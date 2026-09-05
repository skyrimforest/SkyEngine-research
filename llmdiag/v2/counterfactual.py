"""方向7 v2: 反事实执行器 (方法论 §4 判分第三层; HANDOFF §5-3)
================================================================
解析诊断答案的干预建议 → 改引擎配置 → 同 seed 重仿真 → ΔKPI 判分。

  ΔKPI = base_makespan − rerun_makespan   (正值 = 干预使完工更早)
  成功: ΔKPI ≥ δ·base (δ=5%)，或 基线未完工→干预后完工
  对齐 CRN: fault_delta_makespan = twin − injected (负值=故障伤害量)。
  恢复率 recovery_ratio = ΔKPI / (−fault_delta_makespan)，即"干预找回了几成应恢复量"。

干预词表 (方法论 §3 药房, 引擎可执行四旋钮):
  padding(α)  → processing_time_config 确定性放大 (multiplier_uniform low=high=1+α)
  fleet(K±n)  → num_agv 增减
  periodic_revision(T) → trigger_override="periodic-T"
  assigner(name) → assigner
  none → 不重仿真, 记 no_intervention

用法 (容器内):
  python llmdiag/v2/counterfactual.py --results llmdiag/v2/results_rule.json \
      --policy greedy-reactive --per-class 1 --out results_cf_rule.json
cpsat 系案例当前不可重仿真 (FJSP /solve 微服务缺失, HANDOFF §3 备注), 自动跳过。
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
ROOT = HERE.parents[1]
TIMEOUT_S = 420.0
DELTA_SUCCESS = 0.05


def parse_intervention(s: str):
    """药房词表 -> (family, arg); 词表外返回 (None, raw)。"""
    s = (s or "").strip()
    if s == "none":
        return "none", ""
    for pat, fam in (
        (r"^padding\(alpha\s*=\s*([\d.]+)\)$", "padding"),
        (r"^fleet\(K\s*=?\s*([+-]?\d+)\)$", "fleet"),
        (r"^periodic_revision\((\d+)\)$", "periodic_revision"),
        (r"^assigner\(([a-z_]+)\)$", "assigner"),
    ):
        m = re.match(pat, s)
        if m:
            return fam, m.group(1)
    return None, s


def apply_to_scn(scn: dict, fam: str, arg: str) -> dict:
    """干预 -> 场景配置增量 (返回新 scn; 与 run_scenarios.run_one 同构)。"""
    s = dict(scn)
    if fam == "fleet":
        s["num_agv"] = max(1, int(scn["num_agv"]) + int(arg)) if arg.startswith("+") \
            else max(1, int(arg))
    elif fam == "periodic_revision":
        s["trigger_override"] = f"periodic-{arg}"
    elif fam == "assigner":
        s["assigner"] = arg
    elif fam == "padding":
        a = float(arg)
        s["processing_time_config"] = {
            "enabled": True, "preset": "none",
            "random_seed": int(scn.get("seed", 42)),
            "default_distribution": {"dist": "multiplier_uniform",
                                     "low": 1.0 + a, "high": 1.0 + a},
        }
    return s


def rerun(scn: dict) -> dict:
    from closeloop.orchestrator import run_closed_episode
    route = "astar" if scn["policy"] == "greedy-reactive" else "rolling_mapf_http"
    return run_closed_episode(
        fjsp_path=ROOT / "data/fjsp_official/brandimarte" / f"{scn['instance']}.json",
        map_file=ROOT / scn["map_file"], map_name=scn["map_name"],
        policy=scn["policy"], exception_config=scn["exception_config"],
        num_agv=scn["num_agv"], seed=scn["seed"], max_steps=4096,
        route_solver_name=route,
        assigner=scn.get("assigner", "nearest"),
        trigger_override=scn.get("trigger_override"),
        processing_time_config=scn.get("processing_time_config"),
    )


def grade(case: dict, scn: dict, intervention: str) -> dict:
    fam, arg = parse_intervention(intervention)
    out = {"case_id": case["case_id"], "intervention": intervention,
           "family": fam or "invalid", "arg": arg}
    if fam is None:
        return {**out, "executable": False, "delta_kpi": None,
                "success": False, "note": f"不在药房词表: {arg}"}
    if fam == "none":
        return {**out, "executable": True, "delta_kpi": None,
                "success": None, "note": "不建议干预"}
    fd = (case.get("meta") or {}).get("fault_delta_makespan")
    base_ms = (case.get("meta") or {}).get("makespan")
    if base_ms is None and case.get("episode"):
        base_ms = case["episode"].get("makespan")
    t0 = time.time()
    try:
        rec = rerun(apply_to_scn(scn, fam, arg))
    except Exception as e:  # noqa: BLE001
        return {**out, "executable": True, "delta_kpi": None,
                "success": False, "note": f"重仿真异常: {type(e).__name__}: {e}"}
    new_ms, new_fin = rec.get("makespan"), rec.get("finished")
    delta = round(float(base_ms) - float(new_ms), 1) if base_ms and new_ms else None
    success = None
    if delta is not None:
        if not case.get("meta", {}).get("finished", True) and new_fin:
            success = True
        else:
            success = delta >= DELTA_SUCCESS * float(base_ms)
    ratio = None
    if delta is not None and isinstance(fd, (int, float)) and fd < 0:
        ratio = round(delta / (-fd), 3)  # 找回的应恢复量占比
    return {**out, "executable": True, "delta_kpi": delta,
            "rerun_makespan": new_ms, "rerun_finished": new_fin,
            "fault_delta_makespan": fd, "recovery_ratio": ratio,
            "success": bool(success), "note": f"同 seed={scn['seed']} 重仿真",
            "wall_time_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="诊断结果文件 (results_rule.json / results_qwen3_4b.json)")
    ap.add_argument("--out", default="", help="默认 results_cf_<同名>.json")
    ap.add_argument("--policy", default="greedy-reactive",
                    help="只重仿真该策略案例 (cpsat 需微服务, 现跳过)")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args()

    cases = {c["case_id"]: c
             for c in json.loads((HERE / "cases_v2.json").read_text())}
    scns = {}
    for l in (HERE / "scenarios_v2.jsonl").read_text().splitlines():
        if l.strip():
            s = json.loads(l)
            scns[s["scen_id"]] = s

    rp = Path(args.results)
    if not rp.is_absolute():
        rp = next((c for c in (ROOT / rp, HERE.parent / rp, rp) if c.exists()),
                  ROOT / rp)
    results = json.loads(rp.read_text())
    answers = results.get("results", results) if isinstance(results, dict) else results
    if isinstance(answers, dict):
        answers = answers.get("cases", [{"case_id": k, **(
            v if isinstance(v, dict) else {"answer": v})}
            for k, v in answers.items()])

    # 按类各取 N 个可重仿真案例 (策略过滤 + 有干预建议)
    from collections import Counter
    picked, cnt = [], Counter()
    for a in answers:
        cid = a.get("case_id", "")
        case = cases.get(cid)
        if not case:
            continue
        cls = case["scen_id"].rsplit("_", 1)[0]
        ivt = a.get("intervention", a.get("answer", {}).get("intervention")
                     if isinstance(a.get("answer"), dict) else None)
        if not ivt or parse_intervention(str(ivt))[0] in (None, "none"):
            continue
        if case["config"].get("policy") != args.policy:
            continue
        if cnt[cls] >= args.per_class:
            continue
        cnt[cls] += 1
        picked.append((case, str(ivt)))

    out_path = Path(args.out) if args.out else \
        HERE / f"results_cf_{rp.stem.replace('results_', '')}.json"
    done: dict = {}
    if out_path.exists():  # 断点续跑
        for r in json.loads(out_path.read_text()):
            if r.get("delta_kpi") is not None or r.get("family") == "none":
                done[r["case_id"]] = r
    out_lines = list(done.values())
    print(f"[cf] 待重仿真 {len(picked)} 案例 (policy={args.policy}, "
          f"已完成 {len(done)}) -> {out_path.name}", flush=True)

    ctx = mp.get_context("fork")
    for case, ivt in picked:
        if case["case_id"] in done:
            continue
        scn = dict(scns[case["scen_id"]])
        tmp = HERE / f".tmp_cf_{case['case_id'].replace('|', '_')}.json"

        def _target():
            try:
                tmp.write_text(json.dumps(grade(case, scn, ivt),
                                          ensure_ascii=False))
            except Exception as e:  # noqa: BLE001
                tmp.write_text(json.dumps(
                    {"case_id": case["case_id"], "error": str(e)[:200]}))

        proc = ctx.Process(target=_target)
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            proc.terminate(); proc.join(5)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            g = {"case_id": case["case_id"], "intervention": ivt,
                 "error": f"hard timeout {args.timeout}s"}
        else:
            g = json.loads(tmp.read_text())
        tmp.unlink(missing_ok=True)
        if "error" in g:
            g.update({"executable": None, "delta_kpi": None, "success": None})
        out_lines = [x for x in out_lines if x["case_id"] != g["case_id"]] + [g]
        out_path.write_text(json.dumps(out_lines, ensure_ascii=False, indent=1))
        print(f"  [cf] {g['case_id']:<28} {str(g.get('intervention')):<22} "
              f"ΔKPI={g.get('delta_kpi')} success={g.get('success')} "
              f"ratio={g.get('recovery_ratio')}", flush=True)

    ok = [g for g in out_lines if g.get("success") is not None]
    summary = {"n": len(out_lines), "graded": len(ok),
               "success_rate": round(sum(1 for g in ok if g["success"]) / len(ok), 3)
               if ok else None,
               "recovery_ratio_mean": round(sum(g["recovery_ratio"] for g in ok
                                                if g.get("recovery_ratio") is not None)
                                            / max(1, sum(1 for g in ok if g.get(
                                                "recovery_ratio") is not None)), 3)
               if any(g.get("recovery_ratio") is not None for g in ok) else None}
    out = {"summary": summary, "results": out_lines}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[cf] {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
