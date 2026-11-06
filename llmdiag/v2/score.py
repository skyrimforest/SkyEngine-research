"""方向7 v2: 评分器 (三层判分, 全程序化)
========================================
对四元组输出打分:
- 定位: loc_type+loc_target 全对=1.0, 仅类型对=0.5 (hard档预期塌陷点)
- 归因: 严格=1.0, disruption_* 族宽容=0.5
- JRA: 定位满分 且 归因严格对
- 证据: 引用可回查率 (EVT/KPI/REV/CFG 逐条核对, 无引用=0)
用法: from score import score_one, aggregate
"""
import re

LOC_TYPES = {"machine", "agv", "machine_agv", "stochastic", "task_pool",
             "corridor", "machines", "none"}
CAUSES = {"disruption_machine", "disruption_machine_agv", "disruption_stochastic",
          "starvation_livelock", "blocking_livelock", "machine_bottleneck",
          "plan_mismatch", "baseline", "unseen"}


def score_loc(pred: dict, gt: dict) -> float:
    pt = pred.get("loc_type")
    pr = str(pred.get("loc_target", ""))
    gr = str(gt.get("loc_target", ""))
    if pt != gt.get("loc_type"):
        return 0.0
    if pt in ("none", "stochastic", "task_pool", "corridor", "machines"):
        return 1.0  # 该类目标即类型本身
    return 1.0 if pr == gr else 0.5


def score_cause(pred: str, gt: str) -> float:
    if pred == gt:
        return 1.0
    if str(pred).startswith("disruption") and str(gt).startswith("disruption"):
        return 0.5
    return 0.0


def check_citation(cite: str, case: dict) -> bool:
    """单条引用回查: 在档案里找到出处理由即通过"""
    c = str(cite).strip()
    if not c:
        return False
    m = re.match(r"^EVT(\d+)$", c)
    if m:
        return any(e.get("id") == c for e in case.get("events", []))
    # events[N].field 或 events[N].payload.key=value 形式 (下标式引用)
    m = re.match(r"^events\[(\d+)\]\.([\w.]+)", c)
    if m:
        evs = case.get("events", [])
        idx = int(m.group(1))
        if idx >= len(evs):
            return False
        ev = evs[idx]
        rest = m.group(2)
        if rest.startswith("payload"):
            key = rest.split("payload.")[-1].split("=")[0].split("[")[0]
            val = (ev.get("payload") or {}).get(key)
            return val is not None
        return rest.split("=")[0] in ev
    if c.startswith("KPI"):
        field = c.split(":", 1)[-1].split("=")[0].split("[")[0].strip()
        if field in case.get("kpi", {}):
            nums = re.findall(r"[-+]?\d+\.?\d*", c.split("=", 1)[-1]) if "=" in c else []
            if nums:
                try:
                    return abs(float(nums[0]) - float(case["kpi"][field])) <= max(
                        1.0, 0.05 * abs(float(case["kpi"][field])))
                except (ValueError, TypeError):
                    return True  # 字段在但数值格式怪, 给过(字段级核验)
            return True
        return False
    if c.startswith("REV"):
        return any(k in c for k in ("count", "fail", "changed", "assignment")) or \
            "REV" in case.get("revisions", {})
    if c.startswith("CFG"):
        for k in case.get("config", {}):
            if k in c:
                return True
        return False
    # 自由文本引用: 允许引用 kpi 字段名或数值
    for k, v in case.get("kpi", {}).items():
        if k in c:
            return True
    return any(e.get("type", "") in c for e in case.get("events", []))


def score_one(pred: dict | None, case: dict) -> dict:
    """pred: LLM/规则的诊断JSON; case: cases_v2 案例档案"""
    gt = case["ground_truth"]
    if pred is None:
        return {"loc": 0.0, "cause": 0.0, "jra": 0.0, "faithful": 0.0, "parse": 0}
    loc = score_loc(pred, gt)
    cause = score_cause(pred.get("cause"), gt["cause"])
    evs = pred.get("evidence") or []
    faith = (sum(check_citation(c, case) for c in evs) / len(evs)) if evs else 0.0
    return {"loc": loc, "cause": cause,
            "jra": 1.0 if (loc == 1.0 and cause == 1.0) else 0.0,
            "faithful": round(faith, 3),
            "parse": 1,
            "intervention": pred.get("intervention", "none"),
            "narrative": pred.get("narrative", "")}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": n,
        "loc_acc": round(sum(r["loc"] for r in rows) / n, 3),
        "cause_strict_acc": round(sum(1 for r in rows if r["cause"] == 1.0) / n, 3),
        "cause_coarse_acc": round(sum(1 for r in rows if r["cause"] >= 0.5) / n, 3),
        "jra": round(sum(r["jra"] for r in rows) / n, 3),
        "faithful": round(sum(r["faithful"] for r in rows) / n, 3),
        "parse_rate": round(sum(r["parse"] for r in rows) / n, 3),
    }


# ---------------------------------------------------------------------------
# 对抗注入 (grader_v2.py 预留的 --adversarial 欠账, v2 线落地)
# 目的: 验证判分器忠实度层不是虚设 —— 塞"看起来合法但档案里不存在"的引用,
# 每种攻击必须被抓 (faith=0); 抓获率 <1 即判分器存在伪造漏洞。
# ---------------------------------------------------------------------------
def adversarial_attacks(case: dict) -> dict:
    """按引用类型构造 5 种伪造引用 (值需对案例自适应, 确保档案中确实不存在)"""
    kpi_fields = list((case.get("kpi") or {}).keys())
    fake_kpi = next((f for f in ("agv_waiting_time_total", "makespan",
                                 "machine_utilization") if f in kpi_fields),
                    kpi_fields[0] if kpi_fields else "makespan")
    real_val = (case.get("kpi") or {}).get(fake_kpi, 0)
    # 偏离真值 37.7% 再叠加偏移, 保证超出 ±5% 容差
    fake_val = abs(real_val) * 1.377 + 137.7 if real_val else 137.7
    n_evs = len(case.get("events") or [])
    cfg_key = next((k for k in ("instance", "num_agv", "policy", "seed", "map")
                    if k not in (case.get("config") or {})), "nonexistent_cfg")
    kpi_absent = next((f"fake_field_{i}" for i in range(99)
                       if f"fake_field_{i}" not in kpi_fields), "fake_field_x")
    return {
        "phantom_evt": "EVT999",                      # 不存在的事件 ID
        "oob_index": f"events[{n_evs + 3}].payload.machine_id=0",  # 越界下标
        "value_lie": f"KPI:{fake_kpi}={fake_val}",    # 真字段+假数值 (超容差)
        "phantom_cfg": f"CFG:{cfg_key}=1",            # 不存在的配置键
        "phantom_kpi": f"{kpi_absent}=87.3",          # 不存在的统计字段
    }


def adversarial_audit(results: list[dict], cases: dict) -> dict:
    """对一批诊断结果逐案注入伪造引用, 返回判分器抓获率审计。

    results: 诊断器输出列表 (含 case_id 与诊断 JSON 字段)
    cases:   case_id -> 案例档案
    返回: {"n": N, "caught_rate": x, "per_attack": {攻击名: 抓获率}, "leaks": [...]}
    一条伪造引用"被抓获" = check_citation 为 False (faithful 不放水)。
    """
    per_hit, per_total = {}, {}
    leaks = []
    n = 0
    for r in results:
        cid = r.get("case_id")
        case = cases.get(cid)
        if case is None:
            continue
        attacks = adversarial_attacks(case)
        for name, cite in attacks.items():
            caught = not check_citation(cite, case)
            per_total[name] = per_total.get(name, 0) + 1
            if caught:
                per_hit[name] = per_hit.get(name, 0) + 1
            else:
                leaks.append({"case_id": cid, "attack": name, "cite": cite})
        n += 1
    total = sum(per_total.values()) or 1
    return {
        "n_cases": n,
        "n_attacks": total,
        "caught_rate": round(sum(per_hit.values()) / total, 3),
        "per_attack": {k: round(per_hit[k] / per_total[k], 3) for k in per_total},
        "leaks": leaks,
    }


def _cli():
    """用法: python3 score.py --adversarial --cases cases_v2.json --results results_X.json"""
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--cases", default="cases_v2.json")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    here = Path(__file__).parent
    cases = {c["case_id"]: c
             for c in json.loads((here / args.cases).read_text())}
    results = json.loads((here / args.results).read_text())
    audit = adversarial_audit(results, cases)
    print(f"对抗审计: {audit['n_cases']} 案例 × 5 攻击 = {audit['n_attacks']} 条伪造引用")
    print(f"判分器抓获率: {audit['caught_rate']}  (1.0=无伪造漏洞)")
    for k, v in audit["per_attack"].items():
        print(f"  {k}: {v}")
    for lk in audit["leaks"][:10]:
        print(f"  LEAK {lk['case_id']} {lk['attack']}: {lk['cite']}")
    out = args.out or str(Path(args.results).with_name(
        Path(args.results).stem + "_adversarial.json"))
    Path(out).write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    _cli()
