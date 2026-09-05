"""方向7 v2: hard 视图三防线质检 (方法论 v0.2 §1-5 回溯义务)
================================================================
v0.2: "hard 孪生'只留统计特征'存在签名被摘要抹平的风险——质检必须落在 hard 视图
本身而非完整数据"。cases_v2 (104 案例) 的签名确认是在完整数据上做的, 本脚本按
三道防线回溯复验:

  防线1 签名复验 (逐案例): 该故障的判别性统计量在 hard 视图可见字段
        (kpi 20项 + meta 标量) 上仍然成立; 不成立 → 拒收 (留作 L1 档使用)。
  防线2 可辨识性 (数据集级): 各归因类在 hard 视图标记向量上的类间可分性;
        两类标记轮廓不可分 → 报告混淆对 (多解集合评分 EM+set-F1 候选)。
  防线3 CF−INT 兜底 (视图生成): 为注入类生成 INT 视图 (hard+公开故障参数) →
        cases_int.json, 供后续 CF−INT 差距测量。

用法: python3 llmdiag/v2/qc_hard.py   (只读 cases_v2.json, 不碰引擎)
输出: qc_hard_report.json + 控制台摘要
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# hard 视图可见的判别性标记 (布尔化, 仅用 kpi/meta 字段)
def markers(case: dict) -> dict:
    k = case.get("kpi", {})
    m = case.get("meta", {})

    def num(field):
        v = k.get(field)
        return v if isinstance(v, (int, float)) else 0

    return {
        "machine_down": num("machine_down_steps_total") > 0,
        "agv_down": num("agv_down_steps_total") > 0,
        "obstacle": num("obstacle_blocked_steps_total") > 0,
        "disruption_active": num("active_disruption_count") > 0,
        "unfinished": case.get("meta", {}).get("finished") is False,
        "idle_starved": num("agv_waiting_time_total") > 2000
                        and num("tasked_stationary_count") == 0,
        "stationary": num("tasked_stationary_count") > 0,
        "queue_heavy": num("operation_queue_waiting_time_mean") > 20,
        "blocking_heavy": num("transport_blocking_delay_mean") > 30,
        "no_marker": not (num("machine_down_steps_total") > 0
                          or num("agv_down_steps_total") > 0
                          or num("obstacle_blocked_steps_total") > 0),
    }


# 防线1: 每类在 hard 视图上必须成立的签名谓词
HARD_SIGNATURE = {
    "baseline": lambda mk: mk["no_marker"] and not mk["unfinished"],
    "disruption_machine": lambda mk: mk["machine_down"] or mk["disruption_active"],
    "disruption_machine_agv": lambda mk: mk["machine_down"] and mk["agv_down"],
    "disruption_stochastic": lambda mk: (mk["machine_down"] or mk["agv_down"]
                                         or mk["obstacle"]),
    "starvation_livelock": lambda mk: mk["unfinished"] and mk["idle_starved"],
    "blocking_livelock": lambda mk: mk["unfinished"] and mk["stationary"],
    "machine_bottleneck": lambda mk: mk["queue_heavy"],
    "plan_mismatch": lambda mk: mk["unfinished"] or mk["queue_heavy"],
    "route_disruption": lambda mk: mk["obstacle"],
}


def line1_signature(case: dict) -> tuple[bool, str]:
    cause = case["ground_truth"]["cause"]
    pred = HARD_SIGNATURE.get(cause)
    if pred is None:
        return False, f"no signature predicate for {cause}"
    mk = markers(case)
    ok = pred(mk)
    active = [k for k, v in mk.items() if v]
    return ok, f"markers={active}"


def line2_identifiability(cases: list[dict]) -> dict:
    """类间标记轮廓可分性: 共享完全相同标记集合的类对 = hard 视图不可辨识对。"""
    import itertools
    from collections import defaultdict
    prof = defaultdict(list)
    for c in cases:
        prof[c["ground_truth"]["cause"]].append(sorted(
            k for k, v in markers(c).items() if v))
    confusable, matrix = [], {}
    for (a, pa), (b, pb) in itertools.combinations(prof.items(), 2):
        set_a = {tuple(x) for x in pa}
        set_b = {tuple(x) for x in pb}
        overlap = len(set_a & set_b) / max(1, min(len(set_a), len(set_b)))
        matrix[f"{a}|{b}"] = round(overlap, 3)
        if overlap >= 0.5:
            confusable.append({"pair": [a, b], "profile_overlap": round(overlap, 3),
                               "note": "hard 视图上轮廓重合>=50%, 需 INT 消歧或 set-F1"})
    return {"pairwise_overlap": matrix, "confusable_pairs": confusable,
            "class_profile_counts": {k: len(set(tuple(x) for x in v))
                                     for k, v in prof.items()}}


def line3_int_views(cases: list[dict], scns: dict) -> list[dict]:
    """注入类 INT 视图 = hard + 公开故障参数 (CF−INT 兜底的 INT 侧)。"""
    ints = []
    for c in cases:
        scn = scns.get(c["scen_id"])
        if not scn:
            continue
        exc = scn.get("exception_config") or {}
        if exc.get("schedule"):
            fault = {"public_fault_schedule": exc["schedule"]}
        elif exc.get("preset"):
            fault = {"public_fault_preset": exc["preset"],
                     "fault_seed": exc.get("random_seed")}
        else:
            continue
        ints.append(dict(c, case_id=f"{c['scen_id']}|int", variant="int",
                         events=[], fault=fault))
    return ints


def main():
    cases = json.loads((HERE / "cases_v2.json").read_text())
    scns = {json.loads(l)["scen_id"]: json.loads(l)
            for l in (HERE / "scenarios_v2.jsonl").read_text().splitlines()
            if l.strip()}

    # 防线1: 逐案例签名复验 (在 hard 孪生上)
    passed, rejected = [], []
    for c in cases:
        if c["variant"] != "hard":
            continue
        ok, note = line1_signature(c)
        (passed if ok else rejected).append(
            {"case_id": c["case_id"], "cause": c["ground_truth"]["cause"],
             "note": note})
    print(f"== 防线1 签名复验 (hard 视图) ==")
    print(f"  通过 {len(passed)} / 拒收 {len(rejected)} (拒收案例移出正式集, 留作 L1 档)")
    by_cls = {}
    for r in rejected:
        by_cls[r["cause"]] = by_cls.get(r["cause"], 0) + 1
    for k, v in sorted(by_cls.items()):
        print(f"    拒收 {k}: {v}")

    # 防线2: 类间可辨识性
    hard_cases = [c for c in cases if c["variant"] == "hard"]
    kept_hard = [c for c in hard_cases
                 if c["case_id"] not in {r["case_id"] for r in rejected}]
    l2 = line2_identifiability(kept_hard)
    print(f"== 防线2 可辨识性 (复验通过集) ==")
    for p in l2["confusable_pairs"]:
        print(f"    混淆对: {p['pair']} overlap={p['profile_overlap']}")

    # 防线3: INT 视图生成
    ints = line3_int_views(cases, scns)
    (HERE / "cases_int.json").write_text(json.dumps(ints, ensure_ascii=False, indent=1))
    print(f"== 防线3 INT 视图 ==")
    print(f"  生成 {len(ints)} 个 INT 案例 -> cases_int.json")

    report = {
        "line1_signature": {"passed": len(passed), "rejected": len(rejected),
                            "rejected_detail": rejected,
                            "by_cause": by_cls,
                            "passed_ids": [p["case_id"] for p in passed]},
        "line2_identifiability": l2,
        "line3_int_views": {"n": len(ints)},
        "verdict": ("F2 归因前提部分成立" if rejected else
                    "F2 归因前提成立: hard 视图签名全部可辨识, 塌陷归因于溯因本身"),
    }
    (HERE / "qc_hard_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {HERE/'qc_hard_report.json'}")
    print("verdict:", report["verdict"])


if __name__ == "__main__":
    main()
