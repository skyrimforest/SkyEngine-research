"""方向7 v2: 规则基线诊断器 (v2案例格式)
==========================================
与 v1 相同的"事件优先+签名"逻辑, 适配双字段定位与 v2 档案。
hard 变体无事件流 -> 只能靠统计签名, 定位目标将不可知 (预期塌陷点)。
用法: python3 llmdiag/v2/diagnoser_rule.py [--cases cases_v2.json] [--out results_rule.json]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def diagnose(case: dict) -> dict:
    kpi = case.get("kpi", {})
    evs = case.get("events", [])
    mach = sorted({e.get("payload", {}).get("machine_id") for e in evs
                   if e.get("type") == "machine_breakdown"})
    agvs = sorted({e.get("payload", {}).get("agv_id") for e in evs
                   if e.get("type") == "agv_breakdown"})
    unfinished = kpi.get("finished") is False
    wait = kpi.get("agv_waiting_time_total") or 0
    stat = kpi.get("tasked_stationary_count") or 0

    if mach and agvs:
        pred = {"loc_type": "machine_agv",
                "loc_target": f"machine:{mach[0]}+agv:{agvs[0]}",
                "cause": "disruption_machine_agv"}
    elif mach:
        pred = {"loc_type": "machine", "loc_target": f"machine:{mach[0]}",
                "cause": "disruption_machine"}
    elif evs and (agvs or any(e.get("type") in ("machine_breakdown", "agv_breakdown") for e in evs)):
        pred = {"loc_type": "stochastic", "loc_target": "stochastic",
                "cause": "disruption_stochastic"}
    elif evs:
        pred = {"loc_type": "stochastic", "loc_target": "stochastic",
                "cause": "disruption_stochastic"}
    elif unfinished and wait > 2000 and not stat:
        pred = {"loc_type": "task_pool", "loc_target": "task_pool",
                "cause": "starvation_livelock"}
    elif unfinished and stat > 0:
        pred = {"loc_type": "corridor", "loc_target": "corridor",
                "cause": "blocking_livelock"}
    elif (kpi.get("machine_down_steps_total") or 0) > 0:
        # 无事件但有停机统计: 类型可判, 目标不可知 (hard档签名推断的极限)
        pred = {"loc_type": "machine", "loc_target": "machine:unknown",
                "cause": "disruption_machine"}
    elif (kpi.get("agv_down_steps_total") or 0) > 0:
        pred = {"loc_type": "stochastic", "loc_target": "stochastic",
                "cause": "disruption_stochastic"}
    else:
        pred = {"loc_type": "none", "loc_target": "none", "cause": "baseline"}

    # 干预: v1 查表 (药房=引擎可执行4旋钮)
    table = {
        "disruption_machine": "assigner(least_congestion)",
        "disruption_machine_agv": "fleet(K+1)",
        "disruption_stochastic": "padding(alpha=0.2)",
        "starvation_livelock": "fleet(K+1)",
        "blocking_livelock": "assigner(random)",
        "machine_bottleneck": "assigner(least_congestion)",
        "baseline": "none",
    }
    pred.update({"evidence": [], "intervention": table.get(pred["cause"], "none"),
                 "narrative": "rule baseline"})
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "cases_v2.json"))
    ap.add_argument("--out", default=str(HERE / "results_rule.json"))
    args = ap.parse_args()
    from score import score_one, aggregate
    cases = json.loads(Path(args.cases).read_text())
    rows = []
    for c in cases:
        rows.append(dict(score_one(diagnose(c), c),
                         case_id=c["case_id"], gt=c["ground_truth"]))
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for variant in ("easy", "hard"):
        sub = [r for r in rows if r["case_id"].endswith(variant)]
        print(variant, aggregate(sub))
    print("->", args.out)


if __name__ == "__main__":
    main()
