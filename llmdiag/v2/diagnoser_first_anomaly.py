"""方向7 v2: G1.2 首异常启发式基线 (RCA前沿借鉴 §二-G1.2; H5 预注册)
====================================================================
strawman 策略: 取时间上最早的事件作答 (first anomaly = root cause 的朴素反向命题)。
- easy 档: 有事件流可跑; hard 档无事件流 → 只能答 baseline (按设计的盲人),
  其塌陷即 H5 "时间陷阱"对照组的一半。
- 时间陷阱率 (注入类): 最早事件 ≠ 计划注入的主事件 (如 stochastic 噪声先于计划注入
  触发) 的比例 —— 首异常启发式被"诱饵事件"带偏的频率, 与 LLM 的 anchoring 呼应
  (LLM_RCA_Failures: anchoring RR<0.55)。

与规则基线同接口: diagnose(case) -> dict (score.py score_one 兼容)。
用法: python3 llmdiag/v2/diagnoser_first_anomaly.py [--variant easy]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from score import score_one, aggregate  # noqa: E402

INTERVENTIONS = {
    "disruption_machine": "padding(alpha=0.2)",
    "disruption_machine_agv": "fleet(K+1)",
    "disruption_stochastic": "padding(alpha=0.1)",
    "unseen": "padding(alpha=0.1)",
}


def first_event(case: dict) -> dict | None:
    evs = [e for e in (case.get("events") or []) if e.get("step") is not None]
    return min(evs, key=lambda e: e["step"]) if evs else None


def diagnose(case: dict) -> dict:
    ev = first_event(case)
    if ev is None:
        # hard 档或无事件: 首异常启发式对统计特征全盲, 只能答基线
        return {"loc_type": "none", "loc_target": "none", "cause": "baseline",
                "evidence": [], "intervention": "none", "narrative": "无事件可考。"}
    typ = ev.get("type")
    payload = ev.get("payload") or {}
    if typ == "machine_breakdown":
        loc_type, loc_target, cause = "machine", f"machine:{payload.get('machine_id')}", "disruption_machine"
    elif typ == "agv_breakdown":
        loc_type, loc_target, cause = "agv", f"agv:{payload.get('agv_id')}", "disruption_machine_agv"
    elif typ == "temporary_obstacle":
        loc_type, loc_target, cause = "stochastic", "stochastic", "unseen"
    else:
        loc_type, loc_target, cause = "stochastic", "stochastic", "disruption_stochastic"
    return {"loc_type": loc_type, "loc_target": loc_target, "cause": cause,
            "evidence": [ev.get("id", "EVT001")],
            "intervention": INTERVENTIONS.get(cause, "none"),
            "narrative": f"首个异常为 {typ}@step{ev.get('step')}。"}


def time_trap(case: dict, pred_cause: str) -> bool | None:
    """时间陷阱: 注入类的最早事件类型 != 计划注入的主事件类型。
    数据源: scenarios_v2.jsonl 的 exception_config.schedule (注入真值)。"""
    gt = case["ground_truth"]["cause"]
    if gt not in ("disruption_machine", "disruption_machine_agv"):
        return None
    scen = SCNS.get(case["scen_id"])
    sched = (scen.get("exception_config") or {}).get("schedule") if scen else None
    if not sched:
        return None
    primary = min(sched, key=lambda s: s["step"])  # 计划注入的最早项
    ev = first_event(case)
    if ev is None:
        return None
    return ev.get("type") != primary.get("type")


SCNS = {}


def main():
    global SCNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="easy", choices=["easy", "hard"])
    args = ap.parse_args()

    SCNS = {json.loads(l)["scen_id"]: json.loads(l)
            for l in (HERE / "scenarios_v2.jsonl").read_text().splitlines()
            if l.strip()}
    cases = [c for c in json.loads((HERE / "cases_v2.json").read_text())
             if c["variant"] == args.variant]

    rows, traps = [], []
    for c in cases:
        pred = diagnose(c)
        row = score_one(pred, c)
        row["case_id"] = c["case_id"]
        row["target_class"] = c["ground_truth"]["cause"]
        row["pred_cause"] = pred["cause"]
        rows.append(row)
        t = time_trap(c, pred["cause"])
        if t is not None:
            traps.append(t)

    out = {"summary": aggregate(rows), "time_trap_rate": round(
        sum(1 for t in traps if t) / len(traps), 3) if traps else None,
        "n_trap_eligible": len(traps), "results": rows}
    p = HERE / f"results_first_anomaly_{args.variant}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[first-anomaly @{args.variant}]",
          json.dumps(out["summary"], ensure_ascii=False))
    if out["time_trap_rate"] is not None:
        print(f"  时间陷阱率 (注入类首事件≠计划注入): {out['time_trap_rate']} "
              f"(n={out['n_trap_eligible']}) — H5 数据点")
    print("->", p)


if __name__ == "__main__":
    main()
