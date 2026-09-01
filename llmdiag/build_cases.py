"""
方向7: 诊断案例数据集构建器
================================
从方向1 的扰动 episode (已知注入的 ground truth) 构建
(轨迹摘要, 事件时间线, episode 汇总, ground truth) JSON 案例,
供规则基线与 LLM 诊断层评测。

用法:
  python llmdiag/build_cases.py            # 复用 closeloop/results_pilot
  python llmdiag/build_cases.py --fresh    # 重新生成 (跑受控 episode)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ground truth 标注: 场景 -> (定位类型, 归因类别)
SCENARIO_GT = {
    "S0_none": ("none", "baseline"),
    "S1_machine@150": ("machine:0", "disruption_machine"),
    "S2_machine+agv": ("machine:0+agv:1", "disruption_machine_agv"),
    "S3_mild_stochastic": ("stochastic", "disruption_stochastic"),
    "S4_harsh": ("machine:0", "disruption_machine"),
}


def summarize_record(rec: dict) -> dict:
    """把 closeloop episode 记录打包为紧凑案例。"""
    cfg = rec.get("config", {})
    s = rec.get("summary", {}) or {}
    scen = cfg.get("scenario", "S0_none")
    loc, cause = SCENARIO_GT.get(scen, ("unknown", "unknown"))
    return {
        "case_id": f"{cfg.get('instance','?')}|{scen}|{cfg.get('policy','?')}",
        "query": "为何该 episode 的 makespan/未完工状态如此? 定位关键资源与原因, 并给干预建议。",
        "trajectory_summary": {
            "makespan": rec.get("makespan"),
            "finished": rec.get("finished"),
            "steps": rec.get("steps"),
            "n_events": rec.get("n_events"),
            "events": rec.get("events_sample", [])[:6],
            "agv_busy": s.get("agv_busy_utilization"),
            "agv_loaded": s.get("agv_loaded_utilization"),
            "agv_waiting_total": s.get("agv_waiting_time_total"),
            "tasked_stationary": s.get("tasked_stationary_count"),
            "blocking_delay_mean": s.get("transport_blocking_delay_mean"),
            "queue_wait_mean": s.get("operation_queue_waiting_time_mean"),
            "machine_waiting_inbound": s.get(
                "machine_waiting_for_inbound_transfer_ratio"),
            "machine_down_steps": s.get("machine_down_steps_total"),
            "agv_down_steps": s.get("agv_down_steps_total"),
            "throughput_jobs": s.get("throughput_jobs"),
        },
        "plan_revisions": {
            "n_revisions": rec.get("n_plan_revisions"),
            "revision_count": rec.get("orchestrator_stats", {}).get(
                "revision_count"),
            "revision_fails": rec.get("orchestrator_stats", {}).get(
                "revision_fail_count"),
        },
        "config_context": {
            "instance": cfg.get("instance"), "num_agv": cfg.get("num_agv"),
            "policy": cfg.get("policy"), "scenario": scen,
        },
        "ground_truth": {"localization": loc, "cause": cause},
    }


def build_from_existing(out_path: Path) -> list:
    cases = []
    for f in sorted(glob.glob(str(ROOT / "closeloop/results_pilot/closeloop_pilot_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "n_runs" in d or d.get("error"):
                continue
            cases.append(summarize_record(d))
    supp = ROOT / "closeloop/results_pilot/closeloop_s4_supplement.json"
    if supp.exists():
        for d in json.loads(supp.read_text()):
            d["config"] = d.get("config", {})
            d["config"]["scenario"] = "S4_harsh"
            cases.append(summarize_record(d))
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=1, default=str))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "llmdiag/cases_v1.json"))
    args = ap.parse_args()
    cases = build_from_existing(Path(args.out))
    print(f"built {len(cases)} cases -> {args.out}")
    from collections import Counter

    print(Counter(c["ground_truth"]["cause"] for c in cases))


if __name__ == "__main__":
    main()
