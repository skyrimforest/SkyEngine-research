"""
方向7: 规则基线诊断器 (LLM 诊断层的对照组)
================================================
阈值+峰值定位的规则诊断: 输入案例 JSON, 输出 (定位, 归因, 证据, 干预),
并与 ground truth 对比给出定位/归因准确率。LLM 版只需实现同样的
diagnose(case) -> dict 接口 (见 prompts.py 的提示模板)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def diagnose(case: dict) -> dict:
    t = case["trajectory_summary"]
    ev = t.get("events") or []
    evidence: list = []

    # 1) 事件优先: 有真实扰动事件则直接定位
    down_machines = sorted({
        e["payload"].get("machine_id")
        for e in ev if e.get("type") == "machine_breakdown"
        and isinstance(e.get("payload"), dict)
    })
    down_agvs = sorted({
        e["payload"].get("agv_id")
        for e in ev if e.get("type") == "agv_breakdown"
        and isinstance(e.get("payload"), dict)
    })

    localization, cause = "none", "baseline"
    if down_machines and down_agvs:
        localization = f"machine:{down_machines[0]}+agv:{down_agvs[0]}"
        cause = "disruption_machine_agv"
        evidence.append(f"events: m_down={down_machines} a_down={down_agvs}")
    elif down_machines:
        localization = f"machine:{down_machines[0]}"
        cause = "disruption_machine"
        evidence.append(f"events: m_down={down_machines}")
    elif ev:
        localization = "stochastic"
        cause = "disruption_stochastic"
        evidence.append(f"events: {len(ev)} stochastic events")
    elif not t.get("finished", True):
        # 无事件但未完工 => 活锁/饥饿类
        if (t.get("agv_waiting_total") or 0) > 2000 and not (t.get("tasked_stationary") or 0):
            localization = "task_pool"
            cause = "starvation_livelock"
            evidence.append(
                f"agv_waiting_total={t.get('agv_waiting_total')} "
                f"tasked_stationary={t.get('tasked_stationary')}")
        else:
            localization = "corridor"
            cause = "blocking_livelock"
            evidence.append(f"tasked_stationary={t.get('tasked_stationary')}")
    elif (t.get("queue_wait_mean") or 0) > 20:
        localization = "machines"
        cause = "machine_bottleneck"
        evidence.append(f"queue_wait_mean={t.get('queue_wait_mean')}")

    # 2) 干预建议 (可被反事实执行器验证)
    interventions = {
        "disruption_machine": "为关键机器增加并行等效机器或预留 slack (padding alpha=0.2)",
        "disruption_machine_agv": "增大 AGV 冗余 (K+1) 并启用周期重规划 periodic-100",
        "disruption_stochastic": "启用 moderate padding 并将分配器换为 least_congestion",
        "starvation_livelock": "启用饥饿看门狗: 强制重指派滞留运输任务",
        "blocking_livelock": "分配器换 random/least_congestion 打破聚簇",
        "machine_bottleneck": "对峰值队列机器启用工艺分流 (机器再指派)",
        "baseline": "无 (基线工况)",
    }.get(cause, "无")

    return {
        "case_id": case["case_id"],
        "localization": localization,
        "cause": cause,
        "evidence": evidence,
        "intervention": interventions,
        "ground_truth": case["ground_truth"],
    }


def evaluate(diags: list) -> dict:
    n = len(diags)
    loc_ok = sum(d["localization"] == d["ground_truth"]["localization"] for d in diags)
    cause_ok = sum(
        d["cause"] == d["ground_truth"]["cause"]
        or (d["cause"].startswith("disruption") and d["ground_truth"]["cause"].startswith("disruption"))
        for d in diags
    )
    return {
        "n": n,
        "localization_acc": round(loc_ok / n, 3) if n else None,
        "cause_acc(coarse)": round(cause_ok / n, 3) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "llmdiag/cases_v1.json"))
    ap.add_argument("--out", default=str(ROOT / "llmdiag/baseline_results.json"))
    args = ap.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    diags = [diagnose(c) for c in cases]
    Path(args.out).write_text(json.dumps(diags, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(evaluate(diags), ensure_ascii=False))
    for d in diags[:8]:
        print(f"{d['case_id'][:44]:<46} pred={d['cause']:<24} gt={d['ground_truth']['cause']}")
    print("->", args.out)


if __name__ == "__main__":
    main()
