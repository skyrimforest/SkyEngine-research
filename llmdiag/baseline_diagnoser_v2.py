"""方向7 v2: 规则基线诊断器 (对照臂, 词表化输出)
================================================
与 v1 的差异:
  1. 干预输出改为引擎可执行词表 (padding/fleet/periodic_revision/assigner/none),
     可直接进反事实执行器 (grader_v2.grade_intervention);
  2. 支持 easy/hard 双档: easy 先看事件 (v1 逻辑), hard 只看统计特征
     —— 按方法论 §6 可检验主张 ②, 规则基线在 hard 档应显著塌陷;
  3. 归因词表含 plan_mismatch / unseen 判据。

diagnose(case, variant) -> dict 接口与 LLM 诊断器保持一致。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _s(case: dict, *keys, default=None):
    v = case["episode"].get("summary", {}) or {}
    for k in keys:
        if isinstance(v.get(k), (int, float)):
            return float(v[k])
    return default


def _events(case: dict) -> list:
    return case["episode"].get("events_all") or []


def _evt_id(case: dict, typ: str, nth: int = 0) -> str:
    """事件在档案中的证据 ID (与 archive_builder 的 EVT 编号一致)。"""
    i = 0
    for j, e in enumerate(_events(case)):
        if isinstance(e, dict) and e.get("type") == typ:
            if i == nth:
                return f"EVT{j:03d}"
            i += 1
    return f"EVT000"


def _kpi_line(case: dict, key: str, val) -> str:
    """与 archive_builder STAT 段完全一致的 KPI 行 (可回查)。"""
    return f"KPI:{key}={val}"


def _diagnose_easy(case: dict) -> tuple[str, str, list]:
    """事件优先逻辑 (v1 规则的直迁)。证据只引用档案真实 ID。"""
    ev = _events(case)
    types = {e.get("type") for e in ev if isinstance(e, dict)}
    m_ids = sorted({e.get("payload", {}).get("machine_id") for e in ev
                    if isinstance(e, dict) and e.get("type") == "machine_breakdown"})
    a_ids = sorted({e.get("payload", {}).get("agv_id") for e in ev
                    if isinstance(e, dict) and e.get("type") == "agv_breakdown"})
    if "machine_breakdown" in types and "agv_breakdown" in types:
        return (f"machine:{m_ids[0]}+agv:{a_ids[0]}", "disruption_machine_agv",
                [_evt_id(case, "machine_breakdown"), _evt_id(case, "agv_breakdown")])
    if "machine_breakdown" in types:
        return f"machine:{m_ids[0]}", "disruption_machine", [_evt_id(case, "machine_breakdown")]
    if "temporary_obstacle" in types:
        return "stochastic", "unseen", [_evt_id(case, "temporary_obstacle")]  # 留出类
    if ev:
        return "stochastic", "disruption_stochastic", [_kpi_line(case, "n_events_total", len(ev))]

    # 无事件: 未完工 => 活锁类; 完工 => 瓶颈/基线
    rec = case["episode"]
    if not rec.get("finished"):
        idle = _s(case, "agv_busy_utilization", default=1.0) < 0.05
        stationary = _s(case, "tasked_stationary_count", default=0)
        if idle and not stationary:
            return "task_pool", "starvation_livelock", [
                _kpi_line(case, "n_events_total", 0),
                f"KPI:agv_busy_utilization(episode均值)="
                f"{_s(case, 'agv_busy_utilization', default=0):.3f}"]
        return "corridor", "blocking_livelock", [
            _kpi_line(case, "n_events_total", 0),
            f"KPI:tasked_stationary_count(episode均值)={stationary:.3f}"]
    q = _s(case, "operation_queue_waiting_time_mean", "queue_wait_mean", default=0.0)
    if q > 20:
        return "machines", "machine_bottleneck", [
            f"KPI:operation_queue_waiting_time_mean(episode均值)={q:.3f}"]
    return "none", "baseline", [_kpi_line(case, "n_events_total", 0)]


def _diagnose_hard(case: dict) -> tuple[str, str, list]:
    """hard 档: 无事件明细, 仅统计特征 + 修订账本签名。"""
    rec = case["episode"]
    n_ev = rec.get("n_events", 0)
    n_rev = rec.get("n_plan_revisions", 0)
    finished = rec.get("finished")
    busy = _s(case, "agv_busy_utilization", default=1.0)
    stationary = _s(case, "tasked_stationary_count", default=0)
    q = _s(case, "operation_queue_waiting_time_mean", "queue_wait_mean", default=0.0)
    blocking = _s(case, "transport_blocking_delay_mean", default=0.0)
    ev = [_kpi_line(case, "n_events_total", n_ev),
          _kpi_line(case, "n_plan_revisions", n_rev)]

    if n_ev and n_rev == 0 and str(rec["config"]["policy"]).startswith("cpsat"):
        return "machines", "plan_mismatch", ev + ["扰动后零修订"]
    if n_ev and busy > 0.3:
        return "stochastic", "disruption_stochastic", ev
    if n_ev:
        return "stochastic", "disruption_stochastic", ev
    if not finished and busy < 0.05:
        return "task_pool", "starvation_livelock", ev + [f"busy={busy:.2f}"]
    if not finished and (stationary or blocking > 50):
        return "corridor", "blocking_livelock", ev + [
            f"stationary={stationary} blocking={blocking:.1f}"]
    if finished and q > 20:
        return "machines", "machine_bottleneck", ev + [f"queue_wait={q:.1f}"]
    if not n_ev:
        return "none", "baseline", ev
    return "machines", "unseen", ev


INTERVENTIONS = {
    "disruption_machine": "padding(alpha=0.2)",
    "disruption_machine_agv": "fleet(K=+1)",
    "disruption_stochastic": "padding(alpha=0.1)",
    "starvation_livelock": "assigner(least_congestion)",
    "blocking_livelock": "assigner(random)",
    "machine_bottleneck": "fleet(K=+1)",
    "plan_mismatch": "periodic_revision(100)",
    "baseline": "none",
    "unseen": "padding(alpha=0.1)",
}


def _norm_fleet(s: str) -> str:
    return s.replace("K=+1", "K=5").replace("K=+2", "K=6")


def diagnose(case: dict, variant: str = "easy") -> dict:
    loc, cause, ev = _diagnose_easy(case) if variant == "easy" \
        else _diagnose_hard(case)
    ivt = _norm_fleet(INTERVENTIONS.get(cause, "none"))
    return {
        "case_id": case["case_id"],
        "localization": loc,
        "cause": cause if cause in ("unseen",) or cause in INTERVENTIONS else "unseen",
        "evidence": ev,
        "intervention": ivt,
        "narrative": "规则基线无叙述。",
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "llmdiag/results_v2/cases.json"))
    ap.add_argument("--variant", default="easy", choices=["easy", "hard"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    from collections import Counter
    import sys
    sys.path.insert(0, str(ROOT))
    from llmdiag.grader_v2 import grade_loc_cause, summarize
    grades = []
    for c in cases:
        ans = diagnose(c, args.variant)
        grades.append({"case_id": c["case_id"],
                       "target_class": c["spec"]["target_class"],
                       **grade_loc_cause(ans, c["ground_truth"])})
    out = args.out or str(Path(args.cases).parent / f"baseline_grades_{args.variant}.json")
    Path(out).write_text(json.dumps(grades, ensure_ascii=False, indent=1))
    by_cls: dict = {}
    for g in grades:
        by_cls.setdefault(g["target_class"], []).append(g)
    print(f"== 规则基线 @{args.variant} ==")
    print(json.dumps(summarize(grades), ensure_ascii=False))
    for cls, gs in sorted(by_cls.items()):
        acc = sum(x["cause_correct"] for x in gs) / len(gs)
        print(f"  {cls:<24} cause_acc={acc:.2f} (n={len(gs)})")
    print("->", out)


if __name__ == "__main__":
    main()
