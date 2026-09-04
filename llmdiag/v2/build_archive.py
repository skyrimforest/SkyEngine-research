"""方向7 v2: 核验器 + 档案构建器 (P1)
====================================
读取 scenarios_v2.jsonl + episodes/*.json:
1) 核验 (Generator-Executor-Verifier):
   - 注入类: 标签由构造为真; 检查注入事件是否进入事件样本 (meta.event_visible, 尽力核验)
   - 涌现类: 签名谓词判定, 不达标 -> 丢弃 (记录 dropped 原因)
   - 基线: 若呈现饥饿/拥塞签名 -> 改标为涌现 (记录 planned vs verified)
2) 档案构建: 剥热力图, 证据ID化 (KPI:*/EVT001.../REV:*/CFG:*),
   每场景产出 easy(含事件)/hard(删事件) 孪生案例
输出: cases_v2.json
用法: python3 llmdiag/v2/build_archive.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# 进入档案的 KPI 字段 (来自 summary, 证据ID: KPI:<field>)
KPI_FIELDS = [
    "makespan", "finished", "steps", "machine_utilization",
    "operation_queue_waiting_time_mean", "tasked_stationary_count",
    "agv_busy_utilization", "agv_waiting_time_total",
    "transport_blocking_delay_mean", "throughput_jobs",
    "tardy_job_count", "total_tardiness", "machine_down_steps_total",
    "agv_down_steps_total", "active_disruption_count",
    "resource_recovery_completed_count", "resource_time_to_recover_mean",
    "obstacle_blocked_steps_total", "resource_disruption_loss_area",
    "swap_conflict_count",
]


def verify(scn: dict, rec: dict) -> tuple[dict | None, str]:
    """返回 (verified_label | None, note)"""
    v, lbl = scn["verifier"], scn["label"]
    s = rec.get("summary", {})
    if "error" in rec:
        return None, f"episode_error: {rec['error'][:60]}"
    if v == "injected_machine":
        return lbl, "by_construction"
    if v == "injected_combo":
        return lbl, "by_construction"
    if v == "preset_stochastic":
        return lbl, "by_construction"
    if v == "baseline":
        # 基线若呈现涌现签名, 改标 (v1 S0 双标签教训)
        if not rec.get("finished", True):
            if (s.get("agv_waiting_time_total") or 0) > 2000 and not (s.get("tasked_stationary_count") or 0):
                return {"loc_type": "task_pool", "loc_target": "task_pool",
                        "cause": "starvation_livelock"}, "baseline_relabeled_starvation"
            if (s.get("tasked_stationary_count") or 0) > 0:
                return {"loc_type": "corridor", "loc_target": "corridor",
                        "cause": "blocking_livelock"}, "baseline_relabeled_blocking"
        if (s.get("operation_queue_waiting_time_mean") or 0) > 20:
            return {"loc_type": "machines", "loc_target": "machines",
                    "cause": "machine_bottleneck"}, "baseline_relabeled_bottleneck"
        return lbl, "ok"
    if v == "emergent_starvation":
        if not rec.get("finished", True) and (s.get("agv_waiting_time_total") or 0) > 2000 \
                and not (s.get("tasked_stationary_count") or 0):
            return lbl, "signature_confirmed"
        return None, (f"drop: starvation signature absent (fin={rec.get('finished')}, "
                      f"agv_wait={s.get('agv_waiting_time_total')}, stationary={s.get('tasked_stationary_count')})")
    if v == "emergent_blocking":
        if not rec.get("finished", True) and (s.get("tasked_stationary_count") or 0) > 0:
            return lbl, "signature_confirmed"
        # 拥塞签名放宽: 完工但走廊阻塞延迟极高且呈现滞留
        if (s.get("transport_blocking_delay_mean") or 0) > 30 and (s.get("tasked_stationary_count") or 0) > 0:
            return lbl, "signature_confirmed_relaxed"
        return None, (f"drop: blocking signature absent (fin={rec.get('finished')}, "
                      f"blocking={s.get('transport_blocking_delay_mean')}, stationary={s.get('tasked_stationary_count')})")
    return None, f"unknown verifier {v}"


def event_visible(rec: dict, scn: dict) -> bool | None:
    """注入事件是否进入事件样本 (尽力核验)"""
    exc = scn.get("exception_config") or {}
    sched = exc.get("schedule") or []
    if not sched:
        return None
    evs = rec.get("events_sample") or []
    for item in sched:
        t = item["type"]
        hit = any(e.get("type") == t and (e.get("payload") or {}).get(
            "machine_id", (e.get("payload") or {}).get("agv_id")) == item.get(
            "machine_id", item.get("agv_id")) for e in evs)
        if not hit:
            return False
    return True


def build_archive(scn: dict, rec: dict, label: dict, note: str) -> dict:
    s = rec.get("summary", {})
    kpi = {k: s.get(k) for k in KPI_FIELDS if s.get(k) is not None}
    evs = [{"id": f"EVT{i+1:03d}", "step": e.get("step"), "type": e.get("type"),
            "payload": e.get("payload")}
           for i, e in enumerate(rec.get("events_sample") or [])]
    ner = rec.get("schedule_nervousness", {})
    ost = rec.get("orchestrator_stats", {})
    base = {
        "scen_id": scn["scen_id"],
        "config": {"instance": scn["instance"], "num_agv": scn["num_agv"],
                   "policy": scn["policy"], "seed": scn["seed"],
                   "map": scn["map_name"]},
        "kpi": kpi,
        "revisions": {"count": rec.get("n_plan_revisions"),
                      "fail_count": ost.get("revision_fail_count"),
                      "changed_operations": ner.get("changed_operation_count"),
                      "machine_assignment_changes": ner.get("machine_assignment_change_count")},
        "query": "为何该 episode 的完工状态如此? 定位关键资源(loc_type/loc_target)与原因(cause), 并给干预建议。",
        "ground_truth": {"loc_type": label["loc_type"],
                         "loc_target": label["loc_target"],
                         "cause": label["cause"]},
        "meta": {"verify_note": note, "n_events_total": rec.get("n_events"),
                 "makespan": rec.get("makespan"), "finished": rec.get("finished")},
    }
    easy = dict(base, case_id=f"{scn['scen_id']}|easy", variant="easy", events=evs)
    hard = dict(base, case_id=f"{scn['scen_id']}|hard", variant="hard", events=[],
                meta=dict(base["meta"], hard_note="事件流已被移除, 仅凭统计特征推断"))
    return easy, hard


def main():
    scns = {json.loads(l)["scen_id"]: json.loads(l)
            for l in (HERE / "scenarios_v2.jsonl").read_text().splitlines() if l.strip()}
    cases, dropped, kept = [], [], 0
    for f in sorted((HERE / "episodes").glob("*.json")):
        sid = f.stem
        if sid not in scns:
            continue
        rec = json.loads(f.read_text())
        label, note = verify(scns[sid], rec)
        if label is None:
            dropped.append({"scen_id": sid, "reason": note})
            continue
        easy, hard = build_archive(scns[sid], rec, label, note)
        if scns[sid]["verifier"].startswith("injected"):
            easy["meta"]["event_visible"] = event_visible(rec, scns[sid])
        cases += [easy, hard]
        kept += 1
    (HERE / "cases_v2.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    (HERE / "build_report.json").write_text(json.dumps(
        {"kept_scenarios": kept, "dropped": dropped, "n_cases": len(cases)},
        ensure_ascii=False, indent=1))
    from collections import Counter
    print(f"kept={kept} dropped={len(dropped)} cases={len(cases)} (easy+hard)")
    print("cause:", Counter(c["ground_truth"]["cause"] for c in cases))
    for d in dropped:
        print("  DROP", d["scen_id"], "->", d["reason"][:90])


if __name__ == "__main__":
    main()
