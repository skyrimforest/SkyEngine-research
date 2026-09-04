"""方向7 v2: 档案构建器 — 证据 ID 化的运行档案 (方法论 v0.1 §1 步骤4/5)
========================================================================
一条数据 = (带证据 ID 的运行档案 ≤8k token, 标准答案) 对。

证据 ID 空间:
  CFG:<key>=<value>                      episode 配置
  EVT<nnn> step=.. type=.. target=..     事件表
  REV<nnn> step=.. trigger=.. status=..  计划修订账本
  KPI:<metric>[lo-hi] mean=X peak=Y      分段 KPI 摘要

三变体 (同一题目):
  easy  全量 (考阅读)
  hard  删除事件明细, 只留统计特征 (考推断)
  INT   hard + 公开注入参数 (CF−INT 难度归因用)
"""

from __future__ import annotations

import json
from pathlib import Path

TOKEN_BUDGET = 8000          # 方法论 §1 硬预算
TARGET_TOKENS = 6200         # 留出查询与答案空间
CHARS_PER_TOKEN = 4

# 档案展示的 KPI 指标优先级 (存在才展示; 键名容忍由 _pick 处理)
KPI_PRIMARY = [
    "agv_busy_utilization", "agv_loaded_utilization", "agv_waiting_time_total",
    "tasked_stationary_count", "transport_blocking_delay_mean",
    "operation_queue_waiting_time_mean", "machine_waiting_for_inbound_transfer_ratio",
    "machine_down_steps_total", "agv_down_steps_total", "throughput_jobs",
    "completed_makespan",
]

# verified label -> (localization 词表, 说明)
LOC_MAP = {
    "baseline": "none", "starvation_livelock": "task_pool",
    "blocking_livelock": "corridor", "machine_bottleneck": "machines",
    "plan_mismatch": "machines", "route_disruption": "stochastic",
    "disruption_stochastic": "stochastic",
    "disruption_machine_agv": "machine",   # 具体目标由事件载荷填充
}


def _est_tokens(s: str) -> int:
    return max(1, len(s) // CHARS_PER_TOKEN)


def _event_line(idx: int, e: dict) -> str:
    typ = e.get("type", "?")
    payload = e.get("payload", {}) if isinstance(e.get("payload"), dict) else {}
    step = e.get("step", e.get("time", "?"))
    target = payload.get("machine_id", payload.get("agv_id", "-"))
    dur = payload.get("duration_steps", payload.get("duration", "-"))
    extra = "".join(
        f" {k}={v}" for k, v in payload.items()
        if k not in ("machine_id", "agv_id", "duration_steps", "duration"))
    return f"EVT{idx:03d} step={step} type={typ} target={target} dur={dur}{extra}"


def _kpi_sections(metrics_series: list[dict], max_lines: int) -> list[str]:
    """逐步序列 -> 分段摘要 (均值/峰值/末端), 每段带可回查 ID。"""
    if not metrics_series:
        return []
    steps = [p.get("step", 0) for p in metrics_series]
    keys = [k for k in KPI_PRIMARY if any(k in p for p in metrics_series)]
    # 补充高方差的其他数值指标 (发现于运行时的键)
    known = set(KPI_PRIMARY) | {"step"}
    extra = [k for k in metrics_series[-1].keys() if k not in known
             and isinstance(metrics_series[-1][k], (int, float))]
    keys += extra[:4]
    lines: list[str] = []
    per_metric = max(3, max_lines // max(1, len(keys)))
    for k in keys:
        pts = [(p.get("step", 0), p[k]) for p in metrics_series
               if isinstance(p.get(k), (int, float))]
        if len(pts) < 3:
            continue
        n_seg = min(per_metric, max(3, len(pts) // 6))
        size = max(1, len(pts) // n_seg)
        for si in range(0, len(pts), size):
            chunk = pts[si:si + size]
            if len(chunk) < 2:
                continue
            vals = [v for _, v in chunk]
            mean = sum(vals) / len(vals)
            peak = max(vals)
            last = vals[-1]
            lines.append(
                f"KPI:{k}[{chunk[0][0]}-{chunk[-1][0]}] "
                f"mean={mean:.2f} peak={peak:.2f} end={last:.2f}")
    return lines[:max_lines]


def _fmt_revisions(revs: list) -> list[str]:
    out = []
    for i, r in enumerate(revs[:40]):
        if isinstance(r, dict):
            step = r.get("step", r.get("at", "?"))
            trig = r.get("trigger", r.get("reason", "?"))
            ok = r.get("ok", r.get("status", r.get("success", "?")))
            out.append(f"REV{i:03d} step={step} trigger={trig} status={ok}")
        else:
            out.append(f"REV{i:03d} {str(r)[:80]}")
    return out


def _trim(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    """超预算时按 KPI -> REV -> EVT 的顺序截尾, CFG 与头部保留。"""
    order = ["KPI", "REV", "EVT"]
    while True:
        text = "\n".join("\n".join(v) for v in sections.values())
        if _est_tokens(text) <= TARGET_TOKENS:
            return sections
        dropped = False
        for sec in order:
            if len(sections.get(sec, [])) > 4:
                sections[sec] = sections[sec][: int(len(sections[sec]) * 0.8)]
                dropped = True
                break
        if not dropped:
            return sections


def build_archive(case: dict, variant: str = "easy") -> dict:
    """case = run_v2_pilot 的完整案例记录 (含 episode record + verify + spec)。"""
    rec = case["episode"]
    cfg = rec.get("config", {})
    events = rec.get("events_all") or []
    stats = rec.get("summary", {}) or {}
    inject = cfg.get("exception_config", {}) or {}

    cfg_lines = [
        f"CFG:instance={cfg.get('instance')}",
        f"CFG:policy={cfg.get('policy')}",
        f"CFG:K={cfg.get('num_agv')}",
        f"CFG:map={cfg.get('map_name')}",
        f"CFG:max_steps={rec.get('steps')}/4096 finished={rec.get('finished')}",
        "CFG:route=astar obs=default collision=priority",
    ]
    kpi_lines = [
        f"KPI:makespan={rec.get('makespan')} finished={rec.get('finished')}",
        f"KPI:n_events_total={rec.get('n_events', 0)}",
        f"KPI:n_plan_revisions={rec.get('n_plan_revisions', 0)}",
    ]
    for k in ("agv_busy_utilization", "agv_loaded_utilization",
              "agv_waiting_time_total", "tasked_stationary_count",
              "transport_blocking_delay_mean",
              "operation_queue_waiting_time_mean",
              "machine_waiting_for_inbound_transfer_ratio",
              "machine_down_steps_total", "agv_down_steps_total",
              "throughput_jobs"):
        v = stats.get(k)
        if isinstance(v, (int, float)):
            kpi_lines.append(f"KPI:{k}(episode均值)={v:.3f}")

    evt_lines = [_event_line(i, e) for i, e in enumerate(events)]
    rev_lines = _fmt_revisions(rec.get("plan_revisions", []))
    kpi_seg = _kpi_sections(rec.get("metrics_series", []), max_lines=40)

    sections = {
        "CFG": cfg_lines, "STAT": kpi_lines, "KPI": kpi_seg,
        "EVT": evt_lines, "REV": rev_lines,
    }

    # 统计特征 (hard 档替代事件明细的"只留统计特征")
    type_counts: dict[str, int] = {}
    for e in events:
        t = e.get("type", "?") if isinstance(e, dict) else "?"
        type_counts[t] = type_counts.get(t, 0) + 1
    stat_line = "STAT:event_type_counts=" + (
        json.dumps(type_counts, ensure_ascii=False) if type_counts else "{}")

    if variant == "easy":
        use = dict(sections)
    elif variant == "hard":
        use = {k: list(v) for k, v in sections.items() if k != "EVT"}
        use["STAT"] = use["STAT"] + [stat_line]
    elif variant == "int":
        use = {k: list(v) for k, v in sections.items() if k != "EVT"}
        use["STAT"] = use["STAT"] + [stat_line]
        if inject.get("schedule"):
            use["FAULT"] = ["FAULT:公开注入参数=" + json.dumps(
                inject["schedule"], ensure_ascii=False)]
        elif inject.get("preset"):
            use["FAULT"] = [f"FAULT:公开注入参数 preset={inject['preset']} "
                            f"seed={inject.get('random_seed')}"]
        else:
            use["FAULT"] = ["FAULT:本案例无注入事件"]
    else:
        raise ValueError(variant)

    use = _trim(use)
    text = "\n".join("\n".join(v) for k, v in use.items() if v)
    return {
        "case_id": case["case_id"],
        "variant": variant,
        "archive_text": text,
        "est_tokens": _est_tokens(text),
        "n_evidence_ids": sum(1 for ln in text.splitlines()
                              if ln.startswith(("EVT", "REV", "KPI:", "CFG:"))),
    }


def ground_truth_of(case: dict) -> dict:
    """标准答案: 注入类由构造, 无事件类由核验器 (verifiers 已跑)。"""
    label = (case.get("verify") or {}).get("label") or case["spec"]["target_class"]
    evs = case["episode"].get("events_all") or []

    def _first_target(*types):
        for e in evs:
            if isinstance(e, dict) and e.get("type") in types:
                p = e.get("payload", {}) or {}
                return p.get("machine_id", p.get("agv_id"))
        return None

    if label == "baseline":
        loc = "none"
    elif label == "disruption_machine":
        loc = f"machine:{_first_target('machine_breakdown')}"
    elif label == "disruption_machine_agv":
        m = _first_target("machine_breakdown")
        a = next((e["payload"].get("agv_id") for e in evs
                  if isinstance(e, dict) and e.get("type") == "agv_breakdown"), None)
        loc = f"machine:{m}+agv:{a}"
    elif label == "route_disruption":
        loc = "stochastic"
    else:
        loc = LOC_MAP.get(label, "unknown")
    return {"localization": loc, "cause": label}
