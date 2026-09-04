"""方向7 v2: 三层程序化判分器 (方法论 v0.1 §4)
================================================
| 输出     | 判法                                   | 指标 |
|----------|----------------------------------------|------|
| 定位/归因| 与 ground truth 比对 (+粗粒度部分分)   | 定位/归因准确率, JRA |
| 证据     | 逐条回查档案 (ID 存在 ∧ 数值一致)      | 忠实度 = 通过/总条数 |
| 干预     | 解析->改配置->同 seed 重仿真->ΔKPI     | 干预成功率 P(ΔKPI≥δ) |

全程零人工标注。对抗注入测试 (塞假证据) 预留 --adversarial 钩子。
"""

from __future__ import annotations

import json
import re

try:
    from llmdiag.episode_runner import apply_intervention
except ImportError:  # 直接以脚本方式运行时
    from episode_runner import apply_intervention

DELTA_SUCCESS = 0.05  # δ: makespan 相对改善 ≥5% 视为干预有效 (试点档)

LOC_VOCAB_PREFIX = ("machine:", "agv:", "stochastic", "task_pool", "corridor",
                    "machines", "none")
CAUSE_VOCAB = {
    "disruption_machine", "disruption_machine_agv", "disruption_stochastic",
    "starvation_livelock", "blocking_livelock", "machine_bottleneck",
    "plan_mismatch", "baseline", "unseen",
}


def parse_intervention(s: str):
    """词表干预 -> (family, arg); 不在词表返回 (None, raw)。"""
    s = (s or "").strip()
    if s == "none":
        return "none", ""
    for pat, fam in (
        (r"^padding\(alpha\s*=\s*([\d.]+)\)$", "padding"),
        (r"^fleet\(K\s*=\s*(-?\d+)\)$", "fleet"),
        (r"^periodic_revision\((\d+)\)$", "periodic_revision"),
        (r"^assigner\(([a-z_]+)\)$", "assigner"),
    ):
        m = re.match(pat, s)
        if m:
            return fam, m.group(1)
    return None, s


def grade_loc_cause(ans: dict, gt: dict) -> dict:
    loc_ok = ans.get("localization") == gt["localization"]
    cause_ok = ans.get("cause") == gt["cause"]
    coarse_ok = cause_ok or (
        str(ans.get("cause", "")).startswith("disruption")
        and str(gt["cause"]).startswith("disruption"))
    # Unseen 逃生舱: 留出类 (route_disruption) 预测 unseen 是正确行为
    escape = bool(gt["cause"] == "route_disruption"
                  and ans.get("cause") == "unseen")
    return {
        "localization_correct": bool(loc_ok),
        "cause_correct": bool(cause_ok),
        "cause_coarse_correct": bool(coarse_ok),
        "unseen_escape": escape,
        "jra": bool(loc_ok and cause_ok),
        "jra_coarse": bool(loc_ok and coarse_ok),
    }


def grade_evidence(ans: dict, archive_text: str) -> dict:
    """逐条回查: 证据串必须能在档案文本中找到 (ID 存在 ∧ 数值一致)。"""
    ev = ans.get("evidence") or []
    if not isinstance(ev, list):
        ev = [str(ev)]
    if not ev:
        return {"faithfulness": None, "n_evidence": 0, "n_verified": 0,
                "lines": []}
    lines = [ln.strip() for ln in archive_text.splitlines() if ln.strip()]
    ok = []
    for item in ev:
        s = str(item).strip()
        # 容忍诊断器给整行或给 ID 两种粒度
        hit = any(s in ln or ln == s or ln.startswith(s) for ln in lines)
        ok.append(hit)
    return {
        "faithfulness": round(sum(ok) / len(ok), 3),
        "n_evidence": len(ev),
        "n_verified": sum(ok),
        "lines": [str(x)[:60] for x in ev],
    }


def grade_intervention(ans: dict, spec: dict, base_record: dict,
                       rerun_fn, base_makespan: float | None = None) -> dict:
    """解析 -> 改配置 -> 同 seed 重仿真 -> ΔKPI。"""
    fam, arg = parse_intervention(ans.get("intervention", "none"))
    if fam is None:
        return {"intervention": ans.get("intervention"), "executable": False,
                "delta_kpi": None, "success": False,
                "note": f"不在干预词表: {arg}"}
    if fam == "none":
        return {"intervention": "none", "executable": True, "delta_kpi": None,
                "success": None, "note": "不建议干预"}
    new_spec = apply_intervention(spec, ans["intervention"])
    if not new_spec.get("intervention_executable", True):
        return {"intervention": ans["intervention"], "executable": False,
                "delta_kpi": None, "success": False,
                "note": "干预不可执行"}
    try:
        rec = rerun_fn(new_spec)
    except Exception as e:  # noqa: BLE001
        return {"intervention": ans["intervention"], "executable": True,
                "delta_kpi": None, "success": False, "note": f"重仿真失败: {e}"}
    base_ms = base_makespan if base_makespan is not None \
        else base_record.get("makespan")
    new_ms = rec.get("makespan")
    if base_ms is None or new_ms is None:
        return {"intervention": ans["intervention"], "executable": True,
                "delta_kpi": None, "success": False, "note": "缺少 makespan"}
    delta = round(float(base_ms) - float(new_ms), 1)  # 正值 = 完工更早
    rel = delta / max(1.0, float(base_ms))
    if not base_record.get("finished") and rec.get("finished"):
        success = True  # 基线未完工而干预后完工: 直接有效
    else:
        success = rel >= DELTA_SUCCESS
    return {"intervention": ans["intervention"], "executable": True,
            "delta_kpi": delta, "delta_relative": round(rel, 4),
            "rerun_makespan": new_ms, "rerun_finished": rec.get("finished"),
            "success": bool(success), "note": f"同 seed={spec['seed']} 重仿真"}


def grade_case(case: dict, variant: str, ans: dict, rerun_fn=None) -> dict:
    """单案例完整判分。rerun_fn=None 时跳过反事实层。"""
    gt = case["ground_truth"]
    out = {"case_id": case["case_id"], "variant": variant,
           "target_class": case["spec"]["target_class"]}
    out.update(grade_loc_cause(ans, gt))
    out["evidence"] = grade_evidence(
        ans, case["archives"][variant]["archive_text"])
    if rerun_fn is not None:
        out["intervention"] = grade_intervention(
            ans, case["spec"], case["episode"], rerun_fn,
            case["episode"].get("makespan"))
    return out


def summarize(grades: list[dict]) -> dict:
    n = len(grades) or 1
    ev = [g["evidence"]["faithfulness"] for g in grades
          if g.get("evidence", {}).get("faithfulness") is not None]
    ivt = [g["intervention"] for g in grades if g.get("intervention")]
    return {
        "n": len(grades),
        "localization_acc": round(sum(g["localization_correct"] for g in grades) / n, 3),
        "cause_acc": round(sum(g["cause_correct"] for g in grades) / n, 3),
        "cause_acc_coarse": round(sum(g["cause_coarse_correct"] for g in grades) / n, 3),
        "jra": round(sum(g["jra"] for g in grades) / n, 3),
        "unseen_escape_rate": (
            round(sum(1 for g in grades if g["target_class"] == "route_disruption"
                      and g.get("unseen_escape")) /
                  max(1, sum(1 for g in grades
                             if g["target_class"] == "route_disruption")), 3)
            if any(g["target_class"] == "route_disruption" for g in grades)
            else None),
        "evidence_faithfulness_mean": round(sum(ev) / len(ev), 3) if ev else None,
        "intervention_success_rate": round(
            sum(1 for i in ivt if i.get("success")) /
            max(1, sum(1 for i in ivt if i.get("success") is not None)), 3)
        if ivt else None,
    }
