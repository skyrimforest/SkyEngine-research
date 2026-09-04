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
