"""方向7 v2: 程序化核验器 — "阅卷人" (方法论 v0.1 §1 步骤3 / §6)
================================================================
两种规则角色严格分开 (§6):
  出题人 = 注入条件本身 (注入类答案由构造为真, 无需检测);
  阅卷人 = 事后核验器, 读全量轨迹算全局谓词 (确诊手段/化验单, 非诊断过程)。

每个核验器返回 (label|None, confidence, facts):
  label=None 表示判据不显著 (案例弃用或降级为其他类);
  facts 是判据依据, 进入档案 ground_truth 附注。
"""

from __future__ import annotations

from typing import Optional


def _f(rec: dict, *keys, default=None):
    """容忍键名差异的浮点取值 (v1/v2 summary 键名混用)。"""
    s = rec.get("summary", {}) or {}
    for k in keys:
        v = s.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return default


def _series(rec: dict, *keys) -> list[tuple[int, float]]:
    """从 metrics_series 提取一条时间序列 (容忍键名)。"""
    out = []
    for p in rec.get("metrics_series", []):
        for k in keys:
            if k in p and isinstance(p[k], (int, float)):
                out.append((int(p.get("step", 0)), float(p[k])))
                break
    return out


def _longest_idle_streak(rec: dict, *, idle_below: float = 0.02,
                         min_steps: int = 100) -> int:
    """AGV 集体空转的最长连续步数 (按降采样序列折算)。"""
    busy = _series(rec, "agv_busy_utilization", "agv_utilization")
    if not busy:
        return 0
    best = cur = 0
    for _, v in busy:
        if v < idle_below:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * 8  # 乘回步距 METRIC_STRIDE


def verify_injection(rec: dict) -> tuple[Optional[str], float, list]:
    """注入类: 标签由构造为真; 核验器只确认事件确实发生且命中目标。"""
    cls = rec.get("target_class")
    evs = rec.get("events_all") or []
    if cls in ("baseline",):
        if rec.get("n_events", 0) == 0 and rec.get("finished"):
            return "baseline", 1.0, ["n_events=0 且正常完工"]
        if rec.get("n_events", 0) == 0:
            # 无事件但未完工: 降级到活锁类判据 (无事件截断大概率是活锁)
            for fb in (verify_starvation, verify_blocking):
                lbl, conf, facts = fb(rec)
                if lbl:
                    return lbl, conf * 0.9, ["无注入事件, 判据降级"] + facts
            return None, 0.0, ["无事件且未完工, 但活锁判据不显著"]
        return None, 0.0, [f"意外事件 n_events={rec['n_events']}"]
    if not evs:
        return None, 0.0, ["注入事件未触发"]
    facts = []
    types = {e.get("type") for e in evs if isinstance(e, dict)}
    if cls == "disruption_machine" and "machine_breakdown" in types:
        facts.append(f"注入事件 {sorted(types)}")
        return cls, 1.0, facts
    if cls == "disruption_machine_agv" and {"machine_breakdown", "agv_breakdown"} <= types:
        facts.append(f"注入事件 {sorted(types)}")
        return cls, 1.0, facts
    if cls == "disruption_stochastic" and types & {
            "machine_breakdown", "agv_breakdown", "temporary_obstacle"}:
        facts.append(f"概率故障流事件 {sorted(types)} x{rec.get('n_events')}")
        return cls, 1.0, facts
    if cls == "route_disruption" and "temporary_obstacle" in types:
        facts.append(f"路由障碍流 x{rec.get('n_events')}")
        return cls, 1.0, facts
    return None, 0.0, [f"事件类型不符: {sorted(types)}"]


def verify_starvation(rec: dict) -> tuple[Optional[str], float, list]:
    """饥饿活锁 = 任务池非空 ∧ 全 AGV 空转持续 ≥N 步 (未完工)。"""
    if rec.get("finished"):
        return None, 0.0, []
    streak = _longest_idle_streak(rec)
    throughput = _f(rec, "throughput_jobs", default=0)
    if streak >= 40 and throughput == 0:  # 序列按 8 步降采样, 40≈5 个采样点
        return "starvation_livelock", 0.9, [
            f"AGV 连续空转 ≥{streak} 步且零完工 (未完工)"]
    if streak >= 40:
        return "starvation_livelock", 0.7, [f"AGV 连续空转 ≥{streak} 步"]
    return None, 0.0, [f"idle_streak={streak}"]


def verify_blocking(rec: dict) -> tuple[Optional[str], float, list]:
    """走廊拥塞/对峙 = 有任务在身却 stationary, 或阻塞延迟签名显著。"""
    if rec.get("finished"):
        return None, 0.0, []
    stationary = _f(rec, "tasked_stationary_count", default=0)
    blocking = _f(rec, "transport_blocking_delay_mean", default=0.0)
    if stationary and stationary > 0:
        return "blocking_livelock", 0.8, [
            f"带任务 stationary AGV x{stationary:.0f}, 阻塞延迟均值 {blocking:.1f}"]
    if blocking > 8:  # pilot 校准: 健康基线 blocking≈0-3
        return "blocking_livelock", 0.6, [f"阻塞延迟均值 {blocking:.1f}"]
    return None, 0.0, [f"stationary={stationary}, blocking={blocking:.1f}"]


def verify_bottleneck(rec: dict) -> tuple[Optional[str], float, list]:
    """机器瓶颈 = 完工但机器队列等待显著 (v1 阈值 20, v2 用分位数校准)。"""
    if not rec.get("finished"):
        return None, 0.0, []
    q = _f(rec, "operation_queue_waiting_time_mean",
           "queue_wait_mean", default=0.0)
    if q > 5:  # pilot 校准: 健康基线 queue_wait≈1
        return "machine_bottleneck", 0.75, [f"机器队列等待均值 {q:.1f}"]
    return None, 0.0, [f"queue_wait={q:.1f}"]


def verify_plan_mismatch(rec: dict) -> tuple[Optional[str], float, list]:
    """计划失配 = 扰动已发生 ∧ 零修订 ∧ (未完工或 nervousness 显著)。"""
    if not str(rec.get("config", {}).get("policy", "")).startswith("cpsat"):
        return None, 0.0, []
    if rec.get("n_events", 0) == 0:
        return None, 0.0, []
    if rec.get("n_plan_revisions", 0) > 0:
        return None, 0.0, ["已有修订, 非静态失配"]
    nervous = rec.get("schedule_nervousness") or {}
    nerv_val = next((float(v) for v in nervous.values()
                     if isinstance(v, (int, float))), 0.0)
    if not rec.get("finished"):
        return "plan_mismatch", 0.7, [
            f"扰动 x{rec['n_events']} 后零修订且未完工"]
    if nerv_val > 0:
        return "plan_mismatch", 0.6, [
            f"扰动 x{rec['n_events']} 后零修订, nervousness={nerv_val:.2f}"]
    return None, 0.0, ["扰动后零修订但无失配证据"]


# 阅卷顺序: 注入类由构造优先; 无事件类按判据特异性从高到低
def verify_case(rec: dict) -> dict:
    """主入口: 返回 {label, confidence, facts, ambiguous}。"""
    cls = rec.get("target_class")
    if cls in ("baseline", "disruption_machine", "disruption_machine_agv",
               "disruption_stochastic", "route_disruption"):
        label, conf, facts = verify_injection(rec)
    elif cls == "starvation_livelock":
        label, conf, facts = verify_starvation(rec)
        if label is None:  # 判据不显著时允许降级到拥塞
            label, conf, facts = verify_blocking(rec)
            facts = ["目标类判据未命中, 降级拥塞判据"] + facts
    elif cls == "blocking_livelock":
        label, conf, facts = verify_blocking(rec)
        if label is None:
            label, conf, facts = verify_starvation(rec)
            facts = ["目标类判据未命中, 降级饥饿判据"] + facts
    elif cls == "machine_bottleneck":
        label, conf, facts = verify_bottleneck(rec)
    elif cls == "plan_mismatch":
        label, conf, facts = verify_plan_mismatch(rec)
    else:
        label, conf, facts = None, 0.0, ["未知目标类"]
    return {"label": label, "confidence": conf, "facts": facts,
            "ambiguous": label is None or label != cls}
