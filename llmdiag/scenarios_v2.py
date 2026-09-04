"""方向7 v2: 场景网格采样器 (方法论 v0.1 §1)
=================================================
数据不是采集来的, 是制造出来的: 每条数据 = 一次受控注入的仿真运行。

8 类归因 + 1 个 Unseen 留出类 (route_disruption, 标签在手、词表抽走):
  注入类 (出题人=注入条件, 答案由构造为真):
    baseline               无事件基线
    disruption_machine     单机故障 (随机目标机/时刻/时长)
    disruption_machine_agv 机+AGV 双故障
    disruption_stochastic  概率故障流 (moderate_failure preset)
  无事件类 (以运行配置触发, 标签由程序判据 verifiers.py 核验):
    starvation_livelock    任务池非空而 AGV 集体空转 (迷宫+低K+greedy)
    blocking_livelock      走廊对峙型活锁 (迷宫+高K)
    machine_bottleneck     机器队列瓶颈 (瓶颈实例+低K)
    plan_mismatch          CP-SAT 一次成型计划在扰动下失配 (静态臂+故障流)
  Unseen 留出:
    route_disruption       路由障碍流 (routing_disruption preset), 词表中不存在

v1 退化修复: 故障目标/时刻/时长全部随机化 (v1 永远 0 号机 step150)。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

INSTANCES = ["mk01", "mk02", "mk03", "mk04", "mk05"]
# v2 试点: 本机 FJSP 微服务镜像 (skyengine-fjsp-*) 均为旧版 API, 不支持
# worktree 引擎 online_fjsp_gateway 的 /solve+/cancel 无状态接口;
# 故统一 greedy-reactive (进程内 astar, 零微服务依赖)。
# 待新版 fjsp 服务镜像就位后, 恢复四臂: greedy-reactive/cpsat-static/full/partial,
# 并重新纳入 plan_mismatch 类 (静态臂+故障流, 零修订判据)。
PILOT_POLICIES = ["greedy-reactive"]
KS = [3, 4, 6]
MAP_FAMILY = "data/mapf/medium_maps.yaml"

# 每类产出配额 (方法论 §1: 正式 ≥200 条且每类 ≥25; Unseen 单独留出)
CLASS_QUOTA_PILOT = {
    "baseline": 3, "disruption_machine": 3, "disruption_machine_agv": 3,
    "disruption_stochastic": 3, "starvation_livelock": 3, "blocking_livelock": 3,
    "machine_bottleneck": 3, "route_disruption": 3,
}
CLASS_QUOTA_FULL = {
    "baseline": 25, "disruption_machine": 30, "disruption_machine_agv": 30,
    "disruption_stochastic": 25, "starvation_livelock": 25,
    "blocking_livelock": 25, "machine_bottleneck": 25, "route_disruption": 20,
}

# 诊断查询 (方法论 §3 prompt 的 [查询] 段)
QUERIES = {
    "baseline": "该 episode 运行是否正常? 若正常请说明判断依据, 给出干预建议 none。",
    "disruption_machine": "为何该 episode 的 makespan 超预期? 定位关键资源与原因, 给干预建议。",
    "disruption_machine_agv": "为何该 episode 的 makespan 超预期且运输同时受阻? 定位资源与原因, 给干预建议。",
    "disruption_stochastic": "该 episode 的性能波动由何引起? 定位扰动来源, 给干预建议。",
    "starvation_livelock": "为何该 episode 未完工且 AGV 闲置? 判断失效模式, 给干预建议。",
    "blocking_livelock": "为何该 episode 有任务在身却长期无进展? 判断失效模式, 给干预建议。",
    "machine_bottleneck": "为何该 episode 完工时间远超计划? 定位瓶颈资源, 给干预建议。",
    "plan_mismatch": "为何扰动发生后 episode 表现持续恶化? 检查计划是否失配, 给干预建议。",
    "route_disruption": "为何该 episode 的运输时间显著变长? 定位原因, 给干预建议。",
}

# 干预词表 (方法论 §3; watchdog/machine_reroute 引擎未实现, v1 已剔除)
INTERVENTION_VOCAB = (
    "padding(alpha=<a>) alpha∈{0.1,0.2,0.3} | fleet(K=<k>) | "
    "periodic_revision(<T>) | assigner(<name>) | none"
)


@dataclass
class CaseSpec:
    case_id: str
    target_class: str            # 采样时的目标类 (标签仍由 verifiers 核验)
    instance: str
    policy: str
    num_agv: int
    map_name: str
    seed: int
    exception_config: dict = field(default_factory=dict)
    route_solver: str = "astar"  # v2 试点统一进程内 astar, 见 README 偏差说明
    query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _load_healthy_maps() -> list[str]:
    """注入类/基线用"健康地图池" (screen_maps.py 产出的完工地图)。
    无筛选结果时退回 v1 同款 seed-0000。"""
    f = Path(__file__).resolve().parents[1] / ("llmdiag/results_v2/map_screen.json")
    if f.exists():
        pool = [m["map"] for m in json.loads(f.read_text()).get("healthy", [])]
        if pool:
            return pool
    return ["medium-mazes-seed-0000"]


def _mk_case(rng: random.Random, cls: str, idx: int) -> CaseSpec:
    """按类采样一个场景; 注入参数全随机化。"""
    case_id = f"v2_{cls}_{idx:03d}"
    inst = rng.choice(INSTANCES)
    seed = rng.randrange(1000, 999999)
    healthy = cls in ("baseline", "disruption_machine",
                      "disruption_machine_agv", "disruption_stochastic",
                      "machine_bottleneck", "route_disruption")
    if healthy:
        map_name = rng.choice(_load_healthy_maps())
    else:  # 活锁类: 病态地图正是其病灶, 全域采样
        map_name = f"medium-mazes-seed-{rng.randrange(0, 16):04d}"

    # 健康类网格收缩: mk04 超出 4096 步预算, K=6 是 greedy 阻塞峰值 (方向5)
    # —— 否则对照组被活锁污染 (截断非注入所致)
    if cls == "baseline":
        policy, k = rng.choice(PILOT_POLICIES), rng.choice([3, 4])
        inst = rng.choice(["mk01", "mk02"])
        exc = {"enabled": False, "random_seed": seed}
    elif cls == "disruption_machine":
        policy, k = rng.choice(PILOT_POLICIES), rng.choice([3, 4])
        inst = rng.choice(["mk01", "mk02"])
        step = rng.randrange(100, 300)
        exc = {"enabled": True, "random_seed": seed, "schedule": [
            {"step": step, "type": "machine_breakdown",
             "machine_id": rng.randrange(0, 5),          # 目标机随机 (v1 恒 0)
             "duration_steps": rng.randrange(30, 80)},
        ]}
    elif cls == "disruption_machine_agv":
        policy, k = rng.choice(PILOT_POLICIES), rng.choice([3, 4])
        inst = rng.choice(["mk01", "mk02"])
        exc = {"enabled": True, "random_seed": seed, "schedule": [
            {"step": rng.randrange(100, 260), "type": "machine_breakdown",
             "machine_id": rng.randrange(0, 5),
             "duration_steps": rng.randrange(30, 80)},
            {"step": rng.randrange(260, 430), "type": "agv_breakdown",
             "agv_id": rng.randrange(0, k),
             "duration_steps": rng.randrange(15, 40)},
        ]}
    elif cls == "disruption_stochastic":
        policy, k = rng.choice(PILOT_POLICIES), rng.choice([3, 4])
        inst = rng.choice(["mk01", "mk02"])
        exc = {"enabled": True, "random_seed": seed, "preset": "moderate_failure"}
    elif cls == "starvation_livelock":
        # E2 签名: 迷宫 + 低 K + greedy ⇒ 任务池非空而 AGV 集体空转
        policy, k = "greedy-reactive", rng.choice([3, 4])
        exc = {"enabled": False, "random_seed": seed}
    elif cls == "blocking_livelock":
        # 方向5 签名: 窄走廊 + 高 K ⇒ 对峙阻塞
        policy, k = "greedy-reactive", rng.choice([6])
        exc = {"enabled": False, "random_seed": seed}
    elif cls == "machine_bottleneck":
        # 瓶颈实例 + 低 K ⇒ 机器队列堆积 (greedy 也可触发, 判据与策略无关)
        policy, k = rng.choice(PILOT_POLICIES), rng.choice([3, 4])
        inst = rng.choice(["mk02", "mk03"])
        exc = {"enabled": False, "random_seed": seed}
    elif cls == "route_disruption":
        policy, k = rng.choice(PILOT_POLICIES), rng.choice([3, 4])
        inst = rng.choice(["mk01", "mk02"])
        exc = {"enabled": True, "random_seed": seed, "preset": "routing_disruption"}
    else:
        raise ValueError(
            f"unknown class {cls} (plan_mismatch 需 cpsat 微服务, 见文件头说明)")

    return CaseSpec(
        case_id=case_id, target_class=cls, instance=inst, policy=policy,
        num_agv=k, map_name=map_name, seed=seed, exception_config=exc,
        route_solver="astar", query=QUERIES[cls],
    )


def sample_grid(per_class: int | None = None, master_seed: int = 20260904,
                quota: dict | None = None) -> list[CaseSpec]:
    """按类配额采样场景网格 (确定性: 同 master_seed 同网格)。"""
    quota = quota or (CLASS_QUOTA_PILOT if per_class in (None, 0) else
                      {k: per_class for k in CLASS_QUOTA_PILOT})
    if per_class:
        quota = {k: per_class for k in CLASS_QUOTA_PILOT}
    rng = random.Random(master_seed)
    specs = []
    for cls, n in quota.items():
        for i in range(n):
            specs.append(_mk_case(rng, cls, i))
    return specs


if __name__ == "__main__":
    for s in sample_grid(per_class=2)[:6]:
        print(s.case_id, s.instance, s.policy, f"K={s.num_agv}",
              s.map_name, s.exception_config.get("preset") or
              (s.exception_config.get("schedule") or "") or "no-event")
