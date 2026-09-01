"""
方向1: 闭环在线协同 — 恢复编排器 (Recovery Orchestrator)
========================================================
引擎现状: JobSolver 的重调度 API (request_plan_revision /
request_partial_schedule_repair) 已存在但无人调用——即调度层在扰动下
缺少闭环。本模块补上这个策略层, 构成被研究的"闭环协同策略"空间:

  策略 = (求解器 σ_J, 触发器 τ, 修复范围 κ)
    τ ∈ {never, event, periodic-K}
    κ ∈ {full(全域残差重规划), partial(围绕故障机器的局部修复)}

论文对应: 论文_1_闭环在线协同 §方法
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from common.engine_adapter import (
    ENGINE_ROOT,
    build_configs,
    load_fjsp_json,
    load_map,
)

import sys

if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from sky_executor.grid_factory.factory.grid_factory_env import (  # noqa: E402
    GridFactoryEnv,
)
from sky_executor.grid_factory.factory.Component.Coordinator.coordinator import (  # noqa: E402
    Coordinator,
)

try:
    from sky_executor.grid_factory.factory.Utils.transfer_estimator import (  # noqa: E402
        TransferTimeEstimator,
    )
except Exception:  # pragma: no cover
    TransferTimeEstimator = None

# 会触发重调度决策的事件类型 (扰动发生与恢复都算)
DISTURBANCE_EVENT_TYPES = {
    "machine_breakdown",
    "agv_breakdown",
    "temporary_obstacle",
    "machine_recovered",
    "machine_repair",
    "agv_recovered",
    "agv_repair",
}

QUALIFYING_PREFIXES = ("machine", "agv")


class RecoveryOrchestrator:
    """观测事件流并按 (τ, κ) 策略调用调度层重规划。"""

    def __init__(
        self,
        coordinator: Coordinator,
        env: GridFactoryEnv,
        *,
        trigger: str = "event",           # never | event | periodic-K
        scope: str = "full",              # full | partial
        min_replan_interval: int = 10,    # 两次重规划的最小间隔(步), 防重规划风暴
        verbose: bool = False,
    ):
        self.coordinator = coordinator
        self.env = env
        self.trigger = trigger
        self.scope = scope
        self.min_replan_interval = min_replan_interval
        self.verbose = verbose
        self.last_replan_step = -10**9
        self.period = None
        if trigger.startswith("periodic-"):
            self.period = int(trigger.split("-", 1)[1])
        self.stats = {
            "revision_count": 0,
            "revision_fail_count": 0,
            "triggered_events": [],
            "planner_wall_s": 0.0,
        }

    # -- 触发判定 ---------------------------------------------------------
    def _should_replan(self, step: int, events: list) -> Optional[dict]:
        if self.trigger == "never":
            return None
        if self.period is not None:
            if step > 0 and step % self.period == 0:
                return {"type": "periodic", "step": step}
            return None
        # event 触发: 只看机器/AGV 类事件 (路由层自身处理障碍)
        for e in events:
            etype = str(e.get("type", ""))
            if etype in DISTURBANCE_EVENT_TYPES or any(
                etype.startswith(p) for p in QUALIFYING_PREFIXES
            ):
                if step - self.last_replan_step >= self.min_replan_interval:
                    return e
        return None

    # -- 执行重规划 -------------------------------------------------------
    def on_step(self, obs: dict, step: int) -> None:
        injector = getattr(self.env, "exception_injector", None)
        if injector is None:
            return
        events = injector.get_step_events() or []
        ev = self._should_replan(step, events)
        if ev is None:
            return

        solver = self.coordinator.job_solver
        if not hasattr(solver, "request_plan_revision"):
            return  # 规则求解器天然闭环, 无需编排
        if not getattr(solver, "initialized", False):
            return

        job_obs = obs.get("job_observation")
        if job_obs is None:
            return

        t0 = time.time()
        self.last_replan_step = step
        activated = False
        prefer_partial = (
            self.scope == "partial" and ev.get("type") == "machine_breakdown"
        )
        # 策略显式: scope=partial 直接局部修复; 否则 U4 全域 -> (机器事件) U3 兜底
        if prefer_partial and hasattr(solver, "request_partial_schedule_repair"):
            try:
                revision = solver.request_partial_schedule_repair(
                    job_obs,
                    execution_env=self.env,
                    route_solver=self.coordinator.route_solver,
                    failed_machine_id=int(ev.get("machine_id", -1)),
                    trigger={"type": "orchestrator-scoped", "event": ev, "step": step},
                )
                if revision is not None:
                    self.stats["revision_count"] += 1
                    activated = True
            except Exception as e:  # noqa: BLE001
                self.stats["revision_fail_count"] += 1
                if self.verbose:
                    print(f"[orchestrator] scoped partial failed at step {step}: {e}")
        if not activated:
            try:
                revision = solver.request_plan_revision(
                    job_obs,
                    recovery_level=4,  # U4 = joint_full_rescheduling
                    trigger={"type": "orchestrator", "event": ev, "step": step},
                    activate=True,
                    activation_env=self.env,
                    route_solver=self.coordinator.route_solver,
                )
                self.stats["revision_count"] += 1
                activated = str(getattr(revision, "status", "")) != "activation_failed"
                if not activated:
                    self.stats["revision_fail_count"] += 1
            except Exception as e:  # noqa: BLE001
                self.stats["revision_fail_count"] += 1
                if self.verbose:
                    print(f"[orchestrator] U4 full revision failed at step {step}: {e}")
        if not activated and ev.get("type") == "machine_breakdown" and hasattr(
            solver, "request_partial_schedule_repair"
        ):
            try:
                revision = solver.request_partial_schedule_repair(
                    job_obs,
                    execution_env=self.env,
                    route_solver=self.coordinator.route_solver,
                    failed_machine_id=int(ev.get("machine_id", -1)),
                    trigger={"type": "orchestrator-fallback", "event": ev, "step": step},
                )
                if revision is not None:
                    self.stats["revision_count"] += 1
                    self.stats["partial_fallback_count"] = (
                        self.stats.get("partial_fallback_count", 0) + 1
                    )
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"[orchestrator] U3 partial fallback failed at step {step}: {e}")
        if not activated:
            # 允许稍后事件(如修复完成)再次触发, 而非被最小间隔挡住
            self.last_replan_step = step - self.min_replan_interval
        self.stats["planner_wall_s"] += time.time() - t0
        self.stats["triggered_events"].append({"step": step, "event": ev})


def run_closed_episode(
    fjsp_path: str | Path,
    map_file: str | Path,
    map_name: Optional[str] = None,
    *,
    policy: str = "cpsat-full",       # greedy-reactive | cpsat-static | cpsat-full | cpsat-partial
    exception_config: Optional[dict] = None,
    num_agv: int = 4,
    seed: int = 42,
    max_steps: int = 4096,
    assigner: str = "nearest",
    route_solver_name: str = "rolling_mapf_http",
    route_solver_kwargs: Optional[dict] = None,
    fjsp_service_url: str = "http://fjsp:8002",
    fjsp_time_limit: float = 10.0,
    mapf_service_url: str = "http://mapf:8001",
    mapf_time_limit_ms: int = 500,
    verbose: bool = False,
    trigger_override: Optional[str] = None,
    processing_time_config: Optional[dict] = None,
) -> dict:
    """运行一个带扰动注入与闭环策略的 episode。

    policy:
      greedy-reactive : 规则调度, 每步重建计划 (规则级闭环基线)
      cpsat-static    : CP-SAT 一次规划, 扰动下不修正 (开环对照, à la generate-then-refine)
      cpsat-full      : CP-SAT + 事件触发全域残差重规划
      cpsat-partial   : CP-SAT + 事件触发局部修复 (围绕故障机器)
    trigger_override: 覆盖触发器 (如 "periodic-100", 供鲁棒性实验复用)。
    """
    t0 = time.time()
    trigger, scope = "never", "full"
    job_solver_name = "online_fjsp"
    if policy == "greedy-reactive":
        job_solver_name = "greedy"
    elif policy == "cpsat-static":
        trigger = "never"
    elif policy == "cpsat-full":
        trigger, scope = "event", "full"
    elif policy == "cpsat-partial":
        trigger, scope = "event", "partial"
    else:
        raise ValueError(f"unknown policy {policy}")
    if trigger_override is not None:
        trigger = trigger_override

    fjsp_data = load_fjsp_json(fjsp_path)
    grid_map = load_map(map_file, map_name)
    observation_type = "MAPF" if route_solver_name == "rolling_mapf_http" else "default"
    collision_system = "soft" if route_solver_name == "rolling_mapf_http" else "priority"

    grid_cfg, machine_cfg, job_cfg = build_configs(
        fjsp_data, grid_map,
        num_agv=num_agv, seed=seed, max_episode_steps=max_steps,
        observation_type=observation_type, collision_system=collision_system,
    )
    env = GridFactoryEnv(
        grid_config=grid_cfg, machine_config=machine_cfg, job_config=job_cfg,
        random_target=False, exception_config=exception_config,
        processing_time_config=processing_time_config,
    )
    obs, info = env.reset()

    transfer_estimator = None
    if job_solver_name == "greedy" and TransferTimeEstimator is not None:
        try:
            transfer_estimator = TransferTimeEstimator(
                env.pogema_env.machines,
                env.pogema_env.grid.get_obstacles(),
                use_feedback=True,
            )
        except Exception:
            transfer_estimator = None

    job_kwargs = {}
    if job_solver_name == "online_fjsp":
        job_kwargs = {
            "service_url": fjsp_service_url,
            "algorithm": "cp_sat",
            "config": {"time_limit": fjsp_time_limit, "num_workers": 1, "seed": seed},
        }
    if route_solver_kwargs is None and route_solver_name == "rolling_mapf_http":
        route_solver_kwargs = {
            "service_url": mapf_service_url,
            "time_limit_ms": mapf_time_limit_ms,
            "lns_init_algo": "EECBS",
            "planning_horizon": 10,
            "execution_window": 5,
        }

    coordinator = Coordinator(
        job_solver=job_solver_name,
        route_solver=route_solver_name,
        assigner=assigner,
        job_solver_kwargs=job_kwargs,
        route_solver_kwargs=route_solver_kwargs,
        transfer_time_estimator=transfer_estimator,
    )
    orch = RecoveryOrchestrator(
        coordinator, env, trigger=trigger, scope=scope, verbose=verbose
    )

    finished = False
    steps = 0
    events_total = []
    for i in range(max_steps):
        actions = coordinator.decide(obs)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps = i + 1
        injector = getattr(env, "exception_injector", None)
        if injector is not None:
            evs = injector.get_step_events() or []
            if evs:
                events_total.extend(evs)
        orch.on_step(obs, steps)
        if transfer_estimator is not None:
            for agent in obs.get("task_observation", {}).get("agents", []):
                if agent.finished_tasks:
                    for task in agent.finished_tasks:
                        transfer_estimator.update_from_task(task)
        if terminations.get("job_done"):
            finished = True
            break
        if all(truncations.values()):
            break

    summary = env.metrics_hub.get_episode_summary()
    nervous = {}
    try:
        nervous = coordinator.get_schedule_nervousness()
    except Exception:
        pass
    plan_revisions = []
    jd = getattr(coordinator.job_solver, "get_plan_revisions", None)
    if callable(jd):
        try:
            plan_revisions = jd()
        except Exception:
            pass

    return {
        "config": {
            "fjsp": str(fjsp_path), "map_file": str(map_file), "map_name": map_name,
            "num_agv": num_agv, "seed": seed, "policy": policy,
            "exception_config": exception_config,
        },
        "finished": finished,
        "steps": steps,
        "makespan": summary.get("completed_makespan") if finished else max_steps,
        "summary": summary,
        "orchestrator_stats": orch.stats,
        "n_events": len(events_total),
        "events_sample": events_total[:10],
        "schedule_nervousness": {
            k: v for k, v in nervous.items() if k != "schedule_revision_events"
        } if isinstance(nervous, dict) else {},
        "n_plan_revisions": len(plan_revisions),
        "wall_time_s": round(time.time() - t0, 2),
    }
