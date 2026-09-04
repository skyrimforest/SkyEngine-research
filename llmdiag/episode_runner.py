"""方向7 v2: 插桩版 episode runner (方法论 v0.1 §1 步骤2/§2 数据采集)
======================================================================
复刻 closeloop.run_closed_episode 的引擎管线, 但为"档案构建+判分"补齐采集:
  - 全量事件历史 (injector.get_event_history), 供证据 ID 表
  - KPI 逐步序列 (等距降采样), 供分段摘要与拐点检测
  - 修订账本全量 (job_solver.get_plan_revisions), 供 plan_mismatch 判据
  - 干预配置增量 (fleet/periodic_revision/assigner/padding), 供反事实执行器
      * padding(α) = 加工时间确定性放大 (multiplier_uniform low=high=1+α)

偏差说明: v2 试点统一 route_solver=astar (进程内) + default 观测 + priority 碰撞,
不依赖 MAPF 微服务 (mapf-classical 镜像本机缺失); CP-SAT 仍走 fjsp 微服务。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.engine_adapter import (  # noqa: E402
    build_configs, load_fjsp_json, load_map, _wait_for_service,
)
from closeloop.orchestrator import RecoveryOrchestrator  # noqa: E402

import os  # noqa: E402

FJSP_URL = os.getenv("FJSP_SERVICE_URL", "http://fjsp:8002")

METRIC_STRIDE = 8  # KPI 序列降采样步距


def apply_intervention(spec_dict: dict, intervention: str) -> dict:
    """把词表干预解析为 spec 覆盖 (返回新 spec dict; 不可执行返回原样+标记)。"""
    import json as _json
    import re

    spec = dict(spec_dict)
    spec["exception_config"] = _json.loads(_json.dumps(spec.get("exception_config", {})))
    spec["intervention_applied"] = intervention
    spec["intervention_executable"] = True

    s = (intervention or "none").strip()
    if s in ("", "none"):
        return spec
    m = re.match(r"^fleet\(K\s*=\s*(\d+)\)$", s)
    if m:
        spec["num_agv"] = int(m.group(1))
        return spec
    m = re.match(r"^periodic_revision\((\d+)\)$", s)
    if m:
        spec["trigger_override"] = f"periodic-{m.group(1)}"
        return spec
    m = re.match(r"^assigner\(([a-z_]+)\)$", s)
    if m:
        spec["assigner"] = m.group(1)
        return spec
    m = re.match(r"^padding\(alpha\s*=\s*([\d.]+)\)$", s)
    if m:
        a = float(m.group(1))
        spec["processing_time_config"] = {
            "enabled": True, "preset": "none", "random_seed": spec.get("seed", 42),
            "default_distribution": {"dist": "multiplier_uniform",
                                     "low": 1.0 + a, "high": 1.0 + a},
        }
        return spec
    spec["intervention_executable"] = False
    return spec


def run_instrumented_episode(spec: dict, *, max_steps: int = 4096,
                             obs_radius: int = 64) -> dict:
    """跑一个带插桩的 episode。spec 为 scenarios_v2.CaseSpec.to_dict()。

    obs_radius=64 (全图视野): greedy+astar 在局部视野下目标出窗即随机游走,
    会批量截断 (引擎修复记录 b5f2bf2: mk01 1262步->393步)。
    """
    t0 = time.time()
    policy = spec["policy"]
    seed = int(spec["seed"])
    fjsp_path = ROOT / "data" / "fjsp_official" / "brandimarte" / f"{spec['instance']}.json"
    map_file = ROOT / "data" / "mapf" / "medium_maps.yaml"
    num_agv = int(spec["num_agv"])
    assigner = spec.get("assigner", "nearest")
    ptc = spec.get("processing_time_config")

    # --- 策略 -> 触发器映射 (与 run_closed_episode 一致) ---
    trigger, scope = "never", "full"
    job_solver = "online_fjsp"
    if policy == "greedy-reactive":
        job_solver = "greedy"
    elif policy == "cpsat-static":
        pass
    elif policy == "cpsat-full":
        trigger, scope = "event", "full"
    elif policy == "cpsat-partial":
        trigger, scope = "event", "partial"
    else:
        raise ValueError(policy)
    if spec.get("trigger_override"):
        trigger = spec["trigger_override"]

    fjsp_data = load_fjsp_json(fjsp_path)
    grid_map = load_map(map_file, spec.get("map_name"))
    grid_cfg, machine_cfg, job_cfg = build_configs(
        fjsp_data, grid_map, num_agv=num_agv, seed=seed,
        obs_radius=obs_radius, max_episode_steps=max_steps,
        observation_type="default", collision_system="priority",
    )

    from sky_executor.grid_factory.factory.grid_factory_env import GridFactoryEnv
    env = GridFactoryEnv(
        grid_config=grid_cfg, machine_config=machine_cfg, job_config=job_cfg,
        random_target=False, exception_config=spec.get("exception_config"),
        processing_time_config=ptc,
    )
    obs, info = env.reset()

    # greedy 臂必需: 运输时间估计器 (与 run_closed_episode 一致), 缺失则
    # 运输分派失灵, episode 全部截断 (throughput=0)
    transfer_estimator = None
    if job_solver == "greedy":
        try:
            from sky_executor.grid_factory.factory.Utils.transfer_estimator import (
                TransferTimeEstimator,
            )
            transfer_estimator = TransferTimeEstimator(
                env.pogema_env.machines,
                env.pogema_env.grid.get_obstacles(),
                use_feedback=True,
            )
        except Exception:
            transfer_estimator = None

    job_kwargs, route_kwargs = {}, {}
    route_solver = spec.get("route_solver", "astar")
    if job_solver == "online_fjsp":
        job_kwargs = {"service_url": FJSP_URL, "algorithm": "cp_sat",
                      "config": {"time_limit": 10.0, "num_workers": 1, "seed": seed}}
        _wait_for_service("FJSP", FJSP_URL, 120.0)

    from sky_executor.grid_factory.factory.Component.Coordinator.coordinator import (
        Coordinator,
    )
    coordinator = Coordinator(
        job_solver=job_solver, route_solver=route_solver,
        assigner=assigner, job_solver_kwargs=job_kwargs,
        route_solver_kwargs=route_kwargs,
        transfer_time_estimator=transfer_estimator,
    )
    orch = RecoveryOrchestrator(
        coordinator, env, trigger=trigger, scope=scope, verbose=False,
    )

    finished, steps = False, 0
    events_all: list = []
    metrics_series: list = []
    feedback_task_ids: set = set()
    for i in range(max_steps):
        actions = coordinator.decide(obs)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps = i + 1
        if transfer_estimator is not None:
            for agent in obs.get("task_observation", {}).get("agents", []):
                for task in (getattr(agent, "finished_tasks", None) or []):
                    if task.task_id not in feedback_task_ids:
                        transfer_estimator.update_from_task(task)
                        feedback_task_ids.add(task.task_id)
        injector = getattr(env, "exception_injector", None)
        if injector is not None:
            evs = injector.get_step_events() or []
            if evs:
                events_all.extend(evs)
        orch.on_step(obs, steps)
        if (i + 1) % METRIC_STRIDE == 0:
            m = infos.get("metrics", {}) or {}
            metrics_series.append(
                {"step": i + 1, **{k: v for k, v in m.items()
                                   if isinstance(v, (int, float))}})
        if terminations.get("job_done"):
            finished = True
            break
        if all(truncations.values()):
            break

    summary = env.metrics_hub.get_episode_summary()
    if injector is not None:
        hist = injector.get_event_history()
        if hist:
            events_all = hist
    try:
        nervous = coordinator.get_schedule_nervousness()
    except Exception:
        nervous = {}
    plan_revisions = []
    jd = getattr(coordinator.job_solver, "get_plan_revisions", None)
    if callable(jd):
        try:
            plan_revisions = jd() or []
        except Exception:
            plan_revisions = []

    return {
        "case_id": spec["case_id"],
        "config": {
            "instance": spec["instance"], "num_agv": num_agv, "seed": seed,
            "policy": policy, "map_name": spec.get("map_name"),
            "exception_config": spec.get("exception_config"),
            "route_solver": route_solver, "assigner": assigner,
            "trigger": trigger, "scope": scope,
            "processing_time_config": ptc,
        },
        "target_class": spec.get("target_class"),
        "finished": finished,
        "steps": steps,
        "makespan": summary.get("completed_makespan") if finished else max_steps,
        "summary": summary,
        "orchestrator_stats": orch.stats,
        "events_all": events_all,
        "n_events": len(events_all),
        "metrics_series": metrics_series,
        "plan_revisions": plan_revisions,
        "n_plan_revisions": len(plan_revisions),
        "schedule_nervousness": {
            k: v for k, v in (nervous or {}).items()
            if k != "schedule_revision_events"} if isinstance(nervous, dict) else {},
        "intervention_applied": spec.get("intervention_applied", "none"),
        "wall_time_s": round(time.time() - t0, 2),
    }
