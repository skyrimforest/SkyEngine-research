"""
方向4: Lifelong FJSP+MAPF — 到达门控 (Arrival Gate)
===================================================
把静态 episode 改造成订单持续到达的 lifelong 场景:
  - 工件 j 带到达时刻 a_j (确定性节拍或泊松过程);
  - ArrivalGateJobSolver 在 t < a_j 时把工件 j 从 job_observation 中滤除
    (对任意内层 JobSolver 通用——规则式与 CP-SAT 式均可);
  - 内层为 CP-SAT (replayable) 时, 新工件到达触发一次全域残差修订
    (承诺一致, 不打断在制/在途);
  - 指标: 稳态吞吐 (完工工件数/千步)、总 makespan、AGV 利用率。

论文对应: 论文_4_Lifelong任务流 §方法
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
from sky_executor.grid_factory.factory.Component.JobSolver.template_solver.job_solver import (  # noqa: E402
    JobSolver,
)

try:
    from sky_executor.grid_factory.factory.Utils.transfer_estimator import (  # noqa: E402
        TransferTimeEstimator,
    )
except Exception:  # pragma: no cover
    TransferTimeEstimator = None


def deterministic_arrivals(n_jobs: int, cadence: int, seed: int = 42) -> list:
    """工件 i 的到达时刻 = i * cadence (确定性节拍, pilot 用, 便于归因)。"""
    return [i * cadence for i in range(n_jobs)]


class ArrivalGateJobSolver(JobSolver):
    """到达门控包装器: t < a_j 的工件不进入内层求解器的观测。"""

    def __init__(self, inner: JobSolver, arrivals: list, env=None, route_solver=None):
        super().__init__()
        self.inner = inner
        self.arrivals = arrivals
        self.env = env
        self.route_solver = route_solver
        self._released = 0
        self._now = 0
        self.stats = {"arrival_revisions": 0, "arrival_revision_fails": 0}

    def bind(self, env, route_solver):
        self.env = env
        self.route_solver = route_solver
        return self

    def set_context(self, task_observation: dict) -> None:
        self._now = int(task_observation.get("env_timeline", self._now))
        if hasattr(self.inner, "set_context"):
            # 注意: 透传完整 task_observation (任务池本身是全局真实的)
            self.inner.set_context(task_observation)

    def _released_jobs(self, obs: dict) -> dict:
        jobs = obs.get("jobs", [])
        keep = [j for j in jobs if self._job_id(j) < len(self.arrivals)
                and self._now >= self.arrivals[self._job_id(j)]]
        return {"jobs": keep, "machines": obs.get("machines")}

    @staticmethod
    def _job_id(j) -> int:
        jid = getattr(j, "job_id", None)
        if jid is None and isinstance(j, dict):
            jid = j.get("job_id", 0)
        return int(jid)

    def plan(self, obs: dict) -> dict:
        gated = self._released_jobs(obs)
        n_now = sum(
            1 for a in self.arrivals if self._now >= a
        )
        newly = n_now - self._released
        if newly > 0:
            self._released = n_now
            # CP-SAT 内层: 到达即修订 (贪心内层每步重建, 无需处理)
            if getattr(self.inner, "initialized", False) and hasattr(
                self.inner, "request_plan_revision"
            ):
                try:
                    self.inner.request_plan_revision(
                        gated,
                        recovery_level=4,
                        trigger={"type": "job_arrival",
                                 "n_released": self._released, "step": self._now},
                        activate=True,
                        activation_env=self.env,
                        route_solver=self.route_solver,
                    )
                    self.stats["arrival_revisions"] += 1
                except Exception:  # noqa: BLE001
                    self.stats["arrival_revision_fails"] += 1
        return self.inner.plan(gated)

    # 透传内层的审计接口 (Coordinator 会探测)
    def get_schedule_audit(self):
        fn = getattr(self.inner, "get_schedule_audit", None)
        return fn() if callable(fn) else None


def run_lifelong_episode(
    fjsp_path: str | Path,
    map_file: str | Path,
    map_name: Optional[str] = None,
    *,
    job_solver: str = "greedy",          # greedy | online_fjsp
    cadence: int = 50,                   # 到达节拍(步/件); 0 = 全部立即可用
    num_agv: int = 4,
    seed: int = 42,
    max_steps: int = 4096,
    assigner: str = "nearest",
    route_solver_name: str = "astar",
    route_solver_kwargs: Optional[dict] = None,
    fjsp_time_limit: float = 10.0,
    stop_at_horizon: bool = False,
) -> dict:
    t0 = time.time()
    fjsp_data = load_fjsp_json(fjsp_path)
    grid_map = load_map(map_file, map_name)
    n_jobs = len(fjsp_data["jobs"])
    arrivals = deterministic_arrivals(n_jobs, cadence, seed) if cadence > 0 else [0] * n_jobs

    observation_type = "MAPF" if route_solver_name == "rolling_mapf_http" else "default"
    collision_system = "soft" if route_solver_name == "rolling_mapf_http" else "priority"
    grid_cfg, machine_cfg, job_cfg = build_configs(
        fjsp_data, grid_map, num_agv=num_agv, seed=seed,
        max_episode_steps=max_steps,
        observation_type=observation_type, collision_system=collision_system,
    )
    env = GridFactoryEnv(
        grid_config=grid_cfg, machine_config=machine_cfg, job_config=job_cfg,
        random_target=False,
    )
    obs, info = env.reset()

    transfer_estimator = None
    if job_solver == "greedy" and TransferTimeEstimator is not None:
        try:
            transfer_estimator = TransferTimeEstimator(
                env.pogema_env.machines,
                env.pogema_env.grid.get_obstacles(),
                use_feedback=True,
            )
        except Exception:
            transfer_estimator = None

    job_kwargs = {}
    if job_solver == "online_fjsp":
        job_kwargs = {
            "service_url": "http://fjsp:8002",
            "algorithm": "cp_sat",
            "config": {"time_limit": fjsp_time_limit, "num_workers": 1, "seed": seed},
        }
    if route_solver_kwargs is None and route_solver_name == "rolling_mapf_http":
        route_solver_kwargs = {
            "service_url": "http://mapf:8001",
            "time_limit_ms": 500, "lns_init_algo": "EECBS",
            "planning_horizon": 10, "execution_window": 5,
        }

    coordinator = Coordinator(
        job_solver=job_solver,
        route_solver=route_solver_name,
        assigner=assigner,
        job_solver_kwargs=job_kwargs,
        route_solver_kwargs=route_solver_kwargs,
        transfer_time_estimator=transfer_estimator,
    )
    # 用到达门控包装内层求解器
    gate = ArrivalGateJobSolver(coordinator.job_solver, arrivals).bind(
        env, coordinator.route_solver
    )
    coordinator.job_solver = gate

    finished = False
    steps = 0
    for i in range(max_steps):
        actions = coordinator.decide(obs)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps = i + 1
        if transfer_estimator is not None:
            for agent in obs.get("task_observation", {}).get("agents", []):
                if agent.finished_tasks:
                    for task in agent.finished_tasks:
                        transfer_estimator.update_from_task(task)
        if terminations.get("job_done") and not stop_at_horizon:
            finished = True
            break
        if all(truncations.values()):
            break

    summary = env.metrics_hub.get_episode_summary()
    # lifelong 视角指标
    horizon_throughput = summary.get("throughput_jobs", 0) / max(steps, 1) * 1000
    return {
        "config": {
            "fjsp": str(fjsp_path), "map_file": str(map_file), "map_name": map_name,
            "num_agv": num_agv, "seed": seed, "job_solver": job_solver,
            "cadence": cadence, "n_jobs": n_jobs,
            "route_solver": route_solver_name, "assigner": assigner,
        },
        "finished": finished,
        "steps": steps,
        "makespan": summary.get("completed_makespan"),
        "throughput_jobs": summary.get("throughput_jobs"),
        "throughput_per_1k_steps": round(horizon_throughput, 3),
        "agv_busy": summary.get("agv_busy_utilization"),
        "agv_loaded": summary.get("agv_loaded_utilization"),
        "queue_wait_mean": summary.get("operation_queue_waiting_time_mean"),
        "gate_stats": gate.stats,
        "summary": summary,
        "wall_time_s": round(time.time() - t0, 2),
    }
