"""
sky_research.common.engine_adapter
==================================
对 SkyEngine-confirmation-v2-1 (GridFactoryEnv + Coordinator) 的程序化封装。

设计目标:
  - 复刻 run.py 的标准流程 (解析实例 -> 构建环境 -> Coordinator -> episode 循环),
    但去掉 SVG 逐帧绘制与 HTTP 微服务依赖, 便于批量基准实验;
  - 所有随机性由 seed 控制 (地图上 AGV/机器摆放均由 seed 派生), 保证可复现;
  - 返回结构化结果 dict (episode summary + 运行配置), 由上层写入 JSON/CSV。

引用的引擎组件 (均位于 SkyEngine-confirmation-v2-1):
  - GridFactoryEnv: pogema 网格世界 + 机器加工 + AGV 运输的一体化环境
  - Coordinator: JobSolver(FJSP 调度) + Assigner(AGV-任务分配) + RouteSolver(MAPF 路由)
  - TransferTimeEstimator: BFS 距离 + 历史反馈的运输时间估计器 (调度层的运输感知)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

ENGINE_ROOT = Path(__file__).resolve().parents[2] / "SkyEngine-confirmation-v2-1"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from pogema import GridConfig  # noqa: E402

from sky_executor.grid_factory.factory.Utils.structure import (  # noqa: E402
    JobConfig,
    MachineConfig,
)
from sky_executor.grid_factory.factory.grid_factory_env import (  # noqa: E402
    GridFactoryEnv,
)
from sky_executor.grid_factory.factory.Component.Coordinator.coordinator import (  # noqa: E402
    Coordinator,
)
from sky_executor.grid_factory.factory.Utils.machine import (  # noqa: E402
    largest_free_region,
)

try:  # 运输时间估计器为可选依赖 (仅 greedy job solver 使用)
    from sky_executor.grid_factory.factory.Utils.transfer_estimator import (  # noqa: E402
        TransferTimeEstimator,
    )
except Exception:  # pragma: no cover
    TransferTimeEstimator = None


def load_fjsp_json(path: str | Path) -> dict:
    """读取引擎格式 FJSP 实例 JSON: {machines:int, jobs:[[[{processing,machine}]...]]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_map(map_file: str | Path, map_name: Optional[str] = None) -> Any:
    """从 yaml 地图文件读取指定名称的地图 (字符串型 ASCII map)。

    map_file 为多实例 yaml; map_name 为 None 时取第一个 key。
    """
    import yaml

    with open(map_file, "r", encoding="utf-8") as f:
        maps = yaml.safe_load(f)
    if map_name is None:
        map_name = next(iter(maps))
    if map_name not in maps:
        raise KeyError(
            f"map '{map_name}' not in {map_file}; available={list(maps)[:10]}..."
        )
    return maps[map_name]


def build_configs(
    fjsp_data: dict,
    grid_map: Any,
    *,
    num_agv: int = 4,
    seed: int = 42,
    obs_radius: int = 5,
    max_episode_steps: int = 2048,
    observation_type: str = "default",
    collision_system: str = "priority",
):
    """由 FJSP 实例 + 地图 + AGV 数构建 (GridConfig, MachineConfig, JobConfig)。

    与 run.py 的差异: 完全显式传参, 不读环境变量。
    AGV 初始位置与机器位置在最大连通自由区域内按 seed 派生采样。
    """
    jobs = fjsp_data["jobs"]
    machine_num = fjsp_data["machines"]

    custom_jobs = []
    for job in jobs:
        job_ops = []
        for alternatives in job:
            # JSON 中 alternatives 可能是 dict 列表 {"processing","machine"} 或
            # [machine, time] 对; 统一转成 (processing, machine) 再交换为 (machine, time)
            alt_pairs = [
                (alt["processing"], alt["machine"])
                if isinstance(alt, dict)
                else (alt[1], alt[0])
                for alt in alternatives
            ]
            machine_options_with_time = [(mid, pt) for pt, mid in alt_pairs]
            job_ops.append((machine_options_with_time, 0))
        custom_jobs.append(job_ops)

    job_cfg = JobConfig(
        num_jobs=len(jobs),
        total_machines=machine_num,
        seed=seed,
        strategy="custom_time",
        custom_jobs=custom_jobs,
    )

    raw_grid = (
        [[0 if cell == "." else 1 for cell in line] for line in grid_map.split()]
        if isinstance(grid_map, str)
        else grid_map
    )
    connected_cells = sorted(largest_free_region(raw_grid))
    if len(connected_cells) < num_agv:
        raise ValueError(
            f"free cells {len(connected_cells)} < num_agv {num_agv}"
        )
    initial_positions = random.Random(seed).sample(connected_cells, num_agv)
    machine_candidates = [p for p in connected_cells if p not in initial_positions]
    if len(machine_candidates) < machine_num:
        raise ValueError(
            f"machine candidates {len(machine_candidates)} < machines {machine_num}"
        )
    raw_machine_positions = random.Random(seed + 100003).sample(
        machine_candidates, machine_num
    )
    machine_cfg = MachineConfig(
        num_machines=machine_num,
        strategy="custom",
        seed=seed,
        custom_positions=[
            (row + obs_radius, col + obs_radius) for row, col in raw_machine_positions
        ],
    )
    grid_cfg = GridConfig(
        map=grid_map,
        num_agents=num_agv,
        seed=seed,
        max_episode_steps=max_episode_steps,
        obs_radius=obs_radius,
        on_target="restart",
        collision_system=collision_system,
        observation_type=observation_type,
        agents_xy=None,
        targets_xy=None,
    )
    # 与 run.py 一致: GridConfig 校验后再写入确定性起点, 防止被随机生成分支覆盖
    grid_cfg.agents_xy = initial_positions
    grid_cfg.targets_xy = list(initial_positions)
    grid_cfg.num_agents = num_agv
    return grid_cfg, machine_cfg, job_cfg


DEFAULT_ROLLING_MAPF_KWARGS = {
    "time_limit_ms": 1000,
    "flg_star": False,
    "forbid_follow": True,
    "refine_plan_threshold": 0,
    "refine_time_limit_ms": 500,
    "probe_refinement_rate": 0.0,
    "probe_time_limit_ms": 100,
    "suboptimality": 1.2,
    "planning_horizon": 0,
    "execution_window": 0,
    "lg_window": 20,
    "lns_neighbor_size": 8,
    "lns_init_algo": "EECBS",
    "lns_replan_algo": "PP",
    "seed": 42,
}


def _rolling_kwargs_from_env(**overrides) -> dict:
    """与 run.py 中 rolling_mapf_http 的环境变量默认保持一致, 允许显式覆盖。"""
    kw = dict(DEFAULT_ROLLING_MAPF_KWARGS)
    kw["flg_star"] = os.getenv("MAPF_STAR", "0") == "1"
    kw["forbid_follow"] = os.getenv("COLLISION_SYSTEM") != "soft"
    kw["seed"] = int(os.getenv("SEED", "42"))
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return kw


def _wait_for_service(name: str, url: str, max_wait: float = 120.0) -> bool:
    """轮询微服务 /health 直到就绪 (与 run.py 行为一致)。"""
    import requests as _req

    health_url = url.rstrip("/") + "/health"
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            if _req.get(health_url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"[Warmup] WARNING: {name} ({url}) not ready within {max_wait}s")
    return False


def run_episode(
    fjsp_path: str | Path,
    map_file: str | Path,
    map_name: Optional[str] = None,
    *,
    num_agv: int = 4,
    seed: int = 42,
    max_steps: int = 4096,
    obs_radius: int = 5,
    job_solver: str = "greedy",
    route_solver: str = "astar",
    assigner: str = "random",
    transfer_aware: bool = True,
    job_solver_kwargs: Optional[dict] = None,
    route_solver_kwargs: Optional[dict] = None,
    observation_type: Optional[str] = None,
    collision_system: Optional[str] = None,
    metrics_interval: int = 0,
    warmup_timeout: float = 120.0,
) -> dict:
    """运行一个完整 episode, 返回 {config, summary, steps, wall_time, finished}。

    微服务求解器:
      - job_solver="online_fjsp": FJSP 微服务 (CP-SAT 等), 需 FJSP_SERVICE_URL 可达
      - route_solver="rolling_mapf_http": MAPF 微服务 (classical C++ / lacam / pibt),
        需要 MAPF_SERVICE_URL 可达; 自动启用 OBS_TYPE=MAPF + COLLISION_SYSTEM=soft
    """
    t0 = time.time()

    # --- 与 run.py 一致的 OBS/碰撞系统联动规则 ---
    if observation_type is None:
        if route_solver == "rolling_mapf_http":
            observation_type = os.getenv("OBS_TYPE", "MAPF")
        elif route_solver == "http":
            mapf_image = os.getenv("MAPF_IMAGE", "")
            observation_type = "MAPF" if "gpt" in mapf_image.lower() else "default"
        else:
            observation_type = os.getenv("OBS_TYPE", "default")
    if collision_system is None:
        collision_system = (
            os.getenv("COLLISION_SYSTEM", "soft")
            if route_solver == "rolling_mapf_http"
            else os.getenv("COLLISION_SYSTEM", "priority")
        )

    fjsp_data = load_fjsp_json(fjsp_path)
    grid_map = load_map(map_file, map_name)

    grid_cfg, machine_cfg, job_cfg = build_configs(
        fjsp_data,
        grid_map,
        num_agv=num_agv,
        seed=seed,
        obs_radius=obs_radius,
        max_episode_steps=max_steps,
        observation_type=observation_type,
        collision_system=collision_system,
    )
    env = GridFactoryEnv(
        grid_config=grid_cfg,
        machine_config=machine_cfg,
        job_config=job_cfg,
        random_target=False,
    )
    obs, info = env.reset()

    transfer_estimator = None
    if transfer_aware and job_solver == "greedy" and TransferTimeEstimator is not None:
        try:
            obstacles = env.pogema_env.grid.get_obstacles()
            machines = env.pogema_env.machines
            transfer_estimator = TransferTimeEstimator(
                machines, obstacles, use_feedback=True
            )
        except Exception:
            transfer_estimator = None

    if job_solver_kwargs is None and job_solver == "online_fjsp":
        job_solver_kwargs = {
            "service_url": os.getenv("FJSP_SERVICE_URL", "http://fjsp:8002"),
            "algorithm": "cp_sat",
            "config": {
                "time_limit": float(os.getenv("FJSP_TIME_LIMIT", "30")),
                "num_workers": int(os.getenv("FJSP_NUM_WORKERS", "1")),
                "seed": seed,
            },
        }
    if route_solver_kwargs is None and route_solver == "rolling_mapf_http":
        route_solver_kwargs = _rolling_kwargs_from_env(seed=seed)
        route_solver_kwargs["forbid_follow"] = collision_system != "soft"
    # 显式注入 service_url: 构造函数默认值在模块导入时读环境变量, 直接调用时不可靠
    if route_solver in {"http", "rolling_mapf_http"} and route_solver_kwargs is not None:
        route_solver_kwargs.setdefault(
            "service_url", os.getenv("MAPF_SERVICE_URL", "http://mapf:8001")
        )

    if job_solver in {"http", "online_fjsp"}:
        _wait_for_service(
            "FJSP", os.getenv("FJSP_SERVICE_URL", "http://fjsp:8002"), warmup_timeout
        )
    if route_solver in {"http", "rolling_mapf_http"}:
        _wait_for_service(
            "MAPF", os.getenv("MAPF_SERVICE_URL", "http://mapf:8001"), warmup_timeout
        )

    coordinator = Coordinator(
        job_solver=job_solver,
        route_solver=route_solver,
        assigner=assigner,
        job_solver_kwargs=job_solver_kwargs or {},
        route_solver_kwargs=route_solver_kwargs or {},
        transfer_time_estimator=transfer_estimator,
    )

    feedback_task_ids: set = set()
    finished = False
    steps = 0
    metrics_trace: list[dict] = []
    for i in range(max_steps):
        actions = coordinator.decide(obs)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps = i + 1

        if transfer_estimator is not None:
            for agent in obs.get("task_observation", {}).get("agents", []):
                if agent.finished_tasks:
                    for task in agent.finished_tasks:
                        if task.task_id not in feedback_task_ids:
                            transfer_estimator.update_from_task(task)
                            feedback_task_ids.add(task.task_id)

        if metrics_interval and (i + 1) % metrics_interval == 0:
            metrics_trace.append({"step": i + 1, **infos.get("metrics", {})})

        if terminations.get("job_done"):
            finished = True
            break
        if all(truncations.values()):
            break

    summary = env.metrics_hub.get_episode_summary()
    return {
        "config": {
            "fjsp": str(fjsp_path),
            "map_file": str(map_file),
            "map_name": map_name,
            "num_agv": num_agv,
            "seed": seed,
            "job_solver": job_solver,
            "route_solver": route_solver,
            "assigner": assigner,
            "transfer_aware": bool(transfer_estimator is not None),
            "max_steps": max_steps,
        },
        "finished": finished,
        "steps": steps,
        "wall_time_s": round(time.time() - t0, 3),
        "summary": summary,
        "metrics_trace": metrics_trace,
    }
