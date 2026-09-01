"""
方向2: FJSP+MAPF 统一基准套件定义
=================================
因子化设计: FJSP 实例族 × 地图族 × AGV 密度 × 求解器组合 × 种子。

实例族: Brandimarte mk 系列 (含文献最优 makespan, 可计算运输开销比)
地图族: medium-mazes (走廊型, 37% 障碍) / validation-random (开阔型, 15% 障碍)
求解器组合: JobSolver × RouteSolver × Assigner 的三层笛卡尔积的子集
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- 求解器组合 (name -> 配置) --------------------------------------------
# job: greedy=内置优先级贪婪(运输感知); cp_sat=微服务精确求解
# route: astar=内置合作A*; eecbs=微服务 rolling EECBS
# assign: random / nearest / coupling_hungarian / least_congestion ...

SOLVER_COMBOS = {
    # 规则基线 (无微服务, 快)
    "greedy+astar+random": dict(
        job_solver="greedy", route_solver="astar", assigner="random"
    ),
    "greedy+astar+nearest": dict(
        job_solver="greedy", route_solver="astar", assigner="nearest"
    ),
    "greedy+astar+couplingH": dict(
        job_solver="greedy", route_solver="astar", assigner="coupling_hungarian"
    ),
    # 精确调度 + 搜索路由 (微服务)
    # 注: cpsat+eecbs+couplingH 在迷宫+agv>=6 出现重规划风暴 (EECBS 满载不收敛),
    # pilot 中剔除, 作为"分配器x滚动路由交互失稳"的发现写入论文, 留待方向3研究。
    "cpsat+eecbs+nearest": dict(
        job_solver="online_fjsp",
        route_solver="rolling_mapf_http",
        assigner="nearest",
        job_solver_kwargs={
            "service_url": "http://fjsp:8002",
            "algorithm": "cp_sat",
            "config": {"time_limit": 10.0, "num_workers": 1},
        },
        route_solver_kwargs={
            "service_url": "http://mapf:8001",
            "time_limit_ms": 500,
            "lns_init_algo": "EECBS",
            # RHCR 风格有限视野: 避免长视野计划随执行偏差失效引发重规划风暴
            "planning_horizon": 10,
            "execution_window": 5,
        },
    ),
    "cpsat+eecbs+couplingH": dict(
        job_solver="online_fjsp",
        route_solver="rolling_mapf_http",
        assigner="coupling_hungarian",
        job_solver_kwargs={
            "service_url": "http://fjsp:8002",
            "algorithm": "cp_sat",
            "config": {"time_limit": 10.0, "num_workers": 1},
        },
        route_solver_kwargs={
            "service_url": "http://mapf:8001",
            "time_limit_ms": 500,
            "lns_init_algo": "EECBS",
            "planning_horizon": 10,
            "execution_window": 5,
        },
    ),
}

MAP_FAMILIES = {
    "maze": ("data/mapf/medium_maps.yaml", "medium-mazes-seed-0000"),
    "random": ("data/mapf/random_maps.yaml", "validation-random-seed-000"),
}

# pilot 用小实例; 大规模实验再放开 (须 >2h 的先评估)
PILOT_INSTANCES = ["mk01", "mk02", "mk05"]
FULL_INSTANCES = ["mk01", "mk02", "mk04", "mk05", "mk07", "mk09", "mk10"]


@dataclass
class SuiteConfig:
    instances: list = field(default_factory=lambda: list(PILOT_INSTANCES))
    map_families: list = field(default_factory=lambda: ["maze", "random"])
    num_agvs: list = field(default_factory=lambda: [2, 4, 6])
    solver_combos: list = field(default_factory=lambda: list(SOLVER_COMBOS))
    seeds: list = field(default_factory=lambda: [42])
    max_steps: int = 4096

    def iter_runs(self):
        for inst in self.instances:
            for mfam in self.map_families:
                for nagv in self.num_agvs:
                    for combo in self.solver_combos:
                        for seed in self.seeds:
                            yield dict(
                                instance=inst,
                                map_family=mfam,
                                num_agv=nagv,
                                combo=combo,
                                seed=seed,
                                max_steps=self.max_steps,
                            )


def instance_path(name: str) -> Path:
    return ROOT / "data" / "fjsp_official" / "brandimarte" / f"{name}.json"


def instance_optimum(name: str) -> int | None:
    """从官方索引读取纯 FJSP (无运输) 最优 makespan, 用于运输开销比。"""
    idx = ROOT / "data" / "fjsp_official" / "instances.json"
    import json

    for e in json.loads(idx.read_text()):
        if e["name"] == name:
            return e.get("optimum")
    return None


def instance_stats(name: str) -> dict:
    import json

    d = json.loads(instance_path(name).read_text())
    n_ops = sum(len(j) for j in d["jobs"])
    flex = sum(len(op) for j in d["jobs"] for op in j) / max(n_ops, 1)
    return {"jobs": len(d["jobs"]), "machines": d["machines"], "ops": n_ops,
            "avg_flexibility": round(flex, 2)}
