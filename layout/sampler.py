"""布局采样器: 随机 / 分散(远点贪心) / 聚集 / 微扰。全部确定性(显式 rng)。"""
from __future__ import annotations

import random


def random_layout(rng: random.Random, free: list, free_set: set,
                  n_machines: int, n_agv: int, exclude=()) -> dict:
    """随机采样: 机器先、AGV 后, 互不重叠; exclude=额外禁用格。"""
    pool = [p for p in free if p not in exclude]
    machines = rng.sample(pool, n_machines)
    rest = [p for p in pool if p not in set(machines)]
    agvs = rng.sample(rest, n_agv)
    return {"machine_positions": machines, "agv_positions": agvs}


def spread_layout(rng: random.Random, free: list, free_set: set,
                  n_machines: int, n_agv: int) -> dict:
    """分散: 远点贪心(每步选离已选集最远的格) —— 敏感性冒烟用上界布局。"""
    first = rng.choice(free)
    chosen = [first]
    while len(chosen) < n_machines:
        best, best_d = None, -1
        for p in free:
            if p in chosen:
                continue
            d = min(abs(p[0]-q[0]) + abs(p[1]-q[1]) for q in chosen)
            if d > best_d:
                best, best_d = p, d
        chosen.append(best)
    rest = [p for p in free if p not in set(chosen)]
    agvs = rng.sample(rest, n_agv)
    return {"machine_positions": chosen, "agv_positions": agvs}


def cluster_layout(rng: random.Random, free: list, free_set: set,
                   n_machines: int, n_agv: int) -> dict:
    """聚集: 以随机种子格为中心按距离取最近 n_machines 格 —— 敏感性冒烟用下界布局。"""
    center = rng.choice(free)
    by_dist = sorted(free, key=lambda p: (abs(p[0]-center[0]) + abs(p[1]-center[1]), p))
    machines = by_dist[:n_machines]
    rest = by_dist[n_machines:]
    agvs = rng.sample(rest, n_agv)
    return {"machine_positions": machines, "agv_positions": agvs}


def perturb_layout(rng: random.Random, layout: dict, free_set: set) -> dict:
    """微扰: 随机挑一台机器挪到相邻空格(管线三单机挪动分析的基本算子)。"""
    machines = [tuple(p) for p in layout["machine_positions"]]
    i = rng.randrange(len(machines))
    r, c = machines[i]
    cand = [(r+dr, c+dc) for dr, dc in ((0,1),(0,-1),(1,0),(-1,0))]
    free_cand = [p for p in cand if p in free_set and p not in machines]
    if not free_cand:
        return layout
    machines[i] = rng.choice(free_cand)
    return {"machine_positions": machines, "agv_positions": list(layout["agv_positions"])}
