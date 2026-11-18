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
    """微扰: 随机挑一台机器挪到相邻空格(避开其余机器与AGV起点)。"""
    machines = [tuple(p) for p in layout["machine_positions"]]
    agvs = {tuple(p) for p in layout.get("agv_positions", [])}
    i = rng.randrange(len(machines))
    r, c = machines[i]
    cand = [(r+dr, c+dc) for dr, dc in ((0,1),(0,-1),(1,0),(-1,0))]
    free_cand = [p for p in cand if p in free_set and p not in machines and p not in agvs]
    if not free_cand:
        return layout
    machines[i] = rng.choice(free_cand)
    return {"machine_positions": machines, "agv_positions": list(layout["agv_positions"])}


def expected_flow(fjsp_data: dict) -> list:
    """机器间期望转移强度: 相邻工序对各候选机器对均分 1 单位流量(柔性摊开)。"""
    n = fjsp_data["machines"]
    flow = [[0.0] * n for _ in range(n)]
    for job in fjsp_data["jobs"]:
        for a, b in zip(job, job[1:]):
            ms_a = [alt["machine"] for alt in a]
            ms_b = [alt["machine"] for alt in b]
            w = 1.0 / (len(ms_a) * len(ms_b))
            for i in ms_a:
                for j in ms_b:
                    flow[i][j] += w
                    if i != j:
                        pass
    return flow


def from_to_layout(rng: random.Random, free: list, free_set: set,
                   n_machines: int, n_agv: int, flow: list) -> dict:
    """经典 from-to 启发式布局: 按总流量降序逐台放置, 曼哈顿距离最小化。
    兼作 M2 基线(经典方法在耦合场景灵不灵)与好区域采样。"""
    order = sorted(range(n_machines), key=lambda m: -sum(flow[m]))
    placed = {}
    first = rng.choice(free)
    placed[order[0]] = first
    used = {first}
    for m in order[1:]:
        best, best_cost = None, None
        for cell in free:
            if cell in used:
                continue
            cost = sum(
                (flow[m][p] + flow[p][m]) * (abs(cell[0] - c[0]) + abs(cell[1] - c[1]))
                for p, c in placed.items())
            if best_cost is None or cost < best_cost:
                best, best_cost = cell, cost
        placed[m] = best
        used.add(best)
    machines = [placed[m] for m in range(n_machines)]
    rest = [p for p in free if p not in used]
    agvs = rng.sample(rest, n_agv)
    return {"machine_positions": machines, "agv_positions": agvs}
