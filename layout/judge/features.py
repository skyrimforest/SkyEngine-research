"""第一档特征提取器: 13 项布局-工艺特征 (设计书 §3.1)。
距离一律用迷宫真实 BFS 步数; 期望流量按柔性选择均匀摊开。
F3 注: 引擎无显式原料区坐标, 以"到最近 AGV 起点的加权距离"为代理(已声明)。
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from layout.sampler import expected_flow  # noqa: E402
from layout.legality import free_cells  # noqa: E402


def _bfs(grid_free: set, src):
    """单源 BFS: 返回 {cell: dist} 与 parent 表(用于路径重建)。"""
    dist = {src: 0}
    parent = {src: None}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            nxt = (nr, nc)
            if nxt in grid_free and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                parent[nxt] = (r, c)
                q.append(nxt)
    return dist, parent


def _path(parent: dict, src, dst):
    p, cur = [], dst
    while cur is not None and cur != src:
        p.append(cur)
        cur = parent.get(cur)
    return p  # dst..src+1 (不含 src)


def compute_features(map_file, map_name, machine_positions, agv_positions,
                     fjsp_data: dict, num_agv: int) -> dict:
    free, free_set = free_cells(map_file, map_name)
    machines = [tuple(p) for p in machine_positions]
    agvs = [tuple(p) for p in agv_positions]
    n = len(machines)
    flow = expected_flow(fjsp_data)

    bfs = {m: _bfs(free_set, m) for m in machines}
    # --- F1/F2: 流量×距离 ---
    wdist_total, wdist_max = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            if i == j or flow[i][j] == 0:
                continue
            d = bfs[machines[i]][0].get(machines[j])
            if d is None:
                d = 999  # 不连通(不应发生, 合法性已保证)
            w = flow[i][j] * d
            wdist_total += w
            wdist_max = max(wdist_max, w)
    # --- F3: 到最近 AGV 起点的加权距离(原料区代理) ---
    out_flow = [sum(flow[m]) + sum(flow[k][m] for k in range(n)) for m in range(n)]
    depot_d = 0.0
    for i in range(n):
        dmin = min(bfs[machines[i]][0].get(a, 999) for a in agvs)
        depot_d += out_flow[i] * dmin
    # --- F4-F7: 路径交通 (每对流量摊到其最短路径各格) ---
    traffic = {}
    total_traffic = 0.0
    for i in range(n):
        for j in range(n):
            if i == j or flow[i][j] == 0:
                continue
            for cell in _path(bfs[machines[i]][1], machines[i], machines[j]):
                traffic[cell] = traffic.get(cell, 0.0) + flow[i][j]
                total_traffic += flow[i][j]
    cell_loads = sorted(traffic.values(), reverse=True)
    hottest = cell_loads[0] if cell_loads else 0.0
    k5 = max(1, int(len(cell_loads) * 0.05))
    top5_share = sum(cell_loads[:k5]) / total_traffic if total_traffic else 0.0
    herfindahl = sum(v * v for v in cell_loads) / (total_traffic ** 2) if total_traffic else 0.0
    corridors = sum(1 for v in cell_loads if v > 0.01 * total_traffic)
    # --- F8-F10: 空间结构 ---
    n_free = len(free)
    density = n / n_free
    nbrs = [sum(1 for dr, dc in ((0,1),(0,-1),(1,0),(-1,0))
                if (m[0]+dr, m[1]+dc) in free_set) for m in machines]
    # --- F11/F12: 柔性 ---
    flat = [v for row in flow for v in row if v > 0]
    p = [v / sum(flat) for v in flat] if flat else [1.0]
    entropy = -sum(q * math.log(q) for q in p) / math.log(max(2, len(p))) if len(p) > 1 else 0.0
    alts = [len(op) for job in fjsp_data["jobs"] for op in job]
    mean_alts = sum(alts) / len(alts)
    return {
        "F1_wdist_total": wdist_total, "F2_wdist_max": wdist_max,
        "F3_depot_wdist": depot_d, "F4_hottest_cell": hottest,
        "F5_top5_share": top5_share, "F6_traffic_hhi": herfindahl,
        "F7_corridor_cnt": corridors, "F8_free_cells": n_free,
        "F9_density": density, "F10_mean_nbrs": sum(nbrs) / n, "F10b_min_nbrs": min(nbrs),
        "F11_flow_entropy": entropy, "F12_mean_alts": mean_alts,
        "F13_num_agv": num_agv, "F13b_num_machines": n,
    }
