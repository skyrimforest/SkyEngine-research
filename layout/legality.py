"""布局合法性检查 (管线一前置)。
坐标体系: raw 网格 (row, col), 与 build_configs 的显式入参一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.engine_adapter import load_map, largest_free_region  # noqa: E402


def free_cells(map_file: str | Path, map_name: str | None = None):
    """最大连通自由区坐标列表(排序, 确定性) + 集合。"""
    grid_map = load_map(map_file, map_name)
    raw = ([[0 if c == "." else 1 for c in line] for line in grid_map.split()]
           if isinstance(grid_map, str) else grid_map)
    cells = sorted(largest_free_region(raw))
    return cells, set(cells)


def check_layout(free_set: set, machine_positions, agv_positions=None) -> tuple[bool, str]:
    """返回 (合法?, 原因)。规则: 在自由区内/不重复/机器不压 AGV 起点。"""
    machines = [tuple(p) for p in machine_positions]
    if len(set(machines)) != len(machines):
        return False, "机器坐标重复"
    bad = [p for p in machines if p not in free_set]
    if bad:
        return False, f"机器出最大连通自由区: {bad[:3]}"
    if agv_positions is not None:
        agvs = [tuple(p) for p in agv_positions]
        if len(set(agvs)) != len(agvs):
            return False, "AGV 起点重复"
        bad = [p for p in agvs if p not in free_set]
        if bad:
            return False, f"AGV 起点出自由区: {bad[:3]}"
        clash = set(agvs) & set(machines)
        if clash:
            return False, f"AGV 起点压机器: {sorted(clash)[:3]}"
    return True, "ok"
