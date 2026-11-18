"""M3 腿1: 随机搜索基准线 (judge 无关的对照底线)。
固定仿真预算下, 纯随机采样布局能找到多好 —— 预算-质量曲线 + 经典 from-to 参照点。
用法 (容器内): python layout/search_baseline.py [--budget 150]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layout.legality import free_cells, check_layout  # noqa: E402
from layout import sampler  # noqa: E402
from layout.batches import make_batch  # noqa: E402
from layout.evaluator import run_grid  # noqa: E402

MAP_FILE = ROOT / "data/mapf/random_maps.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="000")
    ap.add_argument("--instance", default="mk01")
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--num-agv", type=int, default=4)
    ap.add_argument("--out", default="layout/results_m2/search_baseline.json")
    a = ap.parse_args()

    map_name = f"validation-random-seed-{a.map}"
    free, free_set = free_cells(MAP_FILE, map_name)
    fjsp = json.loads((ROOT / "data/fjsp_official/brandimarte" / f"{a.instance}.json").read_text())
    n_m = fjsp["machines"]
    flow = sampler.expected_flow(fjsp)
    bfile = make_batch(a.instance, 0, 6)

    # 候选: budget 个随机布局, 每个只跑 1 集(b0×o0) —— 搜索中的廉价评估
    rng = random.Random(zlib.crc32(f"search|{a.map}|{a.instance}".encode()))
    eps, lays = [], []
    for i in range(a.budget):
        while True:
            lay = sampler.random_layout(rng, free, free_set, n_m, a.num_agv)
            ok, _ = check_layout(free_set, lay["machine_positions"], lay["agv_positions"])
            if ok:
                break
        machines = [tuple(p) for p in lay["machine_positions"]]
        mset = set(machines)
        rest = [p for p in free if p not in mset]
        agvs = random.Random(700000 + i).sample(rest, a.num_agv)
        lays.append({"i": i, "machines": machines})
        eps.append(((str(bfile), machines, agvs, a.num_agv, 42, 4096,
                     str(MAP_FILE), map_name), {"layout_id": f"rs_{i}", "batch_id": 0, "origin_id": 0}))
    # 参照点: 经典 from-to 启发式 (同协议 1 集)
    lay_ft = sampler.from_to_layout(random.Random(7), free, free_set, n_m, a.num_agv, flow)
    ft_machines = [tuple(p) for p in lay_ft["machine_positions"]]
    ft_rest = [p for p in free if p not in set(ft_machines)]
    ft_agvs = random.Random(700999).sample(ft_rest, a.num_agv)
    eps.append(((str(bfile), ft_machines, ft_agvs, a.num_agv, 42, 4096,
                 str(MAP_FILE), map_name), {"layout_id": "from_to_ref", "batch_id": 0, "origin_id": 0}))

    results = run_grid(eps, timeout=600.0)
    mk = {meta["layout_id"]: row.get("makespan") for row, meta in results}
    ft_mk = mk.pop("from_to_ref")

    # best-so-far 曲线
    curve, best = [], float("inf")
    for i in range(a.budget):
        v = mk[f"rs_{i}"]
        best = min(best, v)
        curve.append(best)
    checkpoints = {str(n): curve[n - 1] for n in (10, 25, 50, 100, a.budget) if n <= a.budget}

    # 精选确认: 单集前5名 → 2批次×2起点 复核
    top5 = sorted(mk, key=lambda k: mk[k])[:5]
    confirm_eps = []
    for k in top5:
        i = int(k.split("_")[1])
        machines = lays[i]["machines"]
        mset = set(machines)
        for b in (0, 1):
            bf = make_batch(a.instance, b, 6)
            for o in range(2):
                rest = [p for p in free if p not in mset]
                agvs = random.Random(700000 + o * 1009 + i).sample(rest, a.num_agv)
                confirm_eps.append(((str(bf), machines, agvs, a.num_agv, 42 + o, 4096,
                                     str(MAP_FILE), map_name),
                                    {"layout_id": k, "batch_id": b, "origin_id": o}))
    cres = run_grid(confirm_eps, timeout=600.0)
    agg = {}
    for row, meta in cres:
        agg.setdefault(meta["layout_id"], []).append(row.get("makespan"))
    confirmed = {k: sum(v) / len(v) for k, v in agg.items()}
    best_confirmed = min(confirmed.values())

    out = {
        "map": map_name, "instance": a.instance, "budget": a.budget,
        "curve_checkpoints": checkpoints,
        "single_episode_best": curve[-1],
        "top5_confirmed": confirmed,
        "best_confirmed_mean": best_confirmed,
        "from_to_reference_1ep": ft_mk,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
