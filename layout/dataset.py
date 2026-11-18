"""M2 数据集生成器: 多样性布局采样 × 三重 hold-out (1118layout)
================================================================
切分纪律 (设计书 D4, 自第一天生效并冻结):
  训练:   地图 {000,001,002} × 实例 {mk01,mk02} × 批次 {0,1}
  留出A:  未见地图 {003,004} × 训练实例 × 训练批次
  留出B:  未见实例 {mk03} × 训练地图 {000} × 训练批次
  留出C:  未见批次 {10,11} × 训练地图 {000} × 训练实例 {mk01}
布局混合: 50% 随机 + 25% from-to 启发式 + 25% 随机的单机微扰 (好区域覆盖)。
用法 (容器内 /work/SkyEngine-research-llmdiag):
  nohup python layout/dataset.py > layout/results_m2/gen.log 2>&1 &
"""
from __future__ import annotations

import json
import random
import sys
import zlib
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layout.legality import free_cells, check_layout  # noqa: E402
from layout import sampler  # noqa: E402
from layout.batches import make_batch  # noqa: E402
from layout.evaluator import run_grid, set_map, RANDOM_MAP_FILE  # noqa: E402

OUT = ROOT / "layout/results_m2"
MAP_SEED = lambda s: f"validation-random-seed-{s}"  # noqa: E731

TRAIN_MAPS, HOLD_MAPS = ["000", "001", "002"], ["003", "004"]
TRAIN_INST, HOLD_INST = ["mk01", "mk02"], ["mk03"]
TRAIN_BATCHES, HOLD_BATCHES = [0, 1], [10, 11]
LAYOUT_MIX = {"random": 0.5, "from_to": 0.25, "perturb": 0.25}


def make_layouts(rng, free, free_set, fjsp_data, n_agv, n_layouts):
    """按混合比例采样 n_layouts 个合法布局。"""
    n_m = fjsp_data["machines"]
    flow = sampler.expected_flow(fjsp_data)
    lays = []
    for i in range(n_layouts):
        kind = rng.choices(list(LAYOUT_MIX), weights=list(LAYOUT_MIX.values()))[0]
        if kind == "random":
            lay = sampler.random_layout(rng, free, free_set, n_m, n_agv)
        elif kind == "from_to":
            lay = sampler.from_to_layout(rng, free, free_set, n_m, n_agv, flow)
        else:  # perturb: 对一个随机布局做 1-3 次单机微扰
            lay = sampler.random_layout(rng, free, free_set, n_m, n_agv)
            for _ in range(rng.randint(1, 3)):
                lay = sampler.perturb_layout(rng, lay, free_set)
        ok, why = check_layout(free_set, lay["machine_positions"], lay["agv_positions"])
        if not ok:
            raise ValueError(f"采样非法: {why}")
        lay["kind"] = kind
        lays.append(lay)
    return lays


def run_split(split, map_seeds, instances, batch_ids, n_layouts, num_agv=4,
              batch_jobs={"mk01": 6, "mk02": 6, "mk03": 9}):
    set_map("random")
    free_cache, flow_cache = {}, {}
    episodes, layout_index = [], []
    for ms in map_seeds:
        map_name = MAP_SEED(ms)
        free, free_set = free_cells(RANDOM_MAP_FILE, map_name)
        free_cache[ms] = (free, free_set)
        for inst in instances:
            fjsp = json.loads((ROOT / "data/fjsp_official/brandimarte" / f"{inst}.json").read_text())
            bfiles = {b: make_batch(inst, b, batch_jobs[inst]) for b in batch_ids}
            rng = random.Random(zlib.crc32(f"{split}|{ms}|{inst}".encode()))
            lays = make_layouts(rng, free, free_set, fjsp, num_agv, n_layouts)
            for li, lay in enumerate(lays):
                lid = f"{split}_{ms}_{inst}_L{li}"
                layout_index.append({"layout_id": lid, "map": map_name,
                                     "instance": inst, "kind": lay["kind"],
                                     "machine_positions": lay["machine_positions"]})
                machines = [tuple(p) for p in lay["machine_positions"]]
                mset = set(machines)
                for b in batch_ids:
                    for o in range(2):
                        orng = random.Random(700000 + o * 1009 + li * 7 + int(ms))
                        rest = [p for p in free if p not in mset]
                        agvs = orng.sample(rest, num_agv)
                        episodes.append((
                            (str(bfiles[b]), machines, agvs, num_agv, 42 + o, 4096,
                             str(RANDOM_MAP_FILE), map_name),
                            {"layout_id": lid, "batch_id": b, "origin_id": o}))
    out = OUT / f"{split}.jsonl"
    t0 = time.time()
    results = run_grid(episodes, timeout=600.0)
    with out.open("w") as f:
        for row, meta in results:
            rec = {"split": split, **meta, **row}
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"[{split}] {len(episodes)} 集 -> {out.name} ({time.time()-t0:.0f}s)", flush=True)
    return layout_index


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"generated": time.strftime("%F %T"),
                "mix": LAYOUT_MIX, "num_agv": 4, "max_steps": 4096,
                "batch_jobs": {"mk01": 6, "mk02": 6, "mk03": 9}}
    idx = {}
    idx["train"] = run_split("train", TRAIN_MAPS, TRAIN_INST, TRAIN_BATCHES, 40)
    idx["holdout_maps"] = run_split("holdout_maps", HOLD_MAPS, TRAIN_INST, TRAIN_BATCHES, 12)
    idx["holdout_instance"] = run_split("holdout_instance", ["000"], HOLD_INST, TRAIN_BATCHES, 12)
    idx["holdout_batches"] = run_split("holdout_batches", ["000"], ["mk01"], HOLD_BATCHES, 12)
    manifest["layouts"] = idx
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("M2_DATASET_DONE", flush=True)


if __name__ == "__main__":
    main()
