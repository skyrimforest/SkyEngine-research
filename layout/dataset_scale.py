"""M3 腿2: 单图加量 —— 每图每实例新增 60 布局(L40-L99), 第一档最后一次机会。
用法 (容器内): nohup python layout/dataset_scale.py > layout/results_m2/scale.log 2>&1 &
"""
from __future__ import annotations

import json
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layout.legality import free_cells, check_layout  # noqa: E402
from layout import sampler  # noqa: E402
from layout.batches import make_batch  # noqa: E402
from layout.evaluator import run_grid  # noqa: E402
from layout.dataset import make_layouts, MAP_SEED, TRAIN_MAPS, TRAIN_INST  # noqa: E402

M2 = ROOT / "layout/results_m2"
EXTRA = 60  # L40..L99


def main():
    t0 = time.time()
    MAP_FILE = ROOT / "data/mapf/random_maps.yaml"
    all_eps, index = [], []
    for ms in TRAIN_MAPS:
        map_name = MAP_SEED(ms)
        free, free_set = free_cells(MAP_FILE, map_name)
        for inst in TRAIN_INST:
            fjsp = json.loads((ROOT / "data/fjsp_official/brandimarte" / f"{inst}.json").read_text())
            bfiles = {b: make_batch(inst, b, 6) for b in (0, 1)}
            rng = random.Random(zlib.crc32(f"scale|{ms}|{inst}".encode()))
            lays = make_layouts(rng, free, free_set, fjsp, 4, EXTRA)
            for li, lay in enumerate(lays):
                lid = f"train_{ms}_{inst}_L{40 + li}"
                index.append({"layout_id": lid, "map": map_name, "instance": inst,
                              "kind": lay["kind"], "machine_positions": lay["machine_positions"]})
                machines = [tuple(p) for p in lay["machine_positions"]]
                mset = set(machines)
                for b in (0, 1):
                    for o in range(2):
                        rest = [p for p in free if p not in mset]
                        agvs = random.Random(700000 + o * 1009 + (40 + li) * 7 + int(ms)).sample(rest, 4)
                        all_eps.append(((str(bfiles[b]), machines, agvs, 4, 42 + o, 4096,
                                         str(MAP_FILE), map_name),
                                        {"layout_id": lid, "batch_id": b, "origin_id": o}))
    out = M2 / "train_scale.jsonl"
    results = run_grid(all_eps, timeout=600.0)
    with out.open("w") as f:
        for row, meta in results:
            f.write(json.dumps({"split": "train_scale", **meta, **row}, default=str) + "\n")
    (M2 / "manifest_scale.json").write_text(json.dumps(
        {"generated": time.strftime("%F %T"), "extra_layouts": index}, ensure_ascii=False, indent=1))
    print(f"SCALE_DONE {len(all_eps)} 集 ({time.time()-t0:.0f}s) -> train_scale.jsonl", flush=True)


if __name__ == "__main__":
    main()
