"""M3 多种子复跑: 3 种子 × 两臂, 把 3.67x 变成均值±范围。
用法 (容器内): nohup python layout/search_pilot_seeds.py > layout/results_m2/seeds.log 2>&1 &
"""
from __future__ import annotations

import json
import random
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "layout/judge"))

from layout.legality import free_cells, check_layout  # noqa: E402
from layout import sampler  # noqa: E402
from layout.batches import make_batch  # noqa: E402
from layout.evaluator import run_grid  # noqa: E402
from features import compute_features  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

MAP_FILE = ROOT / "data/mapf/random_maps.yaml"
M2 = ROOT / "layout/results_m2"
MAP_NAME = "validation-random-seed-000"
INST = "mk01"
K = 4


def eval_layouts(cands, batches, origins, tag, quiet=True):
    free, free_set = free_cells(MAP_FILE, MAP_NAME)
    eps = []
    for i, machines in cands.items():
        mset = set(machines)
        rest = [p for p in free if p not in mset]
        for b in batches:
            bf = make_batch(INST, b, 6)
            for o in range(origins):
                agvs = random.Random(700000 + o * 1009 + i).sample(rest, K)
                eps.append(((str(bf), [tuple(p) for p in machines], agvs, K, 42 + o, 4096,
                             str(MAP_FILE), MAP_NAME),
                            {"layout_id": str(i), "batch_id": b, "origin_id": o}))
    res = run_grid(eps, timeout=600.0, quiet=quiet)
    agg = {}
    for row, meta in res:
        agg.setdefault(int(meta["layout_id"]), []).append(row.get("makespan", 4096))
    return {i: sum(v) / len(v) for i, v in agg.items()}


def train_judge():
    man = json.loads((M2 / 'manifest.json').read_text())
    man_s = json.loads((M2 / 'manifest_scale.json').read_text())
    layouts = {l['layout_id']: l for lst in man['layouts'].values() for l in lst}
    for l in man_s['extra_layouts']:
        layouts[l['layout_id']] = l
    rows = []
    for f in ('train.jsonl', 'train_scale.jsonl'):
        rows += [json.loads(l) for l in (M2 / f).read_text().splitlines() if 'error' not in l]
    bf = json.loads(Path(make_batch(INST, 0, 6)).read_text())
    agg = {}
    for r in rows:
        meta = layouts[r['layout_id']]
        if meta['map'] != MAP_NAME or meta['instance'] != INST:
            continue
        if int(r['layout_id'].split('_L')[-1]) % 2 != 0:
            continue
        agg.setdefault(r['layout_id'], {'machines': r['machine_positions'],
                                        'agvs': r['agv_positions'], 'ys': []})['ys'].append(r['makespan'])
    X, y = [], []
    for lid, d in agg.items():
        f = compute_features(MAP_FILE, MAP_NAME, d['machines'], d['agvs'], bf, K)
        X.append(list(f.values()))
        y.append(sum(d['ys']) / len(d['ys']))
    m = HistGradientBoostingRegressor(max_iter=400, random_state=42)
    m.fit(X, y)
    return m


def run_arm(seed_id: str, seed_int: int, judge):
    free, free_set = free_cells(MAP_FILE, MAP_NAME)
    fjsp = json.loads((ROOT / "data/fjsp_official/brandimarte" / f"{INST}.json").read_text())
    n_m = fjsp["machines"]
    rng = random.Random(zlib.crc32(f"pilot|{seed_id}".encode()))

    def rand_machines():
        while True:
            lay = sampler.random_layout(rng, free, free_set, n_m, K)
            ok, _ = check_layout(free_set, lay["machine_positions"], lay["agv_positions"])
            if ok:
                return [tuple(p) for p in lay["machine_positions"]]

    cands_a = {i: rand_machines() for i in range(100)}
    singles = eval_layouts(cands_a, (0,), 1, "A")
    top5 = sorted(singles, key=singles.get)[:5]
    confirm = eval_layouts({i: cands_a[i] for i in top5}, (0, 1), 2, "A")
    best_a = min(confirm.values())

    bf = json.loads(Path(make_batch(INST, 0, 6)).read_text())
    used = 0
    pool = {1000 + i: rand_machines() for i in range(400)}

    def score_pool(p):
        Xs = []
        for machines in p.values():
            agvs = random.Random(7).sample([q for q in free if q not in set(machines)], K)
            Xs.append(list(compute_features(MAP_FILE, MAP_NAME, machines, agvs, bf, K).values()))
        return dict(zip(p.keys(), judge.predict(Xs)))

    pred = score_pool(pool)
    top5b = sorted(pred, key=pred.get)[:5]
    ev = eval_layouts({i: pool[i] for i in top5b}, (0, 1), 2, "B")
    used += 20
    best_key = min(ev, key=ev.get)
    best_b = ev[best_key]
    for rnd in range(3):
        base = pool[best_key]
        neigh = {}
        for j in range(40):
            lay = {"machine_positions": [tuple(p) for p in base],
                   "agv_positions": random.Random(7).sample(
                       [q for q in free if q not in set(base)], K)}
            lay = sampler.perturb_layout(random.Random(9000 + seed_int * 10000 + rnd * 100 + j),
                                         lay, free_set)
            neigh[2000 + rnd * 100 + j] = [tuple(p) for p in lay["machine_positions"]]
        pred_n = score_pool(neigh)
        top3 = sorted(pred_n, key=pred_n.get)[:3]
        ev_n = eval_layouts({i: neigh[i] for i in top3}, (0, 1), 2, "B")
        used += 12
        mk = min(ev_n, key=ev_n.get)
        if ev_n[mk] < best_b:
            best_b = ev_n[mk]
            best_key = mk
            pool[mk] = neigh[mk]
    return best_a, best_b, used


def main():
    judge = train_judge()
    results = []
    for seed_int, sid in enumerate(("s1", "s2", "s3"), 1):
        ba, bb, used = run_arm(sid, seed_int, judge)
        results.append({"seed": sid, "armA_120eps": ba, "armB_eps": used, "armB_best": bb,
                        "quality_ratio": round(ba / bb, 2)})
        print(f"[{sid}] A(120集)={ba:.0f}  B({used}集)={bb:.0f}  质量比={ba/bb:.2f}x", flush=True)
    ratios = [r["quality_ratio"] for r in results]
    bests_a = [r["armA_120eps"] for r in results]
    bests_b = [r["armB_best"] for r in results]
    summary = {"runs": results,
               "ratio_mean": sum(ratios) / len(ratios),
               "ratio_min": min(ratios), "ratio_max": max(ratios),
               "armA_mean": sum(bests_a) / len(bests_a),
               "armB_mean": sum(bests_b) / len(bests_b)}
    print("\n== 多种子汇总 ==")
    print(f"质量比: 均值 {summary['ratio_mean']:.2f}x (范围 {min(ratios):.2f}-{max(ratios):.2f}x)")
    print(f"均值布局: A {summary['armA_mean']:.0f} vs B {summary['armB_mean']:.0f}")
    (M2 / "search_pilot_seeds.json").write_text(json.dumps(summary, indent=1))
    print("SEEDS_DONE")


if __name__ == "__main__":
    main()
