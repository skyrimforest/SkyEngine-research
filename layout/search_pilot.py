"""M3 经济性试点: 等预算下 judge引导两段式搜索 vs 纯随机搜索。
问题: 0.4 档的弱 judge 能否已产生真实仿真节省? (架构问题, 与 judge 强弱正交)
臂A 随机: 100候选×1集 + top5×4集复核 = 120 集
臂B 两段式: 400候选judge粗筛(0集) → top5×4集 → 3轮微扰爬山(40扰/轮→judge→top3×4集) ≈ 56 集
用法 (容器内): python layout/search_pilot.py
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


def eval_layouts(cands, batches=(0, 1), origins=2):
    """候选布局逐个多批复核, 返回 {idx: mean_makespan}。"""
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
    res = run_grid(eps, timeout=600.0)
    agg = {}
    for row, meta in res:
        agg.setdefault(int(meta["layout_id"]), []).append(row.get("makespan", 4096))
    return {i: sum(v) / len(v) for i, v in agg.items()}


def train_judge():
    """逐图布局级 judge: map000 偶数布局训练。"""
    man = json.loads((M2 / 'manifest.json').read_text())
    man_s = json.loads((M2 / 'manifest_scale.json').read_text())
    layouts = {l['layout_id']: l for lst in man['layouts'].values() for l in lst}
    for l in man_s['extra_layouts']:
        layouts[l['layout_id']] = l
    rows = []
    for f in ('train.jsonl', 'train_scale.jsonl'):
        rows += [json.loads(l) for l in (M2 / f).read_text().splitlines() if 'error' not in l]
    bf = json.loads(Path(make_batch(INST, 0, 6)).read_text())
    X, y = [], []
    agg = {}
    for r in rows:
        meta = layouts[r['layout_id']]
        if meta['map'] != MAP_NAME or meta['instance'] != INST:
            continue
        if int(r['layout_id'].split('_L')[-1]) % 2 != 0:
            continue
        agg.setdefault(r['layout_id'], {'machines': r['machine_positions'],
                                        'agvs': r['agv_positions'], 'ys': []})['ys'].append(r['makespan'])
    for lid, d in agg.items():
        f = compute_features(MAP_FILE, MAP_NAME, d['machines'], d['agvs'], bf, K)
        X.append(list(f.values()))
        y.append(sum(d['ys']) / len(d['ys']))
    m = HistGradientBoostingRegressor(max_iter=400, random_state=42)
    m.fit(X, y)
    return m


def main():
    free, free_set = free_cells(MAP_FILE, MAP_NAME)
    fjsp = json.loads((ROOT / "data/fjsp_official/brandimarte" / f"{INST}.json").read_text())
    n_m = fjsp["machines"]
    rng = random.Random(zlib.crc32(b"pilot|000|mk01"))

    def rand_machines():
        while True:
            lay = sampler.random_layout(rng, free, free_set, n_m, K)
            ok, _ = check_layout(free_set, lay["machine_positions"], lay["agv_positions"])
            if ok:
                return [tuple(p) for p in lay["machine_positions"]]

    # ===== 臂A: 纯随机 (120 集) =====
    cands_a = {i: rand_machines() for i in range(100)}
    singles = eval_layouts(cands_a, batches=(0,), origins=1)          # 100 集
    top5 = sorted(singles, key=singles.get)[:5]
    confirm = eval_layouts({i: cands_a[i] for i in top5})             # 20 集
    best_a = min(confirm.values())
    print(f"臂A 随机: 120集, 单集最优{min(singles.values()):.0f}, 复核后最优 {best_a:.0f}")

    # ===== 臂B: judge 两段式 =====
    judge = train_judge()
    bf = json.loads(Path(make_batch(INST, 0, 6)).read_text())
    used_eps = 0
    pool = {1000 + i: rand_machines() for i in range(400)}           # 400 候选, 0 集

    def score_pool(pool):
        Xs = []
        for machines in pool.values():
            agvs = random.Random(7).sample([p for p in free if p not in set(machines)], K)
            Xs.append(list(compute_features(MAP_FILE, MAP_NAME, machines, agvs, bf, K).values()))
        pred = judge.predict(Xs)
        return {i: p for i, p in zip(pool.keys(), pred)}

    pred = score_pool(pool)
    top5b = sorted(pred, key=pred.get)[:5]
    ev = eval_layouts({i: pool[i] for i in top5b})                    # 20 集
    used_eps += 20
    best_layout = min(ev, key=ev.get)
    best_b = ev[best_layout]
    print(f"臂B 粗筛段: 20集, 复核最优 {best_b:.0f} (真值排名 {sorted(ev.values()).index(best_b)+1}/5)")

    for rnd in range(3):                                              # 微扰爬山 3 轮
        neigh = {}
        base = pool[best_layout]
        for j in range(40):
            lay = {"machine_positions": [tuple(p) for p in base],
                   "agv_positions": random.Random(7).sample(
                       [p for p in free if p not in set(base)], K)}
            lay = sampler.perturb_layout(random.Random(9000 + rnd * 100 + j), lay, free_set)
            neigh[2000 + rnd * 100 + j] = [tuple(p) for p in lay["machine_positions"]]
        pred_n = score_pool(neigh)
        top3 = sorted(pred_n, key=pred_n.get)[:3]
        ev_n = eval_layouts({i: neigh[i] for i in top3})              # 12 集
        used_eps += 12
        m_best = min(ev_n, key=ev_n.get)
        print(f"臂B 爬山轮{rnd+1}: 12集, 本轮最优 {ev_n[m_best]:.0f}")
        if ev_n[m_best] < best_b:
            best_b = ev_n[m_best]
            best_layout = m_best  # 存键(需在 pool/neigh 键空间可寻址)
            pool[m_best] = neigh[m_best]  # 登记进池供下轮微扰
    print(f"\n== 结论 == 等效预算下: 随机(120集)最优 {best_a:.0f} vs 两段式({used_eps}集)最优 {best_b:.0f}")
    print(f"两段式每集收益: {best_a / best_b:.2f}x; 集数节省 {120 - used_eps} 集 ({(120-used_eps)/120:.0%})")
    (M2 / "search_pilot.json").write_text(json.dumps(
        {"armA_best": best_a, "armA_eps": 120, "armB_best": best_b, "armB_eps": used_eps}, indent=1))


if __name__ == "__main__":
    main()
