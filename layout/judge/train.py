"""第一档训练器: GBDT 回归 + 三重 hold-out 排序评测 (M2 判活: Spearman>=0.5)。
用法 (容器内, 数据集生成完成后):
  python layout/judge/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "layout/judge"))

from features import compute_features  # noqa: E402
from layout.batches import make_batch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from scipy.stats import spearmanr, kendalltau  # noqa: E402

M2 = ROOT / "layout/results_m2"
MAP_FILE = ROOT / "data/mapf/random_maps.yaml"


def load_split(name):
    rows = [json.loads(l) for l in (M2 / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    by_layout = {}
    for r in rows:
        if "error" in r:
            continue
        d = by_layout.setdefault(r["layout_id"], {"mk": [], "fin": [],
                                                  "agvs": r.get("agv_positions"),
                                                  "machines": r.get("machine_positions")})
        d["mk"].append(r["makespan"])
        d["fin"].append(1 if r.get("finished") else 0)
    return by_layout


def load_rows(split):
    """逐集行 (布局×批次×起点), 不聚合 —— 特征按该批次流量算。"""
    rows = [json.loads(l) for l in (M2 / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if "error" not in r]


def featurize(rows, manifest):
    layouts = {l["layout_id"]: l
               for lst in manifest["layouts"].values() for l in lst}
    bjobs = manifest.get("batch_jobs", {"mk01": 6, "mk02": 6, "mk03": 9})
    batch_flow, fjsp_cache = {}, {}
    X, y, ids = [], [], []
    for r in rows:
        meta = layouts[r["layout_id"]]
        inst = meta["instance"]
        key = (inst, r["batch_id"])
        if key not in batch_flow:
            bf = make_batch(inst, r["batch_id"], bjobs[inst])
            batch_flow[key] = json.loads(Path(bf).read_text())
        if inst not in fjsp_cache:
            fjsp_cache[inst] = json.loads(
                (ROOT / "data/fjsp_official/brandimarte" / f"{inst}.json").read_text())
        num_agv = 4
        feats = compute_features(MAP_FILE, meta["map"],
                                 r["machine_positions"], r["agv_positions"],
                                 batch_flow[key], num_agv)
        X.append(feats)
        y.append(r["makespan"])
        ids.append(r["layout_id"])
    return X, y, ids


def topk_hit(pred, true, k=10):
    if len(pred) < k:
        return None
    pp = sorted(range(len(pred)), key=lambda i: pred[i])[:k]
    tp = sorted(range(len(true)), key=lambda i: true[i])[:k]
    return len(set(pp) & set(tp))


def main():
    manifest = json.loads((M2 / "manifest.json").read_text())
    X, y, ids = featurize(load_rows("train"), manifest)
    print(f"训练集(逐集级) {len(X)} | makespan 域 [{min(y):.0f}, {max(y):.0f}]")
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                          max_depth=None, random_state=42)
    model.fit([list(f.values()) for f in X], y)
    print("\n== 三重 hold-out 排序评测 (判活线 Spearman>=0.5) ==")
    for split in ("holdout_maps", "holdout_instance", "holdout_batches"):
        try:
            ho_rows = load_rows(split)
        except FileNotFoundError:
            print(f"{split}: 文件不存在, 跳过")
            continue
        Xh, yh, idsh = featurize(ho_rows, manifest)
        pred = model.predict([list(f.values()) for f in Xh])
        sp = spearmanr(pred, yh).statistic
        kt = kendalltau(pred, yh).statistic
        hit = topk_hit(list(pred), yh, k=min(10, max(1, len(yh) // 3)))
        print(f"{split}: n={len(yh)} Spearman={sp:.3f} Kendall={kt:.3f} "
              f"topK命中={hit}/{min(10, len(yh)//3)*1 if hit is not None else '-'} "
              f"| true范围[{min(yh):.0f},{max(yh):.0f}]")
    # 训练内自检 + 特征重要性(置换)
    pred_tr = model.predict([list(f.values()) for f in X])
    print(f"\n训练内 Spearman={spearmanr(pred_tr, y).statistic:.3f} (过拟合参照)")
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, [list(f.values()) for f in X], y,
                                n_repeats=5, random_state=42)
    names = list(X[0].keys())
    top = sorted(zip(names, pi.importances_mean), key=lambda t: -t[1])[:6]
    print("特征重要性 Top6:", [(n, round(v, 3)) for n, v in top])
    (M2 / "judge_v1.json").write_text(json.dumps({
        "model": "HistGradientBoosting", "n_train": len(X),
        "features": names,
        "importance_top": [(n, float(v)) for n, v in top],
    }, ensure_ascii=False, indent=1))
    import pickle
    with open(M2 / "judge_v1.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": names}, f)
    print("-> judge_v1.pkl / judge_v1.json")


if __name__ == "__main__":
    main()
