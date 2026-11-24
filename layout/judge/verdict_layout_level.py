import json, sys
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'layout/judge'))
from features import compute_features
from layout.batches import make_batch
from sklearn.ensemble import HistGradientBoostingRegressor

M2 = ROOT/'layout/results_m2'
man = json.loads((M2/'manifest.json').read_text())
man_s = json.loads((M2/'manifest_scale.json').read_text())
layouts = {l['layout_id']: l for lst in man['layouts'].values() for l in lst}
for l in man_s['extra_layouts']: layouts[l['layout_id']] = l
bjobs = {'mk01':6,'mk02':6,'mk03':9}; bflow = {}

def feats_of(r):
    meta = layouts[r['layout_id']]; inst = meta['instance']
    k = (inst, r['batch_id'])
    if k not in bflow:
        bflow[k] = json.loads(Path(make_batch(inst, r['batch_id'], bjobs[inst])).read_text())
    return compute_features(ROOT/'data/mapf/random_maps.yaml', meta['map'],
                            r['machine_positions'], r['agv_positions'], bflow[k], 4)

rows = []
for f in ('train.jsonl','train_scale.jsonl'):
    rows += [json.loads(l) for l in (M2/f).read_text().splitlines() if 'error' not in l]
data = [(r['layout_id'], feats_of(r), r['makespan']) for r in rows]
def Lidx(lid): return int(lid.split('_L')[-1])

# 布局级聚合: pred 与 y 都按布局取均值 (搜索的真实使用粒度)
agg = defaultdict(lambda: defaultdict(dict))
for lid, f, y in data:
    d = agg[layouts[lid]['map']][Lidx(lid) % 2].setdefault(lid, {'f': f, 'ys': []})
    d['ys'].append(y)

print('=== 布局级裁决: 同图未见布局, 布局均值粒度 ===')
for mode, pooled in (('逐图模型', False), ('三图合训', True)):
    tr_rows = [(x['f'], sum(x['ys'])/len(x['ys'])) for mp,grp in agg.items() for x in grp[0].values()]
    if pooled:
        m = HistGradientBoostingRegressor(max_iter=400, random_state=42)
        m.fit([list(f.values()) for f,_ in tr_rows], [y for _,y in tr_rows])
    for mp, grp in sorted(agg.items()):
        if not pooled:
            m = HistGradientBoostingRegressor(max_iter=400, random_state=42)
            tr = [(x['f'], sum(x['ys'])/len(x['ys'])) for x in grp[0].values()]
            m.fit([list(f.values()) for f,_ in tr], [y for _,y in tr])
        ho = grp[1].values()
        preds = [m.predict([list(x['f'].values())])[0] for x in ho]
        ys = [sum(x['ys'])/len(x['ys']) for x in ho]
        sp = spearmanr(preds, ys).statistic
        # top-10 命中
        k = min(10, len(ys)//3)
        pp = sorted(range(len(preds)), key=lambda i: preds[i])[:k]
        tp = sorted(range(len(ys)), key=lambda i: ys[i])[:k]
        hit = len(set(pp)&set(tp))
        print('[%s] %s: 布局n=%d Spearman=%.3f top%d命中=%d/%d' % (mode, mp, len(ys), sp, k, hit, k))
