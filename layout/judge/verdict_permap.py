import json, sys
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict
ROOT = Path('/work/SkyEngine-research-llmdiag')
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
print('总行数(逐集级):', len(rows))
data = [(r['layout_id'], feats_of(r), r['makespan']) for r in rows]

def Lidx(lid): return int(lid.split('_L')[-1])
by_map = defaultdict(lambda: defaultdict(list))
for lid, f, y in data:
    by_map[layouts[lid]['map']][Lidx(lid) % 2].append((f, y))

print('\n=== 生死判: 同图未见布局 (奇偶分, 每图100布局) ===')
print('-- 模式A: 逐图独立模型 --')
for mp, grp in sorted(by_map.items()):
    tr = grp[0]; ho = grp[1]
    m = HistGradientBoostingRegressor(max_iter=400, random_state=42)
    m.fit([list(x[0].values()) for x in tr], [x[1] for x in tr])
    pred = m.predict([list(x[0].values()) for x in ho])
    print('%s: train=%d holdout=%d Spearman=%.3f' % (mp, len(tr), len(ho), spearmanr(pred, [x[1] for x in ho]).statistic))
print('-- 模式B: 三图合训, 图内留出 --')
tr_all = [x for mp,grp in by_map.items() for x in grp[0]]
m = HistGradientBoostingRegressor(max_iter=400, random_state=42)
m.fit([list(x[0].values()) for x in tr_all], [x[1] for x in tr_all])
for mp, grp in sorted(by_map.items()):
    ho = grp[1]
    pred = m.predict([list(x[0].values()) for x in ho])
    print('%s: holdout=%d Spearman=%.3f' % (mp, len(ho), spearmanr(pred, [x[1] for x in ho]).statistic))
