"""订单批次: 同实例按批次种子抽作业子集 → 不同订单混合(布局泛化性的批次轴)。"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_batch(instance: str, batch_id: int, n_jobs: int,
               out_dir: Path = None) -> Path:
    """从 data/fjsp_official/brandimarte/{instance}.json 抽前 n_jobs 个作业(确定性)。"""
    src = ROOT / "data/fjsp_official/brandimarte" / f"{instance}.json"
    data = json.loads(src.read_text())
    jobs = data["jobs"]
    if n_jobs >= len(jobs):
        return src  # 全量即原文件
    rng = random.Random(9000 + batch_id)
    subset = rng.sample(range(len(jobs)), n_jobs)
    out = dict(data)
    out["jobs"] = [jobs[i] for i in sorted(subset)]
    out_dir = out_dir or (ROOT / "layout/_batches")
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{instance}_b{batch_id}_j{n_jobs}.json"
    f.write_text(json.dumps(out))
    return f
