"""
sky_research.common.fjsp_io
===========================
经典 FJSP 基准实例 (Brandimarte mk 系列等) -> SkyEngine JSON 格式转换。

SkyEngine JSON 格式 (parse_fjsp_instance):
  {
    "machines": <int>,
    "jobs": [                       # 每个 job
      [                             # 每道工序
        [{"processing": <int>, "machine": <int>}, ...]   # 可选 (机器, 时长)
      ]
    ]
  }

Brandimarte 文本格式:
  第 1 行: <job 数> <机器数>
  之后每行一个 job: <工序数> ( <可选机器数> (<机器id> <时长>)* )* )
"""

from __future__ import annotations

import json
from pathlib import Path


def parse_brandimarte(txt_path: str | Path) -> dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    n_jobs, n_machines = map(int, lines[0].split())
    jobs = []
    for job_id in range(n_jobs):
        nums = list(map(int, lines[1 + job_id].split()))
        num_operations = nums[0]
        idx = 1
        operations = []
        for _ in range(num_operations):
            k = nums[idx]
            idx += 1
            alternatives = []
            for _ in range(k):
                machine_id = nums[idx]
                proc_time = nums[idx + 1]
                idx += 2
                alternatives.append(
                    {"processing": proc_time, "machine": machine_id - 1}
                )
            operations.append(alternatives)
        jobs.append(operations)
    assert len(jobs) == n_jobs, f"{txt_path}: job 数不匹配"
    return {"machines": n_machines, "jobs": jobs}


def convert_family(
    classic_dir: str | Path, out_dir: str | Path, family: str | None = None
) -> list[Path]:
    """把 classic_dir 下 (可选按 family 子目录) 的所有 .txt 转为 out_dir/*.json"""
    classic_dir = Path(classic_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_files = (
        sorted((classic_dir / family).glob("*.txt")) if family else sorted(classic_dir.glob("*.txt"))
    )
    written = []
    for txt in src_files:
        data = parse_brandimarte(txt)
        out = out_dir / f"{(family + '_') if family else ''}{txt.stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f)
        written.append(out)
    return written


def load_instance_index(instances_json: str | Path) -> list[dict]:
    """读取 dataset/fjsp-instances/instances.json 索引 (name/jobs/machines/optimum/path)"""
    with open(instances_json, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--classic-dir", required=True, help="经典实例根目录 (含 family 子目录)")
    ap.add_argument("--out-dir", required=True, help="输出 JSON 目录")
    ap.add_argument("--families", nargs="*", default=["brandimarte"])
    args = ap.parse_args()
    for fam in args.families:
        files = convert_family(args.classic_dir, args.out_dir, fam)
        print(f"[{fam}] converted {len(files)} instances -> {args.out_dir}")
