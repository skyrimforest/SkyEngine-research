"""
E6: 到达场景扩容 (论文 R5 的正式支撑实验)
================================================
矩阵: {mk01, mk02, mk05} x 迷宫 x agv4 x cadence {10,20,50,100}
      x 承诺模式 {legacy, soft} x seeds {42,43,44}   = 72 ep
对照: 批量上限 (cadence=0, soft) 每实例 1 集 x 3 种子 = 9 ep
输出: results/icaps_E6.jsonl

用法: python lifelong/run_full.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSTANCES = ["mk01", "mk02", "mk05"]
CADENCES = [10, 20, 50, 100]
MODES = ["legacy", "soft"]
SEEDS = [42, 43, 44]
EPISODE_TIMEOUT = 900.0


def build_runs():
    runs = []
    for inst in INSTANCES:
        for cad in CADENCES:
            for mode in MODES:
                for seed in SEEDS:
                    runs.append(dict(instance=inst, cadence=cad,
                                     mode=mode, seed=seed))
        for seed in SEEDS:  # 批量上限锚点
            runs.append(dict(instance=inst, cadence=0, mode="soft", seed=seed))
    return runs


def _worker(run, out_file):
    from lifelong.gate import run_lifelong_episode

    os.environ["FJSP_SOFT_COMMIT"] = "1" if run["mode"] == "soft" else "0"
    rec = {k: run[k] for k in ("instance", "cadence", "mode", "seed")}
    try:
        res = run_lifelong_episode(
            ROOT / "data" / "fjsp_official" / "brandimarte" / f"{run['instance']}.json",
            ROOT / "data" / "mapf" / "medium_maps.yaml",
            "medium-mazes-seed-0000",
            job_solver="online_fjsp", route_solver_name="rolling_mapf_http",
            cadence=run["cadence"], num_agv=4, seed=run["seed"],
            max_steps=6000,
        )
        rec.update(
            finished=res.get("finished"), steps=res.get("steps"),
            jobs=res.get("throughput_jobs"),
            thr_per_1k=res.get("throughput_per_1k_steps"),
            makespan=res.get("makespan"),
            revisions=res.get("gate_stats", {}).get("arrival_revisions", 0),
            revision_fails=res.get("gate_stats", {}).get("arrival_revision_fails", 0),
            starved=res.get("gate_stats", {}).get("arrival_starved", 0),
            wall_s=res.get("wall_time_s"),
        )
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"{type(e).__name__}: {str(e)[:120]}")
    Path(out_file).write_text(json.dumps(rec, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = ROOT / "results" / "icaps_E6.jsonl"
    runs = build_runs()
    if args.resume and out.exists():
        done = set()
        for line in out.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add("|".join(str(d.get(k)) for k in
                                  ("instance", "cadence", "mode", "seed")))
            except Exception:
                pass
        runs = [r for r in runs
                if "|".join(str(r[k]) for k in ("instance", "cadence", "mode", "seed")) not in done]
    if not (args.resume and out.exists()):
        out.write_text("")
    print(f"[E6] {len(runs)} runs -> {out}", flush=True)
    ctx = mp.get_context("fork")
    t0 = time.time()
    for i, run in enumerate(runs):
        ts = time.time()
        tmp = ROOT / "results" / f".e6_{os.getpid()}_{ts:.0f}.json"
        proc = ctx.Process(target=_worker, args=(run, str(tmp)))
        proc.start()
        proc.join(EPISODE_TIMEOUT)
        if proc.is_alive():
            proc.terminate(); proc.join(5)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            rec = {**run, "error": f"hard timeout {EPISODE_TIMEOUT}s"}
        else:
            try:
                rec = json.loads(tmp.read_text())
            except Exception as e:
                rec = {**run, "error": f"unreadable: {e}"}
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        err = rec.get("error")
        status = f"ERR {err[:40]}" if err else (
            f"jobs={rec.get('jobs')} thr={rec.get('thr_per_1k')} "
            f"rev={rec.get('revisions')}")
        print(f"[E6 {i+1}/{len(runs)}] {run['instance']}|cad{run['cadence']}"
              f"|{run['mode']}|s{run['seed']} -> {status} ({time.time()-ts:.0f}s)",
              flush=True)
    print(f"[E6] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
