"""
方向4 pilot: Lifelong 吞吐 x 到达节拍 x 求解器
==============================================
矩阵: {mk01} x 迷宫 x agv4 x cadence {0(全量), 20, 50, 100}
      x job_solver {greedy(astar), online_fjsp(eecbs)} x seed42
= 8 episodes (预计 5-10 分钟)

用法: python lifelong/run_pilot.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifelong.gate import run_lifelong_episode  # noqa: E402

CADENCES = [0, 20, 50, 100]
SOLVERS = ["greedy+astar", "cpsat+eecbs"]


def _worker(cadence, solver, out_file):
    if solver == "greedy+astar":
        kw = dict(job_solver="greedy", route_solver_name="astar")
    else:
        kw = dict(job_solver="online_fjsp", route_solver_name="rolling_mapf_http")
    try:
        res = run_lifelong_episode(
            ROOT / "data" / "fjsp_official" / "brandimarte" / "mk01.json",
            ROOT / "data" / "mapf" / "medium_maps.yaml",
            "medium-mazes-seed-0000",
            cadence=cadence, num_agv=4, seed=42, max_steps=4096,
            **kw,
        )
        res["cadence"], res["solver"] = cadence, solver
    except Exception as e:  # noqa: BLE001
        res = {"cadence": cadence, "solver": solver,
               "error": f"{type(e).__name__}: {e}"}
    Path(out_file).write_text(json.dumps(res, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "results" / f"lifelong_pilot_{int(time.time())}.jsonl"
    out.parent.mkdir(exist_ok=True)
    runs = [(c, s) for c in CADENCES for s in SOLVERS]
    if args.limit:
        runs = runs[: args.limit]
    out.write_text(json.dumps({"n_runs": len(runs)}) + "\n")
    print(f"[lifelong] {len(runs)} runs -> {out}")
    for i, (c, s) in enumerate(runs):
        t0 = time.time()
        tmp = ROOT / "results" / f".ll_{t0:.0f}_{c}_{s.replace('+','_')}.json"
        ctx = mp.get_context("fork")
        proc = ctx.Process(target=_worker, args=(c, s, str(tmp)))
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            proc.terminate(); proc.join(5)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            rec = {"cadence": c, "solver": s, "error": f"hard timeout {args.timeout}s"}
        else:
            try:
                rec = json.loads(tmp.read_text())
            except Exception as e:
                rec = {"cadence": c, "solver": s, "error": f"unreadable: {e}"}
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        err = rec.get("error")
        status = f"ERR {err[:60]}" if err else (
            f"thr={rec.get('throughput_per_1k_steps')} "
            f"jobs={rec.get('throughput_jobs')} mk={rec.get('makespan')} "
            f"wall={rec.get('wall_time_s')}s"
        )
        print(f"[{i+1}/{len(runs)}] cadence={c} {s} -> {status}", flush=True)
    print("[lifelong] done ->", out)


if __name__ == "__main__":
    main()
