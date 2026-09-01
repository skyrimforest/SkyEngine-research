"""
方向5 pilot: 拥塞诱导的准死锁观测研究
=====================================
研究问题: 有限投放位(机器单元) + 栅格走廊下, "带任务静止"(tasked_stationary)
与运输阻塞延迟如何随 (AGV 密度, 地图族, 路由求解器) 增长——这是死锁风险的
可观测前兆指标。形式化部分见论文 (MAPF-with-holding + 有限缓冲规约)。

矩阵: {mk01} x {maze, random} x K {4,6,8} x route {astar, eecbs} x seed42
= 12 episodes

用法: python deadlock/run_pilot.py
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

from common.engine_adapter import run_episode  # noqa: E402

EECBS_KWARGS = {
    "service_url": "http://mapf:8001", "time_limit_ms": 500,
    "lns_init_algo": "EECBS", "planning_horizon": 10, "execution_window": 5,
}

RUNS = [
    # (map_family, file, name, K, route)
    ("maze", "medium_maps.yaml", "medium-mazes-seed-0000", 4, "astar"),
    ("maze", "medium_maps.yaml", "medium-mazes-seed-0000", 6, "astar"),
    ("maze", "medium_maps.yaml", "medium-mazes-seed-0000", 8, "astar"),
    ("maze", "medium_maps.yaml", "medium-mazes-seed-0000", 4, "eecbs"),
    ("maze", "medium_maps.yaml", "medium-mazes-seed-0000", 6, "eecbs"),
    ("maze", "medium_maps.yaml", "medium-mazes-seed-0000", 8, "eecbs"),
    ("random", "random_maps.yaml", "validation-random-seed-000", 4, "astar"),
    ("random", "random_maps.yaml", "validation-random-seed-000", 6, "astar"),
    ("random", "random_maps.yaml", "validation-random-seed-000", 8, "astar"),
    ("random", "random_maps.yaml", "validation-random-seed-000", 4, "eecbs"),
    ("random", "random_maps.yaml", "validation-random-seed-000", 6, "eecbs"),
    ("random", "random_maps.yaml", "validation-random-seed-000", 8, "eecbs"),
]


def _worker(map_file, map_name, k, route, out_file):
    try:
        res = run_episode(
            fjsp_path=ROOT / "data" / "fjsp_official" / "brandimarte" / "mk01.json",
            map_file=ROOT / "data" / "mapf" / map_file, map_name=map_name,
            num_agv=k, seed=42, max_steps=4096,
            job_solver="greedy",
            route_solver="astar" if route == "astar" else "rolling_mapf_http",
            route_solver_kwargs=None if route == "astar" else dict(EECBS_KWARGS),
            assigner="nearest", transfer_aware=(route == "astar"),
            metrics_interval=50,
        )
        s = res["summary"]
        rec = {
            "map_file": map_file, "num_agv": k, "route": route,
            "finished": res["finished"], "makespan": s.get("completed_makespan"),
            "tasked_stationary": s.get("tasked_stationary_count"),
            "blocking_delay_mean": s.get("transport_blocking_delay_mean"),
            "agv_waiting_total": s.get("agv_waiting_time_total"),
            "swap_conflicts": s.get("swap_conflict_count"),
            "agv_busy": s.get("agv_busy_utilization"),
            "wall_time_s": res["wall_time_s"],
            "stationary_trace": [
                {"step": m.get("step"), "stationary": m.get("tasked_stationary_count"),
                 "swap": m.get("swap_conflict_count")}
                for m in res.get("metrics_trace", [])
            ],
        }
    except Exception as e:  # noqa: BLE001
        rec = {"map_file": map_file, "num_agv": k, "route": route,
               "error": f"{type(e).__name__}: {e}"}
    Path(out_file).write_text(json.dumps(rec, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "results" / f"deadlock_pilot_{int(time.time())}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n_runs": len(RUNS)}) + "\n")
    print(f"[deadlock] {len(RUNS)} runs -> {out}")
    for i, (fam, mf, mn, k, route) in enumerate(RUNS):
        t0 = time.time()
        tmp = ROOT / "results" / f".dl_{t0:.0f}_{fam}_{k}_{route}.json"
        ctx = mp.get_context("fork")
        proc = ctx.Process(target=_worker, args=(mf, mn, k, route, str(tmp)))
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            proc.terminate(); proc.join(5)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            rec = {"map_file": mf, "num_agv": k, "route": route,
                   "error": f"hard timeout {args.timeout}s"}
        else:
            try:
                rec = json.loads(tmp.read_text())
            except Exception as e:
                rec = {"map_file": mf, "num_agv": k, "route": route, "error": str(e)}
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        rec["map_family"] = fam
        with open(out, "a") as fo:
            fo.write(json.dumps(rec, default=str) + "\n")
        err = rec.get("error")
        status = f"ERR {err[:50]}" if err else (
            f"mk={rec.get('makespan')} stationary={rec.get('tasked_stationary')} "
            f"blocking={rec.get('blocking_delay_mean')}")
        print(f"[{i+1}/{len(RUNS)}] {fam}|agv{k}|{route} -> {status}", flush=True)
    print("[deadlock] done ->", out)


if __name__ == "__main__":
    main()
