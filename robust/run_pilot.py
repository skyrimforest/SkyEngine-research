"""
方向6 pilot: 加工时长不确定性下的鲁棒性
=======================================
引擎机制: ProcessingTimeSampler 在工序开工时采样真实时长
(名义时长供调度), 预设 mild/moderate/high (multiplier_uniform)。

研究问题: 名义时长上开环规划的 CP-SAT 计划, 在执行方差下退化多少?
反应式 greedy 天然鲁棒? 周期性重规划 (orchestrator periodic-K) 能否
恢复鲁棒性?

矩阵: {mk01, mk02} x 迷宫 x agv4 x 方差 {none, mild, moderate, high}
      x 策略 {greedy+astar, cpsat-static+eecbs, cpsat-periodic100+eecbs}
= 24 episodes

用法: python robust/run_pilot.py [--limit N]
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
from bench.suite import instance_path  # noqa: E402

VARIANCE = {
    "none": None,
    "mild": {"enabled": True, "preset": "mild_variance", "random_seed": 42},
    "moderate": {"enabled": True, "preset": "moderate_variance", "random_seed": 42},
    "high": {"enabled": True, "preset": "high_variance", "random_seed": 42},
}

MAP_FILE = ROOT / "data" / "mapf" / "medium_maps.yaml"
MAP_NAME = "medium-mazes-seed-0000"

CPSAT_KW = {"service_url": "http://fjsp:8002", "algorithm": "cp_sat",
            "config": {"time_limit": 10.0, "num_workers": 1}}
EECBS_KW = {"service_url": "http://mapf:8001", "time_limit_ms": 500,
            "lns_init_algo": "EECBS", "planning_horizon": 10, "execution_window": 5}

POLICIES = {
    "greedy+astar": dict(job_solver="greedy", route_solver="astar",
                         route_solver_kwargs=None, transfer_aware=True),
    "cpsat-static": dict(job_solver="online_fjsp", route_solver="rolling_mapf_http",
                         route_solver_kwargs=dict(EECBS_KW), transfer_aware=False),
    "cpsat-periodic100": dict(job_solver="online_fjsp", route_solver="rolling_mapf_http",
                              route_solver_kwargs=dict(EECBS_KW), transfer_aware=False,
                              periodic_revision=100),
}


def _worker(inst, varname, polname, out_file):
    pol = POLICIES[polname]
    pt_cfg = VARIANCE[varname]
    try:
        # 周期修订策略复用方向1的编排器: 需要走 closeloop 的运行器
        if pol.get("periodic_revision"):
            from closeloop.orchestrator import run_closed_episode

            res = run_closed_episode(
                ROOT / "data" / "fjsp_official" / "brandimarte" / f"{inst}.json",
                MAP_FILE, MAP_NAME,
                policy="cpsat-full",
                exception_config=None,
                num_agv=4, seed=42, max_steps=4096,
                trigger_override=f"periodic-{pol['periodic_revision']}",
            )
            rec = {"makespan": res.get("makespan"),
                   "finished": res.get("finished"),
                   "revisions": res["orchestrator_stats"]["revision_count"]}
        else:
            res = run_episode(
                fjsp_path=instance_path(inst), map_file=MAP_FILE, map_name=MAP_NAME,
                num_agv=4, seed=42, max_steps=4096,
                job_solver=pol["job_solver"], route_solver=pol["route_solver"],
                assigner="nearest",
                job_solver_kwargs=dict(CPSAT_KW) if pol["job_solver"] == "online_fjsp" else None,
                route_solver_kwargs=pol["route_solver_kwargs"],
                transfer_aware=pol["transfer_aware"],
                processing_time_config=pt_cfg,
            )
            s = res["summary"]
            rec = {"makespan": s.get("completed_makespan") if res["finished"] else 4096,
                   "finished": res["finished"], "revisions": 0,
                   "queue_wait": s.get("operation_queue_waiting_time_mean"),
                   "blocking": s.get("transport_blocking_delay_mean")}
        rec.update(instance=inst, variance=varname, policy=polname,
                   wall_time_s=res.get("wall_time_s"))
    except Exception as e:  # noqa: BLE001
        rec = {"instance": inst, "variance": varname, "policy": polname,
               "error": f"{type(e).__name__}: {e}"}
    Path(out_file).write_text(json.dumps(rec, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "results" / f"robust_pilot_{int(time.time())}.jsonl"
    out.parent.mkdir(exist_ok=True)
    runs = [(i, v, p) for i in ["mk01", "mk02"] for v in VARIANCE for p in POLICIES]
    if args.limit:
        runs = runs[: args.limit]
    out.write_text(json.dumps({"n_runs": len(runs)}) + "\n")
    print(f"[robust] {len(runs)} runs -> {out}")
    for i, (inst, var, pol) in enumerate(runs):
        t0 = time.time()
        tmp = ROOT / "results" / f".rb_{t0:.0f}_{inst}_{var}_{pol}.json"
        ctx = mp.get_context("fork")
        proc = ctx.Process(target=_worker, args=(inst, var, pol, str(tmp)))
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            proc.terminate(); proc.join(5)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            rec = {"instance": inst, "variance": var, "policy": pol,
                   "error": f"hard timeout {args.timeout}s"}
        else:
            try:
                rec = json.loads(tmp.read_text())
            except Exception as e:
                rec = {"instance": inst, "variance": var, "policy": pol, "error": str(e)}
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        err = rec.get("error")
        status = f"ERR {err[:50]}" if err else f"mk={rec.get('makespan')} rev={rec.get('revisions')}"
        print(f"[{i+1}/{len(runs)}] {inst}|{var}|{pol} -> {status}", flush=True)
    print("[robust] done ->", out)


if __name__ == "__main__":
    main()
