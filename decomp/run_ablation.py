"""
方向3: 分解代价 (Price of Decomposition, PoD) 消融实验
=====================================================
在固定 (实例, 地图, K) 下, 用 2x2 因子 + 反馈开关分离分层协同的信息损失:

                 路由层局部反应 (astar)   路由层全局滚动 (eecbs)
  调度层反应式     greedy+astar (基准)     greedy+eecbs  (A3)
  调度层协调式     cpsat+astar    (A2)     cpsat+eecbs   (锚点)

  A1 反馈开关: greedy+astar ± transfer_aware (R->J 信息反馈价值)

PoD 定义 (论文 §3):
  PoD_total = [C(11) - C(22)] / C(22)          # 分层基准 vs 协调锚点
  PoD_sched = [C(21) - C(22)] / C(22)          # 纯调度层损失
  PoD_route = [C(12) - C(22)] / C(22)          # 纯路由层损失
  交互项      = PoD_total - PoD_sched - PoD_route
  反馈价值    = [C(greedy+astar, no fb) - C(greedy+astar, fb)] / C(., fb)

用法: python decomp/run_ablation.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.suite import instance_optimum, instance_path  # noqa: E402
from common.engine_adapter import run_episode  # noqa: E402

MAP_FILE = ROOT / "data" / "mapf" / "medium_maps.yaml"
MAP_NAME = "medium-mazes-seed-0000"

EECBS_KWARGS = {
    "service_url": "http://mapf:8001",
    "time_limit_ms": 500,
    "lns_init_algo": "EECBS",
    "planning_horizon": 10,
    "execution_window": 5,
}

# 消融组合: name -> (job, route, assigner, job_kwargs, route_kwargs, transfer_aware)
ARMS = {
    "greedy+astar": ("greedy", "astar", "nearest", None, None, True),
    "greedy+astar-nofb": ("greedy", "astar", "nearest", None, None, False),
    "greedy+eecbs": ("greedy", "rolling_mapf_http", "nearest", None, dict(EECBS_KWARGS), False),
    "cpsat+astar": ("online_fjsp", "astar", "nearest",
                    {"service_url": "http://fjsp:8002", "algorithm": "cp_sat",
                     "config": {"time_limit": 10.0, "num_workers": 1}}, None, False),
    "cpsat+eecbs": ("online_fjsp", "rolling_mapf_http", "nearest",
                    {"service_url": "http://fjsp:8002", "algorithm": "cp_sat",
                     "config": {"time_limit": 10.0, "num_workers": 1}},
                    dict(EECBS_KWARGS), False),
}


def _worker(arm, inst, num_agv, out_file):
    job, route, assigner, jk, rk, fb = ARMS[arm]
    rec = {"arm": arm, "instance": inst, "num_agv": num_agv, "seed": 42,
           "fjsp_optimum": instance_optimum(inst)}
    try:
        res = run_episode(
            fjsp_path=instance_path(inst), map_file=MAP_FILE, map_name=MAP_NAME,
            num_agv=num_agv, seed=42, max_steps=4096,
            job_solver=job, route_solver=route, assigner=assigner,
            job_solver_kwargs=jk, route_solver_kwargs=rk,
            transfer_aware=fb, warmup_timeout=10.0,
        )
        s = res["summary"]
        rec.update(
            finished=res["finished"], steps=res["steps"],
            makespan=s.get("completed_makespan") if res["finished"] else 4096,
            agv_busy=s.get("agv_busy_utilization"),
            blocking=s.get("transport_blocking_delay_mean"),
            wall_time_s=res["wall_time_s"],
        )
    except Exception as e:  # noqa: BLE001
        rec.update(finished=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-800:])
    Path(out_file).write_text(json.dumps(rec, default=str))


def run_arm(arm, inst, num_agv, timeout_s=180.0):
    t0 = time.time()
    tmp = ROOT / "results" / f".decomp_{t0:.0f}_{arm.replace('+','_')}_{inst}_{num_agv}.json"
    ctx = mp.get_context("fork")
    proc = ctx.Process(target=_worker, args=(arm, inst, num_agv, str(tmp)))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate(); proc.join(5)
        if proc.is_alive():
            proc.kill(); proc.join(5)
        rec = {"arm": arm, "instance": inst, "num_agv": num_agv, "seed": 42,
               "finished": False, "error": f"hard timeout {timeout_s}s"}
    else:
        try:
            rec = json.loads(tmp.read_text())
        except Exception as e:
            rec = {"arm": arm, "instance": inst, "num_agv": num_agv,
                   "finished": False, "error": f"unreadable: {e}"}
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    rec["run_wall_time_s"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "results" / f"decomp_pilot_{int(time.time())}.jsonl"
    out.parent.mkdir(exist_ok=True)
    runs = [(a, i, k) for i in ["mk01", "mk02"] for k in [2, 4, 6] for a in ARMS]
    if args.limit:
        runs = runs[: args.limit]
    out.write_text(json.dumps({"n_runs": len(runs),
                               "started": time.strftime("%F %T")}) + "\n")
    print(f"[decomp] {len(runs)} runs -> {out}")
    for i, (arm, inst, k) in enumerate(runs):
        rec = run_arm(arm, inst, k, timeout_s=args.timeout)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        status = "ERR:" + rec.get("error", "")[:50] if rec.get("error") else f"{rec['run_wall_time_s']}s"
        print(f"[{i+1}/{len(runs)}] {arm}|{inst}|agv{k} -> makespan={rec.get('makespan')} ({status})",
              flush=True)
    print("[decomp] done ->", out)


if __name__ == "__main__":
    main()
