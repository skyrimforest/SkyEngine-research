"""
方向1 pilot: 闭环策略 x 扰动场景 的小规模对比
================================================
矩阵: {mk01, mk02} x 迷宫 x agv4 x {S0无扰动, S1单机故障, S2机+AGV故障, S3随机mild}
      x {greedy-reactive, cpsat-static, cpsat-full, cpsat-partial} x seed42
= 2 x 4 x 4 = 32 episodes (预计 10-15 分钟)

用法: python closeloop/run_pilot.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from closeloop.orchestrator import run_closed_episode  # noqa: E402
from bench.suite import instance_optimum  # noqa: E402

POLICIES = ["greedy-reactive", "cpsat-static", "cpsat-full", "cpsat-partial"]

# 扰动场景: step 定在 makespan 的 1/3 处附近 (mk01 cpsat 约 420 步)
SCENARIOS = {
    "S0_none": None,
    "S1_machine@150": {
        "enabled": True,
        "random_seed": 42,
        "schedule": [
            {"step": 150, "type": "machine_breakdown", "machine_id": 0,
             "duration_steps": 40},
        ],
    },
    "S2_machine+agv": {
        "enabled": True,
        "random_seed": 42,
        "schedule": [
            {"step": 150, "type": "machine_breakdown", "machine_id": 0,
             "duration_steps": 40},
            {"step": 300, "type": "agv_breakdown", "agv_id": 1,
             "duration_steps": 20},
        ],
    },
    "S3_mild_stochastic": {"enabled": True, "random_seed": 42,
                           "preset": "mild_failure"},
}

INSTANCES = ["mk01", "mk02"]


def run_one(inst, scen, policy, timeout_s=300.0):
    fjsp = ROOT / "data" / "fjsp_official" / "brandimarte" / f"{inst}.json"

    def _alarm(signum, frame):
        raise TimeoutError(f"episode timeout {timeout_s}s")

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout_s))
    try:
        res = run_closed_episode(
            fjsp_path=fjsp,
            map_file=ROOT / "data" / "mapf" / "medium_maps.yaml",
            map_name="medium-mazes-seed-0000",
            policy=policy,
            exception_config=SCENARIOS[scen],
            num_agv=4, seed=42, max_steps=4096,
        )
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}", "config": {"instance": inst}}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    res["config"].update(instance=inst, scenario=scen, policy=policy)
    res["fjsp_optimum"] = instance_optimum(inst)
    if res.get("makespan") and res.get("fjsp_optimum"):
        res["overhead"] = round(res["makespan"] / res["fjsp_optimum"], 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        ROOT / "results" / f"closeloop_pilot_{int(time.time())}.jsonl"
    )
    out.parent.mkdir(exist_ok=True)
    runs = [(i, s, p) for i in INSTANCES for s in SCENARIOS for p in POLICIES]
    if args.limit:
        runs = runs[: args.limit]
    out.write_text(json.dumps({"n_runs": len(runs),
                               "started": time.strftime("%F %T")}) + "\n")
    print(f"[closeloop] {len(runs)} runs -> {out}")
    for i, (inst, scen, policy) in enumerate(runs):
        rec = run_one(inst, scen, policy, timeout_s=args.timeout)
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        if "error" in rec:
            print(f"[{i+1}/{len(runs)}] {inst}|{scen}|{policy} -> ERR "
                  f"{rec['error'][:80]}", flush=True)
        else:
            os = rec.get("orchestrator_stats", {})
            print(f"[{i+1}/{len(runs)}] {inst}|{scen}|{policy} -> "
                  f"makespan={rec.get('makespan')} steps={rec.get('steps')} "
                  f"rev={os.get('revision_count')} wall={rec.get('wall_time_s')}s",
                  flush=True)
    print("[closeloop] done ->", out)


if __name__ == "__main__":
    main()
