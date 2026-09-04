"""方向7 v2: 健康地图池筛选
============================
迷宫地图硬度差异极大 (E2: 9/99 饥饿活锁), 注入类/基线需要"能完工"的对照地图。
对候选地图各跑一次 mk01+greedy+K=4 基线, 完工者进入健康池。
结果缓存 results_v2/map_screen.json, scenarios_v2 读取之。

用法: python llmdiag/screen_maps.py [--seeds 16] [--instance mk01]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmdiag.episode_runner import run_instrumented_episode  # noqa: E402
from llmdiag.scenarios_v2 import CaseSpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16,
                    help="筛选 medium-mazes-seed-0000..(N-1)")
    ap.add_argument("--instance", default="mk01")
    ap.add_argument("--num-agv", type=int, default=4)
    args = ap.parse_args()

    out = ROOT / "llmdiag" / "results_v2" / "map_screen.json"
    out.parent.mkdir(exist_ok=True)
    pool = {"healthy": [], "pathological": [], "screened_at": time.strftime("%F %T"),
            "config": {"instance": args.instance, "num_agv": args.num_agv}}
    for i in range(args.seeds):
        name = f"medium-mazes-seed-{i:04d}"
        spec = CaseSpec(
            case_id=f"screen_{name}", target_class="baseline",
            instance=args.instance, policy="greedy-reactive",
            num_agv=args.num_agv, map_name=name, seed=42 + i,
            exception_config={"enabled": False, "random_seed": 42 + i},
            route_solver="astar", query="",
        ).to_dict()
        t0 = time.time()
        try:
            rec = run_instrumented_episode(spec)
            finished, ms = bool(rec["finished"]), rec.get("makespan")
        except Exception as e:  # noqa: BLE001
            finished, ms = False, f"ERR:{e}"
        (pool["healthy"] if finished else pool["pathological"]).append(
            {"map": name, "makespan": ms, "wall": round(time.time() - t0, 1)})
        print(f"{name}: finished={finished} makespan={ms} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out.write_text(json.dumps(pool, ensure_ascii=False, indent=1))
    print(f"健康池 {len(pool['healthy'])}/{args.seeds} -> {out}")


if __name__ == "__main__":
    main()
