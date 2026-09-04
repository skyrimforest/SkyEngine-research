"""方向7 v2: 跑批器 (容器内运行)
================================
读取 scenarios_v2.jsonl, 逐条跑 episode (mp fork 隔离 + 硬超时),
结果写 episodes/<scen_id>.json。断点续跑: 已存在且非 error 的跳过。
用法: docker exec -w /work/sky_research skyresearch python llmdiag/v2/run_scenarios.py [--limit N] [--ids a,b]
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
ROOT = HERE.parents[1]
TIMEOUT_S = 300.0


def run_one(scn: dict):
    from closeloop.orchestrator import run_closed_episode
    route = "astar" if scn["policy"] == "greedy-reactive" else "rolling_mapf_http"
    return run_closed_episode(
        fjsp_path=ROOT / "data/fjsp_official/brandimarte" / f"{scn['instance']}.json",
        map_file=ROOT / scn["map_file"], map_name=scn["map_name"],
        policy=scn["policy"], exception_config=scn["exception_config"],
        num_agv=scn["num_agv"], seed=scn["seed"], max_steps=4096,
        route_solver_name=route,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args()
    outdir = HERE / "episodes"
    outdir.mkdir(exist_ok=True)
    only = set(args.ids.split(",")) if args.ids else None

    scns = [json.loads(l) for l in (HERE / "scenarios_v2.jsonl").read_text().splitlines() if l.strip()]
    if only:
        scns = [s for s in scns if s["scen_id"] in only]
    done = 0
    for scn in scns:
        f = outdir / f"{scn['scen_id']}.json"
        if f.exists() and "error" not in json.loads(f.read_text()):
            done += 1
            continue
        tmp = outdir / f".tmp_{scn['scen_id']}.json"

        def _target():
            t0 = time.time()
            try:
                res = run_one(scn)
                res["wall_time_s"] = round(time.time() - t0, 2)
                for k in ("agv_transit_heatmap", "agv_occupancy_heatmap"):
                    res.pop(k, None)  # 记录瘦身: 热力图不进档案也不进库
            except Exception as e:  # noqa: BLE001
                res = {"error": f"{type(e).__name__}: {e}"}
            tmp.write_text(json.dumps(res, default=str))

        ctx = mp.get_context("fork")
        p = ctx.Process(target=_target)
        p.start()
        p.join(args.timeout)
        if p.is_alive():
            p.terminate(); p.join(5)
            if p.is_alive():
                p.kill(); p.join(5)
            res = {"error": f"hard timeout {args.timeout}s"}
        else:
            res = json.loads(tmp.read_text()) if tmp.exists() else {"error": "no output"}
        tmp.unlink(missing_ok=True)
        f.write_text(json.dumps(res, default=str))
        done += 1
        status = "ERR" if "error" in res else f"mk={res.get('makespan')} fin={res.get('finished')}"
        print(f"[{done}/{len(scns)}] {scn['scen_id']}: {status}", flush=True)
        if args.limit and done >= args.limit:
            break


if __name__ == "__main__":
    main()
