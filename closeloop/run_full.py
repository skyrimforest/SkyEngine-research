"""
方向1 ICAPS 冲刺: 全量实验战役
===============================
四个实验相 (顺序执行, 每相独立 JSONL, 断点续跑, 单 episode 子进程隔离):

E1 主矩阵 (soft=on):  {mk01,mk02,mk05,mk07} x 迷宫 x AGV{2,4,6}
   x {S0,S1,S4,S3} x {greedy-reactive, cpsat-static, cpsat-full, cpsat-partial}
   x seeds{42,43,44}                            = 576 ep
E2 地图稳健性 (soft=on): {mk01,mk05} x 随机地图 x 其余同 E1 = 288 ep
E3 遗留承诺对照 (soft=off): {mk01,mk05} x 迷宫 x agv4 x {S1,S4,S3}
   x {cpsat-full, cpsat-partial} x 3 seeds       = 36 ep
E4 罚金扫描 (soft=on): {mk01,mk05} x 迷宫 x agv4 x {S1,S4}
   x cpsat-full x allowance{50,100,200,400} x 3 seeds = 48 ep

用法:
  python closeloop/run_full.py --phase smoke   # 小规模确认
  python closeloop/run_full.py --phase all     # 全战役 (后台, 约3-6h)
  python closeloop/run_full.py --phase E1 --resume
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

MAPS = {
    "maze": ("data/mapf/medium_maps.yaml", "medium-mazes-seed-0000"),
    "random": ("data/mapf/random_maps.yaml", "validation-random-seed-000"),
}

POLICIES = ["greedy-reactive", "cpsat-static", "cpsat-full", "cpsat-partial"]
INSTANCES_E1 = ["mk01", "mk02", "mk05", "mk07"]
INSTANCES_E2 = ["mk01", "mk05"]
SEEDS = [42, 43, 44]
EPISODE_TIMEOUT = 1200.0  # myopic 臂含 EECBS rollout, 需要更长墙钟


def scenario(scen: str, seed: int):
    if scen == "S0":
        return None
    if scen == "S1":  # 单机中等故障
        return {"enabled": True, "random_seed": seed,
                "schedule": [{"step": 150, "type": "machine_breakdown",
                              "machine_id": 0, "duration_steps": 40}]}
    if scen == "S4":  # 单机强故障
        return {"enabled": True, "random_seed": seed,
                "schedule": [{"step": 80, "type": "machine_breakdown",
                              "machine_id": 0, "duration_steps": 200}]}
    if scen == "S3":  # 随机温和故障
        return {"enabled": True, "random_seed": seed, "preset": "mild_failure"}
    raise ValueError(scen)


def build_runs(phase: str):
    runs = []
    if phase == "smoke":
        for inst in ["mk01"]:
            for scen in ["S1"]:
                for pol in ["greedy-reactive", "cpsat-static", "cpsat-full", "cpsat-partial"]:
                    runs.append(dict(phase="smoke", instance=inst, map="maze",
                                     num_agv=4, scen=scen, policy=pol, seed=42,
                                     soft=True, allowance=200))
        return runs
    if phase in ("E1", "all"):
        for inst in INSTANCES_E1:
            for nagv in [2, 4, 6]:
                for scen in ["S0", "S1", "S4", "S3"]:
                    for pol in POLICIES:
                        for seed in SEEDS:
                            runs.append(dict(phase="E1", instance=inst, map="maze",
                                             num_agv=nagv, scen=scen, policy=pol,
                                             seed=seed, soft=True, allowance=200))
        if phase == "E1":
            return runs
    if phase in ("E2", "all"):
        for inst in INSTANCES_E2:
            for nagv in [2, 4, 6]:
                for scen in ["S0", "S1", "S4", "S3"]:
                    for pol in POLICIES:
                        for seed in SEEDS:
                            runs.append(dict(phase="E2", instance=inst, map="random",
                                             num_agv=nagv, scen=scen, policy=pol,
                                             seed=seed, soft=True, allowance=200))
        if phase == "E2":
            return runs
    if phase in ("E3", "all"):
        for inst in INSTANCES_E2:
            for scen in ["S1", "S4", "S3"]:
                for pol in ["cpsat-full", "cpsat-partial"]:
                    for seed in SEEDS:
                        runs.append(dict(phase="E3", instance=inst, map="maze",
                                         num_agv=4, scen=scen, policy=pol,
                                         seed=seed, soft=False, allowance=200))
        if phase == "E3":
            return runs
    if phase in ("E4", "all"):
        for inst in INSTANCES_E2:
            for scen in ["S1", "S4"]:
                for allowance in [50, 100, 200, 400]:
                    for seed in SEEDS:
                        runs.append(dict(phase="E4", instance=inst, map="maze",
                                         num_agv=4, scen=scen, policy="cpsat-full",
                                         seed=seed, soft=True, allowance=allowance))
        if phase == "E4":
            return runs
    if phase in ("E5", "all"):
        # 路由求解器敏感性: 需配合 scripts/run_E5.sh 换 mapf 服务镜像。
        # Python 版 lacam/pibt 服务在 500ms 滚动预算内无法完成初始化,
        # E5 专用放宽到 3s (引擎 HTTP 超时同步放宽到 30s)。
        route_img = os.getenv("E5_ROUTE", "unknown")
        for inst in ["mk01", "mk05"]:
            for scen in ["S0", "S3", "S4"]:
                for pol in ["cpsat-static", "cpsat-full"]:
                    for seed in SEEDS:
                        runs.append(dict(phase="E5", instance=inst, map="maze",
                                         num_agv=4, scen=scen, policy=pol,
                                         seed=seed, soft=True, allowance=200,
                                         route_img=route_img,
                                         mapf_time_limit_ms=3000))
        if phase == "E5":
            return runs
    if phase in ("E7", "all"):
        # value-aware 触发对照: 三策略统一滚动 EECBS 路由 (修复引擎 +
        # myopic 回滚后的会话重置钩子, 使快照/回滚与远程会话兼容)
        for inst in ["mk01", "mk05"]:
            for nagv in [4, 6]:
                for scen in ["S1", "S4", "S3"]:
                    for pol in ["cpsat-static", "cpsat-full", "cpsat-myopic"]:
                        for seed in SEEDS:
                            runs.append(dict(phase="E7", instance=inst, map="maze",
                                             num_agv=nagv, scen=scen, policy=pol,
                                             seed=seed, soft=True, allowance=200,
                                             route="rolling_mapf_http"))
        if phase == "E7":
            return runs
    return runs


def _worker(run: dict, out_file: str):
    from closeloop.orchestrator import run_closed_episode

    map_file, map_name = MAPS[run["map"]]
    os.environ["FJSP_SOFT_COMMIT"] = "1" if run["soft"] else "0"
    os.environ["FJSP_SOFT_TRAVEL_ALLOWANCE"] = str(run["allowance"])
    rec = {k: run[k] for k in
           ("phase", "instance", "map", "num_agv", "scen", "policy", "seed",
            "soft", "allowance")}
    if "route_img" in run:
        rec["route_img"] = run["route_img"]
    route_kwargs = None
    if run.get("mapf_time_limit_ms"):
        # E5: 放宽滚动预算以适配 Python 版路由服务
        os.environ["HTTP_TIMEOUT"] = "30"
        route_kwargs = {
            "service_url": "http://mapf:8001",
            "time_limit_ms": int(run["mapf_time_limit_ms"]),
            "planning_horizon": 10,
            "execution_window": 5,
        }
    if run.get("route") == "astar":
        route_name = "astar"
    elif run["policy"] == "greedy-reactive":
        route_name = "astar"
    else:
        route_name = "rolling_mapf_http"
    try:
        res = run_closed_episode(
            fjsp_path=ROOT / "data" / "fjsp_official" / "brandimarte" / f"{run['instance']}.json",
            map_file=ROOT / map_file, map_name=map_name,
            policy=run["policy"],
            exception_config=scenario(run["scen"], run["seed"]),
            num_agv=run["num_agv"], seed=run["seed"], max_steps=4096,
            route_solver_name=route_name,
            route_solver_kwargs=route_kwargs,
        )
        os_ = res.get("orchestrator_stats", {})
        rec.update(
            finished=res.get("finished"), steps=res.get("steps"),
            makespan=res.get("makespan"),
            revisions=os_.get("revision_count", 0),
            revision_fails=os_.get("revision_fail_count", 0),
            partial_fallbacks=os_.get("partial_fallback_count", 0),
            myopic_evals=os_.get("myopic_evals", 0),
            myopic_replans=os_.get("myopic_replans", 0),
            myopic_skips=os_.get("myopic_skips", 0),
            n_events=res.get("n_events"),
            agv_busy=res.get("summary", {}).get("agv_busy_utilization"),
            wall_s=res.get("wall_time_s"),
        )
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"{type(e).__name__}: {str(e)[:150]}")
    Path(out_file).write_text(json.dumps(rec, default=str))


def run_key(r: dict) -> str:
    keys = ("phase", "instance", "map", "num_agv", "scen", "policy",
            "seed", "soft", "allowance", "route_img")
    return "|".join(str(r.get(k)) for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="smoke",
                    choices=["smoke", "E1", "E2", "E3", "E4", "E5", "E7", "all"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    phases = ["E1", "E2", "E3", "E4"] if args.phase == "all" else [args.phase]
    for phase in phases:
        runs = build_runs(phase)
        if args.limit:
            runs = runs[: args.limit]
        out = ROOT / "results" / f"icaps_{phase}.jsonl"
        if args.resume and out.exists():
            done = set()
            for line in out.read_text().splitlines():
                try:
                    d = json.loads(line)
                    done.add(run_key(d))
                except Exception:
                    pass
            runs = [r for r in runs if run_key(r) not in done]
        if not (args.resume and out.exists()):
            out.write_text("")
        print(f"[{phase}] {len(runs)} runs -> {out}", flush=True)
        ctx = mp.get_context("fork")
        t_phase = time.time()
        for i, run in enumerate(runs):
            t0 = time.time()
            tmp = ROOT / "results" / f".icaps_{phase}_{os.getpid()}_{t0:.0f}.json"
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
                f"mk={rec.get('makespan')} rev={rec.get('revisions')}")
            print(f"[{phase} {i+1}/{len(runs)}] {run['instance']}|{run['map']}"
                  f"|agv{run['num_agv']}|{run['scen']}|{run['policy']}|s{run['seed']}"
                  f" -> {status} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{phase}] done in {(time.time()-t_phase)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
