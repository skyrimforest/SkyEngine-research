"""
E8: 修复引擎 vs 旧引擎 配对对照 (论文新证据)
================================================
同 26 个格子, 两臂:
  fixed = 三个修复开关全开 (dedupe+reserve+samecell)
  legacy = 全关 (复现旧引擎行为: 幽灵投递/同格死锁/无效同格运输)
配对记录 makespan / 完工 / 幽灵工序数 / 多余投递数。
输出: results/icaps_pilot_ab.jsonl
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAP_FILE = ROOT / "data" / "mapf" / "medium_maps.yaml"
MAP_NAME = "medium-mazes-seed-0000"
SEEDS = [42, 43, 44]
EPISODE_TIMEOUT = 300.0


def scenario(scen: str, seed: int):
    if scen == "S1":
        return {"enabled": True, "random_seed": seed,
                "schedule": [{"step": 150, "type": "machine_breakdown",
                              "machine_id": 0, "duration_steps": 40}]}
    if scen == "S4":
        return {"enabled": True, "random_seed": seed,
                "schedule": [{"step": 80, "type": "machine_breakdown",
                              "machine_id": 0, "duration_steps": 200}]}
    return None


def _episode(run: dict) -> dict:
    # 开关由父进程通过环境变量传入子进程
    from common.engine_adapter import build_configs, load_fjsp_json, load_map
    from closeloop.orchestrator import RecoveryOrchestrator
    from sky_executor.grid_factory.factory.Component.Coordinator.coordinator import (
        Coordinator,
    )
    from sky_executor.grid_factory.factory.grid_factory_env import GridFactoryEnv

    fjsp = load_fjsp_json(
        ROOT / "data/fjsp_official/brandimarte" / f"{run['instance']}.json")
    gmap = load_map(MAP_FILE, MAP_NAME)
    gc, mc, jc = build_configs(
        fjsp, gmap, num_agv=run["num_agv"], seed=run["seed"],
        max_episode_steps=4096, observation_type="MAPF",
        collision_system="soft")
    env = GridFactoryEnv(grid_config=gc, machine_config=mc, job_config=jc,
                         random_target=False,
                         exception_config=scenario(run["scen"], run["seed"]))
    obs, _ = env.reset()
    coord = Coordinator(
        job_solver="online_fjsp", route_solver="rolling_mapf_http",
        assigner="nearest",
        job_solver_kwargs={
            "service_url": "http://fjsp:8002", "algorithm": "cp_sat",
            "config": {"time_limit": 10.0, "num_workers": 1,
                       "seed": run["seed"], "soft_commitments": True,
                       "soft_commitment_travel_allowance": 200}},
        route_solver_kwargs={
            "service_url": "http://mapf:8001", "time_limit_ms": 500,
            "lns_init_algo": "EECBS", "planning_horizon": 10,
            "execution_window": 5})
    trigger = {"cpsat-static": "never", "cpsat-full": "event"}[run["policy"]]
    orch = RecoveryOrchestrator(coord, env, trigger=trigger, scope="full")

    fin, prevst, preva = set(), {}, {}
    ghosts = defaultdict(list)
    deliveries = defaultdict(int)
    t0 = time.time()
    steps = 0
    term = {}
    for i in range(4096):
        actions = coord.decide(obs)
        obs, _r, term, trunc, _i = env.step(actions)
        steps = i + 1
        for job in env.pogema_env.jobs:
            for o in job.ops:
                k = (job.job_id, o.op_id)
                st = str(getattr(o, "status", "?"))
                on = st in ("PROCESSING", "SUSPENDED")
                a = getattr(o, "arrive_machine_at", -1)
                if a >= 0 and a != preva.get(k, -1):
                    deliveries[k] += 1
                preva[k] = a
                if st == "FINISHED":
                    fin.add(k)
                if on and prevst.get(k) == "off" and k in fin:
                    ghosts[k].append(steps)
                prevst[k] = "on" if on else "off"
        orch.on_step(obs, steps)
        if term.get("job_done"):
            break
    done = bool(term.get("job_done"))
    return {
        **run, "finished": done, "steps": steps,
        "makespan": steps if done else None,
        "n_finished_ops": len(fin),
        "revisions": orch.stats.get("revision_count", 0),
        "ghost_ops": len(ghosts),
        "extra_deliveries": sum(v - 1 for v in deliveries.values() if v > 1),
        "wall_s": round(time.time() - t0, 1),
    }


def _worker(run: dict, out_file: str):
    try:
        rec = _episode(run)
    except Exception as e:  # noqa: BLE001
        rec = {**run, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    Path(out_file).write_text(json.dumps(rec, default=str))


def build_runs():
    runs = []
    for inst in ("mk01", "mk02"):
        for scen in ("S1", "S4"):
            for pol in ("cpsat-static", "cpsat-full"):
                for seed in SEEDS:
                    runs.append({"instance": inst, "scen": scen,
                                 "policy": pol, "seed": seed, "num_agv": 4})
    for pol in ("cpsat-static", "cpsat-full"):
        runs.append({"instance": "mk05", "scen": "S1", "policy": pol,
                     "seed": 42, "num_agv": 4})
    return runs


def main():
    out = ROOT / "results" / "icaps_pilot_ab.jsonl"
    out.write_text("")
    ctx = mp.get_context("fork")
    arms = {
        "legacy": {"AGV_TRANSFER_DEDUPE": "0",
                   "AGV_CELL_RESERVE": "0", "AGV_SAMECELL_SKIP": "0"},
        "fixed": {"AGV_TRANSFER_DEDUPE": "1",
                  "AGV_CELL_RESERVE": "1", "AGV_SAMECELL_SKIP": "1"},
    }
    jobs = [(arm, run) for run in build_runs() for arm in arms]
    print(f"[E8-ab] {len(jobs)} runs -> {out}", flush=True)
    for i, (arm, run) in enumerate(jobs):
        t0 = time.time()
        os.environ.update(arms[arm])
        tmp = ROOT / "results" / f".ab_{os.getpid()}_{t0:.0f}.json"
        proc = ctx.Process(target=_worker, args=({**run, "arm": arm}, str(tmp)))
        proc.start()
        proc.join(EPISODE_TIMEOUT)
        if proc.is_alive():
            proc.terminate(); proc.join(5)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            rec = {**run, "arm": arm,
                   "error": f"hard timeout {EPISODE_TIMEOUT}s"}
        else:
            try:
                rec = json.loads(tmp.read_text())
            except Exception as e:
                rec = {**run, "arm": arm, "error": f"unreadable: {e}"}
        tmp.unlink(missing_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        err = rec.get("error")
        st = (f"ERR {err[:44]}" if err else
              f"mk={rec.get('makespan')} ghost={rec.get('ghost_ops')} "
              f"extradel={rec.get('extra_deliveries')}")
        print(f"[E8-ab {i+1}/{len(jobs)}] {arm:6s} {run['instance']}|"
              f"{run['scen']}|{run['policy']}|s{run['seed']} -> {st} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[E8-ab] done", flush=True)


if __name__ == "__main__":
    main()
