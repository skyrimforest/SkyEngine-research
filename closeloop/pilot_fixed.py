"""
引擎修复后试点重跑 (dedupe + 终点唯一不变量)
=============================================
目的: 度量修复对 R2 (static vs full) 结论的影响 + 旧楔死格能否完工。
P1 核心: {mk01,mk02} x 迷宫 x agv4 x {S1,S4} x {static,full} x 3 seeds = 24
P2 楔死: mk05 x 迷宫 x agv4 x S1 x {static,full} x seed42 = 2   (旧引擎该格 censored)
输出: results/icaps_pilot_fixed.jsonl
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
    total_ops = sum(len(j) for j in fjsp["jobs"])
    extra_del = sum(v - 1 for v in deliveries.values() if v > 1)
    summ = env.pogema_env.summary() if hasattr(env.pogema_env, "summary") else {}
    return {
        **run, "finished": done, "steps": steps,
        "makespan": steps if done else None,
        "n_finished_ops": len(fin), "n_total_ops": total_ops,
        "revisions": orch.stats.get("revision_count", 0),
        "revision_fails": orch.stats.get("revision_fail_count", 0),
        "ghost_ops": len(ghosts), "ghost_events": dict(ghosts),
        "extra_deliveries": extra_del,
        "agv_travel": summ.get("agv_travel_steps"),
        "agv_wait": summ.get("agv_wait_steps"),
        "wall_s": round(time.time() - t0, 1),
        "dedupe": os.environ.get("AGV_TRANSFER_DEDUPE", "1"),
        "reserve": os.environ.get("AGV_CELL_RESERVE", "1"),
    }


def _worker(run: dict, out_file: str):
    try:
        rec = _episode(run)
    except Exception as e:  # noqa: BLE001
        rec = {**run, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    Path(out_file).write_text(json.dumps(rec, default=str))


def main():
    runs = []
    for inst in ("mk01", "mk02"):
        for scen in ("S1", "S4"):
            for pol in ("cpsat-static", "cpsat-full"):
                for seed in SEEDS:
                    runs.append({"instance": inst, "scen": scen,
                                 "policy": pol, "seed": seed, "num_agv": 4,
                                 "phase": "P1"})
    for pol in ("cpsat-static", "cpsat-full"):
        runs.append({"instance": "mk05", "scen": "S1", "policy": pol,
                     "seed": 42, "num_agv": 4, "phase": "P2"})
    out = ROOT / "results" / "icaps_pilot_fixed.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add("|".join(str(d.get(k)) for k in
                                  ("phase", "instance", "scen", "policy", "seed")))
            except Exception:
                pass
    else:
        out.write_text("")
    runs = [r for r in runs if "|".join(
        str(r[k]) for k in ("phase", "instance", "scen", "policy", "seed")
    ) not in done]
    print(f"[pilot] {len(runs)} runs -> {out}", flush=True)
    ctx = mp.get_context("fork")
    for i, run in enumerate(runs):
        t0 = time.time()
        tmp = ROOT / "results" / f".pilot_{os.getpid()}_{t0:.0f}.json"
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
        tmp.unlink(missing_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        err = rec.get("error")
        st = (f"ERR {err[:50]}" if err else
              f"mk={rec.get('makespan')} ghost={rec.get('ghost_ops')} "
              f"extradel={rec.get('extra_deliveries')} rev={rec.get('revisions')}")
        print(f"[pilot {i+1}/{len(runs)}] {run['instance']}|{run['scen']}|"
              f"{run['policy']}|s{run['seed']} -> {st} ({time.time()-t0:.0f}s)",
              flush=True)
    print("[pilot] done", flush=True)


if __name__ == "__main__":
    main()
