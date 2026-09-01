"""
方向2: 基准套件执行器
=====================
用法 (容器内):
  python bench/run_suite.py --suite pilot --out results/bench_pilot.jsonl
  python bench/run_suite.py --suite pilot --limit 5   # 冒烟

输出: JSONL, 每行一个 episode 的完整记录 (config + summary + 派生指标)。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.suite import (  # noqa: E402
    MAP_FAMILIES,
    SOLVER_COMBOS,
    SuiteConfig,
    instance_optimum,
    instance_path,
    instance_stats,
)
from common.engine_adapter import run_episode  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _episode_worker(run: dict, out_file: str) -> None:
    """子进程入口: 跑一个 episode 并把记录写入 out_file。"""
    combo = SOLVER_COMBOS[run["combo"]]
    map_file, map_name = MAP_FAMILIES[run["map_family"]]
    rec = {
        "instance": run["instance"],
        "map_family": run["map_family"],
        "map_name": map_name,
        "num_agv": run["num_agv"],
        "combo": run["combo"],
        "seed": run["seed"],
        **{f"inst_{k}": v for k, v in instance_stats(run["instance"]).items()},
        "fjsp_optimum": instance_optimum(run["instance"]),
    }
    try:
        res = run_episode(
            fjsp_path=instance_path(run["instance"]),
            map_file=ROOT / map_file,
            map_name=map_name,
            num_agv=run["num_agv"],
            seed=run["seed"],
            max_steps=run["max_steps"],
            job_solver=combo["job_solver"],
            route_solver=combo["route_solver"],
            assigner=combo["assigner"],
            job_solver_kwargs=combo.get("job_solver_kwargs"),
            route_solver_kwargs=combo.get("route_solver_kwargs"),
            transfer_aware=combo["job_solver"] == "greedy",
            warmup_timeout=10.0,
        )
        s = res["summary"]
        makespan = s.get("completed_makespan") if res["finished"] else run["max_steps"]
        opt = rec["fjsp_optimum"]
        rec.update(
            finished=res["finished"],
            steps=res["steps"],
            makespan=makespan,
            transport_overhead=(makespan / opt) if (opt and makespan) else None,
            agv_busy=s.get("agv_busy_utilization"),
            agv_loaded=s.get("agv_loaded_utilization"),
            machine_util=s.get("machine_utilization"),
            transport_delay_ratio=s.get("transport_delay_ratio"),
            empty_pickup_mean=s.get("empty_pickup_time_mean"),
            blocking_delay_mean=s.get("transport_blocking_delay_mean"),
            queue_wait_mean=s.get("operation_queue_waiting_time_mean"),
            swap_conflicts=s.get("swap_conflict_count"),
            stationary=s.get("tasked_stationary_count"),
            wall_time_s=res["wall_time_s"],
        )
    except Exception as e:  # noqa: BLE001
        rec.update(finished=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1500:])
    Path(out_file).write_text(json.dumps(rec, ensure_ascii=False, default=str))


def episode_record(run: dict, timeout_s: float = 0.0) -> dict:
    """在隔离子进程中运行 episode; 超时则 SIGKILL, 保证套件永不卡死。"""
    import multiprocessing as mp

    t0 = time.time()
    ctx = mp.get_context("fork")
    tmp = ROOT / "results" / f".ep_{os.getpid()}_{t0:.0f}.json"
    proc = ctx.Process(target=_episode_worker, args=(run, str(tmp)))
    proc.start()
    proc.join(timeout_s if timeout_s > 0 else None)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        rec = {
            "instance": run["instance"],
            "map_family": run["map_family"],
            "num_agv": run["num_agv"],
            "combo": run["combo"],
            "seed": run["seed"],
            "fjsp_optimum": instance_optimum(run["instance"]),
            "finished": False,
            "error": f"hard timeout after {timeout_s}s (process terminated)",
        }
    else:
        try:
            rec = json.loads(tmp.read_text())
        except Exception as e:
            rec = {"instance": run["instance"], "finished": False,
                   "error": f"worker result unreadable: {e}"}
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    rec["run_wall_time_s"] = round(time.time() - t0, 2)
    return rec


def run_key(run: dict) -> str:
    return "|".join(
        str(x) for x in [run["instance"], run["map_family"], run["num_agv"],
                         run["combo"], run["seed"]]
    )


def load_done_keys(out: Path) -> set:
    keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "manifest" not in d:
                keys.add("|".join(str(x) for x in
                                  [d.get("instance"), d.get("map_family"),
                                   d.get("num_agv"), d.get("combo"), d.get("seed")]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="pilot", choices=["pilot", "full", "custom"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 (冒烟)")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--num-agvs", nargs="*", type=int, default=None)
    ap.add_argument("--combos", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="跳过输出文件中已存在的 episode (断点续跑)")
    ap.add_argument("--episode-timeout", type=float, default=300.0,
                    help="单 episode 墙钟超时(秒), 0 关闭")
    args = ap.parse_args()

    cfg = SuiteConfig()
    if args.suite == "full":
        from bench.suite import FULL_INSTANCES

        cfg.instances = list(FULL_INSTANCES)
        cfg.seeds = [42, 43, 44]
    if args.instances:
        cfg.instances = args.instances
    if args.num_agvs:
        cfg.num_agvs = args.num_agvs
    if args.combos:
        cfg.solver_combos = args.combos
    if args.seeds:
        cfg.seeds = args.seeds

    runs = list(cfg.iter_runs())
    if args.limit:
        runs = runs[: args.limit]

    out = Path(args.out) if args.out else ROOT / "results" / f"bench_{args.suite}_{int(time.time())}.jsonl"
    out.parent.mkdir(exist_ok=True)
    if args.resume:
        done = load_done_keys(out)
        n_before = len(runs)
        runs = [r for r in runs if run_key(r) not in done]
        print(f"[bench] resume: {n_before - len(runs)} done episodes skipped")
    manifest = {
        "git_sha": git_sha(),
        "suite": args.suite,
        "n_runs": len(runs),
        "config": {
            "instances": cfg.instances,
            "map_families": cfg.map_families,
            "num_agvs": cfg.num_agvs,
            "combos": cfg.solver_combos,
            "seeds": cfg.seeds,
            "max_steps": cfg.max_steps,
        },
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not (args.resume and out.exists()):
        out.write_text(json.dumps({"manifest": manifest}, ensure_ascii=False) + "\n")
    else:
        with open(out, "a") as f:
            f.write(json.dumps({"manifest_resumed": manifest}, ensure_ascii=False) + "\n")

    print(f"[bench] {len(runs)} runs -> {out}")
    t_start = time.time()
    n_err = 0
    for i, run in enumerate(runs):
        rec = episode_record(run, timeout_s=args.episode_timeout)
        if rec.get("error"):
            n_err += 1
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        mk = rec.get("makespan")
        status = "ERR:" + str(rec.get("error", ""))[:60] if rec.get("error") else f"{rec['run_wall_time_s']}s"
        print(f"[{i+1}/{len(runs)}] {run['instance']}|{run['map_family']}|agv{run['num_agv']}"
              f"|{run['combo']}|s{run['seed']} -> makespan={mk} ({status})", flush=True)
    print(f"[bench] done in {time.time()-t_start:.0f}s, errors={n_err}, out={out}")


if __name__ == "__main__":
    main()
