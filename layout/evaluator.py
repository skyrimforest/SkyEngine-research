"""管线一·布局真值仿真打分器 (1118layout)
============================================================
布局 × 订单批次 × AGV 起点 → 逐集仿真 → JSONL; 含聚合与冒烟。

用法 (容器内 /work/SkyEngine-research-llmdiag 下):
  python layout/evaluator.py --smoke                     # 确定性 + 聚集vs分散 敏感性冒烟
  python layout/evaluator.py --instance mk01 --layouts 8 --batches 2 --origins 2 \
      --out layout/results_pilot/pilot.jsonl             # 迷你试点
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layout.legality import free_cells, check_layout  # noqa: E402
from layout import sampler  # noqa: E402
from layout.batches import make_batch  # noqa: E402

MAP_FILE = ROOT / "data/mapf/medium_maps.yaml"
MAP_NAME = "medium-mazes-seed-0000"
RANDOM_MAP_FILE = ROOT / "data/mapf/random_maps.yaml"
RANDOM_MAP_NAME = "validation-random-seed-000"
KPI_KEYS = ["agv_waiting_time_total", "tasked_stationary_count",
            "transport_delay_ratio", "machine_utilization",
            "agv_busy_utilization", "machine_load_variance"]


CUR_MAP = [MAP_FILE, MAP_NAME]


def _run_one(args):
    """子进程: 单集仿真 (fork 隔离, 超时由父进程 join 控制)。"""
    (fjsp_path, machine_positions, agv_positions, num_agv, seed, max_steps, map_file, map_name) = args
    from common.engine_adapter import run_episode
    t0 = time.time()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        pass
    try:
        import contextlib as _cl, io as _io
        with _cl.redirect_stdout(_io.StringIO()):
            res = run_episode(
                fjsp_path=fjsp_path, map_file=map_file, map_name=map_name,
                num_agv=num_agv, seed=seed, max_steps=max_steps,
                job_solver="greedy", route_solver="astar",
                machine_positions=machine_positions,
                agv_positions=agv_positions,
            )
        s = res.get("summary", {}) or {}
        mk = s.get("completed_makespan") if res.get("finished") else max_steps
        lay = (res.get("config", {}) or {}).get("layout", {}) or {}
        row = {
            "makespan": mk, "finished": res.get("finished"),
            "steps": res.get("steps"), "wall_time_s": round(time.time() - t0, 2),
            "layout_hash": lay.get("layout_hash"),
            "agv_origin_hash": lay.get("agv_origin_hash"),
            "machine_positions": lay.get("machine_positions"),
            "agv_positions": lay.get("agv_positions"),
            **{k: s.get(k) for k in KPI_KEYS},
        }
        return row, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def run_grid(episodes, timeout=600.0):
    """顺序 fork 隔离执行 [(args, meta), ...] -> [row|error, ...]"""
    out = []
    ctx = mp.get_context("fork")
    for i, (args, meta) in enumerate(episodes):
        q = ctx.Queue()
        p = ctx.Process(target=_wrapper, args=(args, q))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join(5)
            if p.is_alive():
                p.kill(); p.join(5)
            row = {"error": f"hard timeout {timeout}s"}
        else:
            row = q.get() if not q.empty() else {"error": "no output"}
        out.append((row, meta))
        mk = row.get("makespan", row.get("error", "?"))
        print(f"[{i+1}/{len(episodes)}] L{meta['layout_id']} b{meta['batch_id']} "
              f"o{meta['origin_id']} -> mk={mk} ({row.get('wall_time_s','?')}s)", flush=True)
    return out


def _wrapper(args, q):
    row, err = _run_one(args)
    q.put(row if err is None else {"error": err})


def build_layouts(n_layouts, layout_seed, free, free_set, n_machines, n_agv):
    rng = random.Random(layout_seed)
    layouts = []
    for i in range(n_layouts):
        lay = sampler.random_layout(rng, free, free_set, n_machines, n_agv)
        ok, why = check_layout(free_set, lay["machine_positions"], lay["agv_positions"])
        if not ok:
            raise ValueError(f"采样布局非法: {why}")
        lay["layout_id"] = i
        layouts.append(lay)
    return layouts


def evaluate(instance, layouts, n_batches, batch_jobs, n_origins, num_agv,
             out_path, max_steps=4096, timeout=600.0, map_key="random"):
    set_map(map_key)
    free, free_set = free_cells(CUR_MAP[0], CUR_MAP[1])
    batch_files = [make_batch(instance, b, batch_jobs) for b in range(n_batches)]
    episodes = []
    for lay in layouts:
        machines = [tuple(p) for p in lay["machine_positions"]]
        mset = set(machines)
        for b, bf in enumerate(batch_files):
            for o in range(n_origins):
                orng = random.Random(700000 + o * 1009 + lay["layout_id"])
                rest = [p for p in free if p not in mset]
                agvs = orng.sample(rest, num_agv)
                meta = {"layout_id": lay["layout_id"],
                        "layout_hash": None, "batch_id": b, "origin_id": o}
                episodes.append((
                    (str(bf), machines, agvs, num_agv, 42 + o, max_steps,
                     str(CUR_MAP[0]), CUR_MAP[1]), meta))
    results = run_grid(episodes, timeout=timeout)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row, meta in results:
            rec = {**meta, "instance": instance, **row}
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"-> {out}")
    aggregate(out)


def set_map(map_key):
    if map_key == "random":
        CUR_MAP[0], CUR_MAP[1] = RANDOM_MAP_FILE, RANDOM_MAP_NAME
    else:
        CUR_MAP[0], CUR_MAP[1] = MAP_FILE, MAP_NAME


def aggregate(out_path):
    rows = [json.loads(l) for l in Path(out_path).read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "error" not in r]
    by_layout = {}
    for r in ok:
        by_layout.setdefault(r["layout_id"], []).append(r["makespan"])
    stats = {lid: {"mean": sum(v)/len(v), "min": min(v), "max": max(v), "n": len(v)}
             for lid, v in sorted(by_layout.items())}
    means = [s["mean"] for s in stats.values()]
    if means:
        gmean = sum(means) / len(means)
        spread = (max(means) - min(means)) / gmean * 100
        import math
        cv = (math.sqrt(sum((m-gmean)**2 for m in means)/len(means)) / gmean) * 100
        print(f"\n== 聚合 == 布局数 {len(stats)} | 全局均值 {gmean:.1f} | "
              f"最好 {min(means):.1f} 最差 {max(means):.1f} | "
              f"谱宽 {spread:.1f}% | CV {cv:.1f}%  (判停线: 谱宽<5% → 方向降级)")
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"错误集: {len(errs)} ({errs[0]['error'][:60]})")


def smoke(instance="mk01", num_agv=4, max_steps=4096, map_key="random"):
    """两道冒烟: 确定性(同布局同起点跑两遍逐位一致) + 敏感性(聚集 vs 分散)。"""
    set_map(map_key)
    free, free_set = free_cells(CUR_MAP[0], CUR_MAP[1])
    data = json.loads((ROOT / "data/fjsp_official/brandimarte" / f"{instance}.json").read_text())
    n_m = data["machines"]
    rng = random.Random(1115)
    lay_a = sampler.spread_layout(rng, free, free_set, n_m, num_agv)
    lay_b = sampler.cluster_layout(rng, free, free_set, n_m, num_agv)
    print(f"地图自由格 {len(free)} | 机器 {n_m} | AGV {num_agv}")
    eps = []
    for rep in range(2):  # 确定性: spread 布局跑两遍
        eps.append(((str(ROOT / f"data/fjsp_official/brandimarte/{instance}.json"),
                     [tuple(p) for p in lay_a["machine_positions"]],
                     [tuple(p) for p in lay_a["agv_positions"]], num_agv, 42, max_steps,
                     str(CUR_MAP[0]), CUR_MAP[1]),
                    {"layout_id": f"spread_r{rep}", "batch_id": 0, "origin_id": 0}))
    eps.append(((str(ROOT / f"data/fjsp_official/brandimarte/{instance}.json"),
                 [tuple(p) for p in lay_b["machine_positions"]],
                 [tuple(p) for p in lay_b["agv_positions"]], num_agv, 42, max_steps,
                 str(CUR_MAP[0]), CUR_MAP[1]),
                {"layout_id": "cluster", "batch_id": 0, "origin_id": 0}))
    res = run_grid(eps, timeout=900)
    mks = {meta["layout_id"]: row.get("makespan") for row, meta in res}
    print("\n== 冒烟1 确定性 ==", mks.get("spread_r0"), "vs", mks.get("spread_r1"),
          "->", "PASS 逐位一致" if mks.get("spread_r0") == mks.get("spread_r1") else "FAIL")
    print("== 冒烟2 敏感性 == spread:", mks.get("spread_r0"), "| cluster:", mks.get("cluster"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="mk01")
    ap.add_argument("--layouts", type=int, default=8)
    ap.add_argument("--layout-seed", type=int, default=1115)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch-jobs", type=int, default=6)
    ap.add_argument("--origins", type=int, default=2)
    ap.add_argument("--num-agv", type=int, default=4)
    ap.add_argument("--out", default="layout/results_pilot/pilot.jsonl")
    ap.add_argument("--map", default="random", choices=["random", "maze"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke(a.instance, a.num_agv, map_key=a.map)
        return
    set_map(a.map)
    free, free_set = free_cells(CUR_MAP[0], CUR_MAP[1])
    data = json.loads((ROOT / "data/fjsp_official/brandimarte" / f"{a.instance}.json").read_text())
    layouts = build_layouts(a.layouts, a.layout_seed, free, free_set,
                            data["machines"], a.num_agv)
    evaluate(a.instance, layouts, a.batches, a.batch_jobs, a.origins,
             a.num_agv, a.out, map_key=a.map)


if __name__ == "__main__":
    main()
