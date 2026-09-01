"""
方向2: 基准结果分析 — JSONL -> CSV + Markdown 汇总表 + 图
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEY_METRICS = [
    "makespan",
    "transport_overhead",
    "agv_busy",
    "agv_loaded",
    "empty_pickup_mean",
    "blocking_delay_mean",
    "queue_wait_mean",
    "wall_time_s",
]


def load_records(path: Path) -> list[dict]:
    recs = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if "manifest" in d:
            continue
        recs.append(d)
    return recs


def agg_table(recs: list[dict], group_keys: list[str], metric: str) -> dict:
    groups = defaultdict(list)
    for r in recs:
        if r.get("error"):
            continue
        key = tuple(r.get(k) for k in group_keys)
        v = r.get(metric)
        if isinstance(v, (int, float)):
            groups[key].append(float(v))
    return {
        k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}
        for k, v in groups.items()
    }


def fmt(x):
    return f"{x:.1f}" if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    src = Path(args.jsonl)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent / (src.stem + "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(src)
    n_err = sum(1 for r in recs if r.get("error"))
    print(f"records={len(recs)} errors={n_err}")

    # 表1: combo 主表 (全实例/地图/AGV 平均)
    lines = ["# 基准汇总 (按求解器组合)", "",
             "| combo | makespan(mean/min) | overhead | agv_busy | agv_loaded | empty_pick | blocking | n |",
             "|---|---|---|---|---|---|---|---|"]
    t = agg_table(recs, ["combo"], "makespan")
    oh = agg_table(recs, ["combo"], "transport_overhead")
    ab = agg_table(recs, ["combo"], "agv_busy")
    al = agg_table(recs, ["combo"], "agv_loaded")
    ep = agg_table(recs, ["combo"], "empty_pickup_mean")
    bd = agg_table(recs, ["combo"], "blocking_delay_mean")
    for combo in sorted(t):
        r = t[combo]
        lines.append(
            f"| {combo[0]} | {r['mean']:.0f}/{r['min']:.0f} "
            f"| {fmt(oh.get(combo, {}).get('mean'))} "
            f"| {fmt(ab.get(combo, {}).get('mean'))} "
            f"| {fmt(al.get(combo, {}).get('mean'))} "
            f"| {fmt(ep.get(combo, {}).get('mean'))} "
            f"| {fmt(bd.get(combo, {}).get('mean'))} | {r['n']} |"
        )
    (out_dir / "table_combo.md").write_text("\n".join(lines))

    # 表2: AGV 密度 x combo 的 makespan
    lines = ["# Makespan x AGV 密度 (按 combo)", ""]
    t = agg_table(recs, ["combo", "num_agv"], "makespan")
    combos = sorted({k[0] for k in t})
    agvs = sorted({k[1] for k in t})
    lines.append("| combo | " + " | ".join(f"agv={a}" for a in agvs) + " |")
    lines.append("|---" * (len(agvs) + 1) + "|")
    for c in combos:
        row = [c]
        for a in agvs:
            row.append(fmt(t.get((c, a), {}).get("mean")))
        lines.append("| " + " | ".join(row) + " |")
    (out_dir / "table_agv.md").write_text("\n".join(lines))

    # 表3: 地图族 x combo
    t = agg_table(recs, ["combo", "map_family"], "makespan")
    lines = ["# Makespan x 地图族 (按 combo)", "", "| combo | maze | random |", "|---|---|---|"]
    for c in sorted({k[0] for k in t}):
        lines.append(
            f"| {c} | {fmt(t.get((c, 'maze'), {}).get('mean'))} "
            f"| {fmt(t.get((c, 'random'), {}).get('mean'))} |"
        )
    (out_dir / "table_map.md").write_text("\n".join(lines))

    # 图: makespan vs agv per combo; overhead vs agv
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = agg_table(recs, ["combo", "num_agv"], "makespan")
        combos = sorted({k[0] for k in t})
        agvs = sorted({k[1] for k in t})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for c in combos:
            ys = [t.get((c, a), {}).get("mean") for a in agvs]
            axes[0].plot(agvs, ys, marker="o", label=c)
        axes[0].set_xlabel("num AGVs"); axes[0].set_ylabel("makespan (steps)")
        axes[0].set_title("Makespan vs AGV fleet size"); axes[0].legend(fontsize=7)

        oh = agg_table(recs, ["combo", "num_agv"], "transport_overhead")
        for c in combos:
            ys = [oh.get((c, a), {}).get("mean") for a in agvs]
            axes[1].plot(agvs, ys, marker="s", label=c)
        axes[1].set_xlabel("num AGVs"); axes[1].set_ylabel("C_int / C*_FJSP")
        axes[1].set_title("Transport overhead ratio vs fleet size"); axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_makespan_agv.png", dpi=150)
        print("figures ->", out_dir / "fig_makespan_agv.png")
    except Exception as e:
        print("plot skipped:", e)

    for f in ["table_combo.md", "table_agv.md", "table_map.md"]:
        print(f"--- {f} ---")
        print((out_dir / f).read_text())
    print("analysis dir:", out_dir)


if __name__ == "__main__":
    main()
