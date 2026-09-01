"""方向3: PoD (分解代价) 计算 — 从 decomp ablation JSONL 生成分解表。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    """-> {(instance, num_agv): {arm: makespan}}"""
    data = defaultdict(dict)
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if "manifest" in d or "n_runs" in d or d.get("error") or not d.get("finished"):
            continue
        data[(d["instance"], d["num_agv"])][d["arm"]] = d["makespan"]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    src = Path(args.jsonl)
    out = Path(args.out) if args.out else src.parent / (src.stem + "_pod.md")

    data = load(src)
    lines = ["# 分解代价 PoD 分解表", "",
             "| inst | K | PoD_total | PoD_sched | PoD_route | 交互项 | 反馈价值 |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for (inst, k), arms in sorted(data.items()):
        def C(arm):
            return arms.get(arm)

        c11, c12, c21, c22 = C("greedy+astar"), C("greedy+eecbs"), C("cpsat+astar"), C("cpsat+eecbs")
        nofb = C("greedy+astar-nofb")
        if not (c11 and c22):
            continue
        pod_total = (c11 - c22) / c22
        pod_sched = (c21 - c22) / c22 if c21 is not None else None
        pod_route = (c12 - c22) / c22 if c12 is not None else None
        inter = None
        if pod_sched is not None and pod_route is not None:
            inter = pod_total - pod_sched - pod_route
        fb = (nofb - c11) / c11 if nofb else None

        def f(x):
            return f"{x*100:+.0f}%" if x is not None else "—"

        lines.append(f"| {inst} | {k} | {f(pod_total)} | {f(pod_sched)} | {f(pod_route)} | {f(inter)} | {f(fb)} |")
        rows.append(dict(inst=inst, K=k, pod_total=pod_total, pod_sched=pod_sched,
                         pod_route=pod_route, inter=inter, fb=fb))

    # K 聚合
    by_k = defaultdict(list)
    for r in rows:
        by_k[r["K"]].append(r)
    lines += ["", "# 按 K 聚合 (均值)", "", "| K | PoD_total | PoD_sched | PoD_route | 交互 | n |",
              "|---|---|---|---|---|---|"]
    for k, rs in sorted(by_k.items()):
        def m(key):
            vals = [r[key] for r in rs if r[key] is not None]
            return f"{sum(vals)/len(vals)*100:+.0f}%" if vals else "—"

        lines.append(f"| {k} | {m('pod_total')} | {m('pod_sched')} | {m('pod_route')} | {m('inter')} | {len(rs)} |")

    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print("\n->", out)


if __name__ == "__main__":
    main()
