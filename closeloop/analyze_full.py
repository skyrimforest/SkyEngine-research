"""
方向1 ICAPS 战役结果分析
=========================
输入: results/icaps_E{1,2,3,4}.jsonl
输出: closeloop/results_icaps/ 下的 markdown 表 + CSV

分析内容:
  T1 主表: makespan (scenario x policy) 均值±std, 含删失计数 (timeout/livelock)
  T2 扰动遗憾: RRegret = mk(S)/mk(S0) 按 (policy, scen)
  T3 遗留承诺对照: 修订成功率 legacy vs soft
  T4 罚金曲线: makespan/revisions vs travel_allowance
  S1 统计检验: 策略两两配对 Wilcoxon 符号秩 (纯python实现, 无scipy依赖)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "closeloop" / "results_icaps"


def load(phase: str) -> list[dict]:
    f = ROOT / "results" / f"icaps_{phase}.jsonl"
    if not f.exists():
        return []
    rows = []
    for line in f.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def censored(r: dict) -> bool:
    return bool(r.get("error")) or not r.get("finished")


def fmt_ms(vals: list) -> str:
    if not vals:
        return "—"
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.0f}±{s:.0f}"


def wilcoxon_signed(pairs: list) -> tuple:
    """配对 Wilcoxon 符号秩检验 (n<25 时用精确正态近似已足够论文级展示)。

    pairs: [(a_i, b_i)], 检验 a-b 的中位数是否为 0。
    返回 (W, z, p_two_sided)。纯 python 实现。
    """
    import math

    diffs = [a - b for a, b in pairs if a != b]
    n = len(diffs)
    if n < 5:
        return (None, None, None)
    ranked = sorted((abs(d), i) for i, d in enumerate(diffs))
    ranks = [0.0] * n
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ranked[k][1]] = avg
        i = j
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma if sigma else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (w_plus, z, p)


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    e1, e2 = load("E1"), load("E2")
    e3, e4 = load("E3"), load("E4")
    main_rows = e1 + e2

    # ---- T1 主表 ----
    lines = ["# T1 主表: makespan (scenario x policy)", "",
             "| scen | policy | mk (mean±std) | n | censored(timeout/未完工) |",
             "|---|---|---|---|---|"]
    groups = defaultdict(list)
    cens = defaultdict(int)
    total = defaultdict(int)
    for r in main_rows:
        key = (r.get("scen"), r.get("policy"))
        total[key] += 1
        if censored(r):
            cens[key] += 1
        elif r.get("makespan"):
            groups[key].append(r["makespan"])
    for scen in ["S0", "S1", "S4", "S3"]:
        for pol in ["greedy-reactive", "cpsat-static", "cpsat-full", "cpsat-partial"]:
            k = (scen, pol)
            lines.append(f"| {scen} | {pol} | {fmt_ms(groups.get(k, []))} "
                         f"| {total.get(k, 0)} | {cens.get(k, 0)} |")
    (OUT / "T1_main.md").write_text("\n".join(lines))

    # ---- T2 扰动遗憾 ----
    base = {}
    for r in main_rows:
        if r.get("scen") == "S0" and not censored(r) and r.get("makespan"):
            key = (r.get("instance"), r.get("map"), r.get("num_agv"),
                   r.get("policy"), r.get("seed"))
            base[key] = r["makespan"]
    regret = defaultdict(list)
    for r in main_rows:
        if r.get("scen") == "S0" or censored(r) or not r.get("makespan"):
            continue
        key = (r.get("instance"), r.get("map"), r.get("num_agv"),
               r.get("policy"), r.get("seed"))
        if key in base and base[key]:
            regret[(r["scen"], r["policy"])].append(r["makespan"] / base[key])
    lines = ["# T2 扰动遗憾 RRegret = mk(S)/mk(S0)", "",
             "| scen | policy | RRegret (mean±std) | n |", "|---|---|---|---|"]
    for scen in ["S1", "S4", "S3"]:
        for pol in ["greedy-reactive", "cpsat-static", "cpsat-full", "cpsat-partial"]:
            v = regret.get((scen, pol), [])
            lines.append(f"| {scen} | {pol} | {fmt_ms(v) if not v else f'{mean(v):.3f}±{(stdev(v) if len(v)>1 else 0):.3f}'} | {len(v)} |")
    (OUT / "T2_regret.md").write_text("\n".join(lines))

    # ---- T3 遗留 vs 软承诺 ----
    lines = ["# T3 承诺语义对照 (修订激活)", "",
             "| mode | policy | episodes | revisions(ok) | fails | 成功率 |",
             "|---|---|---|---|---|---|"]
    soft_rows = [r for r in e1 + e2 if r.get("policy") in ("cpsat-full", "cpsat-partial")]
    for mode, rows in [("soft", soft_rows), ("legacy", e3)]:
        agg = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            k = r.get("policy")
            agg[k][0] += 1
            agg[k][1] += r.get("revisions", 0) or 0
            agg[k][2] += r.get("revision_fails", 0) or 0
        for pol, (n, ok, fail) in sorted(agg.items()):
            rate = f"{ok/(ok+fail)*100:.0f}%" if (ok + fail) else "—"
            lines.append(f"| {mode} | {pol} | {n} | {ok} | {fail} | {rate} |")
    (OUT / "T3_commitment.md").write_text("\n".join(lines))

    # ---- T4 罚金曲线 ----
    lines = ["# T4 罚金参数扫描 (travel_allowance)", "",
             "| allowance | scen | mk (mean±std) | revisions(ep) | n |",
             "|---|---|---|---|---|"]
    g4 = defaultdict(list)
    r4 = defaultdict(list)
    for r in e4:
        if censored(r) or not r.get("makespan"):
            continue
        k = (r.get("allowance"), r.get("scen"))
        g4[k].append(r["makespan"])
        r4[k].append(r.get("revisions", 0) or 0)
    for allowance in [50, 100, 200, 400]:
        for scen in ["S1", "S4"]:
            k = (allowance, scen)
            rev = mean(r4[k]) if r4[k] else 0
            lines.append(f"| {allowance} | {scen} | {fmt_ms(g4.get(k, []))} "
                         f"| {rev:.1f} | {len(g4.get(k, []))} |")
    (OUT / "T4_penalty.md").write_text("\n".join(lines))

    # ---- S1 配对检验: soft-full vs static (S1/S4/S3) ----
    lines = ["# S1 配对 Wilcoxon 检验: cpsat-full vs cpsat-static (soft=on)", "",
             "| scen | n_pairs | W+ | z | p |", "|---|---|---|---|---|"]
    for scen in ["S1", "S4", "S3"]:
        pairs = []
        lookup = {}
        for r in main_rows:
            if censored(r) or not r.get("makespan") or r.get("scen") != scen:
                continue
            key = (r.get("instance"), r.get("map"), r.get("num_agv"), r.get("seed"))
            lookup[(key, r.get("policy"))] = r["makespan"]
        for key in {k for (k, p) in lookup if p == "cpsat-static"}:
            a = lookup.get((key, "cpsat-full"))
            b = lookup.get((key, "cpsat-static"))
            if a and b:
                pairs.append((a, b))
        w, z, p = wilcoxon_signed(pairs)
        lines.append(f"| {scen} | {len(pairs)} | {w} | "
                     f"{f'{z:.2f}' if z is not None else '—'} | "
                     f"{f'{p:.4f}' if p is not None else '—'} |")
    (OUT / "S1_tests.md").write_text("\n".join(lines))

    for f in sorted(OUT.glob("*.md")):
        print(f"\n===== {f.name} =====")
        print(f.read_text())
    print("\n->", OUT)


if __name__ == "__main__":
    main()
