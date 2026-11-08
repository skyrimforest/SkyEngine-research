"""案例页真实数据图: KPI 趋势(故障vs孪生) / 故障事件时间线 / 真实迷宫地图"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figs"
FIG.mkdir(exist_ok=True)
PRIMARY, BLUE, GREY = "#6E0F6D", "#024282", "#A5A5A5"

ep = json.loads((HERE / "episodes_v21/v21_mismatch_02.json").read_text())
twin = json.loads((HERE / "episodes_v21/v21_mismatch_02_T.json").read_text())

def series(rec, key):
    return [(p["step"], p[key]) for p in rec["metrics_series"] if key in p]

# ---------- 图1: 完工进度 孪生 vs 故障 ----------
fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=200)
t_s, t_y = zip(*series(twin, "throughput_jobs"))
f_s, f_y = zip(*series(ep, "throughput_jobs"))
ax.plot(t_s, t_y, color=BLUE, lw=2.2, label=f"No-fault twin  (makespan {twin['makespan']})")
ax.plot(f_s, f_y, color=PRIMARY, lw=2.2, label=f"Faulty run  (makespan {ep['makespan']})")
# 故障期阴影: machine_down_steps_total 增量区间
downs = series(ep, "machine_down_steps_total")
prev_step, prev_val = None, 0
for st, v in downs:
    if prev_step is not None and v > prev_val:
        ax.axvspan(prev_step, st, color=PRIMARY, alpha=0.10, lw=0)
    prev_step, prev_val = st, v
ax.set_xlabel("simulation step", fontsize=11)
ax.set_ylabel("jobs completed", fontsize=11)
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E8E8E8", lw=0.6)
ax.set_axisbelow(True)
ax.set_xlim(0, 1560)
ax.annotate(f"fault cost = {ep['makespan'] - twin['makespan']} steps (+{(ep['makespan']/twin['makespan']-1)*100:.1f}%)",
            xy=(0.985, 0.06), xycoords="axes fraction", ha="right", fontsize=11, color=PRIMARY, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig_kpi.png", bbox_inches="tight")
plt.close(fig)

# ---------- 图2: 故障事件时间线 ----------
evs = [e for e in ep["events_sample"] if e["type"] == "machine_breakdown"]
fig, ax = plt.subplots(figsize=(9.6, 3.2), dpi=200)
mids = sorted({e["payload"]["machine_id"] for e in evs})
hl = {6, 7}  # LLM 引用的机器
for e in evs[:80]:
    m = e["payload"]["machine_id"]
    y = mids.index(m)
    col = PRIMARY if m in hl else GREY
    ax.plot([e["step"], e["step"] + e["payload"]["duration_steps"]], [y, y],
            color=col, lw=5 if m in hl else 3, alpha=0.9 if m in hl else 0.55,
            solid_capstyle="butt")
ax.set_yticks(range(len(mids)))
ax.set_yticklabels([f"machine {m}" + ("  ← cited by LLM" if m in hl else "") for m in mids], fontsize=9)
ax.set_xlabel("simulation step  (bar = breakdown duration)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#E8E8E8", lw=0.6)
ax.set_axisbelow(True)
ax.set_title(f"machine_breakdown events (first 80 of {len(evs)}), probabilistic fault stream", fontsize=10, color="#555")
fig.tight_layout()
fig.savefig(FIG / "fig_events.png", bbox_inches="tight")
plt.close(fig)

# ---------- 图3: 真实迷宫地图 ----------
import yaml
maps = yaml.safe_load(open("/work/SkyEngine-research-llmdiag/data/mapf/medium_maps.yaml"))
amap = maps["medium-mazes-seed-0000"].strip("\n").split("\n")
grid = np.array([[0 if ch in ".G" else 1 for ch in row] for row in amap])
fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=200)
ax.imshow(grid, cmap="Greys", interpolation="nearest")
ax.set_title("real map: medium-mazes-seed-0000\n(37% obstacle maze, from dataset)", fontsize=9, color="#555")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("#CCC")
fig.tight_layout()
fig.savefig(FIG / "fig_map.png", bbox_inches="tight")
plt.close(fig)

print("figs saved:", [p.name for p in FIG.glob("*.png")])
