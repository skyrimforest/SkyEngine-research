"""
左右结构可视化帧生成器
======================
左: 当前 job 的选择情况 —— 作业-工序甘特图
    (已完成=实心绿 / 加工中=蓝 / 机器口排队=琥珀 / 计划中=灰描边+计划机器号;
     红色带 = 机器故障区间; 竖线 = 当前时刻)
右: 小车运行情况 —— 引擎内生接口 draw_svg_with_machines_and_targets
    (栅格地图 + 障碍 + 机器节点 + AGV 轨迹/位置 + 当前激活目标高亮)

场景: 论文标准场景 mk01 x 迷宫 x agv4 x S1(机器0@150 坏40步) x soft-commit 修订
输出: results/vis/frame_*.png (左右合成) + rollout.gif
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FJSP_SOFT_COMMIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from common.engine_adapter import build_configs, load_fjsp_json, load_map
from closeloop.orchestrator import RecoveryOrchestrator, run_closed_episode  # noqa: F401
from sky_executor.grid_factory.factory.Component.Coordinator.coordinator import Coordinator
from sky_executor.grid_factory.factory.Utils.pic_drawer import (
    draw_svg_with_machines_and_targets,
)
from sky_executor.grid_factory.factory.grid_factory_env import GridFactoryEnv

OUT = ROOT / "results" / "vis"
OUT.mkdir(parents=True, exist_ok=True)

CAPTURES = [1, 149, 155, 260, 400, 10**9]  # 最后一项=终点
SCEN = {"enabled": True, "random_seed": 42,
        "schedule": [{"step": 150, "type": "machine_breakdown",
                      "machine_id": 0, "duration_steps": 40}]}

# ---- 颜色 ----
C_DONE, C_PROC, C_QUEUE, C_PLAN = "#2E8B57", "#1E6FA8", "#E8A13C", "#B9C4CC"
STATUS = {"FINISHED": "done", "PROCESSING": "proc", "SUSPENDED": "proc"}


def run_and_collect():
    """跑标准场景, 逐步记录: 每道工序的状态跃迁 + 机器故障区间 + 计划(审计)。"""
    fjsp = load_fjsp_json(ROOT / "data/fjsp_official/brandimarte/mk01.json")
    gmap = load_map(ROOT / "data/mapf/medium_maps.yaml", "medium-mazes-seed-0000")
    gc, mc, jc = build_configs(fjsp, gmap, num_agv=4, seed=42,
                               max_episode_steps=2048,
                               observation_type="MAPF", collision_system="soft")
    env = GridFactoryEnv(grid_config=gc, machine_config=mc, job_config=jc,
                         random_target=False, exception_config=SCEN)
    obs, _ = env.reset()
    coord = Coordinator(
        job_solver="online_fjsp",
        route_solver="rolling_mapf_http",
        assigner="nearest",
        job_solver_kwargs={"service_url": "http://fjsp:8002", "algorithm": "cp_sat",
                           "config": {"time_limit": 10.0, "num_workers": 1, "seed": 42,
                                      "soft_commitments": True,
                                      "soft_commitment_travel_allowance": 200}},
        route_solver_kwargs={"service_url": "http://mapf:8001",
                             "time_limit_ms": 500, "lns_init_algo": "EECBS",
                             "planning_horizon": 10, "execution_window": 5},
    )
    orch = RecoveryOrchestrator(coord, env, trigger="event", scope="full")

    n_jobs, n_ops = len(fjsp["jobs"]), sum(len(j) for j in fjsp["jobs"])
    trans = []          # (t, job, op, status_kind, machine_hint)
    downs = []          # (t_start, t_end, machine)
    audit_fn = getattr(coord.job_solver, "get_schedule_audit", None)

    def snapshot(t):
        penv = env.pogema_env
        qm = {}
        for m in penv.machines:
            for o in getattr(m, "input_queue", []) or []:
                qm[(o.job_id, o.op_id)] = m.id
            if getattr(m, "current_op", None) is not None:
                co = m.current_op
                qm[(co.job_id, co.op_id)] = m.id
        audit = audit_fn() if audit_fn else {}
        assign = audit.get("machine_assignments", {}) if audit else {}
        plan_start = audit.get("planned_start_times", {}) if audit else {}
        for job in penv.jobs:
            for o in job.ops:
                k = (int(job.job_id), int(o.op_id))
                st = STATUS.get(str(getattr(o, "status", "PENDING")),
                                "queue" if k in qm else "pending")
                if st == "pending":
                    mid = assign.get(f"{k[0]}:{k[1]}")
                    ps = plan_start.get(f"{k[0]}:{k[1]}")
                else:
                    mid, ps = qm.get(k), None
                trans.append(dict(t=t, job=k[0], op=k[1], st=st,
                                  m=(int(mid) if mid is not None else None),
                                  ps=(float(ps) if ps is not None else None)))

    t, done = 0, False
    prev_down = {}
    for i in range(2048):
        actions = coord.decide(obs)
        obs, _r, term, trunc, _i = env.step(actions)
        t = i + 1
        penv = env.pogema_env
        for m in penv.machines:
            rem = int(getattr(m, "repair_remaining", 0) or 0)
            if m.id not in prev_down and rem > 0:
                prev_down[m.id] = t
            elif m.id in prev_down and rem == 0:
                downs.append((prev_down.pop(m.id), t, m.id))
        if i + 1 in CAPTURES[:-1]:
            snapshot(i + 1)
        orch.on_step(obs, i + 1)
        if term.get("job_done"):
            done = True
            break
    snapshot(t)
    for mid, ts in prev_down.items():
        downs.append((ts, t, mid))
    return env, trans, downs, t, done


def op_intervals(trans):
    """由逐帧快照推出每道工序的时间区间: kind 变更即区间边界。"""
    by_op = {}
    for r in trans:
        by_op.setdefault((r["job"], r["op"]), []).append(r)
    out = {}
    for k, rows in by_op.items():
        rows.sort(key=lambda r: r["t"])
        segs = []
        cur = None
        for r in rows:
            if cur and cur["st"] != r["st"]:
                segs.append(cur); cur = None
            if cur is None:
                cur = dict(st=r["st"], t0=r["t"], t1=r["t"], m=r["m"], ps=r["ps"])
            else:
                cur["t1"] = r["t"]; cur["m"] = r["m"] if r["m"] is not None else cur["m"]
        if cur:
            segs.append(cur)
        out[k] = segs
    return out


def left_panel(intervals, n_jobs, downs, t_now, total, path):
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots(figsize=(7.8, 6.6), dpi=130)
    h = 0.36
    labeled_jobs = set()
    for (j, o), segs in intervals.items():
        # 偶数工序上细行 / 奇数工序下细行: 流水式重叠(前序加工中, 后序已到机器口排队)天然分row
        y = (n_jobs - 1 - j) + (0.22 if o % 2 == 0 else -0.22)
        if j not in labeled_jobs:
            labeled_jobs.add(j)
            ax.text(-12, n_jobs - 1 - j, f"J{j}", ha="right", va="center",
                    fontsize=9.5, color="#334155", fontweight="bold")
        for s in segs:
            x0, x1 = s["t0"], min(s["t1"], t_now)
            if x1 <= x0:
                x1 = x0 + max(0.4, (t_now - x0) * 0.02)
            if s["st"] == "done":
                ax.add_patch(mpatches.Rectangle((x0, y - h / 2), x1 - x0, h,
                                                facecolor=C_DONE, edgecolor="none"))
            elif s["st"] == "proc":
                ax.add_patch(mpatches.Rectangle((x0, y - h / 2), x1 - x0, h,
                                                facecolor=C_PROC, edgecolor="none"))
            elif s["st"] == "queue":
                ax.add_patch(mpatches.Rectangle((x0, y - h / 2), x1 - x0, h,
                                                facecolor=C_QUEUE, edgecolor="none"))
            else:  # pending: 计划开始(虚线框, 审计数据)
                ps = s["ps"] if s["ps"] is not None else x0
                ax.add_patch(mpatches.Rectangle((ps, y - h / 2), 26, h,
                                                facecolor="none", edgecolor=C_PLAN,
                                                linestyle="--", linewidth=1.1))
            # 机器号竖排嵌在色块内(单字符宽, 不可能与相邻标签压盖); 图例说明数字=机器id
            if (s["st"] in ("proc", "queue") and s["m"] is not None
                    and (x1 - x0) > 24):
                boxc = "#1E6FA8" if s["st"] == "proc" else "#E8A13C"
                ax.text((x0 + x1) / 2, y, f"{s['m']}", ha="center", va="center",
                        fontsize=6,
                        color="white" if s["st"] == "proc" else "#3B2B12",
                        fontweight="bold", rotation=90,
                        path_effects=[pe.withStroke(linewidth=2.0,
                                                    foreground=boxc)])
    for (d0, d1, mid) in downs:
        ax.axvspan(d0, min(d1, t_now), color="#E11D48", alpha=0.28, zorder=0)
        if t_now > d0:
            ax.text((d0 + min(d1, t_now)) / 2, n_jobs - 0.45,
                    f"M{mid} DOWN", ha="center", fontsize=8.5,
                    color="#BE123C", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.axvline(t_now, color="#0F1E2E", linewidth=1.4)
    ax.set_xlim(0, max(total, t_now) * 1.02)
    ax.set_ylim(-0.95, n_jobs + 0.15)
    ax.set_yticks([])
    ax.set_xlabel("time (steps)", fontsize=10)
    ax.set_title(f"Job-Operation Status  @t={t_now}", fontsize=12, color="#0B3D4D",
                 fontweight="bold", loc="left")
    handles = [mpatches.Patch(color=C_DONE, label="finished"),
               mpatches.Patch(color=C_PROC, label="processing (digit = machine id)"),
               mpatches.Patch(color=C_QUEUE, label="queued at machine (digit = id)"),
               mpatches.Patch(facecolor="none", edgecolor=C_PLAN, linestyle="--",
                              label="planned (audit)"),
               mpatches.Patch(facecolor="#E11D48", alpha=0.28, label="machine down"),
               plt.Line2D([0], [0], color="#0F1E2E", linewidth=1.4,
                          label="current time")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              fontsize=8, framealpha=0.9, ncol=3)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    # QA: 文本包围盒两两求交, 任何重叠都打印出来(不依赖人眼)
    fig.canvas.draw()
    tb = [(t.get_text(), t.get_window_extent()) for t in ax.texts]
    bad = [(a[0], b[0]) for i, a in enumerate(tb) for b in tb[i + 1:]
           if a[1].overlaps(b[1])]
    if bad:
        print(f"[left_panel] TEXT OVERLAP @t={t_now}: {bad[:8]} "
              f"(total {len(bad)})")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def right_panel(env, t, path):
    svg = draw_svg_with_machines_and_targets(env.pogema_env, int(t))
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(path),
                     output_width=832, output_height=832)


def compose(l, r, out):
    from PIL import Image
    a, b = Image.open(l), Image.open(r)
    h = max(a.height, b.height)
    def scale(im):
        w = int(im.width * h / im.height)
        return im.resize((w, h))
    a, b = scale(a), scale(b)
    gap = 24
    canvas = Image.new("RGB", (a.width + gap + b.width, h), "white")
    canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + gap, 0))
    canvas.save(out)


def main():
    env, trans, downs, total, done = run_and_collect()
    print(f"episode done={done} makespan={total} downs={downs}")
    intervals = op_intervals(trans)
    n_jobs = len(env.pogema_env.jobs)
    frames = []
    tmp = OUT / ".tmp"
    tmp.mkdir(exist_ok=True)
    for t in CAPTURES:
        tt = min(t, total)
        lp = tmp / f"l_{tt}.png"; rp = tmp / f"r_{tt}.png"
        left_panel(intervals, n_jobs, downs, tt, total, lp)
        right_panel(env, tt, rp)
        fp = OUT / f"frame_t{tt:04d}.png"
        compose(lp, rp, fp)
        frames.append(fp)
        print("frame ->", fp.name)
    from PIL import Image
    ims = [Image.open(f) for f in frames]
    ims[0].save(OUT / "rollout.gif", save_all=True, append_images=ims[1:],
                duration=1100, loop=0)
    print("gif ->", OUT / "rollout.gif")


if __name__ == "__main__":
    main()
