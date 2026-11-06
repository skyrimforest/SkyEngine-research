"""方向7 v2: G1.1 统计基线 (R4, 评审行动计划)
================================================
动机 (评审核对二 20260906): F4 基率忽视的对照物 —— 一个不用 LLM 的便宜统计基线:
在 KPI 时间序列上做结构断点检测 + 扰动签名映射, 回答"不用语言模型, 序列统计能诊断到什么程度"。

观测纪律: 只消费档案可见信息 (score 侧与 LLM 臂完全同判分):
  - metrics_series (run_closed_episode 新采, 步距 8)
  - 汇总 KPI (kpi), 修订账本 (revisions.count)
  不读 exception_config / ground_truth / 孪生 (注入时刻表不泄露)。

冻结决策树 (一次定死):
  1. 断点检测: 对吞吐类序列 (throughput_jobs / operation_queue_waiting_time_mean) 做
     CUSUM 式 argmax 断点 (两侧各 ≥4 样本), 得 s* 与退化幅度。
  2. 扰动签名: machine_down_steps_total / agv_down_steps_total 的增量分布
     (恢复平台期 = 离散增量; 持续多段 = 流)。
  3. 映射:
     机+AGV 均有增量        -> disruption_machine_agv
     仅机增量, 单段(≤3段)   -> disruption_machine
     仅机增量, 多段(>3段)   -> disruption_stochastic (若同时 revisions.count==0
                                且完工且 tardiness 高 -> plan_mismatch)
     未完工 & AGV总等待>2000 & stationary==0 -> starvation_livelock
     未完工 & stationary>0  -> blocking_livelock
     其余                   -> baseline

已知盲区 (即实验要量化的东西): loc_target 只能到类型级 (序列里没有机器号),
machine 类 loc_target 给 "machine" -> score_loc 的 0.5 部分分正是预期观测点。

用法 (容器内, 档案与 episode 就绪后):
  python3 llmdiag/v2/granger_baseline.py \
      --episodes episodes_v21 --scen scenarios_v21.jsonl --cases cases_v21.json \
      --out results_granger_v21.json
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

THROUGHPUT_KEYS = ("throughput_jobs", "operation_queue_waiting_time_mean")
MIN_SIDE = 4  # 断点两侧最少样本数


def _series(rec: dict, key: str) -> list:
    return [(p["step"], p[key]) for p in rec.get("metrics_series", [])
            if key in p and isinstance(p[key], (int, float))]


def detect_break(rec: dict) -> tuple[int | None, float]:
    """CUSUM 式断点: argmax |mean(after) - mean(before)| 归一化 (std), 两侧 ≥MIN_SIDE"""
    for key in THROUGHPUT_KEYS:
        s = _series(rec, key)
        if len(s) < MIN_SIDE * 2:
            continue
        vals = [v for _, v in s]
        std = max(1e-9, _std(vals))
        best, best_score = None, 0.0
        for i in range(MIN_SIDE, len(s) - MIN_SIDE):
            pre, post = vals[:i], vals[i:]
            score = abs(_mean(post) - _mean(pre)) / std
            if score > best_score:
                best_score, best = score, s[i][0]
        return best, best_score
    return None, 0.0


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5


def increments(series: list) -> list:
    """阶梯序列的增量步位置 (恢复平台期 = 离散跳变)"""
    steps = []
    for (s0, v0), (s1, v1) in zip(series, series[1:]):
        if v1 > v0:
            steps.append(s1)
    return steps


def diagnose(rec: dict) -> dict:
    s = rec.get("summary", {}) or {}
    md = increments(_series(rec, "machine_down_steps_total"))
    ad = increments(_series(rec, "agv_down_steps_total"))
    kpi_wait = s.get("agv_waiting_time_total") or 0
    stationary = s.get("tasked_stationary_count") or 0
    finished = rec.get("finished", True)
    rev = (rec.get("orchestrator_stats", {}) or {}).get("plan_revisions", None)
    # revisions.count 在档案构建时从 n_plan_revisions 取; episode 记录里是 n_plan_revisions
    n_rev = rec.get("n_plan_revisions")
    tardy = s.get("total_tardiness") or 0
    active = s.get("active_disruption_count") or 0

    ev, brk = [], detect_break(rec)

    def cite(*items):
        return list(items)

    # 决策树 (冻结; 顺序即优先级)
    if md and ad:
        pred = {"loc_type": "machine_agv", "loc_target": "machine", "cause": "disruption_machine_agv"}
        ev = cite(f"KPI:machine_down_steps_total={s.get('machine_down_steps_total')}",
                  f"KPI:agv_down_steps_total={s.get('agv_down_steps_total')}")
    elif md and len(md) > 3:
        # 多段机增量 = 故障流; 静态臂(零修订)+完工+高 tardy = 计划失配
        if (n_rev == 0) and finished and tardy > 0:
            pred = {"loc_type": "stochastic", "loc_target": "stochastic", "cause": "plan_mismatch"}
        else:
            pred = {"loc_type": "stochastic", "loc_target": "stochastic", "cause": "disruption_stochastic"}
        ev = cite(f"KPI:machine_down_steps_total={s.get('machine_down_steps_total')}",
                  f"KPI:active_disruption_count={active}")
    elif md:
        pred = {"loc_type": "machine", "loc_target": "machine", "cause": "disruption_machine"}
        ev = cite(f"KPI:machine_down_steps_total={s.get('machine_down_steps_total')}")
        if brk[0] is not None:
            ev.append(f"KPI:operation_queue_waiting_time_mean={s.get('operation_queue_waiting_time_mean')}")
    elif (not finished) and kpi_wait > 2000 and not stationary:
        pred = {"loc_type": "task_pool", "loc_target": "task_pool", "cause": "starvation_livelock"}
        ev = cite(f"KPI:agv_waiting_time_total={kpi_wait}", "KPI:tasked_stationary_count=0")
    elif (not finished) and stationary > 0:
        pred = {"loc_type": "corridor", "loc_target": "corridor", "cause": "blocking_livelock"}
        ev = cite(f"KPI:tasked_stationary_count={stationary}")
    else:
        pred = {"loc_type": "none", "loc_target": "none", "cause": "baseline"}
        ev = cite(f"KPI:makespan={s.get('completed_makespan', rec.get('makespan'))}")
    pred["evidence"] = ev
    pred["intervention"] = "none"  # 冻结: 统计基线不开处方
    pred["narrative"] = "统计基线 (G1.1): 序列签名映射, 无叙述。"
    pred["_break_step"] = brk[0]
    pred["_break_score"] = round(brk[1], 2)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="episodes_v21")
    ap.add_argument("--scen", default="scenarios_v21.jsonl")
    ap.add_argument("--cases", default="cases_v21.json",
                    help="档案 (判分用); 不存在则只出诊断不判分")
    ap.add_argument("--out", default="results_granger_v21.json")
    args = ap.parse_args()

    rows = []
    for f in sorted((HERE / args.episodes).glob("*.json")):
        rec = json.loads(f.read_text())
        if "error" in rec or not rec.get("metrics_series"):
            continue
        pred = diagnose(rec)
        rows.append({"case_id": f"{f.stem}|easy", "scen_id": f.stem,
                     "raw": pred, **{k: pred[k] for k in
                                     ("loc_type", "loc_target", "cause")}})
    (HERE / args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"{len(rows)} 条诊断 -> {args.out}")

    # 判分 (档案存在时)
    cf = HERE / args.cases
    if cf.exists():
        from score import score_one, aggregate
        cases = {c["case_id"]: c for c in json.loads(cf.read_text())}
        scored = []
        for r in rows:
            c = cases.get(r["case_id"])
            if c is None:
                continue
            scored.append(dict(score_one(r["raw"], c), case_id=r["case_id"],
                               gt=c["ground_truth"]))
        agg = aggregate(scored)
        print(json.dumps(agg, ensure_ascii=False))
        (HERE / args.out).with_name(args.out.replace(".json", "_scored.json")) \
            .write_text(json.dumps(scored, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
