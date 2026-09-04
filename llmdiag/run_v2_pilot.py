"""方向7 v2: 出题-判卷闭环编排 (方法论 v0.1 §7 工程件 ①②④ + 场景生成器)
==========================================================================
阶段 (全部可断点续跑, 工件落 llmdiag/results_v2/):
  sample    场景网格 -> specs.json
  run       逐案例跑插桩 episode (子进程隔离+硬超时) -> episodes.jsonl
  verify    核验器标注 (阅卷人) + ground truth -> cases.json
  archive   easy/hard/INT 三变体档案 -> archives.json
  baseline  规则基线 easy/hard 双档诊断+判分 -> baseline_grades_*.json
  llm       LLM 臂 (需 LLM_BASE_URL/LLM_API_KEY/LLM_MODEL) -> llm_grades_*.json
  report    汇总 -> report.md

用法:
  python llmdiag/run_v2_pilot.py --stage all --per-class 3
  python llmdiag/run_v2_pilot.py --stage run --per-class 3     # 断点续跑
  python llmdiag/run_v2_pilot.py --stage llm --variant hard    # LLM hard 档
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "llmdiag" / "results_v2"
EPISODE_TIMEOUT_S = 900.0


def stage_sample(args):
    from llmdiag.scenarios_v2 import sample_grid
    specs = sample_grid(per_class=args.per_class,
                        master_seed=args.master_seed)
    OUT.mkdir(exist_ok=True)
    (OUT / "specs.json").write_text(
        json.dumps([s.to_dict() for s in specs], ensure_ascii=False, indent=1))
    from collections import Counter
    print(f"[sample] {len(specs)} 场景 -> {OUT/'specs.json'}",
          dict(Counter(s.target_class for s in specs)))


def _run_one(spec_dict: dict) -> dict:
    from llmdiag.episode_runner import run_instrumented_episode
    return run_instrumented_episode(spec_dict)


def stage_run(args):
    from llmdiag.scenarios_v2 import CaseSpec
    specs = json.loads((OUT / "specs.json").read_text())
    done: set = set()
    if (OUT / "episodes.jsonl").exists():
        for line in (OUT / "episodes.jsonl").read_text().splitlines():
            try:
                rec = json.loads(line)
                if "error" not in rec:
                    done.add(rec["case_id"])
            except Exception:
                continue
    todo = [s for s in specs if s["case_id"] not in done]
    print(f"[run] {len(specs)} 场景, 已完成 {len(done)}, 待跑 {len(todo)}")
    out = open(OUT / "episodes.jsonl", "a")
    ctx = mp.get_context("fork")
    for i, s in enumerate(todo):
        t0 = time.time()
        tmp = OUT / f".ep_{s['case_id']}.json"
        proc = ctx.Process(target=lambda: tmp.write_text(
            json.dumps(_run_one(s), ensure_ascii=False, default=str)))
        proc.start()
        proc.join(EPISODE_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate(); proc.join(5)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            rec = {"case_id": s["case_id"],
                   "error": f"hard timeout {EPISODE_TIMEOUT_S}s"}
        else:
            try:
                rec = json.loads(tmp.read_text())
            except Exception as e:  # noqa: BLE001
                rec = {"case_id": s["case_id"], "error": f"unreadable: {e}"}
        tmp.unlink(missing_ok=True)
        out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        out.flush()
        mark = rec.get("error") or f"mk={rec.get('makespan')} fin={rec.get('finished')} ev={rec.get('n_events')}"
        print(f"  [{i+1}/{len(todo)}] {s['case_id']:<38} {mark} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out.close()


def _load_cases(args):
    from llmdiag.verifiers import verify_case
    from llmdiag.archive_builder import build_archive, ground_truth_of
    from llmdiag.scenarios_v2 import sample_grid
    specs = {s["case_id"]: s for s in
             json.loads((OUT / "specs.json").read_text())}
    cases = []
    for line in (OUT / "episodes.jsonl").read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if "error" in rec or rec["case_id"] not in specs:
            continue
        case = {"case_id": rec["case_id"], "spec": specs[rec["case_id"]],
                "episode": rec}
        case["verify"] = verify_case(rec)
        case["ground_truth"] = ground_truth_of(case)
        case["archives"] = {v: build_archive(case, v)
                            for v in ("easy", "hard", "int")}
        cases.append(case)
    return cases


def stage_verify_archive(args):
    cases = _load_cases(args)
    (OUT / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, default=str))
    amb = [c["case_id"] for c in cases if c["verify"]["ambiguous"]]
    print(f"[verify+archive] {len(cases)} 案例 (弃用/歧义 {len(amb)}: "
          f"{amb[:6]}...)")

    toks = [c["archives"][v]["est_tokens"] for c in cases
            for v in ("easy", "hard", "int")]
    print(f"  档案 token 估算: max={max(toks)} mean={sum(toks)//len(toks)} "
          f"(预算 {8000})")


def stage_baseline(args):
    from llmdiag.baseline_diagnoser_v2 import diagnose
    from llmdiag.grader_v2 import grade_case, summarize
    cases = json.loads((OUT / "cases.json").read_text())
    for variant in ("easy", "hard"):
        grades = [grade_case(c, variant, diagnose(c, variant))
                  for c in cases if not c["verify"]["ambiguous"]]
        p = OUT / f"baseline_grades_{variant}.json"
        p.write_text(json.dumps(grades, ensure_ascii=False, indent=1))
        print(f"[baseline:{variant}] {json.dumps(summarize(grades), ensure_ascii=False)}")
        # 分档位塌陷对比 (方法论 §6 主张②)
        inj = [g for g in grades if g["target_class"] in (
            "disruption_machine", "disruption_machine_agv",
            "disruption_stochastic")]
        if inj:
            n = len(inj)
            print(f"    注入类子集 cause_acc="
                  f"{sum(g['cause_correct'] for g in inj)/n:.3f} (n={n})")


def _rerun_from_spec(spec_with_delta: dict) -> dict:
    from llmdiag.episode_runner import run_instrumented_episode
    return run_instrumented_episode(spec_with_delta)


def stage_counterfactual(args):
    """对基线干预做反事实执行 (同 seed 重仿真)。默认每类抽 1 个案例。"""
    from llmdiag.baseline_diagnoser_v2 import diagnose
    from llmdiag.grader_v2 import grade_intervention
    cases = json.loads((OUT / "cases.json").read_text())
    cases = [c for c in cases if not c["verify"]["ambiguous"]]
    picked, seen = [], set()
    for c in cases:
        if c["spec"]["target_class"] not in seen:
            seen.add(c["spec"]["target_class"])
            picked.append(c)
    out_lines = []
    for c in picked:
        ans = diagnose(c, "easy")
        g = grade_intervention(ans, c["spec"], c["episode"], _rerun_from_spec)
        g["case_id"] = c["case_id"]
        out_lines.append(g)
        print(f"  [cf] {c['case_id']:<38} {g['intervention']:<24} "
              f"ΔKPI={g.get('delta_kpi')} success={g.get('success')} "
              f"({g.get('note', '')[:30]})")
    (OUT / "baseline_counterfactual.json").write_text(
        json.dumps(out_lines, ensure_ascii=False, indent=1))
    ok = [g for g in out_lines if g.get("success") is not None]
    if ok:
        print(f"[counterfactual] 成功率 {sum(g['success'] for g in ok)}/{len(ok)}")


def stage_llm(args):
    from llmdiag.llm_diagnoser import diagnose as llm_diagnose
    from llmdiag.grader_v2 import grade_case, summarize
    cases = [c for c in json.loads((OUT / "cases.json").read_text())
             if not c["verify"]["ambiguous"]]
    grades = []
    for i, c in enumerate(cases):
        ans = llm_diagnose(c, args.variant, grounding=not args.no_grounding)
        grades.append(grade_case(c, args.variant, ans))
        print(f"  [llm {i+1}/{len(cases)}] {c['case_id']} "
              f"cause={ans['cause']:<24} meta={ans.get('meta')}", flush=True)
    tag = args.variant + ("_noground" if args.no_grounding else "")
    p = OUT / f"llm_grades_{tag}.json"
    p.write_text(json.dumps(grades, ensure_ascii=False, indent=1))
    print(f"[llm:{tag}] {json.dumps(summarize(grades), ensure_ascii=False)}")
    print("->", p)


def stage_report(args):
    from llmdiag.grader_v2 import summarize
    lines = ["# 方向7 v2 出题-判卷闭环报告", "",
             f"生成: {time.strftime('%F %T')} (master_seed=20260904, "
             f"route=astar 统一, 见 llmdiag/README.md 偏差说明)", ""]
    for variant in ("easy", "hard"):
        p = OUT / f"baseline_grades_{variant}.json"
        if p.exists():
            s = summarize(json.loads(p.read_text()))
            lines += [f"## 规则基线 @{variant}", f"```json",
                      json.dumps(s, ensure_ascii=False, indent=1), "```", ""]
    cf = OUT / "baseline_counterfactual.json"
    if cf.exists():
        lines += ["## 反事实执行 (基线干预, 每类 1 例)", "```json",
                  json.dumps(json.loads(cf.read_text()), ensure_ascii=False,
                             indent=1), "```", ""]
    for tag in ("easy", "hard", "easy_noground", "hard_noground"):
        p = OUT / f"llm_grades_{tag}.json"
        if p.exists():
            s = summarize(json.loads(p.read_text()))
            lines += [f"## LLM 臂 @{tag}", f"```json",
                      json.dumps(s, ensure_ascii=False, indent=1), "```", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"[report] -> {OUT/'report.md'}")


STAGES = ["sample", "run", "verify_archive", "baseline", "counterfactual",
          "llm", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all"] + STAGES,
                    help="all = sample->run->verify_archive->baseline->report")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--master-seed", type=int, default=20260904)
    ap.add_argument("--variant", default="hard", choices=["easy", "hard"],
                    help="llm 阶段使用的档案档位")
    ap.add_argument("--no-grounding", action="store_true",
                    help="llm 臂 A/B: 剥离接地硬约束")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    seq = (STAGES if args.stage == "all" else [args.stage])
    if args.stage == "all":
        seq = ["sample", "run", "verify_archive", "baseline", "report"]
    for st in seq:
        globals()[f"stage_{st}"](args)


if __name__ == "__main__":
    main()
