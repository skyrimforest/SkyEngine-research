"""方向7 × Cloud-OpsBench 跨域公平对比 (复现实验)
====================================================
目标: 同模型/同服务栈/同协议 (T=0, ReAct, 20 步) 在两个域上测同一诊断能力。
- 他们的域: K8s 云运维 RCA (754 案例快照回放, github.com/LLM4Ops/Cloud-OpsBench)
- 我们的域: FJSP+MAPF 耦合调度诊断 (cases_v2, 104 案例)
公平性口径:
  1. 模型与服务栈完全一致: qwen3:8b/14b @ 本机 Ollama (Metal), temperature=0
  2. 协议完全一致: 他们的 agentic harness 原样运行 (20 步预算, 12 工具),
     评测用他们的 evaluation.py (CA/FA/JRA + 过程指标), 我们零改动
  3. 子集分层: 按难度比例 (easy 391/medium 189/hard 174) 抽 48 例, 确定性种子
引用纪律: 他们论文的已发表数字 (Qwen3-8B JRA 0.19 OB) 作锚点核对复现偏差。

用法 (容器内):
  python llmdiag/v2/crossbench_cob.py sample            # 生成子集清单
  python llmdiag/v2/crossbench_cob.py run --model qwen3:8b
  python llmdiag/v2/crossbench_cob.py eval --model qwen3:8b
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
COB_ROOT = Path("/work/codebase/Cloud-OpsBench")
AGENT_ROOT = COB_ROOT / "agents" / "cloudops_agent"
SAVE_ROOT = COB_ROOT / "traces_local"          # 轨迹输出 (不入库)
OUR_SCRIPTS = HERE                              # 驱动所在
STRAT = {  # 每层抽样配额 (与总体 391/189/174 ≈ 52/25/23 比例一致)
    ("boutique", "scheduling", "easy"): 6,
    ("boutique", "startup", "easy"): 3,
    ("boutique", "runtime", "easy"): 3,
    ("boutique", "admission", "medium"): 3,
    ("boutique", "service", "medium"): 3,
    ("boutique", "infrastructure", "medium"): 2,
    ("boutique", "codedefect", "hard"): 3,
    ("boutique", "performance", "hard"): 2,
    ("trainticket", "runtime", "easy"): 5,
    ("trainticket", "startup", "easy"): 3,
    ("trainticket", "service", "medium"): 4,
    ("trainticket", "performance", "hard"): 6,
}


def sample_manifest() -> list:
    rng = random.Random(20261107)
    out = []
    for (sysname, cat, diff), quota in STRAT.items():
        cases = sorted((p.name for p in (COB_ROOT / "benchmark" / sysname / cat).iterdir()
                        if p.is_dir() and (p / "metadata.json").exists()),
                       key=lambda x: int(x) if x.isdigit() else x)
        out += [{"system": sysname, "fault_category": cat, "difficulty": diff,
                 "case_name": c} for c in rng.sample(cases, min(quota, len(cases)))]
    (HERE / "cob_subset.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{len(out)} 案例 -> cob_subset.json")
    return out


def run_one(scn: dict, model: str):
    """子进程内: 导入他们的 harness 跑单案例 (fork 隔离 + 硬超时在外层)"""
    sys.path.insert(0, str(AGENT_ROOT))
    import run as cob_run  # noqa: PLC0415
    case_path = COB_ROOT / "benchmark" / scn["system"] / scn["fault_category"] / scn["case_name"]
    config = {
        "model": {"model": model, "provider": "openai_compatible",
                  "api_base": "http://192.168.5.2:11435/v1", "api_key": "ollama",
                  "temperature": 0, "max_tokens": 4096, "timeout": 300,
                  # 思考链开关经 11435 翻译代理落到 Ollama 原生 think 参数;
                  # 他们论文为 vLLM+关思考, 我们关思考抢答不合规, 开思考可复现 ReAct
                  # (服务栈差异, 与论文 §Scope 既有口径同型, 论文中如实声明)
                  "enable_thinking": True},
        "diagnosis": {"max_iterations": 20, "system": scn["system"],
                      "fault_category": scn["fault_category"],
                      "case_name": "", "dataset_root": str(COB_ROOT),
                      "save_root": str(SAVE_ROOT)},
    }
    runner = cob_run.build_model_runner(config["model"])
    cob_run.run_case(config, case_path, runner)


def run_subset(model: str):
    scns = json.loads((HERE / "cob_subset.json").read_text())
    ok = fail = skip = 0
    for i, scn in enumerate(scns):
        trace = SAVE_ROOT / scn["system"] / model / scn["fault_category"] / scn["case_name"] / f"{scn['case_name']}.json"
        if trace.exists() and "stop_reason" in json.loads(trace.read_text()):
            skip += 1
            continue
        tmp = trace.parent / ".tmp.json" if trace.parent.exists() else None
        ctx = mp.get_context("fork")

        def _target():
            t0 = time.time()
            try:
                run_one(scn, model)
                print(f"[{i+1}/{len(scns)}] {scn['system']}/{scn['fault_category']}/{scn['case_name']} "
                      f"OK {round(time.time()-t0)}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{i+1}/{len(scns)}] {scn['system']}/{scn['fault_category']}/{scn['case_name']} "
                      f"ERR {type(e).__name__}: {str(e)[:100]}", flush=True)

        p = ctx.Process(target=_target)
        p.start()
        p.join(1200)  # 20 分钟硬超时 (20 步 × 每步最坏 1 分钟)
        if p.is_alive():
            p.terminate(); p.join(5)
            if p.is_alive():
                p.kill(); p.join(5)
            print(f"[{i+1}/{len(scns)}] {scn['system']}/{scn['fault_category']}/{scn['case_name']} TIMEOUT", flush=True)
            fail += 1
            continue
        if trace.exists():
            ok += 1
        else:
            fail += 1
    print(f"done: ok={ok} fail={fail} skip={skip}")


def eval_subset(model: str):
    """逐 (system, category) 写评测配置 -> 调他们的 evaluation.py -> 聚合"""
    import subprocess
    import tempfile
    scns = json.loads((HERE / "cob_subset.json").read_text())
    groups = sorted({(s["system"], s["fault_category"]) for s in scns})
    all_details = []
    for sysname, cat in groups:
        names = sorted({s["case_name"] for s in scns
                        if s["system"] == sysname and s["fault_category"] == cat},
                       key=lambda x: int(x) if x.isdigit() else x)
        cfg = {
            "model": {"model": model},
            "diagnosis": {"system": sysname, "fault_category": cat,
                          "case_name": "", "dataset_root": str(COB_ROOT),
                          "save_root": str(SAVE_ROOT)},
            "evaluation": {"label_root": str(COB_ROOT / "process-label" / sysname / cat),
                           "agent_root": str(SAVE_ROOT / sysname / model / cat),
                           "summary_out": str(SAVE_ROOT / "eval" / f"{sysname}_{cat}_{model}_summary.json"),
                           "detail_out": str(SAVE_ROOT / "eval" / f"{sysname}_{cat}_{model}_detail.json")},
        }
        # 他们 evaluation.py 读 configs/model_configs.yaml (同 RELATIVE 路径), 写临时副本
        cfg_path = AGENT_ROOT / "configs" / "model_configs.yaml"
        backup = cfg_path.read_bytes() if cfg_path.exists() else None
        try:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for sec, kv in cfg.items():
                lines.append(f"{sec}:")
                for k, v in kv.items():
                    lines.append(f"  {k}: {v}")
            cfg_path.write_text("\n".join(lines) + "\n")
            r = subprocess.run([sys.executable, "evaluation.py"], cwd=str(AGENT_ROOT),
                               capture_output=True, text=True, timeout=600)
            print(f"[eval {sysname}/{cat}] {r.stdout.strip().splitlines()[-2] if r.stdout.strip() else r.stderr[-200:]}")
            detail_p = SAVE_ROOT / "eval" / f"{sysname}_{cat}_{model}_detail.json"
            if detail_p.exists():
                for d in json.loads(detail_p.read_text()):
                    if d.get("case_name") in names or names is not None:
                        all_details.append(d)
        finally:
            if backup is not None:
                cfg_path.write_bytes(backup)
    (SAVE_ROOT / "eval" / f"crossbench_{model}_all_details.json").write_text(
        json.dumps(all_details, ensure_ascii=False, indent=1))
    # 聚合: JRA / CA / FA 按难度
    from collections import defaultdict
    agg = defaultdict(list)
    for d in all_details:
        diff = next((s["difficulty"] for s in scns
                     if s["system"] == d.get("system") and s["fault_category"] == d.get("fault_category")
                     and s["case_name"] == str(d.get("case_name"))), "?")
        agg[diff].append(d)
        agg["ALL"].append(d)
    print(f"\n== {model} × Cloud-OpsBench 子集 (n={len(all_details)}) ==")
    for key in ("ALL", "easy", "medium", "hard"):
        rows = agg.get(key) or []
        if not rows:
            continue
        n = len(rows) or 1
        f = lambda k: round(sum(r.get(k, 0) or 0 for r in rows) / n, 3)  # noqa: E731
        print(f"  {key}: n={n} JRA={f('JRA')} CA={f('CA')} FA={f('FA')} "
              f"steps={f('steps')} RAR={f('RAR')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "run", "eval"])
    ap.add_argument("--model", default="qwen3:8b")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample_manifest()
    elif args.cmd == "run":
        run_subset(args.model)
    else:
        eval_subset(args.model)


if __name__ == "__main__":
    main()
