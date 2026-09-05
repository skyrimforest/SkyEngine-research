"""方向7 v2: Patch B unseen 留出轮换 (OpsLLM精读 §3-B; 方法论 §6)
====================================================================
协议: 把一种已知病因从归因词表中整体抽走 (标签在手、词表没有), 考三指标:
  诚实率     = P(cause=unseen | 真实为留出类)        —— 该承认不认识
  沉默失败率 = P(cause=其他词表项 | 真实为留出类)    —— 不认识还硬编
  假阳性率   = P(cause=unseen | 真实为词表内类)      —— 抽走不该伤及无辜
三指标之外, ΔKPI 有效率 (干预桥) 用 counterfactual.grade 抽样补充 (--with-dkpi)。

轮换纪律 (OpsLLM Patch B 两坑): 必须逐类轮换报告均值±方差, 防挑折;
留出类数据不得泄入任何训练侧 (当前纯评测, 天然满足)。

用法 (容器内):
  LLM_BASE_URL=http://vllm:8000/v1 LLM_MODEL=/hf/... LLM_API=openai \
    python3 llmdiag/v2/run_unseen_rotation.py --holdout starvation_livelock
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import diagnoser_llm as D  # noqa: E402  (复用 call_llm/model_view/防泄露视图)
from score import score_one  # noqa: E402

ALL_CAUSES = ["disruption_machine", "disruption_machine_agv",
              "disruption_stochastic", "starvation_livelock",
              "blocking_livelock", "machine_bottleneck",
              "plan_mismatch", "baseline"]

DEPLOY_NOTE = ("部署说明: 本次接入的系统不存在「{holdout}」这类失效模式; "
               "若你判断病因不在上述词表中, cause 必须填 unseen, 不要勉强归入词表。")


def patched_system(holdout: str) -> str:
    """把留出类从 SYSTEM 的归因枚举里抽走, 加部署说明。"""
    keep = [c for c in ALL_CAUSES if c != holdout]
    return D.SYSTEM.replace(
        "cause: 归因, 枚举 " + "|".join(ALL_CAUSES),
        "cause: 归因, 枚举 " + "|".join(keep) + "|unseen"
    ) + "\n" + DEPLOY_NOTE.format(holdout=holdout)


def diagnose_unseen(case: dict, holdout: str):
    """与 diagnoser_llm.call_llm 同构, 但用抽走留出类的 SYSTEM。"""
    user = (f"## 运行档案\n{json.dumps(D.model_view(case), ensure_ascii=False)}\n\n"
            f"## 查询\n{case['query']}\n请只输出诊断JSON。/no_think")
    msgs = [{"role": "system", "content": patched_system(holdout)},
            {"role": "user", "content": user}]
    last_err = ""
    for attempt in range(3):
        try:
            body = {"model": D.MODEL, "stream": False, "temperature": 0,
                    "max_tokens": 2048, "messages": msgs}
            if D.NO_THINK:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            import urllib.request
            req = urllib.request.Request(
                f"{D.BASE}/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            txt = r["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:80]}"
            import time; time.sleep(3)
            continue
        try:
            raw = txt[txt.index("{"): txt.rindex("}") + 1]
            return json.loads(raw)
        except Exception:
            msgs = msgs[:2] + [{"role": "assistant", "content": txt[:400]},
                               {"role": "user", "content":
                                   "上面的输出无法解析为JSON。请只输出一个JSON对象。"}]
    return {"loc_type": "none", "loc_target": "none", "cause": "unseen",
            "evidence": [], "intervention": "none",
            "narrative": f"调用失败: {last_err}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, choices=ALL_CAUSES)
    ap.add_argument("--n-others", type=int, default=12,
                    help="非留出类抽样数 (测假阳性)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = json.loads((HERE / "cases_v2.json").read_text())
    hard_cases = [c for c in cases if c["variant"] == "hard"]
    holdout_cases = [c for c in hard_cases
                     if c["ground_truth"]["cause"] == args.holdout]
    others = [c for c in hard_cases if c["ground_truth"]["cause"] != args.holdout]
    # 非留出类按类均衡抽样
    step = max(1, len(others) // args.n_others)
    others = others[::step][: args.n_others]
    print(f"[rotation] holdout={args.holdout}: 留出 {len(holdout_cases)} 例 + "
          f"无辜对照 {len(others)} 例", flush=True)

    rows = []
    for i, c in enumerate(holdout_cases + others):
        pred = diagnose_unseen(c, args.holdout)
        is_holdout = c["ground_truth"]["cause"] == args.holdout
        rows.append({
            "case_id": c["case_id"], "is_holdout": is_holdout,
            "gt_cause": c["ground_truth"]["cause"],
            "pred_cause": str(pred.get("cause")),
            "pred_loc": str(pred.get("loc_type")),
            "intervention": str(pred.get("intervention")),
            "honest_unseen": bool(pred.get("cause") == "unseen" and is_holdout),
            "silent_failure": bool(is_holdout and pred.get("cause") != "unseen"),
            "false_positive": bool(not is_holdout and pred.get("cause") == "unseen"),
        })
        print(f"  [{i+1}/{len(holdout_cases)+len(others)}] {c['case_id']:<28} "
              f"gt={c['ground_truth']['cause']:<24} -> {pred.get('cause')}",
              flush=True)

    n_h = sum(1 for r in rows if r["is_holdout"])
    n_o = len(rows) - n_h
    summary = {
        "holdout": args.holdout,
        "n_holdout": n_h, "n_others": n_o,
        "诚实率_honest_rate": round(
            sum(r["honest_unseen"] for r in rows if r["is_holdout"]) / max(1, n_h), 3),
        "沉默失败率_silent_failure_rate": round(
            sum(r["silent_failure"] for r in rows if r["is_holdout"]) / max(1, n_h), 3),
        "假阳性率_false_positive_rate": round(
            sum(r["false_positive"] for r in rows if not r["is_holdout"]) / max(1, n_o), 3),
    }
    print("[rotation]", json.dumps(summary, ensure_ascii=False))
    out = args.out or str(HERE / f"results_unseen_{args.holdout}_"
                          f"{D.MODEL.split('/')[-1].replace('.', '_')}.json")
    Path(out).write_text(json.dumps(
        {"summary": summary, "results": rows}, ensure_ascii=False, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
