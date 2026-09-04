"""方向7 v2: LLM 诊断器 (OpenAI兼容端点, 双字段词表, JsonRegen)
================================================================
用法:
  python3 llmdiag/v2/diagnoser_llm.py --cases cases_v2.json --limit 4      # 冒烟
  python3 llmdiag/v2/diagnoser_llm.py --variant all                        # 全量(慢)
环境: LLM_BASE_URL(默认http://localhost:11434) LLM_MODEL(默认qwen3:4b)
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")

SYSTEM = """你是FJSP+MAPF柔性制造仿真系统的调度诊断专家。根据运行档案填写诊断JSON, 字段:
- loc_type: 定位类型, 枚举 machine|agv|machine_agv|stochastic|task_pool|corridor|machines|none
- loc_target: 定位目标, 如 "machine:3"/"agv:1"/"machine:0+agv:2"; 无实体时与loc_type同值(如"task_pool")
- cause: 归因, 枚举 disruption_machine|disruption_machine_agv|disruption_stochastic|starvation_livelock|blocking_livelock|machine_bottleneck|plan_mismatch|baseline|unseen
- evidence: 数组, 每条引用档案中的字段名与数值(如 "agv_waiting_time_total=3137" 或事件描述), 不得引用档案中不存在的数值
- intervention: 引擎可执行干预, 从 padding(alpha=0.2)|fleet(K+1)|periodic_revision(T=100)|assigner(least_congestion)|assigner(random)|none 中选择
- narrative: 中文因果叙述, 只允许使用evidence中出现的事实; 不确定时降低置信度而非编造
只输出一个JSON对象。"""


def call_llm(case: dict) -> tuple[dict | None, int, float]:
    user = (f"## 运行档案\n{json.dumps(case, ensure_ascii=False)}\n\n"
            f"## 查询\n{case['query']}\n请只输出诊断JSON。/no_think")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(3):  # JsonRegen: 解析失败带反馈重试
        body = json.dumps({"model": MODEL, "stream": False, "temperature": 0,
                           "options": {"num_ctx": 8192}, "messages": msgs}).encode()
        req = urllib.request.Request(f"{BASE}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        dt = time.time() - t0
        txt = r["message"]["content"].strip()
        try:
            j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            return j, attempt, dt
        except Exception:
            msgs = msgs[:2] + [{"role": "assistant", "content": txt[:400]},
                               {"role": "user",
                                "content": "上面的输出无法解析为JSON。请只输出一个JSON对象, 不要任何其他文字。/no_think"}]
    return None, 2, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "cases_v2.json"))
    ap.add_argument("--variant", default="all", choices=["all", "easy", "hard"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results_llm.json"))
    args = ap.parse_args()
    from score import score_one, aggregate
    cases = json.loads(Path(args.cases).read_text())
    if args.variant != "all":
        cases = [c for c in cases if c["variant"] == args.variant]
    if args.limit:
        # 冒烟采样: 每种 cause 取前 N/6
        from collections import defaultdict
        by = defaultdict(list)
        for c in cases:
            by[c["ground_truth"]["cause"]].append(c)
        per = max(1, args.limit // max(1, len(by)))
        cases = [c for lst in by.values() for c in lst[:per]][:args.limit]

    out = Path(args.out)
    done = {r["case_id"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    for i, c in enumerate(cases):
        if c["case_id"] in done:
            continue
        pred, retries, dt = call_llm(c)
        row = dict(score_one(pred, c), case_id=c["case_id"], gt=c["ground_truth"],
                   retries=retries, wall_s=round(dt, 1), raw=pred)
        done[c["case_id"]] = row
        out.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1))
        gt = c["ground_truth"]
        print(f"[{i+1}/{len(cases)}] {c['case_id']}: "
              f"loc={row['loc']} cause={row['cause']} faith={row['faithful']} "
              f"({dt:.0f}s, retry={retries}) GT={gt['loc_target']}/{gt['cause']}", flush=True)
    print(json.dumps(aggregate(list(done.values())), ensure_ascii=False))


if __name__ == "__main__":
    main()
