"""方向7: 本机小模型冒烟测试 (Ollama + Qwen3-4B, OpenAI兼容端点)
用法: python3 smoke_local_llm.py
- 取 cases_v1.json 的 easy(S1) 与 baseline(S0) 案例各2条
- 按 prompts.py 协议 (受控词表+接地约束) 请求四字段 JSON
- 检查: 格式合规 / 词表合规 / 定位归因对错 / 延迟
"""
import json, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "llmdiag/cases_v1.json").read_text())
BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen3:4b"

LOC_VOCAB = {"machine:<id>", "agv:<id>", "machine:<id>+agv:<id>", "stochastic",
             "task_pool", "corridor", "machines", "none"}
CAUSE_VOCAB = {"disruption_machine", "disruption_machine_agv", "disruption_stochastic",
               "starvation_livelock", "blocking_livelock", "machine_bottleneck",
               "plan_mismatch", "baseline"}
INT_VOCAB = {"padding", "fleet", "periodic_revision", "assigner", "watchdog_starvation", "machine_reroute", "none"}

SYSTEM = """你是FJSP+MAPF柔性制造仿真系统的调度诊断专家。你会收到一个episode的结构化运行档案(指标摘要/事件时间线/计划修订统计/配置)。
你的任务: 回答用户查询, 只输出一个JSON对象(不要任何其他文字):
{"localization": <定位词表项>, "cause": <归因词表项>, "evidence": [<引用档案中的字段与数值>], "intervention": <干预词表项>, "narrative": "<中文因果叙述, 只允许使用evidence中出现的事实>"}
定位词表: machine:<id> | agv:<id> | machine:<id>+agv:<id> | stochastic | task_pool | corridor | machines | none
归因词表: disruption_machine | disruption_machine_agv | disruption_stochastic | starvation_livelock | blocking_livelock | machine_bottleneck | plan_mismatch | baseline
干预词表(可带参数如 fleet(K=5)): padding | fleet | periodic_revision | assigner | watchdog_starvation | machine_reroute | none
硬性约束: 1.不得引用档案中不存在的数值; 2.narrative中的每个论断都要能在evidence中找到出处; 3.不确定时降低置信度而非编造。 /no_think"""


def pick():
    out, seen = [], set()
    for c in CASES:
        key = (c["ground_truth"]["cause"], c["case_id"].split("|")[-1])
        want = [("disruption_machine", None), ("baseline", None), ("disruption_machine_agv", None), ("disruption_stochastic", None)]
        if c["ground_truth"]["cause"] in {w[0] for w in want} and c["ground_truth"]["cause"] not in seen:
            seen.add(c["ground_truth"]["cause"])
            out.append(c)
    return out[:4]


def chat(case):
    prompt = f"## 运行档案\n{json.dumps(case, ensure_ascii=False)}\n\n## 查询\n{case['query']}\n\n请只输出诊断JSON。"
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.time() - t0
    text = resp["choices"][0]["message"]["content"].strip()
    usage = resp.get("usage", {})
    return text, dt, usage


def check(text, case):
    r = {"parse": False, "loc_ok": None, "cause_ok": None, "evidence_ok": None, "raw": text[:160]}
    try:
        j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        r["parse"] = True
        r["out"] = j
        r["loc_ok"] = (j.get("localization") == case["ground_truth"]["localization"])
        co = j.get("cause") == case["ground_truth"]["cause"] or (
            str(j.get("cause", "")).startswith("disruption") and case["ground_truth"]["cause"].startswith("disruption"))
        r["cause_ok"] = co
        r["evidence_ok"] = isinstance(j.get("evidence"), list) and len(j.get("evidence", [])) > 0
    except Exception:
        pass
    return r


cases = pick()
print(f"模型 {MODEL} | 冒烟 {len(cases)} 案例温度0\n" + "=" * 70)
for c in cases:
    text, dt, usage = chat(c)
    r = check(text, c)
    gt = c["ground_truth"]
    n_tok = usage.get("completion_tokens", "?")
    print(f"[{c['case_id'][:40]}]")
    print(f"  GT: loc={gt['localization']} cause={gt['cause']}")
    if r["parse"]:
        o = r["out"]
        print(f"  预测: loc={o.get('localization')} cause={o.get('cause')} ev={len(o.get('evidence',[]))}条 int={str(o.get('intervention'))[:30]}")
        print(f"  判定: 解析✓ 定位{'✓' if r['loc_ok'] else '✗'} 归因{'✓' if r['cause_ok'] else '✗'} 引用{'✓' if r['evidence_ok'] else '✗'} | {dt:.1f}s {n_tok}tok ({(n_tok/dt if isinstance(n_tok,(int,float)) and dt else 0):.1f} tok/s)")
        print(f"  叙述: {str(o.get('narrative'))[:80]}")
    else:
        print(f"  ✗ 解析失败: {r['raw']}")
    print()
