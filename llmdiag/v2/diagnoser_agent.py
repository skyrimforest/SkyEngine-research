"""方向7 v2: agentic 诊断器 (R3 Step 0, FROZEN)
================================================
评审行动计划 R3 (评审核对二 20260906): agentic Step 0 = 6 个查询工具 + few-shot ReAct,
步数预算 ≤20, 与 single-shot 臂 (diagnoser_llm.py) 完全同判分 (score.score_one), 保证
single-shot vs agentic 可比 (预注册 Q1)。冻结协议: 工具集/提示/预算一次定死, 不调优。

工具 (只读, 作用于防泄露档案视图 model_view, 同单发臂的泄露纪律):
  get_kpi()                    全部 KPI
  get_events_page(page)        事件流分页 (20/页, 含总数)
  get_event_stats()            事件按类型计数 + 首末步
  search_events(key, value)    按 payload/step/type 过滤事件 (≤30 条)
  get_revisions()              计划修订账本
  get_config()                 运行配置

指标 (G4): n_steps / tool_calls 按工具计数 / tools_used 覆盖率;
MTTI = JRA=1 案例的平均步数 (报告期聚合, 逐案落 n_steps)。

环境: LLM_BASE_URL(默认http://localhost:11434) LLM_MODEL(默认qwen3:4b)
      LLM_API: ollama | openai (与 diagnoser_llm.py 一致)
用法:
  python3 llmdiag/v2/diagnoser_agent.py --cases cases_v2.json --limit 4     # 冒烟
  python3 llmdiag/v2/diagnoser_agent.py --variant all                       # 全量
"""
import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")
API = os.environ.get("LLM_API", "ollama")
NO_THINK = os.environ.get("LLM_NO_THINK", "1") == "1"
MAX_STEPS = 20          # 冻结步数预算 (评审核对二 R3)
PAGE_SIZE = 20
SEARCH_CAP = 30

SYSTEM = """你是FJSP+MAPF柔性制造仿真系统的调度诊断专家。你通过工具查询运行档案来诊断,
而不是直接拿到全部档案。协议 (ReAct, 严格遵循):
  Thought: <一句话推理>
  Action: <工具名>(<参数>)
系统会返回 Observation。查询充分后输出最终答案:
  Final: {"loc_type": "...", "loc_target": "...", "cause": "...", "evidence": ["..."], "intervention": "...", "narrative": "..."}

字段协议 (与单发臂一致):
- loc_type: machine|agv|machine_agv|stochastic|task_pool|corridor|machines|none
- loc_target: 如 "machine:3"/"agv:1"/"machine:0+agv:2"; 无实体时与loc_type同值
- cause: disruption_machine|disruption_machine_agv|disruption_stochastic|starvation_livelock|blocking_livelock|machine_bottleneck|plan_mismatch|baseline|unseen
- evidence: 数组, 每条必须引用你工具查询结果中的字段/事件ID与数值 (如 "KPI:makespan=499" 或 "EVT001"); 不得引用查询结果中不存在的内容
- intervention: padding(alpha=<0.1|0.2|0.3>) | fleet(K=<k>) | periodic_revision(<T>) | assigner(<name>) | none
- narrative: 中文因果叙述, 只允许使用evidence中出现的事实

示例 (mini档案, 演示协议):
  Thought: 先看整体KPI判断是否完工。
  Action: get_kpi()
  Observation: {"makespan": 4096, "finished": false, "agv_waiting_time_total": 3100}
  Thought: 未完工(makespan=4096封顶)且AGV等待高, 查事件流找异常。
  Action: get_event_stats()
  Observation: {"counts": {"machine_breakdown": 1}, "first_step": 150, "last_step": 210}
  Thought: 有一台机器故障, 定位到具体故障参数。
  Action: search_events(key=type, value=machine_breakdown)
  Observation: [{"id": "EVT001", "step": 150, "payload": {"machine_id": 2, "duration_steps": 60}}]
  Final: {"loc_type": "machine", "loc_target": "machine:2", "cause": "disruption_machine", "evidence": ["EVT001", "KPI:makespan=4096", "KPI:agv_waiting_time_total=3100"], "intervention": "none", "narrative": "step150 机2故障60步, 系统未完工, 该故障为根因。"}

步数预算 20。到达预算仍未 Final 时, 基于已查询到的信息立即给出 Final。只输出 Thought/Action/Final 行。"""


# ---------------------------------------------------------------- 工具实现
def make_tools(case: dict) -> dict:
    """工具集: 闭包绑定防泄露档案视图 (ground_truth/meta 不出)"""
    view = {"config": case.get("config"), "kpi": case.get("kpi"),
            "events": case.get("events", []),
            "revisions": case.get("revisions"), "query": case.get("query")}
    evs = view["events"]

    def get_kpi():
        return view["kpi"]

    def get_events_page(page=0):
        page = max(0, int(page))
        return {"total": len(evs), "page": page,
                "events": evs[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]}

    def get_event_stats():
        counts, steps = {}, []
        for e in evs:
            counts[e.get("type", "?")] = counts.get(e.get("type", "?"), 0) + 1
            steps.append(e.get("step"))
        return {"counts": counts, "n_events": len(evs),
                "first_step": min(steps) if steps else None,
                "last_step": max(steps) if steps else None}

    def search_events(key="type", value=""):
        hits = []
        for e in evs:
            hay = {"type": e.get("type"), "step": e.get("step"),
                   "id": e.get("id"), **(e.get("payload") or {})}
            if str(hay.get(key)) == str(value):
                hits.append(e)
                if len(hits) >= SEARCH_CAP:
                    break
        return {"n_hits": len(hits), "events": hits}

    def get_revisions():
        return view["revisions"]

    def get_config():
        return view["config"]

    return {"get_kpi": get_kpi, "get_events_page": get_events_page,
            "get_event_stats": get_event_stats, "search_events": search_events,
            "get_revisions": get_revisions, "get_config": get_config}


ACTION_RE = re.compile(r"Action:\s*(\w+)\s*(?:\(([^)]*)\))?")
FINAL_RE = re.compile(r"Final:\s*(\{.*\})", re.S)


def parse_args_list(argstr: str) -> list:
    """解析 Action 括号内参数: 数字/带引号字符串/key=value"""
    argstr = (argstr or "").strip()
    if not argstr:
        return []
    parts = [p.strip() for p in argstr.split(",") if p.strip()]
    out = []
    for p in parts:
        if "=" in p:                       # key=value -> 只取值 (工具签名按位置)
            p = p.split("=", 1)[1].strip()
        if (p.startswith("'") and p.endswith("'")) or \
           (p.startswith('"') and p.endswith('"')):
            p = p[1:-1]
        try:
            out.append(int(p))
        except ValueError:
            out.append(p)
    return out


# ---------------------------------------------------------------- ReAct 循环
def run_agent(case: dict, log: list) -> tuple[dict | None, dict]:
    tools = make_tools(case)
    used = {k: 0 for k in tools}
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"## 查询\n{case['query']}\n请开始诊断 (Thought/Action)。/no_think"}]
    pred = None
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        txt = _chat(msgs)
        log.append({"step": steps, "text": txt[:500]})
        m = FINAL_RE.search(txt)
        if m:
            try:
                body = m.group(1)
                pred = json.loads(body[body.index("{"): body.rindex("}") + 1])
            except Exception:
                pred = None
            break
        a = ACTION_RE.search(txt)
        if not a:
            msgs = msgs + [{"role": "assistant", "content": txt[:600]},
                           {"role": "user",
                            "content": "未识别 Action。请输出 'Action: 工具(参数)' 或 'Final: {json}'。/no_think"}]
            continue
        name = a.group(1)
        if name not in tools:
            obs = {"error": f"unknown tool {name}; 可用: {sorted(tools)}"}
        else:
            used[name] += 1
            try:
                obs = tools[name](*parse_args_list(a.group(2)))
            except Exception as e:  # noqa: BLE001
                obs = {"error": f"{type(e).__name__}: {e}"}
        msgs = msgs + [{"role": "assistant", "content": txt[:600]},
                       {"role": "user",
                        "content": f"Observation: {json.dumps(obs, ensure_ascii=False)[:4000]}\n继续 (Thought/Action) 或给出 Final。/no_think"}]
    if pred is None:  # 预算耗尽: 强制收尾
        msgs = msgs + [{"role": "user",
                        "content": "步数预算已用完。请根据已查询信息立即输出 Final: {json}。/no_think"}]
        txt = _chat(msgs)
        m = FINAL_RE.search(txt)
        if m:
            try:
                pred = json.loads(m.group(1))
            except Exception:
                pred = None
    stats = {"n_steps": steps, "tool_calls": used,
             "n_tool_calls": sum(used.values()),
             "tools_used": sorted(k for k, v in used.items() if v > 0)}
    return pred, stats


def _chat(msgs: list) -> str:
    if API == "openai":
        body = {"model": MODEL, "stream": False, "temperature": 0,
                "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "2048")),
                "messages": msgs}
        if NO_THINK:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        url = f"{BASE}/chat/completions"
    else:
        body = json.dumps({"model": MODEL, "stream": False, "temperature": 0,
                           "options": {"num_ctx": 8192}, "messages": msgs}).encode()
        url = f"{BASE}/api/chat"
    req = urllib.request.Request(url, data=body if isinstance(body, bytes)
                                 else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return (r["choices"][0]["message"]["content"] if API == "openai"
            else r["message"]["content"]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "cases_v2.json"))
    ap.add_argument("--variant", default="all", choices=["all", "easy", "hard"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / f"results_agent_{MODEL.replace(':', '_').replace('/', '_')}.json"))
    args = ap.parse_args()
    from score import score_one, aggregate
    cases = json.loads(Path(args.cases).read_text())
    if args.variant != "all":
        cases = [c for c in cases if c["variant"] == args.variant]
    if args.limit:
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
        t0 = time.time()
        try:
            pred, stats = run_agent(c, log := [])
        except Exception as e:  # noqa: BLE001 — 单案例失败不拖垮全量
            print(f"[{i+1}/{len(cases)}] {c['case_id']}: CALL_FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(5)
            continue
        row = dict(score_one(pred, c), case_id=c["case_id"], gt=c["ground_truth"],
                   wall_s=round(time.time() - t0, 1), raw=pred, agent=stats)
        done[c["case_id"]] = row
        out.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1))
        gt = c["ground_truth"]
        print(f"[{i+1}/{len(cases)}] {c['case_id']}: loc={row['loc']} cause={row['cause']} "
              f"faith={row['faithful']} steps={stats['n_steps']} calls={stats['n_tool_calls']} "
              f"({row['wall_s']}s) GT={gt['loc_target']}/{gt['cause']}", flush=True)
    rows = list(done.values())
    agg = aggregate(rows)
    mt = [r["agent"]["n_steps"] for r in rows if r.get("jra") == 1.0]
    agg["mtti_steps"] = round(sum(mt) / len(mt), 2) if mt else None
    agg["tool_coverage"] = round(sum(len(r["agent"]["tools_used"]) for r in rows)
                                 / (len(rows) or 1), 2)
    print(json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
