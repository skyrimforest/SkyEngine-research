"""方向7 v2: LLM 诊断器 (方法论 v0.1 §3 答案生产)
==================================================
openai-compatible endpoint, 环境变量:
  LLM_BASE_URL   如 https://api.xxx/v1  (本地 vLLM: http://host:8000/v1)
  LLM_API_KEY    sk-... (本地服务可任意)
  LLM_MODEL      模型名
  LLM_TSC        自一致性采样条数, 默认 10 (方法论 §3: 10-20 条聚合)

机制:
  - 固定格式 prompt: [系统提示:受控词表+接地硬约束] + [档案] + [查询]
  - JsonRegen: 解析失败带错误重生成, 循环至可解析 (≤3 次)
  - TSC: temperature=0 采样 LLM_TSC 条, 对 (localization, cause) 多数投票
  - A/B 接地约束: grounding=False 时从系统提示剥离硬约束 (忠实度消融)

用法 (配置 API 后):
  docker exec -w /work/sky_research -e LLM_BASE_URL=... -e LLM_API_KEY=... \
    -e LLM_MODEL=... skyresearch python llmdiag/run_v2_pilot.py --arm llm
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

VOCAB_LOCALIZATION = ("machine:<id> | agv:<id> | machine:<id>+agv:<id> | "
                      "stochastic | task_pool | corridor | machines | none")
VOCAB_CAUSE = ("disruption_machine | disruption_machine_agv | "
               "disruption_stochastic | starvation_livelock | blocking_livelock | "
               "machine_bottleneck | plan_mismatch | baseline | unseen")
VOCAB_INTERVENTION = ("padding(alpha=<a>) | fleet(K=<k>) | periodic_revision(<T>) "
                      "| assigner(<name>) | none")

SYSTEM_PROMPT = """你是 FJSP+MAPF 柔性制造仿真系统的调度诊断专家。
你会收到一个 episode 的带证据 ID 的运行档案(配置 CFG / 统计 STAT / 分段指标 KPI / 事件表 EVT / 修订账本 REV)。

你的任务: 回答用户查询, 输出严格 JSON (不要输出其他文字):
{{"localization": "<定位词表>", "cause": "<归因词表或unseen>",
 "evidence": ["<档案中的证据ID, 如 EVT001 / KPI:xxx[0-64] / CFG:K=4>", ...],
 "intervention": "<干预词表>", "narrative": "<中文因果叙述, 只允许使用 evidence 引用过的事实>"}}

定位词表: {loc}
归因词表: {cause}
干预词表: {ivt}

{hard_constraints}"""

HARD_CONSTRAINTS = """硬性约束:
1. evidence 只能引用档案中真实存在的证据 ID, 不得编造 ID 或数值;
2. narrative 中每个论断都要能在 evidence 中找到出处;
3. 若病因不在归因词表中, cause 必须填 unseen, 并仍尽量给出可执行干预;
4. 不确定时选择更一般的词表项, 而非编造。"""

NO_GROUNDING = "(本臂不施加接地硬约束, 仅作消融对照。)"


def _env() -> dict:
    return {
        "base_url": os.environ.get("LLM_BASE_URL", "").rstrip("/"),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", ""),
        "tsc": int(os.environ.get("LLM_TSC", "10")),
        "timeout": int(os.environ.get("LLM_TIMEOUT", "180")),
    }


def build_messages(case: dict, variant: str = "easy", grounding: bool = True):
    arch = case["archives"][variant]
    sys_prompt = SYSTEM_PROMPT.format(
        loc=VOCAB_LOCALIZATION, cause=VOCAB_CAUSE, ivt=VOCAB_INTERVENTION,
        hard_constraints=HARD_CONSTRAINTS if grounding else NO_GROUNDING)
    user = f"## 运行档案 ({variant} 档, 证据 ID 可回查)\n{arch['archive_text']}\n\n## 查询\n{case['spec']['query']}\n\n请输出诊断 JSON。"
    return [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}]


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    for candidate in (raw, raw.replace("'", '"')):
        try:
            d = json.loads(candidate)
            if isinstance(d, dict):
                return d
        except Exception:
            continue
    return None


def _chat(cfg: dict, messages: list) -> str:
    r = requests.post(
        f"{cfg['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        json={"model": cfg["model"], "messages": messages,
              "temperature": 0, "max_tokens": 1024},
        timeout=cfg["timeout"])
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def diagnose(case: dict, variant: str = "easy", grounding: bool = True,
             cfg: dict | None = None) -> dict:
    """diagnose(case) -> dict 接口 (与规则基线同形), 附带调用元数据。"""
    cfg = cfg or _env()
    if not cfg["base_url"]:
        raise RuntimeError("未配置 LLM_BASE_URL/LLM_API_KEY/LLM_MODEL "
                           "(见 llmdiag/README.md)")
    messages = build_messages(case, variant, grounding)
    t0 = time.time()
    regens, votes = 0, []
    last_err = "empty"
    for attempt in range(cfg["tsc"] + 3):
        try:
            text = _chat(cfg, messages)
        except Exception as e:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            last_err = str(e)
            continue
        d = _extract_json(text)
        if d is None:
            regens += 1
            messages = messages[:2] + [
                {"role": "assistant", "content": text[:400]},
                {"role": "user", "content":
                    "上面的输出不是可解析的 JSON。请转写为 YAML 检查字段后,"
                    "重新只输出符合 schema 的严格 JSON。"}]
            continue
        votes.append(d)
        if len(votes) >= cfg["tsc"]:
            break
    if not votes:
        return {"case_id": case["case_id"], "localization": "none",
                "cause": "unseen", "evidence": [], "intervention": "none",
                "narrative": f"LLM 调用失败: {last_err}",
                "meta": {"n_votes": 0, "regens": regens}}

    def majority(key):
        vals = [str(v.get(key, "")) for v in votes]
        return max(set(vals), key=vals.count)

    ev_votes: dict[str, int] = {}
    for v in votes:
        for e in v.get("evidence", []) if isinstance(v.get("evidence"), list) else []:
            ev_votes[str(e)] = ev_votes.get(str(e), 0) + 1
    evidence = [e for e, _ in sorted(ev_votes.items(),
                                     key=lambda kv: -kv[1])[:8]]
    return {
        "case_id": case["case_id"],
        "localization": majority("localization"),
        "cause": majority("cause"),
        "evidence": evidence,
        "intervention": majority("intervention"),
        "narrative": votes[-1].get("narrative", ""),
        "meta": {"n_votes": len(votes), "regens": regens,
                 "wall_time_s": round(time.time() - t0, 1),
                 "model": cfg["model"], "grounding": grounding},
    }
