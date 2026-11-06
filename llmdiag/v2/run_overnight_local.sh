#!/bin/bash
# 过夜批量: cases_v2 (104案例) 四臂顺序跑 (本机 Ollama, Metal 单模型独占)
# 顺序: agentic-8B (R3 主交付) -> 单发14B (新档位) -> agentic-14B -> 单发8B (补齐对照)
# 用法: nohup bash llmdiag/v2/run_overnight_local.sh > llmdiag/v2/overnight.log 2>&1 &
set -u
ROOT=/work/SkyEngine-research-llmdiag
HOST=http://192.168.5.2:11434

wait_model() {  # 等 ollama 拉完模型
  while ! ~/.local/bin/ollama list 2>/dev/null | grep -q "^$1"; do
    echo "[wait] $1 未就绪, 60s 后重查"; sleep 60
  done
}

run_arm() {  # run_arm <script> <model> <out>
  echo "=== $(date '+%F %T') $2 $1 -> $3 ==="
  docker exec -w "$ROOT" skyresearch \
    env LLM_BASE_URL=$HOST LLM_MODEL=$2 LLM_API=ollama \
    python "llmdiag/v2/$1" --cases llmdiag/v2/cases_v2.json \
    --out "llmdiag/v2/$3" 2>&1 | tail -4
}

wait_model "qwen3:8b"
run_arm diagnoser_agent.py qwen3:8b results_agent_qwen3_8b.json

wait_model "qwen3:14b"
run_arm diagnoser_llm.py    qwen3:14b results_qwen3_14b.json
run_arm diagnoser_agent.py  qwen3:14b results_agent_qwen3_14b.json
run_arm diagnoser_llm.py    qwen3:8b  results_qwen3_8b_local.json

echo "=== $(date '+%F %T') OVERNIGHT_DONE ==="
