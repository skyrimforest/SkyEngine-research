#!/bin/bash
# 链式调度: 等过夜批次(我们自己域的四臂)结束后, 跑 Cloud-OpsBench 复现子集 (8B -> 评测 -> 14B -> 评测)
set -u
LOG=/Users/skyrim/project/SkyEngine-research-llmdiag/llmdiag/v2/overnight.log
V2=/work/SkyEngine-research-llmdiag/llmdiag/v2

echo "[chain] 等待 OVERNIGHT_DONE ..."
while ! grep -q OVERNIGHT_DONE "$LOG" 2>/dev/null; do sleep 120; done
echo "[chain] $(date '+%F %T') 过夜批次完成, 启动 Cloud-OpsBench 复现"

docker exec -w "$V2" skyresearch python crossbench_cob.py run --model qwen3:8b 2>&1 | tail -6
echo "[chain] $(date '+%F %T') 8B 跑批完成, 评测"
docker exec -w "$V2" skyresearch python crossbench_cob.py eval --model qwen3:8b 2>&1 | tail -8

docker exec -w "$V2" skyresearch python crossbench_cob.py run --model qwen3:14b 2>&1 | tail -6
echo "[chain] $(date '+%F %T') 14B 跑批完成, 评测"
docker exec -w "$V2" skyresearch python crossbench_cob.py eval --model qwen3:14b 2>&1 | tail -8

echo "[chain] CROSSBENCH_DONE $(date '+%F %T')"
