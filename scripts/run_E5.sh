#!/bin/bash
# E5: 路由求解器敏感性实验 (回应"文献基线"审稿关切)
# 依次把 mapf 服务换成 lacam / pibt 镜像, 各跑一轮 E5, 最后恢复 classical。
# 注意: 必须在主战役 (E1-E4) 结束后运行, 避免服务争用。
set -e
cd "$(dirname "$0")/.."

wait_healthy () {
  echo "waiting for mapf health (max 120s)..."
  for i in $(seq 1 60); do
    if docker exec skyresearch python -c "import requests; assert requests.get('http://mapf:8001/health', timeout=3).json()['status'] == 'ok'" 2>/dev/null; then
      echo "healthy after ~$((i*2))s"; return 0
    fi
    sleep 2
  done
  echo "mapf not healthy in 120s"; return 1
}

for IMG in lacam pibt; do
  echo "=== mapf image -> skyengine-mapf-$IMG ==="
  MAPF_IMAGE="skyengine-mapf-$IMG:latest" docker compose -f docker/research-stack.yaml up -d mapf
  wait_healthy || { echo "SKIP $IMG (unhealthy)"; continue; }
  docker exec -w /work/sky_research -e E5_ROUTE="$IMG" skyresearch \
    python closeloop/run_full.py --phase E5 --resume
done

echo "=== restore classical ==="
docker compose -f docker/research-stack.yaml up -d mapf
wait_healthy || true
docker exec skyresearch python -c \
  "import requests; print(requests.get('http://mapf:8001/health', timeout=5).json())"
