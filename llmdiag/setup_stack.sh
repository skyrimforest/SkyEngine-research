#!/bin/bash
# llmdiag v2 研究栈一键搭建 (幂等)
# =====================================
# 组件:
#   research-net   docker 网络 (与 research-stack.yaml 同名兼容)
#   fjsp           skyengine-fjsp-best:latest  CP-SAT 微服务 :8002 (cpsat 系策略需要)
#   skyresearch    python:3.11-slim 驱动容器, 挂载 sky_research + 引擎 worktree
#
# 说明: v2 试点统一 route_solver=astar (进程内), 不需要 MAPF 微服务;
#       mapf-classical 镜像在本机缺失, 如需 EECBS 恢复见 llmdiag/README.md。
set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
ENGINE_DIR="$(dirname "$REPO_ROOT")/SkyEngine-confirmation-v2-1"
FJSP_DATA_DIR="$(dirname "$REPO_ROOT")/SkyEngine-FJSP"   # data/fjsp_official 符号链接目标
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "[1/5] docker 网络 research-net"
docker network inspect research-net >/dev/null 2>&1 || docker network create research-net >/dev/null

echo "[2/5] fjsp 微服务"
if docker ps --format '{{.Names}}' | grep -qx fjsp; then
  echo "  fjsp 已在运行"
elif docker ps -a --format '{{.Names}}' | grep -qx fjsp; then
  docker start fjsp >/dev/null && echo "  fjsp 已启动(复用)"
else
  docker run -d --name fjsp --network research-net \
    -e SKY_LOG_LEVEL=INFO -e SEED=42 -e TIME_LIMIT=10 \
    skyengine-fjsp-best:latest >/dev/null
  echo "  fjsp 已创建"
fi

echo "[3/5] skyresearch 驱动容器"
if docker ps -a --format '{{.Names}}' | grep -qx skyresearch; then
  echo "  skyresearch 已存在(如需重建: docker rm -f skyresearch)"
else
  docker run -d --name skyresearch --network research-net \
    -v "$REPO_ROOT":/work/sky_research \
    -v "$ENGINE_DIR":/work/SkyEngine-confirmation-v2-1 \
    -v "$FJSP_DATA_DIR":/work/SkyEngine-FJSP \
    -w /work/sky_research \
    python:3.11-slim sleep infinity >/dev/null
  echo "  skyresearch 已创建"
fi
docker network connect research-net skyresearch 2>/dev/null || true

echo "[4/5] 引擎依赖 (pogema==1.3.1 固定版本)"
docker exec skyresearch sh -c "python -c 'import pogema' 2>/dev/null" || {
  docker exec skyresearch pip install --no-cache-dir -i "$PIP_MIRROR" \
    pogema==1.3.1 pogema-toolbox==0.1.0 pettingzoo "numpy==1.23.5" \
    PyYAML==6.0.1 requests==2.32.5 python-dotenv \
    "matplotlib==3.8.3" "scipy==1.11.4" "ortools==9.10.4067" 2>&1 | tail -2
}

echo "[5/5] 冒烟测试: 引擎模块导入"
docker exec -w /work/sky_research skyresearch python -c "
import sys; sys.path.insert(0, '/work/SkyEngine-confirmation-v2-1')
from pogema import GridConfig
from sky_executor.grid_factory.factory.grid_factory_env import GridFactoryEnv
from sky_executor.grid_factory.factory.Component.Coordinator.coordinator import Coordinator
print('SMOKE_OK: engine imports fine')
"
echo "完成。运行: docker exec -w /work/sky_research skyresearch python llmdiag/run_v2_pilot.py --per-class 3"
