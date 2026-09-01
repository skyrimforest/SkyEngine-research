# sky_research — FJSP+MAPF 引擎科研实验仓

本仓库承载 2026-09 起的七个科研探索方向的实验代码，驱动
`SkyEngine-confirmation-v2-1`（引擎/环境）+ `SkyEngine-FJSP`（调度微服务）+
`SkyEngine-MAPF`（路由微服务）三套已有系统，**不修改上游仓库**。

## 目录约定

```
common/        引擎适配层 (engine_adapter / fjsp_io)
docker/        研究栈 compose (fjsp + mapf 微服务, 去 GPU)
bench/         方向2: 统一基准评测
closeloop/     方向1: 闭环在线协同
decomp/        方向3: 学习引导分解 + 分解代价
lifelong/      方向4: Lifelong FJSP+MAPF
deadlock/      方向5: 有限缓冲与死锁
robust/        方向6: 鲁棒协同
llmdiag/       方向7: LLM 调度诊断解释层
data/          实例与地图 (符号链接到上游数据集)
results/       实验原始输出 (gitignore)
```

## 运行环境

宿主机无 pogema 环境，统一在容器内运行：

```bash
# 1. 驱动容器 (常驻, 挂载整个 codebase)
docker run -d --name skyresearch \
  -v <codebase绝对路径>:/work python:3.11-slim sleep infinity
docker exec skyresearch pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  "pogema==1.3.1" pyyaml matplotlib seaborn requests ortools gymnasium pettingzoo python-dotenv

# 2. 算法微服务栈 (CP-SAT + 经典 MAPF C++)
docker compose -f docker/research-stack.yaml up -d
docker network connect docker_research-net skyresearch

# 3. 跑实验
docker exec -w /work/sky_research skyresearch python <script>
```

## 分支约定

`main` 为基线；每个探索方向一条链式分支（后一个方向从前一个方向的分支拉出，
便于复用前序基建）：

| 分支 | 方向 |
|------|------|
| `0901bench` | 方向2 统一基准评测 |
| `0901closeloop` | 方向1 闭环在线协同 (基于 0901bench) |
| `0901decomp` | 方向3 学习引导分解 (基于 0901closeloop) |
| `0901lifelong` | 方向4 Lifelong (基于 0901decomp) |
| `0901deadlock` | 方向5 有限缓冲死锁 (基于 0901lifelong) |
| `0901robust` | 方向6 鲁棒协同 (基于 0901deadlock) |
| `0901llmdiag` | 方向7 LLM 诊断 (基于 0901robust) |
