# llmdiag — 方向7: LLM 调度诊断解释层

论文: `论文_7_LLM调度诊断/article_llmdiag_v1_cn.tex`
方法论: `papers/7_LLM调度诊断/方法论_v0.1_数据与答案生产规范.md` (2026-09-04)

## 组成

### v2 出题-判卷闭环 (方法论 §7 工程件: 场景生成器 + ①档案构建器 + ②Prompt + ④评分器)

| 文件 | 职责 |
|---|---|
| `scenarios_v2.py` | 8 类场景网格采样器 (注入参数全随机化 + Unseen 留出类) |
| `episode_runner.py` | 插桩 runner: 全事件/KPI 逐步序列/修订账本 + 干预配置增量 |
| `verifiers.py` | 程序化核验器 ("阅卷人", 无事件类判据 + 注入确认) |
| `archive_builder.py` | 证据 ID 档案 (CFG/EVT/REV/KPI) ≤8k token + easy/hard/INT 三变体 |
| `grader_v2.py` | 三层判分: 比对 / 证据回查 / 反事实同 seed 重仿真 (ΔKPI) |
| `run_v2_pilot.py` | 闭环编排 (sample→run→verify→archive→grade→report, 断点续跑) |
| `setup_stack.sh` | 研究栈一键搭建 (fjsp 微服务 + skyresearch 容器 + 固定依赖) |

### v1 遗产 (对照基线)

- `build_cases.py`: v1 案例库 (closeloop 扰动试点, 40 案例, 已知缺陷见论文)
- `baseline_diagnoser.py`: v1 规则基线 (定位 95% / 粗归因 97.5%)
- `prompts.py`: v1 提示模板

### LLM 臂 (方法论 §3)

- `llm_diagnoser.py`: openai-compatible 客户端 (JsonRegen + TSC 多数投票 + 接地 A/B),
  待配置 `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL` 即可运行
- `baseline_diagnoser_v2.py`: 词表化规则基线 (v2 案例库对照臂, easy/hard 双档)

## 运行环境 (docker, 宿主机零依赖)

```bash
bash llmdiag/setup_stack.sh                    # 幂等: 网络+fjsp+skyresearch+依赖
docker exec -w /work/sky_research skyresearch \
  python llmdiag/run_v2_pilot.py --stage all --per-class 3
```

工件: `llmdiag/results_v2/` (specs/episodes/cases/archives/grades/report.md)

```bash
# LLM 臂 (配置 API 后)
docker exec -w /work/sky_research \
  -e LLM_BASE_URL=https://api.xxx/v1 -e LLM_API_KEY=sk-... -e LLM_MODEL=... \
  skyresearch python llmdiag/run_v2_pilot.py --stage llm --variant hard
# 接地消融: --stage llm --no-grounding
```

## 与方法论 v0.1 的偏差 (诚实记录)

1. **v2 试点统一 greedy-reactive + 进程内 astar 路由** (default 观测/priority 碰撞)。
   原因: 本机 `skyengine-fjsp-*` 镜像均为旧版 API, 不支持 worktree 引擎
   `online_fjsp_gateway` 的 `/solve`+`/cancel/{id}` 无状态接口, `skyengine-mapf-classical`
   (EECBS) 镜像亦缺失。**四臂网格与 plan_mismatch 类待微服务恢复后扩网** (代码已就绪,
   见 scenarios_v2.py 头注)。
2. `padding(α)` 干预执行为加工时间确定性放大 (multiplier_uniform low=high=1+α),
   即排程松弛的引擎侧近似。
3. 干预反事实的 Wilcoxon 检验需 ≥30 seeds (正式协议), 试点为单 seed 配对 + δ=5%。
4. Unseen 留出类 = `route_disruption` (temporary_obstacle 流): 标签在手、
   归因词表抽走, 测 LLM 的 unseen 逃生舱 (方法论 §6)。

## 下一步

1. 恢复新版 FJSP 微服务镜像 → 四臂网格 + plan_mismatch 类 (§5 对照设计)
2. 配置 LLM API → llm 臂 easy/hard × 有/无接地 × Unseen 留出
3. 扩网至方法论正式规模 (≥200 案例, 每类 ≥25)
4. 信息预算曲线: 4k/8k/16k 档案档位
