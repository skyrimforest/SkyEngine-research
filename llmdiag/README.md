# llmdiag — 方向7: LLM 调度诊断解释层

论文: `论文_7_LLM调度诊断/article_llmdiag_v1_cn.tex`

## 组成

- `build_cases.py`: 从 closeloop 扰动试点构建诊断案例库
  (`cases_v1.json`, 40 案例, 含注入扰动的 ground truth)
- `baseline_diagnoser.py`: 规则基线诊断器 (对照臂), 评测定位/归因准确率
  - 当前: localization 95%, coarse cause 97.5%
  - 已知标注缺陷: S0 场景的系统性活锁案例 gt 应为双标签 (文档化于论文)
- `prompts.py`: LLM 提示模板 (受控词表 + 证据可回查 + 干预可执行
  + 反事实验证说明)

## 下一步 (待用户配置 LLM API)

1. 实现与 `diagnose(case) -> dict` 同签名的 LLM 诊断器
   (openai-compatible endpoint, env: LLM_BASE_URL/LLM_API_KEY/LLM_MODEL)
2. A/B: 有/无接地约束的忠实度对比
3. 反事实执行器: 对 intervention 词表逐项改配置重仿真, dKPI 评分

```bash
# 容器内复现基线
docker exec -w /work/sky_research skyresearch python llmdiag/build_cases.py
docker exec -w /work/sky_research skyresearch python llmdiag/baseline_diagnoser.py
```
