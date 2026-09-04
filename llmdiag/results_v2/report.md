# 方向7 v2 出题-判卷闭环报告

生成: 2026-09-04 04:26:09 (master_seed=20260904, route=astar 统一, 见 llmdiag/README.md 偏差说明)

## 规则基线 @easy
```json
{
 "n": 16,
 "localization_acc": 0.812,
 "cause_acc": 0.688,
 "cause_acc_coarse": 0.875,
 "jra": 0.688,
 "unseen_escape_rate": 1.0,
 "evidence_faithfulness_mean": 1.0,
 "intervention_success_rate": null
}
```

## 规则基线 @hard
```json
{
 "n": 16,
 "localization_acc": 0.625,
 "cause_acc": 0.5,
 "cause_acc_coarse": 0.875,
 "jra": 0.5,
 "unseen_escape_rate": 0.0,
 "evidence_faithfulness_mean": 0.938,
 "intervention_success_rate": null
}
```

## 反事实执行 (基线干预, 每类 1 例)
```json
[
 {
  "intervention": "none",
  "executable": true,
  "delta_kpi": null,
  "success": null,
  "note": "不建议干预",
  "case_id": "v2_baseline_000"
 },
 {
  "intervention": "padding(alpha=0.2)",
  "executable": true,
  "delta_kpi": 0.0,
  "delta_relative": 0.0,
  "rerun_makespan": 4096,
  "rerun_finished": false,
  "success": false,
  "note": "同 seed=378107 重仿真",
  "case_id": "v2_disruption_machine_000"
 },
 {
  "intervention": "fleet(K=5)",
  "executable": true,
  "delta_kpi": 0.0,
  "delta_relative": 0.0,
  "rerun_makespan": 4096,
  "rerun_finished": false,
  "success": false,
  "note": "同 seed=189385 重仿真",
  "case_id": "v2_disruption_machine_agv_000"
 },
 {
  "intervention": "fleet(K=5)",
  "executable": true,
  "delta_kpi": 0.0,
  "delta_relative": 0.0,
  "rerun_makespan": 4096,
  "rerun_finished": false,
  "success": false,
  "note": "同 seed=222288 重仿真",
  "case_id": "v2_disruption_stochastic_000"
 },
 {
  "intervention": "assigner(random)",
  "executable": true,
  "delta_kpi": 0.0,
  "delta_relative": 0.0,
  "rerun_makespan": 4096,
  "rerun_finished": false,
  "success": false,
  "note": "同 seed=875387 重仿真",
  "case_id": "v2_blocking_livelock_000"
 },
 {
  "intervention": "padding(alpha=0.1)",
  "executable": true,
  "delta_kpi": -1.0,
  "delta_relative": -0.0016,
  "rerun_makespan": 610,
  "rerun_finished": true,
  "success": false,
  "note": "同 seed=435775 重仿真",
  "case_id": "v2_route_disruption_001"
 }
]
```
