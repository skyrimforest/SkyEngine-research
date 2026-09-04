"""方向7 v2: 场景生成器 (P1)
================================
从网格抽样生成诊断场景清单 scenarios_v2.jsonl。
- 注入类 (标签由构造为真): machine / machine+agv combo / mild stochastic preset / baseline
- 涌现类 (标签由核验器判定): starvation recipe / blocking recipe
确定性: random.Random(20260904)。
用法: python3 llmdiag/v2/scenarios.py [--out llmdiag/v2/scenarios_v2.jsonl]
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP_FILE = "data/mapf/medium_maps.yaml"
MAP_NAME = "medium-mazes-seed-0000"
BRAND = ROOT / "data/fjsp_official/brandimarte"

FIELDS = [
    "makespan", "finished", "steps",
    "machine_utilization", "operation_queue_waiting_time_mean",
    "tasked_stationary_count", "agv_busy_utilization", "agv_loaded_utilization",
    "agv_travel_time_total", "agv_waiting_time_total", "transport_delay_ratio",
    "transport_blocking_delay_mean", "machine_waiting_for_inbound_transfer_ratio",
    "throughput_jobs", "tardy_job_count", "total_tardiness",
    "machine_down_steps_total", "agv_down_steps_total",
    "active_disruption_count", "resource_recovery_completed_count",
    "resource_time_to_recover_mean", "obstacle_blocked_steps_total",
    "resource_disruption_loss_area", "swap_conflict_count",
]


def n_machines(inst: str) -> int:
    m = json.loads((BRAND / f"{inst}.json").read_text())["machines"]
    return m if isinstance(m, int) else len(m)


def gen():
    rng = random.Random(20260904)
    instances = ["mk01", "mk02", "mk03"]
    scen = []

    def add(scn_id, inst, policy, K, seed, exc, label, verifier, note=""):
        scen.append({
            "scen_id": scn_id, "instance": inst, "policy": policy,
            "num_agv": K, "seed": seed, "map_file": MAP_FILE, "map_name": MAP_NAME,
            "exception_config": exc, "label": label, "verifier": verifier, "note": note,
        })

    nm = {i: n_machines(i) for i in instances}
    # 1) 注入: 机器故障 16
    for i in range(16):
        inst = instances[i % 3]
        policy = ["cpsat-full", "greedy-reactive"][i % 2]
        K = [4, 6][i % 2]
        mid = rng.randrange(nm[inst])
        step, dur = rng.randrange(100, 251), rng.choice([20, 40, 60])
        add(f"inj_machine_{i:02d}", inst, policy, K, 42 + i,
            {"enabled": True, "random_seed": 42, "schedule": [
                {"step": step, "type": "machine_breakdown", "machine_id": mid,
                 "duration_steps": dur}]},
            {"loc_type": "machine", "loc_target": f"machine:{mid}",
             "cause": "disruption_machine"},
            "injected_machine", f"step={step} dur={dur}")
    # 2) 注入: 机+AGV 双故障 12
    for i in range(12):
        inst = instances[i % 3]
        policy = ["cpsat-full", "greedy-reactive"][i % 2]
        K = [4, 6][i % 2]
        mid, aid = rng.randrange(nm[inst]), rng.randrange(K)
        add(f"inj_combo_{i:02d}", inst, policy, K, 142 + i,
            {"enabled": True, "random_seed": 42, "schedule": [
                {"step": rng.randrange(100, 201), "type": "machine_breakdown",
                 "machine_id": mid, "duration_steps": rng.choice([30, 40, 60])},
                {"step": rng.randrange(220, 301), "type": "agv_breakdown",
                 "agv_id": aid, "duration_steps": rng.choice([15, 20, 30])}]},
            {"loc_type": "machine_agv", "loc_target": f"machine:{mid}+agv:{aid}",
             "cause": "disruption_machine_agv"},
            "injected_combo")
    # 3) 注入: mild stochastic preset 8
    for i in range(8):
        inst = instances[i % 3]
        policy = ["cpsat-full", "greedy-reactive"][i % 2]
        add(f"inj_stoch_{i:02d}", inst, policy, [4, 6][i % 2], 242 + i,
            {"enabled": True, "random_seed": 42, "preset": "mild_failure"},
            {"loc_type": "stochastic", "loc_target": "stochastic",
             "cause": "disruption_stochastic"},
            "preset_stochastic")
    # 4) 基线 6
    for i, inst in enumerate(instances):
        for policy in ["cpsat-full", "greedy-reactive"]:
            add(f"base_{inst}_{policy.split('-')[0]}", inst, policy, 4, 42,
                None, {"loc_type": "none", "loc_target": "none", "cause": "baseline"},
                "baseline")
    # 5) 涌现配方 (标签待核验): 饥饿(极小车队+greedy) 4 / 拥塞(大车队+cp-sat 走廊) 4
    for i, (inst, K, seed) in enumerate([("mk01", 2, 7), ("mk02", 2, 7),
                                         ("mk01", 3, 11), ("mk03", 2, 11)]):
        add(f"emg_starve_{i:02d}", inst, "greedy-reactive", K, seed, None,
            {"loc_type": "task_pool", "loc_target": "task_pool",
             "cause": "starvation_livelock"},
            "emergent_starvation", "预期无事件饥饿签名")
    for i, (inst, seed) in enumerate([("mk01", 7), ("mk02", 7),
                                      ("mk01", 11), ("mk03", 11)]):
        add(f"emg_block_{i:02d}", inst, "cpsat-full", 6, seed, None,
            {"loc_type": "corridor", "loc_target": "corridor",
             "cause": "blocking_livelock"},
            "emergent_blocking", "预期走廊拥塞签名")
    # 6) 注入场景的 CRN 无故障孪生 (同实例/策略/K/seed, 无故障):
    #    孪生未完工 -> 配对注入案例作废; makespan(孪生) vs makespan(故障) = 干预应恢复的 ΔKPI
    twins = [s for s in scen if s["verifier"].startswith("injected")]
    for s in twins:
        add(s["scen_id"] + "_T", s["instance"], s["policy"], s["num_agv"],
            s["seed"], None,
            {"loc_type": "none", "loc_target": "none", "cause": "baseline"},
            "baseline", f"CRN twin of {s['scen_id']}")
    return scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "scenarios_v2.jsonl"))
    args = ap.parse_args()
    scen = gen()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in scen:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(scen)} 场景 -> {out}")
    print(Counter(s["verifier"] for s in scen))


if __name__ == "__main__":
    main()
