"""方向7 v2.1: 四臂网格场景生成器 (欠账补齐)
================================================
v2 冻结数据集 (scenarios_v2.jsonl, 论文口径) 只含 cpsat-full/greedy-reactive 两臂 ——
根因: 本机旧 FJSP 微服务镜像不支持 online_fjsp_gateway 的 /solve+/cancel 无状态接口。
新版镜像 (skyengine-fjsp-best, trae/skycausal-evolution) 已就位后, 本文件恢复四臂网格
并把 plan_mismatch 类纳入 (scenarios_v2.py 头注预留的欠账):

  四臂: greedy-reactive / cpsat-static / cpsat-full / cpsat-partial
       (run_scenarios.py: greedy 臂=进程内 astar, 其余=rolling_mapf_http 微服务)
  plan_mismatch: CP-SAT 一次成型计划 (cpsat-static, trigger=never) 在持续故障流下失配
       —— 标签由构造为真 (静态臂零修订), 核验器检查故障流确实发生且造成退化。

协议与 v2 对齐 (build_archive.py):
  - schema 同 scenarios_v2.jsonl (scen_id/instance/policy/num_agv/seed/map_file/
    map_name/exception_config/label/verifier/note)
  - 注入类配 CRN 无故障孪生 (同实例/策略/K/seed), 孪生未完工则配对作废
  - emergent 配方与 v2 相同 (策略固有, 不扩臂)
  - 确定性: random.Random(20261106)

用法: python3 llmdiag/v2/scenarios_v21.py [--out llmdiag/v2/scenarios_v21.jsonl]
跑批: python3 llmdiag/v2/run_scenarios.py --scen scenarios_v21.jsonl --outdir episodes_v21
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
MAP_FILE = "data/mapf/medium_maps.yaml"
MAP_NAME = "medium-mazes-seed-0000"
INSTANCES = ["mk01", "mk02", "mk03"]
POLICIES = ["greedy-reactive", "cpsat-static", "cpsat-full", "cpsat-partial"]

# plan_mismatch 的查询 (icaps1 线 scenarios_v2.py QUERIES 原文)
QUERY_PLAN_MISMATCH = "为何扰动发生后 episode 表现持续恶化? 检查计划是否失配, 给干预建议。"


def n_machines(inst: str) -> int:
    brand = HERE.parents[1] / "data/fjsp_official/brandimarte"
    m = json.loads((brand / f"{inst}.json").read_text())["machines"]
    return m if isinstance(m, int) else len(m)


def gen():
    rng = random.Random(20261106)
    scen = []

    def add(scn_id, inst, policy, K, seed, exc, label, verifier, note=""):
        scen.append({
            "scen_id": scn_id, "instance": inst, "policy": policy,
            "num_agv": K, "seed": seed, "map_file": MAP_FILE, "map_name": MAP_NAME,
            "exception_config": exc, "label": label, "verifier": verifier, "note": note,
        })

    nm = {i: n_machines(i) for i in INSTANCES}

    # 1) 基线: 3 实例 × 4 臂 = 12
    for inst in INSTANCES:
        for p in POLICIES:
            add(f"v21_base_{inst}_{p}", inst, p, 4, 42, None,
                {"loc_type": "none", "loc_target": "none", "cause": "baseline"},
                "baseline")

    # 2) 注入: 机器故障 4 臂 × 4 = 16 (随机目标机/时刻/时长)
    for i in range(16):
        p = POLICIES[i % 4]
        inst = INSTANCES[(i // 4) % 3]
        K = [4, 6][i % 2]
        mid = rng.randrange(nm[inst])
        step, dur = rng.randrange(100, 251), rng.choice([20, 40, 60])
        add(f"v21_inj_machine_{i:02d}", inst, p, K, 42 + i,
            {"enabled": True, "random_seed": 42, "schedule": [
                {"step": step, "type": "machine_breakdown", "machine_id": mid,
                 "duration_steps": dur}]},
            {"loc_type": "machine", "loc_target": f"machine:{mid}",
             "cause": "disruption_machine"},
            "injected_machine", f"step={step} dur={dur}")

    # 3) 注入: 机+AGV 双故障 4 臂 × 3 = 12
    for i in range(12):
        p = POLICIES[i % 4]
        inst = INSTANCES[(i // 4) % 3]
        K = [4, 6][i % 2]
        mid, aid = rng.randrange(nm[inst]), rng.randrange(K)
        add(f"v21_inj_combo_{i:02d}", inst, p, K, 142 + i,
            {"enabled": True, "random_seed": 42, "schedule": [
                {"step": rng.randrange(100, 201), "type": "machine_breakdown",
                 "machine_id": mid, "duration_steps": rng.choice([30, 40, 60])},
                {"step": rng.randrange(220, 301), "type": "agv_breakdown",
                 "agv_id": aid, "duration_steps": rng.choice([15, 20, 30])}]},
            {"loc_type": "machine_agv", "loc_target": f"machine:{mid}+agv:{aid}",
             "cause": "disruption_machine_agv"},
            "injected_combo")

    # 4) 注入: mild stochastic preset 4 臂 × 2 = 8
    for i in range(8):
        p = POLICIES[i % 4]
        inst = INSTANCES[(i // 4) % 2]
        add(f"v21_inj_stoch_{i:02d}", inst, p, [4, 6][i % 2], 242 + i,
            {"enabled": True, "random_seed": 42, "preset": "mild_failure"},
            {"loc_type": "stochastic", "loc_target": "stochastic",
             "cause": "disruption_stochastic"},
            "preset_stochastic")

    # 5) plan_mismatch (新类): cpsat-static + 持续故障流 = 6
    #    静态臂 trigger=never 零修订, 扰动下计划必然失配 —— 标签由构造为真;
    #    核验器 (build_archive.verify: preset_mismatch) 要求故障流真实发生
    #    (active_disruption_count>0), 未发生则弃。
    for i, (inst, K) in enumerate([("mk01", 4), ("mk02", 4), ("mk03", 4),
                                   ("mk01", 6), ("mk02", 6), ("mk03", 6)]):
        add(f"v21_mismatch_{i:02d}", inst, "cpsat-static", K, 342 + i,
            {"enabled": True, "random_seed": 42, "preset": "moderate_failure"},
            {"loc_type": "stochastic", "loc_target": "stochastic",
             "cause": "plan_mismatch"},
            "preset_mismatch", "静态臂+故障流, 零修订失配 (QUERY: " + QUERY_PLAN_MISMATCH + ")")

    # 6) 涌现配方 (策略固有, 与 v2 相同): 饥饿(greedy+极小车队) 4 / 拥塞(cpsat-full+大车队) 4
    for i, (inst, K, seed) in enumerate([("mk01", 2, 7), ("mk02", 2, 7),
                                         ("mk01", 3, 11), ("mk03", 2, 11)]):
        add(f"v21_emg_starve_{i:02d}", inst, "greedy-reactive", K, seed, None,
            {"loc_type": "task_pool", "loc_target": "task_pool",
             "cause": "starvation_livelock"},
            "emergent_starvation", "预期无事件饥饿签名")
    for i, (inst, seed) in enumerate([("mk01", 7), ("mk02", 7),
                                      ("mk01", 11), ("mk03", 11)]):
        add(f"v21_emg_block_{i:02d}", inst, "cpsat-full", 6, seed, None,
            {"loc_type": "corridor", "loc_target": "corridor",
             "cause": "blocking_livelock"},
            "emergent_blocking", "预期走廊拥塞签名")

    # 7) CRN 无故障孪生: 所有注入类 (含 plan_mismatch) 同实例/策略/K/seed 无故障
    twins = [s for s in scen if s["verifier"].startswith(("injected", "preset_mismatch"))]
    for s in twins:
        add(s["scen_id"] + "_T", s["instance"], s["policy"], s["num_agv"],
            s["seed"], None,
            {"loc_type": "none", "loc_target": "none", "cause": "baseline"},
            "baseline", f"CRN twin of {s['scen_id']}")

    return scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "scenarios_v21.jsonl"))
    args = ap.parse_args()
    scen = gen()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in scen:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(scen)} 场景 -> {out}")
    print("按 verifier:", dict(Counter(s["verifier"] for s in scen)))
    print("按 policy:", dict(Counter(s["policy"] for s in scen)))


if __name__ == "__main__":
    main()
