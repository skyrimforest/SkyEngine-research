"""冒烟测试: 转换 mk01 并跑一个最小 episode, 验证引擎链路可用。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.fjsp_io import parse_brandimarte, load_instance_index
from common.engine_adapter import run_episode

# 1. 转换 mk01
src = ROOT / "data" / "fjsp_classic" / "brandimarte" / "mk01.txt"
data = parse_brandimarte(src)
out = ROOT / "data" / "fjsp" / "brandimarte_mk01.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data))
print(f"mk01: jobs={len(data['jobs'])}, machines={data['machines']}, "
      f"ops={sum(len(j) for j in data['jobs'])}")

# 2. 实例索引 (取 optimum)
idx = {e["name"]: e for e in load_instance_index(ROOT / "data" / "fjsp_classic" / "instances.json")}
print("mk01 optimum =", idx["mk01"]["optimum"])

# 3. 跑一个 episode
res = run_episode(
    fjsp_path=out,
    map_file=ROOT / "data" / "mapf" / "medium_maps.yaml",
    map_name="medium-mazes-seed-0000",
    num_agv=4,
    seed=42,
    max_steps=1024,
    job_solver="greedy",
    route_solver="astar",
    assigner="nearest",
)
print(f"finished={res['finished']} steps={res['steps']} wall={res['wall_time_s']}s")
print("--- summary ---")
for k, v in res["summary"].items():
    print(f"  {k}: {v}")
out_json = ROOT / "results" / "smoke_mk01.json"
out_json.parent.mkdir(exist_ok=True)
out_json.write_text(json.dumps(res, indent=2, default=str))
print("saved ->", out_json)
