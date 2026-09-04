"""探针: 跑一条 S1 episode, 输出记录的字段结构 (供档案构建器对齐真实数据)"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from closeloop.orchestrator import run_closed_episode  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

res = run_closed_episode(
    fjsp_path=ROOT / "data/fjsp_official/brandimarte/mk01.json",
    map_file=ROOT / "data/mapf/medium_maps.yaml",
    map_name="medium-mazes-seed-0000",
    policy="cpsat-full",
    exception_config={"enabled": True, "random_seed": 42, "schedule": [
        {"step": 150, "type": "machine_breakdown", "machine_id": 0, "duration_steps": 40}]},
    num_agv=4, seed=42, max_steps=4096,
    route_solver_name="rolling_mapf_http",
)


def shape(x, depth=0):
    if depth > 2:
        return type(x).__name__
    if isinstance(x, dict):
        return {k: shape(v, depth + 1) for k, v in list(x.items())[:40]}
    if isinstance(x, list):
        return [shape(x[0], depth + 1), f"...{len(x)}"] if x else []
    return type(x).__name__ if not isinstance(x, (int, float, str, bool)) else x


out = {"_keys": sorted(res.keys()), "_shape": shape(res)}
Path(__file__).parent.joinpath("_probe_record.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1, default=str))
print("makespan:", res.get("makespan"), "| finished:", res.get("finished"),
      "| n_events:", res.get("n_events"), "| steps:", res.get("steps"))
print("keys:", sorted(res.keys()))
