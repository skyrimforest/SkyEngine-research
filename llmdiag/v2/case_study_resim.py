"""案例页专用: 三张处方的同种子重仿真裁决 (真实运行)"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from counterfactual import grade  # noqa: E402

scns = {json.loads(l)["scen_id"]: json.loads(l)
        for l in (HERE / "scenarios_v21.jsonl").read_text().splitlines() if l.strip()}
cases = {c["case_id"]: c for c in json.loads((HERE / "cases_v21.json").read_text())}
scn = scns["v21_mismatch_02"]

RX = [
    ("v21_mismatch_02|easy", "padding(alpha=0.2)",        "LLM医生·easy档"),
    ("v21_mismatch_02|hard", "periodic_revision(60)",     "LLM医生·hard档"),
    ("v21_mismatch_02|easy", "fleet(K+1)",                "规则医生"),
]
out = []
for case_id, rx, label in RX:
    case = cases[case_id]
    r = grade(case, scn, rx)
    r["label"] = label
    out.append(r)
    print(label, rx, "-> rerun_mk:", r.get("rerun_makespan"),
          "ΔKPI:", r.get("delta_kpi"), "success:", r.get("success"), flush=True)

(HERE / "case_study_resim.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("saved case_study_resim.json")
