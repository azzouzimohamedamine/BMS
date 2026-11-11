"""Module 2/4 feasibility: capacity & resistance table per condition, and explicit RUL/SOH feasibility verdicts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from bms.data.loader import dataset_ids  # noqa: E402
from bms.evaluation.soc_experiments import ecm_for  # noqa: E402
from bms.models.rul.rul import assert_rul_feasible  # noqa: E402
from bms.models.soh.capacity import condition_capacity_table, soh_percent  # noqa: E402
from bms.preprocessing.pipeline import prepare_condition  # noqa: E402
from bms.utils.config import load_dataset_config, repo_root  # noqa: E402
from bms.utils.errors import InsufficientDataError  # noqa: E402

cfg = load_dataset_config(); out = repo_root() / "results"
conds = {d: prepare_condition(d) for d in dataset_ids(cfg)}
r0 = {d: ecm_for(d, c, cfg).r0 for d, c in conds.items()}
tab = condition_capacity_table(conds, r0)
tab.to_csv(out / "metrics" / "soh_capacity_table.csv", index=False)
print(tab.to_string())
verdict = {}
nom, ver = cfg["cell"]["nominal_capacity_ah"], cfg["cell"]["nominal_capacity_verified"]
try:
    soh_percent(conds["T25C_lowI"].capacity_ah, nom, ver); verdict["soh_percent"] = "computed"
except InsufficientDataError as e:
    verdict["soh_percent"] = f"NOT COMPUTED: {e}"
for d in conds:      # one cycle index per file
    try:
        assert_rul_feasible(np.array([1]), np.array([conds[d].capacity_ah]), nom, ver)
    except InsufficientDataError as e:
        verdict.setdefault("rul", {})[d] = f"NOT FEASIBLE: {e}"
json.dump(verdict, open(out / "experiment_logs" / "soh_rul_feasibility.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(verdict, indent=2))
