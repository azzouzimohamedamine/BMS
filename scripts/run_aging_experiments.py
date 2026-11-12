"""SOH / RUL experiments on the Kaggle ageing datasets. Usage: python scripts/run_aging_experiments.py [--quick]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from bms.evaluation.aging_experiments import run_aging_experiments  # noqa: E402

if __name__ == "__main__":
    out = run_aging_experiments(quick="--quick" in sys.argv); r = out["results"]
    pd.set_option("display.width", 220)
    lobo = r[r.protocol == "leave_one_battery_out"]
    print(lobo.groupby(["provenance", "target", "feature_set", "model"])[["MAE", "RMSE", "R2"]].mean().round(3).to_string())
