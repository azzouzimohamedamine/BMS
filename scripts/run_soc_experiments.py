"""Run all SOC experiments (Module 1). Usage: python scripts/run_soc_experiments.py [--quick]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bms.evaluation.soc_experiments import run_soc_experiments  # noqa: E402

if __name__ == "__main__":
    out = run_soc_experiments(quick="--quick" in sys.argv)
    print(out["results"].query("noise=='clean'")[["protocol", "train", "test", "model", "MAE", "RMSE", "R2", "MaxAE"]].round(3).to_string())
