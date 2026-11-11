"""Run Module 3 experiments. Usage: python scripts/run_fault_experiments.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bms.evaluation.fault_experiments import run_fault_experiments  # noqa: E402

if __name__ == "__main__":
    o = run_fault_experiments()
    print(o["real"].to_string()); print(o["baseline_25C_flag_rate"])
    s = o["sim"]; s = s[s.detector == "any"]
    print(s.pivot_table(index="fault_type", columns="magnitude", values="event_detection_rate").round(2).to_string())
    print(o["diagnostics"].head(8).to_string())
