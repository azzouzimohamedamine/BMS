"""Markdown experiment report generator (used by the dashboard download button and scripts)."""
from __future__ import annotations

import pandas as pd

from bms.integration.engine import BMSReport
from bms.utils.config import repo_root


def build_markdown_report(rep: BMSReport) -> str:
    M = repo_root() / "results" / "metrics"
    lines = [f"# BMS experiment report — {rep.dataset_id}", "",
             "Provenance legend: MEASURED / CALCULATED / ESTIMATED / SIMULATED / UNAVAILABLE.", "",
             "## Measured", *(f"- {k}: {v}" for k, v in rep.measured.items()), "",
             "## SOC (estimated vs CALCULATED reference)", *(f"- {k}: {v}" for k, v in rep.soc.items()), "",
             "## SOH", *(f"- {k}: {v}" for k, v in rep.soh.items()), "",
             "## RUL", *(f"- {k}: {v}" for k, v in rep.rul.items()), "",
             "## Fault diagnosis (statistical anomalies only)", f"- reported windows: {rep.fault.get('n_reported')}",
             f"- unavailable diagnostics: {rep.fault.get('unavailable')}", "", "## Warnings", *(f"- {w}" for w in rep.warnings), ""]
    p = M / "soc_results.csv"
    if p.exists():
        df = pd.read_csv(p); df = df[(df.noise == "clean") & (df.test == rep.dataset_id)]
        if len(df):
            lines += ["## SOC metrics on this condition (clean signals, all protocols)", "",
                      df[["protocol", "train", "model", "MAE", "RMSE", "R2", "MaxAE"]].round(3).to_markdown(index=False)
                      if hasattr(df, "to_markdown") and _has_tabulate() else df[["protocol", "train", "model", "MAE", "RMSE", "R2", "MaxAE"]].round(3).to_string(index=False), ""]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False
