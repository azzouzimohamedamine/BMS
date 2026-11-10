"""Data-quality checks and non-destructive cleaning."""
from __future__ import annotations

from typing import Any

import pandas as pd


def quality_report(df: pd.DataFrame, v_range=(2.0, 4.5), i_abs_max=5.0) -> dict[str, Any]:
    """Missing values, duplicates, invalid readings, sampling statistics (``df`` is not modified)."""
    dt = df["time_s"].diff().dropna()
    return {
        "n_rows": int(len(df)),
        "duration_h": float((df["time_s"].iloc[-1] - df["time_s"].iloc[0]) / 3600),
        "missing_per_column": {c: int(df[c].isna().sum()) for c in ["time_s", "voltage_v", "current_a", "step", "cycle", "temperature_c"]},
        "duplicated_timestamps": int(df["time_s"].duplicated().sum()),
        "non_monotonic_time": int((dt < 0).sum()),
        "dt_median_s": float(dt.median()),
        "dt_max_s": float(dt.max()),
        "n_gaps_gt_5s": int((dt > 5).sum()),
        "voltage_out_of_range": int(((df["voltage_v"] < v_range[0]) | (df["voltage_v"] > v_range[1])).sum()),
        "current_out_of_range": int((df["current_a"].abs() > i_abs_max).sum()),
        "voltage_min_max": [float(df["voltage_v"].min()), float(df["voltage_v"].max())],
        "current_min_max": [float(df["current_a"].min()), float(df["current_a"].max())],
        "cycle_index_unique": sorted(df["cycle"].dropna().unique().astype(int).tolist()),
    }


def clean(df: pd.DataFrame, v_range=(2.0, 4.5), i_abs_max=5.0) -> pd.DataFrame:
    """Drop duplicated timestamps (keep first), enforce monotonic time, drop physically invalid rows."""
    out = df.drop_duplicates(subset="time_s", keep="first").sort_values("time_s")
    ok = out["voltage_v"].between(*v_range) & (out["current_a"].abs() <= i_abs_max)
    return out[ok].reset_index(drop=True)
