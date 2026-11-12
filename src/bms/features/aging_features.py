"""Leak-free, causal features and labels for SOH / RUL on ageing datasets.

* SOH_rel(%) = capacity / FIRST-cycle capacity * 100   (relative to the cell's own initial capacity, NOT a nominal capacity)
* EOL = first cycle with capacity <= eol_fraction * first-cycle capacity ; RUL = EOL - cycle (units: retained records)
* Never used as inputs: RUL, NRUL, battery_life (answers), capacity_norm/diff/ma5 (capacity-derived, unknown normalisation),
  arrhenius_fade_rate (exact function of temp_mean, redundant). Rct/Re only in the exploratory 'ext' set (back-filled values).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL = ["voltage_mean", "voltage_std", "temp_mean", "temp_max", "ic_peak_height", "ic_peak_voltage",
          "ic_curve_energy", "dv_peak_height", "dv_curve_energy"]
EXT = ["Rct", "Re"]
HIST = ["soh_rel", "cap_ma3_rel", "fade_slope5"]
FEATURE_SETS = {
    "soh": {"cycle_only": ["cycle"], "base": ["cycle"] + SIGNAL, "no_cycle": SIGNAL, "ext_rct_re": ["cycle"] + SIGNAL + EXT},
    "rul": {"cap_hist": ["cycle"] + HIST, "full": ["cycle"] + HIST + SIGNAL},
}
FORBIDDEN = {"RUL", "NRUL", "battery_life", "capacity_norm", "capacity_diff", "capacity_ma5", "arrhenius_fade_rate", "rul", "nrul", "life", "eol",
             "capacity"}   # raw capacity is the SOH target itself; for RUL it enters only through soh_rel / history features


def assert_no_leak(features: list[str]) -> None:
    bad = sorted(set(features) & FORBIDDEN)
    if bad:
        raise ValueError(f"Forbidden (leaking/redundant) feature(s): {bad}")


def _causal_slope(y: np.ndarray, x: np.ndarray, k: int) -> float:
    lo = max(0, k - 4)
    if k - lo < 2:
        return 0.0
    return float(np.polyfit(x[lo:k + 1], y[lo:k + 1], 1)[0])


def add_capacity_features(df: pd.DataFrame, eol_fraction: float = 0.70, use_file_rul: bool = False) -> pd.DataFrame:
    """Add soh_rel, causal history features, EOL and RUL per (battery, replica). All history features at cycle k use cycles <= k only."""
    out = []
    for _, g in df.sort_values(["battery_id", "aug_rep", "cycle"]).groupby(["battery_id", "aug_rep"], sort=False):
        g = g.copy(); cap = g.capacity.to_numpy(); cyc = g.cycle.to_numpy().astype(float); q0 = cap[0]
        soh = cap / q0 * 100
        g["soh_rel"] = soh
        g["cap_ma3_rel"] = pd.Series(soh).rolling(3, min_periods=1).mean().to_numpy()
        g["fade_slope5"] = [_causal_slope(soh, cyc, k) for k in range(len(g))]
        hit = np.where(soh <= eol_fraction * 100)[0]
        if use_file_rul and "RUL" in g:                       # augmented replicas: labels of the original trajectory
            g["eol_cycle"] = float(g.battery_life.iloc[0]); g["rul"] = g["RUL"].astype(float)
        else:
            g["eol_cycle"] = float(cyc[hit[0]]) if len(hit) else np.nan
            g["rul"] = g["eol_cycle"] - cyc if len(hit) else np.nan
        g["life_frac"] = cyc / g["eol_cycle"]                 # EVALUATION ONLY (uses the label): never a feature
        out.append(g)
    return pd.concat(out, ignore_index=True)
