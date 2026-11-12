"""Physically motivated RUL baselines for the ageing datasets (causal: only data up to the current cycle)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_mean_life(train_lives: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """RUL = mean training life - current cycle (no battery information used)."""
    return np.maximum(float(np.mean(train_lives)) - np.asarray(cycle, float), 0.0)


def baseline_linear_extrapolation(g: pd.DataFrame, train_lives: np.ndarray, eol_pct: float = 70.0, window: int = 10) -> np.ndarray:
    """At each cycle k fit a line to soh_rel over the last ``window`` cycles and extrapolate to the EOL level.
    Falls back to the mean-life baseline when the fitted slope is not negative; the result is capped at the longest
    training life (an unbounded extrapolation from a near-zero slope is not a physically meaningful RUL)."""
    g = g.sort_values("cycle"); cyc = g.cycle.to_numpy(float); soh = g.soh_rel.to_numpy(float); out = np.empty(len(g))
    for k in range(len(g)):
        lo = max(0, k - window + 1)
        slope = np.polyfit(cyc[lo:k + 1], soh[lo:k + 1], 1)[0] if k - lo >= 2 else 0.0
        raw = (soh[k] - eol_pct) / -slope if slope < 0 else float(np.mean(train_lives)) - cyc[k]
        out[k] = min(max(raw, 0.0), float(np.max(train_lives)))
    return out
