"""Causal feature engineering for SOC estimation (only measurements that exist in every file: V, I)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["v", "i", "v_ma", "dv_lag"]


def build_features(v: np.ndarray, i: np.ndarray, ts_s: float, lag_s: float = 600.0) -> pd.DataFrame:
    """Voltage, current, causal moving-average voltage and voltage change over ``lag_s`` (no future information)."""
    k = max(1, int(round(lag_s / ts_s)))
    sv = pd.Series(v)
    v_ma = sv.rolling(k, min_periods=1).mean().to_numpy()
    dv = (sv - sv.shift(k)).fillna(0.0).to_numpy()
    return pd.DataFrame({"v": v, "i": i, "v_ma": v_ma, "dv_lag": dv})


def make_windows(X: np.ndarray, y: np.ndarray, length: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows (label = target at the last step). Returns (Xw, yw, end_index)."""
    ends = np.arange(length - 1, len(X), stride)
    Xw = np.stack([X[e - length + 1: e + 1] for e in ends])
    return Xw, y[ends], ends
