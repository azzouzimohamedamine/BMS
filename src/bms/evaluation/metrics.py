"""Regression metrics (SOC reported in percentage points when ``scale=100``)."""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float = 100.0) -> dict[str, float]:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = (p - y) * scale
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "MaxAE": float(np.max(np.abs(err))),
            "Bias": float(np.mean(err))}
