"""SOH (Module 2): what the data supports, and strict guards for what it does not.

With the four CALCE files (one cell, ONE cycle per condition, no ageing), a *degradation* SOH cannot be built:
there is no capacity-vs-cycle trajectory, and the nominal capacity is not verified. This module therefore
  * reports measured delivered capacities and an internal-resistance indicator per condition (NOT SOH),
  * refuses to compute SOH(%) unless a verified nominal capacity is supplied,
  * provides the cycle-wise regression pipeline, which raises ``InsufficientDataError`` for data like ours
    (it is exercised only by unit tests on explicitly synthetic arrays).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bms.utils.errors import InsufficientDataError

MIN_CYCLES_FOR_SOH_MODELS = 20


def soh_percent(capacity_ah: float, nominal_ah: float | None, nominal_verified: bool) -> float:
    """SOH(%) = Q / Q_nominal * 100 ; refuses to run without a VERIFIED nominal capacity."""
    if nominal_ah is None or not nominal_verified:
        raise InsufficientDataError("Nominal capacity is not verified from reliable documentation: SOH(%) is not computed.")
    return 100.0 * capacity_ah / nominal_ah


def condition_capacity_table(conditions: dict[str, Any], r0: dict[str, float]) -> pd.DataFrame:
    """Measured delivered capacity (to 2.5 V cut-off) and R0 indicator per test condition. Not an SOH table."""
    ref = conditions["T25C_lowI"].capacity_ah
    rows = []
    for d, c in conditions.items():
        rows.append({"dataset_id": d, "nominal_temp_c(label)": c.nominal_temp_c, "mean_current_a": round(c.mean_current_a, 4),
                     "delivered_capacity_ah": round(c.capacity_ah, 4),
                     "capacity_ratio_vs_T25C_lowI(%)": round(100 * c.capacity_ah / ref, 2),
                     "R0_onset_ohm": round(r0[d], 4), "is_SOH": False})
    return pd.DataFrame(rows)


def fit_cycle_soh_models(features: pd.DataFrame, capacity_ah: np.ndarray, nominal_ah: float, nominal_verified: bool,
                         seed: int = 42) -> dict[str, Any]:
    """Cycle-wise SOH regression (Linear / Random Forest) with chronological split. Needs >= 20 cycles."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(features) < MIN_CYCLES_FOR_SOH_MODELS:
        raise InsufficientDataError(f"Only {len(features)} cycles available; at least {MIN_CYCLES_FOR_SOH_MODELS} are required.")
    y = np.array([soh_percent(q, nominal_ah, nominal_verified) for q in capacity_ah])
    n_tr = int(0.7 * len(y)); n_va = int(0.15 * len(y))
    sl = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, None)}
    out: dict[str, Any] = {}
    for name, est in [("linear", LinearRegression()), ("random_forest", RandomForestRegressor(200, random_state=seed))]:
        m = make_pipeline(StandardScaler(), est).fit(features.iloc[sl["train"]], y[sl["train"]])
        p = m.predict(features.iloc[sl["test"]]); e = p - y[sl["test"]]
        out[name] = {"model": m, "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}
    return out
