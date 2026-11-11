"""Tree-based SOC regressors with leakage-aware blocked cross-validation."""
from __future__ import annotations

import time
from typing import Any, Iterator

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bms.utils.config import get_logger

log = get_logger(__name__)

try:  # optional backend
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover - depends on environment
    XGBRegressor = None
    HAS_XGB = False


def blocked_splits(groups: np.ndarray, n_blocks: int, gap: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Blocked CV: within each group (= one test condition, ordered in time) cut ``n_blocks`` contiguous
    blocks; validation = block k of every group; training excludes ``gap`` samples around the block edges
    (so overlapping/auto-correlated samples never straddle the train/validation boundary)."""
    pos = np.zeros(len(groups), dtype=int); size = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; pos[idx] = np.arange(len(idx)); size[g] = len(idx)
    for k in range(n_blocks):
        val = np.zeros(len(groups), bool); excl = np.zeros(len(groups), bool)
        for g, n in size.items():
            lo, hi = int(n * k / n_blocks), int(n * (k + 1) / n_blocks)
            m = groups == g
            val |= m & (pos >= lo) & (pos < hi)
            excl |= m & (pos >= lo - gap) & (pos < hi + gap)
        yield np.where(~excl)[0], np.where(val)[0]


def _estimator(name: str, hp: dict[str, Any], seed: int):
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1, **hp)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **hp)
    if name == "xgboost":
        return XGBRegressor(random_state=seed, n_jobs=1, **hp)
    raise ValueError(name)


def tune_and_fit(name: str, grid: dict[str, list], X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 n_blocks: int, gap: int, seed: int) -> tuple[Pipeline, dict[str, Any]]:
    """Grid search with blocked CV (MAE); scaler fitted inside each fold on training data only."""
    best, best_hp, rows = np.inf, None, []
    for hp in ParameterGrid(grid):
        maes = []
        for tr, va in blocked_splits(groups, n_blocks, gap):
            pipe = Pipeline([("sc", StandardScaler()), ("m", _estimator(name, hp, seed))]).fit(X[tr], y[tr])
            maes.append(float(np.mean(np.abs(pipe.predict(X[va]) - y[va]))))
        rows.append({"hp": hp, "cv_mae": float(np.mean(maes))})
        if np.mean(maes) < best:
            best, best_hp = float(np.mean(maes)), hp
    t0 = time.time()
    final = Pipeline([("sc", StandardScaler()), ("m", _estimator(name, best_hp, seed))]).fit(X, y)
    return final, {"best_hp": best_hp, "cv_mae": best, "search": rows, "fit_time_s": time.time() - t0}
