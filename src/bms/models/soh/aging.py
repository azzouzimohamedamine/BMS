"""Regression models for SOH / RUL on ageing datasets with leave-one-battery-out tuning (shared by SOH and RUL)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bms.preprocessing.transformers import Winsorizer

RF_GRID = {"n_estimators": [100], "max_depth": [4, None], "min_samples_leaf": [1, 5]}


def make_pipeline(name: str, hp: dict[str, Any] | None = None, seed: int = 42) -> Pipeline:
    """Winsorizer and scaler are fitted on training data only (inside ``fit``)."""
    if name == "linear":
        return Pipeline([("w", Winsorizer()), ("s", StandardScaler()), ("m", LinearRegression())])
    if name == "random_forest":
        return Pipeline([("w", Winsorizer()), ("m", RandomForestRegressor(random_state=seed, n_jobs=1, **(hp or {})))])
    raise ValueError(name)


def fit_model(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42, quick: bool = False
              ) -> tuple[Pipeline, dict[str, Any]]:
    """Linear: plain fit. Random Forest: grid search by inner leave-one-battery-out CV (MAE) on TRAINING batteries only."""
    if name == "linear":
        return make_pipeline(name).fit(X, y), {}
    grid = {k: v[:1] for k, v in RF_GRID.items()} if quick else RF_GRID
    best, best_hp = np.inf, None
    ids = np.unique(groups)
    for hp in ParameterGrid(grid):
        errs = []
        if len(ids) < 2:
            errs = [0.0]
        for g in ids if len(ids) > 1 else []:
            tr, va = groups != g, groups == g
            p = make_pipeline(name, hp, seed).fit(X[tr], y[tr])
            errs.append(float(np.mean(np.abs(p.predict(X[va]) - y[va]))))
        if np.mean(errs) < best:
            best, best_hp = float(np.mean(errs)), hp
    return make_pipeline(name, best_hp, seed).fit(X, y), {"best_hp": best_hp, "inner_cv_mae": best}


def predict_interval(pipe: Pipeline, X: np.ndarray, lo: float = 5, hi: float = 95) -> tuple[np.ndarray, np.ndarray]:
    """Spread of the individual tree predictions of a fitted RF pipeline (a heuristic, NOT a calibrated interval)."""
    Z = pipe.named_steps["w"].transform(X)
    P = np.stack([t.predict(Z) for t in pipe.named_steps["m"].estimators_])
    return np.percentile(P, lo, axis=0), np.percentile(P, hi, axis=0)
