"""Fault/anomaly detection (Module 3): windowed features, threshold detector, IsolationForest, One-Class SVM.

Without fault labels in CALCE data, detections on real data are *statistical anomalies*, NOT validated faults.
Artificial faults (``simulator.py``) are SIMULATED and only used to benchmark the detectors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

FEATS = ["v_std", "dv_max", "i_std", "i_dev", "di_max", "dt_max"]


def window_features(t: np.ndarray, v: np.ndarray, i: np.ndarray, win_s: float = 60.0, hop_s: float = 30.0) -> pd.DataFrame:
    """Operating-point-independent window features computed on raw samples (t in s)."""
    t, v, i = np.asarray(t, float), np.asarray(v, float), np.asarray(i, float)
    starts = np.arange(t[0], t[-1] - win_s, hop_s)
    lo, hi = np.searchsorted(t, starts), np.searchsorted(t, starts + win_s)
    rows = []
    for a, b in zip(lo, hi):
        if b - a < 5:
            rows.append((np.nan,) * 6 + (t[a], np.nan)); continue
        tv, vv, ii = t[a:b], v[a:b], i[a:b]
        dt = np.diff(tv); dt[dt == 0] = np.nan
        rows.append((vv.std(), np.nanmax(np.abs(np.diff(vv)) / dt), ii.std(), ii.mean(), np.abs(np.diff(ii)).max(), dt[~np.isnan(dt)].max(initial=0.0), tv[0], abs(ii.mean())))
    df = pd.DataFrame(rows, columns=FEATS + ["t_start", "i_abs"])
    # current deviation from the local set-point (rolling median over ~1 h of windows)
    df["i_dev"] = (df["i_dev"] - df["i_dev"].rolling(121, center=True, min_periods=5).median()).abs()
    return df.dropna().reset_index(drop=True)


def flatline_flags(feats: pd.DataFrame, min_run: int = 6, i_min_a: float = 0.05) -> np.ndarray:
    """Persistence rule for a frozen voltage channel: >= ``min_run`` consecutive windows with zero voltage
    variation while the cell carries current (single zero-variation windows are normal at 1 mV resolution)."""
    flat = ((feats["v_std"] < 1e-4) & (feats["i_abs"] > i_min_a)).to_numpy()
    out = np.zeros(len(flat), bool); k = 0
    while k < len(flat):
        if flat[k]:
            j = k
            while j < len(flat) and flat[j]:
                j += 1
            if j - k >= min_run:
                out[k:j] = True
            k = j
        else:
            k += 1
    return out


class ThresholdDetector:
    """Robust z-score rule detector: flag a window if any feature exceeds ``k`` MAD-scaled deviations."""

    def __init__(self, k: float = 10.0):
        self.k = k

    def fit(self, X: np.ndarray) -> "ThresholdDetector":
        self.med = np.median(X, 0)
        mad = np.median(np.abs(X - self.med), 0) * 1.4826
        spread = np.quantile(X, 0.99, axis=0) - np.quantile(X, 0.01, axis=0)   # floor: avoids degenerate scales
        self.mad = np.maximum(mad, 0.05 * spread) + 1e-12
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.max(np.abs(X - self.med) / self.mad, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.k


class UnsupervisedSuite:
    """IsolationForest + One-Class SVM + threshold detector, scaler/thresholds fitted on training windows only."""

    def __init__(self, seed: int = 42, nu: float = 0.01, quantile: float = 0.995, k: float = 10.0):
        self.seed, self.nu, self.q, self.k = seed, nu, quantile, k

    def fit(self, X: np.ndarray) -> "UnsupervisedSuite":
        self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed).fit(Z)
        rng = np.random.default_rng(self.seed); sub = Z[rng.choice(len(Z), min(3000, len(Z)), replace=False)]
        self.ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=self.nu).fit(sub)
        self.thr = ThresholdDetector(self.k).fit(X)
        sc0 = self.scores(X)
        self.th = {n: float(np.quantile(sc0[n], self.q)) for n in ("iso", "ocsvm", "threshold")}   # 99.5th pct of TRAINING scores
        return self

    def scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Z = self.sc.transform(X)
        return {"iso": -self.iso.score_samples(Z), "ocsvm": -self.ocsvm.score_samples(Z), "threshold": self.thr.score(X)}

    def flags(self, X: np.ndarray) -> dict[str, np.ndarray]:
        s = self.scores(X)
        return {"iso": s["iso"] > self.th["iso"], "ocsvm": s["ocsvm"] > self.th["ocsvm"], "threshold": s["threshold"] > self.th["threshold"]}
