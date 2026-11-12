"""Small scikit-learn transformers."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to [q_low, q_high] percentiles learned on the TRAINING data only (tames outliers such as dv_peak_height)."""

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low, self.q_high = q_low, q_high

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_, self.hi_ = np.quantile(X, self.q_low, axis=0), np.quantile(X, self.q_high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)
