"""Finite-difference gradient check of the NumPy LSTM/GRU (validates the manual BPTT)."""
import numpy as np
import pytest

from bms.models.soc.rnn_numpy import RNNRegressor


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_gradients_match_finite_differences(cell):
    rng = np.random.default_rng(0)
    m = RNNRegressor(cell, n_in=3, hidden=5, seed=1)
    X, y = rng.normal(size=(4, 6, 3)), rng.normal(size=4)

    def loss():
        return float(np.mean((m.forward(X)[0] - y) ** 2))

    yp, st = m.forward(X)
    grads = m.backward(2 * (yp - y) / len(y), st)
    eps = 1e-6
    for name, arr in m.p.items():
        flat_idx = rng.choice(arr.size, size=min(6, arr.size), replace=False)
        for fi in flat_idx:
            idx = np.unravel_index(fi, arr.shape)
            old = arr[idx]
            arr[idx] = old + eps; lp = loss()
            arr[idx] = old - eps; lm = loss()
            arr[idx] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[name][idx]) < 1e-6 + 1e-4 * abs(num), (cell, name, idx, num, grads[name][idx])
