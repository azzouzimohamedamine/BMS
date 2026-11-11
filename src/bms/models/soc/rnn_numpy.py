"""LSTM and GRU regressors implemented in pure NumPy (manual BPTT + Adam).

Why NumPy: no deep-learning framework was available in the execution environment (no network). The
implementation is validated by a finite-difference gradient check (tests/test_rnn_gradients.py).
Architecture: single recurrent layer -> linear head, sequence-to-one regression.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class RNNRegressor:
    def __init__(self, cell: str = "gru", n_in: int = 2, hidden: int = 16, seed: int = 0):
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        self.cell, self.H, self.D = cell, hidden, n_in
        g = 3 if cell == "gru" else 4
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden)
        self.p: dict[str, np.ndarray] = {
            "W": rng.uniform(-s, s, (n_in, g * hidden)), "U": rng.uniform(-s, s, (hidden, g * hidden)),
            "b": np.zeros(g * hidden), "Wo": rng.uniform(-s, s, (hidden, 1)), "bo": np.zeros(1)}
        if cell == "lstm":
            self.p["b"][hidden:2 * hidden] = 1.0   # forget-gate bias
        self.x_mean = np.zeros(n_in); self.x_std = np.ones(n_in)

    # ---------------------------------------------------------------- forward / backward
    def forward(self, X: np.ndarray):
        B, T, _ = X.shape
        H, p = self.H, self.p
        h = np.zeros((B, H)); c = np.zeros((B, H)); cache = []
        for t in range(T):
            x = X[:, t]
            if self.cell == "gru":
                a = x @ p["W"] + p["b"]
                zr = _sig(a[:, :2 * H] + h @ p["U"][:, :2 * H]); z, r = zr[:, :H], zr[:, H:]
                n = np.tanh(a[:, 2 * H:] + (r * h) @ p["U"][:, 2 * H:])
                cache.append((x, h, z, r, n)); h = (1 - z) * n + z * h
            else:
                a = x @ p["W"] + h @ p["U"] + p["b"]
                i_, f_, o_ = _sig(a[:, :H]), _sig(a[:, H:2 * H]), _sig(a[:, 3 * H:])
                g_ = np.tanh(a[:, 2 * H:3 * H])
                c_new = f_ * c + i_ * g_; tc = np.tanh(c_new)
                cache.append((x, h, c, i_, f_, g_, o_, tc)); c = c_new; h = o_ * tc
        y = (h @ p["Wo"] + p["bo"]).ravel()
        return y, (cache, h)

    def backward(self, dy: np.ndarray, state) -> dict[str, np.ndarray]:
        cache, h_last = state
        H, p = self.H, self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"] = h_last.T @ dy[:, None]; g["bo"] = np.array([dy.sum()])
        dh = dy[:, None] @ p["Wo"].T
        dc = np.zeros_like(dh)
        for t in reversed(range(len(cache))):
            if self.cell == "gru":
                x, h, z, r, n = cache[t]
                dn_pre = dh * (1 - z) * (1 - n ** 2)
                dz_pre = dh * (h - n) * z * (1 - z)
                d_rh = dn_pre @ p["U"][:, 2 * H:].T
                dr_pre = d_rh * h * r * (1 - r)
                dzr = np.concatenate([dz_pre, dr_pre], 1)
                g["W"] += x.T @ np.concatenate([dzr, dn_pre], 1); g["b"] += np.concatenate([dzr, dn_pre], 1).sum(0)
                g["U"][:, :2 * H] += h.T @ dzr; g["U"][:, 2 * H:] += (r * h).T @ dn_pre
                dh = dh * z + d_rh * r + dzr @ p["U"][:, :2 * H].T
            else:
                x, h, c_prev, i_, f_, g_, o_, tc = cache[t]
                do = dh * tc; dct = dc + dh * o_ * (1 - tc ** 2)
                dA = np.concatenate([dct * g_ * i_ * (1 - i_), dct * c_prev * f_ * (1 - f_),
                                     dct * i_ * (1 - g_ ** 2), do * o_ * (1 - o_)], 1)
                g["W"] += x.T @ dA; g["U"] += h.T @ dA; g["b"] += dA.sum(0)
                dh = dA @ p["U"].T; dc = dct * f_
        return g

    # ---------------------------------------------------------------- training / inference
    def predict(self, X: np.ndarray, batch: int = 2048) -> np.ndarray:
        Xs = (X - self.x_mean) / self.x_std
        return np.concatenate([self.forward(Xs[k:k + batch])[0] for k in range(0, len(Xs), batch)])

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, lr: float = 5e-3,
            batch: int = 128, epochs: int = 40, patience: int = 6, seed: int = 0, clip: float = 1.0) -> dict[str, Any]:
        """Adam + early stopping on a validation set; scaling statistics come from the TRAINING data only."""
        self.x_mean = X.reshape(-1, X.shape[-1]).mean(0); self.x_std = X.reshape(-1, X.shape[-1]).std(0) + 1e-8
        Xs, Xv = (X - self.x_mean) / self.x_std, (X_val - self.x_mean) / self.x_std
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}; v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        best, best_p, bad, step, hist = np.inf, None, 0, 0, []
        t0 = time.time()
        for ep in range(epochs):
            idx = rng.permutation(len(Xs))
            for k in range(0, len(idx), batch):
                b = idx[k:k + batch]
                yp, st = self.forward(Xs[b])
                grads = self.backward(2 * (yp - y[b]) / len(b), st)
                norm = np.sqrt(sum(float((gr ** 2).sum()) for gr in grads.values()))
                sc = min(1.0, clip / (norm + 1e-12)); step += 1
                for kk in self.p:
                    gr = grads[kk] * sc
                    m[kk] = 0.9 * m[kk] + 0.1 * gr; v2[kk] = 0.999 * v2[kk] + 0.001 * gr ** 2
                    self.p[kk] -= lr * (m[kk] / (1 - 0.9 ** step)) / (np.sqrt(v2[kk] / (1 - 0.999 ** step)) + 1e-8)
            val = float(np.mean(np.abs(self.forward(Xv)[0] - y_val)))
            hist.append(val)
            if val < best - 1e-5:
                best, bad, best_p = val, 0, {k: v.copy() for k, v in self.p.items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_p is not None:
            self.p = best_p
        return {"val_mae": best, "epochs_run": len(hist), "train_time_s": time.time() - t0, "val_history": hist}

    def save(self, path) -> None:
        np.savez(path, cell=self.cell, H=self.H, D=self.D, x_mean=self.x_mean, x_std=self.x_std, **self.p)

    @classmethod
    def load(cls, path) -> "RNNRegressor":
        z = np.load(path, allow_pickle=False)
        m = cls(str(z["cell"]), int(z["D"]), int(z["H"]))
        m.p = {k: z[k] for k in ("W", "U", "b", "Wo", "bo")}; m.x_mean, m.x_std = z["x_mean"], z["x_std"]
        return m
