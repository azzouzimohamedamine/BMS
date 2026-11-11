"""Extended Kalman Filter SOC estimator on a 1RC ECM (Plett-style, state = [SOC, V1])."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bms.battery_models.ecm import ECMParams


@dataclass
class EKFConfig:
    q_soc: float = 1e-9        # process noise variance of SOC per step
    q_v1: float = 1e-7
    r_meas: float = (5e-3) ** 2  # voltage measurement noise variance (V^2)
    p0_soc: float = 0.05 ** 2
    p0_v1: float = 1e-4


def run_ekf(t_s: np.ndarray, v: np.ndarray, i_a: np.ndarray, params: ECMParams, capacity_ah: float,
            soc0: float, cfg: EKFConfig | None = None) -> np.ndarray:
    """Return the SOC trajectory (fraction). ``i_a`` uses the raw convention (discharge < 0)."""
    cfg = cfg or EKFConfig()
    n = len(v)
    soc_hat = np.empty(n)
    x = np.array([soc0, 0.0])
    P = np.diag([cfg.p0_soc, cfg.p0_v1])
    Qn = np.diag([cfg.q_soc, cfg.q_v1])
    soc_hat[0] = x[0]
    for k in range(1, n):
        dt = t_s[k] - t_s[k - 1]
        i_dis = -0.5 * (i_a[k] + i_a[k - 1])
        a = np.exp(-dt / params.tau)
        # prediction
        x = np.array([x[0] - i_dis * dt / (3600.0 * capacity_ah), a * x[1] + params.r1 * (1 - a) * i_dis])
        F = np.diag([1.0, a])
        P = F @ P @ F.T + Qn
        # update
        soc_c = float(np.clip(x[0], 0.0, 1.0))
        y_pred = float(params.ocv(soc_c)) - params.r0 * (-i_a[k]) - x[1]
        H = np.array([[float(params.docv(soc_c)), -1.0]])
        S = float((H @ P @ H.T)[0, 0]) + cfg.r_meas
        K = (P @ H.T) / S
        x = x + (K * (v[k] - y_pred)).ravel()
        P = (np.eye(2) - K @ H) @ P
        soc_hat[k] = x[0]
    return soc_hat
