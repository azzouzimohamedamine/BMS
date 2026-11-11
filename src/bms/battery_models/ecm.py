"""First-order Thevenin equivalent-circuit model (ECM): identification from the data + OCV curve.

Sign convention inside this module: ``i_dis`` > 0 means discharge (opposite to the raw data convention).
    V = OCV(SOC) - R0*i_dis - V1,     dV1/dt = -V1/(R1*C1) + i_dis/C1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


@dataclass
class ECMParams:
    r0: float
    r1: float
    tau: float                 # R1*C1 in seconds
    soc_grid: np.ndarray
    ocv_grid: np.ndarray
    source: str = ""

    def ocv(self, soc):
        return np.interp(soc, self.soc_grid, self.ocv_grid)

    def docv(self, soc):
        d = np.gradient(self.ocv_grid, self.soc_grid)
        return np.interp(soc, self.soc_grid, d)


def identify_r0_rc(t_rest_before: np.ndarray, v_rest_before: np.ndarray, t_dis: np.ndarray, v_dis: np.ndarray,
                   i_dis: float, fit_window_s: float = 1200.0) -> tuple[float, float, float]:
    """R0, R1, tau from the rest->discharge onset (high-SOC region).

    R0 = instantaneous voltage step / current. R1, tau from a fit of the first ``fit_window_s`` seconds of
    V(t) = a - b*t - i*R1*(1-exp(-t/tau)), where the linear term absorbs the OCV drift (assumption).
    """
    v_pre = float(np.mean(v_rest_before[-10:]))
    t = t_dis - t_dis[0]
    r0 = max((v_pre - float(v_dis[0])) / i_dis, 1e-4)
    m = t <= fit_window_s
    tt, vv = t[m], v_dis[m]

    def res(p):
        a, b, r1, tau = p
        return a - b * tt - i_dis * r1 * (1 - np.exp(-tt / tau)) - vv

    sol = least_squares(res, x0=[vv[0], 1e-6, 0.02, 200.0], bounds=([3.0, 0, 1e-4, 10.0], [5.0, 1e-2, 1.0, 3000.0]))
    _, _, r1, tau = sol.x
    return float(r0), float(r1), float(tau)


def build_ocv_curve(soc: np.ndarray, v: np.ndarray, i_dis: np.ndarray, r_total: float, n: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-OCV from a low-rate discharge: V + i*R_total (R0+R1 steady state), binned, monotone, smoothed.

    This is an approximation (a true OCV needs interrupted/relaxed measurements); error ~ i*dR(SOC).
    """
    ocv_raw = v + i_dis * r_total
    grid = np.linspace(0, 1, n)
    edges = np.linspace(0, 1, n + 1)
    idx = np.clip(np.digitize(soc, edges) - 1, 0, n - 1)
    vals = np.array([np.median(ocv_raw[idx == k]) if np.any(idx == k) else np.nan for k in range(n)])
    ok = ~np.isnan(vals)
    vals = np.interp(grid, grid[ok], vals[ok])
    vals = savgol_filter(vals, 11, 2)
    vals = np.maximum.accumulate(vals)          # OCV must be non-decreasing in SOC
    return grid, vals


def identify_ecm(raw_t: np.ndarray, raw_v: np.ndarray, raw_i: np.ndarray, i0: int, soc_on_raw: np.ndarray,
                 i1: int, source: str = "") -> ECMParams:
    """Identify ECM parameters from raw 1-s data around one discharge segment [i0, i1] (raw indices)."""
    i_dis = float(-np.mean(raw_i[i0:i1 + 1]))
    r0, r1, tau = identify_r0_rc(raw_t[max(0, i0 - 600):i0], raw_v[max(0, i0 - 600):i0], raw_t[i0:i1 + 1], raw_v[i0:i1 + 1], i_dis)
    grid, ocv = build_ocv_curve(soc_on_raw, raw_v[i0:i1 + 1], -raw_i[i0:i1 + 1], r0 + r1)
    return ECMParams(r0, r1, tau, grid, ocv, source)
