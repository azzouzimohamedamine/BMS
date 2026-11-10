"""Coulomb-counting integral and the *calculated* SOC reference.

IMPORTANT (scientific honesty): the CALCE files contain no independently measured SOC. The reference used
here is a CALCULATED quantity: SOC_ref(t) = 1 - Q_dis(t)/Q_dis,total, where Q_dis(t) is the charge removed since
the start of the low-rate discharge and Q_dis,total is the charge delivered until the 2.5 V cut-off.
100 % = state at the beginning of the discharge (after the rest period), 0 % = voltage cut-off of THAT test.
It is therefore test-specific (rate- and temperature-dependent), is affected by current-sensor error, and must
never be presented as a measured ground truth.
"""
from __future__ import annotations

import numpy as np


def cumulative_ah(t_s: np.ndarray, i_a: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of current in Ah (sign preserved; charge > 0)."""
    t, i = np.asarray(t_s, float), np.asarray(i_a, float)
    dq = np.r_[0.0, np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))] / 3600.0
    return dq


def discharge_reference(t_s: np.ndarray, i_a: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (SOC_ref in [0,1], delivered capacity in Ah) for ONE discharge segment (current < 0)."""
    q_removed = -cumulative_ah(t_s, i_a)
    q_total = float(q_removed[-1])
    if q_total <= 0:
        raise ValueError("Segment does not discharge the cell; cannot build a discharge SOC reference.")
    return 1.0 - q_removed / q_total, q_total
