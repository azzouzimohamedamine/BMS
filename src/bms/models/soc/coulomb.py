"""Coulomb-counting SOC baseline."""
from __future__ import annotations

import numpy as np


def coulomb_counting(t_s: np.ndarray, i_a: np.ndarray, capacity_ah: float, soc0: float) -> np.ndarray:
    """SOC(t) = SOC0 + (1/Q) * integral(I dt); I>0 charge. ``capacity_ah`` is a PRIOR calibrated capacity,
    not the capacity delivered in the evaluated test (otherwise the result would be circular)."""
    dq = np.r_[0.0, np.cumsum(0.5 * (i_a[1:] + i_a[:-1]) * np.diff(t_s))] / 3600.0
    return soc0 + dq / capacity_ah
