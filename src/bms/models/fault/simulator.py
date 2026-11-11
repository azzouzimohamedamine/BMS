"""SIMULATED fault injection on top of real signals. Results obtained with it are NOT experimental faults."""
from __future__ import annotations

import numpy as np

FAULT_TYPES = ["voltage_spike", "current_sensor_offset", "voltage_sensor_stuck", "resistance_increase", "sample_dropout"]


def inject(t: np.ndarray, v: np.ndarray, i: np.ndarray, kind: str, start_s: float, dur_s: float, magnitude: float = 1.0
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, v, i, mask) where mask marks samples inside the injected fault (ground truth of the SIMULATION)."""
    t, v, i = t.copy(), v.copy(), i.copy()
    m = (t >= start_s) & (t < start_s + dur_s)
    if kind == "voltage_spike":
        v[m] += 0.15 * magnitude * (np.arange(m.sum()) % 2)
    elif kind == "current_sensor_offset":
        i[m] += 0.05 * magnitude
    elif kind == "voltage_sensor_stuck":
        v[m] = v[np.argmax(m)]
    elif kind == "resistance_increase":
        v[m] += 0.5 * magnitude * i[m]           # extra ohmic drop (R_extra = 0.5 ohm * magnitude)
    elif kind == "sample_dropout":
        keep = ~(m & (np.arange(len(t)) % 20 != 0))
        return t[keep], v[keep], i[keep], m[keep]
    else:
        raise ValueError(kind)
    return t, v, i, m
