"""Charge / discharge / rest identification and segment tables (no Arbin step index needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

REST, CHARGE, DISCHARGE = 0, 1, -1
STATE_NAMES = {REST: "rest", CHARGE: "charge", DISCHARGE: "discharge"}


def label_states(current_a, rest_threshold_a: float = 0.02) -> np.ndarray:
    """+1 charge, -1 discharge, 0 rest (discharge current is negative)."""
    i = np.asarray(current_a, dtype=float)
    s = np.zeros(len(i), dtype=int)
    s[i > rest_threshold_a] = CHARGE
    s[i < -rest_threshold_a] = DISCHARGE
    return s


def segment_table(df: pd.DataFrame, rest_threshold_a: float = 0.02, min_duration_s: float = 600.0) -> pd.DataFrame:
    """One row per contiguous segment; segments shorter than ``min_duration_s`` are absorbed by the previous one."""
    state = label_states(df["current_a"], rest_threshold_a)
    seg_id = np.cumsum(np.r_[True, state[1:] != state[:-1]])
    t = df["time_s"].to_numpy()
    raw = []
    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        raw.append({"state": int(state[idx[0]]), "i0": int(idx[0]), "i1": int(idx[-1])})
    merged: list[dict] = []
    for r in raw:
        dur = t[r["i1"]] - t[r["i0"]]
        if dur >= min_duration_s:
            if merged and merged[-1]["state"] == r["state"]:
                merged[-1]["i1"] = r["i1"]
            else:
                merged.append(dict(r))
        elif merged:
            merged[-1]["i1"] = r["i1"]
    out = pd.DataFrame(merged)
    out["t0"], out["t1"] = t[out["i0"]], t[out["i1"]]
    out["duration_s"] = out["t1"] - out["t0"]
    out["state_name"] = out["state"].map(STATE_NAMES)
    return out
