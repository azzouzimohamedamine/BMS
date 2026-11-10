"""End-to-end preparation of one experimental condition for SOC work."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bms.data.loader import load_dataset
from bms.features.soc_reference import discharge_reference
from bms.preprocessing.cleaning import clean
from bms.preprocessing.segmentation import DISCHARGE, segment_table
from bms.utils.config import load_dataset_config


@dataclass
class ConditionData:
    """Uniformly resampled main-discharge data of one test + metadata."""
    dataset_id: str
    nominal_temp_c: float | None
    capacity_ah: float                 # delivered capacity of this discharge (to cut-off)
    mean_current_a: float
    ts_s: float
    frame: pd.DataFrame                # columns: t_s, v, i, soc_ref  (soc_ref in [0,1])
    meta: dict[str, Any] = field(default_factory=dict)


def resample_uniform(t: np.ndarray, cols: dict[str, np.ndarray], ts_s: float) -> pd.DataFrame:
    """Linear interpolation on a uniform grid (handles event-driven logging with gaps up to 10 s)."""
    grid = np.arange(t[0], t[-1], ts_s)
    out = {"t_s": grid - grid[0]}
    for k, v in cols.items():
        out[k] = np.interp(grid, t, v)
    return pd.DataFrame(out)


def prepare_condition(dataset_id: str, ts_s: float = 20.0, cfg: dict[str, Any] | None = None) -> ConditionData:
    cfg = cfg or load_dataset_config()
    entry = next(d for d in cfg["datasets"] if d["id"] == dataset_id)
    raw = clean(load_dataset(dataset_id, cfg=cfg), tuple(cfg["validity"]["voltage_range_v"]), cfg["validity"]["current_abs_max_a"])
    seg = segment_table(raw, cfg["segmentation"]["rest_current_threshold_a"], cfg["segmentation"]["min_segment_duration_s"])
    dis = seg[seg.state == DISCHARGE].sort_values("duration_s", ascending=False)
    if dis.empty:
        raise ValueError(f"{dataset_id}: no discharge segment found.")
    s = dis.iloc[0]
    d = raw.iloc[int(s.i0): int(s.i1) + 1]
    soc, q = discharge_reference(d["time_s"].to_numpy(), d["current_a"].to_numpy())
    frame = resample_uniform(d["time_s"].to_numpy(), {"v": d["voltage_v"].to_numpy(), "i": d["current_a"].to_numpy(),
                                                       "soc_ref": soc}, ts_s)
    pre = raw.iloc[max(0, int(s.i0) - 600): int(s.i0)]
    return ConditionData(dataset_id, entry.get("nominal_temperature_c"), q, float(d["current_a"].mean()), ts_s, frame,
                         {"v_start": float(d["voltage_v"].iloc[0]), "v_end": float(d["voltage_v"].iloc[-1]),
                          "v_rest_before": float(pre["voltage_v"].mean()) if len(pre) else float("nan"),
                          "duration_s": float(s.duration_s), "raw_slice": (int(s.i0), int(s.i1))})
