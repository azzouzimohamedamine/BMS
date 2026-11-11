"""Diagnostic decision layer: turns detector outputs into an engineering-readable report."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bms.models.fault.detection import FEATS

INTERPRETATION = {
    "dv_max": ("abnormal voltage behaviour (fast voltage change)", "Check voltage-sense wiring / contact resistance and the protocol step transitions at this time."),
    "v_std": ("abnormal voltage variability", "Inspect voltage channel noise, connectors and possible loose contacts."),
    "i_std": ("abnormal current noise", "Inspect current-sensor/shunt, cycler channel calibration."),
    "i_dev": ("current deviating from the local set-point", "Verify cycler set-point, current-sensor offset/drift."),
    "di_max": ("abrupt current change (step or glitch)", "Check whether a protocol step transition occurred; otherwise inspect the current sensor."),
    "dt_max": ("sampling gap / data loss", "Check data-logger continuity and communication with the cycler."),
}


def severity(score: float, threshold: float) -> str:
    r = score / max(threshold, 1e-9)
    return "high" if r > 3 else "medium" if r > 1.7 else "low"


def diagnose(feats: pd.DataFrame, X: np.ndarray, scores: dict[str, np.ndarray], flags: dict[str, np.ndarray],
             thresholds: dict[str, float], suite_thr: ThresholdLike, provenance: str, transition_times: np.ndarray | None = None,
             max_rows: int = 20) -> pd.DataFrame:
    """One row per flagged window (flagged by >=1 detector), ranked by Isolation-Forest score."""
    any_flag = flags["iso"] | flags["ocsvm"] | flags["threshold"] | flags.get("flatline", np.zeros(len(feats), bool))
    rows: list[dict[str, Any]] = []
    for k in np.where(any_flag)[0]:
        z = np.abs(X[k] - suite_thr.med) / suite_thr.mad
        top = FEATS[int(np.argmax(z))]
        interp, action = INTERPRETATION[top]
        near_transition = transition_times is not None and np.any(np.abs(transition_times - feats.t_start.iloc[k]) < 120)
        if near_transition:
            interp = "probable protocol step transition (operational event, not a fault): " + interp
        n_det = int(flags["iso"][k]) + int(flags["ocsvm"][k]) + int(flags["threshold"][k])
        rows.append({"t_start_s": float(feats.t_start.iloc[k]), "relevant_measurement": top, "robust_z": float(z.max()),
                     "iso_score": float(scores["iso"][k]), "severity": severity(scores["iso"][k], thresholds["iso"]),
                     "confidence(detectors_agreeing/3)": n_det / 3, "interpretation": interp,
                     "recommended_investigation": action, "provenance": provenance})
    out = pd.DataFrame(rows)
    return out.sort_values("iso_score", ascending=False).head(max_rows).reset_index(drop=True) if len(out) else out


ThresholdLike = Any
