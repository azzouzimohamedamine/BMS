"""Unified BMS engine: runs SOC / SOH / fault / RUL layers and tags every output with its provenance.

Provenance tags: MEASURED (instrument data), CALCULATED (reference derived from measurements),
ESTIMATED (model output), SIMULATED (synthetic perturbation / injected fault), UNAVAILABLE (insufficient data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bms.evaluation.soc_experiments import ecm_for
from bms.features.engineering import build_features
from bms.models.fault.detection import FEATS, UnsupervisedSuite, flatline_flags, window_features
from bms.models.fault.diagnosis import diagnose
from bms.models.rul.rul import assert_rul_feasible
from bms.models.soc.coulomb import coulomb_counting
from bms.models.soc.ekf import EKFConfig, run_ekf
from bms.models.soh.capacity import soh_percent
from bms.preprocessing.cleaning import clean
from bms.preprocessing.pipeline import ConditionData, prepare_condition
from bms.preprocessing.segmentation import segment_table
from bms.data.loader import load_dataset
from bms.utils.config import load_dataset_config, load_model_config, repo_root
from bms.utils.errors import InsufficientDataError

MEASURED, CALCULATED, ESTIMATED, SIMULATED, UNAVAILABLE = "MEASURED", "CALCULATED", "ESTIMATED", "SIMULATED", "UNAVAILABLE"


@dataclass
class BMSReport:
    dataset_id: str
    measured: dict[str, Any] = field(default_factory=dict)
    soc: dict[str, Any] = field(default_factory=dict)
    soh: dict[str, Any] = field(default_factory=dict)
    fault: dict[str, Any] = field(default_factory=dict)
    rul: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class BMSEngine:
    """Reference condition (capacity + ECM) is the 25 C low-current test, as in the SOC experiments."""

    def __init__(self) -> None:
        self.cfg = load_dataset_config(); self.mc = load_model_config(); self.root = repo_root()
        self.ref_id = self.mc["soc"]["reference_condition"]
        self.ref = prepare_condition(self.ref_id, self.mc["soc"]["ts_s"], self.cfg)
        self.ecm = ecm_for(self.ref_id, self.ref, self.cfg)
        e = self.mc["soc"]["ekf"]
        self.ekf_cfg = EKFConfig(e["q_soc"], e["q_v1"], e["r_meas_v"] ** 2, e["p0_soc"])
        self._suite: UnsupervisedSuite | None = None

    # ------------------------------------------------------------------ SOC
    def soc_estimates(self, d: str, init_soc: float = 1.0, models: tuple[str, ...] = ("random_forest", "hist_gradient_boosting")) -> pd.DataFrame:
        c = prepare_condition(d, self.mc["soc"]["ts_s"], self.cfg); f = c.frame
        out = pd.DataFrame({"t_s": f.t_s, "v": f.v, "i": f.i, "soc_reference": f.soc_ref})
        out["coulomb"] = coulomb_counting(f.t_s.to_numpy(), f.i.to_numpy(), self.ref.capacity_ah, init_soc)
        out["ekf"] = run_ekf(f.t_s.to_numpy(), f.v.to_numpy(), f.i.to_numpy(), self.ecm, self.ref.capacity_ah, init_soc, self.ekf_cfg)
        X = build_features(f.v.to_numpy(), f.i.to_numpy(), c.ts_s, self.mc["soc"]["feature_lag_s"]).to_numpy()
        for m in models:
            p = self.root / "models" / f"soc_{m}__{self.ref_id}.joblib"
            out[m] = joblib.load(p).predict(X) if p.exists() else np.nan
        return out

    # ------------------------------------------------------------------ faults
    def _fit_suite(self) -> UnsupervisedSuite:
        if self._suite is None:
            F = []
            for d in ("T00C_lowI", "T45C_lowI"):
                r = clean(load_dataset(d, cfg=self.cfg))
                F.append(window_features(r.time_s.to_numpy(), r.voltage_v.to_numpy(), r.current_a.to_numpy())[FEATS].to_numpy())
            self._suite = UnsupervisedSuite().fit(np.vstack(F))
        return self._suite

    def fault_report(self, d: str) -> pd.DataFrame:
        suite = self._fit_suite(); r = clean(load_dataset(d, cfg=self.cfg)); t0 = r.time_s.iloc[0]
        F = window_features(r.time_s.to_numpy(), r.voltage_v.to_numpy(), r.current_a.to_numpy()); X = F[FEATS].to_numpy()
        fl = suite.flags(X); fl["flatline"] = flatline_flags(F)
        seg = segment_table(r, self.cfg["segmentation"]["rest_current_threshold_a"], self.cfg["segmentation"]["min_segment_duration_s"])
        trans = np.r_[seg.t0.to_numpy(), seg.t1.to_numpy()] - t0
        return diagnose(F.assign(t_start=F.t_start - t0), X, suite.scores(X), fl, suite.th, suite.thr,
                        "statistical anomaly on real data (not a validated fault)", trans)

    # ------------------------------------------------------------------ unified report
    def analyze(self, d: str) -> BMSReport:
        c: ConditionData = prepare_condition(d, self.mc["soc"]["ts_s"], self.cfg)
        rep = BMSReport(d)
        rep.measured = {"tag": MEASURED, "delivered_capacity_ah": c.capacity_ah, "mean_current_a": c.mean_current_a,
                        "v_start": c.meta["v_start"], "v_end": c.meta["v_end"], "duration_h": c.meta["duration_s"] / 3600,
                        "temperature": "nominal label only (file name); not measured" if c.nominal_temp_c is not None else "unknown",
                        "nominal_temp_label_c": c.nominal_temp_c, "n_cycles_in_file": 1}
        est = self.soc_estimates(d)
        m = {k: float(np.mean(np.abs(est[k] - est.soc_reference)) * 100) for k in ("coulomb", "ekf")}
        rep.soc = {"tag": ESTIMATED, "reference_tag": CALCULATED, "mae_pts_vs_calculated_reference": m}
        try:
            rep.soh = {"tag": ESTIMATED, "soh_percent": soh_percent(c.capacity_ah, self.cfg["cell"]["nominal_capacity_ah"], self.cfg["cell"]["nominal_capacity_verified"])}
        except InsufficientDataError as e:
            rep.soh = {"tag": UNAVAILABLE, "reason": str(e)}; rep.warnings.append(f"SOH: {e}")
        fr = self.fault_report(d)
        rep.fault = {"tag": "STATISTICAL ANOMALY (not validated fault)", "n_reported": int(len(fr)), "top": fr.head(5).to_dict("records"),
                     "unavailable": ["over-temperature (no temperature channel)", "cell imbalance (single-cell data)", "abnormal internal resistance trend (one cycle)"]}
        rep.warnings.append("Fault: temperature and cell-level measurements are absent; corresponding diagnostics are unavailable.")
        try:
            assert_rul_feasible(np.array([1]), np.array([c.capacity_ah]), self.cfg["cell"]["nominal_capacity_ah"], self.cfg["cell"]["nominal_capacity_verified"])
        except InsufficientDataError as e:
            rep.rul = {"tag": UNAVAILABLE, "reason": str(e)}; rep.warnings.append(f"RUL: {e}")
        return rep
