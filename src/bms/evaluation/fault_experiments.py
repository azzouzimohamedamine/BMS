"""Module 3 experiments: (a) statistical anomalies on REAL data, (b) detector benchmark on SIMULATED faults."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from bms.data.loader import dataset_ids, load_dataset
from bms.models.fault.detection import FEATS, UnsupervisedSuite, flatline_flags, window_features
from bms.models.fault.diagnosis import diagnose
from bms.models.fault.simulator import FAULT_TYPES, inject
from bms.preprocessing.cleaning import clean
from bms.preprocessing.segmentation import segment_table
from bms.utils.config import get_logger, load_dataset_config, repo_root, set_seed

log = get_logger(__name__)
WIN, HOP = 60.0, 30.0


def _raw(d: str, cfg):
    return clean(load_dataset(d, cfg=cfg), tuple(cfg["validity"]["voltage_range_v"]), cfg["validity"]["current_abs_max_a"])


def run_fault_experiments(out_dir=None) -> dict[str, Any]:
    set_seed(42); cfg = load_dataset_config(); root = repo_root(); out_dir = out_dir or root / "results"
    raws = {d: _raw(d, cfg) for d in dataset_ids(cfg)}
    feats = {d: window_features(r.time_s.to_numpy(), r.voltage_v.to_numpy(), r.current_a.to_numpy(), WIN, HOP) for d, r in raws.items()}
    train_ids = ["T00C_lowI", "T45C_lowI"]               # 25 C and 1 A are kept unseen
    suite = UnsupervisedSuite().fit(np.vstack([feats[d][FEATS].to_numpy() for d in train_ids]))
    summary, diag_all = [], []
    for d in raws:                                          # (a) real data: statistical anomalies only
        X = feats[d][FEATS].to_numpy(); fl = suite.flags(X); fl["flatline"] = flatline_flags(feats[d]); sc = suite.scores(X)
        seg = segment_table(raws[d], cfg["segmentation"]["rest_current_threshold_a"], cfg["segmentation"]["min_segment_duration_s"])
        trans = np.r_[seg.t0.to_numpy(), seg.t1.to_numpy()] - raws[d].time_s.iloc[0]
        summary.append({"dataset": d, "n_windows": len(X), "used_for_training": d in train_ids,
                        **{f"flagged_{k}": int(v.sum()) for k, v in fl.items()}, "flagged_any": int((fl["iso"] | fl["ocsvm"] | fl["threshold"] | fl["flatline"]).sum())})
        dg = diagnose(feats[d].assign(t_start=feats[d].t_start - raws[d].time_s.iloc[0]), X, sc, fl, suite.th, suite.thr,
                      "statistical anomaly on real data (not a validated fault)", trans)
        if len(dg):
            diag_all.append(dg.assign(dataset=d))
    # (b) simulated faults injected into the unseen 25 C test
    r = raws["T25C_lowI"]; t0 = r.time_s.to_numpy(); v0, i0 = r.voltage_v.to_numpy(), r.current_a.to_numpy()
    span = t0[-1] - t0[0]; starts = [t0[0] + f * span for f in (0.2, 0.4, 0.6, 0.8)]
    rows = []
    base = window_features(t0, v0, i0, WIN, HOP)
    for kind in FAULT_TYPES:
        for mag in (0.25, 0.5, 1.0, 2.0):
            tt, vv, ii, mask = t0, v0, i0, np.zeros(len(t0), bool)
            fault_iv = []
            for s in starts:
                tt, vv, ii, m = inject(tt, vv, ii, kind, s, 300.0, mag)
                fault_iv.append((s, s + 300.0))
            F = window_features(tt, vv, ii, WIN, HOP)
            lab = np.zeros(len(F), bool)
            for a, b in fault_iv:
                lab |= (F.t_start.to_numpy() < b) & (F.t_start.to_numpy() + WIN > a)
            X = F[FEATS].to_numpy(); fl = suite.flags(X); fl["flatline"] = flatline_flags(F)
            for name, f in list(fl.items()) + [("any", fl["iso"] | fl["ocsvm"] | fl["threshold"] | fl["flatline"])]:
                tp = int((f & lab).sum()); fp = int((f & ~lab).sum())
                ev = np.mean([bool(np.any(f[(F.t_start.to_numpy() < b) & (F.t_start.to_numpy() + WIN > a)])) for a, b in fault_iv])
                rows.append({"fault_type": kind, "magnitude": mag, "detector": name, "window_recall": tp / max(lab.sum(), 1),
                             "window_precision": tp / max(tp + fp, 1), "event_detection_rate": float(ev), "false_positive_windows": fp,
                             "n_fault_windows": int(lab.sum()), "n_windows": len(F), "provenance": "SIMULATED fault injected into real T25C_lowI signals"})
    base_fl = suite.flags(base[FEATS].to_numpy()); base_fl["flatline"] = flatline_flags(base)
    out = {"real": pd.DataFrame(summary), "sim": pd.DataFrame(rows),
           "diagnostics": pd.concat(diag_all, ignore_index=True) if diag_all else pd.DataFrame(),
           "baseline_25C_flag_rate": {k: float(v.mean()) for k, v in base_fl.items()}}
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    out["real"].to_csv(out_dir / "metrics" / "fault_real_data_summary.csv", index=False)
    out["sim"].to_csv(out_dir / "metrics" / "fault_simulated_benchmark.csv", index=False)
    out["diagnostics"].to_csv(out_dir / "metrics" / "fault_diagnostics_real.csv", index=False)
    with open(out_dir / "experiment_logs" / "fault_experiments.json", "w", encoding="utf-8") as fh:
        json.dump({"thresholds": {k: float(v) for k, v in suite.th.items()},  "train_ids": train_ids,
                   "baseline_25C_flag_rate": out["baseline_25C_flag_rate"], "window_s": WIN, "hop_s": HOP}, fh, indent=2)
    return out
