"""SOH and RUL experiments on the ageing datasets: leave-one-battery-out, cross-temperature, augmentation ablation.

Provenance is carried into every output row. SIMULATED (PyBaMM) results are never mixed with the EXPERIMENTAL_DERIVED ones.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from bms.data.aging_loader import audit_batteries, load_aging, load_aging_config, usable_batteries
from bms.evaluation.metrics import regression_metrics
from bms.features.aging_features import FEATURE_SETS, add_capacity_features, assert_no_leak
from bms.models.rul.aging import baseline_linear_extrapolation, baseline_mean_life
from bms.models.soh.aging import fit_model, predict_interval
from bms.utils.config import get_logger, repo_root, set_seed

log = get_logger(__name__)
MODELS = ["linear", "random_forest"]


def _split(df, train_ids, test_id, aug=None):
    tr = df[df.battery_id.isin(train_ids) & ~df.is_augmented]
    if aug is not None:
        tr = pd.concat([tr, aug[aug.battery_id.isin(train_ids)]], ignore_index=True)   # replicas of TRAINING batteries only
    te = df[(df.battery_id == test_id) & ~df.is_augmented]
    assert not set(tr.battery_id) & {test_id}, "test battery leaked into training"
    assert not te.is_augmented.any()
    return tr, te


def run_aging_experiments(out_dir=None, quick: bool = False) -> dict[str, Any]:
    set_seed(42); cfg = load_aging_config(); out_dir = out_dir or repo_root() / "results"
    eol, hmin = cfg["eol_fraction"], cfg["min_history_cycles"]
    data, audits = {}, {}
    for ds in ("NASA_FEATURES", "PYBAMM"):
        u, a = usable_batteries(load_aging(ds, cfg), cfg); audits[ds] = a
        data[ds] = add_capacity_features(u, eol)
    aug = load_aging("NASA_AUGMENTED", cfg); aug = aug[aug.is_augmented & aug.battery_id.isin(data["NASA_FEATURES"].battery_id.unique())]
    aug = add_capacity_features(aug, eol, use_file_rul=True)
    rows, preds = [], []

    def evaluate(ds, target, fs, model, protocol, train_ids, test_id, use_aug=False):
        df = data[ds]; feats = FEATURE_SETS[target][fs]; assert_no_leak(feats)
        tr, te = _split(df, train_ids, test_id, aug if use_aug else None)
        ycol = "soh_rel" if target == "soh" else "rul"
        te = te[te.cycle >= hmin] if target == "rul" else te
        tr = tr[tr.cycle >= 1].dropna(subset=[ycol])
        lo = hi = None; hp = {}
        if model.startswith("baseline"):
            lives = df[df.battery_id.isin(train_ids)].groupby("battery_id").eol_cycle.first().to_numpy()
            full = df[df.battery_id == test_id]
            p = (baseline_mean_life(lives, te.cycle.to_numpy()) if model == "baseline_mean_life"
                 else baseline_linear_extrapolation(full, lives)[full.cycle.isin(te.cycle).to_numpy()])
        else:
            pipe, hp = fit_model(model, tr[feats].to_numpy(float), tr[ycol].to_numpy(float), tr.battery_id.to_numpy(), 42, quick)
            p = pipe.predict(te[feats].to_numpy(float))
            if model == "random_forest" and target == "rul":
                lo, hi = predict_interval(pipe, te[feats].to_numpy(float))
        y = te[ycol].to_numpy(float); m = regression_metrics(y, p, scale=1.0)
        r = {"dataset": ds, "provenance": df.provenance.iloc[0], "target": target, "feature_set": fs, "model": model, "protocol": protocol,
             "train_batteries": "+".join(train_ids), "test_battery": test_id, "n_test": len(y), **m, "best_hp": json.dumps(hp, default=str)}
        if target == "rul":
            late = te.life_frac.to_numpy() >= 0.7                     # last 30 % of life (label used for EVALUATION only)
            r["MAE_last30pct_life"] = float(np.mean(np.abs(p - y)[late])) if late.any() else np.nan
            r["MAE_over_life_pct"] = 100 * float(np.mean(np.abs(p - y))) / float(te.eol_cycle.iloc[0])
            if lo is not None:
                r["coverage_90"] = float(np.mean((y >= lo) & (y <= hi)))
        rows.append(r)
        if fs in ("base", "full", "cycle_only", "cap_hist"):
            preds.append(pd.DataFrame({"dataset": ds, "provenance": r["provenance"], "target": target, "feature_set": fs, "model": model,
                                       "protocol": protocol, "test_battery": test_id, "cycle": te.cycle.to_numpy(), "life_frac": te.life_frac.to_numpy(),
                                       "y_true": y, "y_pred": p, "lo": lo if lo is not None else np.nan, "hi": hi if hi is not None else np.nan}))

    for ds in data:
        ids = sorted(data[ds].battery_id.unique())
        for target in ("soh", "rul"):
            models = MODELS + (["baseline_mean_life", "baseline_linear_extrap"] if target == "rul" else [])
            for fs in FEATURE_SETS[target]:
                log.info("%s | %s | feature set %s", ds, target, fs)
                for model in models:
                    if model.startswith("baseline") and fs != "cap_hist":
                        continue
                    for t in ids:
                        evaluate(ds, target, fs, model, "leave_one_battery_out", [b for b in ids if b != t], t)
    log.info("cross-temperature experiments")
    # cross-temperature on the NASA-style data (ambient labels from the file)
    a = audits["NASA_FEATURES"]; a = a[a.usable]; warm = sorted(a[a.ambient_c >= 20].battery_id); cold = sorted(a[a.ambient_c < 20].battery_id)
    for target, fs in (("soh", "base"), ("rul", "full"), ("rul", "cap_hist")):
        for model in MODELS:
            for train, test in ((warm, cold), (cold, warm)):
                for t in test:
                    evaluate("NASA_FEATURES", target, fs, model, f"cross_temperature_{'24to4' if train is warm else '4to24'}", train, t)
    log.info("augmentation ablation")
    # augmentation ablation (replicas of TRAINING batteries added; test = original data of the held-out battery)
    ids = sorted(data["NASA_FEATURES"].battery_id.unique())
    for target, fs in (("soh", "base"), ("rul", "full")):
        for model in MODELS:
            for t in ids:
                evaluate("NASA_FEATURES", target, fs, model, "lobo_with_augmented_training_rows", [b for b in ids if b != t], t, use_aug=True)
    res, pr = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True); (out_dir / "experiment_logs").mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "metrics" / "aging_results.csv", index=False)
    pr.to_csv(out_dir / "metrics" / "aging_predictions.csv.gz", index=False, compression="gzip")
    pd.concat([x.assign(dataset=k) for k, x in audits.items()]).to_csv(out_dir / "metrics" / "aging_battery_audit.csv", index=False)
    return {"results": res, "predictions": pr, "audits": audits}
