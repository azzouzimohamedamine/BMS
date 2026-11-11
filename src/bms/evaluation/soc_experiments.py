"""SOC experiment protocols: cross-temperature / cross-rate generalisation, noise robustness, timing."""
from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bms.battery_models.ecm import ECMParams, identify_ecm
from bms.data.loader import load_dataset
from bms.evaluation.metrics import regression_metrics
from bms.features.engineering import build_features, make_windows
from bms.features.soc_reference import discharge_reference
from bms.models.soc import ml
from bms.models.soc.coulomb import coulomb_counting
from bms.models.soc.ekf import EKFConfig, run_ekf
from bms.models.soc.rnn_numpy import RNNRegressor
from bms.preprocessing.cleaning import clean
from bms.preprocessing.pipeline import ConditionData, prepare_condition
from bms.utils.config import get_logger, load_dataset_config, load_model_config, repo_root, set_seed

log = get_logger(__name__)
LOW_I = ["T00C_lowI", "T25C_lowI", "T45C_lowI"]
# (train conditions, test condition). All test conditions are UNSEEN by the trained models.
PROTOCOLS: list[tuple[tuple[str, ...], str]] = [
    (("T25C_lowI",), "T00C_lowI"), (("T25C_lowI",), "T45C_lowI"), (("T25C_lowI",), "INIT_1A"),
    (("T25C_lowI", "T45C_lowI"), "T00C_lowI"), (("T00C_lowI", "T25C_lowI"), "T45C_lowI"),
    (("T00C_lowI", "T45C_lowI"), "T25C_lowI"), (("T00C_lowI", "T25C_lowI", "T45C_lowI"), "INIT_1A"),
]


def add_noise(v, i, nl: dict, rng) -> tuple[np.ndarray, np.ndarray]:
    """SIMULATED measurement perturbation applied to real signals."""
    return (v + rng.normal(0, nl["sigma_v"], len(v)) if nl["sigma_v"] else v,
            i + nl["bias_i"] + (rng.normal(0, nl["sigma_i"], len(i)) if nl["sigma_i"] else 0.0))


def ecm_for(dataset_id: str, cd: ConditionData, cfg=None) -> ECMParams:
    """Identify the ECM (R0, R1, tau, pseudo-OCV) from one condition's raw 1-s data."""
    cfg = cfg or load_dataset_config()
    raw = clean(load_dataset(dataset_id, cfg=cfg), tuple(cfg["validity"]["voltage_range_v"]), cfg["validity"]["current_abs_max_a"])
    i0, i1 = cd.meta["raw_slice"]
    t = raw.time_s.to_numpy()
    soc, _ = discharge_reference(t[i0:i1 + 1], raw.current_a.to_numpy()[i0:i1 + 1])
    return identify_ecm(t, raw.voltage_v.to_numpy(), raw.current_a.to_numpy(), i0, soc, i1, source=dataset_id)


def _timed(fn, repeats: int = 3) -> tuple[Any, float]:
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    return out, best


def run_soc_experiments(out_dir=None, quick: bool = False) -> dict[str, Any]:
    mc = load_model_config(); sc = mc["soc"]; seed = mc["seed"]; set_seed(seed)
    root = repo_root(); out_dir = out_dir or root / "results"
    ts, L, lag = sc["ts_s"], sc["window_len"], sc["feature_lag_s"]
    conds = {d: prepare_condition(d, ts) for d in LOW_I + ["INIT_1A"]}
    ref = sc["reference_condition"]; q_cal = conds[ref].capacity_ah
    ecm_ref = ecm_for(ref, conds[ref])
    ekf_cfg = EKFConfig(sc["ekf"]["q_soc"], sc["ekf"]["q_v1"], sc["ekf"]["r_meas_v"] ** 2, sc["ekf"]["p0_soc"])
    log.info("Calibrated capacity (from %s) = %.4f Ah ; ECM R0=%.4f R1=%.4f tau=%.0f s", ref, q_cal, ecm_ref.r0, ecm_ref.r1, ecm_ref.tau)

    def arrays(d, noise=None, rng=None):
        f = conds[d].frame
        v, i = f.v.to_numpy(), f.i.to_numpy()
        if noise is not None:
            v, i = add_noise(v, i, noise, rng)
        return f.t_s.to_numpy(), v, i, f.soc_ref.to_numpy()

    def tab(d, noise=None, rng=None):           # feature table, trimmed to samples that have a full window
        t, v, i, y = arrays(d, noise, rng)
        X = build_features(v, i, ts, lag).to_numpy()
        return X[L - 1:], y[L - 1:], np.c_[v, i], t[L - 1:]

    rows, preds, logs = [], [], {"protocols": [], "ecm_reference": {"r0": ecm_ref.r0, "r1": ecm_ref.r1, "tau_s": ecm_ref.tau},
                                 "q_cal_ah": q_cal, "xgboost_available": ml.HAS_XGB}
    post = int(1800 / ts)

    def record(proto, train, test, model, nl, y, p, t, infer_ms, extra=None):
        m = regression_metrics(y, p); m2 = regression_metrics(y[post:], p[post:])
        rows.append({"protocol": proto, "train": "+".join(train), "test": test, "model": model, "noise": nl["name"], **m,
                     "MAE_after_30min": m2["MAE"], "MaxAE_after_30min": m2["MaxAE"], "infer_ms_per_1k": infer_ms, **(extra or {})})
        if nl["name"] == "clean":
            preds.append(pd.DataFrame({"protocol": proto, "train": "+".join(train), "test": test, "model": model, "t_s": t[::3], "soc_ref": y[::3], "soc_pred": p[::3]}))

    # ---------------------------------------------------------------- physics-based baselines (reference-condition protocol)
    for test in [c for c in conds if c != ref]:
        for nl in sc["noise_levels"]:
            rng = np.random.default_rng(seed)
            t, v, i, y = arrays(test, nl, rng); sl = slice(L - 1, None)
            for init, tag in [(1.0, ""), (0.7, "_init70")]:
                for name, fn in [("coulomb" + tag, lambda: coulomb_counting(t, i, q_cal, init)),
                                 ("ekf" + tag, lambda: run_ekf(t, v, i, ecm_ref, q_cal, init, ekf_cfg))]:
                    p, dt = _timed(fn, 1 if "ekf" in name else 3)
                    record("physics_" + ref, (ref,), test, name, nl, y[sl], p[sl], t[sl], 1e6 * dt / len(t))
    for test in conds:                                           # matched (optimistic) EKF
        e = ecm_for(test, conds[test]); nl = sc["noise_levels"][0]
        t, v, i, y = arrays(test); sl = slice(L - 1, None)
        p, dt = _timed(lambda: run_ekf(t, v, i, e, conds[test].capacity_ah, 1.0, ekf_cfg), 1)
        record("ekf_matched_own_params", (test,), test, "ekf_matched", nl, y[sl], p[sl], t[sl], 1e6 * dt / len(t))

    # ---------------------------------------------------------------- data-driven models
    model_dir = root / "models"; model_dir.mkdir(exist_ok=True)
    names = ["random_forest", "hist_gradient_boosting"] + (["xgboost"] if ml.HAS_XGB else [])
    cache: dict[tuple, dict] = {}
    for train, test in PROTOCOLS:
        if train not in cache:
            Xs, ys, gs, Ws, Ys = [], [], [], [], []
            for gi, d in enumerate(train):
                X, y, raw2, _ = tab(d); Xs.append(X); ys.append(y); gs.append(np.full(len(y), gi))
                Wd, yd, _ = make_windows(raw2, np.r_[np.zeros(L - 1), y], L, 1); Ws.append(Wd); Ys.append(yd)
            X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
            fitted, info = {}, {}
            for n in names:
                grid = sc["random_forest"] if n == "random_forest" else sc["hist_gradient_boosting"]
                if n == "xgboost":
                    grid = {"n_estimators": [200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
                if quick:
                    grid = {k: v[:1] for k, v in grid.items()}
                fitted[n], info[n] = ml.tune_and_fit(n, grid, X, y, g, sc["cv_blocks"], L, seed)
                joblib.dump(fitted[n], model_dir / f"soc_{n}__{'+'.join(train)}.joblib")
            # recurrent models: contiguous validation block per training condition (gap = L)
            tr_idx, va_idx = [], []
            off = 0
            for Wd in Ws:
                n_w = len(Wd); lo, hi = int(0.45 * n_w), int(0.45 * n_w) + int(sc["rnn"]["val_fraction"] * n_w)
                va_idx.append(np.arange(lo, hi) + off)
                tr_idx.append(np.r_[np.arange(0, max(0, lo - L)), np.arange(hi + L, n_w)][::sc["window_stride_train"]] + off); off += n_w
            Wall, Yall = np.concatenate(Ws), np.concatenate(Ys); tr, va = np.concatenate(tr_idx), np.concatenate(va_idx)
            for tag, cell, floor in (("gru", "gru", 1e-8), ("lstm", "lstm", 1e-8), ("gru_sf", "gru", np.array([0.05, 0.1])), ("lstm_sf", "lstm", np.array([0.05, 0.1]))):
                net = RNNRegressor(cell, Wall.shape[-1], sc["rnn"]["hidden"], seed)
                r = net.fit(Wall[tr], Yall[tr], Wall[va], Yall[va], sc["rnn"]["lr"], sc["rnn"]["batch"],
                            4 if quick else sc["rnn"]["epochs"], sc["rnn"]["patience"], seed, min_std=floor)
                info[tag] = {k: v for k, v in r.items() if k != "val_history"}
                fitted[tag] = net; net.save(model_dir / f"soc_{tag}__{'+'.join(train)}.npz")
            cache[train] = {"fitted": fitted, "info": info}
            logs["protocols"].append({"train": list(train), "n_train": int(len(y)), "models": {k: {kk: vv for kk, vv in v.items() if kk != "search"} for k, v in info.items()}})
            log.info("trained on %s: %s", train, {k: (round(v.get('cv_mae', v.get('val_mae', 0)), 4)) for k, v in info.items()})
        fitted = cache[train]["fitted"]
        for nl in sc["noise_levels"]:
            rng = np.random.default_rng(seed + 1)
            X, y, raw2, t = tab(test, nl, rng)
            for n, mdl in fitted.items():
                if n.startswith(("gru", "lstm")):
                    Wt, _, _ = make_windows(raw2, np.r_[np.zeros(L - 1), y], L, 1); p, dt = _timed(lambda: mdl.predict(Wt)); n_s = len(Wt)
                else:
                    p, dt = _timed(lambda: mdl.predict(X)); n_s = len(X)
                record("data_driven", train, test, n, nl, y, p, t, 1e6 * dt / n_s)
    df = pd.DataFrame(rows)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True); (out_dir / "experiment_logs").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics" / "soc_results.csv", index=False)
    pd.concat(preds).to_csv(out_dir / "metrics" / "soc_predictions.csv.gz", index=False, compression="gzip")
    import json
    logs["conditions"] = {d: {"capacity_ah": c.capacity_ah, "mean_current_a": c.mean_current_a, "nominal_temp_c": c.nominal_temp_c,
                              **{k: v for k, v in c.meta.items() if k != "raw_slice"}} for d, c in conds.items()}
    with open(out_dir / "experiment_logs" / "soc_experiments.json", "w", encoding="utf-8") as fh:
        json.dump(logs, fh, indent=2, default=float)
    return {"results": df, "log": logs}
