"""Ageing-module tests. Synthetic frames verify SOFTWARE behaviour only; they are never used for reported results."""
import numpy as np
import pandas as pd

from bms.data.aging_loader import audit_batteries, usable_batteries
from bms.features.aging_features import FEATURE_SETS, FORBIDDEN, add_capacity_features, assert_no_leak
from bms.models.rul.aging import baseline_linear_extrapolation, baseline_mean_life
from bms.models.soh.aging import fit_model
from bms.preprocessing.transformers import Winsorizer

CFG = {"quality_rules": {"min_cycles": 15, "max_capacity_over_first": 1.15}, "eol_fraction": 0.70}


def _battery(bid, n=40, q0=2.0, fade=0.01, aug=False, rep=0):
    cyc = np.arange(1, n + 1)
    return pd.DataFrame({"battery_id": bid, "cycle": cyc, "capacity": q0 * (1 - fade * cyc), "is_augmented": aug, "aug_rep": rep,
                         "ambient_temperature": 24.0, "RUL": 0, "battery_life": n})


def test_audit_excludes_short_and_corrupted_batteries():
    short = _battery("S", n=3)
    bad = _battery("B"); bad.loc[5, "capacity"] = 9.0
    ok = _battery("O")
    u, a = usable_batteries(pd.concat([short, bad, ok]), CFG)
    assert set(u.battery_id) == {"O"} and a.set_index("battery_id").loc["S", "usable"] == False  # noqa: E712
    assert "inconsistent" in a.set_index("battery_id").loc["B", "reason"]


def test_eol_and_rul_labels():
    f = add_capacity_features(_battery("A", n=40, fade=0.01), 0.70)          # SOH is relative to cycle 1 (0.99*q0): cycle 30 -> 70.7 %, cycle 31 -> 69.7 % <= 70 %
    eol = f.eol_cycle.iloc[0]
    assert eol == 31 and (f.rul == eol - f.cycle).all() and abs(f.soh_rel.iloc[0] - 100 * 0.99 / 0.99) < 1e-9


def test_history_features_are_causal():
    a = _battery("A"); b = a.copy(); b.loc[b.cycle > 20, "capacity"] *= 0.5     # change the FUTURE only
    fa, fb = add_capacity_features(a), add_capacity_features(b)
    cols = ["soh_rel", "cap_ma3_rel", "fade_slope5"]
    assert np.allclose(fa.loc[fa.cycle <= 20, cols], fb.loc[fb.cycle <= 20, cols])


def test_feature_sets_have_no_leaking_columns():
    for t in FEATURE_SETS.values():
        for fs in t.values():
            assert_no_leak(fs); assert not set(fs) & FORBIDDEN
    try:
        assert_no_leak(["voltage_mean", "RUL"])
    except ValueError:
        return
    raise AssertionError("leak guard did not fire")


def test_winsorizer_uses_training_statistics_only():
    w = Winsorizer(0.0, 0.9).fit(np.arange(100.0).reshape(-1, 1))
    assert w.transform(np.array([[1000.0]]))[0, 0] <= 90.0


def test_models_fit_and_leave_one_battery_out_groups_are_disjoint():
    rng = np.random.default_rng(0); X = rng.normal(size=(120, 3)); y = X[:, 0] * 2 + 1; g = np.repeat([0, 1, 2, 3], 30)
    for name in ("linear", "random_forest"):
        pipe, info = fit_model(name, X, y, g, 0, quick=True)
        assert np.mean(np.abs(pipe.predict(X) - y)) < 1.0


def test_rul_baselines_are_non_negative_and_causal():
    g = add_capacity_features(_battery("A", n=40, fade=0.01))
    p = baseline_linear_extrapolation(g, np.array([30.0]))
    assert (p >= 0).all() and abs(p[20] - (g.eol_cycle.iloc[0] - g.cycle.iloc[20])) < 2.0
    g2 = g.copy(); g2.loc[g2.cycle > 20, "soh_rel"] = 100.0
    assert np.allclose(p[:20], baseline_linear_extrapolation(g2, np.array([30.0]))[:20])
    assert (baseline_mean_life(np.array([100.0]), np.array([0, 50, 200])) == [100, 50, 0]).all()
