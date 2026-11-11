"""Unit tests. Synthetic arrays below are used ONLY to verify software correctness, never as results."""
import numpy as np
import pandas as pd

from bms.battery_models.ecm import ECMParams
from bms.evaluation.metrics import regression_metrics
from bms.features.engineering import build_features, make_windows
from bms.features.soc_reference import cumulative_ah, discharge_reference
from bms.models.rul.rul import assert_rul_feasible, find_eol_cycle, rul_labels
from bms.models.soc.coulomb import coulomb_counting
from bms.models.soc.ekf import run_ekf
from bms.models.soc import ml
from bms.models.soh.capacity import soh_percent, fit_cycle_soh_models
from bms.preprocessing.segmentation import label_states, segment_table
from bms.utils.errors import InsufficientDataError


def _expect_insufficient(fn):
    try:
        fn()
    except InsufficientDataError:
        return
    raise AssertionError("InsufficientDataError expected")


def test_cumulative_ah_constant_current():
    t = np.arange(0, 3601.0); i = np.full_like(t, -1.0)
    assert abs(cumulative_ah(t, i)[-1] + 1.0) < 1e-9


def test_discharge_reference_bounds():
    t = np.arange(0, 7201.0); soc, q = discharge_reference(t, np.full_like(t, -0.5))
    assert abs(q - 1.0) < 1e-9 and soc[0] == 1.0 and abs(soc[-1]) < 1e-12 and np.all(np.diff(soc) <= 0)


def test_label_states_and_segments():
    i = np.r_[np.zeros(1000), np.full(2000, -0.1), np.zeros(1000), np.full(2000, 0.1)]
    df = pd.DataFrame({"time_s": np.arange(len(i), dtype=float), "current_a": i})
    assert set(label_states(i)) == {-1, 0, 1}
    seg = segment_table(df, 0.02, 600)
    assert list(seg.state_name) == ["rest", "discharge", "rest", "charge"]


def test_coulomb_counting_exact_with_right_capacity():
    t = np.arange(0, 3601.0); soc = coulomb_counting(t, np.full_like(t, -1.0), 2.0, 1.0)
    assert abs(soc[-1] - 0.5) < 1e-9


def test_ekf_recovers_from_wrong_initial_soc_on_simulated_ecm():
    grid = np.linspace(0, 1, 101); p = ECMParams(0.05, 0.02, 100.0, grid, 3.2 + 1.0 * grid + 0.1 * np.sin(6 * grid))
    n, dt, q = 1500, 20.0, 2.0
    t = np.arange(n) * dt; i = np.full(n, -0.5)
    soc = 1 - np.cumsum(0.5 * dt / 3600 / q * np.ones(n)); v1 = np.zeros(n)
    for k in range(1, n):
        a = np.exp(-dt / p.tau); v1[k] = a * v1[k - 1] + p.r1 * (1 - a) * 0.5
    v = p.ocv(soc) - p.r0 * 0.5 - v1
    est = run_ekf(t, v, i, p, q, 0.6)
    assert abs(est[-1] - soc[-1]) < 0.02 and abs(est[0] - soc[0]) > 0.3


def test_metrics_perfect_and_biased():
    y = np.linspace(0, 1, 100)
    assert regression_metrics(y, y)["MAE"] == 0
    assert abs(regression_metrics(y, y + 0.01)["MAE"] - 1.0) < 1e-9


def test_features_are_causal():
    v = np.arange(100.0); f1 = build_features(v, v, 20.0, 600)
    v2 = v.copy(); v2[60:] += 100; f2 = build_features(v2, v2, 20.0, 600)
    assert np.allclose(f1.iloc[:60].to_numpy(), f2.iloc[:60].to_numpy())


def test_make_windows_shape_and_labels():
    X = np.arange(20.0).reshape(10, 2); y = np.arange(10.0)
    W, yw, ends = make_windows(X, y, 4, 2)
    assert W.shape == (4, 4, 2) and list(yw) == [3, 5, 7, 9]


def test_blocked_cv_has_no_overlap_or_leakage():
    g = np.r_[np.zeros(100), np.ones(100)]
    for tr, va in ml.blocked_splits(g, 5, 7):
        assert not set(tr) & set(va)
        for gi in (0, 1):
            vv = va[g[va] == gi]; tt = tr[g[tr] == gi]
            assert not np.any((tt > vv.min() - 7) & (tt < vv.max() + 7))


def test_soh_refuses_without_verified_nominal():
    _expect_insufficient(lambda: soh_percent(2.0, None, False))
    _expect_insufficient(lambda: soh_percent(2.0, 2.2, False))
    assert abs(soh_percent(2.0, 2.5, True) - 80.0) < 1e-9


def test_soh_models_need_many_cycles():
    _expect_insufficient(lambda: fit_cycle_soh_models(pd.DataFrame({"a": range(5)}), np.ones(5), 2.0, True))


def test_rul_guards_and_labels():
    _expect_insufficient(lambda: assert_rul_feasible(np.array([1]), np.array([2.0]), 2.2, True))
    _expect_insufficient(lambda: assert_rul_feasible(np.arange(100), np.full(100, 2.1), 2.2, True))   # never reaches EOL
    _expect_insufficient(lambda: assert_rul_feasible(np.arange(100), np.full(100, 1.0), None, False))
    cyc = np.arange(100); cap = 2.2 - 0.005 * cyc                       # synthetic, software test only
    eol = assert_rul_feasible(cyc, cap, 2.2, True)
    assert eol == find_eol_cycle(cyc, cap, 2.2) and list(rul_labels(np.array([0, eol, eol + 5]), eol)) == [eol, 0, 0]
