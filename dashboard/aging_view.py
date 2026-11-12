"""Shared renderer for the SOH and RUL pages: ageing-dataset results with explicit provenance banners."""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from common import read_metrics

BANNER = {
    "NASA_FEATURES": ("info", "🟦 EXPERIMENTAL-DERIVED data. Battery IDs follow NASA naming; per-cycle features were computed by an unknown third party and the "
                      "provenance is NOT verified. 7 usable batteries (2 at 24 °C, 5 at 4 °C): statistical power is low."),
    "PYBAMM": ("warning", "🟧 SIMULATED data (PyBaMM, 6 virtual batteries). Good scores show that a model recovers the simulator's behaviour, "
               "NOT that it works on real batteries."),
}


def render(ds: str, target: str) -> None:
    kind, text = BANNER[ds]
    getattr(st, kind)(text)
    res, pr, summ = read_metrics("aging_results.csv"), read_metrics("aging_predictions.csv.gz"), read_metrics("aging_summary.csv")
    if res is None or pr is None or summ is None:
        st.error("Ageing results not found. Run `python scripts/run_aging_experiments.py` then `python scripts/summarize_aging.py`.")
        return
    s = summ[(summ.dataset == ds) & (summ.target == target)]
    unit = "SOH percentage points (SOH relative to the cell's own first-cycle capacity, NOT to a nominal capacity)" if target == "soh" else \
        "retained records (the 'cycle' column; real cycles are spaced 2–4 apart); EOL = capacity ≤ 70 % of the first-cycle capacity"
    st.caption(f"Pooled metrics over all held-out batteries; errors in {unit}.")
    st.dataframe(s[["protocol", "feature_set", "model", "n_batteries", "MAE", "RMSE", "R2", "MaxAE"]].round(3), use_container_width=True)
    proto = st.selectbox("Protocol", sorted(s.protocol.unique()), key=f"p_{ds}_{target}")
    ss = s[s.protocol == proto]
    fs = st.selectbox("Feature set", sorted(ss.feature_set.unique()), key=f"f_{ds}_{target}")
    model = st.selectbox("Model", sorted(ss[ss.feature_set == fs].model.unique()), key=f"m_{ds}_{target}")
    d = pr[(pr.dataset == ds) & (pr.target == target) & (pr.protocol == proto) & (pr.feature_set == fs) & (pr.model == model)]
    if d.empty:
        st.info("No stored predictions for this selection."); return
    b = st.selectbox("Held-out battery (never seen during training)", sorted(d.test_battery.unique()), key=f"b_{ds}_{target}")
    g = d[d.test_battery == b]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(g.cycle, g.y_true, "k", label="reference (computed from measured capacity)" if target == "soh" else "true RUL (from observed EOL)")
    ax.plot(g.cycle, g.y_pred, "tab:blue", label="estimate")
    if g.lo.notna().any():
        ax.fill_between(g.cycle, g.lo, g.hi, color="tab:blue", alpha=.2, label="5–95 % spread of RF trees (heuristic, not calibrated)")
    ax.set_xlabel("cycle index (retained records)"); ax.set_ylabel("SOH (%)" if target == "soh" else "RUL (records)"); ax.legend(fontsize=7)
    st.pyplot(fig)
    if target == "rul" and "coverage_90" in res:
        c = res[(res.dataset == ds) & (res.protocol == proto) & (res.feature_set == fs) & (res.model == model)].coverage_90.dropna()
        if len(c):
            st.caption(f"Tree-spread coverage of the true RUL (nominal 90 %): {c.mean():.2f} averaged over held-out batteries.")
