import json

import streamlit as st

from common import BADGE, LABELS, ROOT, read_metrics
from aging_view import render

st.title("SOH estimation")
t_calce, t_nasa, t_sim = st.tabs(["CALCE SP20-1 (not feasible)", "NASA-style data (experimental-derived)", "PyBaMM (SIMULATED)"])
with t_calce:
    st.info(BADGE["UNAVAILABLE"] + " — SOH(%) is not computed for the four CALCE files: one cycle per file (no ageing) and an unverified nominal capacity.")
    p = ROOT / "results" / "experiment_logs" / "soh_rul_feasibility.json"
    if p.exists():
        st.caption(json.loads(p.read_text(encoding="utf-8")).get("soh_percent", ""))
    cap = read_metrics("soh_capacity_table.csv")
    if cap is not None:
        cap = cap.copy(); cap["dataset_id"] = cap["dataset_id"].map(LABELS)
        st.dataframe(cap.drop(columns=["is_SOH"]), use_container_width=True)
        st.caption("Measured delivered capacity and R0 per condition. They mix temperature, rate and history effects and are NOT SOH.")
with t_nasa:
    render("NASA_FEATURES", "soh")
with t_sim:
    render("PYBAMM", "soh")
