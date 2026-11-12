import json

import streamlit as st

from common import BADGE, ROOT
from aging_view import render

st.title("RUL prediction")
t_calce, t_nasa, t_sim = st.tabs(["CALCE SP20-1 (not feasible)", "NASA-style data (experimental-derived)", "PyBaMM (SIMULATED)"])
with t_calce:
    st.info(BADGE["UNAVAILABLE"] + " — RUL cannot be defined for the four CALCE files (one cycle each, no capacity trajectory reaching an EOL).")
    p = ROOT / "results" / "experiment_logs" / "soh_rul_feasibility.json"
    if p.exists():
        for k, msg in json.loads(p.read_text(encoding="utf-8")).get("rul", {}).items():
            st.write(f"**{k}** — {msg}")
with t_nasa:
    render("NASA_FEATURES", "rul")
with t_sim:
    render("PYBAMM", "rul")
