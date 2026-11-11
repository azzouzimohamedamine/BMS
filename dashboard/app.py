"""Page 1 — Overview. Run with:  streamlit run dashboard/app.py"""
import json

import streamlit as st

from common import ROOT, BADGE, LABELS, read_metrics

st.set_page_config(page_title="AI-Driven BMS", layout="wide")
st.title("AI-Driven Battery Management System — CALCE SP20-1")
st.caption("Research prototype. All values come from real CALCE measurements, models trained here, or are explicitly labelled SIMULATED.")
st.markdown("**Legend:** " + " · ".join(BADGE.values()))
st.markdown("""
**Scope that the data supports.** The four files cover a *single cell*, *one cycle per condition* (no ageing), at temperature
labels 0/25/45 °C (0.1 A tests) plus a 1 A initial-capacity test. Hence SOC estimation is fully evaluated, whereas
SOH-vs-ageing and RUL are **UNAVAILABLE** (explained on their pages) and fault diagnosis is limited to statistical anomalies.
""")
cap = read_metrics("soh_capacity_table.csv")
if cap is not None:
    st.subheader("Measured operating conditions and delivered capacity " + BADGE["MEASURED"])
    cap = cap.copy(); cap["dataset_id"] = cap["dataset_id"].map(LABELS)
    st.dataframe(cap.drop(columns=["is_SOH"]), use_container_width=True)
    st.caption("Capacity ratio is relative to the 25 °C low-rate test; it is NOT a State of Health (nominal capacity not verified).")
p = ROOT / "results" / "experiment_logs" / "soc_experiments.json"
if p.exists():
    lg = json.loads(p.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated capacity (25 °C, 0.1 A)", f"{lg['q_cal_ah']:.3f} Ah")
    c2.metric("ECM R0 (25 °C)", f"{lg['ecm_reference']['r0']:.3f} Ω")
    c3.metric("ECM τ (25 °C)", f"{lg['ecm_reference']['tau_s']:.0f} s")
else:
    st.warning("SOC experiment log missing: run `python scripts/run_soc_experiments.py`.")
