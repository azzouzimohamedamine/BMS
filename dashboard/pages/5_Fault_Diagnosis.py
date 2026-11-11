import streamlit as st

from common import BADGE, dataset_selector, get_engine, require_results

st.title("Fault diagnosis")
st.warning("No fault labels exist in the CALCE files. Real-data detections are **statistical anomalies**, not validated faults. "
           "Benchmark results below use " + BADGE["SIMULATED"] + " faults injected into real signals.")
d = dataset_selector()
fr = get_engine().fault_report(d)
st.subheader("Diagnostic report (real data, ranked by Isolation-Forest score)")
st.dataframe(fr, use_container_width=True) if len(fr) else st.success("No window exceeded the detection thresholds.")
st.info("Unavailable diagnostics: over-temperature (no temperature channel), cell imbalance (single cell), resistance trend (single cycle).")
real = require_results("fault_real_data_summary.csv")
if real is not None:
    st.subheader("Flagged windows per dataset (detectors fitted on 0 °C + 45 °C windows)"); st.dataframe(real)
sim = require_results("fault_simulated_benchmark.csv")
if sim is not None:
    mag = st.select_slider("Simulated fault magnitude", sorted(sim.magnitude.unique()), value=1.0)
    st.subheader("SIMULATED-fault benchmark " + BADGE["SIMULATED"])
    st.dataframe(sim[sim.magnitude == mag].pivot_table(index="fault_type", columns="detector", values="event_detection_rate").round(2))
