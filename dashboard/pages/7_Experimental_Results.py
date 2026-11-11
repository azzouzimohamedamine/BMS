import streamlit as st

from common import BADGE, LABELS, dataset_selector, get_engine, require_results
from bms.integration.report import build_markdown_report

st.title("Experimental results")
res = require_results("soc_results.csv")
if res is not None:
    clean = res[res.noise == "clean"]
    st.subheader("SOC — MAE (SOC points), clean signals")
    st.dataframe(clean.pivot_table(index=["protocol", "train", "test"], columns="model", values="MAE").round(2), use_container_width=True)
    st.subheader("Noise robustness — SIMULATED perturbations " + BADGE["SIMULATED"])
    sub = res[(res.train == "T25C_lowI") | (res.protocol == "physics_T25C_lowI")]
    st.dataframe(sub.pivot_table(index=["test", "model"], columns="noise", values="MAE").round(2), use_container_width=True)
    st.subheader("Inference cost (ms per 1000 samples, single CPU core)")
    st.dataframe(clean.groupby("model").infer_ms_per_1k.mean().round(3).to_frame())
d = dataset_selector()
rep = get_engine().analyze(d)
st.download_button("Download experiment report (Markdown)", build_markdown_report(rep), file_name=f"bms_report_{d}.md")
for w in rep.warnings:
    st.warning(w)
