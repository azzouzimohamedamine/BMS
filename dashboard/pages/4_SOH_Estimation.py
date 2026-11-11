import streamlit as st

from common import BADGE, LABELS, read_metrics, ROOT
import json

st.title("SOH estimation " + BADGE["UNAVAILABLE"])
v = json.loads((ROOT / "results" / "experiment_logs" / "soh_rul_feasibility.json").read_text(encoding="utf-8")) if (ROOT / "results" / "experiment_logs" / "soh_rul_feasibility.json").exists() else {}
st.error("SOH(%) cannot be computed with the available data. " + str(v.get("soh_percent", "Run scripts/run_soh_rul_feasibility.py.")))
st.markdown("""
* The 4 files contain **one cycle each**: there is no capacity-vs-cycle trajectory, so no degradation or cycle-ageing analysis.
* The nominal capacity of SP20-1 has **not been verified** from reliable documentation, and it is never assumed.
* Below: what *is* measured — delivered capacity and the R0 indicator per condition. These mix temperature, rate and history effects and are **not** SOH.
""")
cap = read_metrics("soh_capacity_table.csv")
if cap is not None:
    cap = cap.copy(); cap["dataset_id"] = cap["dataset_id"].map(LABELS)
    st.dataframe(cap.drop(columns=["is_SOH"]), use_container_width=True)
    st.bar_chart(cap.set_index("dataset_id")["delivered_capacity_ah"])
st.caption("To enable SOH: set `nominal_capacity_ah` and `nominal_capacity_verified: true` in configs/dataset_config.yaml AND provide multi-cycle data.")
