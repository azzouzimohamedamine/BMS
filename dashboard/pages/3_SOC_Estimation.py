import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from common import BADGE, LABELS, dataset_selector, get_engine, require_results

st.title("SOC estimation")
st.warning("The reference SOC is " + BADGE["CALCULATED"] + " by coulomb integration of the measured current over the test "
           "(100 % = start of discharge, 0 % = 2.5 V cut-off of that test). It is not an independently measured ground truth.")
d = dataset_selector()
init = st.sidebar.slider("Initial SOC given to CC / EKF (%)", 40, 100, 100) / 100
est = get_engine().soc_estimates(d, init)
cols = [c for c in ["coulomb", "ekf", "random_forest", "hist_gradient_boosting"] if c in est and est[c].notna().any()]
sel = st.multiselect("Estimators " + BADGE["ESTIMATED"], cols, default=["coulomb", "ekf"])
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(est.t_s / 3600, 100 * est.soc_reference, "k", lw=2, label="reference (calculated)")
for c in sel:
    ax[0].plot(est.t_s / 3600, 100 * est[c], label=c); ax[1].plot(est.t_s / 3600, 100 * (est[c] - est.soc_reference), label=c)
ax[0].set_ylabel("SOC (%)"); ax[1].set_ylabel("error (SOC points)"); ax[1].set_xlabel("time (h)"); ax[0].legend()
st.pyplot(fig)
st.caption("CC and EKF use the capacity and ECM calibrated on the 25 °C test; tree models are trained on 25 °C only. "
           "On the 25 °C test itself these are in-sample and not a generalisation result.")
res = require_results("soc_results.csv")
if res is not None:
    t = res[(res.noise == "clean") & (res.test == d)]
    st.subheader("Model comparison on this condition (all protocols, clean signals)")
    st.dataframe(t[["protocol", "train", "model", "MAE", "RMSE", "R2", "MaxAE", "infer_ms_per_1k"]].round(3), use_container_width=True)
    st.caption("Metrics are in SOC percentage points; the first 10 min (sequence-window warm-up) are excluded for every model. "
               "infer_ms_per_1k = milliseconds per 1000 samples.")
