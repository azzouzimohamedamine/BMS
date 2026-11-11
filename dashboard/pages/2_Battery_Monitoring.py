import matplotlib.pyplot as plt
import streamlit as st

from common import BADGE, LABELS, dataset_selector, get_engine

st.title("Battery monitoring " + BADGE["MEASURED"])
d = dataset_selector()
eng = get_engine()
est = eng.soc_estimates(d)
rep = eng.analyze(d)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Delivered capacity", f"{rep.measured['delivered_capacity_ah']:.3f} Ah")
c2.metric("Mean current", f"{rep.measured['mean_current_a']:.3f} A")
c3.metric("V start → end", f"{rep.measured['v_start']:.3f} → {rep.measured['v_end']:.3f} V")
c4.metric("Cycles in file", rep.measured["n_cycles_in_file"])
st.info(f"Temperature: {rep.measured['temperature']}")
fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(est.t_s / 3600, est.v); ax[0].set_ylabel("Voltage (V)")
ax[1].plot(est.t_s / 3600, est.i); ax[1].set_ylabel("Current (A)"); ax[1].set_xlabel("time (h)")
st.pyplot(fig)
st.caption("Main discharge segment, resampled to 20 s. Capacity = ∫|I|dt to the 2.5 V cut-off.")
