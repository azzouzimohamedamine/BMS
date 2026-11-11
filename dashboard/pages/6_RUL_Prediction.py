import json

import streamlit as st

from common import BADGE, ROOT

st.title("RUL prediction " + BADGE["UNAVAILABLE"])
p = ROOT / "results" / "experiment_logs" / "soh_rul_feasibility.json"
v = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
st.error("Remaining Useful Life cannot be defined with these files.")
for k, msg in v.get("rul", {}).items():
    st.write(f"**{k}** — {msg}")
st.markdown("""
RUL = EOL cycle − current cycle requires (i) a capacity trajectory over many cycles that actually crosses an EOL threshold
(e.g. 80 % of a *verified* nominal capacity) and (ii) a verified nominal capacity. Neither holds. No RUL label, curve or
prediction is shown, because it would have to be invented. The pipeline (`bms.models.rul`) is implemented, unit-tested on
explicitly synthetic arrays for software correctness only, and raises `InsufficientDataError` on these data.
""")
