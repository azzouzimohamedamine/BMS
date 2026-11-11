"""Shared helpers for the Streamlit dashboard (every number shown comes from results/ or live model runs)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

LABELS = {"T00C_lowI": "0 °C label, 0.1 A", "T25C_lowI": "25 °C label, 0.1 A", "T45C_lowI": "45 °C label, 0.1 A", "INIT_1A": "1 A, temperature unknown"}
BADGE = {"MEASURED": "🟦 MEASURED", "CALCULATED": "🟪 CALCULATED", "ESTIMATED": "🟩 ESTIMATED", "SIMULATED": "🟧 SIMULATED", "UNAVAILABLE": "⬜ UNAVAILABLE"}


@st.cache_resource(show_spinner="Initialising BMS engine (ECM identification)…")
def get_engine():
    from bms.integration.engine import BMSEngine
    return BMSEngine()


@st.cache_data
def read_metrics(name: str) -> pd.DataFrame | None:
    p = ROOT / "results" / "metrics" / name
    return pd.read_csv(p) if p.exists() else None


def require_results(name: str) -> pd.DataFrame | None:
    df = read_metrics(name)
    if df is None:
        st.error(f"`results/metrics/{name}` not found. Run the experiment scripts first (see README).")
    return df


def dataset_selector(key: str = "ds") -> str:
    return st.sidebar.selectbox("Experimental condition", list(LABELS), format_func=lambda k: LABELS[k], key=key)
