"""Generate LaTeX tables for the report directly from results/metrics (keeps the report synchronised)."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from bms.utils.config import repo_root  # noqa: E402

R = repo_root(); M = R / "results" / "metrics"; T = R / "report" / "tables"; T.mkdir(parents=True, exist_ok=True)
esc = lambda s: str(s).replace("_", r"\_")
TL = {"T00C_lowI": "0\\,°C", "T25C_lowI": "25\\,°C", "T45C_lowI": "45\\,°C", "INIT_1A": "1\\,A"}
tl = lambda s: "+".join(TL[x] for x in s.split("+"))

def write(name, df, fmt, caption, label, colfmt=None):
    cf = colfmt or "l" * df.index.nlevels + "r" * len(df.columns)
    body = df.to_latex(float_format=lambda x: fmt % x, escape=False, column_format=cf, na_rep="--")
    (T / f"{name}.tex").write_text("\\begin{table}[htbp]\\centering\\footnotesize\n" + body + f"\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table}}\n", encoding="utf-8")

cap = pd.read_csv(M / "soh_capacity_table.csv")
cap = cap.assign(Condition=cap.dataset_id.map(lambda x: TL[x]))[["Condition", "mean_current_a", "delivered_capacity_ah", "capacity_ratio_vs_T25C_lowI(%)", "R0_onset_ohm"]]
cap.columns = ["Condition", "Mean I (A)", "Delivered Q (Ah)", "Q / Q$_{25}$ (\\%)", "R$_0$ ($\\Omega$)"]
write("tab_capacity", cap.set_index("Condition"), "%.3f", "Measured delivered capacity (2.5\\,V cut-off) and R$_0$ at discharge onset per test. Temperatures are file-name labels. These are not SOH values.", "tab:capacity")

r = pd.read_csv(M / "soc_results.csv"); c = r[r.noise == "clean"]
ph = c[(c.protocol == "physics_T25C_lowI")].copy(); ph["Test"] = ph.test.map(TL)
ph["Estimator"] = ph.model.map({"coulomb": "CC (SOC$_0$=100\\%)", "coulomb_init70": "CC (SOC$_0$=70\\%)", "ekf": "EKF (SOC$_0$=100\\%)", "ekf_init70": "EKF (SOC$_0$=70\\%)"})
t = ph.set_index(["Test", "Estimator"])[["MAE", "RMSE", "R2", "MaxAE"]]; t.columns = ["MAE", "RMSE", "R$^2$", "MaxAE"]
write("tab_soc_physics", t, "%.2f", "Physics-based SOC estimators (capacity and ECM calibrated on the 25\\,°C test) on unseen conditions; errors in SOC points, first 10\\,min excluded.", "tab:socphys")

dd = c[c.protocol == "data_driven"].copy(); dd["Train $\\rightarrow$ Test"] = dd.train.map(tl) + " $\\rightarrow$ " + dd.test.map(TL)
order = ["random_forest", "hist_gradient_boosting", "gru", "gru_sf", "lstm", "lstm_sf"]
pv = dd.pivot_table(index="Train $\\rightarrow$ Test", columns="model", values="MAE")[order]; pv.columns = [esc(x) for x in order]
write("tab_soc_datadriven", pv, "%.2f", "Data-driven SOC models: MAE (SOC points) on the unseen test condition (clean signals). \\texttt{*\\_sf}: std-floor scaling variant added post hoc.", "tab:socdd")

n = r[((r.protocol == "data_driven") & (r.train == "T25C_lowI") | (r.protocol == "physics_T25C_lowI")) & r.test.isin(["T00C_lowI", "T45C_lowI"]) & r.model.isin(["coulomb", "ekf"] + order)]
g = n.groupby(["model", "noise"]).MAE.mean().unstack()[["clean", "low", "medium", "high", "i_bias_20mA"]]
g = g.loc[["coulomb", "ekf"] + order]; g.index = [esc(x) for x in g.index]; g.columns = ["clean", "low", "medium", "high", "I bias 20\\,mA"]
write("tab_noise", g, "%.2f", "Noise robustness with \\emph{simulated} perturbations (Table~\\ref{tab:noisedef}) added to the real test signals: mean MAE (SOC points) over the 0\\,°C and 45\\,°C tests, models fitted on 25\\,°C.", "tab:noise")

tm = c.groupby("model").infer_ms_per_1k.mean().to_frame("ms per 1000 samples"); tm.index = [esc(x) for x in tm.index]
write("tab_timing", tm, "%.2f", "Mean inference time (NumPy/scikit-learn/Python, single CPU core). The EKF figure is a plain Python loop.", "tab:timing")

s = pd.read_csv(M / "fault_simulated_benchmark.csv")
for mag in (0.5, 1.0):
    q = s[s.magnitude == mag]
    a = q.pivot_table(index="fault_type", columns="detector", values="event_detection_rate")[["iso", "ocsvm", "threshold", "flatline", "any"]]
    a.index = [esc(x) for x in a.index]; a.columns = ["IF", "OC-SVM", "Threshold", "Flatline", "Any"]
    write(f"tab_fault_sim_{str(mag).replace('.', '')}", a, "%.2f", f"\\emph{{Simulated}} faults (magnitude $\\times${mag}) injected into the real 25\\,°C signals: event detection rate (4 events per fault type).", f"tab:faultsim{str(mag).replace('.', '')}")
rf = pd.read_csv(M / "fault_real_data_summary.csv").set_index("dataset"); rf.index = [TL[x] for x in rf.index]
rf = rf[["n_windows", "flagged_iso", "flagged_ocsvm", "flagged_threshold", "flagged_flatline", "flagged_any"]]; rf.columns = ["windows", "IF", "OC-SVM", "Thr.", "Flat.", "Any"]
write("tab_fault_real", rf, "%.0f", "Windows flagged on the \\emph{real} data (statistical anomalies, not validated faults); detectors fitted on 0\\,°C and 45\\,°C windows.", "tab:faultreal")
for f in (R / "results" / "figures").glob("*.png"):
    shutil.copy(f, R / "report" / "figures" / f.name)
print(sorted(p.name for p in T.glob("*.tex")))

# ---------------- ageing datasets (SOH / RUL) ----------------
if (M / "aging_summary.csv").exists():
    sm = pd.read_csv(M / "aging_summary.csv"); PROV = {"EXPERIMENTAL_DERIVED": "NASA-style", "SIMULATED": "PyBaMM (sim.)"}
    FS = {"cycle_only": "cycle only", "base": "base", "no_cycle": "no cycle", "ext_rct_re": "+Rct,Re", "cap_hist": "capacity history", "full": "history+signals"}
    for tgt, name, unit in (("soh", "tab_aging_soh", "SOH points"), ("rul", "tab_aging_rul", "records")):
        q = sm[(sm.protocol == "leave_one_battery_out") & (sm.target == tgt)].copy()
        q["Data"] = q.provenance.map(PROV); q["Features"] = q.feature_set.map(FS); q["Model"] = q.model.map(lambda x: esc(x))
        q = q.sort_values(["Data", "MAE"]).set_index(["Data", "Features", "Model"])[["n_batteries", "MAE", "RMSE", "R2", "MaxAE"]]
        q.columns = ["Batt.", "MAE", "RMSE", "R$^2$", "MaxAE"]
        write(name, q, "%.2f", f"{tgt.upper()} leave-one-battery-out, pooled over held-out batteries ({unit}). ``PyBaMM (sim.)'' rows are \\emph{{simulated}} data.", f"tab:{name}")
    x = sm[(sm.dataset == "NASA_FEATURES") & (sm.protocol != "leave_one_battery_out")].copy()
    x["Protocol"] = x.protocol.map(lambda s: esc(s)); x["Target"] = x.target.str.upper(); x["Model"] = x.model.map(lambda s: esc(s)); x["Features"] = x.feature_set.map(FS)
    x = x.sort_values(["Target", "Protocol", "Model"]).set_index(["Target", "Protocol", "Features", "Model"])[["n_batteries", "MAE", "RMSE", "R2"]]
    x.columns = ["Batt.", "MAE", "RMSE", "R$^2$"]
    write("tab_aging_transfer", x, "%.2f", "NASA-style data: temperature transfer (24\\,°C$\\leftrightarrow$4\\,°C) and augmentation ablation.", "tab:agingtransfer")
    au = pd.read_csv(M / "aging_battery_audit.csv"); au = au[au.dataset == "NASA_FEATURES"][["battery_id", "n_cycles", "q0", "max_over_first", "ambient_c", "usable"]]
    au.columns = ["Battery", "Cycles", "Q$_1$ (Ah)", "max/first", "Ambient (°C)", "Usable"]
    write("tab_aging_audit", au.set_index("Battery"), "%.2f", "Rule-based audit of the NASA-style batteries (exclusion: fewer than 15 cycles, or capacity above 115\\,\\% of the first cycle).", "tab:agingaudit")
    for f in (R / "results" / "figures").glob("*.png"):
        shutil.copy(f, R / "report" / "figures" / f.name)
