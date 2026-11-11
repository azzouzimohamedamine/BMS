"""Generate all figures from real results (results/metrics) and cached data. Run after the experiment scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bms.preprocessing.pipeline import prepare_condition  # noqa: E402
from bms.data.loader import dataset_ids, load_dataset  # noqa: E402
from bms.utils.config import repo_root  # noqa: E402

R = repo_root(); F = R / "results" / "figures"; F.mkdir(parents=True, exist_ok=True); M = R / "results" / "metrics"
LAB = {"T00C_lowI": "0 °C (0.1 A)", "T25C_lowI": "25 °C (0.1 A)", "T45C_lowI": "45 °C (0.1 A)", "INIT_1A": "1 A (T unknown)"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 9})

def save(name): plt.tight_layout(); plt.savefig(F / name, bbox_inches="tight"); plt.close()

# 1 raw overview
fig, ax = plt.subplots(2, 4, figsize=(15, 5), sharex="col")
for k, d in enumerate(dataset_ids()):
    df = load_dataset(d); t = (df.time_s - df.time_s.iloc[0]) / 3600
    ax[0, k].plot(t, df.voltage_v, lw=.5); ax[1, k].plot(t, df.current_a, lw=.5); ax[0, k].set_title(LAB[d]); ax[1, k].set_xlabel("time (h)")
ax[0, 0].set_ylabel("Voltage (V)"); ax[1, 0].set_ylabel("Current (A)"); save("fig01_raw_overview.png")

# 2 voltage vs reference SOC
conds = {d: prepare_condition(d) for d in dataset_ids()}
plt.figure(figsize=(6, 4))
for d, c in conds.items(): plt.plot(100 * c.frame.soc_ref, c.frame.v, label=f"{LAB[d]}  Q={c.capacity_ah:.3f} Ah")
plt.xlabel("Reference SOC (%) [calculated, per-test normalisation]"); plt.ylabel("Terminal voltage (V)"); plt.legend(); plt.gca().invert_xaxis(); save("fig02_voltage_vs_soc_ref.png")

# 3 trajectories
P = pd.read_csv(M / "soc_predictions.csv.gz"); res = pd.read_csv(M / "soc_results.csv")
fig, ax = plt.subplots(2, 3, figsize=(15, 6), sharex="col")
for k, test in enumerate(["T00C_lowI", "T45C_lowI", "INIT_1A"]):
    pp = P[(P.protocol == "physics_T25C_lowI") & (P.test == test)]
    ref = pp[pp.model == "ekf"]; ax[0, k].plot(ref.t_s / 3600, 100 * ref.soc_ref, "k", lw=2, label="reference (calculated)")
    for m, c in [("coulomb", "tab:red"), ("ekf", "tab:blue"), ("coulomb_init70", "tab:orange"), ("ekf_init70", "tab:green")]:
        q = pp[pp.model == m]; ax[0, k].plot(q.t_s / 3600, 100 * q.soc_pred, c=c, lw=1, ls="--" if "init70" in m else "-", label=m)
        ax[1, k].plot(q.t_s / 3600, 100 * (q.soc_pred - q.soc_ref), c=c, lw=1, ls="--" if "init70" in m else "-")
    ax[0, k].set_title(LAB[test] + " (params from 25 °C)"); ax[1, k].set_xlabel("time (h)")
ax[0, 0].set_ylabel("SOC (%)"); ax[1, 0].set_ylabel("error (pts)"); ax[0, 0].legend(fontsize=7); save("fig03_physics_soc_trajectories.png")

# 4 cross-condition MAE (train 25 C)
d = res[(res.noise == "clean") & (res.protocol == "data_driven") & (res.train == "T25C_lowI")].copy()
ph = res[(res.noise == "clean") & (res.protocol == "physics_T25C_lowI") & res.model.isin(["coulomb", "ekf"])]
d = pd.concat([d, ph]); pv = d.pivot_table(index="test", columns="model", values="MAE")
pv.plot.bar(figsize=(10, 4), logy=True); plt.ylabel("MAE (SOC points, log)"); plt.title("Models fitted/identified on 25 °C, tested on unseen conditions"); plt.xticks(rotation=0); save("fig04_cross_condition_mae.png")

# 5 leave-one-condition-out (data-driven only)
d = res[(res.noise == "clean") & (res.protocol == "data_driven")].copy(); d["case"] = d.train.str.replace("_lowI", "") + " → " + d.test.str.replace("_lowI", "")
d.pivot_table(index="case", columns="model", values="MAE").plot.bar(figsize=(11, 4), logy=True); plt.ylabel("MAE (SOC points, log)"); plt.xticks(rotation=25, ha="right"); save("fig05_data_driven_protocols.png")

# 6 noise robustness
n = res[(res.protocol.isin(["data_driven", "physics_T25C_lowI"])) & ((res.train == "T25C_lowI")) & (res.test.isin(["T00C_lowI", "T45C_lowI"])) & res.model.isin(["coulomb", "ekf", "random_forest", "hist_gradient_boosting", "gru", "gru_sf", "lstm", "lstm_sf"])]
order = ["clean", "low", "medium", "high", "i_bias_20mA"]
g = n.groupby(["model", "noise"]).MAE.mean().unstack()[order]
g.T.plot(marker="o", figsize=(8, 4), logy=True); plt.ylabel("mean MAE over 0 °C and 45 °C tests (SOC pts)"); plt.xlabel("SIMULATED perturbation level"); save("fig06_noise_robustness.png")

# 7 capacity table
cap = pd.read_csv(M / "soh_capacity_table.csv"); fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].bar([LAB[x] for x in cap.dataset_id], cap.delivered_capacity_ah); ax[0].set_ylabel("delivered capacity (Ah)"); ax[0].tick_params(axis="x", rotation=20)
ax[1].bar([LAB[x] for x in cap.dataset_id], cap.R0_onset_ohm, color="tab:orange"); ax[1].set_ylabel("R0 at discharge onset (ohm)"); ax[1].tick_params(axis="x", rotation=20); save("fig07_capacity_and_r0.png")

# 8 fault benchmark
s = pd.read_csv(M / "fault_simulated_benchmark.csv"); s = s[s.magnitude.isin([0.5, 1.0])]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for a, mag in zip(ax, [0.5, 1.0]):
    pv = s[s.magnitude == mag].pivot_table(index="fault_type", columns="detector", values="event_detection_rate")[["iso", "ocsvm", "threshold", "flatline", "any"]]
    im = a.imshow(pv.values, vmin=0, vmax=1, cmap="Blues"); a.set_xticks(range(5)); a.set_xticklabels(pv.columns); a.set_yticks(range(len(pv))); a.set_yticklabels(pv.index)
    for (i, j), v in np.ndenumerate(pv.values): a.text(j, i, f"{v:.2f}", ha="center", va="center", color="k" if v < .6 else "w")
    a.set_title(f"SIMULATED faults, magnitude x{mag}: event detection rate"); a.grid(False)
save("fig08_fault_simulation_detection.png")

# 9 SOC error vs SOC (models fitted on 25 C only)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, test in zip(ax, ["T00C_lowI", "T45C_lowI"]):
    q = P[(P.train == "T25C_lowI") & (P.test == test) & P.model.isin(["ekf", "random_forest", "hist_gradient_boosting", "lstm_sf"])]
    for m, g_ in q.groupby("model"):
        a.plot(100 * g_.soc_ref.values, 100 * (g_.soc_pred - g_.soc_ref).values, lw=.8, label=m)
    a.set_title(LAB[test] + " (fitted on 25 °C)"); a.set_xlabel("reference SOC (%)"); a.set_ylabel("error (pts)"); a.invert_xaxis(); a.legend(fontsize=7)
save("fig09_error_vs_soc.png")
print(sorted(p.name for p in F.glob("*.png")))
