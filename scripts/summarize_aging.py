"""Pooled metrics, data-audit doc and figures for the ageing experiments. Run after run_aging_experiments.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bms.evaluation.metrics import regression_metrics  # noqa: E402
from bms.utils.config import repo_root  # noqa: E402

R = repo_root(); M = R / "results" / "metrics"; F = R / "results" / "figures"
res = pd.read_csv(M / "aging_results.csv"); pr = pd.read_csv(M / "aging_predictions.csv.gz"); au = pd.read_csv(M / "aging_battery_audit.csv")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
TAG = {"EXPERIMENTAL_DERIVED": "NASA-style data (experimental-derived, provenance unverified)", "SIMULATED": "PyBaMM (SIMULATED)"}

# ---- pooled metrics over all held-out batteries
rows = []
for k, g in pr.groupby(["dataset", "provenance", "target", "feature_set", "model", "protocol"]):
    m = regression_metrics(g.y_true, g.y_pred, 1.0)
    rows.append({**dict(zip(["dataset", "provenance", "target", "feature_set", "model", "protocol"], k)), "n_batteries": g.test_battery.nunique(), "n_points": len(g), **m})
summ = pd.DataFrame(rows)
cov = res[res.coverage_90.notna()].groupby(["dataset", "feature_set", "model", "protocol"]).coverage_90.mean().rename("coverage_90_mean").reset_index() if "coverage_90" in res else None
if cov is not None:
    summ = summ.merge(cov, on=["dataset", "feature_set", "model", "protocol"], how="left")
summ.to_csv(M / "aging_summary.csv", index=False)

# ---- figures
def curves(ax, ds):
    d = pr[(pr.dataset == ds) & (pr.target == "soh") & (pr.feature_set == "cycle_only") & (pr.model == "linear") & (pr.protocol == "leave_one_battery_out")]
    for b, g in d.groupby("test_battery"):
        ax.plot(g.cycle, g.y_true, lw=1, label=b)
    ax.set_xlabel("cycle index (retained records)"); ax.set_ylabel("SOH relative to first cycle (%)"); ax.axhline(70, c="k", ls=":", lw=.8); ax.legend(fontsize=6)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
curves(ax[0], "NASA_FEATURES"); ax[0].set_title(TAG["EXPERIMENTAL_DERIVED"], fontsize=8); curves(ax[1], "PYBAMM"); ax[1].set_title(TAG["SIMULATED"], fontsize=8)
plt.tight_layout(); plt.savefig(F / "fig10_aging_capacity_trajectories.png", bbox_inches="tight"); plt.close()

d = pr[(pr.dataset == "NASA_FEATURES") & (pr.target == "soh") & (pr.feature_set == "base") & (pr.model == "random_forest") & (pr.protocol == "leave_one_battery_out")]
bs = sorted(d.test_battery.unique()); fig, ax = plt.subplots(2, 4, figsize=(13, 5.5)); ax = ax.ravel()
for a, b in zip(ax, bs):
    g = d[d.test_battery == b]; a.plot(g.cycle, g.y_true, "k", label="reference (from capacity)"); a.plot(g.cycle, g.y_pred, "tab:blue", label="RF (battery unseen)"); a.set_title(b, fontsize=8)
ax[0].legend(fontsize=6); ax[-1].axis("off"); fig.suptitle("SOH (relative to first-cycle capacity), leave-one-battery-out — " + TAG["EXPERIMENTAL_DERIVED"], fontsize=9)
plt.tight_layout(); plt.savefig(F / "fig11_aging_soh_nasa_lobo.png", bbox_inches="tight"); plt.close()

for ds, name in (("NASA_FEATURES", "fig12_aging_rul_nasa_lobo.png"), ("PYBAMM", "fig12b_aging_rul_pybamm_lobo.png")):
    d = pr[(pr.dataset == ds) & (pr.target == "rul") & (pr.feature_set == "cap_hist") & (pr.model == "random_forest") & (pr.protocol == "leave_one_battery_out")]
    bs = sorted(d.test_battery.unique()); fig, ax = plt.subplots(2, 4, figsize=(13, 5.5)); ax = ax.ravel()
    for a, b in zip(ax, bs):
        g = d[d.test_battery == b]; a.plot(g.cycle, g.y_true, "k", label="true RUL"); a.plot(g.cycle, g.y_pred, "tab:blue", label="RF prediction")
        a.fill_between(g.cycle, g.lo, g.hi, color="tab:blue", alpha=.2, label="5–95 % tree spread"); a.set_title(b, fontsize=8); a.set_xlabel("cycle")
    ax[0].legend(fontsize=6); [x.axis("off") for x in ax[len(bs):]]
    fig.suptitle("RUL (units: retained records), battery unseen — " + TAG["EXPERIMENTAL_DERIVED" if ds == "NASA_FEATURES" else "SIMULATED"], fontsize=9)
    plt.tight_layout(); plt.savefig(F / name, bbox_inches="tight"); plt.close()

lobo = summ[summ.protocol == "leave_one_battery_out"]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a, tgt in zip(ax, ("soh", "rul")):
    p = lobo[lobo.target == tgt].assign(label=lambda x: x.model + "\n" + x.feature_set).pivot_table(index="label", columns="provenance", values="MAE")
    p.plot.bar(ax=a, rot=45); a.set_ylabel("pooled MAE (" + ("SOH points" if tgt == "soh" else "records") + ")"); a.set_title(tgt.upper() + ": leave-one-battery-out"); a.legend(["experimental-derived", "SIMULATED"][:len(p.columns)], fontsize=7)
plt.tight_layout(); plt.savefig(F / "fig13_aging_model_comparison.png", bbox_inches="tight"); plt.close()

xt = summ[summ.protocol.str.startswith("cross_temperature") | (summ.protocol == "leave_one_battery_out")]
xt = xt[(xt.dataset == "NASA_FEATURES") & (((xt.target == "soh") & (xt.feature_set == "base")) | ((xt.target == "rul") & (xt.feature_set == "full")))]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a, tgt in zip(ax, ("soh", "rul")):
    xt[xt.target == tgt].pivot_table(index="protocol", columns="model", values="MAE").plot.bar(ax=a, rot=15); a.set_title(tgt.upper() + " — NASA-style data, temperature transfer"); a.set_ylabel("pooled MAE")
plt.tight_layout(); plt.savefig(F / "fig14_aging_cross_temperature.png", bbox_inches="tight"); plt.close()

# ---- auto-generated audit/results doc
def md(df, nd=2):
    df = df.reset_index(drop=True); f = lambda v: f"{v:.{nd}f}" if isinstance(v, float) else str(v)
    return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.to_numpy())
doc = ["# Ageing datasets (Kaggle): audit and results (auto-generated by `scripts/summarize_aging.py`)", "",
       "Data tags: **EXPERIMENTAL_DERIVED** (NASA-style IDs; features computed by an unknown third party; provenance NOT verified), **SIMULATED** (PyBaMM), **AUGMENTED** (artificial replicas, training rows of training batteries only).", "",
       "## 1. Battery audit (rule-based exclusion)", md(au[["dataset", "battery_id", "n_cycles", "q0", "max_over_first", "min_over_first", "ambient_c", "usable", "reason"]], 3), "",
       "## 2. Pooled leave-one-battery-out metrics (all held-out batteries pooled; SOH in % points, RUL in retained records)",
       md(lobo[["provenance", "target", "feature_set", "model", "n_batteries", "n_points", "MAE", "RMSE", "R2", "MaxAE"]].sort_values(["provenance", "target", "MAE"]), 3), "",
       "## 3. Temperature transfer and augmentation ablation (NASA-style data)",
       md(summ[(summ.dataset == "NASA_FEATURES") & (summ.protocol != "leave_one_battery_out")][["protocol", "target", "feature_set", "model", "n_batteries", "MAE", "RMSE", "R2"]].sort_values(["target", "protocol", "model"]), 3), "",
       "## 4. RUL tree-spread coverage (heuristic 5–95 % interval of the RF trees)",
       md(res[res.coverage_90.notna()].groupby(["dataset", "feature_set", "protocol"]).coverage_90.mean().round(2).reset_index(), 2), ""]
(R / "docs" / "AGING_EXPERIMENTS.md").write_text("\n".join(doc), encoding="utf-8")
print("summary rows", len(summ)); print(sorted(p.name for p in F.glob("fig1[0-4]*.png")))
