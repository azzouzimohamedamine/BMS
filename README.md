# AI-Driven Battery Management System for EVs — CALCE SP20-1

Research software that estimates battery states from CALCE experimental data: **SOC estimation** (Coulomb counting, EKF, Random Forest, gradient boosting, LSTM, GRU), a **guarded SOH/RUL** pipeline, **anomaly detection with a diagnostic layer**, a unified **BMS engine** and a **Streamlit dashboard**.

> **Honest scope statement.** The four supplied files hold a *single cell* with *one cycle per condition* (0.1 A tests at 0/25/45 °C labels + one 1 A capacity test). That supports a rigorous SOC study but **not** SOH-vs-ageing, RUL, cell-imbalance or temperature-fault diagnosis. Those modules are implemented as guarded pipelines that refuse to produce numbers without sufficient data; nothing is fabricated. Details: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Main results (SOC, vs. a *calculated* coulomb-integration reference; SOC points)
| Estimator (calibrated on 25 °C) | 0 °C | 45 °C | 1 A |
|---|---|---|---|
| Coulomb counting, MAE | 6.44 | 3.15 | 3.69 |
| EKF (1-RC ECM), MAE | 0.71 | 1.02 | 0.66 |
| Random Forest, MAE | 0.43 | 1.57 | 13.06 |
| Hist. gradient boosting, MAE | 0.42 | 1.56 | 14.40 |

EKF recovers from a wrong initial SOC (70 % vs 100 %), CC does not; tree models extrapolate poorly to a 10× higher current; the NumPy LSTM/GRU are less accurate and noise-sensitive. Caveats (reference normalisation, in-sample EKF identification, no dynamic loads) are in `docs/LIMITATIONS.md`. Full tables: [`docs/EXPERIMENTAL_RESULTS.md`](docs/EXPERIMENTAL_RESULTS.md); report: `report/`.

## SOH and RUL on ageing datasets (Kaggle CSV files, separate from CALCE)
CALCE cannot support SOH/RUL, so three downloaded CSV files are used in `data/raw/KAGGLE/` and **their provenance is kept separate everywhere**:

| File | Nature | Use |
|---|---|---|
| `features_arrhenius_tempmean.csv` | NASA-style battery IDs, features from an unknown third party (**provenance unverified**) | main SOH/RUL experiments (7 usable batteries) |
| `augmented_full_dataset.csv` | same rows + 3 **artificial** replicas per battery | only as extra *training* rows of training batteries (ablation) |
| `pybamm_synthetic_features.csv` | **PyBaMM simulation** (6 virtual batteries) | pipeline check only; never evidence for real batteries |

Protocol: leave-one-battery-out, plus 24 °C↔4 °C transfer. Leaking columns (`RUL`, `NRUL`, `battery_life`, `capacity_norm/diff/ma5`) and the redundant `arrhenius_fade_rate` are never used as inputs; SOH is relative to the cell's first-cycle capacity (no nominal capacity exists); EOL = 70 % of the first-cycle capacity (the rule the files' own RUL labels follow); RUL is in "retained records". Two batteries are excluded by audit rules (`B0033`: 1 cycle, `B0040`: corrupted capacity).
Highlights (pooled over held-out NASA-style batteries): SOH MAE 3.4 pts with Random Forest on signal features; RUL MAE 17.1 records (linear on capacity history) vs 46.4 for a mean-life baseline; **no temperature transfer and no augmentation gain demonstrated**; the Random Forest cannot extrapolate beyond the longest training life. Details: [`docs/AGING_EXPERIMENTS.md`](docs/AGING_EXPERIMENTS.md), report section "SOH and RUL on ageing datasets".

