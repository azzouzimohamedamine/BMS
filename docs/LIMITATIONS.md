# Limitations 

## Data-imposed
1. **No ageing data** (1 cycle/file) → SOH-vs-cycle, capacity fade, RUL, EOL, cycle-aging features: **not feasible; not computed.**
2. **Nominal capacity unverified** → SOH(%) disabled (`configs/dataset_config.yaml`: `nominal_capacity_verified: false`).
3. **Single cell** → no cell-to-cell generalisation, no imbalance diagnosis.
4. **No temperature measurement**; 0/25/45 °C are file-name labels (a single 23.0 °C reading exists in the "25 °C" file; the 1 A file has no temperature). Over-temperature diagnosis is impossible.
5. **No measured SOC**; the reference is calculated by coulomb integration and normalised per test. This removes the temperature dependence of capacity from the target, so the good cross-temperature scores of the data-driven models must **not** be read as robustness to real capacity changes. CC/EKF carry a fixed calibrated capacity and therefore do show that error. ML-vs-EKF comparisons are consequently not like-for-like.
6. Only constant-current tests (0.1 A, and one 1 A discharge): **no dynamic drive-cycle behaviour** was evaluated; low-rate V≈OCV makes the voltage→SOC mapping almost a lookup table, which flatters tree models. The 1 A test (unknown temperature) shows what happens outside that regime (MAE ≈ 13–14 points for trees, 0.7 for the EKF).
7. Confounded conditions: tests recorded Oct 2015 – Feb 2016; the 45 °C capacity (2.022 Ah) < 25 °C (2.157 Ah), contrary to the usual trend – history/ageing effects cannot be separated from temperature.
8. Statistical power: 3 low-current conditions + 1 high-current condition; no confidence intervals across cells.

## Method/tooling
9. The EKF uses constant R₀, R₁, τ and a pseudo-OCV (from a 0.1 A discharge) – SOC- and temperature-dependence of parameters is not modelled. 25 °C EKF numbers on its own identification condition are in-sample; the EKF "matched" rows are optimistic upper bounds.
10. **XGBoost, PyTorch, TensorFlow, Streamlit and pytest were not installed and no network was available.** Consequences: gradient boosting = scikit-learn `HistGradientBoostingRegressor` (not XGBoost); LSTM/GRU = own NumPy implementation (small, 16 units, CPU) validated by gradient checks, not tuned extensively; the dashboard was executed only headlessly with a stub Streamlit (all pages run without error, visuals not inspected); tests were executed with a minimal pytest-compatible runner (`pytest` itself should be run by the user).
11. RNN hyper-parameters were not optimised (fixed 16 units, 40 epochs); the GRU in particular is weak and unstable across conditions. Do not conclude that GRU/LSTM are inherently worse.
12. Fault detectors: the "frozen voltage" rule fires on constant-voltage (CV) charge phases (expected, not a fault) – it is not protocol-aware. Detections on real data are mostly protocol transitions and logger gaps.
13. **The literature references were written from memory without database access and are not verified** (see `report/README.md`).
14. The CALCE raw data are not redistributed .

21. **Augmented replicas** carry no new information; they were not shown to help and must never enter test sets (enforced in code and tests).
22. No LSTM/GRU were run on the ageing data: ~400 samples from 7 batteries do not justify them.
