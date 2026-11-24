# System architecture

```
CALCE Excel files (data/raw, not versioned)
   │  bms.data.loader  (config-driven, unit conversion, dedup, cache in data/processed/interim)
   ▼
bms.preprocessing   cleaning · segmentation (rest/charge/discharge) · uniform resampling
   ▼
bms.features        SOC reference (calculated) · causal features · windows
   ▼
Estimation layer
   ├─ SOC   bms.models.soc   (CC, EKF on ECM, RF, HGB, LSTM, GRU)      → ESTIMATED
   ├─ SOH   bms.models.soh   (guarded: refuses without verified nominal & cycles) → UNAVAILABLE here
   ├─ Fault bms.models.fault (threshold, IF, OC-SVM, flatline rule; simulator) → STATISTICAL ANOMALY / SIMULATED
   └─ RUL   bms.models.rul   (guarded: refuses without trajectory crossing EOL)  → UNAVAILABLE here
   ▼
bms.integration.engine  BMSEngine.analyze() → BMSReport with provenance tags + warnings
   ▼
dashboard/ (Streamlit)  ·  bms.integration.report 
```
