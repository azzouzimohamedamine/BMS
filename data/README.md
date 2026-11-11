# Data

Place the four original CALCE files in `data/raw/CALCE/` (the folder is git-ignored; redistribution rights of the CALCE data have not been verified, so they are never committed):

* `02_24_2016_SP20-1_0C_lowcurrentOCV.xls` (an OOXML workbook despite the extension)
* `11_5_2015_low_current_OCV_test_SP20-1.xlsx`
* `11_21_2015_low_current_OCV_test_SP20-1.xlsx`
* `10_16_2015_Initial_capacity_SP20-1.xlsx`

Mapping of files → conditions, columns and units lives in `configs/dataset_config.yaml`. `data/processed/interim/*.csv.gz` caches are generated automatically (git-ignored). If a file is missing the loader raises `DatasetMissingError`; no substitution is made. See `docs/DATASET_AUDIT_REPORT.md`.
