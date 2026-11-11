"""Integration tests on the real CALCE files (skipped automatically when the raw files are absent)."""
import numpy as np

from bms.utils.config import load_dataset_config, repo_root


def _have_raw():
    cfg = load_dataset_config()
    return all((repo_root() / cfg["raw_dir"] / d["file"]).exists() for d in cfg["datasets"])


def test_all_datasets_load_with_standard_schema_and_units():
    if not _have_raw():
        return
    from bms.data.loader import dataset_ids, load_dataset
    for d in dataset_ids():
        df = load_dataset(d)
        assert {"time_s", "voltage_v", "current_a"} <= set(df.columns)
        assert 2.0 < df.voltage_v.min() < df.voltage_v.max() < 4.5          # volts, not millivolts
        assert df.current_a.abs().max() < 5                                  # amperes, not milliamperes
        assert df.time_s.is_monotonic_increasing


def test_condition_preparation_reference_soc():
    if not _have_raw():
        return
    from bms.preprocessing.pipeline import prepare_condition
    for d in ("T00C_lowI", "T25C_lowI", "T45C_lowI", "INIT_1A"):
        c = prepare_condition(d)
        f = c.frame
        assert abs(f.soc_ref.iloc[0] - 1) < 1e-6 and f.soc_ref.iloc[-1] < 0.01
        assert 1.5 < c.capacity_ah < 2.5
        assert (f.i < 0).all()
