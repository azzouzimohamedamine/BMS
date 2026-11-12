"""Loader + audit for the Kaggle ageing datasets (kept separate from the CALCE loader).

Every returned frame carries a ``provenance`` column: EXPERIMENTAL_DERIVED, AUGMENTED or SIMULATED.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bms.data.loader import find_raw_file
from bms.utils.config import load_yaml, repo_root
from bms.utils.errors import DatasetMissingError

REQUIRED = ["battery_id", "cycle", "capacity"]


def load_aging_config() -> dict[str, Any]:
    return load_yaml(repo_root() / "configs" / "aging_dataset_config.yaml")


def _entry(cfg: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    e = next((d for d in cfg["datasets"] if d["id"] == dataset_id), None)
    if e is None:
        raise KeyError(f"Unknown ageing dataset '{dataset_id}'. Known: {[d['id'] for d in cfg['datasets']]}")
    return e


def missing_aging_files(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_aging_config(); raw = repo_root() / cfg["raw_dir"]
    return [d["file"] for d in cfg["datasets"] if find_raw_file(d["file"], raw, (".csv",)) is None]


def load_aging(dataset_id: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read one ageing CSV, check required columns/duplicates, sort by battery and cycle."""
    cfg = cfg or load_aging_config(); e = _entry(cfg, dataset_id); raw = repo_root() / cfg["raw_dir"]
    path = find_raw_file(e["file"], raw, (".csv",))
    if path is None:
        raise DatasetMissingError(f"Ageing file not found: {raw / e['file']}. Copy it into data/raw/KAGGLE/ (no substitution is done).")
    df = pd.read_csv(path)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"{path.name}: missing required columns {miss}")
    if "is_augmented" not in df.columns:
        df["is_augmented"] = False
    if "aug_rep" not in df.columns:
        df["aug_rep"] = 0
    df["is_augmented"] = df["is_augmented"].astype(bool)
    df["dataset_id"], df["provenance"] = dataset_id, e["provenance"]
    n0 = len(df); df = df.drop_duplicates(subset=["battery_id", "aug_rep", "cycle"])
    if len(df) != n0:
        raise ValueError(f"{path.name}: {n0 - len(df)} duplicated (battery, replica, cycle) rows")
    return df.sort_values(["battery_id", "aug_rep", "cycle"]).reset_index(drop=True)


def audit_batteries(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Rule-based quality audit per battery (original rows only). Excludes short or corrupted trajectories."""
    cfg = cfg or load_aging_config(); r = cfg["quality_rules"]; eol = cfg["eol_fraction"]
    rows = []
    for bid, g in df[~df.is_augmented].groupby("battery_id"):
        cap = g.sort_values("cycle").capacity.to_numpy(); n = len(cap); reasons = []
        ratio_max, ratio_min = float(cap.max() / cap[0]), float(cap.min() / cap[0])
        if n < r["min_cycles"]:
            reasons.append(f"only {n} cycle(s) (< {r['min_cycles']})")
        if ratio_max > r["max_capacity_over_first"]:
            reasons.append(f"capacity reaches {ratio_max:.2f}x its first-cycle value (inconsistent trajectory)")
        rows.append({"battery_id": bid, "n_cycles": n, "q0": float(cap[0]), "max_over_first": ratio_max, "min_over_first": ratio_min,
                     "reaches_eol": bool(ratio_min <= eol), "ambient_c": float(g.ambient_temperature.iloc[0]) if "ambient_temperature" in g else np.nan,
                     "usable": not reasons, "reason": "; ".join(reasons) or "ok"})
    return pd.DataFrame(rows)


def usable_batteries(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frame restricted to usable batteries, audit table)."""
    a = audit_batteries(df, cfg)
    return df[df.battery_id.isin(a.loc[a.usable, "battery_id"])].copy(), a
