"""Config-driven loader for CALCE/Arbin Excel exports (standard schema, nothing hardcoded)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bms.utils.config import get_logger, load_dataset_config, repo_root
from bms.utils.errors import DatasetMissingError

log = get_logger(__name__)
STD_COLS = ["time_s", "voltage_v", "current_a", "step", "cycle", "temperature_c"]


def _read_sheet(path: Path, sheet: str, usecols: list[str]) -> pd.DataFrame:
    # CALCE '.xls' exports are sometimes OOXML (zip): detect by magic bytes, not by extension.
    engine = "openpyxl" if zipfile.is_zipfile(path) else "xlrd"
    return pd.read_excel(path, sheet_name=sheet, usecols=lambda c: c in usecols, engine=engine)


def load_raw(entry: dict[str, Any], raw_dir: Path) -> pd.DataFrame:
    """Read configured sheets of one raw file and map to the standard schema (raw file never modified)."""
    path = raw_dir / entry["file"]
    if not path.exists():
        raise DatasetMissingError(f"Required raw file not found: {path}. Provide its location; no substitution is done.")
    cols = entry["columns"]
    wanted = list(cols.values()) + ([entry["dedup_on"]] if entry.get("dedup_on") else [])
    df = pd.concat([_read_sheet(path, s, wanted) for s in entry["sheets"]], ignore_index=True)
    if entry.get("dedup_on"):
        n0 = len(df)
        df = df.drop_duplicates(subset=entry["dedup_on"]).drop(columns=entry["dedup_on"])
        if n0 != len(df):
            log.info("%s: removed %d rows duplicated on %s", entry["id"], n0 - len(df), entry["dedup_on"])
    out = pd.DataFrame(index=df.index)
    for std in STD_COLS:
        out[std] = pd.to_numeric(df[cols[std]], errors="coerce") if std in cols else np.nan
    for k, f in entry.get("scale", {}).items():
        out[k] = out[k] * float(f)
    out.insert(0, "dataset_id", entry["id"])
    return out.reset_index(drop=True)


def load_dataset(dataset_id: str, use_cache: bool = True, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load one dataset (standard schema) with a gz-CSV cache in ``data/processed/interim``."""
    cfg = cfg or load_dataset_config()
    root = repo_root()
    entry = next((d for d in cfg["datasets"] if d["id"] == dataset_id), None)
    if entry is None:
        raise KeyError(f"Unknown dataset id '{dataset_id}'. Known: {[d['id'] for d in cfg['datasets']]}")
    cache = root / cfg["processed_dir"] / "interim" / f"{dataset_id}.csv.gz"
    if use_cache and cache.exists():
        return pd.read_csv(cache)
    df = load_raw(entry, root / cfg["raw_dir"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, compression="gzip")
    return df


def dataset_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    return [d["id"] for d in (cfg or load_dataset_config())["datasets"]]
