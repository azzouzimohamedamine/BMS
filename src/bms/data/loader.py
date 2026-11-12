"""Config-driven loader for CALCE/Arbin Excel exports (standard schema, nothing hardcoded)."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bms.utils.config import get_logger, load_dataset_config, repo_root
from bms.utils.errors import DatasetMissingError

log = get_logger(__name__)
STD_COLS = ["time_s", "voltage_v", "current_a", "step", "cycle", "temperature_c"]


def _norm(name: str) -> str:
    """Normalise a file name: case, extension, browser duplicate suffix ' (1)', separators."""
    stem = Path(name).stem.lower().strip()
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return re.sub(r"[\s_\-]+", "_", stem)


def find_raw_file(filename: str, raw_dir: Path) -> Path | None:
    """Exact path first; otherwise the SAME file renamed by the OS/browser (case, '(1)' suffix, .xls/.xlsx
    swap, sub-folder). A different dataset is never substituted: normalised names must be equal."""
    exact = raw_dir / filename
    if exact.exists():
        return exact
    if raw_dir.exists():
        target = _norm(filename)
        for p in sorted(raw_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".xls", ".xlsx") and not p.name.startswith("~$") and _norm(p.name) == target:
                log.warning("Expected '%s' but using the identically named file '%s'. Consider renaming it.", filename, p)
                return p
    return None


def _missing_message(filename: str, raw_dir: Path) -> str:
    present = sorted(p.name for p in raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep") if raw_dir.exists() else []
    return (f"Required raw file not found: {raw_dir / filename}\n"
            f"Folder searched: {raw_dir} (exists: {raw_dir.exists()})\n"
            f"Files currently there: {present if present else 'none'}\n"
            "Copy the original CALCE file with exactly this name into that folder (no substitution is done).")


def missing_files(cfg: dict[str, Any] | None = None) -> list[str]:
    """File names from the configuration that cannot be found in the raw folder (cache is ignored)."""
    cfg = cfg or load_dataset_config()
    raw_dir = repo_root() / cfg["raw_dir"]
    return [d["file"] for d in cfg["datasets"] if find_raw_file(d["file"], raw_dir) is None]


def _read_sheet(path: Path, sheet: str, usecols: list[str]) -> pd.DataFrame:
    # CALCE '.xls' exports are sometimes OOXML (zip): detect by magic bytes, not by extension.
    engine = "openpyxl" if zipfile.is_zipfile(path) else "xlrd"
    return pd.read_excel(path, sheet_name=sheet, usecols=lambda c: c in usecols, engine=engine)


def load_raw(entry: dict[str, Any], raw_dir: Path) -> pd.DataFrame:
    """Read configured sheets of one raw file and map to the standard schema (raw file never modified)."""
    path = find_raw_file(entry["file"], raw_dir)
    if path is None:
        raise DatasetMissingError(_missing_message(entry["file"], raw_dir))
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
