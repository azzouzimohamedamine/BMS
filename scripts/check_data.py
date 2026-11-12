"""Diagnose the raw data folder BEFORE running anything. Usage: python scripts/check_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bms.data.loader import find_raw_file, load_raw  # noqa: E402
from bms.utils.config import load_dataset_config, repo_root  # noqa: E402

cfg = load_dataset_config(); raw_dir = repo_root() / cfg["raw_dir"]
print(f"Raw folder: {raw_dir}  (exists: {raw_dir.exists()})")
present = sorted(p.name for p in raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep") if raw_dir.exists() else []
print("Files found there:", present or "NONE")
ok = True
for e in cfg["datasets"]:
    p = find_raw_file(e["file"], raw_dir)
    if p is None:
        ok = False; print(f"[MISSING] {e['id']:10s} -> {e['file']}"); continue
    try:
        df = load_raw(e, raw_dir)
        print(f"[OK]      {e['id']:10s} {p.name}  rows={len(df)}  V=[{df.voltage_v.min():.2f},{df.voltage_v.max():.2f}]")
    except Exception as exc:  # noqa: BLE001
        ok = False; print(f"[ERROR]   {e['id']:10s} {p.name}: {exc}")
print("\nALL GOOD - you can run the experiments." if ok else "\nFix the items above, then re-run this script.")
sys.exit(0 if ok else 1)
