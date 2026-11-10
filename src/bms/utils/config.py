"""Configuration, paths, seeds and logging helpers."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SEED = 42


def repo_root() -> Path:
    """Return the repository root (directory containing ``configs/``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError("Cannot locate repository root (no 'configs' directory found).")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_dataset_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or repo_root() / "configs" / "dataset_config.yaml")


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or repo_root() / "configs" / "model_config.yaml")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "bms", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger
