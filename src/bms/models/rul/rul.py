"""RUL (Module 4): EOL definition, chronological RUL labelling and models, with feasibility guards.

RUL = EOL_cycle - current_cycle. It is only defined when the observed capacity trajectory actually crosses the
EOL threshold. CALCE SP20-1 files used here contain ONE cycle per condition: ``assert_rul_feasible`` raises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bms.utils.errors import InsufficientDataError

MIN_CYCLES = 30


def find_eol_cycle(cycles: np.ndarray, capacity_ah: np.ndarray, nominal_ah: float, threshold: float = 0.8) -> int | None:
    """First cycle where capacity <= threshold*nominal (None if never reached: no invented EOL)."""
    below = np.where(np.asarray(capacity_ah) <= threshold * nominal_ah)[0]
    return int(np.asarray(cycles)[below[0]]) if below.size else None


def assert_rul_feasible(cycles: np.ndarray, capacity_ah: np.ndarray, nominal_ah: float | None, nominal_verified: bool,
                        threshold: float = 0.8) -> int:
    """Return the EOL cycle or raise ``InsufficientDataError`` explaining why RUL cannot be defined."""
    cycles = np.unique(cycles)
    if nominal_ah is None or not nominal_verified:
        raise InsufficientDataError("Nominal capacity not verified: EOL threshold cannot be defined.")
    if len(cycles) < MIN_CYCLES:
        raise InsufficientDataError(f"{len(cycles)} distinct cycle(s) available (< {MIN_CYCLES}): no degradation trajectory.")
    eol = find_eol_cycle(cycles, capacity_ah, nominal_ah, threshold)
    if eol is None:
        raise InsufficientDataError("Capacity never reaches the EOL threshold: true RUL labels would have to be invented.")
    return eol


def rul_labels(cycles: np.ndarray, eol_cycle: int) -> np.ndarray:
    """RUL = EOL cycle - current cycle (clipped at 0)."""
    return np.maximum(eol_cycle - np.asarray(cycles), 0)


def chronological_split(n: int, train: float = 0.6, val: float = 0.2) -> dict[str, slice]:
    a, b = int(n * train), int(n * (train + val))
    return {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}
