from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .gnss import CLIGHT


NS_TO_M = 1e-9 * CLIGHT


def _zeros(n: int) -> np.ndarray:
    return np.zeros(int(n), dtype=float)


def bias_column_ns(epoch: pd.DataFrame, column: str, n: int) -> Optional[np.ndarray]:
    """Return a bias column in nanoseconds, replacing NaN with zero."""
    if column not in epoch.columns:
        return None
    values = epoch[column].to_numpy(dtype=float, copy=False)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def bias_column_m(epoch: pd.DataFrame, column: str, n: int) -> Optional[np.ndarray]:
    """Return a bias column converted from nanoseconds to meters."""

    values = bias_column_ns(epoch, column, n)
    if values is None:
        return None
    return values * NS_TO_M


def osb_m(
    epoch: pd.DataFrame,
    obs_col: str,
    n: int,
    *,
    include_station: bool = True,
) -> np.ndarray:
    """Return satellite plus optional receiver OSB for one observable in meters."""
    total = _zeros(n)
    for prefix in ("OSB_", "STA_OSB_"):
        if prefix == "STA_OSB_" and not include_station:
            continue
        values = bias_column_m(epoch, f"{prefix}{obs_col}", n)
        if values is not None:
            total += values
    return total


def osb_correction_m(
    epoch: pd.DataFrame,
    obs_col: str,
    n: int,
    *,
    include_station: bool = True,
) -> np.ndarray:
    """Return the additive observation correction for an OSB-corrected observable."""
    return -osb_m(epoch, obs_col, n, include_station=include_station)


def has_satellite_osb(epoch: pd.DataFrame, obs_cols: Iterable[str]) -> bool:
    """Return True only when all requested satellite OSB columns are available."""
    return all(f"OSB_{obs_col}" in epoch.columns for obs_col in obs_cols)


def dcb_between_m(
    epoch: pd.DataFrame,
    obs_a: str,
    obs_b: str,
    n: int,
    *,
    include_station: bool = True,
) -> Optional[np.ndarray]:
    """Return DSB/DCB(obs_a, obs_b) in meters, including reverse-sign lookup."""
    if obs_a == obs_b:
        return _zeros(n)

    total = _zeros(n)
    found = False
    prefixes = ["BIAS_"]
    if include_station:
        prefixes.append("STA_BIAS_")

    for prefix in prefixes:
        direct = bias_column_m(epoch, f"{prefix}{obs_a}_{obs_b}", n)
        if direct is not None:
            total += direct
            found = True
            continue

        reverse = bias_column_m(epoch, f"{prefix}{obs_b}_{obs_a}", n)
        if reverse is not None:
            total -= reverse
            found = True

    if not found:
        return None
    return total


def split_dual_code_dsb_corrections_m(
    epoch: pd.DataFrame,
    code_1: str,
    code_2: str,
    coeff_1: float,
    coeff_2: float,
    n: int,
    *,
    skip_if_satellite_osb: bool = True,
    include_station: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a two-code DSB onto code observables without applying it to phase."""
    zeros = _zeros(n)
    if skip_if_satellite_osb and has_satellite_osb(epoch, (code_1, code_2)):
        return zeros.copy(), zeros.copy(), zeros.copy()

    total = dcb_between_m(
        epoch,
        code_1,
        code_2,
        n,
        include_station=include_station,
    )
    if total is None:
        return zeros.copy(), zeros.copy(), zeros.copy()

    return coeff_2 * total, coeff_1 * total, total
