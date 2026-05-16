from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from ..biases import NS_TO_M, bias_column_m
from ..gnss import mode_signals, signal_spec


BiasSource = Literal["product", "gim", "config", "zero"]


@dataclass(frozen=True)
class STECBiasResult:
    """Resolved bias terms applied to one STEC satellite arc.

    Status:
        Active public diagnostic container returned by
        ``apply_stec_bias_policy``.

    Fields:
        ``satellite_m`` and ``station_m`` contain per-sample corrections in
        meters for the selected geometry-free code pair. ``total_m`` is their
        sum and is added to ``P4`` by ``STECMonitor`` before TEC conversion.
        ``satellite_source`` and ``station_source`` describe which fallback
        branch was used: product OSB/DSB, GIM, config, zero or disabled.
        ``code_1`` and ``code_2`` are the concrete code observable columns
        selected for the configured frequency mode.
    """

    total_m: np.ndarray
    satellite_m: np.ndarray
    station_m: np.ndarray
    satellite_source: str
    station_source: str
    code_1: str
    code_2: str


def apply_stec_bias_policy(
    arc: pd.DataFrame,
    *,
    config,
    mode: str,
    system: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, STECBiasResult]:
    """Apply the TECConfig-controlled STEC bias policy to one satellite arc.

    Status:
        Active and numerically sensitive. This is the central bias policy used
        by ``STECMonitor`` for GPS, Galileo and BeiDou arcs.

    Model:
        STEC uses geometry-free code ``P4 = P2 - P1``. DSB/DCB products are
        already differential terms for the selected code pair, while OSB
        products are observable biases and are differenced as
        ``OSB(P1) - OSB(P2)``.

    Sources:
        The fallback order comes from ``TECConfig.stec_bias_sources`` and may
        include product columns, GIM/IONEX DCB columns, manual config values
        and a zero fallback. Missing-bias behavior is controlled by
        ``TECConfig.stec_missing_bias``.

    Returns:
        A copy of ``arc`` with diagnostic bias columns plus an
        ``STECBiasResult`` containing the arrays that should be applied to
        ``P4``.
    """

    out = arc.copy()
    n = len(out)
    zeros = np.zeros(n, dtype=float)
    code_1, code_2 = _code_pair_for_mode(out, mode)
    sv = _arc_sv(out)
    sources = _configured_sources(config)
    enabled = bool(getattr(config, "stec_bias_enabled", True))

    satellite = zeros.copy()
    station = zeros.copy()
    satellite_source = "disabled"
    station_source = "disabled"

    if enabled and bool(getattr(config, "add_dcb", True)):
        satellite, satellite_source = _resolve_satellite_bias(
            out,
            code_1=code_1,
            code_2=code_2,
            config=config,
            sources=sources,
            system=system,
            sv=sv,
            logger=logger,
        )

    if enabled and _station_bias_enabled(config):
        station, station_source = _resolve_station_bias(
            out,
            code_1=code_1,
            code_2=code_2,
            config=config,
            sources=sources,
            system=system,
            sv=sv,
            logger=logger,
        )

    total = satellite + station
    out["stec_sat_bias_m"] = satellite
    out["stec_station_bias_m"] = station
    out["stec_total_bias_m"] = total
    out["stec_sat_bias_source"] = satellite_source
    out["stec_station_bias_source"] = station_source
    out["stec_bias_code_1"] = code_1
    out["stec_bias_code_2"] = code_2

    return out, STECBiasResult(
        total_m=total,
        satellite_m=satellite,
        station_m=station,
        satellite_source=satellite_source,
        station_source=station_source,
        code_1=code_1,
        code_2=code_2,
    )


def _configured_sources(config) -> tuple[BiasSource, ...]:
    """Return validated bias-source fallback order, always ending with zero."""
    sources = tuple(getattr(config, "stec_bias_sources", ("product", "gim", "config", "zero")))
    allowed = {"product", "gim", "config", "zero"}
    unknown = [source for source in sources if source not in allowed]
    if unknown:
        raise ValueError(f"Unsupported STEC bias source(s): {unknown}. Allowed: {sorted(allowed)}")
    if "zero" not in sources:
        sources = (*sources, "zero")
    return sources


def _station_bias_enabled(config) -> bool:
    """Return whether station/receiver bias correction is enabled by config."""
    if not bool(getattr(config, "add_sta_dcb", False)):
        return False
    return getattr(config, "rcv_dcb_source", None) != "none"


def _resolve_satellite_bias(
    arc: pd.DataFrame,
    *,
    code_1: str,
    code_2: str,
    config,
    sources: Iterable[BiasSource],
    system: str,
    sv: str,
    logger: logging.Logger | None,
) -> tuple[np.ndarray, str]:
    """Resolve satellite bias for one code pair using configured fallbacks."""
    n = len(arc)
    for source in sources:
        if source == "product":
            values = _product_pair_bias_m(arc, code_1, code_2, station=False)
            if values is not None and _has_product_signal(values):
                return values, _product_source_name(arc, code_1, code_2, station=False)
        elif source == "gim":
            values = _gim_bias_m(arc, "bias")
            if values is not None:
                return values, "gim"
        elif source == "config":
            manual = _manual_bias_value(getattr(config, "define_satellite_dcb", None), sv=sv, system=system)
            if manual is not None:
                return np.full(n, manual * NS_TO_M, dtype=float), "config"
        elif source == "zero":
            _handle_missing(config, f"No satellite STEC bias for {sv} ({code_1}/{code_2}); using 0.", logger)
            return np.zeros(n, dtype=float), "zero"
    return np.zeros(n, dtype=float), "zero"


def _resolve_station_bias(
    arc: pd.DataFrame,
    *,
    code_1: str,
    code_2: str,
    config,
    sources: Iterable[BiasSource],
    system: str,
    sv: str,
    logger: logging.Logger | None,
) -> tuple[np.ndarray, str]:
    """Resolve station bias for one code pair using configured fallbacks."""
    n = len(arc)
    if getattr(config, "rcv_dcb_source", None) == "calibrate":
        calibrated = _gim_bias_m(arc, "sta_dcb")
        if calibrated is not None:
            return calibrated, "calibrate"

    for source in sources:
        if source == "product":
            values = _product_pair_bias_m(arc, code_1, code_2, station=True)
            if values is not None and _has_product_signal(values):
                return values, _product_source_name(arc, code_1, code_2, station=True)
        elif source == "gim":
            values = _gim_bias_m(arc, "sta_bias")
            if values is not None:
                return values, "gim"
        elif source == "config":
            manual = _manual_bias_value(
                getattr(config, "define_station_dcb", None),
                sv=sv,
                system=system,
                station=getattr(config, "station_name", None),
            )
            if manual is not None:
                return np.full(n, manual * NS_TO_M, dtype=float), "config"
        elif source == "zero":
            _handle_missing(
                config,
                f"No station STEC bias for {getattr(config, 'station_name', None) or '<unknown>'} "
                f"{system} ({code_1}/{code_2}); using 0.",
                logger,
            )
            return np.zeros(n, dtype=float), "zero"
    return np.zeros(n, dtype=float), "zero"


def _product_pair_bias_m(
    arc: pd.DataFrame,
    code_1: str,
    code_2: str,
    *,
    station: bool,
) -> np.ndarray | None:
    """Return product bias in meters from OSB or DSB columns when available."""
    osb = _product_osb_pair_m(arc, code_1, code_2, station=station)
    if osb is not None:
        return osb
    return _product_dsb_pair_m(arc, code_1, code_2, station=station)


def _product_source_name(arc: pd.DataFrame, code_1: str, code_2: str, *, station: bool) -> str:
    """Return a diagnostic source label for product-derived bias columns."""
    prefix = "STA_" if station else ""
    if f"{prefix}OSB_{code_1}" in arc.columns and f"{prefix}OSB_{code_2}" in arc.columns:
        return "product_osb"
    return "product_dsb"


def _product_osb_pair_m(
    arc: pd.DataFrame,
    code_1: str,
    code_2: str,
    *,
    station: bool,
) -> np.ndarray | None:
    """Difference OSB observable biases as OSB(code_1) - OSB(code_2)."""
    prefix = "STA_OSB_" if station else "OSB_"
    osb_1 = bias_column_m(arc, f"{prefix}{code_1}", len(arc))
    osb_2 = bias_column_m(arc, f"{prefix}{code_2}", len(arc))
    if osb_1 is None or osb_2 is None:
        return None
    return osb_1 - osb_2


def _product_dsb_pair_m(
    arc: pd.DataFrame,
    code_1: str,
    code_2: str,
    *,
    station: bool,
) -> np.ndarray | None:
    """Read direct or reversed differential DSB/DCB product columns."""
    prefix = "STA_BIAS_" if station else "BIAS_"
    direct = bias_column_m(arc, f"{prefix}{code_1}_{code_2}", len(arc))
    if direct is not None:
        return direct
    reverse = bias_column_m(arc, f"{prefix}{code_2}_{code_1}", len(arc))
    if reverse is not None:
        return -reverse
    return None


def _gim_bias_m(arc: pd.DataFrame, column: str) -> np.ndarray | None:
    """Read a GIM/IONEX bias column in ns and convert it to meters."""
    if column not in arc.columns:
        return None
    values = arc[column].to_numpy(dtype=float, copy=False)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) * NS_TO_M


def _has_product_signal(values: np.ndarray) -> bool:
    """Return True when a product bias array contains a non-zero finite signal."""
    finite = np.asarray(values, dtype=float)
    return bool(np.isfinite(finite).any() and np.nanmax(np.abs(finite)) > 0.0)


def _manual_bias_value(value, *, sv: str, system: str, station: str | None = None) -> float | None:
    """Resolve scalar or dictionary manual bias value in nanoseconds."""
    if value is None:
        return None
    if isinstance(value, dict):
        keys = [sv, sv.split("_", 1)[0], system]
        if station:
            keys.extend([station, station.upper()])
        keys.append("default")
        for key in keys:
            if key in value and value[key] is not None:
                return float(value[key])
        return None
    return float(value)


def _handle_missing(config, message: str, logger: logging.Logger | None) -> None:
    """Apply configured missing-bias behavior without changing numeric fallback."""
    mode = getattr(config, "stec_missing_bias", "warn_zero")
    if mode == "raise":
        raise KeyError(message)
    if mode == "warn_zero":
        warnings.warn(message, UserWarning, stacklevel=3)
        (logger or logging.getLogger(__name__)).warning(message)
    elif mode != "zero":
        raise ValueError(f"Unsupported stec_missing_bias mode: {mode!r}")


def _code_pair_for_mode(arc: pd.DataFrame, mode: str) -> tuple[str, str]:
    """Resolve concrete code columns for a dual-frequency STEC mode."""
    signals = mode_signals(mode)
    if len(signals) != 2:
        raise ValueError(f"STEC bias policy requires a dual-frequency mode, got {mode!r}.")
    return tuple(_code_column_for_signal(arc, signal) for signal in signals)  # type: ignore[return-value]


def _code_column_for_signal(arc: pd.DataFrame, signal: str) -> str:
    """Find the best available code observable column for one signal."""
    spec = signal_spec(signal)
    for suffix in spec.suffix_priority:
        col = f"{spec.code_prefix}{suffix}"
        if col in arc.columns:
            return col
    matches = [col for col in arc.columns if col.startswith(spec.code_prefix)]
    if matches:
        return sorted(matches)[0]
    return f"{spec.code_prefix}{spec.suffix_priority[0]}"


def _arc_sv(arc: pd.DataFrame) -> str:
    """Return the satellite identifier for diagnostics."""
    if isinstance(arc.index, pd.MultiIndex) and "sv" in arc.index.names:
        return str(arc.index.get_level_values("sv")[0])
    if "sv" in arc.columns:
        return str(arc["sv"].iloc[0])
    return "<unknown>"
