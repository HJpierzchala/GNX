from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

RINEX2_SUPPORTED_VERSIONS = {"2.11", "2.12"}

GPS_NAV_INTERNAL_COLUMNS = [
    "SVclockBias",
    "SVclockDrift",
    "SVclockDriftRate",
    "IODE",
    "Crs",
    "DeltaN",
    "M0",
    "Cuc",
    "Eccentricity",
    "Cus",
    "sqrtA",
    "Toe",
    "Cic",
    "Omega0",
    "Cis",
    "Io",
    "Crc",
    "omega",
    "OmegaDot",
    "IDOT",
    "CodesL2",
    "GPSWeek",
    "L2Pflag",
    "SVacc",
    "health",
    "TGD",
    "IODC",
    "TransTime",
]

GPS_NAV_REQUIRED_CORE = [
    "SVclockBias",
    "SVclockDrift",
    "SVclockDriftRate",
    "IODE",
    "Crs",
    "DeltaN",
    "M0",
    "Cuc",
    "Eccentricity",
    "Cus",
    "sqrtA",
    "Toe",
    "Cic",
    "Omega0",
    "Cis",
    "Io",
    "Crc",
    "omega",
    "OmegaDot",
    "IDOT",
    "GPSWeek",
    "TGD",
]


def is_supported_rinex2_version(version: Any) -> bool:
    try:
        return f"{float(version):.2f}" in RINEX2_SUPPORTED_VERSIONS
    except (TypeError, ValueError):
        return False


def _normalized_version(version: Any) -> str:
    try:
        return f"{float(version):.2f}"
    except (TypeError, ValueError):
        return str(version)


def _normalize_sv_code(value: Any, default_system: str = "G") -> str:
    token = str(value).strip()
    if not token:
        return token
    prefix = token[0] if token[0].isalpha() else default_system
    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        return token
    return f"{prefix}{int(digits):02d}"


def _ensure_sv_time_index(df: pd.DataFrame, *, strict: bool) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex) and {"sv", "time"} <= set(df.index.names):
        df = df.reset_index()
    elif "sv" not in df.columns or "time" not in df.columns:
        df = df.reset_index()
    if "sv" not in df.columns or "time" not in df.columns:
        raise ValueError("R2OBSE04: input must contain 'sv' and 'time'.")

    df["sv"] = df["sv"].map(_normalize_sv_code)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["sv", "time"])
    df = df[df["sv"].astype(str).str.startswith("G")]

    duplicated = df.duplicated(subset=["sv", "time"])
    if duplicated.any():
        if strict:
            raise ValueError("R2OBSE04: duplicated (sv, time) after normalization.")
        df = df.loc[~duplicated].copy()

    return df.set_index(["sv", "time"]).sort_index()


def _infer_interval_seconds(df: pd.DataFrame) -> float | None:
    if not isinstance(df.index, pd.MultiIndex) or "time" not in df.index.names:
        return None
    times = (
        pd.Index(df.index.get_level_values("time"))
        .drop_duplicates()
        .sort_values()
    )
    if len(times) < 2:
        return None
    dt = times.to_series().diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return None
    return float(dt.median())


def _build_obs_mapping(columns: set[str], mode: str) -> tuple[dict[str, str | None], str]:
    if mode == "L1":
        c1_source = "C1" if "C1" in columns else "P1" if "P1" in columns else None
        if c1_source is None or "L1" not in columns:
            raise ValueError("R2OBSE03: GPS RINEX2 mode 'L1' requires C1/P1 and L1.")
        return {
            "C1": c1_source,
            "C2": None,
            "L1": "L1",
            "L2": None,
            "D1": "D1" if "D1" in columns else None,
            "D2": None,
            "S1": "S1" if "S1" in columns else None,
            "S2": None,
        }, "single_l1"

    if mode == "L2":
        c2_source = "P2" if "P2" in columns else "C2" if "C2" in columns else None
        if c2_source is None or "L2" not in columns:
            raise ValueError("R2OBSE03: GPS RINEX2 mode 'L2' requires P2/C2 and L2.")
        return {
            "C1": None,
            "C2": c2_source,
            "L1": None,
            "L2": "L2",
            "D1": None,
            "D2": "D2" if "D2" in columns else None,
            "S1": None,
            "S2": "S2" if "S2" in columns else None,
        }, "single_l2"

    if mode != "L1L2":
        raise ValueError(f"R2OBSE03: GPS RINEX2 MVP supports only modes L1, L2 and L1L2, got {mode!r}.")

    if "L1" not in columns or "L2" not in columns:
        raise ValueError("R2OBSE03: GPS RINEX2 mode 'L1L2' requires both L1 and L2.")

    for c1_source, c2_source, profile in (
        ("P1", "P2", "p1p2"),
        ("C1", "P2", "c1p2"),
        ("P1", "C2", "p1c2"),
        ("C1", "C2", "c1c2"),
    ):
        if c1_source in columns and c2_source in columns:
            return {
                "C1": c1_source,
                "C2": c2_source,
                "L1": "L1",
                "L2": "L2",
                "D1": "D1" if "D1" in columns else None,
                "D2": "D2" if "D2" in columns else None,
                "S1": "S1" if "S1" in columns else None,
                "S2": "S2" if "S2" in columns else None,
            }, profile

    raise ValueError("R2OBSE03: GPS RINEX2 mode 'L1L2' requires one of {P1/C1}+{P2/C2}.")


def adapt_rinex2_gps_obs(
    raw_df: pd.DataFrame,
    *,
    header: dict[str, Any] | None,
    rinex_info: dict[str, Any] | None,
    system: str,
    mode: str,
    strict: bool = True,
    bias_policy: str = "safe",
    obs_path: str | None = None,
) -> pd.DataFrame:
    version = _normalized_version((rinex_info or {}).get("version"))
    if not is_supported_rinex2_version(version):
        raise ValueError(f"R2OBSE01: unsupported RINEX2 version {version!r}.")
    if system != "G":
        raise ValueError(f"R2OBSE02: GPS-only MVP does not support system {system!r}.")
    if bias_policy != "safe":
        raise ValueError(f"R2OBSE02: unsupported bias policy {bias_policy!r}.")

    normalized = _ensure_sv_time_index(raw_df, strict=strict)
    columns = {str(col) for col in normalized.columns}
    mapping, profile = _build_obs_mapping(columns, mode=mode)

    required_targets = [target for target in ("C1", "C2", "L1", "L2") if mapping.get(target)]
    out = pd.DataFrame(index=normalized.index)
    ordered_targets = ["C1", "C2", "L1", "L2", "D1", "D2", "S1", "S2"]
    for target in ordered_targets:
        source = mapping.get(target)
        if source is not None and source in normalized.columns:
            out[target] = pd.to_numeric(normalized[source], errors="coerce")

    if required_targets:
        out = out.loc[out[required_targets].notna().all(axis=1)]
    if out.empty:
        raise ValueError("R2OBSE05: no rows remain after GPS RINEX2 normalization.")

    recognized = {"C1", "P1", "C2", "P2", "L1", "L2", "D1", "D2", "S1", "S2"}
    ignored = sorted(
        col for col in columns
        if col[:1] in {"C", "P", "L", "D", "S"} and col not in recognized
    )

    warnings: list[dict[str, str]] = []
    if (mode == "L1L2" and profile != "p1p2") or (mode == "L1" and mapping["C1"] != "C1") or (mode == "L2" and mapping["C2"] != "P2"):
        warnings.append({
            "code": "R2OBSW01",
            "message": f"Fallback GPS RINEX2 code profile {profile!r} selected for {obs_path or '<memory>'}.",
        })
    if ignored:
        warnings.append({
            "code": "R2OBSW02",
            "message": f"Ignoring unsupported GPS RINEX2 observables: {', '.join(ignored)}.",
        })
    warnings.append({
        "code": "R2OBSW03",
        "message": "Exact OSB/DSB bias attachment is disabled for GPS RINEX2 MVP.",
    })

    interval = None
    if header is not None:
        interval = header.get("interval")
        if interval is None and "INTERVAL" in header:
            try:
                interval = float(str(header["INTERVAL"]).strip())
            except ValueError:
                interval = None
    if interval is None:
        interval = _infer_interval_seconds(out)
        if interval is not None:
            warnings.append({
                "code": "R2OBSW04",
                "message": "INTERVAL missing in header; interval inferred from observation epochs.",
            })

    time_of_first_obs = None
    time_of_last_obs = None
    if header is not None:
        time_of_first_obs = header.get("t0")
        time_of_last_obs = header.get("t1")
    if time_of_first_obs is None:
        time_of_first_obs = out.index.get_level_values("time").min()
    if time_of_last_obs is None:
        time_of_last_obs = out.index.get_level_values("time").max()
        warnings.append({
            "code": "R2OBSW05",
            "message": "TIME OF LAST OBS missing in header; using last epoch from data.",
        })

    capabilities = {
        "obs_internal_contract_ready": True,
        "nav_internal_contract_ready": False,
        "mode_L1_ready": {"C1", "L1"}.issubset(out.columns),
        "mode_L2_ready": {"C2", "L2"}.issubset(out.columns),
        "mode_L1L2_ready": {"C1", "C2", "L1", "L2"}.issubset(out.columns),
        "geometry_free_ready": {"C1", "C2", "L1", "L2"}.issubset(out.columns),
        "iono_free_ready": {"C1", "C2", "L1", "L2"}.issubset(out.columns),
        "broadcast_tgd_ready": False,
        "antenna_pco_ready": True,
        "doppler_present": any(col in out.columns for col in ("D1", "D2")),
        "snr_present": any(col in out.columns for col in ("S1", "S2")),
        "exact_signal_identity_preserved": False,
        "exact_bia_ready": False,
    }

    bias_eligibility = {
        "nav_tgd": True,
        "bia_osb_exact": False,
        "bia_dsb_exact": False,
        "attach_bia_in_load_obs_data": False,
        "sat_pco_freq_level": True,
        "rec_pco_freq_level": True,
        "phase_shift_supported": False,
    }

    out.attrs["gnx_adapter"] = {
        "source_format": "RINEX2",
        "source_version": version,
        "system": system,
        "mode": mode,
        "code_profile": profile,
        "column_map": {
            key: {"source": value}
            for key, value in mapping.items()
        },
        "capabilities": capabilities,
        "bias_eligibility": bias_eligibility,
        "warnings": warnings,
        "meta_override": {
            "gps_obs": list(out.columns),
            "gal_obs": None,
            "interval": interval,
            "time_of_first_obs": time_of_first_obs,
            "time_of_last_obs": time_of_last_obs,
            "phase_shift_dict": None,
        },
    }
    return out


def adapt_rinex2_gps_nav(
    raw_nav_df: pd.DataFrame,
    *,
    header: dict[str, Any] | None,
    rinex_info: dict[str, Any] | None,
    system: str,
    strict: bool = True,
    nav_path: str | None = None,
) -> pd.DataFrame:
    version = _normalized_version((rinex_info or {}).get("version"))
    if not is_supported_rinex2_version(version):
        raise ValueError(f"R2NAVE01: unsupported RINEX2 version {version!r}.")
    if system != "G":
        raise ValueError(f"R2NAVE01: GPS-only MVP does not support system {system!r}.")

    normalized = _ensure_sv_time_index(raw_nav_df, strict=strict)

    fill_defaults = {
        "CodesL2": 0.0,
        "L2Pflag": 0.0,
        "SVacc": 0.0,
        "health": 0.0,
        "IODC": 0.0,
    }
    warnings: list[dict[str, str]] = []
    for column, default in fill_defaults.items():
        if column not in normalized.columns:
            normalized[column] = default
            warnings.append({
                "code": "R2NAVW02",
                "message": f"Missing NAV column {column!r} filled with default {default!r} for {nav_path or '<memory>'}.",
            })

    if "TransTime" not in normalized.columns:
        normalized["TransTime"] = normalized.index.get_level_values("time")
        warnings.append({
            "code": "R2NAVW02",
            "message": "Missing NAV column 'TransTime' filled from message epoch.",
        })

    missing_core = [column for column in GPS_NAV_REQUIRED_CORE if column not in normalized.columns]
    if missing_core:
        raise ValueError(f"R2NAVE01: missing GPS NAV columns {missing_core}.")

    for column in GPS_NAV_INTERNAL_COLUMNS:
        if column == "TransTime":
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    out = normalized.loc[:, GPS_NAV_INTERNAL_COLUMNS].copy()
    out = out.loc[out[GPS_NAV_REQUIRED_CORE].notna().all(axis=1)]
    if out.empty:
        raise ValueError("R2NAVE02: no valid GPS NAV rows remain after normalization.")

    capabilities = {
        "obs_internal_contract_ready": False,
        "nav_internal_contract_ready": True,
        "mode_L1_ready": False,
        "mode_L2_ready": False,
        "mode_L1L2_ready": False,
        "geometry_free_ready": False,
        "iono_free_ready": False,
        "broadcast_tgd_ready": True,
        "antenna_pco_ready": True,
        "doppler_present": False,
        "snr_present": False,
        "exact_signal_identity_preserved": False,
        "exact_bia_ready": False,
    }

    bias_eligibility = {
        "nav_tgd": True,
        "bia_osb_exact": False,
        "bia_dsb_exact": False,
        "attach_bia_in_load_obs_data": False,
        "sat_pco_freq_level": True,
        "rec_pco_freq_level": True,
        "phase_shift_supported": False,
    }

    out.attrs["gnx_adapter"] = {
        "source_format": "RINEX2",
        "source_version": version,
        "system": system,
        "mode": None,
        "code_profile": None,
        "column_map": {},
        "capabilities": capabilities,
        "bias_eligibility": bias_eligibility,
        "warnings": warnings,
        "meta_override": {
            "gps_obs": None,
            "gal_obs": None,
            "interval": None,
            "time_of_first_obs": out.index.get_level_values("time").min(),
            "time_of_last_obs": out.index.get_level_values("time").max(),
            "phase_shift_dict": None,
        },
    }
    return out
