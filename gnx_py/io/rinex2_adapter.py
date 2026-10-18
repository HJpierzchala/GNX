"""RINEX 2 band mapping: the RINEX 2 observation and navigation frames of the reader turned into the internal
contract of the RINEX 3 path.

Observations (:func:`adapt_rinex2_obs`): the frame of one system as :func:`gnx_py.rinex_obs.read_obs` gives it
(index ``['sv', 'time']``, RINEX 2 codes as written: ``C1``, ``P1``, ``L1`` ...) becomes the band columns of
the RINEX 3 contract (``C1`` from P1 or C1, ``L1`` ...; Galileo E1 band 1, E5a band 5, E5b band 7), index
``['sv', 'time']`` lexsorted. The choices are reported in ``attrs['gnx_adapter']``. The notes that change what a
run computes or show what it cannot use (R2OBSW01 fallback code profile, R2OBSW02 observables of the mode's bands
not used, with the columns taken) are also emitted as one ``UserWarning`` per file and system. R2OBSW04 (no
usable header INTERVAL) and R2OBSW05 (no TIME OF LAST OBS) stay attrs notes: the interval and the first/last epoch come from the data for every RINEX version,
so those records change nothing; a header INTERVAL that disagrees with the data is warned by
``io.GNSSDataProcessor2``.
"""
from __future__ import annotations

import warnings as _warnings
from typing import Any

import pandas as pd

from ..core.gnss import SUPPORTED_MODES_BY_SYSTEM, mode_signals, signal_spec, system_name

# Every RINEX 2 minor version shares one observation-code scheme - two-character
# names with no signal attribute - so the reader accepts the whole 2.x range
# rather than an allow-list. Archives are not tidy about the minor number: BKG
# writes a bare "2" for some stations and "2.11" for others in the same day.
RINEX2_VERSION_RANGE = (2.0, 3.0)

RINEX_VERSION_LABEL = "RINEX VERSION / TYPE"

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
    """Return whether ``version`` is a RINEX 2 version this reader handles."""
    try:
        value = float(version)
    except (TypeError, ValueError):
        return False
    low, high = RINEX2_VERSION_RANGE
    return low <= value < high


def parse_rinex_version_line(line: Any) -> float | None:
    """Return the version encoded in a ``RINEX VERSION / TYPE`` header line.

    The version lives in the first nine columns as ``%9.2f``.
    """
    if not isinstance(line, str):
        return None
    try:
        return float(line[:9])
    except ValueError:
        return None


def resolve_rinex_version(
    header: dict[Any, Any] | None = None,
    rinex_info: dict[Any, Any] | None = None,
) -> float | None:
    """Return the RINEX version of a file, seeing past Hatanaka compression.

    ``georinex.rinexinfo`` reports the version found on the first header line,
    and it reads that line without expanding Hatanaka. For a CRINEX file the
    first line is ``CRINEX VERS   / TYPE``, so ``rinexinfo`` returns the
    *CRINEX* version - 1.0 over RINEX 2 and 3.0 over RINEX 3 - rather than the
    RINEX version. A RINEX 2.11 observation file compressed to CRINEX therefore
    looks like version 1.0 and fails every ``is_supported_rinex2_version``
    check, which is why the RINEX 2 adapter used to be skipped for ``.crx`` and
    ``.??d`` inputs.

    The parsed header keeps the original ``RINEX VERSION / TYPE`` line for both
    RINEX 2 and RINEX 3 observation files, so that line is authoritative here
    and ``rinex_info`` is only a fallback (navigation headers do not keep the
    raw line, and navigation files are never Hatanaka-compressed).

    Returns:
        The version as a float, or None when neither source carries one.
    """
    version = parse_rinex_version_line((header or {}).get(RINEX_VERSION_LABEL))
    if version is not None:
        return version

    for source in (header, rinex_info):
        if not source:
            continue
        try:
            return float(source["version"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


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


def _ensure_sv_time_index(df: pd.DataFrame, *, strict: bool, system: str = "G",
                          where: str = "<memory>") -> pd.DataFrame:
    """``df`` with the index ``['sv', 'time']`` lexsorted: the levels may come in either order (or as columns),
    satellite ids are normalised ('G 7' -> 'G07', a blank system letter -> ``system``), rows of other systems are
    dropped. A duplicated (sv, time) raises R2OBSE04 when ``strict`` (the reader drops duplicates with a warning,
    so this is a guard), else the first record is kept. ``where`` names the file in the errors."""
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex) and {"sv", "time"} <= set(df.index.names):
        df = df.reset_index()
    elif "sv" not in df.columns or "time" not in df.columns:
        df = df.reset_index()
    if "sv" not in df.columns or "time" not in df.columns:
        raise ValueError(f"R2OBSE04: {where}: input must contain 'sv' and 'time'.")

    # one normalisation per distinct id (a station-day has ~30 ids and ~30000 rows)
    ids = {value: _normalize_sv_code(value, system) for value in pd.unique(df["sv"])}
    df["sv"] = df["sv"].map(ids)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["sv", "time"])
    # A RINEX 2 mixed file holds every constellation in one record, so keep only
    # the one being adapted even when the caller already filtered upstream.
    df = df[df["sv"].astype(str).str.startswith(system)]

    duplicated = df.duplicated(subset=["sv", "time"])
    if duplicated.any():
        if strict:
            raise ValueError(f"R2OBSE04: {where}: duplicated (sv, time) after normalization.")
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


# RINEX 2 observation codes carry no signal attribute, so a band exposes at
# most one pseudorange per code type and the adapter has to pick between the
# P-code and the C/A-or-civil code.
#
# GPS: P1/P2 come first, which matches both the broadcast TGD convention and
# the RINEX 3 selection in ``GNSSDataProcessor2.select_obs_types_gps``, whose
# suffix priority starts at "W" and so picks C1W/C2W - the RINEX 3 spelling of
# P1/P2. Preferring C1 instead would silently shift a single-frequency run by
# the P1-C1 bias, about half a metre. C/A-only files still work: the fallback
# picks C1 and the adapter reports the profile through the R2OBSW01 warning.
# Band 5 (L5) arrives with RINEX 2.11 and has no P-code counterpart.
#
# Galileo: RINEX 2.11 gives Galileo its own band numbering, and it is not the
# GPS one - E1 is band 1, E5a band 5, E5b band 7 (band 8 is the E5 AltBOC and
# band 6 is E6, neither of which GNX models). There are no P-codes, so every
# band has a single source.
_R2_CODE_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "G": {
        "C1": ("P1", "C1"),
        "C2": ("P2", "C2"),
        "C5": ("C5",),
    },
    "E": {
        "C1": ("C1",),
        "C5": ("C5",),
        "C7": ("C7",),
    },
}

#: Constellations that RINEX 2 can carry and GNX can map. BeiDou is absent by
#: construction: RINEX 2.11 defines no BeiDou band numbering (converters such as
#: gfzrnx write BeiDou into 2.11 files all the same, with their own numbering).
RINEX2_OBS_SYSTEMS: frozenset[str] = frozenset(_R2_CODE_SOURCES)

#: Why BeiDou cannot be read from a RINEX 2 file (shared with ``io.load_and_filter``).
RINEX2_BEIDOU_MESSAGE = ("BeiDou is not defined in RINEX 2.11 (band numbering undefined); convert the file to "
                         "RINEX 3")


def unmapped_system_message(system: str) -> str:
    """Readable reason why ``system`` cannot be read from RINEX 2 observations."""
    if system == "C":
        return RINEX2_BEIDOU_MESSAGE
    return (f"RINEX 2 observations are not mapped for system {system!r}; mapped systems are "
            f"{sorted(_R2_CODE_SOURCES)}")

# Columns are emitted grouped by kind and then by band, which keeps the column
# order of the pre-existing L1/L2 output unchanged.
_R2_BANDS: tuple[str, ...] = ("1", "2", "5", "7")
_R2_TARGET_ORDER: tuple[str, ...] = tuple(
    f"{kind}{band}" for kind in ("C", "L", "D", "S") for band in _R2_BANDS
)


def _band_of(target: str) -> str:
    return target[1]


def _build_obs_mapping(
    columns: set[str],
    mode: str,
    populated: set[str] | None = None,
    system: str = "G",
    where: str = "<memory>",
) -> tuple[dict[str, str | None], str, bool]:
    """Map the observation columns of a RINEX 2 file onto internal names.

    The internal contract uses the RINEX 3 style band names (``C1``/``L1``,
    ``C5``/``L5`` and so on) without a signal attribute, so downstream prefix
    matching finds them the same way it finds ``C1W`` or ``C5Q``. Band numbers
    are taken from the signal metadata, which already knows that Galileo E5b is
    band 7 rather than band 2.

    Args:
        columns: observation columns the header declares.
        mode: the configured frequency mode.
        populated: columns that actually carry at least one value. Headers do
            promise observables the receiver never recorded - BJFS declares P1
            and leaves it empty for the whole day - and choosing an empty
            column would drop every epoch. Defaults to ``columns``.
        system: the constellation being adapted.
        where: the file named in the errors (R2OBSE03).

    Returns:
        The source-column mapping, a profile label naming the chosen code
        sources, and whether every band got its preferred source.
    """
    if populated is None:
        populated = columns
    sources = _R2_CODE_SOURCES.get(system)
    if sources is None:
        raise ValueError(f"R2OBSE03: {where}: {unmapped_system_message(system)}.")
    try:
        signals = mode_signals(mode)
    except ValueError as exc:
        raise ValueError(f"R2OBSE03: {where}: unsupported RINEX2 mode {mode!r}.") from exc

    foreign = sorted({signal_spec(signal).system for signal in signals} - {system})
    if foreign:
        raise ValueError(
            f"R2OBSE03: {where}: mode {mode!r} belongs to system(s) {foreign}, not {system!r}."
        )

    mapping: dict[str, str | None] = {target: None for target in _R2_TARGET_ORDER}
    chosen: list[str] = []
    preferred: list[str] = []

    for signal in signals:
        spec = signal_spec(signal)
        code_target, phase_target = spec.code_prefix, spec.phase_prefix
        candidates = sources.get(code_target)
        if candidates is None:
            raise ValueError(
                f"R2OBSE03: {where}: signal {signal!r} has no RINEX 2 band equivalent "
                f"(code {code_target!r})."
            )

        # A declared-but-empty column is no use, so prefer one that has data and
        # only fall back to header presence to keep the error message honest.
        code_source = next((name for name in candidates if name in populated), None)
        if code_source is None:
            code_source = next((name for name in candidates if name in columns), None)
        if code_source is None or phase_target not in columns:
            raise ValueError(
                f"R2OBSE03: {where}: {system!r} RINEX2 mode {mode!r} requires "
                f"{'/'.join(candidates)} and {phase_target}; file has "
                f"{sorted(columns)} (with data: {sorted(populated)})."
            )

        mapping[code_target] = code_source
        mapping[phase_target] = phase_target
        band = _band_of(code_target)
        for kind in ("D", "S"):
            optional = f"{kind}{band}"
            if optional in columns:
                mapping[optional] = optional

        chosen.append(code_source)
        preferred.append(candidates[0])

    if len(signals) == 1:
        # 'single_l1' for GPS L1, 'single_e5b' for Galileo E5b: the signal name
        # rather than the band number, which says nothing on its own.
        profile = f"single_{signals[0].lower()}"
    else:
        profile = "".join(name.lower() for name in chosen)

    return mapping, profile, tuple(chosen) == tuple(preferred)


def _header_interval_s(header: dict[str, Any] | None) -> float | None:
    """INTERVAL of the header dict in seconds; None when missing, zero, negative or unreadable."""
    if not header:
        return None
    value = header.get("interval")
    if value is None and "INTERVAL" in header:
        value = str(header["INTERVAL"])[:10]
    try:
        value = float(str(value).strip()) if value is not None else None
    except ValueError:
        return None
    return value if value is not None and value > 0 else None


def _emit_notes(notes: list[dict[str, str]], *, system: str, mode: str, obs_path: str | None) -> None:
    """One ``UserWarning`` with the notes of one (file, system); nothing without notes."""
    if not notes:
        return
    text = "; ".join(f"{note['code']} {note['message']}" for note in notes)
    _warnings.warn(f"RINEX 2 observations {obs_path or '<memory>'}, {system_name(system)} (mode {mode!r}): {text}",
                   UserWarning, stacklevel=3)


#: Notes emitted as a warning by :func:`adapt_rinex2_obs`; R2OBSW03 (bias rule), R2OBSW04 (INTERVAL) and R2OBSW05
#: (TIME OF LAST OBS) are attrs notes only (see the module docstring).
EMITTED_NOTES = ("R2OBSW01", "R2OBSW02")


def adapt_rinex2_obs(
    raw_df: pd.DataFrame,
    *,
    header: dict[str, Any] | None,
    rinex_info: dict[str, Any] | None,
    system: str,
    mode: str,
    strict: bool = True,
    bias_policy: str = "safe",
    obs_path: str | None = None,
    warn: bool = True,
) -> pd.DataFrame:
    """Normalize one constellation of a RINEX 2 observation file.

    Handles GPS and Galileo, the two constellations RINEX 2.11 carries that GNX
    also models. BeiDou is not defined in RINEX 2.11 (no band numbering), and
    GLONASS/SBAS are outside GNX's signal tables.

    ``raw_df``: the system's frame of the reader (index ``['sv', 'time']`` or
    ``['time', 'sv']``, or the two as columns); ``header``: the header dict
    (``ObsHeader.georinex_view()``: ``RINEX VERSION / TYPE``, ``interval``,
    ``TIME OF LAST OBS``). Returns the band columns with the index
    ``['sv', 'time']`` lexsorted (the contract of every observation frame); the
    interval and the first/last epoch in ``attrs['gnx_adapter']['meta_override']``
    come from the data (the header INTERVAL only for fewer than 2 epochs).
    ``warn``: emit the notes R2OBSW01/02 as one ``UserWarning`` (every note is
    in ``attrs['gnx_adapter']['warnings']``).
    """
    where = obs_path or "<memory>"          # every R2OBSE error names the file
    version = _normalized_version(
        resolve_rinex_version(header=header, rinex_info=rinex_info)
    )
    if not is_supported_rinex2_version(version):
        raise ValueError(f"R2OBSE01: {where}: unsupported RINEX2 version {version!r}.")
    if system not in _R2_CODE_SOURCES:
        raise ValueError(f"R2OBSE02: {unmapped_system_message(system)} ({where}).")
    if bias_policy != "safe":
        raise ValueError(f"R2OBSE02: {where}: unsupported bias policy {bias_policy!r}.")

    normalized = _ensure_sv_time_index(raw_df, strict=strict, system=system, where=where)
    if normalized.empty:
        raise ValueError(
            f"R2OBSE04: {where} holds no {system!r} satellites."
        )
    columns = {str(col) for col in normalized.columns}
    populated = {col for col in columns if normalized[col].notna().any()}
    mapping, profile, preferred_sources = _build_obs_mapping(
        columns, mode=mode, populated=populated, system=system, where=where
    )

    code_and_phase = {"C", "L"}
    required_targets = [
        target for target, source in mapping.items()
        if source is not None and target[:1] in code_and_phase
    ]
    out = pd.DataFrame(index=normalized.index)
    for target in _R2_TARGET_ORDER:
        source = mapping.get(target)
        if source is not None and source in normalized.columns:
            out[target] = pd.to_numeric(normalized[source], errors="coerce")

    if required_targets:
        out = out.loc[out[required_targets].notna().all(axis=1)]
    if out.empty:
        coverage = ", ".join(
            f"{target}<-{mapping[target]}: {int(pd.to_numeric(normalized[mapping[target]], errors='coerce').notna().sum())}"
            for target in required_targets
        )
        raise ValueError(
            f"R2OBSE05: {where}: no epoch has every observable the mode needs, so nothing "
            f"remains after {system!r} RINEX2 normalization. "
            f"Non-empty counts: {coverage}."
        )

    # The index stays ['sv', 'time'] lexsorted, the contract of every observation
    # frame (RINEX 2, 3 and 4 alike, bug-fix stage 10). The order once had to be
    # (time, sv) because mark_outages wrote its flags back by position after a
    # time sort; it now restores the caller's row order, and the session results
    # do not depend on the order (measured: SPP <= 9.4e-9 m, PPP 0 m).

    # RINEX 2 observables with data on the mode's own bands that the mode does not use (the
    # alternatives the band choice decided against, e.g. C1 when P1 gives the band-1 code;
    # other bands and Doppler are not reported: a mixed RINEX 2 header lists the types of
    # every system, and GNX never uses Doppler)
    used_sources = {source for source in mapping.values() if source is not None}
    mode_bands = {signal_spec(signal).code_prefix[1:] for signal in mode_signals(mode)}
    ignored = sorted(
        col for col in populated
        if col[:1] in {"C", "P", "L", "S"} and col[1:2] in mode_bands and col not in used_sources
    )
    taken = ", ".join(f"{target} <- {source}" for target, source in mapping.items()
                      if source is not None and target in out.columns)

    warnings: list[dict[str, str]] = []
    if not preferred_sources:
        warnings.append({
            "code": "R2OBSW01",
            "message": (
                f"fallback code profile {profile!r} (the P-code observable of a band is missing or empty, "
                f"its C/A or civil code is used: the P1-C1 / P2-C2 bias is corrected only by a bias product "
                f"that gives it)"
            ),
        })
    if ignored:
        warnings.append({
            "code": "R2OBSW02",
            "message": (f"RINEX 2 observables with data on the bands of mode {mode!r} not used: "
                        f"{', '.join(ignored)} (columns taken: {taken})"),
        })
    warnings.append({
        "code": "R2OBSW03",
        "message": (f"{system} biases attached through the RINEX 2 tracking rule "
                    f"(biases.resolve_rinex2_tracking)"),
    })

    # interval and first/last epoch: from the data (bug-fix stage 10); the header
    # INTERVAL only when the frame has fewer than 2 epochs
    header_interval = _header_interval_s(header)
    interval = _infer_interval_seconds(out)
    if interval is None:
        interval = header_interval
    if header_interval is None and interval is not None:
        warnings.append({
            "code": "R2OBSW04",
            "message": f"INTERVAL missing or unusable in the header; {interval:g} s taken from the observation "
                       f"epochs",
        })

    times = out.index.get_level_values("time")
    time_of_first_obs = times.min()
    time_of_last_obs = times.max()
    if header is not None and not (header.get("TIME OF LAST OBS") or header.get("t1")):
        warnings.append({
            "code": "R2OBSW05",
            "message": "TIME OF LAST OBS missing in the header; the last epoch is taken from the data",
        })
    if warn:
        _emit_notes([note for note in warnings if note["code"] in EMITTED_NOTES], system=system, mode=mode,
                    obs_path=obs_path)

    produced = set(out.columns)
    dual_ready = len([target for target in required_targets if target[:1] == "C"]) == 2

    def _mode_ready(candidate: str) -> bool:
        """Whether the produced columns could serve ``candidate`` as well."""
        try:
            signals = mode_signals(candidate)
        except ValueError:
            return False
        if {signal_spec(signal).system for signal in signals} != {system}:
            return False
        return all(
            {signal_spec(signal).code_prefix, signal_spec(signal).phase_prefix}
            .issubset(produced)
            for signal in signals
        )

    capabilities = {
        "obs_internal_contract_ready": True,
        "nav_internal_contract_ready": False,
        "geometry_free_ready": dual_ready,
        "iono_free_ready": dual_ready,
        "broadcast_tgd_ready": False,
        "antenna_pco_ready": True,
        "doppler_present": any(col.startswith("D") for col in produced),
        "snr_present": any(col.startswith("S") for col in produced),
        "exact_signal_identity_preserved": False,
        "exact_bia_ready": False,
    }
    capabilities.update({
        f"mode_{candidate}_ready": _mode_ready(candidate)
        for candidate in SUPPORTED_MODES_BY_SYSTEM.get(system, ())
    })

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
            # Only the adapted system's observable list is known here; a mixed
            # file is adapted once per constellation, so leaving the others out
            # keeps each report from overwriting its neighbour's metadata.
            "gps_obs": list(out.columns) if system == "G" else None,
            "gal_obs": list(out.columns) if system == "E" else None,
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
    """Normalize a RINEX 2 GPS navigation file.

    Navigation stays GPS-only on purpose, and here the limit really is the
    format: RINEX 2 navigation files hold one constellation each and define
    only GPS, GLONASS and SBAS message layouts. Galileo ephemerides arrive with
    RINEX 3, so a Galileo run reading RINEX 2 observations needs a RINEX 3 (or
    RINEX 4) navigation file for its broadcast orbits - which is fine, the two
    inputs are read independently.
    """
    where = nav_path or "<memory>"          # every R2NAVE error names the file
    version = _normalized_version(
        resolve_rinex_version(header=header, rinex_info=rinex_info)
    )
    if not is_supported_rinex2_version(version):
        raise ValueError(f"R2NAVE01: {where}: unsupported RINEX2 version {version!r}.")
    if system != "G":
        raise ValueError(
            f"R2NAVE01: {where}: RINEX 2 navigation is GPS-only and cannot serve system "
            f"{system!r}; Galileo ephemerides exist only from RINEX 3 onwards."
        )

    normalized = _ensure_sv_time_index(raw_nav_df, strict=strict, system="G", where=where)

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
        # seconds of the GPS week, as RINEX 3 navigation gives it (Sunday-based week)
        epoch = pd.DatetimeIndex(normalized.index.get_level_values("time"))
        normalized["TransTime"] = (((epoch.dayofweek + 1) % 7) * 86400
                                   + (epoch - epoch.normalize()).total_seconds()).to_numpy(float)
        warnings.append({
            "code": "R2NAVW02",
            "message": "Missing NAV column 'TransTime' filled from message epoch.",
        })

    missing_core = [column for column in GPS_NAV_REQUIRED_CORE if column not in normalized.columns]
    if missing_core:
        raise ValueError(f"R2NAVE01: {where}: missing GPS NAV columns {missing_core}.")

    # TransTime stays numeric (seconds of the GPS week, like the RINEX 3 path): read as a
    # datetime it was taken as nanoseconds since 1970 and lost its meaning for the record
    # selector (tie-break between records with the same Toe)
    for column in GPS_NAV_INTERNAL_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    out = normalized.loc[:, GPS_NAV_INTERNAL_COLUMNS].copy()
    out = out.loc[out[GPS_NAV_REQUIRED_CORE].notna().all(axis=1)]
    if out.empty:
        raise ValueError(f"R2NAVE02: {where}: no valid GPS NAV rows remain after normalization.")

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
