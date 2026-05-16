from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CLIGHT = 299_792_458.0


@dataclass(frozen=True)
class SignalSpec:
    name: str
    system: str
    frequency_hz: float
    code_prefix: str
    phase_prefix: str
    antex_frequency: str
    suffix_priority: tuple[str, ...]

    @property
    def wavelength_m(self) -> float:
        return CLIGHT / self.frequency_hz


SYSTEM_NAMES = {
    "G": "GPS",
    "E": "Galileo",
    "C": "BeiDou",
}


SIGNALS: dict[str, SignalSpec] = {
    # GPS
    "L1": SignalSpec("L1", "G", 1575.42e6, "C1", "L1", "G01", ("W", "C", "P", "Y", "L", "X", "Q")),
    "L2": SignalSpec("L2", "G", 1227.60e6, "C2", "L2", "G02", ("W", "L", "P", "Y", "X", "C", "Q")),
    "L5": SignalSpec("L5", "G", 1176.45e6, "C5", "L5", "G05", ("X", "Q", "I")),
    # Galileo
    "E1": SignalSpec("E1", "E", 1575.42e6, "C1", "L1", "E01", ("C", "X", "Z", "B", "A")),
    "E5a": SignalSpec("E5a", "E", 1176.45e6, "C5", "L5", "E05", ("Q", "I", "X")),
    "E5b": SignalSpec("E5b", "E", 1207.14e6, "C7", "L7", "E07", ("Q", "I", "X")),
    # BeiDou / BDS. RINEX 3/4 band identifiers are not GPS-like:
    # B1I uses band 2, B1C uses band 1, B2a uses band 5, B2I/B2b use band 7,
    # and B3I uses band 6.
    "B1I": SignalSpec("B1I", "C", 1561.098e6, "C2", "L2", "C02", ("I", "Q", "X")),
    "B1C": SignalSpec("B1C", "C", 1575.42e6, "C1", "L1", "C01", ("X", "P", "D", "S", "L", "Z")),
    "B2a": SignalSpec("B2a", "C", 1176.45e6, "C5", "L5", "C05", ("X", "P", "D")),
    "B2I": SignalSpec("B2I", "C", 1207.14e6, "C7", "L7", "C07", ("I", "Q", "X")),
    "B2b": SignalSpec("B2b", "C", 1207.14e6, "C7", "L7", "C07", ("Z", "P", "D")),
    "B3I": SignalSpec("B3I", "C", 1268.52e6, "C6", "L6", "C06", ("I", "Q", "X")),
}


MODE_SIGNALS: dict[str, tuple[str, ...]] = {
    "L1": ("L1",),
    "L2": ("L2",),
    "L5": ("L5",),
    "L1L2": ("L1", "L2"),
    "L1L5": ("L1", "L5"),
    "L2L5": ("L2", "L5"),
    "E1": ("E1",),
    "E5a": ("E5a",),
    "E5b": ("E5b",),
    "E1E5a": ("E1", "E5a"),
    "E1E5b": ("E1", "E5b"),
    "E5aE5b": ("E5a", "E5b"),
    "B1I": ("B1I",),
    "B1C": ("B1C",),
    "B2a": ("B2a",),
    "B2I": ("B2I",),
    "B2b": ("B2b",),
    "B3I": ("B3I",),
    "B1IB2I": ("B1I", "B2I"),
    "B1IB3I": ("B1I", "B3I"),
    "B1CB2a": ("B1C", "B2a"),
    "B1CB2b": ("B1C", "B2b"),
    "B1CB3I": ("B1C", "B3I"),
    "B2aB2b": ("B2a", "B2b"),
}


DEFAULT_MODE_BY_SYSTEM = {
    "G": "L1L2",
    "E": "E1E5a",
    "C": "B1IB3I",
}


SUPPORTED_MODES_BY_SYSTEM = {
    system: tuple(mode for mode, signals in MODE_SIGNALS.items() if SIGNALS[signals[0]].system == system)
    for system in SYSTEM_NAMES
}


def normalize_systems(sys: str | Iterable[str]) -> set[str]:
    systems = {sys} if isinstance(sys, str) else set(sys)
    unknown = systems - set(SYSTEM_NAMES)
    if unknown:
        raise ValueError(f"Unsupported GNSS systems: {sorted(unknown)}. Allowed: {sorted(SYSTEM_NAMES)}")
    return systems


def system_name(system: str) -> str:
    try:
        return SYSTEM_NAMES[system]
    except KeyError as exc:
        raise ValueError(f"Unsupported GNSS system: {system!r}") from exc


def mode_signals(mode: str) -> tuple[str, ...]:
    try:
        return MODE_SIGNALS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported GNSS frequency mode: {mode!r}") from exc


def dual_mode_signals(mode: str) -> tuple[str, str]:
    signals = mode_signals(mode)
    if len(signals) != 2:
        raise ValueError(f"Mode {mode!r} is not dual-frequency.")
    return signals


def signal_spec(signal: str) -> SignalSpec:
    try:
        return SIGNALS[signal]
    except KeyError as exc:
        raise ValueError(f"Unsupported GNSS signal: {signal!r}") from exc


def frequency_hz(signal: str) -> float:
    return signal_spec(signal).frequency_hz


def wavelength_m(signal: str) -> float:
    return signal_spec(signal).wavelength_m


def validate_mode_for_system(mode: str, system: str) -> None:
    signals = mode_signals(mode)
    systems = {signal_spec(signal).system for signal in signals}
    if systems != {system}:
        raise ValueError(
            f"Mode {mode!r} belongs to {sorted(systems)}, not system {system!r} ({system_name(system)})."
        )


def ionosphere_free_coefficients(sig1: str, sig2: str) -> tuple[float, float]:
    f1 = frequency_hz(sig1)
    f2 = frequency_hz(sig2)
    den = f1**2 - f2**2
    return f1**2 / den, f2**2 / den


def mode_ionosphere_free_coefficients(mode: str) -> tuple[float, float]:
    sig1, sig2 = dual_mode_signals(mode)
    return ionosphere_free_coefficients(sig1, sig2)


def mode_layout(mode: str) -> list[dict[str, str]]:
    signals = mode_signals(mode)
    if len(signals) == 1:
        spec = signal_spec(signals[0])
        return [
            {
                "signal": spec.name,
                "code": spec.code_prefix,
                "phase": spec.phase_prefix,
                "freq": spec.name,
                "antex": spec.antex_frequency,
                "rec_pco_col": "pco_los",
            }
        ]

    if mode in {"E1E5b", "E5aE5b"}:
        pco_cols = ("pco_los_l1", "pco_los_l5")
    else:
        pco_cols = ("pco_los_l1", "pco_los_l2")
    out: list[dict[str, str]] = []
    for signal, pco_col in zip(signals, pco_cols):
        spec = signal_spec(signal)
        out.append(
            {
                "signal": spec.name,
                "code": spec.code_prefix,
                "phase": spec.phase_prefix,
                "freq": spec.name,
                "antex": spec.antex_frequency,
                "rec_pco_col": pco_col,
            }
        )
    return out


def first_signal(mode: str) -> str:
    return mode_signals(mode)[0]


def code_prefixes(mode: str) -> tuple[str, ...]:
    return tuple(signal_spec(signal).code_prefix for signal in mode_signals(mode))


def phase_prefixes(mode: str) -> tuple[str, ...]:
    return tuple(signal_spec(signal).phase_prefix for signal in mode_signals(mode))


def mode_system(mode: str) -> str:
    signals = mode_signals(mode)
    systems = {signal_spec(signal).system for signal in signals}
    if len(systems) != 1:
        raise ValueError(f"Mixed-system mode definitions are not supported: {mode!r}")
    return next(iter(systems))


def mode_from_config(config, system: str) -> str:
    if system == "G":
        return getattr(config, "gps_freq", DEFAULT_MODE_BY_SYSTEM["G"])
    if system == "E":
        return getattr(config, "gal_freq", DEFAULT_MODE_BY_SYSTEM["E"])
    if system == "C":
        return getattr(config, "bds_freq", DEFAULT_MODE_BY_SYSTEM["C"])
    raise ValueError(f"Unsupported GNSS system: {system!r}")


def is_bds_geo(sv: str) -> bool:
    token = str(sv).split("_", 1)[0]
    if not token.startswith("C"):
        return False
    try:
        prn = int(token[1:3])
    except ValueError:
        return False
    # BDS-2 GEO uses C01-C05; BDS-3 GEO uses C59-C63.
    return 1 <= prn <= 5 or 59 <= prn <= 63


def bds_orbit_type(sv: str) -> str:
    token = str(sv).split("_", 1)[0]
    if not token.startswith("C"):
        raise ValueError(f"Not a BeiDou satellite id: {sv!r}")
    try:
        prn = int(token[1:3])
    except ValueError as exc:
        raise ValueError(f"Invalid BeiDou satellite id: {sv!r}") from exc
    if is_bds_geo(token):
        return "GEO"
    # BDS-2 IGSO: C06-C10, C13, C16. BDS-3 IGSO: C38-C40.
    if prn in {6, 7, 8, 9, 10, 13, 16, 38, 39, 40}:
        return "IGSO"
    return "MEO"
