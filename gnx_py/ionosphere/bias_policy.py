from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from ..models.biases import NS_TO_M
from ..core.gnss import code_signal, mode_signals, signal_spec, system_name
from .ionex import IONEX_DEFAULT_DCB_REFERENCE, DCBReference


BiasSource = Literal["product", "gim", "config", "zero"]


def leveling_weights(ev_deg) -> np.ndarray:
    """Weights of the carrier-to-code levelling and of the per-arc bias constants: ``sin^4(E)``.

    The library formerly wrote them ``1 / S^2`` with ``S = 1 / (err sin^2 E)``, i.e. ``err^2 sin^4 E``; the
    constant ``err^2`` (``TECConfig.tec_zenith_err``, deprecated) cancels in every weighted mean.
    Non-finite elevations get weight 0. Public so that test oracles use the same weights.
    Also available as ``gnx_py.ionosphere.config.leveling_weights``.
    """
    w = np.sin(np.deg2rad(np.asarray(ev_deg, dtype=float))) ** 4
    return np.where(np.isfinite(w), w, 0.0)


@dataclass(frozen=True)
class STECBiasResult:
    """Resolved bias terms applied to one STEC satellite arc.

    Status:
        Active public diagnostic container returned by
        ``apply_stec_bias_policy``.

    Fields:
        One source gives the bias of a whole arc: the first
        source of the chain with a value on any sample of the arc; its values are reduced to
        one constant, the levelling-weighted (``leveling_weights``, sin^4 E) mean over the
        samples it covers (``satellite_coverage``/``station_coverage``: their share), and the
        constant is applied to every sample. A bias that changes inside a continuous phase arc
        (two daily products across midnight: CAS1 2024-131 -> 132 up to 1.38 ns, 3.2 TECU on
        E1/E5a) would otherwise put a step into the levelled STEC.
        ``satellite_arc_m`` and ``station_arc_m`` are these constants in meters for the
        selected geometry-free code pair; ``satellite_m``, ``station_m`` and ``total_m`` hold
        them per sample (``total_m`` is what ``STECMonitor`` adds to the levelled phase).
        ``satellite_sample_m`` / ``station_sample_m`` are diagnostics: the chosen source's own
        value per sample (NaN where it gives none).
        ``satellite_source`` and ``station_source`` name the source: product_osb,
        product_dsb, product_dsb_chain (a chain of the product's DSBs), gim,
        calibrate (station), config, zero or disabled; a product mixing OSB and DSB values in
        one arc gives the labels joined with '+'.
        ``satellite_datum`` is the datum of the satellite biases: ``product:<file>`` (zero-mean
        satellite biases of the product), ``ionex:<file>``, ``user`` or ``none``.
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
    satellite_arc_m: float = 0.0
    station_arc_m: float = 0.0
    satellite_coverage: float = 0.0
    station_coverage: float = 0.0
    satellite_sample_m: np.ndarray | None = None
    station_sample_m: np.ndarray | None = None
    satellite_datum: str = "none"


def apply_stec_bias_policy(
    arc: pd.DataFrame,
    *,
    config,
    mode: str,
    system: str,
    logger: logging.Logger | None = None,
    gim_reference: DCBReference | None = None,
    notified: set | None = None,
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
        and a zero fallback. The station bias starts with the source chosen by
        ``TECConfig.rcv_dcb_source`` ('calibrate': the receiver DCB calibrated
        against the GIM, column ``sta_dcb``; 'gim': the IONEX receiver DCB;
        'defined': ``define_station_dcb``; None: no preferred source) and goes
        on with ``stec_bias_sources``; a preferred source without a value gives
        one message per ``notified`` set (to the log only when the bias falls
        to 'zero' anyway, which has its own warning, or when the calibration
        was not done for a reason already warned about).

    Missing values:
        A bias the source does not give is missing, never 0: an OSB pair needs
        both OSBs (a missing OSB of the second code is not ``OSB(P1) - 0``), a
        NaN GIM DCB (satellite not in the IONEX DCB block, system without DCBs)
        or a NaN calibration is missing as well. A source without a value on
        any sample of the arc is skipped; the first source with a value gives
        the whole arc one constant (levelling-weighted mean over the samples it
        covers, coverage recorded). An arc that reaches 'zero' gets 0
        under ``TECConfig.stec_missing_bias`` ('warn_zero': a UserWarning naming
        the satellite or station, once per satellite or station and
        ``notified`` set, further arcs to the log; 'zero': silent; 'raise':
        KeyError). Product
        columns holding only zeros count as absent (placeholder of an analysis
        centre). A product without the pair's DSB takes the shortest chain of its
        DSBs (e.g. C1W-C5Q = (C1W-C2W) + (C2W-C1C) + (C1C-C5Q), label
        ``product_dsb_chain``; a missing link leaves the sample without a value,
        a station chain needs every STA_ link).

    Datum:
        The satellite and station biases of an arc must refer to one datum (the
        zero-mean condition of the satellite biases): a product station bias is
        used only with product satellite biases (one bias file), an IONEX
        receiver DCB only with the IONEX satellite DCBs; otherwise that station
        source is skipped with one message and the next one applies. When no
        station source of the satellite biases' datum has a value, the first
        skipped one is used instead of 0, with one warning (datum mismatch ~1
        TECU against a missing receiver DCB of 8-22 TECU). A user
        station value ('config') with product satellite biases gives one
        warning. The calibrated receiver DCB ('calibrate') is computed with the
        satellite biases of the chain, so it is in their datum. With
        ``rcv_dcb_source='gim'`` the satellite chain tries 'gim' first (when the
        IONEX receiver DCB applies to the arc's pair and 'gim' is in
        ``stec_bias_sources``; one log line), so that the chosen receiver DCB
        meets its own satellite DCBs instead of being skipped.

    GIM DCB:
        The IONEX DCBs refer to one code pair per system (``gim_reference``,
        read from the IONEX header; None means the IONEX defaults, GPS P1-P2).
        They are used only when the arc's pair is on the same two signals
        (Galileo E5a and E5b count as one, see ``gim_dcb_applies``); otherwise
        the source is skipped (one warning per ``notified`` set) and the next
        configured source applies. An arc without GIM DCB values (all NaN) is
        not checked: the GIM gives no bias for it (next source).

    Returns:
        A copy of ``arc`` with diagnostic bias columns plus an
        ``STECBiasResult`` containing the arrays that should be applied to
        ``P4``.
    """

    out = arc.copy()
    n = len(out)
    code_1, code_2 = _code_pair_for_mode(out, mode)
    sv = _arc_sv(out)
    sources = _configured_sources(config)
    enabled = bool(getattr(config, "stec_bias_enabled", True))
    weights = (leveling_weights(out["ev"].to_numpy(dtype=float)) if "ev" in out.columns
               else np.ones(n, dtype=float))

    satellite_chain = _BiasChain.disabled(n)
    station_chain = _BiasChain.disabled(n)
    # stec_missing_bias='zero' silences the bias messages: the GIM pair notes then go to the log only
    quiet = getattr(config, "stec_missing_bias", "warn_zero") == "zero"
    satellite_sources = _satellite_order(out, config, sources, code_1=code_1, code_2=code_2, system=system,
                                         gim_reference=gim_reference, enabled=enabled, logger=logger,
                                         notified=notified)

    if enabled and bool(getattr(config, "add_dcb", True)):
        satellite_chain = _resolve_satellite_bias(
            out,
            code_1=code_1,
            code_2=code_2,
            config=config,
            sources=satellite_sources,
            system=system,
            sv=sv,
            logger=logger,
            gim_ok=lambda: gim_dcb_applies(code_1, code_2, system=system, reference=gim_reference,
                                           logger=logger, notified=notified, quiet=quiet),
            notified=notified,
            weights=weights,
        )
    satellite_datum = _satellite_datum(config, satellite_chain.label)

    if enabled and _station_bias_enabled(config):
        station_chain = _resolve_station_bias(
            out,
            code_1=code_1,
            code_2=code_2,
            config=config,
            sources=sources,
            system=system,
            sv=sv,
            logger=logger,
            gim_ok=lambda: gim_dcb_applies(code_1, code_2, system=system, reference=gim_reference,
                                           station=getattr(config, "station_name", None) or "",
                                           logger=logger, notified=notified, quiet=quiet),
            # a calibration that ran and gave NaN (column 'sta_dcb', Calibration warned) or was skipped
            # for the GIM DCB pair (pair warning, same message set as TECSession's check) is reported
            calibration_reported=lambda: "sta_dcb" in out.columns or (
                _has_values(out, "bias") and not gim_dcb_applies(
                    code_1, code_2, system=system, reference=gim_reference, logger=logger,
                    notified=notified, quiet=quiet)),
            notified=notified,
            quiet=quiet,
            weights=weights,
            satellite_kind=_source_kind(satellite_chain.label),
            satellite_datum=satellite_datum,
        )

    satellite = np.full(n, satellite_chain.arc_value, dtype=float)
    station = np.full(n, station_chain.arc_value, dtype=float)
    total = satellite + station
    out["stec_sat_bias_m"] = satellite
    out["stec_station_bias_m"] = station
    out["stec_total_bias_m"] = total
    out["stec_sat_bias_source"] = satellite_chain.label
    out["stec_station_bias_source"] = station_chain.label
    out["stec_sat_bias_coverage"] = satellite_chain.coverage
    out["stec_station_bias_coverage"] = station_chain.coverage
    out["stec_bias_datum"] = satellite_datum
    out["stec_bias_code_1"] = code_1
    out["stec_bias_code_2"] = code_2

    return out, STECBiasResult(
        total_m=total,
        satellite_m=satellite,
        station_m=station,
        satellite_source=satellite_chain.label,
        station_source=station_chain.label,
        code_1=code_1,
        code_2=code_2,
        satellite_arc_m=float(satellite_chain.arc_value),
        station_arc_m=float(station_chain.arc_value),
        satellite_coverage=float(satellite_chain.coverage),
        station_coverage=float(station_chain.coverage),
        satellite_sample_m=satellite_chain.values,
        station_sample_m=station_chain.values,
        satellite_datum=satellite_datum,
    )


def _file_names(path) -> str:
    """File name(s) of a path setting (one path or a list of paths) for the datum label."""
    from pathlib import Path

    if path is None:
        return ""
    paths = path if isinstance(path, (list, tuple)) else [path]
    return "+".join(Path(str(p)).name for p in paths)


def _source_kind(label: str) -> str:
    """Kind of an arc's bias source label: product, gim, calibrate, config, zero or disabled."""
    return "product" if str(label).startswith("product") else str(label)


def _satellite_datum(config, label: str) -> str:
    """Datum of the satellite biases of an arc: ``product:<file>`` (zero-mean satellite biases of the bias
    product), ``ionex:<file>`` (IONEX DCB block), ``user`` (define_satellite_dcb) or ``none``."""
    kind = _source_kind(label)
    if kind == "product":
        return f"product:{_file_names(getattr(config, 'dcb_path', None))}"
    if kind == "gim":
        return f"ionex:{_file_names(getattr(config, 'gim_path', None))}"
    if kind == "config":
        return "user"
    return "none"


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


class _Mute:
    """Logger stand-in that drops messages (a check repeated later with its own messages)."""

    def info(self, *args, **kwargs) -> None:
        pass

    warning = info


def _satellite_order(arc: pd.DataFrame, config, sources: tuple, *, code_1: str, code_2: str, system: str,
                     gim_reference: DCBReference | None, enabled: bool, logger: logging.Logger | None,
                     notified: set | None) -> tuple:
    """Satellite source order of an arc: ``stec_bias_sources``, with 'gim' moved first when the station bias is
    the IONEX receiver DCB (``rcv_dcb_source='gim'``, a value in the arc, its own pair).

    The IONEX receiver DCB refers to the GIM's satellite DCBs (one datum per arc). With the
    product first it would be skipped and the station bias would come from the product or fall to 0 (BRUX
    2024-035 G L1L2: CAS1 has no C1W-C2W receiver DSB, 0 instead of the GIM's 22.4 TECU). A 'gim' left out of
    ``stec_bias_sources`` is not added; an arc without a GIM satellite DCB still takes the next source (its
    station source is then chosen by the datum rule)."""
    if not (enabled and bool(getattr(config, "add_dcb", True)) and _station_bias_enabled(config)
            and getattr(config, "rcv_dcb_source", None) == "gim" and "gim" in sources and sources[0] != "gim"
            and _has_values(arc, "sta_bias")):
        return sources
    station = getattr(config, "station_name", None) or ""
    if not gim_dcb_applies(code_1, code_2, system=system, reference=gim_reference, station=station,
                           logger=_Mute(), notified=None, quiet=True):
        return sources
    _notify(f"rcv_dcb_source='gim': the satellite biases of {system_name(system)} ({code_1}-{code_2}) come from the "
            f"IONEX DCB block first, the datum of the GIM receiver DCB of {station or '<unknown>'} (stec_bias_sources "
            f"{tuple(sources)} otherwise)", logger, notified, quiet=True)
    return ("gim",) + tuple(source for source in sources if source != "gim")


class _BiasChain:
    """Bias of one arc from one source.

    Sources are tried in the order of the chain; the first one with a finite value on any sample
    takes the whole arc: its values become one constant, the levelling-weighted (sin^4 E) mean over
    the samples it covers (``coverage``: their share), applied to every sample. A source without a
    value on any sample leaves the arc to the next one. ``values`` keeps the chosen source's own
    per-sample values (diagnostic, NaN where it gives none)."""

    def __init__(self, n: int, weights: np.ndarray | None = None):
        self.n = n
        self.weights = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
        self.values = np.full(n, np.nan, dtype=float)
        self.arc_value: float | None = None
        self.label = ""
        self.coverage = 0.0

    @classmethod
    def disabled(cls, n: int) -> "_BiasChain":
        chain = cls(n)
        chain.arc_value, chain.label = 0.0, "disabled"
        return chain

    def complete(self) -> bool:
        return self.arc_value is not None

    def take(self, values: np.ndarray | None, label) -> bool:
        """Take the arc when ``values`` has a finite sample (and no earlier source took it); True when taken."""
        if values is None or self.complete():
            return False
        values = np.asarray(values, dtype=float)
        covered = np.isfinite(values)
        if not covered.any():
            return False
        labels = label if isinstance(label, np.ndarray) else np.full(self.n, label, dtype=object)
        w = self.weights[covered]
        v = values[covered]
        ok = np.isfinite(w) & (w > 0)
        self.arc_value = float(np.dot(w[ok], v[ok]) / w[ok].sum()) if ok.any() else float(np.mean(v))
        self.values = np.where(covered, values, np.nan)
        self.label = _source_summary(labels[covered], ("product",))
        self.coverage = float(covered.mean())
        return True

    def zero(self, config, what: str, arc: pd.DataFrame, logger: logging.Logger | None,
             notified: set | None = None) -> None:
        """An arc no source gave a value gets 0 under stec_missing_bias (warn_zero / zero / raise). One message
        per ``what`` (satellite or station and code pair) and run (``notified``): further arcs of the same
        satellite or station go to the log at INFO level."""
        if self.complete():
            return
        key = ("stec_missing_bias", what)
        if notified is not None and key in notified:
            if getattr(config, "stec_missing_bias", "warn_zero") == "warn_zero":
                (logger or logging.getLogger(__name__)).info("%s; using 0.", what)
        else:
            if notified is not None:
                notified.add(key)
            _handle_missing(config, what, logger)
        self.arc_value, self.label, self.coverage = 0.0, "zero", 0.0
        self.values = np.zeros(self.n, dtype=float)


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
    gim_ok=lambda: True,
    notified: set | None = None,
    weights: np.ndarray | None = None,
) -> _BiasChain:
    """Resolve the satellite bias of one code pair along the configured chain (one source per arc)."""
    n = len(arc)
    chain = _BiasChain(n, weights)
    # one message per satellite and run, not per arc ('G05_0_1' -> 'G05')
    missing = f"No satellite STEC bias for {sv.split('_', 1)[0]} ({code_1}/{code_2})"
    for source in sources:
        if source == "product":
            chain.take(*_product_pair_bias_m(arc, code_1, code_2, station=False))
        elif source == "gim":
            # IONEX DCBs of another code pair are not a bias of this pair (ION-029); the pair is
            # checked only when the GIM has a DCB for this satellite. A NaN DCB (satellite not in
            # the IONEX DCB block, system without DCBs) is missing, not 0 (ION-010)
            if _has_values(arc, "bias") and gim_ok():
                chain.take(_bias_column_m(arc, "bias"), "gim")
        elif source == "config":
            manual = _manual_bias_value(getattr(config, "define_satellite_dcb", None), sv=sv, system=system)
            if manual is not None:
                chain.take(np.full(n, manual * NS_TO_M, dtype=float), "config")
        elif source == "zero":
            chain.zero(config, missing, arc, logger, notified)
        if chain.complete():
            break
    chain.zero(config, missing, arc, logger, notified)
    return chain


#: first station source per rcv_dcb_source (None: the stec_bias_sources order only)
_RCV_FIRST_SOURCE = {"calibrate": "calibrate", "gim": "gim", "defined": "config"}

#: station sources allowed per kind of the arc's satellite source (one datum per arc):
#: a product station bias only with product satellite biases (the same bias file), an IONEX receiver DCB
#: only with the IONEX satellite DCBs; without satellite biases (zero, disabled) any station source
_STATION_DATUM = {"product": {"product": True, "gim": False, "config": False},
                  "gim": {"product": False, "gim": True, "config": False}}


def _datum_ok(station_source: str, satellite_kind: str) -> bool:
    return _STATION_DATUM.get(station_source, {}).get(satellite_kind, True)


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
    gim_ok=lambda: True,
    calibration_reported=lambda: False,
    notified: set | None = None,
    quiet: bool = False,
    weights: np.ndarray | None = None,
    satellite_kind: str = "disabled",
    satellite_datum: str = "none",
) -> _BiasChain:
    """Resolve the station bias of one code pair (one source per arc).

    The source chosen by ``rcv_dcb_source`` comes first, then ``stec_bias_sources``
    without it. A chosen source without a value gives one message per ``notified`` set, to the
    log only when the arc falls to 'zero' (that warning covers it) or when
    ``calibration_reported()`` says the missing calibration was already warned about.
    A product or IONEX station bias in another datum than the arc's satellite biases
    (``satellite_kind``) is skipped with one message (``_STATION_DATUM``) and the next source
    applies; when no other station source has a value, the first skipped one is used rather
    than 0, with one warning (a datum mismatch is ~1 TECU, a missing receiver DCB 8-22 TECU at
    BRUX / NKLG). A user value ('config') with product satellite biases gives one warning.
    """
    n = len(arc)
    station = getattr(config, "station_name", None) or "<unknown>"
    rcv = getattr(config, "rcv_dcb_source", None)
    first = _RCV_FIRST_SOURCE.get(rcv)
    order = _station_order(config, sources)
    chain = _BiasChain(n, weights)
    missing = f"No station STEC bias for {station} {system} ({code_1}/{code_2})"
    first_left = False   # the chosen source left the arc without a value (message after the chain)
    skipped: list = []   # station biases in another datum: (what, values, labels), message after the chain
    other = f"{_DATUM_TEXT.get(satellite_kind, satellite_kind)} ({satellite_datum})"

    def datum_rejected(source: str, what: str, values, labels) -> bool:
        if _datum_ok(source, satellite_kind):
            return False
        skipped.append((what, values, labels))
        return True

    def datum_fallback() -> None:
        # no station source in the satellite biases' datum has a value: the first one skipped, not 0
        if chain.complete() or not skipped:
            return
        what, values, labels = skipped.pop(0)
        chain.take(values, labels)
        _notify(f"Station bias of {station} ({system_name(system)}, {code_1}-{code_2}) from {what} used with the "
                f"satellite biases of {other}, another datum: no station source of their datum has a value "
                f"(station and satellite biases should refer to one datum; the mismatch is ~1 TECU, a station "
                f"bias of 0 would miss the whole receiver DCB)", logger, notified, quiet)

    for source in order:
        missing_before = not chain.complete()
        rejected = False
        if source == "calibrate":
            # receiver DCB calibrated against the GIM with the chain's satellite biases (TECSession);
            # NaN = calibration without data
            chain.take(_bias_column_m(arc, "sta_dcb"), "calibrate")
        elif source == "product":
            values, labels = _product_pair_bias_m(arc, code_1, code_2, station=True)
            if values is not None:
                rejected = datum_rejected(
                    "product", f"the bias product ({_file_names(getattr(config, 'dcb_path', None))})", values, labels)
                if not rejected:
                    chain.take(values, labels)
        elif source == "gim":
            if _has_values(arc, "sta_bias"):
                rejected = not gim_ok() or datum_rejected(
                    "gim", "the IONEX receiver DCB", _bias_column_m(arc, "sta_bias"), "gim")
                if not rejected:
                    chain.take(_bias_column_m(arc, "sta_bias"), "gim")
        elif source == "config":
            manual = _manual_bias_value(getattr(config, "define_station_dcb", None), sv=sv, system=system,
                                        station=getattr(config, "station_name", None))
            if manual is not None and chain.take(np.full(n, manual * NS_TO_M, dtype=float), "config") \
                    and satellite_kind == "product":
                _notify(f"define_station_dcb (user value) of {station} ({system_name(system)}, {code_1}-{code_2}) "
                        f"used with the satellite biases of {satellite_datum}: the station value must refer to "
                        f"the datum of that product (zero-mean satellite biases)", logger, notified, quiet)
        elif source == "zero":
            datum_fallback()
            chain.zero(config, missing, arc, logger, notified)
        if source == first and missing_before and not chain.complete() and not rejected:
            # a pair or datum rejection has its own message
            first_left = True
        if chain.complete():
            break
    datum_fallback()
    chain.zero(config, missing, arc, logger, notified)
    for what, _, _ in skipped:
        _notify(f"Station bias of {station} ({system_name(system)}, {code_1}-{code_2}) from {what} not used: "
                f"the satellite biases of the arc come from {other}, another datum (station and satellite biases "
                f"must refer to one datum); the station bias comes from the next source of stec_bias_sources",
                logger, notified, quiet)
    if first_left:
        # a warning only when another source really replaces the chosen one and nothing said so yet
        reported = chain.label == "zero" or (first == "calibrate" and calibration_reported())
        _notify(f"rcv_dcb_source={rcv!r}: {_RCV_NO_VALUE[rcv]} for {station} ({system_name(system)}, "
                f"{code_1}-{code_2}); the station bias comes from the next source of stec_bias_sources "
                f"{tuple(source for source in order[1:])}", logger, notified, quiet or reported)
    return chain


_DATUM_TEXT = {"product": "the bias product", "gim": "the IONEX DCB block", "config": "define_satellite_dcb"}


def _station_order(config, sources: Iterable[BiasSource]) -> list:
    """Station source order: the source chosen by rcv_dcb_source, then stec_bias_sources without it."""
    first = _RCV_FIRST_SOURCE.get(getattr(config, "rcv_dcb_source", None))
    return ([first] if first else []) + [source for source in sources if source != first]


_RCV_NO_VALUE = {
    "calibrate": "no receiver DCB calibrated against the GIM (calibration skipped or without usable data, "
                 "see the warnings/log of the run)",
    "gim": "the GIM gives no receiver DCB",
    "defined": "define_station_dcb gives no value",
}


def _product_pair_bias_m(
    arc: pd.DataFrame,
    code_1: str,
    code_2: str,
    *,
    station: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Product bias in meters and its label per sample: the OSB pair where both OSBs are given,
    else the DSB/DCB of the pair, else a chain of the product's DSBs (``product_dsb_chain``). (None, None) when the product gives none of them.

    A product part holding only zeros counts as absent (placeholder of an analysis centre).
    """
    n = len(arc)
    values = np.full(n, np.nan, dtype=float)
    labels = np.full(n, "", dtype=object)
    for get, label in ((_product_osb_pair_m, "product_osb"), (_product_dsb_pair_m, "product_dsb"),
                       (_product_dsb_chain_m, "product_dsb_chain")):
        if np.isfinite(values).all():
            break
        if get is _product_dsb_chain_m:
            # the chain is searched for the samples still without a value (its links must cover them)
            part = get(arc, code_1, code_2, station=station, need=~np.isfinite(values))
        else:
            part = get(arc, code_1, code_2, station=station)
        if part is None or not _has_product_signal(part):
            continue
        fill = ~np.isfinite(values) & np.isfinite(part)
        values[fill] = part[fill]
        labels[fill] = label
    if not np.isfinite(values).any():
        return None, None
    return values, labels


def _product_osb_pair_m(
    arc: pd.DataFrame,
    code_1: str,
    code_2: str,
    *,
    station: bool,
) -> np.ndarray | None:
    """Difference OSB observable biases as OSB(code_1) - OSB(code_2).

    NaN where either OSB is missing: a satellite without OSB(code_2) has no pair bias, not
    OSB(code_1) - 0.
    """
    prefix = "STA_OSB_" if station else "OSB_"
    osb_1 = _bias_column_m(arc, f"{prefix}{code_1}")
    osb_2 = _bias_column_m(arc, f"{prefix}{code_2}")
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
    """Read direct or reversed differential DSB/DCB product columns (NaN where missing)."""
    prefix = "STA_BIAS_" if station else "BIAS_"
    direct = _bias_column_m(arc, f"{prefix}{code_1}_{code_2}")
    reverse = _bias_column_m(arc, f"{prefix}{code_2}_{code_1}")
    if direct is None:
        return None if reverse is None else -reverse
    if reverse is not None:
        direct = np.where(np.isfinite(direct), direct, -reverse)
    return direct


def _product_dsb_chain_m(
    arc: pd.DataFrame,
    code_1: str,
    code_2: str,
    *,
    station: bool,
    need: np.ndarray | None = None,
) -> np.ndarray | None:
    """B(code_1) - B(code_2) in meters from the shortest chain of the product's DSBs present in the arc,
   as ``gnx_py.biases._product_dsb_m`` does for SPP/PPP: e.g. GPS C1W-C5Q =
    (C1W-C2W) + (C2W-C1C) + (C1C-C5Q) when CAS1 gives no C1W-C5Q DSB. A link without a value on a sample
    leaves that sample NaN (missing, not 0). A station chain uses only the receiver DSBs (``STA_BIAS_``), so
    it needs every link of the station. None without a chain; columns holding only zeros are no link.

    ``need`` (default: every sample) are the samples to give a value: the search uses only links with a value
    on some of them, and samples a shortest chain leaves without a value get the next chain over the links
    that remain (a link with a value on none of them is no link for them)."""
    prefix = "STA_BIAS_" if station else "BIAS_"
    links: dict[str, tuple[str, str, np.ndarray]] = {}
    for col in sorted(c for c in arc.columns if isinstance(c, str) and c.startswith(prefix)):
        parts = col[len(prefix):].split("_")
        if len(parts) != 2:
            continue
        values = _bias_column_m(arc, col)
        if not _has_product_signal(values):
            continue
        links[col] = (parts[0], parts[1], values)
    if code_1 == code_2 or not links:
        return None
    n = len(arc)
    need = np.ones(n, dtype=bool) if need is None else np.asarray(need, dtype=bool).copy()
    total = np.full(n, np.nan, dtype=float)
    found, excluded = False, set()
    while need.any():
        usable = {col: link for col, link in links.items()
                  if col not in excluded and np.isfinite(link[2][need]).any()}
        path = _dsb_path(usable, code_1, code_2)
        if path is None:
            break
        found = True
        part = np.zeros(n, dtype=float)
        for col, sign in path:
            # B_parent - B_node = sign * DSB(col) along the path; NaN propagates (missing link)
            part = part + sign * links[col][2]
        fill = need & np.isfinite(part)
        if not fill.any():
            # no sample still needed has every link of this chain: drop its sparsest link and search again
            excluded.add(min(path, key=lambda step: int(np.isfinite(links[step[0]][2][need]).sum()))[0])
            continue
        total[fill] = part[fill]
        need &= ~fill
    return total if found else None


def _dsb_path(links: dict, code_1: str, code_2: str) -> list[tuple[str, float]] | None:
    """Shortest chain (breadth-first) of DSB links from code_1 to code_2: [(column, sign), ...], None without."""
    graph: dict[str, list] = {}
    for col, (a, b, _) in links.items():
        graph.setdefault(a, []).append((b, col, 1.0))
        graph.setdefault(b, []).append((a, col, -1.0))
    if code_1 not in graph:
        return None
    prev: dict[str, tuple[str, str, float]] = {}
    queue, seen = [code_1], {code_1}
    while queue and code_2 not in seen:
        node = queue.pop(0)
        for nxt, col, sign in graph.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = (node, col, sign)
                queue.append(nxt)
    if code_2 not in seen:
        return None
    path, node = [], code_2
    while node != code_1:
        parent, col, sign = prev[node]
        path.append((col, sign))
        node = parent
    return path


def _bias_column_m(arc: pd.DataFrame, column: str) -> np.ndarray | None:
    """Bias column in ns converted to meters, missing values kept as NaN; None without the column."""
    if column not in arc.columns:
        return None
    values = arc[column].to_numpy(dtype=float) * NS_TO_M
    values[~np.isfinite(values)] = np.nan
    return values


def _has_values(arc: pd.DataFrame, column: str) -> bool:
    """Return whether the bias column of the arc holds at least one finite value."""
    return column in arc.columns and bool(np.isfinite(arc[column].to_numpy(dtype=float)).any())


def _source_summary(labels: np.ndarray, order: Iterable[str] = ()) -> str:
    """One label for an arc: the source of all its samples, or the sources joined with '+' in the
    chain ``order`` ('product' standing for product_osb, product_dsb then product_dsb_chain)."""
    rank = {}
    for source in order:
        for label in (("product_osb", "product_dsb", "product_dsb_chain") if source == "product" else (source,)):
            rank.setdefault(label, len(rank))
    seen = list(dict.fromkeys(str(label) for label in labels))
    seen.sort(key=lambda label: rank.get(label, len(rank)))
    return "+".join(seen) if seen else "disabled"


# Signals treated as one for the GIM DCB pair check, with one warning: the Galileo E5a and E5b
# satellite biases nearly coincide (CAS DSB 2024-035: C5Q-C7Q 0.35 ns std, max 1.2 ns, i.e.
# ~1-3 TECU on E1/E5b), while dropping the E1/E5a GIM DCB leaves the whole C1C-C7Q spread
# (3.8 ns std). GPS L5 (P1-P2 DCB on L1/L5: 8 ns std) and BDS B3I (B1I/B2I DCB on B1I/B3I:
# 5.7 ns std) are not merged. The group name 'E5' stands for "E5a or E5b"; it is not the E5
# AltBOC signal (code C8x, not modelled by GNX).
_GIM_SIGNAL_GROUPS = {"E5a": "E5", "E5b": "E5"}


def gim_dcb_applies(
    code_1: str,
    code_2: str,
    *,
    system: str,
    reference: DCBReference | None = None,
    station: str | None = None,
    logger: logging.Logger | None = None,
    notified: set | None = None,
    quiet: bool = False,
) -> bool:
    """Return whether IONEX DCBs may be used as the bias of the code pair ``code_1 - code_2``.

    The pair must be on the two signals of the reference observables of the
    IONEX DCB block (satellite DCBs, or the receiver DCB of ``station``), e.g.
    a P1-P2 (C1W-C2W) DCB is no bias of C1W-C5Q: C1W-C5Q = (C1W-C2W) + (C2W-C5Q).
    Satellite DCBs: other tracking modes of the same signals are accepted; for
    GPS C/A, P(Y) and L1C/L2C the inter-tracking bias (P1-C1 up to ~2 ns, ~6
    TECU on L1/L2) stays uncorrected, which gives one warning. Galileo E5a and
    E5b are accepted for each other (``_GIM_SIGNAL_GROUPS``), also with one
    warning. A receiver DCB (``station`` given) is used only for exactly its
    signals and, for GPS, its tracking modes;
    otherwise False with a station message (next source).

    Messages are given once per ``notified`` set; ``quiet`` (``stec_missing_bias='zero'``)
    sends them to the log only.
    """
    ref = (reference.codes(system, station) if reference is not None
           else IONEX_DEFAULT_DCB_REFERENCE.get(system))
    what = f"receiver DCB of {station}" if station else "satellite DCBs"
    if ref is None:
        _notify(f"GIM {what} of {system_name(system)}: the IONEX file states no reference observables; "
                f"not used for {code_1}-{code_2} (next source of stec_bias_sources)", logger, notified, quiet)
        return False
    sig = (_code_signal(system, code_1), _code_signal(system, code_2))
    ref_sig = (_code_signal(system, ref[0]), _code_signal(system, ref[1]))
    if station:
        # bug-fix stage 9 (ION-063): a receiver DCB applies to its own signals and tracking modes only. The
        # E5a-E5b bias of a receiver is not the small satellite one (BRUX / NKLG 7.07 / 8.60 ns calibrated,
        # CAS1 6.97 / 8.91 ns: an E1/E5a receiver DCB on E1/E5b is ~18 TECU off), and the GPS C1C-C1W receiver
        # bias is CAS1 1.33-1.48 ns (BRUX, NKLG, NYA2; ~4 TECU on L1/L2)
        if None in sig or sig != ref_sig:
            _notify(f"GIM {what} of {system_name(system)} refer to {ref[0]}-{ref[1]} ({ref_sig[0]}/{ref_sig[1]}); "
                    f"not used for the pair {code_1}-{code_2} ({sig[0]}/{sig[1]}): a receiver DCB applies to its own "
                    f"signals only{' (the receiver E5a-E5b bias is several ns)' if system == 'E' else ''}; the bias "
                    f"comes from the next source of stec_bias_sources", logger, notified, quiet)
            return False
        if system == "G" and (_gps_tracking(code_1), _gps_tracking(code_2)) != (_gps_tracking(ref[0]),
                                                                                _gps_tracking(ref[1])):
            _notify(f"GIM {what} of GPS refer to {ref[0]}-{ref[1]}; not used for the pair {code_1}-{code_2}: a "
                    f"receiver DCB applies to its own GPS code types only (receiver C1C-C1W bias ~1.3-1.5 ns = "
                    f"~4 TECU on L1/L2); the bias comes from the next source of stec_bias_sources",
                    logger, notified, quiet)
            return False
        return True
    group = tuple(_GIM_SIGNAL_GROUPS.get(s, s) for s in sig)
    ref_group = tuple(_GIM_SIGNAL_GROUPS.get(s, s) for s in ref_sig)
    if None in sig or group != ref_group:
        _notify(f"GIM {what} of {system_name(system)} refer to {ref[0]}-{ref[1]} ({ref_sig[0]}/{ref_sig[1]}); "
                f"not used for the pair {code_1}-{code_2} ({sig[0]}/{sig[1]}): the bias comes from the next "
                f"source of stec_bias_sources", logger, notified, quiet)
        return False
    if sig != ref_sig:
        _notify(f"GIM {what} of {system_name(system)} refer to {ref[0]}-{ref[1]} ({ref_sig[0]}/{ref_sig[1]}), "
                f"used for {code_1}-{code_2} ({sig[0]}/{sig[1]}): the E5a-E5b bias is not corrected "
                f"(satellites ~0.35 ns std, up to ~1.2 ns = ~3 TECU on E1/E5b)", logger, notified, quiet)
    if system == "G" and (_gps_tracking(code_1), _gps_tracking(code_2)) != (_gps_tracking(ref[0]),
                                                                            _gps_tracking(ref[1])):
        _notify(f"GIM {what} of GPS refer to {ref[0]}-{ref[1]}, used for {code_1}-{code_2}: the bias between "
                f"these tracking modes (P1-C1 type, up to ~2 ns = ~6 TECU on L1/L2) is not corrected",
                logger, notified, quiet)
    return True


def _code_signal(system: str, code: str) -> str | None:
    """Signal ('L1', 'E5a', 'B2I', ...) of a code observable of ``system``, None if unknown."""
    try:
        return code_signal(system, code)
    except ValueError:
        return None


def _gps_tracking(code: str) -> str:
    """GPS code family with its own satellite bias: C/A, P(Y) (W, P, Y, ...) or L1C/L2C (S, L, X).

    RINEX 2 codes reach the policy without attribute ('C1', 'C2'); they count as P(Y), the
    adapter's preferred source (P1/P2). A C/A fallback is reported by the adapter (R2OBSW01).
    """
    attr = code[2:]
    if code[1] == "5":
        return "L5"
    if attr == "C":
        return "C/A"
    return "L1C/L2C" if attr in ("S", "L", "X") else "P(Y)"


def _notify(message: str, logger: logging.Logger | None, notified: set | None, quiet: bool = False) -> None:
    """Warn once per ``notified`` set (one STEC run), else once per message (warnings registry).

    ``quiet`` logs the message at INFO level without a UserWarning.
    """
    if notified is not None:
        if message in notified:
            return
        notified.add(message)
    log = logger or logging.getLogger(__name__)
    if quiet:
        log.info(message)
        return
    warnings.warn(message, UserWarning, stacklevel=4)
    log.warning(message)


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


def _handle_missing(config, message: str, logger: logging.Logger | None, partial: bool = False) -> None:
    """Apply configured missing-bias behavior without changing numeric fallback."""
    mode = getattr(config, "stec_missing_bias", "warn_zero")
    if mode == "raise":
        raise KeyError(f"{message} (stec_missing_bias='raise').")
    message = f"{message}; using 0{' there' if partial else ''}."
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
    """Find the best available code observable column for one signal.

    The signal's tracking attributes in ``suffix_priority`` order, then the bare RINEX 2 code (e.g. ``C7``),
    then another attribute of the same signal (GPS C2S / C2D on L2, ``gnss.code_signal``). A code with the
    same prefix that belongs to another signal (BDS C7D is B2b, not B2I) is not used: ValueError
    naming the signal and the foreign code. Without any code of the prefix the
    signal's first code name is returned (no product column will match it).
    """
    spec = signal_spec(signal)
    for suffix in spec.suffix_priority:
        col = f"{spec.code_prefix}{suffix}"
        if col in arc.columns:
            return col
    if spec.code_prefix in arc.columns:
        return spec.code_prefix
    same_prefix = sorted(col for col in arc.columns if isinstance(col, str)
                         and len(col) == len(spec.code_prefix) + 1 and col.startswith(spec.code_prefix))
    # another tracking attribute of the same signal (GPS C2S, C2D on L2) is that signal's code
    own = [col for col in same_prefix if _code_signal(spec.system, col) == signal]
    if own:
        return own[0]
    foreign = same_prefix
    if foreign:
        raise ValueError(f"no {signal} code observable ({', '.join(spec.code_prefix + s for s in spec.suffix_priority)}) "
                         f"in the observations; {', '.join(foreign)} belong(s) to another signal and is not used "
                         f"for {signal}")
    return f"{spec.code_prefix}{spec.suffix_priority[0]}"


def _arc_sv(arc: pd.DataFrame) -> str:
    """Return the satellite identifier for diagnostics."""
    if isinstance(arc.index, pd.MultiIndex) and "sv" in arc.index.names:
        return str(arc.index.get_level_values("sv")[0])
    if "sv" in arc.columns:
        return str(arc["sv"].iloc[0])
    return "<unknown>"
