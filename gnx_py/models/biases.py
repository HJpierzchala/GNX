from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.gnss import CLIGHT, code_signal, gamma, ionosphere_free_coefficients


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
    """Return True only when all requested satellite OSB columns carry estimates.

    Some analysis centres list the code OSBs of their clock-reference pair as
    exactly 0.0 for every satellite when they did not estimate them (CODE MGEX
    does this for the BeiDou C2I/C6I pair). Taking those zeros as values drops
    the satellite DSB from per-signal processing, so columns holding only zeros
    count as absent and a DSB product in ``dcb_path`` can supply the bias.
    """
    cols = [f"OSB_{obs_col}" for obs_col in obs_cols]
    if not all(col in epoch.columns for col in cols):
        return False
    if not cols:
        return True
    values = epoch[cols].to_numpy(dtype=float)
    return bool(np.any(np.isfinite(values) & (values != 0.0)))


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
        placeholder_osb = (
            f"OSB_{code_1}" in epoch.columns
            and f"OSB_{code_2}" in epoch.columns
            and not has_satellite_osb(epoch, (code_1, code_2))
        )
        if placeholder_osb:
            warnings.warn(
                f"satellite OSB for {code_1}/{code_2} are all zero (not estimated by the bias "
                f"product) and no DSB product provides {code_1}-{code_2}: the satellite DSB is "
                f"not corrected in per-signal code observations; add a DSB/DCB file (e.g. CAS) "
                f"to dcb_path",
                UserWarning,
                stacklevel=2,
            )
        return zeros.copy(), zeros.copy(), zeros.copy()

    return coeff_2 * total, coeff_1 * total, total


# --------------------------------------------------------------------------- satellite code group delays
#
# One place for the satellite code-bias (group-delay) convention of every supported signal, used by SPP
# and meant for PPP. ``b(code)`` below is the satellite code bias of a code observable *relative to the
# satellite clock datum* in metres: the corrected code is ``P - b`` and a code combination
# ``sum(w_i * P_i)`` gets ``-sum(w_i * b_i)``. With broadcast clocks ``P + c*dt_sv - b`` is the code of
# the ICD single-frequency user equation (e.g. IS-GPS-200 ``dt_L1P = dt_sv - TGD``: b = c*TGD).

#: Precise (IGS/MGEX) satellite clocks refer to the ionosphere-free combination of these codes; product
#: OSBs are relative to that datum.
PRECISE_CLOCK_REFERENCE = {"G": ("C1W", "C2W"), "E": ("C1C", "C5Q"), "C": ("C2I", "C6I")}

_BDS_CNAV = ("CNV1", "CNV2", "CNV3")
# codes whose broadcast group delay the message defines exactly (used as anchors when a product
# DSB/OSB supplies the inter-signal part the message lacks, e.g. GPS L5)
_BROADCAST_ANCHOR_CODES = {"G": ("C1W", "C2W"), "E": (), "C": ("C6I", "C2I", "C7I")}
# precise clocks: GPS signal -> P(Y) code whose product bias a code without one takes (ICD assignment
# of the group delay when the inter-signal correction is not known)
_PRECISE_ISC_ANCHOR = {"G": {"L1": "C1W", "L2": "C2W", "L5": "C1W"}}

# typical sizes of the corrections a run may lack (2024-035: BRDC00IGS, BRD400DLR B-CNAV, CAS DSB)
_TYPICAL = {
    "G": "GPS TGD -18..+7 ns (-5.3..+2.1 m), L1P(Y)-L5 code bias -4..+8 ns (-1.2..+2.3 m) per satellite, "
         "amplified 1.26x in IF(L1,L5) and 11.3x in IF(L2,L5)",
    "E": "Galileo BGD -15..+8 ns (-4.6..+2.3 m) per satellite",
    "C": "BeiDou group delays -45..+27 ns (-13..+8 m) per satellite",
}
# ... of single terms (first matching key of the missing-term text)
_TYPICAL_TERM = {
    "G": (("ISC_L5", "L1P(Y)-L5 code bias -4..+8 ns (-1.2..+2.3 m) per satellite, amplified 1.26x in "
                     "IF(L1,L5) and 11.3x in IF(L2,L5)"),
          ("ISC_L2C", "L2C-L2P(Y) code bias below 1 ns (0.3 m) per satellite"),
          ("TGD", "GPS TGD -18..+7 ns (-5.3..+2.1 m) per satellite")),
    "C": (("approximation of it", "BeiDou TGD_B1Cp - TGD1 -0.7..+7.7 ns (-0.2..+2.3 m) per satellite, 3.8 ns rms"),
          ("TGD_B1Cp", "BeiDou TGD_B1Cp -42..+27 ns (-12.7..+8.0 m) per satellite"),
          ("TGD_B2ap", "BeiDou TGD_B2ap -19..+27 ns (-5.6..+8.0 m) per satellite"),
          ("TGD_B2bI", "BeiDou TGD_B2bI -18..+27 ns (-5.4..+8.2 m) per satellite"),
          ("ISC_B", "BeiDou ISC of the B1C/B2a data components 1..3 ns (0.3..0.9 m) per satellite")),
}


@dataclass(frozen=True)
class CodeBiasCorrection:
    """Satellite group-delay correction of a code observable or code combination.

    ``correction_m``: metres added to ``sum(w_i * P_i)`` to refer it to the satellite clock datum (with
    broadcast clocks ``P + c*dt_sv + correction`` is the ICD-corrected code); NaN where no correction
    can be formed for that observation.
    ``source``: per observation, how the correction was obtained: ``"broadcast"`` (navigation-message
    TGD/BGD/ISC), ``"osb"``, ``"dsb"`` (product biases), ``"broadcast+osb"``/``"broadcast+dsb"`` (message
    group delay of an anchor code plus a product inter-signal bias), ``"none"`` (NaN).
    ``missing``: corrections that were not available: either the reason for NaN values (also per
    satellite, e.g. a satellite without a navigation record or absent from the bias product), or an
    ICD inter-signal term taken as zero in ``correction_m`` (e.g. GPS ISC_L5 without CNAV,
    IS-GPS-705 20.3.3.3.1.2), or a per-satellite replacement (a satellite absent from the bias
    product that takes the next source). The caller decides how to report them.
    ``biases_m``: per code, the satellite code bias ``b`` [m] that ``correction_m`` is made of
    (``correction_m = -sum(w_i * b_i)``), e.g. for filters that correct each code separately.
    ``per_satellite``: ``(template, satellite)`` for the entries of ``missing`` that name one
    satellite (``template.format(satellites=satellite)``), so that a caller can report them once
    for all satellites (:func:`group_satellite_messages`).
    """

    correction_m: np.ndarray
    source: np.ndarray
    missing: tuple[str, ...] = ()
    biases_m: tuple[np.ndarray, ...] = ()
    per_satellite: tuple[tuple[str, str], ...] = ()


def _scaled_column(epoch: pd.DataFrame, column: str, scale: float) -> Optional[np.ndarray]:
    if column not in epoch.columns:
        return None
    return epoch[column].to_numpy(dtype=float) * scale


class _ProductColumns:
    """Satellite bias columns (``OSB_*``, ``BIAS_*``) a product gives for the rows of one epoch frame.

    A column with no finite value in the frame is not given by the product (the bias reader leaves
    observables/pairs a product does not give as NaN, ``GNSSDataProcessor2.read_bia``); a column holding
    only zeros is not a product value either (an analysis centre's placeholder for biases it did not
    estimate, see has_satellite_osb). Scanned once per frame (one array), the DSB graph is built on
    first use.
    """

    __slots__ = ("estimated", "_graph")

    def __init__(self, epoch: pd.DataFrame):
        positions = [i for i, c in enumerate(epoch.columns) if isinstance(c, str) and c.startswith(("OSB_", "BIAS_"))]
        cols = [epoch.columns[i] for i in positions]
        estimated: frozenset = frozenset()
        if cols:
            values = epoch.iloc[:, positions].to_numpy(dtype=float)
            flags = (np.isfinite(values) & (values != 0.0)).any(axis=0)
            estimated = frozenset(c for c, flag in zip(cols, flags) if flag)
        self.estimated = estimated
        self._graph: Optional[dict[str, list[tuple[str, str, float]]]] = None

    @property
    def graph(self) -> dict[str, list[tuple[str, str, float]]]:
        """Code -> [(neighbour, column, sign)]: DSB(a, b) = B_a - B_b from the BIAS_<a>_<b> columns."""
        if self._graph is None:
            graph: dict[str, list[tuple[str, str, float]]] = {}
            for col in sorted(c for c in self.estimated if c.startswith("BIAS_")):
                parts = col[5:].split("_")
                if len(parts) != 2:
                    continue
                a, b = parts
                graph.setdefault(a, []).append((b, col, 1.0))
                graph.setdefault(b, []).append((a, col, -1.0))
            self._graph = graph
        return self._graph


def _product_osb_m(epoch: pd.DataFrame, code: str, include_station: bool, product: _ProductColumns
                   ) -> Optional[np.ndarray]:
    if f"OSB_{code}" not in product.estimated:
        return None
    sat = _scaled_column(epoch, f"OSB_{code}", NS_TO_M)
    if include_station:
        sta = _scaled_column(epoch, f"STA_OSB_{code}", NS_TO_M)
        if sta is not None:
            sat = sat + np.nan_to_num(sta)
    return sat


def _product_dsb_m(epoch: pd.DataFrame, code_a: str, code_b: str, product: _ProductColumns,
                   include_station: bool, station_only: bool = False) -> Optional[np.ndarray]:
    """B_a - B_b in metres from a chain of satellite DSBs (shortest path), None without a chain
    (``station_only``: the receiver part of that chain alone). With ``include_station`` (or ``station_only``)
    the receiver part adds the station DSB (``STA_<column>``) of every link of the chain; a link without a
    station value counts 0. The receiver part is a receiver DCB: the receiver clock and inter-system bias absorb
    it only on ionosphere-free and single-code routes."""
    n = len(epoch)
    if code_a == code_b:
        return np.zeros(n)
    graph = product.graph
    prev: dict[str, tuple[str, str, float]] = {}
    queue, seen = [code_a], {code_a}
    while queue and code_b not in seen:
        node = queue.pop(0)
        for nxt, col, sign in graph.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = (node, col, sign)
                queue.append(nxt)
    if code_b not in seen:
        return None
    total = np.zeros(n)
    node = code_b
    while node != code_a:
        parent, col, sign = prev[node]
        # walking parent -> node: B_parent - B_node = sign * DSB(col)
        value = np.zeros(n) if station_only else epoch[col].to_numpy(dtype=float) * NS_TO_M
        if (include_station or station_only) and f"STA_{col}" in epoch.columns:
            value = value + np.nan_to_num(epoch[f"STA_{col}"].to_numpy(dtype=float) * NS_TO_M)
        total += sign * value
        node = parent
    return total


def _product_difference_m(epoch, code_a, code_b, product, include_station):
    """B_a - B_b from the product: OSB difference when both OSBs exist, else a DSB chain."""
    osb_a = _product_osb_m(epoch, code_a, include_station, product)
    osb_b = _product_osb_m(epoch, code_b, include_station, product) if code_a != code_b else osb_a
    if osb_a is not None and osb_b is not None:
        return osb_a - osb_b, "osb"
    dsb = _product_dsb_m(epoch, code_a, code_b, product, include_station)
    if dsb is not None:
        return dsb, "dsb"
    return None, None


def _galileo_inav_rows(epoch: pd.DataFrame, nav_message: Optional[str]) -> np.ndarray:
    """True where the row's Galileo clock is the I/NAV (E1,E5b) one, False for F/NAV (E1,E5a)."""
    n = len(epoch)
    if nav_message is None and "nav_message" in epoch.columns:
        values = epoch["nav_message"].astype(str).to_numpy()
        if np.isin(values, ("I/NAV", "F/NAV")).all():
            return values == "I/NAV"
    if nav_message is None and "DataSrc" in epoch.columns:
        # RINEX 3.04 Galileo data sources: bit 9 (512) clock E5b,E1; bit 8 (256) clock E5a,E1
        src = np.nan_to_num(epoch["DataSrc"].to_numpy(dtype=float)).astype(np.int64)
        if np.all((src & 768) > 0):
            return (src & 512) > 0
    if nav_message in ("I/NAV", "F/NAV"):
        return np.full(n, nav_message == "I/NAV")
    raise ValueError(
        "Galileo broadcast group delays need the navigation message of the clock: pass "
        "nav_message='I/NAV' or 'F/NAV' (or give the DataSrc column)"
    )


# GPS code attributes of the P(Y) code (L1/L2). A RINEX 2 code has no attribute ("C1", "C2"): the
# RINEX 2 adapter fills it from P1/P2 first (C/A, L2C only when the file has no P-code), so it is
# taken as P(Y) like C1W/C2W
_GPS_PY_ATTRS = {"W", "P", "Y", "D", ""}


# BeiDou PRNs of BDS-3 satellites (C01-C18: BDS-2 and test satellites)
_BDS3_FIRST_PRN = 19


def _row_satellites(epoch: pd.DataFrame) -> Optional[np.ndarray]:
    """Satellite id (e.g. 'C20') of every row, from the ``sv`` index level or column."""
    if "sv" in (epoch.index.names or ()):
        return epoch.index.get_level_values("sv").astype(str).str[:3].to_numpy()
    if "sv" in epoch.columns:
        return epoch["sv"].astype(str).str[:3].to_numpy()
    return None


def _row_prn(epoch: pd.DataFrame) -> np.ndarray:
    """PRN number of every row (-1 where unknown)."""
    sv = _row_satellites(epoch)
    if sv is None:
        return np.full(len(epoch), -1)
    return pd.to_numeric(pd.Series(sv).str[1:3], errors="coerce").fillna(-1).astype(int).to_numpy()


# --------------------------------------------------------------------------- BeiDou precise clock datum
#
# The precise BeiDou clocks of an analysis centre refer to the ionosphere-free combination of one signal pair, and
# that pair is a choice of the centre: IF(B1I,B3I) for every MGEX product of 2024 (COD/GRG/GFZ 2024-035, declared
# C2I C6I / L2I L6I in the Bias-SINEX header, and confirmed by the broadcast-clock test below), IF(B1C,B2a) for the
# BDS-3 clocks of COD MGEX on 2026-213 (GRG 2026 declares L1P L5P). GNX assumed IF(C2I,C6I) for every BeiDou
# precise clock (PRECISE_CLOCK_REFERENCE): with the wrong datum every BDS-3 code correction is off by
# D1 - D2 = -k1 c TGD1 + c (k1' TGD_B1Cp - k2' TGD_B2ap), 3.7 m rms per satellite in 2024 and 9.0 m in 2026, which
# took SPP C IF(B1I,B3I) on BRUX 2026-213 from 2.36 m to 11.43 m rms, silently (bug-fix stage 13b, SPP-080).
# The datum is resolved once per session (:class:`PreciseClockDatum`, session_guards.resolve_precise_clock_datum)
# per satellite generation: BDS-2 satellites transmit no B1C/B2a, so their clocks always refer to IF(B1I,B3I);
# the BDS-3 GEO satellites transmit B1I, B3I and B2b but no open-service B1C/B2a (no B-CNAV TGD_B1Cp/TGD_B2ap), so
# they keep IF(B1I,B3I) as well (not verified: no 2026 product with GEO clocks was measured).
# PRECISE_CLOCK_REFERENCE itself is never changed at run time.

#: BeiDou precise clock datum pairs by name (``bds_precise_clock_datum`` setting): the codes the IF datum is formed
#: of. The tracking attribute is not part of the datum (attributes of one band differ by ~0.1 ns).
BDS_DATUM_PAIRS = {"B1IB3I": ("C2I", "C6I"), "B1CB2a": ("C1P", "C5P")}

#: Satellite generations of BeiDou, as far as the clock datum is concerned: BDS-2, BDS-3 MEO/IGSO, BDS-3 GEO (ANTEX
#: block BEIDOU-3G*: C59-C62 until 2025, C01-C04 in igs20.atx at 2026-08-01).
BDS2, BDS3, BDS3_GEO = "BDS-2", "BDS-3", "BDS-3 GEO"
#: PRNs of the BDS-3 GEO satellites for the PRN rule (no ANTEX entry, no B-CNAV evidence)
_BDS3_GEO_PRNS = frozenset(range(59, 64))

#: Broadcast-clock datum test (BRUX/COD/GRG/GFZ 2024-035 and COD 2026-213): the between-satellite scatter
#: [m] of the BDS-3 precise-minus-broadcast clocks a hypothesis must reach (hourly winners 0.43-0.81 m), the factor
#: by which the other hypothesis must be larger (hourly 4.2-12.5) and, with TGD1 only (RINEX 3 navigation), the
#: scatter from which IF(B1I,B3I) is rejected (2026: 8.98-9.14 m hourly); fewer BDS-3 satellites than
#: DATUM_TEST_MIN_SATS leave it undetermined.
DATUM_TEST_ACCEPT_M = 1.5
DATUM_TEST_RATIO = 2.5
DATUM_TEST_REJECT_M = 4.5
DATUM_TEST_MIN_SATS = 6
#: BeiDou satellites left out of the datum test: GEO (broadcast orbit and clock of the lowest quality) and the
#: in-orbit test satellites
DATUM_TEST_EXCLUDED = frozenset(f"C{n:02d}" for n in (1, 2, 3, 4, 5, 17, 18, 31, 56, 57, 58, 59, 60, 61, 62, 63))

#: BDT = GPST - 14 s (BDS-SIS-ICD-B1I 5.2.4.5); navigation records carry toc in BDT
_BDT_MINUS_GPST_S = -14.0


def bds_pair_codes(pair) -> tuple[str, str]:
    """Codes of a BeiDou datum pair given by name (``'B1IB3I'``), by codes or by phase observables of a
    Bias-SINEX header (``('L1P', 'L5P')`` -> ``('C1P', 'C5P')``); ValueError for anything else."""
    if isinstance(pair, str):
        if pair in BDS_DATUM_PAIRS:
            return BDS_DATUM_PAIRS[pair]
        raise ValueError(f"unknown BeiDou clock datum {pair!r} (one of {', '.join(BDS_DATUM_PAIRS)})")
    obs = [str(o).strip().upper() for o in pair]
    if len(obs) != 2 or any(len(o) != 3 or o[0] not in "CL" for o in obs):
        raise ValueError(f"a BeiDou clock datum is two code or phase observables, got {pair!r}")
    codes = tuple("C" + o[1:] for o in obs)
    for code in codes:
        code_signal("C", code)  # raises for a band GNX does not know (e.g. B2a+b, band 8)
    return codes


def bds_pair_name(codes: Sequence[str]) -> str:
    """``IF(B1C,B2a)`` for the codes ``('C1P', 'C5P')``."""
    return f"IF({','.join(code_signal('C', c) for c in codes)})"


def _same_signals(a: Sequence[str], b: Sequence[str]) -> bool:
    return sorted(code_signal("C", c) for c in a) == sorted(code_signal("C", c) for c in b)


@dataclass(frozen=True)
class PreciseClockDatum:
    """IF datum of the precise BeiDou clocks of one session, per satellite generation.

    ``refs``: generation (:data:`BDS2`, :data:`BDS3`) -> the two codes of the datum, e.g.
    ``{'BDS-3': ('C1P', 'C5P'), 'BDS-3 GEO': ('C2I', 'C6I'), 'BDS-2': ('C2I', 'C6I')}`` (a generation not in
    ``refs`` takes PRECISE_CLOCK_REFERENCE); ``generations``: satellite (``'C20'``) -> generation
    (``generation_sources`` says how it was decided: 'antex', 'b-cnav' or 'prn'); a satellite not listed takes
    the PRN rule (C59-C63 BDS-3 GEO, other PRNs from C19 BDS-3). ``source``: 'setting'
    (``bds_precise_clock_datum``), 'declared' (the SATELLITE_CLOCK_REFERENCE_OBSERVABLES of a bias product of
    the clocks' analysis centre), 'detected' (the broadcast-clock test, :func:`detect_bds_clock_datum`) or
    'assumed' (IF(B1I,B3I), undetermined).
    ``detail``: the test statistics and the evidence (for the warning texts and the result attributes).
    :func:`code_group_delay_correction` and :func:`satellite_code_bias_m` take it as ``clock_datum`` and use, per
    row, the datum of the row's satellite instead of :data:`PRECISE_CLOCK_REFERENCE`. ``osb_as_given``: every
    OSB product of dcb_path declares this BDS-3 datum (SATELLITE_CLOCK_REFERENCE_OBSERVABLES, e.g. GRG 2026 L1P
    L5P), so its OSBs are used as given (a phase-defined datum leaves k1' OSB(C1P) - k2' OSB(C5P) != 0 on
    purpose); otherwise the OSBs are referred to the datum (:func:`satellite_code_bias_m`).
    ``unmoved``: BDS-3 satellites whose codes cannot be moved to this datum (no B-CNAV TGD_B1Cp/TGD_B2ap, no
    product bias in it; their group delay is taken as 0, ``session_guards.resolve_precise_clock_datum`` warns);
    ``unmoved_sigma_m`` / ``unmoved_code_sigma_m``: the size of that unmodelled term on the mode's observable (the
    IF combination of a dual-frequency mode, the code of a single-frequency one) and on a single code (None: as
    ``unmoved_sigma_m``), added in quadrature to the standard deviation of their code observations in SPP and the
    PPP IF / GRAPHIC filters, respectively in the uncombined PPP filters (:func:`unmoved_code_variance_m2`).
    With broadcast clocks (``source='broadcast'``) only ``generations`` is read: which band-7 code of a satellite
    is B2b-I (BDS-3) and which B2I (BDS-2).
    """

    refs: dict
    generations: dict = field(default_factory=dict)
    source: str = "assumed"
    detail: dict = field(default_factory=dict)
    generation_sources: dict = field(default_factory=dict)
    system: str = "C"
    osb_as_given: bool = False
    unmoved: frozenset = frozenset()
    unmoved_sigma_m: float = 0.0
    unmoved_code_sigma_m: Optional[float] = None

    def generation_of(self, sv: str) -> str:
        sv = str(sv)[:3]
        gen = self.generations.get(sv)
        if gen is not None:
            return gen
        return _prn_generation(sv)

    def refs_of(self, sv: str) -> tuple[str, ...]:
        return tuple(self.refs.get(self.generation_of(sv), PRECISE_CLOCK_REFERENCE[self.system]))

    @property
    def is_default(self) -> bool:
        """True when every generation uses :data:`PRECISE_CLOCK_REFERENCE` (the datum GNX assumed before)."""
        default = PRECISE_CLOCK_REFERENCE[self.system]
        return all(tuple(r) == default for r in self.refs.values())

    def as_dict(self) -> dict:
        """Plain description for result attributes: source, datum per generation, statistics."""
        return {"source": self.source,
                "datum": {gen: bds_pair_name(codes) for gen, codes in sorted(self.refs.items())},
                "codes": {gen: list(codes) for gen, codes in sorted(self.refs.items())},
                "osb_as_given": self.osb_as_given,
                "unmoved": sorted(self.unmoved), "unmoved_sigma_m": round(float(self.unmoved_sigma_m), 3),
                "unmoved_code_sigma_m": round(float(self._code_sigma_m()), 3),
                **{k: v for k, v in self.detail.items() if not isinstance(v, (pd.DataFrame, np.ndarray))}}

    def _code_sigma_m(self) -> float:
        return float(self.unmoved_sigma_m if self.unmoved_code_sigma_m is None else self.unmoved_code_sigma_m)

    def unmoved_variance_m2(self, satellites, *, uncombined: bool = False) -> np.ndarray:
        """Extra code variance [m^2] of each satellite of ``satellites``: ``unmoved_sigma_m**2`` (the mode's
        observable) or, with ``uncombined``, ``unmoved_code_sigma_m**2`` (a single code) for the satellites of
        ``unmoved``, 0 for the others."""
        sv = np.asarray([str(s)[:3] for s in satellites], dtype=object)
        if not self.unmoved or sv.size == 0:
            return np.zeros(sv.size)
        sigma = self._code_sigma_m() if uncombined else float(self.unmoved_sigma_m)
        return np.where(np.isin(sv, sorted(self.unmoved)), sigma ** 2, 0.0)


#: lower bound [m] of the standard deviation added to the code observations of BDS-3 satellites whose codes cannot
#: be moved to the precise clock datum: the smallest datum offset per satellite measured (IF(B1I,B3I) against
#: IF(B1C,B2a) 3-10 m per satellite, BRUX 2026-213)
UNMOVED_CODE_SIGMA_FLOOR_M = 3.0


def unmoved_code_variance_m2(owner, system: str, frame_or_satellites, *, uncombined: bool = False) -> np.ndarray:
    """Extra code variance [m^2] per row of ``frame_or_satellites`` (an epoch frame with an ``sv`` index level or
    column, or a sequence of satellite ids) for the BDS-3 satellites whose codes cannot be moved to the precise
    clock datum of the run (``PreciseClockDatum.unmoved`` of ``_notes.run_memo(owner, 'precise_clock_datum')``,
    resolved by ``session_guards.resolve_precise_clock_datum``); zeros otherwise. SPP and every PPP filter add it to
    the variance of those code observations: their group delay is taken
    as 0 and the down-weighting keeps the unmodelled 3-10 m out of the position. The size is that of the observable
    the filter uses: the mode's combination (SPP, PPP IF / GRAPHIC) or, with ``uncombined``, a single code (the
    uncombined PPP filters; only the IF part of the code biases reaches an IF observable)."""
    from ..core import _notes

    frame = isinstance(frame_or_satellites, pd.DataFrame)
    if not frame:
        frame_or_satellites = list(frame_or_satellites)
    n = len(frame_or_satellites)
    datum = _notes.run_memo(owner, "precise_clock_datum").get("C") if system == "C" else None
    if datum is None or not getattr(datum, "unmoved", None) or n == 0:
        return np.zeros(n)
    sv = _row_satellites(frame_or_satellites) if frame else np.asarray(frame_or_satellites, dtype=object)
    if sv is None:
        return np.zeros(n)
    return datum.unmoved_variance_m2(sv, uncombined=uncombined)


def bias_product_clock_references(path) -> dict:
    """SATELLITE_CLOCK_REFERENCE_OBSERVABLES of the +BIAS/DESCRIPTION block of a Bias-SINEX file, per system
    (``{'C': ['L1P', 'L5P'], 'G': ['C1W', 'C2W']}``; empty when the header has none). Reads the header only (any
    compression of :mod:`gnx_py.compression`); the same parsing as ``GNSSDataProcessor2.read_bia``."""
    from ..io.compression import head_lines

    out: dict = {}
    for line in head_lines(path, "bias product", 1 << 18):
        if line.startswith("+BIAS/SOLUTION"):
            break
        tokens = line.split()
        if not line.startswith("*") and "SATELLITE_CLOCK_REFERENCE_OBSERVABLES" in tokens:
            given = tokens[tokens.index("SATELLITE_CLOCK_REFERENCE_OBSERVABLES") + 1:]
            if len(given) > 1:
                out[given[0]] = given[1:]
    return out


def bias_product_mode(path) -> Optional[str]:
    """Bias mode of a Bias-SINEX product: ``'ABSOLUTE'`` (OSB) or ``'RELATIVE'`` (DSB), from ``BIAS_MODE`` of the
    +BIAS/DESCRIPTION block, else from the bias-mode letter (``A`` / ``R``) of the ``%=BIA`` header line (Bias-SINEX
    1.00 section 3.1); None when the header gives neither (e.g. a CODE .DCB file). Reads the header only, as
    :func:`bias_product_clock_references` (before v1.0.0 OSB products were told by "OSB" in the file name)."""
    from ..io.compression import head_lines

    letter = None
    for line in head_lines(path, "bias product", 1 << 18):
        if line.startswith("+BIAS/SOLUTION"):
            break
        tokens = line.split()
        if line.startswith("%=BIA") and len(tokens) >= 8 and letter is None:
            # %=BIA version agency creation-time data-agency start end bias-mode number
            letter = {"A": "ABSOLUTE", "R": "RELATIVE"}.get(tokens[7].upper())
        elif not line.startswith("*") and "BIAS_MODE" in tokens:
            given = tokens[tokens.index("BIAS_MODE") + 1:]
            if given and given[0].upper() in ("ABSOLUTE", "RELATIVE"):
                return given[0].upper()
    return letter


def _datum_groups(epoch: pd.DataFrame, system: str,
                  clock_datum: Optional[PreciseClockDatum]) -> list[tuple[tuple, Optional[np.ndarray]]]:
    """[(datum codes, row mask)] of the rows of ``epoch`` (mask None: every row). Without a datum of the
    system: PRECISE_CLOCK_REFERENCE for all rows."""
    default = PRECISE_CLOCK_REFERENCE[system]
    if clock_datum is None or clock_datum.system != system or len(epoch) == 0:
        return [(default, None)]
    sv = _row_satellites(epoch)
    if sv is None:
        return [(default, None)]
    by_sv = {s: clock_datum.refs_of(s) for s in pd.unique(sv)}
    groups = list(dict.fromkeys(by_sv.values()))
    if len(groups) == 1:
        return [(groups[0], None)]
    keys = np.array([groups.index(by_sv[s]) for s in sv])
    return [(refs, keys == i) for i, refs in enumerate(groups)]


def bds_bcnav_satellites(nav: Optional[pd.DataFrame]) -> set:
    """BeiDou satellites with B-CNAV1/B-CNAV2 records (RINEX 4 CNV1/CNV2: the B1C and B2a messages) in the
    navigation frame ``nav`` (index level ``sv``, column ``nav_message``): only BDS-3 MEO/IGSO satellites transmit
    them. CNV3 (B2b) is left out: the BDS-3 GEO satellites broadcast PPP-B2b."""
    if nav is None or len(nav) == 0 or "nav_message" not in nav.columns or "sv" not in (nav.index.names or ()):
        return set()
    rows = nav["nav_message"].astype(str).isin(("CNV1", "CNV2")).to_numpy()
    return set(pd.Index(nav.index.get_level_values("sv")[rows]).astype(str).str[:3])


def _prn_generation(sv: str) -> str:
    """Generation of a BeiDou satellite from its PRN alone: C59-C63 BDS-3 GEO, C19 and above BDS-3, else BDS-2
    (wrong for C01-C04, C06-C08 and C11-C14 after 2023, see :func:`bds_satellite_generations`)."""
    prn = int(sv[1:3]) if sv[1:3].isdigit() else -1
    if prn in _BDS3_GEO_PRNS:
        return BDS3_GEO
    return BDS3 if prn >= _BDS3_FIRST_PRN else BDS2


def bds_satellite_generations(satellites: Iterable[str], *, antex=None, epoch=None,
                              cnav_satellites: Iterable[str] = ()) -> tuple[dict, dict]:
    """``({satellite: generation}, {satellite: how})`` of BeiDou satellites: the block of the ANTEX entry valid at
    ``epoch`` (``BEIDOU-3*`` / ``BEIDOU-2*``; ``antex``: ``gnx_py.antex.Antex`` or ``AntennaModel``), else B-CNAV
    records of the satellite in the navigation data (only BDS-3 MEO/IGSO transmits B1C/B2a: ``cnav_satellites``,
    see :func:`bds_bcnav_satellites`), else the PRN (C59-C63 BDS-3 GEO, C19 and above BDS-3). The PRN alone is
    wrong after 2023: C06-C08 and C11-C14 belong to BDS-3 satellites in 2026 (igs20.atx: C220/C221/C224
    BEIDOU-3I, C232-C235 BEIDOU-3M; COD0MGXFIN 2026-213 lists only BDS-3) and C01-C04 to BDS-3 GEO satellites
    (BEIDOU-3G-CAST). A BEIDOU-3G* block is :data:`BDS3_GEO`: no open-service B1C/B2a, the datum of its clocks
    stays IF(B1I,B3I)."""
    cnav = {str(s)[:3] for s in cnav_satellites}
    gens: dict = {}
    how: dict = {}
    for sv in sorted({str(s)[:3] for s in satellites if str(s)[:1] == "C"}):
        block = ""
        if antex is not None:
            try:
                entry = antex.satellite(sv, epoch)  # gnx_py.antex.Antex
            except TypeError:
                entry = antex.satellite(sv)  # gnx_py.antex.AntennaModel: the entries of its first epoch
            block = str(getattr(entry, "type", "") or "").strip().upper()
        if block.startswith("BEIDOU-3G"):
            gens[sv], how[sv] = BDS3_GEO, "antex"
        elif block.startswith("BEIDOU-3"):
            gens[sv], how[sv] = BDS3, "antex"
        elif block.startswith("BEIDOU-2"):
            gens[sv], how[sv] = BDS2, "antex"
        elif sv in cnav:
            gens[sv], how[sv] = BDS3, "b-cnav"
        else:
            gens[sv], how[sv] = _prn_generation(sv), "prn"
    return gens, how


def detect_bds_clock_datum(sp3: pd.DataFrame, nav: Optional[pd.DataFrame], generations: dict, *,
                           start=None, end=None) -> dict:
    """Broadcast-clock test of the precise BeiDou clock datum (clock only, no orbit, no antenna model).

    ``sp3``: precise records indexed (time, sv) [GPST] with ``clk`` [s]; ``nav``: BeiDou navigation records
    indexed (sv, time = toc [BDT]) with the clock polynomial, ``TGD1`` (D1/D2) and, from RINEX 4 B-CNAV records,
    ``TGD_B1Cp``/``TGD_B2ap`` [s]; ``generations``: satellite -> generation. For every BDS-3 satellite (GEO and
    test satellites left out) and every 5-min SP3 epoch in ``[start, end]`` (at most 24 h) with a D1/D2 record
    within 1 h, ``y = c (dt_sp3 - dt_brd)``; the broadcast clock refers to B3I (BDS-SIS-ICD-B1I 5.2.4.10), so a
    precise clock of the datum IF(B1I,B3I) gives ``y = -k1 c TGD1 + common(t)`` (H1, k1 = 2.944) and one of
    IF(B1C,B2a) ``y = -c (k1' TGD_B1Cp - k2' TGD_B2ap) + common(t)`` (H2, k1' = 2.261, k2' = 1.261, ICD B1C/B2a
    7.6), TGD per satellite the median of the day. Residuals minus their epoch median over the BDS-3 satellites,
    median per satellite, standard deviation over the satellites [m]: the smaller one is the datum. Two-sided
    (TGD_B1Cp/TGD_B2ap of at least DATUM_TEST_MIN_SATS satellites): H accepted when its scatter is at most
    DATUM_TEST_ACCEPT_M and the other one DATUM_TEST_RATIO times larger. One-sided (TGD1 only): IF(B1I,B3I)
    accepted below DATUM_TEST_ACCEPT_M, rejected above DATUM_TEST_REJECT_M.

    Returns ``{'datum': 'B1IB3I' | 'B1CB2a' | None, 'two_sided': bool, 'std_h1_m', 'std_h2_m', 'n_bds3',
    'n_bds3_h2', 'reason'}`` (None: undetermined, ``reason`` says why).
    """
    out = {"datum": None, "two_sided": False, "std_h1_m": None, "std_h2_m": None, "n_bds3": 0, "n_bds3_h2": 0,
           "reason": ""}
    if sp3 is None or len(sp3) == 0 or nav is None or len(nav) == 0 or "TGD1" not in nav.columns:
        out["reason"] = "no BeiDou precise or broadcast clocks with TGD1"
        return out
    rec = sp3.reset_index()
    rec["sv"] = rec["sv"].astype(str).str[:3]
    rec["time"] = pd.to_datetime(rec["time"])
    rec = rec[rec["sv"].map(lambda s: generations.get(s) == BDS3) & ~rec["sv"].isin(DATUM_TEST_EXCLUDED)]
    clk = pd.to_numeric(rec["clk"], errors="coerce").to_numpy(dtype=float)
    rec = rec[np.isfinite(clk) & (np.abs(clk) < 0.9)]
    t = rec["time"]
    rec = rec[(t.dt.second == 0) & (t.dt.minute % 5 == 0) & (t.dt.microsecond == 0)]
    if start is not None:
        rec = rec[rec["time"] >= pd.Timestamp(start)]
    if len(rec):
        t0 = rec["time"].min()
        stop = t0 + pd.Timedelta(hours=24) if end is None else min(pd.Timestamp(end), t0 + pd.Timedelta(hours=24))
        rec = rec[rec["time"] <= stop]
    rec = rec.drop_duplicates(["time", "sv"])
    navf = nav.reset_index()
    navf["sv"] = navf["sv"].astype(str).str[:3]
    navf["time"] = pd.to_datetime(navf["time"])
    message = navf["nav_message"].astype(str) if "nav_message" in navf.columns else pd.Series("D1", index=navf.index)
    d12 = navf[message.isin(["D1", "D2"]) | ~message.isin(_BDS_CNAV)]
    if "SVclockBias" in d12:
        d12 = d12[np.isfinite(pd.to_numeric(d12["SVclockBias"], errors="coerce"))]
    else:
        d12 = d12.iloc[0:0]
    tgd1 = pd.to_numeric(d12["TGD1"], errors="coerce").groupby(d12["sv"]).median()
    tgd_ca = None
    if {"TGD_B1Cp", "TGD_B2ap"} <= set(navf.columns):
        cn = navf[message.isin(["CNV1", "CNV2"])]
        tgd_ca = cn[["TGD_B1Cp", "TGD_B2ap"]].apply(pd.to_numeric, errors="coerce").groupby(cn["sv"]).median().dropna()
    k1, _ = ionosphere_free_coefficients("B1I", "B3I")
    c1, c2 = ionosphere_free_coefficients("B1C", "B2a")
    rows = []
    for sv, g in rec.groupby("sv"):
        nv = d12[d12["sv"] == sv].sort_values("time")
        if nv.empty or sv not in tgd1.index or not np.isfinite(tgd1[sv]):
            continue
        t_bdt = (g["time"] + pd.Timedelta(seconds=_BDT_MINUS_GPST_S)).to_numpy(dtype="datetime64[ns]")
        toc = nv["time"].to_numpy(dtype="datetime64[ns]")
        if len(toc) > 1:
            pos = np.clip(np.searchsorted(toc, t_bdt), 1, max(len(toc) - 1, 1))
        else:
            pos = np.zeros(len(t_bdt), int)
        if len(toc) > 1:
            left = pos - 1
            nearer = np.abs(t_bdt - toc[left]) <= np.abs(toc[pos] - t_bdt)
            pos = np.where(nearer, left, pos)
        dt = (t_bdt - toc[pos]) / np.timedelta64(1, "s")
        a = nv[["SVclockBias", "SVclockDrift", "SVclockDriftRate"]].apply(pd.to_numeric, errors="coerce")
        a = a.fillna({"SVclockDrift": 0.0, "SVclockDriftRate": 0.0}).to_numpy(dtype=float)[pos]
        dt_brd = a[:, 0] + a[:, 1] * dt + a[:, 2] * dt ** 2
        y = CLIGHT * (g["clk"].to_numpy(dtype=float) - dt_brd)
        ok = (np.abs(dt) <= 3600.0) & np.isfinite(y)
        d1 = -k1 * CLIGHT * float(tgd1[sv])
        d2 = np.nan
        if tgd_ca is not None and sv in tgd_ca.index:
            d2 = -CLIGHT * (c1 * float(tgd_ca.loc[sv, "TGD_B1Cp"]) - c2 * float(tgd_ca.loc[sv, "TGD_B2ap"]))
        for tt, yy in zip(g["time"].to_numpy()[ok], y[ok]):
            rows.append((tt, sv, yy, d1, d2))
    if not rows:
        out["reason"] = "no BDS-3 satellite with precise and broadcast clocks at the same epochs"
        return out
    df = pd.DataFrame(rows, columns=["t", "sv", "y", "d1", "d2"])

    def scatter(d):
        e = df["y"] - d
        frame = df.assign(e=e)[np.isfinite(e)]
        if frame.empty:
            return None, 0
        common = frame.groupby("t")["e"].median()
        resid = frame["e"] - frame["t"].map(common)
        per_sat = resid.groupby(frame["sv"]).median()
        return (float(per_sat.std()) if len(per_sat) > 1 else None), int(per_sat.size)

    s1, n1 = scatter(df["d1"])
    s2, n2 = scatter(df["d2"])
    out.update(std_h1_m=None if s1 is None else round(s1, 3), std_h2_m=None if s2 is None else round(s2, 3),
               n_bds3=n1, n_bds3_h2=n2)
    if n1 < DATUM_TEST_MIN_SATS or s1 is None:
        out["reason"] = f"{n1} BDS-3 satellites with precise and broadcast clocks (at least {DATUM_TEST_MIN_SATS})"
        return out
    if n2 >= DATUM_TEST_MIN_SATS and s2 is not None:
        out["two_sided"] = True
        if s1 <= DATUM_TEST_ACCEPT_M and s2 >= DATUM_TEST_RATIO * s1:
            out["datum"] = "B1IB3I"
        elif s2 <= DATUM_TEST_ACCEPT_M and s1 >= DATUM_TEST_RATIO * s2:
            out["datum"] = "B1CB2a"
        else:
            out["reason"] = (f"scatter IF(B1I,B3I) {s1:.2f} m, IF(B1C,B2a) {s2:.2f} m: neither at most "
                             f"{DATUM_TEST_ACCEPT_M} m with the other {DATUM_TEST_RATIO}x larger")
        return out
    if s1 <= DATUM_TEST_ACCEPT_M:
        out["datum"] = "B1IB3I"
    elif s1 >= DATUM_TEST_REJECT_M:
        # IF(B1I,B3I) rejected; the other datum of the MGEX products is IF(B1C,B2a) (not testable without B-CNAV)
        out["datum"] = "B1CB2a"
        out["reason"] = (f"IF(B1I,B3I) rejected (scatter {s1:.2f} m >= {DATUM_TEST_REJECT_M} m); IF(B1C,B2a) not "
                         f"tested (no B-CNAV TGD_B1Cp/TGD_B2ap in the navigation data)")
    else:
        out["reason"] = (f"scatter IF(B1I,B3I) {s1:.2f} m between {DATUM_TEST_ACCEPT_M} and {DATUM_TEST_REJECT_M} m; "
                         f"IF(B1C,B2a) not testable without B-CNAV records (RINEX 4 navigation)")
    return out


#: GPS CNAV inter-signal corrections of a code (signal -> tracking attribute -> fields of the navigation frame):
#: IS-GPS-200 30.3.3.3.1.1.1 (L1 C/A, L2C: dt_i = dt - TGD + ISC_i) and IS-GPS-705 20.3.3.3.1.2 (L5 I5/Q5). The
#: RINEX 4 reader puts them, with TGD_CNAV, on the GPS rows from the latest CNAV record of the satellite (contract
#: since v1.0.0); a combined I+Q (X) tracking takes the mean of the two.
_GPS_CNAV_ISC = {
    "L1": {"C": ("ISC_L1CA",)},
    "L2": {"S": ("ISC_L2C",), "L": ("ISC_L2C",), "X": ("ISC_L2C",)},
    "L5": {"I": ("ISC_L5I5",), "Q": ("ISC_L5Q5",), "X": ("ISC_L5I5", "ISC_L5Q5")},
}
#: |TGD_CNAV - TGD_LNAV| above which the two messages are reported as inconsistent [s]
GPS_CNAV_TGD_TOL_S = 1e-9
#: IS-GPS-200 30.3.3.3.1.1 / IS-GPS-705: the 13-bit pattern 1000000000000 of TGD and ISC_* means "not available";
#: scaled by 2^-35 s that is -2^-23 s (-119.2 ns, 35.7 m), a value a navigation writer may pass through.
GPS_CNAV_NOT_AVAILABLE_S = -2.0 ** -23


def _cnav_column_m(epoch: pd.DataFrame, column: str) -> Optional[np.ndarray]:
    """CNAV group-delay field ``column`` [s] of the rows in metres, NaN where it holds the "not available"
    pattern (:data:`GPS_CNAV_NOT_AVAILABLE_S`); None when the frame has no such column."""
    raw = _scaled_column(epoch, column, 1.0)
    if raw is None:
        return None
    raw = np.where(np.abs(raw - GPS_CNAV_NOT_AVAILABLE_S) <= 1e-15, np.nan, raw)
    return raw * CLIGHT


def _gps_cnav_bias_m(epoch: pd.DataFrame, code: str) -> Optional[np.ndarray]:
    """``b(code) = c (TGD - ISC)`` from the CNAV fields of the rows (TGD of the same CNAV record, else the LNAV
    TGD); NaN on rows without a CNAV record; None when the code has no CNAV term or the frame no CNAV field."""
    try:
        names = _GPS_CNAV_ISC.get(code_signal("G", code), {}).get(code[2:3])
    except ValueError:
        return None
    if not names:
        return None
    cols = [c for c in (_cnav_column_m(epoch, name) for name in names) if c is not None]
    if not cols:
        return None
    stack = np.vstack(cols)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    isc = np.where(count > 0, np.where(finite, stack, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan)
    tgd = _cnav_column_m(epoch, "TGD_CNAV")
    lnav = _scaled_column(epoch, "TGD", CLIGHT)
    if tgd is None:
        tgd = lnav
    elif lnav is not None:
        tgd = np.where(np.isfinite(tgd), tgd, lnav)
    if tgd is None:
        return None
    out = tgd - isc
    return out if np.isfinite(out).any() else None


def _gps_isc_missing(code: str) -> list[str]:
    """GPS inter-signal correction a code needs beyond the L1/L2 P(Y) group delay (none for C/A, whose
    P1-C/A bias, 0.7 ns rms, the ICD leaves to ISC_L1C/A of CNAV)."""
    signal, attr = code_signal("G", code), code[2:3]
    if signal == "L2" and attr not in _GPS_PY_ATTRS:
        # IS-GPS-200 30.3.3.3.1.1.1: dt_L2C = dt - TGD + ISC_L2C; without CNAV the L2C code takes
        # the L2 P(Y) group delay gamma12*TGD
        return [f"L2C - L2P(Y) code bias of {code} (CNAV ISC_L2C not available; {code} is taken as L2 P(Y))"]
    if signal == "L5":
        return [f"ISC_L5 ({code} - C1W code bias, CNAV / DSB product)"]
    return []


def _datum_label(system: str, refs: Sequence[str]) -> str:
    """Precise clock datum codes for a message: ``IF(C2I,C6I)``; a BeiDou datum other than PRECISE_CLOCK_REFERENCE
    names its signals as well, ``IF(C1P,C5P) = IF(B1C,B2a)``."""
    text = f"IF({refs[0]},{refs[1]})" if len(refs) == 2 else str(refs[0])
    if system == "C" and len(refs) == 2 and tuple(refs) != tuple(PRECISE_CLOCK_REFERENCE["C"]):
        text += f" = {bds_pair_name(refs)}"
    return text


def _bcnav_taken_as_zero(field: str, note: str = "", refs: Optional[Sequence[str]] = None) -> str:
    if refs is None or tuple(refs) == tuple(PRECISE_CLOCK_REFERENCE["C"]):
        shift = "the TGD1 shift of the precise clock datum IF(B1I,B3I)"
    else:
        shift = f"the shift of the precise clock datum {_datum_label('C', refs)}"
    return (f"broadcast {field} (B-CNAV{note}; a RINEX 3 navigation file has no B-CNAV records) is taken as 0: "
            f"only {shift} is applied")


def _broadcast_bias_m(epoch: pd.DataFrame, system: str, code: str, nav_message: Optional[str],
                      bcnav_zero: bool = False, cnav: bool = False,
                      clock_datum: Optional["PreciseClockDatum"] = None,
                      refs: Optional[Sequence[str]] = None) -> tuple[np.ndarray, bool, list[str]]:
    """``b(code)`` relative to the broadcast clock datum from the navigation-message fields.

    Returns (values [m], exact, missing): ``exact`` is False when the ICD equation needs an
    inter-signal term the message does not carry (taken as zero, named in ``missing``); values are
    NaN where a required field is absent. ``bcnav_zero`` (precise clocks, the datum shift known):
    a BeiDou B-CNAV group delay absent from the navigation data (TGD_B1Cp, TGD_B2ap, TGD_B2bI) is
    such a term as well, as ISC_L5 for GPS L5; TGD_B1Cp is approximated by TGD1 (named in ``missing``).
    ``cnav``: a GPS code with a CNAV inter-signal correction on the rows (RINEX 4: ISC_L1CA, ISC_L2C,
    ISC_L5I5/Q5) takes ``c (TGD - ISC)`` there; callers ask for it after a
    product inter-signal term, which keeps priority. ``clock_datum`` (precise BeiDou clocks): the satellite
    generations of the session decide which band-7 codes are B2b-I (BDS-3, GEO included) instead of the PRN rule;
    ``refs``: the datum codes, for the message texts. With broadcast clocks the sessions pass the satellite
    generations of the run as well (``source='broadcast'``).
    """
    n = len(epoch)
    nan = np.full(n, np.nan)
    signal = code_signal(system, code)
    attr = code[2:3]

    if system == "G":
        tgd = _scaled_column(epoch, "TGD", CLIGHT)
        if tgd is None:
            return nan, False, ["broadcast TGD"]
        # IS-GPS-200 20.3.3.3.3.2 (LNAV): dt_L1P(Y) = dt - TGD, dt_L2P(Y) = dt - gamma12*TGD; a C/A
        # user applies TGD as for L1 P(Y) (the P1-C/A bias only comes with CNAV ISC_L1C/A);
        # IS-GPS-705 20.3.3.3.1.2: dt_L5 = dt - TGD + ISC_L5 (ISC_L5 = t_L1P(Y) - t_L5, CNAV only)
        if signal == "L1":
            value, exact, missing = tgd, attr in _GPS_PY_ATTRS, []
        elif signal == "L2":
            value, exact, missing = gamma("L1", "L2") * tgd, attr in _GPS_PY_ATTRS, _gps_isc_missing(code)
        else:
            value, exact, missing = tgd, False, _gps_isc_missing(code)
        isc = _gps_cnav_bias_m(epoch, code) if cnav and not exact else None
        if isc is not None:
            have = np.isfinite(isc)
            value = np.where(have, isc, value)
            if have.all():
                return value, True, []
            # rows without a CNAV record (e.g. GPS IIR) keep the LNAV group delay
            missing = [f"{m} on rows without a CNAV record" for m in missing]
        return value, exact, missing

    if system == "E":
        bgd = {"E5a": _scaled_column(epoch, "BGDe5a", CLIGHT), "E5b": _scaled_column(epoch, "BGDe5b", CLIGHT)}
        if bgd["E5a"] is None and bgd["E5b"] is None:
            return nan, False, ["broadcast BGD"]
        inav = _galileo_inav_rows(epoch, nav_message)
        # Galileo OS SIS ICD 5.1.5 with the clock of IF(E1,Ey) (y = E5b for I/NAV, E5a for F/NAV):
        #   dt_E1 = dt - BGD(E1,Ey), dt_Ey = dt - (fE1/fEy)^2 BGD(E1,Ey), and for the other E5 signal
        #   dt_Ez = dt - [BGD(E1,Ey) + ((fE1/fEz)^2 - 1) BGD(E1,Ez)]  (from dt_IF(E1,Ez) = dt_E1 + BGD(E1,Ez))
        out = nan.copy()
        missing: list[str] = []
        for is_inav, y in ((True, "E5b"), (False, "E5a")):
            rows = inav == is_inav
            if not rows.any():
                continue
            if bgd[y] is None:
                missing.append(f"broadcast BGD(E1,{y})")
                continue
            value = bgd[y][rows].copy()
            if signal != "E1":
                if signal == y:
                    value = gamma("E1", y) * bgd[y][rows]
                elif is_inav and bgd[signal] is not None:
                    # only I/NAV carries both BGDs; the F/NAV BGD(E1,E5b) field is a placeholder
                    value = value + (gamma("E1", signal) - 1.0) * bgd[signal][rows]
                else:
                    missing.append(f"broadcast BGD(E1,{signal}) ({'I/NAV' if is_inav else 'F/NAV'} record)")
                    continue
            out[rows] = value
        return out, True, missing

    if system == "C":
        cnav_rows = np.zeros(n, dtype=bool)
        if "nav_message" in epoch.columns:
            cnav_rows = epoch["nav_message"].astype(str).isin(_BDS_CNAV).to_numpy()
        if signal == "B3I":
            return np.zeros(n), True, []
        bds3_b2b = np.zeros(n, dtype=bool)
        if signal == "B2I":
            # BDS-3 satellites transmit no B2I: their band-7 I/Q/X code (RINEX 3.02-3.04 receivers)
            # is B2b-I, whose group delay is TGD_B2bI (the D1 TGD2 field of BDS-3 repeats TGD1)
            sv = _row_satellites(epoch) if clock_datum is not None and clock_datum.system == system else None
            if sv is not None:
                # the session's generations (ANTEX block, B-CNAV records): the PRN rule is wrong after 2023,
                # C06-C08/C11-C14 are BDS-3 MEO/IGSO and C01-C04 BDS-3 GEO satellites in 2026 (bug-fix stage 13b;
                # precise clocks since 13b, broadcast clocks since 13d, NQ-SPP-32)
                bds3_b2b = np.array([clock_datum.generation_of(s) in (BDS3, BDS3_GEO) for s in sv], dtype=bool)
            else:
                bds3_b2b = _row_prn(epoch) >= _BDS3_FIRST_PRN
        if signal in ("B1I", "B2I"):
            # BDS-SIS-ICD-B1I 5.2.4.10: dt_B1I = dt - TGD1, dt_B2I = dt - TGD2 (D1/D2 only; the RINEX 4
            # reader fills TGD1/TGD2 of B-CNAV records with 0.0)
            field = "TGD1" if signal == "B1I" else "TGD2"
            value = _scaled_column(epoch, field, CLIGHT)
            missing: list[str] = []
            if value is None:
                value, missing = nan.copy(), [f"broadcast {field} (D1/D2)"]
            value = np.where(cnav_rows, np.nan, value)
            if bds3_b2b.any():
                b2b = _scaled_column(epoch, "TGD_B2bI", CLIGHT)
                if b2b is None and bcnav_zero:
                    missing.append(_bcnav_taken_as_zero("TGD_B2bI", f", {code} of BDS-3 satellites is B2b-I",
                                                        refs))
                    b2b = np.zeros(n)
                elif b2b is None:
                    missing.append(f"broadcast TGD_B2bI (B-CNAV; {code} of BDS-3 satellites is B2b-I)")
                    b2b = nan
                value = np.where(bds3_b2b, b2b, value)
            return value, not missing, missing
        # BDS-SIS-ICD-B1C 7.6 / B2a 7.6: dt_B1Cp = dt - TGD_B1Cp, dt_B1Cd = dt - (TGD_B1Cp + ISC_B1Cd),
        # dt_B2ap = dt - TGD_B2ap, dt_B2ad = dt - (TGD_B2ap + ISC_B2ad); BDS-SIS-ICD-B2b 7.5:
        # dt_B2bI = dt - TGD_B2bI. Combined data+pilot tracking (X/Z) follows the pilot convention.
        field, isc = {"B1C": ("TGD_B1Cp", "ISC_B1Cd"), "B2a": ("TGD_B2ap", "ISC_B2ad"),
                      "B2b": ("TGD_B2bI", None)}[signal]
        value = _scaled_column(epoch, field, CLIGHT)
        if value is None and bcnav_zero and signal == "B1C":
            # author decision (bug-fix stage 8, NQ-SPP-23): B1C (1575.42 MHz) takes the D1/D2 TGD1 of B1I
            # (1561.098 MHz) as a stand-in for TGD_B1Cp, so b_prc(B1C) = b_prc(B1I) = -k2 c TGD1 (BRUX
            # 2024-035 SPP B1C on GRG orbits with GIM: 3D RMS 12.2 m with the TGD1 shift of the datum alone,
            # 1.7 m with the stand-in; TGD_B1Cp - TGD1 of BRD400DLR: 3.8 ns rms); B2a/B2b keep the shift
            # alone (no stand-in of a near frequency)
            tgd1 = _scaled_column(epoch, "TGD1", CLIGHT)
            if tgd1 is not None:
                note = (f"broadcast TGD_B1Cp (B-CNAV; a RINEX 3 navigation file has no B-CNAV records): TGD1 of "
                        f"B1I is used as an approximation of it")
                if attr == "D":
                    note += "; ISC_B1Cd of the data component is taken as 0"
                return np.where(cnav_rows, np.nan, tgd1), False, [note]
        if value is None and bcnav_zero:
            return np.zeros(n), False, [_bcnav_taken_as_zero(field, refs=refs)]
        if value is None:
            return nan, False, [f"broadcast {field} (B-CNAV)"]
        if isc is not None and attr == "D":
            isc_m = _scaled_column(epoch, isc, CLIGHT)
            if isc_m is None:
                return value, False, [f"broadcast {isc} (data component {code})"]
            value = value + isc_m
        return value, True, []

    raise ValueError(f"Unsupported GNSS system: {system!r}")


def _product_isc_anchor_m(epoch, system, code, refs, product, include_station):
    """GPS code without a product bias and without navigation data: the product bias of the P(Y) code
    the ICD gives its group delay to when the inter-signal correction is unknown (L1 C/A, L2C, L5 as
    L1/L2 P(Y): IS-GPS-200 20.3.3.3.3.2, IS-GPS-705 20.3.3.3.1.2); None when not applicable."""
    anchor = _PRECISE_ISC_ANCHOR.get(system, {}).get(code_signal(system, code))
    if anchor is None or anchor == code:
        return None
    b_anchor, kind = _product_osb_m(epoch, anchor, include_station, product), "osb"
    if b_anchor is None:
        b_anchor, kind = _product_datum_bias_m(epoch, system, anchor, refs, product, include_station)
    if b_anchor is None:
        return None
    return b_anchor, kind, [f"product bias of {code} (not in the bias product; that of {anchor} is used)"] \
        + _gps_isc_missing(code)


def _datum_bias_m(epoch, system, refs, bias_of) -> tuple[Optional[np.ndarray], list[str]]:
    """IF combination k1*b(ref1) - k2*b(ref2) (or b(ref) for a single-signal datum)."""
    if len(refs) == 1:
        return bias_of(refs[0])
    sig1, sig2 = (code_signal(system, ref) for ref in refs)
    k1, k2 = ionosphere_free_coefficients(sig1, sig2)
    b1, m1 = bias_of(refs[0])
    b2, m2 = bias_of(refs[1])
    if b1 is None or b2 is None:
        return None, m1 + m2
    return k1 * b1 - k2 * b2, m1 + m2


def _broadcast_reference_codes(system: str, nav_message: Optional[str]) -> Optional[tuple[str, ...]]:
    """Codes of the broadcast clock datum: GPS IF(C1W,C2W), Galileo IF(C1C,C7Q|C5Q), BeiDou C6I."""
    if system == "G":
        return "C1W", "C2W"
    if system == "C":
        return ("C6I",)
    if system == "E" and nav_message in ("I/NAV", "F/NAV"):
        return ("C1C", "C7Q") if nav_message == "I/NAV" else ("C1C", "C5Q")
    return None


def _product_datum_bias_m(epoch, system, code, refs, product, include_station):
    """``B_code - [k1 B_ref1 - k2 B_ref2]`` (or ``B_code - B_ref``) from product OSBs or DSB chains."""
    osb = _product_osb_m(epoch, code, include_station, product)
    if osb is not None:
        ref_osb, _ = _datum_bias_m(
            epoch, system, refs, lambda ref: (_product_osb_m(epoch, ref, include_station, product), []),
        )
        if ref_osb is not None:
            return osb - ref_osb, "osb"
    # with DSB chains D(code,ref) = B_code - B_ref: k1 D(code,ref1) - k2 D(code,ref2), since k1 - k2 = 1
    dsb, _ = _datum_bias_m(
        epoch, system, refs, lambda ref: (_product_dsb_m(epoch, code, ref, product, include_station), []),
    )
    if dsb is not None:
        return dsb, "dsb"
    return None, None


def satellite_code_bias_m(
    epoch: pd.DataFrame,
    system: str,
    code: str,
    *,
    clock: str,
    nav_message: Optional[str] = None,
    include_station: bool = False,
    product_first: bool = False,
    use_product: bool = True,
    broadcast_fallback: bool = True,
    _product: Optional[_ProductColumns] = None,
    _isc_anchor: bool = True,
    _bcnav_zero: bool = True,
    clock_datum: Optional[PreciseClockDatum] = None,
    _refs: Optional[tuple] = None,
) -> tuple[np.ndarray, str, list[str]]:
    """Satellite code bias ``b(code)`` [m] relative to the clock datum, its source and missing terms.

    See :func:`code_group_delay_correction` for the conventions; NaN where unavailable. Precise clocks:
    ``use_product=False`` skips the product biases, ``broadcast_fallback=False`` the broadcast ones;
    ``clock_datum`` (:class:`PreciseClockDatum`, BeiDou): the datum of each row's satellite instead of
    PRECISE_CLOCK_REFERENCE (rows of different datums are evaluated apart and merged). Under a datum other than
    PRECISE_CLOCK_REFERENCE a product OSB is taken as given only when the OSB products declare that datum
    (``clock_datum.osb_as_given``); otherwise it refers to the product's own clock datum (e.g. a CAS or 2024-style
    OSB product with C2I/C6I against COD 2026 clocks of IF(B1C,B2a)) and is referred to the session's datum with
    the product's OSBs of the datum codes, ``OSB(code) - [k1 OSB(ref1) - k2 OSB(ref2)]``, else the DSB chain to
    the datum codes, else the broadcast group delays moved to the datum.
    """
    n = len(epoch)
    if clock == "precise" and _refs is None:
        groups = _datum_groups(epoch, system, clock_datum)
        if len(groups) > 1:
            kwargs = dict(clock=clock, nav_message=nav_message, include_station=include_station,
                          product_first=product_first, use_product=use_product,
                          broadcast_fallback=broadcast_fallback, _isc_anchor=_isc_anchor, _bcnav_zero=_bcnav_zero,
                          clock_datum=clock_datum)
            values = np.full(n, np.nan)
            sources: list[str] = []
            missing: list[str] = []
            for refs, rows in groups:
                idx = np.flatnonzero(rows)
                b, src, miss = satellite_code_bias_m(epoch.iloc[idx], system, code, _refs=refs, **kwargs)
                values[idx] = np.broadcast_to(np.asarray(b, dtype=float), (idx.size,))
                sources.append(src)
                missing.extend(m for m in miss if m not in missing)
            return values, _source_label(sources), missing
        _refs = groups[0][0]
    product = _ProductColumns(epoch) if _product is None else _product
    if clock == "precise":
        refs = tuple(_refs)
        label = _datum_label(system, refs)
        foreign_osb = False
        if use_product:
            # product: the OSB is relative to the product's clock datum, taken as given for
            # PRECISE_CLOCK_REFERENCE; a DSB chain to the reference codes gives b = B_code - [k1 B_ref1 - k2 B_ref2]
            osb = _product_osb_m(epoch, code, include_station, product)
            if osb is not None and (refs == tuple(PRECISE_CLOCK_REFERENCE[system])
                                    or (clock_datum is not None and clock_datum.osb_as_given)):
                return osb, "osb", []
            # another BeiDou datum (SPP-080): the OSB referred to it by the product's OSBs of the datum codes
            # (_product_datum_bias_m: OSB(code) - [k1 OSB(ref1) - k2 OSB(ref2)]), else the DSB chain
            dsb, kind = _product_datum_bias_m(epoch, system, code, refs, product, include_station)
            if dsb is not None:
                return dsb, kind, []
            foreign_osb = osb is not None
        if not broadcast_fallback:
            anchored = _product_isc_anchor_m(epoch, system, code, refs, product, include_station) \
                if use_product and _isc_anchor else None
            if anchored is not None:
                return anchored
            return (np.full(n, np.nan), "none",
                    [f"product bias of {code} (OSB, or DSB to the precise clock datum {label}"
                     + (f"; {_foreign_osb_note(code, refs)}" if foreign_osb else "")
                     + "), and broadcast group delays for all codes of the combination"])
        # no product bias: broadcast group delay moved to the precise datum,
        # b_prc(code) = b_brd(code) - [k1 b_brd(ref1) - k2 b_brd(ref2)]
        def broadcast_of(ref, bcnav_zero=False):
            # GPS: with the CNAV inter-signal corrections where the navigation data carry them (no product here)
            value, _, miss = _broadcast_bias_m(epoch, system, ref, nav_message, bcnav_zero=bcnav_zero, cnav=True,
                                               clock_datum=clock_datum, refs=refs)
            return value, miss

        b_ref, missing_ref = _datum_bias_m(epoch, system, refs, broadcast_of)
        datum_known = b_ref is not None and bool(np.isfinite(b_ref).any())
        # BeiDou B1C/B2a/B2b without B-CNAV records (RINEX 3 navigation file): the available part, the
        # shift of the datum (-k1 c TGD1), with the missing B-CNAV group delay taken as 0, as GPS L5
        # takes c TGD without ISC_L5 (B1C on GRG orbits 2024-035: 24.6 -> 12.2 m); B1C takes TGD1 as a
        # stand-in for TGD_B1Cp (stage 8, see _broadcast_bias_m). Single codes only
        # (``_bcnav_zero``): in a combination the missing terms cancel most of the shift (BDS-3 per
        # satellite, 2024-035 BRD400DLR: B1C 9.2 -> 4.6 m, B2a 15.1 -> 4.3 m, but IF(B1C,B2a) 7.9 ->
        # 13.4 m, IF(B1C,B3I) 3.3 -> 13.1 m with the shift alone)
        b_code, missing = broadcast_of(code, bcnav_zero=datum_known and _bcnav_zero)
        if np.isfinite(b_code).any() and datum_known:
            if foreign_osb:
                missing = [f"product bias of {code} in the precise clock datum {label} "
                           f"({_foreign_osb_note(code, refs)}; the broadcast group delay moved to that datum is "
                           f"used)"] + missing
            elif use_product and product.estimated:
                missing = [f"product bias of {code} (not in the bias product; the broadcast group delay "
                           f"is used)"] + missing
            elif system == "C":
                # accepted fallback (author decision, bug-fix stage 7), reported: the broadcast BeiDou group
                # delays are not in the product datum (2024-035: TGD1 - CAS DSB(C2I-C6I) -3.75 ns on BDS-2,
                # +1.45 ns on BDS-3, ~3 m on B1I); GPS and Galileo broadcast clocks share the IF datum
                missing = [f"product bias of {code} (no bias product given; the broadcast group delay moved to "
                           f"the precise clock datum {label} is used)"] + missing
            return b_code - b_ref, "broadcast", missing
        anchored = _product_isc_anchor_m(epoch, system, code, refs, product, include_station) \
            if use_product and _isc_anchor else None
        if anchored is not None:
            return anchored
        lacking = ", ".join(dict.fromkeys(missing + missing_ref)) or "navigation fields"
        return (np.full(n, np.nan), "none",
                [f"product bias of {code} (OSB, or DSB to the precise clock datum {label}"
                 f"{'; ' + _foreign_osb_note(code, refs) if foreign_osb else ''}) "
                 f"and broadcast group delay ({lacking}; nav_path)"])

    if clock != "broadcast":
        raise ValueError(f"clock must be 'broadcast' or 'precise', got {clock!r}")
    refs = _broadcast_reference_codes(system, nav_message)
    if product_first and refs is not None:
        b, kind = _product_datum_bias_m(epoch, system, code, refs, product, include_station)
        if b is not None:
            return b, kind, []
    # BeiDou: the satellite generations of the session decide B2I / B2b-I (``clock_datum``, NQ-SPP-32)
    value, exact, missing = _broadcast_bias_m(epoch, system, code, nav_message, clock_datum=clock_datum)
    if exact and not missing:
        return value, "broadcast", []
    # the message lacks an inter-signal term: take it from the product, anchored to a code whose
    # broadcast group delay is exact: b(code) = b_brd(anchor) + (B_code - B_anchor)
    for anchor in _BROADCAST_ANCHOR_CODES[system]:
        anchor_value, anchor_exact, _ = _broadcast_bias_m(epoch, system, anchor, nav_message,
                                                          clock_datum=clock_datum)
        if not anchor_exact or not np.isfinite(anchor_value).any():
            continue
        diff, kind = _product_difference_m(epoch, code, anchor, product, include_station)
        if diff is not None:
            return anchor_value + diff, f"broadcast+{kind}", []
    if system == "G":
        # no product inter-signal term: the CNAV one of the navigation data (RINEX 4), where given
        value, _, missing = _broadcast_bias_m(epoch, system, code, nav_message, cnav=True)
    return value, ("broadcast" if np.isfinite(value).any() else "none"), missing


def _foreign_osb_note(code: str, refs: Sequence[str]) -> str:
    """Why a product OSB is not used under a BeiDou datum other than PRECISE_CLOCK_REFERENCE."""
    return (f"the product OSB of {code} refers to the product's own clock datum and the product has no OSB of "
            f"{' and '.join(refs)} to refer it to this one")


def _is_precise_datum(system: str, codes: Sequence[str], weights: Sequence[float],
                      refs: Optional[Sequence[str]] = None) -> bool:
    """True when ``sum(w_i * P_i)`` is the ionosphere-free combination of the precise clock datum codes
    (``refs``, default PRECISE_CLOCK_REFERENCE; GPS L1 C/A, L2C and L5 codes stand for the P(Y) code the ICD
    gives their group delay to; Galileo and BeiDou codes of the datum signals with another tracking mode, e.g.
    C1X/C5X, C2X/C6X, stand for the datum codes up to their inter-mode bias, ~0.1 ns)."""
    refs = PRECISE_CLOCK_REFERENCE.get(system) if refs is None else tuple(refs)
    if refs is None or len(codes) != 2:
        return False
    if system == "G":
        anchors = _PRECISE_ISC_ANCHOR.get(system, {})
    else:
        anchors = {code_signal(system, ref): ref for ref in refs}
    try:
        mapped = [anchors.get(code_signal(system, code), code) for code in codes]
    except ValueError:
        return False
    if mapped == list(refs[::-1]):
        mapped, weights = mapped[::-1], tuple(weights)[::-1]
    if mapped != list(refs):
        return False
    k1, k2 = ionosphere_free_coefficients(*(code_signal(system, ref) for ref in refs))
    return bool(np.allclose(weights, (k1, -k2), rtol=1e-6))


def _row_names(epoch: pd.DataFrame, rows: np.ndarray, sv: Optional[np.ndarray]) -> list[str]:
    if sv is None:
        return [f"{int(rows.sum())} row(s) without a satellite id"]
    return sorted(set(sv[rows]))


def _satellites_missing(epoch: pd.DataFrame, codes: Sequence[str], per_code,
                        product_note: str = "") -> list[tuple[str, str]]:
    """Satellites whose bias is NaN although its code has a source (a satellite without a navigation
    record nearby, or absent from the bias product), as ``(template, satellite)``: a whole missing
    field (``missing`` of the code, e.g. ``broadcast TGD_B2bI``) already explains its NaN rows; one
    taken as 0 (a B-CNAV group delay beside the TGD1 datum shift) does not, the NaN rows then lack
    the datum shift (no D1/D2 record)."""
    sv = None
    lacking: dict[tuple[str, str], list[str]] = {}
    for code, (b, source, miss) in zip(codes, per_code):
        rows = ~np.isfinite(b)
        absent = any(m.startswith("broadcast ") and " is taken as 0" not in m for m in miss)
        if source in ("none", "datum") or not rows.any() or absent:
            continue
        sv = _row_satellites(epoch) if sv is None else sv
        kind = "product" if source in ("osb", "dsb") else "broadcast"
        for name in _row_names(epoch, rows, sv):
            lacking.setdefault((name, kind), []).append(code)
    out = []
    for (name, kind), lacking_codes in sorted(lacking.items()):
        what = ", ".join(dict.fromkeys(lacking_codes))
        if kind == "product":
            out.append((f"product bias of {what} for {{satellites}} (not in the bias product{product_note})", name))
        else:
            out.append((f"broadcast group delay of {what} for {{satellites}} (no navigation record with it)", name))
    return out


def _source_label(sources: Iterable[str]) -> str:
    labels = sorted(set(s for s in sources if s != "none"))
    return "+".join(dict.fromkeys(p for s in labels for p in s.split("+"))) or "none"


def _station_part_m(epoch, system, code, refs, source, product) -> np.ndarray:
    """Receiver part of the product bias of ``code`` as ``source`` gives it: ``STA_OSB_<code>`` of an
    OSB, the receiver DSBs to the precise clock datum of a DSB chain (``STA_BIAS_`` columns); 0 else."""
    n = len(epoch)
    if source == "osb":
        sta = _scaled_column(epoch, f"STA_OSB_{code}", NS_TO_M)
        return np.zeros(n) if sta is None else np.nan_to_num(sta)
    if source == "dsb":
        d, _ = _datum_bias_m(epoch, system, refs,
                             lambda ref: (_product_dsb_m(epoch, code, ref, product, True, station_only=True), []))
        return np.zeros(n) if d is None else np.nan_to_num(d)
    return np.zeros(n)


def _one_product_source(epoch, system, codes, per_code, refs, product, include_station, broadcast_biases):
    """Precise clocks, a combination whose codes all have product biases but from two products: the OSB of one
    product for some codes, the DSB chain of another product of dcb_path for the others (an OSB product without
    the pair's second code, e.g. COD/GRG/GFZ without Galileo C7Q or GPS C5Q, with CAS DSB). The realizations of
    the two products differ per satellite (COD/CAS GPS D12 0.26 ns rms, GRG/CAS 1.26 ns) and the mix multiplies
    the difference by the IF coefficients (x31.2 in IF(L2,L5), x42.8 in IF(E5a,E5b); BRUX 2024-035 E5aE5b COD+CAS
    6-h static 2.2 cm / t_c10 229.5 min against 1.1 cm / 55.5 min with CAS alone). On the rows where the mix
    gives every code:

    1. every code from the DSB product, ``b = k1 D(code,ref1) - k2 D(code,ref2)`` (one source);
    2. else the code without OSB tied to its partner's OSB, ``b = OSB(partner) + D(code, partner)`` (the OSB
       product's error enters once, not amplified);
    3. else the broadcast group delays of all codes moved to the datum (``broadcast_biases()``);
    4. else the mix as before (named).

    Returns (per_code, {row label: rows} or None, missing texts); per_code unchanged when the sources are not
    mixed (OSB-only and DSB-only products, single codes: bit-identical)."""
    sources = [source for _, source, _ in per_code]
    if not ({"osb", "dsb"} <= set(sources)):
        return per_code, None, []
    n = len(epoch)
    osb_codes = [code for code, source in zip(codes, sources) if source == "osb"]
    dsb_codes = [code for code, source in zip(codes, sources) if source == "dsb"]
    current = np.vstack([np.broadcast_to(np.asarray(b, dtype=float), (n,)) for b, _, _ in per_code])
    mixed = np.isfinite(current).all(axis=0)
    if not mixed.any():
        return per_code, None, []
    values = current.copy()
    left = mixed.copy()
    chosen: dict[str, np.ndarray] = {}
    what = ", ".join(codes)
    datum = _datum_label(system, refs)
    head = f"satellite OSB of {', '.join(dsb_codes)} (not in the OSB product of dcb_path, {', '.join(osb_codes)} is)"
    missing: list[str] = []

    # 1. every code from the DSB product
    dsb = []
    for code in codes:
        d, _ = _datum_bias_m(epoch, system, refs,
                             lambda ref, c=code: (_product_dsb_m(epoch, c, ref, product, include_station), []))
        dsb.append(np.full(n, np.nan) if d is None else np.asarray(d, dtype=float))
    dsb = np.vstack(dsb)
    rows = left & np.isfinite(dsb).all(axis=0)
    if rows.any():
        values[:, rows] = dsb[:, rows]
        chosen["dsb"] = rows
        left &= ~rows
        missing.append(f"{head}: the combination of {what} is taken as a whole from the DSB product of dcb_path "
                       f"(DSBs to the precise clock datum {datum}; one source for all its codes)")
    # 2. the code without OSB tied to its partner's OSB
    if left.any():
        anchor = current.copy()
        for i, code in enumerate(codes):
            if sources[i] != "dsb":
                continue
            tied = np.full(n, np.nan)
            for partner in osb_codes:
                d = _product_dsb_m(epoch, code, partner, product, include_station)
                if d is not None:
                    tied = current[codes.index(partner)] + d
                    break
            anchor[i] = tied
        rows = left & np.isfinite(anchor).all(axis=0)
        if rows.any():
            values[:, rows] = anchor[:, rows]
            chosen["dsb+osb"] = rows
            left &= ~rows
            missing.append(f"{head}: {', '.join(dsb_codes)} is taken as the OSB of {', '.join(osb_codes)} plus the "
                           f"DSB between them of the DSB product of dcb_path (no DSB chain of every code to the "
                           f"precise clock datum {datum})")
    # 3. the broadcast group delays of all codes
    if left.any():
        broadcast = broadcast_biases()
        bro = np.vstack([np.broadcast_to(np.asarray(b, dtype=float), (n,)) for b, _, _ in broadcast])
        rows = left & np.isfinite(bro).all(axis=0)
        if rows.any():
            values[:, rows] = bro[:, rows]
            chosen["broadcast"] = rows
            left &= ~rows
            missing.append(f"{head}: the combination of {what} is taken as the broadcast group delays of all its "
                           f"codes moved to the precise clock datum {datum} (no DSB chain between its codes)")
            missing.extend(f"{m} (in the broadcast group delays of the combination)"
                           for _, _, miss in broadcast for m in miss)
    if left.any():
        missing.append(f"{head}: the OSB of {', '.join(osb_codes)} is used with the DSB chain of "
                       f"{', '.join(dsb_codes)} of another product of dcb_path (no DSB chain between the codes, "
                       f"no broadcast group delays; the difference of the two products is amplified by the IF "
                       f"coefficients)")
    out = [(values[i], source, miss) for i, (_, source, miss) in enumerate(per_code)]
    return out, chosen, missing


def _tie_to_osb_partner(epoch, codes, per_code, product, include_station):
    """Precise clocks, a combination where some codes have a product OSB and the others no product bias to the
    datum: each of the others tied to an OSB code by a DSB chain, ``b = OSB(partner) + D(code, partner)``
    (fallback: the OSB product's realization enters once). Returns (per_code, partner names) or None
    when a code has no such chain."""
    osb = [(code, b) for code, (b, source, _) in zip(codes, per_code) if source == "osb"]
    if not osb:
        return None
    out, partners = [], []
    for code, (b, source, miss) in zip(codes, per_code):
        if source != "none":
            out.append((b, source, miss))
            continue
        for partner, b_partner in osb:
            d = _product_dsb_m(epoch, code, partner, product, include_station)
            if d is not None:
                out.append((np.asarray(b_partner, dtype=float) + d, "dsb+osb", []))
                partners.append(partner)
                break
        else:
            return None
    return out, ", ".join(dict.fromkeys(partners))


def _fill_product_gaps(epoch, system, codes, weights, per_code, broadcast_biases, product, include_station,
                       refs=None):
    """Precise clocks, every code of the combination given by the product: rows (satellites, epochs)
    the product leaves NaN for one of the codes take, for all codes of the combination, the next
    source (one source per row, as for whole codes): the product DSBs to the precise clock datum,
    the broadcast group delays moved to that datum (``broadcast_biases()``, navigation data needed),
    0 for the clock-datum combination itself (its group delay by definition); else they stay NaN.
    With ``include_station`` the replaced rows keep the receiver part of the codes' product source
    (one receiver bias for all satellites of the system, absorbed by the receiver clock).

    Returns (per_code, row labels, [(template, satellite)] of the replaced rows, missing terms of the
    broadcast group delays used).
    """
    n = len(epoch)
    values = np.vstack([np.broadcast_to(np.asarray(b, dtype=float), (n,)) for b, _, _ in per_code])
    gaps = ~np.isfinite(values).all(axis=0)
    labels = np.full(n, _source_label(source for _, source, _ in per_code), dtype=object)
    if not gaps.any():
        return per_code, labels, [], []
    values = values.copy()
    refs = PRECISE_CLOCK_REFERENCE[system] if refs is None else tuple(refs)
    sv = _row_satellites(epoch)
    what = ", ".join(codes)
    replaced: list[tuple[str, str]] = []
    station = np.zeros_like(values)
    if include_station:
        for i, (code, (_, source, _)) in enumerate(zip(codes, per_code)):
            station[i] = _station_part_m(epoch, system, code, refs, source, product)

    def take(rows, new_values, label, how):
        values[:, rows] = new_values[:, rows] + station[:, rows]
        labels[rows] = label
        if how:
            replaced.extend((f"product bias of {what} for {{satellites}} (not in the bias product): {how}", name)
                            for name in _row_names(epoch, rows, sv))
        return gaps & ~rows

    if any(source == "osb" for _, source, _ in per_code):
        # another product of dcb_path may give DSBs for the satellite: b = k1 D(code,ref1) - k2 D(code,ref2)
        dsb = []
        for code in codes:
            d, _ = _datum_bias_m(epoch, system, refs,
                                 lambda ref, c=code: (_product_dsb_m(epoch, c, ref, product, False), []))
            dsb.append(np.full(n, np.nan) if d is None else np.asarray(d, dtype=float))
        dsb = np.vstack(dsb)
        rows = gaps & np.isfinite(dsb).all(axis=0)
        if rows.any():
            gaps = take(rows, dsb, "dsb", "the group delay is taken as that of the product DSBs to the precise "
                                          f"clock datum {_datum_label(system, refs)}")
    # the combination of the clock datum (e.g. IF(C2I,C6I) of a BDS GEO GFZ gives no biases for) has
    # the satellite group delay 0 by definition, nothing is missing; its codes take the broadcast
    # group delays moved to the datum when available (their IF is 0 as well, the DCB of the pair is
    # kept for filters that correct each code), else 0
    datum = _is_precise_datum(system, codes, weights, refs)
    broadcast_missing: list[str] = []
    if gaps.any():
        broadcast = broadcast_biases()
        bro = np.vstack([np.broadcast_to(np.asarray(b, dtype=float), (n,)) for b, _, _ in broadcast])
        rows = gaps & np.isfinite(bro).all(axis=0)
        if rows.any() and datum:
            gaps = take(rows, bro, "datum", "")
        elif rows.any():
            gaps = take(rows, bro, "broadcast", "the group delay is taken as the broadcast one moved to the "
                                                f"precise clock datum {_datum_label(system, refs)}")
            # these terms concern only the satellites on the broadcast group delays
            broadcast_missing = [f"{m} (in the broadcast group delays of the satellites not in the bias product)"
                                 for _, _, miss in broadcast for m in miss]
    if gaps.any() and datum:
        gaps = take(gaps, np.zeros_like(values), "datum", "")
    per_code = [(values[i], source, miss) for i, (_, source, miss) in enumerate(per_code)]
    return per_code, labels, replaced, broadcast_missing


def _station_difference_m(epoch, system, code, nav_message, product, clock_datum=None) -> np.ndarray:
    """Receiver part of ``B_code - B_anchor`` for the anchor :func:`satellite_code_bias_m` takes with
    broadcast clocks (``STA_OSB_`` difference, or the receiver DSB chain); 0 when none."""
    n = len(epoch)
    for anchor in _BROADCAST_ANCHOR_CODES[system]:
        anchor_value, anchor_exact, _ = _broadcast_bias_m(epoch, system, anchor, nav_message, clock_datum=clock_datum)
        if not anchor_exact or not np.isfinite(anchor_value).any():
            continue
        if _product_osb_m(epoch, code, False, product) is not None \
                and _product_osb_m(epoch, anchor, False, product) is not None:
            sta = [_scaled_column(epoch, f"STA_OSB_{c}", NS_TO_M) for c in (code, anchor)]
            sta = [np.zeros(n) if s is None else np.nan_to_num(s) for s in sta]
            return sta[0] - sta[1]
        dsb = _product_dsb_m(epoch, code, anchor, product, True, station_only=True)
        if dsb is not None:
            return np.nan_to_num(dsb)
    return np.zeros(n)


def _fill_broadcast_product_gaps(epoch, system, codes, per_code, nav_message, per_satellite,
                                 product=None, include_station=False, clock_datum=None):
    """Broadcast clocks: a code whose inter-signal term comes from the product (``broadcast+osb``/
    ``broadcast+dsb``, e.g. GPS L5 ISC_L5 from a DSB) keeps, on the rows of satellites the product does
    not give, the message group delay without that term (as with no product; with ``include_station``
    plus the receiver part of the product term, as on the other rows), the satellites named in
    ``per_satellite``."""
    sv = None
    out = []
    for code, (b, source, miss) in zip(codes, per_code):
        rows = ~np.isfinite(b)
        if source.startswith("broadcast+") and rows.any():
            plain, _, plain_missing = _broadcast_bias_m(epoch, system, code, nav_message, cnav=True,
                                                        clock_datum=clock_datum)
            rows &= np.isfinite(plain)
            if rows.any():
                if include_station and product is not None:
                    plain = plain + _station_difference_m(epoch, system, code, nav_message, product,
                                                          clock_datum=clock_datum)
                b = np.where(rows, plain, b)
                sv = _row_satellites(epoch) if sv is None else sv
                terms = ", ".join(plain_missing) or "the inter-signal term the message lacks"
                per_satellite.extend(
                    (f"product bias of {code} for {{satellites}} (not in the bias product): {terms} is taken as 0 "
                     f"for them", name) for name in _row_names(epoch, rows, sv))
        out.append((b, source, miss))
    return out


def group_satellite_messages(missing: Iterable[str], per_satellite: Iterable[tuple[str, str]]) -> list[str]:
    """Entries of ``missing`` that name one satellite (``per_satellite``, see
    :class:`CodeBiasCorrection`) merged into one entry per template naming all their satellites."""
    single = {template.format(satellites=name): template for template, name in per_satellite}
    names: dict[str, list[str]] = {}
    for template, name in per_satellite:
        names.setdefault(template, [])
        if name not in names[template]:
            names[template].append(name)
    out: list[str] = []
    for item in missing:
        template = single.get(item)
        if template is None:
            text = item
        else:
            text = template.format(satellites=", ".join(sorted(names[template])))
        if text not in out:
            out.append(text)
    return out


def code_group_delay_correction(
    epoch: pd.DataFrame,
    system: str,
    codes: Sequence[str],
    weights: Sequence[float] = (1.0,),
    *,
    clock: str,
    nav_message: Optional[str] = None,
    include_station: bool = False,
    product_first: bool = False,
    zero_missing: bool = False,
    clock_datum: Optional[PreciseClockDatum] = None,
) -> CodeBiasCorrection:
    """Satellite group-delay correction of a code observable or code combination, all GNSS in one place.

    ``epoch``: rows = observations of one system; broadcast fields as read from the navigation file
    (``TGD``; ``BGDe5a``/``BGDe5b``; ``TGD1``/``TGD2``, ``TGD_B1Cp``/``TGD_B2ap``/``ISC_B1Cd``/``ISC_B2ad``/
    ``TGD_B2bI``, seconds) and product biases (``OSB_<code>``, ``BIAS_<code1>_<code2>`` DSB, ns).
    ``codes``/``weights``: the RINEX code observables of the observable and their coefficients, e.g.
    ``("C1C",), (1.0,)`` or ``("C1C", "C5Q"), (k1, -k2)``. ``clock``: ``"broadcast"`` or ``"precise"``.
    ``nav_message``: Galileo clock message ``"I/NAV"``/``"F/NAV"`` (else taken from the DataSrc
    column); BeiDou D1/D2 and B-CNAV rows are told apart by the ``nav_message`` column.
    ``include_station``: add the receiver biases (``STA_`` columns) of the product.
    ``product_first`` (broadcast clocks only): take the whole code bias from the product when it
    gives it relative to the broadcast clock datum (GPS IF(C1W,C2W), Galileo IF(C1C,C7Q|C5Q),
    BeiDou C6I) instead of the message group delays.
    ``zero_missing``: take the bias of a code that has none as zero instead of a NaN correction; on
    such a row the other codes of a combination are taken as zero too (correcting one code of the
    pair alone moves the combination by that code's whole weighted bias; PPP applies the same rule).
    The missing values are still named in ``missing``.

    Returns :class:`CodeBiasCorrection`: ``-sum(w_i * b_i)`` [m], with ``b`` the satellite code bias of
    each code relative to the clock datum:

    ========  ====================  ==========================================================
    system    clock datum           b(code), broadcast clock (ICD)
    ========  ====================  ==========================================================
    GPS       LNAV: IF(L1P, L2P)    L1: c TGD; L2: c (f1/f2)^2 TGD; L5: c (TGD - ISC_L5)
                                    (IS-GPS-200 20.3.3.3.3.2, IS-GPS-705 20.3.3.3.1.2)
    Galileo   I/NAV: IF(E1, E5b)    E1: c BGD_y; Ey: c (fE1/fEy)^2 BGD_y;
              F/NAV: IF(E1, E5a)    Ez: c [BGD_y + ((fE1/fEz)^2 - 1) BGD_z] (OS SIS ICD 5.1.5)
    BeiDou    D1/D2, B-CNAV: B3I    B1I: c TGD1; B2I: c TGD2; B3I: 0; B1C: c TGD_B1Cp (+ISC_B1Cd
                                    for C1D); B2a: c TGD_B2ap (+ISC_B2ad for C5D); B2b: c TGD_B2bI
    ========  ====================  ==========================================================

    Broadcast clocks: the message group delay is used; an inter-signal term the message lacks
    (GPS ISC_L5, ISC_L2C; BDS ISC of data components) comes from the product as
    ``b_brd(anchor) + (B_code - B_anchor)`` (anchor C1W/C2W, BDS C6I/C2I/C7I), else it is taken as
    zero and named in ``missing``.
    Precise clocks (datum IF(C1W,C2W), IF(C1C,C5Q), IF(C2I,C6I)): ``b = OSB(code)``; else from DSBs
    ``b = k1 D(code,ref1) - k2 D(code,ref2)``; else the broadcast group delay moved to the precise
    datum ``b_brd(code) - [k1 b_brd(ref1) - k2 b_brd(ref2)]`` (e.g. BDS B1I: -k2 c TGD1; a single BDS
    code without B-CNAV records: B1C takes TGD1 of B1I as an approximation of TGD_B1Cp, so
    ``b = -k2 c TGD1`` as for B1I (ISC_B1Cd of C1D taken as 0), B2a and B2b only the datum shift
    -k1 c TGD1 with the B-CNAV group delay taken as 0, all named in ``missing``, as GPS L5 takes c TGD
    without ISC_L5; not in a combination, where the missing terms cancel most of the shift); else (no
    navigation data), for a GPS L1 C/A, L2C or L5 code, the product bias of C1W/C2W/C1W, to which
    the ICD gives its group delay when the ISC is not known (L2C and L5 name the ISC in
    ``missing``); else NaN.
    A satellite (row) the product leaves NaN for a code it gives for other satellites takes the
    next source on that row, for all codes of the combination: product DSBs to the datum, the
    broadcast group delays moved to the datum, 0 for the datum combination; the satellites are
    named in ``missing``/``per_satellite`` (NaN when no source is left).
    With precise clocks a broadcast group delay is never added to a product bias, and the codes of
    one combination take their biases from one source: when one code has no product bias, the
    combination of the clock datum codes (GPS IF(L1,L2) with C/A or L2C codes as well) gets 0, any
    other all the broadcast group delays (without navigation data the GPS product biases above),
    and when those are not available either, it gets no
    correction (NaN): the biases of the codes against the IF datum are large and correlated (BDS
    OSB C6I up to 140 ns), so correcting one code of the pair alone is worse than correcting none.
    Product columns without a finite value, or holding only zeros (a placeholder for biases an analysis
    centre did not estimate), count as absent; NaN rows of a product column (a satellite or epoch the
    product does not cover) give NaN biases.
    One source with a product: when the codes of a combination take their
    product biases from different products (the OSB product gives one code, the DSB chain of another product
    of dcb_path the other one, e.g. COD OSB without C7Q + CAS DSB for Galileo E5aE5b), every code of the
    combination takes the DSB product's biases (``k1 D(code,ref1) - k2 D(code,ref2)``); without a DSB chain of
    every code to the datum, the code without OSB is tied to its partner, ``b = OSB(partner) + D(code,partner)``;
    else all codes take the broadcast group delays moved to the datum (the realizations of two products differ
    by 0.2-1.3 ns per satellite and the mix multiplies that by the IF coefficients, x31 in IF(L2,L5), x43 in
    IF(E5a,E5b)). Named in ``missing``.
    ``clock_datum``: :class:`PreciseClockDatum` of the session (BeiDou precise clocks): the datum codes of each
    row's satellite generation replace PRECISE_CLOCK_REFERENCE (rows of different datums are evaluated apart
    and merged); None: PRECISE_CLOCK_REFERENCE (as before v1.0.0). With broadcast clocks the sessions
    pass the satellite generations of the run (``source='broadcast'``): a band-7 code is B2b-I (TGD_B2bI) on a
    BDS-3 satellite and B2I (TGD2) on a BDS-2 one by the generation, not by the PRN; None: the PRN rule.
    """
    codes = tuple(codes)
    weights = tuple(float(w) for w in weights)
    if len(codes) != len(weights) or not codes:
        raise ValueError("codes and weights must be non-empty and of equal length")
    kwargs = dict(clock=clock, nav_message=nav_message, include_station=include_station,
                  product_first=product_first, zero_missing=zero_missing, clock_datum=clock_datum)
    if clock != "precise":
        return _code_group_delay(epoch, system, codes, weights, refs=None, **kwargs)
    groups = _datum_groups(epoch, system, clock_datum)
    if len(groups) == 1:
        return _code_group_delay(epoch, system, codes, weights, refs=groups[0][0], **kwargs)
    n = len(epoch)
    total = np.full(n, np.nan)
    source = np.full(n, "none", dtype=object)
    used = [np.full(n, np.nan) for _ in codes]
    missing: list[str] = []
    per_satellite: list[tuple[str, str]] = []
    for refs, rows in groups:
        idx = np.flatnonzero(rows)
        part = _code_group_delay(epoch.iloc[idx], system, codes, weights, refs=refs, **kwargs)
        total[idx] = part.correction_m
        source[idx] = part.source
        for i, b in enumerate(part.biases_m):
            used[i][idx] = b
        missing.extend(m for m in part.missing if m not in missing)
        per_satellite.extend(p for p in part.per_satellite if p not in per_satellite)
    return CodeBiasCorrection(correction_m=total, source=source, missing=tuple(missing), biases_m=tuple(used),
                              per_satellite=tuple(per_satellite))


def _code_group_delay(epoch, system, codes, weights, *, clock, nav_message, include_station, product_first,
                      zero_missing, refs, clock_datum=None) -> CodeBiasCorrection:
    """:func:`code_group_delay_correction` with the precise clock datum codes ``refs`` of all rows
    (``clock_datum``: the session's satellite generations, for the band-7 codes of BDS-3)."""
    n = len(epoch)
    columns = _ProductColumns(epoch)

    def code_biases(**kw):
        return [satellite_code_bias_m(epoch, system, code, clock=clock, nav_message=nav_message,
                                      include_station=include_station, product_first=product_first,
                                      _product=columns, _bcnav_zero=len(codes) == 1, _refs=refs,
                                      clock_datum=clock_datum, **kw)
                for code in codes]

    missing: list[str] = []
    per_satellite: list[tuple[str, str]] = []
    row_labels = None
    product_note = ""
    if clock == "precise":
        # the codes of one combination take their biases from one source: realizations differ per
        # satellite and the difference is amplified by the IF coefficient when sources are mixed
        # (GPS 2024-035, L1/L5 x2.26: broadcast TGD - CAS DSB TGD = -5.3 +- 1.1 ns with 3 ns between
        # GPS III and IIF; GRG and CAS C1W biases 5.5 ns apart between the same blocks)
        per_code = code_biases(broadcast_fallback=False, _isc_anchor=False)
        lacking = [code for code, (_, source, _) in zip(codes, per_code) if source == "none"]
        if not lacking:
            one_source = None
            if len(codes) > 1:
                # one product per combination (NQ-PPP-30): an OSB product and the DSB chain of another one
                per_code, one_source, mix_missing = _one_product_source(
                    epoch, system, codes, per_code, refs, columns, include_station,
                    lambda: code_biases(use_product=False))
                missing.extend(m for m in mix_missing if m not in missing)
            # a satellite the product does not give takes the next source on its rows (PPP-023)
            per_code, row_labels, replaced, broadcast_missing = _fill_product_gaps(
                epoch, system, codes, weights, per_code, lambda: code_biases(use_product=False), columns,
                include_station, refs)
            if one_source:
                for label, rows in one_source.items():
                    row_labels[rows] = label
            per_satellite.extend(replaced)
            missing.extend(m for m in broadcast_missing if m not in missing)
            product_note = ", no other source for it"
        elif len(codes) == 1:
            per_code = code_biases()
        elif _is_precise_datum(system, codes, weights, refs):
            # the combination is the clock datum itself (GPS: up to the ISCs the ICD leaves out, as
            # C1C or C2L for C1W or C2W): its satellite group delay is 0 by definition
            per_code = [(np.zeros(n), "datum", _gps_isc_missing(code) if system == "G" else [])
                        for code in codes]
        elif lacking:
            foreign = [code for code in lacking if refs != tuple(PRECISE_CLOCK_REFERENCE[system])
                       and _product_osb_m(epoch, code, include_station, columns) is not None]
            product = (f"product bias of {', '.join(lacking)} (OSB, or DSB to the precise clock datum "
                       f"{_datum_label(system, refs)}"
                       + "".join(f"; {_foreign_osb_note(code, refs)}" for code in foreign) + ")")
            # the code without a product datum bias tied to an OSB partner of the combination by a DSB of
            # another product of dcb_path, b = OSB(partner) + D(code, partner) (NQ-PPP-30, R4d 3.1)
            tied = _tie_to_osb_partner(epoch, codes, per_code, columns, include_station)
            if tied is not None:
                per_code, partners = tied
                missing.append(f"{product}: {', '.join(lacking)} is taken as the OSB of {partners} plus the DSB "
                               f"between them of the DSB product of dcb_path (one OSB realization for the "
                               f"combination)")
            else:
                broadcast = code_biases(use_product=False)
                anchored = None
                if any(source == "none" for _, source, _ in broadcast):
                    anchored = code_biases(broadcast_fallback=False)
                if all(source != "none" for _, source, _ in broadcast):
                    per_code = broadcast
                    missing.append(f"{product}; the combination takes the broadcast group delays of all its codes")
                elif all(source != "none" for _, source, _ in anchored):
                    # no navigation data: GPS L1 C/A, L2C, L5 codes take the product bias of C1W/C2W
                    per_code = anchored
                else:
                    # correcting one code of the pair alone is worse than none: the biases against the IF
                    # datum are large and correlated (BDS OSB C6I up to 140 ns: applied without the C1P
                    # one, IF(B1C,B3I) moved 75 m on 2024-035)
                    per_code = [(np.full(n, np.nan), "none", []) for _ in codes]
                    missing.append(f"{product} and broadcast group delays of all its codes; no satellite group "
                                   f"delay is applied to the combination")
    else:
        per_code = code_biases()
        if clock == "broadcast":
            per_code = _fill_broadcast_product_gaps(epoch, system, codes, per_code, nav_message, per_satellite,
                                                    columns, include_station, clock_datum=clock_datum)
    per_satellite.extend(_satellites_missing(epoch, codes, per_code, product_note))
    missing.extend(template.format(satellites=name) for template, name in per_satellite)
    partial = np.zeros(n, dtype=bool)
    if zero_missing and len(codes) > 1:
        values = np.vstack([np.broadcast_to(np.asarray(b, dtype=float), (n,)) for b, _, _ in per_code])
        partial = (~np.isfinite(values)).any(axis=0) & np.isfinite(values).any(axis=0)
        if partial.any():
            missing.append("the group delays of the other codes of the combination on the same satellite and "
                           "epoch (taken as 0 with the missing one)")
    total = np.zeros(n)
    sources: list[str] = []
    used: list[np.ndarray] = []
    for (b, source, miss), weight in zip(per_code, weights):
        b = np.broadcast_to(np.asarray(b, dtype=float), (n,))
        if zero_missing:
            b = np.where(np.isfinite(b) & ~partial, b, 0.0)
        total -= weight * b
        used.append(np.array(b, dtype=float))
        sources.append(source)
        missing.extend(m for m in miss if m not in missing)
    label = row_labels if row_labels is not None else _source_label(sources)
    source = np.where(np.isfinite(total), label, "none").astype(object)
    return CodeBiasCorrection(correction_m=total, source=source, missing=tuple(missing), biases_m=tuple(used),
                              per_satellite=tuple(per_satellite))


#: navigation-message group delays (seconds) per system, as named by the navigation readers
BROADCAST_GROUP_DELAY_FIELDS = {
    # GPS: LNAV TGD; CNAV TGD and ISCs on RINEX 4 rows (bug-fix stage 10)
    "G": ("TGD", "ISC_L1CA", "ISC_L2C", "ISC_L5I5", "ISC_L5Q5", "TGD_CNAV"),
    "E": ("BGDe5a", "BGDe5b"),
    "C": ("TGD1", "TGD2", "TGD_B1Cp", "TGD_B2ap", "ISC_B1Cd", "ISC_B2ad", "TGD_B2bI"),
}


def attach_broadcast_group_delays(obs: pd.DataFrame, nav: Optional[pd.DataFrame], system: str, *,
                                  only_missing: bool = False) -> pd.DataFrame:
    """Add to ``obs`` (index levels ``sv``, ``time``) the group delays of the navigation record nearest in time.

    For precise orbits (:func:`code_group_delay_correction` uses them only for codes without a
    product bias) and to complete broadcast rows whose record type lacks a field (``only_missing``:
    e.g. BeiDou B1CB2b takes CNV3 records with TGD_B2bI only; TGD_B1Cp comes from the CNV1/CNV2
    record of the satellite, all B-CNAV clocks referring to B3I). Galileo takes I/NAV records, the
    only ones with both BGD(E1,E5a) and BGD(E1,E5b) (F/NAV when there is none) and marks the rows
    with a numeric ``DataSrc`` (512 I/NAV, 256 F/NAV) when they have none; BeiDou takes TGD1/TGD2 from D1/D2 records only (the RINEX 4 reader fills
    them with 0.0 on B-CNAV records). Returns ``obs`` unchanged when there is nothing to add.
    """
    if nav is None or len(nav) == 0 or obs is None or len(obs) == 0:
        return obs
    fields = [f for f in BROADCAST_GROUP_DELAY_FIELDS[system] if f in nav.columns]
    if only_missing:
        fields = [f for f in fields if f not in obs.columns]
    if not fields or not {"sv", "time"}.issubset(obs.index.names):
        return obs
    nav = nav.reset_index()
    nav["sv"] = nav["sv"].astype(str).str[:3]
    nav_message = None
    if system == "E":
        if "DataSrc" not in nav.columns:
            return obs
        # RINEX 3.04 DataSrc bit 9 (512): clock of E5b,E1 = I/NAV; bit 8 (256): E5a,E1 = F/NAV
        src = pd.to_numeric(nav["DataSrc"], errors="coerce").fillna(0).astype(np.int64)
        inav = nav[(src & 512) > 0]
        nav, nav_message = (inav, "I/NAV") if len(inav) else (nav[(src & 256) > 0], "F/NAV")
        if nav_message == "F/NAV":
            fields = [f for f in fields if f != "BGDe5b"]

    left = obs.index.to_frame(index=False)[["time", "sv"]]
    left["sv"] = left["sv"].astype(str).str[:3]
    left["time"] = pd.to_datetime(left["time"])
    left["_row"] = np.arange(len(left))
    left = left.sort_values("time")
    out = obs.copy()
    for field in fields:
        rows = nav[np.isfinite(pd.to_numeric(nav[field], errors="coerce"))]
        if system == "C" and field in ("TGD1", "TGD2") and "nav_message" in rows.columns:
            rows = rows[~rows["nav_message"].isin(_BDS_CNAV)]
        if rows.empty:
            continue
        right = rows[["time", "sv", field]].copy()
        right["time"] = pd.to_datetime(right["time"])
        right[field] = right[field].astype(float)
        merged = pd.merge_asof(left, right.sort_values("time"), on="time", by="sv", direction="nearest")
        values = np.full(len(out), np.nan)
        values[merged["_row"].to_numpy()] = merged[field].to_numpy(dtype=float)
        out[field] = values
    if nav_message is not None and not only_missing and "DataSrc" not in out.columns:
        # numeric marker of the clock message (RINEX 3.04 DataSrc bits), read by the group-delay helper
        out["DataSrc"] = 512 if nav_message == "I/NAV" else 256
    if system == "G" and {"TGD", "TGD_CNAV"} <= set(out.columns):
        _warn_cnav_tgd(out)
    return out


def _warn_cnav_tgd(frame: pd.DataFrame) -> None:
    """One warning when the CNAV TGD of a satellite differs from its LNAV TGD by more than
    :data:`GPS_CNAV_TGD_TOL_S` (the two messages give the same group delay; the CNAV ISCs are used with the
    CNAV TGD)."""
    d = np.abs(_cnav_column_m(frame, "TGD_CNAV") / CLIGHT - frame["TGD"].to_numpy(dtype=float))
    bad = np.isfinite(d) & (d > GPS_CNAV_TGD_TOL_S)
    if not bad.any():
        return
    sv = _row_satellites(frame)
    names = sorted(set(sv[bad])) if sv is not None else []
    warnings.warn(f"GPS navigation data: CNAV TGD differs from LNAV TGD by up to {np.nanmax(d[bad]) * 1e9:.2f} ns "
                  f"(more than {GPS_CNAV_TGD_TOL_S * 1e9:g} ns) for {', '.join(names) or 'some satellites'}; the "
                  f"CNAV inter-signal corrections are applied with the CNAV TGD", UserWarning, stacklevel=3)


def missing_correction_hint(system: str, missing: Optional[str] = None) -> str:
    """Typical size of the satellite group delays of a system, or of the missing term ``missing``
    (an entry of :attr:`CodeBiasCorrection.missing`) when it has its own (for warnings)."""
    if missing:
        for key, hint in _TYPICAL_TERM.get(system, ()):
            if key in missing:
                return hint
    return _TYPICAL.get(system, "")


# --------------------------------------------------------------------------- RINEX 2 tracking modes
#
# RINEX 2 observation codes carry no tracking mode ("C5": L5 I, Q or I+Q), Bias-SINEX products give RINEX 3
# codes, and the biases of the tracking modes of one band differ per satellite (2024-035: CAS1 GPS C5X - C5Q
# sd 0.53 ns, -1.53..+0.89 ns; GRG Galileo C5Q - C5X sd 0.70 ns), amplified 1.26x in IF(L1,L5) and 11.3x in
# IF(L2,L5). One RINEX 3 code is resolved per (system, RINEX 2 observable), once per observation file, and
# where it came from is reported (rule accepted by the author, bug-fix stage 8):
#
#   fixed  GPS P1 -> C1W, P2 -> C2W, C1 -> C1C; BeiDou C2 -> C2I, C7 -> C7I, C6 -> C6I
#   0      the ``rinex2_tracking`` setting, e.g. {"G": {"C5": "C5X"}} (as gfzrnx -pr3rx2 or BNC's per-station
#          priorities)
#   1      the converter's header COMMENT block, exact after validation, for phases too: gfzrnx
#          "RINEX 3 -> 2 TYPE CONVERSION DETAILS:" ("   G C5Q -> C5"; gfzrnx documentation, RINEX File Format
#          Conversion, "Used Observation Types") and IGS ACC cnvrnx3-rnx2 (O. Colombo) "GPS DATA TYPES COPIED
#          FROM THE ORIGINAL RINEX 3 FILE:" ("@ C1C L1C C2W ..." as 4-character fields, blank fields kept,
#          by position in "# / TYPES OF OBSERV"). An entry whose RINEX 2 code is not in "# / TYPES OF
#          OBSERV" has no effect; one is kept when the band digits match, a P code maps to W/P/Y/D, it
#          agrees with the fixed codes and it is the only code given for that RINEX 2 code. A band the block
#          leaves out follows the tracking family of the codes it gives (phases too; with the setting's
#          codes of the same system)
#   2      station biases of the station in the loaded bias product, when they give exactly one attribute
#          of the band, and their family for the bands they leave out; not used when they mix families
#          (weaker: an analysis centre may itself have read RINEX 2 with some default)
#   3      the receiver type (REC # / TYPE, rcvr_ant.tab names, prefix match): RINEX2_RECEIVER_TRACKING
#   4      the default: pilot tracking (G C2L, C5Q; E C1C, C5Q, C7Q, C8Q), the majority of the IGS network
#          (CAS 2024-035 station list: GPS L5 C5Q 186 vs C5X 118 stations, E1 C1C 186 vs C1X 110) and the
#          Galileo clock reference of COD/GFZ (C1C/C5Q). RTKLIB takes I+Q (X) instead; GNX does not follow it
#
# A receiver tracks one family on all bands (Montenbruck, Hauschild, Steigenberger 2014, NAVIGATION 61(3),
# Table 1; CAS 2024-035: 182 stations all pilot, ~112 all I+Q, none mixed), so families are never mixed
# (where the evidence fits several families, the receiver's is taken when it is one of them). A
# phase observable without a block entry or setting takes the attribute of its band's code (P1 -> L1W). A
# product without the resolved code gives the bias of another attribute of the band (same family first) with
# a warning; without any attribute of the band the bias is missing.

#: RINEX 2 observables whose RINEX 3 code does not depend on the receiver
RINEX2_FIXED_CODES = {
    "G": {"P1": "C1W", "P2": "C2W", "C1": "C1C"},
    "C": {"C2": "C2I", "C7": "C7I", "C6": "C6I"},
}

#: tracking families: system -> RINEX 2 code -> RINEX 3 code
RINEX2_TRACKING_FAMILIES = {
    "pilot": {"G": {"C2": "C2L", "C5": "C5Q"},
              "E": {"C1": "C1C", "C5": "C5Q", "C7": "C7Q", "C8": "C8Q"}},
    "pilot, L2C as C2S": {"G": {"C2": "C2S", "C5": "C5Q"},
                          "E": {"C1": "C1C", "C5": "C5Q", "C7": "C7Q", "C8": "C8Q"}},
    "I+Q": {"G": {"C2": "C2X", "C5": "C5X"},
            "E": {"C1": "C1X", "C5": "C5X", "C7": "C7X", "C8": "C8X"}},
}
RINEX2_DEFAULT_FAMILY = "pilot"

#: receiver types (rcvr_ant.tab name prefixes) -> tracking family (Montenbruck et al. 2014, Table 1; CAS
#: 2024-035 station biases); TPS NET-G3A/G5: L2C documented as C2L, the other bands by family
RINEX2_RECEIVER_TRACKING = (
    (("SEPT POLARX4", "SEPT POLARX5", "SEPT ASTERX3", "SEPT ASTERX4", "NOV OEM6"), "pilot"),
    (("LEICA GR10", "LEICA GR25", "LEICA GR30", "LEICA GR50", "STONEX SC2200"), "pilot, L2C as C2S"),
    (("TRIMBLE NETR9", "TRIMBLE ALLOY", "JAVAD TRE_3", "JAVAD TRE_G3TH", "JAVAD TR_G3TH"), "I+Q"),
    (("TPS NET-G3A", "TPS NET-G5"), "pilot"),
)

#: RINEX 2 code -> the RINEX 3 codes of its band, pilot family first (station-bias evidence, product fallback)
RINEX2_BIAS_CODES = {
    "G": {"P1": ("C1W",), "C1": ("C1C",), "P2": ("C2W",), "C2": ("C2L", "C2S", "C2X"), "C5": ("C5Q", "C5X", "C5I")},
    "E": {"C1": ("C1C", "C1X"), "C5": ("C5Q", "C5X", "C5I"), "C7": ("C7Q", "C7X", "C7I"),
          "C8": ("C8Q", "C8X", "C8I")},
    "C": {"C2": ("C2I",), "C7": ("C7I",), "C6": ("C6I",)},
}

_R2_OBSERVABLE = re.compile(r"^[CLPDS][0-9]$")
_R3_OBSERVABLE = re.compile(r"^[CLDS][0-9][A-Z]$")
_P_CODE_ATTRS = "WPYD"


def rinex2_bias_candidates(system: str, source: str, phase: bool = False) -> tuple[str, ...]:
    """RINEX 3 codes of the band of a RINEX 2 code, pilot family first (``RINEX2_BIAS_CODES``).

    ``source``: the RINEX 2 code of the band (``P1``, ``C1``, ``C5`` ...); ``phase``: the phase codes
    with the same attributes (``L1W`` for P1). Empty for a code without a RINEX 3 counterpart. The code a
    file's observable takes is chosen by ``resolve_rinex2_tracking``.
    """
    codes = RINEX2_BIAS_CODES.get(system, {}).get(str(source).upper(), ())
    return tuple("L" + code[1:] for code in codes) if phase else codes


def rinex2_code_problem(system: str, rinex2: object, rinex3: object) -> Optional[str]:
    """Why ``rinex3`` cannot be the RINEX 3 code of the RINEX 2 observable ``rinex2`` (None when it can)."""
    if not isinstance(rinex2, str) or not _R2_OBSERVABLE.match(rinex2):
        return f"{rinex2!r} is not a RINEX 2 observable (C1, P2, L5, ...)"
    if not isinstance(rinex3, str) or not _R3_OBSERVABLE.match(rinex3):
        return f"{rinex3!r} is not a RINEX 3 observable (C5Q, L2W, ...)"
    if rinex2[1] != rinex3[1]:
        return f"{system} {rinex3} is band {rinex3[1]}, {rinex2} band {rinex2[1]}"
    kind = "C" if rinex2[0] == "P" else rinex2[0]
    if rinex3[0] != kind:
        return f"{system} {rinex3} is not a{'n' if kind in 'LS' else ''} {kind} observable like {rinex2}"
    if rinex2[0] == "P" and rinex3[2] not in _P_CODE_ATTRS:
        return f"{system} {rinex2} is a P(Y) code: its RINEX 3 code has attribute W, P, Y or D (got {rinex3})"
    return None


def rinex2_setting_problem(system: str, rinex2: object, rinex3: object) -> Optional[str]:
    """Why ``{system: {rinex2: rinex3}}`` cannot be a ``rinex2_tracking`` entry (None when it can): besides
    ``rinex2_code_problem``, the observable takes product biases (C, P, L) and ``rinex3`` is a tracking mode
    of its band (``RINEX2_BIAS_CODES``; a phase: the attribute of a code of its band)."""
    table = RINEX2_BIAS_CODES.get(system, {})
    if isinstance(rinex2, str) and _R2_OBSERVABLE.match(rinex2) and rinex2[0] in "CP" and rinex2 not in table:
        return f"{system} has no RINEX 2 observable {rinex2} with product biases ({', '.join(sorted(table))})"
    problem = rinex2_code_problem(system, rinex2, rinex3)
    if problem:
        return problem
    if rinex2[0] in "DS":
        return f"{rinex2} takes no product bias (only code and phase observables C, P, L do)"
    if rinex2[0] == "C" and rinex3 not in table[rinex2]:
        hint = f"; the P(Y) code is P{rinex2[1]}" if f"P{rinex2[1]}" in table else ""
        return f"{system} {rinex2} is one of {'/'.join(table[rinex2])} (got {rinex3}){hint}"
    if rinex2[0] == "L":
        band = [code for codes in table.values() for code in codes if code[1] == rinex2[1]]
        attrs = {code[2] for code in band} | (set(_P_CODE_ATTRS) if f"P{rinex2[1]}" in table else set())
        if not band:
            return f"{system} has no band {rinex2[1]} observable with product biases"
        if rinex3[2] not in attrs:
            return f"{system} {rinex2} has attribute {'/'.join(sorted(attrs))} (got {rinex3})"
    return None


def _code_families(system: str, rinex2: str, rinex3: str) -> set:
    """Tracking families in which the RINEX 2 code ``rinex2`` of ``system`` is ``rinex3``."""
    return {family for family, table in RINEX2_TRACKING_FAMILIES.items()
            if table.get(system, {}).get(rinex2) == rinex3}


def receiver_tracking_family(receiver: Optional[str]) -> Optional[str]:
    """Tracking family of a receiver type (REC # / TYPE), None when RINEX2_RECEIVER_TRACKING has none."""
    name = " ".join(str(receiver or "").upper().split())
    if not name:
        return None
    for prefixes, family in RINEX2_RECEIVER_TRACKING:
        if name.startswith(prefixes):
            return family
    return None


def read_rinex_header_lines(path, max_lines: int = 20000) -> list[str]:
    """Header lines of a RINEX observation file up to END OF HEADER (plain, gzip, Unix compress, bzip2, zip or
    Hatanaka: the header of a CRINEX file is plain text after its two CRINEX lines). Empty when the file cannot
    be read. Opened through :func:`gnx_py.compression.head_lines` (no georinex)."""
    if not path:
        return []
    from pathlib import Path

    from ..io.compression import head_lines

    try:
        head = head_lines(Path(str(path)).expanduser(), "observation file", (max_lines + 1) * 82)
    except (OSError, ValueError):
        return []
    lines: list[str] = []
    for i, line in enumerate(head):
        lines.append(line.rstrip("\r\n"))
        if "END OF HEADER" in line[60:] or i >= max_lines:
            break
    return lines


@dataclass(frozen=True)
class Rinex2HeaderTracking:
    """What the header of a RINEX 2 observation file says about its tracking modes.

    ``entries``: {system: {RINEX 2 observable: RINEX 3 observable}} of the converter's COMMENT block, after
    validation; ``converter``: its source ("gfzrnx-2.1.12 conversion block", None without a block);
    ``receiver``: REC # / TYPE; ``issues``: block entries left out, with the reason; ``types``: the
    "# / TYPES OF OBSERV" list.
    """
    entries: dict
    converter: Optional[str] = None
    receiver: Optional[str] = None
    issues: tuple = ()
    types: tuple = ()


_GFZRNX_MARK = "RINEX 3 -> 2 TYPE CONVERSION DETAILS"
_GFZRNX_ENTRY = re.compile(r"^\s*([GRECJIS])\s+([CLDS][0-9][A-Z])\s*->\s*([CLDSP][0-9])\s*$")
_GFZRNX_VERSION = re.compile(r"(gfzrnx-\S+)\s+FILE CONVERSION", re.IGNORECASE)
_CNVRNX3_MARK = re.compile(r"^\s*(GPS|GLO|GAL|BDS|QZS|SBS|IRN)\s+DATA TYPES COPIED FROM THE ORIGINAL RINEX 3 FILE")
_CNVRNX3_SYSTEMS = {"GPS": "G", "GLO": "R", "GAL": "E", "BDS": "C", "QZS": "J", "SBS": "S", "IRN": "I"}


def _header_records(lines: Optional[Sequence[str]], header: Optional[dict]):
    """(comment payloads in order, receiver type, observation types) from raw header lines, else from the
    georinex header dict (its COMMENT is the payloads joined by one blank)."""
    comments: list[str] = []
    receiver, types = None, []
    if lines:
        for line in lines:
            label, payload = line[60:].strip(), line[:60]
            if label == "COMMENT":
                comments.append(payload)
                continue
            if comments and comments[-1].strip():
                comments.append("")  # another record ends a block of COMMENT lines
            if label == "REC # / TYPE / VERS":
                receiver = payload[20:40].strip() or None
            elif label == "# / TYPES OF OBSERV":
                types.extend(payload[6:].split())
            elif label == "END OF HEADER":
                break
        return comments, receiver, types
    header = header or {}
    text = header.get("COMMENT") or ""
    if isinstance(text, (list, tuple)):
        comments = [str(c)[:60] for c in text]
    elif text:
        text = str(text)
        if len(text) >= 60 and all(text[i] == " " for i in range(60, len(text), 61)):
            comments = [text[i:i + 60] for i in range(0, len(text), 61)]
        else:
            # unknown joining: the gfzrnx entries alone, up to a blank COMMENT (a long run of blanks)
            start = text.find(_GFZRNX_MARK)
            if start >= 0:
                block = re.split(r" {100,}", text[start:])[0]
                comments = [_GFZRNX_MARK + ":"] + [
                    f"   {m.group(1)} {m.group(2)} -> {m.group(3)}"
                    for m in re.finditer(r"([GRECJIS])\s+([CLDS][0-9][A-Z])\s*->\s*([CLDSP][0-9])\b", block)]
                comments += [m.group(0) for m in _GFZRNX_VERSION.finditer(text)]
    rec = header.get("REC # / TYPE / VERS")
    if isinstance(rec, str):
        receiver = rec[20:40].strip() or None
    fields = header.get("fields")
    if isinstance(fields, (list, tuple)):
        types = [str(f) for f in fields]
    return comments, receiver, types


def rinex2_header_tracking(lines: Optional[Sequence[str]] = None, header: Optional[dict] = None
                           ) -> Rinex2HeaderTracking:
    """Tracking-mode information of a RINEX 2 observation header (``Rinex2HeaderTracking``).

    ``lines``: the raw header lines (``read_rinex_header_lines``); ``header``: the georinex header dict,
    used when no lines are given. See the RINEX 2 tracking-mode rule above.
    """
    comments, receiver, types = _header_records(lines, header)
    raw: list[tuple[str, str, str]] = []  # (system, RINEX 3 code, RINEX 2 code)
    issues: list[str] = []
    converter = None
    version = next((m.group(1) for c in comments for m in [_GFZRNX_VERSION.search(c)] if m), None)
    i = 0
    while i < len(comments):
        text = comments[i]
        if _GFZRNX_MARK in text:
            converter = f"{version or 'gfzrnx'} conversion block"
            i += 1
            while i < len(comments) and comments[i].strip():
                entry = comments[i].strip()
                match = _GFZRNX_ENTRY.match(comments[i])
                if match:
                    raw.append((match.group(1), match.group(2), match.group(3)))
                elif set(entry) != {"-"}:
                    issues.append(f"unrecognised conversion-block line {entry!r}")
                i += 1
            continue
        match = _CNVRNX3_MARK.match(text)
        if match:
            converter = converter or "cnvrnx3-rnx2 comment block"
            system = _CNVRNX3_SYSTEMS[match.group(1)]
            codes: list[str] = []
            fixed_layout = True
            i += 1
            while i < len(comments) and comments[i].lstrip().startswith("@"):
                # written as ("@ ",14a4): 14 fields of 4 characters, blank where a RINEX 2 type has no
                # RINEX 3 source (e.g. C2 without L2C), so the positions are the fixed fields
                line = comments[i][comments[i].index("@"):]
                fields = [line[2 + 4 * k:6 + 4 * k].strip() for k in range(14)]
                if any(code and not _R3_OBSERVABLE.match(code) for code in fields):
                    fields, fixed_layout = line[1:].split(), False  # not the fixed layout: codes in order
                codes.extend(fields)
                i += 1
            given_codes = [code for code in codes if code]
            if not types or any(codes[len(types):]) or (not fixed_layout and len(codes) != len(types)):
                issues.append(f"{system} cnvrnx3-rnx2 list of {len(given_codes)} codes does not match the "
                              f"{len(types)} RINEX 2 observation types (positions unknown)")
                continue
            pairs = [(system, r3, r2) for r3, r2 in zip(codes, types) if r3 and _R2_OBSERVABLE.match(r2)]
            bad = [f"{r3} -> {r2}: {rinex2_code_problem(s, r2, r3)}" for s, r3, r2 in pairs
                   if rinex2_code_problem(s, r2, r3)]
            if bad:
                issues.append(f"{system} cnvrnx3-rnx2 list misaligned with # / TYPES OF OBSERV ({'; '.join(bad[:3])})")
                continue
            raw.extend(pairs)
            continue
        i += 1

    entries: dict[str, dict[str, str]] = {}
    given: dict[tuple[str, str], set] = {}
    for system, r3, r2 in raw:
        if types and r2 not in types:
            continue  # an observable the file does not have (e.g. filtered after conversion): no effect
        problem = rinex2_code_problem(system, r2, r3)
        fixed = RINEX2_FIXED_CODES.get(system, {}).get(r2)
        if problem is None and fixed is not None and r3[0] == "C" and r3 != fixed:
            problem = f"{system} {r2} is always taken as {fixed} (fixed; set rinex2_tracking to change it)"
        if problem:
            issues.append(f"{system} {r3} -> {r2}: {problem}")
            continue
        given.setdefault((system, r2), set()).add(r3)
    for (system, r2), codes in sorted(given.items()):
        if len(codes) > 1:
            issues.append(f"{system} {r2}: conflicting entries {sorted(codes)}")
            continue
        entries.setdefault(system, {})[r2] = next(iter(codes))
    return Rinex2HeaderTracking(entries=entries, converter=converter if raw or issues else None,
                                receiver=receiver, issues=tuple(issues), types=tuple(types))


def _family_evidence(given: dict, prefer: Optional[str] = None) -> tuple[Optional[str], list[str]]:
    """Common tracking family of the codes ``given`` ({system: {RINEX 2: RINEX 3}}; a phase, Doppler or
    SNR entry counts as the code of its band and attribute), with the codes that show it; None when they
    show none (``shown`` empty) or several families (``shown`` not empty: mixed, nothing is inferred).
    Where several families fit, ``prefer`` (the receiver's family) when it is one of them."""
    families: Optional[set] = None
    codes: list[str] = []
    others: list[str] = []
    for system in sorted(given):
        for r2, r3 in sorted(given[system].items()):
            if r2[:1] in "LDS" and r3[:1] == r2[:1]:
                found = _code_families(system, "C" + r2[1:], "C" + r3[1:])
            else:
                found = _code_families(system, r2, r3)
            if not found:
                continue
            families = found if families is None else families & found
            (others if r2[:1] in "LDS" else codes).append(f"{system} {r3}")
    shown = codes or others  # the codes are named; phases (Doppler, SNR) only when no code shows a family
    if not families:
        return None, shown
    if prefer in families:
        return prefer, shown
    return next(f for f in RINEX2_TRACKING_FAMILIES if f in families), shown


def resolve_rinex2_tracking(system: str, observables: Iterable[str], info: Optional[Rinex2HeaderTracking] = None,
                            override: Optional[dict] = None, station_codes: Iterable[str] = (),
                            station_source: Optional[str] = None) -> dict[str, tuple[str, str]]:
    """RINEX 3 code and its source for each RINEX 2 observable (``P1``, ``C5``, ``L2`` ...) of ``system``.

    The rule above: the setting ``override`` ({system: {RINEX 2: RINEX 3}}), the fixed codes, the converter
    block of ``info`` (``rinex2_header_tracking``) and the family of its codes (with the codes the setting
    gives for ``system``), the station biases ``station_codes`` (RINEX 3 codes of the station's biases in
    the product named ``station_source``) and their family, the receiver type of ``info``, the default
    pilot family. Families are never mixed: where several fit the evidence, the receiver's family is taken
    when it is one of them; station biases of mixed families are not used. A phase (or Doppler, SNR)
    observable resolves only from the setting or the block (else it is left out: it takes the attribute
    of its band's code).
    """
    info = info or Rinex2HeaderTracking(entries={})
    override = {str(s): dict(v) for s, v in (override or {}).items()}
    own = override.get(system, {})
    block = info.entries
    station_codes = set(station_codes)
    out: dict[str, tuple[str, str]] = {}
    receiver_family = receiver_tracking_family(info.receiver)
    # the block's family holds for the receiver (all systems); the setting's only for its own system
    evidence = {s: dict(codes) for s, codes in block.items()}
    if own:
        evidence[system] = {**evidence.get(system, {}), **own}
    block_family, shown = _family_evidence(evidence, prefer=receiver_family)
    shown_in = " and ".join(([f"the {info.converter or 'conversion block'}"] if block else [])
                            + (["rinex2_tracking"] if own else []))
    # station biases: a band with exactly one attribute; their family fills the bands they leave out
    station: dict[str, str] = {}
    for r2, codes in RINEX2_BIAS_CODES.get(system, {}).items():
        present = [code for code in codes if code in station_codes]
        if len(present) == 1 and r2 not in RINEX2_FIXED_CODES.get(system, {}):
            station[r2] = present[0]
    station_family, station_shown = _family_evidence({system: station}, prefer=receiver_family)
    if station_family is None and station_shown:
        station = {}  # mixed families: the analysis centre's own RINEX 2 defaults, not the receiver's
    station_note = f"station biases in {station_source or 'the bias product'}"
    for r2 in observables:
        r2 = str(r2)
        if r2 in own:
            out[r2] = (own[r2], "rinex2_tracking setting")
            continue
        fixed = RINEX2_FIXED_CODES.get(system, {}).get(r2)
        if fixed is not None:
            out[r2] = (fixed, "fixed: RINEX 2 P1/P2 are P(Y), C1 is C/A")
            continue
        if r2 in block.get(system, {}):
            out[r2] = (block[system][r2], info.converter or "conversion block")
            continue
        if not r2.startswith("C") or r2 not in RINEX2_BIAS_CODES.get(system, {}):
            continue
        family_code = RINEX2_TRACKING_FAMILIES.get(block_family or "", {}).get(system, {}).get(r2)
        if family_code is not None:
            out[r2] = (family_code, f"tracking family ({block_family}) of {', '.join(shown)} in {shown_in}")
            continue
        if r2 in station:
            out[r2] = (station[r2], station_note)
            continue
        code = RINEX2_TRACKING_FAMILIES.get(station_family or "", {}).get(system, {}).get(r2)
        if station and code is not None:
            out[r2] = (code, f"tracking family ({station_family}) of {', '.join(station_shown)} in {station_note}")
            continue
        code = RINEX2_TRACKING_FAMILIES.get(receiver_family or "", {}).get(system, {}).get(r2)
        if code is not None:
            out[r2] = (code, f"receiver {info.receiver} ({receiver_family} tracking)")
            continue
        code = RINEX2_TRACKING_FAMILIES[RINEX2_DEFAULT_FAMILY].get(system, {}).get(r2)
        if code is not None:
            out[r2] = (code, f"default {RINEX2_DEFAULT_FAMILY} tracking")
    return out


def rinex2_fallback_codes(system: str, rinex2: str, resolved: str) -> tuple[str, ...]:
    """Other RINEX 3 codes of the band of ``resolved`` (the code of the RINEX 2 observable ``rinex2``) a
    product may give instead, same tracking family first (codes and phases alike)."""
    table = RINEX2_BIAS_CODES.get(system, {})
    if resolved[0] == "C":
        # a code keeps to the tracking modes of its RINEX 2 code (C2: L2C, never the P(Y) code C2W)
        code_r2, band = rinex2, list(table.get(rinex2, ()))
    else:
        # the phases of one band are aligned (RINEX 3 phase alignment): any attribute of the band
        code_r2 = "C" + rinex2[1:]
        band = [resolved[0] + code[1:] for codes in table.values() for code in codes if code[1] == resolved[1]]
    families = _code_families(system, code_r2, "C" + resolved[1:])
    same = [code for code in band if _code_families(system, code_r2, "C" + code[1:]) & families]
    ordered = same + [code for code in band if code not in same]
    return tuple(dict.fromkeys(code for code in ordered if code != resolved))


_STATION_ID9 = re.compile(r"^[A-Z0-9]{4}[0-9]{2}[A-Z]{3}$")


def match_station(name, candidates) -> list:
    """Candidates naming the station ``name``, best first (case-insensitive).

    Station ids come as the 4-character site code (``BRUX``: Bias-SINEX STATION of most products, RINEX 2),
    the 9-character id (``BRUX00BEL``: site, monument and receiver number, country; RINEX 3 file names,
    Bias-SINEX may give it as well) or the site code with the DOMES number (IONEX ``BRUX 13101M010``).
    An exact match comes first, then the candidates with the same 4-character site code; two different
    9-character ids (``BRUX00BEL``/``BRUX01BEL``, another monument or receiver at the site) do not match.
    """
    if name is None:
        return []
    key = str(name).strip().upper().replace(" ", "")
    if not key:
        return []
    exact, site = [], []
    for candidate in candidates:
        other = str(candidate).strip().upper().replace(" ", "")
        if not other:
            continue
        if other == key:
            exact.append(candidate)
        elif (len(key) >= 4 and other[:4] == key[:4]
              and not (_STATION_ID9.match(key) and _STATION_ID9.match(other))):
            site.append(candidate)
    return exact + site
