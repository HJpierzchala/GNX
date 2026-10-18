"""Configuration and high-level control for GNSS orbit/clock/DCB processing.

This module contains:
- SISConfig: a configuration container for file paths, modes, and processing flags.
- SISSession: an orchestrator that runs the end-to-end pipeline (file classification,
  orbit handling, eclipse detection, antenna PCO/PCV application, DCB handling,
  and merging of outputs). SISController is its deprecated former name (see below).

Conventions:
- Time is UTC unless stated otherwise.
- Units follow SI where applicable (meters, seconds), unless specified.
- Array-like inputs follow NumPy broadcasting semantics when relevant.

Notes:
- External resources (SP3, ATX, DCB/OSB) are provided via SISConfig fields.
- Session methods document behavior and expected outputs without changing any logic.
"""

import gzip
import os
import re
import warnings
from dataclasses import dataclass  # noqa: F401 - re-exported by gnx_py.orbits (compatibility)
import logging as _logging
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd

from ..io import GNSSDataProcessor2, read_sp3
from ..core.time import arange_datetime, check_time_limit, interval_epochs
from ..core.tools import DDPreprocessing, DefaultConfig
from ..core.coordinates import BrdcGenerator, SP3InterpolatorOptimized
from ..core.session_errors import guarded_session_run
from ..core import _notes
from ..core.validation import did_you_mean
from ..core.configuration import MISSING_SAT_ANTENNA_POLICIES, SATELLITE_ATTITUDE_MODES, _check_attitude_laws
from ..models.tides import get_sun_ecef
from ..models.antex import build_antenna_model
from ..core.gnss import (SUPPORTED_MODES_BY_SYSTEM, bds_orbit_type, gamma, ionosphere_free_coefficients, is_bds_geo,
                    mode_ionosphere_free_coefficients, mode_signals, signal_spec)


_logger = _logging.getLogger(__name__)


# Legacy default argument set of SISSession.run() (0.98^2, 1, 0.141^2, -2*0.98).
# Passing this set (to rounding, with a FutureWarning) or leaving the coefficients as None
# selects the per-system literature weights below; any other value is used as given.
DEFAULT_SISRE_COEFFICIENTS = {
    "kr": 0.9604,
    "kb": 1.0,
    "kac": 0.019881,
    "krb": -1.960,
}

# SISRE weight factors (w_R, w_AC^2) after Montenbruck et al. (2015, GPS Solut 19:321-333,
# Tab. 3) and Montenbruck et al. (2018, Adv Space Res 61:3020-3038). They follow from the
# satellite altitude: the line-of-sight projection averaged over the visible Earth.
#   SISRE^2 = (w_R*R - c*dT)^2 + w_AC^2*(A^2 + C^2)
SISRE_WEIGHTS = {
    "G": (0.98, 1.0 / 49.0),
    "R": (0.98, 1.0 / 45.0),
    "E": (0.98, 1.0 / 61.0),
    "J": (0.99, 1.0 / 126.0),   # QZSS IGSO/QZO and GEO
}

# BeiDou per orbit type: (w_R, 1/w_AC^2). GEO (0.99, 126) is deliberately absent: the
# SIS path excludes BDS GEO satellites.
BDS_SISRE_ALPHA_BETA = {
    "MEO": (0.98, 54.0),
    "IGSO": (0.99, 126.0),
}

BDS_LEGACY_BROADCAST_TGD_SIGNALS = {"B1I", "B2I", "B3I"}
BDS_MODERN_BROADCAST_SIGNALS = {"B1C", "B2a", "B2b"}
BDS_LEGACY_IF_PRECISE_CLOCK_MODE = "B1IB3I"
#: Tracking attributes of the bias-product observable of BeiDou B1C and B2a in the orbit comparison, best first
#: (since v1.0.0): the pilot P first, because the broadcast group delays TGD_B1Cp and
#: TGD_B2ap the precise side is compared with refer to the pilot (BDS-SIS-ICD-B1C 7.6, B2a 7.6), then the
#: data+pilot X, then the data D (its bias differs by ISC_B1Cd/ISC_B2ad). Before, the receiver priority of
#: `gnx_py.gnss` (X first) took C1X/C5X from a product that gives both (BRUX 2024-035 CAS1: 0.026 m median, 0.29 m
#: max on the precise TGD). The observables the observation processing picks are unchanged.
BDS_PRODUCT_OBSERVABLE_ORDER = {"B1C": ("P", "X", "D"), "B2a": ("P", "X", "D")}

# Galileo broadcast orbit/clock reference APC per mode (OS SIS ICD 5.1.1: one ephemeris for the
# apparent CoP common to all frequencies; no E5a/E5b clock, E5aE5b uses the I/NAV E1/E5b clock)
GAL_BROADCAST_APC_REFERENCE = {"E5aE5b": "E1E5b"}
# Galileo modes compared with the I/NAV clock (IF(E1,E5b) datum; stage-3 message selection),
# the precise IGS/MGEX clocks referring to IF(E1,E5a) (C1C/C5Q)
GAL_INAV_CLOCK_MODES = ("E1", "E5b", "E1E5b", "E5aE5b")


def _gal_apc_label(mode: str) -> str:
    """Phase-centre label of a Galileo mode: ``E1E5b`` -> ``IF(E1,E5b)``, ``E1`` -> ``E1``."""
    signals = mode_signals(mode)
    return f"IF({','.join(signals)})" if len(signals) > 1 else signals[0]


BDS_CLOCK_CONVENTION_NOTE = (
    "BDS precise clocks are tied to an IF clock reference chosen by the analysis "
    "centre: IF(B1I,B3I) (C2I/C6I) in the MGEX products of 2024, IF(B1C,B2a) for the "
    "BDS-3 clocks of COD MGEX in 2026 (the bias product declares it, "
    "SATELLITE_CLOCK_REFERENCE_OBSERVABLES; mode B1IB3I converts them with the "
    "B-CNAV group delays of a RINEX 4 navigation file). Broadcast TGD is a user "
    "correction for a selected signal or IF combination, not a generic DCB/OSB "
    "replacement. SIS clock errors are directly comparable only after both clocks "
    "are expressed in the same datum."
)


# --------------------------------------------------------------------------- settings checks
# Systems of the comparison: R and J only SP3 x SP3 (no broadcast branch, PCO or bias
# conversion for them); the broadcast, satellite-PCO and bias branches cover G, E and C.
SIS_SYSTEMS = ("G", "E", "C", "R", "J")
SIS_FULL_SYSTEMS = ("G", "E", "C")
# Signal modes of the satellite-PCO step (DDPreprocessing.sat_fixed_system); BeiDou: every mode.
SIS_PCO_MODES = {
    "G": ("L1", "L1L2", "L1L5", "L5"),
    "E": ("E1", "E5a", "E1E5a", "E1E5b", "E5aE5b"),
}
SIS_INPUT_TYPES = ("RINEX_NAV", "SP3")
#: Signal modes of the bias-product comparison (`compare_dcb=True`) per system, documented in `SISConfig`
#: (the modes the datum algebra defines but this version does not enable -- Galileo
#: E1E5b/E5aE5b, BeiDou single-signal DSB, BeiDou pairs reachable only through DSB chains -- keep a readable
#: ValueError). BeiDou: two-signal modes with OSB or DSB (the mode's own pair), single-signal modes with OSB only.
SIS_COMPARE_DCB_MODES = {
    "G": ("L1", "L2", "L5", "L1L2", "L1L5", "L2L5"),
    "E": ("E1", "E5b", "E5a", "E1E5a"),
    "C": tuple(SUPPORTED_MODES_BY_SYSTEM["C"]),
}
# pandas groupby reductions usable as the per-epoch clock datum (`groupby.transform` and `.agg`
# both accept them and return a number); a callable is accepted as well
SIS_CLOCK_BIAS_FUNCTIONS = ("mean", "median", "min", "max", "first", "last", "sum", "prod", "std", "var",
                            "sem", "skew", "quantile")
_SYSTEM_ALIASES = {"GPS": "G", "GALILEO": "E", "GAL": "E", "BEIDOU": "C", "BDS": "C", "GLONASS": "R",
                   "GLO": "R", "QZSS": "J"}

_SP3_RE = re.compile(r'\.sp3(?:\.gz)?$', re.IGNORECASE)
# RINEX 3:  …_<CONST><TYPE>.(rnx|crx)[.gz], e.g. …_MN.rnx (mixed navigation), …_GO.crx.gz
_RINEX3_RE = re.compile(r'_(?P<const>[A-Z])(?P<dtype>[ONM])\.(?:rnx|crx)(?:\.gz)?$', re.IGNORECASE)
# RINEX 2 navigation: *.YYn[.gz] (e.g. brdc2400.24n.gz)
_RINEX2_NAV_RE = re.compile(r'\.\d{2}n(?:\.gz)?$', re.IGNORECASE)
# RINEX 2 observation: *.rnx|*.crx[.gz]
_RINEX2_OBS_RE = re.compile(r'\.(rnx|crx)(?:\.gz)?$', re.IGNORECASE)


def classify_orbit_file(path: str | Path) -> str:
    """File type of an orbit input from its name: ``"SP3"``, ``"RINEX_NAV"``, ``"RINEX_OBS"``,
    ``"RINEX_MET"``, ``"RINEX_UNKNOWN"`` or ``"UNKNOWN"`` (only SP3 and RINEX_NAV are compared)."""
    name = Path(path).name  # RINEX 3 type letters are matched case-sensitively
    if _SP3_RE.search(name.lower()):
        return "SP3"
    m3 = _RINEX3_RE.search(name)
    if m3:
        return {"N": "RINEX_NAV", "O": "RINEX_OBS", "M": "RINEX_MET"}.get(m3.group('dtype').upper(),
                                                                         "RINEX_UNKNOWN")
    if _RINEX2_NAV_RE.search(name.lower()):
        return "RINEX_NAV"
    if _RINEX2_OBS_RE.search(name.lower()):
        return "RINEX_OBS"
    return "UNKNOWN"


def _did_you_mean(value, choices) -> str:
    """``"; did you mean 'X'?"`` for a case variant or a close typo of one of ``choices``
    (the shared `gnx_py.validation.did_you_mean`, so every configuration hints alike)."""
    hit = did_you_mean(value, choices)
    return f"; did you mean {hit!r}?" if hit is not None else ""


def _check_choice(name: str, value, choices, allow_none: bool = False, context: str = "", hint: bool = True):
    """``value`` if it is one of ``choices`` (or None when allowed), else a ValueError
    (with a did-you-mean hint for a case variant or typo unless ``hint=False``)."""
    if (value is None and allow_none) or (isinstance(value, str) and value in choices):
        return value
    valid = f"{list(choices)}{' or None' if allow_none else ''}"
    tip = _did_you_mean(value, choices) if hint else ""
    raise ValueError(f"{name} must be one of {valid}{context} (got {value!r}){tip}")


def _check_flag(name: str, value) -> bool:
    """A switch: True/False (numpy bools and the integers 0/1 accepted); a string is rejected."""
    if isinstance(value, (bool, np.bool_)) or (isinstance(value, (int, np.integer)) and value in (0, 1)):
        return bool(value)
    raise ValueError(f"{name} must be True or False (got {value!r})")


def _is_real(value) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))


def _check_input_path(name: str, value, required: bool = False):
    """One existing file (a one-element list/tuple is unwrapped); None when not given."""
    if value is None:
        if required:
            raise ValueError(f"{name} is required (got None)")
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a single file path (got {len(value)} paths: {list(value)!r})")
        value = value[0]
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a file path (str or Path), got {type(value).__name__}: {value!r}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{name}: file not found: {value}")
    if path.is_dir():
        raise ValueError(f"{name}: expected a file, got a directory: {value}")
    return value


def _check_bias_paths(name: str, value):
    """One existing bias product, or a list of them (e.g. the products of two days for a comparison
    across midnight); a one-element list is unwrapped; None when not given."""
    if isinstance(value, (list, tuple)) and len(value) > 1:
        return [_check_input_path(f"{name}[{k}]", path) for k, path in enumerate(value)]
    if isinstance(value, (list, tuple)) and not value:
        raise ValueError(f"{name} must be a file path or a list of file paths (got an empty {type(value).__name__})")
    return _check_input_path(name, value)


def _check_orbit_path(name: str, value):
    """An existing orbit input of a compared type (RINEX navigation or SP3)."""
    value = _check_input_path(name, value, required=True)
    kind = classify_orbit_file(value)
    if kind not in SIS_INPUT_TYPES:
        raise ValueError(
            f"{name}: unsupported file type {kind} ({value}); supported are RINEX navigation files "
            "(RINEX 3/4 '*_xN.rnx', RINEX 2 '*.YYn', optionally .gz) and SP3 files ('*.sp3', '*.sp3.gz')")
    return value


def _check_neighbour_sp3(name: str, value):
    """An existing adjacent-day SP3 file (``prev_sp3``, ``next_sp3``, ``*_0``), or None."""
    value = _check_input_path(name, value)
    if value is not None:
        kind = classify_orbit_file(value)
        if kind != "SP3":
            raise ValueError(f"{name}: expected an SP3 file ('*.sp3', '*.sp3.gz') (got {kind}: {value})")
    return value


class SISConfig:
    """
    Configuration container for signal-in-space orbit and clock comparison.

    Status:
        Active public API. `SISConfig` is the recommended entry point for
        programmatic SIS/SISRE comparisons. It is intentionally lightweight and
        mostly stores paths and switches consumed by `SISSession`.

    Purpose:
        The object describes one comparison run: two orbit/clock inputs, the
        GNSS system, signal/frequency mode, optional bias products, time window,
        and modeling switches such as eclipse masking and satellite PCO.

    Supported modes:
        GPS (`system="G"`) and Galileo (`system="E"`) support broadcast-vs-SP3
        and SP3-vs-SP3 style comparisons when the requested signal mode matches
        the available navigation and bias products. BeiDou (`system="C"`) is
        supported for legacy `B1IB3I` MEO/IGSO validation and guarded modern
        BDS experiments. BeiDou GEO is deliberately excluded from the SIS/SISRE
        comparison path until GEO-specific validation is added.

    Main parameters:
        orb_path_0, orb_path_1:
            Input orbit products. RINEX navigation files are treated as
            broadcast ephemerides; `.sp3`/`.sp3.gz` files are treated as precise
            orbit/clock products.
        interval:
            Target processing interval in seconds. The session converts this
            to minutes for epoch generation and interpolation.
        system:
            GNSS constellation code: `"G"` for GPS, `"E"` for Galileo, `"C"`
            for BeiDou (every input pair, satellite PCO and bias comparison);
            `"R"` (GLONASS) and `"J"` (QZSS) only for SP3 x SP3 comparisons with
            `apply_satellite_pco=False` and `compare_dcb=False`.
        gps_mode, gal_mode, bds_mode:
            Signal or ionosphere-free mode used by broadcast generation and
            group-delay conversion. Recommended BDS validation mode is
            `bds_mode="B1IB3I"`. Single-signal BDS modes such as `"B1I"` and
            modern BDS modes such as `"B1C"`, `"B2a"` or `"B2b"` are reported
            with warning/requires-validation clock-convention status unless the
            clock datum and OSB/DSB stack are explicitly consistent.
        dcb_path_0, dcb_path_1, compare_dcb:
            Optional DCB/OSB/BIA products. When `compare_dcb` is false, GPS and
            Galileo receive zero TGD placeholders, while BeiDou broadcast NAV
            inputs use the broadcast group-delay fields available in the NAV
            data. When `compare_dcb` is true, matching DSB/OSB products are read
            and converted to the internal `TGD` column; every SP3 input needs its
            bias file, a NAV input uses its broadcast group delays (its
            `dcb_path_*` is not read). A bias file given with `compare_dcb=False`
            is not read either (UserWarning at `run()`). `TGD` [m] is chosen so
            that ``clk - TGD`` is the same clock on both sides: GPS the L1 P(Y)
            clock (LNAV TGD; product: -DSB(C1W-C2W)/(gamma-1)), Galileo the E1
            clock (BGD(E1,E5b) of I/NAV for E1/E5b, BGD(E1,E5a) of F/NAV for
            E5a/E1E5a; product: -DSB(C1C-Cx)/(gamma-1), and for E1/E5b the
            precise clock is moved to the I/NAV IF(E1,E5b) datum), BeiDou the
            clock of the mode's signal or IF combination (broadcast TGD1/TGD2 or
            CNAV TGD_B1Cp/TGD_B2ap relative to B3I; product OSBs in the IF(C2I,C6I)
            datum, DSBs tied to that datum through C2I-C6I). Each bias record
            applies from its BIAS_START to its BIAS_END; a comparison across
            midnight takes a list of the products of the compared days (of one
            analysis centre). A product without the
            system or the codes of the mode, or without a record valid at the
            compared epochs, gives a ValueError at `run()`; satellites (or epochs)
            the product does not cover get NaN `TGD`/`dTGD`/`sisre` and one
            RuntimeWarning listing them (`sisre_notgd` is kept).
            Modes with `compare_dcb=True` (`SIS_COMPARE_DCB_MODES`): GPS every
            `gps_mode` (DSB C1W-C2W or OSB C1W/C2W); Galileo E1, E5b, E5a and
            E1E5a (C1C-C5Q and, for E1/E5b, C1C-C7Q); BeiDou two-signal modes
            with an OSB product of both signals or a DSB product holding the
            mode's own pair (tied to the IF(C2I,C6I) clock datum by DSB chains),
            single-signal modes with an OSB product only. Galileo E1E5b and
            E5aE5b, BeiDou single-signal modes with a DSB product and BeiDou
            pairs a DSB product gives only through chains (e.g. B2aB2b) raise a
            ValueError naming the supported modes: the datum algebra defines them
            (E5aE5b = -K[BGD(E1,E5b) - BGD(E1,E5a)], K = 15.009), but they are not
            enabled in this version.
        atx_path, apply_satellite_pco:
            Optional antenna model path and switch for satellite phase-center
            correction of precise SP3 coordinates. For GPS broadcast-vs-SP3 in a
            mode other than L1L2 the precise orbit is moved to the IF(L1,L2) APC the
            broadcast orbit refers to (IS-GPS-200), and the radial offset of the
            user's APC from it is reported in `dR_apc_signal` (information only, as
            below). For BeiDou broadcast-vs-SP3 in a mode other than B3I the precise
            orbit is moved to the B3I APC, the signal the broadcast clock and every
            message group delay refer to (BDS-SIS-ICD-B1I 5.2.4.10), with
            `dR_apc_signal` likewise. For Galileo `gal_mode="E5aE5b"`
            broadcast-vs-SP3 the precise orbit is moved to IF(E1,E5b), the clock
            pair of the I/NAV message used for that mode: a broadcast ephemeris
            refers to the apparent phase centre common to all frequencies (OS SIS
            ICD 5.1.1) and there is no E5a/E5b clock; the Galileo HAS reference is
            I/NAV as well (HAS SIS ICD 7.1.2). The radial offset of the user's
            IF(E5a,E5b) phase centre from IF(E1,E5b) (positive outward, about
            -0.6 m with igs20) is reported in the column `dR_apc_signal`; it is
            information only and enters neither `dR` nor `sisre`.
        satellite_attitude, attitude_laws, attitude_path:
            Satellite attitude of the PCO step, as in
            ``gnx_py.configuration.Config``: None/``"model"`` (default) the
            attitude laws of the ANTEX block, or the ORBEX attitude of
            `attitude_path` where its analysis centre is the one of the SP3
            input; ``"nominal"`` yaw steering with BeiDou GEO orbit-normal.
            During noon/midnight turns the horizontal PCO turns with the body
            (GPS IIF x = 0.394 m: up to 0.79 m of dA/dC against nominal yaw;
            broadcast vs GRG 2024-035 at 300 s G32 0.38 / 0.25 m). The BeiDou
            GEO frame is orbit-normal (as in PPP, where the yaw-steering frame
            misplaced the GEO APC by up to 1.11 m, BEIDOU-2G), but BeiDou GEO
            is excluded from SIS.
            The output column `att_regime` reports the attitude regime per
            satellite-epoch (0 yaw steering, 7 orbit-normal, 8 ORBEX, others
            noon/midnight turn regimes of ``gnx_py.attitude``).
        missing_sat_antenna:
            A satellite of an SP3 input without a valid ANTEX entry (PRN
            within VALID FROM/UNTIL) for a frequency of the PCO step:
            ``"exclude"`` (default) leaves it out of the comparison and of the
            epoch clock datum (`dt_mean`) with one UserWarning naming the
            satellites, the ANTEX file and the date; ``"raise"`` stops `run()`
            with a ValueError.
        prev_sp3_0, next_sp3_0, prev_sp3, next_sp3:
            Adjacent-day SP3 files used to stabilize Chebyshev interpolation at
            day boundaries: `prev_sp3_0`/`next_sp3_0` belong to `orb_path_0`,
            `prev_sp3`/`next_sp3` to `orb_path_1`. They must be SP3 files of the
            days directly before/after the nominal file (checked, ValueError
            otherwise); given for an input that is not SP3 they are not used
            (UserWarning at `run()`).
        clock_bias, clock_bias_function:
            Controls removal of a common per-epoch clock datum before SISRE
            computation. The function is passed to pandas groupby transform/agg:
            one of `SIS_CLOCK_BIAS_FUNCTIONS` (`"mean"`, `"median"`, `"min"`,
            ...) or a callable. BeiDou: per epoch and satellite generation
            (BDS-2, BDS-3; from the ANTEX block of `atx_path`, else B-CNAV
            records, else the PRN), the two generations having independent
            clock datums; a generation with fewer than 3 satellites at an epoch
            has no datum there (NaN `sisre`, one RuntimeWarning).
        apply_eclipse, extend_eclipse, extension_time:
            Controls satellite eclipse flagging and optional post-eclipse mask
            extension. Units for `extension_time` are minutes (>= 0; None is
            accepted only while the extension is not used).
        tlim:
            Optional `[start_datetime, end_datetime]` processing window.

    Validation:
        Every setting is checked when it is set: an unknown `system` or signal
        mode (`'g'`, `'l1l2'`) is rejected with a ValueError listing the valid
        values (no silent normalisation; the same for `clock_bias_function`
        names such as `'Mean'`), out-of-range numbers (`interval <= 0`,
        `extension_time < 0`) with the allowed range, and every given file path
        must exist (FileNotFoundError naming the setting; a one-element list is
        accepted as that path; `dcb_path_0`/`dcb_path_1` also take a list of
        files). Orbit inputs must be RINEX navigation or SP3
        files. Requirements that combine settings (`atx_path` for satellite PCO
        of an SP3 input, `dcb_path_0`/`dcb_path_1` for the SP3 inputs of
        `compare_dcb=True`, the signal mode of the PCO step, the systems with a
        broadcast, PCO or bias branch) are checked at the start of
        `SISSession.run()`.

    Changing settings:
        A setting may be changed after construction, also after a
        `SISSession` was built on this object: the new value is validated
        exactly as in the constructor (`bds_mode=None` or `""` again selects `B1IB3I`,
        `tlim` is parsed again) and the session reads the configuration at
        run time, so the next `run()` uses it. Assigning a name that is not a
        setting (a typo such as `apply_eclpise`) raises AttributeError.

    Comparison and representation:
        Two configurations are equal when every setting is equal (paths compared
        as file-system paths, so ``"a.sp3"`` equals ``Path("a.sp3")``; a callable
        `clock_bias_function` by identity); the run state a session keeps on the
        object is not compared. ``repr()`` lists every setting. A configuration
        is mutable and therefore not hashable (before v1.0.0 the former
        dataclass declared no fields, so any two configurations compared equal
        and the repr was ``SISConfig()``).

    Warnings:
        A large BDS clock/SIS component can indicate a clock-datum mismatch
        between broadcast corrections and precise clock products rather than a
        physical orbit/clock error. Inspect `clock_convention_status` and
        `clock_convention_note` on the output before interpreting BDS results.
    """

    def __init__(self, orb_path_0, orb_path_1, interval, prev_sp3=None, next_sp3=None, dcb_path_0=None, dcb_path_1=None, gps_mode=None ,gal_mode=None, bds_mode=None, atx_path=None, system='G',
                 prev_sp3_0=None, next_sp3_0=None, tlim=None,compare_dcb=False, clock_bias=True, clock_bias_function='mean',apply_eclipse=True,apply_satellite_pco=True,extend_eclipse = True, extension_time=30,
                 missing_sat_antenna='exclude', satellite_attitude=None, attitude_laws=None, attitude_path=None):
        """
                Initialize SIS configuration.

                The constructor wires runtime options, file paths, and modeling flags
                (e.g., orbit/DCB/ATX sources, interval/time limits, eclipse/PCO switches).

                Notes:
                    - Every value is validated on assignment (`_validated`), here and on any
                      later change; cross-setting requirements are checked by `SISSession.run()`.
                    - Paths must point to existing files; orbit inputs to RINEX NAV or SP3 files.

                Raises:
                    ValueError: If a value is outside its valid set or range.
                    FileNotFoundError: If a given input file does not exist.
                """

        self.orb_path_0:Union[str,Path] = orb_path_0
        self.orb_path_1: Union[str, Path] = orb_path_1

        self.compare_dcb = compare_dcb
        self.dcb_path_0: Optional[Union[str, Path, None]] = dcb_path_0
        self.dcb_path_1: Optional[Union[str, Path, None]] = dcb_path_1

        self.atx_path: Union[str, Path] =atx_path
        self.system: str = system
        self.interval: Union[float, int] = interval

        self.gps_mode: Union[str, None] = gps_mode
        self.gal_mode:Union[str, None] = gal_mode
        self.bds_mode: Union[str, None] = bds_mode  # None -> "B1IB3I"

        self.apply_eclipse: bool = apply_eclipse
        self.apply_satellite_pco:bool=apply_satellite_pco
        # satellite of an SP3 input without a valid ANTEX entry: 'exclude' (warning) or 'raise' (bug-fix stage 11)
        self.missing_sat_antenna: str = missing_sat_antenna
        # satellite attitude of the PCO step (bug-fix stage 11c, as gnx_py.configuration.Config)
        self.satellite_attitude: Optional[str] = satellite_attitude
        self.attitude_laws: Optional[dict] = attitude_laws
        self.attitude_path = attitude_path

        self.prev_sp3_0: Optional[Union[str, Path, None]] = prev_sp3_0
        self.next_sp3_0: Optional[Union[str, Path, None]] = next_sp3_0
        self.prev_sp3: Optional[Union[str, Path, None]] = prev_sp3
        self.next_sp3: Optional[Union[str, Path, None]] = next_sp3

        self.clock_bias:bool=clock_bias
        self.clock_bias_function:str=clock_bias_function

        # set (and validated) also with apply_eclipse=False, so that switching the eclipse
        # flag on later finds them
        self.extend_eclipse:bool = extend_eclipse
        self.extension_time : Union[int, float] = extension_time
        # [start, end] as Timestamps (strings are accepted), checked before any file is read
        self.tlim: Union[List[datetime, datetime], None] = tlim

    _ORBIT_PATH_FIELDS = ("orb_path_0", "orb_path_1")
    _NEIGHBOUR_FIELDS = ("prev_sp3_0", "next_sp3_0", "prev_sp3", "next_sp3")
    _PATH_FIELDS = ("dcb_path_0", "dcb_path_1", "atx_path", "attitude_path") + _NEIGHBOUR_FIELDS
    _FLAG_FIELDS = ("compare_dcb", "clock_bias", "apply_eclipse", "apply_satellite_pco", "extend_eclipse")
    _FIELDS = _ORBIT_PATH_FIELDS + _PATH_FIELDS + _FLAG_FIELDS + (
        "system", "interval", "gps_mode", "gal_mode", "bds_mode", "clock_bias_function", "extension_time", "tlim",
        "missing_sat_antenna", "satellite_attitude", "attitude_laws")

    #: settings in the order of the constructor (``repr`` and ``==``)
    _REPR_ORDER = ("orb_path_0", "orb_path_1", "interval", "system", "gps_mode", "gal_mode", "bds_mode",
                   "prev_sp3_0", "next_sp3_0", "prev_sp3", "next_sp3", "compare_dcb", "dcb_path_0", "dcb_path_1",
                   "atx_path", "apply_satellite_pco", "missing_sat_antenna", "satellite_attitude", "attitude_laws",
                   "attitude_path", "tlim", "clock_bias", "clock_bias_function", "apply_eclipse", "extend_eclipse",
                   "extension_time")

    __hash__ = None  # mutable settings: not hashable (as the former dataclass)

    def _settings(self) -> dict:
        """{setting: value} of every setting (the run state kept under ``_``-names is not a setting)."""
        return {name: getattr(self, name, None) for name in self._REPR_ORDER}

    @classmethod
    def _comparable(cls, name: str, value):
        """`value` of setting `name` in a form that compares by meaning: paths as normalised strings (a list of
        bias files as a tuple of them), other lists as tuples."""
        if name in cls._ORBIT_PATH_FIELDS + cls._PATH_FIELDS:
            if isinstance(value, (list, tuple)):
                return tuple(os.path.normpath(os.fspath(v)) for v in value)
            return None if value is None else os.path.normpath(os.fspath(value))
        if isinstance(value, list):
            return tuple(value)
        return value

    def __eq__(self, other):
        if not isinstance(other, SISConfig):
            return NotImplemented
        mine, theirs = self._settings(), other._settings()
        for name in self._REPR_ORDER:
            a, b = self._comparable(name, mine[name]), self._comparable(name, theirs[name])
            if callable(a) or callable(b):
                if a is not b:
                    return False
            elif not bool(a == b):
                return False
        return True

    def __repr__(self) -> str:
        items = ", ".join(f"{name}={value!r}" for name, value in self._settings().items())
        return f"{type(self).__name__}({items})"

    def __setattr__(self, name, value):
        """Validate (and normalise) a setting on every assignment; reject unknown names."""
        if name not in self._FIELDS and not name.startswith("_"):
            raise AttributeError(
                f"SISConfig has no setting {name!r}{_did_you_mean(name, self._FIELDS)}; "
                f"settings: {list(self._FIELDS)}")
        object.__setattr__(self, name, self._validated(name, value))

    def _validated(self, name: str, value):
        """`value` checked for setting `name` (ValueError/FileNotFoundError naming the setting)."""
        if name in self._ORBIT_PATH_FIELDS:
            return _check_orbit_path(name, value)
        if name in self._NEIGHBOUR_FIELDS:
            return _check_neighbour_sp3(name, value)
        if name in ("dcb_path_0", "dcb_path_1", "attitude_path"):
            return _check_bias_paths(name, value)
        if name in self._PATH_FIELDS:
            return _check_input_path(name, value)
        if name in self._FLAG_FIELDS:
            return _check_flag(name, value)
        if name == "system":
            if isinstance(value, str) and value not in SIS_SYSTEMS and value.strip().upper() in _SYSTEM_ALIASES:
                code = _SYSTEM_ALIASES[value.strip().upper()]
                raise ValueError(f"system must be one of {list(SIS_SYSTEMS)} (got {value!r}); did you mean {code!r}?")
            return _check_choice(name, value, SIS_SYSTEMS)
        if name == "gps_mode":
            return _check_choice(name, value, SUPPORTED_MODES_BY_SYSTEM["G"], allow_none=True)
        if name == "gal_mode":
            return _check_choice(name, value, SUPPORTED_MODES_BY_SYSTEM["E"], allow_none=True)
        if name == "bds_mode":
            if value is None or (isinstance(value, str) and value == ""):  # not given -> default
                return BDS_LEGACY_IF_PRECISE_CLOCK_MODE
            return _check_choice(name, value, SUPPORTED_MODES_BY_SYSTEM["C"], allow_none=True)
        if name == "interval":
            # a non-positive interval would build an endless epoch grid
            if not _is_real(value):
                raise ValueError(f"interval must be a number of seconds > 0 (got {value!r})")
            if not np.isfinite(value) or not value > 0:
                raise ValueError(f"interval must be > 0 s (got {value!r})")
            return value
        if name == "extension_time":
            if value is None:  # allowed while unused; run() requires it when the extension is on
                return None
            if not _is_real(value):
                raise ValueError(f"extension_time must be a number of minutes >= 0 (got {value!r})")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"extension_time must be >= 0 min (got {value!r})")
            return value
        if name == "clock_bias_function":
            if callable(value):
                return value
            if not isinstance(value, str):
                raise TypeError(f"clock_bias_function must be one of {list(SIS_CLOCK_BIAS_FUNCTIONS)} "
                                f"or a callable (got {value!r})")
            return _check_choice(name, value, SIS_CLOCK_BIAS_FUNCTIONS, context=" or a callable")
        if name == "tlim":
            return None if value is None else list(check_time_limit(value, "tlim"))
        if name == "missing_sat_antenna":
            return _check_choice(name, value, MISSING_SAT_ANTENNA_POLICIES)
        if name == "satellite_attitude":
            return _check_choice(name, value, SATELLITE_ATTITUDE_MODES, allow_none=True)
        if name == "attitude_laws":
            return _check_attitude_laws(value)
        return value


class SISSession:
    """
    High-level session for SIS/SISRE orbit and clock comparisons.

    Status:
        Active public API. The class is the main orchestration layer for the
        `gnx_py.orbits` package. It also contains guarded BDS helpers that
        require caution because they encode clock datum, TGD, OSB and DSB
        assumptions.

    Pipeline:
        1. Classify both orbit inputs as RINEX navigation, SP3, observation or
           unknown files.
        2. Generate broadcast coordinates/clocks or interpolate precise SP3
           products onto the requested epoch grid.
        3. Optionally apply satellite PCO corrections to precise products.
        4. Merge broadcast TGD or external DCB/OSB/DSB products into a common
           `TGD` representation in meters.
        5. Align both products to common satellite/epoch pairs.
        6. Compute coordinate differences, clock differences, RAC components
           and SISRE/SIS-style metrics.
        7. Annotate BDS outputs with clock-convention status when relevant.

    Outputs:
        `run()` returns a pandas DataFrame indexed by satellite and epoch. Core
        columns include position differences (`dx`, `dy`, `dz`), RAC components
        (`dR`, `dA`, `dC`), clock error in meters (`dt`), group delay difference
        (`dTGD`) and SISRE variants (`sisre`, `sisre_orb`, `sisre_notgd`).

    Limitations:
        BeiDou GEO is unsupported in this path. Legacy BDS `B1IB3I` MEO/IGSO is
        the validated broadcast-vs-precise clock comparison mode; precise BDS-3
        clocks in IF(B1C,B2a) (COD MGEX 2026) are compared at IF(B1I,B3I)
        through the broadcast B-CNAV group delays of a RINEX 4 navigation file
        (`_convert_bds_precise_clock_datum`). Modern BDS
        modes and single-signal legacy modes are exposed for guarded validation
        and should be interpreted through the clock-convention metadata.
    """

    # Earth rotation rate [rad/s] (WGS-84/IERS), used to turn the Earth-fixed satellite
    # velocity into the inertial one that defines the orbit plane of the RAC frame.
    _OMEGA_E = 7.2921151467e-5

    def __init__(self, config: SISConfig):
        """
                Initialize session state and bind configuration.

                Notes:
                    - Binds constants (e.g., speed of light) and establishes internal state.
                    - Does not perform I/O; data loading happens in dedicated methods.
                    - Keeps no copy of the settings: `system` and everything else are read
                      from `config` at run time, so a later change of the config is honoured.
                """

        self.config=config
        self.clight = 299792458
        self.output_cols = ['dR','dA','dC','dt','dt_mean','dTGD','dTGD_mean','sisre','sisre_orb','sisre_notgd', 'dx', 'dy',
       'dz']

    @property
    def system(self) -> str:
        """GNSS system of the comparison: always the current `config.system` (no copy)."""
        return self.config.system

    @system.setter
    def system(self, value):
        self.config.system = value

    def classify_file(self, path: str | Path) -> str:
        """
        Classify an input file and determine its handling route.

        Returns:
            str: ``"SP3"``, ``"RINEX_NAV"``, ``"RINEX_OBS"``, ``"RINEX_MET"``,
            ``"RINEX_UNKNOWN"`` or ``"UNKNOWN"`` (from the file name, see
            `classify_orbit_file`); only SP3 and RINEX_NAV inputs are compared.
        """

        return classify_orbit_file(path)

    def process_broadcast_orbit(self, path: str | Path):
        """
        Process broadcast ephemerides and align them to the target epoch grid.

        Status:
            Active. Supports GPS, Galileo and BeiDou broadcast branches through
            `BrdcGenerator` and the configured signal mode.

        Returns:
            pandas.DataFrame: Per-satellite broadcast coordinates, clocks and
            available navigation delay fields aligned to the session epochs.

        Raises:
            ValueError: If the configured system is unsupported or no broadcast
            data are available for that system.
        """

        prc =   GNSSDataProcessor2(nav_path=path, sys=self.config.system)
        if self.config.tlim is not None:
            nav = prc.load_broadcast_orbit(tlim=self.config.tlim)
        else:
            nav = prc.load_broadcast_orbit()
        if self.system =='G':
            orbit = nav.gps_orb
            mode = self.config.gps_mode
            tol='4H'
        elif self.system =='E':
            orbit = nav.gal_orb
            mode = self.config.gal_mode
            tol = '45T'
        elif self.system == 'C':
            orbit = nav.bds_orb
            mode = self.config.bds_mode
            tol = '2H'
        else:
            raise ValueError(f"Unsupported sys: {self.system}")
        if orbit is None or orbit.empty:
            raise ValueError(f"No broadcast orbit data found for system {self.system}.")
        if self.system == 'C':
            # B-CNAV group delays of the file for the IF(B1C,B2a) precise-clock conversion (N-ORB-24): kept from
            # this read, so the navigation file is not read (and warned about) once more
            self._bds_bcnav_tgd = self._bcnav_medians(orbit)

        broadcast_interpolator = BrdcGenerator(system=self.system,
                                               interval=self.config.interval,
                                               mode=mode,
                                               nav=orbit,tolerance=tol,
                                               tlim=self.config.tlim)
        crd = broadcast_interpolator.generate()
        crd[['x_apc','y_apc','z_apc']] = crd[['x','y','z']]
        return crd

    def process_precise_orbit(self, path: str | Path, prev_path: str|Path, next_path:str|Path,
                              neighbour_params=("prev_sp3", "next_sp3")):
        """
        Process precise SP3 orbits and align them to the target epoch grid.

        Status:
            Active. For BDS, GEO satellites are filtered out before
            interpolation because the current SISRE path is validated for
            MEO/IGSO only.

        Args:
            path: Nominal-day SP3 file.
            prev_path: Previous-day SP3 file used for boundary interpolation.
            next_path: Next-day SP3 file used for boundary interpolation.
            neighbour_params: Config names of `prev_path`/`next_path`, used in
                error messages.

        Returns:
            pandas.DataFrame: Interpolated satellite positions and clocks,
            indexed by satellite and time.

        Notes:
            The method uses Chebyshev interpolation and optionally adjacent-day
            SP3 products to reduce edge effects.
        """

        sp3 =   read_sp3(path=path, sys=self.system)
        sp3 = self._filter_precise_bds_geo(sp3, path)
        if sp3.empty:
            raise ValueError(f"No precise SP3 orbit data found for system {self.system}: {path}")
        precise_interp = SP3InterpolatorOptimized(sp3_dataframe=sp3)
        # Each product is extended with its OWN neighbour-day files (prev_sp3_0/next_sp3_0
        # for orb_path_0, prev_sp3/next_sp3 for orb_path_1).
        prev_sp3 = self._read_neighbour_sp3(prev_path, sp3, path, side="prev", param=neighbour_params[0])
        next_sp3 = self._read_neighbour_sp3(next_path, sp3, path, side="next", param=neighbour_params[1])
        epochs = self._precise_epoch_grid(sp3, prev_sp3 is not None, next_sp3 is not None)
        precise_interp.include_adjacent_data(prev_sp3_df=prev_sp3,next_sp3_df=next_sp3)
        interpolated = precise_interp.run(epochs=epochs, method='chebyshev')
        interpolated.index.names = ['time', 'sv']
        interpolated = interpolated.swaplevel()
        return interpolated

    def _precise_epoch_grid(self, sp3: pd.DataFrame, has_prev: bool, has_next: bool) -> list:
        """
        Epochs at which the precise product is interpolated, on the grid of the broadcast
        side (`BrdcGenerator._epoch_grid`), so that both products share their epochs.

        With `tlim`: its epochs at whole multiples of the interval from midnight. Without
        it: the observation day(s) of the file, 00:00 to 24:00 minus one interval, the
        24:00 node of a daily file being the next day's first epoch; without a neighbour
        day the grid stops at the first/last node instead of extrapolating past it.
        """
        interval = self.config.interval
        if self.config.tlim is not None:
            start, end = self.config.tlim
        else:
            t = sp3.index.get_level_values('time')
            first, last = t.min(), t.max()
            day0 = (first + pd.Timedelta(hours=12)).floor('D')
            day1 = (last - pd.Timedelta(hours=12)).floor('D') + pd.Timedelta(days=1)
            if day1 > day0:
                start, end = day0, day1 - pd.Timedelta(seconds=interval)
            else:  # sub-daily file: the span of its nodes
                start, end = first, last
            if not has_prev:
                start = max(start, first)
            if not has_next:
                end = min(end, last)
        epochs = interval_epochs(start, end, interval)
        if epochs.empty:
            raise ValueError(
                f"No epoch of the {interval} s grid lies within {start} .. {end} "
                f"(tlim={self.config.tlim}); widen tlim or shorten the interval."
            )
        return list(epochs.to_pydatetime())

    def _filter_precise_bds_geo(self, sp3: pd.DataFrame, path: str | Path) -> pd.DataFrame:
        """
        Remove BeiDou GEO satellites from precise-orbit SIS processing.

        Status:
            Active guard rail / unsupported-case handling.

        The current SISRE coefficient and validation stack covers BDS MEO and
        IGSO satellites. GEO satellites are therefore skipped with a runtime
        warning instead of being allowed to enter RAC/SISRE calculations with
        coefficients that have not been validated for GEO geometry.
        """

        if self.system != "C" or sp3.empty:
            return sp3
        sv_index = sp3.index.get_level_values("sv")
        geo_sats = sorted({sv for sv in sv_index if is_bds_geo(sv)})
        if not geo_sats:
            return sp3
        warnings.warn(
            "BeiDou SIS/SISRE precise-orbit processing currently supports BDS MEO/IGSO only; "
            f"skipping GEO satellites from {path}: {', '.join(geo_sats)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        out = sp3.loc[~sv_index.map(is_bds_geo)].copy()
        return out

    # Largest admissible gap between the nominal-day SP3 and a neighbour-day file; real
    # products leave at most one node spacing (5-15 min) between consecutive days.
    _NEIGHBOUR_SP3_MAX_GAP = pd.Timedelta(hours=1)

    @staticmethod
    def _sp3_agency(path) -> str | None:
        """Agency of an SP3 product (header line 1, columns 57-60), None if unreadable."""
        try:
            opener = gzip.open if str(path).lower().endswith(".gz") else open
            with opener(path, "rt", encoding="ascii", errors="ignore") as fh:
                line = fh.readline()
        except (OSError, EOFError):
            return None
        agency = line[56:60].strip() if line.startswith("#") else ""
        return agency or None

    def _read_neighbour_sp3(self, neighbour_path, sp3: pd.DataFrame, path, side: str, param: str):
        """
        Read an adjacent-day SP3 file for boundary interpolation and check that it really
        is the day before (`side="prev"`) or after (`side="next"`) the nominal file `path`.

        The neighbour gets the same BDS GEO filter as the nominal file, otherwise GEO
        satellites re-enter the comparison through the boundary nodes.

        Raises:
            ValueError: If the neighbour file does not adjoin the nominal-day file.
        """

        if neighbour_path is None:
            return None
        nb = read_sp3(path=neighbour_path, sys=self.system)
        nb = self._filter_precise_bds_geo(nb, neighbour_path)
        if nb.empty:
            warnings.warn(f"Neighbour SP3 file {neighbour_path} has no satellites of system {self.system}; "
                          "it is ignored.", RuntimeWarning, stacklevel=2)
            return None
        t_main = sp3.index.get_level_values("time")
        t_nb = nb.index.get_level_values("time")
        m0, m1, n0, n1 = t_main.min(), t_main.max(), t_nb.min(), t_nb.max()
        gap = self._NEIGHBOUR_SP3_MAX_GAP
        if side == "prev":
            adjacent = n0 < m0 and n1 < m1 and m0 - n1 <= gap
            expected = f"to end at the start of {path} ({m0})"
        else:
            adjacent = n0 > m0 and n1 > m1 and n0 - m1 <= gap
            expected = f"to start at the end of {path} ({m1})"
        if not adjacent:
            raise ValueError(
                f"{param} file {neighbour_path} (spanning {n0} - {n1}) is not the adjacent {side} day of "
                f"{path} (spanning {m0} - {m1}); it is expected {expected} within "
                f"{gap.total_seconds() / 3600:g} h."
            )
        # Clocks of different analysis centres refer to different clock datums (and orbits
        # to slightly different realisations): gluing them at the day boundary puts a jump
        # into the interpolated clock near midnight.
        ac_main, ac_nb = self._sp3_agency(path), self._sp3_agency(neighbour_path)
        if ac_main and ac_nb and ac_main.upper() != ac_nb.upper():
            warnings.warn(
                f"{param} file {neighbour_path} comes from agency {ac_nb}, {path} from {ac_main}; "
                "boundary interpolation mixes two clock datums. Use neighbour-day files of the same product.",
                RuntimeWarning, stacklevel=2)
        return nb

    def set_common_epochs(self, orbit_0:pd.DataFrame, orbit_1:pd.DataFrame):
        """
                Align two time series to a common epoch grid.

                Returns:
                    tuple(pd.DataFrame, pd.DataFrame): Both inputs aligned to the same epoch grid,
                    following the session's alignment rules.

                Notes:
                    - The specific return structure follows the internal data model in use.
                """

        cmn = orbit_0.index.intersection(orbit_1.index)
        if len(cmn) == 0:
            # An empty comparison is never a valid SIS result: the two epoch grids (or the
            # satellite sets) do not overlap.
            def _span(orbit):
                if orbit.empty:
                    return "no data"
                t = orbit.index.get_level_values("time")
                return (f"{t.min()} - {t.max()}, {t.nunique()} epochs, "
                        f"{orbit.index.get_level_values('sv').nunique()} satellites")

            raise ValueError(
                f"No common satellite/epoch pairs between orb_path_0 ({self.config.orb_path_0}: "
                f"{_span(orbit_0)}) and orb_path_1 ({self.config.orb_path_1}: {_span(orbit_1)}) "
                f"for system {self.system}; check tlim, interval and that both products cover the same "
                f"day and satellites."
            )
        return orbit_0.loc[cmn].copy(), orbit_1.loc[cmn].copy()

    def _ensure_apc_columns(self, orbit: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that orbit coordinates expose antenna-phase-center columns.

        Status:
            Active compatibility helper.

        Broadcast products generally carry geometric coordinates as `x`, `y`,
        `z`; precise products may later be corrected to `x_apc`, `y_apc`,
        `z_apc`. This helper keeps both branches compatible without changing
        coordinate values when APC-specific columns are absent.
        """

        out = orbit.copy()
        for raw_col, apc_col in (("x", "x_apc"), ("y", "y_apc"), ("z", "z_apc")):
            if apc_col not in out.columns:
                if raw_col not in out.columns:
                    raise KeyError(f"Orbit table is missing both {apc_col!r} and {raw_col!r}.")
                out[apc_col] = out[raw_col]
        return out

    def _bds_clock_convention_policy(self, orbit_0_type: str, orbit_1_type: str) -> dict[str, object]:
        """
        Classify BDS clock-datum comparability for the current run.

        Status:
            Active, requires caution.

        Returns:
            dict: Metadata describing whether the selected BDS mode is directly
            comparable, warning-only, requires validation, or unsupported.

        Notes:
            Broadcast BDS TGD/ISC fields are user corrections for a selected
            signal or ionosphere-free combination. Precise SP3 clocks can be
            tied to a different code or IF datum. This method does not alter
            the numerical result; it records whether the result can be
            interpreted as a direct SIS clock error.
        """

        mode = self.config.bds_mode
        policy = {
            "system": self.system,
            "mode": mode,
            "status": "ok",
            "directly_comparable": True,
            "note": "",
        }
        if self.system != "C":
            return policy

        try:
            signals = mode_signals(mode)
        except ValueError as exc:
            return {
                **policy,
                "status": "unsupported",
                "directly_comparable": False,
                "note": str(exc),
            }

        compares_broadcast_to_precise = {"RINEX_NAV", "SP3"}.issubset({orbit_0_type, orbit_1_type})
        if not compares_broadcast_to_precise:
            return {
                **policy,
                "note": "No broadcast-vs-precise BDS clock datum comparison is performed in this run.",
            }

        has_external_bias = bool(
            self.config.compare_dcb and (self.config.dcb_path_0 is not None or self.config.dcb_path_1 is not None)
        )
        signal_set = set(signals)

        if signal_set.intersection(BDS_MODERN_BROADCAST_SIGNALS):
            return {
                **policy,
                "status": "requires_validation",
                "directly_comparable": False,
                "note": (
                    f"BeiDou mode {mode} contains modern signals {sorted(signal_set.intersection(BDS_MODERN_BROADCAST_SIGNALS))}. "
                    "Broadcast-vs-precise SIS clock comparison requires RINEX 4 CNAV ISC/TGD fields and a clock-reference "
                    "consistent OSB/DSB stack for the selected observables. "
                    + BDS_CLOCK_CONVENTION_NOTE
                ),
            }

        if mode == BDS_LEGACY_IF_PRECISE_CLOCK_MODE:
            # the datum of the SP3's BDS-3 clocks (bug-fix stage 13b, SPP-080): declared or tested on the broadcast
            # clocks; IF(B1C,B2a) (COD MGEX 2026) puts every BDS-3 clock of this mode 3-10 m off per satellite
            datum = self._bds_precise_clock_datum(orbit_0_type, orbit_1_type)
            if datum.get("datum") not in (None, BDS_LEGACY_IF_PRECISE_CLOCK_MODE):
                found = (f"the BDS-3 precise clocks refer to IF(B1C,B2a) ({datum.get('source')}: "
                         f"{datum.get('reason') or 'between-satellite scatter of precise minus broadcast clocks'} "
                         f"IF(B1I,B3I) {datum.get('std_h1_m')} m, IF(B1C,B2a) {datum.get('std_h2_m')} m)")
                conversion = self.__dict__.get("_bds_datum_conversion") or {}
                if conversion.get("via"):
                    # N-ORB-24 (bug-fix stage 13d): the precise group delay of the BDS-3 satellites carries the
                    # datum difference, so clk - TGD is the IF(B1I,B3I) clock on both sides
                    return {
                        **policy,
                        "note": (
                            f"{found}; {conversion['note']} BeiDou B1I/B3I ionosphere-free mode, legacy IF "
                            "broadcast datum correction a*TGD1."
                        ),
                    }
                return {
                    **policy,
                    "status": "warning",
                    "directly_comparable": False,
                    "note": (
                        f"{found}, not IF(B1I,B3I) as mode B1IB3I assumes, and they could not be converted "
                        f"({conversion.get('reason') or 'no conversion attempted'}): the BDS-3 SIS clock errors "
                        "carry the per-satellite difference of the two datums (3-10 m). " + BDS_CLOCK_CONVENTION_NOTE
                    ),
                }
            return {
                **policy,
                "note": (
                    "BeiDou B1I/B3I ionosphere-free mode uses the legacy IF broadcast datum correction "
                    "a*TGD1 and is the validated legacy mode for precise C2I/C6I-referenced clocks"
                    + (f" (precise clock datum IF(B1I,B3I) {datum['source']})." if datum.get("datum") else ".")
                ),
            }

        if len(signals) == 1:
            if has_external_bias:
                return {
                    **policy,
                    "status": "warning",
                    "directly_comparable": False,
                    "note": (
                        f"BeiDou single-signal mode {mode} uses a signal-specific broadcast group delay, while the "
                        "precise clock may be tied to an IF reference such as C2I/C6I. The external bias product is "
                        "used where possible, but clock-stack consistency must be verified before interpreting the "
                        "SIS clock component as a true signal clock error. "
                        + BDS_CLOCK_CONVENTION_NOTE
                    ),
                }
            return {
                **policy,
                "status": "requires_validation",
                "directly_comparable": False,
                "note": (
                    f"BeiDou single-signal mode {mode} is not directly comparable with a precise IF clock without "
                    "an explicit clock-datum conversion from a matching OSB/DSB product. Use B1IB3I for legacy "
                    "BDS SIS clock validation, or enable compare_dcb with a clock-reference consistent product. "
                    + BDS_CLOCK_CONVENTION_NOTE
                ),
            }

        return {
            **policy,
            "status": "requires_validation",
            "directly_comparable": False,
            "note": (
                f"BeiDou mode {mode} is not the validated legacy precise-clock comparison mode. Its broadcast "
                "TGD combination may be valid for user corrections, but SIS clock comparison needs a documented "
                "precise-clock datum conversion for the selected observables. "
                + BDS_CLOCK_CONVENTION_NOTE
            ),
        }

    def _bds_precise_clock_datum(self, orbit_0_type: str, orbit_1_type: str) -> dict:
        """IF datum of the precise BeiDou clocks of the SP3 input, once per run (``session_guards.bds_clock_datum_of_files``): declared by a bias product of the SP3's analysis centre, else
        tested against the broadcast clocks of the RINEX navigation input; ``{'datum': None}`` when undetermined."""
        memo = _notes.run_memo(self.config, "precise_clock_datum")
        if "C" not in memo:
            from ..core.session_guards import bds_clock_datum_of_files

            sides = [(orbit_0_type, self.config.orb_path_0, self.config.dcb_path_0),
                     (orbit_1_type, self.config.orb_path_1, self.config.dcb_path_1)]
            sp3 = next(((p, d) for t, p, d in sides if t == "SP3"), (None, None))
            nav = next((p for t, p, _ in sides if t == "RINEX_NAV"), None)
            try:
                memo["C"] = bds_clock_datum_of_files(sp3[0], nav, atx_path=self.config.atx_path, dcb_paths=sp3[1])
            except (OSError, ValueError) as exc:
                memo["C"] = {"datum": None, "reason": str(exc)}
        return memo["C"]

    #: fewest satellites of one BeiDou generation with a clock difference at an epoch for that generation's own clock
    #: datum: with fewer the generation's `dt_mean`,
    #: `dTGD_mean`, `sisre` and `sisre_notgd` are NaN at that epoch (one or two satellites would be compared with
    #: (almost) themselves)
    BDS_CLOCK_DATUM_MIN_SATS = 3

    def _bds_generations_of(self, satellites, epoch, cnav_satellites=()) -> dict:
        """{satellite: generation} of BeiDou satellites (``biases.bds_satellite_generations``): the block of the
        ANTEX entry of `atx_path` valid at `epoch`, else B-CNAV records (RINEX 4 B1C/B2a messages, only BDS-3
        MEO/IGSO), else the PRN (C59-C63 BDS-3 GEO, C19 and above BDS-3). The PRN alone is wrong after 2023
        (C06-C08 and C11-C14 belong to BDS-3 satellites in 2026)."""
        from ..models.biases import bds_satellite_generations

        atx = None
        if self.config.atx_path:
            from ..models.antex import load_antex
            try:
                atx = load_antex(self.config.atx_path)
            except (OSError, ValueError):
                atx = None
        when = None if epoch is None or pd.isna(epoch) else pd.Timestamp(epoch).to_pydatetime()
        gens, _ = bds_satellite_generations({str(s)[:3] for s in satellites}, antex=atx, epoch=when,
                                            cnav_satellites=cnav_satellites)
        return gens

    @staticmethod
    def _bcnav_satellites(*orbits) -> set:
        """BeiDou satellites whose broadcast rows carry B-CNAV group delays TGD_B1Cp and TGD_B2ap (RINEX 4)."""
        out = set()
        for orbit in orbits:
            if orbit is None or not {"TGD_B1Cp", "TGD_B2ap"} <= set(orbit.columns):
                continue
            ok = (np.isfinite(pd.to_numeric(orbit["TGD_B1Cp"], errors="coerce").to_numpy(float))
                  & np.isfinite(pd.to_numeric(orbit["TGD_B2ap"], errors="coerce").to_numpy(float)))
            out |= set(pd.Index(orbit.index.get_level_values("sv")[ok]).astype(str).str[:3])
        return out

    def _bds_cnav_satellites(self, *orbits) -> set:
        """BeiDou satellites with B-CNAV1/B-CNAV2 group delays (only BDS-3 MEO/IGSO transmit B1C/B2a): those of the
        navigation file read in this run (`_bds_bcnav_tgd`, kept by `process_broadcast_orbit` from the whole file,
        as the datum test of ``session_guards`` sees them) and any row of ``orbits`` carrying them. The interpolated
        broadcast orbit holds only the selected D1/D2 rows, whose B-CNAV fields are empty: without the kept
        values the B-CNAV step of the generation would never apply, and without an ANTEX the PRN rule would type
        the BDS-3 satellites on reused PRNs (C06-C08, C11-C14 in 2026) as BDS-2."""
        out = set(self._bcnav_satellites(*orbits))
        kept = self.__dict__.get("_bds_bcnav_tgd")
        if kept is not None and len(kept):
            out |= set(pd.Index(kept.dropna(how="all").index).astype(str).str[:3])
        return out

    def _convert_bds_precise_clock_datum(self, orbit_0: pd.DataFrame, orbit_1: pd.DataFrame, orbit_0_type: str,
                                         orbit_1_type: str):
        """
        BeiDou B1IB3I broadcast vs precise with BDS-3 precise clocks in IF(B1C,B2a).

        The broadcast clock refers to B3I and every message group delay is relative to B3I (BDS-SIS-ICD-B1I
        5.2.4.10, B1C/B2a 7.6), so the IF(B1I,B3I) clock is ``dt - k1 TGD1`` and the IF(B1C,B2a) clock ``dt - (k1'
        TGD_B1Cp - k2' TGD_B2ap)`` (k1 = 2.944, k1' = 2.261, k2' = 1.261). A precise clock of a BDS-3 satellite in
        IF(B1C,B2a) (COD MGEX 2026; ``biases.PreciseClockDatum``: declared by a bias product of the SP3's
        analysis centre or detected on the broadcast clocks) therefore gets the group delay of the mode relative to
        its datum, ``TGD = c [k1 TGD1 - (k1' TGD_B1Cp - k2' TGD_B2ap)]``, so that ``clk - TGD`` is the IF(B1I,B3I)
        clock on both sides (the broadcast side keeps ``TGD = c k1 TGD1``; for an IF(B1I,B3I) precise clock the
        same formula gives the former 0). TGD1 is that of the broadcast D1/D2 row of the same satellite and
        epoch, TGD_B1Cp and TGD_B2ap the median of the satellite's B-CNAV1/B-CNAV2 records in the navigation
        file (RINEX 4; the D1/D2 rows do not carry them), as in the datum test. BDS-2 and BDS-3 GEO satellites
        (no open-service B1C/B2a) keep IF(B1I,B3I) and TGD 0. `dt` and `sisre_notgd` stay the raw clock
        difference.

        With `compare_dcb=True` the precise group delay comes from the bias product and nothing is converted
        (status 'warning'): a product of the SP3's analysis centre that declares the IF(B1C,B2a) datum should give
        its OSBs in that datum, but no such day is graded yet (the note says so); any other product leaves the
        datum unconverted. Without B-CNAV group delays (RINEX 3
        navigation) nothing is converted (status 'warning'); a BDS-3 satellite without them while others have them
        gets NaN `TGD`/`dTGD`/`sisre` with one RuntimeWarning naming it.

        Returns the two orbits (the precise one with the converted `TGD`); the outcome is kept for the
        clock-convention status (`_bds_datum_conversion`).
        """
        self._bds_datum_conversion = None
        if (self.system != "C" or self.config.bds_mode != BDS_LEGACY_IF_PRECISE_CLOCK_MODE
                or {orbit_0_type, orbit_1_type} != {"RINEX_NAV", "SP3"}):
            return orbit_0, orbit_1
        datum = self._bds_precise_clock_datum(orbit_0_type, orbit_1_type)
        if datum.get("datum") in (None, BDS_LEGACY_IF_PRECISE_CLOCK_MODE):
            return orbit_0, orbit_1
        from ..models.biases import BDS3

        nav, sp3 = (orbit_0, orbit_1) if orbit_0_type == "RINEX_NAV" else (orbit_1, orbit_0)
        sp3_dcb = self.config.dcb_path_1 if orbit_1_type == "SP3" else self.config.dcb_path_0
        if self.config.compare_dcb:
            if datum.get("source") == "declared" and sp3_dcb is not None:
                # OSBs in the clock datum give k1 OSB(C2I) - k2 OSB(C6I) as the IF(B1I,B3I) offset (Bias-SINEX), so
                # the comparison should be at IF(B1I,B3I); not validated on data (no declaring OSB day in the
                # validation data) and the product used is not checked to be the declaring one when dcb_path is a list:
                # 'warning' until a graded row exists (N-ORB-24)
                self._bds_datum_conversion = {
                    "via": None,
                    "reason": (f"compare_dcb=True with the bias product declaring that datum ({datum.get('reason')}): "
                               "its OSBs should put the BDS-3 clocks at IF(B1I,B3I), but this path is not "
                               "validated on data; compare_dcb=False with a RINEX 4 navigation file (B-CNAV "
                               "group delays) is the validated conversion"),
                }
            else:
                self._bds_datum_conversion = {
                    "via": None,
                    "reason": ("compare_dcb=True with a bias product that does not declare the IF(B1C,B2a) datum: "
                               "give the OSB product of the SP3's analysis centre, or compare_dcb=False with a "
                               "RINEX 4 navigation file (B-CNAV group delays)"),
                }
            return orbit_0, orbit_1
        sv = pd.Index(sp3.index.get_level_values("sv")).astype(str).str[:3]
        gens = self.__dict__.get("_bds_generations") or {}
        bds3 = np.array([gens.get(s) == BDS3 for s in sv], dtype=bool)
        if "TGD1" not in nav.columns or not bds3.any():
            self._bds_datum_conversion = {
                "via": None,
                "reason": ("no broadcast group delay TGD1 in the navigation input" if bds3.any()
                           else "no BDS-3 satellite in the comparison"),
            }
            return orbit_0, orbit_1
        for col in ("TGD_B1Cp", "TGD_B2ap"):
            if col not in nav.columns:
                nav[col] = np.nan
        k1, _ = ionosphere_free_coefficients("B1I", "B3I")
        c1, c2 = ionosphere_free_coefficients("B1C", "B2a")
        value = {col: pd.to_numeric(nav[col], errors="coerce").to_numpy(float) for col in ("TGD1", "TGD_B1Cp",
                                                                                              "TGD_B2ap")}
        # the selected D1/D2 rows carry no B-CNAV fields: the B-CNAV group delays of the satellite from the
        # navigation file (median of its B-CNAV1/B-CNAV2 records, as the datum test of biases.detect_bds_clock_datum)
        bcnav = self._bds_bcnav_group_delays(self.config.orb_path_0 if orbit_0_type == "RINEX_NAV"
                                             else self.config.orb_path_1)
        for col in ("TGD_B1Cp", "TGD_B2ap"):
            fill = bcnav[col].reindex(sv).to_numpy(float) if col in bcnav else np.full(len(sv), np.nan)
            value[col] = np.where(np.isfinite(value[col]), value[col], fill)
        tgd = self.clight * (k1 * value["TGD1"] - (c1 * value["TGD_B1Cp"] - c2 * value["TGD_B2ap"]))
        usable = bds3 & np.isfinite(tgd)
        if not usable.any():
            self._bds_datum_conversion = {
                "via": None,
                "reason": ("no BDS-3 satellite of the comparison has B-CNAV group delays TGD_B1Cp/TGD_B2ap in the "
                           "navigation input (a RINEX 3 file has none); give a RINEX 4 navigation file"),
            }
            return orbit_0, orbit_1
        lacking = sorted(set(sv[bds3 & ~np.isfinite(tgd)]))
        sp3["TGD"] = np.where(bds3, tgd, pd.to_numeric(sp3["TGD"], errors="coerce").to_numpy(float))
        converted = sorted(set(sv[usable]))
        if lacking:
            self._warn_once(
                f"BeiDou B1IB3I: the BDS-3 precise clocks refer to IF(B1C,B2a), and the navigation input gives no "
                f"B-CNAV group delays TGD_B1Cp/TGD_B2ap for {', '.join(lacking)}: their clocks cannot be moved to "
                "IF(B1I,B3I), so their TGD, dTGD and sisre are NaN (sisre_notgd is kept).")
        self._bds_datum_conversion = {
            "via": "b-cnav",
            "converted": converted,
            "lacking": lacking,
            "note": (f"the precise group delay of {len(converted)} BDS-3 satellites is c[k1 TGD1 - (k1' TGD_B1Cp - "
                     "k2' TGD_B2ap)] from the B-CNAV broadcast group delays, so their clocks are compared at "
                     "IF(B1I,B3I)" + (f"; {', '.join(lacking)} without B-CNAV group delays: NaN" if lacking else "")
                     + "."),
        }
        return orbit_0, orbit_1

    @staticmethod
    def _bcnav_medians(nav: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Per BeiDou satellite the median TGD_B1Cp and TGD_B2ap [s] of the B-CNAV1/B-CNAV2 records (RINEX 4
        ``CNV1``/``CNV2``) of a navigation frame indexed by ``sv``; an empty frame without them (RINEX 3)."""
        empty = pd.DataFrame(columns=["TGD_B1Cp", "TGD_B2ap"], dtype=float)
        if nav is None or nav.empty or "nav_message" not in nav.columns \
                or not {"TGD_B1Cp", "TGD_B2ap"} <= set(nav.columns) or "sv" not in (nav.index.names or ()):
            return empty
        rows = nav[nav["nav_message"].astype(str).isin(("CNV1", "CNV2")).to_numpy()]
        if rows.empty:
            return empty
        values = rows[["TGD_B1Cp", "TGD_B2ap"]].apply(pd.to_numeric, errors="coerce")
        sv = pd.Index(rows.index.get_level_values("sv")).astype(str).str[:3]
        return values.groupby(sv.to_numpy()).median()

    def _bds_bcnav_group_delays(self, nav_path) -> pd.DataFrame:
        """B-CNAV group delays per BeiDou satellite (`_bcnav_medians`) of the navigation input: those kept by
        `process_broadcast_orbit` in this run, else read from `nav_path`."""
        kept = self.__dict__.get("_bds_bcnav_tgd")
        if kept is not None:
            return kept
        from ..io.rinex_nav import RinexNavError, read_navigation

        try:
            nav = read_navigation(str(nav_path), "C").bds
        except (OSError, ValueError, RinexNavError):
            return self._bcnav_medians(None)
        return self._bcnav_medians(nav)

    def _bds_generation_keys(self, compare: pd.DataFrame) -> np.ndarray:
        """Generation of every row's BeiDou satellite (`_bds_generations` of the run, else from the ANTEX/PRN at the
        first epoch)."""
        sv = pd.Index(compare.index.get_level_values("sv")).astype(str).str[:3]
        gens = self.__dict__.get("_bds_generations")
        if not gens or not set(sv) <= set(gens):
            t = compare.index.get_level_values("time")
            gens = {**(gens or {}), **self._bds_generations_of(set(sv), t.min() if len(t) else None,
                                                                cnav_satellites=self._bds_cnav_satellites(compare))}
        return np.array([gens.get(s, "") for s in sv], dtype=object)

    def _annotate_clock_convention(
        self,
        compared: pd.DataFrame,
        orbit_0_type: str,
        orbit_1_type: str,
    ) -> pd.DataFrame:
        """
        Attach BDS clock-convention metadata to a comparison result.

        Status:
            Active output annotation.

        The method writes both DataFrame attributes and explicit columns so that
        downstream code, CSV exports and tests can see the same status. It does
        not modify orbit, clock, TGD or SISRE values.
        """

        policy = self._bds_clock_convention_policy(orbit_0_type, orbit_1_type)
        if self.system != "C":
            return compared

        out = compared.copy()
        for key, value in policy.items():
            out.attrs[f"clock_convention_{key}"] = value
        out["clock_convention_status"] = policy["status"]
        out["clock_convention_mode"] = policy["mode"]
        out["clock_convention_directly_comparable"] = bool(policy["directly_comparable"])
        out["clock_convention_note"] = policy["note"]

        if policy["status"] != "ok":
            warnings.warn(
                f"BeiDou SIS clock convention status {policy['status']} for mode {policy['mode']}: {policy['note']}",
                RuntimeWarning,
                stacklevel=2,
            )
        return out

    def eclipse_periods(self, orbit:pd.DataFrame):
        """
                Identify satellite eclipse periods and optionally extend masks around ingress/egress.

                Returns:
                    pd.DataFrame: df with Per-satellite, per-epoch eclipse flags/masks usable
                    for filtering or modeling.

                Notes:
                    - The exact umbra/penumbra criteria follow project definitions.
                """

        # --- stałe ---------------------------------------------------------
        A_E = 6_378_137.0  # równikowy promień Ziemi [m]

        if not all(col in orbit.columns for col in ('xs', 'ys', 'zs')):
            # the Sun position comes with the satellite PCO step only (sat_fixed_system): without PCO (or for
            # NAV x NAV) compute it here, once per epoch, so the flags do not depend on the PCO switch (ORB-005)
            t = orbit.index.get_level_values('time')
            sun = get_sun_ecef(pd.DatetimeIndex(t.unique()))
            orbit[['xs', 'ys', 'zs']] = sun[['x', 'y', 'z']].reindex(pd.DatetimeIndex(t)).to_numpy(float)

        # --- wektory -------------------------------------------------------
        r_sat = orbit[['x', 'y', 'z']].to_numpy()  # (N,3)
        r_sun = orbit[['xs', 'ys', 'zs']].to_numpy()  # (N,3)

        norm_sat = np.linalg.norm(r_sat, axis=1)
        norm_sun = np.linalg.norm(r_sun, axis=1)

        cos_phi = (r_sat * r_sun).sum(1) / (norm_sat * norm_sun)

        # --- warunki Navipedia --------------------------------------------
        cond1 = cos_phi < 0  # (za Ziemią)
        # clipped: at cos_phi = -1 (satellite exactly opposite the Sun) rounding gave sqrt(-2e-16) = NaN, no flag
        cond2 = norm_sat * np.sqrt(np.clip(1 - cos_phi ** 2, 0.0, None)) < A_E  # (w cylindrze)

        orbit['eclipse_raw'] = cond1 & cond2  # flaga "czysty" cień

        def extend_post_eclipse(group, minutes=30):
            """Extend a per-satellite eclipse mask for a fixed number of minutes."""

            t = group.index.get_level_values('time')
            eclipse = group['eclipse_raw'].to_numpy()
            shadow = eclipse.copy()

            last_exit = None
            for i, is_ecl in enumerate(eclipse):
                if is_ecl:
                    last_exit = t[i]  # cały czas w cieniu
                elif last_exit is not None and (t[i] - last_exit) <= pd.Timedelta(minutes=minutes):
                    shadow[i] = True  # +30 min po wyjściu
                else:
                    last_exit = None

            group['eclipse'] = shadow
            return group
        if self.config.extend_eclipse:
            orbit = orbit.groupby(level='sv', group_keys=False).apply(extend_post_eclipse,minutes=self.config.extension_time)
        return orbit

    def _apply_prn_change_rule(self, orbit_0: pd.DataFrame, orbit_1: pd.DataFrame):
        """The PRN-change rule for every comparison with an ANTEX (`atx_path`), with or without the
        satellite PCO: a PRN whose satellite (SVN) of the first common epoch has no ANTEX entry from an epoch
        before the last one on (``antex.build_antenna_model``, ``prn_changes``: a PRN reassigned to another
        satellite) leaves both inputs after that epoch, with one UserWarning per run naming the PRN, the SVNs and
        the epoch. Without `atx_path` nothing is checked (the orbit products give no SVN)."""
        if not self.config.atx_path or orbit_0 is None or len(orbit_0) == 0:
            return orbit_0, orbit_1
        times = orbit_0.index.get_level_values('time')
        antenna = build_antenna_model(self.config.atx_path, None, times.min(), (self.system,),
                                      missing_policy=self.config.missing_sat_antenna, epoch_end=times.max(),
                                      notes_owner=self.config)
        return antenna.drop_after_prn_change(orbit_0), antenna.drop_after_prn_change(orbit_1)

    def apply_satellite_pco(self, precise_orbit:pd.DataFrame, pco_mode=None, apc: str = "mode", sp3_path=None):
        """
                Apply the satellite antenna PCO to the precise (centre-of-mass) coordinates.

                The ANTEX is read by ``gnx_py.antex`` (``build_antenna_model``, cached per file): satellite
                entries valid at the first epoch of the orbit (PRN within VALID FROM/UNTIL), PCO of the
                mode's signal or IF combination by its ANTEX frequency (GPS G05 from G01 for
                frequency-independent entries, with a warning). SIS compares orbits, so the PCV (a range
                term along the line of sight) does not enter.

                Args:
                    pco_mode: Galileo signal or IF mode whose ANTEX PCO is applied;
                        ``None`` (default) takes the configured `gal_mode`.
                    apc: ``"mode"`` (default) the APC of the mode; ``"broadcast"`` the APC a
                        broadcast orbit refers to (GPS: IF(L1,L2), BeiDou: B3I, whatever the mode,
                        with the column `dR_apc_signal` = outward radial of APC(mode) -
                        APC(reference); ``DDPreprocessing.sat_fixed_system``).
                    sp3_path: the SP3 file(s) of the orbit; its 'PCV:' label is checked against
                        the ANTEX (one ``AntexVersionWarning`` when they differ).

                Returns:
                    pd.DataFrame: Orbit data with the APC coordinates (`x_apc`, `y_apc`, `z_apc`) and the
                    Sun position (`xs`, `ys`, `zs`) per satellite and epoch.

                PRN changes:
                    A PRN whose ANTEX entry valid at the first epoch ends before the last epoch (a PRN
                    reassigned to another satellite) gives one UserWarning
                    naming the PRN, the SVNs and the change epoch; its rows after that epoch are dropped.

                Missing satellites:
                    A satellite without a valid ANTEX entry for a frequency of the mode follows
                    `missing_sat_antenna`: "exclude" (default) drops it with one UserWarning naming the
                    satellites, the ANTEX file and the date, so it leaves the comparison (common
                    satellite/epoch pairs) and the epoch clock datum `dt_mean`; "raise" raises ValueError.
                """
        times = precise_orbit.index.get_level_values('time')
        antenna = build_antenna_model(self.config.atx_path, None, times.min() if len(times) else None,
                                      (self.system,), missing_policy=self.config.missing_sat_antenna,
                                      sp3_path=sp3_path, epoch_end=times.max() if len(times) else None,
                                      notes_owner=self.config)
        # a PRN whose satellite has no ANTEX entry from an epoch within the comparison on (reassigned to another
        # satellite): its rows after that epoch leave the comparison (one UserWarning per run; D9). run() applies
        # the same rule to both inputs whenever atx_path is given (`_apply_prn_change_rule`)
        precise_orbit = antenna.drop_after_prn_change(precise_orbit)
        # SIS never uses the troposphere: day_of_year=None (the day of the orbit) gives no mismatch warning
        cfg = DefaultConfig(gps_freq=self.config.gps_mode, gal_freq=pco_mode or self.config.gal_mode,
                            bds_freq=self.config.bds_mode, day_of_year=None,
                            satellite_attitude=self.config.satellite_attitude,
                            attitude_laws=self.config.attitude_laws, attitude_path=self.config.attitude_path)
        cfg.missing_sat_antenna = self.config.missing_sat_antenna
        # the analysis centre of the SP3 input decides whether the ORBEX of attitude_path is used
        cfg.sp3_path = sp3_path
        processor = DDPreprocessing(df=precise_orbit, flh=None, xyz=np.array([1e7, 1e7, 1e7]),
                                        phase_shift_dict=None, sat_pco=None, rec_pco=None, antenna_h=None,
                                        system=self.system, config=cfg, antenna=antenna)
        precise_orbit[['xe', 'ye', 'ze']] = precise_orbit[['x', 'y', 'z']]
        processor.sat_fixed_system(apc=apc)
        precise_apc = processor.df.copy().swaplevel()
        return precise_apc


    def compare_orbits(self, orbit_0:pd.DataFrame, orbit_1:pd.DataFrame, kr, kb, kac, krb, clock_bias=True, clock_bias_function='mean'):
        """
        Compare two orbit/clock solutions on shared satellite/epoch pairs.

        Status:
            Active numerical core. Do not change the RAC/SISRE equations without
            cross-system regression tests.

        Inputs:
            orbit_0, orbit_1:
                DataFrames indexed by `sv,time`, containing APC coordinates
                (`x_apc`, `y_apc`, `z_apc`), raw coordinates (`x`, `y`, `z`),
                velocity (`vx`, `vy`, `vz`), clock (`clk`) and group delay
                (`TGD`). `orbit_1` or `orbit_0` should also carry Sun
                coordinates (`xs`, `ys`, `zs`) when eclipse flags are requested.
            kr, kb, kac, krb:
                Coefficients of the SISRE quadratic form
                ``sisre^2 = kr*dR^2 + kb*dt''^2 + kac*(dA^2 + dC^2) + krb*dR*dt''``
                with ``dR`` positive outward and ``dt''`` the datum- and TGD-reduced
                clock difference in meters. Montenbruck et al. (2015) correspond to
                ``kr = w_R^2, kb = 1, kac = w_AC^2, krb = -2*w_R``. ``None`` (or the
                legacy set `DEFAULT_SISRE_COEFFICIENTS`) selects the per-satellite
                literature weights (`SISRE_WEIGHTS`, `BDS_SISRE_ALPHA_BETA`).
            clock_bias:
                If true, remove a common epoch clock and TGD datum before SISRE
                calculation.

        Outputs:
            pandas.DataFrame: Original comparison table plus `dx`, `dy`, `dz`,
            clock error `dt` in meters, group-delay error `dTGD`, RAC components
            (`dR`, `dA`, `dC`), `sisre`, `sisre_orb` and `sisre_notgd`
            (`SISSession.run` adds `dR_apc_signal` for GPS modes other than
            L1L2, BeiDou modes other than B3I and Galileo E5aE5b broadcast-vs-SP3
            with satellite PCO, see `SISConfig`). All
            differences are orbit_0 minus orbit_1. RAC axes: R = r/|r| (outward),
            C = r x v/|r x v| (normal of the inertial orbit plane, v = Earth-fixed
            velocity + w_E x r), A = C x R (along-track). `dt_mean` is the epoch
            clock datum (all satellites with a clock, used by `sisre_notgd`; BeiDou
            per epoch and satellite generation, `_bds_generation_clock_datum`);
            `dTGD_mean` is set so that ``dt'' = (dt - dt_mean) - (dTGD - dTGD_mean)``
            is reduced by the clock and group-delay datums of the satellites
            having both (it equals the plain group-delay datum when none is
            missing).

        Units:
            Positions and RAC components are meters. Clock differences are
            converted from seconds to meters with the speed of light.
        """
        _logger.debug("SIS compare orbit_0 columns=%s", list(orbit_0.columns))
        _logger.debug("SIS compare orbit_1 columns=%s", list(orbit_1.columns))
        orbit_0[['dx', 'dy', 'dz', 'dt','dTGD']] = (
                    orbit_0[['x_apc', 'y_apc', 'z_apc', 'clk','TGD']] - orbit_1[['x_apc', 'y_apc', 'z_apc', 'clk','TGD']]).values
        orbit_0['clk_0'] = orbit_0['clk']
        orbit_0['clk_1'] = orbit_1['clk']
        compare = orbit_0.copy()
        compare['dt']*=self.clight
        compare = self._attach_sisre_coefficients(compare, kr=kr, kb=kb, kac=kac, krb=krb)
        if all(col in orbit_1.columns for col in ['xs', 'ys', 'zs']):
            compare[['xs', 'ys', 'zs']] = orbit_1[['xs', 'ys', 'zs']].copy()
        elif all(col in orbit_0.columns for col in ['xs', 'ys', 'zs']):
            compare[['xs', 'ys', 'zs']] = orbit_0[['xs', 'ys', 'zs']].copy()

        # RAC frame of the orbit_0 satellite: R outward (the convention of the SISRE
        # literature, so that a positive dR lengthens the modelled range), C along the
        # orbit normal r x v, A = C x R completing the right-handed triad. The orbit plane
        # is an inertial one: vx/vy/vz are Earth-fixed, so the Earth rotation w_E x r is
        # added back (without it C tilts by 20-30 deg for MEO, up to 65 deg for BDS IGSO).
        r = compare[['x_apc', 'y_apc', 'z_apc']].to_numpy(float)
        v = compare[['vx', 'vy', 'vz']].to_numpy(float)
        v = v + np.cross(np.array([0.0, 0.0, self._OMEGA_E]), r)
        e_r = r / np.linalg.norm(r, axis=1, keepdims=True)
        h = np.cross(r, v)
        e_c = h / np.linalg.norm(h, axis=1, keepdims=True)
        e_a = np.cross(e_c, e_r)
        dxyz = compare[['dx', 'dy', 'dz']].to_numpy(float)
        compare['dR'] = np.sum(e_r * dxyz, axis=1)
        compare['dA'] = np.sum(e_a * dxyz, axis=1)
        compare['dC'] = np.sum(e_c * dxyz, axis=1)
        if clock_bias and self.system == 'C':
            compare = self._bds_generation_clock_datum(compare, clock_bias_function)
        elif clock_bias:
            # Clock datum of sisre_notgd: every satellite with a clock difference.
            compare['dt_mean'] = compare['dt'].groupby(level='time').transform(clock_bias_function)
            # Datum of the TGD-reduced clock dt'' (sisre): clock and group-delay means over the
            # same satellites, otherwise a satellite lacking one of them enters one mean only
            # and shifts dt'' of every satellite at that epoch. dTGD_mean absorbs the offset
            # between the two clock means, so dt'' = (dt - dt_mean) - (dTGD - dTGD_mean)
            # still holds; with no missing values it is the plain group-delay datum.
            valid = compare['dt'].notna() & compare['dTGD'].notna()
            joint = compare.loc[valid, ['dt', 'dTGD']].groupby(level='time').agg(clock_bias_function)
            joint = joint.reindex(compare.index.get_level_values('time'))
            compare['dTGD_mean'] = (joint['dTGD'].to_numpy(float)
                                    + compare['dt_mean'].to_numpy(float) - joint['dt'].to_numpy(float))
        else:
            compare['dt_mean'] = 0.0
            compare['dTGD_mean'] = 0.0

        # SISRE (Montenbruck et al. 2015, 2018): the range error averaged over the visible
        # Earth, sisre^2 = (w_R*dR - dt'')^2 + w_AC^2*(dA^2 + dC^2), written as the quadratic
        # form kr*dR^2 + kb*dt''^2 + kac*(dA^2 + dC^2) + krb*dR*dt''. The radial error and the
        # clock error enter the range with opposite signs, hence krb = -2*w_R with dR outward.
        dR = compare['dR'].to_numpy(float)
        ac2 = (compare['dA'] ** 2 + compare['dC'] ** 2).to_numpy(float)
        dt = (compare['dt'] - compare['dt_mean']).to_numpy(float)
        dt_tgd = dt - (compare['dTGD'] - compare['dTGD_mean']).to_numpy(float)
        kr_i = compare['_sisre_kr'].to_numpy(float)
        kb_i = compare['_sisre_kb'].to_numpy(float)
        kac_i = compare['_sisre_kac'].to_numpy(float)
        krb_i = compare['_sisre_krb'].to_numpy(float)

        def _sisre(dclk):
            terms = kr_i * dR ** 2 + kb_i * dclk ** 2 + kac_i * ac2
            rad = terms + krb_i * dR * dclk
            # the literature form is a perfect square in (dR, dclk): cancellation may leave
            # a radicand of -1e-16 relative, which is zero, not an invalid weight set
            rad = np.where((rad < 0) & (rad >= -1e-12 * np.abs(terms)), 0.0, rad)
            return np.sqrt(rad)

        compare['sisre'] = _sisre(dt_tgd)
        compare['sisre_orb'] = np.sqrt(kr_i * dR ** 2 + kac_i * ac2)
        compare['sisre_notgd'] = _sisre(dt)
        return compare

    def _bds_generation_clock_datum(self, compare: pd.DataFrame, clock_bias_function) -> pd.DataFrame:
        """`dt_mean` and `dTGD_mean` of BeiDou per epoch AND satellite generation. BDS-2 and BDS-3 have independent clock datums -- the broadcast BDS-2 and BDS-3 clocks
        against one precise product differ by a common offset of about 6 m (2024-035) -- which one mean over both
        generations left in every satellite's `dt''`; each generation is now reduced by its own epoch datum, as
        the clock ensemble a SISRE clock term is free of. The generation comes from the ANTEX block valid at the
        first epoch, else B-CNAV records, else the PRN (`_bds_generations_of`). A generation with fewer than
        `BDS_CLOCK_DATUM_MIN_SATS` satellites at an epoch gets no datum there (NaN, one RuntimeWarning)."""
        time = compare.index.get_level_values('time')
        gen = self._bds_generation_keys(compare)
        keys = [time, gen]
        dt = compare['dt']
        n_dt = dt.notna().groupby(keys).transform('sum').to_numpy()
        dt_mean = dt.groupby(keys).transform(clock_bias_function).to_numpy(float)
        enough_dt = n_dt >= self.BDS_CLOCK_DATUM_MIN_SATS
        compare['dt_mean'] = np.where(enough_dt, dt_mean, np.nan)
        valid = (dt.notna() & compare['dTGD'].notna()).to_numpy()
        joint = compare.loc[valid, ['dt', 'dTGD']].groupby([time[valid], gen[valid]]).agg(clock_bias_function)
        n_joint = pd.Series(valid, index=compare.index).groupby(keys).transform('sum').to_numpy()
        at = joint.reindex(pd.MultiIndex.from_arrays([time, gen]))
        dtgd_mean = (at['dTGD'].to_numpy(float) + dt_mean - at['dt'].to_numpy(float))
        compare['dTGD_mean'] = np.where(n_joint >= self.BDS_CLOCK_DATUM_MIN_SATS, dtgd_mean, np.nan)
        short = (~enough_dt & dt.notna().to_numpy()) | ((n_joint < self.BDS_CLOCK_DATUM_MIN_SATS) & valid)
        if short.any():
            sv = pd.Index(compare.index.get_level_values('sv')).astype(str).str[:3]
            labels = sorted({f"{g or 'unknown'}: {s}" for g, s in zip(gen[short], sv[short])})
            self._warn_once(
                f"BeiDou: fewer than {self.BDS_CLOCK_DATUM_MIN_SATS} satellites of one generation with a clock "
                f"difference at {pd.Index(time[short]).nunique()} epochs ({', '.join(labels)}): the clock datum "
                "is taken per generation (BDS-2 and BDS-3 clocks have independent datums), so these rows have no "
                "dt_mean/dTGD_mean and their sisre and sisre_notgd are NaN there.")
        return compare

    def _antex_block(self, sv: str, epoch) -> Optional[str]:
        """ANTEX block (TYPE) of ``sv`` valid at ``epoch`` from `atx_path`, or None without one (since v1.0.0
        the BeiDou orbit type of the SISRE weights follows the satellite)."""
        if not self.config.atx_path:
            return None
        from ..models.antex import load_antex

        try:
            entry = load_antex(self.config.atx_path).satellite(
                sv, None if epoch is None or pd.isna(epoch) else pd.Timestamp(epoch).to_pydatetime())
        except (OSError, ValueError):
            return None
        return entry.type.strip() if entry is not None else None

    #: geocentric radius [m] above which a BeiDou orbit is geosynchronous (GEO/IGSO 42 164 km, MEO 27 906 km), and
    #: Earth-fixed speed [m/s] above which a geosynchronous one is IGSO (GEO about 0, IGSO 55 deg 1.3-2.8 km/s)
    BDS_GEOSYNC_RADIUS_M = 35.0e6
    BDS_IGSO_MIN_ECEF_SPEED = 500.0

    def _bds_orbit_type_of_orbit(self, compare: pd.DataFrame, rows, sv: str) -> str:
        """Orbit type (GEO/IGSO/MEO) of BeiDou ``sv`` without an ANTEX block: from its orbit in ``compare`` (median
        geocentric radius and Earth-fixed speed of the rows ``rows``), so it holds after the PRN reassignment of
        2026-04 (C06-C08 BDS-3 IGSO, C13/C38-C40 MEO), where the PRN table of ``gnss.bds_orbit_type`` is wrong.
        The PRN table when the rows carry no position/velocity."""
        pos = [c for c in (("x_apc", "y_apc", "z_apc"), ("x", "y", "z")) if set(c) <= set(compare.columns)]
        if not pos or not {"vx", "vy", "vz"} <= set(compare.columns):
            return bds_orbit_type(sv)
        sub = compare.loc[rows]
        r = np.linalg.norm(sub[list(pos[0])].to_numpy(float), axis=1)
        v = np.linalg.norm(sub[["vx", "vy", "vz"]].to_numpy(float), axis=1)
        r, v = np.nanmedian(r) if np.isfinite(r).any() else np.nan, np.nanmedian(v) if np.isfinite(v).any() else np.nan
        if not (np.isfinite(r) and np.isfinite(v)):
            return bds_orbit_type(sv)
        if r < self.BDS_GEOSYNC_RADIUS_M:
            return "MEO"
        return "IGSO" if v > self.BDS_IGSO_MIN_ECEF_SPEED else "GEO"

    def _attach_sisre_coefficients(self, compare: pd.DataFrame, kr, kb, kac, krb) -> pd.DataFrame:
        """
        Attach SISRE weighting coefficients to each comparison row.

        Status:
            Active. Coefficients given by the caller are copied to every row.
            A coefficient left as ``None`` (or the whole legacy default set
            `DEFAULT_SISRE_COEFFICIENTS`) takes the per-satellite literature
            value: ``kr = w_R^2``, ``kb = 1``, ``kac = w_AC^2``, ``krb = -2*w_R``
            with ``(w_R, w_AC^2)`` from `SISRE_WEIGHTS` (GPS, GLONASS, Galileo,
            QZSS) or `BDS_SISRE_ALPHA_BETA` (BeiDou MEO/IGSO).

        Raises:
            ValueError: If a BDS GEO satellite (or a system without literature
            weights) needs default weights; GEO SISRE is not validated here.
        """

        given = {"kr": kr, "kb": kb, "kac": kac, "krb": krb}
        if all(given[k] is not None and np.isclose(float(given[k]), DEFAULT_SISRE_COEFFICIENTS[k], rtol=1e-6, atol=0.0)
               for k in given):
            # The former defaults (GPS-only weights, applied to every system) are read as
            # "literature weights": they were the default of a formula that is now fixed.
            warnings.warn(
                "SISRE coefficients kr, kb, kac, krb equal the former defaults "
                f"{tuple(DEFAULT_SISRE_COEFFICIENTS.values())}; the per-satellite literature weights "
                "(Montenbruck et al. 2015) are used instead. Leave them as None (the default) to "
                "select these weights without this warning.",
                FutureWarning,
                stacklevel=3,
            )
            given = dict.fromkeys(given)
        for key, value in given.items():
            compare[f'_sisre_{key}'] = np.nan if value is None else float(value)
        missing = [k for k, v in given.items() if v is None]
        if not missing:
            return compare

        sv_index = compare.index.get_level_values("sv")
        for sv in sv_index.unique():
            sys_code = str(sv)[0]
            if sys_code == "C":
                block = self._antex_block(sv, compare.index.get_level_values("time").min())
                orbit_type = (self._bds_orbit_type_of_orbit(compare, sv_index == sv, sv) if block is None
                              else bds_orbit_type(sv, block=block))
                if orbit_type == "GEO":
                    raise ValueError(
                        f"BeiDou GEO satellite {sv} is not supported by the SIS/SISRE orbit comparison path."
                    )
                w_r, beta = BDS_SISRE_ALPHA_BETA[orbit_type]
                w_ac2 = 1.0 / beta
            elif sys_code in SISRE_WEIGHTS:
                w_r, w_ac2 = SISRE_WEIGHTS[sys_code]
            else:
                raise ValueError(f"No default SISRE weights for satellite {sv}; pass kr, kb, kac and krb explicitly.")
            defaults = {"kr": w_r ** 2, "kb": 1.0, "kac": w_ac2, "krb": -2.0 * w_r}
            mask = sv_index == sv
            for key in missing:
                compare.loc[mask, f'_sisre_{key}'] = defaults[key]
        return compare

        # --- GŁÓWNA METODA PUBLICZNA ---
    def process_dcb_and_merge(self, orbit_0, orbit_1, orbit_0_type, orbit_1_type):
        """
        Ingest DCB/OSB resources, convert to appropriate group delays if needed,
        and merge them with the orbit/measurement data.

        Status:
            Active bias/clock-datum policy. Requires caution for BDS because the
            selected signal mode must match the broadcast fields and any
            external OSB/DSB product.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame): Orbit/measurement table with per-satellite delay fields merged.

        Raises:
            ValueError: With `compare_dcb=True`, if an SP3 input has no bias file
                (`dcb_path_0`/`dcb_path_1`); NAV inputs use their broadcast group delays.
                Also if a bias product has no record of the system, lacks the codes of
                the signal mode, or has no record valid at the compared epochs.
            RuntimeError: If DCB parsing or conversion fails.

        Warns:
            RuntimeWarning: Once per run and product (from `run()`, after the common
                epochs are set), naming the compared satellites (and epochs) for which the
                bias product gives no group delay; their `TGD` is NaN.
        """

        self._warned_messages = set()
        self._missing_tgd = {}
        self._bias_products_read = {}
        if not self.config.compare_dcb:
            if self.system == 'C':
                orbit_0 = self._merge_bds_broadcast_tgd_without_external_dcb(orbit_0, orbit_0_type)
                orbit_1 = self._merge_bds_broadcast_tgd_without_external_dcb(orbit_1, orbit_1_type)
                return orbit_0, orbit_1
            if (self.system == 'E' and self.config.gal_mode in GAL_INAV_CLOCK_MODES
                    and {orbit_0_type, orbit_1_type} == {'RINEX_NAV', 'SP3'}):
                if self.config.gal_mode in ('E1', 'E5b'):
                    remedy = ("compare_dcb=True with a bias product giving C1C-C5Q and C1C-C7Q "
                              "(DSB, or OSB C1C/C5Q/C7Q) removes it.")
                else:
                    remedy = (f"compare_dcb is not supported for {self.config.gal_mode} in this version, "
                              "so the offset cannot be removed here.")
                self._warn_once(
                    f"Galileo {self.config.gal_mode} uses the I/NAV clock, referred to IF(E1,E5b), while precise "
                    "clocks refer to IF(E1,E5a): the per-satellite difference BGD(E1,E5a) - BGD(E1,E5b) "
                    "(typically 0.2 m, up to about 0.6 m) stays in dt and sisre; " + remedy)
            orbit_0['TGD'] = 0.0
            orbit_1['TGD'] = 0.0
            return orbit_0, orbit_1
        # every SP3 input has its bias product (a NAV input uses its broadcast group delays,
        # converted to metres like the external ones), otherwise a readable error
        self._check_dcb_paths(orbit_0_type, orbit_1_type)
        if self.system == 'G':
            dcb_0, type_dcb_0 = self._get_gps_dcb(orbit_0, orbit_0_type, self.config.dcb_path_0)
            dcb_1, type_dcb_1 = self._get_gps_dcb(orbit_1, orbit_1_type, self.config.dcb_path_1)
            dcb_0 = self._convert_gps_dcb_to_tgd(dcb_0, type_dcb_0)
            dcb_1 = self._convert_gps_dcb_to_tgd(dcb_1, type_dcb_1)
            orbit_0 = self._merge_tgd_with_orbit(orbit_0, dcb_0)
            orbit_1 = self._merge_tgd_with_orbit(orbit_1, dcb_1)
        elif self.system == 'E':
            gal_field = self._get_gal_field()
            dcb_0, type_dcb_0 = self._get_gal_dcb(orbit_0, orbit_0_type, self.config.dcb_path_0, gal_field)
            dcb_1, type_dcb_1 = self._get_gal_dcb(orbit_1, orbit_1_type, self.config.dcb_path_1, gal_field)
            dcb_0_out = self._convert_gal_dcb_to_tgd(dcb_0, type_dcb_0)
            dcb_1_out = self._convert_gal_dcb_to_tgd(dcb_1, type_dcb_1)
            orbit_0 = self._merge_gal_tgd_with_orbit(orbit_0, dcb_0_out, type_dcb_0)
            orbit_1 = self._merge_gal_tgd_with_orbit(orbit_1, dcb_1_out, type_dcb_1)
        elif self.system == 'C':
            bds_field = self._get_bds_field()
            dcb_0, type_dcb_0 = self._get_bds_dcb(orbit_0, orbit_0_type, self.config.dcb_path_0, bds_field)
            dcb_1, type_dcb_1 = self._get_bds_dcb(orbit_1, orbit_1_type, self.config.dcb_path_1, bds_field)
            dcb_0_out = self._convert_bds_dcb_to_tgd(dcb_0, type_dcb_0)
            dcb_1_out = self._convert_bds_dcb_to_tgd(dcb_1, type_dcb_1)
            orbit_0 = self._merge_gal_tgd_with_orbit(orbit_0, dcb_0_out, type_dcb_0)
            _logger.debug("SIS BDS orbit_0 columns after TGD merge=%s", list(orbit_0.columns))
            orbit_1 = self._merge_gal_tgd_with_orbit(orbit_1, dcb_1_out, type_dcb_1)
            _logger.debug("SIS BDS orbit_1 columns after TGD merge=%s", list(orbit_1.columns))
        else:
            raise ValueError(f"Unsupported sys: {self.system}")
        return orbit_0, orbit_1

    # --- POMOCNICZE METODY DLA GPS ---
    def _get_gps_dcb(self, orbit, orbit_type, dcb_path):
        """
        Load or resolve GPS DCB/OSB resource from the provided source.

        Returns:
            Any: Parsed DCB/OSB structure suitable for conversion/merging.
        """

        if dcb_path is not None and orbit_type != 'RINEX_NAV':
            return self._read_bias_product_at_orbit(orbit, dcb_path)
        elif orbit_type == 'RINEX_NAV':
            dcb, = orbit.pop('TGD').to_frame() * self.clight,
            return dcb, 'brdc'
        else:
            return None, None

    def _convert_gps_dcb_to_tgd(self, dcb, type_dcb):
        """
        Convert GPS DCB/OSB data to the broadcast TGD convention, in metres.

        IS-GPS-200 (20.3.3.3.3.2): T_GD = (t_L1P - t_L2P) / (1 - gamma), gamma = (f_L1/f_L2)^2,
        so a P1-P2 (C1W-C2W) code bias gives TGD = -DSB / (gamma - 1) = -1.5457 * DSB. With
        OSBs the DSB is OSB_C1W - OSB_C2W; in the IF(C1W,C2W) datum of IGS clocks (the LNAV
        clock datum as well) that TGD equals OSB_C1W. Both sides of the comparison therefore
        carry the L1 P(Y) group delay, ``clk - TGD`` being the L1 P(Y) clock, for every
        `gps_mode` (LNAV has no L5 group delay; ISC_L5 is a CNAV parameter, IS-GPS-705).

        Returns:
            table-like or dict: Per-satellite delay values aligned with the internal data model.
        """

        k2 = -1.0 / (gamma('L1', 'L2') - 1.0)
        if type_dcb in ('DSB', 'OSB'):
            self._require_bias_codes(dcb, type_dcb, [('C1W', 'C2W')])
            source = dcb
        if type_dcb == 'DSB':
            dcb = k2 * dcb[['BIAS_C1W_C2W']] * 1e-09 * self.clight
            dcb = self._product_tgd(dcb.rename(columns={'BIAS_C1W_C2W': 'TGD'}), source, 'DSB C1W-C2W')
        elif type_dcb == 'OSB':
            dcb = k2 * (dcb[['OSB_C1W']] - dcb[['OSB_C2W']].values) * 1e-09 * self.clight
            dcb = self._product_tgd(dcb.rename(columns={'OSB_C1W': 'TGD'}), source, 'OSB C1W, C2W')
        # brdc or None -> return as is
        return dcb

    # --- POMOCNICZE METODY DLA GALILEO ---
    def _get_gal_field(self):
        """
        Resolve a Galileo-specific configuration or data field.

        Returns:
            Any: Field value according to session/config state.
        """

        gal_mode_map = {
            'E1': 'BGDe5b',
            'E5b': 'BGDe5b',
            'E1E5b': 'BGDe5b',
            'E5a': 'BGDe5a',
            'E1E5a': 'BGDe5a'
        }
        gal_field = gal_mode_map.get(self.config.gal_mode)
        if gal_field is None:
            raise ValueError(f"Unsupported gal_mode: {self.config.gal_mode}")
        return gal_field

    def _get_gal_dcb(self, orbit, orbit_type, dcb_path, gal_field):
        """
        Load or resolve Galileo DCB/OSB resource from the provided source.

        Returns:
            Any: Parsed Galileo DCB/OSB structure.
        """

        if dcb_path is not None and orbit_type != 'RINEX_NAV':
            return self._read_bias_product_at_orbit(orbit, dcb_path)
        elif orbit_type == 'RINEX_NAV':
            dcb = orbit.pop(gal_field).to_frame() * self.clight
            dcb = dcb.rename(columns={gal_field: 'TGD'})
            return dcb, 'brdc'
        else:
            return None, None

    # Galileo BGD(E1,Ex) of a message: second signal and its precise-product code observable
    _GAL_BGD_SIGNAL = {'BGDe5a': 'E5a', 'BGDe5b': 'E5b'}
    _GAL_BGD_OBS = {'E5a': 'C5Q', 'E5b': 'C7Q'}

    def _gal_product_bgd(self, dcb, type_dcb, signal):
        """
        BGD(E1, `signal`) in metres from a DSB/OSB product, None when the product lacks it.

        OS SIS ICD 5.1.5: BGD(E1,Ex) = (t_E1 - t_Ex) / (1 - gamma), gamma = (f_E1/f_Ex)^2,
        so BGD = -DSB(C1C-Cx) / (gamma - 1): -1.2606*DSB for E1/E5a, -1.4220*DSB for E1/E5b.
        With OSBs the DSB is OSB_C1C - OSB_Cx.
        """
        if dcb is None:
            return None
        obs = self._GAL_BGD_OBS[signal]
        if type_dcb == 'DSB':
            col = f'BIAS_C1C_{obs}'
            if col not in dcb.columns:
                return None
            dsb = dcb[col]
        elif type_dcb == 'OSB':
            if 'OSB_C1C' not in dcb.columns or f'OSB_{obs}' not in dcb.columns:
                return None
            dsb = dcb['OSB_C1C'] - dcb[f'OSB_{obs}']
        else:
            return None
        return -dsb / (gamma('E1', signal) - 1.0) * 1e-9 * self.clight

    def _convert_gal_dcb_to_tgd(self, dcb, type_dcb):
        """
        Convert Galileo DCB/OSB data to the broadcast BGD convention, in metres.

        Both sides carry the BGD(E1,Ex) of the message the mode uses (message
        selection: E1/E5b -> I/NAV, BGD(E1,E5b); E5a/E1E5a -> F/NAV, BGD(E1,E5a)), so that
        ``clk - TGD`` is the E1 clock on both sides (OS SIS ICD 5.1.5).

        The I/NAV clock refers to IF(E1,E5b), the IGS/MGEX precise clock to IF(E1,E5a)
        (C1C/C5Q). For E1/E5b the product therefore also gives the shift of the precise
        clock to the I/NAV datum, IF(E1,E5b) - IF(E1,E5a) = BGD(E1,E5a) - BGD(E1,E5b), returned
        as column `_clk_datum` [s] and added to the precise clock on merge: without it the
        per-satellite offset (typically 0.2 m, up to 0.6 m) stays in `dt`, `sisre` and
        `sisre_notgd`. A product without the C1C-C5Q bias keeps the IF(E1,E5a) clock (warned).

        Raises:
            ValueError: If the product has no C1C-Cx bias of the message's BGD for Galileo
                (e.g. E1/E5b with an OSB product without C7Q), instead of a zero TGD.

        Returns:
            table-like or dict: Per-satellite delay values aligned with the internal data model.
        """

        mode = self.config.gal_mode
        if mode not in SIS_COMPARE_DCB_MODES['E']:
            raise ValueError(
                f"Unsupported gal_mode for compare_dcb: {mode!r} (supported: {', '.join(SIS_COMPARE_DCB_MODES['E'])}); "
                "E1E5b and E5aE5b are not enabled in this version (see SISConfig, compare_dcb); use "
                "compare_dcb=False for them")
        if type_dcb == 'brdc':
            return dcb  # broadcast BGD of the message, already in metres
        signal = self._GAL_BGD_SIGNAL[self._get_gal_field()]
        obs = self._GAL_BGD_OBS[signal]
        # a product without the codes of the message's BGD gives no Galileo group delay at all
        self._require_bias_codes(dcb, type_dcb, [('C1C', obs)])
        bgd = self._gal_product_bgd(dcb, type_dcb, signal)
        codes = f"DSB C1C-{obs}" if type_dcb == 'DSB' else f"OSB C1C, {obs}"
        out = self._product_tgd(bgd.to_frame(name='TGD'), dcb, codes)
        if signal == 'E5b':
            bgd_e5a = (self._gal_product_bgd(dcb, type_dcb, 'E5a')
                       if self._has_bias_codes(dcb, type_dcb, [('C1C', 'C5Q')]) else None)
            if bgd_e5a is None:
                self._warn_once(
                    f"Galileo {mode}: the bias product has no C1C-C5Q bias, so the precise clocks stay in "
                    "their IF(E1,E5a) datum while the I/NAV clock refers to IF(E1,E5b); the per-satellite "
                    "difference BGD(E1,E5a) - BGD(E1,E5b) (typically 0.2 m, up to about 0.6 m) remains in dt "
                    "and sisre.")
            else:
                out['_clk_datum'] = (bgd - bgd_e5a) / self.clight
        return out

    def _warn_once(self, message: str) -> None:
        """One RuntimeWarning per message and run (both inputs pass the same conversion)."""
        seen = self.__dict__.setdefault('_warned_messages', set())
        if message not in seen:
            seen.add(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    # --- POMOCNICZE METODY DLA BEIDOU ---
    def _get_bds_field(self):
        """
        Resolve the legacy BDS broadcast TGD field for the configured mode.

        Status:
            Active BDS compatibility helper.

        Modern BDS CNAV modes do not map to the legacy `TGD1`/`TGD2` fields, so
        this returns `None` for modern signals and lets the guarded RINEX 4 /
        OSB/DSB path handle them.
        """

        signals = mode_signals(self.config.bds_mode)
        if any(signal in BDS_MODERN_BROADCAST_SIGNALS for signal in signals):
            return None
        if signals == ("B3I",):
            return None
        if "B3I" in signals and "B1I" in signals:
            return "TGD1"
        if "B3I" in signals and "B2I" in signals:
            return "TGD2"
        if "B1I" in signals:
            return "TGD1"
        if "B2I" in signals:
            return "TGD2"
        raise ValueError(f"Unsupported bds_mode for broadcast group delay: {self.config.bds_mode}")

    def _bds_signal_tgd_seconds(self, orbit: pd.DataFrame, signal: str) -> pd.Series:
        """
        Read a signal-specific BDS broadcast group delay in seconds.

        Status:
            Active, requires validation for modern BDS.

        `B3I` is the legacy reference signal in the B1I/B3I datum and therefore
        contributes zero delay here. Modern signals require RINEX 4 CNAV
        signal-specific fields; missing or empty fields raise immediately.
        """

        zero = pd.Series(0.0, index=orbit.index)
        if signal == "B3I":
            return zero
        if signal == "B1I":
            field = "TGD1"
        elif signal == "B2I":
            field = "TGD2"
        elif signal == "B1C":
            field = "TGD_B1Cp"
        elif signal == "B2a":
            field = "TGD_B2ap"
        elif signal == "B2b":
            field = "TGD_B2bI"
        else:
            raise ValueError(
                f"Unsupported BeiDou broadcast group delay for signal {signal}: "
                "modern B1C/B2a/B2b need RINEX 4 CNAV ISC/TGD fields or validated OSB handling."
            )
        if field not in orbit.columns:
            raise ValueError(f"BeiDou broadcast orbit does not contain required {field} field.")
        values = orbit[field]
        if values.isna().all():
            raise ValueError(
                f"BeiDou broadcast orbit contains {field}, but it is empty for mode {self.config.bds_mode}. "
                "This mode needs fields from multiple RINEX 4 CNAV message types or a validated OSB/DSB product."
            )
        return values

    def _bds_broadcast_group_delay_to_clock_datum(self, orbit: pd.DataFrame) -> pd.DataFrame:
        """
        Convert BDS broadcast group delays to the internal meter-level TGD.

        Status:
            Active clock-datum helper, requires caution.

        The output is a DataFrame named `TGD` in meters. Single-signal modes use
        that signal's broadcast delay. Two-signal modes use the
        ionosphere-free coefficients from `mode_ionosphere_free_coefficients`.
        Modern modes are only accepted when the required RINEX 4 fields are
        present; otherwise the method raises instead of silently producing a
        misleading clock correction.
        """

        signals = mode_signals(self.config.bds_mode)
        unsupported_signals = set(signals) - BDS_LEGACY_BROADCAST_TGD_SIGNALS
        has_modern_broadcast_fields = bool({"TGD_B1Cp", "TGD_B2ap", "TGD_B2bI"}.intersection(orbit.columns))
        if unsupported_signals and not has_modern_broadcast_fields:
            raise ValueError(
                "Unsupported BeiDou broadcast clock/TGD mode "
                f"{self.config.bds_mode!r}: signals {sorted(unsupported_signals)} require "
                "RINEX 4 BDS CNAV signal-specific ISC/TGD fields or validated OSB handling."
            )
        if len(signals) == 1:
            tgd_seconds = self._bds_signal_tgd_seconds(orbit, signals[0])
        elif len(signals) == 2:
            a, b = mode_ionosphere_free_coefficients(self.config.bds_mode)
            tgd_seconds = (
                a * self._bds_signal_tgd_seconds(orbit, signals[0])
                - b * self._bds_signal_tgd_seconds(orbit, signals[1])
            )
        else:
            raise ValueError(f"Unsupported BeiDou mode for broadcast group delay: {self.config.bds_mode}")
        return tgd_seconds.to_frame(name="TGD") * self.clight

    def _merge_bds_broadcast_tgd_without_external_dcb(self, orbit: pd.DataFrame, orbit_type: str) -> pd.DataFrame:
        """
        Merge broadcast BDS TGD when no external bias product is configured.

        Status:
            Active fallback policy.

        Broadcast NAV inputs use the BDS TGD/ISC fields available in the orbit
        table. Precise products receive zero TGD so that later comparison code
        can operate on a uniform schema.
        """

        if orbit_type == 'RINEX_NAV':
            tgd = self._bds_broadcast_group_delay_to_clock_datum(orbit)
            return orbit.join(tgd)
        orbit['TGD'] = 0.0
        return orbit

    def _get_bds_dcb(self, orbit, orbit_type, dcb_path, bds_field):
        """
        Load or derive a BDS bias source for the selected orbit product.

        Status:
            Active BDS bias policy helper.

        External precise products read BIA/OSB/DSB data. Broadcast navigation
        inputs derive `TGD` from the NAV fields so they share the same meter
        convention as external products.
        """

        if dcb_path is not None and orbit_type != 'RINEX_NAV':
            return self._read_bias_product_at_orbit(orbit, dcb_path)
        if orbit_type == 'RINEX_NAV':
            return self._bds_broadcast_group_delay_to_clock_datum(orbit), 'brdc'
        return None, None

    @staticmethod
    def _bds_observation_candidates(signal: str) -> list[str]:
        """
        Return prioritized bias-product observable codes for a logical signal name.

        Status:
            Active helper used by OSB/DSB matching.

        B1C and B2a: pilot, data+pilot, data (`BDS_PRODUCT_OBSERVABLE_ORDER`, the observable of the
        broadcast TGD_B1Cp/TGD_B2ap); every other signal the tracking priority of `gnx_py.gnss`.
        """

        spec = signal_spec(signal)
        order = BDS_PRODUCT_OBSERVABLE_ORDER.get(signal, spec.suffix_priority)
        return [f"{spec.code_prefix}{suffix}" for suffix in order]

    @staticmethod
    def _available_bds_bias_observables(dcb: pd.DataFrame) -> set[str]:
        """
        Read advertised BDS OSB observables from a parsed BIA product.

        Status:
            Active metadata helper. Empty metadata is treated as unknown rather
            than as a hard failure, because older parsers may not populate attrs.
        """

        availability = dcb.attrs.get("bias_observables_by_system", {})
        values = availability.get("C", []) if isinstance(availability, dict) else []
        return set(values)

    @staticmethod
    def _available_bds_bias_pairs(dcb: pd.DataFrame) -> set[str]:
        """
        Read advertised BDS DSB observable pairs from a parsed BIA product.

        Status:
            Active metadata helper for pair selection and error reporting.
        """

        availability = dcb.attrs.get("bias_pairs_by_system", {})
        values = availability.get("C", []) if isinstance(availability, dict) else []
        return set(values)

    def _resolve_bds_osb_observable(self, dcb: pd.DataFrame, signal: str) -> str:
        """
        Select the best OSB observable column for a BDS signal.

        Status:
            Active, requires caution for mixed bias products.

        The resolver follows the observable suffix priority from `gnx_py.gnss`
        and checks both DataFrame columns and optional parser metadata. It raises
        a descriptive error when the requested signal is not represented.
        """

        available = self._available_bds_bias_observables(dcb)
        candidates = self._bds_observation_candidates(signal)
        for obs in candidates:
            if f"OSB_{obs}" in dcb.columns and (not available or obs in available):
                return obs
        raise ValueError(
            f"No BDS OSB observable found for signal {signal} in DCB/OSB product {dcb.attrs.get('bias_path', '')}. "
            f"Tried {candidates}; available BDS observables: {sorted(available) or 'unknown'}"
            f"{self._placeholder_note(dcb)}."
        )

    def _resolve_bds_dsb_pair(self, dcb: pd.DataFrame, signal1: str, signal2: str) -> tuple[str, str, str]:
        """
        Select a DSB observable pair for a two-signal BDS mode.

        Status:
            Active, requires caution.

        Returns:
            tuple: `(obs1, obs2, direction)`, where direction is `"direct"` or
            `"reverse"` depending on how the BIA product stores the pair.
        """

        available = self._available_bds_bias_pairs(dcb)
        candidates1 = self._bds_observation_candidates(signal1)
        candidates2 = self._bds_observation_candidates(signal2)
        pairs = [(obs1, obs2) for obs1 in candidates1 for obs2 in candidates2]
        if (signal1 in BDS_PRODUCT_OBSERVABLE_ORDER) != (signal2 in BDS_PRODUCT_OBSERVABLE_ORDER):
            # one signal of BDS_PRODUCT_OBSERVABLE_ORDER (B1C with B2b): the other signal's priority decides
            # first, so the pilot order changes only the B1C/B2a observable (N-ORB-19, decision D9 narrow) and
            # B1CB2b keeps its former CAS pair C1X-C7Z. Not a settled choice: TGD_B2bI refers to B2b-I, the data
            # component (RINEX 7D), so C7D may be the better B2b observable (open item, N-ORB-19 addendum)
            other = 1 if signal1 in BDS_PRODUCT_OBSERVABLE_ORDER else 0
            pairs.sort(key=lambda pair: (candidates2.index(pair[1]) if other else candidates1.index(pair[0])))
        for obs1, obs2 in pairs:  # stable sort: within a rank the former nested order
            direct = f"{obs1}_{obs2}"
            reverse = f"{obs2}_{obs1}"
            if f"BIAS_{direct}" in dcb.columns and (not available or direct in available):
                return obs1, obs2, "direct"
            if f"BIAS_{reverse}" in dcb.columns and (not available or reverse in available):
                return obs1, obs2, "reverse"
        raise ValueError(
            f"No BDS DSB pair found for mode {self.config.bds_mode} in DCB product {dcb.attrs.get('bias_path', '')}. "
            f"Tried {candidates1} x {candidates2}; available BDS pairs: {sorted(available) or 'unknown'}"
            f"{self._placeholder_note(dcb)}."
        )

    def _bds_bias_columns(self, dcb: pd.DataFrame) -> tuple[str, str]:
        """
        Return OSB observables used by a two-signal BDS mode.

        Status:
            Active helper retained for diagnostics and tests. Single-signal
            modes return empty strings because they do not form an IF pair.
        """

        signals = mode_signals(self.config.bds_mode)
        if len(signals) == 1:
            return "", ""
        return (
            self._resolve_bds_osb_observable(dcb, signals[0]),
            self._resolve_bds_osb_observable(dcb, signals[1]),
        )

    def _convert_bds_dcb_to_tgd(self, dcb, type_dcb):
        """
        Convert BDS OSB/DSB/BRDC bias data to the internal TGD convention.

        Status:
            Active BDS clock/bias conversion helper, requires caution.

        OSB inputs are converted from nanoseconds to meters for either a
        single-signal datum or an ionosphere-free combination (a*OSB_1 - b*OSB_2).
        DSB inputs give the same quantity through OSBs tied to the IF(C2I,C6I)
        clock datum (`_bds_osb_from_dsb`); they are accepted for two-signal
        modes only. Broadcast (`brdc`) data are already in the internal
        representation and are returned unchanged.
        """

        if type_dcb in {'brdc', None} or dcb is None:
            return dcb
        signals = mode_signals(self.config.bds_mode)
        if type_dcb == 'OSB':
            if len(signals) == 1:
                obs = self._resolve_bds_osb_observable(dcb, signals[0])
                col = f"OSB_{obs}"
                out = dcb[[col]] * 1e-9 * self.clight
                return self._product_tgd(out.rename(columns={col: 'TGD'}), dcb, f"OSB {obs}")
            if len(signals) == 2:
                obs1 = self._resolve_bds_osb_observable(dcb, signals[0])
                obs2 = self._resolve_bds_osb_observable(dcb, signals[1])
                col1, col2 = f"OSB_{obs1}", f"OSB_{obs2}"
                a, b = mode_ionosphere_free_coefficients(self.config.bds_mode)
                out = (a * dcb[[col1]] - b * dcb[[col2]].values) * 1e-9 * self.clight
                return self._product_tgd(out.rename(columns={col1: 'TGD'}), dcb, f"OSB {obs1}, {obs2}")
        if type_dcb == 'DSB':
            if len(signals) != 2:
                raise ValueError(
                    f"BDS DSB products are accepted for two-signal modes only, not for the single-signal mode "
                    f"{self.config.bds_mode} (not enabled in this version, see SISConfig, compare_dcb); use a "
                    "matching OSB product instead."
                )
            # Same quantity as the OSB branch: a*OSB_1 - b*OSB_2, the OSBs being tied to the
            # IF(C2I,C6I) datum of the precise clocks through the DSBs (a single DSB of the
            # mode's pair, e.g. -b*DSB(C2I-C6I) = OSB_C2I for B1IB3I, is the group delay of the
            # first signal, not of the IF combination). The mode's own pair must be in the
            # product (its observables are those of the pair).
            obs1, obs2, _ = self._resolve_bds_dsb_pair(dcb, signals[0], signals[1])
            osb = self._bds_osb_from_dsb(dcb)
            missing = [obs for obs in (obs1, obs2) if obs not in osb]
            if missing:
                raise ValueError(
                    f"BDS DSB product {dcb.attrs.get('bias_path', '')}: no DSB chain ties {', '.join(missing)} (mode {self.config.bds_mode}) to the "
                    f"C2I-C6I clock datum; observables reachable: {sorted(osb)}.")
            a, b = mode_ionosphere_free_coefficients(self.config.bds_mode)
            out = (a * osb[obs1] - b * osb[obs2]) * 1e-9 * self.clight
            return self._product_tgd(out.to_frame(name='TGD'), dcb, f"DSB chain to {obs1}, {obs2}")
        return None

    def _bds_osb_from_dsb(self, dcb: pd.DataFrame) -> dict[str, pd.Series]:
        """
        BDS code OSBs [ns] in the IF(C2I,C6I) clock datum of IGS/MGEX precise clocks, from DSBs.

        The datum pins the anchor pair: with D = DSB(C2I-C6I) and the IF coefficients
        (a, b) of B1I/B3I, a*OSB_C2I - b*OSB_C6I = 0 and OSB_C2I - OSB_C6I = D give
        OSB_C2I = -b*D and OSB_C6I = -a*D. Every other observable follows along DSB chains
        (OSB_Y = OSB_X - DSB(X-Y)), e.g. C7I from C2I-C7I, C1X from C1X-C6I, C5X from C1X-C5X;
        D itself may come from a chain (C2I-C7I and C7I-C6I) when C2I-C6I is not given.

        Offsets relative to C2I are grown breadth-first per satellite: the shortest chain
        wins, ties go to the pair column names in sorted order (deterministic when a product
        has redundant loops such as C1P-C7D and C1X-C7D), and a satellite without a value in
        one DSB is filled from another chain.

        Raises:
            ValueError: If no DSB chain links C2I and C6I (no tie to the clock datum).
        """
        available = self._available_bds_bias_pairs(dcb)
        pairs = []
        for col in sorted(dcb.columns):
            parts = col.split('_')
            if len(parts) == 3 and parts[0] == 'BIAS' and (not available or f"{parts[1]}_{parts[2]}" in available):
                pairs.append((parts[1], parts[2], col))
        # rel[X] = OSB_X - OSB_C2I [ns]; a DSB X-Y is OSB_X - OSB_Y
        rel = {'C2I': pd.Series(0.0, index=dcb.index)}
        filled = -1
        while filled != sum(int(s.notna().sum()) for s in rel.values()):
            filled = sum(int(s.notna().sum()) for s in rel.values())
            known = dict(rel)  # one chain step per pass: shorter chains take precedence
            for x, y, col in pairs:
                for src, dst, step in ((x, y, -dcb[col]), (y, x, dcb[col])):
                    if src in known:
                        cand = known[src] + step
                        rel[dst] = rel[dst].combine_first(cand) if dst in rel else cand
        if 'C6I' not in rel or rel['C6I'].isna().all():
            raise ValueError(
                f"BDS DSB product has no C2I-C6I bias (direct or chained): DSBs of mode {self.config.bds_mode} "
                "cannot be tied to the IF(C2I,C6I) datum of the precise clocks; use a product with C2I-C6I "
                "or an OSB product.")
        _, b = ionosphere_free_coefficients('B1I', 'B3I')
        osb_c2i = b * rel['C6I']  # -b*D with D = OSB_C2I - OSB_C6I = -rel['C6I']
        return {obs: osb_c2i + offset for obs, offset in rel.items()}

    # --- PRODUKT OPÓŹNIEŃ W EPOKACH ORBIT ---
    _SYSTEM_NAMES = {'G': 'GPS', 'E': 'Galileo', 'C': 'BeiDou'}

    def _read_bias_product_at_orbit(self, orbit, dcb_path):
        """
        Read a Bias-SINEX product (or the products of several days, given as a list) and
        take its records valid at the orbit rows (sv, time).

        Each record applies from its BIAS_START to its BIAS_END (to the end of that second:
        a daily record ending at 23:59:59 still holds at 24:00), as in the observation
        processing (`GNSSDataProcessor2._bias_values_at`); a record of another day is never
        used (the former nearest-start merge took it silently), so a comparison across
        midnight needs the products of both days. A row without a valid record of its
        satellite is NaN. Biases given as 0.0 for every satellite of the system are
        placeholders, absent (`read_bia`); for the product's own clock-reference pair
        they stand as 0.0 (that combination has no satellite bias by definition).

        Returns:
            tuple(pd.DataFrame, str): bias columns [ns] indexed like `orbit`, with the
            products' availability attrs (union over the files), `attrs['bias_path']` and
            `attrs['bias_name']` (the file names), and the file type.

        Raises:
            ValueError: If the product has no record of the configured system, none of its
                records of that system is valid at the orbit epochs, or the listed files
                mix OSB and DSB products.
        """
        paths = list(dcb_path) if isinstance(dcb_path, (list, tuple)) else [dcb_path]
        # a file given for both inputs is read (and its reader warnings given) once per run
        read = self.__dict__.setdefault('_bias_products_read', {})
        for path in paths:
            if str(path) not in read:
                read[str(path)] = GNSSDataProcessor2().read_bia(path=path)
        products = [read[str(path)] for path in paths]
        types = sorted({type_dcb for _, type_dcb in products})
        if len(types) > 1:
            raise ValueError(
                f"Bias products {', '.join(map(str, paths))} mix {' and '.join(types)} files; give the "
                "products of one analysis centre and type for the compared days")
        type_dcb = types[0]
        label = str(paths[0]) if len(paths) == 1 else " + ".join(map(str, paths))
        sys_ = self.system
        name = self._SYSTEM_NAMES.get(sys_, sys_)
        attrs = {}
        for key in ('bias_observables_by_system', 'bias_pairs_by_system'):
            union = {}
            for dcb, _ in products:
                for system, values in dcb.attrs.get(key, {}).items():
                    union.setdefault(system, set()).update(values)
            attrs[key] = {system: sorted(values) for system, values in union.items()}
        if 'clock_reference_observables' in products[0][0].attrs:
            attrs['clock_reference_observables'] = products[0][0].attrs['clock_reference_observables']
        # biases given as 0.0 for every satellite (placeholders, e.g. COD MGX BeiDou C2I/C6I) are absent
        # (GNSSDataProcessor2.read_bia), as in SPP, PPP and STEC
        placeholders = sorted({code for dcb, _ in products
                               for code in dcb.attrs.get('placeholder_biases_by_system', {}).get(sys_, [])})
        attrs['placeholder_biases_by_system'] = {sys_: placeholders} if placeholders else {}
        datum = self._placeholder_datum_codes(attrs, placeholders)
        if placeholders and not datum:
            # once per run and file (the clock-reference pair of the product takes them as 0.0, see below)
            warned = self.__dict__.setdefault('_warned_messages', set())
            for path, (dcb, _) in zip(paths, products):
                codes = dcb.attrs.get('placeholder_biases_by_system', {}).get(sys_, [])
                if codes and ('placeholder', str(path)) not in warned:
                    warned.add(('placeholder', str(path)))
                    warnings.warn(GNSSDataProcessor2._placeholder_bias_message(
                        path, sys_, codes, "no group delay is taken from them"),
                        UserWarning, stacklevel=3)
        given = (set(attrs['bias_observables_by_system'].get(sys_, []))
                 | set(attrs['bias_pairs_by_system'].get(sys_, [])))
        if not given and not datum:
            raise ValueError(
                f"Bias product {label} has no {name} ({sys_}) bias records{self._placeholder_note(attrs)}: "
                f"compare_dcb=True needs the {name} group delays of the precise clocks; give a product with "
                f"{name} biases or set compare_dcb=False")
        sv = orbit.index.get_level_values('sv').astype(str).to_numpy()
        times = orbit.index.get_level_values('time').to_numpy(dtype='datetime64[ns]')
        # records of all files (one day each, or overlapping: the record that starts last applies)
        parts = [dcb.reset_index() for dcb, _ in products]
        for part in parts:
            part.attrs = {}  # concat compares the attrs (frames among them)
        records = pd.concat(parts, ignore_index=True, sort=False)
        values = GNSSDataProcessor2._bias_values_at(sv, times, records, 'sv')
        if datum:
            # the mode is the clock-datum pair of the product: its ionosphere-free combination carries no
            # satellite bias by definition, so 0.0 stands for the placeholders within the product's validity
            system_records = records[records['sv'].astype(str).str.startswith(sys_)]
            start = system_records['time'].min()
            end = system_records['time_end'].max() + pd.Timedelta(seconds=1)
            covered = (times >= np.datetime64(start)) & (times <= np.datetime64(end))
            for code in datum:
                values[f"OSB_{code}"] = np.where(covered, 0.0, np.nan)
            attrs['bias_observables_by_system'] = {
                **attrs['bias_observables_by_system'],
                sys_: sorted(set(attrs['bias_observables_by_system'].get(sys_, [])) | set(datum))}
        out = pd.DataFrame(values, index=orbit.index)
        out.attrs.update(attrs)
        out.attrs['bias_path'] = label
        out.attrs['bias_name'] = ", ".join(Path(str(path)).name for path in paths)
        if out.empty or not out.notna().to_numpy().any():
            records = records[records['sv'].astype(str).str.startswith(sys_)]
            raise ValueError(
                f"Bias product {label}: no {name} record is valid at the compared epochs "
                f"{pd.Timestamp(times.min())} .. {pd.Timestamp(times.max())} (its {name} records are valid "
                f"{records['time'].min()} .. {records['time_end'].max()}); give the "
                "bias product of the compared day")
        return out, type_dcb

    def _placeholder_datum_codes(self, attrs: dict, placeholders: list) -> list:
        """The product's clock-reference OSB codes of the configured system when the mode is that pair
        and the product gives them as placeholders (0.0 for every satellite), else []."""
        reference = list((attrs.get('clock_reference_observables') or {}).get(self.system, []))
        signals = mode_signals(self._signal_mode()) if self._signal_mode() else ()
        if len(reference) != 2 or len(signals) != 2 or not set(reference) <= set(placeholders):
            return []
        for code, signal in zip(reference, signals):
            if code not in self._bds_observation_candidates(signal):
                return []
        return reference

    def _placeholder_note(self, dcb_or_attrs) -> str:
        """`` (its ... biases ... are 0.0 placeholders ...)`` for an error message, '' without placeholders."""
        attrs = dcb_or_attrs if isinstance(dcb_or_attrs, dict) else dcb_or_attrs.attrs
        codes = (attrs.get('placeholder_biases_by_system') or {}).get(self.system, [])
        if not codes:
            return ""
        return (f" (its {self._SYSTEM_NAMES.get(self.system, self.system)} biases {', '.join(codes)} are 0.0 for "
                f"every satellite: placeholders for biases the product does not estimate, taken as absent)")

    def _has_bias_codes(self, dcb, type_dcb, pairs) -> bool:
        """True if the product gives, for the configured system, the DSB pairs or both OSBs of `pairs`."""
        return not self._missing_bias_codes(dcb, type_dcb, pairs)[0]

    def _missing_bias_codes(self, dcb, type_dcb, pairs):
        """(missing codes, codes given) of the configured system for the (obs1, obs2) `pairs`."""
        sys_ = self.system
        if type_dcb == 'DSB':
            given = set(dcb.attrs.get('bias_pairs_by_system', {}).get(sys_, []))
            missing = [f"{a}-{b}" for a, b in pairs
                       if f"{a}_{b}" not in given or f"BIAS_{a}_{b}" not in dcb.columns]
            return missing, sorted(p.replace('_', '-') for p in given)
        given = set(dcb.attrs.get('bias_observables_by_system', {}).get(sys_, []))
        codes = dict.fromkeys(code for pair in pairs for code in pair)
        missing = [code for code in codes if code not in given or f"OSB_{code}" not in dcb.columns]
        return missing, sorted(given)

    def _require_bias_codes(self, dcb, type_dcb, pairs) -> None:
        """ValueError naming product, system and codes when the product lacks the biases of `pairs`."""
        missing, given = self._missing_bias_codes(dcb, type_dcb, pairs)
        if missing:
            name = self._SYSTEM_NAMES.get(self.system, self.system)
            raise ValueError(
                f"Bias product {dcb.attrs.get('bias_path', '')} has no {type_dcb} {', '.join(missing)} for {name}, "
                f"needed for the {name} group delay of mode {self._signal_mode()} with compare_dcb=True "
                f"({name} {type_dcb}s in the product: {', '.join(given) or 'none'}){self._placeholder_note(dcb)}; "
                "give a product with these biases or set compare_dcb=False")

    def _signal_mode(self):
        return {'G': self.config.gps_mode, 'E': self.config.gal_mode, 'C': self.config.bds_mode}.get(self.system)

    @staticmethod
    def _product_tgd(tgd: pd.DataFrame, dcb: pd.DataFrame, codes: str) -> pd.DataFrame:
        """Tag a TGD table converted from a product with its path and codes (for the missing-value warning)."""
        tgd.attrs = {'bias_path': dcb.attrs.get('bias_path'), 'bias_name': dcb.attrs.get('bias_name'),
                     'bias_codes': codes}
        return tgd

    def _note_missing_tgd(self, orbit: pd.DataFrame, tgd: pd.DataFrame) -> None:
        """
        Remember the orbit rows with a clock but without a group delay from the bias product
        (satellite absent from it, without the mode's codes, or without a record valid at the
        epoch); `_warn_missing_tgd` reports those that are compared.
        """
        path = tgd.attrs.get('bias_path')
        if path is None or 'TGD' not in tgd.columns:
            return  # broadcast group delays
        has_clk = orbit['clk'].notna().to_numpy() if 'clk' in orbit.columns else np.ones(len(orbit), bool)
        missing = tgd['TGD'].reindex(orbit.index).isna().to_numpy() & has_clk
        if missing.any():
            key = (tgd.attrs.get('bias_name') or Path(str(path)).name, tgd.attrs.get('bias_codes'))
            notes = self.__dict__.setdefault('_missing_tgd', {})
            notes[key] = orbit.index[missing] if key not in notes else notes[key].union(orbit.index[missing])

    def _warn_missing_tgd(self, orbit_0: pd.DataFrame, orbit_1: pd.DataFrame) -> None:
        """
        After `set_common_epochs`: one RuntimeWarning per bias product naming the compared
        satellites (with clocks on both sides) it gives no group delay for, with the share of
        epochs when partial. Their TGD, dTGD and sisre stay NaN.
        """
        notes = self.__dict__.pop('_missing_tgd', {})
        if not notes:
            return
        clocks = np.ones(len(orbit_0), bool)
        for orbit in (orbit_0, orbit_1):
            if 'clk' in orbit.columns:
                clocks &= orbit['clk'].notna().to_numpy()
        compared = orbit_0.index[clocks]
        sv = compared.get_level_values('sv')
        n_rows = pd.Series(1, index=sv).groupby(level=0).sum()
        name = self._SYSTEM_NAMES.get(self.system, self.system)
        for (product, codes), rows in notes.items():
            hit = compared.isin(rows)
            if not hit.any():
                continue
            n_missing = pd.Series(hit, index=sv).groupby(level=0).sum()
            whole = [s for s in n_missing.index if 0 < n_missing[s] == n_rows[s]]
            part = [s for s in n_missing.index if 0 < n_missing[s] < n_rows[s]]
            listed = ""
            if whole:
                # no record of the codes: satellite absent from the product, or not transmitting the signals
                listed = (f"{', '.join(whole)} (no record of these codes for them: absent from the product or "
                          "not transmitting these signals" + (", e.g. BeiDou-2 has no B1C/B2a/B2b and BeiDou-3 no B2I"
                                                              if self.system == 'C' else "") + ")")
            if part:
                # satellites covered only at some epochs (e.g. a record ending before the last epoch)
                times = compared.get_level_values('time')
                gap = times[hit & sv.isin(part)]
                listed += (("; " if whole else "") + f"{', '.join(part)} at {gap.nunique()} of "
                           f"{times.nunique()} epochs ({gap.min()} .. {gap.max()}, no record valid then; a "
                           "comparison across midnight needs the products of both days: dcb_path_0/dcb_path_1 "
                           "take a list)")
            self._warn_once(
                f"{name} {self._signal_mode()}: the bias product {product} gives no group delay "
                f"({codes}) for {listed}: their TGD, dTGD and sisre are NaN there (sisre_notgd is kept).")

    # --- ŁĄCZENIE TGD Z ORBITAMI (wiersze orbity: sv, time) ---
    @staticmethod
    def _time_ordered(orbit):
        """Rows in time order, in the order of the former `sort_values('time')` within an epoch (outputs keep their row order)."""
        times = orbit.index.get_level_values('time')
        if times.is_monotonic_increasing:
            return orbit  # already ordered (a second unstable sort would permute an epoch's rows)
        return orbit.iloc[np.argsort(times.asi8, kind='quicksort')]

    @staticmethod
    def _join_tgd(orbit, tgd):
        """Columns of `tgd` (indexed by the orbit rows) joined to `orbit`, rows in time order."""
        orbit = SISSession._time_ordered(orbit)
        tgd = tgd.reindex(orbit.index)
        for col in tgd.columns:
            orbit[col] = tgd[col].to_numpy()
        return orbit

    def _merge_tgd_with_orbit(self, orbit, dcb):
        """
        Merge GPS per-satellite group delays with orbit/measurement data.

        Returns:
            table-like: Merged structure with delay columns appended.
        """

        if dcb is not None:
            self._note_missing_tgd(orbit, dcb)
            orbit = self._join_tgd(orbit, dcb)
        return orbit

    def _merge_gal_tgd_with_orbit(self, orbit, dcb_out, type_dcb):
        """
        Merge Galileo per-satellite group delays with orbit/measurement data.

        Returns:
            table-like: Merged structure with delay columns appended.
        """

        orbit = self._time_ordered(orbit)
        if dcb_out is not None:
            self._note_missing_tgd(orbit, dcb_out)
            orbit = self._join_tgd(orbit, dcb_out)
            if '_clk_datum' in orbit.columns:
                # precise clock moved to the clock datum of the broadcast message (Galileo I/NAV)
                shift = orbit.pop('_clk_datum')
                missing = shift.isna() & orbit['clk'].notna()
                if missing.any():
                    sats = sorted(set(orbit.index.get_level_values('sv')[missing.to_numpy()]))
                    self._warn_once(
                        f"Galileo {self.config.gal_mode}: no C1C-C5Q/C1C-C7Q bias for {', '.join(sats)}; their "
                        "precise clocks stay in the IF(E1,E5a) datum (I/NAV: IF(E1,E5b)).")
                orbit['clk'] = orbit['clk'] + shift.fillna(0.0)
        else:
            if type_dcb != 'brdc':
                orbit['TGD'] = 0.0
        return orbit

    # --- SPRAWDZENIE USTAWIEŃ PRZED STARTEM ---
    def _check_dcb_paths(self, orbit_0_type: str, orbit_1_type: str) -> None:
        """compare_dcb=True: every SP3 input needs its bias product (dcb_path_0/dcb_path_1)."""
        for k, orbit_type in enumerate((orbit_0_type, orbit_1_type)):
            name = f"dcb_path_{k}"
            if orbit_type == 'SP3' and getattr(self.config, name) is None:
                raise ValueError(
                    f"{name} is required when compare_dcb=True and orb_path_{k} is an SP3 file "
                    f"({getattr(self.config, f'orb_path_{k}')}): a precise product carries no group "
                    f"delays; give a DCB/OSB/DSB file as {name} or set compare_dcb=False")

    def _sisre_default_kr(self) -> list[float]:
        """Literature kr = w_R^2 of the configured system (BeiDou: MEO and IGSO)."""
        if self.system == 'C':
            return [w_r ** 2 for w_r, _ in BDS_SISRE_ALPHA_BETA.values()]
        if self.system in SISRE_WEIGHTS:
            return [SISRE_WEIGHTS[self.system][0] ** 2]
        return []

    def _check_sisre_coefficients(self, kr, kb, kac, krb) -> None:
        """
        SISRE weights: finite, kr/kb/kac >= 0 and |krb| <= 2*sqrt(kr*kb), so that the
        quadratic form kr*dR^2 + kb*dt^2 + krb*dR*dt is never negative (no NaN sisre).
        A coefficient left as None takes the literature value of the configured system.
        """
        given = {"kr": kr, "kb": kb, "kac": kac, "krb": krb}
        for key, value in given.items():
            if value is None:
                continue
            if not _is_real(value) or not np.isfinite(value):
                raise ValueError(f"{key} must be a finite number or None (got {value!r})")
            if key != "krb" and value < 0:
                raise ValueError(f"{key} must be >= 0 (got {value!r})")
        if krb is None:
            return
        krs = [float(kr)] if kr is not None else self._sisre_default_kr()
        if not krs:
            return  # no literature weights: _attach_sisre_coefficients names the missing ones
        kr_min, kb_ = min(krs), (1.0 if kb is None else float(kb))
        bound = 2.0 * np.sqrt(kr_min * kb_)
        if abs(float(krb)) > bound * (1.0 + 1e-9) + 1e-12:
            kr_txt = f"{kr_min:g}" if kr is not None else f"{kr_min:g} (literature value)"
            raise ValueError(
                f"krb must be in [{-bound:g}, {bound:g}] for kr={kr_txt}, kb={kb_:g} (got {krb!r}): "
                "|krb| <= 2*sqrt(kr*kb) keeps sisre^2 = kr*dR^2 + kb*dt^2 + krb*dR*dt + ... >= 0")

    def _check_run_settings(self, orbit_0_type: str, orbit_1_type: str, kr, kb, kac, krb) -> None:
        """
        Requirements that combine settings, checked before any file is read (the single
        values were checked when they were set, see `SISConfig`).

        Raises:
            FileNotFoundError: If a configured file disappeared after it was set.
            ValueError: For an unsupported input type or system, a missing ANTEX/bias
                file, a signal mode the satellite-PCO step does not support, a missing
                `extension_time` with the eclipse extension on, or invalid SISRE weights.

        Warns:
            UserWarning: A bias file given with `compare_dcb=False`, or neighbour-day SP3
                files given for an input that is not SP3 (both are not used).
        """
        cfg = self.config
        for name in SISConfig._ORBIT_PATH_FIELDS + SISConfig._PATH_FIELDS:
            paths = getattr(cfg, name)
            for path in (paths if isinstance(paths, list) else [paths]):  # a list: bias products of several days
                if path is not None and not Path(path).exists():
                    raise FileNotFoundError(f"{name}: file not found: {path}")
        types = (orbit_0_type, orbit_1_type)
        for k, orbit_type in enumerate(types):
            if orbit_type not in SIS_INPUT_TYPES:
                _check_orbit_path(f"orb_path_{k}", getattr(cfg, f"orb_path_{k}"))  # raises the ValueError
        system = _check_choice("system", cfg.system, SIS_SYSTEMS)
        full = list(SIS_FULL_SYSTEMS)
        if system not in SIS_FULL_SYSTEMS:
            if 'RINEX_NAV' in types:
                k = types.index('RINEX_NAV')
                raise ValueError(
                    f"system must be one of {full} for a broadcast navigation input (got {system!r}; "
                    f"orb_path_{k} is a RINEX navigation file); system {system!r} is supported only "
                    "for SP3 x SP3 comparisons")
            if cfg.apply_satellite_pco:
                raise ValueError(f"system must be one of {full} when apply_satellite_pco=True (got {system!r}); "
                                 "set apply_satellite_pco=False")
            if cfg.compare_dcb:
                raise ValueError(f"system must be one of {full} when compare_dcb=True (got {system!r}); "
                                 "set compare_dcb=False")
        if cfg.apply_satellite_pco and 'SP3' in types:
            if cfg.atx_path is None:
                k = types.index('SP3')
                raise ValueError(
                    f"atx_path is required when apply_satellite_pco=True and orb_path_{k} is an SP3 file: "
                    "give an ANTEX file (e.g. igs20.atx) as atx_path or set apply_satellite_pco=False")
            mode_name = {"G": "gps_mode", "E": "gal_mode"}.get(system)
            if mode_name is not None:
                # a valid mode of the system (checked when set) that the PCO step lacks: no hint
                _check_choice(mode_name, getattr(cfg, mode_name), SIS_PCO_MODES[system],
                              context=" when apply_satellite_pco=True", hint=False)
        if cfg.compare_dcb:
            self._check_dcb_paths(orbit_0_type, orbit_1_type)
        else:
            given = [n for n in ("dcb_path_0", "dcb_path_1") if getattr(cfg, n) is not None]
            if given:
                warnings.warn(f"{', '.join(given)} given but compare_dcb=False: the bias file is not read "
                              "(TGD = 0, BeiDou NAV: broadcast group delays); set compare_dcb=True to "
                              "compare group delays", UserWarning, stacklevel=4)  # run() wrapper
        # neighbour days extend an SP3 input only; prev_sp3/next_sp3 belong to orb_path_1 and
        # prev_sp3_0/next_sp3_0 to orb_path_0. A warning, not an error: setting both pairs keeps a
        # configuration valid when the two inputs are swapped.
        for k, names in ((0, ("prev_sp3_0", "next_sp3_0")), (1, ("prev_sp3", "next_sp3"))):
            given = [n for n in names if getattr(cfg, n) is not None]
            if given and types[k] != 'SP3':
                warnings.warn(f"{', '.join(given)} given but orb_path_{k} is not an SP3 file "
                              f"({getattr(cfg, f'orb_path_{k}')}): neighbour-day SP3 files extend only "
                              f"an SP3 orb_path_{k} and are not used ({', '.join(names)} belong to "
                              f"orb_path_{k})", UserWarning, stacklevel=4)
        if cfg.apply_eclipse and cfg.extend_eclipse and cfg.extension_time is None:
            raise ValueError("extension_time is required when apply_eclipse=True and extend_eclipse=True "
                             "(got None): give the post-eclipse extension in minutes >= 0 or set "
                             "extend_eclipse=False")
        self._check_sisre_coefficients(kr, kb, kac, krb)

    @guarded_session_run("SISSession")
    def run(self, kr=None, kb=None, kac=None, krb=None):
        """
        Execute the configured SIS processing pipeline.

        Status:
            Active public method. This method is intentionally procedural so the
            routing remains explicit for broadcast-vs-precise, SP3-vs-SP3 and
            bias-product comparisons.

        Steps (high level):
            1) Input classification
            2) Orbit processing (broadcast or precise)
            3) Optional eclipse masking
            4) Optional satellite PCO/PCV application
            5) DCB ingestion and conversion to group delays
            6) Final merging and output preparation

        Args:
            kr, kb, kac, krb: SISRE coefficients of
                ``sisre^2 = kr*dR^2 + kb*dt''^2 + kac*(dA^2 + dC^2) + krb*dR*dt''``
                (``dR`` outward). ``None`` takes the per-satellite literature value
                (Montenbruck et al. 2015/2018: ``w_R^2, 1, w_AC^2, -2*w_R``); the
                former defaults ``0.9604, 1.0, 0.019881, -1.960`` passed together
                select the literature weights as well (with a FutureWarning).

        Returns:
            table-like: Final product of the pipeline (e.g., DataFrame with orbit
            coordinates, clocks, and per-satellite delays).

        Raises:
            SessionExecutionError: `run()` is wrapped by `guarded_session_run`, so every
                error comes out as `SessionExecutionError` (a RuntimeError) whose `.cause`
                is the original exception. Before any file is read, settings that do not
                fit together (see `_check_run_settings`) give a cause ValueError or
                FileNotFoundError naming the setting, e.g. satellite PCO of an SP3 input
                without `atx_path`, `compare_dcb=True` without the bias file of an SP3
                input, or SISRE weights that would make `sisre` NaN. (Invalid single
                values are rejected earlier, as plain ValueError, when set on `SISConfig`.)

        Warns:
            UserWarning: For a given file that is not used: a bias file with
                `compare_dcb=False`, or a neighbour-day SP3 of an input that is not SP3.
        """

        # warn-once state is per run (bug-fix stage 13b): the configuration's keys (gnx_py._notes) and the
        # session's own messages, so a second run() of the session or of its configuration warns again
        _notes.clear_run_notes(self.config)
        self._warned_messages = set()
        self._bds_bcnav_tgd = None
        # ---- orbit interpolation
        apc_signal = None  # radial offset of the user's APC from the broadcast reference APC
        orbit_0_type = self.classify_file(self.config.orb_path_0)
        orbit_1_type = self.classify_file(self.config.orb_path_1)
        _logger.debug("SIS pipeline event=classified orbit_0_type=%s orbit_1_type=%s", orbit_0_type, orbit_1_type)
        self._check_run_settings(orbit_0_type, orbit_1_type, kr=kr, kb=kb, kac=kac, krb=krb)
        if orbit_0_type == 'RINEX_NAV':
            orbit_0 = self.process_broadcast_orbit(path=self.config.orb_path_0)
        elif orbit_0_type =='SP3':
            orbit_0 = self.process_precise_orbit(path=self.config.orb_path_0,prev_path=self.config.prev_sp3_0,
                                                 next_path=self.config.next_sp3_0,
                                                 neighbour_params=("prev_sp3_0", "next_sp3_0"))
            if self.config.apply_satellite_pco:
                orbit_0, apc_signal = self._apply_reference_pco(orbit_0, other_type=orbit_1_type,
                                                                sp3_path=self.config.orb_path_0)

        if orbit_1_type == 'RINEX_NAV':
            orbit_1 = self.process_broadcast_orbit(path=self.config.orb_path_1)
        elif orbit_1_type == 'SP3':
            orbit_1 = self.process_precise_orbit(path=self.config.orb_path_1,prev_path=self.config.prev_sp3,
                                                 next_path=self.config.next_sp3)
            if self.config.apply_satellite_pco:
                orbit_1, apc_signal_1 = self._apply_reference_pco(orbit_1, other_type=orbit_0_type,
                                                                  sp3_path=self.config.orb_path_1)
                apc_signal = apc_signal if apc_signal_1 is None else apc_signal_1
        _logger.debug("SIS pipeline event=orbits-processed")
        # ---- DCB comparison
        orbit_0, orbit_1 = self.process_dcb_and_merge(orbit_0, orbit_1, orbit_0_type, orbit_1_type)
        orbit_0 = self._ensure_apc_columns(orbit_0)
        orbit_1 = self._ensure_apc_columns(orbit_1)
        _logger.debug("SIS pipeline event=dcb-processed")

        # ---- SISRE computation
        orbit_0, orbit_1 = self.set_common_epochs(orbit_0, orbit_1)
        orbit_0, orbit_1 = self._apply_prn_change_rule(orbit_0, orbit_1)
        self._warn_missing_tgd(orbit_0, orbit_1)
        self._bds_generations = None
        if self.system == 'C':
            # generation of every BeiDou satellite (ANTEX at the first epoch, B-CNAV, PRN): clock datum per
            # generation and the IF(B1C,B2a) precise-clock conversion (bug-fix stage 13d)
            t = orbit_0.index.get_level_values('time')
            self._bds_generations = self._bds_generations_of(
                set(orbit_0.index.get_level_values('sv')), t.min() if len(t) else None,
                cnav_satellites=self._bds_cnav_satellites(orbit_0, orbit_1))
            orbit_0, orbit_1 = self._convert_bds_precise_clock_datum(orbit_0, orbit_1, orbit_0_type, orbit_1_type)
        _logger.debug("SIS pipeline event=common-epochs-processed epochs=%d", len(orbit_0))
        compared_orbits = self.compare_orbits(orbit_0=orbit_0, orbit_1=orbit_1,
                                              clock_bias=self.config.clock_bias,
                                              clock_bias_function=self.config.clock_bias_function,
                                              kr=kr, krb=krb, kac=kac,kb=kb)
        if self.config.apply_eclipse:
            compared_orbits = self.eclipse_periods(orbit=compared_orbits)
            self.output_cols += ['eclipse_raw', 'eclipse']
        att = [o['att_regime'] for o in (orbit_0, orbit_1) if 'att_regime' in o.columns]
        if att:
            # attitude regime of the precise input (the first one when both are precise), reporting only
            idx = compared_orbits.index
            compared_orbits['att_regime'] = att[0].reorder_levels(idx.names).reindex(idx).to_numpy(float)
            if 'att_regime' not in self.output_cols:
                self.output_cols.append('att_regime')
        compared_orbits = self._annotate_clock_convention(compared_orbits, orbit_0_type, orbit_1_type)
        if apc_signal is not None:
            # compare_orbits keeps the columns of orbit_0 only: copy by (sv, time) for both orders
            idx = compared_orbits.index
            compared_orbits['dR_apc_signal'] = (
                apc_signal.reorder_levels(idx.names).reindex(idx).to_numpy(float))
            compared_orbits.attrs['apc_reference'], compared_orbits.attrs['apc_signal'] = (
                self._broadcast_apc_labels())
            if 'dR_apc_signal' not in self.output_cols:
                self.output_cols.append('dR_apc_signal')
        return compared_orbits#[self.output_cols]

    #: APC label of the reference a broadcast orbit refers to and the mode whose APC equals it, per system with
    #: a reference (``antex.broadcast_apc_reference``): GPS IF(L1,L2), BeiDou B3I
    _BROADCAST_REFERENCE = {"G": ("IF(L1,L2)", "L1L2"), "C": ("B3I", "B3I")}

    def _system_mode(self) -> str:
        return {"G": self.config.gps_mode, "E": self.config.gal_mode, "C": self.config.bds_mode}[self.system]

    def _broadcast_reference_applies(self, other_type: str) -> bool:
        """GPS or BeiDou precise orbit compared with a broadcast one in a mode whose APC is not the broadcast
        reference (GPS: any mode but L1L2; BeiDou: any mode but B3I)."""
        ref = self._BROADCAST_REFERENCE.get(self.system)
        return ref is not None and other_type == 'RINEX_NAV' and self._system_mode() != ref[1]

    def _broadcast_apc_labels(self):
        """(apc_reference, apc_signal) attrs of a comparison at the broadcast reference APC."""
        ref = self._BROADCAST_REFERENCE.get(self.system)
        if ref is not None:
            return ref[0], _gal_apc_label(self._system_mode())
        return (_gal_apc_label(GAL_BROADCAST_APC_REFERENCE[self.config.gal_mode]),
                _gal_apc_label(self.config.gal_mode))

    def _apply_reference_pco(self, precise_orbit: pd.DataFrame, other_type: str, sp3_path=None):
        """
        Satellite PCO of a precise orbit compared with a broadcast one, at the broadcast reference APC.

        A broadcast orbit and clock refer to the APC of the message's clock-reference signal
        (``antex.broadcast_apc_reference``) whatever the user's
        signal; the broadcast group delays map it to the signal.

        GPS: IF(L1,L2) (IS-GPS-200 20.3.3.4.3). For every GPS mode other than L1L2 the precise
        orbit is moved to the IF(L1,L2) APC (``sat_fixed_system(apc='broadcast')``), not to the
        APC of the mode: at the mode's APC the GPS IIIA L1/L5 PCO difference entered dR as an
        orbit error (BRUX 2024-035 BRDC vs COD, 24 h: GPS IIIA mean dR -0.747 m for L1 and
        -1.227 m for L5, +0.015 m at IF(L1,L2) like the other blocks; R_rms L5 0.562 -> 0.157 m).

        BeiDou: B3I, the signal the broadcast clock refers to (BDS-SIS-ICD-B1I 3.0 5.2.4.10,
        B1C 1.0 7.6, B2a 1.0 7.6, B2b 1.0 7.5: every message group delay is relative to B3I), for
        D1/D2 and B-CNAV records alike. For every BeiDou mode other than B3I the precise orbit is
        moved to the B3I APC. Measured (2024-035/131/132, COD and GRG, IGS20_2388): BeiDou-3 mean dR
        -0.085..+0.046 m at B3I for every block, against +0.20..+0.40 m at B1I, +0.69..+1.12 m at
        IF(B1I,B3I) and -1.04..-1.68 m at the CoM; T-BRDC R_rms MEO median 0.717 -> 0.077 m.
        The igs20 BeiDou-2 PCO fits no candidate (B3I +0.34..+1.72 m, day-dependent).

        Galileo: a broadcast ephemeris refers to the apparent phase centre common to all
        frequencies (OS SIS ICD 5.1.1) and its clock to the signal pair of its message. For
        modes listed in `GAL_BROADCAST_APC_REFERENCE` (E5aE5b: no E5a/E5b clock, I/NAV
        E1/E5b clock) the precise orbit is moved to the reference IF combination instead of
        the user's signal mode; the other Galileo modes stay at the APC of the mode (the
        measured orbit point lies near the mean of the frequency PCOs).

        The user's APC is then reported, not compared: the returned series `dR_apc_signal` is
        the radial (outward) component of APC(user mode) minus APC(reference), indexed by
        (sv, time). A user corrects that difference through group delays/code biases, so it
        enters neither dR nor SISRE.

        Returns:
            tuple(pd.DataFrame, pd.Series | None): the APC orbit and `dR_apc_signal` (None
            when the reference equals the user's mode, i.e. every other system/mode/pair).
        """
        if self._broadcast_reference_applies(other_type):
            apc = self.apply_satellite_pco(precise_orbit=precise_orbit, apc='broadcast', sp3_path=sp3_path)
            return apc, apc['dR_apc_signal'].copy()
        ref = None
        if self.system == 'E' and other_type == 'RINEX_NAV':
            ref = GAL_BROADCAST_APC_REFERENCE.get(self.config.gal_mode)
        if ref is None:
            return self.apply_satellite_pco(precise_orbit=precise_orbit, sp3_path=sp3_path), None
        signal_apc = self.apply_satellite_pco(precise_orbit=precise_orbit.copy(), sp3_path=sp3_path)
        apc = self.apply_satellite_pco(precise_orbit=precise_orbit, pco_mode=ref)
        sig = signal_apc[['x_apc', 'y_apc', 'z_apc']].reindex(apc.index).to_numpy(float)
        com = apc[['x', 'y', 'z']].to_numpy(float)
        e_r = com / np.linalg.norm(com, axis=1, keepdims=True)
        d_apc = sig - apc[['x_apc', 'y_apc', 'z_apc']].to_numpy(float)
        apc['dR_apc_signal'] = np.sum(d_apc * e_r, axis=1)
        return apc, apc['dR_apc_signal'].copy()


class _DeprecatedSISControllerMeta(type):
    """Metaclass of the `SISController` alias: isinstance/issubclass follow `SISSession`.

    `SISController` is a subclass of `SISSession`, so objects created through the old name
    are `SISSession` instances. This metaclass makes the reverse hold as well:
    `isinstance(SISSession(cfg), SISController)` and `issubclass(SISSession, SISController)`
    are True, so old checks written against `SISController` accept sessions built with the
    new name. The override applies to the alias itself only; user subclasses of
    `SISController` keep the default checks.
    """

    def __instancecheck__(cls, instance):
        if cls is SISController:
            return isinstance(instance, SISSession)
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if cls is SISController:
            return issubclass(subclass, SISSession)
        return super().__subclasscheck__(subclass)


class SISController(SISSession, metaclass=_DeprecatedSISControllerMeta):
    """Deprecated alias of `SISSession`.

    Deprecated:
        `SISController` was renamed to `SISSession`, consistent with `PPPConfig`/`PPPSession`
        and `TECConfig`/`TECSession` (`SISConfig` keeps its name). The old name still works
        and behaves exactly as `SISSession`; each instantiation emits one DeprecationWarning.
        It will be removed only in a future major release. Replace
        `SISController(config).run()` with `SISSession(config).run()`.
    """

    def __init__(self, config: SISConfig):
        warnings.warn("SISController is deprecated and will be removed in a future major release; "
                      "use SISSession instead (same arguments and behaviour, "
                      "e.g. SISSession(config).run()).", DeprecationWarning, stacklevel=2)
        super().__init__(config)
