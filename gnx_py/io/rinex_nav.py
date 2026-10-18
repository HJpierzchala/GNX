"""RINEX 2/3/4 broadcast navigation reader, ionosphere coefficients and broadcast health.

Files are opened through :mod:`gnx_py.compression` (plain, gzip, Unix compress, bzip2, zip); blank lines before the
RINEX VERSION / TYPE line are skipped (as by the observation and SP3 readers). RINEX 3 GPS LNAV,
Galileo and BeiDou D1/D2 records are read from their fixed 4 x 19-character slots; RINEX 4 records are walked by
their '>' lines (EPH, ION; STO, EOP and other systems are counted in ``attrs['rinex4_skipped']``). RINEX 2 GPS
navigation (file type N) is read by GNX itself from the fixed format of RINEX 2.11 Table A4 (I2,5(1X,I2.2),F5.1,
3D19.12 / 3X,4D19.12): each field from its own 19-character slot, 'D', 'd', 'E', 'e' exponents and an exponent
written without its letter, blank fields, a last line without its trailing fields and lines padded past column 80
are read; in every version a field the line ends inside is read only when it holds a complete number (mantissa and
exponent; a number cut short by the end of a cut file is NaN, never a shorter number) and blanks inside a number are
ignored as a Fortran D19.12 read ignores them when what remains is a complete number. Blank lines inside a record
are skipped (every BROADCAST ORBIT line of a G, E or C record holds a field the format requires), so they never
shift the lines after them into other fields; a CR CR LF file reads as a CR LF file. The records are normalised by
``rinex2_adapter.adapt_rinex2_gps_nav`` as before (no georinex). A copy of the same ephemeris (equal clock and orbit
fields, IODE to IODC; transmission time, fit interval and spares may differ) is dropped with a note and the earliest
transmission is kept; a conflicting copy raises :class:`RinexNavError` naming R2OBSE04. Truncated records and
records with a blank or unreadable required field (clock, Keplerian set, issue of data, transmission time; GPS also
TGD: ``_REQUIRED``) are skipped with one warning per file, in every version. RINEX 2 GLONASS (G), SBAS
(H) and the unofficial RINEX 2 Galileo files (teqc "2.12" type E, RTKLIB type L) are refused with a readable
message.
Format references: RINEX 2.11 Table A3/A4, RINEX 3.05 Tables A5-A13, RINEX 4.02 Tables A6-A35, IS-GPS-200
20.3.3.5.2.5, IS-GPS-705 20.3.3.3.1.2, Galileo OS SIS ICD 5.1.6/5.1.9.3, BDS-SIS-ICD-B1I 3.0 5.2.4.7. Written
from the specifications; no GPL code.

Interface (stable; additions only with defaults)
------------------------------------------------
* :class:`RinexNavError` (``ValueError``) names the file, the RINEX version and the systems (requested and in the
  file). No print, no traceback on stdout.
* :func:`nav_info` reads the header (and, for a mixed file, the system letters of the record starts) into a
  :class:`NavInfo`.
* :func:`check_nav_systems` raises before any record is read when the file cannot carry a requested system:
  RINEX 2 file type N carries GPS only ("<path>: RINEX 2.xx navigation (file type N) carries GPS ephemerides only;
  system(s) E, C cannot be read from it - use a RINEX 3/4 mixed navigation file"); RINEX 2 types G and H
  (GLONASS, SBAS) and the RINEX 2 Galileo variants E and L (``nav_info`` systems ('E',)) are refused; an unknown
  version raises with the file and the version named; a RINEX 3/4 file without records of a requested system
  raises "<path> (RINEX 3.04, systems in file: G) has no Galileo ephemerides".
* :func:`read_navigation` returns a :class:`NavRead`:

  - ``gps``, ``gal``, ``bds``: frames exactly as today's ``io.OrbitData`` frames (index ``(sv, time)``, time
    = toc, georinex column names such as ``SVclockBias``, ``sqrtA``, ``Toe``, ``TransTime``; a second record of
    one satellite at one toc gets the ``_1`` suffix on ``sv``, ``Exx_1``, as today), None for a system not
    requested;
  - Galileo frames carry ``DataSrc`` and the health columns ``E1B_DVS``, ``E1B_SHS``, ``E5a_DVS``, ``E5a_SHS``,
    ``E5b_DVS``, ``E5b_SHS`` as today;
  - frames read from RINEX 4 carry ``nav_message`` with the record's message token ('LNAV'; 'INAV', 'FNAV';
    'D1', 'D2', 'CNV1', 'CNV2', 'CNV3'), as the RINEX 4 BeiDou frames do today; RINEX 2/3 frames have no
    ``nav_message`` (BroadcastInterp filters BeiDou by it only when present);
  - GPS rows from RINEX 4 carry ``ISC_L1CA``, ``ISC_L2C``, ``ISC_L5I5``, ``ISC_L5Q5`` and ``TGD_CNAV`` [s] from the
    latest CNAV record of the same satellite at or before the row's toc (NaN when there is none);
  - ``iono``: the ionosphere table (:data:`IONO_COLUMNS`), see below; ``info``: the :class:`NavInfo`;
    ``notes``: readable remarks (skipped record kinds, dropped duplicates, rejected ionosphere records).

* Ionosphere table (one row per broadcast coefficient set): ``system`` ('G', 'E', 'C', 'J'), ``sv`` (the
  broadcasting satellite, None for header records), ``message`` (the RINEX 4 token, e.g. 'LNAV', 'CNVX',
  'IFNV', 'D1D2'; the header label for header records, e.g. 'GPSA', 'BDSA', 'ION ALPHA'), ``model`` in
  {'KLOB', 'NEQG', 'BDGIM'}, ``t_tx`` (transmission time, GPST; NaT for header records), ``coeffs`` (tuple:
  KLOB (alpha0..alpha3, beta0..beta3) in s, s/semicircle^n; NEQG (ai0, ai1, ai2); BDGIM the 9 alpha in TECU),
  ``source`` in {'R2 header', 'R3 header', 'R4 ION'}, ``sv_class`` ('GEO', 'IGSO', 'MEO' for BeiDou from
  ``gnss.bds_orbit_type``; None otherwise), ``plausible`` (bool) and ``reason`` (why a set was rejected, None
  when plausible). Screens, per model: GPS Klobuchar amplitude over |geomagnetic latitude| <= 1/3 semicircle with
  5 ns + max amplitude <= 100 ns; BeiDou Klobuchar A2 over |latitude| in [0, 1/3] with max A2 > 0 and 5 ns + max
  A2 <= 100 ns; a Klobuchar set whose 8 raw counts are all equal is rejected (the C48 0x55 filler); NeQuick
  ai0 > 0; BDGIM (BDS-SIS-ICD-B2b-1.0 §7.7.1, Table 7-8) every alpha on the 1/8 TECU grid and inside
  the range of its broadcast word (alpha5 has the negative scale factor -2^-3, so alpha5 <= 0), and a set whose
  nine raw words are all equal modulo 256 is rejected (the C48 0x2AA/0xAA filler of 2024-035); a grid or range
  reason names the first failing alpha and its Table 7-8 range. The model itself is :mod:`gnx_py.ionosphere.bdgim`.
  One warning per rejected record names the record and the file. QZSS ('J') Klobuchar sets are fitted to the area
  around Japan and exceed the global bound by design (220-480 ns on the validation data); they get the filler test only (GNX
  does not use them). Other '> ION' kinds (NavIC) and IRNA/IRNB header records are noted, not read.
* :func:`legacy_iono` gives the values ``io.OrbitData`` carries today: ``{'gpsa', 'gpsb', 'gala', 'bdsa',
  'bdsb'}``, each a list of floats (gpsa/gpsb/bdsa/bdsb 4 values, gala 3) or None. GPS and Galileo: the first
  plausible set, header first, then LNAV, CNVX, IFNV. BeiDou: one set, GEO first, then IGSO, then MEO.
* :func:`broadcast_iono_at` gives per epoch the latest plausible set with ``t_tx`` at or before the epoch (header
  sets count as valid from the start), for the BeiDou Klobuchar within the highest-priority BeiDou class present
  (BDGIM takes any satellite and class); before the first set, the earliest. :func:`read_iono_table` is the
  table of a file alone, cached per file.

Broadcast health (``HealthTable``, used by ``gnx_py.io``)
--------------------------------------------------------
* :func:`health_table` turns a navigation frame into a :class:`HealthTable`: one flag series per base satellite
  and per message type. Galileo: E1-B and E5b flags from I/NAV records only, E5a flags from F/NAV records only
  (RINEX 3.05 Table A8, 4.02 Table A13: the fields that do not apply are written as 0); DVS = 1 or SHS in {1, 3}
  is an outage. GPS: ``health != 0``. BeiDou: ``SatH1`` (or ``health``) != 0. Record time: the transmission time
  (``TransTime`` converted to GPST), falling back to toc.
* :func:`health_flags_at`: a flag holds from its record until the next record of the same satellite and message
  type, for at most :data:`HEALTH_MAX_AGE` (4 h for G and E, 2 h for C); ``merge_asof`` backward over all
  records, healthy ones included, so a healthy record clears an earlier flag. An epoch with no record in force
  (before the first record of the series, e.g. the start of a daily file, or after a gap longer than the maximum
  age) takes the next record when it is at most the maximum age ahead, as the ephemeris selector uses that record
  for the orbit. This replaces the record-copy overwrite of ``io.screen_navigation_message`` (last copy wins).
* BeiDou has one flag column: the satellite health of D1/D2 (``SatH1``) and of B-CNAV1/2/3 (``health``) are both
  satellite-level and each message series is held separately, so a satellite is flagged while any of its series
  says it is unhealthy (on the validation data 0-82 epochs per satellite and day come from CNAV records alone).
* ``io.GNSSDataProcessor2.screen_navigation_message(df, system)`` returns a :class:`HealthTable`;
  ``io.GNSSDataProcessor2.mark_outages(obs_df, system, outage_info, ...)`` accepts a :class:`HealthTable` (in-force
  rule above) or the legacy list (G, C) / dict (E) and writes the columns of :data:`HEALTH_COLUMNS` in
  ``obs_df``'s own row order.

Public API::

    nav_info(path) -> NavInfo
    check_nav_systems(path, systems) -> NavInfo
    read_navigation(path, systems) -> NavRead
    read_iono_table(path, systems=None) -> DataFrame
    legacy_iono(iono) -> dict
    broadcast_iono_at(iono, system, model, epochs) -> ndarray (n, k)
    health_table(nav_df, system) -> HealthTable
    health_flags_at(table, obs_index) -> DataFrame[bool]
    galileo_data_source_columns(df) -> DataFrame         (io.interpret_data_src delegates here)
    galileo_health_columns(df, health_col, prefix) -> DataFrame   (io.add_galileo_health_cols delegates here)
    clear_cache() -> None

Notes on the frames (C2): RINEX 3 frames keep georinex's row order ((time, sv) lexicographic, index
levels (sv, time)) and RINEX 4 GPS and Galileo frames use the RINEX 3 order; RINEX 2 GPS frames are sorted
(sv, time) by the adapter, as before; RINEX 4 BeiDou frames keep today's (sv, time) order without the ``_1``
suffix (a D1 and a CNV1 record at one toc share an index entry, as today).
RINEX 3 frames differ from the frames of the former reader only in these columns: BeiDou ``AODC``, which now holds the broadcast value (georinex mis-aligned the
record tail and the old repair set it to 0), BeiDou ``spare1``-``spare3`` and GPS ``spare0``/``spare1`` (each
record's own slots, blank -> NaN; georinex decided per satellite whether the spare fields exist). Every frame
carries ``attrs['nav_path']``;
RINEX 4 frames also ``attrs['rinex_version']`` and ``attrs['rinex4_skipped']`` (record kind -> count).
"""
from __future__ import annotations

import os
import re
import threading
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import compression
from ..core.gnss import bds_orbit_type
from .rinex2_adapter import GPS_NAV_REQUIRED_CORE, adapt_rinex2_gps_nav
from .rinex_obs import RinexWarning

__all__ = [
    "HEALTH_COLUMNS", "HEALTH_MAX_AGE", "IONO_COLUMNS", "IONO_MODELS", "HealthTable", "NavInfo", "NavRead",
    "RinexNavError", "broadcast_iono_at", "check_nav_systems", "clear_cache", "galileo_data_source_columns",
    "galileo_health_columns", "health_flags_at", "health_table", "legacy_iono", "nav_info", "read_iono_table",
    "read_navigation",
]

#: Columns of the ionosphere table, in this order.
IONO_COLUMNS = ("system", "sv", "message", "model", "t_tx", "coeffs", "source", "sv_class", "plausible", "reason")

#: Number of coefficients per broadcast ionosphere model (the width k of :func:`broadcast_iono_at`).
IONO_MODELS = {"KLOB": 8, "NEQG": 3, "BDGIM": 9}

#: Output columns of :func:`health_flags_at` (and of ``io.mark_outages``) per system.
HEALTH_COLUMNS = {
    "G": ("gps_outage",),
    "C": ("bds_outage",),
    "E": ("E1B_DVS_outage", "E1B_SHS_outage", "E5a_DVS_outage", "E5a_SHS_outage", "E5b_DVS_outage",
          "E5b_SHS_outage"),
}

#: Longest time a broadcast health flag stays in force without a newer record of the same message type.
HEALTH_MAX_AGE = {"G": pd.Timedelta(hours=4), "E": pd.Timedelta(hours=4), "C": pd.Timedelta(hours=2)}

#: Systems whose ephemerides GNX reads, and their names in messages.
_SYSTEM_NAMES = {"G": "GPS", "E": "Galileo", "C": "BeiDou", "R": "GLONASS", "J": "QZSS", "I": "NavIC",
                 "S": "SBAS"}
_SUPPORTED = ("G", "E", "C")
_SYSTEM_ORDER = "GRECJIS"

#: RINEX 2 navigation file types (RINEX 2.11 Table A3 and section 5.3): N GPS, G GLONASS, H geostationary SBAS;
#: Galileo ephemerides in RINEX 2 are not part of RINEX 2.11 and are written in two unofficial variants (teqc's
#: "2.12" file type E, RTKLIB's file type L), recognised only to refuse them with a readable message.
_R2_FILE_TYPES = {"N": ("G",), "G": ("R",), "H": ("S",), "E": ("E",), "L": ("E",)}

#: BDT = GPST - 14 s; a BDT time tag is moved to GPST by adding this offset.
_BDT_TO_GPST = pd.Timedelta(seconds=14)

#: Seconds of a week, and the RINEX "transmission time not known" threshold (0.9999E9; some writers 9.999E9).
_WEEK_S = 604800.0
_TRANS_UNKNOWN_S = 9e8

# ----------------------------------------------------------------------------------------------- record layouts
# BROADCAST ORBIT lines 1-7 (RINEX 3.05 Tables A6, A8, A14; RINEX 4.02 Tables A6, A12/A13, A23): four 19-character
# fields from column 5. Names as georinex gives them, so the frames keep their column names.
_GPS_LNAV = (
    ("IODE", "Crs", "DeltaN", "M0"),
    ("Cuc", "Eccentricity", "Cus", "sqrtA"),
    ("Toe", "Cic", "Omega0", "Cis"),
    ("Io", "Crc", "omega", "OmegaDot"),
    ("IDOT", "CodesL2", "GPSWeek", "L2Pflag"),
    ("SVacc", "health", "TGD", "IODC"),
    ("TransTime", "FitIntvl", "spare0", "spare1"),
)
_GAL = (
    ("IODnav", "Crs", "DeltaN", "M0"),
    ("Cuc", "Eccentricity", "Cus", "sqrtA"),
    ("Toe", "Cic", "Omega0", "Cis"),
    ("Io", "Crc", "omega", "OmegaDot"),
    ("IDOT", "DataSrc", "GALWeek", "spare0"),
    ("SISA", "health", "BGDe5a", "BGDe5b"),
    ("TransTime", "spare1", "spare2", "spare3"),
)
_BDS_D1D2 = (
    ("AODE", "Crs", "DeltaN", "M0"),
    ("Cuc", "Eccentricity", "Cus", "sqrtA"),
    ("Toe", "Cic", "Omega0", "Cis"),
    ("Io", "Crc", "omega", "OmegaDot"),
    ("IDOT", "spare0", "BDTWeek", "spare1"),
    ("SVacc", "SatH1", "TGD1", "TGD2"),
    ("TransTime", "AODC", "spare2", "spare3"),
)
# GPS CNAV (RINEX 4.02 Table A10): only the group delays are used (IS-GPS-200 30.3.3.3.1.1); broadcast orbit 8
# is t_tm, wn_op and the flags.
_GPS_CNAV = (
    ("ADot", "Crs", "DeltaN", "M0"),
    ("Cuc", "Eccentricity", "Cus", "sqrtA"),
    ("Top", "Cic", "Omega0", "Cis"),
    ("Io", "Crc", "omega", "OmegaDot"),
    ("IDOT", "DeltaN0dot", "URAI_NED0", "URAI_NED1"),
    ("URAI_ED", "health", "TGD", "URAI_NED2"),
    ("ISC_L1CA", "ISC_L2C", "ISC_L5I5", "ISC_L5Q5"),
    ("t_tm", "wn_op", "flags", "spare0"),
)
_CLOCK = ("SVclockBias", "SVclockDrift", "SVclockDriftRate")
_LAYOUT3 = {"G": _GPS_LNAV, "E": _GAL, "C": _BDS_D1D2}
_ISC_COLUMNS = ("ISC_L1CA", "ISC_L2C", "ISC_L5I5", "ISC_L5Q5", "TGD_CNAV")
_BDS_MESSAGES = ("D1", "D2", "CNV1", "CNV2", "CNV3")

# Galileo DataSrc bits (RINEX 3.05 Table A8): bit 0 I/NAV E1-B, 1 F/NAV E5a-I, 2 I/NAV E5b-I, 8 af0-af2/Toc/SISA
# for E5a,E1, 9 for E5b,E1 (bits 3-4 reserved).
_DATA_SRC_BITS = {"I/NAV_E1-B": 0, "F/NAV_E5a-I": 1, "I/NAV_E5b-I": 2, "dt_E5a_E1": 8, "dt_E5b_E1": 9}

# Klobuchar scale factors (IS-GPS-200 Table 20-X; BDS-SIS-ICD-B1I 3.0 Table 5-11 uses the same): alpha0..3, beta0..3.
_KLOB_SCALE = np.array([2.0 ** -30, 2.0 ** -27, 2.0 ** -24, 2.0 ** -24, 2.0 ** 11, 2.0 ** 14, 2.0 ** 16, 2.0 ** 16])
_KLOB_NIGHT_S = 5e-9
_KLOB_MAX_S = 100e-9
_PHI_GPS = np.linspace(-1.0 / 3.0, 1.0 / 3.0, 241)
_PHI_BDS = np.linspace(0.0, 1.0 / 3.0, 121)

# BDGIM broadcast words alpha1..alpha9 (BDS-SIS-ICD-B2b-1.0 §7.7.1, Table 7-8): bits, two's complement or not, and
# the scale factor [TECU]. alpha5 is an unsigned word with the negative scale factor -2^-3 (range -31.875..0).
_BDGIM_BITS = (10, 8, 8, 8, 8, 8, 8, 8, 8)
_BDGIM_SIGNED = (False, True, False, False, False, True, True, True, True)
_BDGIM_SCALE = (1 / 8, 1 / 8, 1 / 8, 1 / 8, -1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8)
#: largest distance of a value from the 1/8 TECU grid accepted as print rounding [LSB]
_BDGIM_GRID_TOL = 1e-3

# '> ION <sv> <message>' records (RINEX 4.02 Tables A35-A37) and header IONOSPHERIC CORR labels (RINEX 3.05 A5).
_ION_MODEL4 = {("G", "LNAV"): "KLOB", ("G", "CNVX"): "KLOB", ("J", "LNAV"): "KLOB", ("J", "CNVX"): "KLOB",
               ("C", "D1D2"): "KLOB", ("C", "CNVX"): "BDGIM", ("E", "IFNV"): "NEQG"}
_ION_HEADER_PAIRS = {"GPSA": ("GPSB", "G"), "QZSA": ("QZSB", "J"), "BDSA": ("BDSB", "C")}
_LEGACY_ORDER = {"R2 header": 0, "R3 header": 0, "LNAV": 1, "CNVX": 2, "IFNV": 3, "D1D2": 1}
_CLASS_ORDER = {"GEO": 0, "IGSO": 1, "MEO": 2}


class RinexNavError(ValueError):
    """A navigation file that cannot give the requested systems; the message names the file, the version and the
    systems."""


@dataclass(frozen=True)
class NavInfo:
    """Header facts of a navigation file.

    ``version``: RINEX version as written (2.11, 3.04, 4.02). ``file_type``: column 21 of RINEX VERSION / TYPE
    ('N'; RINEX 2 'G' GLONASS, 'H' SBAS). ``systems``: the system letters the file carries, in 'GRECJIS' order:
    RINEX 2 N -> ('G',), G -> ('R',), H -> ('S',); RINEX 3/4 single-system files their letter; mixed files the
    letters of the record starts found in the data."""

    version: float
    file_type: str
    systems: tuple
    path: str


@dataclass
class NavRead:
    """Result of :func:`read_navigation` (interface in the module docstring)."""

    gps: Optional[pd.DataFrame]
    gal: Optional[pd.DataFrame]
    bds: Optional[pd.DataFrame]
    iono: pd.DataFrame
    info: NavInfo
    notes: list = field(default_factory=list)


@dataclass(frozen=True)
class HealthTable:
    """Broadcast health of one system as a series of records.

    ``records``: one row per broadcast record copy, with columns ``sv`` (base id, 'E07' for 'E07_1'), ``t``
    (record time, GPST datetime64[ns]: transmission time, else toc), ``message`` ('INAV' or 'FNAV' for Galileo,
    from ``DataSrc`` in RINEX 3; 'LNAV' for GPS; 'D1' / 'D2' or the RINEX 4 token for BeiDou) and one float
    column per name of ``flag_columns``: 1.0 the record marks the signal as not usable, 0.0 healthy, NaN the
    record's message type does not carry that flag (so it neither sets nor clears it). Sorted by ``t``.
    ``flag_columns`` equals ``HEALTH_COLUMNS[system]``. ``max_age``: ``HEALTH_MAX_AGE[system]``. ``path``: the
    navigation file, when known."""

    system: str
    records: pd.DataFrame
    flag_columns: tuple
    max_age: pd.Timedelta
    path: Optional[str] = None


# ---------------------------------------------------------------------------------------------------- file text
_TEXT_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_IONO_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_TEXT_CACHE_SIZE = 2
_IONO_CACHE_SIZE = 8


def _file_key(path) -> tuple:
    st = os.stat(path)
    return os.path.realpath(os.fspath(path)), st.st_size, st.st_mtime_ns


#: Blank lines before the first line of a file (the header starts at the first non-blank line).
_LEADING_BLANK_LINES = re.compile(r"\A(?:[ \t\x1a]*\n)+")


def _text(path) -> str:
    """Decompressed text of ``path`` (any compression :mod:`gnx_py.compression` reads), cached for the last two files (key: path, size, mtime). Blank lines
    before the RINEX VERSION / TYPE line are dropped, as the observation and SP3 readers skip them (the format
    makes that line the first record; a blank line before it carries nothing)."""
    key = _file_key(path)
    with _CACHE_LOCK:
        if key in _TEXT_CACHE:
            _TEXT_CACHE.move_to_end(key)
            return _TEXT_CACHE[key]
    text = compression.open_text(path, what="RINEX navigation file").getvalue()
    text = _LEADING_BLANK_LINES.sub("", text, count=1)
    with _CACHE_LOCK:
        _TEXT_CACHE[key] = text
        while len(_TEXT_CACHE) > _TEXT_CACHE_SIZE:
            _TEXT_CACHE.popitem(last=False)
    return text


def clear_cache() -> None:
    """Forget the cached file texts and ionosphere tables."""
    with _CACHE_LOCK:
        _TEXT_CACHE.clear()
        _IONO_CACHE.clear()


def _fmt_version(version: float) -> str:
    return f"{version:.2f}"


def _names(systems) -> str:
    return ", ".join(_SYSTEM_NAMES.get(s, s) for s in systems)


def _header_end(lines: list, path) -> int:
    for k, line in enumerate(lines):
        if line[60:].strip() == "END OF HEADER" or "END OF HEADER" in line:
            return k
    raise RinexNavError(f"{path}: no END OF HEADER line; not a complete RINEX navigation file")


def _parse_info(text: str, path) -> NavInfo:
    first = text[:200].split("\n", 1)[0]
    label = first[60:].strip()
    if label.startswith("CRINEX"):
        raise RinexNavError(f"{path} is a Hatanaka-compressed (CRINEX) observation file, not a navigation file")
    if label != "RINEX VERSION / TYPE":
        raise RinexNavError(f"{path} is not a RINEX file (first line has no 'RINEX VERSION / TYPE' label)")
    try:
        version = float(first[:9])
    except ValueError:
        raise RinexNavError(f"{path}: unreadable RINEX version {first[:9].strip()!r}") from None
    file_type = first[20:21].upper()
    major = int(version)
    if major not in (2, 3, 4):
        raise RinexNavError(f"{path}: RINEX version {first[:9].strip()} is not supported (GNX reads RINEX 2, 3 "
                            "and 4 navigation files)")
    if major == 2:
        if file_type not in _R2_FILE_TYPES:
            raise RinexNavError(f"{path} (RINEX {_fmt_version(version)}, file type {file_type!r}) is not a "
                                "navigation file")
        return NavInfo(version, file_type, _R2_FILE_TYPES[file_type], os.fspath(path))
    if file_type != "N":
        raise RinexNavError(f"{path} (RINEX {_fmt_version(version)}, file type {file_type!r}) is not a navigation "
                            "file")
    letter = first[40:41].upper()
    if letter in _SYSTEM_NAMES:
        return NavInfo(version, file_type, (letter,), os.fspath(path))
    # mixed file ('M' or blank): the system letters of the record starts after the header (a COMMENT line may
    # look like a record start)
    end = text.find("END OF HEADER")
    body = text[text.find("\n", end) + 1:] if end >= 0 else text
    if major == 3:
        found = set(re.findall(r"^([A-Z])[ \d]\d \d{4} ", body, flags=re.M))
    else:
        found = set(re.findall(r"^> EPH ([A-Z])", body, flags=re.M))
    systems = tuple(s for s in _SYSTEM_ORDER if s in found)
    return NavInfo(version, file_type, systems, os.fspath(path))


def nav_info(path) -> NavInfo:
    """Version, file type and systems of the navigation file ``path`` (any compression :mod:`gnx_py.compression` reads).

    Raises :class:`RinexNavError` for a file that is not a RINEX navigation file or has an unknown version."""
    return _parse_info(_text(path), path)


def _requested(systems, path) -> tuple:
    if isinstance(systems, str):
        systems = systems.replace(",", "").replace(" ", "")
    elif systems is None or not isinstance(systems, Iterable):
        raise RinexNavError(f"{path}: systems must be letters such as 'GEC' or ['G', 'E'], got {systems!r}; GNX reads "
                            "G, E and C ephemerides")
    letters = {str(x).upper() for x in systems}
    req = tuple(s for s in _SYSTEM_ORDER if s in letters)
    other = sorted(letters - set(_SYSTEM_ORDER))
    if other or not req:
        raise RinexNavError(f"{path}: unknown system letter(s) {other or list(systems)}; GNX reads G, E and C "
                            "ephemerides")
    return req


def check_nav_systems(path, systems: Iterable[str] | str) -> NavInfo:
    """Raise :class:`RinexNavError` before any record is read when ``path`` cannot give every system of
    ``systems`` (letters 'G', 'E', 'C'); return its :class:`NavInfo` otherwise."""
    info = nav_info(path)
    req = _requested(systems, path)
    version = _fmt_version(info.version)
    unsupported = [s for s in req if s not in _SUPPORTED]
    if unsupported:
        raise RinexNavError(f"{path}: system(s) {', '.join(unsupported)} requested; GNX reads GPS (G), Galileo (E) "
                            "and BeiDou (C) ephemerides only")
    if int(info.version) == 2:
        if info.systems == ("E",):
            raise RinexNavError(f"{path}: RINEX {version} Galileo navigation (file type {info.file_type}) cannot be "
                                "read: Galileo is not defined in RINEX 2.11 navigation files and GNX does not read the "
                                "unofficial RINEX 2 Galileo variants - use a RINEX 3/4 navigation file")
        if info.file_type != "N":
            raise RinexNavError(f"{path}: RINEX {version} navigation (file type {info.file_type}, "
                                f"{_names(info.systems)}) cannot be read: GNX reads GPS, Galileo and BeiDou "
                                "ephemerides")
        missing = [s for s in req if s != "G"]
        if missing:
            raise RinexNavError(f"{path}: RINEX {version} navigation (file type N) carries GPS ephemerides only; "
                                f"system(s) {', '.join(missing)} cannot be read from it - use a RINEX 3/4 mixed "
                                "navigation file")
        return info
    missing = [s for s in req if s not in info.systems]
    if missing:
        raise RinexNavError(f"{path} (RINEX {version}, systems in file: {', '.join(info.systems) or 'none'}) has no "
                            f"{_names(missing)} ephemerides")
    return info


# ---------------------------------------------------------------------------------------------- field parsing
_FORTRAN_EXP = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$")


def _number(s: str) -> float:
    """A Fortran D/E field: 'D' or 'd' exponents, and the exponent letter left out ('0.1770-07', which a Fortran
    D12.4 read accepts); NaN when unreadable."""
    s = s.replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        m = _FORTRAN_EXP.match(s)
        return float(f"{m.group(1)}e{m.group(2)}") if m else np.nan


#: A complete Fortran D/E number: mantissa and a two-digit exponent (with or without its letter, as D19.12 and
#: D12.4 write it). A number cut short loses its end, the exponent first.
_COMPLETE_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[DdEe][+-]?|[+-])\d\d")


def _slot(line: str, a: int, width: int) -> float:
    """The D/E field of ``line[a:a + width]``: NaN when blank or unreadable. A slot the line ends inside is read only
    when it holds a complete number (a number cut by the end of a cut file loses its exponent, so it is NaN and a
    shorter number is never read; a complete number in a short line, written one column to the left by a writer
    that drops a leading blank, as BRDC00GOP 2025-032 does, is read). Blanks inside the number are ignored, as a
    Fortran D19.12 read ignores them, when what remains is a complete number with its exponent (galileo3.15n
    writes ``. 000000000000D+00``: 0.0; RTKLIB reads 0.0 as well)."""
    s = line[a:a + width]
    if len(s) < width and s.strip() and not _COMPLETE_NUMBER.fullmatch(s.strip().replace(" ", "")):
        return np.nan
    s = s.strip()
    if not s:
        return np.nan
    value = _number(s)
    if value != value and " " in s:
        joined = s.replace(" ", "")
        if _COMPLETE_NUMBER.fullmatch(joined):
            value = _number(joined)
    return value


def _floats(line: str, start: int = 4, count: int = 4, width: int = 19) -> list:
    """``count`` fixed-width D/E fields from column ``start`` (0-based) (:func:`_slot`: a blank or unreadable field
    is NaN, a number cut by the end of the file too)."""
    return [_slot(line, start + width * k, width) for k in range(count)]


def _clock_line(line: str) -> tuple:
    """SV / EPOCH / SV CLK line of a RINEX 3/4 record: (sv as written, toc, [af0, af1, af2])."""
    sv = line[:3].strip()
    second = float(line[21:23])
    sec_int = int(second)
    epoch = pd.Timestamp(datetime(int(line[4:8]), int(line[9:11]), int(line[12:14]), int(line[15:17]),
                                  int(line[18:20]), sec_int, int(round((second - sec_int) * 1_000_000))))
    return sv, epoch, _floats(line, start=23, count=3)


#: Fields without which a record gives no usable ephemeris: the clock, the Keplerian set, the issue of data and the
#: transmission time, which the format requires in every record (0.9999E9 when not known), so a record cut inside
#: its last line never stays with a NaN transmission time; GPS also the TGD, as ``GPS_NAV_REQUIRED_CORE`` of the
#: RINEX 2 path. A record with one of them blank or unreadable is skipped with the truncated-record warning (as the
#: RINEX 2 path does), so a field lost or shifted by a broken line never reaches the orbit as NaN silently. On the
#: corpus (85 navigation files) no G, E or C record has one of them blank.
_KEPLER = ("Crs", "DeltaN", "M0", "Cuc", "Eccentricity", "Cus", "sqrtA", "Toe", "Cic", "Omega0", "Cis", "Io", "Crc",
           "omega", "OmegaDot", "IDOT")
_REQUIRED = {"G": _CLOCK + ("IODE",) + _KEPLER + ("TGD", "TransTime"),
             "E": _CLOCK + ("IODnav",) + _KEPLER + ("TransTime",),
             "C": _CLOCK + ("AODE",) + _KEPLER + ("TransTime",),
             "C-CNV": _CLOCK + _KEPLER + ("TransTime",)}


def _complete(row: dict, key: str) -> bool:
    """Whether every field of ``_REQUIRED[key]`` of a record is a number."""
    return all(row.get(name) == row.get(name) and row.get(name) is not None for name in _REQUIRED[key])


def _record_lines(lines: list) -> list:
    """The non-blank lines of a record: a blank line inside a record carries no field (every BROADCAST ORBIT line
    of a G, E or C record holds at least one field the format requires), so it is skipped instead of shifting the
    lines after it into other fields (corpus review F2)."""
    return [ln for ln in lines if ln.strip()]


def _orbit_row(sv: str, epoch, clock, layout, block) -> dict:
    row = {"time": epoch, "sv": sv.replace(" ", "0"), "SVclockBias": clock[0], "SVclockDrift": clock[1],
           "SVclockDriftRate": clock[2]}
    for names, line in zip(layout, block):
        row.update(zip(names, _floats(line)))
    return row


def _suffix_copies(out: pd.DataFrame) -> pd.DataFrame:
    """georinex convention: the second record of one satellite at one toc is 'Exx_1', the third 'Exx_2' (file
    order)."""
    copy_no = out.groupby(["sv", "time"], sort=False).cumcount()
    out["sv"] = np.where(copy_no > 0, out["sv"] + "_" + copy_no.astype(str), out["sv"])
    return out


def _time_major(out: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Index (sv, time) in (time, sv) lexicographic row order, float columns ``columns`` (as georinex's frames)."""
    out = out.set_index(["time", "sv"]).sort_index()
    return out[columns].astype(float).swaplevel()


def _empty_frame(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns), index=pd.MultiIndex.from_arrays(
        [np.array([], dtype=object), pd.DatetimeIndex([])], names=["sv", "time"]), dtype=float)


def galileo_data_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the boolean DataSrc columns 'I/NAV_E1-B', 'F/NAV_E5a-I', 'I/NAV_E5b-I', 'dt_E5a_E1', 'dt_E5b_E1'
    (RINEX 3.05 Table A8 bits 0, 1, 2, 8, 9) to a Galileo navigation frame; a missing or negative DataSrc sets
    none. Returns a new frame (``io.GNSSDataProcessor2.interpret_data_src``)."""
    src = pd.to_numeric(df["DataSrc"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(src) & (src >= 0)
    word = np.where(ok, np.where(ok, src, 0.0), 0.0).astype(np.int64)
    bits = pd.DataFrame({name: ((word >> bit) & 1).astype(bool) for name, bit in _DATA_SRC_BITS.items()},
                        index=df.index)
    return pd.concat([df, bits], axis=1)


def galileo_health_columns(df: pd.DataFrame, health_col: str = "health", prefix: str = "") -> pd.DataFrame:
    """Decode the Galileo "SV health" word (RINEX 3.05 Table A8, 4.02 Table A13: bit 0 E1-B DVS, bits 1-2 E1-B
    HS, bit 3 E5a DVS, bits 4-5 E5a HS, bit 6 E5b DVS, bits 7-8 E5b HS) into the columns E1B_DVS, E1B_SHS,
    E5a_DVS, E5a_SHS, E5b_DVS, E5b_SHS (DVS int 0/1, SHS categorical 0-3). A word that is not an integer in
    0..511 cannot be decoded and is flagged DVS = 1 for every signal, so the record is never taken as healthy.
    Writes into ``df`` and returns it (``io.GNSSDataProcessor2.add_galileo_health_cols``)."""
    health = pd.to_numeric(df[health_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(health) & (health >= 0) & (health <= 511) & (np.round(health) == health)
    w = np.where(valid, np.round(np.where(valid, health, 0.0)), 0).astype(np.int64)
    dvs = np.array([0, 1])
    hs = np.array([0, 1, 2, 3])
    df[prefix + "E1B_DVS"] = dvs[np.where(valid, w & 0b1, 1)]
    df[prefix + "E1B_SHS"] = hs[(w >> 1) & 0b11]
    df[prefix + "E5a_DVS"] = dvs[np.where(valid, (w >> 3) & 0b1, 1)]
    df[prefix + "E5a_SHS"] = hs[(w >> 4) & 0b11]
    df[prefix + "E5b_DVS"] = dvs[np.where(valid, (w >> 6) & 0b1, 1)]
    df[prefix + "E5b_SHS"] = hs[(w >> 7) & 0b11]
    cat = pd.CategoricalDtype(hs, ordered=False)
    for name in ("E1B_SHS", "E5a_SHS", "E5b_SHS"):
        df[prefix + name] = df[prefix + name].astype(cat)
    return df


def _galileo_frame(rows: list, path, nav_message: bool, notes: list) -> pd.DataFrame:
    """Galileo frame as ``io._load_rinex3_galileo_nav`` + ``interpret_data_src`` + ``add_galileo_health_cols``."""
    columns = (["nav_message"] if nav_message else []) + list(_CLOCK) + [n for names in _GAL for n in names]
    if not rows:
        return _empty_frame(columns)
    out = pd.DataFrame(rows)
    health = out["health"]
    trans = out["TransTime"]
    # TransTime is referred to the week of the record and may run past the week end or start before it; anything
    # further out is junk, except the RINEX "not known" value.
    corrupt = ~(health.between(0, 511) & (health.round() == health)
                & (trans.between(-_WEEK_S, 2 * _WEEK_S) | (trans >= _TRANS_UNKNOWN_S)))
    if corrupt.any():
        msg = (f"{path}: {int(corrupt.sum())} Galileo navigation records with an invalid health word or transmission "
               "time were skipped")
        warnings.warn(msg, RinexWarning, stacklevel=3)
        notes.append(msg)
        out = out.loc[~corrupt]
    out = _suffix_copies(out)
    if nav_message:
        msg_col = out.set_index(["time", "sv"]).sort_index()["nav_message"]
        frame = _time_major(out, columns[1:])
        frame.insert(0, "nav_message", msg_col.swaplevel().to_numpy())
    else:
        frame = _time_major(out, columns)
    frame = galileo_data_source_columns(frame)
    return galileo_health_columns(frame)


# ------------------------------------------------------------------------------------------------ RINEX 3 body
def _read_rinex3(lines: list, start: int, req: tuple, path, notes: list) -> dict:
    rows = {s: [] for s in req}
    truncated = Counter()
    n = len(lines)
    i = start
    while i < n:
        line = lines[i]
        if not line[:1].isalpha():
            i += 1
            continue
        j = i + 1
        while j < n and not lines[j][:1].isalpha():
            j += 1
        s = line[0]
        if s in rows:
            block = _record_lines(lines[i + 1:j])
            if len(block) < 7:
                truncated[s] += 1
            else:
                try:
                    sv, epoch, clock = _clock_line(line)
                except ValueError:
                    truncated[s] += 1
                else:
                    row = _orbit_row(sv, epoch, clock, _LAYOUT3[s], block[:7])
                    if _complete(row, s):
                        rows[s].append(row)
                    else:
                        truncated[s] += 1
        i = j
    for s, k in truncated.items():
        msg = f"{path}: {k} truncated or unreadable {_SYSTEM_NAMES[s]} navigation record(s) skipped"
        warnings.warn(msg, RinexWarning, stacklevel=3)
        notes.append(msg)
    frames = {}
    for s in req:
        if s == "E":
            frames[s] = _galileo_frame(rows[s], path, False, notes)
            continue
        columns = list(_CLOCK) + [n for names in _LAYOUT3[s] for n in names]
        if not rows[s]:
            frames[s] = _empty_frame(columns + (["health"] if s == "C" else []))
            continue
        frame = _time_major(_suffix_copies(pd.DataFrame(rows[s])), columns)
        if s == "C":
            frame["health"] = frame["SatH1"]
        frames[s] = frame
    return frames


# ------------------------------------------------------------------------------------------------ RINEX 4 body
def _rinex4_records(lines: list, start: int):
    head, body = None, []
    for line in lines[start:]:
        if line.startswith(">"):
            if head is not None:
                yield head, body
            head, body = line, []
        elif head is not None:
            body.append(line)
    if head is not None:
        yield head, body


def _bds4_row(message: str, lines: list) -> Optional[dict]:
    """A RINEX 4 BeiDou EPH record (Tables A23-A26), as ``io._parse_rinex4_bds_eph_record`` reads it."""
    lines = _record_lines(lines)
    if not lines:
        return None
    sv, epoch, clock = _clock_line(lines[0])
    data = [_floats(line) for line in lines[1:]]
    row = {"sv": sv, "time": epoch, "nav_message": message, "SVclockBias": clock[0], "SVclockDrift": clock[1],
           "SVclockDriftRate": clock[2]}
    if len(data) < 7:
        return None                      # truncated: counted and warned about by the caller
    if message in ("D1", "D2"):
        vals = [v for fields in data for v in fields]
        names = [n for names in _BDS_D1D2 for n in names]
        row.update({name: vals[i] if i < len(vals) else np.nan for i, name in enumerate(names)})
        row["health"] = row.get("SatH1")
        return row if _complete(row, "C") else None
    o1, o2, o3, o4, o5, o6 = data[:6]
    row.update({
        "ADot": o1[0], "Crs": o1[1], "DeltaN": o1[2], "M0": o1[3],
        "Cuc": o2[0], "Eccentricity": o2[1], "Cus": o2[2], "sqrtA": o2[3],
        "Toe": o3[0], "Cic": o3[1], "Omega0": o3[2], "Cis": o3[3],
        "Io": o4[0], "Crc": o4[1], "omega": o4[2], "OmegaDot": o4[3],
        "IDOT": o5[0], "DeltaNdot": o5[1], "SatType": o5[2], "Top": o5[3],
        "SISAI_oe": o6[0], "SISAI_ocb": o6[1], "SISAI_oc1": o6[2], "SISAI_oc2": o6[3],
        # compatibility fields of the BeiDou propagator
        "AODE": 0.0, "spare0": o5[1], "BDTWeek": 0.0, "SVacc": o6[0], "TGD1": 0.0, "TGD2": 0.0,
    })
    if message in ("CNV1", "CNV2"):
        if len(data) < 9:
            return None
        o7, o8, o9 = data[6], data[7], data[8]
        if message == "CNV1":
            row["ISC_B1Cd"] = o7[0]
        else:
            row["ISC_B2ad"] = o7[1]
        row["TGD_B1Cp"] = o7[2]
        row["TGD_B2ap"] = o7[3]
        row.update({"SISMAI": o8[0], "SatH1": o8[1], "IntegrityFlags": o8[2], "AODC": o8[3], "IODC": o8[3],
                    "TransTime": o9[0], "IODE": o9[3], "AODE": o9[3], "health": o8[1]})
        return row if _complete(row, "C-CNV") else None
    o7 = data[6]
    o8 = data[7] if len(data) > 7 else [np.nan] * 4
    row.update({"SISMAI": o7[0], "SatH1": o7[1], "IntegrityFlags": o7[2], "TGD_B2bI": o7[3], "TransTime": o8[0],
                "AODC": 0.0, "IODC": 0.0, "IODE": 0.0, "AODE": 0.0, "health": o7[1]})
    return row if _complete(row, "C-CNV") else None


def _ion4_row(system: str, sv: str, message: str, lines: list) -> Optional[dict]:
    model = _ION_MODEL4.get((system, message))
    if model is None or not lines:
        return None
    first = lines[0]
    try:
        epoch = pd.Timestamp(datetime(int(first[4:8]), int(first[9:11]), int(first[12:14]), int(first[15:17]),
                                      int(first[18:20]), int(first[21:23])))
    except ValueError:
        return None
    vals = _floats(first, start=23, count=3) + [v for line in lines[1:] for v in _floats(line)]
    coeffs = tuple(float(v) for v in vals[:IONO_MODELS[model]])
    if len(coeffs) < IONO_MODELS[model]:
        coeffs = coeffs + (np.nan,) * (IONO_MODELS[model] - len(coeffs))
    if system == "C":
        epoch = epoch + _BDT_TO_GPST
    return {"system": system, "sv": sv, "message": message, "model": model, "t_tx": epoch, "coeffs": coeffs,
            "source": "R4 ION"}


def _attach_cnav(gps: pd.DataFrame, cnav: list) -> pd.DataFrame:
    """ISC_L1CA, ISC_L2C, ISC_L5I5, ISC_L5Q5 and TGD_CNAV of the latest CNAV record of the same satellite at or
    before each LNAV toc (NaN when there is none)."""
    if gps.empty:
        for name in _ISC_COLUMNS:
            gps[name] = np.nan
        return gps
    left = pd.DataFrame({"base": gps.index.get_level_values("sv").str[:3],
                         "time": gps.index.get_level_values("time"), "__row": np.arange(len(gps))})
    left = left.sort_values("time", kind="stable")
    if cnav:
        right = pd.DataFrame(cnav).sort_values("time", kind="stable")
        merged = pd.merge_asof(left, right, on="time", by="base", direction="backward")
    else:
        merged = left.assign(**{name: np.nan for name in _ISC_COLUMNS})
    merged = merged.sort_values("__row")
    for name in _ISC_COLUMNS:
        gps[name] = merged[name].to_numpy(dtype=float)
    return gps


def _read_rinex4(lines: list, start: int, req: tuple, info: NavInfo, path, notes: list):
    gps_rows, gal_rows, bds_rows, cnav, ion = [], [], [], [], []
    skipped = Counter()
    truncated = Counter()
    n_eph = 0
    for head, body in _rinex4_records(lines, start):
        body = _record_lines(body)
        parts = head[1:].split()
        if len(parts) < 3:
            skipped["unreadable record line"] += 1
            continue
        kind, sv, message = parts[0], parts[1], (parts[2] if len(parts) > 2 else "")
        system = sv[:1]
        if kind == "ION":
            row = _ion4_row(system, sv, message, body)
            if row is None:
                skipped[f"ION {system} {message}"] += 1
            else:
                ion.append(row)
            continue
        if kind != "EPH":
            skipped[f"{kind} {system} {message}"] += 1
            continue
        n_eph += 1
        if system == "G" and "G" in req and message in ("LNAV", "CNAV"):
            layout = _GPS_LNAV if message == "LNAV" else _GPS_CNAV
            if len(body) < len(layout) + 1:
                truncated["G"] += 1
                continue
            try:
                s, epoch, clock = _clock_line(body[0])
            except ValueError:
                truncated["G"] += 1
                continue
            row = _orbit_row(s, epoch, clock, layout, body[1:len(layout) + 1])
            if message == "LNAV" and not _complete(row, "G"):
                truncated["G"] += 1
                continue
            if message == "LNAV":
                row["nav_message"] = message
                gps_rows.append(row)
            else:
                cnav.append({"base": row["sv"], "time": epoch, "ISC_L1CA": row["ISC_L1CA"],
                             "ISC_L2C": row["ISC_L2C"], "ISC_L5I5": row["ISC_L5I5"], "ISC_L5Q5": row["ISC_L5Q5"],
                             "TGD_CNAV": row["TGD"]})
        elif system == "E" and "E" in req and message in ("INAV", "FNAV"):
            if len(body) < 8:
                truncated["E"] += 1
                continue
            try:
                s, epoch, clock = _clock_line(body[0])
            except ValueError:
                truncated["E"] += 1
                continue
            row = _orbit_row(s, epoch, clock, _GAL, body[1:8])
            if not _complete(row, "E"):
                truncated["E"] += 1
                continue
            row["nav_message"] = message
            gal_rows.append(row)
        elif system == "C" and "C" in req and message in _BDS_MESSAGES:
            try:
                row = _bds4_row(message, body)
            except ValueError:
                row = None
            if row is None:
                truncated["C"] += 1
            else:
                bds_rows.append(row)
        else:
            skipped[f"EPH {system} {message}"] += 1
    if n_eph == 0:
        raise RinexNavError(f"{path} (RINEX {_fmt_version(info.version)}): no EPH record could be parsed; check that "
                            "the file is complete")
    for s, k in truncated.items():
        msg = f"{path}: {k} truncated or unreadable {_SYSTEM_NAMES[s]} navigation record(s) skipped"
        warnings.warn(msg, RinexWarning, stacklevel=3)
        notes.append(msg)
    if skipped:
        notes.append(f"{path}: RINEX 4 records not used: "
                     + ", ".join(f"{k} {v}" for k, v in sorted(skipped.items())))
    frames = {}
    if "G" in req:
        columns = list(_CLOCK) + [n for names in _GPS_LNAV for n in names]
        if gps_rows:
            out = _suffix_copies(pd.DataFrame(gps_rows))
            msg_col = out.set_index(["time", "sv"]).sort_index()["nav_message"].to_numpy()
            frame = _time_major(out, columns)
            frame.insert(0, "nav_message", msg_col)
        else:
            frame = _empty_frame(["nav_message"] + columns)
        frames["G"] = _attach_cnav(frame, cnav)
    if "E" in req:
        frames["E"] = _galileo_frame(gal_rows, path, True, notes)
    if "C" in req:
        if bds_rows:
            out = pd.DataFrame(bds_rows).set_index(["sv", "time"]).sort_index()
            out.attrs["rinex4_bds_message_types"] = sorted(out["nav_message"].dropna().unique().tolist())
        else:
            out = _empty_frame(["nav_message"])
        frames["C"] = out
    for frame in frames.values():
        frame.attrs["rinex_version"] = info.version
        frame.attrs["rinex4_skipped"] = dict(skipped)
    return frames, ion


# ------------------------------------------------------------------------------------------------ RINEX 2 body
#: number of lines of a RINEX 2 GPS record: PRN / EPOCH / SV CLK and BROADCAST ORBIT 1-7 (RINEX 2.11 Table A4)
_R2_RECORD_LINES = 8

#: the 31 values of a RINEX 2 GPS record in file order: SV clock (3), BROADCAST ORBIT 1-7 (4 each; the last two
#: of BROADCAST ORBIT 7 are spares). The frame keeps the first 29 (through ``FitIntvl``), the names georinex gave.
_R2_GPS_NAMES = _CLOCK + tuple(n for names in _GPS_LNAV for n in names)
_R2_FRAME_NAMES = _R2_GPS_NAMES[:-2]

#: values compared to tell a copy of the same ephemeris from a conflicting record: SV clock (3), BROADCAST ORBIT 1-6
#: (IODE ... IODC); BROADCAST ORBIT 7 (transmission time, fit interval, spares) may differ between copies
_R2_EPHEMERIS_VALUES = slice(0, 3 + 6 * 4)
_R2_TRANS_TIME = 3 + 6 * 4

#: positions (in the 31 values of a record) of the fields without which ``adapt_rinex2_gps_nav`` drops the record
_R2_REQUIRED = np.array([k for k, name in enumerate(_R2_GPS_NAMES)
                         if name in GPS_NAV_REQUIRED_CORE or name == "TransTime"])


def _r2_record_start(line: str) -> bool:
    """RINEX 2.11 Table A4: I2,1X,I2.2,... (PRN in columns 1-2); BROADCAST ORBIT lines start with 3X."""
    return len(line) >= 22 and line[0:2].strip().isdigit() and line[2:3] == " "


def _r2_fields(line: str, start: int, count: int) -> list:
    """``count`` D19.12 fields of a RINEX 2 line from column ``start`` (0-based). A blank or unreadable slot, a slot
    past the end of the line, and a slot the line ends inside that does not hold a complete number (a number cut
    short, :func:`_slot`) are NaN."""
    return [_slot(line, start + 19 * k, 19) for k in range(count)]


def _r2_record(block: list):
    """One RINEX 2 GPS record (RINEX 2.11 Table A4): ``(key, toc, values)``.

    PRN / EPOCH / SV CLK: I2, 5(1X,I2.2) (year as two digits: 80-99 -> 19xx, 00-79 -> 20xx), F5.1 seconds, 3D19.12
    from column 23; BROADCAST ORBIT 1-7: 3X, 4D19.12. Each field is read from its own 19-character slot
    (:func:`_r2_fields`), so a blank field or a line without its trailing fields (as the format allows for the last
    line) gives NaN in those slots and never shifts a later field; 'D', 'd', 'E', 'e' exponents and an exponent
    written without its letter ('0.1770-07', which a Fortran D19.12 read accepts) are read (:func:`_number`); columns
    after 79 are not read. ``values`` holds 31 floats (:data:`_R2_GPS_NAMES`). Raises ValueError for an unreadable
    PRN or epoch."""
    head = block[0]
    prn = int(head[0:2])
    year = int(head[2:5])
    year += 1900 if year >= 80 else 2000
    sec = float(head[17:22])
    key = (prn, year, int(head[5:8]), int(head[8:11]), int(head[11:14]), int(head[14:17]), sec)
    sec_int = int(sec)
    toc = pd.Timestamp(datetime(year, key[2], key[3], key[4], key[5], sec_int,
                                int(round((sec - sec_int) * 1_000_000))))
    values = _r2_fields(head, 22, 3) + [v for line in block[1:] for v in _r2_fields(line, 3, 4)]
    return key, toc, np.array(values, dtype=float)


def _r2_rows(lines: list, start: int, path, notes: list) -> list:
    """The records of a RINEX 2 GPS navigation file as ``(prn, toc, values)``, in file order (the
    readability rules of the module docstring).

    A copy of the same ephemeris (equal clock and orbit values; the transmission time, fit interval and spares may
    differ) is dropped with a note and the earliest transmission kept; a conflicting copy raises
    :class:`RinexNavError` (R2OBSE04). A record cut short (fewer than eight non-blank lines before the next record or
    the end of the file), with an unreadable PRN or epoch, or with a blank or unreadable clock or orbit field of
    ``GPS_NAV_REQUIRED_CORE`` (which the adapter would drop silently) or transmission time is skipped with one
    warning per file. Blank lines inside a record are skipped; lines outside any record (blank lines, stray text)
    are ignored."""
    rows = []
    seen = {}                       # key -> position in ``rows``
    dropped = []
    bad = 0
    i, n = start, len(lines)
    while i < n:
        if not _r2_record_start(lines[i]):
            i += 1
            continue
        j = i + 1
        block = [lines[i]]
        while j < n and len(block) < _R2_RECORD_LINES and not _r2_record_start(lines[j]):
            if lines[j].strip():            # a blank line inside a record carries no field (_record_lines)
                block.append(lines[j])
            j += 1
        i = j
        if len(block) < _R2_RECORD_LINES:
            bad += 1
            continue
        try:
            key, toc, values = _r2_record(block)
        except ValueError:
            bad += 1
            continue
        if np.isnan(values[_R2_REQUIRED]).any():
            bad += 1
            continue
        if key in seen:
            pos = seen[key]
            first = rows[pos][2]
            if np.array_equal(first[_R2_EPHEMERIS_VALUES], values[_R2_EPHEMERIS_VALUES], equal_nan=True):
                dropped.append(key)
                t_first, t_new = first[_R2_TRANS_TIME], values[_R2_TRANS_TIME]
                if np.isfinite(t_new) and (not np.isfinite(t_first) or t_new < t_first):
                    rows[pos] = (key[0], toc, values)                      # keep the earliest transmission
                continue
            prn, y, mo, d, h, mi, sec = key
            raise RinexNavError(
                f"{path}: RINEX 2 navigation record of G{prn:02d} at {y:04d}-{mo:02d}-{d:02d} "
                f"{h:02d}:{mi:02d}:{sec:04.1f} appears twice with different values (R2OBSE04: duplicated "
                "(sv, time)); remove the wrong copy")
        seen[key] = len(rows)
        rows.append((key[0], toc, values))
    if bad:
        msg = f"{path}: {bad} truncated or unreadable GPS navigation record(s) skipped"
        warnings.warn(msg, RinexWarning, stacklevel=4)
        notes.append(msg)
    if dropped:
        notes.append(f"{path}: {len(dropped)} duplicate RINEX 2 navigation record(s) of the same ephemeris dropped "
                     "(earliest transmission kept): "
                     + ", ".join(f"G{k[0]:02d} {k[1]:04d}-{k[2]:02d}-{k[3]:02d} {k[4]:02d}:{k[5]:02d}"
                                 for k in dropped[:5]) + (" ..." if len(dropped) > 5 else ""))
    return rows


def _read_rinex2(lines: list, start: int, info: NavInfo, path, notes: list) -> pd.DataFrame:
    """GPS frame of a RINEX 2 navigation file (file type N): the records of :func:`_r2_rows` as a frame indexed
    (time, sv) with the 29 columns of :data:`_R2_FRAME_NAMES`, normalised by ``rinex2_adapter.adapt_rinex2_gps_nav``
    (index (sv, time), ``GPS_NAV_INTERNAL_COLUMNS``, ``attrs['gnx_adapter']``)."""
    rows = _r2_rows(lines, start + 1, path, notes)
    if not rows:
        return _empty_frame(list(_CLOCK))
    index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([r[1] for r in rows]).as_unit("ns"),
                                       [f"G{r[0]:02d}" for r in rows]], names=["time", "sv"])
    raw = pd.DataFrame(np.vstack([r[2] for r in rows])[:, :len(_R2_FRAME_NAMES)], index=index,
                       columns=list(_R2_FRAME_NAMES))
    version = {"version": info.version}
    try:
        return adapt_rinex2_gps_nav(raw, header=version, rinex_info=version, system="G", strict=True,
                                    nav_path=os.fspath(path))
    except ValueError as exc:
        raise RinexNavError(f"{path} (RINEX {_fmt_version(info.version)}, GPS): the navigation records cannot be "
                            f"read ({exc})") from None


# ------------------------------------------------------------------------------------------- ionosphere table
def _iono_fields(line: str, start: int, free: tuple) -> list:
    """Four D12.4 fields from column ``start``; when one does not read, the free-format tokens of
    ``line[free[0]:free[1]]`` (writers that do not keep the columns) if they give more values."""
    vals = _floats(line, start=start, count=4, width=12)
    payload = line[free[0]:free[1]]
    if any(np.isnan(v) for v in vals) and payload.strip():
        # free-format fallback (writers that do not keep the D12.4 columns)
        parsed = [_number(t) for t in payload.split()][:4]
        if any(np.isnan(v) for v in parsed):
            parsed = []
        if len(parsed) > sum(not np.isnan(v) for v in vals):
            vals = parsed + [np.nan] * (4 - len(parsed))
    return vals


def _header_iono(lines: list, end: int, major: int) -> tuple:
    """Header ionosphere records: RINEX 2 ION ALPHA / ION BETA (2X,4D12.4) and RINEX 3/4 IONOSPHERIC CORR
    (A4,1X,4D12.4,1X,A1,1X,I2: label, values, time mark, SV id). Returns (rows, notes)."""
    rows, notes = [], []
    if major == 2:
        rec = {}
        for line in lines[:end]:
            label = line[60:].strip()
            if label in ("ION ALPHA", "ION BETA"):
                rec[label] = _iono_fields(line, 2, (0, 60))
        if "ION ALPHA" in rec and "ION BETA" in rec:
            rows.append({"system": "G", "sv": None, "message": "ION ALPHA", "model": "KLOB", "t_tx": pd.NaT,
                         "coeffs": tuple(rec["ION ALPHA"] + rec["ION BETA"]), "source": "R2 header"})
        elif rec:
            notes.append("header ION ALPHA without ION BETA (or the reverse); not used")
        return rows, notes
    corr = []
    for line in lines[:end]:
        if line[60:].strip() != "IONOSPHERIC CORR":
            continue
        key = line[0:4].strip()
        mark = line[54:55].strip() or None
        sv_id = line[56:58].strip()
        corr.append((key, _iono_fields(line, 5, (4, 54)), mark, sv_id))
    used = set()
    for k, (key, vals, mark, sv_id) in enumerate(corr):
        if key == "GAL":
            rows.append({"system": "E", "sv": None, "message": "GAL", "model": "NEQG", "t_tx": pd.NaT,
                         "coeffs": tuple(vals[:3]), "source": "R3 header"})
            continue
        if key not in _ION_HEADER_PAIRS:
            if key not in {b for b, _ in _ION_HEADER_PAIRS.values()}:
                notes.append(f"IONOSPHERIC CORR {key!r} not used")
            continue
        partner, system = _ION_HEADER_PAIRS[key]
        match = next((m for m, (key2, _, mark2, sv2) in enumerate(corr)
                      if m not in used and key2 == partner and mark2 == mark and sv2 == sv_id), None)
        if match is None:
            notes.append(f"IONOSPHERIC CORR {key} {mark or ''} {sv_id} without its {partner} record; not used")
            continue
        used.add(match)
        sv = f"{system}{int(sv_id):02d}" if sv_id.isdigit() else None
        rows.append({"system": system, "sv": sv, "message": key, "model": "KLOB", "t_tx": pd.NaT,
                     "coeffs": tuple(vals + corr[match][1]), "source": "R3 header"})
    return rows, notes


def _bdgim_word_check(c) -> tuple:
    """(raw words, problem) of the alpha values ``c`` [TECU] (BDS-SIS-ICD-B2b-1.0 Table 7-8): the nine unsigned
    broadcast words and None, or None and the first problem, naming the coefficient (off the 1/8 TECU grid beyond
    print rounding, or outside the range of its word). The value itself is in the record the warning names; the
    problem text has a bounded set of forms so that per-file summaries of reasons stay short.

    alpha5's negative scale factor is applied before the words are formed: the word is alpha5 / (-1/8)."""
    out = np.empty(len(_BDGIM_BITS), dtype=np.int64)
    for i, (value, bits, signed, scale) in enumerate(zip(c, _BDGIM_BITS, _BDGIM_SIGNED, _BDGIM_SCALE)):
        q = float(value) / scale
        k = round(q)
        if abs(q - k) > _BDGIM_GRID_TOL:
            return None, f"BDGIM alpha{i + 1} not on the 1/8 TECU grid (BDS-SIS-ICD-B2b Table 7-8)"
        lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        if not lo <= k <= hi:
            v_lo, v_hi = sorted((lo * scale, hi * scale))
            return None, (f"BDGIM alpha{i + 1} outside its BDS-SIS-ICD-B2b Table 7-8 range "
                          f"{v_lo + 0.0:g}..{v_hi + 0.0:g} TECU")
        out[i] = k % (1 << bits)
    return out, None


def _bdgim_raw_words(c) -> Optional[np.ndarray]:
    """The nine unsigned BDGIM broadcast words of ``c`` [TECU], or None (see :func:`_bdgim_word_check`)."""
    return _bdgim_word_check(c)[0]


def _screen(row: dict) -> Optional[str]:
    """Why a broadcast ionosphere set is not plausible (None when it is), per model (module docstring)."""
    c = np.asarray(row["coeffs"], dtype=float)
    model = row["model"]
    if c.size != IONO_MODELS[model] or not np.all(np.isfinite(c)):
        return "missing or unreadable coefficient"
    if model == "NEQG":
        return None if c[0] > 0 else f"NeQuick ai0 = {c[0]:g} <= 0"
    if model == "BDGIM":
        raw, problem = _bdgim_word_check(c)
        if raw is None:
            return problem
        low = raw % 256
        if np.all(low == low[0]):
            return f"all nine raw BDGIM words equal 0x{int(low[0]):02X} modulo 256 (filler record)"
        return None
    counts = np.round(c / _KLOB_SCALE)
    if np.all(counts == counts[0]):
        return f"all eight raw Klobuchar counts equal {int(counts[0])} (filler record)"
    alpha = c[:4]
    if row["system"] == "J":
        # QZSS LNAV/CNAV sets are fitted to the area around Japan (IS-QZSS-PNT 5.2.2.2.5.2) and are not bounded
        # over all latitudes; GNX does not use them, so only the filler test applies
        return None
    if row["system"] == "C":
        amp = np.polyval(alpha[::-1], _PHI_BDS)
        peak = float(np.max(amp))
        if peak <= 0:
            return "BeiDou Klobuchar amplitude A2 <= 0 for every |latitude| <= 1/3 semicircle (no daytime term)"
    else:
        amp = np.polyval(alpha[::-1], _PHI_GPS)
        peak = float(max(np.max(amp), 0.0))
    if _KLOB_NIGHT_S + peak > _KLOB_MAX_S:
        return f"Klobuchar vertical delay reaches {(_KLOB_NIGHT_S + peak) * 1e9:.1f} ns (> 100 ns)"
    return None


def _iono_frame(rows: list) -> pd.DataFrame:
    for row in rows:
        if row["system"] == "C" and row["sv"]:
            try:
                row["sv_class"] = bds_orbit_type(row["sv"])
            except ValueError:
                row["sv_class"] = None
        else:
            row["sv_class"] = None
        reason = _screen(row)
        row["plausible"] = reason is None
        row["reason"] = reason
    frame = pd.DataFrame(rows, columns=list(IONO_COLUMNS))
    frame["t_tx"] = pd.to_datetime(frame["t_tx"]).astype("datetime64[ns]")
    frame["plausible"] = frame["plausible"].astype(bool)
    frame = frame.astype({"system": object, "sv": object, "message": object, "model": object, "source": object,
                          "sv_class": object, "reason": object})
    return frame


def _iono_label(row) -> str:
    if row.source == "R4 ION":
        return f"'> ION {row.sv} {row.message}' {row.t_tx}"
    return f"header {row.message}" + (f" {row.sv}" if row.sv else "")


def _warn_rejected(frame: pd.DataFrame, path) -> None:
    for row in frame.loc[~frame["plausible"]].itertuples(index=False):
        warnings.warn(f"{path}: broadcast ionosphere record {_iono_label(row)} rejected: {row.reason}", RinexWarning,
                      stacklevel=3)


def _build_iono(text: str, path, ion4: Optional[list] = None) -> tuple:
    """(table, notes) of a file; ``ion4``: its '> ION' rows when the RINEX 4 body was already walked."""
    lines = text.splitlines()
    info = _parse_info(text, path)
    end = _header_end(lines, path)
    rows, notes = _header_iono(lines, end, int(info.version))
    notes = [f"{path}: {note}" for note in notes]
    if int(info.version) >= 4:
        if ion4 is None:
            ion4 = []
            for head, body in _rinex4_records(lines, end + 1):
                parts = head[1:].split()
                if len(parts) >= 3 and parts[0] == "ION":
                    row = _ion4_row(parts[1][:1], parts[1], parts[2], _record_lines(body))
                    if row is not None:
                        ion4.append(row)
        rows.extend(dict(row) for row in ion4)
    return _iono_frame(rows), notes


def _iono_cached(path, ion4: Optional[list] = None) -> tuple:
    key = _file_key(path)
    with _CACHE_LOCK:
        hit = _IONO_CACHE.get(key)
        if hit is not None:
            _IONO_CACHE.move_to_end(key)
    if hit is None:
        hit = _build_iono(_text(path), path, ion4)
        with _CACHE_LOCK:
            _IONO_CACHE[key] = hit
            while len(_IONO_CACHE) > _IONO_CACHE_SIZE:
                _IONO_CACHE.popitem(last=False)
    return hit


def read_iono_table(path, systems: Optional[Iterable[str] | str] = None) -> pd.DataFrame:
    """The ionosphere table (:data:`IONO_COLUMNS`) of ``path``: header records and RINEX 4 '> ION' records,
    screened. Cached per file (path, size, modification time); every call returns a new frame.

    ``systems`` (for example ``"GE"``): a :class:`RinexWarning` only for the rejected records of these systems and
    GPS, whose Klobuchar set every system may use (the rule of :func:`read_navigation`); None warns for every
    system. The table always holds every record."""
    frame = _iono_cached(path)[0].copy()
    _warn_rejected(frame if systems is None else frame[frame["system"].isin(_warn_scope(systems))], path)
    return frame


def _warn_scope(systems) -> set:
    """The systems whose rejected ionosphere records warn: ``systems`` plus GPS."""
    return {c for c in "".join(str(s) for s in systems).upper() if c.isalpha()} | {"G"}


def _priority(frame: pd.DataFrame) -> pd.DataFrame:
    """Plausible sets in selection order: BeiDou class (GEO, IGSO, MEO, unknown), header before broadcast
    records, then message (LNAV, CNVX, IFNV), then transmission time, then file order."""
    if frame.empty:
        return frame
    out = frame.assign(
        __cls=frame["sv_class"].map(lambda c: _CLASS_ORDER.get(c, 3)).where(frame["system"] == "C", 0),
        __src=frame["source"].map(lambda s: 0 if s != "R4 ION" else 1),
        __msg=frame["message"].map(lambda m: _LEGACY_ORDER.get(m, 0)),
        __t=frame["t_tx"].fillna(pd.Timestamp.min),
        __pos=np.arange(len(frame)))
    out = out.sort_values(["__cls", "__src", "__msg", "__t", "__pos"], kind="stable")
    return out.drop(columns=["__cls", "__src", "__msg", "__t", "__pos"])


def legacy_iono(iono: pd.DataFrame) -> dict:
    """``{'gpsa', 'gpsb', 'gala', 'bdsa', 'bdsb'}`` from the ionosphere table (selection rules in the module
    docstring); a value is a list of floats or None."""
    out = {"gpsa": None, "gpsb": None, "gala": None, "bdsa": None, "bdsb": None}
    if iono is None or iono.empty:
        return out
    ok = iono[iono["plausible"].astype(bool)]
    for system, model, keys in (("G", "KLOB", ("gpsa", "gpsb")), ("C", "KLOB", ("bdsa", "bdsb")),
                                ("E", "NEQG", ("gala",))):
        cand = _priority(ok[(ok["system"] == system) & (ok["model"] == model)])
        if cand.empty:
            continue
        coeffs = [float(v) for v in cand["coeffs"].iloc[0]]
        if model == "KLOB":
            out[keys[0]], out[keys[1]] = coeffs[:4], coeffs[4:8]
        else:
            out[keys[0]] = coeffs[:3]
    return out


def broadcast_iono_at(iono: pd.DataFrame, system: str, model: str, epochs) -> np.ndarray:
    """Coefficients in force at each of ``epochs`` (GPST, array-like of datetime64), shape (n, k) with k =
    ``IONO_MODELS[model]``; rows of NaN when the table has no plausible set of that system and model.

    The BeiDou class priority (GEO, IGSO, MEO) applies to the BeiDou Klobuchar ('C', 'KLOB') only; a BDGIM set
    is global and is taken from any satellite or class."""
    if model not in IONO_MODELS:
        raise ValueError(f"unknown broadcast ionosphere model {model!r}; one of {sorted(IONO_MODELS)}")
    k = IONO_MODELS[model]
    t = np.atleast_1d(np.asarray(pd.to_datetime(np.atleast_1d(epochs)), dtype="datetime64[ns]"))
    out = np.full((t.size, k), np.nan)
    if iono is None or iono.empty:
        return out
    cand = iono[(iono["system"] == system) & (iono["model"] == model) & iono["plausible"].astype(bool)]
    if cand.empty:
        return out
    if system == "C" and model == "KLOB":
        cls = cand["sv_class"].map(lambda c: _CLASS_ORDER.get(c, 3))
        cand = cand[cls == cls.min()]
    header = _priority(cand[cand["t_tx"].isna()])
    dated = cand[cand["t_tx"].notna()].assign(__pos=np.arange(int(cand["t_tx"].notna().sum())))
    dated = dated.sort_values(["t_tx", "__pos"], kind="stable")
    base = np.array(header["coeffs"].iloc[0], dtype=float) if not header.empty else None
    if dated.empty:
        out[:] = base
        return out
    times = dated["t_tx"].to_numpy(dtype="datetime64[ns]")
    sets = np.array([np.asarray(c, dtype=float) for c in dated["coeffs"]])
    idx = np.searchsorted(times, t, side="right") - 1
    out[idx >= 0] = sets[idx[idx >= 0]]
    out[idx < 0] = base if base is not None else sets[0]
    return out


# ------------------------------------------------------------------------------------------------- navigation
def read_navigation(path, systems: Iterable[str] | str) -> NavRead:
    """Broadcast ephemerides of ``systems`` and the ionosphere table of ``path`` (interface in the module docstring).

    Calls :func:`check_nav_systems` first; an empty requested system after the read raises
    :class:`RinexNavError`."""
    info = check_nav_systems(path, systems)
    req = _requested(systems, path)
    text = _text(path)
    lines = text.splitlines()
    end = _header_end(lines, path)
    notes: list = []
    major = int(info.version)
    ion4 = None
    if major == 2:
        frames = {"G": _read_rinex2(lines, end, info, path, notes)}
    elif major == 3:
        frames = _read_rinex3(lines, end + 1, req, path, notes)
    else:
        frames, ion4 = _read_rinex4(lines, end + 1, req, info, path, notes)
    table, iono_notes = _iono_cached(path, ion4)
    iono = table.copy()
    # a warning for the records of the requested systems and GPS (whose Klobuchar set every system may use by
    # default); the notes list every rejected record
    _warn_rejected(iono[iono["system"].isin(_warn_scope(req))], path)
    notes.extend(iono_notes)
    for row in iono.loc[~iono["plausible"]].itertuples(index=False):
        notes.append(f"{path}: broadcast ionosphere record {_iono_label(row)} rejected: {row.reason}")
    empty = [s for s in req if frames.get(s) is None or frames[s].empty]
    if empty:
        raise RinexNavError(f"{path} (RINEX {_fmt_version(info.version)}, systems in file: "
                            f"{', '.join(info.systems) or 'none'}) has no readable {_names(empty)} ephemerides")
    for frame in frames.values():
        frame.attrs["nav_path"] = os.fspath(path)
    return NavRead(gps=frames.get("G"), gal=frames.get("E"), bds=frames.get("C"), iono=iono, info=info,
                   notes=notes)


# ----------------------------------------------------------------------------------------------------- health
def _sow(times: pd.DatetimeIndex) -> np.ndarray:
    """Seconds of the (Sunday-based) week of ``times`` in their own time scale."""
    return (((times.dayofweek + 1) % 7) * 86400 + (times - times.normalize()).total_seconds()).to_numpy(float)


def _record_times(nav: pd.DataFrame, system: str) -> pd.DatetimeIndex:
    """Transmission time of each record in GPST (TransTime referred to the toc's week, wrapped to within half a
    week of toc), toc when the transmission time is missing or 'not known'; BDT moved by +14 s."""
    toc = pd.DatetimeIndex(nav.index.get_level_values("time")).astype("datetime64[ns]")
    if "TransTime" in nav.columns:
        trans = pd.to_numeric(nav["TransTime"], errors="coerce").to_numpy(float)
    else:
        trans = np.full(len(nav), np.nan)
    ok = np.isfinite(trans) & (trans < _TRANS_UNKNOWN_S) & (np.abs(trans) < 3 * _WEEK_S)
    d = np.where(ok, trans, 0.0) - _sow(toc)
    d = np.where(ok, (d + _WEEK_S / 2) % _WEEK_S - _WEEK_S / 2, 0.0)
    t = toc + pd.to_timedelta(np.round(d * 1e9).astype(np.int64), unit="ns")
    if system == "C":
        t = t + _BDT_TO_GPST
    return t


def _flag(values, test) -> np.ndarray:
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return np.where(np.isfinite(v), test(v).astype(float), np.nan)


def health_table(nav_df: Optional[pd.DataFrame], system: str) -> HealthTable:
    """:class:`HealthTable` of the navigation frame ``nav_df`` of ``system`` ('G', 'E' or 'C'); an empty table for
    None or an empty frame."""
    system = str(system).upper()
    if system not in HEALTH_COLUMNS:
        raise ValueError(f"Unknown sys: {system}")
    cols = HEALTH_COLUMNS[system]
    path = None if nav_df is None else nav_df.attrs.get("nav_path")
    if nav_df is None or nav_df.empty:
        empty = pd.DataFrame({"sv": pd.Series(dtype=object), "t": pd.Series(dtype="datetime64[ns]"),
                              "message": pd.Series(dtype=object), **{c: pd.Series(dtype=float) for c in cols}})
        return HealthTable(system, empty, cols, HEALTH_MAX_AGE[system], path)
    sv = pd.Index(nav_df.index.get_level_values("sv")).astype(str).str.split("_").str[0]
    t = _record_times(nav_df, system)
    has_msg = "nav_message" in nav_df.columns
    if system == "G":
        message = nav_df["nav_message"].fillna("LNAV").to_numpy(object) if has_msg else np.full(len(nav_df), "LNAV")
        flags = {cols[0]: _flag(nav_df["health"], lambda v: v != 0)}
    elif system == "C":
        if has_msg:
            message = nav_df["nav_message"].to_numpy(object)
        else:
            message = np.array(["D2" if bds_orbit_type(s) == "GEO" else "D1" for s in sv], dtype=object)
        source = nav_df["SatH1"] if "SatH1" in nav_df.columns else nav_df["health"]
        if "SatH1" in nav_df.columns and "health" in nav_df.columns:
            source = nav_df["SatH1"].where(nav_df["SatH1"].notna(), nav_df["health"])
        flags = {cols[0]: _flag(source, lambda v: v != 0)}
    else:
        if has_msg:
            message = nav_df["nav_message"].to_numpy(object)
        else:
            src = pd.to_numeric(nav_df["DataSrc"], errors="coerce").to_numpy(float)
            fnav = np.isfinite(src) & ((np.where(np.isfinite(src), src, 0).astype(np.int64) >> 1) & 1).astype(bool)
            message = np.where(fnav, "FNAV", "INAV").astype(object)
        h = pd.to_numeric(nav_df["health"], errors="coerce").to_numpy(float)
        valid = np.isfinite(h) & (h >= 0) & (h <= 511) & (np.round(h) == h)
        w = np.where(valid, np.round(np.where(valid, h, 0.0)), 0).astype(np.int64)
        fields = {"E1B_DVS": (w & 1) == 1, "E1B_SHS": np.isin((w >> 1) & 3, (1, 3)),
                  "E5a_DVS": ((w >> 3) & 1) == 1, "E5a_SHS": np.isin((w >> 4) & 3, (1, 3)),
                  "E5b_DVS": ((w >> 6) & 1) == 1, "E5b_SHS": np.isin((w >> 7) & 3, (1, 3))}
        is_f = message == "FNAV"
        flags = {}
        for name, bad in fields.items():
            value = np.where(valid, bad, True).astype(float)          # undecodable word: not usable
            carried = is_f if name.startswith("E5a") else ~is_f
            flags[name + "_outage"] = np.where(carried, value, np.nan)
    records = pd.DataFrame({"sv": np.asarray(sv, dtype=object), "t": t.to_numpy(), "message": message, **flags})
    records = records.sort_values("t", kind="stable").reset_index(drop=True)
    return HealthTable(system, records, cols, HEALTH_MAX_AGE[system], path)


def health_flags_at(table: HealthTable, obs_index: pd.MultiIndex) -> pd.DataFrame:
    """Boolean outage flags in force at the rows of ``obs_index`` (levels 'sv' and 'time', either order, any row
    order; time naive GPST): a frame with the same index, in the same row order, and the columns
    ``table.flag_columns``. Per satellite and message type: the latest record at or before the epoch within
    ``table.max_age``; where there is none, the next record within ``table.max_age`` ahead (the ephemeris selector
    uses that record too); False where neither exists."""
    n = len(obs_index)
    out = {c: np.zeros(n, dtype=bool) for c in table.flag_columns}
    rec = table.records
    if n and not rec.empty:
        times = pd.DatetimeIndex(obs_index.get_level_values("time"))
        if times.tz is not None:
            raise ValueError(f"health_flags_at: the observation times are timezone-aware ({times.tz}); GNX times "
                             "are naive GPST")
        obs = pd.DataFrame({"sv": pd.Index(obs_index.get_level_values("sv")).astype(str).to_numpy(object),
                            "time": times.astype("datetime64[ns]"), "__row": np.arange(n)})
        obs = obs.sort_values("time", kind="stable")
        rows = obs["__row"].to_numpy()
        for col in table.flag_columns:
            for message, sub in rec.loc[rec[col].notna(), ["sv", "t", "message", col]].groupby("message", sort=False):
                right = (sub.groupby(["sv", "t"], as_index=False)[col].max()
                         .rename(columns={"t": "time"}).sort_values("time", kind="stable"))
                right["time"] = right["time"].astype("datetime64[ns]")
                value = pd.merge_asof(obs, right, on="time", by="sv", direction="backward",
                                      tolerance=table.max_age)[col].to_numpy(float)
                none = np.isnan(value)
                if none.any():
                    ahead = pd.merge_asof(obs.loc[none], right, on="time", by="sv", direction="forward",
                                          tolerance=table.max_age)[col].to_numpy(float)
                    value[none] = ahead
                out[col][rows] |= value == 1.0
    return pd.DataFrame(out, index=obs_index)
