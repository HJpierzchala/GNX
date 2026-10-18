"""RINEX 2/3/4 observation reader.

One reader for RINEX 2.10/2.11, 3.0x and 4.0x observation files, plain or compressed: gzip, Unix compress (.Z),
bzip2 and zip through :mod:`gnx_py.compression`, and Hatanaka CRINEX 1/3 through the RNXCMP ``crx2rnx`` program
bundled with the ``hatanaka`` package (MIT; RNXCMP falls under the GSI terms, Hatanaka 2008), started through
:func:`gnx_py._proc.run` (posix_spawn, no fork) with the exit-code handling of
``hatanaka.crx2rnx``. Everything is decompressed in memory and nothing is
written to disk, so read-only directories and parallel reads of one file are safe. It replaces the gfzrnx
``tabulate`` path, georinex for observations and the private parser of :mod:`gnx_py.cycle_slips`; all of them
share the output contract below.

Format references: RINEX 2.11 (IGS, Table A1/A2), RINEX 3.05 and 4.02 (IGS/RTCM, Tables A2/A3, §4.1.4, §5.4,
§6.7). Written from the specifications; no GPL code.

Interface (stable; additions only with defaults)
------------------------------------------------
Errors and warnings

* :class:`RinexObsError` (``ValueError``): a file that cannot be read; the message names the file and the problem
  (not an observation file, unreadable version, no END OF HEADER, decompression failure, GLONASS/UTC time tags,
  observation types changed by an event-flag-3/4 block inside the data or by a second header inside the data
  of a concatenated file, a RINEX 2 type list with fewer codes than it declares).
* :class:`RinexWarning` (``UserWarning``): one warning per file and kind for a readable defect (duplicate
  (sv, time) records, a truncated last epoch, a cut compressed file, flag 2/3 events, RINEX 3.02 BeiDou with
  both band 1 and band 2, headers inside the data of a concatenated file, RINEX 2 epochs skipped whole, value
  fields that are not numbers, records wider than the types, a RINEX 3/4 system whose type list cannot be used,
  stray bytes after the gzip data).

Header: :func:`read_header` returns an :class:`ObsHeader` (fields documented on the class). Every record is read
by column. ``approx_position`` is None when the record is missing, unreadable, zero or off the Earth's surface
(``|r|`` outside :data:`APPROX_XYZ_MIN_M` .. :data:`APPROX_XYZ_MAX_M`, the bounds shared with
``cycle_slips.GEO_XYZ_MIN_M/MAX_M`` and ``gnx_py.apriori``); each such case adds a readable entry to ``issues``.

Data: :func:`read_obs` returns an :class:`ObsRead`. Frames (one per satellite system):

* index ``['sv', 'time']``, lexsorted and unique; ``sv`` like ``'G01'``; ``time`` datetime64[ns] in GPST;
* float64 columns, one per observation type of the requested kinds, in header order (RINEX 3/4 codes such as
  ``'C1C'``; RINEX 2 codes as written, ``'C1'``, ``'P2'``, ``'L5'``; the RINEX 2 band mapping stays in
  :mod:`gnx_py.rinex2_adapter`); a code never appears twice in a frame. The sessions then keep one code and one
  phase column per band of the mode (the signal priority of :mod:`gnx_py.gnss`), so a lookup by the
  two-character prefix (``'C1'``, ``'L2'``) finds one column whichever match it takes;
* only rows with at least one value of the requested kinds;
* blank fields and 0.0 are NaN for every observation type, Doppler included (RINEX 2.11 Table A2, 3.05 and
  4.02 §6.7); ``SYS / SCALE FACTOR`` is applied;
* BDT tags are moved to GPST (+14 s) before ``tlim`` is applied; ``tlim`` is inclusive and in GPST;
* epochs with flag 0 or 1 are data; flags 2-5 are skipped with their special records, flag 6 records are
  skipped; flags 2 (start of moving antenna) and 3 (new site occupation) give one :class:`RinexWarning` per file
  listing the epochs;
* ``attrs``: ``rinex_version``, ``time_system``, ``path``, ``epoch_flags`` (dict), ``issues`` (list of str), and
  ``phase_shift`` and ``phase_shift_events``, stored JSON-safe and
  read-only (``DataFrame.to_parquet`` writes ``attrs`` as JSON; pandas deep-copies ``attrs`` on nearly every
  operation that makes a new frame and these two share one immutable object, so the copy costs nothing however
  many records a file has). Read them in the contract form with :func:`phase_shift_of` and
  :func:`phase_shift_events_of` (also after a parquet / JSON round trip):

  - ``phase_shift_of(frame)``: the ``SYS / PHASE SHIFT`` records of the header, ``{(system, code): {'value':
    float [cycles], 'sats': list[str] | None, 'by_sat': dict[str, float] | None}}`` (:func:`phase_shift_map`;
    ``sats`` None = every satellite of the system; the satellite lists continue on the following lines). Stored
    as ``attrs['phase_shift'] = {'G L1C': {'value', 'sats', 'by_sat'}}`` (key "system code");
  - ``phase_shift_events_of(frame)``: the ``SYS / PHASE SHIFT`` records inside event-flag 2-5 blocks of the data,
    one tuple ``(epoch, system, code, sats, value)`` per record in file order (``epoch`` the event record's time
    as a GPST Timestamp; a block with blank epoch fields takes the next data epoch of the file, None when none
    follows); every block of the file, also outside ``tlim`` (an event before the window sets the shift in force
    in it). Stored as ``attrs['phase_shift_events'] = [[ISO epoch or None, system, code, sats, value], ...]``;
  - every system's records in every frame of the file (keys and records name the system); both empty for a file
    without such records and for RINEX 2 (the record does not exist there); the old text form stays in
    ``ObsHeader.phase_shift_text``. ``ObsHeader.phase_shift`` and ``ObsRead.phase_shift_events`` hold the contract
    form directly.

LLI frames (``lli=True``): the same index as the system's frame, one float column per phase code of the frame
(named as the phase, e.g. ``'L1C'``), values 0-7, 0 for a blank LLI, NaN where the phase is missing; bit 0 is
forced on every phase of a flag-1 epoch and the other bits are kept as written (bit 1, :data:`LLI_HALF_CYCLE`,
the half-cycle ambiguity of RINEX 3/4; :func:`lli_half_cycle` gives it as a boolean frame). LLI never
enters the observation frames (the phase screens ``filter(regex='^L')`` would see the columns).

Time system: from ``TIME OF FIRST OBS`` (columns 49-51 in RINEX 2 and 3/4; the first known token of columns
44-60, for writers that print the seconds one column wider), otherwise the file's own system in
single-system files; GPS, GAL, QZS and IRN tags are used as written (Galileo System Time is taken as GPST: the
GST - GPST offset, a few nanoseconds, is not applied), BDT is shifted by +14 s, GLO and UTC raise
:class:`RinexObsError` ("GLONASS/UTC time tags are not supported; convert to GPS time"); a mixed file without the
field is taken as GPS with a header issue.

Cache: process-wide LRU of one entry keyed by (file size, blake2b-128 digest of the raw bytes, ``tlim`` in ns),
guarded by a lock. The parsed arrays of every system are cached; every call builds new frames, so callers may
modify what they get. :func:`clear_cache` empties it.

Public API::

    read_header(path) -> ObsHeader
    read_obs(path, systems=None, tlim=None, kinds=("C", "P", "L", "S"), lli=False) -> ObsRead
    lli_for(path, system, phases, tlim=None) -> DataFrame
    lli_half_cycle(lli) -> DataFrame
    parse_phase_shift_records(payloads) -> (records, issues), phase_shift_map(records) -> (map, issues)
    phase_shift_of(frame | attrs) -> dict, phase_shift_events_of(frame | attrs) -> list
    clear_cache() -> None

Implementation notes
--------------------
* Additions with defaults: ``read_obs(..., ssi=False)`` returns the signal strength indicators in
  ``ObsRead.ssi`` (frames like the LLI frames, one column per observation type of the frame, digits 1-9, 0 for
  blank, NaN where the value is missing).
* Scan: the decompressed bytes are walked with ``bytes.find`` (no list of all lines is built). The records of
  each system are converted to fixed-width arrays and then to numbers in chunks of ``_PARSE_CHUNK`` records, so
  the text is never held twice; ``tlim`` (after the BDT shift) is applied while scanning. Measured (tracemalloc
  peak / decompressed size, time): G+E+C validation days with LLI 2.3-2.5 (0.85-0.88 s per 30 s station-day,
  Hatanaka + gzip), a synthetic 1 Hz hour 2.9 plain and Hatanaka + gzip (memory limit
  checked: at most 4), so the decompression is not streamed. The cache then holds about
  10 bytes per observation field (value, LLI, SSI) of every system.
* Epoch times are exact to the written digits (1 ns). RINEX 4.02 picoseconds (columns 58-62 of the epoch record,
  I5.5 after 1X) are read to the nanosecond (floored); a CRINEX round trip through RNXCMP 4.1.0 drops them (the
  tags keep 100 ns).
* RINEX 2: epoch flag in column 29, satellite count in columns 30-32, satellites from column 33 (12 per line,
  continuation lines), five observations per line (continuation lines), a blank system character is GPS
  ('G 7' is G07), years 80-99 are 19xx and 00-79 20xx. The one type list of the header applies to the records
  of every system letter in the data: ``systems=None`` gives the letters of ``ObsHeader.obs_types``, and a
  letter outside RINEX 2.11 (C, J, I, written by gfzrnx) is read when it is requested. RINEX 3/4: flag in
  column 32, satellite count in columns 33-35 (I3); after a truncated epoch the scan resynchronises on the next
  '>' record.
* LLI digits 0-7 are read as written; the undefined digits 8 and 9 (the LLI has three bits) are read as 0 and
  counted in one RinexWarning.
* A truncated last epoch keeps its complete records and gives one RinexWarning. A last line without a newline
  is dropped when the epoch declares more records after it or when it does not reach the value of its last field
  (writers end every line with a newline and drop trailing blank fields, so a shorter line may be cut at a field
  boundary), so a value or an LLI digit cut off never enters a frame and no value is lost silently.
* Cut compressed files (a cut download) are read like cut plain files: a gzip or bzip2 stream that ends before
  its end marker keeps its readable start; when crx2rnx stops with "truncated" at the end of its input, the
  RINEX text it wrote before stopping (a prefix of the full output) is read. Each gives one RinexWarning. Damage elsewhere (a CRC error, garbage in the
  middle of a CRINEX file) raises RinexObsError. The bytes read once for the cache key are the bytes
  decompressed (the file is read once).
* RINEX 2 resynchronisation: an epoch record found where a special record of an event block or an observation
  line is expected ends the block or the epoch early (as in RINEX 3/4); see the corpus review note below for what
  is kept.
* Duplicate (sv, time) records keep the first record of the file (one RinexWarning).
* RINEX 3.00-3.02 BeiDou: band-1 codes are band 2 (B1I; RINEX 3.05 Table 19 note). When the header lists the
  band-2 code as well, the band-2 field is read where it holds a value and the band-1 field otherwise (one
  RinexWarning).
* An event-flag-3 or -4 block that repeats ``SYS / # / OBS TYPES`` (or ``# / TYPES OF OBSERV``) with other types
  raises RinexObsError ("observation types change inside the data (event flag 4 at line n); split the file");
  repeating the header's own list is accepted.
* A whole header inside the data (a 'RINEX VERSION / TYPE' line where an epoch record is expected, up to its END
  OF HEADER: files joined with ``cat``; the format carries header records inside the data only in event-flag-3/4
  blocks) is read like the first header (joined CRINEX files are decompressed part by part, because crx2rnx would
  skip the second CRINEX header and decode the records after it with the first header's types). When it changes
  how the records after it are read (RINEX 2/3/4 layout, the RINEX 3.00-3.02 BeiDou rule, observation types,
  SYS / SCALE FACTOR, time system) RinexObsError ("a second header inside the data at line n ... changes how the
  records after it are read; split the file"), because those records would otherwise be read with the first
  header's types. Otherwise it is skipped (one RinexWarning per file, naming the line and any MARKER, receiver,
  antenna, DELTA H/E/N or a position more than 100 m away that differ: the observations are then read as one
  station). A header line found where a record line of a truncated epoch is expected ends that epoch (as an epoch
  record does).
* Corpus review (corpusfix2): RINEX 2 records carry no satellite id, so an epoch whose lines do not fit its
  satellite count (a new epoch record where a record line is expected, a record line with a value past the fields
  of the types, or a line left over before the next epoch record) is skipped whole with one RinexWarning; an
  epoch record whose satellite list does not read (an unreadable token, a count that does not match the tokens, a
  missing or unreadable continuation line) is skipped with its records. An epoch record with 11 or 12 satellites
  (system letters in columns 63 and 66) is recognised as one when the scan resynchronises. An epoch cut by the end
  of the file or by the header of a concatenated file keeps its complete records. Satellite numbers are read as a
  Fortran I2 (blanks ignored: 'G7 ' and 'G 7' are G07; blank or 00 unreadable). A value field that is not a
  number (Fortran overflow asterisks, junk, a NUL byte) is NaN and counted in one RinexWarning with the first
  such field. A type list with fewer codes than declared cannot be matched to the record fields: RINEX 2 raises,
  a RINEX 3/4 system is not read (one RinexWarning; ``ObsHeader.unreadable_types``); a RINEX 3/4 line repeating
  the system letter while the list is short of its count is a continuation line. Records holding values past
  the fields of the types are counted in one RinexWarning (those values are not read). A UTF-8 byte order mark
  is dropped and CR CR LF line ends read as one line end (:mod:`gnx_py.compression`); stray bytes after the gzip
  data are ignored with one RinexWarning, as gunzip ignores them.
* Blanks inside the seconds field of an epoch record or of TIME OF FIRST/LAST OBS are ignored, as a Fortran F
  read ignores them by default (``' 0 .0000000'`` is 0 s; georinex and RTKLIB read it so too); seconds 60 roll
  over into the next minute (``60.0000000``: GPS time has no leap second).
* Warnings are emitted when the file is parsed (a cache miss); a cache hit returns the same texts in
  ``ObsRead.notes`` without warning again. Texts, ``header.path`` and ``attrs`` name the path of the call, also
  when the cache entry was parsed from an identical file at another path.
"""
from __future__ import annotations

import bz2
import copy
import functools
import hashlib
import math
import os
import re
import threading
import warnings
import zipfile
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, NamedTuple, Optional

import numpy as np
import pandas as pd
from ..core import _proc
from . import compression

__all__ = [
    "APPROX_XYZ_MAX_M", "APPROX_XYZ_MIN_M", "BDT_TO_GPST_S", "OBS_KINDS", "ObsHeader", "ObsRead", "ReceiverInfo",
    "RinexObsError", "RinexWarning", "clear_cache", "lli_for", "read_header", "read_obs",
    "LLI_SLIP", "LLI_HALF_CYCLE", "LLI_BOC", "lli_half_cycle", "parse_phase_shift_records", "phase_shift_map",
    "phase_shift_of", "phase_shift_events_of",
]

#: Plausible geocentric distance of a station position from a header [m]; one pair of bounds shared with
#: ``cycle_slips.GEO_XYZ_MIN_M/MAX_M`` (aliases of these) and the a-priori position (``gnx_py.apriori``).
APPROX_XYZ_MIN_M = 6.30e6
APPROX_XYZ_MAX_M = 6.40e6

#: BDT = GPST - 14 s (BDS-SIS-ICD-B1I 3.0 §4.2): a BDT time tag is moved to GPST by adding this many seconds.
BDT_TO_GPST_S = 14

#: Kinds of observation types (first character of the code): code, P code (RINEX 2), phase, Doppler, SNR.
OBS_KINDS = ("C", "P", "L", "D", "S")

#: RINEX system letters a caller may request (RINEX 3.05 §3.5; RINEX 2.11 uses G, R, E, S).
_SYSTEM_LETTERS = "GRECJIS"

#: Records converted to numbers per chunk and system (bounds the text held at once).
_PARSE_CHUNK = 20000

#: Default time system of a single-system file (RINEX 3.05 Table A2, TIME OF FIRST OBS; SBAS: GPS).
_SYSTEM_TIME = {"G": "GPS", "R": "GLO", "E": "GAL", "C": "BDT", "J": "QZS", "I": "IRN", "S": "GPS"}
#: Time systems whose tags are used as written (aligned with GPS time), and their shift to GPST [s].
_TIME_SHIFT_S = {"GPS": 0, "GAL": 0, "QZS": 0, "IRN": 0, "BDT": BDT_TO_GPST_S}
#: Time system tokens of TIME OF FIRST OBS (RINEX 3.05 Table A2; GLO and UTC are refused by read_obs).
_KNOWN_TIME_SYSTEMS = ("GPS", "GLO", "GAL", "BDT", "QZS", "IRN", "UTC")

#: Units of the file-name data frequency token (RINEX 3.05 §A1): interval [s] per unit; C and Z are rates.
_NAME_RATE_HZ = {"C": 100.0, "Z": 1.0}
_NAME_STEP_S = {"S": 1.0, "M": 60.0, "H": 3600.0, "D": 86400.0}
_NAME_TOKEN = re.compile(r"_\d{2}[MHDYU]_(\d{2})([CZSMHDU])_[A-Z]{2}")

_NS = 1_000_000_000
_EPOCH0 = np.datetime64("1970-01-01T00:00:00", "ns")

#: Bits of the RINEX loss-of-lock indicator (LLI, 0-7) as the LLI frames carry it.
#: RINEX 3.03 §5.4/Table A3, 4.00 §5.2.13/§6.7.1: bit 0 lost lock between previous and this observation (slip possible;
#: the reader also sets it on every phase of a flag-1 epoch), bit 1 half-cycle ambiguity/slip possible ("software
#: not capable of handling half cycles should skip this observation"; valid for the current epoch only), bit 2 BOC
#: tracking of an MBOC-modulated signal (Galileo). RINEX 2.11 (Table A2, data record) defines bit 1 as "opposite
#: wavelength factor to the one of WAVELENGTH FACT L1/2" (a half-wavelength observation when the default factor is
#: 1) and bit 2 as observation under anti-spoofing. RTKLIB reads bit 1 the same way for every version (LLI_HALFC)
#: but uses it differently by mode: PPP resets the ambiguity on bit 0 or bit 1 (2.4.2 p13 ppp.c:715 detslp_ll
#: ``LLI & 3``; 2.5.1 ex ppp.c:462 ``LLI_SLIP | LLI_HALFC``); the half-cycle rule proper (a slip when bit 1
#: toggles, the ambiguity kept out of integer fixing while it is set) is relative positioning only (2.4.2
#: rtkpos.c:592; 2.5.1 ex rtkpos.c:708-720, 836-837 and ddidx :1532/:1548).
LLI_SLIP = 1
LLI_HALF_CYCLE = 2
LLI_BOC = 4


class RinexObsError(ValueError):
    """An observation file that cannot be read; the message names the file and the problem."""


class RinexWarning(UserWarning):
    """A readable defect of a RINEX file (one warning per file and kind); the message names the file."""


class ReceiverInfo(NamedTuple):
    """``REC # / TYPE / VERS``: serial (columns 1-20), type (21-40) and firmware version (41-60); None if blank."""

    serial: Optional[str]
    type: Optional[str]
    version: Optional[str]


@dataclass
class ObsHeader:
    """Header of an observation file, every record read by column.

    ``obs_types``: ``{system: [codes in header order]}``. RINEX 3/4: from ``SYS / # / OBS TYPES`` (a RINEX 3.02
    BeiDou band-1 code is listed as band 2, with an issue). RINEX 2: the single ``# / TYPES OF OBSERV`` list,
    stored under the file's own system letter ('G' when blank), and for a mixed file ('M') under each RINEX 2.11
    system letter 'G', 'R', 'E' and 'S'.

    ``time_of_first_obs`` / ``time_of_last_obs`` are the header times as written (naive, in ``time_system``, not
    moved to GPST); None when missing or unreadable. ``time_system`` is resolved as described in the module
    docstring ('GPS', 'GAL', 'BDT', 'QZS', 'IRN'; GLO and UTC are refused by :func:`read_obs`).

    ``issues``: readable notes about header defects (a missing, zero, unreadable or implausible APPROX POSITION
    XYZ, a missing DELTA H/E/N, a blank antenna type, a count that does not match the listed types, a header
    INTERVAL more than 1 % off the data interval, ...). ``raw``: ``{label: [payloads]}`` with the line[:60]
    payload of every header line in file order (labels stripped).
    """

    path: str
    version: float                                   # the true RINEX version (2.11, 3.04, 4.02; never CRINEX's)
    file_type: str                                   # 'O'
    system: str                                      # 'G', 'R', 'E', 'C', 'J', 'I', 'S' or 'M' (RINEX 2 blank -> 'G')
    obs_types: dict = field(default_factory=dict)
    marker_name: Optional[str] = None
    marker_number: Optional[str] = None
    receiver: Optional[ReceiverInfo] = None
    antenna_serial: Optional[str] = None             # ANT # / TYPE columns 1-20, None if blank
    antenna_type: Optional[str] = None               # columns 21-36, None if blank
    antenna_radome: Optional[str] = None             # columns 37-40, 'NONE' if blank (None without the record)
    approx_position: Optional[np.ndarray] = None     # ECEF [m], shape (3,); None: missing, zero, junk, off Earth
    antenna_delta_hen: Optional[np.ndarray] = None   # [m] (H, E, N), shape (3,); None when missing or unreadable
    interval_header: Optional[float] = None          # INTERVAL [s]; None for missing, 0 or junk
    time_of_first_obs: Optional[pd.Timestamp] = None
    time_of_last_obs: Optional[pd.Timestamp] = None
    time_system: Optional[str] = None
    leap_seconds: Optional[int] = None
    phase_shift_text: Optional[str] = None           # SYS / PHASE SHIFT payloads joined with ' ' (None if absent)
    scale_factors: dict = field(default_factory=dict)  # {system: {code: factor}} from SYS / SCALE FACTOR
    comments: list = field(default_factory=list)     # COMMENT payloads, stripped, in file order
    issues: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    n_header_lines: int = 0                          # lines up to and including END OF HEADER
    unreadable_types: dict = field(default_factory=dict)  # {system or '*' (RINEX 2): why its type list is unusable}
    #: SYS / PHASE SHIFT records of the header (RINEX 3/4) as :func:`phase_shift_map` gives them:
    #: {(system, code): {'value': cycles, 'sats': list | None, 'by_sat': dict | None}}; empty
    #: without records and for RINEX 2
    phase_shift: dict = field(default_factory=dict)

    def georinex_view(self) -> dict:
        """The header as ``georinex.rinexheader`` gives it, for the callers that used that dict (io.py,
        configuration.py), without georinex.

        Keys: every label of ``raw`` with its line[:60] payloads joined by ``' '`` (georinex's joining), except
        that RINEX 3/4 ``SYS / # / OBS TYPES`` is not a key and RINEX 2 ``# / TYPES OF OBSERV`` is the list of
        codes; plus ``'version'`` (the true version, also for CRINEX), ``'filetype'`` ('O'), ``'rinextype'``
        ('obs'), ``'systems'`` (the header system character, 'M' for mixed), ``'fields'`` (RINEX 3/4: ``{system:
        [codes]}`` in header order; RINEX 2: the list of codes), ``'t0'`` and ``'t1'`` (datetime, only when
        TIME OF FIRST/LAST OBS is readable), ``'interval'`` (float, only when INTERVAL parses) and
        ``'position'`` (list of 3 floats, only when ``approx_position`` is plausible). Unlike georinex, no
        'position_geodetic' and no 'fields_ind'/'Fmax'.
        """
        out: dict = {}
        rinex2 = self.version < 3
        for label, payloads in self.raw.items():
            if label == "SYS / # / OBS TYPES" and not rinex2:
                continue
            if label == "# / TYPES OF OBSERV" and rinex2:
                continue
            out[label] = " ".join(payloads)
        out["version"] = float(self.version)
        out["filetype"] = self.file_type
        out["rinextype"] = "obs"
        out["systems"] = self.system
        if rinex2:
            codes = _rinex2_type_list(self)
            out["# / TYPES OF OBSERV"] = list(codes)
            out["fields"] = list(codes)
        else:
            out["fields"] = {s: list(c) for s, c in self.obs_types.items()}
        for key, t in (("t0", self.time_of_first_obs), ("t1", self.time_of_last_obs)):
            if t is not None:
                out[key] = t.to_pydatetime(warn=False)
        if self.raw.get("INTERVAL"):
            value = _to_float(self.raw["INTERVAL"][0][:10])
            if value is not None:
                out["interval"] = value
        if self.approx_position is not None:
            out["position"] = [float(v) for v in self.approx_position]
        return out


@dataclass
class ObsRead:
    """Result of :func:`read_obs`.

    ``frames``: ``{system: DataFrame or None}`` for every requested system (every system with observation types
    in the header when ``systems`` is None); None when the file has no row of that system inside ``tlim``.
    ``lli``: ``{system: DataFrame}`` (empty unless ``lli=True``). ``interval_s``: the median spacing of the data
    epochs read; with fewer than 2 epochs the header INTERVAL, then the file-name token with its units (C = 100
    Hz, Z = Hz, S, M, H, D, U); None if none is usable; ``interval_source`` says which ('data', 'header',
    'name'). ``t_first`` / ``t_last``: first and last data epoch read (GPST), None without data.
    ``epoch_flags``: counts of the epoch records of the whole file by flag (0-6), regardless of ``tlim`` and
    ``systems``. ``notes``: the texts of the warnings emitted and other readable remarks about the data.
    ``ssi``: ``{system: DataFrame}`` (empty unless ``ssi=True``).
    ``phase_shift_events``: the ``SYS / PHASE SHIFT`` records of the event-flag 2-5 blocks inside the data
    (RINEX 3/4), in file order, as ``(epoch, system, code, sats, value)`` (since v1.0.0; see the module
    docstring).

    The data epochs of ``interval_s``, ``t_first`` and ``t_last`` are the flag-0/1 epochs inside ``tlim`` of
    the whole file (every system), so they do not depend on ``systems``.
    """

    header: ObsHeader
    frames: dict = field(default_factory=dict)
    lli: dict = field(default_factory=dict)
    interval_s: Optional[float] = None
    interval_source: Optional[str] = None
    t_first: Optional[pd.Timestamp] = None
    t_last: Optional[pd.Timestamp] = None
    epoch_flags: Counter = field(default_factory=Counter)
    notes: list = field(default_factory=list)
    ssi: dict = field(default_factory=dict)
    phase_shift_events: list = field(default_factory=list)


# ============================================================================ small helpers
def _to_float(text) -> Optional[float]:
    """Float of a fixed-width field (D exponents allowed); None when blank or unreadable."""
    if isinstance(text, bytes):
        text = text.decode("ascii", errors="replace")
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _to_int(text) -> Optional[int]:
    if isinstance(text, bytes):
        text = text.decode("ascii", errors="replace")
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _seconds_ns(text: str) -> int:
    """Seconds field (F11.7 / F13.7) in integer nanoseconds, exact to the written digits; ValueError if junk.

    Blanks inside the field are ignored, as a Fortran F read ignores them by default (``BLANK='NULL'``; RINEX
    defines its records in Fortran formats): ``' 0 .0000000'`` is 0 s. georinex and RTKLIB read such a field the
    same way."""
    text = text.replace(" ", "")
    if not text:
        raise ValueError("blank seconds")
    neg = text.startswith("-")
    ip, _, fp = text.lstrip("+-").partition(".")
    if not (ip or fp) or (ip and not ip.isdigit()) or (fp and not fp.isdigit()):
        raise ValueError(f"unreadable seconds {text!r}")
    ns = int(ip or 0) * _NS + int((fp + "000000000")[:9])
    return -ns if neg else ns


_DAY_NS: dict = {}


def _day_ns(y: int, mo: int, d: int) -> int:
    """Nanoseconds from 1970-01-01 to the date (ValueError for an impossible date)."""
    key = (y, mo, d)
    value = _DAY_NS.get(key)
    if value is None:
        value = int((np.datetime64(datetime(y, mo, d), "ns") - _EPOCH0).astype(np.int64))
        _DAY_NS[key] = value
    return value


def _header_time(payload: str) -> Optional[pd.Timestamp]:
    """TIME OF FIRST/LAST OBS: 5I6, F13.7 (exact to the written digits); None when unreadable."""
    try:
        y, mo, d, h, mi = (int(payload[i:i + 6]) for i in range(0, 30, 6))
        ns = _day_ns(y, mo, d) + (h * 3600 + mi * 60) * _NS + _seconds_ns(payload[30:43])
    except (ValueError, TypeError, OverflowError):
        return None
    return pd.Timestamp(ns, unit="ns")


def _rinex2_type_list(hdr: ObsHeader) -> list:
    """The one RINEX 2 type list of the header (the same list is stored under every system letter)."""
    for codes in hdr.obs_types.values():
        return list(codes)
    return []


def _bds_b1_as_b2(codes: list) -> list:
    """RINEX 3.00-3.02 BeiDou codes with band 1 read as band 2 (B1I; RINEX 3.05 Table 19 note)."""
    return [c[0] + "2" + c[2:] if len(c) == 3 and c[1] == "1" else c for c in codes]


def _norm_systems(systems) -> Optional[list]:
    if systems is None:
        return None
    letters = list(systems) if not isinstance(systems, str) else list(systems)
    out = []
    for s in letters:
        s = str(s).strip().upper()
        if len(s) != 1 or s not in _SYSTEM_LETTERS:
            raise ValueError(f"systems must be RINEX system letters ({', '.join(_SYSTEM_LETTERS)}); got {systems!r}")
        if s not in out:
            out.append(s)
    return out


def _tlim_ns(tlim) -> Optional[tuple]:
    """``tlim`` as (start, end) integer nanoseconds (naive GPST wall time); None bound -> open."""
    if tlim is None:
        return None
    try:
        t0, t1 = tlim
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tlim must be (start, end); got {tlim!r}") from exc
    lo = -(2 ** 62) if t0 is None else int(pd.Timestamp(t0).value)
    hi = 2 ** 62 if t1 is None else int(pd.Timestamp(t1).value)
    return lo, hi


def _fmt_ns(ns: int) -> str:
    return str(pd.Timestamp(int(ns), unit="ns"))


def _interval_from_name(path) -> Optional[float]:
    """Sampling interval [s] from the data frequency token of a RINEX 3 long file name (``_30S_``, ``_01Z_``,
    ``_15M_`` ...; C = 100 Hz, Z = Hz, S, M, H, D; U = unspecified); None when absent or unspecified."""
    m = _NAME_TOKEN.search(os.path.basename(os.fspath(path)).upper())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    if n <= 0:
        return None
    if unit in _NAME_RATE_HZ:
        return 1.0 / (n * _NAME_RATE_HZ[unit])
    if unit in _NAME_STEP_S:
        return n * _NAME_STEP_S[unit]
    return None


# ============================================================================ bytes
_WHAT = "RINEX observation file"


def _file_bytes(path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _is_crinex(raw: bytes) -> bool:
    return b"COMPACT RINEX" in raw.lstrip()[:80]


def _stream_prefix(raw: bytes, fmt: str) -> tuple:
    """(readable start, truncated) of gzip or bzip2 data, member after member; a damaged member (CRC, invalid
    block) raises as in :mod:`gzip` / :mod:`bz2`."""
    out = []
    data = raw
    while data:
        d = zlib.decompressobj(31) if fmt == "gzip" else bz2.BZ2Decompressor()
        out.append(d.decompress(data))
        if not d.eof:
            return b"".join(out), True
        data = d.unused_data
    return b"".join(out), False


def _outer(raw: bytes, path, notes: list) -> bytes:
    """``raw`` with gzip, Unix compress, bzip2 or zip undone in memory (the decoders of
    :mod:`gnx_py.compression`, applied to the bytes already read, so the file is read once).

    A gzip or bzip2 stream that ends before its end marker (a cut download) keeps its readable start; a note
    goes to ``notes`` and the record scan drops the cut epoch. Stray bytes after the gzip data are ignored with a
    note (as ``gunzip`` ignores them). Any other damage raises RinexObsError."""
    fmt = compression.detect_format(raw[:4])
    if fmt == "plain":
        return raw
    try:
        data, stray = compression.decompress(raw, fmt)
        notes.extend(stray)
        return data
    except (EOFError, OSError, ValueError, zlib.error, zipfile.BadZipFile) as exc:
        err = compression.CompressedFileError(path, fmt, _WHAT, f"{type(exc).__name__}: {exc}")
        if fmt in ("gzip", "bzip2") and "ended before the end-of-stream marker" in str(exc):
            try:
                data, truncated = _stream_prefix(raw, fmt)
            except (EOFError, OSError, ValueError, zlib.error):
                truncated = False
            if truncated:
                notes.append(f"the {fmt} data end before their end-of-stream marker (a cut file): the "
                             f"{len(data)} bytes decompressed before the cut are read")
                return data
        raise RinexObsError(str(err)) from exc


def _crx2rnx_exe() -> Optional[str]:
    """The ``crx2rnx`` program bundled with hatanaka, or None."""
    try:
        from importlib.resources import files
        exe = files("hatanaka.bin").joinpath("crx2rnx" + (".exe" if os.name == "nt" else ""))
        return str(exe) if exe.is_file() else None
    except (ImportError, ModuleNotFoundError, OSError, TypeError):
        return None


class _Crx2rnxError(Exception):
    """crx2rnx stopped with an error (exit status other than 0 and 2); the message is hatanaka's text."""


def _crx2rnx_run(exe: str, raw: bytes) -> tuple:
    """(stdout, stderr) of ``exe -`` on ``raw`` through :func:`gnx_py._proc.run` (posix_spawn, no fork), with the
    exit-status handling of ``hatanaka.crx2rnx`` (hatanaka 2.8, ``hatanaka._check``): status 0 or 2 is a success
    and a non-empty stderr (or status 2 without one) becomes a UserWarning ``"crx2rnx: <text>"``, any other status
    raises :class:`_Crx2rnxError` with hatanaka's message text."""
    proc = _proc.run([exe, "-"], input=raw)
    stderr = proc.stderr.decode("ascii", errors="backslashreplace").strip()
    if proc.returncode not in (0, 2):
        stderr = re.sub("^ +", "", stderr, flags=re.M)
        stderr = re.sub("\n(?!WARNING|ERROR)", " ", stderr)
        stderr = re.sub("^ERROR *: *", "", stderr, flags=re.M)
        raise _Crx2rnxError(stderr)
    if proc.returncode == 2 and not stderr:
        warnings.warn("crx2rnx: exited with an unspecified warning", UserWarning, stacklevel=3)
    if stderr:
        stderr = re.sub("^ +", "", stderr.strip(), flags=re.M)
        stderr = stderr.replace("\n", " ")
        stderr = re.sub("^ *WARNING *:? *", "\n", stderr, flags=re.M)
        stderr = re.sub("^\n", "", stderr)
        warnings.warn(f"crx2rnx: {stderr}", UserWarning, stacklevel=3)
    return proc.stdout, proc.stderr


def _crx2rnx(raw: bytes, path, notes: list) -> bytes:
    """Hatanaka (CRINEX 1/3) decompression in memory through the bundled RNXCMP ``crx2rnx`` (stdout mode, so
    the end of the file is never lost), started by :func:`_crx2rnx_run` (posix_spawn, no fork); crx2rnx's own
    warnings are passed on, not filtered.

    A CRINEX text cut short (a cut download): crx2rnx stops with "truncated" at the end of its input (also for a
    last line without a newline, so a number cut in half is never decoded; measured on 300 random cuts of a real
    file and on cuts inside the last number); the RINEX text it wrote before stopping, a prefix of the full
    output, is then read and the record scan drops the cut epoch. A note goes to ``notes``. Any other crx2rnx
    error, or a "truncated" stop before the end of the input, raises RinexObsError."""
    exe = _crx2rnx_exe()
    if exe is None:
        raise RinexObsError(f"{path}: Hatanaka-compressed (CRINEX) observations need the 'hatanaka' package "
                            "(pip install hatanaka) and its bundled crx2rnx program")
    try:
        out, _ = _crx2rnx_run(exe, raw)
        return out
    except _Crx2rnxError as exc:
        out = _crx2rnx_cut(raw, str(exc))
        if out is None:
            raise RinexObsError(f"{path}: CRINEX (Hatanaka) data cannot be decompressed ({exc})") from exc
        notes.append(f"the CRINEX data end in the middle of an epoch (a cut file; crx2rnx: "
                     f"{' '.join(str(exc).split())[:160]}): the RINEX text decompressed before the cut is read")
        return out


def _crx2rnx_cut(raw: bytes, message: str) -> Optional[bytes]:
    """RINEX text crx2rnx writes before it stops at the end of a cut CRINEX input ("truncated in the middle",
    interrupted at the last line of ``raw`` or after it); None for any other error. The bundled program is run
    again through :func:`gnx_py._proc.run` to collect that text (cut files only; the first run's output is not
    kept, as with hatanaka.crx2rnx before v1.0.0)."""
    m = re.search(r"truncated.*?interrupted after reading the line\s+(\d+)", message, re.S)
    if not m or int(m.group(1)) < raw.count(b"\n"):
        return None
    exe = _crx2rnx_exe()
    if exe is None:
        return None
    proc = _proc.run([exe, "-"], input=raw)
    if b"truncated" not in proc.stderr:
        return None
    return proc.stdout


#: Label of the first line of a CRINEX file (columns 61-80).
_CRINEX_LABEL = b"CRINEX VERS   / TYPE"


def _crinex_parts(raw: bytes) -> list:
    """``raw`` (CRINEX) split before every CRINEX VERS / TYPE line after the first: CRINEX files joined with ``cat``.
    crx2rnx skips a second CRINEX header ("skip until an initialized epoch is found") and decodes the records after
    it with the first header's types (measured: C1C and L1C swapped for a reordered type list, no error), so each
    part is decompressed on its own and its RINEX header then reaches :func:`_embedded_header`. One part for an
    ordinary file (one ``bytes.find`` over the data)."""
    k = raw.find(_CRINEX_LABEL, raw.find(b"\n") + 1)
    if k < 0:
        return [raw]
    starts = [0]
    while k >= 0:
        line0 = raw.rfind(b"\n", 0, k) + 1
        if k - line0 == 60:
            starts.append(line0)
        k = raw.find(_CRINEX_LABEL, k + len(_CRINEX_LABEL))
    return [raw[a:b] for a, b in zip(starts, starts[1:] + [len(raw)])]


def _plain(raw: bytes, path, notes: list) -> bytes:
    """Plain RINEX bytes of a file read as ``raw``; a UTF-8 byte order mark dropped and line ends normalised to
    ``\\n`` (:func:`compression.normalise_line_ends`: CR LF, CR CR LF and bare CR). Notes about a cut compressed
    stream go to ``notes``. Joined CRINEX files are decompressed part by part (:func:`_crinex_parts`)."""
    raw = compression.strip_bom(_outer(raw, path, notes))
    if _is_crinex(raw):
        raw = compression.normalise_line_ends(raw)    # crx2rnx reads CR LF but not CR CR LF or bare CR
        parts = _crinex_parts(raw)
        if len(parts) == 1:
            raw = _crx2rnx(raw, path, notes)
        else:
            out = [_crx2rnx(part, path, notes) for part in parts]
            raw = b"".join(o if o.endswith(b"\n") else o + b"\n" for o in out)
    return compression.normalise_line_ends(raw)


# ============================================================================ header
def _header_lines(buf: bytes, path) -> tuple:
    """(lines of the header from RINEX VERSION / TYPE to END OF HEADER exclusive, offset of the first data byte).

    CRINEX lines before the RINEX header are skipped (the RINEX header is kept verbatim in CRINEX)."""
    n = len(buf)
    if not buf.strip():
        raise RinexObsError(f"{path}: empty file")
    pos = len(compression.BOM) if buf.startswith(compression.BOM) else 0
    lines = []
    started = False
    while pos < n:
        j = buf.find(b"\n", pos)
        j = n if j < 0 else j
        ln = buf[pos:j].decode("ascii", errors="replace").rstrip("\r")
        pos = j + 1
        if not started:
            if not ln.strip():
                continue
            if "CRINEX" in ln[60:80] or "COMPACT RINEX" in ln[:80]:
                continue
            if "RINEX VERSION / TYPE" not in ln[60:80]:
                raise RinexObsError(f"{path}: not a RINEX file (first line is not 'RINEX VERSION / TYPE': "
                                    f"{ln[:80]!r})")
            started = True
        if "END OF HEADER" in ln and "COMMENT" not in ln[60:80]:
            return lines, pos
        lines.append(ln)
    raise RinexObsError(f"{path}: no END OF HEADER")


def _parse_header(lines: list, path) -> tuple:
    """(:class:`ObsHeader`, layout) from the header lines. ``layout``: ``{system: list of the codes of the
    record fields in file order}`` (RINEX 2: ``{'*': codes}``), the field order of the data records."""
    first = lines[0]
    version = _to_float(first[:9])
    if version is None:
        raise RinexObsError(f"{path}: unreadable RINEX version {first[:9]!r}")
    file_type = first[20:21].upper()
    if file_type != "O":
        kind = {"N": "navigation", "G": "GLONASS navigation", "H": "SBAS navigation", "C": "clock",
                "M": "meteorological"}.get(file_type, f"type {file_type!r}")
        raise RinexObsError(f"{path}: not an observation file (RINEX {version:.2f} {kind} file)")
    if not 2.0 <= version < 5.0:
        raise RinexObsError(f"{path}: RINEX version {version:.2f} is not supported (2.xx, 3.xx and 4.xx are)")
    rinex2 = version < 3
    system = first[40:41].strip().upper() or "G"
    hdr = ObsHeader(path=os.fspath(path), version=float(version), file_type=file_type, system=system,
                    n_header_lines=len(lines) + 1)
    raw: dict = {}
    for ln in lines:
        label = ln[60:80].strip()
        raw.setdefault(label, []).append(ln[:60])
    hdr.raw = raw
    layout = _parse_obs_types(hdr, rinex2)
    _parse_header_records(hdr)
    return hdr, layout


def _parse_obs_types(hdr: ObsHeader, rinex2: bool) -> dict:
    """Observation types of the header (``hdr.obs_types``) and the record layout (returned).

    A list with fewer codes than its count declares (a continuation line missing) cannot be matched to the record
    fields: every field after the gap would be read under another type. Such a system is left out of the layout
    and of ``obs_types`` and named in ``hdr.unreadable_types`` (RINEX 2: key '*', the one list of the file;
    :func:`read_obs` then raises). A list with more codes than declared is read as listed. RINEX 3/4: a line that
    repeats the system letter while the list is still short of its count is a continuation line; a second list
    of one system that differs from the first makes that system unreadable as well."""
    raw = hdr.raw
    if rinex2:
        codes: list = []
        declared = None
        for payload in raw.get("# / TYPES OF OBSERV", []):
            if declared is None:
                declared = _to_int(payload[:6])
            codes += payload[6:60].split()
        if declared is not None and declared != len(codes):
            hdr.issues.append(f"# / TYPES OF OBSERV: {declared} types declared, {len(codes)} listed")
            if declared > len(codes):
                hdr.unreadable_types["*"] = f"# / TYPES OF OBSERV: {declared} types declared, {len(codes)} listed"
        if not codes:
            hdr.issues.append("# / TYPES OF OBSERV missing: no observation can be read")
        if len(set(codes)) < len(codes):
            hdr.issues.append("# / TYPES OF OBSERV: codes listed twice are read once: "
                              + ", ".join(sorted({c for c in codes if codes.count(c) > 1})))
        letters = ("G", "R", "E", "S") if hdr.system == "M" else (hdr.system,)
        hdr.obs_types = {s: list(dict.fromkeys(codes)) for s in letters} if codes else {}
        return {"*": codes}
    lists: list = []                                  # [system, declared, codes] per list, header order
    for payload in raw.get("SYS / # / OBS TYPES", []):
        head = payload[:1].strip().upper()
        if head:
            n_decl = _to_int(payload[3:6])
            last = lists[-1] if lists else None
            if not (last is not None and last[0] == head and last[1] is not None and len(last[2]) < last[1]
                    and n_decl in (None, last[1])):
                lists.append([head, n_decl, []])
        if lists:
            lists[-1][2] += payload[7:60].split()
    layout: dict = {}
    declared: dict = {}
    for s, n_decl, codes in lists:
        if s in layout:
            if (layout[s], declared[s]) != (codes, n_decl):
                hdr.unreadable_types[s] = (f"SYS / # / OBS TYPES {s}: two different type lists "
                                           f"({' '.join(layout[s])} / {' '.join(codes)})")
            else:
                hdr.issues.append(f"SYS / # / OBS TYPES {s}: the type list is written twice")
            continue
        layout[s], declared[s] = codes, n_decl
        if n_decl is not None and n_decl != len(codes):
            hdr.issues.append(f"SYS / # / OBS TYPES {s}: {n_decl} types declared, {len(codes)} listed")
            if n_decl > len(codes):
                hdr.unreadable_types[s] = f"SYS / # / OBS TYPES {s}: {n_decl} types declared, {len(codes)} listed"
    for s in hdr.unreadable_types:
        layout.pop(s, None)
        hdr.issues.append(f"{hdr.unreadable_types[s]}: the record fields cannot be matched to the types; the {s} "
                          "observations are not read")
    if not layout and not hdr.unreadable_types:
        hdr.issues.append("SYS / # / OBS TYPES missing: no observation can be read")
    obs_types = {}
    for s, codes in layout.items():
        out = list(codes)
        if s == "C" and hdr.version < 3.025:
            mapped = _bds_b1_as_b2(codes)
            if mapped != codes:
                hdr.issues.append(f"RINEX {hdr.version:.2f} BeiDou band-1 codes read as band 2 (B1I): "
                                  + ", ".join(f"{a}->{b}" for a, b in zip(codes, mapped) if a != b))
            out = mapped
        if len(set(out)) < len(out):
            hdr.issues.append(f"SYS / # / OBS TYPES {s}: codes listed twice are read once: "
                              + ", ".join(sorted({c for c in out if out.count(c) > 1})))
        obs_types[s] = list(dict.fromkeys(out))
    hdr.obs_types = obs_types
    return layout


def _parse_header_records(hdr: ObsHeader) -> None:
    raw = hdr.raw

    def one(label):
        v = raw.get(label)
        return v[0].ljust(60) if v else None

    rec = one("MARKER NAME")
    if rec is not None:
        hdr.marker_name = rec[:60].strip() or None
    rec = one("MARKER NUMBER")
    if rec is not None:
        hdr.marker_number = rec[:20].strip() or None
    rec = one("REC # / TYPE / VERS")
    if rec is not None:
        hdr.receiver = ReceiverInfo(rec[0:20].strip() or None, rec[20:40].strip() or None,
                                    rec[40:60].strip() or None)
    # ANT # / TYPE: serial 1-20, type 21-36, radome 37-40 (IGS convention; antex.receiver_antenna_from_header)
    rec = one("ANT # / TYPE")
    if rec is None:
        hdr.issues.append("ANT # / TYPE missing")
    else:
        hdr.antenna_serial = rec[0:20].strip() or None
        hdr.antenna_type = rec[20:36].strip() or None
        hdr.antenna_radome = rec[36:40].strip() or "NONE"
        if hdr.antenna_type is None:
            hdr.issues.append("ANT # / TYPE: antenna type (columns 21-36) is blank; fields found: serial "
                              f"{rec[0:20].strip()!r}, type '', radome {rec[36:40].strip()!r}")
    # APPROX POSITION XYZ: 3F14.4
    rec = one("APPROX POSITION XYZ")
    if rec is None:
        hdr.issues.append("APPROX POSITION XYZ missing")
    else:
        xyz, split = _xyz_record(rec)
        if split:
            hdr.issues.append(f"APPROX POSITION XYZ is not in 3F14.4 columns, read by splitting: {rec.rstrip()!r}")
        if xyz is None:
            hdr.issues.append(f"APPROX POSITION XYZ unreadable: {rec.rstrip()!r}")
        elif not any(xyz):
            hdr.issues.append("APPROX POSITION XYZ is (0, 0, 0)")
        else:
            r = math.sqrt(sum(v * v for v in xyz))
            if APPROX_XYZ_MIN_M <= r <= APPROX_XYZ_MAX_M:
                hdr.approx_position = np.array(xyz, dtype=float)
            else:
                hdr.issues.append(f"APPROX POSITION XYZ ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) is off the "
                                  f"Earth's surface: |r| = {r:.0f} m outside {APPROX_XYZ_MIN_M:.0f}-"
                                  f"{APPROX_XYZ_MAX_M:.0f} m")
    # ANTENNA: DELTA H/E/N: 3F14.4
    rec = one("ANTENNA: DELTA H/E/N")
    if rec is None:
        hdr.issues.append("ANTENNA: DELTA H/E/N missing")
    else:
        hen, split = _xyz_record(rec)
        if split:
            hdr.issues.append(f"ANTENNA: DELTA H/E/N is not in 3F14.4 columns, read by splitting: {rec.rstrip()!r}")
        if hen is None:
            hdr.issues.append(f"ANTENNA: DELTA H/E/N unreadable: {rec.rstrip()!r}")
        else:
            hdr.antenna_delta_hen = np.array(hen, dtype=float)
    # INTERVAL: F10.3; missing, 0 or junk -> None
    rec = one("INTERVAL")
    if rec is not None:
        value = _to_float(rec[:10])
        if value is None:
            hdr.issues.append(f"INTERVAL unreadable: {rec[:10].strip()!r}")
        elif value <= 0:
            hdr.issues.append(f"INTERVAL {value:g} is not a sampling interval")
        else:
            hdr.interval_header = value
    # TIME OF FIRST / LAST OBS: 5I6, F13.7, 5X, A3
    rec = one("TIME OF FIRST OBS")
    tsys = None
    if rec is None:
        hdr.issues.append("TIME OF FIRST OBS missing")
    else:
        hdr.time_of_first_obs = _header_time(rec)
        if hdr.time_of_first_obs is None:
            hdr.issues.append(f"TIME OF FIRST OBS unreadable: {rec.rstrip()!r}")
        tsys = _time_system_field(rec)
    rec = one("TIME OF LAST OBS")
    if rec is not None:
        hdr.time_of_last_obs = _header_time(rec)
        if tsys is None:
            tsys = _time_system_field(rec)
    if tsys is None:
        if hdr.system == "M":
            tsys = "GPS"
            hdr.issues.append("TIME OF FIRST OBS has no time system in a mixed file: GPS time assumed")
        else:
            tsys = _SYSTEM_TIME.get(hdr.system, "GPS")
    hdr.time_system = tsys
    rec = one("LEAP SECONDS")
    if rec is not None:
        hdr.leap_seconds = _to_int(rec[:6])
    if raw.get("SYS / PHASE SHIFT"):
        hdr.phase_shift_text = " ".join(p.rstrip() for p in raw["SYS / PHASE SHIFT"])
        if hdr.version >= 3:                         # RINEX 2 has no such record
            records, issues = parse_phase_shift_records(raw["SYS / PHASE SHIFT"])
            hdr.phase_shift, more = phase_shift_map(records)
            hdr.issues += issues + more
    hdr.comments = [p.strip() for p in raw.get("COMMENT", [])]
    hdr.scale_factors = _scale_factors(hdr)


_PS_SAT = re.compile(r"([A-Z])\s*(\d{1,2})")


def parse_phase_shift_records(payloads: Iterable[str]) -> tuple:
    """(records, issues) of RINEX 3/4 ``SYS / PHASE SHIFT`` lines (the line[:60] payloads, in file order).

    RINEX 3.05 / 4.02 Table A2: system A1 (column 1), carrier-phase code 1X,A3 (columns 3-5), correction applied
    [cycles] 1X,F8.5 (columns 7-14), number of satellites 2X,I2.2 (columns 17-18; 0 or blank: every satellite of
    the system), list of satellites 10(1X,A3) from column 19, continued on lines that start with 18 blanks.
    Each record is ``(system, code, value [cycles], sats)``: ``sats`` the listed satellites ('G01'; a blank in the
    number is read as a Fortran I2 reads it, 'G 1' is G01) in listing order, None for every satellite of the
    system. A blank correction is 0.0 (no shift; the value ``io.parse_phase_shift`` gives). A line with only the
    system letter (RINEX 3.01: no correction for that system) gives no record. ``issues``: readable notes about
    defects (a list shorter or longer than its count, a satellite of another system or unreadable, a continuation
    line without a record); the records are read as far as they go."""
    records: list = []
    issues: list = []
    current = None                                   # [system, code, value, sats or None, declared]

    def close():
        if current is None:
            return
        system, code, value, sats, declared = current
        if declared and len(sats) != declared:
            issues.append(f"SYS / PHASE SHIFT {system} {code}: {declared} satellites declared, {len(sats)} listed "
                          f"(the listed ones are used)")
        records.append((system, code, value, sats if declared else None))

    def add_sats(text: str, system: str, code: str, sats: list) -> None:
        text = text.ljust(42)
        for k in range(10):
            tok = text[4 * k + 1:4 * k + 4]
            if not tok.strip():
                continue
            m = _PS_SAT.fullmatch(tok.strip())
            if m is None or int(m.group(2)) == 0:
                issues.append(f"SYS / PHASE SHIFT {system} {code}: unreadable satellite {tok!r} skipped")
            elif m.group(1) != system:
                issues.append(f"SYS / PHASE SHIFT {system} {code}: satellite {tok.strip()} of another system "
                              "skipped")
            else:
                sats.append(f"{system}{int(m.group(2)):02d}")

    for payload in payloads:
        line = str(payload)[:60].ljust(60)
        head = line[:1].strip().upper()
        if head:
            close()
            current = None
            code = line[2:5].strip()
            if not code:
                continue                             # system letter only (RINEX 3.01): no record
            value = _to_float(line[5:14])
            if value is None:
                if line[5:14].strip():
                    issues.append(f"SYS / PHASE SHIFT {head} {code}: unreadable correction "
                                  f"{line[5:14].strip()!r} read as 0.0")
                value = 0.0
            declared = _to_int(line[14:18]) or 0
            sats: list = []
            if declared:
                add_sats(line[18:60], head, code, sats)
            current = [head, code, float(value), sats, declared]
        elif line.strip():
            if current is None or not current[4]:
                issues.append(f"SYS / PHASE SHIFT continuation line without a satellite list before it skipped: "
                              f"{line.rstrip()!r}")
                continue
            add_sats(line[18:60], current[0], current[1], current[3])
    close()
    return records, issues


def phase_shift_map(records: Iterable[tuple]) -> tuple:
    """(map, issues): the ``SYS / PHASE SHIFT`` records of one header as ``{(system, code): {'value': float
    [cycles], 'sats': list[str] | None, 'by_sat': dict[str, float] | None}}``.

    ``by_sat`` maps every listed satellite to its correction; None when no record of the key lists satellites.
    ``value``/``sats`` give the contract view: the records of one key with one correction give that value and the
    union of their lists (None when one of them has no list: every satellite of the system). Records of one key
    with different corrections (Javad receivers list per-satellite values, e.g. BeiDou L2I 0.0 for C02 C05 C60 and
    0.5 for 29 other satellites) give the value of most listed satellites (a tie: the earlier record), ``sats`` =
    the satellites with that value, and ``by_sat`` for the others. A key with a record without a list takes that
    record's value (``sats`` None) for the satellites no list names. A satellite listed twice with different
    corrections keeps the first one (an issue names it). The per-satellite rule is therefore ``by_sat[sv]`` when
    listed, else ``value`` when ``sats`` is None, else no shift; ``value``/``sats`` alone are exact whenever the
    records of a key agree (every corpus file but the Javad ones)."""
    grouped: dict = {}
    for system, code, value, sats in records:
        grouped.setdefault((system, code), []).append((float(value), None if sats is None else list(sats)))
    out: dict = {}
    issues: list = []
    for key, recs in grouped.items():
        by_sat: dict = {}
        conflicts: list = []
        default = None
        for value, sats in recs:
            if sats is None:
                if default is None:
                    default = value
                elif default != value:
                    conflicts.append(f"every satellite {default:g} / {value:g}")
                continue
            for sv in sats:
                if sv in by_sat and by_sat[sv] != value:
                    conflicts.append(f"{sv} {by_sat[sv]:g} / {value:g}")
                by_sat.setdefault(sv, value)
        if conflicts:
            issues.append(f"SYS / PHASE SHIFT {key[0]} {key[1]}: conflicting corrections, the first one is used: "
                          + ", ".join(conflicts[:10]) + (f" and {len(conflicts) - 10} more" if len(conflicts) > 10
                                                         else ""))
        if default is not None:
            value, sats = default, None
        else:
            counts = Counter(by_sat.values())
            value = max((v for v, _ in recs), key=lambda v: counts.get(v, 0))   # ties: the earlier record
            sats = [sv for sv, v in by_sat.items() if v == value]
        out[key] = {"value": value, "sats": sats, "by_sat": by_sat or None}
    return out, issues


def lli_half_cycle(lli: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame (same index and columns as the LLI frame ``lli`` of :func:`read_obs` / :func:`lli_for`):
    True where the LLI of that phase has bit 1 (:data:`LLI_HALF_CYCLE`, half-cycle ambiguity possible)
    set, False where it is clear, blank or the phase is missing."""
    values = lli.to_numpy(dtype=float, copy=False)
    known = np.isfinite(values)
    flags = np.zeros(values.shape, dtype=bool)
    flags[known] = (values[known].astype(np.int64) & LLI_HALF_CYCLE) != 0
    return pd.DataFrame(flags, index=lli.index, columns=lli.columns)



# ---------------------------------------------------------------------------- phase-shift attrs of the frames
def _read_only(self, *args, **kwargs):
    raise TypeError(f"{type(self).__name__}: the phase-shift attrs of rinex_obs frames are read-only (shared by "
                    "every frame of a file); take a copy with phase_shift_of / phase_shift_events_of")


class _FrozenDict(dict):
    """A read-only dict of the phase-shift ``attrs``. pandas deep-copies ``DataFrame.attrs`` in
    ``NDFrame.__finalize__`` (every slice, ``copy``, ``groupby`` group, ...); an immutable value can be shared, so
    ``__deepcopy__`` returns it and the copy costs nothing. json, pickle and ``==`` see a plain dict."""
    __slots__ = ()
    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _read_only

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


class _FrozenList(list):
    """The list counterpart of :class:`_FrozenDict` (json arrays; ``==`` compares as a list)."""
    __slots__ = ()
    __setitem__ = __delitem__ = __iadd__ = __imul__ = append = clear = extend = insert = pop = remove = \
        reverse = sort = _read_only

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (list(self),)


def _phase_shift_attrs(table: Optional[dict] = None, events: Iterable[tuple] = ()) -> dict:
    """``{'phase_shift': ..., 'phase_shift_events': ...}`` of the frames (JSON-safe, read-only) from the contract
    form: ``table`` as :func:`phase_shift_map` gives it, ``events`` ``(epoch ns / Timestamp / None, system, code,
    sats or None, value)``."""
    stored = _FrozenDict({
        f"{system} {code}": _FrozenDict({
            "value": float(rec["value"]),
            "sats": None if rec.get("sats") is None else _FrozenList(rec["sats"]),
            "by_sat": None if rec.get("by_sat") is None else _FrozenDict(
                (sv, float(v)) for sv, v in rec["by_sat"].items())})
        for (system, code), rec in (table or {}).items()})
    stored_events = _FrozenList(
        _FrozenList([None if t is None else pd.Timestamp(t, unit="ns" if isinstance(t, (int, np.integer)) else None)
                     .isoformat(), system, code,
                     None if sats is None else _FrozenList(sats), float(value)])
        for t, system, code, sats, value in events)
    return {"phase_shift": stored, "phase_shift_events": stored_events}


#: key of a header record in the stored form: system letter, blank, RINEX 3/4 observation code
_PHASE_SHIFT_KEY = re.compile(r"^[A-Z] [A-Z0-9]{2,3}$")


def _attrs_entry(source, name: str):
    """``attrs[name]`` of a frame or an attrs dict, or ``source`` when it is the entry itself. A dict that holds
    neither ``name`` nor header-record keys (tuples or ``"G L1C"``) is an attrs dict without the entry: None (a
    frame with other attrs, e.g. ``{'time_scale': 'GPST'}``, or with the events only)."""
    if isinstance(source, (pd.DataFrame, pd.Series)):
        return source.attrs.get(name)
    if isinstance(source, dict):
        if name in source:                              # an attrs dict
            return source[name]
        if name == "phase_shift_events" or not all(
                (isinstance(k, tuple) and len(k) == 2) or (isinstance(k, str) and _PHASE_SHIFT_KEY.match(k))
                for k in source):
            return None
    return source                                       # the entry itself


def phase_shift_of(source) -> dict:
    """The header ``SYS / PHASE SHIFT`` records of a frame of :func:`read_obs` / ``GNSSDataProcessor2`` in the
    the form of :func:`phase_shift_map`: ``{(system, code): {'value': float [cycles], 'sats': list[str] | None,
    'by_sat': dict[str, float] | None}}``, new mutable objects. ``source``: the frame, its ``attrs`` or
    ``attrs['phase_shift']``, also as a parquet / JSON round trip gives them back (plain dicts and lists). Empty
    when the frame has none (RINEX 2, no record, a frame not from the reader). Per satellite: ``by_sat[sv]`` when
    listed, else ``value`` when ``sats`` is None, else no shift (:func:`phase_shift_map`)."""
    table = _attrs_entry(source, "phase_shift")
    out: dict = {}
    for key, rec in (table or {}).items():
        system, code = key if isinstance(key, tuple) else str(key).split(" ", 1)
        out[(system, code)] = {"value": float(rec["value"]),
                               "sats": None if rec.get("sats") is None else list(rec["sats"]),
                               "by_sat": None if rec.get("by_sat") is None else
                               {sv: float(v) for sv, v in rec["by_sat"].items()}}
    return out


def phase_shift_events_of(source) -> list:
    """The ``SYS / PHASE SHIFT`` records of the event-flag 2-5 blocks of the data: ``[(epoch GPST Timestamp or None, system, code, sats list[str] | None, value [cycles]), ...]`` in
    file order, one tuple per record (not merged: a block may restate a list with another value; the first record
    of a satellite in a block is the writer's first statement). ``source`` as for :func:`phase_shift_of`."""
    events = _attrs_entry(source, "phase_shift_events")
    return [(None if t is None else pd.Timestamp(t), system, code, None if sats is None else list(sats),
             float(value)) for t, system, code, sats, value in (events or ())]


def _time_system_field(payload: str) -> Optional[str]:
    """Time system of TIME OF FIRST/LAST OBS: A3 in columns 49-51; writers that print the seconds one column
    wider move it to 50-52, so the first known token in columns 44-60 is taken. None when blank; an unknown
    token is returned as written (refused by :func:`read_obs`)."""
    tokens = re.findall(r"[A-Z]+", payload[43:60].upper())
    for tok in tokens:
        if tok in _KNOWN_TIME_SYSTEMS:
            return tok
    return tokens[0] if tokens else None


def _xyz_record(rec: str) -> tuple:
    """(three floats or None, read by splitting) of a 3F14.4 record; when the columns do not read (writers that
    use 3F15.4), the payload split on blanks, if it holds exactly three numbers."""
    xyz = [_to_float(rec[i:i + 14]) for i in (0, 14, 28)]
    if all(v is not None for v in xyz):
        return xyz, False
    parts = [_to_float(t) for t in rec[:60].split()]
    if len(parts) == 3 and all(v is not None for v in parts):
        return parts, True
    return None, False


def _scale_factors(hdr: ObsHeader) -> dict:
    """``SYS / SCALE FACTOR`` (RINEX 3/4: A1, 1X, I4, 2X, I2, 12(1X,A3); continuation lines 10X, 12(1X,A3);
    0 or blank types: every type of the system)."""
    out: dict = {}
    cur = None
    for payload in hdr.raw.get("SYS / SCALE FACTOR", []):
        payload = payload.ljust(60)
        head = payload[:1].strip().upper()
        codes = payload[10:58].split()
        if head:
            factor = _to_int(payload[2:6])
            if factor is None or factor <= 0:
                hdr.issues.append(f"SYS / SCALE FACTOR unreadable: {payload.rstrip()!r}")
                cur = None
                continue
            cur = (head, factor)
            if not codes and not _to_int(payload[8:10]):
                codes = list(hdr.obs_types.get(head, []))
        if cur is None:
            continue
        s, factor = cur
        if s == "C" and hdr.version < 3.025:
            codes = _bds_b1_as_b2(codes)
        for c in codes:
            out.setdefault(s, {})[c] = float(factor)
    return out


def read_header(path) -> ObsHeader:
    """Header of the observation file ``path`` (any supported compression; CRINEX gives the true version).

    Raises :class:`RinexObsError` for a file that is not a RINEX observation file or has no END OF HEADER.
    The header of a CRINEX file is read without Hatanaka decompression (CRINEX keeps it verbatim); of a plain
    file only the first 256 kB are read when the header ends there. The file is opened once. A cut gzip or
    bzip2 stream gives the header of its readable start, with a note in ``issues``."""
    with open(path, "rb") as fh:
        head = fh.read(1 << 18)
        lines = None
        if compression.detect_format(head[:4]) == "plain" and b"END OF HEADER" in head:
            try:
                lines, _ = _header_lines(_bare_cr(head), path)
            except RinexObsError:
                lines = None   # e.g. 'END OF HEADER' only inside a COMMENT of a longer header: read the whole file
        if lines is None:
            raw = head + fh.read()
    notes: list = []
    if lines is None:
        del head
        lines, _ = _header_of(_bare_cr(_outer(raw, path, notes)), path, notes)
    hdr, _ = _parse_header(lines, path)
    hdr.issues += notes
    return hdr


def _header_of(buf: bytes, path, notes: list) -> tuple:
    """:func:`_header_lines`; an error names the cut compressed stream of ``notes`` as well."""
    try:
        return _header_lines(buf, path)
    except RinexObsError as exc:
        if not notes:
            raise
        raise RinexObsError(f"{exc} ({'; '.join(notes)})") from exc


def _bare_cr(buf: bytes) -> bytes:
    """``buf`` with bare CR line ends made LF (CR LF and CR CR LF are stripped line by line by
    :func:`_header_lines`)."""
    if b"\n" not in buf[:8192] and b"\r" in buf[:8192]:
        return buf.replace(b"\r", b"\n")
    return buf


# ============================================================================ data
@dataclass
class _SysData:
    """Parsed records of one system, sorted by (PRN, time), duplicates dropped."""

    codes: list                 # output codes, header order
    fields: np.ndarray          # record field of each output code
    prn: np.ndarray             # int16
    time: np.ndarray            # int64 ns, GPST
    vals: np.ndarray            # float64 [n, len(codes)]
    lli: np.ndarray             # uint8 [n, len(codes)] (0-7; bit 0 forced on phases of flag-1 epochs)
    ssi: np.ndarray             # uint8 [n, len(codes)]


@dataclass
class _Parsed:
    header: ObsHeader
    systems: dict
    epoch_flags: Counter
    interval_s: Optional[float]
    interval_source: Optional[str]
    t_first: Optional[int]
    t_last: Optional[int]
    notes: list
    warn_texts: list
    # SYS / PHASE SHIFT records of the event-flag 2-5 blocks, file order: (epoch ns GPST or None, system, code,
    # sats or None, value [cycles]) (contract C4)
    phase_shift_events: list = field(default_factory=list)


class _Pending:
    """Record lines of one system waiting for conversion, and the converted chunks."""

    __slots__ = ("lines", "eps", "chunks", "nt", "width", "offset")

    def __init__(self, nt: int, width: int, offset: int = 3):
        self.lines: list = []
        self.eps: list = []
        self.chunks: list = []
        self.nt = nt
        self.width = width
        self.offset = offset


def _float_or_nan(x) -> float:
    try:
        return float(x)
    except ValueError:
        return np.nan


def _flush(p: _Pending) -> None:
    """Convert the pending record lines of one system to numbers (one chunk).

    Satellite number (columns 2-3) as a Fortran I2 read gives it, blanks ignored: 'G 7' and 'G7 ' are G07 (RTKLIB
    reads both so too); a blank or zero number is unreadable. A value field that is not a number (Fortran overflow
    asterisks, junk, a NUL byte, which numpy's bytes type would drop and so read a shorter number) is NaN and
    counted, as is an LLI or SSI character that is neither a digit nor blank."""
    lines = p.lines
    if not lines:
        return
    m, w, nt, off = len(lines), p.width, p.nt, p.offset
    rec = np.frombuffer(b"".join([ln.ljust(w)[:w] for ln in lines]), dtype=np.uint8).reshape(m, w)
    eps = np.asarray(p.eps, dtype=np.int64)
    p.lines, p.eps = [], []
    a = rec[:, 1].astype(np.int16) - 48
    b = rec[:, 2].astype(np.int16) - 48
    da, db = (a >= 0) & (a <= 9), (b >= 0) & (b <= 9)
    prn = np.where(da & db, a * 10 + b, np.where(da, a, b)).astype(np.int16)
    ok = (da | (rec[:, 1] == 32)) & (db | (rec[:, 2] == 32)) & (da | db) & (prn > 0)
    body = rec[:, off:off + 16 * nt].reshape(m, nt, 16)
    txt = np.ascontiguousarray(body[:, :, :14]).reshape(m * nt, 14)
    present = (txt != 32).any(axis=1)
    vals = np.full(m * nt, np.nan)
    unreadable = np.zeros(m * nt, dtype=bool)
    if present.any():
        raw = txt[present]
        sel = raw.view("S14").ravel()
        try:
            v = sel.astype(np.float64)
        except ValueError:  # a damaged field: element by element, NaN where unreadable
            v = np.array([_float_or_nan(x) for x in sel], dtype=np.float64)
        bad = ~np.isfinite(v)
        if (raw == 0).any():
            bad |= (raw == 0).any(axis=1)
        v[bad] = np.nan
        vals[present] = v
        unreadable[present] = bad
    vals = vals.reshape(m, nt)
    unreadable = unreadable.reshape(m, nt)
    vals[vals == 0.0] = np.nan
    c = body[:, :, 14]
    lli = np.where((c >= 48) & (c <= 55), c - 48, 0).astype(np.uint8)     # 0-7; blank, 8, 9 and junk: 0
    undefined = int(((c == 56) | (c == 57))[ok].sum())
    flag_junk = ~(((c >= 48) & (c <= 57)) | (c == 32))
    c = body[:, :, 15]
    ssi = np.where((c >= 48) & (c <= 57), c - 48, 0).astype(np.uint8)
    flag_junk |= ~(((c >= 48) & (c <= 57)) | (c == 32))
    unreadable &= ok[:, None]
    n_bad = int(unreadable.sum())
    n_junk = int(flag_junk[ok].sum())
    sample = None
    if n_bad:
        r, k = np.argwhere(unreadable)[0]
        sample = bytes(body[r, k, :14]).decode("ascii", errors="replace")
    elif n_junk:
        r, k = np.argwhere(flag_junk & ok[:, None])[0]
        sample = bytes(body[r, k, :]).decode("ascii", errors="replace")
    p.chunks.append((prn[ok], eps[ok], vals[ok], lli[ok], ssi[ok], int((~ok).sum()), undefined,
                     n_bad + n_junk, sample))


def _complete_last_line(line: bytes, prefix: int, nfields: int) -> bool:
    """A last line without a newline (the last line of the last record the epoch declares) is complete only when it
    reaches the value of its last field (column ``prefix + 16 * (nfields - 1) + 14``). Writers end every line with
    a newline (every observation file of the corpus does) and drop trailing blank fields, so a shorter line may be
    a line cut at a field boundary or between a value and its LLI: it is dropped (with the truncated-epoch warning)
    instead of losing its last values or reading a cut LLI as 0."""
    return len(line.rstrip()) >= prefix + 16 * max(nfields - 1, 0) + 14


class _ScanState:
    """Counters and lists of one scan."""

    def __init__(self):
        self.flags = Counter()
        self.times: list = []            # data epochs inside tlim (GPST ns)
        self.flag1: list = []            # indices (into times) of flag-1 epochs
        self.events: dict = {2: [], 3: []}
        self.bad_lines = 0
        self.first_bad = None
        self.truncated_mid = 0
        self.truncated_last = None       # (epoch text, complete records, declared)
        self.cut_record = False
        self.unknown_sys = Counter()
        self.dropped: dict = {"early": [], "misfit": [], "extra": []}   # RINEX 2 epochs skipped whole:
        #                                  (epoch text, record, declared, lines skipped)
        self.bad_epochs: list = []       # RINEX 2 epoch records with an unreadable satellite list: (line, reason)
        self.overflow = Counter()        # records with values past the fields of the header's types, per system
        self.embedded: list = []         # headers inside the data: (line, site changes or problem, complete)
        # SYS / PHASE SHIFT lines of event-flag 2-5 blocks (RINEX 3/4), file order: [epoch ns (GPST) or None,
        # [line[:60] payloads]]; a block whose epoch fields are blank takes the next data epoch (ps_pending)
        self.phase_shift_blocks: list = []
        self.ps_pending: list = []


def _line_no(buf: bytes, pos: int) -> int:
    """Line number (1-based, from the first line of the decompressed file) of the byte offset ``pos``."""
    return buf.count(b"\n", 0, pos) + 1


def _bad(st: _ScanState, buf: bytes, pos: int) -> None:
    st.bad_lines += 1
    if st.first_bad is None:
        st.first_bad = _line_no(buf, pos)


def _type_change(block: list, layout: dict, rinex2: bool, path, flag: int, buf: bytes, line_pos: int,
                 hdr: Optional[ObsHeader] = None) -> None:
    """Raise when an event-flag-3/4 block changes how the records after it are read: observation types other than
    the header's, or (RINEX 3/4) a SYS / SCALE FACTOR other than the header's (the line number of the message is
    counted only when raising). Other header records of the block (COMMENT, SYS / PHASE SHIFT, PRN / # OF OBS,
    the site records of a new occupation) do not change the reading."""
    label = "# / TYPES OF OBSERV" if rinex2 else "SYS / # / OBS TYPES"
    rows = [b.decode("ascii", errors="replace") for b in block if label.encode() in b[60:80]]
    changed = ""
    if rows:
        if rinex2:
            codes = []
            for r in rows:
                codes += r[6:60].split()
            new = {"*": codes}
        else:
            new, cur = {}, None
            for r in rows:
                if r[:1].strip():
                    cur = r[:1].upper()
                    new[cur] = []
                if cur is not None:
                    new[cur] += r[7:60].split()
        if any(layout.get(s) != c for s, c in new.items()):
            changed = "observation types change inside the data"
            detail = "; ".join(f"{s}: {' '.join(c)}" for s, c in new.items())
    if not changed and not rinex2 and hdr is not None:
        scale = [b.decode("ascii", errors="replace")[:60] for b in block if b"SYS / SCALE FACTOR" in b[60:80]]
        if scale:
            tmp = ObsHeader(path=hdr.path, version=hdr.version, file_type="O", system=hdr.system,
                            obs_types=hdr.obs_types, raw={"SYS / SCALE FACTOR": scale})
            new_f = _scale_factors(tmp)
            for s, factors in new_f.items():
                old_f = hdr.scale_factors.get(s, {})
                if any(factors.get(c, 1.0) != old_f.get(c, 1.0) for c in set(factors) | set(old_f)):
                    changed = "scale factors change inside the data"
                    detail = "; ".join(f"{s}: " + " ".join(f"{c} {f:g}" for c, f in fs.items())
                                       for s, fs in new_f.items())
                    break
    if changed:
        raise RinexObsError(f"{path}: {changed} (event flag {flag} at line {_line_no(buf, line_pos)}): {detail}; "
                            "split the file at this epoch")


#: Label of the SYS / PHASE SHIFT record (columns 61-80), read also inside event-flag 2-5 blocks.
_PS_LABEL = b"SYS / PHASE SHIFT"

#: Label of the first header line (columns 61-80); found in the data section it starts a header of a concatenated
#: file.
_RVT = b"RINEX VERSION / TYPE"
#: An APPROX POSITION XYZ of a header inside the data that moved by more than this [m] describes another site.
_EMBEDDED_MOVE_M = 100.0


def _is_rvt(line: bytes) -> bool:
    """A 'RINEX VERSION / TYPE' line: the label in columns 61-80."""
    return line[60:80].rstrip() == _RVT


def _layout_rules(hdr: ObsHeader) -> tuple:
    """How the records of a header's version are laid out: RINEX 2, RINEX 4 (picoseconds) and the RINEX 3.00-3.02
    BeiDou band-1 rule (when the header lists BeiDou types)."""
    return hdr.version < 3, hdr.version >= 4, hdr.version < 3.025 and "C" in hdr.obs_types


def _tag_system(hdr: ObsHeader) -> str:
    return (hdr.time_system or "GPS").upper()


def _site_changes(a: ObsHeader, b: ObsHeader) -> list:
    """Records of header ``b`` that describe another station or set-up than header ``a``."""
    out = []
    for label, x, y in (("MARKER NAME", a.marker_name, b.marker_name),
                        ("MARKER NUMBER", a.marker_number, b.marker_number),
                        ("REC # / TYPE / VERS", a.receiver, b.receiver),
                        ("ANT # / TYPE", (a.antenna_serial, a.antenna_type, a.antenna_radome),
                         (b.antenna_serial, b.antenna_type, b.antenna_radome))):
        if x != y:
            out.append(label)
    da, db = a.antenna_delta_hen, b.antenna_delta_hen
    if (da is None) != (db is None) or (da is not None and not np.array_equal(da, db)):
        out.append("ANTENNA: DELTA H/E/N")
    pa, pb = a.approx_position, b.approx_position
    if pa is not None and pb is not None:
        move = float(np.linalg.norm(pa - pb))
        if move > _EMBEDDED_MOVE_M:
            out.append(f"APPROX POSITION XYZ ({move:.0f} m away)")
    return out


def _embedded_header(buf: bytes, pos: int, hdr: ObsHeader, layout: dict, path, st: "_ScanState",
                     is_epoch) -> int:
    """A header inside the data section, from its RINEX VERSION / TYPE line at ``pos`` to END OF HEADER (a file
    made by concatenating RINEX files; the format has no such record, RINEX 2.11 §5 and RINEX 3.05 §5 carry header
    records inside the data only in event-flag-3/4 blocks). Returns the offset after it.

    The header is read like the first one. When it changes how the records after it are read (another layout,
    other observation types, other scale factors, another time system), RinexObsError: those records would be
    read with the first header's types ("split the file"). Otherwise it is skipped and noted in ``st.embedded``
    with the records that describe another station or set-up. The block ends at END OF HEADER, or before an
    epoch record when END OF HEADER is missing (an incomplete header: the records it has are compared, and it is
    noted as incomplete); a header cut by the end of the file is only noted."""
    n = len(buf)
    start = pos
    lines = []
    ended = False
    while pos < n:
        j = buf.find(b"\n", pos)
        j = n if j < 0 else j
        raw = buf[pos:j]
        if lines and is_epoch(raw):
            break                                    # END OF HEADER missing: the data go on here
        pos = j + 1
        ln = raw.decode("ascii", errors="replace").rstrip("\r")
        if ln[60:80].strip() == "END OF HEADER":
            ended = True
            break
        lines.append(ln)

    def line_no() -> int:                            # counted only where a message names the line
        return _line_no(buf, start)

    if not ended and pos >= n:
        st.embedded.append((line_no(), ["the file ends inside it"], False))
        return pos
    try:
        second, layout2 = _parse_header(lines, path)
    except RinexObsError as exc:
        raise RinexObsError(f"{path}: a second header starts inside the data at line {line_no()} (a concatenated "
                            f"file?) and cannot be read: {str(exc).replace(f'{path}: ', '', 1)}") from None
    raw2 = second.raw
    diffs = []
    if _layout_rules(second) != _layout_rules(hdr):
        diffs.append(f"RINEX {second.version:.2f} after RINEX {hdr.version:.2f}")
    if (ended or layout2) and layout2 != layout:
        changed = sorted(set(layout) | set(layout2), key=lambda s: (s != "*", s))
        diffs.append("observation types " + "; ".join(
            f"{'' if s == '*' else s + ': '}{' '.join(layout2.get(s, [])) or '(none)'}"
            for s in changed if layout.get(s) != layout2.get(s)))
    if (ended or "SYS / SCALE FACTOR" in raw2) and second.scale_factors != hdr.scale_factors:
        diffs.append("SYS / SCALE FACTOR")
    if (ended or "TIME OF FIRST OBS" in raw2) and _tag_system(second) != _tag_system(hdr):
        diffs.append(f"time system {second.time_system} after {hdr.time_system}")
    if diffs:
        k = line_no()
        raise RinexObsError(f"{path}: a second header inside the data at line {k} (a concatenated file?) "
                            f"changes how the records after it are read ({'; '.join(diffs)}); split the file at "
                            f"line {k}")
    if not ended:
        st.embedded.append((line_no(), ["no END OF HEADER before the next epoch record"], False))
        return pos
    changes = _site_changes(hdr, second)
    # the line number is named in the warning for the first header and the first one with site changes only
    first_site = changes and not any(e[1] for e in st.embedded if e[2])
    st.embedded.append((line_no() if not st.embedded or first_site else None, changes, True))
    return pos


def _event_epoch(ln: bytes, parse) -> str:
    """Time of an event record (GPST) for a message, or '(no epoch)' when its epoch fields are blank."""
    try:
        return _fmt_ns(parse(ln))
    except (ValueError, TypeError):
        return "(no epoch)"


def _is_epoch3(ln: bytes) -> bool:
    """A RINEX 3/4 epoch record ('>' in column 1)."""
    return ln[:1] == b">"


def _epoch3_ns(ln: bytes, rinex4: bool) -> int:
    """'> yyyy mm dd hh mm ss.sssssss' -> ns since 1970 (exact to the written digits; RINEX 4.02 picoseconds in
    columns 58-62 floored to ns)."""
    ns = _day_ns(int(ln[2:6]), int(ln[7:9]), int(ln[10:12])) + (int(ln[13:15]) * 3600 + int(ln[16:18]) * 60) * _NS \
        + _seconds_ns(ln[18:29].decode("ascii"))
    if rinex4:
        ps = ln[57:62]
        if len(ps) == 5 and ps.isdigit():
            ns += int(ps) // 1000
    return ns


def _epoch2_ns(ln: bytes) -> int:
    """' yy mm dd hh mm ss.sssssss' -> ns since 1970 (RINEX 2: 80-99 -> 19xx, 00-79 -> 20xx)."""
    yy = int(ln[1:3])
    y = yy + (1900 if yy >= 80 else 2000)
    return _day_ns(y, int(ln[4:6]), int(ln[7:9])) + (int(ln[10:12]) * 3600 + int(ln[13:15]) * 60) * _NS \
        + _seconds_ns(ln[15:26].decode("ascii"))


def _scan3(buf: bytes, pos: int, hdr: ObsHeader, layout: dict, lim, shift: int, path) -> tuple:
    """Walk the records of a RINEX 3/4 file. Returns ({system: _Pending}, _ScanState)."""
    n = len(buf)
    find = buf.find
    st = _ScanState()
    rinex4 = hdr.version >= 4
    pend = {s.encode(): _Pending(len(c), 3 + 16 * len(c)) for s, c in layout.items() if c}
    times = st.times
    flags = st.flags
    lo, hi = lim if lim is not None else (-(2 ** 62), 2 ** 62)
    chunk = _PARSE_CHUNK
    is_epoch = _is_epoch3
    while pos < n:
        j = find(b"\n", pos)
        if j < 0:
            j = n
        if buf[pos] != 62:  # not '>'
            if _is_rvt(buf[pos:j]):
                pos = _embedded_header(buf, pos, hdr, layout, path, st, is_epoch)
                continue
            if buf[pos:j].strip():
                _bad(st, buf, pos)
            pos = j + 1
            continue
        ln = buf[pos:j]
        line_pos = pos
        pos = j + 1
        fl = ln[31:32]
        nrec = _to_int(ln[32:35])
        if nrec is None or nrec < 0 or not (fl.isdigit() or fl in (b"", b" ")):
            _bad(st, buf, line_pos)
            continue
        flag = int(fl) if fl.isdigit() else 0
        if flag > 6:
            _bad(st, buf, line_pos)
            continue
        flags[flag] += 1
        if 2 <= flag <= 5:
            block = []
            for _ in range(nrec):
                if pos >= n:
                    break
                j = find(b"\n", pos)
                j = n if j < 0 else j
                rl = buf[pos:j]
                if rl[:1] == b">" and not rl[60:80].strip():
                    break  # an epoch record: the special records were fewer than declared
                block.append(rl)
                pos = j + 1
            if flag in (3, 4):
                _type_change(block, layout, False, path, flag, buf, line_pos, hdr)
            if flag in (2, 3):
                st.events[flag].append(_event_epoch(ln, lambda x: _epoch3_ns(x, rinex4) + shift))
            shifts = [b[:60].decode("ascii", errors="replace") for b in block if b[60:80].strip() == _PS_LABEL]
            if shifts:
                try:
                    t_event = _epoch3_ns(ln, rinex4) + shift
                except ValueError:
                    t_event = None
                    st.ps_pending.append(len(st.phase_shift_blocks))
                st.phase_shift_blocks.append([t_event, shifts])
            continue
        take = flag <= 1
        if take:
            try:
                te = _epoch3_ns(ln, rinex4) + shift
            except ValueError:
                _bad(st, buf, line_pos)
                take = False
            else:
                if st.ps_pending:                    # event blocks without an epoch: the next data epoch
                    for i in st.ps_pending:
                        st.phase_shift_blocks[i][0] = te
                    st.ps_pending.clear()
                take = lo <= te <= hi
        if take:
            e = len(times)
            times.append(te)
            if flag == 1:
                st.flag1.append(e)
        k = 0
        while k < nrec:
            if pos >= n:
                st.truncated_last = (_event_epoch(ln, lambda x: _epoch3_ns(x, rinex4) + shift), k, nrec)
                break
            j = find(b"\n", pos)
            if buf[pos] == 62 or (buf[pos] == 32 and _is_rvt(buf[pos:n if j < 0 else j])):
                # '>' (or the first line of a header: a concatenated file): the epoch was truncated, resynchronise
                st.truncated_mid += 1
                break
            if j < 0:
                j = n
                p = pend.get(buf[pos:pos + 1])
                if k + 1 < nrec or not _complete_last_line(buf[pos:j], 3, p.nt if p is not None else 0):
                    st.cut_record = True
                    st.truncated_last = (_event_epoch(ln, lambda x: _epoch3_ns(x, rinex4) + shift), k, nrec)
                    pos = n
                    break
            if take:
                rl = buf[pos:j]
                p = pend.get(rl[:1])
                if p is not None:
                    if len(rl) > p.width and rl[p.width:].strip():
                        st.overflow[rl[:1].decode("ascii", errors="replace")] += 1
                    p.lines.append(rl)
                    p.eps.append(e)
                    if len(p.lines) >= chunk:
                        _flush(p)
                elif rl.strip():
                    st.unknown_sys[rl[:1].decode("ascii", errors="replace")] += 1
            pos = j + 1
            k += 1
    for p in pend.values():
        _flush(p)
    return {s.decode(): p for s, p in pend.items()}, st


@functools.lru_cache(maxsize=4096)
def _sv2(tok: bytes) -> Optional[bytes]:
    """RINEX 2 satellite token ('G01', 'G 1', 'G1 ', ' 1', '  1') -> b'G01' (the number read as a Fortran I2, blanks
    ignored; a blank system letter is GPS); None when unreadable, blank or zero."""
    if len(tok) < 3:
        return None
    s = tok[:1]
    if s == b" ":
        s = b"G"
    elif not s.isalpha():
        return None
    num = tok[1:3].replace(b" ", b"")
    if not num.isdigit() or int(num) == 0:
        return None
    return s.upper() + b"%02d" % int(num)


def _is_epoch2(ln: bytes) -> bool:
    """A RINEX 2 epoch line: flag digit in column 29, satellite count in 30-32, and a time with its decimal point
    in column 19 (or blank time fields for an event without a significant epoch)."""
    fl = ln[28:29]
    if not (fl.isdigit() or fl == b" "):
        return False
    if _to_int(ln[29:32]) is None:
        return False
    if ln[18:19] == b".":
        return True
    return not ln[:26].strip() and fl in b"2345"


_EVENT_FLAG_DIGITS = frozenset((b"2", b"3", b"4", b"5"))
_LETTER = re.compile(rb"[A-Za-z]")


def _sat_tokens_ok(line: bytes, k: int) -> bool:
    """The ``k`` satellite tokens of an epoch or continuation line (columns 33-35, 36-38, ...) are readable and
    nothing but blanks and the receiver clock offset (columns 69-80, no letters) follows them."""
    if any(_sv2(line[32 + 3 * i:35 + 3 * i]) is None for i in range(k)):
        return False
    return not line[32 + 3 * k:68].strip() and not _LETTER.search(line[68:80])


def _epoch2_start(rl: bytes) -> bool:
    """A RINEX 2 epoch record, not a header line of an event block (its label is in columns 61-80) and not a
    record line. Letters in columns 61-80 are the system letters of the 11th and 12th satellite (columns 63 and
    66) of an epoch line with 11 or more satellites; a line is an epoch record when its satellite tokens read
    (:func:`_sat_tokens_ok`). An event record (flags 2-5) has no satellite list, so letters there are a label."""
    if not _is_epoch2(rl):
        return False
    if not _LETTER.search(rl[60:80]):
        return True
    if rl[28:29] in _EVENT_FLAG_DIGITS:
        return False
    return _sat_tokens_ok(rl, min(12, int(rl[29:32])))


def _sat_list2(buf: bytes, ln: bytes, pos: int, nsat: int) -> tuple:
    """(satellite tokens, offset after the list, problem) of a RINEX 2 data epoch whose epoch line is ``ln`` and
    whose continuation lines (32X, 12(A1,I2)) start at ``pos``. ``problem`` is None for a readable list, otherwise
    why it is not: an unreadable token, a token where the count says the list has ended (a count too low), a blank
    token where it has not (a count too high), a continuation line that is not one, or the end of the file inside
    the list ('cut')."""
    n = len(buf)
    toks: list = []
    line, left, first = ln, nsat, True
    while True:
        k = min(12, left)
        if not first and line[:32].strip():
            return None, pos, "a satellite continuation line is missing (a record line follows)"
        if not _sat_tokens_ok(line, k):
            return None, pos, (f"the satellite list does not hold the {nsat} satellites the epoch record declares"
                               if first else f"a satellite continuation line does not hold the {k} satellites the "
                               f"count of {nsat} leaves for it")
        toks += [line[32 + 3 * i:35 + 3 * i] for i in range(k)]
        left -= k
        if left <= 0:
            return toks, pos, None
        if pos >= n:
            return None, pos, "cut"
        j = buf.find(b"\n", pos)
        j = n if j < 0 else j
        line, pos, first = buf[pos:j], j + 1, False


def _skip_to_epoch2(buf: bytes, pos: int) -> tuple:
    """(offset of the next line at or after ``pos`` that looks like a RINEX 2 epoch record (:func:`_is_epoch2`:
    the main scan then reads or rejects it) or starts a header, record-like lines skipped). A record line never has
    a decimal point in column 19 (F14.3 fields put it in columns 11, 27, 43, ...). Blank lines and header-like
    lines (letters in columns 61-80, where a record line has numbers, LLI and SSI digits: a stray COMMENT) are
    passed over without being counted; the main scan counts the header-like ones as unreadable lines."""
    n = len(buf)
    skipped = 0
    while pos < n:
        j = buf.find(b"\n", pos)
        j = n if j < 0 else j
        ln = buf[pos:j]
        if _is_epoch2(ln) or _is_rvt(ln):
            break
        if ln.strip() and not _LETTER.search(ln[60:80]):
            skipped += 1
        pos = j + 1
    return pos, skipped


def _scan2(buf: bytes, pos: int, hdr: ObsHeader, layout: dict, lim, shift: int, path) -> tuple:
    """Walk the records of a RINEX 2 file. Returns ({system: _Pending}, _ScanState). The one type list of the
    header applies to the records of every system letter found in the data.

    RINEX 2 records carry no satellite id: the n-th record of an epoch belongs to the n-th satellite of its list,
    so a line missing or extra anywhere in an epoch moves every record after it to another satellite or type.
    An epoch whose lines do not fit its satellite count (a new epoch record where a record line is expected, a
    record line holding a value past the fields of the types, or a line left over before the next epoch record)
    is therefore skipped whole, and an epoch record whose satellite list does not read is skipped with its
    records. An epoch cut by the end of the file or by a header of a concatenated file keeps its complete
    records (the lines before the cut are the epoch's own)."""
    n = len(buf)
    find = buf.find
    st = _ScanState()
    codes = layout["*"]
    nt = len(codes)
    nl = max(1, math.ceil(nt / 5))
    last_fields = nt - 5 * (nl - 1) if nt else 5
    tail = 16 * last_fields if nt else 80  # columns of the fields on the last line of a record
    width = 3 + 80 * nl
    pend: dict = {}
    times = st.times
    flags = st.flags
    lo, hi = lim if lim is not None else (-(2 ** 62), 2 ** 62)
    chunk = _PARSE_CHUNK
    while pos < n:
        j = find(b"\n", pos)
        j = n if j < 0 else j
        ln = buf[pos:j]
        line_pos = pos
        if not ln.strip():
            pos = j + 1
            continue
        if not _is_epoch2(ln):
            if _is_rvt(ln):
                pos = _embedded_header(buf, pos, hdr, layout, path, st, _epoch2_start)
                continue
            _bad(st, buf, line_pos)
            pos = j + 1
            continue
        pos = j + 1
        fl = ln[28:29]
        flag = int(fl) if fl.isdigit() else 0
        nsat = int(ln[29:32])
        if flag > 6:
            _bad(st, buf, line_pos)
            continue
        if 2 <= flag <= 5:
            flags[flag] += 1
            block = []
            for _ in range(nsat):
                if pos >= n:
                    break
                j = find(b"\n", pos)
                j = n if j < 0 else j
                rl = buf[pos:j]
                if _epoch2_start(rl):
                    break  # an epoch record: the special records were fewer than declared
                block.append(rl)
                pos = j + 1
            if flag in (3, 4):
                _type_change(block, layout, True, path, flag, buf, line_pos, hdr)
            if flag in (2, 3):
                st.events[flag].append(_event_epoch(ln, lambda x: _epoch2_ns(x) + shift))
            continue
        # satellite list: 12 per line from column 33, continuation lines for more
        toks, after, problem = _sat_list2(buf, ln, pos, nsat)
        if problem == "cut":
            flags[flag] += 1
            st.truncated_last = (_event_epoch(ln, lambda x: _epoch2_ns(x) + shift), 0, nsat)
            pos = n
            break
        if problem is not None:
            pos, skipped = _skip_to_epoch2(buf, pos)
            st.bad_epochs.append((_line_no(buf, line_pos), problem, skipped))
            continue
        flags[flag] += 1
        pos = after
        take = flag <= 1
        if take:
            try:
                te = _epoch2_ns(ln) + shift
            except ValueError:
                _bad(st, buf, line_pos)
                take = False
            else:
                take = lo <= te <= hi
        e = len(times)
        records = []                     # (record number, lines) of this epoch, kept when the epoch fits
        dropped = None
        cut = False
        for k in range(nsat):
            lines = []
            early = misfit = False
            for i in range(nl):
                if pos >= n:
                    cut = True
                    break
                j = find(b"\n", pos)
                if j < 0:
                    j = n
                    if k + 1 < nsat or i + 1 < nl or not _complete_last_line(buf[pos:j], 0, last_fields):
                        st.cut_record = True
                        cut = True
                        pos = n
                        break
                rl = buf[pos:j]
                if _is_rvt(rl):
                    cut = True     # a header of a concatenated file: the epoch was cut at the end of its part
                    break
                if (rl[18:19] == b"." or (rl[28:29] in _EVENT_FLAG_DIGITS and not rl[:26].strip())) \
                        and _epoch2_start(rl):
                    early = True   # a new epoch record started early: a line of this epoch is missing
                    break
                if (i + 1 == nl and len(rl) > tail and rl[tail:].strip()) or (len(rl) > 80 and rl[80:].strip()):
                    misfit = True  # a value past the last field of a record: a line is missing or extra
                    break
                lines.append(rl)
                pos = j + 1
            if cut:
                if pos < n:
                    st.truncated_mid += 1      # ended by a header line
                else:
                    st.truncated_last = (_event_epoch(ln, lambda x: _epoch2_ns(x) + shift), k, nsat)
                break
            if early:
                dropped = ("early", k, 0)
                break
            if misfit:
                pos, skipped = _skip_to_epoch2(buf, pos)
                dropped = ("misfit", k, skipped)
                break
            records.append((k, lines))
        if dropped is None and not cut:
            nxt, skipped = _skip_to_epoch2(buf, pos)
            if skipped:
                dropped = ("extra", nsat, skipped)   # record lines left over: a line of the epoch is extra
                pos = nxt                            # (header-like lines after it are left to the main scan)
        if dropped is not None:
            kind, k, skipped = dropped
            st.dropped[kind].append((_event_epoch(ln, lambda x: _epoch2_ns(x) + shift), k, nsat, skipped))
            continue
        if not take:
            continue
        times.append(te)
        if flag == 1:
            st.flag1.append(e)
        if not nt:
            continue
        for k, lines in records:
            sv = _sv2(toks[k])
            p = pend.get(sv[:1])
            if p is None:
                p = pend[sv[:1]] = _Pending(nt, width)
            p.lines.append(sv + b"".join([b.ljust(80)[:80] for b in lines]))
            p.eps.append(e)
            if len(p.lines) >= chunk:
                _flush(p)
    for p in pend.values():
        _flush(p)
    return {s.decode(): p for s, p in pend.items()}, st


def _finish_system(p: _Pending, codes_file: list, out_codes: list, times: np.ndarray, flag1: np.ndarray,
                   scale: dict, version: float, system: str) -> tuple:
    """(_SysData or None, duplicates dropped, unreadable satellites, undefined LLI digits, (unreadable fields,
    first one's text)) of one system."""
    if not p.chunks:
        return None, 0, 0, 0, (0, None)
    bad_prn = sum(c[5] for c in p.chunks)
    undefined = sum(c[6] for c in p.chunks)
    unreadable = (sum(c[7] for c in p.chunks), next((c[8] for c in p.chunks if c[8] is not None), None))
    prn = np.concatenate([c[0] for c in p.chunks])
    eps = np.concatenate([c[1] for c in p.chunks])
    vals = np.concatenate([c[2] for c in p.chunks]) if len(p.chunks) > 1 else p.chunks[0][2]
    lli = np.concatenate([c[3] for c in p.chunks]) if len(p.chunks) > 1 else p.chunks[0][3]
    ssi = np.concatenate([c[4] for c in p.chunks]) if len(p.chunks) > 1 else p.chunks[0][4]
    p.chunks = []
    if not len(prn):
        return None, 0, bad_prn, undefined, unreadable
    tt = times[eps]
    if len(flag1):
        rows = np.isin(eps, flag1)
        if rows.any():
            ph = np.array([c[:1] == "L" for c in codes_file])
            lli[np.ix_(rows, ph)] |= 1
    # output fields: header order; RINEX <= 3.02 BeiDou band 1 read as band 2: the band-2 field where it holds a
    # value, the band-1 field otherwise (a satellite tracked on one of them only keeps its data)
    mapped = _bds_b1_as_b2(codes_file) if system == "C" and version < 3.025 else list(codes_file)
    fields, alts = [], []
    for code in out_codes:
        cand = [k for k, c in enumerate(mapped) if c == code]
        native = [k for k in cand if codes_file[k] == code]
        first = (native or cand)[0]
        fields.append(first)
        alts.append([k for k in cand if k != first])
    fields = np.asarray(fields, dtype=np.int64)
    order = np.lexsort((tt, prn))
    prn, tt = prn[order], tt[order]
    dup = np.zeros(len(order), dtype=bool)
    dup[1:] = (prn[1:] == prn[:-1]) & (tt[1:] == tt[:-1])
    n_dup = int(dup.sum())
    keep = order[~dup] if n_dup else order
    if n_dup:
        prn, tt = prn[~dup], tt[~dup]
    sel = np.ix_(keep, fields)
    out_vals, out_lli, out_ssi = vals[sel], lli[sel], ssi[sel]
    for j, alt in enumerate(alts):
        for k in alt:
            a = vals[keep, k]
            fill = np.isnan(out_vals[:, j]) & ~np.isnan(a)
            if fill.any():
                out_vals[fill, j] = a[fill]
                out_lli[fill, j] = lli[keep, k][fill]
                out_ssi[fill, j] = ssi[keep, k][fill]
    for j, code in enumerate(out_codes):
        f = scale.get(code)
        if f and f != 1.0:
            out_vals[:, j] /= f
    return _SysData(codes=list(out_codes), fields=fields, prn=prn.astype(np.int16), time=tt.astype(np.int64),
                    vals=out_vals, lli=out_lli, ssi=out_ssi), n_dup, bad_prn, undefined, unreadable


def _parse(box: list, path, lim) -> _Parsed:
    """Parse a whole observation file within ``lim`` for every system. ``box``: a list holding the file's bytes;
    they are taken out of it, so the text is held once and freed after the scan.

    ``notes`` and ``warn_texts`` of the result do not name the file (a cache hit may come through another path
    to the same bytes); :func:`read_obs` puts the path in front."""
    cut: list = []
    buf = _plain(box.pop(), path, cut)
    lines, pos = _header_of(buf, path, cut)
    hdr, layout = _parse_header(lines, path)
    tsys = (hdr.time_system or "GPS").upper()
    rinex2 = hdr.version < 3
    if tsys in ("GLO", "UTC"):
        raise RinexObsError(f"{path}: time tags in {tsys} time: GLONASS/UTC time tags are not supported; convert "
                            "to GPS time")
    if tsys not in _TIME_SHIFT_S:
        raise RinexObsError(f"{path}: time system {tsys!r} of TIME OF FIRST OBS is not supported (GPS, GAL, QZS, "
                            "IRN and BDT are; GLO and UTC time tags must be converted to GPS time)")
    if rinex2 and "*" in hdr.unreadable_types:
        raise RinexObsError(f"{path}: {hdr.unreadable_types['*']}: the record fields cannot be matched to the types "
                            "(a header line of the type list missing?); no observation can be read")
    shift = _TIME_SHIFT_S[tsys] * _NS
    notes: list = []
    warn_texts: list = list(cut)
    if shift:
        notes.append(f"{tsys} time tags moved to GPST (+{_TIME_SHIFT_S[tsys]} s)")
    if rinex2:
        pend, st = _scan2(buf, pos, hdr, layout, lim, shift, path)
    else:
        pend, st = _scan3(buf, pos, hdr, layout, lim, shift, path)
    del buf
    times = np.asarray(st.times, dtype=np.int64)
    flag1 = np.asarray(st.flag1, dtype=np.int64)
    systems = {}
    dups = {}
    bad_prn = undefined_lli = 0
    unreadable = Counter()
    unreadable_sample = None
    for s, p in pend.items():
        codes_file = layout["*"] if rinex2 else layout.get(s, [])
        out_codes = hdr.obs_types.get(s) or list(dict.fromkeys(codes_file))
        data, n_dup, n_bad, n_undef, (n_unread, sample) = _finish_system(
            p, codes_file, out_codes, times, flag1, hdr.scale_factors.get(s, {}), hdr.version, s)
        bad_prn += n_bad
        undefined_lli += n_undef
        if n_unread:
            unreadable[s] += n_unread
            unreadable_sample = unreadable_sample or sample
        if n_dup:
            dups[s] = n_dup
        if data is not None:
            systems[s] = data
    # warnings (one per file and kind)
    if dups:
        warn_texts.append("duplicate (sv, time) records dropped, the first of the file kept: "
                          + ", ".join(f"{s} {k}" for s, k in dups.items()))
    if st.truncated_last is not None:
        ep, k, nrec = st.truncated_last
        warn_texts.append(f"the last epoch {ep} is truncated: {k} of {nrec} satellite records complete "
                          "(the complete records are kept"
                          + (", a record cut in the middle of a line is dropped: the file ends without a newline "
                             "inside it)" if st.cut_record else ")"))
    if st.truncated_mid and not rinex2:
        warn_texts.append(f"{st.truncated_mid} epoch(s) hold fewer satellite records than declared (a new "
                          "epoch record started early); their complete records are kept")
    elif st.truncated_mid:
        warn_texts.append(f"{st.truncated_mid} epoch(s) end at a header line inside the data before all their "
                          "satellite records (a concatenated file: the part before it was cut); their complete "
                          "records are kept")
    why = "RINEX 2 records carry no satellite id, so the records after the defect would belong to other satellites"
    for kind, text in (("early", "hold fewer record lines than their satellite count declares (a new epoch record "
                                 "follows early: a line is missing)"),
                       ("misfit", f"hold a record line that does not fit the {len(layout.get('*', []))} observation "
                                  "types (a line missing or extra)"),
                       ("extra", "hold more record lines than their satellite count declares (a line is extra)")):
        found = st.dropped[kind]
        if found:
            ep, k, nrec, _ = found[0]
            where = f"first at {ep}" + (f", record {k + 1} of {nrec}" if kind != "extra" else "")
            skipped = sum(m[3] for m in found)
            warn_texts.append(f"{len(found)} epoch(s) {text}; {where}: these epochs are skipped whole ({why})"
                              + (f"; {skipped} line(s) after the defect skipped" if skipped else ""))
    if st.bad_epochs:
        line, reason, _ = st.bad_epochs[0]
        warn_texts.append(f"{len(st.bad_epochs)} epoch record(s) with an unreadable satellite list skipped with their "
                          f"records ({sum(b[2] for b in st.bad_epochs)} line(s)); first at line {line}: {reason}")
    if st.overflow:
        warn_texts.append("records holding values past the fields of the header's observation types (not read): "
                          + ", ".join(f"{s} {k}" for s, k in sorted(st.overflow.items())))
    if unreadable:
        warn_texts.append(f"{sum(unreadable.values())} observation field(s) that are not numbers read as missing "
                          f"(NaN; {', '.join(f'{s} {k}' for s, k in sorted(unreadable.items()))}; first "
                          f"{unreadable_sample!r})")
    events = []
    for flag, what in ((2, "start of moving antenna (event flag 2: kinematic from that epoch)"),
                       (3, "new site occupation (event flag 3)")):
        ev = st.events[flag]
        if ev:
            events.append(f"{what} at " + ", ".join(ev[:10]) + (f" and {len(ev) - 10} more" if len(ev) > 10 else ""))
    if events:
        warn_texts.append("; ".join(events) + "; the observations before and after are read as one station "
                          "(split the file to process the occupations separately)")
    if st.embedded:
        complete = [e for e in st.embedded if e[2]]
        text = (f"{len(st.embedded)} header(s) inside the data (a concatenated file?), first at line "
                f"{st.embedded[0][0]}")
        if complete:
            text += ": the records after them have the observation types of the first header and are read"
        cut = [e for e in st.embedded if not e[2]]
        if cut:
            text += f"; the header at line {cut[0][0]} is incomplete ({cut[0][1][0]})"
            if not cut[0][1][0].startswith("the file ends"):
                text += ": the records after it are read with the observation types of the first header"
        site = [e for e in complete if e[1]]
        if site:
            text += (f"; the header at line {site[0][0]} describes another station or set-up "
                     f"({', '.join(site[0][1])}): the observations before and after it are read as one station "
                     "(split the file to process them separately)")
        warn_texts.append(text)
    if st.bad_lines or bad_prn:
        warn_texts.append(f"{st.bad_lines + bad_prn} unreadable line(s) or satellite(s) skipped"
                          + (f" (first at line {st.first_bad})" if st.first_bad else ""))
    for s, reason in hdr.unreadable_types.items():
        k = st.unknown_sys.pop(s, 0)
        warn_texts.append(f"{reason}: the record fields cannot be matched to the types, so the {s} observations "
                          f"are not read ({k} record(s) skipped)")
    if st.unknown_sys:
        warn_texts.append("records of system(s) without observation types in the header skipped: "
                          + ", ".join(f"{s or '?'} {k}" for s, k in sorted(st.unknown_sys.items())))
    if undefined_lli:
        warn_texts.append(f"{undefined_lli} LLI digit(s) 8 or 9 (undefined: the LLI has three bits, 0-7) read as 0")
    if not rinex2 and "C" in layout and hdr.version < 3.025:
        mapped = _bds_b1_as_b2(layout["C"])
        both = sorted({c for c, m in zip(layout["C"], mapped) if m != c and m in layout["C"]})
        if both or len(set(mapped)) < len(mapped):
            warn_texts.append(f"RINEX {hdr.version:.2f} BeiDou lists band 1 and band 2 codes "
                              f"({', '.join(both)} and their band-2 twins): the band-2 field is read where it holds "
                              "a value, the band-1 field otherwise")
    # data interval, first and last epoch
    ut = np.unique(times)
    interval_s = interval_source = None
    if len(ut) >= 2:
        interval_s = float(np.median(np.diff(ut))) / _NS
        interval_source = "data"
        if hdr.interval_header is not None and abs(hdr.interval_header - interval_s) > 0.01 * interval_s:
            hdr.issues.append(f"INTERVAL {hdr.interval_header:g} s of the header differs from the data interval "
                              f"{interval_s:g} s by more than 1 %")
    elif hdr.interval_header is not None:
        interval_s, interval_source = float(hdr.interval_header), "header"
    else:
        from_name = _interval_from_name(path)
        if from_name is not None:
            interval_s, interval_source = from_name, "name"
    notes = warn_texts + notes
    ps_events = []
    for t_event, payloads in st.phase_shift_blocks:
        records, issues = parse_phase_shift_records(payloads)
        when = "(no epoch)" if t_event is None else _fmt_ns(t_event)
        hdr.issues += [f"event block at {when}: {text}" for text in issues]
        ps_events += [(t_event, system, code, sats, value) for system, code, value, sats in records]
    return _Parsed(header=hdr, systems=systems, epoch_flags=st.flags, interval_s=interval_s,
                   interval_source=interval_source, t_first=int(ut[0]) if len(ut) else None,
                   t_last=int(ut[-1]) if len(ut) else None, notes=notes, warn_texts=warn_texts,
                   phase_shift_events=ps_events)


# ============================================================================ cache and frames
_CACHE: "OrderedDict[tuple, _Parsed]" = OrderedDict()
_CACHE_SIZE = 1
_LOCK = threading.Lock()


def clear_cache() -> None:
    """Empty the process-wide reader cache."""
    with _LOCK:
        _CACHE.clear()


def _load(path, tlim) -> tuple:
    """(_Parsed, parsed now) of ``path`` within ``tlim`` from the cache, parsing on a miss."""
    raw = _file_bytes(path)
    lim = _tlim_ns(tlim)
    key = (len(raw), hashlib.blake2b(raw, digest_size=16).digest(), lim)
    box = [raw]
    del raw
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit, False
        parsed = _parse(box, path, lim)
        _CACHE[key] = parsed
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return parsed, True


_SV_NAMES: dict = {}


def _sv_names(system: str) -> np.ndarray:
    names = _SV_NAMES.get(system)
    if names is None:
        names = np.array([f"{system}{i:02d}" for i in range(100)], dtype=object)
        _SV_NAMES[system] = names
    return names


def _index(data: _SysData, rows: np.ndarray, system: str) -> pd.MultiIndex:
    prn = data.prn[rows]
    tt = data.time[rows]
    up, sv_codes = np.unique(prn, return_inverse=True)
    ut, t_codes = np.unique(tt, return_inverse=True)
    levels = [pd.Index(_sv_names(system)[up], dtype=object), pd.DatetimeIndex(ut.astype("datetime64[ns]"))]
    return pd.MultiIndex(levels=levels, codes=[sv_codes.ravel(), t_codes.ravel()], names=["sv", "time"],
                         verify_integrity=False)


def _frames_of(parsed: _Parsed, system: str, kinds: tuple, want_lli: bool, want_ssi: bool, attrs: dict) -> tuple:
    """(frame or None, LLI frame or None, SSI frame or None) of one system."""
    data = parsed.systems.get(system)
    if data is None:
        return None, None, None
    cols = [j for j, c in enumerate(data.codes) if c[:1] in kinds]
    if not cols:
        return None, None, None
    vals = data.vals[:, cols]
    rows = ~np.isnan(vals).all(axis=1)
    if not rows.any():
        return None, None, None
    rows = np.flatnonzero(rows)
    vals = vals[rows]
    names = [data.codes[j] for j in cols]
    idx = _index(data, rows, system)
    df = pd.DataFrame(vals, index=idx, columns=names)
    df.attrs = copy.deepcopy(attrs)
    lli_df = ssi_df = None
    if want_lli:
        ph = [k for k, j in enumerate(cols) if data.codes[j][:1] == "L"]
        lv = data.lli[rows][:, [cols[k] for k in ph]].astype(float)
        lv[np.isnan(vals[:, ph])] = np.nan
        lli_df = pd.DataFrame(lv, index=idx.copy(), columns=[names[k] for k in ph])
        lli_df.attrs = copy.deepcopy(attrs)
    if want_ssi:
        sv_ = data.ssi[rows][:, cols].astype(float)
        sv_[np.isnan(vals)] = np.nan
        ssi_df = pd.DataFrame(sv_, index=idx.copy(), columns=names)
        ssi_df.attrs = copy.deepcopy(attrs)
    return df, lli_df, ssi_df


def read_obs(path, systems: Optional[Iterable[str] | str] = None, tlim=None,
             kinds: Iterable[str] = ("C", "P", "L", "S"), lli: bool = False, ssi: bool = False) -> ObsRead:
    """Observations of ``path`` as one frame per satellite system (contract in the module docstring).

    ``systems``: a string of system letters (``'G'``, ``'GEC'``) or an iterable of letters; None reads every
    system of the header. ``tlim``: ``(start, end)``, anything ``pd.Timestamp`` accepts, inclusive, in GPST.
    ``kinds``: first characters of the observation types to keep (see :data:`OBS_KINDS`). ``lli``: also return
    the LLI frames. ``ssi`` (addition): also return the signal strength frames. Raises :class:`RinexObsError`
    (see the module docstring); ``OSError`` for a missing or unreadable file; ``ValueError`` for a ``systems``
    argument that is not made of RINEX system letters."""
    want = _norm_systems(systems)
    kinds = tuple(str(k)[:1].upper() for k in kinds)
    parsed, fresh = _load(path, tlim)
    name = os.fspath(path)
    warn_texts = [f"{name}: {text}" for text in parsed.warn_texts]
    if fresh:
        for text in warn_texts:
            warnings.warn(text, RinexWarning, stacklevel=2)
    hdr = copy.deepcopy(parsed.header)
    hdr.path = name                                        # a cache hit may come through another path
    if want is None:
        want = list(hdr.obs_types)
    ps_events = [(None if t is None else pd.Timestamp(t, unit="ns"), system, code,
                  None if sats is None else list(sats), value)
                 for t, system, code, sats, value in parsed.phase_shift_events]
    attrs = {"rinex_version": hdr.version, "time_system": hdr.time_system, "path": name,
             "epoch_flags": dict(parsed.epoch_flags), "issues": list(hdr.issues) + warn_texts,
             **_phase_shift_attrs(hdr.phase_shift, parsed.phase_shift_events)}
    interval_s, interval_source = parsed.interval_s, parsed.interval_source
    if interval_source in (None, "name"):
        # the file-name fallback belongs to the path of this call: the cache key holds the content, not the name
        from_name = _interval_from_name(path)
        interval_s, interval_source = (from_name, "name") if from_name is not None else (None, None)
    out = ObsRead(header=hdr, interval_s=interval_s, interval_source=interval_source,
                  t_first=None if parsed.t_first is None else pd.Timestamp(parsed.t_first, unit="ns"),
                  t_last=None if parsed.t_last is None else pd.Timestamp(parsed.t_last, unit="ns"),
                  epoch_flags=Counter(parsed.epoch_flags), notes=[f"{name}: {text}" for text in parsed.notes],
                  phase_shift_events=ps_events)
    for s in want:
        df, lli_df, ssi_df = _frames_of(parsed, s, kinds, lli, ssi, attrs)
        out.frames[s] = df
        if df is not None:
            if lli:
                out.lli[s] = lli_df
            if ssi:
                out.ssi[s] = ssi_df
    return out


def lli_for(path, system: str, phases: Iterable[str], tlim=None) -> pd.DataFrame:
    """LLI of the phase codes ``phases`` of one system (index ``['sv', 'time']``, one float column per phase,
    as the LLI frames of :func:`read_obs`; a requested phase absent from the file is an all-NaN column).

    Served from the reader cache: for the slip layer (``cycle_slips``) after a session has read the same file.
    The rows are those of ``read_obs(path, system, tlim, kinds=('L',), lli=True)`` (records with at least one
    phase); a system without phases gives an empty frame with the requested columns."""
    phases = [str(p) for p in phases]
    res = read_obs(path, systems=system, tlim=tlim, kinds=("L",), lli=True)
    df = res.lli.get(system)
    if df is None:
        empty = pd.MultiIndex.from_arrays([np.array([], dtype=object), pd.DatetimeIndex([], dtype="datetime64[ns]")],
                                          names=["sv", "time"])
        return pd.DataFrame({p: np.zeros(0) for p in phases}, index=empty)
    return df.reindex(columns=phases)
