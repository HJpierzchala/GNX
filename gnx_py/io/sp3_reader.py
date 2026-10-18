"""SP3 precise orbit and clock files (SP3-a, -b, -c and -d), GNX's own reader.

Written from the format descriptions (SP3-a: Remondi 1989; SP3-c: Hilla 2002/2010; SP3-d: Hilla 2016, IGS/ACC);
no code of another reader. It replaces ``georinex.load_sp3`` in :func:`gnx_py.io.read_sp3`; georinex stays an
independent reference for the validation only. Files are opened through :mod:`gnx_py.compression`, so plain, gzip, Unix
compress (.Z), bzip2 and single-member zip files read alike, detected from their first bytes.

Layout read (columns 1-based, as in the format descriptions)
-------------------------------------------------------------
* line 1 ``#cP2024  2  4  0  0  0.00000000     288 ORBIT IGS20 FIT  GRG``: version (column 2, ``a``-``d``),
  position/velocity flag (column 3, ``P`` or ``V``), first epoch, number of epochs (33-39; not checked: real files
  write a wrong count), data used (41-45), coordinate system (47-51), orbit type (53-55), agency (57-60);
* line 2 ``##``: GPS week, seconds of week, epoch interval, MJD and fraction of day (read leniently, ``None`` when a
  field does not parse);
* ``+`` lines: number of satellites (columns 4-6; SP3-a/b/c write it in 5-6), then 17 satellite ids per line in
  columns 10-12, 13-15, ...; place holders ``  0``/`` 00`` are skipped. SP3-a ids carry no system letter (``  1``
  is ``G01``); every id is normalised to letter + two digits (``G 1`` -> ``G01``);
* ``++`` lines: accuracy exponents in the order of the ``+`` list (``0`` = unknown);
* first ``%c`` line: file type (columns 4-5) and time system (columns 10-12; ``ccc`` or blank -> ``None``, as in
  SP3-a which has no ``%c`` line: GPS time);
* first ``%f`` line: base of the position/velocity (columns 4-13) and clock/rate (15-26) standard deviations;
* ``/*`` comment lines (kept, the 3 leading characters removed; accepted anywhere in the file);
* epoch lines ``*  yyyy mm dd hh mm ss.ssssssss`` (seconds 60, minutes 60 and hour 24 roll over into the next
  minute/hour/day, as some writers produce them; blanks inside the seconds field are ignored, as a Fortran F read
  ignores them, like the observation reader does);
* ``P`` records: satellite (2-4), x, y, z in km (5-18, 19-32, 33-46), clock in microseconds (47-60), standard
  deviation exponents (62-63, 65-66, 68-69, 71-73), clock event flag ``E`` (75), clock prediction flag ``P`` (76),
  manoeuvre flag ``M`` (79), orbit prediction flag ``P`` (80);
* ``EP`` records (after the ``P`` record of their satellite): standard deviations x, y, z in mm (5-8, 10-13, 15-18),
  clock in ps (20-26) and the correlations xy, xz, xc, yz, yc, zc times 1e7 (28-35, 37-44, 46-53, 55-62, 64-71,
  73-80);
* ``V`` records (after the ``P`` record of their satellite): velocity in dm/s and clock rate in 1e-4 microseconds/s,
  same columns as ``P``; ``EV`` records as ``EP`` (1e-4 mm/s, 1e-4 ps/s);
* ``EOF``: end of the data (lines after it are ignored; a file without it reads to its end, as ``minimal.sp3c`` of
  the corpus). A file without it that holds fewer epochs than line 1 declares ends at an epoch boundary of a cut
  file (or is an excerpt, as ``example1.sp3a`` of the corpus): it is read, with one :class:`SP3Warning` naming the
  file and the counts (corpus review F8; a cut inside an epoch or inside a number raises, see below).

Values are kept as written, with the format's markers: an absent position is ``0.000000`` in x, y and z, a bad or
absent clock (and clock rate) ``999999.999999`` (:data:`BAD_CLOCK` from 999999.0 up). :func:`gnx_py.io.read_sp3`
turns them into NaN per quantity.

Checks (every one raises :class:`SP3Error`, a ``ValueError`` naming the file, the line number and the line)
-------------------------------------------------------------------------------------------------------------
* an empty file, or a first non-blank line that is not an SP3 line 1 (HTML page, IONEX, RINEX under an SP3 name);
* no ``+`` satellite list, or no epoch line after the header (a header-only file);
* an epoch line or a record whose numbers do not parse; a blank clock field (SP3 writes 999999.999999 for a missing
  clock); a ``P`` or ``V`` record that ends before column 60 inside a number cut short (fewer than the six decimals
  of F14.6: the last line of a cut file, whose value would otherwise be read as a shorter number);
* an epoch not after the previous one;
* every epoch must hold one ``P`` record for each satellite of the ``+`` list, in that order (the layout of the
  format; also how a cut file shows: its last epoch misses satellites); the message names what is missing, what is
  not in the header or that the order differs;
* a ``V``/``EP``/``EV`` record that does not follow the ``P`` (``EV``: ``V``) record of its satellite;
* any other non-blank line in the data part.

Public API::

    read_sp3_file(path) -> SP3File      (header, records frame)
    read_sp3_header(path) -> SP3Header
    sp3_sv(code) -> str                 ('  1' -> 'G01', 'G 1' -> 'G01')
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import compression as _compression

__all__ = ["BAD_CLOCK", "SP3Error", "SP3File", "SP3Header", "SP3Warning", "read_sp3_file", "read_sp3_header",
           "sp3_sv"]

#: SP3 clock (and clock rate) values from this up mark a bad or absent value (the format writes 999999.999999).
BAD_CLOCK = 999999.0

_WHAT = "SP3 file"
_VERSIONS = "abcd"
_PLACEHOLDERS = ("", "0", "00", "000")


class SP3Error(ValueError):
    """A file that is not a readable SP3 file; ``path`` and ``line`` (1-based, ``None`` for the whole file) name the
    place."""

    def __init__(self, path, message: str, line: Optional[int] = None, text: Optional[str] = None):
        self.path = os.fspath(path)
        self.line = line
        where = f"SP3 file {self.path}" + (f", line {line}" if line is not None else "")
        tail = f": {text.rstrip()[:80]!r}" if text is not None else ""
        super().__init__(f"{where}: {message}{tail}")


class SP3Warning(UserWarning):
    """A readable defect of an SP3 file (a file without EOF that ends before the epochs line 1 declares); the
    message names the file."""


@dataclass
class SP3Header:
    """The header of an SP3 file (lines before the first epoch line)."""
    version: str                       # 'a', 'b', 'c' or 'd'
    pos_vel: str                       # 'P' (positions) or 'V' (positions and velocities)
    start: pd.Timestamp                # first epoch of line 1
    n_epochs: Optional[int]            # as written on line 1 (real files write wrong counts)
    data_used: str
    coord_sys: str
    orbit_type: str
    agency: str
    gps_week: Optional[int] = None
    sow: Optional[float] = None
    interval: Optional[float] = None   # seconds
    mjd: Optional[int] = None
    fmjd: Optional[float] = None
    n_sv: int = 0                      # as declared on the first '+' line
    satellites: List[str] = field(default_factory=list)
    accuracy: Dict[str, int] = field(default_factory=dict)  # exponent per satellite ('++' lines); 0 = unknown
    file_type: Optional[str] = None    # first '%c' line, columns 4-5
    time_system: Optional[str] = None  # first '%c' line, columns 10-12; None for SP3-a/b or 'ccc'
    base_pos: Optional[float] = None   # first '%f' line, columns 4-13 (None when 0 or absent)
    base_clk: Optional[float] = None   # first '%f' line, columns 15-26 (None when 0 or absent)
    comments: List[str] = field(default_factory=list)


@dataclass
class SP3File:
    """An SP3 file as read.

    ``records``: one row per ``P`` record in file order (epochs ascending, satellites in the order of the header),
    index ``(time, sv)`` (``datetime64[ns]``, ``str``), columns

    * ``x``, ``y``, ``z`` (km), ``clk`` (microseconds), as written (markers kept);
    * ``vx``, ``vy``, ``vz`` (dm/s), ``dclk`` (1e-4 microseconds/s): NaN for a record without a ``V`` record;
    * ``sx``, ``sy``, ``sz``, ``sclk``: standard deviation exponents of the ``P`` record (NaN when blank);
    * ``clk_event``, ``clk_pred``, ``maneuver``, ``orb_pred``: the flags of the ``P`` record (bool);
    * only when the file has ``EP`` (``EV``) records: ``ep_sx``, ``ep_sy``, ``ep_sz``, ``ep_sclk``, ``ep_cxy``,
      ``ep_cxz``, ``ep_cxc``, ``ep_cyz``, ``ep_cyc``, ``ep_czc`` (``ev_...``), NaN for a record without one.

    ``velocity``: whether the file has ``V`` records; ``eof``: whether it ends with an ``EOF`` line. ``notes``: the
    texts of the warnings emitted."""
    path: str
    header: SP3Header
    records: pd.DataFrame
    velocity: bool
    eof: bool
    notes: List[str] = field(default_factory=list)

    @property
    def time_system(self) -> Optional[str]:
        return self.header.time_system


def sp3_sv(code: str) -> str:
    """SP3 satellite id normalised to letter + two digits: SP3-a ``'  1'`` -> ``'G01'``, ``'G 1'`` -> ``'G01'``."""
    code = code.strip().rjust(3)
    return ("G" if code[0] == " " else code[0]) + code[1:].replace(" ", "0")


# ------------------------------------------------------------------------------------------------- parsing helpers
def _epoch_ns(text: str) -> int:
    """Nanoseconds since 1970 of the date fields of an epoch line or line 1 (columns 4-31). Seconds 60, minutes 60
    and hour 24 roll over. Raises ValueError (the caller names the line)."""
    year, month, day = int(text[3:7]), int(text[8:10]), int(text[11:13])
    hour, minute = int(text[14:16]), int(text[17:19])
    sec = text[20:31].replace(" ", "")          # blanks inside the F11.8 field are ignored, as a Fortran F read does
    whole, _, frac = sec.partition(".")
    if not whole.isdigit() or (frac and not frac.isdigit()) or len(frac) > 9:
        raise ValueError(f"seconds field {sec!r}")
    day0 = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "ns")   # ValueError for an impossible date
    if not (0 <= hour <= 24 and 0 <= minute <= 60 and int(whole) <= 60):
        raise ValueError("hour, minute or second out of range")
    ns = ((hour * 60 + minute) * 60 + int(whole)) * 1_000_000_000 + int((frac or "0").ljust(9, "0"))
    return int(day0.astype(np.int64)) + ns


#: A complete F14.6 number (six decimals): a number cut short by the end of a cut file has fewer.
_COMPLETE_F146 = re.compile(r"[+-]?\d*\.\d{6}")


def _cut_record(line: str) -> Optional[int]:
    """The column a ``P``/``V`` record line ends in when it ends before column 60 inside a number cut short (its
    last number has fewer than six decimals: the last line of a cut file); None for a complete line (60 columns or
    more, or a shorter line whose last number is complete, e.g. a blank clock field: checked on its own)."""
    used = len(line.rstrip())
    if used >= 60:
        return None
    tokens = line[4:used].split()
    if tokens and _COMPLETE_F146.fullmatch(tokens[-1]):
        return None
    return used


def _float(text: str) -> float:
    """``float`` of a fixed-width field; blank or bad text raises ValueError."""
    return float(text)


def _opt_int(text: str) -> float:
    text = text.strip()
    return float(int(text)) if text else np.nan


_EP_NAMES = ("sx", "sy", "sz", "sclk", "cxy", "cxz", "cxc", "cyz", "cyc", "czc")
_EP_SLICES = ((4, 8), (9, 13), (14, 18), (19, 26), (27, 35), (36, 44), (45, 53), (54, 62), (63, 71), (72, 80))


def _ep_values(line: str) -> list:
    return [_opt_int(line[a:b]) for a, b in _EP_SLICES]


def _parse_header_line1(path, line: str, lineno: int) -> SP3Header:
    if len(line) < 2 or line[0] != "#" or line[1] not in _VERSIONS:
        raise SP3Error(path, "not an SP3 file (line 1 must start with '#a', '#b', '#c' or '#d')", lineno, line)
    pos_vel = line[2:3]
    if pos_vel not in ("P", "V"):
        raise SP3Error(path, "line 1: position/velocity flag (column 3) is neither 'P' nor 'V'", lineno, line)
    try:
        start = pd.Timestamp(_epoch_ns(line), unit="ns")
    except (ValueError, IndexError) as exc:
        raise SP3Error(path, f"line 1: first epoch (columns 4-31) does not parse ({exc})", lineno, line) from None
    try:
        n_epochs: Optional[int] = int(line[32:39])
    except ValueError:
        n_epochs = None
    return SP3Header(version=line[1], pos_vel=pos_vel, start=start, n_epochs=n_epochs,
                     data_used=line[40:45].strip(), coord_sys=line[46:51].strip(), orbit_type=line[52:55].strip(),
                     agency=line[56:60].strip())


def _parse_line2(header: SP3Header, line: str) -> None:
    for name, sl, conv in (("gps_week", slice(3, 7), int), ("sow", slice(8, 23), float),
                           ("interval", slice(24, 38), float), ("mjd", slice(39, 44), int),
                           ("fmjd", slice(45, 60), float)):
        try:
            setattr(header, name, conv(line[sl]))
        except ValueError:
            pass


def _ids(line: str) -> List[str]:
    return [line[k:k + 3] for k in range(9, 60, 3)]


# ------------------------------------------------------------------------------------------------------- readers
def _lines(path):
    """Numbered lines of the decoded file (compression undone); CompressedFileError for a broken archive."""
    text = _compression.open_text(path, what=_WHAT)
    for lineno, line in enumerate(text, start=1):
        yield lineno, line.rstrip("\n")


def _read(path, header_only: bool):
    path = os.fspath(path)
    lines = _lines(path)
    header = None
    first_lineno = None
    for lineno, line in lines:
        if line.strip() and line.strip() != "\x1a":
            header = _parse_header_line1(path, line, lineno)
            first_lineno = lineno
            break
    if header is None:
        raise SP3Error(path, "empty file (no SP3 line 1)")

    plus_lines: List[str] = []
    acc_lines: List[str] = []
    seen_c = seen_f = False
    epoch_line = None
    for lineno, line in lines:
        if line.startswith("*"):
            epoch_line = (lineno, line)
            break
        if line.startswith("##"):
            _parse_line2(header, line)
        elif line.startswith("++"):
            acc_lines.append(line)
        elif line.startswith("+"):
            if not plus_lines:
                try:
                    header.n_sv = int(line[3:6])
                except ValueError:
                    raise SP3Error(path, "number of satellites (columns 4-6 of the first '+' line) does not "
                                         "parse", lineno, line) from None
            plus_lines.append(line)
        elif line.startswith("%c") and not seen_c:
            seen_c = True
            header.file_type = line[3:5].strip() or None
            value = line[9:12].strip()
            header.time_system = value if value and value.lower() != "ccc" else None
        elif line.startswith("%f") and not seen_f:
            seen_f = True
            for name, sl in (("base_pos", slice(3, 13)), ("base_clk", slice(14, 26))):
                try:
                    value = float(line[sl])
                except ValueError:
                    value = 0.0
                setattr(header, name, value or None)
        elif line.startswith("/*"):
            header.comments.append(line[3:].rstrip())
    if not plus_lines:
        raise SP3Error(path, f"no satellite list ('+' lines) after line 1 (line {first_lineno})")
    ids = [code for ln in plus_lines for code in _ids(ln) if code.strip() not in _PLACEHOLDERS]
    header.satellites = [sp3_sv(code) for code in (ids[:header.n_sv] if header.n_sv else ids)]
    exps = [code for ln in acc_lines for code in _ids(ln)]
    for sv, code in zip(header.satellites, exps):
        try:
            header.accuracy[sv] = int(code)
        except ValueError:
            header.accuracy[sv] = 0
    if header_only:
        return header, None, None
    if epoch_line is None:
        raise SP3Error(path, "no epoch line ('*') after the header: the file holds no orbit records")
    return header, epoch_line, lines


def read_sp3_header(path) -> SP3Header:
    """The header of an SP3 file (lines before the first epoch line); :class:`SP3Error` as described above."""
    return _read(path, header_only=True)[0]


def read_sp3_file(path) -> SP3File:
    """Read an SP3 file (plain or compressed): header and all records (see :class:`SP3File`)."""
    path = os.fspath(path)
    header, (lineno, line), lines = _read(path, header_only=False)
    expected = header.satellites
    n_hdr = len(expected)
    order = {sv: k for k, sv in enumerate(expected)}

    times: List[int] = []
    svs: List[str] = []
    pos: List[tuple] = []
    sdev: List[tuple] = []
    flags: List[tuple] = []
    vel: Dict[int, tuple] = {}
    ep: Dict[int, list] = {}
    ev: Dict[int, list] = {}

    state = {"epoch_lineno": None, "epoch_text": None, "svs": [], "last_p": None, "last_v": None, "prev_ns": None}

    def check_epoch(cut: bool = False):
        got = state["svs"]
        if state["epoch_lineno"] is None or got == expected:
            return
        missing = [sv for sv in expected if sv not in got]
        extra = [sv for sv in got if sv not in order]
        dup = sorted({sv for sv in got if got.count(sv) > 1})
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"not in the header {extra}")
        if dup:
            parts.append(f"repeated {dup}")
        raise SP3Error(path, f"epoch {state['epoch_text'].strip()!r} lists other satellite records than the header "
                             f"('+' lines, {n_hdr} satellites): {'; '.join(parts) or 'another order'}; every epoch "
                             f"must hold one P record per header satellite in header order "
                             f"{'(the file ends in this epoch without an EOF line: a cut file?)' if cut else '(a cut or merged file?)'}",
                       state["epoch_lineno"])

    def new_epoch(lineno: int, line: str):
        check_epoch()
        try:
            ns = _epoch_ns(line)
        except (ValueError, IndexError) as exc:
            raise SP3Error(path, f"epoch line does not parse ({exc})", lineno, line) from None
        if state["prev_ns"] is not None and ns <= state["prev_ns"]:
            raise SP3Error(path, "epoch not after the previous epoch", lineno, line)
        state.update(epoch_lineno=lineno, epoch_text=line, svs=[], last_p=None, last_v=None, prev_ns=ns)

    new_epoch(lineno, line)
    eof = False
    for lineno, line in lines:
        head = line[:1]
        if head == "P":
            sv = sp3_sv(line[1:4])
            used = _cut_record(line)
            if used is not None:
                where = ("position fields (x, y, z F14.6 end in column 46)" if used < 46 else
                         "clock field (F14.6 ends in column 60)")
                raise SP3Error(path, f"the record of epoch {state['epoch_text'].strip()!r} ends in column {used}, "
                                     f"inside its {where}, with a number cut short: a cut line (a cut file?)",
                               lineno, line)
            if not line[46:60].strip():
                raise SP3Error(path, f"blank clock field in the record of epoch {state['epoch_text'].strip()!r} "
                                     f"(SP3 writes 999999.999999 for a missing clock)", lineno, line)
            try:
                rec = (_float(line[4:18]), _float(line[18:32]), _float(line[32:46]), _float(line[46:60]))
                sd = (_opt_int(line[61:63]), _opt_int(line[64:66]), _opt_int(line[67:69]), _opt_int(line[70:73]))
            except ValueError:
                raise SP3Error(path, "position/clock record does not parse", lineno, line) from None
            row = len(times)
            times.append(state["prev_ns"])
            svs.append(sv)
            pos.append(rec)
            sdev.append(sd)
            flags.append((line[74:75] == "E", line[75:76] == "P", line[78:79] == "M", line[79:80] == "P"))
            state["svs"].append(sv)
            state["last_p"], state["last_v"] = (sv, row), None
        elif head == "*":
            new_epoch(lineno, line)
        elif line.startswith("EP"):
            if state["last_p"] is None:
                raise SP3Error(path, "EP record without a P record before it in its epoch", lineno, line)
            try:
                ep[state["last_p"][1]] = _ep_values(line)
            except ValueError:
                raise SP3Error(path, "EP record does not parse", lineno, line) from None
        elif line.startswith("EV"):
            if state["last_v"] is None:
                raise SP3Error(path, "EV record without a V record before it in its epoch", lineno, line)
            try:
                ev[state["last_v"][1]] = _ep_values(line)
            except ValueError:
                raise SP3Error(path, "EV record does not parse", lineno, line) from None
        elif head == "V":
            sv = sp3_sv(line[1:4])
            last = state["last_p"]
            if last is None or last[0] != sv or last[1] in vel:
                raise SP3Error(path, f"V record of {sv} does not follow the P record of {sv}", lineno, line)
            used = _cut_record(line)
            if used is not None:
                raise SP3Error(path, f"the velocity record ends in column {used}, inside its fields (F14.6 ending in "
                                     f"columns 18, 32, 46 and 60), with a number cut short: a cut line (a cut file?)",
                               lineno, line)
            if not line[46:60].strip():
                raise SP3Error(path, "blank clock-rate field in the velocity record (SP3 writes 999999.999999 "
                                     "for a missing rate)", lineno, line)
            try:
                vel[last[1]] = (_float(line[4:18]), _float(line[18:32]), _float(line[32:46]), _float(line[46:60]))
            except ValueError:
                raise SP3Error(path, "velocity record does not parse", lineno, line) from None
            state["last_v"] = (sv, last[1])
        elif line.startswith("EOF"):
            eof = True
            break
        elif line.startswith("/*") or not line.strip() or line.strip() == "\x1a":
            continue
        else:
            raise SP3Error(path, "unknown record in the data part (expected '*', 'P', 'EP', 'V', 'EV' or 'EOF')",
                           lineno, line)
    check_epoch(cut=not eof)
    notes: List[str] = []
    n_epochs = len(set(times)) if times else 0
    if not eof and header.n_epochs and n_epochs < header.n_epochs:
        notes.append(f"SP3 file {path}: the file ends after {n_epochs} of the {header.n_epochs} epochs its line 1 "
                     "declares, without an EOF line (a cut file?); the epochs read are used")
        warnings.warn(notes[-1], SP3Warning, stacklevel=2)

    n = len(times)
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(np.asarray(times, dtype="datetime64[ns]")), pd.Index(svs, dtype=object)],
        names=["time", "sv"])
    pos_a = np.asarray(pos, dtype=float).reshape(n, 4)
    vel_a = np.full((n, 4), np.nan)
    if vel:
        rows = np.fromiter(vel.keys(), dtype=np.int64, count=len(vel))
        vel_a[rows] = np.asarray(list(vel.values()), dtype=float)
    sd_a = np.asarray(sdev, dtype=float).reshape(n, 4)
    fl_a = np.asarray(flags, dtype=bool).reshape(n, 4)
    data = {"x": pos_a[:, 0], "y": pos_a[:, 1], "z": pos_a[:, 2], "clk": pos_a[:, 3],
            "vx": vel_a[:, 0], "vy": vel_a[:, 1], "vz": vel_a[:, 2], "dclk": vel_a[:, 3],
            "sx": sd_a[:, 0], "sy": sd_a[:, 1], "sz": sd_a[:, 2], "sclk": sd_a[:, 3],
            "clk_event": fl_a[:, 0], "clk_pred": fl_a[:, 1], "maneuver": fl_a[:, 2], "orb_pred": fl_a[:, 3]}
    for prefix, table in (("ep_", ep), ("ev_", ev)):
        if table:
            arr = np.full((n, len(_EP_NAMES)), np.nan)
            rows = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
            arr[rows] = np.asarray(list(table.values()), dtype=float)
            for k, name in enumerate(_EP_NAMES):
                data[prefix + name] = arr[:, k]
    records = pd.DataFrame(data, index=index)
    return SP3File(path=path, header=header, records=records, velocity=bool(vel), eof=eof, notes=notes)
