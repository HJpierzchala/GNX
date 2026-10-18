"""Ocean tide loading (OTL): BLQ reader and the RTKLIB ``tide_oload`` displacement.

BLQ files come from the Onsala ocean loading service (http://holt.oso.chalmers.se/loading/, Scherneck & Bos).
Each station block has a name line (two spaces and the station name), ``$$`` comment lines and six rows of 11
values, columns M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa:

* rows 1-3: amplitudes [m] of the radial (up), tangential west and tangential south displacement;
* rows 4-6: Greenwich phase lags [deg] of the same three components.

The displacement is the RTKLIB 2.4.2p13 ``tide_oload`` (ppp.c) model: the 11 constituents with the astronomical
arguments of the IERS ``ARG.f`` routine (RTKLIB's version: Doodson angles h0, s0, p0 at 0h of the day, angular speed
times the seconds of day), no nodal modulation and no minor constituents (HARDISP sums 342). The argument time is
UT: UTC, or UT1 when Earth rotation parameters are given (``tidedisp``: tut = tutc + UT1-UTC).

The ENU displacement is rotated to ECEF with the geocentric latitude, as ``tidedisp`` does (the BLQ radial
direction is the spherical one); the difference to the geodetic frame is below 0.04 mm for 1 cm of radial loading.

Model error of the 11-constituent truncation against HARDISP (IERS 2010 7.1.2): the missing nodal factors f, u of
the 11 constituents and the missing minor tides (nu2, mu2, L2, T2, 2N2, M1, J1, OO1, rho1, sigma1, ...; HARDISP
interpolates their admittance) are each of the order of 1 mm radial for a station with ~1 cm of M2 loading.

:func:`merge_blq` joins several BLQ files (e.g. one per station) into one file for ``ocean_loading_path``.
"""
from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from ..io.compression import head_lines, open_text

_LOG = logging.getLogger(__name__)

#: constituent order of the BLQ columns and of RTKLIB's odisp array
OTL_CONSTITUENTS: Tuple[str, ...] = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "Mf", "Mm", "Ssa")

# RTKLIB tide_oload args[][5]: angular speed [rad/s], multipliers of h0, s0, p0, and of 2 pi. Ported from RTKLIB
# 2.4.2p13 src/ppp.c (Copyright (C) 2010-2018 by T.TAKASU, All rights reserved; BSD 2-clause licence), together
# with the argument arithmetic of otl_arguments: see THIRD_PARTY_NOTICES, section 3.
_OTL_ARGS = np.array([
    [1.40519E-4, 2.0, -2.0, 0.0, 0.00],   # M2
    [1.45444E-4, 0.0, 0.0, 0.0, 0.00],    # S2
    [1.37880E-4, 2.0, -3.0, 1.0, 0.00],   # N2
    [1.45842E-4, 2.0, 0.0, 0.0, 0.00],    # K2
    [0.72921E-4, 1.0, 0.0, 0.0, 0.25],    # K1
    [0.67598E-4, 1.0, -2.0, 0.0, -0.25],  # O1
    [0.72523E-4, -1.0, 0.0, 0.0, -0.25],  # P1
    [0.64959E-4, 1.0, -3.0, 1.0, -0.25],  # Q1
    [0.53234E-5, 0.0, 2.0, 0.0, 0.00],    # Mf
    [0.26392E-5, 0.0, 1.0, -1.0, 0.00],   # Mm
    [0.03982E-5, 2.0, 0.0, 0.0, 0.00],    # Ssa
])
_PI = 3.1415926535897932  # RTKLIB PI
_D2R = _PI / 180.0
_EPOCH_1975 = np.datetime64("1975-01-01", "D")

PathLike = Union[str, os.PathLike]


# ============================================================================ BLQ
@dataclass(frozen=True)
class BLQStation:
    """Ocean loading coefficients of one station of a BLQ file.

    ``amplitude`` and ``phase`` are (3, 11) arrays: rows radial (up), west, south; columns
    :data:`OTL_CONSTITUENTS`; amplitudes in metres, phases (Greenwich lag) in degrees.
    """

    name: str
    amplitude: np.ndarray
    phase: np.ndarray
    label: str = ""
    lon_lat_height: Optional[Tuple[float, float, Optional[float]]] = None
    comments: Tuple[str, ...] = ()
    line: int = 0

    def odisp(self) -> np.ndarray:
        """The RTKLIB ``odisp`` layout: odisp[j + 6 i] (j = 0..2 amplitudes radial/west/south, 3..5 phases)."""
        out = np.empty(66)
        for i in range(11):
            out[6 * i:6 * i + 3] = self.amplitude[:, i]
            out[6 * i + 3:6 * i + 6] = self.phase[:, i]
        return out


@dataclass(frozen=True)
class BLQFile:
    """Contents of a BLQ file: the stations in file order, the ocean tide model and the CMC flag of the header.

    ``model`` is the ocean tide model the header names (e.g. ``'FES2014b'``; None when not stated) and ``cmc``
    the centre-of-mass correction flag (``$$ CMC: YES``: the displacement includes the geocentre motion caused by
    the ocean tide, as GNSS orbits in the centre-of-mass frame need; None when not stated).
    """

    path: str
    stations: Dict[str, BLQStation] = field(default_factory=dict)
    model: Optional[str] = None
    cmc: Optional[bool] = None
    header: Tuple[str, ...] = ()

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.stations)

    def find(self, name: Optional[str]) -> Optional[BLQStation]:
        """The station block for ``name`` (None when absent).

        Matching is case-insensitive on the 4-character site code; a 9-character id (``BRUX00BEL``) or a name
        with the DOMES number matches the 4-character BLQ name and the other way round (``biases.match_station``).
        An exact name comes first; two different blocks of the same site raise ``ValueError``.
        """
        from .biases import match_station

        if name is None or not str(name).strip():
            return None
        hits = match_station(name, list(self.stations))
        if not hits:
            return None
        key = str(name).strip().upper().replace(" ", "")
        exact = [h for h in hits if h.upper().replace(" ", "") == key]
        if exact:
            return self.stations[exact[0]]
        if len(hits) > 1:
            raise ValueError(f"ocean_loading_path {self.path}: station {name!r} matches several BLQ blocks "
                             f"({', '.join(hits)}); set station_name to one of them")
        return self.stations[hits[0]]


_MODEL_RX = [
    re.compile(r"ocean\s+tide\s+model\s*[:=]\s*([A-Za-z][\w.\-+]*)", re.I),
    re.compile(r"^\$\$\s*([A-Za-z][\w.\-+]*?)(?:_PP)?\s+ID\s*:", re.I),
    re.compile(r"^\$\$\s*Complete\s+([A-Za-z][\w.\-+]*)", re.I),
]
_CMC_RX = re.compile(r"\bCMC\s*[:=]?\s*(YES|NO)\b", re.I)
_LONLAT_RX = re.compile(r"lon/lat\s*:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)(?:\s+([-+0-9.eE]+))?", re.I)
_NUMBER_RX = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


def _open_text(path: Path):
    # gzip, Unix compress (.Z), bzip2 and zip by content, in memory (bug-fix stage 10: .Z gave a format error)
    return open_text(path, what="BLQ file", encoding="latin-1")


def _is_number(token: str) -> bool:
    return bool(_NUMBER_RX.match(token))


def _model_of(lines: Iterable[str]) -> Optional[str]:
    for rx in _MODEL_RX:
        for ln in lines:
            m = rx.search(ln)
            if m:
                return m.group(1).rstrip(",.;")
    return None


def _cmc_of(lines: Iterable[str]) -> Optional[bool]:
    for ln in lines:
        m = _CMC_RX.search(ln)
        if m:
            return m.group(1).upper() == "YES"
    return None


def parse_blq(text: str, path: str = "<string>") -> BLQFile:
    """Parse BLQ ``text`` (see :func:`read_blq`)."""
    lines = text.splitlines()
    header: list = []
    # header: the $$ lines up to '$$ END HEADER' (Onsala); a file without that line has no separate header
    end = next((i for i, ln in enumerate(lines) if re.match(r"^\$\$\s*END\s+HEADER", ln, re.I)), None)
    start = 0
    if end is not None:
        header = [ln.rstrip() for ln in lines[:end + 1]]
        start = end + 1

    def err(lineno: int, msg: str) -> ValueError:
        return ValueError(f"ocean_loading_path {path}: line {lineno}: {msg}")

    stations: Dict[str, BLQStation] = {}
    i = start
    n = len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()
        if not s or s.startswith("$$"):
            i += 1
            continue
        tokens = s.split()
        if len(tokens) > 1 and _is_number(tokens[0]):
            raise err(i + 1, f"a row of numbers outside a station block (no station name line before it): {s[:60]!r}")
        name_line = i + 1
        label = s
        name = tokens[0].upper()
        comments: list = []
        rows: list = []
        i += 1
        while i < n and len(rows) < 6:
            r = lines[i].strip()
            if not r:
                i += 1
                continue
            if r.startswith("$$"):
                comments.append(r)
                i += 1
                continue
            toks = r.split()
            if len(toks) == 1 or not _is_number(toks[0]):
                # a name line (a station code may be numeric, e.g. 7090): the block is truncated
                raise err(i + 1, f"station {name}: {len(rows)} of 6 coefficient rows before the next name line "
                                 f"{r[:40]!r} (3 amplitude rows and 3 phase rows of 11 values)")
            try:
                vals = [float(t.replace("D", "E").replace("d", "e")) for t in toks]
            except ValueError:
                raise err(i + 1, f"station {name}: non-numeric value in {r[:60]!r}") from None
            if len(vals) != 11:
                raise err(i + 1, f"station {name}: {len(vals)} values in a coefficient row, expected 11 "
                                 f"({' '.join(OTL_CONSTITUENTS)})")
            rows.append(vals)
            i += 1
        if len(rows) < 6:
            raise err(i, f"station {name}: file ends after {len(rows)} of 6 coefficient rows")
        arr = np.asarray(rows, dtype=float)
        amp, pha = arr[:3], arr[3:]
        if not np.all(np.isfinite(arr)):
            raise err(name_line, f"station {name}: non-finite coefficient")
        if np.any(amp < 0):
            raise err(name_line, f"station {name}: negative amplitude in rows 1-3 (amplitudes in metres)")
        if np.any(amp > 1.0):
            raise err(name_line, f"station {name}: amplitude above 1 m in rows 1-3; BLQ amplitudes are metres "
                                 f"(largest {amp.max():.4g})")
        if np.any(np.abs(pha) > 360.0):
            raise err(name_line, f"station {name}: phase outside [-360, 360] deg in rows 4-6")
        if name in stations:
            raise err(name_line, f"station {name} appears twice (first block at line {stations[name].line})")
        lonlat = None
        for c in comments:
            m = _LONLAT_RX.search(c)
            if m:
                lonlat = (float(m.group(1)), float(m.group(2)), float(m.group(3)) if m.group(3) else None)
                break
        stations[name] = BLQStation(name=name, amplitude=amp.copy(), phase=pha.copy(), label=label,
                                    lon_lat_height=lonlat, comments=tuple(comments), line=name_line)
    if not stations:
        raise ValueError(f"ocean_loading_path {path}: no station block found (a BLQ file has, per station, a name "
                         f"line and 6 rows of 11 coefficients)")
    station_comments = [c for st in stations.values() for c in st.comments]
    model = _model_of(header) or _model_of(station_comments)
    cmc = _cmc_of(header)
    if cmc is None:
        cmc = _cmc_of(station_comments)
    return BLQFile(path=path, stations=stations, model=model, cmc=cmc, header=tuple(header))


@functools.lru_cache(maxsize=16)
def _read_blq_cached(path: str, mtime_ns: int, size: int) -> BLQFile:
    p = Path(path)
    with _open_text(p) as fh:
        text = fh.read()
    blq = parse_blq(text, str(p))
    _LOG.info("BLQ %s: %d stations, model %s, CMC %s", p.name, len(blq.stations), blq.model,
              {True: "yes", False: "no", None: "not stated"}[blq.cmc])
    return blq


def read_blq(path: PathLike) -> BLQFile:
    """Read an Onsala BLQ file (plain, ``.gz``, ``.Z``, bzip2 or zip); several stations per file.

    Raises ``FileNotFoundError`` for a missing file and ``ValueError`` naming the file and line for a malformed
    one (a coefficient row without 11 values, a block with fewer than 6 rows, a non-numeric value, a negative or
    implausible amplitude, a phase outside +-360 deg, a station given twice, no station at all).
    The result is cached per (path, modification time, size).
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"ocean_loading_path: file not found: {path}")
    st = p.stat()
    return _read_blq_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)


def _fmt_coeff(v: float) -> str:
    return f"{v:.10g}"


def format_blq(stations: Iterable[BLQStation], model: Optional[str] = None, cmc: Optional[bool] = None,
               header: Iterable[str] = ()) -> str:
    """BLQ text of ``stations`` that :func:`parse_blq` reads back exactly (coefficients with 10 significant
    digits); the header states ``model`` and ``cmc`` when given."""
    out = [ln if ln.startswith("$$") else f"$$ {ln}" for ln in header]
    if model:
        out.append(f"$$ Ocean tide model: {model}")
    if cmc is not None:
        out.append(f"$$ CMC: {'YES' if cmc else 'NO'}")
    out.append("$$ END HEADER")
    for st in stations:
        out.append(f"  {st.name}")
        out.extend(c if c.startswith("$$") else f"$$ {c}" for c in st.comments)
        for row in np.vstack([st.amplitude, st.phase]):
            out.append(" " + " ".join(_fmt_coeff(float(v)) for v in row))
    return "\n".join(out) + "\n"


def merge_blq(paths: Iterable[PathLike], out_path: PathLike) -> BLQFile:
    """Merge several BLQ files (plain or compressed, e.g. one file per station from the Onsala service) into one
    file at ``out_path`` and return it read back.

    A station in several files is kept once when its coefficients are identical and raises ``ValueError``
    otherwise; files of different ocean tide models or CMC flags raise ``ValueError`` (a model or flag stated in
    one file and absent in another is taken from the file that states it).
    """
    files = [read_blq(p) for p in paths]
    if not files:
        raise ValueError("merge_blq: no BLQ file given")
    models = {f.model for f in files if f.model}
    cmcs = {f.cmc for f in files if f.cmc is not None}
    if len(models) > 1:
        raise ValueError("merge_blq: the files name different ocean tide models: "
                         + ", ".join(f"{Path(f.path).name}={f.model}" for f in files))
    if len(cmcs) > 1:
        raise ValueError("merge_blq: the files have different CMC flags: "
                         + ", ".join(f"{Path(f.path).name}={f.cmc}" for f in files))
    merged: Dict[str, BLQStation] = {}
    origin: Dict[str, str] = {}
    for f in files:
        for name, st in f.stations.items():
            if name in merged:
                prev = merged[name]
                if not (np.array_equal(prev.amplitude, st.amplitude) and np.array_equal(prev.phase, st.phase)):
                    raise ValueError(f"merge_blq: station {name} has different coefficients in {origin[name]} and "
                                     f"{f.path}")
                continue
            merged[name] = st
            origin[name] = f.path
    header = [f"$$ merged by gnx_py.ocean_loading.merge_blq from: {', '.join(Path(f.path).name for f in files)}"]
    text = format_blq(merged.values(), model=next(iter(models), None), cmc=next(iter(cmcs), None), header=header)
    out = Path(out_path).expanduser()
    out.write_text(text, encoding="latin-1")
    return read_blq(out)


# ============================================================================ displacement
def _as_datetime64(times) -> np.ndarray:
    t = np.asarray(times)
    if t.dtype.kind != "M":
        import pandas as pd

        t = pd.to_datetime(np.atleast_1d(times)).to_numpy()
    return np.atleast_1d(t).astype("datetime64[ns]")


def otl_arguments(times_ut) -> np.ndarray:
    """Astronomical arguments [rad] of the 11 constituents at UT epochs, (N, 11), as RTKLIB ``tide_oload``.

    h0, s0, p0 at 0h UT of the day from days since 1975-01-01 (``ARG.f``), plus the angular speed times the
    seconds of day.
    """
    t = _as_datetime64(times_ut)
    day0 = t.astype("datetime64[D]")
    fday = (t - day0.astype("datetime64[ns]")).astype("int64") * 1e-9
    days = (day0 - _EPOCH_1975).astype("int64").astype(float) + 1.0
    tc = (27392.500528 + 1.000000035 * days) / 36525.0
    t2, t3 = tc * tc, tc * tc * tc
    h0 = (279.69668 + 36000.768930485 * tc + 3.03E-4 * t2) * _D2R
    s0 = (270.434358 + 481267.88314137 * tc - 0.001133 * t2 + 1.9E-6 * t3) * _D2R
    p0 = (334.329653 + 4069.0340329577 * tc - 0.010325 * t2 - 1.2E-5 * t3) * _D2R
    a = np.column_stack([fday, h0, s0, p0, np.full_like(fday, 2.0 * _PI)])
    # same summation order as RTKLIB (j = 0..4)
    ang = np.zeros((len(t), 11))
    for j in range(5):
        ang += a[:, j:j + 1] * _OTL_ARGS[None, :, j]
    return ang


def ocean_loading_enu(station: BLQStation, times_ut) -> np.ndarray:
    """OTL displacement (east, north, up) [m] at UT epochs, (N, 3): RTKLIB ``tide_oload``."""
    ang = otl_arguments(times_ut)                                   # (N, 11)
    amp = np.asarray(station.amplitude, dtype=float)                # (3, 11) radial, west, south
    pha = np.asarray(station.phase, dtype=float) * _D2R
    dp = np.einsum("ji,nji->nj", amp, np.cos(ang[:, None, :] - pha[None, :, :]))  # (N, 3)
    return np.column_stack([-dp[:, 1], -dp[:, 2], dp[:, 0]])


def geocentric_enu_matrix(xyz) -> np.ndarray:
    """ECEF -> ENU rotation (rows east, north, up) at the geocentric latitude and longitude of ``xyz``
    (RTKLIB ``tidedisp``: pos = asin(z / |r|), atan2(y, x))."""
    r = np.asarray(xyz, dtype=float).reshape(3)
    lat = np.arcsin(r[2] / np.linalg.norm(r))
    lon = np.arctan2(r[1], r[0])
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def enu_to_ecef(xyz, denu) -> np.ndarray:
    """Rotate (N, 3) ENU displacements at ``xyz`` (geocentric frame) to ECEF."""
    return np.asarray(denu, dtype=float) @ geocentric_enu_matrix(xyz)


def ocean_loading_ecef(station: BLQStation, times_ut, xyz) -> np.ndarray:
    """OTL displacement in ECEF [m], (N, 3) (RTKLIB ``tidedisp`` with opt=2)."""
    return enu_to_ecef(xyz, ocean_loading_enu(station, times_ut))


# ============================================================================ station name
def rinex_marker_name(obs_path: Optional[PathLike]) -> str:
    """MARKER NAME of a RINEX observation header ('' when absent or unreadable; any compression of
    ``gnx_py.compression``, the Hatanaka header is plain text)."""
    if not obs_path:
        return ""
    p = Path(obs_path)
    try:
        head = head_lines(p, "observation file", 200_000)
    except (OSError, ValueError):
        return ""
    for k, ln in enumerate(head):
        label = ln[60:].strip()
        if label == "MARKER NAME":
            return ln[:60].strip()
        if label == "END OF HEADER" or k > 400:
            break
    return ""


def station_names_for(config) -> Tuple[str, ...]:
    """Names to look the station up by: ``station_name``, else the RINEX MARKER NAME, else the first 4
    characters of the observation file name (RINEX 2 and 3 names start with the site code)."""
    names = []
    configured = getattr(config, "station_name", None)
    if configured is not None and str(configured).strip():
        return (str(configured).strip(),)
    obs = getattr(config, "obs_path", None)
    marker = rinex_marker_name(obs)
    if marker:
        names.append(marker)
    if obs:
        base = Path(obs).name
        if len(base) >= 4 and re.match(r"^[A-Za-z0-9]{4}", base):
            names.append(base[:4])
    return tuple(dict.fromkeys(names))


def missing_station_message(path, blq: BLQFile, names: Tuple[str, ...]) -> str:
    """The warning text for a station that is not in the BLQ file."""
    shown = ', '.join(blq.names[:12]) + (' ...' if len(blq.names) > 12 else '')
    who = ' / '.join(repr(n) for n in names) if names else 'unknown (no station_name, MARKER NAME or file name)'
    return (f"ocean_loading_path {path}: station {who} not in the BLQ file (stations: {shown}); "
            f"no ocean tide loading applied")


def resolve_station(config) -> Tuple[Optional[BLQStation], BLQFile, Tuple[str, ...]]:
    """(station block or None, BLQ file, names tried) for ``config.ocean_loading_path`` and the station of the
    session (:func:`station_names_for`). ``ValueError`` when a name matches several blocks (:meth:`BLQFile.find`)."""
    blq = read_blq(getattr(config, "ocean_loading_path"))
    names = station_names_for(config)
    for name in names:
        station = blq.find(name)
        if station is not None:
            return station, blq, names
    return None, blq, names


def warn_missing_station_once(config, path, blq: BLQFile, names: Tuple[str, ...], stacklevel: int = 3) -> None:
    """The missing-station UserWarning, once per run of the configuration (``gnx_py._notes``; the session
    clears the keys at the start of each run), BLQ file and station names: the session validation and the
    preprocessing of every system look the station up, only the first one warns."""
    from ..core import _notes

    key = ("otl_missing", str(Path(path).expanduser().resolve()), tuple(names))
    _notes.warn_once(config, key, missing_station_message(path, blq, names), UserWarning, stacklevel=stacklevel + 1)
