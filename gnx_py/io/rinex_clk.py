"""RINEX clock files: 30 s satellite clocks for the sessions and the slip layer (``cycle_slips``).

Format: RINEX clock 3.00-3.04 (IGS): header ``ANALYSIS CENTER`` (columns 1-3 the centre), ``TIME SYSTEM ID``
(columns 4-6; blank or missing means GPS), data records ``AS <sv> yyyy mm dd hh mm ss.ssssss n bias [sigma]``.
Only ``AS`` (satellite) records are read; ``AR`` (receiver) and other records are skipped. A record whose bias is
not a complete E19.12 value (no exponent, or a cut one: the last line of a cut file) is skipped and counted with
the unreadable records, so a cut file never gives a shorter number. Files are opened
through :mod:`gnx_py.compression`, so plain, gzip, Unix compress (.Z) and bzip2 files all read (today a .Z file
reads as 0 satellites without an error, cycle_slips.py:1476-1483). Written from the format description; no GPL
code. Interpolation conventions after RTKLIB-ex (BSD-2) ``pephclk``, read for the convention only.

Interface (stable; additions only with defaults)
------------------------------------------------
* :func:`read_rinex_clk` returns a :class:`ClkFile`; a time system other than GPS raises :class:`RinexClkError`
  (a ``ValueError``) naming the file and the system; a file without ``AS`` records gives an empty ``series`` and
  a note.
* :func:`usable_clk_paths` keeps the semantics of ``cycle_slips.usable_clk_paths`` (cycle_slips.py:1485-1505):
  ``(None, None)`` without clock files; ``(None, reason)`` without an SP3 file or when the analysis centre of a
  clock file (first three characters of its name, or its ANALYSIS CENTER header line) is not the centre of every
  SP3 file; ``(paths, None)`` otherwise.
* :func:`apply_clk_clocks` replaces the SP3 clock of an emission-interpolated satellite frame by the CLK clock at
  the emission time:

  - ``clk_raw`` becomes the CLK value at ``time - t_em`` and ``clk = clk_raw + clk_rel``;
  - each file is its own series: no interpolation from one file to the next, 1 s edge tolerance
    (:data:`CLK_EDGE_TOL_NS`) at a file's first and last node;
  - an epoch on a node takes the node; otherwise linear interpolation between two nodes at most
    :data:`CLK_MAX_GAP_STEPS` x the file's interval apart (no bridging of gaps, e.g. the C48 gap 18:22:30-18:26:30
    of 2024-035);
  - a row that no file covers keeps its SP3 clock;
  - a ``clk_src`` column says 'clk' or 'sp3' per row, and ``attrs['clk_cover']`` holds the share of the rows
    that got a CLK clock;
  - a bool ``clk_gap`` column marks the rows inside a gap of a file: between two
    nodes of the satellite in that file more than :data:`CLK_MAX_GAP_STEPS` intervals apart, and covered by no
    other file. ``gap_policy='drop'`` (``Config.clk_gap_policy``, the default of the sessions) leaves these rows
    out, as an ordinary data gap (RTKLIB ``pephclk``, preceph.c: "prec clock outage", the satellite is not used at
    that epoch, no SP3 fallback once clock files are loaded); ``'sp3'`` keeps them with the SP3 clock (before v1.0.0). Rows outside a file's span (another day, before the first or after the last node of the satellite)
    keep the SP3 clock under both policies;
  - one message gives the share of rows that fell back and their satellites: logging WARNING when the covered
    share is below :data:`CLK_COVER_WARN`, INFO otherwise (as the slip layer does today); dropped gap rows count
    as not covered.

Public API::

    read_rinex_clk(path) -> ClkFile
    usable_clk_paths(clk_path, sp3_path) -> (paths | None, reason | None)
    apply_clk_clocks(frame, files, gap_policy='sp3') -> DataFrame
"""
from __future__ import annotations

import io
import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .compression import read_bytes

__all__ = [
    "CLK_COVER_WARN", "CLK_EDGE_TOL_NS", "CLK_GAP_POLICIES", "CLK_MAX_GAP_STEPS", "ClkFile", "RinexClkError",
    "apply_clk_clocks", "read_rinex_clk", "usable_clk_paths",
]

#: Tolerance at the first and last node of a clock file [ns].
CLK_EDGE_TOL_NS = 1_000_000_000

#: Two clock nodes are interpolated between only when at most this many file intervals apart.
CLK_MAX_GAP_STEPS = 1.5

#: What becomes of the rows inside a clock-file gap (``apply_clk_clocks``; ``configuration.CLK_GAP_POLICIES``).
CLK_GAP_POLICIES = ("drop", "sp3")

#: Covered share of the rows below which the fallback to SP3 clocks is reported as a WARNING (shared with
#: ``cycle_slips.CLK_COVER_WARN``, cycle_slips.py:955).
CLK_COVER_WARN = 0.9

log = logging.getLogger(__name__)

#: Number of parsed clock files kept (a session uses a few days at most).
_CACHE_SIZE = 4
_CACHE: "OrderedDict[tuple, ClkFile]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


class RinexClkError(ValueError):
    """A clock file that cannot be used; the message names the file and the problem."""


@dataclass
class ClkFile:
    """Satellite clocks of one RINEX clock file.

    ``ac``: the ANALYSIS CENTER of the header (first three characters, upper case), None when absent.
    ``time_system``: 'GPS' (the only one accepted). ``interval_s``: the median spacing of the satellite records
    [s], None with fewer than two epochs. ``series``: ``{sv: (t_ns, clk_s)}`` with ``t_ns`` int64 nanoseconds
    since 1970 (GPST), strictly increasing, and ``clk_s`` float64 clock biases [s]. ``path``: the file.
    ``notes``: readable remarks (skipped records, a missing TIME SYSTEM ID taken as GPS, no ``AS`` record)."""

    ac: Optional[str]
    time_system: str
    interval_s: Optional[float]
    series: dict
    path: Optional[str] = None
    notes: list = field(default_factory=list)


def _sv_id(token: str) -> Optional[str]:
    """'G01' -> 'G01', 'G1' -> 'G01'; None when ``token`` is not a satellite number."""
    if len(token) < 2 or not token[0].isalpha() or not token[1:].isdigit():
        return None
    return f"{token[0].upper()}{int(token[1:]):02d}"


#: A complete E19.12 clock value (the data values of a RINEX clock record): a mantissa and a two- or three-digit
#: exponent; a value cut by the end of a cut file (no exponent, or a cut one) is not read as a shorter number.
_E_VALUE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)[EeDd][+-]?\d{2,3}")


def _epoch_ns(fields) -> int:
    """GPST nanoseconds since 1970 of ``(year, month, day, hour, minute, second)`` tokens."""
    y, mo, d, h, mi = (int(f) for f in fields[:5])
    base = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}", "ns").astype(np.int64)
    return int(base) + int(round(float(fields[5]) * 1e9))


def _parse(path) -> ClkFile:
    name = os.fspath(path)
    # the decompressed bytes once, read line by line (a 36 MB file is not copied into a text buffer)
    stream = io.BytesIO(read_bytes(path, what="RINEX clock file"))
    version, file_type, ac, tsys, ended = None, None, None, None, False
    notes: list = []
    for raw in stream:
        ln = raw.decode("ascii", "replace").rstrip("\r\n")
        # labels start in column 61 (RINEX clock 2.0x-3.02) or 66 (3.04, 9-character station names)
        label = ln[60:].strip()
        if label.endswith("RINEX VERSION / TYPE"):
            try:
                version = float(ln[:9])
            except ValueError:
                version = None
            # column 21 (RINEX clock 2.0x-3.02, F9.2,11X,A1) or 22 (3.04, F4.2,17X,A1)
            file_type = "C" if "C" in ln[20:22] else ln[20:22].strip()
        elif label.endswith("ANALYSIS CENTER"):
            ac = ln[:3].strip().upper() or None
        elif label.endswith("TIME SYSTEM ID"):
            tsys = ln[:60].strip().upper() or None
        elif label.endswith("END OF HEADER"):
            ended = True
            break
    if version is None or file_type != "C":
        raise RinexClkError(f"{name}: not a RINEX clock file (no RINEX VERSION / TYPE record of type 'C')")
    if not ended:
        raise RinexClkError(f"{name}: RINEX clock header without END OF HEADER (truncated file?)")
    if tsys is None:
        tsys = "GPS"
        notes.append("no TIME SYSTEM ID in the header: GPS time taken")
    if tsys != "GPS":
        raise RinexClkError(f"{name}: clock epochs in time system {tsys} (TIME SYSTEM ID); only GPS-time clock "
                            f"files can be used with GNX's GPST observations")

    epochs: dict = {}
    svs, tns, vals = [], [], []
    skipped = 0
    for raw in stream:
        if not raw.startswith(b"AS "):
            continue
        tok = raw[3:].decode("ascii", "replace").split()
        sv = _sv_id(tok[0]) if tok else None
        rest = tok[1:]
        if sv is None and len(tok) > 1 and len(tok[0]) == 1 and tok[1].isdigit():
            sv, rest = _sv_id(tok[0] + tok[1]), tok[2:]          # 'G 1' of old writers
        try:
            if sv is None or len(rest) < 8 or int(rest[6]) < 1:
                raise ValueError
            key = tuple(rest[:6])
            t = epochs.get(key)
            if t is None:
                t = epochs[key] = _epoch_ns(rest)
            if not _E_VALUE.fullmatch(rest[7]):
                raise ValueError          # not a complete E19.12 value: a cut line
            value = float(rest[7].replace("D", "E").replace("d", "e"))
        except (ValueError, OverflowError):
            skipped += 1
            continue
        svs.append(sv)
        tns.append(t)
        vals.append(value)
    if skipped:
        notes.append(f"{skipped} unreadable AS record(s) skipped")

    series: dict = {}
    interval = None
    if svs:
        sv_a = np.asarray(svs)
        t_a = np.asarray(tns, dtype=np.int64)
        v_a = np.asarray(vals, dtype=np.float64)
        order = np.lexsort((t_a, sv_a))
        sv_a, t_a, v_a = sv_a[order], t_a[order], v_a[order]
        cuts = np.flatnonzero(sv_a[1:] != sv_a[:-1]) + 1
        dup = 0
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, sv_a.size]):
            t, v = t_a[lo:hi], v_a[lo:hi]
            keep = np.r_[True, np.diff(t) > 0]
            dup += int((~keep).sum())
            series[str(sv_a[lo])] = (t[keep], v[keep])
        if dup:
            notes.append(f"{dup} repeated AS record(s) of one satellite and epoch: the first one kept")
        u = np.unique(t_a)
        if u.size > 1:
            interval = float(np.median(np.diff(u))) / 1e9
    else:
        notes.append("no satellite clock (AS) record")
    return ClkFile(ac=ac, time_system=tsys, interval_s=interval, series=series, path=name, notes=notes)


def read_rinex_clk(path) -> ClkFile:
    """Satellite clocks of the RINEX clock file ``path`` (any compression :mod:`gnx_py.compression` reads).

    Cached per file (path, size, modification time; a few entries); the arrays of a cached file are shared and
    must not be modified."""
    st = os.stat(path)
    key = (os.path.abspath(os.fspath(path)), st.st_size, st.st_mtime_ns)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    parsed = _parse(path)
    with _CACHE_LOCK:
        _CACHE[key] = parsed
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return parsed


def _name_centre(path) -> str:
    """Analysis centre of a product file name: its first three characters, upper case."""
    return os.path.basename(os.fspath(path))[:3].upper()


def usable_clk_paths(clk_path, sp3_path) -> tuple:
    """``(paths, reason)``: the clock files to use with the orbits of ``sp3_path``, or ``(None, why not)``;
    ``(None, None)`` when ``clk_path`` names no file. Semantics in the module docstring."""
    from ..core.validation import iter_paths

    clk = list(iter_paths(clk_path))
    if not clk:
        return None, None
    sp3 = list(iter_paths(sp3_path))
    if not sp3:
        return None, "no SP3 file (sp3_path) to compare its analysis centre with"
    centres = {_name_centre(p) for p in sp3}
    for p in clk:
        mine = {_name_centre(p)}
        if not centres <= mine:
            # the ANALYSIS CENTER line may name the centre the file name does not (COD files say 'COM')
            try:
                mine.add(read_rinex_clk(p).ac)
            except (OSError, ValueError) as exc:
                return None, f"{os.fspath(p)} cannot be read ({exc})"
        if not centres <= mine:
            return None, (f"analysis centre of {_name_centre(p)!r} differs from the SP3 "
                          f"({', '.join(sorted(centres))}); only clocks of the SP3's own centre are used")
    return clk, None


def _interp_file(clk: ClkFile, sv: str, tq: np.ndarray, gap: Optional[np.ndarray] = None) -> np.ndarray:
    """Clock of ``sv`` at ``tq`` [ns] from one file: the node on a node, else linear between two nodes at most
    CLK_MAX_GAP_STEPS intervals apart, within CLK_EDGE_TOL_NS of the file's first and last node; NaN elsewhere.
    ``gap`` (bool, as ``tq``): set True where ``tq`` lies strictly between two nodes of ``sv`` more than
    CLK_MAX_GAP_STEPS intervals apart (a gap of the file)."""
    out = np.full(tq.size, np.nan)
    s = clk.series.get(sv)
    if s is None or tq.size == 0:
        return out
    t, v = s
    if t.size < 2:
        if t.size == 1:
            on = tq == t[0]
            out[on] = v[0]
        return out
    step = clk.interval_s * 1e9 if clk.interval_s else float(np.median(np.diff(t)))
    inr = (tq >= t[0] - CLK_EDGE_TOL_NS) & (tq <= t[-1] + CLK_EDGE_TOL_NS)
    if not inr.any():
        return out
    q = tq[inr]
    k = np.searchsorted(t, q)
    kk = np.minimum(k, t.size - 1)
    on = t[kk] == q
    i1 = np.clip(k, 1, t.size - 1)
    i0 = i1 - 1
    span = (t[i1] - t[i0]).astype(np.float64)
    w = (q - t[i0]).astype(np.float64) / span
    val = v[i0] + w * (v[i1] - v[i0])
    ok = (span <= CLK_MAX_GAP_STEPS * step) & np.isfinite(val)
    val = np.where(ok, val, np.nan)
    val[on] = v[kk[on]]
    out[inr] = val
    if gap is not None:
        inside = (q > t[0]) & (q < t[-1]) & ~on & (span > CLK_MAX_GAP_STEPS * step)
        gap[np.flatnonzero(inr)[inside]] = True
    return out


def apply_clk_clocks(frame: pd.DataFrame, files: Iterable[ClkFile], gap_policy: str = "sp3") -> pd.DataFrame:
    """Copy of the emission-interpolated satellite frame ``frame`` with the CLK clocks of ``files`` (rules in the
    module docstring).

    ``frame``: the output of ``coordinates.lagrange_emission_interp`` (index levels 'sv' and 'time' in either
    order; columns ``t_em`` [s], ``clk_raw``, ``clk_rel`` and ``clk`` [s]). The index and row order are kept;
    ``clk_src`` and ``clk_gap`` are added. ``gap_policy``: 'sp3' keeps the rows inside a clock-file gap with the
    SP3 clock, 'drop' leaves them out (the other rows keep their order). An empty frame or no file returns the
    frame unchanged (with ``clk_src`` 'sp3', ``clk_gap`` False)."""
    if gap_policy not in CLK_GAP_POLICIES:
        raise ValueError(f"gap_policy must be one of {CLK_GAP_POLICIES}, got {gap_policy!r}")
    files = [f for f in (files or ()) if f is not None]
    out = frame.copy()
    n = len(out)
    out["clk_src"] = "sp3"
    out["clk_gap"] = False
    if n == 0 or not files:
        out.attrs["clk_cover"] = None if n == 0 else 0.0
        return out
    t_ns = np.asarray(out.index.get_level_values("time"), dtype="datetime64[ns]").astype(np.int64)
    sv = pd.Index(out.index.get_level_values("sv").astype(str)).str.split("_").str[0].to_numpy()
    t_em = out["t_em"].to_numpy(dtype=np.float64)
    valid = np.isfinite(t_em)
    tq = t_ns - np.rint(np.where(valid, t_em, 0.0) * 1e9).astype(np.int64)
    new = np.full(n, np.nan)
    in_gap = np.zeros(n, dtype=bool)
    rows = np.flatnonzero(valid)
    for s, pos in pd.Series(rows).groupby(sv[rows]).indices.items():
        idx = rows[pos]
        for f in files:
            need = idx[np.isnan(new[idx])]
            if need.size == 0:
                break
            gap = np.zeros(need.size, dtype=bool)
            new[need] = _interp_file(f, str(s), tq[need], gap)
            in_gap[need] |= gap
    use = np.isfinite(new)
    in_gap &= ~use
    raw = out["clk_raw"].to_numpy(dtype=np.float64, copy=True)
    clk = out["clk"].to_numpy(dtype=np.float64, copy=True)
    if "clk_rel" in out.columns:
        clk[use] = new[use] + out["clk_rel"].to_numpy(dtype=np.float64)[use]
    else:
        clk[use] = clk[use] - raw[use] + new[use]
    raw[use] = new[use]
    out["clk_raw"] = raw
    out["clk"] = clk
    out["clk_src"] = np.where(use, "clk", "sp3")
    out["clk_gap"] = in_gap
    n_valid = int(valid.sum())
    cover = float(use.sum()) / n_valid if n_valid else 0.0
    out.attrs["clk_cover"] = cover
    names = ", ".join(os.path.basename(f.path) if f.path else "?" for f in files)
    drop = gap_policy == "drop" and bool(in_gap.any())
    fallback = valid & ~use & ~in_gap if drop else valid & ~use

    def listed(rows) -> str:
        sats = sorted(set(sv[rows].tolist()))
        return ", ".join(sats[:20]) + (f" and {len(sats) - 20} more" if len(sats) > 20 else "")

    if fallback.any() or drop:
        parts = []
        if fallback.any():
            parts.append(f"{int(fallback.sum())} ({100.0 * fallback.sum() / max(n_valid, 1):.1f} %) keep the SP3 clock "
                         f"(no clock node within the rules: another day, a gap of more than {CLK_MAX_GAP_STEPS:g} "
                         f"intervals): {listed(fallback)}")
        if drop:
            parts.append(f"{int(in_gap.sum())} inside a gap of the clock files are left out (clk_gap_policy='drop', "
                         f"as an ordinary data gap): {listed(in_gap)}")
        log.log(logging.WARNING if cover < CLK_COVER_WARN else logging.INFO,
                "clk_path (%s) gives the clock of %.1f %% of %d satellite-epochs; %s", names, 100.0 * cover, n_valid,
                "; ".join(parts))
    else:
        log.info("clk_path (%s) gives the clock of all %d satellite-epochs", names, n_valid)
    if drop:
        out = out[~in_gap]
    return out
