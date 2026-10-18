"""Checks shared by the SPP, PPP and TEC sessions before and between their processing steps.

* :func:`no_observations_error`: the readable "no observations" error, naming the systems, the observation file,
  ``time_limit`` and the time span of the file.
* :func:`load_sp3`: the SP3 file(s) of a session read once (all systems), with the coverage of every observed
  system checked before any interpolation: a system the SP3 files do not cover at all (another day) or have no
  satellite of is dropped with a warning, or a ``ValueError`` names the files, their spans, the observation span
  and ``time_limit`` when none is left. Replaces the SP3 reads of ``PPPSession.run``, of ``SPPSession`` once per
  system and of ``TECSession``; today an SP3 of another day ends in "No objects to concatenate".
* :func:`session_clk_files`: the RINEX clock files of ``clk_path`` usable with the SP3 (``rinex_clk``), with the
  same coverage check (a warning, and the SP3 clocks, when they cover none of the observations).
* :func:`resolve_precise_clock_datum`: the IF datum of the precise BeiDou clocks per satellite generation, once per
  SPP/PPP session before the epoch loop: the ``bds_precise_clock_datum`` setting, the
  declaration of a bias product of the clocks' analysis centre, the broadcast-clock test, else IF(B1I,B3I) with a
  warning; a datum the run's codes cannot be moved to is refused with a readable ``ValueError``.
  :func:`bds_clock_datum_of_files` gives the same answer for the SIS comparison, which reads its own files.
* :func:`drop_empty_systems`: after the satellite positions and the elevation mask, a system without
  observations is dropped with one warning naming the reason; with none left the error starts with "No GPS,
  Galileo or BeiDou observations after preprocessing" and gives the reason per system (built from SPP's former
  guard).
* :func:`screen_observations`: the broadcast-health and phase screen of ``screen=True`` (SPP, PPP, TEC): the
  outage flags in force (``io.mark_outages``), for Galileo those of the mode's signals (E1 -> E1-B, E5a, E5b; DVS
  and SHS; ``coordinates._GAL_SIGNAL_HEALTH``), and the presence of the mode's phases (``notna``: a negative
  phase is a valid observation).
* :func:`require_phases`: PPP and TEC need the carrier phases of their mode; a frame without a phase observable
  with data for a signal of the mode (a code-only file, which ``io`` reads for SPP) gives a ``ValueError`` naming
  the file, the mode and the missing phases (it used to end in "Lack of dual frequency data!" from the cycle-slip
  detector; since v1.0.0).
* :func:`product_solutions` / :func:`warn_mixed_solutions`: the solution identifiers of IGS long product names
  (``COD0MGXFIN``, ``GRG0MGXFIN``) with the days of their files; SP3 files of more than one solution in
  ``sp3_path`` give one UserWarning in :func:`load_sp3`, i.e. in SPP, PPP and TEC (orbits and
  clocks of different analysis centres disagree at day boundaries). The PPP-AR product check
  (``PPPSession._warn_ar_product_mismatch``) uses the same parsing.

Nothing here changes a frame that passes: the guards drop systems or rows, or raise.
"""
from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .gnss import SIGNALS, code_signal, ionosphere_free_coefficients, mode_signals, system_name

__all__ = [
    "PHASE_COLUMN", "PRODUCT_ID", "SP3_SYSTEMS", "bds_clock_datum_of_files", "describe_solutions",
    "drop_empty_systems", "file_span", "load_sp3",
    "no_observations_error", "observation_span", "outage_columns", "phase_present", "product_solutions",
    "require_phases", "resolve_precise_clock_datum", "screen_observations", "session_clk_files", "solution_years",
    "sp3_coverage", "sp3_gaps",
    "warn_mixed_solutions",
]

#: Phase observation columns (RINEX 3 phase codes, 'L1C', and the RINEX 2 adapter's 'L1').
PHASE_COLUMN = re.compile(r"^L\d[A-Z]?$")

#: Systems read from the SP3 files.
SP3_SYSTEMS = ("G", "E", "C")

#: Tolerance at the first and last SP3 node (``coordinates.SP3_EDGE_TOLERANCE_S``).
_EDGE = pd.Timedelta(seconds=1)

#: A gap between an SP3 span and the observations counts as at a day edge when the span ends this close to
#: midnight (then the neighbour-day file is the remedy).
_DAY_EDGE = pd.Timedelta(hours=1)


# ----------------------------------------------------------------------------------------------- spans and text
def _times(frame) -> Optional[pd.DatetimeIndex]:
    if frame is None or len(frame) == 0:
        return None
    names = list(getattr(frame.index, "names", []) or [])
    if "time" not in names:
        return None
    return pd.DatetimeIndex(frame.index.get_level_values("time"))


def observation_span(frames: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, tuple]:
    """``{system: (first epoch, last epoch, number of epochs)}`` of the frames that hold observations."""
    out = {}
    for s, frame in frames.items():
        t = _times(frame)
        if t is None:
            continue
        u = t.unique()
        out[s] = (u.min(), u.max(), int(u.size))
    return out


def _pair(first, last) -> Optional[tuple]:
    try:
        first = pd.Timestamp(first) if first is not None else None
        last = pd.Timestamp(last) if last is not None else None
    except (TypeError, ValueError):
        return None
    if first is None or last is None or pd.isna(first) or pd.isna(last):
        return None
    return first, last


def file_span(obs_data=None, frames: Optional[dict] = None) -> Optional[tuple]:
    """``(first, last)`` epoch of the observation file in GPST: the first and last epoch the reader found in the
    data (``obs_data.meta[9:11]``), else the span of ``frames``, else TIME OF FIRST / LAST OBS of the header
    (``obs_data.header``; as written, a spliced or cut file may keep the original times); None when unknown."""
    meta = getattr(obs_data, "meta", None)
    if meta is not None and len(meta) > 10:
        span = _pair(meta[9], meta[10])
        if span is not None:
            return span
    spans = observation_span(frames or {})
    if spans:
        return min(v[0] for v in spans.values()), max(v[1] for v in spans.values())
    header = getattr(obs_data, "header", None)
    return _pair(getattr(header, "time_of_first_obs", None), getattr(header, "time_of_last_obs", None))


def _fmt(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _limit_text(time_limit) -> str:
    return "time_limit " + ("none" if not time_limit else f"{_fmt(time_limit[0])} - {_fmt(time_limit[1])}")


def _window_text(time_limit, span) -> str:
    fs = "unknown" if span is None else f"{_fmt(span[0])} - {_fmt(span[1])}"
    return f"{_limit_text(time_limit)}; observations in the file {fs} (GPST)"


def no_observations_error(prefix: str, systems: Iterable[str], obs_path, time_limit=None, span=None) -> ValueError:
    """``ValueError("<prefix> <obs_path> (<systems>; time_limit ..., file span ...)")``: the readable error of a
    selected system without observations (in the file, or in the time window)."""
    names = ", ".join(f"{s} ({system_name(s)})" for s in sorted(systems))
    hint = ""
    if time_limit and span is not None and (pd.Timestamp(time_limit[1]) < span[0] or
                                            pd.Timestamp(time_limit[0]) > span[1]):
        hint = "; time_limit lies outside the file"
    return ValueError(f"{prefix} {obs_path} ({names}; {_window_text(time_limit, span)}{hint})")


def _names(paths) -> str:
    from .validation import iter_paths

    return ", ".join(Path(str(p)).name for p in iter_paths(paths)) or "none"


# ------------------------------------------------------------------------------------- product solutions
#: Solution identifier of an IGS long product name (centre, version, campaign, type; group 1), followed by
#: ``_YYYYDDDHHMM_``: 'GRG0MGXFIN' in GRG0MGXFIN_20240350000_01D_01D_OSB.BIA (matched on the upper-case name).
PRODUCT_ID = re.compile(r"^([A-Z0-9]{3}[0-9][A-Z0-9]{3}[A-Z]{3})_[0-9]{11}_")


def product_solutions(paths, contents: Iterable[str], pattern: Optional[re.Pattern] = None) -> Dict[str, list]:
    """``{solution id: [(year, day of year, file name), ...]}`` of the IGS long names in ``paths`` whose content
    field is one of ``contents`` (e.g. ``("ORB", "CLK")`` for ``*_ORB.SP3`` / ``*_CLK.CLK``, ``("OSB",)``); other
    names (short names, other contents) are not listed. ``pattern``: a regex matched on the upper-case name, with
    the solution id as group 1 directly followed by ``_YYYYDDD`` (default :data:`PRODUCT_ID`); a match without
    that day stamp is not listed. Files in the order given."""
    from .validation import iter_paths

    pattern = PRODUCT_ID if pattern is None else pattern
    kinds = tuple(contents)
    out: Dict[str, list] = {}
    for p in iter_paths(paths):
        name = Path(str(p)).name
        upper = name.upper()
        m = pattern.match(upper)
        if not m or not any(f"_{k}." in upper for k in kinds):
            continue
        stamp = upper[m.end(1) + 1:m.end(1) + 8]          # YYYYDDD after the solution id
        if upper[m.end(1):m.end(1) + 1] != "_" or len(stamp) != 7 or not stamp.isdigit():
            continue                                        # a custom pattern without the day stamp
        out.setdefault(m.group(1), []).append((int(stamp[:4]), int(stamp[4:]), name))
    return out


def _days_text(entries, years) -> str:
    days = sorted({(y, d) for y, d, _ in entries})
    if len(years) == 1:
        return ", ".join(f"{d:03d}" for _, d in days)
    return ", ".join(f"{y}-{d:03d}" for y, d in days)


def solution_years(*solutions: Dict[str, list]) -> set:
    """The years of the files of one or more :func:`product_solutions` results."""
    return {y for sols in solutions for entries in sols.values() for y, _, _ in entries}


def describe_solutions(solutions: Dict[str, list], years: Optional[Iterable[int]] = None) -> str:
    """'COD0MGXFIN (212, 213) and GRG0MGXFIN (214)': each solution of :func:`product_solutions` with the days of
    its files (day of year; 'YYYY-DDD' when the files span more than one year), solutions in alphabetical order.
    ``years``: the years that decide the format (default: those of ``solutions``); pass
    :func:`solution_years` of every listed product so that texts naming several products share one format."""
    years = solution_years(solutions) if years is None else set(years)
    parts = [f"{sol} ({_days_text(entries, years)})" for sol, entries in sorted(solutions.items())]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def warn_mixed_solutions(paths, session: str, name: str = "sp3_path", stacklevel: int = 1) -> Optional[str]:
    """One UserWarning when the SP3 / clock files of ``paths`` (IGS long names, ``*_ORB.SP3`` / ``*_CLK.CLK``)
    come from more than one
    solution: "<session>: sp3_path mixes COD0MGXFIN (212, 213) and GRG0MGXFIN (214): ...".
    A warning, not an error (the files are used as given). Returns the message, or None when nothing was said
    (one solution, or no long names). ``stacklevel`` counts from the caller (1: the caller's line)."""
    solutions = product_solutions(paths, ("ORB", "CLK"))
    if len(solutions) < 2:
        return None
    message = (f"{session}: {name} mixes {describe_solutions(solutions)}: orbits and clocks of different analysis "
               f"centres disagree at day boundaries; use one product (DownloadConfig picks one by default)")
    warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
    return message


# ------------------------------------------------------------------------------------------------ coverage
def sp3_coverage(sp3_df: pd.DataFrame) -> Dict[str, tuple]:
    """``{system: (first node, last node)}`` of the SP3 records with a position (index levels 'time', 'sv')."""
    if sp3_df is None or len(sp3_df) == 0:
        return {}
    d = sp3_df.reset_index()[["time", "sv", "x"]]
    d = d[d["x"].notna() & d["time"].notna()]
    sys_letter = d["sv"].astype(str).str[:1]
    g = d.groupby(sys_letter)["time"]
    return {s: (lo, hi) for s, lo, hi in zip(g.min().index, g.min().to_numpy(), g.max().to_numpy())}


def sp3_gaps(sp3_df: pd.DataFrame) -> Dict[str, list]:
    """``{system: [(last node before, first node after), ...]}``: the gaps of the SP3 epochs of a system (any of
    its satellites) longer than ``coordinates.SP3_GAP_FACTOR`` x the median node spacing, e.g. the day between
    the files of D-1 and D+1."""
    from .coordinates import SP3_GAP_FACTOR

    if sp3_df is None or len(sp3_df) == 0:
        return {}
    d = sp3_df.reset_index()[["time", "sv", "x"]]
    d = d[d["x"].notna() & d["time"].notna()]
    out = {}
    for s, times in d.groupby(d["sv"].astype(str).str[:1])["time"]:
        t = np.unique(np.asarray(times, dtype="datetime64[ns]"))
        if t.size < 3:
            continue
        step = np.diff(t)
        cut = np.flatnonzero(step > SP3_GAP_FACTOR * np.median(step))
        if cut.size:
            out[s] = [(pd.Timestamp(t[k]), pd.Timestamp(t[k + 1])) for k in cut]
    return out


def _day_edge_advice(lo, hi, before: pd.DatetimeIndex, after: pd.DatetimeIndex) -> Optional[str]:
    """'add the next-day / previous-day file' when the uncovered epochs lie just beyond a span ending at a day
    edge; None otherwise."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    parts = []
    if len(after):
        midnight = hi.normalize() + pd.Timedelta(days=1)
        if midnight - hi <= _DAY_EDGE and after.max() < midnight + pd.Timedelta(days=1):
            parts.append("the next-day file")
        else:
            return None
    if len(before):
        if lo - lo.normalize() <= _DAY_EDGE and before.min() >= lo.normalize() - pd.Timedelta(days=1):
            parts.append("the previous-day file")
        else:
            return None
    return " and ".join(parts) if parts else None


def _coverage(spans: Dict[str, tuple], frames: Dict[str, pd.DataFrame], key=lambda s: s,
              gaps: Optional[Dict[str, list]] = None) -> Dict[str, dict]:
    """Per observed system: 'state' none | partial | full, the uncovered epochs (before, after the span and
    'inside' its ``gaps``) and the day-edge advice."""
    out = {}
    for s, frame in frames.items():
        t = _times(frame)
        if t is None:
            continue
        u = t.unique().sort_values()
        span = spans.get(key(s))
        if span is None:
            out[s] = dict(state="absent", n=int(u.size), uncovered=u)
            continue
        lo, hi = pd.Timestamp(span[0]), pd.Timestamp(span[1])
        before, after = u[u < lo - _EDGE], u[u > hi + _EDGE]
        holes = (gaps or {}).get(key(s), [])
        inside = u[:0]
        for g0, g1 in holes:
            inside = inside.append(u[(u > g0 + _EDGE) & (u < g1 - _EDGE)])
        n_out = len(before) + len(after) + len(inside)
        state = "full" if n_out == 0 else ("none" if n_out == u.size else "partial")
        advice = _day_edge_advice(lo, hi, before, after) if state == "partial" and not len(inside) else None
        out[s] = dict(state=state, n=int(u.size), before=before, after=after, inside=inside, span=(lo, hi),
                      gaps=holes, advice=advice)
    return out


def load_sp3(paths, frames: Dict[str, Optional[pd.DataFrame]], time_limit=None, session: str = "PPP",
             obs_path=None, span=None) -> tuple:
    """The SP3 files ``paths`` read once (systems G, E, C), and the observed systems they serve.

    ``frames``: ``{system: observation frame}`` of the session. Returns ``(sp3_df, kept)``: ``sp3_df`` indexed
    ``(time, sv)`` with x, y, z in metres and ``clk`` in seconds (as the sessions built it before), ``kept`` the
    systems of ``frames`` with observations inside the SP3 span. A system without SP3 records ("No precise SP3
    records found for system C ...") or without any observation inside the span is dropped with one
    RuntimeWarning; when none is left a ValueError names the files, their spans, the observation span and
    ``time_limit`` (``span``: the file's first and last epoch, :func:`file_span`). A partial overlap at a day
    edge is left to the interpolators' warning ("add the next-day SP3 file"); a partial overlap elsewhere (a
    file of a neighbouring day ending inside the observations) or observations inside a gap of the SP3 epochs
    (:func:`sp3_gaps`, e.g. the files of D-1 and D+1 for observations of D) get one warning naming the uncovered
    epochs and their count; when every epoch of a system is uncovered the system is dropped as above. SP3 files
    of more than one solution (IGS long names) give one UserWarning first (:func:`warn_mixed_solutions`).
    """
    from ..io import read_sp3
    from .validation import iter_paths

    files = list(iter_paths(paths))
    warn_mixed_solutions(files, session, stacklevel=3)
    parts = [read_sp3(str(f)) for f in files]
    parts = [df for df in parts if not df.empty]
    if parts:
        sp3_df = pd.concat(parts)
        # keep both records of a day-boundary epoch (24:00 / 00:00): one may lack the clock (999999.999999)
        # that the other has; lagrange_interp merges them per quantity
        sp3_df = sp3_df.reset_index().drop_duplicates().set_index(["time", "sv"])
        sp3_df[["x", "y", "z"]] = sp3_df[["x", "y", "z"]].to_numpy() * 1e3
        sp3_df["clk"] = sp3_df["clk"].to_numpy() * 1e-6
    else:
        sp3_df = pd.DataFrame(columns=["x", "y", "z", "clk"],
                              index=pd.MultiIndex.from_arrays([[], []], names=["time", "sv"]))
    spans = sp3_coverage(sp3_df)
    cov = _coverage(spans, {s: f for s, f in frames.items() if f is not None}, gaps=sp3_gaps(sp3_df))
    names = _names(files)
    obs_span = observation_span(frames)
    problems = {}
    # a system without observation rows (screened out, time window) has nothing to check: kept, so that
    # drop_empty_systems names its reason
    kept = [s for s, f in frames.items() if f is not None and len(f) == 0]
    for s, c in cov.items():
        label = f"{s} ({system_name(s)})"
        if c["state"] == "absent":
            problems[s] = f"No precise SP3 records found for system {label} in {names}"
            continue
        lo, hi = c["span"]
        holes = "".join(f" except {_fmt(g0)} - {_fmt(g1)}" for g0, g1 in c["gaps"])
        if c["state"] == "none":
            o = obs_span[s]
            problems[s] = (f"the SP3 file(s) {names} cover {_fmt(lo)} - {_fmt(hi)}{holes}, the {system_name(s)} "
                           f"observations {_fmt(o[0])} - {_fmt(o[1])} ({_limit_text(time_limit)}): "
                           f"no observation epoch is covered; give the SP3 file(s) of the observation day(s)")
            continue
        kept.append(s)
        if c["state"] == "partial" and c["advice"] is None:
            out = c["before"].append(c["after"]).append(c["inside"])
            warnings.warn(
                f"{session}: {len(out)} of {c['n']} {system_name(s)} observation epochs ({_fmt(out.min())} - "
                f"{_fmt(out.max())}) lie outside the SP3 span {_fmt(lo)} - {_fmt(hi)}{holes} of {names} and get "
                f"no satellite position; give the SP3 file(s) of the observation day(s)",
                RuntimeWarning, stacklevel=3)
    if not kept:
        span = span if span is not None else file_span(None, frames)
        detail = "; ".join(f"{s}: {r}" for s, r in sorted(problems.items()))
        raise ValueError(f"{session}: no satellite position for any observation of {obs_path or 'the observation file'}"
                         f": {detail} ({_window_text(time_limit, span)})")
    for s, why in sorted(problems.items()):
        warnings.warn(f"{session}: {system_name(s)} ({s}) is not processed: {why}; continuing with "
                      f"{', '.join(sorted(kept))}", RuntimeWarning, stacklevel=3)
    return sp3_df, sorted(kept)


def session_clk_files(config, frames: Dict[str, Optional[pd.DataFrame]], session: str = "PPP") -> Optional[list]:
    """The parsed clock files of ``config.clk_path`` the session's filter uses (``rinex_clk.ClkFile`` list), or
    None: no ``clk_path``, broadcast orbits, files of another analysis centre than the SP3 (one warning each,
    the SP3 clocks are kept), files that cannot be read (a time system other than GPS, a damaged archive: one
    warning, the SP3 clocks are kept) or files that cover none of the observations (one warning)."""
    from ..io.rinex_clk import read_rinex_clk, usable_clk_paths
    from .validation import iter_paths

    if not list(iter_paths(getattr(config, "clk_path", None))):
        return None
    if getattr(config, "orbit_type", "precise") != "precise":
        warnings.warn(f"{session}: clk_path is used with precise orbits only; orbit_type='broadcast' keeps the "
                      f"broadcast clocks", UserWarning, stacklevel=3)
        return None
    paths, why = usable_clk_paths(config.clk_path, getattr(config, "sp3_path", None))
    if paths is None:
        warnings.warn(f"{session}: clk_path is not used ({why}); the clocks of sp3_path are kept",
                      UserWarning, stacklevel=3)
        return None
    try:
        files = [read_rinex_clk(p) for p in paths]
    except (OSError, ValueError) as exc:
        # the same handling as a file that cannot be compared by its name (usable_clk_paths): SP3 clocks kept
        warnings.warn(f"{session}: clk_path is not used ({exc}); the clocks of sp3_path are kept",
                      UserWarning, stacklevel=3)
        return None
    nodes = [t for f in files for (t, _) in f.series.values() if t.size]
    if not nodes:
        warnings.warn(f"{session}: clk_path {_names(paths)} holds no satellite clock (AS) record; the clocks of "
                      f"sp3_path are kept", UserWarning, stacklevel=3)
        return None
    lo = pd.Timestamp(min(int(t[0]) for t in nodes))
    hi = pd.Timestamp(max(int(t[-1]) for t in nodes))
    cov = _coverage({"all": (lo, hi)}, {s: f for s, f in frames.items() if f is not None}, key=lambda s: "all")
    if cov and all(c["state"] == "none" for c in cov.values()):
        span = file_span(None, frames)
        warnings.warn(f"{session}: clk_path {_names(paths)} covers {_fmt(lo)} - {_fmt(hi)}, none of the "
                      f"observations ({_window_text(getattr(config, 'time_limit', None), span)}); the clocks of "
                      f"sp3_path are kept. Give the clock files of the observation day(s)",
                      UserWarning, stacklevel=3)
        return None
    return files


# ------------------------------------------------------------------------------- BeiDou precise clock datum
def _centre(path) -> str:
    """Analysis centre of a product file name: its first three characters, upper case."""
    return Path(str(path)).name[:3].upper()


def _declared_bds_references(config, sp3_centres: set) -> tuple:
    """``(same, other)``: the BeiDou SATELLITE_CLOCK_REFERENCE_OBSERVABLES of the bias products of ``dcb_path``
    from the analysis centre of the SP3 files (``same``: [(file name, codes)]) and from other centres
    (``other``). A header that cannot be read or gives no BeiDou line is left out."""
    from ..models.biases import bds_pair_codes, bias_product_clock_references
    from .validation import iter_paths

    same, other = [], []
    for path in iter_paths(getattr(config, "dcb_path", None)):
        try:
            given = bias_product_clock_references(path).get("C")
            codes = bds_pair_codes(given) if given else None
        except (OSError, ValueError):
            continue
        if codes is None:
            continue
        (same if sp3_centres == {_centre(path)} else other).append((Path(str(path)).name, codes))
    return same, other


def _satellites_not_movable(obs: pd.DataFrame, nav: Optional[pd.DataFrame], mode: str, datum,
                            satellites) -> tuple[Optional[list], float, float]:
    """``(stuck, size_m, code_size_m)``: ``stuck``, the BDS-3 satellites of ``satellites`` whose ``mode`` codes get
    no group-delay correction under ``datum`` (``biases.PreciseClockDatum``), decided with the helper the run
    itself uses: one row per satellite (its first observation, with the product biases the reader attached), the
    broadcast group delays of ``nav`` attached as the sessions do, ``biases.code_group_delay_correction(...,
    clock='precise', clock_datum=datum)`` for the codes of the observation file of each mode signal (the signal's
    first code, e.g. C2I, when the frame names none); a satellite is movable when one code per signal gives a finite
    correction. The size of the correction the stuck satellites lack, as RMS over the movable satellites (NaN without a movable
    satellite): ``size_m`` of the mode's observable (the IF combination of a dual-frequency mode, the code of a
    single-frequency one: SPP, the IF and GRAPHIC filters of PPP), ``code_size_m`` of a single code (the largest
    RMS of the mode's codes: the uncombined PPP filters). ``(None, nan, nan)`` without a mode or satellite."""
    from itertools import product as combinations

    from ..models.biases import attach_broadcast_group_delays, code_group_delay_correction

    signals = mode_signals(mode) if mode else ()
    if not signals or obs is None or "sv" not in (obs.index.names or ()):
        return None, float("nan"), float("nan")
    sv = pd.Index(obs.index.get_level_values("sv")).astype(str).str[:3]
    rows = obs[sv.isin(list(satellites))]
    rows = rows[~pd.Index(rows.index.get_level_values("sv")).astype(str).str[:3].duplicated()]
    if len(rows) == 0:
        return None, float("nan"), float("nan")
    candidates = []
    for signal in signals:
        cols = []
        for col in rows.columns:
            if isinstance(col, str) and len(col) == 3 and col[0] == "C":
                try:
                    if code_signal("C", col) == signal:
                        cols.append(col)
                except ValueError:
                    continue
        if not cols:
            spec = SIGNALS[signal]
            cols = [spec.code_prefix + spec.suffix_priority[0]]
        candidates.append(cols)
    probe = attach_broadcast_group_delays(rows, nav, "C")
    weights = (1.0,)
    if len(signals) == 2:
        k1, k2 = ionosphere_free_coefficients(*signals)
        weights = (k1, -k2)
    movable = np.zeros(len(probe), dtype=bool)
    # per row: the correction of the combination and the bias of each code, from the first combination that gives it
    size = np.full((len(probe), 1 + len(signals)), np.nan)
    for codes in combinations(*candidates):
        result = code_group_delay_correction(probe, "C", codes, weights, clock="precise", clock_datum=datum)
        finite = np.isfinite(result.correction_m)
        fill = finite & ~movable
        size[fill, 0] = result.correction_m[fill]
        for i, b in enumerate(result.biases_m[:len(signals)]):
            size[fill, 1 + i] = np.asarray(b, dtype=float)[fill]
        movable |= finite
    names = pd.Index(probe.index.get_level_values("sv")).astype(str).str[:3]
    moved = size[movable]
    rms = [float(np.sqrt(np.mean(col[np.isfinite(col)] ** 2))) if np.isfinite(col).any() else float("nan")
           for col in moved.T]
    codes_rms = [r for r in rms[1:] if np.isfinite(r)]
    return sorted(set(names[~movable])), rms[0], (max(codes_rms) if codes_rms else float("nan"))


def _is_osb_product(path) -> bool:
    """True for an absolute-bias (OSB) Bias-SINEX product: ``BIAS_MODE ABSOLUTE`` in its +BIAS/DESCRIPTION block,
    else the bias-mode letter ``A`` of its ``%=BIA`` header line (``biases.bias_product_mode``); a file that cannot
    be read or has neither (e.g. a CODE .DCB file) is not one. Before v1.0.0 "OSB" in the file name."""
    from ..models.biases import bias_product_mode

    try:
        return bias_product_mode(path) == "ABSOLUTE"
    except (OSError, ValueError):
        return False


def _run_generations(config, sats, *, nav: Optional[pd.DataFrame], obs: pd.DataFrame, antenna=None) -> tuple:
    """``(generations, how)`` of the BeiDou satellites ``sats`` of a run: the ANTEX block valid at the first
    observation (``antenna``, else ``config.atx_path``), else B-CNAV records of ``nav``, else the PRN
    (``biases.bds_satellite_generations``); shared by the precise and the broadcast path."""
    from ..models.biases import bds_bcnav_satellites, bds_satellite_generations

    times = _times(obs)
    epoch0 = times.min().to_pydatetime() if times is not None else None
    atx = getattr(antenna, "atx", None)
    if atx is None and getattr(config, "atx_path", None):
        from ..models.antex import load_antex
        try:
            atx = load_antex(config.atx_path)
        except (OSError, ValueError):
            atx = None
    return bds_satellite_generations(sats, antex=atx, epoch=epoch0, cnav_satellites=bds_bcnav_satellites(nav))


def _broadcast_generations(config, *, nav: Optional[pd.DataFrame], obs: pd.DataFrame, antenna=None):
    """The BeiDou satellite generations of a broadcast-orbit run (``biases.PreciseClockDatum`` with
    ``source='broadcast'``, refs IF(B1I,B3I); :func:`_run_generations`). None without BeiDou observations."""
    from ..models.biases import PRECISE_CLOCK_REFERENCE, BDS2, BDS3, BDS3_GEO, PreciseClockDatum

    if obs is None or len(obs) == 0 or "sv" not in (obs.index.names or ()):
        return None
    sats = {s for s in pd.Index(obs.index.get_level_values("sv")).astype(str).str[:3] if s.startswith("C")}
    if not sats:
        return None
    generations, how = _run_generations(config, sats, nav=nav, obs=obs, antenna=antenna)
    default = PRECISE_CLOCK_REFERENCE["C"]
    return PreciseClockDatum(refs={BDS3: default, BDS3_GEO: default, BDS2: default}, generations=generations,
                             source="broadcast", generation_sources=how)


def resolve_precise_clock_datum(config, *, sp3_df: Optional[pd.DataFrame], nav: Optional[pd.DataFrame],
                                obs: Optional[pd.DataFrame], antenna=None, session: str = "PPP"):
    """The IF datum of the precise BeiDou clocks of a session (``biases.PreciseClockDatum``), resolved once
    before the epoch loop, or None (broadcast orbits, no BeiDou observation).

    Generation of each satellite: the ANTEX block valid at the first observation (``antenna``, else
    ``config.atx_path``), else B-CNAV records in the navigation data, else the PRN (``biases.
    bds_satellite_generations``). BDS-2 satellites always take IF(B1I,B3I). The BDS-3 datum, in this order:

    a) ``config.bds_precise_clock_datum`` ('B1IB3I' / 'B1CB2a'; 'auto' goes on); a disagreeing test is warned;
    b) declared: SATELLITE_CLOCK_REFERENCE_OBSERVABLES 'C' of a bias product of ``dcb_path`` from the analysis
       centre of the SP3 files (first three characters of the file names; band 2/6/1/5 = B1I/B3I/B1C/B2a, the
       tracking attribute ignored); a disagreeing test is warned;
    c) detected: ``biases.detect_bds_clock_datum`` on the SP3 and navigation clocks of the observation span;
       a declaration of another centre is evidence only (a disagreement is warned);
    d) assumed: IF(B1I,B3I) with one warning naming the ~10 m risk and the remedies.

    A datum other than IF(B1I,B3I) needs the codes of ``bds_freq`` moved to it: by the bias product (OSB/DSB of
    the codes) or by the B-CNAV TGD_B1Cp/TGD_B2ap of the navigation data; with neither a ``ValueError`` explains
    the remedies (``bds_freq='B1CB2a'``, a RINEX 4 navigation file, the analysis centre's OSB product), where the
    run would otherwise end ~10 m off without a message. Whether the codes can be moved is decided per observed
    BDS-3 satellite with the bias helper itself (:func:`_satellites_not_movable`: an OSB of another clock datum
    counts only when the product's OSBs of the datum codes refer it, a DSB only with a chain to the datum codes;
    only satellites with records in the SP3 files, the others never reach the adjustment): none movable -> the
    ValueError; some -> one warning naming the others, whose code group delay is taken as 0 and whose code
    observations SPP and every PPP filter down-weight (``PreciseClockDatum.unmoved``; the RMS over the movable
    satellites of the correction of the observable the filter uses, at least ``biases.UNMOVED_CODE_SIGMA_FLOOR_M``,
    added in quadrature to their code sigma: ``unmoved_sigma_m`` of the mode's combination for SPP and the PPP IF /
    GRAPHIC filters, ``unmoved_code_sigma_m`` of a single code for the uncombined PPP filters). BDS-3 GEO satellites (no open-service
    B1C/B2a) keep IF(B1I,B3I) like BDS-2 (named in a warning). ``osb_as_given``: every absolute-bias (OSB) product
    of dcb_path (``BIAS_MODE ABSOLUTE`` in its header; before: an "OSB" file name) declares the
    BDS-3 datum, so its OSBs are used as given. The result is kept for the run in
    ``_notes.run_memo(config, 'precise_clock_datum')['C']`` (read by the SPP solver and the PPP filters).

    Broadcast orbits: no clock datum (None), but the satellite generations of the run (ANTEX block valid at the
    first observation, B-CNAV records, PRN) are kept in ``_notes.run_memo(config, 'bds_generations')['C']``
    (a ``PreciseClockDatum`` with ``source='broadcast'``): they decide whether a band-7 code is B2I (BDS-2, group
    delay TGD2) or B2b-I (BDS-3, TGD_B2bI), instead of the PRN rule that is wrong after 2023.
    """
    import logging

    from . import _notes
    from ..models.biases import (BDS2, BDS3, BDS3_GEO, BDS_DATUM_PAIRS, PRECISE_CLOCK_REFERENCE,
                         UNMOVED_CODE_SIGMA_FLOOR_M, PreciseClockDatum, _same_signals, bds_pair_codes,
                         bds_pair_name, detect_bds_clock_datum)
    from .validation import iter_paths

    memo = _notes.run_memo(config, "precise_clock_datum")
    memo.pop("C", None)
    generations_memo = _notes.run_memo(config, "bds_generations")
    generations_memo.pop("C", None)
    if obs is None or len(obs) == 0:
        return None
    if getattr(config, "orbit_type", "precise") != "precise":
        # broadcast clocks: the satellite generations only (B2I / B2b-I of the band-7 codes, NQ-SPP-32)
        generations_memo["C"] = _broadcast_generations(config, nav=nav, obs=obs, antenna=antenna)
        return None
    times = _times(obs)
    sats = set(pd.Index(obs.index.get_level_values("sv")).astype(str).str[:3]) if "sv" in obs.index.names else set()
    sp3_c = None
    if sp3_df is not None and len(sp3_df):
        sv = pd.Index(sp3_df.index.get_level_values("sv")).astype(str)
        sp3_c = sp3_df[sv.str.startswith("C")]
        sats |= set(sv[sv.str.startswith("C")].str[:3])
    # the satellite generation: ANTEX block valid at the first epoch, B-CNAV records, PRN
    generations, how = _run_generations(config, sats, nav=nav, obs=obs, antenna=antenna)
    observed = {s for s in pd.Index(obs.index.get_level_values("sv")).astype(str).str[:3]} \
        if "sv" in obs.index.names else set()
    bds3_observed = sorted(s for s in observed if generations.get(s) == BDS3)
    # the satellites that can reach the adjustment: an observed satellite without SP3 records is dropped by the
    # session (e.g. the test satellites C56/C58, which COD 2026-213 does not give)
    sp3_sats = set(pd.Index(sp3_c.index.get_level_values("sv")).astype(str).str[:3]) \
        if sp3_c is not None and len(sp3_c) else None
    bds3_used = [s for s in bds3_observed if sp3_sats is None or s in sp3_sats]

    sp3_centres = {_centre(p) for p in iter_paths(getattr(config, "sp3_path", None))}
    sp3_names = _names(getattr(config, "sp3_path", None))
    same, other = _declared_bds_references(config, sp3_centres)
    test = detect_bds_clock_datum(sp3_c, nav, generations,
                                  start=None if times is None else times.min(),
                                  end=None if times is None else times.max()) \
        if sp3_c is not None and nav is not None and len(nav) else {"datum": None, "reason": "no navigation data"}
    tested = test.get("datum")
    tested_codes = BDS_DATUM_PAIRS[tested] if tested else None
    setting = getattr(config, "bds_precise_clock_datum", "auto") or "auto"
    h2 = (f"{test.get('std_h2_m')} m" if test.get("std_h2_m") is not None
          else "not testable without B-CNAV TGD_B1Cp/TGD_B2ap")
    stats = (f"between-satellite scatter of the BDS-3 precise minus broadcast clocks: IF(B1I,B3I) "
             f"{test.get('std_h1_m')} m, IF(B1C,B2a) {h2}, {test.get('n_bds3', 0)} satellites")
    notes: list = []
    if setting != "auto":
        codes, source = bds_pair_codes(setting), "setting"
        if tested_codes is not None and not _same_signals(codes, tested_codes):
            notes.append(f"bds_precise_clock_datum='{setting}' is used although the broadcast-clock test finds "
                         f"{bds_pair_name(tested_codes)} ({stats})")
    elif same:
        name, codes = same[0]
        source = "declared"
        if tested_codes is not None and not _same_signals(codes, tested_codes):
            notes.append(f"the bias product {name} declares the BeiDou clock datum {bds_pair_name(codes)}, the "
                         f"broadcast-clock test finds {bds_pair_name(tested_codes)} ({stats}); the declaration is "
                         f"used (set bds_precise_clock_datum to override)")
    elif tested_codes is not None:
        codes, source = tested_codes, "detected"
    else:
        codes, source = PRECISE_CLOCK_REFERENCE["C"], "assumed"
    for name, declared in other:
        if not _same_signals(declared, codes):
            notes.append(f"the bias product {name} of another analysis centre declares the BeiDou clock datum "
                         f"{bds_pair_name(declared)}, the clocks of {sp3_names} are taken as {bds_pair_name(codes)} "
                         f"({source}{'; ' + stats if tested else ''})")
    if _same_signals(codes, PRECISE_CLOCK_REFERENCE["C"]):
        # IF(B1I,B3I) in any tracking mode (C2I C6I, L2I L6I): the datum GNX assumed before, bit-identical
        codes = PRECISE_CLOCK_REFERENCE["C"]
    # BDS-3 GEO: B1I/B3I/B2b only, no B1C/B2a (their clocks cannot refer to IF(B1C,B2a)); BDS-2 likewise
    refs = {BDS3: tuple(codes), BDS3_GEO: PRECISE_CLOCK_REFERENCE["C"], BDS2: PRECISE_CLOCK_REFERENCE["C"]}
    detail = {"std_h1_m": test.get("std_h1_m"), "std_h2_m": test.get("std_h2_m"), "n_bds3": test.get("n_bds3", 0),
              "test": tested, "two_sided": test.get("two_sided", False), "declared": [list(c) for _, c in same],
              "generation_prn_rule": sorted(s for s, h in how.items() if h == "prn" and s in observed)}
    # the OSBs of dcb_path are used as given only when every OSB product declares this BDS-3 datum (a product of
    # the clocks' centre, or one of the same datum); else they are referred to it (biases.satellite_code_bias_m)
    osb_files = [p for p in iter_paths(getattr(config, "dcb_path", None)) if _is_osb_product(p)]
    declared_by = {name: c for name, c in same + other}
    osb_as_given = bool(osb_files) and all(
        Path(str(p)).name in declared_by and _same_signals(declared_by[Path(str(p)).name], codes) for p in osb_files)
    datum = PreciseClockDatum(refs=refs, generations=generations, source=source, detail=detail,
                              generation_sources=how, osb_as_given=osb_as_given)
    mode = getattr(config, "bds_freq", None)
    for text in notes:
        _notes.warn_once(config, ("bds_clock_datum", text), f"{session}: {text}", UserWarning, stacklevel=3)
    if source == "assumed" and bds3_observed:
        evidence = f" ({test.get('reason')})" if test.get("reason") else ""
        _notes.warn_once(
            config, ("bds_clock_datum", "assumed"),
            f"{session}: the datum of the BeiDou precise clocks of {sp3_names} is not determined{evidence}: they are "
            f"taken as IF(B1I,B3I). Products of 2026 refer their BDS-3 clocks to IF(B1C,B2a) (COD MGEX 2026-213); "
            f"with the wrong datum every BDS-3 code is off by 3-10 m per satellite (SPP C IF(B1I,B3I) BRUX 2026-213 "
            f"11.4 m instead of 2.4 m rms). Give a RINEX 4 navigation file in nav_path (the datum is then tested "
            f"on the broadcast clocks), the analysis centre's OSB product in dcb_path (its header declares the "
            f"datum), or set bds_precise_clock_datum", UserWarning, stacklevel=3)
    if not _same_signals(codes, PRECISE_CLOCK_REFERENCE["C"]):
        bds2 = sorted(s for s in observed if generations.get(s) == BDS2)
        geo = sorted(s for s in observed if generations.get(s) == BDS3_GEO)
        if bds2 or geo:
            which = " and ".join(part for part in (f"the BDS-2 satellites {', '.join(bds2)}" if bds2 else "",
                                                   f"the BDS-3 GEO satellites {', '.join(geo)}" if geo else "")
                                 if part)
            _notes.warn_once(
                config, ("bds_clock_datum", "bds2"),
                f"{session}: the BDS-3 clocks of {sp3_names} refer to {bds_pair_name(codes)} ({source}); {which} "
                f"transmit no open-service B1C/B2a and are taken in IF(B1I,B3I) (not verified)",
                UserWarning, stacklevel=3)
        signals = mode_signals(mode) if mode else ()
        # the mode's combination is the datum itself (its group delay is 0): nothing to move
        datum_combination = len(signals) == 2 and sorted(signals) == sorted(code_signal("C", c) for c in codes)
        stuck, size_m, code_size_m = _satellites_not_movable(obs, nav, mode, datum, bds3_used) \
            if bds3_used and not datum_combination else (None, float("nan"), float("nan"))
        if stuck and len(stuck) == len(bds3_used):
            raise ValueError(
                f"{session}: the BeiDou precise clocks of {sp3_names} refer to {bds_pair_name(codes)} for the "
                f"BDS-3 satellites ({source}{'; ' + stats if tested else ''}), and the {mode} codes cannot be "
                f"moved to that datum: the navigation data (nav_path) have no B-CNAV TGD_B1Cp/TGD_B2ap and no "
                f"bias product in dcb_path gives the codes in that datum (an OSB with the OSBs of "
                f"{' and '.join(codes)}, or DSBs chained to them). Use bds_freq='B1CB2a' (the datum combination), a "
                f"RINEX 4 navigation file with B-CNAV records in nav_path (e.g. BRD400DLR), or the analysis centre's "
                f"OSB product in dcb_path; bds_precise_clock_datum='B1IB3I' forces the old datum (3-10 m per "
                f"satellite off when it is wrong)")
        if stuck:
            # author decision of 2026-10-15 (NQ-SPP-33): the warning stays and the code observations of these
            # satellites are down-weighted by the size of the missing term, so that they do not drive the position:
            # the term of the observable each filter uses (the mode's combination: SPP, PPP IF / GRAPHIC; a single
            # code: the uncombined PPP filters), at least the floor
            def floor(value):
                return max(float(value) if np.isfinite(value) else 0.0, UNMOVED_CODE_SIGMA_FLOOR_M)

            sigma, code_sigma = floor(size_m), floor(code_size_m)
            datum = dataclasses.replace(datum, unmoved=frozenset(stuck), unmoved_sigma_m=sigma,
                                        unmoved_code_sigma_m=code_sigma)
            what = f"{sigma:.1f} m" if len(signals) == 1 or np.isclose(sigma, code_sigma, atol=0.05) else \
                (f"{sigma:.1f} m on the {mode} combination (SPP, PPP IF), {code_sigma:.1f} m on each code "
                 f"(uncombined PPP)")
            _notes.warn_once(
                config, ("bds_clock_datum", "not movable"),
                f"{session}: the {mode} codes of the BDS-3 satellites {', '.join(stuck)} cannot be moved to the "
                f"precise clock datum {bds_pair_name(codes)} ({source}): no B-CNAV TGD_B1Cp/TGD_B2ap record of "
                f"these satellites in nav_path and no product bias in that datum; their code group delay is missing "
                f"(taken as 0, 3-10 m per satellite off). SPP and PPP down-weight their code observations: the "
                f"size of the missing term, {what} (the RMS of that correction over the other BDS-3 satellites, at "
                f"least {UNMOVED_CODE_SIGMA_FLOOR_M:.0f} m), is added in quadrature to their code standard deviation, "
                f"so they do not drive the position", UserWarning, stacklevel=3)
        logging.getLogger(__name__).info("%s: BeiDou precise clock datum %s for BDS-3 (%s; %s)", session,
                                         bds_pair_name(codes), source, stats)
    memo["C"] = datum
    return datum


def bds_clock_datum_of_files(sp3_paths, nav_path, *, atx_path=None, dcb_paths=None) -> dict:
    """The IF datum of the precise BeiDou clocks of SP3 files, for a comparison that reads the files itself (the
    SIS comparison): the declaration of a bias product of ``dcb_paths`` from the SP3's analysis
    centre, else the broadcast-clock test (``biases.detect_bds_clock_datum``) against the BeiDou records of
    ``nav_path``. Returns ``{'datum': 'B1IB3I' | 'B1CB2a' | None, 'source': 'declared' | 'detected' | None,
    'std_h1_m', 'std_h2_m', 'reason'}``; None datum: undetermined (no file, too few satellites)."""
    from ..models.biases import BDS_DATUM_PAIRS, PRECISE_CLOCK_REFERENCE, _same_signals, bds_bcnav_satellites, \
        bds_satellite_generations, detect_bds_clock_datum
    from ..io import read_sp3
    from .validation import iter_paths

    sp3_paths = [p for p in iter_paths(sp3_paths)]
    out = {"datum": None, "source": None, "std_h1_m": None, "std_h2_m": None, "reason": "no SP3 file"}
    if not sp3_paths:
        return out
    same, _ = _declared_bds_references(SimpleNamespace(dcb_path=dcb_paths), {_centre(p) for p in sp3_paths})
    if same:
        codes = same[0][1]
        name = "B1IB3I" if _same_signals(codes, PRECISE_CLOCK_REFERENCE["C"]) else \
            next((k for k, v in BDS_DATUM_PAIRS.items() if _same_signals(codes, v)), None)
        return {**out, "datum": name, "source": "declared", "reason": f"declared by {same[0][0]}"}
    if not nav_path:
        return {**out, "reason": "no navigation file"}
    from ..io.rinex_nav import read_navigation

    sp3 = pd.concat([read_sp3(str(p)) for p in sp3_paths])
    sv = pd.Index(sp3.index.get_level_values("sv")).astype(str)
    sp3 = sp3[sv.str.startswith("C")].reset_index().drop_duplicates(["time", "sv"]).set_index(["time", "sv"])
    sp3["clk"] = sp3["clk"].to_numpy(dtype=float) * 1e-6
    nav = read_navigation(str(nav_path), "C").bds
    atx = None
    if atx_path:
        from ..models.antex import load_antex
        try:
            atx = load_antex(atx_path)
        except (OSError, ValueError):
            atx = None
    times = pd.DatetimeIndex(sp3.index.get_level_values("time"))
    epoch = times.min().to_pydatetime() if len(times) else None
    gens, _ = bds_satellite_generations(set(pd.Index(sp3.index.get_level_values("sv")).astype(str).str[:3]),
                                        antex=atx, epoch=epoch, cnav_satellites=bds_bcnav_satellites(nav))
    test = detect_bds_clock_datum(sp3, nav, gens)
    return {"datum": test["datum"], "source": "detected" if test["datum"] else None, "std_h1_m": test["std_h1_m"],
            "std_h2_m": test["std_h2_m"], "reason": test.get("reason", "")}


# ------------------------------------------------------------------------------------------- empty systems
def drop_empty_systems(crd_by_sys: Dict[str, pd.DataFrame], counts: Dict[str, dict], session: str, obs_path,
                       ev_mask, orbit_type: str = "precise", products=None) -> Dict[str, pd.DataFrame]:
    """Frames of the systems that keep observations; the others are dropped with one RuntimeWarning
    ("<session>: no usable observation of system(s) ['C'] in <file> (C: nothing above ev_mask=80 deg); solving
    with ['E', 'G'] only."). With none left: ValueError "<session>: No GPS, Galileo or BeiDou observations after
    preprocessing (<file>): <reason per system>".

    ``counts[s]``: ``{'obs': rows before the satellite positions, 'orbit': rows with a satellite position}``
    (missing keys are not reported); the reason is the first empty stage: no observation (after the time window
    and the screening), no satellite position, nothing above ``ev_mask``. ``'reported': True`` marks a system
    whose loss was already reported (:func:`load_sp3`): it is not named in the warning again.
    """
    def reason(s) -> str:
        c = counts.get(s, {})
        if c.get("obs", 1) == 0:
            return "no observation in the time window after the screening"
        if c.get("orbit", 1) == 0:
            names = f" {_names(products)}" if products else ""
            return f"no satellite position (the {orbit_type} orbit products{names} do not cover the observations)"
        return f"nothing above ev_mask={ev_mask} deg (check ev_mask and the a-priori position)"

    empty = sorted(s for s, crd in crd_by_sys.items() if crd is None or len(crd) == 0)
    if not empty:
        return crd_by_sys
    why = {s: reason(s) for s in empty}
    left = sorted(set(crd_by_sys) - set(empty))
    if not left:
        raise ValueError(f"{session}: No GPS, Galileo or BeiDou observations after preprocessing ({obs_path}): "
                         + "; ".join(f"{s} ({system_name(s)}): {r}" for s, r in why.items()))
    empty = [s for s in empty if not counts.get(s, {}).get("reported")]
    why = {s: why[s] for s in empty}
    if not empty:
        return {s: crd_by_sys[s] for s in left}
    warnings.warn(
        f"{session}: no usable observation of system(s) {empty} in {obs_path} "
        f"({'; '.join(f'{s}: {r}' for s, r in why.items())}); solving with {left} only.",
        RuntimeWarning, stacklevel=3)
    return {s: crd_by_sys[s] for s in left}


# ------------------------------------------------------------------------------------------------ screening
def outage_columns(system: str, mode: Optional[str], columns) -> list:
    """Outage-flag columns of ``io.mark_outages`` that reject a row of ``system`` in ``mode``: 'gps_outage',
    'bds_outage', or the Galileo DVS/SHS flags of the mode's signals (all three signals without a mode)."""
    if system == "G":
        names = ["gps_outage"]
    elif system == "C":
        names = ["bds_outage"]
    elif system == "E":
        from .coordinates import _GAL_SIGNAL_HEALTH, _galileo_health_signals

        names = []
        for signal in (_galileo_health_signals(mode) if mode else tuple(_GAL_SIGNAL_HEALTH)):
            spec = _GAL_SIGNAL_HEALTH.get(signal)
            if spec is not None:
                names += [f"{spec[2]}_outage", f"{spec[1]}_outage"]       # DVS, SHS
    else:
        names = []
    cols = set(columns)
    return [c for c in names if c in cols]


def _flag_true(values) -> np.ndarray:
    v = np.asarray(values)
    if v.dtype == bool:
        return v
    return np.asarray(pd.notna(v), dtype=bool) & np.asarray(v == True, dtype=bool)  # noqa: E712 - elementwise


def phase_present(frame: pd.DataFrame, system: str, mode: Optional[str]) -> np.ndarray:
    """Rows with every phase of the mode present (all phase columns when none maps to the mode)."""
    phases = [c for c in frame.columns if isinstance(c, str) and PHASE_COLUMN.match(c)]
    if mode:
        signals = set(mode_signals(mode))
        own = []
        for c in phases:
            try:
                if code_signal(system, "C" + c[1:]) in signals:
                    own.append(c)
            except ValueError:
                continue
        phases = own or phases
    if not phases:
        return np.ones(len(frame), dtype=bool)
    return frame[phases].notna().all(axis=1).to_numpy(dtype=bool)


def require_phases(obs_by_sys: Dict[str, Optional[pd.DataFrame]], modes: Dict[str, Optional[str]],
                   systems: Iterable[str], obs_path, session: str) -> None:
    """ValueError when a frame of ``systems`` has no phase observable with data for a signal of its mode.

    PPP and TEC use carrier phases. ``io`` reads a code-only file (no phase observable with data) for SPP, so the
    check belongs to these sessions: the message names the session, the system, the mode, the missing signals, the
    phase observables of the frame and ``obs_path``. A frame that is None or empty is left to the empty-system
    guards; a phase column that maps to no signal of GNX makes the check give up for that frame (nothing raised).
    """
    for s in systems:
        frame = obs_by_sys.get(s)
        mode = modes.get(s)
        if frame is None or len(frame) == 0 or not mode:
            continue
        try:
            signals = mode_signals(mode)
        except ValueError:
            continue
        with_data = {}
        unknown = False
        for c in frame.columns:
            if not (isinstance(c, str) and PHASE_COLUMN.match(c)) or not frame[c].notna().any():
                continue
            try:
                with_data.setdefault(code_signal(s, "C" + c[1:]), []).append(c)
            except ValueError:
                unknown = True
        missing = [sig for sig in signals if sig not in with_data]
        if not missing or unknown:
            continue
        present = [c for c in frame.columns if isinstance(c, str) and PHASE_COLUMN.match(c)]
        found = ", ".join(f"{c}{'' if frame[c].notna().any() else ' (empty)'}" for c in present) or "none"
        raise ValueError(
            f"{session} needs carrier phases: {system_name(s)} mode {mode!r} has no {'/'.join(missing)} phase "
            f"observable with data in {obs_path} (phase observables read: {found}); a code-only file can be "
            "processed by SPP")


def screen_observations(processor, obs_by_sys: Dict[str, Optional[pd.DataFrame]], broadcast,
                        modes: Dict[str, Optional[str]], systems: Iterable[str]) -> Dict[str, Optional[pd.DataFrame]]:
    """``screen=True``: rows of ``systems`` without a broadcast outage in force and with the mode's phases.

    The flags come from ``processor.screen_navigation_message`` / ``processor.mark_outages`` (in-force health
    of the navigation records); GPS rejects on 'gps_outage', BeiDou on 'bds_outage', Galileo on the
    DVS and SHS flags of the mode's signals (the TEC session built 'E1_*' names that never
    matched). The flag columns stay in the frames."""
    out = dict(obs_by_sys)
    navs = {"G": getattr(broadcast, "gps_orb", None), "E": getattr(broadcast, "gal_orb", None),
            "C": getattr(broadcast, "bds_orb", None)}
    for s in systems:
        obs = out.get(s)
        if obs is None:
            continue
        health = processor.screen_navigation_message(navs.get(s), s)
        obs = processor.mark_outages(obs, s, health)
        bad = np.zeros(len(obs), dtype=bool)
        for col in outage_columns(s, modes.get(s), obs.columns):
            bad |= _flag_true(obs[col].to_numpy())
        keep = ~bad & phase_present(obs, s, modes.get(s))
        out[s] = obs[keep] if not keep.all() else obs
    return out
