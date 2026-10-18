import datetime
import math
import warnings
import numpy as np
import pandas as pd
import gnx_py.core.time
from . import _notes
from .utils import calculate_distance
from typing import Optional, Union
from .gnss import (
    CLIGHT,
    SYSTEM_NAMES,
    code_prefixes,
    frequency_hz,
    is_bds_geo,
    mode_ionosphere_free_coefficients,
    mode_layout,
    mode_signals,
    signal_spec,
)

try:
    RankWarning = np.exceptions.RankWarning  # NumPy >=2.0
except AttributeError:
    try:
        RankWarning = np.RankWarning        # NumPy <2.0
    except AttributeError:
        RankWarning = Warning

# ---------------- Numba guard ----------------
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

# ---------------- Stałe ----------------
C_LIGHT = CLIGHT           # [m/s]
pi = np.pi
WEEK = 604800.0


_ANY_CODE_PREFIXES = ("C1", "C2", "C5", "C6", "C7")


def _select_code_column(df: pd.DataFrame, mode: str | None) -> str:
    """Return the code observable used to turn a range into a travel time.

    ``mode`` picks the band, which matters for BeiDou where the band numbers do
    not follow the GPS ones. Without a mode the search falls back to band order,
    which is only right by coincidence for some modes - so callers should pass
    the mode they already know.
    """
    mode = mode or ""
    try:
        preferred_prefixes = code_prefixes(mode)
    except ValueError:
        preferred_prefixes = _ANY_CODE_PREFIXES

    for prefix in preferred_prefixes:
        matches = [c for c in df.columns if c.startswith(prefix)]
        if matches:
            return matches[0]

    fallback = [c for c in df.columns if c.startswith(_ANY_CODE_PREFIXES)]
    if fallback:
        return fallback[0]

    observation_like = sorted(
        c for c in map(str, df.columns) if c[:1] in {"C", "P", "L", "D", "S"}
    )
    raise KeyError(
        f"No code observation column for mode {mode or '<unset>'!r}: looked for "
        f"{list(preferred_prefixes)} then {list(_ANY_CODE_PREFIXES)}, but the frame "
        f"only offers {observation_like or 'no observation-like columns'}. "
        "A RINEX 2 file that was not routed through the RINEX 2 adapter still "
        "carries its original P1/P2 names and lands here."
    )

def _process_satellite_interpolation(args):
    """
        Processes interpolation for a single satellite.

        args – dictionary containing:
          - sat: satellite identifier
          - sat_df: DataFrame with data for the satellite (sorted by toc)
          - epoch_times: NumPy array of continuous epoch times (in seconds)
          - epochs: list of epochs (datetime) corresponding to epoch_times
          - method: interpolation method ('chebyshev' or 'cubic_spline')
          - window_size: initial window size
          - min_window_size: minimum window size
          - delta_t: small time step for calculating velocity (seconds)
          - step: SP3 node spacing in seconds (from the file; 300 s if absent)
          - clk_common: the system's common clock steps (see SP3_CLOCK_COMMON_MIN_SATS), optional
        Returns (results, skipped): a list of dictionaries with interpolation results for a given
        satellite and a list of (sat, epoch, reason, quantity) for epochs that the SP3 nodes do
        not cover (see _sp3_locate) and that are therefore left out instead of extrapolated.
        """
    sat = args['sat']
    sat_df = args['sat_df']
    epoch_times = np.asarray(args['epoch_times'], dtype=np.float64)
    epochs = args['epochs']
    nodes = _SP3SatNodes(sat_df, args.get('step'), args['min_window_size'], args.get('clk_common'))
    results, skipped = [], []
    for i, (epoch_time, epoch) in enumerate(zip(epoch_times, epochs)):
        row, why = nodes.evaluate(i, epoch_time, epoch_times, args['method'], args['window_size'],
                                  args['min_window_size'], args['delta_t'])
        if row is None:
            skipped.append((sat, epoch) + why)
            continue
        row.update({'sv': sat, 'epoch': epoch})
        results.append(row)
    return results, skipped


# ---------------- SP3 nodes: spacing, coverage, missing values ----------------
# The SP3 interpolators never leave the nodes. An epoch is served only between two valid nodes
# of one contiguous arc (consecutive nodes at most SP3_GAP_FACTOR node steps apart, i.e. up to
# two missing nodes in a row), or within SP3_EDGE_TOLERANCE_S of an arc end. The tolerance is
# there for the signal travel time (0.07-0.14 s): the emission epoch of a 00:00:00 observation
# lies just before the first node of a daily file. Beyond it the error grows fast: a 5-min SP3
# extrapolated by 5 min is off by 0.7-8 m in orbit and 0.1-4 m in clock (GRG 2024-035, 95th
# percentile), so such satellite-epochs are left out and reported; the neighbour-day file
# covers them. Positions and clocks are located over their own valid nodes, so a missing clock
# (999999.999999) does not remove the position node and vice versa.
SP3_EDGE_TOLERANCE_S = 1.0
SP3_GAP_FACTOR = 3.0
# An arc must hold at least SP3_MIN_ARC_NODES position nodes: an isolated 5-node arc of 5-min
# nodes is off by up to 0.6 m (0.2 m at the 95th percentile), 6 nodes by 6 cm, 8 nodes by 3 mm.
SP3_MIN_ARC_NODES = 8
# Two files giving the same epoch (day boundary) must agree on the position within this [m].
SP3_BOUNDARY_TOL_M = 1.0
_SP3_INTERP_COLUMNS = ['sv', 'epoch', 'x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'dt_rel', 'clk_raw']

_SP3_SKIP_BEFORE, _SP3_SKIP_AFTER, _SP3_SKIP_GAP, _SP3_SKIP_CLOCK_JUMP = 1, 2, 3, 4
_SP3_SKIP_NO_SAT = 5  # satellite absent from the SP3 file(s)

# SP3 clocks are interpolated linearly between two valid clock nodes, which is wrong by up to the
# size of a clock jump between them: COD0MGXFIN 2024-035 C48 resets by -27.5 us between 18:20 and
# 18:30 (the 18:25 clock missing), 8.3 km. Each interval of an arc is checked against its
# neighbouring intervals: the clock change not explained by a neighbour's rate,
# |r - r_nb| * (t1 - t0) * c, is 0.004-0.011 m (median), at most 0.54 m (99.9th percentile) and
# 0.75 m (IGS 15-min G08) on the 593,188 GPS/Galileo/BeiDou intervals of 28 daily SP3 products
# (COD, GFZ, GRG, IGS; 2024 days 034-036, 130-133); above 1 m only two intervals, both across a
# missing clock node (COD 2024-035 C48 8.3 km,
# COD 2024-131 C25 4.8 m). Above SP3_CLOCK_JUMP_TOL_M for every neighbour the interval is a clock
# discontinuity and the epochs inside it get no clock. An interval across missing clock node(s)
# is interpolated only when it could be checked (RTKLIB pephclk never bridges a missing node).
SP3_CLOCK_JUMP_TOL_M = 2.0
# A step shared by all satellites of a system is a change of the product's clock datum, which
# the receiver clock / inter-system bias absorbs, not a satellite clock jump: GFZ0MGXRAP daily
# files joined at 00:00 step every BeiDou clock by +15.15 m (2024 035/036, 14.80..15.48 m), +2.99 m
# (130/131) and -3.47 m (132/133), <= 0.61 m left per satellite after the median. On an interval
# (same two node epochs) shared by at least SP3_CLOCK_COMMON_MIN_SATS satellites of the system,
# the median signed mismatch is removed before the test.
SP3_CLOCK_COMMON_MIN_SATS = 4


def _sp3_node_step(t) -> float:
    """Nominal SP3 node spacing [s]: the median spacing of the distinct node epochs."""
    u = np.unique(np.asarray(t, dtype=np.float64))
    d = np.diff(u)
    d = d[d > 0]
    return float(np.median(d)) if d.size else float('nan')


def _sp3_locate(t, tq, max_gap, min_nodes, tol=SP3_EDGE_TOLERANCE_S):
    """Node arc serving each query time.

    ``t`` are sorted node times; an arc is a run of nodes without a spacing above ``max_gap``.
    Returns (lo, hi, reason): nodes ``t[lo:hi]`` of the arc serving the query, and reason 0 when
    served, 1 before the first node, 2 after the last node, 3 inside a gap of the nodes (or in an
    arc with fewer than ``min_nodes`` nodes). lo = hi = -1 where not served.
    """
    t = np.asarray(t, dtype=np.float64)
    tq = np.asarray(tq, dtype=np.float64)
    lo = np.full(tq.size, -1, dtype=np.int64)
    hi = np.full(tq.size, -1, dtype=np.int64)
    reason = np.full(tq.size, _SP3_SKIP_GAP, dtype=np.int8)
    if t.size == 0:
        return lo, hi, reason
    brk = np.flatnonzero(np.diff(t) > max_gap) + 1
    starts = np.concatenate(([0], brk)).astype(np.int64)
    stops = np.concatenate((brk, [t.size])).astype(np.int64)
    a = np.searchsorted(t[starts] - tol, tq, side='right') - 1
    ac = np.clip(a, 0, None)
    ok = (a >= 0) & (tq <= t[stops[ac] - 1] + tol) & ((stops[ac] - starts[ac]) >= min_nodes)
    lo[ok], hi[ok] = starts[ac[ok]], stops[ac[ok]]
    reason[ok] = 0
    reason[~ok & (tq < t[0] - tol)] = _SP3_SKIP_BEFORE
    reason[~ok & (tq > t[-1] + tol)] = _SP3_SKIP_AFTER
    return lo, hi, reason


def _sp3_clock_rate_mismatch(t, clk, max_gap):
    """Signed clock change [m] of each interval ``t[k]..t[k+1]`` not explained by the rate of the
    previous / next interval of its arc (NaN without that neighbour), and the in-arc flags.

    ``t`` sorted clock node times [s], ``clk`` their clocks [s]. An interval of duplicate node
    times (dt = 0) has no rate: it is outside the arcs and nobody's neighbour.
    """
    t = np.asarray(t, dtype=np.float64)
    clk = np.asarray(clk, dtype=np.float64)
    dt = np.diff(t)
    arc = (dt > 0) & (dt <= max_gap)
    has_prev = arc & np.concatenate(([False], arc[:-1]))
    has_next = arc & np.concatenate((arc[1:], [False]))
    # explicit masks (no floating-point warning): a rate only for dt > 0, a mismatch only where the interval and
    # its neighbour are in the arc (both then have dt > 0); NaN elsewhere
    rate = np.divide(np.diff(clk), dt, out=np.full(dt.shape, np.nan), where=dt > 0)
    r_prev = np.concatenate(([np.nan], rate[:-1]))
    r_next = np.concatenate((rate[1:], [np.nan]))
    s_prev = np.full(dt.shape, np.nan)
    s_next = np.full(dt.shape, np.nan)
    s_prev[has_prev] = (rate[has_prev] - r_prev[has_prev]) * dt[has_prev] * C_LIGHT
    s_next[has_next] = (rate[has_next] - r_next[has_next]) * dt[has_next] * C_LIGHT
    return arc, s_prev, s_next


def _sp3_clock_common_steps(series, min_sats=SP3_CLOCK_COMMON_MIN_SATS):
    """Clock steps common to the satellites of one system (see SP3_CLOCK_COMMON_MIN_SATS).

    ``series``: iterable of (t, clk, max_gap) of the system's satellites as in
    _sp3_clock_rate_mismatch. Returns {(t0, t1): (median signed mismatch against the previous,
    against the next interval) [m]} for the intervals shared by at least ``min_sats`` satellites
    (0.0 where fewer have that neighbour).
    """
    parts = []
    for t, clk, max_gap in series:
        t = np.asarray(t, dtype=np.float64)
        if t.size < 2:
            continue
        _, s_prev, s_next = _sp3_clock_rate_mismatch(t, clk, max_gap)
        parts.append(np.column_stack((t[:-1], t[1:], s_prev, s_next)))
    if not parts:
        return {}
    d = pd.DataFrame(np.concatenate(parts), columns=['t0', 't1', 'p', 'n'])
    g = d.groupby(['t0', 't1'], sort=False)[['p', 'n']]
    med = g.median().where(g.count() >= min_sats).dropna(how='all').fillna(0.0)
    return {k: (p, n) for k, p, n in zip(med.index, med['p'].to_numpy(), med['n'].to_numpy())}


def _sp3_clock_intervals(t, clk, max_gap, step, common=None):
    """Which intervals between consecutive valid clock nodes may be interpolated.

    ``t`` sorted clock node times [s], ``clk`` their clocks [s], arcs as in _sp3_locate, ``step``
    the nominal node spacing [s], ``common`` the system's steps of _sp3_clock_common_steps (with
    the same time scale as ``t``). Returns (bad, mismatch) per interval ``t[k]..t[k+1]``: bad 0
    when usable, _SP3_SKIP_CLOCK_JUMP for a clock discontinuity, _SP3_SKIP_GAP for an interval
    across missing clock node(s) without a neighbouring interval to check it against (or across
    a gap between arcs); mismatch is the clock change [m] not explained by the closer of the
    neighbouring rates, less the system's common step (NaN without a neighbour). A neighbour
    serves as reference only if it agrees with its own other neighbour (or has none), so that a
    jump in the second interval of an arc does not condemn the first. See SP3_CLOCK_JUMP_TOL_M.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2:
        return np.zeros(0, dtype=np.int8), np.zeros(0)
    dt = np.diff(t)
    arc, s_prev, s_next = _sp3_clock_rate_mismatch(t, clk, max_gap)
    if common:
        off = np.array([common.get(k, (0.0, 0.0)) for k in zip(t[:-1].tolist(), t[1:].tolist())])
        s_prev, s_next = s_prev - off[:, 0], s_next - off[:, 1]
    a_prev, a_next = np.abs(s_prev), np.abs(s_next)
    # a neighbour is a reference unless its mismatch exceeds the tolerance (NaN, no neighbour: a reference)
    big_prev = np.greater(a_prev, SP3_CLOCK_JUMP_TOL_M, out=np.zeros(a_prev.shape, dtype=bool),
                          where=~np.isnan(a_prev))
    big_next = np.greater(a_next, SP3_CLOCK_JUMP_TOL_M, out=np.zeros(a_next.shape, dtype=bool),
                          where=~np.isnan(a_next))
    ref_prev = np.concatenate(([False], ~big_prev[:-1]))
    ref_next = np.concatenate((~big_next[1:], [False]))
    mismatch = np.fmin(np.where(ref_prev, a_prev, np.nan), np.where(ref_next, a_next, np.nan))
    checked = np.isfinite(mismatch)
    bridge = dt > 1.5 * step if np.isfinite(step) and step > 0 else np.zeros(dt.size, dtype=bool)
    bad = np.zeros(dt.size, dtype=np.int8)
    bad[~arc | (bridge & ~checked)] = _SP3_SKIP_GAP
    bad[arc & checked & (mismatch > SP3_CLOCK_JUMP_TOL_M)] = _SP3_SKIP_CLOCK_JUMP
    return bad, mismatch


def _sp3_skip_message(skipped, what="SP3 interpolation"):
    """One warning text for satellite-epochs left out by the SP3 interpolators (None if none).

    ``skipped``: iterable of (sat, epoch, reason, quantity) with the reasons of _sp3_locate, and
    (sat, epoch, _SP3_SKIP_CLOCK_JUMP, 'clock', (node epoch 0, node epoch 1, mismatch [m])).
    """
    skipped = list(skipped)
    if not skipped:
        return None
    by = {r: [s for s in skipped if s[2] == r]
          for r in (_SP3_SKIP_BEFORE, _SP3_SKIP_AFTER, _SP3_SKIP_GAP, _SP3_SKIP_CLOCK_JUMP, _SP3_SKIP_NO_SAT)}
    parts = []
    if by[_SP3_SKIP_NO_SAT]:
        sats = sorted({str(s[0]) for s in by[_SP3_SKIP_NO_SAT]})
        parts.append(f"{len(by[_SP3_SKIP_NO_SAT])} from satellites absent from the SP3 file: {', '.join(sats)}")
    edge = f"SP3 nodes are not extrapolated beyond {SP3_EDGE_TOLERANCE_S:g} s"
    if by[_SP3_SKIP_AFTER]:
        ep = sorted({pd.Timestamp(s[1]) for s in by[_SP3_SKIP_AFTER]})
        parts.append(f"{len(by[_SP3_SKIP_AFTER])} after the last SP3 node ({ep[0]} - {ep[-1]}; {edge}): "
                     "add the next-day SP3 file")
    if by[_SP3_SKIP_BEFORE]:
        ep = sorted({pd.Timestamp(s[1]) for s in by[_SP3_SKIP_BEFORE]})
        parts.append(f"{len(by[_SP3_SKIP_BEFORE])} before the first SP3 node ({ep[0]} - {ep[-1]}; {edge}): "
                     "add the previous-day SP3 file")
    if by[_SP3_SKIP_GAP]:
        sats = sorted({f"{s[0]} ({s[3]})" for s in by[_SP3_SKIP_GAP]})
        parts.append(f"{len(by[_SP3_SKIP_GAP])} in gaps of missing SP3 positions/clocks: {', '.join(sats)}")
    if by[_SP3_SKIP_CLOCK_JUMP]:
        # one entry per node span, listing the satellites that jump on it
        spans = {}
        for s in by[_SP3_SKIP_CLOCK_JUMP]:
            spans.setdefault((pd.Timestamp(s[4][0]), pd.Timestamp(s[4][1])), {})[s[0]] = float(s[4][2])
        items = ", ".join(f"{t0} - {t1} [" + ", ".join(
            f"{sv} {m / 1e3:.1f} km" if m >= 1e3 else f"{sv} {m:.1f} m" for sv, m in sorted(svs.items())) + "]"
            for (t0, t1), svs in sorted(spans.items()))
        parts.append(f"{len(by[_SP3_SKIP_CLOCK_JUMP])} in a clock discontinuity (clock change between two "
                     f"nodes off the neighbouring clock rates by more than {SP3_CLOCK_JUMP_TOL_M:g} m, not "
                     f"interpolated): {items}")
    return f"{what}: {len(skipped)} satellite-epochs left out: " + "; ".join(parts) + "."


def _sp3_merge_duplicate_nodes(df, sv_col, t_col, blk_col, pos_cols, clk_col, tol):
    """One node per (satellite, epoch) where SP3 files overlap (24:00 of a day = 00:00 of the next).

    ``blk_col`` tells the files apart. The record with both a position and a clock wins, so that
    orbit and clock come from one solution (a CODE 24:00 node has no clock, its 00:00 node of the
    next day has both); between two equal records the file that starts at the epoch wins (the
    later file, as RTKLIB combpeph does), which keeps day D's own 00:00 node in day D. A quantity
    the winner lacks is taken from the other record. If the two positions differ by more than
    ``tol`` (frame units) the files disagree on the orbit (an untracked satellite carried by one
    of them; daily solutions otherwise meet within cm) and the position node is dropped.
    The result does not depend on the order in which the files are listed.
    Returns (frame without ``blk_col``, list of (sv, epoch, distance) of dropped positions).
    """
    dup = df.duplicated([sv_col, t_col], keep=False).to_numpy()
    if not dup.any():
        return df.drop(columns=blk_col), []
    t_blk = df.groupby([sv_col, blk_col], sort=False)[t_col]
    d = df.loc[dup].copy()
    d['__bfirst'] = t_blk.transform('min').loc[dup]
    d['__blast'] = t_blk.transform('max').loc[dup]
    full = d[pos_cols].notna().all(axis=1)
    if clk_col is not None:
        full &= d[clk_col].notna()
    d['__full'] = full
    d = d.sort_values([sv_col, t_col, '__full', '__bfirst', '__blast'],
                      ascending=[True, True, False, False, False], kind='mergesort')
    g = d.groupby([sv_col, t_col], sort=False)
    merged = g.first()  # first non-missing value per column, in the order of preference
    pos = g[pos_cols]
    spread = np.sqrt(((pos.max() - pos.min()) ** 2).sum(axis=1)).reindex(merged.index)
    bad = spread > tol
    merged.loc[bad, pos_cols] = np.nan
    dropped = [(sv, t, float(s)) for (sv, t), s in spread[bad].items()]
    merged = merged.reset_index()
    out = pd.concat([df.loc[~dup], merged], ignore_index=True)
    return out[[c for c in df.columns if c != blk_col]], dropped


def _sp3_boundary_message(dropped, scale_m=1.0):
    """Warning text for position nodes dropped by _sp3_merge_duplicate_nodes (None if none)."""
    if not dropped:
        return None
    items = ", ".join(f"{sv} {pd.Timestamp(t)} ({d * scale_m:.2f} m)" for sv, t, d in dropped)
    return (f"SP3 files disagree on the position at an epoch both carry (more than "
            f"{SP3_BOUNDARY_TOL_M:g} m): {items}; that position node is not used.")


def _sp3_cheb_fit_eval(toc_window, x_window, y_window, z_window, epoch_time, method, delta_t):
    """Chebyshev polynomial through the window nodes [km]: position [km] and velocity [m/s]."""
    t_min = toc_window.min()
    t_max = toc_window.max()
    # Normalizacja czasu do przedziału [0, 2] (zachowujemy proporcje)
    toc_normalized = 2 * (toc_window - t_min) / (t_max - t_min)
    epoch_time_normalized = 2 * (epoch_time - t_min) / (t_max - t_min)
    epoch_time_plus_dt_normalized = 2 * ((epoch_time + delta_t) - t_min) / (t_max - t_min)
    epoch_time_minus_dt_normalized = 2 * ((epoch_time - delta_t) - t_min) / (t_max - t_min)

    degree = len(toc_normalized) - 1

    if method == 'chebyshev':
        # Dopasowanie wielomianu Czebyszewa
        cheb_x = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, x_window, deg=degree)
        cheb_y = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, y_window, deg=degree)
        cheb_z = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, z_window, deg=degree)
    else:
        raise ValueError("Unsupported interpolation method.")
    pos = (cheb_x(epoch_time_normalized), cheb_y(epoch_time_normalized), cheb_z(epoch_time_normalized))
    # Obliczenie prędkości metodą różnic centralnych (skalujemy do metrów na sekundę)
    vel = tuple((f(epoch_time_plus_dt_normalized) - f(epoch_time_minus_dt_normalized)) / (2 * delta_t) * 1000
                for f in (cheb_x, cheb_y, cheb_z))
    return pos, vel


class _SP3SatNodes:
    """Valid SP3 nodes of one satellite: positions and clocks each over their own valid nodes."""

    def __init__(self, sat_df, step, min_window_size, clk_common=None):
        toc = sat_df['toc'].to_numpy(dtype=np.float64)  # ciągły czas (w sekundach)
        xyz = sat_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        clk = sat_df['clk'].to_numpy(dtype=np.float64)
        self.step = self.node_spacing(toc, step)
        self.max_gap = SP3_GAP_FACTOR * self.step
        p = np.isfinite(xyz).all(axis=1)
        c = np.isfinite(clk)
        self.tp, self.xyz = toc[p], xyz[p]
        self.tc, self.clk = toc[c], clk[c]
        self.ec = sat_df['epoch'].to_numpy()[c]
        # clock intervals not to be interpolated (clock discontinuity, unchecked missing node);
        # ``clk_common``: the system's common clock steps (see SP3_CLOCK_COMMON_MIN_SATS)
        self.clk_bad, self.clk_mis = _sp3_clock_intervals(self.tc, self.clk * 1e-6, self.max_gap, self.step,
                                                          clk_common)
        self.min_window_size = min_window_size
        self.min_arc = max(int(min_window_size), SP3_MIN_ARC_NODES)
        self._loc = None

    @staticmethod
    def node_spacing(toc, step):
        """Node spacing [s]: ``step`` if valid, else the satellite's own, else 300 s."""
        step = float(step) if step is not None and np.isfinite(step) and step > 0 else _sp3_node_step(toc)
        return step if np.isfinite(step) else 300.0

    @staticmethod
    def common_clock_steps(sat_data, step):
        """{system: _sp3_clock_common_steps} over the satellites of ``sat_data`` {sat: frame}."""
        series = {}
        for sat, sat_df in sat_data.items():
            toc = sat_df['toc'].to_numpy(dtype=np.float64)
            clk = sat_df['clk'].to_numpy(dtype=np.float64)
            c = np.isfinite(clk)
            max_gap = SP3_GAP_FACTOR * _SP3SatNodes.node_spacing(toc, step)
            series.setdefault(str(sat)[:1], []).append((toc[c], clk[c] * 1e-6, max_gap))
        return {s: _sp3_clock_common_steps(v) for s, v in series.items()}

    def locate(self, epoch_times):
        pl, ph, pr = _sp3_locate(self.tp, epoch_times, self.max_gap, self.min_arc)
        cl, ch, cr = _sp3_locate(self.tc, epoch_times, self.max_gap, 2)
        self._loc = (pl, ph, pr, cl, ch, cr)

    def _clock_skip(self, k, epoch_time):
        """Skip reason for an epoch off the nodes on clock interval ``k``, or None if usable."""
        bad = self.clk_bad[k]
        if not bad or epoch_time == self.tc[k] or epoch_time == self.tc[k + 1]:
            return None
        if bad != _SP3_SKIP_CLOCK_JUMP:
            return int(bad), 'clock'
        return _SP3_SKIP_CLOCK_JUMP, 'clock', (self.ec[k], self.ec[k + 1], float(self.clk_mis[k]))

    def evaluate(self, i, epoch_time, epoch_times, method, window_size, min_window_size, delta_t):
        """Interpolated position/clock row at epoch ``i`` of ``epoch_times``, or (None, why)."""
        if self._loc is None:
            self.locate(epoch_times)
        pl, ph, pr, cl, ch, cr = self._loc
        if pr[i]:
            return None, (int(pr[i]), 'position')
        if cr[i]:
            return None, (int(cr[i]), 'clock')
        toc = self.tp[pl[i]:ph[i]]
        xyz = self.xyz[pl[i]:ph[i]]
        # Chebyshev window: the window_size - 1 nodes centred on the epoch (+-(window_size-1)/2 node
        # steps). Near an arc end (or a missing node) the time window holds fewer nodes; then the
        # same number of nodes is taken from the arc, shifted towards its inside, instead of a
        # shorter one-sided window - with 15-min nodes a degree-5 fit in the last interval is off
        # by up to 36 m (7 m at the 95th percentile), the 11-node window by 1 cm (0.9 m max).
        half_window = (window_size - 1) * self.step / 2
        indices = np.where((toc >= epoch_time - half_window) & (toc < epoch_time + half_window))[0]
        n_win = window_size - 1
        if len(indices) < n_win:
            n = min(n_win, len(toc))
            if n < min_window_size:
                return None, (_SP3_SKIP_GAP, 'position')
            left = int(np.clip(np.searchsorted(toc, epoch_time) - n // 2, 0, len(toc) - n))
            indices = np.arange(left, left + n)
        (ix, iy, iz), (vx, vy, vz) = _sp3_cheb_fit_eval(toc[indices], xyz[indices, 0], xyz[indices, 1],
                                                         xyz[indices, 2], epoch_time, method, delta_t)

        # Interpolacja korekty zegara – liniowa interpolacja pomiędzy dwoma najbliższymi
        # poprawnymi węzłami zegara (w obrębie tolerancji na krańcach łuku – ekstrapolacja liniowa)
        tc = self.tc[cl[i]:ch[i]]
        cc = self.clk[cl[i]:ch[i]]
        idx = np.searchsorted(tc, epoch_time)
        if idx == 0:
            idx0, idx1 = 0, 1
        elif idx == len(tc):
            idx0, idx1 = len(tc) - 2, len(tc) - 1
        else:
            idx0, idx1 = idx - 1, idx
        # no clock inside a clock discontinuity (see SP3_CLOCK_JUMP_TOL_M); on a node the node value
        why = self._clock_skip(cl[i] + idx0, epoch_time)
        if why is not None:
            return None, why
        t0, t1 = tc[idx0], tc[idx1]
        clk0, clk1 = cc[idx0], cc[idx1]
        if epoch_time == t1:
            interpolated_clk = clk1
        elif epoch_time == t0:
            interpolated_clk = clk0
        else:
            interpolated_clk = clk0 + (clk1 - clk0) * (epoch_time - t0) / (t1 - t0) if t1 != t0 else clk0
        interpolated_clk *= 1e-6  # przeskalowanie jednostek

        c = 299792458
        r_vec = np.array([ix, iy, iz]) * 1000  # m
        v_vec = np.array([vx, vy, vz])
        delta_t_rel = -2 * np.dot(r_vec, v_vec) / c ** 2  # w sekundach
        return {
            'x': ix, 'y': iy, 'z': iz,
            'clk': interpolated_clk + delta_t_rel,
            'vx': vx, 'vy': vy, 'vz': vz,
            'dt_rel': delta_t_rel,
            'clk_raw': interpolated_clk,
        }, None


class SP3InterpolatorOptimized:
    def __init__(self, sp3_dataframe):
        """
        Inicjalizacja z danymi SP3.
        """
        self.sp3_df = sp3_dataframe.copy()
        self.prepare_data()

    def prepare_data(self):
        # Konwersja kolumny 'epoch' do daty
        self.sp3_df.loc[:,'epoch'] = pd.to_datetime(self.sp3_df['epoch'], utc=False)
        self.sp3_df['gps_week'], self.sp3_df['tow'] = zip(*self.sp3_df['epoch'].apply(self.datetime2gpsweek_and_tow))
        min_gps_week = self.sp3_df['gps_week'].min()
        self.sp3_df['continuous_time'] = (self.sp3_df['gps_week'] - min_gps_week) * 604800 + self.sp3_df['tow']
        self.sp3_df['toc'] = self.sp3_df['continuous_time']
        self.satellites = self.sp3_df['sat'].unique()
        # node spacing of the product (5 min, 15 min, 30 s ...) read from the node epochs
        self.node_step = _sp3_node_step(self.sp3_df['toc'].to_numpy(dtype=np.float64))
        self.sat_data = {}
        for sat in self.satellites:
            sat_df = self.sp3_df[self.sp3_df['sat'] == sat].sort_values('toc')
            self.sat_data[sat] = sat_df
        # clock steps shared by a system's satellites (clock datum change, e.g. joined daily files)
        self.clk_common = _SP3SatNodes.common_clock_steps(self.sat_data, self.node_step)

    def datetime2gpsweek_and_tow(self, dt):
        gps_epoch = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
        delta = dt - gps_epoch
        delta_seconds = delta.total_seconds()
        gps_week = int(delta_seconds // 604800)
        tow = delta_seconds % 604800
        return gps_week, tow

    def interpolate(self, epochs, method='chebyshev', window_size=12, min_window_size=5, delta_t=0.5e-6):
        """
        Interpolates the satellite positions at the given epochs, one satellite after the other.

        Runs in the calling process: no process pool, so no child process is
        forked or spawned (a fork of a process that has loaded PROJ crashed in PROJ's atfork handler, and
        ``multiprocessing``'s 'spawn' forks as well on macOS), and the number of processes is bounded by the
        caller. A thread pool does not help: the per-epoch Chebyshev fits are GIL-bound (measured on a COD 30 s
        day of GPS: 18.0 s sequential, 21.6 s with 4 threads, 9.9 s with the former 8-process pool including its
        start-up). The results are identical to the former pool's (same worker function per satellite).
        """
        interpolated_positions = []
        # Przeliczenie epok na GPS week i ciągły czas
        epoch_weeks, epoch_tows = zip(*[self.datetime2gpsweek_and_tow(epoch) for epoch in epochs])
        min_gps_week = self.sp3_df['gps_week'].min()
        epoch_times = np.array([(week - min_gps_week) * 604800 + tow for week, tow in zip(epoch_weeks, epoch_tows)])

        # Argumenty interpolacji – każdy satelita osobno (liczony kolejno w tym procesie)
        tasks = []
        for sat in self.satellites:
            sat_df = self.sat_data[sat]
            if sat_df.empty:
                continue
            task = {
                'sat': sat,
                'sat_df': sat_df,
                'epoch_times': epoch_times,
                'epochs': epochs,
                'method': method,
                'window_size': window_size,
                'min_window_size': min_window_size,
                'delta_t': delta_t,
                'step': self.node_step,
                'clk_common': self.clk_common.get(str(sat)[:1]),
            }
            tasks.append(task)

        # one satellite after the other, in the calling process (no child process; see the docstring)
        results = [_process_satellite_interpolation(task) for task in tasks]

        # Scal wyniki – results to lista list słowników
        skipped = []
        for sat_results, sat_skipped in results:
            interpolated_positions.extend(sat_results)
            skipped.extend(sat_skipped)
        msg = _sp3_skip_message(skipped)
        if msg:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        interpolated_df = pd.DataFrame(interpolated_positions,
                                       columns=None if interpolated_positions else _SP3_INTERP_COLUMNS)
        interpolated_df.set_index(['sv', 'epoch'], inplace=True)
        return interpolated_df

    def include_adjacent_data(self, prev_sp3_df=None, next_sp3_df=None):
        frames = []
        for k, f in enumerate((prev_sp3_df, self.sp3_df, next_sp3_df)):
            if f is None:
                continue
            f = f.copy()
            f['epoch'] = pd.to_datetime(f['epoch'])
            f['__file'] = k
            frames.append(f)
        combined_df = pd.concat(frames, ignore_index=True)
        # Day-boundary epoch present in two files (24:00 of one day = 00:00 of the next): one node,
        # chosen by content (see _sp3_merge_duplicate_nodes), so whichever file is the nominal
        # day the boundary epoch gets the same values. Positions are in km here.
        self.sp3_df, dropped = _sp3_merge_duplicate_nodes(combined_df, 'sat', 'epoch', '__file',
                                                          ['x', 'y', 'z'], 'clk', SP3_BOUNDARY_TOL_M * 1e-3)
        msg = _sp3_boundary_message(dropped, scale_m=1e3)
        if msg:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        self.prepare_data()

    def run(self, epochs, method, prev_sp3_df=None, next_sp3_df=None):
        if (prev_sp3_df is not None) or (next_sp3_df is not None):
            self.include_adjacent_data(prev_sp3_df=prev_sp3_df,next_sp3_df=next_sp3_df)
        interpolated_positions = self.interpolate(epochs=epochs, method=method)
        interpolated_positions = interpolated_positions.swaplevel(0, 1)
        interpolated_positions[['x', 'y', 'z']] = interpolated_positions[['x', 'y', 'z']].apply(lambda x: x * 1000)
        return interpolated_positions

class CrdWrapper:
    """
    This class performs:
    - azimuth, elevation computation
    - earth rotation coordinates correction
    - emission time computation
    It is designed for other satellite coordinates interpolations classes to inherit from
    """
    def __init__(self, obs:pd.DataFrame, flh, xyz_a, mode, epochs:Optional=None):
        self.obs = obs
        self.epochs = epochs
        self.flh = flh
        self.xyz_a = xyz_a
        self.mode = mode

    def elevation_azimuth(self):
        df = self.obs.copy()
        lat, lon, alt = self.flh[0], self.flh[1], self.flh[2]
        xyz_point = np.asarray(self.xyz_a)
        flh_point = np.deg2rad([lat, lon, alt])
        fi, la, h = flh_point[0], flh_point[1], flh_point[2]

        e = np.array([
            - np.sin(la),
            np.cos(la),
            0
        ])
        n = np.array([
            -np.cos(la) * (np.sin(fi)),
            -np.sin(la) * np.sin(fi),
            np.cos(fi)
        ])

        u = np.array([
            np.cos(la) * np.cos(fi),
            np.sin(la) * np.cos(fi),
            np.sin(fi)
        ])

        xyz = df[['xe', 'ye', 'ze']].to_numpy()
        dist = calculate_distance(xyz, xyz_point)
        p = np.array([
            (xyz[:, 0] - xyz_point[0]) / dist,
            (xyz[:, 1] - xyz_point[1]) / dist,
            (xyz[:, 2] - xyz_point[2]) / dist
        ]).T

        ev = np.arcsin(np.dot(p, u))
        az = np.arctan2(np.dot(p, e), np.dot(p, n))
        az = (az + 2 * np.pi) % (2 * np.pi)
        df['az'] = np.rad2deg(az)
        df['ev'] = np.rad2deg(ev)
        return df

    # def correct_crd(self):
    #     xyz= self.obs[['x', 'y', 'z']].to_numpy()
    #     xyze = np.array(list(map(lambda row: correct_sat_coordinates(row,xyz_a=self.xyz_a),xyz)))
    #     xyze = pd.DataFrame(index=self.obs.index,data={'xe':xyze[:,0],'ye':xyze[:,1],'ze':xyze[:,2]})
    #     self.obs[['xe_ur', 'ye_ur','ze_ur']] = self.obs[['x', 'y', 'z']].copy()
    #     self.obs[['xe', 'ye', 'ze']] = xyze
    #     return self.obs
    def correct_crd(self):
        xyz = self.obs[['x', 'y', 'z']].to_numpy(dtype=np.float64, copy=False)
        a = np.asarray(self.xyz_a, dtype=np.float64)

        dx = xyz[:, 0] - a[0]
        dy = xyz[:, 1] - a[1]
        dz = xyz[:, 2] - a[2]

        c = 299792458.0
        omega = 7.2921159e-5

        dT = np.sqrt(dx * dx + dy * dy + dz * dz) / c
        omw = dT * omega

        cos0 = np.cos(omw)
        sin0 = np.sin(omw)

        xe = cos0 * xyz[:, 0] + sin0 * xyz[:, 1]
        ye = -sin0 * xyz[:, 0] + cos0 * xyz[:, 1]
        ze = xyz[:, 2]

        self.obs[['xe_ur', 'ye_ur', 'ze_ur']] = self.obs[['x', 'y', 'z']].to_numpy(copy=True)
        self.obs[['xe', 'ye', 'ze']] = np.column_stack((xe, ye, ze))
        return self.obs

class CustomWrapper(CrdWrapper):
    def __init__(self, obs, epochs, flh, xyz_a, mode):
        super().__init__(obs, epochs, flh, xyz_a, mode)
        self.obs = obs
        self.epochs = epochs
        self.flh = flh
        self.xyz_a = xyz_a
        self.mode = mode

    def run(self):
        self.obs = self.correct_crd()
        self.obs = self.elevation_azimuth()
        return self.obs




def make_ionofree(obs_df,  mode,sys='G'):
    def first_col(prefix):
        return [c for c in obs_df.columns if c.startswith(prefix)][0]

    signals = mode_signals(mode)
    if len(signals) == 1:
        obs_df['P3'] = obs_df[first_col(signal_spec(signals[0]).code_prefix)]
        return

    sig1, sig2 = signals
    k1, k2 = mode_ionosphere_free_coefficients(mode)
    c1 = first_col(signal_spec(sig1).code_prefix)
    c2 = first_col(signal_spec(sig2).code_prefix)
    obs_df['P3'] = k1 * obs_df[c1] - k2 * obs_df[c2]


def eccentric_anomaly(mk, e, pi):
    """
    This function is to calculate eccentric anomaly iteratively
    :param mk: mean anomaly at the epoch t
    :param e: eccentricity 'e' from nav message
    :param pi: pi number
    :return: eccentric anomaly [rad] <0, 2pi>
    """
    ek = mk
    for i in range(14):
        ek_old = ek
        ek = mk + e * np.sin(ek)
        diff = (ek - ek_old) #% (2 * pi)
        if abs(diff) < 1e-8:
            return ek% (2 * pi)
    return None


def _eccentric_anomaly_vec(mk, e, max_iter=30, tol=1e-14):
    """Solve Kepler's equation E - e sin E = M (rad) by Newton iteration.

    Every element iterates until its own correction is below ``tol``: a stop
    decided for the whole batch left each epoch with a residual of up to 1e-10
    rad (mm on the orbit) that depended on which other epochs were computed in
    the same batch, so a time-limited run differed from the daily one.
    """
    mk = np.asarray(mk, dtype=float)
    e = np.broadcast_to(np.asarray(e, dtype=float), mk.shape)
    ek = mk.copy()
    active = np.ones(mk.shape, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        ea, ea_e = ek[idx], e[idx]
        step = (ea - ea_e * np.sin(ea) - mk[idx]) / (1.0 - ea_e * np.cos(ea))
        ek[idx] = ea - step
        active[idx] = np.abs(step) >= tol
    return np.mod(ek, 2 * pi)


def _numpy_broadcast_interpolation_batch(message_toc, observation_toc, messages):
    mi = 3.986005e14
    ome = 7.2921151467e-05
    c = 299792458.0

    a0 = messages[:, 0]
    a1 = messages[:, 1]
    a2 = messages[:, 2]

    t_toc = observation_toc - message_toc
    dT = np.where(t_toc < -302400.0, t_toc + WEEK, np.where(t_toc > 302400.0, t_toc - WEEK, t_toc))
    dt = a0 + a1 * dT + a2 * dT ** 2
    dt_dot = a1 + 2.0 * a2 * dT

    toe = messages[:, 11]
    t_toe = observation_toc - toe
    tk = np.where(t_toe > 302400.0, t_toe - WEEK, np.where(t_toe < -302400.0, t_toe + WEEK, t_toe))

    sqrta = messages[:, 10]
    a = sqrta ** 2
    n0 = np.sqrt(mi / a ** 3)
    n = n0 + messages[:, 5]
    mk = np.mod(messages[:, 6] + n * tk, 2 * pi)
    e = messages[:, 8]
    ek = _eccentric_anomaly_vec(mk, e)

    v = np.arctan2(np.sqrt(1.0 - e ** 2) * np.sin(ek), np.cos(ek) - e)
    u = np.mod(messages[:, 17] + v, 2 * pi)

    cus = messages[:, 9]
    cuc = messages[:, 7]
    duk = cus * np.sin(2.0 * u) + cuc * np.cos(2.0 * u)

    crs = messages[:, 4]
    crc = messages[:, 16]
    drk = crs * np.sin(2.0 * u) + crc * np.cos(2.0 * u)

    cis = messages[:, 14]
    cic = messages[:, 12]
    idot = messages[:, 19]
    dik = cis * np.sin(2.0 * u) + cic * np.cos(2.0 * u) + idot * tk

    uk = u + duk
    rk = a * (1.0 - e * np.cos(ek)) + drk
    ik = messages[:, 15] + dik

    omk = np.mod(messages[:, 13] + (messages[:, 18] - ome) * tk - ome * toe, 2 * pi)
    xi = rk * np.cos(uk)
    eta = rk * np.sin(uk)

    cos_omk = np.cos(omk)
    sin_omk = np.sin(omk)
    cos_ik = np.cos(ik)
    sin_ik = np.sin(ik)

    x = xi * cos_omk - eta * cos_ik * sin_omk
    y = xi * sin_omk + eta * cos_ik * cos_omk
    z = eta * sin_ik

    em_dot = 1.0 / (1.0 - e * np.cos(ek))
    v_dot = np.sqrt((1.0 + e) / (1.0 - e)) * (1.0 / (np.cos(ek / 2.0) ** 2)) * (
            1.0 / (1.0 + np.tan(v / 2.0) ** 2)) * em_dot * n

    om_dot = messages[:, 18]
    u_dot = v_dot + 2.0 * v_dot * (cus * np.cos(2.0 * u) - cuc * np.sin(2.0 * u))
    small_om_dot = om_dot - ome
    i_dot = idot + 2.0 * v_dot * (cis * np.cos(2.0 * u) - cic * np.sin(2.0 * u))
    r_dot = a * e * np.sin(ek) * em_dot * n + 2.0 * v_dot * (crs * np.cos(2.0 * u) - crc * np.sin(2.0 * u))

    xi_dot = r_dot * np.cos(uk) - rk * np.sin(uk) * u_dot
    eta_dot = r_dot * np.sin(uk) + rk * np.cos(uk) * u_dot

    x_dot = ((cos_omk * xi_dot - cos_ik * sin_omk * eta_dot - xi * sin_omk * small_om_dot)
             - eta * cos_ik * cos_omk * small_om_dot
             + eta * sin_ik * sin_omk * i_dot)
    y_dot = ((sin_omk * xi_dot + cos_ik * cos_omk * eta_dot + xi * cos_omk * small_om_dot)
             - eta * cos_ik * sin_omk * small_om_dot
             - eta * sin_ik * cos_omk * i_dot)
    z_dot = sin_ik * eta_dot + eta * np.cos(ik) * i_dot

    dt_rel = -2.0 * (x * x_dot + y * y_dot + z * z_dot) / c ** 2
    dt_sat = dt + dt_rel
    return np.column_stack([x, y, z, dt_sat, x_dot, y_dot, z_dot, dt_dot])


def _gal_numpy_broadcast_interpolation_batch(message_toc, observation_toc, messages):
    mi = 3.986004418e14
    ome = 7.2921151467e-05
    c = 299792458.0

    a0 = messages[:, 0]
    a1 = messages[:, 1]
    a2 = messages[:, 2]

    t_toc = observation_toc - message_toc
    dT = np.where(t_toc < -302400.0, t_toc + WEEK, np.where(t_toc > 302400.0, t_toc - WEEK, t_toc))
    dt = a0 + a1 * dT + a2 * dT ** 2
    dt_dot = a1 + 2.0 * a2 * dT

    toe = messages[:, 11]
    t_toe = observation_toc - toe
    tk = np.where(t_toe > 302400.0, t_toe - WEEK, np.where(t_toe < -302400.0, t_toe + WEEK, t_toe))

    sqrta = messages[:, 10]
    a = sqrta ** 2
    n0 = np.sqrt(mi / a ** 3)
    n = n0 + messages[:, 5]
    mk = np.mod(messages[:, 6] + n * tk, 2 * pi)
    e = messages[:, 8]
    ek = _eccentric_anomaly_vec(mk, e)

    v = np.arctan2(np.sqrt(1.0 - e ** 2) * np.sin(ek), np.cos(ek) - e)
    u = np.mod(messages[:, 17] + v, 2 * pi)

    cus = messages[:, 9]
    cuc = messages[:, 7]
    duk = cus * np.sin(2.0 * u) + cuc * np.cos(2.0 * u)

    crs = messages[:, 4]
    crc = messages[:, 16]
    drk = crs * np.sin(2.0 * u) + crc * np.cos(2.0 * u)

    cis = messages[:, 14]
    cic = messages[:, 12]
    idot = messages[:, 19]
    dik = cis * np.sin(2.0 * u) + cic * np.cos(2.0 * u) + idot * tk

    uk = u + duk
    rk = a * (1.0 - e * np.cos(ek)) + drk
    ik = messages[:, 15] + dik

    omk = np.mod(messages[:, 13] + (messages[:, 18] - ome) * tk - ome * toe, 2 * pi)
    xi = rk * np.cos(uk)
    eta = rk * np.sin(uk)

    cos_omk = np.cos(omk)
    sin_omk = np.sin(omk)
    cos_ik = np.cos(ik)
    sin_ik = np.sin(ik)

    x = xi * cos_omk - eta * cos_ik * sin_omk
    y = xi * sin_omk + eta * cos_ik * cos_omk
    z = eta * sin_ik

    em_dot = 1.0 / (1.0 - e * np.cos(ek))
    v_dot = np.sqrt((1.0 + e) / (1.0 - e)) * (1.0 / (np.cos(ek / 2.0) ** 2)) * (
            1.0 / (1.0 + np.tan(v / 2.0) ** 2)) * em_dot * n

    om_dot = messages[:, 18]
    u_dot = v_dot + 2.0 * v_dot * (cus * np.cos(2.0 * u) - cuc * np.sin(2.0 * u))
    small_om_dot = om_dot - ome
    i_dot = idot + 2.0 * v_dot * (cis * np.cos(2.0 * u) - cic * np.sin(2.0 * u))
    r_dot = a * e * np.sin(ek) * em_dot * n + 2.0 * v_dot * (crs * np.cos(2.0 * u) - crc * np.sin(2.0 * u))

    xi_dot = r_dot * np.cos(uk) - rk * np.sin(uk) * u_dot
    eta_dot = r_dot * np.sin(uk) + rk * np.cos(uk) * u_dot

    x_dot = ((cos_omk * xi_dot - cos_ik * sin_omk * eta_dot - xi * sin_omk * small_om_dot)
             - eta * cos_ik * cos_omk * small_om_dot
             + eta * sin_ik * sin_omk * i_dot)
    y_dot = ((sin_omk * xi_dot + cos_ik * cos_omk * eta_dot + xi * cos_omk * small_om_dot)
             - eta * cos_ik * sin_omk * small_om_dot
             - eta * sin_ik * cos_omk * i_dot)
    z_dot = sin_ik * eta_dot + eta * np.cos(ik) * i_dot

    dt_rel = -2.0 * (x * x_dot + y * y_dot + z * z_dot) / c ** 2
    dt_sat = dt + dt_rel
    return np.column_stack([x, y, z, dt_sat, x_dot, y_dot, z_dot, dt_dot])


def numpy_broadcast_interpolation(message_toc, observation_toc, message, with_rel=True, rel_sep=False):
    """
    This function interpolates satellite position at given epoch using GPS broadcast navigation message
    :param message_toc: toc epoch of broadcast message
    :param observation_toc: toc epoch at which interpolation is done
    :param message: broadcast message
    :return: X,Y,Z dte (coordinates + SV clock offset correction)
    ['SVclockBias'          0,
    SVclockDrift',          1
    'SVclockDriftRate',     2
    'IODE',                 3
    'Crs',                  4
    'DeltaN',               5
    'M0',                   6
    'Cuc',                  7
    'Eccentricity',         8
    'Cus',                  9
    'sqrtA',                10
    'Toe',                  11
    'Cic',                  12
    'Omega0',               13
    'Cis',                  14
    'Io',                   15
    'Crc',                  16
    'omega',                17
    'OmegaDot',             18
    'IDOT',                 19
    'CodesL2',              20
    'GPSWeek',              21
    'L2Pflag',              22
    'SVacc',                23
    'health',               24
    'TGD',                  25
    'IODC',                 26
    'TransTime']            27
    """

    # constants
    mi = 3.986005e14  # potencjal ziemski
    # pi = 3.1415926535898  # rad
    ome = 7.2921151467e-05
    c = 299792458  # m/s
    # SV clock correction (GPST to UTC)
    a0 = message[0]
    a1 = message[1]
    a2 = message[2]
    t_toc = observation_toc - message_toc
    if t_toc < -302400:
        dT = t_toc + 604800
    elif t_toc > 302400:
        dT = t_toc - 604800
    else:
        dT = t_toc
    dt = a0 + a1 * dT + a2 * dT ** 2
    # if message[-1] =='G27':
    #     print(f'{message[-1], message[0:3], observation_toc, message_toc, dT}')
    dt_dot = a1 + 2 * a2 * dT
    # ephemeris reference epoch
    toe = message[11]
    t_toe = observation_toc - toe
    if t_toe > 302400:
        tk = t_toe - 604800
    elif t_toe < -302400:
        tk = t_toe + 604800
    else:
        tk = t_toe
    # Satellite orbit major axis
    sqrta = message[10]
    a = sqrta ** 2
    # Satellite average motion
    n0 = np.sqrt(mi / a ** 3)
    # corrected average motion
    dn = message[5]
    n = n0 + dn
    # mean anomaly at tk epoch
    m0 = message[6]
    mk = m0 + n * tk  # should be <0, 2pi>
    mk = mk % (2 * pi)
    # Eccentric anomaly
    e = message[8]
    ek = eccentric_anomaly(mk, e, pi)
    ek = ek #% (2 * pi)
    # True anomaly
    v = math.atan2((np.sqrt(1 - e ** 2) * np.sin(ek)), (np.cos(ek) - e))
    # latitude argument
    u = message[17] + v
    u = u % (2 * pi)
    # latitude argument correction
    cus = message[9]
    cuc = message[7]
    duk = cus * np.sin(2 * u) + cuc * np.cos(2 * u)
    # leading beam correction
    crs = message[4]
    crc = message[16]
    drk = crs * np.sin(2 * u) + crc * np.cos(2 * u)
    # orbit inclination correction
    cis = message[14]
    cic = message[12]
    idot = message[19]
    dik = cis * np.sin(2 * u) + cic * np.cos(2 * u) + idot * tk
    # corrected latitude argument
    uk = u + duk
    # corrected leading beam
    rk = a * (1 - e * np.cos(ek)) + drk
    # corrected orbit inclination angle
    i0 = message[15]
    ik = i0 + dik
    # corrected length of ascending orbit node
    om0 = message[13]
    omv = message[18]
    omk = om0 + (omv - ome) * tk - ome * toe
    omk = omk % (2 * pi)
    # coordinates of the satellite in the plane of orbit
    xi = rk * np.cos(uk)
    eta = rk * np.sin(uk)
    # geocentric coordinates of the satellite

    x = xi * np.cos(omk) - eta * np.cos(ik) * np.sin(omk)
    y = xi * np.sin(omk) + eta * np.cos(ik) * np.cos(omk)
    z = eta * np.sin(ik)

    # The product of the mean and eccentric anomaly velocity
    em_dot = 1 / (1 - e * np.cos(ek))
    # True anomaly velocity
    v_dot = np.sqrt((1 + e) / (1 - e)) * (1 / (np.cos(ek / 2) ** 2)) * (1 / (1 + np.tan(v / 2) ** 2)) * em_dot * n
    # argument of latitude velocity
    # cus = cus
    # cuc = cuc
    om_dot = message[18]
    u_dot = v_dot + 2 * v_dot * (cus * np.cos(2 * u) - cuc * np.sin(2 * u))
    small_om_dot = om_dot - ome
    # inclination rate
    cis = cis
    cic = cic
    i_dot = idot + 2 * v_dot * (cis * np.cos(2 * u) - cic * np.sin(2 * u))
    # leading beam rate
    r_dot = a * e * np.sin(ek) * em_dot * n + 2 * v_dot * (crs * np.cos(2 * u) - crc * np.sin(2 * u))
    # satellite velocity in the plane of an orbit
    xi_dot = r_dot * np.cos(uk) - rk * np.sin(uk) * u_dot
    eta_dot = r_dot * np.sin(uk) + rk * np.cos(uk) * u_dot
    # satellite geocentric velocity

    x_dot = (((np.cos(omk) * xi_dot - np.cos(ik) * np.sin(omk) * eta_dot
               - xi * np.sin(omk) * small_om_dot)
              - eta * np.cos(ik) * np.cos(omk) * small_om_dot)
             + eta * np.sin(ik) * np.sin(omk) * i_dot)

    y_dot = (((np.sin(omk) * xi_dot + np.cos(ik) * np.cos(omk) * eta_dot
               + xi * np.cos(omk) * small_om_dot)
              - eta * np.cos(ik) * np.sin(omk) * small_om_dot)
             - eta * np.sin(ik) * np.cos(omk) * i_dot)
    z_dot = np.sin(ik) * eta_dot + eta * np.cos(ik) * i_dot
    # relativistic effects correction
    dt_rel = -2 * (x * x_dot + y * y_dot + z * z_dot) / c ** 2
    dt_sat = dt + dt_rel
    if with_rel:
        if rel_sep:
            return np.array([x, y, z, dt, dt_rel]), np.array([x_dot, y_dot, z_dot, dt_dot])
        else:
            return np.array([x, y, z, dt_sat,x_dot, y_dot, z_dot, dt_dot])
    else:
        return np.array([x, y, z, dt]), np.array([x_dot, y_dot, z_dot, dt_dot])


def gal_numpy_broadcast_interpolation(message_toc, observation_toc, message, with_rel=True, rel_sep=False):
    """
    This function interpolates satellite position at given epoch using Galileo broadcast navigation message
    :param message_toc: toc epoch of broadcast message
    :param observation_toc: toc epoch at which interpolation is done
    :param message: broadcast message
    :return: X,Y,Z dte (coordinates + SV clock offset correction)
    ['SVclockBias'          0,
    SVclockDrift',          1
    'SVclockDriftRate',     2
    'IODnav',               3
    'Crs',                  4
    'DeltaN',               5
    'M0',                   6
    'Cuc',                  7
    'Eccentricity',         8
    'Cus',                  9
    'sqrtA',                10
    'Toe',                  11
    'Cic',                  12
    'Omega0',               13
    'Cis',                  14
    'Io',                   15
    'Crc',                  16
    'omega',                17
    'OmegaDot',             18
    'IDOT',                 19
    'DataSrc',              20
    'GALWeek',              21
    'spare0'                22
    SISA                    23
    health                  24
    BGDe5a                  25
    BGDe5b                  26
    TransTime               27
    spare1                  28
    spare2                  29
    spare3                  30
    """

    # constants
    mi = 3.986004418e14
    pi = 3.1415926535898  # rad
    ome = 0.000072921151467  # rad/s
    c = 299792458  # m/s
    # SV clock correction (GPST to UTC)
    a0 = message[0]
    a1 = message[1]
    a2 = message[2]
    t_toc = observation_toc - message_toc
    if t_toc < -302400:
        dT = t_toc + 604800
    elif t_toc > 302400:
        dT = t_toc - 604800
    else:
        dT = t_toc
    dt = a0 + a1 * dT + a2 * dT ** 2
    dt_dot = a1 + 2 * a2 *dT
    # ephemeris reference epoch
    toe = message[11]
    t_toe = observation_toc - toe
    if t_toe > 302400:
        tk = t_toe - 604800
    elif t_toe < -302400:
        tk = t_toe + 604800
    else:
        tk = t_toe
    # Satellite orbit major axis
    sqrta = message[10]
    a = sqrta ** 2
    # Satellite average motion
    n0 = np.sqrt(mi / a ** 3)
    # corrected average motion
    dn = message[5]
    n = n0 + dn
    # mean anomaly at tk epoch
    m0 = message[6]
    mk = m0 + n * tk  # should be <0, 2pi>
    mk = mk % (2 * pi)
    # Eccentric anomaly
    e = message[8]
    ek = eccentric_anomaly(mk, e, pi)

    # True anomaly
    v = math.atan2((np.sqrt(1 - e ** 2) * np.sin(ek)), (np.cos(ek) - e))
    # latitude argument
    u = message[17] + v
    u = u % (2 * pi)
    # latitude argument correction
    cus = message[9]
    cuc = message[7]
    duk = cus * np.sin(2 * u) + cuc * np.cos(2 * u)
    # leading beam correction
    crs = message[4]
    crc = message[16]
    drk = crs * np.sin(2 * u) + crc * np.cos(2 * u)
    # orbit inclination correction
    cis = message[14]
    cic = message[12]
    idot = message[19]
    dik = cis * np.sin(2 * u) + cic * np.cos(2 * u) + idot * tk
    # corrected latitude argument
    uk = u + duk
    # corrected leading beam
    rk = a * (1 - e * np.cos(ek)) + drk
    # corrected orbit inclination angle
    i0 = message[15]
    ik = i0 + dik
    # corrected length of ascending orbit node
    om0 = message[13]
    omv = message[18]
    omk = om0 + (omv - ome) * tk - ome * toe
    omk = omk % (2 * pi)
    # coordinates of the satellite in the plane of orbit
    xi = rk * np.cos(uk)
    eta = rk * np.sin(uk)
    # geocentric coordinates of the satellite
    x = xi * np.cos(omk) - eta * np.cos(ik) * np.sin(omk)
    y = xi * np.sin(omk) + eta * np.cos(ik) * np.cos(omk)
    z = eta * np.sin(ik)
    # The product of the mean and eccentric anomaly velocity
    em_dot = 1 / (1 - e * np.cos(ek))
    # True anomaly velocity
    v_dot = np.sqrt((1 + e) / (1 - e)) * (1 / (np.cos(ek / 2) ** 2)) * (1 / (1 + np.tan(v / 2) ** 2)) * em_dot * n
    # argument of latitude velocity
    cus = cus
    cuc = cuc
    om_dot = message[18]
    u_dot = v_dot + 2 * v_dot * (cus * np.cos(2 * u) - cuc * np.sin(2 * u))
    small_om_dot = om_dot - ome
    # inclination rate
    cis = cis
    cic = cic
    i_dot = idot + 2 * v_dot * (cis * np.cos(2 * u) - cic * np.sin(2 * u))
    # leading beam rate
    r_dot = a * e * np.sin(ek) * em_dot * n + 2 * v_dot * (crs * np.cos(2 * u) - crc * np.sin(2 * u))
    # satellite velocity in the plane of an orbit
    xi_dot = r_dot * np.cos(uk) - rk * np.sin(uk) * u_dot
    eta_dot = r_dot * np.sin(uk) + rk * np.cos(uk) * u_dot
    # satellite geocentric velocity
    x_dot = (((np.cos(omk) * xi_dot - np.cos(ik) * np.sin(omk) * eta_dot
               - xi * np.sin(omk) * small_om_dot)
              - eta * np.cos(ik) * np.cos(omk) * small_om_dot)
             + eta * np.sin(ik) * np.sin(omk) * i_dot)

    y_dot = (((np.sin(omk) * xi_dot + np.cos(ik) * np.cos(omk) * eta_dot
               + xi * np.cos(omk) * small_om_dot)
              - eta * np.cos(ik) * np.sin(omk) * small_om_dot)
             - eta * np.sin(ik) * np.cos(omk) * i_dot)
    z_dot = np.sin(ik) * eta_dot + eta * np.cos(ik) * i_dot
    # relativistic effects correction
    dt_rel = -2 * (x * x_dot + y * y_dot + z * z_dot) / c ** 2
    dt_sat = dt + dt_rel
    if with_rel:
        if rel_sep:
            return np.array([x, y, z, dt, dt_rel]), np.array([x_dot, y_dot, z_dot, dt_dot])
        else:
            return np.array([x, y, z, dt_sat,x_dot, y_dot, z_dot, dt_dot])
    else:
        return np.array([x, y, z, dt]), np.array([x_dot, y_dot, z_dot, dt_dot])


# BDS-SIS-ICD constants: CGCS2000 GM and Earth rotation rate. The rotation rate
# differs from the WGS-84 value used by GPS/Galileo (7.2921151467e-5); with the
# wrong one the node longitude drifts by dOmega * t (metres late in the week).
GM_BDS = 3.986004418e14
OMEGA_E_BDS = 7.2921150e-5
BDS_CNAV_MESSAGES = frozenset({"CNV1", "CNV2", "CNV3"})
# BeiDou message matrix layout (``BrdcGenerator.bds_mes_cols``/``BroadcastInterp.bds_mes_cols``).
BDS_MES_COLS = ('SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'AODE', 'Crs',
                'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic',
                'Omega0', 'Cis', 'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'spare0',
                'BDTWeek', 'SVacc', 'SatH1', 'TGD1', 'TGD2', 'TransTime',
                'AODC')
# Column positions of the B-CNAV rate terms appended after ``BDS_MES_COLS``
# (see ``_bds_message_matrix``); D1/D2 rows carry zeros there.
_BDS_ADOT_COL = len(BDS_MES_COLS)
_BDS_DN0DOT_COL = _BDS_ADOT_COL + 1


def _bds_numpy_broadcast_interpolation_batch(message_toc, observation_toc, messages):
    """Vectorised BeiDou MEO/IGSO broadcast propagation (BDS-SIS-ICD).

    ``messages`` follows ``bds_mes_cols``; two optional trailing columns hold the
    B-CNAV rate terms Adot and dn0dot (RINEX 4 CNV1/CNV2/CNV3). For B-CNAV
    (BDS-SIS-ICD-B1C/B2a, B2b) the semi-major axis and the mean-motion correction
    vary linearly in time:  A_k = A0 + Adot*tk,  n_A = n0 + dn0 + dn0dot*tk/2,
    with A0 = sqrtA^2 (RINEX 4 stores sqrt(A_ref + dA)). For D1/D2 both rates are
    zero and the expressions reduce to the classic D1/D2 user algorithm.
    GEO satellites (extra -5 deg rotation) are excluded before this point.
    """
    mi = GM_BDS
    ome = OMEGA_E_BDS
    c = 299792458.0

    messages = np.asarray(messages, dtype=float)
    if messages.ndim == 1:
        messages = messages[None, :]
    n_rows = messages.shape[0]
    if messages.shape[1] == _BDS_DN0DOT_COL + 1:
        a_dot = messages[:, _BDS_ADOT_COL]
        dn0_dot = messages[:, _BDS_DN0DOT_COL]
    elif messages.shape[1] == len(BDS_MES_COLS):
        # D1/D2 layout without the rate columns
        a_dot = np.zeros(n_rows)
        dn0_dot = np.zeros(n_rows)
    else:
        raise ValueError(
            f"BeiDou message rows must have {len(BDS_MES_COLS)} (D1/D2) or "
            f"{len(BDS_MES_COLS) + 2} (with Adot, dn0dot) columns, got {messages.shape[1]}."
        )

    a0 = messages[:, 0]
    a1 = messages[:, 1]
    a2 = messages[:, 2]

    t_toc = observation_toc - message_toc
    dT = np.where(t_toc < -302400.0, t_toc + WEEK, np.where(t_toc > 302400.0, t_toc - WEEK, t_toc))
    dt = a0 + a1 * dT + a2 * dT ** 2
    dt_dot = a1 + 2.0 * a2 * dT

    toe = messages[:, 11]
    t_toe = observation_toc - toe
    tk = np.where(t_toe > 302400.0, t_toe - WEEK, np.where(t_toe < -302400.0, t_toe + WEEK, t_toe))

    sqrta = messages[:, 10]
    a_ref = sqrta ** 2
    a = a_ref + a_dot * tk
    n0 = np.sqrt(mi / a_ref ** 3)
    n = n0 + messages[:, 5] + 0.5 * dn0_dot * tk
    # dM/dt: the dn0dot term is quadratic in tk inside M_k
    m_rate = n + 0.5 * dn0_dot * tk
    mk = np.mod(messages[:, 6] + n * tk, 2 * pi)
    e = messages[:, 8]
    ek = _eccentric_anomaly_vec(mk, e)
    # Newton polish: the fixed-point stop at 1e-8 rad leaves up to e*1e-8 rad,
    # i.e. millimetres for the IGSO orbits (e ~ 0.01, r ~ 42000 km).
    for _ in range(2):
        ek = ek - (ek - e * np.sin(ek) - mk) / (1.0 - e * np.cos(ek))

    v = np.arctan2(np.sqrt(1.0 - e ** 2) * np.sin(ek), np.cos(ek) - e)
    u = np.mod(messages[:, 17] + v, 2 * pi)

    cus = messages[:, 9]
    cuc = messages[:, 7]
    duk = cus * np.sin(2.0 * u) + cuc * np.cos(2.0 * u)

    crs = messages[:, 4]
    crc = messages[:, 16]
    drk = crs * np.sin(2.0 * u) + crc * np.cos(2.0 * u)

    cis = messages[:, 14]
    cic = messages[:, 12]
    idot = messages[:, 19]
    dik = cis * np.sin(2.0 * u) + cic * np.cos(2.0 * u) + idot * tk

    uk = u + duk
    rk = a * (1.0 - e * np.cos(ek)) + drk
    ik = messages[:, 15] + dik

    omk = np.mod(messages[:, 13] + (messages[:, 18] - ome) * tk - ome * toe, 2 * pi)
    xi = rk * np.cos(uk)
    eta = rk * np.sin(uk)

    cos_omk = np.cos(omk)
    sin_omk = np.sin(omk)
    cos_ik = np.cos(ik)
    sin_ik = np.sin(ik)

    x = xi * cos_omk - eta * cos_ik * sin_omk
    y = xi * sin_omk + eta * cos_ik * cos_omk
    z = eta * sin_ik

    em_dot = 1.0 / (1.0 - e * np.cos(ek))
    v_dot = np.sqrt((1.0 + e) / (1.0 - e)) * (1.0 / (np.cos(ek / 2.0) ** 2)) * (
            1.0 / (1.0 + np.tan(v / 2.0) ** 2)) * em_dot * m_rate

    om_dot = messages[:, 18]
    u_dot = v_dot + 2.0 * v_dot * (cus * np.cos(2.0 * u) - cuc * np.sin(2.0 * u))
    small_om_dot = om_dot - ome
    i_dot = idot + 2.0 * v_dot * (cis * np.cos(2.0 * u) - cic * np.sin(2.0 * u))
    r_dot = (a * e * np.sin(ek) * em_dot * m_rate + a_dot * (1.0 - e * np.cos(ek))
             + 2.0 * v_dot * (crs * np.cos(2.0 * u) - crc * np.sin(2.0 * u)))

    xi_dot = r_dot * np.cos(uk) - rk * np.sin(uk) * u_dot
    eta_dot = r_dot * np.sin(uk) + rk * np.cos(uk) * u_dot

    x_dot = ((cos_omk * xi_dot - cos_ik * sin_omk * eta_dot - xi * sin_omk * small_om_dot)
             - eta * cos_ik * cos_omk * small_om_dot
             + eta * sin_ik * sin_omk * i_dot)
    y_dot = ((sin_omk * xi_dot + cos_ik * cos_omk * eta_dot + xi * cos_omk * small_om_dot)
             - eta * cos_ik * sin_omk * small_om_dot
             - eta * sin_ik * cos_omk * i_dot)
    z_dot = sin_ik * eta_dot + eta * np.cos(ik) * i_dot

    # Relativistic clock term -2 r.v / c^2 (r.v = rk * r_dot). The B-CNAV Adot
    # drift of the fitted semi-major axis is not an orbital eccentricity effect:
    # the ICD term F*e*sqrt(A)*sin(Ek) does not contain it, so it is left out here
    # (otherwise ~-2*A*Adot/c, i.e. centimetres of clock error for IGSO).
    dt_rel = -2.0 * ((x * x_dot + y * y_dot + z * z_dot) - rk * a_dot * (1.0 - e * np.cos(ek))) / c ** 2
    dt_sat = dt + dt_rel
    return np.column_stack([x, y, z, dt_sat, x_dot, y_dot, z_dot, dt_dot])


def bds_numpy_broadcast_interpolation(message_toc, observation_toc, message, with_rel=True, rel_sep=False):
    """BeiDou broadcast interpolation for BDS MEO/IGSO messages (D1/D2 and B-CNAV).

    Single-epoch wrapper around ``_bds_numpy_broadcast_interpolation_batch`` (CGCS2000
    constants). ``message`` follows ``BDS_MES_COLS``; a B-CNAV (CNV1/CNV2/CNV3) message
    must append Adot and dn0dot, otherwise it is propagated with the D1/D2 algorithm
    (constant A and dn), which is off by tens of metres for IGSO.
    The broadcast pipeline skips BDS GEO PRNs before this function is called, because
    those satellites require the BeiDou GEO-specific -5 deg coordinate rotation.
    """
    row = _bds_numpy_broadcast_interpolation_batch(
        np.atleast_1d(np.asarray(message_toc, dtype=float)),
        np.atleast_1d(np.asarray(observation_toc, dtype=float)),
        np.asarray(message, dtype=float)[None, :],
    )[0]
    x, y, z, dt_sat, x_dot, y_dot, z_dot, dt_dot = row
    if with_rel and not rel_sep:
        return row
    # clock polynomial alone; the batch result carries polynomial + relativistic term
    d_toc = float(_wrap_signed(float(observation_toc) - float(message_toc)))
    dt = message[0] + message[1] * d_toc + message[2] * d_toc ** 2
    dt_rel = dt_sat - dt
    if with_rel:
        return np.array([x, y, z, dt, dt_rel]), np.array([x_dot, y_dot, z_dot, dt_dot])
    return np.array([x, y, z, dt]), np.array([x_dot, y_dot, z_dot, dt_dot])


def _bds_message_matrix(merged: pd.DataFrame, needed) -> np.ndarray:
    """``bds_mes_cols`` message matrix plus the B-CNAV rate columns (Adot, dn0dot).

    D1/D2 records get zero rates. A B-CNAV record whose rate is missing keeps NaN, so
    the row is rejected instead of silently propagated with the D1/D2 algorithm.
    """
    if list(needed) != list(BDS_MES_COLS):
        raise ValueError("BeiDou message columns must follow BDS_MES_COLS.")
    messages = merged[needed].to_numpy(dtype=float, copy=False)
    n_rows = len(merged)
    if "nav_message" in merged.columns:
        is_cnav = merged["nav_message"].isin(BDS_CNAV_MESSAGES).to_numpy()
    else:
        is_cnav = np.zeros(n_rows, dtype=bool)
    rates = []
    for col in ("ADot", "DeltaNdot"):
        if col in merged.columns:
            vals = merged[col].to_numpy(dtype=float, copy=True)
        else:
            vals = np.full(n_rows, np.nan)
        rates.append(np.where(is_cnav, vals, np.nan_to_num(vals, nan=0.0)))
    return np.column_stack([messages] + rates)



def _wrap_dist_sow(a_sow: np.ndarray, b_sow: np.ndarray) -> np.ndarray:
    d = np.abs(a_sow[:, None] - b_sow[None, :])
    return np.minimum(d, WEEK - d)

BDS_MINUS_GPS_SECONDS = -14.0

def _obs_toc_for_system(times, system: str):
    toc = np.asarray(gnx_py.core.time.datetime2toc(t=times), dtype=float)
    if system == "C":
        # BeiDou broadcast Toe/Toc values are expressed in BDT. Mixed RINEX
        # observation epochs are GPST in the current SPP workflow, and
        # BDT = GPST - 14 s.
        toc = np.mod(toc + BDS_MINUS_GPS_SECONDS, WEEK)
    return toc

#: Systems whose broadcast record is chosen among the records already transmitted at the epoch first
#: (``_pick_nav_rows``; since v1.0.0). A Galileo record carries its Toe at or before the start of
#: its transmission (the batch is issued every 10 min and used forward of Toe), so the nearest Toe could pick a
#: record broadcast up to hours after the epoch and fitted for later times: RINEX 4 file of 2024-035, E13 at
#: 06:30 took the record with Toe 10:30 transmitted at 10:41, 55 m from the RINEX 3 run. A related, not identical
#: rule: RTKLIB-EX 2.5.1 keeps only Galileo records with Toe before the epoch (ephemeris.c:458 seleph, "AOD<=0");
#: GNX requires the transmission time (TransTime, or Toe when it is unknown) at or before the epoch, which picks a
#: record about 11 min older (Galileo TransTime - Toe on BRDC00IGS 2024-035: median 11.1, p95 14.6 min). GPS
#: needs no preference (a
#: record is transmitted two hours before its Toe, IS-GPS-200 20.3.4.4, so the nearest Toe has always been
#: sent); BeiDou keeps the nearest-Toe rule of RTKLIB seleph.
_TRANSMITTED_FIRST_SYSTEMS = frozenset({"E"})


def _broadcast_validity_seconds(system: str) -> float:
    """Largest |epoch - Toe| at which a broadcast ephemeris of the system is used.

    GPS 2 h (RTKLIB MAXDTOE 7200 s), Galileo 3 h (RTKLIB 2.4.2 MAXDTOE_GAL 10800 s, rtklib.h:202; 4 h before v1.0.0: a Galileo record 3-4 h from its Toe is outside the validity RTKLIB 2.4.2 grants;
    RTKLIB-EX 2.5.1 keeps 4 h, rtklib.h:264 MAXDTOE_GAL 14400 s, with its Toe-before-the-epoch rule), BeiDou 2 h.
    """
    if system == "E":
        return 10800.0
    if system == "C":
        # BeiDou D1/D2 ephemerides are refreshed every hour: 2 h tolerates one missed
        # update, while 3-4 h after Toe the orbit error grows to 10-35 m (vs COD).
        return 7200.0
    return 7200.0

def _bds_message_types_for_mode(mode: str | None) -> set[str]:
    try:
        signals = set(mode_signals(mode or "B1IB3I"))
    except ValueError:
        return {"D1", "D2"}
    if not signals.intersection({"B1C", "B2a", "B2b"}):
        return {"D1", "D2"}
    if "B2b" in signals:
        return {"CNV3"}
    if "B2a" in signals:
        return {"CNV2"}
    return {"CNV1"}

def _bds_drop_incomplete_cnav(nav: pd.DataFrame) -> pd.DataFrame:
    """Drop B-CNAV records whose Adot or dn0dot field is blank.

    Without these rates a CNV1/CNV2/CNV3 orbit cannot be propagated. Removing the
    record (not the column) keeps one blank field from disabling B-CNAV for the
    whole run and lets the Toe selector fall back to a neighbouring record.
    """
    if "nav_message" not in nav.columns:
        return nav
    is_cnav = nav["nav_message"].isin(BDS_CNAV_MESSAGES)
    rate_cols = [c for c in ("ADot", "DeltaNdot") if c in nav.columns]
    if len(rate_cols) < 2:
        return nav[~is_cnav]
    return nav[~is_cnav | nav[rate_cols].notna().all(axis=1)]


def _bds_drop_mixed_d1d2(nav: pd.DataFrame) -> pd.DataFrame:
    """Drop D1/D2 records whose Toe differs from their clock epoch (toc).

    In a D1/D2 data set toc and Toe are the same epoch. A record pairing the clock
    and first orbit words of one data set with the Toe and remaining words of the
    next one (merged BRD400DLR files: C10 2024-035 toc 15:00 / Toe 16:00) propagates
    to an orbit thousands of km off, so it must never reach the record selector.
    """
    if "Toe" not in nav.columns or "time" not in nav.index.names or nav.empty:
        return nav
    is_d = nav["nav_message"].isin(("D1", "D2")) if "nav_message" in nav.columns else \
        pd.Series(True, index=nav.index)
    toc = pd.DatetimeIndex(nav.index.get_level_values("time"))
    # seconds of the (Sunday-based) week of the record's own time scale (BDT for BDS)
    toc_sow = (((toc.dayofweek.to_numpy() + 1) % 7) * 86400.0
               + (toc - toc.normalize()).total_seconds().to_numpy())
    mixed = is_d.to_numpy() & (np.abs(_wrap_signed(nav["Toe"].to_numpy(dtype=float) - toc_sow)) > 0.5)
    # GEO records are dropped later anyway: warn only about the satellites that are processed
    shown = [(sv, t) for sv, t in nav.index[mixed] if not is_bds_geo(sv)]
    if shown:
        warnings.warn(
            f"{len(shown)} BeiDou D1/D2 navigation record(s) with Toe != toc (inconsistent data set) "
            f"rejected: " + ", ".join(f"{sv} {pd.Timestamp(t):%Y-%m-%d %H:%M} BDT" for sv, t in shown[:10]),
            RuntimeWarning, stacklevel=3,
        )
    return nav[~mixed]

def _wrap_signed(t):
    return (t + WEEK/2) % WEEK - WEEK/2


def _ensure_obs_block(df_obs, sv_hint=None):
    """
        Returns df with columns 'sv','time','toc' (without messing with df.columns!).
        Accepts both MultiIndex (sv,time) and regular index.
        """
    if not isinstance(df_obs, pd.DataFrame):
        df_obs = pd.DataFrame(df_obs)

    if ('sv' not in df_obs.columns) or ('time' not in df_obs.columns):
        df_obs = df_obs.reset_index()
    if 'sv' not in df_obs.columns and sv_hint is not None:
        df_obs['sv'] = sv_hint
    if 'toc' not in df_obs.columns:
        raise ValueError("df_obs must have 'toc' column.")

    return df_obs.loc[:, ['sv', 'time', 'toc']]


def _ensure_nav_block(df_orb, sv_hint=None):
    """
        Returns df with columns NAV + 'sv','time','Toe','nav_toc' (without messing with df.columns!).
        """
    if not isinstance(df_orb, pd.DataFrame):
        df_orb = pd.DataFrame(df_orb)

    if ('sv' not in df_orb.columns) or ('time' not in df_orb.columns):
        df_orb = df_orb.reset_index()
    if 'sv' not in df_orb.columns and sv_hint is not None:
        df_orb['sv'] = sv_hint

    need = ['sv','time','Toe','nav_toc']
    miss = [c for c in need if c not in df_orb.columns]
    if miss:
        raise ValueError(f"df_orb missing columns: {miss} (make sure _add_nav_toc() was already there).")

    return df_orb


def _prep_nav_arrays(system: str, nav_block: pd.DataFrame):
    """
    Prepare nav arrays + iloc mapping (so we can filter and still return indices
    compatible with nav_block.iloc[...] later).
    """
    toe = nav_block['Toe'].to_numpy(float)
    trans = None
    trans_sow = None
    if 'TransTime' in nav_block.columns:
        raw = pd.to_numeric(nav_block['TransTime'], errors='coerce').to_numpy(float)
        # RINEX writes 0.9999E9 for an unknown transmission time: such a record must not
        # win the "latest transmission" tie-break (NaT is ignored there)
        raw = np.where(raw < 9e8, raw, np.nan)
        trans_sow = raw
        # TransTime is seconds of the record's own week and may run below 0 or past
        # 604800 s: rank absolute times (record epoch + offset wrapped to +-half a week),
        # so a record sent after the Saturday/Sunday midnight is the later one
        if {'time', 'nav_toc'} <= set(nav_block.columns):
            offset = _wrap_signed(raw - nav_block['nav_toc'].to_numpy(float))
            trans = pd.Series(pd.DatetimeIndex(nav_block['time']) + pd.to_timedelta(offset, unit='s'))
        else:
            trans = pd.Series(pd.to_timedelta(raw, unit='s') + pd.Timestamp(0))

    if 'SISA' in nav_block.columns:
        quality = nav_block['SISA'].to_numpy(float)
    elif 'SVacc' in nav_block.columns:
        quality = nav_block['SVacc'].to_numpy(float)
    else:
        quality = np.zeros_like(toe, dtype=float)

    orig_iloc = np.arange(len(nav_block), dtype=int)

    # Health filter (keep only healthy broadcast messages where the system exposes it)
    health_col = 'health' if 'health' in nav_block.columns else 'SatH1' if 'SatH1' in nav_block.columns else None
    if system in {'G', 'C'} and health_col is not None:
        good = (nav_block[health_col].to_numpy() == 0)
        toe = toe[good]
        quality = quality[good]
        orig_iloc = orig_iloc[good]
        if trans is not None:
            trans = trans.iloc[good].reset_index(drop=True)
        if trans_sow is not None:
            trans_sow = trans_sow[good]

    return toe, trans, quality, orig_iloc, trans_sow


def _pick_nav_rows(
    system: str,
    toc_vec: np.ndarray,
    nav_block: pd.DataFrame,
    nearest_h: float | None = None,
    allow_out_of_window: bool = True,
) -> np.ndarray:
    """
    Row of ``nav_block`` (``iloc``) whose ephemeris is used at each epoch, -1 if none.

    Post-processing rule (RTKLIB seleph convention): the
    record with the nearest Toe, ``|age| = |wrap_signed(toc - Toe)| <= nearest_h``
    (default: the validity interval of the system, :func:`_broadcast_validity_seconds`).
    Galileo (``_TRANSMITTED_FIRST_SYSTEMS``) takes the nearest
    Toe among the records already transmitted at the epoch (TransTime <= epoch; a record
    without a known transmission time counts as transmitted from its Toe) when one lies
    within the interval, else the nearest Toe within the interval; the other systems
    take the nearest Toe whether or not the record had been transmitted. Records with
    the same Toe are ranked by the latest known transmission time, then by the best
    SISA/URA, then by file order. With ``allow_out_of_window`` (GPS) the record nearest
    in Toe is taken whatever its age when none lies within the interval.

    The choice at an epoch depends only on that epoch and the records, never on the
    epochs processed before it: a run limited to a time window, a run at another
    sampling interval and the daily run use the same ephemeris at a common epoch.
    """
    valid_h = _broadcast_validity_seconds(system)
    if nearest_h is None:
        nearest_h = valid_h

    toe, trans, quality, orig_iloc, trans_sow = _prep_nav_arrays(system, nav_block)
    transmitted_first = system in _TRANSMITTED_FIRST_SYSTEMS
    if transmitted_first:
        # seconds of week from which each record is known to be on the air: its transmission time, else its Toe
        sent_sow = toe if trans_sow is None else np.where(np.isfinite(trans_sow), trans_sow, toe)

    N = len(toc_vec)
    choice = np.full(N, -1, dtype=int)
    M = len(toe)
    if M == 0 or N == 0:
        return choice

    # tie-break rank: known transmission time (latest first), then quality, then file order
    trans_key = np.full(M, -np.inf)
    if trans is not None:
        tv = pd.to_datetime(pd.Series(trans).reset_index(drop=True))
        known = tv.notna().to_numpy()
        trans_key[known] = tv[known].astype('int64').to_numpy(dtype=float)
    q = np.asarray(quality, dtype=float)
    # SISA/URA: lower is better; a negative SISA (Galileo NAPA, no accuracy prediction) ranks last
    q = np.where(np.isfinite(q) & (q >= 0), q, np.inf)
    order = np.lexsort((np.arange(M), q, -trans_key))
    rank = np.empty(M, dtype=int)
    rank[order] = np.arange(M)

    def _pick(toc: np.ndarray) -> np.ndarray:
        dist = np.abs(_wrap_signed(toc[:, None] - toe[None, :]))  # [n, M]
        stages = [dist <= nearest_h]
        if transmitted_first:
            sent = _wrap_signed(toc[:, None] - sent_sow[None, :]) >= 0.0
            stages.insert(0, stages[0] & sent)
        if allow_out_of_window:
            stages.append(np.ones_like(stages[0]))

        pick = np.full(len(toc), -1, dtype=int)
        todo = np.ones(len(toc), dtype=bool)
        for mask in stages:
            best = np.where(mask, dist, np.inf).min(axis=1)
            rows = todo & np.isfinite(best)
            if not rows.any():
                continue
            tied = mask[rows] & (np.abs(dist[rows] - best[rows, None]) < 1e-9)
            pick[rows] = np.where(tied, rank[None, :], M).argmin(axis=1)
            todo &= ~rows
            if not todo.any():
                break
        return pick

    # blocks of epochs bound the [epochs x records] work arrays (a 1 Hz day of Galileo
    # would need GBs at once); the choice is per epoch, so blocking does not change it
    block = 4096
    pick = np.concatenate([_pick(toc_vec[i:i + block]) for i in range(0, N, block)])
    ok = pick >= 0
    choice[ok] = orig_iloc[pick[ok]]
    return choice


def _select_by_toe_for_sat(system, obs_sv, nav_sv):
    toc_vec = obs_sv['toc'].to_numpy(float)
    # nearest Toe (Galileo: among the records already transmitted first); only GPS falls back to a
    # record outside its validity interval (Galileo 3 h and BeiDou 2 h epochs without a record are
    # refused, not extrapolated)
    choice = _pick_nav_rows(
        system=system,
        toc_vec=toc_vec,
        nav_block=nav_sv,
        nearest_h=None,
        allow_out_of_window=(system == "G"),
    )

    keep = choice >= 0
    if not keep.any():
        return pd.DataFrame()

    keep_idx = np.flatnonzero(keep)
    nav_sel = nav_sv.iloc[choice[keep_idx]].reset_index(drop=True)
    obs_sel = obs_sv.iloc[keep_idx].reset_index(drop=True)

    nav_sel = nav_sel.copy()
    nav_sel['sv'] = obs_sel['sv'].to_numpy(copy=False)
    nav_sel['time'] = obs_sel['time'].to_numpy(copy=False)

    merged = pd.concat(
        [obs_sel, nav_sel], axis=1
    )
    return merged


def _brdc_no_ephemeris_reason(system, nav_block, toc) -> str:
    """Why no broadcast ephemeris was selected for a satellite (text for the aggregated warning)."""
    try:
        toe = nav_block['Toe'].to_numpy(float)
        health_col = ('health' if 'health' in nav_block.columns
                      else 'SatH1' if 'SatH1' in nav_block.columns else None)
        if system in {'G', 'C'} and health_col is not None:
            # same health rule as _prep_nav_arrays (IS-GPS-200 20.3.3.3.1.4: any non-zero health is unusable)
            h = pd.to_numeric(nav_block[health_col], errors='coerce').to_numpy(float)
            good = h == 0
            if not good.any():
                n_missing = int(np.sum(~np.isfinite(h)))
                vals = sorted({f"{v:g}" for v in h[np.isfinite(h)]})
                if not vals:
                    return f"{health_col} missing in all {h.size} records"
                why = f"unhealthy, {health_col}={'/'.join(vals)}"
                return why + (f", {health_col} missing in {n_missing} records" if n_missing else "")
            toe = toe[good]
        toc = np.asarray(toc, dtype=float)
        if toe.size == 0 or toc.size == 0:
            return "no navigation record"
        valid_s = _broadcast_validity_seconds(system)
        outside = int(np.sum(np.min(_wrap_dist_sow(toc, toe), axis=1) > valid_s))
        return (f"no record within {valid_s / 3600:g} h of Toe at {outside} of {toc.size} epochs, "
                f"{toe.size} records")
    except Exception as exc:  # the reason is diagnostic only
        return f"no ephemeris selected ({type(exc).__name__})"


def _warn_brdc_unusable(system, no_eph, failed, stage, stacklevel=3):
    """One RuntimeWarning per system for satellites left out of a broadcast computation.

    ``no_eph``: {sv: reason} for satellites without a usable ephemeris; ``failed``:
    {sv: exception} for satellites whose ``stage`` raised (the satellite is skipped).
    """
    name = SYSTEM_NAMES.get(system, system)
    if no_eph:
        warnings.warn(
            f"{name} satellites without a usable broadcast ephemeris (left out): "
            + ", ".join(f"{sv} ({why})" for sv, why in sorted(no_eph.items())) + ".",
            RuntimeWarning, stacklevel=stacklevel,
        )
    if failed:
        def _short(exc):
            text = str(exc).replace("\n", " ")
            return f"{type(exc).__name__}: {text[:120]}" if text else type(exc).__name__
        warnings.warn(
            f"{name} broadcast {stage} failed for {len(failed)} satellite(s), left out: "
            + "; ".join(f"{sv} ({_short(e)})" for sv, e in sorted(failed.items())) + ".",
            RuntimeWarning, stacklevel=stacklevel,
        )


def _normalize_brdc_interp_result(res):
    if isinstance(res, (tuple, list)):
        a = np.atleast_1d(res[0]).astype(float, copy=False).ravel()
        b = np.atleast_1d(res[1]).astype(float, copy=False).ravel()
        return np.concatenate([a, b], axis=0)
    return np.atleast_1d(res).astype(float, copy=False).ravel()


def _run_brdc_interp(mes_toc, obs_toc, messages, interp_fun):
    n = len(messages)
    valid_mask = np.zeros(n, dtype=bool)
    if n == 0:
        return None, valid_mask

    finite_in = np.isfinite(mes_toc) & np.isfinite(obs_toc) & np.all(np.isfinite(messages), axis=1)
    if not finite_in.any():
        return None, valid_mask

    batch_fun = None
    if interp_fun is numpy_broadcast_interpolation:
        batch_fun = _numpy_broadcast_interpolation_batch
    elif interp_fun is gal_numpy_broadcast_interpolation:
        batch_fun = _gal_numpy_broadcast_interpolation_batch
    elif interp_fun is bds_numpy_broadcast_interpolation:
        batch_fun = _bds_numpy_broadcast_interpolation_batch

    if batch_fun is not None:
        try:
            crd = batch_fun(mes_toc[finite_in], obs_toc[finite_in], messages[finite_in])
            finite_out = np.all(np.isfinite(crd), axis=1)
            kept_idx = np.flatnonzero(finite_in)[finite_out]
            valid_mask[kept_idx] = True
            if kept_idx.size:
                return crd[finite_out], valid_mask
            return None, valid_mask
        except Exception:
            pass

    rows = []
    lengths = []
    for idx, (mt, ot, msg) in enumerate(zip(mes_toc, obs_toc, messages)):
        try:
            row = _normalize_brdc_interp_result(interp_fun(mt, ot, msg))
        except Exception:
            continue
        if not np.all(np.isfinite(row)):
            continue
        rows.append(row)
        lengths.append(row.size)
        valid_mask[idx] = True

    if not rows:
        return None, valid_mask
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent broadcast interpolation output sizes: {sorted(set(lengths))}")
    return np.vstack(rows), valid_mask

class BrdcGenerator:
    mes_cols = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE', 'Crs', 'DeltaN', 'M0',
        'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic', 'Omega0', 'Cis',
        'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'CodesL2', 'GPSWeek',
        'L2Pflag', 'SVacc', 'health', 'TGD', 'IODC', 'TransTime'
    ]
    gal_mes_cols = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODnav', 'Crs',
                    'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic',
                    'Omega0', 'Cis', 'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'DataSrc',
                    'GALWeek', 'SISA', 'health', 'BGDe5a', 'BGDe5b', 'TransTime']
    bds_mes_cols = list(BDS_MES_COLS)

    def __init__(self, system, interval, mode, nav, tolerance ='1H', tlim=None):
        self.sys = system
        self.interval = interval
        self.mode = mode
        self.nav = nav
        self.SEC_TO_FREQ = {
            1: "1s",
            2: "2s",
            5: "5s",
            10: "10s",
            15: "15s",
            30: "30s",
            60: "1T",  # 1 minuta
            120: "2T",
            300: "5T",
            600: "10T",
            900: "15T",
            1800: "30T",
            3600: "1H",  # 1 godzina
            7200: "2H",
            10800: "3H",
            21600: "6H",
            43200: "12H",
            86400: "1D"  # 1 doba
        }
        self.table = None
        self.tolerance = tolerance
        self.tlim = tlim

    def _epoch_grid(self) -> pd.DatetimeIndex:
        """Output epochs: the ``tlim`` window, otherwise the observation day(s) of the file.

        With ``tlim`` the grid holds the epochs of ``tlim[0]`` .. ``tlim[1]`` (inclusive) at
        whole multiples of the interval from midnight, as the SP3 grid and the daily run,
        so a window shares its epochs with both (a start off that grid moves to the next
        grid epoch). Without
        it the grid covers the observation day(s) at whole multiples of the interval:
        a daily NAV file also carries the last messages of the previous day (GPS
        23:59:44, Galileo from 23:30) and the first of the next one, which must not
        shift the grid off the SP3 epochs. Epochs farther than the validity interval
        from every record of the system are left out; a cut of ``tlim`` is reported.
        """
        step = pd.Timedelta(seconds=self.interval)
        valid = pd.Timedelta(seconds=_broadcast_validity_seconds(self.sys))
        valid_h = valid.total_seconds() / 3600
        times = self.nav.index.get_level_values('time')
        first, last = times.min(), times.max()
        if self.tlim is not None:
            start, end = pd.Timestamp(self.tlim[0]), pd.Timestamp(self.tlim[1])
            grid = gnx_py.core.time.interval_epochs(start, end, self.interval)
            if grid.empty:
                raise ValueError(
                    f"tlim {start} .. {end} holds no epoch of the {self.interval} s grid "
                    f"(multiples of the interval from midnight); widen tlim or shorten the interval."
                )
        else:
            # observation days: dates lying more than half a day inside the record span
            day0 = (first + pd.Timedelta(hours=12)).floor('D')
            day1 = (last - pd.Timedelta(hours=12)).floor('D') + pd.Timedelta(days=1)
            if day1 > day0:
                start, end = day0, day1 - step
            else:  # sub-daily file: the span of its records
                start, end = first.ceil(step), last.floor(step)
            grid = pd.date_range(start, end, freq=step)
        usable = grid[(grid >= first - valid) & (grid <= last + valid)]
        if usable.empty:
            raise ValueError(
                f"No broadcast epochs for system {self.sys}: the requested epochs {start} .. {end} "
                f"(tlim/interval {self.interval} s) lie farther than {valid_h:g} h from every record "
                f"of the navigation file ({first} .. {last})."
            )
        if self.tlim is not None and len(usable) < len(grid):
            warnings.warn(
                f"Broadcast epoch grid of system {self.sys} truncated to {usable[0]} .. {usable[-1]}: "
                f"tlim asks for {start} .. {end}, but the navigation records span {first} .. {last} "
                f"and an ephemeris is used at most {valid_h:g} h from its Toe.",
                RuntimeWarning,
                stacklevel=3,
            )
        return usable

    def _add_obs_toc(self):
        obs_toc = _obs_toc_for_system(self.table.index.get_level_values('time').tolist(), self.sys)
        self.table['toc'] = obs_toc

    def _add_nav_toc(self):
        nav_toc = gnx_py.core.time.datetime2toc(t=self.nav.index.get_level_values('time').tolist())
        self.nav['nav_toc'] = nav_toc

    def _clear_suffixes(self, df):
        df = df.reset_index().copy()
        df['prn'] = df['sv'].astype(str).str[:3]
        df = df.drop(columns='sv')
        df.set_index(['prn', 'time'], inplace=True)
        df.index.names = ['sv', 'time']
        return df

    def select_messages(self, df_obs: pd.DataFrame, df_orb: pd.DataFrame) -> pd.DataFrame:
        """
                Broadcast selection (``_select_by_toe_for_sat``, rule of ``_pick_nav_rows``):
                - operates via satellite,
                - selects based on minimum |toc_obs - Toe| (SOW) with wrapping,
                - applies the validity windows of ``_broadcast_validity_seconds`` (GPS 2 h, Galileo 3 h,
                  BeiDou 2 h) and the health filters,
                - Galileo: the nearest Toe among the records already
                  transmitted at the epoch when one lies within the window, else the nearest Toe,
                - resolves ties by TransTime and quality (SISA/SVacc).
                Returns a DataFrame joined 1:1 with epochs, but only for those for which
                we found a valid ephemeris; no "forward/nearest" after 'time'.
                """
        # 'toc' in df_obs and 'nav_toc' in df_orb are required (added earlier in the pipeline)
        if 'toc' not in df_obs.columns:
            raise ValueError("df_obs must contain 'toc' (seconds-of-week); call _add_obs_toc() first.")
        if 'nav_toc' not in df_orb.columns:
            raise ValueError("df_orb must contain 'nav_toc' (seconds-of-week); call _add_nav_toc() first.")
        if 'Toe' not in df_orb.columns:
            raise ValueError("df_orb must contain 'Toe' (seconds-of-week).")

        sv_values = df_obs['sv'].unique()
        if sv_values.size == 1:
            sv = sv_values[0]
            nav_sv = df_orb
            if 'sv' in df_orb.columns:
                if not (df_orb['sv'].nunique() == 1 and df_orb['sv'].iloc[0] == sv):
                    nav_sv = df_orb[df_orb['sv'] == sv]
            if nav_sv.empty:
                return pd.DataFrame()
            sel = _select_by_toe_for_sat(self.sys, df_obs, nav_sv)
            if sel is None or sel.empty:
                return pd.DataFrame()
            if '__rank__' in sel.columns:
                sel = sel.drop(columns='__rank__')
            return sel

        out_list = []
        failed = {}
        for sv, obs_sv in df_obs.groupby('sv', sort=False):
            try:
                nav_sv = df_orb[df_orb['sv'] == sv]
                if nav_sv.empty:
                    continue
                sel = _select_by_toe_for_sat(self.sys, obs_sv, nav_sv)

                if not sel.empty:
                    out_list.append(sel)
            except Exception as e:
                failed[sv] = e
                continue
        _warn_brdc_unusable(self.sys, {}, failed, "ephemeris selection")
        if not out_list:
            return pd.DataFrame()
        merged = pd.concat(out_list, ignore_index=False)
        if '__rank__' in merged.columns:
            merged = merged.drop(columns='__rank__')
        return merged

    def prepare_table(self):
        if self.sys == 'G':
            sats_av = self.nav.index.get_level_values('sv').unique().tolist()
        elif self.sys == 'C':
            if "nav_message" in self.nav.columns:
                allowed_messages = _bds_message_types_for_mode(self.mode)
                self.nav = self.nav[self.nav["nav_message"].isin(allowed_messages)]
                self.nav = _bds_drop_incomplete_cnav(self.nav)
                if self.nav.empty:
                    raise ValueError(
                        f"No BeiDou navigation messages {sorted(allowed_messages)} available for mode {self.mode}."
                    )
            # RINEX 3 and RINEX 4 alike: a record mixing two D1/D2 data sets is never selectable
            self.nav = _bds_drop_mixed_d1d2(self.nav)
            self.nav = self._clear_suffixes(df=self.nav)
            geo_sats = sorted(
                sv for sv in self.nav.index.get_level_values('sv').unique().tolist()
                if is_bds_geo(sv)
            )
            if geo_sats:
                warnings.warn(
                    "BeiDou broadcast interpolation currently supports BDS MEO/IGSO only; "
                    f"skipping GEO satellites: {', '.join(geo_sats)}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.nav = self.nav[~self.nav.index.get_level_values('sv').map(is_bds_geo)]
            if 'SatH1' in self.nav.columns:
                self.nav = self.nav[self.nav['SatH1'] == 0]
            sats_av = self.nav.index.get_level_values('sv').unique().tolist()
        elif self.sys == 'E':
            self.nav = self._clear_suffixes(df=self.nav)
            # all records (I/NAV and F/NAV) carry the health used to screen the epochs below
            health_nav = self.nav

            # Message selection only: health is not a record filter here, since dropping an
            # unhealthy record would make the selector fall back to an older, stale ephemeris.
            if self.mode == 'E1':
                war =(self.nav['I/NAV_E1-B']==True)& (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]

            elif self.mode == 'E5b':
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]
            elif self.mode in ('E1E5b', 'E5aE5b'):
                # E5a/E5b has no clock of its own: I/NAV is the only message carrying both
                # BGD(E1,E5a) and BGD(E1,E5b) needed by an E5a/E5b user, and the Galileo HAS
                # reference message (HAS ICD 7.1.2, Table 21); the clock is the E1/E5b one.
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]
            elif self.mode in ['E5a','E1E5a']:
                war = (self.nav['F/NAV_E5a-I']==True)& (self.nav['dt_E5a_E1'] == True)
                self.nav = self.nav[war]
            sats_av = self.nav.index.get_level_values('sv').unique().tolist()
        if not sats_av:
            raise ValueError(
                f"No usable {self.sys} broadcast satellites after system-specific screening."
            )
        idx = pd.MultiIndex.from_product(
            [sats_av, self._epoch_grid()],
            names=["sv", "time"])
        self.table = pd.DataFrame(index=idx)
        if self.sys == 'E':
            # epochs at which a satellite is unhealthy for the signals of the mode
            unhealthy = _galileo_unhealthy_obs_mask(self.table, health_nav, self.mode)
            self.table = self.table[~unhealthy]

    def get_coordinates(self):
        output = []
        no_eph, failed = {}, {}
        if self.sys == 'E':
            needed = self.gal_mes_cols
            interp_fun = gal_numpy_broadcast_interpolation
        elif self.sys == 'G':
            needed = self.mes_cols
            interp_fun = numpy_broadcast_interpolation
        elif self.sys == 'C':
            needed = self.bds_mes_cols
            interp_fun = bds_numpy_broadcast_interpolation
        else:
            raise ValueError(f"Unknown sys: {self.sys}")

        nav_by_sv = {}
        if 'sv' in self.nav.columns:
            for sv, df in self.nav.groupby('sv', sort=False):
                nav_by_sv[sv] = _ensure_nav_block(df, sv_hint=sv)
        else:
            for sv in self.table.index.get_level_values('sv').unique():
                try:
                    nav_by_sv[sv] = _ensure_nav_block(self.nav.loc[sv], sv_hint=sv)
                except Exception:
                    continue

        for sv, df_obs_grp in self.table.groupby('sv', sort=False):
            try:
                # OBS
                df_obs_block = _ensure_obs_block(df_obs_grp, sv_hint=sv)

                # NAV
                df_orb_block = nav_by_sv.get(sv)
                if df_orb_block is None or df_orb_block.empty:
                    continue

                merged = _select_by_toe_for_sat(self.sys, df_obs_block, df_orb_block)
                if merged is None or merged.empty:
                    no_eph[sv] = _brdc_no_ephemeris_reason(self.sys, df_orb_block, df_obs_block['toc'])
                    continue

                # Przygotowanie do interpolacji
                mes_toc = merged['nav_toc'].to_numpy(dtype=float, copy=False)
                obs_toc = merged['toc'].to_numpy(dtype=float, copy=False)

                # Na wszelki wypadek: sprawdź, czy wszystkie kolumny istnieją
                missing = [c for c in needed if c not in merged.columns]
                if missing:
                    raise ValueError(f"NAV columns are missing in merged: {missing}")

                # Wymuś float i kształt (N, K)
                if self.sys == 'C':
                    messages = _bds_message_matrix(merged, needed)
                else:
                    messages = merged[needed].to_numpy(dtype=float, copy=False)
                if messages.ndim != 2:
                    raise ValueError(f"messages has the wrong dimension: {messages.ndim}")

                crd, keep_mask = _run_brdc_interp(mes_toc, obs_toc, messages, interp_fun)
                if crd is None:
                    continue
                if not keep_mask.all():
                    merged = merged.loc[keep_mask].copy()
                    mes_toc = mes_toc[keep_mask]
                    obs_toc = obs_toc[keep_mask]

                cols8 = ['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot']
                cols9 = ['x', 'y', 'z', 'clk', 'clk_rel', 'vx', 'vy', 'vz', 'clk_dot']
                cols = cols8 if crd.shape[1] == 8 else (
                    cols9 if crd.shape[1] == 9 else [f'c{i}' for i in range(crd.shape[1])])

                merged = merged.loc[:,~merged.columns.duplicated()]
                merged = merged.set_index(['sv', 'time'])

                crd = pd.DataFrame(crd, index=merged.index, columns=cols)

                output.append(merged.join(crd))




            except Exception as e:
                failed[sv] = e
                continue

        _warn_brdc_unusable(self.sys, no_eph, failed, "orbit computation")
        if not output:
            raise ValueError("No ephemeris selected for all SVs (check validity windows, health and 'nav_toc').")
        return pd.concat(output)

    def generate(self):
        self.prepare_table()
        self._add_obs_toc()
        self._add_nav_toc()
        crd = self.get_coordinates()
        if self.sys == 'E':
            crd = crd[~_galileo_record_unhealthy(crd, self.mode)]
        return crd


# Galileo signal -> (message carrying its health bits, SHS column, DVS column).
# I/NAV (E1-B / E5b-I) transmits the E1-B and E5b health, F/NAV (E5a-I) the E5a health.
_GAL_SIGNAL_HEALTH = {
    'E1': ('inav', 'E1B_SHS', 'E1B_DVS'),
    'E5b': ('inav', 'E5b_SHS', 'E5b_DVS'),
    'E5a': ('fnav', 'E5a_SHS', 'E5a_DVS'),
}


def _galileo_health_signals(mode) -> tuple:
    """Galileo signals whose health ``mode`` depends on.

    Without a (known) mode the signal in use is not known, so a satellite is
    screened on all of E1, E5a and E5b: unusable on any of them rejects it.
    """
    try:
        return tuple(mode_signals(mode))
    except ValueError:
        return tuple(_GAL_SIGNAL_HEALTH)


def _galileo_napa(df: pd.DataFrame) -> np.ndarray:
    """Records with a SISA of NAPA (No Accuracy Prediction Available, -1 in RINEX 3.04).

    The Galileo OS SDD treats NAPA like DVS=1 (marginal SIS, no accuracy
    guarantee), so the signals of that message are taken as unusable.
    """
    if 'SISA' not in df.columns:
        return np.zeros(len(df), dtype=bool)
    sisa = pd.to_numeric(df['SISA'], errors='coerce').to_numpy(dtype=float)
    return np.isfinite(sisa) & (sisa < 0.0)


def _galileo_unhealthy_obs_mask(obs: pd.DataFrame, nav: pd.DataFrame, mode: str) -> np.ndarray:
    """Flag observations of Galileo satellites unhealthy for the signals of ``mode``.

    The health in force at an epoch is that of the latest record (of the message
    carrying the signal's health) issued at or before the epoch; before the first
    record the earliest one is used. A signal is unusable when its SHS is 1 (out of
    service) or 3 (in test), its DVS is 1 (navigation data without guarantee) or
    the SISA of the record is NAPA; SHS 2 (extended operations) is kept. Without
    a known ``mode`` all Galileo signals are screened. ``obs`` and ``nav`` are
    indexed by ``(sv, time)`` with plain PRNs (``E14``), as after ``_clear_suffixes``.
    """
    flags = np.zeros(len(obs), dtype=bool)
    signals = _galileo_health_signals(mode)
    if obs.empty or nav is None or nav.empty:
        return flags

    obs_r = pd.DataFrame({
        'sv': np.asarray(obs.index.get_level_values('sv'), dtype=object),
        'time': obs.index.get_level_values('time'),
        '__row': np.arange(len(obs)),
    }).sort_values('time', kind='stable')
    nav_r = nav.reset_index()
    for signal in signals:
        spec = _GAL_SIGNAL_HEALTH.get(signal)
        if spec is None:
            continue
        message, shs_col, dvs_col = spec
        if message == 'inav':
            src_cols = ['I/NAV_E1-B', 'I/NAV_E5b-I']
        else:
            src_cols = ['F/NAV_E5a-I']
        needed = src_cols + [shs_col, dvs_col]
        if any(c not in nav_r.columns for c in needed):
            continue
        sel = nav_r[src_cols].astype(bool).any(axis=1).to_numpy()
        if not sel.any():
            continue
        rec = pd.DataFrame({
            'sv': np.asarray(nav_r.loc[sel, 'sv'], dtype=object),
            'time': nav_r.loc[sel, 'time'].to_numpy(),
            'bad': (nav_r.loc[sel, shs_col].astype(int).isin([1, 3]).to_numpy()
                    | (nav_r.loc[sel, dvs_col].astype(int) == 1).to_numpy()
                    | _galileo_napa(nav_r.loc[sel])).astype(float),
        })
        # several records of one message at one epoch (E1-B and E5b-I pages): unhealthy if any says so
        rec = rec.groupby(['sv', 'time'], as_index=False)['bad'].max().sort_values('time', kind='stable')
        back = pd.merge_asof(obs_r, rec, on='time', by='sv', direction='backward')
        fwd = pd.merge_asof(obs_r, rec, on='time', by='sv', direction='forward')
        bad = back['bad'].fillna(fwd['bad']).fillna(0.0).to_numpy(dtype=float) > 0.0
        flags[back['__row'].to_numpy()] |= bad
    return flags


def _galileo_record_unhealthy(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Rows whose own (selected) Galileo record flags a signal of ``mode`` unusable.

    Only the health bits of the message the record carries are judged (E1-B/E5b
    for I/NAV, E5a for F/NAV): the other bits are not transmitted in it. This
    complements ``_galileo_unhealthy_obs_mask`` when the selector holds an older
    record across a health change. Without a known ``mode`` all signals are judged.
    """
    flags = np.zeros(len(df), dtype=bool)
    signals = _galileo_health_signals(mode)
    napa = _galileo_napa(df)
    for signal in signals:
        spec = _GAL_SIGNAL_HEALTH.get(signal)
        if spec is None:
            continue
        message, shs_col, dvs_col = spec
        src_cols = ['I/NAV_E1-B', 'I/NAV_E5b-I'] if message == 'inav' else ['F/NAV_E5a-I']
        if any(c not in df.columns for c in src_cols + [shs_col, dvs_col]):
            continue
        carries = df[src_cols].fillna(False).astype(bool).any(axis=1).to_numpy()
        shs = pd.to_numeric(df[shs_col].astype(object), errors='coerce').fillna(0).to_numpy()
        dvs = pd.to_numeric(df[dvs_col].astype(object), errors='coerce').fillna(0).to_numpy()
        flags |= carries & (np.isin(shs, (1, 3)) | (dvs == 1) | napa)
    return flags


class BroadcastInterp:
    mes_cols = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE', 'Crs', 'DeltaN', 'M0',
        'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic', 'Omega0', 'Cis',
        'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'CodesL2', 'GPSWeek',
        'L2Pflag', 'SVacc', 'health', 'TGD', 'IODC', 'TransTime'
    ]
    gal_mes_cols = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODnav', 'Crs',
                    'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic',
                    'Omega0', 'Cis', 'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'DataSrc',
                    'GALWeek', 'SISA', 'health', 'BGDe5a', 'BGDe5b', 'TransTime']
    bds_mes_cols = list(BDS_MES_COLS)

    def __init__(self, obs, nav, mode, sys, emission_time=True,tolerance='2H'):
        self.obs = obs
        self.nav = nav
        self.mode = mode
        self.sys = sys
        self.emission_time = emission_time
        self.tolerance=tolerance

    def _clear_suffixes(self, df):
        df = df.reset_index().copy()
        df['prn'] = df['sv'].astype(str).str[:3]
        df = df.drop(columns='sv')
        df.set_index(['prn', 'time'], inplace=True)
        df.index.names = ['sv', 'time']
        return df

    def _add_obs_toc(self, emission=False):
        self.obs = self.obs.copy()
        if emission:
            obs_toc = _obs_toc_for_system(self.obs['em_epoch'].tolist(), self.sys)
            self.obs.loc[:, 'toc'] = obs_toc
        else:
            obs_toc = _obs_toc_for_system(self.obs.index.get_level_values('time').tolist(), self.sys)
            self.obs.loc[:, 'toc'] = obs_toc
        return self.obs

    def _add_nav_toc(self):
        nav_toc = gnx_py.core.time.datetime2toc(t=self.nav.index.get_level_values('time').tolist())
        self.nav['nav_toc'] = nav_toc
        return self.nav

    def select_messages(self, df_obs: pd.DataFrame, df_orb: pd.DataFrame) -> pd.DataFrame:
        """
        Broadcast message selection (``_select_by_toe_for_sat``, rule of ``_pick_nav_rows``):
        - operates via satellite,
        - selects based on minimum |toc_obs - Toe| (SOW) with wrapping,
        - applies the validity windows of ``_broadcast_validity_seconds`` (GPS 2 h, Galileo 3 h, BeiDou 2 h)
          and the health filters,
        - Galileo: the nearest Toe among the records already transmitted at
          the epoch when one lies within the window, else the nearest Toe,
        - resolves ties by TransTime and quality (SISA/SVacc).
        Returns a DataFrame joined 1:1 with epochs, but only for those for which
        we found a valid ephemeris; no "forward/nearest" after 'time'.
        """
        # 'toc' in df_obs and 'nav_toc' in df_orb are required (added earlier in the pipeline)
        if 'toc' not in df_obs.columns:
            raise ValueError("df_obs must contain 'toc' (seconds-of-week); call _add_obs_toc() first.")
        if 'nav_toc' not in df_orb.columns:
            raise ValueError("df_orb must contain 'nav_toc' (seconds-of-week); call _add_nav_toc() first.")
        if 'Toe' not in df_orb.columns:
            raise ValueError("df_orb must contain 'Toe' (seconds-of-week).")

        # Zakładamy, że df_obs i df_orb mają kolumnę 'sv'
        sv_values = df_obs['sv'].unique()
        if sv_values.size == 1:
            sv = sv_values[0]
            nav_sv = df_orb
            if 'sv' in df_orb.columns:
                if not (df_orb['sv'].nunique() == 1 and df_orb['sv'].iloc[0] == sv):
                    nav_sv = df_orb[df_orb['sv'] == sv]
            if nav_sv.empty:
                return pd.DataFrame()
            sel = _select_by_toe_for_sat(self.sys, df_obs, nav_sv)
            if sel is None or sel.empty:
                return pd.DataFrame()
            if '__rank__' in sel.columns:
                sel = sel.drop(columns='__rank__')
            return sel

        out_list = []
        failed = {}
        for sv, obs_sv in df_obs.groupby('sv', sort=False):
            try:
                nav_sv = df_orb[df_orb['sv'] == sv]
                if nav_sv.empty:
                    continue
                sel = _select_by_toe_for_sat(self.sys, obs_sv, nav_sv)

                if not sel.empty:
                    out_list.append(sel)
            except Exception as e:
                failed[sv] = e
                continue
        _warn_brdc_unusable(self.sys, {}, failed, "ephemeris selection")
        if not out_list:
            return pd.DataFrame()
        merged = pd.concat(out_list, ignore_index=False)
        # usuń pomocnicze kolumny jeśli jakieś dodałeś
        if '__rank__' in merged.columns:
            merged = merged.drop(columns='__rank__')
        return merged

    def get_coordinates(self):
        output = []
        no_eph, failed = {}, {}
        if self.sys == 'E':
            needed = self.gal_mes_cols
            interp_fun = gal_numpy_broadcast_interpolation
        elif self.sys == 'G':
            needed = self.mes_cols
            interp_fun = numpy_broadcast_interpolation
        elif self.sys == 'C':
            needed = self.bds_mes_cols
            interp_fun = bds_numpy_broadcast_interpolation
        else:
            raise ValueError(f"Unknown sys: {self.sys}")

        nav_by_sv = {}
        if 'sv' in self.nav.columns:
            for sv, df in self.nav.groupby('sv', sort=False):
                nav_by_sv[sv] = _ensure_nav_block(df, sv_hint=sv)
        else:
            for sv in self.obs.index.get_level_values('sv').unique():
                try:
                    nav_by_sv[sv] = _ensure_nav_block(self.nav.loc[sv], sv_hint=sv)
                except Exception:
                    continue

        for sv, df_obs_grp in self.obs.groupby('sv', sort=False):
            try:
                # OBS
                df_obs_block = _ensure_obs_block(df_obs_grp, sv_hint=sv)

                # NAV
                df_orb_block = nav_by_sv.get(sv)
                if df_orb_block is None or df_orb_block.empty:
                    continue

                merged = _select_by_toe_for_sat(self.sys, df_obs_block, df_orb_block)
                if merged is None or merged.empty:
                    no_eph[sv] = _brdc_no_ephemeris_reason(self.sys, df_orb_block, df_obs_block['toc'])
                    continue

                mes_toc = merged['nav_toc'].to_numpy(dtype=float, copy=False)
                obs_toc = merged['toc'].to_numpy(dtype=float, copy=False)

                missing = [c for c in needed if c not in merged.columns]
                if missing:
                    raise ValueError(f"Missing data: {missing}\n Check your broadcast orbit")

                # Wymuś float i kształt (N, K)
                if self.sys == 'C':
                    messages = _bds_message_matrix(merged, needed)
                else:
                    messages = merged[needed].to_numpy(dtype=float, copy=False)
                if messages.ndim != 2:
                    raise ValueError(f"Dimension error : {messages.ndim}")

                crd, keep_mask = _run_brdc_interp(mes_toc, obs_toc, messages, interp_fun)
                if crd is None:
                    continue
                if not keep_mask.all():
                    merged = merged.loc[keep_mask].copy()
                    mes_toc = mes_toc[keep_mask]
                    obs_toc = obs_toc[keep_mask]

                cols8 = ['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot']
                cols9 = ['x', 'y', 'z', 'clk', 'clk_rel', 'vx', 'vy', 'vz', 'clk_dot']
                cols = cols8 if crd.shape[1] == 8 else (
                    cols9 if crd.shape[1] == 9 else [f'c{i}' for i in range(crd.shape[1])])

                merged = merged.loc[:, ~merged.columns.duplicated()]
                merged = merged.set_index(['sv', 'time'])

                crd = pd.DataFrame(crd, index=merged.index, columns=cols)

                output.append(merged.join(crd))




            except Exception as e:
                failed[sv] = e
                continue

        _warn_brdc_unusable(self.sys, no_eph, failed, "orbit computation")
        if not output:
            raise ValueError("No ephemeris selected for all SVs (check validity windows, health and 'nav_toc').")
        return pd.concat(output)

    def grab_emission_time(self):
        clight = 299792458
        if 'P3' in self.obs.columns:
            ccol='P3'
        else:
            ccol = _select_code_column(self.obs, self.mode)
        obs_time = self.obs.index.get_level_values('time')
        self.obs['obs_time'] = obs_time
        t_em = ((self.obs[ccol] / clight) + self.obs['clk'])
        self.obs['em_epoch'] = self.obs['obs_time'] - pd.to_timedelta(t_em, unit='s')
        return self.obs

    def _clear_reception_data(self):
        if self.sys == 'G':
            self.obs = self.obs.drop(columns=['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot'] + self.mes_cols)
        elif self.sys == 'E':
            self.obs = self.obs.drop(columns=['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot'] + self.gal_mes_cols)
        elif self.sys == 'C':
            self.obs = self.obs.drop(columns=['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot'] + self.bds_mes_cols)
        return self.obs

    def interpolate(self):

        init_obs = self.obs.copy()
        if self.sys== 'E':
            self.nav = self._clear_suffixes(df=self.nav)
            self.nav = self.nav.dropna(how='any', axis=1).dropna(how='all', axis=0)
            # Health is judged on all records (E1E5a needs the E1-B bits of I/NAV and the
            # E5a bits of F/NAV) and rejects observations, not records: dropping unhealthy
            # records would only make the selector fall back to an older, stale ephemeris.
            unhealthy = _galileo_unhealthy_obs_mask(self.obs, self.nav, self.mode)
            if unhealthy.any():
                rejected = sorted(set(self.obs.index.get_level_values('sv')[unhealthy]))
                warnings.warn(
                    f"Galileo satellites unhealthy for {self.mode} (SHS 1/3 or DVS 1) rejected at "
                    f"{int(unhealthy.sum())} observations: {', '.join(map(str, rejected))}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.obs = self.obs[~unhealthy]
                init_obs = init_obs[~unhealthy]
            if self.mode == 'E1':
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]

            elif self.mode == 'E5b':
                # I/NAV record with the E1/E5b clock, as for E1E5b; health is handled above
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]
            elif self.mode in ('E1E5b', 'E5aE5b'):
                # E5a/E5b has no clock of its own: I/NAV is the only message carrying both
                # BGD(E1,E5a) and BGD(E1,E5b) needed by an E5a/E5b user, and the Galileo HAS
                # reference message (HAS ICD 7.1.2, Table 21); the clock is the E1/E5b one.
                # E5a health still comes from the F/NAV records (checked above).
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]
            elif self.mode in ['E5a', 'E1E5a']:
                war = (self.nav['F/NAV_E5a-I'] == True) & (self.nav['dt_E5a_E1'] == True)
                self.nav = self.nav[war]
            self.obs = self._add_obs_toc(emission=False)
            self.nav = self._add_nav_toc()
            result = self.get_coordinates()
            self.obs = pd.merge(left=init_obs,right=result,on=['sv','time'],how='right')
            if self.emission_time:
                self.obs = self.grab_emission_time()
                self._clear_reception_data()
                self.obs = self._add_obs_toc(emission=True)
                result = self.get_coordinates()
            # a held (older) record may still flag the signals unusable
            result = result[~_galileo_record_unhealthy(result, self.mode)]
        if self.sys == 'C':
            if "nav_message" in self.nav.columns:
                # RINEX 4 carries D1/D2 and B-CNAV (CNV1/2/3) ephemerides of the same
                # satellite side by side; their clock parameters refer to different
                # signals and their orbit models differ, so keep only the message type
                # of the processed signals (never select across message types). This
                # must precede dropna(axis=1): the CNAV-only columns are NaN on D1/D2.
                allowed_messages = _bds_message_types_for_mode(self.mode)
                self.nav = self.nav[self.nav["nav_message"].isin(allowed_messages)]
                # a blank Adot/dn0dot drops that record here, not the whole column below
                self.nav = _bds_drop_incomplete_cnav(self.nav)
                if self.nav.empty:
                    raise ValueError(
                        f"No BeiDou navigation messages {sorted(allowed_messages)} available for mode {self.mode}."
                    )
            # RINEX 3 and RINEX 4 alike: a record mixing two D1/D2 data sets is never selectable
            self.nav = _bds_drop_mixed_d1d2(self.nav)
            self.nav = self.nav.dropna(how='any', axis=1).dropna(how='all', axis=0)
            self.nav = self._clear_suffixes(df=self.nav)
            geo_sats = sorted(
                sv for sv in self.nav.index.get_level_values('sv').unique().tolist()
                if is_bds_geo(sv)
            )
            if geo_sats:
                warnings.warn(
                    "BeiDou broadcast interpolation currently supports BDS MEO/IGSO only; "
                    f"skipping GEO satellites: {', '.join(geo_sats)}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.nav = self.nav[~self.nav.index.get_level_values('sv').map(is_bds_geo)]
            if 'SatH1' in self.nav.columns:
                self.nav = self.nav[self.nav['SatH1'] == 0]
            if self.nav.empty:
                raise ValueError(
                    "BeiDou broadcast interpolation currently supports BDS MEO/IGSO only; "
                    "GEO PRNs C01-C05/C59-C63 were excluded."
                )
            self.obs = self.obs[~self.obs.index.get_level_values('sv').map(is_bds_geo)]
            self.obs = self._add_obs_toc(emission=False)
            self.nav = self._add_nav_toc()
            result = self.get_coordinates()
            self.obs = pd.merge(left=init_obs, right=result, on=['sv', 'time'], how='right')
            if self.emission_time:
                self.obs = self.grab_emission_time()
                self._clear_reception_data()
                self.obs = self._add_obs_toc(emission=True)
                result = self.get_coordinates()
        if self.sys == 'G':

            self.nav = self.nav.dropna(how='any', axis=1).dropna(how='all', axis=0)
            self.nav = self._clear_suffixes(df=self.nav)
            self.obs = self._add_obs_toc(emission=False)
            self.nav = self._add_nav_toc()
            result = self.get_coordinates()
            self.obs = pd.merge(left=init_obs,right=result,on=['sv','time'],how='right')
            if self.emission_time:
                self.obs = self.grab_emission_time()
                self._clear_reception_data()
                self.obs = self._add_obs_toc(emission=True)
                result = self.get_coordinates()
        output = pd.merge(left=init_obs,right=result,on=['sv','time'],how='right')
        return output


def correct_sat_coordinates(xyz_s, xyz_a):
    """
    Correction of satellite coordinates with respect to earth rotation during signal travel
    :param xyz_s: satellite crd at the epoch
    :param xyz_a: station crd
    :return: xyz corrected by earth rotation
    """
    omega = 7.2921159e-5
    c = 299792458

    dT = (np.sqrt((xyz_s[0] - xyz_a[0]) ** 2 + (xyz_s[1] - xyz_a[1]) ** 2 + (xyz_s[2] - xyz_a[2]) ** 2)) / c
    omw = dT * omega

    cos0 = np.cos(omw)
    sin0 = np.sin(omw)
    rot = np.array([[cos0, sin0, 0],
                    [-sin0, cos0, 0],
                    [0, 0, 1]])
    xyz_s2 = rot.dot(xyz_s)
    return xyz_s2




# ---------------- Utils ----------------
def _to_seconds(dt_any) -> np.ndarray:
    dt64 = np.asarray(dt_any, dtype='datetime64[ns]')
    # int64 nanosekundy -> sekundy z ułamkami
    return dt64.astype('int64') / 1e9
def _median_dt_seconds(t_s: np.ndarray) -> float:
    return np.inf if t_s.size < 2 else float(np.median(np.diff(t_s)))

# ---------------- Barycentric Lagrange ----------------
@njit(cache=False, fastmath=False)
def _bary_weights(xw):
    n = xw.size
    w = np.empty(n, dtype=np.float64)
    for j in range(n):
        s = 1.0
        xj = xw[j]
        for k in range(n):
            if k != j:
                diff = xj - xw[k]
                if diff == 0.0:
                    for q in range(n):
                        w[q] = np.nan
                    return w
                s *= diff
        w[j] = 1.0 / s
    return w

@njit(cache=True, fastmath=True)
def _bary_eval_vec(xw, w, Yw, xq):
    m, D = Yw.shape
    for j in range(m):
        if w[j] != w[j]:  # NaN
            out_nan = np.empty(D, dtype=np.float64)
            for d in range(D):
                out_nan[d] = np.nan
            return out_nan
    num = np.zeros(D, dtype=np.float64)
    den = 0.0
    for j in range(m):
        dj = xq - xw[j]
        if dj == 0.0:
            out = np.empty(D, dtype=np.float64)
            for d in range(D):
                out[d] = Yw[j, d]
            return out
        tmp = w[j] / dj
        for d in range(D):
            num[d] += tmp * Yw[j, d]
        den += tmp
    inv = 1.0 / den
    out = np.empty(D, dtype=np.float64)
    for d in range(D):
        out[d] = num[d] * inv
    return out

@njit(parallel=True, cache=True, fastmath=True)
def lagrange_local_vec(x, Y, xq, m=12, max_gap=np.inf):
    N = x.size
    D = Y.shape[1]
    Q = xq.size
    out = np.empty((Q, D), dtype=np.float64)
    r = m if N >= m else N
    for i in range(Q):
        if r < 2:
            for d in range(D):
                out[i, d] = np.nan
            continue
        xi = xq[i]
        k = np.searchsorted(x, xi)
        left = k - r // 2
        if left < 0: left = 0
        right = left + r
        if right > N:
            right = N
            left = right - r
        nearest = 1e300
        if k < N and left <= k < right:
            tmp = abs(x[k] - xi)
            if tmp < nearest: nearest = tmp
        if k-1 >= 0 and left <= (k-1) < right:
            tmp = abs(x[k-1] - xi)
            if tmp < nearest: nearest = tmp
        if nearest > max_gap:
            for d in range(D):
                out[i, d] = np.nan
            continue
        xw = x[left:right]
        Yw = Y[left:right, :]
        w  = _bary_weights(xw)
        out[i, :] = _bary_eval_vec(xw, w, Yw, xi)
    return out
@njit(cache=True, fastmath=True)
def lin_interp1d_vec(x, y, xq, max_gap=np.inf):
    """
        Linear 1D interpolation (x,y)->y(xq) for sorted x.
        Returns NaN if the query is out of range or the nearest node is further than max_gap.
        """
    N = x.size
    Q = xq.size
    out = np.empty(Q, dtype=np.float64)
    for i in range(Q):
        xi = xq[i]
        k = np.searchsorted(x, xi)

        # trafienie dokładnie w węzeł
        if k < N and x[k] == xi:
            out[i] = y[k]
            continue

        i0 = k - 1
        i1 = k
        if i0 < 0 or i1 >= N:
            out[i] = np.nan
            continue

        # test luki (do najbliższego węzła)
        d0 = abs(xi - x[i0])
        d1 = abs(x[i1] - xi)
        if (d0 if d0 < d1 else d1) > max_gap:
            out[i] = np.nan
            continue

        denom = x[i1] - x[i0]
        if denom == 0.0:
            out[i] = np.nan
            continue

        t = (xi - x[i0]) / denom
        out[i] = (1.0 - t) * y[i0] + t * y[i1]
    return out

@njit(cache=True, fastmath=True)
def lin_interp1d_vec(x, y, xq, max_gap=np.inf):
    """
        Linear 1D interpolation (x,y)->y(xq) for sorted x.
        Returns NaN if the query is out of range or the nearest node is farther than max_gap.
        """
    N = x.size
    Q = xq.size
    out = np.empty(Q, dtype=np.float64)
    for i in range(Q):
        xi = xq[i]
        k = np.searchsorted(x, xi)

        # trafienie dokładnie w węzeł
        if k < N and x[k] == xi:
            out[i] = y[k]
            continue

        i0 = k - 1
        i1 = k
        if i0 < 0 or i1 >= N:
            out[i] = np.nan
            continue

        # test luki (do najbliższego węzła)
        d0 = abs(xi - x[i0])
        d1 = abs(x[i1] - xi)
        if (d0 if d0 < d1 else d1) > max_gap:
            out[i] = np.nan
            continue

        denom = x[i1] - x[i0]
        if denom == 0.0:
            out[i] = np.nan
            continue

        t = (xi - x[i0]) / denom
        out[i] = (1.0 - t) * y[i0] + t * y[i1]
    return out

# ---------------- Front: MultiIndex (time, sv) ----------------
def _sp3_lagrange_clock_nodes(g, gap_factor):
    """Valid clock nodes of one satellite's SP3 rows ``g`` (columns time, clk [s]): sorted node
    times [s], clocks [s], node epochs, and the arc gap limit and node spacing [s] of its rows."""
    t_all = _to_seconds(g['time'].to_numpy())
    dt_med = _median_dt_seconds(np.unique(t_all))
    max_gap = gap_factor * dt_med if np.isfinite(dt_med) else np.inf
    gc = g.dropna(subset=['clk'])
    gc = gc.loc[~gc['time'].duplicated(keep='first')]
    tc = _to_seconds(gc['time'].to_numpy())
    cc = gc['clk'].to_numpy(dtype=np.float64)
    order = np.argsort(tc, kind='stable')
    return tc[order], cc[order], gc['time'].to_numpy()[order], max_gap, dt_med


def lagrange_interp(
    obs: pd.DataFrame,
    sp3: pd.DataFrame,
    m: int = 12,
    gap_factor: float = 3.0,
    owner=None,
) -> pd.DataFrame:
    """
        For epochs (time, sv) from 'obs', interpolates from 'sp3' the position (x,y,z) [m],
        optionally the clock 'clk' [s], calculates the velocities (vx,vy,vz) [m/s] using the central difference method
        ±0.5 s on arc-relative time and relativistic clock correction:
            clk_rel = -2 * dot(r, v) / c^2   [s]
        Returns a DataFrame with MultiIndex (time, sv) and columns:
            x, y, z, clk, vx, vy, vz, clk_rel

        The SP3 node check warns (RuntimeWarning) about position nodes that two SP3 files give differently at an
        epoch both carry, and about satellite-epochs the nodes do not cover. ``owner`` is the object that keeps the
        warn-once state of the run (the session passes its configuration, see :mod:`gnx_py._notes`): each
        condition is then warned once per run, although a session interpolates every system at reception and at
        emission time. ``owner=None`` keeps no state: every call warns.
        """
    out_cols = ['x', 'y', 'z', 'clk_raw', 'clk', 'vx', 'vy', 'vz', 'clk_rel']
    if obs.empty:
        return pd.DataFrame(index=obs.index.copy(), columns=out_cols, dtype=np.float64)

    obs = obs.copy(); sp3 = sp3.copy()
    # obs.index = obs.index.set_names(['time', 'sv'])
    obs.index = (
        obs.index
        .set_names(['time', 'sv'])
        if isinstance(obs.index.levels[0][0], pd.Timestamp)
        else obs.index.reorder_levels([1, 0]).set_names(['time', 'sv'])
    )

    sp3.index = sp3.index.set_names(['time', 'sv'])
    # An epoch given by two files (24:00 / 00:00 of consecutive days) becomes one node, chosen by
    # content and independent of the order of the files (see _sp3_merge_duplicate_nodes). The
    # files are told apart by the row order: each file lists a satellite's epochs increasing, so
    # an epoch not later than the satellite's previous row starts the next file.
    s3 = sp3.reset_index()
    s3 = s3.loc[s3['time'].notna()]
    prev_t = s3.groupby('sv', sort=False)['time'].shift()
    s3['__file'] = (s3['time'] <= prev_t).groupby(s3['sv']).cumsum()
    s3, dropped = _sp3_merge_duplicate_nodes(s3, 'sv', 'time', '__file', ['x', 'y', 'z'],
                                             'clk' if 'clk' in s3.columns else None, SP3_BOUNDARY_TOL_M)
    msg = _sp3_boundary_message(dropped)
    if msg:
        # the same SP3 frame gives the same dropped nodes at every call of the run: one warning per text
        _notes.warn_once(owner, ("sp3_boundary_nodes", msg), msg, RuntimeWarning, stacklevel=3)
    sp3 = s3.set_index(['time', 'sv'])

    obs_sorted = obs.sort_index()
    times_obs = obs_sorted.index.get_level_values('time').to_numpy()
    sv_obs    = obs_sorted.index.get_level_values('sv').to_numpy()
    t_obs_s   = np.ascontiguousarray(_to_seconds(times_obs), dtype=np.float64)

    # mapowanie SV -> listy wierszy
    sv_to_rows: dict[str, list[int]] = {}
    for i, sv in enumerate(sv_obs.tolist()):
        sv_to_rows.setdefault(sv, []).append(i)

    # bufor wyjściowy
    n = len(obs_sorted)
    out_xyz  = np.full((n, 3), np.nan, dtype=np.float64)
    out_clk  = np.full(n, np.nan, dtype=np.float64)
    out_vel  = np.full((n, 3), np.nan, dtype=np.float64)
    out_rel  = np.full(n, np.nan, dtype=np.float64)
    skipped = []  # satellite-epochs the SP3 nodes do not cover

    # clock steps shared by a system's satellites (clock datum change, e.g. joined daily files),
    # over all satellites of the product (see SP3_CLOCK_COMMON_MIN_SATS)
    clk_common = {}
    if 'clk' in sp3.columns:
        series = {}
        for sv, g in sp3.reset_index().dropna(subset=['time']).groupby('sv', sort=False):
            tc, cc, _, max_gap, _ = _sp3_lagrange_clock_nodes(g, gap_factor)
            series.setdefault(str(sv)[:1], []).append((tc, cc, max_gap))
        clk_common = {s: _sp3_clock_common_steps(v) for s, v in series.items()}

    for sv, row_idx in sv_to_rows.items():
        try:
            g = sp3.xs(sv, level='sv', drop_level=False).reset_index()
        except KeyError:
            continue

        # Positions and clocks are each taken from their own valid nodes: an SP3 node may carry a
        # position without a clock (999999.999999) or a clock without a position (0.000000).
        has_clk = 'clk' in g.columns
        g = g.dropna(subset=['time'])
        if g.empty:
            continue
        t_all = _to_seconds(g['time'].to_numpy())
        dt_med  = _median_dt_seconds(np.unique(t_all))
        max_gap = gap_factor * dt_med if np.isfinite(dt_med) else np.inf

        gp = g.dropna(subset=['x', 'y', 'z'])
        gp = gp.loc[~gp['time'].duplicated(keep='first')]
        t_sv = _to_seconds(gp['time'].to_numpy())
        XYZ  = gp[['x','y','z']].to_numpy(dtype=np.float64, copy=False)
        order = np.argsort(t_sv, kind='stable')
        t_sv  = np.ascontiguousarray(t_sv[order], dtype=np.float64)
        XYZ   = np.ascontiguousarray(XYZ[order, :], dtype=np.float64)

        idx = np.asarray(row_idx, dtype=np.int64)
        tq  = np.ascontiguousarray(t_obs_s[idx], dtype=np.float64)

        # No extrapolation: each query is served by the contiguous node arc around it (see
        # _sp3_locate; SP3_EDGE_TOLERANCE_S covers the signal travel time at the file edges), and
        # the Lagrange window never reaches over a gap of missing nodes into another arc.
        min_nodes = SP3_MIN_ARC_NODES
        lo, hi, why = _sp3_locate(t_sv, tq, max_gap, min_nodes)
        skipped.extend((sv, times_obs[idx[q]], int(why[q]), 'position') for q in np.flatnonzero(why))
        pos = np.full((tq.size, 3), np.nan)
        vel = np.full((tq.size, 3), np.nan)
        for a in np.unique(lo[lo >= 0]):
            sel = np.flatnonzero(lo == a)
            ta, Xa = t_sv[a:hi[sel[0]]], XYZ[a:hi[sel[0]]]
            tqa = np.ascontiguousarray(tq[sel])
            pos[sel] = lagrange_local_vec(ta, Xa, tqa, m=int(m), max_gap=max_gap)
            # velocity by central difference with a 0.5 s half step on time relative to the first
            # node of the arc, where t ± h is exact. On absolute Unix seconds (~1.7e9 s, ulp
            # 2.4e-7 s) the former ±5 µs step (DT_VEL) was realised as 42 ulp and made every
            # velocity 0.136 % too large (PPP-065). Truncation error |d³r/dt³|·h²/6 ≈ 1e-9 |v|.
            t0, h = ta[0], 0.5
            tr = np.ascontiguousarray(ta - t0)
            r_m = lagrange_local_vec(tr, Xa, np.ascontiguousarray(tqa - t0 - h), m=int(m), max_gap=max_gap)
            r_p = lagrange_local_vec(tr, Xa, np.ascontiguousarray(tqa - t0 + h), m=int(m), max_gap=max_gap)
            vel[sel] = (r_p - r_m) / (2.0 * h)   # [m/s]
        out_xyz[idx, :] = pos

        if has_clk:
            # linear interpolation between the two valid clock nodes around the epoch
            tc, cc, te, _, _ = _sp3_lagrange_clock_nodes(g, gap_factor)
            clk_bad, clk_mis = _sp3_clock_intervals(tc, cc, max_gap, dt_med, clk_common.get(str(sv)[:1]))
            clo, chi, cwhy = _sp3_locate(tc, tq, max_gap, 2)
            skipped.extend((sv, times_obs[idx[q]], int(cwhy[q]), 'clock')
                           for q in np.flatnonzero((cwhy != 0) & (why == 0)))
            ok = clo >= 0
            clk = np.full(tq.size, np.nan)
            if ok.any():
                k = np.searchsorted(tc, tq[ok])
                i0 = np.clip(k - 1, clo[ok], chi[ok] - 2)
                i1 = i0 + 1
                w = (tq[ok] - tc[i0]) / (tc[i1] - tc[i0])
                clk[ok] = cc[i0] + w * (cc[i1] - cc[i0])
                # an epoch on a node takes the node value
                on = (k < tc.size) & (tc[np.minimum(k, tc.size - 1)] == tq[ok])
                clk[np.flatnonzero(ok)[on]] = cc[k[on]]
                # no clock inside a clock discontinuity (see SP3_CLOCK_JUMP_TOL_M)
                cut = (clk_bad[i0] != 0) & ~on
                for q, k0 in zip(np.flatnonzero(ok)[cut], i0[cut]):
                    clk[q] = np.nan
                    if why[q]:
                        continue
                    if clk_bad[k0] == _SP3_SKIP_CLOCK_JUMP:
                        skipped.append((sv, times_obs[idx[q]], _SP3_SKIP_CLOCK_JUMP, 'clock',
                                        (te[k0], te[k0 + 1], float(clk_mis[k0]))))
                    else:
                        skipped.append((sv, times_obs[idx[q]], int(clk_bad[k0]), 'clock'))
            out_clk[idx] = clk

        out_vel[idx, :] = vel

        # relativistic clock correction: -2 * (r·v) / c^2 [s]
        # (r, v in ECEF – standard PPP practice for "eccentricity correction")
        dot_rv = np.einsum('ij,ij->i', pos, vel)   # [m^2/s]
        clk_rel = -2.0 * dot_rv / (C_LIGHT**2)    # [s]
        out_rel[idx] = clk_rel

    msg = _sp3_skip_message(skipped, "SP3 interpolation (Lagrange)")
    if msg:
        # one warning per run for a set of (satellite, reason): the emission-time call of a session repeats the
        # satellites of its reception-time call with epochs shifted by the signal travel time
        key = ("sp3_interpolation_skipped", frozenset((str(s[0]), int(s[2])) for s in skipped))
        _notes.warn_once(owner, key, msg, RuntimeWarning, stacklevel=3)

    clk_corrected = out_clk + out_rel
    out = pd.DataFrame(
        np.column_stack([out_xyz,
                         out_clk.reshape(-1,1),
                         clk_corrected.reshape(-1,1),
                         out_vel,
                         out_rel.reshape(-1,1)]),
        index=obs_sorted.index,
        columns=out_cols
    )
    return out

def lagrange_reception_interp(obs:pd.DataFrame,  sp3_df:pd.DataFrame,
                             degree:Optional[int]=12, gap_factor:Optional[Union[float,int]]=3.0, owner=None):
    """

    :param obs: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with observation data
    :param sp3_df: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with SP3 data
    :param degree: Optional[int], degree of interpolation, default: 12
    :param gap_factor: Optional[Union[float,int]], gap factor for interpolation, default: 3.0
    :param owner: keeper of the run's warn-once state (the session's configuration; see :func:`lagrange_interp`),
        default None: every call warns
    :return: tuple(pd.DataFrame, index: pd.MultiIndex (time, sv),pd.DataFrame, index: pd.MultiIndex (time,sv)),
    tupe of dataframes: one with observation data, satellite positions and clocks,
    and second dataframe with interpolated positions and clocks
    """
    interp = lagrange_interp(obs=obs,
                             sp3=sp3_df,
                             m=degree,
                             gap_factor=gap_factor,
                             owner=owner)
    return obs.join(interp), interp

def lagrange_emission_interp(obs:pd.DataFrame, positions:pd.DataFrame, sp3_df:pd.DataFrame,
                             degree:Optional[int]=12, gap_factor:Optional[Union[float,int]]=3.0,
                             mode:Optional[str]=None, owner=None):
    """

    :param obs: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with observation data
    :param positions: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with reception time positions
    :param sp3_df: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with SP3 data
    :param degree: Optional[int], degree of interpolation, default: 12
    :param gap_factor: Optional[Union[float,int]], gap factor for interpolation, default: 3.0
    :param mode: Optional[str], frequency mode picking the code band; falls back to
        ``obs.attrs['mode']`` and then to band order
    :param owner: keeper of the run's warn-once state (the session's configuration; see :func:`lagrange_interp`),
        default None: every call warns
    :return: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with observation data, satellite positions and clocks
    """
    interp_cols = ['x', 'y', 'z', 'clk_raw', 'clk', 'vx', 'vy', 'vz', 'clk_rel']
    obs_crd = obs.join(positions)
    obs_crd.reset_index(inplace=True)
    obs_crd['__rowid'] = np.arange(len(obs_crd), dtype=np.int64)
    if 'P3' in obs_crd.columns:
        obs_crd['t_em'] = (obs_crd['P3'] / C_LIGHT) + (obs_crd['clk'])
    else:
        col = _select_code_column(obs_crd, mode if mode is not None else obs.attrs.get("mode"))
        obs_crd['t_em'] = (obs_crd[col] / C_LIGHT) + (obs_crd['clk'])
    obs_crd['time_em'] = obs_crd['time'] - pd.to_timedelta(obs_crd['t_em'].values, unit='s')
    obs_crd['time_rec'] = obs_crd['time'].copy()

    obs_for_interp = obs_crd.copy()
    obs_for_interp['time'] = obs_for_interp['time_em'].copy()
    obs_for_interp.set_index(['time', 'sv'], inplace=True)
    valid_time = obs_for_interp.index.get_level_values('time').notna()

    obs_final = obs_crd.drop(columns=[c for c in interp_cols if c in obs_crd.columns])
    obs_final = obs_final.set_index('__rowid')
    for col in interp_cols:
        obs_final[col] = np.nan

    if valid_time.any():
        interp_input = obs_for_interp.loc[valid_time]
        interp_em = lagrange_interp(
            obs=interp_input,
            sp3=sp3_df,
            m=degree,
            gap_factor=gap_factor,
            owner=owner,
        )
        rowids = interp_input.sort_index()['__rowid'].to_numpy(dtype=np.int64, copy=False)
        obs_final.loc[rowids, interp_em.columns] = interp_em.to_numpy(copy=False)

    obs_final.reset_index(inplace=True)
    obs_final['time'] = obs_final['time_rec'].copy()
    obs_final = obs_final.drop(['__rowid', 'time_rec', 'time_em'], axis=1)
    obs_final.set_index(['time', 'sv'], inplace=True)
    return obs_final
