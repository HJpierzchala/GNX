"""A-priori receiver position from code observations.

The sessions need an a-priori position for the elevations (mask, weights, troposphere, ionosphere), for the
Earth-rotation correction of the satellite positions and as the EKF start of PPP. It used to come only from the
RINEX header (APPROX POSITION XYZ): without that record PPP and TEC crashed far downstream and
SPP stopped with a ValueError.

:func:`code_position` computes the position from the satellite states alone: the frames after the satellite
interpolation and before ``coordinates.CustomWrapper`` (columns x, y, z [m, ECEF at emission, not rotated], clk
[s, with the relativistic term] and the code - ``P3``, the mode's ionosphere-free code or single code), none of
which depends on the receiver position. Least squares from the geocentre on at most :data:`MAX_EPOCHS` epochs
spread over the session, with the Earth rotation during the signal travel (omega * P / c), one receiver clock per
system present at the epoch and, once the position is near the surface, an elevation mask of :data:`MASK_DEG`
and a 2.4 m / sin(el) troposphere. The median of the epoch solutions is the position and the median distance from
it the spread. A moving receiver (``kinematic=True``; kinematic PPP, ``q_crd > 0``) takes the first
:data:`MIN_EPOCHS_REPLACE` epoch solutions of the session instead: its start position, with no limit on the spread.
The code solution is the antenna (reference point plus phase centre); :func:`resolve_position` moves it to the
marker by the header's ANTENNA: DELTA H/E/N before comparing it with APPROX POSITION XYZ (a marker position) and
before using it (NKLG 2024-132, DELTA H 3.04 m: the code solution 4.1 m above the SINEX marker, 1.1 m after
the offset; a mast of several metres would otherwise exceed the 10 m threshold).

:func:`resolve_position` chooses the session's position, in this order:

1. ``Config.approx_position`` when set (ECEF metres);
2. the header position, when it is plausible (|r| within ``rinex_obs.APPROX_XYZ_MIN_M`` .. ``MAX_M``) and within
   the threshold T of the code position (or when the code position cannot be computed);
3. the code position, with one :class:`gnx_py.rinex_obs.RinexWarning`. A header position is replaced only by a
   code position of at least :data:`MIN_EPOCHS_REPLACE` epoch solutions; without a header a code position of fewer
   epochs is used with a warning that names it weak.

T = max(``floor_m``, 3 sqrt(p_crd), 4 spread): PPP passes 10 m and its ``p_crd`` (10 m^2 combined, 100 m^2
uncombined) for a static dual-frequency run on precise orbits and clocks, SPP and TEC 100 m. With a
single-frequency code the code position carries the ionospheric delay (several metres in height, tens of metres in
a storm), broadcast clocks and group delays leave metres (BeiDou broadcast: 10.7-25.1 m at BRUX 2024-035), a
moving receiver's start rests on 5 epochs, and a close-pair ionosphere-free code (L2L5, E5aE5b, B2aB2b) amplifies
code noise and biases 16.6-27.5 times (14.0-14.2 m at BRUX and NYA2 2024-035 with precise orbits), so PPP passes a
floor of 100 m for such runs too.
Measured (BRUX 2024-035): the code position is 0.44-1.18 m from the SINEX coordinates, the
RINEX headers of the validation data 0.14-3.07 m; PPP after 1 h from the code position equals the run from the header within 0.00-0.03 cm
with precise orbits (0.5 cm with broadcast orbits). Written from first principles; no GPL code.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.conversion import ecef2geodetic
from ..io.rinex_obs import APPROX_XYZ_MAX_M, APPROX_XYZ_MIN_M, RinexWarning

__all__ = ["MASK_DEG", "MAX_EPOCHS", "MAX_PDOP", "MAX_SPREAD_M", "MIN_EPOCHS_REPLACE", "antenna_offset_ecef",
           "code_position", "plausible_position", "resolve_position"]

log = logging.getLogger(__name__)

C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5
#: Epochs used by the code solution (spread evenly over the session).
MAX_EPOCHS = 20
#: Elevation mask of the code solution once the position is near the surface [deg].
MASK_DEG = 10.0
#: Largest median distance of the epoch solutions from their median accepted as a static solution [m].
MAX_SPREAD_M = 100.0
#: An epoch solution needs at least one redundant satellite and a PDOP not above this value.
MAX_PDOP = 10.0
#: Epoch solutions needed before a plausible header position is replaced by the code position (BDS-2 B2I at BRUX:
#: 4 epochs with 5 satellites gave a solution 2.9 km off, spread 39 m); also the epochs of a moving receiver's
#: start position.
MIN_EPOCHS_REPLACE = 5
_E2 = 6.69437999014e-3


def plausible_position(xyz) -> bool:
    """True for three finite coordinates whose geocentric distance lies on the Earth's surface
    (``rinex_obs.APPROX_XYZ_MIN_M`` .. ``APPROX_XYZ_MAX_M``)."""
    if xyz is None:
        return False
    a = np.asarray(xyz, dtype=float).reshape(-1)
    if a.size != 3 or not np.all(np.isfinite(a)):
        return False
    return bool(APPROX_XYZ_MIN_M <= np.linalg.norm(a) <= APPROX_XYZ_MAX_M)


def _up(x: np.ndarray) -> np.ndarray:
    lat = np.arctan2(x[2], np.hypot(x[0], x[1]) * (1.0 - _E2))
    lon = np.arctan2(x[1], x[0])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _rows(states_by_sys: Dict[str, Optional[pd.DataFrame]], modes: Dict[str, Optional[str]]) -> pd.DataFrame:
    from ..core.coordinates import _select_code_column

    rows = []
    for s, df in states_by_sys.items():
        if df is None or len(df) == 0 or not {"x", "y", "z", "clk"} <= set(df.columns):
            continue
        try:
            col = "P3" if "P3" in df.columns else _select_code_column(df, modes.get(s))
        except KeyError:
            continue
        d = pd.DataFrame({
            "time": np.asarray(df.index.get_level_values("time"), dtype="datetime64[ns]"),
            "sys": s,
            "P": df[col].to_numpy(dtype=float),
            "x": df["x"].to_numpy(dtype=float), "y": df["y"].to_numpy(dtype=float),
            "z": df["z"].to_numpy(dtype=float), "clk": df["clk"].to_numpy(dtype=float),
        })
        d = d[np.isfinite(d[["P", "x", "y", "z", "clk"]].to_numpy()).all(axis=1) & (d["P"] > 1.0e7)]
        if len(d):
            rows.append(d)
    if not rows:
        raise ValueError("no satellite with a position, a clock and a code observation")
    return pd.concat(rows, ignore_index=True)


def _solve_epoch(X: np.ndarray, P: np.ndarray, si: np.ndarray, ns: int, mask_deg: float) -> Optional[tuple]:
    """Position, clocks [m], residual RMS [m] and the number of used satellites of one epoch, or None.

    One clock per system with a used satellite at this epoch (a system absent at the epoch, or below the mask,
    adds no column), and at least one redundant satellite."""
    x = np.zeros(3)
    cdt = np.zeros(ns)
    use = np.ones(P.size, dtype=bool)
    sel = np.eye(ns)[si]

    def enough(mask) -> bool:
        return int(mask.sum()) >= 3 + np.unique(si[mask]).size + 1

    for _ in range(3):                              # passes: from the geocentre, then with mask and outlier screen
        for _ in range(15):
            if not enough(use):
                return None
            present = np.unique(si[use])
            rho = np.linalg.norm(X - x, axis=1)
            u = (X - x) / rho[:, None]
            tro = np.zeros(P.size)
            if np.linalg.norm(x) > 6.0e6:
                el = np.arcsin(np.clip(u @ _up(x), -1.0, 1.0))
                tro = 2.4 / np.sin(np.clip(el, np.deg2rad(3.0), None))
            H = np.column_stack([-u, sel[:, present]])
            v = P - rho - cdt[si] - tro
            dx, *_ = np.linalg.lstsq(H[use], v[use], rcond=None)
            x += dx[:3]
            cdt[present] += dx[3:]
            if np.linalg.norm(dx[:3]) < 1e-3:
                break
        rho = np.linalg.norm(X - x, axis=1)
        u = (X - x) / rho[:, None]
        el = np.arcsin(np.clip(u @ _up(x), -1.0, 1.0)) if np.linalg.norm(x) > 6.0e6 else np.full(P.size, np.pi / 2)
        tro = 2.4 / np.sin(np.clip(el, np.deg2rad(3.0), None))
        res = P - rho - cdt[si] - tro
        rms = float(np.sqrt(np.mean(res[use] ** 2)))
        new = (el > np.deg2rad(mask_deg)) & (np.abs(res - np.median(res[use])) <= max(30.0, 4.0 * rms))
        if not enough(new) or np.array_equal(new, use):
            break
        use = new
    if not enough(use) or not np.all(np.isfinite(x)):
        return None
    present = np.unique(si[use])
    rho = np.linalg.norm(X - x, axis=1)
    H = np.column_stack([-(X - x) / rho[:, None], sel[:, present]])[use]
    try:
        pdop = float(np.sqrt(np.trace(np.linalg.inv(H.T @ H)[:3, :3])))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(pdop) or pdop > MAX_PDOP:
        return None
    return x, cdt, rms, int(use.sum())


def _epoch_solution(e: pd.DataFrame, systems, mask_deg: float) -> Optional[tuple]:
    P = e["P"].to_numpy()
    th = OMEGA_E * P / C_LIGHT
    xs, ys = e["x"].to_numpy(), e["y"].to_numpy()
    X = np.column_stack([np.cos(th) * xs + np.sin(th) * ys, -np.sin(th) * xs + np.cos(th) * ys, e["z"].to_numpy()])
    Pc = P + C_LIGHT * e["clk"].to_numpy()
    si = np.searchsorted(systems, e["sys"].to_numpy())
    return _solve_epoch(X, Pc, si, len(systems), mask_deg)


def code_position(states_by_sys: Dict[str, Optional[pd.DataFrame]], modes: Dict[str, Optional[str]],
                  max_epochs: int = MAX_EPOCHS, mask_deg: float = MASK_DEG,
                  kinematic: Optional[bool] = False) -> tuple:
    """Receiver antenna position [m] from code observations and satellite states, starting at the geocentre.

    ``states_by_sys``: ``{system: frame}`` after the satellite interpolation and before ``CustomWrapper``
    (index levels 'sv' and 'time' in either order; x, y, z, clk and P3 or the mode's code columns).
    ``kinematic``: False - the median of up to ``max_epochs`` epoch solutions spread over the session (a static
    receiver); True - the median of the first :data:`MIN_EPOCHS_REPLACE` epoch solutions (the start position of a
    moving receiver, no spread limit); None - static, and the start position when the static solutions spread by
    more than :data:`MAX_SPREAD_M` (SPP: either kind of receiver).
    Returns ``(xyz, info)`` with ``info = {'epochs', 'spread_m', 'rms_m', 'systems', 'kinematic'}``; ValueError
    when no epoch has enough satellites, the solution is off the Earth's surface or (static) the epoch solutions
    spread by more than :data:`MAX_SPREAD_M`."""
    A = _rows(states_by_sys, modes)
    systems = sorted(A["sys"].unique())
    per_epoch = A.groupby("time")["sys"]
    counts, n_sys = per_epoch.size(), per_epoch.nunique()
    good = counts.index[(counts.to_numpy() >= 4 + n_sys.to_numpy())].to_numpy()
    if good.size == 0:
        raise ValueError(f"no epoch with a redundant satellite (4 + one per system; {', '.join(systems)})")
    groups = {pd.Timestamp(k): v for k, v in A.groupby("time", sort=False).indices.items()}

    def solve(order, wanted):
        sols, tried = [], 0
        for k in order:
            if len(sols) >= wanted:
                break
            tried += 1
            out = _epoch_solution(A.iloc[groups[pd.Timestamp(good[k])]], systems, mask_deg)
            if out is not None:
                sols.append(out)
        if not sols:
            raise ValueError(f"no epoch solution with a redundant satellite and PDOP <= {MAX_PDOP:g} ({tried} "
                             f"epochs tried)")
        pos = np.array([s[0] for s in sols])
        xyz = np.median(pos, axis=0)
        spread = float(np.median(np.linalg.norm(pos - xyz, axis=1)))
        rms = float(np.median([s[2] for s in sols]))
        return xyz, spread, rms, len(sols)

    budget = 10 * max_epochs
    moving = bool(kinematic)
    if moving:
        xyz, spread, rms, n = solve(range(min(good.size, budget)), MIN_EPOCHS_REPLACE)
    else:
        # epochs spread over the session; when an epoch gives no solution (no redundancy, PDOP), further epochs
        # are tried, at most 10 x max_epochs in all
        first = np.unique(np.linspace(0, good.size - 1, min(max_epochs, good.size)).round().astype(int))
        rest = np.setdiff1d(np.arange(good.size), first)
        order = np.r_[first, rest[np.unique(np.linspace(0, rest.size - 1, min(9 * max_epochs, rest.size))
                                            .round().astype(int))] if rest.size else rest]
        xyz, spread, rms, n = solve(order, max_epochs)
        if spread > MAX_SPREAD_M and kinematic is None:
            moving = True
            xyz, spread, rms, n = solve(range(min(good.size, budget)), MIN_EPOCHS_REPLACE)
    r = float(np.linalg.norm(xyz))
    if not (APPROX_XYZ_MIN_M <= r <= APPROX_XYZ_MAX_M):
        raise ValueError(f"the code solution did not converge (|r| {r:.0f} m, epoch spread {spread:.1f} m)")
    if not moving and spread > MAX_SPREAD_M:
        raise ValueError(f"the code solutions of {n} epochs spread by {spread:.1f} m (more than {MAX_SPREAD_M:g} m): "
                         f"a moving receiver? Kinematic PPP (q_crd > 0) starts from the receiver's first position")
    return xyz, {"epochs": n, "spread_m": spread, "rms_m": rms, "systems": "".join(systems), "kinematic": moving}


def antenna_offset_ecef(xyz, delta_hen) -> np.ndarray:
    """ECEF vector [m] of the antenna offset ``delta_hen`` (ANTENNA: DELTA H/E/N: up, east, north from the marker)
    at the position ``xyz``; zeros when ``delta_hen`` is None or not three finite numbers."""
    try:
        h, e, n = (float(v) for v in np.asarray(delta_hen, dtype=float).reshape(3))
    except (TypeError, ValueError):
        return np.zeros(3)
    if not np.all(np.isfinite([h, e, n])):
        return np.zeros(3)
    x = np.asarray(xyz, dtype=float).reshape(3)
    lat = np.arctan2(x[2], np.hypot(x[0], x[1]) * (1.0 - _E2))
    lon = np.arctan2(x[1], x[0])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return e * east + n * north + h * _up(x)


def resolve_position(header_xyz, header_flh, states_by_sys: Dict[str, Optional[pd.DataFrame]],
                     modes: Dict[str, Optional[str]], *, floor_m: float, p_crd: Optional[float] = None,
                     approx_position=None, obs_path=None, session: str = "PPP", delta_hen=None,
                     kinematic: Optional[bool] = False) -> tuple:
    """The session's a-priori marker position: ``(xyz, flh, source)`` with ``source`` in {'approx_position',
    'header', 'code'} (order and threshold in the module docstring). ``header_flh`` (the reader's geodetic
    position) is returned unchanged with the header position. ``delta_hen``: ANTENNA: DELTA H/E/N of the header
    (the code position is moved from the antenna to the marker by it). ``kinematic``: see :func:`code_position`.
    ValueError when neither the header nor the code observations give a position."""
    if approx_position is not None:
        xyz = np.asarray(approx_position, dtype=float).reshape(3)
        return xyz, ecef2geodetic(ecef=xyz, deg=True), "approx_position"
    good_header = plausible_position(header_xyz)
    try:
        code, info = code_position(states_by_sys, modes, kinematic=kinematic)
        code = code - antenna_offset_ecef(code, delta_hen)
        why = None
    except ValueError as exc:
        code, info, why = None, None, str(exc)
    name = obs_path if obs_path is not None else "the observation file"
    if code is not None:
        what_code = (f"{info['epochs']} epochs" + (", start of a moving receiver" if info["kinematic"] else "")
                     + f", spread {info['spread_m']:.2f} m")
    if good_header:
        hxyz = np.asarray(header_xyz, dtype=float).reshape(3)
        hflh = header_flh
        if hflh is None or np.asarray(hflh, dtype=float).size != 3:
            hflh = ecef2geodetic(ecef=hxyz, deg=True)
        if code is None:
            log.info("%s: header position of %s not checked against the code observations (%s)", session, name, why)
            return hxyz, hflh, "header"
        d = float(np.linalg.norm(hxyz - code))
        limit = max(float(floor_m), 3.0 * float(np.sqrt(p_crd)) if p_crd else 0.0, 4.0 * info["spread_m"])
        if info["epochs"] < MIN_EPOCHS_REPLACE:
            log.info("%s: header position of %s kept: %.1f m from a code position of %d epoch(s) only (weak "
                     "geometry)", session, name, d, info["epochs"])
            return hxyz, hflh, "header"
        if d <= limit:
            log.info("%s: header position of %s is %.2f m from the code position (limit %.1f m)", session, name, d,
                     limit)
            return hxyz, hflh, "header"
        warnings.warn(
            f"{session}: APPROX POSITION XYZ of {name} is {d:.1f} m from the marker position from code "
            f"observations (more than {limit:.1f} m); a-priori position from code observations ({what_code}). Set "
            f"approx_position (ECEF metres) to impose one.",
            RinexWarning, stacklevel=3)
        return code, ecef2geodetic(ecef=code, deg=True), "code"
    if header_xyz is None:
        what = f"APPROX POSITION XYZ missing in {name}"
    else:
        a = np.asarray(header_xyz, dtype=float).reshape(-1)
        what = (f"APPROX POSITION XYZ of {name} is not a position on the Earth's surface "
                f"({', '.join(f'{v:.3f}' for v in a)})")
    if code is None:
        raise ValueError(f"{session}: {what}, and the a-priori position cannot be computed from the code "
                         f"observations ({why}); set approx_position (ECEF metres)")
    weak = ""
    if info["epochs"] < MIN_EPOCHS_REPLACE:
        weak = (f"; weak: only {info['epochs']} epoch solution(s) with enough satellites, the position may be tens "
                f"of metres off - set approx_position (ECEF metres) if the solution does not converge")
    warnings.warn(f"{session}: {what}; a-priori position from code observations ({what_code}){weak}",
                  RinexWarning, stacklevel=3)
    return code, ecef2geodetic(ecef=code, deg=True), "code"
