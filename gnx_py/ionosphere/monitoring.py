from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import warnings

from .utils import _check_choice, _check_number
from .ionex import (DEFAULT_SHELL_HEIGHT_KM as _CODE_IPP_SHELL_KM, MSLM_ALPHA as _MSLM_ALPHA,
                    MSLM_HEIGHT_KM as _MSLM_HEIGHT_KM, mapping_mslm as _mapping_mslm, mapping_slm as _mapping_slm)

# The plotting code moved to ``gnx_py.viz`` (see the bottom of this module), so
# nothing here uses matplotlib any more. These imports stay because
# ``gnx_py/ionosphere/__init__.py`` does ``from .monitoring import *`` and its
# ``__all__`` snapshots these names as part of the historical public surface:
# dropping them breaks ``from gnx_py.ionosphere import *``, and therefore
# ``import gnx_py``. Pruning them is an API change and needs a deprecation step.
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import MatplotlibDeprecationWarning

logger = logging.getLogger(__name__)

# --- SIDX, GIX ---
Re = 6371e3
IONO_H = 450e3
MIN_ELEV_DEG = 10.0
MAX_PAIR_PER_EPOCH = 200000  # random thinning fuse
# --- ROT/ROTI ---
ROTI_WINDOW_MIN = 5            # window for STD
ROTI_MIN_SAMPLES = 3           # min. number of ROTs in the window to calculate STD
MAX_GAP_S       = 180.0        # cut off jumps with large time gaps
SPLIT_BY_DAY    = True         # do not differentiate across midnight (GPST day; reset per day)
# Time scale of every epoch handled here: naive GPS time (GPST), the scale of the TEC session
# (RINEX receiver epochs). Outputs carry ``attrs["time_scale"] = "GPST"``.
TIME_SCALE      = "GPST"
DDOF_POP        = 0            # according to the definition ROTI = sqrt(<ROT^2>-<ROT>^2)

# The plotting constants FIGSIZE and DPI now live in ``gnx_py.viz.core`` and are
# imported back at the bottom of this module, so they keep one definition.


@dataclass
class Region:
    """
    Definition of the region covered by the ROTI/GIX/SIDX measurement.

    Latitudes in [-90, 90] deg with ``lat_min <= lat_max``; longitudes in
    [-180, 180] deg. ``lon_min > lon_max`` describes a box crossing the
    antimeridian (e.g. ``lon_min=170, lon_max=-170`` spans 170E..190E); note
    that swapped bounds such as ``lon_min=20, lon_max=0`` therefore select the
    other 340 deg of longitude, not an error.
    The bounds are checked on construction, at the entry of
    :func:`compute_sid`/:func:`compute_gix`/:func:`compute_roti_links` and
    again by :meth:`contains`, so a bound changed afterwards is validated too.
    """
    lat_min: float = -90.0
    lat_max: float =  90.0
    lon_min: float = -180.0
    lon_max: float =  180.0

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        for name in ('lat_min', 'lat_max'):
            _check_number(f"Region.{name}", getattr(self, name), ge=-90, le=90, unit="deg")
        for name in ('lon_min', 'lon_max'):
            _check_number(f"Region.{name}", getattr(self, name), ge=-180, le=180, unit="deg")
        if self.lat_min > self.lat_max:
            raise ValueError(f"Region.lat_min must be <= Region.lat_max "
                             f"(got lat_min={self.lat_min!r}, lat_max={self.lat_max!r})")

    def contains(self, lat, lon) -> np.ndarray:
        self._validate()
        lon = ((lon + 180) % 360) - 180
        in_lat = (lat >= self.lat_min) & (lat <= self.lat_max)
        if self.lon_min <= self.lon_max:
            in_lon = (lon >= self.lon_min) & (lon <= self.lon_max)
        else:  # box crossing the antimeridian
            in_lon = (lon >= self.lon_min) | (lon <= self.lon_max)
        return in_lat & in_lon

def _check_percentile_pair(name: str, q) -> Tuple[float, float]:
    """Raise ValueError unless ``q`` is (low, high) percentiles with 0 <= low <= high <= 100."""
    msg = f"{name} must be a pair (low, high) of percentiles in [0, 100] (got {q!r})"
    try:
        lo, hi = q
        _check_number(name, lo, ge=0, le=100)
        _check_number(name, hi, ge=0, le=100)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if lo > hi:
        raise ValueError(f"{name} must satisfy low <= high (got {q!r})")
    # fractions such as (0.05, 0.95): a non-integer value with high <= 1
    # (whole-number pairs such as (0, 1) are valid percentiles)
    if hi <= 1 and not (float(lo).is_integer() and float(hi).is_integer()):
        raise ValueError(f"{name} are percentiles in [0, 100], not fractions (got {q!r}); "
                         f"did you mean ({lo * 100:g}, {hi * 100:g})?")
    return float(lo), float(hi)


def _check_region(region) -> None:
    """Raise ValueError unless ``region`` is None or a :class:`Region` with valid bounds."""
    if region is None:
        return
    if not isinstance(region, Region):
        raise ValueError(f"region must be a Region or None (got {type(region).__name__}: {region!r})")
    region._validate()


# --- time scale: naive GPST ---
def _gps_minus_utc_s(utc_naive: np.ndarray) -> np.ndarray:
    """GPS-UTC [s] at naive-UTC ``datetime64[ns]`` epochs (leap-second table of :mod:`gnx_py.time`)."""
    from ..core.time import _LEAP_TABLE

    since = np.array([np.datetime64(d.replace(tzinfo=None), 'ns') for d, _ in _LEAP_TABLE])
    value = np.array([0] + [v for _, v in _LEAP_TABLE], dtype=np.int64)
    return value[np.searchsorted(since, np.asarray(utc_naive, dtype='datetime64[ns]'), side='right')]


def _aware_warning(who: str, name: str, leap) -> None:
    leaps = sorted({int(v) for v in np.atleast_1d(leap)})
    warnings.warn(
        f"{who}: {name!r} is timezone-aware; it is converted to UTC and then to GPS time "
        f"(GPST = UTC + {'/'.join(str(v) for v in leaps)} s). GNX monitoring epochs are naive GPST, "
        "the time scale of the TEC session; pass naive GPST epochs to avoid this conversion.",
        UserWarning, stacklevel=4)


def _to_gpst(values, *, name: str = "time", who: str = "") -> pd.Series:
    """Epochs (a column) as naive GPST ``datetime64[ns]``.

    Naive epochs are taken as GPST and returned unchanged. Timezone-aware epochs are converted to
    UTC and then to GPST by adding the GPS-UTC leap seconds, with one UserWarning. Strings are
    parsed; numbers raise TypeError (they would silently become 1970 epochs).
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(s.dtype):
        if pd.api.types.is_numeric_dtype(s.dtype) and len(s):
            raise TypeError(f"{who}: {name!r} is numeric ({s.dtype}), not datetimes; convert the epochs "
                            "explicitly (naive GPST)")
        try:
            s = pd.to_datetime(s)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{who}: {name!r} cannot be read as datetimes (dtype {s.dtype}): {exc}") from None
    if getattr(s.dt, 'tz', None) is not None:
        utc = s.dt.tz_convert('UTC').dt.tz_localize(None)
        leap = _gps_minus_utc_s(utc.to_numpy())
        if len(s):
            _aware_warning(who, name, leap[~pd.isna(utc.to_numpy())])
        s = utc + pd.to_timedelta(leap, unit='s')
    return s.astype('datetime64[ns]')


def _epoch_to_gpst(epoch, *, name: str = "epoch", who: str = "", warn: bool = True) -> pd.Timestamp:
    """One epoch as a naive GPST Timestamp (same rules as :func:`_to_gpst`).

    ``warn=False`` skips the UserWarning for a timezone-aware epoch: for a caller that converts an epoch a
    function before it already converted (and warned about).
    """
    if isinstance(epoch, (bool, np.bool_)) or isinstance(epoch, (int, float, np.integer, np.floating)):
        raise TypeError(f"{who}: {name} is numeric ({epoch!r}), not a datetime (naive GPST expected)")
    try:
        ts = pd.Timestamp(epoch)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{who}: {name} {epoch!r} cannot be read as a datetime") from exc
    if ts is pd.NaT:
        raise ValueError(f"{who}: {name} is NaT")
    if ts.tzinfo is not None:
        utc = ts.tz_convert('UTC').tz_localize(None)
        leap = _gps_minus_utc_s(np.array([utc.to_datetime64()]))[0]
        if warn:
            _aware_warning(who, name, leap)
        ts = utc + pd.Timedelta(seconds=int(leap))
    return ts


def _gpst_to_tai_time(t: pd.Timestamp) -> Time:
    """astropy Time of a naive GPST epoch: GPST = TAI - 19 s (exact, no leap-second table)."""
    return Time(np.datetime64(pd.Timestamp(t) + pd.Timedelta(seconds=19), 'ns'), scale='tai')


def _tag(obj, **extra):
    obj.attrs["time_scale"] = TIME_SCALE
    obj.attrs.update(extra)
    return obj


# --- mapping functions ---
_MAPPINGS = ['slm', 'mslm']


def mapping_M(elev_deg: np.ndarray, Re_m: float = Re, H_m: float = IONO_H) -> np.ndarray:
    """
    Single-layer (SLM) mapping function STEC/VTEC, sin z' = Re/(Re+H) cos E.

    Delegates to :func:`gnx_py.ionosphere.ionex.mapping_slm`; the elevation is clipped to
    [0.01, 89.99] deg. A pandas Series input returns a Series on the same index (as before).

    :param elev_deg: elevation in degrees
    :param Re_m: mean earth radius in meters
    :param H_m: Ionospheric shell height in meters
    :return: mapping function value
    """
    el = np.clip(np.asarray(elev_deg, dtype=float), 0.01, 89.99)
    out = _mapping_slm(el, R=Re_m / 1e3, H=H_m / 1e3)
    if isinstance(elev_deg, pd.Series):
        return pd.Series(out, index=elev_deg.index, name=elev_deg.name)
    return out


def _mapping_values(elev_deg, mapping: str, iono_h_m: float) -> np.ndarray:
    """M at ``iono_h_m``: 'slm' (single layer) or 'mslm' (Schaer's modified single layer, alpha 0.9782,
    mapping height ``iono_h_m``; the NTCM-G form at 450 km)."""
    if mapping == 'slm':
        return mapping_M(elev_deg, H_m=iono_h_m)
    el = np.clip(np.asarray(elev_deg, dtype=float), 0.01, 89.99)
    return _mapping_mslm(el, R=Re / 1e3, H=iono_h_m / 1e3, alpha=_MSLM_ALPHA)


def _implied_shell_height_m(M, elev_deg, Re_m: float = Re) -> np.ndarray:
    """Invert the single-layer M = 1/sqrt(1 - (Re cos E/(Re+H))^2) for H [m]."""
    e = np.deg2rad(np.clip(np.asarray(elev_deg, dtype=float), 0.01, 89.99))
    m2 = np.asarray(M, dtype=float) ** 2
    # explicit mask: M = 0 gives 1/M^2 = inf as before (then the clip floor), without a division by zero
    inv_m2 = np.divide(1.0, m2, out=np.full(np.shape(m2), np.inf), where=m2 != 0)
    a = np.sqrt(np.clip(1.0 - inv_m2, 1e-12, None))
    return Re_m * np.cos(e) / a - Re_m


def _mapping_candidates(ev, mapping: Optional[str], iono_h_m: float):
    """(label, M) pairs an 'M' column may equal. ``mapping=None`` accepts every candidate;
    an explicit ``mapping`` accepts only that function. The CODE MSLM (mapping height
    ``ionex.MSLM_HEIGHT_KM``) is a candidate only when ``iono_h_m`` is the CODE IPP shell
    (``ionex.DEFAULT_SHELL_HEIGHT_KM``, 450 km), the shell its M belongs to."""
    out = []
    if mapping in (None, 'slm'):
        out.append((f"SLM at {iono_h_m / 1e3:g} km", _mapping_values(ev, 'slm', iono_h_m)))
    if mapping in (None, 'mslm'):
        out.append((f"MSLM at {iono_h_m / 1e3:g} km", _mapping_values(ev, 'mslm', iono_h_m)))
        if abs(iono_h_m / 1e3 - _CODE_IPP_SHELL_KM) <= 1e-6:
            el = np.clip(np.asarray(ev, dtype=float), 0.01, 89.99)
            out.append((f"CODE MSLM ({_MSLM_HEIGHT_KM:g} km, alpha {_MSLM_ALPHA:g})",
                        _mapping_mslm(el, R=Re / 1e3, H=_MSLM_HEIGHT_KM, alpha=_MSLM_ALPHA)))
    return out


def _resolve_M(obs: pd.DataFrame, *, mapping: Optional[str], iono_h_m: float, use_M_column: bool, who: str,
               rtol: float = 1e-6) -> np.ndarray:
    """The mapping-function values used by SIDX/GIX.

    No 'M' column (or an all-NaN one, with a UserWarning): ``mapping`` ('slm' when None) at
    ``iono_h_m``. A supplied 'M' is kept when ``use_M_column=True`` or when it equals, within
    ``rtol`` (relative), one of the accepted functions: with ``mapping=None`` the SLM at
    ``iono_h_m``, the MSLM at ``iono_h_m`` or, when ``iono_h_m`` is 450 km, the CODE MSLM
    (506.7 km, alpha 0.9782); with an explicit ``mapping`` only that function ('mslm': at
    ``iono_h_m`` or the CODE MSLM).
    Otherwise a UserWarning names ``iono_h_m`` (and, when the column has the single-layer
    shape, the shell height it implies) and M is recomputed with ``mapping`` at ``iono_h_m``.
    """
    ev = obs['ev'].to_numpy(dtype=float)
    used = mapping or 'slm'
    ref = _mapping_values(ev, used, iono_h_m)
    if 'M' not in obs.columns:
        return ref
    M = obs['M'].to_numpy(dtype=float)
    if use_M_column or M.size == 0:
        return M
    fin = np.isfinite(M)
    if not fin.any():
        warnings.warn(f"{who}: the 'M' column is all NaN; M is recomputed with mapping={used!r} at iono_h_m="
                      f"{iono_h_m / 1e3:g} km", UserWarning, stacklevel=3)
        return ref
    cands = _mapping_candidates(ev, mapping, iono_h_m)
    for _label, c in cands:
        if np.max(np.abs(M[fin] / c[fin] - 1.0)) <= rtol:
            return M
    rel = float(np.max(np.abs(M[fin] / ref[fin] - 1.0)))
    # implied single-layer height: reported only when one SLM height reproduces the column (1e-3)
    sel = fin & (ev < 60.0)
    shape = "it does not have the single-layer shape (no one SLM height reproduces it)"
    if sel.any():
        H = float(np.nanmedian(_implied_shell_height_m(M[sel], ev[sel])))
        if np.isfinite(H) and H > 0:
            fit = mapping_M(ev[fin], H_m=H)
            if np.max(np.abs(M[fin] / fit - 1.0)) <= 1e-3:
                shape = f"as a single-layer M it implies a shell height of ~{H / 1e3:.0f} km"
    accepted = " or ".join(label for label, _c in cands)
    warnings.warn(
        f"{who}: the 'M' column is not the {accepted} mapping function (iono_h_m={iono_h_m / 1e3:g} km, "
        f"mapping={mapping!r}; max relative difference to the recomputed M {rel:.3g}; {shape}); M is "
        f"recomputed with mapping={used!r} at iono_h_m (pass use_M_column=True to keep the column as given)",
        UserWarning, stacklevel=3)
    return ref

def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_m: float) -> np.ndarray:
    """
    Distance on a large circle between two points
    :param lat1: first point latitude
    :param lon1: first point longitude
    :param lat2: second point latitude
    :param lon2: second point longitude
    :param radius_m: great circle radius in meters
    :return: distance in meters
    """
    phi1, lbd1, phi2, lbd2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dphi, dlbd = phi2 - phi1, lbd2 - lbd1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlbd/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return (radius_m * c) / 1000.0

def initial_bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    azimuth between a pair of points
    """
    phi1, lbd1, phi2, lbd2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlbd = lbd2 - lbd1
    y = np.sin(dlbd) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlbd)
    theta = np.arctan2(y, x)
    return (np.rad2deg(theta) + 360) % 360

def spherical_midpoint(lat1, lon1, lat2, lon2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Midpoint between a pair of points

    """
    phi1, lbd1, phi2, lbd2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    x1, y1, z1 = np.cos(phi1)*np.cos(lbd1), np.cos(phi1)*np.sin(lbd1), np.sin(phi1)
    x2, y2, z2 = np.cos(phi2)*np.cos(lbd2), np.cos(phi2)*np.sin(lbd2), np.sin(phi2)
    xm, ym, zm = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
    hyp = np.hypot(xm, ym)
    latm = np.arctan2(zm, hyp)
    lonm = np.arctan2(ym, xm)
    return np.rad2deg(latm), (np.rad2deg(lonm)+540)%360-180

# --- SIDX ---
def compute_sid(
    df: pd.DataFrame,
    region: Optional[Region] = None,
    min_elev_deg: float = MIN_ELEV_DEG,
    iono_h_m: float = IONO_H,
    stec_scale: float = 1.0,
    max_gap_s: Optional[float] = 300,      # reject differences when gap > max_gap_s
    split_on_day_change: bool = True,        # DO NOT count differences across midnight (GPST day)
    agg: str = "median",              # 'median', 'mean', 'max' or 'p95'
    mapping: Optional[Literal['slm', 'mslm']] = None,
    use_M_column: bool = False,
) -> pd.Series:
    """
    SIDX ≈ < dltSTEC / (M·dltt) >  (TECU/min).

    Status:
        Active monitoring helper. It is unit-sensitive: output is TECU/min, not
        TECU/s. Use with leveled STEC or a documented scale factor.

    Differentiation only within the same link (name, sv) and — if enabled —
    within the same GPST day (no differences across the GPST midnight).
    ``sv`` must identify a continuous arc (the TEC session's 'G01_0_4' labels do);
    with plain PRNs two passes of one satellite are joined unless ``max_gap_s``
    separates them.
    Required columns: name, sv, time, stec, ev, lat_ipp, lon_ipp.

    Time: epochs are naive GPST. Timezone-aware epochs are converted to UTC and
    then to GPST (+leap seconds) with one UserWarning; strings are parsed and
    numbers raise TypeError. The result is indexed by naive GPST epochs and
    carries ``attrs["time_scale"] = "GPST"``.

    Mapping function: without an 'M' column (or with an all-NaN one, which warns),
    ``mapping`` ('slm' single layer, the default when ``mapping=None``; 'mslm'
    Schaer's modified single layer with alpha 0.9782) at ``iono_h_m``. A supplied
    'M' is used when ``use_M_column=True`` or when it equals (relative 1e-6) an
    accepted function: with ``mapping=None`` the SLM or the MSLM at ``iono_h_m``
    or, with ``iono_h_m`` = 450 km (the CODE IPP shell), the CODE MSLM (H 506.7 km,
    alpha 0.9782); with an explicit ``mapping`` only that function. Otherwise a UserWarning names ``iono_h_m`` (and the shell height
    the column implies, when it has the single-layer shape) and M is recomputed.

    Raises ValueError (naming the argument) for missing columns, ``min_elev_deg``
    outside [0, 90] deg, non-positive ``iono_h_m``/``stec_scale``/``max_gap_s``,
    an ``agg`` other than 'median', 'mean', 'max', 'p95' or a ``mapping`` other
    than 'slm', 'mslm'.
    """
    need = {'name','sv','time','stec','ev','lat_ipp','lon_ipp'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Columns are missing: {sorted(miss)}")
    _check_number("min_elev_deg", min_elev_deg, ge=0, le=90, unit="deg")
    _check_number("iono_h_m", iono_h_m, gt=0, unit="m")
    _check_number("stec_scale", stec_scale, gt=0)
    _check_number("max_gap_s", max_gap_s, gt=0, unit="s", allow_none=True)
    _check_choice("agg", agg, ['median', 'mean', 'max', 'p95'])
    if mapping is not None:
        _check_choice("mapping", mapping, _MAPPINGS)
    _check_region(region)

    dfa = df.copy()
    dfa['time'] = _to_gpst(dfa['time'], who="compute_sid")

    # elevation mask
    dfa = dfa[dfa['ev'] >= min_elev_deg].copy()
    if dfa.empty:
        logger.info("[IONO MONITOR] metric=SIDX event=no-data-after-elevation-mask")
        return _tag(pd.Series(dtype=float, name='SIDX_tecu_per_min',
                              index=pd.DatetimeIndex([], dtype='datetime64[ns]', name='time')))

    # units and mapping function (ION-058: an 'M' column is checked against iono_h_m)
    dfa['stec_tecu'] = dfa['stec'] * stec_scale
    dfa['M'] = _resolve_M(dfa, mapping=mapping, iono_h_m=iono_h_m, use_M_column=use_M_column,
                          who="compute_sid")

    # GPST day for the arc division
    if split_on_day_change:
        dfa['day'] = dfa['time'].dt.floor('D')  # GPST day
        grp_cols = ['name','sv','day']
    else:
        grp_cols = ['name','sv']

    # differentiation within groups (link + possibly day)
    dfa = dfa.sort_values(grp_cols + ['time']).reset_index(drop=True)
    dfa['dt_s']  = dfa.groupby(grp_cols, observed=True)['time'].diff().dt.total_seconds()
    dfa['dSTEC'] = dfa.groupby(grp_cols, observed=True)['stec_tecu'].diff()

    rate = dfa.dropna(subset=['dt_s','dSTEC','M'])
    rate = rate[rate['dt_s'] > 0]

    # discard large time gaps
    if max_gap_s is not None:
        rate = rate[rate['dt_s'] <= float(max_gap_s)]

    # IPP region mask (if specified)
    if region is not None:
        inreg = region.contains(rate['lat_ipp'].to_numpy(), rate['lon_ipp'].to_numpy())
        rate = rate[inreg]

    # instantaneous indicator for individual links in TECU/min
    rate['sid_link'] = (rate['dSTEC'] / (rate['M'] * rate['dt_s'])) * 60.0

    # post-epoch aggregation (average/median across links)
    if agg == "median":
        sid = rate.groupby('time')['sid_link'].median().sort_index()
    elif agg == "mean":
        sid = rate.groupby('time')['sid_link'].mean().sort_index()
    elif agg == "max":
        sid = rate.groupby('time')['sid_link'].max().sort_index()
    else:  # "p95" (validated at entry)
        sid = rate.groupby('time')['sid_link'].quantile(0.95, interpolation='linear').sort_index()

    sid.name = 'SIDX_tecu_per_min'
    return _tag(sid)


# --- GIX ---


# --- auxiliary: layered thinning by Δs and azimuth---
def _waterfill(counts: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Equal quotas over the groups with capacities ``counts`` summing to ``target`` (or to
    ``counts.sum()`` when smaller): ``left // n_open`` each, the remainder to randomly chosen
    groups, and the deficit of a group that runs out passed on to the others (water-filling)."""
    quota = np.zeros(counts.size, dtype=int)
    left = int(min(target, counts.sum()))
    open_groups = counts > 0
    while left > 0 and open_groups.any():
        idx_open = np.flatnonzero(open_groups)
        base, rem = divmod(left, idx_open.size)
        alloc = np.zeros(counts.size, dtype=int)
        alloc[idx_open] = base
        if rem:
            alloc[rng.choice(idx_open, size=rem, replace=False)] += 1
        take = np.minimum(alloc, counts - quota)
        quota += take
        left -= int(take.sum())
        open_groups = (counts - quota) > 0
    return quota


def _stratified_subsample(mask_idx: np.ndarray, d_km: np.ndarray, az_deg: np.ndarray,
                          target: int, n_dist_bins: int = 4, n_az_bins: int = 8,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Returns ``target`` indices of mask_idx (all of them when there are no more),
    stratified over the cells (distance quantile bin x azimuth sector).

    Equal-random allocation: the target is shared equally among the non-empty
    distance bins and then, within each bin, equally among its non-empty azimuth sectors;
    at each level the remainder goes to randomly chosen groups and a group with fewer pairs
    than its quota passes the deficit on to the others (water-filling). Cells are drawn in
    random order and nothing is truncated, so there is no shortest-dipole-first bias, and
    with ``target`` >= the number of distance bins every distance bin is represented. The
    distance bins are population quantiles, so the dipole-length distribution is kept in
    expectation; the equal azimuth quotas make the sample isotropic by design (GIX of a
    thinned anisotropic network is therefore not the population GIX).
    """
    if rng is None:
        rng = np.random.default_rng()
    if mask_idx.size <= target:
        return mask_idx

    dsel = d_km[mask_idx]
    azsel = az_deg[mask_idx]

    # cells
    d_bins = np.quantile(dsel, np.linspace(0, 1, n_dist_bins+1))
    # to avoid duplicate boundaries with low variability
    d_bins = np.unique(d_bins)
    if d_bins.size <= 2:  # fallback
        d_bins = np.array([dsel.min(), dsel.max()])
    az_bins = np.linspace(0, 360, n_az_bins+1)

    d_cat = np.clip(np.digitize(dsel, d_bins, right=True)-1, 0, len(d_bins)-2)
    az_cat = np.clip(np.digitize(azsel, az_bins, right=True)-1, 0, len(az_bins)-2)

    n_d, n_a = len(d_bins) - 1, len(az_bins) - 1
    counts = np.bincount(d_cat * n_a + az_cat, minlength=n_d * n_a).reshape(n_d, n_a)
    quota_d = _waterfill(counts.sum(axis=1), target, rng)
    chosen = []
    for di in rng.permutation(n_d):
        if quota_d[di] == 0:
            continue
        quota_a = _waterfill(counts[di], int(quota_d[di]), rng)
        for ai in rng.permutation(n_a):
            if quota_a[ai] > 0:
                cell = mask_idx[(d_cat == di) & (az_cat == ai)]
                chosen.append(rng.choice(cell, size=int(quota_a[ai]), replace=False))
    return np.concatenate(chosen) if chosen else np.array([], dtype=mask_idx.dtype)

_DIP_PRESETS = {
    "30-250": (30.0, 250.0),
    "50-500": (50.0, 500.0),
    "50-1000": (50.0, 1000.0),
}

GIX_TIME_COLUMNS = ['time', 'GIX_mtecu_km', 'GIXx_mean_mtecu_km', 'GIXy_mean_mtecu_km', 'GIXS_mtecu_km',
                    'GIXSx_mtecu_km', 'GIXSy_mtecu_km', 'GIXPx95_plus_mtecu_km', 'GIXPx95_minus_mtecu_km',
                    'GIXPy95_plus_mtecu_km', 'GIXPy95_minus_mtecu_km', 'GIXP95_mtecu_km', 'n_pairs',
                    'GIX_LS_mtecu_km', 'GIX_LSx_mtecu_km', 'GIX_LSy_mtecu_km']
GIX_PAIR_COLUMNS = ['time', 'CP_lat', 'CP_lon', 'grad_abs_mtecu_km', 'grad_x_mtecu_km', 'grad_y_mtecu_km',
                    'dipole_km', 'Mi', 'Mj', 'dVTEC', 'sv_ii', 'sv_jj', 'name_ii', 'name_jj']
#: smallest/largest eigenvalue of sum(u u^T) below which the pair directions are too collinear
#: for the least-squares gradient (GIX_LS_* are then NaN)
GIX_LS_MIN_EIG_RATIO = 1e-2


def _empty_gix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Typed empty (times_df, pairs_df) with the full column lists."""
    def col(c):
        if c == 'time':
            return pd.Series(dtype='datetime64[ns]')
        if c == 'n_pairs':
            return pd.Series(dtype='int64')
        if c in ('sv_ii', 'sv_jj', 'name_ii', 'name_jj'):
            return pd.Series(dtype=object)
        return pd.Series(dtype=float)
    return (_tag(pd.DataFrame({c: col(c) for c in GIX_TIME_COLUMNS})),
            _tag(pd.DataFrame({c: col(c) for c in GIX_PAIR_COLUMNS})))


def _ls_gradient(g: np.ndarray, sin_az: np.ndarray, cos_az: np.ndarray) -> Tuple[float, float]:
    """Least-squares horizontal gradient (east, north) from pair gradients g = G . u, u = (sin az, cos az)."""
    if g.size < 2:
        return float('nan'), float('nan')
    A = np.array([[np.sum(sin_az * sin_az), np.sum(sin_az * cos_az)],
                  [np.sum(sin_az * cos_az), np.sum(cos_az * cos_az)]])
    ev = np.linalg.eigvalsh(A)
    if not np.all(np.isfinite(ev)) or ev[-1] <= 0 or ev[0] / ev[-1] < GIX_LS_MIN_EIG_RATIO:
        return float('nan'), float('nan')
    gx, gy = np.linalg.solve(A, np.array([np.sum(g * sin_az), np.sum(g * cos_az)]))
    return float(gx), float(gy)


# --- GIX, GIXSx/GIXSy i percentyle kierunkowe ± ---
def compute_gix(
    df: pd.DataFrame,
    region: Optional[Region] = None,
    min_elev_deg: float = MIN_ELEV_DEG,
    dipole_preset: str = "30-250",          # "30-250" | "50-500" | "50-1000"
    dmin_km: Optional[float] = None,        #overwrites the preset if specified
    dmax_km: Optional[float] = None,
    iono_h_m: float = IONO_H,
    stec_scale: float = 1.0,
    max_delta_M: float = 0.2,               # filter |Mi−Mj|
    max_pairs_per_epoch: int = MAX_PAIR_PER_EPOCH,
    stratified_sampling: bool = True,       # layered thinning Δs×azimuth
    rng: Optional[np.random.Generator] = None,
    mapping: Optional[Literal['slm', 'mslm']] = None,
    use_M_column: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input (required columns): name,sv,time,lat_ipp,lon_ipp,stec,ev,(optional M).

    Status:
      Active monitoring helper for GIX/GIX-family diagnostics. It can be
      computationally heavy for dense station networks; pair thinning protects
      memory/runtime but changes sampling.

    Definition (Jakowski & Hoque 2019, as restated by Cahuasquí et al. 2025,
    JSWSC 15, 45, eq. 1-2): for every IPP pair (dipole) the gradient
    dVTEC/ds is projected on the pair azimuth, x = g sin(az), y = g cos(az), and
    GIX = |(mean x, mean y)|. This projected mean is kept for comparability with
    the published index. Note its magnitude: for pairs spread evenly in azimuth
    E[u u^T] = I/2, so GIX ≈ |∇VTEC|/2 (a 50 mTECU/km linear field gives GIX
    ≈ 25). The least-squares gradient from the same pairs,
    G = (Σ u u^T)^-1 Σ (dVTEC/ds) u, is exact for any pair geometry and is
    returned as GIX_LS_mtecu_km / GIX_LSx_mtecu_km / GIX_LSy_mtecu_km (NaN when
    the pair directions are nearly collinear, eigenvalue ratio of Σ u u^T below
    ``GIX_LS_MIN_EIG_RATIO``).

    Returns:
      - times_df (1 record/epoch) with fields:
        ['time',
         'GIX_mtecu_km',
         'GIXx_mean_mtecu_km','GIXy_mean_mtecu_km',
         'GIXS_mtecu_km',       # std(|∇TEC|), as in the article – variant on modules
         'GIXSx_mtecu_km','GIXSy_mtecu_km',  # std of signed components
         'GIXPx95_plus_mtecu_km','GIXPx95_minus_mtecu_km',
         'GIXPy95_plus_mtecu_km','GIXPy95_minus_mtecu_km',
         'GIXP95_mtecu_km',     # p95 from |∇TEC|
         'n_pairs',
         'GIX_LS_mtecu_km','GIX_LSx_mtecu_km','GIX_LSy_mtecu_km']   # least-squares gradient
      - pairs_df (each pair) with fields (CP_lat/lon, grad_x/y/abs, dipole_km, Mi, Mj, dVTEC,
        sv_ii/sv_jj, name_ii/name_jj)
      Both frames always have these columns (typed and empty when no pair is found) and
      carry ``attrs["time_scale"] = "GPST"``.

    Time: naive GPST; timezone-aware epochs are converted to UTC and then to GPST
    (+leap seconds) with one UserWarning.

    Pairs: every two IPPs of an epoch form a candidate pair, whatever their station,
    satellite or system (same-station pairs of two satellites and same-satellite pairs of
    two stations included; mixed G/E/C input is allowed); a pair is kept when
    its dipole length is inside the range and ``|Mi - Mj| <= max_delta_M``. Filter ``df``
    by ``sv`` before the call to pair one system only. The region mask (pair centre point)
    is applied before the thinning, so ``max_pairs_per_epoch`` counts pairs inside the
    region. Stratified thinning is the equal-random allocation of
    :func:`_stratified_subsample`.

    Mapping function: see :func:`compute_sid` (``mapping``, ``use_M_column``; an 'M'
    column that is not an accepted mapping function at ``iono_h_m`` is recomputed with
    a UserWarning).

    Dipole range: ``dipole_preset`` ('30-250' | '50-500' | '50-1000'), or both
    ``dmin_km`` and ``dmax_km`` (0 < dmin_km < dmax_km), which replace the preset
    (``dipole_preset`` is then ignored, e.g. a placeholder such as 'custom').
    Raises ValueError (naming the argument) for missing columns (including 'sv'),
    an unknown preset, only one of ``dmin_km``/``dmax_km``, ``min_elev_deg``
    outside [0, 90] deg, non-positive ``iono_h_m``/``stec_scale``, negative
    ``max_delta_M``, ``max_pairs_per_epoch < 1`` or an unknown ``mapping``.
    """
    need = {'name','sv','time','lat_ipp','lon_ipp','stec','ev'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Columns missing: {sorted(miss)}")

    if (dmin_km is None) != (dmax_km is None):
        raise ValueError("dmin_km and dmax_km must be given together (they replace the dipole_preset range); "
                         f"got dmin_km={dmin_km!r}, dmax_km={dmax_km!r}")
    if dmin_km is None:
        _check_choice("dipole_preset", dipole_preset, list(_DIP_PRESETS),
                      extra="Or give dmin_km and dmax_km explicitly.")
        dmin_km, dmax_km = _DIP_PRESETS[dipole_preset]
    else:
        _check_number("dmin_km", dmin_km, gt=0, unit="km")
        _check_number("dmax_km", dmax_km, gt=0, unit="km")
        if dmax_km <= dmin_km:
            raise ValueError(f"dmax_km must be > dmin_km (got dmin_km={dmin_km!r}, dmax_km={dmax_km!r})")
    _check_number("min_elev_deg", min_elev_deg, ge=0, le=90, unit="deg")
    _check_number("iono_h_m", iono_h_m, gt=0, unit="m")
    _check_number("stec_scale", stec_scale, gt=0)
    _check_number("max_delta_M", max_delta_M, ge=0, allow_none=True)
    max_pairs_per_epoch = _check_number("max_pairs_per_epoch", max_pairs_per_epoch, ge=1, integer=True)
    if mapping is not None:
        _check_choice("mapping", mapping, _MAPPINGS)
    _check_region(region)

    obs = df.copy()
    obs['time'] = _to_gpst(obs['time'], who="compute_gix")
    obs = obs[obs['ev'] >= float(min_elev_deg)].copy()
    if obs.empty:
        return _empty_gix()

    obs['stec_tecu'] = obs['stec'] * float(stec_scale)
    obs['M'] = _resolve_M(obs, mapping=mapping, iono_h_m=iono_h_m, use_M_column=use_M_column,
                          who="compute_gix")
    obs['VTEC'] = obs['stec_tecu'] / obs['M']
    obs['lon_ipp'] = ((obs['lon_ipp'] + 180) % 360) - 180

    if rng is None:
        rng = np.random.default_rng()

    times, pair_rows = [], []

    for epoch, sub in obs.groupby('time'):
        sub = sub.reset_index(drop=True)
        n = len(sub)
        if n < 2:
            continue

        ii, jj = np.triu_indices(n, k=1)

        if ii.size == 0:
            continue

        lat, lon = sub['lat_ipp'].to_numpy(), sub['lon_ipp'].to_numpy()
        # dipole lengths (over a shell)
        d_km_all = great_circle_distance_km(lat[ii], lon[ii], lat[jj], lon[jj], radius_m=Re + iono_h_m)

        # pair azimuth (direction j->i)
        az_all = initial_bearing_deg(lat[jj], lon[jj], lat[ii], lon[ii])

        mask = (d_km_all >= dmin_km) & (d_km_all <= dmax_km)
        if not np.any(mask):
            continue

        Mi_all = sub['M'].to_numpy()[ii]
        Mj_all = sub['M'].to_numpy()[jj]
        if max_delta_M is not None:
            mask &= (np.abs(Mi_all - Mj_all) <= float(max_delta_M))
            if not np.any(mask):
                continue

        idx = np.where(mask)[0]
        # pair midpoint (for region mask and visualization)
        cp_lat_all, cp_lon_all = spherical_midpoint(lat[ii[idx]], lon[ii[idx]], lat[jj[idx]], lon[jj[idx]])

        # region mask BEFORE thinning (ION-068): the pair budget is spent inside the region
        if region is not None:
            inreg = region.contains(cp_lat_all, cp_lon_all)
            if not np.any(inreg):
                continue
            idx, cp_lat_all, cp_lon_all = idx[inreg], cp_lat_all[inreg], cp_lon_all[inreg]

        # thinning: regular or layered Δs×azimuth
        if idx.size > max_pairs_per_epoch:
            if stratified_sampling:
                keep = _stratified_subsample(np.arange(idx.size), d_km_all[idx], az_all[idx],
                                             target=max_pairs_per_epoch, rng=rng)
            else:
                keep = rng.choice(idx.size, size=max_pairs_per_epoch, replace=False)
            idx, cp_lat, cp_lon = idx[keep], cp_lat_all[keep], cp_lon_all[keep]
        else:
            cp_lat, cp_lon = cp_lat_all, cp_lon_all

        # final choice
        ii_s, jj_s = ii[idx], jj[idx]
        dsel   = d_km_all[idx]
        Mi, Mj = Mi_all[idx], Mj_all[idx]
        az     = az_all[idx]

        # gradient (TECU/km) – signed
        dVTEC = sub['VTEC'].to_numpy()[ii_s] - sub['VTEC'].to_numpy()[jj_s]
        sv_ii, sv_jj = sub['sv'].to_numpy()[ii_s], sub['sv'].to_numpy()[jj_s]
        name_ii, name_jj = sub['name'].to_numpy()[ii_s], sub['name'].to_numpy()[jj_s]
        grad_signed = dVTEC / dsel
        grad_abs    = np.abs(grad_signed)

        # breakdown into x/y components (WE/NS) – signed projections on the pair azimuth
        sindlt, cosdlt = np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))
        grad_x = grad_signed * sindlt # E-W
        grad_y = grad_signed * cosdlt # N-S

        if grad_abs.size == 0:
            continue

        # --- METRICS  ---
        # mean of the projections (published GIX, ~|grad|/2 for isotropic pairs)
        gx_mean = float(np.mean(grad_x)) * 1000.0  # mTECU/km
        gy_mean = float(np.mean(grad_y)) * 1000.0
        gix     = float(np.hypot(gx_mean, gy_mean))

        # least-squares gradient from the same pairs (exact for any geometry)
        ls_x, ls_y = _ls_gradient(grad_signed, sindlt, cosdlt)
        ls_x, ls_y = ls_x * 1000.0, ls_y * 1000.0

        # dyspersja GIXS:
        # (a) std on modules |∇TEC|
        gixs_abs = float(np.std(grad_abs)) * 1000.0
        # (b) std of signed components (directional dispersion)
        gixs_x   = float(np.std(grad_x)) * 1000.0
        gixs_y   = float(np.std(grad_y)) * 1000.0

        # directional percentiles ± (separately for signs)
        def _p95_signed(arr: np.ndarray) -> Tuple[float, float]:
            pos = arr[arr > 0]
            neg = arr[arr < 0]
            p_pos = float(np.percentile(pos, 95)) if pos.size else 0.0
            p_neg = float(np.percentile(neg, 5))  if neg.size else 0.0  # "95% of the smallest" < 0
            return p_pos*1000.0, p_neg*1000.0

        px95_plus,  px95_minus  = _p95_signed(grad_x)
        py95_plus,  py95_minus  = _p95_signed(grad_y)

        # p95 from modules (global)
        gixp95 = float(np.percentile(grad_abs, 95)) * 1000.0

        times.append({
            'time': epoch,
            'GIX_mtecu_km': gix,
            'GIXx_mean_mtecu_km': gx_mean,
            'GIXy_mean_mtecu_km': gy_mean,
            'GIXS_mtecu_km': gixs_abs,      # std(|∇TEC|)
            'GIXSx_mtecu_km': gixs_x,       # std(∇TEC_x)
            'GIXSy_mtecu_km': gixs_y,       # std(∇TEC_y)
            'GIXPx95_plus_mtecu_km':  px95_plus,   # 95. perc. positive in x
            'GIXPx95_minus_mtecu_km': px95_minus,  # 5. perc. (negative) in x
            'GIXPy95_plus_mtecu_km':  py95_plus,
            'GIXPy95_minus_mtecu_km': py95_minus,
            'GIXP95_mtecu_km': gixp95,      # 95.perc. of |∇TEC|
            'n_pairs': int(grad_abs.size),
            'GIX_LS_mtecu_km': float(np.hypot(ls_x, ls_y)),
            'GIX_LSx_mtecu_km': ls_x,
            'GIX_LSy_mtecu_km': ls_y,
        })

        pair_rows.append(pd.DataFrame({
            'time': epoch,
            'CP_lat': cp_lat, 'CP_lon': cp_lon,
            'grad_abs_mtecu_km': grad_abs * 1000.0,
            'grad_x_mtecu_km':   grad_x  * 1000.0,   # signed
            'grad_y_mtecu_km':   grad_y  * 1000.0,   # signed
            'dipole_km': dsel,
            'Mi': Mi, 'Mj': Mj, 'dVTEC':dVTEC,'sv_ii':sv_ii, 'sv_jj':sv_jj,
            'name_ii':name_ii, 'name_jj':name_jj,
        }))

    if not times:
        return _empty_gix()
    times_df = pd.DataFrame(times, columns=GIX_TIME_COLUMNS).sort_values('time').reset_index(drop=True)
    pairs_df = pd.concat(pair_rows, ignore_index=True)[GIX_PAIR_COLUMNS]
    return _tag(times_df), _tag(pairs_df)


"""
ROTI for GNSS data (STEC/VTEC) — according to Carmo et al. (2021), Method 4 (Cherniak 2018).
DataFrame input: columns ['time','stec','vtec','lat_ipp','lon_ipp','name','sv'] (+ possibly 'ev').
- ROT = dltTEC / dltt  (TECU/min)
- ROTI = std(ROT) in a 5-minute window (population std, ddof=0)
- epochs are naive GPST (outputs carry attrs["time_scale"] = "GPST")

Functions:
- compute_roti_links(...)        : calculate ROT/ROTI per link (name, sv) over time, gap-resistant and north-resistant
- roti_snapshot_grid(...)        : aggregate ROTI to a geographic grid in a time window around an epoch
- compute_roti_global_scale(...) : percentile-based (vmin, vmax) shared by a series of ROTI maps

Plotting lives in ``gnx_py.viz``: ``plot_roti_series`` in :mod:`gnx_py.viz.ionosphere`,
``plot_roti_snapshot`` and ``make_roti_snapshots`` in :mod:`gnx_py.viz.maps` (the last two
are imported back at the bottom of this module).

By default, we operate on STEC (Carmo recommendation — Method 4), but you can switch to VTEC
with ``compute_roti_links(..., tec_source='vtec')``.
"""



from astropy.time import Time


def _ensure_sm_cols(
    dfa: pd.DataFrame,
    transformer,                  # gnx.SolarGeomagneticTransformer()
    shell_height_m: float = 450e3
) -> pd.DataFrame:
    """
    Adds columns 'lat_sm', 'lon_sm' based on 'lat_ipp', 'lon_ipp', 'time' (naive GPST).
    Performs vector transformation for each epoch.

    ``lat_ipp``/``lon_ipp`` are passed to ``transformer.geodetic_to_sm`` as geodetic coordinates
    at height ``shell_height_m``. The TEC session computes them with the spherical thin-shell
    formula started from the geodetic receiver latitude (``ionex.TECInterpolator.get_ipp``), so
    ``lat_ipp`` is close to geodetic already: exact at zenith and within about 0.21 deg of the
    geodetic latitude of the true line-of-sight/shell intersection down to 15 deg elevation. It
    must not be converted as if it were geocentric (that adds 0.18 deg at 50 N). The epochs are
    GPST: the transform time is TAI = GPST + 19 s (exact), not the GPST label read as UTC.
    """
    if transformer is None:
        raise ValueError("For mode='SM', specify a transformer with the geodetic_to_sm(..., obstime=Time) method.")
    if not {'time','lat_ipp','lon_ipp'}.issubset(dfa.columns):
        raise ValueError("The columns 'time', 'lat_ipp', and 'lon_ipp' are missing..")

    dfa = dfa.copy()
    dfa['time'] = _to_gpst(dfa['time'], who="_ensure_sm_cols")
    dfa = dfa.reset_index(drop=True)

    lat_sm = np.empty(len(dfa))
    lon_sm = np.empty(len(dfa))
    lat_rad_all = np.deg2rad(dfa['lat_ipp'].to_numpy(dtype=float))
    h_all = np.full(len(dfa), float(shell_height_m))
    lon_rad_all = np.deg2rad(dfa['lon_ipp'].to_numpy(dtype=float))
    groups = dfa.groupby('time', sort=False).groups
    for t, block_idx in groups.items():
        idx = np.asarray(block_idx, dtype=int)
        sm_lat, sm_lon, _h = transformer.geodetic_to_sm(
            lat_rad=lat_rad_all[idx], lon_rad=lon_rad_all[idx], h_m=h_all[idx], obstime=_gpst_to_tai_time(t)
        )
        lat_sm[idx] = np.rad2deg(sm_lat)
        lon_deg = (np.rad2deg(sm_lon) + 180.0) % 360.0 - 180.0
        lon_sm[idx] = lon_deg

    dfa['lat_sm'] = lat_sm
    dfa['lon_sm'] = lon_sm
    return _tag(dfa)


ROTI_COLUMNS = ['time', 'name', 'sv', 'lat_ipp', 'lon_ipp', 'TEC', 'ROT_mtecu_per_s', 'ROT_tecu_per_min',
                'ROTI_tecu_per_min', 'ROTI_mtecu_per_s', 'rm5']


def _empty_frame(columns, object_cols=(), int_cols=()) -> pd.DataFrame:
    def col(c):
        if c == 'time':
            return pd.Series(dtype='datetime64[ns]')
        if c in object_cols:
            return pd.Series(dtype=object)
        if c in int_cols:
            return pd.Series(dtype='int64')
        return pd.Series(dtype=float)
    return _tag(pd.DataFrame({c: col(c) for c in columns}))


# ---------- ROT/ROTI per link ----------
def compute_roti_links(
    df: pd.DataFrame,
    tec_source: Literal['stec','vtec']='stec',
    tec_scale: float = 1.0,
    window_min: int = ROTI_WINDOW_MIN,
    min_samples: int = ROTI_MIN_SAMPLES,
    max_gap_s: Optional[float] = MAX_GAP_S,
    split_on_day_change: bool = SPLIT_BY_DAY,
    region: Optional[Region] = None,
    min_elev_deg: Optional[float] = MIN_ELEV_DEG,
    detrend_5min: bool = True,
    # --- NOWE ---
    coord_mode: Literal['GEO','SM'] = 'GEO',
    sm_transformer = None,
    sm_shell_height_m: float = 450e3,
    region_frame: Literal['GEO','SM','AUTO'] = 'AUTO',
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['time','name','sv','lat_ipp','lon_ipp','TEC','ROT_mtecu_per_s','ROT_tecu_per_min',
       'ROTI_tecu_per_min','ROTI_mtecu_per_s','rm5'] (+ 'lat_sm','lon_sm' whenever SM coordinates
    are requested by ``coord_mode`` or ``region_frame``). An empty result has the same typed
    columns. The frame carries ``attrs["time_scale"] = "GPST"``.

    ROT is calculated for each link (name, sv) and (optionally) within the same GPST day, only
    between adjacent samples of one contiguous segment: a step is a ROT only when the two samples
    are neighbours in the link as given (no sample removed between them by the elevation mask or
    the region filter), the time step is positive and at most ``max_gap_s``. The 5-min detrend,
    ROT and the ROTI window run per segment, so no ROT spans a gap (Pi et al. 1997).
    ``detrend_5min``: the TEC is reduced by its trailing 5-min mean (window [t - 5 min, t],
    past samples only; column 'rm5') before the differencing; the window does not follow
    ``window_min``. Kept trailing on purpose: a centred window would use later
    samples; the analytic ramp tests do not depend on it.
    ROTI = rolling std(ROT) over time (window 'window_min'), ddof=0, min_periods='min_samples'.
    One link gives the same values alone as inside a network.

    ``sv`` must identify a continuous arc (the TEC session's 'G01_0_4' labels do): with plain
    PRNs two passes of one satellite are joined unless ``max_gap_s`` separates them.
    ``name`` and ``sv`` must not be NaN (ValueError naming the column and the row count).
    ``lon_ipp`` is normalised to [-180, 180) at entry. ``time``: naive GPST; timezone-aware
    epochs are converted to UTC and then to GPST (+leap seconds) with one UserWarning.

    Status:
      Active monitoring helper. SM coordinate mode depends on
      ``SolarGeomagneticTransformer`` and should be validated for the target
      epoch/region.

    ``tec_source`` is 'stec' or 'vtec', ``coord_mode`` 'GEO' or 'SM',
    ``region_frame`` 'GEO', 'SM' or 'AUTO' (exact spelling). Raises ValueError
    (naming the argument) for any other value, non-positive ``tec_scale``/
    ``window_min``/``max_gap_s``/``sm_shell_height_m``, ``min_samples < 1`` or
    ``min_elev_deg`` outside [0, 90] deg.
    """
    need = {'time','stec','vtec','lat_ipp','lon_ipp','name','sv'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Columns missing: {sorted(miss)}")
    _check_choice("tec_source", tec_source, ['stec', 'vtec'])
    _check_number("tec_scale", tec_scale, gt=0)
    _check_number("window_min", window_min, gt=0, unit="min")
    min_samples = _check_number("min_samples", min_samples, ge=1, integer=True)
    _check_number("max_gap_s", max_gap_s, gt=0, unit="s", allow_none=True)
    _check_number("min_elev_deg", min_elev_deg, ge=0, le=90, unit="deg", allow_none=True)
    _check_choice("coord_mode", coord_mode, ['GEO', 'SM'])
    _check_choice("region_frame", region_frame, ['GEO', 'SM', 'AUTO'])
    _check_number("sm_shell_height_m", sm_shell_height_m, gt=0, unit="m")
    _check_region(region)
    bad = {c: int(df[c].isna().sum()) for c in ('name', 'sv') if df[c].isna().any()}
    if bad:
        raise ValueError("compute_roti_links: the link key columns hold NaN ("
                         + ", ".join(f"{c!r}: {n} of {len(df)} rows" for c, n in bad.items())
                         + "); every row needs a station 'name' and a satellite/arc 'sv'")

    sm = coord_mode == 'SM' or region_frame == 'SM'
    out_cols = ROTI_COLUMNS + (['lat_sm', 'lon_sm'] if sm else [])

    dfa = df.copy()
    dfa['time'] = _to_gpst(dfa['time'], who="compute_roti_links")
    dfa['lon_ipp'] = ((dfa['lon_ipp'].astype(float) + 180.0) % 360.0) - 180.0
    dfa = dfa.sort_values(['name','sv','time']).reset_index(drop=True)
    # position of every sample in its link BEFORE any mask: a step is a ROT only between neighbours
    dfa['_pos'] = dfa.groupby(['name', 'sv'], sort=False, observed=True).cumcount()

    # elevation mask
    if min_elev_deg is not None and 'ev' in dfa.columns:
        dfa = dfa[dfa['ev'] >= float(min_elev_deg)].copy()

    # TEC source
    dfa['TEC'] = dfa[tec_source].astype(float) * tec_scale

    # --- SM columns  ---
    if sm and not dfa.empty:
        dfa = _ensure_sm_cols(dfa, sm_transformer, shell_height_m=sm_shell_height_m)
    elif sm and sm_transformer is None:
        raise ValueError("For mode='SM', specify a transformer with the geodetic_to_sm(..., obstime=Time) method.")

    # --- region filter (in selected reference frame); the removed samples break the link ---
    if region is not None and not dfa.empty:
        rf = coord_mode if region_frame == 'AUTO' else region_frame
        latv = dfa['lat_sm'] if rf == 'SM' else dfa['lat_ipp']
        lonv = dfa['lon_sm'] if rf == 'SM' else dfa['lon_ipp']
        inreg = region.contains(latv.to_numpy(), lonv.to_numpy())
        dfa = dfa[inreg].copy()

    if dfa.empty:
        return _empty_frame(out_cols, object_cols=('name', 'sv'))

    # division into GPST days, so as not to differentiate across midnight (daily jumps)
    link = ['name', 'sv'] + (['day'] if split_on_day_change else [])
    if split_on_day_change:
        dfa['day'] = dfa['time'].dt.floor('D')
    dfa = dfa.sort_values(link + ['time'])
    dt = dfa.groupby(link, sort=False, observed=True)['time'].diff().dt.total_seconds()
    dpos = dfa.groupby(link, sort=False, observed=True)['_pos'].diff()
    step_ok = dt.notna() & (dt > 0) & (dpos == 1)
    if max_gap_s is not None:
        step_ok &= dt <= float(max_gap_s)
    # segment = contiguous piece of a link; the first sample of a link always opens one
    dfa['_seg'] = (~step_ok).groupby([dfa[c] for c in link], sort=False, observed=True).cumsum()
    grp = link + ['_seg']

    # per-segment rolling through transform: a 1-D result for any number of links (ION-060)
    if detrend_5min:
        rm5 = dfa.set_index('time').groupby(grp, sort=False, observed=True)['TEC'].transform(
            lambda s: s.rolling('5min', min_periods=1, closed='both').mean()).to_numpy()
        dfa['rm5'] = rm5
        dfa['TEC_d'] = dfa['TEC'].to_numpy() - rm5
    else:
        dfa['TEC_d'] = dfa['TEC']
        dfa['rm5'] = 0.0

    # differentiation within segments
    dtec = dfa.groupby(grp, sort=False, observed=True)['TEC_d'].diff()
    dts = dt.where(step_ok)

    # ROT [TECU/min] and [mTECU/s]
    dfa['ROT_tecu_per_min'] = (dtec / dts) * 60.0
    dfa['ROT_mtecu_per_s'] = (dtec / dts) * 1000.0

    # Rolling STD over time (time window) per segment, ddof=0 (population definition)
    tix = dfa.set_index('time')
    for src, dst in (('ROT_tecu_per_min', 'ROTI_tecu_per_min'), ('ROT_mtecu_per_s', 'ROTI_mtecu_per_s')):
        dfa[dst] = tix.groupby(grp, sort=False, observed=True)[src].transform(
            lambda s: s.rolling(f'{window_min}min', min_periods=min_samples, closed='both').std(ddof=DDOF_POP)
        ).to_numpy()

    out = dfa[out_cols].copy()
    return _tag(out.sort_values(['time','name','sv']).reset_index(drop=True))



# ------ ------ ------ ------ ------ ------ ROTI PLOTTING ------- ------ ------ ------ ------ ------
SNAPSHOT_COLUMNS = ['ix', 'iy', 'lon', 'lat', 'ROTI', 'n']


# --- 1) GRID for given epoch (snapshot) ---
def roti_snapshot_grid(
    roti_df: pd.DataFrame,
    epoch,
    window_min: int = 5,
    mode: Literal['center','trailing'] = 'center',
    res_deg: float = 1.0,
    min_points: int = 5,
    agg: Literal['median','mean','max','Q95'] = 'median',
    # --- NOWE ---
    coord_mode: Literal['GEO','SM'] = 'GEO'
) -> pd.DataFrame:
    """
    Aggregates ROTI into a grid in a time window around (or before) 'epoch'.
    Returns: DataFrame ['ix','iy','lon','lat','ROTI','n'] (1 row per grid cell; the same typed
    columns when empty). ``ix`` counts cells from -180 deg and ``lon`` is the mean of the
    cell's longitudes in [-180, 180), so ``lon`` and ``ix`` agree whatever convention the
    input longitudes use.

    Time: ``epoch`` and ``roti_df['time']`` are naive GPST; timezone-aware values are
    converted to UTC and then to GPST (+leap seconds) with one UserWarning each, so naive and
    aware inputs can be mixed; strings are parsed, numbers raise TypeError.

    ``mode``: 'center' (window centred on ``epoch``) or 'trailing' (window
    ending at ``epoch``). ``agg``: 'median', 'mean', 'max' or 'Q95' (95th
    percentile). ``coord_mode='SM'`` needs the 'lat_sm'/'lon_sm' columns of
    ``compute_roti_links(..., coord_mode='SM')``. ``window_min=0`` takes only
    the samples at ``epoch`` itself. Any other value, a negative
    ``window_min``, a non-positive ``res_deg`` or ``min_points < 1`` raises
    ValueError naming the argument.
    """
    _check_number("window_min", window_min, ge=0, unit="min")
    _check_choice("mode", mode, ['center', 'trailing'])
    _check_number("res_deg", res_deg, gt=0, unit="deg")
    _check_number("min_points", min_points, ge=1, integer=True)
    _check_choice("agg", agg, ['median', 'mean', 'max', 'Q95'])
    _check_choice("coord_mode", coord_mode, ['GEO', 'SM'])

    empty = _empty_frame(SNAPSHOT_COLUMNS, int_cols=('ix', 'iy', 'n'))
    if roti_df.empty:
        return empty

    if coord_mode == 'SM':
        miss = {'lat_sm', 'lon_sm'} - set(roti_df.columns)
        if miss:
            raise ValueError(f"coord_mode='SM' needs the columns {sorted(miss)} in roti_df "
                             "(compute them with compute_roti_links(..., coord_mode='SM', sm_transformer=...))")

    t = _to_gpst(roti_df['time'], who="roti_snapshot_grid")
    t0 = _epoch_to_gpst(epoch, who="roti_snapshot_grid")
    if mode == 'center':
        half = pd.Timedelta(minutes=window_min/2)
        t1, t2 = t0 - half, t0 + half
    else:  # 'trailing' – backward window
        t1, t2 = t0 - pd.Timedelta(minutes=window_min), t0

    sel = ((t >= t1) & (t <= t2)).to_numpy()
    sub = roti_df[sel].copy()
    sub = sub[~sub['ROTI_tecu_per_min'].isna()]
    if sub.empty:
        return empty

    lat_col = 'lat_sm' if coord_mode == 'SM' else 'lat_ipp'
    lon_col = 'lon_sm' if coord_mode == 'SM' else 'lon_ipp'

    sub['_lon'] = ((sub[lon_col].astype(float) + 180) % 360) - 180   # ION-067: one convention
    sub['ix'] = np.floor((sub['_lon'] + 180)/res_deg).astype(int)
    sub['iy'] = np.floor((sub[lat_col] + 90)/res_deg).astype(int)

    base_agg = dict(lon=('_lon','mean'), lat=(lat_col,'mean'),
                    ROTI=('ROTI_tecu_per_min', {'median':'median','mean':'mean','max':'max'}.get(agg, 'median')),
                    n=('ROTI_tecu_per_min','size'))
    if agg == 'Q95':
        base_agg['ROTI'] = ("ROTI_tecu_per_min", lambda x: x.quantile(0.95))

    g = sub.groupby(['ix','iy']).agg(**base_agg).reset_index()
    return _tag(g[g['n'] >= int(min_points)].reset_index(drop=True)[SNAPSHOT_COLUMNS])


# --- 2) Global scale (fixed vmin/vmax for all snapshots) ---
def compute_roti_global_scale(
    roti_df: pd.DataFrame,
    window_min: int = 5,
    q_low_high: Tuple[float,float] = (5.0, 95.0)
) -> Tuple[float, float]:
    """
    Calculates global (vmin, vmax) based on ROTI percentiles across the entire set.
    For frame comparability, it is worth calculating this after roughly filtering out NaN.

    ``q_low_high`` are percentiles in [0, 100] with low <= high (``make_roti_snapshots``
    passes its ``q_scale`` here); fractions such as (0.05, 0.95) are rejected,
    since they would give a near-zero colour scale.
    """
    _check_percentile_pair("q_low_high (q_scale)", q_low_high)
    s = roti_df['ROTI_tecu_per_min'].dropna()
    if s.empty:
        return (0.0, 1.0)
    vmin = float(np.percentile(s, q_low_high[0]))
    vmax = float(np.percentile(s, q_low_high[1]))
    if vmin >= vmax:
        vmax = vmin + 1e-3
    return vmin, vmax


# --- Plotting -------------------------------------------------------------
# The ROTI/GIX plots that used to be defined here now live in ``gnx_py.viz``,
# alongside the rest of the GNX plotting code: ``plot_roti_snapshot``,
# ``make_roti_snapshots`` and ``plot_gix`` in :mod:`gnx_py.viz.maps`, and
# ``plot_gix_family`` in :mod:`gnx_py.viz.ionosphere`. They are imported back
# here so that ``gnx_py.ionosphere`` -- and therefore ``gnx_py`` -- keeps
# exporting them under their original names.
#
# This import sits at the end of the module deliberately: the plots pull this
# module's grid and geodesy helpers, so every name above must exist before
# ``gnx_py.viz`` is reached.
from ..viz.core import DPI, FIGSIZE
from ..viz.ionosphere import plot_gix_family
from ..viz.maps import make_roti_snapshots, plot_gix, plot_roti_snapshot
