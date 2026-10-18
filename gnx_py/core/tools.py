from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from ..models.tides import get_sun_ecef, get_tides, site_displacement, solid_tide_step2
from .conversion import ecef2geodetic
from .utils import calculate_distance
from .time import datetime2toc, datetime2doy
from ..ionosphere.models import (klobuchar, klobuchar_array, klobuchar_bds, ntcm_vtec, BDT_MINUS_GPST_S,
                                 BDS_KLOBUCHAR_H_KM, BDS_KLOBUCHAR_R_KM)
from ..ionosphere.bdgim import bdgim_slant, gpst_to_bdt
from ..ionosphere.utils import stec_mf
from ..models.troposphere import saastamoinen,niell,tropospheric_delay
from ..ionosphere.ionex import GIMReader, TECInterpolator
from . import _notes
from ..io import rinex_obs as _rinex_obs
from ..models.corrections import orbit_normal_frame, rel_path_corr, process_windup_correction_vectorized
from ..models.biases import match_station
from ..models.antex import (  # noqa: F401 - ANTEX_FREQUENCY_HZ kept importable from tools
    ANTEX_FREQUENCY_HZ,
    AntennaModel,
    broadcast_apc_reference,
    missing_satellite_antennas,
    receiver_frequency,
    satellite_frequency,
)
from .gnss import (
    CLIGHT,
    first_signal,
    SYSTEM_NAMES,
    frequency_hz,
    gamma,
    is_bds_geo,
    mode_ionosphere_free_coefficients,
    mode_layout,
    mode_signals,
    signal_spec,
    wavelength_m,
)


_LOG = logging.getLogger(__name__)


def _satellite_body_frame(r_sat, r_sun):
    """Unit axes (i, j, k) of the IGS satellite body frame, one row per satellite position [m].

    k points from the satellite to the Earth centre, j = k x e_sun normalised (e_sun: unit vector
    satellite -> Sun), i = j x k (RTKLIB ``satantoff``; IGS ANTEX satellite convention). Where the Sun is
    on the k axis (j undefined) j and i are 0, so only the z offset is applied there.
    """
    r_sat = np.atleast_2d(np.asarray(r_sat, dtype=float))
    r_sun = np.atleast_2d(np.asarray(r_sun, dtype=float))
    k = -r_sat / np.linalg.norm(r_sat, axis=1, keepdims=True)
    e = r_sun - r_sat
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    j = np.cross(k, e)
    nj = np.linalg.norm(j, axis=1, keepdims=True)
    ok = nj > 1e-12
    j = np.where(ok, j / np.where(ok, nj, 1.0), 0.0)
    i = np.cross(j, k)
    return i, j, k


def _satellite_azimuth_deg(u_sr, x_b, y_b) -> np.ndarray:
    """Azimuth [deg, 0-360) of the line of sight in the satellite body frame, the angle of the satellite PCV grid.

    ``u_sr``: unit vector satellite -> receiver; ``x_b``, ``y_b``: body axes. ANTEX 1.4 section 3: the azimuth
    counts clockwise from the y-axis toward the x-axis when looking in the direction of the negative z-axis
    (toward deep space), so +y is 0 deg and +x is 90 deg.
    """
    a = np.degrees(np.arctan2(np.einsum('ij,ij->i', u_sr, x_b), np.einsum('ij,ij->i', u_sr, y_b)))
    return np.mod(a, 360.0)


def _attitude_module():
    """``gnx_py.attitude`` (published attitude laws); a function so that tests can replace the
    attitude model."""
    from ..models import attitude

    return attitude


def _warn_once(config, key, message: str, category=UserWarning, stacklevel: int = 3) -> None:
    """``warnings.warn`` once per run of the configuration and ``key`` (the session validation and the
    preprocessing of every system may reach the same condition; ``gnx_py._notes``, cleared at the start of each
    session run)."""
    _notes.warn_once(config, ("attitude",) + (key if isinstance(key, tuple) else (key,)), message, category,
              stacklevel=stacklevel + 1)


def _product_ac(config) -> Optional[str]:
    """Analysis centre of the precise orbits and clocks (first three characters of every ``sp3_path`` name), or
    None for broadcast orbits, without SP3 files or for SP3 files of several centres."""
    if getattr(config, "orbit_type", "precise") == "broadcast":
        return None
    from ..io.orbex import analysis_centre
    from .validation import iter_paths

    centres = {analysis_centre(p) for p in iter_paths(getattr(config, "sp3_path", None))}
    return next(iter(centres)) if len(centres) == 1 else None


def _orbit_product(config) -> Optional[str]:
    """The orbit product ``gnx_py.orbex.check_product`` compares an ORBEX file with: the ``sp3_path`` name when every
    SP3 file is of one product line (e.g. 'GRG0MGXFIN'), the analysis centre when they are of one centre but several
    lines, else None (broadcast orbits, no SP3 file, several centres: every ORBEX file goes)."""
    ac = _product_ac(config)
    if ac is None:
        return None
    from ..io.orbex import product_id
    from .validation import iter_paths

    paths = iter_paths(getattr(config, "sp3_path", None))
    return os.fspath(paths[0]) if len({product_id(p) for p in paths}) == 1 else ac


_ORBEX_CACHE: dict = {}
_ORBEX_CACHE_MAX = 2


def _read_orbex_cached(paths):
    """``gnx_py.orbex.read_orbex(paths)``, cached by (resolved path, mtime, size) of every file."""
    from ..io.orbex import read_orbex

    key = []
    for p in paths:
        st = os.stat(p)
        key.append((os.path.realpath(os.fspath(p)), st.st_mtime_ns, st.st_size))
    key = tuple(key)
    hit = _ORBEX_CACHE.get(key)
    if hit is None:
        hit = read_orbex(list(paths))
        if len(_ORBEX_CACHE) >= _ORBEX_CACHE_MAX:
            _ORBEX_CACHE.pop(next(iter(_ORBEX_CACHE)))
        _ORBEX_CACHE[key] = hit
    return hit


def _warn_attitude_path_unused(config, stacklevel: int = 3) -> None:
    """One UserWarning per configuration when ``attitude_path`` is set with ``satellite_attitude='nominal'`` (the
    nominal attitude reads no ORBEX file)."""
    if getattr(config, "attitude_path", None) and getattr(config, "satellite_attitude", None) == "nominal":
        _warn_once(config, ("attitude_path_unused",),
                   "attitude_path is not used with satellite_attitude='nominal' (yaw steering, BeiDou GEO "
                   "orbit-normal); satellite_attitude='model' uses the ORBEX attitude", UserWarning,
                   stacklevel=stacklevel + 1)


def orbex_for_config(config, stacklevel: int = 3):
    """The ORBEX attitude of ``config.attitude_path`` the session uses, or None.

    The product rule is ``gnx_py.orbex.check_product``: an ORBEX of another analysis centre than the precise orbits
    and clocks (``sp3_path``) is not used, with one UserWarning naming both centres (the clocks absorb the wind-up of
    their own centre's attitude, and centres fly some noon/midnight turns the other way round: one wind-up cycle,
    about 10.7 cm of the ionosphere-free phase); ``attitude_allow_mixed_products=True`` (PPPConfig only; SPP, TEC and SIS
    configurations have no such field) uses it anyway.
    The same centre's file of another product line is used, with one UserWarning. With broadcast orbits the file
    is used as given (the broadcast clocks carry no analysis-centre attitude). The decision and its warning are
    made once per run of the configuration (the session validation and the preprocessing of every system ask
    again; ``gnx_py._notes.run_memo``, cleared at the start of each session run).
    """
    from ..io.orbex import check_product
    from .validation import iter_paths

    paths = iter_paths(getattr(config, "attitude_path", None))
    if not paths:
        return None
    obx = _read_orbex_cached(paths)
    product = _orbit_product(config)
    allow = bool(getattr(config, "attitude_allow_mixed_products", False))
    key = (tuple(os.fspath(p) for p in paths), product, allow)
    done = _notes.run_memo(config, "attitude_orbex_use")
    if key not in done:
        done[key] = check_product(obx, product, allow_mixed=allow, stacklevel=stacklevel + 1)
    return obx if done[key] else None


#: GPS blocks without an L5 signal (ANTEX satellite type, columns 1-20)
_GPS_BLOCKS_WITHOUT_L5 = ("BLOCK I", "BLOCK II", "BLOCK IIA", "BLOCK IIR-A", "BLOCK IIR-B", "BLOCK IIR-M")


def _transmits_l5(entry) -> bool:
    """False for an ANTEX GPS entry of a block without L5 (I, II, IIA, IIR); True otherwise (IIF, III,
    legacy tables without the block name)."""
    if entry is None:
        return False
    return entry.type.strip().upper() not in _GPS_BLOCKS_WITHOUT_L5

# —————————————————————————————————————————————
# GIM evaluation (SPP, PPP, TEC and compare_models share it)
# —————————————————————————————————————————————

#: accepted values of ``Config.gim_time_interpolation``
GIM_TIME_INTERPOLATIONS = ("rotated", "earth_fixed")


def nav_header_bds_coefficients(nav_path):
    """BeiDou Klobuchar coefficients (BDSA, BDSB) of the navigation file ``nav_path``.

    One set, as ``io`` gives it to the broadcast data (``OrbitData.bdsa``/``bdsb``): the ionosphere table of
    ``rinex_nav.read_iono_table`` (header records and RINEX 4 '> ION' records, screened per model) reduced by
    ``rinex_nav.legacy_iono`` - GEO first, then IGSO, then MEO (formerly the first header record
    from georinex, or the last one with the satellite number appended from the basic header reader).
    (None, None) without a path, without a plausible BeiDou Klobuchar set or when the file cannot be read.
    """
    if not nav_path:
        return None, None
    from ..io.rinex_nav import legacy_iono, read_iono_table

    try:
        legacy = legacy_iono(read_iono_table(nav_path, systems="C"))
    except (OSError, ValueError):
        return None, None
    a, b = legacy.get('bdsa'), legacy.get('bdsb')
    if a is None or b is None:
        return None, None
    return list(a)[:4], list(b)[:4]


def _broadcast_iono_rows(iono, system: str, model: str, epochs):
    """Coefficients of ``system``/``model`` in force at each of ``epochs`` (GPST datetime64), shape (n, k), from the
    broadcast ionosphere table ``iono`` (``rinex_nav.broadcast_iono_at``: RINEX 4 '> ION' records, the latest at or
    before the epoch; RINEX 3 / 2 the one header set), or None without a table or without a plausible set. A row
    without a set in force takes the earliest one (the table's rule), so every row is finite when one is."""
    if iono is None or not isinstance(iono, pd.DataFrame) or len(iono) == 0 or len(epochs) == 0:
        return None
    from ..io.rinex_nav import IONO_MODELS, broadcast_iono_at

    coeffs = np.asarray(broadcast_iono_at(iono, system, model, epochs), dtype=float)
    if coeffs.ndim != 2 or coeffs.shape[0] != len(epochs) or coeffs.shape[1] < IONO_MODELS[model] \
            or not np.isfinite(coeffs).all():
        return None
    return coeffs


def _bds_klobuchar_rows(iono, epochs) -> tuple:
    """BDS Klobuchar coefficients in force at each of ``epochs`` (GPST datetime64): ``(alpha (n, 4), beta (n, 4))``
    (:func:`_broadcast_iono_rows`), or ``(None, None)`` without a table or without a plausible BeiDou set."""
    coeffs = _broadcast_iono_rows(iono, 'C', 'KLOB', epochs)
    if coeffs is None:
        return None, None
    return coeffs[:, :4], coeffs[:, 4:8]


#: rejected BDGIM records count for a run when their transmission time is within [first epoch - this, last epoch]
BDGIM_REJECTED_WINDOW = np.timedelta64(6, 'h')


class RunNotes:
    """Run-wide notes of one session run, shared by the :class:`DDPreprocessing` instances of the run. A message about the whole run (the BDGIM age and lead warnings, and the rejected BDGIM records
    inside the run window) is given once, by the first processed system that meets it, whichever systems the run
    processes; ``iono_report`` collects the broadcast ionosphere report of each processed system.

    ``nav_systems``: the systems whose rejected ionosphere records the navigation reader already warned about (the
    ``systems`` given to ``rinex_nav.read_navigation``; the reader adds GPS). None: unknown, then the configured
    ``sys`` of the instance decides.
    """

    def __init__(self, nav_systems=None):
        self.nav_systems = None if nav_systems is None else {c for c in "".join(str(s) for s in nav_systems)
                                                             if c.isalpha()}
        #: broadcast ionosphere report per processed system (``DDPreprocessing.iono_report``)
        self.iono_report: dict = {}
        self._given: set = set()

    def first(self, key) -> bool:
        """True the first time ``key`` is met in this run."""
        if key in self._given:
            return False
        self._given.add(key)
        return True


def _bdgim_rows(iono, epochs) -> tuple:
    """BDGIM coefficients in force at each of ``epochs`` (GPST datetime64): ``(alpha (n, 9) [TECU], info)``.

    The coefficients come from ``rinex_nav.broadcast_iono_at(iono, 'C', 'BDGIM', epochs)`` (RINEX 4 '> ION Cxx CNVX'
    records: the latest plausible set with ``t_tx`` at or before the epoch, from any BeiDou satellite; before the
    first set the earliest one). ``alpha`` is None without a table or without a plausible BDGIM set (with an empty
    ``epochs`` and a plausible set: shape (0, 9)). ``info``:

    - ``n_records``, ``n_rejected``, ``reasons`` (sorted, distinct): every BDGIM record of the table;
    - ``rejected_in_window`` [(sv, t_tx, reason)] and ``n_rejected_in_window``: rejected records with ``t_tx``
      within [first epoch - :data:`BDGIM_REJECTED_WINDOW`, last epoch];
    - ``sets`` [(sv, t_tx)]: the sets used, in time order; ``first_set`` / ``last_set``: the ``t_tx`` span of the
      plausible sets of the table;
    - ``age_h`` (n,): epoch - ``t_tx`` of the set in force [h] (NaN for a row before the first set); ``t_tx`` is
      that of the latest record with the same coefficients from any satellite at or before the epoch, so the age
      is the time since the set was last transmitted (a re-broadcast resets it), not since it was first issued;
      ``max_age_h`` (NaN without such rows); ``n_before_first``: rows that take the earliest set; ``lead_h`` (n,):
      ``t_tx`` of that earliest set - epoch [h] for those rows (NaN for the others), ``max_lead_h`` (NaN without
      such rows).
    """
    t = np.asarray(epochs, dtype='datetime64[ns]').reshape(-1)
    info = {'n_records': 0, 'n_rejected': 0, 'reasons': [], 'rejected_in_window': [], 'n_rejected_in_window': 0,
            'sets': [], 'first_set': None, 'last_set': None, 'age_h': np.full(t.size, np.nan),
            'max_age_h': np.nan, 'n_before_first': 0, 'lead_h': np.full(t.size, np.nan), 'max_lead_h': np.nan}
    if iono is None or not isinstance(iono, pd.DataFrame) or len(iono) == 0 or 'model' not in iono.columns:
        return None, info
    rec = iono[(iono['system'] == 'C') & (iono['model'] == 'BDGIM')]
    ok = rec['plausible'].astype(bool).to_numpy()
    bad = rec[~ok]
    info['n_records'], info['n_rejected'] = int(len(rec)), int(len(bad))
    info['reasons'] = sorted({str(r) for r in bad['reason']})
    finite_t = t[~np.isnat(t)]
    if len(bad) and finite_t.size:
        lo, hi = finite_t.min() - BDGIM_REJECTED_WINDOW, finite_t.max()
        bt = bad['t_tx'].to_numpy(dtype='datetime64[ns]')
        inside = ~np.isnat(bt) & (bt >= lo) & (bt <= hi)
        info['rejected_in_window'] = [(row.sv, pd.Timestamp(row.t_tx), str(row.reason))
                                      for row in bad[inside].itertuples(index=False)]
        info['n_rejected_in_window'] = len(info['rejected_in_window'])
    good = rec[ok]
    if good.empty:
        return None, info
    from ..io.rinex_nav import broadcast_iono_at

    alpha = np.asarray(broadcast_iono_at(iono, 'C', 'BDGIM', t), dtype=float)
    if alpha.shape != (t.size, 9) or not np.isfinite(alpha).all():
        return None, info
    dated = good[good['t_tx'].notna()].assign(__pos=np.arange(int(good['t_tx'].notna().sum())))
    dated = dated.sort_values(['t_tx', '__pos'], kind='stable')
    if not dated.empty:
        info['first_set'], info['last_set'] = pd.Timestamp(dated['t_tx'].iloc[0]), pd.Timestamp(dated['t_tx'].iloc[-1])
    # the set of each row: among the records whose coefficients are the row's, the latest at or before the epoch
    # (the earliest before the first one), so the report follows whatever rule broadcast_iono_at applies
    times = dated['t_tx'].to_numpy(dtype='datetime64[ns]')
    coeffs = np.array([np.asarray(c, dtype=float) for c in dated['coeffs']]).reshape(len(dated), -1)
    svs = dated['sv'].to_list()
    used = {}
    if t.size and len(dated):
        uniq, inverse = np.unique(alpha, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        for u, a in enumerate(uniq):
            match = np.flatnonzero(np.all(coeffs == a, axis=1))
            if match.size == 0:
                continue                                    # a header set (no transmission time)
            rows = np.flatnonzero(inverse == u)
            k = np.searchsorted(times[match], t[rows], side='right') - 1
            before = k < 0
            info['n_before_first'] += int(np.count_nonzero(before & ~np.isnat(t[rows])))
            pick = match[np.where(before, 0, k)]
            age = (t[rows] - times[pick]) / np.timedelta64(1, 'h')
            info['age_h'][rows] = np.where(before, np.nan, age)
            info['lead_h'][rows] = np.where(before, -age, np.nan)
            for j in np.unique(pick):
                used[int(j)] = (svs[j], pd.Timestamp(times[j]))
    info['sets'] = [used[j] for j in sorted(used)]
    ages = info['age_h'][np.isfinite(info['age_h'])]
    info['max_age_h'] = float(ages.max()) if ages.size else np.nan
    leads = info['lead_h'][np.isfinite(info['lead_h'])]
    info['max_lead_h'] = float(leads.max()) if leads.size else np.nan
    return alpha, info


def bdgim_missing_message(nav_path, info: dict, what: str, alternatives: str) -> str:
    """Text of the ValueError for a run that asks for BDGIM when ``nav_path`` has no usable BDGIM set: the file and its RINEX version, no records or how many were rejected and why, and the hint."""
    from ..io.rinex_nav import nav_info

    source = str(nav_path) if nav_path else 'the navigation data'
    if nav_path:
        try:
            source += f" (RINEX {nav_info(nav_path).version})"
        except (OSError, ValueError):                   # an unreadable file: its own reader error comes first
            pass
    if info.get('n_records', 0) == 0:
        why = "has no BDGIM records ('> ION Cxx CNVX', RINEX 4)"
    else:
        why = (f"has {info['n_records']} BDGIM records, all rejected: {'; '.join(info.get('reasons') or [])}")
    return (f"{what}: {source} {why}. Use a RINEX 4 navigation file such as BRD400DLR (it also carries the G/E/C "
            f"ephemerides) as nav_path, or {alternatives}")


#: accepted values of ``Config.bds_klobuchar``: GPS coefficients scaled to the BeiDou signal (default) or the
#: BDS-SIS-ICD-B1I model with BDSA/BDSB (opt-in)
BDS_KLOBUCHAR_MODELS = ("gps", "icd")


def bds_klobuchar_model(config) -> str:
    """``config.bds_klobuchar`` ('gps' when the object has no such setting); ValueError for another value.

    'gps' (default): BeiDou takes the GPS Klobuchar coefficients scaled to its signal, as before v1.0.0. 'icd' (opt-in): ``ionosphere.models.klobuchar_bds`` with the BDSA/BDSB header coefficients; the
    regional BDS-2 coefficients were worse than GPS-scaled on 5 of 9 station-days (see
    ``Config.bds_klobuchar``).
    """
    value = getattr(config, 'bds_klobuchar', 'gps')
    if value not in BDS_KLOBUCHAR_MODELS:
        raise ValueError(f"bds_klobuchar must be one of {list(BDS_KLOBUCHAR_MODELS)} (got {value!r})")
    return value


def gpst_leap_seconds(times) -> np.ndarray:
    """GPST - UTC [s] at GPST epochs (``time._LEAP_TABLE``; 18 s since 2017)."""
    from .time import _LEAP_TABLE

    t = np.asarray(times, dtype="datetime64[ns]")
    since = np.array([np.datetime64(d.replace(tzinfo=None), "ns") + np.timedelta64(v, "s")
                      for d, v in _LEAP_TABLE], dtype="datetime64[ns]")
    vals = np.array([v for _, v in _LEAP_TABLE], dtype=np.int64)
    k = np.searchsorted(since, t, side="right") - 1
    return np.where(k >= 0, vals[np.clip(k, 0, None)], 0)


def gim_epochs_utc(times_gpst, first, last):
    """GPST epochs as the UTC epochs of IONEX maps, clamped at the coverage edges.

    IONEX map epochs are UTC (IONEX 1.0); observation epochs are GPST. Each epoch
    becomes GPST - leap(t). An epoch within leap + 1 s (19 s) before the first map
    is clamped to it, so 00:00:00 GPST of a daily file (23:59:42 UTC of the day
    before) keeps the first map instead of turning NaN. The shift only moves epochs
    earlier, so nothing is clamped after the last map: an epoch past it stays
    outside (no extrapolation), as before the UTC conversion.

    Returns
    -------
    (numpy.ndarray datetime64[ns], numpy.ndarray bool)
        UTC epochs to interpolate at and the mask of epochs outside the maps.
    """
    t = np.asarray(times_gpst, dtype="datetime64[ns]")
    leap = gpst_leap_seconds(t)
    t_utc = t - leap.astype("timedelta64[s]").astype("timedelta64[ns]")
    tol = (leap + 1).astype("timedelta64[s]").astype("timedelta64[ns]")
    first = np.datetime64(pd.Timestamp(first).to_datetime64(), "ns")
    last = np.datetime64(pd.Timestamp(last).to_datetime64(), "ns")
    t_utc = np.where((t_utc < first) & (t_utc >= first - tol), first, t_utc)
    outside = (t_utc < first) | (t_utc > last)
    return t_utc, outside


def gim_slant_tec(data, times_gpst, ev, az, xyz, *, mapping: str = "auto",
                  time_interpolation: str = "rotated", rms: bool = False, notes=None) -> dict:
    """GIM VTEC, mapping function and IPPs for receiver-satellite lines of sight.

    Parameters
    ----------
    data : GIMData
        Maps of ``GIMReader.read``.
    times_gpst : array_like of datetime64
        GPST epochs (converted to UTC, see :func:`gim_epochs_utc`).
    ev, az : array_like
        Elevation and azimuth [deg].
    xyz : array_like
        Receiver ECEF [m].
    mapping : {"auto", "slm", "mslm"}
        GIM mapping function (``TECInterpolator``; ``Config.gim_mapping_function``).
    time_interpolation : {"rotated", "earth_fixed"}
        IONEX eq. 3 (rotated, default) or eq. 2 (``Config.gim_time_interpolation``).
    rms : bool
        Also interpolate the RMS maps (key 'rms'; None when the file has none).
    notes : object, optional
        Owner of the warn-once keys (``gnx_py._notes``; the session's configuration, so the "auto" mapping
        warning comes once per run); None: once per ``data`` object.

    Returns
    -------
    dict
        'vtec' [TECU], 'm_f' (STEC/VTEC), 'lat_ipp'/'lon_ipp' [deg; spherical thin-shell formula started
        from the geodetic receiver latitude, i.e. about geodetic, within 0.21 deg, exact at zenith],
        'outside' (epochs outside the maps), 't_utc', 'mapping' ("slm"/"mslm"), 'rms'.
    """
    if time_interpolation not in GIM_TIME_INTERPOLATIONS:
        raise ValueError(f"gim_time_interpolation must be one of {list(GIM_TIME_INTERPOLATIONS)} "
                         f"(got {time_interpolation!r})")
    # thin shell from the IONEX header (HGT1, BASE RADIUS), not a fixed 450/6371 km
    interp = TECInterpolator(data, mapping=mapping, notes=notes)
    first, last = data.time_span()
    t_utc, outside = gim_epochs_utc(times_gpst, first, last)
    lat_deg, lon_deg = ecef2geodetic(xyz)[:2]
    ev = np.asarray(ev, dtype=float)
    lat_ipp, lon_ipp = interp.get_ipp(ev, np.asarray(az, dtype=float), lat_deg, lon_deg)
    rotate = time_interpolation == "rotated"
    # linear in time between consecutive maps, bilinear in space; IPPs beyond the
    # outermost grid row (+-87.5 deg) take that row (see ionex.interpolate_gim)
    vtec = interp.vtec(t_utc, lat_ipp, lon_ipp, rotate=rotate)
    out = dict(vtec=vtec, m_f=interp.mapping_function(ev), lat_ipp=lat_ipp, lon_ipp=lon_ipp,
               outside=outside, t_utc=t_utc, mapping=interp.mapping_used, rms=None)
    if rms and data.rms is not None:
        out["rms"] = interp.vtec(t_utc, lat_ipp, lon_ipp, field="RMS", rotate=rotate)
    return out


def check_gim_coverage(data, gim: dict, times_gpst, *, source, what: str) -> None:
    """Coverage rule of the GIM paths (``read_inx``, ``compare_models`` 'gim').

    VTEC is only interpolated between maps (no extrapolation in time). A GIM of another day
    gives NaN everywhere: raise the readable error of ``GIMData.check_coverage``. A partial
    overlap (session across midnight, maps ending before 24:00) only loses the uncovered epochs:
    RuntimeWarning. ``gim`` is the result of :func:`gim_slant_tec` for ``times_gpst``.
    """
    times = np.asarray(times_gpst, dtype="datetime64[ns]")
    outside = np.asarray(gim['outside'], dtype=bool)
    if times.size and outside.all():
        data.check_coverage(gim['t_utc'].min(), gim['t_utc'].max(), what="observations (UTC)")
    if outside.any():
        first, last = data.time_span()
        warnings.warn(
            f"GIM {source} covers {first} to {last} (UTC) only: {int(outside.sum())} of {outside.size} "
            f"{what} ({pd.Timestamp(times[outside].min())} to "
            f"{pd.Timestamp(times[outside].max())} GPST) lie outside the maps; their 'ion' is NaN",
            RuntimeWarning, stacklevel=3)

# —————————————————————————————————————————————
# SYS / PHASE SHIFT (contract C4; bug-fix stage 13, NQ-PPP-38)
# —————————————————————————————————————————————

_FRAME_OF_SYSTEM = {"G": "gps", "E": "gal", "C": "bds"}


def observation_phase_shift_attrs(obs_data, system: str) -> Optional[dict]:
    """SYS / PHASE SHIFT information of the observation reader for ``system`` as ``{'phase_shift':
    {(sys, code): {'value', 'sats', 'by_sat'}}, 'phase_shift_events': [(epoch, sys, code, sats, value), ...]}``,
    the contract form as ``rinex_obs.phase_shift_of`` / ``rinex_obs.phase_shift_events_of`` give it (new mutable
    objects), read once per run and system from ``attrs`` of the system's frame of ``obs_data``
    (``ObsData.gps``/``gal``/``bds``), else of the first frame carrying them (every frame of a file carries the
    records of all systems). The sessions pass the result to :class:`DDPreprocessing` explicitly; the per-epoch
    states never carry the attrs. None when no frame carries them (a frame not from the reader: no phase-shift
    handling, as before v1.0.0)."""
    own = _FRAME_OF_SYSTEM.get(system)
    for name in [own] + [n for n in ("gps", "gal", "bds") if n != own]:
        frame = getattr(obs_data, name, None) if name else None
        attrs = getattr(frame, "attrs", None)
        if attrs and ("phase_shift" in attrs or "phase_shift_events" in attrs):
            return {"phase_shift": _rinex_obs.phase_shift_of(attrs),
                    "phase_shift_events": _rinex_obs.phase_shift_events_of(attrs)}
    return None


def phase_alignment_blocks(frame: pd.DataFrame, system: str, phase_columns, phase_shift=None,
                           events=None) -> tuple[np.ndarray, list]:
    """Changes of the phase alignment inside the observation data, per satellite and processed phase.

    RINEX 3.01+ (RINEX 3.03 section 9.1, Table 25): the phases of the file are already aligned to the reference
    signal, phi_RINEX = phi_original + dphi, and SYS / PHASE SHIFT reports the correction dphi [cycles] applied by
    the receiver or converter, for the listed satellites (all satellites of the system without a list). A record
    inside the data (event flag 4 header block) reports a correction in force from its epoch on. A changed dphi
    documents how the original phase was aligned, not a step of the aligned phase: the file's phases stay aligned
    to the reference signal on both sides, so the arc is kept (measured: at the 28 measurable changes of
    WUH200CHN 2024-141 the geometry-free phase steps by 0.2 cm median, not by the 6.2-6.4 cm a quarter cycle
    would give; RTKLIB reads no SYS / PHASE SHIFT record, rinex.c decode_obsh). A real step is a cycle slip and
    is left to the slip detectors. The changes are reported (:meth:`DDPreprocessing.apply_phase_shift`).

    ``frame``: rows indexed by ``sv`` and ``time`` (GPST); ``phase_columns``: the processed phase observables of
    the frame (e.g. ``['L1C', 'L2W']``); ``phase_shift`` / ``events``: in the form of the observation reader or the
    reader's stored form (JSON-safe ``"G L1C"`` keys, ISO epochs; read with ``rinex_obs.phase_shift_of`` /
    ``rinex_obs.phase_shift_events_of``, also after a parquet / JSON round trip). The correction in force at the
    start follows the reader's per-satellite rule (``rinex_obs.phase_shift_map``): ``by_sat[sv]`` when the header
    lists the satellite, else the record's value when it has no satellite list, else 0 (an observable with a
    record); unknown for an observable without a header record. Records of one epoch (one event block) are read
    as the header is: the first statement of a satellite is the writer's (a converter listing a satellite under
    two values in one block restates the header), the satellites a block does not name keep their correction. A
    record repeating the value in force changes nothing; one whose value differs, or whose predecessor is
    unknown, does when the satellite has data of that phase before and from the epoch on.

    Returns ``(blocks, changes)``: per row of ``frame`` the number of changes of its satellite up to the row (int
    array), and the changes as ``(epoch, obs_type, sv, value_before | None, value_after)``.
    """
    n = len(frame)
    blocks = np.zeros(n, dtype=np.int64)
    events = _rinex_obs.phase_shift_events_of(events)
    if not events or n == 0:
        return blocks, []
    phase_columns = [c for c in phase_columns if c in frame.columns]
    if not phase_columns:
        return blocks, []
    svs = frame.index.get_level_values("sv").astype(str).to_numpy()
    times = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("time"))).as_unit("ns").asi8
    all_sats = sorted(set(svs))

    def sats_of(sats):
        return all_sats if sats is None else sats

    in_force: dict = {}
    default: dict = {}          # obs -> correction of the satellites the header lists for no value of its own
    for (sys_, obs), entry in _rinex_obs.phase_shift_of(phase_shift).items():
        if sys_ != system or obs not in phase_columns:
            continue
        default[obs] = entry["value"] if entry["sats"] is None else 0.0
        if entry["by_sat"]:
            for sv, value in entry["by_sat"].items():
                in_force[(obs, sv)] = value
        elif entry["sats"] is not None:
            for sv in entry["sats"]:
                in_force[(obs, sv)] = entry["value"]

    def before(obs, sv):
        if (obs, sv) in in_force:
            return in_force[(obs, sv)]
        return default.get(obs)

    by_epoch: dict = {}
    for epoch, sys_, obs, sats, value in events:
        if sys_ != system or obs not in phase_columns or epoch is None:
            continue
        by_epoch.setdefault(epoch, []).append((obs, sats, value))
    changes = []
    for epoch in sorted(by_epoch):
        t_ns = epoch.value
        after: dict = {}
        for obs, sats, value in by_epoch[epoch]:
            for sv in sats_of(sats):
                after.setdefault((obs, sv), value)          # first statement of the satellite in the block
        for (obs, sv), value in after.items():
            old = before(obs, sv)
            in_force[(obs, sv)] = value
            if old is not None and abs(value - old) < 1e-9:
                continue
            rows = svs == sv
            col = np.isfinite(frame[obs].to_numpy(dtype=float)) & rows
            if not (col & (times < t_ns)).any() or not (col & (times >= t_ns)).any():
                continue
            changes.append((epoch, obs, sv, old, value))
    for epoch, sv in dict.fromkeys((c[0], c[2]) for c in changes):
        blocks[(svs == sv) & (times >= epoch.value)] += 1
    return blocks, changes


# —————————————————————————————————————————————
# Cycle-slip detection
# —————————————————————————————————————————————

class _PrefixSums:
    """Weighted prefix sums of a series: least-squares steps over any window in O(1).

    Epochs of weight 0 (NaN, outliers) are left out of every window. The series is taken
    relative to its first finite value, so a large constant offset (e.g. L1 - L2 of a receiver
    that does not align phase to code) does not cost precision in the sums.
    """

    def __init__(self, y: np.ndarray, weight: Optional[np.ndarray] = None):
        y = np.asarray(y, dtype=float)
        w = np.isfinite(y).astype(float)
        if weight is not None:
            w = w * weight
        used = np.flatnonzero(w > 0)
        y = np.where(w > 0, y - (y[used[0]] if len(used) else 0.0), 0.0)
        x = np.arange(len(y), dtype=float)

        def cum(a):
            return np.concatenate(([0.0], np.cumsum(a)))

        self.n = cum(w)
        self.y, self.x, self.xx, self.xy = cum(w * y), cum(w * x), cum(w * x * x), cum(w * x * y)

    def step(self, lo, j, hi, linear: bool, skip: int = 0):
        """Step of the series between epochs ``j-1`` and ``j`` fitted to the epochs ``[lo, j)`` and ``[j+skip, hi)``.

        ``linear``: one slope common to both sides (ionospheric drift) and a step, otherwise the
        difference of the two means. ``skip``: epochs left out after ``j``. Returns ``(step, v)``,
        ``v`` the variance of the step per unit variance of one epoch (white noise); a side
        without data gives step 0 and ``v`` = inf. Scalars or arrays of indices.
        """
        r = j + skip
        n_l, n_r = self.n[j] - self.n[lo], self.n[hi] - self.n[r]
        both = (n_l > 0) & (n_r > 0)
        # explicit mask (no floating-point warning): a window side without data divides by 1 instead of 0; its
        # values are replaced by (0, inf) below, so only windows with data on both sides are computed as fitted
        n_l, n_r = np.where(both, n_l, 1.0), np.where(both, n_r, 1.0)
        y_l, y_r = (self.y[j] - self.y[lo]) / n_l, (self.y[hi] - self.y[r]) / n_r
        step, v = y_r - y_l, 1.0 / n_l + 1.0 / n_r
        if linear:
            x_l, x_r = (self.x[j] - self.x[lo]) / n_l, (self.x[hi] - self.x[r]) / n_r
            sxx = (self.xx[j] - self.xx[lo] - n_l * x_l * x_l) + (self.xx[hi] - self.xx[r] - n_r * x_r * x_r)
            sxy = (self.xy[j] - self.xy[lo] - n_l * x_l * y_l) + (self.xy[hi] - self.xy[r] - n_r * x_r * y_r)
            ok = sxx > 1e-9
            sxx = np.where(ok, sxx, 1.0)
            dx = x_r - x_l
            step = step - np.where(ok, sxy / sxx, 0.0) * dx
            v = v + np.where(ok, dx * dx / sxx, 0.0)
        return np.where(both, step, 0.0), np.where(both, v, np.inf)


def _fit_rss(x, y, side, linear: bool) -> float:
    """Residual sum of squares of a fit with one level per ``side`` (False/True) and, if ``linear``, one common slope."""
    rss = sxx = sxy = 0.0
    for s in (False, True):
        sel = side == s
        if sel.any():
            dx, dy = x[sel] - x[sel].mean(), y[sel] - y[sel].mean()
            rss, sxx, sxy = rss + dy @ dy, sxx + dx @ dx, sxy + dx @ dy
    if linear and sxx > 1e-9:
        rss -= sxy * sxy / sxx
    return rss


def _level_shifts(y, *, linear: bool, k: float, floor, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Epochs at which a series that is smooth between cycle slips changes its level.

    The step at epoch ``j`` is fitted to ``window`` epochs before and ``window`` epochs from
    ``j`` (``_PrefixSums.step``). Its noise is measured on the series itself: robust scale
    (1.4826 x median absolute deviation) of the normalised steps over +-max(20, 4*window)
    epochs, so the threshold follows elevation, receiver, multipath and ionospheric activity.
    A step larger than ``max(k * noise, floor)`` (``floor`` a scalar or one value per epoch)
    starts a test of two explanations on the epochs around it: a level step at one of the next
    ``window`` epochs, or a single deviating epoch (code or phase outlier). The one with the
    smaller residual sum of squares wins (same number of parameters) if it is significant: the
    epoch deviates by more than the threshold, or the step stays above it without the epoch on
    either side. An outlier (at most two within a window) is left out of every later window and
    the test is repeated without it; a step is a slip, and the windows never reach back across
    it. NaN epochs are left out like outliers.
    Returns ``(flags, threshold, outliers)``: True at the first epoch of a new arc, the threshold
    of a step at each epoch (the smallest detectable slip there), True at outlier epochs.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    floor = np.broadcast_to(np.asarray(floor, dtype=float), (n,))
    flags = np.zeros(n, dtype=bool)
    outliers = np.zeros(n, dtype=bool)
    weight = np.isfinite(y).astype(float)
    sums = _PrefixSums(y, weight)
    j = np.arange(1, n)
    full_step = np.zeros(n)
    full_var = np.full(n, np.inf)
    full_step[1:], full_var[1:] = sums.step(np.maximum(0, j - window), j, np.minimum(n, j + window), linear)
    z = np.where(np.isfinite(full_var), full_step / np.sqrt(full_var), np.nan)
    if not np.isfinite(z).any():
        return flags, floor.copy(), outliers
    half = max(20, 4 * window)

    def local_noise(values):
        dev = pd.Series(np.abs(values - np.nanmedian(values)))
        return 1.4826 * dev.rolling(2 * half + 1, center=True, min_periods=1).median().to_numpy()

    def threshold(noise_, var, floor_):
        # a step without data on one side (var = inf) is never significant
        ok = np.isfinite(var)
        return np.maximum(np.where(ok, k * noise_ * np.sqrt(np.where(ok, var, 0.0)), np.inf), floor_)

    noise = local_noise(z)
    finite = np.isfinite(noise)
    noise = np.where(finite, noise, noise[finite].max())  # long NaN runs: the largest noise of the arc
    noise[0] = noise[1]
    full_thr = threshold(noise, full_var, floor)
    candidates = np.flatnonzero(np.abs(full_step) > full_thr)
    if len(candidates):
        # a slip disturbs the steps of the 2*window-1 epochs around it: measure the noise without them
        quiet = z.copy()
        for c in candidates:
            quiet[max(0, c - window + 1):c + window] = np.nan
        if np.isfinite(quiet).any():
            clean = local_noise(quiet)
            noise = np.where(np.isfinite(clean), clean, noise)
            noise[0] = noise[1]
            full_thr = threshold(noise, full_var, floor)
            candidates = np.flatnonzero(np.abs(full_step) > full_thr)

    x_all = np.arange(n, dtype=float)
    start, i = 0, 1
    while i < n:
        if i >= start + window:
            # windows not clipped by a slip: the precomputed steps apply
            pos = np.searchsorted(candidates, i)
            if pos == len(candidates):
                break
            i = int(candidates[pos])
            step, thr = full_step[i], full_thr[i]
        else:
            step, v = sums.step(start, i, min(n, i + window), linear)
            thr = threshold(noise[i], v, floor[i])
        if abs(step) > thr:
            # step at slip in [i, i+window) or outlier at o in [i-window, i+window): an outlier
            # at o moves the steps of the epochs o-window+1 ... o+window
            lo, hi = max(start, i - window), min(n, i + 2 * window)
            used = weight[lo:hi] > 0
            x, ys = x_all[lo:hi][used], y[lo:hi][used]
            slip, best_slip = i, np.inf
            for jj in range(i, min(n, i + window)):
                if weight[jj] > 0:
                    rss = _fit_rss(x, ys, x >= jj, linear)
                    if rss < best_slip:
                        best_slip, slip = rss, jj
            out, best_out = -1, np.inf
            for o in range(lo, min(n, i + window)):
                if weight[o] > 0:
                    keep = x != o
                    rss = _fit_rss(x[keep], ys[keep], np.zeros(keep.sum(), dtype=bool), linear)
                    if rss < best_out:
                        best_out, out = rss, o
            # more than two outliers within a window are a disturbance: end the arc there instead
            if best_out < best_slip and outliers[max(0, out - window):out + window + 1].sum() < 2:
                keep = x != out
                fit = np.polyfit(x[keep], ys[keep], 1) if linear and keep.sum() > 2 else [ys[keep].mean()]
                dev = y[out] - np.polyval(fit, float(out))
                if abs(dev) <= max(k * noise[out] * np.sqrt(1.0 + 1.0 / keep.sum()), floor[out]):
                    i += 1  # neither a significant outlier nor a step: noise
                    continue
                outliers[out] = True
                weight[out] = 0.0
                sums = _PrefixSums(y, weight)
                jj = np.arange(max(1, out - window + 1), min(n, out + window + 1))
                full_step[jj], full_var[jj] = sums.step(np.maximum(0, jj - window), jj,
                                                        np.minimum(n, jj + window), linear)
                full_thr[jj] = threshold(noise[jj], full_var[jj], floor[jj])
                candidates = np.flatnonzero(np.abs(full_step) > full_thr)
                i = max(start + 1, out - window + 1)
                continue
            # a slip is a step that stays without the epoch on either side of it (a single large
            # noise value is not a slip)
            tests = [sums.step(max(start, slip - window), slip, min(n, slip + window), linear),
                     sums.step(max(start, slip - window), slip, min(n, slip + 1 + window), linear, skip=1),
                     sums.step(max(start, slip - 1 - window), slip - 1, min(n, slip + window), linear, skip=1)]
            if any(np.isfinite(v) and abs(s) <= threshold(noise[slip], v, floor[slip]) for s, v in tests) \
                    or not np.isfinite(tests[0][1]):
                i += 1
                continue
            flags[slip] = True
            start, i = slip, slip + 1
            continue
        i += 1
    full_thr[0] = full_thr[1]
    return flags, full_thr, outliers


class CSDetector:
    """Cycle-slip detection and arc segmentation of one system's observations.

    Arcs are first split at data gaps longer than the sampling interval, then at cycle slips;
    arcs shorter than ``min_arc_lenth`` minutes are dropped. Each arc gets the label
    ``<sv>_<gap arc>_<slip arc>`` (the filters treat it as a satellite with its own ambiguity).

    Tests (``run(detector=None)``), each a least-squares level shift of a combination that is
    smooth between slips, with a threshold scaled by the noise measured on the data itself
    (``_level_shifts``):

    * dual-frequency modes: geometry-free phase L1 - L2 [m] (one slope common to 2.5 min on
      each side; ``GF_K`` x noise, at least ``GF_FLOOR_M``) and Melbourne-Wuebbena
      (f1 L1 - f2 L2)/(f1 - f2) - (f1 P1 + f2 P2)/(f1 + f2) in wide-lane cycles (means of 2.5 min;
      ``MW_K`` x noise, at least ``MW_FLOOR_CYC``). The Melbourne-Wuebbena test carries the close
      pairs (L2/L5, E5a/E5b, B2a/B2b: wide-lane 5.9-9.8 m, noise ~0.01 cycle), whose
      geometry-free signature of an equal slip on both signals is only n*|lambda2 - lambda1|
      (0.65-1.06 cm per cycle) - the former fixed threshold |lambda2 - lambda1| was below the
      noise of the 3-point prediction and cut arcs every few epochs. Where
      Melbourne-Wuebbena would see any slip n1 != n2 and saw none, the geometry-free threshold is
      at least 0.5*|lambda1 - lambda2| (2.7 cm for L1/L2): only n1 = n2 slips are left to find;
    * single-frequency modes: code minus phase (P - L) / lambda in cycles (2 x ionosphere drift
      as a common slope over 5 min on each side; ``CMC_K`` x noise, at least
      ``CMC_FLOOR_CYC``). Code noise limits it to slips of several cycles;
    * single deviating epochs found by the same tests do not split the arc next to them: a phase
      outlier (geometry-free) becomes a one-epoch arc, dropped as a short arc; a code outlier
      (Melbourne-Wuebbena or code minus phase only) stays, for the code screening of the filters;
    * dual frequency, where neither test would detect a one-cycle slip on one signal (threshold
      above the shorter wavelength and above one wide-lane cycle, e.g. under scintillation or at
      the lowest elevations) phase continuity cannot be verified: those epochs become one-epoch
      arcs, dropped as short arcs (``BREAK_UNVERIFIABLE``, ``BREAK_MAX_INTERVAL_S``);
    * the RINEX loss-of-lock indicator (bit 0, and a change of bit 1) of the processed phases:
      ``<phase>lli`` columns of the frame, else the side read of the slip context (``context``,
      ``gnx_py.cycle_slips.read_aux_observations``);
    * receiver millisecond clock jumps (``CLOCK_JUMP_GUARD``) are removed from the detector's
      copies of the codes first, so a 1 ms code-only jump breaks no arc (whenever the tests run,
      also with ``cycle_slip_aux_carrier`` and ``cycle_slip_geometry`` False; without the guard such
      a jump breaks the arcs of the satellites tracked at that epoch);
    * third carrier (L1 layer, with a ``context`` whose RINEX file has the carrier
      of ``gnss.AUX_SIGNALS``, at every sampling interval since v1.0.0 (calibrated at 30 s,
      measured up to 300 s; ``cycle_slips.AUX_MAX_INTERVAL_S`` = inf, a finite value switches it off
      above that interval); ``gnx_py.cycle_slips.third_carrier_flags``): the same pair tests on the cross pairs (third
      carrier, each processed signal), geometry-free flags only in a quiet ionosphere, the
      ionosphere-free geometry-free triple combination, slips on the third carrier alone attributed
      and not breaking the arc, unverifiable epochs kept where the cross pairs verify them, and a
      common-mode guard against ionospheric events that hit most satellites at once (the 2024-05-10
      flare; geometry-free-only flags dropped from one epoch before to ``CM_TAIL`` = 3 after); no
      products needed;
    * geometry (L2 layer, ``gnx_py.cycle_slips.GeometrySlipDetector``, with a ``context`` of precise
      orbits, or in the TEC session of broadcast orbits whose frames name their records (a record switch starts a new piece), at 30 s sampling, ``cycle_slip_geometry``): the epoch-differenced residual of every
      carrier (processed signals, plus the third carrier for the close pairs and single frequency)
      of every satellite of all systems of the run against the frame's orbits and clocks, one
      receiver-clock change per epoch (a kinematic run keeps this static model while the receiver
      does not move; a moving receiver is tracked at its integrated position),
      ionosphere-weighted chi2 test at 7 sigma (T1); windowed
      ionosphere-free level shifts for wide pairs (T2); a flag whose integer vector is a slip on the
      third carrier alone vetoes the third-carrier flags around it;
    * fusion (``gnx_py.cycle_slips.fuse_flags``): a T1 or third-carrier flag that no other test
      raises within one epoch is kept only if the clock-corrected ionosphere-free level steps by at
      least 7 sigma and 7 cm over the next 10 epochs (close pair without a third carrier: sigma_I
      <= 2 cm or a 5 sigma / 5 cm step; single frequency without a second band: sigma_I <= 3 cm; wide
      pair: 5 sigma / 5 cm of IF(a, b) for T1 flags, 5 cm for third-carrier flags, so a slip on the
      third carrier alone does not break a wide-pair arc); a flag without a complete window is kept.
      Without the geometry layer (sampling other than 30 s, broadcast orbits in PPP or without their records, or
      no result of the layer)
      a third-carrier flag that no other test, LLI or common-mode event raises within one epoch breaks
      the arc only when the integer estimate of the cross pairs cannot exclude a slip on the processed
      signals (``cycle_slips._no_processed_slip``).

    Found at the slip epoch with all layers (library bench, 30 s, 30 s CLK, BRUX / NKLG / NYA2
    2024-035 / NYA2 2024-132, held-out BRUX / NKLG / NYA2 2024-131; today in brackets): (1, 1) L2/L5
    100 % at all seven (58.7 / 32.7 / 2.2 / 7.0; 47.3 / 30.6 / 2.1), (1, 1, 1) 100 / 99.7 / 96.1 /
    97.4; 99.4 / 98.9 / 97.1 %; (1, 1) E5a/E5b 100 % (1.6-42.7), (1, 1, 1) 96.2-100 %; (1, 1)
    B2a/B2b 100 % (19.5-38.0); (1, 1) L1/L2 100 / 100 / 99.6 / 99.6; 100 / 98.8 / 98.7 % (95.4 /
    84.4 / 8.8 / 21.5; 86.5 / 71.0 / 11.8); one cycle on L1 with L2W 99.6-100 % (0 %). New
    ambiguities per satellite-hour on clean data: at most +0.016 over today (L2/L5 0 / 0.030 / 0.118
    / 0.140; 0.070 / 0.039 / 0.096 against 0 / 0.030 / 0.104 / 0.140; 0.070 / 0.039 / 0.088).
    Epochs lost on L1/L2 (unverifiable, short arcs) 0.06 / 0.48 / 0.26 / 0.80 % against 0.15 /
    0.81 / 4.44 / 3.86 %. A slip on the third carrier alone breaks the arc in 0-3.9 % of the cases.

    Basis of the constants (BRUX/NKLG/NYA2 2024-035, BRUX 2024-132, 30 s): 0.00-0.22 detected
    slips per satellite and hour in every mode (the former detector: up to 24 per hour for the
    close pairs, 2-36 per hour for every pair at NKLG/NYA2 and on the storm day); the
    unverifiable epochs are 0-0.3 % of the data at BRUX, up to 4-9 % at NYA2; synthetic slips of
    1, 2 and 5 cycles on one signal found at the slip epoch in 98-100 % of the cases. Other
    sampling intervals: windows in time (at least 3/5 epochs); at 120-300 s the linear
    geometry-free model leaves ionospheric curvature, which gives more (mostly spurious) slips.
    RTKLIB (ppp.c) compares epoch-to-epoch changes: geometry-free
    ``thresslip`` = 0.05 m and Melbourne-Wuebbena 10 m (``THRES_MW_JUMP``), plus LLI; Blewitt
    (1990, TurboEdit) tests Melbourne-Wuebbena against its running mean (4 sigma).

    Fallbacks (same bench): broadcast orbits or sampling other than 30 s (no geometry): (1, 1) on
    L2/L5 95.6-99.7 %, (1, 1, 1) at NYA2 5.3-11.7 %, L1/L2 (1, 1) at NYA2 59.0-61.1 %; close pair
    without a third carrier: (1, 1) L2/L5 98.8 / 87.5 / 17.2 / 32.6 %; single frequency without a
    second band, one cycle on L1: 99.2 / 90.6 / 20.9 / 38.8 %; 5-min SP3 clocks (NYA2 2024-035):
    (1, 1, 1) L2/L5 85.9 %, L1/L2 (1, 1) 62.6 % (``clk_path``: 30 s clocks); kinematic: a static
    receiver keeps the static numbers, a simulated 2 km circle (tracked) (1, 1, 1) 100 % (BRUX) / 94.7 %
    (NYA2, 0-8 h), GPS only or single frequency without a second band on it: motion guard. Without products and
    third carrier (other modes, RINEX 2, ``cycle_slip_aux_carrier=False``) the limits of the
    combinations stay (BRUX/NYA2 2024-035):

    * equal slips (n, n) on a close pair - two carriers close in frequency, L2/L5 (1227.60 /
      1176.45 MHz), E5a/E5b (1176.45 / 1207.14 MHz), B2a/B2b, whose wavelengths differ by about
      1 cm: Melbourne-Wuebbena does not change (the wide-lane ambiguity N1 - N2 stays the same) and
      the geometry-free phase changes only by n*|lambda1 - lambda2|, 1.06 cm per cycle for L2/L5,
      0.65 cm for E5a/E5b, near its noise. Found: (1, 1) 75 % (L2/L5) / 35 % (E5a/E5b), (2, 2)
      79 / 72.5 %, (5, 5) 96 / 97.5 %. On L1/L2 (5.4 cm per cycle) at NYA2 (polar ionosphere):
      (1, 1) 12 %, (2, 2) 27 %. A slip on one signal only, or n1 != n2, changes Melbourne-Wuebbena
      and is found in 98-100 % of the cases;
    * single frequency: code minus phase is code-noise limited (code noise and multipath of tens
      of cm against a 19-25 cm wavelength): 5 cycles found 29 % (L1) / 55 % (E1) / 28 % (B1I),
      10 cycles 79 / 97.5 / 45 %, 20 cycles 100 / 100 / 95 %; a one-cycle phase outlier is not
      isolated (below the code-minus-phase noise);
    * at 120-300 s sampling the linear geometry-free model leaves ionospheric curvature: about
      0.14 spurious slips per satellite and hour at 300 s (BRUX, GPS); each one only starts a new
      ambiguity. The third carrier runs there too (since v1.0.0; it was off
      above 60 s before). Bench BRUX/NKLG/NYA2 2024-035, 24 h thinned from 30 s, without -> with
      it: (1, 1) on L2/L5 and E5a/E5b 0.0-27.4 -> 81.0-97.2 % at 120 s, 0.4-3.5 -> 42.1-98.7 % at
      300 s; one cycle on single-frequency L1/E1 0.0-1.1 -> 63.9-84.2 % (120 s), 0.0-0.6 ->
      6.2-58.8 % (300 s); (1, 1, 1) stays low without the geometry layer (0.6-9.4 % at 300 s).
      Its cost: a slip on the third carrier alone breaks the processed arc in 25.1-93.9 % (120 s)
      and 4.8-96.0 % (300 s) of the cases (0-8.9 % at 30 s with the geometry attribution), new
      ambiguities per satellite-hour on clean data +0.000-0.016 (120 s) and +0.000-0.033 (300 s;
      BRUX L2/L5 0.074 -> 0.107), epochs lost up to +1.9 points (NYA2 L2/L5 at 300 s 0.75 ->
      2.60 %). Storm day 2024-132: same pattern, third carrier alone breaks the arc 3.5-97.0 %,
      new ambiguities up to +0.032 (BRUX L1 at 120 s 0.000 -> 0.032), epochs lost up to +1.5
      points (NYA2 L2/L5 at 300 s 2.27 -> 3.80 %). Default modes L1/L2 and E1/E5a (quiet and
      storm): (1, 1) 0.6-65.1 -> 52.5-100 %, third carrier alone breaks the arc 29.8-100 %, new
      ambiguities up to +0.020, epochs lost up to +0.9 points. At 300 s on single frequency a
      flag is about as likely to come from a slip of the third carrier as of the processed signal
      (found ~ broken, e.g. NKLG L1 16.5 vs 17.0 %). TEC sessions pass the same slip context, so
      they get the layer above 60 s too (not benched);
    * the RINEX loss-of-lock indicators are used only when the frame carries ``<phase>lli``
      columns or a slip context reads them (the PPP and TEC sessions do, from RINEX 3 files, with
      ``cycle_slip_aux_carrier=True``).

    An undetected slip stays in the ambiguity of its arc until the arc ends; the phase-jump
    screening of the filters (on by default on the constrained uncombined routes) and, for PPP-AR,
    the fix validation catch the larger ones.
    """

    #: geometry-free test: threshold in noise sigmas, minimum [m]
    GF_K = 5.0
    GF_FLOOR_M = 0.004
    #: Melbourne-Wuebbena test: threshold in noise sigmas, minimum [wide-lane cycles]
    MW_K = 5.0
    MW_FLOOR_CYC = 0.5
    #: time on each side of a step [s] (both dual-frequency tests; 5 epochs at 30 s), at least 3
    #: epochs for the geometry-free test (ionospheric curvature) and 5 for Melbourne-Wuebbena (code noise)
    DUAL_WINDOW_S = 150.0
    #: dual frequency: end the arc where neither test would detect a one-cycle slip on one signal,
    #: for sampling intervals up to ``BREAK_MAX_INTERVAL_S`` (the constants are calibrated on 30 s
    #: data; at 60 s the rule costs about as much as at 30 s, at 120-300 s ionospheric curvature
    #: inflates the geometry-free noise and the rule would drop 5-10 % of a quiet day)
    BREAK_UNVERIFIABLE = True
    BREAK_MAX_INTERVAL_S = 60.0
    #: single-frequency code-minus-phase test: threshold in noise sigmas, minimum [cycles], time on each
    #: side [s]. The minimum keeps code level jumps of 1-2 cycles (semi-codeless C2W at BRUX) from splitting
    #: arcs: a new single-frequency ambiguity takes hours to converge again
    CMC_K = 5.0
    CMC_FLOOR_CYC = 2.0
    SINGLE_WINDOW_S = 300.0
    #: remove receiver millisecond clock jumps from the detector's copies of the codes before the tests
    #: (``gnx_py.cycle_slips.clock_jump_corrections``): a 1 ms code-only jump would break every arc through
    #: Melbourne-Wuebbena and code minus phase
    CLOCK_JUMP_GUARD = True

    def __init__(self,obs, phase_shift_dict, dcb, sys='G',mode='L1L2',interval=30, context=None):
        #: ``gnx_py.cycle_slips.SlipContext`` of the system (LLI and third carrier from the RINEX file, the
        #: master switch); None: the frame's own columns and every test of the mode
        self.context = context
        self.obs = obs
        self.phase_shift_dict = phase_shift_dict
        self.sys = sys

        self.c = 299792458
        self.sampling_rate = interval
        self.M = 5  # interpolation and CS repair window size
        self.dcb = dcb
        self.mode=mode
        self.F = {signal: frequency_hz(signal) for signal in {
            'L1', 'L2', 'L5', 'E1', 'E5a', 'E5b', 'B1I', 'B1C', 'B2a', 'B2I', 'B2b', 'B3I'
        }}
        self.l = {signal: wavelength_m(signal) for signal in self.F}
        if self.obs.index.get_level_values(0).name == 'time':
            self.obs = self.obs.swaplevel(0, 1)

    def _signal_columns(self):
        """``[(code column, phase column), ...]`` of the mode's signals (indicator columns excluded)."""
        def first(prefix):
            return [col for col in self.obs.columns
                    if col.startswith(prefix) and not col.endswith(('lli', 'ssi'))][0]

        signals = mode_signals(self.mode)
        if len(signals) not in (1, 2):
            raise IndexError(f'Unsupported signal mode {self.mode!r} for cycle-slip detection')
        out = []
        for signal in signals:
            spec = signal_spec(signal)
            try:
                out.append((first(spec.code_prefix), first(spec.phase_prefix)))
            except IndexError:
                kind = 'dual' if len(signals) == 2 else 'single'
                raise IndexError(f'Lack of {kind} frequency data!') from None
        return signals, out

    def make_geometry_free(self):
        signals, cols = self._signal_columns()
        if len(signals) != 2:
            raise IndexError('Lack of dual frequency data!')
        (p1, l1), (p2, l2) = cols
        p4 = self.obs[p2] - self.obs[p1]
        l4 = self.obs[l1] - self.obs[l2]
        self.obs['P4'] = p4
        self.obs['L4'] = l4

    def split_obs_dt(self):
        """
        Splits the observations of every PRN into pieces continuous in time, at gaps longer than sampling_rate.
        """
        out = {}
        for prn, group in self.obs.groupby('sv'):
            # Przepisujemy 'time' jako kolumnę (upewnij się, że jest typu Timestamp!)
            time_idx = group.index.get_level_values('time')
            times = pd.to_datetime(time_idx)
            # Różnice czasowe między epokami (w sekundach)
            dt = times.to_series().diff().dt.total_seconds().fillna(0).to_numpy()
            # Gdzie przerwa > sampling_rate, tam zaczyna się nowy fragment
            breaks = (dt > self.sampling_rate)
            # Każdemu fragmentowi przypisujemy numer
            frag_id = breaks.cumsum()
            # Dodajemy do DataFrame
            df = group.reset_index()
            df['frag'] = frag_id
            # Splitujemy po frag_id
            for n, frag in df.groupby('frag'):
                key = f"{prn}_{n}"
                # Ustawiamy indeks 'time', usuwamy kolumny indeksowe
                frag = frag.set_index('time').drop(columns=['sv', 'frag'])
                out[key] = frag
        # Łączymy – klucz główny to sv_x, drugi poziom to czas
        return pd.concat(out, keys=out.keys(), names=['sv', 'time'])

    def drop_short_arcs(self, df, t):
        """

        :param df: dataframe with index sv orz multiindex with sv level
        :param t: minimum arc length (minutes)
        :return: filtered df
        """
        # min_arc_len = int(self.sampling_rate/60 * t)
        min_arc_len = int(t / (self.sampling_rate/60))
        filtered_df = df.groupby('arc').filter(lambda x: len(x)>min_arc_len)
        return filtered_df


    @classmethod
    def _window(cls, span_s, interval, minimum):
        """Epochs on each side of a step for ``span_s`` seconds at ``interval`` seconds, at least ``minimum``."""
        return max(minimum, int(round(span_s / cls._interval(interval))))

    @staticmethod
    def _interval(interval):
        """Sampling interval [s]; 30 s when unknown."""
        return float(interval) if interval and float(interval) > 0 else 30.0

    @staticmethod
    def _isolate(flags, epochs):
        """Make each of ``epochs`` (bool mask) a one-epoch arc, dropped as a short arc."""
        flags = flags | epochs
        flags[1:] |= epochs[:-1]
        flags[:1] = False  # the arc starts there anyway
        return flags

    @classmethod
    def dual_frequency_slips(cls, l1, l2, p1, p2, mode, detector=None, lli=None, interval=30.0):
        """Slip flags of one continuous arc of a dual-frequency mode (True: first epoch after a slip).

        ``l1``/``l2``: phases [m], ``p1``/``p2``: codes [m] of the mode's signals; ``detector``:
        None (geometry-free and Melbourne-Wuebbena), "GF" or "MW"; ``lli``: loss-of-lock
        indicators of the epochs (one column per signal, see :meth:`_lli_slips`); ``interval``:
        sampling interval [s]. Geometry-free (phase) outlier epochs become one-epoch arcs.
        """
        s1, s2 = mode_signals(mode)
        t = cls._pair_tests(l1, l2, p1, p2, frequency_hz(s1), frequency_hz(s2), interval, detector=detector)
        return cls._isolate(t["slips"], t["outliers"] | t["unverifiable"]) | cls._lli_slips(lli, len(t["slips"]))

    @classmethod
    def _pair_tests(cls, l1, l2, p1, p2, f1, f2, interval=30.0, detector=None) -> dict:
        """Geometry-free and Melbourne-Wuebbena tests of one continuous arc of any two carriers.

        ``l1``/``l2`` phases [m], ``p1``/``p2`` codes [m] at the frequencies ``f1``/``f2`` [Hz];
        ``detector`` as in :meth:`dual_frequency_slips`. Returns the details without isolation:
        ``slips`` (MW | GF flags), ``mw_flags``/``gf_flags``, ``mw_thr`` [wide-lane cycles] /
        ``gf_thr`` [m] (smallest detectable step per epoch; None for a test not run),
        ``outliers`` (geometry-free phase outliers), ``unverifiable`` (epochs where neither test
        would see a one-cycle slip on one signal), the series ``mw``/``gf``, the windows ``w_mw``/
        ``w_gf`` and the wavelengths ``lam``. The mode's pair (:meth:`dual_frequency_slips`) and
        the cross pairs with a third carrier (``gnx_py.cycle_slips``) use the same tests.
        """
        lam1, lam2 = CLIGHT / f1, CLIGHT / f2
        l1, l2, p1, p2 = (np.asarray(a, dtype=float) for a in (l1, l2, p1, p2))
        n = len(l1)
        out = {"lam": (lam1, lam2), "mw": None, "gf": None, "mw_thr": None, "gf_thr": None,
               "w_mw": cls._window(cls.DUAL_WINDOW_S, interval, 5),
               "w_gf": cls._window(cls.DUAL_WINDOW_S, interval, 3)}
        mw_flags = np.zeros(n, dtype=bool)
        gf_flags = np.zeros(n, dtype=bool)
        outliers = np.zeros(n, dtype=bool)
        unverifiable = np.zeros(n, dtype=bool)
        gf_floor = np.full(n, cls.GF_FLOOR_M)
        if detector in (None, "MW"):
            wide_lane = CLIGHT / abs(f1 - f2)
            mw = ((f1 * l1 - f2 * l2) / (f1 - f2) - (f1 * p1 + f2 * p2) / (f1 + f2)) / wide_lane
            # (an epoch deviating in Melbourne-Wuebbena only is a code outlier: neither a slip nor a
            # dropped epoch - a dropped epoch restarts the ambiguity; the filters screen code)
            mw_flags, mw_thr, _ = _level_shifts(mw, linear=False, k=cls.MW_K, floor=cls.MW_FLOOR_CYC,
                                                window=out["w_mw"])
            out["mw"], out["mw_thr"] = mw, mw_thr
            if detector is None:
                # where Melbourne-Wuebbena would see any slip n1 != n2 and saw none, a slip can only
                # be n1 = n2 = n, geometry-free signature n*(lambda1 - lambda2): steps below half of
                # it are disturbances (multipath, ionosphere), not slips
                near = mw_flags.copy()
                near[1:] |= mw_flags[:-1]
                near[:-1] |= mw_flags[1:]
                gf_floor = np.where((mw_thr < 1.0) & ~near,
                                    max(cls.GF_FLOOR_M, 0.5 * abs(lam1 - lam2)), cls.GF_FLOOR_M)
        if detector in (None, "GF"):
            gf = l1 - l2
            gf_flags, gf_thr, gf_out = _level_shifts(gf, linear=True, k=cls.GF_K, floor=gf_floor,
                                                     window=out["w_gf"])
            out["gf"], out["gf_thr"] = gf, gf_thr
            outliers |= gf_out
        if detector is None and cls.BREAK_UNVERIFIABLE and cls._interval(interval) <= cls.BREAK_MAX_INTERVAL_S and n:
            # continuity not verifiable: neither test would see a one-cycle slip on one signal
            # (geometry-free signature lambda, Melbourne-Wuebbena 1 wide-lane cycle), e.g. under
            # scintillation; such epochs become one-epoch arcs (dropped as short arcs)
            unverifiable = (out["gf_thr"] > min(lam1, lam2)) & (out["mw_thr"] > 1.0)
        out.update(slips=mw_flags | gf_flags, mw_flags=mw_flags, gf_flags=gf_flags, outliers=outliers,
                   unverifiable=unverifiable)
        return out

    @classmethod
    def single_frequency_slips(cls, phase, code, signal, lli=None, interval=30.0):
        """Slip flags of one continuous arc of a single-frequency mode (code minus phase test).

        ``phase``/``code`` [m] of ``signal``; ``lli`` and ``interval`` as in :meth:`dual_frequency_slips`.
        """
        cmc = (np.asarray(code, dtype=float) - np.asarray(phase, dtype=float)) / wavelength_m(signal)
        # a single deviating epoch is a code outlier (code noise >> phase noise): not a slip, and
        # kept (as in dual_frequency_slips)
        flags, _, _ = _level_shifts(cmc, linear=True, k=cls.CMC_K, floor=cls.CMC_FLOOR_CYC,
                                    window=cls._window(cls.SINGLE_WINDOW_S, interval, 5))
        return flags | cls._lli_slips(lli, len(flags))

    @staticmethod
    def _lli_slips(lli, n):
        """Slips from RINEX loss-of-lock indicators, one column per signal (or one value per epoch).

        Bit 0 (loss of lock, slip possible) marks a slip. Bit 1 (half-cycle ambiguity) may stay set
        over many epochs; only a change of it, when the receiver resolves or loses the half
        cycle, is a phase jump (RTKLIB ppp.c treats ``LLI & 3`` as a slip at every epoch).
        """
        if lli is None:
            return np.zeros(n, dtype=bool)
        lli = np.nan_to_num(np.asarray(lli, dtype=float), nan=0.0).astype(np.int64).reshape(n, -1)
        half = (lli & 2).astype(bool)
        flags = (lli & 1).astype(bool)
        flags[1:] |= half[1:] != half[:-1]
        flags = flags.any(axis=1)
        flags[:1] = False  # a new arc starts there anyway
        return flags

    def split_obs_cs(self, detector=None):
        """Split the continuous arcs of ``split_obs_dt`` at cycle slips (column ``arc``: ``<sv>_<n>``).

        Order: LLI, the millisecond clock-jump guard, today's tests of the mode, the third-carrier tests
        and the common-mode guard over all satellites (``context`` with a third carrier, ``detector``
        None), the geometry layer of the receiver run (``context.geometry_rows``, computed once for all
        systems), fusion with the confirmation of flags that one layer alone raises and the attribution
        veto (``cycle_slips.fuse_flags``; unverifiable epochs kept where the cross pairs or the geometry
        verify them), isolation and labels. A context with ``tests=False`` (``cycle_slips_detection=False``)
        splits at LLI only. ``make_geometry_free`` stays before the split: no phase is repaired in 8c.
        A change (set or clear) of the optional per-row column ``lli_half_cycle`` (RINEX LLI bit 1)
        is an LLI break, also with ``tests=False``.
        """
        if detector not in (None, "GF", "MW"):
            raise ValueError(f"detector must be one of [None, 'GF', 'MW'] (got {detector!r})")
        from .cycle_slips import (AUX_MAX_INTERVAL_S, _log_once, clock_jump_corrections, common_mode_events,
                                  fuse_flags, half_cycle_breaks, third_carrier_flags)

        signals, cols = self._signal_columns()
        obs = self.obs.copy()
        ctx = self.context
        tests = ctx is None or ctx.tests
        dual = len(signals) == 2
        interval = self.sampling_rate
        n_rows = len(obs)
        piece_keys = obs.index.get_level_values('sv')
        piece_id = pd.factorize(piece_keys)[0]
        # nanosecond times whatever the unit of the frame (datetime64[s] / [ms] / [us] frames are allowed)
        times = pd.DatetimeIndex(pd.to_datetime(obs.index.get_level_values('time'))).as_unit('ns')
        t_s = ((times - times.min()) / pd.Timedelta(seconds=1)).to_numpy(dtype=float) if n_rows else np.zeros(0)
        epoch = np.rint(t_s / self._interval(interval)).astype(np.int64)
        # L0: side-read LLI and third carrier (gnx_py.cycle_slips), aligned to the rows
        aux = ctx.aux_frame() if ctx is not None else None
        aux_rows = None
        if aux is not None and n_rows:
            prn = np.array([str(s).rsplit('_', 1)[0] for s in piece_keys])
            aux_rows = aux.reindex(pd.MultiIndex.from_arrays([prn, times], names=['sv', 'time']))
        lli_cols = [f"{phase}lli" for _, phase in cols if f"{phase}lli" in obs.columns]
        if lli_cols:
            lli_all = obs[lli_cols].to_numpy(dtype=float)
        else:
            side = [f"{phase}lli" for _, phase in cols if aux_rows is not None and f"{phase}lli" in aux_rows.columns]
            lli_all = aux_rows[side].to_numpy(dtype=float) if side else None
        codes = [obs[p].to_numpy(dtype=float) for p, _ in cols]
        phases = [obs[l].to_numpy(dtype=float) for _, l in cols]
        freqs = [frequency_hz(s) for s in signals]
        use_l1 = (tests and detector is None and aux_rows is not None and ctx.aux_signal is not None
                  and np.isfinite(aux_rows['aux_L'].to_numpy(dtype=float)).any())
        if use_l1 and self._interval(interval) > AUX_MAX_INTERVAL_S:
            # calibrated at 30 s; on at every interval since bug-fix stage 9 unless AUX_MAX_INTERVAL_S is set
            # to a finite value (cycle_slips)
            _log_once(f"{self.sys} {self.mode}", f"third carrier off at {self._interval(interval):g} s sampling "
                      f"(measured up to {AUX_MAX_INTERVAL_S:g} s); today's tests plus LLI")
            use_l1 = False
        if use_l1:
            fc = frequency_hz(ctx.aux_signal)
            lc = aux_rows['aux_L'].to_numpy(dtype=float)
            pc = aux_rows['aux_P'].to_numpy(dtype=float)
            lli_c = aux_rows['aux_lli'].to_numpy(dtype=float)
            if ctx.windup_applied and 'phw' in obs.columns:
                # the frame phases carry -lambda*phw (TEC): the third carrier the same
                lc = lc - CLIGHT / fc * obs['phw'].to_numpy(dtype=float)
        # L0: receiver millisecond clock jumps removed from the detector's copies of the codes
        if tests and n_rows and self.CLOCK_JUMP_GUARD:
            jump = clock_jump_corrections(epoch, piece_id, codes[0] - phases[0])
            if jump.any():
                codes = [c - jump for c in codes]
                if use_l1:
                    pc = pc - jump
        groups = obs.groupby(level='sv', sort=False).indices
        per = {}
        gf_only = np.zeros(n_rows, dtype=bool)
        active = np.zeros(n_rows, dtype=bool)
        # L0: half-cycle status of the frame (contract C3 column, bug-fix stage 13b NQ-PPP-37): a change starts an arc
        half = obs['lli_half_cycle'] if 'lli_half_cycle' in obs.columns else None
        for sv, rows in groups.items():
            lli_flags = self._lli_slips(lli_all[rows] if lli_all is not None else None, len(rows))
            if half is not None:
                lli_flags = lli_flags | half_cycle_breaks(half.iloc[rows])
            if not tests:  # cycle_slips_detection=False: gaps and LLI only
                per[sv] = (lli_flags, None, None, None, None)
                continue
            if dual:
                t = self._pair_tests(phases[0][rows], phases[1][rows], codes[0][rows], codes[1][rows],
                                     freqs[0], freqs[1], interval, detector=detector)
                slips, outl, unver = t['slips'], t['outliers'], t['unverifiable']
            else:
                slips = self.single_frequency_slips(phases[0][rows], codes[0][rows], signals[0], interval=interval)
                outl = unver = np.zeros(len(rows), dtype=bool)
            l1 = None
            if use_l1:
                l1 = third_carrier_flags([p[rows] for p in phases], [c[rows] for c in codes], freqs, slips,
                                         lc[rows], pc[rows], fc, lli_c[rows], interval)
                gf_only[rows], active[rows] = l1['gf_only'], l1['active']
            per[sv] = (lli_flags, slips, outl, unver, l1)
        # L1: common-mode (flare) guard over all satellites of the system
        event = common_mode_events(epoch, gf_only, active, piece_id) if use_l1 else np.zeros(n_rows, bool)
        # L2: geometry layer (computed once per receiver run for all systems), aligned to the rows
        geo = None
        if tests and detector is None and ctx is not None and n_rows:
            prn = np.array([str(s).rsplit('_', 1)[0] for s in piece_keys])
            geo = ctx.geometry_rows(prn, times.asi8)
        arc = np.empty(n_rows, dtype=object)
        for sv, rows in groups.items():
            lli_flags, slips, outl, unver, l1 = per[sv]
            if slips is None:
                flags = lli_flags
            else:
                # L3: fusion with the confirmation of flags that one layer alone raises (cycle_slips.fuse_flags)
                g = None if geo is None else {k: v[rows] for k, v in geo.items()}
                slips, iso = fuse_flags(slips, lli_flags, outl, unver, l1, event[rows], g)
                flags = self._isolate(slips, iso) | lli_flags
            arc_nr = np.cumsum(flags) + 1
            arc[rows] = [f"{sv}_{nr}" for nr in arc_nr]
        obs['arc'] = arc
        return obs


    def rolling_lagrange_pred(self, l4, predict_at):
        """
        Former slip detector (not used by ``run``; kept for compatibility).
        Rolling 3-point Lagrange prediction: for every window of 3 samples the value of the polynomial at predict_at.
        predict_at = 3 -> predicts the value at i+3 from i, i+1, i+2
        Returns: array of shape (len(l4)-2,)
        """
        k = predict_at
        n = len(l4)
        if n < k:
            return np.array([])
        # Wagi Lagrange'a dla punktów 0,1,2 w punkcie 'predict_at'
        # Wzory analityczne:
        x = np.arange(k)
        L = np.ones(k)
        for j in range(k):
            for m in range(k):
                if m != j:
                    L[j] *= (predict_at - x[m]) / (x[j] - x[m])
        # Teraz rolling dot product:
        # Rolling okna 3 elementowe:
        strided = np.lib.stride_tricks.sliding_window_view(l4, window_shape=k)
        # shape = (n-k+1, k)
        preds = np.dot(strided, L)
        return preds

    def lagrange_filter_vectorized(self, l4, ev, mode, l):
        """
        Former slip detector (not used by ``run``; kept for compatibility).
        Vectorised version of lagrange_filter, without a loop over the points.
        """
        max_elevation_index = np.argmax(ev)
        indices = []

        try:
            signals = mode_signals(mode)
            t = abs(wavelength_m(signals[1]) - wavelength_m(signals[0])) if len(signals) == 2 else 0.5
        except ValueError:
            t = 0.5

        # W prawo od max_elevation_index:
        right_start = max_elevation_index
        if right_start < len(l4) - 3:
            l4_right = l4[right_start:]
            preds_right = self.rolling_lagrange_pred(l4_right, predict_at=3)[:-1]
            n_pred = len(preds_right)
            actuals_right = l4[right_start + 3:right_start + 3 + n_pred]
            assert len(preds_right) == len(
                actuals_right), f"preds_right: {len(preds_right)}, actuals_right: {len(actuals_right)}"
            diffs = preds_right - actuals_right

            outlier_idx = np.where(np.abs(diffs) > t)[0]
            indices.extend(right_start + 3 + outlier_idx)

        # W lewo od max_elevation_index:
        left_end = max_elevation_index
        if left_end > 2:
            l4_left = l4[:left_end + 1][::-1]
            preds_left = self.rolling_lagrange_pred(l4_left, predict_at=3)[:-1]
            n_pred = len(preds_left)
            actuals_left = l4_left[3:3 + n_pred]
            assert len(preds_left) == len(
                actuals_left), f"preds_left: {len(preds_left)}, actuals_left: {len(actuals_left)}"
            outlier_idx = np.where(np.abs(preds_left - actuals_left) > t)[0]
            # Indeksy przeliczyć do oryginalnej kolejności!
            indices.extend(left_end - 3 - outlier_idx)

        indices = np.unique(indices)
        differences = []  # Możesz dorzucić jeśli chcesz

        return np.array(differences), indices



    def run(self, min_arc_lenth=30, detector: Optional[Literal["MW","GF"]]=None):
        """Arcs of continuous phase: data gaps, cycle slips, short arcs dropped.

        ``detector``: None - every test of the mode (dual frequency: geometry-free and
        Melbourne-Wuebbena; single frequency: code minus phase), "GF" or "MW" - only that
        dual-frequency test. LLI is used where the frame carries ``<phase>lli`` columns, else
        where the slip context reads it from the RINEX 3 file (``context``; not with
        ``cycle_slip_aux_carrier=False``).
        """
        if len(self._signal_columns()[0]) == 2:
            self.make_geometry_free()
        self.obs = self.split_obs_dt()
        self.obs = self.split_obs_cs(detector=detector)
        # SV_dt_cs
        self.obs = self.drop_short_arcs(df=self.obs,t=min_arc_lenth)
        return self.obs

@dataclass
class DefaultConfig:
    """Internal helper: the minimal configuration :class:`DDPreprocessing` reads when no session configuration is
    given.

    Used by the SIS session (``gnx_py.orbits``) and by direct ``DDPreprocessing`` calls; it is not a session
    configuration (no validation, no input paths) and is kept importable for that use only. Every setting is a
    real dataclass field: ``DefaultConfig(day_of_year=35, ionosphere_model="gim", gim_path=...)`` works
    (``day_of_year``, ``gim_path`` and ``ionosphere_model`` were once class attributes without annotations). The fields keep their positional order (``gps_freq`` ... ``bds_klobuchar`` first); the
    satellite/receiver PCO and the antenna height are ``DDPreprocessing`` arguments (``sat_pco``, ``rec_pco``,
    ``antenna_h``), not fields. Sessions use :class:`gnx_py.configuration.SPPConfig`, ``PPPConfig``,
    ``TECConfig`` and ``gnx_py.orbits.SISConfig``.
    """
    gps_freq: str = "L1L2"  # Domyślne wartości
    gal_freq: str = "E1E5a"
    bds_freq: str = "B1IB3I"
    station_name: Optional[str] = 'Station'
    bds_klobuchar: str = "gps"
    # fields added after the original positional order (gps_freq ... bds_klobuchar) stays unchanged
    #: None: the day of year of the first observation epoch
    day_of_year: Optional[int] = None
    troposphere_model: Optional[str] = None
    gim_path: Optional[str] = None
    ionosphere_model: Optional[str] = None
    #: satellite without an ANTEX entry: 'exclude' (one warning) or 'raise' as the session configurations; None
    #: takes the policy of the antenna model, else 'exclude'
    missing_sat_antenna: Optional[str] = None
    #: satellite attitude (as ``Config``): None/'model' attitude laws or ORBEX, 'nominal' yaw
    #: steering; ``attitude_laws`` {law family: law}; ``attitude_path`` ORBEX file(s)
    satellite_attitude: Optional[str] = None
    attitude_laws: Optional[dict] = None
    attitude_path: Optional[Union[str, Path, list]] = None

class DDPreprocessing:
    def _day_of_year(self) -> int:
        """Day of year for the troposphere model.

        ``config.day_of_year=None`` takes the day of year of the first observation epoch;
        a configured value that matches none of the observation days is used as given,
        with a warning.
        """
        configured = getattr(self.configuration, "day_of_year", None)
        obs_days = []
        has_time = isinstance(self.df, pd.DataFrame) and "time" in (self.df.index.names or [])
        try:
            if has_time:
                times = pd.DatetimeIndex(self.df.index.get_level_values("time")).dropna()
                if len(times):
                    obs_days = sorted({int(d) for d in times.dayofyear.unique()})
                    first = int(times.min().dayofyear)
        except (TypeError, ValueError):
            obs_days = []
        if configured is None:
            if obs_days:
                return first
            if not has_time and getattr(self.configuration, "troposphere_model", None) in ("niell", "collins"):
                warnings.warn(
                    "day_of_year=None and the observations carry no 'time' index: the troposphere model "
                    "uses day_of_year=1; set day_of_year in the configuration", UserWarning, stacklevel=3)
            return 1
        uses_doy = getattr(self.configuration, "troposphere_model", None) in ("niell", "collins")
        if uses_doy and obs_days and int(configured) not in obs_days:
            warnings.warn(
                f"day_of_year={configured} differs from the day of year of the observations "
                f"({', '.join(str(d) for d in obs_days)}); the troposphere model uses day_of_year={configured}",
                UserWarning, stacklevel=3)
        return configured

    def __init__(
        self,
        df,
        flh,
        xyz,

        sat_pco,
        rec_pco,
        antenna_h,
        system,
        config: Optional[Union["PPPConfig","DefaultConfig","SPPConfig"]] = DefaultConfig,
        gpsa=None,
        gpsb=None,
        gala=None,
        phase_shift_dict=None,
        bdsa=None,
        bdsb=None,
        *,
        antenna: Optional[AntennaModel] = None,
        geometry_free: bool = False,
        iono: Optional[pd.DataFrame] = None,
        run_notes: Optional[RunNotes] = None,
        phase_shift_attrs: Optional[dict] = None,
    ):
        """
        Preprocessing class for double difference solution.

        Parameters:
        - df: DataFrame with observations and coordinates
        - flh: (lat, lon, height)
        - xyz: ECEF coordinates
        - config: PPPConfig instance
        - gpsa/gpsb/gala: model parameters if needed, passed optionally
        - bdsa/bdsb: BeiDou Klobuchar coefficients (BDSA/BDSB of the navigation header), used only with
          ``config.bds_klobuchar='icd'``; then without them (and without BDSA/BDSB in the ``nav_path``
          header) BeiDou takes the GPS Klobuchar coefficients, with a warning. The default 'gps' always
          takes the GPS coefficients scaled to the BeiDou signal
        - antenna: :class:`gnx_py.antex.AntennaModel` of the session (receiver and satellite PCO + PCV,
          built once by ``antex.build_antenna_model``). When given it replaces ``sat_pco``/``rec_pco``;
          without it the legacy dicts ({(PRN, freq): xyz mm}, {(antenna, radome, freq): neu mm}) are used,
          PCO only. Satellites without a valid entry follow ``config.missing_sat_antenna`` ('exclude'
          with one warning, default; 'raise': ValueError)
        - iono: the broadcast ionosphere table of the navigation file (``OrbitData.iono``, as
          ``rinex_nav`` reads it). With ``config.bds_klobuchar='icd'`` BeiDou takes per row the BDS Klobuchar set
          in force at the epoch (``rinex_nav.broadcast_iono_at``: RINEX 4 '> ION' records, GEO first, the latest
          at or before the epoch; RINEX 3 one GEO-first header set) before ``bdsa``/``bdsb`` and the header
        - geometry_free: True for the TEC session (geometry-free combinations): the satellite antenna term of
          broadcast orbits is kept per signal (full PCO and, since v1.0.0, PCV), so K (PCO_1 - PCO_2)
          . u and the PCV difference enter STEC as with precise orbits and STEC does not
          depend on the orbit product
        - run_notes: the :class:`RunNotes` of the session run, shared by its instances (one warning per run for a
          message about the whole run; the ionosphere report per system). None: this instance is a run of its own
        - phase_shift_dict: SYS / PHASE SHIFT values of the header as {'G L2L': cycles} (``io.parse_phase_shift``).
          They are corrections the file's phases already carry (RINEX 3.03 section 9.1), so they are not added
          (since v1.0.0; before it they were added to every satellite of the system, a double
          correction; a non-zero value now gives one warning per run). The sessions pass None
        - phase_shift_attrs: SYS / PHASE SHIFT of the observation reader (``{'phase_shift': ...,
          'phase_shift_events': ...}``, :func:`observation_phase_shift_attrs`; the contract form or the reader's
          stored form); None: ``df.attrs`` when it holds them. Records inside the data that change the
          correction reported for a processed phase are reported; the arcs are kept (:meth:`apply_phase_shift`)
        """
        self.geometry_free = bool(geometry_free)
        self.df = df
        self.flh = flh
        self.xyz = xyz
        self.configuration = config if config is not None else DefaultConfig()

        self.doy = self._day_of_year()
        self.SYS = system
        if self.SYS == 'G':
            self.mode = self.configuration.gps_freq
        elif self.SYS == 'E':
            self.mode = self.configuration.gal_freq
        elif self.SYS == 'C':
            self.mode = getattr(self.configuration, 'bds_freq', 'B1IB3I')
        else:
            raise ValueError(f"Unsupported GNSS system: {self.SYS}")
        self.iono_source = self.configuration.ionosphere_model
        self.antenna_height = antenna_h
        self.phase_shift = phase_shift_dict
        self.phase_shift_attrs = phase_shift_attrs
        #: in-data SYS / PHASE SHIFT changes of this system's processed phases (:func:`phase_alignment_blocks`)
        self.phase_shift_changes: list = []
        self.pco = sat_pco
        self.rec_pco = rec_pco
        self.antenna = antenna
        self._antenna_legacy: Optional[AntennaModel] = None
        self._missing_checked = False
        # Sun per epoch and satellite attitude per (time, sv), shared by the PCO, PCV, wind-up and turn guard
        self._sun_cache: Optional[pd.DataFrame] = None
        self._att_cache: Optional[pd.DataFrame] = None
        #: satellite attitude of the run: mode, rows per source, satellites without velocity, turn guard counts
        self.attitude_report: dict = {}
        #: broadcast ionosphere of the run (ionosphere_model='bdgim', :meth:`compute_bdgim`): sets used, ages,
        #: clamped rows, rejected records in the window
        self.iono_report: dict = {}
        self.run_notes = run_notes if run_notes is not None else RunNotes()

        self.F = {signal: frequency_hz(signal) for signal in {
            'L1', 'L2', 'L5', 'E1', 'E5a', 'E5b', 'B1I', 'B1C', 'B2a', 'B2I', 'B2b', 'B3I'
        }}

        self.l = {signal: wavelength_m(signal) for signal in self.F}

        # Wartości pomocnicze (jeśli model wymaga ich użycia)
        self.gpsa = gpsa
        self.gpsb = gpsb
        self.gala = gala
        self.bdsa = bdsa
        self.bdsb = bdsb
        self.iono = iono

        self.gim_file = self.configuration.gim_path

    # ------------------------------------------------------------------ antenna model
    def _antenna_model(self) -> AntennaModel:
        """The session's :class:`AntennaModel`; without one, a PCO-only model from the legacy dicts."""
        if self.antenna is not None:
            return self.antenna
        if self._antenna_legacy is None:
            t0 = None
            if isinstance(self.df, pd.DataFrame) and "time" in (self.df.index.names or []) and len(self.df):
                t0 = pd.Timestamp(self.df.index.get_level_values("time").min()).to_pydatetime()
            self._antenna_legacy = AntennaModel.from_legacy(self.pco, self.rec_pco, epoch0=t0,
                                                            missing_policy=self._missing_policy())
        return self._antenna_legacy

    def _missing_policy(self) -> str:
        """config.missing_sat_antenna, else the policy the antenna model was built with, else 'exclude'."""
        policy = (getattr(self.configuration, "missing_sat_antenna", None)
                  or getattr(self.antenna, "missing_policy", None) or "exclude")
        if policy not in ("exclude", "raise"):
            raise ValueError(f"missing_sat_antenna must be 'exclude' or 'raise', got {policy!r}")
        return policy

    def _broadcast_orbits(self) -> bool:
        return getattr(self.configuration, "orbit_type", "precise") == "broadcast"

    def _broadcast_no_satellite_term(self) -> bool:
        """Broadcast orbits: the satellite antenna term is 0 for every system and mode.

        The broadcast orbit and clock refer to the APC of the message's clock-reference signal (GPS IF(L1,L2);
        Galileo the APC of its ground segment; BeiDou B3I, :func:`gnx_py.antex.broadcast_apc_reference`) and the
        broadcast group delays (TGD/BGD/ISC, BeiDou TGD relative to B3I) map it to the user's signal, so they
        already hold the (nearly nadir-independent) PCO difference of the signal. RTKLIB (satposs with
        EPHOPT_BRDC) and gLAB apply no satellite PCO then; PCO(signal) - PCO(IF) on top double-counted it
        (SPP G L1 BRUX 2024-035: mean residual of GPS IIIA +0.507 m vs -0.123 m for the other satellites,
        +0.089/-0.045 m without the term). BeiDou took the full signal PCO before v1.0.0 (reference open); without it C-BRDC-kin-IF BRUX 2024-035 6 h p95 2.210 -> 1.668 m, NKLG 2.577 -> 0.932 m, static
        IF 0.729 -> 0.657 m (PCO(signal) - PCO(B3I) gives 1.658 / 0.914 / 0.711 m, no better than none).
        Not for a geometry-free (TEC) session: there the per-signal PCO difference is a property of the satellite
        antenna, independent of the orbit product (STEC broadcast = precise, plan 4.2), and K (PCO_1 -
        PCO_2) . u does not depend on the reference point (measured: <= 1.6e-8 TECU).
        """
        return self._broadcast_orbits() and not getattr(self, "geometry_free", False)

    def _sis_broadcast_reference(self):
        """APC a broadcast orbit refers to, for moving a precise orbit there (SIS), as
        :func:`gnx_py.antex.broadcast_apc_reference`: GPS IF(L1, L2) (IS-GPS-200; SIS 2024-035 BRDC vs COD,
        IGS20_2388: GPS IIIA mean dR +0.015 m like the other blocks for L1, L5 and L1L5, against
        -0.747/-1.227/-0.142 m at the signal APC) and BeiDou B3I (2024-035/131/132,
        COD and GRG: BeiDou-3 mean dR -0.085..+0.046 m at B3I, R_rms MEO median 0.717 -> 0.077 m against the
        APC of the mode). None for Galileo: the precise orbit stays at the APC of the mode. Galileo broadcast orbits
        do not refer to the IF APC of the ANTEX frequencies (FOC mean dR +0.17..+0.25 m at the IF of the message
        pair against -0.07..+0.10 m at E1, E5a and E5b; the zero lies near the mean of the frequency PCOs)."""
        if self.SYS not in ("G", "C"):
            return None
        return broadcast_apc_reference(self.SYS, self.mode)

    def _satellite_frequencies_needed(self, apc: str = "mode") -> list:
        """ANTEX frequencies a satellite must have for the satellite antenna step.

        ``apc='mode'``: the mode's signals, none with broadcast orbits outside a geometry-free session (no
        satellite term); ``apc='broadcast'`` (precise orbit moved to the broadcast APC, SIS): the reference's
        frequencies, G01/G02 for GPS and C06 (B3I) for BeiDou (the comparison needs only them; the reported
        ``dR_apc_signal`` is NaN for a satellite without an entry for the mode), the mode's frequencies for
        Galileo (:meth:`_sis_broadcast_reference`).
        """
        mode_freqs = [signal_spec(s).antex_frequency for s in mode_signals(self.mode)]
        if apc == "broadcast":
            ref = self._sis_broadcast_reference()
            if ref is None:
                return mode_freqs
            return list(dict.fromkeys(ref[0]))
        if self._broadcast_no_satellite_term():
            return []
        return mode_freqs

    def apply_prn_change_rule(self) -> None:
        """PRN-change rule: the rows of a PRN in the session antenna model's ``prn_changes`` (its
        satellite has no ANTEX entry from an epoch within the session on, e.g. the PRN reassigned to another
        satellite) after that epoch are dropped, in every session that built the model (``atx_path`` with
        ``sat_pco`` or ``rec_pco``), whether or not the satellite antenna term is applied: the broadcast record
        selection and the receiver-side models would otherwise mix the two satellites. No-op without a model."""
        if self.antenna is not None and isinstance(self.df, pd.DataFrame) and len(self.df):
            self.df = self.antenna.drop_after_prn_change(self.df)

    def apply_missing_satellite_rule(self, need: Optional[list] = None) -> list:
        """Satellites without a valid ANTEX entry for every frequency in ``need`` (default: what the
        satellite term of the mode needs, :meth:`_satellite_frequencies_needed`; nothing for broadcast
        orbits outside a geometry-free session): 'exclude' drops their rows with one warning naming the PRNs,
        the ANTEX and the date; 'raise' raises ValueError. When no satellite of the frame has an entry (an
        ANTEX without satellite blocks, e.g. a station calibration file, or an empty legacy table) ValueError
        is raised under either policy. Returns the PRNs (once per instance). G05 taken from G01 (frequency-independent
        entries of satellites that transmit L5) is reported in one warning as well."""
        if self._missing_checked or not isinstance(self.df, pd.DataFrame) or self.df.empty:
            return []
        self._missing_checked = True
        # a PRN whose satellite has no ANTEX entry from an epoch within the session on (reassigned to another
        # satellite; warned when the model was built, decision D9 of bug-fix stage 13): its rows after that epoch
        # are dropped whatever the satellite term needs (also done by run(), apply_prn_change_rule)
        self.apply_prn_change_rule()
        if need is None:
            need = self._satellite_frequencies_needed()
        if not need:
            return []
        model = self._antenna_model()
        sv_level = self.df.index.get_level_values("sv")
        prns = sorted({str(s) for s in sv_level.unique()})
        missing = missing_satellite_antennas(model, prns, need)
        when = (f"{model.epoch0:%Y-%m-%d}" if model.epoch0 is not None
                else f"{pd.Timestamp(self.df.index.get_level_values('time').min()):%Y-%m-%d}")
        if "G05" in need:
            sub = [p for p in prns if p not in missing and _transmits_l5(model.satellite(p))
                   and not model.satellite(p).has("G05")]
            if sub:
                warnings.warn(
                    f"{len(sub)} GPS satellites have no G05 calibration in {model.atx_name}: the "
                    f"frequency-independent G01 values are used for L5 ({', '.join(sub)})", UserWarning, stacklevel=3)
        if not missing:
            return []
        name = SYSTEM_NAMES.get(self.SYS, self.SYS)
        msg = (f"{name} satellites without a valid ANTEX entry for {', '.join(need)} in {model.atx_name} on "
               f"{when}: {', '.join(missing)}")
        if len(missing) == len(prns):
            raise ValueError(
                f"no {name} satellite has a valid ANTEX entry for {', '.join(need)} in {model.atx_name} on "
                f"{when} ({', '.join(missing)}): use an IGS ANTEX that covers the session, or switch the "
                f"satellite antenna correction off (sat_pco=False)")
        if self._missing_policy() == "raise":
            raise ValueError(msg + " (missing_sat_antenna='raise'; use an ANTEX that covers the session, or "
                                   "missing_sat_antenna='exclude')")
        warnings.warn(msg + "; their observations are excluded (missing_sat_antenna='exclude')", UserWarning,
                      stacklevel=3)
        self.df = self.df[~sv_level.isin(missing)].copy()
        return missing

    def _sun_rows(self, times) -> np.ndarray:
        """Sun ECEF [m] for every row time (one ephemeris evaluation per unique epoch, cached per instance and
        shared by the satellite PCO, the attitude and the wind-up)."""
        times = pd.DatetimeIndex(times)
        if len(times) == 0:
            return np.zeros((0, 3))
        uniq = times.unique()
        cache = self._sun_cache
        new = uniq if cache is None else uniq[~uniq.isin(cache.index)]
        if len(new):
            sun = get_sun_ecef(list(new))
            xyz = np.column_stack([np.asarray([v[0] if isinstance(v, (list, np.ndarray)) else v for v in sun[c]],
                                              dtype=float) for c in ("x", "y", "z")])
            part = pd.DataFrame(xyz, index=pd.DatetimeIndex(new), columns=["x", "y", "z"])
            cache = part if cache is None else pd.concat([cache, part])
            self._sun_cache = cache
        return cache[["x", "y", "z"]].to_numpy(dtype=float)[cache.index.get_indexer(times)]

    # ------------------------------------------------------------------ satellite attitude (bug-fix stage 11c)
    #: largest spacing [s] of a satellite's rows over which the attitude takes a finite-difference velocity of the
    #: positions when the frame has no vx/vy/vz (synthetic frames)
    ATTITUDE_GRADIENT_MAX_DT_S = 300.0

    def _attitude_mode(self) -> str:
        """``config.satellite_attitude``: 'model' (also for None, the default) or 'nominal'."""
        mode = getattr(self.configuration, "satellite_attitude", None)
        return "model" if mode is None else mode

    def _first_epoch(self):
        if isinstance(self.df, pd.DataFrame) and "time" in (self.df.index.names or []) and len(self.df):
            return pd.Timestamp(self.df.index.get_level_values("time").min()).to_pydatetime()
        return None

    def _satellite_blocks(self, svs) -> dict:
        """{PRN: ANTEX entry or None}. The entry of the session's antenna model (the one its PCO uses, valid at the
        first epoch), else the entry of ``atx_path`` valid at the first epoch, else None."""
        out = {}
        model, atx = self.antenna, None
        if model is None and getattr(self.configuration, "atx_path", None):
            from ..models.antex import load_antex

            try:
                atx = load_antex(self.configuration.atx_path)
            except (OSError, ValueError):
                atx = None
        epoch0 = self._first_epoch()
        for sv in svs:
            e = model.satellite(sv) if model is not None else (atx.satellite(sv, epoch0) if atx is not None else None)
            # a model made from the legacy PCO dicts has no block names
            out[sv] = e if e is not None and e.type.strip().upper() != "LEGACY" else None
        return out

    @staticmethod
    def _is_geo(sv: str, entry) -> bool:
        """BeiDou GEO by the ANTEX block (BEIDOU-2G, BEIDOU-3G-*), else by the PRN (``gnss.is_bds_geo``)."""
        if entry is not None and str(sv).startswith("C"):
            return entry.type.strip().upper().startswith(("BEIDOU-2G", "BEIDOU-3G"))
        return is_bds_geo(sv)

    def _attitude_velocity(self, df: pd.DataFrame, svs: np.ndarray, t_s: np.ndarray, r: np.ndarray) -> np.ndarray:
        """ECEF velocity [m/s] per row: vx/vy/vz of the orbit interpolation (precise: Lagrange, broadcast: the
        ephemeris, SIS: Chebyshev), else the second-order finite difference of a satellite's positions over at
        least three consecutive rows at most ATTITUDE_GRADIENT_MAX_DT_S apart (central inside, one-sided second
        order at the ends: about 0.04 deg of the orbit normal at 300 s); NaN where neither exists, also on runs of
        one or two rows (a first-order difference would tilt the orbit normal by the ECEF Coriolis acceleration,
        up to about 1 deg at 300 s)."""
        v = np.full_like(r, np.nan)
        if {"vx", "vy", "vz"} <= set(df.columns):
            v = df[["vx", "vy", "vz"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(v).all(axis=1)
        for sv in pd.unique(svs[bad]):
            rows = np.flatnonzero(svs == sv)
            rows = rows[np.argsort(t_s[rows], kind="stable")]
            tt = t_s[rows]
            cut = np.flatnonzero((np.diff(tt) > self.ATTITUDE_GRADIENT_MAX_DT_S) | (np.diff(tt) <= 0)) + 1
            for seg in np.split(np.arange(rows.size), cut):
                if seg.size < 3 or not np.isfinite(r[rows[seg]]).all():
                    continue
                g = np.gradient(r[rows[seg]], tt[seg], axis=0, edge_order=2)
                fill = bad[rows[seg]]
                v[rows[seg][fill]] = g[fill]
        return v

    @staticmethod
    def _yaw_of_axes(r: np.ndarray, v: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """Yaw [deg] of the body x axis in the orbit frame, x_b = cos(yaw) e_T - sin(yaw) e_N (IGS convention,
        yaw 0 = orbit-normal attitude; ``gnx_py.attitude.orbit_frame`` and ``yaw_of_axes``)."""
        from ..models import attitude

        _, e_t, e_n = attitude.orbit_frame(r, v)
        return np.degrees(attitude.yaw_of_axes(x_b, e_t, e_n))

    def _turn_guard_mode(self) -> str:
        """``attitude_turn_guard`` of a PPP configuration ('auto', 'all', 'off'); 'off' for every other session
        (SPP, TEC and SIS configurations have no such field)."""
        mode = getattr(self.configuration, "attitude_turn_guard", None)
        return "off" if mode is None else mode

    def _static_receiver(self) -> bool:
        """True for static PPP (effective q_crd == 0)."""
        cfg = self.configuration
        eff = getattr(cfg, "effective", None)
        q = eff().get("q_crd", 0.0) if callable(eff) else getattr(cfg, "q_crd", 0.0)
        try:
            return float(q) == 0.0
        except (TypeError, ValueError):
            return True

    def _attitude(self, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Satellite body axes per row of the frame, one provider for the satellite PCO (sat_pco_los,
        sat_fixed_system), the satellite PCV azimuth and the wind-up.

        Columns x1..x3, y1..y3, z1..z3 (IGS body axes in ECEF), yaw_deg, regime, source and guard, one row per
        row of ``frame`` (default ``self.df``), indexed (time, sv). Computed once per instance: rows looked up
        again (after rows were dropped or the index reordered) come from the cached result.

        ``config.satellite_attitude``:
        - 'nominal': yaw steering everywhere (RTKLIB satantoff / windupcorr, as GNX before v1.0.0), BeiDou
          GEO orbit-normal for the PCO and the wind-up alike;
        - 'model' (None, the default): ORBEX attitude of the product's analysis centre (``attitude_path``) where
          it covers a satellite's arc, else the published attitude law of the satellite block (``gnx_py.attitude``;
          block from the ANTEX entry). Rows without a velocity get the nominal frame with one UserWarning.
        ``guard``: rows of the turn guard (``attitude_turn_guard``, PPP only), dropped at the end of ``run``.
        """
        df = self.df if frame is None else frame
        key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(df.index.get_level_values("time")),
                                         df.index.get_level_values("sv").astype(str)], names=["time", "sv"])
        if self._att_cache is not None:
            hit = self._att_cache.reindex(key)
            if len(hit) == 0 or hit["regime"].notna().all():
                return hit
        times = pd.DatetimeIndex(key.get_level_values("time"))
        svs = key.get_level_values("sv").to_numpy()
        n = len(df)
        r = df[["xe", "ye", "ze"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sun = self._sun_rows(times)
        t_s = np.asarray((times - pd.Timestamp("1980-01-06")).total_seconds(), dtype=float)  # GPS seconds
        v = self._attitude_velocity(df, svs, t_s, r)
        blocks = self._satellite_blocks(pd.unique(svs))
        mode = self._attitude_mode()

        # nominal frame: yaw steering (as before stage 11c, bit-identical), GEO orbit-normal
        x, y, z = _satellite_body_frame(r, sun) if n else (np.zeros((0, 3)),) * 3
        x, y, z = x.copy(), y.copy(), z.copy()
        regime = np.zeros(n, dtype=np.int8)
        source = np.zeros(n, dtype=np.int8)
        guard = np.zeros(n, dtype=bool)
        yaw = np.full(n, np.nan)
        for sv in pd.unique(svs):
            if not self._is_geo(sv, blocks[sv]):
                continue
            rows = np.flatnonzero(svs == sv)
            rows = rows[np.argsort(t_s[rows], kind="stable")]
            # rows without a velocity take the finite difference over the satellite's rows
            ex, ey = orbit_normal_frame(times[rows], r[rows], velocities=v[rows])
            x[rows], y[rows], z[rows] = ex, ey, np.cross(ex, ey)
            regime[rows] = 7

        report = {"mode": mode, "rows": int(n), "source": {}, "no_velocity": [], "guard": "off"}
        if mode == "nominal":
            _warn_attitude_path_unused(self.configuration, stacklevel=4)
        if mode == "model" and n:
            self._model_attitude(svs, t_s, r, v, sun, blocks, x, y, z, yaw, regime, source, guard, report)
        has_v = np.isfinite(v).all(axis=1)
        nominal = has_v & np.isnan(yaw)
        if nominal.any():
            yaw[nominal] = self._yaw_of_axes(r[nominal], v[nominal], x[nominal])
        report["source"] = {name: int((source == code).sum()) for code, name in ((0, "nominal"), (1, "law"),
                                                                              (2, "orbex")) if (source == code).any()}
        self.attitude_report = report
        out = pd.DataFrame(np.column_stack([x, y, z]), index=key,
                           columns=[f"{a}{c}" for a in "xyz" for c in (1, 2, 3)])
        out["yaw_deg"] = yaw
        out["regime"] = regime
        out["source"] = source
        out["guard"] = guard
        self._att_cache = out if self._att_cache is None else pd.concat(
            [self._att_cache, out[~out.index.isin(self._att_cache.index)]])
        return out

    def _model_attitude(self, svs, t_s, r, v, sun, blocks, x, y, z, yaw, regime, source, guard, report) -> None:
        """'model' attitude in place: ``gnx_py.attitude.satellite_attitude`` for all rows; rows without a velocity
        keep the nominal frame (one UserWarning)."""
        cfg = self.configuration
        explicit = getattr(cfg, "satellite_attitude", None) == "model"
        if all(b is None for b in blocks.values()):
            msg = (f"satellite_attitude='model' needs the satellite blocks of an ANTEX file (atx_path): {self.SYS} "
                   f"uses the nominal attitude (yaw steering, BeiDou GEO orbit-normal)")
            if explicit:
                _warn_once(cfg, ("attitude_no_atx", self.SYS), msg, UserWarning, stacklevel=4)
            else:
                _LOG.info(msg)
        att = _attitude_module()
        epoch0 = self._first_epoch()
        families = {}
        for sv, e in blocks.items():
            if e is not None:
                families[sv] = att.law_family(e.type.strip(), e.svn.strip(), sv, epoch0)
            else:
                families[sv] = "bds_geo" if is_bds_geo(sv) else "unknown"
        guard_mode = self._turn_guard_mode()
        report["guard"] = guard_mode
        orbex = orbex_for_config(cfg, stacklevel=5)
        product_ac = _product_ac(cfg)
        broadcast = self._broadcast_orbits()
        static = self._static_receiver()
        # one signal of this system: single-frequency PPP (route 'single' or an uncombined single-signal mode)
        single_frequency = len(mode_signals(self.mode)) == 1 if getattr(self, "mode", None) else False
        has_v = np.isfinite(v).all(axis=1) & np.isfinite(r).all(axis=1)
        missing = sorted(set(svs[~has_v]))
        if missing:
            report["no_velocity"] = missing
            _warn_once(cfg, ("attitude_no_velocity", self.SYS, tuple(missing)),
                       f"satellite attitude: {int((~has_v).sum())} {self.SYS} rows have no satellite velocity "
                       f"(no vx/vy/vz and fewer than three consecutive rows at most "
                       f"{self.ATTITUDE_GRADIENT_MAX_DT_S:.0f} s apart): "
                       f"nominal attitude there ({', '.join(missing)})", UserWarning, stacklevel=4)
        # one call for all rows (closed form per row; ORBEX per satellite and continuous arc); rows without a
        # velocity come back with the nominal frame (regime 9)
        a = att.satellite_attitude(svs, t_s, r, v, sun, families, mode="model",
                                   laws=getattr(cfg, "attitude_laws", None), orbex=orbex, guard=guard_mode,
                                   product_ac=product_ac, static=static, broadcast=broadcast,
                                   single_frequency=single_frequency)
        x[:], y[:], z[:] = a.x, a.y, a.z
        yaw[:] = a.yaw_deg
        regime[:] = a.regime
        source[:] = a.source
        guard[:] = a.guard

    def _sat_pco_xyz(self, model: AntennaModel, svs: np.ndarray, freq: str) -> np.ndarray:
        """Body-frame PCO [m] per row for ANTEX frequency ``freq`` (NaN where no entry)."""
        out = np.full((len(svs), 3), np.nan)
        for s in pd.unique(svs):
            e = model.satellite(s)
            f = satellite_frequency(e, freq, warn=False) if e is not None else None
            if f is not None:
                out[svs == s] = e.pco_xyz_m(f)
        return out

    def _sat_pcv(self, model: AntennaModel, svs: np.ndarray, freq: str, nadir_deg: np.ndarray,
                 az_deg: Optional[np.ndarray] = None) -> np.ndarray:
        """Satellite PCV [m] per row: at the nadir angle and, for an entry with an azimuth grid (DAZI > 0, e.g.
        the IGS20 Galileo IOV/FOC entries), the body-frame azimuth ``az_deg`` (:func:`_satellite_azimuth_deg`); NOAZI otherwise (RTKLIB satantpcv). 0 without PCV."""
        out = np.zeros(len(svs))
        if not model.pcv:
            return out
        for s in pd.unique(svs):
            e = model.satellite(s)
            f = satellite_frequency(e, freq, warn=False) if e is not None else None
            if f is not None:
                m = svs == s
                out[m] = e.pcv_m(f, nadir_deg[m], None if az_deg is None else az_deg[m])
        return out

    def _sat_pco_for_mode(self, model: AntennaModel, svs: np.ndarray) -> np.ndarray:
        """PCO [m] of the mode per row: the signal's, or the IF combination of a dual-frequency mode."""
        signals = mode_signals(self.mode)
        freqs = [signal_spec(s).antex_frequency for s in signals]
        if len(signals) == 1:
            return self._sat_pco_xyz(model, svs, freqs[0])
        k1, k2 = mode_ionosphere_free_coefficients(self.mode)
        return k1 * self._sat_pco_xyz(model, svs, freqs[0]) - k2 * self._sat_pco_xyz(model, svs, freqs[1])

    def sat_fixed_system(self, apc: str = "mode"):
        """Satellite CoM -> APC in the coordinates ('crd'): x_apc/y_apc/z_apc = xe/ye/ze + R PCO.

        ``apc='mode'``: PCO of the mode's signal (or its IF combination) by the ``gnss.SIGNALS`` ANTEX
        frequency, GPS G05 from G01 for frequency-independent entries. A frame of broadcast orbits
        (``config.orbit_type='broadcast'``, any system) is already at the broadcast APC: shift 0.

        ``apc='broadcast'`` (SIS: a precise orbit compared with a broadcast one): the APC broadcast orbits
        refer to whatever the user's signal, GPS IF(L1,L2) and BeiDou B3I, and
        the column ``dR_apc_signal`` holds the radial (outward) component of APC(mode) minus APC(reference) [m]
        (NaN without an entry for the mode), reported, not compared: a user maps the reference to the signal
        through the broadcast group delays. Galileo: the mode's APC as with ``'mode'``
        (:meth:`_sis_broadcast_reference`).

        Satellites without an entry for the frequencies used follow the missing-satellite rule. The result
        keeps the Sun position (xs, ys, zs), the body axes (i, j, k: x, y, z of :meth:`_attitude`, the satellite
        attitude of ``config.satellite_attitude``; BeiDou GEO orbit-normal), the PCO and the attitude columns
        att_yaw_deg, att_regime and att_source (reporting only).
        """
        if apc not in ("mode", "broadcast"):
            raise ValueError(f"apc must be 'mode' or 'broadcast', got {apc!r}")
        self.apply_missing_satellite_rule(self._satellite_frequencies_needed(apc))
        model = self._antenna_model()
        ps = self.df.copy()
        sun = self._sun_rows(ps.index.get_level_values('time'))
        ps['xs'], ps['ys'], ps['zs'] = sun[:, 0], sun[:, 1], sun[:, 2]
        numeric_cols = ['xe', 'ye', 'ze', 'xs', 'ys', 'zs']
        ps[numeric_cols] = ps[numeric_cols].apply(pd.to_numeric, errors='coerce')
        ps = ps.dropna(subset=numeric_cols)
        xyz = ps[['xe', 'ye', 'ze']].to_numpy(dtype=float)
        svs = ps.index.get_level_values('sv').astype(str).to_numpy()
        ref = self._sis_broadcast_reference() if apc == "broadcast" else None
        if len(ps):
            att = self._attitude(ps)
            i, j, k = (att[[f'{a}1', f'{a}2', f'{a}3']].to_numpy(dtype=float) for a in 'xyz')
            ps['att_yaw_deg'] = att['yaw_deg'].to_numpy()
            ps['att_regime'] = att['regime'].to_numpy()
            ps['att_source'] = att['source'].to_numpy()
            e = ps[['xs', 'ys', 'zs']].to_numpy(dtype=float) - xyz
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            if ref is not None:
                (f1, f2), (a, b) = ref
                pco = a * self._sat_pco_xyz(model, svs, f1) - b * self._sat_pco_xyz(model, svs, f2)
            elif apc == "mode" and self._broadcast_no_satellite_term():
                pco = np.zeros((len(ps), 3))
            else:
                pco = self._sat_pco_for_mode(model, svs)
            pco = np.nan_to_num(pco)
        else:
            i = j = k = e = pco = np.zeros((0, 3))
        for name, arr in (('pco', pco), ('k', k), ('e', e), ('i', i), ('j', j)):
            for c in range(3):
                ps[f'{name}{c + 1}'] = arr[:, c]
        delta = pco[:, [0]] * i + pco[:, [1]] * j + pco[:, [2]] * k
        ps[['x_apc', 'y_apc', 'z_apc']] = xyz + delta
        if ref is not None:
            d = self._sat_pco_for_mode(model, svs) - pco if len(ps) else np.zeros((0, 3))
            d_xyz = d[:, [0]] * i + d[:, [1]] * j + d[:, [2]] * k
            ps['dR_apc_signal'] = np.einsum('ij,ij->i', d_xyz, -k) if len(ps) else np.zeros(0)
        if list(ps.index.names) != ['time', 'sv']:
            ps = ps.reset_index().set_index(['time', 'sv'])
        self.df = ps.copy()

    def sat_pco_los(self):
        """``sat_pco_los_<signal>`` [m]: satellite antenna term on the line of sight, per signal of the mode.

        Value = (R PCO_sig) . u_sr - PCV_sat(nadir, azimuth), u_sr the unit vector satellite -> receiver, so
        the range model is rho(CoM) - sat_pco_los (routes: obs + sat_pco_los). R holds the body axes of
        :meth:`_attitude` (``config.satellite_attitude``). Precise orbits (CoM): full PCO and PCV, by azimuth
        for entries with an azimuth grid. Broadcast orbits, every system: 0 for every
        signal (the orbit is at the APC of the message's clock-reference signal, GPS IF(L1,L2), BeiDou B3I, and
        the broadcast group delays map it to the signal; RTKLIB/gLAB apply no satellite term,
        :meth:`_broadcast_no_satellite_term`; broadcast BeiDou kept the full signal PCO before v1.0.0). The geometry-free (TEC) session keeps PCO and PCV with broadcast orbits too (S9-D20: STEC
        does not depend on the orbit product).
        """
        self.apply_missing_satellite_rule()
        df = self.df
        signals = mode_signals(self.mode)
        if len(df) == 0 or self._broadcast_no_satellite_term():
            for sig in signals:
                self.df[f'sat_pco_los_{sig}'] = 0.0
            return
        model = self._antenna_model()
        times = df.index.get_level_values('time')
        xyz = df[['xe', 'ye', 'ze']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        att = self._attitude()
        i, j, k = (att[[f'{a}1', f'{a}2', f'{a}3']].to_numpy(dtype=float) for a in 'xyz')
        r = np.asarray(self.xyz, dtype=float).reshape(3)
        u_sr = r - xyz
        u_sr = u_sr / np.linalg.norm(u_sr, axis=1, keepdims=True)
        # nadir angle at the satellite between k (to the Earth centre) and the receiver (RTKLIB satantpcv)
        nadir = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', u_sr, k), -1.0, 1.0)))
        az = _satellite_azimuth_deg(u_sr, i, j)
        svs = df.index.get_level_values('sv').astype(str).to_numpy()
        # PCV with precise orbits; with broadcast orbits only in the geometry-free (TEC) session, where the
        # satellite term enters STEC whatever the orbit product (S9-D20; the azimuth-dependent Galileo PCV of
        # stage 11c moved E5aE5b STEC broadcast vs precise by 0.013 TECU median without it). Stage 11c extends
        # S9-D20, which read "full PCO, no PCV" for broadcast orbits, to the PCV (confirmed by the author on 2026-10-12)
        apply_pcv = not self._broadcast_orbits() or self.geometry_free
        for sig in signals:
            freq = signal_spec(sig).antex_frequency
            p = np.nan_to_num(self._sat_pco_xyz(model, svs, freq))
            delta = p[:, [0]] * i + p[:, [1]] * j + p[:, [2]] * k
            val = np.einsum('ij,ij->i', delta, u_sr)
            if apply_pcv:
                val = val - self._sat_pcv(model, svs, freq, nadir, az)
            self.df[f'sat_pco_los_{sig}'] = val


    def phase2meters(self):
        self.df= self.df.copy()
        for signal in mode_signals(self.mode):
            spec = signal_spec(signal)
            cols = [col for col in self.df.columns
                    if col.startswith(spec.phase_prefix) and not col.endswith(('lli', 'ssi'))]
            if cols:
                self.df.loc[:, cols] = self.df[cols].to_numpy(copy=False) * wavelength_m(signal)


    def apply_phase_shift(self):
        """SYS / PHASE SHIFT of the observation file (RINEX 3.01+).

        The phases are used as written: in a RINEX 3.01+ file every phase is already aligned to the reference
        signal of its band and SYS / PHASE SHIFT reports the correction applied, phi_RINEX = phi_original + dphi
        (RINEX 3.03 section 9.1, Table 25; RTKLIB's RINEX writer adds the shift to the phase and writes the same
        value in the header, rinex.c outrnxobsb of RTKLIB-EX 2.5.1). Before v1.0.0 GNX added the header
        values a second time: the PPP and TEC sessions passed ``ObsData.meta[-1]``, the header dictionary of
        ``io.parse_phase_shift``, and every non-zero value (e.g. GPS L2L / L5Q, Galileo E5a / E5b -0.25 cycles of
        many Septentrio / Trimble files) shifted the phase of every satellite of the system, against the satellite
        lists of the record: a constant per arc for float PPP and TEC, a quarter or half cycle on the shifted phases
        against IGS OSB phase biases for PPP-AR. Removed; a dictionary passed directly gives one warning.

        A SYS / PHASE SHIFT record inside the data (``phase_shift_events``, an event-flag header block) that
        changes the correction in force for a processed phase of a satellite is found
        (:func:`phase_alignment_blocks`, ``phase_shift_changes``) and reported with one UserWarning per run and
        system (the first instance of the run names its satellites); the phases stay as written and the arcs are
        kept: the file's phases are aligned on both sides of the record, a changed correction compensates a
        change of the original phase (RINEX 3.03 section 9.1; WUH200CHN 2024-141: geometry-free step 0.2 cm
        median at 28 measurable changes against 6.2-6.4 cm for a quarter cycle). RTKLIB ignores the record
        (rinex.c decode_obsh) and keeps the ambiguity as well.
        """
        cfg = self.configuration
        legacy = {k: v for k, v in (self.phase_shift or {}).items()
                  if str(k).startswith(f"{self.SYS} ") and v}
        if legacy:
            _notes.warn_once(
                cfg, ("phase_shift_header_values", self.SYS, tuple(sorted(legacy.items()))),
                f"SYS / PHASE SHIFT {', '.join(f'{k} {v:+g}' for k, v in sorted(legacy.items()))} cycles: corrections "
                f"already applied to the phases of the file (RINEX 3.03 section 9.1), not applied again (GNX-py "
                f"before v1.0.0 added them a second time)",
                UserWarning, stacklevel=3)
        info = self.phase_shift_attrs
        if info is None:
            info = getattr(self.df, "attrs", None) or {}
        events = info.get("phase_shift_events") if isinstance(info, dict) else None
        if not events or not isinstance(self.df, pd.DataFrame) or self.df.empty:
            return
        phase_cols = [c for c in self.df.columns
                      if any(c.startswith(signal_spec(sig).phase_prefix) for sig in mode_signals(self.mode))
                      and not c.endswith(('lli', 'ssi'))]
        _, changes = phase_alignment_blocks(self.df, self.SYS, phase_cols, info.get("phase_shift"), events)
        self.phase_shift_changes = changes
        if not changes:
            return
        sats = sorted({c[2] for c in changes})
        obs = sorted({c[1] for c in changes})
        epochs = sorted({c[0] for c in changes})
        _notes.warn_once(
            cfg, ("phase_shift_events", self.SYS),
            f"SYS / PHASE SHIFT records inside the observation data change the phase alignment of "
            f"{SYSTEM_NAMES.get(self.SYS, self.SYS)} {', '.join(obs)} for {len(sats)} satellite"
            f"{'s' if len(sats) != 1 else ''} ({', '.join(sats)}) at {len(epochs)} epoch"
            f"{'s' if len(epochs) != 1 else ''} (first {epochs[0]}): the phases are used as written and the arcs "
            f"are kept (RINEX 3.0x phases stay aligned to the reference signal; the record reports the correction "
            f"applied, not a phase step)", UserWarning, stacklevel=3)

    #: 1-sigma error of a broadcast ionosphere model (Klobuchar, NTCM) as a fraction of its delay;
    #: the value RTKLIB uses for the broadcast model (ERR_BRDCI = 0.5, RTKLIB 2.4.2 ppp.c:63 and pntpos.c:34, the
    #: PPP variance at ppp.c:562)
    BROADCAST_IONO_ERR_FRACTION = 0.5
    #: 'ion_rms' floor of every broadcast model [TECU at the zenith, times the model's mapping function]: BDGIM since
    #: v1.0.0 (0.8 m (2.7 ns) on L1, about half the 5 ns night term of the Klobuchar; share of BDGIM
    #: errors beyond 3 sigma 10.2 % -> 4.4 % on the validation data), Klobuchar and NTCM since v1.0.0 (the
    #: model error is correlated over hours, but the constraint enters every epoch; 50 % of a small night delay
    #: weighted it so much that the Klobuchar-constrained BRUX 2024-035 rows converged slower than without a
    #: constraint, GPS-only uncombined: convergence to 10 cm after 50.5 against 41.5 min unconstrained, 46.0 with the floor)
    BDGIM_SIGMA_FLOOR_TECU = 5.0
    BROADCAST_SIGMA_FLOOR_TECU = BDGIM_SIGMA_FLOOR_TECU
    #: a BDGIM set older than this [h] at an epoch gives one warning per run; it is used
    BDGIM_MAX_AGE_WARN_H = 6.0
    #: TECU -> metres of slant delay at frequency f: K / f**2 (GNX constant; the ICD's 40.28e16 stays in the model)
    IONO_K = 40.3e16

    def broadcast_iono_rms(self):
        """``ion_rms`` [m] of a model without RMS maps: hypot(50 % of the modelled slant delay, 5 TECU x the
        model's mapping function), the rule of BDGIM (:meth:`compute_bdgim`) since v1.0.0 (before: 50 % of the delay alone).

        The column has the meaning of the GIM path (1-sigma of the model delay of the first
        signal), so the PPP constraints with ``use_iono_rms=True`` stay soft instead of
        taking a zero sigma. The mapping function is the model's own: GPS Klobuchar its obliquity factor
        1 + 16 (0.53 - E)^3 (E in semicircles), BeiDou Klobuchar with ``bds_klobuchar='icd'`` the thin shell of
        BDS-SIS-ICD-B1I 3.0 5.2.4.7 (1 / sqrt(1 - (R cos E / (R + h))^2), R 6378 km, h 375 km), NTCM its MSLM
        (alpha 0.9782 at 450 km).
        """
        ion = np.abs(self.df['ion'].to_numpy(dtype=float))
        mf = self._broadcast_mapping()
        floor = self.IONO_K / frequency_hz(first_signal(self.mode)) ** 2 * self.BROADCAST_SIGMA_FLOOR_TECU * mf
        self.df['ion_rms'] = np.hypot(self.BROADCAST_IONO_ERR_FRACTION * ion, floor)

    def _broadcast_mapping(self) -> np.ndarray:
        """Mapping function of the active broadcast model at the rows' elevations (Klobuchar obliquity factor, or
        the NTCM-G MSLM)."""
        ev = self.df['ev'].to_numpy(dtype=float)
        if self.configuration.ionosphere_model == 'ntcm':
            return np.asarray(stec_mf(ev, no=1), dtype=float)
        if getattr(self, 'SYS', None) == 'C' and bds_klobuchar_model(self.configuration) == 'icd':
            # BDS-SIS-ICD-B1I 3.0 5.2.4.7: the BeiDou Klobuchar's own single-layer slant factor
            x = BDS_KLOBUCHAR_R_KM / (BDS_KLOBUCHAR_R_KM + BDS_KLOBUCHAR_H_KM) * np.cos(np.deg2rad(ev))
            return 1.0 / np.sqrt(1.0 - x * x)
        e = ev / 180.0
        return 1.0 + 16.0 * (0.53 - e) ** 3

    def compute_klobuchar(self):
        """Klobuchar 'ion' [m] at the first signal of the mode.

        GPS and Galileo: GPS coefficients (IS-GPS-200). BeiDou: by ``config.bds_klobuchar`` - 'gps' (default)
        the GPS coefficients scaled to the first signal; 'icd' (opt-in) the BDS-SIS-ICD-B1I 3.0 5.2.4.7 model
        with BDSA/BDSB (given, else from the ``nav_path`` header, else 'gps' with a warning).
        """
        self.df = self.df.copy()
        if self.SYS == 'C' and bds_klobuchar_model(self.configuration) == 'icd':
            t = self.df.index.get_level_values('time').to_numpy('datetime64[ns]')
            # per row: the BDS Klobuchar set in force at the epoch (broadcast ionosphere table, bug-fix stage 10)
            bdsa, bdsb = _bds_klobuchar_rows(self.iono, t)
            if bdsa is None and (self.bdsa is None or self.bdsb is None):
                # the sessions pass gpsa/gpsb/gala only: take BDSA/BDSB from the header of nav_path
                bdsa, bdsb = nav_header_bds_coefficients(getattr(self.configuration, 'nav_path', None))
            elif bdsa is None:
                bdsa, bdsb = self.bdsa, self.bdsb
            if bdsa is not None and bdsb is not None:
                # BDS-SIS-ICD-B1I 3.0 5.2.4.7: B1I delay from BDSA/BDSB, BDT = GPST - 14 s
                sod_gpst = (t - t.astype('datetime64[D]')).astype('timedelta64[ns]').astype(np.int64) / 1e9
                ion = klobuchar_bds(azimuth=self.df['az'].to_numpy(dtype=float),
                                    elev=self.df['ev'].to_numpy(dtype=float), fi=self.flh[0], lambda_=self.flh[1],
                                    sod_bdt=sod_gpst + BDT_MINUS_GPST_S, alfa=bdsa, beta=bdsb)
                self.df['ion'] = ion * gamma('B1I', first_signal(self.mode))
                return
            warnings.warn(
                "ionosphere_model='klobuchar' with bds_klobuchar='icd' for BeiDou: no BeiDou Klobuchar "
                "coefficients (BDSA/BDSB of the navigation header) were given; the GPS Klobuchar model "
                f"(GPSA/GPSB) scaled to {first_signal(self.mode)} is used", UserWarning, stacklevel=2)
        epochs = self.df.index.get_level_values('time').tolist()
        # per row the GPS set in force at the epoch (the broadcast parameters of the epoch, IS-GPS-200 20.3.3.5.2.5;
        # a RINEX 4 file carries one '> ION' record per broadcast set; bug-fix stage 13d, D8), else the session's
        # one set (gpsa/gpsb: RINEX 3 / 2 header)
        sets = _broadcast_iono_rows(self.iono, 'G', 'KLOB', np.asarray(epochs, dtype='datetime64[ns]'))
        if sets is None:
            self._require_broadcast_iono('klobuchar', 'GPSA/GPSB (RINEX 2: ION ALPHA/ION BETA)', self.gpsa,
                                         self.gpsb)
            alfa, beta = self.gpsa, self.gpsb
        else:
            alfa, beta = sets[:, :4].T, sets[:, 4:8].T

        toc_list = datetime2toc(epochs)
        az = self.df['az'].to_numpy()
        ev = self.df['ev'].to_numpy()

        ion = klobuchar_array(azimuth=az, elev=ev, fi=self.flh[0], lambda_=self.flh[1],
                              tow=np.asarray(toc_list, dtype=float), beta=beta, alfa=alfa)
        # the model gives the L1 group delay (IS-GPS-200 20.3.3.5.2.5: for L2 multiply by gamma);
        # 'ion' is the delay of the first signal of the mode, as for the GIM and NTCM paths
        self.df['ion'] = ion * gamma('L1', first_signal(self.mode))

    def enu2ecef(self, x):
        """
        Rotation matrix from ENU to ECEF for 3D matrixes (obviously)
        :param x: np.array([east, north, up]) vector
        :return:
        """
        flh = np.array([np.deg2rad(self.flh[0]),
                        np.deg2rad(self.flh[1])
                        ])
        rot = np.array([[-np.sin(flh[1]), -np.cos(flh[1]) * np.sin(flh[0]), np.cos(flh[1]) * np.cos(flh[0])],
                        [np.cos(flh[1]), -np.sin(flh[1]) * np.sin(flh[0]), np.sin(flh[1]) * np.cos(flh[0])],
                        [0, np.cos(flh[0]), np.sin(flh[0])]])
        return rot.dot(x)

    def compute_antenna_height(self):
        self.antenna_height = np.array([self.antenna_height[1], self.antenna_height[2], self.antenna_height[0]])
        self.xyz + self.enu2ecef(x=self.antenna_height)
        return self.xyz + self.enu2ecef(x=self.antenna_height)

    def project_antenna_height(self):
        xyz = self.xyz
        dist = calculate_distance(self.df[['xe', 'ye', 'ze']].to_numpy(), xyz)
        dx = self.df['xe'].to_numpy() - xyz[0]
        dy = self.df['ye'].to_numpy() - xyz[1]
        dz = self.df['ze'].to_numpy() - xyz[2]
        e = -np.column_stack((dx / dist, dy / dist, dz / dist))
        enu = np.array([self.antenna_height[1], self.antenna_height[2],self.antenna_height[0]])
        ecef = self.enu2ecef(x=enu)
        ah_los = e.dot(ecef)
        self.df['ah_los'] = ah_los

    def _rec_pco_enu(self, freq: str, entry=None) -> np.ndarray:
        """Receiver PCO [m] of ANTEX frequency ``freq`` as East, North, Up (ANTEX gives North, East, Up),
        the order :meth:`enu2ecef` expects."""
        entry = self._antenna_model().receiver if entry is None else entry
        if entry is None or not entry.has(freq):
            return np.zeros(3)
        return entry.pco_neu_m(freq)[[1, 0, 2]]

    def compute_pco_at_los(self):
        """Receiver antenna term per band column of the mode (``gnss.mode_layout``), and ``pco_los_l3``
        (IF combination) for dual-frequency modes.

        Value = -PCO_enu . u + PCV(zenith, azimuth) [m], u receiver -> satellite, so the range model is
        rho(ARP) + pco_los (routes: obs - pco_los). PCV: azimuth grid when the entry has one (DAZI > 0),
        else NOAZI; zenith = 90 - 'ev', azimuth 'az' [deg, clockwise from north]. A frequency without a
        calibration takes the nearest calibrated one with a warning; an uncalibrated antenna gives 0 with a
        warning.
        """
        model = self._antenna_model()
        rec = model.receiver
        layout = mode_layout(self.mode)
        cols = [band["rec_pco_col"] for band in layout]
        if rec is None:
            if model.atx is not None and model.receiver_type:
                what = f"receiver antenna {model.receiver_label()} is not in {model.atx_name}"
            elif model.atx is not None:
                what = "the RINEX header gives no receiver antenna type"
            else:
                what = "no receiver antenna calibration was given (rec_pco)"
            warnings.warn(f"{what}: the receiver PCO and PCV are 0 ({self.SYS} {self.mode})", UserWarning,
                          stacklevel=2)
            for c in cols:
                self.df[c] = 0.0
            if len(layout) == 2:
                self.df['pco_los_l3'] = 0.0
            return

        xyz = np.asarray(self.xyz, dtype=float).reshape(3)
        coords = self.df[['xe', 'ye', 'ze']].to_numpy(dtype=float)
        dist = calculate_distance(coords, xyz)
        # unit vector satellite -> receiver (= -u)
        e = -(coords - xyz) / np.asarray(dist, dtype=float).reshape(-1, 1)
        use_pcv = model.pcv and 'ev' in self.df.columns
        if use_pcv:
            zen = 90.0 - self.df['ev'].to_numpy(dtype=float)
            az = self.df['az'].to_numpy(dtype=float) if 'az' in self.df.columns else None
        values = []
        for band in layout:
            freq, _note = receiver_frequency(rec, band["antex"])
            if freq is None:
                warnings.warn(f"receiver antenna {model.receiver_label()}: no calibration usable for "
                              f"{band['antex']}: its PCO and PCV are 0", UserWarning, stacklevel=2)
                v = np.zeros(len(self.df))
            else:
                v = e.dot(self.enu2ecef(x=self._rec_pco_enu(freq, rec)))
                if use_pcv:
                    pcv = rec.pcv_m(freq, zen, az)
                    v = v + np.where(np.isfinite(pcv), pcv, 0.0)
            self.df[band["rec_pco_col"]] = v
            values.append(v)
        if len(layout) == 2:
            k1, k2 = mode_ionosphere_free_coefficients(self.mode)
            self.df['pco_los_l3'] = k1 * values[0] - k2 * values[1]

    def _require_broadcast_iono(self, model: str, records: str, *coefficients) -> None:
        """ValueError naming the model and the missing IONOSPHERIC CORR records when the navigation header
        gave no coefficients (a RINEX 4 header has none, a RINEX 2 header no GAL record; the
        None reached the model as a TypeError)."""
        if all(c is not None for c in coefficients):
            return
        nav = getattr(self.configuration, 'nav_path', None)
        raise ValueError(
            f"ionosphere_model={model!r} needs the {records} coefficients of the navigation header "
            f"(IONOSPHERIC CORR); {nav or 'the navigation data'} has none for {self.SYS} {self.mode}. Use a "
            f"navigation file whose header carries them, or ionosphere_model='gim' (gim_path)")

    def compute_ntcm(self):
        """NTCM-G 'ion' [m] at the first signal of the mode, per epoch with the Galileo broadcast set (ai0, ai1,
        ai2) in force at the epoch (RINEX 4 '> ION' records, :func:`_broadcast_iono_rows`; BRD400DLR changes the
        set at 14:11 on 2024-035 and at 17:21 on 2024-132), else the session's one set
        (``gala``: RINEX 3 header)."""
        times = self.df.index.get_level_values('time')
        uniq = pd.DatetimeIndex(times.unique())
        sets = _broadcast_iono_rows(self.iono, 'E', 'NEQG', uniq.to_numpy(dtype='datetime64[ns]'))
        if sets is None:
            self._require_broadcast_iono('ntcm', 'GAL (RINEX 2: none)', self.gala)
            gala_at = {t: self.gala for t in uniq}
        else:
            gala_at = {t: [float(v) for v in row[:3]] for t, row in zip(uniq, sets)}

        grouped = self.df.groupby(level='time')

        # Define the wrapper function
        def apply_run_NTCM_for_gal(group):
            az = group['az'].values
            ev = group['ev'].values
            epoch = group.name  # The group name is the time epoch
            DOY = datetime2doy(epoch)
            # Call the function
            stec = ntcm_vtec(
                pos=self.flh,
                DOY=DOY,
                ev=ev,
                az=az,
                epoch=epoch,
                gala=gala_at[pd.Timestamp(epoch)],
                mf_no=1  # NTCM-G MSLM (alpha 0.9782 at 450 km, spec sect. 2.5.8 eq. 32-33)
            )
            return pd.Series(stec, index=group.index)

        # Apply the function to each group
        result = grouped.apply(apply_run_NTCM_for_gal)
        result.index = result.index.droplevel(0)  # Remove the first level

        # Assign the result back to the DataFrame
        stec = result
        K = 40.3 * 10 ** 16
        f = frequency_hz(first_signal(self.mode))
        self.df['ion'] = K * (stec / (f ** 2))

    def compute_bdgim(self):
        """BDGIM 'ion' and 'ion_rms' [m] at the first signal of the mode (``ionosphere_model='bdgim'``; BDS-SIS-ICD-B2b-1.0 §7.7, :mod:`gnx_py.ionosphere.bdgim`). G, E and C alike: the model is global.

        - Per row the set in force at the epoch (``_bdgim_rows``: latest plausible '> ION Cxx CNVX' record at or
          before the epoch, any BeiDou satellite; selection in GPST), evaluated at BDT = GPST - 14 s with the
          receiver's geodetic latitude/longitude.
        - 'ion' = 40.3e16 / f1**2 * max(STEC, 0) (the GNX constant, as GIM, Klobuchar and NTCM; a negative
          model STEC gives 0 and is counted in ``iono_report['n_negative_clamped']``).
        - 'ion_rms' = 40.3e16 / f1**2 * hypot(0.5 |STEC|, 5 TECU x MF) from the unclamped STEC, so a clamped
          0 never makes the PPP constraint rigid.
        - Rows with a negative elevation get NaN in both.
        - One UserWarning per run (``run_notes``, shared by the instances of the session run) when a set in force
          is more than 6 h old, and one when rows precede the first set of the file by more than 6 h (they take
          that set; the set is used); and, when the navigation reader did not warn for BeiDou records (its
          systems, ``RunNotes.nav_systems``, without 'C'), one naming the rejected BDGIM records inside the window.
        - ValueError naming ``nav_path`` and its RINEX version when there is no usable BDGIM set; no
          fallback to another model.
        """
        self.df = self.df.copy()
        t = self.df.index.get_level_values('time').to_numpy('datetime64[ns]')
        alpha, info = _bdgim_rows(self.iono, t)
        if alpha is None:
            raise ValueError(bdgim_missing_message(
                getattr(self.configuration, 'nav_path', None), info,
                f"ionosphere_model='bdgim' ({self.SYS} {self.mode})", "ionosphere_model='klobuchar' / 'gim'"))
        az = self.df['az'].to_numpy(dtype=float)
        ev = self.df['ev'].to_numpy(dtype=float)
        r = bdgim_slant(alpha, self.flh[0], self.flh[1], az, ev, gpst_to_bdt(t))
        stec = np.asarray(r['stec'], dtype=float).reshape(-1)
        mf = np.asarray(r['mf'], dtype=float).reshape(-1)
        scale = self.IONO_K / frequency_hz(first_signal(self.mode)) ** 2
        neg = stec < 0.0
        self.df['ion'] = scale * np.where(neg, 0.0, stec)
        self.df['ion_rms'] = scale * np.hypot(0.5 * np.abs(stec), self.BDGIM_SIGMA_FLOOR_TECU * mf)
        old = info['age_h'] > self.BDGIM_MAX_AGE_WARN_H
        ahead = info['lead_h'] > self.BDGIM_MAX_AGE_WARN_H
        self.iono_report = {
            'model': 'bdgim', 'system': self.SYS, 'mode': self.mode, 'rows': int(t.size),
            'sets': info['sets'], 'max_age_h': info['max_age_h'], 'n_rows_older_than_warn': int(np.count_nonzero(old)),
            'n_before_first': info['n_before_first'], 'max_lead_h': info['max_lead_h'],
            'n_rows_lead_over_warn': int(np.count_nonzero(ahead)), 'n_negative_clamped': int(np.count_nonzero(neg)),
            'n_rejected_in_window': info['n_rejected_in_window'], 'rejected_in_window': info['rejected_in_window'],
            'reasons': sorted({reason for _sv, _t, reason in info['rejected_in_window']}),
        }
        self.run_notes.iono_report[self.SYS] = self.iono_report
        _LOG.info("BDGIM %s %s: %d rows, %d sets, max age %.1f h, %d rows before the first set (up to %.1f h), %d "
                  "negative values clamped to 0, %d rejected records in the window", self.SYS, self.mode, t.size,
                  len(info['sets']), info['max_age_h'], info['n_before_first'], info['max_lead_h'],
                  int(np.count_nonzero(neg)), info['n_rejected_in_window'])
        span = (f"{info['first_set']:%Y-%m-%d %H:%M:%S}..{info['last_set']:%Y-%m-%d %H:%M:%S} GPST"
                if info['first_set'] is not None else "no dated set")
        if old.any() and self.run_notes.first('bdgim_age'):
            warnings.warn(
                f"BDGIM set in force is up to {info['max_age_h']:.1f} h old for {int(np.count_nonzero(old))} rows "
                f"({self.SYS} {self.mode}; the BDGIM sets of nav_path cover {span}); the set is used",
                UserWarning, stacklevel=2)
        if ahead.any() and self.run_notes.first('bdgim_lead'):
            warnings.warn(
                f"{int(np.count_nonzero(ahead))} rows precede the first BDGIM set of nav_path by up to "
                f"{info['max_lead_h']:.1f} h ({self.SYS} {self.mode}; the BDGIM sets of nav_path cover {span}); "
                f"they take that set: is nav_path of the observation day?", UserWarning, stacklevel=2)
        reader = self.run_notes.nav_systems
        if reader is None:
            configured = getattr(self.configuration, 'sys', None)
            with_bds = self.SYS == 'C' if configured is None else 'C' in configured
        else:
            with_bds = 'C' in reader
        if info['n_rejected_in_window'] and not with_bds and self.run_notes.first('bdgim_rejected'):
            # the reader warns for the rejected records of its systems and GPS only
            names = "; ".join(f"'> ION {sv} CNVX' {tt} ({reason})" for sv, tt, reason in info['rejected_in_window'])
            warnings.warn(
                f"ionosphere_model='bdgim': {info['n_rejected_in_window']} BDGIM record(s) of "
                f"{getattr(self.configuration, 'nav_path', None) or 'the navigation data'} inside the run window "
                f"were rejected and not used: {names}", UserWarning, stacklevel=2)

    def compute_troposphere_vectorized(self):
        ev = self.df['ev'].to_numpy()
        tro = np.array(list(map(lambda row: saastamoinen(row, h=self.flh[2]), ev)))
        self.df['tro'] = tro


    def read_inx(self):
        K = 40.3 * 10 ** 16
        f = frequency_hz(first_signal(self.mode))

        parser = GIMReader(tec_path=self.gim_file,dcb_path=self.gim_file)
        data = parser.read(dcb=True)
        times = self.df.index.get_level_values('time').to_numpy('datetime64[ns]')
        use_rms = getattr(self.configuration, 'use_iono_rms', False)
        # GPST epochs -> UTC map epochs (clamped within 19 s before the first map); mapping
        # function by header or setting; rotated time interpolation by default (IONEX eq. 3)
        gim = gim_slant_tec(data, times, self.df['ev'].to_numpy(), self.df['az'].to_numpy(), self.xyz,
                            mapping=getattr(self.configuration, 'gim_mapping_function', 'auto'),
                            time_interpolation=getattr(self.configuration, 'gim_time_interpolation', 'rotated'),
                            rms=bool(use_rms), notes=self.configuration)
        # VTEC is only interpolated between maps (no extrapolation in time): a GIM of another
        # day gives NaN everywhere (SPP: LinAlgError, PPP: empty constraint set) -> readable error;
        # a partial overlap (session across midnight, maps ending before 24:00) only loses the
        # uncovered epochs, which the NaN-ion paths of SPP/PPP drop -> warning
        check_gim_coverage(data, gim, times, source=self.gim_file, what=f"{self.SYS} observations")
        outside = gim['outside']
        lat_ipp, lon_ipp, m_f, vtec = gim['lat_ipp'], gim['lon_ipp'], gim['m_f'], gim['vtec']
        self.df['lat_ipp'] = lat_ipp
        self.df['lon_ipp'] = lon_ipp
        self.df['ion'] = K * vtec * m_f / f ** 2
        no_value = np.isnan(vtec) & ~outside
        if no_value.any():
            sv = (self.df.index.get_level_values('sv') if 'sv' in self.df.index.names
                  else pd.Index(['?'] * len(self.df)))
            bad_sv = sorted(set(map(str, sv[no_value])))
            warnings.warn(
                f"GIM {self.gim_file}: no VTEC value (9999 'no value' in the map, IPP outside a regional "
                f"grid or missing elevation/azimuth) for {int(no_value.sum())} of "
                f"{no_value.size} {self.SYS} observations (satellites {', '.join(bad_sv[:10])}"
                f"{' ...' if len(bad_sv) > 10 else ''}; first epoch {pd.Timestamp(times[no_value][0])}); "
                f"their 'ion' is NaN", RuntimeWarning, stacklevel=2)
        if use_rms:
            if data.rms is None:
                # RMS maps are optional in IONEX: the GIM is used without its uncertainty
                warnings.warn(
                    f"use_iono_rms=True but the GIM {self.gim_file} has no RMS maps; the ionosphere "
                    f"is not weighted with the GIM RMS (no 'ion_rms' column)", RuntimeWarning, stacklevel=2)
            else:
                rms = gim['rms']
                self.df['ion_rms'] = K * rms * m_f / f ** 2
                # RMS maps may span fewer epochs than the TEC maps (ionex warns on reading)
                no_rms = np.isnan(rms) & np.isfinite(vtec)
                if no_rms.any():
                    warnings.warn(
                        f"GIM {self.gim_file}: no RMS value for {int(no_rms.sum())} of {no_rms.size} {self.SYS} "
                        f"observations with a VTEC value (first epoch {pd.Timestamp(times[no_rms][0])}); "
                        f"their 'ion_rms' is NaN", RuntimeWarning, stacklevel=2)

        # an IONEX without the DCB AUX block gives an empty frame: no 'bias' columns, the
        # STEC bias policy then falls back (and warns) on its own
        if data.dcb is not None and not data.dcb.empty:

            sat_dcb = data.dcb[(data.dcb['entry_type'] == 'satellite') &
                     data.dcb['sv'].str.startswith(self.SYS)]
            sat_dcb = sat_dcb.set_index('sv')

            self.df = self.df.join(sat_dcb,on='sv')
            if self.configuration.station_name is not None:
                # IONEX gives the site code with the DOMES number ('BRUX13101M010'); station_name may be the
                # 4-character site code or the 9-character id, in any case (ION-011)
                stations = data.dcb[(data.dcb['entry_type'] == 'station') & (data.dcb['system'] == self.SYS)]
                names = match_station(self.configuration.station_name, stations['prn_or_site'].astype(str).unique())
                if names:
                    sta_bias = stations[stations['prn_or_site'].astype(str) == names[0]]
                    self.df['sta_bias'] = sta_bias['bias'].values[0]


    def compute_niell(self):
        tro, dry, wet, me_wet = niell(ev=self.df['ev'].to_numpy(),lat=self.flh[0],lon=self.flh[1], h=self.flh[2],
                                      doy=self.doy)
        self.df=self.df.copy()
        self.df['tro'] = 0.0
        self.df.loc[:, 'tro'] = tro
        self.df.loc[:, 'dry'] = dry
        self.df.loc[:, 'wet'] = wet
        self.df.loc[:, 'me_wet'] = me_wet

    def compute_collins(self):
        ev = self.df['ev'].to_numpy()
        tro_co = np.fromiter(
            (
                tropospheric_delay(
                    f=self.flh[0], h=self.flh[2], elevation=el, doy=self.doy
                )
                for el in ev
            ),
            dtype=float,
            count=ev.size,
        )
        # 'tro' is the column every route reads; 'tro_co' stays as an alias for older readers
        self.df['tro'] = tro_co
        self.df['tro_co'] = tro_co


    def _tides_enabled(self) -> bool:
        """Any site-displacement term on: solid tide, ocean tide loading (BLQ) or pole tide (ERP)."""
        cfg = self.configuration
        return bool(getattr(cfg, 'solid_tides', True) or getattr(cfg, 'ocean_loading_path', None)
                    or getattr(cfg, 'erp_path', None))

    def _ocean_loading_station(self):
        """BLQ block of this station (``config.ocean_loading_path``), or None when the station is not in the file:
        one UserWarning per configuration (in a session ``validate_for_run`` gives it before the preprocessing of
        the first system)."""
        cfg = self.configuration
        path = getattr(cfg, 'ocean_loading_path', None)
        if not path:
            return None
        from ..models.ocean_loading import resolve_station, warn_missing_station_once

        station, blq, names = resolve_station(cfg)
        if station is not None:
            _LOG.info("ocean tide loading: station %s of %s (model %s)", station.name, path, blq.model)
            return station
        warn_missing_station_once(cfg, path, blq, names, stacklevel=3)
        return None

    def compute_tides(self):
        """Site displacement dx/dy/dz [m, ECEF] per epoch and its line-of-sight projection ``tides_los``.

        Solid tide (``solid_tides``: step 1 of ``tides.get_tides`` plus, with ``solid_tide_step2``, the IERS 2010
        step-2 terms), ocean tide loading (``ocean_loading_path``) and pole tide (``erp_path``).
        """
        cfg = self.configuration
        epochs = self.df.index.get_level_values('time').unique().tolist()
        solid = bool(getattr(cfg, 'solid_tides', True))
        if solid:
            tides, radial, north, sun_tides, moon_tides = get_tides(xyz=self.xyz,datetimes=epochs)
            tides['N'] = north
            tides['R']=radial
        else:
            tides = pd.DataFrame({'dx': 0.0, 'dy': 0.0, 'dz': 0.0},
                                 index=pd.DatetimeIndex(pd.to_datetime(epochs), name='time'))
        step2 = solid and bool(getattr(cfg, 'solid_tide_step2', True))
        blq_station = self._ocean_loading_station()
        erp = None
        if getattr(cfg, 'erp_path', None):
            from ..io.erp import read_erp

            erp = read_erp(cfg.erp_path)
        if step2 or blq_station is not None or erp is not None:
            extra = site_displacement(self.xyz, tides.index, solid=False, blq_station=blq_station, erp=erp,
                                      mean_pole=getattr(cfg, 'pole_tide_mean_pole', 'secular'))
            add = extra[['dx', 'dy', 'dz']].to_numpy(dtype=float)
            if step2:
                add = add + solid_tide_step2(self.xyz, tides.index)
            tides[['dx', 'dy', 'dz']] = tides[['dx', 'dy', 'dz']].to_numpy(dtype=float) + add
        dX = self.df[['xe','ye','ze']].to_numpy() - self.xyz
        dist = np.linalg.norm(dX, axis=1)
        LOS = -dX / dist[:, np.newaxis]
        self.df[['LOS1', 'LOS2', 'LOS3']] = LOS
        self.df = self.df.join(tides,on=['time'])

        # tides_los
        los = self.df[['LOS1', 'LOS2', 'LOS3']].to_numpy()
        tides = self.df[['dx','dy','dz']].to_numpy()
        tides_los = np.sum(los*tides,axis=1)
        self.df['tides_los']=tides_los

    def compute_corrections(self, get_windup=True, get_rel=True):
        """``phw`` (phase wind-up [cycles], per satellite unwrapped as RTKLIB windupcorr) and ``dprel`` (relativistic
        path delay [m]) per row.

        The wind-up takes the satellite x/y axes of :meth:`_attitude` (``config.satellite_attitude``: nominal yaw
        steering, BeiDou GEO orbit-normal, or the attitude laws / ORBEX), the same frame as the satellite PCO, and
        writes the attitude columns att_yaw_deg, att_regime and att_source (reporting only).
        """
        if len(self.df) == 0:
            # no row (everything masked or no satellite position): the columns, no attitude or Sun position
            # (pd.concat of an empty dict raised "No objects to concatenate", PPP-042)
            self.df = self.df.assign(phw=pd.Series(dtype=float), dprel=pd.Series(dtype=float))
            return
        out_dict = {}
        if get_windup:
            att = self._attitude()
            ax = att[['x1', 'x2', 'x3']].to_numpy(dtype=float)
            ay = att[['y1', 'y2', 'y3']].to_numpy(dtype=float)
            sun = self._sun_rows(self.df.index.get_level_values("time"))
            self.df['att_yaw_deg'] = att['yaw_deg'].to_numpy()
            self.df['att_regime'] = att['regime'].to_numpy()
            self.df['att_source'] = att['source'].to_numpy()

        for sv, pos in self.df.groupby("sv", sort=False).indices.items():
            gr = self.df.iloc[pos]
            times = pd.to_datetime(gr.index.get_level_values("time"))
            satpositions = gr[["xe", "ye", "ze"]].to_numpy()

            if get_windup:
                windup = process_windup_correction_vectorized(times, satpositions, self.xyz, rsun_all=sun[pos],
                                                              sat_axes=(ax[pos], ay[pos]))
            else:
                windup = np.zeros(len(gr))

            if get_rel:
                path_rel = rel_path_corr(satpositions, self.xyz)
            else:
                path_rel = np.zeros(len(gr))

            out_dict[sv] = pd.DataFrame({"phw": windup, "dprel": path_rel}, index=times)

        out = pd.concat(out_dict, keys=out_dict.keys(), names=["sv", "time"])
        self.df = self.df.join(out)

    def has_phase_observations(self) -> bool:
        phase_columns = [col for col in self.df.columns if col.startswith('L')]
        return len(phase_columns) > 0

    def run(self):
        # decision D9: a PRN reassigned during the session leaves it from the change on (before every model)
        self.apply_prn_change_rule()
        if self.has_phase_observations():
            self.phase2meters()
            self.apply_phase_shift()
        else:
            _LOG.debug("No phase observations (%s %s)", self.SYS, self.mode)
        if self.configuration.sat_pco in ('los', 'crd'):
            # satellites without a valid ANTEX entry: excluded with one warning, or ValueError
            self.apply_missing_satellite_rule()
        if self.configuration.sat_pco=='los':
            self.sat_pco_los()
        elif self.configuration.sat_pco=='crd':
            self.sat_fixed_system()

        if self.configuration.antenna_h:
            self.project_antenna_height()
        if self.configuration.rec_pco:
            self.compute_pco_at_los()
        if self.configuration.troposphere_model =='niell':
            self.compute_niell()
        elif self.configuration.troposphere_model =='saastamoinen':
            self.compute_troposphere_vectorized()
        elif self.configuration.troposphere_model =='collins':
            self.compute_collins()
        if self.configuration.ionosphere_model =='gim':
            self.read_inx()
        elif self.configuration.ionosphere_model =='klobuchar':
            self.compute_klobuchar()
            self.broadcast_iono_rms()
        elif self.configuration.ionosphere_model =='ntcm':
            self.compute_ntcm()
            self.broadcast_iono_rms()
        elif self.configuration.ionosphere_model == 'bdgim':
            # sets its own ion_rms (5 TECU floor, D3): broadcast_iono_rms is not applied
            self.compute_bdgim()
        if self.configuration.windup or self.configuration.rel_path:
            self.compute_corrections(get_windup=self.configuration.windup, get_rel=self.configuration.rel_path)
        if self._tides_enabled():
            self.compute_tides()
        self._zero_switched_off_corrections()
        self._apply_turn_guard()
        return self.df

    def _apply_turn_guard(self) -> None:
        """Drop the rows of the attitude turn guard (``attitude_turn_guard``, PPP only).

        Without the ORBEX attitude of the product's analysis centre, rows within GUARD_HALF_WINDOW_S of a
        noon/midnight turn at small |beta| (``gnx_py.attitude``) are removed: analysis centres fly some of these
        turns the other way round than the published laws, and their clocks then differ from the model by one
        wind-up cycle. The next row of the satellite opens a new arc (``split_obs_dt``), so its ambiguity
        restarts after the turn. One UserWarning per configuration, system and input gives the counts. The guard is
        part of the 'model' attitude: with ``satellite_attitude='nominal'`` no row is guarded
        (``attitude_report['guard']`` stays 'off').
        """
        mode = self._turn_guard_mode() if self._attitude_mode() == "model" else "off"
        if mode == "off" or not isinstance(self.df, pd.DataFrame) or self.df.empty:
            return
        if not {"xe", "ye", "ze"} <= set(self.df.columns):
            return
        guard = self._attitude()["guard"].to_numpy(dtype=bool)
        report = self.attitude_report
        report["guard"] = mode
        report["guarded_rows"] = int(guard.sum())
        if not guard.any():
            report["guarded_satellites"] = []
            return
        svs = self.df.index.get_level_values("sv").astype(str).to_numpy()
        sats = sorted(set(svs[guard]))
        report["guarded_satellites"] = sats
        path = getattr(self.configuration, "attitude_path", None)
        # once per run, system and guarded set (gnx_py._notes; the session clears the keys at the start of each
        # run): the session validation and every system's preprocessing reach the same rows. obs_path / sp3_path
        # stay in the key for a configuration used outside a session run (no clearing), where other input data
        # warns again
        cfg = self.configuration
        key = ("turn_guard", self.SYS, mode, tuple(sats), int(guard.sum()),
               repr(getattr(cfg, "obs_path", None)), repr(getattr(cfg, "sp3_path", None)))
        _warn_once(cfg, key,
                   f"attitude turn guard ({mode!r}): {int(guard.sum())} {SYSTEM_NAMES.get(self.SYS, self.SYS)} "
                   f"observations of {len(sats)} satellite{'s' if len(sats) != 1 else ''} ({', '.join(sats)}) "
                   f"near noon/midnight turns at small |beta| are dropped and their ambiguities restart after the "
                   f"turn, because the clocks of analysis centres can differ from the attitude laws by a wind-up "
                   f"cycle there. "
                   + ("The ORBEX attitude (attitude_path) does not cover them or is not used; " if path else "")
                   + "attitude_turn_guard='off' keeps them with the attitude laws", UserWarning, stacklevel=4)
        self.df = self.df[~guard].copy()

    def _zero_switched_off_corrections(self):
        """A switched-off correction is a zero term: write phw/dprel = 0 when absent and, without solid tides,
        the line of sight LOS1..3 (as compute_tides), dx = dy = dz = 0 and tides_los = 0, so every route
        finds its columns."""
        df = self.df
        if 'phw' not in df.columns:
            df['phw'] = 0.0
        if 'dprel' not in df.columns:
            df['dprel'] = 0.0
        if not self._tides_enabled() and 'tides_los' not in df.columns:
            dX = df[['xe', 'ye', 'ze']].to_numpy(dtype=float) - np.asarray(self.xyz, dtype=float).reshape(3)
            dist = np.linalg.norm(dX, axis=1)
            df[['LOS1', 'LOS2', 'LOS3']] = -dX / dist[:, np.newaxis]
            df[['dx', 'dy', 'dz']] = 0.0
            df['tides_los'] = 0.0
        self.df = df
