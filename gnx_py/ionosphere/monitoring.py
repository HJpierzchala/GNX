from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="The PostScript backend does not support transparency*"
)

# --- SIDX, GIX ---
Re = 6371e3
IONO_H = 450e3
MIN_ELEV_DEG = 10.0
DMIN_KM = 30.0
DMAX_KM = 50.0
MAX_PAIR_PER_EPOCH = 200000  # random thinning fuse
# --- ROT/ROTI ---
ROTI_WINDOW_MIN = 5            # window for STD
ROTI_MIN_SAMPLES = 3           # min. number of ROTs in the window to calculate STD
MAX_GAP_S       = 180.0        # cut off jumps with large time gaps
SPLIT_BY_DAY    = True         # do not differentiate by UTC midnight (reset per day)
DDOF_POP        = 0            # according to the definition ROTI = sqrt(<ROT^2>-<ROT>^2)

# --- global standard for plotting ---
FIGSIZE = (10, 10)
DPI = 600
_GS_HEIGHT_RATIOS = (1.0, 0.06)
_GS_HSPACE = 0.05

@dataclass
class Region:
    """
    Definition of the region covered by the ROTI/GIX/SIDX measurement
    """
    lat_min: float = -90.0
    lat_max: float =  90.0
    lon_min: float = -180.0
    lon_max: float =  180.0
    def contains(self, lat, lon) -> np.ndarray:
        lon = ((lon + 180) % 360) - 180
        return (lat >= self.lat_min) & (lat <= self.lat_max) & \
               (lon >= self.lon_min) & (lon <= self.lon_max)

def mapping_M(elev_deg: np.ndarray, Re_m: float = Re, H_m: float = IONO_H) -> np.ndarray:
    """
    Mapping function (this shell model)
    :param elev_deg: elevation in degrees
    :param Re_m: mean earth radius in meters
    :param H_m: Ionospheric shell height in meters
    :return: mapping function value
    """
    el = np.deg2rad(np.clip(elev_deg, 0.01, 89.99))
    a = (Re_m * np.cos(el)) / (Re_m + H_m)
    return 1.0 / np.sqrt(1.0 - a * a)

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
    split_on_day_change: bool = True,        # DO NOT count differences by midnight (UTC)
    agg: str = "median",              # 'median' or 'mean'
) -> pd.Series:
    """
    SIDX ≈ < dltSTEC / (M·dltt) >  (TECU/min).
    Differentiation only within the same link (name, sv) and — if enabled —
    within the same UTC day (no differences across midnight).
    Required columns: name, sv, time, stec, ev, lat_ipp, lon_ipp.
    """
    need = {'name','sv','time','stec','ev','lat_ipp','lon_ipp'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Columns are missing: {sorted(miss)}")

    dfa = df.copy()
    dfa['time'] = pd.to_datetime(dfa['time'], utc=True)

    # elevation mask
    dfa = dfa[dfa['ev'] >= min_elev_deg].copy()
    if dfa.empty:
        print(f'No data after elevation mask application')
        return pd.Series(dtype=float, name='SIDX_tecu_per_min')

    # units and mapping function
    dfa['stec_tecu'] = dfa['stec'] * stec_scale
    dfa['M'] = dfa['M'] if 'M' in dfa.columns else mapping_M(dfa['ev'].to_numpy(), H_m=iono_h_m)

    # identification of UTC time for arc division
    if split_on_day_change:
        dfa['day'] = dfa['time'].dt.floor('D')  # UTC doba
        grp_cols = ['name','sv','day']
    else:
        grp_cols = ['name','sv']

    # differentiation within groups (link + possibly day)
    dfa = dfa.sort_values(grp_cols + ['time']).reset_index(drop=True)
    dfa['dt_s']  = dfa.groupby(grp_cols)['time'].diff().dt.total_seconds()
    dfa['dSTEC'] = dfa.groupby(grp_cols)['stec_tecu'].diff()

    rate = dfa.dropna(subset=['dt_s','dSTEC','M'])
    rate = rate[rate['dt_s'] > 0]

    # discard large time gaps
    if max_gap_s is not None:
        rate = rate[rate['dt_s'] <= float(max_gap_s)]

    # IPP region mask (if specified)
    if region is not None:
        inreg = region.contains(rate['lat_ipp'].to_numpy(), rate['lon_ipp'].to_numpy())
        rate = rate[inreg]

    # instantaneous indicator for individual links
    rate['sid_link'] = rate['dSTEC'] / (rate['M'] * rate['dt_s'])  # TECU/s

    # post-epoch aggregation (average/median across links)
    if agg == "median":
        sid = rate.groupby('time')['sid_link'].median().sort_index()
    elif agg == "mean":
        sid = rate.groupby('time')['sid_link'].mean().sort_index()
    elif agg == "max":
        sid = rate.groupby('time')['sid_link'].max().sort_index()
    elif agg == "p95":
        sid = rate.groupby('time')['sid_link'].quantile(0.95, interpolation='linear').sort_index()
    else:
        raise ValueError("agg must be 'median', 'mean', 'max', or 'p95'")


    # scaling of units → mTECU/s
    sid = sid * 1e3
    sid.name = 'SIDX_tecu_per_min'
    return sid

# --- GIX ---


# --- auxiliary: layered thinning by Δs and azimuth---
def _stratified_subsample(mask_idx: np.ndarray, d_km: np.ndarray, az_deg: np.ndarray,
                          target: int, n_dist_bins: int = 4, n_az_bins: int = 8,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
   Returns a subset of mask_idx (int) indices in number <= target,
    sampled uniformly from 2D baskets: Δs×azimuth.
    """
    if rng is None:
        rng = np.random.default_rng()
    if mask_idx.size <= target:
        return mask_idx

    dsel = d_km[mask_idx]
    azsel = az_deg[mask_idx]

    # koszyki
    d_bins = np.quantile(dsel, np.linspace(0, 1, n_dist_bins+1))
    # to avoid duplicate boundaries with low variability
    d_bins = np.unique(d_bins)
    if d_bins.size <= 2:  # fallback
        d_bins = np.array([dsel.min(), dsel.max()])
    az_bins = np.linspace(0, 360, n_az_bins+1)

    d_cat = np.clip(np.digitize(dsel, d_bins, right=True)-1, 0, len(d_bins)-2)
    az_cat = np.clip(np.digitize(azsel, az_bins, right=True)-1, 0, len(az_bins)-2)

    # ultimately equally distributed among the basket
    n_cells = (len(d_bins)-1) * (len(az_bins)-1)
    per_cell = max(1, target // n_cells)

    chosen = []
    for di in range(len(d_bins)-1):
        for ai in range(len(az_bins)-1):
            cell = mask_idx[(d_cat == di) & (az_cat == ai)]
            if cell.size == 0:
                continue
            if cell.size <= per_cell:
                chosen.append(cell)
            else:
                pick = rng.choice(cell, size=per_cell, replace=False)
                chosen.append(pick)

    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    # if still short of the target, select randomly from the remaining ones
    if chosen.size < target:
        remaining_pool = np.setdiff1d(mask_idx, chosen, assume_unique=False)
        need = target - chosen.size
        if remaining_pool.size > 0:
            extra = rng.choice(remaining_pool, size=min(need, remaining_pool.size), replace=False)
            chosen = np.concatenate([chosen, extra])

    return chosen[:target]

_DIP_PRESETS = {
    "30-250": (30.0, 250.0),
    "50-500": (50.0, 500.0),
    "50-1000": (50.0, 1000.0),
}
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
    rng: Optional[np.random.Generator] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input (required columns): name,time,lat_ipp,lon_ipp,stec,ev,(optional M).
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
         'n_pairs']
      - pairs_df (each pair) with fields as in your case (CP_lat/lon, grad_x/y/abs, Δs, Mi, Mj)
    """
    need = {'name','time','lat_ipp','lon_ipp','stec','ev'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Columns missing: {sorted(miss)}")

    if dipole_preset not in _DIP_PRESETS and (dmin_km is None or dmax_km is None):
        raise ValueError("Specify the correct dipole_preset ('30-250'|'50-500'|'50-1000') "
                         "or explicitly dmin_km and dmax_km.")

    if dmin_km is None or dmax_km is None:
        dmin_km, dmax_km = _DIP_PRESETS[dipole_preset]

    obs = df.copy()
    obs['time'] = pd.to_datetime(obs['time'], utc=True)
    obs = obs[obs['ev'] >= float(min_elev_deg)].copy()
    if obs.empty:
        return pd.DataFrame(), pd.DataFrame()

    obs['stec_tecu'] = obs['stec'] * float(stec_scale)
    obs['M'] = obs['M'] if 'M' in obs.columns else mapping_M(obs['ev'].to_numpy(), H_m=iono_h_m)
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

        # dipole lengths (over a shell)
        d_km_all = great_circle_distance_km(
            sub['lat_ipp'].to_numpy()[ii], sub['lon_ipp'].to_numpy()[ii],
            sub['lat_ipp'].to_numpy()[jj], sub['lon_ipp'].to_numpy()[jj],
            radius_m=Re + iono_h_m
        )

        # pair azimuth (direction j->i)
        az_all = initial_bearing_deg(
            sub['lat_ipp'].to_numpy()[jj], sub['lon_ipp'].to_numpy()[jj],
            sub['lat_ipp'].to_numpy()[ii], sub['lon_ipp'].to_numpy()[ii]
        )

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

        # thinning: regular or layered Δs×azimuth
        if idx.size > max_pairs_per_epoch:
            if stratified_sampling:
                idx = _stratified_subsample(idx, d_km_all, az_all,
                                            target=max_pairs_per_epoch, rng=rng)
            else:
                idx = rng.choice(idx, size=max_pairs_per_epoch, replace=False)

        # final choice
        ii, jj = ii[idx], jj[idx]
        dsel   = d_km_all[idx]
        Mi, Mj = Mi_all[idx], Mj_all[idx]
        az     = az_all[idx]

        # gradient (TECU/km) – signed
        dVTEC = sub['VTEC'].to_numpy()[ii] - sub['VTEC'].to_numpy()[jj]
        sv_ii, sv_jj = sub['sv'].to_numpy()[ii], sub['sv'].to_numpy()[jj]
        name_ii, name_jj = sub['name'].to_numpy()[ii], sub['name'].to_numpy()[jj]
        grad_signed = dVTEC / dsel
        grad_abs    = np.abs(grad_signed)

        # breakdown into x/y components (WE/NS) – signed
        sindlt, cosdlt = np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))
        grad_x = grad_signed * sindlt # E-W
        grad_y = grad_signed * cosdlt # N-S

        # pair midpoint (for region mask and visualization)
        cp_lat, cp_lon = spherical_midpoint(
            sub['lat_ipp'].to_numpy()[ii], sub['lon_ipp'].to_numpy()[ii],
            sub['lat_ipp'].to_numpy()[jj], sub['lon_ipp'].to_numpy()[jj]
        )

        if region is not None:
            inreg = region.contains(cp_lat, cp_lon)
            if not np.any(inreg):
                continue
            grad_abs, grad_x, grad_y = grad_abs[inreg], grad_x[inreg], grad_y[inreg]
            dVTEC = dVTEC[inreg]
            sv_ii = sv_ii[inreg]
            sv_jj = sv_jj[inreg]
            name_ii = name_ii[inreg]
            name_jj = name_jj[inreg]
            cp_lat, cp_lon = cp_lat[inreg], cp_lon[inreg]
            dsel, Mi, Mj = dsel[inreg], Mi[inreg], Mj[inreg]

        if grad_abs.size == 0:
            continue

        # --- METRICS  ---
        # mean
        gx_mean = float(np.mean(grad_x)) * 1000.0  # mTECU/km
        gy_mean = float(np.mean(grad_y)) * 1000.0
        gix     = float(np.hypot(gx_mean, gy_mean))

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
            'n_pairs': int(grad_abs.size)
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

    times_df = pd.DataFrame(times).sort_values('time').reset_index(drop=True)
    pairs_df = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    return times_df, pairs_df

"""
ROTI for GNSS data (STEC/VTEC) — according to Carmo et al. (2021), Method 4 (Cherniak 2018).
DataFrame input: columns ['time','stec','vtec','lat_ipp','lon_ipp','name','sv'] (+ possibly 'ev').
- ROT = dltTEC / dltt  (TECU/min)
- ROTI = std(ROT) in a 5-minute window (population std, ddof=0)

Functions:
- compute_roti_links(...) : calculate ROT/ROTI per link (name, sv) over time, gap-resistant and north-resistant
- grid_roti_map(...)      : aggregate ROTI to a geographic grid (map)
- plot_roti_series(...)   : ROTI time series (summarized by station or per satellite)
- plot_roti_map(...)      : ROTI map (scatter-grid)
- save_roti_netcdf(...)   : save results to NetCDF (xarray)

By default, we operate on STEC (Carmo recommendation — Method 4), but you can switch to VTEC.
"""



from astropy.time import Time


def _ensure_sm_cols(
    dfa: pd.DataFrame,
    transformer,                  # gnx.SolarGeomagneticTransformer()
    shell_height_m: float = 450e3
) -> pd.DataFrame:
    """
    Adds columns 'lat_sm', 'lon_sm' based on 'lat_ipp', 'lon_ipp', 'time' (UTC).
    Performs vector transformation for each epoch.
    """
    if transformer is None:
        raise ValueError("For mode='SM', specify a transformer with the geodetic_to_sm(..., obstime=Time) method.")
    if not {'time','lat_ipp','lon_ipp'}.issubset(dfa.columns):
        raise ValueError("The columns 'time', 'lat_ipp', and 'lon_ipp' are missing..")

    dfa = dfa.copy()
    dfa['time'] = pd.to_datetime(dfa['time'], utc=True)
    dfa = dfa.reset_index(drop=True)

    lat_sm = np.empty(len(dfa)); lon_sm = np.empty(len(dfa))
    for t, block_idx in dfa.groupby('time', sort=False).groups.items():
        b = dfa.loc[block_idx]
        lat_rad = np.deg2rad(b['lat_ipp'].to_numpy())
        lon_rad = np.deg2rad(b['lon_ipp'].to_numpy())
        h      = np.full_like(lat_rad, shell_height_m, dtype=float)

        sm_lat, sm_lon, _h = transformer.geodetic_to_sm(
            lat_rad=lat_rad, lon_rad=lon_rad, h_m=h, obstime=Time(t, scale='utc')
        )
        idx = np.asarray(block_idx, dtype=int)
        lat_sm[idx] = np.rad2deg(sm_lat)
        lon_deg = (np.rad2deg(sm_lon) + 180.0) % 360.0 - 180.0
        lon_sm[idx] = lon_deg

    dfa['lat_sm'] = lat_sm
    dfa['lon_sm'] = lon_sm
    return dfa


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
      ['time','name','sv','lat_ipp','lon_ipp','TEC','ROT_tecu_per_min','ROTI_tecu_per_min']
    ROT calculated for (name, sv) and (optionally) within the same day.
    ROTI = rolling std(ROT) over time (window 'window_min'), ddof=0, min_periods='min_samples'.
    """
    need = {'time','stec','vtec','lat_ipp','lon_ipp','name','sv'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Columns missing: {sorted(miss)}")

    dfa = df.copy()
    dfa['time'] = pd.to_datetime(dfa['time'], utc=True)
    dfa = dfa.sort_values(['name','sv','time']).reset_index(drop=True)

    # elevation map
    if min_elev_deg is not None and 'ev' in dfa.columns:
        dfa = dfa[dfa['ev'] >= float(min_elev_deg)].copy()

    # TEC source
    col = 'stec' if tec_source == 'stec' else 'vtec'
    dfa['TEC'] = dfa[col].astype(float) * tec_scale

        # --- SM columns  ---
    if coord_mode == 'SM' or region_frame == 'SM':
        dfa = _ensure_sm_cols(dfa, sm_transformer, shell_height_m=sm_shell_height_m)

    # --- region filter (in selected reference frame) ---
    if region is not None:
        rf = coord_mode if region_frame == 'AUTO' else region_frame
        latv = dfa['lat_sm'] if rf == 'SM' else dfa['lat_ipp']
        lonv = dfa['lon_sm'] if rf == 'SM' else dfa['lon_ipp']
        inreg = region.contains(latv.to_numpy(), lonv.to_numpy())
        dfa = dfa[inreg].copy()

    if dfa.empty:
        return pd.DataFrame(columns=[
            'time','name','sv',
            'lat_ipp','lon_ipp','TEC','ROT_tecu_per_min','ROTI_tecu_per_min'
        ] + (['lat_sm','lon_sm'] if 'lat_sm' in dfa.columns else []))


    # division into UTC days, so as not to differentiate by midnight (daily jumps)
    if split_on_day_change:
        dfa['day'] = dfa['time'].dt.floor('D')
        grp_cols = ['name', 'sv', 'day']
    else:
        grp_cols = ['name', 'sv']
    dfa = dfa.sort_values(grp_cols + ['time'])
    if detrend_5min:
        # we calculate the rolling mean only on the columns ['time','TEC'] within each group,
        # so that apply gets a mini-DataFrame WITHOUT grouping columns
        rm5 = (
            dfa.groupby(grp_cols, group_keys=False)
               .apply(
                   lambda g: g[['time','TEC']]
                             .rolling('5min', on='time', min_periods=1, closed='both')['TEC']
                             .mean(),
                   include_groups=False
               )
        )
        # rm5 has an index of ORIGINAL LINES -> can be safely inserted into dfa
        dfa['TEC_d'] = dfa['TEC'] - rm5
        dfa['rm5'] = rm5.values

    else:
        dfa['TEC_d'] = dfa['TEC']
        dfa['rm5'] = 0.0

    # differentiation within groups (link [+ day])
    dfa['dt_s']  = dfa.groupby(grp_cols)['time'].diff().dt.total_seconds()
    dfa['dTEC']  = dfa.groupby(grp_cols)['TEC_d'].diff()

    # cut out illogical steps in time
    valid = dfa['dt_s'].notna() & (dfa['dt_s'] > 0)
    if max_gap_s is not None:
        valid &= dfa['dt_s'] <= float(max_gap_s)
    dfa.loc[~valid, ['dt_s','dTEC']] = np.nan

    # ROT [TECU/min]
    dfa['ROT_tecu_per_min'] = (dfa['dTEC'] / dfa['dt_s']) * 60.0
    dfa['ROT_mtecu_per_s'] = (dfa['dTEC'] / dfa['dt_s']) * 1000
    # jest w TECU/min

    # Rolling STD over time (time window) for each group
    # Note: we use rolling(time-based) with min_periods, ddof=0 (population definition)
    def _rolling_std(g: pd.DataFrame, col) -> pd.Series:
        g = g.set_index('time')
        # tylko kolumna ROT; lat/lon nie są potrzebne do STD
        return g[col].rolling(f'{window_min}min', min_periods=min_samples,closed='both').std(ddof=DDOF_POP)

    roti = dfa.groupby(grp_cols, group_keys=False).apply(_rolling_std,'ROT_tecu_per_min')
    roti_mtecu_s = dfa.groupby(grp_cols, group_keys=False).apply(_rolling_std, 'ROT_mtecu_per_s')
    dfa['ROTI_tecu_per_min'] = roti.values
    dfa['ROTI_mtecu_per_s'] = roti_mtecu_s.values

    out_cols = ['time','name','sv','lat_ipp','lon_ipp','TEC','ROT_mtecu_per_s','ROT_tecu_per_min','ROTI_tecu_per_min','ROTI_mtecu_per_s','rm5']
    if coord_mode == 'SM' or 'lat_sm' in dfa.columns:
        out_cols += ['lat_sm','lon_sm']
    out = dfa[out_cols].copy()
    return out.sort_values(['time','name','sv']).reset_index(drop=True)



# ------ ------ ------ ------ ------ ------ ROTI PLOTTING ------- ------ ------ ------ ------ ------
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
    Returns: DataFrame ['lon','lat','ROTI','n'] (1 point/grid cell).
    """
    if roti_df.empty:
        return pd.DataFrame(columns=['lon','lat','ROTI','n'])

    t0 = pd.to_datetime(epoch, utc=True)
    if mode == 'center':
        half = pd.Timedelta(minutes=window_min/2)
        t1, t2 = t0 - half, t0 + half
    elif mode =='trailing':  # 'trailing' – backward window
        t1, t2 = t0 - pd.Timedelta(minutes=window_min), t0
    else: #top
        t1, t2 = t0, t0 + pd.Timedelta(minutes=window_min)

    sub = roti_df[(roti_df['time'] >= t1) & (roti_df['time'] <= t2)].copy()
    sub = sub[~sub['ROTI_tecu_per_min'].isna()]
    if sub.empty:
        return pd.DataFrame(columns=['lon','lat','ROTI','n'])

    lat_col = 'lat_sm' if coord_mode == 'SM' and 'lat_sm' in sub.columns else 'lat_ipp'
    lon_col = 'lon_sm' if coord_mode == 'SM' and 'lon_sm' in sub.columns else 'lon_ipp'

    sub['ix'] = np.floor((((sub[lon_col] + 180) % 360) - 180 + 180)/res_deg).astype(int)
    sub['iy'] = np.floor((sub[lat_col] + 90)/res_deg).astype(int)

    base_agg = dict(lon=(lon_col,'mean'), lat=(lat_col,'mean'),
                    ROTI=('ROTI_tecu_per_min', {'median':'median','mean':'mean','max':'max'}.get(agg, 'median')),
                    n=('ROTI_tecu_per_min','size'))
    if agg == 'Q95':
        base_agg['ROTI'] = ("ROTI_tecu_per_min", lambda x: x.quantile(0.95))

    g = sub.groupby(['ix','iy']).agg(**base_agg).reset_index()
    return g[g['n'] >= int(min_points)].reset_index(drop=True)

# --- 2) Global scale (fixed vmin/vmax for all snapshots) ---
def compute_roti_global_scale(
    roti_df: pd.DataFrame,
    window_min: int = 5,
    q_low_high: Tuple[float,float] = (5.0, 95.0)
) -> Tuple[float, float]:
    """
    Calculates global (vmin, vmax) based on ROTI percentiles across the entire set.
    For frame comparability, it is worth calculating this after roughly filtering out NaN.
    """
    s = roti_df['ROTI_tecu_per_min'].dropna()
    if s.empty:
        return (0.0, 1.0)
    vmin = float(np.percentile(s, q_low_high[0]))
    vmax = float(np.percentile(s, q_low_high[1]))
    if vmin >= vmax:
        vmax = vmin + 1e-3
    return vmin, vmax





def _new_map_axes(proj, *, figsize=FIGSIZE, dpi=DPI):
    """Tworzy zawsze identyczny układ: ax (mapa) + cax (colorbar pod spodem)."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(2, 1, height_ratios=_GS_HEIGHT_RATIOS, hspace=_GS_HSPACE)
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    cax = fig.add_subplot(gs[1, 0])
    return fig, ax, cax


def plot_roti_snapshot(
    roti_df: pd.DataFrame,
    epoch,
    window_min: int = 5,
    mode: Literal['center','trailing'] = 'center',
    res_deg: float = 1.0,
    min_points: int = 5,
    agg: Literal['median','mean','max','Q95'] = 'median',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None,
    title_prefix: str = "ROTI",
    map_background: bool = True,
    coast_res: str = "50m",
    draw: Literal['tiles','scatter'] = 'tiles',
    coord_mode: Literal['GEO','SM'] = 'GEO',
    extent: Optional[list] = None,   # [lon_min, lon_max, lat_min, lat_max]
    cmap_name: str = "viridis",
):
    grid = roti_snapshot_grid(
        roti_df, epoch=epoch, window_min=window_min, mode=mode,
        res_deg=res_deg, min_points=min_points, agg=agg, coord_mode=coord_mode
    )
    if grid.empty:
        print("Brak punktów w oknie – pomijam rysowanie.")
        return

    # skala kolorów
    if vmin is None or vmax is None:
        vmin = float(np.percentile(grid['ROTI'], 5)) if vmin is None else vmin
        vmax = float(np.percentile(grid['ROTI'], 95)) if vmax is None else vmax
        if vmin >= vmax:
            vmax = vmin + 1e-3

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # extent (jeśli nie podany – wylicz z danych)
    lon = ((grid['lon'].to_numpy() + 180) % 360) - 180
    lat = grid['lat'].to_numpy()
    if extent is None:
        margin_lon = max(2.0, res_deg * 2)
        margin_lat = max(2.0, res_deg * 2)
        lon_min, lon_max = np.nanmin(lon) - margin_lon, np.nanmax(lon) + margin_lon
        lat_min, lat_max = max(-90, np.nanmin(lat) - margin_lat), min(90, np.nanmax(lat) + margin_lat)
        extent = [lon_min, lon_max, lat_min, lat_max]

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        proj = ccrs.PlateCarree()
        fig, ax, cax = _new_map_axes(proj)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--', alpha=0.6)

        if map_background and coord_mode == 'GEO':
            ax.add_feature(cfeature.OCEAN.with_scale(coast_res), facecolor="#dfe9f3")
            ax.add_feature(cfeature.LAND.with_scale(coast_res),  facecolor="#ececec")
            ax.add_feature(cfeature.COASTLINE.with_scale(coast_res), linewidth=0.6)
            ax.add_feature(cfeature.BORDERS.with_scale(coast_res), linewidth=0.4)

        if draw == 'tiles':
            for _, r in grid.iterrows():
                west  = r['ix'] * res_deg - 180.0
                south = r['iy'] * res_deg -  90.0
                rect = mpatches.Rectangle(
                    (west, south), res_deg, res_deg,
                    transform=ccrs.PlateCarree(),
                    facecolor=cmap(norm(r['ROTI'])),
                    edgecolor='none'
                )
                ax.add_patch(rect)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        else:
            ax.scatter(
                grid['lon'], grid['lat'],
                c=grid['ROTI'],
                s=60, marker='s',
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(),
                edgecolor='none',
                linewidths=0
            )
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        def place_cax_below(ax, cax, gap=0.006, height=0.018):
            """
            gap, height w jednostkach figury (0..1).
            """
            pos = ax.get_position()
            cax.set_position([pos.x0, pos.y0 - gap - height, pos.width, height])


        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label('ROTI [TECU/min]')
        fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08)
        place_cax_below(ax, cax, gap=0.05, height=0.018)

        ts = pd.to_datetime(epoch, utc=True)
        ax.set_title(f"{title_prefix} — {ts}  ({window_min} min window, {mode})")

        if save_path:
            fig.savefig(save_path, dpi=600)
            plt.close(fig)

    except ImportError:
        # fallback bez cartopy: identyczny layout osi danych
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        gs = gridspec.GridSpec(2, 1, height_ratios=_GS_HEIGHT_RATIOS, hspace=_GS_HSPACE)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[1, 0])

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(alpha=0.3)

        if draw == 'tiles':
            for _, r in grid.iterrows():
                west  = r['ix'] * res_deg - 180.0
                south = r['iy'] * res_deg -  90.0
                ax.add_patch(
                    plt.Rectangle((west, south), res_deg, res_deg,
                                  facecolor=cmap(norm(r['ROTI'])), edgecolor='none')
                )
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        else:
            ax.scatter(grid['lon'], grid['lat'], c=grid['ROTI'], s=60, marker='s', cmap=cmap, norm=norm)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label('ROTI [TECU/min]')

        ts = pd.to_datetime(epoch, utc=True)
        ax.set_title(f"{title_prefix} — {ts}  ({window_min} min window, {mode})")
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')

        if save_path:
            fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08)

            fig.savefig(save_path, dpi=DPI)
            # fig.savefig(
            #     save_path.split('.')[0] + '.tiff',
            #     dpi=DPI,
            #     format="tiff",
            #     pil_kwargs={"compression": "tiff_lzw"}
            # )
            plt.close(fig)


# --- 4) Generator snapshotów (pętla po epokach) ---
def make_roti_snapshots(
    roti_df: pd.DataFrame,
    epochs: Optional[pd.DatetimeIndex] = None,
    every_n: int = 1,
    window_min: int = 5,
    mode: Literal['center','trailing'] = 'center',
    res_deg: float = 1.0,
    min_points: int = 5,
    agg: Literal['median','mean','max','Q95'] = 'median',
    global_scale: bool = True,
    q_scale: Tuple[float,float] = (5.0, 95.0),
    out_dir: Optional[str] = None,
    filename_fmt: str = "roti_{:%Y%m%d_%H%M%S}.png",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # --- NOWE ---
    coord_mode: Literal['GEO','SM'] = 'GEO',
    extent=None
):
    """
    Tworzy serię snapshotów ROTI.
    - epochs: lista epok (UTC). Jeśli None → weźmie posortowane unikalne czasy z roti_df.
    - global_scale=True → stała skala kolorów (percentyle z całego zbioru).
    """
    if roti_df.empty:
        print("Brak danych ROTI."); return

    # lista epok
    if epochs is None:
        epochs = pd.to_datetime(roti_df['time'].dropna().unique())
        epochs = pd.DatetimeIndex(np.sort(epochs))
    # decymacja co 'every_n'
    epochs = epochs[::max(1, int(every_n))]

    # globalna skala
    if vmin is None or vmax is None:
        if global_scale:
            vmin, vmax = compute_roti_global_scale(roti_df, window_min=window_min, q_low_high=q_scale)


    # pętla po epokach
    for t in epochs:
        save_path_i = None
        if out_dir:
            save_path_i = f"{out_dir.rstrip('/')}/{filename_fmt.format(pd.to_datetime(t).to_pydatetime())}"
        plot_roti_snapshot(
            roti_df, epoch=t, window_min=window_min, mode=mode,
            res_deg=res_deg, min_points=min_points, agg=agg,
            vmin=vmin, vmax=vmax, save_path=save_path_i, coord_mode=coord_mode,map_background=True,extent=extent
        )





def plot_gix(
    pairs_df: pd.DataFrame,
    epoch,
    clip_outliers: bool = True,
    *,
    vmin=None,
    vmax=None,
    vclip=(0.01, 0.995),
    log_scale: bool = False,
    cmap: str = 'viridis',

    projection: str = 'PlateCarree',
    central_longitude: float = 0.0,
    extent=None,

    figsize=FIGSIZE,
    dpi=DPI,

    point_size=16,
    point_alpha=0.8,
    point_edgecolor='none',
    point_zorder=3,

    draw_coastlines=True,
    coast_resolution='110m',
    coast_linewidth=0.7,
    draw_borders=True,
    borders_linewidth=0.5,
    draw_land=True,
    land_facecolor='0.9',
    ocean_facecolor='0.95',
    draw_ocean=True,

    draw_gridlines=True,
    gridlines_kwargs=None,
    title_prefix=r"GIX |$\nabla$TEC|",
    title_kwargs=None,

    cbar_label=r'|$\nabla$TEC| [mTECU/km]',
    annotate_stats=True,
    stats_loc='lower left',
    stats_fontsize=9,
    stats_alpha=0.7,

    save_path: Optional[str] = None,
):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LogNorm

    # --- wybór epoki i czyszczenie NaN ---
    pairs_df = pairs_df.dropna(subset=['CP_lat', 'CP_lon']).copy()
    sub = pairs_df[pairs_df['time'] == epoch]
    if sub.empty:
        raise ValueError(f"Brak par dla epoki {epoch}")

    lats = sub['CP_lat'].to_numpy()
    lons = sub['CP_lon'].to_numpy()
    vals = sub['grad_abs_mtecu_km'].to_numpy()

    if clip_outliers:
        p99 = np.nanpercentile(vals, 99)
        mask = vals <= p99
        lats, lons, vals = lats[mask], lons[mask], vals[mask]

    order = np.argsort(vals)
    lats, lons, vals = lats[order], lons[order], vals[order]

    if vmin is None or vmax is None:
        lo_p, hi_p = vclip
        vmin_auto = np.nanpercentile(vals, lo_p * 100)
        vmax_auto = np.nanpercentile(vals, hi_p * 100)
        if vmin is None: vmin = float(vmin_auto)
        if vmax is None: vmax = float(vmax_auto)

    # --- projekcja ---
    proj_cls = getattr(ccrs, projection, ccrs.PlateCarree)
    proj = proj_cls(central_longitude=central_longitude) \
        if 'central_longitude' in proj_cls.__init__.__code__.co_varnames else proj_cls()

    # --- identyczny layout jak ROTI: ax + cax ---
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(2, 1, height_ratios=_GS_HEIGHT_RATIOS, hspace=_GS_HSPACE)
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    cax = fig.add_subplot(gs[1, 0])

    # extent
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # tło
    if draw_ocean:
        ax.set_facecolor(ocean_facecolor)
    if draw_land:
        ax.add_feature(cfeature.LAND.with_scale(coast_resolution),
                       facecolor=land_facecolor, edgecolor='none', zorder=0)

    # kontury
    if draw_coastlines:
        ax.coastlines(resolution=coast_resolution, linewidth=coast_linewidth, zorder=1)
    if draw_borders:
        ax.add_feature(cfeature.BORDERS.with_scale(coast_resolution),
                       linewidth=borders_linewidth, zorder=2)

    # gridlines (bez labeli -> spójny layout)
    if draw_gridlines:
        gl_kwargs = dict(draw_labels=True, linestyle='--', linewidth=0.4, alpha=0.6)
        if gridlines_kwargs is not None:
            gl_kwargs.update(gridlines_kwargs)
        ax.gridlines(**gl_kwargs)

    # norm
    if log_scale:
        positive_vals = vals[vals > 0]
        if positive_vals.size == 0:
            raise ValueError("Brak dodatnich wartości do log_scale=True.")
        if vmin <= 0:
            vmin = float(np.nanmin(positive_vals))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin_sc = vmax_sc = None
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        vmin_sc, vmax_sc = None, None

    # scatter
    sc = ax.scatter(
        lons, lats,
        c=vals,
        s=point_size,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=point_alpha,
        edgecolor=point_edgecolor,
        linewidths=0,
        zorder=point_zorder,
    )

    def place_cax_below(ax, cax, gap=0.006, height=0.018):
        """
        gap, height w jednostkach figury (0..1).
        """
        pos = ax.get_position()
        cax.set_position([pos.x0, pos.y0 - gap - height, pos.width, height])

    # colorbar zawsze w cax i zawsze horizontal
    cb = plt.colorbar(sc, cax=cax, orientation='horizontal',pad=0.02)
    cb.set_label(cbar_label)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08)
    place_cax_below(ax, cax, gap=0.05, height=0.018)

    # title
    ttl_opts = dict(fontsize=12)
    if title_kwargs is not None:
        ttl_opts.update(title_kwargs)
    ax.set_title(f"{title_prefix} — {pd.to_datetime(epoch)}", **ttl_opts)

    # stats
    if annotate_stats:
        mn = float(np.nanmin(vals))
        mx = float(np.nanmax(vals))
        med = float(np.nanmedian(vals))
        p95 = float(np.nanpercentile(vals, 95))
        text = f"min={mn:.2f}, median={med:.2f}, max={mx:.2f}\np95={p95:.2f}"

        loc2xy = {
            'lower left':  (0.02, 0.02),
            'lower right': (0.98, 0.02),
            'upper left':  (0.02, 0.98),
            'upper right': (0.98, 0.98),
        }
        x, y = loc2xy.get(stats_loc, (0.02, 0.02))

        ax.text(
            x, y, text,
            transform=ax.transAxes,
            fontsize=stats_fontsize,
            ha='left' if 'left' in stats_loc else 'right',
            va='bottom' if 'lower' in stats_loc else 'top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=stats_alpha),
            zorder=4,
        )

    if save_path:
        fig.savefig(save_path, dpi=dpi)
        # fig.savefig(
        #     save_path.split('.')[0] + '.tiff',
        #     dpi=dpi,
        #     format="tiff",
        #     pil_kwargs={"compression": "tiff_lzw"}
        # )

        plt.close(fig)

    return fig, ax, sc




def plot_gix_family(times, reg, dmin, dmax, figsize=(12,9), dpi=100):
    """
        Creates a 3-panel GIX family chart:
            (a) GIX      = sqrt(gx_bar^2 + gy_bar^2)
            (b) GIXσ     = std(|grad(VTEC)|)
            (c) GIX95    = 95%(|grad(VTEC)|)

        Parameters:
            times : DataFrame containing columns:
                - 'time'
                - 'GIX_mtecu_km'
                - 'GIXS_mtecu_km'
                - 'GIXP95_mtecu_km'
            reg   : region name (str)
            dmin, dmax : dipole length range
            figsize : figure size
            dpi : density to save
        """

    t     = times['time'].values
    gix_  = times['GIX_mtecu_km'].values
    gixs  = times['GIXS_mtecu_km'].values
    gix95 = times['GIXP95_mtecu_km'].values

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, dpi=dpi)

    # ---- Nagłówek całości ----
    fig.suptitle(
        fr'GIX family for {reg}'
        '\n Period: 2024 / 035, sampling: 30 s\n'
        fr'Dipole length range: {dmin}-{dmax} km',
        fontsize=14,
        fontweight='bold'
    )

    panel_titles = [
        r'(a) $GIX$',
        r'(b) $GIX_\sigma$',
        r'(c) $GIX_{95}$'
    ]

    # ------------- PANEL 1: GIX -------------
    ax = axes[0]
    ax.plot(t, gix_, lw=0.7, color='tab:green', alpha=0.9)
    ax.scatter(t, gix_, s=4, color='tab:green', alpha=0.4, edgecolor='none')
    ax.set_title(panel_titles[0], loc='left', fontsize=11)
    ax.set_ylabel(r'$GIX$ [mTECU/km]', fontsize=11)
    ax.legend(
        [r'$GIX = \sqrt{\bar{g}_x^{\,2} + \bar{g}_y^{\,2}}$'],
        loc='upper right', fontsize=10, frameon=True
    )

    # ------------- PANEL 2: GIX_sigma -------------
    ax = axes[1]
    ax.plot(t, gixs, lw=0.7, color='tab:red', alpha=0.9)
    ax.scatter(t, gixs, s=4, color='tab:red', alpha=0.4, edgecolor='none')
    ax.set_title(panel_titles[1], loc='left', fontsize=11)
    ax.set_ylabel(r'$GIX_\sigma$ [mTECU/km]', fontsize=11)
    ax.legend(
        [r'$GIX_\sigma = \sigma\!\left(|\nabla \mathrm{VTEC}|\right)$'],
        loc='upper right', fontsize=10, frameon=True
    )

    # ------------- PANEL 3: GIX_95 -------------
    ax = axes[2]
    ax.plot(t, gix95, lw=0.7, color='tab:blue', alpha=0.9)
    ax.scatter(t, gix95, s=4, color='tab:blue', alpha=0.4, edgecolor='none')
    ax.set_title(panel_titles[2], loc='left', fontsize=11)
    ax.set_ylabel(r'$GIX_{95}$ [mTECU/km]', fontsize=11)
    ax.set_xlabel('Time [GPST]', fontsize=11)
    ax.legend(
        [r'$GIX_{95} = Q_{95}\!\left(|\nabla \mathrm{VTEC}|\right)$'],
        loc='upper right', fontsize=10, frameon=True
    )

    # ---- Layout formatting ----
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    return fig, axes
