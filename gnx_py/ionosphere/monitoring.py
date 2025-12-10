from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Optional, Literal, Tuple
from typing import Union

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.dates import DateFormatter
# --- SIDX, GIX ---
Re = 6371e3
IONO_H = 450e3
MIN_ELEV_DEG = 10.0
DMIN_KM = 30.0
DMAX_KM = 50.0
MAX_PAIR_PER_EPOCH = 200000  # bezpiecznik losowego przerzedzania
# --- ROT/ROTI ---
ROTI_WINDOW_MIN = 5            # okno dla STD
ROTI_MIN_SAMPLES = 3           # min. liczba ROT w oknie, żeby wyliczyć STD
MAX_GAP_S       = 180.0        # odetnij skoki przy dużych lukach czasowych
SPLIT_BY_DAY    = True         # nie różnicować przez północ UTC (reset per doba)
DDOF_POP        = 0            # zgodnie z def. ROTI = sqrt(<ROT^2>-<ROT>^2)


@dataclass
class Region:
    lat_min: float = -90.0
    lat_max: float =  90.0
    lon_min: float = -180.0
    lon_max: float =  180.0
    def contains(self, lat, lon) -> np.ndarray:
        lon = ((lon + 180) % 360) - 180
        return (lat >= self.lat_min) & (lat <= self.lat_max) & \
               (lon >= self.lon_min) & (lon <= self.lon_max)

# --- Narzędzia geometryczne ---
def mapping_M(elev_deg: np.ndarray, Re_m: float = Re, H_m: float = IONO_H) -> np.ndarray:
    el = np.deg2rad(np.clip(elev_deg, 0.01, 89.99))
    a = (Re_m * np.cos(el)) / (Re_m + H_m)
    return 1.0 / np.sqrt(1.0 - a * a)

def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_m: float) -> np.ndarray:
    phi1, lbd1, phi2, lbd2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dphi, dlbd = phi2 - phi1, lbd2 - lbd1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlbd/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return (radius_m * c) / 1000.0

def initial_bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    phi1, lbd1, phi2, lbd2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlbd = lbd2 - lbd1
    y = np.sin(dlbd) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlbd)
    theta = np.arctan2(y, x)
    return (np.rad2deg(theta) + 360) % 360

def spherical_midpoint(lat1, lon1, lat2, lon2) -> Tuple[np.ndarray, np.ndarray]:
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
    max_gap_s: Optional[float] = 300,      # odrzuć różnice, gdy przerwa > max_gap_s
    split_on_day_change: bool = True,        # NIE licz różnic przez północ (UTC)
    robust_sigma_clip: Optional[float] = 5.0,# klip po epoce w oparciu o MAD (None=wyłącz)
    agg: str = "median",              # 'median' lub 'mean'
) -> pd.Series:
    """
    SIDX ≈ < dltSTEC / (M·dltt) >  (TECU/min).
    Różnicowanie tylko w obrębie tego samego linku (name, sv) i — jeśli włączone —
    w obrębie tej samej doby UTC (brak różnic przez północ).
    Wymagane kolumny: name, sv, time, stec, ev, lat_ipp, lon_ipp.
    """
    need = {'name','sv','time','stec','ev','lat_ipp','lon_ipp'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Brakuje kolumn: {sorted(miss)}")

    dfa = df.copy()
    dfa['time'] = pd.to_datetime(dfa['time'], utc=True)

    # filtr elewacji
    dfa = dfa[dfa['ev'] >= min_elev_deg].copy()
    if dfa.empty:
        print(f'No data after elevation mask application')
        return pd.Series(dtype=float, name='SIDX_tecu_per_min')

    # jednostki i mapping
    dfa['stec_tecu'] = dfa['stec'] * stec_scale
    dfa['M'] = dfa['M'] if 'M' in dfa.columns else mapping_M(dfa['ev'].to_numpy(), H_m=iono_h_m)

    # identyfikacja doby UTC do podziału łuków
    if split_on_day_change:
        dfa['day'] = dfa['time'].dt.floor('D')  # UTC doba
        grp_cols = ['name','sv','day']
    else:
        grp_cols = ['name','sv']

    # różnicowanie wewnątrz grup (link + ewentualnie doba)
    dfa = dfa.sort_values(grp_cols + ['time']).reset_index(drop=True)
    dfa['dt_s']  = dfa.groupby(grp_cols)['time'].diff().dt.total_seconds()
    dfa['dSTEC'] = dfa.groupby(grp_cols)['stec_tecu'].diff()

    rate = dfa.dropna(subset=['dt_s','dSTEC','M'])
    rate = rate[rate['dt_s'] > 0]

    # odrzuć duże przerwy czasowe
    if max_gap_s is not None:
        rate = rate[rate['dt_s'] <= float(max_gap_s)]

    # maska regionu IPP (jeśli podano)
    if region is not None:
        inreg = region.contains(rate['lat_ipp'].to_numpy(), rate['lon_ipp'].to_numpy())
        rate = rate[inreg]

    # wskaźnik chwilowy dla pojedynczych linków
    rate['sid_link'] = rate['dSTEC'] / (rate['M'] * rate['dt_s'])  # TECU/s

    # ROBUST: sigma-clipping po epoce (MAD)
    if robust_sigma_clip is not None:
        k = float(robust_sigma_clip)
        def _clip_epoch(g):
            if len(g) < 5:
                return g
            med = g['sid_link'].median()
            mad = 1.4826 * np.median(np.abs(g['sid_link'] - med))
            if mad == 0 or not np.isfinite(mad):
                return g
            return g[np.abs(g['sid_link'] - med) <= k * mad]
        rate = rate.groupby('time', group_keys=False).apply(_clip_epoch)

    # agregacja po epoce (średnia/mediana po linkach)
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


    # skalowanie jednostek → TECU/min
    sid = sid * 1e3 #* 60
    sid.name = 'SIDX_tecu_per_min'
    return sid

# --- GIX ---


# --- pomocniczo: warstwowe przerzedzanie po Δs i azymucie ---
def _stratified_subsample(mask_idx: np.ndarray, d_km: np.ndarray, az_deg: np.ndarray,
                          target: int, n_dist_bins: int = 4, n_az_bins: int = 8,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Zwraca podzbiór indeksów mask_idx (int) w liczbie <= target,
    próbkując równomiernie z koszyków 2D: Δs×azymut.
    """
    if rng is None:
        rng = np.random.default_rng()
    if mask_idx.size <= target:
        return mask_idx

    dsel = d_km[mask_idx]
    azsel = az_deg[mask_idx]

    # koszyki
    d_bins = np.quantile(dsel, np.linspace(0, 1, n_dist_bins+1))
    # aby uniknąć duplikatów granic przy małej zmienności
    d_bins = np.unique(d_bins)
    if d_bins.size <= 2:  # fallback
        d_bins = np.array([dsel.min(), dsel.max()])
    az_bins = np.linspace(0, 360, n_az_bins+1)

    d_cat = np.clip(np.digitize(dsel, d_bins, right=True)-1, 0, len(d_bins)-2)
    az_cat = np.clip(np.digitize(azsel, az_bins, right=True)-1, 0, len(az_bins)-2)

    # docelowo po równo na koszyk
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

    # jeśli jeszcze brakuje do target, dobierz losowo z pozostałych
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
# --- GIX „1:1”: dodane GIXSx/GIXSy i percentyle kierunkowe ± ---
def compute_gix(
    df: pd.DataFrame,
    region: Optional[Region] = None,
    min_elev_deg: float = MIN_ELEV_DEG,
    dipole_preset: str = "30-250",          # "30-250" | "50-500" | "50-1000"
    dmin_km: Optional[float] = None,        # nadpisze preset, jeśli podane
    dmax_km: Optional[float] = None,
    iono_h_m: float = IONO_H,
    stec_scale: float = 1.0,
    max_delta_M: float = 0.2,               # filtr |Mi−Mj|
    max_pairs_per_epoch: int = MAX_PAIR_PER_EPOCH,
    stratified_sampling: bool = True,       # warstwowe przerzedzanie Δs×azymut
    rng: Optional[np.random.Generator] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wejście (wymagane kolumny): name,time,lat_ipp,lon_ipp,stec,ev,(opcjonalnie M).
    Zwraca:
      - times_df (1 rekord/epoka) z polami:
        ['time',
         'GIX_mtecu_km',
         'GIXx_mean_mtecu_km','GIXy_mean_mtecu_km',
         'GIXS_mtecu_km',       # std(|∇TEC|), jak w artykule – wariant na modułach
         'GIXSx_mtecu_km','GIXSy_mtecu_km',  # std składowych podpisanych
         'GIXPx95_plus_mtecu_km','GIXPx95_minus_mtecu_km',
         'GIXPy95_plus_mtecu_km','GIXPy95_minus_mtecu_km',
         'GIXP95_mtecu_km',     # p95 z |∇TEC|
         'n_pairs']
      - pairs_df (każda para) z polami jak u Ciebie (CP_lat/lon, grad_x/y/abs, Δs, Mi, Mj)
    """
    need = {'name','time','lat_ipp','lon_ipp','stec','ev'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Brakuje kolumn: {sorted(miss)}")

    if dipole_preset not in _DIP_PRESETS and (dmin_km is None or dmax_km is None):
        raise ValueError("Podaj poprawny dipole_preset ('30-250'|'50-500'|'50-1000') "
                         "albo jawnie dmin_km i dmax_km.")

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

        # odległości dipoli (w powłoce)
        d_km_all = great_circle_distance_km(
            sub['lat_ipp'].to_numpy()[ii], sub['lon_ipp'].to_numpy()[ii],
            sub['lat_ipp'].to_numpy()[jj], sub['lon_ipp'].to_numpy()[jj],
            radius_m=Re + iono_h_m
        )

        # azymut par (kierunek j->i)
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

        # przerzedzanie: zwykłe lub warstwowe Δs×azymut
        if idx.size > max_pairs_per_epoch:
            if stratified_sampling:
                idx = _stratified_subsample(idx, d_km_all, az_all,
                                            target=max_pairs_per_epoch, rng=rng)
            else:
                idx = rng.choice(idx, size=max_pairs_per_epoch, replace=False)

        # finalny wybór
        ii, jj = ii[idx], jj[idx]
        dsel   = d_km_all[idx]
        Mi, Mj = Mi_all[idx], Mj_all[idx]
        az     = az_all[idx]

        # gradient (TECU/km) – ze znakiem
        dVTEC = sub['VTEC'].to_numpy()[ii] - sub['VTEC'].to_numpy()[jj]
        sv_ii, sv_jj = sub['sv'].to_numpy()[ii], sub['sv'].to_numpy()[jj]
        name_ii, name_jj = sub['name'].to_numpy()[ii], sub['name'].to_numpy()[jj]
        grad_signed = dVTEC / dsel
        grad_abs    = np.abs(grad_signed)

        # rozkład na składowe x/y (WE/NS) – podpisane
        sindlt, cosdlt = np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))
        grad_x = grad_signed * sindlt # E-W
        grad_y = grad_signed * cosdlt # N-S

        # środek pary (do maski regionu i wizualizacji)
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

        # --- METRYKI 1:1 ---
        # średnia wektorowa (eq. 12)
        gx_mean = float(np.mean(grad_x)) * 1000.0  # mTECU/km
        gy_mean = float(np.mean(grad_y)) * 1000.0
        gix     = float(np.hypot(gx_mean, gy_mean))

        # dyspersja GIXS:
        # (a) std na modułach |∇TEC|
        gixs_abs = float(np.std(grad_abs)) * 1000.0
        # (b) std składowych podpisanych (kierunkowa dyspersja)
        gixs_x   = float(np.std(grad_x)) * 1000.0
        gixs_y   = float(np.std(grad_y)) * 1000.0

        # percentyle kierunkowe ± (osobno dla znaków)
        def _p95_signed(arr: np.ndarray) -> Tuple[float, float]:
            pos = arr[arr > 0]
            neg = arr[arr < 0]
            p_pos = float(np.percentile(pos, 95)) if pos.size else 0.0
            p_neg = float(np.percentile(neg, 5))  if neg.size else 0.0  # „95% najmniejszych” < 0
            return p_pos*1000.0, p_neg*1000.0

        px95_plus,  px95_minus  = _p95_signed(grad_x)
        py95_plus,  py95_minus  = _p95_signed(grad_y)

        # p95 z modułów (globalny)
        gixp95 = float(np.percentile(grad_abs, 95)) * 1000.0

        times.append({
            'time': epoch,
            'GIX_mtecu_km': gix,
            'GIXx_mean_mtecu_km': gx_mean,
            'GIXy_mean_mtecu_km': gy_mean,
            'GIXS_mtecu_km': gixs_abs,      # std(|∇TEC|)
            'GIXSx_mtecu_km': gixs_x,       # std(∇TEC_x)
            'GIXSy_mtecu_km': gixs_y,       # std(∇TEC_y)
            'GIXPx95_plus_mtecu_km':  px95_plus,   # 95. perc. dodatnich w x
            'GIXPx95_minus_mtecu_km': px95_minus,  # 5. perc. (wart. ujemne) w x
            'GIXPy95_plus_mtecu_km':  py95_plus,
            'GIXPy95_minus_mtecu_km': py95_minus,
            'GIXP95_mtecu_km': gixp95,      # 95. perc. z |∇TEC|
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

# --- plotting ---
# --- Agregacja do siatki (quiver-grid) ---
def grid_mean_vectors(pairs_df: pd.DataFrame, time_sel, res_deg=1.0, min_pairs=20) -> pd.DataFrame:
    """
    Agreguje pary do komórek siatki (1 strzałka/komórkę).
    Zwraca kolumny: lon, lat, gx, gy, gabs, n
    """
    sub = pairs_df[pairs_df['time'] == pd.to_datetime(time_sel, utc=True)].copy()
    if sub.empty:
        return pd.DataFrame(columns=['lon','lat','gx','gy','gabs','n'])
    sub['ix'] = np.floor((sub['CP_lon']+180)/res_deg).astype(int)
    sub['iy'] = np.floor((sub['CP_lat']+ 90)/res_deg).astype(int)
    g = sub.groupby(['ix','iy'])
    agg = g.agg(
        lon = ('CP_lon','mean'),
        lat = ('CP_lat','mean'),
        gx  = ('grad_x_mtecu_km','mean'),     # signed mean
        gy  = ('grad_y_mtecu_km','mean'),     # signed mean
        gabs=('grad_abs_mtecu_km','median'),  # robust kolor
        n   = ('grad_x_mtecu_km','size')
    ).reset_index(drop=True)
    return agg[agg['n'] >= int(min_pairs)].reset_index(drop=True)

# --- helper: pewność, że 'time' to datetime64[ns, UTC] ---
def _ensure_time_utc(series):
    t = pd.to_datetime(series, utc=True)
    # dropna żeby uniknąć problemów z rysowaniem
    return t

"""
ROTI dla danych GNSS (STEC/VTEC) — zgodnie z Carmo et al. (2021), Metoda 4 (Cherniak 2018).
Wejście DataFrame: kolumny ['time','stec','vtec','lat_ipp','lon_ipp','name','sv'] (+ ewentualnie 'ev').
- ROT = dltTEC / dltt  (TECU/min)
- ROTI = std(ROT) w oknie 5 minut (population std, ddof=0)

Funkcje:
- compute_roti_links(...) : licz ROT/ROTI per link (name, sv) w czasie, odporne na luki i północ
- grid_roti_map(...)      : zagreguj ROTI do siatki geograficznej (mapa)
- plot_roti_series(...)   : seria czasowa ROTI (sumarycznie po stacji lub per satelita)
- plot_roti_map(...)      : mapa ROTI (scatter-grid)
- save_roti_netcdf(...)   : zapis wyników do NetCDF (xarray)

Domyślnie operujemy na STEC (rekomendacja Carmo — Metoda 4), ale można przełączyć na VTEC.
"""


# ---------- Parametry domyślne ----------


from astropy.time import Time


def _ensure_sm_cols(
    dfa: pd.DataFrame,
    transformer,                  # np. SolarGeomagneticTransformer()
    shell_height_m: float = 450e3
) -> pd.DataFrame:
    """
    Dodaje kolumny 'lat_sm','lon_sm' na podstawie 'lat_ipp','lon_ipp','time' (UTC).
    Wykonuje transformację wektorowo dla każdej epoki.
    """
    if transformer is None:
        raise ValueError("Dla mode='SM' podaj transformer z metodą geodetic_to_sm(..., obstime=Time).")
    if not {'time','lat_ipp','lon_ipp'}.issubset(dfa.columns):
        raise ValueError("Brakuje kolumn: 'time','lat_ipp','lon_ipp'.")

    dfa = dfa.copy()
    dfa['time'] = pd.to_datetime(dfa['time'], utc=True)
    dfa = dfa.reset_index(drop=True)   # <-- KLUCZOWE

    lat_sm = np.empty(len(dfa)); lon_sm = np.empty(len(dfa))
    for t, block_idx in dfa.groupby('time', sort=False).groups.items():
        b = dfa.loc[block_idx]           # teraz block_idx == pozycje 0..N-1
        lat_rad = np.deg2rad(b['lat_ipp'].to_numpy())
        lon_rad = np.deg2rad(b['lon_ipp'].to_numpy())
        h      = np.full_like(lat_rad, shell_height_m, dtype=float)

        sm_lat, sm_lon, _h = transformer.geodetic_to_sm(
            lat_rad=lat_rad, lon_rad=lon_rad, h_m=h, obstime=Time(t, scale='utc')
        )
        idx = np.asarray(block_idx, dtype=int)   # jawnie jako pozycje
        lat_sm[idx] = np.rad2deg(sm_lat)
        lon_deg = (np.rad2deg(sm_lon) + 180.0) % 360.0 - 180.0
        lon_sm[idx] = lon_deg

    dfa['lat_sm'] = lat_sm
    dfa['lon_sm'] = lon_sm
    return dfa


# ---------- Główna funkcja: ROT/ROTI per link ----------
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
    sm_transformer = None,                 # obiekt z geodetic_to_sm(...)
    sm_shell_height_m: float = 450e3,
    region_frame: Literal['GEO','SM','AUTO'] = 'AUTO',  # w której ramie stosować 'region'
) -> pd.DataFrame:
    """
    Zwraca DataFrame z kolumnami:
      ['time','name','sv','lat_ipp','lon_ipp','TEC','ROT_tecu_per_min','ROTI_tecu_per_min']
    ROT liczony dla (name, sv) i (opcjonalnie) w obrębie tej samej doby.
    ROTI = rolling std(ROT) po czasie (okno 'window_min'), ddof=0, min_periods='min_samples'.
    """
    need = {'time','stec','vtec','lat_ipp','lon_ipp','name','sv'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Brakuje kolumn: {sorted(miss)}")

    dfa = df.copy()
    dfa['time'] = pd.to_datetime(dfa['time'], utc=True)
    dfa = dfa.sort_values(['name','sv','time']).reset_index(drop=True)

    # filtr elewacji (jeśli dostępna)
    if min_elev_deg is not None and 'ev' in dfa.columns:
        dfa = dfa[dfa['ev'] >= float(min_elev_deg)].copy()

    # wybór źródła TEC
    col = 'stec' if tec_source == 'stec' else 'vtec'
    dfa['TEC'] = dfa[col].astype(float) * tec_scale

        # --- SM kolumny (opcjonalnie) ---
    if coord_mode == 'SM' or region_frame == 'SM':
        dfa = _ensure_sm_cols(dfa, sm_transformer, shell_height_m=sm_shell_height_m)

    # --- filtr regionu (w wybranej ramie) ---
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


    # podział na doby UTC, aby nie różnicować przez północ (skoki dzienne)
    if split_on_day_change:
        dfa['day'] = dfa['time'].dt.floor('D')
        grp_cols = ['name', 'sv', 'day']
    else:
        grp_cols = ['name', 'sv']
    dfa = dfa.sort_values(grp_cols + ['time'])
    if detrend_5min:
        # liczymy rolling mean tylko na kolumnach ['time','TEC'] wewnątrz każdej grupy,
        # dzięki czemu apply dostaje mini-DataFrame BEZ kolumn grupujących
        rm5 = (
            dfa.groupby(grp_cols, group_keys=False)
               .apply(
                   lambda g: g[['time','TEC']]
                             .rolling('5min', on='time', min_periods=1, closed='both')['TEC']
                             .mean(),
                   include_groups=False  # pandas>=2.2; jeśli masz starszego, zobacz wariant B poniżej
               )
        )
        # rm5 ma indeks ORYGINALNYCH WIERSZY -> można bezpiecznie wstawić do dfa
        dfa['TEC_d'] = dfa['TEC'] - rm5
        dfa['rm5'] = rm5.values

    else:
        dfa['TEC_d'] = dfa['TEC']
        dfa['rm5'] = 0.0

    # różnicowanie w obrębie grup (link [+ doba])
    dfa['dt_s']  = dfa.groupby(grp_cols)['time'].diff().dt.total_seconds()
    dfa['dTEC']  = dfa.groupby(grp_cols)['TEC_d'].diff()

    # obetnij nielogiczne kroki czasu
    valid = dfa['dt_s'].notna() & (dfa['dt_s'] > 0)
    if max_gap_s is not None:
        valid &= dfa['dt_s'] <= float(max_gap_s)
    dfa.loc[~valid, ['dt_s','dTEC']] = np.nan

    # ROT [TECU/min]
    dfa['ROT_tecu_per_min'] = (dfa['dTEC'] / dfa['dt_s']) * 60.0
    dfa['ROT_mtecu_per_s'] = (dfa['dTEC'] / dfa['dt_s']) * 1000
    # jest w TECU/min
    # zeby zrobic mTECU/s - pomnozyc przez 1000 i  NIE MNOZYC przez 60

    # Rolling STD po czasie (okno czasowe) dla każdej grupy
    # Uwaga: używamy rolling(time-based) z min_periods, ddof=0 (definicja populacyjna)
    def _rolling_std(g: pd.DataFrame, col) -> pd.Series:
        g = g.set_index('time')
        # tylko kolumna ROT; lat/lon nie są potrzebne do STD
        return g[col].rolling(f'{window_min}min', min_periods=min_samples,closed='both').std(ddof=DDOF_POP)

    roti = dfa.groupby(grp_cols, group_keys=False).apply(_rolling_std,'ROT_tecu_per_min')
    roti_mtecu_s = dfa.groupby(grp_cols, group_keys=False).apply(_rolling_std, 'ROT_mtecu_per_s')
    # roti = (
    # dfa.groupby(grp_cols, group_keys=False)[['time','ROT_tecu_per_min']]
    #    .apply(_rolling_std)
    # )
    dfa['ROTI_tecu_per_min'] = roti.values
    dfa['ROTI_mtecu_per_s'] = roti_mtecu_s.values

    out_cols = ['time','name','sv','lat_ipp','lon_ipp','TEC','ROT_mtecu_per_s','ROT_tecu_per_min','ROTI_tecu_per_min','ROTI_mtecu_per_s','rm5']
    if coord_mode == 'SM' or 'lat_sm' in dfa.columns:
        out_cols += ['lat_sm','lon_sm']
    out = dfa[out_cols].copy()
    return out.sort_values(['time','name','sv']).reset_index(drop=True)



# ------ ------ ------ ------ ------ ------ ROTI PLOTTING ------- ------ ------ ------ ------ ------
# --- 1) GRID dla wskazanej epoki (snapshot) ---
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
    Agreguje ROTI do siatki w oknie czasowym wokół (lub przed) 'epoch'.
    Zwraca: DataFrame ['lon','lat','ROTI','n'] (po 1 punkcie/komórkę siatki).
    """
    if roti_df.empty:
        return pd.DataFrame(columns=['lon','lat','ROTI','n'])

    t0 = pd.to_datetime(epoch, utc=True)
    if mode == 'center':
        half = pd.Timedelta(minutes=window_min/2)
        t1, t2 = t0 - half, t0 + half
    elif mode =='trailing':  # 'trailing' – okno wsteczne
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

# --- 2) Skala globalna (stałe vmin/vmax dla wszystkich snapshotów) ---
def compute_roti_global_scale(
    roti_df: pd.DataFrame,
    window_min: int = 5,
    q_low_high: Tuple[float,float] = (5.0, 95.0)
) -> Tuple[float, float]:
    """
    Liczy globalne (vmin, vmax) oparte na percentylach ROTI w całym zbiorze.
    Dla porównywalności klatek warto to policzyć po zgrubnym odfiltrowaniu NaN.
    """
    s = roti_df['ROTI_tecu_per_min'].dropna()
    if s.empty:
        return (0.0, 1.0)
    vmin = float(np.percentile(s, q_low_high[0]))
    vmax = float(np.percentile(s, q_low_high[1]))
    if vmin >= vmax:
        vmax = vmin + 1e-3
    return vmin, vmax



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
    # --- NOWE ---
    coord_mode: Literal['GEO','SM'] = 'GEO'
):
    grid = roti_snapshot_grid(
        roti_df, epoch=epoch, window_min=window_min, mode=mode,
        res_deg=res_deg, min_points=min_points, agg=agg, coord_mode=coord_mode
    )
    if grid.empty:
        print("Brak punktów w oknie – pomijam rysowanie."); return

    # skala kolorów
    if vmin is None or vmax is None:
        vmin = float(np.percentile(grid['ROTI'], 5)) if vmin is None else vmin
        vmax = float(np.percentile(grid['ROTI'], 95)) if vmax is None else vmax
        if vmin >= vmax: vmax = vmin + 1e-3
    cmap = plt.get_cmap('viridis'); norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # obrys mapy
    lon = ((grid['lon'].to_numpy() + 180) % 360) - 180
    lat = grid['lat'].to_numpy()
    margin_lon = max(2.0, res_deg*2); margin_lat = max(2.0, res_deg*2)
    lon_min, lon_max = np.nanmin(lon)-margin_lon, np.nanmax(lon)+margin_lon
    lat_min, lat_max = max(-90, np.nanmin(lat)-margin_lat), min(90, np.nanmax(lat)+margin_lat)

    # rysowanie z Cartopy (tło opcjonalne)
    try:
        import cartopy.crs as ccrs, cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 5.5))
        ax = plt.axes(projection=proj)

        if map_background and coord_mode == 'GEO':
            ax.add_feature(cfeature.OCEAN.with_scale(coast_res), facecolor="#dfe9f3")
            ax.add_feature(cfeature.LAND.with_scale(coast_res),  facecolor="#ececec")
            ax.add_feature(cfeature.COASTLINE.with_scale(coast_res), linewidth=0.6)
            ax.add_feature(cfeature.BORDERS.with_scale(coast_res), linewidth=0.4)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False

        if draw == 'tiles':
            # kafle: prostokąty o wymiarach res_deg x res_deg (bez nachodzenia)
            for _, r in grid.iterrows():
                # krawędzie komórki z indeksów ix, iy
                west  = r['ix']*res_deg - 180.0
                south = r['iy']*res_deg -  90.0
                color = cmap(norm(r['ROTI']))
                rect = mpatches.Rectangle((west, south), res_deg, res_deg,
                                          transform=proj, facecolor=color, edgecolor='none')
                ax.add_patch(rect)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, label='ROTI [TECU/min]')
        else:
            # fallback: punktowe kwadraty (mogą lekko nachodzić przy małym res_deg)
            sc = ax.scatter(grid['lon'], grid['lat'], c=grid['ROTI'], s=60, marker='s',
                            vmin=vmin, vmax=vmax, transform=proj)
            cb = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, label='ROTI [TECU/min]')

        ax.set_title(f"{title_prefix} ({coord_mode}) — {pd.to_datetime(epoch, utc=True)}  (okno {window_min} min, {mode})")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150); plt.close(fig)
        else: plt.show()
    except ImportError:
        # bez Cartopy – narysujemy „tiles” w osi danych (też się nie będą nachodzić)
        plt.figure(figsize=(9,5))
        if draw == 'tiles':
            for _, r in grid.iterrows():
                west  = r['ix']*res_deg - 180.0
                south = r['iy']*res_deg -  90.0
                color = cmap(norm(r['ROTI']))
                rect = plt.Rectangle((west, south), res_deg, res_deg, facecolor=color, edgecolor='none')
                plt.gca().add_patch(rect)
            plt.xlim(lon_min, lon_max); plt.ylim(lat_min, lat_max)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            plt.colorbar(sm, label='ROTI [TECU/min]')
        else:
            sc = plt.scatter(grid['lon'], grid['lat'], c=grid['ROTI'], s=60, marker='s', vmin=vmin, vmax=vmax)
            plt.colorbar(sc, label='ROTI [TECU/min]')
        plt.title(f"{title_prefix} — {pd.to_datetime(epoch, utc=True)}  (okno {window_min} min, {mode})")
        plt.xlabel('lon'); plt.ylabel('lat'); plt.grid(alpha=0.3); plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150); plt.close()
        else: plt.show()


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
    coord_mode: Literal['GEO','SM'] = 'GEO'
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
            vmin=vmin, vmax=vmax, save_path=save_path_i, coord_mode=coord_mode,map_background=True
        )

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LogNorm


def plot_gix(
    pairs_df,
    epoch,
    clip_outliers = True,
    *,
    # --- dane / skala ---
    vmin=None,
    vmax=None,
    vclip=(0.01, 0.995),          # percentyle do automatycznego przycięcia
    log_scale=False,             # log10 skala dla kolorów
    cmap='viridis',

    # --- projekcja i zasięg mapy ---
    projection='PlateCarree',    # 'PlateCarree', 'Robinson', 'Mollweide', ...
    central_longitude=0.0,
    extent=None,                 # [lon_min, lon_max, lat_min, lat_max]

    # --- figura / oś ---
    ax=None,                     # możesz podać istniejący ax, np. do subplotów
    figsize=(10, 7),

    # --- punkty (scatter) ---
    point_size=16,
    point_alpha=0.8,
    point_edgecolor='none',
    point_zorder=3,

    # --- tło mapy ---
    draw_coastlines=True,
    coast_resolution='110m',
    coast_linewidth=0.7,
    draw_borders=True,
    borders_linewidth=0.5,
    draw_land=True,
    land_facecolor='0.9',
    ocean_facecolor='0.95',      # jednolite tło oceanu
    draw_ocean=True,

    # --- siatka / opisy ---
    draw_gridlines=True,
    gridlines_kwargs=None,
    title_prefix=r"GIX |$\nabla$TEC|",
    title_kwargs=None,

    # --- colorbar ---
    #r"$\nabla$TEC"
    cbar_label=r'|$\nabla$TEC| [mTECU/km]',
    cbar_orientation='vertical',
    cbar_pad=0.02,
    cbar_kwargs=None,

    # --- adnotacje statystyk ---
    annotate_stats=True,
    stats_loc='lower left',      # 'lower left', 'lower right', 'upper left', ...
    stats_fontsize=9,
    stats_alpha=0.7,
):
    """
    Wykres punktowy GIX dla jednej epoki z dużą kontrolą estetyki.
    Zwraca (fig, ax, sc) – figurę, oś i obiekt scatter.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in create_collection",
        category=RuntimeWarning,
        module="shapely.creation"
    )

    # --- wybór epoki i czyszczenie NaN ---
    pairs_df = pairs_df.dropna(subset=['CP_lat', 'CP_lon']).copy()
    sub = pairs_df[pairs_df['time'] == epoch]
    if sub.empty:
        raise ValueError(f"Brak par dla epoki {epoch}")

    lats = sub['CP_lat'].to_numpy()
    lons = sub['CP_lon'].to_numpy()
    vals = sub['grad_abs_mtecu_km'].to_numpy()

    # --- usuwanie outlierów powyżej 99 percentyla ---
    if clip_outliers:
        p99 = np.nanpercentile(vals, 99)
        mask = vals <= p99

        lats = lats[mask]
        lons = lons[mask]
        vals = vals[mask]

    # --- sortowanie po wartości gradientu (rosnąco) ---
    # dzięki temu najmniejsze wartości są rysowane jako pierwsze,
    # a „gorące” punkty ładnie przebijają się na wierzch
    order = np.argsort(vals)
    lats = lats[order]
    lons = lons[order]
    vals = vals[order]

    # --- automatyczne przycinanie zakresu (percentyle) ---
    # jeśli vmin/vmax nie zostały podane, używamy percentyli z vclip
    if vmin is None or vmax is None:
        lo_p, hi_p = vclip
        vmin_auto = np.nanpercentile(vals, lo_p * 100)
        vmax_auto = np.nanpercentile(vals, hi_p * 100)

        if vmin is None:
            vmin = float(vmin_auto)
        if vmax is None:
            vmax = float(vmax_auto)

    # --- wybór projekcji ---
    if isinstance(projection, str):
        proj_cls = getattr(ccrs, projection, ccrs.PlateCarree)
        proj = proj_cls(central_longitude=central_longitude) \
               if 'central_longitude' in proj_cls.__init__.__code__.co_varnames \
               else proj_cls()
    else:
        # użytkownik może podać już gotowy obiekt projekcji
        proj = projection

    # --- tworzenie figury / osi ---
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
    else:
        fig = ax.figure

    # --- zasięg mapy ---
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # --- tło ocean/land ---
    if draw_ocean:
        ax.set_facecolor(ocean_facecolor)
    if draw_land:
        land = cfeature.LAND.with_scale(coast_resolution)
        ax.add_feature(land, facecolor=land_facecolor, edgecolor='none', zorder=0)

    # --- kontury / granice ---
    if draw_coastlines:
        ax.coastlines(resolution=coast_resolution, linewidth=coast_linewidth, zorder=1)
    if draw_borders:
        borders = cfeature.BORDERS.with_scale(coast_resolution)
        ax.add_feature(borders, linewidth=borders_linewidth, zorder=2)

    # --- siatka geograficzna ---
    if draw_gridlines:
        gl_kwargs = dict(
            draw_labels=True,
            linestyle='--',
            linewidth=0.4,
            alpha=0.6,
        )
        if gridlines_kwargs is not None:
            gl_kwargs.update(gridlines_kwargs)
        gl = ax.gridlines(**gl_kwargs)
        # w cartopy >=0.20 atrybuty label_* mogą mieć inne nazwy – dopasuj w razie czego

    # --- przygotowanie normalizacji kolorów ---
    if log_scale:
        # minimalna wartość > 0, żeby log10 miało sens
        positive_vals = vals[vals > 0]
        if positive_vals.size == 0:
            raise ValueError("Brak dodatnich wartości do log_scale=True.")
        if vmin <= 0:
            vmin = float(np.nanmin(positive_vals))
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # --- scatter ---
    sc = ax.scatter(
        lons, lats,
        c=vals,
        s=point_size,
        cmap=cmap,
        vmin=None if log_scale else vmin,
        vmax=None if log_scale else vmax,
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=point_alpha,
        edgecolor=point_edgecolor,
        linewidths=0,
        zorder=point_zorder,
    )

    # --- colorbar ---
    cbar_opts = dict(orientation=cbar_orientation, pad=cbar_pad)
    if cbar_kwargs is not None:
        cbar_opts.update(cbar_kwargs)
    cbar = plt.colorbar(sc, ax=ax, **cbar_opts)
    cbar.set_label(cbar_label)

    # --- tytuł ---
    ttl_opts = dict(fontsize=12)
    if title_kwargs is not None:
        ttl_opts.update(title_kwargs)
    ax.set_title(f"{title_prefix} – {epoch}", **ttl_opts)

    # --- adnotacja statystyki w rogu ---
    if annotate_stats:
        mn = float(np.nanmin(vals))
        mx = float(np.nanmax(vals))
        med = float(np.nanmedian(vals))
        p95 = float(np.nanpercentile(vals, 95))

        text = (
            f"min={mn:.2f}, median={med:.2f}, max={mx:.2f}\n"
            f"p95={p95:.2f}"
        )

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
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                edgecolor='0.7',
                alpha=stats_alpha
            ),
            zorder=4,
        )

    return fig, ax, sc

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_gix_family(times, reg, dmin, dmax, figsize=(12,9), dpi=100):
    """
    Tworzy 3-panelowy wykres rodzinny GIX:
        (a) GIX      = sqrt(gx_bar^2 + gy_bar^2)
        (b) GIXσ     = std(|grad(VTEC)|)
        (c) GIX95    = 95%(|grad(VTEC)|)

    Parametry:
        times : DataFrame zawierający kolumny:
            - 'time'
            - 'GIX_mtecu_km'
            - 'GIXS_mtecu_km'
            - 'GIXP95_mtecu_km'
        reg   : nazwa regionu (str)
        dmin, dmax : zakres długości dipola
        figsize : rozmiar figury
        dpi : gęstość do zapisu
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

    # ---- Formatowanie osi czasu, siatki, itd. ----
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    return fig, axes
