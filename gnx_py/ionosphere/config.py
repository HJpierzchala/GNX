
"""GNSS TEC processing configuration and controllers.

This module provides:
- Configuration dataclasses for TEC processing.
- Utilities for single‑receiver DCB calibration based on GIM data.
- STEC monitor for producing leveled TEC measurements per satellite arc.
- High‑level controllers (TECController, TECSession) orchestrating the workflow:
  loading data, screening, orbit interpolation, preprocessing, cycle‑slip detection,
  and TEC estimation, with optional model comparison.

Conventions:
- Angles are in degrees unless otherwise specified.
- Time values follow the input RINEX/GNSS library conventions.
- DCB values are expressed in nanoseconds (ns) where noted; conversions to seconds
  or meters are explicitly stated in the code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Literal, Set, Final, Any

import numpy as np
import pandas as pd
from ..coordinates import lagrange_emission_interp, lagrange_reception_interp, CustomWrapper
from .models import klobuchar, ntcm_vtec
from ..io import OrbitData, read_sp3
from ..time import datetime2toc, datetime2doy
from datetime import datetime
from ..configuration import TECConfig
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import cheby1, filtfilt
from ..io import GNSSDataProcessor2, read_sp3
from ..coordinates import make_ionofree, emission_interp, BroadcastInterp
from ..tools import CSDetector
from ..tools import DDPreprocessing


def apply_savgol_filter_1d(
    y: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Zastosuj filtr Savitzky-Golay do jednowymiarowego sygnału (np. TEC).

    - Obsługuje różne długości łuków.
    - Robi prostą interpolację w miejsce NaN przed filtrowaniem,
      po filtrze NaN-y są przywracane na swoje pozycje.
    - Automatycznie dopasowuje długość okna, gdy łuk jest krótki.

    Parameters
    ----------
    y : np.ndarray
        Dane wejściowe (1D), mogą zawierać NaN.
    window_length : int
        Długość okna w próbkach (musi być nieparzysta).
    polyorder : int
        Rząd wielomianu używanego w filtrze.

    Returns
    -------
    y_filt : np.ndarray
        Wygładzony sygnał (1D), NaN-y zachowane tam, gdzie były.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y

    # maska wartości ważnych
    valid = np.isfinite(y)
    if valid.sum() < polyorder + 2:
        # Za mało punktów, żeby cokolwiek sensownego policzyć
        return y

    # prosta interpolacja NaN-ów (po indeksie)
    if not valid.all():
        x = np.arange(n)
        y_interp = np.interp(x, x[valid], y[valid])
    else:
        y_interp = y.copy()

    # dopasuj długość okna do długości łuku
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
    if wl <= polyorder:
        # nie da się zastosować SG przy tak małej liczbie próbek
        return y

    y_sg = savgol_filter(y_interp, window_length=wl, polyorder=polyorder, mode="interp")

    # przywróć NaN-y tam gdzie brak obserwacji
    y_sg[~valid] = np.nan
    return y_sg

def _estimate_dt_seconds(t: np.ndarray) -> float:
    """Pomocniczo: oszacuj krok czasowy [s] z wektora t (float lub datetime64)."""
    t = np.asarray(t)
    if t.size < 2:
        return 1.0  # cokolwiek, i tak filtra nie użyjemy przy tak krótkim łuku

    if np.issubdtype(t.dtype, np.datetime64):
        # konwersja datetime64 -> sekundy (float)
        tt = t.astype("datetime64[ns]").astype("int64") * 1e-9
    else:
        tt = t.astype(float)

    diffs = np.diff(tt)
    # odfiltruj ewentualne zera / dziwne outliery
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def apply_cheby_lowpass_1d(
    t: np.ndarray,
    y: np.ndarray,
    cutoff_period_s: float = 600.0,
    order: int = 4,
    ripple_db: float = 0.5,
) -> np.ndarray:
    """
    Zastosuj dolnoprzepustowy filtr Czebyszewa typu I (zero-phase, filtfilt)
    do sygnału 1D (np. TEC).

    Parameters
    ----------
    t : np.ndarray
        Wektory czasów (float albo datetime64) tego samego rozmiaru co `y`.
        Może być to np. `gr.index.get_level_values('time').values`.
    y : np.ndarray
        Dane wejściowe (1D), mogą zawierać NaN.
    cutoff_period_s : float
        Okres odcięcia [s]; ~czasy krótsze niż ten będą tłumione.
        Np. 600 s => fc ≈ 1/600 Hz.
    order : int
        Rząd filtru (4–6 zazwyczaj wystarcza).
    ripple_db : float
        Dopuszczalna falistość (ripple) w paśmie przepustowym [dB].

    Returns
    -------
    y_filt : np.ndarray
        Wygładzony sygnał, NaN-y przywrócone na oryginalne pozycje.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y

    valid = np.isfinite(y)
    if valid.sum() < max(8, 2 * order + 1):
        # za mało punktów na sensowny filtr IIR
        return y

    # dt [s] z wektora czasów
    dt = _estimate_dt_seconds(np.asarray(t))
    fs = 1.0 / dt  # Hz
    fc = 1.0 / float(cutoff_period_s)  # Hz
    # znormalizowana częstotliwość odcięcia do Nyquista
    wn = 2.0 * fc / fs  # = 2*dt/cutoff_period_s

    # zabezpieczenia
    if wn >= 1.0:
        wn = 0.99
    if wn <= 0.0:
        # cutoff_period_s >> długość łuku – filtr praktycznie nic nie zrobi
        return y

    # uzupełnij NaN przez interpolację (po czasie)
    x = np.arange(n)
    if not valid.all():
        y_work = np.interp(x, x[valid], y[valid])
    else:
        y_work = y.copy()

    # współczynniki filtru Chebyshev typu I
    b, a = cheby1(N=order, rp=ripple_db, Wn=wn, btype="low", analog=False)

    # filtfilt potrzebuje wystarczająco długiego sygnału
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        # przy bardzo krótkich łukach przełącz się na zwykłe lfilter lub zwróć oryginał
        # żeby nie komplikować: tu po prostu zwracamy oryginalne dane
        return y

    y_filt = filtfilt(b, a, y_work, method="pad")

    # przywróć NaN-y
    y_filt[~valid] = np.nan
    return y_filt

class Calibration:
    """
    Single receiver calibration using background GIM data.

    Workflow:
    1) Smooth L4 observation and divide by speed of light to obtain the uncalibrated
       ionospheric time delay (seconds).
    2) Apply satellite DCB to smoothed L4 [sec] observation.
    3) Retrieve ionospheric delay from GIM (typically in meters at L1) and convert
       to time delay.
    4) Subtract the GIM‑derived ionospheric delay from the smoothed L4 observation;
       the residual bias is interpreted as receiver DCB (seconds).
    5) For each epoch, average residuals with elevation weighting to estimate DCB.
    6) Take the median across epochs to obtain the final receiver DCB (nanoseconds).
    """


    c = 299792458
    K = 40.3 * 10 ** 16
    def __init__(self,name,df,sys, mode,elev_curoff=30):
        """Initialize single‑receiver DCB calibration.

                Args:
                    name: Station name (identifier).
                    df: Preprocessed observation DataFrame containing L4/P4, bias and ion columns.
                    sys: Constellation code ("G" or "E").
                    mode: Frequency pair mode, e.g., "L1L2" or "E1E5a".
                    elev_curoff: Elevation cutoff (degrees) used for calibration.
                """

        self.name = name
        self.df = df
        self.sys = sys
        self.mode = mode
        self.FREQ_DICT = {'L1':1575.42e06, 'L2':1227.60e06,
                          'E1':1575.42e06,'E5a':1176.45e06,'E5b':1207.14e06}
        self.C = 299792458
        self.elev_curoff = elev_curoff
        self.dcb_u = None

    def _get_factor(self):
        """Compute frequency‑dependent factors for TEC and DCB conversions.

                Returns:
                    tuple[float, float]: (coef, tecu2meters)
                        - coef: converts modeled ionospheric delay to time delay contribution
                                in seconds when forming geometry‑free combinations.
                        - tecu2meters: conversion factor from TECU to meters at f1.
                """

        m1, m2 = self.mode[:2], self.mode[2:]
        f1, f2 = self.FREQ_DICT[m1], self.FREQ_DICT[m2]


        tecu2meters = (self.K/f1**2)
        coef= ((self.K*(f1**2-f2**2))/(self.c*(f1**2*f2**2)))
        return coef, tecu2meters

    def _data_prep(self):
        """Prepare and filter data, compute modeled iono and per‑sample DCB residuals.

                Mutates:
                    self.df: Resets/sets index, filters by elevation, and adds:
                             - model: GIM ionospheric delay converted to meters at f1.
                             - dcb_u: per‑sample receiver DCB estimate in nanoseconds.
                """

        self.df = self.df.reset_index()
        self.df['sv'] = self.df['sv'].str.split('_').str[0]
        self.df= self.df.set_index(['time','sv'])
        self.df=self.df[self.df['ev']>self.elev_curoff]
        coef, tecu2meters = self._get_factor()
        self.df['model'] = self.df['ion']/tecu2meters
        self.df['dcb_u'] = (coef*self.df['model'] - (self.df['L4']/self.c + self.df['bias']*1e-09))/1e-09


    def calibrate(self):
        """Estimate receiver DCB using elevation‑weighted epoch medians.

                Returns:
                    float: Receiver DCB in nanoseconds.
                """

        self._data_prep()
        est=[]
        for epoch, gr in self.df.groupby('time'):
            if len(gr)>1:
                weights = np.sin(np.deg2rad(gr['ev'].values))**4
                mn = np.average(gr['dcb_u'].values,weights=weights)
                est.append(mn)
            else:
                est.append(gr['dcb_u'].values[0])
        self.dcb_u = np.median(np.array(est))
        return self.dcb_u

class STECMonitor:
    """Slant ionospheric TEC measurement class.

     Produces leveled TEC measurements per satellite arc using:
     - Phase leveling of geometry‑free combinations.
     - Median leveling for robust smoothing at boundaries.
     - Optional inclusion of satellite and station DCBs depending on configuration.
     """

    def __init__(self, mode, sys, obs, config, flh:Optional[Union[np.typing.ArrayLike, None]]=None, broadcast:Optional[Union[OrbitData, None]]  = None):
        """Initialize STEC monitor.

                Args:
                    mode: Frequency mode, e.g., "L1L2" or "E1E5a".
                    sys: Constellation code ("G"/"E").
                    obs: Observation DataFrame indexed by ['sv','time'] or similar, with L4/P4 and metadata.
                    config: TECConfig instance controlling processing options.
                    flh: Receiver geodetic coordinates (phi, lambda, h) if required by models.
                    broadcast: Optional broadcast orbit and related parameters.
                """

        self.obs = obs # must have arc, L4 and P4
        self.mode = mode # must be F1F2
        self.sys = sys
        self.config: TECConfig =config
        self.flh= flh
        self.broadcast= broadcast
        self.FREQ_DICT = {'L1':1575.42e06, 'L2':1227.60e06,
                          'E1':1575.42e06,'E5a':1176.45e06}
        self.GAMMA = 1/40.3
        self.C = 299792458

    def _get_factor(self):
        """Return the K‑factor to convert geometry‑free phase to TEC.

                This factor maps LI [meters] to TEC units (TECU) for the selected frequency pair.

                Returns:
                    float: Conversion factor for geometry‑free phases to TECU.
                """

        m1, m2 = self.mode[:2], self.mode[2:]
        f1, f2 = self.FREQ_DICT[m1], self.FREQ_DICT[m2]
        return self.GAMMA * ((f1**2*f2**2)/(f1**2-f2**2))

    import numpy as np

    def phase_leveling(self, code, phase, elev):
        """Level phase geometry-free observables to code geometry-free observables (JPL-GIM style)."""
        code = np.asarray(code, dtype=float)
        phase = np.asarray(phase, dtype=float)
        elev = np.asarray(elev, dtype=float)
        if code.shape != phase.shape:
            raise ValueError("code and phase must have the same shape")

        n = len(phase)
        if n == 0:
            return np.array([], dtype=float)

        err = float(self.config.tec_zenith_err)
        window_size = int(self.config.leveling_ws)

        # Bezpieczny rozmiar okna
        window_size = max(1, min(window_size, n))
        half_window = max(1, window_size // 2)

        # Wagi ~ 1 / (err * sin(elev)^2)^2; zabezpieczenie na sin(0)=0 i NaN
        s = np.sin(np.deg2rad(elev))
        with np.errstate(divide='ignore', invalid='ignore'):
            S = 1.0 / (err * s ** 2)
            weights = 1.0 / (S ** 2)
        # Uczyść wagi z inf/NaN
        weights = np.where(np.isfinite(weights), weights, 0.0)

        leveled = np.array(phase, copy=True)

        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            w = weights[start:end]
            c = code[start:end]
            p = phase[i]

            # Usuwamy NaN z okna
            m = np.isfinite(w) & np.isfinite(c)
            w = w[m];
            c = c[m]

            denom = w.sum()
            if denom <= 0.0 or c.size == 0:
                # brak sensownych danych — nie korygujemy punktu
                continue

            # numerator = sum(w * (code - phase_i))
            leveling_diff = np.dot(w, (c - p)) / denom
            leveled[i] = p + leveling_diff

        return leveled

    def median_tec_leveling(self, code_tec, phase_tec):
        """Smooth code-derived TEC with phase-derived TEC using robust median window (NaN-safe)."""
        code_tec = np.asarray(code_tec, dtype=float)
        phase_tec = np.asarray(phase_tec, dtype=float)
        if code_tec.shape != phase_tec.shape:
            raise ValueError("code_tec and phase_tec must have the same shape")

        n = len(code_tec)
        if n == 0:
            return np.array([], dtype=float)

        ws = int(self.config.median_leveling_ws)
        ws = max(1, min(ws, n))
        k = max(1, ws // 2)  # gwarantuje niepuste okno

        out = np.empty(n, dtype=float)

        for i in range(n):
            start = max(0, i - k)
            end = min(n, i + k + 1)  # +1 żeby okno było symetryczne inkluzywne

            # różnice względem bieżącego phase_tec[i]
            diff = code_tec[start:end] - phase_tec[i]

            # NaN-safe median
            if diff.size == 0:
                md = 0.0
            else:
                md = np.nanmedian(diff)
                if not np.isfinite(md):  # przypadek: same NaN-y
                    md = 0.0

            out[i] = phase_tec[i] + md

        return out





    def _process_arc(self,gr):
        """Process a single satellite arc: apply DCBs, level phase and compute TEC.

        Steps:
        - Optionally add satellite and station DCB to code observable.
        - Apply phase leveling and compute leveled TEC.
        - Compute median‑leveled TEC using code/phase combinations.

        Args:
            gr (pd.DataFrame): Single‑arc observations indexed by time for one SV.

        Returns:
            pd.DataFrame: Arc DataFrame with added columns:
                         ['leveled_phase','leveled_tec','code_tec','phase_tec','median_leveled_tec'].
        """

        K = self._get_factor()
        sv = gr.index.get_level_values('sv').tolist()[0]
        t = gr.index.get_level_values('time').values

        dcb = 0.0
        if self.config.add_dcb:
            if 'bias' in gr.columns:
                dcb = gr['bias']*1e-09*self.C
            else:
                print('WARNING! No bias for sv: ', sv)
                dcb  = 0.0

        if self.config.add_sta_dcb:
            if self.config.rcv_dcb_source =='gim':
                dcb += np.asarray(gr.get('sta_bias',0.0))*1e-09*self.C


            elif self.config.rcv_dcb_source =='defined':
                gr['sta_dcb'] = self.config.define_station_dcb
                dcb+=gr['sta_dcb']*1e-09*self.C
            elif self.config.rcv_dcb_source =='calibrate':
                dcb += gr.get('sta_dcb',0.0)*1e-09*self.C


        code = gr['P4'].values+dcb
        phase = gr['L4'].values
        ev = gr['ev'].values

        leveled_phase = self.phase_leveling(code=code,phase=phase,elev=ev)
        leveled_tec = K*leveled_phase

        code_tec, phase_tec = code*K, phase*K
        median_leveled_tec = self.median_tec_leveling(code_tec=code_tec,phase_tec=phase_tec)



        gr=gr.copy()
        gr['leveled_phase'] = leveled_phase
        gr['leveled_tec'] = leveled_tec
        gr['code_tec'] = code_tec

        gr['leveled_tec_sg'] = apply_savgol_filter_1d(
            gr['code_tec'].to_numpy(),
            window_length=getattr(self.config, 'leveling_ws', 11),  # np. 11
            polyorder=getattr(self.config, 'sg_polyorder', 1),  # np. 2
        )

        # --- Chebyshev low-pass na poziomie łuku ---
        gr['leveled_tec_cheby'] = apply_cheby_lowpass_1d(
            t=t,
            y=gr['code_tec'].to_numpy(),
            cutoff_period_s=getattr(self.config, 'cheby_cutoff_s', 1800.0),  # np. 600.0
            order=getattr(self.config, 'cheby_order', 2),  # np. 4
            ripple_db=getattr(self.config, 'cheby_ripple_db', 0.5),  # np. 0.5
        )

        gr['phase_tec'] = phase_tec
        gr['L4'] = leveled_phase
        gr['P4'] = code
        gr['median_leveled_tec'] = median_leveled_tec
        return gr

    def compare_models(self):
        """Compute and attach ionospheric model  for comparison.

               Based on config.compare_models, evaluates selected models (e.g., 'klobuchar',
               'ntcm','gim') and adds their slant ionospheric delays (meters at L1) to self.obs.
               """

        cm = self.config.compare_models
        if self.config.ionosphere_model in cm:
            cm = cm.remove(self.config.ionosphere_model)
        if cm:
            toc = datetime2toc(t=self.obs.index.get_level_values('time').tolist())
            gpsa, gpsb, gala = self.broadcast.gpsa, self.broadcast.gpsb, self.broadcast.gala
            ev, az = self.obs['ev'].to_numpy(), self.obs['az'].to_numpy()

            for model in cm:
                if model == 'klobuchar':
                    if model in self.obs.columns:
                        continue
                    ion = np.array(list(map(lambda row: klobuchar(
                        azimuth=row[0], elev=row[1], fi=self.flh[0], lambda_=self.flh[1],
                        tow=row[2], beta=gpsb, alfa=gpsa), zip(az, ev, toc))))
                    self.obs[model] = ion

                elif model == 'ntcm':
                    if model in self.obs.columns:
                        continue
                    flh = self.flh

                    def apply_run_NTCM_for_gal(group):
                        az = group['az'].values
                        ev = group['ev'].values
                        epoch = group.name  # The group name is the time epoch
                        DOY = datetime2doy(epoch)
                        # Call the function
                        stec = ntcm_vtec(
                            pos=flh,
                            DOY=DOY,
                            ev=ev,
                            az=az,
                            epoch=epoch,
                            gala=gala,
                            mf_no=2  # or self.mf_no if applicable
                        )
                        return pd.Series(stec, index=group.index)

                    grouped = self.obs.groupby('time')
                    result = grouped.apply(apply_run_NTCM_for_gal)
                    result.index = result.index.droplevel(0)
                    stec = result
                    K = 40.3 * 10 ** 16

                    f = 1575.42e06
                    ion = K * (stec / (f ** 2))
                    self.obs[model] = ion
    def run(self):
        """Run the STEC monitor over all arcs and optionally compare models.

        Returns:
            pd.DataFrame: Observations augmented with leveled TEC quantities
                          (and model columns if compare_models is enabled).
        """

        obs_out = self.obs.groupby('sv',group_keys=False).apply(self._process_arc)
        self.obs = obs_out.copy()
        if self.config.compare_models:
            self.compare_models()
        return self.obs


class TECSession:
    """High‑level controller for a TEC measurement session.

    Orchestrates data loading, optional broadcast‑based screening, precise/broadcast
    orbit interpolation, preprocessing (including PCO/PCV, wind‑up, path corrections),
    cycle‑slip detection, and TEC computation via STECMonitor.
    """


    def __init__(self,
                 config: TECConfig,

                 logger: Optional[logging.Logger] = None,
                 gnss_api: Optional[Any]=None,
                 use_sys = Optional[Literal['G','E']]) -> None:
        """Initialize the session controller with external orbit inputs.

        Args:
            config: TECConfig with parameters and paths.

            logger: Optional logger; if None, a default one is created.
            gnss_api: Optional GNSS backend module; if None, import the default.
            use_sys: Constellation to process ('G' or 'E').
        """

        self.config: TECConfig = config
        self.log: logging.Logger = logger or self._default_logger()
        self.use_sys = self.config.sys


    def run(self):
        """Execute the TEC pipeline using provided positions/SP3 inputs.

        Returns:
            pd.DataFrame: TEC output DataFrame with per‑arc results.
        """

        processor =  GNSSDataProcessor2(
            obs_path=self.config.obs_path,
            nav_path=self.config.nav_path,
            dcb_path=self.config.dcb_path,
            sys=self.use_sys,
            use_gfz=self.config.use_gfz,
            atx_path=self.config.atx_path,
            mode=self.config.gps_freq,
            galileo_modes=self.config.gal_freq
        )
        obs_data = self._load_obs_data(processor)
        self.config.interval = obs_data.interval
        xyz, flh = obs_data.meta[4], obs_data.meta[5]

        if self.config.nav_path:
            broadcast = self._load_nav_data(processor)
            if self.config.screen:
                gps_obs, gal_obs = self._screen_data(processor, obs_data.gps, obs_data.gal, broadcast)
            else:
                gps_obs, gal_obs = obs_data.gps, obs_data.gal
        else:
            gps_obs, gal_obs = obs_data.gps, obs_data.gal
            broadcast = None
        if self.use_sys == 'E':
            if gal_obs is None:
                raise FileNotFoundError(f'No Galileo observations for this file: {self.config.obs_path}')
            gps_obs = gal_obs.copy()
            mode = self.config.gal_freq
        elif self.use_sys =='G':
            mode = self.config.gps_freq

        # positions, sp3 = self._interpolate_reception(gps_info=obs_data.meta,interval=obs_data.interval)
        if self.config.orbit_type == "broadcast":
            obs_gps_crd = self._interpolate_broadcast(obs_df=gps_obs,
                                                      xyz=xyz.copy(),
                                                      flh=flh.copy(),
                                                      sys='G',
                                                      mode=self.config.gps_freq,
                                                      orbit=broadcast)
        elif self.config.orbit_type == "precise":

            obs_gps_crd = self._interpolate_lgr(obs=gps_obs, xyz=xyz.copy(), flh=flh.copy(),
                                                    mode=self.config.gps_freq)
        if 'ev' in obs_gps_crd.columns:
            obs_gps_crd = obs_gps_crd[obs_gps_crd['ev']>self.config.ev_mask]
        obs_gps_crd = self._preprocess(obs=obs_gps_crd,
                                       flh=flh.copy(),
                                       xyz=xyz.copy(),
                                       broadcast=broadcast,
                                       phase_shift=obs_data.meta[-1],
                                       sat_pco=obs_data.sat_pco,
                                       rec_pco=obs_data.rec_pco,
                                       antenna_h=obs_data.meta[3],
                                       system=self.use_sys)

        obs_gps_crd = self._detect_cycle_slips(obs_df=obs_gps_crd,
                                               mode=mode,
                                               system=self.use_sys,
                                               interval=obs_data.interval
                                               )


        if self.config.rcv_dcb_source == 'calibrate':
            self.config.add_dcb=False

            monitor =  STECMonitor(obs=obs_gps_crd, mode=mode, config=self.config, sys=self.use_sys, flh=flh,
                                            broadcast=broadcast)
            obs_tec = monitor.run()
            calib = Calibration(name=self.config.station_name, df=obs_tec, sys=self.use_sys, mode=mode)
            dcb_u = calib.calibrate()
            obs_tec['sta_dcb'] = dcb_u
            # po kalibracji aplikujemy DCB satelitow do pomiarow
            self.config.add_dcb=True
            monitor =  STECMonitor(obs=obs_tec, mode=mode, config=self.config,
                                            sys=self.use_sys, flh=flh, broadcast=broadcast)
            obs_tec = monitor.run()
        else:
            monitor =  STECMonitor(obs=obs_gps_crd, mode=mode, config=self.config, sys=self.use_sys, flh=flh,
                                            broadcast=broadcast)
            obs_tec = monitor.run()
        return obs_tec

    def _load_obs_data(self, processor):
        """Load dual‑frequency observations via backend."""

        obs = processor.load_obs_data()
        return obs

    def _load_nav_data(self, processor):
        """Load broadcast navigation messages for screening."""
        broadcast = processor.load_broadcast_orbit()
        return broadcast

    def _screen_data(self, processor, gps_obs, gal_obs, broadcast):
        """Filter observations using broadcast NAV availability and basic quality checks.

        Returns:
            tuple[pd.DataFrame, Optional[pd.DataFrame]]: (gps_obs_filtered, gal_obs_filtered)
        """
        if self.config.system_includes('G'):
            gps_nav = processor.screen_navigation_message(broadcast.gps_orb, 'G')
            gps_obs = processor.mark_outages(gps_obs, 'G', gps_nav)
            gps_obs = gps_obs[gps_obs['gps_outage'] == False]
            gps_obs = gps_obs[(gps_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        if self.config.system_includes('E') and gal_obs is not None:
            gal_nav = processor.screen_navigation_message(broadcast.gal_orb, 'E')
            gal_obs = processor.mark_outages(gal_obs, 'E', gal_nav)
            gal_obs = gal_obs[(gal_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        return gps_obs, gal_obs

    def _interpolate_broadcast(self, obs_df, xyz, flh,sys,mode, orbit, tolerance):
        make_ionofree(obs_df,sys,mode)
        if sys == 'G':
            orb = orbit.gps_orb
        elif sys == 'E':
            orb = orbit.gal_orb
        else:
            raise ValueError("Invalid sys: %s", sys)

        interpolator =BroadcastInterp(obs=obs_df,mode=mode,sys=sys,nav=orb,emission_time=True)
        obs_crd = interpolator.interpolate()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _interpolate_lgr(self, obs, flh, xyz, mode):
        obs = obs.swaplevel('sv','time')
        start = datetime.now()
        sp3 = [read_sp3(f) for f in self.config.sp3_path]
        sp3_df = pd.concat(sp3)
        sp3_df = sp3_df.reset_index().drop_duplicates()
        sp3_df = sp3_df.set_index(['time', 'sv'])
        sp3_df[['x','y','z']]=sp3_df[['x','y','z']].apply(lambda x: x*1e3)
        sp3_df['clk'] = sp3_df['clk'].apply(lambda x: x * 1e-6)
        _, positions = lagrange_reception_interp(obs=obs,sp3_df=sp3_df)
        obs_crd = lagrange_emission_interp(obs=obs,positions=positions,sp3_df=sp3_df)
        end=datetime.now()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _interpolate_emission(self, obs_df,xyz, flh,sys,mode, positions, sp3):
        """Interpolate satellite positions at emission times and add geometry.

        Returns:
            pd.DataFrame: Observations with emission coordinates and geometric angles.
        """


        make_ionofree(obs_df,sys,mode)
        obs_crd = emission_interp(
            obs=obs_df,
            crd=positions,
            prev_sp3_df=sp3[0],
            next_sp3_df=sp3[-1],
            sp3_df=sp3[1]
        )

        wrapper =  CustomWrapper(obs=obs_crd,epochs=None,flh=flh.copy(),xyz_a=xyz.copy(),
                                          mode=mode)
        obs_crd = wrapper.run()
        return obs_crd
    def _preprocess(self, obs, flh, xyz,
                    broadcast,phase_shift,sat_pco, rec_pco, antenna_h, system):
        """Apply preprocessing: PCO/PCV, wind‑up, path corrections, and model terms.

        Returns:
            pd.DataFrame: Preprocessed observations ready for cycle‑slip detection.
        """
        if self.config.skip_sat:
            obs = obs[~obs.index.get_level_values('sv').isin(self.config.skip_sat)]
        if broadcast is None:
            gpsa, gpsb, gala = None, None, None
        else:
            gpsa, gpsb, gala = broadcast.gpsa, broadcast.gpsb, broadcast.gala
        prc =  DDPreprocessing(df=obs, flh=flh.copy(), xyz=xyz.copy(), config=self.config, gpsa=gpsa,
                                  gpsb=gpsb, gala=gala, phase_shift_dict=phase_shift, sat_pco=sat_pco, rec_pco=rec_pco,
                                  antenna_h=antenna_h, system=system)
        obs_preprocessed = prc.run()
        # Local time
        Lon = flh[1]
        time_utc = obs_preprocessed.index.get_level_values('time').to_numpy()
        offset = np.timedelta64(int(Lon * 3600 / 15), 's')
        time_local = time_utc + offset
        obs_preprocessed['LT']=time_local
        return obs_preprocessed

    def _detect_cycle_slips(self, obs_df, mode, system, interval):
        """Detect cycle slips basing on dL1-dL2 difference and form continuous arcs.

        Returns:
            pd.DataFrame: DataFrame indexed by ['sv','time'] with arc segmentation.
        """



        detector =  CSDetector(obs=obs_df,phase_shift_dict=None,dcb=None,sys=system,mode=mode,
                                     interval=interval*60)
        obs_df = detector.run(min_arc_lenth=self.config.min_arc_len)
        obs_df = obs_df.reset_index()
        obs_df = obs_df.set_index(['arc', 'time'])
        obs_df.index.names = ['sv', 'time']
        obs_df = obs_df.drop(columns=['sv'])
        return obs_df

    @staticmethod
    def _default_logger() -> logging.Logger:  # noqa: D401 – simple phrase
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        return logging.getLogger("TEC")

    @staticmethod
    def _import_gnss_api():  # -> ModuleType
        """Import the supported GNSS backend 'gnx_py'.

        Raises:
            ImportError: If the backend is not installed.
        """

        import importlib, sys, logging

        try:
            module = importlib.import_module("gnx_py")
            sys.modules.setdefault("gps", module)
            return module
        except ModuleNotFoundError as exc:
            logging.getLogger(__name__).error("Brak biblioteki 'gnx_py'.")
            raise ImportError(
                "GNSS backend 'gnx_py' nie jest zainstalowany (pip install gnx_py)"
            ) from exc