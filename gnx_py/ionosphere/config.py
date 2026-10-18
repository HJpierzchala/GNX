
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
- Time index: GPS time (GPST), naive (no time zone). The local time column ``LT`` is
  GPST + longitude x 240 s; the 18 s GPST-UTC difference is negligible there.
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
from ..core.coordinates import lagrange_emission_interp, lagrange_reception_interp, CustomWrapper
from .models import klobuchar, ntcm_vtec
from ..io import OrbitData, read_sp3  # read_sp3, datetime: re-exported by gnx_py.ionosphere (__all__)
from ..core.time import datetime2toc, datetime2doy
from datetime import datetime
from ..core.configuration import TECConfig
from ..core.validation import check_choice, check_number
import warnings
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import cheby1, filtfilt
from ..io import GNSSDataProcessor2
from ..core.coordinates import make_ionofree, BroadcastInterp
from ..core.tools import CSDetector
from ..core.tools import DDPreprocessing, RunNotes, observation_phase_shift_attrs
from ..core.session_errors import guarded_session_run
from ..core import _notes
from ..models.apriori import resolve_position
from ..core.session_guards import (drop_empty_systems, file_span, load_sp3, no_observations_error, require_phases,
                              screen_observations)
from ..core.gnss import SIGNALS, SUPPORTED_MODES_BY_SYSTEM, frequency_hz, gamma, mode_layout, mode_signals, system_name
from .bias_policy import apply_stec_bias_policy, gim_dcb_applies, leveling_weights, _code_pair_for_mode
from ..models.biases import NS_TO_M
from .ionex import GIMReader


def _gim_dcb_reference(config):
    """Reference observables of the IONEX DCBs of a GIM run (None: no GIM, IONEX defaults apply)."""
    path = getattr(config, "gim_path", None)
    if getattr(config, "ionosphere_model", None) != "gim" or not path or not Path(path).is_file():
        return None
    return GIMReader.read_dcb_reference(path)


def apply_savgol_filter_1d(
    y: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter to a one-dimensional signal (e.g. TEC).

    - Handles arcs of any length.
    - Fills NaN by simple interpolation before filtering; the NaN are put back
      at their positions after the filter.
    - Shortens the window automatically when the arc is short.

    Parameters
    ----------
    y : np.ndarray
        Input data (1D), may contain NaN.
    window_length : int
        Window length in samples (must be odd).
    polyorder : int
        Order of the polynomial used by the filter.

    Returns
    -------
    y_filt : np.ndarray
        Smoothed signal (1D), NaN kept where they were.
    """
    check_number("window_length (sg_window)", window_length, ge=1, integer=True)
    check_number("polyorder (sg_polyorder)", polyorder, ge=0, integer=True)
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
    """Helper: estimate the time step [s] of the vector t (float or datetime64)."""
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
    Apply a low-pass Chebyshev type I filter (zero phase, filtfilt) to a 1D
    signal (e.g. TEC).

    Parameters
    ----------
    t : np.ndarray
        Epochs (float or datetime64), the same size as `y`, for example
        `gr.index.get_level_values('time').values`.
    y : np.ndarray
        Input data (1D), may contain NaN.
    cutoff_period_s : float
        Cut-off period [s]; variations faster than this are damped.
        E.g. 600 s => fc ≈ 1/600 Hz.
    order : int
        Filter order (4-6 is usually enough).
    ripple_db : float
        Allowed passband ripple [dB].

    The filter has unit DC gain (a TEC level passes unchanged). With filtfilt the
    passband gain then ripples between 1 and ``10**(ripple_db/10)`` (1.122 for 0.5 dB) for an
    even ``order``, and between ``10**(-ripple_db/10)`` (0.891) and 1 for an odd ``order``.

    Returns
    -------
    y_filt : np.ndarray
        Smoothed signal, NaN restored at their original positions.
    """
    check_number("cutoff_period_s (cheby_cutoff_s)", cutoff_period_s, gt=0, unit="s")
    check_number("order (cheby_order)", order, ge=1, integer=True)
    check_number("ripple_db (cheby_ripple_db)", ripple_db, gt=0, unit="dB")
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
    # unit DC gain (bug-fix stage 9, ION-059): an even-order Chebyshev-I design has DC gain 10^(-rp/20)
    # (0.944 for 0.5 dB, squared by filtfilt to 0.891: a constant 30 TECU came out as 26.74). With the DC
    # normalised the passband ripples between gain 1 and 10^(rp/10) after filtfilt (1.122 for 0.5 dB); an
    # odd order already has DC gain 1 (unchanged) and ripples between 10^(-rp/10) and 1
    b = b * (a.sum() / b.sum())

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

    Status:
        Legacy/active compatibility path. It is still used by ``TECSession``
        when ``add_sta_dcb=True`` and ``rcv_dcb_source="calibrate"``, but it
        should be changed only with strong numerical regression tests.

    Purpose:
        Estimate a receiver DCB from leveled STEC observations by aligning them
        to a background GIM ionosphere value. The result is a single station
        bias in nanoseconds.

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
        self.FREQ_DICT = {name: spec.frequency_hz for name, spec in SIGNALS.items()}
        self.C = 299792458
        self.elev_curoff = check_number("elev_curoff", elev_curoff, ge=0, lt=90, unit="deg")
        self.dcb_u = None

    def _get_factor(self):
        """Compute frequency‑dependent factors for TEC and DCB conversions.

                Returns:
                    tuple[float, float]: (coef, tecu2meters)
                        - coef: converts modeled ionospheric delay to time delay contribution
                                in seconds when forming geometry‑free combinations.
                        - tecu2meters: conversion factor from TECU to meters at f1.
                """

        m1, m2 = mode_signals(self.mode)
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

                Samples without a GIM ionosphere ('ion'), satellite DCB ('bias') or 'L4' value (IPP
                outside the maps, satellite not in the IONEX DCB block) are left out. Without any usable
                sample the result is NaN, with a warning; the STEC bias policy then takes the next
                station bias source.

                Returns:
                    float: Receiver DCB in nanoseconds (NaN without usable data).
                """

        self._data_prep()
        usable = np.isfinite(self.df['dcb_u'].to_numpy(dtype=float))
        if not usable.all():
            left_out = sorted(set(self.df.index.get_level_values('sv')[~usable]))
            logging.getLogger(__name__).warning(
                "Receiver DCB calibration of %s (%s %s): %d of %d samples without a GIM ionosphere, satellite "
                "DCB or L4 value left out (satellites %s)", self.name, self.sys, self.mode, int((~usable).sum()),
                usable.size, ", ".join(left_out[:12]) + (" ..." if len(left_out) > 12 else ""))
            self.df = self.df[usable]
        if self.df.empty:
            warnings.warn(
                f"Receiver DCB calibration of {self.name} ({self.sys} {self.mode}): no sample above the "
                f"{self.elev_curoff} deg elevation cutoff with a GIM ionosphere, satellite DCB and L4 value; "
                f"no calibrated receiver DCB (NaN)", UserWarning, stacklevel=2)
            self.dcb_u = float('nan')
            return self.dcb_u
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

     Status:
     Active core computation class. Requires caution: ``_process_arc`` changes
     the measured geometry-free code/phase combinations and feeds downstream
     TEC values.

     Purpose:
     Process already-prepared dual-frequency observations for one GNSS system
     and one signal pair. The class assumes ``P4`` and ``L4`` geometry-free
     observables, elevation angles, arc labels and optional bias/model columns
     are already present.

     Supported systems:
     GPS, Galileo and BeiDou are selected by ``sys`` plus the configured
     ``mode``. BDS ``B1IB3I`` has direct regression coverage in this branch.

     Bias policy:
     Satellite and station biases are resolved through
     ``apply_stec_bias_policy`` using ``TECConfig``: one constant per arc
     each, added to the levelled phase.

     Produces leveled TEC measurements per satellite arc using:
     - Carrier-to-code levelling of the geometry-free phase (whole arc by
       default, ``leveling_ws`` samples as an option; weights ``leveling_weights``).
     - The median variant of the levelling (``median_leveled_tec``).
     - Four calibration modes (``stec_uncal``, ``stec_sat``, ``stec_rcv``,
       ``stec_full`` = ``leveled_tec``) depending on configuration.
     """

    def __init__(self, mode, sys, obs, config, flh:Optional[Union[np.typing.ArrayLike, None]]=None, broadcast:Optional[Union[OrbitData, None]]  = None,
                 bias_notes: Optional[set] = None):
        """Initialize STEC monitor.

                Args:
                    mode: Frequency mode, e.g., "L1L2" or "E1E5a".
                    sys: Constellation code ("G"/"E").
                    obs: Observation DataFrame indexed by ['sv','time'] or similar, with L4/P4 and metadata.
                    config: TECConfig instance controlling processing options.
                    flh: Receiver geodetic coordinates (phi, lambda, h) if required by models.
                    broadcast: Optional broadcast orbit and related parameters.
                    bias_notes: Optional set of bias messages already given in this run (shared
                        by the monitors of one TECSession run); None starts a new one.
                """

        self.obs = obs # must have arc, L4 and P4
        self.mode = mode # must be F1F2
        self.sys = sys
        self.config: TECConfig =config
        self.flh= flh
        self.broadcast= broadcast
        self.FREQ_DICT = {name: spec.frequency_hz for name, spec in SIGNALS.items()}
        self.GAMMA = 1/40.3
        self.C = 299792458
        # code pairs of the IONEX DCBs ('bias'/'sta_bias' columns of a GIM run) and the
        # messages already given for this run
        self.gim_reference = _gim_dcb_reference(config)
        self._bias_notes: set = bias_notes if bias_notes is not None else set()

    def _get_factor(self):
        """Return the K‑factor to convert geometry‑free phase to TEC.

                K = (1/40.3) f1^2 f2^2 / (f1^2 - f2^2) [electrons/m^2 per metre of L4] for the selected
                frequency pair: the STEC columns are in el/m^2 (divide by 1e16 for TECU; the ``_tecu``
                columns are).

                Returns:
                    float: Conversion factor of geometry‑free phases [m] to el/m^2.
                """

        m1, m2 = mode_signals(self.mode)
        f1, f2 = self.FREQ_DICT[m1], self.FREQ_DICT[m2]
        return self.GAMMA * ((f1**2*f2**2)/(f1**2-f2**2))


    def _level(self, code, phase, elev, ws):
        """Carrier-to-code levelling in the units of ``code``/``phase``.

        ``L4_lev = L4 + sum w (P4 - L4) / sum w`` with ``w = leveling_weights(E)`` (sin^4 E) over the finite
        samples of the whole arc (``ws=None``: one offset per arc; Mannucci et al. 1998, Radio Sci 33:565;
        Ciraolo et al. 2007, J Geod 81:111), or over a centred window of ``ws`` samples (``i - h .. i + h``,
        ``h = max(1, ws // 2)``) for each sample. A sample without any code in its window (or an arc without
        code) gives NaN: there is nothing to level it to.
        """
        code = np.asarray(code, dtype=float)
        phase = np.asarray(phase, dtype=float)
        if code.shape != phase.shape:
            raise ValueError("code and phase must have the same shape")
        n = len(phase)
        if n == 0:
            return np.array([], dtype=float)
        w = leveling_weights(elev)
        d = code - phase
        ok = np.isfinite(d) & (w > 0)
        if ws is None:
            if not ok.any():
                return np.full(n, np.nan)
            return phase + float(np.dot(w[ok], d[ok]) / w[ok].sum())
        ws = max(1, min(int(ws), n))
        h = max(1, ws // 2)
        wd = np.where(ok, w * np.where(ok, d, 0.0), 0.0)
        ww = np.where(ok, w, 0.0)
        cs_wd = np.concatenate([[0.0], np.cumsum(wd)])
        cs_w = np.concatenate([[0.0], np.cumsum(ww)])
        i = np.arange(n)
        lo, hi = np.maximum(0, i - h), np.minimum(n, i + h + 1)
        num, den = cs_wd[hi] - cs_wd[lo], cs_w[hi] - cs_w[lo]
        # no floating-point exception is possible here (bug-fix stage 13b dropped np.errstate): the divisor is 1
        # where den <= 0, and num is a difference of finite cumulative sums (w = sin^4 E in [0, 1], d finite where
        # used, every other sample 0)
        off = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        return phase + off

    def phase_leveling(self, code, phase, elev):
        """Level the geometry-free phase to the geometry-free code (``TECConfig.leveling_ws``: None = arc).

        Returns the levelled phase ``L4 + weighted mean(P4 - L4)`` (see :meth:`_level`).
        """
        return self._level(code, phase, elev, self.config.leveling_ws)

    def median_tec_leveling(self, code_tec, phase_tec):
        """Robust carrier-to-code levelling: ``phase_tec + nanmedian(code_tec - phase_tec)`` over the whole arc
        (``TECConfig.median_leveling_ws=None``) or a centred window of ``median_leveling_ws`` samples
        (``i - k .. i + k``, ``k = max(1, ws // 2)``). NaN where the arc or window has no code (formerly
        ``phase[i] + median(code - phase[i])``, i.e. a moving median of code TEC)."""
        code_tec = np.asarray(code_tec, dtype=float)
        phase_tec = np.asarray(phase_tec, dtype=float)
        if code_tec.shape != phase_tec.shape:
            raise ValueError("code_tec and phase_tec must have the same shape")
        n = len(code_tec)
        if n == 0:
            return np.array([], dtype=float)
        d = code_tec - phase_tec
        ws = self.config.median_leveling_ws
        if ws is None:
            return phase_tec + (np.nanmedian(d) if np.isfinite(d).any() else np.nan)
        ws = max(1, min(int(ws), n))
        k = max(1, ws // 2)
        med = pd.Series(d).rolling(window=2 * k + 1, center=True, min_periods=1).median().to_numpy()
        return phase_tec + med

    def _process_arc(self,gr):
        """Process a single satellite arc: level the phase, apply the arc's bias constants, compute TEC.

        Status:
            Active and requires caution. This is the main numerical path for
            STEC arc processing; changes can alter produced TEC values.

        Steps:
        - The uncorrected code ``P4`` is levelled once: ``L4_lev`` (:meth:`phase_leveling`).
        - The bias policy gives one satellite and one station bias constant per arc
          (``apply_stec_bias_policy``), added after the levelling, which is exact
          for a constant (``level(P4 + b) = L4_lev + b``).
        - Four calibration modes, all in el/m^2 like ``leveled_tec`` (K: TEC per metre of L4):
          ``stec_uncal = K L4_lev`` (still holds both code DCBs of the pair),
          ``stec_sat = stec_uncal + K DCB_s``, ``stec_rcv = stec_uncal + K DCB_r``,
          ``stec_full = stec_uncal + K (DCB_s + DCB_r)``; ``leveled_tec`` is ``stec_full``.
          ``stec_sat_bias_tecu`` / ``stec_rcv_bias_tecu`` are the constants in TECU,
          ``stec_bias_datum`` the datum of the satellite biases.
        - ``median_leveled_tec``: the median variant, with the same constants.
        - ``leveled_tec_sg`` / ``leveled_tec_cheby``: Savitzky-Golay (``sg_window``,
          ``sg_polyorder``) and zero-phase Chebyshev-I (``cheby_*``, unit DC gain) filters
          of ``leveled_tec``.
        - ``leveling_std_tecu``: per-arc levelling-quality diagnostic, the weighted
          (sin^4 E) std of ``K (P4 - L4_lev)`` in TECU (code noise and multipath around the
          levelled phase; not a formal error of the offset).

        Args:
            gr (pd.DataFrame): Single‑arc observations indexed by time for one SV.

        Returns:
            pd.DataFrame: Arc DataFrame with the columns above plus ``leveled_phase`` (levelled
            phase with the bias constants, m), ``code_tec`` (code with the constants),
            ``phase_tec`` (K x raw L4) and the bias diagnostics of the policy. ``P4`` and ``L4``
            are left as they came in.
        """

        K = self._get_factor()
        t = gr.index.get_level_values('time').values

        gr, bias = apply_stec_bias_policy(
            gr,
            config=self.config,
            mode=self.mode,
            system=self.sys,
            logger=logging.getLogger(__name__),
            gim_reference=self.gim_reference,
            notified=self._bias_notes,
        )

        code = gr['P4'].to_numpy(dtype=float)
        phase = gr['L4'].to_numpy(dtype=float)
        ev = gr['ev'].to_numpy(dtype=float)
        sat_m, sta_m = float(bias.satellite_arc_m), float(bias.station_arc_m)
        total_m = sat_m + sta_m

        l4_lev = self.phase_leveling(code=code, phase=phase, elev=ev)
        stec_uncal = K * l4_lev

        w = leveling_weights(ev)
        r = code - l4_lev
        ok = np.isfinite(r) & (w > 0)
        if ok.any():
            mean_r = np.dot(w[ok], r[ok]) / w[ok].sum()
            lev_std = float(np.sqrt(np.dot(w[ok], (r[ok] - mean_r) ** 2) / w[ok].sum()) * K / 1e16)
        else:
            lev_std = float('nan')

        gr = gr.copy()
        gr['leveled_phase'] = l4_lev + total_m
        gr['stec_uncal'] = stec_uncal
        gr['stec_sat'] = stec_uncal + K * sat_m
        gr['stec_rcv'] = stec_uncal + K * sta_m
        gr['stec_full'] = stec_uncal + K * total_m
        gr['leveled_tec'] = gr['stec_full']
        gr['stec_sat_bias_tecu'] = K * sat_m / 1e16
        gr['stec_rcv_bias_tecu'] = K * sta_m / 1e16
        gr['leveling_std_tecu'] = lev_std
        gr['code_tec'] = K * (code + total_m)
        gr['phase_tec'] = K * phase
        gr['median_leveled_tec'] = (self.median_tec_leveling(code_tec=K * code, phase_tec=K * phase)
                                    + K * total_m)

        leveled_tec = gr['leveled_tec'].to_numpy(dtype=float)
        gr['leveled_tec_sg'] = apply_savgol_filter_1d(
            leveled_tec, window_length=self.config.sg_window, polyorder=self.config.sg_polyorder)
        gr['leveled_tec_cheby'] = apply_cheby_lowpass_1d(
            t=t, y=leveled_tec, cutoff_period_s=self.config.cheby_cutoff_s, order=self.config.cheby_order,
            ripple_db=self.config.cheby_ripple_db)
        return gr

    def compare_models(self):
        """Compute and attach ionospheric model values for comparison.

               Status:
               Experimental diagnostic path. It can add Klobuchar/NTCM/BDGIM/GIM
               comparison columns, but is not the primary STEC measurement
               path and should be validated for each model/product setup.

               Based on config.compare_models, evaluates the selected models ('klobuchar',
               'ntcm', 'bdgim', 'gim'; the background ``ionosphere_model`` is skipped, it is 'ion') and adds
               per model two columns: ``<model>`` - the slant ionospheric group delay in metres at
               the first signal of the mode, like the background model column 'ion'; and
               ``<model>_tecu`` - the same as slant TEC in TECU (the STEC columns of the monitor
               are in el/m^2).

               - 'klobuchar': GPS and Galileo take the GPS coefficients (GPSA/GPSB, IS-GPS-200).
                 BeiDou follows ``bds_klobuchar``: 'gps' (default) the GPS coefficients scaled to the
                 first signal; 'icd' (opt-in) the BDS model (BDS-SIS-ICD-B1I 3.0 5.2.4.7, BDSA/BDSB,
                 BDT = GPST - 14 s), or the GPS coefficients with a warning when the navigation header
                 has no BDSA/BDSB. The regional BDS-2 coefficients were worse than GPS-scaled on 5 of
                 9 station-days (``Config.bds_klobuchar``).
               - 'ntcm': NTCM-G (Galileo coefficients) with its MSLM mapping (alpha 0.9782 at 450 km).
               - 'bdgim': the BeiDou-3 global broadcast model (BDS-SIS-ICD-B2b-1.0 §7.7,
                 ``gnx_py.ionosphere.bdgim``) with, per row, the '> ION Cxx CNVX'
                 set in force (a RINEX 4 nav_path; ValueError naming the file and its version
                 without one), evaluated at BDT = GPST - 14 s; a negative model STEC gives 0, as in
                 the background model, so ``bdgim_tecu`` is the clamped model STEC.
               - 'gim': the IONEX file ``gim_path`` with the session's GIM settings
                 (``gim_mapping_function``, ``gim_time_interpolation``, GPST -> UTC epochs).
               """
        from ..core.tools import (bds_klobuchar_model, _bds_klobuchar_rows, _bdgim_rows, bdgim_missing_message,
                             check_gim_coverage, gim_slant_tec)
        from .bdgim import bdgim_slant, gpst_to_bdt
        from .ionex import GIMReader as _GIMReader
        from .models import klobuchar_array, klobuchar_bds, BDT_MINUS_GPST_S
        from ..core.conversion import geodetic2ecef

        # the background model is already in the data; a new list leaves the user's config unchanged
        cm = [m for m in (self.config.compare_models or []) if m != self.config.ionosphere_model]
        if not cm:
            return
        times = self.obs.index.get_level_values('time')
        ev, az = self.obs['ev'].to_numpy(dtype=float), self.obs['az'].to_numpy(dtype=float)
        # the models give the delay at L1 (GPS), B1I (BDS) or as TEC; 'ion' is at the first signal of the mode
        signal = mode_signals(self.mode)[0]
        f = frequency_hz(signal)
        K = 40.3 * 10 ** 16
        broadcast = self.broadcast

        def coefficients(*names):
            values = [getattr(broadcast, n, None) for n in names]
            if any(v is None for v in values):
                raise ValueError(f"compare_models: the navigation header (nav_path) gives no "
                                 f"{'/'.join(n.upper() for n in names)} coefficients")
            return values

        for model in cm:
            if model in self.obs.columns:
                continue
            if model == 'klobuchar':
                icd = self.sys == 'C' and bds_klobuchar_model(self.config) == 'icd'
                t = times.to_numpy('datetime64[ns]')
                # per row: the BDS set in force at the epoch (broadcast ionosphere table, GEO first, bug-fix
                # stage 10), else the one BDSA/BDSB set of the broadcast data
                bdsa, bdsb = _bds_klobuchar_rows(getattr(broadcast, 'iono', None), t) if icd else (None, None)
                if icd and bdsa is None:
                    bdsa, bdsb = getattr(broadcast, 'bdsa', None), getattr(broadcast, 'bdsb', None)
                bds = icd and bdsa is not None and bdsb is not None
                if bds:
                    sod = (t - t.astype('datetime64[D]')).astype('timedelta64[ns]').astype(np.int64) / 1e9
                    ion = klobuchar_bds(azimuth=az, elev=ev, fi=self.flh[0], lambda_=self.flh[1],
                                        sod_bdt=sod + BDT_MINUS_GPST_S, alfa=bdsa, beta=bdsb)
                    ion = ion * gamma('B1I', signal)
                else:
                    gpsa, gpsb = coefficients('gpsa', 'gpsb')
                    if icd:
                        warnings.warn("compare_models 'klobuchar' with bds_klobuchar='icd' for BeiDou: no "
                                      "BDSA/BDSB in the navigation header; the GPS Klobuchar model scaled to "
                                      f"{signal} is used", UserWarning, stacklevel=2)
                    toc = np.asarray(datetime2toc(t=times.tolist()), dtype=float)
                    ion = klobuchar_array(azimuth=az, elev=ev, fi=self.flh[0], lambda_=self.flh[1],
                                          tow=toc, beta=gpsb, alfa=gpsa) * gamma('L1', signal)

            elif model == 'ntcm':
                (gala,) = coefficients('gala')
                flh = self.flh

                def apply_run_NTCM_for_gal(group):
                    epoch = group.name  # The group name is the time epoch
                    stec = ntcm_vtec(pos=flh, DOY=datetime2doy(epoch), ev=group['ev'].values,
                                     az=group['az'].values, epoch=epoch, gala=gala,
                                     mf_no=1)  # NTCM-G MSLM (alpha 0.9782 at 450 km, spec eq. 32-33)
                    return pd.Series(stec, index=group.index)

                result = self.obs.groupby('time').apply(apply_run_NTCM_for_gal)
                result.index = result.index.droplevel(0)
                ion = (K * result.reindex(self.obs.index) / f ** 2).to_numpy(dtype=float)

            elif model == 'bdgim':
                t = times.to_numpy('datetime64[ns]')
                alpha, info = _bdgim_rows(getattr(broadcast, 'iono', None), t)
                if alpha is None:
                    raise ValueError(bdgim_missing_message(
                        getattr(self.config, 'nav_path', None), info,
                        f"compare_models 'bdgim' ({self.sys} {self.mode})", "drop 'bdgim' from compare_models"))
                r = bdgim_slant(alpha, self.flh[0], self.flh[1], az, ev, gpst_to_bdt(t))
                ion = K * np.maximum(np.asarray(r['stec'], dtype=float).reshape(-1), 0.0) / f ** 2

            elif model == 'gim':
                if not getattr(self.config, 'gim_path', None):
                    raise ValueError("compare_models 'gim' needs gim_path (IONEX file)")
                data = _GIMReader(tec_path=self.config.gim_path).read(dcb=False)
                flh = np.asarray(self.flh, dtype=float)
                xyz = geodetic2ecef(np.array([np.deg2rad(flh[0]), np.deg2rad(flh[1]), flh[2]]))
                gim = gim_slant_tec(data, times.to_numpy('datetime64[ns]'), ev, az, xyz,
                                    mapping=getattr(self.config, 'gim_mapping_function', 'auto'),
                                    time_interpolation=getattr(self.config, 'gim_time_interpolation', 'rotated'),
                                    notes=self.config)
                # same coverage rule as read_inx: another day raises, a partial overlap warns
                check_gim_coverage(data, gim, times.to_numpy('datetime64[ns]'), source=self.config.gim_path,
                                   what=f"{self.sys} compare_models epochs")
                ion = K * gim['vtec'] * gim['m_f'] / f ** 2
            else:
                raise ValueError(f"compare_models: unknown model {model!r} (allowed: 'klobuchar', 'ntcm', 'bdgim', "
                                 f"'gim')")
            self.obs[model] = ion
            self.obs[f"{model}_tecu"] = np.asarray(ion, dtype=float) * f ** 2 / K

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

    Status:
    Active public session driver for STEC/TEC runs. It orchestrates one
    constellation at a time and routes to ``STECMonitor`` after loading,
    interpolation, preprocessing and cycle-slip detection.

    Supported systems:
    GPS (G), Galileo (E) and BeiDou (C). The concrete signal pair is selected
    from ``TECConfig.gps_freq``, ``gal_freq`` or ``bds_freq``.

    Bias/clock assumptions:
    Bias products and GIM DCB values are carried through preprocessing columns
    and then resolved by ``STECMonitor``. Receiver DCB calibration is optional
    and should be treated as model-dependent.

    Orchestrates data loading, optional broadcast‑based screening, precise/broadcast
    orbit interpolation, preprocessing (including PCO/PCV, wind‑up, path corrections),
    cycle‑slip detection, and TEC computation via STECMonitor.
    """

    CLIGHT: float = 299792458.0
    FREQ_DICT: dict[str, float] = {
        name: spec.frequency_hz for name, spec in SIGNALS.items()
    }


    def __init__(self,
                 config: TECConfig,

                 logger: Optional[logging.Logger] = None,
                 gnss_api: Optional[Any]=None,
                 use_sys: Optional[Literal['G','E','C']] = None) -> None:
        """Initialize the session controller with external orbit inputs.

        Args:
            config: TECConfig with parameters and paths.

            logger: Optional logger; if None, a default one is created.
            gnss_api: Optional GNSS backend module; if None, import the default.
            use_sys: Constellation to process ('G' or 'E').
        """

        self.config: TECConfig = config
        self.log: logging.Logger = logger or self._default_logger()
        if use_sys is not None:
            if not isinstance(use_sys, str):
                raise ValueError(f"use_sys must be one of ['C', 'E', 'G'] (got {use_sys!r})")
            check_choice("use_sys", use_sys, ("C", "E", "G"))
        selected = {use_sys} if use_sys is not None else ({self.config.sys} if isinstance(self.config.sys, str) else set(self.config.sys))
        if len(selected) != 1:
            raise ValueError("TECSession processes one constellation at a time; choose exactly one of 'G', 'E' or 'C'.")
        self.use_sys = next(iter(selected))
        # the signal pair of the processed system must be set (use_sys may differ from config.sys)
        freq_setting = {"G": "gps_freq", "E": "gal_freq", "C": "bds_freq"}[self.use_sys]
        check_choice(freq_setting, getattr(self.config, freq_setting), SUPPORTED_MODES_BY_SYSTEM[self.use_sys],
                     suffix=f" for system {self.use_sys!r}")
        # settings, model/orbit/PCO paths and the given input files, before run() reads anything;
        # obs_path itself is required at the start of run(). The validation warns as part of this session's first
        # run (bug-fix stage 13b): it starts from empty warn-once keys, and the first run() starts from a copy of
        # the keys it left, even when another session on this configuration ran or was created in between
        _notes.clear_run_notes(self.config)
        self.config.validate_for_run(required=())
        self._first_run_notes = _notes.snapshot_run_notes(self.config)
        sg_window = int(self.config.sg_window) + (1 if int(self.config.sg_window) % 2 == 0 else 0)  # odd, as applied
        if sg_window <= self.config.sg_polyorder:
            warnings.warn(
                f"sg_window={self.config.sg_window} <= sg_polyorder={self.config.sg_polyorder}: the "
                "Savitzky-Golay filter cannot be applied, leveled_tec_sg equals the unfiltered levelled TEC",
                UserWarning, stacklevel=2)


    @guarded_session_run("TECSession")
    def run(self):
        """Execute the TEC pipeline using provided positions/SP3 inputs.

        ``station_name=None`` takes the RINEX MARKER NAME for this run (receiver biases, result
        name); a ``station_name`` whose first 4 characters differ from the MARKER NAME gives a
        warning.

        Returns:
            pd.DataFrame: TEC output DataFrame with per‑arc results.

        Raises:
            SessionExecutionError whose cause is a ValueError naming the system, signal pair, station and file
            when the cycle-slip detection leaves no arc (the run returned an empty frame
            without the TEC columns, or a KeyError with receiver-DCB calibration).
        """
        station_name = self.config.station_name
        # warn-once keys are per run (bug-fix stage 13b): a configuration reused for another run warns again; the
        # first run of this session continues the notes of the validation in __init__ (gnx_py._notes)
        _notes.restore_run_notes(self.config, self.__dict__.pop("_first_run_notes", None))
        try:
            return self._run()
        finally:
            if station_name is None and self.config.station_name is not None:
                self.config.station_name = None

    def _processor(self):
        return GNSSDataProcessor2(
            obs_path=self.config.obs_path,
            nav_path=self.config.nav_path,
            dcb_path=self.config.dcb_path,
            sys=self.use_sys,
            use_gfz=self.config.use_gfz,
            # the ANTEX is read by gnx_py.antex (session_antenna_model in _run), not by io (bug-fix stage 11)
            atx_path=None,
            mode=self.config.gps_freq,
            galileo_modes=self.config.gal_freq,
            beidou_modes=self.config.bds_freq,
            station_name=self.config.station_name,
            rinex2_tracking=self.config.rinex2_tracking,
        )

    @staticmethod
    def _header_marker(processor) -> str:
        """MARKER NAME of the observation file header ('' when the header has none or cannot be read here; the
        name is then taken after reading, :meth:`_check_station_name`)."""
        try:
            header = processor._get_obs_header()
        except Exception:  # noqa: BLE001 - the full read reports file problems
            return ""
        marker = str(header.get("MARKER NAME", "") or "").strip() if isinstance(header, dict) else ""
        return "" if marker == "None" else marker

    def _check_station_name(self, obs_data) -> bool:
        """Compare ``station_name`` with the RINEX MARKER NAME (first 4 characters, any case).
        A mismatch gives a warning; ``station_name=None`` takes the MARKER NAME. True when it was taken."""
        marker = str(obs_data.meta[0]).strip() if obs_data.meta is not None and obs_data.meta[0] else ""
        if not marker or marker == "None":
            return False
        if self.config.station_name is None:
            self.log.info("station_name=None: using the RINEX MARKER NAME %r", marker)
            self.config.station_name = marker
            return True
        if str(self.config.station_name).strip()[:4].upper() != marker[:4].upper():
            warnings.warn(
                f"station_name={self.config.station_name!r} differs from the RINEX MARKER NAME {marker!r} of "
                f"{Path(str(self.config.obs_path)).name} (first 4 characters): the receiver biases and the result "
                f"name use station_name", UserWarning, stacklevel=4)
        return False

    def _run(self):
        # settings changed after construction are checked again before any file is read
        self.config.validate_for_run()

        processor = self._processor()
        if self.config.station_name is None and self.config.dcb_path:
            # the receiver biases of the product are attached while reading, by station name: take the MARKER
            # NAME from the header first, so the observations are read once
            marker = self._header_marker(processor)
            if marker:
                self.log.info("station_name=None: using the RINEX MARKER NAME %r", marker)
                self.config.station_name = marker
                processor = self._processor()
        obs_data = self._load_obs_data(processor)
        if self._check_station_name(obs_data) and self.config.dcb_path:
            # the receiver biases of the product are attached while reading, by station name
            processor = self._processor()
            obs_data = self._load_obs_data(processor)
        self.config.interval = obs_data.interval

        if self.config.nav_path:
            broadcast = self._load_nav_data(processor)
            if self.config.screen:
                gps_obs, gal_obs, bds_obs = self._screen_data(processor, obs_data.gps, obs_data.gal, obs_data.bds, broadcast)
            else:
                gps_obs, gal_obs, bds_obs = obs_data.gps, obs_data.gal, obs_data.bds
        else:
            gps_obs, gal_obs, bds_obs = obs_data.gps, obs_data.gal, obs_data.bds
            broadcast = None

        frames = {'G': gps_obs, 'E': gal_obs, 'C': bds_obs}
        if self.use_sys not in frames:
            raise ValueError(f"Unsupported TEC system: {self.use_sys}")
        mode = {'G': self.config.gps_freq, 'E': self.config.gal_freq, 'C': self.config.bds_freq}[self.use_sys]
        raw = {'G': obs_data.gps, 'E': obs_data.gal, 'C': obs_data.bds}[self.use_sys]
        span = file_span(obs_data, {'G': obs_data.gps, 'E': obs_data.gal, 'C': obs_data.bds})
        if raw is None:
            # a readable ValueError naming time_limit and the file span (formerly FileNotFoundError, bug-fix stage 10)
            raise no_observations_error(f"No {system_name(self.use_sys)} observations for this file:",
                                        [self.use_sys], self.config.obs_path, self.config.time_limit, span)
        # carrier phases of the mode's signals (a code-only file is read for SPP only; stage-10 revision)
        require_phases({self.use_sys: raw}, {self.use_sys: mode}, [self.use_sys], self.config.obs_path, "TEC")
        work_obs = frames[self.use_sys].copy()
        # antenna model (receiver and satellite PCO/PCV), once per session
        antenna = self.config.session_antenna_model(processor, [raw], {self.use_sys})

        products = self.config.sp3_path if self.config.orbit_type == "precise" else self.config.nav_path
        if len(work_obs) == 0:
            # every observation screened out (or outside the time window): readable error with the reason
            drop_empty_systems({self.use_sys: work_obs}, {self.use_sys: {"obs": 0}}, "TEC", self.config.obs_path,
                               self.config.ev_mask, orbit_type=self.config.orbit_type, products=products)
        sp3_df = None
        if self.config.orbit_type == "precise":
            # the SP3 files once, with their coverage of the observations checked (PPP-030 for TEC)
            sp3_df, _ = load_sp3(self.config.sp3_path, {self.use_sys: work_obs}, time_limit=self.config.time_limit,
                                 session="TEC", obs_path=self.config.obs_path, span=span)
        elif self.config.orbit_type != "broadcast":
            raise ValueError(f"orbit_type must be one of ['broadcast', 'precise'] (got {self.config.orbit_type!r})")
        # satellite states first, then the a-priori position (header, approx_position or code observations, D3)
        states = self._satellite_states(obs=work_obs, sys=self.use_sys, mode=mode, orbit=broadcast, sp3_df=sp3_df)
        modes = {self.use_sys: mode}
        xyz, flh, _ = resolve_position(obs_data.meta[4], obs_data.meta[5], {self.use_sys: states}, modes,
                                       floor_m=100.0, approx_position=getattr(self.config, "approx_position", None),
                                       obs_path=self.config.obs_path, session="TEC", delta_hen=obs_data.meta[3])
        xyz, flh = np.asarray(xyz, dtype=float), np.asarray(flh, dtype=float)
        n_obs = len(work_obs)
        obs_gps_crd = CustomWrapper(obs=states, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(), mode=mode).run()
        del states
        n_orbit = int(obs_gps_crd['ev'].notna().sum()) if 'ev' in obs_gps_crd.columns else len(obs_gps_crd)
        if 'ev' in obs_gps_crd.columns:
            obs_gps_crd = obs_gps_crd[obs_gps_crd['ev']>self.config.ev_mask]
        # nothing left (orbits, mask): readable error naming the reason (PPP-042 for TEC)
        drop_empty_systems({self.use_sys: obs_gps_crd}, {self.use_sys: {"obs": n_obs, "orbit": n_orbit}}, "TEC",
                           self.config.obs_path, self.config.ev_mask, orbit_type=self.config.orbit_type,
                           products=products)
        n_above_mask = len(obs_gps_crd)
        # run notes of this run (bug-fix stage 10b): the broadcast ionosphere report (run_notes.iono_report); the
        # reader warned for the rejected records of use_sys (and GPS), not of config.sys
        self.run_notes = RunNotes(nav_systems=self.use_sys)
        # SYS / PHASE SHIFT: the phases of the file are used as written (RINEX 3.0x applies the correction); the
        # reader's in-data records (contract C4) are reported, the arcs kept (DDPreprocessing.apply_phase_shift)
        obs_gps_crd = self._preprocess(obs=obs_gps_crd,
                                       flh=flh.copy(),
                                       xyz=xyz.copy(),
                                       broadcast=broadcast,
                                       phase_shift=None,
                                       sat_pco=obs_data.sat_pco,
                                       rec_pco=obs_data.rec_pco,
                                       antenna_h=obs_data.meta[3],
                                       system=self.use_sys,
                                       antenna=antenna,
                                       phase_shift_attrs=observation_phase_shift_attrs(obs_data, self.use_sys))
        obs_gps_crd = self._apply_observation_corrections(
            obs_df=obs_gps_crd,
            mode=mode,
            system=self.use_sys,
        )

        n_before_slips = len(obs_gps_crd)
        obs_gps_crd = self._detect_all_cycle_slips({self.use_sys: obs_gps_crd}, {self.use_sys: mode},
                                                   obs_data.interval, xyz)[self.use_sys]
        if len(obs_gps_crd) == 0:
            # every arc rejected (bug-fix stage 13d): the run returned an empty frame without the TEC columns and
            # without a message, or a KeyError in the receiver-DCB calibration
            raise ValueError(self._no_arc_message(mode, n_before_slips, obs_data.interval, n_above_mask))


        bias_notes: set = set()   # bias messages of this run: each one is given once
        if self._should_calibrate_station_bias(obs_gps_crd, notified=bias_notes):
            # N-ION-9: the second pass runs on the same observations plus the calibrated receiver DCB, so
            # phase_tec is K x raw L4 on both paths
            obs_gps_crd = obs_gps_crd.copy()
            obs_gps_crd['sta_dcb'] = self._calibrate_station_bias(obs_gps_crd, mode, flh, broadcast)
        monitor =  STECMonitor(obs=obs_gps_crd, mode=mode, config=self.config, sys=self.use_sys, flh=flh,
                                        broadcast=broadcast, bias_notes=bias_notes)
        obs_tec = monitor.run()
        obs_tec['lat'] = flh[0]
        obs_tec['lon'] = flh[1]
        obs_tec['h'] = flh[2]
        # ION-020: the processed system, not repr(config.sys)
        obs_tec['name'] = f"{self.config.station_name}_{self.use_sys}"
        return obs_tec

    def _load_obs_data(self, processor):
        """Load dual‑frequency observations via backend."""

        obs = processor.load_obs_data(tlim=self.config.time_limit)
        return obs

    def _load_nav_data(self, processor):
        """Load broadcast navigation messages for screening."""
        broadcast = processor.load_broadcast_orbit(tlim=self.config.time_limit)
        return broadcast

    def _screen_data(self, processor, gps_obs, gal_obs, bds_obs, broadcast):
        """Filter observations using broadcast NAV availability and basic quality checks.

        Only the processed system (``use_sys``) is screened (formerly ``config.sys``, which
        crashed for sys='G', use_sys='E' and skipped Galileo for sys='E', use_sys='G'). The outage flags in
        force are applied (``session_guards.screen_observations``); Galileo rows are dropped on the DVS/SHS
        flags of the mode's own signals only: E1 -> E1-B, E5a, E5b (formerly the names 'E1_*' were built,
        which never matched). Rows need every phase of the mode (``notna``: negative phases are kept).

        Returns:
            tuple: (gps_obs, gal_obs, bds_obs), the processed system's frame filtered
        """
        modes = {'G': self.config.gps_freq, 'E': self.config.gal_freq, 'C': self.config.bds_freq}
        out = screen_observations(processor, {'G': gps_obs, 'E': gal_obs, 'C': bds_obs}, broadcast, modes,
                                  [self.use_sys])
        return out['G'], out['E'], out['C']

    def _satellite_states(self, obs, sys, mode, orbit=None, sp3_df=None):
        """Satellite positions and clocks at emission (broadcast or precise orbits), before any use of the
        receiver position. TEC does not use the satellite clock, so ``clk_path`` is not applied here."""
        if self.config.orbit_type == "broadcast":
            make_ionofree(obs_df=obs, mode=mode, sys=sys)
            if sys not in ('G', 'E', 'C'):
                raise ValueError("Invalid sys: %s", sys)
            orb = {'G': orbit.gps_orb, 'E': orbit.gal_orb, 'C': orbit.bds_orb}[sys]
            interpolator = BroadcastInterp(obs=obs, mode=mode, sys=sys, nav=orb, emission_time=True)
            return interpolator.interpolate()
        obs = obs.swaplevel('sv', 'time')
        # the run's configuration keeps the warn-once state of the SP3 node check (once per run, not per call)
        _, positions = lagrange_reception_interp(obs=obs, sp3_df=sp3_df, owner=self.config)
        return lagrange_emission_interp(obs=obs, positions=positions, sp3_df=sp3_df, mode=mode, owner=self.config)

    def _preprocess(self, obs, flh, xyz,
                    broadcast,phase_shift,sat_pco, rec_pco, antenna_h, system, antenna=None, phase_shift_attrs=None):
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
        # geometry_free: broadcast orbits of every system keep the per-signal satellite PCO and PCV, so STEC does not
        # depend on the orbit product (bug-fix stage 11 integration; positioning routes take no satellite term with
        # broadcast orbits). K (PCO_1 - PCO_2) . u does not depend on the point the broadcast orbit refers to, so the
        # BeiDou B3I reference of decision D4 (bug-fix stage 13) leaves STEC unchanged (R4f: <= 1.6e-8 TECU)
        prc =  DDPreprocessing(df=obs, flh=flh.copy(), xyz=xyz.copy(), config=self.config, gpsa=gpsa,
                                  gpsb=gpsb, gala=gala, phase_shift_dict=phase_shift, sat_pco=sat_pco, rec_pco=rec_pco,
                                  antenna_h=antenna_h, system=system, antenna=antenna, geometry_free=True,
                                  iono=getattr(broadcast, 'iono', None), run_notes=getattr(self, 'run_notes', None),
                                  phase_shift_attrs=phase_shift_attrs)
        obs_preprocessed = prc.run()
        # Local (solar) time: GPST + longitude x 240 s (the 18 s GPST-UTC difference is negligible here)
        Lon = flh[1]
        time_gps = obs_preprocessed.index.get_level_values('time').to_numpy()
        offset = np.timedelta64(int(Lon * 3600 / 15), 's')
        time_local = time_gps + offset
        obs_preprocessed['LT']=time_local
        return obs_preprocessed

    def _detect_all_cycle_slips(self, frames_by_sys, mode_by_sys, interval, xyz):
        """Arcs of continuous phase of the frames (``interval`` in minutes, ``xyz`` the a-priori position [m]).

        The slip context (``gnx_py.cycle_slips.build_slip_context``: LLI and the third carrier from the RINEX
        file, the ``cycle_slips_detection`` master switch) is built from exactly these frames; their phases
        already carry the wind-up correction (``_apply_observation_corrections``), so the third carrier gets it
        too. Then :meth:`_detect_cycle_slips` runs per system.
        """
        from ..core.cycle_slips import GEO_BROADCAST_TEC, build_slip_context

        self._slip_context = build_slip_context(
            self.config, frames_by_sys, mode_by_sys, xyz=xyz, interval_s=float(interval) * 60.0,
            windup_applied=bool(getattr(self.config, 'windup', False)), kinematic=False,
            broadcast_geometry=GEO_BROADCAST_TEC)
        try:
            return {sys: self._detect_cycle_slips(obs_df=df, mode=mode_by_sys[sys], system=sys, interval=interval)
                    for sys, df in frames_by_sys.items()}
        finally:
            # the context belongs to these frames: a later direct call must not align its third carrier
            self._slip_context = None

    def _detect_cycle_slips(self, obs_df, mode, system, interval):
        """Detect cycle slips and form continuous arcs (``gnx_py.tools.CSDetector``).

        With the slip context of :meth:`_detect_all_cycle_slips`: RINEX loss-of-lock indicators, the
        third-carrier tests and, at 30 s sampling with precise orbits or (since v1.0.0,
        ``cycle_slips.GEO_BROADCAST_TEC``) broadcast orbits, the geometry layer (``gnx_py.cycle_slips``; one system, so its receiver-clock estimate has no helper system; the frame
        phases already carry the wind-up, which the layer then leaves out), which find equal slips on a
        close pair and on L1/L2 under the polar ionosphere; an undetected (1, 1) slip moves L4 by 0.45 TECU
        per cycle on L2L5 / E5aE5b and 0.51 TECU on L1L2. NYA2 2024-035 / 132 L1L2 with precise orbits: the
        epochs dropped as unverifiable fall from 4.44 / 3.86 % to 0.26 / 0.80 % (broadcast: 3.98 / 3.59 % with
        the layer off; since v1.0.0 with the layer and a new piece at each record switch: NYA2 2024-035 4.01 ->
        0.39 %).

        Returns:
            pd.DataFrame: DataFrame indexed by ['sv','time'] with arc segmentation.
        """
        # STEC is the geometry-free combination of two signals (the detector also takes
        # single-frequency modes, for PPP)
        if len(mode_signals(mode)) != 2:
            raise IndexError(f'Lack of dual frequency data! STEC needs a dual-frequency mode (got {mode!r})')
        context = (getattr(self, '_slip_context', None) or {}).get(system)
        detector =  CSDetector(obs=obs_df,phase_shift_dict=None,dcb=None,sys=system,mode=mode,
                                     interval=interval*60, context=context)
        obs_df = detector.run(min_arc_lenth=self.config.min_arc_len)
        obs_df = obs_df.reset_index()
        obs_df = obs_df.set_index(['arc', 'time'])
        obs_df.index.names = ['sv', 'time']
        obs_df = obs_df.drop(columns=['sv'])
        return obs_df

    def _no_arc_message(self, mode: str, n_rows: int, interval, n_above_mask: Optional[int] = None) -> str:
        """Text of the ValueError of a run with no arc left. ``n_rows`` = observations that
        reached the cycle-slip detection; 0 means the preprocessing (``skip_sat``, ``DDPreprocessing``) had
        already removed every one of the ``n_above_mask`` observations above the elevation mask."""
        cfg = self.config
        station = cfg.station_name or "the station"
        if n_rows == 0:
            above = "" if n_above_mask is None else f"all {n_above_mask} "
            return (
                f"TEC: no {system_name(self.use_sys)} {mode} observation of {station} reaches the cycle-slip "
                f"detection ({Path(str(cfg.obs_path)).name}): the preprocessing removed {above}observations above "
                f"ev_mask={cfg.ev_mask} deg. Typical causes: skip_sat naming every satellite left, or the "
                f"preprocessing excluding every satellite (for example no antenna model in the ANTEX file).")
        return (
            f"TEC: no {system_name(self.use_sys)} {mode} arc of {station} is left after the cycle-slip detection "
            f"({Path(str(cfg.obs_path)).name}): {n_rows} observations above ev_mask={cfg.ev_mask} deg were split "
            f"into arcs and every arc was cut or was shorter than min_arc_len={cfg.min_arc_len} min (sampling "
            f"{float(interval) * 60:g} s). Typical causes: loss-of-lock indicators set on most epochs of one signal "
            f"(cycle_slip_aux_carrier=False ignores the RINEX LLI), a signal tracked on a few epochs only, or a "
            f"min_arc_len longer than the data.")

    def _should_calibrate_station_bias(self, obs_df: pd.DataFrame, notified: Optional[set] = None) -> bool:
        """Whether the receiver DCB is calibrated against the GIM (``add_sta_dcb`` with
        ``rcv_dcb_source='calibrate'``, the bias policy enabled, a GIM background model with its 'ion'
        column). The satellite biases come from the chain (:meth:`_calibrate_station_bias`), so the GIM
        DCB block is not needed."""
        if not bool(getattr(self.config, "add_sta_dcb", False)):
            return False
        if getattr(self.config, "rcv_dcb_source", None) != "calibrate":
            return False
        if not bool(getattr(self.config, "stec_bias_enabled", True)):
            return False
        if getattr(self.config, "ionosphere_model", None) != "gim" or "ion" not in obs_df.columns:
            self.log.warning(
                "Skipping STEC receiver DCB calibration for %s: the calibration needs a GIM background model "
                "(ionosphere_model='gim' with its 'ion' column; got %r). Configured fallback policy will be "
                "used instead.",
                self.use_sys,
                getattr(self.config, "ionosphere_model", None),
            )
            return False
        return True

    def _calibrate_station_bias(self, obs_df: pd.DataFrame, mode: str, flh, broadcast) -> float:
        """Receiver DCB [ns] calibrated against the GIM with the satellite biases the chain applies.

        A first STECMonitor pass levels the arcs with the satellite
        biases of ``stec_bias_sources`` (one constant per arc) and without a station bias; ``Calibration``
        takes the uncalibrated levelled phase and those satellite biases (not the GIM DCB block), so the
        receiver DCB is in the datum of the applied satellite biases. Samples whose satellite bias ended at
        'zero' are left out; without any, NaN (the station bias policy then takes the next source).
        Elevation cutoff: ``TECConfig.calibration_min_elev``.
        """
        cfg = self.config
        saved = {k: getattr(cfg, k) for k in ("add_dcb", "add_sta_dcb", "stec_missing_bias", "compare_models")}
        try:
            # first pass: satellite biases only, missing ones silently 0 (the second pass warns or raises
            # under the user's stec_missing_bias); its own message set, so the second pass gives every message.
            # No comparison models: the second pass computes them (the IONEX file is read once for them)
            cfg.add_dcb, cfg.add_sta_dcb, cfg.stec_missing_bias, cfg.compare_models = True, False, "zero", False
            first = STECMonitor(obs=obs_df, mode=mode, config=cfg, sys=self.use_sys, flh=flh,
                                broadcast=broadcast, bias_notes=set()).run()
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)
        src = first["stec_sat_bias_source"].astype(str)
        applied = ~src.isin(["zero", "disabled"]).to_numpy()
        if not applied.any():
            code_1, code_2 = _code_pair_for_mode(obs_df, mode)
            self.log.warning(
                "Skipping STEC receiver DCB calibration for %s: no satellite bias of the %s pair (%s-%s) from "
                "stec_bias_sources %s on any arc. Configured fallback policy will be used instead.",
                self.use_sys, mode, code_1, code_2, tuple(cfg.stec_bias_sources))
            return float("nan")
        frame = pd.DataFrame({
            "ev": first["ev"].to_numpy(dtype=float),
            "ion": first["ion"].to_numpy(dtype=float),
            # uncalibrated levelled geometry-free phase [m] and the satellite bias of the chain [ns]
            "L4": (first["leveled_phase"] - first["stec_total_bias_m"]).to_numpy(dtype=float),
            "bias": np.where(applied, first["stec_sat_bias_m"].to_numpy(dtype=float) / NS_TO_M, np.nan),
        }, index=first.index)
        calib = Calibration(name=cfg.station_name, df=frame, sys=self.use_sys, mode=mode,
                            elev_curoff=cfg.calibration_min_elev)
        return calib.calibrate()

    def _mode_layout(self, mode: str, system: str) -> list[dict[str, str]]:
        return mode_layout(mode)

    def _get_corr(self, obs_df: pd.DataFrame, enabled: bool, column: str) -> np.ndarray:
        if enabled and column in obs_df.columns:
            return obs_df[column].to_numpy(copy=False)
        return np.zeros(len(obs_df), dtype=float)

    def _apply_observation_corrections(self, obs_df: pd.DataFrame, mode: str, system: str) -> pd.DataFrame:
        """Apply sat/rec PCO and phase windup to raw code/phase observations.

        Convention is aligned with PPP UDUC:
        - code:  C <- C - rec_pco + sat_pco
        - phase: L <- L - rec_pco + sat_pco - lambda * phw
        If a correction is disabled or missing, zeros are applied.
        """
        obs_df = obs_df.copy()
        layout = self._mode_layout(mode=mode, system=system)
        sat_enabled = bool(self.config.sat_pco)
        rec_enabled = bool(self.config.rec_pco)
        windup_enabled = bool(self.config.windup)

        for band in layout:
            code_cols = [c for c in obs_df.columns if c.startswith(band["code"])]
            phase_cols = [c for c in obs_df.columns if c.startswith(band["phase"])]
            if not code_cols and not phase_cols:
                continue

            rec = self._get_corr(obs_df=obs_df, enabled=rec_enabled, column=band["rec_pco_col"])
            sat = self._get_corr(
                obs_df=obs_df,
                enabled=sat_enabled,
                column=f"sat_pco_los_{band['freq']}",
            )
            common = -rec + sat

            if code_cols:
                obs_df.loc[:, code_cols] = obs_df[code_cols].to_numpy(copy=False) + common[:, None]

            if phase_cols:
                phw = self._get_corr(obs_df=obs_df, enabled=windup_enabled, column="phw")
                lam = self.CLIGHT / frequency_hz(band["freq"])
                phase_corr = common - lam * phw
                obs_df.loc[:, phase_cols] = obs_df[phase_cols].to_numpy(copy=False) + phase_corr[:, None]

        return obs_df

    @staticmethod
    def _default_logger() -> logging.Logger:  # noqa: D401 – simple phrase
        logger = logging.getLogger("TEC")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

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
            logging.getLogger(__name__).error("The library 'gnx_py' is not available.")
            raise ImportError(
                "GNSS backend 'gnx_py' is not installed (pip install gnx_py)"
            ) from exc
