"""PPP session orchestration and filter selection.

This module contains the in-package ``PPPSession`` class, which is the main
runtime coordinator for PPP processing. It is separate from the top-level
``PPPSession.py`` script: the script is a runnable driver, while this module
loads/preprocesses data and routes a ``PPPConfig`` instance to the concrete
combined, uncombined, single-system, mixed-system, and ionosphere-constrained
filter classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Set, Union, Literal, Final, Generic

import pandas as pd

from .ppp_gnss import PPPDualFreqMultiGNSS
from .ppp_single import PPPDualFreqSingleGNSS,PPPSingleFreqSingleGNSS
from .ppp_uduc import (
    PPPUdGenericMixedGNSS,
    PPPUdMultiGNSS,
    PPPUdSingleGNSS,
    PPPUdMixedGNSS,
    PPPUducSFMultiGNSS,
    PPPUducSFSingleGNSS,
    PPPFilterMultiGNSSIonConst,
    PPPFilterMultiGNSSIonConstGEC,
)
from ..conversion import ecef2geodetic
from ..coordinates import SP3InterpolatorOptimized, make_ionofree, emission_interp, CustomWrapper, BroadcastInterp, lagrange_reception_interp, lagrange_emission_interp
from ..io import read_sp3, parse_sinex, GNSSDataProcessor2
from ..ionosphere.config import TECConfig, STECMonitor
from ..ionosphere.ionex import GIMReader
from ..time import arange_datetime
from ..tools import CSDetector, DDPreprocessing
from ..configuration import PPPConfig
from ..session_errors import guarded_session_run
from ..gnss import frequency_hz, mode_signals
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

SINGLE_FREQUENCY_MODES: Final[set[str]] = {"L1", "L2", "L5", "E1", "E5a", "E5b", "B1I", "B1C", "B2a", "B2I", "B2b", "B3I"}
DUAL_FREQUENCY_MODES: Final[set[str]] = {
    "L1L2",
    "L1L5",
    "L2L5",
    "E1E5a",
    "E1E5b",
    "E5aE5b",
    "B1IB2I",
    "B1IB3I",
    "B1CB2a",
    "B1CB2b",
    "B1CB3I",
    "B2aB2b",
}


@dataclass(slots=True)
class PPPResult:
    """Container for PPP solution products and diagnostics.

    Status:
        Active public result container returned by ``PPPSession.run``.

    Fields:
        ``solution`` is the epoch-indexed position/filter output produced by a
        PPP filter. Typical columns include ENU errors, receiver clock, ZTD,
        ISB/DCB estimates, satellite counts, residual summaries, and PPP-AR
        diagnostic columns when ambiguity resolution is enabled.

        ``residuals_gps``, ``residuals_gal`` and ``residuals_bds`` hold
        per-system residual/observation diagnostics when the selected filter
        returns them. Some legacy two-system filters return GPS/Galileo frames
        positionally; the session normalizes those artifacts into these fields.

        ``convergence`` is the convergence time reported by the filter, usually
        in hours or ``None`` when the filter did not declare convergence.
    """
    solution: Union["pd.DataFrame",None]  =None
    residuals_gps: Union["pd.DataFrame",None] = None
    residuals_gal: Union["pd.DataFrame",None] = None
    residuals_bds: Union["pd.DataFrame",None] = None
    convergence: Union[float, int, None] = None

    # Convenience exporters --------------------------------------------------
    def to_netcdf(self, path: Path, **kwargs: Any) -> None:
        if not path.suffix:
            path = path.with_suffix(".nc")
        self.solution.to_xarray().to_netcdf(path, **kwargs)


class PPPSession:  # noqa: D101 – docstring below
    """High-level orchestrator for a PPP processing session.

    Purpose:
        Drives the in-package PPP workflow: observation loading, optional
        broadcast navigation loading, precise/broadcast orbit interpolation,
        preprocessing, cycle-slip detection, optional STEC/GIM preparation,
        filter selection, filter execution, and normalization into
        ``PPPResult``.

    Status:
        Active session/router class. It is the supported in-package entry point
        for PPP execution and the authoritative location for filter routing.

    Config routing:
        The session reads ``PPPConfig`` but does not define configuration
        defaults itself. ``positioning_mode='combined'`` routes to ionosphere-
        free combined filters. ``positioning_mode='uncombined'`` routes by
        active systems, signal mode cardinality, and ``use_iono_constr``.
        ``positioning_mode='single'`` is treated as a deprecated alias for
        uncombined single-frequency routing.

    Supported systems:
        GPS (``G``), Galileo (``E``) and BeiDou (``C``), depending on available
        observations and configured modes.

    Warnings:
        This class is a routing/orchestration layer, not a mathematical PPP
        model. Numerical behavior lives in the selected filter class.
    """


    def __init__(
        self,
        config: PPPConfig,
    ) -> None:
        self.config: PPPConfig = config
        self.FREQ_DICT = {
            signal: frequency_hz(signal)
            for signal in ("L1", "L2", "L5", "E1", "E5a", "E5b", "B1I", "B1C", "B2a", "B2I", "B2b", "B3I")
        }

    @guarded_session_run("PPPSession")
    def run(self):

        processor = GNSSDataProcessor2(
                atx_path=self.config.atx_path,
                obs_path=self.config.obs_path,
                dcb_path=self.config.dcb_path,
                nav_path=self.config.nav_path,
                mode=self.config.gps_freq,
                sys=self.config.sys,
                galileo_modes=self.config.gal_freq,
                beidou_modes=self.config.bds_freq,
                use_gfz = self.config.use_gfz,
                station_name=self.config.station_name,
            )

        selected = set(self.config.sys)
        allowed = {'G', 'E', 'C'}
        unknown = selected - allowed
        if unknown:
            raise ValueError(f"Unsupported systems in config.sys: {unknown}. Allowed: {allowed}")
        obs_data = self._load_obs_data(processor)
        xyz, flh = obs_data.meta[4], obs_data.meta[5]

        obs_by_sys = {'G': obs_data.gps, 'E': obs_data.gal, 'C': obs_data.bds}

        missing = [s for s in selected if obs_by_sys.get(s) is None]
        if missing:
            raise ValueError(f"Selected systems {missing} but no observations found in {self.config.obs_path}")

        broadcast = None

        if self.config.nav_path:
            broadcast = self._load_nav_data(processor)

        if self.config.nav_path and self.config.screen:
            gps_obs, gal_obs, bds_obs = self._screen_data(processor, obs_by_sys['G'], obs_by_sys['E'], obs_by_sys['C'], broadcast)
            obs_by_sys['G'], obs_by_sys['E'], obs_by_sys['C'] = gps_obs, gal_obs, bds_obs
        # sp3_df tylko gdy potrzebny
        sp3_df = None
        if self.config.orbit_type == "precise":
            sp3 = [read_sp3(f) for f in self.config.sp3_path]
            sp3_df = (pd.concat(sp3)
                      .reset_index().drop_duplicates()
                      .set_index(['time', 'sv']))
            sp3_df[['x', 'y', 'z']] = sp3_df[['x', 'y', 'z']].values * 1e3
            sp3_df['clk'] = sp3_df['clk'].values * 1e-6
        mode_by_sys = {'G': self.config.gps_freq, 'E': self.config.gal_freq, 'C': self.config.bds_freq}

        crd_by_sys = {}
        for sys in selected:
            obs = obs_by_sys[sys]
            mode = mode_by_sys[sys]

            if self.config.orbit_type == "broadcast":
                crd = self._interpolate_broadcast(
                    obs_df=obs, xyz=xyz.copy(), flh=flh.copy(),
                    sys=sys, mode=mode, orbit=broadcast,
                    tolerance=self.config.broadcast_tolerance
                )
            else:  # precise
                crd = self._interpolate_lgr(
                    obs=obs, xyz=xyz.copy(), flh=flh.copy(),
                    mode=mode, degree=self.config.interpolation_degree,
                    sp3_df=sp3_df
                )

            if 'ev' in crd.columns:
                crd = crd[crd['ev'] > self.config.ev_mask]

            crd_by_sys[sys] = crd

        for sys in selected:
            crd_by_sys[sys] = self._preprocess(
                obs=crd_by_sys[sys],
                flh=flh.copy(),
                xyz=xyz.copy(),
                broadcast=broadcast,
                phase_shift=obs_data.meta[-1],
                sat_pco=obs_data.sat_pco,
                rec_pco=obs_data.rec_pco,
                antenna_h=obs_data.meta[3],
                system=sys
            )

        for sys in selected:
            crd_by_sys[sys] = self._detect_cycle_slips(
                obs_df=crd_by_sys[sys],
                mode=mode_by_sys[sys],
                system=sys,
                interval=obs_data.interval
            )

        if self.config.ionosphere_model == "measure STEC":
            K = 40.3e16
            for sys in selected:
                mode = mode_by_sys[sys]
                f = self.FREQ_DICT[mode_signals(mode)[0]]
                tec2m = K / (f ** 2)

                df = self._measure_STEC(df=crd_by_sys[sys], system=sys)
                df.loc[:, 'ion'] = (df['leveled_tec'] / 1e16) * tec2m
                crd_by_sys[sys] = df

        for sys in selected:
            crd_by_sys[sys] = self._sanitize(df=crd_by_sys[sys])

        obs_gps_crd = crd_by_sys.get('G')
        obs_gal_crd = crd_by_sys.get('E')
        obs_bds_crd = crd_by_sys.get('C')
        result = self._run_filter(
            obs_data=obs_data,
            obs_gps_crd=obs_gps_crd,
            obs_gal_crd=obs_gal_crd,
            obs_bds_crd=obs_bds_crd,
            interval=obs_data.interval
        )
        return result

    def _sanitize(self, df):
        cols = ['xe', 'ye', 'ze', 'clk']
        return df.dropna(subset=cols)

    def _measure_STEC(self, df:pd.DataFrame, system:str):
        # DOWNLOAD DCB FROM GIM
        parser = GIMReader(tec_path=self.config.gim_path,dcb_path=self.config.gim_path)
        data= parser.read()
        if data.dcb is not None:
            sat_dcb = data.dcb[(data.dcb['entry_type'] == 'satellite') &
                               data.dcb['sv'].str.startswith(system)]
            sat_dcb = sat_dcb.set_index('sv')
            df['sv_main'] = df.index.get_level_values('sv')
            df['sv_main'] = df['sv_main'].apply(lambda x: x[:3])
            sat_dcb = sat_dcb.reset_index()
            merged = df.merge(sat_dcb, left_on='sv_main',right_on='sv',how='left')
            merged = merged.set_index(df.index)
            merged = merged.drop(columns='sv')
            df=merged.copy()
            if 'bias' in df.columns:
                df['bias'] = df['bias'].fillna(0.0)

            name = self.config.station_name
            station = (data.dcb[(data.dcb['entry_type'] == 'station') &
                                (data.dcb['prn_or_site'].str.startswith(name))])
            sta_bias = station[station['sys'] == system]
            if not sta_bias.empty:
                df['sta_bias'] = sta_bias['bias'].values[0]
            else:
                df['sta_bias'] = 0.0
        # MEASURE STEC
        ## create a config
        tec_config =  TECConfig(station_name = self.config.station_name,day_of_year = self.config.day_of_year)
        ## create a monitor
        if system =='G':
            mode = self.config.gps_freq
        elif system =='E':
            mode = self.config.gal_freq
        elif system == 'C':
            mode = self.config.bds_freq
        else:
            raise ValueError(f"Unsupported system for STEC measurement: {system}")
        monitor =  STECMonitor(config=tec_config,sys=system, mode=mode,obs=df)
        out = monitor.run()
        return out

    def _load_nav_data(self, processor):
        """ load broadcast navigation messages for screening"""
        broadcast = processor.load_broadcast_orbit(tlim=self.config.time_limit)
        return broadcast

    # ‑‑‑ private helpers ----------------------------------------------------
    def _load_obs_data(self, processor):  # noqa: D401 – simple phrase OK
        """Read observation & navigation files, apply basic screenings."""
        obs = processor.load_obs_data(tlim=self.config.time_limit)

        return obs

    def _screen_data(self, processor, gps_obs, gal_obs, bds_obs, broadcast):
        if self.config.system_includes('G') and gps_obs is not None:
            gps_nav = processor.screen_navigation_message(broadcast.gps_orb, 'G')
            gps_obs = processor.mark_outages(gps_obs, 'G', gps_nav)
            gps_obs = gps_obs[gps_obs['gps_outage'] == False]
            gps_obs = gps_obs[(gps_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        if self.config.system_includes('E') and gal_obs is not None:
            gal_nav = processor.screen_navigation_message(broadcast.gal_orb, 'E')
            gal_obs = processor.mark_outages(gal_obs, 'E', gal_nav)
            gal_obs = gal_obs[(gal_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        if self.config.system_includes('C') and bds_obs is not None:
            bds_nav = processor.screen_navigation_message(broadcast.bds_orb, 'C')
            bds_obs = processor.mark_outages(bds_obs, 'C', bds_nav)
            bds_obs = bds_obs[bds_obs['bds_outage'] == False]
            bds_obs = bds_obs[(bds_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        return gps_obs, gal_obs, bds_obs


    def _interpolate_broadcast(self, obs_df, xyz, flh,sys,mode, orbit, tolerance):
        make_ionofree(obs_df=obs_df, mode=mode, sys=sys)
        if sys == 'G':
            orb = orbit.gps_orb
        elif sys == 'E':
            orb = orbit.gal_orb
        elif sys == 'C':
            orb = orbit.bds_orb
        else:
            raise ValueError("Invalid sys: %s", sys)

        interpolator =BroadcastInterp(obs=obs_df,mode=mode,sys=sys,nav=orb,emission_time=True,tolerance=tolerance)
        obs_crd = interpolator.interpolate()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _interpolate_lgr(self, obs, flh, xyz, mode,degree, sp3_df):
        make_ionofree(obs_df=obs, mode=mode)
        obs = obs.swaplevel('sv','time')


        _, positions = lagrange_reception_interp(obs=obs,sp3_df=sp3_df,degree=degree)
        obs_crd = lagrange_emission_interp(obs=obs,positions=positions,sp3_df=sp3_df,degree=degree)
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _preprocess(self, obs, flh, xyz,
                    broadcast,phase_shift,sat_pco, rec_pco, antenna_h, system):

        if broadcast is None:
            gpsa, gpsb, gala = None, None, None
        else:
            gpsa, gpsb, gala = broadcast.gpsa, broadcast.gpsb, broadcast.gala
        prc =  DDPreprocessing(df=obs, flh=flh.copy(), xyz=xyz.copy(), config=self.config, gpsa=gpsa,
                                  gpsb=gpsb, gala=gala, phase_shift_dict=phase_shift, sat_pco=sat_pco, rec_pco=rec_pco,
                                  antenna_h=antenna_h, system=system)
        obs_preprocessed = prc.run()
        return obs_preprocessed

    def _detect_cycle_slips(self, obs_df, mode, system, interval):
        detector =  CSDetector(obs=obs_df,phase_shift_dict=None,dcb=None,sys=system,mode=mode,
                                     interval=interval*60)
        obs_df = detector.run(min_arc_lenth=self.config.min_arc_len)
        obs_df = obs_df.reset_index()
        obs_df = obs_df.set_index(['arc', 'time'])
        obs_df['arc']=obs_df.index.get_level_values('arc')
        obs_df.index.names = ['sv', 'time']
        obs_df = obs_df.drop(columns=['sv'])
        return obs_df

    def create_if_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        """Create a combined ionosphere-free PPP filter from active systems.

        Mixed-system input routes to ``PPPDualFreqMultiGNSS`` with a reference
        constellation clock and per-non-reference ISB states. Single-system
        input routes to ``PPPDualFreqSingleGNSS`` for GPS, Galileo or BeiDou.
        All branches require dual-frequency modes because the observables are
        formed as ionosphere-free combinations before filtering.

        Status:
            Active routing helper. It should not be changed without checking
            combined single-system and mixed-system regression tests.
        """
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq
        bds_mode = self.config.bds_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty
        has_bds = isinstance(obs_bds_crd, pd.DataFrame) and not obs_bds_crd.empty

        if sum((has_gps, has_gal, has_bds)) >= 2:
            return PPPDualFreqMultiGNSS(
                gps_obs=obs_gps_crd.copy() if has_gps else None,
                gal_obs=obs_gal_crd.copy() if has_gal else None,
                bds_obs=obs_bds_crd.copy() if has_bds else None,
                gps_mode=gps_mode,
                gal_mode=gal_mode,
                bds_mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
            )
        elif has_gps:
            # Single-GNSS / GPS
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gps_crd.copy(),
                gps_mode=gps_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
                system="G",
            )
        elif has_gal:
            # Single-GNSS na Galileo (we pass the GAL mode as gps_mode)
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gal_crd.copy(),
                gps_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
                system="E",
            )
        elif has_bds:
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_bds_crd.copy(),
                gps_mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
                system="C",
            )
        else:
            raise ValueError("No GPS, Galileo or BeiDou observations after preprocessing (None/empty).")

    def _frequency_kind(self, mode: str) -> str:
        if mode in SINGLE_FREQUENCY_MODES:
            return "single"
        if mode in DUAL_FREQUENCY_MODES:
            return "dual"
        raise ValueError(f"Unsupported uncombined frequency mode: {mode}")

    def _create_uncombined_single_system_filter(self, obs, mode, ekf, pos0, interval):
        """Create the single-system uncombined filter for one signal mode.

        Single-frequency modes route either to ``PPPUducSFSingleGNSS`` when
        ionospheric constraints are enabled, or to ``PPPSingleFreqSingleGNSS``
        without constraints. Dual-frequency modes route to ``PPPUdSingleGNSS``.

        Status:
            Active routing helper for single-system uncombined PPP.
        """
        kind = self._frequency_kind(mode)
        use_iono = bool(self.config.use_iono_constr)

        if kind == "single":
            if use_iono:
                return PPPUducSFSingleGNSS(
                    obs=obs,
                    mode=mode,
                    ekf=ekf,
                    pos0=pos0,
                    tro=True,
                    interval=interval,
                    use_iono_rms=self.config.use_iono_rms,
                    sigma_iono_0=self.config.sigma_iono_0,
                    sigma_iono_end=self.config.sigma_iono_end,
                    t_end=self.config.t_end,
                    config=self.config,
                )
            return PPPSingleFreqSingleGNSS(
                obs=obs,
                mode=mode,
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        return PPPUdSingleGNSS(
            obs=obs,
            mode=mode,
            ekf=ekf,
            pos0=pos0,
            tro=True,
            interval=interval,
            config=self.config,
        )

    def create_uncombined_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        """Create the uncombined PPP filter selected by systems and constraints.

        Routing summary:
            * Multiple systems without ionospheric constraints use the generic
              mixed uncombined model.
            * G/E/C or broadcast constrained mixed runs use the generic G/E/C
              ionospheric-constraint branch.
            * Legacy G/E dual-frequency constrained runs use
              ``PPPFilterMultiGNSSIonConst`` as the reference implementation.
            * Older G/E mixed/single-frequency paths remain available for
              compatibility and require caution before behavioral changes.

        Status:
            Active routing helper. Several legacy/reference classes remain
            reachable here, so status labels must not be interpreted as removal
            permission.
        """
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq
        bds_mode = self.config.bds_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty
        has_bds = isinstance(obs_bds_crd, pd.DataFrame) and not obs_bds_crd.empty
        use_iono = bool(self.config.use_iono_constr)
        iono_constraints_model = getattr(self.config, "uncombined_iono_constraints_model", "legacy")

        active = [s for s, present in (('G', has_gps), ('E', has_gal), ('C', has_bds)) if present]
        if len(active) > 1 and not use_iono:
            obs_by_system = {
                "G": obs_gps_crd,
                "E": obs_gal_crd,
                "C": obs_bds_crd,
            }
            mode_by_system = {
                "G": gps_mode,
                "E": gal_mode,
                "C": bds_mode,
            }
            return PPPUdGenericMixedGNSS(
                obs_by_system={system: obs_by_system[system] for system in active},
                mode_by_system={system: mode_by_system[system] for system in active},
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if use_iono and len(active) > 1 and (
            has_bds
            or iono_constraints_model == "gec"
            or self.config.orbit_type == "broadcast"
        ):
            obs_by_system = {
                "G": obs_gps_crd,
                "E": obs_gal_crd,
                "C": obs_bds_crd,
            }
            mode_by_system = {
                "G": gps_mode,
                "E": gal_mode,
                "C": bds_mode,
            }
            return PPPFilterMultiGNSSIonConstGEC(
                obs_by_system={system: obs_by_system[system] for system in active},
                mode_by_system={system: mode_by_system[system] for system in active},
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if has_bds and has_gps and not has_gal:
            return PPPUdMixedGNSS(
                gps_obs=obs_gps_crd,
                gps_mode=gps_mode,
                gal_obs=obs_bds_crd,
                gal_mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if has_bds and has_gal and not has_gps:
            return PPPUdMixedGNSS(
                gps_obs=obs_gal_crd,
                gps_mode=gal_mode,
                gal_obs=obs_bds_crd,
                gal_mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if has_gps and has_gal:
            gps_kind = self._frequency_kind(gps_mode)
            gal_kind = self._frequency_kind(gal_mode)

            if gps_kind == "single" and gal_kind == "single":
                if use_iono:
                    return PPPUdMixedGNSS(
                        gps_obs=obs_gps_crd,
                        gps_mode=gps_mode,
                        gal_obs=obs_gal_crd,
                        gal_mode=gal_mode,
                        ekf=ekf,
                        pos0=pos0,
                        tro=True,
                        interval=interval,
                        config=self.config,
                    )
                return PPPUducSFMultiGNSS(
                    gps_obs=obs_gps_crd,
                    gps_mode=gps_mode,
                    gal_obs=obs_gal_crd,
                    gal_mode=gal_mode,
                    ekf=ekf,
                    pos0=pos0,
                    tro=True,
                    interval=interval,
                    config=self.config,
                )

            if gps_kind == "dual" and gal_kind == "dual":
                if use_iono:
                    return PPPFilterMultiGNSSIonConst(
                        gps_obs=obs_gps_crd,
                        gps_mode=gps_mode,
                        gal_obs=obs_gal_crd,
                        gal_mode=gal_mode,
                        ekf=ekf,
                        pos0=pos0,
                        tro=True,
                        est_dcb=True,
                        interval=interval,
                        use_iono_rms=self.config.use_iono_rms,
                        config=self.config,
                    )
                return PPPUdMultiGNSS(
                    gps_obs=obs_gps_crd,
                    gps_mode=gps_mode,
                    gal_obs=obs_gal_crd,
                    gal_mode=gal_mode,
                    ekf=ekf,
                    pos0=pos0,
                    tro=True,
                    interval=interval,
                    config=self.config,
                )

            return PPPUdMixedGNSS(
                gps_obs=obs_gps_crd,
                gps_mode=gps_mode,
                gal_obs=obs_gal_crd,
                gal_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if has_gps:
            return self._create_uncombined_single_system_filter(
                obs=obs_gps_crd,
                mode=gps_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
            )

        if has_gal:
            return self._create_uncombined_single_system_filter(
                obs=obs_gal_crd,
                mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
            )

        if has_bds:
            return self._create_uncombined_single_system_filter(
                obs=obs_bds_crd,
                mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
            )

        raise ValueError("No GPS, Galileo or BeiDou observations after preprocessing (None/empty).")

    def create_uduc_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        return self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)

    def create_sf_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        return self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, obs_bds_crd, interval) -> PPPResult:  # noqa: D401 – simple phrase
        """Select, run, and normalize the configured PPP filter."""
        from filterpy.kalman import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(dim_x=1,dim_z=1)
        pos0=obs_data.meta[4]

        if self.config.positioning_mode == 'combined':
            ppp_filter = self.create_if_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)
        elif self.config.positioning_mode == 'uncombined':
            ppp_filter = self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)
        elif self.config.positioning_mode == 'single':
            warnings.warn(
                "positioning_mode='single' is deprecated; use positioning_mode='uncombined' with a single-frequency mode.",
                DeprecationWarning,
                stacklevel=2,
            )
            ppp_filter = self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)
        else:
            raise ValueError(f"Unsupported positioning_mode: {self.config.positioning_mode}")


        sta_name = obs_data.meta[0][:4].upper()

        if self.config.sinex_path:
            snx = parse_sinex(self.config.sinex_path)
            if sta_name in snx.index:
                ref = snx.loc[sta_name].to_numpy()
                flh =  ecef2geodetic(ecef=ref, deg=True)
                ppp_result, ppp_gps, ppp_gal, conv_time = ppp_filter.run_filter(ref=ref.copy(),
                                                   flh=flh.copy(),
                                                    reset_every=self.config.reset_every,
                                                    trace_filter=self.config.trace_filter)

            else:
                ppp_result,ppp_gps, ppp_gal, conv_time = ppp_filter.run_filter(trace_filter=False,reset_every=self.config.reset_every)
        else:
            ppp_result, ppp_gps, ppp_gal, conv_time = ppp_filter.run_filter(
                                                                            reset_every=self.config.reset_every,
                                                                            trace_filter=self.config.trace_filter)



        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty
        has_bds = isinstance(obs_bds_crd, pd.DataFrame) and not obs_bds_crd.empty

        if isinstance(ppp_gps, dict):
            residuals_gps = ppp_gps.get('G')
            residuals_gal = ppp_gps.get('E')
            residuals_bds = ppp_gps.get('C')
        else:
            residuals_gps, residuals_gal, residuals_bds = ppp_gps, ppp_gal, None
            if has_bds and not has_gps and not has_gal:
                residuals_gps, residuals_gal, residuals_bds = None, None, ppp_gps
            elif has_bds and has_gps and not has_gal:
                residuals_gps, residuals_gal, residuals_bds = ppp_gps, None, ppp_gal
            elif has_bds and has_gal and not has_gps:
                residuals_gps, residuals_gal, residuals_bds = None, ppp_gps, ppp_gal

        return PPPResult(solution=ppp_result,residuals_gps=residuals_gps, residuals_gal=residuals_gal, residuals_bds=residuals_bds, convergence=conv_time)
