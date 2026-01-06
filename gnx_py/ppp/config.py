from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Set, Union, Literal, Final, Generic

import pandas as pd

from .ppp_gnss import PPPDualFreqMultiGNSS
from .ppp_single import PPPDualFreqSingleGNSS,PPPSingleFreqSingleGNSS
from .ppp_uduc import PPPUdMultiGNSS, PPPUdSingleGNSS,PPPUducSFMultiGNSS, PPPFilterMultiGNSSIonConst
from ..conversion import ecef2geodetic
from ..coordinates import SP3InterpolatorOptimized, make_ionofree, emission_interp, CustomWrapper, BroadcastInterp, lagrange_reception_interp, lagrange_emission_interp
from ..io import read_sp3, parse_sinex, GNSSDataProcessor2
from ..ionosphere.config import TECConfig, STECMonitor
from ..ionosphere.ionex import GIMReader
from ..time import arange_datetime
from ..tools import CSDetector, DDPreprocessing
from ..configuration import PPPConfig
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@dataclass(slots=True)
class PPPResult:
    """Container holding PPP solution and auxiliary data."""
    solution: Union["pd.DataFrame",None]  =None
    residuals_gps: Union["pd.DataFrame",None] = None
    residuals_gal: Union["pd.DataFrame",None] = None
    convergence: Union[float, int, None] = None

    # Convenience exporters --------------------------------------------------
    def to_netcdf(self, path: Path, **kwargs: Any) -> None:
        if not path.suffix:
            path = path.with_suffix(".nc")
        self.solution.to_xarray().to_netcdf(path, **kwargs)


class PPPSession:  # noqa: D101 – docstring below
    """High‑level orchestrator for PPP processing session.

    The controller is intentionally *thin*: heavy lifting is delegated to helper
    objects from the underlying GNX library
    """


    def __init__(
        self,
        config: PPPConfig,
    ) -> None:
        self.config: PPPConfig = config
        self.FREQ_DICT = {'L1': 1575.42e06, 'L2': 1227.60e06,
                          'E1': 1575.42e06, 'E5a': 1176.45e06}

    def run(self):

        processor = GNSSDataProcessor2(
                atx_path=self.config.atx_path,
                obs_path=self.config.obs_path,
                dcb_path=self.config.dcb_path,
                nav_path=self.config.nav_path,
                mode=self.config.gps_freq,
                sys=self.config.sys,
                galileo_modes=self.config.gal_freq,
                use_gfz = self.config.use_gfz,
            )

        selected = set(self.config.sys)
        allowed = {'G', 'E'}
        unknown = selected - allowed
        if unknown:
            raise ValueError(f"Unsupported systems in config.sys: {unknown}. Allowed: {allowed}")

        obs_data = self._load_obs_data(processor)
        xyz, flh = obs_data.meta[4], obs_data.meta[5]

        obs_by_sys = {'G': obs_data.gps, 'E': obs_data.gal}

        missing = [s for s in selected if obs_by_sys.get(s) is None]
        if missing:
            raise ValueError(f"Selected systems {missing} but no observations found in {self.config.obs_path}")

        broadcast = None
        if self.config.nav_path:
            broadcast = self._load_nav_data(processor)

        if self.config.nav_path and self.config.screen:
            gps_obs, gal_obs = self._screen_data(processor, obs_by_sys['G'], obs_by_sys['E'], broadcast)
            obs_by_sys['G'], obs_by_sys['E'] = gps_obs, gal_obs

        # sp3_df tylko gdy potrzebny
        sp3_df = None
        if self.config.orbit_type == "precise":
            sp3 = [read_sp3(f) for f in self.config.sp3_path]
            sp3_df = (pd.concat(sp3)
                      .reset_index().drop_duplicates()
                      .set_index(['time', 'sv']))
            sp3_df[['x', 'y', 'z']] = sp3_df[['x', 'y', 'z']].values * 1e3
            sp3_df['clk'] = sp3_df['clk'].values * 1e-6

        mode_by_sys = {'G': self.config.gps_freq, 'E': self.config.gal_freq}

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
                f = self.FREQ_DICT[mode]
                tec2m = K / (f ** 2)

                df = self._measure_STEC(df=crd_by_sys[sys], system=sys)
                df.loc[:, 'ion'] = (df['leveled_tec'] / 1e16) * tec2m
                crd_by_sys[sys] = df

        for sys in selected:
            crd_by_sys[sys] = self._sanitize(df=crd_by_sys[sys])

        obs_gps_crd = crd_by_sys.get('G')
        obs_gal_crd = crd_by_sys.get('E')

        result = self._run_filter(
            obs_data=obs_data,
            obs_gps_crd=obs_gps_crd,
            obs_gal_crd=obs_gal_crd,
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
            # df = df.join(sat_dcb, on='sv')
            # print(df[['bias']])
            # if self.config.station_name is not None:
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
        monitor =  STECMonitor(config=tec_config,sys=system, mode=mode,obs=df)
        out = monitor.run()
        return out

    def _load_nav_data(self, processor):
        """ load broadcast navigation messages for screening"""
        broadcast = processor.load_broadcast_orbit()
        return broadcast

    # ‑‑‑ private helpers ----------------------------------------------------
    def _load_obs_data(self, processor):  # noqa: D401 – simple phrase OK
        """Read observation & navigation files, apply basic screenings."""
        obs = processor.load_obs_data()

        return obs

    def _screen_data(self, processor, gps_obs, gal_obs, broadcast):
        if self.config.system_includes('G') and gps_obs is not None:
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

    def create_if_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval):
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty

        if has_gps and has_gal:
            print('GPS + GAL')
            # Multi-GNSS (GPS + Galileo)
            return PPPDualFreqMultiGNSS(
                gps_obs=obs_gps_crd.copy(),
                gal_obs=obs_gal_crd.copy(),
                gps_mode=gps_mode,
                gal_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
            )
        elif has_gps:
            print('GPS')
            # Single-GNSS / GPS
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gps_crd.copy(),
                gps_mode=gps_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
            )
        elif has_gal:
            print('GAL')
            # Single-GNSS na Galileo (we pass the GAL mode as gps_mode)
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gal_crd.copy(),
                gps_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
            )
        else:
            raise ValueError("No GPS or Galileo observations after preprocessing (None/empty).")

    def create_uduc_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval):
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty

        if self.config.use_iono_constr and (has_gps and has_gal):
            print('GPS + GAL Constrained')
            return  PPPFilterMultiGNSSIonConst(gps_obs=obs_gps_crd,gps_mode=gps_mode,
                                              gal_obs=obs_gal_crd,gal_mode=gal_mode,
                                              ekf=ekf,pos0=pos0,tro=True,est_dcb=True,interval=interval,use_iono_rms=self.config.use_iono_rms,
                                               config=self.config)
        else:
            if has_gps and has_gal:
                print('GPS + GAL Unconstrained')
                return  PPPUdMultiGNSS(gps_obs=obs_gps_crd,gps_mode=gps_mode,
                                              gal_obs=obs_gal_crd,gal_mode=gal_mode,
                                              ekf=ekf,pos0=pos0,tro=True, interval=interval,config=self.config)
            elif has_gps:
                print('GPS')
                return  PPPUdSingleGNSS(obs=obs_gps_crd,mode=self.config.gps_freq,ekf=ekf,pos0=pos0,tro=True,interval=interval,config=self.config)
            elif has_gal:
                print('GAL')
                return  PPPUdSingleGNSS(obs=obs_gal_crd,mode=self.config.gal_freq,ekf=ekf,pos0=pos0,tro=True,interval=interval,config=self.config)
    def create_sf_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval):
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty
        if has_gps and has_gal:
            print('GPS + GAL')
            return PPPUducSFMultiGNSS(gps_obs=obs_gps_crd,gps_mode=gps_mode,
                                              gal_obs=obs_gal_crd,gal_mode=gal_mode,
                                              ekf=ekf,pos0=pos0,tro=True,interval=interval,
                                      config=self.config)
        else:
            if has_gps:
                print('GPS')
                obs = obs_gps_crd
                mode = gps_mode
            elif has_gal:
                print('GAL')
                obs = obs_gal_crd
                mode = gal_mode
            return PPPSingleFreqSingleGNSS(obs=obs,mode=mode,ekf=ekf,pos0=pos0,interval=interval,config=self.config)

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, interval) -> PPPResult:  # noqa: D401 – simple phrase
        from filterpy.kalman import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(dim_x=1,dim_z=1)
        pos0=obs_data.meta[4]

        if self.config.positioning_mode == 'combined':
            print('Creating IF filter')
            ppp_filter = self.create_if_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval)
        elif self.config.positioning_mode == 'uncombined':
            print('Creating UDUC filter')
            ppp_filter = self.create_uduc_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval)
        elif self.config.positioning_mode == 'single':
            print('Creating Single filter')
            ppp_filter = self.create_sf_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval)


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



        return PPPResult(solution=ppp_result,residuals_gps=ppp_gps, residuals_gal=ppp_gal, convergence=conv_time)

