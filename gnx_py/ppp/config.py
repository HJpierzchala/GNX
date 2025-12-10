from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Set, Union, Literal, Final, Generic

import pandas as pd

from .ppp_gnss import PPPDualFreqMultiGNSS
from .ppp_single import PPPDualFreqSingleGNSS
from .ppp_uduc import PPPUdMultiGNSS, PPPUdSingleGNSS,PPPUducSFMultiGNSS, PPPFilterMultiGNSSIonConst
from .preprocessing import DDPreprocessing
from ..conversion import ecef2geodetic
from ..coordinates import SP3InterpolatorOptimized, make_ionofree, emission_interp, CustomWrapper, BroadcastInterp, lagrange_reception_interp, lagrange_emission_interp
from ..io import read_sp3, parse_sinex, GNSSDataProcessor2
from ..ionosphere.config import TECConfig, STECMonitor
from ..ionosphere.ionex import GIMReader
from ..time import arange_datetime
from ..tools import CSDetector
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
    objects from the underlying GNSS library – injected via the constructor or
    at call‑time so they can be mocked in unit‑tests.
    """

    #: Magic numbers collected here for clarity (defaults – may be overridden)
    _DEFAULT_P_CRD: Final[float] = 1e3
    _DEFAULT_Q_DT: Final[float] = 9e9
    _DEFAULT_MAX_ARC_LEN_MIN: Final[int] = 60

    def __init__(
        self,
        config: PPPConfig,
        positions: Optional[Union[pd.DataFrame, None]]=None,
        sp3: Optional[Union[list,None]]=None,
        logger: Optional[logging.Logger] = None,

    ) -> None:
        self.config: PPPConfig = config
        self.FREQ_DICT = {'L1': 1575.42e06, 'L2': 1227.60e06,
                          'E1': 1575.42e06, 'E5a': 1176.45e06}
        self.positions=positions 
        self.sp3=sp3 

    # ‑‑‑ public API ---------------------------------------------------------
    def run(self) -> PPPResult:  # noqa: C901 – orchestration, acceptable
        # --------- Tworzymy processor dla przetwarzania danych wsadowych
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
        # --------- Pobieramy dane obserwacyjne
        obs_data = self._load_obs_data(processor)
        # --------- Jezeli wybralismy G+E, a E nie ma => błąd
        if self.config.sys == {'G','E'}:
            if obs_data.gal is None:
                raise ValueError(f'Selected systems: Galileo and GPS. No galileo observations for: {self.config.obs_path}')
        # --------- przyblizone wspolrzedne z naglowka
        xyz, flh = obs_data.meta[4], obs_data.meta[5]
        # --------- Jezeli uzywamy danych broadcast, wczytujemy je
        if self.config.nav_path:
            broadcast = self._load_nav_data(processor)
            # --------- jezeli chcemy screenowac obserwacje, robimy to tutaj
            if self.config.screen:
                gps_obs, gal_obs = self._screen_data(processor, obs_data.gps, obs_data.gal, broadcast)
            else:
                gps_obs, gal_obs = obs_data.gps, obs_data.gal
        else:
            gps_obs, gal_obs = obs_data.gps, obs_data.gal
            broadcast = None


        # -------------- GPS: Interpolacja na czas emisji sygnału
        if gps_obs is not None:
            if self.config.orbit_type == "broadcast":
                obs_gps_crd = self._interpolate_broadcast(obs_df=gps_obs,
                                                        xyz=xyz.copy(),
                                                        flh=flh.copy(),
                                                        sys='G',
                                                        mode=self.config.gps_freq,
                                                        orbit=broadcast,
                                                        tolerance=self.config.broadcast_tolerance)
            elif self.config.orbit_type == "precise":

                obs_gps_crd = self._interpolate_lgr(obs=gps_obs,xyz=xyz.copy(),flh=flh.copy(),
                                                        mode=self.config.gps_freq,degree=self.config.interpolation_degree)
            if 'ev' in obs_gps_crd.columns:
                obs_gps_crd = obs_gps_crd[obs_gps_crd['ev'] > self.config.ev_mask]

        else:
            obs_gps_crd = None

        # ---------- GALILEO: Interpolacja i maska elewacji
        if gal_obs is not None:
            if self.config.orbit_type == "broadcast":
                obs_gal_crd = self._interpolate_broadcast(obs_df=gal_obs,
                                                        xyz=xyz.copy(),
                                                        flh=flh.copy(),
                                                        sys='E',
                                                        mode=self.config.gal_freq,
                                                        orbit=broadcast,
                                                        tolerance=self.config.broadcast_tolerance)
            elif self.config.orbit_type == "precise":

                obs_gal_crd = self._interpolate_lgr(obs=gal_obs, xyz=xyz.copy(), flh=flh.copy(),
                                                        mode=self.config.gal_freq,degree=self.config.interpolation_degree)
            if 'ev' in obs_gal_crd.columns:
                obs_gal_crd = obs_gal_crd[obs_gal_crd['ev'] > self.config.ev_mask]
        else:
            obs_gal_crd = None




        # --------- Preprocessing jest taki sam dla UDUC i IF, z wyjatkiem pobierania informacji o jonosferze ---------
        # -- Preprocessing obserwacji GPS
        if obs_gps_crd is not None:
            obs_gps_crd = self._preprocess(obs=obs_gps_crd,
                                           flh=flh.copy(),
                                           xyz=xyz.copy(),
                                           broadcast=broadcast,
                                           phase_shift=obs_data.meta[-1],
                                           sat_pco=obs_data.sat_pco,
                                           rec_pco=obs_data.rec_pco,
                                           antenna_h=obs_data.meta[3],
                                           system='G')

        # -- Preprocessing obserwacji GAL
        if obs_gal_crd is not None:
            obs_gal_crd = self._preprocess(obs=obs_gal_crd,
                                           flh=flh.copy(),
                                           xyz=xyz.copy(),
                                           broadcast=broadcast,
                                           phase_shift=obs_data.meta[-1],
                                           sat_pco=obs_data.sat_pco,
                                           rec_pco=obs_data.rec_pco,
                                           antenna_h=obs_data.meta[3],
                                           system='E')
        # --------- Przeskoki fazy - musza byc dwie czestotliwosci
        # cycle slips GPS

        if obs_gps_crd is not None:
            obs_gps_crd = self._detect_cycle_slips(obs_df=obs_gps_crd,
                                                   mode=self.config.gps_freq,
                                                   system='G',
                                                   interval=obs_data.interval
                                                   )



        if obs_gal_crd is not None:
            # cycle slips GAL
            obs_gal_crd = self._detect_cycle_slips(obs_df=obs_gal_crd,
                                                   mode=self.config.gal_freq,
                                                   system='E',
                                                   interval=obs_data.interval
                                                   )

        # --------- Jezeli chcemy uzyc pomierzonego STEC. Uwaga - DCB odbiornika pobierane z pliku IONEX ---------
        # --------- Przyda sie zewnetrzne zrodlo sta_bias
        if self.config.ionosphere_model =="measure STEC":
            fg = self.FREQ_DICT[self.config.gps_freq]
            K=40.3e16
            tec2m = K /fg**2


            obs_gps_crd = self._measure_STEC(df=obs_gps_crd,system='G')
            obs_gps_crd.loc[:,'ion'] = (obs_gps_crd['leveled_tec']/1e16) * tec2m
            if obs_gal_crd is not None:
                fe = self.FREQ_DICT[getattr(self.config,'gal_freq','E1E5a')]
                tec2m = K/fe**2
                obs_gal_crd = self._measure_STEC(df=obs_gal_crd,system='E')
                obs_gal_crd.loc[:,'ion'] = (obs_gal_crd['leveled_tec']/1e16)*tec2m

        result=self._run_filter(obs_data=obs_data,obs_gps_crd=obs_gps_crd,obs_gal_crd=obs_gal_crd,interval=obs_data.interval)
        return result

    def _measure_STEC(self, df:pd.DataFrame, system:str):
        # POBIERAMY DCB Z GIM
        parser = GIMReader(tec_path=self.config.gim_path,dcb_path=self.config.gim_path)
        data= parser.read()
        if data.dcb is not None:
            sat_dcb = data.dcb[(data.dcb['entry_type'] == 'satellite') &
                               data.dcb['sv'].str.startswith(system)]
            sat_dcb = sat_dcb.set_index('sv')
            # print(sat_dcb)
            # print(sat_dcb.info())
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
            sta_bias = station[station['system'] == system]
            if not sta_bias.empty:
                df['sta_bias'] = sta_bias['bias'].values[0]
            else:
                df['sta_bias'] = 0.0
        # MIERZYMY STEC
        ## tworzymy config
        tec_config =  TECConfig(station_name = self.config.station_name,day_of_year = self.config.day_of_year)
        ## tworzymy Monitor
        if system =='G':
            mode = self.config.gps_freq
        elif system =='E':
            mode = self.config.gal_freq
        monitor =  STECMonitor(config=tec_config,sys=system, mode=mode,obs=df)
        out = monitor.run()
        return out

        # return result
    def _load_nav_data(self, processor):
        """ load broadcast navigation messages for screening"""
        broadcast = processor.load_broadcast_orbit()
        return broadcast

    # ‑‑‑ private helpers ----------------------------------------------------
    def _load_obs_data(self, processor):  # noqa: D401 – simple phrase OK
        """Read observation & navigation files, apply basic screenings."""
        # Actual implementation should call the GNSS library. Here we only log.
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
            raise ValueError("Invalid system: %s", sys)

        interpolator =BroadcastInterp(obs=obs_df,mode=mode,sys=sys,nav=orb,emission_time=True)
        obs_crd = interpolator.interpolate()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _interpolate_lgr(self, obs, flh, xyz, mode,degree):
        make_ionofree(obs_df=obs, mode=mode)
        obs = obs.swaplevel('sv','time')
        start = datetime.now()
        sp3 = [read_sp3(f) for f in self.config.sp3_path]
        sp3_df = pd.concat(sp3)
        sp3_df = sp3_df.reset_index().drop_duplicates()
        sp3_df = sp3_df.set_index(['time', 'sv'])
        sp3_df[['x','y','z']]=sp3_df[['x','y','z']].apply(lambda x: x*1e3)
        sp3_df['clk'] = sp3_df['clk'].apply(lambda x: x * 1e-6)
        _, positions = lagrange_reception_interp(obs=obs,sp3_df=sp3_df,degree=degree)
        obs_crd = lagrange_emission_interp(obs=obs,positions=positions,sp3_df=sp3_df,degree=degree)
        end=datetime.now()
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
            # Multi-GNSS (GPS + Galileo)
            return PPPDualFreqMultiGNSS(
                gps_obs=obs_gps_crd.copy(),
                gal_obs=obs_gal_crd.copy(),
                gps_mode=gps_mode,
                gal_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval
            )
        elif has_gps:
            # Single-GNSS na GPS
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gps_crd.copy(),
                gps_mode=gps_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval
            )
        elif has_gal:
            # Single-GNSS na Galileo (przekazujemy tryb GAL jako gps_mode —
            # jeśli konstruktor ma inną sygnaturę, dostosuj nazwę parametru)
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gal_crd.copy(),
                gps_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval
            )
        else:
            raise ValueError("Brak obserwacji GPS ani Galileo po preprocessingu (None/empty).")

    def create_uduc_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval):
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty

        if self.config.use_iono_constr:
            return  PPPFilterMultiGNSSIonConst(gps_obs=obs_gps_crd,gps_mode=gps_mode,
                                              gal_obs=obs_gal_crd,gal_mode=gal_mode,
                                              ekf=ekf,pos0=pos0,tro=True,est_dcb=True,interval=interval,use_iono_rms=self.config.use_iono_rms)
        else:
            if has_gps and has_gal:
                return  PPPUdMultiGNSS(gps_obs=obs_gps_crd,gps_mode=gps_mode,
                                              gal_obs=obs_gal_crd,gal_mode=gal_mode,
                                              ekf=ekf,pos0=pos0,tro=True, interval=interval)
            elif has_gps:
                return  PPPUdSingleGNSS(obs=obs_gps_crd,mode=self.config.gps_freq,ekf=ekf,pos0=pos0,tro=True,interval=interval)
            elif has_gal:
                return  PPPUdSingleGNSS(obs=obs_gal_crd,mode=self.config.gal_freq,ekf=ekf,pos0=pos0,tro=True,interval=interval)
    def create_sf_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval):
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty
        if has_gps and has_gal:
            return PPPUducSFMultiGNSS(gps_obs=obs_gps_crd,gps_mode=gps_mode,
                                              gal_obs=obs_gal_crd,gal_mode=gal_mode,
                                              ekf=ekf,pos0=pos0,tro=True,interval=interval)
        else:
            raise NotImplementedError("Lack of data for one or more systems")

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, interval) -> PPPResult:  # noqa: D401 – simple phrase
        from filterpy.kalman import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(dim_x=1,dim_z=1)
        pos0=obs_data.meta[4]

        if self.config.positioning_mode == 'combined':
            ppp_filter = self.create_if_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval)
        elif self.config.positioning_mode == 'uncombined':
            ppp_filter = self.create_uduc_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval)
        elif self.config.positioning_mode == 'single':
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
    # ‑‑‑ static helpers -----------------------------------------------------
    @staticmethod
    def _default_logger() -> logging.Logger:  # noqa: D401 – simple phrase
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        return logging.getLogger("PPP")

    @staticmethod
    def _import_gnss_api():  # -> ModuleType
        """Import jedynego wspieranego backendu `gps_lib`.

        Rzuca ImportError, jeśli biblioteka nie jest zainstalowana.
        """
        import importlib, sys, logging

        try:
            module = importlib.import_module("gps_lib")
            # (opcjonalnie) alias, jeżeli gdzieś indziej spodziewamy się `import gps`
            sys.modules.setdefault("gps", module)
            return module
        except ModuleNotFoundError as exc:
            logging.getLogger(__name__).error("Brak biblioteki 'gps_lib'.")
            raise ImportError(
                "GNSS backend 'gps_lib' nie jest zainstalowany (pip install gps_lib)"
            ) from exc
