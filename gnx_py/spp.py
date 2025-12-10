from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Literal, Set, Final, Any
from pathlib import Path
from datetime import datetime
from .configuration import Config
import pandas as pd
import logging
import numpy as np
from .utils import calculate_distance
from .conversion import ecef_to_enu, ecef2geodetic
from .coordinates import BroadcastInterp, lagrange_emission_interp, lagrange_reception_interp, CustomWrapper, make_ionofree
from .io import GNSSDataProcessor2, read_sp3, parse_sinex
from .ppp.preprocessing import DDPreprocessing

@dataclass(slots=True)
class SPPResult:
    """Container holding PPP solution and auxiliary data."""
    solution: Union["pd.DataFrame",None]  =None
    residuals_gps: Union["pd.DataFrame",None] = None
    residuals_gal: Union["pd.DataFrame",None] = None
    covariance_info: Union["pd.DataFrame",None] = None

    # Convenience exporters --------------------------------------------------
    def to_netcdf(self, path: Path, **kwargs: Any) -> None:
        if not path.suffix:
            path = path.with_suffix(".nc")
        self.solution.to_xarray().to_netcdf(path, **kwargs)

@dataclass()
class SPPConfig(Config):

    solver: Literal["LS"] ="LS"



class SinglePointPositioning:

    def __init__(self, config, gps_obs: pd.DataFrame, gal_obs: Union[pd.DataFrame, None], gps_mode, gal_mode,
                 solver='LSQ', xyz_apr=np.zeros(3)):
        self.config = config
        self.gps_obs = gps_obs
        self.gal_obs = gal_obs
        self.gps_mode = gps_mode
        self.gal_mode = gal_mode
        self.solver = solver
        self.FREQ_DICT = {'L1': 1575.42e06, 'E1': 1575.42e06,
                          'L2': 1227.60e06, 'E5a': 1176.45e06,
                          'L5': 1176.450e06, 'E5b': 1207.14e06}
        self.xyz_apr = xyz_apr

    def hjacobian(self, x, gps_sats, gal_sats=None):

        # GPS
        dxyz = -(gps_sats - x[:3])
        dnorm = np.linalg.norm(dxyz, axis=1)
        A = np.zeros(shape=(len(gps_sats), 4)) if gal_sats is None else np.zeros((len(gps_sats) + len(gal_sats), 5))
        for i in range(len(gps_sats)):
            if gal_sats is not None:
                A[i, :] = [dxyz[i, 0] / dnorm[i], dxyz[i, 1] / dnorm[i], dxyz[i, 2] / dnorm[i], 1, 0]
            else:
                A[i, :] = [dxyz[i, 0] / dnorm[i], dxyz[i, 1] / dnorm[i], dxyz[i, 2] / dnorm[i], 1]
        # GAL
        if gal_sats is not None:
            dxyz_gal = -(gal_sats - x[:3])
            dnorm_gal = np.linalg.norm(dxyz_gal, axis=1)
            for i in range(len(gal_sats)):
                idx = len(gps_sats) + i
                A[idx, :] = [dxyz_gal[i, 0] / dnorm_gal[i], dxyz_gal[i, 1] / dnorm_gal[i], dxyz_gal[i, 2] / dnorm_gal[i],
                             0, 1]
        return A

    def hx(self, x, gps_sats, gal_sats=None):

        dxyz = -(gps_sats - x[:3])
        dist = np.sqrt(dxyz[:, 0] ** 2 + dxyz[:, 1] ** 2 + dxyz[:, 2] ** 2)
        if gal_sats is None:
            return dist + x[-1]
        else:
            dxyz_gal = -(gal_sats - x[:3])
            dist_gal = np.sqrt(dxyz_gal[:, 0] ** 2 + dxyz_gal[:, 1] ** 2 + dxyz_gal[:, 2] ** 2)
            hx_gal = dist_gal + x[-1]
            hx_gps = dxyz + x[-2]
            return np.concatenate((hx_gps, hx_gal))

    def code_screening(self, omc, err, n_sigma=3):
        md = np.median(omc)
        mask = np.where(np.abs(omc) > n_sigma * md)
        if len(mask[0]) > len(omc) / 2:
            for ind in mask:
                err[ind] /= 100
        return err

    def observed(self, gps_epoch, gal_epoch):
        """
                Builds 'observed' vectors (metres) for GPS and Galileo depending on the mode.
                Assumptions regarding column units:
                  - C1*/C2*/C5* pseudo-ranges: metres
                  - tro, ion, dprel: metres (if you have any in seconds, convert them beforehand)
                  - clk, TGD, BGD*: seconds  (clk understood as dt_r - dt_s)
                Clock convention:
                  - we assume clk = dt_r - dt_s  ⇒  we add +c*(clk - TGD/BGD)
        """
        C = 299_792_458.0

        def first_col(df, prefixes):
            """Return the name of the first existing column starting with a given prefix."""
            for p in prefixes:
                for c in df.columns:
                    if c.startswith(p):
                        return c
            return None

        def get_arr(df, name, default=0.0, dtype='float64'):
            """Get the column as an np.ndarray; if missing, fill with a constant."""
            if name in df.columns:
                return np.asarray(df[name].values, dtype=dtype)
            # brak kolumny: zwróć stałą o długości N
            N = len(df)
            return np.full(N, default, dtype=dtype)

        def get_any(df, names, default=0.0):
            """Get the first one from the list of possible columns; if there are none, return 0."""
            for n in names:
                if n in df.columns:
                    return np.asarray(df[n].values, dtype='float64')
            return np.full(len(df), default, dtype='float64')

        # ---------- GPS ----------
        gps_obs = None
        if gps_epoch is not None and len(gps_epoch) > 0:
            if self.gps_mode == 'L1':
                c1 = first_col(gps_epoch, ['C1C', 'C1W', 'C1', 'C1P'])
                if c1 is None:
                    raise ValueError("GPS L1: column C1* missing in gps_epoch.")
                P = get_arr(gps_epoch, c1)

                tro = get_any(gps_epoch, ['tro'], 0.0)
                ion = get_any(gps_epoch, ['ion'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                TGD_s = get_any(gps_epoch, ['TGD'], 0.0)  # s
                dtr = get_any(gps_epoch, ['dtr'], 0.0)
                pco_l1 = get_any(gps_epoch, ['pco_los_l1'], 0.0)


                clk_term_m = C * (clk_s - TGD_s)
                gps_obs = P - tro - ion - dprel + clk_term_m - dtr + pco_l1

            elif self.gps_mode == 'L1L2':
                # IF(L1,L2): k1*C1 - k2*C2  + c*clk  - tro  - dprel
                c1 = first_col(gps_epoch, ['C1C', 'C1W', 'C1', 'C1P'])
                c2 = first_col(gps_epoch, ['C2W', 'C2L', 'C2P', 'C2'])
                if c1 is None or c2 is None:
                    raise ValueError("GPS L1L2: no C1* or C2* columns in gps_epoch.")

                f1 = self.FREQ_DICT['L1']
                f2 = self.FREQ_DICT['L2']
                k1 = (f1 ** 2) / (f1 ** 2 - f2 ** 2)
                k2 = (f2 ** 2) / (f1 ** 2 - f2 ** 2)

                P1 = get_arr(gps_epoch, c1)
                P2 = get_arr(gps_epoch, c2)
                tro = get_any(gps_epoch, ['tro'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                pco_l1 = get_any(gps_epoch, ['pco_los_l1'], 0.0)
                pco_l2 = get_any(gps_epoch, ['pco_los_l2'], 0.0)
                pco = (k1*pco_l1 - k2*pco_l2)
                gps_obs = (k1 * P1 - k2 * P2) + C * clk_s - tro - dprel + pco

            else:
                raise ValueError(f"Unknown gps_mode: {self.gps_mode}")

        # ---------- Galileo ----------
        gal_obs = None
        if gal_epoch is not None and len(gal_epoch) > 0 and getattr(self, 'gal_mode', None):
            if self.gal_mode == 'E1':
                # E1 single: P_E1 - tro - ion - dprel + c*(clk - BGD_E1*)
                c1 = first_col(gal_epoch, ['C1C', 'C1X', 'C1A', 'C1'])
                if c1 is None:
                    c1 = first_col(gal_epoch, ['C1B'])
                if c1 is None:
                    raise ValueError("GAL E1: column C1* missing in gal_epoch.")

                P = get_arr(gal_epoch, c1)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                ion = get_any(gal_epoch, ['ion'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)

                # BGD (sekundy) – próbujemy kilka popularnych nazw
                BGD_s = get_any(
                    gal_epoch,
                    ['BGD', 'BGDe5a', 'BGDe5b', 'BGD_E1E5a', 'BGD_E1E5b'],
                    0.0
                )

                clk_term_m = C * (clk_s - BGD_s)
                gal_obs = P - tro - ion - dprel + clk_term_m

            elif self.gal_mode == 'E1E5a':
                # IF(E1,E5a): k1*C1 - k2*C5  + c*clk  - tro  - dprel
                c1 = first_col(gal_epoch, ['C1C', 'C1X', 'C1W', 'C1'])
                c5 = first_col(gal_epoch, ['C5Q', 'C5X', 'C5', 'C5A'])
                if c1 is None or c5 is None:
                    raise ValueError("GAL E1E5a: columns C1* or C5* are missing in gal_epoch.")

                f1 = self.FREQ_DICT['E1']
                f5a = self.FREQ_DICT['E5a']
                k1 = (f1 ** 2) / (f1 ** 2 - f5a ** 2)
                k2 = (f5a ** 2) / (f1 ** 2 - f5a ** 2)

                P1 = get_arr(gal_epoch, c1)
                P5 = get_arr(gal_epoch, c5)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)

                gal_obs = (k1 * P1 - k2 * P5) + C * clk_s - tro - dprel

            else:
                raise ValueError(f"Unknown gal_mode: {self.gal_mode}")

        # ---------- Zwracanie zgodnie z Twoją konwencją ----------
        if gal_obs is None:
            return gps_obs, gps_obs, None
        else:
            all_obs = np.concatenate((gps_obs, gal_obs)) if gps_obs is not None else gal_obs
            return all_obs, gps_obs, gal_obs

    def lsq(self, ref=None):
        results_rows = []  # <= zbiorczy output (po epokach)

        xyz_apr = self.xyz_apr.copy()
        if self.gal_obs is not None:
            cmn = self.gps_obs.index.get_level_values('time').intersection(self.gal_obs.index.get_level_values('time'))
            epochs = sorted(cmn.get_level_values('time').unique().tolist())
        else:
            epochs = sorted(self.gps_obs.index.get_level_values('time').unique().tolist())
        for num, t in enumerate(epochs):
            gps_epoch = self.gps_obs.loc[(slice(None), t), :]
            gps_sats = gps_epoch[['xe', 'ye', 'ze']].to_numpy()
            if self.gal_obs is not None:
                gal_epoch = self.gal_obs.loc[(slice(None), t), :]
                gal_sats = gal_epoch[['xe', 'ye', 'ze']].to_numpy()
                if len(gal_sats) + len(gps_sats)<5:continue
            else:
                gal_sats = None
                gal_epoch = None
                if len(gps_sats)<5:continue

            A = self.hjacobian(x=xyz_apr, gps_sats=gps_sats, gal_sats=gal_sats)

            observed, observed_gps, observed_gal = self.observed(gps_epoch=gps_epoch, gal_epoch=gal_epoch)
            computed_gps = calculate_distance(gps_sats, xyz_apr[:3])
            L_gps = observed_gps - computed_gps

            if self.gal_obs is not None:
                computed_gal = calculate_distance(gal_sats, xyz_apr[:3])
                L_gal = observed_gal - computed_gal
                L_all = np.concatenate((L_gps, L_gal))
            else:
                computed_gal = None
                L_gal = None
                L_all = L_gps

            # Wagi
            err_gps = 0.5 / np.sin(np.deg2rad(gps_epoch['ev'])) ** 2
            err_gps = self.code_screening(L_gps, err_gps)
            if self.gal_obs is not None:
                err_gal = 0.5 / np.sin(np.deg2rad(gal_epoch['ev'])) ** 2
                err_gal = self.code_screening(L_gal, err_gal)
                err = np.concatenate((err_gps, err_gal))
            else:
                err = err_gps

            W = np.linalg.pinv(np.diag(err))
            Q = np.linalg.pinv(A.T.dot(W).dot(A))
            X = Q.dot(A.T.dot(W).dot(L_all))

            # Iteracje (pozycja + dtr)
            obl = xyz_apr[:3] + X[:3]

            tol = 1e-3
            max_iter = 5
            it = 0

            gps_epoch = gps_epoch.copy()
            gps_epoch.loc[:, 'dtr'] = X[-1]

            while True:
                x_prev = X.copy()
                A = self.hjacobian(x=obl, gps_sats=gps_sats, gal_sats=gal_sats)
                observed, observed_gps, observed_gal = self.observed(gps_epoch=gps_epoch, gal_epoch=gal_epoch)
                computed_gps = calculate_distance(gps_sats, obl)
                L_gps = observed_gps - computed_gps

                if self.gal_obs is not None:
                    computed_gal = calculate_distance(gal_sats, obl)
                    L_gal = observed_gal - computed_gal
                    L_all = np.concatenate((L_gps, L_gal))
                else:
                    L_all = L_gps

                W = np.linalg.pinv(np.diag(err))
                Q = np.linalg.pinv(A.T.dot(W).dot(A))
                X = Q.dot(A.T.dot(W).dot(L_all))

                obl = obl + X[:3]
                gps_epoch.loc[:, 'dtr'] = X[-1]

                it += 1
                if np.max(np.abs(X - x_prev)) < tol:
                    break
                if it >= max_iter:
                    break


            if self.gal_obs is not None:
                L_all = np.concatenate((L_gps, L_gal))
            else:
                L_all = L_gps
            V_all = A @ X - L_all


            n_gps = len(L_gps)
            self.gps_obs.loc[(slice(None), t), 'V'] = V_all[:n_gps]
            if self.gal_obs is not None:
                self.gal_obs.loc[(slice(None), t), 'V'] = V_all[n_gps:]


            sig = np.sqrt(np.clip(np.diag(Q), 0.0, np.inf))
            sig_x, sig_y, sig_z = (sig[0], sig[1], sig[2])
            sig_dtr = sig[3] if len(sig) > 3 else np.nan

            rms_v = float(np.sqrt(np.mean(V_all ** 2))) if V_all.size else np.nan

            results_rows.append({
                'time': t,
                'x': float(obl[0]),
                'y': float(obl[1]),
                'z': float(obl[2]),
                'dtr_gps': float(X[-2]) if gal_epoch is not None else float(X[-1]),
                'dtr_gal':float(X[-1]) if gal_epoch is not None else None,
                'sig_x': float(sig_x),
                'sig_y': float(sig_y),
                'sig_z': float(sig_z),
                'sig_dtr': float(sig_dtr),
                'n_gps': int(n_gps),
                'n_gal': int(len(L_gal)) if L_gal is not None else 0,
                'iters': int(it),
                'rms_v': rms_v,
            })

            xyz_apr[:3] = obl

            if ref is not None:
                dif = ref - obl
                ref_flh = ecef2geodetic(ref)
                denu = ecef_to_enu(dif, flh=ref_flh,degrees=True)
                results_rows[-1].update({'de': float(denu[0]), 'dn': float(denu[1]), 'du': float(denu[2])})
                if getattr(self.config,'track_filter',False):
                    print(f'Epoch: {t} error de: {denu[0]} dn: {denu[1]} du: {denu[2]}')

        results_df = pd.DataFrame(results_rows)#.set_index('time').sort_index()

        return results_df, self.gps_obs, (self.gal_obs if self.gal_obs is not None else None)



class SPPSession:  # noqa: D101 – docstring below
    """High‑level orchestrator for PPP processing session.

    The controller is intentionally *thin*: heavy lifting is delegated to helper
    objects from the underlying GNSS library – injected via the constructor or
    at call‑time so they can be mocked in unit‑tests.
    """


    def __init__(
        self,
        config: SPPConfig,
        logger: Optional[logging.Logger] = None,

    ) -> None:
        self.config: SPPConfig = config
        self.FREQ_DICT = {'L1': 1575.42e06, 'L2': 1227.60e06,
                          'E1': 1575.42e06, 'E5a': 1176.45e06}


    # ‑‑‑ public API ---------------------------------------------------------
    def run(self) -> SPPResult:  # noqa: C901 – orchestration, acceptable

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
        obs_data = self._load_obs_data(processor)

        if self.config.sys == {'G','E'}:
            if obs_data.gal is None:
                raise ValueError(f'Selected systems: Galileo and GPS. No galileo observations for: {self.config.obs_path}')

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

                # ELEVATION MASK
            if 'ev' in obs_gps_crd.columns:
                obs_gps_crd = obs_gps_crd[obs_gps_crd['ev'] > self.config.ev_mask]
        else:
            obs_gps_crd = None


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



        result = self._run_filter(obs_data=obs_data,obs_gps_crd=obs_gps_crd, obs_gal_crd=obs_gal_crd, interval=obs_data.interval)

        return result

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, interval)->SPPResult:
        solver = SinglePointPositioning(config=self.config,gps_obs=obs_gps_crd,gal_obs=obs_gal_crd,
                                        gps_mode=self.config.gps_freq,gal_mode=self.config.gal_freq,xyz_apr=obs_data.meta[4])
        if self.config.sinex_path:
            sta_name = obs_data.meta[0][:4].upper()
            snx = parse_sinex(self.config.sinex_path)
            if sta_name in snx.index:
                ref = snx.loc[sta_name].to_numpy()
                result, gps_result, gal_result = solver.lsq(ref=ref)
            else:
                result, gps_result, gal_result = solver.lsq()
        else:
            result, gps_result, gal_result = solver.lsq()
        return SPPResult(solution=result,residuals_gps=gps_result,residuals_gal=gal_result)


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

        interpolator =BroadcastInterp(obs=obs_df,mode=mode,sys=sys,nav=orb,emission_time=True,tolerance=tolerance)
        obs_crd = interpolator.interpolate()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _interpolate_lgr(self, obs, flh, xyz, mode,degree):
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



