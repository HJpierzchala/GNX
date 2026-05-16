"""Mixed-constellation combined PPP filter classes.

This module contains ionosphere-free combined PPP implementations for multiple
constellations. The active generic class supports GPS, Galileo and BeiDou with
a reference receiver clock and inter-system-bias states. Older G/E-specific
classes remain for compatibility and should be changed only with numerical
regression coverage.
"""

import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter

from ..conversion import ecef_to_enu
from ..utils import calculate_distance
from ..configuration import PPPConfig
from .ppp_helpers import (
    code_screening as ppp_code_screening,
    phase_residuals_outliers,
    trace_epoch_summary as ppp_trace_epoch,
    trace_message as ppp_trace,
)
from .pppar import (
    PPPARSettings,
    advance_arc_age,
    apply_conventional_pppar,
    arc_age_array,
    combined_if_pppar_enabled,
    pppar_diagnostic_columns,
)
from .ppp_single import PPPDualFreqSingleGNSS as _SingleCombinedBiasModel
from ..gnss import SIGNALS, mode_ionosphere_free_coefficients, mode_layout, mode_signals, signal_spec

FREQ_DICT_ALL = {name: spec.frequency_hz for name, spec in SIGNALS.items()}

class PPPSingleFreqMultiGNSS:
    """Single-frequency mixed GPS/Galileo PPP without ionospheric constraints.

    Purpose:
        Legacy mixed-system single-frequency path. It is retained for
        compatibility with older routing and validation scenarios.

    Status:
        Legacy compatibility class. It is exported through the broad PPP package
        API, but the current ``PPPSession`` routing prefers uncombined generic
        classes for mixed single-frequency work. Do not remove until public API
        usage and historical tests are audited.

    Model:
        Uses one code/phase pair per satellite for GPS and Galileo, with a
        reference GPS clock, Galileo ISB, optional ZTD, and one ambiguity per
        satellite. It does not estimate slant ionosphere states.

    State vector:
        ``[x, y, z, dtr_G, isb_E, ztd?, N_G..., N_E...]``.

    Supported systems/modes:
        GPS plus Galileo single-frequency modes.

    Limitations:
        Legacy path. Prefer the generic uncombined/mixed classes for new work,
        especially when BeiDou or ionospheric constraints are involved.
    """

    def __init__(self, config:PPPConfig,gps_obs, gal_obs, gps_mode, gal_mode, ekf, tro=False, pos0=None,interval=0.5):

        """
        PPP Single Frequency Multi GNSS estimator
        :param gps_obs: pd.DataFrame, gps observations
        :param gal_obs: pd.DataFrame, gal observations
        :param gps_mode: str, gps signals
        :param gal_mode: str, gal signals
        :param ekf: filterpy.ExtendedKalmanFilter instance
        :param tro: bool, estimate ZTD
        :param pos0: np.ndarray, initial position
        :param interval: float, interval of observations
        """
        self.cfg = config
        self.gps_obs = gps_obs
        self.gal_obs =gal_obs
        self.gps_mode =gps_mode
        self.gal_mode = gal_mode
        self.tro =tro
        self.ekf = ekf
        self.FREQ_DICT = {'L1' :1575.42e06 ,'E1' :1575.42e06,
                          'L2' :1227.60e06 ,'E5a' :1176.45e06,
                          'L5' :1176.450e06 ,'E5b' :1207.14e06}
        self.CLIGHT = 299792458
        self.base_dim = 6 if self.tro else 5
        self.pos0 = pos0
        self.interval=interval

    def HJacobian(self, x, gps_satellites, gal_satellites):
        """
        Hjacobian matrix
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :param gal_satellites: np.ndarray, gal satellite coordinates & mapping function
        :return: Hjacobian matrix
        """
        num_gps = len(gps_satellites)
        num_gal = len(gal_satellites)
        N = num_gps + num_gal
        # State: x,y,z, dt_gps, tro, isb, N1...Nn
        dim_state = self.base_dim + N

        receiver = x[:3]
        # GPS PART
        H1 = np.zeros((2 * num_gps, dim_state))
        distances_gps = np.linalg.norm((gps_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
        for i in range(num_gps):
            # Code observation (wiersz i)
            H1[i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[i, 3] = 1.0  # clock
            H1[i, 4] = 0.0 # ISB
            H1[i,5]=gps_satellites[i,-1]

            # Phase observation (wiersz N+i)
            H1[num_gps + i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[num_gps + i, 3] = 1.0  # clock
            if self.tro:
                H1[num_gps+i,5] = gps_satellites[i,-1]
                H1[num_gps+i,4] = 0.0 # ISB

            H1[num_gps + i, self.base_dim + i] = 1  # AMB
        # GAL PART
        H2 = np.zeros((2 * num_gal, dim_state))
        distances_gal = np.linalg.norm((gal_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
        for i in range(num_gal):
            # Code observation (wiersz i)
            H2[i, 0:3] = (receiver - gal_satellites[i, :3]) / distances_gal[i]
            H2[i, 3] = 1.0  # clock
            H2[i, 4] = 1.0  # isb
            H2[i, 5] = gal_satellites[i,-1]
            if self.tro:
                H2[num_gal+i, 5] = gal_satellites[i,-1]
                H2[num_gal + i, 4] = 1.0 # ISB
            # Phase observation (wiersz N+i)
            H2[num_gal + i, 0:3] = (receiver - gal_satellites[i, :3]) / distances_gal[i]
            H2[num_gal + i, 3] = 1.0  # clock

            H2[num_gal + i, (self.base_dim + num_gps) + i] = 1  # AMB
        H = np.vstack((H1, H2))
        return H

    def Hx(self, x ,gps_satellites, gal_satellites):
        """
        Predicted observations vector
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :param gal_satellites: np.ndarray, gal satellite coordinates & mapping function
        :return: predicted obs matrix
        """
        num_gps = len(gps_satellites)
        num_gal = len(gal_satellites)
        receiver = self.ekf.x[:3]
        clock = self.ekf.x[3]
        isb = self.ekf.x[4]
        if self.tro:
            tro = self.ekf.x[5]
            gal_tro = gal_satellites[:,-1]*tro
            gps_tro = gps_satellites[:,-1]*tro
        else:
            gps_tro = 0.0
            gal_tro = 0.0
        # GPS prediction
        distances_gps = np.linalg.norm((gps_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
        h_code_gps = distances_gps + clock+gps_tro
        ambiguities_gps = x[self.base_dim:self.base_dim + num_gps]
        h_phase_gps = distances_gps + clock + ambiguities_gps+gps_tro
        # GAL prediction
        distances_gal = np.linalg.norm((gal_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
        h_code_gal = distances_gal + clock + isb+gal_tro
        ambiguities_gal = x[self.base_dim + num_gps:]
        h_phase_gal = distances_gal + clock + isb + ambiguities_gal + gal_tro
        return np.concatenate((h_code_gps, h_phase_gps, h_code_gal, h_phase_gal))

    def rebuild_state(self, x_old, P_old, Q_old, prev_sats, curr_sats, base_dim=6):
        """
        Rebuilding of state vector and KF matrices after satellite visibility change
        :param x_old: np.ndarray, old state vector
        :param P_old: np.ndarray, old prior error matrix
        :param Q_old: np.ndarray, old process noise matrix
        :param prev_sats: list, previous satellites
        :param curr_sats: list, current satellites
        :param base_dim: int, dimension of state vector
        :return: new X, new P, Q
        """
        old_dim = base_dim + len(prev_sats)
        new_dim = base_dim + len(curr_sats)

        # 1) Nowy wektor stanu
        new_x = np.zeros(new_dim)
        # Skopiuj parametry bazowe (X, Y, Z, clock, tropo)
        new_x[:base_dim] = x_old[:base_dim]

        # Kopiuj ambiguities dla satelit wspólnych
        prev_map = {sat: i for i, sat in enumerate(prev_sats)}
        curr_map = {sat: i for i, sat in enumerate(curr_sats)}
        common_sats = [sat for sat in curr_sats if sat in prev_map]
        for sat in common_sats:
            old_i = prev_map[sat]
            new_i = curr_map[sat]
            new_x[base_dim + new_i] = x_old[base_dim + old_i]

        # 2) Nowa macierz P
        new_P = np.zeros((new_dim, new_dim))
        new_P[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]

        for satA in common_sats:
            oA = base_dim + prev_map[satA]
            nA = base_dim + curr_map[satA]
            new_P[nA, :base_dim] = P_old[oA, :base_dim]
            new_P[:base_dim, nA] = P_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_map[satB]
                nB = base_dim + curr_map[satB]
                new_P[nA, nB] = P_old[oA, oB]
                new_P[nB, nA] = P_old[oB, oA]

        # Dla nowych satelit – ustaw duże niepewności
        new_sats = set(curr_sats) - set(prev_sats)
        for sat in new_sats:
            i_new = base_dim + curr_map[sat]
            new_P[i_new, i_new] = self.cfg.p_amb

        # 3) Nowa macierz Q
        new_Q = np.zeros((new_dim, new_dim))
        new_Q[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]
        for satA in common_sats:
            oA = base_dim + prev_map[satA]
            nA = base_dim + curr_map[satA]
            new_Q[nA, :base_dim] = Q_old[oA, :base_dim]
            new_Q[:base_dim, nA] = Q_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_map[satB]
                nB = base_dim + curr_map[satB]
                new_Q[nA, nB] = Q_old[oA, oB]
                new_Q[nB, nA] = Q_old[oB, oA]
        # Nowym satelitom daj Q = 0
        for sat in new_sats:
            i_new = base_dim + curr_map[sat]
            new_Q[i_new, i_new] = self.cfg.q_amb

        return new_x, new_P, new_Q

    def init_filter(self):
        """
        Initialization of EKF for PPP estimation

        :return: tuple(list, list), satellites  & observation epochs lists
        """
        gps_epochs = sorted(self.gps_obs.index.get_level_values('time').unique().tolist())
        gal_epochs = sorted(self.gal_obs.index.get_level_values('time').unique().tolist())
        epochs = sorted(list((set(gps_epochs) & set(gal_epochs))))
        gps_obs0 = self.gps_obs.loc[(slice(None), gps_epochs[0]), :]
        gal_obs0 = self.gal_obs.loc[(slice(None), gal_epochs[0]), :]

        gps0 = gps_obs0.index.get_level_values('sv').tolist()
        gal0 = gal_obs0.index.get_level_values('sv').tolist()
        N0 = len(gps0) + len(gal0)

        dimx = self.base_dim + N0
        dimz = 2 * N0
        # x
        if self.pos0 is None:

            xyz0 = np.array([6371.0e3, 0.0, 0.0])
            if self.tro:
                initial_state = np.concatenate((xyz0, [0.0],[0.0],[0.0], np.zeros(N0)))
            else:
                initial_state = np.concatenate((xyz0, [0.0],[0.0], np.zeros(N0)))
        else:
            if self.tro:
                initial_state = np.concatenate((self.pos0, [0.0],[0.0],[0.0], np.zeros(N0)))
            else:
                initial_state = np.concatenate((self.pos0, [0.0],[0.0], np.zeros(N0)))
        self.ekf = ExtendedKalmanFilter(dim_x=dimx, dim_z=dimz)
        self.ekf._I = np.eye(self.ekf.dim_x)

        self.ekf.x = initial_state.copy()
        # Q
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.zeros(N0))))
        self.ekf.Q[:3, :3] = self.cfg.q_crd
        self.ekf.Q[3, 3] = self.cfg.q_dt

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(N0, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.P[4, 4] = self.cfg.p_dt

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0 + gal0

        return old_sats, epochs

    def _prepare_obs(self):

        if self.gps_mode == 'L1':
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C1')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L1')][0]
        if self.gal_mode == 'E1':
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C1')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L1')][0]

        if self.gps_mode == 'L5':
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C5')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L5')][0]
        if self.gal_mode == 'E5':
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C5')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L5')][0]

        gps_data = (gps_c1_col, gps_l1_col,)
        gal_data = (gal_c1_col, gal_l1_col)

        self.gps_obs = self.gps_obs.copy()
        self.gal_obs = self.gal_obs.copy()

        if 'pco_los' not in self.gal_obs.columns.tolist():
            self.gal_obs.loc[:, 'pco_los'] = 0.0

        if 'pco_los' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los'] = 0.0

        return gps_data, gal_data

    def code_screening(self, x, satellites, code_obs, thr=1):
        """
        Screening of code observations for outliers.
        """
        return ppp_code_screening(x=x, satellites=satellites, code_obs=code_obs, thr=thr)

    def phase_residuals_screening(self, sat_list, phase_residuals_dict,num,thr=10,sys='G',len_gps=None):
        """
        Screening of phase residuals for outliers based of prefit differences between epochs
        :param sat_list: list, satellites list
        :param phase_residuals_dict: dict, phase observations prefit residuals
        :param num: int, epoch number
        :param thr: thershold for filtering
        :param sys: str, sys
        :param len_gps: int, number of GPS observations
        :return: Modify state vector and prior matrix (x & P)
        """
        if sys =='E':
            assert len_gps is not None
        reset_svs = []
        prefit_entries = []
        for idx, sv in enumerate(sat_list):
            prev_residual = phase_residuals_dict[num - 1].get(sv)
            current_residual = phase_residuals_dict[num].get(sv)
            if prev_residual is not None and current_residual is not None:
                prefit_entries.append((idx, current_residual - prev_residual))
        if prefit_entries:
            prefit_diff = np.fromiter(
                (entry[1] for entry in prefit_entries),
                dtype=float,
                count=len(prefit_entries),
            )
            median_prefit_diff = np.median(prefit_diff)
        else:
            median_prefit_diff = 0.0

        for idx, residual in prefit_entries:
            if np.abs(residual-median_prefit_diff)>thr:
                outlier_idx = idx
                if sys =='E':
                    outlier_idx += len_gps
                self.ekf.x[self.base_dim + outlier_idx] = 0.0
                self.ekf.P[
                    self.base_dim + outlier_idx, self.base_dim + outlier_idx] = 1e3
                reset_svs.append(sat_list[idx])
        return reset_svs

    def reset_filter(self, epoch):
        """
        reinitialize (reset) Kalman filter. Init filter function but based on given epoch

        """
        # satellites at epoch
        gps_obs0 = self.gps_obs.loc[(slice(None), epoch), :]
        gal_obs0 = self.gal_obs.loc[(slice(None), epoch), :]
        gps0 = gps_obs0.index.get_level_values('sv').tolist()
        gal0 = gal_obs0.index.get_level_values('sv').tolist()
        # number of ambiguities
        N0 = len(gps0) + len(gal0)
        dimx = self.base_dim + N0
        dimz = 2 * N0
        initial_state = np.concatenate((self.pos0, [0.0], [0.0], [0.0], np.zeros(N0)))
        # self.ekf = ExtendedKalmanFilter(dim_x=dimx, dim_z=dimz)
        self.ekf.dim_x=dimx
        self.ekf.dim_z=dimz
        self.ekf._I = np.eye(self.ekf.dim_x)

        self.ekf.x = initial_state.copy()

        # Q
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.zeros(N0))))
        self.ekf.Q[:3, :3] = self.cfg.q_crd
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[5, 5] = self.cfg.q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(N0, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.P[4, 4] = self.cfg.p_dt
        self.ekf.P[5, 5] = self.cfg.p_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0 + gal0

        return old_sats

    def run_filter(self, ref=None, flh=None, add_dcb=True,trace_filter=False,reset_every=0):
        """
        Kalman filter PPP Float estimation
        :param ref: np.ndarray, reference position in ECEF
        :param flh: np.ndarray, reference position in BLH

        :param add_dcb: bool, whether to add dcb
        :param trace_filter:bool, whether to trace filter
        :param reset_every: int, number of epochs to reset filter
        :return: tuple(pd.Dataframe, pd.Dataframe, pd.Dataframe, float), solution, gps&gal residuals, convergence time
        """
        old_sats, epochs = self.init_filter()

        gps_data, gal_data = self._prepare_obs()
        gps_c1, gps_l1 = gps_data
        gal_c1, gal_l1 = gal_data
        self.gps_obs = self.gps_obs.sort_values(by='sv')
        self.gal_obs = self.gal_obs.sort_values(by='sv')
        gps_epochs = {
            t: df for t, df in self.gps_obs.groupby(level=1, sort=False)
        }
        gal_epochs = {
            t: df for t, df in self.gal_obs.groupby(level=1, sort=False)
        }
        r_cache = {}
        phase_residuals_trace = {}
        phase_residuals_trace_gal = {}
        result = []
        result_gal=[]
        result_gps=[]
        xyz=None
        T0 = epochs[0]
        conv_time=None
        gps_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("G")}
        gal_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("E")}
        ar_cfg = PPPARSettings(
            enabled=bool(getattr(self.cfg, "pppar_enabled", False)),
            warmup_epochs=int(getattr(self.cfg, "pppar_warmup_epochs", 60)),
            min_ambiguities=int(getattr(self.cfg, "pppar_min_ambiguities", 4)),
            ratio_threshold=float(getattr(self.cfg, "pppar_ratio_threshold", 2.0)),
            constraint_sigma_cycles=float(getattr(self.cfg, "pppar_constraint_sigma_cycles", 1e-3)),
            constraint_sigma_floor_cycles=getattr(self.cfg, "pppar_constraint_sigma_floor_cycles", 1e-3),
            min_lock_epochs=getattr(self.cfg, "pppar_min_lock_epochs", None),
            min_candidate_elevation_deg=getattr(self.cfg, "pppar_min_candidate_elevation_deg", None),
            use_float_ratio_covariance=bool(getattr(self.cfg, "pppar_use_float_ratio_covariance", True)),
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", False)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
        )
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)
                    gps_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("G")}
                    gal_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("E")}

                    reset_epoch = True
                    T0 = t
            gps_epoch = gps_epochs.get(t)
            gal_epoch = gal_epochs.get(t)
            if gps_epoch is None or gal_epoch is None:
                continue

            if len(gal_epoch) + len(gps_epoch)<4:
                ppp_trace(
                    trace_filter,
                    self.__class__.__name__,
                    "insufficient-satellites",
                    epoch=num,
                    time=t,
                    n_sats=len(gal_epoch) + len(gps_epoch),
                    min_sats=4,
                )
                continue

            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_gal_sats = gal_epoch.index.get_level_values('sv').tolist()
            curr_sats = curr_gps_sats + curr_gal_sats
            if curr_sats != old_sats:
                new_x, new_P, new_Q = self.rebuild_state(self.ekf.x.copy(), self.ekf.P.copy(), self.ekf.Q.copy(),
                                                         old_sats, curr_sats,
                                                         self.base_dim)
                self.ekf.x, self.ekf.P, self.ekf.Q = new_x, new_P, new_Q
                self.ekf.F = np.eye(len(self.ekf.x))
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = 2 * len(curr_sats)
                self.ekf._I = np.eye(self.ekf.dim_x)
            old_sats = curr_sats
            gps_arc_age = {sv: gps_arc_age.get(sv, 0) for sv in curr_gps_sats}
            gal_arc_age = {sv: gal_arc_age.get(sv, 0) for sv in curr_gal_sats}
            gps_clk, gps_tro, gps_ah_los, gps_sat_pco_los, gps_dprel, gps_pco1, gps_ion, gps_phw = [
                gps_epoch[col].to_numpy()
                for col in
                ['clk', 'tro', 'ah_los',
                 'sat_pco_los',
                 'dprel', 'pco_los', 'ion', 'phw']]
            gal_clk, gal_tro, gal_ah_los, gal_sat_pco_los, gal_dprel, gal_pco1, gal_ion, gal_phw = [
                gal_epoch[col].to_numpy()
                for col in
                ['clk', 'tro', 'ah_los',
                 'sat_pco_los',
                 'dprel', 'pco_los', 'ion', 'phw']]


            gps_p1_c1 = np.asarray(gps_epoch.get(f'OSB_{gps_c1}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl1_l1 = np.asarray(gps_epoch.get(f'OSB_{gps_l1}',0.0)) * 1e-09 * self.CLIGHT

            gal_p1_c1 = np.asarray(gal_epoch.get(f'OSB_{gal_c1}')) * 1e-09 * self.CLIGHT
            gal_pl1_l1 = np.asarray(gal_epoch.get(f'OSB_{gal_l1}')) * 1e-09 * self.CLIGHT

            los_gps = gps_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            gps_tides = gps_epoch[['dx', 'dy', 'dz']].to_numpy()
            gps_tides_los = np.sum(los_gps * gps_tides, axis=1)

            los_gal = gal_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            gal_tides = gal_epoch[['dx', 'dy', 'dz']].to_numpy()
            gal_tides_los = np.sum(los_gal * gal_tides, axis=1)

            GP3_code = (
                    (gps_epoch[f'{gps_c1}'].to_numpy() - gps_pco1 -gps_p1_c1)
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel
                    - gps_ion - gps_tides_los
            )

            GL3_phase = (
                    (gps_epoch[f'{gps_l1}'].to_numpy() - gps_pco1-gps_pl1_l1)
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel
                    + gps_ion -  ((self.CLIGHT / self.FREQ_DICT[self.gps_mode]) / (2 * np.pi)) * gps_phw - gps_tides_los
            )

            EP3_code = (
                    (gal_epoch[f'{gal_c1}'].to_numpy() - gal_pco1-gal_p1_c1)
                    + gal_clk * self.CLIGHT - gal_tro - gal_ah_los + gal_sat_pco_los - gal_dprel
                    - gal_ion - gal_tides_los
            )

            EL3_phase = (
                    (gal_epoch[f'{gal_l1}'].to_numpy() - gal_pco1-gal_pl1_l1)
                    + gal_clk * self.CLIGHT - gal_tro - gal_ah_los + gal_sat_pco_los - gal_dprel
                    + gal_ion -  ((self.CLIGHT / self.FREQ_DICT[self.gal_mode]) / (2 * np.pi)) * gal_phw - gal_tides_los
            )

            z = np.concatenate((GP3_code, GL3_phase, EP3_code, EL3_phase))
            self.ekf.dim_z = len(z)
            sat_positions_gps = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            sat_positions_gal = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            dist_gps = calculate_distance(sat_positions_gps[:, :3].copy(), self.ekf.x[:3].copy())
            dist_gal = calculate_distance(sat_positions_gal[:, :3].copy(), self.ekf.x[:3].copy())
            prefit_gps_phase = GL3_phase - dist_gps
            prefit_gal_phase = EL3_phase - dist_gal
            phase_residuals_trace[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_phase)}
            phase_residuals_trace_gal[num] = {sv: r for sv, r in zip(curr_gal_sats, prefit_gal_phase)}
            gps_phase_reset_svs: set[str] = set()
            gal_phase_reset_svs: set[str] = set()
            if num >= 60:
                gps_phase_reset_svs.update(self.phase_residuals_screening(sat_list=curr_gps_sats,phase_residuals_dict=phase_residuals_trace,num=num))
                gal_phase_reset_svs.update(self.phase_residuals_screening(sat_list=curr_gal_sats, phase_residuals_dict=phase_residuals_trace_gal,
                                           num=num, sys='E', len_gps=len(curr_gps_sats)))

            mask_gps = self.code_screening(x=self.ekf.x[:3].copy(),satellites=sat_positions_gps,code_obs=GP3_code,thr=10)
            mask_gps2 = np.concatenate((mask_gps, mask_gps))
            mask_gal = self.code_screening(x=self.ekf.x[:3].copy(), satellites=sat_positions_gal, code_obs=EP3_code,
                                           thr=10)
            mask_gal2 = np.concatenate((mask_gal, mask_gal))
            mask2 = np.concatenate((mask_gps2, mask_gal2))

            ev_gps = gps_epoch['ev'].to_numpy(copy=False)
            ev_gal = gal_epoch['ev'].to_numpy(copy=False)
            sin_el_gps = np.sin(np.deg2rad(ev_gps))
            sin_el_gal = np.sin(np.deg2rad(ev_gal))
            inv_sin2_gps = 1.0 / np.square(sin_el_gps)
            inv_sin2_gal = 1.0 / np.square(sin_el_gal)
            sigma_code_gps = 1 + 0.0025 * inv_sin2_gps
            sigma_phase_gps = 1e-4 + 0.00025 * inv_sin2_gps
            sigma_code_gal = 1 + 0.0025 * inv_sin2_gal
            sigma_phase_gal = 1e-4 + 0.00025 * inv_sin2_gal
            R_vec = np.concatenate((sigma_code_gps, sigma_phase_gps, sigma_code_gal, sigma_phase_gal))
            R_vec[~mask2] = 1e12

            m = R_vec.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = R_vec
            self.ekf.R = R

            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=(sat_positions_gps, sat_positions_gal),
                                    Hx=self.Hx, hx_args=(sat_positions_gps, sat_positions_gal))
            gps_arc_age = advance_arc_age(gps_arc_age, curr_gps_sats, gps_phase_reset_svs)
            gal_arc_age = advance_arc_age(gal_arc_age, curr_gal_sats, gal_phase_reset_svs)
            gps_ar_age = arc_age_array(gps_arc_age, curr_gps_sats)
            gal_ar_age = arc_age_array(gal_arc_age, curr_gal_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_conventional_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(curr_gps_sats),
                    n_gal=len(curr_gal_sats),
                    gps_ev=ev_gps,
                    gal_ev=ev_gal,
                    lambda_gps=self.CLIGHT / self.FREQ_DICT[self.gps_mode],
                    lambda_gal=self.CLIGHT / self.FREQ_DICT[self.gal_mode],
                    settings=ar_cfg,
                    gps_age=gps_ar_age,
                    gal_age=gal_ar_age,
                )

            dtr = self.ekf.x[3]
            isb = self.ekf.x[4]
            if self.tro:
                tro = self.ekf.x[5]
            else:
                tro = 0.0
            if ref is not None and flh is not None:

                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx,flh=flh,degrees=True)).flatten()
            else:
                enu = np.array([0.0, 0.0, 0.0])
            gps_epoch['dtr'] = dtr
            gps_epoch['ztd'] = tro
            gps_epoch['L3'] = GL3_phase
            gps_epoch['P3'] = GP3_code
            gps_epoch['N1'] = self.ekf.x[self.base_dim:self.base_dim + len(gps_epoch)].copy()
            gps_epoch['v'] = self.ekf.y[len(gps_epoch):2 * len(gps_epoch)]
            gps_epoch['p_n'] = np.diag(self.ekf.P[self.base_dim:self.base_dim + len(gps_epoch),
                                        self.base_dim:self.base_dim + len(gps_epoch)])

            gal_epoch['dtr'] = dtr
            gal_epoch['isb'] = isb
            gal_epoch['L3'] = EL3_phase
            gal_epoch['P3'] = EP3_code
            gal_epoch['N1'] = self.ekf.x[self.base_dim + len(gps_epoch):].copy()
            gal_epoch['p_n'] = np.diag(self.ekf.P[self.base_dim + len(gps_epoch):,
                                        self.base_dim + len(gps_epoch):])
            gal_epoch['v'] = self.ekf.y[2 * len(gps_epoch) + len(gal_epoch):]

            result_gps.append(gps_epoch)
            result_gal.append(gal_epoch)
            # Convergence checking logic
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time
            xyz = self.ekf.x[:3].copy()
            df_epoch = pd.DataFrame(
                {'de': [enu[0]], 'dn': [enu[1]], 'du': [enu[2]], 'dtr': [dtr], 'isb': [isb], 'ztd': [tro],
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]],
                 'ar_fixed': [0 if ar_diag is None else ar_diag.fixed_ambiguities],
                 'ar_ratio': [np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min],
                 'ar_ok': [False if ar_diag is None else ar_diag.accepted],
                 'ar_gps_min_age': [int(np.min(gps_ar_age)) if gps_ar_age.size else np.nan],
                 'ar_gal_min_age': [int(np.min(gal_ar_age)) if gal_ar_age.size else np.nan]},
                index=pd.DatetimeIndex([t], name='time'))
            for key, value in pppar_diagnostic_columns(ar_diag).items():
                df_epoch[key] = value
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=df_epoch.iloc[0],
                )
            result.append(df_epoch)
        df_result = pd.concat(result)
        df_gps = pd.concat(result_gps)
        df_gal = pd.concat(result_gal)
        if conv_time:
            ct = conv_time.total_seconds() / 3600
            if trace_filter:
                ppp_trace(
                    trace_filter,
                    self.__class__.__name__,
                    "convergence",
                    threshold_m=0.005,
                    minutes=conv_time.total_seconds() / 60,
                    hours=conv_time.total_seconds() / 3600,
                )
        else:
            ct=None
        df_result['ct_min'] = ct
        return df_result, df_gps, df_gal, ct


class _LegacyPPPDualFreqMultiGNSS:
    """Legacy GPS/Galileo dual-frequency combined PPP implementation.

    Purpose:
        Historical G/E ionosphere-free mixed PPP class kept as a reference for
        older behavior. It is not the preferred public mixed combined class.

    Status:
        Legacy/reference class. It is not routed by the current PPP session and
        the leading underscore keeps it out of normal star exports, but it is
        kept in-module as a comparison point for the active generic combined
        implementation.

    Model:
        Forms ionosphere-free observables for GPS and Galileo, estimates one
        GPS receiver clock, one Galileo ISB, optional ZTD, and one IF ambiguity
        per satellite.

    State vector:
        ``[x, y, z, dtr_G, isb_E, ztd?, N_IF_G..., N_IF_E...]``.

    PPP-AR support:
        Contains conventional IF AR hooks, but combined IF AR remains
        experimental and configuration-gated.

    Warnings:
        Treat as legacy/reference code. Do not consolidate or remove without
        direct numerical comparison against historical G/E runs.
    """

    def __init__(self,config:PPPConfig, gps_obs, gal_obs, gps_mode, gal_mode, ekf, tro=True, pos0=None,interval=0.5):

        """
        PPP Dual Frequency, Multi GNSS estimator
        :param gps_obs: pd.DataFrame, gps observations
        :param gal_obs: pd.DataFrame, gal observations
        :param gps_mode: str, gps signals
        :param gal_mode: str, gal signals
        :param ekf: filterpy.ExtendedKalmanFilter instance
        :param tro: bool, estimate ZTD
        :param pos0: np.ndarray, initial position
        :param interval: float, interval of observations
        """
        self.cfg=config
        self.gps_obs = gps_obs
        self.gal_obs = gal_obs
        self.gps_mode = gps_mode
        self.gal_mode = gal_mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = {'L1': 1575.42e06, 'E1': 1575.42e06,
                          'L2': 1227.60e06, 'E5a': 1176.45e06,
                          'L5': 1176.450e06, 'E5b': 1207.14e06}
        self.LAMBDA_DICT={}
        self.CLIGHT = 299792458
        self.base_dim = 6 if self.tro else 5
        self.pos0 = pos0
        self.interval=interval

    def HJacobian(self, x, gps_satellites, gal_satellites):
        """
        Hjacobian matrix
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :param gal_satellites: np.ndarray, gal satellite coordinates & mapping function
        :return: Hjacobian matrix
        """
        num_gps = len(gps_satellites)
        num_gal = len(gal_satellites)
        N = num_gps + num_gal
        # State: x,y,z, dt_gps, tro, isb, N1...Nn
        dim_state = self.base_dim + N

        receiver = x[:3]
        # GPS PART
        H1 = np.zeros((2 * num_gps, dim_state))

        distances_gps = np.linalg.norm((gps_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)

        for i in range(num_gps):
            # Code observation (wiersz i)
            H1[i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[i, 3] = 1.0  # clock
            H1[i, 5] = gps_satellites[i, -1]  # trop

            # Phase observation (wiersz N+i)
            H1[num_gps + i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[num_gps + i, 3] = 1.0  # clock
            H1[num_gps + i, 5] = gps_satellites[i, -1]  # trop
            H1[num_gps + i, self.base_dim + i] = 1  # AMB
        # GAL PART
        H2 = np.zeros((2 * num_gal, dim_state))
        distances_gal = np.linalg.norm((gal_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
        for i in range(num_gal):
            # Code observation (wiersz i)
            H2[i, 0:3] = (receiver - gal_satellites[i, :3]) / distances_gal[i]
            H2[i, 3] = 1.0  # clock
            H2[i, 4] = 1.0  # isb
            H2[i, 5] = gal_satellites[i, -1]  # trop

            # Phase observation (wiersz N+i)
            H2[num_gal + i, 0:3] = (receiver - gal_satellites[i, :3]) / distances_gal[i]
            H2[num_gal + i, 3] = 1.0  # clock
            H2[num_gal + i, 4] = 1.0  # isb
            H2[num_gal + i, 5] = gal_satellites[i, -1]  # trop
            H2[num_gal + i, (self.base_dim + num_gps) + i] = 1  # AMB
        H = np.vstack((H1, H2))
        return H

    def Hx(self, x, gps_satellites, gal_satellites):
        """
        Predicted observations vector
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :param gal_satellites: np.ndarray, gal satellite coordinates & mapping function
        :return: predicted obs matrix
        """
        num_gps = len(gps_satellites)
        num_gal = len(gal_satellites)
        receiver = x[:3]
        clock = x[3]
        tropo = x[5]
        isb = x[4]
        # GPS prediction
        distances_gps = np.linalg.norm((gps_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)

        h_code_gps = distances_gps + clock + gps_satellites[:, -1] * tropo
        ambiguities_gps = x[self.base_dim:self.base_dim + num_gps]
        h_phase_gps = distances_gps + clock + ambiguities_gps + gps_satellites[:, -1] * tropo
        # GAL prediction
        distances_gal = np.linalg.norm((gal_satellites[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
        h_code_gal = distances_gal + clock + isb + gal_satellites[:, -1] * tropo
        ambiguities_gal = x[self.base_dim + num_gps:]
        h_phase_gal = distances_gal + clock + isb + ambiguities_gal + gal_satellites[:, -1] * tropo
        return np.concatenate((h_code_gps, h_phase_gps, h_code_gal, h_phase_gal))

    def rebuild_state(self, x_old, P_old, Q_old, prev_sats, curr_sats, base_dim=6):
        """
        Rebuilding of state vector and KF matrices after satellite visibility change
        :param x_old: np.ndarray, old state vector
        :param P_old: np.ndarray, old prior error matrix
        :param Q_old: np.ndarray, old process noise matrix
        :param prev_sats: list, previous satellites
        :param curr_sats: list, current satellites
        :param base_dim: int, dimension of state vector
        :return: new X, new P, Q
        """
        old_dim = base_dim + len(prev_sats)
        new_dim = base_dim + len(curr_sats)

        # 1) Nowy wektor stanu
        new_x = np.zeros(new_dim)
        # Skopiuj parametry bazowe (X, Y, Z, clock, tropo)
        new_x[:base_dim] = x_old[:base_dim]

        # Kopiuj ambiguities dla satelit wspólnych
        prev_map = {sat: i for i, sat in enumerate(prev_sats)}
        curr_map = {sat: i for i, sat in enumerate(curr_sats)}
        common_sats = [sat for sat in curr_sats if sat in prev_map]
        for sat in common_sats:
            old_i = prev_map[sat]
            new_i = curr_map[sat]
            new_x[base_dim + new_i] = x_old[base_dim + old_i]

        # 2) Nowa macierz P
        new_P = np.zeros((new_dim, new_dim))
        new_P[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]

        for satA in common_sats:
            oA = base_dim + prev_map[satA]
            nA = base_dim + curr_map[satA]
            new_P[nA, :base_dim] = P_old[oA, :base_dim]
            new_P[:base_dim, nA] = P_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_map[satB]
                nB = base_dim + curr_map[satB]
                new_P[nA, nB] = P_old[oA, oB]
                new_P[nB, nA] = P_old[oB, oA]

        # Dla nowych satelit – ustaw duże niepewności
        new_sats = set(curr_sats) - set(prev_sats)
        for sat in new_sats:
            i_new = base_dim + curr_map[sat]
            new_P[i_new, i_new] = self.cfg.p_amb

        # 3) Nowa macierz Q
        new_Q = np.zeros((new_dim, new_dim))
        new_Q[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]
        for satA in common_sats:
            oA = base_dim + prev_map[satA]
            nA = base_dim + curr_map[satA]
            new_Q[nA, :base_dim] = Q_old[oA, :base_dim]
            new_Q[:base_dim, nA] = Q_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_map[satB]
                nB = base_dim + curr_map[satB]
                new_Q[nA, nB] = Q_old[oA, oB]
                new_Q[nB, nA] = Q_old[oB, oA]
        # Nowym satelitom daj Q = 0
        for sat in new_sats:
            i_new = base_dim + curr_map[sat]
            new_Q[i_new, i_new] = self.cfg.q_amb

        return new_x, new_P, new_Q

    def init_filter(self):
        """
        Initialization of EKF for PPP estimation
        :return: tuple(list, list), satellites  & observation epochs lists
        """
        gps_epochs = sorted(self.gps_obs.index.get_level_values('time').unique().tolist())
        gal_epochs = sorted(self.gal_obs.index.get_level_values('time').unique().tolist())
        epochs = sorted(list((set(gps_epochs) & set(gal_epochs))))
        gps_obs0 = self.gps_obs.loc[(slice(None), gps_epochs[0]), :]
        gal_obs0 = self.gal_obs.loc[(slice(None), gal_epochs[0]), :]

        gps0 = gps_obs0.index.get_level_values('sv').tolist()
        gal0 = gal_obs0.index.get_level_values('sv').tolist()
        N0 = len(gps0) + len(gal0)

        dimx = self.base_dim + N0
        dimz = 2 * N0
        # x
        if self.pos0 is None:

            xyz0 = np.array([6371.0e3, 0.0, 0.0])
            initial_state = np.concatenate((xyz0, [0.0], [0.0], [0.0], np.zeros(N0)))
        else:
            initial_state = np.concatenate((self.pos0, [0.0], [0.0], [0.0], np.zeros(N0)))
        self.ekf = ExtendedKalmanFilter(dim_x=dimx, dim_z=dimz)
        self.ekf._I = np.eye(self.ekf.dim_x)

        self.ekf.x = initial_state.copy()

        # Q
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.zeros(N0))))
        self.ekf.Q[:3, :3] = self.cfg.q_crd
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[5, 5] = self.cfg.q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(N0, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.P[4, 4] = self.cfg.p_dt
        self.ekf.P[5, 5] = self.cfg.p_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0 + gal0

        return old_sats, epochs

    def _prepare_obs(self):
        if self.gps_mode == 'L1L2':
            GF1 = self.FREQ_DICT['L1']
            GF2 = self.FREQ_DICT['L2']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.gps_obs.columns if c.startswith('C2')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.gps_obs.columns if c.startswith('L2')][0]

        if self.gps_mode == 'L1L5':
            GF1 = self.FREQ_DICT['L1']
            GF2 = self.FREQ_DICT['L5']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.gps_obs.columns if c.startswith('C5')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.gps_obs.columns if c.startswith('L5')][0]

        if self.gal_mode == 'E1E5a':
            EF1 = self.FREQ_DICT['E1']
            EF2 = self.FREQ_DICT['E5a']
            agal, bgal = EF1 ** 2 / (EF1 ** 2 - EF2 ** 2), EF2 ** 2 / (EF1 ** 2 - EF2 ** 2)
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C1')][0]
            gal_c2_col = [c for c in self.gal_obs.columns if c.startswith('C5')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L1')][0]
            gal_l2_col = [c for c in self.gal_obs.columns if c.startswith('L5')][0]
        elif self.gal_mode == 'E1E5b':
            EF1 = self.FREQ_DICT['E1']
            EF2 = self.FREQ_DICT['E5b']
            agal, bgal = EF1 ** 2 / (EF1 ** 2 - EF2 ** 2), EF2 ** 2 / (EF1 ** 2 - EF2 ** 2)
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C1')][0]
            gal_c2_col = [c for c in self.gal_obs.columns if c.startswith('C7')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L1')][0]
            gal_l2_col = [c for c in self.gal_obs.columns if c.startswith('L7')][0]

        self.LAMBDA_DICT[self.gps_mode] = agps * (self.CLIGHT/GF1) - bgps * (self.CLIGHT/GF2)
        self.LAMBDA_DICT[self.gal_mode] = agal * (self.CLIGHT/EF1 )- bgal * (self.CLIGHT/EF2)

        gps_data = (agps, bgps, gps_c1_col, gps_c2_col, gps_l1_col, gps_l2_col)
        gal_data = (agal, bgal, gal_c1_col, gal_c2_col, gal_l1_col, gal_l2_col)

        self.gps_obs = self.gps_obs.copy()
        self.gal_obs = self.gal_obs.copy()

        if 'pco_los_l1' not in self.gal_obs.columns.tolist():
            self.gal_obs.loc[:, 'pco_los_l1'] = 0.0
        if 'pco_los_l5' not in self.gal_obs.columns.tolist():
            self.gal_obs.loc[:, 'pco_los_l5'] = 0.0

        if 'pco_los_l1' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los_l1'] = 0.0
        if 'pco_los_l2' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los_l2'] = 0.0

        return gps_data, gal_data

    def code_screening(self, x, satellites, code_obs, thr=1):
        """
        Screening of code observations for outliers.
        """
        return ppp_code_screening(x=x, satellites=satellites, code_obs=code_obs, thr=thr)

    def phase_residuals_screening(self, sat_list, phase_residuals_dict,num,thr=1,sys='G',len_gps=None):
        """
        Screening of phase residuals for outliers based of prefit differences between epochs
        :param sat_list: list, satellites list
        :param phase_residuals_dict: dict, phase observations prefit residuals
        :param num: int, epoch number
        :param thr: thershold for filtering
        :param sys: str, sys
        :param len_gps: int, number of GPS observations
        :return: Modify state vector and prior matrix (x & P)
        """
        if sys =='E':
            assert len_gps is not None
        reset_svs = []
        prefit_entries = []
        for idx, sv in enumerate(sat_list):
            prev_residual = phase_residuals_dict[num - 1].get(sv)
            current_residual = phase_residuals_dict[num].get(sv)
            if prev_residual is not None and current_residual is not None:
                prefit_entries.append((idx, current_residual - prev_residual))
        if prefit_entries:
            prefit_diff = np.fromiter(
                (entry[1] for entry in prefit_entries),
                dtype=float,
                count=len(prefit_entries),
            )
            median_prefit_diff = np.median(prefit_diff)
        else:
            median_prefit_diff = 0.0

        for idx, residual in prefit_entries:
            if np.abs(residual-median_prefit_diff)>thr:
                outlier_idx = idx
                if sys =='E':
                    outlier_idx += len_gps
                self.ekf.x[self.base_dim + outlier_idx] = 0.0
                self.ekf.P[
                    self.base_dim + outlier_idx, self.base_dim + outlier_idx] = 400
                reset_svs.append(sat_list[idx])
        return reset_svs

    def reset_filter(self, epoch):
        """
        reinitialize (reset) Kalman filter. Init filter function but based on given epoch
        """
        # satellites at epoch
        gps_obs0 = self.gps_obs.loc[(slice(None), epoch), :]
        gal_obs0 = self.gal_obs.loc[(slice(None), epoch), :]
        gps0 = gps_obs0.index.get_level_values('sv').tolist()
        gal0 = gal_obs0.index.get_level_values('sv').tolist()
        # number of ambiguities
        N0 = len(gps0) + len(gal0)
        dimx = self.base_dim + N0
        dimz = 2 * N0
        initial_state = np.concatenate((self.pos0, [0.0], [0.0], [0.0], np.zeros(N0)))
        # self.ekf = ExtendedKalmanFilter(dim_x=dimx, dim_z=dimz)
        self.ekf.dim_x=dimx
        self.ekf.dim_z=dimz
        self.ekf._I = np.eye(self.ekf.dim_x)

        self.ekf.x = initial_state.copy()

        # Q
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.zeros(N0))))
        self.ekf.Q[:3, :3] = self.cfg.q_crd
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[5, 5] = self.cfg.q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(N0, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.P[4, 4] = self.cfg.p_dt
        self.ekf.P[5, 5] = self.cfg.p_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0 + gal0

        return old_sats




    def run_filter(self, ref=None, flh=None,add_dcb=True,trace_filter=False, reset_every = 180):
        """
        Kalman filter PPP Float estimation
        :param ref: np.ndarray, reference position in ECEF
        :param flh: np.ndarray, reference position in BLH
        :param add_dcb: bool, whether to add dcb
        :param trace_filter:bool, whether to trace filter
        :param reset_every: int, number of epochs to reset filter
        :return: tuple(pd.Dataframe, pd.Dataframe, pd.Dataframe, float), solution, gps&gal residuals, convergence time
        """
        old_sats, epochs = self.init_filter(

        )

        gps_data, gal_data = self._prepare_obs()
        agps, bgps, gps_c1, gps_c2, gps_l1, gps_l2 = gps_data
        agal, bgal, gal_c1, gal_c2, gal_l1, gal_l2 = gal_data
        self.gps_obs = self.gps_obs.sort_values(by='sv')
        self.gal_obs = self.gal_obs.sort_values(by='sv')
        gps_epochs = {
            t: df for t, df in self.gps_obs.groupby(level=1, sort=False)
        }
        gal_epochs = {
            t: df for t, df in self.gal_obs.groupby(level=1, sort=False)
        }
        r_cache = {}
        phase_residuals_trace = {}
        phase_residuals_trace_gal = {}
        result = []
        result_gps = []
        result_gal =[]
        ar_cfg = PPPARSettings(
            enabled=combined_if_pppar_enabled(self.cfg),
            warmup_epochs=int(getattr(self.cfg, "pppar_warmup_epochs", 60)),
            min_ambiguities=int(getattr(self.cfg, "pppar_min_ambiguities", 4)),
            ratio_threshold=float(getattr(self.cfg, "pppar_ratio_threshold", 2.0)),
            constraint_sigma_cycles=float(getattr(self.cfg, "pppar_constraint_sigma_cycles", 1e-3)),
            constraint_sigma_floor_cycles=getattr(self.cfg, "pppar_constraint_sigma_floor_cycles", 1e-3),
            min_lock_epochs=getattr(self.cfg, "pppar_min_lock_epochs", None),
            min_candidate_elevation_deg=getattr(self.cfg, "pppar_min_candidate_elevation_deg", None),
            use_float_ratio_covariance=bool(getattr(self.cfg, "pppar_use_float_ratio_covariance", True)),
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", False)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
        )
        ar_disabled_reason = (
            "combined_if_ar_disabled"
            if bool(getattr(self.cfg, "pppar_enabled", False)) and not ar_cfg.enabled
            else None
        )
        xyz = None
        conv_time = None
        T0 = epochs[0]
        gps_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("G")}
        gal_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("E")}
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num !=0):
                    old_sats = self.reset_filter(epoch=t)
                    gps_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("G")}
                    gal_arc_age = {sv: 0 for sv in old_sats if str(sv).startswith("E")}
                    reset_epoch =True
                    T0 = t

            gps_epoch = gps_epochs.get(t)
            gal_epoch = gal_epochs.get(t)
            if gps_epoch is None or gal_epoch is None:
                continue

            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_gal_sats = gal_epoch.index.get_level_values('sv').tolist()
            curr_sats = curr_gps_sats + curr_gal_sats
            if curr_sats != old_sats:
                new_x, new_P, new_Q = self.rebuild_state(self.ekf.x.copy(), self.ekf.P.copy(), self.ekf.Q.copy(),
                                                         old_sats, curr_sats,
                                                         self.base_dim)
                self.ekf.x, self.ekf.P, self.ekf.Q = new_x, new_P, new_Q
                self.ekf.F = np.eye(len(self.ekf.x))
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = 2 * len(curr_sats)
                self.ekf._I = np.eye(self.ekf.dim_x)
            old_sats = curr_sats
            gps_arc_age = {sv: gps_arc_age.get(sv, 0) for sv in curr_gps_sats}
            gal_arc_age = {sv: gal_arc_age.get(sv, 0) for sv in curr_gal_sats}
            gps_sta_pco1, gps_sta_pco2 = self.gps_mode[:2], self.gps_mode[2:]
            gal_sta_pco1, gal_sta_pco2 = self.gal_mode[:2], self.gal_mode[2:]
            gps_clk, gps_tro, gps_ah_los,  gps_dprel, gps_pco1, gps_pco2, sat_pco_L1, sat_pco_L2,gps_tides_los = [np.asarray(gps_epoch.get(col,0.0))
                                                                                            for col in
                                                                                            ['clk', 'tro', 'ah_los',
                                                                                             'dprel', 'pco_los_l1',
                                                                                             'pco_los_l2',f'sat_pco_los_{gps_sta_pco1}',
                                                                                             f'sat_pco_los_{gps_sta_pco2}','tides_los']]

            gal_clk, gal_tro, gal_ah_los,  gal_dprel, gal_pco1, gal_pco2,sat_pco_E1, sat_pco_E5a, gal_tides_los  = [np.asarray(gal_epoch.get(col,0.0))
                                                                                            for col in
                                                                                            ['clk', 'tro', 'ah_los',
                                                                                             'dprel', 'pco_los_l1',
                                                                                             'pco_los_l2',f'sat_pco_los_{gal_sta_pco1}',
                                                                                        f'sat_pco_los_{gal_sta_pco2}','tides_los']]

            # los_gps = gps_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            # gps_tides = gps_epoch[['dx', 'dy', 'dz']].to_numpy()
            # gps_tides_los = np.sum(los_gps * gps_tides, axis=1)
            # gps_epoch['tides_los'] = gps_tides_los


            # los_gal = gal_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            # gal_tides = gal_epoch[['dx', 'dy', 'dz']].to_numpy()
            # gal_tides_los = np.sum(los_gal * gal_tides, axis=1)
            # gal_epoch['tides_los'] = gal_tides_los

            # gal_tides_los += np.sum(los_gal * dR_gal, axis=1)
            gps_p1_c1 = np.asarray(gps_epoch.get(f'OSB_{gps_c1}',0.0)) * 1e-09 * self.CLIGHT
            gps_p2_c2 = np.asarray(gps_epoch.get(f'OSB_{gps_c2}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl1_l1 = np.asarray(gps_epoch.get(f'OSB_{gps_l1}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl2_l2 = np.asarray(gps_epoch.get(f'OSB_{gps_l2}',0.0)) * 1e-09 * self.CLIGHT

            gal_p1_c1 = np.asarray(gal_epoch.get(f'OSB_{gal_c1}',0.0)) * 1e-09 * self.CLIGHT
            gal_p2_c2 = np.asarray(gal_epoch.get(f'OSB_{gal_c2}',0.0)) * 1e-09 * self.CLIGHT
            gal_pl1_l1 = np.asarray(gal_epoch.get(f'OSB_{gal_l1}',0.0)) * 1e-09 * self.CLIGHT
            gal_pl2_l2 = np.asarray(gal_epoch.get(f'OSB_{gal_l2}',0.0)) * 1e-09 * self.CLIGHT



            GP3_code = (
                    (agps * (gps_epoch[f'{gps_c1}'].to_numpy() - gps_pco1 - gps_p1_c1+sat_pco_L1) -
                     bgps * (gps_epoch[f'{gps_c2}'].to_numpy() - gps_pco2 - gps_p2_c2+sat_pco_L2))
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los  - gps_dprel  - gps_tides_los
            )

            GL3_phase = (
                    (agps * (gps_epoch[f'{gps_l1}'].to_numpy() - gps_pco1 - gps_pl1_l1+sat_pco_L1) -
                     bgps * (gps_epoch[f'{gps_l2}'].to_numpy() - gps_pco2 - gps_pl2_l2+sat_pco_L2))
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel  - gps_tides_los
                    - (self.LAMBDA_DICT[self.gps_mode] / (2 * np.pi)) * gps_epoch['phw'].to_numpy()
            )

            EP3_code = (
                    (agal * (gal_epoch[f'{gal_c1}'].to_numpy() - gal_pco1 - gal_p1_c1+sat_pco_E1) -
                     bgal * (gal_epoch[f'{gal_c2}'].to_numpy() - gal_pco2 - gal_p2_c2+sat_pco_E5a))
                    + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_tides_los
            )

            EL3_phase = (
                    (agal * (gal_epoch[f'{gal_l1}'].to_numpy() - gal_pco1 - gal_pl1_l1+sat_pco_E1) -
                     bgal * (gal_epoch[f'{gal_l2}'].to_numpy() - gal_pco2 - gal_pl2_l2+sat_pco_E5a))
                    + gal_clk * self.CLIGHT - gal_tro - gal_ah_los  - gal_dprel  - gal_tides_los
                    - (self.LAMBDA_DICT[self.gal_mode]/ (2 * np.pi)) * gal_epoch['phw'].to_numpy()
            )

            z = np.concatenate((GP3_code, GL3_phase, EP3_code, EL3_phase))
            self.ekf.dim_z = len(z)

            sat_positions_gps = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            sat_positions_gal = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            dist_gps = calculate_distance(sat_positions_gps[:, :3].copy(), self.ekf.x[:3].copy())
            dist_gal = calculate_distance(sat_positions_gal[:, :3].copy(), self.ekf.x[:3].copy())
            prefit_gps_phase = GL3_phase - dist_gps
            prefit_gal_phase = EL3_phase - dist_gal

            phase_residuals_trace[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_phase)}
            phase_residuals_trace_gal[num] = {sv: r for sv, r in zip(curr_gal_sats, prefit_gal_phase)}

            # OBSERVATION SCREENING
            mask_gps = self.code_screening(x=self.ekf.x[:3].copy(),satellites = sat_positions_gps[:,:3],code_obs=GP3_code)
            gps_phase_reset_svs: set[str] = set()
            gal_phase_reset_svs: set[str] = set()
            if num >= 60:
                gps_phase_reset_svs.update(self.phase_residuals_screening(sat_list=curr_gps_sats,phase_residuals_dict=phase_residuals_trace,num=num))
            mask_gps2 = np.concatenate((mask_gps, mask_gps))

            mask_gal = self.code_screening(x=self.ekf.x[:3].copy(),satellites=sat_positions_gal[:,:3],code_obs=EP3_code)
            if num >=60:
                gal_phase_reset_svs.update(self.phase_residuals_screening(sat_list=curr_gal_sats,phase_residuals_dict=phase_residuals_trace_gal,num=num,sys='E',len_gps=len(curr_gps_sats)))
            mask_gal2 = np.concatenate((mask_gal, mask_gal))

            # # ostateczna maska na wszystkie pomiary
            mask2 = np.concatenate((mask_gps2, mask_gal2))

            ev_gps = gps_epoch['ev'].to_numpy(copy=False)
            ev_gal = gal_epoch['ev'].to_numpy(copy=False)
            sin_el_gps = np.sin(np.deg2rad(ev_gps))
            sin_el_gal = np.sin(np.deg2rad(ev_gal))
            inv_sin2_gps = 1.0 / np.square(sin_el_gps)
            inv_sin2_gal = 1.0 / np.square(sin_el_gal)
            sigma_code_gps = 1 + 0.0025 * inv_sin2_gps
            sigma_phase_gps = 1e-4 + 0.0003 * inv_sin2_gps
            sigma_code_gal = 1 + 0.0025 * inv_sin2_gal
            sigma_phase_gal = 1e-4 + 0.0003 * inv_sin2_gal
            r_vec = np.concatenate((sigma_code_gps, sigma_phase_gps, sigma_code_gal, sigma_phase_gal))
            r_vec[~mask2] = 1e12

            m = r_vec.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = r_vec
            self.ekf.R = R
            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=(sat_positions_gps, sat_positions_gal),
                                    Hx=self.Hx, hx_args=(sat_positions_gps, sat_positions_gal))
            gps_arc_age = advance_arc_age(gps_arc_age, curr_gps_sats, gps_phase_reset_svs)
            gal_arc_age = advance_arc_age(gal_arc_age, curr_gal_sats, gal_phase_reset_svs)
            gps_ar_age = arc_age_array(gps_arc_age, curr_gps_sats)
            gal_ar_age = arc_age_array(gal_arc_age, curr_gal_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_conventional_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(curr_gps_sats),
                    n_gal=len(curr_gal_sats),
                    gps_ev=ev_gps,
                    gal_ev=ev_gal,
                    lambda_gps=self.LAMBDA_DICT[self.gps_mode],
                    lambda_gal=self.LAMBDA_DICT[self.gal_mode],
                    settings=ar_cfg,
                    gps_age=gps_ar_age,
                    gal_age=gal_ar_age,
                )

            dtr = self.ekf.x[3]
            isb = self.ekf.x[4]
            tro = self.ekf.x[5]

            # Convergence checking logic
            if not reset_epoch or reset_every ==0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time

            xyz = self.ekf.x[:3].copy()  # Update position for the next epoch
            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx,flh=flh,degrees=True)).flatten()
            else:
                enu = np.array([0.0, 0.0, 0.0])
            # # # Save epoch results
            df_epoch = pd.DataFrame(
                {'de': [enu[0]], 'dn': [enu[1]], 'du': [enu[2]], 'dtr': [dtr], 'isb': [isb], 'ztd': [tro],
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]]},
                index=pd.DatetimeIndex([t], name='time'))
            df_epoch['ar_fixed'] = 0 if ar_diag is None else int(ar_diag.fixed_ambiguities)
            df_epoch['ar_ratio'] = np.nan if ar_diag is None or ar_diag.ratio_min is None else float(ar_diag.ratio_min)
            df_epoch['ar_ok'] = False if ar_diag is None else bool(ar_diag.accepted)
            df_epoch['ar_gps_min_age'] = int(np.min(gps_ar_age)) if gps_ar_age.size else np.nan
            df_epoch['ar_gal_min_age'] = int(np.min(gal_ar_age)) if gal_ar_age.size else np.nan
            df_epoch['ar_disabled_reason'] = ar_disabled_reason
            for key, value in pppar_diagnostic_columns(ar_diag).items():
                df_epoch[key] = value
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=df_epoch.iloc[0],
                    innovation_norm=float(np.linalg.norm(self.ekf.y)),
                    innovation_size=len(self.ekf.y),
                )
            result.append(df_epoch)

            gps_epoch['v'] = self.ekf.y[len(gps_epoch):2*len(gps_epoch)]
            gps_epoch['vc'] = self.ekf.y[:len(gps_epoch)]

            gal_epoch['v'] = self.ekf.y[2*len(gps_epoch)+len(gal_epoch):]
            gal_epoch['vc'] = self.ekf.y[2 * len(gps_epoch): 2 * len(gps_epoch) + len(gal_epoch)]


            result_gps.append(gps_epoch)
            result_gal.append(gal_epoch)
        df_result = pd.concat(result)
        df_obs_gps = pd.concat(result_gps)
        df_obs_gal = pd.concat(result_gal)

        if conv_time is not None:
            if trace_filter:
                ppp_trace(
                    trace_filter,
                    self.__class__.__name__,
                    "convergence",
                    threshold_m=0.005,
                    minutes=conv_time.total_seconds() / 60,
                    hours=conv_time.total_seconds() / 3600,
                )

            ct = conv_time.total_seconds() / 3600
        else:
            ct = None
        df_result['ct_min'] = ct
        return df_result,df_obs_gps, df_obs_gal, ct


class PPPDualFreqMultiGNSS:
    """Generic ionosphere-free combined PPP for multiple constellations.

    Purpose:
        Active mixed-system combined PPP branch selected by ``PPPSession`` when
        multiple constellations are available in ``positioning_mode='combined'``.

    Status:
        Active production path for mixed combined PPP. Requires caution around
        mixed-constellation bias assumptions and any BeiDou-specific clock/bias
        handling.

    Model:
        Builds ionosphere-free code and phase combinations independently per
        constellation. One configured/reference constellation supplies the
        receiver clock state; every other active constellation receives an ISB
        state. Ambiguities are stored as one IF ambiguity per ``system:sv``.

    State vector:
        ``[x, y, z, dtr_ref, ISB_nonref..., ztd?, N_IF(system:sv)...]``.

    Supported systems/modes:
        GPS, Galileo and BeiDou dual-frequency modes known to ``gnx_py.gnss``.
        At least two non-empty systems are required.

    PPP-AR support:
        Conventional combined IF AR can be attempted, but only when
        ``pppar_enabled`` and ``pppar_combined_if_ar_enabled`` are both true.
        It is experimental and disabled by default.
        BeiDou-containing AR combinations should be treated as not fully
        validated unless covered by dedicated phase-bias regression tests.

    Bias / clock assumptions:
        The reference-system receiver clock absorbs the common receiver time
        datum. Non-reference clock differences are represented by ISB states.
        Satellite/receiver code-bias corrections are expected to be handled
        before or during observable preparation.

    Limitations:
        Does not estimate slant ionosphere states because the observables are
        ionosphere-free combinations. Mixed-system behavior depends on
        consistent orbit/clock and bias products across constellations.
    """

    SYSTEM_PRIORITY = ("G", "E", "C")

    def __init__(
        self,
        config: PPPConfig,
        gps_obs=None,
        gal_obs=None,
        bds_obs=None,
        gps_mode="L1L2",
        gal_mode="E1E5a",
        bds_mode="B1IB3I",
        ekf=None,
        tro=True,
        pos0=None,
        interval=0.5,
        reference_system=None,
    ):
        self.cfg = config
        self.ekf = ekf
        self.tro = tro
        self.pos0 = pos0
        self.interval = interval
        self.CLIGHT = 299792458.0
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.LAMBDA_DICT = {}

        obs_by_system = {"G": gps_obs, "E": gal_obs, "C": bds_obs}
        mode_by_system = {"G": gps_mode, "E": gal_mode, "C": bds_mode}
        self.obs_by_system = {
            sys: obs.copy()
            for sys, obs in obs_by_system.items()
            if isinstance(obs, pd.DataFrame) and not obs.empty
        }
        self.mode_by_system = {sys: mode_by_system[sys] for sys in self.obs_by_system}
        self.systems = [sys for sys in self.SYSTEM_PRIORITY if sys in self.obs_by_system]
        if len(self.systems) < 2:
            raise ValueError("Generic mixed combined PPP requires at least two non-empty systems.")

        self.reference_system = reference_system or next(sys for sys in self.SYSTEM_PRIORITY if sys in self.systems)
        if self.reference_system not in self.systems:
            raise ValueError(f"Reference system {self.reference_system!r} is not present in mixed PPP data.")
        self.nonref_systems = [sys for sys in self.systems if sys != self.reference_system]
        self.isb_indices = {sys: 4 + idx for idx, sys in enumerate(self.nonref_systems)}
        self.trop_index = 4 + len(self.nonref_systems) if self.tro else None
        self.base_dim = (self.trop_index + 1) if self.tro else (4 + len(self.nonref_systems))

        self.groups = {}
        self._prepared = False
        self.current_amb_keys = []

    def _first_prefixed(self, df, prefix: str, mode: str) -> str:
        for col in df.columns:
            if str(col).startswith(prefix):
                return col
        raise ValueError(f"{mode}: required observation prefix {prefix!r} is missing.")

    def _prepare_obs(self):
        self.groups = {}
        for system in self.systems:
            mode = self.mode_by_system[system]
            obs = self.obs_by_system[system].copy()
            signals = mode_signals(mode)
            if len(signals) != 2:
                raise ValueError(f"Combined mixed PPP requires dual-frequency modes, got {system}:{mode}.")
            sig1, sig2 = signals
            spec1, spec2 = signal_spec(sig1), signal_spec(sig2)
            if spec1.system != system or spec2.system != system:
                raise ValueError(f"Mode {mode!r} does not belong to system {system!r}.")

            a_if, b_if = mode_ionosphere_free_coefficients(mode)
            c1 = self._first_prefixed(obs, spec1.code_prefix, mode)
            c2 = self._first_prefixed(obs, spec2.code_prefix, mode)
            l1 = self._first_prefixed(obs, spec1.phase_prefix, mode)
            l2 = self._first_prefixed(obs, spec2.phase_prefix, mode)
            layout = mode_layout(mode)
            pco_col1 = layout[0]["rec_pco_col"]
            pco_col2 = layout[1]["rec_pco_col"]
            for pco_col in (pco_col1, pco_col2):
                if pco_col not in obs.columns:
                    obs.loc[:, pco_col] = 0.0

            f1 = self.FREQ_DICT[sig1]
            f2 = self.FREQ_DICT[sig2]
            lambda_if = a_if * (self.CLIGHT / f1) - b_if * (self.CLIGHT / f2)
            self.LAMBDA_DICT[mode] = lambda_if
            self.groups[system] = {
                "system": system,
                "mode": mode,
                "obs": obs.sort_index(),
                "coeff": (a_if, b_if),
                "columns": {"C1": c1, "C2": c2, "L1": l1, "L2": l2},
                "signals": (sig1, sig2),
                "rec_pco_cols": (pco_col1, pco_col2),
                "lambda": lambda_if,
            }
            self.obs_by_system[system] = obs.sort_index()
        self._prepared = True
        return self.groups

    def _epoch_maps(self):
        return {
            system: {t: df for t, df in group["obs"].groupby(level="time", sort=False)}
            for system, group in self.groups.items()
        }

    def _common_epochs(self, epoch_maps):
        epoch_sets = [set(epoch_maps[system].keys()) for system in self.systems]
        epochs = sorted(set.intersection(*epoch_sets))
        if not epochs:
            raise ValueError(f"No common epochs for mixed PPP systems {self.systems}.")
        return epochs

    def _amb_keys(self, epoch_groups):
        return [(group["system"], sv) for group in epoch_groups for sv in group["svs"]]

    def _init_or_reset_filter(self, amb_keys):
        n_amb = len(amb_keys)
        dimx = self.base_dim + n_amb
        dimz = 2 * n_amb
        xyz0 = np.array([6371.0e3, 0.0, 0.0]) if self.pos0 is None else np.asarray(self.pos0, dtype=float)
        base = [*xyz0, 0.0]
        base.extend([0.0] * len(self.nonref_systems))
        if self.tro:
            base.append(0.0)
        initial_state = np.concatenate((np.asarray(base, dtype=float), np.zeros(n_amb)))

        self.ekf = ExtendedKalmanFilter(dim_x=dimx, dim_z=dimz)
        self.ekf._I = np.eye(dimx)
        self.ekf.x = initial_state.copy()
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.full(n_amb, self.cfg.q_amb))))
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(n_amb, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.Q[:3, :3] = self.cfg.q_crd
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.Q[3, 3] = self.cfg.q_dt
        for system, idx in self.isb_indices.items():
            # PPPConfig preset for positioning_mode='combined' sets p_isb/q_isb to 0.0,
            # but mixed combined PPP requires estimating ISB(s). We default to p_dt and
            # keep ISB constant unless the user configured otherwise.
            p_isb = getattr(self.cfg, "p_isb", 0.0)
            q_isb = getattr(self.cfg, "q_isb", 0.0)
            self.ekf.P[idx, idx] = self.cfg.p_dt if not p_isb else float(p_isb)
            self.ekf.Q[idx, idx] = 0.0 if not q_isb else float(q_isb)
        if self.tro:
            self.ekf.P[self.trop_index, self.trop_index] = self.cfg.p_tro
            self.ekf.Q[self.trop_index, self.trop_index] = self.cfg.q_tro
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        self.current_amb_keys = list(amb_keys)

    def init_filter(self, epoch_maps, epochs):
        first_groups = self._build_epoch_groups(epochs[0], epoch_maps)
        amb_keys = self._amb_keys(first_groups)
        self._init_or_reset_filter(amb_keys)
        return amb_keys

    def rebuild_state(self, x_old, P_old, Q_old, prev_keys, curr_keys):
        old_dim = self.base_dim + len(prev_keys)
        new_dim = self.base_dim + len(curr_keys)
        new_x = np.zeros(new_dim)
        new_P = np.zeros((new_dim, new_dim))
        new_Q = np.zeros((new_dim, new_dim))
        new_x[:self.base_dim] = x_old[:self.base_dim]
        new_P[:self.base_dim, :self.base_dim] = P_old[:self.base_dim, :self.base_dim]
        new_Q[:self.base_dim, :self.base_dim] = Q_old[:self.base_dim, :self.base_dim]

        prev_map = {key: i for i, key in enumerate(prev_keys)}
        curr_map = {key: i for i, key in enumerate(curr_keys)}
        common = [key for key in curr_keys if key in prev_map]
        for key in common:
            old_i = self.base_dim + prev_map[key]
            new_i = self.base_dim + curr_map[key]
            new_x[new_i] = x_old[old_i]
            new_P[new_i, :self.base_dim] = P_old[old_i, :self.base_dim]
            new_P[:self.base_dim, new_i] = P_old[:self.base_dim, old_i]
            new_Q[new_i, :self.base_dim] = Q_old[old_i, :self.base_dim]
            new_Q[:self.base_dim, new_i] = Q_old[:self.base_dim, old_i]
            for other in common:
                old_j = self.base_dim + prev_map[other]
                new_j = self.base_dim + curr_map[other]
                new_P[new_i, new_j] = P_old[old_i, old_j]
                new_Q[new_i, new_j] = Q_old[old_i, old_j]
        for key in set(curr_keys) - set(prev_keys):
            idx = self.base_dim + curr_map[key]
            new_P[idx, idx] = self.cfg.p_amb
            new_Q[idx, idx] = self.cfg.q_amb
        return new_x, new_P, new_Q

    def _col_np(self, epoch, col, default=0.0):
        value = epoch.get(col, default)
        return np.asarray(value, dtype=float)

    def _tides_los(self, epoch):
        if "tides_los" in epoch.columns:
            return epoch["tides_los"].to_numpy(dtype=float, copy=False)
        if {"LOS1", "LOS2", "LOS3", "dx", "dy", "dz"} <= set(epoch.columns):
            los = epoch[["LOS1", "LOS2", "LOS3"]].to_numpy(dtype=float)
            tides = epoch[["dx", "dy", "dz"]].to_numpy(dtype=float)
            return np.sum(los * tides, axis=1)
        return np.zeros(len(epoch), dtype=float)

    def _bias_corrections(self, system, epoch, columns, coeff):
        a_if, b_if = coeff
        n = len(epoch)
        c1, c2, l1, l2 = columns["C1"], columns["C2"], columns["L1"], columns["L2"]
        code_osb_1 = self._col_np(epoch, f"OSB_{c1}", 0.0) * 1e-9 * self.CLIGHT
        code_osb_2 = self._col_np(epoch, f"OSB_{c2}", 0.0) * 1e-9 * self.CLIGHT
        phase_osb_1 = self._col_np(epoch, f"OSB_{l1}", 0.0) * 1e-9 * self.CLIGHT
        phase_osb_2 = self._col_np(epoch, f"OSB_{l2}", 0.0) * 1e-9 * self.CLIGHT

        code_corr = -(a_if * code_osb_1 - b_if * code_osb_2)
        if system == "C" and getattr(self.cfg, "orbit_type", None) == "precise":
            code_corr = _SingleCombinedBiasModel._bds_precise_clock_bias_correction_m(
                epoch=epoch,
                weighted_codes=[(c1, a_if), (c2, -b_if)],
                n=n,
            )
        phase_corr = code_corr - (a_if * phase_osb_1 - b_if * phase_osb_2)
        return code_corr, phase_corr

    def _build_epoch_groups(self, t, epoch_maps):
        epoch_groups = []
        for system in self.systems:
            group = self.groups[system]
            epoch = epoch_maps[system].get(t)
            if epoch is None or epoch.empty:
                return []
            epoch = epoch.copy()
            a_if, b_if = group["coeff"]
            columns = group["columns"]
            sig1, sig2 = group["signals"]
            pco_col1, pco_col2 = group["rec_pco_cols"]

            pco1 = self._col_np(epoch, pco_col1, 0.0)
            pco2 = self._col_np(epoch, pco_col2, 0.0)
            sat_pco1 = self._col_np(epoch, f"sat_pco_los_{sig1}", 0.0)
            sat_pco2 = self._col_np(epoch, f"sat_pco_los_{sig2}", 0.0)
            sat_pco_if = a_if * sat_pco1 - b_if * sat_pco2
            code_bias_corr, phase_bias_corr = self._bias_corrections(system, epoch, columns, group["coeff"])

            common_corr = (
                self._col_np(epoch, "clk", 0.0) * self.CLIGHT
                - self._col_np(epoch, "tro", 0.0)
                - self._col_np(epoch, "ah_los", 0.0)
                + sat_pco_if
                - self._col_np(epoch, "dprel", 0.0)
                - self._tides_los(epoch)
            )
            code_if = (
                a_if * (epoch[columns["C1"]].to_numpy(dtype=float, copy=False) - pco1)
                - b_if * (epoch[columns["C2"]].to_numpy(dtype=float, copy=False) - pco2)
                + code_bias_corr
                + common_corr
            )
            phase_if = (
                a_if * (epoch[columns["L1"]].to_numpy(dtype=float, copy=False) - pco1)
                - b_if * (epoch[columns["L2"]].to_numpy(dtype=float, copy=False) - pco2)
                + phase_bias_corr
                + common_corr
                - (group["lambda"] / (2 * np.pi)) * self._col_np(epoch, "phw", 0.0)
            )

            epoch_groups.append(
                {
                    "system": system,
                    "epoch": epoch,
                    "svs": epoch.index.get_level_values("sv").tolist(),
                    "sat_positions": epoch[["xe", "ye", "ze", "me_wet"]].to_numpy(dtype=float),
                    "code": code_if,
                    "phase": phase_if,
                    "ev": epoch["ev"].to_numpy(dtype=float, copy=False) if "ev" in epoch.columns else np.full(len(epoch), 45.0),
                    "code_bias_corr": code_bias_corr,
                    "phase_bias_corr": phase_bias_corr,
                    "mode": group["mode"],
                    "lambda": group["lambda"],
                }
            )
        return epoch_groups

    def HJacobian(self, x, epoch_groups):
        dim_state = self.base_dim + sum(len(group["svs"]) for group in epoch_groups)
        receiver = x[:3]
        blocks = []
        amb_offset = self.base_dim
        for group in epoch_groups:
            sats = group["sat_positions"]
            n = len(sats)
            H = np.zeros((2 * n, dim_state))
            distances = np.linalg.norm((sats[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
            isb_idx = self.isb_indices.get(group["system"])
            for i in range(n):
                los_der = (receiver - sats[i, :3]) / distances[i]
                for row in (i, n + i):
                    H[row, 0:3] = los_der
                    H[row, 3] = 1.0
                    if isb_idx is not None:
                        H[row, isb_idx] = 1.0
                    if self.tro:
                        H[row, self.trop_index] = sats[i, -1]
                H[n + i, amb_offset + i] = 1.0
            blocks.append(H)
            amb_offset += n
        return np.vstack(blocks)

    def Hx(self, x, epoch_groups):
        receiver = x[:3]
        clock = x[3]
        trop = x[self.trop_index] if self.tro else 0.0
        out = []
        amb_offset = self.base_dim
        for group in epoch_groups:
            sats = group["sat_positions"]
            n = len(sats)
            distances = np.linalg.norm((sats[:, :3] - receiver).astype(np.float64, copy=False), axis=1)
            isb = x[self.isb_indices[group["system"]]] if group["system"] in self.isb_indices else 0.0
            trop_part = sats[:, -1] * trop if self.tro else 0.0
            amb = x[amb_offset:amb_offset + n]
            out.append(np.concatenate((distances + clock + isb + trop_part, distances + clock + isb + trop_part + amb)))
            amb_offset += n
        return np.concatenate(out)

    def code_screening(self, x, satellites, code_obs, thr=3):
        return ppp_code_screening(x=x, satellites=satellites[:, :3], code_obs=code_obs, thr=thr)

    def _weight_vectors(self, group, mask):
        ev = group["ev"]
        sin_el = np.sin(np.deg2rad(ev))
        inv_sin2 = 1.0 / np.square(sin_el)
        if group["system"] == "C":
            code_base, phase_base = 9.0, 9e-4
        else:
            code_base, phase_base = 1.0, 1e-4
        sigma_code = code_base + 0.0025 * inv_sin2
        sigma_phase = phase_base + 0.0003 * inv_sin2
        r_vec = np.concatenate((sigma_code, sigma_phase))
        r_vec[~np.concatenate((mask, mask))] = 1e12
        return r_vec

    def _disable_unsupported_ar(self, ar_cfg):
        if not ar_cfg.enabled:
            return ar_cfg
        if self.systems == ["G", "E"]:
            return ar_cfg
        return PPPARSettings(
            enabled=False,
            warmup_epochs=ar_cfg.warmup_epochs,
            min_ambiguities=ar_cfg.min_ambiguities,
            ratio_threshold=ar_cfg.ratio_threshold,
            constraint_sigma_cycles=ar_cfg.constraint_sigma_cycles,
            constraint_sigma_floor_cycles=ar_cfg.constraint_sigma_floor_cycles,
            min_lock_epochs=ar_cfg.min_lock_epochs,
            min_candidate_elevation_deg=ar_cfg.min_candidate_elevation_deg,
            use_float_ratio_covariance=ar_cfg.use_float_ratio_covariance,
            partial_fixing_enabled=ar_cfg.partial_fixing_enabled,
            partial_min_ambiguities=ar_cfg.partial_min_ambiguities,
            wide_lane_max_frac_cycles=ar_cfg.wide_lane_max_frac_cycles,
        )

    def run_filter(self, ref=None, flh=None, add_dcb=True, trace_filter=False, reset_every=180):
        if not self._prepared:
            self._prepare_obs()
        epoch_maps = self._epoch_maps()
        epochs = self._common_epochs(epoch_maps)
        old_keys = self.init_filter(epoch_maps, epochs)

        ar_cfg = self._disable_unsupported_ar(PPPARSettings(
            enabled=combined_if_pppar_enabled(self.cfg),
            warmup_epochs=int(getattr(self.cfg, "pppar_warmup_epochs", 60)),
            min_ambiguities=int(getattr(self.cfg, "pppar_min_ambiguities", 4)),
            ratio_threshold=float(getattr(self.cfg, "pppar_ratio_threshold", 2.0)),
            constraint_sigma_cycles=float(getattr(self.cfg, "pppar_constraint_sigma_cycles", 1e-3)),
            constraint_sigma_floor_cycles=getattr(self.cfg, "pppar_constraint_sigma_floor_cycles", 1e-3),
            min_lock_epochs=getattr(self.cfg, "pppar_min_lock_epochs", None),
            min_candidate_elevation_deg=getattr(self.cfg, "pppar_min_candidate_elevation_deg", None),
            use_float_ratio_covariance=bool(getattr(self.cfg, "pppar_use_float_ratio_covariance", True)),
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", False)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
        ))
        ar_disabled_reason = (
            "combined_if_ar_disabled"
            if bool(getattr(self.cfg, "pppar_enabled", False)) and not ar_cfg.enabled
            else None
        )

        r_cache = {}
        result = []
        residuals_by_system = {system: [] for system in self.systems}
        phase_traces = {system: {} for system in self.systems}
        xyz = None
        conv_time = None
        T0 = epochs[0]
        arc_age = {key: 0 for key in old_keys}

        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0 and ((num * self.interval) % reset_every == 0) and num != 0:
                epoch_groups = self._build_epoch_groups(t, epoch_maps)
                self._init_or_reset_filter(self._amb_keys(epoch_groups))
                old_keys = list(self.current_amb_keys)
                arc_age = {key: 0 for key in old_keys}
                reset_epoch = True
                T0 = t

            epoch_groups = self._build_epoch_groups(t, epoch_maps)
            if not epoch_groups:
                continue
            curr_keys = self._amb_keys(epoch_groups)
            if curr_keys != old_keys:
                new_x, new_P, new_Q = self.rebuild_state(
                    self.ekf.x.copy(), self.ekf.P.copy(), self.ekf.Q.copy(), old_keys, curr_keys
                )
                self.ekf.x, self.ekf.P, self.ekf.Q = new_x, new_P, new_Q
                self.ekf.F = np.eye(len(self.ekf.x))
                self.ekf.F[3, 3] = 0.0
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf._I = np.eye(self.ekf.dim_x)
            old_keys = curr_keys
            self.current_amb_keys = curr_keys
            arc_age = {key: arc_age.get(key, 0) for key in curr_keys}

            z = np.concatenate([np.concatenate((group["code"], group["phase"])) for group in epoch_groups])
            self.ekf.dim_z = len(z)

            masks = []
            phase_reset_counts = {}
            phase_reset_keys = set()
            amb_offset = 0
            for group in epoch_groups:
                n = len(group["svs"])
                dist = calculate_distance(group["sat_positions"][:, :3].copy(), self.ekf.x[:3].copy())
                prefit_phase = group["phase"] - dist
                phase_traces[group["system"]][num] = {sv: r for sv, r in zip(group["svs"], prefit_phase)}
                mask = self.code_screening(
                    x=self.ekf.x[:3].copy(),
                    satellites=group["sat_positions"],
                    code_obs=group["code"],
                    thr=3,
                )
                reset_count = 0
                if num > 60:
                    for idx in phase_residuals_outliers(group["svs"], phase_traces[group["system"]], num, thr=1):
                        state_idx = self.base_dim + amb_offset + idx
                        self.ekf.x[state_idx] = 0.0
                        self.ekf.P[state_idx, state_idx] = 400.0
                        phase_reset_keys.add((group["system"], group["svs"][idx]))
                        reset_count += 1
                masks.append(mask)
                phase_reset_counts[group["system"]] = reset_count
                amb_offset += n

            r_vec = np.concatenate([self._weight_vectors(group, mask) for group, mask in zip(epoch_groups, masks)])
            m = r_vec.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = r_vec
            self.ekf.R = R

            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=epoch_groups, Hx=self.Hx, hx_args=epoch_groups)
            arc_age = advance_arc_age(arc_age, curr_keys, phase_reset_keys)
            age_by_key = dict(arc_age)
            postfit = z - self.Hx(self.ekf.x, epoch_groups)
            innovation = np.asarray(self.ekf.y, dtype=float)

            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs and self.systems == ["G", "E"]:
                g_group = next(group for group in epoch_groups if group["system"] == "G")
                e_group = next(group for group in epoch_groups if group["system"] == "E")
                self.ekf.x, self.ekf.P, ar_diag = apply_conventional_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(g_group["svs"]),
                    n_gal=len(e_group["svs"]),
                    gps_ev=g_group["ev"],
                    gal_ev=e_group["ev"],
                    gps_age=arc_age_array(age_by_key, [("G", sv) for sv in g_group["svs"]]),
                    gal_age=arc_age_array(age_by_key, [("E", sv) for sv in e_group["svs"]]),
                    lambda_gps=self.groups["G"]["lambda"],
                    lambda_gal=self.groups["E"]["lambda"],
                    settings=ar_cfg,
                )

            dtr = self.ekf.x[3]
            ztd = self.ekf.x[self.trop_index] if self.tro else 0.0
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0
                    elif position_diff > 0.005:
                        conv_time = None
            xyz = self.ekf.x[:3].copy()

            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
            else:
                enu = np.array([0.0, 0.0, 0.0])

            row = {
                "de": enu[0],
                "dn": enu[1],
                "du": enu[2],
                "dtr": dtr,
                "ztd": ztd,
                "x": self.ekf.x[0],
                "y": self.ekf.x[1],
                "z": self.ekf.x[2],
                "reference_system": self.reference_system,
                "systems": "+".join(self.systems),
                "n_sats_total": len(curr_keys),
                "n_states": len(self.ekf.x),
                "n_code_rejected_total": 0,
                "n_phase_reset_total": sum(phase_reset_counts.values()),
                "ar_fixed": 0 if ar_diag is None else int(ar_diag.fixed_ambiguities),
                "ar_ratio": np.nan if ar_diag is None or ar_diag.ratio_min is None else float(ar_diag.ratio_min),
                "ar_ok": False if ar_diag is None else bool(ar_diag.accepted),
                "ar_disabled_reason": ar_disabled_reason,
            }
            row.update(pppar_diagnostic_columns(ar_diag))
            if len(self.nonref_systems) == 1:
                row["isb"] = self.ekf.x[self.isb_indices[self.nonref_systems[0]]]
            for system, idx in self.isb_indices.items():
                row[f"isb_{system}"] = self.ekf.x[idx]

            z_cursor = 0
            amb_cursor = self.base_dim
            for group, mask in zip(epoch_groups, masks):
                system = group["system"]
                n = len(group["svs"])
                group_post = postfit[z_cursor:z_cursor + 2 * n]
                group_prefit = innovation[z_cursor:z_cursor + 2 * n]
                code_post = group_post[:n]
                phase_post = group_post[n:]
                code_prefit = group_prefit[:n]
                phase_prefit = group_prefit[n:]
                used_mask = np.asarray(mask, dtype=bool)
                row["n_code_rejected_total"] += int(np.count_nonzero(~used_mask))
                row[f"n_sats_{system}"] = n
                row[f"n_code_rejected_{system}"] = int(np.count_nonzero(~used_mask))
                row[f"n_phase_reset_{system}"] = phase_reset_counts.get(system, 0)
                group_ages = arc_age_array(age_by_key, [(system, sv) for sv in group["svs"]])
                row[f"ar_{system}_min_age"] = int(np.min(group_ages)) if group_ages.size else np.nan
                if np.any(used_mask):
                    code_common = float(np.median(code_post[used_mask]))
                    phase_common = float(np.median(phase_post[used_mask]))
                    code_scatter = code_post - code_common
                    phase_scatter = phase_post - phase_common
                    row[f"code_res_rms_{system}"] = float(np.sqrt(np.mean(np.square(code_post[used_mask]))))
                    row[f"phase_res_rms_{system}"] = float(np.sqrt(np.mean(np.square(phase_post[used_mask]))))
                    row[f"code_res_scatter_rms_{system}"] = float(np.sqrt(np.mean(np.square(code_scatter[used_mask]))))
                    row[f"phase_res_scatter_rms_{system}"] = float(np.sqrt(np.mean(np.square(phase_scatter[used_mask]))))
                else:
                    code_common = phase_common = np.nan
                    code_scatter = np.full(n, np.nan)
                    phase_scatter = np.full(n, np.nan)
                    row[f"code_res_rms_{system}"] = np.nan
                    row[f"phase_res_rms_{system}"] = np.nan
                    row[f"code_res_scatter_rms_{system}"] = np.nan
                    row[f"phase_res_scatter_rms_{system}"] = np.nan

                epoch = group["epoch"].copy()
                epoch["system"] = system
                epoch["reference_system"] = self.reference_system
                epoch["is_reference"] = system == self.reference_system
                epoch["dtr"] = dtr
                epoch["ztd"] = ztd
                epoch["isb"] = 0.0 if system == self.reference_system else self.ekf.x[self.isb_indices[system]]
                epoch["P3"] = group["code"]
                epoch["L3"] = group["phase"]
                epoch["Nif"] = self.ekf.x[amb_cursor:amb_cursor + n].copy()
                epoch["prefit_code_res"] = code_prefit
                epoch["prefit_phase_res"] = phase_prefit
                epoch["postfit_code_res"] = code_post
                epoch["postfit_phase_res"] = phase_post
                epoch["postfit_code_res_demeaned"] = code_scatter
                epoch["postfit_phase_res_demeaned"] = phase_scatter
                epoch["res"] = epoch["postfit_phase_res"]
                epoch["code_bias_corr"] = group["code_bias_corr"]
                epoch["phase_bias_corr"] = group["phase_bias_corr"]
                epoch["p_if"] = np.diag(self.ekf.P[amb_cursor:amb_cursor + n, amb_cursor:amb_cursor + n])
                residuals_by_system[system].append(epoch)

                z_cursor += 2 * n
                amb_cursor += n

            result.append(pd.DataFrame(row, index=pd.DatetimeIndex([t], name="time")))
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result[-1].iloc[0],
                )

        df_result = pd.concat(result)
        ct = None if conv_time is None else conv_time.total_seconds() / 3600
        df_result["ct_min"] = ct
        residual_frames = {
            system: pd.concat(frames) if frames else None
            for system, frames in residuals_by_system.items()
        }
        return df_result, residual_frames, None, ct
