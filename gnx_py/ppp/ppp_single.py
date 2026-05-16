"""Single-constellation PPP filter classes.

The classes in this module cover single-system PPP variants where all active
observations belong to one constellation. Combined dual-frequency filters use
ionosphere-free observables; single-frequency filters estimate a float
single-frequency ambiguity without ionospheric constraints. Uncombined dual-
frequency constrained models live in ``ppp_uduc.py``.
"""

import warnings

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
from ..gnss import SIGNALS, frequency_hz, mode_ionosphere_free_coefficients, mode_layout, mode_signals, signal_spec

FREQ_DICT_ALL = {name: spec.frequency_hz for name, spec in SIGNALS.items()}

class PPPSingleFreqSingleGNSS:
    """Single-frequency, single-system PPP filter without ionospheric constraints.

    Purpose:
        Legacy/compatibility float PPP path for one constellation and one
        signal. It is selected for uncombined single-frequency operation when
        ``PPPConfig.use_iono_constr`` is false.

    Status:
        Legacy compatibility path, but still reachable from session routing and
        covered by tests. Prefer ``PPPUducSFSingleGNSS`` for constrained
        single-frequency work. This is a deprecated-candidate only after a
        replacement strategy exists for no-constraint single-frequency runs.

    Model:
        Uses one code and one phase observable per satellite. The ionospheric
        delay is not estimated as a separate state and is not constrained by
        external STEC/GIM data in this class.

    State vector:
        ``[x, y, z, dtr, ztd?, N_s1, ..., N_sn]`` where ``ztd`` is present when
        ``tro=True`` and ``N_si`` is the float phase ambiguity in meters for
        satellite ``si``.

    Supported systems/modes:
        Any single-frequency mode known to ``gnx_py.gnss`` for GPS, Galileo or
        BeiDou, provided the input observation columns match the selected mode.

    PPP-AR support:
        Not applied here.

    Limitations:
        This path is useful for compatibility and simple smoke runs, but it is
        weaker than constrained uncombined single-frequency PPP because the
        ionosphere is not modeled independently.
    """

    def __init__(self, config:PPPConfig, obs, mode, ekf, tro=False, pos0=None, interval=0.5):

        """
        PPP Single Frequency, Single GNSS estimator
        :param obs: pd.DataFrame, GNSS observations
        :param mode: str, mode/signals eg. L1L2, E1E5a
        :param ekf: filterpy.ExtendedKalmxanFilter instance
        :param tro: bool, estimate ZTD
        :param pos0: np.ndarray, initial position
        :param interval: float, interval of observations
        """
        self.cfg=config
        self.obs = obs
        self.mode =mode
        self.tro =tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.CLIGHT = 299792458
        self.base_dim = 5 if self.tro else 4
        self.pos0 = pos0
        self.interval=interval

    def HJacobian(self, x, gps_satellites):
        """
        Hjacobian matrix
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, satellite coordinates & mapping function
=       :return: Hjacobian matrix
        """
        num_gps = len(gps_satellites)
        N = num_gps
        # State: x,y,z, dt_gps, tro, isb, N1...Nn
        dim_state = self.base_dim + N

        receiver = x[:3]
        # GPS PART
        H1 = np.zeros((2 * num_gps, dim_state))
        distances_gps = np.linalg.norm(gps_satellites[:, :3] - receiver, axis=1)
        for i in range(num_gps):
            # Code observation (wiersz i)
            H1[i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[i, 3] = 1.0  # clock
            if self.tro:
                H1[i, 4] = gps_satellites[i, -1]

            # Phase observation (wiersz N+i)
            H1[num_gps + i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[num_gps + i, 3] = 1.0  # clock
            if self.tro:
                H1[num_gps+i, 4] = gps_satellites[i,-1]

            H1[num_gps + i, self.base_dim + i] = 1  # AMB

        return H1

    def Hx(self, x ,gps_satellites):
        """
        Predicted observations vector
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :return: predicted obs matrix
        """
        num_gps = len(gps_satellites)
        receiver = x[:3]
        clock = x[3]
        if self.tro:
            tro = gps_satellites[:, -1] * x[4]
        else:
            tro = 0.0
        # GPS prediction
        distances_gps = np.linalg.norm(gps_satellites[:, :3] - receiver, axis=1)
        h_code_gps = distances_gps + clock + tro
        ambiguities_gps = x[self.base_dim:self.base_dim + num_gps]
        h_phase_gps = distances_gps + clock + tro + ambiguities_gps

        return np.concatenate((h_code_gps, h_phase_gps))

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

        epochs = sorted(self.obs.index.get_level_values('time').unique().tolist())
        
        obs0 = self.obs.loc[(slice(None), epochs[0]), :]
    

        gps0 = obs0.index.get_level_values('sv').tolist()
        N0 = len(gps0) 

        dimx = self.base_dim + N0
        dimz = 2 * N0
        # x
        if self.pos0 is None:

            xyz0 = np.array([6371.0e3, 0.0, 0.0])
            if self.tro:
                initial_state = np.concatenate((xyz0, [0.0],[0.0], np.zeros(N0)))
            else:
                initial_state = np.concatenate((xyz0, [0.0], np.zeros(N0)))
        else:
            if self.tro:
                initial_state = np.concatenate((self.pos0, [0.0],[0.0], np.zeros(N0)))
            else:
                initial_state = np.concatenate((self.pos0, [0.0], np.zeros(N0)))
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

        if self.tro:
            self.ekf.P[4,4] = self.cfg.p_tro
            self.ekf.Q[4, 4] = self.cfg.q_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0 

        return old_sats, epochs

    def reset_filter(self, epoch):

        """
        reinitialize (reset) Kalman filter. Init filter function but based on given epoch
        """
        obs0 = self.obs.loc[(slice(None), epoch), :]

        gps0 = obs0.index.get_level_values('sv').tolist()
        N0 = len(gps0)

        dimx = self.base_dim + N0
        dimz = 2 * N0
        # x
        if self.pos0 is None:

            xyz0 = np.array([6371.0e3, 0.0, 0.0])
            if self.tro:
                initial_state = np.concatenate((xyz0, [0.0], [0.0], np.zeros(N0)))
            else:
                initial_state = np.concatenate((xyz0, [0.0], np.zeros(N0)))
        else:
            if self.tro:
                initial_state = np.concatenate((self.pos0, [0.0], [0.0], np.zeros(N0)))
            else:
                initial_state = np.concatenate((self.pos0, [0.0], np.zeros(N0)))
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

        if self.tro:
            self.ekf.P[4, 4] = self.cfg.p_tro
            self.ekf.Q[4, 4] = self.cfg.q_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0

        return old_sats

    def _prepare_obs(self):
        signals = mode_signals(self.mode)
        if len(signals) != 1:
            raise ValueError(f"Mode {self.mode} is not single-frequency.")
        spec = signal_spec(signals[0])
        gps_c1_col = [c for c in self.obs.columns if c.startswith(spec.code_prefix)][0]
        gps_l1_col = [c for c in self.obs.columns if c.startswith(spec.phase_prefix)][0]
        data = (gps_c1_col, gps_l1_col)

        self.obs = self.obs.copy()
        if 'pco_los' not in self.obs.columns.tolist():
            self.obs.loc[:, 'pco_los'] = 0.0
        return data

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
        for idx in phase_residuals_outliers(
            sat_list=sat_list, phase_residuals_dict=phase_residuals_dict, num=num, thr=thr
        ):
            outlier_idx = idx
            if sys == 'E':
                outlier_idx += len_gps
            self.ekf.x[self.base_dim + outlier_idx] = 0.0
            self.ekf.P[
                self.base_dim + outlier_idx, self.base_dim + outlier_idx
            ] = 1e3
            reset_svs.append(sat_list[idx])
        return reset_svs

    def run_filter(self, ref=None, flh=None,trace_filter=False, reset_every =0):
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

        observ_data = self._prepare_obs()
        gps_c1, gps_l1 = observ_data

        result = []
        result_gps = []
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
        self.obs = self.obs.sort_values(by='sv')
        obs_epochs = {t: df for t, df in self.obs.groupby(level=1, sort=False)}
        r_cache = {}
        xyz = None
        T0 = epochs[0]
        conv_time = None
        arc_age = {sv: 0 for sv in old_sats}
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)
                    arc_age = {sv: 0 for sv in old_sats}
                    reset_epoch = True
                    T0 = t
            gps_epoch = obs_epochs.get(t)
            if gps_epoch is None:
                continue
            if len(gps_epoch)<4:
                ppp_trace(
                    trace_filter,
                    self.__class__.__name__,
                    "insufficient-satellites",
                    epoch=num,
                    time=t,
                    n_sats=len(gps_epoch),
                    min_sats=4,
                )
                continue
            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_sats = curr_gps_sats 
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
            arc_age = {sv: arc_age.get(sv, 0) for sv in curr_sats}
            gps_clk, gps_tro, gps_ah_los, gps_sat_pco_los, gps_dprel, gps_pco1, gps_ion, gps_phw = [
                gps_epoch[col].to_numpy()
                for col in
                ['clk', 'tro', 'ah_los',
                 f'sat_pco_los_{self.mode}',
                 'dprel', 'pco_los', 'ion', 'phw']]

            gps_osb_c1 = np.asarray(gps_epoch.get(f'OSB_{gps_c1}',0.0)) * 1e-09 * self.CLIGHT
            gps_osb_c2 = np.asarray(gps_epoch.get(f'OSB_{gps_l1}',0.0)) * 1e-09 * self.CLIGHT


            GP3_code = (
                    (gps_epoch[f'{gps_c1}'].to_numpy() - gps_pco1-gps_osb_c1)
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel 
                    - gps_ion
            )

            GL3_phase = (
                    (gps_epoch[f'{gps_l1}'].to_numpy() - gps_pco1-gps_osb_c2)
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel
                    + gps_ion -  ((self.CLIGHT / self.FREQ_DICT[self.mode]) / (2 * np.pi)) * gps_epoch['phw'].to_numpy()
            )


            z = np.concatenate((GP3_code, GL3_phase))
            self.ekf.dim_z = len(z)
            sat_positions_gps = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            mask = self.code_screening(x=self.ekf.x[:3].copy(), code_obs=GP3_code,satellites=sat_positions_gps[:,:3],
                                       thr=3)
            mask2 = np.concatenate((mask,mask))
            ev_gps = gps_epoch['ev'].to_numpy(copy=False)
            sin_el = np.sin(np.deg2rad(ev_gps))
            inv_sin2 = 1.0 / np.square(sin_el)
            sigma_code_gps = 0.5 + 0.0025 * inv_sin2
            sigma_phase_gps = 9e-4 + 0.00025 * inv_sin2
            R_vec = np.concatenate((sigma_code_gps, sigma_phase_gps))
            if num > 30:
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

            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=(sat_positions_gps),
                                    Hx=self.Hx, hx_args=(sat_positions_gps))
            arc_age = advance_arc_age(arc_age, curr_sats)
            ar_age = arc_age_array(arc_age, curr_sats)
            gps_ev = gps_epoch['ev'].to_numpy(copy=False) if 'ev' in gps_epoch.columns else np.full(
                len(curr_gps_sats), 45.0, dtype=float
            )
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                lam = self.CLIGHT / self.FREQ_DICT[self.mode]
                self.ekf.x, self.ekf.P, ar_diag = apply_conventional_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(curr_gps_sats),
                    n_gal=0,
                    gps_ev=gps_ev,
                    gal_ev=np.empty(0, dtype=float),
                    lambda_gps=lam,
                    lambda_gal=lam,
                    settings=ar_cfg,
                    gps_age=ar_age,
                )
            if not reset_epoch or reset_every ==0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time

            xyz = self.ekf.x[:3].copy()  # Update position for the next epoch

            dtr = self.ekf.x[3]
            isb = 0.0
            if self.tro:
                tro =self.ekf.x[4]
            else:
                tro = 0.0
            if ref is not None and flh is not None:

                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx,flh=flh,degrees=True)).flatten()
            else:
                enu = np.array([0.0, 0.0, 0.0])

            df_epoch = pd.DataFrame(
                {'de': [enu[0]], 'dn': [enu[1]], 'du': [enu[2]], 'dtr': [dtr], 'isb': [isb], 'ztd': [tro],
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]],
                 'ar_fixed': [0 if ar_diag is None else ar_diag.fixed_ambiguities],
                 'ar_ratio': [np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min],
                 'ar_ok': [False if ar_diag is None else ar_diag.accepted],
                 'ar_min_age': [int(np.min(ar_age)) if ar_age.size else np.nan]},
                index=pd.DatetimeIndex([t], name='time'))
            for key, value in pppar_diagnostic_columns(ar_diag).items():
                df_epoch[key] = value
            if trace_filter:
                if reset_epoch:
                    ppp_trace(trace_filter, self.__class__.__name__, "reset", epoch=num, time=t)
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=df_epoch.iloc[0],
                )
            result.append(df_epoch)
            gps_epoch['dtr'] = dtr
            gps_epoch['ztd'] = tro
            gps_epoch['L3'] = GL3_phase
            gps_epoch['P3'] = GP3_code
            gps_epoch['Nif'] = self.ekf.x[self.base_dim:self.base_dim + len(gps_epoch)].copy()
            gps_epoch['res'] = self.ekf.y[len(gps_epoch):2 * len(gps_epoch)]
            gps_epoch['p_if'] = np.diag(self.ekf.P[self.base_dim:self.base_dim + len(gps_epoch),
                                        self.base_dim:self.base_dim + len(gps_epoch)])

            result_gps.append(gps_epoch)
        df_gps = pd.concat(result_gps)
        df_result = pd.concat(result)
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
        return df_result, df_gps, None, None


class PPPDualFreqSingleGNSS:
    """Dual-frequency combined PPP filter for one constellation.

    Purpose:
        Active single-system ionosphere-free PPP branch used when
        ``PPPConfig.positioning_mode='combined'`` and only one constellation is
        present after preprocessing.

    Status:
        Active production path for single-system combined PPP.

    Model:
        Builds ionosphere-free code and phase combinations from the configured
        dual-frequency mode. The filter estimates receiver position, receiver
        clock for the selected constellation, optional ZTD, and one IF
        ambiguity per satellite.

    State vector:
        ``[x, y, z, dtr, ztd?, N_IF_s1, ..., N_IF_sn]`` where ``dtr`` is the
        receiver clock in meters and ``N_IF`` is the combined phase ambiguity.

    Supported systems/modes:
        GPS, Galileo or BeiDou single-system operation. The mode must contain
        two signals from the same constellation, for example ``L1L2``,
        ``E1E5a`` or ``B1IB3I``.

    PPP-AR support:
        Conventional IF AR can be attempted through ``pppar.py`` only when the
        configuration explicitly enables it. Combined IF AR is experimental and
        guarded by ``pppar_combined_if_ar_enabled``.
        BeiDou combined AR is blocked in this class until BDS phase-bias
        handling is validated.

    Limitations:
        This class does not estimate slant ionosphere states. Bias handling is
        limited to the combined-observable model and depends on preprocessing.
    """

    def __init__(self, config:PPPConfig,gps_obs,  gps_mode, ekf, tro=True, pos0=None,interval=0.5, system=None):

        """
        PPP Dual Frequency, Single GNSS estimator
        :param gps_obs: pd.DataFrame, gps observations
        :param gps_mode: str, signals eg L1L2, E1E5a
        :param ekf: filterpy.ExtendedKalmanFilter instance
        :param tro: bool, estimate ZTD
        :param pos0: np.ndarray, initial position
        :param interval: float, interval of observations
        """
        self.cfg = config
        self.gps_obs = gps_obs
        self.gps_mode = gps_mode
        mode_systems = {signal_spec(sig).system for sig in mode_signals(gps_mode)}
        if len(mode_systems) != 1:
            raise ValueError(f"Single-system PPP mode {gps_mode!r} mixes constellations: {sorted(mode_systems)}")
        inferred_system = next(iter(mode_systems))
        if system is not None and system != inferred_system:
            raise ValueError(f"Mode {gps_mode!r} belongs to system {inferred_system}, not {system}.")
        self.system = inferred_system
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.LAMBDA_DICT = {}
        self.CLIGHT = 299792458
        self.base_dim = 5 if self.tro else 4
        self.pos0 = pos0
        self.interval = interval
        self.if_coefficients = None
        self.if_wavelength = None
        self.obs_columns = None
        self.rec_pco_columns = None

    def HJacobian(self, x, gps_satellites):
        """
        Hjacobian matrix
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, satellite coordinates & mapping function
=       :return: Hjacobian matrix
        """
        num_gps = len(gps_satellites)
        N = num_gps
        # State: x,y,z, dt_gps, tro, isb, N1...Nn
        dim_state = self.base_dim + N

        receiver = x[:3]
        # GPS PART
        H1 = np.zeros((2 * num_gps, dim_state))
        distances_gps = np.linalg.norm(gps_satellites[:, :3] - receiver, axis=1)
        for i in range(num_gps):
            # Code observation (wiersz i)
            H1[i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[i, 3] = 1.0  # clock
            H1[i, 4] = gps_satellites[i, -1]  # trop

            # Phase observation (wiersz N+i)
            H1[num_gps + i, 0:3] = (receiver - gps_satellites[i, :3]) / distances_gps[i]
            H1[num_gps + i, 3] = 1.0  # clock
            H1[num_gps + i, 4] = gps_satellites[i, -1]  # trop
            H1[num_gps + i, self.base_dim + i] = 1  # AMB

        return H1

    def Hx(self, x, gps_satellites):
        """
        Predicted observations vector
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :return: predicted obs matrix
        """
        num_gps = len(gps_satellites)
        receiver = x[:3]
        clock = x[3]
        tropo = x[4]
        # GPS prediction
        distances_gps = np.linalg.norm(gps_satellites[:, :3] - receiver, axis=1)
        h_code_gps = distances_gps + clock + gps_satellites[:, -1] * tropo
        ambiguities_gps = x[self.base_dim:]
        h_phase_gps = distances_gps + clock + ambiguities_gps + gps_satellites[:, -1] * tropo
        return np.concatenate((h_code_gps, h_phase_gps))

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
        epochs = sorted(self.gps_obs.index.get_level_values('time').unique().tolist())
        gps_obs0 = self.gps_obs.loc[(slice(None), epochs[0]), :]

        gps0 = gps_obs0.index.get_level_values('sv').tolist()
        N0 = len(gps0)

        dimx = self.base_dim + N0
        dimz = 2 * N0
        # x
        if self.pos0 is None:

            xyz0 = np.array([6371.0e3, 0.0, 0.0])
            initial_state = np.concatenate((xyz0, [0.0], [0.0], np.zeros(N0)))
        else:
            initial_state = np.concatenate((self.pos0, [0.0], [0.0], np.zeros(N0)))
        self.ekf = ExtendedKalmanFilter(dim_x=dimx, dim_z=dimz)
        self.ekf._I = np.eye(self.ekf.dim_x)
        self.ekf.x = initial_state.copy()

        # Q
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.zeros(N0))))
        self.ekf.Q[:3, :3] = self.cfg.q_crd
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(N0, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.P[4, 4] = self.cfg.p_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0

        return old_sats, epochs

    def reset_filter(self, epoch):

        """
        reinitialize (reset) Kalman filter. Init filter function but based on given epoch
        """

        gps_obs0 = self.gps_obs.loc[(slice(None), epoch), :]

        gps0 = gps_obs0.index.get_level_values('sv').tolist()
        N0 = len(gps0)

        dimx = self.base_dim + N0
        dimz = 2 * N0
        # x
        if self.pos0 is None:

            xyz0 = np.array([6371.0e3, 0.0, 0.0])
            initial_state = np.concatenate((xyz0, [0.0], [0.0], np.zeros(N0)))
        else:
            initial_state = np.concatenate((self.pos0, [0.0], [0.0], np.zeros(N0)))
        self.ekf = ExtendedKalmanFilter(dim_x=dimx, dim_z=dimz)
        self.ekf._I = np.eye(self.ekf.dim_x)
        self.ekf.x = initial_state.copy()

        # Q
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.zeros(N0))))
        self.ekf.Q[:3, :3] = self.cfg.q_crd
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(N0, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.P[4, 4] = self.cfg.p_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0

        return old_sats

    def _prepare_obs(self):
        signals = mode_signals(self.gps_mode)
        if len(signals) != 2:
            raise ValueError(f"Ionosphere-free PPP requires a dual-frequency mode, got {self.gps_mode!r}.")

        sig1, sig2 = signals
        spec1, spec2 = signal_spec(sig1), signal_spec(sig2)

        def _first_prefixed(prefix: str) -> str:
            for col in self.gps_obs.columns:
                if col.startswith(prefix):
                    return col
            raise ValueError(f"{self.gps_mode}: required observation prefix {prefix!r} is missing.")

        agps, bgps = mode_ionosphere_free_coefficients(self.gps_mode)
        GF1 = self.FREQ_DICT[sig1]
        GF2 = self.FREQ_DICT[sig2]
        gps_c1_col = _first_prefixed(spec1.code_prefix)
        gps_c2_col = _first_prefixed(spec2.code_prefix)
        gps_l1_col = _first_prefixed(spec1.phase_prefix)
        gps_l2_col = _first_prefixed(spec2.phase_prefix)

        self.LAMBDA_DICT[self.gps_mode] = agps * (self.CLIGHT / GF1) - bgps * (self.CLIGHT / GF2)
        self.if_coefficients = (agps, bgps)
        self.if_wavelength = self.LAMBDA_DICT[self.gps_mode]

        layout = mode_layout(self.gps_mode)
        pco_col1 = layout[0]["rec_pco_col"]
        pco_col2 = layout[1]["rec_pco_col"]
        self.obs_columns = {
            "code_1": gps_c1_col,
            "code_2": gps_c2_col,
            "phase_1": gps_l1_col,
            "phase_2": gps_l2_col,
        }
        self.rec_pco_columns = (pco_col1, pco_col2)

        gps_data = (agps, bgps, gps_c1_col, gps_c2_col, gps_l1_col, gps_l2_col, sig1, sig2, pco_col1, pco_col2)

        self.gps_obs = self.gps_obs.copy()



        for pco_col in (pco_col1, pco_col2):
            if pco_col not in self.gps_obs.columns.tolist():
                self.gps_obs.loc[:, pco_col] = 0.0

        return gps_data

    @staticmethod
    def _bias_col_m(epoch: pd.DataFrame, column: str, n: int):
        if column not in epoch.columns:
            return None
        values = epoch[column].to_numpy(dtype='float64', copy=False)
        return np.nan_to_num(values, nan=0.0) * 1e-9 * 299792458

    @classmethod
    def _dcb_between_m(cls, epoch: pd.DataFrame, obs_a: str, obs_b: str, n: int):
        if obs_a == obs_b:
            return np.zeros(n, dtype='float64')

        direct = cls._bias_col_m(epoch, f"BIAS_{obs_a}_{obs_b}", n)
        if direct is not None:
            return direct

        reverse = cls._bias_col_m(epoch, f"BIAS_{obs_b}_{obs_a}", n)
        if reverse is not None:
            return -reverse

        return None

    @staticmethod
    def _bds_reference_codes(code_col: str):
        if code_col in {"C2I", "C6I", "C7I"}:
            return "C2I", "C6I"
        if code_col.startswith("C1"):
            suffix = code_col[-1]
            return code_col, f"C5{suffix}"
        if code_col.startswith("C5"):
            suffix = code_col[-1]
            return f"C1{suffix}", code_col
        if code_col.startswith("C7"):
            suffix = code_col[-1]
            if suffix == "Z":
                return "C1X", code_col
            return f"C1{suffix}", f"C5{suffix}"
        return None

    @classmethod
    def _bds_precise_clock_bias_correction_m(cls, epoch: pd.DataFrame, weighted_codes, n: int) -> np.ndarray:
        """Translate a BDS code combination to the precise-clock reference."""
        if not weighted_codes:
            return np.zeros(n, dtype='float64')

        osb_terms = []
        all_osb_available = True
        for code_col, coeff in weighted_codes:
            osb = cls._bias_col_m(epoch, f"OSB_{code_col}", n)
            if osb is None:
                all_osb_available = False
                break
            osb_terms.append(coeff * osb)
        if all_osb_available:
            return -np.sum(osb_terms, axis=0)

        ref = cls._bds_reference_codes(weighted_codes[0][0])
        if ref is None:
            return np.zeros(n, dtype='float64')

        anchor, ref_second = ref
        ref_sig1 = signal_spec("B1C" if anchor.startswith("C1") else "B1I").name
        ref_sig2 = signal_spec("B2a" if ref_second.startswith("C5") else "B3I").name
        _, k_ref_second = mode_ionosphere_free_coefficients(ref_sig1 + ref_sig2)

        ref_dcb = cls._dcb_between_m(epoch, anchor, ref_second, n)
        if ref_dcb is None:
            return np.zeros(n, dtype='float64')

        corr = k_ref_second * ref_dcb
        for code_col, coeff in weighted_codes:
            dcb = cls._dcb_between_m(epoch, anchor, code_col, n)
            if dcb is not None:
                corr = corr + coeff * dcb
        return corr

    def code_screening(self, x, satellites, code_obs, thr=1):
        """
        Screening of code observations for outliers.
        """
        return ppp_code_screening(x=x, satellites=satellites[:, :3], code_obs=code_obs, thr=thr)

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
        for idx in phase_residuals_outliers(
            sat_list=sat_list, phase_residuals_dict=phase_residuals_dict, num=num, thr=thr
        ):
            outlier_idx = idx
            if sys == 'E':
                outlier_idx += len_gps
            self.ekf.x[self.base_dim + outlier_idx] = 0.0
            self.ekf.P[
                self.base_dim + outlier_idx, self.base_dim + outlier_idx
            ] = 1e3
            reset_svs.append(sat_list[idx])
        return reset_svs

    def run_filter(self, ref=None, flh=None, trace_filter=False, reset_every=0):
        """
        Kalman filter PPP Float estimation
        :param ref: np.ndarray, reference position in ECEF
        :param flh: np.ndarray, reference position in BLH
        :param trace_filter:bool, whether to trace filter
        :param reset_every: int, number of epochs to reset filter
        :return: tuple(pd.Dataframe, pd.Dataframe, pd.Dataframe, float), solution, gps&gal residuals, convergence time
        """
        old_sats, epochs = self.init_filter()

        gps_data = self._prepare_obs()
        agps, bgps, gps_c1, gps_c2, gps_l1, gps_l2, sig1, sig2, pco_col1, pco_col2 = gps_data
        phase_residuals_trace = {}
        result = []
        result_gps = []
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
        if self.system == "C" and ar_cfg.enabled:
            warnings.warn(
                "PPP-AR is disabled for BeiDou single-system combined PPP until BDS phase-bias "
                "handling is validated.",
                RuntimeWarning,
                stacklevel=2,
            )
            ar_cfg = PPPARSettings(
                enabled=False,
                warmup_epochs=ar_cfg.warmup_epochs,
                min_ambiguities=ar_cfg.min_ambiguities,
                ratio_threshold=ar_cfg.ratio_threshold,
                constraint_sigma_cycles=ar_cfg.constraint_sigma_cycles,
            )
        ar_disabled_reason = None
        if bool(getattr(self.cfg, "pppar_enabled", False)) and not ar_cfg.enabled:
            ar_disabled_reason = "combined_if_ar_disabled"
            if self.system == "C" and bool(getattr(self.cfg, "pppar_combined_if_ar_enabled", False)):
                ar_disabled_reason = "bds_combined_ar_disabled"
        self.gps_obs = self.gps_obs.sort_values(by='sv')
        gps_epochs = {
            t: df for t, df in self.gps_obs.groupby(level=1, sort=False)
        }
        r_cache = {}
        xyz = None
        conv_time = None
        T0 = epochs[0]
        arc_age = {sv: 0 for sv in old_sats}
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)
                    arc_age = {sv: 0 for sv in old_sats}
                    reset_epoch = True
                    T0 = t
            gps_epoch = gps_epochs.get(t)
            if gps_epoch is None:
                continue

            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_sats = curr_gps_sats
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
            arc_age = {sv: arc_age.get(sv, 0) for sv in curr_sats}
            gps_clk, gps_tro, gps_ah_los, gps_dprel, gps_pco1, gps_pco2 = [
                np.asarray(gps_epoch.get(col, 0.0))
                for col in ['clk', 'tro', 'ah_los', 'dprel', pco_col1, pco_col2]
            ]
            gps_sat_pco_l1 = np.asarray(gps_epoch.get(f'sat_pco_los_{sig1}', 0.0))
            gps_sat_pco_l2 = np.asarray(gps_epoch.get(f'sat_pco_los_{sig2}', 0.0))
            gps_sat_pco_los = agps * gps_sat_pco_l1 - bgps * gps_sat_pco_l2

            los_gps = gps_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            gps_tides = gps_epoch[['dx', 'dy', 'dz']].to_numpy()
            gps_tides_los = np.sum(los_gps * gps_tides, axis=1)

            gps_p1_c1 = np.asarray(gps_epoch.get(f'OSB_{gps_c1}',0.0)) * 1e-09 * self.CLIGHT
            gps_p2_c2 = np.asarray(gps_epoch.get(f'OSB_{gps_c2}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl1_l1 = np.asarray(gps_epoch.get(f'OSB_{gps_l1}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl2_l2 = np.asarray(gps_epoch.get(f'OSB_{gps_l2}',0.0)) * 1e-09 * self.CLIGHT

            code_if_bias_corr = -(agps * gps_p1_c1 - bgps * gps_p2_c2)
            if self.system == "C" and getattr(self.cfg, "orbit_type", None) == "precise":
                code_if_bias_corr = self._bds_precise_clock_bias_correction_m(
                    epoch=gps_epoch,
                    weighted_codes=[(gps_c1, agps), (gps_c2, -bgps)],
                    n=len(gps_epoch),
                )
            phase_if_bias_corr = code_if_bias_corr - (agps * gps_pl1_l1 - bgps * gps_pl2_l2)

            GP3_code = (
                    (agps * (gps_epoch[f'{gps_c1}'].to_numpy() - gps_pco1) -
                     bgps * (gps_epoch[f'{gps_c2}'].to_numpy() - gps_pco2))
                    + code_if_bias_corr
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel - gps_tides_los
            )

            GL3_phase = (
                    (agps * (gps_epoch[f'{gps_l1}'].to_numpy() - gps_pco1) -
                     bgps * (gps_epoch[f'{gps_l2}'].to_numpy() - gps_pco2))
                    + phase_if_bias_corr
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel - gps_tides_los
                    - (self.LAMBDA_DICT[self.gps_mode] / (2 * np.pi)) * gps_epoch['phw'].to_numpy()
            )



            z = np.concatenate((GP3_code, GL3_phase))

            self.ekf.dim_z = len(z)
            sat_positions_gps = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            dist_gps = calculate_distance(sat_positions_gps[:, :3].copy(), self.ekf.x[:3].copy())

            prefit_gps = GP3_code - dist_gps
            prefit_gps_phase = GL3_phase - dist_gps
            phase_residuals_trace[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_phase)}
            # DOBRE, ale zamiast wielkosci rezyduoow filtr medianowy

            phase_reset_svs: set[str] = set()
            if num > 60:
                phase_reset_count = 0
                # Oblicz różnice prefitu z dwóch epok
                prefit_entries = []
                for idx, sv in enumerate(curr_gps_sats):
                    prev_residual = phase_residuals_trace[num - 1].get(sv)
                    current_residual = phase_residuals_trace[num].get(sv)

                    if prev_residual is not None and current_residual is not None:
                        prefit_entries.append((idx, current_residual - prev_residual))

                # Oblicz medianę różnic prefitu
                if prefit_entries:
                    prefit_diff = np.fromiter(
                        (entry[1] for entry in prefit_entries),
                        dtype=float,
                        count=len(prefit_entries),
                    )
                    median_prefit_diff = np.median(prefit_diff)
                else:
                    median_prefit_diff = 0.0

                threshold = 1  # or 3 std

                # Sprawdzanie, czy jakaś obserwacja wychodzi poza medianę (np. 2 razy większa różnica)
                for idx, residual in prefit_entries:

                    if np.abs(residual - median_prefit_diff) > threshold:
                        self.ekf.x[self.base_dim + idx] = 0.0  # Resetowanie stanu
                        self.ekf.P[
                            self.base_dim + idx, self.base_dim + idx] = 400  # Wysoka niepewność dla outliera
                        phase_reset_svs.add(curr_gps_sats[idx])
                        phase_reset_count += 1
            else:
                phase_reset_count = 0

            mask_gps = self.code_screening(x=self.ekf.x[:3].copy(),satellites=sat_positions_gps,code_obs=GP3_code,thr=3)
            mask_gps2 = np.concatenate((mask_gps, mask_gps))


            ev_gps = gps_epoch['ev'].to_numpy(copy=False)
            sin_el = np.sin(np.deg2rad(ev_gps))
            inv_sin2 = 1.0 / np.square(sin_el)
            sigma_code_gps = 9 + 0.0025 * inv_sin2
            sigma_phase_gps = 9e-4 + 0.0003 * inv_sin2
            R_vec = np.concatenate((sigma_code_gps, sigma_phase_gps))
            R_vec[~mask_gps2] = 1e12

            m = R_vec.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = R_vec
            self.ekf.R = R

            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=sat_positions_gps,
                                    Hx=self.Hx, hx_args=sat_positions_gps)
            arc_age = advance_arc_age(arc_age, curr_sats, phase_reset_svs)
            ar_age = arc_age_array(arc_age, curr_sats)
            innovation = np.asarray(self.ekf.y, dtype=float)
            postfit = z - self.Hx(self.ekf.x, sat_positions_gps)
            n_obs = len(gps_epoch)
            code_innov = postfit[:n_obs]
            phase_innov = postfit[n_obs:2 * n_obs]
            used_mask = np.asarray(mask_gps, dtype=bool)
            if np.any(used_mask):
                code_res_common = float(np.median(code_innov[used_mask]))
                phase_res_common = float(np.median(phase_innov[used_mask]))
                code_scatter = code_innov - code_res_common
                phase_scatter = phase_innov - phase_res_common
                code_res_rms = float(np.sqrt(np.mean(np.square(code_innov[used_mask]))))
                phase_res_rms = float(np.sqrt(np.mean(np.square(phase_innov[used_mask]))))
                code_res_scatter_rms = float(np.sqrt(np.mean(np.square(code_scatter[used_mask]))))
                phase_res_scatter_rms = float(np.sqrt(np.mean(np.square(phase_scatter[used_mask]))))
            else:
                code_res_common = np.nan
                phase_res_common = np.nan
                code_scatter = np.full_like(code_innov, np.nan)
                phase_scatter = np.full_like(phase_innov, np.nan)
                code_res_rms = np.nan
                phase_res_rms = np.nan
                code_res_scatter_rms = np.nan
                phase_res_scatter_rms = np.nan
            gps_ev = gps_epoch['ev'].to_numpy(copy=False) if 'ev' in gps_epoch.columns else np.full(
                len(curr_gps_sats), 45.0, dtype=float
            )
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                lam = self.LAMBDA_DICT[self.gps_mode]
                self.ekf.x, self.ekf.P, ar_diag = apply_conventional_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(curr_gps_sats),
                    n_gal=0,
                    gps_ev=gps_ev,
                    gal_ev=np.empty(0, dtype=float),
                    lambda_gps=lam,
                    lambda_gal=lam,
                    settings=ar_cfg,
                    gps_age=ar_age,
                )

            dtr = self.ekf.x[3]
            tro = self.ekf.x[4]

            # Convergence checking logic
            if not reset_epoch or reset_every==0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.05:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.05:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time

            xyz = self.ekf.x[:3].copy()  # Update position for the next epoch

            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx,flh=flh,degrees=True)).flatten()
            else:
                enu = np.array([0.0, 0.0, 0.0])
            df_epoch = pd.DataFrame(
                {'de': [enu[0]], 'dn': [enu[1]], 'du': [enu[2]], 'dtr': [dtr], 'ztd': [tro],
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]],
                 'system': [self.system], 'mode': [self.gps_mode],
                 'n_sats': [len(curr_gps_sats)],
                 'n_amb': [len(curr_gps_sats)],
                 'n_states': [len(self.ekf.x)],
                 'n_code_rejected': [int(np.count_nonzero(~used_mask))],
                 'n_phase_reset': [phase_reset_count],
                 'code_res_rms': [code_res_rms],
                 'phase_res_rms': [phase_res_rms],
                 'code_res_common': [code_res_common],
                 'phase_res_common': [phase_res_common],
                 'code_res_scatter_rms': [code_res_scatter_rms],
                 'phase_res_scatter_rms': [phase_res_scatter_rms],
                 'ar_fixed': [0 if ar_diag is None else ar_diag.fixed_ambiguities],
                 'ar_ratio': [np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min],
                 'ar_ok': [False if ar_diag is None else ar_diag.accepted],
                 'ar_min_age': [int(np.min(ar_age)) if ar_age.size else np.nan],
                 'ar_disabled_reason': [ar_disabled_reason]},
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
            gps_epoch['dtr'] = dtr
            gps_epoch['ztd'] = tro
            gps_epoch['L3'] = GL3_phase
            gps_epoch['P3'] = GP3_code
            gps_epoch['Nif'] = self.ekf.x[self.base_dim:self.base_dim + len(gps_epoch)].copy()
            gps_epoch['prefit_code_res'] = innovation[:len(gps_epoch)]
            gps_epoch['prefit_phase_res'] = innovation[len(gps_epoch):2 * len(gps_epoch)]
            gps_epoch['postfit_code_res'] = postfit[:len(gps_epoch)]
            gps_epoch['postfit_phase_res'] = postfit[len(gps_epoch):2 * len(gps_epoch)]
            gps_epoch['postfit_code_res_demeaned'] = code_scatter
            gps_epoch['postfit_phase_res_demeaned'] = phase_scatter
            gps_epoch['res'] = gps_epoch['postfit_phase_res']
            gps_epoch['code_bias_corr'] = code_if_bias_corr
            gps_epoch['phase_bias_corr'] = phase_if_bias_corr
            gps_epoch['p_if'] = np.diag(self.ekf.P[self.base_dim:self.base_dim + len(gps_epoch),
                                        self.base_dim:self.base_dim + len(gps_epoch)])

            result_gps.append(gps_epoch)

        df_result = pd.concat(result)
        df_obs_gps = pd.concat(result_gps)

        if trace_filter:
            if conv_time:
                ppp_trace(
                    trace_filter,
                    self.__class__.__name__,
                    "convergence",
                    threshold_m=0.05,
                    minutes=conv_time.total_seconds() / 60,
                    hours=conv_time.total_seconds() / 3600,
                )
                ct = conv_time.total_seconds() / 3600
            else:
                ct = None
        else:
            ct = None if conv_time is None else conv_time.total_seconds() / 3600
        return df_result, df_obs_gps,None, ct
