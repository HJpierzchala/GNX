import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter

from ..conversion import ecef_to_enu
from ..utils import calculate_distance
from ..configuration import PPPConfig

class PPPSingleFreqSingleGNSS:
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
        self.FREQ_DICT = {'L1' :1575.42e06 ,'E1' :1575.42e06,
                          'L2' :1227.60e06 ,'E5a' :1176.45e06,
                          'L5' :1176.450e06 ,'E5b' :1207.14e06}
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
        isb = 0.0
        # GPS prediction
        distances_gps = np.linalg.norm(gps_satellites[:, :3] - receiver, axis=1)
        h_code_gps = distances_gps + clock
        ambiguities_gps = x[self.base_dim:self.base_dim + num_gps]
        h_phase_gps = distances_gps + clock + ambiguities_gps

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
        common_sats = set(prev_sats) & set(curr_sats)
        for sat in common_sats:
            old_i = prev_sats.index(sat)
            new_i = curr_sats.index(sat)
            new_x[base_dim + new_i] = x_old[base_dim + old_i]

        # 2) Nowa macierz P
        new_P = np.zeros((new_dim, new_dim))
        new_P[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]

        for satA in common_sats:
            oA = base_dim + prev_sats.index(satA)
            nA = base_dim + curr_sats.index(satA)
            new_P[nA, :base_dim] = P_old[oA, :base_dim]
            new_P[:base_dim, nA] = P_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_sats.index(satB)
                nB = base_dim + curr_sats.index(satB)
                new_P[nA, nB] = P_old[oA, oB]
                new_P[nB, nA] = P_old[oB, oA]

        # Dla nowych satelit – ustaw duże niepewności
        new_sats = set(curr_sats) - set(prev_sats)
        for sat in new_sats:
            i_new = base_dim + curr_sats.index(sat)
            new_P[i_new, i_new] = self.cfg.p_amb

        # 3) Nowa macierz Q
        new_Q = np.zeros((new_dim, new_dim))
        new_Q[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]
        for satA in common_sats:
            oA = base_dim + prev_sats.index(satA)
            nA = base_dim + curr_sats.index(satA)
            new_Q[nA, :base_dim] = Q_old[oA, :base_dim]
            new_Q[:base_dim, nA] = Q_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_sats.index(satB)
                nB = base_dim + curr_sats.index(satB)
                new_Q[nA, nB] = Q_old[oA, oB]
                new_Q[nB, nA] = Q_old[oB, oA]
        # Nowym satelitom daj Q = 0
        for sat in new_sats:
            i_new = base_dim + curr_sats.index(sat)
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
        if self.mode in ['L1','E1']:
            gps_c1_col = [c for c in self.obs.columns if c.startswith('C1')][0]
            gps_l1_col = [c for c in self.obs.columns if c.startswith('L1')][0]
            data = (gps_c1_col, gps_l1_col)
        self.obs = self.obs.copy()
        if 'pco_los' not in self.obs.columns.tolist():
            self.obs.loc[:, 'pco_los'] = 0.0
        return data

    def code_screening(self, x, satellites, code_obs, thr=1):
        """
        Screening of code observations for outliers
        :param x: np.ndarray, reciever coordinates
        :param satellites: np.ndarray, satellite coordinates
        :param code_obs: np.ndarray, code observations
        :param thr: [float,int], threshold for median filter
        :return: np.ndarray, array of bools, outlier markers
        """
        dist = calculate_distance(satellites, x)
        prefit = code_obs-dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad >n_sat/2:
            mask = np.ones(n_sat,dtype=bool)
        return mask

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
        prefit_diff = []
        for sv in sat_list:
            prev_residual = phase_residuals_dict[num - 1].get(sv)
            current_residual = phase_residuals_dict[num].get(sv)
            if prev_residual is not None and current_residual is not None:
                prefit_diff.append(current_residual-prev_residual)
        median_prefit_diff = np.median(prefit_diff) if len(prefit_diff)>0 else 0

        for sv, residual in zip(sat_list, prefit_diff):
            if np.abs(residual-median_prefit_diff)>thr:
                outlier_idx = sat_list.index(sv)
                if sys =='E':
                    outlier_idx += len_gps
                self.ekf.x[self.base_dim + outlier_idx] = 0.0
                self.ekf.P[
                    self.base_dim + outlier_idx, self.base_dim + outlier_idx] = 1e3

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
        xyz = None
        T0 = epochs[0]
        conv_time = None
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)
                    reset_epoch = True
                    T0 = t
            gps_epoch = self.obs.loc[(slice(None), t), :].sort_values(by='sv')
            if len(gps_epoch)<4:
                print('Less than 4 satellites in sight! ')
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
            ev_gps = gps_epoch['ev'].to_numpy()
            sigma_code_gps = 0.5 + 0.0025 / np.sin(np.deg2rad(ev_gps)) ** 2
            sigma_phase_gps = 9e-4 + 0.00025 / np.sin(np.deg2rad(ev_gps)) ** 2
            R_vec = np.concatenate((sigma_code_gps, sigma_phase_gps))
            if num > 30:
                R_vec[~mask2] = 1e12

            self.ekf.R = np.diag(R_vec)

            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=(sat_positions_gps),
                                    Hx=self.Hx, hx_args=(sat_positions_gps))
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
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]]},
                index=pd.DatetimeIndex([t], name='time'))
            if trace_filter:
                print(df_epoch[['de', 'dn', 'du', 'dtr', 'isb', 'ztd']])
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
                print('Convergence time < 5 mm: ', conv_time.total_seconds() / 60,
                  ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")

            ct = conv_time.total_seconds() / 3600
        else:
            ct = None
        df_result['ct_min'] = ct
        return df_result, df_gps, None, None


class PPPDualFreqSingleGNSS:

    def __init__(self, config:PPPConfig,gps_obs,  gps_mode, ekf, tro=True, pos0=None,interval=0.5):

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
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = {'L1': 1575.42e06, 'E1': 1575.42e06,
                          'L2': 1227.60e06, 'E5a': 1176.45e06,
                          'L5': 1176.450e06, 'E5b': 1207.14e06}
        self.CLIGHT = 299792458
        self.base_dim = 5 if self.tro else 4
        self.pos0 = pos0
        self.interval = interval

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
        common_sats = set(prev_sats) & set(curr_sats)
        for sat in common_sats:
            old_i = prev_sats.index(sat)
            new_i = curr_sats.index(sat)
            new_x[base_dim + new_i] = x_old[base_dim + old_i]

        # 2) Nowa macierz P
        new_P = np.zeros((new_dim, new_dim))
        new_P[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]

        for satA in common_sats:
            oA = base_dim + prev_sats.index(satA)
            nA = base_dim + curr_sats.index(satA)
            new_P[nA, :base_dim] = P_old[oA, :base_dim]
            new_P[:base_dim, nA] = P_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_sats.index(satB)
                nB = base_dim + curr_sats.index(satB)
                new_P[nA, nB] = P_old[oA, oB]
                new_P[nB, nA] = P_old[oB, oA]

        # Dla nowych satelit – ustaw duże niepewności
        new_sats = set(curr_sats) - set(prev_sats)
        for sat in new_sats:
            i_new = base_dim + curr_sats.index(sat)
            new_P[i_new, i_new] = self.cfg.p_amb

        # 3) Nowa macierz Q
        new_Q = np.zeros((new_dim, new_dim))
        new_Q[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]
        for satA in common_sats:
            oA = base_dim + prev_sats.index(satA)
            nA = base_dim + curr_sats.index(satA)
            new_Q[nA, :base_dim] = Q_old[oA, :base_dim]
            new_Q[:base_dim, nA] = Q_old[:base_dim, oA]
            for satB in common_sats:
                oB = base_dim + prev_sats.index(satB)
                nB = base_dim + curr_sats.index(satB)
                new_Q[nA, nB] = Q_old[oA, oB]
                new_Q[nB, nA] = Q_old[oB, oA]
        # Nowym satelitom daj Q = 0
        for sat in new_sats:
            i_new = base_dim + curr_sats.index(sat)
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
        if self.gps_mode == 'E1E5a':
            GF1 = self.FREQ_DICT['L1']
            GF2 = self.FREQ_DICT['E5a']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.gps_obs.columns if c.startswith('C5')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.gps_obs.columns if c.startswith('L5')][0]
        if self.gps_mode == 'E1E5b':
            GF1 = self.FREQ_DICT['L1']
            GF2 = self.FREQ_DICT['E5b']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.gps_obs.columns if c.startswith('C7')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.gps_obs.columns if c.startswith('L7')][0]


        gps_data = (agps, bgps, gps_c1_col, gps_c2_col, gps_l1_col, gps_l2_col)

        self.gps_obs = self.gps_obs.copy()



        if 'pco_los_l1' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los_l1'] = 0.0
        if 'pco_los_l2' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los_l2'] = 0.0

        return gps_data

    def code_screening(self, x, satellites, code_obs, thr=1):
        """
        Screening of code observations for outliers
        :param x: np.ndarray, reciever coordinates
        :param satellites: np.ndarray, satellite coordinates
        :param code_obs: np.ndarray, code observations
        :param thr: [float,int], threshold for median filter
        :return: np.ndarray, array of bools, outlier markers
        """
        dist = calculate_distance(satellites[:,:3], x)
        prefit = code_obs-dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad >n_sat/2:
            mask = np.ones(n_sat,dtype=bool)
        return mask

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
        prefit_diff = []
        for sv in sat_list:
            prev_residual = phase_residuals_dict[num - 1].get(sv)
            current_residual = phase_residuals_dict[num].get(sv)
            if prev_residual is not None and current_residual is not None:
                prefit_diff.append(current_residual-prev_residual)
        median_prefit_diff = np.median(prefit_diff) if len(prefit_diff)>0 else 0

        for sv, residual in zip(sat_list, prefit_diff):
            if np.abs(residual-median_prefit_diff)>thr:
                outlier_idx = sat_list.index(sv)
                if sys =='E':
                    outlier_idx += len_gps
                self.ekf.x[self.base_dim + outlier_idx] = 0.0
                self.ekf.P[
                    self.base_dim + outlier_idx, self.base_dim + outlier_idx] = 1e3

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
        agps, bgps, gps_c1, gps_c2, gps_l1, gps_l2 = gps_data
        phase_residuals_trace = {}
        result = []
        result_gps = []
        xyz = None
        conv_time = None
        T0 = epochs[0]
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)
                    reset_epoch = True
                    T0 = t
            gps_epoch = self.gps_obs.loc[(slice(None), t), :].sort_values(by='sv')

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
            gps_clk, gps_tro, gps_ah_los, gps_sat_pco_los, gps_dprel, gps_pco1, gps_pco2 = [np.asarray(gps_epoch.get(col, 0.0))
                                                                                            for col in
                                                                                            ['clk', 'tro', 'ah_los',
                                                                                             'sat_pco_los_L1',
                                                                                             'dprel', 'pco_los_l1',
                                                                                             'pco_los_l2']]

            los_gps = gps_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            gps_tides = gps_epoch[['dx', 'dy', 'dz']].to_numpy()
            gps_tides_los = np.sum(los_gps * gps_tides, axis=1)

            gps_p1_c1 = np.asarray(gps_epoch.get(f'OSB_{gps_c1}',0.0)) * 1e-09 * self.CLIGHT
            gps_p2_c2 = np.asarray(gps_epoch.get(f'OSB_{gps_c2}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl1_l1 = np.asarray(gps_epoch.get(f'OSB_{gps_l1}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl2_l2 = np.asarray(gps_epoch.get(f'OSB_{gps_l2}',0.0)) * 1e-09 * self.CLIGHT


            GP3_code = (
                    (agps * (gps_epoch[f'{gps_c1}'].to_numpy() - gps_pco1 - gps_p1_c1) -
                     bgps * (gps_epoch[f'{gps_c2}'].to_numpy() - gps_pco2 - gps_p2_c2))
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel - gps_tides_los
            )

            GL3_phase = (
                    (agps * (gps_epoch[f'{gps_l1}'].to_numpy() - gps_pco1 - gps_pl1_l1 - gps_p1_c1) -
                     bgps * (gps_epoch[f'{gps_l2}'].to_numpy() - gps_pco2 - gps_pl2_l2-gps_p2_c2))
                    + gps_clk * self.CLIGHT - gps_tro - gps_ah_los + gps_sat_pco_los - gps_dprel - gps_tides_los
                    - (0.106 / (2 * np.pi)) * gps_epoch['phw'].to_numpy()
            )



            z = np.concatenate((GP3_code, GL3_phase))

            self.ekf.dim_z = len(z)
            sat_positions_gps = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            dist_gps = calculate_distance(sat_positions_gps[:, :3].copy(), self.ekf.x[:3].copy())

            prefit_gps = GP3_code - dist_gps
            prefit_gps_phase = GL3_phase - dist_gps
            phase_residuals_trace[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_phase)}
            # DOBRE, ale zamiast wielkosci rezyduoow filtr medianowy

            if num > 60:
                # Oblicz różnice prefitu z dwóch epok
                prefit_diff = []
                for sv in curr_gps_sats:
                    prev_residual = phase_residuals_trace[num - 1].get(sv)
                    current_residual = phase_residuals_trace[num].get(sv)

                    if prev_residual is not None and current_residual is not None:
                        prefit_diff.append(current_residual - prev_residual)

                # Oblicz medianę różnic prefitu
                median_prefit_diff = np.median(prefit_diff) if len(prefit_diff) > 0 else 0

                threshold = 1  # or 3 std

                # Sprawdzanie, czy jakaś obserwacja wychodzi poza medianę (np. 2 razy większa różnica)
                for sv, residual in zip(curr_gps_sats, prefit_diff):

                    # print(f'SV: {sv} Residual: {residual}  THR: {threshold}')
                    if np.abs(residual - median_prefit_diff) > threshold:
                        # print(f'Outlier detected for {sv}: Residual difference {residual}, threshold {threshold}')
                        outlier_idx = curr_gps_sats.index(sv)
                        self.ekf.x[self.base_dim + outlier_idx] = 0.0  # Resetowanie stanu
                        self.ekf.P[
                            self.base_dim + outlier_idx, self.base_dim + outlier_idx] = 400  # Wysoka niepewność dla outliera

            mask_gps = self.code_screening(x=self.ekf.x[:3].copy(),satellites=sat_positions_gps,code_obs=GP3_code,thr=3)
            mask_gps2 = np.concatenate((mask_gps, mask_gps))


            ev_gps = gps_epoch['ev'].to_numpy()
            sigma_code_gps = 9 + 0.0025 / np.sin(np.deg2rad(ev_gps)) ** 2
            sigma_phase_gps = 9e-4 + 0.0003 / np.sin(np.deg2rad(ev_gps)) ** 2
            R_vec = np.concatenate((sigma_code_gps, sigma_phase_gps))
            R_vec[~mask_gps2] = 1e12

            self.ekf.R = np.diag(R_vec)

            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=sat_positions_gps,
                                    Hx=self.Hx, hx_args=sat_positions_gps)

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
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]]},
                index=pd.DatetimeIndex([t], name='time'))
            if trace_filter:
                print(df_epoch)
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

        df_result = pd.concat(result)
        df_obs_gps = pd.concat(result_gps)

        if trace_filter:
            if conv_time:
                print('Convergence time < 5 cm: ', conv_time.total_seconds() / 60,
                      ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")
                ct = conv_time.total_seconds() / 3600
            else:
                ct = None
        else:
            ct = conv_time.total_seconds() / 3600
        return df_result, df_obs_gps,None, ct

