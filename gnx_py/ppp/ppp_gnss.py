import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter

from ..conversion import ecef_to_enu
from ..utils import calculate_distance


class PPPSingleFreqMultiGNSS:

    def __init__(self, gps_obs, gal_obs, gps_mode, gal_mode, ekf, tro=False, pos0=None,interval=0.5):

        """

        :param gps_obs:
        :param gal_obs:
        :param gps_mode:
        :param gal_mode:
        :param ekf: ExtendedKalmanFilter
        :param tro:
        """
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

        :param x:
        :param gps_satellites:
        :param gal_satellites:
        :return:
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

        :param x:
        :param gps_satellites:
        :param gal_satellites:
        :return:
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

        :param x_old:
        :param P_old:
        :param Q_old:
        :param prev_sats:
        :param curr_sats:
        :param base_dim:
        :return:
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
            new_P[i_new, i_new] = 400.0

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
            new_Q[i_new, i_new] = 0.0

        return new_x, new_P, new_Q

    def init_filter(self, p_crd=10.0, p_dt=9e9, p_amb=1e3, p_tro=0.25, q_crd=0.0, q_dt=9e9, q_tro=0.00025):

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
        self.ekf.Q[:3, :3] = q_crd
        self.ekf.Q[3, 3] = q_dt

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, p_crd), np.full(N0, p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = p_dt
        self.ekf.P[4, 4] = p_dt

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

    def reset_filter(self, epoch,p_crd=10.0, p_dt=9e9, p_amb=1e3, p_tro=0.25, q_crd=0.0, q_dt=9e9, q_tro=0.00025):
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
        self.ekf.Q[:3, :3] = q_crd
        self.ekf.Q[3, 3] = q_dt
        self.ekf.Q[5, 5] = q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, p_crd), np.full(N0, p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = p_dt
        self.ekf.P[4, 4] = p_dt
        self.ekf.P[5, 5] = p_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0 + gal0

        return old_sats

    def run_filter(self, ref=None, flh=None, p_crd=10.0, p_dt=9e9, p_amb=1e3, p_tro=0.25, q_crd=0.0, q_dt=9e9,
                   q_tro=0.00025, add_dcb=True,trace_filter=False,reset_every=0):
        old_sats, epochs = self.init_filter(
            p_crd=p_crd,
            p_dt=p_dt,
            p_amb=p_amb,
            p_tro=p_tro,
            q_crd=q_crd,
            q_dt=q_dt,
            q_tro=q_tro,
        )

        gps_data, gal_data = self._prepare_obs()
        gps_c1, gps_l1 = gps_data
        gal_c1, gal_l1 = gal_data
        phase_residuals_trace = {}
        phase_residuals_trace_gal = {}
        result = []
        result_gal=[]
        result_gps=[]
        xyz=None
        T0 = epochs[0]
        conv_time=None
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)

                    reset_epoch = True
                    T0 = t
            print(t)
            gps_epoch = self.gps_obs.loc[(slice(None), t), :].sort_values(by='sv')


            gal_epoch = self.gal_obs.loc[(slice(None), t), :].sort_values(by='sv')

            if len(gal_epoch) + len(gps_epoch)<4:
                print('Less than 4 satellites in sight! ')
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

            if add_dcb:
                try:

                    gps_p1_c1 = gps_epoch[f'OSB_{gps_c1}'].to_numpy() * 1e-09 * self.CLIGHT
                    gps_pl1_l1 = gps_epoch[f'OSB_{gps_l1}'].to_numpy() * 1e-09 * self.CLIGHT

                except Exception as e:
                    gps_p1_c1 = 0.0
                    gps_pl1_l1 = 0.0

                try:
                    gal_p1_c1 = gal_epoch[f'OSB_{gal_c1}'].to_numpy() * 1e-09 * self.CLIGHT
                    gal_pl1_l1 = gal_epoch[f'OSB_{gal_l1}'].to_numpy() * 1e-09 * self.CLIGHT

                except Exception as e:
                    gal_p1_c1 = 0.0
                    gal_pl1_l1 = 0.0
            else:
                gps_p1_c1 = 0.0
                gps_pl1_l1 = 0.0
                gal_p1_c1 = 0.0
                gal_pl1_l1 = 0.0
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
            if num >= 60:
                self.phase_residuals_screening(sat_list=curr_gps_sats,phase_residuals_dict=phase_residuals_trace,num=num)
                self.phase_residuals_screening(sat_list=curr_gal_sats, phase_residuals_dict=phase_residuals_trace_gal,
                                           num=num, sys='E', len_gps=len(curr_gps_sats))

            mask_gps = self.code_screening(x=self.ekf.x[:3].copy(),satellites=sat_positions_gps,code_obs=GP3_code,thr=10)
            mask_gps2 = np.concatenate((mask_gps, mask_gps))
            mask_gal = self.code_screening(x=self.ekf.x[:3].copy(), satellites=sat_positions_gal, code_obs=EP3_code,
                                           thr=10)
            mask_gal2 = np.concatenate((mask_gal, mask_gal))
            mask2 = np.concatenate((mask_gps2, mask_gal2))



            ev_gps = gps_epoch['ev'].to_numpy()
            ev_gal = gal_epoch['ev'].to_numpy()
            sigma_code_gps = 1 + 0.00025 / np.sin(np.deg2rad(ev_gps)) ** 2
            sigma_phase_gps = 1e-4 + 0.00025 / np.sin(np.deg2rad(ev_gps)) ** 2
            sigma_code_gal = 1 + 0.00025 / np.sin(np.deg2rad(ev_gal)) ** 2
            sigma_phase_gal = 1e-4 + 0.00025 / np.sin(np.deg2rad(ev_gal)) ** 2
            R_vec = np.concatenate((sigma_code_gps, sigma_phase_gps, sigma_code_gal, sigma_phase_gal))
            R_vec[~mask2] = 1e12

            self.ekf.R = np.diag(R_vec)

            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=(sat_positions_gps, sat_positions_gal),
                                    Hx=self.Hx, hx_args=(sat_positions_gps, sat_positions_gal))

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
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]]},
                index=pd.DatetimeIndex([t], name='time'))
            if trace_filter:
                print(df_epoch[['de', 'dn', 'du']])
                print('\n')
                print(df_epoch[['dtr', 'isb', 'ztd']])
                print('===' * 30, '\n')
            result.append(df_epoch)
        df_result = pd.concat(result)
        df_gps = pd.concat(result_gps)
        df_gal = pd.concat(result_gal)
        if conv_time:
            ct = conv_time.total_seconds() / 3600
            if trace_filter:
                print('Convergence time < 5 mm: ', conv_time.total_seconds() / 60,
                  ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")
        else:
            ct=None
        df_result['ct_min'] = ct
        return df_result, df_gps, df_gal, ct


class PPPDualFreqMultiGNSS:

    def __init__(self, gps_obs, gal_obs, gps_mode, gal_mode, ekf, tro=True, pos0=None,interval=0.5):

        """

        :param gps_obs:
        :param gal_obs:
        :param gps_mode:
        :param gal_mode:
        :param ekf: ExtendedKalmanFilter
        :param tro:
        """
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

        :param x:
        :param gps_satellites:
        :param gal_satellites:
        :return:
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

        :param x:
        :param gps_satellites:
        :param gal_satellites:
        :return:
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

        :param x_old:
        :param P_old:
        :param Q_old:
        :param prev_sats:
        :param curr_sats:
        :param base_dim:
        :return:
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
            new_P[i_new, i_new] = 400.0

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
            new_Q[i_new, i_new] = 0.0

        return new_x, new_P, new_Q

    def init_filter(self, p_crd=10.0, p_dt=9e9, p_amb=400, p_tro=0.25, q_crd=0.0, q_dt=9e9, q_tro=0.00025):

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
        self.ekf.Q[:3, :3] = q_crd
        self.ekf.Q[3, 3] = q_dt
        self.ekf.Q[5, 5] = q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, p_crd), np.full(N0, p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = p_dt
        self.ekf.P[4, 4] = p_dt
        self.ekf.P[5, 5] = p_tro

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
        dist = calculate_distance(satellites, x)
        prefit = code_obs-dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad >n_sat/2:
            mask = np.ones(n_sat,dtype=bool)
        return mask

    def phase_residuals_screening(self, sat_list, phase_residuals_dict,num,thr=1,sys='G',len_gps=None):
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
                    self.base_dim + outlier_idx, self.base_dim + outlier_idx] = 400

    def reset_filter(self, epoch,p_crd=10.0, p_dt=9e9, p_amb=400, p_tro=0.25, q_crd=0.0, q_dt=9e9, q_tro=0.00025):
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
        self.ekf.Q[:3, :3] = q_crd
        self.ekf.Q[3, 3] = q_dt
        self.ekf.Q[5, 5] = q_tro

        # P
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, p_crd), np.full(N0, p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = 1e12
        self.ekf.P[3, 3] = p_dt
        self.ekf.P[4, 4] = p_dt
        self.ekf.P[5, 5] = p_tro

        # F
        self.ekf.F = np.eye(dimx)
        self.ekf.F[3, 3] = 0.0
        old_sats = gps0 + gal0

        return old_sats




    def run_filter(self, ref=None, flh=None, p_crd=10.0, p_dt=9e9, p_amb=400, p_tro=0.25, q_crd=0.0, q_dt=9e9,
                   q_tro=0.00025, add_dcb=True,trace_filter=False, reset_every = 180):
        old_sats, epochs = self.init_filter(
            p_crd=p_crd,
            p_dt=p_dt,
            p_amb=p_amb,
            p_tro=p_tro,
            q_crd=q_crd,
            q_dt=q_dt,
            q_tro=q_tro,
        )

        gps_data, gal_data = self._prepare_obs()
        agps, bgps, gps_c1, gps_c2, gps_l1, gps_l2 = gps_data
        agal, bgal, gal_c1, gal_c2, gal_l1, gal_l2 = gal_data
        phase_residuals_trace = {}
        phase_residuals_trace_gal = {}
        result = []
        result_gps = []
        result_gal =[]
        xyz = None
        conv_time = None
        T0 = epochs[0]
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num !=0):
                    old_sats = self.reset_filter(epoch=t)
                    reset_epoch =True
                    T0 = t

            gps_epoch = self.gps_obs.loc[(slice(None), t), :].sort_values(by='sv')
            gal_epoch = self.gal_obs.loc[(slice(None), t), :].sort_values(by='sv')

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
            gps_sta_pco1, gps_sta_pco2 = self.gps_mode[:2], self.gps_mode[2:]
            gal_sta_pco1, gal_sta_pco2 = self.gal_mode[:2], self.gal_mode[2:]
            gps_clk, gps_tro, gps_ah_los,  gps_dprel, gps_pco1, gps_pco2, sat_pco_L1, sat_pco_L2 = [np.asarray(gps_epoch.get(col,0.0))
                                                                                            for col in
                                                                                            ['clk', 'tro', 'ah_los',
                                                                                             'dprel', 'pco_los_l1',
                                                                                             'pco_los_l2',f'sat_pco_los_{gps_sta_pco1}',
                                                                                             f'sat_pco_los_{gps_sta_pco2}']]
            gal_clk, gal_tro, gal_ah_los,  gal_dprel, gal_pco1, gal_pco2,sat_pco_E1, sat_pco_E5a  = [np.asarray(gal_epoch.get(col,0.0))
                                                                                            for col in
                                                                                            ['clk', 'tro', 'ah_los',
                                                                                             'dprel', 'pco_los_l1',
                                                                                             'pco_los_l5',f'sat_pco_los_{gal_sta_pco1}',
                                                                                        f'sat_pco_los_{gal_sta_pco2}']]

            los_gps = gps_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            gps_tides = gps_epoch[['dx', 'dy', 'dz']].to_numpy()
            gps_tides_los = np.sum(los_gps * gps_tides, axis=1)
            gps_epoch['tides_los'] = gps_tides_los


            los_gal = gal_epoch[['LOS1', 'LOS2', 'LOS3']].to_numpy()
            gal_tides = gal_epoch[['dx', 'dy', 'dz']].to_numpy()
            gal_tides_los = np.sum(los_gal * gal_tides, axis=1)
            gal_epoch['tides_los'] = gal_tides_los

            # gal_tides_los += np.sum(los_gal * dR_gal, axis=1)

            if add_dcb:
                try:

                    gps_p1_c1 = gps_epoch[f'OSB_{gps_c1}'].to_numpy() * 1e-09 * self.CLIGHT
                    gps_p2_c2 = gps_epoch[f'OSB_{gps_c2}'].to_numpy() * 1e-09 * self.CLIGHT
                    gps_pl1_l1 = gps_epoch[f'OSB_{gps_l1}'].to_numpy() * 1e-09 * self.CLIGHT
                    gps_pl2_l2 = gps_epoch[f'OSB_{gps_l2}'].to_numpy() * 1e-09 * self.CLIGHT

                except Exception as e:
                    gps_p1_c1 = 0.0
                    gps_p2_c2 = 0.0
                    gps_pl1_l1 = 0.0
                    gps_pl2_l2 = 0.0

                try:
                    gal_p1_c1 = gal_epoch[f'OSB_{gal_c1}'].to_numpy() * 1e-09 * self.CLIGHT
                    gal_p2_c2 = gal_epoch[f'OSB_{gal_c2}'].to_numpy() * 1e-09 * self.CLIGHT
                    gal_pl1_l1 = gal_epoch[f'OSB_{gal_l1}'].to_numpy() * 1e-09 * self.CLIGHT
                    gal_pl2_l2 = gal_epoch[f'OSB_{gal_l2}'].to_numpy() * 1e-09 * self.CLIGHT

                except Exception as e:
                    gal_p1_c1 = 0.0
                    gal_p2_c2 = 0.0
                    gal_pl1_l1 = 0.0
                    gal_pl2_l2 = 0.0
            else:
                gps_p1_c1 = 0.0
                gps_p2_c2 = 0.0
                gps_pl1_l1 = 0.0
                gps_pl2_l2 = 0.0
                gal_p1_c1 = 0.0
                gal_p2_c2 = 0.0
                gal_pl1_l1 = 0.0
                gal_pl2_l2 = 0.0


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
            if num >= 60:
                self.phase_residuals_screening(sat_list=curr_gps_sats,phase_residuals_dict=phase_residuals_trace,num=num)
            mask_gps2 = np.concatenate((mask_gps, mask_gps))

            mask_gal = self.code_screening(x=self.ekf.x[:3].copy(),satellites=sat_positions_gal[:,:3],code_obs=EP3_code)
            if num >=60:
                self.phase_residuals_screening(sat_list=curr_gal_sats,phase_residuals_dict=phase_residuals_trace_gal,num=num,sys='E',len_gps=len(curr_gps_sats))
            mask_gal2 = np.concatenate((mask_gal, mask_gal))

            # # ostateczna maska na wszystkie pomiary
            mask2 = np.concatenate((mask_gps2, mask_gal2))

            ev_gps = gps_epoch['ev'].to_numpy()
            ev_gal = gal_epoch['ev'].to_numpy()
            sigma_code_gps = 1 + 0.0025 / np.sin(np.deg2rad(ev_gps)) ** 2
            sigma_phase_gps = 1e-4 + 0.0003 / np.sin(np.deg2rad(ev_gps)) ** 2
            sigma_code_gal = 1 + 0.0025 / np.sin(np.deg2rad(ev_gal)) ** 2
            sigma_phase_gal = 1e-4 + 0.0003 / np.sin(np.deg2rad(ev_gal)) ** 2
            r_vec = np.concatenate((sigma_code_gps, sigma_phase_gps, sigma_code_gal, sigma_phase_gal))
            r_vec[~mask2] = 1e12

            self.ekf.R = np.diag(r_vec)
            self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=(sat_positions_gps, sat_positions_gal),
                                    Hx=self.Hx, hx_args=(sat_positions_gps, sat_positions_gal))

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
            mxyz = np.diag(self.ekf.P[:3,:3])
            df_epoch = pd.DataFrame(
                {'de': [enu[0]], 'dn': [enu[1]], 'du': [enu[2]], 'dtr': [dtr], 'isb': [isb], 'ztd': [tro],
                 'x': [self.ekf.x[0]], 'y': [self.ekf.x[1]], 'z': [self.ekf.x[2]] ,'mx':mxyz[0], 'my':mxyz[1],'mz':mxyz[2]},
                index=pd.DatetimeIndex([t], name='time'))
            if trace_filter:
                print(self.ekf.y)
                print(df_epoch[['de', 'dn', 'du']])
                print('\n')
                print(df_epoch[['dtr', 'isb', 'ztd']])
                print('===' * 30, '\n')
            result.append(df_epoch)

            gps_epoch['dtr'] = dtr
            gps_epoch['ztd'] = tro
            gps_epoch['L3'] = GL3_phase
            gps_epoch['P3'] = GP3_code
            gps_epoch['Nif'] = self.ekf.x[self.base_dim:self.base_dim+len(gps_epoch)].copy()
            gps_epoch['v'] = self.ekf.y[len(gps_epoch):2*len(gps_epoch)]
            gps_epoch['vc'] = self.ekf.y[:len(gps_epoch)]
            gps_epoch['p_if'] = np.diag(self.ekf.P[self.base_dim:self.base_dim+len(gps_epoch),
                            self.base_dim:self.base_dim+len(gps_epoch)])

            gal_epoch['dtr'] = dtr
            gal_epoch['isb'] = isb
            gal_epoch['L3'] = EL3_phase
            gal_epoch['P3'] = EP3_code
            gal_epoch['Nif'] = self.ekf.x[self.base_dim+len(gps_epoch): ].copy()
            gal_epoch['p_if'] = np.diag(self.ekf.P[self.base_dim + len(gps_epoch):,
                                        self.base_dim + len(gps_epoch):])
            gal_epoch['v'] = self.ekf.y[2*len(gps_epoch)+len(gal_epoch):]
            gal_epoch['vc'] = self.ekf.y[2 * len(gps_epoch): 2 * len(gps_epoch) + len(gal_epoch)]


            result_gps.append(gps_epoch)
            result_gal.append(gal_epoch)
        df_result = pd.concat(result)
        df_obs_gps = pd.concat(result_gps)
        df_obs_gal = pd.concat(result_gal)

        if conv_time is not None:
            if trace_filter:
                print('Convergence time < 5 mm: ', conv_time.total_seconds() / 60,
                  ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")

            ct = conv_time.total_seconds() / 3600
        else:
            ct = None
        df_result['ct_min'] = ct
        return df_result,df_obs_gps, df_obs_gal, ct



