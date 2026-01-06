import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter

from ..conversion import ecef_to_enu
from ..utils import calculate_distance
from ..configuration import PPPConfig

class PPPUducSFMultiGNSS:
    def __init__(self, config:PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, interval=0.5):
        self.cfg=config
        self.gps_obs = gps_obs
        self.gps_mode = gps_mode
        self.gal_obs = gal_obs
        self.gal_mode = gal_mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = {'L1': 1575.42e06, 'E1': 1575.42e06,
                          'L2': 1227.60e06, 'E5a': 1176.45e06,
                          'L5': 1176.450e06, 'E5b': 1207.14e06}
        self.LAMBDA_DICT = {}
        self.CLIGHT = 299792458
        self.base_dim = 6 if self.tro else 5
        self.pos0 = pos0
        self.interval=interval

    def Hjacobian(self, x, gps_satellites, gal_satellites):
        C = self.CLIGHT
        # Częstotliwości GPS
        F1_GPS = self.FREQ_DICT[self.gps_mode]
        L1 = C / F1_GPS
        MU1 = 1.0

        # Częstotliwości Galileo
        F1_GAL = self.FREQ_DICT[self.gal_mode]
        E1 = C / F1_GAL
        MU1_GAL = 1.0

        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        rec_xyz = x[:3].copy()

        # Jednostkowe wektory do satelitów GPS
        rho_gps = gps_satellites[:, :3] - rec_xyz
        e_gps = rho_gps / np.linalg.norm(rho_gps, axis=1)[:, None]
        m_wet_gps = gps_satellites[:, 3]

        # Jednostkowe wektory do satelitów Galileo
        rho_gal = gal_satellites[:, :3] - rec_xyz
        e_gal = rho_gal / np.linalg.norm(rho_gal, axis=1)[:, None]
        m_wet_gal = gal_satellites[:, 3]

        # Parametry wspólne: XYZ, clk_gps, ISB_galileo, ZTD
        COL_X, COL_Y, COL_Z = 0, 1, 2
        COL_CLK = 3
        COL_ISB = 4
        COL_ZTD = 5

        base_dim = 6  # wspólne parametry (XYZ + clk_gps + isb + ztd)

        # 2 równania na każdy satelita: P1, L1, P2, L2
        H = np.zeros((2 * (n_gps + n_gal), base_dim + 2 * (n_gps + n_gal)))

        # ---------------- GPS ----------------
        for s in range(n_gps):
            row = 2 * s
            ex, ey, ez = -e_gps[s]
            mw = m_wet_gps[s]

            for i in range(2):
                H[row + i, [COL_X, COL_Y, COL_Z]] = [ex, ey, ez]
                H[row + i, COL_CLK] = 1.0  # wspólny zegar GPS
                H[row + i, COL_ZTD] = mw

            col_iono = base_dim + 2 * s
            col_N1 = col_iono + 1

            # Efekt IONO
            H[row + 0, col_iono] = +MU1  # P1
            H[row + 1, col_iono] = -MU1  # L1


            # Ambiguity
            H[row + 1, col_N1] = L1

        # ---------------- Galileo ----------------
        for s in range(n_gal):
            row = 2 * (n_gps + s)
            ex, ey, ez = -e_gal[s]
            mw = m_wet_gal[s]

            for i in range(2):
                H[row + i, [COL_X, COL_Y, COL_Z]] = [ex, ey, ez]
                H[row + i, COL_CLK] = 1.0  # współdzielony zegar
                H[row + i, COL_ISB] = 1.0  # inter-sys bias dla Galileo
                H[row + i, COL_ZTD] = mw

            col_iono = base_dim + 2 * (n_gps + s)
            col_N1 = col_iono + 1

            # Efekt IONO (na Galileo)
            H[row + 0, col_iono] = +MU1_GAL  # E1 code
            H[row + 1, col_iono] = -MU1_GAL  # E1 phase


            # Ambiguity (Galileo)
            H[row + 1, col_N1] = E1

        return H

    def Hx(self, x: np.ndarray, gps_satellites: np.ndarray, gal_satellites: np.ndarray) -> np.ndarray:
        x_state = x.copy()
        C = self.CLIGHT
        # GPS
        F1_GPS = self.FREQ_DICT[self.gps_mode]
        L1 = C / F1_GPS
        MU1 = 1.0

        # Galileo
        F1_GAL = self.FREQ_DICT[self.gal_mode]
        E1 = C / F1_GAL
        MU1_GAL = 1.0

        xr, yr, zr = x_state[0:3]
        clk = x_state[3]
        isb = x_state[4]  # inter-sys bias (dla Galileo)
        zwd = x_state[5]

        base_dim = self.base_dim
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]

        # Z góry rozmiar wektora predykcji: 4 obserwacje na każdy satelita
        z_hat = np.empty(2 * (n_gps + n_gal))

        # --- GPS ---
        sat_xyz = gps_satellites[:, :3]
        m_wet = gps_satellites[:, 3]
        rho_vec = sat_xyz - np.array([xr, yr, zr])
        rho = np.linalg.norm(rho_vec, axis=1)

        for s in range(n_gps):
            i = base_dim + 2 * s
            I_s = x_state[i]
            N1_s = x_state[i + 1]

            geom = rho[s]
            mw = m_wet[s]

            z_hat[2 * s + 0] = geom + clk + mw * zwd + MU1 * I_s  # P1
            z_hat[2 * s + 1] = geom + clk + mw * zwd - MU1 * I_s + L1 * N1_s  # L1

        # --- Galileo ---
        sat_xyz = gal_satellites[:, :3]
        m_wet = gal_satellites[:, 3]
        rho_vec = sat_xyz - np.array([xr, yr, zr])
        rho = np.linalg.norm(rho_vec, axis=1)

        for s in range(n_gal):
            i = base_dim + 2 * (n_gps + s)
            I_s = x_state[i]
            N1_s = x_state[i + 1]

            geom = rho[s]
            mw = m_wet[s]

            row = 2 * (n_gps + s)

            z_hat[row + 0] = geom + clk + isb + mw * zwd + MU1_GAL * I_s  # E1 code
            z_hat[row + 1] = geom + clk + isb + mw * zwd - MU1_GAL * I_s + E1 * N1_s  # E1 phase

        return z_hat

    def rebuild_state(self, x_old, P_old, Q_old,
                      prev_gps, prev_gal, curr_gps, curr_gal,
                      P4_gps_dict: dict, P4_gal_dict: dict):
        """
        Aktualizuje stan filtra Kalman w sytuacji, gdy zmienia się widoczność satelitów GPS i Galileo.
        Dla nowych satelitów inicjalizuje I_s na podstawie obserwacji P4.

        Parametry:
        - prev_gps / prev_gal: listy satelitów poprzedniej epoki
        - curr_gps / curr_gal: listy satelitów bieżącej epoki
        - P4_gps_dict: słownik {sv: P4} dla GPS
        - P4_gal_dict: słownik {sv: P4} dla Galileo

        Zakładamy:
        - self.base_dim = 6 (X, Y, Z, clk_GPS, ISB_GAL, ZTD)
        """

        base_dim = self.base_dim
        n_prev = len(prev_gps) + len(prev_gal)
        n_curr = len(curr_gps) + len(curr_gal)

        old_dim = base_dim + 2 * n_prev
        new_dim = base_dim + 2 * n_curr

        x_new = np.zeros(new_dim)
        x_new[:base_dim] = x_old[:base_dim]

        P_new = np.zeros((new_dim, new_dim))
        P_new[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]

        Q_new = np.zeros((new_dim, new_dim))
        Q_new[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]

        # Oznaczenia systemowe, np. G:G01, E:E11
        def tag(system: str, sv: str) -> str:
            return f"{system}:{sv}"

        prev_all = [tag("G", sv) for sv in prev_gps] + [tag("E", sv) for sv in prev_gal]
        curr_all = [tag("G", sv) for sv in curr_gps] + [tag("E", sv) for sv in curr_gal]

        common = set(prev_all) & set(curr_all)

        # --- przenoszenie wspólnych satelitów ---
        for prn in common:
            i_old = prev_all.index(prn)
            i_new = curr_all.index(prn)
            for k in range(2):
                xo = base_dim + 2 * i_old + k
                xn = base_dim + 2 * i_new + k
                x_new[xn] = x_old[xo]

                P_new[xn, :base_dim] = P_old[xo, :base_dim]
                P_new[:base_dim, xn] = P_old[:base_dim, xo]
                Q_new[xn, :base_dim] = Q_old[xo, :base_dim]
                Q_new[:base_dim, xn] = Q_old[:base_dim, xo]

                for prn2 in common:
                    j_old = prev_all.index(prn2)
                    j_new = curr_all.index(prn2)
                    for l in range(2):
                        yo = base_dim + 2 * j_old + l
                        yn = base_dim + 2 * j_new + l
                        P_new[xn, yn] = P_old[xo, yo]
                        Q_new[xn, yn] = Q_old[xo, yo]

        # --- nowe satelity: zainicjalizuj I_s z P4 ---
        new_only = set(curr_all) - set(prev_all)
        for prn in new_only:
            j = curr_all.index(prn)
            idx_I = base_dim + 2 * j
            idx_N1 = idx_I + 1

            # Domyślny init
            x_new[idx_I] = 0.0

            # Próba inicjalizacji z P4
            if prn.startswith("G:"):
                sv = prn[2:]
                if sv in P4_gps_dict:
                    x_new[idx_I] = P4_gps_dict[sv]
            elif prn.startswith("E:"):
                sv = prn[2:]
                if sv in P4_gal_dict:
                    x_new[idx_I] = P4_gal_dict[sv]

            # Ustaw P, Q
            P_new[idx_I, idx_I] = self.cfg.p_iono
            Q_new[idx_I, idx_I] = self.cfg.q_iono*(self.interval*60)/3600
            P_new[idx_N1, idx_N1] = self.cfg.p_amb

        return x_new, P_new, Q_new

    def reset_filter(self, epoch, clk0: float = 0.0, zwd0: float = 0.0):
        # Lista satelitów GPS i Galileo w pierwszej epoce
        gps_sats = self.gps_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        gal_sats = self.gal_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()

        n_gps = len(gps_sats)
        n_gal = len(gal_sats)

        # Wymiary wektora stanu i obserwacji
        base_dim = self.base_dim  # X, Y, Z, clk_gps, isb_galileo, ztd
        dim_x = base_dim + 2 * (n_gps + n_gal)
        dim_z = 2 * (n_gps + n_gal)

        # Wektor stanu
        x0 = np.zeros(dim_x)
        x0[0:3] = self.pos0  # XYZ
        x0[3] = clk0  # clk_gps
        x0[4] = 0.0  # ISB dla Galileo
        x0[5] = zwd0  # ZTD

        # Inicjalizacja EKF
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[3, 3] = self.cfg.p_dt  # clk
        self.ekf.P[4, 4] = self.cfg.p_isb  # isb galileo
        self.ekf.P[5, 5] = self.cfg.p_tro  # zwd

        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_isb
        self.ekf.Q[5, 5] = self.cfg.q_tro

        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0  # model zegara: zresetuj co epokę (opcjonalnie)

        # Inicjalizacja GPS: I_s, N1, N2
        P4_gps = self.gps_obs.loc[(slice(None), epoch), 'P4'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 2 * i
            idx_N1 = idx_I + 1

            I_init = P4_gps[i]  # zakładamy P4 ≈ I_s * (MU2 - 1)

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb

        # Inicjalizacja Galileo: I_s, N1, N2
        P4_gal = self.gal_obs.loc[(slice(None), epoch), 'P4'].to_numpy()
        for j, sv in enumerate(gal_sats):
            k = n_gps + j  # Galileo są po GPS
            idx_I = base_dim + 2 * k
            idx_N1 = idx_I + 1

            I_init = P4_gal[j]  # analogicznie jak dla GPS

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb

        return gps_sats, gal_sats

    def init_filter(self, clk0: float = 0.0, zwd0: float = 0.0):
        # Częstotliwości do przeliczenia P4 → I


        # Wspólna pierwsza epoka
        all_times = sorted(set(self.gps_obs.index.get_level_values('time').unique()) &
                           set(self.gal_obs.index.get_level_values('time').unique()))
        first_ep = all_times[0]

        # Lista satelitów GPS i Galileo w pierwszej epoce
        gps_sats = self.gps_obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()
        gal_sats = self.gal_obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()

        n_gps = len(gps_sats)
        n_gal = len(gal_sats)

        # Wymiary wektora stanu i obserwacji
        base_dim = self.base_dim  # X, Y, Z, clk_gps, isb_galileo, ztd
        dim_x = base_dim + 2 * (n_gps + n_gal)
        dim_z = 2 * (n_gps + n_gal)

        # Wektor stanu
        x0 = np.zeros(dim_x)
        x0[0:3] = self.pos0  # XYZ
        x0[3] = clk0  # clk_gps
        x0[4] = 0.0  # ISB dla Galileo
        x0[5] = zwd0  # ZTD

        # Inicjalizacja EKF
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[3, 3] = self.cfg.p_dt  # clk
        self.ekf.P[4, 4] = self.cfg.p_isb  # isb galileo
        self.ekf.P[5, 5] = self.cfg.p_tro  # zwd

        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_isb
        self.ekf.Q[5, 5] = self.cfg.q_tro

        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0  # model zegara: zresetuj co epokę (opcjonalnie)

        # Inicjalizacja GPS: I_s, N1, N2
        P4_gps = self.gps_obs.loc[(slice(None), first_ep), 'P4'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 2 * i
            idx_N1 = idx_I + 1

            I_init = P4_gps[i]  # zakładamy P4 ≈ I_s * (MU2 - 1)

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb

        # Inicjalizacja Galileo: I_s, N1, N2
        P4_gal = self.gal_obs.loc[(slice(None), first_ep), 'P4'].to_numpy()
        for j, sv in enumerate(gal_sats):
            k = n_gps + j  # Galileo są po GPS
            idx_I = base_dim + 2 * k
            idx_N1 = idx_I + 1

            I_init = P4_gal[j]  # analogicznie jak dla GPS

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb

        return gps_sats, gal_sats, all_times


    def _prepare_obs(self):
        if self.gps_mode == 'L1':
            GF1 = self.FREQ_DICT['L1']
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C1')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L1')][0]

        if self.gal_mode == 'E1':
            GF1 = self.FREQ_DICT['E1']
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C1')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L1')][0]

        if self.gps_mode == 'L2':
            GF1 = self.FREQ_DICT['L2']
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C2')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L2')][0]

        if self.gal_mode == 'E5a':
            GF1 = self.FREQ_DICT['E5a']
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C5')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L5')][0]
        if self.gps_mode == 'L5':
            GF1 = self.FREQ_DICT['L5']
            gps_c1_col = [c for c in self.gps_obs.columns if c.startswith('C5')][0]
            gps_l1_col = [c for c in self.gps_obs.columns if c.startswith('L5')][0]

        if self.gal_mode == 'E5b':
            GF1 = self.FREQ_DICT['E5b']
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C7')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L7')][0]


        gps_data = (gps_c1_col, gps_l1_col)
        gal_data = (gal_c1_col, gal_l1_col)
        self.gps_obs = self.gps_obs.copy()

        if 'pco_los_l1' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los_l1'] = 0.0


        if 'pco_los_l1' not in self.gal_obs.columns.tolist():
            self.gal_obs.loc[:, 'pco_los_l1'] = 0.0


        return gps_data, gal_data

    def extract_ambiguities(self, x, P, n_sats, base_dim=5):
        """
        Wyciąga wektor ambiguł N1, N2 oraz ich macierz kowariancji z pełnego stanu i macierzy P.

        Parameters:
            x        -- wektor stanu EKF (1D array)
            P        -- macierz kowariancji (2D array)
            n_sats   -- liczba satelitów (czyli bloków [I_s, N1_s, N2_s])
            base_dim -- liczba parametrów globalnych (default=5)

        Returns:
            N_vec    -- wektor [N1_1, N2_1, N1_2, N2_2, ..., N1_n, N2_n]
            P_N      -- macierz kowariancji odpowiadająca N_vec
        """
        idxs_N = []
        for i in range(n_sats):
            idx_N1 = base_dim + 2 * i + 1
            idx_N2 = base_dim + 2 * i + 2
            idxs_N.extend([idx_N1, idx_N2])

        N_vec = x[idxs_N]
        P_N = P[np.ix_(idxs_N, idxs_N)]

        return N_vec, P_N

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

    def run_filter(self, clk0=0.0, ref=None, flh=None, zwd0=0.0, trace_filter=False, reset_every=0):
        old_gps_sats, old_gal_sats, all_times = self.init_filter(clk0=clk0, zwd0=zwd0)
        gps_data, gal_data = self._prepare_obs()
        gps_c1, gps_l1 = gps_data
        gal_c1, gal_l1 = gal_data
        result = []
        e, g = [], []
        xyz = None
        conv_time = None
        T0 = all_times[0]
        for num, t in enumerate(all_times):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_gps_sats, old_gal_sats = self.reset_filter(epoch=t)
                    reset_epoch = True
                    T0 = t
            gps_epoch = self.gps_obs.loc[(slice(None), t), :].sort_values(by='sv')
            gal_epoch = self.gal_obs.loc[(slice(None), t), :].sort_values(by='sv')

            def safe_get(df, col, length=None):
                """Zwraca kolumnę jako numpy, a jeśli brak – wektor zer o podanej długości."""
                if col in df.columns:
                    return df[col].to_numpy()
                elif length is not None:
                    return np.zeros(length)
                else:
                    return np.zeros(len(df))

            # --- GPS ---
            gps_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', 'pco_los_l1', 'pco_los_l2',
                f'sat_pco_los_{self.gps_mode}',
                'phw', 'me_wet'
            ]
            gps_len = len(gps_epoch)
            (gps_clk, gps_tro, gps_ah_los, gps_dprel, gps_pco1, gps_pco2,
             sat_pco_L1, phw, mwet) = [safe_get(gps_epoch, c, gps_len) for c in gps_cols]

            # --- Galileo ---
            gal_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', 'pco_los_l1', 'pco_los_l5',
                f'sat_pco_los_{self.gal_mode}',
                'phw'
            ]
            gal_len = len(gal_epoch)
            (gal_clk, gal_tro, gal_ah_los, gal_dprel, gal_pco1, gal_pco2,
             sat_pco_E1, gal_phw) = [safe_get(gal_epoch, c, gal_len) for c in gal_cols]

            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_gal_sats = gal_epoch.index.get_level_values('sv').tolist()
            if (curr_gps_sats != old_gps_sats) or (curr_gal_sats != old_gal_sats):
                # Tworzymy słowniki P4 dla nowej epoki
                P4_gps_dict = gps_epoch['P4'].to_dict()
                P4_gal_dict = gal_epoch['P4'].to_dict()

                # Aktualizacja stanu filtra EKF
                self.ekf.x, self.ekf.P, self.ekf.Q = self.rebuild_state(
                    x_old=self.ekf.x,
                    P_old=self.ekf.P,
                    Q_old=self.ekf.Q,
                    prev_gps=old_gps_sats,
                    prev_gal=old_gal_sats,
                    curr_gps=curr_gps_sats,
                    curr_gal=curr_gal_sats,
                    P4_gps_dict=P4_gps_dict,
                    P4_gal_dict=P4_gal_dict
                )

                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = 2 * len(curr_gps_sats) + 2 * len(curr_gal_sats)
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)

            old_gps_sats = curr_gps_sats
            old_gal_sats = curr_gal_sats

            tides = gps_epoch['tides_los'].to_numpy()
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gps_p1_c1 = gps_epoch.get(f'OSB_{gps_c1}',0.0) * 1e-09 * self.CLIGHT
            gps_pl1_l1 = gps_epoch.get(f'OSB_{gps_l1}',0.0) * 1e-09 * self.CLIGHT
            C1 = gps_epoch[
                     gps_c1].to_numpy() - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p1_c1 - tides

            L1 = gps_epoch[
                     gps_l1].to_numpy() - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl1_l1 - tides - (
                             self.CLIGHT / self.FREQ_DICT[self.gps_mode]) * phw


            gal_tides = gal_epoch['tides_los'].to_numpy()
            gal_satellites = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gal_p1_c1 = gal_epoch.get(f'OSB_{gal_c1}',0.0) * 1e-09 * self.CLIGHT
            gal_pl1_l1 = gal_epoch.get(f'OSB_{gal_l1}',0.0) * 1e-09 * self.CLIGHT

            EC1 = gal_epoch[
                      gal_c1].to_numpy() - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_p1_c1 - gal_tides

            EL1 = gal_epoch[
                      gal_l1].to_numpy() - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_pl1_l1 - gal_tides - (
                              self.CLIGHT / self.FREQ_DICT[self.gal_mode]) * gal_phw


            ZGPS = np.vstack((C1, L1 )).T.reshape(-1)
            ZGAL = np.vstack((EC1, EL1)).T.reshape(-1)

            Z = np.concatenate((ZGPS, ZGAL))
            gps_c1_mask = self.code_screening(x=self.ekf.x[:3],code_obs=C1,thr=30,satellites=gps_satellites[:,:3])
            gal_c1_mask = self.code_screening(x=self.ekf.x[:3], code_obs=EC1, thr=30, satellites=gal_satellites[:,:3])

            # WEIGHTS GPS
            ev_gps = np.deg2rad(gps_epoch['ev'].to_numpy())
            R = np.zeros(2 * len(C1))
            for k in range(len(curr_gps_sats)):
                base = 2 * k

                R[base + 0] = 1 / np.sin(ev_gps[k]) if gps_c1_mask[k] else 1e12

                R[base + 1] = 0.001 / np.sin(ev_gps[k])#if gps_c1_mask[k] else 1e12

            # WEIGHTS GAL
            ev_gal = np.deg2rad(gal_epoch['ev'].to_numpy())
            RG = np.zeros(2 * len(EC1))
            for k in range(len(curr_gal_sats)):
                base = 2 * k

                RG[base + 0] = 1 / np.sin(ev_gal[k]) if gal_c1_mask[k] else 1e12

                RG[base + 1] = 0.001 / np.sin(ev_gal[k])#if gal_c1_mask[k] else 1e12

            Rdiag = np.diag(np.concatenate((R, RG)))



            self.ekf.R = Rdiag

            self.ekf.predict_update(z=Z,
                                    HJacobian=self.Hjacobian,
                                    args=(gps_satellites, gal_satellites),
                                    Hx=self.Hx,
                                    hx_args=(gps_satellites, gal_satellites))
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time

            xyz = self.ekf.x[:3].copy()
            dtr = self.ekf.x[3]
            isb = self.ekf.x[4]
            ztd = self.ekf.x[5]
            stec = self.ekf.x[self.base_dim:][::2]
            n1 = self.ekf.x[self.base_dim:][1::2]

            gps_stec = stec[:len(curr_gps_sats)]
            gal_stec = stec[len(curr_gps_sats):]
            n1_gps = n1[:len(curr_gps_sats)]

            n1_gal = n1[len(curr_gps_sats):]

            gps_epoch['stec'] = gps_stec
            gps_epoch['n1'] = n1_gps


            gal_epoch['stec'] = gal_stec
            gal_epoch['n1'] = n1_gal

            g.append(gps_epoch)
            e.append(gal_epoch)
            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
                result.append(pd.DataFrame(data={'de': enu[0], 'dn': enu[1], 'du': enu[2], 'dtr': dtr, 'ztd': ztd,
                                                 'x': xyz[0], 'y': xyz[1], 'z': xyz[2],'isb':isb},
                                           index=pd.DatetimeIndex(data=[t],name='time')))
            else:
                enu = np.zeros(3)
                result.append(pd.DataFrame(data={'de': enu[0], 'dn': enu[1], 'du': enu[2], 'dtr': dtr, 'ztd': ztd,
                                                 'x': xyz[0], 'y': xyz[1], 'z': xyz[2],'isb':isb},
                                           index=pd.DatetimeIndex(data=[t],name='time')))
            if trace_filter:
                print(result[-1])
        if conv_time is not None:
            if trace_filter:
                print('Convergence time < 5 mm: ', conv_time.total_seconds() / 60,
                      ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")
            ct = conv_time.total_seconds() / 3600
        else:
            ct = None
        gps_result = pd.concat(g)
        gal_result = pd.concat(e)
        result = pd.concat(result)
        result['ct_min']=ct
        return result, gps_result, gal_result, ct

class PPPUdMultiGNSS:
    def __init__(self, config:PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, interval=0.5):
        self.cfg=config
        self.gps_obs = gps_obs
        self.gps_mode = gps_mode
        self.gal_obs = gal_obs
        self.gal_mode = gal_mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = {'L1': 1575.42e06, 'E1': 1575.42e06,
                          'L2': 1227.60e06, 'E5a': 1176.45e06,
                          'L5': 1176.450e06, 'E5b': 1207.14e06}
        self.LAMBDA_DICT = {}
        self.CLIGHT = 299792458
        self.base_dim = 6 if self.tro else 5
        self.pos0 = pos0
        self.interval=interval

    def Hjacobian(self, x, gps_satellites, gal_satellites):
        C = self.CLIGHT
        # Częstotliwości GPS
        F1_GPS = self.FREQ_DICT[self.gps_mode[:2]]
        F2_GPS = self.FREQ_DICT[self.gps_mode[2:]]
        L1 = C / F1_GPS
        L2 = C / F2_GPS
        MU1 = 1.0
        MU2 = (F1_GPS / F2_GPS) ** 2

        # Częstotliwości Galileo
        F1_GAL = self.FREQ_DICT[self.gal_mode[:2]]
        F2_GAL = self.FREQ_DICT[self.gal_mode[2:]]
        E1 = C / F1_GAL
        E5a = C / F2_GAL
        MU1_GAL = 1.0
        MU2_GAL = (F1_GAL / F2_GAL) ** 2

        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        rec_xyz = x[:3].copy()

        # Jednostkowe wektory do satelitów GPS
        rho_gps = gps_satellites[:, :3] - rec_xyz
        e_gps = rho_gps / np.linalg.norm(rho_gps, axis=1)[:, None]
        m_wet_gps = gps_satellites[:, 3]

        # Jednostkowe wektory do satelitów Galileo
        rho_gal = gal_satellites[:, :3] - rec_xyz
        e_gal = rho_gal / np.linalg.norm(rho_gal, axis=1)[:, None]
        m_wet_gal = gal_satellites[:, 3]

        # Parametry wspólne: XYZ, clk_gps, ISB_galileo, ZTD
        COL_X, COL_Y, COL_Z = 0, 1, 2
        COL_CLK = 3
        COL_ISB = 4
        COL_ZTD = 5

        base_dim = 6  # wspólne parametry (XYZ + clk_gps + isb + ztd)

        # 4 równania na każdy satelita: P1, L1, P2, L2
        H = np.zeros((4 * (n_gps + n_gal), base_dim + 3 * (n_gps + n_gal)))

        # ---------------- GPS ----------------
        for s in range(n_gps):
            row = 4 * s
            ex, ey, ez = -e_gps[s]
            mw = m_wet_gps[s]

            for i in range(4):
                H[row + i, [COL_X, COL_Y, COL_Z]] = [ex, ey, ez]
                H[row + i, COL_CLK] = 1.0  # wspólny zegar GPS
                H[row + i, COL_ZTD] = mw

            col_iono = base_dim + 3 * s
            col_N1 = col_iono + 1
            col_N2 = col_iono + 2

            # Efekt IONO
            H[row + 0, col_iono] = +MU1  # P1
            H[row + 1, col_iono] = -MU1  # L1
            H[row + 2, col_iono] = +MU2  # P2
            H[row + 3, col_iono] = -MU2  # L2

            # Ambiguity
            H[row + 1, col_N1] = L1
            H[row + 3, col_N2] = L2

        # ---------------- Galileo ----------------
        for s in range(n_gal):
            row = 4 * (n_gps + s)
            ex, ey, ez = -e_gal[s]
            mw = m_wet_gal[s]

            for i in range(4):
                H[row + i, [COL_X, COL_Y, COL_Z]] = [ex, ey, ez]
                H[row + i, COL_CLK] = 1.0  # współdzielony zegar
                H[row + i, COL_ISB] = 1.0  # inter-sys bias dla Galileo
                H[row + i, COL_ZTD] = mw

            col_iono = base_dim + 3 * (n_gps + s)
            col_N1 = col_iono + 1
            col_N2 = col_iono + 2

            # Efekt IONO (na Galileo)
            H[row + 0, col_iono] = +MU1_GAL  # E1 code
            H[row + 1, col_iono] = -MU1_GAL  # E1 phase
            H[row + 2, col_iono] = +MU2_GAL  # E5a code
            H[row + 3, col_iono] = -MU2_GAL  # E5a phase

            # Ambiguity (Galileo)
            H[row + 1, col_N1] = E1
            H[row + 3, col_N2] = E5a

        return H

    def Hx(self, x: np.ndarray, gps_satellites: np.ndarray, gal_satellites: np.ndarray) -> np.ndarray:
        x_state = x.copy()
        C = self.CLIGHT

        # GPS
        F1_GPS = self.FREQ_DICT[self.gps_mode[:2]]
        F2_GPS = self.FREQ_DICT[self.gps_mode[2:]]
        L1 = C / F1_GPS
        L2 = C / F2_GPS
        MU1 = 1.0
        MU2 = (F1_GPS / F2_GPS) ** 2

        # Galileo
        F1_GAL = self.FREQ_DICT[self.gal_mode[:2]]
        F2_GAL = self.FREQ_DICT[self.gal_mode[2:]]
        E1 = C / F1_GAL
        E5a = C / F2_GAL
        MU1_GAL = 1.0
        MU2_GAL = (F1_GAL / F2_GAL) ** 2

        xr, yr, zr = x_state[0:3]
        clk = x_state[3]
        isb = x_state[4]  # inter-sys bias (dla Galileo)
        zwd = x_state[5]

        base_dim = self.base_dim
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]

        # Z góry rozmiar wektora predykcji: 4 obserwacje na każdy satelita
        z_hat = np.empty(4 * (n_gps + n_gal))

        # --- GPS ---
        sat_xyz = gps_satellites[:, :3]
        m_wet = gps_satellites[:, 3]
        rho_vec = sat_xyz - np.array([xr, yr, zr])
        rho = np.linalg.norm(rho_vec, axis=1)

        for s in range(n_gps):
            i = base_dim + 3 * s
            I_s = x_state[i]
            N1_s = x_state[i + 1]
            N2_s = x_state[i + 2]

            geom = rho[s]
            mw = m_wet[s]

            z_hat[4 * s + 0] = geom + clk + mw * zwd + MU1 * I_s  # P1
            z_hat[4 * s + 1] = geom + clk + mw * zwd - MU1 * I_s + L1 * N1_s  # L1
            z_hat[4 * s + 2] = geom + clk + mw * zwd + MU2 * I_s  # P2
            z_hat[4 * s + 3] = geom + clk + mw * zwd - MU2 * I_s + L2 * N2_s  # L2

        # --- Galileo ---
        sat_xyz = gal_satellites[:, :3]
        m_wet = gal_satellites[:, 3]
        rho_vec = sat_xyz - np.array([xr, yr, zr])
        rho = np.linalg.norm(rho_vec, axis=1)

        for s in range(n_gal):
            i = base_dim + 3 * (n_gps + s)
            I_s = x_state[i]
            N1_s = x_state[i + 1]
            N2_s = x_state[i + 2]

            geom = rho[s]
            mw = m_wet[s]

            row = 4 * (n_gps + s)

            z_hat[row + 0] = geom + clk + isb + mw * zwd + MU1_GAL * I_s  # E1 code
            z_hat[row + 1] = geom + clk + isb + mw * zwd - MU1_GAL * I_s + E1 * N1_s  # E1 phase
            z_hat[row + 2] = geom + clk + isb + mw * zwd + MU2_GAL * I_s  # E5a code
            z_hat[row + 3] = geom + clk + isb + mw * zwd - MU2_GAL * I_s + E5a * N2_s  # E5a phase

        return z_hat

    def rebuild_state(self, x_old, P_old, Q_old,
                      prev_gps, prev_gal, curr_gps, curr_gal,
                      P4_gps_dict: dict, P4_gal_dict: dict):
        """
        Aktualizuje stan filtra Kalman w sytuacji, gdy zmienia się widoczność satelitów GPS i Galileo.
        Dla nowych satelitów inicjalizuje I_s na podstawie obserwacji P4.

        Parametry:
        - prev_gps / prev_gal: listy satelitów poprzedniej epoki
        - curr_gps / curr_gal: listy satelitów bieżącej epoki
        - P4_gps_dict: słownik {sv: P4} dla GPS
        - P4_gal_dict: słownik {sv: P4} dla Galileo

        Zakładamy:
        - self.base_dim = 6 (X, Y, Z, clk_GPS, ISB_GAL, ZTD)
        """

        base_dim = self.base_dim
        n_prev = len(prev_gps) + len(prev_gal)
        n_curr = len(curr_gps) + len(curr_gal)

        old_dim = base_dim + 3 * n_prev
        new_dim = base_dim + 3 * n_curr

        x_new = np.zeros(new_dim)
        x_new[:base_dim] = x_old[:base_dim]

        P_new = np.zeros((new_dim, new_dim))
        P_new[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]

        Q_new = np.zeros((new_dim, new_dim))
        Q_new[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]

        # Oznaczenia systemowe, np. G:G01, E:E11
        def tag(system: str, sv: str) -> str:
            return f"{system}:{sv}"

        prev_all = [tag("G", sv) for sv in prev_gps] + [tag("E", sv) for sv in prev_gal]
        curr_all = [tag("G", sv) for sv in curr_gps] + [tag("E", sv) for sv in curr_gal]

        common = set(prev_all) & set(curr_all)

        # --- przenoszenie wspólnych satelitów ---
        for prn in common:
            i_old = prev_all.index(prn)
            i_new = curr_all.index(prn)
            for k in range(3):
                xo = base_dim + 3 * i_old + k
                xn = base_dim + 3 * i_new + k
                x_new[xn] = x_old[xo]

                P_new[xn, :base_dim] = P_old[xo, :base_dim]
                P_new[:base_dim, xn] = P_old[:base_dim, xo]
                Q_new[xn, :base_dim] = Q_old[xo, :base_dim]
                Q_new[:base_dim, xn] = Q_old[:base_dim, xo]

                for prn2 in common:
                    j_old = prev_all.index(prn2)
                    j_new = curr_all.index(prn2)
                    for l in range(3):
                        yo = base_dim + 3 * j_old + l
                        yn = base_dim + 3 * j_new + l
                        P_new[xn, yn] = P_old[xo, yo]
                        Q_new[xn, yn] = Q_old[xo, yo]

        # --- nowe satelity: zainicjalizuj I_s z P4 ---
        new_only = set(curr_all) - set(prev_all)
        for prn in new_only:
            j = curr_all.index(prn)
            idx_I = base_dim + 3 * j
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2

            # Domyślny init
            x_new[idx_I] = 0.0

            # Próba inicjalizacji z P4
            if prn.startswith("G:"):
                sv = prn[2:]
                if sv in P4_gps_dict:
                    x_new[idx_I] = P4_gps_dict[sv]
            elif prn.startswith("E:"):
                sv = prn[2:]
                if sv in P4_gal_dict:
                    x_new[idx_I] = P4_gal_dict[sv]

            # Ustaw P, Q
            P_new[idx_I, idx_I] = self.cfg.p_iono
            Q_new[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600
            P_new[idx_N1, idx_N1] = self.cfg.p_amb
            P_new[idx_N2, idx_N2] = self.cfg.p_amb

        return x_new, P_new, Q_new
    def reset_filter(self, epoch, clk0: float = 0.0, zwd0: float = 0.0):
        # Lista satelitów GPS i Galileo w pierwszej epoce
        gps_sats = self.gps_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        gal_sats = self.gal_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()

        n_gps = len(gps_sats)
        n_gal = len(gal_sats)

        # Wymiary wektora stanu i obserwacji
        base_dim = self.base_dim  # X, Y, Z, clk_gps, isb_galileo, ztd
        dim_x = base_dim + 3 * (n_gps + n_gal)
        dim_z = 4 * (n_gps + n_gal)

        # Wektor stanu
        x0 = np.zeros(dim_x)
        x0[0:3] = self.pos0  # XYZ
        x0[3] = clk0  # clk_gps
        x0[4] = 0.0  # ISB dla Galileo
        x0[5] = zwd0  # ZTD

        # Inicjalizacja EKF
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[3, 3] = self.cfg.p_dt # clk
        self.ekf.P[4, 4] = self.cfg.p_dt  # isb galileo
        self.ekf.P[5, 5] = self.cfg.p_tro  # zwd

        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_dt
        self.ekf.Q[5, 5] = self.cfg.q_tro

        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0  # model zegara: zresetuj co epokę (opcjonalnie)

        # Inicjalizacja GPS: I_s, N1, N2
        P4_gps = self.gps_obs.loc[(slice(None), epoch), 'P4'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 3 * i
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2

            I_init = P4_gps[i]  # zakładamy P4 ≈ I_s * (MU2 - 1)

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb

        # Inicjalizacja Galileo: I_s, N1, N2
        P4_gal = self.gal_obs.loc[(slice(None), epoch), 'P4'].to_numpy()
        for j, sv in enumerate(gal_sats):
            k = n_gps + j  # Galileo są po GPS
            idx_I = base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2

            I_init = P4_gal[j]  # analogicznie jak dla GPS

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb

        return gps_sats, gal_sats

    def init_filter(self, clk0: float = 0.0, zwd0: float = 0.0):
        # Częstotliwości do przeliczenia P4 → I


        # Wspólna pierwsza epoka
        all_times = sorted(set(self.gps_obs.index.get_level_values('time').unique()) &
                           set(self.gal_obs.index.get_level_values('time').unique()))
        first_ep = all_times[0]

        # Lista satelitów GPS i Galileo w pierwszej epoce
        gps_sats = self.gps_obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()
        gal_sats = self.gal_obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()

        n_gps = len(gps_sats)
        n_gal = len(gal_sats)

        # Wymiary wektora stanu i obserwacji
        base_dim = self.base_dim  # X, Y, Z, clk_gps, isb_galileo, ztd
        dim_x = base_dim + 3 * (n_gps + n_gal)
        dim_z = 4 * (n_gps + n_gal)

        # Wektor stanu
        x0 = np.zeros(dim_x)
        x0[0:3] = self.pos0  # XYZ
        x0[3] = clk0  # clk_gps
        x0[4] = 0.0  # ISB dla Galileo
        x0[5] = zwd0  # ZTD

        # Inicjalizacja EKF
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[3, 3] = self.cfg.p_dt  # clk
        self.ekf.P[4, 4] = self.cfg.p_dt  # isb galileo
        self.ekf.P[5, 5] = self.cfg.p_tro  # zwd

        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_dt
        self.ekf.Q[5, 5] = self.cfg.q_tro

        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0  # model zegara: zresetuj co epokę (opcjonalnie)

        # Inicjalizacja GPS: I_s, N1, N2
        P4_gps = self.gps_obs.loc[(slice(None), first_ep), 'P4'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 3 * i
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2

            I_init = P4_gps[i]  # zakładamy P4 ≈ I_s * (MU2 - 1)

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb

        # Inicjalizacja Galileo: I_s, N1, N2
        P4_gal = self.gal_obs.loc[(slice(None), first_ep), 'P4'].to_numpy()
        for j, sv in enumerate(gal_sats):
            k = n_gps + j  # Galileo są po GPS
            idx_I = base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2

            I_init = P4_gal[j]  # analogicznie jak dla GPS

            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_amb
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600

            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb

        return gps_sats, gal_sats, all_times


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
            GF1 = self.FREQ_DICT['E1']
            GF2 = self.FREQ_DICT['E5a']
            agal, bgal = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C1')][0]
            gal_c2_col = [c for c in self.gal_obs.columns if c.startswith('C5')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L1')][0]
            gal_l2_col = [c for c in self.gal_obs.columns if c.startswith('L5')][0]
        if self.gal_mode == 'E1E5b':
            GF1 = self.FREQ_DICT['E1']
            GF2 = self.FREQ_DICT['E5b']
            agal, bgal = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gal_c1_col = [c for c in self.gal_obs.columns if c.startswith('C1')][0]
            gal_c2_col = [c for c in self.gal_obs.columns if c.startswith('C7')][0]
            gal_l1_col = [c for c in self.gal_obs.columns if c.startswith('L1')][0]
            gal_l2_col = [c for c in self.gal_obs.columns if c.startswith('L7')][0]

        gps_data = (agps, bgps, gps_c1_col, gps_c2_col, gps_l1_col, gps_l2_col)
        gal_data = (agal, bgal, gal_c1_col, gal_c2_col, gal_l1_col, gal_l2_col)
        self.gps_obs = self.gps_obs.copy()

        if 'pco_los_l1' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los_l1'] = 0.0
        if 'pco_los_l2' not in self.gps_obs.columns.tolist():
            self.gps_obs.loc[:, 'pco_los_l2'] = 0.0

        if 'pco_los_l1' not in self.gal_obs.columns.tolist():
            self.gal_obs.loc[:, 'pco_los_l1'] = 0.0
        if 'pco_los_l5' not in self.gal_obs.columns.tolist():
            self.gal_obs.loc[:, 'pco_los_l5'] = 0.0

        return gps_data, gal_data

    def extract_ambiguities(self, x, P, n_sats, base_dim=5):
        """
        Wyciąga wektor ambiguł N1, N2 oraz ich macierz kowariancji z pełnego stanu i macierzy P.

        Parameters:
            x        -- wektor stanu EKF (1D array)
            P        -- macierz kowariancji (2D array)
            n_sats   -- liczba satelitów (czyli bloków [I_s, N1_s, N2_s])
            base_dim -- liczba parametrów globalnych (default=5)

        Returns:
            N_vec    -- wektor [N1_1, N2_1, N1_2, N2_2, ..., N1_n, N2_n]
            P_N      -- macierz kowariancji odpowiadająca N_vec
        """
        idxs_N = []
        for i in range(n_sats):
            idx_N1 = base_dim + 3 * i + 1
            idx_N2 = base_dim + 3 * i + 2
            idxs_N.extend([idx_N1, idx_N2])

        N_vec = x[idxs_N]
        P_N = P[np.ix_(idxs_N, idxs_N)]

        return N_vec, P_N

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

    def run_filter(self, clk0=0.0, ref=None, flh=None, zwd0=0.0, trace_filter=False, reset_every=0):
        old_gps_sats, old_gal_sats, all_times = self.init_filter(clk0=clk0, zwd0=zwd0)
        gps_data, gal_data = self._prepare_obs()
        agps, bgps, gps_c1, gps_c2, gps_l1, gps_l2 = gps_data
        _, _, gal_c1, gal_c2, gal_l1, gal_l2 = gal_data
        result = []
        e, g = [], []
        xyz = None
        conv_time = None
        T0 = all_times[0]
        for num, t in enumerate(all_times):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_gps_sats, old_gal_sats = self.reset_filter(epoch=t)
                    reset_epoch = True
                    T0 = t
            gps_epoch = self.gps_obs.loc[(slice(None), t), :].sort_values(by='sv')
            gal_epoch = self.gal_obs.loc[(slice(None), t), :].sort_values(by='sv')

            def safe_get(df, col, length=None):
                """Zwraca kolumnę jako numpy, a jeśli brak – wektor zer o podanej długości."""
                if col in df.columns:
                    return df[col].to_numpy()
                elif length is not None:
                    return np.zeros(length)
                else:
                    return np.zeros(len(df))

            # --- GPS ---
            gps_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', 'pco_los_l1', 'pco_los_l2',
                f'sat_pco_los_{self.gps_mode[:2]}',
                f'sat_pco_los_{self.gps_mode[2:]}',
                'phw', 'me_wet'
            ]
            gps_len = len(gps_epoch)
            (gps_clk, gps_tro, gps_ah_los, gps_dprel, gps_pco1, gps_pco2,
             sat_pco_L1, sat_pco_L2, phw, mwet) = [safe_get(gps_epoch, c, gps_len) for c in gps_cols]

            # --- Galileo ---
            gal_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', 'pco_los_l1', 'pco_los_l5',
                f'sat_pco_los_{self.gal_mode[:2]}',
                f'sat_pco_los_{self.gal_mode[2:]}',
                'phw'
            ]
            gal_len = len(gal_epoch)
            (gal_clk, gal_tro, gal_ah_los, gal_dprel, gal_pco1, gal_pco2,
             sat_pco_E1, sat_pco_E5a, gal_phw) = [safe_get(gal_epoch, c, gal_len) for c in gal_cols]

            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_gal_sats = gal_epoch.index.get_level_values('sv').tolist()
            if (curr_gps_sats != old_gps_sats) or (curr_gal_sats != old_gal_sats):
                # Tworzymy słowniki P4 dla nowej epoki
                P4_gps_dict = gps_epoch['P4'].to_dict()
                P4_gal_dict = gal_epoch['P4'].to_dict()

                # Aktualizacja stanu filtra EKF
                self.ekf.x, self.ekf.P, self.ekf.Q = self.rebuild_state(
                    x_old=self.ekf.x,
                    P_old=self.ekf.P,
                    Q_old=self.ekf.Q,
                    prev_gps=old_gps_sats,
                    prev_gal=old_gal_sats,
                    curr_gps=curr_gps_sats,
                    curr_gal=curr_gal_sats,
                    P4_gps_dict=P4_gps_dict,
                    P4_gal_dict=P4_gal_dict
                )

                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = 4 * len(curr_gps_sats) + 4 * len(curr_gal_sats)
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)

            old_gps_sats = curr_gps_sats
            old_gal_sats = curr_gal_sats

            tides = gps_epoch['tides_los'].to_numpy()
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gps_p1_c1 = gps_epoch.get(f'OSB_{gps_c1}',0.0) * 1e-09 * self.CLIGHT
            gps_p2_c2 = gps_epoch.get(f'OSB_{gps_c2}',0.0) * 1e-09 * self.CLIGHT
            gps_pl1_l1 = gps_epoch.get(f'OSB_{gps_l1}',0.0) * 1e-09 * self.CLIGHT
            gps_pl2_l2 = gps_epoch.get(f'OSB_{gps_l2}',0.0) * 1e-09 * self.CLIGHT
            C1 = gps_epoch[
                     gps_c1].to_numpy() - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p1_c1 - tides
            C2 = gps_epoch[
                     gps_c2].to_numpy() - gps_pco2 + sat_pco_L2 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p2_c2 - tides
            L1 = gps_epoch[
                     gps_l1].to_numpy() - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl1_l1 - tides - (
                             self.CLIGHT / self.FREQ_DICT[self.gps_mode[:2]]) * phw
            L2 = gps_epoch[
                     gps_l2].to_numpy() - gps_pco2 + sat_pco_L2 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl2_l2 - tides - (
                             self.CLIGHT / self.FREQ_DICT[self.gps_mode[2:]]) * phw

            gal_tides = gal_epoch['tides_los'].to_numpy()
            gal_satellites = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gal_p1_c1 = gal_epoch.get(f'OSB_{gal_c1}',0.0) * 1e-09 * self.CLIGHT
            gal_p2_c2 = gal_epoch.get(f'OSB_{gal_c2}',0.0) * 1e-09 * self.CLIGHT
            gal_pl1_l1 = gal_epoch.get(f'OSB_{gal_l1}',0.0) * 1e-09 * self.CLIGHT
            gal_pl2_l2 = gal_epoch.get(f'OSB_{gal_l2}',0.0) * 1e-09 * self.CLIGHT
            EC1 = gal_epoch[
                      gal_c1].to_numpy() - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_p1_c1 - gal_tides
            EC2 = gal_epoch[
                      gal_c2].to_numpy() - gal_pco2 + sat_pco_E5a + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_p2_c2 - gal_tides
            EL1 = gal_epoch[
                      gal_l1].to_numpy() - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_pl1_l1 - gal_tides - (
                              self.CLIGHT / self.FREQ_DICT[self.gal_mode[:2]]) * gal_phw
            EL2 = gal_epoch[
                      gal_l2].to_numpy() - gal_pco2 + sat_pco_E5a + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_pl2_l2 - gal_tides - (
                              self.CLIGHT / self.FREQ_DICT[self.gal_mode[2:]]) * gal_phw

            ZGPS = np.vstack((C1, L1, C2, L2)).T.reshape(-1)
            ZGAL = np.vstack((EC1, EL1, EC2, EL2)).T.reshape(-1)

            Z = np.concatenate((ZGPS, ZGAL))
            gps_c1_mask = self.code_screening(x=self.ekf.x[:3],code_obs=C1,thr=30,satellites=gps_satellites[:,:3])
            gps_c2_mask = self.code_screening(x=self.ekf.x[:3], code_obs=C2, thr=30, satellites=gps_satellites[:,:3])
            gal_c1_mask = self.code_screening(x=self.ekf.x[:3], code_obs=EC1, thr=30, satellites=gal_satellites[:,:3])
            gal_c2_mask = self.code_screening(x=self.ekf.x[:3], code_obs=EC2, thr=30, satellites=gal_satellites[:,:3])

            # WEIGHTS GPS
            ev_gps = np.deg2rad(gps_epoch['ev'].to_numpy())
            R = np.zeros(4 * len(C1))
            for k in range(len(curr_gps_sats)):
                base = 4 * k

                R[base + 0] = 1 / np.sin(ev_gps[k]) if gps_c1_mask[k] else 1e12

                R[base + 1] = 0.001 / np.sin(ev_gps[k])#if gps_c1_mask[k] else 1e12

                R[base + 2] = 1 / np.sin(ev_gps[k]) if gps_c2_mask[k] else 1e12

                R[base + 3] = 0.001 / np.sin(ev_gps[k])#if gps_c2_mask[k] else 1e12

            # WEIGHTS GAL
            ev_gal = np.deg2rad(gal_epoch['ev'].to_numpy())
            RG = np.zeros(4 * len(EC1))
            for k in range(len(curr_gal_sats)):
                base = 4 * k

                RG[base + 0] = 1 / np.sin(ev_gal[k]) if gal_c1_mask[k] else 1e12

                RG[base + 1] = 0.001 / np.sin(ev_gal[k])#if gal_c1_mask[k] else 1e12

                RG[base + 2] = 1 / np.sin(ev_gal[k]) if gal_c2_mask[k] else 1e12

                RG[base + 3] = 0.001 / np.sin(ev_gal[k]) #if gal_c2_mask[k] else 1e12

            Rdiag = np.diag(np.concatenate((R, RG)))



            self.ekf.R = Rdiag

            self.ekf.predict_update(z=Z,
                                    HJacobian=self.Hjacobian,
                                    args=(gps_satellites, gal_satellites),
                                    Hx=self.Hx,
                                    hx_args=(gps_satellites, gal_satellites))
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time

            xyz = self.ekf.x[:3].copy()
            dtr = self.ekf.x[3]
            isb = self.ekf.x[4]
            ztd = self.ekf.x[5]
            stec = self.ekf.x[self.base_dim:][::3]
            n1 = self.ekf.x[self.base_dim:][1::3]
            n2 = self.ekf.x[self.base_dim:][2::3]

            gps_stec = stec[:len(curr_gps_sats)]
            gal_stec = stec[len(curr_gps_sats):]
            n1_gps = n1[:len(curr_gps_sats)]
            n2_gps = n2[:len(curr_gps_sats)]

            n1_gal = n1[len(curr_gps_sats):]
            n2_gal = n2[len(curr_gps_sats):]

            gps_epoch['stec'] = gps_stec
            gps_epoch['n1'] = n1_gps
            gps_epoch['n2'] = n2_gps


            gal_epoch['stec'] = gal_stec
            gal_epoch['n1'] = n1_gal
            gal_epoch['n2'] = n2_gal
            g.append(gps_epoch)
            e.append(gal_epoch)
            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
                result.append(pd.DataFrame(data={'de': enu[0], 'dn': enu[1], 'du': enu[2], 'dtr': dtr, 'ztd': ztd,
                                                 'x': xyz[0], 'y': xyz[1], 'z': xyz[2],'isb':isb},
                                           index=pd.DatetimeIndex(data=[t],name='time')))
            else:
                enu = np.zeros(3)
                result.append(pd.DataFrame(data={'de': enu[0], 'dn': enu[1], 'du': enu[2], 'dtr': dtr, 'ztd': ztd,
                                                 'x': xyz[0], 'y': xyz[1], 'z': xyz[2],'isb':isb},
                                           index=pd.DatetimeIndex(data=[t],name='time')))
            if trace_filter:
                print(result[-1])
        if conv_time is not None:
            if trace_filter:
                print('Convergence time < 5 mm: ', conv_time.total_seconds() / 60,
                      ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")
            ct = conv_time.total_seconds() / 3600
        else:
            ct = None
        gps_result = pd.concat(g)
        gal_result = pd.concat(e)
        result = pd.concat(result)
        result['ct_min']=ct
        return result, gps_result, gal_result, ct


class PPPUdSingleGNSS:
    def __init__(self, config:PPPConfig, obs, mode, ekf, pos0, tro=True,interval=0.5):
        self.cfg=config
        self.obs = obs
        self.mode = mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = {'L1': 1575.42e06, 'E1': 1575.42e06,
                          'L2': 1227.60e06, 'E5a': 1176.45e06,
                          'L5': 1176.450e06, 'E5b': 1207.14e06}
        self.LAMBDA_DICT = {}
        self.CLIGHT = 299792458
        self.base_dim = 5 if self.tro else 4
        self.pos0 = pos0
        self.interval=interval

    def Hjacobian(self, x, gps_satellites):
        C = self.CLIGHT
        F1 = self.FREQ_DICT[self.mode[:2]]
        F2 = self.FREQ_DICT[self.mode[2:]]
        L1 = C / F1
        L2 = C / F2
        MU1 = 1.0
        MU2 = (F1 / F2) ** 2

        sat_xyz = gps_satellites[:, :3]
        m_wet = gps_satellites[:, 3]
        n = sat_xyz.shape[0]
        rec_xyz = x[:3].copy()
        rho_vec = sat_xyz - rec_xyz
        rho_norm = np.linalg.norm(rho_vec, axis=1, keepdims=True)
        e_vec = rho_vec / rho_norm

        n_params = self.base_dim + 3 * n
        H = np.zeros((4 * n, n_params))
        COL_X, COL_Y, COL_Z, COL_CLK, COL_ZTD = 0, 1, 2, 3, 4

        for s in range(n):
            row = 4 * s
            ex, ey, ez = -e_vec[s]
            mw = m_wet[s]

            for i in range(4):
                H[row + i, [COL_X, COL_Y, COL_Z]] = [ex, ey, ez]
                H[row + i, COL_CLK] = 1.0
                H[row + i, COL_ZTD] = mw

            col_iono = self.base_dim + 3 * s
            col_N1 = col_iono + 1
            col_N2 = col_iono + 2

            H[row + 0, col_iono] = +MU1
            H[row + 1, col_iono] = -MU1
            H[row + 2, col_iono] = +MU2
            H[row + 3, col_iono] = -MU2

            H[row + 1, col_N1] = L1
            H[row + 3, col_N2] = L2
        # print(H)
        return H

    def Hx(self, x: np.ndarray, gps_satellites: np.ndarray) -> np.ndarray:
        x_state = x.copy()
        C = self.CLIGHT
        F1 = self.FREQ_DICT[self.mode[:2]]
        F2 = self.FREQ_DICT[self.mode[2:]]
        L1 = C / F1
        L2 = C / F2
        MU1 = 1.0
        MU2 = (F1 / F2) ** 2

        xr, yr, zr = x_state[0:3]
        clk = x_state[3]
        zwd = x_state[4]
        n = gps_satellites.shape[0]
        base_dim = self.base_dim

        sat_xyz = gps_satellites[:, :3]
        m_wet = gps_satellites[:, 3]
        rho_vec = sat_xyz - np.array([xr, yr, zr])
        rho = np.linalg.norm(rho_vec, axis=1)

        z_hat = np.empty(4 * n)

        for s in range(n):
            i = base_dim + 3 * s
            I_s = x_state[i]
            N1_s = x_state[i + 1]
            N2_s = x_state[i + 2]

            geom = rho[s]
            mw = m_wet[s]

            z_hat[4 * s + 0] = geom + clk + mw * zwd + MU1 * I_s
            z_hat[4 * s + 1] = geom + clk + mw * zwd - MU1 * I_s + L1 * N1_s
            z_hat[4 * s + 2] = geom + clk + mw * zwd + MU2 * I_s
            z_hat[4 * s + 3] = geom + clk + mw * zwd - MU2 * I_s + L2 * N2_s

        return z_hat

    def check_dim_consistency(self, Z, Hx):
        if Z.shape != Hx.shape:
            raise ValueError(f"Dimension mismatch: Z {Z.shape} vs Hx {Hx.shape}")

    def rebuild_state(self, x_old, P_old, Q_old, prev_sats, curr_sats):
        base_dim = self.base_dim
        n_prev = len(prev_sats)
        n_curr = len(curr_sats)
        old_dim = base_dim + 3 * n_prev
        new_dim = base_dim + 3 * n_curr

        x_new = np.zeros(new_dim)
        x_new[:base_dim] = x_old[:base_dim]

        P_new = np.zeros((new_dim, new_dim))
        P_new[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]

        Q_new = np.zeros((new_dim, new_dim))
        Q_new[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]

        common = set(prev_sats) & set(curr_sats)
        for prn in common:
            i_old = prev_sats.index(prn)
            i_new = curr_sats.index(prn)
            for k in range(3):
                xo = base_dim + 3 * i_old + k
                xn = base_dim + 3 * i_new + k
                x_new[xn] = x_old[xo]

                P_new[xn, :base_dim] = P_old[xo, :base_dim]
                P_new[:base_dim, xn] = P_old[:base_dim, xo]
                Q_new[xn, :base_dim] = Q_old[xo, :base_dim]
                Q_new[:base_dim, xn] = Q_old[:base_dim, xo]

                for prn2 in common:
                    j_old = prev_sats.index(prn2)
                    j_new = curr_sats.index(prn2)
                    for l in range(3):
                        yo = base_dim + 3 * j_old + l
                        yn = base_dim + 3 * j_new + l
                        P_new[xn, yn] = P_old[xo, yo]
                        Q_new[xn, yn] = Q_old[xo, yo]

        new_only = set(curr_sats) - set(prev_sats)
        for prn in new_only:
            j = curr_sats.index(prn)
            idx_I = base_dim + 3 * j
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            P_new[idx_I, idx_I] = self.cfg.p_iono#100.0
            P_new[idx_N1, idx_N1] = self.cfg.p_amb#1e6
            P_new[idx_N2, idx_N2] = self.cfg.p_amb#1e6

        return x_new, P_new, Q_new

    def reset_filter(self, epoch, clk0: float = 0.0, zwd0: float = 0.0):
        sats0 = self.obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        n0 = len(sats0)
        dim_x = self.base_dim + 3 * n0
        dim_z = 4 * n0

        x0 = np.zeros(dim_x)
        x0[:3] = self.pos0
        x0[3] = clk0
        x0[4] = zwd0

        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2

        # self.ekf.P[:3,:3] = 0.0

        self.ekf.P[3, 3] = self.cfg.p_dt#1e9
        self.ekf.P[4, 4] = self.cfg.p_tro#2

        P4 = self.obs.loc[(slice(None), epoch), 'P4'].to_numpy()
        I_init = P4  # / (MU2 - 1.0)
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt * (self.interval * 60) / 3600 #9e9
        self.ekf.Q[4, 4] = self.cfg.q_tro#0.025
        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0

        for k in range(n0):
            idx_I = self.base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            self.ekf.P[idx_I, idx_I] =self.cfg.p_iono# 100
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb #1e6

            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb #1e6
            self.ekf.x[idx_I] = I_init[k]  # ← inicjalizacja I_s z P4

        return sats0

    def init_filter(self, clk0: float = 0.0, zwd0: float = 0.0):
        gps_epochs = sorted(self.obs.index.get_level_values('time').unique())
        first_ep = gps_epochs[0]
        sats0 = self.obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()
        n0 = len(sats0)
        dim_x = self.base_dim + 3 * n0
        dim_z = 4 * n0

        x0 = np.zeros(dim_x)
        x0[:3] = self.pos0
        x0[3] = clk0
        x0[4] = zwd0

        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2

        # self.ekf.P[:3,:3] = 0.0

        self.ekf.P[3, 3] = self.cfg.p_dt #1e9
        self.ekf.P[4, 4] = self.cfg.p_tro #2

        P4 = self.obs.loc[(slice(None), first_ep), 'P4'].to_numpy()
        I_init = P4  # / (MU2 - 1.0)
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt #9e9
        self.ekf.Q[4, 4] = self.cfg.q_tro #0.025
        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0

        for k in range(n0):
            idx_I = self.base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono #100
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono * (self.interval * 60) / 3600#25
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb#1e6

            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb #1e6
            self.ekf.x[idx_I] = I_init[k]  # ← inicjalizacja I_s z P4

        return sats0, gps_epochs

    def _prepare_obs(self):
        if self.mode == 'L1L2':
            GF1 = self.FREQ_DICT['L1']
            GF2 = self.FREQ_DICT['L2']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.obs.columns if c.startswith('C2')][0]
            gps_l1_col = [c for c in self.obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.obs.columns if c.startswith('L2')][0]

        if self.mode == 'L1L5':
            GF1 = self.FREQ_DICT['L1']
            GF2 = self.FREQ_DICT['L5']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.obs.columns if c.startswith('C5')][0]
            gps_l1_col = [c for c in self.obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.obs.columns if c.startswith('L5')][0]

        if self.mode == 'E1E5a':
            GF1 = self.FREQ_DICT['E1']
            GF2 = self.FREQ_DICT['E5a']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.obs.columns if c.startswith('C5')][0]
            gps_l1_col = [c for c in self.obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.obs.columns if c.startswith('L5')][0]

        if self.mode == 'E1E5b':
            GF1 = self.FREQ_DICT['E1']
            GF2 = self.FREQ_DICT['E5b']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.obs.columns if c.startswith('C7')][0]
            gps_l1_col = [c for c in self.obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.obs.columns if c.startswith('L7')][0]

        if self.mode == 'L1L5':
            GF1 = self.FREQ_DICT['L1']
            GF2 = self.FREQ_DICT['L5']
            agps, bgps = GF1 ** 2 / (GF1 ** 2 - GF2 ** 2), GF2 ** 2 / (GF1 ** 2 - GF2 ** 2)
            gps_c1_col = [c for c in self.obs.columns if c.startswith('C1')][0]
            gps_c2_col = [c for c in self.obs.columns if c.startswith('C5')][0]
            gps_l1_col = [c for c in self.obs.columns if c.startswith('L1')][0]
            gps_l2_col = [c for c in self.obs.columns if c.startswith('L5')][0]

        gps_data = (agps, bgps, gps_c1_col, gps_c2_col, gps_l1_col, gps_l2_col)

        self.obs = self.obs.copy()

        if 'pco_los_l1' not in self.obs.columns.tolist():
            self.obs.loc[:, 'pco_los_l1'] = 0.0
        if 'pco_los_l2' not in self.obs.columns.tolist():
            self.obs.loc[:, 'pco_los_l2'] = 0.0
        if 'pco_los_l5' not in self.obs.columns.tolist():
            self.obs.loc[:, 'pco_los_l5'] = 0.0

        return gps_data

    def extract_ambiguities(self, x, P, n_sats, base_dim=5):
        """
        Wyciąga wektor ambiguł N1, N2 oraz ich macierz kowariancji z pełnego stanu i macierzy P.

        Parameters:
            x        -- wektor stanu EKF (1D array)
            P        -- macierz kowariancji (2D array)
            n_sats   -- liczba satelitów (czyli bloków [I_s, N1_s, N2_s])
            base_dim -- liczba parametrów globalnych (default=5)

        Returns:
            N_vec    -- wektor [N1_1, N2_1, N1_2, N2_2, ..., N1_n, N2_n]
            P_N      -- macierz kowariancji odpowiadająca N_vec
        """
        idxs_N = []
        for i in range(n_sats):
            idx_N1 = base_dim + 3 * i + 1
            idx_N2 = base_dim + 3 * i + 2
            idxs_N.extend([idx_N1, idx_N2])

        N_vec = x[idxs_N]
        P_N = P[np.ix_(idxs_N, idxs_N)]

        return N_vec, P_N

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

    def run_filter(self, epochs=None, clk0=0.0, ref=None, flh=None, zwd0=0.0, trace_filter=False, reset_every=0):
        old_sats, all_epochs = self.init_filter(clk0=clk0, zwd0=zwd0)
        gps_data = self._prepare_obs()
        agps, bgps, gps_c1, gps_c2, gps_l1, gps_l2 = gps_data
        if epochs is None:
            epochs = all_epochs
        result = []
        obs_result =[]
        xyz = None
        conv_time = None
        T0 = all_epochs[0]
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)
                    reset_epoch = True
                    T0 = t
            gps_epoch = self.obs.loc[(slice(None), t), :].sort_values(by='sv')

            def safe_get(df, col, length=None):
                """Zwraca kolumnę jako numpy, a jeśli brak – wektor zer o podanej długości."""
                if col in df.columns:
                    return df[col].to_numpy()
                elif length is not None:
                    return np.zeros(length)
                else:
                    return np.zeros(len(df))

            # --- GPS ---
            gps_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', 'pco_los_l1', 'pco_los_l2',
                f'sat_pco_los_{self.mode[:2]}',
                f'sat_pco_los_{self.mode[2:]}',
                'phw', 'me_wet'
            ]
            gps_len = len(gps_epoch)
            (gps_clk, gps_tro, gps_ah_los, gps_dprel, gps_pco1, gps_pco2,
             sat_pco_L1, sat_pco_L2, phw, mwet) = [safe_get(gps_epoch, c, gps_len) for c in gps_cols]


            curr_sats = gps_epoch.index.get_level_values('sv').tolist()

            if curr_sats != old_sats:
                new_x, new_P, new_Q = self.rebuild_state(
                    self.ekf.x.copy(), self.ekf.P.copy(), self.ekf.Q.copy(),
                    old_sats, curr_sats
                )
                self.ekf.x, self.ekf.P, self.ekf.Q = new_x, new_P, new_Q
                self.ekf.dim_x = len(new_x)
                self.ekf.dim_z = 4 * len(curr_sats)
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)

            old_sats = curr_sats

            tides = gps_epoch['tides_los'].to_numpy()
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gps_p1_c1 = np.asarray(gps_epoch.get(f'OSB_{gps_c1}',0.0)) * 1e-09 * self.CLIGHT
            gps_p2_c2 = np.asarray(gps_epoch.get(f'OSB_{gps_c2}',0.0) )* 1e-09 * self.CLIGHT
            gps_pl1_l1 = np.asarray(gps_epoch.get(f'OSB_{gps_l1}',0.0)) * 1e-09 * self.CLIGHT
            gps_pl2_l2 = np.asarray(gps_epoch.get(f'OSB_{gps_l2}',0.0)) * 1e-09 * self.CLIGHT
            C1 = gps_epoch[
                     gps_c1].to_numpy() - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p1_c1 - tides
            C2 = gps_epoch[
                     gps_c2].to_numpy() - gps_pco2 + sat_pco_L2 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p2_c2 - tides
            L1 = gps_epoch[
                     gps_l1].to_numpy() - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl1_l1 - tides - (
                             self.CLIGHT / self.FREQ_DICT[self.mode[:2]]) * phw
            L2 = gps_epoch[
                     gps_l2].to_numpy() - gps_pco2 + sat_pco_L2 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl2_l2 - tides - (
                             self.CLIGHT / self.FREQ_DICT[self.mode[2:]]) * phw

            Z = np.vstack((C1, L1, C2, L2)).T.reshape(-1)

            gps_c1_mask = self.code_screening(x=self.ekf.x[:3], code_obs=C1, thr=30, satellites=gps_satellites[:, :3])
            gps_c2_mask = self.code_screening(x=self.ekf.x[:3], code_obs=C2, thr=30, satellites=gps_satellites[:, :3])
            R_diag = np.zeros(4*len(curr_sats))
            for k in range(len(curr_sats)):
                base = 4*k
                R_diag[base+0] = 1 if gps_c1_mask[k] else 1e12
                R_diag[base+1] = 1
                R_diag[base + 2] = 1 if gps_c2_mask[k] else 1e12
                R_diag[base + 3] = 1
            self.ekf.R = np.diag(R_diag)

            self.ekf.predict_update(z=Z,
                                    HJacobian=self.Hjacobian,
                                    args=(gps_satellites,),
                                    Hx=self.Hx,
                                    hx_args=(gps_satellites,))

            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time
            xyz = self.ekf.x[:3].copy()
            dtr = self.ekf.x[3].copy()
            ztd = self.ekf.x[4].copy()
            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
                result.append(pd.DataFrame(data={'de': enu[0], 'dn': enu[1], 'du': enu[2], 'dtr': dtr, 'ztd': ztd,
                                                 'x':xyz[0],'y':xyz[1],'z':xyz[2]},
                                           index=pd.DatetimeIndex(data=[t])))
            if trace_filter:
                print(result[-1])
        if conv_time is not None:
            if trace_filter:
                print('Convergence time < 5 mm: ', conv_time.total_seconds() / 60,
                      ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")
            ct = conv_time.total_seconds() / 3600
        else:
            ct = None
        output = pd.concat(result)
        output['ct_min'] = ct
        return output, self.obs, self.obs, ct


class PPPFilterMultiGNSSIonConst:
    """
    PPP Undifferenced uncombinad EKF estimator. Ionospheric constraints approach.
    """
    def __init__(self, config:PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, est_dcb=False, interval=0.5,
                 use_iono_rms=True, sigma_iono_0=1.1,
                 sigma_iono_end=2.3, t_end=30):
        """

        :param gps_obs: pd.DataFrame, gps observations
        :param gps_mode: str, gps signals
        :param gal_obs: pd.DataFrame, gal observations
        :param gal_mode: str, gal signals
        :param ekf: filterpy.ExtendedKalmanFilter instance
        :param pos0: np.ndarray, initial position
        :param tro: bool, estimate ZTD
        :param est_dcb: bool, estimate reciever DCB
        :param interval: float, interval of observations
        :param use_iono_rms: bool, use Ionospheric constraints
        :param sigma_iono_0: [float, int] ionospheric contraint error at first epoch
        :param sigma_iono_end:  [float, int] ionospheric constraint error at last epoch
        :param t_end: int, epoch number after which sigma_iono stops at sigma_iono_end
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
        self.CLIGHT = 299792458
        self.base_dim = 6 if tro else 5  # X,Y,Z,clk,ISB_gal,ZTD
        self.est_dcb = est_dcb
        self.pos0 = pos0
        self.interval = interval
        # --------
        self.gps_obs = self.gps_obs.dropna(subset=['ion'])
        self.gal_obs = self.gal_obs.dropna(subset=['ion'])
        if 'ion_rms' in self.gps_obs.columns:
            self.gps_obs = self.gps_obs.dropna(subset=['ion_rms'])
        if 'ion_rms' in self.gal_obs.columns:
            self.gal_obs = self.gal_obs.dropna(subset=['ion_rms'])
        self.use_iono_rms = use_iono_rms
        self.sigma_iono_0 = sigma_iono_0
        self.sigma_iono_end = sigma_iono_end
        self.t_end = t_end

    def dcb_param_indices(self, n_gps, n_gal):
        """
        Automatically find DCB indices in vector state
        :param n_gps:int, number of gps satellites
        :param n_gal:int, number of gal satellites
        :return: int
        """
        if not self.est_dcb:
            return []
        base = self.base_dim + 3 * (n_gps + n_gal)
        # 4 parametry: C1_GPS, C2_GPS, C1_GAL (E1), C2_GAL (E5a)
        return [base + i for i in range(4)]

    def extract_dcb(self, x, P, n_gps, n_gal):
        """
        Automatically extract DCB from x and P matrices
        :param x: np.ndarray, state vector
        :param P: np.ndarray a-priori error matrix
        :param n_gps: int, number of gps satellites
        :param n_gal: int, number of gal satellites
        :return: subset of x and P for DCBs
        """
        if not self.est_dcb:
            return None, None
        dcb_idx = self.dcb_param_indices(n_gps, n_gal)
        return x[dcb_idx], P[np.ix_(dcb_idx, dcb_idx)]

    def Hjacobian(self, x, gps_satellites, gal_satellites):
        """
        Hjacobian matrix
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :param gal_satellites: np.ndarray, gal satellite coordinates & mapping function
        :return: Hjacobian matrix
        """
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        pos0 = x[:3].copy()
        base_dim = self.base_dim
        dcb_block = 4 if self.est_dcb else 0
        dim = base_dim + 3 * (n_gps + n_gal) + dcb_block
        H = np.zeros((5 * (n_gps + n_gal), dim))
        # GPS
        C = self.CLIGHT
        F1_GPS = self.FREQ_DICT[self.gps_mode[:2]]
        F2_GPS = self.FREQ_DICT[self.gps_mode[2:]]
        L1 = C / F1_GPS
        L2 = C / F2_GPS
        MU1 = 1.0
        MU2 = (F1_GPS / F2_GPS) ** 2
        e_gps = ((gps_satellites[:, :3] - pos0).astype(np.float64, copy=False)) / \
                np.linalg.norm((gps_satellites[:, :3] - pos0).astype(np.float64, copy=False), axis=1)[:, None]
        m_wet_gps = gps_satellites[:, 3]
        # GALILEO

        F1_GAL = self.FREQ_DICT[self.gal_mode[:2] if self.gal_mode!='E5aE5b' else self.gal_mode[:3]]
        F2_GAL = self.FREQ_DICT[self.gal_mode[2:] if self.gal_mode!='E5aE5b' else self.gal_mode[3:]]
        E1 = C / F1_GAL
        E5a = C / F2_GAL
        MU1_GAL = 1.0
        MU2_GAL = (F1_GAL / F2_GAL) ** 2
        e_gal = ((gal_satellites[:, :3] - pos0).astype(np.float64, copy=False)) / \
                np.linalg.norm((gal_satellites[:, :3] - pos0).astype(np.float64, copy=False), axis=1)[:, None]
        m_wet_gal = gal_satellites[:, 3]

        COL_X, COL_Y, COL_Z, COL_CLK, COL_ISB, COL_ZTD = 0, 1, 2, 3, 4, 5

        for s in range(n_gps):
            row = 5 * s
            ex, ey, ez = -e_gps[s]
            mw = m_wet_gps[s]
            for i in range(4):
                H[row + i, [COL_X, COL_Y, COL_Z]] = [ex, ey, ez]
                H[row + i, COL_CLK] = 1.0
                H[row + i, COL_ZTD] = mw
            col_iono = base_dim + 3 * s
            col_N1 = col_iono + 1
            col_N2 = col_iono + 2
            H[row + 0, col_iono] = +MU1
            H[row + 1, col_iono] = -MU1
            H[row + 2, col_iono] = +MU2
            H[row + 3, col_iono] = -MU2
            H[row + 1, col_N1] = L1
            H[row + 3, col_N2] = L2
            if self.est_dcb:
                dcb_base = base_dim + 3 * (n_gps + n_gal)
                H[row + 0, dcb_base + 0] = 1.0  # DCB_C1_GPS
                H[row + 1, dcb_base + 0] = 1.0  # DCB_L1_GPS==C1
                H[row + 2, dcb_base + 1] = 1.0  # DCB_C2_GPS
                H[row + 3, dcb_base + 1] = 1.0  # DCB_L2_GPS==C2

                # ROZW tez na L1 i L2 ?
            H[row + 4, col_iono] = 1.0  # constraint

        for s in range(n_gal):
            row = 5 * (n_gps + s)
            ex, ey, ez = -e_gal[s]
            mw = m_wet_gal[s]
            for i in range(4):
                H[row + i, [COL_X, COL_Y, COL_Z]] = [ex, ey, ez]
                H[row + i, COL_CLK] = 1.0
                H[row + i, COL_ISB] = 1.0  # ISB Galileo
                H[row + i, COL_ZTD] = mw
            col_iono = base_dim + 3 * (n_gps + s)
            col_N1 = col_iono + 1
            col_N2 = col_iono + 2
            H[row + 0, col_iono] = +MU1_GAL
            H[row + 1, col_iono] = -MU1_GAL
            H[row + 2, col_iono] = +MU2_GAL
            H[row + 3, col_iono] = -MU2_GAL
            H[row + 1, col_N1] = E1
            H[row + 3, col_N2] = E5a
            if self.est_dcb:
                dcb_base = base_dim + 3 * (n_gps + n_gal)
                H[row + 0, dcb_base + 2] = 1.0  # DCB_C1_GAL
                H[row + 1, dcb_base + 2] = 1.0  # DCB_L1_GAL==C1
                H[row + 2, dcb_base + 3] = 1.0  # DCB_C2_GAL
                H[row + 3, dcb_base + 3] = 1.0  # DCB_C2_GAL==C2
            H[row + 4, col_iono] = 1.0
        return H

    def Hx(self, x, gps_satellites, gal_satellites):
        """
        Predicted observations vector
        :param x: np.ndarray, state vector
        :param gps_satellites: np.ndarray, gps satellite coordinates & mapping function
        :param gal_satellites: np.ndarray, gal satellite coordinates & mapping function
        :return: predicted obs matrix
        """
        C = self.CLIGHT
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        F1_GPS = self.FREQ_DICT[self.gps_mode[:2]]
        F2_GPS = self.FREQ_DICT[self.gps_mode[2:]]
        L1 = C / F1_GPS
        L2 = C / F2_GPS
        MU1 = 1.0
        MU2 = (F1_GPS / F2_GPS) ** 2
        F1_GAL = self.FREQ_DICT[self.gal_mode[:2] if self.gal_mode!='E5aE5b' else self.gal_mode[:3]]
        F2_GAL = self.FREQ_DICT[self.gal_mode[2:] if self.gal_mode!='E5aE5b' else self.gal_mode[3:]]
        E1 = C / F1_GAL
        E5a = C / F2_GAL
        MU1_GAL = 1.0
        MU2_GAL = (F1_GAL / F2_GAL) ** 2
        xr, yr, zr = x[0:3]
        clk = x[3]
        isb = x[4]
        zwd = x[5]
        dcb = x[-4:] if self.est_dcb else np.zeros(8)
        z_hat = np.empty(5 * (n_gps + n_gal))
        sat_xyz = gps_satellites[:, :3]
        m_wet = gps_satellites[:, 3]
        rho_vec = (sat_xyz - np.array([xr, yr, zr])).astype(np.float64, copy=False)
        rho = np.linalg.norm(rho_vec.astype(np.float64, copy=False), axis=1)
        for s in range(n_gps):
            i = self.base_dim + 3 * s
            I_s = x[i]
            N1_s = x[i + 1]
            N2_s = x[i + 2]
            geom = rho[s]
            mw = m_wet[s]
            z_hat[5 * s + 0] = geom + clk + mw * zwd + MU1 * I_s + dcb[0]
            z_hat[5 * s + 1] = geom + clk + mw * zwd - MU1 * I_s + L1 * N1_s
            z_hat[5 * s + 2] = geom + clk + mw * zwd + MU2 * I_s + dcb[1]
            z_hat[5 * s + 3] = geom + clk + mw * zwd - MU2 * I_s + L2 * N2_s
            z_hat[5 * s + 4] = I_s
        sat_xyz = gal_satellites[:, :3]
        m_wet = gal_satellites[:, 3]
        rho_vec = sat_xyz - np.array([xr, yr, zr])
        rho = np.linalg.norm(rho_vec.astype(np.float64, copy=False), axis=1)
        for s in range(n_gal):
            i = self.base_dim + 3 * (n_gps + s)
            I_s = x[i]
            N1_s = x[i + 1]
            N2_s = x[i + 2]
            geom = rho[s]
            mw = m_wet[s]
            row = 5 * (n_gps + s)
            z_hat[row + 0] = geom + clk + isb + mw * zwd + MU1_GAL * I_s + dcb[2]
            z_hat[row + 1] = geom + clk + isb + mw * zwd - MU1_GAL * I_s + E1 * N1_s
            z_hat[row + 2] = geom + clk + isb + mw * zwd + MU2_GAL * I_s + dcb[3]
            z_hat[row + 3] = geom + clk + isb + mw * zwd - MU2_GAL * I_s + E5a * N2_s
            z_hat[row + 4] = I_s
        return z_hat

    def _prepare_obs(self):
        def find_cols(obs, mode, prefix_map):

            if mode =='E5aE5b':
                F1 = self.FREQ_DICT[mode[:3]]
                F2 = self.FREQ_DICT[mode[3:]]
            else:
                F1 = self.FREQ_DICT[mode[:2]]
                F2 = self.FREQ_DICT[mode[2:]]

            a = F1 ** 2 / (F1 ** 2 - F2 ** 2)
            b = F2 ** 2 / (F1 ** 2 - F2 ** 2)
            cols = {}
            for typ, prefix in prefix_map.items():
                for c in obs.columns:
                    if c.startswith(prefix):
                        cols[typ] = c

            return a, b, cols
        if self.gps_mode in ['L1L2']:
            gps_map = {'C1': 'C1', 'C2': 'C2', 'L1': 'L1', 'L2': 'L2'}
        elif self.gps_mode in ['L1L5']:
            gps_map = {'C1': 'C1', 'C2': 'C5', 'L1': 'L1', 'L2': 'L5'}
        elif self.gps_mode in ['L2L5']:
            gps_map = {'C1': 'C2', 'C2': 'C5', 'L1': 'L2', 'L2': 'L5'}
        if self.gal_mode in ['E1E5a']:
            gal_map = {'C1': 'C1', 'C2': 'C5', 'L1': 'L1', 'L2': 'L5'}
        elif self.gal_mode in ['E1E5b']:
            gal_map = {'C1': 'C1', 'C2': 'C7', 'L1': 'L1', 'L2': 'L7'}
        elif self.gal_mode in ['E5aE5b']:
            gal_map = {'C1': 'C5', 'C2': 'C7', 'L1': 'L5', 'L2': 'L7'}
        agps, bgps, gps_cols = find_cols(self.gps_obs, self.gps_mode, gps_map)
        agal, bgal, gal_cols = find_cols(self.gal_obs, self.gal_mode, gal_map)
        return (agps, bgps, gps_cols), (agal, bgal, gal_cols)

    def init_filter(self, clk0=0.0, zwd0=0.0):
        """
        Initialization of EKF for PPP estimation
        :param clk0: [int, float] initial clock value
        :param zwd0: [int, float] initial ZTD value
        """
        all_times = sorted(set(self.gps_obs.index.get_level_values('time').unique()) &
                           set(self.gal_obs.index.get_level_values('time').unique()))
        first_ep = all_times[0]
        gps_sats = self.gps_obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()
        gal_sats = self.gal_obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()
        n_gps = len(gps_sats)
        n_gal = len(gal_sats)
        base_dim = self.base_dim
        dcb_block = 4 if self.est_dcb else 0
        dim_x = base_dim + 3 * (n_gps + n_gal) + dcb_block
        dim_z = 5 * (n_gps + n_gal)
        x0 = np.zeros(dim_x)
        x0[0:3] = self.pos0
        x0[3] = clk0
        x0[4] = 0.0  # ISB GALILEO
        x0[5] = zwd0
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 100
        self.ekf.P[3, 3] = self.cfg.p_dt#1e9 # clk
        self.ekf.P[4, 4] = self.cfg.p_isb#100 # isb
        self.ekf.P[5, 5] = self.cfg.p_tro#2.0 # tro
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt #1e9 # clk
        self.ekf.Q[4, 4] = self.cfg.q_isb#1.0 # isb
        self.ekf.Q[5, 5] = self.cfg.q_tro#0.025 # tro
        self.ekf.F = np.eye(dim_x)
        # self.ekf.F[3, 3] = 0.0

        P4_gps = self.gps_obs.loc[(slice(None), first_ep), 'ion'].to_numpy()
        P4_gal = self.gal_obs.loc[(slice(None), first_ep), 'ion'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 3 * i
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gps[i]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono#1.5
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono * (self.interval * 60) / 3600 #2.0 * (self.interval * 60) / 3600
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb#1e6
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb#1e6
        for j, sv in enumerate(gal_sats):
            k = n_gps + j
            idx_I = base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gal[j]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono#1.5
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono * (self.interval * 60) / 3600
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb
        if self.est_dcb:
            dcb_indices = self.dcb_param_indices(n_gps, n_gal)
            for idx in dcb_indices:
                self.ekf.P[idx, idx] = self.cfg.p_dcb#1e2
                self.ekf.Q[idx, idx] = self.cfg.q_dcb
        # zakładam: dcb_block=4 (zalecane; patrz pkt 2)
        base = self.base_dim + 3 * (n_gps + n_gal)
        idx_c1gps = base + 0  # DCB_C1_GPS
        idx_c1gal = base + 2
        self.ekf.x[idx_c1gps] = 0.0
        self.ekf.P[idx_c1gps, idx_c1gps] = 1e-12
        self.ekf.Q[idx_c1gps, idx_c1gps] = 0.0

        self.ekf.x[idx_c1gal] = 0.0
        self.ekf.P[idx_c1gal, idx_c1gal] = 1e-12
        self.ekf.Q[idx_c1gal, idx_c1gal] = 0.0

        return gps_sats, gal_sats, all_times

    def reset_filter(self, epoch, clk0=0.0, zwd0=0.0):
        """
        reinitialize (reset) Kalman filter. Init filter function but based on given epoch
        """
        gps_sats = self.gps_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        gal_sats = self.gal_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        n_gps = len(gps_sats)
        n_gal = len(gal_sats)
        base_dim = self.base_dim
        dcb_block = 4 if self.est_dcb else 0
        dim_x = base_dim + 3 * (n_gps + n_gal) + dcb_block
        dim_z = 5 * (n_gps + n_gal)
        x0 = np.zeros(dim_x)
        x0[0:3] = self.pos0
        x0[3] = clk0
        x0[4] = 0.0  # ISB GALILEO
        x0[5] = zwd0
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 100
        self.ekf.P[3, 3] = self.cfg.p_dt  # 1e9 # clk
        self.ekf.P[4, 4] = self.cfg.p_isb  # 100 # isb
        self.ekf.P[5, 5] = self.cfg.p_tro  # 2.0 # tro
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt  # 1e9 # clk
        self.ekf.Q[4, 4] = self.cfg.q_isb  # 1.0 # isb
        self.ekf.Q[5, 5] = self.cfg.q_tro  # 0.025 # tro
        self.ekf.F = np.eye(dim_x)
        # self.ekf.F[3, 3] = 0.0

        P4_gps = self.gps_obs.loc[(slice(None), epoch), 'ion'].to_numpy()
        P4_gal = self.gal_obs.loc[(slice(None), epoch), 'ion'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 3 * i
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gps[i]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono  # 1.5
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono * (
                        self.interval * 60) / 3600  # 2.0 * (self.interval * 60) / 3600
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb  # 1e6
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb  # 1e6
        for j, sv in enumerate(gal_sats):
            k = n_gps + j
            idx_I = base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gal[j]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono  # 1.5
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono * (self.interval * 60) / 3600
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb
        if self.est_dcb:
            dcb_indices = self.dcb_param_indices(n_gps, n_gal)
            for idx in dcb_indices:
                self.ekf.P[idx, idx] = self.cfg.p_dcb  # 1e2
                self.ekf.Q[idx, idx] = self.cfg.q_dcb
        # zakładam: dcb_block=4 (zalecane; patrz pkt 2)
        base = self.base_dim + 3 * (n_gps + n_gal)
        idx_c1gps = base + 0  # DCB_C1_GPS
        idx_c1gal = base + 2
        self.ekf.x[idx_c1gps] = 0.0
        self.ekf.P[idx_c1gps, idx_c1gps] = 1e-12
        self.ekf.Q[idx_c1gps, idx_c1gps] = 0.0

        self.ekf.x[idx_c1gal] = 0.0
        self.ekf.P[idx_c1gal, idx_c1gal] = 1e-12
        self.ekf.Q[idx_c1gal, idx_c1gal] = 0.0

        return gps_sats, gal_sats

    def rebuild_state(self, x_old, P_old, Q_old, prev_gps, prev_gal, curr_gps, curr_gal):
        """
        Rebuilding of state vector and KF matrices after satellite visibility change
        :param x_old: np.ndarray, old state vector
        :param P_old: np.ndarray, old prior error matrix
        :param Q_old: np.ndarray, old process noise matrix
        :param prev_gps: list, previous GPS satellites
        :param prev_gal: list, previous GAL satellites
        :param curr_gps: list, current GPS satellites
        :param curr_gal: list, current GAL satellites
        :return: new X, new P, Q
        """
        base_dim = self.base_dim
        n_prev = len(prev_gps) + len(prev_gal)
        n_curr = len(curr_gps) + len(curr_gal)
        dcb_block = 4 if self.est_dcb else 0
        old_dim = base_dim + 3 * n_prev + dcb_block
        new_dim = base_dim + 3 * n_curr + dcb_block
        x_new = np.zeros(new_dim)
        x_new[:base_dim] = x_old[:base_dim]
        P_new = np.zeros((new_dim, new_dim))
        P_new[:base_dim, :base_dim] = P_old[:base_dim, :base_dim]
        Q_new = np.zeros((new_dim, new_dim))
        Q_new[:base_dim, :base_dim] = Q_old[:base_dim, :base_dim]

        def tag(system, sv):
            return f"{system}:{sv}"

        prev_all = [tag("G", sv) for sv in prev_gps] + [tag("E", sv) for sv in prev_gal]
        curr_all = [tag("G", sv) for sv in curr_gps] + [tag("E", sv) for sv in curr_gal]
        common = set(prev_all) & set(curr_all)
        for prn in common:
            i_old = prev_all.index(prn)
            i_new = curr_all.index(prn)
            for k in range(3):
                xo = base_dim + 3 * i_old + k
                xn = base_dim + 3 * i_new + k
                x_new[xn] = x_old[xo]
                P_new[xn, :base_dim] = P_old[xo, :base_dim]
                P_new[:base_dim, xn] = P_old[:base_dim, xo]
                Q_new[xn, :base_dim] = Q_old[xo, :base_dim]
                Q_new[:base_dim, xn] = Q_old[:base_dim, xo]
                for prn2 in common:
                    j_old = prev_all.index(prn2)
                    j_new = curr_all.index(prn2)
                    for l in range(3):
                        yo = base_dim + 3 * j_old + l
                        yn = base_dim + 3 * j_new + l
                        P_new[xn, yn] = P_old[xo, yo]
                        Q_new[xn, yn] = Q_old[xo, yo]
        # Nowe satelity
        new_only = set(curr_all) - set(prev_all)
        for prn in new_only:
            j = curr_all.index(prn)
            idx_I = base_dim + 3 * j
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            P_new[idx_I, idx_I] =self.cfg.p_iono# 10.0
            Q_new[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600#5.0
            P_new[idx_N1, idx_N1] = self.cfg.p_amb#1e6
            P_new[idx_N2, idx_N2] = self.cfg.p_amb#1e6
        # DCB przeniesienie
        if self.est_dcb:
            dcb_idx_new = self.dcb_param_indices(len(curr_gps), len(curr_gal))
            dcb_idx_old = self.dcb_param_indices(len(prev_gps), len(prev_gal))
            for i, idx in enumerate(dcb_idx_new):
                if i < len(dcb_idx_old):
                    x_new[idx] = x_old[dcb_idx_old[i]]
                    P_new[idx, idx] = P_old[dcb_idx_old[i], dcb_idx_old[i]]
                    Q_new[idx, idx] = Q_old[dcb_idx_old[i], dcb_idx_old[i]]
                else:
                    P_new[idx, idx] = self.cfg.p_dcb#1e2
                    Q_new[idx, idx] = self.cfg.q_dcb#1e-4
        Q_new[3, 3] = 9e9
        # zakładam: dcb_block=4 (zalecane; patrz pkt 2)


        return x_new, P_new, Q_new

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
        dist = dist.astype(np.float64, copy=False)
        prefit = code_obs - dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad > n_sat / 2:
            mask = np.ones(n_sat, dtype=bool)
        return mask, prefit

    def phase_residuals_screening(self, sat_list, phase_residuals_dict, num, thr=1, sys='G', len_gps=None, freq='n1'):
        """
        Screening of phase residuals for outliers based of prefit differences between epochs
        :param sat_list: list, satellites list
        :param phase_residuals_dict: dict, phase observations prefit residuals
        :param num: int, epoch number
        :param thr: thershold for filtering
        :param sys: str, sys
        :param len_gps: int, number of GPS observations
        :param: freq: str, signal frequency
        :return: Modify state vector and prior matrix (x & P)
        """
        if sys == 'E':
            assert len_gps is not None
        prefit_diff = []
        N = 5
        for sv in sat_list:
            # Sprawdź, czy dla każdej z poprzednich N epok satelita ma residual
            if all(
                    phase_residuals_dict.get(num - offset - 1, {}).get(sv) is not None
                    for offset in range(N)
            ) and phase_residuals_dict[num].get(sv) is not None:
                prev_residual = phase_residuals_dict[num - 1][sv]
                current_residual = phase_residuals_dict[num][sv]
                prefit_diff.append(current_residual - prev_residual)
        median_prefit_diff = np.median(prefit_diff) if len(prefit_diff) > 0 else 0
        for n_item, (sv, residual) in enumerate(zip(sat_list, prefit_diff)):

            if np.abs(residual - median_prefit_diff) > thr:

                outlier_idx = sat_list.index(sv)

                if sys == 'E':
                    outlier_idx += len_gps
                base = 3 * outlier_idx
                if freq == 'n1':
                    n1 = base + 1
                elif freq == 'n2':
                    n1 = base + 2

                self.ekf.x[self.base_dim + n1] = 0.0
                self.ekf.P[
                    self.base_dim + n1, self.base_dim + n1] = 1e6
                self.ekf.Q[self.base_dim + n1, self.base_dim + n1] = 0.0

    def run_filter(self, clk0=0.0, ref=None, flh=None, zwd0=0.0, trace_filter=False, reset_every=0):
        """
        Kalman Filter PPP-UDUC Float estimation
        :param clk0: [float, int] initial clock value
        :param ref: np.ndarray, reference coordinates in ECEF
        :param flh: np.ndarray, reference coordinates in BLH
        :param zwd0: [float, int], initial ZTD value
        :param trace_filter:bool, whether to trace filter
        :param reset_every: int, number of epochs to reset filter
        :return: tuple(pd.Dataframe, pd.Dataframe, pd.Dataframe, float), solution, gps&gal residuals, convergence time
        """
        gps_sats, gal_sats, epochs = self.init_filter(clk0=clk0, zwd0=zwd0)
        (agps, bgps, gps_cols), (agal, bgal, gal_cols) = self._prepare_obs()
        # sigma_iono_0=1.1, sigma_iono_end=3,t_end = 30 for iono rms
        # 0.01 2.5 for no iono rms
        result = []
        result_gps = []
        result_gal = []

        t_end = self.t_end / self.interval

        sigma_iono_t = self.sigma_iono_0 + (self.sigma_iono_end - self.sigma_iono_0) / t_end * np.arange(t_end)
        xyz = None
        conv_time = None
        T0 = epochs[0]
        prt_gps_l1 = {}
        prt_gps_l2 = {}
        prt_gal_l1 = {}
        prt_gal_l2 = {}
        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    print('RESET: ', t)
                    gps_sats, gal_sats = self.reset_filter(epoch=t)
                    reset_epoch = True
                    T0 = t
                    # n_i = 0

            gps_epoch = self.gps_obs.loc[(slice(None), t), :].sort_values(by='sv')
            gal_epoch = self.gal_obs.loc[(slice(None), t), :].sort_values(by='sv')
            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_gal_sats = gal_epoch.index.get_level_values('sv').tolist()
            if (curr_gps_sats != gps_sats) or (curr_gal_sats != gal_sats):
                self.ekf.x, self.ekf.P, self.ekf.Q = self.rebuild_state(
                    self.ekf.x, self.ekf.P, self.ekf.Q,
                    gps_sats, gal_sats, curr_gps_sats, curr_gal_sats
                )
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = 5 * (len(curr_gps_sats) + len(curr_gal_sats))
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)

                n_gps = len(gps_epoch); n_gal = len(gal_epoch)
                base = self.base_dim + 3 * (n_gps + n_gal)
                idx_c1gps = base + 0  # DCB_C1_GPS
                idx_c1gal = base + 2
                self.ekf.x[idx_c1gps] = 0.0
                self.ekf.P[idx_c1gps, idx_c1gps] = 1e-12
                self.ekf.Q[idx_c1gps, idx_c1gps] = 0.0

                self.ekf.x[idx_c1gal] = 0.0
                self.ekf.P[idx_c1gal, idx_c1gal] = 1e-12
                self.ekf.Q[idx_c1gal, idx_c1gal] = 0.0
            gps_sats = curr_gps_sats
            gal_sats = curr_gal_sats
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            gal_satellites = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            # ---- Obserwacje i constrainty ----

            def get_obs(epoch, cols, sys):
                if sys == "G":
                    _mode = self.gps_mode
                    sat_pco1_col = f"sat_pco_los_{_mode[:2]}"
                    sat_pco2_col = f"sat_pco_los_{_mode[2:]}"
                    pco2_col = "pco_los_l2"
                    f1 = self.FREQ_DICT[_mode[:2]]
                    f2 = self.FREQ_DICT[_mode[2:]]
                else:
                    _mode = self.gal_mode
                    if _mode == 'E5aE5b':
                        sat_pco1_col = f"sat_pco_los_{_mode[:3]}"
                        sat_pco2_col = f"sat_pco_los_{_mode[3:]}"
                        f1 = self.FREQ_DICT[_mode[:3]]
                        f2 = self.FREQ_DICT[_mode[3:]]
                    else:
                        sat_pco1_col = f"sat_pco_los_{_mode[:2]}"
                        sat_pco2_col = f"sat_pco_los_{_mode[2:]}"
                        f1 = self.FREQ_DICT[_mode[:2]]
                        f2 = self.FREQ_DICT[_mode[2:]]
                    pco2_col = "pco_los_l2"

                # helper: fast column -> numpy (no copy if possible)
                def col_np(name, default=0.0):
                    if name in epoch.columns:
                        return epoch[name].to_numpy(copy=False)
                    if np.isscalar(default):
                        return default
                    return np.asarray(default)

                clk = col_np("clk")
                tro = col_np("tro")
                ah = col_np("ah_los")
                dprel = col_np("dprel")
                tides = col_np("tides_los")
                phw = col_np("phw")
                ion = col_np("ion")

                pco1 = col_np("pco_los_l1", 0.0)
                pco2 = col_np(pco2_col, 0.0)

                sat_pco1 = col_np(sat_pco1_col, 0.0)
                sat_pco2 = col_np(sat_pco2_col, 0.0)

                # OSB: KONIECZNIE jako numpy, nie Series
                osb_c1 = col_np(f"OSB_{cols['C1']}", 0.0)
                osb_c2 = col_np(f"OSB_{cols['C2']}", 0.0)
                osb_l1 = col_np(f"OSB_{cols['L1']}", 0.0)
                osb_l2 = col_np(f"OSB_{cols['L2']}", 0.0)

                # stałe tylko raz
                cl = self.CLIGHT
                clk_cl = clk * cl

                osb_c1 = osb_c1 * 1e-09 * cl
                osb_c2 = osb_c2 * 1e-09 * cl
                osb_l1 = osb_l1 * 1e-09 * cl
                osb_l2 = osb_l2 * 1e-09 * cl

                C1 = epoch[cols["C1"]].to_numpy(copy=False)
                C2 = epoch[cols["C2"]].to_numpy(copy=False)
                L1 = epoch[cols["L1"]].to_numpy(copy=False)
                L2 = epoch[cols["L2"]].to_numpy(copy=False)

                # precompute phw factors

                phw1 = (cl / f1) * phw
                phw2 = (cl / f2) * phw

                common = (-pco1 + sat_pco1 + clk_cl - tro - ah - dprel - tides)

                c1 = C1 + common - osb_c1
                c2 = C2 + (-pco2 + sat_pco2 + clk_cl - tro - ah - dprel - tides) - osb_c2

                l1 = L1 + common - phw1 - osb_l1
                l2 = L2 + (-pco2 + sat_pco2 + clk_cl - tro - ah - dprel - tides) - phw2 - osb_l2

                return c1, l1, c2, l2, ion

            C1, L1, C2, L2, IONO = get_obs(gps_epoch, gps_cols, 'G')
            EC1, EL1, EC2, EL2, EIONO = get_obs(gal_epoch, gal_cols, 'E')
            Z = np.hstack([np.vstack((C1, L1, C2, L2, IONO)).T.reshape(-1),
                           np.vstack((EC1, EL1, EC2, EL2, EIONO)).T.reshape(-1)])

            dist_gps = calculate_distance(gps_satellites[:, :3].copy(), self.ekf.x[:3].copy())
            dist_gal = calculate_distance(gal_satellites[:, :3].copy(), self.ekf.x[:3].copy())
            prefit_gps_l1 = L1 - dist_gps
            prefit_gps_l2 = L2 - dist_gps
            prt_gps_l1[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_l1)}
            prt_gps_l2[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_l2)}
            if num >= 60:
                self.phase_residuals_screening(sat_list=curr_gps_sats, phase_residuals_dict=prt_gps_l1, num=num,
                                               freq='n1')
                self.phase_residuals_screening(sat_list=curr_gps_sats, phase_residuals_dict=prt_gps_l2, num=num,
                                               freq='n2')

            prefit_gal_l1 = EL1 - dist_gal
            prefit_gal_l2 = EL2 - dist_gal
            prt_gal_l1[num] = {sv: r for sv, r in zip(curr_gal_sats, prefit_gal_l1)}
            prt_gal_l2[num] = {sv: r for sv, r in zip(curr_gal_sats, prefit_gal_l2)}
            if num >= 60:
                self.phase_residuals_screening(sat_list=curr_gal_sats, phase_residuals_dict=prt_gal_l1, num=num,
                                               sys='E', len_gps=len(curr_gps_sats), freq='n1')
                self.phase_residuals_screening(sat_list=curr_gal_sats, phase_residuals_dict=prt_gal_l2, num=num,
                                               sys='E', len_gps=len(curr_gps_sats), freq='n2')
            if num < t_end:
                sigma_ion = sigma_iono_t[num]
            else:
                sigma_ion = sigma_iono_t[-1]

            ev_gps = gps_epoch['ev'].to_numpy()
            ev_gal = gal_epoch['ev'].to_numpy()
            sigma_code_gps = 0.3 + 0.0025 / np.sin(np.deg2rad(ev_gps))  # **2
            sigma_phase_gps = 1e-4 + 0.0003 / np.sin(np.deg2rad(ev_gps))  # **2
            if self.use_iono_rms:
                sigma_ion_gps = gps_epoch[
                                    'ion_rms'].to_numpy() * sigma_ion  # sigma_ion/np.sin(np.deg2rad(ev_gps)) #np.full_like(ev_gps, sigma_ion)
                sigma_ion_gal = gal_epoch[
                                    'ion_rms'].to_numpy() * sigma_ion  # sigma_ion/np.sin(np.deg2rad(ev_gal)) #np.full_like(ev_gal, sigma_ion)
            else:
                sigma_ion_gps = np.full_like(ev_gps, sigma_ion)
                sigma_ion_gal = np.full_like(ev_gal, sigma_ion)

            sigma_code_gal = 0.3 + 0.0025 / np.sin(np.deg2rad(ev_gal))  # **2
            sigma_phase_gal = 1e-4 + 0.0003 / np.sin(np.deg2rad(ev_gal))  # **2

            W = []
            gps_c1_mask, prefit_c1 = self.code_screening(x=self.ekf.x[:3], satellites=gps_satellites[:, :3],
                                                         code_obs=C1, thr=10)
            gps_c2_mask, prefit_c2 = self.code_screening(x=self.ekf.x[:3], satellites=gps_satellites[:, :3],
                                                         code_obs=C2, thr=10)

            gal_c1_mask, prefit_e1 = self.code_screening(x=self.ekf.x[:3], satellites=gal_satellites[:, :3],
                                                         code_obs=EC1, thr=10)
            gal_c2_mask, prefit_e2 = self.code_screening(x=self.ekf.x[:3], satellites=gal_satellites[:, :3],
                                                         code_obs=EC2, thr=10)

            for i, (sc, sp, si) in enumerate(zip(sigma_code_gps, sigma_phase_gps, sigma_ion_gps)):
                sc1 = sc
                sc2 = sc
                if not gps_c1_mask[i]:
                    sc1 = 1e12
                if not gps_c2_mask[i]:
                    sc2 = 1e12
                W.extend([sc1, sp, sc2, sp, si])

            for i, (sc, sp, si) in enumerate(zip(sigma_code_gal, sigma_phase_gal, sigma_ion_gal)):
                sc1 = sc
                sc2 = sc
                if not gal_c1_mask[i]:
                    sc1 = 1e12
                if not gal_c2_mask[i]:
                    sc2 = 1e12
                W.extend([sc1, sp, sc2, sp, si])

            W = np.array(W, dtype=np.float32)

            self.ekf.R = np.diag(W)

            self.ekf.predict_update(z=Z,
                                    HJacobian=self.Hjacobian,
                                    args=(gps_satellites, gal_satellites),
                                    Hx=self.Hx,
                                    hx_args=(gps_satellites, gal_satellites))

            dtr = self.ekf.x[3]
            isb = self.ekf.x[4]
            ztd = self.ekf.x[5]

            # ---- convergence time measure
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time

            # --- data collecting
            xyz = self.ekf.x[:3].copy()
            dcb = self.ekf.x[-4:]
            x_gps = self.ekf.x[self.base_dim:self.base_dim + 3 * len(gps_epoch)]
            x_gal = self.ekf.x[self.base_dim + 3 * len(gps_epoch):self.base_dim + 3 * (len(gps_epoch) + len(gal_epoch))]
            stec_gps = x_gps[::3]
            stec_gal = x_gal[::3]

            n1_gps = x_gps[1::3]
            n2_gps = x_gps[2::3]

            n1_gal = x_gal[1::3]
            n2_gal = x_gal[2::3]



            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
            else:
                enu = np.array([0.0, 0.0, 0.0])
            result.append({'time': t, 'de': enu[0], 'dn': enu[1], 'du': enu[2],
                           'dtr': dtr, 'isb': isb, 'ztd': ztd, 'dcb_gps_c1': dcb[0], 'dcb_gps_c2': dcb[1],
                           'dcb_gal_c1': dcb[2], 'dcb_gal_c2': dcb[3],  'xr': self.ekf.x[0], 'yr': self.ekf.x[1],
                           'zr': self.ekf.x[2]})

            y_gps = self.ekf.y[:5 * len(curr_gps_sats)].copy()
            v_gps_c1 = y_gps[0::5]
            v_gps_l1 = y_gps[1::5]
            v_gps_c2 = y_gps[2::5]
            v_gps_l2 = y_gps[3::5]

            y_gal = self.ekf.y[5 * len(curr_gps_sats):].copy()
            v_gal_c1 = y_gal[0::5]
            v_gal_l1 = y_gal[1::5]
            v_gal_c2 = y_gal[2::5]
            v_gal_l2 = y_gal[3::5]

            gps_epoch = gps_epoch.assign(
                Idelay=stec_gps,
                n1=n1_gps,
                C1=C1, L1=L1,
                C2=C2, L2=L2,
                n2=n2_gps,
                vc1=v_gps_c1, vl1=v_gps_l1,
                vc2=v_gps_c2, vl2=v_gps_l2,
            )

            gal_epoch = gal_epoch.assign(
                Idelay=stec_gal,
                n1=n1_gal,
                n2=n2_gal,
                vc1=v_gal_c1, vl1=v_gal_l1,
                vc2=v_gal_c2, vl2=v_gal_l2,
            )

            result_gps.append(gps_epoch)
            result_gal.append(gal_epoch)



            if trace_filter:
                if reset_epoch:
                    print('===' * 30, '  RESET  ', '===' * 30)
                print(result[-1])
                print('===' * 30)
                print('\n\n')

        df_result = pd.DataFrame(result)
        df_result = df_result.set_index(['time'])
        df_obs_gps = pd.concat(result_gps)
        df_obs_gal = pd.concat(result_gal)
        if conv_time is not None:
            if trace_filter:
                print('Convergence time < 5 mm: ', conv_time.total_seconds() / 60,
                      ' [min] ', conv_time.total_seconds() / 3600, ' [h]' if conv_time is not None else "Not reached")

            ct = conv_time.total_seconds() / 3600
        else:
            ct = None
        return df_result, df_obs_gps, df_obs_gal, ct


