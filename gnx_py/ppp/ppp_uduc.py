"""Undifferenced uncombined PPP filter classes.

This module holds the main uncombined PPP branches. State vectors keep code,
phase, ionosphere and ambiguity terms in the filter instead of forming
ionosphere-free observables up front. The file includes active generic G/E/C
paths, older GPS/Galileo reference paths, single-system filters, and
ionospheric-constraint variants. Because these filters encode the numerical PPP
models directly, changes should be backed by focused regression tests.
"""

import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter
from typing import Sequence

from ..conversion import ecef_to_enu
from ..utils import calculate_distance
from ..configuration import PPPConfig
from ..biases import (
    bias_column_m,
    has_satellite_osb,
    osb_m,
    split_dual_code_dsb_corrections_m,
)
from ..gnss import MODE_SIGNALS as GNSS_MODE_SIGNALS, SIGNALS, frequency_hz
from .ppp_helpers import (
    phase_residuals_outliers,
    trace_epoch_summary as ppp_trace_epoch,
    trace_message as ppp_trace,
)
from .pppar import (
    PPPARSettings,
    advance_arc_age,
    apply_indexed_uncombined_pppar,
    apply_uncombined_pppar,
    arc_age_array,
    pppar_diagnostic_columns,
)

FREQ_DICT_ALL = {name: spec.frequency_hz for name, spec in SIGNALS.items()}

def _unit_vectors_and_ranges(sat_xyz: np.ndarray, rec_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return line-of-sight design vectors and geometric ranges."""
    rho_vec = sat_xyz - rec_xyz
    rho = np.linalg.norm(rho_vec, axis=1)
    e = np.zeros_like(rho_vec, dtype=float)
    valid = rho > 0.0
    e[valid] = -rho_vec[valid] / rho[valid, None]
    return e, rho


def _covariance_measurement_update(P: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply a covariance-only EKF measurement update for AR ratio covariance."""
    if H.size == 0:
        return P
    PHt = P @ H.T
    S = H @ PHt + R
    try:
        K = np.linalg.solve(S.T, PHt.T).T
    except np.linalg.LinAlgError:
        return P
    I = np.eye(P.shape[0], dtype=float)
    KH = K @ H
    P_new = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
    return 0.5 * (P_new + P_new.T)


def _reset_covariance_states(P: np.ndarray, indices: Sequence[int], variance: float) -> None:
    """Reset covariance rows/columns for selected state indices in place."""
    for idx in indices:
        if idx < 0 or idx >= P.shape[0]:
            continue
        P[idx, :] = 0.0
        P[:, idx] = 0.0
        P[idx, idx] = variance


def _iono_constraint_sigma(
    epoch: pd.DataFrame,
    num: int,
    interval: float,
    use_iono_rms: bool,
    sigma_iono_0: float,
    sigma_iono_end: float,
    t_end: int,
) -> np.ndarray:
    """Return per-satellite ionospheric constraint sigma for one epoch.

    The sigma ramps from ``sigma_iono_0`` to ``sigma_iono_end`` over ``t_end``
    minutes/epochs as encoded by the existing model and can be scaled by
    ``ion_rms`` when available.
    """
    interval = max(float(interval), 1e-12)
    sigma_iono_0 = 1.1 if sigma_iono_0 is None else sigma_iono_0
    sigma_iono_end = 3.0 if sigma_iono_end is None else sigma_iono_end
    t_end = 30 if t_end is None else t_end
    t_end_epochs = max(float(t_end) / interval, 1.0)
    frac = min(float(num), t_end_epochs) / t_end_epochs
    sigma = float(sigma_iono_0) + (float(sigma_iono_end) - float(sigma_iono_0)) * frac

    if use_iono_rms and 'ion_rms' in epoch.columns:
        ion_sigma = epoch['ion_rms'].to_numpy(copy=False).astype(float, copy=False) * sigma
        ion_sigma = np.where(np.isfinite(ion_sigma), ion_sigma, float(sigma_iono_end))
    else:
        ion_sigma = np.full(len(epoch), sigma, dtype=float)

    return np.clip(np.nan_to_num(ion_sigma, nan=1e12, posinf=1e12, neginf=1e12), 1e-6, 1e12)


MODE_SIGNALS = GNSS_MODE_SIGNALS

SIGNAL_COLUMNS = {
    name: (spec.code_prefix, spec.phase_prefix)
    for name, spec in SIGNALS.items()
}


def _mode_signals(mode: str) -> tuple[str, ...]:
    """Return configured signal names for an uncombined mode."""
    try:
        return MODE_SIGNALS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported uncombined mode: {mode}") from exc


def _dual_mode_signals(mode: str) -> tuple[str, str]:
    """Return the two signal names for a dual-frequency uncombined mode."""
    signals = _mode_signals(mode)
    if len(signals) != 2:
        raise ValueError(f"Mode {mode} is not dual-frequency.")
    return signals


def _first_prefixed_column(obs: pd.DataFrame, prefix: str) -> str:
    """Find the first observation column matching a signal prefix."""
    for col in obs.columns:
        if col.startswith(prefix):
            return col
    raise IndexError(f"Lack of {prefix} observations.")


def _dual_mode_columns(obs: pd.DataFrame, mode: str):
    """Resolve dual-frequency IF coefficients and observation columns."""
    s1, s2 = _dual_mode_signals(mode)
    f1 = frequency_hz(s1)
    f2 = frequency_hz(s2)
    a = f1 ** 2 / (f1 ** 2 - f2 ** 2)
    b = f2 ** 2 / (f1 ** 2 - f2 ** 2)
    c1_prefix, l1_prefix = SIGNAL_COLUMNS[s1]
    c2_prefix, l2_prefix = SIGNAL_COLUMNS[s2]
    return (
        a,
        b,
        _first_prefixed_column(obs, c1_prefix),
        _first_prefixed_column(obs, c2_prefix),
        _first_prefixed_column(obs, l1_prefix),
        _first_prefixed_column(obs, l2_prefix),
    )


def _p4_to_iono_state(p4: np.ndarray, mode: str) -> np.ndarray:
    """Convert geometry-free code combination to the modeled ionosphere state."""
    sig1, sig2 = _dual_mode_signals(mode)
    f1 = frequency_hz(sig1)
    f2 = frequency_hz(sig2)
    scale = (f1 / f2) ** 2 - 1.0
    if np.isclose(scale, 0.0):
        return np.zeros_like(np.asarray(p4, dtype=float))
    return np.asarray(p4, dtype=float) / scale


def _dual_rec_pco_columns(mode: str) -> tuple[str, str]:
    """Return receiver PCO columns used by the selected dual-frequency mode."""
    if mode in {"E1E5b", "E5aE5b"}:
        return "pco_los_l1", "pco_los_l5"
    return "pco_los_l1", "pco_los_l2"


def _mode_layout(mode: str, system: str) -> list[dict[str, str]]:
    """Build per-signal observation metadata for a single or dual mode."""
    signals = _mode_signals(mode)
    if len(signals) == 1:
        signal = signals[0]
        code, phase = SIGNAL_COLUMNS[signal]
        return [{"code": code, "phase": phase, "freq": signal, "rec_pco_col": "pco_los"}]

    pco1, pco2 = _dual_rec_pco_columns(mode)
    layout = []
    for signal, pco_col in zip(signals, (pco1, pco2)):
        code, phase = SIGNAL_COLUMNS[signal]
        layout.append({"code": code, "phase": phase, "freq": signal, "rec_pco_col": pco_col})
    return layout


def _elevation_variance_scale(ev_deg: np.ndarray) -> np.ndarray:
    """Return the 1/sin(elevation) variance scale with a low-elevation floor."""
    ev = np.asarray(ev_deg, dtype=float)
    return 1.0 / np.clip(np.sin(np.deg2rad(ev)), 1e-3, None)


def _apply_uncombined_measurement_variances(
    r_diag: np.ndarray,
    code_masks: list[np.ndarray],
    phase_masks: list[np.ndarray],
    ev_deg: np.ndarray,
    rows_per_sat: int,
    code_var: float = 1.0,
    phase_var: float = 1.0,
) -> None:
    """Fill uncombined code/phase variances in an epoch R diagonal."""
    scale = _elevation_variance_scale(ev_deg)
    for band_idx, code_mask in enumerate(code_masks):
        code_row = 2 * band_idx
        phase_row = code_row + 1
        phase_mask = phase_masks[band_idx]
        r_diag[code_row::rows_per_sat] = np.where(code_mask, code_var * scale, 1e12)
        r_diag[phase_row::rows_per_sat] = np.where(phase_mask, phase_var * scale, 1e12)


class PPPUducSFMultiGNSS:
    """Legacy mixed GPS/Galileo single-frequency uncombined PPP.

    Purpose:
        Compatibility path for mixed single-frequency GPS/Galileo processing.
        It predates the generic mixed uncombined implementation and remains
        reachable from routing for selected G/E single-frequency no-constraint
        scenarios.

    Status:
        Legacy/reachable class. It is still selected by ``PPPSession`` for
        specific GPS+Galileo single-frequency no-constraint routing and is not
        safe to remove without replacing that route and its regression tests.

    Model:
        Uses code and phase observables per satellite without forming
        ionosphere-free combinations. The state carries one receiver clock, one
        Galileo ISB, optional ZTD, and per-satellite ionosphere/ambiguity
        blocks.

    State vector:
        ``[x, y, z, dtr_G, isb_E, ztd?, (I, N)_G..., (I, N)_E...]``.

    Supported systems:
        GPS and Galileo only.

    Limitations:
        Legacy G/E-specific class. Prefer ``PPPUdGenericMixedGNSS`` for new
        mixed no-constraint work and ``PPPFilterMultiGNSSIonConstGEC`` for
        constrained G/E/C validation.
    """

    def __init__(self, config:PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, interval=0.5):
        self.cfg=config
        self.gps_obs = gps_obs
        self.gps_mode = gps_mode
        self.gal_obs = gal_obs
        self.gal_mode = gal_mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
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

        e_gps, _ = _unit_vectors_and_ranges(gps_satellites[:, :3], rec_xyz)
        e_gal, _ = _unit_vectors_and_ranges(gal_satellites[:, :3], rec_xyz)
        m_wet_gps = gps_satellites[:, 3]
        m_wet_gal = gal_satellites[:, 3]

        # Parametry wspólne: XYZ, clk_gps, ISB_galileo, ZTD
        COL_X, COL_Y, COL_Z = 0, 1, 2
        COL_CLK = 3
        COL_ISB = 4
        COL_ZTD = 5 if self.tro else None

        base_dim = self.base_dim

        # 2 równania na każdy satelita: P1, L1, P2, L2
        H = np.zeros((2 * (n_gps + n_gal), base_dim + 2 * (n_gps + n_gal)))

        # ---------------- GPS ----------------
        if n_gps:
            gps_rows = slice(0, 2 * n_gps)
            H[gps_rows, [COL_X, COL_Y, COL_Z]] = np.repeat(e_gps, 2, axis=0)
            H[gps_rows, COL_CLK] = 1.0
            if self.tro:
                H[gps_rows, COL_ZTD] = np.repeat(m_wet_gps, 2)

            idx = np.arange(n_gps)
            row0 = 2 * idx
            row1 = row0 + 1
            col_iono = base_dim + 2 * idx
            H[row0, col_iono] = MU1
            H[row1, col_iono] = -MU1
            H[row1, col_iono + 1] = L1

        # ---------------- Galileo ----------------
        if n_gal:
            gal_rows = slice(2 * n_gps, 2 * (n_gps + n_gal))
            H[gal_rows, [COL_X, COL_Y, COL_Z]] = np.repeat(e_gal, 2, axis=0)
            H[gal_rows, COL_CLK] = 1.0
            H[gal_rows, COL_ISB] = 1.0
            if self.tro:
                H[gal_rows, COL_ZTD] = np.repeat(m_wet_gal, 2)

            idx = np.arange(n_gal)
            row0 = 2 * (n_gps + idx)
            row1 = row0 + 1
            col_iono = base_dim + 2 * (n_gps + idx)
            H[row0, col_iono] = MU1_GAL
            H[row1, col_iono] = -MU1_GAL
            H[row1, col_iono + 1] = E1

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

        clk = x_state[3]
        isb = x_state[4]  # inter-sys bias (dla Galileo)
        zwd = x_state[5] if self.tro else 0.0

        base_dim = self.base_dim
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]

        z_hat = np.empty(2 * (n_gps + n_gal))

        if n_gps:
            gps_state = x_state[base_dim:base_dim + 2 * n_gps].reshape(n_gps, 2)
            I_gps = gps_state[:, 0]
            N1_gps = gps_state[:, 1]
            _, rho_gps = _unit_vectors_and_ranges(gps_satellites[:, :3], x_state[:3])
            mw_gps = gps_satellites[:, 3]
            geom = rho_gps
            gps_block = np.column_stack((
                geom + clk + mw_gps * zwd + MU1 * I_gps,
                geom + clk + mw_gps * zwd - MU1 * I_gps + L1 * N1_gps,
            ))
            z_hat[:2 * n_gps] = gps_block.reshape(-1)

        if n_gal:
            gal_state = x_state[base_dim + 2 * n_gps:base_dim + 2 * (n_gps + n_gal)].reshape(n_gal, 2)
            I_gal = gal_state[:, 0]
            N1_gal = gal_state[:, 1]
            _, rho_gal = _unit_vectors_and_ranges(gal_satellites[:, :3], x_state[:3])
            mw_gal = gal_satellites[:, 3]
            geom = rho_gal
            gal_block = np.column_stack((
                geom + clk + isb + mw_gal * zwd + MU1_GAL * I_gal,
                geom + clk + isb + mw_gal * zwd - MU1_GAL * I_gal + E1 * N1_gal,
            ))
            start = 2 * n_gps
            z_hat[start:start + 2 * n_gal] = gal_block.reshape(-1)

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

        prev_map = {prn: i for i, prn in enumerate(prev_all)}
        curr_map = {prn: i for i, prn in enumerate(curr_all)}
        common = [prn for prn in curr_all if prn in prev_map]

        if common:
            prev_idx = np.fromiter((prev_map[prn] for prn in common), dtype=int)
            curr_idx = np.fromiter((curr_map[prn] for prn in common), dtype=int)
            offsets = np.arange(2)
            old_state_idx = (base_dim + 2 * prev_idx[:, None] + offsets[None, :]).ravel()
            new_state_idx = (base_dim + 2 * curr_idx[:, None] + offsets[None, :]).ravel()

            x_new[new_state_idx] = x_old[old_state_idx]
            base_idx = np.arange(base_dim)
            P_new[np.ix_(new_state_idx, base_idx)] = P_old[np.ix_(old_state_idx, base_idx)]
            P_new[np.ix_(base_idx, new_state_idx)] = P_old[np.ix_(base_idx, old_state_idx)]
            Q_new[np.ix_(new_state_idx, base_idx)] = Q_old[np.ix_(old_state_idx, base_idx)]
            Q_new[np.ix_(base_idx, new_state_idx)] = Q_old[np.ix_(base_idx, old_state_idx)]
            P_new[np.ix_(new_state_idx, new_state_idx)] = P_old[np.ix_(old_state_idx, old_state_idx)]
            Q_new[np.ix_(new_state_idx, new_state_idx)] = Q_old[np.ix_(old_state_idx, old_state_idx)]

        # --- nowe satelity: zainicjalizuj I_s z P4 ---
        new_only = set(curr_all) - set(prev_all)
        for prn in new_only:
            j = curr_map[prn]
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
        if self.tro:
            x0[5] = zwd0  # ZTD

        # Inicjalizacja EKF
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[3, 3] = self.cfg.p_dt  # clk
        self.ekf.P[4, 4] = self.cfg.p_isb  # isb galileo
        if self.tro:
            self.ekf.P[5, 5] = self.cfg.p_tro  # zwd

        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_isb
        if self.tro:
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
        if self.tro:
            x0[5] = zwd0  # ZTD

        # Inicjalizacja EKF
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[3, 3] = self.cfg.p_dt  # clk
        self.ekf.P[4, 4] = self.cfg.p_isb  # isb galileo
        if self.tro:
            self.ekf.P[5, 5] = self.cfg.p_tro  # zwd

        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_isb
        if self.tro:
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
        offsets = np.array([1, 2])
        idxs_N = (base_dim + 2 * np.arange(n_sats)[:, None] + offsets[None, :]).ravel()
        N_vec = x[idxs_N]
        P_N = P[np.ix_(idxs_N, idxs_N)]

        return N_vec, P_N

    def code_screening(self, x, satellites, code_obs, thr=1):
        sat_xyz = np.asarray(satellites, dtype=float)
        ref_xyz = np.asarray(x, dtype=float)
        dist = np.linalg.norm(sat_xyz - ref_xyz, axis=1)
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
        result_rows = []
        result_times = []
        e, g = [], []
        self.gps_obs = self.gps_obs.sort_values(by='sv')
        self.gal_obs = self.gal_obs.sort_values(by='sv')
        gps_epochs = {
            t: df for t, df in self.gps_obs.groupby(level=1, sort=False)
        }
        gal_epochs = {
            t: df for t, df in self.gal_obs.groupby(level=1, sort=False)
        }
        gps_osb_c1_col = f'OSB_{gps_c1}'
        gps_osb_l1_col = f'OSB_{gps_l1}'
        gal_osb_c1_col = f'OSB_{gal_c1}'
        gal_osb_l1_col = f'OSB_{gal_l1}'
        gps_has_osb_c1 = gps_osb_c1_col in self.gps_obs.columns
        gps_has_osb_l1 = gps_osb_l1_col in self.gps_obs.columns
        gal_has_osb_c1 = gal_osb_c1_col in self.gal_obs.columns
        gal_has_osb_l1 = gal_osb_l1_col in self.gal_obs.columns
        r_cache = {}
        xyz = None
        conv_time = None
        T0 = all_times[0]
        gps_arc_age = {sv: 0 for sv in old_gps_sats}
        gal_arc_age = {sv: 0 for sv in old_gal_sats}
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
        if self.system == "C" and ar_cfg.enabled:
            import warnings
            warnings.warn(
                "PPP-AR is disabled for BeiDou uncombined PPP until BDS phase-bias handling is validated.",
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
        for num, t in enumerate(all_times):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_gps_sats, old_gal_sats = self.reset_filter(epoch=t)
                    gps_arc_age = {sv: 0 for sv in old_gps_sats}
                    gal_arc_age = {sv: 0 for sv in old_gal_sats}
                    reset_epoch = True
                    T0 = t
            gps_epoch = gps_epochs.get(t)
            gal_epoch = gal_epochs.get(t)
            if gps_epoch is None or gal_epoch is None:
                continue

            def safe_get(df, col, length=None):
                """Zwraca kolumnę jako numpy, a jeśli brak – wektor zer o podanej długości."""
                if col in df.columns:
                    return df[col].to_numpy(copy=False)
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

            tides = gps_epoch['tides_los'].to_numpy(copy=False)
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gps_p1_c1 = osb_m(gps_epoch, gps_c1, gps_len)
            gps_pl1_l1 = osb_m(gps_epoch, gps_l1, gps_len)
            C1 = gps_epoch[gps_c1].to_numpy(copy=False) - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p1_c1 - tides
            L1 = gps_epoch[gps_l1].to_numpy(copy=False) - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl1_l1 - tides - (
                self.CLIGHT / self.FREQ_DICT[self.gps_mode]) * phw

            gal_tides = gal_epoch['tides_los'].to_numpy(copy=False)
            gal_satellites = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gal_p1_c1 = osb_m(gal_epoch, gal_c1, gal_len)
            gal_pl1_l1 = osb_m(gal_epoch, gal_l1, gal_len)

            EC1 = gal_epoch[gal_c1].to_numpy(copy=False) - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_p1_c1 - gal_tides
            EL1 = gal_epoch[gal_l1].to_numpy(copy=False) - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_pl1_l1 - gal_tides - (
                self.CLIGHT / self.FREQ_DICT[self.gal_mode]) * gal_phw

            ZGPS = np.empty(2 * len(C1))
            ZGPS[0::2] = C1
            ZGPS[1::2] = L1
            ZGAL = np.empty(2 * len(EC1))
            ZGAL[0::2] = EC1
            ZGAL[1::2] = EL1

            Z = np.concatenate((ZGPS, ZGAL))
            if not np.all(np.isfinite(Z)):
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "non-finite-observation",
                        epoch=num,
                        time=t,
                    )
                continue
            if not np.all(np.isfinite(Z)):
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "non-finite-observation",
                        epoch=num,
                        time=t,
                    )
                continue
            gps_c1_mask = self.code_screening(x=self.ekf.x[:3],code_obs=C1,thr=30,satellites=gps_satellites[:,:3])
            gal_c1_mask = self.code_screening(x=self.ekf.x[:3], code_obs=EC1, thr=30, satellites=gal_satellites[:,:3])

            # WEIGHTS GPS
            ev_gps = np.deg2rad(gps_epoch['ev'].to_numpy(copy=False))
            inv_sin_gps = 1.0 / np.sin(ev_gps)
            R = np.empty(2 * len(C1))
            R[0::2] = np.where(gps_c1_mask, inv_sin_gps, 1e12)
            R[1::2] = 0.001 * inv_sin_gps

            # WEIGHTS GAL
            ev_gal = np.deg2rad(gal_epoch['ev'].to_numpy(copy=False))
            inv_sin_gal = 1.0 / np.sin(ev_gal)
            RG = np.empty(2 * len(EC1))
            RG[0::2] = np.where(gal_c1_mask, inv_sin_gal, 1e12)
            RG[1::2] = 0.001 * inv_sin_gal

            r_diag = np.concatenate((R, RG))
            r_diag = np.nan_to_num(r_diag, nan=1e12, posinf=1e12, neginf=1e12)
            m = r_diag.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = r_diag
            self.ekf.R = R
            updated = False
            for k in range(5):
                try:
                    self.ekf.predict_update(
                        z=Z,
                        HJacobian=self.Hjacobian,
                        args=(gps_satellites, gal_satellites),
                        Hx=self.Hx,
                        hx_args=(gps_satellites, gal_satellites),
                    )
                    updated = True
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-6 * (10.0 ** k)
                    self.ekf.R.flat[::m + 1] += jitter
            if not updated:
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "singular-update-reset",
                        epoch=num,
                        time=t,
                    )
                old_gps_sats, old_gal_sats = self.reset_filter(epoch=t)
                gps_arc_age = {sv: 0 for sv in old_gps_sats}
                gal_arc_age = {sv: 0 for sv in old_gal_sats}
                continue
            gps_arc_age = advance_arc_age(gps_arc_age, curr_gps_sats)
            gal_arc_age = advance_arc_age(gal_arc_age, curr_gal_sats)
            gps_ar_age = arc_age_array(gps_arc_age, curr_gps_sats)
            gal_ar_age = arc_age_array(gal_arc_age, curr_gal_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                groups = []
                if curr_gps_sats:
                    gps_idx = np.arange(len(curr_gps_sats))
                    groups.append(
                        {
                            "n1_idx": self.base_dim + 2 * gps_idx + 1,
                            "n2_idx": None,
                            "ev": gps_epoch['ev'].to_numpy(copy=False),
                            "age": gps_ar_age,
                        }
                    )
                if curr_gal_sats:
                    gal_idx = len(curr_gps_sats) + np.arange(len(curr_gal_sats))
                    groups.append(
                        {
                            "n1_idx": self.base_dim + 2 * gal_idx + 1,
                            "n2_idx": None,
                            "ev": gal_epoch['ev'].to_numpy(copy=False),
                            "age": gal_ar_age,
                        }
                    )
                self.ekf.x, self.ekf.P, ar_diag = apply_indexed_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    ambiguity_groups=groups,
                    settings=ar_cfg,
                )
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
            else:
                enu = np.zeros(3)
            result_rows.append({
                'de': enu[0],
                'dn': enu[1],
                'du': enu[2],
                'dtr': dtr,
                'ztd': ztd,
                'x': xyz[0],
                'y': xyz[1],
                'z': xyz[2],
                'isb': isb,
                'ar_fixed': 0 if ar_diag is None else ar_diag.fixed_ambiguities,
                'ar_ratio': np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min,
                'ar_ok': False if ar_diag is None else ar_diag.accepted,
                'ar_gps_min_age': int(np.min(gps_ar_age)) if gps_ar_age.size else np.nan,
                'ar_gal_min_age': int(np.min(gal_ar_age)) if gal_ar_age.size else np.nan,
            })
            result_rows[-1].update(pppar_diagnostic_columns(ar_diag))
            result_times.append(t)
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result_rows[-1],
                )
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
        gps_result = pd.concat(g)
        gal_result = pd.concat(e)
        result = pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time'))
        result['ct_min']=ct
        return result, gps_result, gal_result, ct


class PPPUducSFSingleGNSS:
    """Single-frequency uncombined PPP with ionospheric constraints.

    Purpose:
        Active single-system constrained branch for one signal when
        ``PPPConfig.use_iono_constr`` is true.

    Status:
        Active constrained single-frequency path. Requires caution for
        ionosphere constraint weighting and datum assumptions.

    Model:
        Uses code and phase observables plus one ionosphere pseudo-observation
        per satellite. The ionosphere constraint comes from the preprocessed
        ``ion`` column and can be weighted with ``ion_rms`` through
        ``_iono_constraint_sigma``.

    State vector:
        ``[x, y, z, dtr, ztd?, (I, N)_s1, ..., (I, N)_sn]`` where each
        satellite block contains slant ionosphere delay and a float ambiguity.

    Supported systems/modes:
        Any single-frequency GPS, Galileo or BeiDou mode supported by
        ``gnx_py.gnss`` and present in the observation columns.

    Inputs:
        Observations must include orbit/clock/preprocessing columns and an
        ``ion`` column. ``ion_rms`` is required only when the configuration asks
        to use RMS-weighted ionospheric constraints.

    PPP-AR support:
        Indexed uncombined AR may be attempted for the single ambiguity group
        when enabled, but validation should be treated cautiously.
        BeiDou AR is explicitly disabled elsewhere in the uncombined code until
        BDS phase-bias handling is validated.

    Limitations:
        The model depends strongly on the quality and datum of the external
        ionospheric constraint. Do not compare it directly with unconstrained
        single-frequency PPP as if they were the same observable model.
    """

    def __init__(
        self,
        config: PPPConfig,
        obs,
        mode,
        ekf,
        pos0,
        tro=True,
        interval=0.5,
        use_iono_rms=True,
        sigma_iono_0=1.1,
        sigma_iono_end=2.3,
        t_end=30,
    ):
        self.cfg = config
        self.obs = obs
        self.mode = mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.CLIGHT = 299792458
        self.base_dim = 5 if tro else 4  # X, Y, Z, clk, ZTD(optional)
        self.pos0 = pos0
        self.interval = interval
        self.use_iono_rms = use_iono_rms
        self.sigma_iono_0 = sigma_iono_0
        self.sigma_iono_end = sigma_iono_end
        self.t_end = t_end

    def _obs_mode_columns(self):
        signals = _mode_signals(self.mode)
        if len(signals) != 1:
            raise ValueError(f"Mode {self.mode} is not single-frequency.")
        code, phase = SIGNAL_COLUMNS[signals[0]]
        return code, phase, 'pco_los'

    def Hjacobian(self, x, satellites):
        n = satellites.shape[0]
        C = self.CLIGHT
        f1 = self.FREQ_DICT[self.mode]
        lam = C / f1
        rec_xyz = x[:3].copy()
        e_vec, _ = _unit_vectors_and_ranges(satellites[:, :3], rec_xyz)
        dim = self.base_dim + 2 * n
        H = np.zeros((3 * n, dim))
        col_x, col_y, col_z = 0, 1, 2
        col_clk = 3
        col_ztd = 4
        for s in range(n):
            row = 3 * s
            ex, ey, ez = e_vec[s]
            H[row + 0, [col_x, col_y, col_z]] = [ex, ey, ez]
            H[row + 1, [col_x, col_y, col_z]] = [ex, ey, ez]
            H[row + 0, col_clk] = 1.0
            H[row + 1, col_clk] = 1.0
            if self.tro:
                mw = satellites[s, 3]
                H[row + 0, col_ztd] = mw
                H[row + 1, col_ztd] = mw
            col_iono = self.base_dim + 2 * s
            col_n1 = col_iono + 1
            H[row + 0, col_iono] = 1.0
            H[row + 1, col_iono] = -1.0
            H[row + 1, col_n1] = lam
            H[row + 2, col_iono] = 1.0
        return H

    def Hx(self, x, satellites):
        n = satellites.shape[0]
        C = self.CLIGHT
        f1 = self.FREQ_DICT[self.mode]
        lam = C / f1
        clk = x[3]
        zwd = x[4] if self.tro else 0.0
        z_hat = np.empty(3 * n)
        if n == 0:
            return z_hat
        state = x[self.base_dim:self.base_dim + 2 * n].reshape(n, 2)
        I = state[:, 0]
        N1 = state[:, 1]
        _, rho = _unit_vectors_and_ranges(satellites[:, :3], x[:3])
        mw = satellites[:, 3] if self.tro else 0.0
        code = rho + clk + mw * zwd + I
        phase = rho + clk + mw * zwd - I + lam * N1
        z_hat[0::3] = code
        z_hat[1::3] = phase
        z_hat[2::3] = I
        return z_hat

    def rebuild_state(self, x_old, P_old, Q_old, prev_sats, curr_sats):
        base = self.base_dim
        n_prev = len(prev_sats)
        n_curr = len(curr_sats)
        new_dim = base + 2 * n_curr
        x_new = np.zeros(new_dim)
        x_new[:base] = x_old[:base]
        P_new = np.zeros((new_dim, new_dim))
        P_new[:base, :base] = P_old[:base, :base]
        Q_new = np.zeros((new_dim, new_dim))
        Q_new[:base, :base] = Q_old[:base, :base]
        prev_map = {sv: i for i, sv in enumerate(prev_sats)}
        curr_map = {sv: i for i, sv in enumerate(curr_sats)}
        common = [sv for sv in curr_sats if sv in prev_map]
        if common:
            prev_idx = np.fromiter((prev_map[sv] for sv in common), dtype=int)
            curr_idx = np.fromiter((curr_map[sv] for sv in common), dtype=int)
            offs = np.arange(2)
            old_state = (base + 2 * prev_idx[:, None] + offs[None, :]).ravel()
            new_state = (base + 2 * curr_idx[:, None] + offs[None, :]).ravel()
            x_new[new_state] = x_old[old_state]
            b = np.arange(base)
            P_new[np.ix_(new_state, b)] = P_old[np.ix_(old_state, b)]
            P_new[np.ix_(b, new_state)] = P_old[np.ix_(b, old_state)]
            Q_new[np.ix_(new_state, b)] = Q_old[np.ix_(old_state, b)]
            Q_new[np.ix_(b, new_state)] = Q_old[np.ix_(b, old_state)]
            P_new[np.ix_(new_state, new_state)] = P_old[np.ix_(old_state, old_state)]
            Q_new[np.ix_(new_state, new_state)] = Q_old[np.ix_(old_state, old_state)]
        new_only = set(curr_sats) - set(prev_sats)
        for sv in new_only:
            j = curr_map[sv]
            idx_i = base + 2 * j
            idx_n = idx_i + 1
            P_new[idx_i, idx_i] = self.cfg.p_iono
            Q_new[idx_i, idx_i] = self.cfg.q_iono * (self.interval * 60) / 3600
            P_new[idx_n, idx_n] = self.cfg.p_amb
        return x_new, P_new, Q_new

    def _init_filter_common(self, epoch, clk0=0.0, zwd0=0.0):
        sats = self.obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        n = len(sats)
        dim_x = self.base_dim + 2 * n
        dim_z = 3 * n
        x0 = np.zeros(dim_x)
        x0[:3] = self.pos0
        x0[3] = clk0
        if self.tro:
            x0[4] = zwd0
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        if self.tro:
            self.ekf.P[4, 4] = self.cfg.p_tro
            self.ekf.Q[4, 4] = self.cfg.q_tro
        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0
        if 'ion' in self.obs.columns:
            ion_init = self.obs.loc[(slice(None), epoch), 'ion'].to_numpy()
        elif 'P4' in self.obs.columns:
            ion_init = self.obs.loc[(slice(None), epoch), 'P4'].to_numpy()
        else:
            ion_init = np.zeros(n, dtype=float)
        for k in range(n):
            idx_i = self.base_dim + 2 * k
            idx_n = idx_i + 1
            self.ekf.x[idx_i] = ion_init[k]
            self.ekf.P[idx_i, idx_i] = self.cfg.p_iono
            self.ekf.Q[idx_i, idx_i] = self.cfg.q_iono * (self.interval * 60) / 3600
            self.ekf.P[idx_n, idx_n] = self.cfg.p_amb
        return sats

    def init_filter(self, clk0=0.0, zwd0=0.0):
        epochs = sorted(self.obs.index.get_level_values('time').unique())
        first_ep = epochs[0]
        sats = self._init_filter_common(first_ep, clk0=clk0, zwd0=zwd0)
        return sats, epochs

    def reset_filter(self, epoch, clk0=0.0, zwd0=0.0):
        return self._init_filter_common(epoch, clk0=clk0, zwd0=zwd0)

    def _prepare_obs(self):
        c_prefix, l_prefix, pco_col = self._obs_mode_columns()
        c_col = [c for c in self.obs.columns if c.startswith(c_prefix)][0]
        l_col = [c for c in self.obs.columns if c.startswith(l_prefix)][0]
        self.obs = self.obs.copy()
        if pco_col not in self.obs.columns:
            self.obs.loc[:, pco_col] = 0.0
        if 'me_wet' not in self.obs.columns:
            self.obs.loc[:, 'me_wet'] = 0.0
        return c_col, l_col, pco_col

    def code_screening(self, x, satellites, code_obs, thr=1):
        sat_xyz = np.asarray(satellites, dtype=float)
        ref_xyz = np.asarray(x, dtype=float)
        dist = np.linalg.norm(sat_xyz - ref_xyz, axis=1)
        prefit = code_obs - dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad > n_sat / 2:
            mask = np.ones(n_sat, dtype=bool)
        return mask

    def run_filter(self, clk0=0.0, ref=None, flh=None, zwd0=0.0, trace_filter=False, reset_every=0):
        old_sats, all_epochs = self.init_filter(clk0=clk0, zwd0=zwd0)
        c_col, l_col, pco_col = self._prepare_obs()
        result_rows = []
        result_times = []
        self.obs = self.obs.sort_values(by='sv')
        obs_epochs = {t: df for t, df in self.obs.groupby(level=1, sort=False)}
        osb_c_col = f'OSB_{c_col}'
        osb_l_col = f'OSB_{l_col}'
        has_osb_c = osb_c_col in self.obs.columns
        has_osb_l = osb_l_col in self.obs.columns
        r_cache = {}
        xyz = None
        conv_time = None
        T0 = all_epochs[0]
        arc_age = {sv: 0 for sv in old_sats}
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
        for num, t in enumerate(all_epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                    arc_age = {sv: 0 for sv in old_sats}
                    reset_epoch = True
                    T0 = t
            epoch = obs_epochs.get(t)
            if epoch is None:
                continue
            curr_sats = epoch.index.get_level_values('sv').tolist()
            if curr_sats != old_sats:
                self.ekf.x, self.ekf.P, self.ekf.Q = self.rebuild_state(
                    self.ekf.x.copy(),
                    self.ekf.P.copy(),
                    self.ekf.Q.copy(),
                    old_sats,
                    curr_sats,
                )
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = 3 * len(curr_sats)
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)
            old_sats = curr_sats
            tides = epoch['tides_los'].to_numpy(copy=False) if 'tides_los' in epoch.columns else 0.0
            satellites = epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            clk = epoch['clk'].to_numpy(copy=False) if 'clk' in epoch.columns else 0.0
            tro = epoch['tro'].to_numpy(copy=False) if 'tro' in epoch.columns else 0.0
            ah_los = epoch['ah_los'].to_numpy(copy=False) if 'ah_los' in epoch.columns else 0.0
            dprel = epoch['dprel'].to_numpy(copy=False) if 'dprel' in epoch.columns else 0.0
            rec_pco = epoch[pco_col].to_numpy(copy=False)
            sat_pco = (
                epoch[f'sat_pco_los_{self.mode}'].to_numpy(copy=False)
                if f'sat_pco_los_{self.mode}' in epoch.columns
                else 0.0
            )
            phw = epoch['phw'].to_numpy(copy=False) if 'phw' in epoch.columns else 0.0
            osb_c = osb_m(epoch, c_col, len(epoch))
            osb_l = osb_m(epoch, l_col, len(epoch))
            C1 = (
                epoch[c_col].to_numpy(copy=False) - rec_pco + sat_pco + clk * self.CLIGHT
                - tro - ah_los - dprel - osb_c - tides
            )
            L1 = (
                epoch[l_col].to_numpy(copy=False) - rec_pco + sat_pco + clk * self.CLIGHT
                - tro - ah_los - dprel - osb_l - tides
                - (self.CLIGHT / self.FREQ_DICT[self.mode]) * phw
            )
            if 'ion' in epoch.columns:
                I_obs = epoch['ion'].to_numpy(copy=False)
            elif 'P4' in epoch.columns:
                I_obs = epoch['P4'].to_numpy(copy=False)
            else:
                I_obs = np.zeros(len(epoch), dtype=float)
            Z = np.empty(3 * len(C1))
            Z[0::3] = C1
            Z[1::3] = L1
            Z[2::3] = I_obs
            if not np.all(np.isfinite(Z)):
                continue
            code_mask = self.code_screening(
                x=self.ekf.x[:3],
                code_obs=C1,
                thr=30,
                satellites=satellites[:, :3],
            )
            ev = np.deg2rad(epoch['ev'].to_numpy(copy=False)) if 'ev' in epoch.columns else np.deg2rad(
                np.full(len(epoch), 30.0)
            )
            inv_sin = 1.0 / np.sin(ev)
            r_diag = np.empty(3 * len(C1))
            r_diag[0::3] = np.where(code_mask, inv_sin, 1e12)
            r_diag[1::3] = 0.001 * inv_sin
            if self.use_iono_rms and ('ion_rms' in epoch.columns):
                ion_sigma = epoch['ion_rms'].to_numpy(copy=False)
                ion_sigma = np.where(np.isfinite(ion_sigma), ion_sigma, self.sigma_iono_end)
                ion_sigma = np.clip(ion_sigma, 1e-6, 1e6)
            else:
                frac = min(num, self.t_end) / max(self.t_end, 1)
                ion_sigma = np.full(
                    len(C1),
                    self.sigma_iono_0 + (self.sigma_iono_end - self.sigma_iono_0) * frac,
                    dtype=float,
                )
            r_diag[2::3] = ion_sigma
            r_diag = np.nan_to_num(r_diag, nan=1e12, posinf=1e12, neginf=1e12)
            m = r_diag.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = r_diag
            self.ekf.R = R
            self.ekf.predict_update(
                z=Z,
                HJacobian=self.Hjacobian,
                args=(satellites,),
                Hx=self.Hx,
                hx_args=(satellites,),
            )
            arc_age = advance_arc_age(arc_age, curr_sats)
            ar_age = arc_age_array(arc_age, curr_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_indexed_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    ambiguity_groups=[
                        {
                            "n1_idx": self.base_dim + 2 * np.arange(len(curr_sats)) + 1,
                            "n2_idx": None,
                            "ev": epoch['ev'].to_numpy(copy=False)
                            if 'ev' in epoch.columns
                            else np.full(len(curr_sats), 45.0, dtype=float),
                            "age": ar_age,
                        }
                    ],
                    settings=ar_cfg,
                )
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0
                    elif position_diff > 0.005:
                        conv_time = None
            xyz = self.ekf.x[:3].copy()
            dtr = self.ekf.x[3]
            ztd = self.ekf.x[4] if self.tro else 0.0
            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
            else:
                enu = np.zeros(3)
            result_rows.append({
                'de': enu[0],
                'dn': enu[1],
                'du': enu[2],
                'dtr': dtr,
                'ztd': ztd,
                'x': xyz[0],
                'y': xyz[1],
                'z': xyz[2],
                'ar_fixed': 0 if ar_diag is None else ar_diag.fixed_ambiguities,
                'ar_ratio': np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min,
                'ar_ok': False if ar_diag is None else ar_diag.accepted,
                'ar_min_age': int(np.min(ar_age)) if ar_age.size else np.nan,
            })
            result_rows[-1].update(pppar_diagnostic_columns(ar_diag))
            result_times.append(t)
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result_rows[-1],
                )
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
        result = pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time'))
        result['ct_min'] = ct
        return result, self.obs, None, ct


class PPPUdGenericMixedGNSS:
    """Generic mixed-system uncombined PPP filter.

    Purpose:
        Active generic no-constraint mixed PPP branch for GPS, Galileo and
        BeiDou. It is selected when multiple systems are active and
        ``PPPConfig.use_iono_constr`` is false.

    Status:
        Active generic mixed no-constraint path. Requires caution when changing
        bias handling, conservative BDS weighting, or the merged
        ionosphere/receiver-bias interpretation.

    Model:
        Keeps uncombined code/phase observables in the EKF. A reference
        constellation supplies the receiver clock; other constellations use ISB
        states. Each satellite gets a per-signal state block with ionosphere and
        ambiguity states according to the selected mode.

    State vector:
        ``[x, y, z, dtr_ref, ISB_nonref..., ztd?, rx_signal_bias?,
        per-system/satellite blocks...]``. For the common dual-frequency
        no-constraint case, each satellite block is effectively
        ``[I, N1, N2]``.

    Supported systems/modes:
        GPS, Galileo and BeiDou in the generic system order ``G, E, C``.
        Single- or dual-frequency layouts are resolved from ``mode_by_system``.

    Bias / ionosphere assumptions:
        Without ionospheric constraints, residual receiver signal bias and
        ionosphere datum effects can be partly merged in the estimated
        ionosphere/bias states. The optional receiver signal-bias states absorb
        remaining inter-signal code datum mismatches after available bias
        products are applied.

    PPP-AR support:
        Uses indexed uncombined PPP-AR when enabled and when candidate groups
        pass arc-age, elevation, wide-lane and ratio gates.
        BDS-containing ambiguity groups require dedicated validation before AR
        behavior should be treated as production-stable.

    Limitations:
        This is a flexible active path, but mixed no-constraint G/E/C solutions
        are sensitive to bias product consistency. Treat changes to state layout
        or bias handling as numerical model changes.
    """

    SYSTEM_ORDER = ("G", "E", "C")
    BDS_CODE_VAR_SCALE_MIXED_NOCONSTR = 12.0
    BDS_PHASE_VAR_SCALE_MIXED_NOCONSTR = 144.0

    def __init__(
        self,
        config: PPPConfig,
        obs_by_system: dict[str, pd.DataFrame],
        mode_by_system: dict[str, str],
        ekf,
        pos0,
        tro=True,
        interval=0.5,
        reference_system: str | None = None,
        estimate_signal_bias: bool | None = None,
        signal_bias_start_band: int = 1,
    ):
        self.cfg = config
        self.use_iono_constr = bool(getattr(self.cfg, "use_iono_constr", False))
        self.use_iono_rms = bool(getattr(self.cfg, "use_iono_rms", True))
        self.sigma_iono_0 = getattr(self.cfg, "sigma_iono_0", 1.1)
        self.sigma_iono_end = getattr(self.cfg, "sigma_iono_end", 3.0)
        self.t_end = getattr(self.cfg, "t_end", 30)
        self.obs_by_system = {
            system: obs.copy()
            for system, obs in obs_by_system.items()
            if isinstance(obs, pd.DataFrame) and not obs.empty
        }
        self.mode_by_system = {
            system: mode_by_system[system]
            for system in self.SYSTEM_ORDER
            if system in self.obs_by_system
        }
        self.systems = tuple(system for system in self.SYSTEM_ORDER if system in self.obs_by_system)
        if len(self.systems) < 2:
            raise ValueError("Generic mixed PPP requires at least two active systems.")

        self.reference_system = reference_system or self.systems[0]
        if self.reference_system not in self.systems:
            raise ValueError(f"Reference system {self.reference_system!r} is not active.")
        self.isb_systems = tuple(system for system in self.systems if system != self.reference_system)
        self.isb_cols = {system: 4 + i for i, system in enumerate(self.isb_systems)}
        self.col_ztd = 4 + len(self.isb_systems) if tro else None

        # After applying available satellite/station bias products we still keep
        # one residual code-datum state per system/frequency pair in mixed float
        # PPP to absorb remaining inter-system observable datum mismatches.
        self.estimate_signal_bias = (not self.use_iono_constr) if estimate_signal_bias is None else bool(estimate_signal_bias)
        self.signal_bias_q = 0.0 if self.estimate_signal_bias else float(getattr(self.cfg, "q_dcb", 0.0))
        self.signal_bias_start_band = int(signal_bias_start_band)
        self.anchor_first_signal_bias = False
        self.bias_state_cols: dict[str, dict[int, int]] = {}
        next_col = 4 + len(self.isb_systems) + (1 if tro else 0)
        for system in self.systems:
            band_cols: dict[int, int] = {}
            if self.estimate_signal_bias:
                for band_idx in range(self.signal_bias_start_band, len(_mode_signals(self.mode_by_system[system]))):
                    band_cols[band_idx] = next_col
                    next_col += 1
            self.bias_state_cols[system] = band_cols
        self.base_dim = next_col
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.CLIGHT = 299792458
        self.pos0 = pos0
        self.interval = interval
        self.code_prefit_threshold = float(getattr(self.cfg, "uncombined_code_prefit_threshold", 30.0))
        self.phase_jump_reset_threshold = float(getattr(self.cfg, "uncombined_phase_screen_threshold", 10.0))
        phase_gate_systems = getattr(self.cfg, "uncombined_mixed_phase_code_gate_systems", None)
        if phase_gate_systems is None:
            phase_gate_systems = ()
        elif isinstance(phase_gate_systems, str):
            phase_gate_systems = tuple(s.strip() for s in phase_gate_systems.split(",") if s.strip())
        self.phase_code_gate_systems = tuple(phase_gate_systems)
        self.conservative_bds_weighting = (
            (not self.use_iono_constr)
            and ("C" in self.systems)
            and (len(self.systems) > 1)
        )
        self.enable_phase_jump_resets = False
        if self.use_iono_constr:
            for system, obs in self.obs_by_system.items():
                if "ion" not in obs.columns:
                    raise ValueError("use_iono_constr requires an 'ion' column in all mixed observations.")
                self.obs_by_system[system] = obs.dropna(subset=["ion"])
                if self.use_iono_rms and "ion_rms" in obs.columns:
                    self.obs_by_system[system] = self.obs_by_system[system].dropna(subset=["ion_rms"])

    def _state_dim(self, system: str) -> int:
        return 1 + len(_mode_signals(self.mode_by_system[system]))

    def _obs_dim(self, system: str) -> int:
        return 2 * len(_mode_signals(self.mode_by_system[system])) + int(self.use_iono_constr)

    def _state_specs(self, sats_by_system: dict[str, list[str]]) -> tuple[dict[str, tuple[int, int]], int]:
        specs: dict[str, tuple[int, int]] = {}
        offset = self.base_dim
        for system in self.systems:
            state_dim = self._state_dim(system)
            for sv in sats_by_system.get(system, []):
                specs[f"{system}:{sv}"] = (offset, state_dim)
                offset += state_dim
        return specs, offset

    def _system_state_start(self, sats_by_system: dict[str, list[str]], system: str) -> int:
        offset = self.base_dim
        for current in self.systems:
            if current == system:
                return offset
            offset += self._state_dim(current) * len(sats_by_system.get(current, []))
        raise KeyError(system)

    def _system_obs_start(self, sats_by_system: dict[str, list[str]], system: str) -> int:
        offset = 0
        for current in self.systems:
            if current == system:
                return offset
            offset += self._obs_dim(current) * len(sats_by_system.get(current, []))
        raise KeyError(system)

    def _signal_bias_col(self, system: str, band_idx: int) -> int | None:
        return self.bias_state_cols.get(system, {}).get(band_idx)

    def _satellite_blocks(self, satellite_args) -> dict[str, np.ndarray]:
        return {system: np.asarray(block) for system, block in zip(self.systems, satellite_args)}

    def _sat_counts_from_blocks(self, blocks: dict[str, np.ndarray]) -> dict[str, list[str]]:
        return {system: [f"{system}{i:03d}" for i in range(blocks[system].shape[0])] for system in self.systems}

    def Hjacobian(self, x, *satellite_args):
        blocks = self._satellite_blocks(satellite_args)
        pseudo_sats = self._sat_counts_from_blocks(blocks)
        _, dim_x = self._state_specs(pseudo_sats)
        dim_z = sum(self._obs_dim(system) * blocks[system].shape[0] for system in self.systems)
        H = np.zeros((dim_z, dim_x))
        for system in self.systems:
            n = blocks[system].shape[0]
            if not n:
                continue
            self._fill_h_system(
                H=H,
                x=x,
                satellites=blocks[system],
                system=system,
                row_start=self._system_obs_start(pseudo_sats, system),
                state_start=self._system_state_start(pseudo_sats, system),
            )
        return H

    def _fill_h_system(self, H, x, satellites, system, row_start, state_start):
        mode = self.mode_by_system[system]
        signals = _mode_signals(mode)
        state_dim = self._state_dim(system)
        obs_dim = self._obs_dim(system)
        f_ref = self.FREQ_DICT[signals[0]]
        rec_xyz = x[:3].copy()
        e_vec, _ = _unit_vectors_and_ranges(satellites[:, :3], rec_xyz)
        m_wet = satellites[:, 3]

        for sat_idx in range(satellites.shape[0]):
            row = row_start + obs_dim * sat_idx
            state = state_start + state_dim * sat_idx
            for band_idx, signal in enumerate(signals):
                code_row = row + 2 * band_idx
                phase_row = code_row + 1
                freq = self.FREQ_DICT[signal]
                mu = (f_ref / freq) ** 2
                lam = self.CLIGHT / freq
                H[[code_row, phase_row], 0:3] = e_vec[sat_idx]
                H[[code_row, phase_row], 3] = 1.0
                if system != self.reference_system:
                    H[[code_row, phase_row], self.isb_cols[system]] = 1.0
                if self.tro:
                    H[[code_row, phase_row], self.col_ztd] = m_wet[sat_idx]
                bias_col = self._signal_bias_col(system, band_idx)
                if bias_col is not None:
                    H[code_row, bias_col] = 1.0
                    if getattr(self, "legacy_phase_bias_jacobian", False):
                        H[phase_row, bias_col] = 1.0
                H[code_row, state] = mu
                H[phase_row, state] = -mu
                H[phase_row, state + 1 + band_idx] = lam
            if self.use_iono_constr:
                H[row + 2 * len(signals), state] = 1.0

    def Hx(self, x, *satellite_args):
        blocks = self._satellite_blocks(satellite_args)
        pseudo_sats = self._sat_counts_from_blocks(blocks)
        dim_z = sum(self._obs_dim(system) * blocks[system].shape[0] for system in self.systems)
        z_hat = np.empty(dim_z)
        for system in self.systems:
            n = blocks[system].shape[0]
            if not n:
                continue
            self._fill_hx_system(
                z_hat=z_hat,
                x=x,
                satellites=blocks[system],
                system=system,
                row_start=self._system_obs_start(pseudo_sats, system),
                state_start=self._system_state_start(pseudo_sats, system),
            )
        return z_hat

    def _fill_hx_system(self, z_hat, x, satellites, system, row_start, state_start):
        mode = self.mode_by_system[system]
        signals = _mode_signals(mode)
        state_dim = self._state_dim(system)
        obs_dim = self._obs_dim(system)
        f_ref = self.FREQ_DICT[signals[0]]
        clk = x[3]
        isb = 0.0 if system == self.reference_system else x[self.isb_cols[system]]
        zwd = x[self.col_ztd] if self.tro else 0.0
        _, rho = _unit_vectors_and_ranges(satellites[:, :3], x[:3])
        mw = satellites[:, 3] if self.tro else 0.0
        n = satellites.shape[0]
        state = x[state_start:state_start + state_dim * n].reshape(n, state_dim)
        ion = state[:, 0]
        block = np.empty((n, obs_dim))
        base = rho + clk + isb + mw * zwd
        for band_idx, signal in enumerate(signals):
            freq = self.FREQ_DICT[signal]
            mu = (f_ref / freq) ** 2
            lam = self.CLIGHT / freq
            amb = state[:, 1 + band_idx]
            bias_col = self._signal_bias_col(system, band_idx)
            band_bias = 0.0 if bias_col is None else x[bias_col]
            block[:, 2 * band_idx] = base + mu * ion + band_bias
            block[:, 2 * band_idx + 1] = base - mu * ion + lam * amb
        if self.use_iono_constr:
            block[:, -1] = ion
        z_hat[row_start:row_start + obs_dim * n] = block.reshape(-1)

    def _prepare_system_obs(self, obs: pd.DataFrame, mode: str, system: str):
        obs = obs.copy()
        layout = []
        for band in _mode_layout(mode, system):
            resolved = dict(band)
            resolved["code_col"] = _first_prefixed_column(obs, band["code"])
            resolved["phase_col"] = _first_prefixed_column(obs, band["phase"])
            if band["rec_pco_col"] not in obs.columns:
                obs.loc[:, band["rec_pco_col"]] = 0.0
            layout.append(resolved)
        if "me_wet" not in obs.columns:
            obs.loc[:, "me_wet"] = 0.0
        return obs, layout

    def _prepare_obs(self):
        layouts = {}
        for system in self.systems:
            obs, layout = self._prepare_system_obs(
                self.obs_by_system[system],
                self.mode_by_system[system],
                system,
            )
            self.obs_by_system[system] = obs
            layouts[system] = layout
        return layouts

    def _col_np(self, epoch, name, default=0.0):
        if name in epoch.columns:
            return epoch[name].to_numpy(copy=False)
        if np.isscalar(default):
            return np.full(len(epoch), default, dtype=float)
        return np.asarray(default)

    def _bias_col_m(self, epoch: pd.DataFrame, column: str, n: int):
        return bias_column_m(epoch, column, n)

    def _dcb_between_m(self, epoch: pd.DataFrame, prefix: str, obs_a: str, obs_b: str, n: int):
        if obs_a == obs_b:
            return np.zeros(n, dtype=float)
        direct = self._bias_col_m(epoch, f"{prefix}{obs_a}_{obs_b}", n)
        if direct is not None:
            return direct
        reverse = self._bias_col_m(epoch, f"{prefix}{obs_b}_{obs_a}", n)
        if reverse is not None:
            return -reverse
        return None

    def _dual_code_dsb_corrections_m(self, epoch: pd.DataFrame, layout: list[dict[str, str]], n: int):
        corr = [np.zeros(n, dtype=float) for _ in layout]
        total_dcb = np.zeros(n, dtype=float)
        if len(layout) != 2:
            return corr, total_dcb

        code_1 = layout[0]["code_col"]
        code_2 = layout[1]["code_col"]
        sig_1 = layout[0]["freq"]
        sig_2 = layout[1]["freq"]
        f1 = self.FREQ_DICT[sig_1]
        f2 = self.FREQ_DICT[sig_2]
        a = f1 ** 2 / (f1 ** 2 - f2 ** 2)
        b = f2 ** 2 / (f1 ** 2 - f2 ** 2)
        corr_1, corr_2, total_dcb = split_dual_code_dsb_corrections_m(
            epoch,
            code_1,
            code_2,
            coeff_1=a,
            coeff_2=b,
            n=n,
        )
        corr[0] = corr_1
        corr[1] = corr_2
        return corr, total_dcb

    @staticmethod
    def _has_absolute_code_osb(epoch: pd.DataFrame, layout: list[dict[str, str]]) -> bool:
        return has_satellite_osb(epoch, (band["code_col"] for band in layout))

    def _corrected_code_observables(self, epoch: pd.DataFrame, layout: list[dict[str, str]], n: int):
        if len(layout) >= 2 and not self._has_absolute_code_osb(epoch, layout):
            code_dcb_corrs, dcb_total = self._dual_code_dsb_corrections_m(epoch, layout, n)
        else:
            code_dcb_corrs = [np.zeros(n, dtype=float) for _ in layout]
            dcb_total = np.zeros(n, dtype=float)

        clk = self._col_np(epoch, "clk")
        tro = self._col_np(epoch, "tro")
        ah = self._col_np(epoch, "ah_los")
        dprel = self._col_np(epoch, "dprel")
        tides = self._col_np(epoch, "tides_los")
        code_values = []
        for band_idx, band in enumerate(layout):
            rec_pco = self._col_np(epoch, band["rec_pco_col"])
            sat_pco = self._col_np(epoch, f"sat_pco_los_{band['freq']}")
            osb_code = osb_m(epoch, band["code_col"], n)
            code = (
                epoch[band["code_col"]].to_numpy(copy=False)
                - rec_pco
                + sat_pco
                + clk * self.CLIGHT
                - tro
                - ah
                - dprel
                - osb_code
                + code_dcb_corrs[band_idx]
                - tides
            )
            code_values.append(code)
        return code_values, code_dcb_corrs, dcb_total

    def _system_measurements(self, system, epoch, satellites, layout, num, strict_phase_code_gate=False):
        n = len(epoch)
        obs_dim = self._obs_dim(system)
        block = np.empty((n, obs_dim))
        r_diag = np.empty((n, obs_dim))
        clk = self._col_np(epoch, "clk")
        tro = self._col_np(epoch, "tro")
        ah = self._col_np(epoch, "ah_los")
        dprel = self._col_np(epoch, "dprel")
        tides = self._col_np(epoch, "tides_los")
        phw = self._col_np(epoch, "phw")
        ev_deg = self._col_np(epoch, "ev", np.full(n, 30.0))
        code_masks = []
        phase_masks = []
        code_values, code_dcb_corrs, dcb_total = self._corrected_code_observables(epoch, layout, n)

        for band_idx, band in enumerate(layout):
            rec_pco = self._col_np(epoch, band["rec_pco_col"])
            sat_pco = self._col_np(epoch, f"sat_pco_los_{band['freq']}")
            osb_phase = osb_m(epoch, band["phase_col"], n)
            freq = self.FREQ_DICT[band["freq"]]
            code = code_values[band_idx]
            phase = (
                epoch[band["phase_col"]].to_numpy(copy=False)
                - rec_pco
                + sat_pco
                + clk * self.CLIGHT
                - tro
                - ah
                - dprel
                - osb_phase
                - tides
                - (self.CLIGHT / freq) * phw
            )
            code_mask = self.code_screening(
                x=self.ekf.x[:3],
                satellites=satellites[:, :3],
                code_obs=code,
                thr=self.code_prefit_threshold,
            )
            code_masks.append(code_mask)
            phase_mask = code_mask if strict_phase_code_gate else np.ones(n, dtype=bool)
            phase_masks.append(phase_mask)
            block[:, 2 * band_idx] = code
            block[:, 2 * band_idx + 1] = phase
        code_mask_matrix = np.column_stack(code_masks) if code_masks else np.ones((n, 0), dtype=bool)
        phase_mask_matrix = np.column_stack(phase_masks) if phase_masks else np.ones((n, 0), dtype=bool)
        both_code_rejected = ~np.any(code_mask_matrix, axis=1)
        for band_idx in range(len(layout)):
            phase_mask_matrix[:, band_idx] = phase_mask_matrix[:, band_idx] & ~both_code_rejected

        inv_sin = _elevation_variance_scale(ev_deg)
        for band_idx in range(len(layout)):
            r_diag[:, 2 * band_idx] = np.where(code_mask_matrix[:, band_idx], inv_sin, 1e12)
            r_diag[:, 2 * band_idx + 1] = np.where(phase_mask_matrix[:, band_idx], 0.001 * inv_sin, 1e12)

        if self.use_iono_constr:
            block[:, -1] = epoch["ion"].to_numpy(copy=False)
            r_diag[:, -1] = _iono_constraint_sigma(
                epoch,
                num,
                self.interval,
                self.use_iono_rms,
                self.sigma_iono_0,
                self.sigma_iono_end,
                self.t_end,
            )

        if self.conservative_bds_weighting and system == "C":
            r_diag[:, 0::2] *= self.BDS_CODE_VAR_SCALE_MIXED_NOCONSTR
            r_diag[:, 1::2] *= self.BDS_PHASE_VAR_SCALE_MIXED_NOCONSTR

        return block.reshape(-1), r_diag.reshape(-1), {
            "code_masks": code_mask_matrix,
            "phase_masks": phase_mask_matrix,
            "phase_used": np.any(phase_mask_matrix, axis=1),
            "strict_phase_code_gate": strict_phase_code_gate,
            "layout": layout,
            "code_dcb_corrs": code_dcb_corrs,
            "dcb_total": dcb_total,
        }

    def _model_iono(self, epoch):
        if "ion" in epoch.columns:
            ion = epoch["ion"].to_numpy(dtype=float, copy=False)
            if np.any(np.isfinite(ion)):
                return np.nan_to_num(ion, nan=0.0, posinf=0.0, neginf=0.0)
        return np.zeros(len(epoch), dtype=float)

    def _estimate_system_iono_datum(self, system, epoch, layout):
        if (
            self.use_iono_constr
            or len(layout) < 2
            or "ion" not in epoch.columns
            or self._has_absolute_code_osb(epoch, layout)
        ):
            return 0.0
        n = len(epoch)
        ion_model = self._model_iono(epoch)
        code_values, _, _ = self._corrected_code_observables(epoch, layout, n)
        ref_signal = layout[0]["freq"]
        ref_code = code_values[0]
        ref_freq = self.FREQ_DICT[ref_signal]
        offsets = []
        for band_idx, band in enumerate(layout[1:], start=1):
            freq = self.FREQ_DICT[band["freq"]]
            mu = (ref_freq / freq) ** 2
            denom = mu - 1.0
            if abs(denom) < 1e-12:
                continue
            merged_iono = (code_values[band_idx] - ref_code) / denom
            samples = merged_iono - ion_model
            finite = np.isfinite(samples)
            if np.any(finite):
                offsets.append(float(np.nanmedian(samples[finite])))
        if not offsets:
            return 0.0
        return float(np.median(offsets))

    def _initial_iono(self, system, epoch):
        return self._model_iono(epoch)

    def _initial_signal_bias_values(self, system, epoch, layout):
        if not self.estimate_signal_bias or len(layout) < 2:
            return {}

        n = len(epoch)
        ion_model = self._model_iono(epoch)
        code_values, _, _ = self._corrected_code_observables(epoch, layout, n)
        ref_signal = layout[0]["freq"]
        ref_code = code_values[0]
        ref_freq = self.FREQ_DICT[ref_signal]
        bias_values = {}
        for band_idx, band in enumerate(layout[1:], start=1):
            freq = self.FREQ_DICT[band["freq"]]
            mu = (ref_freq / freq) ** 2
            gf = code_values[band_idx] - ref_code
            samples = gf - (mu - 1.0) * ion_model
            finite = np.isfinite(samples)
            if not np.any(finite):
                continue
            bias_values[band_idx] = float(np.nanmedian(samples[finite]))
        return bias_values

    def _init_filter_common(self, epoch, clk0=0.0, zwd0=0.0):
        sats_by_system = {
            system: self.obs_by_system[system]
            .loc[(slice(None), epoch), :]
            .index.get_level_values("sv")
            .tolist()
            for system in self.systems
        }
        _, dim_x = self._state_specs(sats_by_system)
        dim_z = sum(self._obs_dim(system) * len(sats_by_system[system]) for system in self.systems)
        x0 = np.zeros(dim_x)
        x0[:3] = self.pos0 if self.pos0 is not None else np.array([6371.0e3, 0.0, 0.0])
        x0[3] = clk0
        if self.tro:
            x0[self.col_ztd] = zwd0

        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[:3, :3] = np.eye(3) * self.cfg.p_crd
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        for system, col in self.isb_cols.items():
            self.ekf.P[col, col] = self.cfg.p_isb
            self.ekf.Q[col, col] = self.cfg.q_isb
        if self.tro:
            self.ekf.P[self.col_ztd, self.col_ztd] = self.cfg.p_tro
            self.ekf.Q[self.col_ztd, self.col_ztd] = self.cfg.q_tro
        for band_cols in self.bias_state_cols.values():
            for band_idx, col in band_cols.items():
                if self.anchor_first_signal_bias and band_idx == 0:
                    self.ekf.x[col] = 0.0
                    self.ekf.P[col, col] = 1e-12
                    self.ekf.Q[col, col] = 0.0
                else:
                    self.ekf.P[col, col] = self.cfg.p_dcb
                    self.ekf.Q[col, col] = self.signal_bias_q
        if getattr(self, "layouts", None):
            for system in self.systems:
                epoch_df = self.obs_by_system[system].loc[(slice(None), epoch), :]
                for band_idx, value in self._initial_signal_bias_values(system, epoch_df, self.layouts[system]).items():
                    col = self._signal_bias_col(system, band_idx)
                    if col is not None:
                        self.ekf.x[col] = value
        self._configure_transition_matrix()

        for system in self.systems:
            epoch_df = self.obs_by_system[system].loc[(slice(None), epoch), :]
            self._init_system_state(system, sats_by_system, self._initial_iono(system, epoch_df))
        return sats_by_system

    def _init_system_state(self, system, sats_by_system, ion_values):
        start = self._system_state_start(sats_by_system, system)
        state_dim = self._state_dim(system)
        for i, _sv in enumerate(sats_by_system[system]):
            idx_i = start + state_dim * i
            self.ekf.x[idx_i] = ion_values[i]
            self.ekf.P[idx_i, idx_i] = self.cfg.p_iono
            self.ekf.Q[idx_i, idx_i] = self.cfg.q_iono * (self.interval * 60) / 3600
            for amb in range(1, state_dim):
                self.ekf.P[idx_i + amb, idx_i + amb] = self.cfg.p_amb
                self.ekf.Q[idx_i + amb, idx_i + amb] = getattr(self.cfg, "q_amb", 0.0)

    def init_filter(self, clk0=0.0, zwd0=0.0):
        self.layouts = self._prepare_obs()
        epoch_sets = [
            set(self.obs_by_system[system].index.get_level_values("time").unique())
            for system in self.systems
        ]
        epochs = sorted(set.intersection(*epoch_sets))
        if not epochs:
            raise ValueError(f"No common epochs for mixed uncombined systems {self.systems}.")
        return self._init_filter_common(epochs[0], clk0=clk0, zwd0=zwd0), epochs

    def _configure_transition_matrix(self):
        self.ekf.F = np.eye(self.ekf.dim_x)
        self.ekf.F[3, 3] = 0.0

    def reset_filter(self, epoch, clk0=0.0, zwd0=0.0):
        return self._init_filter_common(epoch, clk0=clk0, zwd0=zwd0)

    def rebuild_state(self, x_old, P_old, Q_old, prev_sats, curr_sats, ion_by_system):
        old_specs, _ = self._state_specs(prev_sats)
        new_specs, new_dim = self._state_specs(curr_sats)
        x_new = np.zeros(new_dim)
        P_new = np.zeros((new_dim, new_dim))
        Q_new = np.zeros((new_dim, new_dim))
        x_new[:self.base_dim] = x_old[:self.base_dim]
        P_new[:self.base_dim, :self.base_dim] = P_old[:self.base_dim, :self.base_dim]
        Q_new[:self.base_dim, :self.base_dim] = Q_old[:self.base_dim, :self.base_dim]

        for tag, (new_start, new_dim_state) in new_specs.items():
            if tag in old_specs:
                old_start, old_dim_state = old_specs[tag]
                common_dim = min(old_dim_state, new_dim_state)
                old_idx = np.arange(old_start, old_start + common_dim)
                new_idx = np.arange(new_start, new_start + common_dim)
                x_new[new_idx] = x_old[old_idx]
                P_new[np.ix_(new_idx, new_idx)] = P_old[np.ix_(old_idx, old_idx)]
                Q_new[np.ix_(new_idx, new_idx)] = Q_old[np.ix_(old_idx, old_idx)]
                base_idx = np.arange(self.base_dim)
                P_new[np.ix_(new_idx, base_idx)] = P_old[np.ix_(old_idx, base_idx)]
                P_new[np.ix_(base_idx, new_idx)] = P_old[np.ix_(base_idx, old_idx)]
                Q_new[np.ix_(new_idx, base_idx)] = Q_old[np.ix_(old_idx, base_idx)]
                Q_new[np.ix_(base_idx, new_idx)] = Q_old[np.ix_(base_idx, old_idx)]
                continue

            system, sv = tag.split(":", 1)
            sat_idx = curr_sats[system].index(sv)
            x_new[new_start] = ion_by_system[system][sat_idx]
            P_new[new_start, new_start] = self.cfg.p_iono
            Q_new[new_start, new_start] = self.cfg.q_iono * (self.interval * 60) / 3600
            for amb in range(1, new_dim_state):
                P_new[new_start + amb, new_start + amb] = self.cfg.p_amb
                Q_new[new_start + amb, new_start + amb] = getattr(self.cfg, "q_amb", 0.0)

        return x_new, P_new, Q_new

    def code_screening(self, x, satellites, code_obs, thr=1):
        sat_xyz = np.asarray(satellites, dtype=float)
        ref_xyz = np.asarray(x, dtype=float)
        dist = np.linalg.norm(sat_xyz - ref_xyz, axis=1)
        prefit = code_obs - dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad > n_sat / 2:
            mask = np.ones(n_sat, dtype=bool)
        return mask

    def _system_state_values(self, system, sats_by_system):
        n_sats = len(sats_by_system[system])
        state_dim = self._state_dim(system)
        if n_sats == 0:
            return np.empty((0, state_dim))
        start = self._system_state_start(sats_by_system, system)
        return self.ekf.x[start:start + state_dim * n_sats].reshape(n_sats, state_dim)

    def _ambiguity_groups(self, sats_by_system, epoch_by_system, age_by_system=None):
        groups = []
        for system in self.systems:
            n_sats = len(sats_by_system[system])
            if not n_sats:
                continue
            state_dim = self._state_dim(system)
            state_start = self._system_state_start(sats_by_system, system)
            sat_idx = np.arange(n_sats)
            n1_idx = state_start + state_dim * sat_idx + 1
            n2_idx = state_start + state_dim * sat_idx + 2 if state_dim > 2 else None
            group = {
                "n1_idx": n1_idx,
                "n2_idx": n2_idx,
                "ev": epoch_by_system[system]["ev"].to_numpy(copy=False),
            }
            if age_by_system is not None:
                group["age"] = arc_age_array(age_by_system.get(system, {}), sats_by_system[system])
            groups.append(group)
        return groups

    @staticmethod
    def _residual_stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.nan, np.nan
        rms = float(np.sqrt(np.mean(np.square(values))))
        med = float(np.median(values))
        scatter = float(np.sqrt(np.mean(np.square(values - med))))
        return rms, scatter

    def _system_residual_output(self, system, epoch, prefit_block, postfit_block, diag, state_values):
        out = epoch.copy()
        out["system"] = system
        out["reference_system"] = self.reference_system
        out["system_isb"] = 0.0 if system == self.reference_system else self.ekf.x[self.isb_cols[system]]
        out["dtr"] = self.ekf.x[3]
        out["ztd"] = self.ekf.x[self.col_ztd] if self.tro else 0.0
        for band_idx, _band in enumerate(diag["layout"], start=1):
            out[f"prefit_code_res_{band_idx}"] = prefit_block[:, 2 * (band_idx - 1)]
            out[f"prefit_phase_res_{band_idx}"] = prefit_block[:, 2 * (band_idx - 1) + 1]
            out[f"postfit_code_res_{band_idx}"] = postfit_block[:, 2 * (band_idx - 1)]
            out[f"postfit_phase_res_{band_idx}"] = postfit_block[:, 2 * (band_idx - 1) + 1]
            out[f"code_used_{band_idx}"] = diag["code_masks"][:, band_idx - 1]
            out[f"phase_used_{band_idx}"] = diag["phase_masks"][:, band_idx - 1]
            out[f"code_dcb_corr_{band_idx}"] = diag["code_dcb_corrs"][band_idx - 1]
            bias_col = self._signal_bias_col(system, band_idx - 1)
            out[f"signal_bias_{band_idx}"] = 0.0 if bias_col is None else self.ekf.x[bias_col]
        out["phase_used"] = diag["phase_used"]
        out["strict_phase_code_gate"] = diag["strict_phase_code_gate"]
        out["dcb_total_m"] = diag["dcb_total"]
        out["I_state"] = state_values[:, 0]
        for amb_idx in range(1, state_values.shape[1]):
            out[f"N{amb_idx}_state"] = state_values[:, amb_idx]
        return out

    def run_filter(self, clk0=0.0, ref=None, flh=None, zwd0=0.0, trace_filter=False, reset_every=0):
        old_sats_by_system, all_times = self.init_filter(clk0=clk0, zwd0=zwd0)
        layouts = self.layouts
        for system in self.systems:
            self.obs_by_system[system] = self.obs_by_system[system].sort_values(by="sv")
        epoch_maps = {
            system: {t: df for t, df in self.obs_by_system[system].groupby(level=1, sort=False)}
            for system in self.systems
        }
        result_rows = []
        result_times = []
        residuals = {system: [] for system in self.systems}
        r_cache = {}
        xyz = None
        conv_time = None
        T0 = all_times[0]
        age_by_system = {
            system: {sv: 0 for sv in old_sats_by_system[system]}
            for system in self.systems
        }
        phase_traces = {
            system: {band_idx: {} for band_idx in range(len(layouts[system]))}
            for system in self.systems
        }
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

        for num, t in enumerate(all_times):
            reset_epoch = False
            if reset_every != 0 and ((num * self.interval) % reset_every == 0) and (num != 0):
                old_sats_by_system = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                age_by_system = {
                    system: {sv: 0 for sv in old_sats_by_system[system]}
                    for system in self.systems
                }
                reset_epoch = True
                T0 = t

            epoch_by_system = {system: epoch_maps[system].get(t) for system in self.systems}
            if any(epoch is None for epoch in epoch_by_system.values()):
                continue

            curr_sats_by_system = {
                system: epoch_by_system[system].index.get_level_values("sv").tolist()
                for system in self.systems
            }
            ion_by_system = {
                system: self._initial_iono(system, epoch_by_system[system])
                for system in self.systems
            }
            if curr_sats_by_system != old_sats_by_system:
                self.ekf.x, self.ekf.P, self.ekf.Q = self.rebuild_state(
                    x_old=self.ekf.x.copy(),
                    P_old=self.ekf.P.copy(),
                    Q_old=self.ekf.Q.copy(),
                    prev_sats=old_sats_by_system,
                    curr_sats=curr_sats_by_system,
                    ion_by_system=ion_by_system,
                )
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = sum(
                    self._obs_dim(system) * len(curr_sats_by_system[system])
                    for system in self.systems
                )
                self.ekf._I = np.eye(self.ekf.dim_x)
                self._configure_transition_matrix()
            old_sats_by_system = curr_sats_by_system

            satellite_blocks = [
                epoch_by_system[system][["xe", "ye", "ze", "me_wet"]].to_numpy()
                for system in self.systems
            ]
            def build_epoch_measurements(strict_phase_code_gate=False):
                z_blocks = []
                r_blocks = []
                diagnostics = {}
                for system, satellites in zip(self.systems, satellite_blocks):
                    z_sys, r_sys, diag = self._system_measurements(
                        system,
                        epoch_by_system[system],
                        satellites,
                        layouts[system],
                        num,
                        strict_phase_code_gate=(
                            strict_phase_code_gate or system in self.phase_code_gate_systems
                        ),
                    )
                    z_blocks.append(z_sys)
                    r_blocks.append(r_sys)
                    diagnostics[system] = diag
                z_epoch = np.concatenate(z_blocks)
                r_epoch = np.nan_to_num(np.concatenate(r_blocks), nan=1e12, posinf=1e12, neginf=1e12)
                prefit_epoch = z_epoch - self.Hx(self.ekf.x, *satellite_blocks)
                return z_epoch, prefit_epoch, r_epoch, diagnostics

            Z, prefit, r_diag, diag_by_system = build_epoch_measurements(strict_phase_code_gate=False)
            if not np.all(np.isfinite(Z)):
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "non-finite-observation",
                        epoch=num,
                        time=t,
                    )
                continue

            phase_resets_by_system = {system: 0 for system in self.systems}
            phase_reset_svs_by_system = {system: set() for system in self.systems}
            if self.enable_phase_jump_resets:
                cursor = 0
                z_cursor = 0
                for system in self.systems:
                    n_sats = len(curr_sats_by_system[system])
                    obs_dim = self._obs_dim(system)
                    size = obs_dim * n_sats
                    pre_block = prefit[cursor:cursor + size].reshape(n_sats, obs_dim)
                    z_block = Z[z_cursor:z_cursor + size].reshape(n_sats, obs_dim)
                    cursor += size
                    z_cursor += size
                    if n_sats == 0:
                        continue
                    if getattr(self, "legacy_phase_jump_resets", False):
                        _, rho_phase = _unit_vectors_and_ranges(
                            satellite_blocks[self.systems.index(system)][:, :3],
                            self.ekf.x[:3],
                        )
                    state_dim = self._state_dim(system)
                    state_start = self._system_state_start(curr_sats_by_system, system)
                    for band_idx in range(len(layouts[system])):
                        if getattr(self, "legacy_phase_jump_resets", False):
                            residual_source = z_block[:, 2 * band_idx + 1] - rho_phase
                        else:
                            residual_source = pre_block[:, 2 * band_idx + 1]
                        phase_traces[system][band_idx][num] = {
                            sv: float(residual)
                            for sv, residual in zip(
                                curr_sats_by_system[system],
                                residual_source,
                            )
                        }
                        if num < 60:
                            continue
                        for idx in phase_residuals_outliers(
                            curr_sats_by_system[system],
                            phase_traces[system][band_idx],
                            num,
                            thr=self.phase_jump_reset_threshold,
                        ):
                            amb_col = state_start + state_dim * idx + 1 + band_idx
                            self.ekf.x[amb_col] = 0.0
                            self.ekf.P[amb_col, amb_col] = float(
                                max(getattr(self.cfg, "p_amb", 1e6), 1e3)
                            )
                            phase_resets_by_system[system] += 1
                            phase_reset_svs_by_system[system].add(curr_sats_by_system[system][idx])

            def set_measurement_noise(diag_values):
                m = diag_values.size
                R = r_cache.get(m)
                if R is None:
                    R = np.zeros((m, m), dtype=np.float64)
                    r_cache[m] = R
                else:
                    R.fill(0.0)
                R.flat[::m + 1] = diag_values
                self.ekf.R = R
                return m

            m = set_measurement_noise(r_diag)
            updated = False
            for k in range(5):
                try:
                    self.ekf.predict_update(
                        z=Z,
                        HJacobian=self.Hjacobian,
                        args=tuple(satellite_blocks),
                        Hx=self.Hx,
                        hx_args=tuple(satellite_blocks),
                    )
                    updated = True
                    break
                except np.linalg.LinAlgError:
                    self.ekf.R.flat[::m + 1] += 1e-6 * (10.0 ** k)
            if not updated:
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "singular-update-reset",
                        epoch=num,
                        time=t,
                    )
                old_sats_by_system = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                age_by_system = {
                    system: {sv: 0 for sv in old_sats_by_system[system]}
                    for system in self.systems
                }
                continue

            for system in self.systems:
                age_by_system[system] = advance_arc_age(
                    age_by_system.get(system, {}),
                    curr_sats_by_system[system],
                    phase_reset_svs_by_system[system],
                )

            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_indexed_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    ambiguity_groups=self._ambiguity_groups(curr_sats_by_system, epoch_by_system, age_by_system),
                    settings=ar_cfg,
                )

            postfit = Z - self.Hx(self.ekf.x, *satellite_blocks)
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0
                    elif position_diff > 0.005:
                        conv_time = None

            xyz = self.ekf.x[:3].copy()
            dtr = self.ekf.x[3]
            ztd = self.ekf.x[self.col_ztd] if self.tro else 0.0
            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
            else:
                enu = np.zeros(3)

            result_row = {
                "de": enu[0],
                "dn": enu[1],
                "du": enu[2],
                "dtr": dtr,
                "ztd": ztd,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "reference_system": self.reference_system,
                "n_states": int(len(self.ekf.x)),
                "n_sats_total": int(sum(len(curr_sats_by_system[s]) for s in self.systems)),
                "n_phase_reset": int(sum(phase_resets_by_system.values())),
                "ar_fixed": 0 if ar_diag is None else ar_diag.fixed_ambiguities,
                "ar_ratio": np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min,
                "ar_ok": False if ar_diag is None else ar_diag.accepted,
            }
            for system in self.isb_systems:
                result_row[f"isb_{system}"] = self.ekf.x[self.isb_cols[system]]
            for system in self.systems:
                result_row[f"n_phase_reset_{system}"] = phase_resets_by_system[system]
                ages = arc_age_array(age_by_system.get(system, {}), curr_sats_by_system[system])
                result_row[f"ar_{system}_min_age"] = int(np.min(ages)) if ages.size else np.nan
            for system, band_cols in self.bias_state_cols.items():
                for band_idx, col in band_cols.items():
                    result_row[f"signal_bias_{system}_{band_idx + 1}"] = self.ekf.x[col]

            code_all = []
            phase_all = []
            cursor = 0
            for system in self.systems:
                n_sats = len(curr_sats_by_system[system])
                obs_dim = self._obs_dim(system)
                size = obs_dim * n_sats
                pre_block = prefit[cursor:cursor + size].reshape(n_sats, obs_dim)
                post_block = postfit[cursor:cursor + size].reshape(n_sats, obs_dim)
                cursor += size
                diag = diag_by_system[system]
                state_values = self._system_state_values(system, curr_sats_by_system)
                residuals[system].append(
                    self._system_residual_output(
                        system,
                        epoch_by_system[system],
                        pre_block,
                        post_block,
                        diag,
                        state_values,
                    )
                )

                code_values = []
                for band_idx in range(len(diag["layout"])):
                    mask = diag["code_masks"][:, band_idx]
                    code_values.append(post_block[:, 2 * band_idx][mask])
                    code_all.append(post_block[:, 2 * band_idx][mask])
                phase_values = []
                for band_idx in range(len(diag["layout"])):
                    phase_values.append(post_block[:, 2 * band_idx + 1][diag["phase_used"]])
                    phase_all.append(post_block[:, 2 * band_idx + 1][diag["phase_used"]])
                code_values = np.concatenate(code_values) if code_values else np.empty(0)
                phase_values = np.concatenate(phase_values) if phase_values else np.empty(0)
                code_rms, code_scatter = self._residual_stats(code_values)
                phase_rms, phase_scatter = self._residual_stats(phase_values)
                result_row[f"n_sats_{system}"] = n_sats
                result_row[f"n_code_rejected_{system}"] = int(np.count_nonzero(~diag["code_masks"]))
                result_row[f"code_res_scatter_rms_{system}"] = code_scatter
                result_row[f"phase_res_scatter_rms_{system}"] = phase_scatter
                result_row[f"code_res_rms_{system}"] = code_rms
                result_row[f"phase_res_rms_{system}"] = phase_rms

            code_all = np.concatenate(code_all) if code_all else np.empty(0)
            phase_all = np.concatenate(phase_all) if phase_all else np.empty(0)
            result_row["n_code_rejected_total"] = int(sum(
                result_row[f"n_code_rejected_{system}"] for system in self.systems
            ))
            result_row["code_res_rms"], result_row["code_res_scatter_rms"] = self._residual_stats(code_all)
            result_row["phase_res_rms"], result_row["phase_res_scatter_rms"] = self._residual_stats(phase_all)
            result_row.update(pppar_diagnostic_columns(ar_diag))
            result_rows.append(result_row)
            result_times.append(t)
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result_row,
                )

        ct = conv_time.total_seconds() / 3600 if conv_time is not None else None
        result = pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name="time"))
        result["ct_min"] = ct
        residual_map = {
            system: (pd.concat(frames) if frames else self.obs_by_system[system].iloc[0:0].copy())
            for system, frames in residuals.items()
        }
        return result, residual_map, None, ct


class PPPFilterMultiGNSSIonConstGEC(PPPUdGenericMixedGNSS):
    """Generic G/E/C uncombined PPP with ionospheric constraints.

    Purpose:
        Active/developing constrained branch for mixed GPS, Galileo and BeiDou.
        It is selected for constrained multi-system runs that include BeiDou,
        request the generic ``gec`` constraint model, or use broadcast routing
        where the generic path is required.

    Model:
        Inherits the generic mixed uncombined state layout and adds one
        ionosphere pseudo-observation per satellite from the ``ion`` column.
        Constraint sigmas follow the same ramp/RMS policy used by the legacy
        constrained branch.

    State vector:
        ``[x, y, z, dtr_ref, ISB_nonref..., ztd?, per-satellite (I, N1, N2)
        blocks..., optional system code-bias datum states]`` as defined by the
        generic superclass.

    Supported systems:
        GPS, Galileo and BeiDou, including G/E/C combinations supported by the
        configured modes and preprocessing.

    Status:
        Active G/E/C constrained path, still less historically validated than
        ``PPPFilterMultiGNSSIonConst`` for GPS+Galileo.
        Treat as active/developing rather than legacy.

    Limitations:
        Do not treat this as a drop-in replacement for the legacy G/E reference
        branch without numerical regression tests. BeiDou support depends on
        consistent BDS orbit/clock and bias products.
    """

    def __init__(
        self,
        config: PPPConfig,
        obs_by_system: dict[str, pd.DataFrame],
        mode_by_system: dict[str, str],
        ekf,
        pos0,
        tro=True,
        interval=0.5,
        reference_system: str | None = None,
    ):
        super().__init__(
            config=config,
            obs_by_system=obs_by_system,
            mode_by_system=mode_by_system,
            ekf=ekf,
            pos0=pos0,
            tro=tro,
            interval=interval,
            reference_system=reference_system,
            estimate_signal_bias=False,
        )
        if not self.use_iono_constr:
            raise ValueError("PPPFilterMultiGNSSIonConstGEC requires ionospheric constraints.")
        self.code_prefit_threshold = 10.0
        self.enable_phase_jump_resets = True
        self.legacy_phase_jump_resets = True
        self.phase_jump_reset_threshold = 1.0

    def _state_specs(self, sats_by_system: dict[str, list[str]]) -> tuple[dict[str, tuple[int, int]], int]:
        specs: dict[str, tuple[int, int]] = {}
        offset = self.base_dim
        for system in self.systems:
            for sv in sats_by_system.get(system, []):
                specs[f"{system}:{sv}"] = (offset, 3)
                offset += 3
        offset += 2 * len(self.systems)
        return specs, offset

    def _state_dim(self, system: str) -> int:
        signals = _mode_signals(self.mode_by_system[system])
        if len(signals) != 2:
            raise ValueError("PPPFilterMultiGNSSIonConstGEC currently supports dual-frequency modes only.")
        return 3

    def _dcb_cols(self, sats_by_system: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
        offset = self.base_dim + sum(3 * len(sats_by_system.get(system, [])) for system in self.systems)
        return {system: (offset + 2 * idx, offset + 2 * idx + 1) for idx, system in enumerate(self.systems)}

    def _init_legacy_dcb_states(self, sats_by_system: dict[str, list[str]]) -> None:
        for _system, (idx_c1, idx_c2) in self._dcb_cols(sats_by_system).items():
            self.ekf.x[idx_c1] = 0.0
            self.ekf.P[idx_c1, idx_c1] = 1e-12
            self.ekf.Q[idx_c1, idx_c1] = 0.0
            self.ekf.P[idx_c2, idx_c2] = self.cfg.p_dcb
            self.ekf.Q[idx_c2, idx_c2] = self.cfg.q_dcb

    def _init_filter_common(self, epoch, clk0=0.0, zwd0=0.0):
        sats_by_system = super()._init_filter_common(epoch, clk0=clk0, zwd0=zwd0)
        self._init_legacy_dcb_states(sats_by_system)
        return sats_by_system

    def rebuild_state(self, x_old, P_old, Q_old, prev_sats, curr_sats, ion_by_system):
        _, new_dim = self._state_specs(curr_sats)
        x_new = np.zeros(new_dim)
        P_new = np.zeros((new_dim, new_dim))
        Q_new = np.zeros((new_dim, new_dim))

        base = self.base_dim
        x_new[:base] = x_old[:base]
        P_new[:base, :base] = P_old[:base, :base]
        Q_new[:base, :base] = Q_old[:base, :base]

        prev_all = [
            f"{system}:{sv}"
            for system in self.systems
            for sv in prev_sats.get(system, [])
        ]
        curr_all = [
            f"{system}:{sv}"
            for system in self.systems
            for sv in curr_sats.get(system, [])
        ]
        prev_map = {tag: idx for idx, tag in enumerate(prev_all)}
        curr_map = {tag: idx for idx, tag in enumerate(curr_all)}
        common = [tag for tag in curr_all if tag in prev_map]

        if common:
            old_idx = np.array([prev_map[tag] for tag in common], dtype=int)
            new_idx = np.array([curr_map[tag] for tag in common], dtype=int)
            offs = np.array([0, 1, 2], dtype=int)
            old_state = (base + 3 * old_idx[:, None] + offs[None, :]).ravel()
            new_state = (base + 3 * new_idx[:, None] + offs[None, :]).ravel()
            x_new[new_state] = x_old[old_state]
            P_new[np.ix_(new_state, new_state)] = P_old[np.ix_(old_state, old_state)]
            Q_new[np.ix_(new_state, new_state)] = Q_old[np.ix_(old_state, old_state)]
            base_idx = np.arange(base)
            P_new[np.ix_(new_state, base_idx)] = P_old[np.ix_(old_state, base_idx)]
            P_new[np.ix_(base_idx, new_state)] = P_old[np.ix_(base_idx, old_state)]
            Q_new[np.ix_(new_state, base_idx)] = Q_old[np.ix_(old_state, base_idx)]
            Q_new[np.ix_(base_idx, new_state)] = Q_old[np.ix_(base_idx, old_state)]

        for tag in curr_all:
            if tag in prev_map:
                continue
            idx = curr_map[tag]
            idx_i = base + 3 * idx
            idx_n1 = idx_i + 1
            idx_n2 = idx_i + 2
            x_new[idx_i] = 0.0
            P_new[idx_i, idx_i] = self.cfg.p_iono
            Q_new[idx_i, idx_i] = self.cfg.q_iono * (self.interval * 60) / 3600
            P_new[idx_n1, idx_n1] = self.cfg.p_amb
            P_new[idx_n2, idx_n2] = self.cfg.p_amb

        old_dcb = self._dcb_cols(prev_sats)
        new_dcb = self._dcb_cols(curr_sats)
        for system in self.systems:
            if system in old_dcb and system in new_dcb:
                old_idx = np.array(old_dcb[system], dtype=int)
                new_idx = np.array(new_dcb[system], dtype=int)
                x_new[new_idx] = x_old[old_idx]
                P_new[np.ix_(new_idx, new_idx)] = P_old[np.ix_(old_idx, old_idx)]
                Q_new[np.ix_(new_idx, new_idx)] = Q_old[np.ix_(old_idx, old_idx)]
        for _system, (idx_c1, idx_c2) in new_dcb.items():
            x_new[idx_c1] = 0.0
            P_new[idx_c1, idx_c1] = 1e-12
            Q_new[idx_c1, idx_c1] = 0.0
            if P_new[idx_c2, idx_c2] == 0.0:
                P_new[idx_c2, idx_c2] = self.cfg.p_dcb
                Q_new[idx_c2, idx_c2] = self.cfg.q_dcb
        return x_new, P_new, Q_new

    def _fill_h_system(self, H, x, satellites, system, row_start, state_start):
        mode = self.mode_by_system[system]
        signals = _mode_signals(mode)
        obs_dim = self._obs_dim(system)
        f_ref = self.FREQ_DICT[signals[0]]
        rec_xyz = x[:3].copy()
        e_vec, _ = _unit_vectors_and_ranges(satellites[:, :3], rec_xyz)
        m_wet = satellites[:, 3]

        dcb_offset = H.shape[1] - 2 * len(self.systems)
        dcb_cols = {current: (dcb_offset + 2 * idx, dcb_offset + 2 * idx + 1)
                    for idx, current in enumerate(self.systems)}

        for sat_idx in range(satellites.shape[0]):
            row = row_start + obs_dim * sat_idx
            state = state_start + 3 * sat_idx
            for band_idx, signal in enumerate(signals):
                code_row = row + 2 * band_idx
                phase_row = code_row + 1
                freq = self.FREQ_DICT[signal]
                mu = (f_ref / freq) ** 2
                lam = self.CLIGHT / freq
                H[[code_row, phase_row], 0:3] = e_vec[sat_idx]
                H[[code_row, phase_row], 3] = 1.0
                if system != self.reference_system:
                    H[[code_row, phase_row], self.isb_cols[system]] = 1.0
                if self.tro:
                    H[[code_row, phase_row], self.col_ztd] = m_wet[sat_idx]
                H[code_row, state] = mu
                H[phase_row, state] = -mu
                H[phase_row, state + 1 + band_idx] = lam
                H[[code_row, phase_row], dcb_cols[system][band_idx]] = 1.0
            H[row + 2 * len(signals), state] = 1.0

    def _fill_hx_system(self, z_hat, x, satellites, system, row_start, state_start):
        mode = self.mode_by_system[system]
        signals = _mode_signals(mode)
        obs_dim = self._obs_dim(system)
        f_ref = self.FREQ_DICT[signals[0]]
        clk = x[3]
        isb = 0.0 if system == self.reference_system else x[self.isb_cols[system]]
        zwd = x[self.col_ztd] if self.tro else 0.0
        _, rho = _unit_vectors_and_ranges(satellites[:, :3], x[:3])
        mw = satellites[:, 3] if self.tro else 0.0
        n = satellites.shape[0]
        state = x[state_start:state_start + 3 * n].reshape(n, 3)
        dcb_offset = len(x) - 2 * len(self.systems)
        dcb_cols = {current: (dcb_offset + 2 * idx, dcb_offset + 2 * idx + 1)
                    for idx, current in enumerate(self.systems)}
        dcb_values = x[list(dcb_cols[system])]
        block = np.empty((n, obs_dim))
        base = rho + clk + isb + mw * zwd
        for band_idx, signal in enumerate(signals):
            freq = self.FREQ_DICT[signal]
            mu = (f_ref / freq) ** 2
            lam = self.CLIGHT / freq
            block[:, 2 * band_idx] = base + mu * state[:, 0] + dcb_values[band_idx]
            block[:, 2 * band_idx + 1] = base - mu * state[:, 0] + lam * state[:, 1 + band_idx]
        block[:, -1] = state[:, 0]
        z_hat[row_start:row_start + obs_dim * n] = block.reshape(-1)

    def _configure_transition_matrix(self):
        self.ekf.F = np.eye(self.ekf.dim_x)

    def _system_measurements(self, system, epoch, satellites, layout, num, strict_phase_code_gate=False):
        block, r_diag, diag = super()._system_measurements(
            system=system,
            epoch=epoch,
            satellites=satellites,
            layout=layout,
            num=num,
            strict_phase_code_gate=strict_phase_code_gate,
        )
        n = len(epoch)
        obs_dim = self._obs_dim(system)
        if n == 0:
            return block, r_diag, diag

        r_matrix = r_diag.reshape(n, obs_dim)
        ev_deg = self._col_np(epoch, "ev", np.full(n, 30.0))
        inv_sin = _elevation_variance_scale(ev_deg)
        legacy_code_var = 0.3 + 0.0025 * inv_sin
        legacy_phase_var = 1e-4 + 0.0003 * inv_sin
        for band_idx in range(len(layout)):
            r_matrix[:, 2 * band_idx] = np.where(
                diag["code_masks"][:, band_idx],
                legacy_code_var,
                1e12,
            )
            r_matrix[:, 2 * band_idx + 1] = np.where(
                diag["phase_masks"][:, band_idx],
                legacy_phase_var,
                1e12,
            )
        return block, r_matrix.reshape(-1), diag


class PPPUdMixedGNSS:
    """Specific mixed two-system uncombined PPP filter.

    Purpose:
        Older/specialized mixed-system class used by routing for several
        two-system combinations and mixed single/dual-frequency cases. It is
        still reachable and should be considered maintained for compatibility,
        but not the preferred generic G/E/C implementation.

    Status:
        Requires caution. This is a legacy/specific two-system compatibility
        path that remains reachable from ``PPPSession``. It is not a removal
        candidate until its routed combinations are covered by the generic
        mixed implementation and regression tests.

    Model:
        Uses uncombined code/phase observables for two systems represented by
        the historical ``gps`` and ``gal`` argument names. Depending on the
        selected modes, each satellite block contains ionosphere plus one or
        two ambiguity states. With ``use_iono_constr`` it adds ionospheric
        pseudo-observations from ``ion``/``ion_rms`` columns.

    State vector:
        ``[x, y, z, dtr_first, isb_second, ztd?, per-satellite blocks...]``.

    Supported systems:
        Historically GPS/Galileo, but routing may pass GPS/BeiDou or
        Galileo/BeiDou through the same two-system slots.

    PPP-AR support:
        Limited to paths wired through indexed uncombined AR; treat acceptance
        diagnostics as validation aids rather than proof that all mixed
        frequency combinations are equally mature.

    Limitations:
        Requires caution before edits because the argument names no longer
        fully describe the possible systems. Prefer ``PPPUdGenericMixedGNSS``
        for new no-constraint G/E/C work.
    """

    def __init__(self, config: PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, interval=0.5):
        self.cfg = config
        self.gps_obs = gps_obs
        self.gps_mode = gps_mode
        self.gal_obs = gal_obs
        self.gal_mode = gal_mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.CLIGHT = 299792458
        self.base_dim = 6 if self.tro else 5
        self.pos0 = pos0
        self.interval = interval
        self.use_iono_constr = bool(getattr(self.cfg, "use_iono_constr", False))
        self.use_iono_rms = bool(getattr(self.cfg, "use_iono_rms", True))
        self.sigma_iono_0 = getattr(self.cfg, "sigma_iono_0", 1.1)
        self.sigma_iono_end = getattr(self.cfg, "sigma_iono_end", 3.0)
        self.t_end = getattr(self.cfg, "t_end", 30)
        if self.use_iono_constr:
            if 'ion' not in self.gps_obs.columns or 'ion' not in self.gal_obs.columns:
                raise ValueError("use_iono_constr requires an 'ion' column in GPS and Galileo observations.")
            self.gps_obs = self.gps_obs.dropna(subset=['ion'])
            self.gal_obs = self.gal_obs.dropna(subset=['ion'])
            if self.use_iono_rms and 'ion_rms' in self.gps_obs.columns:
                self.gps_obs = self.gps_obs.dropna(subset=['ion_rms'])
            if self.use_iono_rms and 'ion_rms' in self.gal_obs.columns:
                self.gal_obs = self.gal_obs.dropna(subset=['ion_rms'])

    def _state_dim(self, mode: str) -> int:
        return 1 + len(_mode_signals(mode))

    def _obs_dim(self, mode: str) -> int:
        return 2 * len(_mode_signals(mode)) + int(self.use_iono_constr)

    def _state_specs(self, gps_sats, gal_sats):
        specs = {}
        offset = self.base_dim
        gps_dim = self._state_dim(self.gps_mode)
        for sv in gps_sats:
            specs[f"G:{sv}"] = (offset, gps_dim)
            offset += gps_dim
        gal_dim = self._state_dim(self.gal_mode)
        for sv in gal_sats:
            specs[f"E:{sv}"] = (offset, gal_dim)
            offset += gal_dim
        return specs, offset

    def _system_state_start(self, n_gps: int, system: str) -> int:
        if system == "G":
            return self.base_dim
        return self.base_dim + self._state_dim(self.gps_mode) * n_gps

    def _system_obs_start(self, n_gps: int, system: str) -> int:
        if system == "G":
            return 0
        return self._obs_dim(self.gps_mode) * n_gps

    def Hjacobian(self, x, gps_satellites, gal_satellites):
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        dim_x = (
            self.base_dim
            + self._state_dim(self.gps_mode) * n_gps
            + self._state_dim(self.gal_mode) * n_gal
        )
        dim_z = self._obs_dim(self.gps_mode) * n_gps + self._obs_dim(self.gal_mode) * n_gal
        H = np.zeros((dim_z, dim_x))
        if n_gps:
            self._fill_h_system(
                H=H,
                x=x,
                satellites=gps_satellites,
                mode=self.gps_mode,
                system="G",
                row_start=0,
                state_start=self._system_state_start(n_gps, "G"),
            )
        if n_gal:
            self._fill_h_system(
                H=H,
                x=x,
                satellites=gal_satellites,
                mode=self.gal_mode,
                system="E",
                row_start=self._system_obs_start(n_gps, "E"),
                state_start=self._system_state_start(n_gps, "E"),
            )
        return H

    def _fill_h_system(self, H, x, satellites, mode, system, row_start, state_start):
        signals = _mode_signals(mode)
        state_dim = self._state_dim(mode)
        obs_dim = self._obs_dim(mode)
        f_ref = self.FREQ_DICT[signals[0]]
        rec_xyz = x[:3].copy()
        e_vec, _ = _unit_vectors_and_ranges(satellites[:, :3], rec_xyz)
        m_wet = satellites[:, 3]
        col_clk = 3
        col_isb = 4
        col_ztd = 5 if self.tro else None

        for sat_idx in range(satellites.shape[0]):
            row = row_start + obs_dim * sat_idx
            state = state_start + state_dim * sat_idx
            for band_idx, signal in enumerate(signals):
                code_row = row + 2 * band_idx
                phase_row = code_row + 1
                freq = self.FREQ_DICT[signal]
                mu = (f_ref / freq) ** 2
                lam = self.CLIGHT / freq
                H[[code_row, phase_row], 0:3] = e_vec[sat_idx]
                H[[code_row, phase_row], col_clk] = 1.0
                if system == "E":
                    H[[code_row, phase_row], col_isb] = 1.0
                if self.tro:
                    H[[code_row, phase_row], col_ztd] = m_wet[sat_idx]
                H[code_row, state] = mu
                H[phase_row, state] = -mu
                H[phase_row, state + 1 + band_idx] = lam
            if self.use_iono_constr:
                H[row + 2 * len(signals), state] = 1.0

    def Hx(self, x, gps_satellites, gal_satellites):
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        dim_z = self._obs_dim(self.gps_mode) * n_gps + self._obs_dim(self.gal_mode) * n_gal
        z_hat = np.empty(dim_z)
        if n_gps:
            self._fill_hx_system(
                z_hat=z_hat,
                x=x,
                satellites=gps_satellites,
                mode=self.gps_mode,
                system="G",
                row_start=0,
                state_start=self._system_state_start(n_gps, "G"),
            )
        if n_gal:
            self._fill_hx_system(
                z_hat=z_hat,
                x=x,
                satellites=gal_satellites,
                mode=self.gal_mode,
                system="E",
                row_start=self._system_obs_start(n_gps, "E"),
                state_start=self._system_state_start(n_gps, "E"),
            )
        return z_hat

    def _fill_hx_system(self, z_hat, x, satellites, mode, system, row_start, state_start):
        signals = _mode_signals(mode)
        state_dim = self._state_dim(mode)
        obs_dim = self._obs_dim(mode)
        f_ref = self.FREQ_DICT[signals[0]]
        clk = x[3]
        isb = x[4] if system == "E" else 0.0
        zwd = x[5] if self.tro else 0.0
        _, rho = _unit_vectors_and_ranges(satellites[:, :3], x[:3])
        mw = satellites[:, 3] if self.tro else 0.0
        n = satellites.shape[0]
        state = x[state_start:state_start + state_dim * n].reshape(n, state_dim)
        ion = state[:, 0]
        block = np.empty((n, obs_dim))
        base = rho + clk + isb + mw * zwd
        for band_idx, signal in enumerate(signals):
            freq = self.FREQ_DICT[signal]
            mu = (f_ref / freq) ** 2
            lam = self.CLIGHT / freq
            amb = state[:, 1 + band_idx]
            block[:, 2 * band_idx] = base + mu * ion
            block[:, 2 * band_idx + 1] = base - mu * ion + lam * amb
        if self.use_iono_constr:
            block[:, -1] = ion
        z_hat[row_start:row_start + obs_dim * n] = block.reshape(-1)

    def _prepare_system_obs(self, obs: pd.DataFrame, mode: str, system: str):
        obs = obs.copy()
        layout = []
        for band in _mode_layout(mode, system):
            resolved = dict(band)
            resolved["code_col"] = _first_prefixed_column(obs, band["code"])
            resolved["phase_col"] = _first_prefixed_column(obs, band["phase"])
            if band["rec_pco_col"] not in obs.columns:
                obs.loc[:, band["rec_pco_col"]] = 0.0
            layout.append(resolved)
        if 'me_wet' not in obs.columns:
            obs.loc[:, 'me_wet'] = 0.0
        return obs, layout

    def _prepare_obs(self):
        self.gps_obs, gps_layout = self._prepare_system_obs(self.gps_obs, self.gps_mode, "G")
        self.gal_obs, gal_layout = self._prepare_system_obs(self.gal_obs, self.gal_mode, "E")
        return gps_layout, gal_layout

    def _col_np(self, epoch, name, default=0.0):
        if name in epoch.columns:
            return epoch[name].to_numpy(copy=False)
        if np.isscalar(default):
            return default
        return np.asarray(default)

    def _system_measurements(self, epoch, satellites, mode, layout, num):
        n = len(epoch)
        obs_dim = self._obs_dim(mode)
        block = np.empty((n, obs_dim))
        r_diag = np.empty((n, obs_dim))
        clk = self._col_np(epoch, "clk")
        tro = self._col_np(epoch, "tro")
        ah = self._col_np(epoch, "ah_los")
        dprel = self._col_np(epoch, "dprel")
        tides = self._col_np(epoch, "tides_los")
        phw = self._col_np(epoch, "phw")
        ev = np.deg2rad(self._col_np(epoch, "ev", np.full(n, 30.0)))
        inv_sin = 1.0 / np.sin(ev)
        code_dcb_corrs = [np.zeros(n, dtype=float) for _ in layout]
        if len(layout) == 2:
            code_1 = layout[0]["code_col"]
            code_2 = layout[1]["code_col"]
            f1 = self.FREQ_DICT[layout[0]["freq"]]
            f2 = self.FREQ_DICT[layout[1]["freq"]]
            a = f1 ** 2 / (f1 ** 2 - f2 ** 2)
            b = f2 ** 2 / (f1 ** 2 - f2 ** 2)
            corr_1, corr_2, _dcb_total = split_dual_code_dsb_corrections_m(
                epoch,
                code_1,
                code_2,
                coeff_1=a,
                coeff_2=b,
                n=n,
            )
            code_dcb_corrs[0] = corr_1
            code_dcb_corrs[1] = corr_2

        for band_idx, band in enumerate(layout):
            rec_pco = self._col_np(epoch, band["rec_pco_col"])
            sat_pco = self._col_np(epoch, f"sat_pco_los_{band['freq']}")
            osb_code = osb_m(epoch, band["code_col"], n)
            osb_phase = osb_m(epoch, band["phase_col"], n)
            freq = self.FREQ_DICT[band["freq"]]
            code = (
                epoch[band["code_col"]].to_numpy(copy=False)
                - rec_pco
                + sat_pco
                + clk * self.CLIGHT
                - tro
                - ah
                - dprel
                - osb_code
                + code_dcb_corrs[band_idx]
                - tides
            )
            phase = (
                epoch[band["phase_col"]].to_numpy(copy=False)
                - rec_pco
                + sat_pco
                + clk * self.CLIGHT
                - tro
                - ah
                - dprel
                - osb_phase
                - tides
                - (self.CLIGHT / freq) * phw
            )
            code_mask = self.code_screening(
                x=self.ekf.x[:3],
                satellites=satellites[:, :3],
                code_obs=code,
                thr=30,
            )
            block[:, 2 * band_idx] = code
            block[:, 2 * band_idx + 1] = phase
            r_diag[:, 2 * band_idx] = np.where(code_mask, inv_sin, 1e12)
            r_diag[:, 2 * band_idx + 1] = 0.001 * inv_sin

        if self.use_iono_constr:
            block[:, -1] = epoch['ion'].to_numpy(copy=False)
            r_diag[:, -1] = _iono_constraint_sigma(
                epoch,
                num,
                self.interval,
                self.use_iono_rms,
                self.sigma_iono_0,
                self.sigma_iono_end,
                self.t_end,
            )

        return block.reshape(-1), r_diag.reshape(-1)

    def _init_filter_common(self, epoch, clk0=0.0, zwd0=0.0):
        gps_sats = self.gps_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        gal_sats = self.gal_obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        _, dim_x = self._state_specs(gps_sats, gal_sats)
        dim_z = self._obs_dim(self.gps_mode) * len(gps_sats) + self._obs_dim(self.gal_mode) * len(gal_sats)

        x0 = np.zeros(dim_x)
        if self.pos0 is None:
            x0[:3] = np.array([6371.0e3, 0.0, 0.0])
        else:
            x0[:3] = self.pos0
        x0[3] = clk0
        x0[4] = 0.0
        if self.tro:
            x0[5] = zwd0

        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf._I = np.eye(dim_x)
        self.ekf.x = x0.copy()
        self.ekf.P = np.eye(dim_x) * 1e2
        self.ekf.P[:3, :3] = np.eye(3) * self.cfg.p_crd
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.P[4, 4] = self.cfg.p_isb
        if self.tro:
            self.ekf.P[5, 5] = self.cfg.p_tro
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        self.ekf.Q[4, 4] = self.cfg.q_isb
        if self.tro:
            self.ekf.Q[5, 5] = self.cfg.q_tro
        self.ekf.F = np.eye(dim_x)
        self.ekf.F[3, 3] = 0.0

        gps_ion = self._initial_iono(self.gps_obs.loc[(slice(None), epoch), :])
        gal_ion = self._initial_iono(self.gal_obs.loc[(slice(None), epoch), :])
        self._init_system_state("G", gps_sats, gps_ion, len(gps_sats))
        self._init_system_state("E", gal_sats, gal_ion, len(gps_sats))
        return gps_sats, gal_sats

    def _initial_iono(self, epoch):
        if 'ion' in epoch.columns:
            ion = epoch['ion'].to_numpy(dtype=float, copy=False)
            if np.any(np.isfinite(ion)):
                return np.nan_to_num(ion, nan=0.0, posinf=0.0, neginf=0.0)
        return np.zeros(len(epoch), dtype=float)

    def _init_system_state(self, system, sats, ion_values, n_gps):
        mode = self.gps_mode if system == "G" else self.gal_mode
        state_start = self._system_state_start(n_gps, system)
        state_dim = self._state_dim(mode)
        for i, _sv in enumerate(sats):
            idx_i = state_start + state_dim * i
            self.ekf.x[idx_i] = ion_values[i]
            self.ekf.P[idx_i, idx_i] = self.cfg.p_iono
            self.ekf.Q[idx_i, idx_i] = self.cfg.q_iono * (self.interval * 60) / 3600
            for amb in range(1, state_dim):
                self.ekf.P[idx_i + amb, idx_i + amb] = self.cfg.p_amb
                self.ekf.Q[idx_i + amb, idx_i + amb] = self.cfg.q_amb

    def init_filter(self, clk0=0.0, zwd0=0.0):
        epochs = sorted(set(self.gps_obs.index.get_level_values('time').unique()) &
                        set(self.gal_obs.index.get_level_values('time').unique()))
        first_ep = epochs[0]
        gps_sats, gal_sats = self._init_filter_common(first_ep, clk0=clk0, zwd0=zwd0)
        return gps_sats, gal_sats, epochs

    def reset_filter(self, epoch, clk0=0.0, zwd0=0.0):
        return self._init_filter_common(epoch, clk0=clk0, zwd0=zwd0)

    def rebuild_state(self, x_old, P_old, Q_old, prev_gps, prev_gal, curr_gps, curr_gal, gps_ion, gal_ion):
        old_specs, _ = self._state_specs(prev_gps, prev_gal)
        new_specs, new_dim = self._state_specs(curr_gps, curr_gal)
        x_new = np.zeros(new_dim)
        P_new = np.zeros((new_dim, new_dim))
        Q_new = np.zeros((new_dim, new_dim))
        x_new[:self.base_dim] = x_old[:self.base_dim]
        P_new[:self.base_dim, :self.base_dim] = P_old[:self.base_dim, :self.base_dim]
        Q_new[:self.base_dim, :self.base_dim] = Q_old[:self.base_dim, :self.base_dim]

        for tag, (new_start, new_dim_state) in new_specs.items():
            if tag in old_specs:
                old_start, old_dim_state = old_specs[tag]
                common_dim = min(old_dim_state, new_dim_state)
                old_idx = np.arange(old_start, old_start + common_dim)
                new_idx = np.arange(new_start, new_start + common_dim)
                x_new[new_idx] = x_old[old_idx]
                P_new[np.ix_(new_idx, new_idx)] = P_old[np.ix_(old_idx, old_idx)]
                Q_new[np.ix_(new_idx, new_idx)] = Q_old[np.ix_(old_idx, old_idx)]
                base_idx = np.arange(self.base_dim)
                P_new[np.ix_(new_idx, base_idx)] = P_old[np.ix_(old_idx, base_idx)]
                P_new[np.ix_(base_idx, new_idx)] = P_old[np.ix_(base_idx, old_idx)]
                Q_new[np.ix_(new_idx, base_idx)] = Q_old[np.ix_(old_idx, base_idx)]
                Q_new[np.ix_(base_idx, new_idx)] = Q_old[np.ix_(base_idx, old_idx)]
            else:
                sys_tag, sv = tag.split(":", 1)
                sat_list = curr_gps if sys_tag == "G" else curr_gal
                ion_values = gps_ion if sys_tag == "G" else gal_ion
                sat_idx = sat_list.index(sv)
                x_new[new_start] = ion_values[sat_idx]
                P_new[new_start, new_start] = self.cfg.p_iono
                Q_new[new_start, new_start] = self.cfg.q_iono * (self.interval * 60) / 3600
                for amb in range(1, new_dim_state):
                    P_new[new_start + amb, new_start + amb] = self.cfg.p_amb
                    Q_new[new_start + amb, new_start + amb] = self.cfg.q_amb

        return x_new, P_new, Q_new

    def code_screening(self, x, satellites, code_obs, thr=1):
        sat_xyz = np.asarray(satellites, dtype=float)
        ref_xyz = np.asarray(x, dtype=float)
        dist = np.linalg.norm(sat_xyz - ref_xyz, axis=1)
        prefit = code_obs - dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad > n_sat / 2:
            mask = np.ones(n_sat, dtype=bool)
        return mask

    def _system_state_values(self, system, n_gps, n_sats):
        mode = self.gps_mode if system == "G" else self.gal_mode
        start = self._system_state_start(n_gps, system)
        state_dim = self._state_dim(mode)
        if n_sats == 0:
            return np.empty((0, state_dim))
        return self.ekf.x[start:start + state_dim * n_sats].reshape(n_sats, state_dim)

    def _ar_ambiguity_groups(self, n_gps, n_gal, gps_ev, gal_ev, gps_age=None, gal_age=None):
        groups = []
        for system, n_sats, mode, ev, age in (
            ("G", n_gps, self.gps_mode, gps_ev, gps_age),
            ("E", n_gal, self.gal_mode, gal_ev, gal_age),
        ):
            if n_sats == 0:
                continue
            state_dim = self._state_dim(mode)
            state_start = self._system_state_start(n_gps, system)
            sat_idx = np.arange(n_sats)
            n1_idx = state_start + state_dim * sat_idx + 1
            n2_idx = state_start + state_dim * sat_idx + 2 if state_dim > 2 else None
            group = {
                "n1_idx": n1_idx,
                "n2_idx": n2_idx,
                "ev": ev,
            }
            if age is not None:
                group["age"] = age
            groups.append(group)
        return groups

    def run_filter(self, clk0=0.0, ref=None, flh=None, zwd0=0.0, trace_filter=False, reset_every=0):
        old_gps_sats, old_gal_sats, all_times = self.init_filter(clk0=clk0, zwd0=zwd0)
        gps_layout, gal_layout = self._prepare_obs()
        self.gps_obs = self.gps_obs.sort_values(by='sv')
        self.gal_obs = self.gal_obs.sort_values(by='sv')
        gps_epochs = {t: df for t, df in self.gps_obs.groupby(level=1, sort=False)}
        gal_epochs = {t: df for t, df in self.gal_obs.groupby(level=1, sort=False)}
        result_rows = []
        result_times = []
        gps_results = []
        gal_results = []
        r_cache = {}
        xyz = None
        conv_time = None
        T0 = all_times[0]
        gps_arc_age = {sv: 0 for sv in old_gps_sats}
        gal_arc_age = {sv: 0 for sv in old_gal_sats}
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

        for num, t in enumerate(all_times):
            reset_epoch = False
            if reset_every != 0 and ((num * self.interval) % reset_every == 0) and (num != 0):
                old_gps_sats, old_gal_sats = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                gps_arc_age = {sv: 0 for sv in old_gps_sats}
                gal_arc_age = {sv: 0 for sv in old_gal_sats}
                reset_epoch = True
                T0 = t

            gps_epoch = gps_epochs.get(t)
            gal_epoch = gal_epochs.get(t)
            if gps_epoch is None or gal_epoch is None:
                continue

            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_gal_sats = gal_epoch.index.get_level_values('sv').tolist()
            gps_ion = self._initial_iono(gps_epoch)
            gal_ion = self._initial_iono(gal_epoch)
            if (curr_gps_sats != old_gps_sats) or (curr_gal_sats != old_gal_sats):
                self.ekf.x, self.ekf.P, self.ekf.Q = self.rebuild_state(
                    x_old=self.ekf.x.copy(),
                    P_old=self.ekf.P.copy(),
                    Q_old=self.ekf.Q.copy(),
                    prev_gps=old_gps_sats,
                    prev_gal=old_gal_sats,
                    curr_gps=curr_gps_sats,
                    curr_gal=curr_gal_sats,
                    gps_ion=gps_ion,
                    gal_ion=gal_ion,
                )
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = (
                    self._obs_dim(self.gps_mode) * len(curr_gps_sats)
                    + self._obs_dim(self.gal_mode) * len(curr_gal_sats)
                )
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)
            old_gps_sats = curr_gps_sats
            old_gal_sats = curr_gal_sats

            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            gal_satellites = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            z_gps, r_gps = self._system_measurements(gps_epoch, gps_satellites, self.gps_mode, gps_layout, num)
            z_gal, r_gal = self._system_measurements(gal_epoch, gal_satellites, self.gal_mode, gal_layout, num)
            Z = np.concatenate((z_gps, z_gal))
            if not np.all(np.isfinite(Z)):
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "non-finite-observation",
                        epoch=num,
                        time=t,
                    )
                continue

            r_diag = np.concatenate((r_gps, r_gal))
            r_diag = np.nan_to_num(r_diag, nan=1e12, posinf=1e12, neginf=1e12)
            m = r_diag.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = r_diag
            self.ekf.R = R

            updated = False
            for k in range(5):
                try:
                    self.ekf.predict_update(
                        z=Z,
                        HJacobian=self.Hjacobian,
                        args=(gps_satellites, gal_satellites),
                        Hx=self.Hx,
                        hx_args=(gps_satellites, gal_satellites),
                    )
                    updated = True
                    break
                except np.linalg.LinAlgError:
                    self.ekf.R.flat[::m + 1] += 1e-6 * (10.0 ** k)
            if not updated:
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "singular-update-reset",
                        epoch=num,
                        time=t,
                    )
                old_gps_sats, old_gal_sats = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                gps_arc_age = {sv: 0 for sv in old_gps_sats}
                gal_arc_age = {sv: 0 for sv in old_gal_sats}
                continue
            gps_arc_age = advance_arc_age(gps_arc_age, curr_gps_sats)
            gal_arc_age = advance_arc_age(gal_arc_age, curr_gal_sats)
            gps_ar_age = arc_age_array(gps_arc_age, curr_gps_sats)
            gal_ar_age = arc_age_array(gal_arc_age, curr_gal_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_indexed_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    ambiguity_groups=self._ar_ambiguity_groups(
                        n_gps=len(curr_gps_sats),
                        n_gal=len(curr_gal_sats),
                        gps_ev=gps_epoch['ev'].to_numpy(copy=False),
                        gal_ev=gal_epoch['ev'].to_numpy(copy=False),
                        gps_age=gps_ar_age,
                        gal_age=gal_ar_age,
                    ),
                    settings=ar_cfg,
                )

            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(self.ekf.x[:3] - xyz)
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0
                    elif position_diff > 0.005:
                        conv_time = None

            xyz = self.ekf.x[:3].copy()
            dtr = self.ekf.x[3]
            isb = self.ekf.x[4]
            ztd = self.ekf.x[5] if self.tro else 0.0
            if ref is not None and flh is not None:
                dx = ref - self.ekf.x[:3]
                enu = np.array(ecef_to_enu(dXYZ=dx, flh=flh, degrees=True)).flatten()
            else:
                enu = np.zeros(3)
            result_rows.append({
                'de': enu[0],
                'dn': enu[1],
                'du': enu[2],
                'dtr': dtr,
                'ztd': ztd,
                'x': xyz[0],
                'y': xyz[1],
                'z': xyz[2],
                'isb': isb,
                'ar_fixed': 0 if ar_diag is None else ar_diag.fixed_ambiguities,
                'ar_ratio': np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min,
                'ar_ok': False if ar_diag is None else ar_diag.accepted,
                'ar_gps_min_age': int(np.min(gps_ar_age)) if gps_ar_age.size else np.nan,
                'ar_gal_min_age': int(np.min(gal_ar_age)) if gal_ar_age.size else np.nan,
            })
            result_rows[-1].update(pppar_diagnostic_columns(ar_diag))
            result_times.append(t)

            gps_epoch = gps_epoch.copy()
            gal_epoch = gal_epoch.copy()
            gps_state = self._system_state_values("G", len(curr_gps_sats), len(curr_gps_sats))
            gal_state = self._system_state_values("E", len(curr_gps_sats), len(curr_gal_sats))
            if len(gps_state):
                gps_epoch['stec'] = gps_state[:, 0]
                gps_epoch['n1'] = gps_state[:, 1]
                if gps_state.shape[1] > 2:
                    gps_epoch['n2'] = gps_state[:, 2]
            if len(gal_state):
                gal_epoch['stec'] = gal_state[:, 0]
                gal_epoch['n1'] = gal_state[:, 1]
                if gal_state.shape[1] > 2:
                    gal_epoch['n2'] = gal_state[:, 2]
            gps_results.append(gps_epoch)
            gal_results.append(gal_epoch)
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result_rows[-1],
                )

        ct = conv_time.total_seconds() / 3600 if conv_time is not None else None
        result = pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time'))
        result['ct_min'] = ct
        gps_result = pd.concat(gps_results) if gps_results else self.gps_obs.iloc[0:0].copy()
        gal_result = pd.concat(gal_results) if gal_results else self.gal_obs.iloc[0:0].copy()
        return result, gps_result, gal_result, ct


class PPPUdMultiGNSS:
    """Legacy GPS/Galileo dual-frequency uncombined PPP filter.

    Purpose:
        Historical G/E uncombined dual-frequency branch without external
        ionospheric constraints. It remains reachable when GPS and Galileo
        dual-frequency observations are active and ``use_iono_constr`` is
        false.

    Status:
        Legacy/reachable GPS+Galileo path. It remains routed for G/E dual-
        frequency no-constraint compatibility, but new mixed no-constraint work
        should prefer ``PPPUdGenericMixedGNSS``.

    Model:
        Estimates receiver position, GPS receiver clock, Galileo ISB, optional
        ZTD, and per-satellite ``[I, N1, N2]`` blocks from uncombined code and
        phase observables.

    State vector:
        ``[x, y, z, dtr_G, isb_E, ztd?, (I, N1, N2)_G..., (I, N1, N2)_E...]``.

    Supported systems:
        GPS and Galileo only.

    PPP-AR support:
        Uncombined AR hooks are present in related branches; this class should
        be validated carefully before enabling or changing AR behavior.

    Limitations:
        Legacy G/E-specific path. New mixed no-constraint work should prefer
        ``PPPUdGenericMixedGNSS``.
    """

    def __init__(self, config:PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, interval=0.5):
        self.cfg=config
        self.gps_obs = gps_obs
        self.gps_mode = gps_mode
        self.gal_obs = gal_obs
        self.gal_mode = gal_mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.LAMBDA_DICT = {}
        self.CLIGHT = 299792458
        self.base_dim = 6 if self.tro else 5
        self.pos0 = pos0
        self.interval=interval
        self.use_iono_constr = bool(getattr(self.cfg, "use_iono_constr", False))
        self.use_iono_rms = bool(getattr(self.cfg, "use_iono_rms", True))
        self.sigma_iono_0 = getattr(self.cfg, "sigma_iono_0", 1.1)
        self.sigma_iono_end = getattr(self.cfg, "sigma_iono_end", 3.0)
        self.t_end = getattr(self.cfg, "t_end", 30)
        if self.use_iono_constr and self.gps_obs is not None and self.gal_obs is not None:
            if 'ion' not in self.gps_obs.columns or 'ion' not in self.gal_obs.columns:
                raise ValueError("use_iono_constr requires an 'ion' column in GPS and Galileo observations.")
            self.gps_obs = self.gps_obs.dropna(subset=['ion'])
            self.gal_obs = self.gal_obs.dropna(subset=['ion'])
            if self.use_iono_rms and 'ion_rms' in self.gps_obs.columns:
                self.gps_obs = self.gps_obs.dropna(subset=['ion_rms'])
            if self.use_iono_rms and 'ion_rms' in self.gal_obs.columns:
                self.gal_obs = self.gal_obs.dropna(subset=['ion_rms'])
        elif self.gps_obs is None or self.gal_obs is None:
            self.use_iono_constr = False
        self.obs_per_sat = 5 if self.use_iono_constr else 4

    def Hjacobian(self, x, gps_satellites, gal_satellites):
        C = self.CLIGHT
        # Częstotliwości GPS
        gps_f1, gps_f2 = _dual_mode_signals(self.gps_mode)
        F1_GPS = self.FREQ_DICT[gps_f1]
        F2_GPS = self.FREQ_DICT[gps_f2]
        L1 = C / F1_GPS
        L2 = C / F2_GPS
        MU1 = 1.0
        MU2 = (F1_GPS / F2_GPS) ** 2

        # Częstotliwości Galileo
        gal_f1, gal_f2 = _dual_mode_signals(self.gal_mode)
        F1_GAL = self.FREQ_DICT[gal_f1]
        F2_GAL = self.FREQ_DICT[gal_f2]
        E1 = C / F1_GAL
        E5a = C / F2_GAL
        MU1_GAL = 1.0
        MU2_GAL = (F1_GAL / F2_GAL) ** 2

        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        rec_xyz = x[:3].copy()

        e_gps, _ = _unit_vectors_and_ranges(gps_satellites[:, :3], rec_xyz)
        e_gal, _ = _unit_vectors_and_ranges(gal_satellites[:, :3], rec_xyz)
        m_wet_gps = gps_satellites[:, 3]
        m_wet_gal = gal_satellites[:, 3]

        # Parametry wspólne: XYZ, clk_gps, ISB_galileo, ZTD
        COL_X, COL_Y, COL_Z = 0, 1, 2
        COL_CLK = 3
        COL_ISB = 4
        COL_ZTD = 5

        base_dim = 6  # wspólne parametry (XYZ + clk_gps + isb + ztd)
        rows_per_sat = self.obs_per_sat

        # 4 obserwacje GNSS plus opcjonalny constraint jonosferyczny.
        H = np.zeros((rows_per_sat * (n_gps + n_gal), base_dim + 3 * (n_gps + n_gal)))

        # ---------------- GPS ----------------
        if n_gps:
            idx = np.arange(n_gps)
            gps_rows = (rows_per_sat * idx[:, None] + np.arange(4)).ravel()
            H[np.ix_(gps_rows, [COL_X, COL_Y, COL_Z])] = np.repeat(e_gps, 4, axis=0)
            H[gps_rows, COL_CLK] = 1.0
            H[gps_rows, COL_ZTD] = np.repeat(m_wet_gps, 4)

            row0 = rows_per_sat * idx
            row1 = row0 + 1
            row2 = row0 + 2
            row3 = row0 + 3
            col_iono = base_dim + 3 * idx
            H[row0, col_iono] = MU1
            H[row1, col_iono] = -MU1
            H[row2, col_iono] = MU2
            H[row3, col_iono] = -MU2
            H[row1, col_iono + 1] = L1
            H[row3, col_iono + 2] = L2
            if self.use_iono_constr:
                H[row0 + 4, col_iono] = 1.0

        # ---------------- Galileo ----------------
        if n_gal:
            idx = np.arange(n_gal)
            gal_rows = (rows_per_sat * (n_gps + idx)[:, None] + np.arange(4)).ravel()
            H[np.ix_(gal_rows, [COL_X, COL_Y, COL_Z])] = np.repeat(e_gal, 4, axis=0)
            H[gal_rows, COL_CLK] = 1.0
            H[gal_rows, COL_ISB] = 1.0
            H[gal_rows, COL_ZTD] = np.repeat(m_wet_gal, 4)

            row0 = rows_per_sat * (n_gps + idx)
            row1 = row0 + 1
            row2 = row0 + 2
            row3 = row0 + 3
            col_iono = base_dim + 3 * (n_gps + idx)
            H[row0, col_iono] = MU1_GAL
            H[row1, col_iono] = -MU1_GAL
            H[row2, col_iono] = MU2_GAL
            H[row3, col_iono] = -MU2_GAL
            H[row1, col_iono + 1] = E1
            H[row3, col_iono + 2] = E5a
            if self.use_iono_constr:
                H[row0 + 4, col_iono] = 1.0

        return H

    def Hx(self, x: np.ndarray, gps_satellites: np.ndarray, gal_satellites: np.ndarray) -> np.ndarray:
        x_state = x.copy()
        C = self.CLIGHT

        # GPS
        gps_f1, gps_f2 = _dual_mode_signals(self.gps_mode)
        F1_GPS = self.FREQ_DICT[gps_f1]
        F2_GPS = self.FREQ_DICT[gps_f2]
        L1 = C / F1_GPS
        L2 = C / F2_GPS
        MU1 = 1.0
        MU2 = (F1_GPS / F2_GPS) ** 2

        # Galileo
        gal_f1, gal_f2 = _dual_mode_signals(self.gal_mode)
        F1_GAL = self.FREQ_DICT[gal_f1]
        F2_GAL = self.FREQ_DICT[gal_f2]
        E1 = C / F1_GAL
        E5a = C / F2_GAL
        MU1_GAL = 1.0
        MU2_GAL = (F1_GAL / F2_GAL) ** 2

        clk = x_state[3]
        isb = x_state[4]  # inter-sys bias (dla Galileo)
        zwd = x_state[5]

        base_dim = self.base_dim
        n_gps = gps_satellites.shape[0]
        n_gal = gal_satellites.shape[0]
        rows_per_sat = self.obs_per_sat

        z_hat = np.empty(rows_per_sat * (n_gps + n_gal))

        if n_gps:
            gps_state = x_state[base_dim:base_dim + 3 * n_gps].reshape(n_gps, 3)
            I_gps = gps_state[:, 0]
            N1_gps = gps_state[:, 1]
            N2_gps = gps_state[:, 2]
            _, rho_gps = _unit_vectors_and_ranges(gps_satellites[:, :3], x_state[:3])
            mw_gps = gps_satellites[:, 3]
            geom = rho_gps
            gps_block = np.column_stack((
                geom + clk + mw_gps * zwd + MU1 * I_gps,
                geom + clk + mw_gps * zwd - MU1 * I_gps + L1 * N1_gps,
                geom + clk + mw_gps * zwd + MU2 * I_gps,
                geom + clk + mw_gps * zwd - MU2 * I_gps + L2 * N2_gps,
            ))
            if self.use_iono_constr:
                gps_block = np.column_stack((gps_block, I_gps))
            z_hat[:rows_per_sat * n_gps] = gps_block.reshape(-1)

        if n_gal:
            gal_state = x_state[base_dim + 3 * n_gps:base_dim + 3 * (n_gps + n_gal)].reshape(n_gal, 3)
            I_gal = gal_state[:, 0]
            N1_gal = gal_state[:, 1]
            N2_gal = gal_state[:, 2]
            _, rho_gal = _unit_vectors_and_ranges(gal_satellites[:, :3], x_state[:3])
            mw_gal = gal_satellites[:, 3]
            geom = rho_gal
            gal_block = np.column_stack((
                geom + clk + isb + mw_gal * zwd + MU1_GAL * I_gal,
                geom + clk + isb + mw_gal * zwd - MU1_GAL * I_gal + E1 * N1_gal,
                geom + clk + isb + mw_gal * zwd + MU2_GAL * I_gal,
                geom + clk + isb + mw_gal * zwd - MU2_GAL * I_gal + E5a * N2_gal,
            ))
            if self.use_iono_constr:
                gal_block = np.column_stack((gal_block, I_gal))
            start = rows_per_sat * n_gps
            z_hat[start:start + rows_per_sat * n_gal] = gal_block.reshape(-1)

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

        prev_map = {prn: i for i, prn in enumerate(prev_all)}
        curr_map = {prn: i for i, prn in enumerate(curr_all)}
        common = [prn for prn in curr_all if prn in prev_map]

        if common:
            prev_idx = np.fromiter((prev_map[prn] for prn in common), dtype=int)
            curr_idx = np.fromiter((curr_map[prn] for prn in common), dtype=int)
            offsets = np.arange(3)
            old_state_idx = (base_dim + 3 * prev_idx[:, None] + offsets[None, :]).ravel()
            new_state_idx = (base_dim + 3 * curr_idx[:, None] + offsets[None, :]).ravel()

            x_new[new_state_idx] = x_old[old_state_idx]
            base_idx = np.arange(base_dim)
            P_new[np.ix_(new_state_idx, base_idx)] = P_old[np.ix_(old_state_idx, base_idx)]
            P_new[np.ix_(base_idx, new_state_idx)] = P_old[np.ix_(base_idx, old_state_idx)]
            Q_new[np.ix_(new_state_idx, base_idx)] = Q_old[np.ix_(old_state_idx, base_idx)]
            Q_new[np.ix_(base_idx, new_state_idx)] = Q_old[np.ix_(base_idx, old_state_idx)]
            P_new[np.ix_(new_state_idx, new_state_idx)] = P_old[np.ix_(old_state_idx, old_state_idx)]
            Q_new[np.ix_(new_state_idx, new_state_idx)] = Q_old[np.ix_(old_state_idx, old_state_idx)]

        # --- nowe satelity: zainicjalizuj I_s z P4 ---
        new_only = set(curr_all) - set(prev_all)
        for prn in new_only:
            j = curr_map[prn]
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
        dim_z = self.obs_per_sat * (n_gps + n_gal)

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
        dim_z = self.obs_per_sat * (n_gps + n_gal)

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
        gps_data = _dual_mode_columns(self.gps_obs, self.gps_mode)
        gal_data = _dual_mode_columns(self.gal_obs, self.gal_mode)
        self.gps_obs = self.gps_obs.copy()
        self.gal_obs = self.gal_obs.copy()

        for col in _dual_rec_pco_columns(self.gps_mode):
            if col not in self.gps_obs.columns.tolist():
                self.gps_obs.loc[:, col] = 0.0
        for col in _dual_rec_pco_columns(self.gal_mode):
            if col not in self.gal_obs.columns.tolist():
                self.gal_obs.loc[:, col] = 0.0

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
        offsets = np.array([1, 2])
        idxs_N = (base_dim + 3 * np.arange(n_sats)[:, None] + offsets[None, :]).ravel()
        N_vec = x[idxs_N]
        P_N = P[np.ix_(idxs_N, idxs_N)]

        return N_vec, P_N

    def code_screening(self, x, satellites, code_obs, thr=1):
        sat_xyz = np.asarray(satellites, dtype=float)
        ref_xyz = np.asarray(x, dtype=float)
        dist = np.linalg.norm(sat_xyz - ref_xyz, axis=1)
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
        agal, bgal, gal_c1, gal_c2, gal_l1, gal_l2 = gal_data
        result_rows = []
        result_times = []
        e, g = [], []
        self.gps_obs = self.gps_obs.sort_values(by='sv')
        self.gal_obs = self.gal_obs.sort_values(by='sv')
        gps_epochs = {
            t: df for t, df in self.gps_obs.groupby(level=1, sort=False)
        }
        gal_epochs = {
            t: df for t, df in self.gal_obs.groupby(level=1, sort=False)
        }
        gps_osb_c1_col = f'OSB_{gps_c1}'
        gps_osb_c2_col = f'OSB_{gps_c2}'
        gps_osb_l1_col = f'OSB_{gps_l1}'
        gps_osb_l2_col = f'OSB_{gps_l2}'
        gal_osb_c1_col = f'OSB_{gal_c1}'
        gal_osb_c2_col = f'OSB_{gal_c2}'
        gal_osb_l1_col = f'OSB_{gal_l1}'
        gal_osb_l2_col = f'OSB_{gal_l2}'
        gps_has_osb_c1 = gps_osb_c1_col in self.gps_obs.columns
        gps_has_osb_c2 = gps_osb_c2_col in self.gps_obs.columns
        gps_has_osb_l1 = gps_osb_l1_col in self.gps_obs.columns
        gps_has_osb_l2 = gps_osb_l2_col in self.gps_obs.columns
        gal_has_osb_c1 = gal_osb_c1_col in self.gal_obs.columns
        gal_has_osb_c2 = gal_osb_c2_col in self.gal_obs.columns
        gal_has_osb_l1 = gal_osb_l1_col in self.gal_obs.columns
        gal_has_osb_l2 = gal_osb_l2_col in self.gal_obs.columns
        gps_sig1, gps_sig2 = _dual_mode_signals(self.gps_mode)
        gal_sig1, gal_sig2 = _dual_mode_signals(self.gal_mode)
        gps_pco1_col, gps_pco2_col = _dual_rec_pco_columns(self.gps_mode)
        gal_pco1_col, gal_pco2_col = _dual_rec_pco_columns(self.gal_mode)
        r_cache = {}
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
        xyz = None
        conv_time = None
        T0 = all_times[0]
        gps_arc_age = {sv: 0 for sv in old_gps_sats}
        gal_arc_age = {sv: 0 for sv in old_gal_sats}
        for num, t in enumerate(all_times):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_gps_sats, old_gal_sats = self.reset_filter(epoch=t)
                    gps_arc_age = {sv: 0 for sv in old_gps_sats}
                    gal_arc_age = {sv: 0 for sv in old_gal_sats}
                    reset_epoch = True
                    T0 = t
            gps_epoch = gps_epochs.get(t)
            gal_epoch = gal_epochs.get(t)
            if gps_epoch is None or gal_epoch is None:
                continue

            def safe_get(df, col, length=None):
                """Zwraca kolumnę jako numpy, a jeśli brak – wektor zer o podanej długości."""
                if col in df.columns:
                    return df[col].to_numpy(copy=False)
                elif length is not None:
                    return np.zeros(length)
                else:
                    return np.zeros(len(df))

            # --- GPS ---
            gps_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', gps_pco1_col, gps_pco2_col,
                f'sat_pco_los_{gps_sig1}',
                f'sat_pco_los_{gps_sig2}',
                'phw', 'me_wet'
            ]
            gps_len = len(gps_epoch)
            (gps_clk, gps_tro, gps_ah_los, gps_dprel, gps_pco1, gps_pco2,
             sat_pco_L1, sat_pco_L2, phw, mwet) = [safe_get(gps_epoch, c, gps_len) for c in gps_cols]

            # --- Galileo ---
            gal_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', gal_pco1_col, gal_pco2_col,
                f'sat_pco_los_{gal_sig1}',
                f'sat_pco_los_{gal_sig2}',
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
                self.ekf.dim_z = self.obs_per_sat * (len(curr_gps_sats) + len(curr_gal_sats))
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)

            old_gps_sats = curr_gps_sats
            old_gal_sats = curr_gal_sats
            gps_arc_age = {sv: gps_arc_age.get(sv, 0) for sv in curr_gps_sats}
            gal_arc_age = {sv: gal_arc_age.get(sv, 0) for sv in curr_gal_sats}

            tides = gps_epoch['tides_los'].to_numpy(copy=False)
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gps_p1_c1 = osb_m(gps_epoch, gps_c1, gps_len)
            gps_p2_c2 = osb_m(gps_epoch, gps_c2, gps_len)
            gps_pl1_l1 = osb_m(gps_epoch, gps_l1, gps_len)
            gps_pl2_l2 = osb_m(gps_epoch, gps_l2, gps_len)
            gps_code_corr_c1, gps_code_corr_c2, _gps_dcb_total = split_dual_code_dsb_corrections_m(
                gps_epoch,
                gps_c1,
                gps_c2,
                coeff_1=agps,
                coeff_2=bgps,
                n=gps_len,
            )
            C1 = gps_epoch[gps_c1].to_numpy(copy=False) - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p1_c1 + gps_code_corr_c1 - tides
            C2 = gps_epoch[gps_c2].to_numpy(copy=False) - gps_pco2 + sat_pco_L2 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_p2_c2 + gps_code_corr_c2 - tides
            L1 = gps_epoch[gps_l1].to_numpy(copy=False) - gps_pco1 + sat_pco_L1 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl1_l1 - tides - (
                self.CLIGHT / self.FREQ_DICT[gps_sig1]) * phw
            L2 = gps_epoch[gps_l2].to_numpy(copy=False) - gps_pco2 + sat_pco_L2 + gps_clk * self.CLIGHT - gps_tro - gps_ah_los - gps_dprel - gps_pl2_l2 - tides - (
                self.CLIGHT / self.FREQ_DICT[gps_sig2]) * phw

            gal_tides = gal_epoch['tides_los'].to_numpy(copy=False)
            gal_satellites = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gal_p1_c1 = osb_m(gal_epoch, gal_c1, gal_len)
            gal_p2_c2 = osb_m(gal_epoch, gal_c2, gal_len)
            gal_pl1_l1 = osb_m(gal_epoch, gal_l1, gal_len)
            gal_pl2_l2 = osb_m(gal_epoch, gal_l2, gal_len)
            gal_code_corr_c1, gal_code_corr_c2, _gal_dcb_total = split_dual_code_dsb_corrections_m(
                gal_epoch,
                gal_c1,
                gal_c2,
                coeff_1=agal,
                coeff_2=bgal,
                n=gal_len,
            )
            EC1 = gal_epoch[gal_c1].to_numpy(copy=False) - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_p1_c1 + gal_code_corr_c1 - gal_tides
            EC2 = gal_epoch[gal_c2].to_numpy(copy=False) - gal_pco2 + sat_pco_E5a + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_p2_c2 + gal_code_corr_c2 - gal_tides
            EL1 = gal_epoch[gal_l1].to_numpy(copy=False) - gal_pco1 + sat_pco_E1 + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_pl1_l1 - gal_tides - (
                self.CLIGHT / self.FREQ_DICT[gal_sig1]) * gal_phw
            EL2 = gal_epoch[gal_l2].to_numpy(copy=False) - gal_pco2 + sat_pco_E5a + gal_clk * self.CLIGHT - gal_tro - gal_ah_los - gal_dprel - gal_pl2_l2 - gal_tides - (
                self.CLIGHT / self.FREQ_DICT[gal_sig2]) * gal_phw

            if self.use_iono_constr:
                IONO = gps_epoch['ion'].to_numpy(copy=False)
                EIONO = gal_epoch['ion'].to_numpy(copy=False)
                ZGPS = np.empty(5 * len(C1))
                ZGPS[0::5] = C1
                ZGPS[1::5] = L1
                ZGPS[2::5] = C2
                ZGPS[3::5] = L2
                ZGPS[4::5] = IONO
                ZGAL = np.empty(5 * len(EC1))
                ZGAL[0::5] = EC1
                ZGAL[1::5] = EL1
                ZGAL[2::5] = EC2
                ZGAL[3::5] = EL2
                ZGAL[4::5] = EIONO
            else:
                ZGPS = np.empty(4 * len(C1))
                ZGPS[0::4] = C1
                ZGPS[1::4] = L1
                ZGPS[2::4] = C2
                ZGPS[3::4] = L2
                ZGAL = np.empty(4 * len(EC1))
                ZGAL[0::4] = EC1
                ZGAL[1::4] = EL1
                ZGAL[2::4] = EC2
                ZGAL[3::4] = EL2

            Z = np.concatenate((ZGPS, ZGAL))
            gps_c1_mask = self.code_screening(x=self.ekf.x[:3],code_obs=C1,thr=30,satellites=gps_satellites[:,:3])
            gps_c2_mask = self.code_screening(x=self.ekf.x[:3], code_obs=C2, thr=30, satellites=gps_satellites[:,:3])
            gal_c1_mask = self.code_screening(x=self.ekf.x[:3], code_obs=EC1, thr=30, satellites=gal_satellites[:,:3])
            gal_c2_mask = self.code_screening(x=self.ekf.x[:3], code_obs=EC2, thr=30, satellites=gal_satellites[:,:3])

            # WEIGHTS GPS
            ev_gps = np.deg2rad(gps_epoch['ev'].to_numpy(copy=False))
            inv_sin_gps = 1.0 / np.sin(ev_gps)

            # WEIGHTS GAL
            ev_gal = np.deg2rad(gal_epoch['ev'].to_numpy(copy=False))
            inv_sin_gal = 1.0 / np.sin(ev_gal)
            if self.use_iono_constr:
                R = np.empty(5 * len(C1))
                R[0::5] = np.where(gps_c1_mask, inv_sin_gps, 1e12)
                R[1::5] = 0.001 * inv_sin_gps
                R[2::5] = np.where(gps_c2_mask, inv_sin_gps, 1e12)
                R[3::5] = 0.001 * inv_sin_gps
                R[4::5] = _iono_constraint_sigma(
                    gps_epoch,
                    num,
                    self.interval,
                    self.use_iono_rms,
                    self.sigma_iono_0,
                    self.sigma_iono_end,
                    self.t_end,
                )

                RG = np.empty(5 * len(EC1))
                RG[0::5] = np.where(gal_c1_mask, inv_sin_gal, 1e12)
                RG[1::5] = 0.001 * inv_sin_gal
                RG[2::5] = np.where(gal_c2_mask, inv_sin_gal, 1e12)
                RG[3::5] = 0.001 * inv_sin_gal
                RG[4::5] = _iono_constraint_sigma(
                    gal_epoch,
                    num,
                    self.interval,
                    self.use_iono_rms,
                    self.sigma_iono_0,
                    self.sigma_iono_end,
                    self.t_end,
                )
            else:
                R = np.empty(4 * len(C1))
                R[0::4] = np.where(gps_c1_mask, inv_sin_gps, 1e12)
                R[1::4] = 0.001 * inv_sin_gps
                R[2::4] = np.where(gps_c2_mask, inv_sin_gps, 1e12)
                R[3::4] = 0.001 * inv_sin_gps

                RG = np.empty(4 * len(EC1))
                RG[0::4] = np.where(gal_c1_mask, inv_sin_gal, 1e12)
                RG[1::4] = 0.001 * inv_sin_gal
                RG[2::4] = np.where(gal_c2_mask, inv_sin_gal, 1e12)
                RG[3::4] = 0.001 * inv_sin_gal

            r_diag = np.concatenate((R, RG))
            r_diag = np.nan_to_num(r_diag, nan=1e12, posinf=1e12, neginf=1e12)
            m = r_diag.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = r_diag
            self.ekf.R = R
            updated = False
            for k in range(5):
                try:
                    self.ekf.predict_update(
                        z=Z,
                        HJacobian=self.Hjacobian,
                        args=(gps_satellites, gal_satellites),
                        Hx=self.Hx,
                        hx_args=(gps_satellites, gal_satellites),
                    )
                    updated = True
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-6 * (10.0 ** k)
                    self.ekf.R.flat[::m + 1] += jitter
            if not updated:
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "singular-update-reset",
                        epoch=num,
                        time=t,
                    )
                old_gps_sats, old_gal_sats = self.reset_filter(epoch=t)
                gps_arc_age = {sv: 0 for sv in old_gps_sats}
                gal_arc_age = {sv: 0 for sv in old_gal_sats}
                continue
            gps_arc_age = advance_arc_age(gps_arc_age, curr_gps_sats)
            gal_arc_age = advance_arc_age(gal_arc_age, curr_gal_sats)
            gps_ar_age = arc_age_array(gps_arc_age, curr_gps_sats)
            gal_ar_age = arc_age_array(gal_arc_age, curr_gal_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(curr_gps_sats),
                    n_gal=len(curr_gal_sats),
                    gps_ev=gps_epoch['ev'].to_numpy(copy=False),
                    gal_ev=gal_epoch['ev'].to_numpy(copy=False),
                    settings=ar_cfg,
                    gps_age=gps_ar_age,
                    gal_age=gal_ar_age,
                )
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
            else:
                enu = np.zeros(3)
            result_rows.append({
                'de': enu[0],
                'dn': enu[1],
                'du': enu[2],
                'dtr': dtr,
                'ztd': ztd,
                'x': xyz[0],
                'y': xyz[1],
                'z': xyz[2],
                'isb': isb,
                'ar_fixed': 0 if ar_diag is None else int(ar_diag.fixed_ambiguities),
                'ar_ratio': np.nan if ar_diag is None or ar_diag.ratio_min is None else float(ar_diag.ratio_min),
                'ar_ok': False if ar_diag is None else bool(ar_diag.accepted),
                'ar_gps_min_age': int(np.min(gps_ar_age)) if gps_ar_age.size else np.nan,
                'ar_gal_min_age': int(np.min(gal_ar_age)) if gal_ar_age.size else np.nan,
            })
            result_rows[-1].update(pppar_diagnostic_columns(ar_diag))
            if ar_diag is not None:
                result_rows[-1]['ar_fixed'] = int(ar_diag.fixed_ambiguities)
                result_rows[-1]['ar_ratio'] = np.nan if ar_diag.ratio_min is None else float(ar_diag.ratio_min)
                result_rows[-1]['ar_ok'] = bool(ar_diag.accepted)
            result_times.append(t)
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result_rows[-1],
                )
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
        gps_result = pd.concat(g)
        gal_result = pd.concat(e)
        result = pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time'))
        result['ct_min']=ct
        return result, gps_result, gal_result, ct


class PPPUdSingleGNSS:
    """Dual-frequency uncombined PPP filter for one constellation.

    Purpose:
        Active single-system uncombined dual-frequency branch for GPS, Galileo
        or BeiDou. It is selected when only one constellation is active and the
        configured mode contains two signals.

    Status:
        Active single-system uncombined dual-frequency path. Requires caution
        for receiver signal-bias states, BDS bias products, and ionospheric
        constraint datum handling.

    Model:
        Uses uncombined code and phase observations on both frequencies. The
        base model estimates receiver position, clock, optional ZTD, and
        per-satellite ``[I, N1, N2]`` states. When ionospheric constraints are
        enabled, an external ``ion`` pseudo-observation constrains the slant
        ionosphere state and optional receiver signal-bias datum states may be
        estimated for code bias handling.

    State vector:
        No constraints:
            ``[x, y, z, dtr, ztd?, (I, N1, N2)_s1, ...]``.
        With constraints and receiver signal-bias estimation:
            ``[x, y, z, dtr, ztd?, rx_code_bias_1, rx_code_bias_2,
            (I, N1, N2)_s1, ...]``.

    Supported systems/modes:
        GPS, Galileo and BeiDou dual-frequency modes known to ``gnx_py.gnss``.

    Bias / ionosphere handling:
        Satellite OSB/DCB corrections are read from preprocessed columns when
        available. In constrained mode, receiver signal-bias states protect the
        ionosphere datum from absorbing all code-bias mismatch.

    PPP-AR support:
        Uses indexed uncombined AR when enabled and when lock-age/candidate
        gates pass.
        BeiDou AR is disabled in current uncombined paths until BDS phase-bias
        handling is validated.

    Limitations:
        This class is numerically sensitive to the ionosphere/bias datum. Any
        change to state ordering, bias coefficients or constraint weighting
        requires regression against representative G/E/C data.
    """

    def __init__(self, config:PPPConfig, obs, mode, ekf, pos0, tro=True,interval=0.5):
        self.cfg=config
        self.obs = obs
        self.mode = mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.LAMBDA_DICT = {}
        self.CLIGHT = 299792458
        self.pos0 = pos0
        self.interval=interval
        self.use_iono_constr = bool(getattr(self.cfg, "use_iono_constr", False))
        self.use_iono_rms = bool(getattr(self.cfg, "use_iono_rms", True))
        self.sigma_iono_0 = getattr(self.cfg, "sigma_iono_0", 1.1)
        self.sigma_iono_end = getattr(self.cfg, "sigma_iono_end", 3.0)
        self.t_end = getattr(self.cfg, "t_end", 30)
        self.rx_dcb_col = None
        self.rx_signal_bias_cols = None
        self.base_dim = 5 if self.tro else 4
        if self.use_iono_constr and bool(getattr(self.cfg, "uncombined_est_rx_dcb", True)):
            self.rx_signal_bias_cols = (self.base_dim, self.base_dim + 1)
            # Backward-compatible diagnostic alias: report the estimated
            # second-frequency receiver signal bias as rx_dcb_m.
            self.rx_dcb_col = self.base_dim + 1
            self.base_dim += 2
        if self.use_iono_constr and self.obs is not None:
            if 'ion' not in self.obs.columns:
                raise ValueError("use_iono_constr requires an 'ion' column in observations.")
            self.obs = self.obs.dropna(subset=['ion'])
            if self.use_iono_rms and 'ion_rms' in self.obs.columns:
                self.obs = self.obs.dropna(subset=['ion_rms'])
        elif self.obs is None:
            self.use_iono_constr = False
        self.obs_per_sat = 5 if self.use_iono_constr else 4
        try:
            sig1, _ = _dual_mode_signals(self.mode)
            self.system = SIGNALS[sig1].system
        except Exception:
            # Fallback for unexpected single-frequency modes (should not happen for PPPUdSingleGNSS).
            self.system = "G"

    def _rx_dcb_coefficients(self) -> tuple[float, float]:
        sig1, sig2 = _dual_mode_signals(self.mode)
        f1 = self.FREQ_DICT[sig1]
        f2 = self.FREQ_DICT[sig2]
        return f2 ** 2 / (f1 ** 2 - f2 ** 2), f1 ** 2 / (f1 ** 2 - f2 ** 2)

    def _has_rx_signal_bias_states(self) -> bool:
        return self.rx_signal_bias_cols is not None

    def _init_rx_signal_bias_states(self) -> None:
        if not self._has_rx_signal_bias_states():
            return
        idx_c1, idx_c2 = self.rx_signal_bias_cols
        self.ekf.x[idx_c1] = 0.0
        self.ekf.P[idx_c1, idx_c1] = 1e-12
        self.ekf.Q[idx_c1, idx_c1] = 0.0
        self.ekf.P[idx_c2, idx_c2] = self.cfg.p_dcb
        self.ekf.Q[idx_c2, idx_c2] = self.cfg.q_dcb

    def _anchor_rx_signal_bias_datum(self) -> None:
        if not self._has_rx_signal_bias_states():
            return
        idx_c1, _idx_c2 = self.rx_signal_bias_cols
        self.ekf.x[idx_c1] = 0.0
        self.ekf.P[idx_c1, idx_c1] = 1e-12
        self.ekf.Q[idx_c1, idx_c1] = 0.0

    def _configure_transition_matrix(self) -> None:
        self.ekf.F = np.eye(self.ekf.dim_x)
        if not (
            self.use_iono_constr
            and bool(getattr(self.cfg, "uncombined_iono_clock_identity", True))
        ):
            self.ekf.F[3, 3] = 0.0

    def _single_iono_obs_variances(self, ev_deg: np.ndarray, n_sats: int) -> tuple[np.ndarray, np.ndarray]:
        if self.use_iono_constr and bool(getattr(self.cfg, "uncombined_iono_legacy_weighting", True)):
            inv_sin = _elevation_variance_scale(ev_deg)
            code_var = (
                float(getattr(self.cfg, "uncombined_iono_code_var_base", 0.3))
                + float(getattr(self.cfg, "uncombined_iono_code_var_elev_coeff", 0.0025)) * inv_sin
            )
            phase_var = (
                float(getattr(self.cfg, "uncombined_iono_phase_var_base", 1e-4))
                + float(getattr(self.cfg, "uncombined_iono_phase_var_elev_coeff", 0.0003)) * inv_sin
            )
            return code_var, phase_var

        code_var = float(getattr(self.cfg, "uncombined_code_var", 1.0))
        phase_var = float(getattr(self.cfg, "uncombined_phase_var", 1.0))
        return np.full(n_sats, code_var, dtype=float), np.full(n_sats, phase_var, dtype=float)

    def _code_bias_correction_m(self, epoch: pd.DataFrame, code_col: str, n: int) -> np.ndarray:
        """Return additive observable-specific code bias correction (meters)."""
        return -osb_m(epoch, code_col, n)

    def _phase_bias_correction_m(self, epoch: pd.DataFrame, phase_col: str, n: int) -> np.ndarray:
        """Return additive observable-specific phase bias correction (meters)."""
        return -osb_m(epoch, phase_col, n)

    def _bias_col_m(self, epoch: pd.DataFrame, column: str, n: int):
        return bias_column_m(epoch, column, n)

    def _dcb_between_m(self, epoch: pd.DataFrame, prefix: str, obs_a: str, obs_b: str, n: int):
        if obs_a == obs_b:
            return np.zeros(n, dtype=float)

        direct = self._bias_col_m(epoch, f"{prefix}{obs_a}_{obs_b}", n)
        if direct is not None:
            return direct

        reverse = self._bias_col_m(epoch, f"{prefix}{obs_b}_{obs_a}", n)
        if reverse is not None:
            return -reverse

        return None

    def _code_dsb_corrections_m(
        self,
        epoch: pd.DataFrame,
        code_1: str,
        code_2: str,
        coeff_1: float,
        coeff_2: float,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return split_dual_code_dsb_corrections_m(
            epoch,
            code_1,
            code_2,
            coeff_1=coeff_1,
            coeff_2=coeff_2,
            n=n,
            skip_if_satellite_osb=False,
        )

    def _initial_iono_values(self, epoch: pd.DataFrame) -> np.ndarray:
        if 'ion' in epoch.columns:
            ion = epoch['ion'].to_numpy(dtype=float, copy=False)
            if np.any(np.isfinite(ion)):
                return np.nan_to_num(ion, nan=0.0, posinf=0.0, neginf=0.0)

        return np.zeros(len(epoch), dtype=float)

    def _seed_iono_states_from_epoch(
        self,
        sats: list[str],
        epoch: pd.DataFrame,
        only_sats: set[str] | None = None,
    ) -> None:
        if epoch is None or len(epoch) == 0:
            return

        ion_seed = self._initial_iono_values(epoch)
        ion_by_sv = {
            sv: ion for sv, ion in zip(
                epoch.index.get_level_values('sv').tolist(),
                ion_seed,
            )
        }
        for idx, sv in enumerate(sats):
            if only_sats is not None and sv not in only_sats:
                continue
            ion = ion_by_sv.get(sv)
            if ion is None or not np.isfinite(ion):
                continue
            self.ekf.x[self.base_dim + 3 * idx] = float(ion)

    def Hjacobian(self, x, gps_satellites):
        C = self.CLIGHT
        sig1, sig2 = _dual_mode_signals(self.mode)
        F1 = self.FREQ_DICT[sig1]
        F2 = self.FREQ_DICT[sig2]
        L1 = C / F1
        L2 = C / F2
        MU1 = 1.0
        MU2 = (F1 / F2) ** 2

        sat_xyz = gps_satellites[:, :3]
        m_wet = gps_satellites[:, 3]
        n = sat_xyz.shape[0]
        rec_xyz = x[:3].copy()
        e_vec, _ = _unit_vectors_and_ranges(sat_xyz, rec_xyz)

        n_params = self.base_dim + 3 * n
        rows_per_sat = self.obs_per_sat
        H = np.zeros((rows_per_sat * n, n_params))
        COL_X, COL_Y, COL_Z, COL_CLK, COL_ZTD = 0, 1, 2, 3, 4

        if n:
            idx = np.arange(n)
            rows = (rows_per_sat * idx[:, None] + np.arange(4)).ravel()
            H[np.ix_(rows, [COL_X, COL_Y, COL_Z])] = np.repeat(e_vec, 4, axis=0)
            H[rows, COL_CLK] = 1.0
            H[rows, COL_ZTD] = np.repeat(m_wet, 4)

            row0 = rows_per_sat * idx
            row1 = row0 + 1
            row2 = row0 + 2
            row3 = row0 + 3
            col_iono = self.base_dim + 3 * idx
            H[row0, col_iono] = MU1
            H[row1, col_iono] = -MU1
            H[row2, col_iono] = MU2
            H[row3, col_iono] = -MU2
            H[row1, col_iono + 1] = L1
            H[row3, col_iono + 2] = L2
            if self._has_rx_signal_bias_states():
                rx_c1_col, rx_c2_col = self.rx_signal_bias_cols
                # Receiver signal-bias states define the observable datum for
                # each frequency, so the same bias enters code and phase rows.
                H[row0, rx_c1_col] = 1.0
                H[row1, rx_c1_col] = 1.0
                H[row2, rx_c2_col] = 1.0
                H[row3, rx_c2_col] = 1.0
            if self.use_iono_constr:
                H[row0 + 4, col_iono] = 1.0

        return H

    def Hx(self, x: np.ndarray, gps_satellites: np.ndarray) -> np.ndarray:
        x_state = x.copy()
        C = self.CLIGHT
        sig1, sig2 = _dual_mode_signals(self.mode)
        F1 = self.FREQ_DICT[sig1]
        F2 = self.FREQ_DICT[sig2]
        L1 = C / F1
        L2 = C / F2
        MU1 = 1.0
        MU2 = (F1 / F2) ** 2

        clk = x_state[3]
        zwd = x_state[4]
        n = gps_satellites.shape[0]
        base_dim = self.base_dim
        rows_per_sat = self.obs_per_sat

        z_hat = np.empty(rows_per_sat * n)

        if n:
            gps_state = x_state[base_dim:base_dim + 3 * n].reshape(n, 3)
            I = gps_state[:, 0]
            N1 = gps_state[:, 1]
            N2 = gps_state[:, 2]
            _, rho = _unit_vectors_and_ranges(gps_satellites[:, :3], x_state[:3])
            m_wet = gps_satellites[:, 3]
            geom = rho
            block = np.column_stack((
                geom + clk + m_wet * zwd + MU1 * I,
                geom + clk + m_wet * zwd - MU1 * I + L1 * N1,
                geom + clk + m_wet * zwd + MU2 * I,
                geom + clk + m_wet * zwd - MU2 * I + L2 * N2,
            ))
            if self._has_rx_signal_bias_states():
                rx_c1_col, rx_c2_col = self.rx_signal_bias_cols
                block[:, 0] += x_state[rx_c1_col]
                block[:, 1] += x_state[rx_c1_col]
                block[:, 2] += x_state[rx_c2_col]
                block[:, 3] += x_state[rx_c2_col]
            if self.use_iono_constr:
                block = np.column_stack((block, I))
            z_hat[:] = block.reshape(-1)

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

        prev_map = {sv: i for i, sv in enumerate(prev_sats)}
        curr_map = {sv: i for i, sv in enumerate(curr_sats)}
        common = [sv for sv in curr_sats if sv in prev_map]

        if common:
            prev_idx = np.fromiter((prev_map[sv] for sv in common), dtype=int)
            curr_idx = np.fromiter((curr_map[sv] for sv in common), dtype=int)
            offsets = np.arange(3)
            old_state_idx = (base_dim + 3 * prev_idx[:, None] + offsets[None, :]).ravel()
            new_state_idx = (base_dim + 3 * curr_idx[:, None] + offsets[None, :]).ravel()

            x_new[new_state_idx] = x_old[old_state_idx]
            base_idx = np.arange(base_dim)
            P_new[np.ix_(new_state_idx, base_idx)] = P_old[np.ix_(old_state_idx, base_idx)]
            P_new[np.ix_(base_idx, new_state_idx)] = P_old[np.ix_(base_idx, old_state_idx)]
            Q_new[np.ix_(new_state_idx, base_idx)] = Q_old[np.ix_(old_state_idx, base_idx)]
            Q_new[np.ix_(base_idx, new_state_idx)] = Q_old[np.ix_(base_idx, old_state_idx)]
            P_new[np.ix_(new_state_idx, new_state_idx)] = P_old[np.ix_(old_state_idx, old_state_idx)]
            Q_new[np.ix_(new_state_idx, new_state_idx)] = Q_old[np.ix_(old_state_idx, old_state_idx)]

        new_only = set(curr_sats) - set(prev_sats)
        for prn in new_only:
            j = curr_map[prn]
            idx_I = base_dim + 3 * j
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            P_new[idx_I, idx_I] = self.cfg.p_iono#100.0
            Q_new[idx_I, idx_I] = self.cfg.q_iono * (self.interval * 60) / 3600
            P_new[idx_N1, idx_N1] = self.cfg.p_amb#1e6
            P_new[idx_N2, idx_N2] = self.cfg.p_amb#1e6

        return x_new, P_new, Q_new

    def reset_filter(self, epoch, clk0: float = 0.0, zwd0: float = 0.0):
        sats0 = self.obs.loc[(slice(None), epoch), :].index.get_level_values('sv').tolist()
        n0 = len(sats0)
        dim_x = self.base_dim + 3 * n0
        dim_z = self.obs_per_sat * n0

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

        epoch0 = self.obs.loc[(slice(None), epoch), :]
        I_init = self._initial_iono_values(epoch0)
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt * (self.interval * 60) / 3600 #9e9
        self.ekf.Q[4, 4] = self.cfg.q_tro#0.025
        self._init_rx_signal_bias_states()
        self._configure_transition_matrix()

        for k in range(n0):
            idx_I = self.base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            self.ekf.P[idx_I, idx_I] =self.cfg.p_iono# 100
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono* (self.interval * 60) / 3600
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb #1e6

            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb #1e6
            self.ekf.x[idx_I] = I_init[k]  # seed I_s from the selected ionosphere model constraint

        return sats0

    def init_filter(self, clk0: float = 0.0, zwd0: float = 0.0):
        gps_epochs = sorted(self.obs.index.get_level_values('time').unique())
        first_ep = gps_epochs[0]
        sats0 = self.obs.loc[(slice(None), first_ep), :].index.get_level_values('sv').tolist()
        n0 = len(sats0)
        dim_x = self.base_dim + 3 * n0
        dim_z = self.obs_per_sat * n0

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

        first_epoch = self.obs.loc[(slice(None), first_ep), :]
        I_init = self._initial_iono_values(first_epoch)
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt #9e9
        self.ekf.Q[4, 4] = self.cfg.q_tro #0.025
        self._init_rx_signal_bias_states()
        self._configure_transition_matrix()

        for k in range(n0):
            idx_I = self.base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono #100
            self.ekf.Q[idx_I, idx_I] = self.cfg.q_iono * (self.interval * 60) / 3600#25
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb#1e6

            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb #1e6
            self.ekf.x[idx_I] = I_init[k]  # seed I_s from the selected ionosphere model constraint

        return sats0, gps_epochs

    def _prepare_obs(self):
        gps_data = _dual_mode_columns(self.obs, self.mode)

        self.obs = self.obs.copy()

        for col in _dual_rec_pco_columns(self.mode):
            if col not in self.obs.columns.tolist():
                self.obs.loc[:, col] = 0.0

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
        offsets = np.array([1, 2])
        idxs_N = (base_dim + 3 * np.arange(n_sats)[:, None] + offsets[None, :]).ravel()
        N_vec = x[idxs_N]
        P_N = P[np.ix_(idxs_N, idxs_N)]

        return N_vec, P_N

    def code_screening(self, x, satellites, code_obs, thr=1):
        sat_xyz = np.asarray(satellites, dtype=float)
        ref_xyz = np.asarray(x, dtype=float)
        dist = np.linalg.norm(sat_xyz - ref_xyz, axis=1)
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
        result_rows = []
        result_times = []
        obs_result =[]
        self.obs = self.obs.sort_values(by='sv')
        obs_epochs = {t: df for t, df in self.obs.groupby(level=1, sort=False)}
        gps_osb_c1_col = f'OSB_{gps_c1}'
        gps_osb_c2_col = f'OSB_{gps_c2}'
        gps_osb_l1_col = f'OSB_{gps_l1}'
        gps_osb_l2_col = f'OSB_{gps_l2}'
        gps_has_osb_c1 = gps_osb_c1_col in self.obs.columns
        gps_has_osb_c2 = gps_osb_c2_col in self.obs.columns
        gps_has_osb_l1 = gps_osb_l1_col in self.obs.columns
        gps_has_osb_l2 = gps_osb_l2_col in self.obs.columns
        code_var = float(getattr(self.cfg, "uncombined_code_var", 1.0))
        phase_var = float(getattr(self.cfg, "uncombined_phase_var", 1.0))
        code_prefit_threshold = float(getattr(self.cfg, "uncombined_code_prefit_threshold", 30.0))
        phase_screen_enabled = bool(getattr(self.cfg, "uncombined_phase_screen", False))
        phase_screen_threshold = float(getattr(self.cfg, "uncombined_phase_screen_threshold", 10.0))
        phase_screen_warmup_epochs = int(getattr(self.cfg, "uncombined_phase_screen_warmup_epochs", 60))
        phase_screen_legacy_source = False
        if self.use_iono_constr:
            code_prefit_threshold = float(
                getattr(self.cfg, "uncombined_iono_code_prefit_threshold", 10.0)
            )
            phase_screen_enabled = bool(getattr(self.cfg, "uncombined_iono_phase_screen", True))
            phase_screen_threshold = float(
                getattr(self.cfg, "uncombined_iono_phase_screen_threshold", 1.0)
            )
            phase_screen_warmup_epochs = int(
                getattr(self.cfg, "uncombined_iono_phase_screen_warmup_epochs", 60)
            )
            phase_screen_legacy_source = bool(
                getattr(self.cfg, "uncombined_iono_phase_screen_legacy_source", True)
            )
        sig1, sig2 = _dual_mode_signals(self.mode)
        pco1_col, pco2_col = _dual_rec_pco_columns(self.mode)
        r_cache = {}
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
        xyz = None
        conv_time = None
        T0 = all_epochs[0]
        arc_age = {sv: 0 for sv in old_sats}
        phase_traces_1: dict[int, dict[str, float]] = {}
        phase_traces_2: dict[int, dict[str, float]] = {}
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

            def safe_get(df, col, length=None):
                """Zwraca kolumnę jako numpy, a jeśli brak – wektor zer o podanej długości."""
                if col in df.columns:
                    return df[col].to_numpy(copy=False)
                elif length is not None:
                    return np.zeros(length)
                else:
                    return np.zeros(len(df))

            # --- GPS ---
            gps_cols = [
                'clk', 'tro', 'ah_los',
                'dprel', pco1_col, pco2_col,
                f'sat_pco_los_{sig1}',
                f'sat_pco_los_{sig2}',
                'phw', 'me_wet'
            ]
            gps_len = len(gps_epoch)
            (gps_clk, gps_tro, gps_ah_los, gps_dprel, gps_pco1, gps_pco2,
             sat_pco_L1, sat_pco_L2, phw, mwet) = [safe_get(gps_epoch, c, gps_len) for c in gps_cols]


            curr_sats = gps_epoch.index.get_level_values('sv').tolist()

            if curr_sats != old_sats:
                new_sats = set(curr_sats) - set(old_sats)
                new_x, new_P, new_Q = self.rebuild_state(
                    self.ekf.x.copy(), self.ekf.P.copy(), self.ekf.Q.copy(),
                    old_sats, curr_sats
                )
                self.ekf.x, self.ekf.P, self.ekf.Q = new_x, new_P, new_Q
                self.ekf.dim_x = len(new_x)
                self.ekf.dim_z = self.obs_per_sat * len(curr_sats)
                self.ekf._I = np.eye(self.ekf.dim_x)
                self._configure_transition_matrix()
                self._anchor_rx_signal_bias_datum()
                self._seed_iono_states_from_epoch(curr_sats, gps_epoch, new_sats)

            old_sats = curr_sats
            arc_age = {sv: arc_age.get(sv, 0) for sv in curr_sats}

            tides = gps_epoch['tides_los'].to_numpy(copy=False)
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            gps_p1_c1 = osb_m(gps_epoch, gps_c1, gps_len)
            gps_p2_c2 = osb_m(gps_epoch, gps_c2, gps_len)
            gps_pl1_l1 = osb_m(gps_epoch, gps_l1, gps_len)
            gps_pl2_l2 = osb_m(gps_epoch, gps_l2, gps_len)

            code_corr_c1 = np.zeros(gps_len, dtype=float)
            code_corr_c2 = np.zeros(gps_len, dtype=float)
            dcb_total = np.zeros(gps_len, dtype=float)
            if not has_satellite_osb(gps_epoch, (gps_c1, gps_c2)):
                code_corr_c1, code_corr_c2, dcb_total = self._code_dsb_corrections_m(
                    gps_epoch,
                    gps_c1,
                    gps_c2,
                    agps,
                    bgps,
                    gps_len,
                )

            C1 = (
                gps_epoch[gps_c1].to_numpy(copy=False)
                - gps_pco1
                + sat_pco_L1
                + gps_clk * self.CLIGHT
                - gps_tro
                - gps_ah_los
                - gps_dprel
                - gps_p1_c1
                + code_corr_c1
                - tides
            )
            C2 = (
                gps_epoch[gps_c2].to_numpy(copy=False)
                - gps_pco2
                + sat_pco_L2
                + gps_clk * self.CLIGHT
                - gps_tro
                - gps_ah_los
                - gps_dprel
                - gps_p2_c2
                + code_corr_c2
                - tides
            )
            L1 = (
                gps_epoch[gps_l1].to_numpy(copy=False)
                - gps_pco1
                + sat_pco_L1
                + gps_clk * self.CLIGHT
                - gps_tro
                - gps_ah_los
                - gps_dprel
                - gps_pl1_l1
                - tides
                - (self.CLIGHT / self.FREQ_DICT[sig1]) * phw
            )
            L2 = (
                gps_epoch[gps_l2].to_numpy(copy=False)
                - gps_pco2
                + sat_pco_L2
                + gps_clk * self.CLIGHT
                - gps_tro
                - gps_ah_los
                - gps_dprel
                - gps_pl2_l2
                - tides
                - (self.CLIGHT / self.FREQ_DICT[sig2]) * phw
            )

            if self.use_iono_constr:
                IONO = gps_epoch['ion'].to_numpy(copy=False)
                Z = np.empty(5 * len(C1))
                Z[0::5] = C1
                Z[1::5] = L1
                Z[2::5] = C2
                Z[3::5] = L2
                Z[4::5] = IONO
            else:
                Z = np.empty(4 * len(C1))
                Z[0::4] = C1
                Z[1::4] = L1
                Z[2::4] = C2
                Z[3::4] = L2

            gps_c1_mask = self.code_screening(
                x=self.ekf.x[:3],
                code_obs=C1,
                thr=code_prefit_threshold,
                satellites=gps_satellites[:, :3],
            )
            gps_c2_mask = self.code_screening(
                x=self.ekf.x[:3],
                code_obs=C2,
                thr=code_prefit_threshold,
                satellites=gps_satellites[:, :3],
            )

            rows_per_sat = 5 if self.use_iono_constr else 4
            z_hat_prefit = self.Hx(self.ekf.x, gps_satellites)
            prefit = Z - z_hat_prefit
            n_sats = len(curr_sats)
            prefit_mat = prefit.reshape(n_sats, rows_per_sat)

            if phase_screen_legacy_source:
                _, rho_phase = _unit_vectors_and_ranges(gps_satellites[:, :3], self.ekf.x[:3])
                phase_trace_1 = L1 - rho_phase
                phase_trace_2 = L2 - rho_phase
            else:
                phase_trace_1 = prefit_mat[:, 1]
                phase_trace_2 = prefit_mat[:, 3]
            phase_traces_1[num] = {sv: float(r) for sv, r in zip(curr_sats, phase_trace_1)}
            phase_traces_2[num] = {sv: float(r) for sv, r in zip(curr_sats, phase_trace_2)}
            phase_reset_svs: set[str] = set()
            if phase_screen_enabled and num >= phase_screen_warmup_epochs:
                for idx in phase_residuals_outliers(curr_sats, phase_traces_1, num, thr=phase_screen_threshold):
                    idx_n1 = self.base_dim + 3 * idx + 1
                    self.ekf.x[idx_n1] = 0.0
                    self.ekf.P[idx_n1, idx_n1] = float(max(getattr(self.cfg, "p_amb", 1e6), 1e3))
                    phase_reset_svs.add(curr_sats[idx])
                for idx in phase_residuals_outliers(curr_sats, phase_traces_2, num, thr=phase_screen_threshold):
                    idx_n2 = self.base_dim + 3 * idx + 2
                    self.ekf.x[idx_n2] = 0.0
                    self.ekf.P[idx_n2, idx_n2] = float(max(getattr(self.cfg, "p_amb", 1e6), 1e3))
                    phase_reset_svs.add(curr_sats[idx])

            gps_ev = gps_epoch['ev'].to_numpy(copy=False) if 'ev' in gps_epoch.columns else np.full(
                n_sats, 45.0, dtype=float
            )

            if self.use_iono_constr:
                code_var_epoch, phase_var_epoch = self._single_iono_obs_variances(gps_ev, n_sats)
                R_diag = np.ones(5 * n_sats, dtype=float)
                R_diag[0::5] = np.where(gps_c1_mask, code_var_epoch, 1e12)
                R_diag[2::5] = np.where(gps_c2_mask, code_var_epoch, 1e12)
                both_code_rejected = (~gps_c1_mask) & (~gps_c2_mask)
                R_diag[1::5] = np.where(both_code_rejected, 1e12, phase_var_epoch)
                R_diag[3::5] = np.where(both_code_rejected, 1e12, phase_var_epoch)
                R_diag[4::5] = _iono_constraint_sigma(
                    gps_epoch,
                    num,
                    self.interval,
                    self.use_iono_rms,
                    self.sigma_iono_0,
                    self.sigma_iono_end,
                    self.t_end,
                )
            else:
                R_diag = np.ones(4 * n_sats, dtype=float)
                both_code_rejected = (~gps_c1_mask) & (~gps_c2_mask)
                R_diag[0::4] = np.where(gps_c1_mask, code_var, 1e12)
                R_diag[2::4] = np.where(gps_c2_mask, code_var, 1e12)
                R_diag[1::4] = np.where(both_code_rejected, 1e12, phase_var)
                R_diag[3::4] = np.where(both_code_rejected, 1e12, phase_var)

            R_diag = np.nan_to_num(R_diag, nan=1e12, posinf=1e12, neginf=1e12)
            m = R_diag.size
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)
            R.flat[::m + 1] = R_diag
            self.ekf.R = R

            self.ekf.predict_update(z=Z,
                                    HJacobian=self.Hjacobian,
                                    args=(gps_satellites,),
                                    Hx=self.Hx,
                                    hx_args=(gps_satellites,))
            self._anchor_rx_signal_bias_datum()
            arc_age = advance_arc_age(arc_age, curr_sats, phase_reset_svs)
            ar_age = arc_age_array(arc_age, curr_sats)
            postfit = Z - self.Hx(self.ekf.x, gps_satellites)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(curr_sats),
                    n_gal=0,
                    gps_ev=gps_ev,
                    gal_ev=np.empty(0, dtype=float),
                    settings=ar_cfg,
                    gps_age=ar_age,
                )

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
            else:
                enu = np.zeros(3)
            n_sats = len(curr_sats)
            n_code_rejected_c1 = int(np.count_nonzero(~gps_c1_mask))
            n_code_rejected_c2 = int(np.count_nonzero(~gps_c2_mask))
            n_code_rejected_total = n_code_rejected_c1 + n_code_rejected_c2

            prefit_mat = prefit.reshape(n_sats, rows_per_sat)
            postfit_mat = postfit.reshape(n_sats, rows_per_sat)
            used_c1 = np.asarray(gps_c1_mask, dtype=bool)
            used_c2 = np.asarray(gps_c2_mask, dtype=bool)

            code_used = np.concatenate((postfit_mat[:, 0][used_c1], postfit_mat[:, 2][used_c2]))
            if code_used.size:
                code_rms = float(np.sqrt(np.mean(np.square(code_used))))
                code_common = float(np.median(code_used))
                code_scatter_rms = float(np.sqrt(np.mean(np.square(code_used - code_common))))
            else:
                code_rms = np.nan
                code_scatter_rms = np.nan

            used_phase = ~(~used_c1 & ~used_c2)
            phase_used = np.concatenate((postfit_mat[:, 1][used_phase], postfit_mat[:, 3][used_phase]))
            if phase_used.size:
                phase_rms = float(np.sqrt(np.mean(np.square(phase_used))))
                phase_common = float(np.median(phase_used))
                phase_scatter_rms = float(np.sqrt(np.mean(np.square(phase_used - phase_common))))
            else:
                phase_rms = np.nan
                phase_scatter_rms = np.nan
            rx_bias_1 = np.nan
            rx_bias_2 = np.nan
            if self._has_rx_signal_bias_states():
                rx_bias_1 = float(self.ekf.x[self.rx_signal_bias_cols[0]])
                rx_bias_2 = float(self.ekf.x[self.rx_signal_bias_cols[1]])
            result_rows.append({
                'de': enu[0],
                'dn': enu[1],
                'du': enu[2],
                'dtr': dtr,
                'ztd': ztd,
                'x': xyz[0],
                'y': xyz[1],
                'z': xyz[2],
                'rx_dcb_m': rx_bias_2,
                'rx_code_bias_1_m': rx_bias_1,
                'rx_code_bias_2_m': rx_bias_2,
                'n_sats': n_sats,
                'n_states': int(len(self.ekf.x)),
                'n_code_rejected_c1': n_code_rejected_c1,
                'n_code_rejected_c2': n_code_rejected_c2,
                'n_code_rejected_total': n_code_rejected_total,
                'code_res_rms': code_rms,
                'phase_res_rms': phase_rms,
                'code_res_scatter_rms': code_scatter_rms,
                'phase_res_scatter_rms': phase_scatter_rms,
                'ar_fixed': 0 if ar_diag is None else ar_diag.fixed_ambiguities,
                'ar_ratio': np.nan if ar_diag is None or ar_diag.ratio_min is None else ar_diag.ratio_min,
                'ar_ok': False if ar_diag is None else ar_diag.accepted,
                'ar_min_age': int(np.min(ar_age)) if ar_age.size else np.nan,
            })
            result_rows[-1].update(pppar_diagnostic_columns(ar_diag))
            result_times.append(t)
            epoch_out = gps_epoch.copy()
            epoch_out['system'] = self.system
            epoch_out['dtr'] = dtr
            epoch_out['ztd'] = ztd
            epoch_out['prefit_code_res_1'] = prefit_mat[:, 0]
            epoch_out['prefit_phase_res_1'] = prefit_mat[:, 1]
            epoch_out['prefit_code_res_2'] = prefit_mat[:, 2]
            epoch_out['prefit_phase_res_2'] = prefit_mat[:, 3]
            epoch_out['postfit_code_res_1'] = postfit_mat[:, 0]
            epoch_out['postfit_phase_res_1'] = postfit_mat[:, 1]
            epoch_out['postfit_code_res_2'] = postfit_mat[:, 2]
            epoch_out['postfit_phase_res_2'] = postfit_mat[:, 3]

            epoch_out['code_used_1'] = used_c1
            epoch_out['code_used_2'] = used_c2
            if self.use_iono_constr:
                epoch_out['prefit_iono_res'] = prefit_mat[:, 4]
                epoch_out['postfit_iono_res'] = postfit_mat[:, 4]
            epoch_out['code_dcb_corr_1'] = code_corr_c1
            epoch_out['code_dcb_corr_2'] = code_corr_c2
            epoch_out['dcb_total_m'] = dcb_total
            epoch_out['rx_dcb_m'] = 0.0 if not self._has_rx_signal_bias_states() else rx_bias_2
            epoch_out['rx_code_bias_1_m'] = 0.0 if not self._has_rx_signal_bias_states() else rx_bias_1
            epoch_out['rx_code_bias_2_m'] = 0.0 if not self._has_rx_signal_bias_states() else rx_bias_2
            state_block = self.ekf.x[self.base_dim:self.base_dim + 3 * n_sats].reshape(n_sats, 3)
            epoch_out['I_state'] = state_block[:, 0]
            epoch_out['N1_state'] = state_block[:, 1]
            epoch_out['N2_state'] = state_block[:, 2]
            obs_result.append(epoch_out)
            if trace_filter:
                if reset_epoch:
                    ppp_trace(trace_filter, self.__class__.__name__, "reset", epoch=num, time=t)
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result_rows[-1],
                )
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
        output = pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time'))
        output['ct_min'] = ct
        residuals = pd.concat(obs_result) if obs_result else None
        return output, residuals, residuals, ct


class PPPFilterMultiGNSSIonConst:
    """Legacy/reference GPS+Galileo uncombined PPP with ionospheric constraints.

    Purpose:
        Important reference implementation for constrained dual-frequency
        GPS+Galileo uncombined PPP. It is still used by routing for legacy G/E
        constrained runs and should not be changed without strong numerical
        regression evidence.

    Status:
        Legacy/reference and still routed. This is not dead code; it is the
        benchmark implementation for constrained GPS+Galileo behavior and a
        required comparison target for newer G/E/C constrained work.

    Model:
        Uses uncombined code and phase observations on two frequencies plus one
        ionosphere pseudo-observation per satellite. GPS provides the reference
        receiver clock; Galileo is represented by an ISB state. Optional
        receiver DCB states can be estimated after the satellite blocks.

    State vector:
        ``[x, y, z, dtr_G, isb_E, ztd?, (I, N1, N2)_G...,
        (I, N1, N2)_E..., dcb_states?]``.

    Supported systems:
        GPS and Galileo only.

    PPP-AR support:
        Uncombined AR hooks are present in this branch. Arc age and phase-jump
        resets are part of the safety model; changing them can change
        ambiguity-resolution behavior.

    Limitations / warnings:
        Treat this as legacy/reference, not dead code. It is a key comparison
        target for newer generic G/E/C constrained work. Edits to measurement
        weighting, ionospheric sigmas, DCB placement or state rebuild logic are
        numerical model changes.
    """

    def __init__(self, config: PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, est_dcb=False,
                 interval=0.5,
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
        self.cfg = config
        self.gps_obs = gps_obs
        self.gal_obs = gal_obs
        self.gps_mode = gps_mode
        self.gal_mode = gal_mode
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
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

        F1_GAL = self.FREQ_DICT[self.gal_mode[:2] if self.gal_mode != 'E5aE5b' else self.gal_mode[:3]]
        F2_GAL = self.FREQ_DICT[self.gal_mode[2:] if self.gal_mode != 'E5aE5b' else self.gal_mode[3:]]
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
        F1_GAL = self.FREQ_DICT[self.gal_mode[:2] if self.gal_mode != 'E5aE5b' else self.gal_mode[:3]]
        F2_GAL = self.FREQ_DICT[self.gal_mode[2:] if self.gal_mode != 'E5aE5b' else self.gal_mode[3:]]
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

            if mode == 'E5aE5b':
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
        self.ekf.P[3, 3] = self.cfg.p_dt  # 1e9 # clk
        self.ekf.P[4, 4] = self.cfg.p_isb  # 100 # isb
        self.ekf.P[5, 5] = self.cfg.p_tro  # 2.0 # tro
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt  # 1e9 # clk
        self.ekf.Q[4, 4] = self.cfg.q_isb  # 1.0 # isb
        self.ekf.Q[5, 5] = self.cfg.q_tro  # 0.025 # tro
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
        base = self.base_dim
        dcb_block = 4 if self.est_dcb else 0

        n_prev = len(prev_gps) + len(prev_gal)
        n_curr = len(curr_gps) + len(curr_gal)

        old_dim = base + 3 * n_prev + dcb_block
        new_dim = base + 3 * n_curr + dcb_block

        x_new = np.zeros(new_dim)
        P_new = np.zeros((new_dim, new_dim))
        Q_new = np.zeros((new_dim, new_dim))

        # base block
        x_new[:base] = x_old[:base]
        P_new[:base, :base] = P_old[:base, :base]
        Q_new[:base, :base] = Q_old[:base, :base]

        def tagG(sv):
            return f"G:{sv}"

        def tagE(sv):
            return f"E:{sv}"

        prev_all = [tagG(sv) for sv in prev_gps] + [tagE(sv) for sv in prev_gal]
        curr_all = [tagG(sv) for sv in curr_gps] + [tagE(sv) for sv in curr_gal]

        prev_map = {prn: i for i, prn in enumerate(prev_all)}
        curr_map = {prn: i for i, prn in enumerate(curr_all)}

        common = [prn for prn in curr_all if prn in prev_map]  # kolejność jak w curr
        if common:
            old_idx = np.array([prev_map[p] for p in common], dtype=int)
            new_idx = np.array([curr_map[p] for p in common], dtype=int)

            # indeksy stanów (I,N1,N2) dla wszystkich common sat
            offs = np.array([0, 1, 2], dtype=int)
            old_state = (base + 3 * old_idx[:, None] + offs[None, :]).ravel()
            new_state = (base + 3 * new_idx[:, None] + offs[None, :]).ravel()

            # x
            x_new[new_state] = x_old[old_state]

            # P/Q: sat-sat blok naraz
            P_new[np.ix_(new_state, new_state)] = P_old[np.ix_(old_state, old_state)]
            Q_new[np.ix_(new_state, new_state)] = Q_old[np.ix_(old_state, old_state)]

            # cross z base
            b = np.arange(base)
            P_new[np.ix_(new_state, b)] = P_old[np.ix_(old_state, b)]
            P_new[np.ix_(b, new_state)] = P_old[np.ix_(b, old_state)]
            Q_new[np.ix_(new_state, b)] = Q_old[np.ix_(old_state, b)]
            Q_new[np.ix_(b, new_state)] = Q_old[np.ix_(b, old_state)]

        # nowe satelity: tylko diagonalne init
        new_only = [prn for prn in curr_all if prn not in prev_map]
        for prn in new_only:
            j = curr_map[prn]
            idx_I = base + 3 * j
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            P_new[idx_I, idx_I] = self.cfg.p_iono
            Q_new[idx_I, idx_I] = self.cfg.q_iono * (self.interval * 60) / 3600
            P_new[idx_N1, idx_N1] = self.cfg.p_amb
            P_new[idx_N2, idx_N2] = self.cfg.p_amb

        # DCB block transfer (jak miałeś)
        if self.est_dcb:
            dcb_new = self.dcb_param_indices(len(curr_gps), len(curr_gal))
            dcb_old = self.dcb_param_indices(len(prev_gps), len(prev_gal))
            for i, idx in enumerate(dcb_new):
                if i < len(dcb_old):
                    x_new[idx] = x_old[dcb_old[i]]
                    P_new[idx, idx] = P_old[dcb_old[i], dcb_old[i]]
                    Q_new[idx, idx] = Q_old[dcb_old[i], dcb_old[i]]
                else:
                    P_new[idx, idx] = self.cfg.p_dcb
                    Q_new[idx, idx] = self.cfg.q_dcb

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
        sat_xyz = np.asarray(satellites, dtype=float)
        ref_xyz = np.asarray(x, dtype=float)
        dist = np.linalg.norm(sat_xyz - ref_xyz, axis=1)
        prefit = code_obs - dist
        median_prefit = np.median(prefit)
        mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
        n_sat = len(prefit)
        n_bad = np.count_nonzero(~mask)
        if n_bad > n_sat / 2:
            mask = np.ones(n_sat, dtype=bool)
        return mask, prefit

    def phase_residuals_screening(self, sat_list, phase_residuals_dict, num, thr=1, sys='G', len_gps=None, freq='n1'):
        if sys == 'E':
            assert len_gps is not None

        N = 5
        # mapowanie sv -> pozycja w sat_list (O(n) raz)
        idx_map = {sv: i for i, sv in enumerate(sat_list)}

        valid_svs = []
        diffs = []
        reset_svs = []

        curr = phase_residuals_dict.get(num, {})
        prev1 = phase_residuals_dict.get(num - 1, {})

        # sprawdź komplet N poprzednich epok (identyczne kryterium jak u Ciebie)
        for sv in sat_list:
            ok = True
            for offset in range(N):
                if phase_residuals_dict.get(num - offset - 1, {}).get(sv) is None:
                    ok = False
                    break
            if not ok:
                continue
            if curr.get(sv) is None or prev1.get(sv) is None:
                continue
            valid_svs.append(sv)
            diffs.append(curr[sv] - prev1[sv])

        if diffs:
            med = np.median(diffs)
        else:
            med = 0.0

        for sv, d in zip(valid_svs, diffs):
            if abs(d - med) > thr:
                outlier_idx = idx_map[sv]
                if sys == 'E':
                    outlier_idx += len_gps
                base = 3 * outlier_idx
                if freq == 'n1':
                    nidx = base + 1
                else:
                    nidx = base + 2
                ii = self.base_dim + nidx
                self.ekf.x[ii] = 0.0
                self.ekf.P[ii, ii] = 1e6
                self.ekf.Q[ii, ii] = 0.0
                reset_svs.append(sv)
        return reset_svs

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
        ar_float_P = self.ekf.P.copy()
        (agps, bgps, gps_cols), (agal, bgal, gal_cols) = self._prepare_obs()
        # sigma_iono_0=1.1, sigma_iono_end=3,t_end = 30 for iono rms
        # 0.01 2.5 for no iono rms
        result = []
        result_gps = []
        result_gal = []

        # NEW
        gps_rows = []  # index (MultiIndex)
        gps_data = []  # dict lub ndarray

        gal_rows = []
        gal_data = []

        t_end = self.t_end / self.interval

        sigma_iono_t = self.sigma_iono_0 + (self.sigma_iono_end - self.sigma_iono_0) / t_end * np.arange(t_end)
        xyz = None
        conv_time = None
        T0 = epochs[0]
        prt_gps_l1 = {}
        prt_gps_l2 = {}
        prt_gal_l1 = {}
        prt_gal_l2 = {}
        gps_arc_age: dict[str, int] = {}
        gal_arc_age: dict[str, int] = {}
        self.gps_obs.sort_values(by='sv', inplace=True)
        self.gal_obs.sort_values(by='sv', inplace=True)
        gps_epochs = {
            t: df for t, df in self.gps_obs.groupby(level=1, sort=False)
        }
        gal_epochs = {
            t: df for t, df in self.gal_obs.groupby(level=1, sort=False)
        }
        r_cache = {}
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

        def _reset_ar_float_ambiguities(system: str, svs: set[str], curr_gps: list[str], curr_gal: list[str]) -> None:
            if not svs or ar_float_P.shape != self.ekf.P.shape:
                return
            indices: list[int] = []
            if system == "G":
                sat_lookup = {sv: idx for idx, sv in enumerate(curr_gps)}
                for sv in svs:
                    if sv in sat_lookup:
                        base = self.base_dim + 3 * sat_lookup[sv]
                        indices.extend([base + 1, base + 2])
            else:
                sat_lookup = {sv: idx for idx, sv in enumerate(curr_gal)}
                for sv in svs:
                    if sv in sat_lookup:
                        base = self.base_dim + 3 * (len(curr_gps) + sat_lookup[sv])
                        indices.extend([base + 1, base + 2])
            _reset_covariance_states(ar_float_P, indices, float(self.cfg.p_amb))

        for num, t in enumerate(epochs):
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    ppp_trace(trace_filter, self.__class__.__name__, "reset", epoch=num, time=t)
                    gps_sats, gal_sats = self.reset_filter(epoch=t)
                    ar_float_P = self.ekf.P.copy()
                    gps_arc_age = {sv: 0 for sv in gps_sats}
                    gal_arc_age = {sv: 0 for sv in gal_sats}
                    reset_epoch = True
                    T0 = t
                    # n_i = 0

            gps_epoch = gps_epochs.get(t)
            gal_epoch = gal_epochs.get(t)
            if gps_epoch is None or gal_epoch is None:
                continue
            curr_gps_sats = gps_epoch.index.get_level_values('sv').tolist()
            curr_gal_sats = gal_epoch.index.get_level_values('sv').tolist()
            if (curr_gps_sats != gps_sats) or (curr_gal_sats != gal_sats):
                old_x = self.ekf.x
                old_Q = self.ekf.Q
                self.ekf.x, self.ekf.P, self.ekf.Q = self.rebuild_state(
                    self.ekf.x, self.ekf.P, self.ekf.Q,
                    gps_sats, gal_sats, curr_gps_sats, curr_gal_sats
                )
                if ar_float_P.shape == old_Q.shape:
                    _, ar_float_P, _ = self.rebuild_state(
                        old_x, ar_float_P, old_Q,
                        gps_sats, gal_sats, curr_gps_sats, curr_gal_sats
                    )
                else:
                    ar_float_P = self.ekf.P.copy()
                self.ekf.dim_x = len(self.ekf.x)
                self.ekf.dim_z = 5 * (len(curr_gps_sats) + len(curr_gal_sats))
                self.ekf._I = np.eye(self.ekf.dim_x)
                self.ekf.F = np.eye(self.ekf.dim_x)

                n_gps = len(gps_epoch);
                n_gal = len(gal_epoch)
                base = self.base_dim + 3 * (n_gps + n_gal)
                idx_c1gps = base + 0  # DCB_C1_GPS
                idx_c1gal = base + 2
                self.ekf.x[idx_c1gps] = 0.0
                self.ekf.P[idx_c1gps, idx_c1gps] = 1e-12
                self.ekf.Q[idx_c1gps, idx_c1gps] = 0.0

                self.ekf.x[idx_c1gal] = 0.0
                self.ekf.P[idx_c1gal, idx_c1gal] = 1e-12
                self.ekf.Q[idx_c1gal, idx_c1gal] = 0.0
                if ar_float_P.shape == self.ekf.P.shape:
                    ar_float_P[idx_c1gps, :] = 0.0
                    ar_float_P[:, idx_c1gps] = 0.0
                    ar_float_P[idx_c1gps, idx_c1gps] = 1e-12
                    ar_float_P[idx_c1gal, :] = 0.0
                    ar_float_P[:, idx_c1gal] = 0.0
                    ar_float_P[idx_c1gal, idx_c1gal] = 1e-12
            gps_sats = curr_gps_sats
            gal_sats = curr_gal_sats
            gps_arc_age = {sv: gps_arc_age.get(sv, 0) for sv in curr_gps_sats}
            gal_arc_age = {sv: gal_arc_age.get(sv, 0) for sv in curr_gal_sats}
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

            dist_gps = calculate_distance(gps_satellites[:, :3], self.ekf.x[:3])
            dist_gal = calculate_distance(gal_satellites[:, :3], self.ekf.x[:3])
            prefit_gps_l1 = L1 - dist_gps
            prefit_gps_l2 = L2 - dist_gps
            prt_gps_l1[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_l1)}
            prt_gps_l2[num] = {sv: r for sv, r in zip(curr_gps_sats, prefit_gps_l2)}
            gps_phase_reset_svs: set[str] = set()
            if num >= 60:
                gps_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gps_sats, phase_residuals_dict=prt_gps_l1, num=num, freq='n1'
                    )
                )
                gps_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gps_sats, phase_residuals_dict=prt_gps_l2, num=num, freq='n2'
                    )
                )
            for sv in gps_phase_reset_svs:
                gps_arc_age[sv] = 0
            _reset_ar_float_ambiguities("G", gps_phase_reset_svs, curr_gps_sats, curr_gal_sats)

            prefit_gal_l1 = EL1 - dist_gal
            prefit_gal_l2 = EL2 - dist_gal
            prt_gal_l1[num] = {sv: r for sv, r in zip(curr_gal_sats, prefit_gal_l1)}
            prt_gal_l2[num] = {sv: r for sv, r in zip(curr_gal_sats, prefit_gal_l2)}
            gal_phase_reset_svs: set[str] = set()
            if num >= 60:
                gal_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gal_sats, phase_residuals_dict=prt_gal_l1, num=num,
                        sys='E', len_gps=len(curr_gps_sats), freq='n1'
                    )
                )
                gal_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gal_sats, phase_residuals_dict=prt_gal_l2, num=num,
                        sys='E', len_gps=len(curr_gps_sats), freq='n2'
                    )
                )
            for sv in gal_phase_reset_svs:
                gal_arc_age[sv] = 0
            _reset_ar_float_ambiguities("E", gal_phase_reset_svs, curr_gps_sats, curr_gal_sats)
            if num < t_end:
                sigma_ion = sigma_iono_t[num]
            else:
                sigma_ion = sigma_iono_t[-1]

            ev_gps = gps_epoch['ev'].to_numpy(copy=False)
            ev_gal = gal_epoch['ev'].to_numpy(copy=False)
            inv_sin_gps = 1.0 / np.sin(np.deg2rad(ev_gps))
            inv_sin_gal = 1.0 / np.sin(np.deg2rad(ev_gal))
            sigma_code_gps = 0.3 + 0.0025 * inv_sin_gps  # **2
            sigma_phase_gps = 1e-4 + 0.0003 * inv_sin_gps  # **2
            if self.use_iono_rms:
                sigma_ion_gps = gps_epoch[
                                    'ion_rms'].to_numpy(copy=False) * sigma_ion  # sigma_ion/np.sin(np.deg2rad(ev_gps)) #np.full_like(ev_gps, sigma_ion)
                sigma_ion_gal = gal_epoch[
                                    'ion_rms'].to_numpy(copy=False) * sigma_ion  # sigma_ion/np.sin(np.deg2rad(ev_gal)) #np.full_like(ev_gal, sigma_ion)
            else:
                sigma_ion_gps = np.full_like(ev_gps, sigma_ion)
                sigma_ion_gal = np.full_like(ev_gal, sigma_ion)

            sigma_code_gal = 0.3 + 0.0025 * inv_sin_gal  # **2
            sigma_phase_gal = 1e-4 + 0.0003 * inv_sin_gal  # **2

            gps_c1_mask, prefit_c1 = self.code_screening(x=self.ekf.x[:3], satellites=gps_satellites[:, :3],
                                                         code_obs=C1, thr=10)
            gps_c2_mask, prefit_c2 = self.code_screening(x=self.ekf.x[:3], satellites=gps_satellites[:, :3],
                                                         code_obs=C2, thr=10)

            gal_c1_mask, prefit_e1 = self.code_screening(x=self.ekf.x[:3], satellites=gal_satellites[:, :3],
                                                         code_obs=EC1, thr=10)
            gal_c2_mask, prefit_e2 = self.code_screening(x=self.ekf.x[:3], satellites=gal_satellites[:, :3],
                                                         code_obs=EC2, thr=10)

            gps_sc1 = np.where(gps_c1_mask, sigma_code_gps, 1e12)
            gps_sc2 = np.where(gps_c2_mask, sigma_code_gps, 1e12)
            W_gps = np.column_stack(
                (gps_sc1, sigma_phase_gps, gps_sc2, sigma_phase_gps, sigma_ion_gps)
            ).ravel()

            gal_sc1 = np.where(gal_c1_mask, sigma_code_gal, 1e12)
            gal_sc2 = np.where(gal_c2_mask, sigma_code_gal, 1e12)
            W_gal = np.column_stack(
                (gal_sc1, sigma_phase_gal, gal_sc2, sigma_phase_gal, sigma_ion_gal)
            ).ravel()

            W = np.concatenate((W_gps, W_gal))
            W = np.nan_to_num(W, nan=1e12, posinf=1e12, neginf=1e12)

            # W = np.array(W, dtype=np.float32)
            #
            # self.ekf.R = np.diag(W)
            m = self.ekf.dim_z
            R = r_cache.get(m)
            if R is None:
                R = np.zeros((m, m), dtype=np.float64)
                r_cache[m] = R
            else:
                R.fill(0.0)

            # W musi mieć długość m
            R.flat[::m + 1] = W  # wstaw przekątną bez tworzenia diag
            self.ekf.R = R
            if not np.all(np.isfinite(Z)):
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "non-finite-observation",
                        epoch=num,
                        time=t,
                    )
                continue
            updated = False
            for k in range(5):
                try:
                    self.ekf.predict_update(
                        z=Z,
                        HJacobian=self.Hjacobian,
                        args=(gps_satellites, gal_satellites),
                        Hx=self.Hx,
                        hx_args=(gps_satellites, gal_satellites),
                    )
                    updated = True
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-6 * (10.0 ** k)
                    self.ekf.R.flat[::m + 1] += jitter
            if updated and ar_cfg.use_float_ratio_covariance:
                if ar_float_P.shape == self.ekf.P.shape:
                    P_pred_ar = self.ekf.F @ ar_float_P @ self.ekf.F.T + self.ekf.Q
                    H_ar = self.Hjacobian(self.ekf.x, gps_satellites, gal_satellites)
                    ar_float_P = _covariance_measurement_update(P_pred_ar, H_ar, self.ekf.R)
                else:
                    ar_float_P = self.ekf.P.copy()
            elif updated:
                ar_float_P = self.ekf.P.copy()
            if not updated:
                if trace_filter:
                    ppp_trace(
                        trace_filter,
                        self.__class__.__name__,
                        "singular-update-reset",
                        epoch=num,
                        time=t,
                    )
                gps_sats, gal_sats = self.reset_filter(epoch=t)
                ar_float_P = self.ekf.P.copy()
                gps_arc_age = {sv: 0 for sv in gps_sats}
                gal_arc_age = {sv: 0 for sv in gal_sats}
                continue

            for sv in curr_gps_sats:
                gps_arc_age[sv] = gps_arc_age.get(sv, 0) + 1
            for sv in curr_gal_sats:
                gal_arc_age[sv] = gal_arc_age.get(sv, 0) + 1
            gps_ar_age = np.array([gps_arc_age.get(sv, 0) for sv in curr_gps_sats], dtype=int)
            gal_ar_age = np.array([gal_arc_age.get(sv, 0) for sv in curr_gal_sats], dtype=int)

            ar_diag = None

            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                self.ekf.x, self.ekf.P, ar_diag = apply_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    base_dim=self.base_dim,
                    n_gps=len(curr_gps_sats),
                    n_gal=len(curr_gal_sats),
                    gps_ev=ev_gps,
                    gal_ev=ev_gal,
                    settings=ar_cfg,
                    gps_age=gps_ar_age,
                    gal_age=gal_ar_age,
                    ratio_P=ar_float_P if ar_cfg.use_float_ratio_covariance else None,
                )



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
            row = {'time': t, 'de': enu[0], 'dn': enu[1], 'du': enu[2],
                   'dtr': dtr, 'isb': isb, 'ztd': ztd, 'dcb_gps_c1': dcb[0], 'dcb_gps_c2': dcb[1],
                   'dcb_gal_c1': dcb[2], 'dcb_gal_c2': dcb[3], 'xr': self.ekf.x[0], 'yr': self.ekf.x[1],
                   'zr': self.ekf.x[2],
                   'ar_fixed': 0 if ar_diag is None else int(ar_diag.fixed_ambiguities),
                   'ar_ratio': np.nan if ar_diag is None or ar_diag.ratio_min is None else float(ar_diag.ratio_min),
                   'ar_ok': False if ar_diag is None else bool(ar_diag.accepted),
                   'ar_ratio_covariance': 'float' if ar_cfg.use_float_ratio_covariance else 'filtered',
                   'ar_gps_min_age': int(np.min(gps_ar_age)) if gps_ar_age.size else np.nan,
                   'ar_gal_min_age': int(np.min(gal_ar_age)) if gal_ar_age.size else np.nan}
            row.update(pppar_diagnostic_columns(ar_diag))
            result.append(row)

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

            # zapisz index (sv,time)
            gps_rows.append(gps_epoch.index)

            # zapisz dane jako ndarray (najlepiej 2D)
            gps_data.append(
                np.column_stack([
                    stec_gps,
                    n1_gps,
                    n2_gps,
                    C1, L1, C2, L2,
                    v_gps_c1, v_gps_l1,
                    v_gps_c2, v_gps_l2,
                ])
            )

            # zapisz index (sv,time)
            gal_rows.append(gal_epoch.index)

            # zapisz dane jako ndarray (najlepiej 2D)
            gal_data.append(
                np.column_stack([
                    stec_gal,
                    n1_gal,
                    n2_gal,
                    EC1, EL1, EC2, EL2,
                    v_gal_c1, v_gal_l1,
                    v_gal_c2, v_gal_l2,
                ])
            )

            if trace_filter:
                if reset_epoch:
                    ppp_trace(trace_filter, self.__class__.__name__, "reset", epoch=num, time=t)
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result[-1],
                )

        df_result = pd.DataFrame(result)
        df_result = df_result.set_index(['time'])
        # sklej index
        gps_index = gps_rows[0].append(gps_rows[1:])

        # sklej dane
        gps_array = np.vstack(gps_data)

        cols = [
            "Idelay", "n1", "n2",
            "C1", "L1", "C2", "L2",
            "vc1", "vl1", "vc2", "vl2",
        ]

        df_obs_gps = pd.DataFrame(
            gps_array,
            index=gps_index,
            columns=cols,
        )

        # sklej index
        gal_index = gal_rows[0].append(gal_rows[1:])

        # sklej dane
        gal_array = np.vstack(gal_data)

        cols = [
            "Idelay", "n1", "n2",
            "C1", "L1", "C2", "L2",
            "vc1", "vl1", "vc2", "vl2",
        ]

        df_obs_gal = pd.DataFrame(
            gal_array,
            index=gal_index,
            columns=cols,
        )

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
        return df_result, df_obs_gps, df_obs_gal, ct
