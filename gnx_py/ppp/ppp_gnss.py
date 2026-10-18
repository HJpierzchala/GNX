"""Mixed-constellation combined PPP filter classes.

This module contains ionosphere-free combined PPP implementations for multiple
constellations. The active generic class supports GPS, Galileo and BeiDou with
a reference receiver clock and inter-system-bias states. The G/E-specific
single-frequency and legacy dual-frequency classes, which the session never
selected, were removed in v1.0.0.
"""

import numpy as np
import pandas as pd
from .ppp_helpers import GNXExtendedKalmanFilter as ExtendedKalmanFilter

from ..core.conversion import ecef_to_enu
from ..core.utils import calculate_distance
from ..core.configuration import PPPConfig
from .ppp_helpers import (
    clock_priors,
    code_screening as ppp_code_screening,
    ar_phase_kwargs,
    convergence_hours,
    finish_solution,
    phase_residuals_outliers,
    epoch_process_noise,
    random_walk_variance,
    residual_frames,
    trace_epoch_summary as ppp_trace_epoch,
)
from .pppar import (
    PPPARSettings,
    advance_arc_age,
    ar_eligible_mask,
    apply_conventional_pppar,
    arc_age_array,
    combined_if_pppar_enabled,
    keep_float_state,
    pppar_diagnostic_columns,
    restore_float_state,
)
from .ppp_single import PPPDualFreqSingleGNSS as _SingleCombinedBiasModel
from ..core.gnss import SIGNALS, mode_ionosphere_free_coefficients, mode_layout, mode_signals, signal_spec
from ..models.biases import unmoved_code_variance_m2

FREQ_DICT_ALL = {name: spec.frequency_hz for name, spec in SIGNALS.items()}

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
        The AR step runs only when ``pppar_enabled`` and
        ``pppar_combined_if_ar_enabled`` are both true, and fixes no
        ambiguity: the IF ambiguity is not an integer without the wide lane
        (``pppar.apply_conventional_pppar``).

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
    #: columns the filter adds to each system's observation rows in its residual frames
    RESIDUAL_COLUMNS = (
        "system", "reference_system", "is_reference", "dtr", "ztd", "isb", "P3", "L3", "Nif",
        "prefit_code_res", "prefit_phase_res", "postfit_code_res", "postfit_phase_res",
        "postfit_code_res_demeaned", "postfit_phase_res_demeaned", "res", "code_bias_corr", "phase_bias_corr", "p_if",
    )

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
        q_amb = random_walk_variance(self.cfg.q_amb, self.interval)
        self.ekf.Q = np.diag(np.concatenate((np.zeros(self.base_dim), np.full(n_amb, q_amb))))
        self.ekf.P = np.diag(np.concatenate((np.full(self.base_dim, self.cfg.p_crd), np.full(n_amb, self.cfg.p_amb))))
        if self.pos0 is None:
            self.ekf.P[:3, :3] = np.eye(3) * 1e12
        self.ekf.Q[:3, :3] = np.eye(3) * self.cfg.q_crd  # independent random walk per axis
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.Q[3, 3] = self.cfg.q_dt
        for system, idx in self.isb_indices.items():
            # The PPPConfig preset for positioning_mode='combined' leaves p_isb unset (None):
            # mixed combined PPP estimates the ISB(s) starting from p_dt and keeps them
            # constant unless the user configured otherwise. An explicit p_isb (also 0.0)
            # is honoured. Since bug-fix stage 13b an ISB that starts with p_dt (> clock_prior_sigma_m^2)
            # takes the RTKLIB-style prior at its first epoch: the mean of its system's code residuals
            # minus the receiver clock, variance clock_prior_sigma_m^2 (StatePrior, run_filter).
            p_isb = getattr(self.cfg, "p_isb", None)
            q_isb = getattr(self.cfg, "q_isb", 0.0)
            self.ekf.P[idx, idx] = self.cfg.p_dt if p_isb is None else float(p_isb)
            self.ekf.Q[idx, idx] = 0.0 if not q_isb else random_walk_variance(q_isb, self.interval)
        if self.tro:
            self.ekf.P[self.trop_index, self.trop_index] = self.cfg.p_tro
            self.ekf.Q[self.trop_index, self.trop_index] = random_walk_variance(self.cfg.q_tro, self.interval)
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
            new_Q[idx, idx] = random_walk_variance(self.cfg.q_amb, self.interval)
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
                - group["lambda"] * self._col_np(epoch, "phw", 0.0)  # lambda_IF = c/(f1+f2), phw in cycles
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
        # NQ-SPP-33 (author decision 2026-10-15): BDS-3 codes not moved to the clock datum are down-weighted
        sigma_code = sigma_code + unmoved_code_variance_m2(self.cfg, group["system"], group["svs"])
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
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", True)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
            hold=bool(getattr(self.cfg, "pppar_fix_and_hold", False)),
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
        arc_age: dict = {}  # PPP-069: the first advance_arc_age gives 0

        ar_float_state = None
        for num, t in enumerate(epochs):
            # the previous epoch was reported fixed; without fix-and-hold the filter continues float
            restore_float_state(self.ekf, ar_float_state)
            ar_float_state = None
            reset_epoch = False
            if reset_every != 0 and ((num * self.interval) % reset_every == 0) and num != 0:
                epoch_groups = self._build_epoch_groups(t, epoch_maps)
                self._init_or_reset_filter(self._amb_keys(epoch_groups))
                old_keys = list(self.current_amb_keys)
                arc_age = {}  # PPP-069: every arc restarts at age 0
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

            # receiver clock (all code rows) and the inter-system biases that start with p_dt (their system's code
            # rows), RTKLIB udclk_ppp (bug-fix stage 13b, A1(b))
            code_rows, isb_rows, cursor = [], {}, 0
            for group in epoch_groups:
                rows = np.arange(cursor, cursor + len(group["svs"]))
                code_rows.append(rows)
                if group["system"] in self.isb_indices:
                    isb_rows[self.isb_indices[group["system"]]] = rows
                cursor += 2 * len(group["svs"])
            self.ekf.state_priors = clock_priors(self.cfg.clock_prior_sigma_m, np.concatenate(code_rows), isb_rows)
            with epoch_process_noise(self.ekf, t, self.interval):
                self.ekf.predict_update(z=z, HJacobian=self.HJacobian, args=epoch_groups, Hx=self.Hx,
                                        hx_args=epoch_groups)
            arc_age = advance_arc_age(arc_age, curr_keys, phase_reset_keys)
            age_by_key = dict(arc_age)
            postfit = z - self.Hx(self.ekf.x, epoch_groups)
            innovation = np.asarray(self.ekf.y, dtype=float)

            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs and self.systems == ["G", "E"]:
                g_group = next(group for group in epoch_groups if group["system"] == "G")
                e_group = next(group for group in epoch_groups if group["system"] == "E")
                # float phase residuals and weights of the IF phase rows (contract C3, NQ-PPP-20)
                phase_rows, amb_idx, z_cursor, amb_cursor = [], [], 0, self.base_dim
                for group in epoch_groups:
                    n = len(group["svs"])
                    phase_rows.append(z_cursor + n + np.arange(n))
                    amb_idx.append(amb_cursor + np.arange(n))
                    z_cursor += 2 * n
                    amb_cursor += n
                ar_inputs = ar_phase_kwargs(postfit, np.diag(self.ekf.R),
                                            np.concatenate(phase_rows), np.concatenate(amb_idx))
                ar_float_state = keep_float_state(self.ekf, ar_cfg)
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
                    gps_eligible=ar_eligible_mask(g_group.get("epoch")),
                    gal_eligible=ar_eligible_mask(e_group.get("epoch")),
                    **ar_inputs,
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

        df_result = finish_solution(pd.concat(result) if result else None, conv_time)
        residuals = residual_frames(residuals_by_system, self.obs_by_system, self.RESIDUAL_COLUMNS)
        return df_result, residuals, None, convergence_hours(conv_time)
