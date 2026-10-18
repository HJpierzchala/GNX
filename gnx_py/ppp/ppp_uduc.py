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
from .ppp_helpers import GNXExtendedKalmanFilter as ExtendedKalmanFilter
from typing import Sequence

from ..core.conversion import ecef_to_enu
from ..core.utils import calculate_distance
from ..core.configuration import PPPConfig
from ..models.biases import (
    bias_column_m,
    has_satellite_osb,
    osb_m,
    split_dual_code_dsb_corrections_m,
    unmoved_code_variance_m2,
)
from ..core.gnss import MODE_SIGNALS as GNSS_MODE_SIGNALS, SIGNALS, frequency_hz
from ..core import _notes
# private aliases: ``gnx_py.ppp`` star-imports this module (bug-fix stage 13d, D6)
from pathlib import Path as _Path
from ..models.biases import NS_TO_M as _NS_TO_M
from ..core.gnss import code_signal as _code_signal
from ..ionosphere.ionex import IONEX_DEFAULT_DCB_REFERENCE as _IONEX_DEFAULT_DCB_REFERENCE
from ..ionosphere.ionex import GIMReader as _GIMReader
from .ppp_helpers import (
    ar_phase_kwargs,
    clock_priors,
    convergence_hours,
    finish_solution,
    phase_residuals_outliers,
    epoch_process_noise,
    gap_scaled_process_noise,
    random_walk_variance,
    reset_prior_states,
    residual_frames,
    trace_epoch_summary as ppp_trace_epoch,
    trace_message as ppp_trace,
    used_rows,
)
from .pppar import (
    PPPARSettings,
    advance_arc_age,
    ar_eligible_mask,
    apply_indexed_uncombined_pppar,
    apply_uncombined_pppar,
    arc_age_array,
    float_position,
    keep_float_state,
    pppar_diagnostic_columns,
    restore_float_state,
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
    """Apply a covariance-only EKF measurement update for AR ratio covariance.

    Rows with ``R >= REJECTED_R`` are left out, as in the filter's own update (A1(a))."""
    keep = used_rows(R)
    if keep.size != H.shape[0]:
        H, R = H[keep], R[np.ix_(keep, keep)]
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


def _init_coordinate_states(ekf, cfg) -> None:
    """Initial covariance and process noise of the receiver coordinates (states 0-2).

    ``p_crd`` and ``q_crd`` are per-axis variances [m^2]: the three axes start and
    move independently (``q_crd`` per epoch, 0.0 static, > 0 kinematic random walk),
    as in the combined filters.
    """
    ekf.P[:3, :3] = np.eye(3) * float(cfg.p_crd)
    ekf.Q[:3, :3] = np.eye(3) * float(cfg.q_crd)


def _reset_covariance_states(P: np.ndarray, indices: Sequence[int], variance: float) -> None:
    """Reset covariance rows/columns for selected state indices in place."""
    for idx in indices:
        if idx < 0 or idx >= P.shape[0]:
            continue
        P[idx, :] = 0.0
        P[:, idx] = 0.0
        P[idx, idx] = variance


def _carry_over_states(
    x_old: np.ndarray,
    P_old: np.ndarray,
    Q_old: np.ndarray,
    x_new: np.ndarray,
    P_new: np.ndarray,
    Q_new: np.ndarray,
    old_idx: Sequence[int],
    new_idx: Sequence[int],
) -> None:
    """Copy retained states with their full joint covariance block, in place.

    On a satellite-set change a filter keeps the receiver states, the states of
    the satellites still tracked and any receiver bias states. ``old_idx`` and
    ``new_idx`` list all of them in matching order; copying their covariance
    block in one go keeps every correlation between them (satellite-satellite,
    satellite-receiver, bias-anything). Copying it block by block drops some of
    those correlations and leaves an inconsistent covariance matrix.
    """
    old_idx = np.asarray(old_idx, dtype=int)
    new_idx = np.asarray(new_idx, dtype=int)
    if old_idx.size == 0:
        return
    x_new[new_idx] = x_old[old_idx]
    P_new[np.ix_(new_idx, new_idx)] = P_old[np.ix_(old_idx, old_idx)]
    Q_new[np.ix_(new_idx, new_idx)] = Q_old[np.ix_(old_idx, old_idx)]


def _ramp_plus_ion_rms(ramp, ion_rms, fallback: float) -> np.ndarray:
    """Variance [m^2] of the ionospheric pseudo-observation with the model error: ``ramp + ion_rms**2``.

    ``ramp``: the constraint variance of the ``sigma_iono_0`` -> ``sigma_iono_end`` ramp [m^2] (scalar or one per
    satellite); ``ion_rms``: the model error [m] (GIM RMS map, or the modelled error of a broadcast model); a
    non-finite ``ion_rms`` takes ``fallback`` [m^2]. Variances of independent errors add, as in RTKLIB, which sums the
    model variances into the observation variance (ppp.c ``res_ppp``; ``iontec``: the RMS map squared, broadcast
    ``SQR(ERR_BRDCI * ion)``). Shared by ``_iono_constraint_sigma`` (R7, R2, single frequency) and R4c (before v1.0.0 it was ``ion_rms * ramp``, a metre times a variance).
    """
    ion_rms = np.asarray(ion_rms, dtype=float)
    finite = np.isfinite(ion_rms)
    return np.where(finite, ramp + np.square(np.where(finite, ion_rms, 0.0)), float(fallback))


def _iono_constraint_sigma(
    epoch: pd.DataFrame,
    num,
    interval: float,
    use_iono_rms: bool,
    sigma_iono_0: float,
    sigma_iono_end: float,
    t_end: int,
) -> np.ndarray:
    """Return the per-satellite variance [m^2] of the ionospheric pseudo-observation for one epoch.

    The ramp goes from ``sigma_iono_0`` to ``sigma_iono_end`` over ``t_end``
    minutes/epochs; its values are variances [m^2]. With ``use_iono_rms`` and an
    ``ion_rms`` column [m] (the GIM RMS map, or the modelled broadcast error) the
    variance is ``ramp + ion_rms**2``, the model error added to the ramp as
    RTKLIB adds the variance of its ionosphere model (``iontec``: RMS map
    squared; broadcast ``SQR(ERR_BRDCI * ion)``). Before v1.0.0
    it was ``ion_rms * ramp`` (a metre times a variance), 2-3x
    tighter at BRUX and slower: C-COD static UCi t_c10 224.5 min.
    ``num`` is the epoch count since the start (one ramp for all satellites) or
    an array with one count per satellite of ``epoch`` (one ramp per arc from
    its own start). A non-finite ``ion_rms`` takes ``sigma_iono_end``.
    """
    interval = max(float(interval), 1e-12)
    sigma_iono_0 = 1.1 if sigma_iono_0 is None else sigma_iono_0
    sigma_iono_end = 3.0 if sigma_iono_end is None else sigma_iono_end
    t_end = 30 if t_end is None else t_end
    t_end_epochs = max(float(t_end) / interval, 1.0)
    if np.ndim(num):
        # per-satellite epoch counts (the age of each arc's ionosphere state, PPPUducSFSingleGNSS)
        frac = np.minimum(np.asarray(num, dtype=float), t_end_epochs) / t_end_epochs
    else:
        frac = min(float(num), t_end_epochs) / t_end_epochs
    sigma = float(sigma_iono_0) + (float(sigma_iono_end) - float(sigma_iono_0)) * frac

    if use_iono_rms and 'ion_rms' in epoch.columns:
        ion_sigma = _ramp_plus_ion_rms(sigma, epoch['ion_rms'].to_numpy(copy=False), sigma_iono_end)
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


def _uncombined_obs_variances(cfg, ev_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Code and phase variances [m^2] of the uncombined filters without the legacy weighting.

    ``uncombined_code_var`` and ``uncombined_phase_var`` are the zenith variances of one
    undifferenced observation, scaled by 1/sin(elevation).
    """
    scale = _elevation_variance_scale(ev_deg)
    return (float(getattr(cfg, "uncombined_code_var", 1.0)) * scale,
            float(getattr(cfg, "uncombined_phase_var", 1e-4)) * scale)


def _code_zenith_var_by_system(cfg, system: str) -> float:
    """Zenith code variance [m^2] of ``system`` from ``uncombined_iono_code_var_by_system`` (default G 0.3, E 0.3,
    C 1.0 m^2, ``PPPConfig.CODE_VAR_BY_SYSTEM_DEFAULT``): the code weight of every route with ionospheric constraints,
    dual-frequency (``_constrained_obs_variances``) and single-frequency (``PPPUducSFSingleGNSS``)."""
    by_system = getattr(cfg, "uncombined_iono_code_var_by_system", None) or PPPConfig.CODE_VAR_BY_SYSTEM_DEFAULT
    given = by_system.get(system, PPPConfig.CODE_VAR_BY_SYSTEM_DEFAULT[system])
    try:
        code_zenith = float(given)
    except (TypeError, ValueError):
        code_zenith = float("nan")
    if not (np.isfinite(code_zenith) and code_zenith > 0.0):
        # assignment and set_param validate the setting; an in-place edit of the dict does not
        raise ValueError(f"uncombined_iono_code_var_by_system[{system!r}] must be a finite variance > 0 m^2, "
                         f"got {given!r}")
    return code_zenith


def _constrained_obs_variances(cfg, ev_deg: np.ndarray, system: str) -> tuple[np.ndarray, np.ndarray]:
    """Code and phase variances [m^2] of the filters with ionospheric constraints (R7, R2) for ``system``.

    Default (``uncombined_iono_legacy_weighting=False``): code
    ``c_sys / sin(elevation)`` with ``c_sys`` from ``uncombined_iono_code_var_by_system``
    (default G 0.3, E 0.3, C 1.0 m^2, ``PPPConfig.CODE_VAR_BY_SYSTEM_DEFAULT``), phase
    ``uncombined_phase_var / sin(elevation)`` (1e-4 m^2).
    ``uncombined_iono_legacy_weighting=True`` (the default before v1.0.0, kept to reproduce old results):
    ``base + elev_coeff / sin(elevation)`` from the ``uncombined_iono_*_var_*`` settings, i.e. code
    0.3 + 0.0025/sin(E) m^2, practically independent of the elevation (0.3025 m^2 at 90 deg, 0.3144 at
    10 deg) and 3.3-18x stronger than the unconstrained routes' 1/sin(E). With the slant ionosphere pinned by
    the GIM every code is direct geometry information, so low-elevation code errors and product code-datum
    errors (COD C06 +4.9 m, a BeiDou satellite without DSB) entered the float ambiguities with zenith weight:
    C/EC/GC/GEC COD static t_c10 157.5/130/65.5/48.5 min, G GRG kinematic t_c20 103.5 min (38.5/29/20/18
    and 10.5 min with this weighting). One value cannot serve every system: 1.0 for all slows Galileo
    (E GRG static 13 -> 44 min), 0.3 for all leaves BeiDou slow (C COD 58.5 min); 1.0 m^2 for BeiDou allows for
    the elevation-dependent satellite-induced code variations of BDS-2 (Wanninger and Beer 2015).
    """
    if not bool(getattr(cfg, "uncombined_iono_legacy_weighting", False)):
        scale = _elevation_variance_scale(ev_deg)
        return (_code_zenith_var_by_system(cfg, system) * scale,
                float(getattr(cfg, "uncombined_phase_var", 1e-4)) * scale)
    inv_sin = _elevation_variance_scale(ev_deg)
    code_var = (float(getattr(cfg, "uncombined_iono_code_var_base", 0.3))
                + float(getattr(cfg, "uncombined_iono_code_var_elev_coeff", 0.0025)) * inv_sin)
    phase_var = (float(getattr(cfg, "uncombined_iono_phase_var_base", 1e-4))
                 + float(getattr(cfg, "uncombined_iono_phase_var_elev_coeff", 0.0003)) * inv_sin)
    return code_var, phase_var


def _sat_bias_ns(obs: pd.DataFrame, column: str) -> np.ndarray | None:
    """Satellite bias column [ns] with missing values kept as NaN; None without the column or a value."""
    if column not in obs.columns:
        return None
    values = obs[column].to_numpy(dtype=float)
    return values if np.isfinite(values).any() else None


def _sat_osb_pair_ns(obs: pd.DataFrame, code_1: str, code_2: str) -> np.ndarray | None:
    """OSB(code_1) - OSB(code_2) [ns] when the product gives both satellite OSBs (``has_satellite_osb``), else None."""
    if not has_satellite_osb(obs, (code_1, code_2)):
        return None
    return obs[f"OSB_{code_1}"].to_numpy(dtype=float) - obs[f"OSB_{code_2}"].to_numpy(dtype=float)


def _sat_dsb_pair_ns(obs: pd.DataFrame, code_1: str, code_2: str) -> np.ndarray | None:
    """Satellite DSB code_1 - code_2 [ns] of a DSB/DCB product (``BIAS_`` columns, direct or reversed), else None."""
    direct = _sat_bias_ns(obs, f"BIAS_{code_1}_{code_2}")
    reverse = _sat_bias_ns(obs, f"BIAS_{code_2}_{code_1}")
    if direct is None:
        return None if reverse is None else -reverse
    if reverse is not None:
        direct = np.where(np.isfinite(direct), direct, -reverse)
    return direct


def _same_signals(system: str, pair_a, pair_b) -> bool:
    try:
        return tuple(_code_signal(system, c) for c in pair_a) == tuple(_code_signal(system, c) for c in pair_b)
    except ValueError:
        return False


def _gim_dcb_pair(cfg, system: str):
    """Code pair the satellite DCBs of the configured IONEX file refer to (stated, else the IONEX default)."""
    path = getattr(cfg, "gim_path", None)
    try:
        if path and _Path(path).is_file():
            return _GIMReader.read_dcb_reference(path).codes(system)
    except (OSError, ValueError):
        pass
    return _IONEX_DEFAULT_DCB_REFERENCE.get(system)


#: Smallest satellite DSB difference [ns] between the bias product and the DSB the ionosphere model is consistent with
#: that ``_align_iono_model_to_code_biases`` corrects. Two DSB estimates of the same satellite agree to about
#: 0.3-0.5 ns (BRUX 2024-035: GRG OSB against the COD IONEX DCBs, Galileo 0.41 ns rms, max 0.85 ns): a smaller
#: difference is the noise of the two estimates, and correcting it trades the product's DSB error for the
#: reference's (Galileo GRG kinematic UCi RMS 0.096 -> 0.161 m with every satellite corrected, the per-satellite
#: GIM-minus-state offsets uncorrelated with the corrections, r = -0.09). The systematic differences are larger:
#: GPS III C1W-C2W 2.1-2.9 ns, BeiDou-2 IGSO and BeiDou-3 MEO C2I-C6I 1.6-2.7 ns (against CAS), and they explain the
#: offsets (r = -0.89 GPS, -0.94 BeiDou; rms 0.59 -> 0.27 m, 0.93 -> 0.31 m on L1 / B1I).
_IONO_DSB_ALIGN_MIN_NS = 1.0


def _align_iono_model_to_code_biases(obs: pd.DataFrame, mode: str, system: str, cfg) -> tuple[pd.DataFrame, dict]:
    """Refer the model ionosphere ``ion`` of a constrained uncombined route to the satellite code-bias datum of
    the filter.

    With per-signal code biases from a product (OSB, or a DSB applied to both codes) the slant ionosphere state of
    a satellite on the first signal ``f1`` of the pair is ``I + beta * (D_used - D_true)``, ``beta = f2^2 / (f1^2 -
    f2^2)``, ``D = b(code_1) - b(code_2)`` the satellite DSB: the bias product's DSB ``D_used`` differs from the
    satellite's true DSB and the state takes the difference. A GIM gives ``I`` in the datum of the satellite DCBs
    estimated with it. The constraint ``ion`` must be in the datum of the state, else the per-satellite
    difference (BRUX 2024-035, GRG OSB: against the COD GIM DCBs GPS 0.68 m rms, up to 1.2 m on L1 for the GPS III
    satellites, Galileo 0.19 m rms; BeiDou C2I-C6I against the CAS DSB 0.93 m rms, up to 1.6 m on B1I) is a
    constant error of the pseudo-observation of that satellite, which the filter, treating it as white noise every
    epoch, takes into the ionosphere states and, in kinematic mode, into the position. So ``ion`` gets
    ``beta * ((D_used - D_ref) - median)`` per satellite, ``D_ref`` the DSB the model is consistent with:

    1. ``ionosphere_model='gim'``: the satellite DCBs of the IONEX file (column ``bias`` [ns]) when they refer to
       the two signals of the pair (``Reference observables`` of the file, else the IONEX default per system;
       Galileo E5a and E5b are not taken for each other); with OSB code biases ``D_used`` is taken on the IONEX
       codes (e.g. GPS C1W-C2W, Galileo C1X-C5X) when the product gives their OSBs, so the product's
       inter-tracking biases carry over;
    2. else, with OSB code biases, a DSB/DCB product of the pair in ``dcb_path`` (``BIAS_`` columns, e.g. CAS
       DCB), for any constraint model.

    ``median`` (over the satellites of the system with both values) is the datum difference of the two DSB sets, a
    common term the receiver code-bias states take. Only a difference of at least ``_IONO_DSB_ALIGN_MIN_NS`` (1 ns)
    from it is corrected: smaller ones are within the agreement of two DSB estimates (their noise, not a datum
    difference); such a satellite, and one without a value, keeps ``ion`` (shift 0, i.e. the median). Without
    code biases from a product or without a reference DSB the frame is returned unchanged; a dual-frequency
    GIM-constrained route with product code biases and no reference warns once per run and system (add a DSB
    product of the pair to ``dcb_path``). A single-frequency mode takes
    :func:`_align_sf_iono_model_to_code_biases`. Returns the frame (a copy when changed) and a summary: ``source``
    ('ionex', 'dsb' or None), ``pair``, ``reference_pair``, ``n_satellites`` corrected, ``unaligned`` (satellites
    without a reference value), ``within_agreement`` (difference below the threshold), ``shift_m`` per corrected
    satellite.
    """
    info = {"system": system, "source": None, "pair": None, "reference_pair": None, "n_satellites": 0,
            "unaligned": (), "within_agreement": (), "shift_m": {}}
    signals = MODE_SIGNALS.get(mode, ())
    if obs is None or obs.empty or "ion" not in obs.columns or len(signals) not in (1, 2):
        return obs, info
    if len(signals) == 1:
        return _align_sf_iono_model_to_code_biases(obs, signals[0], system, cfg, info)
    try:
        code_1, code_2 = (_first_prefixed_column(obs, SIGNAL_COLUMNS[s][0]) for s in signals)
    except (IndexError, KeyError):
        return obs, info
    info["pair"] = (code_1, code_2)
    used = _sat_osb_pair_ns(obs, code_1, code_2)
    kind = "osb" if used is not None else None
    if used is None:
        used = _sat_dsb_pair_ns(obs, code_1, code_2)
        kind = "dsb" if used is not None else None
    if used is None:
        return obs, info
    model = getattr(cfg, "ionosphere_model", None)
    diff = None
    if model == "gim" and _sat_bias_ns(obs, "bias") is not None:
        ref_pair = _gim_dcb_pair(cfg, system)
        if ref_pair is not None and _same_signals(system, ref_pair, (code_1, code_2)):
            ref_used = _sat_osb_pair_ns(obs, *ref_pair) if kind == "osb" else None
            diff = (used if ref_used is None else ref_used) - obs["bias"].to_numpy(dtype=float)
            info.update(source="ionex", reference_pair=tuple(ref_pair))
    if diff is None and kind == "osb":
        dsb = _sat_dsb_pair_ns(obs, code_1, code_2)
        if dsb is not None:
            diff = used - dsb
            info.update(source="dsb", reference_pair=(code_1, code_2))
    if diff is None:
        if model == "gim" and kind == "osb":
            # (code biases from a DSB product: the state is in that product's datum, estimated with a GIM or of
            # its own; nothing to align it with)
            _notes.warn_once(
                cfg, ("ppp_iono_constraint_dsb", system, code_1, code_2),
                f"GIM ionospheric constraint of {system} ({code_1}-{code_2}) is not referred to the satellite "
                f"code-bias datum of the bias product: neither the IONEX DCBs nor a DSB product in dcb_path give "
                f"the satellite DSB {code_1}-{code_2}, so the difference between the product's DSB and the DSB "
                f"the GIM is consistent with (up to ~2 m of slant ionosphere per satellite) stays in the "
                f"constraint; add a DSB product of the pair (e.g. CAS DCB) to dcb_path", UserWarning, stacklevel=3)
        return obs, info
    f1, f2 = (frequency_hz(s) for s in signals)
    return _shift_iono_model(obs, diff, f2 ** 2 / (f1 ** 2 - f2 ** 2), info)


def _shift_iono_model(obs: pd.DataFrame, diff: np.ndarray, factor: float, info: dict) -> tuple[pd.DataFrame, dict]:
    """``ion`` + ``factor * ((diff - median) [ns] -> m)`` per satellite whose ``diff - median`` is at least
    ``_IONO_DSB_ALIGN_MIN_NS``; ``diff`` = used minus reference satellite DSB [ns] per row; ``info`` updated."""
    if "sv" in obs.index.names:
        sv = obs.index.get_level_values("sv").astype(str)
    elif "sv" in obs.columns:
        sv = obs["sv"].astype(str)
    else:
        sv = pd.Index(["?"] * len(obs))
    # per satellite, not per arc ('G04_0_1' -> 'G04'): the bias is the satellite's, the mean over satellites
    prn = np.asarray(pd.Index(sv).str.split("_").str[0])
    per_sat = pd.Series(diff, index=prn).groupby(level=0).mean()
    have = per_sat.dropna()
    if have.empty:
        return obs, info
    dev = have - float(have.median())
    aligned = dev[dev.abs() >= _IONO_DSB_ALIGN_MIN_NS]
    info.update(unaligned=tuple(sorted(set(per_sat.index) - set(have.index))),
                within_agreement=tuple(sorted(set(have.index) - set(aligned.index))))
    if aligned.empty:
        return obs, info
    shift = pd.Series(prn).map(aligned * (factor * _NS_TO_M)).to_numpy(dtype=float)
    obs = obs.copy()
    obs["ion"] = obs["ion"].to_numpy(dtype=float) + np.where(np.isfinite(shift), shift, 0.0)
    info.update(n_satellites=int(aligned.size),
                shift_m={k: float(factor * v * _NS_TO_M) for k, v in aligned.items()})
    return obs, info


def _align_sf_iono_model_to_code_biases(obs: pd.DataFrame, signal: str, system: str, cfg,
                                        info: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The GIM ionosphere ``ion`` of a single-frequency code observable in the satellite code-bias datum of the
    product (the rule of :func:`_align_iono_model_to_code_biases`
    for one signal). Used by SPP and by the single-frequency constrained PPP routes.

    A GIM and its satellite DCBs ``D_GIM`` (IONEX ``PRN / BIAS / RMS``, column ``bias`` [ns]) are one consistent
    pair: the map gives ``I + beta (D_GIM - D_true)`` on the first signal of the DCB pair, ``beta = f2^2 / (f1^2 -
    f2^2)``. A code corrected with the product's bias (``OSB = -beta D_used`` on the first signal of an
    IF-consistent product, ``-alpha D_used`` on the second, ``alpha = f1^2 / (f1^2 - f2^2)``) and with that map
    keeps ``(f1 / f_s)^2 beta (D_used - D_GIM)`` on signal ``s``, whatever ``D_true``: so ``ion`` gets that term
    per satellite (``D_used`` the product's OSB pair on the IONEX codes, else its DSB of those codes), with the
    median and the 1-ns threshold of the dual-frequency rule. RTKLIB 2.4.2 takes the IONEX DCBs for
    single-frequency positioning (``pntpos.c`` ``prange``: ``P1_P2`` from the IONEX file, the broadcast TGD only
    where it gives none). BRUX 2024-035, GRG OSB and the COD GIM: the six GPS III satellites differ by 2.7-3.5 ns
    (1.25-1.60 m on L1), every other GPS satellite by at most 0.51 ns; GPS L1 SPP rms 3D 1.444 -> 0.955 m.

    Only with ``ionosphere_model='gim'``, IONEX DCBs of the system and a signal on one of the two frequencies of
    the IONEX code pair (GPS C1W-C2W: L1, L2; Galileo C1C/C1X-C5Q/C5X: E1, E5a); otherwise (e.g. BeiDou with the
    CODE / IGS GIMs, which carry no BeiDou DCBs; GPS L5; Galileo E5b) the frame is returned unchanged. Not warned:
    without a reference the product's DSB is applied as given (the behaviour before v1.0.0).
    """
    if info is None:
        info = {"system": system, "source": None, "pair": None, "reference_pair": None, "n_satellites": 0,
                "unaligned": (), "within_agreement": (), "shift_m": {}}
    if (obs is None or obs.empty or "ion" not in obs.columns or getattr(cfg, "ionosphere_model", None) != "gim"
            or _sat_bias_ns(obs, "bias") is None):
        return obs, info
    ref_pair = _gim_dcb_pair(cfg, system)
    if ref_pair is None:
        return obs, info
    try:
        pair_signals = tuple(_code_signal(system, c) for c in ref_pair)
    except ValueError:
        return obs, info
    if signal not in pair_signals:
        return obs, info
    code_1, code_2 = ref_pair
    used = _sat_osb_pair_ns(obs, code_1, code_2)
    if used is None:
        used = _sat_dsb_pair_ns(obs, code_1, code_2)
    if used is None:
        return obs, info
    info.update(pair=(code_1, code_2), source="ionex", reference_pair=(code_1, code_2))
    f1, f2 = (frequency_hz(s) for s in pair_signals)
    factor = f2 ** 2 / (f1 ** 2 - f2 ** 2) * (f1 / frequency_hz(signal)) ** 2
    return _shift_iono_model(obs, used - obs["bias"].to_numpy(dtype=float), factor, info)


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
        ``_iono_constraint_sigma``. Code variance ``c_sys / sin(elevation)``
        with ``c_sys`` from ``uncombined_iono_code_var_by_system`` (default
        G 0.3, E 0.3, C 1.0 m^2), the setting of the dual-frequency
        constrained routes (before, 1.0 m^2 for
        every system, which ``{"G": 1.0, "E": 1.0, "C": 1.0}`` still gives);
        phase variance ``0.001 / sin(elevation)`` m^2.

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
        ionospheric constraint. With one frequency the GIM error cannot be
        separated from the position, so by default
        (``SF_SIGMA_IONO_END_DEFAULT``) the GIM acts as a prior on the slant
        ionosphere of the satellites in view at the start and is practically
        free after the first epochs; the solution then rests on the
        ionosphere-free code/phase information, as the unconstrained route
        (``PPPSingleFreqSingleGNSS``, GRAPHIC) does. On the validation data this
        is better than GRAPHIC in the first 15 min in some cases (Galileo
        especially) but not on the median, and worse after 15-60 min.
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
        sigma_iono_end=None,
        t_end=30,
    ):
        self.cfg = config
        self.mode = mode
        # the GIM prior in the satellite code-bias datum of the product (bug-fix stage 13d integration review,
        # NQ-SPP-34; dual-frequency routes since D6): the state absorbs the product's DSB, the prior must carry it
        self.iono_dsb_alignment = {}
        if isinstance(obs, pd.DataFrame) and "ion" in obs.columns and len(_mode_signals(mode)) == 1:
            system = SIGNALS[_mode_signals(mode)[0]].system
            obs, self.iono_dsb_alignment[system] = _align_iono_model_to_code_biases(obs, mode, system, config)
        self.obs = obs
        self.tro = tro
        self.ekf = ekf
        self.FREQ_DICT = dict(FREQ_DICT_ALL)
        self.CLIGHT = 299792458
        self.base_dim = 5 if tro else 4  # X, Y, Z, clk, ZTD(optional)
        self.pos0 = pos0
        self.interval = interval
        self.use_iono_rms = use_iono_rms
        self.sigma_iono_0 = sigma_iono_0
        # None (the PPPConfig default) takes the default of this route: a GIM prior at the start, practically
        # free after the first epochs (bug-fix stage 8b, NQ-PPP-23; see SF_SIGMA_IONO_END_DEFAULT)
        self.sigma_iono_end = self.SF_SIGMA_IONO_END_DEFAULT if sigma_iono_end is None else sigma_iono_end
        self.t_end = t_end

    #: route default of ``sigma_iono_end`` (a variance [m^2], like the whole ramp). The ramp is linear in the
    #: variance over ``t_end`` minutes, so with 1e4 (30 s data, t_end 30 min) it is 1.1 m^2 at the first epoch,
    #: about 170 m^2 at the second and 1000 m^2 after 3 min: in practice the GIM is a prior on the slant
    #: ionosphere of the satellites in view at the start; satellites that rise later and new arcs get
    #: practically no GIM information, and the solution rests on the ionosphere-free code/phase information
    #: of the arcs (as GRAPHIC). With one frequency nothing separates the GIM error (1-3 TECU,
    #: time-correlated and elevation-dependent, so it does not average out) from the position: with the
    #: generic 3.0 the GIM error stayed in the solution (12 validation cases, 6 h static: median 0.17-0.27 m
    #: final, 0.76-1.07 m after 1 h; kinematic RMS 0.56-1.05 m) against 0.08-0.11 m, 0.25-0.42 m and
    #: 0.27-0.40 m with 1e4; 10, 30 and 100 converged later than 1e4 (BRUX L1, t_c10 184-261 against 87 min).
    #: Against GRAPHIC: better in the first 15 min in 3-5 of the 12 cases per signal (BRUX 2024-035 E1 RMS
    #: 0.38 against 1.23 m), worse on the median (L1/E1/B1I 1.87/1.84/1.60 against 1.33/0.92/1.36 m) and
    #: after 1 h (0.25/0.42/0.42 against 0.19/0.15/0.17 m).
    SF_SIGMA_IONO_END_DEFAULT = 1e4
    #: GIM prior for every new arc (OFF by default). True: an arc that
    #: starts after the first epoch (a rising satellite, a new arc after a slip or a gap) gets its slant
    #: ionosphere state from the GIM ``ion`` value and the constraint ramp from its own start (``sigma_iono_0`` at
    #: its first epoch, then the same ramp to ``sigma_iono_end`` over ``t_end``), as the arcs of the first epoch.
    #: False (default): one ramp from the first epoch of the run, a new state starts at 0 m with ``p_iono``.
    #: Not neutral on the validation data (SP3grg, 6 h, float; off -> on, median of 12 static cases BRUX/NKLG/NYA2
    #: x 2024-035/132 x 00/12 h, L1 / E1 / B1I): RMS 0-15 min 1.52 -> 1.32 / 1.84 -> 1.84 / 1.60 -> 1.48 m, after
    #: 1 h 0.248 -> 0.269 / 0.422 -> 0.341 / 0.414 -> 0.420 m, final 0.090 -> 0.083 / 0.105 -> 0.100 / 0.071 ->
    #: 0.068 m, but the worst final 0.266 -> 0.203 / 0.129 -> 0.182 / 0.201 -> 0.254 m: final better in 17 of the
    #: 24 BRUX/NKLG cases, worse in all 12 NYA2 cases (polar GIM; e.g. 2024-035 00 h L1 0.154 -> 0.195 m);
    #: kinematic 2024-035 RMS from 60 min, median of 3 stations 0.270 -> 0.364 / 0.329 -> 0.384 / 0.395 ->
    #: 0.264 m. Kept off (on only if it helps or is neutral).
    SF_GIM_PRIOR_NEW_ARCS = False
    #: columns the filter adds to the observation rows in its residual frame (before v1.0.0
    #: the route returned its input observations instead of residuals)
    RESIDUAL_COLUMNS = (
        "system", "dtr", "ztd", "prefit_code_res", "prefit_phase_res", "prefit_iono_res", "postfit_code_res",
        "postfit_phase_res", "postfit_iono_res", "code_used", "I_state", "N_state",
    )

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
            Q_new[idx_i, idx_i] = random_walk_variance(self.cfg.q_iono, self.interval)
            P_new[idx_n, idx_n] = self.cfg.p_amb
            Q_new[idx_n, idx_n] = random_walk_variance(self.cfg.q_amb, self.interval)  # PPP-064
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
            self.ekf.Q[4, 4] = random_walk_variance(self.cfg.q_tro, self.interval)
        _init_coordinate_states(self.ekf, self.cfg)
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
            self.ekf.Q[idx_i, idx_i] = random_walk_variance(self.cfg.q_iono, self.interval)
            self.ekf.P[idx_n, idx_n] = self.cfg.p_amb
            self.ekf.Q[idx_n, idx_n] = random_walk_variance(self.cfg.q_amb, self.interval)  # PPP-064
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
        residual_rows = []
        system = SIGNALS[_mode_signals(self.mode)[0]].system
        code_zenith = _code_zenith_var_by_system(self.cfg, system)       # NQ-PPP-47 [m^2 at zenith]
        self.obs = self.obs.sort_values(by='sv')
        obs_epochs = {t: df for t, df in self.obs.groupby(level=1, sort=False)}
        osb_c_col = f'OSB_{c_col}'
        osb_l_col = f'OSB_{l_col}'
        has_osb_c = osb_c_col in self.obs.columns
        has_osb_l = osb_l_col in self.obs.columns
        r_cache = {}
        xyz = None
        conv_xyz = None  # previous float position (convergence check)
        conv_time = None
        T0 = all_epochs[0]
        arc_age: dict = {}  # PPP-069: the first advance_arc_age gives 0
        # GIM prior for every new arc (SF_GIM_PRIOR_NEW_ARCS): epoch number at which each ionosphere state started
        gim_new_arcs = bool(self.SF_GIM_PRIOR_NEW_ARCS)
        arc_start = {sv: 0 for sv in old_sats}
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
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", True)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
            hold=bool(getattr(self.cfg, "pppar_fix_and_hold", False)),
        )
        ar_float_state = None
        for num, t in enumerate(all_epochs):
            # the previous epoch was reported fixed; without fix-and-hold the filter continues float
            restore_float_state(self.ekf, ar_float_state)
            ar_float_state = None
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                    arc_age = {}  # PPP-069: every arc restarts at age 0
                    # the reset re-initialises every ionosphere state from the GIM: each ramp starts again
                    arc_start = {sv: num for sv in old_sats}
                    reset_epoch = True
                    T0 = t
            epoch = obs_epochs.get(t)
            if epoch is None:
                continue
            curr_sats = epoch.index.get_level_values('sv').tolist()
            fresh = []
            if gim_new_arcs and curr_sats != old_sats:
                prev = set(old_sats)
                fresh = [j for j, sv in enumerate(curr_sats) if sv not in prev]
                arc_start = {sv: (arc_start.get(sv, num) if sv in prev else num) for sv in curr_sats}
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
            if fresh:
                # new arcs start from the GIM slant ionosphere, as the arcs of the first epoch (_init_filter_common)
                j = np.asarray(fresh, dtype=int)
                ok = np.isfinite(I_obs[j])
                self.ekf.x[self.base_dim + 2 * j[ok]] = I_obs[j[ok]]
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
            # NQ-SPP-33 (author decision 2026-10-15): BDS-3 codes not moved to the clock datum are down-weighted
            extra = unmoved_code_variance_m2(self.cfg, system, epoch, uncombined=True)
            # NQ-PPP-47 (bug-fix stage 13d): code c_sys / sin(E) of uncombined_iono_code_var_by_system, as on the
            # dual-frequency constrained routes; the down-weighting term of NQ-SPP-33 comes on top
            r_diag[0::3] = np.where(code_mask, code_zenith * inv_sin + extra, 1e12)
            r_diag[1::3] = 0.001 * inv_sin
            # constraint ramp over t_end minutes, scaled by ion_rms when used: as the dual-frequency filters;
            # with SF_GIM_PRIOR_NEW_ARCS one ramp per arc from its own start
            ramp_num = (np.fromiter((num - arc_start[sv] for sv in curr_sats), dtype=float, count=len(curr_sats))
                        if gim_new_arcs else num)
            r_diag[2::3] = _iono_constraint_sigma(
                epoch,
                ramp_num,
                self.interval,
                self.use_iono_rms,
                self.sigma_iono_0,
                self.sigma_iono_end,
                self.t_end,
            )
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
            # prefit residuals against the previous epoch's state (before the prediction and the clock prior, so the
            # code rows carry the previous epoch's receiver clock), the convention of the other uncombined routes
            # (generic R1/R2, R7); the IF, GRAPHIC and R4c routes report the innovation ekf.y instead
            prefit = Z - self.Hx(self.ekf.x, satellites)
            # receiver clock prior from the code rows (RTKLIB udclk_ppp; bug-fix stage 13b, A1(b))
            self.ekf.state_priors = clock_priors(self.cfg.clock_prior_sigma_m, np.arange(0, m, 3))
            with epoch_process_noise(self.ekf, t, self.interval):
                self.ekf.predict_update(
                    z=Z,
                    HJacobian=self.Hjacobian,
                    args=(satellites,),
                    Hx=self.Hx,
                    hx_args=(satellites,),
                )
            postfit = Z - self.Hx(self.ekf.x, satellites)
            arc_age = advance_arc_age(arc_age, curr_sats)
            ar_age = arc_age_array(arc_age, curr_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                n_sv = len(curr_sats)
                # float phase residuals and weights (rows 3i+1 carry N of satellite i; contract C3, NQ-PPP-20)
                ar_inputs = ar_phase_kwargs(postfit, np.diag(self.ekf.R),
                                            3 * np.arange(n_sv) + 1, self.base_dim + 2 * np.arange(n_sv) + 1,
                                            ekf=self.ekf)
                ar_float_state = keep_float_state(self.ekf, ar_cfg)
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
                            "eligible": ar_eligible_mask(epoch),
                        }
                    ],
                    settings=ar_cfg,
                    **ar_inputs,
                )
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(float_position(self.ekf, ar_float_state) - conv_xyz)
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0
                    elif position_diff > 0.005:
                        conv_time = None
            xyz = self.ekf.x[:3].copy()
            conv_xyz = float_position(self.ekf, ar_float_state)
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
            epoch_out = epoch.copy()
            epoch_out['system'] = system
            epoch_out['dtr'] = dtr
            epoch_out['ztd'] = ztd
            for name, values in (('prefit', prefit), ('postfit', postfit)):
                epoch_out[f'{name}_code_res'] = values[0::3]
                epoch_out[f'{name}_phase_res'] = values[1::3]
                epoch_out[f'{name}_iono_res'] = values[2::3]
            epoch_out['code_used'] = np.asarray(code_mask, dtype=bool)
            state_block = self.ekf.x[self.base_dim:self.base_dim + 2 * len(curr_sats)].reshape(-1, 2)
            epoch_out['I_state'] = state_block[:, 0]
            epoch_out['N_state'] = state_block[:, 1]
            residual_rows.append(epoch_out)
            if trace_filter:
                ppp_trace_epoch(
                    trace_filter,
                    self.__class__.__name__,
                    epoch=num,
                    time=t,
                    row=result_rows[-1],
                )
        if conv_time is not None and trace_filter:
            ppp_trace(
                trace_filter,
                self.__class__.__name__,
                "convergence",
                threshold_m=0.005,
                minutes=conv_time.total_seconds() / 60,
                hours=conv_time.total_seconds() / 3600,
            )
        # contracts C1 / C2 (bug-fix stage 13b): {system: residual frame} (the route returned its input
        # observations), ct_min [min]
        result = finish_solution(pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time')),
                                 conv_time)
        residuals = residual_frames({system: residual_rows}, {system: self.obs}, self.RESIDUAL_COLUMNS)
        return result, residuals, None, convergence_hours(conv_time)


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
    #: columns of a residual frame without rows; a dual-frequency system's frames also carry the
    #: per-band columns of _system_residual_output for bands 1 and 2
    RESIDUAL_COLUMNS = (
        "system", "reference_system", "system_isb", "dtr", "ztd",
        "prefit_code_res_1", "prefit_phase_res_1", "postfit_code_res_1", "postfit_phase_res_1", "code_used_1",
        "phase_used_1", "code_dcb_corr_1", "signal_bias_1", "phase_used", "strict_phase_code_gate", "dcb_total_m",
        "I_state", "N1_state",
    )

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
        self.signal_bias_q = 0.0 if self.estimate_signal_bias else random_walk_variance(
            getattr(self.cfg, "q_dcb", 0.0), interval)
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
        # phase-jump screening of the no-constraint settings (off by default, as R7 without constraints)
        self.enable_phase_jump_resets = bool(getattr(self.cfg, "uncombined_phase_screen", False))
        self.phase_jump_warmup_epochs = int(getattr(self.cfg, "uncombined_phase_screen_warmup_epochs", 60))
        if self.use_iono_constr:
            self.iono_dsb_alignment = {}
            for system, obs in self.obs_by_system.items():
                if "ion" not in obs.columns:
                    raise ValueError("use_iono_constr requires an 'ion' column in all mixed observations.")
                # the GIM in the satellite code-bias datum (bug-fix stage 13d), before the rows without a model value
                # go: the satellite set of the DSB datum (median) is that of the observations
                obs, self.iono_dsb_alignment[system] = _align_iono_model_to_code_biases(
                    obs, self.mode_by_system[system], system, self.cfg)
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

        code_var, phase_var = _uncombined_obs_variances(self.cfg, ev_deg)
        for band_idx in range(len(layout)):
            r_diag[:, 2 * band_idx] = np.where(code_mask_matrix[:, band_idx], code_var, 1e12)
            r_diag[:, 2 * band_idx + 1] = np.where(phase_mask_matrix[:, band_idx], phase_var, 1e12)

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
        # NQ-SPP-33 (author decision 2026-10-15): BDS-3 codes not moved to the clock datum are down-weighted by the
        # size of the missing term, added after the BeiDou noise scale (it is an error size, not a noise level)
        extra = unmoved_code_variance_m2(self.cfg, system, epoch, uncombined=True)
        for band_idx in range(len(layout)):
            r_diag[:, 2 * band_idx] += np.where(code_mask_matrix[:, band_idx], extra, 0.0)

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
        self.ekf.P[3, 3] = self.cfg.p_dt
        self.ekf.Q = np.zeros_like(self.ekf.P)
        self.ekf.Q[3, 3] = self.cfg.q_dt
        _init_coordinate_states(self.ekf, self.cfg)
        for system, col in self.isb_cols.items():
            self.ekf.P[col, col] = self.cfg.p_isb
            self.ekf.Q[col, col] = random_walk_variance(self.cfg.q_isb, self.interval)
        if self.tro:
            self.ekf.P[self.col_ztd, self.col_ztd] = self.cfg.p_tro
            self.ekf.Q[self.col_ztd, self.col_ztd] = random_walk_variance(self.cfg.q_tro, self.interval)
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
            self.ekf.Q[idx_i, idx_i] = random_walk_variance(self.cfg.q_iono, self.interval)
            for amb in range(1, state_dim):
                self.ekf.P[idx_i + amb, idx_i + amb] = self.cfg.p_amb
                self.ekf.Q[idx_i + amb, idx_i + amb] = random_walk_variance(getattr(self.cfg, "q_amb", 0.0),
                                                                            self.interval)

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

        # Receiver states (incl. signal-bias states) plus every retained satellite,
        # carried over together so the satellite-satellite covariances survive.
        keep_old = [np.arange(self.base_dim)]
        keep_new = [np.arange(self.base_dim)]
        for tag, (new_start, new_dim_state) in new_specs.items():
            if tag in old_specs:
                old_start, old_dim_state = old_specs[tag]
                common_dim = min(old_dim_state, new_dim_state)
                keep_old.append(np.arange(old_start, old_start + common_dim))
                keep_new.append(np.arange(new_start, new_start + common_dim))
                continue

            system, sv = tag.split(":", 1)
            sat_idx = curr_sats[system].index(sv)
            x_new[new_start] = ion_by_system[system][sat_idx]
            P_new[new_start, new_start] = self.cfg.p_iono
            Q_new[new_start, new_start] = random_walk_variance(self.cfg.q_iono, self.interval)
            for amb in range(1, new_dim_state):
                P_new[new_start + amb, new_start + amb] = self.cfg.p_amb
                Q_new[new_start + amb, new_start + amb] = random_walk_variance(getattr(self.cfg, "q_amb", 0.0),
                                                                               self.interval)

        _carry_over_states(x_old, P_old, Q_old, x_new, P_new, Q_new,
                           np.concatenate(keep_old), np.concatenate(keep_new))
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

    def _phase_rows(self, sats_by_system):
        """Phase rows of the epoch and the ambiguity state each carries (the AR phase residuals)."""
        rows, amb = [], []
        for system in self.systems:
            n_sats = len(sats_by_system.get(system, []))
            if not n_sats:
                continue
            n_bands = len(_mode_signals(self.mode_by_system[system]))
            sat = np.arange(n_sats)[:, None]
            band = np.arange(n_bands)[None, :]
            rows.append((self._system_obs_start(sats_by_system, system) + self._obs_dim(system) * sat
                         + 2 * band + 1).ravel())
            amb.append((self._system_state_start(sats_by_system, system) + self._state_dim(system) * sat
                        + 1 + band).ravel())
        if not rows:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return np.concatenate(rows), np.concatenate(amb)

    def _clock_prior_rows(self, sats_by_system):
        """Code rows of the epoch (every system and band) and {ISB state index: code rows of its system}."""
        code_rows, isb_rows = [], {}
        for system in self.systems:
            n_sats = len(sats_by_system.get(system, []))
            if not n_sats:
                continue
            obs_dim = self._obs_dim(system)
            start = self._system_obs_start(sats_by_system, system)
            n_bands = len(_mode_signals(self.mode_by_system[system]))
            rows = (start + obs_dim * np.arange(n_sats)[:, None] + 2 * np.arange(n_bands)[None, :]).ravel()
            code_rows.append(rows)
            if system in self.isb_cols:
                isb_rows[self.isb_cols[system]] = rows
        rows = np.concatenate(code_rows) if code_rows else np.zeros(0, dtype=int)
        return rows, isb_rows

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
                "system": system,
                "n1_idx": n1_idx,
                "n2_idx": n2_idx,
                "ev": epoch_by_system[system]["ev"].to_numpy(copy=False),
                "eligible": ar_eligible_mask(epoch_by_system[system]),
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
        conv_xyz = None  # previous float position (convergence check)
        conv_time = None
        T0 = all_times[0]
        age_by_system = {
            system: {}
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
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", True)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
            hold=bool(getattr(self.cfg, "pppar_fix_and_hold", False)),
        )

        ar_float_state = None
        for num, t in enumerate(all_times):
            # the previous epoch was reported fixed; without fix-and-hold the filter continues float
            restore_float_state(self.ekf, ar_float_state)
            ar_float_state = None
            reset_epoch = False
            if reset_every != 0 and ((num * self.interval) % reset_every == 0) and (num != 0):
                old_sats_by_system = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                age_by_system = {
                    system: {}
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
                        if num < self.phase_jump_warmup_epochs:
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
            prior_rows, isb_rows = self._clock_prior_rows(curr_sats_by_system)
            with epoch_process_noise(self.ekf, t, self.interval):
                for k in range(5):
                    try:
                        # receiver clock and ISB priors (RTKLIB udclk_ppp; bug-fix stage 13b, A1(b)), set again for
                        # every attempt (the call consumes them)
                        self.ekf.state_priors = clock_priors(self.cfg.clock_prior_sigma_m, prior_rows, isb_rows)
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
                    system: {}
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
                ar_inputs = ar_phase_kwargs(Z - self.Hx(self.ekf.x, *satellite_blocks),
                                            np.diag(self.ekf.R), *self._phase_rows(curr_sats_by_system), ekf=self.ekf)
                ar_float_state = keep_float_state(self.ekf, ar_cfg)
                self.ekf.x, self.ekf.P, ar_diag = apply_indexed_uncombined_pppar(
                    x=self.ekf.x,
                    P=self.ekf.P,
                    ambiguity_groups=self._ambiguity_groups(curr_sats_by_system, epoch_by_system, age_by_system),
                    settings=ar_cfg,
                    **ar_inputs,
                )

            postfit = Z - self.Hx(self.ekf.x, *satellite_blocks)
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(float_position(self.ekf, ar_float_state) - conv_xyz)
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0
                    elif position_diff > 0.005:
                        conv_time = None

            xyz = self.ekf.x[:3].copy()
            conv_xyz = float_position(self.ekf, ar_float_state)
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

        # contracts C1 / C2 (bug-fix stage 13b): ct_min [min]; an empty residual frame keeps the residual columns
        result = finish_solution(pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name="time")),
                                 conv_time)
        residual_map = residual_frames(residuals, self.obs_by_system, self.RESIDUAL_COLUMNS)
        return result, residual_map, None, convergence_hours(conv_time)


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
        # code gate and phase-jump screening of the constrained settings (defaults 10 m, on, legacy
        # source, 1 m, 60 epochs), as R7 with constraints
        cfg = self.cfg
        self.code_prefit_threshold = float(getattr(cfg, "uncombined_iono_code_prefit_threshold", 10.0))
        self.enable_phase_jump_resets = bool(getattr(cfg, "uncombined_iono_phase_screen", True))
        self.legacy_phase_jump_resets = bool(getattr(cfg, "uncombined_iono_phase_screen_legacy_source", True))
        self.phase_jump_reset_threshold = float(getattr(cfg, "uncombined_iono_phase_screen_threshold", 1.0))
        self.phase_jump_warmup_epochs = int(getattr(cfg, "uncombined_iono_phase_screen_warmup_epochs", 60))

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
            self.ekf.Q[idx_c2, idx_c2] = random_walk_variance(self.cfg.q_dcb, self.interval)

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

        # Receiver states, retained satellites and receiver DCB states are carried
        # over together, so no correlation between them is lost.
        keep_old = [np.arange(base)]
        keep_new = [np.arange(base)]
        if common:
            old_idx = np.array([prev_map[tag] for tag in common], dtype=int)
            new_idx = np.array([curr_map[tag] for tag in common], dtype=int)
            offs = np.array([0, 1, 2], dtype=int)
            keep_old.append((base + 3 * old_idx[:, None] + offs[None, :]).ravel())
            keep_new.append((base + 3 * new_idx[:, None] + offs[None, :]).ravel())

        for tag in curr_all:
            if tag in prev_map:
                continue
            idx = curr_map[tag]
            idx_i = base + 3 * idx
            idx_n1 = idx_i + 1
            idx_n2 = idx_i + 2
            x_new[idx_i] = 0.0
            P_new[idx_i, idx_i] = self.cfg.p_iono
            Q_new[idx_i, idx_i] = random_walk_variance(self.cfg.q_iono, self.interval)
            P_new[idx_n1, idx_n1] = self.cfg.p_amb
            P_new[idx_n2, idx_n2] = self.cfg.p_amb
            # PPP-064 (bug-fix stage 13b): q_amb on the ambiguities of a rising satellite too (Q = 0 before)
            Q_new[idx_n1, idx_n1] = random_walk_variance(self.cfg.q_amb, self.interval)
            Q_new[idx_n2, idx_n2] = random_walk_variance(self.cfg.q_amb, self.interval)

        old_dcb = self._dcb_cols(prev_sats)
        new_dcb = self._dcb_cols(curr_sats)
        for system in self.systems:
            if system in old_dcb and system in new_dcb:
                keep_old.append(np.array(old_dcb[system], dtype=int))
                keep_new.append(np.array(new_dcb[system], dtype=int))
        _carry_over_states(x_old, P_old, Q_old, x_new, P_new, Q_new,
                           np.concatenate(keep_old), np.concatenate(keep_new))
        for _system, (idx_c1, idx_c2) in new_dcb.items():
            # The first code bias of each system is the datum: fixed to zero
            # with no correlation to the other states.
            x_new[idx_c1] = 0.0
            _reset_covariance_states(P_new, [idx_c1], 1e-12)
            _reset_covariance_states(Q_new, [idx_c1], 0.0)
            if P_new[idx_c2, idx_c2] == 0.0:
                P_new[idx_c2, idx_c2] = self.cfg.p_dcb
                Q_new[idx_c2, idx_c2] = random_walk_variance(self.cfg.q_dcb, self.interval)
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
        if not bool(getattr(self.cfg, "uncombined_iono_clock_identity", True)):
            self.ekf.F[3, 3] = 0.0  # receiver clock re-estimated every epoch, as R7

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
        # constrained weighting of the system (PPP-074, bug-fix stage 13b), as R7 and R4c
        code_var, phase_var = _constrained_obs_variances(self.cfg, ev_deg, system)
        # NQ-SPP-33 (author decision 2026-10-15): BDS-3 codes not moved to the clock datum are down-weighted
        code_var = code_var + unmoved_code_variance_m2(self.cfg, system, epoch, uncombined=True)
        for band_idx in range(len(layout)):
            r_matrix[:, 2 * band_idx] = np.where(
                diag["code_masks"][:, band_idx],
                code_var,
                1e12,
            )
            r_matrix[:, 2 * band_idx + 1] = np.where(
                diag["phase_masks"][:, band_idx],
                phase_var,
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

    #: columns the filter adds to the observation rows in its residual frames
    RESIDUAL_COLUMNS = ("stec", "n1", "n2")

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
            # the GIM in the satellite code-bias datum (bug-fix stage 13d), before the rows without a model value go
            self.iono_dsb_alignment = {}
            self.gps_obs, self.iono_dsb_alignment["G"] = _align_iono_model_to_code_biases(
                self.gps_obs, self.gps_mode, "G", self.cfg)
            self.gal_obs, self.iono_dsb_alignment["E"] = _align_iono_model_to_code_biases(
                self.gal_obs, self.gal_mode, "E", self.cfg)
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
        self.ekf.Q[4, 4] = random_walk_variance(self.cfg.q_isb, self.interval)
        if self.tro:
            self.ekf.Q[5, 5] = random_walk_variance(self.cfg.q_tro, self.interval)
        _init_coordinate_states(self.ekf, self.cfg)
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
            self.ekf.Q[idx_i, idx_i] = random_walk_variance(self.cfg.q_iono, self.interval)
            for amb in range(1, state_dim):
                self.ekf.P[idx_i + amb, idx_i + amb] = self.cfg.p_amb
                self.ekf.Q[idx_i + amb, idx_i + amb] = random_walk_variance(self.cfg.q_amb, self.interval)

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

        # Receiver states plus every retained satellite, carried over together so
        # the satellite-satellite covariances survive.
        keep_old = [np.arange(self.base_dim)]
        keep_new = [np.arange(self.base_dim)]
        for tag, (new_start, new_dim_state) in new_specs.items():
            if tag in old_specs:
                old_start, old_dim_state = old_specs[tag]
                common_dim = min(old_dim_state, new_dim_state)
                keep_old.append(np.arange(old_start, old_start + common_dim))
                keep_new.append(np.arange(new_start, new_start + common_dim))
            else:
                sys_tag, sv = tag.split(":", 1)
                sat_list = curr_gps if sys_tag == "G" else curr_gal
                ion_values = gps_ion if sys_tag == "G" else gal_ion
                sat_idx = sat_list.index(sv)
                x_new[new_start] = ion_values[sat_idx]
                P_new[new_start, new_start] = self.cfg.p_iono
                Q_new[new_start, new_start] = random_walk_variance(self.cfg.q_iono, self.interval)
                for amb in range(1, new_dim_state):
                    P_new[new_start + amb, new_start + amb] = self.cfg.p_amb
                    Q_new[new_start + amb, new_start + amb] = random_walk_variance(self.cfg.q_amb, self.interval)

        _carry_over_states(x_old, P_old, Q_old, x_new, P_new, Q_new,
                           np.concatenate(keep_old), np.concatenate(keep_new))
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

    def _ar_ambiguity_groups(self, n_gps, n_gal, gps_ev, gal_ev, gps_age=None, gal_age=None,
                             gps_eligible=None, gal_eligible=None):
        groups = []
        for system, n_sats, mode, ev, age, eligible in (
            ("G", n_gps, self.gps_mode, gps_ev, gps_age, gps_eligible),
            ("E", n_gal, self.gal_mode, gal_ev, gal_age, gal_eligible),
        ):
            if n_sats == 0:
                continue
            state_dim = self._state_dim(mode)
            state_start = self._system_state_start(n_gps, system)
            sat_idx = np.arange(n_sats)
            n1_idx = state_start + state_dim * sat_idx + 1
            n2_idx = state_start + state_dim * sat_idx + 2 if state_dim > 2 else None
            group = {
                "system": system,
                "n1_idx": n1_idx,
                "n2_idx": n2_idx,
                "ev": ev,
                "eligible": eligible,
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
        conv_xyz = None  # previous float position (convergence check)
        conv_time = None
        T0 = all_times[0]
        gps_arc_age = {}
        gal_arc_age = {}
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
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", True)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
            hold=bool(getattr(self.cfg, "pppar_fix_and_hold", False)),
        )

        ar_float_state = None
        for num, t in enumerate(all_times):
            # the previous epoch was reported fixed; without fix-and-hold the filter continues float
            restore_float_state(self.ekf, ar_float_state)
            ar_float_state = None
            reset_epoch = False
            if reset_every != 0 and ((num * self.interval) % reset_every == 0) and (num != 0):
                old_gps_sats, old_gal_sats = self.reset_filter(epoch=t, clk0=clk0, zwd0=zwd0)
                gps_arc_age = {}
                gal_arc_age = {}
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

            # receiver clock (every code row) and Galileo ISB (state 4, its code rows) priors, RTKLIB udclk_ppp
            # (bug-fix stage 13b, A1(b))
            gps_code = (self._obs_dim(self.gps_mode) * np.arange(len(curr_gps_sats))[:, None]
                        + 2 * np.arange(len(gps_layout))[None, :]).ravel()
            gal_code = (len(z_gps) + self._obs_dim(self.gal_mode) * np.arange(len(curr_gal_sats))[:, None]
                        + 2 * np.arange(len(gal_layout))[None, :]).ravel()
            prior_rows = np.concatenate((gps_code, gal_code)).astype(int)
            updated = False
            with epoch_process_noise(self.ekf, t, self.interval):
                for k in range(5):
                    try:
                        self.ekf.state_priors = clock_priors(self.cfg.clock_prior_sigma_m, prior_rows,
                                                             {4: gal_code.astype(int)})
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
                gps_arc_age = {}
                gal_arc_age = {}
                continue
            gps_arc_age = advance_arc_age(gps_arc_age, curr_gps_sats)
            gal_arc_age = advance_arc_age(gal_arc_age, curr_gal_sats)
            gps_ar_age = arc_age_array(gps_arc_age, curr_gps_sats)
            gal_ar_age = arc_age_array(gal_arc_age, curr_gal_sats)
            ar_diag = None
            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                # float phase residuals and weights (contract C3, NQ-PPP-20)
                phase_rows, amb_idx = [], []
                for system, mode, n_sv in (("G", self.gps_mode, len(curr_gps_sats)),
                                           ("E", self.gal_mode, len(curr_gal_sats))):
                    sat = np.arange(n_sv)[:, None]
                    band = np.arange(len(_mode_signals(mode)))[None, :]
                    phase_rows.append((self._system_obs_start(len(curr_gps_sats), system)
                                       + self._obs_dim(mode) * sat + 2 * band + 1).ravel())
                    amb_idx.append((self._system_state_start(len(curr_gps_sats), system)
                                    + self._state_dim(mode) * sat + 1 + band).ravel())
                ar_inputs = ar_phase_kwargs(Z - self.Hx(self.ekf.x, gps_satellites, gal_satellites),
                                            np.diag(self.ekf.R), np.concatenate(phase_rows), np.concatenate(amb_idx),
                                            ekf=self.ekf)
                ar_float_state = keep_float_state(self.ekf, ar_cfg)
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
                        gps_eligible=ar_eligible_mask(gps_epoch),
                        gal_eligible=ar_eligible_mask(gal_epoch),
                    ),
                    settings=ar_cfg,
                    **ar_inputs,
                )

            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(float_position(self.ekf, ar_float_state) - conv_xyz)
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0
                    elif position_diff > 0.005:
                        conv_time = None

            xyz = self.ekf.x[:3].copy()
            conv_xyz = float_position(self.ekf, ar_float_state)
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

        # contracts C1 / C2 (bug-fix stage 13b): {system: residual frame}, ct_min [min]
        result = finish_solution(pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time')),
                                 conv_time)
        residuals = residual_frames({"G": gps_results, "E": gal_results}, {"G": self.gps_obs, "E": self.gal_obs},
                                    self.RESIDUAL_COLUMNS)
        return result, residuals, None, convergence_hours(conv_time)


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
        Uses uncombined wide-lane/narrow-lane AR when enabled, for satellites
        with product phase biases, once lock-age/candidate gates pass.
        BeiDou AR runs only when the bias product gives BDS phase biases
        (e.g. GRG L2I/L6I; COD MGEX has none) and is less validated than GPS
        and Galileo AR.

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
        try:
            sig1, _ = _dual_mode_signals(self.mode)
            self.system = SIGNALS[sig1].system
        except Exception:
            # Fallback for unexpected single-frequency modes (should not happen for PPPUdSingleGNSS).
            self.system = "G"
        self.iono_dsb_alignment = {}
        if self.use_iono_constr and self.obs is not None:
            if 'ion' not in self.obs.columns:
                raise ValueError("use_iono_constr requires an 'ion' column in observations.")
            # the GIM in the satellite code-bias datum (bug-fix stage 13d), before the rows without a model value go
            self.obs, self.iono_dsb_alignment[self.system] = _align_iono_model_to_code_biases(
                self.obs, self.mode, self.system, self.cfg)
            self.obs = self.obs.dropna(subset=['ion'])
            if self.use_iono_rms and 'ion_rms' in self.obs.columns:
                self.obs = self.obs.dropna(subset=['ion_rms'])
        elif self.obs is None:
            self.use_iono_constr = False
        self.obs_per_sat = 5 if self.use_iono_constr else 4

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
        self.ekf.Q[idx_c2, idx_c2] = random_walk_variance(self.cfg.q_dcb, self.interval)

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

    #: columns the filter adds to the observation rows in its residual frame
    RESIDUAL_COLUMNS = (
        "system", "dtr", "ztd", "prefit_code_res_1", "prefit_phase_res_1", "prefit_code_res_2", "prefit_phase_res_2",
        "postfit_code_res_1", "postfit_phase_res_1", "postfit_code_res_2", "postfit_phase_res_2", "code_used_1",
        "code_used_2", "code_dcb_corr_1", "code_dcb_corr_2", "dcb_total_m", "rx_dcb_m", "rx_code_bias_1_m",
        "rx_code_bias_2_m", "I_state", "N1_state", "N2_state",
    )

    def _single_iono_obs_variances(self, ev_deg: np.ndarray, n_sats: int) -> tuple[np.ndarray, np.ndarray]:
        ev_deg = np.broadcast_to(np.asarray(ev_deg, dtype=float), (n_sats,))
        if self.use_iono_constr:
            return _constrained_obs_variances(self.cfg, ev_deg, self.system)
        return _uncombined_obs_variances(self.cfg, ev_deg)

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
            Q_new[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            P_new[idx_N1, idx_N1] = self.cfg.p_amb#1e6
            P_new[idx_N2, idx_N2] = self.cfg.p_amb#1e6
            # PPP-064 (bug-fix stage 13b): q_amb on every ambiguity (it was ignored by this filter)
            Q_new[idx_N1, idx_N1] = random_walk_variance(self.cfg.q_amb, self.interval)
            Q_new[idx_N2, idx_N2] = random_walk_variance(self.cfg.q_amb, self.interval)

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
        self.ekf.Q[3, 3] = self.cfg.q_dt  # per epoch, as init_filter (was scaled to hours here only)
        self.ekf.Q[4, 4] = random_walk_variance(self.cfg.q_tro, self.interval)
        _init_coordinate_states(self.ekf, self.cfg)
        self._init_rx_signal_bias_states()
        self._configure_transition_matrix()

        for k in range(n0):
            idx_I = self.base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            self.ekf.P[idx_I, idx_I] =self.cfg.p_iono# 100
            self.ekf.Q[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb #1e6

            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb #1e6
            self.ekf.Q[idx_N1, idx_N1] = random_walk_variance(self.cfg.q_amb, self.interval)  # PPP-064
            self.ekf.Q[idx_N2, idx_N2] = random_walk_variance(self.cfg.q_amb, self.interval)
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
        self.ekf.Q[4, 4] = random_walk_variance(self.cfg.q_tro, self.interval)
        _init_coordinate_states(self.ekf, self.cfg)
        self._init_rx_signal_bias_states()
        self._configure_transition_matrix()

        for k in range(n0):
            idx_I = self.base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono #100
            self.ekf.Q[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb#1e6

            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb #1e6
            self.ekf.Q[idx_N1, idx_N1] = random_walk_variance(self.cfg.q_amb, self.interval)  # PPP-064
            self.ekf.Q[idx_N2, idx_N2] = random_walk_variance(self.cfg.q_amb, self.interval)
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
        Ambiguity vector N1, N2 and its covariance matrix from the full state and the matrix P.

        Parameters:
            x        -- EKF state vector (1D array)
            P        -- covariance matrix (2D array)
            n_sats   -- number of satellites (blocks [I_s, N1_s, N2_s])
            base_dim -- number of global parameters (default=5)

        Returns:
            N_vec    -- vector [N1_1, N2_1, N1_2, N2_2, ..., N1_n, N2_n]
            P_N      -- covariance matrix of N_vec
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
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", True)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
            hold=bool(getattr(self.cfg, "pppar_fix_and_hold", False)),
        )
        xyz = None
        conv_xyz = None  # previous float position (convergence check)
        conv_time = None
        T0 = all_epochs[0]
        arc_age: dict = {}  # PPP-069: the first advance_arc_age gives 0
        phase_traces_1: dict[int, dict[str, float]] = {}
        phase_traces_2: dict[int, dict[str, float]] = {}
        ar_float_state = None
        for num, t in enumerate(epochs):
            # the previous epoch was reported fixed; without fix-and-hold the filter continues float
            restore_float_state(self.ekf, ar_float_state)
            ar_float_state = None
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    old_sats = self.reset_filter(epoch=t)
                    arc_age = {}  # PPP-069: every arc restarts at age 0
                    reset_epoch = True
                    T0 = t
            gps_epoch = obs_epochs.get(t)
            if gps_epoch is None:
                continue

            def safe_get(df, col, length=None):
                """The column as a NumPy array; a vector of zeros of the given length when it is missing."""
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
                # NQ-SPP-33 (author decision 2026-10-15): BDS-3 codes not moved to the clock datum are down-weighted
                code_var_epoch = code_var_epoch + unmoved_code_variance_m2(self.cfg, self.system, gps_epoch,
                                                                                   uncombined=True)
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
                code_var, phase_var = self._single_iono_obs_variances(gps_ev, n_sats)
                # NQ-SPP-33 (author decision 2026-10-15): BDS-3 codes not moved to the clock datum are down-weighted
                code_var = code_var + unmoved_code_variance_m2(self.cfg, self.system, gps_epoch,
                                                                       uncombined=True)
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

            # receiver clock prior from the code rows of both bands (RTKLIB udclk_ppp; bug-fix stage 13b, A1(b))
            code_rows = (rows_per_sat * np.arange(n_sats)[:, None] + np.array([0, 2])[None, :]).ravel()
            self.ekf.state_priors = clock_priors(self.cfg.clock_prior_sigma_m, code_rows)
            with epoch_process_noise(self.ekf, t, self.interval):
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
                # float phase residuals and weights: rows 1 / 3 of each satellite carry N1 / N2 (C3, NQ-PPP-20)
                sat = np.arange(n_sats)[:, None]
                ar_inputs = ar_phase_kwargs(postfit, np.diag(self.ekf.R),
                                            (rows_per_sat * sat + np.array([1, 3])).ravel(),
                                            (self.base_dim + 3 * sat + np.array([1, 2])).ravel(), ekf=self.ekf)
                ar_float_state = keep_float_state(self.ekf, ar_cfg)
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
                    gps_eligible=ar_eligible_mask(gps_epoch),
                    **ar_inputs,
                )

            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(float_position(self.ekf, ar_float_state) - conv_xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time
            xyz = self.ekf.x[:3].copy()
            conv_xyz = float_position(self.ekf, ar_float_state)
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
        if conv_time is not None and trace_filter:
            ppp_trace(
                trace_filter,
                self.__class__.__name__,
                "convergence",
                threshold_m=0.005,
                minutes=conv_time.total_seconds() / 60,
                hours=conv_time.total_seconds() / 3600,
            )
        # contracts C1 / C2 (bug-fix stage 13b, PPP-033): {system: residual frame} (the frame was returned twice, so
        # a GPS run filled residuals_gal with a copy), ct_min [min]
        output = finish_solution(pd.DataFrame(result_rows, index=pd.DatetimeIndex(result_times, name='time')),
                                 conv_time)
        residuals = residual_frames({self.system: obs_result}, {self.system: self.obs}, self.RESIDUAL_COLUMNS)
        return output, residuals, None, convergence_hours(conv_time)


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

    #: columns of the per-satellite residual frames: slant ionosphere, ambiguities, corrected code and
    #: phase [m] and their innovations
    RESIDUAL_COLUMNS = ("Idelay", "n1", "n2", "C1", "L1", "C2", "L2", "vc1", "vl1", "vc2", "vl2")

    def __init__(self, config: PPPConfig, gps_obs, gps_mode, gal_obs, gal_mode, ekf, pos0, tro=True, est_dcb=False,
                 interval=0.5,
                 use_iono_rms=True, sigma_iono_0=1.1,
                 sigma_iono_end=3.0, t_end=30):
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
        :param t_end: [float, int] length [min] of the constraint ramp from sigma_iono_0 to
            sigma_iono_end (PPPConfig.t_end; the ramp counts from the first epoch)
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
        # receiver signal-bias states also in the phase rows (uncombined_iono_signal_bias_phase_jacobian),
        # in Hjacobian and Hx alike; the Jacobian alone had them, so every epoch the phase residuals
        # missed the L2/E5a code-bias state (metres) and the update pushed it into ZTD and height
        # (PPP-061, 2-4 m height error in the first minutes, hidden by the loose q_tro before PPP-060)
        self._phase_signal_bias = bool(getattr(config, "uncombined_iono_signal_bias_phase_jacobian", True))
        self.pos0 = pos0
        self.interval = interval
        # --------
        # the GIM in the satellite code-bias datum (bug-fix stage 13d), before the rows without a model value go
        self.iono_dsb_alignment = {}
        self.gps_obs, self.iono_dsb_alignment["G"] = _align_iono_model_to_code_biases(
            self.gps_obs, self.gps_mode, "G", self.cfg)
        self.gal_obs, self.iono_dsb_alignment["E"] = _align_iono_model_to_code_biases(
            self.gal_obs, self.gal_mode, "E", self.cfg)
        self.gps_obs = self.gps_obs.dropna(subset=['ion'])
        self.gal_obs = self.gal_obs.dropna(subset=['ion'])
        if 'ion_rms' in self.gps_obs.columns:
            self.gps_obs = self.gps_obs.dropna(subset=['ion_rms'])
        if 'ion_rms' in self.gal_obs.columns:
            self.gal_obs = self.gal_obs.dropna(subset=['ion_rms'])
        self.use_iono_rms = use_iono_rms
        # None (allowed by PPPConfig, the default of sigma_iono_end) takes the defaults of this route:
        # sigma_iono_end 3.0, as the other dual-frequency constrained filters (author decision NQ-PPP-22,
        # revised in bug-fix stage 9: the former 2.3 compensated the PPP-061 model mismatch; after its fix
        # 3.0 converges faster on every static and storm R4c row, e.g. NYA2 6 h t_c10 91 -> 82.5 min, finals
        # within 0.06 cm; kinematic BRUX RMS -0.66 cm on the pre-stage-9 GIM, +0.26 cm with the stage-9 GIM)
        self.sigma_iono_0 = 1.1 if sigma_iono_0 is None else sigma_iono_0
        self.sigma_iono_end = 3.0 if sigma_iono_end is None else sigma_iono_end
        self.t_end = 30 if t_end is None else t_end

    def _configure_transition_matrix(self):
        self.ekf.F = np.eye(self.ekf.dim_x)
        if not bool(getattr(self.cfg, "uncombined_iono_clock_identity", True)):
            self.ekf.F[3, 3] = 0.0  # receiver clock re-estimated every epoch, as R7

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
                H[row + 2, dcb_base + 1] = 1.0  # DCB_C2_GPS
                if self._phase_signal_bias:  # same terms as Hx
                    H[row + 1, dcb_base + 0] = 1.0  # DCB_L1_GPS==C1
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
                H[row + 2, dcb_base + 3] = 1.0  # DCB_C2_GAL
                if self._phase_signal_bias:  # same terms as Hx
                    H[row + 1, dcb_base + 2] = 1.0  # DCB_L1_GAL==C1
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
        # receiver signal-bias terms of the phase rows, the same as in Hjacobian (PPP-061)
        pdcb = dcb if self._phase_signal_bias else np.zeros(4)
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
            z_hat[5 * s + 1] = geom + clk + mw * zwd - MU1 * I_s + L1 * N1_s + pdcb[0]
            z_hat[5 * s + 2] = geom + clk + mw * zwd + MU2 * I_s + dcb[1]
            z_hat[5 * s + 3] = geom + clk + mw * zwd - MU2 * I_s + L2 * N2_s + pdcb[1]
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
            z_hat[row + 1] = geom + clk + isb + mw * zwd - MU1_GAL * I_s + E1 * N1_s + pdcb[2]
            z_hat[row + 2] = geom + clk + isb + mw * zwd + MU2_GAL * I_s + dcb[3]
            z_hat[row + 3] = geom + clk + isb + mw * zwd - MU2_GAL * I_s + E5a * N2_s + pdcb[3]
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
        self.ekf.Q[4, 4] = random_walk_variance(self.cfg.q_isb, self.interval)  # isb
        self.ekf.Q[5, 5] = random_walk_variance(self.cfg.q_tro, self.interval)  # tro
        _init_coordinate_states(self.ekf, self.cfg)
        self._configure_transition_matrix()

        q_amb = random_walk_variance(self.cfg.q_amb, self.interval)
        P4_gps = self.gps_obs.loc[(slice(None), first_ep), 'ion'].to_numpy()
        P4_gal = self.gal_obs.loc[(slice(None), first_ep), 'ion'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 3 * i
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gps[i]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono  # 1.5
            self.ekf.Q[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb  # 1e6
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb  # 1e6
            self.ekf.Q[idx_N1, idx_N1] = q_amb  # PPP-064 (bug-fix stage 13b)
            self.ekf.Q[idx_N2, idx_N2] = q_amb
        for j, sv in enumerate(gal_sats):
            k = n_gps + j
            idx_I = base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gal[j]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono  # 1.5
            self.ekf.Q[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb
            self.ekf.Q[idx_N1, idx_N1] = q_amb  # PPP-064
            self.ekf.Q[idx_N2, idx_N2] = q_amb
        if self.est_dcb:
            dcb_indices = self.dcb_param_indices(n_gps, n_gal)
            for idx in dcb_indices:
                self.ekf.P[idx, idx] = self.cfg.p_dcb  # 1e2
                self.ekf.Q[idx, idx] = random_walk_variance(self.cfg.q_dcb, self.interval)
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
        self.ekf.Q[4, 4] = random_walk_variance(self.cfg.q_isb, self.interval)  # isb
        self.ekf.Q[5, 5] = random_walk_variance(self.cfg.q_tro, self.interval)  # tro
        _init_coordinate_states(self.ekf, self.cfg)
        self._configure_transition_matrix()

        q_amb = random_walk_variance(self.cfg.q_amb, self.interval)
        P4_gps = self.gps_obs.loc[(slice(None), epoch), 'ion'].to_numpy()
        P4_gal = self.gal_obs.loc[(slice(None), epoch), 'ion'].to_numpy()
        for i, sv in enumerate(gps_sats):
            idx_I = base_dim + 3 * i
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gps[i]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono  # 1.5
            self.ekf.Q[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb  # 1e6
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb  # 1e6
            self.ekf.Q[idx_N1, idx_N1] = q_amb  # PPP-064 (bug-fix stage 13b)
            self.ekf.Q[idx_N2, idx_N2] = q_amb
        for j, sv in enumerate(gal_sats):
            k = n_gps + j
            idx_I = base_dim + 3 * k
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            I_init = P4_gal[j]
            self.ekf.x[idx_I] = I_init
            self.ekf.P[idx_I, idx_I] = self.cfg.p_iono  # 1.5
            self.ekf.Q[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            self.ekf.P[idx_N1, idx_N1] = self.cfg.p_amb
            self.ekf.P[idx_N2, idx_N2] = self.cfg.p_amb
            self.ekf.Q[idx_N1, idx_N1] = q_amb  # PPP-064
            self.ekf.Q[idx_N2, idx_N2] = q_amb
        if self.est_dcb:
            dcb_indices = self.dcb_param_indices(n_gps, n_gal)
            for idx in dcb_indices:
                self.ekf.P[idx, idx] = self.cfg.p_dcb  # 1e2
                self.ekf.Q[idx, idx] = random_walk_variance(self.cfg.q_dcb, self.interval)
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

        # Receiver states, retained satellites and DCB states are carried over
        # together (one joint covariance block), so no correlation is lost.
        keep_old = [np.arange(base)]
        keep_new = [np.arange(base)]

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
            keep_old.append((base + 3 * old_idx[:, None] + offs[None, :]).ravel())
            keep_new.append((base + 3 * new_idx[:, None] + offs[None, :]).ravel())

        # nowe satelity: tylko diagonalne init
        new_only = [prn for prn in curr_all if prn not in prev_map]
        for prn in new_only:
            j = curr_map[prn]
            idx_I = base + 3 * j
            idx_N1 = idx_I + 1
            idx_N2 = idx_I + 2
            P_new[idx_I, idx_I] = self.cfg.p_iono
            Q_new[idx_I, idx_I] = random_walk_variance(self.cfg.q_iono, self.interval)
            P_new[idx_N1, idx_N1] = self.cfg.p_amb
            P_new[idx_N2, idx_N2] = self.cfg.p_amb
            Q_new[idx_N1, idx_N1] = random_walk_variance(self.cfg.q_amb, self.interval)  # PPP-064
            Q_new[idx_N2, idx_N2] = random_walk_variance(self.cfg.q_amb, self.interval)

        # DCB states: part of the joint block above
        if self.est_dcb:
            dcb_new = self.dcb_param_indices(len(curr_gps), len(curr_gal))
            dcb_old = self.dcb_param_indices(len(prev_gps), len(prev_gal))
            for i, idx in enumerate(dcb_new):
                if i < len(dcb_old):
                    keep_old.append(np.array([dcb_old[i]], dtype=int))
                    keep_new.append(np.array([idx], dtype=int))
                else:
                    P_new[idx, idx] = self.cfg.p_dcb
                    Q_new[idx, idx] = random_walk_variance(self.cfg.q_dcb, self.interval)

        _carry_over_states(x_old, P_old, Q_old, x_new, P_new, Q_new,
                           np.concatenate(keep_old), np.concatenate(keep_new))
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

        # check that all N previous epochs are present
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
                # PPP-064 (bug-fix stage 13b): the restarted ambiguity keeps the q_amb random walk (Q was set to 0)
                self.ekf.Q[ii, ii] = random_walk_variance(self.cfg.q_amb, self.interval)
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
        :param reset_every: int, reset period [min] (PPPConfig.reset_every: the filter is reset when
            the epoch index x interval [min] is a multiple of it), 0 = never
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

        t_end = max(self.t_end / self.interval, 1.0)

        sigma_iono_t = self.sigma_iono_0 + (self.sigma_iono_end - self.sigma_iono_0) / t_end * np.arange(t_end)
        # code gate and phase-jump screening of the constrained uncombined settings (PPP-029)
        code_prefit_threshold = float(getattr(self.cfg, "uncombined_iono_code_prefit_threshold", 10.0))
        phase_screen_enabled = bool(getattr(self.cfg, "uncombined_iono_phase_screen", True))
        phase_screen_threshold = float(getattr(self.cfg, "uncombined_iono_phase_screen_threshold", 1.0))
        phase_screen_warmup_epochs = int(getattr(self.cfg, "uncombined_iono_phase_screen_warmup_epochs", 60))
        xyz = None
        conv_xyz = None  # previous float position (convergence check)
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
            partial_fixing_enabled=bool(getattr(self.cfg, "pppar_partial_fixing_enabled", True)),
            partial_min_ambiguities=getattr(self.cfg, "pppar_partial_min_ambiguities", None),
            wide_lane_max_frac_cycles=getattr(self.cfg, "pppar_wide_lane_max_frac_cycles", 0.25),
            hold=bool(getattr(self.cfg, "pppar_fix_and_hold", False)),
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

        ar_float_state = None
        for num, t in enumerate(epochs):
            # the previous epoch was reported fixed; without fix-and-hold the filter continues float
            restore_float_state(self.ekf, ar_float_state)
            ar_float_state = None
            reset_epoch = False
            if reset_every != 0:
                if ((num * self.interval) % reset_every == 0) and (num != 0):
                    ppp_trace(trace_filter, self.__class__.__name__, "reset", epoch=num, time=t)
                    gps_sats, gal_sats = self.reset_filter(epoch=t)
                    ar_float_P = self.ekf.P.copy()
                    gps_arc_age = {}
                    gal_arc_age = {}
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
                self._configure_transition_matrix()

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
            gps_satellites = gps_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()
            gal_satellites = gal_epoch[['xe', 'ye', 'ze', 'me_wet']].to_numpy()

            # ---- Obserwacje i constrainty ----

            def get_obs(epoch, cols, sys):
                if sys == "G":
                    _mode = self.gps_mode
                    sat_pco1_col = f"sat_pco_los_{_mode[:2]}"
                    sat_pco2_col = f"sat_pco_los_{_mode[2:]}"
                    pco2_col = _dual_rec_pco_columns(_mode)[1]  # E1E5b/E5aE5b: pco_los_l5 (PPP-022)
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
                    pco2_col = _dual_rec_pco_columns(_mode)[1]  # E1E5b/E5aE5b: pco_los_l5 (PPP-022)

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

                # satellite plus receiver OSB [m] (OSB_ and STA_OSB_ columns, as the other filters;
                # PPPSession writes product DSBs in the clock datum as OSB_, PPP-038)
                n_obs = len(epoch)
                osb_c1 = osb_m(epoch, cols['C1'], n_obs)
                osb_c2 = osb_m(epoch, cols['C2'], n_obs)
                osb_l1 = osb_m(epoch, cols['L1'], n_obs)
                osb_l2 = osb_m(epoch, cols['L2'], n_obs)

                # stałe tylko raz
                cl = self.CLIGHT
                clk_cl = clk * cl

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
            if phase_screen_enabled and num >= phase_screen_warmup_epochs:
                gps_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gps_sats, phase_residuals_dict=prt_gps_l1, num=num, freq='n1',
                        thr=phase_screen_threshold,
                    )
                )
                gps_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gps_sats, phase_residuals_dict=prt_gps_l2, num=num, freq='n2',
                        thr=phase_screen_threshold,
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
            if phase_screen_enabled and num >= phase_screen_warmup_epochs:
                gal_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gal_sats, phase_residuals_dict=prt_gal_l1, num=num,
                        sys='E', len_gps=len(curr_gps_sats), freq='n1', thr=phase_screen_threshold,
                    )
                )
                gal_phase_reset_svs.update(
                    self.phase_residuals_screening(
                        sat_list=curr_gal_sats, phase_residuals_dict=prt_gal_l2, num=num,
                        sys='E', len_gps=len(curr_gps_sats), freq='n2', thr=phase_screen_threshold,
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
            # variances [m^2] of the constrained weighting per system (PPP-074, bug-fix stage 13b; see
            # _constrained_obs_variances)
            sigma_code_gps, sigma_phase_gps = _constrained_obs_variances(self.cfg, ev_gps, "G")
            sigma_code_gal, sigma_phase_gal = _constrained_obs_variances(self.cfg, ev_gal, "E")
            if self.use_iono_rms and 'ion_rms' in gps_epoch.columns and 'ion_rms' in gal_epoch.columns:
                # the GIM RMS [m] added to the ramp variance (PPP-074; it multiplied the ramp before, as
                # _iono_constraint_sigma did)
                sigma_ion_gps = _ramp_plus_ion_rms(sigma_ion, gps_epoch['ion_rms'].to_numpy(copy=False),
                                                   self.sigma_iono_end)
                sigma_ion_gal = _ramp_plus_ion_rms(sigma_ion, gal_epoch['ion_rms'].to_numpy(copy=False),
                                                   self.sigma_iono_end)
            else:
                sigma_ion_gps = np.full_like(ev_gps, sigma_ion)
                sigma_ion_gal = np.full_like(ev_gal, sigma_ion)

            gps_c1_mask, prefit_c1 = self.code_screening(x=self.ekf.x[:3], satellites=gps_satellites[:, :3],
                                                         code_obs=C1, thr=code_prefit_threshold)
            gps_c2_mask, prefit_c2 = self.code_screening(x=self.ekf.x[:3], satellites=gps_satellites[:, :3],
                                                         code_obs=C2, thr=code_prefit_threshold)

            gal_c1_mask, prefit_e1 = self.code_screening(x=self.ekf.x[:3], satellites=gal_satellites[:, :3],
                                                         code_obs=EC1, thr=code_prefit_threshold)
            gal_c2_mask, prefit_e2 = self.code_screening(x=self.ekf.x[:3], satellites=gal_satellites[:, :3],
                                                         code_obs=EC2, thr=code_prefit_threshold)

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
            # receiver clock (every code row) and Galileo ISB (state 4, its code rows) priors, RTKLIB udclk_ppp
            # (bug-fix stage 13b, A1(b))
            n_sat_all = len(curr_gps_sats) + len(curr_gal_sats)
            code_rows = (5 * np.arange(n_sat_all)[:, None] + np.array([0, 2])[None, :]).ravel()
            gal_code_rows = code_rows[2 * len(curr_gps_sats):]
            with epoch_process_noise(self.ekf, t, self.interval) as process_noise:
                for k in range(5):
                    try:
                        self.ekf.state_priors = clock_priors(self.cfg.clock_prior_sigma_m, code_rows,
                                                             {4: gal_code_rows})
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
                    P_pred_ar = (self.ekf.F @ ar_float_P @ self.ekf.F.T
                                 + gap_scaled_process_noise(self.ekf.Q, process_noise.step))
                    # the clock / ISB priors the filter applied to its own prediction (A1(b))
                    reset_prior_states(P_pred_ar, self.ekf.priors_applied)
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
                gps_arc_age = {}
                gal_arc_age = {}
                continue

            # PPP-069: one arc-age convention for every route (0 at the first epoch of an arc)
            gps_arc_age = advance_arc_age(gps_arc_age, curr_gps_sats, gps_phase_reset_svs)
            gal_arc_age = advance_arc_age(gal_arc_age, curr_gal_sats, gal_phase_reset_svs)
            gps_ar_age = arc_age_array(gps_arc_age, curr_gps_sats)
            gal_ar_age = arc_age_array(gal_arc_age, curr_gal_sats)

            ar_diag = None

            if ar_cfg.enabled and num >= ar_cfg.warmup_epochs:
                # float phase residuals and weights: rows 1 / 3 of each satellite carry N1 / N2 (C3, NQ-PPP-20)
                sat = np.arange(len(curr_gps_sats) + len(curr_gal_sats))[:, None]
                ar_inputs = ar_phase_kwargs(Z - self.Hx(self.ekf.x, gps_satellites, gal_satellites),
                                            np.diag(self.ekf.R), (5 * sat + np.array([1, 3])).ravel(),
                                            (self.base_dim + 3 * sat + np.array([1, 2])).ravel(), ekf=self.ekf)
                ar_float_state = keep_float_state(self.ekf, ar_cfg)
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
                    gps_eligible=ar_eligible_mask(gps_epoch),
                    gal_eligible=ar_eligible_mask(gal_epoch),
                    **ar_inputs,
                )

            dtr = self.ekf.x[3]
            isb = self.ekf.x[4]
            ztd = self.ekf.x[5]

            # ---- convergence time measure
            if not reset_epoch or reset_every == 0:
                if xyz is not None:
                    position_diff = np.linalg.norm(float_position(self.ekf, ar_float_state) - conv_xyz)

                    # If convergence time is not set and position difference is small, set the convergence time
                    if conv_time is None and position_diff < 0.005:
                        conv_time = t - T0  # Set the convergence time
                    elif position_diff > 0.005:  # Reset convergence time if position change exceeds threshold
                        conv_time = None  # Reset the convergence time

            # --- data collecting
            xyz = self.ekf.x[:3].copy()
            conv_xyz = float_position(self.ekf, ar_float_state)
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
                   'dcb_gal_c1': dcb[2], 'dcb_gal_c2': dcb[3], 'x': self.ekf.x[0], 'y': self.ekf.x[1],
                   'z': self.ekf.x[2],
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
        df_result = df_result.set_index(['time']) if len(df_result) else None

        def residual_frame(rows, data):
            # per-satellite rows of the run; an empty frame with the same columns when the system has none
            if not rows:
                return pd.DataFrame(np.empty((0, len(self.RESIDUAL_COLUMNS))), columns=list(self.RESIDUAL_COLUMNS),
                                    index=pd.MultiIndex.from_arrays([[], []], names=["sv", "time"]))
            return pd.DataFrame(np.vstack(data), index=rows[0].append(rows[1:]), columns=list(self.RESIDUAL_COLUMNS))

        df_obs_gps = residual_frame(gps_rows, gps_data)
        df_obs_gal = residual_frame(gal_rows, gal_data)

        if conv_time is not None and trace_filter:
            ppp_trace(
                trace_filter,
                self.__class__.__name__,
                "convergence",
                threshold_m=0.005,
                minutes=conv_time.total_seconds() / 60,
                hours=conv_time.total_seconds() / 3600,
            )
        # contracts C1 / C2 (bug-fix stage 13b, PPP-033 / PPP-034): {system: residual frame}; x, y, z (were xr, yr,
        # zr) and ct_min [min] (missing before)
        return (finish_solution(df_result, conv_time), {"G": df_obs_gps, "E": df_obs_gal}, None,
                convergence_hours(conv_time))
