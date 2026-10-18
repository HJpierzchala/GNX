"""PPP ambiguity-resolution helpers.

This module contains the float-to-fixed ambiguity workflow used by selected PPP
filters. It implements covariance stabilization, LAMBDA integer least-squares
search, ratio testing, lock-time/arc-age gating, optional elevation-based
candidate selection, partial fixing, fix validation and soft constraints
applied to the EKF state (for the reported epoch, or held with fix-and-hold).

Uncombined AR fixes the between-satellite single differences in two steps:
the wide lane N1 - N2 by rounding where that is safe, then N1 conditioned on
it (narrow lane) with LAMBDA; satellites without product phase biases and
phases flagged with a possible half-cycle ambiguity (RINEX LLI bit 1) are not
candidates. The validation tests use the float covariance scaled by the
a-posteriori variance factor of the float phase residuals when the filter
passes them (``phase_residuals`` / ``phase_weights``).

Status:
    Requires caution. These helpers can change numerical PPP behavior even when
    their signatures stay stable. Treat AR candidate selection, covariance
    source, ratio thresholds and hold constraints as model logic.

Combined ionosphere-free filters do not fix ambiguities: the ionosphere-free
ambiguity is an integer only together with the wide lane, which those filters
do not separate (``apply_conventional_pppar``). Uncombined AR is the PPP-AR
path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PPPARSettings:
    """Configuration for PPP ambiguity resolution attempts.

    Status:
        Active AR settings container. Combined IF AR still requires the extra
        experimental gate ``pppar_combined_if_ar_enabled``.

    ``enabled`` gates all AR updates. ``warmup_epochs`` and
    ``min_lock_epochs`` control when an ambiguity arc can become a candidate:
    the arc age (``advance_arc_age``) is the number of epochs of the arc before
    the current one, 0 at the first epoch of every arc (a satellite seen for
    the first time, a slip, a filter reset), so ``min_lock_epochs = N`` makes
    an ambiguity fixable from the (N+1)-th epoch of its arc (``None`` takes
    ``warmup_epochs``).
    ``min_ambiguities``, ``min_candidate_elevation_deg`` and
    ``wide_lane_max_frac_cycles`` screen candidate groups. ``ratio_threshold``
    applies the LAMBDA ratio test. Accepted fixes are applied as soft
    constraints with ``constraint_sigma_cycles`` subject to the optional floor;
    that sigma weights only the constraint applied to the filter state. The
    narrow-lane search is conditioned on the wide-lane integers with the
    internal ``WIDE_LANE_CONDITION_SIGMA_CYCLES`` whatever the setting.

    ``partial_fixing_enabled`` (default on, as ``PPPConfig.pppar_partial_fixing_enabled``)
    allows mature subsets to be fixed when younger ambiguities are present, and
    lets the uncombined engine fix the subset of decorrelated ambiguities whose
    bootstrapped success rate reaches ``PARTIAL_MIN_SUCCESS_RATE`` when the full
    set cannot be fixed. Combined ionosphere-free filters fix nothing (see
    ``apply_conventional_pppar``).

    Fix validation (always on): besides the ratio test, the fixed integer
    combinations must reach the bootstrapped success rate
    ``PARTIAL_MIN_SUCCESS_RATE``, each lie within
    ``NARROW_LANE_MAX_FRAC_CYCLES`` of its float value, be statistically
    consistent with the float ambiguities (squared distance in the metric of
    their covariance below the ``1 - FIX_TEST_ALPHA`` chi-square quantile) and
    the shift of the fixed position must stay within the chi-square(3) quantile
    of the float position covariance; every test uses the float covariance
    scaled by ``FIX_TEST_VARIANCE_FACTOR`` times the a-posteriori variance factor
    of the epoch's float phase residuals (``phase_variance_factor``, floored at
    1; 1 when the filter passes no residuals). ``hold``
    feeds accepted fixes back into the filter (fix-and-hold); without it the
    filter stays float and every epoch is fixed anew from the float state, so
    a wrong fix cannot outlive its epoch.
    """

    enabled: bool = False
    warmup_epochs: int = 60
    min_ambiguities: int = 4
    ratio_threshold: float = 2.0
    constraint_sigma_cycles: float = 1e-3
    constraint_sigma_floor_cycles: Optional[float] = 1e-3
    max_condition_number: float = 1e12
    min_lock_epochs: Optional[int] = None
    min_candidate_elevation_deg: Optional[float] = None
    use_float_ratio_covariance: bool = True
    partial_fixing_enabled: bool = True
    partial_min_ambiguities: Optional[int] = None
    wide_lane_max_frac_cycles: Optional[float] = 0.25
    hold: bool = False


@dataclass(slots=True)
class PPPARDiagnostics:
    """Diagnostics returned by one PPP-AR update attempt.

    Status:
        Active diagnostics container. The flat result columns generated from
        this object are part of PPP validation output.

    The object records whether any group was accepted, the minimum accepted
    ratio, how many ambiguity groups were attempted/accepted/rejected, and a
    reason-count map suitable for stable result columns. ``attempts`` keeps
    compact per-group details for debugging and validation.
    """

    fixed_ambiguities: int = 0
    ratio_min: Optional[float] = None
    accepted: bool = False
    attempted_groups: int = 0
    accepted_groups: int = 0
    rejected_groups: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    attempts: list[dict[str, object]] = field(default_factory=list)
    partial_groups: int = 0
    full_groups: int = 0
    partial_fixed_ambiguities: int = 0
    excluded_no_phase_bias: int = 0
    excluded_mixed_products: int = 0
    excluded_wide_lane: int = 0
    excluded_partial: int = 0
    excluded_half_cycle: int = 0
    #: a-posteriori variance factor of the float phase residuals used by the position check (all groups together;
    #: None when the filter passed no residuals): ``phase_variance_factor`` floored at 1
    phase_variance_factor: Optional[float] = None


#: significance level of the fix validation tests (float-vs-fixed ambiguity distance and fixed-vs-float position)
FIX_TEST_ALPHA = 1e-3
#: variance factor of the float covariance, applied alike in every validation test (float-vs-fixed distance,
#: success rate, wide-lane sigma gate, position check). The filter covariance of the float ambiguities is
#: optimistic (BRUX 2024-035, GPS: float-minus-fixed narrow-lane residuals 1.8 times their formal sigma), so with
#: 1 the distance test rejects part of the good fixes, which errs on the safe side. Scaling it up (4) admitted
#: fixes 1-2 h earlier whose epochs were no better than float in 3D (height degraded: G 6 h 4.45 against 4.46 cm
#: 3D RMS, G without ionospheric constraints 4.9 against 3.6 cm); with 1 and the narrow-lane gate the fixed
#: epochs beat float in 3D on every static BRUX case (G 6 h 3.2 against 3.5 cm). Since v1.0.0 the
#: tests multiply it by the a-posteriori variance factor of the epoch's float phase residuals when the filter passes
#: them (``phase_variance_factor``, floored at 1): a float solution whose phase residuals are larger than
#: their weights say has a float covariance too optimistic by that factor
FIX_TEST_VARIANCE_FACTOR = 1.0
#: sigma (cycles) of the wide-lane integers when the narrow-lane float ambiguities are conditioned on them before the
#: LAMBDA search and the validation tests. Internal and tight: against a rounded wide lane (standard
#: deviation <= WIDE_LANE_MAX_SIGMA_CYCLES = 0.15) it leaves 1e-3 / 0.15 = 0.7 % of the wide-lane sigma, so the
#: narrow-lane covariance is the conditional one. ``pppar_constraint_sigma_cycles`` (plan range 1e-6..0.1) weights
#: only the constraint applied to the filter state; before v1.0.0 it also conditioned the search, so 0.1 cycles
#: left the wide lane practically float and no epoch could be fixed (GPS-only uncombined, BRUX 3 h: 0 fixed epochs, 104 with the
#: default). Equal to the default constraint sigma and floor, so the default runs are unchanged
WIDE_LANE_CONDITION_SIGMA_CYCLES = 1e-3
#: a wide-lane float ambiguity is rounded only when its standard deviation is below this (cycles): the
#: probability of rounding to the wrong integer is then below 1e-3 (2 * Phi(-0.5 / 0.15))
WIDE_LANE_MAX_SIGMA_CYCLES = 0.15
#: narrow-lane gate: every fixed integer combination must lie within this of its float value (cycles), as the
#: wide-lane fraction gate (``wide_lane_max_frac_cycles``, 0.25); a larger residual means the float state is
#: biased (unconverged troposphere, unmodelled errors) even when the ratio test passes
NARROW_LANE_MAX_FRAC_CYCLES = 0.25
#: partial fixing keeps the decorrelated ambiguities whose bootstrapped success rate reaches this
PARTIAL_MIN_SUCCESS_RATE = 0.999
#: name of the per-observation column (bool) set by PPPSession: the satellite has product phase biases for the
#: phase observables of its mode, so its ambiguities can be integer
AR_PHASE_BIAS_COLUMN = "ar_phase_bias"
#: name of the per-observation column (bool) set by PPPSession when PPP-AR is disabled because the phase OSBs
#: come from another solution than the orbits and clocks: ``AR_PHASE_BIAS_COLUMN`` is then False
#: everywhere and the exclusion is reported as ``mixed_products`` instead of ``no_phase_bias``
AR_MIXED_PRODUCTS_COLUMN = "ar_mixed_products"
#: name of the per-observation column (bool) set by PPPSession from the RINEX LLI frames :
#: True when LLI bit 1 (half-cycle ambiguity possible; RINEX 3.03 §5.4 / Table A3, 4.00 §6.7.1, 2.11 Table A2) is set
#: on a phase of the row. Such an ambiguity is kept out of integer fixing while the flag is set, as RTKLIB's relative
#: positioning does (rtkpos.c ddmat: ``!(ssat.slip[f]&2)`` keeps a satellite whose LLI bit 1 is set out of the fixed
#: set, the flag being stored by detslp_ll; RTKLIB 2.4.2 p13 rtkpos.c:585, :592 and :1280/:1292; RTKLIB PPP resets the
#: ambiguity instead, ppp.c:714); the float solution is unaffected and the cycle-slip detector starts a new arc when
#: the flag sets or clears (gnx_py.cycle_slips.half_cycle_breaks). A missing column means no flag (all False)
AR_HALF_CYCLE_COLUMN = "lli_half_cycle"
#: a phase residual whose weight is at or below this [1/m^2] (variance >= 1e10 m^2, the variance the filters give a
#: rejected observation, 1e12 m^2) does not enter the a-posteriori variance factor
PHASE_WEIGHT_REJECTED = 1e-10
#: fewest valid phase residuals for an a-posteriori variance factor; with fewer the factor is 1
PHASE_VARIANCE_MIN_DOF = 2

COMMON_AR_REASONS = (
    "missing_n1",
    "below_min_ambiguities",
    "elevation_size_mismatch",
    "n2_size_mismatch",
    "n1_index_oob",
    "n2_index_oob",
    "lock_age_size_mismatch",
    "young_arc",
    "candidate_selection_gate",
    "no_reference_satellite",
    "wide_lane_gate",
    "ill_conditioned",
    "lambda_error",
    "ratio_covariance_invalid",
    "ratio_reject",
    "no_phase_bias",
    "mixed_products",
    "half_cycle",
    "eligible_size_mismatch",
    "success_rate",
    "narrow_lane_gate",
    "fix_validation",
    "position_check",
    "if_not_integer",
    "accepted",
)


def _record_attempt(diag: PPPARDiagnostics, reason: str, **details: object) -> None:
    diag.attempted_groups += 1
    diag.reason_counts[reason] = diag.reason_counts.get(reason, 0) + 1
    if reason == "accepted":
        diag.accepted_groups += 1
    else:
        diag.rejected_groups += 1
    attempt = {"reason": reason}
    attempt.update(details)
    diag.attempts.append(attempt)


def pppar_diagnostic_columns(diag: Optional[PPPARDiagnostics]) -> dict[str, object]:
    """Return stable, flat PPP-AR diagnostic columns for result DataFrames.

    Missing diagnostics are represented by zeros/``None`` so result schemas do
    not change when AR is disabled, rejected, or not attempted in an epoch.
    """
    row: dict[str, object] = {
        "ar_attempted_groups": 0,
        "ar_accepted_groups": 0,
        "ar_rejected_groups": 0,
        "ar_last_reason": None,
        "ar_partial_groups": 0,
        "ar_full_groups": 0,
        "ar_partial_fixed_ambiguities": 0,
        "ar_excluded_no_phase_bias": 0,
        "ar_excluded_mixed_products": 0,
        "ar_excluded_wide_lane": 0,
        "ar_excluded_partial": 0,
        "ar_excluded_half_cycle": 0,
        "ar_phase_variance_factor": np.nan,
    }
    for reason in COMMON_AR_REASONS:
        row[f"ar_reason_{reason}"] = 0
    if diag is None:
        return row

    row["ar_attempted_groups"] = int(diag.attempted_groups)
    row["ar_accepted_groups"] = int(diag.accepted_groups)
    row["ar_rejected_groups"] = int(diag.rejected_groups)
    row["ar_last_reason"] = diag.attempts[-1]["reason"] if diag.attempts else None
    row["ar_partial_groups"] = int(diag.partial_groups)
    row["ar_full_groups"] = int(diag.full_groups)
    row["ar_partial_fixed_ambiguities"] = int(diag.partial_fixed_ambiguities)
    row["ar_excluded_no_phase_bias"] = int(diag.excluded_no_phase_bias)
    row["ar_excluded_mixed_products"] = int(diag.excluded_mixed_products)
    row["ar_excluded_wide_lane"] = int(diag.excluded_wide_lane)
    row["ar_excluded_partial"] = int(diag.excluded_partial)
    row["ar_excluded_half_cycle"] = int(diag.excluded_half_cycle)
    if diag.phase_variance_factor is not None:
        row["ar_phase_variance_factor"] = float(diag.phase_variance_factor)
    for reason, count in diag.reason_counts.items():
        row[f"ar_reason_{reason}"] = int(count)
    return row



def _effective_min_lock_epochs(settings: PPPARSettings) -> int:
    if settings.min_lock_epochs is None:
        return max(0, int(settings.warmup_epochs))
    return max(0, int(settings.min_lock_epochs))


def _effective_constraint_sigma_cycles(settings: PPPARSettings) -> float:
    sigma = float(settings.constraint_sigma_cycles)
    if settings.constraint_sigma_floor_cycles is None:
        return sigma
    return max(sigma, float(settings.constraint_sigma_floor_cycles))


def _effective_partial_min_ambiguities(settings: PPPARSettings) -> int:
    if settings.partial_min_ambiguities is None:
        return int(settings.min_ambiguities)
    return max(int(settings.min_ambiguities), int(settings.partial_min_ambiguities))


def combined_if_pppar_enabled(config: object) -> bool:
    """Return whether the combined ionosphere-free filters call their AR step.

    ``pppar_enabled`` alone is not enough for combined IF filters; the separate
    ``pppar_combined_if_ar_enabled`` flag is also needed. The step fixes no
    ambiguity (``apply_conventional_pppar``: the ionosphere-free ambiguity is
    not an integer without the wide lane); it only records the reason.

    Status:
        Feature gate. Do not remove or bypass this guard without a wide-lane /
        narrow-lane ionosphere-free AR and its validation.
    """
    return bool(getattr(config, "pppar_enabled", False)) and bool(
        getattr(config, "pppar_combined_if_ar_enabled", False)
    )


def ar_eligible_mask(epoch: object) -> Optional[np.ndarray]:
    """Per-satellite AR eligibility of one epoch block (rows in filter order).

    ``True`` where the satellite carries product phase biases for the phase observables of its mode
    (``AR_PHASE_BIAS_COLUMN``, set by ``PPPSession``) and no phase of the row carries the RINEX half-cycle flag
    (``AR_HALF_CYCLE_COLUMN``, LLI bit 1). ``None`` when the block has neither column (filters used
    outside a session keep every satellite as a candidate). A missing phase-bias column counts as phase biases
    present, a missing half-cycle column (or NaN) as no flag. The returned mask carries ``half_cycle``: the rows
    excluded by the half-cycle flag alone (``apply_indexed_uncombined_pppar`` reports them as ``half_cycle``).
    """
    columns = getattr(epoch, "columns", None)
    if columns is None:
        return None
    has_bias = AR_PHASE_BIAS_COLUMN in columns
    has_half = AR_HALF_CYCLE_COLUMN in columns
    if not has_bias and not has_half:
        return None
    if has_bias:
        values = epoch[AR_PHASE_BIAS_COLUMN].to_numpy()
        mask = np.asarray([bool(v) if v == v else False for v in values], dtype=bool)
    else:
        mask = np.ones(len(epoch), dtype=bool)
    mask = mask.view(_EligibleMask)
    if has_bias and AR_MIXED_PRODUCTS_COLUMN in columns and (
            epoch[AR_MIXED_PRODUCTS_COLUMN].fillna(False).astype(bool).any()):
        # the exclusion is reported as mixed_products (NQ-PPP-24); the filters pass the mask on unchanged
        mask.exclusion_reason = "mixed_products"
    if has_half:
        half = _bool_column(epoch[AR_HALF_CYCLE_COLUMN].to_numpy())
        mask.half_cycle = half & np.asarray(mask)
        mask &= ~half
    return mask


def _bool_column(values: np.ndarray) -> np.ndarray:
    """Per-row bool of an object/bool/float column: NaN / None count as False."""
    return np.asarray([bool(v) if v is not None and v == v else False for v in values], dtype=bool)


class _EligibleMask(np.ndarray):
    """Boolean AR eligibility mask that carries the reason of its exclusions (``exclusion_reason``) and the rows
    excluded by the half-cycle flag alone (``half_cycle``, None when the block had no such column)."""

    exclusion_reason = "no_phase_bias"
    half_cycle: Optional[np.ndarray] = None


def keep_float_state(ekf: object, settings: PPPARSettings) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Float state to restore after the epoch's output when fixes are not held (``settings.hold`` off).

    Callers take it just before the AR update and pass it to ``restore_float_state`` at the start of the
    next epoch: the epoch is reported with the fixed solution while the filter continues from the float one.
    """
    if not settings.enabled or settings.hold:
        return None
    return np.array(ekf.x, dtype=float, copy=True), np.array(ekf.P, dtype=float, copy=True)


def restore_float_state(ekf: object, float_state: Optional[tuple[np.ndarray, np.ndarray]]) -> None:
    """Put back the float state kept by ``keep_float_state`` (no-op when it is ``None``)."""
    if float_state is None:
        return
    x, P = float_state
    if np.shape(ekf.x) == np.shape(x) and np.shape(ekf.P) == np.shape(P):
        ekf.x, ekf.P = x, P


def float_position(ekf: object, float_state: Optional[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Position of the float filter state: the one kept by ``keep_float_state`` when this epoch is reported
    fixed, else the filter's own (float, or held with fix-and-hold). The filters' convergence check follows
    it, so an epoch-to-epoch switch between fixed and float output does not restart the convergence time."""
    x = ekf.x if float_state is None else float_state[0]
    return np.array(x[:3], dtype=float, copy=True)


def advance_arc_age(
    age_by_key: dict[object, int],
    current_keys: Sequence[object],
    reset_keys: Optional[Sequence[object] | set[object]] = None,
) -> dict[object, int]:
    """Advance per-ambiguity age for the current epoch.

    The age is the number of epochs of the arc before the current one: 0 at the
    first epoch of every arc. Keys absent from ``current_keys`` disappear from
    the returned mapping (a satellite that comes back starts a new arc). Keys
    present in ``reset_keys`` (a slip or a re-initialised ambiguity) and keys
    seen for the first time start a new arc at age zero; all other current keys
    increment from their previous value. ``min_lock_epochs = N`` therefore makes
    an ambiguity fixable from the (N+1)-th epoch of its arc, for a new and a
    restarted arc alike (before v1.0.0 a new key started at 1,
    a restarted one at 0, so the gate let the first be fixed one epoch earlier).
    """
    reset = set(reset_keys or ())
    return {
        key: 0 if key in reset or key not in age_by_key else int(age_by_key[key]) + 1
        for key in current_keys
    }


def arc_age_array(age_by_key: dict[object, int], current_keys: Sequence[object]) -> np.ndarray:
    """Return arc ages aligned with the current ambiguity ordering."""
    return np.asarray([int(age_by_key.get(key, 0)) for key in current_keys], dtype=int)


def update_redundancy(ekf: object) -> Optional[np.ndarray]:
    """Local redundancy numbers of the rows of the filter's last measurement update, ``r_i = R_ii (S^-1)_ii``.

    For a Kalman update with the optimal gain the postfit residuals ``v = (I - H K) y`` have the covariance
    ``R - H P+ H' = R S^-1 R`` (S = H P- H' + R, the innovation covariance), so with a diagonal R the expected
    ``v_i^2 / R_ii`` is ``r_i = 1 - (H P+ H')_ii / R_ii = R_ii (S^-1)_ii``, between 0 (the row is fitted
    exactly) and 1 (the row adds nothing to the state): the redundancy number of the row, whose sum over the rows
    is the degrees of freedom of the update (Foerstner 1979; Teunissen, Adjustment theory, 2003, ch. 1.3, the
    diagonal of Q_vv W). Reads ``ekf.R`` (m x m) and ``ekf.S``; ``ekf.rows_used`` (a filter that leaves the
    rejected rows out of the update: S is then over those rows) when the filter sets it. Returns an array over every
    measurement row (NaN for a row left out of the update) or ``None`` when the filter has no usable S.
    """
    R = np.asarray(getattr(ekf, "R", None), dtype=float)
    S = getattr(ekf, "S", None)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or S is None:
        return None
    S = np.asarray(S, dtype=float)
    m = R.shape[0]
    rows = getattr(ekf, "rows_used", None)
    rows = np.arange(m) if rows is None else np.asarray(rows, dtype=int).ravel()
    if S.ndim != 2 or S.shape != (rows.size, rows.size) or (rows.size and (rows.min() < 0 or rows.max() >= m)):
        return None
    out = np.full(m, np.nan)
    if not rows.size:
        return out
    try:
        L_inv = np.linalg.inv(np.linalg.cholesky(0.5 * (S + S.T)))
    except np.linalg.LinAlgError:
        return None
    s_inv_diag = np.sum(L_inv * L_inv, axis=0)  # diag(S^-1) = diag(L^-T L^-1)
    out[rows] = np.clip(np.diag(R)[rows] * s_inv_diag, 0.0, 1.0)
    return out


def ar_phase_series(
    residuals: Sequence[float],
    variances: Sequence[float],
    phase_rows: Sequence[int],
    ambiguity_idx: Sequence[int],
    redundancy: Optional[Sequence[float]] = None,
) -> tuple[pd.Series, pd.Series]:
    """The ``phase_residuals`` / ``phase_weights`` arguments of the AR entry points from one float update.

    ``residuals`` are the a-posteriori (postfit) residuals of the epoch's float update, ``z - h(x_float)`` [m], and
    ``variances`` the diagonal of its observation covariance R [m^2], both over every measurement row;
    ``phase_rows`` are the rows of the phase observations and ``ambiguity_idx`` (same length) the state index of the
    ambiguity each of them carries (N1 for the first carrier, N2 for the second). Returns the residuals and their
    weights [1/m^2] indexed by that state index. Rows the filter rejected (R >= 1e10 m^2) get a weight at or below
    ``PHASE_WEIGHT_REJECTED`` and are ignored by the variance factor.

    ``redundancy`` (over every row, ``update_redundancy`` of the same update) gives the variance factor the degrees
    of freedom of the phase residuals (s0^2 = v'Pv / dof): the weights 1/R of the n
    valid phase rows are scaled by n / sum(r_i) over those rows, so that the factor ``sum(w v^2) / n`` of
    ``phase_variance_factor`` is ``v'R^-1 v / sum(r_i)``. One global share for every system: they share the
    position, troposphere and clock states (per-row 1/(R r_i) is unbiased too but unstable, a row with r near 0
    and a tiny residual dominates it). With less than ``PHASE_VARIANCE_MIN_DOF`` degrees of freedom every weight is
    0 (no factor). Without it the weights are 1/R and the factor ``v'R^-1 v / n``, biased low by the redundancy
    share (about 0.44 on the GPS-only uncombined validation days).
    """
    rows = np.asarray(phase_rows, dtype=int).ravel()
    idx = np.asarray(ambiguity_idx, dtype=int).ravel()
    if rows.shape != idx.shape:
        raise ValueError(f"phase_rows ({rows.size}) and ambiguity_idx ({idx.size}) must have the same length")
    v = np.asarray(residuals, dtype=float).ravel()[rows]
    var = np.asarray(variances, dtype=float).ravel()[rows]
    w = np.zeros_like(var)
    ok = np.isfinite(var) & (var > 0.0)
    w[ok] = 1.0 / var[ok]
    if redundancy is not None:
        r = np.asarray(redundancy, dtype=float).ravel()[rows]
        valid = np.isfinite(v) & (w > PHASE_WEIGHT_REJECTED) & np.isfinite(r)
        dof = float(np.sum(r[valid]))
        w[~valid] = 0.0
        if dof >= PHASE_VARIANCE_MIN_DOF:
            w[valid] *= np.count_nonzero(valid) / dof
        else:
            w[:] = 0.0
    return pd.Series(v, index=idx, dtype=float), pd.Series(w, index=idx, dtype=float)


def _phase_residual_arrays(
    phase_residuals: Optional[pd.Series], phase_weights: Optional[pd.Series]
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Validated (state index, residual, weight) of the valid phase residuals; ``None`` without residuals."""
    if phase_residuals is None and phase_weights is None:
        return None
    if phase_residuals is None or phase_weights is None:
        raise ValueError("phase_residuals and phase_weights must be given together")
    if not isinstance(phase_residuals, pd.Series) or not isinstance(phase_weights, pd.Series):
        raise TypeError("phase_residuals and phase_weights must be pandas Series indexed by the ambiguity state "
                        "index (pppar.ar_phase_series)")
    for series in (phase_residuals, phase_weights):
        if len(series) and not _integer_index(series.index):
            raise TypeError("phase_residuals / phase_weights must be indexed by the integer state index of the "
                            "ambiguity each phase carries (pppar.ar_phase_series), not by satellite or row labels")
    if not phase_residuals.index.is_unique or not phase_weights.index.is_unique:
        raise ValueError("phase_residuals / phase_weights: duplicate ambiguity state indices")
    w_aligned = phase_weights.reindex(phase_residuals.index)
    idx = np.asarray(phase_residuals.index, dtype=float)
    v = pd.to_numeric(phase_residuals, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(w_aligned, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(idx) & np.isfinite(v) & np.isfinite(w) & (w > PHASE_WEIGHT_REJECTED)
    return idx[ok].astype(int), v[ok], w[ok]


def _integer_index(index: pd.Index) -> bool:
    if pd.api.types.is_integer_dtype(index.dtype):
        return True
    if not pd.api.types.is_numeric_dtype(index.dtype) or pd.api.types.is_bool_dtype(index.dtype):
        return False
    values = np.asarray(index, dtype=float)
    return bool(np.all(np.isfinite(values) & (values == np.round(values))))


def phase_variance_factor(
    phase_residuals: Optional[pd.Series],
    phase_weights: Optional[pd.Series],
    state_idx: Optional[Sequence[int]] = None,
) -> tuple[float, int]:
    """A-posteriori variance factor of the float phase residuals and the number of residuals used.

    s0^2 = sum(w_i v_i^2) / n over the n valid residuals (finite, weight above ``PHASE_WEIGHT_REJECTED``) whose
    ambiguity state index is in ``state_idx`` (every residual when ``None``), floored at 1: the validation tests
    then use the float covariance scaled by s0^2 when the residuals are larger than their weights say, and the
    covariance as it is otherwise (a filter's phase weights are not made more optimistic). With the weights of
    ``ar_phase_series(..., redundancy=update_redundancy(ekf))`` (1/R times n / sum of the redundancy numbers) this is
    v'R^-1 v / dof, the definition used here; with plain 1/R weights it is v'R^-1 v / n, biased low by the redundancy
    share (the postfit residuals of a Kalman update have E[v'R^-1 v] = n - tr(R^-1 H P+ H')). Returns ``(1.0, n)``
    with fewer than ``PHASE_VARIANCE_MIN_DOF`` residuals or without residuals (``n`` = 0).
    """
    return _variance_factor(_phase_residual_arrays(phase_residuals, phase_weights), state_idx)


def _variance_factor(
    arrays: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]], state_idx: Optional[Sequence[int]]
) -> tuple[float, int]:
    if arrays is None:
        return 1.0, 0
    idx, v, w = arrays
    if state_idx is not None:
        keep = np.isin(idx, np.asarray(state_idx, dtype=int))
        v, w = v[keep], w[keep]
    n = int(v.size)
    if n < PHASE_VARIANCE_MIN_DOF:
        return 1.0, n
    s0_sq = float(np.sum(w * v * v) / n)
    return (max(1.0, s0_sq) if np.isfinite(s0_sq) else 1.0), n


def _stabilize_cov(Q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    Qs = 0.5 * (Q + Q.T)
    jitter = eps
    for _ in range(6):
        try:
            np.linalg.cholesky(Qs)
            return Qs
        except np.linalg.LinAlgError:
            Qs = Qs + np.eye(Qs.shape[0]) * jitter
            jitter *= 10.0
    return Qs


# ---------------------------------------------------------------------------------------------------------------
# LAMBDA integer least squares: a Python port of RTKLIB 2.4.2 src/lambda.c (T. Takasu; BSD 2-clause licence,
# reproduced in THIRD_PARTY_NOTICES): the LD factorization, the integer Gauss transformations and permutations of
# the LAMBDA reduction, and the modified LAMBDA (MLAMBDA) search. References: P.J.G. Teunissen (1995), The
# least-squares ambiguity decorrelation adjustment: a method for fast GPS integer ambiguity estimation, J. Geod.
# 70, 65-82; P. de Jonge and C. Tiberius (1996), The LAMBDA method for integer ambiguity estimation:
# implementation aspects, LGR series 12, Delft University of Technology; X.-W. Chang, X. Yang and T. Zhou (2005),
# MLAMBDA: a modified LAMBDA method for integer least-squares estimation, J. Geod. 79, 552-565.
# Differences from lambda.c: NumPy row/column indexing instead of column-major arrays (the row update of a
# permutation as one 2 x 2 matrix product), D returned as a (1, n) row, the permutation test without lambda.c's
# 1e-6 margin (the strict test of de Jonge and Tiberius 1996, as GNX-py used before v1.0.0), the conditional
# estimates of the search summed directly instead of through lambda.c's matrix S, the reduction also returns the
# inverse transposed transformation, the fixed candidates are mapped back by the caller
# (``_lambda_search``), and the search raises ``LinAlgError`` instead of returning an error code.
# ---------------------------------------------------------------------------------------------------------------

_LAMBDA_LOOPMAX = 1_000_000  # search iterations before the search gives up (lambda.c: LOOPMAX 10000)


def _round_half_up(x: float) -> float:
    """lambda.c ROUND(x): floor(x + 0.5)."""
    return float(np.floor(x + 0.5))


def _sgn(x: float) -> float:
    """lambda.c SGN(x): -1 for x <= 0, else 1."""
    return -1.0 if x <= 0.0 else 1.0


def ldldecom(Qahat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LD factorization ``Q = L^T diag(D) L`` of an ambiguity covariance (lambda.c ``LD``).

    ``L`` is unit lower triangular (n x n); ``D`` is returned as a (1, n) row. Raises ``LinAlgError`` when a
    pivot is not positive (the covariance is not positive definite).
    """
    A = np.array(Qahat, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    D = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        D[i] = A[i, i]
        if not D[i] > 0.0:
            raise np.linalg.LinAlgError("LD factorization error: the ambiguity covariance is not positive definite")
        a = np.sqrt(D[i])
        for j in range(i + 1):
            L[i, j] = A[i, j] / a
        for j in range(i):
            for k in range(j + 1):
                A[j, k] -= L[i, k] * L[i, j]
        for j in range(i + 1):
            L[i, j] /= L[i, i]
    return L, D.reshape(1, n)


def _gauss(L: np.ndarray, Z: np.ndarray, i: int, j: int) -> None:
    """Integer Gauss transformation of column ``j`` by column ``i`` (lambda.c ``gauss``)."""
    mu = _round_half_up(L[i, j])
    if mu != 0.0:
        n = L.shape[0]
        for k in range(i, n):
            L[k, j] -= mu * L[k, i]
        Z[:, j] -= mu * Z[:, i]


def _perm(L: np.ndarray, D: np.ndarray, j: int, delta: float, Z: np.ndarray) -> None:
    """Permutation of the ambiguities ``j`` and ``j + 1`` (lambda.c ``perm``)."""
    n = L.shape[0]
    eta = D[j] / delta
    lam = D[j + 1] * L[j + 1, j] / delta
    D[j] = eta * D[j + 1]
    D[j + 1] = delta
    if j > 0:  # lambda.c: L[j,k] = -L[j+1,j] L[j,k] + L[j+1,k], L[j+1,k] = eta L[j,k] + lam L[j+1,k] (k < j)
        L[j:j + 2, :j] = np.array([[-L[j + 1, j], 1.0], [eta, lam]]) @ L[j:j + 2, :j]
    L[j + 1, j] = lam
    for k in range(j + 2, n):
        L[k, j], L[k, j + 1] = L[k, j + 1], L[k, j]
    Z[:, [j, j + 1]] = Z[:, [j + 1, j]]


def decorrel(Qahat: np.ndarray, ahat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LAMBDA reduction (decorrelation) of float ambiguities (lambda.c ``reduction``).

    Returns ``(Qzhat, Z, L, D, zhat, iZt)``: the unimodular transformation ``Z`` with ``zhat = Z^T ahat`` and
    ``Qzhat = Z^T Qahat Z = L^T diag(D) L``, the factors ``L`` and ``D`` (a (1, n) row) of ``Qzhat``, and
    ``iZt = (Z^T)^-1`` (integer), which maps decorrelated integers back: ``a = iZt z``.
    """
    Q = np.array(Qahat, dtype=float)
    n = Q.shape[0]
    L, D2 = ldldecom(Q)
    D = D2.reshape(-1).copy()
    Z = np.eye(n, dtype=float)
    j = n - 2
    k = n - 2
    while j >= 0:
        if j <= k:
            for i in range(j + 1, n):
                _gauss(L, Z, i, j)
        delta = D[j] + L[j + 1, j] ** 2 * D[j + 1]
        if delta < D[j + 1]:  # strict test of de Jonge and Tiberius (1996); lambda.c adds a 1e-6 margin
            _perm(L, D, j, delta, Z)
            k = j
            j = n - 2
        else:
            j -= 1
    iZt = np.round(np.linalg.inv(Z.T))
    Qzhat = Z.T @ Q @ Z
    zhat = Z.T @ np.asarray(ahat, dtype=float)
    return Qzhat, Z, L, D.reshape(1, n), zhat, iZt


def ssearch(ahat: np.ndarray, L: np.ndarray, D: np.ndarray, ncands: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """MLAMBDA search of the ``ncands`` best integer vectors (lambda.c ``search``).

    ``ahat``: the decorrelated float ambiguities, ``L`` / ``D`` their LD factors (``decorrel``). Returns the
    candidates as columns (n x ncands) and their squared norms ``(zhat - z)^T Qzhat^-1 (zhat - z)``, both in
    ascending order of the norm.
    """
    zs = np.asarray(ahat, dtype=float).reshape(-1)
    Dv = np.asarray(D, dtype=float).reshape(-1)
    n = zs.size
    m = int(ncands)
    dist = np.zeros(n, dtype=float)
    zb = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    step = np.zeros(n, dtype=float)
    zn = np.zeros((n, m), dtype=float)
    s = np.zeros(m, dtype=float)
    nn = 0
    imax = 0
    maxdist = 1e99
    k = n - 1
    dist[k] = 0.0
    zb[k] = zs[k]
    z[k] = _round_half_up(zb[k])
    y = zb[k] - z[k]
    step[k] = _sgn(y)
    for _ in range(_LAMBDA_LOOPMAX):
        newdist = dist[k] + y * y / Dv[k]
        if newdist < maxdist:
            if k != 0:
                k -= 1
                dist[k] = newdist
                s_k = 0.0  # lambda.c keeps these sums in a matrix S; summed directly here (same terms, ascending i)
                for i in range(k + 1, n):
                    s_k += L[i, k] * (z[i] - zb[i])
                zb[k] = zs[k] + s_k
                z[k] = _round_half_up(zb[k])
                y = zb[k] - z[k]
                step[k] = _sgn(y)
            else:
                if nn < m:
                    if nn == 0 or newdist > s[imax]:
                        imax = nn
                    zn[:, nn] = z
                    s[nn] = newdist
                    nn += 1
                else:
                    if newdist < s[imax]:
                        zn[:, imax] = z
                        s[imax] = newdist
                        imax = int(np.argmax(s))
                    maxdist = s[imax]
                z[0] += step[0]
                y = zb[0] - z[0]
                step[0] = -step[0] - _sgn(step[0])
        else:
            if k == n - 1:
                break
            k += 1
            z[k] += step[k]
            y = zb[k] - z[k]
            step[k] = -step[k] - _sgn(step[k])
    else:
        raise np.linalg.LinAlgError("LAMBDA search failed: iteration limit reached")
    order = np.argsort(s, kind="stable")
    return zn[:, order], s[order]


def _lambda_search(float_amb: np.ndarray, Q_amb: np.ndarray, ncands: int = 2) -> tuple[np.ndarray, float, float]:
    """LAMBDA ILS: best integer vector, ratio second-best/best and the best squared norm
    (float - fixed)^T Q^-1 (float - fixed)."""
    if float_amb.ndim != 1:
        raise ValueError("float_amb must be a 1D vector")
    Q = _stabilize_cov(Q_amb)
    if np.linalg.cond(Q) > 1e16:
        raise np.linalg.LinAlgError("Ambiguity covariance is ill-conditioned")
    frac, incr = np.modf(float_amb)
    _, _, L, D, zhat, iZt = decorrel(Q, frac)
    zfixed, sqnorm = ssearch(zhat, L, D, ncands=max(2, ncands))
    afixed = iZt.dot(zfixed) + incr.reshape(-1, 1)
    best = afixed[:, 0]
    if sqnorm.size >= 2 and sqnorm[0] > 0.0:
        ratio = float(sqnorm[1] / sqnorm[0])
    else:
        ratio = float("inf")
    return best, ratio, float(sqnorm[0])


def lambda_ils(float_amb: np.ndarray, Q_amb: np.ndarray, ncands: int = 2) -> tuple[np.ndarray, float]:
    """Resolve float ambiguities with LAMBDA ILS and return best fix plus ratio.

    The ratio is computed from the two best squared norms. The function raises
    ``LinAlgError`` for unstable covariance/search cases; callers translate
    those failures into PPP-AR diagnostics rather than changing filter state.
    """
    best, ratio, _ = _lambda_search(float_amb, Q_amb, ncands=ncands)
    return best, ratio


def _chi2_limit(dof: int, factor: Optional[float] = None) -> float:
    """``1 - FIX_TEST_ALPHA`` quantile of the chi-square distribution with ``dof`` degrees of freedom, scaled
    by ``factor`` (default ``FIX_TEST_VARIANCE_FACTOR``)."""
    from scipy.stats import chi2

    scale = FIX_TEST_VARIANCE_FACTOR if factor is None else float(factor)
    return scale * float(chi2.ppf(1.0 - FIX_TEST_ALPHA, max(int(dof), 1)))


def _position_shift_ok(x_float: np.ndarray, P_float: np.ndarray, x_fixed: np.ndarray,
                       variance_factor: float = 1.0) -> tuple[bool, float]:
    """Fixed-vs-float position check: the shift of the fixed position, in the metric of the float position
    covariance (scaled by ``FIX_TEST_VARIANCE_FACTOR`` times ``variance_factor``, the a-posteriori variance factor
    of the float phase residuals), must stay below the chi-square(3) limit (a correct fix moves the position within
    the float uncertainty; a wrong integer set pulls it outside).

    It catches gross errors only: early in a session the float height sigma is several centimetres, so a fixed
    height biased by the troposphere/height correlation or unmodelled errors passes it.
    """
    dpos = np.asarray(x_fixed[:3], dtype=float) - np.asarray(x_float[:3], dtype=float)
    try:
        P_pos = _stabilize_cov(np.asarray(P_float[:3, :3], dtype=float) * FIX_TEST_VARIANCE_FACTOR
                               * float(variance_factor))
        stat = float(dpos @ np.linalg.solve(P_pos, dpos))
    except np.linalg.LinAlgError:
        return False, float("inf")
    return bool(np.isfinite(stat) and stat <= _chi2_limit(3, factor=1.0)), stat


def build_sd_matrix(n: int, ref_idx: int) -> np.ndarray:
    """Build a single-difference matrix against one reference ambiguity."""
    if n < 2:
        return np.zeros((0, n), dtype=float)
    D = np.zeros((n - 1, n), dtype=float)
    r = 0
    for i in range(n):
        if i == ref_idx:
            continue
        D[r, ref_idx] = 1.0
        D[r, i] = -1.0
        r += 1
    return D


def _select_ref_by_elevation(ev_deg: np.ndarray) -> Optional[int]:
    if ev_deg.size == 0:
        return None
    good = np.isfinite(ev_deg)
    if not np.any(good):
        return None
    idx = np.argmax(np.where(good, ev_deg, -np.inf))
    return int(idx)


def _constrained_update(x: np.ndarray, P: np.ndarray, H: np.ndarray, y: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    if H.size == 0:
        return x, P
    R = np.eye(H.shape[0], dtype=float) * (sigma * sigma)
    PHt = P @ H.T
    S = H @ PHt + R
    try:
        K = np.linalg.solve(S.T, PHt.T).T
    except np.linalg.LinAlgError:
        return x, P
    innov = y - H @ x
    x_new = x + K @ innov
    I = np.eye(P.shape[0], dtype=float)
    KH = K @ H
    P_new = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new


def _bootstrap_success_rates(Q_amb: np.ndarray, variance_factor: float = 1.0) -> tuple[np.ndarray, tuple]:
    """Bootstrapped success rate of the trailing decorrelated ambiguities (element i: the set i..n-1, which the
    LAMBDA search conditions each on those after it; conditional variances scaled by
    ``FIX_TEST_VARIANCE_FACTOR`` times ``variance_factor``) and the decorrelation ``(Z, L, D)`` of ``Q_amb``
    itself."""
    from scipy.stats import norm

    _, Z, L, D, _, _ = decorrel(Q_amb, np.zeros(Q_amb.shape[0]))
    scale = FIX_TEST_VARIANCE_FACTOR * float(variance_factor)
    p_each = 2.0 * norm.cdf(0.5 / np.sqrt(np.maximum(D[0] * scale, 1e-300))) - 1.0
    return np.cumprod(p_each[::-1])[::-1], (Z, L, D)


def _partial_lambda(
    float_amb: np.ndarray, Q_amb: np.ndarray, min_fixed: int, variance_factor: float = 1.0
) -> Optional[tuple[np.ndarray, np.ndarray, float, float, float]]:
    """LAMBDA search of the largest decorrelated subset whose bootstrapped success rate reaches
    ``PARTIAL_MIN_SUCCESS_RATE`` (partial ambiguity resolution, Teunissen et al. 1999).

    The subset is chosen from the conditional variances of the decorrelated ambiguities alone, not from
    their values, so the ratio test on it keeps its meaning. ``min_fixed`` equal to the dimension asks for the
    full set (no partial fixing). ``variance_factor`` scales the conditional variances of the success rate (the
    a-posteriori variance factor of the float phase residuals; the search and the ratio do not depend on a scale
    of ``Q_amb``). Returns ``(Hz, fixed, ratio, distance, success_rate)``: the integer
    combinations of the input ambiguities that are fixed (rows), their integer values, the ratio test value,
    the squared distance of the best candidate and the success rate; ``None`` when fewer than ``min_fixed``
    combinations reach the success rate.
    """
    Q = _stabilize_cov(Q_amb)
    if np.linalg.cond(Q) > 1e16:
        raise np.linalg.LinAlgError("Ambiguity covariance is ill-conditioned")
    frac, incr = np.modf(float_amb)
    tail, (Z, L, D) = _bootstrap_success_rates(Q, variance_factor)
    zhat = Z.T.dot(frac)
    m = zhat.size
    if min_fixed >= m:
        start = 0
    else:
        ok = np.flatnonzero(tail >= PARTIAL_MIN_SUCCESS_RATE)
        start = int(ok[0]) if ok.size else m
        if m - start < min_fixed:
            return None
    zfixed, sqnorm = ssearch(zhat[start:], L[start:, start:], D[:, start:], ncands=2)
    ratio = float(sqnorm[1] / sqnorm[0]) if sqnorm.size >= 2 and sqnorm[0] > 0.0 else float("inf")
    Hz = Z[:, start:].T
    fixed = zfixed[:, 0] + Hz @ incr
    return Hz, fixed, ratio, float(sqnorm[0]), float(tail[start])


def apply_conventional_pppar(
    x: np.ndarray,
    P: np.ndarray,
    base_dim: int,
    n_gps: int,
    n_gal: int,
    gps_ev: Optional[Sequence[float]],
    gal_ev: Optional[Sequence[float]],
    lambda_gps: float,
    lambda_gal: float,
    settings: PPPARSettings,
    gps_age: Optional[Sequence[int]] = None,
    gal_age: Optional[Sequence[int]] = None,
    ratio_P: Optional[np.ndarray] = None,
    gps_eligible: Optional[Sequence[bool]] = None,
    gal_eligible: Optional[Sequence[bool]] = None,
    *,
    phase_residuals: Optional[pd.Series] = None,
    phase_weights: Optional[pd.Series] = None,
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    """PPP-AR on combined ionosphere-free ambiguity states: no ambiguity is fixed.

    The ionosphere-free ambiguity in narrow-lane cycles is N1 + f2 / (f1 - f2) * N_WL (GPS L1/L2: N1 +
    3.529 N_WL; Galileo E1/E5a: N1 + 2.949 N_WL). It is an integer only together with the wide lane N_WL,
    which the ionosphere-free filters neither estimate nor take from the Melbourne-Wuebbena combination.
    Fixing it directly gives integers that are wrong by up to half a narrow-lane cycle (5 cm): on
    BRUX 2024-035 the validated fixes were still worse than the float solution on the same epochs in every
    case (G 6 h 2.8 against 2.2 cm 3D RMS, G+E kinematic 20.6 against 14.5 cm). Each system with
    ambiguities records the reason ``if_not_integer`` and the state is returned unchanged (float); the session
    warns when ``pppar_combined_if_ar_enabled`` is set. PPP-AR runs in the uncombined filters (wide lane,
    then narrow lane: ``apply_indexed_uncombined_pppar``).

    The signature is kept for the filters that call it; the elevation, lock-age, eligibility, covariance and
    phase-residual arguments (``phase_residuals`` / ``phase_weights``) are not
    used.
    """
    diag = PPPARDiagnostics()
    if not settings.enabled:
        return x, P, diag
    for system, n in (("G", n_gps), ("E", n_gal)):
        if n:
            _record_attempt(diag, "if_not_integer", system=system, candidates=int(n))
    return x, P, diag


def apply_uncombined_pppar(
    x: np.ndarray,
    P: np.ndarray,
    base_dim: int,
    n_gps: int,
    n_gal: int,
    gps_ev: Optional[Sequence[float]],
    gal_ev: Optional[Sequence[float]],
    settings: PPPARSettings,
    gps_age: Optional[Sequence[int]] = None,
    gal_age: Optional[Sequence[int]] = None,
    ratio_P: Optional[np.ndarray] = None,
    gps_eligible: Optional[Sequence[bool]] = None,
    gal_eligible: Optional[Sequence[bool]] = None,
    *,
    phase_residuals: Optional[pd.Series] = None,
    phase_weights: Optional[pd.Series] = None,
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    """Apply PPP-AR to the standard uncombined dual-frequency state layout.

    The expected layout is ``base_dim + 3 * sat + [I, N1, N2]`` for each GPS
    and Galileo satellite. This wrapper builds explicit N1/N2 index groups and
    delegates the actual candidate selection, wide-lane gate, ratio test and
    soft hold update to ``apply_indexed_uncombined_pppar`` (``phase_residuals`` /
    ``phase_weights`` are passed on unchanged).

    Status:
        Active uncombined AR helper for standard GPS/Galileo-style state
        layouts. BDS use requires caller-side validation and may be disabled by
        filter classes.
    """
    def _n1_indices(n: int, start_sat: int) -> np.ndarray:
        sat_ids = np.arange(start_sat, start_sat + n)
        return base_dim + 3 * sat_ids + 1

    def _n2_indices(n: int, start_sat: int) -> np.ndarray:
        sat_ids = np.arange(start_sat, start_sat + n)
        return base_dim + 3 * sat_ids + 2

    groups: list[dict[str, object]] = []
    if n_gps:
        groups.append(
            {
                "system": "G",
                "n1_idx": _n1_indices(n_gps, 0),
                "n2_idx": _n2_indices(n_gps, 0),
                "ev": gps_ev,
                "age": gps_age,
                "eligible": gps_eligible,
            }
        )
    if n_gal:
        groups.append(
            {
                "system": "E",
                "n1_idx": _n1_indices(n_gal, n_gps),
                "n2_idx": _n2_indices(n_gal, n_gps),
                "ev": gal_ev,
                "age": gal_age,
                "eligible": gal_eligible,
            }
        )

    return apply_indexed_uncombined_pppar(x=x, P=P, ambiguity_groups=groups, settings=settings, ratio_P=ratio_P,
                                          phase_residuals=phase_residuals, phase_weights=phase_weights)


def _check_residual_index(phase_residuals: pd.Series, groups: Sequence[dict[str, object]]) -> None:
    """A residual Series that names none of the groups' ambiguity states was built with another index convention
    (the state index of the ambiguity each phase carries); it would silently give no factor."""
    if not len(phase_residuals):
        return
    states = []
    for group in groups:
        for key, alias in (("n1_idx", "n1"), ("n2_idx", "n2")):
            raw = group.get(key, group.get(alias))
            if raw is None:
                continue
            try:
                states.append(np.asarray(raw, dtype=int).ravel())
            except (TypeError, ValueError):
                continue
    if not states:
        return
    if not np.isin(np.asarray(phase_residuals.index, dtype=float), np.concatenate(states)).any():
        raise ValueError("phase_residuals: no index is an ambiguity state index of the groups (contract C3: index "
                         "by the state index of the ambiguity each phase carries, pppar.ar_phase_series)")


def _select_wide_lane_reference(
    wl: np.ndarray,
    Q_wl: np.ndarray,
    ev: np.ndarray,
    max_frac: Optional[float],
    variance_factor: float = 1.0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Reference satellite of the wide-lane single differences and the candidates that round safely.

    Every candidate is tried as the reference; the one leaving the most single differences that round
    safely wins (ties: higher elevation). A difference rounds safely when its fractional part is within
    ``max_frac`` cycles and its standard deviation (the float covariance scaled by ``FIX_TEST_VARIANCE_FACTOR``
    times ``variance_factor``) below ``WIDE_LANE_MAX_SIGMA_CYCLES``. A satellite whose
    wide lane carries a receiver-satellite-specific bias (on BRUX 2024-035 the GPS III satellites G04/G11/G14
    and the Galileo IOV satellites E11/E12/E19 sit 0.3-0.5 cycles off the others in the Melbourne-Wuebbena
    combination, with GRG and COD alike) is left out instead of spoiling the reference or the whole group.
    Returns the reference index, the mask of safe candidates (reference included) and the fractional parts.
    """
    n = wl.size
    var = np.diag(Q_wl)
    limit = 0.5 if max_frac is None else float(max_frac)
    best = (-1, -np.inf, 0, np.ones(n, dtype=bool), np.zeros(n))
    for r in range(n):
        sd = wl - wl[r]
        frac = sd - np.round(sd)
        sd_var = var + var[r] - 2.0 * Q_wl[:, r]
        sd_sigma = np.sqrt(np.maximum(sd_var * FIX_TEST_VARIANCE_FACTOR * float(variance_factor), 0.0))
        ok = (np.abs(frac) <= limit) & (sd_sigma <= WIDE_LANE_MAX_SIGMA_CYCLES)
        ok[r] = True
        count = int(np.count_nonzero(ok))
        elev = ev[r] if np.isfinite(ev[r]) else -np.inf
        if count > best[0] or (count == best[0] and elev > best[1]):
            best = (count, elev, r, ok, frac)
    return best[2], best[3], best[4]


def apply_indexed_uncombined_pppar(
    x: np.ndarray,
    P: np.ndarray,
    ambiguity_groups: Sequence[dict[str, object]],
    settings: PPPARSettings,
    ratio_P: Optional[np.ndarray] = None,
    *,
    phase_residuals: Optional[pd.Series] = None,
    phase_weights: Optional[pd.Series] = None,
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    """Apply uncombined PPP-AR to explicit ambiguity state indices.

    Each group supplies N1 indices, optional N2 indices, elevations, optional
    arc ages, an optional ``eligible`` mask (satellites without product phase
    biases are not candidates; ``ar_eligible_mask`` also leaves out the phases
    flagged with a possible half-cycle ambiguity) and an optional
    ``half_cycle`` mask (True: LLI bit 1 set on a phase of that satellite, the
    ambiguity is not a candidate while it is set). Accepted fixes are
    applied as soft single-difference constraints, preserving the EKF state
    dimension.

    ``phase_residuals`` / ``phase_weights`` (default
    ``None`` = no factor): the postfit residuals of the epoch's float phase
    observations [m] and their weights [1/m^2], pandas Series indexed by the
    state index of the ambiguity each phase carries (``ar_phase_series``; an
    index that is not integer is a TypeError, one that names none of the
    groups' states a ValueError). Each group's validation uses the float
    covariance scaled by the a-posteriori variance factor of the residuals of
    its candidate ambiguities (after the phase-bias, half-cycle and elevation
    screening; ``phase_variance_factor``, floored at 1), the position check
    that of the candidates of all groups together.

    Model (dual-frequency groups, wide lane then narrow lane):
        1. Between-satellite single differences of the wide-lane float
           ambiguities N1 - N2 are rounded where that is safe (fraction within
           ``wide_lane_max_frac_cycles``, standard deviation below
           ``WIDE_LANE_MAX_SIGMA_CYCLES``); the reference satellite is the one
           that keeps the most candidates, the others are left out.
        2. The N1 single differences, conditioned on the fixed wide lanes, are
           searched with LAMBDA. With ``partial_fixing_enabled`` the search
           keeps the decorrelated combinations whose bootstrapped success rate
           reaches ``PARTIAL_MIN_SUCCESS_RATE`` (chosen from their conditional
           variances, not their values), at least the partial minimum; without
           it the full set must reach that success rate (reason
           ``success_rate`` otherwise).
        3. The fix is validated: ratio test, success rate, narrow-lane gate
           (every fixed combination within ``NARROW_LANE_MAX_FRAC_CYCLES`` of
           its float value), chi-square test of the distance between the float
           and the fixed ambiguities (``FIX_TEST_ALPHA``), and after all groups
           the fixed-vs-float position check, all on the float covariance
           scaled by ``FIX_TEST_VARIANCE_FACTOR`` and the variance factor of
           the phase residuals. N1 and N2 = N1 - wide lane are then constrained
           together with ``pppar_constraint_sigma_cycles``; step 2 conditions
           the narrow lane on the wide lane with the internal
           ``WIDE_LANE_CONDITION_SIGMA_CYCLES``.
        Groups without N2 (single frequency) skip step 1.

    Status:
        Active low-level AR engine. Requires caution because callers can pass
        arbitrary state indices; correctness depends on the filter's state
        layout and bias model.
    """
    diag = PPPARDiagnostics()
    if not settings.enabled:
        return x, P, diag
    H_rows = []
    y_rows = []
    ratios = []
    pending: list[tuple[int, dict[str, object]]] = []
    sigma_c = _effective_constraint_sigma_cycles(settings)
    # NQ-PPP-20: a-posteriori variance factor of the float phase residuals (1 without residuals)
    residual_arrays = _phase_residual_arrays(phase_residuals, phase_weights)
    if residual_arrays is not None:
        _check_residual_index(phase_residuals, ambiguity_groups)
    screened_state_idx: list[np.ndarray] = []

    def _system_fix(group_index: int, group: dict[str, object]) -> None:
        system = group.get("system")
        n1_idx_raw = group.get("n1_idx", group.get("n1"))
        if n1_idx_raw is None:
            _record_attempt(diag, "missing_n1", group_index=group_index, system=system)
            return
        n1_idx = np.asarray(n1_idx_raw, dtype=int)
        ev_raw = group.get("ev", group.get("elev", np.empty(0, dtype=float)))
        ev = np.asarray(ev_raw, dtype=float)
        n = n1_idx.size
        n2_idx_raw = group.get("n2_idx", group.get("n2"))
        n2_idx = None if n2_idx_raw is None else np.asarray(n2_idx_raw, dtype=int)
        if n2_idx is not None and n2_idx.size != n:
            _record_attempt(
                diag,
                "n2_size_mismatch",
                group_index=group_index,
                system=system,
                candidates=n,
                n2_size=int(n2_idx.size),
            )
            return
        if ev.size != n:
            _record_attempt(
                diag,
                "elevation_size_mismatch",
                group_index=group_index,
                system=system,
                candidates=n,
                elev_size=int(ev.size),
            )
            return

        age_raw = group.get("age")
        age = None
        if age_raw is not None:
            age = np.asarray(age_raw, dtype=int)
            if age.size != n:
                _record_attempt(
                    diag,
                    "lock_age_size_mismatch",
                    group_index=group_index,
                    system=system,
                    candidates=n,
                    age_size=int(age.size),
                )
                return

        # satellites without product phase biases: their ambiguities are not integer (NQ-PPP-18, PPP-016); phases
        # with a possible half-cycle ambiguity (RINEX LLI bit 1) stay out of the fix while flagged (NQ-PPP-37)
        eligible_raw = group.get("eligible")
        half_raw = group.get("half_cycle")
        if eligible_raw is not None or half_raw is not None:
            # no phase bias, or phase biases of another solution than the clocks (NQ-PPP-24)
            exclusion_reason = getattr(eligible_raw, "exclusion_reason", "no_phase_bias")
            eligible = np.ones(n, dtype=bool) if eligible_raw is None else np.asarray(eligible_raw, dtype=bool)
            half_only = getattr(eligible_raw, "half_cycle", None)
            half_only = (np.zeros(eligible.size, dtype=bool) if half_only is None
                         else np.asarray(half_only, dtype=bool))
            half = None if half_raw is None else np.asarray(half_raw, dtype=bool)
            if eligible.size != n or half_only.size != n or (half is not None and half.size != n):
                # a mask not aligned with the ambiguities cannot tell which satellites have phase biases
                _record_attempt(
                    diag,
                    "eligible_size_mismatch",
                    group_index=group_index,
                    system=system,
                    candidates=n,
                    eligible_size=int(eligible.size),
                    half_cycle_size=None if half is None else int(half.size),
                )
                return
            if half is not None:
                half_only = half_only | (half & eligible)
                eligible = eligible & ~half
            if not eligible.all():
                n_half = int(np.count_nonzero(half_only))
                n_bias = int(np.count_nonzero(~eligible)) - n_half
                if exclusion_reason == "mixed_products":
                    diag.excluded_mixed_products += n_bias
                else:
                    diag.excluded_no_phase_bias += n_bias
                diag.excluded_half_cycle += n_half
                if np.count_nonzero(eligible) < settings.min_ambiguities:
                    if n_half and (not n_bias
                                   or np.count_nonzero(eligible | half_only) >= settings.min_ambiguities):
                        reason = "half_cycle"
                    else:
                        reason = exclusion_reason
                    _record_attempt(
                        diag,
                        reason,
                        group_index=group_index,
                        system=system,
                        candidates=n,
                        with_phase_bias=int(np.count_nonzero(eligible | half_only)),
                        half_cycle=n_half,
                    )
                    return
                n1_idx = n1_idx[eligible]
                ev = ev[eligible]
                if n2_idx is not None:
                    n2_idx = n2_idx[eligible]
                if age is not None:
                    age = age[eligible]
                n = int(np.count_nonzero(eligible))

        min_elev = settings.min_candidate_elevation_deg
        if min_elev is not None:
            elev_mask = ev >= float(min_elev)
            selected_count = int(np.count_nonzero(elev_mask))
            if selected_count < settings.min_ambiguities:
                _record_attempt(
                    diag,
                    "candidate_selection_gate",
                    group_index=group_index,
                    system=system,
                    candidates=n,
                    selected_candidates=selected_count,
                    min_candidate_elevation_deg=float(min_elev),
                )
                return
            if selected_count < n:
                n1_idx = n1_idx[elev_mask]
                ev = ev[elev_mask]
                if n2_idx is not None:
                    n2_idx = n2_idx[elev_mask]
                if age is not None:
                    age = age[elev_mask]
                n = selected_count

        # NQ-PPP-20: the group's validation uses the variance factor of the residuals of its candidate ambiguities
        # (after the phase-bias, half-cycle and elevation screening; before the lock-age, wide-lane and partial
        # selection, which choose among sound phases), so a flagged phase cannot veto the fix of the others
        group_state_idx = n1_idx if n2_idx is None else np.concatenate([n1_idx, n2_idx])
        screened_state_idx.append(group_state_idx)
        vf, vf_dof = _variance_factor(residual_arrays, group_state_idx)

        partial_applied = False
        original_candidates = n
        partial_min = _effective_partial_min_ambiguities(settings)
        if age is not None:
            min_lock = _effective_min_lock_epochs(settings)
            if min_lock > 0:
                mature = age >= min_lock
                mature_count = int(np.count_nonzero(mature))
                if mature_count < n:
                    if not settings.partial_fixing_enabled or mature_count < partial_min:
                        _record_attempt(
                            diag,
                            "young_arc",
                            group_index=group_index,
                            system=system,
                            candidates=n,
                            mature_candidates=mature_count,
                            min_lock_epochs=min_lock,
                            min_age=int(np.min(age)) if age.size else None,
                            max_age=int(np.max(age)) if age.size else None,
                            partial_min_ambiguities=partial_min,
                        )
                        return
                    n1_idx = n1_idx[mature]
                    ev = ev[mature]
                    if n2_idx is not None:
                        n2_idx = n2_idx[mature]
                    n = mature_count
                    partial_applied = True

        if n < settings.min_ambiguities:
            _record_attempt(
                diag,
                "below_min_ambiguities",
                group_index=group_index,
                system=system,
                candidates=n,
                min_ambiguities=settings.min_ambiguities,
            )
            return
        if np.any(n1_idx < 0) or np.any(n1_idx >= x.size):
            _record_attempt(diag, "n1_index_oob", group_index=group_index, system=system, candidates=n)
            return
        if n2_idx is not None and (np.any(n2_idx < 0) or np.any(n2_idx >= x.size)):
            _record_attempt(diag, "n2_index_oob", group_index=group_index, system=system, candidates=n)
            return
        cov_source = P if ratio_P is None else ratio_P
        if cov_source.shape != P.shape:
            _record_attempt(
                diag,
                "ratio_covariance_invalid",
                group_index=group_index,
                system=system,
                candidates=n,
                covariance_shape=cov_source.shape,
                state_shape=P.shape,
            )
            return

        # 1. wide lane: reference satellite and safely rounded single differences
        x_c, P_c = x, cov_source
        if n2_idx is not None:
            wl = x[n1_idx] - x[n2_idx]
            Q_wl = (cov_source[np.ix_(n1_idx, n1_idx)] + cov_source[np.ix_(n2_idx, n2_idx)]
                    - cov_source[np.ix_(n1_idx, n2_idx)] - cov_source[np.ix_(n2_idx, n1_idx)])
            ref, wl_ok, wl_frac = _select_wide_lane_reference(wl, Q_wl, ev, settings.wide_lane_max_frac_cycles,
                                                              variance_factor=vf)
            wl_count = int(np.count_nonzero(wl_ok))
            if wl_count < n:
                diag.excluded_wide_lane += n - wl_count
            if wl_count < settings.min_ambiguities:
                _record_attempt(
                    diag,
                    "wide_lane_gate",
                    group_index=group_index,
                    system=system,
                    candidates=n,
                    wide_lane_candidates=wl_count,
                    wl_max_abs_frac=float(np.max(np.abs(wl_frac))) if wl_frac.size else 0.0,
                    wl_max_frac_threshold=settings.wide_lane_max_frac_cycles,
                )
                return
            if wl_count < n:
                partial_applied = True
                ref = int(np.flatnonzero(np.flatnonzero(wl_ok) == ref)[0])
                n1_idx, n2_idx, ev = n1_idx[wl_ok], n2_idx[wl_ok], ev[wl_ok]
                n = wl_count
            D_wl = build_sd_matrix(n, ref)
            wl_fixed = np.round(D_wl @ (x[n1_idx] - x[n2_idx]))
            H_wl = np.zeros((D_wl.shape[0], x.size), dtype=float)
            H_wl[:, n1_idx] = D_wl
            H_wl[:, n2_idx] -= D_wl
            # narrow lane conditioned on the fixed wide lane with the internal tight sigma (PPP-063: not the
            # constraint sigma of the setting, which weights only the constraint applied to the filter)
            x_c, P_c = _constrained_update(x, cov_source, H_wl, wl_fixed, sigma=WIDE_LANE_CONDITION_SIGMA_CYCLES)
        else:
            ref = _select_ref_by_elevation(ev)
            if ref is None:
                _record_attempt(diag, "no_reference_satellite", group_index=group_index, system=system,
                                candidates=n)
                return

        # 2. N1 single differences (conditioned on the fixed wide lanes) with LAMBDA; partial fixing keeps the
        # decorrelated combinations whose bootstrapped success rate reaches PARTIAL_MIN_SUCCESS_RATE
        D = build_sd_matrix(n, ref)
        n1_sd_f = D @ x_c[n1_idx]
        Qsd = D @ P_c[np.ix_(n1_idx, n1_idx)] @ D.T
        min_fixed = max(int(partial_min) - 1, 1) if settings.partial_fixing_enabled else D.shape[0]
        try:
            condition = float(np.linalg.cond(_stabilize_cov(Qsd)))
            if condition > settings.max_condition_number:
                _record_attempt(
                    diag,
                    "ill_conditioned",
                    group_index=group_index,
                    system=system,
                    candidates=n,
                    sd_candidates=int(D.shape[0]),
                    condition=condition,
                )
                return
            search = _partial_lambda(n1_sd_f, Qsd, min_fixed, variance_factor=vf)
        except Exception as exc:  # pragma: no cover - defensive guard
            _record_attempt(
                diag,
                "lambda_error",
                group_index=group_index,
                system=system,
                candidates=n,
                sd_candidates=int(D.shape[0]),
                error=type(exc).__name__,
            )
            return
        if search is None:
            # too imprecise to fix: fewer than the minimum number of combinations reach the success rate
            _record_attempt(
                diag,
                "success_rate",
                group_index=group_index,
                system=system,
                candidates=n,
                sd_candidates=int(D.shape[0]),
                min_fixed=int(min_fixed),
                success_rate_below=PARTIAL_MIN_SUCCESS_RATE,
                variance_factor=vf,
            )
            return
        Hz, fixed_z, ratio, sq_best, success_rate = search
        if Hz.shape[0] < D.shape[0]:
            diag.excluded_partial += int(D.shape[0] - Hz.shape[0])
            partial_applied = True
        if ratio < settings.ratio_threshold:
            _record_attempt(
                diag,
                "ratio_reject",
                group_index=group_index,
                system=system,
                candidates=n,
                sd_candidates=int(Hz.shape[0]),
                ratio=ratio,
                threshold=settings.ratio_threshold,
            )
            return
        # 3. validation: precise enough to be fixed (the partial subset reaches the success rate by
        # construction), every fixed combination close to its float value, and the best integer set
        # consistent with the float ambiguities and their covariance
        if success_rate < PARTIAL_MIN_SUCCESS_RATE:
            _record_attempt(
                diag,
                "success_rate",
                group_index=group_index,
                system=system,
                candidates=n,
                sd_candidates=int(Hz.shape[0]),
                ratio=ratio,
                success_rate=success_rate,
                variance_factor=vf,
            )
            return
        nl_residual = float(np.max(np.abs(Hz @ n1_sd_f - fixed_z)))
        if nl_residual > NARROW_LANE_MAX_FRAC_CYCLES:
            _record_attempt(
                diag,
                "narrow_lane_gate",
                group_index=group_index,
                system=system,
                candidates=n,
                sd_candidates=int(Hz.shape[0]),
                ratio=ratio,
                nl_max_residual=nl_residual,
                nl_max_frac_threshold=NARROW_LANE_MAX_FRAC_CYCLES,
            )
            return
        chi2_limit = _chi2_limit(Hz.shape[0], factor=FIX_TEST_VARIANCE_FACTOR * vf)
        if not np.isfinite(sq_best) or sq_best > chi2_limit:
            _record_attempt(
                diag,
                "fix_validation",
                group_index=group_index,
                system=system,
                candidates=n,
                sd_candidates=int(Hz.shape[0]),
                ratio=ratio,
                distance=sq_best,
                limit=chi2_limit,
                success_rate=success_rate,
                variance_factor=vf,
            )
            return
        HD = Hz @ D
        Hc = np.zeros((HD.shape[0], x.size), dtype=float)
        Hc[:, n1_idx] = HD
        H_rows.append(Hc)
        y_rows.append(fixed_z)
        if n2_idx is not None:
            # N2 = N1 - wide lane on the same integer combinations
            H2 = np.zeros((HD.shape[0], x.size), dtype=float)
            H2[:, n2_idx] = HD
            H_rows.append(H2)
            y_rows.append(fixed_z - Hz @ wl_fixed)
        ratios.append(ratio)
        pending.append((int(Hz.shape[0]), dict(
            group_index=group_index,
            system=system,
            candidates=n,
            original_candidates=original_candidates,
            sd_candidates=int(Hz.shape[0]),
            ratio=ratio,
            distance=sq_best,
            limit=chi2_limit,
            nl_max_residual=nl_residual,
            success_rate=success_rate,
            partial=partial_applied,
            variance_factor=vf,
            variance_dof=vf_dof,
        )))

    for group_index, group in enumerate(ambiguity_groups):
        _system_fix(group_index, group)

    # the position check: variance factor of the candidate phases of every group together (of every phase
    # residual of the epoch when no group got that far)
    vf_all, _ = _variance_factor(residual_arrays,
                                 np.concatenate(screened_state_idx) if screened_state_idx else None)
    if residual_arrays is not None:
        diag.phase_variance_factor = vf_all
    if not H_rows:
        return x, P, diag
    H = np.vstack(H_rows)
    y = np.concatenate(y_rows)
    x_new, P_new = _constrained_update(x, P, H, y, sigma=sigma_c)
    position_ok, shift_stat = _position_shift_ok(x, P if ratio_P is None else ratio_P, x_new,
                                                 variance_factor=vf_all)
    for sd_count, details in pending:
        if not position_ok:
            _record_attempt(diag, "position_check", position_shift_stat=shift_stat, **details)
            continue
        diag.fixed_ambiguities += sd_count
        if details["partial"]:
            diag.partial_groups += 1
            diag.partial_fixed_ambiguities += sd_count
        else:
            diag.full_groups += 1
        _record_attempt(diag, "accepted", **details)
    if not position_ok:
        return x, P, diag
    diag.accepted = True
    diag.ratio_min = float(np.min(ratios)) if ratios else None
    return x_new, P_new, diag
