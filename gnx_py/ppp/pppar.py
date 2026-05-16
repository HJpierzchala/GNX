"""PPP ambiguity-resolution helpers.

This module contains the float-to-fixed ambiguity workflow used by selected PPP
filters. It implements covariance stabilization, LAMBDA integer least-squares
search, ratio testing, lock-time/arc-age gating, optional elevation-based
candidate selection, partial fixing, and soft hold constraints applied back to
the EKF state.

Status:
    Requires caution. These helpers can change numerical PPP behavior even when
    their signatures stay stable. Treat AR candidate selection, covariance
    source, ratio thresholds and hold constraints as model logic.

Combined ionosphere-free AR is intentionally guarded by a separate
configuration flag and is considered experimental. Uncombined AR is the primary
path covered by the current PPP tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(slots=True)
class PPPARSettings:
    """Configuration for PPP ambiguity resolution attempts.

    Status:
        Active AR settings container. Combined IF AR still requires the extra
        experimental gate ``pppar_combined_if_ar_enabled``.

    ``enabled`` gates all AR updates. ``warmup_epochs`` and
    ``min_lock_epochs`` control when an ambiguity arc can become a candidate.
    ``min_ambiguities``, ``min_candidate_elevation_deg`` and
    ``wide_lane_max_frac_cycles`` screen candidate groups. ``ratio_threshold``
    applies the LAMBDA ratio test. Accepted fixes are applied as soft
    constraints with ``constraint_sigma_cycles`` subject to the optional floor.

    ``partial_fixing_enabled`` allows mature subsets to be fixed when younger
    ambiguities are present. Combined ionosphere-free callers should also check
    ``combined_if_pppar_enabled`` because IF AR remains experimental.
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
    partial_fixing_enabled: bool = False
    partial_min_ambiguities: Optional[int] = None
    wide_lane_max_frac_cycles: Optional[float] = 0.25


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
    """Return whether experimental combined ionosphere-free PPP-AR may run.

    ``pppar_enabled`` alone is not enough for combined IF filters; the separate
    ``pppar_combined_if_ar_enabled`` flag prevents accidental activation of a
    less-validated AR branch.

    Status:
        Experimental feature gate. Do not remove or bypass this guard without
        dedicated combined IF AR validation.
    """
    return bool(getattr(config, "pppar_enabled", False)) and bool(
        getattr(config, "pppar_combined_if_ar_enabled", False)
    )


def advance_arc_age(
    age_by_key: dict[object, int],
    current_keys: Sequence[object],
    reset_keys: Optional[Sequence[object] | set[object]] = None,
) -> dict[object, int]:
    """Advance per-ambiguity age for the current epoch.

    Keys absent from ``current_keys`` disappear from the returned mapping.
    Keys present in ``reset_keys`` start a new arc at age zero; all other
    current keys increment from their previous value.
    """
    reset = set(reset_keys or ())
    return {
        key: 0 if key in reset else int(age_by_key.get(key, 0)) + 1
        for key in current_keys
    }


def arc_age_array(age_by_key: dict[object, int], current_keys: Sequence[object]) -> np.ndarray:
    """Return arc ages aligned with the current ambiguity ordering."""
    return np.asarray([int(age_by_key.get(key, 0)) for key in current_keys], dtype=int)


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


def ldldecom(Qahat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LDL-style decomposition used by the LAMBDA decorrelation step."""
    n = len(Qahat)
    L = np.zeros((n, n), dtype=float)
    D = np.empty((1, n), dtype=float)
    D[:] = np.nan
    Q = Qahat.copy().astype(float)
    for i in range(n - 1, -1, -1):
        D[0, i] = Q[i, i]
        L[i, 0:i + 1] = Q[i, 0:i + 1] / np.sqrt(Q[i, i])
        for j in range(0, i):
            Q[j, 0:j + 1] -= L[i, 0:j + 1] * L[i, j]
        L[i, 0:i + 1] /= L[i, i]
    return L, D


def decorrel(Qahat: np.ndarray, ahat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decorrelate float ambiguities for LAMBDA integer least-squares search."""
    n = len(Qahat)
    iZt = np.eye(n, dtype=float)
    i1 = n - 2
    sw = 1
    L, D = ldldecom(Qahat.copy())
    while sw:
        i = n - 1
        sw = 0
        while (not sw) and (i > 0):
            i -= 1
            if i <= i1:
                for j in range(i + 1, n):
                    mu = np.round(L[j, i])
                    if mu != 0.0:
                        L[j:n, i] -= mu * L[j:n, j]
                        iZt[:, j] += mu * iZt[:, i]
            delta = D[0, i] + (L[i + 1, i] ** 2) * D[0, i + 1]
            if delta < D[0, i + 1]:
                lam = D[0, i + 1] * L[i + 1, i] / delta
                eta = D[0, i] / delta
                D[0, i] = eta * D[0, i + 1]
                D[0, i + 1] = delta
                L[i:i + 2, 0:i] = np.array([[-L[i + 1, i], 1.0], [eta, lam]]).dot(L[i:i + 2, 0:i])
                L[i + 1, i] = lam
                if i == 0:
                    L[i + 2:n, i:i + 2] = L[i + 2:n, i + 1::-1].copy()
                    iZt[:, i:i + 2] = iZt[:, i + 1::-1].copy()
                else:
                    L[i + 2:n, i:i + 2] = L[i + 2:n, i + 1:i - 1:-1].copy()
                    iZt[:, i:i + 2] = iZt[:, i + 1:i - 1:-1].copy()
                i1 = i
                sw = 1
    Z = np.round(np.linalg.inv(iZt.T))
    Qzhat = Z.T.dot(Qahat).dot(Z)
    zhat = Z.T.dot(ahat)
    return Qzhat, Z, L, D, zhat, iZt


def ssearch(ahat: np.ndarray, L: np.ndarray, D: np.ndarray, ncands: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Search integer ambiguity candidates in the decorrelated domain."""
    n = len(ahat)
    afixed = np.zeros((n, ncands), dtype=float)
    sqnorm = np.inf * np.ones(ncands, dtype=float)
    dist = np.zeros(n, dtype=float)
    zb = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    step = np.zeros(n, dtype=float)
    k = n - 1
    zb[k] = ahat[k]
    z[k] = np.round(zb[k])
    y = zb[k] - z[k]
    step[k] = np.sign(y) if y != 0 else 1.0
    maxdist = np.inf
    nn = 0
    endsearch = False
    while not endsearch:
        newdist = dist[k] + (y * y) / D[0, k]
        if newdist < maxdist:
            if k != 0:
                dist[k - 1] = newdist
                S = 0.0
                for i in range(k, n):
                    S += L[i, k - 1] * (z[i] - zb[i])
                zb[k - 1] = ahat[k - 1] + S
                z[k - 1] = np.round(zb[k - 1])
                y = zb[k - 1] - z[k - 1]
                step[k - 1] = np.sign(y) if y != 0 else 1.0
                k -= 1
            else:
                if nn < ncands:
                    afixed[:, nn] = z.copy()
                    sqnorm[nn] = newdist
                    nn += 1
                    if nn == ncands:
                        order = np.argsort(sqnorm)
                        sqnorm = sqnorm[order]
                        afixed = afixed[:, order]
                        maxdist = sqnorm[-1]
                elif newdist < sqnorm[-1]:
                    afixed[:, -1] = z.copy()
                    sqnorm[-1] = newdist
                    order = np.argsort(sqnorm)
                    sqnorm = sqnorm[order]
                    afixed = afixed[:, order]
                    maxdist = sqnorm[-1]
                z[0] = z[0] + step[0]
                y = zb[0] - z[0]
                step[0] = -step[0] - np.sign(step[0])
        else:
            if k == n - 1:
                endsearch = True
            else:
                k += 1
                z[k] = z[k] + step[k]
                y = zb[k] - z[k]
                step[k] = -step[k] - np.sign(step[k])
    if nn == 0:
        raise np.linalg.LinAlgError("LAMBDA search failed to produce candidates")
    return afixed, sqnorm


def lambda_ils(float_amb: np.ndarray, Q_amb: np.ndarray, ncands: int = 2) -> tuple[np.ndarray, float]:
    """Resolve float ambiguities with LAMBDA ILS and return best fix plus ratio.

    The ratio is computed from the two best squared norms. The function raises
    ``LinAlgError`` for unstable covariance/search cases; callers translate
    those failures into PPP-AR diagnostics rather than changing filter state.
    """
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
    return best, ratio


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
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    """Apply PPP-AR to combined ionosphere-free ambiguity states.

    Model:
        Ambiguities are grouped per constellation as one IF ambiguity per
        satellite. A highest-elevation satellite is selected as reference,
        single differences are fixed with LAMBDA, and accepted fixes are held
        by soft constraints in meters.

    Status:
        This path is experimental in the current codebase and should normally
        be activated only when both ``pppar_enabled`` and
        ``pppar_combined_if_ar_enabled`` are true.

    BDS warning:
        BeiDou combined AR requires phase-bias validation. Some callers disable
        BDS AR explicitly; mixed BDS cases should not be treated as validated by
        this helper alone.
    """
    diag = PPPARDiagnostics()
    if not settings.enabled:
        return x, P, diag
    H_rows = []
    y_rows = []
    ratios = []

    def _system_fix(
        system: str,
        start: int,
        n: int,
        ev_raw: Optional[Sequence[float]],
        lam: float,
        age_raw: Optional[Sequence[int]],
    ) -> None:
        if n == 0:
            return
        ev = np.asarray(ev_raw, dtype=float) if ev_raw is not None else np.empty(0, dtype=float)
        if ev.size != n:
            _record_attempt(
                diag,
                "elevation_size_mismatch",
                system=system,
                candidates=n,
                elev_size=int(ev.size),
            )
            return
        age = None
        if age_raw is not None:
            age = np.asarray(age_raw, dtype=int)
            if age.size != n:
                _record_attempt(
                    diag,
                    "lock_age_size_mismatch",
                    system=system,
                    candidates=n,
                    age_size=int(age.size),
                )
                return

        amb_idx = np.arange(start, start + n)
        min_elev = settings.min_candidate_elevation_deg
        if min_elev is not None:
            elev_mask = ev >= float(min_elev)
            selected_count = int(np.count_nonzero(elev_mask))
            if selected_count < settings.min_ambiguities:
                _record_attempt(
                    diag,
                    "candidate_selection_gate",
                    system=system,
                    candidates=n,
                    selected_candidates=selected_count,
                    min_candidate_elevation_deg=float(min_elev),
                )
                return
            if selected_count < n:
                amb_idx = amb_idx[elev_mask]
                ev = ev[elev_mask]
                if age is not None:
                    age = age[elev_mask]
                n = selected_count

        partial_applied = False
        original_candidates = n
        if age is not None:
            min_lock = _effective_min_lock_epochs(settings)
            if min_lock > 0:
                mature = age >= min_lock
                mature_count = int(np.count_nonzero(mature))
                if mature_count < n:
                    partial_min = _effective_partial_min_ambiguities(settings)
                    if not settings.partial_fixing_enabled or mature_count < partial_min:
                        _record_attempt(
                            diag,
                            "young_arc",
                            system=system,
                            candidates=n,
                            mature_candidates=mature_count,
                            min_lock_epochs=min_lock,
                            min_age=int(np.min(age)) if age.size else None,
                            max_age=int(np.max(age)) if age.size else None,
                            partial_min_ambiguities=partial_min,
                        )
                        return
                    amb_idx = amb_idx[mature]
                    ev = ev[mature]
                    n = mature_count
                    partial_applied = True

        if n < settings.min_ambiguities:
            _record_attempt(
                diag,
                "below_min_ambiguities",
                system=system,
                candidates=n,
                min_ambiguities=settings.min_ambiguities,
            )
            return
        ref = _select_ref_by_elevation(ev)
        if ref is None:
            _record_attempt(diag, "no_reference_satellite", system=system, candidates=n)
            return
        if np.any(amb_idx < 0) or np.any(amb_idx >= x.size):
            _record_attempt(diag, "n1_index_oob", system=system, candidates=n)
            return
        amb_m = x[amb_idx]
        D = build_sd_matrix(n, ref)
        cov_source = P if ratio_P is None else ratio_P
        if cov_source.shape != P.shape:
            _record_attempt(
                diag,
                "ratio_covariance_invalid",
                system=system,
                candidates=n,
                covariance_shape=cov_source.shape,
                state_shape=P.shape,
            )
            return
        Qm = cov_source[np.ix_(amb_idx, amb_idx)]
        Qc = Qm / (lam * lam)
        try:
            amb_sd_f = D @ (amb_m / lam)
            Qsd = D @ Qc @ D.T
            condition = float(np.linalg.cond(_stabilize_cov(Qsd)))
            if condition > settings.max_condition_number:
                _record_attempt(
                    diag,
                    "ill_conditioned",
                    system=system,
                    candidates=n,
                    sd_candidates=int(D.shape[0]),
                    condition=condition,
                )
                return
            fixed_sd_cycles, ratio = lambda_ils(amb_sd_f, Qsd, ncands=2)
        except Exception as exc:  # pragma: no cover - defensive guard
            _record_attempt(
                diag,
                "lambda_error",
                system=system,
                candidates=n,
                sd_candidates=int(D.shape[0]),
                error=type(exc).__name__,
            )
            return
        if ratio < settings.ratio_threshold:
            _record_attempt(
                diag,
                "ratio_reject",
                system=system,
                candidates=n,
                sd_candidates=int(D.shape[0]),
                ratio=ratio,
                threshold=settings.ratio_threshold,
            )
            return
        Hc = np.zeros((D.shape[0], x.size), dtype=float)
        Hc[:, amb_idx] = D
        yc = fixed_sd_cycles * lam
        H_rows.append(Hc)
        y_rows.append(yc)
        ratios.append(ratio)
        diag.fixed_ambiguities += int(D.shape[0])
        if partial_applied:
            diag.partial_groups += 1
            diag.partial_fixed_ambiguities += int(D.shape[0])
        else:
            diag.full_groups += 1
        _record_attempt(
            diag,
            "accepted",
            system=system,
            candidates=n,
            original_candidates=original_candidates,
            sd_candidates=int(D.shape[0]),
            ratio=ratio,
            partial=partial_applied,
        )

    _system_fix("G", base_dim, n_gps, gps_ev, lambda_gps, gps_age)
    _system_fix("E", base_dim + n_gps, n_gal, gal_ev, lambda_gal, gal_age)

    if not H_rows:
        return x, P, diag
    H = np.vstack(H_rows)
    y = np.concatenate(y_rows)
    sigma_m = _effective_constraint_sigma_cycles(settings) * min(lambda_gps, lambda_gal)
    x_new, P_new = _constrained_update(x, P, H, y, sigma=sigma_m)
    diag.accepted = True
    diag.ratio_min = float(np.min(ratios)) if ratios else None
    return x_new, P_new, diag

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
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    """Apply PPP-AR to the standard uncombined dual-frequency state layout.

    The expected layout is ``base_dim + 3 * sat + [I, N1, N2]`` for each GPS
    and Galileo satellite. This wrapper builds explicit N1/N2 index groups and
    delegates the actual candidate selection, wide-lane gate, ratio test and
    soft hold update to ``apply_indexed_uncombined_pppar``.

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
            }
        )

    return apply_indexed_uncombined_pppar(x=x, P=P, ambiguity_groups=groups, settings=settings, ratio_P=ratio_P)


def apply_indexed_uncombined_pppar(
    x: np.ndarray,
    P: np.ndarray,
    ambiguity_groups: Sequence[dict[str, object]],
    settings: PPPARSettings,
    ratio_P: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    """Apply uncombined PPP-AR to explicit ambiguity state indices.

    Each group supplies N1 indices, optional N2 indices, elevations, and
    optional arc ages. Dual-frequency groups pass a wide-lane fractional gate
    before LAMBDA fixing; groups without N2 are treated as single-frequency and
    skip that gate. Accepted fixes are applied as soft single-difference
    constraints, preserving the EKF state dimension.

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

        partial_applied = False
        original_candidates = n
        if age is not None:
            min_lock = _effective_min_lock_epochs(settings)
            if min_lock > 0:
                mature = age >= min_lock
                mature_count = int(np.count_nonzero(mature))
                if mature_count < n:
                    partial_min = _effective_partial_min_ambiguities(settings)
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
        ref = _select_ref_by_elevation(ev)
        if ref is None:
            _record_attempt(diag, "no_reference_satellite", group_index=group_index, system=system, candidates=n)
            return
        n1 = x[n1_idx]
        D = build_sd_matrix(n, ref)
        if n2_idx is not None:
            n2 = x[n2_idx]
            wl_float = D @ (n1 - n2)
            wl_fixed = np.round(wl_float)
            wl_frac_abs = np.abs(wl_float - wl_fixed)
            wide_lane_threshold = settings.wide_lane_max_frac_cycles
            if wide_lane_threshold is not None:
                wide_lane_threshold = max(0.0, float(wide_lane_threshold))
                if np.any(wl_frac_abs > wide_lane_threshold):
                    _record_attempt(
                        diag,
                        "wide_lane_gate",
                        group_index=group_index,
                        system=system,
                        candidates=n,
                        sd_candidates=int(D.shape[0]),
                        wl_max_abs_frac=float(np.max(wl_frac_abs)) if wl_frac_abs.size else 0.0,
                        wl_max_frac_threshold=wide_lane_threshold,
                    )
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
        Qn1 = cov_source[np.ix_(n1_idx, n1_idx)]
        try:
            n1_sd_f = D @ n1
            Qsd = D @ Qn1 @ D.T
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
            fixed_sd, ratio = lambda_ils(n1_sd_f, Qsd, ncands=2)
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
        if ratio < settings.ratio_threshold:
            _record_attempt(
                diag,
                "ratio_reject",
                group_index=group_index,
                system=system,
                candidates=n,
                sd_candidates=int(D.shape[0]),
                ratio=ratio,
                threshold=settings.ratio_threshold,
            )
            return
        Hc = np.zeros((D.shape[0], x.size), dtype=float)
        Hc[:, n1_idx] = D
        H_rows.append(Hc)
        y_rows.append(fixed_sd)
        ratios.append(ratio)
        diag.fixed_ambiguities += int(D.shape[0])
        if partial_applied:
            diag.partial_groups += 1
            diag.partial_fixed_ambiguities += int(D.shape[0])
        else:
            diag.full_groups += 1
        _record_attempt(
            diag,
            "accepted",
            group_index=group_index,
            system=system,
            candidates=n,
            original_candidates=original_candidates,
            sd_candidates=int(D.shape[0]),
            ratio=ratio,
            partial=partial_applied,
        )

    for group_index, group in enumerate(ambiguity_groups):
        _system_fix(group_index, group)

    if not H_rows:
        return x, P, diag
    H = np.vstack(H_rows)
    y = np.concatenate(y_rows)
    x_new, P_new = _constrained_update(x, P, H, y, sigma=_effective_constraint_sigma_cycles(settings))
    diag.accepted = True
    diag.ratio_min = float(np.min(ratios)) if ratios else None
    return x_new, P_new, diag
