from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class PPPARSettings:
    enabled: bool = False
    warmup_epochs: int = 60
    min_ambiguities: int = 4
    ratio_threshold: float = 2.0
    constraint_sigma_cycles: float = 1e-3
    max_condition_number: float = 1e12


@dataclass(slots=True)
class PPPARDiagnostics:
    fixed_ambiguities: int = 0
    ratio_min: Optional[float] = None
    accepted: bool = False


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
    # print('===' * 30)
    # print(" (sqnorm[1]/sqnorm[0]) ratio =", ratio)
    # print('sqnorm[0] =', sqnorm[0])
    # print('sqnorm[1] =', sqnorm[1])
    # print('float_amb =', float_amb)
    # print('afixed =', afixed)
    # print('==='*30)
    return best, ratio


def build_sd_matrix(n: int, ref_idx: int) -> np.ndarray:
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
        K = PHt @ np.linalg.inv(S)
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
    gps_ev: np.ndarray,
    gal_ev: np.ndarray,
    lambda_gps: float,
    lambda_gal: float,
    settings: PPPARSettings,
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    diag = PPPARDiagnostics()
    H_rows = []
    y_rows = []
    ratios = []

    def _system_fix(start: int, n: int, ev: np.ndarray, lam: float):
        if n < settings.min_ambiguities:
            return
        ref = _select_ref_by_elevation(ev)
        if ref is None:
            return
        amb_idx = np.arange(start, start + n)
        amb_m = x[amb_idx]
        D = build_sd_matrix(n, ref)
        Qm = P[np.ix_(amb_idx, amb_idx)]
        Qc = Qm / (lam * lam)
        try:
            amb_sd_f = D @ (amb_m / lam)
            Qsd = D @ Qc @ D.T
            if np.linalg.cond(_stabilize_cov(Qsd)) > settings.max_condition_number:
                return
            fixed_sd_cycles, ratio = lambda_ils(amb_sd_f, Qsd, ncands=2)
        except Exception:
            return
        if ratio < settings.ratio_threshold:
            return
        Hc = np.zeros((D.shape[0], x.size), dtype=float)
        Hc[:, amb_idx] = D
        yc = fixed_sd_cycles * lam
        H_rows.append(Hc)
        y_rows.append(yc)
        ratios.append(ratio)
        diag.fixed_ambiguities += int(D.shape[0])

    _system_fix(base_dim, n_gps, gps_ev, lambda_gps)
    _system_fix(base_dim + n_gps, n_gal, gal_ev, lambda_gal)

    if not H_rows:
        return x, P, diag
    H = np.vstack(H_rows)
    y = np.concatenate(y_rows)
    sigma_m = settings.constraint_sigma_cycles * min(lambda_gps, lambda_gal)
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
    gps_ev: np.ndarray,
    gal_ev: np.ndarray,
    settings: PPPARSettings,
) -> tuple[np.ndarray, np.ndarray, PPPARDiagnostics]:
    diag = PPPARDiagnostics()
    H_rows = []
    y_rows = []
    ratios = []

    def _n1_indices(n: int, start_sat: int) -> np.ndarray:
        sat_ids = np.arange(start_sat, start_sat + n)
        return base_dim + 3 * sat_ids + 1

    def _n2_indices(n: int, start_sat: int) -> np.ndarray:
        sat_ids = np.arange(start_sat, start_sat + n)
        return base_dim + 3 * sat_ids + 2

    def _system_fix(n: int, ev: np.ndarray, sat_offset: int):
        if n < settings.min_ambiguities:
            return
        ref = _select_ref_by_elevation(ev)
        if ref is None:
            return
        amb_idx = _n1_indices(n, sat_offset)
        n2_idx = _n2_indices(n, sat_offset)
        n1 = x[amb_idx]
        n2 = x[n2_idx]
        D = build_sd_matrix(n, ref)
        # WL check (|float-round| <= 0.25 cycles) as in UC PPP-AR procedure.
        wl_float = D @ (n1 - n2)
        wl_fixed = np.round(wl_float)
        if np.any(np.abs(wl_float - wl_fixed) > 0.25):
            return
        Qn1 = P[np.ix_(amb_idx, amb_idx)]
        try:
            n1_sd_f = D @ n1
            Qsd = D @ Qn1 @ D.T
            if np.linalg.cond(_stabilize_cov(Qsd)) > settings.max_condition_number:
                return
            fixed_sd, ratio = lambda_ils(n1_sd_f, Qsd, ncands=2)
        except Exception:
            return
        if ratio < settings.ratio_threshold:
            return
        Hc = np.zeros((D.shape[0], x.size), dtype=float)
        Hc[:, amb_idx] = D
        H_rows.append(Hc)
        y_rows.append(fixed_sd)
        ratios.append(ratio)
        diag.fixed_ambiguities += int(D.shape[0])

    _system_fix(n_gps, gps_ev, sat_offset=0)
    _system_fix(n_gal, gal_ev, sat_offset=n_gps)

    if not H_rows:
        return x, P, diag
    H = np.vstack(H_rows)
    y = np.concatenate(y_rows)
    x_new, P_new = _constrained_update(x, P, H, y, sigma=settings.constraint_sigma_cycles)
    diag.accepted = True
    diag.ratio_min = float(np.min(ratios)) if ratios else None
    return x_new, P_new, diag
