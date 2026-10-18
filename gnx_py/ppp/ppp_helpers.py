"""Shared PPP helper utilities.

The functions here are intentionally small and side-effect-light. Trace helpers
provide a single controlled text output path for ``trace_filter`` diagnostics;
screening helpers implement simple residual gates used by several legacy and
active PPP filters. ``GNXExtendedKalmanFilter`` is the EKF of every PPP filter
(filterpy's with a Cholesky gain instead of ``inv(S)``).

Status:
    Active shared helper module. Residual screening helpers are numerical
    safeguards and should be changed only with filter-level regression tests.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
import scipy.linalg
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.stats import logpdf

from ..core.utils import calculate_distance
from .pppar import ar_phase_series, update_redundancy

#: measurement variance [m^2] from which a row counts as rejected: the filters keep a rejected or masked observation
#: in z with R = 1e12 (code gate, outlier screening, missing value) and GNXExtendedKalmanFilter leaves every row with
#: R >= REJECTED_R (or a non-finite R) out of the gain, as RTKLIB leaves rejected observations out of the measurement
#: vector (ppp.c res_ppp: excluded and outlier rows ``continue`` before ``nv++``), since v1.0.0
REJECTED_R = 1e10


def used_rows(R) -> np.ndarray:
    """Indices of the rows of ``R`` that enter the update: finite diagonal below :data:`REJECTED_R`."""
    d = np.diagonal(np.atleast_2d(np.asarray(R, dtype=float)))
    return np.flatnonzero(np.isfinite(d) & (d < REJECTED_R))


@dataclass(frozen=True)
class StatePrior:
    """A receiver clock / inter-system-bias state re-initialised from the code residuals of the epoch.

    RTKLIB (ppp.c:55 ``VAR_CLK = SQR(100.0)``, ``udclk_ppp`` ppp.c:660-678) re-initialises the receiver clock at
    every epoch (``initx``: value, variance, correlations zero). GNX does the same in
    :meth:`GNXExtendedKalmanFilter.predict_update` for a state whose predicted variance is larger than ``variance``
    (the white-noise clock of the presets, ``q_dt`` 1e9 / 9e9 m^2, at every epoch; an inter-system bias that starts
    with ``p_dt``, at its first epoch): its value becomes the weighted mean of the code residuals without that state
    over ``rows`` (weights 1/R; rows with ``R >= REJECTED_R`` and rows that also observe a later state of the same
    epoch's priors are left out), its variance ``variance`` [m^2] (``PPPConfig.clock_prior_sigma_m`` squared) and its
    correlations 0. RTKLIB takes the clock of its SPP solution; the weighted mean of the code residuals is the same
    least-squares clock with the filter's predicted geometry.

    ``index``: state index; ``rows``: indices of the code rows of z that observe it; ``reference``: True for the
    receiver clock, which the inter-system biases after it are defined relative to (when the clock prior cannot be
    applied, the bias priors after it are not applied either; a bias prior that cannot be applied does not block the
    other biases).
    """

    index: int
    rows: np.ndarray
    variance: float
    reference: bool = False


def clock_priors(sigma_m, code_rows, isb=None, clock_index=3):
    """Priors of the receiver clock (``clock_index``, every code row in ``code_rows``) and of the inter-system
    biases ``isb`` = {state index: code rows of that system}; ``sigma_m`` = ``PPPConfig.clock_prior_sigma_m`` [m].

    The clock comes first: a row of a system whose bias is re-initialised in the same epoch does not enter the clock
    value (``StatePrior``), and the bias then takes the mean of its rows minus that clock.
    """
    var = float(sigma_m) ** 2
    out = [StatePrior(int(clock_index), np.asarray(code_rows, dtype=int), var, reference=True)]
    for index, rows in (isb or {}).items():
        out.append(StatePrior(int(index), np.asarray(rows, dtype=int), var))
    return tuple(out)


def spd_gain(PHT, S):
    """Kalman gain K = PHT S^-1 for a symmetric positive (semi)definite innovation covariance ``S``.

    ``S`` is symmetrised and factored by Cholesky (``K = cho_solve(cho_factor(S), PHT.T).T``); when it is not
    numerically positive definite the gain falls back to a symmetric indefinite (LDL^T) solve. S^-1 is never formed:
    an explicit inverse (filterpy ``inv(S)``) loses the common-mode innovation of the white-noise receiver clock
    (``p_dt = 9e9`` m^2 against 4e-4 m^2 phase noise, cond(S) up to 3e15 with the R = 1e12 rows of rejected
    satellites): 0.25-1.7 m of clock and postfit-residual error and scipy's LinAlgWarning.
    A non-finite S raises ValueError, as ``inv(S)`` did.
    """
    S = 0.5 * (S + S.T)
    try:
        c = scipy.linalg.cho_factor(S, lower=True)
        return scipy.linalg.cho_solve(c, PHT.T, check_finite=False).T, S
    except np.linalg.LinAlgError:
        return scipy.linalg.solve(S, PHT.T, assume_a="sym").T, S


# GNXExtendedKalmanFilter.predict_update / .update are adapted from filterpy 1.4.5, filterpy/kalman/EKF.py
# (ExtendedKalmanFilter.predict_update and .update), under the following notice:
#
#   The MIT License (MIT)
#
#   Copyright (c) 2015 Roger R. Labbe Jr
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
class GNXExtendedKalmanFilter(ExtendedKalmanFilter):
    """filterpy ``ExtendedKalmanFilter`` with the gain of :func:`spd_gain`, rejected rows
    left out and the receiver-clock prior of RTKLIB (A1).

    ``predict_update`` and ``update`` follow filterpy 1.4.5 ``EKF.py`` (lines 172-252 and 254-341; MIT, Copyright (c)
    2015 Roger R. Labbe Jr, notice above) with ``SI = inv(S); K = PHT SI`` replaced by the Cholesky solve on the
    symmetrised S; the Joseph-form covariance update and the stored attributes are kept (``x_prior`` / ``P_prior`` copy the state before the prediction, as
    filterpy does). ``SI`` is formed only when read (filterpy's ``mahalanobis``); GNX reads only ``y``.
    Every PPP filter imports it as ``ExtendedKalmanFilter``.

    Since v1.0.0:

    * (a) the rows with ``R >= REJECTED_R`` (the filters' 1e12 marker of a rejected observation) are left out of S,
      the gain and the Joseph update, which run on the compact system (rows ``rows_used``); with R = 1e12 a rejected
      row still had a weight p_dt / (R + p_dt) = 0.9 % on the white-noise clock (a 1 km outlier moved its prior by
      ~9 m) and pushed cond(S) to 3e15. ``y`` (the innovation z - h(x) of the predicted state with the priors of (b))
      stays full length, every row included; ``K`` is full width with zero columns for the left-out rows; ``S`` (and
      ``SI``, ``mahalanobis``, ``log_likelihood``) are those of the compact system.
    * (b) ``state_priors``: a tuple of :class:`StatePrior` set by the filter before ``predict_update`` / ``update``
      (consumed by the call): each state whose predicted variance exceeds the prior's variance is re-initialised
      from the code residuals (``StatePrior``), as RTKLIB's ``udclk_ppp``. ``priors_applied`` lists the
      ``(index, variance)`` pairs applied by the last call (a filter that propagates a second covariance applies the
      same resets to it).
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        self._SI = None
        self.state_priors = None
        self.priors_applied = ()
        self.rows_used = None
        super().__init__(dim_x, dim_z, dim_u)

    @property
    def SI(self):
        """Inverse of the innovation covariance S (of the rows used), computed from S on request."""
        if self._SI is None:
            self._SI = spd_gain(np.eye(self.S.shape[0]), self.S)[0]
        return self._SI

    @SI.setter
    def SI(self, value):
        self._SI = value

    def _used_innovation(self):
        y = np.asarray(self.y, dtype=float).reshape(-1)
        return y if self.rows_used is None else y[self.rows_used]

    @property
    def mahalanobis(self):
        """Mahalanobis distance of the innovation of the rows used (filterpy's, on the compact system)."""
        if self._mahalanobis is None:
            y = self._used_innovation()
            self._mahalanobis = sqrt(float(np.dot(np.dot(y, self.SI), y))) if y.size else 0.0
        return self._mahalanobis

    @property
    def log_likelihood(self):
        """Log-likelihood of the innovation of the rows used (filterpy's, on the compact system)."""
        if self._log_likelihood is None:
            y = self._used_innovation()
            self._log_likelihood = float(logpdf(x=y, cov=self.S)) if y.size else 0.0
        return self._log_likelihood

    @property
    def likelihood(self):
        """exp(log_likelihood), at least the smallest positive float (as filterpy)."""
        if self._likelihood is None:
            self._likelihood = float(np.exp(self.log_likelihood))
            if self._likelihood == 0:
                self._likelihood = float(np.finfo(float).tiny)
        return self._likelihood

    def _apply_state_priors(self, x, P, z, H, Hx, hx_args, R):
        """Re-initialise the states of ``self.state_priors`` (see :class:`StatePrior`); returns x, P.

        A prior without a usable code row (every row rejected) is not applied and its state keeps its prediction, as
        before v1.0.0. When that prior is the receiver clock (``StatePrior.reference``) the inter-system
        bias priors after it, which are defined relative to the clock, are not applied either; an inter-system bias
        without a usable row only skips itself (the biases are not defined relative to each other)."""
        priors, self.state_priors = self.state_priors, None
        self.priors_applied = ()
        if not priors:
            return x, P
        todo = [p for p in priors if 0 <= p.index < len(x) and P[p.index, p.index] > p.variance]
        if not todo:
            return x, P
        z = np.asarray(z, dtype=float).reshape(-1)
        r_var = np.diagonal(np.atleast_2d(np.asarray(R, dtype=float)))
        applied = []
        x = np.array(x, dtype=float, copy=True)
        P = np.array(P, dtype=float, copy=True)
        for k, prior in enumerate(todo):
            rows = np.asarray(prior.rows, dtype=int).reshape(-1)
            rows = rows[(rows >= 0) & (rows < len(z))]
            h = H[rows, prior.index]
            var = r_var[rows]
            ok = (h != 0.0) & np.isfinite(var) & (var > 0.0) & (var < REJECTED_R) & np.isfinite(z[rows])
            later = [q.index for q in todo[k + 1:]]
            if later and rows.size:
                ok &= np.all(H[np.ix_(rows, later)] == 0.0, axis=1)
            if ok.any():
                x_wo = x.copy()
                x_wo[prior.index] = 0.0
                res = z[rows] - np.asarray(Hx(x_wo, *hx_args), dtype=float).reshape(-1)[rows]
                ok &= np.isfinite(res)
            if not ok.any():
                if prior.reference:
                    break
                continue
            w = 1.0 / var[ok]
            hh = h[ok]
            x[prior.index] = float(np.sum(w * hh * res[ok]) / np.sum(w * hh * hh))
            P[prior.index, :] = 0.0
            P[:, prior.index] = 0.0
            P[prior.index, prior.index] = prior.variance
            applied.append((prior.index, prior.variance))
        self.priors_applied = tuple(applied)
        return x, P

    def _measurement_update(self, x, P, H, R):
        """Gain and Joseph update on the rows used (``self.y`` already set, full length); sets K, S, x, P."""
        y = np.asarray(self.y, dtype=float).reshape(-1)
        m = y.size
        keep = used_rows(R) if m else np.zeros(0, dtype=int)
        self.rows_used = keep
        if keep.size == m:
            Hk, Rk, yk = H, R, self.y
        else:
            Hk, Rk, yk = H[keep], R[np.ix_(keep, keep)], y[keep]
        if keep.size:
            PHT = np.dot(P, Hk.T)
            Kk, self.S = spd_gain(PHT, np.dot(Hk, PHT) + Rk)
            x = x + np.dot(Kk, yk)
            I_KH = self._I - np.dot(Kk, Hk)
            P = np.dot(I_KH, P).dot(I_KH.T) + np.dot(Kk, Rk).dot(Kk.T)
        else:
            Kk, self.S = np.zeros((len(x), 0)), np.zeros((0, 0))
        if keep.size == m:
            self.K = Kk
        else:
            self.K = np.zeros((len(x), m))
            self.K[:, keep] = Kk
        self._SI = None
        self.x, self.P = x, P

    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        """Predict and update in one step (filterpy ``predict_update`` with the Cholesky gain, rejected rows left
        out and the state priors applied after the prediction)."""
        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)
        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F, B, P, Q, R, x = self.F, self.B, self.P, self.Q, self.R, self.x
        H = HJacobian(x, *args)

        # predict step
        x = np.dot(F, x) + np.dot(B, u)
        P = np.dot(F, P).dot(F.T) + Q

        # save prior (filterpy keeps the state before the prediction here)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        # receiver clock / ISB priors (A1(b)), then the update on the rows used (A1(a))
        x, P = self._apply_state_priors(x, P, z, H, Hx, hx_args, R)
        self.y = z - Hx(x, *hx_args)
        self._measurement_update(x, P, H, R)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(), residual=np.subtract):
        """Update step (filterpy ``update`` with the Cholesky gain, rejected rows left out, state priors applied
        to the current state)."""
        if z is None:
            self.state_priors = None
            self.priors_applied = ()
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return
        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R
        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, *args)
        x, P = self._apply_state_priors(self.x, self.P, z, H, Hx, hx_args, R)
        hx = Hx(x, *hx_args)
        self.y = residual(z, hx)
        # Joseph form: P = (I-KH)P(I-KH)' + KRK' on the rows used
        self._measurement_update(x, P, H, R)

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


def reset_prior_states(P, applied):
    """Apply the covariance part of ``GNXExtendedKalmanFilter.priors_applied`` to another covariance ``P`` (in
    place): the row and column of each state zeroed and its variance set (as the filter did to its own P)."""
    for index, variance in applied or ():
        if 0 <= index < P.shape[0]:
            P[index, :] = 0.0
            P[:, index] = 0.0
            P[index, index] = variance
    return P


def random_walk_variance(rate, interval_min):
    """Process-noise variance added to a random-walk state over one epoch.

    ``rate``: spectral density of the random walk [state unit^2 per second] - ``q_tro``,
    ``q_iono``, ``q_isb``, ``q_dcb`` and ``q_amb`` of ``PPPConfig``; ``interval_min``: the epoch
    interval in minutes, as the PPP filters store it (``interval``). The variance grows with the
    time between epochs, so the state wanders the same amount per hour at 1 s and at 300 s
    sampling (before v1.0.0 the troposphere noise was a variance per epoch). ``q_crd`` and ``q_dt``
    are not rates: they are variances per epoch (white-noise-like coordinates and clock).
    """
    return float(rate) * float(interval_min) * 60.0


# per-epoch process variance [state unit^2] from which a state counts as re-estimated every epoch
# (white-noise switch, e.g. q_dt = 1e9); such states are not scaled over data gaps
_FREE_PROCESS_VARIANCE = 1e4


def epoch_step(t_prev, t, interval_min):
    """Time from the previous filter prediction to epoch ``t`` in nominal epochs (>= 1).

    ``None`` or an unreadable time difference gives 1 (the first epoch, a new filter).
    """
    if t_prev is None or interval_min is None or not float(interval_min) > 0.0:
        return 1.0
    try:
        dt = (t - t_prev).total_seconds()
    except (AttributeError, TypeError):
        try:
            dt = float((np.datetime64(t) - np.datetime64(t_prev)) / np.timedelta64(1, "s"))
        except (TypeError, ValueError):
            return 1.0
    step = dt / (float(interval_min) * 60.0)
    return step if np.isfinite(step) and step > 1.0 else 1.0


def gap_scaled_process_noise(Q, step, n_epoch_states=4):
    """Process-noise matrix for a prediction over ``step`` nominal epochs.

    The filters build ``Q`` for one nominal epoch (``random_walk_variance``). After a data gap
    (or skipped epochs) the random-walk states (troposphere, ionosphere, ISB, receiver biases,
    ambiguities) must grow with the elapsed time, as in RTKLIB (``prn * |tt|``). The coordinates
    and the receiver clock (the first ``n_epoch_states`` states of every PPP filter) keep their
    per-epoch variances (``q_crd``, ``q_dt``), and so does any state that is already re-estimated
    every epoch (variance >= ``_FREE_PROCESS_VARIANCE``).
    """
    if not step > 1.0:
        return Q
    Qs = np.array(Q, dtype=float, copy=True)
    scale = np.abs(Qs) < _FREE_PROCESS_VARIANCE
    scale[:n_epoch_states, :] = False
    scale[:, :n_epoch_states] = False
    Qs[scale] *= step
    return Qs


class epoch_process_noise:
    """Context manager: ``ekf.Q`` scaled to the time since the previous prediction of ``ekf``.

    Wraps the ``predict_update`` call(s) of one epoch. The time of the last prediction is kept on
    the filter object, so a re-created filter (reset) starts with one nominal epoch. At the
    nominal sampling (``step == 1``) ``ekf.Q`` is left untouched; afterwards ``ekf.Q`` is restored.
    """

    def __init__(self, ekf, t, interval_min):
        self.ekf = ekf
        self.step = epoch_step(getattr(ekf, "_gnx_last_predict_time", None), t, interval_min)
        ekf._gnx_last_predict_time = t
        self._q = None

    def __enter__(self):
        if self.step > 1.0:
            self._q = self.ekf.Q
            self.ekf.Q = gap_scaled_process_noise(self._q, self.step)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._q is not None:
            self.ekf.Q = self._q
            self._q = None
        return False


def _format_trace_value(value):
    """Format scalar trace fields without dumping large arrays or DataFrames."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value).replace("\n", " ")


def trace_message(enabled, component, event, epoch=None, time=None, **fields):
    """Emit one controlled PPP trace line when ``enabled`` is true.

    Output format:
        ``[PPP TRACE] Component epoch=... time=... event=... key=value``

    This is the only intentional print sink in the PPP package. Normal library
    execution should pass ``enabled=False`` via ``trace_filter``.
    """
    if not enabled:
        return
    parts = ["[PPP TRACE]", component]
    if epoch is not None:
        parts.append(f"epoch={_format_trace_value(epoch)}")
    if time is not None:
        parts.append(f"time={_format_trace_value(time)}")
    parts.append(f"event={event}")
    for key, value in fields.items():
        if value is not None:
            parts.append(f"{key}={_format_trace_value(value)}")
    print(" ".join(parts))


def trace_epoch_summary(enabled, component, epoch=None, time=None, row=None, **fields):
    """Emit a compact per-epoch PPP trace summary.

    ``row`` may be a dict-like result row or pandas Series. Only a stable set
    of high-signal fields is emitted so trace mode does not dump full
    DataFrames or large residual vectors.
    """
    summary_keys = (
        "system",
        "systems",
        "mode",
        "reference_system",
        "de",
        "dn",
        "du",
        "dtr",
        "isb",
        "ztd",
        "n_sats",
        "n_sats_total",
        "n_states",
        "n_code_rejected",
        "n_code_rejected_total",
        "n_phase_reset",
        "n_phase_reset_total",
        "ar_fixed",
        "ar_ratio",
        "ar_ok",
    )
    payload = {}
    if row is not None:
        for key in summary_keys:
            value = row.get(key) if hasattr(row, "get") else None
            if value is not None:
                payload[key] = value
    payload.update(fields)
    trace_message(enabled, component, "epoch-summary", epoch=epoch, time=time, **payload)


def code_screening(x, satellites, code_obs, thr=1):
    """Return a boolean mask for code observations passing a median gate.

    The prefit code residuals are formed against geometric range from ``x``.
    Values outside ``median +/- thr`` are rejected, unless that would reject
    more than half the epoch, in which case all finite observations are kept.
    """
    dist = calculate_distance(satellites, x)
    prefit = code_obs - dist
    finite = np.isfinite(prefit)
    if not np.any(finite):
        return np.zeros(prefit.shape, dtype=bool)
    median_prefit = np.median(prefit[finite])
    mask = finite & (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
    n_sat = len(prefit)
    n_bad = np.count_nonzero(~mask)
    if n_bad > n_sat / 2:
        mask = finite.copy()
    return mask


def phase_residuals_outliers(sat_list, phase_residuals_dict, num, thr=10):
    """Find satellites with large epoch-to-epoch phase residual jumps.

    The function compares current and previous prefit residuals by satellite,
    removes the epoch median jump, and returns indices whose residual jump is
    larger than ``thr``. Callers decide whether those indices reset ambiguity
    states or only feed diagnostics.

    ``phase_residuals_dict`` is ``{epoch number: {satellite: residual}}``. The
    previous epoch is the latest one before ``num`` that has residuals: an
    epoch a filter skipped (e.g. a non-finite observation vector) has none,
    and the comparison then spans the skipped epoch like a data gap (before v1.0.0 the next epoch raised ``KeyError``). Without residuals
    at ``num`` or before it, nothing is flagged.
    """
    current = phase_residuals_dict.get(num)
    previous = phase_residuals_dict.get(num - 1)
    if previous is None:
        earlier = [k for k in phase_residuals_dict if k < num]
        previous = phase_residuals_dict[max(earlier)] if earlier else None
    if current is None or previous is None:
        return []
    prefit_entries = []
    for idx, sv in enumerate(sat_list):
        prev_residual = previous.get(sv)
        current_residual = current.get(sv)
        if prev_residual is not None and current_residual is not None:
            prefit_entries.append((idx, current_residual - prev_residual))
    if prefit_entries:
        prefit_diff = np.fromiter(
            (entry[1] for entry in prefit_entries),
            dtype=float,
            count=len(prefit_entries),
        )
        median_prefit_diff = np.median(prefit_diff)
    else:
        median_prefit_diff = 0.0
    return [
        idx
        for idx, residual in prefit_entries
        if np.abs(residual - median_prefit_diff) > thr
    ]


# ======================================================================= filter outputs (C1, C2; bug-fix stage 13b)
#: columns every PPP solution frame has: receiver position x, y, z [m, ECEF] and the
#: convergence time ct_min [min] (NaN when the filter did not converge)
SOLUTION_COLUMNS = ("x", "y", "z", "ct_min")


def finish_solution(frame, conv_time) -> pd.DataFrame:
    """The solution frame of a filter run in the common schema.

    ``frame``: the per-epoch rows (a DataFrame, possibly empty or None); ``conv_time``: the filter's convergence
    time (``pandas.Timedelta``/``datetime.timedelta`` or None). Every route returns x, y, z [m, ECEF] and ct_min
    [min, float]: the convergence time in minutes on every row, NaN when the position did not settle (before v1.0.0 ct_min held hours, IF single-system and R4c had no ct_min and R4c named the position
    xr, yr, zr). An empty solution keeps these columns with zero rows and a ``time`` DatetimeIndex.
    """
    if frame is None or (isinstance(frame, pd.DataFrame) and frame.empty and len(frame.columns) == 0):
        frame = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
    frame = frame.rename(columns={"xr": "x", "yr": "y", "zr": "z"})
    if len(frame) == 0 and not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_axis(pd.DatetimeIndex([], name="time"), axis=0)
    for col in SOLUTION_COLUMNS[:3]:
        if col not in frame.columns:
            frame[col] = pd.Series(np.nan, index=frame.index, dtype=float)
    ct = np.nan if conv_time is None else float(conv_time.total_seconds()) / 60.0
    frame["ct_min"] = pd.Series(ct, index=frame.index, dtype=float)
    return frame


def ar_phase_kwargs(residuals, variances, phase_rows, ambiguity_idx, ekf=None) -> dict:
    """The ``phase_residuals`` / ``phase_weights`` keywords of a PPP-AR entry point.

    ``residuals``: the postfit residuals ``z - h(x)`` of the epoch's float update [m] over every row; ``variances``:
    the diagonal of its R [m^2]; ``phase_rows``: the phase rows; ``ambiguity_idx`` (same length): the state index of
    the ambiguity each phase row carries; ``ekf``: the filter just after that update, whose redundancy numbers
    (``pppar.update_redundancy``) give the variance factor its degrees of freedom (s0^2 = v'R^-1 v / dof). Returns
    ``pppar.ar_phase_series`` as keywords (a rejected row, R >= ``REJECTED_R``, gets a weight the AR step ignores).
    Every AR entry point of ``ppp/pppar.py`` takes them (``apply_conventional_pppar`` accepts and ignores them).
    """
    redundancy = None if ekf is None else update_redundancy(ekf)
    v, w = ar_phase_series(residuals, variances, phase_rows, ambiguity_idx, redundancy=redundancy)
    return {"phase_residuals": v, "phase_weights": w}


def convergence_hours(conv_time):
    """The convergence time returned by ``run_filter`` (4th element, ``PPPResult.convergence``): hours, or None."""
    return None if conv_time is None else conv_time.total_seconds() / 3600


def empty_residual_frame(template, columns) -> pd.DataFrame:
    """A residual frame with no rows: the columns of ``template`` (the system's observation frame) plus ``columns``
    (a processed system without residual rows gets an empty frame with the usual columns, not None)."""
    if isinstance(template, pd.DataFrame):
        out = template.iloc[0:0].copy()
    else:
        out = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["sv", "time"]))
    for col in columns:
        if col not in out.columns:
            out[col] = pd.Series(dtype=float)
    return out


def residual_frames(frames_by_system, templates, columns) -> dict:
    """{system: residual DataFrame} for every processed system: the concatenated per-epoch
    frames, or :func:`empty_residual_frame` of ``templates[system]`` and ``columns`` when the system has none."""
    out = {}
    for system, frames in frames_by_system.items():
        frames = [f for f in frames if f is not None]
        out[system] = pd.concat(frames) if frames else empty_residual_frame(templates.get(system), columns)
    return out
