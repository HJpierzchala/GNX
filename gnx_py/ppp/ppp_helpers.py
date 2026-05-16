"""Shared PPP helper utilities.

The functions here are intentionally small and side-effect-light. Trace helpers
provide a single controlled text output path for ``trace_filter`` diagnostics;
screening helpers implement simple residual gates used by several legacy and
active PPP filters.

Status:
    Active shared helper module. Residual screening helpers are numerical
    safeguards and should be changed only with filter-level regression tests.
"""

import numpy as np

from ..utils import calculate_distance


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
    """
    prefit_entries = []
    for idx, sv in enumerate(sat_list):
        prev_residual = phase_residuals_dict[num - 1].get(sv)
        current_residual = phase_residuals_dict[num].get(sv)
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
