import numpy as np

from ..utils import calculate_distance


def code_screening(x, satellites, code_obs, thr=1):
    """
    Screening of code observations for outliers.
    """
    dist = calculate_distance(satellites, x)
    prefit = code_obs - dist
    median_prefit = np.median(prefit)
    mask = (prefit >= (median_prefit - thr)) & (prefit <= (median_prefit + thr))
    n_sat = len(prefit)
    n_bad = np.count_nonzero(~mask)
    if n_bad > n_sat / 2:
        mask = np.ones(n_sat, dtype=bool)
    return mask


def phase_residuals_outliers(sat_list, phase_residuals_dict, num, thr=10):
    """
    Find outlier indices based on phase residual prefit differences between epochs.
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
