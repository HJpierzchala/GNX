"""Statistical helpers of the PPP tutorial (``precise_point_positioning.ipynb``)."""

import numpy as np
import pandas as pd


def measure_convergence_time(df: pd.DataFrame,
                             threshold: float = 0.05,
                             mode: str = '3D',
                             min_stable_duration: int = 10) -> float:
    """
    Convergence time of a PPP solution measured on its accuracy: the first
    epoch at which the 2D or 3D error has stayed below ``threshold`` for
    ``min_stable_duration`` consecutive epochs.

    Unlike ``solution["ct_min"]`` of GNX-py, which the filter derives from the
    stability of its own position estimate, this measure needs the reference
    coordinates (the ``de``, ``dn``, ``du`` columns).

    Parameters
    ----------
    df : pd.DataFrame
        Solution indexed by time, with the columns ``de``, ``dn`` and ``du`` [m].
    threshold : float
        Convergence threshold of the error, in metres.
    mode : str
        ``'2D'`` (east and north) or ``'3D'`` (east, north and up).
    min_stable_duration : int
        Number of consecutive epochs below the threshold required to accept
        convergence.

    Returns
    -------
    float
        Time in minutes from the first epoch of ``df`` to the epoch at which
        the error has stayed below the threshold for ``min_stable_duration``
        epochs; ``np.nan`` when that never happens.
    """
    if not set(['de', 'dn', 'du']).issubset(df.columns):
        raise ValueError("DataFrame must contain 'de', 'dn', 'du' columns.")

    df = df.copy()
    if len(df) < min_stable_duration:
        return np.nan

    if mode == '2D':
        df['error'] = np.sqrt(df['de']**2 + df['dn']**2)
    elif mode == '3D':
        df['error'] = np.sqrt(df['de']**2 + df['dn']**2 + df['du']**2)
    else:
        raise ValueError("mode must be '2D' or '3D'")

    # first run of min_stable_duration consecutive epochs below the threshold
    below_thresh = df['error'] < threshold
    counter = 0
    for i in range(len(df)):
        if below_thresh.iloc[i]:
            counter += 1
            if counter >= min_stable_duration:
                conv_time = (df.index[i] - df.index[0]).total_seconds() / 60
                return conv_time
        else:
            counter = 0

    return np.nan  # no convergence
