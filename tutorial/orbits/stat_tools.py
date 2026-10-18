import numpy as np
import pandas as pd

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def orbit_stats(
    df: pd.DataFrame,
    columns=None,              # list of columns or a single column name
    stats=('min', 'max', 'mean', 'median', '95%', 'rms')  # statistics to compute
):
    """
    Compute per-satellite statistics of the selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a MultiIndex (sv, time), e.g. the result of `SISSession.run()`.
    columns : str or list
        Column or list of columns to analyse.
    stats : tuple or list
        Statistics to compute: 'min', 'max', 'mean', 'median', '95%' (the 95th
        percentile of the signed values), 'rms'. NaN values are left out.

    Returns
    -------
    stat_df : pd.DataFrame
        DataFrame indexed by sv, with one column per column and statistic
        (named `<column>_<statistic>`, e.g. `sisre_rms`).
    """
    if columns is None:
        columns = df.columns
    if isinstance(columns, str):
        columns = [columns]
    columns = [col for col in columns if col in df.columns]
    if not columns:
        raise ValueError("None of the requested columns are present in the input dataframe.")

    # statistic names -> functions
    stat_funcs = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'median': np.median,
        '95%': lambda x: np.percentile(x, 95),
        'rms': rms,
    }
    # check the requested statistics
    for s in stats:
        if s not in stat_funcs:
            raise ValueError(f"Statistic '{s}' is not supported. Allowed: {list(stat_funcs)}")

    # results
    result = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            continue
        grouped = series.groupby('sv')
        col_stats = {}
        for stat in stats:
            col_stats[stat] = grouped.apply(stat_funcs[stat])
        # column names such as 'sisre_mean'
        col_stats_df = pd.concat(
            [col_stats[stat].rename(f'{col}_{stat}') for stat in stats], axis=1
        )
        result[col] = col_stats_df

    # join the columns (when there are several)
    if not result:
        raise ValueError("Requested columns are present, but none contain numeric values.")
    if len(result) == 1:
        return next(iter(result.values()))
    else:
        return pd.concat(result.values(), axis=1)
