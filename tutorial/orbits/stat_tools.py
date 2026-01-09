import numpy as np
import pandas as pd

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def orbit_stats(
    df: pd.DataFrame,
    columns=None,              # lista kolumn lub pojedyncza nazwa kolumny
    stats=('min', 'max', 'mean', 'median', '95%', 'rms')  # lista wybranych statystyk
):
    """
    Oblicza statystyki dla wybranych kolumn po sv.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z MultiIndex (sv, time).
    columns : str or list
        Kolumna lub lista kolumn do analizy.
    stats : tuple or list
        Statystyki do obliczenia: 'min', 'max', 'mean', 'median', '95%', 'rms'
    
    Returns
    -------
    stat_df : pd.DataFrame
        DataFrame: indeks sv, kolumny: [kolumna_statystyka]
    """
    if columns is None:
        columns = df.columns
    if isinstance(columns, str):
        columns = [columns]
    
    # Mapowanie nazw statystyk na funkcje
    stat_funcs = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'median': np.median,
        '95%': lambda x: np.percentile(x, 95),
        'rms': rms,
    }
    # Sprawdź dostępność żądanych statystyk
    for s in stats:
        if s not in stat_funcs:
            raise ValueError(f"Statystyka '{s}' nieobsługiwana. Dozwolone: {list(stat_funcs)}")

    # Wyniki
    result = {}
    for col in columns:
        grouped = df[col].groupby('sv')
        col_stats = {}
        for stat in stats:
            col_stats[stat] = grouped.apply(stat_funcs[stat])
        # Nadaj nazwę kolumnom np. 'sisre_mean'
        col_stats_df = pd.concat(
            [col_stats[stat].rename(f'{col}_{stat}') for stat in stats], axis=1
        )
        result[col] = col_stats_df

    # Połącz (jeśli kilka kolumn)
    if len(result) == 1:
        return next(iter(result.values()))
    else:
        return pd.concat(result.values(), axis=1)

