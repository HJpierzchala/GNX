import pandas as pd
import numpy as np

def measure_convergence_time(df: pd.DataFrame,
                             threshold: float = 0.05,
                             mode: str = '3D',
                             min_stable_duration: int = 10) -> float:
    """
    Oblicza czas konwergencji PPP jako pierwszą epokę, po której błąd
    (2D lub 3D) spada poniżej progu i już nie rośnie powyżej niego.
    
    Parametry:
    ----------
    df : pd.DataFrame
        DataFrame z indeksem time i kolumnami ['de', 'dn', 'du'].
    threshold : float
        Próg błędu konwergencji w metrach.
    mode : str
        '2D' lub '3D' — wybór sposobu obliczania błędu.
    min_stable_duration : int
        Liczba kolejnych epok poniżej progu wymagana do uznania konwergencji.
    
    Zwraca:
    -------
    float:
        Czas (w minutach od początku), w którym nastąpiła trwała konwergencja.
        Zwraca np.nan jeśli konwergencja nie wystąpiła.
    """
    if not set(['de', 'dn', 'du']).issubset(df.columns):
        raise ValueError("DataFrame must contain 'de', 'dn', 'du' columns.")

    df = df.copy()
    interval = (df.index.tolist()[1]-df.index.tolist()[0]).total_seconds()/60

    if mode == '2D':
        df['error'] = np.sqrt(df['de']**2 + df['dn']**2)
    elif mode == '3D':
        df['error'] = np.sqrt(df['de']**2 + df['dn']**2 + df['du']**2)
    else:
        raise ValueError("mode must be '2D' or '3D'")

    # Wyszukiwanie konwergencji
    below_thresh = df['error'] < threshold
    counter = 0
    for i in range(len(df)):
        if below_thresh.iloc[i]:
            counter += 1
            if counter >= min_stable_duration:
                conv_time = (df.index[i] - df.index[0]).total_seconds() / 60
                return conv_time*interval
        else:
            counter = 0

    return np.nan  # jeśli nie osiągnięto konwergencji
