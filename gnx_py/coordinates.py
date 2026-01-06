import concurrent.futures
import datetime
import math
import warnings
import numpy as np
import pandas as pd
import gnx_py.time
from .utils import calculate_distance
from typing import Optional, Union

try:
    RankWarning = np.exceptions.RankWarning  # NumPy >=2.0
except AttributeError:
    try:
        RankWarning = np.RankWarning        # NumPy <2.0
    except AttributeError:
        RankWarning = Warning

# ---------------- Numba guard ----------------
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

# ---------------- Stałe ----------------
C_LIGHT = 299_792_458.0           # [m/s]
DT_VEL  = 5e-6                  # [s] – pół-okno do prędkości (±0.5 µs)
pi = np.pi
WEEK = 604800.0

def _process_satellite_interpolation(args):
    """
        Processes interpolation for a single satellite.

        args – dictionary containing:
          - sat: satellite identifier
          - sat_df: DataFrame with data for the satellite (sorted by toc)
          - epoch_times: NumPy array of continuous epoch times (in seconds)
          - epochs: list of epochs (datetime) corresponding to epoch_times
          - method: interpolation method ('chebyshev' or 'cubic_spline')
          - window_size: initial window size
          - min_window_size: minimum window size
          - delta_t: small time step for calculating velocity (seconds)
        Returns a list of dictionaries with interpolation results for a given satellite.
        """
    sat = args['sat']
    sat_df = args['sat_df']
    epoch_times = args['epoch_times']
    epochs = args['epochs']
    method = args['method']
    window_size = args['window_size']
    min_window_size = args['min_window_size']
    delta_t = args['delta_t']

    results = []
    toc = sat_df['toc'].values  # ciągły czas (w sekundach)
    x = sat_df['x'].values
    y = sat_df['y'].values
    z = sat_df['z'].values
    clk = sat_df['clk'].values

    for epoch_time, epoch in zip(epoch_times, epochs):
        window_found = False
        current_window_size = window_size

        while current_window_size >= min_window_size and not window_found:
            half_window = (current_window_size - 1) * 300 / 2  # 5-minutowe interwały (300 s)
            indices = np.where((toc >= epoch_time - half_window) & (toc < epoch_time + half_window))[0]

            if len(indices) >= min_window_size:
                window_found = True
                toc_window = toc[indices]
                x_window = x[indices]
                y_window = y[indices]
                z_window = z[indices]
                clk_window = clk[indices]

                t_min = toc_window.min()
                t_max = toc_window.max()
                # Normalizacja czasu do przedziału [0, 2] (zachowujemy proporcje)
                toc_normalized = 2 * (toc_window - t_min) / (t_max - t_min)
                epoch_time_normalized = 2 * (epoch_time - t_min) / (t_max - t_min)
                epoch_time_plus_dt_normalized = 2 * ((epoch_time + delta_t) - t_min) / (t_max - t_min)
                epoch_time_minus_dt_normalized = 2 * ((epoch_time - delta_t) - t_min) / (t_max - t_min)

                degree = len(toc_normalized) - 1

                if method == 'chebyshev':
                    with warnings.catch_warnings():
                        # Dopasowanie wielomianu Czebyszewa
                        cheb_x = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, x_window, deg=degree)
                        cheb_y = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, y_window, deg=degree)
                        cheb_z = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, z_window, deg=degree)
                    interpolated_x = cheb_x(epoch_time_normalized)
                    interpolated_y = cheb_y(epoch_time_normalized)
                    interpolated_z = cheb_z(epoch_time_normalized)
                    interpolated_x_dt = cheb_x(epoch_time_plus_dt_normalized)
                    interpolated_y_dt = cheb_y(epoch_time_plus_dt_normalized)
                    interpolated_z_dt = cheb_z(epoch_time_plus_dt_normalized)
                    interpolated_x_min_dt = cheb_x(epoch_time_minus_dt_normalized)
                    interpolated_y_min_dt = cheb_y(epoch_time_minus_dt_normalized)
                    interpolated_z_min_dt = cheb_z(epoch_time_minus_dt_normalized)
                else:
                    raise ValueError("Unsupported interpolation method.")

                # Obliczenie prędkości metodą różnic centralnych (skalujemy do metrów na sekundę)
                vx = (interpolated_x_dt - interpolated_x_min_dt) / (2 * delta_t) * 1000
                vy = (interpolated_y_dt - interpolated_y_min_dt) / (2 * delta_t) * 1000
                vz = (interpolated_z_dt - interpolated_z_min_dt) / (2 * delta_t) * 1000

                # Interpolacja korekty zegara – liniowa interpolacja pomiędzy dwoma najbliższymi punktami
                idx = np.searchsorted(toc_window, epoch_time)
                if idx == 0:
                    idx0, idx1 = 0, 1
                elif idx == len(toc_window):
                    idx0, idx1 = len(toc_window) - 2, len(toc_window) - 1
                else:
                    idx0, idx1 = idx - 1, idx
                t0, t1 = toc_window[idx0], toc_window[idx1]
                clk0, clk1 = clk_window[idx0], clk_window[idx1]
                if epoch_time == t1:
                    interpolated_clk = clk1
                elif epoch_time == t0:
                    interpolated_clk = clk0
                else:
                    if t1 != t0:
                        interpolated_clk = clk0 + (clk1 - clk0) * (epoch_time - t0) / (t1 - t0)
                    else:
                        interpolated_clk = clk0
                interpolated_clk *= 1e-6  # przeskalowanie jednostek

                c = 299792458
                r_vec = np.array([interpolated_x, interpolated_y, interpolated_z]) * 1000  # m
                v_vec = np.array([vx, vy, vz])
                dot_product = np.dot(r_vec, v_vec)
                delta_t_rel = -2 * dot_product / c ** 2  # w sekundach
                corrected_clk = interpolated_clk + delta_t_rel

                results.append({
                    'sv': sat,
                    'epoch': epoch,
                    'x': interpolated_x,
                    'y': interpolated_y,
                    'z': interpolated_z,
                    'clk': corrected_clk,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'dt_rel': delta_t_rel,
                    'clk_raw':interpolated_clk
                })
            else:
                current_window_size -= 1


        if not window_found:
            max_half = (window_size - 1) * 300 / 2
            t_min = toc.min()
            t_max = toc.max()
            start = max(t_min, epoch_time - max_half)
            end = min(t_max, epoch_time + max_half)
            indices = np.where((toc >= start) & (toc <= end))[0]

            if len(indices) >= min_window_size:
                toc_window = toc[indices]
                x_window = x[indices]
                y_window = y[indices]
                z_window = z[indices]
                clk_window = clk[indices]

                t_min = toc_window.min()
                t_max = toc_window.max()
                toc_normalized = 2 * (toc_window - t_min) / (t_max - t_min)
                epoch_time_normalized = 2 * (epoch_time - t_min) / (t_max - t_min)
                epoch_time_plus_dt_normalized = 2 * ((epoch_time + delta_t) - t_min) / (t_max - t_min)
                epoch_time_minus_dt_normalized = 2 * ((epoch_time - delta_t) - t_min) / (t_max - t_min)

                degree = len(toc_normalized) - 1

                if method == 'chebyshev':
                    with warnings.catch_warnings():
                        cheb_x = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, x_window, deg=degree)
                        cheb_y = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, y_window, deg=degree)
                        cheb_z = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, z_window, deg=degree)
                    interpolated_x = cheb_x(epoch_time_normalized)
                    interpolated_y = cheb_y(epoch_time_normalized)
                    interpolated_z = cheb_z(epoch_time_normalized)
                    interpolated_x_dt = cheb_x(epoch_time_plus_dt_normalized)
                    interpolated_y_dt = cheb_y(epoch_time_plus_dt_normalized)
                    interpolated_z_dt = cheb_z(epoch_time_plus_dt_normalized)
                    interpolated_x_min_dt = cheb_x(epoch_time_minus_dt_normalized)
                    interpolated_y_min_dt = cheb_y(epoch_time_minus_dt_normalized)
                    interpolated_z_min_dt = cheb_z(epoch_time_minus_dt_normalized)
                else:
                    raise ValueError("Unsupported interpolation method.")

                vx = (interpolated_x_dt - interpolated_x_min_dt) / (2 * delta_t) * 1000
                vy = (interpolated_y_dt - interpolated_y_min_dt) / (2 * delta_t) * 1000
                vz = (interpolated_z_dt - interpolated_z_min_dt) / (2 * delta_t) * 1000

                idx = np.searchsorted(toc_window, epoch_time)
                if idx == 0:
                    idx0, idx1 = 0, 1
                elif idx == len(toc_window):
                    idx0, idx1 = len(toc_window) - 2, len(toc_window) - 1
                else:
                    idx0, idx1 = idx - 1, idx
                t0, t1 = toc_window[idx0], toc_window[idx1]
                clk0, clk1 = clk_window[idx0], clk_window[idx1]
                if epoch_time == t1:
                    interpolated_clk = clk1
                elif epoch_time == t0:
                    interpolated_clk = clk0
                else:
                    interpolated_clk = clk0 + (clk1 - clk0) * (epoch_time - t0) / (t1 - t0) if t1 != t0 else clk0
                interpolated_clk *= 1e-6

                c = 299792458
                r_vec = np.array([interpolated_x, interpolated_y, interpolated_z]) * 1000
                v_vec = np.array([vx, vy, vz])
                dot_product = np.dot(r_vec, v_vec)
                delta_t_rel = -2 * dot_product / c ** 2
                corrected_clk = interpolated_clk + delta_t_rel

                results.append({
                    'sv': sat,
                    'epoch': epoch,
                    'x': interpolated_x,
                    'y': interpolated_y,
                    'z': interpolated_z,
                    'clk': corrected_clk,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'dt_rel': delta_t_rel,
                    'clk_raw': interpolated_clk
                })

    return results


class SP3InterpolatorOptimized:
    def __init__(self, sp3_dataframe):
        """
        Inicjalizacja z danymi SP3.
        """
        self.sp3_df = sp3_dataframe.copy()
        self.prepare_data()

    def prepare_data(self):
        # Konwersja kolumny 'epoch' do daty
        self.sp3_df.loc[:,'epoch'] = pd.to_datetime(self.sp3_df['epoch'], utc=False)
        self.sp3_df['gps_week'], self.sp3_df['tow'] = zip(*self.sp3_df['epoch'].apply(self.datetime2gpsweek_and_tow))
        min_gps_week = self.sp3_df['gps_week'].min()
        self.sp3_df['continuous_time'] = (self.sp3_df['gps_week'] - min_gps_week) * 604800 + self.sp3_df['tow']
        self.sp3_df['toc'] = self.sp3_df['continuous_time']
        self.satellites = self.sp3_df['sat'].unique()
        self.sat_data = {}
        for sat in self.satellites:
            sat_df = self.sp3_df[self.sp3_df['sat'] == sat].sort_values('toc')
            self.sat_data[sat] = sat_df

    def datetime2gpsweek_and_tow(self, dt):
        gps_epoch = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
        delta = dt - gps_epoch
        delta_seconds = delta.total_seconds()
        gps_week = int(delta_seconds // 604800)
        tow = delta_seconds % 604800
        return gps_week, tow

    def interpolate(self, epochs, method='chebyshev', window_size=12, min_window_size=5, delta_t=0.5e-6):
        """
        Przeprowadza interpolację pozycji satelitów dla podanych epok, równolegle przetwarzając dane dla każdego satelity.
        """
        interpolated_positions = []
        # Przeliczenie epok na GPS week i ciągły czas
        epoch_weeks, epoch_tows = zip(*[self.datetime2gpsweek_and_tow(epoch) for epoch in epochs])
        min_gps_week = self.sp3_df['gps_week'].min()
        epoch_times = np.array([(week - min_gps_week) * 604800 + tow for week, tow in zip(epoch_weeks, epoch_tows)])

        # Przygotowanie argumentów dla równoległego przetwarzania – każdy satelita osobno
        tasks = []
        for sat in self.satellites:
            sat_df = self.sat_data[sat]
            if sat_df.empty:
                continue
            task = {
                'sat': sat,
                'sat_df': sat_df,
                'epoch_times': epoch_times,
                'epochs': epochs,
                'method': method,
                'window_size': window_size,
                'min_window_size': min_window_size,
                'delta_t': delta_t
            }
            tasks.append(task)

        # Równoległe przetwarzanie z użyciem ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(_process_satellite_interpolation, tasks))

        # Scal wyniki – results to lista list słowników
        for sat_results in results:
            interpolated_positions.extend(sat_results)

        interpolated_df = pd.DataFrame(interpolated_positions)
        interpolated_df.set_index(['sv', 'epoch'], inplace=True)
        return interpolated_df

    def include_adjacent_data(self, prev_sp3_df=None, next_sp3_df=None):
        if prev_sp3_df is not None:
            prev_sp3_df['epoch'] = pd.to_datetime(prev_sp3_df['epoch'])
            combined_df = pd.concat([prev_sp3_df, self.sp3_df])
            self.sp3_df = combined_df.reset_index(drop=True)
        if next_sp3_df is not None:
            next_sp3_df['epoch'] = pd.to_datetime(next_sp3_df['epoch'])
            combined_df = pd.concat([self.sp3_df, next_sp3_df])
            self.sp3_df = combined_df.reset_index(drop=True)
        self.sp3_df = self.sp3_df.drop_duplicates(subset=['sat', 'epoch'], keep='first')
        self.prepare_data()

    def run(self, epochs, method, prev_sp3_df=None, next_sp3_df=None):
        if (prev_sp3_df is not None) or (next_sp3_df is not None):
            self.include_adjacent_data(prev_sp3_df=prev_sp3_df,next_sp3_df=next_sp3_df)
        interpolated_positions = self.interpolate(epochs=epochs, method=method)
        interpolated_positions = interpolated_positions.swaplevel(0, 1)
        interpolated_positions[['x', 'y', 'z']] = interpolated_positions[['x', 'y', 'z']].apply(lambda x: x * 1000)
        return interpolated_positions

def _process_satellite_tuple(args):
    """
    Przetwarza interpolację dla jednego tuple: (sat, [epoch1, epoch2, ...], [rec_epoch1, ...]).
    Zwraca listę słowników z wynikami interpolacji dla danego satelity.

    args – słownik zawierający:
      - sat: identyfikator satelity
      - epochs: lista epok (datetime) do interpolacji (czas emisji sygnału)
      - rec_epochs: odpowiadająca lista epok obserwacji (do indeksowania wyniku)
      - sat_df: DataFrame z danymi SP3 dla satelity (posortowany według 'toc')
      - method: metoda interpolacji ('chebyshev' lub 'cubic_spline')
      - window_size: początkowy rozmiar okna
      - min_window_size: minimalny rozmiar okna
      - delta_t: mały krok czasu (w sekundach) do obliczania prędkości
      - datetime2gpsweek_and_tow: funkcja konwertująca datetime na (gps_week, tow)
      - min_gps_week: minimalna wartość gps_week z całego zbioru
    """
    sat = args['sat']
    epochs = args['epochs']
    rec_epochs = args['rec_epochs']
    sat_df = args['sat_df']
    method = args['method']
    window_size = args['window_size']
    min_window_size = args['min_window_size']
    delta_t = args['delta_t']
    datetime2gpsweek_and_tow = args['datetime2gpsweek_and_tow']
    min_gps_week = args['min_gps_week']

    results = []
    toc = sat_df['toc'].values
    x = sat_df['x'].values
    y = sat_df['y'].values
    z = sat_df['z'].values
    clk = sat_df['clk'].values

    # Konwersja listy epok na ciągłe czasy
    epoch_weeks, epoch_tows = zip(*[datetime2gpsweek_and_tow(epoch) for epoch in epochs])
    epoch_times = np.array([(week - min_gps_week) * 604800 + tow for week, tow in zip(epoch_weeks, epoch_tows)])

    # Dla każdej epoki z listy (dla satelity)
    for epoch_time, epoch, rec_epoch in zip(epoch_times, epochs, rec_epochs):
        window_found = False
        current_window_size = window_size

        while current_window_size >= min_window_size and not window_found:
            half_window = (current_window_size - 1) * 300 / 2  # Zakładamy 5-minutowe interwały (300 s)
            indices = np.where((toc >= epoch_time - half_window) & (toc < epoch_time + half_window))[0]

            if len(indices) >= min_window_size:
                window_found = True
                toc_window = toc[indices]
                x_window = x[indices]
                y_window = y[indices]
                z_window = z[indices]
                clk_window = clk[indices]

                t_min = toc_window.min()
                t_max = toc_window.max()
                # Normalizacja czasu – zachowujemy oryginalne przeliczenie (2 * ...)
                toc_normalized = 2 * (toc_window - t_min) / (t_max - t_min)
                epoch_time_normalized = 2 * (epoch_time - t_min) / (t_max - t_min)
                epoch_time_plus_dt_normalized = 2 * ((epoch_time + delta_t) - t_min) / (t_max - t_min)
                epoch_time_minus_dt_normalized = 2 * ((epoch_time - delta_t) - t_min) / (t_max - t_min)

                if method == 'chebyshev':
                    with warnings.catch_warnings():
                        # warnings.simplefilter('ignore', RankWarning)
                        cheb_x = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, x_window,
                                                                       deg=len(toc_normalized) - 1)
                        cheb_y = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, y_window,
                                                                       deg=len(toc_normalized) - 1)
                        cheb_z = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, z_window,
                                                                       deg=len(toc_normalized) - 1)
                    interpolated_x = cheb_x(epoch_time_normalized)
                    interpolated_y = cheb_y(epoch_time_normalized)
                    interpolated_z = cheb_z(epoch_time_normalized)
                    interpolated_x_dt = cheb_x(epoch_time_plus_dt_normalized)
                    interpolated_y_dt = cheb_y(epoch_time_plus_dt_normalized)
                    interpolated_z_dt = cheb_z(epoch_time_plus_dt_normalized)
                    interpolated_x_min_dt = cheb_x(epoch_time_minus_dt_normalized)
                    interpolated_y_min_dt = cheb_y(epoch_time_minus_dt_normalized)
                    interpolated_z_min_dt = cheb_z(epoch_time_minus_dt_normalized)

                else:
                    raise ValueError("Unsupported interpolation method.")

                vx = (interpolated_x_dt - interpolated_x_min_dt) / (2 * delta_t) * 1000
                vy = (interpolated_y_dt - interpolated_y_min_dt) / (2 * delta_t) * 1000
                vz = (interpolated_z_dt - interpolated_z_min_dt) / (2 * delta_t) * 1000

                idx = np.searchsorted(toc_window, epoch_time)
                if idx == 0:
                    idx0, idx1 = 0, 1
                elif idx == len(toc_window):
                    idx0, idx1 = len(toc_window) - 2, len(toc_window) - 1
                else:
                    idx0, idx1 = idx - 1, idx
                t0 = toc_window[idx0]
                t1 = toc_window[idx1]
                clk0 = clk_window[idx0]
                clk1 = clk_window[idx1]
                if epoch_time == t1:
                    interpolated_clk = clk1
                elif epoch_time == t0:
                    interpolated_clk = clk0
                else:
                    if t1 != t0:
                        interpolated_clk = clk0 + (clk1 - clk0) * (epoch_time - t0) / (t1 - t0)
                    else:
                        interpolated_clk = clk0
                interpolated_clk *= 1e-6

                c = 299792458  # prędkość światła w m/s
                r_vec = np.array([interpolated_x, interpolated_y, interpolated_z]) * 1000
                v_vec = np.array([vx, vy, vz])
                dot_product = np.dot(r_vec, v_vec)
                delta_t_rel = -2 * dot_product / c ** 2
                corrected_clk = interpolated_clk + delta_t_rel

                results.append({
                    'sv': sat,
                    'epoch': rec_epoch,  # Używamy rec_epoch jako docelowego indeksu
                    'em_epoch': epoch,  # epoka emisji sygnału
                    'x': interpolated_x,
                    'y': interpolated_y,
                    'z': interpolated_z,
                    'clk': corrected_clk,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'dt_rel': delta_t_rel,
                    'clk_raw':interpolated_clk
                })
            else:
                current_window_size -= 1

        # if not window_found:
        #     print(f"No sufficient data to interpolate satellite {sat} at epoch {epoch}")
        if not window_found:
            # Próba rozszerzenia okna tylko w kierunku dostępnych danych
            max_half = (window_size - 1) * 300 / 2
            t_min = toc.min()
            t_max = toc.max()
            start = max(t_min, epoch_time - max_half)
            end = min(t_max, epoch_time + max_half)
            indices = np.where((toc >= start) & (toc <= end))[0]

            if len(indices) >= min_window_size:
                toc_window = toc[indices]
                x_window = x[indices]
                y_window = y[indices]
                z_window = z[indices]
                clk_window = clk[indices]

                t_min = toc_window.min()
                t_max = toc_window.max()
                toc_normalized = 2 * (toc_window - t_min) / (t_max - t_min)
                epoch_time_normalized = 2 * (epoch_time - t_min) / (t_max - t_min)
                epoch_time_plus_dt_normalized = 2 * ((epoch_time + delta_t) - t_min) / (t_max - t_min)
                epoch_time_minus_dt_normalized = 2 * ((epoch_time - delta_t) - t_min) / (t_max - t_min)

                degree = len(toc_normalized) - 1

                if method == 'chebyshev':
                    with warnings.catch_warnings():
                        # warnings.simplefilter('ignore', RankWarning)
                        cheb_x = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, x_window, deg=degree)
                        cheb_y = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, y_window, deg=degree)
                        cheb_z = np.polynomial.chebyshev.Chebyshev.fit(toc_normalized, z_window, deg=degree)
                    interpolated_x = cheb_x(epoch_time_normalized)
                    interpolated_y = cheb_y(epoch_time_normalized)
                    interpolated_z = cheb_z(epoch_time_normalized)
                    interpolated_x_dt = cheb_x(epoch_time_plus_dt_normalized)
                    interpolated_y_dt = cheb_y(epoch_time_plus_dt_normalized)
                    interpolated_z_dt = cheb_z(epoch_time_plus_dt_normalized)
                    interpolated_x_min_dt = cheb_x(epoch_time_minus_dt_normalized)
                    interpolated_y_min_dt = cheb_y(epoch_time_minus_dt_normalized)
                    interpolated_z_min_dt = cheb_z(epoch_time_minus_dt_normalized)

                else:
                    raise ValueError("Unsupported interpolation method.")

                vx = (interpolated_x_dt - interpolated_x_min_dt) / (2 * delta_t) * 1000
                vy = (interpolated_y_dt - interpolated_y_min_dt) / (2 * delta_t) * 1000
                vz = (interpolated_z_dt - interpolated_z_min_dt) / (2 * delta_t) * 1000

                idx = np.searchsorted(toc_window, epoch_time)
                if idx == 0:
                    idx0, idx1 = 0, 1
                elif idx == len(toc_window):
                    idx0, idx1 = len(toc_window) - 2, len(toc_window) - 1
                else:
                    idx0, idx1 = idx - 1, idx
                t0, t1 = toc_window[idx0], toc_window[idx1]
                clk0, clk1 = clk_window[idx0], clk_window[idx1]
                if epoch_time == t1:
                    interpolated_clk = clk1
                elif epoch_time == t0:
                    interpolated_clk = clk0
                else:
                    interpolated_clk = clk0 + (clk1 - clk0) * (epoch_time - t0) / (t1 - t0) if t1 != t0 else clk0
                interpolated_clk *= 1e-6

                c = 299792458
                r_vec = np.array([interpolated_x, interpolated_y, interpolated_z]) * 1000
                v_vec = np.array([vx, vy, vz])
                dot_product = np.dot(r_vec, v_vec)
                delta_t_rel = -2 * dot_product / c ** 2
                corrected_clk = interpolated_clk + delta_t_rel

                results.append({
                    'sv': sat,
                    'epoch': epoch,
                    'x': interpolated_x,
                    'y': interpolated_y,
                    'z': interpolated_z,
                    'clk': corrected_clk,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'dt_rel': delta_t_rel,
                    'clk_raw': interpolated_clk
                })
            else:
                print(f"No sufficient data (even at edge) to interpolate satellite {sat} at epoch {epoch}")

    return results


class SP3CustomInterpolatorOptimized(SP3InterpolatorOptimized):
    def interpolate_for_tuples(self, satellite_epochs_tuples, method='chebyshev', window_size=12, min_window_size=5,
                               delta_t=0.5e-6):
        """
        Interpoluje pozycje i korekty zegara dla podanych tuple:
          (satellite, [epoch1, epoch2, ...], [obs_epoch1, obs_epoch2, ...]).
        Zastosowano równoległe przetwarzanie na poziomie satelitów.
        Zwraca DataFrame z wynikami.
        """
        all_results = []
        # Przygotowanie wspólnych danych
        min_gps_week = self.sp3_df['gps_week'].min()

        # Przygotowanie listy zadań dla równoległego przetwarzania
        tasks = []
        for tup in satellite_epochs_tuples:
            sat, epochs, rec_epochs = tup
            if sat not in self.sat_data or self.sat_data[sat].empty:
                print(f"Brak danych dla satelity {sat}")
                continue
            task = {
                'sat': sat,
                'epochs': epochs,
                'rec_epochs': rec_epochs,
                'sat_df': self.sat_data[sat],
                'method': method,
                'window_size': window_size,
                'min_window_size': min_window_size,
                'delta_t': delta_t,
                'datetime2gpsweek_and_tow': self.datetime2gpsweek_and_tow,
                'min_gps_week': min_gps_week
            }
            tasks.append(task)

        # Równoległe przetwarzanie – każdy satelita osobno
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(_process_satellite_tuple, tasks))

        # Scal wyniki – results to lista list słowników
        for sat_results in results:
            all_results.extend(sat_results)

        interpolated_df = pd.DataFrame(all_results)
        if not interpolated_df.empty:
            interpolated_df.set_index(['sv', 'epoch'], inplace=True)
        return interpolated_df

class CrdWrapper:
    """
    This class performs:
    - azimuth, elevation computation
    - earth rotation coordinates correction
    - emission time computation
    It is designed for other satellite coordinates interpolations classes to inherit from
    """
    def __init__(self, obs:pd.DataFrame, flh, xyz_a, mode, epochs:Optional=None):
        self.obs = obs
        self.epochs = epochs
        self.flh = flh
        self.xyz_a = xyz_a
        self.mode = mode

    def elevation_azimuth(self):
        df = self.obs.copy()
        lat, lon, alt = self.flh[0], self.flh[1], self.flh[2]
        xyz_point = np.asarray(self.xyz_a)
        flh_point = np.deg2rad([lat, lon, alt])
        fi, la, h = flh_point[0], flh_point[1], flh_point[2]

        e = np.array([
            - np.sin(la),
            np.cos(la),
            0
        ])
        n = np.array([
            -np.cos(la) * (np.sin(fi)),
            -np.sin(la) * np.sin(fi),
            np.cos(fi)
        ])

        u = np.array([
            np.cos(la) * np.cos(fi),
            np.sin(la) * np.cos(fi),
            np.sin(fi)
        ])

        xyz = df[['xe', 'ye', 'ze']].to_numpy()
        dist = calculate_distance(xyz, xyz_point)
        p = np.array([
            (xyz[:, 0] - xyz_point[0]) / dist,
            (xyz[:, 1] - xyz_point[1]) / dist,
            (xyz[:, 2] - xyz_point[2]) / dist
        ]).T

        ev = np.arcsin(np.dot(p, u))
        az = np.arctan2(np.dot(p, e), np.dot(p, n))
        az = (az + 2 * np.pi) % (2 * np.pi)
        df['az'] = np.rad2deg(az)
        df['ev'] = np.rad2deg(ev)
        return df

    # def correct_crd(self):
    #     xyz= self.obs[['x', 'y', 'z']].to_numpy()
    #     xyze = np.array(list(map(lambda row: correct_sat_coordinates(row,xyz_a=self.xyz_a),xyz)))
    #     xyze = pd.DataFrame(index=self.obs.index,data={'xe':xyze[:,0],'ye':xyze[:,1],'ze':xyze[:,2]})
    #     self.obs[['xe_ur', 'ye_ur','ze_ur']] = self.obs[['x', 'y', 'z']].copy()
    #     self.obs[['xe', 'ye', 'ze']] = xyze
    #     return self.obs
    def correct_crd(self):
        xyz = self.obs[['x', 'y', 'z']].to_numpy(dtype=np.float64, copy=False)
        a = np.asarray(self.xyz_a, dtype=np.float64)

        dx = xyz[:, 0] - a[0]
        dy = xyz[:, 1] - a[1]
        dz = xyz[:, 2] - a[2]

        c = 299792458.0
        omega = 7.2921159e-5

        dT = np.sqrt(dx * dx + dy * dy + dz * dz) / c
        omw = dT * omega

        cos0 = np.cos(omw)
        sin0 = np.sin(omw)

        xe = cos0 * xyz[:, 0] + sin0 * xyz[:, 1]
        ye = -sin0 * xyz[:, 0] + cos0 * xyz[:, 1]
        ze = xyz[:, 2]

        self.obs[['xe_ur', 'ye_ur', 'ze_ur']] = self.obs[['x', 'y', 'z']].to_numpy(copy=True)
        self.obs[['xe', 'ye', 'ze']] = np.column_stack((xe, ye, ze))
        return self.obs

    def get_emission_epochs(self):
        _c = 299792458
        if self.mode == 'L1' or self.mode =='L1L2':
            c1_col = [col for col in self.obs.columns if col.startswith('C1')][0]
        elif self.mode == 'L5':
            c1_col = [col for col in self.obs.columns if col.startswith('C5')][0]

        if self.mode == 'L1':
            s = (self.obs[c1_col].values / _c ) - self.obs['dt_clock'].values + self.obs['TGD'].values
        elif self.mode == 'L1L2':
            s = (self.obs[c1_col].values / _c ) - self.obs['dt_clock'].values #+ self.obs['TGD'].values

        self.obs['sec'] = pd.to_timedelta(s, unit='s')
        self.obs['t_emission'] = self.obs['obs_time'] - self.obs['sec']
        return self.obs


class CustomWrapper(CrdWrapper):
    def __init__(self, obs, epochs, flh, xyz_a, mode):
        super().__init__(obs, epochs, flh, xyz_a, mode)
        self.obs = obs
        self.epochs = epochs
        self.flh = flh
        self.xyz_a = xyz_a
        self.mode = mode

    def run(self):
        self.obs = self.correct_crd()
        self.obs = self.elevation_azimuth()
        return self.obs




def emission_interp(obs, crd, prev_sp3_df, sp3_df, next_sp3_df):
    """
    Łączy dane obserwacyjne z interpolowanymi współrzędnymi.
    Funkcja pozostaje bez zmian, poza tym że wykorzystuje zoptymalizowaną
    wersję interpolatora (SP3CustomInterpolator) do obliczeń.
    :param obs - dataframe z obserwacjami (jeden sys)
    :param crd - dataframe z pozycjami satelitow (wszystkie systemy)
    :param prev_sp3_df: dataframe z dnia poprzedniego
    :param next_sp3_df: dataframe z dnia nastepnego
    :param sp3_df: dataframe z dnia obecnego
    """
    C = 299792458
    obs_reset = obs.reset_index().copy()
    crd_reset = crd.reset_index().copy()
    # laczymy dataframy zeby miec wiersz z zegarem do liczenia emisji sygnalu
    obs_crd = pd.merge(obs_reset, crd_reset, left_on=['sv', 'time'], right_on=['sv', 'epoch'])
    obs_crd.drop(columns=['epoch'], inplace=True)
    obs_crd = obs_crd.set_index(['sv', 'time'])
    if 'P3' in obs_crd.columns:
        t_em = (obs_crd['P3'] / C) + obs_crd['clk']
        obs_crd['t_em'] = t_em
    else:
        c1_col = [col for col in obs_crd.columns if col.startswith('C')][0]
        t_em = (obs_crd[c1_col] / C) + obs_crd['clk']
        obs_crd['t_em'] = t_em
    obs_crd['obs_time'] = obs_crd.index.get_level_values('time')
    obs_crd['emission_time'] = obs_crd['obs_time'] - pd.to_timedelta(t_em, unit='s')
    # Tworzymy tuple: (sat, list(emission_time), list(obs_time))
    satellite_epochs_tuples = [
        (sv, group['emission_time'].tolist(), group['obs_time'].tolist())
        for sv, group in obs_crd.reset_index().groupby('sv')
    ]
    custom_interpolator = SP3CustomInterpolatorOptimized(sp3_df)
    custom_interpolator.include_adjacent_data(prev_sp3_df=prev_sp3_df, next_sp3_df=next_sp3_df)
    # Współrzędne w czasie emisji sygnału
    interpolated_final = custom_interpolator.interpolate_for_tuples(
        satellite_epochs_tuples=satellite_epochs_tuples
    )
    # obs = obs.loc[obs_crd.index.reorder_levels(order=[1, 0])]
    # interpolated_final['t_em'] = pd.to_timedelta(t_em, unit='s').to_numpy()
    # obs['emission_time'] = obs_crd['emission_time'].copy().to_numpy()
    interpolated_final[['x','y','z']] = interpolated_final[['x','y','z']].apply(lambda x: x * 1000)
    interpolated_final.index.names = ['sv', 'time']
    interpolated_final=interpolated_final.sort_values(by='time')
    interpolated_final=interpolated_final.swaplevel(0,1)
    obs_crd = pd.merge(obs, interpolated_final, left_on=['sv', 'time'], right_on=['sv', 'time'])
    return obs_crd


def make_ionofree(obs_df,  mode,sys='G'):
    F1 = 1575.42e06
    F2 = 1227.60e06
    FE1a = F1
    FE5a = 1176.45e06
    FE5b = 1207.140e06

    # if sys == 'G':
    if mode in ['L1','L1L2']:
        c1 = [c for c in obs_df.columns if c.startswith('C1')][0]
        c2 = [c for c in obs_df.columns if c.startswith('C2')][0]
        obs_df['P3'] = 2.545 * obs_df[c1] - 1.545 * obs_df[c2]
    elif mode in ['L1L5','L5']:
        c1 = [c for c in obs_df.columns if c.startswith('C1')][0]
        c5 = [c for c in obs_df.columns if c.startswith('C5')][0]
        obs_df['P3'] = 2.2606 * obs_df[c1] - 1.2606 * obs_df[c5]
# elif sys =='E':
    if mode in ['E1','E1E5a','E5a']:
        K1 = FE1a ** 2 / (FE1a ** 2 - FE5a ** 2)
        K2 = FE5a ** 2 / (FE1a ** 2 - FE5a ** 2)
        c1 = [c for c in obs_df.columns if c.startswith('C1')][0]
        c5 = [c for c in obs_df.columns if c.startswith('C5')][0]
        obs_df['P3'] = K1 * obs_df[c1] - K2 * obs_df[c5]
    elif mode in ['E1E5b','E5b']:
        K1 = FE1a ** 2 / (FE1a ** 2 - FE5b ** 2)
        K2 = FE5b ** 2 / (FE1a ** 2 - FE5b ** 2)
        c1 = [c for c in obs_df.columns if c.startswith('C1')][0]
        c7 = [c for c in obs_df.columns if c.startswith('C7')][0]
        obs_df['P3'] = K1 * obs_df[c1] - K2 * obs_df[c7]
    # else:
    #     raise NotImplementedError('Unknown sys!')


def eccentric_anomaly(mk, e, pi):
    """
    This function is to calculate eccentric anomaly iteratively
    :param mk: mean anomaly at the epoch t
    :param e: eccentricity 'e' from nav message
    :param pi: pi number
    :return: eccentric anomaly [rad] <0, 2pi>
    """
    ek = mk
    for i in range(14):
        ek_old = ek
        ek = mk + e * np.sin(ek)
        diff = (ek - ek_old) #% (2 * pi)
        if abs(diff) < 1e-8:
            return ek% (2 * pi)
    print('Convergence not reached, diff: ', diff)
    return None


def numpy_broadcast_interpolation(message_toc, observation_toc, message, with_rel=True, rel_sep=False):
    """
    This function interpolates satellite position at given epoch using GPS broadcast navigation message
    :param message_toc: toc epoch of broadcast message
    :param observation_toc: toc epoch at which interpolation is done
    :param message: broadcast message
    :return: X,Y,Z dte (coordinates + SV clock offset correction)
    ['SVclockBias'          0,
    SVclockDrift',          1
    'SVclockDriftRate',     2
    'IODE',                 3
    'Crs',                  4
    'DeltaN',               5
    'M0',                   6
    'Cuc',                  7
    'Eccentricity',         8
    'Cus',                  9
    'sqrtA',                10
    'Toe',                  11
    'Cic',                  12
    'Omega0',               13
    'Cis',                  14
    'Io',                   15
    'Crc',                  16
    'omega',                17
    'OmegaDot',             18
    'IDOT',                 19
    'CodesL2',              20
    'GPSWeek',              21
    'L2Pflag',              22
    'SVacc',                23
    'health',               24
    'TGD',                  25
    'IODC',                 26
    'TransTime']            27
    """

    # constants
    mi = 3.986005e14  # potencjal ziemski
    # pi = 3.1415926535898  # rad
    ome = 7.2921151467e-05
    c = 299792458  # m/s
    # SV clock correction (GPST to UTC)
    a0 = message[0]
    a1 = message[1]
    a2 = message[2]
    t_toc = observation_toc - message_toc
    if t_toc < -302400:
        dT = t_toc + 604800
    elif t_toc > 302400:
        dT = t_toc - 604800
    else:
        dT = t_toc
    dt = a0 + a1 * dT + a2 * dT ** 2
    # if message[-1] =='G27':
    #     print(f'{message[-1], message[0:3], observation_toc, message_toc, dT}')
    dt_dot = a1 + 2 * a2 * dT
    # ephemeris reference epoch
    toe = message[11]
    t_toe = observation_toc - toe
    if t_toe > 302400:
        tk = t_toe - 604800
    elif t_toe < -302400:
        tk = t_toe + 604800
    else:
        tk = t_toe
    # Satellite orbit major axis
    sqrta = message[10]
    a = sqrta ** 2
    # Satellite average motion
    n0 = np.sqrt(mi / a ** 3)
    # corrected average motion
    dn = message[5]
    n = n0 + dn
    # mean anomaly at tk epoch
    m0 = message[6]
    mk = m0 + n * tk  # should be <0, 2pi>
    mk = mk % (2 * pi)
    # Eccentric anomaly
    e = message[8]
    ek = eccentric_anomaly(mk, e, pi)
    ek = ek #% (2 * pi)
    # True anomaly
    v = math.atan2((np.sqrt(1 - e ** 2) * np.sin(ek)), (np.cos(ek) - e))
    # latitude argument
    u = message[17] + v
    u = u % (2 * pi)
    # latitude argument correction
    cus = message[9]
    cuc = message[7]
    duk = cus * np.sin(2 * u) + cuc * np.cos(2 * u)
    # leading beam correction
    crs = message[4]
    crc = message[16]
    drk = crs * np.sin(2 * u) + crc * np.cos(2 * u)
    # orbit inclination correction
    cis = message[14]
    cic = message[12]
    idot = message[19]
    dik = cis * np.sin(2 * u) + cic * np.cos(2 * u) + idot * tk
    # corrected latitude argument
    uk = u + duk
    # corrected leading beam
    rk = a * (1 - e * np.cos(ek)) + drk
    # corrected orbit inclination angle
    i0 = message[15]
    ik = i0 + dik
    # corrected length of ascending orbit node
    om0 = message[13]
    omv = message[18]
    omk = om0 + (omv - ome) * tk - ome * toe
    omk = omk % (2 * pi)
    # coordinates of the satellite in the plane of orbit
    xi = rk * np.cos(uk)
    eta = rk * np.sin(uk)
    # geocentric coordinates of the satellite

    x = xi * np.cos(omk) - eta * np.cos(ik) * np.sin(omk)
    y = xi * np.sin(omk) + eta * np.cos(ik) * np.cos(omk)
    z = eta * np.sin(ik)

    # The product of the mean and eccentric anomaly velocity
    em_dot = 1 / (1 - e * np.cos(ek))
    # True anomaly velocity
    v_dot = np.sqrt((1 + e) / (1 - e)) * (1 / (np.cos(ek / 2) ** 2)) * (1 / (1 + np.tan(v / 2) ** 2)) * em_dot * n
    # argument of latitude velocity
    # cus = cus
    # cuc = cuc
    om_dot = message[18]
    u_dot = v_dot + 2 * v_dot * (cus * np.cos(2 * u) - cuc * np.sin(2 * u))
    small_om_dot = om_dot - ome
    # inclination rate
    cis = cis
    cic = cic
    i_dot = idot + 2 * v_dot * (cis * np.cos(2 * u) - cic * np.sin(2 * u))
    # leading beam rate
    r_dot = a * e * np.sin(ek) * em_dot * n + 2 * v_dot * (crs * np.cos(2 * u) - crc * np.sin(2 * u))
    # satellite velocity in the plane of an orbit
    xi_dot = r_dot * np.cos(uk) - rk * np.sin(uk) * u_dot
    eta_dot = r_dot * np.sin(uk) + rk * np.cos(uk) * u_dot
    # satellite geocentric velocity

    x_dot = (((np.cos(omk) * xi_dot - np.cos(ik) * np.sin(omk) * eta_dot
               - xi * np.sin(omk) * small_om_dot)
              - eta * np.cos(ik) * np.cos(omk) * small_om_dot)
             + eta * np.sin(ik) * np.sin(omk) * i_dot)

    y_dot = (((np.sin(omk) * xi_dot + np.cos(ik) * np.cos(omk) * eta_dot
               + xi * np.cos(omk) * small_om_dot)
              - eta * np.cos(ik) * np.sin(omk) * small_om_dot)
             - eta * np.sin(ik) * np.cos(omk) * i_dot)
    z_dot = np.sin(ik) * eta_dot + eta * np.cos(ik) * i_dot
    # relativistic effects correction
    dt_rel = -2 * (x * x_dot + y * y_dot + z * z_dot) / c ** 2
    dt_sat = dt + dt_rel
    if with_rel:
        if rel_sep:
            return np.array([x, y, z, dt, dt_rel]), np.array([x_dot, y_dot, z_dot, dt_dot])
        else:
            return np.array([x, y, z, dt_sat,x_dot, y_dot, z_dot, dt_dot])
    else:
        return np.array([x, y, z, dt]), np.array([x_dot, y_dot, z_dot, dt_dot])


def gal_numpy_broadcast_interpolation(message_toc, observation_toc, message, with_rel=True, rel_sep=False):
    """
    This function interpolates satellite position at given epoch using Galileo broadcast navigation message
    :param message_toc: toc epoch of broadcast message
    :param observation_toc: toc epoch at which interpolation is done
    :param message: broadcast message
    :return: X,Y,Z dte (coordinates + SV clock offset correction)
    ['SVclockBias'          0,
    SVclockDrift',          1
    'SVclockDriftRate',     2
    'IODnav',               3
    'Crs',                  4
    'DeltaN',               5
    'M0',                   6
    'Cuc',                  7
    'Eccentricity',         8
    'Cus',                  9
    'sqrtA',                10
    'Toe',                  11
    'Cic',                  12
    'Omega0',               13
    'Cis',                  14
    'Io',                   15
    'Crc',                  16
    'omega',                17
    'OmegaDot',             18
    'IDOT',                 19
    'DataSrc',              20
    'GALWeek',              21
    'spare0'                22
    SISA                    23
    health                  24
    BGDe5a                  25
    BGDe5b                  26
    TransTime               27
    spare1                  28
    spare2                  29
    spare3                  30
    """

    # constants
    mi = 3.986004418e14
    pi = 3.1415926535898  # rad
    ome = 0.000072921151467  # rad/s
    c = 299792458  # m/s
    # SV clock correction (GPST to UTC)
    a0 = message[0]
    a1 = message[1]
    a2 = message[2]
    t_toc = observation_toc - message_toc
    if t_toc < -302400:
        dT = t_toc + 604800
    elif t_toc > 302400:
        dT = t_toc - 604800
    else:
        dT = t_toc
    dt = a0 + a1 * dT + a2 * dT ** 2
    dt_dot = a1 + 2 * a2 *dT
    # ephemeris reference epoch
    toe = message[11]
    t_toe = observation_toc - toe
    if t_toe > 302400:
        tk = t_toe - 604800
    elif t_toe < -302400:
        tk = t_toe + 604800
    else:
        tk = t_toe
    # Satellite orbit major axis
    sqrta = message[10]
    a = sqrta ** 2
    # Satellite average motion
    n0 = np.sqrt(mi / a ** 3)
    # corrected average motion
    dn = message[5]
    n = n0 + dn
    # mean anomaly at tk epoch
    m0 = message[6]
    mk = m0 + n * tk  # should be <0, 2pi>
    mk = mk % (2 * pi)
    # Eccentric anomaly
    e = message[8]
    ek = eccentric_anomaly(mk, e, pi)

    # True anomaly
    v = math.atan2((np.sqrt(1 - e ** 2) * np.sin(ek)), (np.cos(ek) - e))
    # latitude argument
    u = message[17] + v
    u = u % (2 * pi)
    # latitude argument correction
    cus = message[9]
    cuc = message[7]
    duk = cus * np.sin(2 * u) + cuc * np.cos(2 * u)
    # leading beam correction
    crs = message[4]
    crc = message[16]
    drk = crs * np.sin(2 * u) + crc * np.cos(2 * u)
    # orbit inclination correction
    cis = message[14]
    cic = message[12]
    idot = message[19]
    dik = cis * np.sin(2 * u) + cic * np.cos(2 * u) + idot * tk
    # corrected latitude argument
    uk = u + duk
    # corrected leading beam
    rk = a * (1 - e * np.cos(ek)) + drk
    # corrected orbit inclination angle
    i0 = message[15]
    ik = i0 + dik
    # corrected length of ascending orbit node
    om0 = message[13]
    omv = message[18]
    omk = om0 + (omv - ome) * tk - ome * toe
    omk = omk % (2 * pi)
    # coordinates of the satellite in the plane of orbit
    xi = rk * np.cos(uk)
    eta = rk * np.sin(uk)
    # geocentric coordinates of the satellite
    x = xi * np.cos(omk) - eta * np.cos(ik) * np.sin(omk)
    y = xi * np.sin(omk) + eta * np.cos(ik) * np.cos(omk)
    z = eta * np.sin(ik)
    # The product of the mean and eccentric anomaly velocity
    em_dot = 1 / (1 - e * np.cos(ek))
    # True anomaly velocity
    v_dot = np.sqrt((1 + e) / (1 - e)) * (1 / (np.cos(ek / 2) ** 2)) * (1 / (1 + np.tan(v / 2) ** 2)) * em_dot * n
    # argument of latitude velocity
    cus = cus
    cuc = cuc
    om_dot = message[18]
    u_dot = v_dot + 2 * v_dot * (cus * np.cos(2 * u) - cuc * np.sin(2 * u))
    small_om_dot = om_dot - ome
    # inclination rate
    cis = cis
    cic = cic
    i_dot = idot + 2 * v_dot * (cis * np.cos(2 * u) - cic * np.sin(2 * u))
    # leading beam rate
    r_dot = a * e * np.sin(ek) * em_dot * n + 2 * v_dot * (crs * np.cos(2 * u) - crc * np.sin(2 * u))
    # satellite velocity in the plane of an orbit
    xi_dot = r_dot * np.cos(uk) - rk * np.sin(uk) * u_dot
    eta_dot = r_dot * np.sin(uk) + rk * np.cos(uk) * u_dot
    # satellite geocentric velocity
    x_dot = (((np.cos(omk) * xi_dot - np.cos(ik) * np.sin(omk) * eta_dot
               - xi * np.sin(omk) * small_om_dot)
              - eta * np.cos(ik) * np.cos(omk) * small_om_dot)
             + eta * np.sin(ik) * np.sin(omk) * i_dot)

    y_dot = (((np.sin(omk) * xi_dot + np.cos(ik) * np.cos(omk) * eta_dot
               + xi * np.cos(omk) * small_om_dot)
              - eta * np.cos(ik) * np.sin(omk) * small_om_dot)
             - eta * np.sin(ik) * np.cos(omk) * i_dot)
    z_dot = np.sin(ik) * eta_dot + eta * np.cos(ik) * i_dot
    # relativistic effects correction
    dt_rel = -2 * (x * x_dot + y * y_dot + z * z_dot) / c ** 2
    dt_sat = dt + dt_rel
    if with_rel:
        if rel_sep:
            return np.array([x, y, z, dt, dt_rel]), np.array([x_dot, y_dot, z_dot, dt_dot])
        else:
            return np.array([x, y, z, dt_sat,x_dot, y_dot, z_dot, dt_dot])
    else:
        return np.array([x, y, z, dt]), np.array([x_dot, y_dot, z_dot, dt_dot])



def _wrap_dist_sow(a_sow: np.ndarray, b_sow: np.ndarray) -> np.ndarray:
    d = np.abs(a_sow[:, None] - b_sow[None, :])
    return np.minimum(d, WEEK - d)

def _wrap_signed(t):
    return (t + WEEK/2) % WEEK - WEEK/2


def _ensure_obs_block(df_obs, sv_hint=None):
    """
        Returns df with columns 'sv','time','toc' (without messing with df.columns!).
        Accepts both MultiIndex (sv,time) and regular index.
        """
    if not isinstance(df_obs, pd.DataFrame):
        df_obs = pd.DataFrame(df_obs)

    if ('sv' not in df_obs.columns) or ('time' not in df_obs.columns):
        df_obs = df_obs.reset_index()
    if 'sv' not in df_obs.columns and sv_hint is not None:
        df_obs['sv'] = sv_hint
    if 'toc' not in df_obs.columns:
        raise ValueError("df_obs must have 'toc' column.")

    return df_obs[['sv','time','toc']].copy()


def _ensure_nav_block(df_orb, sv_hint=None):
    """
        Returns df with columns NAV + 'sv','time','Toe','nav_toc' (without messing with df.columns!).
        """
    if not isinstance(df_orb, pd.DataFrame):
        df_orb = pd.DataFrame(df_orb)

    if ('sv' not in df_orb.columns) or ('time' not in df_orb.columns):
        df_orb = df_orb.reset_index()
    if 'sv' not in df_orb.columns and sv_hint is not None:
        df_orb['sv'] = sv_hint

    need = ['sv','time','Toe','nav_toc']
    miss = [c for c in need if c not in df_orb.columns]
    if miss:
        raise ValueError(f"df_orb missing columns: {miss} (make sure _add_nav_toc() was already there).")

    return df_orb.copy()
import numpy as np
import pandas as pd


def _prep_nav_arrays(system: str, nav_block: pd.DataFrame):
    """
    Prepare nav arrays + iloc mapping (so we can filter and still return indices
    compatible with nav_block.iloc[...] later).
    """
    toe = nav_block['Toe'].to_numpy(float)
    trans = pd.to_datetime(nav_block['TransTime']) if 'TransTime' in nav_block.columns else None

    if 'SISA' in nav_block.columns:
        quality = nav_block['SISA'].to_numpy(float)
    elif 'SVacc' in nav_block.columns:
        quality = nav_block['SVacc'].to_numpy(float)
    else:
        quality = np.zeros_like(toe, dtype=float)

    orig_iloc = np.arange(len(nav_block), dtype=int)

    # GPS health filter (keep only healthy)
    if system == 'G' and 'health' in nav_block.columns:
        good = (nav_block['health'].to_numpy() == 0)
        toe = toe[good]
        quality = quality[good]
        orig_iloc = orig_iloc[good]
        if trans is not None:
            trans = trans.iloc[good].reset_index(drop=True)

    return toe, trans, quality, orig_iloc


def _pick_nav_rows_pref_future_else_past_with_hold(
    system: str,
    toc_vec: np.ndarray,
    nav_block: pd.DataFrame,
    hold_sec: float = None,
    switch_thresh_sec: float = 900.0,
    nearest_h: float | None = None,
) -> np.ndarray:
    """
    GPS-friendly hybrid:
      1) prefer FUTURE:   -valid_h <= age <= 0
      2) fallback PAST:    0 <= age <= valid_h
      3) fallback NEAREST: |age| <= nearest_h (default=valid_h), otherwise any

    Stickiness (hold) + thresholding as before.

    age := wrap_signed(toc - Toe):
      age < 0  -> Toe in FUTURE
      age > 0  -> Toe in PAST
    """
    valid_h = 14400.0 if system == 'E' else 7200.0
    if hold_sec is None:
        hold_sec = 3600.0 if system == 'G' else 7200.0
    if nearest_h is None:
        nearest_h = valid_h

    toe, trans, quality, orig_iloc = _prep_nav_arrays(system, nav_block)

    N = len(toc_vec)
    choice = np.full(N, -1, dtype=int)
    if len(toe) == 0:
        return choice

    age = _wrap_signed(toc_vec[:, None] - toe[None, :])  # [N, M]

    future_mask = (age <= 0.0) & (age >= -valid_h)
    past_mask   = (age >= 0.0) & (age <=  valid_h)
    near_mask   = (np.abs(age) <= nearest_h)

    def _tiebreak(idxs: np.ndarray):
        if len(idxs) == 1 or trans is None:
            return idxs[0]
        latest_t = trans.iloc[idxs].values
        tmax = latest_t.max()
        idxs2 = idxs[latest_t == tmax]
        if len(idxs2) == 1:
            return idxs2[0]
        q = quality[idxs2]
        return idxs2[np.argmin(q)]

    def pick_best(i: int):
        # 1) future: pick closest future (max age, i.e. closest to 0 from below)
        m = future_mask[i]
        if m.any():
            best_age = np.max(age[i, m])
            idxs = np.flatnonzero(m & (np.abs(age[i] - best_age) < 1e-9))
            return _tiebreak(idxs), 'future'

        # 2) past: pick freshest past (min age)
        m = past_mask[i]
        if m.any():
            best_age = np.min(age[i, m])
            idxs = np.flatnonzero(m & (np.abs(age[i] - best_age) < 1e-9))
            return _tiebreak(idxs), 'past'

        # 3) nearest: minimal |age| (prefer in-window near_mask; if empty, use all)
        m = near_mask[i]
        if not m.any():
            m = np.ones(age.shape[1], dtype=bool)
        best_abs = np.min(np.abs(age[i, m]))
        idxs = np.flatnonzero(m & (np.abs(np.abs(age[i]) - best_abs) < 1e-9))
        return _tiebreak(idxs), 'nearest'

    i0_idx, _ = pick_best(0)
    if i0_idx < 0:
        return choice
    cur_fidx = i0_idx
    cur_toe = toe[cur_fidx]
    cur_start_toc = toc_vec[0]
    choice[0] = orig_iloc[cur_fidx]

    for i in range(1, N):
        # if current went missing (shouldn't), re-pick
        if cur_fidx < 0:
            fidx, _ = pick_best(i)
            cur_fidx = fidx
            if cur_fidx >= 0:
                cur_toe = toe[cur_fidx]
                cur_start_toc = toc_vec[i]
                choice[i] = orig_iloc[cur_fidx]
            continue

        cur_age = _wrap_signed(toc_vec[i] - cur_toe)
        still_valid = (np.abs(cur_age) <= valid_h)  # “reasonable” validity
        if not still_valid:
            fidx, _ = pick_best(i)
            cur_fidx = fidx
            if cur_fidx >= 0:
                cur_toe = toe[cur_fidx]
                cur_start_toc = toc_vec[i]
                choice[i] = orig_iloc[cur_fidx]
            continue

        elapsed_hold = _wrap_signed(toc_vec[i] - cur_start_toc)
        if elapsed_hold < hold_sec:
            choice[i] = orig_iloc[cur_fidx]
            continue

        cand_fidx, cand_kind = pick_best(i)
        if cand_fidx < 0:
            choice[i] = orig_iloc[cur_fidx]
            continue

        cand_age = age[i, cand_fidx]

        # Decide if switch is worth it: compare absolute closeness to zero
        gain = (abs(cur_age) - abs(cand_age))

        # If we were forced into fallback (e.g., future->past after 22:00),
        # we still keep the same switching rule; it naturally switches once.
        if gain >= switch_thresh_sec:
            cur_fidx = cand_fidx
            cur_toe = toe[cur_fidx]
            cur_start_toc = toc_vec[i]
            choice[i] = orig_iloc[cur_fidx]
        else:
            choice[i] = orig_iloc[cur_fidx]

    return choice


def _pick_nav_rows_pref_past_else_future_with_hold(
    system: str,
    toc_vec: np.ndarray,
    nav_block: pd.DataFrame,
    hold_sec: float = None,
    switch_thresh_sec: float = 900.0,
    nearest_h: float | None = None,
) -> np.ndarray:
    """
    Galileo-friendly hybrid:
      1) prefer PAST
      2) fallback FUTURE
      3) fallback NEAREST
    """
    valid_h = 14400.0 if system == 'E' else 7200.0
    if hold_sec is None:
        hold_sec = 10800.0 if system == 'E' else 3600.0
    if nearest_h is None:
        nearest_h = valid_h

    toe, trans, quality, orig_iloc = _prep_nav_arrays(system, nav_block)

    N = len(toc_vec)
    choice = np.full(N, -1, dtype=int)
    if len(toe) == 0:
        return choice

    age = _wrap_signed(toc_vec[:, None] - toe[None, :])  # [N, M]

    past_mask   = (age >= 0.0) & (age <=  valid_h)
    future_mask = (age <= 0.0) & (age >= -valid_h)
    near_mask   = (np.abs(age) <= nearest_h)

    def _tiebreak(idxs: np.ndarray):
        if len(idxs) == 1 or trans is None:
            return idxs[0]
        latest_t = trans.iloc[idxs].values
        tmax = latest_t.max()
        idxs2 = idxs[latest_t == tmax]
        if len(idxs2) == 1:
            return idxs2[0]
        q = quality[idxs2]
        return idxs2[np.argmin(q)]

    def pick_best(i: int):
        # 1) past: freshest past (min age)
        m = past_mask[i]
        if m.any():
            best_age = np.min(age[i, m])
            idxs = np.flatnonzero(m & (np.abs(age[i] - best_age) < 1e-9))
            return _tiebreak(idxs), 'past'

        # 2) future: closest future (max age)
        m = future_mask[i]
        if m.any():
            best_age = np.max(age[i, m])
            idxs = np.flatnonzero(m & (np.abs(age[i] - best_age) < 1e-9))
            return _tiebreak(idxs), 'future'

        # 3) nearest: minimal |age|
        m = near_mask[i]
        if not m.any():
            m = np.ones(age.shape[1], dtype=bool)
        best_abs = np.min(np.abs(age[i, m]))
        idxs = np.flatnonzero(m & (np.abs(np.abs(age[i]) - best_abs) < 1e-9))
        return _tiebreak(idxs), 'nearest'

    f0, _ = pick_best(0)
    if f0 < 0:
        return choice
    cur_fidx = f0
    cur_toe = toe[cur_fidx]
    cur_start_toc = toc_vec[0]
    choice[0] = orig_iloc[cur_fidx]

    for i in range(1, N):
        if cur_fidx < 0:
            fidx, _ = pick_best(i)
            cur_fidx = fidx
            if cur_fidx >= 0:
                cur_toe = toe[cur_fidx]
                cur_start_toc = toc_vec[i]
                choice[i] = orig_iloc[cur_fidx]
            continue

        cur_age = _wrap_signed(toc_vec[i] - cur_toe)
        still_valid = (np.abs(cur_age) <= valid_h)
        if not still_valid:
            fidx, _ = pick_best(i)
            cur_fidx = fidx
            if cur_fidx >= 0:
                cur_toe = toe[cur_fidx]
                cur_start_toc = toc_vec[i]
                choice[i] = orig_iloc[cur_fidx]
            continue

        elapsed_hold = _wrap_signed(toc_vec[i] - cur_start_toc)
        if elapsed_hold < hold_sec:
            choice[i] = orig_iloc[cur_fidx]
            continue

        cand_fidx, _ = pick_best(i)
        if cand_fidx < 0:
            choice[i] = orig_iloc[cur_fidx]
            continue

        cand_age = age[i, cand_fidx]
        gain = (abs(cur_age) - abs(cand_age))
        if gain >= switch_thresh_sec:
            cur_fidx = cand_fidx
            cur_toe = toe[cur_fidx]
            cur_start_toc = toc_vec[i]
            choice[i] = orig_iloc[cur_fidx]
        else:
            choice[i] = orig_iloc[cur_fidx]

    return choice


def _select_by_toe_for_sat(system, obs_sv, nav_sv, picker='_gps_pref_future'):
    toc_vec = obs_sv['toc'].to_numpy(float)

    if picker == '_gps_pref_future':
        choice = _pick_nav_rows_pref_future_else_past_with_hold(
            system=system,
            toc_vec=toc_vec,
            nav_block=nav_sv,
            hold_sec=(30 * 60),
            switch_thresh_sec=900.0,
            nearest_h=None,
        )
    elif picker == '_gal_pref_past':
        choice = _pick_nav_rows_pref_past_else_future_with_hold(
            system=system,
            toc_vec=toc_vec,
            nav_block=nav_sv,
            hold_sec=(30 * 60),
            switch_thresh_sec=900.0,
            nearest_h=None,
        )
    else:
        raise ValueError(f"Unknown picker: {picker}")

    keep = choice >= 0
    if not keep.any():
        return pd.DataFrame()

    nav_sel = nav_sv.iloc[choice[keep]].copy()
    obs_sel = obs_sv.loc[keep].copy()

    nav_sel['sv'] = obs_sel['sv'].values
    nav_sel['time'] = obs_sel['time'].values

    merged = pd.concat(
        [obs_sel.reset_index(drop=True),
         nav_sel.reset_index(drop=True)], axis=1
    )
    return merged

class BrdcGenerator:
    mes_cols = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE', 'Crs', 'DeltaN', 'M0',
        'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic', 'Omega0', 'Cis',
        'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'CodesL2', 'GPSWeek',
        'L2Pflag', 'SVacc', 'health', 'TGD', 'IODC', 'TransTime'
    ]
    gal_mes_cols = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODnav', 'Crs',
                    'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic',
                    'Omega0', 'Cis', 'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'DataSrc',
                    'GALWeek', 'SISA', 'health', 'BGDe5a', 'BGDe5b', 'TransTime']

    def __init__(self, system, interval, mode, nav, tolerance ='1H'):
        self.sys = system
        self.interval = interval
        self.mode = mode
        self.nav = nav
        self.SEC_TO_FREQ = {
            1: "1s",
            2: "2s",
            5: "5s",
            10: "10s",
            15: "15s",
            30: "30s",
            60: "1T",  # 1 minuta
            120: "2T",
            300: "5T",
            600: "10T",
            900: "15T",
            1800: "30T",
            3600: "1H",  # 1 godzina
            7200: "2H",
            10800: "3H",
            21600: "6H",
            43200: "12H",
            86400: "1D"  # 1 doba
        }
        self.table = None
        self.tolerance = tolerance

    def _add_obs_toc(self):
        obs_toc = gnx_py.time.datetime2toc(t=self.table.index.get_level_values('time').tolist())
        self.table['toc'] = obs_toc

    def _add_nav_toc(self):
        nav_toc = gnx_py.time.datetime2toc(t=self.nav.index.get_level_values('time').tolist())
        self.nav['nav_toc'] = nav_toc

    def _clear_suffixes(self, df):
        df = df.reset_index().copy()
        df['prn'] = df.apply(lambda row: row['sv'][:3], axis=1)
        df = df.drop(columns='sv')
        df.set_index(['prn', 'time'], inplace=True)
        df.index.names = ['sv', 'time']
        return df

    def select_messages(self, df_obs: pd.DataFrame, df_orb: pd.DataFrame) -> pd.DataFrame:
        """
                New broadcast selection:
                - operates via satellite,
                - selects based on minimum |toc_obs - Toe| (SOW) with wrapping,
                - applies RTKLIB validity windows (GPS=2h, GAL=4h) validity windows and health filters,
                - resolves ties by TransTime and quality (SISA/SVacc).
                Returns a DataFrame joined 1:1 with epochs, but only for those for which
                we found a valid ephemeris; no "forward/nearest" after 'time'.
                """
        # Upewnij się, że mamy 'toc' w df_obs i 'nav_toc' w df_orb (dodajesz je wyżej w pipeline)
        if 'toc' not in df_obs.columns:
            raise ValueError("df_obs must contain 'toc' (seconds-of-week); call _add_obs_toc() first.")
        if 'nav_toc' not in df_orb.columns:
            raise ValueError("df_orb must contain 'nav_toc' (seconds-of-week); call _add_nav_toc() first.")
        if 'Toe' not in df_orb.columns:
            raise ValueError("df_orb must contain 'Toe' (seconds-of-week).")

        out_list = []
        for sv, obs_sv in df_obs.groupby('sv', sort=False):
            try:
                nav_sv = df_orb[df_orb['sv'] == sv]
                if nav_sv.empty:
                    continue
                if self.sys == 'G':
                    picker = '_gps_pref_future'  # prefer future, fallback past (np. po 22), fallback nearest
                else:
                    picker = '_gal_pref_past'  # prefer past, fallback future, fallback nearest
                sel = _select_by_toe_for_sat(self.sys, obs_sv, nav_sv, picker)

                if not sel.empty:
                    out_list.append(sel)
            except Exception as e:
                print(f"[select_messages] {sv}: {e}")
                continue
        if not out_list:
            return pd.DataFrame()
        merged = pd.concat(out_list, ignore_index=False)
        if '__rank__' in merged.columns:
            merged = merged.drop(columns='__rank__')
        return merged

    def prepare_table(self):
        freq = self.SEC_TO_FREQ[self.interval]
        if self.sys == 'G':
            sats_av = self.nav.index.get_level_values('sv').unique().tolist()
        elif self.sys == 'E':
            self.nav = self._clear_suffixes(df=self.nav)

            if self.mode == 'E1':
                war =(self.nav['I/NAV_E1-B']==True)& (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]

            elif self.mode == 'E5b':
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True) & (
                            self.nav['E5b_SHS'].isin([0, 2]) & (self.nav['E5b_DVS'] == 0))
                self.nav = self.nav[war]
            elif self.mode == 'E1E5b':
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True) & (
                        self.nav['E5b_SHS'].isin([0, 2]) & (self.nav['E5b_DVS'] == 0))
                self.nav = self.nav[war]
            elif self.mode in ['E5a','E1E5a']:
                war = (self.nav['F/NAV_E5a-I']==True)& (self.nav['dt_E5a_E1'] == True) & (
                        self.nav['E5a_SHS'].isin([0, 2]) & (self.nav['E5a_DVS'] == 0))
                self.nav = self.nav[war]
            sats_av = self.nav.index.get_level_values('sv').unique().tolist()
        t0 = sorted(self.nav.index.get_level_values('time').unique().tolist())[0].floor(freq='min')
        t1 = sorted(self.nav.index.get_level_values('time').unique().tolist())[-1].floor(freq='min')
        idx = pd.MultiIndex.from_product(
            [sats_av, pd.date_range(t0, t1, freq=freq)],
            names=["sv", "time"])
        self.table = pd.DataFrame(index=idx)

    def get_coordinates(self):
        output = []
        for sv, df_obs_grp in self.table.groupby('sv'):
            try:
                # OBS
                df_obs_block = _ensure_obs_block(df_obs_grp, sv_hint=sv)

                # NAV
                df_orb_sv = self.nav.loc[sv]
                df_orb_block = _ensure_nav_block(df_orb_sv, sv_hint=sv)

                merged = self.select_messages(df_obs_block, df_orb_block)
                if merged is None or merged.empty:
                    try:
                        toe = _ensure_nav_block(df_orb_sv, sv_hint=sv)['Toe'].to_numpy(float)
                        toc = _ensure_obs_block(df_obs_grp, sv_hint=sv)['toc'].to_numpy(float)
                        D = _wrap_dist_sow(toc, toe) if toe.size and toc.size else None
                        min_d = np.min(D, axis=1) if D is not None else np.array([])
                        within = np.sum(min_d <= (7200.0 if self.sys == 'G' else 14400.0)) if min_d.size else 0
                        print(f"[diag] {sv}: epochs={len(toc)}, NAV={len(toe)}, in_window={within}")
                    except Exception:
                        pass
                    continue

                if merged is None or merged.empty:
                    continue

                # Przygotowanie do interpolacji
                mes_toc = merged['nav_toc'].to_numpy(dtype=float)
                obs_toc = merged['toc'].to_numpy(dtype=float)

                if self.sys == 'E':
                    needed = self.gal_mes_cols
                    interp_fun = gal_numpy_broadcast_interpolation
                elif self.sys == 'G':
                    needed = self.mes_cols
                    interp_fun = numpy_broadcast_interpolation
                else:
                    raise ValueError(f"Unknown sys: {self.sys}")

                # Na wszelki wypadek: sprawdź, czy wszystkie kolumny istnieją
                missing = [c for c in needed if c not in merged.columns]
                if missing:
                    raise ValueError(f"NAV columns are missing in merged: {missing}")

                # Wymuś float i kształt (N, K)
                messages = merged[needed].to_numpy(dtype=float, copy=True)
                if messages.ndim != 2:
                    raise ValueError(f"messages has the wrong dimension: {messages.ndim}")

                rows = []
                bad_rows = 0
                for mt, ot, msg in zip(mes_toc, obs_toc, messages):
                    try:
                        res = interp_fun(mt, ot, msg)

                        if isinstance(res, (tuple, list)):
                            a = np.atleast_1d(res[0]).astype(float).ravel()
                            b = np.atleast_1d(res[1]).astype(float).ravel()
                            row = np.concatenate([a, b], axis=0)
                        else:
                            row = np.atleast_1d(res).astype(float).ravel()

                        rows.append(row)
                    except Exception:
                        bad_rows += 1
                        rows.append(None)
                        continue

                if any(r is None for r in rows):
                    keep_mask = np.array([r is not None for r in rows])
                    rows = [r for r in rows if r is not None]
                    merged = merged.loc[keep_mask].copy()
                    mes_toc = mes_toc[keep_mask]
                    obs_toc = obs_toc[keep_mask]

                if not rows:
                    continue

                unique_lengths = {len(r) for r in rows}
                if len(unique_lengths) != 1:
                    # jeżeli masz mieszankę 8/9 itp. – dopełnij do maksymalnej długości zerami
                    maxlen = max(unique_lengths)
                    rows = [np.pad(r, (0, maxlen - len(r)), mode='constant') for r in rows]

                crd = np.vstack(rows)

                cols8 = ['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot']
                cols9 = ['x', 'y', 'z', 'clk', 'clk_rel', 'vx', 'vy', 'vz', 'clk_dot']
                cols = cols8 if crd.shape[1] == 8 else (
                    cols9 if crd.shape[1] == 9 else [f'c{i}' for i in range(crd.shape[1])])

                merged = merged.loc[:,~merged.columns.duplicated()]
                merged = merged.set_index(['sv', 'time'])

                crd = pd.DataFrame(crd, index=merged.index, columns=cols)

                output.append(merged.join(crd))




            except Exception as e:
                try:
                    print(
                        f"For sv: {sv} | obs_cols={list(df_obs_grp.columns) if hasattr(df_obs_grp, 'columns') else 'n/a'} "
                        f"| nav_cols={list(self.nav.columns) if hasattr(self.nav, 'columns') else 'n/a'}")
                except Exception:
                    pass
                print('For sv: ', sv, e)
                continue

        if not output:
            raise ValueError("No ephemeris selected for all SVs (check validity windows, health and 'nav_toc').")
        return pd.concat(output)

    def generate(self):
        self.prepare_table()
        self._add_obs_toc()
        self._add_nav_toc()
        crd = self.get_coordinates()
        return crd


class BroadcastInterp:
    mes_cols = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE', 'Crs', 'DeltaN', 'M0',
        'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic', 'Omega0', 'Cis',
        'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'CodesL2', 'GPSWeek',
        'L2Pflag', 'SVacc', 'health', 'TGD', 'IODC', 'TransTime'
    ]
    gal_mes_cols = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODnav', 'Crs',
                    'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic',
                    'Omega0', 'Cis', 'Io', 'Crc', 'omega', 'OmegaDot', 'IDOT', 'DataSrc',
                    'GALWeek', 'SISA', 'health', 'BGDe5a', 'BGDe5b', 'TransTime']

    def __init__(self, obs, nav, mode, sys, emission_time=True,tolerance='2H'):
        self.obs = obs
        self.nav = nav
        self.mode = mode
        self.sys = sys
        self.emission_time = emission_time
        self.tolerance=tolerance

    def _clear_suffixes(self, df):
        df = df.reset_index().copy()
        df['prn'] = df.apply(lambda row: row['sv'][:3], axis=1)
        df = df.drop(columns='sv')
        df.set_index(['prn', 'time'], inplace=True)
        df.index.names = ['sv', 'time']
        return df

    def _add_obs_toc(self, emission=False):
        if emission:
            obs_toc = gnx_py.time.datetime2toc(t=self.obs['em_epoch'].tolist())
            self.obs['toc'] = obs_toc
        else:
            obs_toc = gnx_py.time.datetime2toc(t=self.obs.index.get_level_values('time').tolist())
            self.obs['toc'] = obs_toc
        return self.obs

    def _add_nav_toc(self):
        nav_toc = gnx_py.time.datetime2toc(t=self.nav.index.get_level_values('time').tolist())
        self.nav['nav_toc'] = nav_toc
        return self.nav

    def select_messages(self, df_obs: pd.DataFrame, df_orb: pd.DataFrame) -> pd.DataFrame:
        """
        Broadcast message selection:
        - operates via satellite,
        - selects based on minimum |toc_obs - Toe| (SOW) with wrapping,
        - applies validity windows (GPS=2h, GAL=4h) validity windows and health filters,
        - resolves ties by TransTime and quality (SISA/SVacc).
        Returns a DataFrame joined 1:1 with epochs, but only for those for which
        we found a valid ephemeris; no "forward/nearest" after 'time'.
        """
        # Upewnij się, że mamy 'toc' w df_obs i 'nav_toc' w df_orb (dodajesz je wyżej w pipeline)
        if 'toc' not in df_obs.columns:
            raise ValueError("df_obs must contain 'toc' (seconds-of-week); call _add_obs_toc() first.")
        if 'nav_toc' not in df_orb.columns:
            raise ValueError("df_orb must contain 'nav_toc' (seconds-of-week); call _add_nav_toc() first.")
        if 'Toe' not in df_orb.columns:
            raise ValueError("df_orb must contain 'Toe' (seconds-of-week).")

        # Zakładamy, że df_obs i df_orb mają kolumnę 'sv'
        out_list = []
        for sv, obs_sv in df_obs.groupby('sv', sort=False):
            try:
                nav_sv = df_orb[df_orb['sv'] == sv]
                if nav_sv.empty:
                    continue
                if self.sys == 'G':
                    picker = '_gps_pref_future'  # prefer future, fallback past (np. po 22), fallback nearest
                else:
                    picker = '_gal_pref_past'  # prefer past, fallback future, fallback nearest
                sel = _select_by_toe_for_sat(self.sys, obs_sv, nav_sv, picker)

                if not sel.empty:
                    out_list.append(sel)
            except Exception as e:
                print(f"[select_messages] {sv}: {e}")
                continue
        if not out_list:
            return pd.DataFrame()
        merged = pd.concat(out_list, ignore_index=False)
        # usuń pomocnicze kolumny jeśli jakieś dodałeś
        if '__rank__' in merged.columns:
            merged = merged.drop(columns='__rank__')
        return merged

    def get_coordinates(self):
        output = []
        for sv, df_obs_grp in self.obs.groupby('sv'):
            try:
                # OBS
                df_obs_block = _ensure_obs_block(df_obs_grp, sv_hint=sv)

                # NAV
                df_orb_sv = self.nav.loc[sv]
                df_orb_block = _ensure_nav_block(df_orb_sv, sv_hint=sv)

                merged = self.select_messages(df_obs_block, df_orb_block)
                if merged is None or merged.empty:
                    try:
                        toe = _ensure_nav_block(df_orb_sv, sv_hint=sv)['Toe'].to_numpy(float)
                        toc = _ensure_obs_block(df_obs_grp, sv_hint=sv)['toc'].to_numpy(float)
                        D = _wrap_dist_sow(toc, toe) if toe.size and toc.size else None
                        min_d = np.min(D, axis=1) if D is not None else np.array([])
                        within = np.sum(min_d <= (7200.0 if self.sys == 'G' else 14400.0)) if min_d.size else 0
                        print(f"[diag] {sv}: epochs={len(toc)}, NAV={len(toe)}, in_window={within}")
                    except Exception:
                        pass
                    continue

                if merged is None or merged.empty:
                    continue

                mes_toc = merged['nav_toc'].to_numpy(dtype=float)
                obs_toc = merged['toc'].to_numpy(dtype=float)

                if self.sys == 'E':
                    needed = self.gal_mes_cols
                    interp_fun = gal_numpy_broadcast_interpolation
                elif self.sys == 'G':
                    needed = self.mes_cols
                    interp_fun = numpy_broadcast_interpolation
                else:
                    raise ValueError(f"Unknown sys: {self.sys}")

                missing = [c for c in needed if c not in merged.columns]
                if missing:
                    raise ValueError(f"Missing data: {missing}\n Check your broadcast orbit")

                # Wymuś float i kształt (N, K)
                messages = merged[needed].to_numpy(dtype=float, copy=True)
                if messages.ndim != 2:
                    raise ValueError(f"Dimension error : {messages.ndim}")



                rows = []
                bad_rows = 0
                for mt, ot, msg in zip(mes_toc, obs_toc, messages):
                    try:
                        res = interp_fun(mt, ot, msg)

                        if isinstance(res, (tuple, list)):
                            a = np.atleast_1d(res[0]).astype(float).ravel()
                            b = np.atleast_1d(res[1]).astype(float).ravel()
                            row = np.concatenate([a, b], axis=0)
                        else:
                            row = np.atleast_1d(res).astype(float).ravel()

                        rows.append(row)
                    except Exception:
                        bad_rows += 1
                        rows.append(None)
                        continue

                if any(r is None for r in rows):
                    keep_mask = np.array([r is not None for r in rows])
                    rows = [r for r in rows if r is not None]
                    merged = merged.loc[keep_mask].copy()
                    mes_toc = mes_toc[keep_mask]
                    obs_toc = obs_toc[keep_mask]

                if not rows:
                    continue

                unique_lengths = {len(r) for r in rows}
                if len(unique_lengths) != 1:
                    maxlen = max(unique_lengths)
                    rows = [np.pad(r, (0, maxlen - len(r)), mode='constant') for r in rows]

                crd = np.vstack(rows)

                cols8 = ['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot']
                cols9 = ['x', 'y', 'z', 'clk', 'clk_rel', 'vx', 'vy', 'vz', 'clk_dot']
                cols = cols8 if crd.shape[1] == 8 else (
                    cols9 if crd.shape[1] == 9 else [f'c{i}' for i in range(crd.shape[1])])

                merged = merged.loc[:, ~merged.columns.duplicated()]
                merged = merged.set_index(['sv', 'time'])

                crd = pd.DataFrame(crd, index=merged.index, columns=cols)

                output.append(merged.join(crd))




            except Exception as e:
                try:
                    print(
                        f"For sv: {sv} | obs_cols={list(df_obs_grp.columns) if hasattr(df_obs_grp, 'columns') else 'n/a'} "
                        f"| nav_cols={list(self.nav.columns) if hasattr(self.nav, 'columns') else 'n/a'}")
                except Exception:
                    pass
                print('For sv: ', sv, e)
                continue

        if not output:
            raise ValueError("No ephemeris selected for all SVs (check validity windows, health and 'nav_toc').")
        return pd.concat(output)

    def grab_emission_time(self):
        clight = 299792458
        if 'P3' in self.obs.columns:
            ccol='P3'
        else:
            ccol = [c for c in self.obs.columns if c.startswith('C')][0]
        obs_time = self.obs.index.get_level_values('time')
        self.obs['obs_time'] = obs_time
        t_em = ((self.obs[ccol] / clight) + self.obs['clk'])
        self.obs['em_epoch'] = self.obs['obs_time'] - pd.to_timedelta(t_em, unit='s')
        return self.obs

    def _clear_reception_data(self):
        if self.sys == 'G':
            self.obs = self.obs.drop(columns=['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot'] + self.mes_cols)
        elif self.sys == 'E':
            self.obs = self.obs.drop(columns=['x', 'y', 'z', 'clk', 'vx', 'vy', 'vz', 'clk_dot'] + self.gal_mes_cols)
        return self.obs

    def interpolate(self):

        init_obs = self.obs.copy()
        if self.sys== 'E':
            self.nav = self._clear_suffixes(df=self.nav)
            self.nav = self.nav.dropna(how='any', axis=1).dropna(how='all', axis=0)
            if self.mode == 'E1':
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]

            elif self.mode == 'E5b':
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True) & (
                        self.nav['E5b_SHS'].isin([0, 2]) & (self.nav['E5b_DVS'] == 0))
                self.nav = self.nav[war]
            elif self.mode == 'E1E5b':
                war = (self.nav['I/NAV_E1-B'] == True) & (self.nav['dt_E5b_E1'] == True)
                self.nav = self.nav[war]
            elif self.mode in ['E5a', 'E1E5a']:
                war = (self.nav['F/NAV_E5a-I'] == True) & (self.nav['dt_E5a_E1'] == True)
                self.nav = self.nav[war]
            self.obs = self._add_obs_toc(emission=False)
            self.nav = self._add_nav_toc()
            result = self.get_coordinates()
            self.obs = pd.merge(left=init_obs,right=result,on=['sv','time'],how='right')
            if self.emission_time:
                self.obs = self.grab_emission_time()
                self._clear_reception_data()
                self.obs = self._add_obs_toc(emission=True)
                result = self.get_coordinates()
        if self.sys == 'G':

            self.nav = self.nav.dropna(how='any', axis=1).dropna(how='all', axis=0)
            self.nav = self._clear_suffixes(df=self.nav)
            self.obs = self._add_obs_toc(emission=False)
            self.nav = self._add_nav_toc()
            result = self.get_coordinates()
            self.obs = pd.merge(left=init_obs,right=result,on=['sv','time'],how='right')
            if self.emission_time:
                self.obs = self.grab_emission_time()
                self._clear_reception_data()
                self.obs = self._add_obs_toc(emission=True)
                result = self.get_coordinates()
        output = pd.merge(left=init_obs,right=result,on=['sv','time'],how='right')
        return output


def correct_sat_coordinates(xyz_s, xyz_a):
    """
    Correction of satellite coordinates with respect to earth rotation during signal travel
    :param xyz_s: satellite crd at the epoch
    :param xyz_a: station crd
    :return: xyz corrected by earth rotation
    """
    omega = 7.2921159e-5
    c = 299792458

    dT = (np.sqrt((xyz_s[0] - xyz_a[0]) ** 2 + (xyz_s[1] - xyz_a[1]) ** 2 + (xyz_s[2] - xyz_a[2]) ** 2)) / c
    omw = dT * omega

    cos0 = np.cos(omw)
    sin0 = np.sin(omw)
    rot = np.array([[cos0, sin0, 0],
                    [-sin0, cos0, 0],
                    [0, 0, 1]])
    xyz_s2 = rot.dot(xyz_s)
    return xyz_s2




# ---------------- Utils ----------------
def _to_seconds(dt_any) -> np.ndarray:
    dt64 = np.asarray(dt_any, dtype='datetime64[ns]')
    # int64 nanosekundy -> sekundy z ułamkami
    return dt64.astype('int64') / 1e9
def _median_dt_seconds(t_s: np.ndarray) -> float:
    return np.inf if t_s.size < 2 else float(np.median(np.diff(t_s)))

# ---------------- Barycentric Lagrange ----------------
@njit(cache=False, fastmath=False)
def _bary_weights(xw):
    n = xw.size
    w = np.empty(n, dtype=np.float64)
    for j in range(n):
        s = 1.0
        xj = xw[j]
        for k in range(n):
            if k != j:
                diff = xj - xw[k]
                if diff == 0.0:
                    for q in range(n):
                        w[q] = np.nan
                    return w
                s *= diff
        w[j] = 1.0 / s
    return w

@njit(cache=True, fastmath=True)
def _bary_eval_vec(xw, w, Yw, xq):
    m, D = Yw.shape
    for j in range(m):
        if w[j] != w[j]:  # NaN
            out_nan = np.empty(D, dtype=np.float64)
            for d in range(D):
                out_nan[d] = np.nan
            return out_nan
    num = np.zeros(D, dtype=np.float64)
    den = 0.0
    for j in range(m):
        dj = xq - xw[j]
        if dj == 0.0:
            out = np.empty(D, dtype=np.float64)
            for d in range(D):
                out[d] = Yw[j, d]
            return out
        tmp = w[j] / dj
        for d in range(D):
            num[d] += tmp * Yw[j, d]
        den += tmp
    inv = 1.0 / den
    out = np.empty(D, dtype=np.float64)
    for d in range(D):
        out[d] = num[d] * inv
    return out

@njit(parallel=True, cache=True, fastmath=True)
def lagrange_local_vec(x, Y, xq, m=12, max_gap=np.inf):
    N = x.size
    D = Y.shape[1]
    Q = xq.size
    out = np.empty((Q, D), dtype=np.float64)
    r = m if N >= m else N
    for i in range(Q):
        if r < 2:
            for d in range(D):
                out[i, d] = np.nan
            continue
        xi = xq[i]
        k = np.searchsorted(x, xi)
        left = k - r // 2
        if left < 0: left = 0
        right = left + r
        if right > N:
            right = N
            left = right - r
        nearest = 1e300
        if k < N and left <= k < right:
            tmp = abs(x[k] - xi)
            if tmp < nearest: nearest = tmp
        if k-1 >= 0 and left <= (k-1) < right:
            tmp = abs(x[k-1] - xi)
            if tmp < nearest: nearest = tmp
        if nearest > max_gap:
            for d in range(D):
                out[i, d] = np.nan
            continue
        xw = x[left:right]
        Yw = Y[left:right, :]
        w  = _bary_weights(xw)
        out[i, :] = _bary_eval_vec(xw, w, Yw, xi)
    return out
@njit(cache=True, fastmath=True)
def lin_interp1d_vec(x, y, xq, max_gap=np.inf):
    """
        Linear 1D interpolation (x,y)->y(xq) for sorted x.
        Returns NaN if the query is out of range or the nearest node is further than max_gap.
        """
    N = x.size
    Q = xq.size
    out = np.empty(Q, dtype=np.float64)
    for i in range(Q):
        xi = xq[i]
        k = np.searchsorted(x, xi)

        # trafienie dokładnie w węzeł
        if k < N and x[k] == xi:
            out[i] = y[k]
            continue

        i0 = k - 1
        i1 = k
        if i0 < 0 or i1 >= N:
            out[i] = np.nan
            continue

        # test luki (do najbliższego węzła)
        d0 = abs(xi - x[i0])
        d1 = abs(x[i1] - xi)
        if (d0 if d0 < d1 else d1) > max_gap:
            out[i] = np.nan
            continue

        denom = x[i1] - x[i0]
        if denom == 0.0:
            out[i] = np.nan
            continue

        t = (xi - x[i0]) / denom
        out[i] = (1.0 - t) * y[i0] + t * y[i1]
    return out

@njit(cache=True, fastmath=True)
def lin_interp1d_vec(x, y, xq, max_gap=np.inf):
    """
        Linear 1D interpolation (x,y)->y(xq) for sorted x.
        Returns NaN if the query is out of range or the nearest node is farther than max_gap.
        """
    N = x.size
    Q = xq.size
    out = np.empty(Q, dtype=np.float64)
    for i in range(Q):
        xi = xq[i]
        k = np.searchsorted(x, xi)

        # trafienie dokładnie w węzeł
        if k < N and x[k] == xi:
            out[i] = y[k]
            continue

        i0 = k - 1
        i1 = k
        if i0 < 0 or i1 >= N:
            out[i] = np.nan
            continue

        # test luki (do najbliższego węzła)
        d0 = abs(xi - x[i0])
        d1 = abs(x[i1] - xi)
        if (d0 if d0 < d1 else d1) > max_gap:
            out[i] = np.nan
            continue

        denom = x[i1] - x[i0]
        if denom == 0.0:
            out[i] = np.nan
            continue

        t = (xi - x[i0]) / denom
        out[i] = (1.0 - t) * y[i0] + t * y[i1]
    return out

# ---------------- Front: MultiIndex (time, sv) ----------------
def lagrange_interp(
    obs: pd.DataFrame,
    sp3: pd.DataFrame,
    m: int = 12,
    gap_factor: float = 3.0,
) -> pd.DataFrame:
    """
        For epochs (time, sv) from 'obs', interpolates from 'sp3' the position (x,y,z) [m],
        optionally the clock 'clk' [s], calculates the velocities (vx,vy,vz) [m/s] using the central difference method
        ±0.5 µs and relativistic clock correction:
            clk_rel = -2 * dot(r, v) / c^2   [s]
        Returns a DataFrame with MultiIndex (time, sv) and columns:
            x, y, z, clk, vx, vy, vz, clk_rel
        """
    obs = obs.copy(); sp3 = sp3.copy()
    # obs.index = obs.index.set_names(['time', 'sv'])
    obs.index = (
        obs.index
        .set_names(['time', 'sv'])
        if isinstance(obs.index.levels[0][0], pd.Timestamp)
        else obs.index.reorder_levels([1, 0]).set_names(['time', 'sv'])
    )

    sp3.index = sp3.index.set_names(['time', 'sv'])

    obs_sorted = obs.sort_index()
    times_obs = obs_sorted.index.get_level_values('time').to_numpy()
    sv_obs    = obs_sorted.index.get_level_values('sv').to_numpy()
    t_obs_s   = np.ascontiguousarray(_to_seconds(times_obs), dtype=np.float64)

    # mapowanie SV -> listy wierszy
    sv_to_rows: dict[str, list[int]] = {}
    for i, sv in enumerate(sv_obs.tolist()):
        sv_to_rows.setdefault(sv, []).append(i)

    # bufor wyjściowy
    n = len(obs_sorted)
    out_xyz  = np.full((n, 3), np.nan, dtype=np.float64)
    out_clk  = np.full(n, np.nan, dtype=np.float64)
    out_vel  = np.full((n, 3), np.nan, dtype=np.float64)
    out_rel  = np.full(n, np.nan, dtype=np.float64)

    for sv, row_idx in sv_to_rows.items():
        try:
            g = sp3.xs(sv, level='sv', drop_level=False).reset_index()
        except KeyError:
            continue

        # tylko potrzebne kolumny; odfiltruj NaNy
        cols = ['time','x','y','z'] + (['clk'] if 'clk' in g.columns else [])
        g = g[cols].dropna(subset=['time','x','y','z'])
        # g = g.groupby('time', as_index=False).mean(numeric_only=True)
        g = g.loc[~g['time'].duplicated(keep='first')]

        # ndarray
        t_sv = _to_seconds(g['time'].to_numpy())
        XYZ  = g[['x','y','z']].to_numpy(dtype=np.float64, copy=False)
        has_clk = 'clk' in g.columns
        if has_clk:
            CLK = g[['clk']].to_numpy(dtype=np.float64, copy=False)

        if t_sv.size == 0:
            continue

        # sort + C
        order = np.argsort(t_sv)
        t_sv  = np.ascontiguousarray(t_sv[order], dtype=np.float64)
        XYZ   = np.ascontiguousarray(XYZ[order, :], dtype=np.float64)
        if has_clk:
            CLK = np.ascontiguousarray(CLK[order, :], dtype=np.float64)

        idx = np.asarray(row_idx, dtype=np.int64)
        tq  = np.ascontiguousarray(t_obs_s[idx], dtype=np.float64)

        dt_med  = _median_dt_seconds(t_sv)
        max_gap = gap_factor * dt_med if np.isfinite(dt_med) else np.inf

        pos = lagrange_local_vec(t_sv, XYZ, tq, m=int(m), max_gap=max_gap)
        out_xyz[idx, :] = pos

        if has_clk:
            # CLK jako wektor 1D float64, C-contiguous
            clk_vec = np.ascontiguousarray(CLK.ravel(), dtype=np.float64)
            clk = lin_interp1d_vec(t_sv, clk_vec, tq, max_gap=max_gap)
            out_clk[idx] = clk

        if np.isfinite(DT_VEL) and DT_VEL > 0.0:
            tq_minus = np.ascontiguousarray(tq - DT_VEL, dtype=np.float64)
            tq_plus  = np.ascontiguousarray(tq + DT_VEL, dtype=np.float64)
            r_m = lagrange_local_vec(t_sv, XYZ, tq_minus, m=int(m), max_gap=max_gap)
            r_p = lagrange_local_vec(t_sv, XYZ, tq_plus , m=int(m), max_gap=max_gap)
            vel = (r_p - r_m) / (2.0 * DT_VEL)   # [m/s]
            out_vel[idx, :] = vel

            # relativistic clock correction: -2 * (r·v) / c^2 [s]
            # (r, v in ECEF – standard PPP practice for "eccentricity correction")
            dot_rv = np.einsum('ij,ij->i', pos, vel)   # [m^2/s]
            clk_rel = -2.0 * dot_rv / (C_LIGHT**2)    # [s]
            out_rel[idx] = clk_rel

    clk_corrected = out_clk + out_rel
    out = pd.DataFrame(
        np.column_stack([out_xyz,
                         out_clk.reshape(-1,1),
                         clk_corrected.reshape(-1,1),
                         out_vel,
                         out_rel.reshape(-1,1)]),
        index=obs_sorted.index,
        columns=['x','y','z','clk_raw','clk','vx','vy','vz','clk_rel']
    )
    return out

def lagrange_reception_interp(obs:pd.DataFrame,  sp3_df:pd.DataFrame,
                             degree:Optional[int]=12, gap_factor:Optional[Union[float,int]]=3.0):
    """

    :param obs: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with observation data
    :param sp3_df: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with SP3 data
    :param degree: Optional[int], degree of interpolation, default: 12
    :param gap_factor: Optional[Union[float,int]], gap factor for interpolation, default: 3.0
    :return: tuple(pd.DataFrame, index: pd.MultiIndex (time, sv),pd.DataFrame, index: pd.MultiIndex (time,sv)),
    tupe of dataframes: one with observation data, satellite positions and clocks,
    and second dataframe with interpolated positions and clocks
    """
    interp = lagrange_interp(obs=obs,
                             sp3=sp3_df,
                             m=degree,
                             gap_factor=gap_factor)
    return obs.join(interp), interp

def lagrange_emission_interp(obs:pd.DataFrame, positions:pd.DataFrame, sp3_df:pd.DataFrame,
                             degree:Optional[int]=12, gap_factor:Optional[Union[float,int]]=3.0):
    """

    :param obs: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with observation data
    :param positions: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with reception time positions
    :param sp3_df: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with SP3 data
    :param degree: Optional[int], degree of interpolation, default: 12
    :param gap_factor: Optional[Union[float,int]], gap factor for interpolation, default: 3.0
    :return: pd.DataFrame, index: pd.MultiIndex (time, sv), dataframe with observation data, satellite positions and clocks
    """
    obs_crd = obs.join(positions)
    obs_crd.reset_index(inplace=True)
    if 'P3' in obs_crd.columns:
        obs_crd['t_em'] = (obs_crd['P3'] / C_LIGHT) + (obs_crd['clk'])
    else:
        col = [c for c in obs_crd.columns if c.startswith(('C1','C2','C5','C7'))][0]
        obs_crd['t_em'] = (obs_crd[col] / C_LIGHT) + (obs_crd['clk'])
    obs_crd['time_em'] = obs_crd['time'] - pd.to_timedelta(obs_crd['t_em'].values, unit='s')
    obs_crd['time_rec'] = obs_crd['time'].copy()
    obs_crd['time'] = obs_crd['time_em'].copy()
    obs_crd.set_index(['time', 'sv'], inplace=True)
    interp_em = lagrange_interp(
        obs=obs_crd,
        sp3=sp3_df,
        m=degree,
        gap_factor=gap_factor
    )
    obs_final = obs_crd.merge(interp_em, how="left", on=["time", "sv"], suffixes=("_old", ""))
    obs_final = obs_final.drop([c for c in obs_final.columns if c.endswith("_old")], axis=1)
    obs_final.reset_index(inplace=True)
    obs_final['time'] = obs_final['time_rec'].copy()
    obs_final = obs_final.drop(['time_rec', 'time_em'], axis=1)
    obs_final.set_index(['time', 'sv'], inplace=True)
    return obs_final
