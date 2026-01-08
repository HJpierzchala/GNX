from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from .tides import get_sun_ecef, get_tides
from .conversion import ecef2geodetic
from .utils import calculate_distance
from .time import datetime2toc, datetime2doy
from .ionosphere.models import klobuchar,ntcm_vtec
from .troposphere import saastamoinen,niell,tropospheric_delay
from .ionosphere.ionex import GIMReader, TECInterpolator
from .corrections import rel_path_corr, process_windup_correction_vectorized

# —————————————————————————————————————————————
# 3) Metody „wkładalne” do Twojej klasy
# —————————————————————————————————————————————

class CSDetector:
    def __init__(self,obs, phase_shift_dict, dcb, sys='G',mode='L1L2',interval=30):
        self.obs = obs
        self.phase_shift_dict = phase_shift_dict
        self.sys = sys

        self.c = 299792458
        self.sampling_rate = interval
        self.M = 5  # interpolation and CS repair window size
        self.dcb = dcb
        self.mode=mode
        self.F = {'f1': 1575.42e06,
                  'f2': 1227.60e06,
                  'f5': 1176.45e06,
                  'fe1a': 1575.420e06,
                  'fe5a': 1176.450e06,
                  'fe5b': 1207.140e06}
        self.l = {'L1': 299792458 / 1575.42e06,
                  'L5': 299792458 / 1176.45e06,
                  'L2': 299792458 / 1227.60e06,
                  'E5a': 299792458 / 1176.45e06,
                  'E5b': 299792458 / 1207.140e06}
        if self.obs.index.get_level_values(0).name == 'time':
            self.obs = self.obs.swaplevel(0, 1)

    def make_geometry_free(self):
        try:
            if self.mode in ['L1', 'L2','L1L2']:
                p1 = [col for col in self.obs.columns if col.startswith('C1')][0]
                p2 = [col for col in self.obs.columns if col.startswith('C2')][0]
                l1 = [col for col in self.obs.columns if col.startswith('L1')][0]
                l2 = [col for col in self.obs.columns if col.startswith('L2')][0]

            if self.mode =='L1L5' or self.mode =='L5':
                p1 = [col for col in self.obs.columns if col.startswith('C1')][0]
                p2 = [col for col in self.obs.columns if col.startswith('C5')][0]
                l1 = [col for col in self.obs.columns if col.startswith('L1')][0]
                l2 = [col for col in self.obs.columns if col.startswith('L5')][0]
            if self.mode =='L2L5':
                p1 = [col for col in self.obs.columns if col.startswith('C2')][0]
                p2 = [col for col in self.obs.columns if col.startswith('C5')][0]
                l1 = [col for col in self.obs.columns if col.startswith('L2')][0]
                l2 = [col for col in self.obs.columns if col.startswith('L5')][0]

            elif self.mode in ['E1E5a','E1','E5a']:
                p1 = [col for col in self.obs.columns if col.startswith('C1')][0]
                p2 = [col for col in self.obs.columns if col.startswith('C5')][0]
                l1 = [col for col in self.obs.columns if col.startswith('L1')][0]
                l2 = [col for col in self.obs.columns if col.startswith('L5')][0]
            elif self.mode == 'E1E5b' or self.mode =='E5b':
                p1 = [col for col in self.obs.columns if col.startswith('C1')][0]
                p2 = [col for col in self.obs.columns if col.startswith('C7')][0]
                l1 = [col for col in self.obs.columns if col.startswith('L1')][0]
                l2 = [col for col in self.obs.columns if col.startswith('L7')][0]
            elif self.mode == 'E5aE5b':
                p1 = [col for col in self.obs.columns if col.startswith('C5')][0]
                p2 = [col for col in self.obs.columns if col.startswith('C7')][0]
                l1 = [col for col in self.obs.columns if col.startswith('L5')][0]
                l2 = [col for col in self.obs.columns if col.startswith('L7')][0]

        except IndexError:
            raise IndexError('Lack of dual frequency data!')




        p4 = self.obs[p2] - self.obs[p1]
        l4 = self.obs[l1] - self.obs[l2]
        self.obs['P4'] = p4
        self.obs['L4'] = l4

    def split_obs_dt(self):
        """
        Dzieli obserwacje dla każdego PRN na fragmenty ciągłe w czasie, na podstawie przerw większych niż sampling_rate.
        """
        out = {}
        for prn, group in self.obs.groupby('sv'):
            # Przepisujemy 'time' jako kolumnę (upewnij się, że jest typu Timestamp!)
            time_idx = group.index.get_level_values('time')
            times = pd.to_datetime(time_idx)
            # Różnice czasowe między epokami (w sekundach)
            dt = times.to_series().diff().dt.total_seconds().fillna(0).to_numpy()
            # Gdzie przerwa > sampling_rate, tam zaczyna się nowy fragment
            breaks = (dt > self.sampling_rate)
            # Każdemu fragmentowi przypisujemy numer
            frag_id = breaks.cumsum()
            # Dodajemy do DataFrame
            df = group.reset_index()
            df['frag'] = frag_id
            # Splitujemy po frag_id
            for n, frag in df.groupby('frag'):
                key = f"{prn}_{n}"
                # Ustawiamy indeks 'time', usuwamy kolumny indeksowe
                frag = frag.set_index('time').drop(columns=['sv', 'frag'])
                out[key] = frag
        # Łączymy – klucz główny to sv_x, drugi poziom to czas
        return pd.concat(out, keys=out.keys(), names=['sv', 'time'])

    def drop_short_arcs(self, df, t):
        """

        :param df: dataframe with index sv orz multiindex with sv level
        :param t: minimum arc length (minutes)
        :return: filtered df
        """
        # min_arc_len = int(self.sampling_rate/60 * t)
        min_arc_len = int(t / (self.sampling_rate/60))
        filtered_df = df.groupby('arc').filter(lambda x: len(x)>min_arc_len)
        return filtered_df


    def split_obs_cs(self):
        """
        Dzieli obserwacje na łuki (arcs) na podstawie detekcji outlierów z filtra Lagrange'a.
        Przyspieszona wersja: batchowe operacje, bez pętli po wierszach.
        """
        # Tworzymy kopię DataFrame z nową kolumną 'arc' = None
        obs = self.obs.copy()
        obs['arc'] = None

        for sv, group in obs.groupby('sv'):
            ev = group['ev'].to_numpy()
            l4 = group['L4'].to_numpy()
            # Szybki Lagrange filter
            difs, inds = self.lagrange_filter_vectorized(l4, ev, mode=self.mode, l=self.l)
            if len(inds) == 0:
                # Cały łuk = jedna etykieta
                obs.loc[group.index, 'arc'] = f"{sv}_1"
                continue

            # Wyznacz czasy podziałów
            split_idx = np.sort(inds)
            # Dodaj 0 na początek i N na koniec, by obsłużyć całość
            split_idx = np.concatenate([[0], split_idx, [len(group)]])
            # Nadaj etykiety
            for arc_nr, (i_start, i_end) in enumerate(zip(split_idx[:-1], split_idx[1:]), 1):
                mask = np.zeros(len(group), dtype=bool)
                mask[i_start:i_end] = True
                idxs = group.index[mask]
                obs.loc[idxs, 'arc'] = f"{sv}_{arc_nr}"

        # Jeśli jakieś łuki nie zostały przypisane, przypisz po prostu etykietę PRN
        obs['arc'] = obs['arc'].fillna(obs.index.get_level_values('sv').to_series(index=obs.index))

        # Drop short arcs:
        return obs


    def rolling_lagrange_pred(self, l4, predict_at):
        """
        Szybki rolling Lagrange 3-punktowy: dla każdego okna 3 wyznacza wartość wielomianu w punkcie predict_at.
        predict_at = 3 -> przewiduje wartość dla i+3 na podstawie i,i+1,i+2
        Zwraca: array shape (len(l4)-2,)
        """
        k = predict_at
        n = len(l4)
        if n < k:
            return np.array([])
        # Wagi Lagrange'a dla punktów 0,1,2 w punkcie 'predict_at'
        # Wzory analityczne:
        x = np.arange(k)
        L = np.ones(k)
        for j in range(k):
            for m in range(k):
                if m != j:
                    L[j] *= (predict_at - x[m]) / (x[j] - x[m])
        # Teraz rolling dot product:
        # Rolling okna 3 elementowe:
        strided = np.lib.stride_tricks.sliding_window_view(l4, window_shape=k)
        # shape = (n-k+1, k)
        preds = np.dot(strided, L)
        return preds

    def lagrange_filter_vectorized(self, l4, ev, mode, l):
        """
        Wektoryzowana wersja lagrange_filter, bez pętli po punktach!
        """
        max_elevation_index = np.argmax(ev)
        indices = []

        if mode in ['L1L2', 'L1']:
            t = np.abs(l['L2'] - l['L1'])
        elif mode in ['L1L5', 'L5']:
            t = np.abs(l['L5'] - l['L1'])
        elif mode in ['E1E5a', 'E1']:
            t = np.abs(l['E5a'] - l['L1'])
        elif mode == 'E1E5b' or mode == 'E5b':
            t = np.abs(l['E5b'] - l['L1'])
        else:
            t = 0.5

        # W prawo od max_elevation_index:
        right_start = max_elevation_index
        if right_start < len(l4) - 3:
            l4_right = l4[right_start:]
            preds_right = self.rolling_lagrange_pred(l4_right, predict_at=3)[:-1]
            n_pred = len(preds_right)
            actuals_right = l4[right_start + 3:right_start + 3 + n_pred]
            assert len(preds_right) == len(
                actuals_right), f"preds_right: {len(preds_right)}, actuals_right: {len(actuals_right)}"
            diffs = preds_right - actuals_right

            outlier_idx = np.where(np.abs(diffs) > t)[0]
            indices.extend(right_start + 3 + outlier_idx)

        # W lewo od max_elevation_index:
        left_end = max_elevation_index
        if left_end > 2:
            l4_left = l4[:left_end + 1][::-1]
            preds_left = self.rolling_lagrange_pred(l4_left, predict_at=3)[:-1]
            n_pred = len(preds_left)
            actuals_left = l4_left[3:3 + n_pred]
            assert len(preds_left) == len(
                actuals_left), f"preds_left: {len(preds_left)}, actuals_left: {len(actuals_left)}"
            outlier_idx = np.where(np.abs(preds_left - actuals_left) > t)[0]
            # Indeksy przeliczyć do oryginalnej kolejności!
            indices.extend(left_end - 3 - outlier_idx)

        indices = np.unique(indices)
        differences = []  # Możesz dorzucić jeśli chcesz

        return np.array(differences), indices



    def run(self, min_arc_lenth=30, detector: Optional[Literal["MW","GF"]]="GF"):
        self.make_geometry_free()
        self.obs = self.split_obs_dt()
        self.obs = self.split_obs_cs()
        # SV_dt_cs
        self.obs = self.drop_short_arcs(df=self.obs,t=min_arc_lenth)
        return self.obs

@dataclass
class DefaultConfig:
    gps_freq: str = "L1L2"  # Domyślne wartości
    gal_freq: str = "E1E5a"
    station_name: Optional[str] = 'Station'
    day_of_year = 1
    gim_path=None
    ionosphere_model=None

class DDPreprocessing:
    def __init__(
        self,
        df,
        flh,
        xyz,

        sat_pco,
        rec_pco,
        antenna_h,
        system,
        config: Optional[Union["PPPConfig","DefaultConfig","SPPConfig"]] = DefaultConfig,
        gpsa=None,
        gpsb=None,
        gala=None,
        phase_shift_dict=None
    ):
        """
        Preprocessing class for double difference solution.

        Parameters:
        - df: DataFrame with observations and coordinates
        - flh: (lat, lon, height)
        - xyz: ECEF coordinates
        - config: PPPConfig instance
        - gpsa/gpsb/gala: model parameters if needed, passed optionally
        """
        self.df = df
        self.flh = flh
        self.xyz = xyz
        self.configuration = config if config is not None else DefaultConfig()

        self.doy = self.configuration.day_of_year or 1
        self.SYS = system
        self.mode = self.configuration.gps_freq if self.SYS=='G' else self.configuration.gal_freq
        self.iono_source = self.configuration.ionosphere_model
        self.antenna_height = antenna_h
        self.phase_shift = phase_shift_dict
        self.pco = sat_pco
        self.rec_pco = rec_pco

        self.F = {
            'L1': 1575.42e06,
            'L2': 1227.60e06,
            'L5': 1176.45e06,
            'E1': 1575.420e06,
            'E5a': 1176.450e06,
            'E5b': 1207.140e06
        }

        self.l = {
            'L1': 299792458 / 1575.42e06,
            'L5': 299792458 / 1176.45e06,
            'L2': 299792458 / 1227.60e06,
            'E5a': 299792458 / 1176.45e06,
            'E5b': 299792458 / 1207.140e06
        }

        # Wartości pomocnicze (jeśli model wymaga ich użycia)
        self.gpsa = gpsa
        self.gpsb = gpsb
        self.gala = gala

        self.gim_file = self.configuration.gim_path

    def sat_fixed_system(self):

        epochs = self.df.index.get_level_values('time').unique().tolist()

        sunpos = get_sun_ecef(epochs)
        sunpos = sunpos.rename(columns={'x':'xs','y':'ys','z':'zs'})

        # Check for nested arrays or lists in sunpos columns
        for col in sunpos.columns:
            sunpos[col] = sunpos[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)


        # Convert sunpos columns to numeric types
        sunpos[['xs', 'ys', 'zs']] = sunpos[['xs', 'ys', 'zs']].astype(np.float64)

        ps = self.df.join(sunpos, how='left', on='time')

        # Ensure numeric types and handle missing values
        numeric_cols = ['xe', 'ye', 'ze', 'xs', 'ys', 'zs']
        ps[numeric_cols] = ps[numeric_cols].apply(pd.to_numeric, errors='coerce')
        ps = ps.dropna(subset=numeric_cols)
        # Vectorized calculation of k
        xyz = ps[['xe', 'ye', 'ze']].values
        norms_xyz = np.linalg.norm(xyz, axis=1, keepdims=True) # sprawdzone
        k = -xyz / norms_xyz # sprawdzone
        k_df = pd.DataFrame(k, columns=['k1', 'k2', 'k3'], index=ps.index)

        # Vectorized calculation of e (unit vector from receiver to satellite)
        sunpos_values = ps[['xs', 'ys', 'zs']].values
        d = sunpos_values - xyz
        norms_d = np.linalg.norm(d, axis=1, keepdims=True)
        e = d / norms_d
        e_df = pd.DataFrame(e, columns=['e1', 'e2', 'e3'], index=ps.index)

        # Vectorized calculation of j
        j = np.cross(k, e) # sprawdzone
        j_df = pd.DataFrame(j, columns=['j1', 'j2', 'j3'], index=ps.index)

        # Vectorized calculation of i
        i = np.cross(j, k)
        i_df = pd.DataFrame(i, columns=['i1', 'i2', 'i3'], index=ps.index)

        # Prepare pco DataFrame
        unique_sv = ps.index.get_level_values('sv').unique()
        if self.SYS == 'G':
            if self.mode == 'L1':

                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco[(ind, f'{ind[0]}01')]) / 1000

            elif self.mode == 'L1L2':
                f1 = self.F['L1']
                f2 = self.F['L2']
                pco_dict = {}
                for ind in unique_sv:

                    pco_dict[ind] = np.array(
                        (f1 ** 2 * np.array(self.pco[(ind, f'{ind[0]}01')]) - f2 ** 2 * np.array(
                            self.pco[(ind, f'{ind[0]}02')])) / (
                                f1 ** 2 - f2 ** 2)
                    ) / 1000
            elif self.mode == 'L1L5':
                f1 = self.F['L1']
                f2 = self.F['L5']
                pco_dict = {}
                for ind in unique_sv:
                    pco_l1 = np.array(self.pco[(ind, f'{ind[0]}01')])
                    key_l5 = (ind, 'G05')
                    if key_l5 in self.pco.keys():
                        pco_l5 = np.array(self.pco[key_l5])
                    else:
                        pco_l5 = np.array([0.0, 0.0, 0.0])

                    pco_dict[ind] = np.array(
                        (f1 ** 2 * pco_l1 - f2 ** 2 * pco_l5) / (
                                f1 ** 2 - f2 ** 2)
                    ) / 1000

            elif self.mode == 'L5':
                pco_dict ={}
                for ind in unique_sv:
                    key = (ind, 'G05')
                    if key in self.pco.keys():
                        pco_dict[ind] = np.array(self.pco[(ind, 'G05')]) / 1000
                    else:
                        pco_dict[ind] = np.array([0.0, 0.0, 0.0])
        elif self.SYS =='E':
            if self.mode == 'E1':
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco[(ind, f'{ind[0]}01')]) / 1000
            if self.mode == 'E5a':
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco[(ind, f'{ind[0]}05')]) / 1000

            elif self.mode == 'E1E5a':
                fe1a = self.F['E1']
                fe5a = self.F['E5a']
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(
                        (fe1a ** 2 * np.array(self.pco[(ind, f'{ind[0]}01')]) - fe5a ** 2 * np.array(
                            self.pco[(ind, f'{ind[0]}05')])) / (
                                fe1a ** 2 - fe5a ** 2)
                    ) / 1000
            elif self.mode == 'E1E5b':
                fe1a = self.F['E1']
                fe5b = self.F['E5b']
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(
                        (fe1a ** 2 * np.array(self.pco[(ind, f'{ind[0]}01')]) - fe5b ** 2 * np.array(
                            self.pco[(ind, f'{ind[0]}07')])) / (
                                fe1a ** 2 - fe5b ** 2)
                    ) / 1000

            elif self.mode == 'E5aE5b':
                fe1a = self.F['E5a']
                fe5b = self.F['E5b']
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(
                        (fe1a ** 2 * np.array(self.pco[(ind, f'{ind[0]}05')]) - fe5b ** 2 * np.array(
                            self.pco[(ind, f'{ind[0]}07')])) / (
                                fe1a ** 2 - fe5b ** 2)
                    ) / 1000

        # PCO DF = pco dla danego MODE
        # to co ponizej ,az do funkcji, musi byc wykonane dla kazdego mode
        # dlatego lepiej naliczac los niz dodawac, chyba ze w URE
        # dla L1L2 tryb SINGLE
        # tworzymy liste pco_df_L1, pco_df_L1
        # liczymy delta_L1, delta_L2 -> sat_pco_los_L1, sat_pco_los_L2

        # PARAMETR STERUJACY (pco at los, apply to crd)
        pco_df = pd.DataFrame.from_dict(pco_dict, orient='index', columns=['pco1', 'pco2', 'pco3'])
        pco_df.index.name = 'sv'
        # Merge pco into ps
        ps = ps.reset_index()
        ps = ps.merge(pco_df, on='sv', how='left')
        ps.set_index(['time', 'sv'], inplace=True)

        # Add k, e, i, j into ps
        ps = ps.join(k_df)
        ps = ps.join(e_df)
        ps = ps.join(i_df)
        ps = ps.join(j_df)

        # Compute rotation matrices
        i_vectors = ps[['i1', 'i2', 'i3']].values
        j_vectors = ps[['j1', 'j2', 'j3']].values
        k_vectors = ps[['k1', 'k2', 'k3']].values
        rot_matrices = np.stack((i_vectors, j_vectors, k_vectors), axis=2)  # Shape: (N, 3, 3)

        # Get pco_vectors
        pco_vectors = ps[['pco1', 'pco2', 'pco3']].values  # Shape: (N, 3)

        # Compute delta (vectorized matrix multiplication)
        delta = np.einsum('ijk,ik->ij', rot_matrices, pco_vectors)  # Shape: (N, 3)

        # Update 'xe', 'ye', 'ze' with delta
        ps[['x_apc', 'y_apc', 'z_apc']] = ps[['xe', 'ye', 'ze']]+delta

        self.df = ps.copy()

    def sat_pco_los(self):
        epochs = self.df.index.get_level_values('time').unique().tolist()
        sunpos = get_sun_ecef(epochs)
        sunpos = sunpos.rename(columns={'x': 'xs', 'y': 'ys', 'z': 'zs'})

        # Check for nested arrays or lists in sunpos columns
        for col in sunpos.columns:
            sunpos[col] = sunpos[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

        # Convert sunpos columns to numeric types
        sunpos[['xs', 'ys', 'zs']] = sunpos[['xs', 'ys', 'zs']].astype(np.float64)

        ps = self.df.join(sunpos, how='left', on='time')

        # Ensure numeric types and handle missing values
        numeric_cols = ['xe', 'ye', 'ze', 'xs', 'ys', 'zs']
        ps[numeric_cols] = ps[numeric_cols].apply(pd.to_numeric, errors='coerce')
        ps = ps.dropna(subset=numeric_cols)
        # Vectorized calculation of k
        xyz = ps[['xe', 'ye', 'ze']].values
        norms_xyz = np.linalg.norm(xyz, axis=1, keepdims=True)  # sprawdzone
        k = -xyz / norms_xyz  # sprawdzone
        k_df = pd.DataFrame(k, columns=['k1', 'k2', 'k3'], index=ps.index)

        # Vectorized calculation of e (unit vector from receiver to satellite)
        sunpos_values = ps[['xs', 'ys', 'zs']].values
        d = sunpos_values - xyz
        norms_d = np.linalg.norm(d, axis=1, keepdims=True)
        e = d / norms_d
        e_df = pd.DataFrame(e, columns=['e1', 'e2', 'e3'], index=ps.index)

        # Vectorized calculation of j
        j = np.cross(k, e)  # sprawdzone
        j_df = pd.DataFrame(j, columns=['j1', 'j2', 'j3'], index=ps.index)

        # Vectorized calculation of i
        i = np.cross(j, k)
        i_df = pd.DataFrame(i, columns=['i1', 'i2', 'i3'], index=ps.index)

        # Prepare pco DataFrame
        unique_sv = ps.index.get_level_values('sv').unique()
        def pco_LOS(delta, xyz_sat, xyz_sta):
            dist = calculate_distance(xyz_sat, xyz_sta)
            dx = xyz_sat[:,0] - xyz_sta[0]
            dy = xyz_sat[:,1] - xyz_sta[1]
            dz = xyz_sat[:,2] - xyz_sta[2]
            e = -np.column_stack((dx / dist, dy / dist, dz / dist))
            pco_los  = np.einsum('ij,ij->i', delta, e)
            return pco_los

        def process_frequency(freq, unique_sv, ps,k_df, i_df, j_df, e_df, xyz_sat):
            if freq == 'L1':
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco[(ind, f'{ind[0]}01')]) / 1000
            elif freq == 'L2':
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco[(ind, f'{ind[0]}02')]) / 1000
            elif freq == 'L5':
                pco_dict = {}
                for ind in unique_sv:
                    val = self.pco.get((ind, f'{ind[0]}05'),[0.0,0.0,0.0])
                    pco_dict[ind] = np.array(val) / 1000
            elif freq == 'E1':
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco[(ind, f'{ind[0]}01')]) / 1000
            elif freq == 'E5a':
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco[(ind, f'{ind[0]}05')]) / 1000
            elif freq == 'E5b':
                pco_dict = {}
                for ind in unique_sv:
                    pco_dict[ind] = np.array(self.pco.get((ind, f'{ind[0]}07'),[0.0,0.0,0.0])) / 1000
            pco_df = pd.DataFrame.from_dict(pco_dict, orient='index', columns=['pco1', 'pco2', 'pco3'])
            pco_df.index.name = 'sv'
            ps = ps.reset_index()
            ps = ps.merge(pco_df, on='sv', how='left')
            ps.set_index(['time', 'sv'], inplace=True)

            # Add k, e, i, j into ps
            ps = ps.join(k_df)
            ps = ps.join(e_df)
            ps = ps.join(i_df)
            ps = ps.join(j_df)

            # Compute rotation matrices
            i_vectors = ps[['i1', 'i2', 'i3']].values
            j_vectors = ps[['j1', 'j2', 'j3']].values
            k_vectors = ps[['k1', 'k2', 'k3']].values
            rot_matrices = np.stack((i_vectors, j_vectors, k_vectors), axis=2)  # Shape: (N, 3, 3)

            # Get pco_vectors
            pco_vectors = ps[['pco1', 'pco2', 'pco3']].values  # Shape: (N, 3)

            # Compute delta (vectorized matrix multiplication)
            delta = np.einsum('ijk,ik->ij', rot_matrices, pco_vectors)  # Shape: (N, 3)
            pco_los = pco_LOS(delta=delta, xyz_sat=xyz_sat, xyz_sta=self.xyz.copy())
            self.df[f'sat_pco_los_{freq}'] = pco_los

        if self.mode in ['L1','L2','L5']:
            process_frequency(freq=self.mode, unique_sv=unique_sv, ps=ps, k_df=k_df, j_df=j_df, i_df=i_df,
                              e_df=e_df, xyz_sat=xyz)
        elif self.mode in ['E1','E5a','E5b']:
            process_frequency(freq=self.mode, unique_sv=unique_sv, ps=ps, k_df=k_df, j_df=j_df, i_df=i_df,
                              e_df=e_df, xyz_sat=xyz)
        elif self.mode == 'L1L2':
            for freq in ['L1','L2']:
                process_frequency(freq=freq,unique_sv=unique_sv,ps=ps,k_df=k_df,j_df=j_df,i_df=i_df,
                                  e_df=e_df,xyz_sat=xyz)
        elif self.mode == 'L1L5':
            for freq in ['L1','L5']:
                process_frequency(freq=freq,unique_sv=unique_sv,ps=ps,k_df=k_df,j_df=j_df,i_df=i_df,
                                  e_df=e_df,xyz_sat=xyz)
        elif self.mode == 'L2L5':
            for freq in ['L2','L5']:
                process_frequency(freq=freq,unique_sv=unique_sv,ps=ps,k_df=k_df,j_df=j_df,i_df=i_df,
                                  e_df=e_df,xyz_sat=xyz)

        elif self.mode == 'E1E5a':
            for freq in ['E1','E5a']:
                process_frequency(freq=freq,unique_sv=unique_sv,ps=ps,k_df=k_df,j_df=j_df,i_df=i_df,
                                  e_df=e_df,xyz_sat=xyz)
        elif self.mode == 'E1E5b':
            for freq in ['E1','E5b']:
                process_frequency(freq=freq,unique_sv=unique_sv,ps=ps,k_df=k_df,j_df=j_df,i_df=i_df,
                                  e_df=e_df,xyz_sat=xyz)
        elif self.mode == 'E5aE5b':
            for freq in ['E5a','E5b']:
                process_frequency(freq=freq,unique_sv=unique_sv,ps=ps,k_df=k_df,j_df=j_df,i_df=i_df,
                                  e_df=e_df,xyz_sat=xyz)




    def phase2meters(self):
        self.df= self.df.copy()
        if self.SYS =='G':
            if self.mode in ['L1L2','L1','L2']:
                self.df[[col for col in self.df.columns if col.startswith('L1')]] *= self.l['L1']
                self.df[[col for col in self.df.columns if col.startswith('L2')]] *= self.l['L2']
            if self.mode in ['L1L5','L5']:
                self.df[[col for col in self.df.columns if col.startswith('L1')]] *= self.l['L1']
                self.df[[col for col in self.df.columns if col.startswith('L5')]] *= self.l['L5']
            if self.mode in ['L2L5']:
                self.df[[col for col in self.df.columns if col.startswith('L2')]] *= self.l['L2']
                self.df[[col for col in self.df.columns if col.startswith('L5')]] *= self.l['L5']
        elif self.SYS == 'E':
            if self.mode in ['E1E5a','E1','E5a']:
                self.df[[col for col in self.df.columns if col.startswith('L1')]] *= self.l['L1']
                self.df[[col for col in self.df.columns if col.startswith('L5')]] *= self.l['E5a']
            if self.mode in ['E1E5b','E5b']:
                self.df[[col for col in self.df.columns if col.startswith('L1')]] *= self.l['L1']
                self.df[[col for col in self.df.columns if col.startswith('L7')]] *= self.l['E5b']
            if self.mode in ['E5aE5b']:
                self.df[[col for col in self.df.columns if col.startswith('L5')]] *= self.l['E5a']
                self.df[[col for col in self.df.columns if col.startswith('L7')]] *= self.l['E5b']


    def apply_phase_shift(self):
        # Pobieramy kolumny, których nazwy zaczynają się od "L" (np. L1C, L2P itd.)
        cols = [col for col in self.df.columns if col.startswith('L')]

        if self.phase_shift is None:
            print('No phase shift applied')
            return

        # Tryb L1L2: rozróżniamy kolumny L1 oraz L2
        elif self.mode in ['L1L2','L1','L2']:
            for c in cols:
                key = f"{self.SYS} {c}"
                shift = self.phase_shift.get(key, 0.0)
                # Jeśli kolumna dotyczy L2, używamy długości fali L2, w przeciwnym wypadku L1
                if c.startswith("L2"):
                    self.df[c] = self.df[c] + self.l['L2'] * np.float32(shift)
                elif c.startswith("L1"):
                    self.df[c] = self.df[c] + self.l['L1'] * np.float32(shift)
        elif self.mode in ['L1L5','L5']:
            for c in cols:
                key = f"{self.SYS} {c}"
                shift = self.phase_shift.get(key, 0.0)

                # Jeśli kolumna dotyczy L2, używamy długości fali L2, w przeciwnym wypadku L1
                if c.startswith("L1"):
                    self.df[c] = self.df[c] + self.l['L1'] * np.float32(shift)
                elif c.startswith("L5"):
                    self.df[c] = self.df[c] + self.l['L5'] * np.float32(shift)
        elif self.mode in ['L2L5']:
            for c in cols:
                key = f"{self.SYS} {c}"
                shift = self.phase_shift.get(key, 0.0)

                # Jeśli kolumna dotyczy L2, używamy długości fali L2, w przeciwnym wypadku L1
                if c.startswith("L2"):
                    self.df[c] = self.df[c] + self.l['L2'] * np.float32(shift)
                elif c.startswith("L5"):
                    self.df[c] = self.df[c] + self.l['L5'] * np.float32(shift)

        elif self.mode in ['E1E5b','E5b']:
            for c in cols:
                key = f"{self.SYS} {c}"
                shift = self.phase_shift.get(key, 0.0)

                # Jeśli kolumna dotyczy L2, używamy długości fali L2, w przeciwnym wypadku L1

                if c.startswith("L1"):
                    self.df[c] = self.df[c] + self.l['L1'] * np.float32(shift)
                elif c.startswith("L7"):
                    self.df[c] = self.df[c] + self.l['E5b'] * np.float32(shift)
        elif self.mode in ['E5aE5b']:
            for c in cols:
                key = f"{self.SYS} {c}"
                shift = self.phase_shift.get(key, 0.0)

                # Jeśli kolumna dotyczy L2, używamy długości fali L2, w przeciwnym wypadku L1

                if c.startswith("L5"):
                    self.df[c] = self.df[c] + self.l['E5a'] * np.float32(shift)
                elif c.startswith("L7"):
                    self.df[c] = self.df[c] + self.l['E5b'] * np.float32(shift)

        elif self.mode in ['E1E5a','E1','E5a']:
            for c in cols:
                key = f"{self.SYS} {c}"
                shift = self.phase_shift.get(key, 0.0)

                # Jeśli kolumna dotyczy L2, używamy długości fali L2, w przeciwnym wypadku L1

                if c.startswith("L1"):
                    self.df[c] = self.df[c] + self.l['L1'] * np.float32(shift)
                elif c.startswith("L5"):
                    self.df[c] = self.df[c] + self.l['E5a'] * np.float32(shift)


        else:
            print('Unknown mode')

    def compute_klobuchar(self):
        self.df = self.df.copy()
        epochs = self.df.index.get_level_values('time').tolist()

        toc_list = datetime2toc(epochs)
        az = self.df['az'].to_numpy()
        ev = self.df['ev'].to_numpy()

        ion = np.array(list(map(lambda row: klobuchar(
            azimuth=row[0], elev=row[1], fi=self.flh[0], lambda_=self.flh[1],
            tow=row[2], beta=self.gpsb, alfa=self.gpsa), zip(az, ev, toc_list))))
        self.df['ion'] = ion

    def enu2ecef(self, x):
        """
        Rotation matrix from ENU to ECEF for 3D matrixes (obviously)
        :param x: np.array([east, north, up]) vector
        :return:
        """
        flh = np.array([np.deg2rad(self.flh[0]),
                        np.deg2rad(self.flh[1])
                        ])
        rot = np.array([[-np.sin(flh[1]), -np.cos(flh[1]) * np.sin(flh[0]), np.cos(flh[1]) * np.cos(flh[0])],
                        [np.cos(flh[1]), -np.sin(flh[1]) * np.sin(flh[0]), np.sin(flh[1]) * np.cos(flh[0])],
                        [0, np.cos(flh[0]), np.sin(flh[0])]])
        return rot.dot(x)

    def compute_antenna_height(self):
        self.antenna_height = np.array([self.antenna_height[1], self.antenna_height[2], self.antenna_height[0]])
        self.xyz + self.enu2ecef(x=self.antenna_height)
        return self.xyz + self.enu2ecef(x=self.antenna_height)

    def project_antenna_height(self):
        xyz = self.xyz
        dist = calculate_distance(self.df[['xe', 'ye', 'ze']].to_numpy(), xyz)
        dx = self.df['xe'].to_numpy() - xyz[0]
        dy = self.df['ye'].to_numpy() - xyz[1]
        dz = self.df['ze'].to_numpy() - xyz[2]
        e = -np.column_stack((dx / dist, dy / dist, dz / dist))
        enu = np.array([self.antenna_height[1], self.antenna_height[2],self.antenna_height[0]])
        ecef = self.enu2ecef(x=enu)
        ah_los = e.dot(ecef)
        self.df['ah_los'] = ah_los

    def compute_pco_at_los(self):
        """
        Oblicza korekcję PCO wzdłuż wektora widzenia (LOS) i zapisuje wynik
        do odpowiednich kolumn DataFrame, w zależności od systemu (E lub G) i trybu.
        """
        # Pobranie pozycji odbiornika oraz współrzędnych obserwacyjnych
        xyz = self.xyz
        coords = self.df[['xe', 'ye', 'ze']].to_numpy()
        dist = calculate_distance(coords, xyz)
        dx = coords[:, 0] - xyz[0]
        dy = coords[:, 1] - xyz[1]
        dz = coords[:, 2] - xyz[2]
        # Wektor jednostkowy skierowany do satelity (korekta ujemna)
        e = -np.column_stack((dx / dist, dy / dist, dz / dist))

        # Jeśli brak danych PCO, ustawiamy korekcję na 0 i kończymy funkcję
        if not any(self.rec_pco.items()):
            print('Reciever PCO not found in Antex file!')
            print('Reciever PCO applied: ', self.rec_pco)
            self.df['pco_los'] = 0.0
            return

        def get_pco(key_suffix):
            """
            Pomocnicza funkcja do pobierania wartości PCO.
            Zwraca wartość przeliczoną na kilometry.
            """
            for key, value in self.rec_pco.items():
                if key[-1] == key_suffix:
                    return np.array(value) / 1000.
            return np.array([0.0, 0.0, 0.0])

        # Gałąź dla systemu E
        if self.SYS == 'E':
            if self.mode == 'E1':
                try:
                    pco = get_pco('E01')
                except KeyError:
                    pco = get_pco('G01')
                pco_ecef = self.enu2ecef(x=pco)
                self.df['pco_los'] = e.dot(pco_ecef)
            elif self.mode =='E5a':
                pco = get_pco('E05')
                pco_ecef = self.enu2ecef(x=pco)
                self.df['pco_los'] = e.dot(pco_ecef)
            elif self.mode =='E5b':
                pco = get_pco('E07')
                pco_ecef = self.enu2ecef(x=pco)
                self.df['pco_los'] = e.dot(pco_ecef)
            elif self.mode == 'L1L2':
                raise ValueError("Tryb L1L2 nie jest zaimplementowany dla systemu E")
            elif self.mode == 'E1E5a':
                f1 = self.F['E1']
                fe5a = self.F['E5a']
                try:
                    try:
                        pco_L1 = get_pco('E01')
                    except KeyError:
                        pco_L1 = get_pco('G01')
                    pco_L2 = get_pco('E05')
                except KeyError:
                    pco_L1 = np.array([0.0, 0.0, 0.0])
                    pco_L2 = np.array([0.0, 0.0, 0.0])
                pco_if = (f1 ** 2 * pco_L1 - fe5a ** 2 * pco_L2) / (f1 ** 2 - fe5a ** 2)
                pco_l1 = self.enu2ecef(x=pco_L1)
                pco_l2 = self.enu2ecef(x=pco_L2)
                pco_if_ecef = self.enu2ecef(x=pco_if)
                self.df['pco_los_l1'] = e.dot(pco_l1)
                self.df['pco_los_l2'] = e.dot(pco_l2)
                self.df['pco_los_l3'] = e.dot(pco_if_ecef)
            elif self.mode == 'E1E5b':
                f1 = self.F['E1']
                fe5b = self.F['E5b']
                try:
                    try:
                        pco_L1 = get_pco('E01')
                    except KeyError:
                        pco_L1 = get_pco('G01')
                    pco_L2 = get_pco('E07')
                except KeyError:
                    pco_L1 = np.array([0.0, 0.0, 0.0])
                    pco_L2 = np.array([0.0, 0.0, 0.0])
                pco_if = (f1 ** 2 * pco_L1 - fe5b ** 2 * pco_L2) / (f1 ** 2 - fe5b ** 2)
                pco_l1 = self.enu2ecef(x=pco_L1)
                pco_l2 = self.enu2ecef(x=pco_L2)
                pco_if_ecef = self.enu2ecef(x=pco_if)
                self.df['pco_los_l1'] = e.dot(pco_l1)
                self.df['pco_los_l5'] = e.dot(pco_l2)
                self.df['pco_los_l3'] = e.dot(pco_if_ecef)

            elif self.mode == 'E5aE5b':
                f1 = self.F['E5a']
                fe5b = self.F['E5b']
                try:
                    try:
                        pco_L1 = get_pco('E05')
                    except KeyError:
                        pco_L1 = get_pco('G01')
                    pco_L2 = get_pco('E07')
                except KeyError:
                    pco_L1 = np.array([0.0, 0.0, 0.0])
                    pco_L2 = np.array([0.0, 0.0, 0.0])
                pco_if = (f1 ** 2 * pco_L1 - fe5b ** 2 * pco_L2) / (f1 ** 2 - fe5b ** 2)
                pco_l1 = self.enu2ecef(x=pco_L1)
                pco_l2 = self.enu2ecef(x=pco_L2)
                pco_if_ecef = self.enu2ecef(x=pco_if)
                self.df['pco_los_l1'] = e.dot(pco_l1)
                self.df['pco_los_l5'] = e.dot(pco_l2)
                self.df['pco_los_l3'] = e.dot(pco_if_ecef)
            else:
                raise ValueError(f"Nieobsługiwany tryb dla systemu E: {self.mode}")

        # Gałąź dla systemu G
        elif self.SYS == 'G':

            if self.mode == 'L1':
                pco = get_pco('G01')
                pco_ecef = self.enu2ecef(x=pco)
                self.df['pco_los'] = e.dot(pco_ecef)
            elif self.mode =='L2':
                pco = get_pco('G02')
                pco_ecef = self.enu2ecef(x=pco)
                self.df['pco_los'] = e.dot(pco_ecef)
            elif self.mode =='L5':
                pco = get_pco('G05')
                pco_ecef = self.enu2ecef(x=pco)
                self.df['pco_los'] = e.dot(pco_ecef)
            elif self.mode == 'L1L2':
                f1 = self.F['L1']
                f2 = self.F['L2']
                pco_L1 = get_pco('G01')
                pco_L2 = get_pco('G02')
                pco_if = (f1 ** 2 * pco_L1 - f2 ** 2 * pco_L2) / (f1 ** 2 - f2 ** 2)
                pco_l1 = self.enu2ecef(x=pco_L1)
                pco_l2 = self.enu2ecef(x=pco_L2)
                pco_if_ecef = self.enu2ecef(x=pco_if)
                self.df['pco_los_l1'] = e.dot(pco_l1)
                self.df['pco_los_l2'] = e.dot(pco_l2)
                self.df['pco_los_l3'] = e.dot(pco_if_ecef)
            elif self.mode == 'L1L5':
                f1 = self.F['L1']
                f5 = self.F['L5']  # Założenie: klucz 'f5' istnieje w self.F
                pco_L1 = get_pco('G01')
                pco_L2 = get_pco('G05')
                pco_if = (f1 ** 2 * pco_L1 - f5 ** 2 * pco_L2) / (f1 ** 2 - f5 ** 2)
                pco_l1 = self.enu2ecef(x=pco_L1)
                pco_l2 = self.enu2ecef(x=pco_L2)
                pco_if_ecef = self.enu2ecef(x=pco_if)
                self.df['pco_los_l1'] = e.dot(pco_l1)
                self.df['pco_los_l2'] = e.dot(pco_l2)
                self.df['pco_los_l3'] = e.dot(pco_if_ecef)
            elif self.mode == 'L2L5':
                f1 = self.F['L2']
                f5 = self.F['L5']  # Założenie: klucz 'f5' istnieje w self.F
                pco_L1 = get_pco('G02')
                pco_L2 = get_pco('G05')
                pco_if = (f1 ** 2 * pco_L1 - f5 ** 2 * pco_L2) / (f1 ** 2 - f5 ** 2)
                pco_l1 = self.enu2ecef(x=pco_L1)
                pco_l2 = self.enu2ecef(x=pco_L2)
                pco_if_ecef = self.enu2ecef(x=pco_if)
                self.df['pco_los_l2'] = e.dot(pco_l1)
                self.df['pco_los_l2'] = e.dot(pco_l2)
                self.df['pco_los_l3'] = e.dot(pco_if_ecef)
            elif self.mode == 'E1E5b':
                raise ValueError("Tryb E1E5b jest zaimplementowany jedynie dla systemu E")
            else:
                raise ValueError(f"Nieobsługiwany tryb dla systemu G: {self.mode}")

        else:
            raise ValueError(f"Nieobsługiwany sys (SYS): {self.SYS}")

    def compute_ntcm(self):

        grouped = self.df.groupby(level='time')

        # Define the wrapper function
        def apply_run_NTCM_for_gal(group):
            az = group['az'].values
            ev = group['ev'].values
            epoch = group.name  # The group name is the time epoch
            DOY = datetime2doy(epoch)
            # Call the function
            stec = ntcm_vtec(
                pos=self.flh,
                DOY=DOY,
                ev=ev,
                az=az,
                epoch=epoch,
                gala=self.gala,
                mf_no=2  # or self.mf_no if applicable
            )
            return pd.Series(stec, index=group.index)

        # Apply the function to each group
        result = grouped.apply(apply_run_NTCM_for_gal)
        result.index = result.index.droplevel(0)  # Remove the first level

        # Assign the result back to the DataFrame
        stec = result
        K = 40.3 * 10 ** 16
        if self.mode in ['L1','L1L2','L1L5']:
            f = self.F['L1']
            ion = K * (stec / (f ** 2))
            self.df['ion'] = ion
        elif self.mode in ['L2','L2L5']:
            f = self.F['L2']
            ion = K * (stec / (f ** 2))
            self.df['ion'] = ion
        elif self.mode =='L5':
            f = self.F['L5']
            ion = K * (stec / (f ** 2))
            self.df['ion'] = ion


        elif self.mode in ['E1','E1E5a','E1E5b']:
            f = self.F['L1']
            ion = K * (stec / (f ** 2))
            self.df['ion'] = ion
        elif self.mode in ['E5a','E5aE5b']:
            f = self.F['E5a']
            ion = K * (stec / (f ** 2))
            self.df['ion'] = ion
        elif self.mode =='E5b':
            f = self.F['E5b']
            ion = K * (stec / (f ** 2))
            self.df['ion'] = ion
        else:
            raise ValueError('Unkown mode')

    def compute_troposphere_vectorized(self):
        ev = self.df['ev'].to_numpy()
        tro = np.array(list(map(lambda row: saastamoinen(row, h=self.flh[2]), ev)))
        self.df['tro'] = tro


    def read_inx(self):
        K = 40.3 * 10 ** 16
        if self.mode in ['L1','E1', 'L1L2','E1E5a','L1L5','E1E5b']:
            f = self.F['L1']
        elif self.mode in ['L5','E5a','E5aE5b']:
            f = self.F['E5a']
        elif self.mode in ['E5b']:
            f = self.F['E5b']
        elif self.mode in ['L2','L2L5']:
            f = self.F['L2']

        parser = GIMReader(tec_path=self.gim_file,dcb_path=self.gim_file)
        data = parser.read(dcb=True)
        interp = TECInterpolator(data)
        geodetic = ecef2geodetic(self.xyz)
        lat_deg, lon_deg = geodetic[0], geodetic[1]
        # lat_deg, lon_deg = np.rad2deg(lat_deg), np.rad2deg(lon_deg)
        lat_ipp, lon_ipp = interp.get_ipp(self.df['ev'].to_numpy(), self.df['az'].to_numpy(),
                                          lat_deg, lon_deg)
        self.df['lat_ipp'] = lat_ipp
        self.df['lon_ipp'] = lon_ipp
        m_f = 1 / np.sqrt(1 - (interp.R / (interp.R + interp.ish)) ** 2 * np.cos(np.deg2rad(self.df['ev'])) ** 2)
        self.df['ion'] = K * (
                data.tec.TEC.interp(
                    time=('p', self.df.index.get_level_values('time').to_numpy('datetime64[ns]')),
                    lat=('p', lat_ipp),
                    lon=('p', lon_ipp),
                    method='linear').values
                * m_f / f ** 2
        )
        if getattr(self.configuration,'use_iono_rms',False):
            self.df['ion_rms'] = K * (
                    data.rms.RMS.interp(
                        time=('p', self.df.index.get_level_values('time').to_numpy('datetime64[ns]')),
                        lat=('p', lat_ipp),
                        lon=('p', lon_ipp),
                        method='linear').values
                    * m_f / f ** 2
            )

        if data.dcb is not None:

            sat_dcb = data.dcb[(data.dcb['entry_type'] == 'satellite') &
                     data.dcb['sv'].str.startswith(self.SYS)]
            sat_dcb = sat_dcb.set_index('sv')

            self.df = self.df.join(sat_dcb,on='sv')
            if self.configuration.station_name is not None:
                name = self.configuration.station_name
                station = (data.dcb[(data.dcb['entry_type'] == 'station') &
                                    (data.dcb['prn_or_site'].str.startswith(name))])
                sta_bias = station[station['system'] == self.SYS]
                if not sta_bias.empty:
                    self.df['sta_bias'] = sta_bias['bias'].values[0]


    def compute_niell(self):
        tro, dry, wet, me_wet = niell(ev=self.df['ev'].to_numpy(),lat=self.flh[0],lon=self.flh[1], h=self.flh[2],
                                      doy=self.doy)
        self.df=self.df.copy()
        self.df['tro'] = 0.0
        self.df.loc[:, 'tro'] = tro
        self.df.loc[:, 'dry'] = dry
        self.df.loc[:, 'wet'] = wet
        self.df.loc[:, 'me_wet'] = me_wet

    def compute_collins(self):
        self.df['tro_co'] = self.df.apply(lambda row: tropospheric_delay(f=self.flh[0],
                                                                         h=self.flh[2],
                                                                         elevation=row['ev'],
                                                                         doy=self.doy),
                                          axis=1)


    def compute_tides(self):
        epochs = self.df.index.get_level_values('time').unique().tolist()
        tides, radial, north, sun_tides, moon_tides = get_tides(xyz=self.xyz,datetimes=epochs)
        tides['N'] = north
        tides['R']=radial
        dX = self.df[['xe','ye','ze']].to_numpy() - self.xyz
        dist = np.linalg.norm(dX, axis=1)
        LOS = -dX / dist[:, np.newaxis]
        self.df[['LOS1', 'LOS2', 'LOS3']] = LOS
        self.df = self.df.join(tides,on=['time'])

        # tides_los
        los = self.df[['LOS1', 'LOS2', 'LOS3']].to_numpy()
        tides = self.df[['dx','dy','dz']].to_numpy()
        tides_los = np.sum(los*tides,axis=1)
        self.df['tides_los']=tides_los

    def compute_corrections(self, get_windup=True, get_rel=True):
        out_dict = {}

        # 1) unikalne epoki raz
        all_times = pd.to_datetime(self.df.index.get_level_values("time"))
        uniq_times = pd.Index(all_times.unique()).sort_values()

        # 2) słońce raz dla wszystkich epok
        sun_df = None
        if get_windup:
            sun_df = get_sun_ecef(uniq_times)  # index=time, cols x y z


        for sv, gr in self.df.groupby("sv", sort=False):
            times = pd.to_datetime(gr.index.get_level_values("time"))
            satpositions = gr[["xe", "ye", "ze"]].to_numpy()

            if get_windup:
                rsun_all = sun_df.loc[times, ["x", "y", "z"]].to_numpy()
                windup = process_windup_correction_vectorized(times, satpositions, self.xyz, rsun_all=rsun_all)
            else:
                windup = np.zeros(len(gr))

            if get_rel:
                path_rel = rel_path_corr(satpositions, self.xyz)
            else:
                path_rel = np.zeros(len(gr))

            out_dict[sv] = pd.DataFrame({"phw": windup, "dprel": path_rel}, index=times)

        out = pd.concat(out_dict, keys=out_dict.keys(), names=["sv", "time"])
        self.df = self.df.join(out)

    def _compute_corrections(self, get_windup=True, get_rel=True):
        out_dict = {}
        for ind, gr in self.df.groupby('sv'):
            times = pd.to_datetime(gr.index.get_level_values('time'))
            satpositions = gr[['xe', 'ye', 'ze']].to_numpy()
            if get_windup:
                windup = process_windup_correction_vectorized(times, satpositions, self.xyz.copy())
            else:
                windup = np.zeros(len(gr))
            if get_rel:
                path_rel = rel_path_corr(satpositions, self.xyz.copy())
            else:
                path_rel = np.zeros(len(gr))

            out = pd.DataFrame({
                'phw': windup,
                'dprel': path_rel
            }, index=times)
            out_dict[ind] = out
        out = pd.concat(out_dict, keys=out_dict.keys(), names=['sv', 'time'])
        self.df = self.df.join(out)

    def has_phase_observations(self) -> bool:
        phase_columns = [col for col in self.df.columns if col.startswith('L')]
        return len(phase_columns) > 0

    def run(self):
        if self.has_phase_observations():
            self.phase2meters()
            self.apply_phase_shift()
        else:
            print("No phase observations")
        if self.configuration.sat_pco=='los':
            self.sat_pco_los()
        elif self.configuration.sat_pco=='crd':
            self.sat_fixed_system()

        if self.configuration.antenna_h:
            self.project_antenna_height()
        if self.configuration.rec_pco:
            self.compute_pco_at_los()
        if self.configuration.troposphere_model =='niell':
            self.compute_niell()
        elif self.configuration.troposphere_model =='saastamoinen':
            self.compute_troposphere_vectorized()
        elif self.configuration.troposphere_model =='collins':
            self.compute_collins()
        if self.configuration.ionosphere_model =='gim':
            self.read_inx()
        elif self.configuration.ionosphere_model =='klobuchar':
            self.compute_klobuchar()
            self.df['ion_rms'] = 0.0
        elif self.configuration.ionosphere_model =='ntcm':
            self.compute_ntcm()
            self.df['ion_rms'] = 0.0
        if self.configuration.windup or self.configuration.rel_path:
            self.compute_corrections(get_windup=self.configuration.windup, get_rel=self.configuration.rel_path)
        if self.configuration.solid_tides:
            self.compute_tides()
        return self.df


