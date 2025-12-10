from typing import Union, Optional, Literal
import os
import numpy as np
import pandas as pd



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







