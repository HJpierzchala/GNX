from datetime import datetime, timedelta

import seaborn as sns
import matplotlib as mpl
sns.set_context("paper", font_scale=1.6)
sns.set_style("ticks")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

import pandas as pd
from IPython.display import Markdown, display

def df_tail(
    df: pd.DataFrame,
    nrows: int = 10,
    ncols: int = 6,
    index: bool = False,
    floatfmt: str = ".3f",
    truncate_str: int | None = 30,
    reset_index: bool = True,
):
    """
    Wyświetl końcowe wiersze DataFrame jako czytelną tabelę Markdown (do PDF),
    z ograniczeniem liczby wierszy/kolumn, cyfr i długości stringów.

    Parametry:
    - nrows: ile ostatnich wierszy pokazać
    - ncols: ile pierwszych kolumn pokazać
    - index: czy pokazywać index w tabeli
    - floatfmt: format liczb zmiennoprzecinkowych
    - truncate_str: maksymalna długość stringów (None = bez ucinania)
    - reset_index: czy wrzucić MultiIndex do kolumn
    """
    sub = df.copy()

    if reset_index:
        sub = sub.reset_index()

    sub = sub.iloc[-nrows:, :ncols].copy()

    # Spłaszczenie MultiIndex kolumn, jeśli jest
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = [
            "_".join(map(str, lvl)).strip()
            for lvl in sub.columns
        ]

    # 1) Zaokrąglanie kolumn numerycznych wg floatfmt
    ndigits = None
    if floatfmt.startswith(".") and floatfmt[1:].rstrip("f").isdigit():
        ndigits = int(floatfmt[1:].rstrip("f"))

    if ndigits is not None:
        num_cols = sub.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            sub[num_cols] = sub[num_cols].astype(float).round(ndigits)

    # 2) Ucinanie długich stringów w kolumnach tekstowych
    if truncate_str is not None:
        obj_cols = sub.select_dtypes(include=["object", "string"]).columns

        def _truncate(s: str) -> str:
            return (s[:truncate_str] + "…") if len(s) > truncate_str else s

        for col in obj_cols:
            sub[col] = sub[col].astype(str).map(_truncate)

    # 3) Generacja Markdown
    md = sub.to_markdown(
        index=index,
        tablefmt="github",
        floatfmt=floatfmt,
    )

    display(Markdown(md))

def df_head(
    df: pd.DataFrame,
    nrows: int = 10,
    ncols: int = 6,
    index: bool = False,
    floatfmt: str = ".3f",
    truncate_str: int | None = 30,
    reset_index: bool = True,
):
    """
    Wyświetl DataFrame jako czytelną tabelę Markdown (do PDF),
    z ograniczeniem liczby wierszy/kolumn, cyfr i długości stringów.

    Parametry:
    - nrows: ile pierwszych wierszy pokazać
    - ncols: ile pierwszych kolumn pokazać
    - index: czy pokazywać index w tabeli
    - floatfmt: format liczb zmiennoprzecinkowych (np. '.2f', '.3f')
    - truncate_str: maksymalna długość stringów (None = bez ucinania)
    - reset_index: czy wrzucić MultiIndex do kolumn (True = zwykle czytelniej)
    """
    sub = df.copy()

    if reset_index:
        sub = sub.reset_index()

    sub = sub.iloc[:nrows, :ncols].copy()

    # Spłaszczenie MultiIndex kolumn, jeśli jest
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = [
            "_".join(map(str, lvl)).strip()
            for lvl in sub.columns
        ]

    # 1) Zaokrąglanie kolumn numerycznych na podstawie floatfmt
    ndigits = None
    # oczekujemy formatu typu '.3f' albo '.3'
    if floatfmt.startswith(".") and floatfmt[1:].rstrip("f").isdigit():
        ndigits = int(floatfmt[1:].rstrip("f"))

    if ndigits is not None:
        num_cols = sub.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            sub[num_cols] = sub[num_cols].astype(float).round(ndigits)

    # 2) Ucinanie długich stringów TYLKO w kolumnach tekstowych
    if truncate_str is not None:
        obj_cols = sub.select_dtypes(include=["object", "string"]).columns

        def _truncate(s: str) -> str:
            return (s[:truncate_str] + "…") if len(s) > truncate_str else s

        for col in obj_cols:
            sub[col] = sub[col].astype(str).map(_truncate)

    # 3) Generacja Markdown – floatfmt nadal działa na kolumnach numerycznych
    md = sub.to_markdown(
        index=index,
        tablefmt="github",
        floatfmt=floatfmt,
    )

    display(Markdown(md))

def plot_residuals(df: pd.DataFrame,
                   cut_init_time: int = 30,
                   skip_obs: int = 5,
                   obs_type: str = 'phase',
                   x_axis: str = 'time',
                   title: str = None,
                   figsize: tuple = (12, 6)):
    
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex with levels ('sv', 'time') or ('time', 'sv')")

    if obs_type not in ['phase', 'code']:
        raise ValueError(f"obs_type must be 'phase' or 'code', got '{obs_type}'")

    if x_axis not in ['time', 'elev']:
        raise ValueError(f"x_axis must be 'time' or 'elev', got '{x_axis}'")

    all_times = df.index.get_level_values('time').unique().sort_values()

    if len(all_times) <= cut_init_time:
        raise ValueError(f"Not enough time epochs to cut {cut_init_time} initial steps")

    res_trimmed = df[df.index.get_level_values('time').isin(all_times[cut_init_time:])]

    fig, ax = plt.subplots(figsize=figsize)
    seen_prns = set()

    for sv, gr in res_trimmed.groupby('sv'):
        times = gr.index.get_level_values('time')
        if len(times) <= skip_obs:
            continue  # pomiń SV, który ma za mało epok

        first_time_sv = sorted(times)[skip_obs]
        gr_filtered = gr[times > first_time_sv]

        if gr_filtered.empty:
            continue

        prn = sv.split('_')[0]
        label = prn if prn not in seen_prns else None
        seen_prns.add(prn)

        # Wybór kolumny
        col = 'v' if obs_type == 'phase' else 'vc'
        if col not in gr_filtered.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

        # Wybór osi X
        if x_axis == 'time':
            x = gr_filtered.index.get_level_values('time')
        else:  # elevation
            if 'ev' not in gr_filtered.columns:
                raise KeyError("Column 'ev' (elevation) not found in DataFrame")
            x = gr_filtered['ev']

        ax.scatter(x, gr_filtered[col], label=label, s=2)

    # Styl
    if x_axis == 'time':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel("Time")
    else:
        ax.set_xlabel("Elevation [deg]")

    sys = prn[0] if 'prn' in locals() else '?'
    title_map = {'G': 'GPS', 'E': 'Galileo', 'phase': 'Carrier-phase', 'code': 'Pseudorange'}
    if title is None:
        ax.set_title(f"{title_map.get(sys, sys)} {title_map[obs_type]} residuals")
    else:
        ax.set_title(title)

    ax.set_ylabel("Residual [m]")
    ax.grid(True)

    # Legenda poza wykresem
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=2, title='PRN')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # zostawia miejsce na legendę
    plt.show()



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Optional, Union

def plot_positioning_series(sol: pd.DataFrame,
                             cut_init_time: int = 0,
                             figsize: tuple = (12, 8),
                             color: str = 'red',
                             title: str = None,
                             save_path: str = None,
                             ct: Optional[Union[None, int]] = None ):
    """
    Wykres komponentów pozycji i parametrów PPP:
    - dE, dN, dU, dtr, ISB

    Parametry:
    ----------
    sol : pd.DataFrame
        DataFrame z kolumnami: 'de', 'dn', 'du', 'dtr', 'isb', indeks = czas.
    cut_init_time : int
        Liczba początkowych epok do pominięcia (domyślnie 0).
    figsize : tuple
        Rozmiar figury matplotlib.
    color : str
        Kolor wykresu.
    title : str
        Tytuł główny (opcjonalnie).
    save_path : str
        Ścieżka zapisu do pliku (opcjonalnie).
    """

    required_cols = ['de', 'dn', 'du', 'dtr', 'isb']
    for col in required_cols:
        if col not in sol.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Przycięcie danych
    if cut_init_time > 0:
        sol = sol.iloc[cut_init_time:]

    cols = ['de', 'dn', 'du', 'dtr', 'isb']
    names = ['East [m]', 'North [m]', 'Up [m]', 'Receiver Clock [m]', 'ISB [m]']
    fig, ax = plt.subplots(5, 1, figsize=figsize, sharex=True)

    times = sol.index.to_list()

    for i, (c, name) in enumerate(zip(cols, names)):
        ax[i].plot(times, sol[c], label=name, color=color)
        ax[i].grid()
        ax[i].legend(loc='upper right', fontsize=9)
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        if ct is not None:
            if i <3:
                ax[i].axvline(x=times[0]+timedelta(minutes=int(ct)))
    ax[-1].set_xlabel("Time (UTC)")
    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_uduc_residuals(df: pd.DataFrame,
                   cut_init_time: int = 30,
                   skip_obs: int = 5,
                   obs_type: str = 'phase',
                   x_axis: str = 'time',
                   title: str = None,
                   figsize: tuple = (12, 6),
                   mode='L1L2'):
    
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex with levels ('sv', 'time') or ('time', 'sv')")

    if obs_type not in ['phase', 'code']:
        raise ValueError(f"obs_type must be 'phase' or 'code', got '{obs_type}'")


    if mode not in ['L1L2','E1E5a','E1E5b','L1L5']:
        raise ValueError(f'Unknown mode: {mode}')

    if x_axis not in ['time', 'elev']:
        raise ValueError(f"x_axis must be 'time' or 'elev', got '{x_axis}'")

    all_times = df.index.get_level_values('time').unique().sort_values()

    if len(all_times) <= cut_init_time:
        raise ValueError(f"Not enough time epochs to cut {cut_init_time} initial steps")

    res_trimmed = df[df.index.get_level_values('time').isin(all_times[cut_init_time:])]

    fig, ax = plt.subplots(figsize=figsize,nrows=2,ncols=1)
    seen_prns = set()
    col_map = {('phase',0):'vl1', ('phase',1):'vl2', 
               ('code', 0):'vc1', ('code',1):'vc2',}
    #### automatyczne wykrywanie modu 


    for sv, gr in res_trimmed.groupby('sv'):
        times = gr.index.get_level_values('time')
        if len(times) <= skip_obs:
            continue  # pomiń SV, który ma za mało epok

        first_time_sv = sorted(times)[skip_obs]
        gr_filtered = gr[times > first_time_sv]

        if gr_filtered.empty:
            continue

        prn = sv.split('_')[0]
        label = prn if prn not in seen_prns else None
        seen_prns.add(prn)

        # # Wybór kolumny
        # col = 'vl' if obs_type == 'phase' else 'vc'
        # if col not in gr_filtered.columns:
        #     raise KeyError(f"Column '{col}' not found in DataFrame")

        # Wybór osi X
        if x_axis == 'time':
            x = gr_filtered.index.get_level_values('time')
        else:  # elevation
            if 'ev' not in gr_filtered.columns:
                raise KeyError("Column 'ev' (elevation) not found in DataFrame")
            x = gr_filtered['ev']
        modes = [mode[:2], mode[2:]]
        for i in range(2):
            signal = modes[i]
            col = col_map[(obs_type,i)]

            ax[i].scatter(x, gr_filtered[col].values, label=label, s=2)
            # Styl
            if i == 1:
                if x_axis == 'time':
                    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax[i].set_xlabel("Time")
                else:
                    ax[i].set_xlabel("Elevation [deg]")

            sys = prn[0] if 'prn' in locals() else '?'
            title_map = {'G': 'GPS', 'E': 'Galileo', 'phase': 'Carrier-phase', 'code': 'Pseudorange'}
            if title is None:
                ax[0].set_title(f"{title_map.get(sys, sys)} {title_map[obs_type]} residuals")
            else:
                ax[0].set_title(title)

            ax[i].set_ylabel(f"{signal} Residuals [m]")
            ax[i].grid(True)

    

    # Legenda poza wykresem
    ax[0].legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=2, title='PRN')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # zostawia miejsce na legendę
    plt.show()