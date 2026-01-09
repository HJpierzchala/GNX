import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.dates as mdates
from typing import Union
from pathlib import Path
time_format = mdates.DateFormatter('%H:%M')
import matplotlib as mpl
import seaborn as sns
# Styl publikacyjny
sns.set_context("paper", font_scale=1.3)
sns.set_style("ticks")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from matplotlib.gridspec import GridSpec
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

def plot_sisre(df: pd.DataFrame, out_path: Union[Path, str, None], name: Union[str, None]):
    # Figura z włączonym constrained_layout – lepiej zarządza labelkami
    fig = plt.figure(figsize=(15, 20)) #, constrained_layout=True

    # 4 wiersze, 2 kolumny; ostatni wiersz trochę wyższy (na 3 SISRE pod sobą)
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 2], figure=fig)

    # --- GÓRNA CZĘŚĆ: 2 kolumny × 3 wiersze (6 wykresów) ---
    # trochę większe odstępy w pionie (hspace)
    gs_top = gs[:3, :].subgridspec(3, 2, hspace=0.4)

    # --- DOLNA CZĘŚĆ: 1 kolumna × 3 wiersze (3 wykresy SISRE) ---
    # jeszcze większe odstępy między trzema dolnymi
    gs_bottom = gs[3, :].subgridspec(3, 1, hspace=0.6)

    labels = {
        'dR': 'Radial [m]',
        'dA': 'Along-track [m]',
        'dC': 'Cross-track [m]',
        'dt': 'Clock [m]',
        'dt_mean': 'Clock avg [m]',
        'dTGD': 'dTGD [m]',
        'sisre': 'SISE [m]',
        'sisre_orb': 'SISE (ORB) [m]',
        'sisre_notgd': 'SISE (NO TGD) [m]',
    }

    axes = {}

    # Góra: 2 kolumny × 3 wiersze
    top_layout = [
        ('dR',       (0, 0)),
        ('dA',       (0, 1)),
        ('dC',       (1, 0)),
        ('dt',       (1, 1)),
        ('dt_mean',  (2, 0)),
        ('dTGD',     (2, 1)),
    ]
    for comp, (r, c) in top_layout:
        axes[comp] = fig.add_subplot(gs_top[r, c])

    # Dół: 3 SISRE pod sobą (pełna szerokość)
    bottom_layout = [
        ('sisre',       0),
        ('sisre_orb',   1),
        ('sisre_notgd', 2),
    ]
    for comp, r in bottom_layout:
        axes[comp] = fig.add_subplot(gs_bottom[r, 0])

    # Kolory dla PRN
    prns = df.index.get_level_values('sv').unique()
    palette = sns.color_palette("deep", len(prns))
    color_map = dict(zip(prns, palette))

    # Globalny zakres czasu
    xmin = df.reset_index()['time'].min()
    xmax = df.reset_index()['time'].max()

    # Rysowanie
    for sv, gr in df.groupby('sv'):
        gr = gr.reset_index()
        for comp, ax_comp in axes.items():
            if comp not in gr.columns:
                continue
            ax_comp.scatter(
                gr['time'],
                gr[comp],
                s=1,
                color=color_map[sv],
                label=sv if comp == 'dt_mean' else None  # legenda tylko z dt_mean
            )

    # Formatowanie osi
    for comp, ax_comp in axes.items():
        ax_comp.set_ylabel(labels[comp])
        ax_comp.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax_comp.set_xlim(xmin, xmax)
        ax_comp.xaxis.set_major_formatter(time_format)
        # trochę odsuwamy y-label od osi, żeby nie wchodził na sąsiadów
        ax_comp.yaxis.labelpad = 8

    # Oś X: opis i ticki TYLKO na najniższym SISRE
    axes['sisre_notgd'].set_xlabel("Time [UTC]")
    for comp, ax_comp in axes.items():
        if comp != 'sisre_notgd':
            # ukryj podpisy ticków osi X (dolne trzy + wszystkie górne)
            ax_comp.tick_params(labelbottom=False)

    # Legenda – z panelu 'dt_mean'
    handles, labels_ = axes['dt_mean'].get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        title="SV",
        loc='center left',
        bbox_to_anchor=(0.99, 0.5),
        fontsize=9,
        title_fontsize=10,
        markerscale=3,
        ncol=1,
        frameon=False
    )

    if out_path is not None:
        name = name if name is not None else 'fig'
        plt.savefig(f"{out_path}/{name}_sisre.png", dpi=600)


def plot_sv_stats(
    stats_df,
    column_stat,
    sv_list=None,            # None = wszystkie sv, lub lista np. ['G01', 'G03']
    kind='bar',              # 'bar', 'plot', 'scatter'
    title=None,
    figsize=(10, 5),
    xlabel='SV',
    ylabel=None,
    **kwargs
):
    """
    Wizualizuje statystyki dla sv.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame zwrócony przez calc_sv_stats (index: sv, kolumny: np. 'sisre_mean').
    column_stat : str
        Nazwa kolumny do rysowania (np. 'sisre_mean').
    sv_list : list or None
        Lista sv do wyświetlenia. None = wszystkie.
    kind : str
        Typ wykresu: 'bar', 'plot', 'scatter'.
    title : str
        Tytuł wykresu.
    figsize : tuple
        Rozmiar figury.
    xlabel, ylabel : str
        Opisy osi.
    kwargs : dict
        Dodatkowe parametry przekazywane do matplotlib.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    # Wybranie satelitów
    df = stats_df.copy()
    if sv_list is not None:
        df = df[df.index.isin(sv_list)]

    y = df[column_stat]
    x = df.index.astype(str)
    
    fig, ax = plt.subplots(figsize=figsize)

    if kind == 'bar':
        ax.bar(x, y, **kwargs)
    elif kind == 'plot':
        ax.plot(x, y, marker='o', **kwargs)
    elif kind == 'scatter':
        ax.scatter(x, y, **kwargs)
    else:
        raise ValueError("kind must be 'bar', 'plot' or 'scatter'")
    
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = column_stat
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(column_stat)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return fig, ax

# ---- PRZYKŁAD UŻYCIA ----
# plot_sv_stats(stats_df, column_stat='sisre_mean', kind='bar')
# plot_sv_stats(stats_df, column_stat='sisre_rms', kind='scatter', sv_list=['G01', 'G02', 'G03'])
# plot_sv_stats(stats_df, column_stat='iono_95%', kind='plot')
