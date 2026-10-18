"""Plots for orbit, clock and SIS(RE) comparison results.

These consume the frames produced by the orbits/SIS tooling: a per-satellite
error frame with a ``(sv, time)`` MultiIndex for :func:`plot_sisre`, and a
per-satellite statistics frame indexed by ``sv`` for :func:`plot_sv_stats`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import (
    _check_dpi,
    categorical_palette,
    finalize,
    format_time_axis,
    make_fig_ax,
    no_figure_leak,
)

# Error components, in panel order, with axis labels. The first six sit in a
# two-column block; the three SISE variants span the full width below them,
# because they are the summary the other six explain.
_TOP_COMPONENTS = (
    ("dR", "Radial [m]", (0, 0)),
    ("dA", "Along-track [m]", (0, 1)),
    ("dC", "Cross-track [m]", (1, 0)),
    ("dt", "Clock [m]", (1, 1)),
    ("dt_mean", "Clock avg [m]", (2, 0)),
    ("dTGD", "dTGD [m]", (2, 1)),
)
_BOTTOM_COMPONENTS = (
    ("sisre", "SISE [m]", 0),
    ("sisre_orb", "SISE (ORB) [m]", 1),
    ("sisre_notgd", "SISE (NO TGD) [m]", 2),
)

# The panel whose handles feed the figure-level legend.
_LEGEND_SOURCE = "dt_mean"
# The bottom panel, which is the only one keeping x tick labels.
_X_AXIS_PANEL = "sisre_notgd"

__all__ = [
    "plot_sisre",
    "plot_sv_stats",
]


@no_figure_leak
def plot_sisre(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    xlabel: str = "Time [GPST]",
    figsize: tuple[float, float] = (15, 20),
    save_path: str | Path | None = None,
    close: bool = False,
    dpi: int = 600,
) -> Figure:
    """Plot per-satellite orbit, clock and SISE errors as a nine-panel figure.

    Panels are created only for the components present in ``df``. Every
    satellite gets one colour from
    :func:`~gnx_py.viz.core.categorical_palette`, distinct for up to 60
    satellites and shared across all panels, and a single figure
    level legend; only the bottom panel keeps x tick labels, so the stack reads
    as one time axis. A frame with supported columns but no rows gives the
    empty panel layout without legend entries.

    Parameters
    ----------
    df:
        Error frame with a ``(sv, time)`` MultiIndex. Supported columns are
        ``dR``, ``dA``, ``dC``, ``dt``, ``dt_mean``, ``dTGD``, ``sisre``,
        ``sisre_orb`` and ``sisre_notgd``.
    title:
        Figure title.
    xlabel:
        Label of the bottom panel's time axis. The default names GPS time,
        the scale of SP3 and broadcast-ephemeris epochs; pass another label if
        the epochs were converted.
    figsize:
        Figure size in inches. This plot always creates its own figure, since
        it owns a nine-panel layout.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.
    dpi:
        Resolution used when saving.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the panels.

    Raises
    ------
    ValueError
        When the index is not a MultiIndex, or none of the supported
        components are present.
    KeyError
        When the MultiIndex has no ``sv`` or no ``time`` level.
    """
    # Every check runs before the figure exists, so a bad input leaves none open.
    _check_dpi(dpi)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "SIS error frame must have a MultiIndex with levels ('sv', 'time')."
        )
    missing_levels = [name for name in ("sv", "time") if name not in df.index.names]
    if missing_levels:
        raise KeyError(
            f"SIS error frame has no index level {missing_levels[0]!r}; expected levels "
            f"('sv', 'time'), got {list(df.index.names)}."
        )
    supported = [c for c, _, _ in _TOP_COMPONENTS + _BOTTOM_COMPONENTS]
    if not any(component in df.columns for component in supported):
        raise ValueError(
            f"No supported error columns found. Expected any of: {supported}."
        )

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 2], figure=fig)
    top = grid[:3, :].subgridspec(3, 2, hspace=0.4)
    bottom = grid[3, :].subgridspec(3, 1, hspace=0.6)

    axes: dict[str, Axes] = {}
    labels: dict[str, str] = {}
    for component, label, (row, col) in _TOP_COMPONENTS:
        if component in df.columns:
            axes[component] = fig.add_subplot(top[row, col])
            labels[component] = label
    for component, label, row in _BOTTOM_COMPONENTS:
        if component in df.columns:
            axes[component] = fig.add_subplot(bottom[row, 0])
            labels[component] = label

    svs = df.index.get_level_values("sv").unique()
    # An empty frame has no satellite to colour (categorical_palette needs >= 1).
    colors = dict(zip(svs, categorical_palette(len(svs)))) if len(svs) else {}

    times = df.index.get_level_values("time")
    xmin, xmax = (times.min(), times.max()) if len(times) else (None, None)

    legend_source = _LEGEND_SOURCE if _LEGEND_SOURCE in axes else next(iter(axes))
    for sv, group in df.groupby("sv"):
        group_times = group.index.get_level_values("time")
        for component, ax in axes.items():
            ax.scatter(
                group_times,
                group[component],
                s=1,
                color=colors[sv],
                # Label once, on a single panel, so the figure legend has one
                # entry per satellite rather than nine.
                label=sv if component == legend_source else None,
            )

    for component, ax in axes.items():
        ax.set_ylabel(labels[component])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        if xmin is not None:
            ax.set_xlim(xmin, xmax)
        format_time_axis(ax, autofmt=False)
        ax.yaxis.labelpad = 8

    x_panel = _X_AXIS_PANEL if _X_AXIS_PANEL in axes else list(axes)[-1]
    axes[x_panel].set_xlabel(xlabel)
    for component, ax in axes.items():
        if component != x_panel:
            ax.tick_params(labelbottom=False)

    handles, handle_labels = axes[legend_source].get_legend_handles_labels()
    by_label = dict(zip(handle_labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        title="SV",
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=9,
        title_fontsize=10,
        markerscale=3,
        ncol=1,
        frameon=False,
    )

    if title:
        fig.suptitle(title, fontsize=14)
    return finalize(fig, save_path=save_path, dpi=dpi, close=close)


@no_figure_leak
def plot_sv_stats(
    stats: pd.DataFrame | pd.Series,
    column: str | None = None,
    *,
    sv_list: list[str] | None = None,
    sv_col: str = "sv",
    kind: Literal["bar", "barh", "plot", "scatter"] = "bar",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    save_path: str | Path | None = None,
    close: bool = False,
    **kwargs: object,
) -> Figure:
    """Plot one per-satellite statistic across satellites.

    Parameters
    ----------
    stats:
        Statistics frame, as returned by the orbit statistics helpers. The
        satellite ids may sit in the index or in an ``sv`` column: a summary
        written with ``to_csv(index=False)`` and read back has them in a
        column. A :class:`~pandas.Series` indexed by satellite, such as
        ``frame.groupby(level="sv").size()``, is accepted too.
    column:
        Column to plot, such as ``"sisre_rms_m"``. Required for a frame. For a
        Series it only names the values; None uses the Series name, or
        ``"value"`` when it has none.
    sv_list:
        Satellites to include. None plots all of them. Satellites missing
        from ``stats`` are ignored; when none of them is present, the figure
        is drawn with empty axes rather than raising.
    sv_col:
        Name of the satellite id column, used when the frame carries one.
        Otherwise the index is used. Without this, a summary read from CSV
        would be labelled with row numbers instead of PRNs.
    kind:
        Chart type: ``"bar"``, ``"barh"``, ``"plot"`` or ``"scatter"``.
        ``"barh"`` draws horizontal bars, with the satellites on the y axis
        (the first one at the bottom) and the values on the x axis.
    title:
        Plot title. None uses ``column``.
    xlabel:
        Label of the x axis. None uses ``"SV"``, or ``column`` for
        ``"barh"``.
    ylabel:
        Label of the y axis. None uses ``column``, or ``"SV"`` for
        ``"barh"``.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.
    **kwargs:
        Extra keyword arguments forwarded to the underlying matplotlib call.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the plot.

    Raises
    ------
    KeyError
        When ``column`` is not in ``stats``.
    TypeError
        When ``stats`` is a frame and ``column`` is not given.
    ValueError
        When ``kind`` is not a supported chart type.
    """
    if isinstance(stats, pd.Series):
        if column is None:
            column = "value" if stats.name is None else str(stats.name)
        stats = stats.to_frame(column)
    elif column is None:
        raise TypeError("column is required when stats is a DataFrame.")

    if column not in stats.columns:
        raise KeyError(
            f"Column {column!r} not found in stats. Available: {list(stats.columns)}."
        )
    if kind not in ("bar", "barh", "plot", "scatter"):
        raise ValueError(
            f"kind must be 'bar', 'barh', 'plot' or 'scatter', got {kind!r}."
        )

    svs = stats[sv_col] if sv_col in stats.columns else stats.index.to_series()
    svs = svs.astype(str)

    frame = stats if sv_list is None else stats[svs.isin([str(sv) for sv in sv_list])]
    x = (frame[sv_col] if sv_col in frame.columns else frame.index).astype(str)
    y = frame[column]

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    if kind == "barh":
        ax.barh(x, y, **kwargs)
    elif kind == "bar":
        ax.bar(x, y, **kwargs)
    elif kind == "plot":
        ax.plot(x, y, marker="o", **kwargs)
    else:
        ax.scatter(x, y, **kwargs)

    # Horizontal bars swap the axes: satellites go on y, values on x.
    horizontal = kind == "barh"
    sv_label, value_label = "SV", column
    if horizontal:
        ax.set_xlabel(xlabel if xlabel is not None else value_label)
        ax.set_ylabel(ylabel if ylabel is not None else sv_label)
    else:
        ax.set_xlabel(xlabel if xlabel is not None else sv_label)
        ax.set_ylabel(ylabel if ylabel is not None else value_label)
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(title if title is not None else column)
    ax.grid(True, axis="x" if horizontal else "y", linestyle="--", alpha=0.7)

    if owns_fig:
        fig.tight_layout()
    return finalize(fig, save_path=save_path, close=close)
