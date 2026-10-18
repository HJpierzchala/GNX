"""Non-projected ionosphere plots: STEC arcs, VTEC, activity indexes.

This module covers the ionosphere plots that live on a plain time or scatter
axis. Projected maps of the same quantities (ROTI snapshots, GIX maps, VTEC
model maps) live in :mod:`gnx_py.viz.maps`, which is where Cartopy is used.

Following the convention of :mod:`gnx_py.ionosphere.monitoring`, plotting is
kept separate from metric computation: these functions summarise a frame for
display, but the indexes themselves are computed by the ionosphere package.
"""
from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Iterator, Literal, Mapping, NamedTuple, Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .core import (
    _check_dpi,
    _check_sv_scatter_options,
    _scatter_by_sv,
    constellation_label,
    finalize,
    format_time_axis,
    make_fig_ax,
    make_fig_axes,
    no_figure_leak,
    resolve_time_axis,
    to_utc_naive,
)

# Default STEC series drawn by :func:`plot_stec`. Any of these columns may be
# absent from a given TEC frame; missing ones are skipped.
_DEFAULT_LEFT_SERIES: tuple[dict, ...] = (
    {
        "col": "code_tec",
        "label": "Code STEC",
        "color": "tab:blue",
        "y_label": "CODE STEC [TECU]",
        "y_color": "tab:blue",
    },
    {"col": "leveled_tec", "label": "Phase levelled STEC", "color": "tab:green"},
    {"col": "leveled_tec_sg", "label": "S-G filter smoothed STEC", "color": "orange"},
)
_DEFAULT_RIGHT_SERIES: tuple[dict, ...] = (
    {
        "col": "phase_tec",
        "label": "Phase STEC",
        "color": "tab:red",
        "y_label": "PHASE STEC [TECU]",
        "y_color": "tab:red",
    },
)

# Panels of the GIX family plot: (column, colour, panel title, y label, legend).
_GIX_PANELS = (
    (
        "GIX_mtecu_km",
        "tab:green",
        r"(a) $GIX$",
        r"$GIX$ [mTECU/km]",
        r"$GIX = \sqrt{\bar{g}_x^{\,2} + \bar{g}_y^{\,2}}$",
    ),
    (
        "GIXS_mtecu_km",
        "tab:red",
        r"(b) $GIX_\sigma$",
        r"$GIX_\sigma$ [mTECU/km]",
        r"$GIX_\sigma = \sigma\!\left(|\nabla \mathrm{VTEC}|\right)$",
    ),
    (
        "GIXP95_mtecu_km",
        "tab:blue",
        r"(c) $GIX_{95}$",
        r"$GIX_{95}$ [mTECU/km]",
        r"$GIX_{95} = Q_{95}\!\left(|\nabla \mathrm{VTEC}|\right)$",
    ),
)

_STATS_ORDER = ["Model", "N", "Bias", "STD", "RMS", "MAE", "Corr", "R2", "Slope", "Intercept"]
_ROUNDED_STATS = ["Bias", "STD", "RMS", "MAE", "Corr", "R2", "Slope", "Intercept"]

# Default title stems of :func:`plot_vtec_error_series`, keyed by the upper-cased
# metric name. A metric missing here is titled "<metric> of the VTEC difference".
_ERROR_METRIC_TITLES = {
    "MAE": "Mean absolute VTEC difference",
    "MEDAE": "Median absolute VTEC difference",
    "RMS": "RMS VTEC difference",
    "RMSE": "RMS VTEC difference",
    "BIAS": "Mean VTEC difference",
    "STD": "Standard deviation of the VTEC difference",
}

# Units of the ROTI columns written by ``compute_roti_links``, keyed by the
# column-name suffix; used when :func:`plot_roti_series` gets no ``units``.
_ROTI_UNIT_SUFFIXES = (
    ("_mtecu_per_s", "mTECU/s"),
    ("_tecu_per_min", "TECU/min"),
)

__all__ = [
    "STECComparisonReport",
    "STECModelComparator",
    "plot_gix_family",
    "plot_point_timeseries",
    "plot_roti_series",
    "plot_sidx_series",
    "plot_stec",
    "plot_stec_arcs",
    "plot_vtec_error_series",
    "plot_vtec_zenith",
]


def _draw_stec_series(
    ax: Axes,
    data: pd.DataFrame,
    times: Sequence,
    series: Sequence[Mapping],
    colors: Iterator,
) -> tuple[list, list, str | None, str | None]:
    """Scatter one axis' worth of STEC series; skip columns that are absent.

    A series without a ``"color"`` takes the next colour from ``colors``, which
    the caller shares between the left and the twin axis so the two never
    repeat a colour.
    """
    handles: list = []
    labels: list = []
    y_label: str | None = None
    y_color: str | None = None

    for spec in series:
        column = spec["col"]
        if column not in data.columns:
            continue

        label = spec.get("label", column)
        color = spec.get("color")
        if color is None:
            color = next(colors)
        handle = ax.scatter(
            times,
            data[column],
            label=label,
            color=color,
            marker=spec.get("marker", "o"),
            alpha=spec.get("alpha", 0.7),
            edgecolors=spec.get("edgecolor", "k"),
            s=spec.get("s", 10),
        )
        handles.append(handle)
        labels.append(label)

        # The first drawn series names and colours the axis.
        if y_label is None:
            y_label = spec.get("y_label", f"{label} [TECU]")
            y_color = spec.get("y_color") or color

    return handles, labels, y_label, y_color


def _check_stec_series(name: str, series: object) -> list:
    """Return the series specs of :func:`plot_stec` as a list, or raise.

    Every spec must be dict-like (it supports ``spec["col"]`` and
    ``spec.get(...)``, as a dict or a ``pd.Series`` does) with a hashable
    ``"col"`` entry; the check runs before any figure exists.
    """
    if isinstance(series, (str, collections.abc.Mapping)) or not isinstance(
        series, collections.abc.Iterable
    ):
        raise TypeError(
            f"{name} must be a sequence of mappings with a 'col' key "
            f"(got {type(series).__name__})"
        )
    specs = list(series)
    for i, spec in enumerate(specs):
        # Duck-typed like _draw_stec_series, which only uses spec["col"] and
        # spec.get(...): a str has neither .get nor a 'col' entry.
        if isinstance(spec, (str, bytes)) or not (
            hasattr(spec, "get") and hasattr(spec, "__getitem__")
        ):
            raise TypeError(
                f"{name}[{i}] must be a mapping with a 'col' key (got {type(spec).__name__})"
            )
        if "col" not in spec:
            raise KeyError(f"{name}[{i}] has no 'col' key naming the column to draw: {spec!r}")
        column = spec["col"]
        try:
            hash(column)
        except TypeError:
            raise TypeError(
                f"{name}[{i}]['col'] must be a single column name "
                f"(got {type(column).__name__} {column!r})"
            ) from None
    return specs


def _stec_time_label(time_col: str) -> str:
    """Default x label of a STEC plot for the epochs held in ``time_col``.

    ``TECSession`` keeps the RINEX observation epochs, which are GPS time: the
    ionosphere models convert them with ``gpst2utc`` before use, and the
    session's ``LT`` column is those epochs shifted by the station longitude.
    """
    if time_col == "LT":
        return "Local Time [LT]"
    if time_col == "time":
        return "Time [GPST]"
    return "Time"


@no_figure_leak
def plot_stec(
    obs_tec: pd.DataFrame,
    prn: str,
    *,
    time_col: str = "time",
    left_series: Sequence[Mapping] | None = None,
    right_series: Sequence[Mapping] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    grid: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot the STEC arc of one satellite, with code and phase on twin axes.

    Code and phase STEC differ by an ambiguity offset of a size that would flatten
    both if drawn on one scale, so phase goes on a right-hand axis by default.

    The values are plotted as given and the axes are labelled TECU, so convert
    first if the frame holds raw electron counts: ``TECSession`` reports STEC in
    el/m², and 1 TECU is 1e16 el/m².

    Parameters
    ----------
    obs_tec:
        TEC frame indexed such that ``obs_tec.loc[prn]`` selects one satellite,
        with its epochs in a column or index level named by ``time_col``. The
        raw ``TECSession`` output, with an ``(sv, time)`` MultiIndex, works
        as is.
    prn:
        Key selecting one arc. ``TECSession`` splits satellites into arcs and
        names them ``"<PRN>_<arc>_<band>"``, such as ``"G02_0_1"``, so this is
        an arc key rather than a bare PRN.
    time_col:
        Name of the epoch column or index level, located with
        :func:`~gnx_py.viz.core.resolve_time_axis`. ``TECSession`` names its
        epoch level ``time`` and adds a local-time column named ``LT``.
    left_series:
        Series drawn on the left axis, as mappings with a required ``"col"``
        key and optional ``"label"``, ``"color"``, ``"marker"``, ``"s"``,
        ``"alpha"``, ``"edgecolor"``, ``"y_label"`` and ``"y_color"`` keys. The
        first drawn series names and colours the axis. A series without a
        ``"color"`` takes the next colour of the ``axes.prop_cycle``, counted
        across both axes so no two uncoloured series share one. None uses the
        code and levelled STEC defaults. Series whose column is absent are
        skipped.
    right_series:
        Series drawn on a twin right axis, same format. None uses the phase
        STEC default; an empty sequence suppresses the right axis.
    title:
        Plot title. None derives it from ``prn``.
    xlabel:
        X-axis label. None derives it from ``time_col``: ``"Local Time [LT]"``
        for ``LT``, ``"Time [GPST]"`` for the session's ``time`` epochs
        (RINEX observation epochs, which are GPS time), and ``"Time"``
        otherwise.
    grid:
        Draw a grid on the left axis.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the plot.

    Raises
    ------
    KeyError
        When ``time_col`` is found neither as a column nor as an index level,
        a series has no ``"col"`` key, or none of the requested series
        columns are present in the frame (also when both ``left_series`` and
        ``right_series`` are empty). All of these are raised before a figure
        is created.
    TypeError
        When the ``time_col`` values cannot be read as datetimes, or a series
        spec is not a mapping.
    """
    left = _check_stec_series(
        "left_series", _DEFAULT_LEFT_SERIES if left_series is None else left_series
    )
    right = _check_stec_series(
        "right_series", _DEFAULT_RIGHT_SERIES if right_series is None else right_series
    )

    data = obs_tec.loc[prn]
    times = resolve_time_axis(data, time_col)
    # "Nothing to draw" is known before the figure exists, so it leaves none behind (VIZ-005).
    requested = [spec["col"] for spec in left + right]
    if not any(column in data.columns for column in requested):
        raise KeyError(
            "None of the requested STEC columns are present in the frame. "
            f"Requested: {requested}. Available: {list(data.columns)}."
        )

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    # One colour source for both axes, so uncoloured series never repeat.
    colors = cycle(mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["k"]))
    handles, labels, y_label, y_color = _draw_stec_series(ax, data, times, left, colors)
    if y_label is not None:
        ax.set_ylabel(y_label, color=y_color)
        ax.tick_params(axis="y", labelcolor=y_color)
        ax.spines["left"].set_color(y_color)

    # Only claim a twin axis once something is actually going on it: the
    # default right-hand series is optional, and an empty twin axis draws a
    # second y scale that means nothing.
    if any(spec["col"] in data.columns for spec in right):
        right_ax = ax.twinx()
        r_handles, r_labels, r_y_label, r_y_color = _draw_stec_series(
            right_ax, data, times, right, colors
        )
        if r_y_label is not None:
            right_ax.set_ylabel(r_y_label, color=r_y_color)
            right_ax.tick_params(axis="y", labelcolor=r_y_color)
            right_ax.spines["right"].set_color(r_y_color)
        handles += r_handles
        labels += r_labels

    ax.set_xlabel(xlabel if xlabel is not None else _stec_time_label(time_col))
    format_time_axis(ax, hour_interval=1, autofmt=owns_fig)
    if grid:
        ax.grid(True, which="major", linestyle="--", alpha=0.6)

    ax.set_title(
        title if title is not None else f"Slant TEC  PRN: {str(prn)[:3]}",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    ax.legend(handles, labels, loc="best", fontsize=8, frameon=True)

    if owns_fig:
        fig.tight_layout()
    return finalize(fig, save_path=save_path, close=close)


@no_figure_leak
def plot_stec_arcs(
    obs_tec: pd.DataFrame,
    *,
    value_col: str = "leveled_tec",
    time_col: str = "time",
    group_by: Literal["prn", "arc"] = "prn",
    s: float = 2,
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "STEC [TECU]",
    legend: bool = True,
    legend_ncol: int = 2,
    legend_fontsize: float = 7,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot the STEC of every satellite on one axes, one colour per PRN.

    This is the multi-satellite counterpart of :func:`plot_stec`, which draws
    one arc in detail. A satellite tracked in several arcs keeps one colour
    and one legend entry. Lay a zenith VTEC line over it by passing the
    returned figure's axes to :func:`plot_vtec_zenith`.

    As with :func:`plot_stec`, the values are plotted as given, so convert
    raw ``TECSession`` output (el/m²) to TECU first.

    Parameters
    ----------
    obs_tec:
        TEC frame with the satellite keys in an ``sv`` index level or column,
        such as the ``(sv, time)`` MultiIndex of ``TECSession`` output. Keys
        may be arc keys such as ``"G01_0"``; the PRN is the part before the
        first ``"_"``.
    value_col:
        Column plotted on the y axis.
    time_col:
        Name of the epoch column or index level, located with
        :func:`~gnx_py.viz.core.resolve_time_axis`; ``"LT"`` plots against
        the session's local-time column.
    group_by:
        ``"prn"`` merges the arcs of each PRN into one series; ``"arc"``
        draws each arc separately. The colours and the legend are per PRN
        either way.
    s:
        Marker size, in points squared.
    title:
        Plot title. None derives it from the constellations drawn.
    xlabel:
        X-axis label.
    ylabel:
        Y-axis label.
    legend:
        Draw a legend with one entry per PRN.
    legend_ncol:
        Number of legend columns.
    legend_fontsize:
        Legend font size.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the plot.

    Raises
    ------
    KeyError
        When ``value_col`` or the satellite keys are missing, or the epochs
        cannot be located.
    ValueError
        When ``value_col`` holds no values to draw, or ``group_by`` is not
        ``"prn"`` or ``"arc"``.
    """
    # Before any figure: an unknown group_by used to fall back to "prn" silently (VIZ-019).
    _check_sv_scatter_options(group_by, "prn", "palette")
    if value_col not in obs_tec.columns:
        raise KeyError(
            f"Column {value_col!r} not found. Available: {list(obs_tec.columns)}."
        )
    if obs_tec[value_col].notna().sum() == 0:
        raise ValueError(f"Column {value_col!r} holds no values to plot.")
    # The satellite keys and the epochs are located before a figure exists, so
    # a typo in time_col does not leave an empty figure behind.
    if "sv" not in obs_tec.columns and "sv" not in obs_tec.index.names:
        raise KeyError("No column or index level named 'sv' holds the satellites.")
    resolve_time_axis(obs_tec, time_col)

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    prns = _scatter_by_sv(
        ax,
        obs_tec,
        value_col,
        time_col=time_col,
        group_by=group_by,
        label_by="prn",
        colors="palette",
        s=s,
    )
    ax.set_title(title if title is not None else f"{constellation_label(prns)} STEC")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)
    format_time_axis(ax, autofmt=False)
    if legend:
        ax.legend(loc="best", ncol=legend_ncol, fontsize=legend_fontsize)

    if owns_fig:
        fig.tight_layout()
    return finalize(fig, save_path=save_path, close=close)


@no_figure_leak
def plot_vtec_zenith(
    zen: pd.DataFrame,
    *,
    time_col: str = "LT",
    use_smoothed: bool = True,
    color: str | None = None,
    lw: float = 1.5,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "VTEC [TECU]",
    legend: bool | None = None,
    legend_ncol: int = 1,
    legend_fontsize: float | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 4),
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot the VTEC time series above a station.

    The line can be laid over another plot through ``ax``, such as the STEC
    arcs of :func:`plot_stec_arcs`; the legend then collects every labelled
    artist on the axes, not just this line.

    Parameters
    ----------
    zen:
        Zenith VTEC frame with a datetime column and a ``vtec`` column, and
        optionally a ``vtec_smooth`` column.
    time_col:
        Name of the datetime column.
    use_smoothed:
        Prefer the ``vtec_smooth`` column when it is present.
    color:
        Line colour. None takes the next colour of the axes' property cycle.
    lw:
        Line width, in points.
    label:
        Legend label of the line. None leaves it unlabelled.
    title:
        Plot title. None uses a generic zenith-VTEC title.
    xlabel:
        X-axis label.
    ylabel:
        Y-axis label.
    legend:
        Draw a legend of every labelled artist on the axes. None draws one
        only when ``label`` is given.
    legend_ncol:
        Number of legend columns.
    legend_fontsize:
        Legend font size. None uses ``legend.fontsize`` from the rcParams.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the plot.

    Raises
    ------
    KeyError
        When the time or VTEC column is missing.
    TypeError
        When the time column holds numbers (local time in hours, say) or
        values that cannot be read as datetimes; strings are converted.
    """
    if time_col not in zen.columns:
        raise KeyError(
            f"Time column {time_col!r} not found. Available: {list(zen.columns)}."
        )

    column = "vtec_smooth" if (use_smoothed and "vtec_smooth" in zen.columns) else "vtec"
    if column not in zen.columns:
        raise KeyError(f"Column {column!r} not found. Available: {list(zen.columns)}.")
    # numbers raise TypeError and strings become datetimes before any figure exists (VIZ-025)
    times = to_utc_naive(resolve_time_axis(zen, time_col))

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    line_kwargs: dict = {"lw": lw}
    if color is not None:
        line_kwargs["color"] = color
    if label is not None:
        line_kwargs["label"] = label
    ax.plot(times, zen[column].to_numpy(), **line_kwargs)
    ax.set_title(title if title is not None else "Zenith VTEC above the station")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls=":", alpha=0.6)
    if legend if legend is not None else label is not None:
        # No explicit handles: artists already on a shared axes, such as the
        # STEC scatter of plot_stec_arcs, join the same legend.
        ax.legend(loc="best", ncol=legend_ncol, fontsize=legend_fontsize)

    if owns_fig:
        fig.tight_layout()
    return finalize(fig, save_path=save_path, close=close)


def _label_time_scale(time_scale, frame, epochs) -> str:
    """Time-scale name for an x label: the explicit argument, else ``frame.attrs["time_scale"]``
    (monitoring outputs carry "GPST"), else "UTC" for timezone-aware epochs (``to_utc_naive``
    plots them as naive UTC) and "GPST" for naive ones."""
    if time_scale is not None:
        return str(time_scale)
    tag = getattr(frame, "attrs", {}).get("time_scale")
    if tag:
        return str(tag)
    return "UTC" if getattr(pd.DatetimeIndex(epochs), "tz", None) is not None else "GPST"


@no_figure_leak
def plot_roti_series(
    roti: pd.DataFrame,
    *,
    value_col: str = "ROTI_tecu_per_min",
    time_col: str = "time",
    quantile: float = 0.95,
    units: str | None = None,
    title: str | None = None,
    time_scale: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot mean and high-quantile ROTI over time.

    ROTI is computed per satellite-station link, so a network produces many
    values per epoch. This collapses them to two lines: the mean, and an upper
    quantile that exposes localised disturbances the mean hides.

    Parameters
    ----------
    roti:
        Per-link ROTI frame holding ``time_col`` and ``value_col``. Already
        aggregated frames, indexed by time, are accepted too.
    value_col:
        Name of the ROTI column.
    time_col:
        Name of the epoch column or index level, located with
        :func:`~gnx_py.viz.core.resolve_time_axis`; a frame without either is
        read from its DatetimeIndex.
    quantile:
        Upper quantile drawn alongside the mean, as a fraction.
    units:
        Units of ``value_col``, shown in the y label. None derives them from
        the column-name suffix written by ``compute_roti_links``
        (``_tecu_per_min`` or ``_mtecu_per_s``) and leaves them out when the
        suffix is not recognised.
    title:
        Plot title. None derives it from ``quantile``.
    time_scale:
        Time scale of the epochs, named in the x label. None (default) takes
        ``roti.attrs["time_scale"]`` (``"GPST"`` for ``compute_roti_links``
        output), else ``"UTC"`` for timezone-aware epochs (they are plotted
        as naive UTC) and ``"GPST"`` for naive ones.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the plot.

    Raises
    ------
    KeyError
        When ``value_col`` is missing, or the epochs cannot be located.
    TypeError
        When the epochs found cannot be read as datetimes.
    ValueError
        When ``quantile`` is outside ``(0, 1)``.
    """
    if value_col not in roti.columns:
        raise KeyError(
            f"Column {value_col!r} not found. Available: {list(roti.columns)}."
        )
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}.")

    epochs = resolve_time_axis(roti, time_col)
    time_scale = _label_time_scale(time_scale, roti, epochs)
    summary = (
        roti.groupby(pd.Index(epochs))[value_col]
        .agg(mean="mean", upper=lambda s: s.quantile(quantile))
        .sort_index()
    )
    times = to_utc_naive(summary.index)

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    ax.plot(times, summary["mean"].to_numpy(), color="tab:blue", lw=1.2, alpha=0.9)
    ax.plot(times, summary["upper"].to_numpy(), color="tab:red", lw=1.2, alpha=0.9)
    if units is None:
        units = next(
            (unit for suffix, unit in _ROTI_UNIT_SUFFIXES if value_col.endswith(suffix)),
            None,
        )
    percent = f"{quantile * 100:g}%"
    ax.set_ylabel(f"ROTI [{units}]" if units else "ROTI")
    ax.set_xlabel(f"Time [{time_scale}]")
    ax.set_title(title if title is not None else f"Mean and {percent} ROTI over time")
    ax.grid(True)
    ax.legend(
        ["Mean ROTI", f"{percent} quantile ROTI"],
        loc="best",
        fontsize=10,
        frameon=True,
    )
    format_time_axis(ax, autofmt=owns_fig)
    return finalize(fig, save_path=save_path, close=close)


@no_figure_leak
def plot_sidx_series(
    sidx: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    title: str | None = None,
    time_scale: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot the Sudden Ionospheric Disturbance Index (SIDX) time series.

    SIDX is a TEC-rate index: per epoch, the median (by default) over
    receiver-satellite links of the slant-TEC rate divided by the mapping
    function, in TECU per minute (see
    :func:`gnx_py.ionosphere.monitoring.compute_sid`).

    Parameters
    ----------
    sidx:
        SIDX values indexed by time, as a Series or a single-column frame.
        The epochs are read from a DatetimeIndex or an index level named
        ``time``. Missing values are dropped.
    value_col:
        Column to plot when ``sidx`` is a frame. None uses its only column.
    title:
        Plot title.
    time_scale:
        Time scale of the epochs, named in the x label. None (default) takes
        ``sidx.attrs["time_scale"]`` (``"GPST"`` for ``compute_sid``
        output), else ``"UTC"`` for timezone-aware epochs (they are plotted
        as naive UTC) and ``"GPST"`` for naive ones.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the plot.

    Raises
    ------
    KeyError
        When ``value_col`` is not in ``sidx``, or the index holds no epochs
        (a RangeIndex, say), which would otherwise plot as 1970.
    TypeError
        When the index values cannot be read as datetimes.
    ValueError
        When ``sidx`` is a frame with no columns, or with several columns and
        ``value_col`` is None.
    """
    if isinstance(sidx, pd.DataFrame):
        if sidx.shape[1] == 0:
            raise ValueError("sidx is an empty frame (no columns): nothing to plot.")
        if value_col is not None:
            if value_col not in sidx.columns:
                raise KeyError(
                    f"Column {value_col!r} not found. Available: {list(sidx.columns)}."
                )
            series = sidx[value_col]
        elif sidx.shape[1] == 1:
            series = sidx.iloc[:, 0]
        else:
            raise ValueError(
                "sidx has several columns; pass value_col to pick one. "
                f"Available: {list(sidx.columns)}."
            )
    else:
        series = sidx

    series = series.dropna()
    # Resolve on an index-only frame, so a column that happens to be named
    # "time" in a multi-column input cannot stand in for the index.
    raw_times = resolve_time_axis(pd.DataFrame(index=series.index), "time")
    time_scale = _label_time_scale(time_scale, sidx, raw_times)
    times = to_utc_naive(raw_times)

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    ax.plot(
        times,
        series.to_numpy(),
        color="tab:red",
        alpha=0.9,
        lw=1.2,
        marker=".",
        markersize=3,
    )
    ax.set_ylabel("SIDX [TECU/min]")
    ax.set_xlabel(f"Time [{time_scale}]")
    ax.set_title(title if title is not None else "SIDX over time")
    ax.grid(True)
    format_time_axis(ax, autofmt=owns_fig)
    return finalize(fig, save_path=save_path, close=close)


def _format_bound(value: float, positive: str, negative: str) -> str:
    """Format one latitude or longitude bound, such as ``57.5°N``."""
    hemisphere = positive if value >= 0 else negative
    return f"{abs(float(value)):g}°{hemisphere}"


def _region_label(reg: object) -> str:
    """Name a region readably: its name, its bounds, or both.

    A string is returned as is. An object with ``lat_min``/``lat_max``/
    ``lon_min``/``lon_max`` attributes, such as
    :class:`gnx_py.ionosphere.Region`, is described by its bounds, after its
    ``name`` when it has one. Anything else falls back to ``str``.
    """
    if isinstance(reg, str):
        return reg
    name = getattr(reg, "name", None)
    try:
        lat = [_format_bound(value, "N", "S") for value in (reg.lat_min, reg.lat_max)]
        lon = [_format_bound(value, "E", "W") for value in (reg.lon_min, reg.lon_max)]
    except (AttributeError, TypeError, ValueError):
        return str(name) if name else str(reg)
    bounds = f"{lat[0]}–{lat[1]}, {lon[0]}–{lon[1]}"
    return f"{name} ({bounds})" if name else bounds


@no_figure_leak
def plot_gix_family(
    times: pd.DataFrame,
    reg: object,
    dmin: float,
    dmax: float,
    *,
    subtitle: str | None = None,
    time_scale: str | None = None,
    figsize: tuple[float, float] = (12, 9),
    dpi: int = 100,
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot the GIX family as three stacked panels.

    The panels are the gradient index itself, its standard deviation, and its
    95th percentile:

    * (a) ``GIX`` = sqrt(gx_bar^2 + gy_bar^2)
    * (b) ``GIX_sigma`` = std(|grad(VTEC)|)
    * (c) ``GIX_95`` = Q95(|grad(VTEC)|)

    Parameters
    ----------
    times:
        Frame holding ``time``, ``GIX_mtecu_km``, ``GIXS_mtecu_km`` and
        ``GIXP95_mtecu_km``.
    reg:
        Region shown in the title: a name, or a region object such as
        :class:`gnx_py.ionosphere.Region`, which is shown by its ``name``
        when it has one and by its latitude/longitude bounds.
    dmin:
        Lower bound of the dipole length range, in km.
    dmax:
        Upper bound of the dipole length range, in km.
    subtitle:
        Extra title line. None derives the analysed period from ``times``.
    time_scale:
        Time scale of the epochs, named in the period line and the x label.
        None (default) takes ``times.attrs["time_scale"]`` (``"GPST"`` for
        ``compute_gix`` output), else ``"UTC"`` for timezone-aware epochs
        (they are plotted as naive UTC) and ``"GPST"`` for naive ones, as in
        :func:`plot_roti_series` and :func:`plot_sidx_series`.
    figsize:
        Figure size in inches.
    dpi:
        Figure resolution.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the three panels.

    Raises
    ------
    KeyError
        When ``time`` or one of the GIX columns is missing.
    TypeError
        When the epochs are numbers or cannot be read as datetimes; strings
        are converted.
    """
    if "time" not in times.columns:
        raise KeyError(f"Column 'time' not found. Available: {list(times.columns)}.")
    missing = [column for column, *_ in _GIX_PANELS if column not in times.columns]
    if missing:
        raise KeyError(
            f"GIX columns not found: {missing}. Available: {list(times.columns)}."
        )

    # numbers raise TypeError and strings become datetimes before any figure exists (VIZ-025)
    raw_t = resolve_time_axis(times, "time")
    time_scale = _label_time_scale(time_scale, times, raw_t)
    t = to_utc_naive(raw_t)
    fig, axes = make_fig_axes(3, figsize=figsize, dpi=dpi)

    if subtitle is None:
        subtitle = f"Period: {t.min():%Y-%m-%d %H:%M}–{t.max():%H:%M} {time_scale}"
    fig.suptitle(
        f"GIX family for {_region_label(reg)}\n{subtitle}\n"
        f"Dipole length range: {dmin}-{dmax} km",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (column, color, panel_title, y_label, legend) in zip(axes, _GIX_PANELS):
        values = times[column].to_numpy()
        ax.plot(t, values, lw=0.7, color=color, alpha=0.9)
        ax.scatter(t, values, s=4, color=color, alpha=0.4, edgecolor="none")
        ax.set_title(panel_title, loc="left", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.legend([legend], loc="best", fontsize=10, frameon=True)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
        ax.tick_params(axis="both", labelsize=10)

    axes[-1].set_xlabel(f"Time [{time_scale}]", fontsize=11)
    format_time_axis(axes[-1], autofmt=False)
    return finalize(fig, save_path=save_path, dpi=dpi, close=close)


@no_figure_leak
def plot_point_timeseries(
    ds: object,
    var: str,
    lat: float,
    lon: float,
    *,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 4),
    save_path: str | Path | None = None,
    close: bool = False,
    dpi: int = 150,
) -> Figure:
    """Plot a gridded variable over time at the grid point nearest a location.

    Parameters
    ----------
    ds:
        :class:`xarray.Dataset` with ``time``, ``lat`` and ``lon`` coordinates.
    var:
        Name of the variable to plot.
    lat:
        Latitude of interest, in degrees.
    lon:
        Longitude of interest, in degrees.
    title:
        Plot title. None reports the variable and the grid point actually used.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
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
        The figure holding the plot.

    Raises
    ------
    KeyError
        When ``var`` is not a variable of ``ds``.
    """
    _check_dpi(dpi)  # before the figure exists, so a bad dpi leaves none open
    if var not in ds.data_vars:
        raise KeyError(
            f"Variable {var!r} not found. Available: {list(ds.data_vars)}."
        )

    point = ds[var].sel(lat=lat, lon=lon, method="nearest")
    series = point.to_series()

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    ax.plot(series.index, series.to_numpy())
    ax.set_xlabel("Time")
    ax.set_ylabel(var)
    if title is None:
        title = (
            f"{var} @ ~({float(point['lat']):.2f}°, {float(point['lon']):.2f}°)"
        )
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.6)

    if owns_fig:
        fig.tight_layout()
    return finalize(fig, save_path=save_path, dpi=dpi, close=close)


@no_figure_leak
def plot_vtec_error_series(
    errors: pd.DataFrame | Mapping[str, Sequence[float] | pd.Series],
    *,
    reference: str = "GIM",
    metric: str = "MAE",
    units: str = "TECU",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_mean_in_title: bool = True,
    marker: str = "o",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot a per-map VTEC error summary of several models against a reference.

    Each model is one line of per-map values, such as the mean absolute
    difference that :func:`~gnx_py.viz.maps.compare_vtec_maps` returns for
    each epoch, so models can be compared over the whole run at a glance.

    Parameters
    ----------
    errors:
        One column per model, named by the model, and one row per compared
        map; or a mapping from model name to its values. With a
        DatetimeIndex the x axis is time: timezone-aware epochs are plotted
        as naive UTC, naive ones as given. Any other index, such as a map
        counter, is plotted by value.
    reference:
        Name of the reference the models were compared against.
    metric:
        Name of the error metric, used in the legend, the y label and the
        default title.
    units:
        Units of the values.
    title:
        First title line. None derives it from ``metric`` and ``reference``,
        such as "Mean absolute VTEC difference against GIM".
    xlabel:
        X-axis label. None names the time scale for a DatetimeIndex, as in
        :func:`plot_roti_series`: ``"Time [<scale>]"`` with the scale taken
        from ``errors.attrs["time_scale"]`` when ``errors`` is a frame that
        carries it, else ``"UTC"`` for timezone-aware epochs and ``"GPST"``
        for naive ones (GNX grid epochs, such as kriging output, are naive
        GPST). ``attrs["time_scale"]`` wins even for timezone-aware epochs,
        which are still plotted as naive UTC: set it only when it names the
        scale of the plotted values. For any other index it is the index
        name, else ``"Map number"``.
    ylabel:
        Y-axis label. None uses ``"<metric> [<units>]"``.
    show_mean_in_title:
        Add a second title line with each model's mean over all maps. Set
        it to False to use ``title`` exactly as given.
    marker:
        Marker drawn at each map.
    ax:
        Axes to draw on. When None, a new figure is created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the plot.

    Raises
    ------
    ValueError
        When the mapping's values differ in length or, for Series, in index,
        or no model has a finite value (all-NaN models are skipped).

    Examples
    --------
    >>> errors = pd.DataFrame({"Kriging": mns, "NTCM": mns_ntcm})  # doctest: +SKIP
    >>> fig = plot_vtec_error_series(errors, reference="GIM", metric="MAE")  # doctest: +SKIP
    """
    # attrs (time_scale) are read from the caller's frame: the numeric
    # conversion below does not carry them over.
    label_source = errors if isinstance(errors, pd.DataFrame) else None
    if isinstance(errors, pd.DataFrame):
        frame = errors.copy()
    else:
        errors = dict(errors)
        try:
            frame = pd.DataFrame(errors)
        except ValueError as exc:
            raise ValueError(
                f"Every model in errors must have one value per map: {exc}"
            ) from exc
        # Series align on their index, so a shorter or shifted Series would be
        # NaN-padded silently instead of failing like a list does.
        lengths = {len(values) for values in errors.values() if np.ndim(values) > 0}
        if len(lengths) > 1 or any(len(frame) != n for n in lengths):
            raise ValueError(
                "Every model in errors must have one value per map, on the same index."
            )
    frame = frame.apply(pd.to_numeric).dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError("errors holds no model with a finite value to plot.")

    is_time = isinstance(frame.index, pd.DatetimeIndex)
    # Only a real DatetimeIndex goes through to_utc_naive: a map counter
    # converted to datetimes would plot as 1970.
    x = to_utc_naive(frame.index) if is_time else frame.index.to_numpy()
    if xlabel is None:
        if is_time:
            # naive epochs are GPST, aware ones are plotted as UTC (VIZ-042, Q100)
            xlabel = f"Time [{_label_time_scale(None, label_source, frame.index)}]"
        else:
            xlabel = frame.index.name or "Map number"

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        ax.plot(x, values, marker=marker, label=f"{metric} for {column}")

    if title is None:
        stem = _ERROR_METRIC_TITLES.get(metric.upper(), f"{metric} of the VTEC difference")
        title = f"{stem} against {reference}"
    if show_mean_in_title:
        means = " | ".join(
            f"{column}: {frame[column].mean():.3f} {units}" for column in frame.columns
        )
        title = f"{title}\n{means}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else f"{metric} [{units}]")
    ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.legend()
    if is_time:
        format_time_axis(ax, autofmt=False)
    elif pd.api.types.is_integer_dtype(frame.index.dtype):
        # A map counter has no half maps.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if owns_fig:
        fig.tight_layout()
    return finalize(fig, save_path=save_path, close=close)


def _as_name_list(name: str, value: object) -> list:
    """Return ``value`` as a list of names; a single string becomes a one-element list.

    The library convention for column arguments: ``"ion"`` means
    ``["ion"]``, not the characters ``"i"``, ``"o"``, ``"n"``.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, collections.abc.Mapping) or not isinstance(
        value, collections.abc.Iterable
    ):
        raise TypeError(
            f"{name} must be a name or a sequence of names (got {type(value).__name__})"
        )
    return list(value)


class STECComparisonReport(NamedTuple):
    """Result of :meth:`STECModelComparator.report`.

    Attributes
    ----------
    stats:
        The statistics from :meth:`STECModelComparator.compute_stats`.
    hexbin:
        The hexbin figure, or None when it was not drawn.
    hist:
        The error-histogram figure, or None when it was not drawn.
    """

    stats: pd.DataFrame
    hexbin: Figure | None
    hist: Figure | None


@dataclass
class STECModelComparator:
    """Compare modelled STEC against measured STEC.

    Computes the agreement statistics (bias, RMS, STD, MAE, correlation, R²,
    and a linear fit) and draws the two figures that make them legible: a
    hexbin of modelled against measured values, and a histogram of the errors.

    Parameters
    ----------
    df:
        Frame holding the measured column and one column per model.
    x_col:
        Name of the measured STEC column.
    model_cols:
        Names of the model columns; every one must be in ``df``, and at least
        one is required. A single name given as a string is read as a
        one-element list. The default
        is the background model 'ion' and the Klobuchar and NTCM columns of
        ``compare_models``; the BDGIM column ('bdgim') is
        not in the default, so a frame without it keeps working: pass e.g.
        ``["ion", "klobuchar", "ntcm", "bdgim"]`` to compare it.
    model_names:
        Labels for the models, in the order of ``model_cols`` and as many as
        there are model columns; a single string is read as a one-element
        list. None uses the column names.
    scale:
        Factor applied to the model values before comparison, for models whose
        output is in different units.
    gridsize:
        Hexbin resolution.
    mincnt:
        Minimum number of points in a hexagon for it to be drawn.

    Raises
    ------
    TypeError
        When ``df`` is not a DataFrame, or ``model_cols`` / ``model_names``
        is neither a name nor a sequence of names.
    ValueError
        When ``model_cols`` is empty, or ``model_names`` does not hold one
        label per model column.
    KeyError
        When ``x_col`` or a model column is not in ``df``.

    Notes
    -----
    Construction normalises ``model_cols`` and ``model_names`` to lists (a
    tuple, array or index becomes a list, ``"ion"`` becomes ``["ion"]``), so
    the attributes hold what is drawn.

    The same checks run again at the start of :meth:`compute_stats`,
    :meth:`plot_hexbin`, :meth:`plot_error_hist` and :meth:`report`, before
    any figure is created, so an attribute changed after construction cannot
    leave a half-drawn figure behind.
    """

    df: pd.DataFrame
    x_col: str = "leveled_tec"
    model_cols: list[str] = field(default_factory=lambda: ["ion", "klobuchar", "ntcm"])
    model_names: list[str] | None = None
    scale: float = 1.0
    gridsize: int = 80
    mincnt: int = 1

    def __post_init__(self) -> None:
        # Validated and normalised at construction (VIZ-006, VIZ-026): a str
        # becomes a one-element list, so the stored lists are what is used.
        self.model_cols, self.model_names = self._checked_models()
        self._check_columns(self.model_cols)

    def _checked_models(self) -> tuple[list, list | None]:
        """Return ``(model_cols, model_names)`` as lists after the argument checks."""
        cols = _as_name_list("model_cols", self.model_cols)
        if not cols:
            raise ValueError("model_cols is empty: name at least one model column to compare.")
        names = None
        if self.model_names is not None:
            names = _as_name_list("model_names", self.model_names)
            if len(names) != len(cols):
                raise ValueError(
                    f"model_names must hold one label per model column: got {len(names)} "
                    f"label(s) {names!r} for {len(cols)} column(s) {cols!r}."
                )
        return cols, names

    def _check_columns(self, cols: list) -> None:
        """Raise unless ``df`` is a frame holding ``x_col`` and every model column."""
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame (got {type(self.df).__name__})")
        available = list(self.df.columns)
        if self.x_col not in self.df.columns:
            raise KeyError(
                f"Measured column x_col={self.x_col!r} not found. Available: {available}."
            )
        missing = [column for column in cols if column not in self.df.columns]
        if missing:
            raise KeyError(f"Model columns not found: {missing}. Available: {available}.")

    def _validated(self) -> tuple[list, list]:
        """Run every check and return ``(model columns, labels)`` for drawing."""
        cols, names = self._checked_models()
        self._check_columns(cols)
        return cols, list(names if names is not None else cols)

    @property
    def labels(self) -> list[str]:
        """Model labels, falling back to the column names."""
        cols, names = self._checked_models()
        return list(names if names is not None else cols)

    def _xy(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the finite (measured, modelled) pairs for one model."""
        x = self.df[self.x_col].to_numpy()
        y = (self.df[model] * self.scale).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    @staticmethod
    def _metrics(x: np.ndarray, y: np.ndarray) -> dict:
        """Compute error statistics of modelled ``y`` against measured ``x``."""
        from scipy import stats

        error = y - x
        n = error.size
        if n < 2:
            return {
                key: np.nan
                for key in ("Bias", "RMS", "STD", "MAE", "Corr", "R2", "Slope", "Intercept")
            } | {"N": int(n)}

        if np.std(x) == 0 or np.std(y) == 0:
            corr = r2 = slope = intercept = np.nan
        else:
            slope, intercept, r, _, _ = stats.linregress(x, y)
            corr = r
            r2 = r**2

        return {
            "Bias": float(np.mean(error)),
            "RMS": float(np.sqrt(np.mean(error**2))),
            "STD": float(np.std(error, ddof=0)),
            "MAE": float(np.mean(np.abs(error))),
            "Corr": corr,
            "R2": r2,
            "Slope": slope,
            "Intercept": intercept,
            "N": int(n),
        }

    def compute_stats(self) -> pd.DataFrame:
        """Compute the comparison statistics, one row per model.

        Returns
        -------
        pandas.DataFrame
            Columns ``Model``, ``N``, ``Bias``, ``STD``, ``RMS``, ``MAE``,
            ``Corr``, ``R2``, ``Slope`` and ``Intercept``.

        Raises
        ------
        KeyError, ValueError, TypeError
            As for the constructor, when an attribute was changed since.
        """
        cols, labels = self._validated()
        rows = [
            {"Model": name, **self._metrics(*self._xy(column))}
            for column, name in zip(cols, labels)
        ]
        stats_df = pd.DataFrame(rows)
        for column in _ROUNDED_STATS:
            if column in stats_df:
                stats_df[column] = stats_df[column].astype(float).round(3)
        return stats_df[_STATS_ORDER]

    @no_figure_leak
    def plot_hexbin(
        self,
        *,
        suptitle: str | None = None,
        figsize: tuple[float, float] = (18, 6),
        save_path: str | Path | None = None,
        close: bool = False,
    ) -> Figure:
        """Plot modelled against measured STEC, one hexbin panel per model.

        Each panel carries the identity line and the fitted regression, so the
        two ways a model can be wrong (offset and slope) are separable by eye.
        All panels share one colour normalisation, so the common colorbar
        reads correctly on every panel.

        Parameters
        ----------
        suptitle:
            Figure title.
        figsize:
            Figure size in inches.
        save_path:
            When given, write the figure to this path.
        close:
            Close the figure after saving; the figure is still returned but no
            longer renders.

        Returns
        -------
        matplotlib.figure.Figure
            The figure holding one panel per model.

        Raises
        ------
        KeyError, ValueError, TypeError
            As for the constructor, before the figure is created.
        """
        from scipy import stats

        import matplotlib.pyplot as plt

        cols, labels = self._validated()  # before the figure (VIZ-006)
        fig, axes = plt.subplots(1, len(cols), figsize=figsize, sharex=True, sharey=True)
        axes = list(np.atleast_1d(axes))

        x_all = self.df[self.x_col].to_numpy()
        finite = np.isfinite(x_all)
        x_min = float(np.min(x_all[finite])) if finite.any() else 0.0
        x_max = float(np.max(x_all[finite])) if finite.any() else 1.0
        bounds = np.array([x_min, x_max])

        hexbins = []
        for ax, column, name in zip(axes, cols, labels):
            x, y = self._xy(column)
            if x.size == 0:
                ax.set_title(f"{name} (no data)")
                continue

            hexbins.append(
                ax.hexbin(x, y, gridsize=self.gridsize, cmap="plasma", mincnt=self.mincnt)
            )
            ax.plot(bounds, bounds, ls="--", color="red", lw=1)

            if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
                slope, intercept, r, _, _ = stats.linregress(x, y)
                ax.plot(
                    bounds,
                    slope * bounds + intercept,
                    color="orange",
                    label=f"Fit: y={slope:.2f}x{intercept:+.2f}\nR²={r**2:.2f}",
                )
                ax.legend(fontsize=9)

            ax.set_title(name)
            ax.grid(True, ls=":", alpha=0.6)
            ax.set_xlabel("Measured STEC [TECU]")

        axes[0].set_ylabel("Modelled STEC [TECU]")
        if hexbins:
            # Each hexbin autoscales to its own counts; one shared norm makes
            # the single colorbar valid for every panel.
            counts = [np.asarray(hexbin.get_array()) for hexbin in hexbins]
            counts = [c for c in counts if c.size]
            if counts:
                norm = Normalize(
                    vmin=min(float(np.min(c)) for c in counts),
                    vmax=max(float(np.max(c)) for c in counts),
                )
                for hexbin in hexbins:
                    hexbin.set_norm(norm)
            cbar = fig.colorbar(
                hexbins[0], ax=axes, orientation="horizontal", fraction=0.05, pad=0.1
            )
            cbar.set_label("Point Count")
        if suptitle:
            fig.suptitle(suptitle, y=1.02)
        return finalize(fig, save_path=save_path, close=close)

    @no_figure_leak
    def plot_error_hist(
        self,
        *,
        bins: int = 60,
        suptitle: str | None = "Error histogram",
        figsize: tuple[float, float] = (18, 5),
        density: bool = False,
        save_path: str | Path | None = None,
        close: bool = False,
    ) -> Figure:
        """Plot the error distribution of each model.

        Parameters
        ----------
        bins:
            Number of histogram bins.
        suptitle:
            Figure title.
        figsize:
            Figure size in inches.
        density:
            Normalise the histogram to a density.
        save_path:
            When given, write the figure to this path.
        close:
            Close the figure after saving; the figure is still returned but no
            longer renders.

        Returns
        -------
        matplotlib.figure.Figure
            The figure holding one panel per model.

        Raises
        ------
        KeyError, ValueError, TypeError
            As for the constructor, before the figure is created.
        """
        import matplotlib.pyplot as plt

        cols, labels = self._validated()  # before the figure (VIZ-006)
        fig, axes = plt.subplots(1, len(cols), figsize=figsize, sharex=True, sharey=True)
        axes = list(np.atleast_1d(axes))

        for ax, column, name in zip(axes, cols, labels):
            x, y = self._xy(column)
            error = y - x
            if error.size == 0:
                ax.set_title(f"{name} (no data)")
                continue

            ax.hist(error, bins=bins, alpha=0.85, edgecolor="k", density=density)
            mean = float(np.mean(error))
            ax.axvline(0, color="red", ls="--", lw=1)
            ax.axvline(mean, color="black", ls="-", lw=1)
            ax.set_title(f"{name}\nBias={mean:.2f}, STD={float(np.std(error)):.2f}")
            ax.set_xlabel("Error [TECU]")
            ax.grid(True, ls=":", alpha=0.6)

        axes[0].set_ylabel("Density" if density else "Count")
        if suptitle:
            fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        return finalize(fig, save_path=save_path, close=close)

    @no_figure_leak
    def report(
        self,
        *,
        title: str = "Modelled vs Measured STEC",
        show_hex: bool = True,
        show_hist: bool = True,
    ) -> STECComparisonReport:
        """Compute the statistics and draw the comparison figures.

        The figures are returned rather than shown, so the caller can save or
        close them; in a notebook they render as the cell's open figures.

        Parameters
        ----------
        title:
            Title of the hexbin figure.
        show_hex:
            Draw the hexbin comparison.
        show_hist:
            Draw the error histogram.

        Returns
        -------
        STECComparisonReport
            Named tuple of the statistics from :meth:`compute_stats`, the
            hexbin figure and the histogram figure; a figure that was not
            drawn is None. It unpacks as ``stats, hexbin, hist = ...``.

        Raises
        ------
        KeyError, ValueError, TypeError
            As for the constructor, before any figure is created. When the
            histogram fails after the hexbin was drawn, the hexbin figure is
            closed before the error propagates.
        """
        stats_df = self.compute_stats()
        hexbin = self.plot_hexbin(suptitle=title) if show_hex else None
        hist = self.plot_error_hist() if show_hist else None
        return STECComparisonReport(stats=stats_df, hexbin=hexbin, hist=hist)
