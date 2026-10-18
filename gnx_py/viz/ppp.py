"""Plots for PPP solutions and PPP filter residuals.

The residual plots expect the ``(sv, time)`` MultiIndex frames produced by the
PPP filters, and the positioning plot expects a :class:`~pandas.DataFrame`
solution indexed by time, as carried by ``PPPResult.solution``.

Residual column names differ between PPP routes: the combined filters expose
compact names (``v``, ``vc``) while the uncombined ones expose per-band names
(``vl1``, ``vl2``, ``vc1``, ``vc2``). The plots resolve those names themselves,
so callers do not have to know which route produced the frame.
"""
from __future__ import annotations

import numbers
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import (
    _check_dpi,
    _check_real,
    _scatter_by_sv,
    add_prn_legend,
    constellation_label,
    finalize,
    format_time_axis,
    make_fig_ax,
    make_fig_axes,
    no_figure_leak,
    resolve_time_axis,
)

# Residual columns tried in order, per observation type. Combined PPP routes
# use the short names; the rest come from filter diagnostics.
_PHASE_COLUMNS = ("v", "postfit_phase_res", "postfit_phase_res_demeaned", "res", "vl1")
_CODE_COLUMNS = ("vc", "postfit_code_res", "postfit_code_res_demeaned", "vc1")

# Uncombined (UDUC) residuals, keyed by (observation type, band index).
_UDUC_COLUMNS = {
    ("phase", 0): "vl1",
    ("phase", 1): "vl2",
    ("code", 0): "vc1",
    ("code", 1): "vc2",
}

_OBS_NAMES = {"phase": "Carrier-phase", "code": "Pseudorange"}

# Solution columns in plotting order, with axis labels.
_SOLUTION_COLUMNS = (
    ("de", "East [m]"),
    ("dn", "North [m]"),
    ("du", "Up [m]"),
    ("dtr", "Receiver Clock [m]"),
    ("isb", "ISB [m]"),
    ("ztd", "ZTD [m]"),
    ("ar_fixed", "Fixed ambiguities [-]"),
    ("ar_ratio", "AR ratio [-]"),
)
_CONVERGENCE_COMPONENTS = frozenset({"de", "dn", "du"})

__all__ = [
    "plot_positioning_series",
    "plot_residuals",
    "plot_uduc_residuals",
]


def _check_count(name: str, value: object) -> int:
    """Return ``value`` as an int, or raise ValueError naming ``name``.

    Epoch counts must be integers >= 0 (bool rejected): a negative count
    would silently cut from the end of the session or arc instead.
    """
    from ..ionosphere.utils import _check_number  # lazy: no import cycle with ionosphere

    if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
        # Checked here rather than in _check_number, whose np.isfinite cannot
        # take a Python int beyond the int64 range (such as 10**30).
        count = int(value)
        if count < 0:
            raise ValueError(f"{name} must be >= 0 (got {count!r})")
        return count
    _check_number(name, value, ge=0, integer=True)
    return int(value)  # type: ignore[arg-type]


def _validate_residuals(
    df: pd.DataFrame,
    *,
    obs_type: str,
    x_axis: str,
    cut_init_time: int,
    skip_obs: int,
    uduc: bool = False,
    value_col: str | None = None,
) -> tuple[pd.DataFrame, int, tuple[str, ...]]:
    """Validate a residual frame and its options, then drop its initial epochs.

    Everything that can fail is checked here, on the whole frame and before a
    figure exists: the options, the ``(sv, time)`` index levels, the epoch
    counts, the ``ev`` column for an elevation axis and the residual columns.
    Checking the whole frame, not the arcs left after trimming, means a frame
    is accepted or rejected the same way whatever ``skip_obs`` is.

    Parameters
    ----------
    uduc:
        Require both per-band columns of ``obs_type`` (the uncombined plot)
        instead of resolving one column.
    value_col:
        Residual column to plot when ``uduc`` is False. None tries the
        defaults of ``obs_type`` in order and uses the first one present.

    Returns
    -------
    tuple
        The frame without its first ``cut_init_time`` epochs, ``skip_obs`` as
        an int, and the residual columns to plot: one for the combined plot,
        the two bands in panel order for the uncombined one.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "Residual frame must have a MultiIndex with levels ('sv', 'time')."
        )
    if obs_type not in _OBS_NAMES:
        raise ValueError(f"obs_type must be 'phase' or 'code', got {obs_type!r}.")
    if x_axis not in ("time", "elev"):
        raise ValueError(f"x_axis must be 'time' or 'elev', got {x_axis!r}.")
    cut_init_time = _check_count("cut_init_time", cut_init_time)
    skip_obs = _check_count("skip_obs", skip_obs)

    missing_levels = [name for name in ("sv", "time") if name not in df.index.names]
    if missing_levels:
        raise KeyError(
            f"Residual frame has no index level {missing_levels[0]!r}; expected levels "
            f"('sv', 'time'), got {list(df.index.names)}."
        )
    if x_axis == "elev" and "ev" not in df.columns:
        raise KeyError("Column 'ev' (elevation) not found in the frame.")

    if uduc:
        columns = tuple(_UDUC_COLUMNS[(obs_type, index)] for index in range(2))
        for name in columns:
            if name not in df.columns:
                raise KeyError(
                    f"Column {name!r} not found; is this an uncombined PPP residual "
                    f"frame? Available: {list(df.columns)}."
                )
    elif value_col is not None:
        if value_col not in df.columns:
            raise KeyError(
                f"Residual column {value_col!r} not found. Available: {list(df.columns)}."
            )
        columns = (value_col,)
    else:
        candidates = _PHASE_COLUMNS if obs_type == "phase" else _CODE_COLUMNS
        column = next((name for name in candidates if name in df.columns), None)
        if column is None:
            raise KeyError(
                f"No {obs_type} residual column found. Tried: {list(candidates)}; "
                f"available: {list(df.columns)}."
            )
        columns = (column,)

    all_times = df.index.get_level_values("time").unique().sort_values()
    if len(all_times) <= cut_init_time:
        raise ValueError(
            f"Cannot cut {cut_init_time} initial epochs: the frame only has "
            f"{len(all_times)}."
        )
    trimmed = df[df.index.get_level_values("time").isin(all_times[cut_init_time:])]
    return trimmed, skip_obs, columns


def _drop_arc_starts(trimmed: pd.DataFrame, *, skip_obs: int) -> pd.DataFrame:
    """Drop the first ``skip_obs`` epochs of every satellite arc.

    The first epochs after a satellite rises carry the ambiguity convergence
    transient, which would dominate the y scale. Exactly ``skip_obs`` epochs
    are dropped, so ``skip_obs=0`` keeps every row; rows sharing an epoch are
    kept or dropped together. Arcs with nothing left are dropped whole. The
    result is grouped satellite by satellite, in sorted ``sv`` order, and each
    satellite's rows keep their original order.
    """
    arcs = []
    for _, group in trimmed.groupby(level="sv"):
        times = group.index.get_level_values("time")
        kept_epochs = times.unique().sort_values()[skip_obs:]
        if len(kept_epochs):
            arcs.append(group[times.isin(kept_epochs)])

    if not arcs:
        return trimmed.iloc[:0]
    return pd.concat(arcs)


def _frame_systems(df: pd.DataFrame) -> list[str]:
    """Sorted system letters of the satellites in the ``sv`` level of ``df``."""
    return sorted({str(sv)[:1] for sv in df.index.get_level_values("sv").unique()})


def _check_mode_systems(mode: str, df: pd.DataFrame, arcs: pd.DataFrame) -> None:
    """Raise ValueError when ``mode`` belongs to a system that is not drawn.

    The system must be among the satellites of the whole frame ``df``, and,
    when anything is left to draw, among those of ``arcs`` (the rows left
    after ``cut_init_time`` and ``skip_obs``): otherwise the panels would be
    labelled with the bands of a constellation none of the plotted points
    belongs to.
    """
    from ..core.gnss import mode_system  # lazy: viz must not import the processing package

    system = mode_system(mode)
    systems = _frame_systems(df)
    if system not in systems:
        raise ValueError(
            f"mode {mode!r} is a signal pair of system {system!r}, but the frame holds "
            f"satellites of {systems}; pass the mode of a constellation in the frame."
        )
    if not arcs.empty:
        drawn = _frame_systems(arcs)
        if system not in drawn:
            raise ValueError(
                f"mode {mode!r} is a signal pair of system {system!r}, but no satellite "
                f"of that system is left after cut_init_time and skip_obs; the plotted "
                f"satellites are of {drawn}. Lower cut_init_time or skip_obs, or pass "
                f"the mode of a plotted constellation."
            )


def _residual_title(prns: list[str], obs_type: str) -> str:
    """Name the plot after the constellations actually drawn."""
    return f"{constellation_label(prns)} {_OBS_NAMES[obs_type]} residuals"


@no_figure_leak
def plot_residuals(
    df: pd.DataFrame,
    *,
    cut_init_time: int = 30,
    skip_obs: int = 5,
    obs_type: Literal["phase", "code"] = "phase",
    x_axis: Literal["time", "elev"] = "time",
    value_col: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | None = None,
    close: bool = False,
) -> Figure:
    """Plot post-fit PPP residuals of every satellite against time or elevation.

    Parameters
    ----------
    df:
        Residual frame with a ``(sv, time)`` MultiIndex. The residual column is
        resolved from the frame: ``v``/``postfit_phase_res``/``vl1`` for phase,
        ``vc``/``postfit_code_res``/``vc1`` for code.
    cut_init_time:
        Number of initial epochs of the whole session to drop, covering filter
        convergence. An integer >= 0.
    skip_obs:
        Number of epochs to drop at the start of each satellite arc, an
        integer >= 0; ``0`` keeps every epoch.
    obs_type:
        Plot carrier-phase or pseudorange residuals.
    x_axis:
        Plot against ``"time"``, or against the ``ev`` elevation column.
    value_col:
        Name of the residual column to plot, in metres, such as
        ``"postfit_phase_res_demeaned"``. None resolves it from the frame as
        described under ``df``. ``obs_type`` still names the plot either way.
    title:
        Plot title. None derives it from the constellations present.
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
        When the index is not a MultiIndex, the options are unknown,
        ``cut_init_time`` or ``skip_obs`` is not an integer >= 0, or there are
        no more epochs than ``cut_init_time``.
    KeyError
        When the index has no ``sv`` or ``time`` level, ``value_col`` is
        missing, no residual column for ``obs_type`` is present, or ``x_axis``
        is ``"elev"`` and the ``ev`` column is missing. These are checked on
        the whole frame, before any arc is trimmed and before a figure exists.
    """
    trimmed, skip_obs, (column,) = _validate_residuals(
        df,
        obs_type=obs_type,
        x_axis=x_axis,
        cut_init_time=cut_init_time,
        skip_obs=skip_obs,
        value_col=value_col,
    )

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    arcs = _drop_arc_starts(trimmed, skip_obs=skip_obs)
    seen_prns: list[str] = []
    if not arcs.empty:
        # One colour per PRN, shared by its arcs, so the per-PRN legend
        # identifies each satellite.
        seen_prns = _scatter_by_sv(
            ax,
            arcs,
            column,
            x_col=None if x_axis == "time" else "ev",
            colors="palette",
            dropna=False,
        )

    if x_axis == "time":
        format_time_axis(ax, autofmt=False)
        ax.set_xlabel("Time")
    else:
        ax.set_xlabel("Elevation [deg]")

    ax.set_ylabel("Residual [m]")
    ax.set_title(title if title is not None else _residual_title(seen_prns, obs_type))
    ax.grid(True)
    add_prn_legend(ax)

    if owns_fig:
        # Leave room on the right for the legend.
        fig.tight_layout(rect=(0, 0, 0.85, 1))
    return finalize(fig, save_path=save_path, close=close)


@no_figure_leak
def plot_uduc_residuals(
    df: pd.DataFrame,
    *,
    cut_init_time: int = 30,
    skip_obs: int = 5,
    obs_type: Literal["phase", "code"] = "phase",
    x_axis: Literal["time", "elev"] = "time",
    mode: str = "L1L2",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | None = None,
    close: bool = False,
) -> Figure:
    """Plot uncombined PPP residuals, one panel per frequency band.

    This is the two-band counterpart of :func:`plot_residuals`: uncombined
    filters estimate each band separately, so each band gets its own panel.
    The panels share the x axis.

    Parameters
    ----------
    df:
        Residual frame with a ``(sv, time)`` MultiIndex and per-band residual
        columns (``vl1``/``vl2`` for phase, ``vc1``/``vc2`` for code).
    cut_init_time:
        Number of initial epochs of the whole session to drop, an integer
        >= 0.
    skip_obs:
        Number of epochs to drop at the start of each satellite arc, an
        integer >= 0; ``0`` keeps every epoch.
    obs_type:
        Plot carrier-phase or pseudorange residuals.
    x_axis:
        Plot against ``"time"``, or against the ``ev`` elevation column.
    mode:
        Dual-frequency signal pair, such as ``"L1L2"``, ``"E1E5a"`` or
        ``"B1IB3I"``: any two-signal mode of :func:`gnx_py.gnss.mode_signals`.
        It labels the panels, so its constellation must be among the
        satellites of ``df`` and among those left to draw after
        ``cut_init_time`` and ``skip_obs``. Satellites of other systems in a
        mixed frame are drawn too, under the same band labels.
    title:
        Title above the first panel. None derives it from the constellations
        present.
    figsize:
        Figure size in inches. This plot always creates its own figure, since
        it needs two axes.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding both panels.

    Raises
    ------
    ValueError
        When the index is not a MultiIndex, an option is unknown,
        ``cut_init_time`` or ``skip_obs`` is not an integer >= 0, there are no
        more epochs than ``cut_init_time``, ``mode`` is not a dual-frequency
        mode, or its constellation has no satellite in ``df`` or, when any
        point is left to draw, none among the satellites left after
        ``cut_init_time`` and ``skip_obs``.
    KeyError
        When the index has no ``sv`` or ``time`` level, a per-band residual
        column is missing, or ``x_axis`` is ``"elev"`` and the ``ev`` column
        is missing. These are checked on the whole frame, before any arc is
        trimmed and before a figure exists.
    """
    # Imported here: gnx_py.viz must not import the processing package at
    # module level.
    from ..core.gnss import dual_mode_signals

    try:
        bands = dual_mode_signals(mode)
    except ValueError as exc:
        raise ValueError(
            "mode must be a dual-frequency signal pair such as 'L1L2' or "
            f"'E1E5a', got {mode!r}."
        ) from exc

    trimmed, skip_obs, band_columns = _validate_residuals(
        df,
        obs_type=obs_type,
        x_axis=x_axis,
        cut_init_time=cut_init_time,
        skip_obs=skip_obs,
        uduc=True,
    )
    arcs = _drop_arc_starts(trimmed, skip_obs=skip_obs)
    _check_mode_systems(mode, df, arcs)

    fig, axes = make_fig_axes(2, figsize=figsize)

    seen_prns: list[str] = []
    if not arcs.empty:
        for index, column in enumerate(band_columns):
            # Both panels draw from the same arcs, so the palette gives a
            # PRN the same colour in each band.
            seen_prns = _scatter_by_sv(
                axes[index],
                arcs,
                column,
                x_col=None if x_axis == "time" else "ev",
                colors="palette",
                dropna=False,
            )

    for index, band in enumerate(bands):
        axes[index].set_ylabel(f"{band} Residuals [m]")
        axes[index].grid(True)

    if x_axis == "time":
        format_time_axis(axes[1], autofmt=False)
        axes[1].set_xlabel("Time")
    else:
        axes[1].set_xlabel("Elevation [deg]")

    axes[0].set_title(
        title if title is not None else _residual_title(seen_prns, obs_type)
    )
    add_prn_legend(axes[0])
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    return finalize(fig, save_path=save_path, close=close)


@no_figure_leak
def plot_positioning_series(
    solution: pd.DataFrame,
    *,
    cut_init_time: int = 0,
    color: str = "red",
    convergence_time: float | None = None,
    time_col: str = "time",
    title: str | None = None,
    xlabel: str = "Time [GPST]",
    figsize: tuple[float, float] = (12, 8),
    save_path: str | None = None,
    close: bool = False,
    dpi: int = 300,
) -> Figure:
    """Plot the PPP solution time series, one panel per estimated parameter.

    Panels are created only for the columns present, in a fixed order: the ENU
    position components first, then receiver clock, ISB, ZTD and the ambiguity
    resolution diagnostics. Columns that are entirely empty are skipped, as in
    :func:`~gnx_py.viz.spp.plot_spp_positioning`, so a run without reference
    coordinates does not get blank ENU panels.

    Parameters
    ----------
    solution:
        Solution frame, as in ``PPPResult.solution``. Supported columns are
        ``de``, ``dn``, ``du``, ``dtr``, ``isb``, ``ztd``, ``ar_fixed`` and
        ``ar_ratio``. Epochs may sit in the index or in a ``time`` column: a
        solution written with ``to_csv`` and read back has them in a column.
    cut_init_time:
        Number of initial epochs to drop, an integer >= 0.
    color:
        Line colour.
    convergence_time:
        Convergence time in minutes, a finite number >= 0. When given, a
        vertical line is drawn at that offset from the first epoch on the
        position panels.
    time_col:
        Name of the epoch column, used when the frame carries one. Otherwise
        the index is used, which is where ``PPPResult.solution`` keeps them.
    title:
        Figure title.
    xlabel:
        Label of the bottom (shared) time axis. The default names GPS time,
        the scale of the RINEX observation epochs the solution is computed
        at; pass another label if the epochs were converted.
    figsize:
        Figure size in inches.
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
        The figure holding one panel per plotted parameter.

    Raises
    ------
    ValueError
        When the frame holds none of the supported columns, or they are all
        empty, ``cut_init_time`` is not an integer >= 0, or
        ``convergence_time`` is not a finite number >= 0 or puts the
        convergence line outside the supported time range.
    KeyError
        When the epochs can be found neither in ``time_col`` nor in the index.
    """
    # Every option is checked before the figure exists, so a bad value leaves
    # none open.
    _check_dpi(dpi)
    cut_init_time = _check_count("cut_init_time", cut_init_time)
    convergence_offset = None
    if convergence_time is not None:
        minutes = _check_real("convergence_time", convergence_time, ge=0)
        try:
            convergence_offset = timedelta(minutes=minutes)
        except OverflowError as exc:
            raise ValueError(
                f"convergence_time is too large for a time offset (got {minutes!r} min)"
            ) from exc
    if cut_init_time > 0:
        solution = solution.iloc[cut_init_time:]

    supported = [(col, name) for col, name in _SOLUTION_COLUMNS if col in solution.columns]
    if not supported:
        raise ValueError(
            "No supported positioning columns found. Expected any of: "
            f"{[col for col, _ in _SOLUTION_COLUMNS]}."
        )
    present = [(col, name) for col, name in supported if solution[col].notna().any()]
    if not present:
        raise ValueError(
            "The supported positioning columns hold no data: "
            f"{[col for col, _ in supported]}."
        )

    times = list(resolve_time_axis(solution, time_col))
    convergence_x = None
    if convergence_offset is not None:
        # timedelta accepts offsets far beyond the pandas Timestamp range
        # (about +-292 years), so the line position itself is checked here,
        # before the figure exists.
        try:
            convergence_x = times[0] + convergence_offset
        except (OverflowError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta) as exc:
            raise ValueError(
                f"convergence_time is too large for a time offset (got {minutes!r} min): "
                f"the first epoch plus it falls outside the supported time range"
            ) from exc
    fig, axes = make_fig_axes(len(present), figsize=figsize)

    for index, (column, name) in enumerate(present):
        axes[index].plot(times, solution[column], label=name, color=color)
        axes[index].grid()
        axes[index].legend(loc="best", fontsize=9)
        format_time_axis(axes[index], autofmt=False)
        if convergence_x is not None and column in _CONVERGENCE_COMPONENTS:
            axes[index].axvline(x=convergence_x)

    axes[-1].set_xlabel(xlabel)
    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return finalize(fig, save_path=save_path, dpi=dpi, close=close)
