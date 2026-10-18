"""Plots for SPP solutions and SPP residuals.

These consume :class:`~gnx_py.spp.SPPResult`: ``solution`` for
:func:`plot_spp_positioning`, and the per-constellation ``residuals_gps`` /
``residuals_gal`` / ``residuals_bds`` frames for :func:`plot_spp_residuals`.

SPP results are shaped differently from PPP results, so these are separate
functions rather than aliases of :mod:`gnx_py.viz.ppp`:

* the SPP solution carries time in a ``time`` column, not in its index, and
  names its receiver clock per constellation (``dtr_gps``) rather than once
  (``dtr``);
* SPP residuals live in one frame per constellation, in a ``V`` column
  (finite exactly for the observations used in the solution); an observation
  dropped by the residual-rejection loop keeps ``V`` NaN and carries its
  residual against the epoch's final solution in ``V_rejected``;
* SPP solves each epoch independently, so there is no filter convergence to
  trim. Every epoch is plotted.

Columns that are present but empty are skipped, because a GPS-only run still
carries ``dtr_gal`` and ``n_gal`` as all-NaN and all-zero columns.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import (
    _check_dpi,
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

# Post-fit residual column written by the SPP least squares.
_RESIDUAL_COLUMN = "V"
# Residual of an observation the rejection loop dropped, against the epoch's
# final solution (``V`` stays NaN for it; see gnx_py.spp.SPPResult).
_REJECTED_RESIDUAL_COLUMN = "V_rejected"
# Flag and reason written by the SPP residual-rejection loop.
_REJECTED_COLUMN = "spp_rejected"
_REJECT_REASON_COLUMN = "spp_reject_reason"

# Axis labels for the solution columns, by column name.
_SOLUTION_LABELS = {
    "de": "East [m]",
    "dn": "North [m]",
    "du": "Up [m]",
    "x": "X [m]",
    "y": "Y [m]",
    "z": "Z [m]",
    "dtr_gps": "GPS clock [m]",
    "dtr_gal": "Galileo clock [m]",
    "dtr_bds": "BeiDou clock [m]",
    "sig_x": "σ X [m]",
    "sig_y": "σ Y [m]",
    "sig_z": "σ Z [m]",
    "sig_dtr": "σ clock [m]",
    "rms_v": "Residual RMS [m]",
    "spp_design_cond": "Design condition [-]",
    "spp_rejected_obs": "Rejected obs [-]",
    "iters": "Iterations [-]",
}

# Position components, plotted by default.
_DEFAULT_COMPONENTS = ("de", "dn", "du")

# Satellite counts, drawn together in one panel.
_SATELLITE_COUNTS = (
    ("n_gps", "GPS", "tab:blue"),
    ("n_gal", "Galileo", "tab:orange"),
    ("n_bds", "BeiDou", "tab:red"),
)

__all__ = [
    "plot_spp_positioning",
    "plot_spp_residuals",
]


def _has_data(series: pd.Series) -> bool:
    """True when a column carries something worth a panel."""
    return not series.isna().all()


def _column_list(columns: object) -> list[str]:
    """``columns`` as a list of names; a single name given as a string is wrapped.

    A string is a sequence of characters, so iterating ``"de"`` would ask for
    columns ``"d"`` and ``"e"``. As elsewhere in GNX, one name is accepted for
    a one-element list.
    """
    if isinstance(columns, str):
        return [columns]
    try:
        names = list(columns)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "columns must be a column name or a sequence of column names, "
            f"got {type(columns).__name__}."
        ) from None
    bad = [name for name in names if not isinstance(name, str)]
    if bad:
        raise TypeError(
            f"columns must hold column names (str), got {bad[0]!r} "
            f"({type(bad[0]).__name__})."
        )
    return names


@no_figure_leak
def plot_spp_residuals(
    residuals: pd.DataFrame,
    *,
    x_axis: Literal["time", "elev"] = "time",
    value_col: str = _RESIDUAL_COLUMN,
    show_rejected: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot post-fit SPP pseudorange residuals of every satellite.

    Pass one constellation's frame at a time, as SPP produces them. To compare
    constellations, draw them onto a shared layout with ``ax``::

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for ax, frame in zip(axes, (r.residuals_gps, r.residuals_gal, r.residuals_bds)):
            if frame is not None:
                plot_spp_residuals(frame, ax=ax)

    Parameters
    ----------
    residuals:
        One constellation's residual frame, with a ``(sv, time)`` MultiIndex
        and a residual column, as carried by ``SPPResult.residuals_gps`` and
        its siblings. The satellites come from an ``sv`` column or index
        level, the epochs from a ``time`` column or index level, located as
        by :func:`~gnx_py.viz.core.resolve_time_axis` (so epochs stored as
        strings, as after a CSV round trip, are converted).
    x_axis:
        Plot against ``"time"``, or against the ``ev`` elevation column.
    value_col:
        Name of the post-fit residual column, in metres.
    show_rejected:
        Draw observations flagged by the SPP rejection loop as black crosses,
        separately from the accepted ones and on the same x axis. Ignored
        when the frame carries no ``spp_rejected`` column. A real SPP frame
        leaves ``V`` NaN for the observations it did not use, so when
        ``value_col`` is ``"V"`` a rejected row without a finite ``V`` is
        drawn at its ``V_rejected`` value (its residual against the solution
        it was excluded from). Rejected rows with no finite value at all,
        such as those of an epoch rejected as a whole, are not drawn; the
        legend entry ``Rejected (n)`` counts the crosses drawn.
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
        When the index is not a MultiIndex, or ``x_axis`` is unknown.
    KeyError
        When the residual column is missing, no ``sv`` column or index level
        holds the satellites, ``x_axis`` is ``"time"`` and the epochs cannot
        be located, or ``x_axis`` is ``"elev"`` and the ``ev`` column is
        missing.
    TypeError
        When ``x_axis`` is ``"time"`` and the epochs cannot be read as
        datetimes (numeric values or unparseable strings).

    Notes
    -----
    Every check runs before the figure is created; a figure this call created
    is closed if it fails anyway (:func:`~gnx_py.viz.core.no_figure_leak`).
    """
    if not isinstance(residuals.index, pd.MultiIndex):
        raise ValueError(
            "SPP residual frame must have a MultiIndex with levels ('sv', 'time')."
        )
    if x_axis not in ("time", "elev"):
        raise ValueError(f"x_axis must be 'time' or 'elev', got {x_axis!r}.")
    if value_col not in residuals.columns:
        raise KeyError(
            f"Residual column {value_col!r} not found. Is this an SPP residual "
            f"frame? Available: {list(residuals.columns)[:10]}."
        )
    if "sv" not in residuals.columns and "sv" not in residuals.index.names:
        raise KeyError(
            "No column or index level named 'sv' holds the satellites; index "
            f"levels: {list(residuals.index.names)}."
        )
    if x_axis == "elev" and "ev" not in residuals.columns:
        raise KeyError("Column 'ev' (elevation) not found in the frame.")
    # One time rule for accepted and rejected rows (Q174), resolved before the
    # figure exists so a frame without readable epochs leaves nothing open.
    times = resolve_time_axis(residuals, "time") if x_axis == "time" else None

    rejected_mask = (
        residuals[_REJECTED_COLUMN].fillna(False).astype(bool).to_numpy()
        if show_rejected and _REJECTED_COLUMN in residuals.columns
        else np.zeros(len(residuals), dtype=bool)
    )
    rejected_values = residuals[value_col].to_numpy(dtype="float64", copy=True)
    if value_col == _RESIDUAL_COLUMN and _REJECTED_RESIDUAL_COLUMN in residuals.columns:
        # V is NaN for every observation SPP did not use; V_rejected holds the
        # residual of those the rejection loop dropped.
        fallback = residuals[_REJECTED_RESIDUAL_COLUMN].to_numpy(dtype="float64")
        rejected_values = np.where(np.isfinite(rejected_values), rejected_values, fallback)
    drawn_rejected = rejected_mask & np.isfinite(rejected_values)

    owns_fig = ax is None
    fig, ax = make_fig_ax(ax, figsize=figsize)

    x_col = None if x_axis == "time" else "ev"
    # One colour per PRN, shared by its arcs, as in the PPP residual plots.
    seen_prns = _scatter_by_sv(
        ax, residuals[~rejected_mask], value_col, x_col=x_col, colors="palette"
    )

    # Rejected observations go on top, in one colour (black differs from every
    # palette colour): the point is to see where the solver dropped data, not
    # which satellite it came from.
    n_rejected = int(np.count_nonzero(drawn_rejected))
    if n_rejected:
        ax.scatter(
            times[drawn_rejected] if x_col is None else residuals[x_col].to_numpy()[drawn_rejected],
            rejected_values[drawn_rejected],
            s=18,
            marker="x",
            color="black",
            linewidths=0.8,
            zorder=3,
            label=f"Rejected ({n_rejected})",
        )

    if x_axis == "time":
        format_time_axis(ax, autofmt=False)
        ax.set_xlabel("Time")
    else:
        ax.set_xlabel("Elevation [deg]")

    ax.set_ylabel("Residual [m]")
    ax.set_title(
        title
        if title is not None
        else f"{constellation_label(seen_prns)} SPP pseudorange residuals"
    )
    ax.grid(True)
    add_prn_legend(ax)

    if owns_fig:
        # Leave room on the right for the legend.
        fig.tight_layout(rect=(0, 0, 0.85, 1))
    return finalize(fig, save_path=save_path, close=close)


@no_figure_leak
def plot_spp_positioning(
    solution: pd.DataFrame,
    *,
    columns: Sequence[str] = _DEFAULT_COMPONENTS,
    show_satellites: bool = True,
    time_col: str = "time",
    color: str = "red",
    title: str | None = None,
    xlabel: str = "Time [GPST]",
    figsize: tuple[float, float] = (12, 8),
    save_path: str | Path | None = None,
    close: bool = False,
    dpi: int = 300,
) -> Figure:
    """Plot the SPP solution time series, one panel per parameter.

    The position error components are plotted by default; pass ``columns`` to
    plot any other solution parameters, such as ``"rms_v"`` or ``"dtr_gps"``.
    The satellite counts share one panel, since reading them against the
    position errors is the point: an error spike usually lines up with a drop
    in satellites.

    Parameters
    ----------
    solution:
        Solution frame, as in ``SPPResult.solution``: epochs in a ``time``
        column, position errors in ``de``, ``dn`` and ``du``, plus per-epoch
        diagnostics.
    columns:
        Solution columns to plot, in order; a single column name may be given
        as a string. Columns that are absent or entirely
        empty are skipped, so a GPS-only run does not get blank Galileo panels.
        When ENU components are requested but none of them hold data, which is
        what a run without reference coordinates produces, a
        :class:`UserWarning` says so.
    show_satellites:
        Add a panel with the per-constellation satellite counts.
    time_col:
        Name of the epoch column. Falls back to the index when the column is
        absent and the index holds datetimes.
    color:
        Line colour of the parameter panels.
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
    KeyError
        When the epochs cannot be located.
    ValueError
        When none of the requested columns hold data.
    TypeError
        When ``columns`` is neither a column name nor a sequence of column
        names.

    Warns
    -----
    UserWarning
        When ENU components are requested but none of them hold data.
    """
    _check_dpi(dpi)  # before the figure exists, so a bad dpi leaves none open
    columns = _column_list(columns)
    times = resolve_time_axis(solution, time_col)

    present = [
        column
        for column in columns
        if column in solution.columns and _has_data(solution[column])
    ]
    counts = (
        [
            (column, label, count_color)
            for column, label, count_color in _SATELLITE_COUNTS
            if column in solution.columns and solution[column].fillna(0).abs().sum() > 0
        ]
        if show_satellites
        else []
    )

    if not present and not counts:
        raise ValueError(
            f"None of the requested columns hold data: {list(columns)}. "
            f"Available: {list(solution.columns)}."
        )

    # de/dn/du are only filled when the run had reference coordinates; without
    # them the figure would silently lose its position panels.
    requested_enu = [column for column in columns if column in _DEFAULT_COMPONENTS]
    if requested_enu and not any(column in present for column in requested_enu):
        warnings.warn(
            f"None of the position components {requested_enu} hold data, so no "
            "position panels are drawn. The SPP run probably had no reference "
            "coordinates (e.g. no SINEX solution for the station).",
            UserWarning,
            stacklevel=2,
        )

    fig, axes = make_fig_axes(len(present) + (1 if counts else 0), figsize=figsize)

    for index, column in enumerate(present):
        axes[index].plot(
            times,
            solution[column],
            label=_SOLUTION_LABELS.get(column, column),
            color=color,
        )
        axes[index].grid()
        axes[index].legend(loc="best", fontsize=9)
        format_time_axis(axes[index], autofmt=False)

    if counts:
        count_ax = axes[-1]
        for column, label, count_color in counts:
            count_ax.plot(times, solution[column], label=label, color=count_color, lw=1.0)
        count_ax.set_ylabel("Satellites [-]")
        count_ax.grid()
        count_ax.legend(loc="best", fontsize=9, ncol=len(counts))
        format_time_axis(count_ax, autofmt=False)

    axes[-1].set_xlabel(xlabel)
    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return finalize(fig, save_path=save_path, dpi=dpi, close=close)
