"""Plotting for GNX results.

The package mirrors the GNX processing modules: :mod:`gnx_py.viz.spp` and
:mod:`gnx_py.viz.ppp` plot positioning solutions and residuals,
:mod:`gnx_py.viz.orbits` plots orbit and SIS(RE) errors, and
:mod:`gnx_py.viz.ionosphere` plots STEC, VTEC and the activity indexes.
Projected maps of any quantity live in :mod:`gnx_py.viz.maps`, which is the
only module that uses Cartopy; :mod:`gnx_py.viz.schematics` holds the
explanatory diagrams and :mod:`gnx_py.viz.tables` the Markdown frame previews.

Three conventions hold across the package:

* **Every plot returns a** :class:`~matplotlib.figure.Figure`. Nothing calls
  ``plt.show()`` and nothing closes a figure unless asked with
  ``close=True``, which every plot accepts, so notebooks render by returning
  and scripts save through ``save_path`` (saved with ``bbox_inches="tight"``).
  The exceptions: :func:`~gnx_py.viz.maps.plot_roti_snapshot` returns None for
  an epoch with no data so that a snapshot loop survives gaps,
  :func:`~gnx_py.viz.maps.compare_vtec_maps` returns ``(fig, mean, median)``,
  and :meth:`~gnx_py.viz.ionosphere.STECModelComparator.report` returns an
  :class:`~gnx_py.viz.ionosphere.STECComparisonReport` named tuple.
* **Single-axes plots accept** ``ax``. Pass one to compose a plot into a larger
  layout; leave it out and the plot creates its own figure. Multi-panel plots
  own their layout and take no ``ax``.
* **Styling is opt-in.** Importing this package does not touch matplotlib's
  global settings; call
  :func:`~gnx_py.viz.core.apply_publication_style` when you want them.

The table previews :func:`~gnx_py.viz.tables.df_head` and
:func:`~gnx_py.viz.tables.df_tail` display their table and return None; pass
``show=False`` to get the Markdown string instead.

The ionosphere map functions are also re-exported by
:mod:`gnx_py.ionosphere.monitoring`, where they used to be defined, so
``gnx_py.plot_gix`` and friends are still importable under their old names.
They now return a single figure where they used to return tuples; see the
package README.

Examples
--------
>>> from gnx_py.viz import apply_publication_style, plot_positioning_series
>>> apply_publication_style()
>>> fig = plot_positioning_series(result.solution, title="PPP")  # doctest: +SKIP
>>> fig.savefig("ppp.png", dpi=300)  # doctest: +SKIP
"""
from __future__ import annotations

from .core import (
    DPI,
    FIGSIZE,
    GNSS_SYSTEM_NAMES,
    SERIES_FIGSIZE,
    TIME_FORMAT,
    add_prn_legend,
    apply_publication_style,
    categorical_palette,
    constellation_label,
    finalize,
    format_time_axis,
    make_fig_ax,
    make_fig_axes,
    make_map_axes,
    place_cax_below,
    require_cartopy,
    resolve_time_axis,
    to_utc_naive,
)
from .ionosphere import (
    STECComparisonReport,
    STECModelComparator,
    plot_gix_family,
    plot_point_timeseries,
    plot_roti_series,
    plot_sidx_series,
    plot_stec,
    plot_stec_arcs,
    plot_vtec_error_series,
    plot_vtec_zenith,
)
from .maps import (
    compare_vtec_maps,
    make_roti_snapshots,
    plot_gix,
    plot_grid_map,
    plot_roti_snapshot,
    plot_vtec_map,
)
from .orbits import plot_sisre, plot_sv_stats
from .ppp import plot_positioning_series, plot_residuals, plot_uduc_residuals
from .schematics import plot_gix_two_points, plot_stec_change
from .spp import plot_spp_positioning, plot_spp_residuals
from .tables import df_head, df_tail

__all__ = [
    "DPI",
    "FIGSIZE",
    "GNSS_SYSTEM_NAMES",
    "SERIES_FIGSIZE",
    "STECComparisonReport",
    "STECModelComparator",
    "TIME_FORMAT",
    "add_prn_legend",
    "apply_publication_style",
    "categorical_palette",
    "compare_vtec_maps",
    "constellation_label",
    "df_head",
    "df_tail",
    "finalize",
    "format_time_axis",
    "make_fig_ax",
    "make_fig_axes",
    "make_map_axes",
    "make_roti_snapshots",
    "place_cax_below",
    "plot_gix",
    "plot_gix_family",
    "plot_gix_two_points",
    "plot_grid_map",
    "plot_point_timeseries",
    "plot_positioning_series",
    "plot_residuals",
    "plot_roti_series",
    "plot_roti_snapshot",
    "plot_sidx_series",
    "plot_sisre",
    "plot_spp_positioning",
    "plot_spp_residuals",
    "plot_stec",
    "plot_stec_arcs",
    "plot_stec_change",
    "plot_sv_stats",
    "plot_uduc_residuals",
    "plot_vtec_error_series",
    "plot_vtec_map",
    "plot_vtec_zenith",
    "require_cartopy",
    "resolve_time_axis",
    "to_utc_naive",
]
