"""Projected maps of ionospheric quantities.

Every Cartopy-backed plot in GNX lives here, so the optional projection stack is
imported in exactly one module and only when a map is actually drawn. The
non-projected counterparts (time series, scatter comparisons) are in
:mod:`gnx_py.viz.ionosphere`.

All maps share the layout from :func:`gnx_py.viz.core.make_map_axes`: the map
with a horizontal colorbar snapped underneath. Maps of different quantities
therefore stack and compare cleanly.

The GNX-internal computations these plots need (the ROTI grid, the global colour
scale) are imported inside the functions rather than at module level. That keeps
``gnx_py.viz`` free of an import cycle with :mod:`gnx_py.ionosphere.monitoring`,
which re-exports the map functions defined here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .core import (
    DPI,
    FIGSIZE,
    _check_dpi,
    _check_real,
    _fill_bounds,
    _resolve_cmap,
    finalize,
    make_map_axes,
    no_figure_leak,
    place_cax_below,
    require_cartopy,
    resolve_time_axis,
    to_utc_naive,
)

LOGGER = logging.getLogger(__name__)

# Fixed figure margins. The map layout deliberately avoids tight_layout, which
# fights with Cartopy's own axes resizing.
_MARGINS = {"left": 0.06, "right": 0.98, "top": 0.94, "bottom": 0.08}

# Basemap styling shared by the ionosphere maps.
_LAND_FACECOLOR = "0.9"
_OCEAN_FACECOLOR = "0.96"
_ROTI_LAND_FACECOLOR = "#ececec"
_ROTI_OCEAN_FACECOLOR = "#dfe9f3"

# ROTI layering: Cartopy puts land and ocean at zorder -1, the ROTI cells sit
# at 1, and coastlines and borders at Cartopy's own default of 1.5 so they stay
# visible on top of the cells. Gridlines (Cartopy default 2) stay above both.
_ROTI_LAYER_ZORDER = 1
_ROTI_FEATURE_ZORDER = 1.5

# Draw order for gridlines that must stay readable on top of a filled mesh
# (the meshes are drawn at zorder 3).
_GRIDLINES_OVER_MESH_ZORDER = 4

_STATS_ANCHORS = {
    "lower left": (0.02, 0.02),
    "lower right": (0.98, 0.02),
    "upper left": (0.02, 0.98),
    "upper right": (0.98, 0.98),
}

# Renderings of the ROTI cells (see _draw_roti_layer).
_ROTI_DRAW = ("tiles", "scatter")

# Relative gap given to a log colour range whose derived bound had to be moved.
_LOG_MIN_RATIO = 1e-3

# Columns plot_gix reads from the pair frame (the epochs may also sit in the index).
_GIX_COLUMNS = ("CP_lat", "CP_lon", "grad_abs_mtecu_km")

__all__ = [
    "compare_vtec_maps",
    "make_roti_snapshots",
    "plot_gix",
    "plot_grid_map",
    "plot_roti_snapshot",
    "plot_vtec_map",
]


def _add_basemap(
    ax,
    cfeature,
    *,
    resolution: str,
    land_facecolor: str | None = _LAND_FACECOLOR,
    ocean_facecolor: str | None = _OCEAN_FACECOLOR,
    coastlines: bool = True,
    coast_linewidth: float = 0.7,
    borders: bool = True,
    borders_linewidth: float = 0.5,
    feature_zorder: float = 4,
) -> None:
    """Draw land, ocean, coastlines and borders on a projected axes.

    Land and ocean keep Cartopy's default zorder of -1, below any data layer;
    ``feature_zorder`` only applies to the coastline and border lines.
    """
    if land_facecolor is not None:
        ax.add_feature(
            cfeature.LAND.with_scale(resolution), facecolor=land_facecolor, edgecolor="none"
        )
    if ocean_facecolor is not None:
        ax.add_feature(
            cfeature.OCEAN.with_scale(resolution), facecolor=ocean_facecolor, edgecolor="none"
        )
    if coastlines:
        ax.coastlines(resolution=resolution, linewidth=coast_linewidth, zorder=feature_zorder)
    if borders:
        ax.add_feature(
            cfeature.BORDERS.with_scale(resolution),
            linewidth=borders_linewidth,
            zorder=feature_zorder,
        )


def _add_gridlines(
    ax,
    *,
    linestyle: str = "--",
    linewidth: float = 0.4,
    alpha: float = 0.7,
    zorder: float | None = None,
):
    """Draw labelled gridlines without the crowded top and right labels.

    ``zorder`` is set on the Gridliner itself: Cartopy draws the grid lines
    when the Gridliner is drawn, so a ``zorder`` passed to ``ax.gridlines``
    only reaches the line collections and does not lift them above a mesh.
    None keeps Cartopy's default of 2.
    """
    gl = ax.gridlines(draw_labels=True, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    gl.top_labels = False
    gl.right_labels = False
    if zorder is not None:
        gl.set_zorder(zorder)
    return gl


def _attach_colorbar(
    fig: Figure, mappable, ax, cax, label: str, *, top: float | None = None
) -> None:
    """Put a horizontal colorbar under the map and pin it to the map's width.

    ``top`` overrides the top figure margin, for titles taller than one line.
    """
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.set_label(label)
    fig.subplots_adjust(**{**_MARGINS, **({"top": top} if top is not None else {})})
    # Cartopy resizes the map axes when the extent is applied, so the colorbar
    # can only be positioned once everything else is in place.
    place_cax_below(ax, cax)


def _fit_title_top(fig: Figure, ax, *, min_fontsize: float = 6.0) -> float:
    """Fit the map title onto the canvas and return the top margin it needs.

    A title wider than the figure is shrunk until it fits; this assumes the
    standard ``_MARGINS`` left and right, as :func:`_attach_colorbar` sets
    them. The returned top margin, a figure fraction, leaves room for every
    line of the title plus the title pad; it never exceeds the standard
    ``_MARGINS["top"]``. Font sizes are in points and the figure size in
    inches, so the result holds at any save dpi.
    """
    # Only Agg-based canvases (inline, macosx, Agg) have a renderer to ask
    # for. Vector canvases (pdf, svg, ps) do not, and Figure._get_renderer
    # would reset fig.dpi to 72 there. So measure with a throwaway Agg
    # renderer at the figure's own dpi, which keeps the units of fig.bbox.
    if hasattr(fig.canvas, "get_renderer"):
        renderer = fig.canvas.get_renderer()
    else:
        from matplotlib.backends.backend_agg import RendererAgg

        renderer = RendererAgg(fig.bbox.width, fig.bbox.height, fig.dpi)
    title = ax.title
    width = title.get_window_extent(renderer).width
    # The title is centred on the map, which the margins centre slightly off
    # the figure centre, so it can use twice the narrower side.
    centre = 0.5 * (_MARGINS["left"] + _MARGINS["right"])
    available = 0.98 * 2.0 * min(centre, 1.0 - centre) * fig.bbox.width
    if width > available:
        title.set_fontsize(max(min_fontsize, title.get_fontsize() * available / width))
    height = title.get_window_extent(renderer).height
    # Title pad plus a couple of points of breathing room, in pixels.
    pad = (mpl.rcParams["axes.titlepad"] + 2.0) * fig.dpi / 72.0
    return min(_MARGINS["top"], 1.0 - (height + pad) / fig.bbox.height)


def _projection_names(ccrs) -> list[str]:
    """Names of the Cartopy projections that can be built from a name alone.

    These are the public :class:`cartopy.crs.Projection` subclasses whose
    constructor needs no argument, so a projection such as ``UTM`` (which
    needs a zone) is left out.
    """
    import inspect

    names = []
    for name, obj in vars(ccrs).items():
        if name.startswith("_") or not isinstance(obj, type):
            continue
        if not issubclass(obj, ccrs.Projection) or obj is ccrs.Projection:
            continue
        parameters = list(inspect.signature(obj).parameters.values())
        if all(
            p.default is not inspect.Parameter.empty
            or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in parameters
        ):
            names.append(name)
    return sorted(names)


def _region_limits(
    values: np.ndarray, lons: np.ndarray, lats: np.ndarray, region: Sequence[float]
) -> tuple[float | None, float | None]:
    """Minimum and maximum of a ``(lat, lon)`` field inside a region.

    Longitudes are compared modulo 360, so a 0..360 grid and a -180..180
    region (or the other way round) still match. Returns
    ``(None, None)`` when no finite value falls inside the region, which leaves
    the colour scale to matplotlib's whole-field default.
    """
    lon_min, lon_max, lat_min, lat_max = (float(v) for v in region)
    lon_span = (lon_max - lon_min) % 360.0 or 360.0
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    in_lon = ((lons - lon_min) % 360.0) <= lon_span
    in_lat = (lats >= lat_min) & (lats <= lat_max)
    inside = np.asarray(values, dtype=float)[np.ix_(in_lat, in_lon)]
    inside = inside[np.isfinite(inside)]
    if inside.size == 0:
        return None, None
    return float(inside.min()), float(inside.max())


def _auto_extent(lon: np.ndarray, lat: np.ndarray, res_deg: float) -> list[float]:
    """Bound the data with a margin that scales with the grid resolution."""
    margin = max(2.0, res_deg * 2)
    return [
        float(np.nanmin(lon) - margin),
        float(np.nanmax(lon) + margin),
        max(-90.0, float(np.nanmin(lat) - margin)),
        min(90.0, float(np.nanmax(lat) + margin)),
    ]


def _as_number(value: object) -> object:
    """Unwrap a zero-dimensional numeric array into its Python number.

    A bound or a coordinate is often taken straight from the data, for example
    ``ds.V.min()`` (a 0-d :class:`xarray.DataArray`) or ``ds.V.min().values``
    (a 0-d :class:`numpy.ndarray`). Those are unwrapped so the number checks
    accept them, as the maps always did. Booleans and everything else
    (None, sequences, strings, other objects) are returned unchanged and left
    to the checks.
    """
    if value is None or isinstance(value, (str, bytes)):
        return value
    try:
        if np.ndim(value) != 0:
            return value
        array = np.asarray(value)
    except Exception:  # not array-like: left to the checks, which name the setting
        return value
    if array.ndim == 0 and array.dtype.kind in "iuf":
        return array.item()
    return value


def _check_extent(extent: object, *, name: str = "extent") -> list[float] | None:
    """Return ``extent`` as four floats, or raise ValueError naming ``name``.

    An extent is ``[lon_min, lon_max, lat_min, lat_max]`` in degrees: four
    finite real numbers (zero-dimensional numeric arrays included). None
    passes through (the map fits the data).
    """
    if extent is None:
        return None
    try:
        values = list(extent)  # type: ignore[call-overload]
    except TypeError:
        values = None
    if values is None or len(values) != 4:
        raise ValueError(
            f"{name} must be [lon_min, lon_max, lat_min, lat_max] in degrees (got {extent!r})"
        )
    return [_check_real(f"{name}[{i}]", _as_number(value)) for i, value in enumerate(values)]


def _check_extent_in_domain(
    ccrs,
    projection,
    extent: Sequence[float],
    *,
    name: str = "extent",
    derived_from: str | None = None,
) -> None:
    """Raise ValueError when ``extent`` cannot be shown in ``projection``.

    :meth:`cartopy.mpl.geoaxes.GeoAxes.set_extent` projects the outline of the
    extent into the map coordinates. For a projection with a limited domain
    (such as ``OSGB`` or ``OSNI``) an extent outside that domain projects to
    an empty geometry, and matplotlib then fails with "Axis limits cannot be
    NaN or Inf" after the figure exists. This runs the same projection first,
    so the error names the extent and the projection and no figure is created.
    Anything other than a :class:`cartopy.crs.Projection` is left to the axes
    constructor. ``derived_from`` names the source of an extent the caller
    did not give (for example ``"the grid"``), so the message does not blame
    an argument that was never passed.
    """
    if not isinstance(projection, ccrs.Projection):
        return
    import shapely.geometry as sgeom

    x1, x2, y1, y2 = extent
    outline = sgeom.LineString([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    plate_carree = ccrs.PlateCarree()
    # set_extent's own shortcut for an extent equal to the projection's boundary
    if plate_carree == projection and projection.boundary.equals(outline):
        return
    projected = projection.project_geometry(outline, plate_carree)
    bounds = np.asarray(projected.bounds, dtype=float)
    if projected.is_empty or bounds.size != 4 or not np.all(np.isfinite(bounds)):
        values = [float(v) for v in extent]
        projection_name = type(projection).__name__
        if derived_from is None:
            raise ValueError(
                f"{name} {values} lies outside the valid domain of projection "
                f"{projection_name!r}; use an extent inside it or another projection"
            )
        raise ValueError(
            f"the extent {values} of {derived_from} (no {name} was given) cannot be shown in "
            f"projection {projection_name!r}: it lies outside the projection's valid domain; "
            f"pass a {name} inside it or use another projection"
        )


def _require_var(ds: object, var: str, *, name: str = "var", dataset: str = "ds") -> None:
    """Raise KeyError unless ``var`` is a data variable of ``ds``."""
    if var not in ds.data_vars:
        raise KeyError(
            f"Variable {var!r} ({name}) not found in {dataset}. Available: {list(ds.data_vars)}."
        )


def _field_at(ds: object, var: str, t_index: int):
    """The 2-D field of ``var`` to draw, and the epoch label for the title.

    A variable with a ``time`` dimension is taken at ``t_index``; a variable
    without one (a single map) is drawn as it is, and then only ``t_index`` 0
    or -1 is accepted. The label is the epoch of the selected field, or an
    empty string when the field carries no single epoch.
    """
    data = ds[var]
    if "time" in data.dims:
        field = data.isel(time=t_index)
    else:
        if t_index not in (0, -1):
            raise IndexError(
                f"t_index={t_index!r} is out of range: {var!r} has no time dimension "
                "(a single map), so only t_index 0 or -1 selects it"
            )
        field = data
    label = ""
    if "time" in field.coords and field["time"].ndim == 0:
        label = str(field["time"].values[()])
    return field, label


def _finite_limits(values: object) -> tuple[float | None, float | None]:
    """Minimum and maximum of the finite values, or ``(None, None)`` when there are none."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def _field_bounds(vmin, vmax, values, *, auto: tuple[float | None, float | None] = (None, None)):
    """Colour bounds of a field map (linear scale), keeping every explicit bound.

    ``auto`` are the automatic bounds the map uses for a missing one (for
    example the limits inside a region); None leaves matplotlib to autoscale
    it, as before. When exactly one bound is given and no automatic value is
    known for the other, the field's own finite minimum or maximum is used,
    which is what matplotlib's autoscale gives, so a contradiction with the
    given bound is caught before the figure and only the derived
    bound moves.
    """
    auto_lo, auto_hi = auto
    if (vmin is None) != (vmax is None) and auto_lo is None and auto_hi is None:
        auto_lo, auto_hi = _finite_limits(values)
    return _fill_bounds(vmin, vmax, auto_lo, auto_hi, strict=False)


def _check_log_bound(name: str, value: object) -> float | None:
    """An explicit bound of a log colour scale: a finite number > 0, or None."""
    if value is None:
        return None
    value = _check_real(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0 with log_scale=True (got {value:g})")
    return value


def _log_bounds(
    vmin: float | None,
    vmax: float | None,
    values: np.ndarray,
    lo_pct: float,
    hi_pct: float,
) -> tuple[float, float]:
    """Complete a logarithmic colour range from the data, keeping every explicit bound.

    The explicit bounds are already checked (> 0, ``vmin < vmax``). A missing
    bound is the data percentile; a percentile <= 0 is replaced by the
    smallest (for ``vmin``) or largest (for ``vmax``) positive value. When that
    leaves the range empty, only the derived bound moves, by a relative
    ``_LOG_MIN_RATIO``.
    """
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError(
            "log_scale=True requires positive grad_abs_mtecu_km values; none found for this epoch."
        )
    lo, hi = vmin, vmax
    if lo is None:
        lo = float(np.nanpercentile(values, lo_pct))
        if lo <= 0:
            lo = float(np.nanmin(positive))
    if hi is None:
        hi = float(np.nanpercentile(values, hi_pct))
        if hi <= 0:
            hi = float(np.nanmax(positive))
    if lo >= hi:
        if vmin is not None:  # explicit vmin above the data: the derived vmax moves
            hi = lo * (1.0 + _LOG_MIN_RATIO)
        elif vmax is not None:  # explicit vmax below the data: the derived vmin moves
            lo = hi / (1.0 + _LOG_MIN_RATIO)
        else:  # both derived (one positive value, or a percentile between 0 and the first positive)
            hi = max(float(np.nanmax(positive)), lo * (1.0 + _LOG_MIN_RATIO))
    return lo, hi


def _check_out_dir(out_dir: object) -> Path:
    """Return ``out_dir`` as a Path to an existing directory (no directory is created)."""
    path = Path(out_dir)  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(
            f"out_dir {str(path)!r} does not exist; create it first (frames are never written "
            "to a directory GNX created on its own)"
        )
    if not path.is_dir():
        raise NotADirectoryError(f"out_dir {str(path)!r} is not a directory")
    return path


def _frame_paths(out_dir: Path, filename_fmt: object, epochs) -> list[Path]:
    """The frame path of every epoch, built before any frame is drawn.

    Raises ValueError naming ``filename_fmt`` when it cannot be formatted with
    an epoch or when two epochs map to the same file name (a pattern without
    an epoch field, or one too coarse for the epoch spacing).
    """
    if not isinstance(filename_fmt, str):
        raise TypeError(
            f"filename_fmt must be a format string such as 'roti_{{:%Y%m%d_%H%M%S}}.png' "
            f"(got {type(filename_fmt).__name__})"
        )
    names = []
    for epoch in epochs:
        try:
            names.append(filename_fmt.format(pd.to_datetime(epoch).to_pydatetime()))
        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"filename_fmt {filename_fmt!r} cannot be formatted with an epoch: {exc!r}; "
                "use one positional field such as 'roti_{:%Y%m%d_%H%M%S}.png'"
            ) from None
    paths = [out_dir / name for name in names]
    unique = len(set(paths))
    if unique < len(paths):
        raise ValueError(
            f"filename_fmt {filename_fmt!r} maps {len(paths)} epochs to {unique} file name(s); "
            "every frame needs its own name (include the epoch down to the frame spacing)"
        )
    return paths


def _grid_summary(field) -> str:
    """One-line description of the lat/lon nodes of a field, for error messages."""
    parts = []
    for coord in ("lat", "lon"):
        if coord not in field.coords:
            parts.append(f"no {coord} coordinate")
            continue
        values = np.asarray(field[coord].values, dtype=float).ravel()
        if values.size == 0:
            parts.append(f"{coord}: 0 nodes")
        else:
            # full precision: grids that differ only by float noise must not look equal
            parts.append(f"{coord} {values.min():.15g}..{values.max():.15g}, {values.size} nodes")
    return "; ".join(parts)


def _draw_roti_layer(ax, grid: pd.DataFrame, *, draw: str, res_deg: float, cmap, norm, transform=None):
    """Draw the ROTI grid either as filled cells or as square markers.

    Both renderings sit at ``_ROTI_LAYER_ZORDER``: above land and ocean, below
    coastlines, borders and gridlines.
    """
    if draw == "tiles":
        for _, row in grid.iterrows():
            west = row["ix"] * res_deg - 180.0
            south = row["iy"] * res_deg - 90.0
            rectangle = mpatches.Rectangle(
                (west, south),
                res_deg,
                res_deg,
                facecolor=cmap(norm(row["ROTI"])),
                edgecolor="none",
                zorder=_ROTI_LAYER_ZORDER,
                **({"transform": transform} if transform is not None else {}),
            )
            ax.add_patch(rectangle)
    else:
        ax.scatter(
            grid["lon"],
            grid["lat"],
            c=grid["ROTI"],
            s=60,
            marker="s",
            cmap=cmap,
            norm=norm,
            edgecolor="none",
            linewidths=0,
            zorder=_ROTI_LAYER_ZORDER,
            **({"transform": transform} if transform is not None else {}),
        )
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


@no_figure_leak
def plot_roti_snapshot(
    roti_df: pd.DataFrame,
    epoch,
    window_min: int = 5,
    mode: Literal["center", "trailing"] = "center",
    res_deg: float = 1.0,
    min_points: int = 5,
    agg: Literal["median", "mean", "max", "Q95"] = "median",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | Path | None = None,
    title_prefix: str = "ROTI",
    map_background: bool = True,
    coast_res: str = "50m",
    draw: Literal["tiles", "scatter"] = "tiles",
    coord_mode: Literal["GEO", "SM"] = "GEO",
    extent: Sequence[float] | None = None,
    cmap_name: str = "viridis",
    *,
    figsize: tuple[float, float] = FIGSIZE,
    dpi: int = DPI,
    close: bool = False,
) -> Figure | None:
    """Plot one ROTI snapshot on a geographic or solar-magnetic grid.

    Plotting is intentionally separated from metric computation: the ROTI grid
    is built by :func:`gnx_py.ionosphere.monitoring.roti_snapshot_grid`, and an
    epoch with no data in its window is reported through the module logger
    rather than raised, so a snapshot loop survives gaps in the data.

    Parameters
    ----------
    roti_df:
        Per-link ROTI frame, as produced by the ionosphere monitoring tools.
    epoch:
        Epoch to plot, in the time scale of ``roti_df`` (naive GPST for
        ``compute_roti_links`` output; a timezone-aware epoch or frame is
        converted to GPST by
        :func:`~gnx_py.ionosphere.monitoring.roti_snapshot_grid`).
    window_min:
        Length of the averaging window, in minutes.
    mode:
        Place the window centred on, or trailing behind, ``epoch``.
    res_deg:
        Grid cell size, in degrees.
    min_points:
        Minimum number of links in a cell for it to be drawn.
    agg:
        Aggregation applied within a cell.
    vmin:
        Lower bound of the colour scale. None uses the 5th percentile of the
        cells.
    vmax:
        Upper bound of the colour scale. None uses the 95th percentile of the
        cells. A given bound is always kept: when the percentile of the other
        bound would leave an empty range, only that derived bound moves (to
        1e-3 TECU/min from the given one). Two given bounds must satisfy
        ``vmin < vmax``.
    save_path:
        When given, write the figure to this path.
    title_prefix:
        Text before the timestamp in the title.
    map_background:
        Draw land, ocean, coastlines and borders. Only applies to ``"GEO"``.
        Land and ocean go below the ROTI cells, coastlines and borders above
        them.
    coast_res:
        Cartopy feature resolution, such as ``"50m"``.
    draw:
        Render cells as filled tiles (``"tiles"``) or as square markers
        (``"scatter"``).
    coord_mode:
        Plot in geographic or solar-magnetic coordinates.
    extent:
        ``[lon_min, lon_max, lat_min, lat_max]`` in degrees. None fits the
        data. An extent outside the valid domain of the map projection
        (an extent entirely beyond +-90 deg latitude) is an error, not
        clipped.
    cmap_name:
        Name of a registered colormap (or a
        :class:`~matplotlib.colors.Colormap`).
    figsize:
        Figure size in inches.
    dpi:
        Figure resolution, also used when saving.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None when the window holds no data. The None case is
        what lets :func:`make_roti_snapshots` walk an epoch range that has
        gaps.

    Raises
    ------
    ValueError
        Before any figure exists: an unknown ``draw`` or ``cmap_name``, a
        bound that is not a finite number, ``vmin >= vmax`` when both are
        given, an ``extent`` that is not four finite numbers, and the
        settings checked by
        :func:`~gnx_py.ionosphere.monitoring.roti_snapshot_grid`; after the
        data are gridded, an ``extent`` outside the valid domain of the
        projection (entirely beyond +-90 deg latitude). A figure created
        before a later failure is closed.
    """
    from ..ionosphere.monitoring import roti_snapshot_grid
    from ..ionosphere.utils import _check_choice

    # 1. arguments
    _check_dpi(dpi)
    _check_choice("draw", draw, _ROTI_DRAW)
    cmap = _resolve_cmap(cmap_name, name="cmap_name")
    if cmap is None:  # None keeps matplotlib's default colormap, as plt.get_cmap(None) did
        cmap = mpl.colormaps[mpl.rcParams["image.cmap"]]
    vmin, vmax = _as_number(vmin), _as_number(vmax)
    _fill_bounds(vmin, vmax, strict=True)  # explicit bounds: finite and vmin < vmax
    extent = _check_extent(extent)
    # 2. data: roti_snapshot_grid validates window_min, mode, res_deg,
    # min_points, agg and coord_mode before any figure exists.
    grid = roti_snapshot_grid(
        roti_df,
        epoch=epoch,
        window_min=window_min,
        mode=mode,
        res_deg=res_deg,
        min_points=min_points,
        agg=agg,
        coord_mode=coord_mode,
    )
    if grid.empty:
        LOGGER.info("[IONO MONITOR] plot=ROTI event=no-points-in-window")
        return None

    # 3. colour range: explicit bounds are kept, only a percentile bound moves
    auto_lo = float(np.percentile(grid["ROTI"], 5)) if vmin is None else None
    auto_hi = float(np.percentile(grid["ROTI"], 95)) if vmax is None else None
    vmin, vmax = _fill_bounds(vmin, vmax, auto_lo, auto_hi, strict=True)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    lon = ((grid["lon"].to_numpy() + 180) % 360) - 180
    lat = grid["lat"].to_numpy()
    if extent is None:
        extent = _auto_extent(lon, lat, res_deg)

    try:
        ccrs, cfeature = require_cartopy()
    except ImportError:
        # Cartopy ships with GNX, but keep the unprojected fallback so a broken
        # projection stack degrades the map instead of losing it.
        fig, ax, cax = make_map_axes(None, figsize=figsize, dpi=dpi)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(alpha=0.3)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        mappable = _draw_roti_layer(
            ax, grid, draw=draw, res_deg=res_deg, cmap=cmap, norm=norm
        )
    else:
        projection = ccrs.PlateCarree()
        _check_extent_in_domain(ccrs, projection, extent)
        # 4. figure
        fig, ax, cax = make_map_axes(projection, figsize=figsize, dpi=dpi)
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())
        ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", linestyle="--", alpha=0.6
        )
        if map_background and coord_mode == "GEO":
            _add_basemap(
                ax,
                cfeature,
                resolution=coast_res,
                land_facecolor=_ROTI_LAND_FACECOLOR,
                ocean_facecolor=_ROTI_OCEAN_FACECOLOR,
                coast_linewidth=0.6,
                borders_linewidth=0.4,
                feature_zorder=_ROTI_FEATURE_ZORDER,
            )
        mappable = _draw_roti_layer(
            ax,
            grid,
            draw=draw,
            res_deg=res_deg,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

    _attach_colorbar(fig, mappable, ax, cax, "ROTI [TECU/min]")
    # roti_snapshot_grid took the epoch to naive GPST (and already warned for an
    # aware epoch), so the title shows that GPST epoch, not the input read as UTC.
    from ..ionosphere.monitoring import _epoch_to_gpst

    t0 = _epoch_to_gpst(epoch, who="plot_roti_snapshot", warn=False)
    ax.set_title(f"{title_prefix} — {t0:%Y-%m-%d %H:%M:%S} GPST  ({window_min} min window, {mode})")

    return finalize(fig, save_path=save_path, dpi=dpi, close=close)


@no_figure_leak
def make_roti_snapshots(
    roti_df: pd.DataFrame,
    epochs: pd.DatetimeIndex | None = None,
    every_n: int = 1,
    window_min: int = 5,
    mode: Literal["center", "trailing"] = "center",
    res_deg: float = 1.0,
    min_points: int = 5,
    agg: Literal["median", "mean", "max", "Q95"] = "median",
    global_scale: bool = True,
    q_scale: tuple[float, float] = (5.0, 95.0),
    out_dir: str | Path | None = None,
    filename_fmt: str = "roti_{:%Y%m%d_%H%M%S}.png",
    vmin: float | None = None,
    vmax: float | None = None,
    coord_mode: Literal["GEO", "SM"] = "GEO",
    extent: Sequence[float] | None = None,
) -> list[Path]:
    """Render a ROTI snapshot per epoch, for animation or review.

    Every frame is drawn by :func:`plot_roti_snapshot`, then saved on the
    full, uncropped canvas and closed. Frames therefore share one pixel size
    (``FIGSIZE`` times ``DPI``) whatever the extent, which video encoders
    need, and a long epoch range does not accumulate open figures.

    Parameters
    ----------
    roti_df:
        Per-link ROTI frame.
    epochs:
        Epochs to render. None uses the sorted unique epochs of ``roti_df``.
    every_n:
        Keep every ``every_n``-th epoch (an integer >= 1).
    window_min:
        Length of the averaging window, in minutes.
    mode:
        Place the window centred on, or trailing behind, each epoch.
    res_deg:
        Grid cell size, in degrees.
    min_points:
        Minimum number of links in a cell for it to be drawn.
    agg:
        Aggregation applied within a cell.
    global_scale:
        Derive one colour scale from the whole dataset, so frames are
        comparable. Only the bound that is not given comes from it: a given
        ``vmin`` or ``vmax`` is kept (if the global percentile of the other
        bound would leave an empty range, that derived bound moves to 1e-3
        TECU/min from the given one). Ignored when both are given.
    q_scale:
        Percentiles used for the global colour scale.
    out_dir:
        Existing directory for the frames; it is not created. None renders
        without writing files.
    filename_fmt:
        Frame filename pattern (a str), formatted with each epoch (one
        positional field, such as ``"roti_{:%Y%m%d_%H%M%S}.png"``). All names
        are built before the first frame is drawn and must differ from each
        other.
    vmin:
        Lower bound of the colour scale.
    vmax:
        Upper bound of the colour scale. Two given bounds must satisfy
        ``vmin < vmax``.
    coord_mode:
        Plot in geographic or solar-magnetic coordinates.
    extent:
        ``[lon_min, lon_max, lat_min, lat_max]`` in degrees, used for every
        frame. None fits each frame's data. An extent outside the valid
        domain of the map projection (an extent entirely beyond +-90 deg
        latitude) is an error.

    Returns
    -------
    list[pathlib.Path]
        Paths of the frames written, in order. Empty when ``out_dir`` is None
        or the input frame is empty.

    Raises
    ------
    FileNotFoundError
        When ``out_dir`` does not exist (NotADirectoryError when it is a
        file); checked before any frame is drawn.
    ValueError
        Before the first frame: an invalid option, a bound that is not a
        finite number, ``vmin >= vmax`` when both are given, an ``extent``
        that is not four finite numbers or lies outside the valid domain of
        the projection, a ``filename_fmt`` that cannot be formatted with an
        epoch or that gives two epochs the same file name, and (with
        ``global_scale``) a global percentile that is not finite where a
        bound is derived from it (pass that bound explicitly).
    TypeError
        When ``filename_fmt`` is not a str (with ``out_dir`` given), before
        the first frame.
    """
    from ..ionosphere.monitoring import _check_percentile_pair, compute_roti_global_scale
    from ..ionosphere.utils import _check_choice, _check_number

    # every option is checked up front, also when there is nothing to draw
    every_n = _check_number("every_n", every_n, ge=1, integer=True)
    _check_number("window_min", window_min, ge=0, unit="min")
    _check_choice("mode", mode, ["center", "trailing"])
    _check_number("res_deg", res_deg, gt=0, unit="deg")
    _check_number("min_points", min_points, ge=1, integer=True)
    _check_choice("agg", agg, ["median", "mean", "max", "Q95"])
    _check_choice("coord_mode", coord_mode, ["GEO", "SM"])
    if global_scale:
        _check_percentile_pair("q_low_high (q_scale)", q_scale)
    vmin, vmax = _as_number(vmin), _as_number(vmax)
    _fill_bounds(vmin, vmax, strict=True)  # explicit bounds: finite and vmin < vmax
    extent = _check_extent(extent)
    if extent is not None:
        # every frame uses this extent in PlateCarree: checked once, before any frame
        try:
            ccrs, _ = require_cartopy()
        except ImportError:  # the unprojected fallback of plot_roti_snapshot draws any extent
            pass
        else:
            _check_extent_in_domain(ccrs, ccrs.PlateCarree(), extent)
    out_path = _check_out_dir(out_dir) if out_dir is not None else None
    if roti_df.empty:
        LOGGER.info("[IONO MONITOR] plot=ROTI event=no-data")
        return []

    if epochs is None:
        epochs = pd.DatetimeIndex(
            np.sort(pd.to_datetime(roti_df["time"].dropna().unique()))
        )
    epochs = epochs[::every_n]
    # every frame name is known (and unique) before the first frame is drawn
    save_paths: list[Path | None] = (
        list(_frame_paths(out_path, filename_fmt, epochs)) if out_path is not None
        else [None] * len(epochs)
    )

    if (vmin is None or vmax is None) and global_scale:
        # the global scale fills only the bound that is not given (NQ-VIZ-7)
        global_lo, global_hi = compute_roti_global_scale(
            roti_df, window_min=window_min, q_low_high=q_scale
        )
        vmin, vmax = _fill_bounds(vmin, vmax, global_lo, global_hi, strict=True)

    written: list[Path] = []
    for epoch, save_path in zip(epochs, save_paths):
        figure = plot_roti_snapshot(
            roti_df,
            epoch=epoch,
            window_min=window_min,
            mode=mode,
            res_deg=res_deg,
            min_points=min_points,
            agg=agg,
            vmin=vmin,
            vmax=vmax,
            map_background=True,
            coord_mode=coord_mode,
            extent=extent,
        )
        if figure is None:
            continue
        # Frames are saved on the full canvas, not cropped: every frame then
        # has the same, even pixel size, which video encoders require.
        finalize(figure, save_path=save_path, dpi=DPI, bbox_inches=None, close=True)
        if save_path is not None:
            written.append(save_path)

    return written


def _naive_epoch(epoch) -> pd.Timestamp:
    """One epoch as a naive Timestamp: numbers raise TypeError, aware values go to UTC."""
    if isinstance(epoch, (bool, np.bool_, int, float, np.integer, np.floating)):
        raise TypeError(f"epoch is numeric ({epoch!r}), not a datetime.")
    try:
        ts = pd.Timestamp(epoch)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"epoch {epoch!r} cannot be read as a datetime.") from exc
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


@no_figure_leak
def plot_gix(
    pairs_df: pd.DataFrame,
    epoch,
    clip_outliers: bool = True,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    vclip: tuple[float, float] = (0.01, 0.995),
    log_scale: bool = False,
    cmap: str = "viridis",
    projection: str = "PlateCarree",
    central_longitude: float = 0.0,
    extent: Sequence[float] | None = None,
    figsize: tuple[float, float] = FIGSIZE,
    dpi: int = DPI,
    point_size: float = 16,
    point_alpha: float = 0.8,
    point_edgecolor: str = "none",
    point_zorder: int = 3,
    draw_coastlines: bool = True,
    coast_resolution: str = "110m",
    coast_linewidth: float = 0.7,
    draw_borders: bool = True,
    borders_linewidth: float = 0.5,
    draw_land: bool = True,
    land_facecolor: str = "0.9",
    ocean_facecolor: str = "0.95",
    draw_ocean: bool = True,
    draw_gridlines: bool = True,
    gridlines_kwargs: dict | None = None,
    title_prefix: str = r"GIX |$\nabla$TEC|",
    title_kwargs: dict | None = None,
    cbar_label: str = r"|$\nabla$TEC| [mTECU/km]",
    annotate_stats: bool = True,
    stats_loc: str = "lower left",
    stats_fontsize: float = 9,
    stats_alpha: float = 0.7,
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot the ionospheric gradient index of every pair for one epoch.

    Each point is the centre point of a station pair, coloured by the magnitude
    of the VTEC gradient across it. Points are drawn in ascending order of
    magnitude so the strongest gradients stay visible where pairs overlap.

    Parameters
    ----------
    pairs_df:
        Pair frame holding ``time``, ``CP_lat``, ``CP_lon`` and
        ``grad_abs_mtecu_km``.
    epoch:
        Epoch to plot. The epochs of ``pairs_df`` and ``epoch`` are both read
        with :func:`~gnx_py.viz.core.resolve_time_axis` (strings are parsed,
        numbers raise TypeError). When ``pairs_df.attrs["time_scale"]`` is
        ``"GPST"`` (the output of ``compute_gix``), the epoch is taken to naive
        GPST with the monitoring rule: naive values are GPST, timezone-aware ones
        are converted to UTC and then to GPST (+leap seconds, one UserWarning),
        as in :func:`~gnx_py.ionosphere.monitoring.roti_snapshot_grid`; the title
        shows the GPST epoch. Frames without that tag are compared on clock
        time: aware values are converted to UTC and the zone is dropped, naive
        ones are compared as they are.
    clip_outliers:
        Drop values above the 99th percentile before scaling the colours.
    vmin:
        Lower bound of the colour scale. None uses the ``vclip`` percentile.
    vmax:
        Upper bound of the colour scale. None uses the ``vclip`` percentile.
        A given bound is always kept; if the percentile of the other bound
        would leave an empty range, only that derived bound moves. Two given
        bounds must satisfy ``vmin <= vmax`` (``vmin < vmax`` and both > 0
        with ``log_scale``).
    vclip:
        Percentiles, as fractions ``0 <= low < high <= 1``, used for the
        automatic colour scale; checked before the figure is created.
    log_scale:
        Use a logarithmic colour scale. A derived bound <= 0 is replaced by
        the smallest (``vmin``) or largest (``vmax``) positive value; a given
        bound <= 0 is an error.
    cmap:
        Name of a registered colormap (or a
        :class:`~matplotlib.colors.Colormap`).
    projection:
        Name of a :mod:`cartopy.crs` projection class that needs no
        constructor argument, such as ``"PlateCarree"``, ``"Mercator"`` or
        ``"LambertConformal"``.
    central_longitude:
        Central longitude, for projections that accept one. It must be a
        finite number for every projection, also one that ignores it.
    extent:
        ``[lon_min, lon_max, lat_min, lat_max]`` in degrees, applied in any
        projection. None fits the data. An extent outside the valid domain of
        the projection (for example central Europe in ``"OSGB"``) is an
        error, not clipped.
    figsize:
        Figure size in inches.
    dpi:
        Figure resolution.
    point_size:
        Marker size.
    point_alpha:
        Marker opacity.
    point_edgecolor:
        Marker edge colour.
    point_zorder:
        Draw order of the markers.
    draw_coastlines:
        Draw coastlines.
    coast_resolution:
        Cartopy feature resolution.
    coast_linewidth:
        Coastline width.
    draw_borders:
        Draw country borders.
    borders_linewidth:
        Border width.
    draw_land:
        Fill land areas.
    land_facecolor:
        Land fill colour.
    ocean_facecolor:
        Ocean fill colour, applied as the axes background.
    draw_ocean:
        Fill the ocean background.
    draw_gridlines:
        Draw labelled gridlines.
    gridlines_kwargs:
        Overrides for the gridline styling.
    title_prefix:
        Text before the timestamp in the title.
    title_kwargs:
        Overrides for the title styling.
    cbar_label:
        Colorbar label.
    annotate_stats:
        Write min, median, max and the 95th percentile onto the map.
    stats_loc:
        Corner for the statistics box: ``"lower left"``, ``"lower right"``,
        ``"upper left"`` or ``"upper right"``.
    stats_fontsize:
        Font size of the statistics box.
    stats_alpha:
        Opacity of the statistics box.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the map.

    Raises
    ------
    ValueError
        All before the figure is created, in this order:

        * the arguments: ``vclip``, ``stats_loc`` not one of the four
          corners, ``projection`` not a supported Cartopy projection name
          (the message lists the supported names; checked before the pairs
          are matched), an unknown ``cmap``, a bound that is not a finite
          number, contradicting bounds, a bound <= 0 with ``log_scale``, an
          ``extent`` that is not four finite numbers;
        * the data: the epoch holds no pairs or no finite value, or
          ``log_scale`` is requested and no positive value remains;
        * an ``extent`` outside the valid domain of the projection.
    KeyError
        When ``pairs_df`` lacks ``CP_lat``, ``CP_lon`` or
        ``grad_abs_mtecu_km``.
    ImportError
        When Cartopy is not installed.
    """
    from matplotlib.colors import LogNorm

    from ..ionosphere.utils import _check_choice

    # 1. arguments
    _check_dpi(dpi)
    ccrs, cfeature = require_cartopy()

    try:
        lo_clip, hi_clip = (float(v) for v in vclip)
    except (TypeError, ValueError):
        raise ValueError(
            f"vclip must be a pair (low, high) of fractions in [0, 1], got {vclip!r}."
        ) from None
    if not (0.0 <= lo_clip < hi_clip <= 1.0):
        raise ValueError(
            f"vclip must be fractions 0 <= low < high <= 1, got {vclip!r}; "
            "percentiles such as (1, 99.5) are given as (0.01, 0.995)."
        )
    _check_choice("stats_loc", stats_loc, list(_STATS_ANCHORS))
    # the projection is an argument: checked before the pairs are matched, so a
    # bad name is reported as such and not as "No pairs found" (NQ-VIZ-8)
    supported = _projection_names(ccrs)
    if projection not in supported:
        raise ValueError(
            f"Unsupported projection {projection!r}. Supported: {', '.join(supported)}."
        )
    central_longitude = _as_number(central_longitude)
    _check_real("central_longitude", central_longitude)
    projection_cls = getattr(ccrs, projection)
    proj = (
        projection_cls(central_longitude=central_longitude)
        if "central_longitude" in projection_cls.__init__.__code__.co_varnames
        else projection_cls()
    )
    cmap = _resolve_cmap(cmap)
    vmin, vmax = _as_number(vmin), _as_number(vmax)
    if log_scale:
        vmin = _check_log_bound("vmin", vmin)
        vmax = _check_log_bound("vmax", vmax)
        if vmin is not None and vmax is not None and vmin >= vmax:
            raise ValueError(
                f"vmin must be < vmax with log_scale=True (got vmin={vmin:g}, vmax={vmax:g})"
            )
    else:
        vmin, vmax = _fill_bounds(vmin, vmax)  # explicit bounds: finite and vmin <= vmax
    extent = _check_extent(extent)

    # 2. data
    missing = [c for c in _GIX_COLUMNS if c not in pairs_df.columns]
    if missing:
        raise KeyError(
            f"pairs_df lacks the column(s) {missing}; plot_gix needs {list(_GIX_COLUMNS)} "
            "(the output of compute_gix)."
        )
    sub = pairs_df.dropna(subset=["CP_lat", "CP_lon"])
    gpst = pairs_df.attrs.get("time_scale") == "GPST"
    if gpst:
        # compute_gix output: naive GPST pair times, so the epoch is taken to GPST
        # with the monitoring rule (aware -> UTC -> +leap seconds), as in
        # roti_snapshot_grid. Lazy import: monitoring imports gnx_py.viz.
        from ..ionosphere.monitoring import _epoch_to_gpst, _to_gpst

        times = pd.Index(resolve_time_axis(sub, "time"))
        pair_times = _to_gpst(pd.Series(times), who="plot_gix").to_numpy()
        epoch_naive = _epoch_to_gpst(epoch, who="plot_gix")
    else:
        # frames without the GPST tag keep the clock-time rule: both sides naive UTC
        pair_times = np.asarray(to_utc_naive(resolve_time_axis(sub, "time")))
        epoch_naive = _naive_epoch(epoch)
    sub = sub[pair_times == np.datetime64(epoch_naive, "ns")]
    if sub.empty:
        raise ValueError(f"No pairs found for epoch {epoch!r}.")

    lats = sub["CP_lat"].to_numpy()
    lons = sub["CP_lon"].to_numpy()
    values = sub["grad_abs_mtecu_km"].to_numpy()
    if not np.isfinite(values).any():
        raise ValueError(f"No finite grad_abs_mtecu_km value for epoch {epoch!r}.")

    if clip_outliers:
        keep = values <= np.nanpercentile(values, 99)
        lats, lons, values = lats[keep], lons[keep], values[keep]

    # Draw weakest first so the strong gradients end up on top.
    order = np.argsort(values)
    lats, lons, values = lats[order], lons[order], values[order]

    # 3. colour range and extent: explicit bounds are kept, only a percentile
    # bound moves
    if log_scale:
        vmin, vmax = _log_bounds(vmin, vmax, values, lo_clip * 100, hi_clip * 100)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        auto_lo = float(np.nanpercentile(values, lo_clip * 100)) if vmin is None else None
        auto_hi = float(np.nanpercentile(values, hi_clip * 100)) if vmax is None else None
        vmin, vmax = _fill_bounds(vmin, vmax, auto_lo, auto_hi)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if extent is not None:
        _check_extent_in_domain(ccrs, proj, extent)

    # 4. figure
    fig, ax, cax = make_map_axes(proj, figsize=figsize, dpi=dpi)

    if extent is not None:
        # The extent is in degrees; set_extent converts it into the map's own
        # coordinates, which set_xlim/set_ylim would not.
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())

    if draw_ocean:
        ax.set_facecolor(ocean_facecolor)
    if draw_land:
        ax.add_feature(
            cfeature.LAND.with_scale(coast_resolution),
            facecolor=land_facecolor,
            edgecolor="none",
            zorder=0,
        )
    if draw_coastlines:
        ax.coastlines(resolution=coast_resolution, linewidth=coast_linewidth, zorder=1)
    if draw_borders:
        ax.add_feature(
            cfeature.BORDERS.with_scale(coast_resolution),
            linewidth=borders_linewidth,
            zorder=2,
        )
    if draw_gridlines:
        gridline_options = {
            "draw_labels": True,
            "linestyle": "--",
            "linewidth": 0.4,
            "alpha": 0.6,
        }
        if gridlines_kwargs is not None:
            gridline_options.update(gridlines_kwargs)
        ax.gridlines(**gridline_options)

    scatter = ax.scatter(
        lons,
        lats,
        c=values,
        s=point_size,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=point_alpha,
        edgecolor=point_edgecolor,
        linewidths=0,
        zorder=point_zorder,
    )

    _attach_colorbar(fig, scatter, ax, cax, cbar_label)

    title_options = {"fontsize": 12}
    if title_kwargs is not None:
        title_options.update(title_kwargs)
    epoch_label = f"{epoch_naive:%Y-%m-%d %H:%M:%S} GPST" if gpst else f"{pd.to_datetime(epoch)}"
    ax.set_title(f"{title_prefix} — {epoch_label}", **title_options)

    if annotate_stats:
        text = (
            f"min={float(np.nanmin(values)):.2f}, "
            f"median={float(np.nanmedian(values)):.2f}, "
            f"max={float(np.nanmax(values)):.2f}\n"
            f"p95={float(np.nanpercentile(values, 95)):.2f}"
        )
        x, y = _STATS_ANCHORS[stats_loc]
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=stats_fontsize,
            ha="left" if "left" in stats_loc else "right",
            va="bottom" if "lower" in stats_loc else "top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": stats_alpha,
            },
            zorder=4,
        )

    return finalize(fig, save_path=save_path, dpi=dpi, close=close)


@no_figure_leak
def plot_vtec_map(
    ds: object,
    var: str,
    *,
    t_index: int = 0,
    region: Sequence[float] | None = None,
    projection: object | None = None,
    data_crs: object | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    draw_coastlines: bool = True,
    draw_borders: bool = True,
    gridlines: bool = True,
    add_colorbar: bool = True,
    cbar_label: str | None = None,
    cmap: str = "plasma",
    coast_res: str = "50m",
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot a gridded ionospheric field on a projected map.

    Parameters
    ----------
    ds:
        :class:`xarray.Dataset` with ``lat`` and ``lon`` coordinates and,
        usually, a ``time`` dimension.
    var:
        Name of the variable to plot, such as ``"V"`` for VTEC.
    t_index:
        Index along the time axis. A variable without a ``time`` dimension (a
        single map) is drawn as it is; ``t_index`` must then be 0 or -1.
    region:
        ``[lon_min, lon_max, lat_min, lat_max]``. None fits the grid. A region
        outside the valid domain of ``projection`` is an error.
    projection:
        Cartopy CRS of the map axes. None uses ``PlateCarree``.
    data_crs:
        Cartopy CRS of the data. None uses ``PlateCarree``, which is right for
        regular latitude/longitude grids.
    vmin:
        Lower bound of the colour scale. None uses the field minimum, taken
        inside ``region`` when one is given and ``data_crs`` is
        ``PlateCarree``, otherwise over the whole field.
    vmax:
        Upper bound of the colour scale. None uses the field maximum, with
        the same rule as ``vmin``. A given bound is always kept; if the other,
        derived bound would leave an empty range, only that one moves. Two
        given bounds must satisfy ``vmin <= vmax``.
    title:
        Map title. None derives it from the variable name and epoch.
    draw_coastlines:
        Draw coastlines.
    draw_borders:
        Draw country borders.
    gridlines:
        Draw labelled gridlines.
    add_colorbar:
        Draw the colorbar under the map.
    cbar_label:
        Colorbar label. None uses the variable name and its ``units``
        attribute, when present.
    cmap:
        Name of a registered colormap (or a
        :class:`~matplotlib.colors.Colormap`).
    coast_res:
        Cartopy feature resolution.
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
        The figure holding the map.

    Raises
    ------
    KeyError
        When ``var`` is not a variable of ``ds``.
    IndexError
        When ``t_index`` is out of range.
    ValueError
        Before the figure is created: an unknown ``cmap``, a bound that is
        not a finite number, ``vmin > vmax`` when both are given, a
        ``region`` that is not four finite numbers or lies outside the valid
        domain of ``projection`` (without ``region``, the extent of the grid
        is checked the same way).
    ImportError
        When Cartopy is not installed.
    """
    # 1. arguments
    _check_dpi(dpi)
    ccrs, cfeature = require_cartopy()
    cmap = _resolve_cmap(cmap)
    vmin, vmax = _as_number(vmin), _as_number(vmax)
    _fill_bounds(vmin, vmax)  # explicit bounds: finite and vmin <= vmax
    region = _check_extent(region, name="region")

    # 2. data
    _require_var(ds, var)
    field, epoch_label = _field_at(ds, var, t_index)

    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    if projection is None:
        projection = ccrs.PlateCarree()

    lats = ds["lat"].values
    lons = ds["lon"].values

    # 3. colour range and extent
    auto = (None, None)
    region_source = None
    if region is None:
        region_source = "the grid"
        region = [
            float(np.min(lons)),
            float(np.max(lons)),
            float(np.min(lats)),
            float(np.max(lats)),
        ]
    elif (vmin is None or vmax is None) and isinstance(data_crs, ccrs.PlateCarree):
        # Scale the colours to what is visible, not to the whole field. This
        # needs the grid to be in degrees, hence the PlateCarree check.
        auto = _region_limits(field.values, lons, lats, region)
    vmin, vmax = _field_bounds(vmin, vmax, field.values, auto=auto)
    _check_extent_in_domain(ccrs, projection, region, name="region", derived_from=region_source)

    # 4. figure
    fig, ax, cax = make_map_axes(projection, figsize=figsize, dpi=dpi)
    ax.set_extent(list(region), crs=ccrs.PlateCarree())
    _add_basemap(
        ax,
        cfeature,
        resolution=coast_res,
        coastlines=draw_coastlines,
        borders=draw_borders,
    )

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        field.values,
        transform=data_crs,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        zorder=3,
    )

    if gridlines:
        _add_gridlines(ax)

    if title is None:
        title = f"{var} @ t={epoch_label}"
    ax.set_title(title, fontsize=12)

    if add_colorbar:
        if cbar_label is None:
            units = ds[var].attrs.get("units", "")
            cbar_label = f"{var} [{units}]" if units else var
        _attach_colorbar(fig, mesh, ax, cax, cbar_label)
    else:
        cax.set_visible(False)
        fig.subplots_adjust(**_MARGINS)

    return finalize(fig, save_path=save_path, dpi=dpi, close=close)


@no_figure_leak
def compare_vtec_maps(
    ds_model: object,
    ds_ref: object,
    time,
    *,
    var_model: str = "V",
    var_ref: str = "TEC",
    region: Sequence[float] | None = None,
    projection: object | None = None,
    data_crs: object | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Blues",
    add_colorbar: bool = True,
    cbar_label: str = r"|ΔTEC| [TECU]",
    title_prefix: str = r"|ΔTEC| = |TEC$_\mathrm{ref}$ - V$_\mathrm{model}$|",
    show_stats_in_title: bool = True,
    coast_res: str = "50m",
    figsize: tuple[float, float] = (6, 4),
    dpi: int = 120,
    save_path: str | Path | None = None,
    close: bool = False,
) -> tuple[Figure, float, float]:
    """Map the absolute difference between a modelled and a reference field.

    Parameters
    ----------
    ds_model:
        :class:`xarray.Dataset` holding the modelled field.
    ds_ref:
        :class:`xarray.Dataset` holding the reference field.
    time:
        Epoch to compare.
    var_model:
        Name of the modelled variable.
    var_ref:
        Name of the reference variable.
    region:
        ``[lon_min, lon_max, lat_min, lat_max]``. None fits the compared
        nodes. It sets the visible map only; the statistics always cover
        every common node. A region outside the valid domain of
        ``projection`` is an error.
    projection:
        Cartopy CRS of the map axes. None uses ``PlateCarree``.
    data_crs:
        Cartopy CRS of the data. None uses ``PlateCarree``.
    vmin:
        Lower bound of the colour scale. None lets matplotlib use the
        smallest difference.
    vmax:
        Upper bound of the colour scale. None lets matplotlib use the
        largest difference. A given bound is always kept; if the other bound,
        taken from the differences, would leave an empty range, only that one
        moves. Two given bounds must satisfy ``vmin <= vmax``.
    cmap:
        Name of a registered colormap (or a
        :class:`~matplotlib.colors.Colormap`).
    add_colorbar:
        Draw the colorbar under the map.
    cbar_label:
        Colorbar label.
    title_prefix:
        Text before the epoch and statistics in the title. The top margin
        grows to fit a multi-line title, and a title wider than the figure is
        drawn smaller, so the title stays on the canvas.
    show_stats_in_title:
        Append the mean and median absolute difference to the title, as
        ``MAE`` and ``MedAE``, in TECU.
    coast_res:
        Cartopy feature resolution.
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
    tuple[matplotlib.figure.Figure, float, float]
        The figure, the mean absolute difference and the median absolute
        difference, both in the units of the compared fields.

    Raises
    ------
    KeyError
        When ``var_model`` or ``var_ref`` is missing, or ``time`` is not an
        epoch of both datasets (exact match).
    ValueError
        Before the figure is created: an unknown ``cmap``, a bound that is
        not a finite number, ``vmin > vmax`` when both are given, a
        ``region`` that is not four finite numbers or lies outside the valid
        domain of ``projection`` (without ``region``, the extent of the
        common nodes is checked the same way), and two grids without a
        common node.
    ImportError
        When Cartopy is not installed.

    Notes
    -----
    The two fields are compared on their common grid nodes: the nodes whose
    latitude and longitude appear in both grids with exactly the same value
    (an inner join of the coordinates, as :mod:`xarray` aligns them; no
    tolerance). Longitudes are not wrapped, so a 0..360 deg grid and a
    -180..180 deg grid share only the nodes in 0..180 deg; bring both to one
    longitude convention first to compare the whole globe. The map and both statistics
    cover exactly those nodes, so a fine model grid can be checked against a
    coarser reference such as a 2.5 x 5 deg global ionosphere map where their
    nodes coincide. Nodes of only one grid are left out without interpolation;
    the module logger reports how many nodes each grid had and how many were
    compared. Two grids without a single common node raise ValueError naming
    both grids.
    """
    # 1. arguments
    _check_dpi(dpi)
    ccrs, cfeature = require_cartopy()
    cmap = _resolve_cmap(cmap)
    vmin, vmax = _as_number(vmin), _as_number(vmax)
    _fill_bounds(vmin, vmax)  # explicit bounds: finite and vmin <= vmax
    region = _check_extent(region, name="region")

    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    if projection is None:
        projection = ccrs.PlateCarree()

    # 2. data: the two fields on their common nodes
    _require_var(ds_model, var_model, name="var_model", dataset="ds_model")
    _require_var(ds_ref, var_ref, name="var_ref", dataset="ds_ref")
    model = ds_model[var_model].sel(time=np.datetime64(time))
    reference = ds_ref[var_ref].sel(time=np.datetime64(time))
    difference = np.abs(reference - model)
    if "time" in difference.dims:
        difference = difference.isel(time=0)
    if difference.size == 0:
        raise ValueError(
            f"compare_vtec_maps: the model grid ({_grid_summary(model)}) and the reference grid "
            f"({_grid_summary(reference)}) have no common lat/lon nodes; the maps are compared "
            "on their common nodes, whose coordinates must match exactly (no tolerance and no "
            "interpolation), so the grids must share at least one node"
        )
    n_common, n_model, n_ref = int(difference.size), int(model.size), int(reference.size)
    if n_common < max(n_model, n_ref):
        LOGGER.info(
            "[VTEC COMPARE] common grid nodes: %d (model %d, reference %d)", n_common, n_model, n_ref
        )

    mean_abs = float(difference.mean().values)
    median_abs = float(difference.median().values)

    lats = difference["lat"].values
    lons = difference["lon"].values
    region_source = None
    if region is None:
        region_source = "the common grid nodes"
        region = [
            float(np.min(lons)),
            float(np.max(lons)),
            float(np.min(lats)),
            float(np.max(lats)),
        ]

    # 3. colour range and extent
    vmin, vmax = _field_bounds(vmin, vmax, difference.values)
    _check_extent_in_domain(ccrs, projection, region, name="region", derived_from=region_source)

    # 4. figure
    fig, ax, cax = make_map_axes(projection, figsize=figsize, dpi=dpi)
    ax.set_extent(list(region), crs=ccrs.PlateCarree())
    _add_basemap(ax, cfeature, resolution=coast_res)
    # Dotted gridlines above the mesh, so the grid stays readable across the
    # difference field.
    _add_gridlines(ax, linestyle=":", linewidth=0.3, zorder=_GRIDLINES_OVER_MESH_ZORDER)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        difference.values,
        transform=data_crs,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        zorder=3,
    )

    epoch = np.datetime_as_string(np.datetime64(time), unit="m")
    title = f"{title_prefix}\nTime: {epoch}"
    if show_stats_in_title:
        title += f"  |  MAE={mean_abs:.3f} TECU, MedAE={median_abs:.3f} TECU"
    ax.set_title(title, fontsize=11)
    # The title spans two lines, more than the standard top margin holds.
    top = _fit_title_top(fig, ax)

    if add_colorbar:
        _attach_colorbar(fig, mesh, ax, cax, cbar_label, top=top)
    else:
        cax.set_visible(False)
        fig.subplots_adjust(**{**_MARGINS, "top": top})

    return finalize(fig, save_path=save_path, dpi=dpi, close=close), mean_abs, median_abs


@no_figure_leak
def plot_grid_map(
    ds: object,
    var: str,
    *,
    t_index: int = 0,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Plot a gridded field on plain latitude/longitude axes, without Cartopy.

    This is the quick look that works anywhere: no projection, no coastlines,
    no optional dependency. Use :func:`plot_vtec_map` for a real map.

    Parameters
    ----------
    ds:
        :class:`xarray.Dataset` with ``lat`` and ``lon`` coordinates.
    var:
        Name of the variable to plot.
    t_index:
        Index along the time axis. A variable without a ``time`` dimension (a
        single map) is drawn as it is; ``t_index`` must then be 0 or -1.
    title:
        Plot title. None derives it from the variable name and epoch.
    vmin:
        Lower bound of the colour scale. None lets matplotlib use the field
        minimum.
    vmax:
        Upper bound of the colour scale. None lets matplotlib use the field
        maximum. A given bound is always kept; if the other bound, taken from
        the field, would leave an empty range, only that one moves. Two given
        bounds must satisfy ``vmin <= vmax``.
    cmap:
        Name of a registered colormap (or a
        :class:`~matplotlib.colors.Colormap`). None uses the matplotlib
        default.
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
        The figure holding the plot.

    Raises
    ------
    KeyError
        When ``var`` is not a variable of ``ds``.
    IndexError
        When ``t_index`` is out of range.
    ValueError
        Before the figure is created: an unknown ``cmap``, a bound that is
        not a finite number, or ``vmin > vmax`` when both are given.
    """
    # 1. arguments
    _check_dpi(dpi)
    cmap = _resolve_cmap(cmap)
    vmin, vmax = _as_number(vmin), _as_number(vmax)
    _fill_bounds(vmin, vmax)  # explicit bounds: finite and vmin <= vmax
    # 2. data
    _require_var(ds, var)
    field, epoch_label = _field_at(ds, var, t_index)
    lon_grid, lat_grid = np.meshgrid(ds["lon"].values, ds["lat"].values)
    # 3. colour range
    vmin, vmax = _field_bounds(vmin, vmax, field.values)

    # 4. figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    mesh = ax.pcolormesh(
        lon_grid, lat_grid, field.values, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    fig.colorbar(mesh, ax=ax, label=var)

    if title is None:
        title = f"{var} @ t={epoch_label}"
    ax.set_title(title)

    fig.tight_layout()
    return finalize(fig, save_path=save_path, dpi=dpi, close=close)
