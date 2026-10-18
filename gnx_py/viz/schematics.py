"""Explanatory diagrams of the ionospheric observables.

These figures illustrate definitions rather than plot measurements: they are
drawn from a handful of scalars and are meant for documentation, teaching and
the manual. Everything else in :mod:`gnx_py.viz` plots real data.

The geodesy is not reimplemented here. The great-circle distance, bearing and
midpoint come from :mod:`gnx_py.ionosphere.monitoring`, the same functions the
GIX computation itself uses, so a diagram cannot drift from the code it
explains.
"""
from __future__ import annotations

from numbers import Real
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .core import _check_real, finalize, make_fig_ax, no_figure_leak

# Mean Earth radius in km. Matches ``gnx_py.ionosphere.monitoring.Re`` (which
# is in metres); kept here so the diagram defaults need no import.
_EARTH_RADIUS_KM = 6371.0
# Single-layer shell height in km. Matches ``monitoring.IONO_H``.
_DEFAULT_SHELL_HEIGHT_KM = 450.0

__all__ = [
    "plot_gix_two_points",
    "plot_stec_change",
]


def _to_cartesian(lat: float, lon: float) -> np.ndarray:
    """Unit vector of a geographic position on a sphere."""
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    return np.array(
        [
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ]
    )


def _to_geographic(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geographic latitude and longitude of a cartesian vector, in degrees."""
    x, y, z = vector
    lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    lon = (np.rad2deg(np.arctan2(y, x)) + 540) % 360 - 180
    return lat, lon


def _check_point(name: str, value: object) -> np.ndarray:
    """Return ``value`` as a float array of two finite numbers, or raise ValueError.

    Each element must be a real number: booleans, strings and other objects
    are rejected, as they are for the scalar settings.
    """
    message = f"{name} must be a pair of finite numbers (x, y) (got {value!r})"
    try:
        items = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if items.shape != (2,) or not all(
        isinstance(item, Real) and not isinstance(item, (bool, np.bool_)) for item in items
    ):
        raise ValueError(message)
    point = items.astype(float)
    if not np.isfinite(point).all():
        raise ValueError(message)
    return point


def _check_latitude(name: str, value: object) -> float:
    """Return a latitude in degrees as a float, or raise ValueError unless it is in (-90, 90)."""
    from ..ionosphere.utils import _check_number  # lazy: no import cycle with ionosphere

    _check_number(name, value, gt=-90, lt=90, unit="deg")
    return float(value)  # type: ignore[arg-type]


def _check_longitude(name: str, value: object) -> float:
    """Return a longitude in degrees as a float, or raise ValueError unless it is in [-180, 180]."""
    from ..ionosphere.utils import _check_number  # lazy: no import cycle with ionosphere

    _check_number(name, value, ge=-180, le=180, unit="deg")
    return float(value)  # type: ignore[arg-type]


def _nearest_longitude(lon, ref: float):
    """Shift ``lon`` by whole turns to the representative nearest ``ref``, in degrees.

    Values already within 180 degrees of ``ref`` are returned unchanged, so a
    pair that does not cross the antimeridian is drawn exactly as given.
    """
    return lon - 360.0 * np.round((np.asarray(lon, dtype=float) - ref) / 360.0)


def _great_circle_path(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the great-circle arc between two points, for drawing.

    This is a drawing helper, not geodesy: it exists so an arc looks like an
    arc rather than a straight line in longitude/latitude space.
    """
    a = _to_cartesian(lat1, lon1)
    b = _to_cartesian(lat2, lon2)
    omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if omega < 1e-9:
        return np.linspace(lat1, lat2, n), np.linspace(lon1, lon2, n)

    t = np.linspace(0, 1, n)
    points = (
        a[:, None] * (np.sin((1 - t) * omega) / np.sin(omega))
        + b[:, None] * (np.sin(t * omega) / np.sin(omega))
    )
    points /= np.linalg.norm(points, axis=0, keepdims=True)
    return _to_geographic(points)


@no_figure_leak
def plot_stec_change(
    *,
    rx_xy: tuple[float, float] = (0.0, 0.0),
    sat_xy_t0: tuple[float, float] = (-5.0, 6.0),
    sat_xy_t1: tuple[float, float] = (5.0, 6.0),
    stec_t0: float = 23.4,
    stec_t1: float = 27.9,
    annotate: bool = True,
    scale: float = 1.1,
    show_iono_arc: bool = True,
    caption: str | None = None,
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Draw how STEC changes as a satellite moves across the sky.

    A schematic in arbitrary units: the receiver, the satellite at two epochs,
    the signal path at each, and a symbolic ionospheric shell. It shows why
    STEC changes even when the ionosphere itself does not, because the path
    length through the shell changes with elevation.

    Parameters
    ----------
    rx_xy:
        Receiver position, in arbitrary units.
    sat_xy_t0:
        Satellite position at the first epoch.
    sat_xy_t1:
        Satellite position at the second epoch.
    stec_t0:
        STEC at the first epoch, in TECU. Annotation only.
    stec_t1:
        STEC at the second epoch, in TECU. Annotation only.
    annotate:
        Label the STEC values and their difference.
    scale:
        Zoom factor of the scene, a finite number > 0. The figure is
        ``(7.5 * scale, 6 * scale)`` inches.
    show_iono_arc:
        Draw the symbolic ionospheric shell.
    caption:
        Caption placed under the figure. None draws no caption.
    save_path:
        When given, write the figure to this path.
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the diagram.

    Raises
    ------
    ValueError
        When ``scale`` is not a finite number > 0, or a position is not a pair
        of finite numbers. Raised before any figure is created.
    """
    # A zero or negative scale used to give a 0x0 (or negative) figure size.
    scale = _check_real("scale", scale, gt=0)
    rx = _check_point("rx_xy", rx_xy)
    sat0 = _check_point("sat_xy_t0", sat_xy_t0)
    sat1 = _check_point("sat_xy_t1", sat_xy_t1)

    fig, ax = make_fig_ax(figsize=(7.5 * scale, 6 * scale))

    ax.scatter(*rx, s=60, marker="o", label="Receiver", zorder=3, color="black")
    ax.scatter(*sat0, s=120, marker="*", label="Satellite @t0", zorder=3, color="royalblue")
    ax.scatter(*sat1, s=120, marker="*", label="Satellite @t1", zorder=3, color="deepskyblue")
    ax.plot(
        [rx[0], sat0[0]], [rx[1], sat0[1]], linewidth=2, label="Link @ t0", color="royalblue"
    )
    ax.plot(
        [rx[0], sat1[0]],
        [rx[1], sat1[1]],
        linewidth=2,
        linestyle="--",
        label="Link @ t1",
        color="deepskyblue",
    )

    if show_iono_arc:
        radius = np.linalg.norm([sat0[0] * 0.8, sat0[1] * 0.8]) * 1.1
        theta = np.linspace(np.pi / 8, np.pi - np.pi / 8, 200)
        x_arc, y_arc = radius * np.cos(theta), radius * np.sin(theta)
        ax.plot(
            x_arc,
            y_arc,
            color="lightblue",
            linewidth=2,
            alpha=0.6,
            label="Ionospheric shell",
        )
        ax.text(
            x_arc[len(x_arc) // 2],
            y_arc[len(y_arc) // 2] + 0.5,
            f"Ionospheric shell (~{_DEFAULT_SHELL_HEIGHT_KM:.0f} km)",
            color="steelblue",
            ha="center",
            fontsize=10,
        )

    if annotate:
        ax.text(sat0[0], sat0[1], f"  STEC(t0)={stec_t0:.2f} TECU", va="bottom", ha="left", fontsize=11)
        ax.text(sat1[0], sat1[1], f"  STEC(t1)={stec_t1:.2f} TECU", va="bottom", ha="left", fontsize=11)
        midpoint = 0.5 * ((rx + sat0) + (rx + sat1)) / 2.0
        ax.text(
            midpoint[0],
            midpoint[1],
            f"ΔSTEC={stec_t1 - stec_t0:+.2f} TECU",
            bbox={"boxstyle": "round,pad=0.3", "fc": "w", "ec": "0.7"},
            fontsize=11,
        )

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Horizontal distance (a.u.)")
    ax.set_ylabel("Height / direction (a.u.)")
    ax.legend(loc="best", bbox_to_anchor=(1, 0.5))
    ax.set_aspect("equal", adjustable="box")

    extent = max(np.abs(sat0).max(), np.abs(sat1).max()) * 1.2
    ax.set_xlim(-extent * scale, extent * scale)
    ax.set_ylim(-extent * 0.1, extent * scale)
    ax.set_axis_off()

    if caption:
        fig.text(0.5, -0.05, caption, ha="center", fontsize=11)
    return finalize(fig, save_path=save_path, close=close)


@no_figure_leak
def plot_gix_two_points(
    *,
    lat_i: float = 52.0,
    lon_i: float = 16.0,
    vtec_i: float = 12.5,
    lat_j: float = 50.5,
    lon_j: float = 12.0,
    vtec_j: float = 10.9,
    iono_height_km: float = _DEFAULT_SHELL_HEIGHT_KM,
    title: str = "GIX for two IPPs (with CP)",
    draw_gc_arc: bool = True,
    arrow_scale: float = 0.55,
    show_cp_arrow: bool = True,
    annotate: bool = True,
    caption: str | None = None,
    figsize: tuple[float, float] = (7.5, 6),
    dpi: int = 120,
    save_path: str | Path | None = None,
    close: bool = False,
) -> Figure:
    """Draw the VTEC gradient between two ionospheric pierce points.

    Shows what one GIX pair contributes: two pierce points on the shell, the
    great-circle arc between them, the bearing from j to i, and the centre
    point the resulting gradient is attributed to.

    Parameters
    ----------
    lat_i:
        Latitude of pierce point i, in degrees, strictly between -90 and 90.
    lon_i:
        Longitude of pierce point i, in degrees, in [-180, 180].
    vtec_i:
        VTEC at pierce point i, in TECU.
    lat_j:
        Latitude of pierce point j, in degrees, strictly between -90 and 90.
    lon_j:
        Longitude of pierce point j, in degrees, in [-180, 180].
    vtec_j:
        VTEC at pierce point j, in TECU.
    iono_height_km:
        Height of the single-layer shell, in km, a finite number >= 0.
    title:
        Plot title.
    draw_gc_arc:
        Draw the true great-circle arc rather than a straight segment.
    arrow_scale:
        Length factor of the bearing arrow, a finite number > 0.
    show_cp_arrow:
        Draw the gradient arrow at the centre point.
    annotate:
        Label the pierce points and the derived quantities.
    caption:
        Caption placed under the figure. None draws no caption.
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
        The figure holding the diagram.

    Raises
    ------
    ValueError
        When a coordinate or VTEC value is not finite, a latitude is not
        strictly inside (-90, 90), a longitude is outside [-180, 180],
        ``iono_height_km`` is negative, ``arrow_scale`` is not > 0 or
        ``dpi`` is not > 0. Raised before any figure is created.

    Notes
    -----
    The arrows are drawn in longitude/latitude degrees, so their east
    component is divided by ``cos(latitude)``. At a pole that factor is zero
    and the bearing is undefined, hence the open latitude interval.

    The longitude axis is anchored at IPP j: point i, the arc and the centre
    point are drawn at the longitude nearest ``lon_j`` (shifted by whole
    turns), so a pair across the antimeridian appears as one short arc and
    the axis may extend past +-180 deg. The labels give the centre point
    longitude in [-180, 180). Two cases stay degenerate as drawings: points
    exactly 180 deg apart in longitude, whose shorter arc passes over a pole
    and jumps in longitude there, and antipodal points, which have no unique
    great circle.
    """
    # Validate before the geodesy and the figure: |lat| = 90 used to draw an
    # arrow about 1e16 degrees long through the 1/cos(lat) factor.
    lat_i = _check_latitude("lat_i", lat_i)
    lat_j = _check_latitude("lat_j", lat_j)
    lon_i = _check_longitude("lon_i", lon_i)
    lon_j = _check_longitude("lon_j", lon_j)
    vtec_i = _check_real("vtec_i", vtec_i)
    vtec_j = _check_real("vtec_j", vtec_j)
    iono_height_km = _check_real("iono_height_km", iono_height_km, ge=0)
    arrow_scale = _check_real("arrow_scale", arrow_scale, gt=0)

    from ..ionosphere.monitoring import (
        great_circle_distance_km,
        initial_bearing_deg,
        spherical_midpoint,
    )

    # The library geodesy takes the radius in metres and returns km.
    shell_radius_m = (_EARTH_RADIUS_KM + float(iono_height_km)) * 1000.0
    distance_km = float(
        great_circle_distance_km(lat_j, lon_j, lat_i, lon_i, shell_radius_m)
    )
    azimuth_deg = float(initial_bearing_deg(lat_j, lon_j, lat_i, lon_i))
    azimuth_rad = np.deg2rad(azimuth_deg)

    delta_vtec = float(vtec_i) - float(vtec_j)
    gradient = delta_vtec / distance_km if distance_km > 0 else np.nan
    gx, gy = gradient * np.sin(azimuth_rad), gradient * np.cos(azimuth_rad)
    cp_lat, cp_lon = spherical_midpoint(lat_j, lon_j, lat_i, lon_i)
    cp_lat, cp_lon = float(cp_lat), float(cp_lon)

    # Drawing frame anchored at IPP j: point i, the arc and the centre point
    # are shifted by whole turns to the longitude nearest lon_j, so a pair
    # across the antimeridian is drawn as one short arc, not split across
    # the axes. Pairs that do not cross it are drawn exactly as given.
    x_i = float(_nearest_longitude(lon_i, lon_j))
    x_cp = float(_nearest_longitude(cp_lon, lon_j))

    fig, ax = make_fig_ax(figsize=figsize, dpi=dpi)

    if draw_gc_arc:
        lat_arc, lon_arc = _great_circle_path(lat_j, lon_j, lat_i, lon_i)
        x_arc = _nearest_longitude(lon_arc, lon_j)
        ax.plot(x_arc, lat_arc, "-", color="tab:blue", lw=2, label="Pair (j→i, great-circle)")
    else:
        ax.plot([lon_j, x_i], [lat_j, lat_i], "-", color="tab:blue", lw=2, label="Pair (j→i)")

    ax.scatter([lon_j], [lat_j], s=80, c="tab:orange", marker="o", zorder=3, label="IPP j")
    ax.scatter([x_i], [lat_i], s=90, c="tab:green", marker="^", zorder=3, label="IPP i")
    ax.scatter([x_cp], [cp_lat], s=70, c="mediumpurple", marker="s", zorder=4, label=r"$CP_{i,j}$")

    # Arrow length in degrees, damped so short and long pairs both stay legible.
    base = max(0.5, min(2.0, distance_km / 400.0)) * arrow_scale
    ax.arrow(
        lon_j,
        lat_j,
        base * np.sin(azimuth_rad) / np.cos(np.deg2rad(lat_j)),
        base * np.cos(azimuth_rad),
        width=0.01,
        head_width=0.1,
        head_length=0.2,
        color="tab:red",
        alpha=0.95,
        length_includes_head=True,
        zorder=4,
        label="Azimuth j→i",
    )

    if show_cp_arrow:
        base_cp = base * 0.6
        ax.arrow(
            x_cp,
            cp_lat,
            base_cp * np.sin(azimuth_rad) / np.cos(np.deg2rad(cp_lat)),
            base_cp * np.cos(azimuth_rad),
            width=0.008,
            head_width=0.1,
            head_length=0.2,
            color="mediumpurple",
            alpha=0.9,
            length_includes_head=True,
            zorder=5,
        )

    if annotate:
        ax.text(lon_j, lat_j, f"  j\nVTEC={vtec_j:.2f} TECU", va="top", ha="left", fontsize=10, color="tab:orange")
        ax.text(x_i, lat_i, f"  i\nVTEC={vtec_i:.2f} TECU", va="bottom", ha="left", fontsize=10, color="tab:green")
        ax.text(
            x_cp + 0.7,
            cp_lat - 0.5,
            f"$CP_{{i,j}}$\nΔs = {distance_km:.1f} km\nlat: {cp_lat:.2f} lon: {cp_lon:.2f}",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.6"},
            ha="left",
            va="bottom",
            fontsize=10,
            color="mediumpurple",
        )

        # Offset the summary box perpendicular to the pair, clear of the arc.
        mid_lon, mid_lat = (x_i + lon_j) / 2, (lat_i + lat_j) / 2
        normal_lon = -np.cos(azimuth_rad) / max(1e-6, np.cos(np.deg2rad(mid_lat)))
        normal_lat = np.sin(azimuth_rad)
        norm = np.hypot(normal_lon, normal_lat)
        ax.text(
            mid_lon - 2 * normal_lon / norm,
            mid_lat - 2 * normal_lat / norm,
            f"az(j→i) = {azimuth_deg:.1f}°\nΔVTEC = {delta_vtec:+.2f} TECU\n"
            f"gₓ = {gx:+.4f}, gᵧ = {gy:+.4f} TECU/km",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.6"},
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    pad_lon = max(1.0, 0.12 * abs(x_i - lon_j) + 1.0)
    pad_lat = max(1.0, 0.12 * abs(lat_i - lat_j) + 1.0)
    ax.set_xlim(min(x_i, lon_j) - pad_lon, max(x_i, lon_j) + pad_lon)
    ax.set_ylim(min(lat_i, lat_j) - pad_lat, max(lat_i, lat_j) + pad_lat)
    fig.tight_layout()

    if caption:
        fig.text(0.5, -0.05, caption, ha="center", fontsize=11)
    return finalize(fig, save_path=save_path, dpi=dpi, close=close)
