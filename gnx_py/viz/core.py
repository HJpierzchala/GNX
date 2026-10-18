"""Shared plotting primitives for :mod:`gnx_py.viz`.

This module holds the pieces every GNX plot needs: the publication style, the
figure/axes construction rules, the map layout used by the ionosphere maps, and
the date-axis formatting shared by every time series.

Two conventions are enforced here and relied on by the rest of the package:

* **The caller owns the figure.** Nothing in :mod:`gnx_py.viz` calls
  ``plt.show()`` and nothing closes a figure unless explicitly asked through
  ``close=True``. Plotting functions return a
  :class:`~matplotlib.figure.Figure`, so a notebook renders it by returning it
  and a script saves it through ``save_path``.
* **Styling is opt-in.** Importing this module does not touch
  :data:`matplotlib.rcParams`; call :func:`apply_publication_style` to do that.
  A library that restyles a user's session on import is impossible to opt out
  of.

Cartopy is imported lazily through :func:`require_cartopy` so that importing
``gnx_py.viz`` stays cheap and non-projected plots keep working regardless of
the Cartopy install.

A third convention keeps a failed call from leaving debris behind:

* **Figures are closed on failure.** A plotting function that raises closes
  every figure it created during that call (:func:`no_figure_leak`), so a
  loop of failing calls does not accumulate open figures. Figures that
  existed before the call, including the one behind a caller's ``ax``, are
  never touched. :func:`finalize` closes the figure when ``close=True`` even
  if saving fails.
"""
from __future__ import annotations

import collections.abc
import functools
import inspect
import numbers
from typing import Callable, Iterable, Literal, Mapping, TypeVar

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Map figure defaults. These values are the historical
# ``gnx_py.ionosphere.monitoring`` constants and are re-exported from there, so
# they are a public compatibility contract: changing them silently changes
# every saved map.
FIGSIZE: tuple[float, float] = (10, 10)
DPI: int = 600

# Default figure size for single-axes time series (the tutorial helpers' size).
SERIES_FIGSIZE: tuple[float, float] = (12, 4)

# Every GNX time axis is labelled the same way.
TIME_FORMAT: str = "%H:%M"

# Constellation names, keyed by the first character of a PRN.
GNSS_SYSTEM_NAMES: dict[str, str] = {
    "C": "BeiDou",
    "E": "Galileo",
    "G": "GPS",
    "I": "NavIC",
    "J": "QZSS",
    "R": "GLONASS",
    "S": "SBAS",
}

# Qualitative colormaps concatenated by :func:`categorical_palette` when more
# colours are needed than one palette holds: 20 + 20 + 20 distinct colours.
_TAB_PALETTES: tuple[str, ...] = ("tab20", "tab20b", "tab20c")
# Seaborn's ``deep`` palette has 10 colours; beyond that it would repeat.
_SEABORN_DEEP_SIZE: int = 10

# Colorbar strip geometry used by the map layout, in figure coordinates.
_GS_HEIGHT_RATIOS: tuple[float, float] = (1.0, 0.06)
_GS_HSPACE: float = 0.05
_CAX_GAP: float = 0.05
_CAX_HEIGHT: float = 0.018

_CARTOPY_HINT = (
    "Cartopy is required for projected map plots. It ships with the GNX "
    "runtime dependencies; install it with `pip install Cartopy==0.24.1` or "
    "`pip install -e .` from the repository root."
)

__all__ = [
    "DPI",
    "FIGSIZE",
    "GNSS_SYSTEM_NAMES",
    "SERIES_FIGSIZE",
    "TIME_FORMAT",
    "add_prn_legend",
    "apply_publication_style",
    "categorical_palette",
    "constellation_label",
    "finalize",
    "format_time_axis",
    "make_fig_ax",
    "make_fig_axes",
    "make_map_axes",
    "place_cax_below",
    "require_cartopy",
    "resolve_time_axis",
    "to_utc_naive",
]

_F = TypeVar("_F", bound=Callable[..., object])

# Options of the per-satellite scatter helper (:func:`_scatter_by_sv`).
_SV_GROUP_BY: tuple[str, ...] = ("arc", "prn")
_SV_LABEL_BY: tuple[str | None, ...] = ("prn", "series", None)
_SV_COLORS: tuple[str, ...] = ("cycle", "palette")

# Smallest colour range :func:`_fill_bounds` gives a bound it derived itself.
_MIN_COLOR_SPAN: float = 1e-3


def no_figure_leak(func: _F) -> _F:
    """Close the figures a plotting function created when it raises.

    A plotting function typically builds its figure first and fails later,
    when a column is missing, a value is out of range or the file cannot be
    written. Without this decorator the half-built figure stays registered
    with :mod:`matplotlib.pyplot` and accumulates in loops and notebooks.

    The wrapper records which pyplot figures are open before the call (by
    identity, not by figure number, because pyplot reuses a freed number).
    When the call raises (any :class:`BaseException`, including
    :class:`KeyboardInterrupt`), every pyplot figure that was not open before
    the call is closed and the original exception propagates unchanged.
    Figures that existed before the call are never closed, so a caller's own
    figure behind an ``ax`` argument survives a failed call. A successful call
    is not affected: the caller owns the figure it gets back.

    Apply it where the function is defined, so every re-export of the function
    (for example from :mod:`gnx_py.ionosphere.monitoring`) is the same object.
    :func:`functools.wraps` keeps the name, docstring and signature.

    Parameters
    ----------
    func:
        The plotting function. Generator and coroutine functions are rejected,
        because their body runs after the wrapper has returned.

    Returns
    -------
    callable
        The wrapped function.

    Raises
    ------
    TypeError
        When ``func`` is a generator or coroutine function.

    Notes
    -----
    If closing a figure itself fails during the cleanup, the original
    exception still propagates; the cleanup failure is attached to it as a
    note (:meth:`BaseException.add_note`) instead of replacing it.
    """
    if (
        inspect.isgeneratorfunction(func)
        or inspect.iscoroutinefunction(func)
        or inspect.isasyncgenfunction(func)
    ):
        raise TypeError(
            f"no_figure_leak cannot guard {func.__qualname__!r}: generator and coroutine "
            "functions run after the wrapper has returned."
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        before = _open_figures()
        try:
            return func(*args, **kwargs)
        except BaseException as exc:
            _close_new_figures(before, exc)
            raise

    return wrapper  # type: ignore[return-value]


def _open_figures() -> dict[int, Figure]:
    """Return the open pyplot figures keyed by ``id()``.

    The dict holds a reference to every figure, so no ``id`` can be reused by
    a new object while the snapshot is alive.
    """
    return {id(fig): fig for fig in _pyplot_figures()}


def _pyplot_figures() -> list[Figure]:
    """The figures currently registered with pyplot, without changing the current figure."""
    from matplotlib._pylab_helpers import Gcf  # pyplot's own figure registry

    return [manager.canvas.figure for manager in Gcf.get_all_fig_managers()]


def _close_new_figures(before: dict[int, Figure], exc: BaseException) -> None:
    """Close every pyplot figure that is not in the snapshot ``before``.

    Figures are compared by identity: a figure created under the number of a
    figure the call closed is still recognised as new. A failure to close one
    figure is recorded as a note on ``exc`` and does not stop the others from
    being closed.
    """
    for fig in _pyplot_figures():
        if before.get(id(fig)) is fig:
            continue
        try:
            plt.close(fig)
        except Exception as close_exc:  # the caller re-raises exc; keep it the primary error
            exc.add_note(
                f"gnx_py.viz: closing figure {fig.number} after this error also failed: {close_exc!r}"
            )


def _check_real(
    name: str,
    value: object,
    *,
    gt: float | None = None,
    ge: float | None = None,
) -> float:
    """Return ``value`` as a float, or raise ValueError naming ``name``.

    ``value`` must be a finite real number (bool rejected) and, when given,
    ``> gt`` or ``>= ge``.
    """
    from ..ionosphere.utils import _check_number  # lazy: no import cycle with ionosphere

    _check_number(name, value, gt=gt, ge=ge)
    return float(value)  # type: ignore[arg-type]


def _check_sv_scatter_options(
    group_by: object,
    label_by: object,
    colors: object,
) -> None:
    """Validate the ``group_by``, ``label_by`` and ``colors`` options of :func:`_scatter_by_sv`.

    Exposed separately so a public plot can check them before it creates a
    figure.

    Raises
    ------
    ValueError
        When ``group_by`` is not ``"arc"`` or ``"prn"``, ``label_by`` not
        ``"prn"``, ``"series"`` or None, or ``colors`` a string other than
        ``"cycle"`` or ``"palette"``.
    TypeError
        When ``colors`` is neither one of those strings nor a mapping from PRN
        to colour.
    """
    from ..ionosphere.utils import _check_choice, _did_you_mean  # lazy: no import cycle

    _check_choice("group_by", group_by, _SV_GROUP_BY)
    _check_choice("label_by", label_by, _SV_LABEL_BY)
    if isinstance(colors, str):
        if colors not in _SV_COLORS:
            msg = (
                f"colors must be one of {list(_SV_COLORS)!r} or a mapping from PRN to colour "
                f"(got {colors!r})"
            )
            hint = _did_you_mean(colors, _SV_COLORS)
            if hint is not None:
                msg += f"; did you mean {hint!r}?"
            raise ValueError(msg)
    elif not isinstance(colors, collections.abc.Mapping):
        raise TypeError(
            f"colors must be 'cycle', 'palette' or a mapping from PRN to colour "
            f"(got {type(colors).__name__})"
        )


def _resolve_cmap(cmap: object, *, name: str = "cmap", allow_none: bool = True) -> mpl.colors.Colormap | None:
    """Resolve a colormap argument before any figure exists.

    Matplotlib looks a colormap name up only when it draws, which is after
    the figure has been created. Resolving it first turns a typo into an
    error that leaves nothing behind.

    Parameters
    ----------
    cmap:
        A registered colormap name (``"viridis"``, ``"plasma_r"``, ...), a
        :class:`~matplotlib.colors.Colormap`, or None when ``allow_none``
        (matplotlib then uses ``rcParams["image.cmap"]``).
    name:
        Parameter name used in the error messages.
    allow_none:
        Accept None and return it unchanged.

    Returns
    -------
    matplotlib.colors.Colormap or None
        The colormap (a copy, for a name) or None.

    Raises
    ------
    ValueError
        When the name is not a registered colormap, or ``cmap`` is None and
        ``allow_none`` is false.
    TypeError
        When ``cmap`` is neither a string, a Colormap nor None.
    """
    if cmap is None:
        if allow_none:
            return None
        raise ValueError(f"{name}: a colormap is required (got None)")
    if isinstance(cmap, mpl.colors.Colormap):
        return cmap
    if not isinstance(cmap, str):
        raise TypeError(
            f"{name} must be a colormap name or a matplotlib Colormap (got {type(cmap).__name__})"
        )
    try:
        return mpl.colormaps[cmap]
    except KeyError:
        raise ValueError(
            f"{name}: unknown colormap {cmap!r}; see matplotlib.colormaps for the registered names"
        ) from None


def _fill_bounds(
    vmin: object,
    vmax: object,
    auto_lo: float | None = None,
    auto_hi: float | None = None,
    *,
    strict: bool = False,
    min_span: float = _MIN_COLOR_SPAN,
) -> tuple[float | None, float | None]:
    """Complete a linear colour range, keeping every explicit bound.

    Rules:

    * An explicit bound (``vmin`` or ``vmax`` not None) is always kept as
      given. It must be a finite real number.
    * Two explicit bounds must satisfy ``vmin <= vmax``, or ``vmin < vmax``
      when ``strict``; otherwise ValueError, before any figure exists.
    * A missing bound is taken from ``auto_lo`` / ``auto_hi`` (typically data
      percentiles). If that leaves the range empty or inverted, only the
      derived bound is moved, to ``min_span`` away from the other one: the
      user's bound never changes. With both bounds derived, the upper one is
      moved. The gap is ``min_span``, or a relative ``1e-9`` of the bound's
      magnitude when that is larger, so it is never lost to rounding.
    * Two derived bounds must be ordered (``auto_lo <= auto_hi``); inverted
      automatic values are a caller error and raise ValueError.
    * A missing bound whose automatic value is None stays None, so matplotlib
      autoscales it when drawing.

    Parameters
    ----------
    vmin, vmax:
        The caller's bounds, None when not given.
    auto_lo, auto_hi:
        Values used for a missing ``vmin`` / ``vmax``. Must be finite when
        they are used; a NaN (all-NaN data) raises ValueError asking for an
        explicit bound.
    strict:
        Reject two equal explicit bounds (a zero-width colour range).
    min_span:
        Minimum width given to a range whose derived bound had to be moved.

    Returns
    -------
    tuple[float | None, float | None]
        ``(vmin, vmax)`` as floats, or None where no bound is known.

    Raises
    ------
    ValueError
        When an explicit bound is not a finite number, the explicit bounds
        contradict each other, a needed automatic bound is not finite, both
        bounds are derived from inverted automatic values, or the range cannot
        be widened within the float range.

    Examples
    --------
    >>> _fill_bounds(None, 0.05, 0.1, 0.4)    # explicit vmax kept, vmin moved
    (0.049, 0.05)
    >>> _fill_bounds(None, None, 2.0, 7.0)
    (2.0, 7.0)
    """
    lo = None if vmin is None else _check_real("vmin", vmin)
    hi = None if vmax is None else _check_real("vmax", vmax)
    if lo is not None and hi is not None:
        if lo > hi or (strict and lo == hi):
            relation = "<" if strict else "<="
            raise ValueError(f"vmin must be {relation} vmax (got vmin={lo:g}, vmax={hi:g})")
        return lo, hi

    explicit_lo, explicit_hi = lo is not None, hi is not None
    if lo is None and auto_lo is not None:
        lo = _auto_bound("vmin", auto_lo)
    if hi is None and auto_hi is not None:
        hi = _auto_bound("vmax", auto_hi)
    if lo is not None and hi is not None and lo >= hi:
        if explicit_lo:
            hi = _widen(lo, +1.0, min_span)
        elif explicit_hi:
            lo = _widen(hi, -1.0, min_span)
        elif lo > hi:
            raise ValueError(
                f"automatic colour bounds are inverted (auto_lo={lo:g} > auto_hi={hi:g})"
            )
        else:
            hi = _widen(lo, +1.0, min_span)
    return lo, hi


def _widen(ref: float, direction: float, min_span: float) -> float:
    """Return a bound ``min_span`` (or ``1e-9 * |ref|``, if larger) beyond ``ref``.

    The relative part keeps the range non-empty for large values, where adding
    a fixed ``min_span`` would be lost to rounding.
    """
    span = max(min_span, abs(ref) * 1e-9)
    bound = ref + direction * span
    if not np.isfinite(bound) or bound == ref:
        raise ValueError(
            f"cannot build a colour range around {ref:g}; pass both vmin and vmax"
        )
    return bound


def _auto_bound(name: str, value: object) -> float:
    """Return a data-derived colour bound as a float, or raise ValueError when it is not finite."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not np.isfinite(value)
    ):
        raise ValueError(
            f"{name} cannot be derived from the data (got {value!r}: no finite values?); "
            f"pass {name} explicitly"
        )
    return float(value)


def apply_publication_style(
    *,
    font_scale: float = 1.4,
    serif: bool = True,
    mathtext: str | None = None,
) -> None:
    """Apply the GNX publication style to the global matplotlib settings.

    This mutates :data:`matplotlib.rcParams` and is therefore opt-in: importing
    :mod:`gnx_py.viz` never calls it. Call it once per session, before
    plotting.

    Seaborn is optional. When it is missing, the seaborn-specific context and
    style are skipped and the matplotlib-level settings are still applied.

    Parameters
    ----------
    font_scale:
        Seaborn font scale. The tutorial helpers historically disagreed here
        (1.6 for PPP, 1.3 for orbits and ionosphere); 1.4 is the shared
        default. Ignored when seaborn is not installed.
    serif:
        Use a Times New Roman serif family and embed TrueType fonts in PDF
        output (``pdf.fonttype = 42``), which is what journals expect.
    mathtext:
        Font set for mathtext (``$...$``) labels, written to
        ``rcParams["mathtext.fontset"]``; ``"cm"`` gives Computer Modern, as
        the tutorial helpers used. None leaves the current setting untouched.

    Returns
    -------
    None
    """
    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    if sns is not None:
        sns.set_context("paper", font_scale=font_scale)
        sns.set_style("ticks")

    if serif:
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["Times New Roman"]

    if mathtext is not None:
        mpl.rcParams["mathtext.fontset"] = mathtext


def constellation_label(prns: Iterable[str]) -> str:
    """Name the constellations a set of PRNs belongs to.

    Used to title per-satellite plots after what they actually contain, rather
    than after whichever satellite happened to be processed last.

    Parameters
    ----------
    prns:
        PRN strings such as ``"G01"``. Only the leading character is read;
        it is looked up in :data:`GNSS_SYSTEM_NAMES`, and an unknown system
        letter is returned as is.

    Returns
    -------
    str
        The constellation name when they all share one, ``"Multi-GNSS"`` when
        they do not, and ``"GNSS"`` when ``prns`` is empty.
    """
    systems = {prn[0] for prn in prns if prn}
    if not systems:
        return "GNSS"
    if len(systems) == 1:
        system = next(iter(systems))
        return GNSS_SYSTEM_NAMES.get(system, system)
    return "Multi-GNSS"


def add_prn_legend(ax: Axes, *, title: str = "PRN", ncol: int = 2) -> None:
    """Place a per-satellite legend outside the axes.

    A GNSS residual plot has one entry per satellite, which covers the data if
    the legend is placed inside. The caller is responsible for leaving room,
    typically with ``fig.tight_layout(rect=(0, 0, 0.85, 1))``.

    Parameters
    ----------
    ax:
        Axes whose handles are collected.
    title:
        Legend title.
    ncol:
        Number of legend columns.

    Returns
    -------
    None
    """
    ax.legend(
        fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=ncol, title=title
    )


def _scatter_by_sv(
    ax: Axes,
    frame: pd.DataFrame,
    value_col: str,
    *,
    x_col: str | None = None,
    time_col: str = "time",
    sv_col: str = "sv",
    group_by: Literal["arc", "prn"] = "arc",
    label_by: Literal["prn", "series"] | None = "prn",
    colors: Literal["cycle", "palette"] | Mapping[str, object] = "cycle",
    s: float = 2,
    dropna: bool = True,
    **scatter_kwargs: object,
) -> list[str]:
    """Scatter one series per satellite arc or PRN onto a single axes.

    This is the drawing loop shared by the per-satellite plots (PPP and SPP
    residuals, STEC arcs). GNX keys satellite arcs as ``"G01_0"``,
    ``"G01_1"``, ...; the PRN is the part before the first ``"_"``, and a
    plain ``"G01"`` key is its own PRN. The legend gets one entry per PRN, not
    one per arc, however the series are grouped.

    Parameters
    ----------
    ax:
        Axes to draw on.
    frame:
        Frame holding the values, the satellite keys and the epochs, such as
        an ``(sv, time)`` MultiIndex residual frame.
    value_col:
        Column plotted on the y axis.
    x_col:
        Column plotted on the x axis, such as ``"ev"``. None plots against
        the epochs, located per series by :func:`resolve_time_axis`.
    time_col:
        Epoch column or index level, used when ``x_col`` is None.
    sv_col:
        Column or index level holding the satellite (arc) keys.
    group_by:
        ``"arc"`` draws one series per distinct ``sv_col`` value;
        ``"prn"`` merges the arcs of each PRN into one series.
    label_by:
        ``"prn"`` labels the first series of each PRN with the PRN, so the
        legend has one entry per satellite; ``"series"`` labels every series
        with its own key (the arc key, or the PRN when grouping by PRN);
        None draws no labels.
    colors:
        ``"cycle"`` leaves colours to the axes' property cycle, so every
        series takes the next colour; ``"palette"`` gives each PRN one colour
        from :func:`categorical_palette`, shared by all its arcs; a mapping
        from PRN to colour sets them explicitly, and PRNs missing from it
        fall back to the cycle.
    s:
        Marker size, in points squared.
    dropna:
        Drop rows whose ``value_col`` is NaN before drawing, and skip series
        left empty.
    **scatter_kwargs:
        Extra keyword arguments forwarded to every
        :meth:`matplotlib.axes.Axes.scatter` call, such as ``marker`` or
        ``alpha``.

    Returns
    -------
    list[str]
        The PRNs drawn, without repeats, in drawing order. Pass them to
        :func:`constellation_label` to title the plot.

    Raises
    ------
    KeyError
        When ``value_col``, ``x_col`` or ``sv_col`` is missing, or the epochs
        cannot be located.
    ValueError
        When ``group_by``, ``label_by`` or a string ``colors`` is not one of
        the values listed above.
    TypeError
        When ``colors`` is neither a string nor a mapping.

    Examples
    --------
    >>> fig, ax = make_fig_ax()
    >>> prns = _scatter_by_sv(ax, residuals, "v")  # doctest: +SKIP
    >>> ax.set_title(f"{constellation_label(prns)} residuals")  # doctest: +SKIP
    >>> # STEC of every PRN against local time, one palette colour per PRN.
    >>> _scatter_by_sv(ax, obs_tec, "leveled_tec", time_col="LT",
    ...                group_by="prn", colors="palette")  # doctest: +SKIP
    """
    _check_sv_scatter_options(group_by, label_by, colors)
    if value_col not in frame.columns:
        raise KeyError(
            f"Column {value_col!r} not found. Available: {list(frame.columns)[:10]}."
        )
    if x_col is not None and x_col not in frame.columns:
        raise KeyError(f"Column {x_col!r} not found in the frame.")
    if sv_col in frame.columns:
        sv_keys = frame[sv_col].to_numpy()
    elif sv_col in frame.index.names:
        sv_keys = frame.index.get_level_values(sv_col).to_numpy()
    else:
        raise KeyError(f"No column or index level named {sv_col!r} holds the satellites.")

    prn_keys = np.array([str(key).split("_")[0] for key in sv_keys], dtype=object)
    if colors == "cycle":
        color_of: Mapping[str, object] = {}
    elif colors == "palette":
        prns = sorted(set(prn_keys))
        # An empty frame (every row filtered out upstream) has no PRN to colour.
        color_of = dict(zip(prns, categorical_palette(len(prns)))) if prns else {}
    else:
        color_of = colors

    drawn: list[str] = []
    for key, series in frame.groupby(sv_keys if group_by == "arc" else prn_keys):
        if dropna:
            series = series[series[value_col].notna()]
        if series.empty:
            continue
        prn = str(key).split("_")[0]
        if label_by == "prn":
            label = prn if prn not in drawn else None
        elif label_by == "series":
            label = str(key)
        else:
            label = None
        if prn not in drawn:
            drawn.append(prn)

        x = resolve_time_axis(series, time_col) if x_col is None else series[x_col]
        kwargs = dict(scatter_kwargs)
        if prn in color_of:
            kwargs.setdefault("color", color_of[prn])
        ax.scatter(x, series[value_col], label=label, s=s, **kwargs)
    return drawn


def categorical_palette(n_colors: int) -> list:
    """Return ``n_colors`` distinct colours for per-satellite series.

    Up to 60 colours are pairwise distinct, which covers a full multi-GNSS
    constellation. With seaborn installed and at most 10 colours, the seaborn
    ``deep`` palette is used. Otherwise the colours come from matplotlib's
    ``tab20``, extended with ``tab20b`` and ``tab20c`` when more than 20 are
    needed; the list order is the same with or without seaborn beyond 10
    colours, so plots do not change colour between environments.

    Parameters
    ----------
    n_colors:
        Number of colours needed, an integer >= 1. Values above 60 wrap
        around, so colours repeat from the 61st on.

    Returns
    -------
    list
        Colours accepted by matplotlib's ``color`` argument.

    Raises
    ------
    ValueError
        When ``n_colors`` is not an integer >= 1 (an empty palette is never a
        meaningful request; callers with possibly empty data check first).
    """
    from ..ionosphere.utils import _check_number  # lazy: no import cycle with ionosphere

    _check_number("n_colors", n_colors, ge=1, integer=True)
    n_colors = int(n_colors)
    if n_colors <= _SEABORN_DEEP_SIZE:
        try:
            import seaborn as sns
        except ModuleNotFoundError:
            pass
        else:
            return sns.color_palette("deep", n_colors)

    # tab20 alone for up to 20 colours, the three tab palettes beyond that.
    names = _TAB_PALETTES[:1] if n_colors <= 20 else _TAB_PALETTES
    colors = [color for name in names for color in plt.get_cmap(name).colors]
    return [colors[i % len(colors)] for i in range(n_colors)]


def _check_dpi(dpi: object, *, allow_none: bool = True) -> None:
    """Raise ValueError unless ``dpi`` is a finite number > 0 (or None when allowed).

    Called before any figure is created, so a bad resolution leaves no
    half-built figure open.
    """
    if dpi is None and allow_none:
        return
    if (
        isinstance(dpi, (bool, np.bool_))
        or not isinstance(dpi, (int, float, np.integer, np.floating))
        or not np.isfinite(dpi)
        or dpi <= 0
    ):
        value = dpi.item() if isinstance(dpi, np.generic) else dpi
        raise ValueError(f"dpi must be > 0 (got {value!r})")


def require_cartopy() -> tuple[object, object]:
    """Import Cartopy on demand and fail with an actionable message.

    Returns
    -------
    tuple[object, object]
        The ``cartopy.crs`` and ``cartopy.feature`` modules.

    Raises
    ------
    ImportError
        When Cartopy is not installed, with an install hint.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError(_CARTOPY_HINT) from exc
    return ccrs, cfeature


@no_figure_leak
def make_fig_ax(
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] = SERIES_FIGSIZE,
    dpi: int | None = None,
    projection: object | None = None,
) -> tuple[Figure, Axes]:
    """Return a figure/axes pair, creating one only when ``ax`` is None.

    This is the single entry point behind the ``ax=None`` convention used by
    every single-axes plot in :mod:`gnx_py.viz`: pass nothing and get a fresh
    figure, or pass an existing axes to compose the plot into a larger layout.

    Parameters
    ----------
    ax:
        Existing axes to draw on. When None, a new figure and axes are created.
    figsize:
        Size of the new figure, in inches. Ignored when ``ax`` is given.
    dpi:
        Resolution of the new figure. None uses the matplotlib default.
        Ignored when ``ax`` is given.
    projection:
        Cartopy CRS for map axes. Ignored when ``ax`` is given; in that case
        the caller is responsible for passing a projected axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        ``(ax.figure, ax)`` when ``ax`` was supplied, otherwise the new pair.

    Notes
    -----
    A new figure is closed again when creating its axes fails, for example
    on an invalid ``projection`` (:func:`no_figure_leak`).
    """
    _check_dpi(dpi)
    if ax is not None:
        return ax.figure, ax

    if projection is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        new_ax = fig.add_subplot(1, 1, 1, projection=projection)
        return fig, new_ax

    fig, new_ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, new_ax


@no_figure_leak
def make_fig_axes(
    nrows: int,
    *,
    figsize: tuple[float, float] = SERIES_FIGSIZE,
    dpi: int | None = None,
    sharex: bool = True,
) -> tuple[Figure, list[Axes]]:
    """Create a vertical stack of axes sharing one x axis.

    Used by the multi-panel plots, which cannot honour an ``ax`` argument
    because they own more than one axes. The axes are always returned as a
    list, including for ``nrows=1``, so callers never special-case the
    single-panel result.

    Parameters
    ----------
    nrows:
        Number of stacked panels. Must be at least 1.
    figsize:
        Figure size in inches.
    dpi:
        Figure resolution. None uses the matplotlib default.
    sharex:
        Share the x axis across panels.

    Returns
    -------
    tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]
        The figure and its panels, top to bottom.

    Raises
    ------
    ValueError
        When ``nrows`` is not an integer >= 1.
    """
    if (
        isinstance(nrows, (bool, np.bool_))
        or not isinstance(nrows, numbers.Integral)
        or nrows < 1
    ):
        raise ValueError(f"nrows must be an integer >= 1, got {nrows!r}.")
    _check_dpi(dpi)
    fig, axes = plt.subplots(int(nrows), 1, figsize=figsize, dpi=dpi, sharex=sharex)
    return fig, list(np.atleast_1d(axes))


@no_figure_leak
def make_map_axes(
    projection: object,
    *,
    figsize: tuple[float, float] = FIGSIZE,
    dpi: int = DPI,
) -> tuple[Figure, Axes, Axes]:
    """Create the standard GNX map layout: a map axes with a colorbar below it.

    Every projected map in GNX uses this identical layout, so that maps of
    different quantities stack and compare cleanly.

    Parameters
    ----------
    projection:
        Cartopy CRS passed to the map axes.
    figsize:
        Figure size in inches.
    dpi:
        Figure resolution.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]
        ``(fig, ax, cax)`` where ``ax`` is the map axes and ``cax`` the
        colorbar axes. Call :func:`place_cax_below` after the map extent is
        final to snap the colorbar to the map.

    Notes
    -----
    The new figure is closed again when creating the map axes fails, for
    example on an invalid ``projection`` (:func:`no_figure_leak`).
    """
    _check_dpi(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    grid = gridspec.GridSpec(2, 1, height_ratios=_GS_HEIGHT_RATIOS, hspace=_GS_HSPACE)
    ax = fig.add_subplot(grid[0, 0], projection=projection)
    cax = fig.add_subplot(grid[1, 0])
    return fig, ax, cax


def place_cax_below(
    ax: Axes,
    cax: Axes,
    *,
    gap: float = _CAX_GAP,
    height: float = _CAX_HEIGHT,
) -> None:
    """Snap a colorbar axes directly beneath a map axes.

    Cartopy resizes a map axes when its extent is set, which leaves a
    grid-allocated colorbar floating. Call this once the extent is final.

    Parameters
    ----------
    ax:
        The map axes whose position drives the placement.
    cax:
        The colorbar axes to reposition in place.
    gap:
        Vertical space between map and colorbar, in figure coordinates
        (0..1); a finite number >= 0.
    height:
        Colorbar height, in figure coordinates (0..1); a finite number > 0.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        When ``gap`` is negative or ``height`` is not positive, or either is
        not a finite number. Nothing is moved in that case.
    """
    gap = _check_real("gap", gap, ge=0)
    height = _check_real("height", height, gt=0)
    pos = ax.get_position()
    cax.set_position([pos.x0, pos.y0 - gap - height, pos.width, height])


def format_time_axis(
    ax: Axes,
    *,
    fmt: str = TIME_FORMAT,
    hour_interval: int | None = None,
    autofmt: bool = True,
) -> None:
    """Format an x axis as a time-of-day axis.

    Parameters
    ----------
    ax:
        Axes whose x axis holds datetimes.
    fmt:
        :class:`~matplotlib.dates.DateFormatter` format string.
    hour_interval:
        When given, place a major tick every ``hour_interval`` hours.
    autofmt:
        Rotate and right-align the tick labels so long timestamps do not
        overlap.

    Returns
    -------
    None
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    if hour_interval is not None:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    if autofmt:
        ax.figure.autofmt_xdate()


def resolve_time_axis(frame: pd.DataFrame, time_col: str = "time") -> pd.Series | pd.DatetimeIndex:
    """Locate the epoch axis of a frame, from a column or the index.

    GNX carries epochs in whichever place each session found natural: the PPP
    solution is indexed by time, the SPP solution keeps a ``time`` column, the
    residual and TEC frames hold them in a ``time`` level of an ``(sv, time)``
    MultiIndex, and all of them become a ``time`` column once written to CSV
    and read back. Reading the index blindly would silently plot a saved
    solution against its row numbers, which looks like a valid plot, so the
    location is resolved here and a frame with no epochs at all is an error
    rather than a bad axis.

    The epochs are looked up in this order:

    1. a column named ``time_col``;
    2. an index level named ``time_col``, of a MultiIndex or of a named single
       index;
    3. a :class:`~pandas.DatetimeIndex`, whatever its name.

    Parameters
    ----------
    frame:
        Frame to inspect.
    time_col:
        Name of the epoch column or index level.

    Returns
    -------
    pandas.Series or pandas.DatetimeIndex
        The epochs, one per row and in row order: the column as a Series, or
        the index level or index as a DatetimeIndex. Values that are not
        datetimes, such as the strings of a CSV read without
        ``parse_dates``, are converted with :func:`pandas.to_datetime`.

    Raises
    ------
    KeyError
        When the frame has neither the column, nor the index level, nor a
        datetime index.
    TypeError
        When the values found cannot be read as datetimes. Numeric values are
        rejected too: converting them would silently give 1970 epochs.
    """
    if time_col in frame.columns:
        values = frame[time_col]
        where = f"column {time_col!r}"
    elif time_col in frame.index.names:
        # get_level_values keeps one value per row, in row order, for both a
        # MultiIndex level and a named single index.
        values = frame.index.get_level_values(time_col)
        where = f"index level {time_col!r}"
    elif isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    else:
        raise KeyError(
            f"Time column {time_col!r} not found, no index level has that name "
            f"and the index is not a DatetimeIndex, so the epochs cannot be "
            f"located. Available columns: {list(frame.columns)}; index levels: "
            f"{list(frame.index.names)}."
        )

    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values
    if pd.api.types.is_numeric_dtype(values.dtype):
        raise TypeError(
            f"Epochs in {where} are numeric ({values.dtype}), not datetimes; "
            "convert them explicitly before plotting."
        )
    try:
        converted = pd.to_datetime(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Epochs in {where} cannot be read as datetimes (dtype {values.dtype})."
        ) from exc
    if isinstance(values, pd.Series):
        return converted
    return pd.DatetimeIndex(converted)


def to_utc_naive(index: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    """Convert a datetime index to timezone-naive UTC.

    Matplotlib date axes mix badly with timezone-aware timestamps, and GNX
    products carry a mix of both. Converting to naive UTC keeps every time axis
    on the same footing.

    Parameters
    ----------
    index:
        Datetime index or series. Naive input is returned unchanged (assumed
        to already be UTC); aware input is converted to UTC, then localised
        away.

    Returns
    -------
    pandas.DatetimeIndex
        Timezone-naive UTC timestamps.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx
    return idx.tz_convert("UTC").tz_localize(None)


def finalize(
    fig: Figure,
    *,
    save_path: str | None = None,
    dpi: int | None = None,
    close: bool = False,
    **savefig_kwargs: object,
) -> Figure:
    """Optionally save a figure, then hand it back to the caller.

    Parameters
    ----------
    fig:
        The figure to finalize.
    save_path:
        When given, write the figure to this path. The file is cropped to the
        drawn content (``bbox_inches="tight"``) unless ``bbox_inches`` is
        passed, so legends and captions placed outside the axes are kept.
    dpi:
        Resolution for the saved file, > 0. None keeps the figure's own dpi.
        A value <= 0 raises ValueError (``make_fig_ax``, ``make_fig_axes``
        and ``make_map_axes`` check it the same way before creating a figure).
    close:
        Close the figure after saving; the figure is still returned but no
        longer renders. Use this when generating many figures in a loop.
        The figure is closed even when the arguments are rejected or saving
        fails (the error still propagates), so a failing loop leaves no
        figures open.
    **savefig_kwargs:
        Extra keyword arguments forwarded to
        :meth:`matplotlib.figure.Figure.savefig`. Pass ``bbox_inches=None``
        to save the full, uncropped canvas.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure, so callers can ``return finalize(fig, ...)``.

    Raises
    ------
    ValueError
        When ``dpi`` is not > 0.
    OSError
        When the file cannot be written, for example because the directory of
        ``save_path`` does not exist (``FileNotFoundError``); nothing creates
        directories.
    """
    try:
        if not (isinstance(dpi, str) and dpi == "figure"):  # matplotlib's "keep the figure dpi" token
            _check_dpi(dpi)
        if save_path is not None:
            savefig_kwargs.setdefault("bbox_inches", "tight")
            fig.savefig(save_path, dpi=dpi if dpi is not None else fig.dpi, **savefig_kwargs)
    finally:
        if close:
            plt.close(fig)
    return fig
