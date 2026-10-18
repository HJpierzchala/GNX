# GNX Visualization

`gnx_py.viz` holds the plotting code for GNX results: PPP solutions and
residuals, orbit and SIS(RE) errors, STEC and VTEC, the ionospheric activity
indexes, and the projected maps of all of them.

The package mirrors the processing modules, so the plot for a result lives in
the module named after the code that produced it:

- `gnx_py.viz.spp`: SPP residuals and positioning time series.
- `gnx_py.viz.ppp`: PPP residuals and positioning time series.
- `gnx_py.viz.orbits`: SIS(RE) panels and per-satellite statistics.
- `gnx_py.viz.ionosphere`: STEC arcs, zenith VTEC, ROTI/SIDX/GIX series, and
  the modelled-versus-measured STEC comparison.
- `gnx_py.viz.maps`: every projected map. This is the only module that uses
  Cartopy.
- `gnx_py.viz.schematics`: explanatory diagrams for documentation.
- `gnx_py.viz.tables`: compact Markdown previews of result frames.
- `gnx_py.viz.core`: the shared style, figure and axis primitives.

Everything is matplotlib. There is no plotly or folium backend, and no plans
for one: matplotlib and Cartopy are already GNX runtime dependencies, so a
plot works wherever GNX itself works.

## Conventions

Four rules hold across the package.

**Every plot returns a `Figure`.** Nothing calls `plt.show()` and nothing
closes a figure unless you ask with `close=True`; the figure is still
returned, it just no longer renders. A notebook renders a figure by returning
it, a script writes one with `save_path=`, and neither has to guess what the
library did to the global pyplot state. Every plot takes `save_path=` and
`close=`, with two exceptions: `make_roti_snapshots` writes and closes each
frame itself (it has `out_dir=` instead), and `STECModelComparator.report()`
takes neither, because it returns its figures for you to save or close.

`save_path=` saves with `bbox_inches="tight"`, so titles and colorbars below
the axes are never clipped. For the full canvas, save through the helper
yourself: `finalize(fig, save_path="out.png", bbox_inches=None)`.

Three functions return something other than a bare `Figure`:

- `plot_roti_snapshot` returns `None` when the requested epoch has no data in
  its window. That is what lets `make_roti_snapshots` walk an epoch range
  containing gaps; `make_roti_snapshots` itself returns the paths it wrote.
- `compare_vtec_maps` returns `(fig, mean, median)`: the figure plus the mean
  and median absolute difference it prints in its title as `MAE` and `MedAE`.
- `STECModelComparator.report()` returns an `STECComparisonReport` named
  tuple, `(stats, hexbin, hist)`.

**Series plots accept `ax`.** Pass one to compose a plot into a larger
layout; leave it out and the plot builds its own figure at `figsize`. The
plots that take `ax` are `plot_residuals`, `plot_spp_residuals`,
`plot_sv_stats`, `plot_stec`, `plot_stec_arcs`, `plot_vtec_zenith`,
`plot_roti_series`, `plot_sidx_series`, `plot_point_timeseries` and
`plot_vtec_error_series`. The others build their own figure and take no
`ax`: the multi-panel plots (`plot_sisre`, `plot_uduc_residuals`,
`plot_positioning_series`, `plot_spp_positioning`, `plot_gix_family` and the
`STECModelComparator` plots), the maps, whose layout carries a projection and a
colorbar (`plot_roti_snapshot`, `plot_gix`, `plot_vtec_map`,
`compare_vtec_maps`, `plot_grid_map`), and the diagrams (`plot_stec_change`,
`plot_gix_two_points`).

**Figures are closed on failure.** A plot that raises closes every figure it
created during that call, so a loop of failing calls does not pile up open
figures. Figures that existed before the call, including the one behind an
`ax` you passed, are never touched. The inputs are checked before the figure
is created wherever they can be (column names, index levels, option names,
numeric ranges, colormaps, colour bounds, map extents), so most errors leave
nothing behind in the first place. With `close=True` the figure is closed even
when saving it fails.

**Styling is opt-in.** Importing `gnx_py.viz` does not touch `matplotlib.rcParams`.
Call `apply_publication_style()` when you want the GNX look; it is a no-op on
the seaborn parts when seaborn is not installed. Pass `mathtext="stix"` (or any
other `mathtext.fontset`) to match the maths in labels to a serif body font.

## Basic Use

```python
import matplotlib.pyplot as plt
from gnx_py.viz import apply_publication_style, plot_positioning_series, plot_residuals

apply_publication_style()

fig = plot_positioning_series(result.solution, convergence_time=15, title="PPP: BRUX")
fig.savefig("ppp_convergence.png", dpi=300)

# Phase residuals against elevation rather than time.
fig = plot_residuals(result.residuals_gps, obs_type="phase", x_axis="elev")
```

`result` is a `PPPResult`: its residuals are per system (`residuals_gps`,
`residuals_gal`, `residuals_bds`, None for a system the run did not process),
and `result.solution["ct_min"]` holds the convergence time in minutes, the unit
`convergence_time` takes.

`plot_residuals` picks the residual column from `obs_type`; pass
`value_col=` to plot a specific column instead. `cut_init_time` drops the
first epochs of the session and `skip_obs` the first epochs of every
satellite arc, exactly that many (`skip_obs=0` keeps every arc whole); both
must be integers >= 0. The residual column, the `sv` and `time` index levels
and, for `x_axis="elev"`, the `ev` column are required even when the trimming
leaves nothing to draw. `plot_positioning_series`
skips columns that hold no data, so a solution without reference coordinates
gets no blank ENU panels; `cut_init_time` must be an integer >= 0 there too,
and `convergence_time` a number of minutes (a string such as `"30"` is an
error, not converted). `plot_uduc_residuals` accepts any dual-frequency
signal pair as `mode` (`"L1L2"`, `"E1E5a"`, `"B1IB3I"`, ...); the mode's
constellation must be one of the frame's and, when anything is left to draw
after the trimming, one of the plotted satellites'. Satellites of other
constellations in the same frame are drawn under the mode's band labels.

The residual plots (`plot_residuals`, `plot_uduc_residuals`,
`plot_spp_residuals`) give each PRN one colour, shared by all its arcs and,
in `plot_uduc_residuals`, by both bands, so the per-PRN legend identifies
every satellite.

RINEX and SP3 epochs are GPS time, so the solution and orbit plots
(`plot_positioning_series`, `plot_spp_positioning`, `plot_sisre`) label their
time axis "Time [GPST]". Pass `xlabel=` if you converted the epochs to
another scale.

Compose several plots into one layout by passing `ax`:

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_residuals(result.residuals_gps, obs_type="phase", ax=axes[0])
plot_residuals(result.residuals_gps, obs_type="code", ax=axes[1])
```

## SPP

SPP results are shaped differently from PPP results, so they have their own
functions rather than aliases: the SPP solution carries time in a `time`
column instead of its index, names the receiver clock per constellation
(`dtr_gps`), and splits residuals into one frame per constellation.

```python
from gnx_py.viz import plot_spp_positioning, plot_spp_residuals

result = SPPSession(config).run()

# Position errors plus the satellite count, which usually explains the spikes.
fig = plot_spp_positioning(result.solution, title="SPP: BRUX, GPS L1/L2")

# Any other solution parameter can be plotted instead.
fig = plot_spp_positioning(result.solution, columns=("de", "dn", "du", "rms_v"))

# Residuals, with the observations the solver rejected marked separately.
fig = plot_spp_residuals(result.residuals_gps, x_axis="elev")
```

Columns that are present but empty are skipped: a GPS-only run still carries
`dtr_gal` and `n_gal` as an all-NaN and an all-zero column, and neither earns
a panel. When none of `de`/`dn`/`du` has data, `plot_spp_positioning` issues
a `UserWarning`: the run had no reference coordinates (typically from SINEX)
to difference against. When nothing at all is left to plot (no position
error and no satellite count), it raises `ValueError` before that warning.
A single column name may be given as a string: `columns="de"` is one panel,
as `columns=["de"]`.

SPP keeps `V` NaN for every observation it did not use, so "finite `V`"
means "used in the solution". An observation the residual-rejection loop
dropped (`spp_reject_reason="postfit_residual"`) carries its residual against
the solution it was excluded from in a separate column, `V_rejected`.
`plot_spp_residuals` draws those as black crosses on the same axis as the
accepted residuals, labelled `Rejected (n)` with `n` the number of crosses
drawn. Observations of an epoch rejected as a whole have no residual and are
not drawn.

`plot_spp_residuals` takes one constellation's frame at a time, since that is
how SPP produces them. Use `ax` to compare constellations in one layout:

```python
frames = {"GPS": result.residuals_gps, "Galileo": result.residuals_gal}
frames = {name: frame for name, frame in frames.items() if frame is not None}

fig, axes = plt.subplots(len(frames), 1, figsize=(12, 8), sharex=True)
for ax, (name, frame) in zip(axes, frames.items()):
    plot_spp_residuals(frame, ax=ax, title=f"{name} residuals")
```

There is no `cut_init_time` here, unlike the PPP residual plots: SPP solves
each epoch independently, so there is no filter convergence to trim.

## Maps

Map functions share one layout: the map with a horizontal colorbar underneath,
so maps of different quantities stack and compare cleanly.

```python
from gnx_py.viz import plot_gix, plot_roti_snapshot, make_roti_snapshots

fig = plot_roti_snapshot(roti_df, epoch=epoch, res_deg=1.0, save_path="roti.png")
fig = plot_gix(pairs_df, epoch=epoch, log_scale=True,
               projection="LambertConformal", extent=[10, 30, 48, 58])

# One frame per epoch, on a shared colour scale, for an animation.
frames = make_roti_snapshots(roti_df, every_n=2, out_dir="frames", global_scale=True)
```

Cartopy is imported on first use, not at import time. `plot_gix` takes any
Cartopy projection that needs no constructor arguments, by name; an unknown
name raises `ValueError` listing the supported ones, before the pairs are
matched to the epoch. `plot_vtec_map` with a `region` scales its colours to
the values inside that region unless you pass `vmin`/`vmax` (for a
`PlateCarree` `data_crs`; other data projections scale to the whole field).
`plot_grid_map` draws a gridded field on plain latitude/longitude axes when a
real projection is not needed; it and `plot_vtec_map` accept a variable
without a `time` dimension.

Extents and regions are in degrees, `[lon_min, lon_max, lat_min, lat_max]`.
Without one, `plot_roti_snapshot` and `plot_gix` fit the map to the data
(`plot_gix` through Cartopy's automatic limits, with a small margin),
`plot_vtec_map` shows the whole grid and `compare_vtec_maps` the common grid
nodes. An extent outside the valid domain of the projection,
such as central Europe in `"OSGB"`, raises a `ValueError` naming the
projection and the extent; it is not clipped to the domain.

Colour bounds you give are kept. When you give one, only the other is
derived from the data, and if the derived one would leave an empty range it
moves, never the one you gave. Two given bounds must satisfy `vmin < vmax`
in the ROTI maps and in the log-scale `plot_gix`, and `vmin <= vmax` in the
linear maps (`plot_gix`, `plot_vtec_map`, `compare_vtec_maps`,
`plot_grid_map`). With `log_scale=True`,
`plot_gix` needs positive data, and a given bound <= 0 is an error rather
than being replaced. `make_roti_snapshots(global_scale=True)` derives only
the bound you did not give from the whole dataset. Unknown option names
(`cmap`, `draw`, `stats_loc`, ...) raise `ValueError` before the figure.

`compare_vtec_maps` compares the two datasets at one epoch present in both
(selected exactly; a missing epoch is a `KeyError`) and on their common grid
nodes: the latitudes and longitudes that appear in both grids with exactly
the same value. A fine model grid can therefore be checked against a coarse
reference such as a 2.5 x 5 degree GIM, and the map, `MAE` and `MedAE` cover
those nodes only. Grids with no common node raise a `ValueError` naming both
grids. A 0..360 grid and a -180..180 grid share only the longitudes 0..180.
`region` crops the displayed map; the statistics always use every common
node.

`make_roti_snapshots` saves every frame on the full, uncropped canvas, so all
frames have the same even pixel size and can go straight into a video encoder
(e.g. ffmpeg with libx264 and `-pix_fmt yuv420p`) or a GIF. Pass a fixed
`extent` as well, so the map region does not move between frames. A single
`plot_roti_snapshot(save_path=...)` is still cropped tight. `out_dir` must be
an existing directory (it is not created; it is checked once, before the
first frame), and `filename_fmt` must give every epoch its own file name; all
names are built before the first frame is drawn. With `out_dir=None` the
frames are rendered and closed without being written, and the call returns
an empty list. The frames use the map defaults (`FIGSIZE`, `DPI`, tiles,
coastlines); for other figure settings call `plot_roti_snapshot` in your own
loop.

## Ionosphere Series and Comparisons

```python
from gnx_py.viz import (
    STECModelComparator, plot_stec, plot_stec_arcs, plot_vtec_error_series, plot_vtec_zenith,
)

fig = plot_stec(obs_tec, "G01", time_col="LT")

# Every satellite's arcs on one axes, with the zenith VTEC laid over them.
fig = plot_stec_arcs(obs_tec, time_col="LT")
plot_vtec_zenith(zen, time_col="LT", ax=fig.axes[0], color="black", lw=2,
                 label="Smoothed VTEC", legend=True, legend_ncol=2)

comparator = STECModelComparator(
    df, x_col="leveled_tec", model_cols=["ion", "klobuchar", "ntcm"],
    model_names=["GIM", "Klobuchar", "NTCM G"],
)
stats, hexbin, hist = comparator.report(title="Modelled vs measured STEC")

# Per-map error of several VTEC models against a reference, one line each.
fig = plot_vtec_error_series({"Kriging": mae_kriging, "NTCM": mae_ntcm}, reference="GIM")
```

`plot_stec` draws code STEC on the left axis and phase STEC on a twin right
axis by default, because the ambiguity offset between them would otherwise
flatten both. Series whose columns are absent from the frame are skipped, so
the defaults work across TEC frames that carry different optional columns,
and a series without a `"color"` takes the next colour of the axes cycle.

`TECSession` keeps its epochs in GPST, so `plot_stec` labels the `time` axis
"Time [GPST]"; pass `xlabel=` to say otherwise. With the default series,
each y axis takes the colour of its first series: code STEC in blue on the
left (levelled STEC in green, smoothed STEC in orange), phase STEC in red on
the right. The axis labels follow
`rcParams`, so `apply_publication_style()` sizes them with the ticks.

The activity-index plots (`plot_roti_series`, `plot_sidx_series`,
`plot_gix_family`) and `plot_vtec_error_series` take the time scale of their
label from the data: the frame's `attrs["time_scale"]` first (the outputs of
`gnx_py.ionosphere.monitoring` carry `"GPST"`, the time scale of the TEC
session), else "UTC" for timezone-aware epochs and "GPST" for naive ones,
since naive GNX epochs are GPS time. The index plots take `time_scale=` and
`plot_vtec_error_series` takes `xlabel=` to say otherwise.

`report()` used to return only the statistics frame. It still draws the same
figures, and now returns them as well so that a script can save or close
them; a figure you switched off with `show_hex`/`show_hist` is `None`.

`STECModelComparator` checks its inputs when it is built: `df` must be a
DataFrame holding `x_col` and every model column (`KeyError` otherwise),
`model_cols` must not be empty, and `model_names`, when given, must have one
name per column. A single string is one column (`model_cols="ion"`).
`model_cols` and `model_names` are stored as lists. `plot_stec` raises
`KeyError` before the figure when none of the requested series is in the
frame.

## Tables

`df_head` and `df_tail` preview a frame or a Series as a compact Markdown
table: the first or last `nrows`, at most `ncols` columns, floats formatted
with `floatfmt`, integers without decimals, and long strings cut at
`truncate_str`.

```python
from gnx_py.viz import df_head, df_tail

df_head(result.solution, nrows=5)            # displays the table, returns None
md = df_tail(stats, ncols=8, show=False)     # displays nothing, returns the Markdown
```

`nrows`, `ncols` and `truncate_str` are integers >= 0, or None for "all";
`df_tail(nrows=0)` shows the header only.

With the default `show=True` the table is displayed as Markdown in a notebook
and printed as Markdown source elsewhere (terminal IPython, plain Python), and
the call returns `None`, so a notebook cell does
not echo it twice. An unnamed `RangeIndex` is dropped, since it only numbers
the rows; any other index, MultiIndex levels included, becomes the leading
columns and counts toward `ncols`. With `reset_index=False` the index stays
the index and appears only with `index=True`.

## Diagrams

`plot_stec_change` and `plot_gix_two_points` draw explanatory diagrams, not
data. Their inputs are checked before the figure: `scale` and `arrow_scale`
must be finite and > 0, positions pairs of finite numbers, latitudes strictly
between -90 and 90 degrees, longitudes in [-180, 180] and `iono_height_km`
>= 0. A pair of points across the antimeridian is drawn as one short arc, so
the longitude axis may extend past +-180.

## Relationship to `gnx_py.ionosphere`

The ROTI and GIX map plots used to live in `gnx_py.ionosphere.monitoring`.
They now live here, and `monitoring` imports them back, so
`gnx_py.plot_gix`, `gnx_py.plot_roti_snapshot`, `gnx_py.plot_gix_family` and
`gnx_py.make_roti_snapshots` are still importable under the same names. Their
behaviour is not unchanged, though; code written against the old versions
needs these adjustments:

- They now return a single `Figure`. `plot_gix` previously returned
  `(fig, ax, sc)` and `plot_gix_family` returned `(fig, axes)`; unpacking
  those tuples now fails. Reach the axes through `fig.axes`.
- `plot_roti_snapshot` previously returned `None` on every path and closed
  the figure after saving. It now returns the figure and keeps it open; pass
  `close=True` in a loop.
- `plot_gix` raises `ValueError` for an unknown projection name instead of
  silently falling back to PlateCarree.
- `plot_gix_family` derives the analysed period from the data. It previously
  wrote a fixed `Period: 2024 / 035, sampling: 30 s` into the title regardless
  of its input. Pass `subtitle=` to set that line yourself. `reg` may be a
  `Region`, shown by its bounds.

Plotting stays separate from metric computation, as it was in `monitoring`:
these functions summarise a frame for display, but the indexes themselves are
computed by `gnx_py.ionosphere`.

## Developer Notes

Every public plot function is wrapped by `gnx_py.viz.core.no_figure_leak`
where it is defined, so the re-exports in `gnx_py.ionosphere.monitoring` are
the same objects. A new plot function needs the decorator (a forced `savefig`
failure must leave no figure open).

`categorical_palette(n)` needs `n >= 1`; callers with possibly empty data
check first. Up to 10 colours it uses seaborn's `deep` palette when seaborn
is installed, so the first colours differ between environments; beyond 10 the
colours are the same with and without seaborn. `apply_publication_style(serif=True)`
asks for Times New Roman, and matplotlib warns (`findfont`) where that font is
missing; that warning comes from matplotlib, not from GNX-py.

No module in `gnx_py.viz` imports `gnx_py.*` at module level. The GNX-internal
computations the maps and diagrams need are imported inside the functions that
use them. That is what allows `gnx_py.ionosphere.monitoring` to re-export from
`gnx_py.viz` without an import cycle; keep it that way.

The geodesy used by the diagrams comes from `gnx_py.ionosphere.monitoring`, so
a diagram cannot drift from the code it explains. Note that
`great_circle_distance_km` there takes its radius **in metres** and returns
kilometres.
