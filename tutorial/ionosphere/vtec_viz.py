
from __future__ import annotations
from typing import Tuple, Optional, Dict
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# Styl publikacyjny
sns.set_context("paper", font_scale=1.3)
sns.set_style("ticks")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
def load_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path)

def detect_variables(ds: xr.Dataset) -> Tuple[str, Optional[str]]:
    data_vars = list(ds.data_vars)
    lower = {name: name.lower() for name in data_vars}
    vtec = None
    ss = None
    if "V" in ds.data_vars:
        vtec = "V"
    else:
        for k, v in lower.items():
            if "vtec" in v:
                vtec = k
                break
    if "SS" in ds.data_vars:
        ss = "SS"
    else:
        for k, v in lower.items():
            if any(s in v for s in ["ss", "sigma", "std", "error"]):
                ss = k
                break
    return vtec, ss

def _nearest_point(ds: xr.Dataset, lat: float, lon: float) -> Dict[str, float]:
    lat_idx = int(np.abs(ds["lat"].values - lat).argmin())
    lon_idx = int(np.abs(ds["lon"].values - lon).argmin())
    return {"lat_idx": lat_idx, "lon_idx": lon_idx,
            "lat": float(ds["lat"].values[lat_idx]),
            "lon": float(ds["lon"].values[lon_idx])}

def plot_map(ds: xr.Dataset, var: str, t_index: int = 0, title: Optional[str] = None,
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             savepath: Optional[str] = None) -> None:
    field = ds[var].isel(time=t_index)
    lats = ds["lat"].values
    lons = ds["lon"].values
    data = field.values
    Lon, Lat = np.meshgrid(lons, lats)
    plt.figure()
    h = plt.pcolormesh(Lon, Lat, data, shading="auto", vmin=vmin, vmax=vmax)
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.colorbar(h, label=var)
    if title is None:
        t_coord = str(np.array(ds["time"].values)[t_index]) if "time" in ds else ""
        title = f"{var} @ t={t_coord}"
    plt.title(title)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def plot_timeseries(ds: xr.Dataset, var: str, lat: float, lon: float,
                    title: Optional[str] = None, savepath: Optional[str] = None) -> None:
    idx = _nearest_point(ds, lat, lon)
    series = ds[var].isel(lat=idx["lat_idx"], lon=idx["lon_idx"]).to_series()
    plt.figure()
    series.plot()
    plt.xlabel("Time")
    plt.ylabel(var)
    if title is None:
        title = f"{var} @ ~({idx['lat']:.2f}°, {idx['lon']:.2f}°)"
    plt.title(title)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def ss_stats(ds: xr.Dataset, ss_var: str):
    arr = ds[ss_var].values.astype(float)
    return {"mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr))}

# --- paste into vtec_viz.py ---
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

def plot_model(
    ds,
    var: str,
    t_index: int = 0,
    region: tuple | None = None,      # (lon_min, lon_max, lat_min, lat_max)
    projection=None,                  # np. ccrs.Mercator()
    data_crs=None,                    # CRS danych; dla lat/lon -> ccrs.PlateCarree()
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    draw_coastlines: bool = True,
    draw_borders: bool = True,
    gridlines: bool = True,
    savepath: str | None = None,
    figsize: tuple = (8, 5),
    dpi: int = 150,
    cmap: str = "plasma",
    # --- sterowanie paskiem kolorów ---
    add_colorbar: bool = True,
    cbar_label: str | None = None,
    cbar_orientation: str = "vertical",   # "vertical" lub "horizontal"
    cbar_shrink: float = 0.9,             # zmniejszenie paska (0–1)
    cbar_pad: float = 0.02,               # odstęp od osi
    cbar_aspect: float = 30,              # stosunek długość / szerokość
    cbar_fraction: float | None = None,   # szerokość jako ułamek osi (gdy None -> domyślne)
    cbar_location: str = "right",         # "right", "bottom", ...
    cbar_kwargs: dict | None = None       # dodatkowe parametry do plt.colorbar
):
    """
    Rysuje mapę 'var' na tle mapy dla wybranego indeksu czasu używając Cartopy.

    Parametry
    ---------
    ds : xr.Dataset
        Zawiera wymiary (time, lat, lon) i zmienną 'var'.
    var : str
        Nazwa zmiennej, np. "V" lub "SS".
    t_index : int
        Indeks czasu (po osi 'time').
    region : tuple
        (lon_min, lon_max, lat_min, lat_max); jeśli None – użyje zakresu z danych.
    projection : cartopy.crs.Projection
        Docelowa projekcja mapy (oś). Np. ccrs.PlateCarree(), ccrs.Mollweide().
    data_crs : cartopy.crs.Projection
        CRS danych; dla regularnych siatek geograficznych -> ccrs.PlateCarree().
    vmin, vmax : float
        Zakres skali kolorów.
    title : str
        Tytuł mapy; jeśli None – zostanie zbudowany z nazwy zmiennej i czasu.
    draw_coastlines, draw_borders, gridlines : bool
        Sterowanie rysowaniem linii brzegowej, granic i siatki.
    savepath : str
        Jeśli podany, zapisze rysunek do pliku (png/pdf/...).
    figsize : tuple
        Rozmiar figury w calach.
    dpi : int
        Rozdzielczość figury.
    cmap : str
        Nazwa colormap (np. "plasma", "viridis", "cividis").

    Parametry paska kolorów
    -----------------------
    add_colorbar : bool
        Czy dodać pasek kolorów.
    cbar_label : str
        Etykieta paska kolorów; jeśli None – użyje 'var' i jednostek z ds[var].attrs['units'].
    cbar_orientation : {"vertical", "horizontal"}
    cbar_shrink : float
        Współczynnik zmniejszenia paska (często 0.8–1.0).
    cbar_pad : float
        Odstęp między osią a paskiem kolorów.
    cbar_aspect : float
        Stosunek długość/szerokość paska.
    cbar_fraction : float
        Szerokość paska względem osi; np. 0.046 domyślnie.
    cbar_location : str
        "right", "left", "top", "bottom".
    cbar_kwargs : dict
        Dodatkowe argumenty przekazywane do plt.colorbar.

    Wymaga: cartopy (zalecana instalacja przez conda-forge).
    """

    def place_cax_below(ax, cax, gap=0.006, height=0.018):
        """
        gap, height w jednostkach figury (0..1).
        """
        pos = ax.get_position()
        cax.set_position([pos.x0, pos.y0 - gap - height, pos.width, height])

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        raise ImportError(
            "Brak Cartopy. Zainstaluj np.: conda install -c conda-forge cartopy"
        ) from e

    # --- CRS danych i projekcji mapy ---
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    if projection is None:
        projection = ccrs.PlateCarree()

    # --- Dane ---
    field = ds[var].isel(time=t_index)
    lats = ds["lat"].values
    lons = ds["lon"].values

    # --- Extent ---
    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
    else:
        lon_min = float(np.min(lons))
        lon_max = float(np.max(lons))
        lat_min = float(np.min(lats))
        lat_max = float(np.max(lats))

    # --- Siatka 2D do pcolormesh ---
    Lon, Lat = np.meshgrid(lons, lats)

    # --- Rysunek / oś z projekcją ---
    # fig, ax = plt.subplots(
    #     1, 1,
    #     figsize=figsize,
    #     dpi=dpi,
    #     subplot_kw={"projection": projection}
    # )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.06], hspace=0.05)  # <- identycznie jak plot_roti/plot_gix
    ax = fig.add_subplot(gs[0, 0], projection=projection)
    cax = fig.add_subplot(gs[1, 0])




    # Zakres mapy w geograficznym CRS
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- Tło / warstwy stylu „Earth/Space” ---
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.9", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="0.96", edgecolor="none")

    if draw_coastlines:
        ax.coastlines(resolution="50m", linewidth=0.7, zorder=4)
    if draw_borders:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, zorder=4)

    # --- Warstwa danych ---
    h = ax.pcolormesh(
        Lon, Lat, field.values,
        transform=data_crs,
        shading="auto",
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        zorder=3
    )

    # --- Siatka geograficzna ---
    if gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.7
        )
        gl.top_labels = False
        gl.right_labels = False

    # --- Tytuł ---
    if title is None:
        t_val = ""
        if "time" in ds:
            t_val = str(np.array(ds["time"].values)[t_index])
        title = f"{var} @ t={t_val}"
    ax.set_title(title, fontsize=12)

    # --- Pasek kolorów ---
    if add_colorbar:
        if cbar_kwargs is None:
            cbar_kwargs = {}

        # domyślna etykieta: nazwa + jednostki z atrybutów
        if cbar_label is None:
            units = ""
            try:
                units = ds[var].attrs.get("units", "")
            except Exception:
                units = ""
            if units:
                cbar_label = f"{var} [{units}]"
            else:
                cbar_label = var

        # budujemy dict z argumentami do colorbara
        cb_args = dict(
            orientation=cbar_orientation,
            pad=cbar_pad,
            shrink=cbar_shrink,
            aspect=cbar_aspect,
            location=cbar_location,
        )
        # fraction dodajemy tylko, jeśli użytkownik podał
        if cbar_fraction is not None:
            cb_args["fraction"] = cbar_fraction

        # ewentualne dodatkowe parametry użytkownika
        cb_args.update(cbar_kwargs)

        # cb = plt.colorbar(h, ax=ax, **cb_args)
        # cb.set_label(cbar_label)
        cb = plt.colorbar(h, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label)

        # stałe marginesy (bez tight)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08)

        # NA KOŃCU: wymuś pozycję cax względem ax
        place_cax_below(ax, cax, gap=0.05, height=0.018)

        # ax.set_xticks([])
        # ax.set_yticks([])

    # --- Zapis / pokazanie ---
    # plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=600)
        # fig.savefig(
        #     savepath.split('.')[0] + '.tiff',
        #     dpi=600,
        #     format="tiff",
        #     pil_kwargs={"compression": "tiff_lzw"}
        # )

    plt.show()

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def compare_plot(
    ds_model,
    ds_ref,
    time,
    var_model: str = "V",
    var_ref: str = "TEC",
    region: tuple | None = None,      # (lon_min, lon_max, lat_min, lat_max)
    projection=None,                  # np. ccrs.PlateCarree()
    data_crs=None,                    # CRS danych; dla lon/lat -> ccrs.PlateCarree()
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Blues",
    figsize: tuple = (6, 4),
    dpi: int = 120,
    # colorbar
    add_colorbar: bool = True,
    cbar_label: str = r"|ΔTEC| [TECU]",
    cbar_orientation: str = "horizontal",
    cbar_shrink: float = 0.8,
    cbar_pad: float = 0.1,
    cbar_aspect: float = 40,
    title_prefix: str = r"|ΔTEC| = |TEC$_\mathrm{ref}$ - V$_\mathrm{model}$|",
    show_stats_in_title: bool = True,
        show=True
):
    """
    Rysuje mapę bezwzględnej różnicy |TEC_ref - V_model| dla jednej epoki.

    Parametry
    ---------
    ds_model : xr.Dataset
        Dataset z modelem, np. 'V'.
    ds_ref : xr.Dataset
        Dataset referencyjny, np. 'TEC'.
    time : datetime / np.datetime64 / cokolwiek akceptuje .sel(time=...)
        Epoka, dla której liczymy różnicę.
    var_model : str
        Nazwa zmiennej w ds_model.
    var_ref : str
        Nazwa zmiennej w ds_ref.
    region : tuple
        (lon_min, lon_max, lat_min, lat_max); jeśli None – użyje pełnego zakresu.
    projection : cartopy.crs.Projection
        Projekcja mapy (oś).
    data_crs : cartopy.crs.Projection
        CRS danych; dla regularnej siatki lon/lat -> ccrs.PlateCarree().
    vmin, vmax : float
        Zakres skali kolorów.
    cmap : str
        Colormap, np. "Blues".
    figsize : tuple
        Rozmiar figury w calach.
    dpi : int
        DPI figury.
    add_colorbar : bool
        Czy dodać pasek kolorów.
    cbar_* :
        Parametry paska kolorów.
    title_prefix : str
        Tekst początkowy tytułu.
    show_stats_in_title : bool
        Czy dopisać ME/medianę w tytule.

    Zwraca
    -------
    fig, ax, mean_abs, median_abs
    """
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    if projection is None:
        projection = ccrs.PlateCarree()

    # --- wybór epoki i zmiennych ---
    model_da = ds_model[var_model].sel(time=np.datetime64(time))
    ref_da   = ds_ref[var_ref].sel(time=np.datetime64(time))

    # różnica bezwzględna
    da = np.abs(ref_da - model_da)

    # wyciągnięcie 2D (gdyby czas jednak został w wymiarach)
    da_t = da if "time" not in da.dims else da.isel(time=0)

    # statystyki
    mean_abs   = float(da_t.mean().values)
    median_abs = float(da_t.median().values)

    # współrzędne
    lats = da_t["lat"].values
    lons = da_t["lon"].values

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
    else:
        lon_min = float(np.min(lons))
        lon_max = float(np.max(lons))
        lat_min = float(np.min(lats))
        lat_max = float(np.max(lats))

    # --- rysunek / oś ---
    fig, ax = plt.subplots(
        1, 1,
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": projection},
    )

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # tło
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.9", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="0.96", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=0.7, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, zorder=4)

    # gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linestyle=":",
        linewidth=0.3,
        alpha=0.7,
        zorder=4
    )
    gl.top_labels = False
    gl.right_labels = False

    # --- mapa różnic: xarray + pcolormesh ---
    cbar_kwargs = None
    if add_colorbar:
        cbar_kwargs = dict(
            orientation=cbar_orientation,
            shrink=cbar_shrink,
            pad=cbar_pad,
            aspect=cbar_aspect,
            label=cbar_label,
        )

    da_t.plot.pcolormesh(
        ax=ax,
        transform=data_crs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
    )

    # --- tytuł ---
    tstr = np.datetime_as_string(np.datetime64(time), unit="m")
    if show_stats_in_title:
        title = (
            f"{title_prefix}\n"
            f"Time: {tstr}  |  ME={mean_abs:.3f} TECU, median={median_abs:.3f} TECU"
        )
    else:
        title = f"{title_prefix}\nTime: {tstr}"

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if not show:
        plt.close()

    return fig, ax, mean_abs, median_abs

