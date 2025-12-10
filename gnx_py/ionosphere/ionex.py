"""IONEX reader and TEC interpolation utilities.

This module provides tools to read Global Ionosphere Maps (GIM) from IONEX/INX
files, parse Differential Code Biases (DCBs), and interpolate Vertical/Slant
Total Electron Content (VTEC/STEC) at requested epochs and positions.

High-level overview:
- GIMReader: reads IONEX/INX TEC maps and optional DCB products, parses meta and
  gridded data, and returns structured containers with TEC grids and statistics.
- TECInterpolator: evaluates/interpolates TEC at ionospheric piercing points
  (IPPs) for given receiver/satellite geometry and epochs.
- GIMData/_DCBEntry: lightweight containers for TEC grids and DCB entries.

Notes:
- The module assumes geographic latitude/longitude grids with fixed resolution
  and supports epoch selection logic for multi-epoch products.
- Interpolation relies on simple spherical shell geometry for IPP computation,
  unless noted otherwise in method-specific documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr
from ..conversion import ecef2geodetic, geodetic2spherical

Number = Union[int, float]
ArrayLike = Union[Sequence[Number], np.ndarray]


@dataclass
class TECInterpolator:
    """Interpolate TEC values from a GIM on a thin-shell ionospheric layer.

        The interpolator provides:
        - Epoch selection from a multi-epoch GIM product.
        - Computation of ionospheric pierce points (IPPs) at a fixed shell radius.
        - Bilinear interpolation of TEC at requested geographic locations or IPPs.

        Attributes
        ----------
        gim : GIMData
            The gridded ionospheric map container including TEC and optional RMS.
        epoch : pandas.Timestamp | None
            Currently selected epoch within the GIM product; None means inferred
            automatically per query when possible.
        ish : float
            Ionospheric shell height above mean Earth radius [meters].
        R : float
            Assumed mean Earth radius [meters].

        Notes
        -----
        - The shell model is a geometric approximation of the ionosphere; actual
          electron density distribution is not modeled here.
        - Interpolation assumes latitude and longitude axes form a rectilinear grid
          with regular spacing.
        """

    gim: GIMData
    epoch: datetime | None = None  # jeśli None → pierwsza epoka w mapie
    ish: float = 450.0  # Ionospheric Shell Height [km]
    R: float = 6371.0   # Promień Ziemi [km]

    # --------------------------------------------------------------------- public

    def __call__(
        self,
        ev_deg: Sequence[float] | np.ndarray,
        az_deg: Sequence[float] | np.ndarray,
        xyz_m: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Interpolate VTEC at given geographic coordinates and an optional epoch.

                Parameters
                ----------
                lat : ArrayLike
                    Geographic latitude(s) in degrees, broadcastable to `lon`.
                lon : ArrayLike
                    Geographic longitude(s) in degrees, broadcastable to `lat`.
                t : pandas.Timestamp | None, optional
                    Epoch to sample. If None, uses the currently selected `epoch`
                    or the nearest available epoch inferred by `_select_epoch`.

                Returns
                -------
                numpy.ndarray
                    Interpolated VTEC values [TECU], with the shape broadcast from
                    inputs `lat` and `lon`.

                Raises
                ------
                ValueError
                    If no suitable epoch can be selected for interpolation.

                Notes
                -----
                - Longitude normalization (e.g., wrapping to [-180, 180) or [0, 360))
                  is assumed consistent with the underlying GIM grid.
                - If RMS is available in `gim`, this method does not modify or return it.
                """

        ev = np.asarray(ev_deg, dtype=float)
        az = np.asarray(az_deg, dtype=float)

        # --- 1) odbiornik ECEF → geograficzne ---------------------------
        geodetic = ecef2geodetic(xyz_m)  # RADIANS
        geocentric = geodetic2spherical(geodetic)
        lat, lon, h = geocentric[0], geocentric[1], geodetic[2]
        lat_deg, lon_deg = np.rad2deg(lat), np.rad2deg(lon)
        # --- 2) IPP ------------------------------------------------------
        lat_ipp, lon_ipp = self.get_ipp(ev, az, lat_deg, lon_deg, self.ish, self.R)

        # --- 3) bilinearna interp na mapie ------------------------------
        da = self._select_epoch()
        tec_ipp = da.interp(lat=xr.DataArray(lat_ipp, dims="p"), lon=xr.DataArray(lon_ipp, dims="p"), method="linear").values

        # --- 4) funkcja mapująca ----------------------------------------
        ev_rad = np.deg2rad(ev)
        m_f = 1.0 / np.sqrt(1.0 - (self.R / (self.R + self.ish)) ** 2 * (np.cos(ev_rad) ** 2))
        return tec_ipp * m_f

    # ----------------------------------------------------------------- helpers

    def _select_epoch(self) -> xr.DataArray:
        """Select an interpolation epoch from the GIM product.

                This helper chooses the appropriate epoch for interpolation, given
                an explicit `t` or using a default/nearest epoch available in `gim`.

                Parameters
                ----------
                t : pandas.Timestamp | None
                    Requested epoch, or None to infer from internal state or data.

                Returns
                -------
                pandas.Timestamp
                    The selected epoch present in the GIM.

                Raises
                ------
                ValueError
                    If `t` is None and no epoch can be inferred, or if `t` is not
                    compatible with the epochs provided by the GIM.
                """

        if self.epoch is None:
            return self.gim.tec.TEC.isel(time=0)
        return self.gim.tec.TEC.sel(time=self.epoch, method="nearest")


    # ----------------------------------------------------------------- IPP LOGIC

    @staticmethod
    def get_ipp(
        ev: np.ndarray | Sequence[float],
        az: np.ndarray | Sequence[float],
        lat: float | np.ndarray,
        lon: float | np.ndarray,
        ish: float = 450.0,
        R: float = 6371.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ionospheric pierce points (IPPs) on a thin shell.

                Given receiver geographic coordinates and line-of-sight azimuth/elevation,
                this computes the sub-ionospheric latitude and longitude at the
                intersection of the line-of-sight with a spherical shell of height `ish`.

                Parameters
                ----------
                lat : ArrayLike
                    Receiver latitude(s) in degrees.
                lon : ArrayLike
                    Receiver longitude(s) in degrees.
                az : ArrayLike
                    Azimuth(s) of the line-of-sight in degrees (clockwise from North).
                el : ArrayLike
                    Elevation(s) of the line-of-sight in degrees.
                ish : float | None, optional
                    Ionospheric shell height above the mean Earth radius [meters].
                    If None, uses the instance attribute `ish`.
                R: float | None, optional
                    mean earth radius in [km]

                Returns
                -------
                tuple[numpy.ndarray, numpy.ndarray]
                    A pair (lat_ipp, lon_ipp) in degrees on the ionospheric shell.

                Notes
                -----
                - The computation assumes a spherical Earth of radius `R` and a thin
                  ionospheric shell at altitude `ish`.
                - All inputs must be broadcastable to a common shape.
                """
        ev = np.array(ev, dtype=float)
        az = np.array(az, dtype=float)
        ev = np.deg2rad(ev)
        az = np.deg2rad(az)

        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        wi_pp = (np.pi / 2) - ev - np.arcsin((R / (R + ish)) * np.cos(ev))
        fi_pp = np.arcsin(np.sin(lat) * np.cos(wi_pp) + np.cos(lat) * np.sin(wi_pp) * np.cos(az))
        la_pp = lon + np.arcsin(((np.sin(wi_pp) * np.sin(az)) / np.cos(fi_pp)))

        # Konwersja wyników na stopnie
        fi_pp = np.rad2deg(fi_pp)
        la_pp = np.rad2deg(la_pp)
        # Normalize latitude (-90 to 90)
        fi_pp = np.clip(fi_pp, -90, 90)
        # Normalize longitude (-180 to 180)
        la_pp = (la_pp + 180) % 360 - 180

        return fi_pp, la_pp


@dataclass
class GIMData:
    """Container for GIM gridded data and metadata.

        Attributes
        ----------
        tec : xr.DataArray | np.ndarray
            Gridded VTEC values [TECU], indexed by time/latitude/longitude.
        rms : xr.DataArray | np.ndarray | None
            Optional RMS (uncertainty) values per grid cell [TECU].
        dcb : dict | None
            Optional Differential Code Bias information associated with the GIM.

        Notes
        -----
        - The concrete array type may be xarray.DataArray for labeled dimensions
          or numpy.ndarray for unlabeled grids, depending on upstream parsing.
        """

    tec: xr.Dataset
    rms: xr.Dataset | None
    dcb: pd.DataFrame | None

    def __repr__(self) -> str:  # pragma: no cover
        """Return a concise string representation of the GIMData instance.

                The representation includes presence/absence of TEC, RMS, and DCB
                fields and may include basic dimension summaries if available.
                """

        rms_txt = "Tak" if self.rms is not None else "Nie"
        return (
            f"<GIMData: {len(self.tec.time)} epok, TEC shape={self.tec.TEC.shape}, "
            f"RMS obecne: {rms_txt}, DCB: {len(self.dcb)} wpisów>"
        )


@dataclass
class _DCBEntry:
    """Internal container for a single DCB record.

    Attributes
    ----------
    system : str
        GNSS system identifier (e.g., 'G', 'E', 'R', etc.).
    prn_or_site : str
        Satellite PRN or ground site/station identifier.
    bias : float
        Differential Code Bias value in nanoseconds or TECU (as per source).
    rms : float | None
        Optional RMS value for the bias in source units.
    entry_type : str
        Entry type/category (e.g., 'SV' for satellite, 'RCV' for receiver).
    sv : bool
        True if the entry refers to a space vehicle (satellite); False for receiver/site.

    Notes
    -----
    - Units and interpretation of `bias` depend on the DCB source; conversion to
      TECU may be required upstream or downstream.
    """

    system: str
    prn_or_site: str
    bias: float
    rms: float
    entry_type: str
    sv: str | None


class GIMReader:
    """Reader for IONEX/INX Global Ionosphere Maps and optional DCB products.

        This class parses IONEX/INX files to extract:
        - TEC maps (possibly multi-epoch) on a latitude/longitude grid,
        - optional RMS maps,
        - optional DCB entries (satellite/receiver biases).

        Attributes
        ----------
        NLAT : int
            Expected number of latitude steps in the grid (as per IONEX header).
        NLON : int
            Expected number of longitude steps in the grid (as per IONEX header).
        tec_path : str | pathlib.Path | None
            Path to the IONEX/INX file containing TEC maps.
        dcb_path : str | pathlib.Path | None
            Path to a file containing DCB entries compatible with the TEC data.

        Examples
        --------
        - Minimal usage:
          reader = GIMReader(tec_path="COD..._GIM.INX")
          gim = reader.read()

        Notes
        -----
        - The reader expects standard-compliant IONEX/INX files; deviations
          may require extending the private parsing helpers.
        """

    NLAT, NLON = 71, 73  # grid size (lat × lon)

    def __init__(self, tec_path: str | Path, dcb_path: str | Path | None = None):
        """Initialize the reader with input file paths.

                Parameters
                ----------
                tec_path : str | pathlib.Path | None, optional
                    Path to the IONEX/INX file with TEC maps.
                dcb_path : str | pathlib.Path | None, optional
                    Path to the DCB file corresponding to the TEC epoch range.

                Notes
                -----
                - Paths can be set to None and provided later to `read`.
                """

        self.tec_path = Path(tec_path)
        self.dcb_path = Path(dcb_path) if dcb_path else self.tec_path

    # ------------------------------------------------------------------ API

    def read(self,dcb=True) -> GIMData:
        """Read the IONEX/INX TEC maps and optional DCB entries.

                Returns
                -------
                GIMData
                    Parsed TEC grid (and RMS if available). If DCB is provided, it is
                    attached to the resulting container.

                Raises
                ------
                FileNotFoundError
                    If the `tec_path` (or `dcb_path`, when used) does not exist.
                ValueError
                    For malformed IONEX/INX content or inconsistent grid dimensions.
                """

        if dcb:
            dcb_df = self._parse_dcb(self.dcb_path)
        else:
            dcb_df=None
        interval_h = self._parse_interval(self.tec_path)
        tec_ds, rms_ds = self._read_maps(self.tec_path, interval_h)
        return GIMData(tec_ds, rms_ds, dcb_df)

    # -------------------------------------------------- header & DCB

    @staticmethod
    def _parse_interval(path: Path) -> float:
        """Parse the time coverage/interval from IONEX header lines.

                Parameters
                ----------
                header_lines : list[str]
                    Raw header lines extracted from the IONEX/INX file.

                Returns
                -------
                float
                    The interval covered by the product

                Raises
                ------
                ValueError
                    If required header fields are missing or inconsistent.
                """

        with path.open() as f:
            for ln in f:
                if "INTERVAL" in ln:
                    return float(ln.split()[0]) / 3600
        raise ValueError("Brak INTERVAL w nagłówku IONEX.")

    @staticmethod
    def _iterate_dcb_lines(path: Path):
        """Yield or collect DCB-related lines from an open file handle.

                Parameters
                ----------
                fh : io.TextIOBase
                    Open text file handle for the DCB source.

                Returns
                -------
                list[str]
                    Lines containing DCB entries or blocks to be parsed.

                Notes
                -----
                - The exact format depends on the DCB product; this method isolates
                  the subset relevant to downstream parsing in `_parse_dcb`.
                """

        in_aux = False
        with path.open() as f:
            for ln in f:
                if "START OF AUX DATA" in ln:
                    in_aux = True; continue
                if "END OF AUX DATA" in ln:
                    break
                if in_aux:
                    yield ln

    @classmethod
    def _parse_dcb(cls, path: Path) -> pd.DataFrame:
        """Parse DCB entries from raw lines.

                Parameters
                ----------
                lines : list[str]
                    Raw DCB lines filtered by `_iterate_dcb_lines`.

                Returns
                -------
                pd.DataFrame
                    Parsed DCB records with system, identifier, bias and optional RMS.

                Raises
                ------
                ValueError
                    If required fields are missing or lines cannot be interpreted.
                """

        entries: list[_DCBEntry] = []
        for ln in cls._iterate_dcb_lines(path):
            p = ln.split()
            if not p:
                continue
            key = p[0]
            if len(key) >= 2 and key[0] in "GE" and key[1].isdigit():
                entries.append(_DCBEntry(key[0], key[1:], float(p[1]), float(p[2]), "satellite", key))
            elif key in "GE" and len(p) >= 3:
                try:
                    float(p[2]); prn = p[1]; bias, rms = float(p[2]), float(p[3])
                except ValueError:
                    prn = p[1] + p[2]; bias, rms = float(p[3]), float(p[4])
                entries.append(_DCBEntry(key, prn, bias, rms, "station", None))
        df = pd.DataFrame(e.__dict__ for e in entries)
        return df[["entry_type", "system", "prn_or_site", "bias", "rms", "sv"]]

    # -------------------------------------------------- TEC / RMS MAPS

    @staticmethod
    def _flatten(xss):
        """Return a flattened numpy array view/copy of the input.

        Parameters
        ----------
        arr : ArrayLike
            Input array-like object (numpy, xarray, list).

        Returns
        -------
        numpy.ndarray
            Flattened array. If input is an xarray.DataArray, the underlying
            numpy array is extracted before flattening.
        """

        return [x for xs in xss for x in xs]

    def _read_maps(self, path: Path, interval_h: float):
        """Read TEC and optional RMS maps from an open IONEX/INX file handle.

                Parameters
                ----------
                fh : io.TextIOBase
                    Open text file handle for the IONEX/INX TEC product.

                Returns
                -------
                tuple
                    A tuple containing:
                        xarray.Dataset with TEC maps,
                        xarray.Dataset with RMS maps (if available).

                Raises
                ------
                ValueError
                    If the grid shape does not match expected NLAT x NLON, or if
                    mandatory IONEX blocks are malformed.
                """

        break_at = int(24 / interval_h + 1)
        tec, rms = {}, {}

        lines = path.read_text().splitlines()

        # header info
        K = next(int(l.split()[0]) for l in lines if "EXPONENT" in l)
        header_end = next(i for i, l in enumerate(lines) if "END OF HEADER" in l)

        def commit(sec, num, ep, rows):
            if len(rows) != self.NLAT:
                raise ValueError("Niekompletna mapa (NLAT != 71)")
            arr = np.array(rows) * 10 ** K
            (tec if sec == "TEC" else rms)[(num, ep)] = arr

        cur_sec, cur_rows, cur_epoch, map_num = None, [], None, None
        i = header_end + 1
        while i < len(lines):
            ln = lines[i]
            if "START OF TEC MAP" in ln:
                cur_sec, cur_rows, map_num = "TEC", [], int(ln.split()[0]); i += 1; continue
            if "START OF RMS MAP" in ln:
                cur_sec, cur_rows, map_num = "RMS", [], int(ln.split()[0]); i += 1; continue
            if any(tag in ln for tag in ("END OF TEC MAP", "END OF RMS MAP")):
                commit(cur_sec, map_num, cur_epoch, cur_rows)
                if "END OF RMS MAP" in ln and f"{break_at}" in ln:
                    break  # reached end-of-day marker
                cur_sec = None; i += 1; continue
            if cur_sec and "EPOCH OF CURRENT MAP" in ln:
                cur_epoch = datetime(*map(int, ln.split()[:6])); i += 1; continue
            if cur_sec and "LAT/LON1/LON2/DLON/H" in ln:
                raw = [l.split() for l in lines[i + 1 : i + 6]]
                cur_rows.append([int(v) for v in self._flatten(raw)][: self.NLON])
                i += 6; continue
            i += 1

        # build xarray datasets
        lats = np.arange(87.5, -87.5 - 0.1, -2.5)
        lons = np.linspace(-180, 180, self.NLON)

        def build(grid_dict: Dict[Tuple[int, datetime], np.ndarray], varname: str):
            if not grid_dict:
                return None
            times = [epoch for (_, epoch) in grid_dict]
            data = np.stack(list(grid_dict.values()), axis=0)
            da = xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords=dict(time=times, lat=lats, lon=lons),
                name=varname,
            )
            if varname == "RMS":
                da.attrs["units"] = "TECU rms"
            else:
                da.attrs["units"] = "TECU"
            return da.to_dataset()

        return build(tec, "TEC"), build(rms, "RMS")
