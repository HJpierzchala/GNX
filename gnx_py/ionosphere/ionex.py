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
- The reader follows the IONEX 1.0/1.1 format (Schaer, Gurtner, Feltens 1998):
  header records are identified by their label in columns 61-80, the map grid
  is built from LAT1/LAT2/DLAT and LON1/LON2/DLON, values are scaled with
  EXPONENT (default -1) and the "no value" flag 9999 becomes NaN. Plain and
  compressed (.INX.gz, .Z, bzip2, zip; gnx_py.compression) files are read.
- Only 2-D maps (single layer at HGT1) are supported. The single-layer height
  HGT1 and the BASE RADIUS of the file define the thin shell used for the
  IPP. The mapping function follows the producer by default (``mapping="auto"``):
  the CODE modified single-layer model (MSLM, H = 506.7 km, alpha = 0.9782) when
  the header DESCRIPTION states it, the single layer at HGT1 for COSZ
  (:func:`mapping_slm`, :func:`mapping_mslm`, :func:`resolve_gim_mapping`).
- Interpolation: bilinear in latitude/longitude, linear in time between
  consecutive maps, earth-fixed (IONEX eq. 2) or rotated by the Earth's
  rotation between the map epochs (eq. 3, ``rotate=True``; the IONEX
  recommendation and the session default, ``Config.gim_time_interpolation``).
  Map epochs are UTC; the sessions convert their GPST epochs before
  interpolating. IPPs beyond the outermost grid
  latitude of a global map (e.g. +-87.5 deg) take the value of that row
  (nearest-row extrapolation, as RTKLIB interptec falls back to the nearest
  grid node); outside a regional grid the value is NaN.
"""

from __future__ import annotations

import functools
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr
from ..core import _notes
from ..core.conversion import ecef2geodetic

Number = Union[int, float]
ArrayLike = Union[Sequence[Number], np.ndarray]

#: IONEX "no value available" flag of TEC/RMS records.
IONEX_NO_VALUE = 9999
#: Exponent used when the (optional) EXPONENT record is absent (IONEX 1.0).
IONEX_DEFAULT_EXPONENT = -1
#: Default thin shell when a GIM header gives none [km].
DEFAULT_SHELL_HEIGHT_KM = 450.0
DEFAULT_BASE_RADIUS_KM = 6371.0

_DCB_COLUMNS = ["entry_type", "system", "prn_or_site", "bias", "rms", "sv"]
_DCB_SYSTEMS = ("G", "E", "C")       # systems whose IONEX DCBs are consumed downstream
_GNSS_LETTERS = "GRECJIS"            # valid IONEX/RINEX system identifiers
_SAT_TOKEN = re.compile(r"^[A-Z]\d{2}$")

#: Code pair the IONEX DCBs refer to when the header states none. IONEX 1.0 defines the
#: DCB block as P1-P2 (GPS C1W-C2W). Galileo and BeiDou DCBs without a statement (CAS
#: maps, block "[P1-P2]") are E1/E5a and B1I/B2I: on 2024-035 the CAS map agrees with the
#: CAS DSB product C1C-C5Q / C2I-C7I to 0.22 / 0.30 ns (demeaned std), against
#: C1C-C7Q 0.39 ns and C2I-C6I 5.7 ns. The E and C entries are thus inferred from CAS: a map
#: of another centre with unstated Galileo/BeiDou DCBs of another pair (e.g. C2I-C6I) would
#: be taken for these pairs; such a file needs its 'Reference observables' statement.
IONEX_DEFAULT_DCB_REFERENCE: Dict[str, Tuple[str, str]] = {
    "G": ("C1W", "C2W"), "E": ("C1C", "C5Q"), "C": ("C2I", "C7I"),
}
# CODE: 'Reference observables for GPS    : C1W-C2W' (satellites of a system) and
# 'Reference observables for BRUX 13101M010 (G): C1W-C2W' (one station) COMMENT records
_DCB_REFERENCE = re.compile(r"^Reference observables for (.+?)\s*:\s*([A-Z]\d[A-Z])\s*-\s*([A-Z]\d[A-Z])\s*$",
                            re.IGNORECASE)
_DCB_REFERENCE_STATION = re.compile(r"^(.+?)\s*\(([A-Z])\)$")
_SYSTEM_BY_NAME = {"GPS": "G", "GALILEO": "E", "BEIDOU": "C", "BDS": "C", "COMPASS": "C",
                   "GLONASS": "R", "QZSS": "J", "IRNSS": "I", "NAVIC": "I", "SBAS": "S"}

#: Mapping height of the CODE modified single-layer model (MSLM) [km] (Schaer 1999).
#: The IPP stays on the IONEX shell HGT1 (450 km); only the mapping uses this height.
MSLM_HEIGHT_KM = 506.7
#: Zenith-angle factor alpha of the MSLM (CODE, Schaer 1999; NTCM-G eq. 32-33).
MSLM_ALPHA = 0.9782
#: Accepted values of ``TECInterpolator(mapping=...)`` / ``Config.gim_mapping_function``.
GIM_MAPPING_FUNCTIONS = ("auto", "slm", "mslm")


# ---------------------------------------------------------------------------
# mapping functions (slant/vertical TEC ratio)
# ---------------------------------------------------------------------------

def _check_shell(R: float, H: float) -> Tuple[float, float]:
    R, H = float(R), float(H)
    if not (np.isfinite(R) and R > 0.0) or not (np.isfinite(H) and H >= 0.0):
        raise ValueError(f"mapping function needs R > 0 and H >= 0 [km], got R={R:g}, H={H:g}")
    if R > 1.0e5 or H > 1.0e5:
        raise ValueError(f"mapping function takes R and H in km, got R={R:g}, H={H:g} (metres?)")
    return R, H


def mapping_slm(ev, R: float = DEFAULT_BASE_RADIUS_KM, H: float = DEFAULT_SHELL_HEIGHT_KM) -> np.ndarray:
    """Single-layer (SLM) mapping function STEC/VTEC = 1/cos z'.

    sin z' = R/(R+H) cos E (IONEX 1.0 "COSZ" with the shell height H).

    Parameters
    ----------
    ev : array_like
        Elevation [deg].
    R : float
        Earth radius [km] (IONEX BASE RADIUS, default 6371).
    H : float
        Shell height [km] (IONEX HGT1, default 450).
    """
    R, H = _check_shell(R, H)
    e = np.deg2rad(np.asarray(ev, dtype=float))
    return 1.0 / np.sqrt(1.0 - (R / (R + H) * np.cos(e)) ** 2)


def mapping_mslm(ev, R: float = DEFAULT_BASE_RADIUS_KM, H: float = MSLM_HEIGHT_KM,
                 alpha: float = MSLM_ALPHA) -> np.ndarray:
    """Modified single-layer (MSLM) mapping function STEC/VTEC = 1/cos z'.

    sin z' = R/(R+H) sin(alpha z), z = 90 deg - E (Schaer 1999). The defaults are
    the CODE GIM model (H = 506.7 km, alpha = 0.9782; the IPP stays at HGT1 = 450 km).
    NTCM-G (model description issue 1.0, eq. 32-33) uses the same form with H = 450 km.

    Parameters
    ----------
    ev : array_like
        Elevation [deg].
    R : float
        Earth radius [km] (default 6371).
    H : float
        Mapping height [km] (default 506.7, CODE).
    alpha : float
        Zenith-angle factor (default 0.9782).
    """
    R, H = _check_shell(R, H)
    z = np.deg2rad(90.0 - np.asarray(ev, dtype=float))
    return 1.0 / np.sqrt(1.0 - (R / (R + H) * np.sin(float(alpha) * z)) ** 2)


_MSLM_DESCRIPTION = re.compile(r"MODIFIED\s+SINGLE[\s-]*LAYER", re.IGNORECASE)


def resolve_gim_mapping(gim, mapping: str = "auto", notes=None) -> str:
    """Mapping function ("slm" or "mslm") of a GIM for the setting ``mapping``.

    "slm" and "mslm" are returned as given. "auto" follows the producer: "mslm"
    when the header DESCRIPTION states a modified single-layer model (CODE:
    "a modified single-layer model mapping function approximating the JPL
    extended slab model", MAPPING FUNCTION NONE), "slm" for MAPPING FUNCTION
    COSZ (IGS, CAS), otherwise "slm" with one UserWarning per file and run: the
    warn-once keys live on ``notes`` (``gnx_py._notes``; the sessions pass their
    configuration, cleared at the start of each run), or on ``gim`` when
    ``notes`` is None (before, once per file and process).
    """
    if mapping not in GIM_MAPPING_FUNCTIONS:
        raise ValueError(f"GIM mapping must be one of {list(GIM_MAPPING_FUNCTIONS)} (got {mapping!r})")
    if mapping != "auto":
        return mapping
    description = getattr(gim, "description", "") or ""
    mf = getattr(gim, "mapping_function", None)
    if _MSLM_DESCRIPTION.search(description):
        return "mslm"
    if mf == "COSZ":
        return "slm"
    src = getattr(gim, "source", None)
    key = ("gim_auto_mapping", str(src) if src is not None else id(gim))
    where = f" {src}" if src is not None else ""
    _notes.warn_once(gim if notes is None else notes, key,
              f"GIM{where}: MAPPING FUNCTION {mf if mf else 'absent'} and no modified single-layer "
              f"model in DESCRIPTION; the single-layer mapping (COSZ at HGT1) is used. Set "
              f"gim_mapping_function='slm' or 'mslm' to choose explicitly", UserWarning, stacklevel=3)
    return "slm"


# ---------------------------------------------------------------------------
# low-level IONEX helpers
# ---------------------------------------------------------------------------

def _read_ionex_lines(path: str | Path) -> list[str]:
    """Return the lines of a plain or compressed IONEX file.

    The downloader delivers GIMs as ``*.INX.gz``; gzip, Unix compress (.Z), bzip2 and zip are undone in memory
    by :mod:`gnx_py.compression`, detected from the file magic, not from the suffix (before v1.0.0 a .Z
    file was refused). IONEX is ASCII, latin-1 decoding never fails on stray bytes in COMMENT/DESCRIPTION
    records. A damaged compressed file raises ``compression.CompressedFileError`` (a ValueError naming it).
    """
    from ..io.compression import read_bytes

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IONEX file not found: {path}")
    try:
        return read_bytes(path, what="IONEX file").decode("latin-1").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read IONEX file {path}: {exc}") from exc


def _label(line: str) -> str:
    """IONEX record label (columns 61-80)."""
    return line[60:80].strip()


def _fixed_floats(body: str, n: int) -> list[float]:
    """Parse ``2X,nF6.1`` (grid header and LAT/LON1/LON2/DLON/H records).

    Falls back to whitespace splitting for files that do not respect the
    fixed columns.
    """
    try:
        return [float(body[2 + 6 * k: 8 + 6 * k]) for k in range(n)]
    except ValueError:
        vals = [float(v) for v in body.split()[:n]]
        if len(vals) != n:
            raise ValueError(f"expected {n} numbers, got {len(vals)}")
        return vals


def _ionex_epoch(body: str) -> datetime:
    """Parse an IONEX epoch record (6I6; some producers write F6.2 seconds)."""
    p = body.split()
    if len(p) < 5:
        raise ValueError(f"expected 'year month day hour min sec', got {body.strip()!r}")
    y, mo, d, h, mi = (int(float(v)) for v in p[:5])
    s = float(p[5]) if len(p) > 5 else 0.0
    # hour 24 is written by some producers for the end-of-day map
    return datetime(y, mo, d) + timedelta(hours=h, minutes=mi, seconds=s)


def _i5_values(line: str) -> list[int]:
    """Parse one data line of a TEC/RMS map (16I5, free format accepted)."""
    s = line.rstrip()
    try:
        return [int(s[k:k + 5]) for k in range(0, len(s), 5) if s[k:k + 5].strip()]
    except ValueError:
        return [int(v) for v in s.split()]


def _grid_axis(first: float, last: float, step: float, what: str) -> np.ndarray:
    """Regular grid axis first..last (inclusive) with spacing step."""
    if step == 0.0:
        if first != last:
            raise ValueError(f"{what}: step 0 with {first} != {last}")
        return np.array([first])
    n = (last - first) / step
    if n < 0 or abs(n - round(n)) > 1e-6:
        raise ValueError(f"{what}: {first}/{last}/{step} is not a regular grid "
                         f"({n:.4f} steps between the first and last node)")
    return np.round(first + step * np.arange(int(round(n)) + 1), 6)


def _cell(axis: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lower node index and fraction along an ascending axis (fraction NaN outside)."""
    i = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, axis.size - 2)
    f = (x - axis[i]) / (axis[i + 1] - axis[i])
    f = np.where((x < axis[0]) | (x > axis[-1]), np.nan, f)
    return i, f


def _wsum(values, weights) -> np.ndarray:
    """Weighted sum where a corner with zero weight does not propagate its NaN."""
    out = 0.0
    for v, w in zip(values, weights):
        out = out + np.where(w == 0.0, 0.0, w * v)
    return np.asarray(out, dtype=float)


def interpolate_gim(da: xr.DataArray, lat: ArrayLike, lon: ArrayLike, times=None,
                    rotate: bool = False) -> np.ndarray:
    """Interpolate a GIM field (TEC or RMS) at IPP latitude/longitude [deg].

    Parameters
    ----------
    da : xarray.DataArray
        Map(s) with dims ``(time, lat, lon)`` or ``(lat, lon)``.
    lat, lon : ArrayLike
        IPP coordinates in degrees (broadcast with ``times``).
    times : ArrayLike of datetime64, optional
        Epochs; required when ``da`` has a time dimension with several maps.
        Epochs outside the map span give NaN (no extrapolation in time).
    rotate : bool, default False
        False: linear interpolation between consecutive earth-fixed maps
        (IONEX 1.0 eq. 2, used by GNX). True: each bracketing map is
        evaluated at the longitude rotated with the Sun, ``lon + 360 deg *
        (t - T_i) / 1 day`` (IONEX eq. 3, RTKLIB ``iontec`` with opt bit 0).
        Applied only to grids that are global in longitude: on a regional grid
        the rotated longitudes (up to +-15 deg with 2 h maps) leave the grid, so
        it is interpolated earth-fixed with a UserWarning.

    Notes
    -----
    - Bilinear in latitude/longitude; a cell corner equal to 9999 (NaN) gives
      NaN unless its weight is zero.
    - Polar rule: latitudes beyond an outermost grid row that lies within one
      DLAT of the pole (IONEX global maps stop at +-87.5 deg) are clamped to
      that row. The cap beyond is < 280 km wide and the polar VTEC gradient
      is small; RTKLIB interptec likewise takes the nearest grid node outside
      the grid.
    - Longitudes are wrapped onto the grid; a global grid without the closing
      meridian (e.g. -180..175) gets it appended. Outside a regional grid (in
      latitude or longitude) the result is NaN.
    """
    grid = np.asarray(da.values, dtype=float)
    has_time = "time" in da.dims
    if not has_time:
        grid = grid[None]
    lats = np.asarray(da["lat"].values, dtype=float)
    lons = np.asarray(da["lon"].values, dtype=float)
    if lats.size < 2 or lons.size < 2:
        raise ValueError("GIM interpolation needs at least two grid nodes in latitude and longitude")
    if lats[0] > lats[-1]:
        lats, grid = lats[::-1], grid[:, ::-1, :]
    if lons[0] > lons[-1]:
        lons, grid = lons[::-1], grid[:, :, ::-1]
    dlon = lons[1] - lons[0]
    if abs(lons[-1] + dlon - lons[0] - 360.0) < 1e-6:
        lons = np.append(lons, lons[0] + 360.0)
        grid = np.concatenate([grid, grid[:, :, :1]], axis=2)
    if rotate and abs(lons[-1] - lons[0] - 360.0) > 1e-6:
        warnings.warn(f"GIM grid covers longitudes {lons[0]:g} to {lons[-1]:g} deg only: rotated time "
                      "interpolation (IONEX eq. 3) needs a global grid; earth-fixed interpolation is used",
                      UserWarning, stacklevel=2)
        rotate = False

    if times is None:
        if grid.shape[0] != 1:
            raise ValueError("times are required to interpolate a multi-epoch GIM")
        lat_b, lon_b = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        t = None
    else:
        t_in = np.asarray(times, dtype="datetime64[ns]").astype(np.int64)
        lat_b, lon_b, t = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float), t_in)

    # polar rule (see Notes): clamp only towards a pole whose outermost row is within one
    # DLAT of it (global maps stop at +-87.5 deg); a regional grid edge gives NaN as in longitude
    dlat = lats[1] - lats[0]
    lat_lo = lats[0] if lats[0] + 90.0 <= dlat + 1e-6 else -np.inf
    lat_hi = lats[-1] if 90.0 - lats[-1] <= dlat + 1e-6 else np.inf
    la = np.clip(lat_b, lat_lo, lat_hi)
    ia, fa = _cell(lats, la)

    def bil(m, shift_deg=0.0):
        lo = lons[0] + np.mod(lon_b + shift_deg - lons[0], 360.0)
        io, fo = _cell(lons, lo)
        corners = (grid[m, ia, io], grid[m, ia + 1, io], grid[m, ia, io + 1], grid[m, ia + 1, io + 1])
        return _wsum(corners, ((1 - fa) * (1 - fo), fa * (1 - fo), (1 - fa) * fo, fa * fo))

    if t is None:
        return bil(np.zeros(la.shape, dtype=int))

    te = np.asarray(da["time"].values, dtype="datetime64[ns]").astype(np.int64) if has_time else None
    if te is None or te.size == 1:
        out = bil(np.zeros(la.shape, dtype=int))
        if te is not None:
            out = np.where(t == te[0], out, np.nan)
        return out
    k = np.clip(np.searchsorted(te, t, side="right") - 1, 0, te.size - 2)
    w = (t - te[k]) / (te[k + 1] - te[k])
    s0 = 360.0 * (t - te[k]) / 86400e9 if rotate else 0.0
    s1 = 360.0 * (t - te[k + 1]) / 86400e9 if rotate else 0.0
    out = _wsum((bil(k, s0), bil(k + 1, s1)), (1.0 - w, w))
    return np.where((t >= te[0]) & (t <= te[-1]), out, np.nan)


class _ShellMethod:
    """``get_ipp`` usable on the class (450/6371 km default shell, the historic
    static behaviour) and on an instance (the IONEX HGT1 / BASE RADIUS of its GIM)."""

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func

        @functools.wraps(self.func)
        def bound(ev, az, lat, lon, ish=None, R=None):
            return self.func(ev, az, lat, lon, obj.ish if ish is None else ish, obj.R if R is None else R)

        return bound


@dataclass
class TECInterpolator:
    """Interpolate TEC values from a GIM on a thin-shell ionospheric layer.

        Status
        ------
        Active utility used by GIM-based workflows (``DDPreprocessing.read_inx``
        for SPP, PPP and TEC sessions). Requires validation when changing IPP
        geometry or mapping function assumptions.

        The interpolator provides:
        - Epoch selection from a multi-epoch GIM product.
        - Computation of ionospheric pierce points (IPPs) on the IONEX shell.
        - Bilinear interpolation of TEC at requested geographic locations or IPPs.

        Attributes
        ----------
        gim : GIMData
            The gridded ionospheric map container including TEC and optional RMS.
        epoch : datetime | None
            Epoch of the map used by ``__call__``; None means the first map.
        ish : float | None
            Ionospheric shell height [km]; None takes HGT1 of the IONEX header.
        R : float | None
            Earth radius of the shell model [km]; None takes BASE RADIUS of the
            IONEX header.
        mapping : {"auto", "slm", "mslm"}
            Mapping function STEC/VTEC. "slm": single layer at the shell height
            (:func:`mapping_slm`, IONEX "COSZ"); "mslm": CODE modified single
            layer (:func:`mapping_mslm`, H = 506.7 km, alpha = 0.9782); "auto"
            (default): the producer's function, MSLM when the header DESCRIPTION
            states a modified single-layer model (CODE), SLM for MAPPING FUNCTION
            COSZ (IGS, CAS), otherwise SLM with one warning per file and run.
        notes : object | None
            Owner of that warning's warn-once key (``gnx_py._notes``): the
            sessions pass their configuration (once per run); None keeps it on
            ``gim`` (once per GIM object).

        Notes
        -----
        - The shell model is a geometric approximation of the ionosphere; actual
          electron density distribution is not modeled here.
        - The IPP is always on the shell ``ish`` (HGT1, 450 km); only the MSLM
          mapping uses its own height.
        """

    gim: GIMData
    epoch: datetime | None = None  # None -> first map of the product
    ish: float | None = None       # Ionospheric shell height [km]; None -> IONEX HGT1
    R: float | None = None         # Earth radius [km]; None -> IONEX BASE RADIUS
    mapping: str = "auto"          # "auto" (by header) | "slm" | "mslm"
    # owner of the warn-once keys of the "auto" warning (gnx_py._notes; a session's configuration), None: the GIM
    notes: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.ish is None:
            self.ish = float(getattr(self.gim, "height_km", DEFAULT_SHELL_HEIGHT_KM))
        if self.R is None:
            self.R = float(getattr(self.gim, "base_radius_km", DEFAULT_BASE_RADIUS_KM))
        self.mapping_used = resolve_gim_mapping(self.gim, self.mapping, notes=self.notes)

    # --------------------------------------------------------------------- public

    def __call__(
        self,
        ev_deg: Sequence[float] | np.ndarray,
        az_deg: Sequence[float] | np.ndarray,
        xyz_m: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Interpolate slant TEC from GIM for receiver/satellite geometry.

                Parameters
                ----------
                ev_deg : ArrayLike
                    Satellite elevation angle(s) in degrees.
                az_deg : ArrayLike
                    Satellite azimuth angle(s) in degrees.
                xyz_m : ArrayLike
                    Receiver ECEF coordinates in meters.

                Returns
                -------
                numpy.ndarray
                    Interpolated slant TEC values [TECU] after thin-shell IPP
                    interpolation and elevation mapping, on the map selected by
                    ``epoch``.

                Notes
                -----
                - The receiver geodetic latitude/longitude [deg] enter the
                  spherical IPP geometry directly (as in RTKLIB ``ionppp``).
                - If RMS is available in `gim`, this method does not modify or return it.
                """

        ev = np.asarray(ev_deg, dtype=float)
        az = np.asarray(az_deg, dtype=float)

        # receiver ECEF -> geodetic latitude/longitude in DEGREES
        lat_deg, lon_deg = ecef2geodetic(xyz_m)[:2]
        lat_ipp, lon_ipp = self.get_ipp(ev, az, lat_deg, lon_deg)
        tec_ipp = interpolate_gim(self._select_epoch(), lat_ipp, lon_ipp)
        return tec_ipp * self.mapping_function(ev)

    def vtec(self, times, lat_ipp: ArrayLike, lon_ipp: ArrayLike, field: str = "TEC",
             rotate: bool = False) -> np.ndarray:
        """VTEC (or its RMS, ``field="RMS"``) [TECU] at IPPs and epochs.

        Linear in time between consecutive maps, bilinear in space; see
        :func:`interpolate_gim` for the polar rule and ``rotate``. ``times`` are
        taken as UTC (IONEX map epochs) and ``rotate`` defaults to False
        (earth-fixed); the sessions go through ``tools.gim_slant_tec``, which
        converts GPST to UTC and rotates by default (``Config.gim_time_interpolation``).
        """
        ds = self.gim.tec if field == "TEC" else self.gim.rms
        if ds is None:
            src = f" {self.gim.source}" if getattr(self.gim, "source", None) is not None else ""
            raise ValueError(f"GIM{src} has no {field} maps")
        return interpolate_gim(ds[field], lat_ipp, lon_ipp, times=times, rotate=rotate)

    def mapping_function(self, ev_deg: ArrayLike) -> np.ndarray:
        """Mapping function STEC/VTEC of ``mapping_used``.

        "slm": 1/cos z', sin z' = R/(R+H) cos E with the shell height H = ``ish``;
        "mslm": sin z' = R/(R+506.7 km) sin(0.9782 z) (CODE, Schaer 1999).
        """
        if self.mapping_used == "mslm":
            return mapping_mslm(ev_deg, R=self.R)
        return mapping_slm(ev_deg, R=self.R, H=self.ish)

    # ----------------------------------------------------------------- helpers

    def _select_epoch(self) -> xr.DataArray:
        """Select the map used by ``__call__``.

                Returns
                -------
                xarray.DataArray
                    The first map when ``epoch`` is None, otherwise the map
                    nearest to ``epoch``.
                """

        if self.epoch is None:
            return self.gim.tec.TEC.isel(time=0)
        return self.gim.tec.TEC.sel(time=self.epoch, method="nearest")

    # ----------------------------------------------------------------- IPP LOGIC

    @_ShellMethod
    def get_ipp(
        ev: np.ndarray | Sequence[float],
        az: np.ndarray | Sequence[float],
        lat: float | np.ndarray,
        lon: float | np.ndarray,
        ish: float = DEFAULT_SHELL_HEIGHT_KM,
        R: float = DEFAULT_BASE_RADIUS_KM,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ionospheric pierce points (IPPs) on a thin shell.

                Given receiver geographic coordinates and line-of-sight azimuth/elevation,
                this computes the sub-ionospheric latitude and longitude at the
                intersection of the line-of-sight with a spherical shell of height `ish`.

                Called on the class (``TECInterpolator.get_ipp``) the default
                450/6371 km shell is used; called on an instance, ``ish`` and
                ``R`` default to the instance values (IONEX HGT1 / BASE RADIUS).

                Parameters
                ----------
                ev : ArrayLike
                    Elevation(s) of the line-of-sight in degrees.
                az : ArrayLike
                    Azimuth(s) of the line-of-sight in degrees (clockwise from North).
                lat : ArrayLike
                    Receiver latitude(s) in degrees.
                lon : ArrayLike
                    Receiver longitude(s) in degrees.
                ish : float, optional
                    Ionospheric shell height above the Earth radius [km].
                R: float, optional
                    Earth radius [km].

                Returns
                -------
                tuple[numpy.ndarray, numpy.ndarray]
                    A pair (lat_ipp, lon_ipp) in degrees on the ionospheric shell.

                Notes
                -----
                - The computation assumes a spherical Earth of radius `R` and a thin
                  ionospheric shell at altitude `ish`.
                - A line of sight crossing the pole (high-latitude receivers looking
                  poleward) puts the IPP on the far meridian, lon + 180 - asin(...),
                  as in RTKLIB ``ionppp``; elsewhere the result is identical to
                  ``gnx_py.ionosphere.utils.get_ipp``.
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
        dla = np.arcsin(((np.sin(wi_pp) * np.sin(az)) / np.cos(fi_pp)))
        # beyond the pole the longitude difference exceeds 90 deg (cos(dlon) < 0) and asin
        # returns the wrong branch
        over_pole = (np.cos(lat) * np.cos(wi_pp) - np.sin(lat) * np.sin(wi_pp) * np.cos(az)) < 0.0
        la_pp = lon + np.where(over_pole, np.pi - dla, dla)

        # radians -> degrees
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

        Status
        ------
        Active data container returned by ``GIMReader``.

        Attributes
        ----------
        tec : xr.Dataset
            Gridded VTEC values [TECU] (variable ``TEC``), indexed by
            time/lat/lon; 9999 ("no value") is NaN.
        rms : xr.Dataset | None
            RMS maps [TECU] (variable ``RMS``); None when the file has none
            (RMS maps are optional in IONEX).
        dcb : pd.DataFrame | None
            DCB records of the IONEX AUX block (columns entry_type, system,
            prn_or_site, bias, rms, sv); an empty frame with these columns when
            the file has no DCB block; None when DCBs were not read.
        height_km : float
            Single-layer height HGT1 of the IONEX header [km].
        base_radius_km : float
            BASE RADIUS of the IONEX header [km].
        source : pathlib.Path | None
            The IONEX file the maps were read from.
        dcb_reference : DCBReference | None
            Code pairs the DCBs refer to (header statements, else the IONEX
            defaults); None when DCBs were not read.
        mapping_function : str | None
            MAPPING FUNCTION record of the header ('COSZ', 'QFAC', 'NONE', ...);
            None when the header has none.
        description : str
            DESCRIPTION records of the header, joined with single spaces ('' when
            absent). The CODE maps state their modified single-layer mapping here.
        """

    tec: xr.Dataset
    rms: xr.Dataset | None
    dcb: pd.DataFrame | None
    height_km: float = DEFAULT_SHELL_HEIGHT_KM
    base_radius_km: float = DEFAULT_BASE_RADIUS_KM
    source: Path | None = None
    dcb_reference: DCBReference | None = None
    mapping_function: str | None = None
    description: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        """Return a concise string representation of the GIMData instance."""

        rms_txt = "yes" if self.rms is not None else "no"
        n_dcb = "not read" if self.dcb is None else f"{len(self.dcb)} records"
        return (
            f"<GIMData: {len(self.tec.time)} maps, TEC shape={self.tec.TEC.shape}, "
            f"RMS maps: {rms_txt}, DCB: {n_dcb}, HGT1={self.height_km} km>"
        )

    def time_span(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """First and last map epoch."""
        t = pd.DatetimeIndex(self.tec.time.values)
        return t.min(), t.max()

    def check_coverage(self, start, end, what: str = "observations") -> None:
        """Raise ValueError unless [start, end] lies within the map epochs.

        VTEC is interpolated in time between maps only, so every epoch must lie
        between the first and the last map (a daily file 00:00-24:00 with the
        24:00 map covers the whole day).
        """
        first, last = self.time_span()
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start < first or end > last:
            src = f" {self.source}" if self.source is not None else ""
            raise ValueError(
                f"GIM{src} does not cover the {what}: IONEX maps span {first} to {last}, "
                f"the {what} span {start} to {end}. Use the IONEX file of the observation date (gim_path)."
            )


@dataclass
class _DCBEntry:
    """Internal container for a single DCB record.

    Attributes
    ----------
    system : str
        GNSS sys identifier (e.g., 'G', 'E', 'R', etc.).
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


@dataclass(frozen=True)
class DCBReference:
    """Code pairs the DCBs of an IONEX AUX block refer to.

    ``satellite`` maps a system ('G', 'E', 'C') to the observables stated for its
    satellites, ``station`` maps (site as in ``GIMData.dcb['prn_or_site']``, e.g.
    'BRUX13101M010', system) to the observables stated for one receiver. Only
    explicit 'Reference observables' COMMENT records (CODE) are stored; ``codes``
    falls back to the station statement, the system statement and
    ``IONEX_DEFAULT_DCB_REFERENCE`` in this order.
    """

    satellite: Dict[str, Tuple[str, str]]
    station: Dict[Tuple[str, str], Tuple[str, str]]

    def codes(self, system: str, station: str | None = None) -> Tuple[str, str] | None:
        """Reference code pair of the satellite (``station=None``) or station DCBs of ``system``."""
        if station:
            key = str(station).replace(" ", "").upper()
            for (site, sys_), pair in self.station.items():
                if sys_ == system and site.upper().startswith(key):
                    return pair
        return self.satellite.get(system, IONEX_DEFAULT_DCB_REFERENCE.get(system))


@dataclass(frozen=True)
class _IonexHeader:
    """Header values needed to read the maps."""

    end: int                    # index of the END OF HEADER line
    interval_s: float           # 0 = irregular map spacing
    n_maps: int | None
    exponent: int
    height_km: float
    base_radius_km: float
    lats: np.ndarray
    lons: np.ndarray
    mapping_function: str | None = None   # MAPPING FUNCTION record (COSZ, QFAC, NONE, ...)
    description: str = ""                  # DESCRIPTION records joined with single spaces


class GIMReader:
    """Reader for IONEX/INX Global Ionosphere Maps and optional DCB products.

        Status
        ------
        Active parser used by ionosphere workflows and by PPP when measuring
        STEC. Requires caution: DCB parsing feeds satellite/station bias policy
        and currently supports GPS, Galileo and BeiDou records.

        This class parses IONEX/INX files (plain or compressed) to extract:
        - TEC maps (possibly multi-epoch) on the latitude/longitude grid of the header,
        - optional RMS maps,
        - optional DCB entries (satellite/receiver biases).

        Attributes
        ----------
        NLAT : int
            Number of latitude rows of the standard 2.5 deg global grid (legacy
            constant; the reader builds the grid from the header).
        NLON : int
            Number of longitude columns of the standard 5 deg global grid
            (legacy constant).
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
        - Records are identified by their label (columns 61-80). Missing RMS
          maps give ``rms=None``; a missing DCB block gives an empty DCB frame.
        - Errors name the file and the line number.
        """

    NLAT, NLON = 71, 73  # standard global grid (87.5..-87.5 / 2.5, -180..180 / 5)

    def __init__(self, tec_path: str | Path, dcb_path: str | Path | None = None):
        """Initialize the reader with input file paths.

                Parameters
                ----------
                tec_path : str | pathlib.Path
                    Path to the IONEX/INX file with TEC maps (``.gz`` allowed).
                dcb_path : str | pathlib.Path | None, optional
                    Path to the IONEX file whose AUX block holds the DCBs;
                    defaults to ``tec_path``.
                """

        self.tec_path = Path(tec_path)
        self.dcb_path = Path(dcb_path) if dcb_path else self.tec_path

    # ------------------------------------------------------------------ API

    def read(self, dcb=True) -> GIMData:
        """Read the IONEX/INX TEC maps and optional DCB entries.

                Returns
                -------
                GIMData
                    Parsed TEC grid (and RMS if available), the shell height
                    HGT1 and BASE RADIUS. With ``dcb=True`` the DCB records of
                    the AUX block (empty frame when the file has none).

                Raises
                ------
                FileNotFoundError
                    If the `tec_path` (or `dcb_path`, when used) does not exist.
                ValueError
                    For malformed IONEX/INX content (message names file and line).
                """

        lines = _read_ionex_lines(self.tec_path)
        header = self._parse_header(lines, self.tec_path)
        if dcb:
            dcb_lines = lines if self.dcb_path == self.tec_path else _read_ionex_lines(self.dcb_path)
            dcb_df = self._parse_dcb_lines(dcb_lines, self.dcb_path)
            dcb_ref = self._parse_dcb_reference(dcb_lines)
        else:
            dcb_df = dcb_ref = None
        tec_ds, rms_ds = self._parse_maps(lines, header, self.tec_path)
        return GIMData(tec_ds, rms_ds, dcb_df, height_km=header.height_km,
                       base_radius_km=header.base_radius_km, source=self.tec_path, dcb_reference=dcb_ref,
                       mapping_function=header.mapping_function, description=header.description)

    # -------------------------------------------------- header & DCB

    @staticmethod
    def _parse_header(lines: list[str], path: Path) -> _IonexHeader:
        """Parse the IONEX header records needed for the maps.

                Raises
                ------
                ValueError
                    If a required record (INTERVAL, HGT1 / HGT2 / DHGT,
                    LAT1 / LAT2 / DLAT, LON1 / LON2 / DLON) is missing or invalid,
                    or the file holds 3-D maps.
                """

        rec: Dict[str, Tuple[int, str]] = {}
        description: list[str] = []
        end = None
        for i, ln in enumerate(lines):
            lab = _label(ln)
            if lab == "END OF HEADER":
                end = i
                break
            if lab == "DESCRIPTION":
                description.append(ln[:60].strip())
            rec.setdefault(lab, (i, ln[:60]))
        if end is None:
            raise ValueError(f"{path}: no END OF HEADER record (not an IONEX file?)")

        def get(label, parse, required=True, default=None):
            if label not in rec:
                if required:
                    raise ValueError(f"{path}: IONEX header has no {label} record")
                return default
            i, body = rec[label]
            try:
                return parse(body)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}, line {i + 1}: invalid {label} record {body.strip()!r} ({exc})") from None

        interval = get("INTERVAL", lambda b: float(b.split()[0]))
        if interval < 0:
            i = rec["INTERVAL"][0]
            raise ValueError(f"{path}, line {i + 1}: INTERVAL {interval:g} s is negative "
                             "(seconds between maps; 0 = irregular spacing)")
        dim = get("MAP DIMENSION", lambda b: int(b.split()[0]), required=False, default=2)
        if dim != 2:
            raise ValueError(f"{path}: MAP DIMENSION {dim}: only 2-D (single-layer) IONEX maps are supported")
        h1, h2, dh = get("HGT1 / HGT2 / DHGT", lambda b: _fixed_floats(b, 3))
        if h1 != h2 or dh != 0.0:
            raise ValueError(f"{path}: HGT1 / HGT2 / DHGT {h1}/{h2}/{dh} describe several layers; "
                             "only 2-D (single-layer) IONEX maps are supported")
        lat1, lat2, dlat = get("LAT1 / LAT2 / DLAT", lambda b: _fixed_floats(b, 3))
        lon1, lon2, dlon = get("LON1 / LON2 / DLON", lambda b: _fixed_floats(b, 3))
        try:
            lats = _grid_axis(lat1, lat2, dlat, "LAT1 / LAT2 / DLAT")
            lons = _grid_axis(lon1, lon2, dlon, "LON1 / LON2 / DLON")
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None
        if lats.size < 2 or lons.size < 2:
            raise ValueError(f"{path}: the map grid needs at least two latitude rows and two longitude columns "
                             f"(LAT {lat1}/{lat2}/{dlat}, LON {lon1}/{lon2}/{dlon})")
        return _IonexHeader(
            end=end,
            interval_s=interval,
            n_maps=get("# OF MAPS IN FILE", lambda b: int(b.split()[0]), required=False),
            exponent=get("EXPONENT", lambda b: int(b.split()[0]), required=False, default=IONEX_DEFAULT_EXPONENT),
            height_km=h1,
            base_radius_km=get("BASE RADIUS", lambda b: float(b.split()[0]), required=False,
                               default=DEFAULT_BASE_RADIUS_KM),
            lats=lats,
            lons=lons,
            mapping_function=get("MAPPING FUNCTION", lambda b: b.split()[0].upper() if b.split() else None,
                                 required=False),
            description=" ".join(" ".join(description).split()),
        )

    @staticmethod
    def _parse_interval(path: Path) -> float:
        """INTERVAL of the IONEX header in hours (0 = irregular map spacing)."""

        lines = _read_ionex_lines(path)
        return GIMReader._parse_header(lines, Path(path)).interval_s / 3600

    @classmethod
    def _parse_dcb(cls, path: Path) -> pd.DataFrame:
        """Parse the DCB records of the IONEX AUX block of ``path``.

                Status
                ------
                Active and requires caution. The parsed bias table is consumed
                by STEC bias handling for GPS, Galileo and BeiDou. Format
                extensions should be covered by product-specific tests.
                """

        return cls._parse_dcb_lines(_read_ionex_lines(path), Path(path))

    @staticmethod
    def _parse_dcb_lines(lines: list[str], path: Path) -> pd.DataFrame:
        """Parse the first ``DIFFERENTIAL CODE BIASES`` AUX block of the header.

                Records (by label): ``PRN / BIAS / RMS`` (e.g. ``G01 -7.047 0.051``)
                and ``STATION / BIAS / RMS`` (system, 4-char site, optional DOMES
                number, bias, rms). A blank system flag (IONEX 1.0 style,
                ``   01 -7.047 0.051``) means GPS. GPS, Galileo and BeiDou records
                are kept; other valid systems are skipped, unknown system tokens
                give one RuntimeWarning per file. The CAS product has further AUX
                blocks (P1-C1, P2-C2); only the first DCB block (P1-P2) is read.

                Returns
                -------
                pd.DataFrame
                    Columns entry_type, system, prn_or_site, bias, rms, sv; an
                    empty frame with these columns when the file has no DCB block.

                Raises
                ------
                ValueError
                    For a malformed record (message names file, line and record).
                """

        entries: list[_DCBEntry] = []
        skipped: list[str] = []
        in_aux = False
        for i, ln in enumerate(lines):
            lab = _label(ln)
            if lab == "END OF HEADER":
                break
            if lab == "START OF AUX DATA":
                in_aux = "DIFFERENTIAL CODE BIASES" in ln[:60].upper()
                continue
            if lab == "END OF AUX DATA":
                if in_aux:
                    break
                continue
            if not in_aux or lab not in ("PRN / BIAS / RMS", "STATION / BIAS / RMS"):
                continue
            p = ln[:60].split()
            where = f"{path}, line {i + 1}: {lab} record {ln[:60].strip()!r}"
            # IONEX 1.0 (3X,A1,...): a blank satellite-system flag in column 4 means GPS
            blank_sys = ln[:4].strip() == "" and len(p) > 0
            if lab == "PRN / BIAS / RMS":
                if blank_sys and p[0].isdigit() and len(p[0]) <= 2:
                    p[0] = f"G{int(p[0]):02d}"
                if len(p) < 3 or not _SAT_TOKEN.match(p[0]) or p[0][0] not in _GNSS_LETTERS:
                    raise ValueError(f"{where}: expected '<sys><prn> <bias> <rms>' (e.g. 'G01 -7.047 0.051')")
                system, values, site = p[0][0], p[1:3], p[0][1:]
            else:
                if not p:
                    raise ValueError(f"{where}: empty record")
                if blank_sys and not (len(p[0]) == 1 and p[0] in _GNSS_LETTERS):
                    p.insert(0, "G")
                if len(p[0]) != 1 or p[0] not in _GNSS_LETTERS:
                    # e.g. 'GE': not a system identifier, the record cannot be attributed
                    skipped.append(f"line {i + 1} ({p[0]!r})")
                    continue
                if len(p) not in (4, 5):
                    raise ValueError(f"{where}: expected '<sys> <site> [<DOMES>] <bias> <rms>'")
                system, values, site = p[0], p[-2:], "".join(p[1:-2])
            try:
                bias, rms = float(values[0]), float(values[1])
            except ValueError:
                raise ValueError(f"{where}: bias/rms are not numbers") from None
            if system not in _DCB_SYSTEMS:
                continue
            if lab == "PRN / BIAS / RMS":
                entries.append(_DCBEntry(system, site, bias, rms, "satellite", p[0]))
            else:
                entries.append(_DCBEntry(system, site, bias, rms, "station", None))
        if skipped:
            warnings.warn(f"{path}: {len(skipped)} STATION / BIAS / RMS record(s) with an unknown system token "
                          f"skipped: {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}",
                          RuntimeWarning, stacklevel=3)
        if not entries:
            return pd.DataFrame({
                "entry_type": pd.Series(dtype=object), "system": pd.Series(dtype=object),
                "prn_or_site": pd.Series(dtype=object), "bias": pd.Series(dtype=float),
                "rms": pd.Series(dtype=float), "sv": pd.Series(dtype=object),
            })
        df = pd.DataFrame(e.__dict__ for e in entries)
        return df[_DCB_COLUMNS]

    @classmethod
    def read_dcb_reference(cls, path: str | Path) -> DCBReference:
        """Code pairs the DCBs of the IONEX file ``path`` refer to (header only, cached per file version)."""
        path = Path(path)
        return _cached_dcb_reference(str(path.resolve()), path.stat().st_mtime_ns)

    @staticmethod
    def _parse_dcb_reference(lines: list[str]) -> DCBReference:
        """Read the 'Reference observables for ...' COMMENT records of the first DCB AUX block.

        A system statement ('... for GALILEO: C1X-C5X') applies to the
        satellite DCBs of that system, a station statement ('... for
        ZIM3 14001M008 (G): C1C-C2W') to that receiver DCB only. Files
        without statements (IGS, CAS) give empty maps, i.e. the IONEX
        defaults of ``IONEX_DEFAULT_DCB_REFERENCE``.
        """

        satellite: Dict[str, Tuple[str, str]] = {}
        station: Dict[Tuple[str, str], Tuple[str, str]] = {}
        in_aux = False
        for ln in lines:
            lab = _label(ln)
            if lab == "END OF HEADER":
                break
            if lab == "START OF AUX DATA":
                in_aux = "DIFFERENTIAL CODE BIASES" in ln[:60].upper()
                continue
            if lab == "END OF AUX DATA":
                if in_aux:
                    break
                continue
            if not in_aux or lab != "COMMENT":
                continue
            m = _DCB_REFERENCE.match(ln[:60].strip())
            if m is None:
                continue
            who, pair = m.group(1).strip(), (m.group(2).upper(), m.group(3).upper())
            sys_ = _SYSTEM_BY_NAME.get(who.upper())
            if sys_ is not None:
                satellite[sys_] = pair
                continue
            st = _DCB_REFERENCE_STATION.match(who)
            if st is not None:
                station[(st.group(1).replace(" ", ""), st.group(2).upper())] = pair
        return DCBReference(satellite=satellite, station=station)

    # -------------------------------------------------- TEC / RMS MAPS

    def _parse_maps(self, lines: list[str], hdr: _IonexHeader, path: Path):
        """Read TEC and optional RMS maps after the header.

                Returns
                -------
                tuple
                    xarray.Dataset with TEC maps, xarray.Dataset with RMS maps
                    (None if the file has none).

                Raises
                ------
                ValueError
                    If a map is incomplete or has no END record (truncated
                    file), a record does not match the header grid, or the
                    file holds no TEC map.
                """

        lats, lons = hdr.lats, hdr.lons
        nlat, nlon = lats.size, lons.size
        maps: Dict[str, Dict[datetime, np.ndarray]] = {"TEC": {}, "RMS": {}}
        fact = 10.0 ** hdr.exponent
        sec, epoch, rows, filled, start = None, None, None, None, None
        i, n = hdr.end + 1, len(lines)
        while i < n:
            ln = lines[i]
            lab = _label(ln)
            if lab in ("START OF TEC MAP", "START OF RMS MAP", "START OF HEIGHT MAP"):
                if sec is not None:
                    # a lost END record would silently drop a map (and widen the time interpolation)
                    raise ValueError(f"{path}, line {i + 1}: {lab} while the {sec} map starting at line "
                                     f"{start + 1} has no END OF {sec} MAP")
                sec, epoch, start = lab.split()[2], None, i
                rows, filled = np.full((nlat, nlon), np.nan), np.zeros(nlat, dtype=bool)
                i += 1
                continue
            if lab in ("END OF TEC MAP", "END OF RMS MAP", "END OF HEIGHT MAP"):
                kind = lab.split()[2]
                if sec != kind:
                    raise ValueError(f"{path}, line {i + 1}: {lab} without START OF {kind} MAP")
                if kind != "HEIGHT":
                    what = f"{path}, {kind} map at line {start + 1}"
                    if epoch is None:
                        raise ValueError(f"{what}: no EPOCH OF CURRENT MAP record")
                    if not filled.all():
                        raise ValueError(f"{what} ({epoch}): incomplete map, {int(filled.sum())} of {nlat} "
                                         f"latitude rows of the LAT1 / LAT2 / DLAT grid")
                    if epoch in maps[kind]:
                        raise ValueError(f"{what}: second {kind} map for epoch {epoch}")
                    maps[kind][epoch] = rows
                sec = None
                i += 1
                continue
            if lab == "END OF FILE":
                break
            if sec is None or sec == "HEIGHT":
                i += 1          # COMMENT records between maps, height maps
                continue
            if lab == "EPOCH OF CURRENT MAP":
                try:
                    epoch = _ionex_epoch(ln[:60])
                except ValueError as exc:
                    raise ValueError(f"{path}, line {i + 1}: invalid EPOCH OF CURRENT MAP ({exc})") from None
            elif lab == "EXPONENT":
                # applies from this map on (as RTKLIB readionexb)
                try:
                    fact = 10.0 ** int(ln[:60].split()[0])
                except (ValueError, IndexError):
                    raise ValueError(f"{path}, line {i + 1}: invalid EXPONENT record {ln.strip()!r}") from None
            elif lab == "LAT/LON1/LON2/DLON/H":
                rec_line = i
                try:
                    lat, lon1, lon2, dlon, hgt = _fixed_floats(ln[:60], 5)
                except ValueError as exc:
                    raise ValueError(f"{path}, line {i + 1}: invalid LAT/LON1/LON2/DLON/H record ({exc})") from None
                k = np.flatnonzero(np.abs(lats - lat) < 1e-6)
                if (k.size != 1 or abs(lon1 - lons[0]) > 1e-6 or abs(lon2 - lons[-1]) > 1e-6
                        or abs(dlon - (lons[1] - lons[0])) > 1e-6 or abs(hgt - hdr.height_km) > 1e-6):
                    raise ValueError(
                        f"{path}, line {i + 1}: {sec} map record LAT {lat} LON {lon1}/{lon2}/{dlon} H {hgt} does "
                        f"not match the header grid (LAT {lats[0]}/{lats[-1]}/{lats[1] - lats[0]}, "
                        f"LON {lons[0]}/{lons[-1]}/{lons[1] - lons[0]}, HGT1 {hdr.height_km})")
                k = int(k[0])
                if filled[k]:
                    raise ValueError(f"{path}, line {i + 1}: latitude {lat} given twice in one {sec} map")
                vals: list[int] = []
                i += 1
                while len(vals) < nlon:
                    # data lines (16I5) fill columns 61-80 too: a record is a line with a text label
                    if i >= n or any(c.isalpha() for c in lines[i][60:80]):
                        raise ValueError(f"{path}, line {rec_line + 1}: {sec} map row LAT {lat} has "
                                         f"{len(vals)} of {nlon} values (LON1 / LON2 / DLON grid)")
                    try:
                        vals.extend(_i5_values(lines[i]))
                    except ValueError:
                        raise ValueError(f"{path}, line {i + 1}: invalid {sec} map values {lines[i].strip()!r}") from None
                    i += 1
                if len(vals) != nlon:
                    raise ValueError(f"{path}, line {rec_line + 1}: {sec} map row LAT {lat} has {len(vals)} values, "
                                     f"the LON1 / LON2 / DLON grid has {nlon}")
                row = np.asarray(vals, dtype=float)
                row[row == IONEX_NO_VALUE] = np.nan      # 9999: no value available
                rows[k] = row * fact
                filled[k] = True
                continue
            i += 1

        if sec is not None:
            raise ValueError(f"{path}: the {sec} map starting at line {start + 1} has no END OF {sec} MAP "
                             f"(truncated file?)")
        if not maps["TEC"]:
            raise ValueError(f"{path}: no TEC map found")
        if hdr.n_maps is not None and hdr.n_maps != len(maps["TEC"]):
            warnings.warn(f"{path}: # OF MAPS IN FILE is {hdr.n_maps}, {len(maps['TEC'])} TEC maps read",
                          RuntimeWarning, stacklevel=3)
        if maps["RMS"] and set(maps["RMS"]) != set(maps["TEC"]):
            # RMS is interpolated on its own epochs: a missing RMS map is bridged silently
            only_rms = [str(t) for t in sorted(set(maps["RMS"]) - set(maps["TEC"]))]
            only_tec = [str(t) for t in sorted(set(maps["TEC"]) - set(maps["RMS"]))]
            warnings.warn(f"{path}: {len(maps['RMS'])} RMS maps for {len(maps['TEC'])} TEC maps; TEC epochs "
                          f"without an RMS map: {only_tec[:3]}{' ...' if len(only_tec) > 3 else ''}, RMS epochs "
                          f"without a TEC map: {only_rms[:3]}{' ...' if len(only_rms) > 3 else ''}",
                          RuntimeWarning, stacklevel=3)

        def build(grid_dict: Dict[datetime, np.ndarray], varname: str):
            if not grid_dict:
                return None
            times = sorted(grid_dict)
            data = np.stack([grid_dict[t] for t in times], axis=0)
            da = xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords=dict(time=np.array(times, dtype="datetime64[ns]"), lat=lats, lon=lons),
                name=varname,
            )
            da.attrs["units"] = "TECU rms" if varname == "RMS" else "TECU"
            ds = da.to_dataset()
            ds.attrs.update(height_km=hdr.height_km, base_radius_km=hdr.base_radius_km,
                            interval_s=hdr.interval_s, source=str(path))
            return ds

        return build(maps["TEC"], "TEC"), build(maps["RMS"], "RMS")


@functools.lru_cache(maxsize=16)
def _cached_dcb_reference(path: str, mtime_ns: int) -> DCBReference:
    """DCB reference observables of one IONEX file version (``mtime_ns`` keys the cache)."""
    return GIMReader._parse_dcb_reference(_read_ionex_lines(path))
