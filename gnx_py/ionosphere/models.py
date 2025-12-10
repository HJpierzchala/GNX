"""Ionopsheric modeling utilities.

This module provides implementations of classic and empirical ionospheric models
and helpers used in GNSS processing, including:
- Klobuchar model (broadcast ionospheric delay model for GPS)
- Components of NTCM-like VTEC modeling
- Helpers for solar geometry and slant-to-vertical mapping

Conventions and units:
- Angles:
  - Latitude/longitude/azimuth/elevation are generally given in degrees unless
    explicitly stated otherwise.
  - Radians are used internally where noted.
- Time:
  - `tow` is GPS time-of-week seconds unless a `datetime` is provided (interpreted
    as a GPST datetime for conversion).
  - Day-of-year (DOY) is counted from 1..365/366.
- Physical quantities:
  - Speed of light constant (C) is in m/s.
  - Klobuchar ionospheric delay returns either meters or seconds depending on the
    specific function (see respective docstrings).
- Arrays:
  - Many functions accept either scalars or NumPy arrays; broadcasting follows
    NumPy semantics.

References:
- GPS ToolBox Klobuchar FORTRAN reference:
  https://geodesy.noaa.gov/gps-toolbox/ovstedal/klobuchar.for

Note:
This module does not alter external I/O or data sources; it focuses on numerical
transformations only.
"""

import math
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from ..conversion import geographic2geomagnetic_latitude,spherical2ecef,ecef2geodetic
from .utils import get_ipp, get_local_time, stec_mf
from ..time import gpst2utc, local2gsow

C = 299792458
PI = 3.1415296535898

def klobuchar(azimuth, elev, fi, lambda_, tow, beta, alfa):
    """
        Compute ionospheric delay using the classic Klobuchar broadcast model.

        This implementation follows the NOAA GPS Toolbox reference implementation.
        The returned delay is a slant ionospheric group delay in meters for a given
        satellite-receiver geometry and broadcast coefficients.

        Parameters:
            azimuth (float): Satellite azimuth in degrees.
            elev (float): Satellite elevation angle in degrees.
            fi (float): Receiver geodetic latitude in degrees.
            lambda_ (float): Receiver geodetic longitude in degrees.
            tow (float): GPS time-of-week in seconds (seconds of day are derived internally).
            beta (array-like of float, length 4): Klobuchar period (GPSB) coefficients.
            alfa (array-like of float, length 4): Klobuchar amplitude (GPSA) coefficients.

        Returns:
            float: Ionospheric slant delay in meters.

        Notes:
            - Elevation is converted to semicircles internally as per the model definition.
            - The mapping (slant factor) is applied within the model.
            - If the model phase |x| > 1.57, a default minimal delay is applied per spec.
        """

    pi = math.pi
    deg2semi = 1.0 / 180.0
    semi2rad = pi
    deg2rad = pi / 180.0
    c = 2.99792458e8  # speed of light

    a = azimuth * deg2rad
    e = elev * deg2semi

    psi = 0.0137 / (e + 0.11) - 0.022

    lat_i = fi * deg2semi + psi * math.cos(a)
    if lat_i > 0.416:
        lat_i = 0.416
    elif lat_i < -0.416:
        lat_i = -0.416

    long_i = lambda_ * deg2semi + (psi * math.sin(a) / math.cos(lat_i * semi2rad))

    lat_m = lat_i + 0.064 * math.cos((long_i - 1.617) * semi2rad)

    t = 4.32e4 * long_i + tow
    t = t % 86400.0  # Seconds of day
    if t > 86400.0:
        t -= 86400.0
    elif t < 0.0:
        t += 86400.0

    sF = 1.0 + 16.0 * (0.53 - e) ** 3  # Slant factor

    PER = beta[0] + beta[1] * lat_m + beta[2] * lat_m ** 2 + beta[3] * lat_m ** 3
    if PER < 72000.0:
        PER = 72000.0

    x = 2.0 * pi * (t - 50400.0) / PER  # Phase of the model (Max at 14.00 = 50400 sec local time)

    AMP = alfa[0] + alfa[1] * lat_m + alfa[2] * lat_m ** 2 + alfa[3] * lat_m ** 3
    if AMP < 0.0:
        AMP = 0.0

    if abs(x) > 1.57:
        dIon1 = sF * 5e-9
    else:
        dIon1 = sF * (5e-9 + AMP * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0))

    dIon1 = c * dIon1
    return dIon1


def sun_dec(doy):
    """
        Compute the Sun's declination angle (radians) for a given day-of-year.

        Parameters:
            doy (int or float or array-like): Day-of-year (1..365/366). Accepts scalars
                or NumPy arrays.

        Returns:
            numpy.ndarray or float: Sun declination angle in radians (broadcasting follows
            NumPy rules).

        Notes:
            - Uses a common approximation: δ ≈ 23.44° · sin(0.9856° · (doy − 80.7)).
            - Output is converted to radians.
        """

    # doy int
    sig = 23.44 * np.sin(0.9856 * (doy - 80.7) * (PI / 180)) * (PI / 180)
    return sig  # sun declination in radians


def sol_zen_deps(fi_pp, sun_declination):
    """
       Compute solar-zenith dependent coefficients used by VTEC modeling.

       Parameters:
           fi_pp (array-like or float): Pierce point geodetic latitude in degrees.
           sun_declination (array-like or float): Sun declination angle in radians.

       Returns:
           tuple[numpy.ndarray or float, numpy.ndarray or float]:
               - cos3x: Auxiliary cosine term for diurnal behavior.
               - cos2x: Auxiliary cosine term with declination dependence.

       Notes:
           - Both inputs can be arrays; outputs broadcast accordingly.
       """

    # radian, radian
    fi_pp = np.deg2rad(fi_pp)
    cos3x = np.cos(fi_pp - sun_declination) + 0.4
    cos2x = np.cos(fi_pp - sun_declination) - (2 / PI) * fi_pp * np.sin(sun_declination)
    return cos3x, cos2x


def klobuchar_delay(fi, lambda_, elev, azimuth, tow, alfa, beta):
    """
        Calculate ionospheric delay using the Klobuchar model (seconds).

        This variant returns the ionospheric correction as a time delay (seconds),
        which is often used before converting to meters via c · τ.

        Parameters:
            fi (float): Receiver geodetic latitude in degrees.
            lambda_ (float): Receiver geodetic longitude in degrees.
            elev (float): Satellite elevation angle in degrees.
            azimuth (float): Satellite azimuth in degrees.
            tow (float or datetime.datetime): GPS time-of-week in seconds, or GPST datetime
                (which will be converted internally).
            alfa (array-like of float, length 4): Amplitude (GPSA) coefficients.
            beta (array-like of float, length 4): Period (GPSB) coefficients.

        Returns:
            float: Ionospheric slant range correction in seconds.

        Notes:
            - Follows the broadcast Klobuchar approach with sub-ionospheric point,
              geomagnetic latitude, and local time computations.
            - If a datetime is provided for `tow`, it is interpreted as GPST.
        """

    pi = math.pi
    c = 2.99792458e8  # speed of light in m/s
    deg2semi = 1.0 / 180.0  # degrees to semicircles
    semi2rad = pi  # semicircles to radians
    deg2rad = pi / 180.0  # degrees to radians

    a = np.deg2rad(azimuth)  # azimuth in radians
    e = elev * deg2semi  # elevation angle in semicircles

    psi = 0.0137 / (e + 0.11) - 0.022  # Earth Centered angle

    lat_i = fi * deg2semi + psi * np.cos(a)  # Subionospheric latitude
    lat_i = np.clip(lat_i, -0.416, 0.416)  # Clipping to range [-0.416, 0.416]

    long_i = lambda_ * deg2semi + (psi * np.sin(a) / np.cos(lat_i * semi2rad))  # Subionospheric longitude

    lat_m = lat_i + 0.064 * np.cos((long_i - 1.617) * semi2rad)  # Geomagnetic latitude

    if isinstance(tow, datetime):
        gps_epoch = datetime(1980, 1, 6)
        delta = tow - gps_epoch
        tow = delta.total_seconds() % 604800  # convert to seconds of GPS week

    t = 4.32e4 * long_i + tow
    t = t % 86400.0  # Seconds of day

    # sf = 1.0 + 16.0 * (0.53 - e) ** 3  # Slant factor

    PER = beta[0] + beta[1] * lat_m + beta[2] * lat_m ** 2 + beta[3] * lat_m ** 3  # Period of model
    PER = max(PER, 72000.0)

    x = 2.0 * pi * (t - 50400.0) / PER  # Phase of the model

    AMP = alfa[0] + alfa[1] * lat_m + alfa[2] * lat_m ** 2 + alfa[3] * lat_m ** 3  # Amplitude of the model
    AMP = max(AMP, 0.0)

    # Ionospheric correction
    if abs(x) > 1.57:
        dIon1 = (5e-9)
    else:
        dIon1 = (5e-9 + AMP * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0))

    return dIon1  # Return ionospheric delay in seconds


def get_vtec(LT, doy, fmag, fmag_deg, klobpar, cos3x, cos2x, sys='GPS'):
    """
        Compute VTEC (Vertical Total Electron Content) using an NTCM model.

        Parameters:
            LT (array-like or float): Local time(s) at the ionospheric pierce points (hours).
            doy (int or float or array-like): Day-of-year.
            fmag (array-like or float): Geomagnetic latitude in radians.
            fmag_deg (array-like or float): Geomagnetic latitude in degrees.
            klobpar (float or array-like): Model driving parameter (derived from broadcast
                coefficients or system-specific parameters).
            cos3x (array-like or float): Solar-zenith dependent coefficient (from sol_zen_deps).
            cos2x (array-like or float): Solar-zenith dependent coefficient (from sol_zen_deps).
            sys (str, optional): GNSS system identifier; affects empirical coefficients.
                Supported values: 'GPS' (default), 'GAL'.

        Returns:
            numpy.ndarray or float: Estimated VTEC in TECU.

        Notes:
            - The model is parameterized with system-dependent constants (k).
            - Inputs may be arrays; broadcasting follows NumPy semantics.
        """

    if sys == 'GPS':
        k = (
        0.87909, 0.17466, -0.02500, 0.06058, 0.00714, 0.01992, -0.02634, -0.31836, 0.99221, 1.01612, 2.59418, 0.35127)
    elif sys == 'GAL':
        k = (
        0.92519, 0.16951, 0.00443, 0.06626, 0.00899, 0.21289, -0.15414, -0.38439, 1.14023, 1.20556, 1.41808, 0.13985)
    VD = (2 * PI * (LT - 14.0)) / 24
    VSD = (2 * PI * LT) / 12
    VTD = (2 * PI * LT) / 8
    F1 = cos3x + cos2x * (
                k[0] * np.cos(VD) + k[1] * np.cos(VSD) + k[2] * np.sin(VSD) + k[3] * np.cos(VTD) + k[4] * np.sin(VTD))

    VA = (2 * PI * (doy - 18)) / 365.25
    VSA = (4 * PI) * (doy - 6) / 365.25
    F2 = 1 + k[5] * np.cos(VA) + k[6] * np.cos(VSA)

    F3 = 1 + k[7] * np.cos(fmag)

    EC1 = - ((fmag_deg - 16) ** 2) / (2 * 12 ** 2)
    EC2 = - ((fmag_deg - (-10)) ** 2) / (2 * 13 ** 2)

    F4 = 1 + k[8] * np.exp(EC1) + k[9] * np.exp(EC2)

    F5 = k[10] + k[11] * klobpar
    vtec = F1 * F2 * F3 * F4 * F5

    return vtec


def run_NTCM_for_gps(pos, DOY, ev, az, epoch, gpsa, gpsb):
    """
        Run an NTCM pipeline to estimate STEC for GPS using Klobpar parameter instead of Azpar.

        Parameters:
            pos (tuple[float, float]): Receiver geodetic coordinates (lat_deg, lon_deg).
            DOY (int): Day-of-year.
            ev (float or array-like): Satellite elevation angle(s) in degrees.
            az (float or array-like): Satellite azimuth angle(s) in degrees.
            epoch (datetime.datetime): GPST epoch of observation.
            gpsa (array-like of float, length 4): Broadcast GPSA (amplitude) coefficients.
            gpsb (array-like of float, length 4): Broadcast GPSB (period) coefficients.

        Returns:
            numpy.ndarray or float: Estimated STEC (slant TEC) in TECU.

        Workflow:
            1) Compute IPP (pierce points) and local solar time.
            2) Derive solar-zenith dependent coefficients and geomagnetic latitude.
            3) Form a driving parameter from broadcast Klobuchar delays at two reference
               locations and local time 14:00:18 via local2gsow.
            4) Evaluate VTEC via get_vtec and map to STEC using stec_mf.

        Notes:
            - The mapping function `stec_mf` is applied to convert VTEC to STEC.
            - Broadcasting is supported for `ev`/`az`.
        """

    fi_pp, la_pp = get_ipp(ev=ev, az=az, lat=pos[0], lon=pos[1])
    UT = gpst2utc(epoch)  # czas UTC
    local_times_list = get_local_time(la_pp, ut=UT)
    sun_declination = sun_dec(DOY)
    cos3x, cos2x = sol_zen_deps(fi_pp, sun_declination)
    fmag, fmag_deg = geographic2geomagnetic_latitude(fi_pp=fi_pp, la_pp=la_pp)

    A = (10, -90)
    B = (-10, -90)

    klobpar_time = local2gsow(datetime(epoch.year, epoch.month, epoch.day, 14, 0, 18))
    dA = klobuchar_delay(fi=A[0], lambda_=A[1], elev=90.0, azimuth=0.0, tow=klobpar_time, alfa=gpsa, beta=gpsb)
    dB = klobuchar_delay(fi=B[0], lambda_=B[1], elev=90.0, azimuth=0.0, tow=klobpar_time, alfa=gpsa, beta=gpsb)

    klobpar = (dA + dB) * C * 6.1587

    vtec = get_vtec(LT=local_times_list, doy=DOY, fmag=fmag, fmag_deg=fmag_deg, klobpar=klobpar, cos3x=cos3x,
                    cos2x=cos2x)
    mf = stec_mf(ev)
    stec = mf * vtec
    return stec


def ntcm_vtec(pos, DOY, ev, az, epoch, gala, mf_no=1,return_vtec=False):
    """
    Run an NTCM pipeline to estimate STEC for Galileo.

    Parameters:
        pos (tuple[float, float]): Receiver geodetic coordinates (lat_deg, lon_deg).
        DOY (int): Day-of-year.
        ev (float or array-like): Satellite elevation angle(s) in degrees.
        az (float or array-like): Satellite azimuth angle(s) in degrees.
        epoch (datetime.datetime): GPST epoch of observation.
        gala (array-like): Galileo system parameters (e.g., two/three coefficients).
        mf_no (int, optional): Mapping function selector passed to stec_mf (default: 1).

    Returns:
        numpy.ndarray or float: Estimated STEC (slant TEC) in TECU.

    Notes:
        - An azimuth-dependent driving parameter `azpar` is formed per provided `gala`.
        - Uses `get_vtec` with sys='GAL' and maps VTEC to STEC via `stec_mf`.
    """

    fi_pp, la_pp = get_ipp(ev=ev, az=az, lat=pos[0], lon=pos[1])
    UT = gpst2utc(epoch)  # czas UTC
    local_times_list = get_local_time(la_pp, ut=UT)
    sun_declination = sun_dec(DOY)
    cos3x, cos2x = sol_zen_deps(fi_pp, sun_declination)
    fmag, fmag_deg = geographic2geomagnetic_latitude(fi_pp=fi_pp, la_pp=la_pp)
    azpar = np.abs(
        np.sqrt(gala[0] ** 2 + 1633.33 * gala[1] ** 2 + 4802000 * gala[2] ** 2 + 3266.67 * gala[0] * gala[2]))

    vtec = get_vtec(LT=local_times_list, doy=DOY, fmag=fmag, fmag_deg=fmag_deg, klobpar=azpar, cos3x=cos3x,
                    cos2x=cos2x, sys='GAL')
    mf = stec_mf(ev, no=mf_no)
    stec = mf * vtec
    if return_vtec:
        return vtec
    else:
        return stec





def compute_ntcm_grid(
    lat_grid,
    lon_grid,
    times,
    gal_alpha,
    shell_height: float = 450e3,
    earth_radius: float = 6371e3,
    doy: int | None = None,
) -> xr.Dataset:
    """
    Oblicz VTEC z modelu NTCM na siatce (lat, lon, time) i zwróć xarray.Dataset.

    Parameters
    ----------
    lat_grid : array-like
        Wektor szerokości geograficznych (deg).
    lon_grid : array-like
        Wektor długości geograficznych (deg).
    times : array-like
        Wektor czasów (np. np.datetime64, list[Timestamp], DatetimeIndex).
    gal_alpha : any
        Parametr/parametry gal_alpha przekazywane do gnx.ntcm_vtec (argument 'gala').
    shell_height : float, optional
        Wysokość powłoki jonosferycznej nad powierzchnią Ziemi [m], domyślnie 450e3.
    earth_radius : float, optional
        Promień Ziemi [m], domyślnie 6371e3.
    doy : int or None, optional
        Day-of-year przekazywany do gnx.ntcm_vtec (argument DOY).
        Jeśli None, zostanie wyliczony z pierwszego elementu `times`.

    Returns
    -------
    xr.Dataset
        Dataset z jedną zmienną 'V' o wymiarach (time, lat, lon),
        z atrybutami jednostek i opisu.
    """

    # Upewnij się, że mamy poprawne wektory i typy
    lat_grid = np.asarray(lat_grid, dtype=float)
    lon_grid = np.asarray(lon_grid, dtype=float)
    times = pd.to_datetime(times)

    # Jeżeli DOY nie podany, wyznacz z pierwszej epoki
    if doy is None:
        doy = times[0].dayofyear

    # Przygotuj pustą tablicę na wyniki: (czas, lat, lon)
    V = np.empty((times.size, lat_grid.size, lon_grid.size), dtype=float)

    # Pętla po siatce i czasie
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            # pozycja IPP na powłoce h = shell_height
            # wektor sferyczny: [lat, lon, r]
            sph = np.array([lat, lon, earth_radius + shell_height], dtype=float)
            ecef = spherical2ecef(sph=sph)
            geodetic = ecef2geodetic(ecef=ecef, deg=True)  # [lat, lon, h]

            for k, t in enumerate(times):
                t = pd.Timestamp(t)
                vtec = ntcm_vtec(
                    pos=geodetic,
                    DOY=doy,
                    ev=0.0,
                    az=0.0,        # VTEC nad punktem
                    epoch=t,
                    gala=gal_alpha,
                    return_vtec=True,
                )
                V[k, i, j] = vtec

    # Tworzenie DataArray
    da_V = xr.DataArray(
        V,
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": lat_grid,
            "lon": lon_grid,
        },
        name="V",
        attrs={
            "units": "TECU",
            "description": "VTEC from NTCM",
        },
    )

    # Dataset w tej samej postaci, co miałeś
    ds_vtec = xr.Dataset({"V": da_V})

    return ds_vtec
