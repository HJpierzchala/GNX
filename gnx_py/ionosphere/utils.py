"""Ionospheric models and utilities for TEC estimation and related GNSS workflows.

This module aggregates model implementations and helpers used across
ionosphere-related processing, including thin-shell assumptions, mapping
functions, and empirical/analytic formulations employed in TEC computation.
The focus is on clarity and composability so that higher-level pipelines
can select and compare different modeling approaches without modifying logic.

Notes:
- All public functions/classes should document units explicitly (e.g., meters,
  kilometers, degrees, radians) and the expected array shapes or broadcasting.
- Where multiple model variants exist, prefer a single entry point with a
  parameter switch while keeping mathematical equivalence clear in the docs.
"""

import numpy as np
C = 299792458
PI = 3.1415296535898

def get_ipp(ev, az, lat, lon, ish=450.0, R=6371.0):
    """Compute ionospheric pierce point (IPP) geographic coordinates.

        Computes the latitude and longitude of the ionospheric pierce point (IPP)
        at a specified thin-shell ionosphere height using the receiver geodetic
        position and line-of-sight defined by elevation and azimuth.

        Args:
            ev (array-like of float): Elevation angles in degrees.
            az (array-like of float): Azimuth angles in degrees.
            lat (array-like of float): Receiver geodetic latitude in degrees.
            lon (array-like of float): Receiver geodetic longitude in degrees.
            ish (float, optional): Ionospheric shell height in kilometers.
                Defaults to 450.0.
            R (float, optional): Mean Earth radius in kilometers. Defaults to 6371.0.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple (lat_ipp_deg, lon_ipp_deg) where:
                - lat_ipp_deg is IPP latitude in degrees, clipped to [-90, 90].
                - lon_ipp_deg is IPP longitude in degrees, normalized to [-180, 180].

        Notes:
            - Inputs are broadcast using NumPy semantics; the result shapes follow
              NumPy broadcasting rules based on the inputs.
            - Elevation and azimuth are assumed to follow the local topocentric
              frame at the receiver location.
        """

    ev = np.array(ev, dtype=float)  # Upewnij się, że `ev` jest tablicą typu float
    az = np.array(az, dtype=float)  # Upewnij się, że `az` jest tablicą typu float
    ev = np.deg2rad(ev)
    az = np.deg2rad(az)

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    wi_pp = (PI / 2) - ev - np.arcsin((R / (R + ish)) * np.cos(ev))
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


def get_local_time(la_pp, ut):
    """Compute local time (solar) from longitude and UTC time.

        Converts a UTC timestamp to local solar time at the specified longitude.

        Args:
            la_pp (float or array-like of float): Longitude in degrees (east positive).
            ut: UTC timestamp-like object exposing hour, minute, and second attributes
                (e.g., datetime.datetime in UTC).

        Returns:
            float or np.ndarray: Local time in decimal hours. The value is not wrapped
            to [0, 24); apply modulo 24 externally if needed.

        Notes:
            - The conversion uses 15 degrees per hour (360°/24 h).
            - If an array of longitudes is provided, NumPy broadcasting applies.
        """

    # Convert longitude from radians to degrees and compute local time
    uth = ut.hour + ut.minute / 60 + ut.second / 3600
    lt = uth + la_pp / 15
    return lt


def stec_mf(ev, ish=450e03, R=6371e03, no=1):
    """Compute slant TEC mapping function (thin-shell ionosphere).

    Implements several mapping function (MF) formulations to convert between
    vertical and slant TEC under a single-layer ionosphere assumption.

    Args:
        ev (array-like of float): Elevation angles in degrees.
        ish (float, optional): Ionospheric shell height in meters. Defaults to 450e03.
        R (float, optional): Mean Earth radius in meters. Defaults to 6371e03.
        no (int, optional): Mapping function variant selector. Must be one of:
            1: Empirical variant using sin(z') with scale 0.9782.
            2: Standard thin-shell geometry using zenith angle at IPP.
            3: Alternative form based on cosine of elevation.
            4: Secant of zenith angle at receiver approximation (1 / cos(z)).

    Returns:
        np.ndarray: Mapping function values (dimensionless), broadcast to the shape
        of the input elevation array.

    Raises:
        AssertionError: If `no` is not in {1, 2, 3, 4}.

    Notes:
        - The mapping function (MF) is typically multiplied by vertical TEC (VTEC)
          to obtain slant TEC (STEC), i.e., STEC ≈ MF · VTEC.
        - Units must be consistent: this function expects `ish` and `R` in meters.
        - Input elevation is converted to radians internally.
    """

    ev = np.array(ev, dtype=float)
    ev = np.deg2rad(ev)
    assert no in [1, 2, 3, 4]
    if no == 1:
        sinz = (R / (R + ish)) * np.sin(0.9782 * ((PI / 2) - ev))
        mf = 1 / (np.sqrt(1 - (sinz ** 2)))
    elif no == 2:
        sinzip = R / (R + ish) * np.sin(np.pi / 2 - ev)
        zips = np.arcsin(sinzip)
        mf = 1 / np.cos(zips)
    elif no == 3:
        mf = (1 - ((R * np.cos(ev) / (R + ish)) ** 2)) ** (-0.5)
    elif no == 4:
        mf = 1 / np.cos(np.pi / 2 - ev)

    return mf
