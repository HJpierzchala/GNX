"""Ionospheric modeling utilities.

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
from ..core.conversion import geographic2geomagnetic_latitude,spherical2ecef,ecef2geodetic
from .utils import get_ipp, get_local_time, stec_mf, _check_choice
from ..core.time import gpst2utc, local2gsow

C = 299792458
PI = np.pi

def klobuchar(azimuth, elev, fi, lambda_, tow, beta, alfa):
    """
        Compute ionospheric delay using the classic Klobuchar broadcast model.

        Status:
            Active broadcast-model helper used by preprocessing and model
            comparison paths. GPS coefficients are required.

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
            float: Ionospheric slant delay in meters on GPS L1; another signal f
                takes ``gnss.gamma('L1', f)`` times this delay (IS-GPS-200 20.3.3.5.2.5).

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


def klobuchar_array(azimuth, elev, fi, lambda_, tow, beta, alfa):
    """Vectorised :func:`klobuchar` (GPS, IS-GPS-200 20.3.3.5.2.5): slant delay [m] on L1.

    Same formulas and clamps as the scalar reference, element-wise over ``azimuth``,
    ``elev`` and ``tow`` (GPS seconds of week or of day).
    """
    a = np.deg2rad(np.asarray(azimuth, dtype=float))
    e = np.asarray(elev, dtype=float) / 180.0
    tow = np.asarray(tow, dtype=float)
    psi = 0.0137 / (e + 0.11) - 0.022
    lat_i = np.clip(fi / 180.0 + psi * np.cos(a), -0.416, 0.416)
    long_i = lambda_ / 180.0 + psi * np.sin(a) / np.cos(lat_i * np.pi)
    lat_m = lat_i + 0.064 * np.cos((long_i - 1.617) * np.pi)
    t = np.mod(4.32e4 * long_i + tow, 86400.0)
    sF = 1.0 + 16.0 * (0.53 - e) ** 3
    per = beta[0] + beta[1] * lat_m + beta[2] * lat_m ** 2 + beta[3] * lat_m ** 3
    per = np.where(per < 72000.0, 72000.0, per)
    x = 2.0 * np.pi * (t - 50400.0) / per
    amp = alfa[0] + alfa[1] * lat_m + alfa[2] * lat_m ** 2 + alfa[3] * lat_m ** 3
    amp = np.where(amp < 0.0, 0.0, amp)
    d = np.where(np.abs(x) > 1.57, sF * 5e-9, sF * (5e-9 + amp * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0)))
    return 2.99792458e8 * d


#: BDS Klobuchar shell (BDS-SIS-ICD-B1I 3.0, 5.2.4.7): Earth radius and ionospheric height [km]
BDS_KLOBUCHAR_R_KM = 6378.0
BDS_KLOBUCHAR_H_KM = 375.0
#: BDT = GPST - 14 s
BDT_MINUS_GPST_S = -14.0


def klobuchar_bds(azimuth, elev, fi, lambda_, sod_bdt, alfa, beta):
    """BeiDou Klobuchar slant ionospheric delay [m] on B1I (BDS-SIS-ICD-B1I 3.0, 5.2.4.7).

    The BDS model differs from the GPS one (IS-GPS-200): a spherical IPP on a
    375 km shell (R = 6378 km) with the geographic IPP latitude, a cosine (not
    its series) for the daytime term, the period A4 clamped to 72000-172800 s,
    and the slant factor 1/sqrt(1 - (R cos E / (R + h))^2). The coefficients
    are the BDSA/BDSB values of the navigation header (RINEX ``BDSA``/``BDSB``),
    in the units of the GPS ones (s/semicircle^n). Vectorised.

    Opt-in in the sessions: ``Config.bds_klobuchar="icd"`` (SPP, PPP, TEC and
    ``compare_models``). The default ``"gps"`` takes the GPS Klobuchar
    coefficients scaled to the BeiDou signal: the BDS-2 broadcast coefficients are fitted for the
    Asia-Pacific service area, and over 9 station-days (BRUX/NKLG/NYA2 x
    2024-035/131/132) the slant RMS against the COD GIM at B1I is 17.95 TECU
    GPS-scaled and 19.58 TECU with this model, worse on 5 of 9 days (1 tie).
    In SPP 29 BeiDou rows got worse and 22 better (NKLG 2024-132 C-only
    rms_3d 3.08 -> 6.75 m). The function itself matches an independent ICD
    implementation to 3.6e-14 m (2000 random cases).
    BDGIM, the global BDS-3 model, is implemented in
    :mod:`gnx_py.ionosphere.bdgim` (BDS-SIS-ICD-B2b-1.0 §7.7;
    ionosphere_model='bdgim').

    Parameters
    ----------
    azimuth, elev : array_like
        Satellite azimuth and elevation [deg].
    fi, lambda_ : float or array_like
        Receiver geographic latitude and longitude [deg].
    sod_bdt : array_like
        BDT seconds of day (or of week; taken modulo 86400). BDT = GPST - 14 s.
    alfa, beta : sequence of 4 floats, or arrays (n, 4)
        BDSA (alpha_n) and BDSB (beta_n) coefficients: one set for every row, or one set per row (the
        broadcast set in force at each epoch, ``rinex_nav.broadcast_iono_at``).

    Returns
    -------
    numpy.ndarray
        Slant group delay [m] at B1I; another signal f takes ``gnss.gamma('B1I', f)``
        times this delay.
    """
    R, h = BDS_KLOBUCHAR_R_KM, BDS_KLOBUCHAR_H_KM
    E = np.deg2rad(np.asarray(elev, dtype=float))
    A = np.deg2rad(np.asarray(azimuth, dtype=float))
    fu = np.deg2rad(np.asarray(fi, dtype=float))
    lu = np.deg2rad(np.asarray(lambda_, dtype=float))
    psi = np.pi / 2 - E - np.arcsin(R / (R + h) * np.cos(E))
    phi_m = np.arcsin(np.sin(fu) * np.cos(psi) + np.cos(fu) * np.sin(psi) * np.cos(A))
    lam_m = lu + np.arcsin(np.sin(psi) * np.sin(A) / np.cos(phi_m))
    x = np.abs(phi_m / np.pi)
    # one set (4,) for all rows, or one set per row (n, 4) (the set in force at each epoch, bug-fix stage 10)
    al = np.asarray(alfa, dtype=float)
    be = np.asarray(beta, dtype=float)
    al = al.T if al.ndim == 2 else al
    be = be.T if be.ndim == 2 else be
    a2 = sum(al[n] * x ** n for n in range(4))
    a2 = np.where(a2 < 0.0, 0.0, a2)
    a4 = sum(be[n] * x ** n for n in range(4))
    a4 = np.clip(a4, 72000.0, 172800.0)
    t = np.mod(np.asarray(sod_bdt, dtype=float) + lam_m * 43200.0 / np.pi, 86400.0)
    iz = np.where(np.abs(t - 50400.0) < a4 / 4.0,
                  5e-9 + a2 * np.cos(2.0 * np.pi * (t - 50400.0) / a4), 5e-9)
    slant = 1.0 / np.sqrt(1.0 - (R * np.cos(E) / (R + h)) ** 2)
    return C * slant * iz


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
                coefficients or sys-specific parameters).
            cos3x (array-like or float): Solar-zenith dependent coefficient (from sol_zen_deps).
            cos2x (array-like or float): Solar-zenith dependent coefficient (from sol_zen_deps).
            sys (str, optional): GNSS sys identifier; affects empirical coefficients.
                Supported values: 'GPS' (default), 'GAL'.

        Returns:
            numpy.ndarray or float: Estimated VTEC in TECU.

        Raises:
            ValueError: If ``sys`` is not 'GPS' or 'GAL'.

        Notes:
            - The model is parameterized with sys-dependent constants (k).
            - Inputs may be arrays; broadcasting follows NumPy semantics.
        """

    _check_choice("sys", sys, ['GPS', 'GAL'], fuzzy=False,
                  extra="(NTCM coefficients are defined for GPS and Galileo only)")
    # coefficients k1..k12 of the NTCM model (Jakowski et al. 2011, J. Geod. 85, 965-974,
    # doi:10.1007/s00190-011-0455-1). 'GAL': Table 3 of the NTCM G Ionospheric Model Description, issue 1.0
    # (European GNSS (Galileo) Open Service, European Union / European GNSS Service Centre, May 2022,
    # https://www.gsc-europa.eu/sites/default/files/NTCM-G_Ionospheric_Model_Description_-_v1.0.pdf), equal digit
    # by digit. 'GPS': the NTCM driven by the GPS Klobuchar coefficients (NTCM-Klobpar; Hoque, Jakowski and
    # Berdermann 2017, GPS Solut. 21, 1563-1572, doi:10.1007/s10291-017-0632-7); this set is not checked against
    # the paper (no open-access copy; the open-access follow-up, Hoque, Jakowski and Berdermann 2018, J. Space
    # Weather Space Clim. 8, A20, doi:10.1051/swsc/2018009, does not list the coefficients).
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

        Status:
            Active model helper but less exercised than the Galileo NTCM path.
            Requires GPSA/GPSB broadcast coefficients and validation for new
            products.

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

    Status:
        Active model helper for Galileo-style NTCM comparisons. Treat as
        model-sensitive: changes require validation against known broadcasts.

    Parameters:
        pos (tuple[float, float]): Receiver geodetic coordinates (lat_deg, lon_deg).
        DOY (int): Day-of-year.
        ev (float or array-like): Satellite elevation angle(s) in degrees.
        az (float or array-like): Satellite azimuth angle(s) in degrees.
        epoch (datetime.datetime): GPST epoch of observation.
        gala (array-like): Galileo sys parameters (e.g., two/three coefficients).
        mf_no (int, optional): Mapping function selector passed to stec_mf (default: 1).

    Returns:
        numpy.ndarray or float: Estimated STEC (slant TEC) in TECU.

    Notes:
        - An azimuth-dependent driving parameter `azpar` is formed per provided `gala`.
        - Uses `get_vtec` with sys='GAL' and maps VTEC to STEC via `stec_mf`.
        - Verification: the 38 rows of Annex D of the NTCM-G model description (European GNSS (Galileo) Open
          Service, NTCM-G Ionospheric Model Description, issue 1.0, 2022) to 4.9e-5 TECU with ``mf_no=1``, and
          an independent implementation (ray/sphere pierce point, pole vector) to 1e-12 TECU on random
          geometry; the SPP session's 'ion' column of an NTCM row equals it.
        - The pierce-point longitude is the description's lambda + asin(...) (``utils.get_ipp``): a pierce point
          beyond the pole is folded onto the near meridian, as Annex D row 6 (82.49 N looking poleward) is
          computed. At NYA2 (78.9 N) that concerns 1.3 % of the GPS L1 rows of a day (up to 5 TECU there); the
          slant RMS against the COD GIM is the same either way.
        - ``gala`` is one set; the sessions (``tools.DDPreprocessing.compute_ntcm``) pass per epoch the set in
          force (RINEX 4 '> ION E' records, the latest at or before the epoch; RINEX 3 the header set; since v1.0.0, before that one set per run).
        - Model accuracy (slant TEC against the COD GIM on the validation rows; Klobuchar
          on the same lines of sight in brackets): BRUX 2024-02-04 10-11 h -17.9 (-4.7) TECU, NYA2 2024-035 -9.7
          mean / 11.9 RMS, NKLG 2024-132 (storm) +21.0..+24.0 mean / 28.5-31.6 RMS (-1.4..+0.1 / 13.6-14.4),
          NYA2 2024-132 +18.5 mean / 19.6 RMS, scale 1.95 (+1.4 / 6.4). On those rows NTCM SPP stays above a
          reference computed with the Klobuchar model; the implementation is not the cause.
    """

    _check_choice("mf_no", mf_no, [1, 2, 3, 4])
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
    shell_height: float | None = None,
    earth_radius: float | None = None,
    doy: int | None = None,
) -> xr.Dataset:
    """
    Calculate VTEC from the NTCM model on a grid (lat, lon, time) and return xarray.Dataset.

    Status
    ------
    Experimental grid-generation helper used by tutorials/research workflows.
    It is not part of the core STEC measurement pipeline.

    The NTCM-G VTEC (eq. 3) is evaluated directly at each grid node, which is
    the ionospheric point (geocentric latitude, as in IONEX), vectorised over
    the grid. No line of sight, elevation or mapping function is involved.

    Parameters
    ----------
    lat_grid : array-like
        Vector of latitudes (deg).
    lon_grid : array-like
        Vector of longitudes (deg).
    times : array-like
        Vector of GPST times (e.g., np.datetime64, list[Timestamp], DatetimeIndex);
        converted to UTC for the local time, as in ``ntcm_vtec``.
    gal_alpha : any
        Galileo broadcast coefficients (ai0, ai1, ai2), as ``gala`` of ``ntcm_vtec``.
    shell_height, earth_radius : float, optional
        Deprecated and unused: the grid nodes are the ionospheric points.
    doy : int or None, optional
        Day-of-year used for every epoch; None takes the day of the first element of `times`.

    Returns
    -------
    xr.Dataset
        Dataset with one variable 'V' with dimensions (time, lat, lon),
        with unit and description attributes.
    """
    import warnings

    if shell_height is not None or earth_radius is not None:
        warnings.warn("compute_ntcm_grid: shell_height and earth_radius are unused (the grid nodes are the "
                      "ionospheric points) and will be removed", DeprecationWarning, stacklevel=2)

    lat_grid = np.asarray(lat_grid, dtype=float)
    lon_grid = np.asarray(lon_grid, dtype=float)
    times = pd.to_datetime(times)

    if doy is None:
        doy = times[0].dayofyear

    gala = gal_alpha
    lat2, lon2 = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    cos3x, cos2x = sol_zen_deps(lat2, sun_dec(doy))
    fmag, fmag_deg = geographic2geomagnetic_latitude(fi_pp=lat2, la_pp=lon2)
    azpar = np.abs(np.sqrt(gala[0] ** 2 + 1633.33 * gala[1] ** 2 + 4802000 * gala[2] ** 2
                           + 3266.67 * gala[0] * gala[2]))
    V = np.empty((times.size, lat_grid.size, lon_grid.size), dtype=float)
    for k, t in enumerate(times):
        lt = get_local_time(lon2, ut=gpst2utc(pd.Timestamp(t)))
        V[k] = get_vtec(LT=lt, doy=doy, fmag=fmag, fmag_deg=fmag_deg, klobpar=azpar, cos3x=cos3x,
                        cos2x=cos2x, sys='GAL')

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

    ds_vtec = xr.Dataset({"V": da_V})
    return ds_vtec
