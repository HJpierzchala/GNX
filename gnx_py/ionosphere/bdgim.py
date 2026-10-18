"""BDGIM, the BeiDou-3 global broadcast ionospheric delay correction model.

Implements the single-frequency model of BDS-SIS-ICD-B2b-1.0 §7.7 (CSNO 2020): the nine broadcast coefficients
(Table 7-8), Eq. 7-5 to 7-16, the broadcast terms of Table 7-9 and the 17 predicted (non-broadcast) terms with the
coefficients and periods of Table 7-10. Implemented from the published algorithm; no third-party code (MIT). The
Table 7-10 numbers below are data transcribed from that table and checked against the published page; the ICD
itself is cited, not copied.

Inputs and units
----------------
* ``alpha``: the nine broadcast coefficients alpha1..alpha9 [TECU], shape (9,) or (..., 9), broadcast against the
  other inputs (one set per row is allowed). RINEX 4 '> ION Cxx CNVX' records carry them in TECU already.
* Angles in degrees (latitude, longitude, azimuth, elevation), as in :mod:`gnx_py.ionosphere.models`; radians are
  used internally. The user latitude/longitude are geodetic, as the ICD takes them. The pierce point (IPP) is
  computed on a spherical Earth, R = 6378 km, shell height 400 km (the values suggested by the ICD).
* VTEC and STEC in TECU (1 TECU = 1e16 electrons/m^2). The model returns TECU; only :func:`bdgim_delay_icd`
  converts to metres with the ICD constant 40.28e16 (Eq. 7-5). GNX sessions convert TECU to metres with the
  GNX constant 40.3e16, as for GIM, Klobuchar and NTCM (difference 0.05 %, at most 1 cm at 20 m).
* Time: BDT, as numpy ``datetime64`` only (a float MJD or seconds raise ``TypeError``). GPST epochs are converted
  with :func:`gpst_to_bdt` (BDT = GPST - 14 s, BDS ICD §3.3). The mean solar longitude (Eq. 7-9) and the
  prediction epoch t_p (Eq. 7-14) are computed from the BDT epoch in integer nanoseconds, so an exact even hour
  is never moved across the hour by float rounding.

Conventions where the ICD leaves a choice
-----------------------------------------
* The one-argument arctan of Eq. 7-7, 7-8 and 7-9 is evaluated as the quadrant-correct two-argument atan2. The
  literal arctan of Eq. 7-7 puts pierce points beyond the pole on the wrong meridian (slant TEC errors up to
  6.7 TECU on the NYA2 lines of sight of 2024-035); with the literal arctan of Eq. 7-8 and 7-9 too,
  the 2024-035 grid against the COD GIM has an RMS of 22.1 instead of 12.4 TECU.
* The third period of Table 7-10 is used as printed, T3 = 0.33 day (not 1/3).
* t_p is the odd hour of the same BDT day nearest to the epoch; an exact even hour is a tie and takes the later
  odd hour (00:00 -> 01:00, 12:00 -> 13:00); 23:xx -> 23:00.
* A negative VTEC is returned as is: the ICD does not clamp the model. The GNX sessions clamp their own
  ionosphere delay at 0 and count the rows.
* Rows with a negative elevation give NaN in every output of :func:`bdgim_slant`; NaT epochs give NaN VTEC.

Interface (additions only with defaults)
----------------------------------------
* Constants ``RE_KM``, ``H_ION_KM``, ``LAT_MPOLE_DEG``, ``LON_MPOLE_DEG``, ``K_ICD`` (golden tests only),
  ``BDT_MINUS_GPST`` (``timedelta64``, equal to :data:`gnx_py.ionosphere.models.BDT_MINUS_GPST_S`), and the
  tables ``NM_BROADCAST``, ``NM_PREDICT``, ``PERIOD_DAYS``, ``A0_COEF``, ``A_COEF``, ``B_COEF``.
* :func:`gpst_to_bdt`, :func:`prediction_epoch`, :func:`bdgim_vtec`, :func:`bdgim_slant`, :func:`bdgim_delay_icd`,
  :func:`compute_bdgim_grid` (``time_scale`` keyword required).
* The lower-level helpers (IPP, geomagnetic and Sun-fixed coordinates, Legendre functions, mapping function) are
  importable for tests; their signatures may change.

Coefficient sets are selected in GPST (``rinex_nav.broadcast_iono_at``; the transmission time is stored in GPST);
the model is evaluated in BDT. This module imports nothing from the readers or the sessions; it is imported by its
module path (``gnx_py.ionosphere.bdgim``) and is not re-exported from :mod:`gnx_py.ionosphere`.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from ..core.time import _gps_utc_offset_seconds
from .models import BDT_MINUS_GPST_S

__all__ = [
    "RE_KM", "H_ION_KM", "LAT_MPOLE_DEG", "LON_MPOLE_DEG", "K_ICD", "BDT_MINUS_GPST",
    "NM_BROADCAST", "NM_PREDICT", "PERIOD_DAYS", "A0_COEF", "A_COEF", "B_COEF",
    "gpst_to_bdt", "prediction_epoch", "bdgim_vtec", "bdgim_slant", "bdgim_delay_icd", "compute_bdgim_grid",
]

# ---------------------------------------------------------------------------------------------- constants
#: Mean radius of the Earth [km] (BDS-SIS-ICD-B2b-1.0 §7.7.2, suggested value)
RE_KM = 6378.0
#: Height of the ionospheric single layer [km] (§7.7.2, suggested value)
H_ION_KM = 400.0
#: Geographic latitude / longitude of the north geomagnetic pole [deg] (§7.7.2)
LAT_MPOLE_DEG = 80.27
LON_MPOLE_DEG = -72.58
#: Eq. 7-5: delay [m] = 40.28e16 / f^2 * STEC [TECU]; used only by :func:`bdgim_delay_icd` (golden tests)
K_ICD = 40.28e16
#: BDT - GPST (BDS ICD §3.3), equal to :data:`gnx_py.ionosphere.models.BDT_MINUS_GPST_S`
BDT_MINUS_GPST = np.timedelta64(int(BDT_MINUS_GPST_S), "s")

#: Table 7-9: (n, m) of the nine broadcast terms A_i (alpha1..alpha9); m < 0 is the sine term
NM_BROADCAST = ((0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, -1), (2, 2), (2, -2))

#: Table 7-10: (n, m) of the 17 predicted terms B_j
NM_PREDICT = ((3, 0), (3, 1), (3, -1), (3, 2), (3, -2), (3, 3), (3, -3), (4, 0), (4, 1), (4, -1), (4, 2), (4, -2),
              (5, 0), (5, 1), (5, -1), (5, 2), (5, -2))

#: Table 7-10: periods T_k [day], k = 1..12 (T3 = 0.33 as printed)
PERIOD_DAYS = np.array([1.0, 0.5, 0.33, 14.6, 27.0, 121.6, 182.51, 365.25, 4028.71, 2014.35, 1342.90, 1007.18])

#: Table 7-10: a_{0,j} [TECU], j = 1..17
A0_COEF = np.array([-0.61, -1.31, -2.00, -0.03, 0.15, -0.48, -0.40, 2.28, -0.16, -0.21, -0.10, -0.13, 0.21, 0.68,
                    1.06, 0.0, -0.12])

#: Table 7-10: a_{k,j} [TECU], shape (12, 17), rows k = 1..12
A_COEF = np.array([
    [-0.51, -0.43, 0.34, -0.01, 0.17, 0.02, -0.06, 0.30, 0.44, -0.28, -0.31, -0.17, 0.04, 0.39, -0.12, 0.12, 0.0],
    [-0.06, -0.05, 0.06, 0.17, 0.15, 0.0, 0.11, -0.05, -0.16, 0.02, 0.11, 0.04, 0.12, 0.07, 0.02, -0.14, -0.14],
    [0.01, -0.03, 0.01, -0.01, 0.05, -0.03, 0.05, -0.03, -0.01, 0.0, -0.08, -0.04, 0.0, -0.02, -0.03, 0.0, -0.03],
    [-0.01, 0.0, 0.01, 0.0, 0.01, 0.0, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.03, 0.01, 0.02, 0.01, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.19, -0.02, 0.12, -0.10, 0.06, 0.0, -0.02, -0.08, -0.02, -0.07, 0.01, 0.03, 0.15, 0.06, -0.05, -0.03, -0.10],
    [-0.18, 0.06, -0.55, -0.02, 0.09, -0.08, 0.0, 0.86, -0.18, -0.05, -0.07, 0.04, 0.14, -0.03, 0.37, -0.11, -0.12],
    [1.09, -0.14, -0.21, 0.52, 0.27, 0.0, 0.11, 0.17, 0.23, 0.35, -0.05, 0.02, -0.60, 0.02, 0.01, 0.27, 0.32],
    [-0.34, -0.09, -1.22, 0.05, 0.15, -0.29, -0.17, 1.58, -0.06, -0.15, 0.00, 0.13, 0.28, -0.08, 0.62, -0.01, -0.04],
    [-0.13, 0.07, -0.37, 0.05, 0.06, -0.11, -0.07, 0.46, 0.00, -0.04, 0.01, 0.07, 0.09, -0.05, 0.15, -0.01, 0.01],
    [-0.06, 0.13, -0.07, 0.03, 0.02, -0.05, -0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.03, 0.08, -0.01, 0.04, 0.01, -0.02, -0.02, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

#: Table 7-10: b_{k,j} [TECU], shape (12, 17), rows k = 1..12
B_COEF = np.array([
    [0.23, -0.20, -0.31, 0.16, -0.03, 0.02, 0.04, 0.18, 0.34, 0.45, 0.19, -0.25, -0.12, 0.18, 0.40, -0.09, 0.21],
    [0.02, -0.08, -0.06, -0.11, 0.15, -0.14, 0.01, 0.01, 0.04, -0.14, -0.05, 0.08, 0.08, -0.01, 0.01, 0.11, -0.12],
    [0.0, -0.02, -0.03, -0.05, -0.01, -0.07, -0.03, -0.01, 0.02, -0.01, 0.03, -0.10, 0.01, 0.05, -0.01, 0.04, 0.00],
    [0.0, -0.02, 0.01, 0.0, -0.01, 0.01, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.09, 0.07, 0.03, 0.06, 0.09, 0.01, 0.02, 0.0, -0.04, -0.02, -0.01, 0.01, -0.10, 0.0, -0.01, 0.02, 0.05],
    [0.15, -0.31, 0.13, 0.05, -0.09, -0.03, 0.06, -0.36, 0.08, 0.05, 0.06, -0.02, -0.05, 0.06, -0.20, 0.04, 0.07],
    [0.50, -0.08, -0.38, 0.36, 0.14, 0.04, 0.0, 0.25, 0.17, 0.27, -0.03, -0.03, -0.32, -0.10, 0.20, 0.10, 0.30],
    [0.0, -0.11, -0.22, 0.01, 0.02, -0.03, -0.01, 0.49, -0.03, -0.02, 0.01, 0.02, 0.04, -0.04, 0.16, -0.02, -0.01],
    [0.05, 0.03, 0.07, 0.02, -0.01, 0.03, 0.02, -0.04, -0.01, -0.01, 0.02, 0.03, 0.02, -0.04, -0.04, -0.01, 0.0],
    [0.03, -0.02, 0.04, -0.01, -0.03, 0.02, 0.01, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.04, -0.02, -0.04, 0.00, -0.01, 0.0, 0.01, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

#: Highest degree of the expansion (Table 7-10)
NMAX = 5
_N_ALPHA = len(NM_BROADCAST)
_MJD_UNIX_EPOCH = 40587                         # MJD of 1970-01-01
_DAY_NS = 86_400_000_000_000
_HOUR = np.timedelta64(1, "h")
_TIME_SCALES = ("BDT", "GPST", "UTC")
#: elements per block of :func:`vtec_rad` over the flattened broadcast shape (about 1 kB of transient memory per
#: element, so about 100 MB per block)
_BLOCK_ROWS = 100_000


# ------------------------------------------------------------------------------------------------- time
def _as_datetime64_ns(t, name: str = "t_bdt") -> np.ndarray:
    """``t`` as a datetime64[ns] array; datetime64 (or naive datetime / Timestamp objects) only, else TypeError."""
    a = np.asarray(t)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[ns]")
    if a.dtype == object and a.size and all(isinstance(x, datetime) and x.tzinfo is None for x in a.flat):
        return a.astype("datetime64[ns]")
    raise TypeError(f"BDGIM: {name} must be numpy datetime64 epochs (BDT for the model; got dtype {a.dtype}); "
                    f"float MJD or seconds are not accepted")


def _mjd_parts(t_bdt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(MJD day number int64, seconds of day float, NaT mask) of BDT epochs, from integer nanoseconds."""
    t = _as_datetime64_ns(t_bdt)
    nat = np.isnat(t)
    ns = np.where(nat, 0, t.astype(np.int64))
    day = np.floor_divide(ns, _DAY_NS)
    sod = (ns - day * _DAY_NS) / 1e9
    return day + _MJD_UNIX_EPOCH, sod, nat


def gpst_to_bdt(t_gpst) -> np.ndarray:
    """BDT epochs (datetime64[ns]) of GPST epochs: BDT = GPST - 14 s (BDS ICD §3.3).

    ``t_gpst`` is numpy datetime64 (scalar or array); floats raise ``TypeError``. Returns an array of the same
    shape (0-d for a scalar)."""
    return np.asarray(_as_datetime64_ns(t_gpst, "t_gpst") + BDT_MINUS_GPST)


def prediction_epoch(t_bdt) -> np.ndarray:
    """Prediction epoch t_p of Eq. 7-14 (datetime64[ns], BDT) of BDT epochs ``t_bdt``.

    t_p is the odd hour (01:00, 03:00, ..., 23:00) of the epoch's BDT day nearest to the epoch; an exact even hour
    is a tie and takes the later odd hour (00:00 -> 01:00, 12:00:00 -> 13:00, 11:59:59.999 -> 11:00). Computed in
    integer nanoseconds (no float rounding). NaT gives NaT."""
    t = _as_datetime64_ns(t_bdt)
    nat = np.isnat(t)
    tt = np.where(nat, np.datetime64(0, "ns"), t)
    day = tt.astype("datetime64[D]")
    hour = (tt - day) // _HOUR
    tp = day + (2 * (hour // 2) + 1) * _HOUR
    return np.asarray(np.where(nat, np.datetime64("NaT", "ns"), tp.astype("datetime64[ns]")))


def _prediction_epoch_mjd(t_bdt) -> np.ndarray:
    """t_p of Eq. 7-14 as a float MJD [day] (the phase argument of Eq. 7-14)."""
    tp = prediction_epoch(t_bdt)
    nat = np.isnat(tp)
    ns = np.where(nat, 0, tp.astype(np.int64))
    day = np.floor_divide(ns, _DAY_NS)
    frac = (ns - day * _DAY_NS) / _DAY_NS
    return np.where(nat, np.nan, (day + _MJD_UNIX_EPOCH) + frac)


def mean_sun_longitude(t_bdt) -> np.ndarray:
    """Mean solar longitude S_lon of Eq. 7-9 [rad]: pi * (1 - 2 * fraction of the BDT day)."""
    _, sod, nat = _mjd_parts(t_bdt)
    return np.where(nat, np.nan, np.pi * (1.0 - 2.0 * (sod / 86400.0)))


# ------------------------------------------------------------------------------------------- geometry
def _wrap_pi(x):
    """Angle wrapped to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def ipp_rad(lat_u, lon_u, az, el, re_km: float = RE_KM, h_km: float = H_ION_KM):
    """Eq. 7-6 and 7-7: Earth-central angle psi and geographic IPP latitude / longitude [rad].

    Inputs in radians (user geodetic latitude / longitude, azimuth, elevation). The longitude increment uses atan2;
    the IPP longitude is wrapped to [-pi, pi)."""
    lat_u, lon_u, az, el = (np.asarray(v, dtype=float) for v in (lat_u, lon_u, az, el))
    psi = np.pi / 2.0 - el - np.arcsin(np.clip(re_km / (re_km + h_km) * np.cos(el), -1.0, 1.0))
    s = np.sin(lat_u) * np.cos(psi) + np.cos(lat_u) * np.sin(psi) * np.cos(az)
    lat_g = np.arcsin(np.clip(s, -1.0, 1.0))
    num = np.sin(psi) * np.sin(az) * np.cos(lat_u)
    den = np.cos(psi) - np.sin(lat_u) * np.sin(lat_g)
    return psi, lat_g, _wrap_pi(lon_u + np.arctan2(num, den))


def geomagnetic_rad(lat_g, lon_g, lat_p: float = math.radians(LAT_MPOLE_DEG),
                    lon_p: float = math.radians(LON_MPOLE_DEG)):
    """Eq. 7-8: geomagnetic latitude / longitude [rad] of geographic points [rad] (centred dipole, pole
    (lat_p, lon_p); geomagnetic longitude 0 on the meridian through the geographic south pole, so the geographic
    north pole lies at pi). The arctan is evaluated as atan2; this equals a rotation into the dipole frame."""
    lat_g, lon_g = np.asarray(lat_g, dtype=float), np.asarray(lon_g, dtype=float)
    s = np.sin(lat_p) * np.sin(lat_g) + np.cos(lat_p) * np.cos(lat_g) * np.cos(lon_g - lon_p)
    lat_m = np.arcsin(np.clip(s, -1.0, 1.0))
    num = np.cos(lat_g) * np.sin(lon_g - lon_p) * np.cos(lat_p)
    den = np.sin(lat_p) * np.sin(lat_m) - np.sin(lat_g)
    return lat_m, np.arctan2(num, den)


def sun_geomagnetic_longitude_rad(t_bdt, lat_p: float = math.radians(LAT_MPOLE_DEG),
                                  lon_p: float = math.radians(LON_MPOLE_DEG)) -> np.ndarray:
    """The subtracted term of Eq. 7-9 [rad]: geomagnetic longitude of the mean-Sun point (0, S_lon) (atan2)."""
    s_lon = mean_sun_longitude(t_bdt)
    return np.arctan2(np.sin(s_lon - lon_p), np.sin(lat_p) * np.cos(s_lon - lon_p))


def sun_fixed_rad(lat_g, lon_g, t_bdt):
    """Eq. 7-8 and 7-9: (phi', lambda') [rad], the geomagnetic latitude and the Sun-fixed geomagnetic longitude of
    geographic points [rad] at BDT epochs (lambda' = 0 on the geomagnetic meridian of the mean Sun), lambda'
    wrapped to [-pi, pi)."""
    lat_m, lon_m = geomagnetic_rad(lat_g, lon_g)
    return lat_m, _wrap_pi(lon_m - sun_geomagnetic_longitude_rad(t_bdt))


def mapping_function(el_rad, re_km: float = RE_KM, h_km: float = H_ION_KM) -> np.ndarray:
    """Mapping function M_F of Eq. 7-16 at elevation ``el_rad`` [rad]: MF(90 deg) = 1, MF(0) = 2.954674."""
    c = re_km / (re_km + h_km) * np.cos(np.asarray(el_rad, dtype=float))
    return 1.0 / np.sqrt(1.0 - c * c)


# ------------------------------------------------------------------------------------- Legendre functions
def _double_factorial_odd(n: int) -> float:
    """(2n - 1)!! with (-1)!! = 1."""
    out = 1.0
    for k in range(2 * n - 1, 0, -2):
        out *= k
    return out


def norm_factor(n: int, m: int) -> float:
    """Normalisation N_{n,m} of Eq. 7-11 (m >= 0)."""
    delta = 1.0 if m == 0 else 0.0
    return math.sqrt(math.factorial(n - m) * (2 * n + 1) * (2.0 - delta) / math.factorial(n + m))


def legendre_unnormalized(x, nmax: int = NMAX) -> np.ndarray:
    """Associated Legendre functions P_{n,m}(x) of Eq. 7-12 (no Condon-Shortley phase): shape
    x.shape + (nmax + 1, nmax + 1), zero for m > n."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape + (nmax + 1, nmax + 1))
    c = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    for m in range(nmax + 1):
        out[..., m, m] = _double_factorial_odd(m) * c ** m
        if m + 1 <= nmax:
            out[..., m + 1, m] = x * (2 * m + 1) * out[..., m, m]
        for n in range(m + 2, nmax + 1):
            out[..., n, m] = ((2 * n - 1) * x * out[..., n - 1, m] - (n + m - 1) * out[..., n - 2, m]) / (n - m)
    return out


_NORM = np.array([[norm_factor(n, m) if m <= n else 0.0 for m in range(NMAX + 1)] for n in range(NMAX + 1)])


def legendre_normalized(x, nmax: int = NMAX) -> np.ndarray:
    """Normalised functions N_{n,m} P_{n,m}(x) of Eq. 7-11 and 7-12, shape x.shape + (nmax + 1, nmax + 1)."""
    out = legendre_unnormalized(x, nmax)
    out *= _NORM[: nmax + 1, : nmax + 1]                            # in place: no second (..., 6, 6) array
    return out


def _basis(pbar, lam, terms) -> np.ndarray:
    """A_i (Eq. 7-10) or B_j (Eq. 7-13) of the (n, m) list ``terms``: shape lam.shape + (len(terms),)."""
    cols = []
    for n, m in terms:
        p = pbar[..., n, abs(m)]
        cols.append(p * (np.cos(m * lam) if m >= 0 else np.sin(-m * lam)))
    return np.stack(cols, axis=-1)


def predicted_coefficients(tp_mjd) -> np.ndarray:
    """beta_j of Eq. 7-14 at the prediction epoch ``tp_mjd`` [MJD day, float]: shape tp.shape + (17,) [TECU]."""
    tp = np.asarray(tp_mjd, dtype=float)
    ph = tp[..., None] * (2.0 * np.pi / PERIOD_DAYS)
    return A0_COEF + np.cos(ph) @ A_COEF + np.sin(ph) @ B_COEF


# --------------------------------------------------------------------------------------------- the model
def _as_alpha(alpha) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0 or a.shape[-1] != _N_ALPHA:
        raise ValueError(f"BDGIM: alpha must have shape (9,) or (..., 9) [TECU]; got shape {a.shape}")
    return a


def _vtec_block(alpha, lat_g, lon_g, t) -> np.ndarray:
    """Eq. 7-15 on one block of rows (inputs already validated; ``t`` datetime64[ns])."""
    lat_sf, lam_sf = sun_fixed_rad(lat_g, lon_g, t)
    pbar = legendre_normalized(np.sin(lat_sf))
    a_i = _basis(pbar, lam_sf, NM_BROADCAST)
    b_j = _basis(pbar, lam_sf, NM_PREDICT)
    beta = predicted_coefficients(_prediction_epoch_mjd(t))
    return np.sum(beta * b_j, axis=-1) + np.sum(alpha * a_i, axis=-1)


def vtec_rad(alpha, lat_g, lon_g, t_bdt) -> np.ndarray:
    """VTEC [TECU] of Eq. 7-15 at geographic IPP latitude / longitude [rad] and BDT epochs (datetime64).

    VTEC = A0 + sum_i alpha_i A_i: the predicted part A0 (Eq. 7-13, 7-14, t_p of :func:`prediction_epoch`) plus the
    broadcast part (Eq. 7-10). Not clamped (a negative value is returned as is). NaT epochs give NaN.

    The transient arrays take about 1 kB per row (Legendre table, 26 basis columns), so inputs with more than
    ``_BLOCK_ROWS`` elements are evaluated in blocks of the flattened broadcast shape (each element is
    independent; the result is the same as one pass, whatever the shape, e.g. a leading length-1 axis)."""
    alpha = _as_alpha(alpha)
    t = _as_datetime64_ns(t_bdt)
    lat_g, lon_g = np.asarray(lat_g, dtype=float), np.asarray(lon_g, dtype=float)
    shape = np.broadcast_shapes(alpha.shape[:-1], lat_g.shape, lon_g.shape, t.shape)
    size = math.prod(shape)
    if size <= _BLOCK_ROWS or not shape:
        return _vtec_block(alpha, lat_g, lon_g, t)
    # flattened inputs: 8 bytes per element each (72 for per-element sets), against about 1 kB inside a block
    la, lo, tt = (np.broadcast_to(x, shape).reshape(-1) for x in (lat_g, lon_g, t))
    one_set = alpha.ndim == 1
    a = alpha if one_set else np.broadcast_to(alpha, shape + (_N_ALPHA,)).reshape(-1, _N_ALPHA)
    out = np.empty(size, dtype=float)
    for s in range(0, size, _BLOCK_ROWS):
        blk = slice(s, s + _BLOCK_ROWS)
        out[blk] = _vtec_block(a if one_set else a[blk], la[blk], lo[blk], tt[blk])
    return out.reshape(shape)


def bdgim_vtec(alpha, lat_deg, lon_deg, t_bdt) -> np.ndarray:
    """BDGIM VTEC [TECU] at a geographic point taken as the IPP (BDS-SIS-ICD-B2b-1.0 §7.7, Eq. 7-8 to 7-15).

    Parameters
    ----------
    alpha : array-like, shape (9,) or (..., 9)
        Broadcast coefficients alpha1..alpha9 [TECU]; broadcast against the other inputs.
    lat_deg, lon_deg : array-like
        Geographic latitude / longitude of the IPP [deg].
    t_bdt : numpy datetime64
        BDT epochs (use :func:`gpst_to_bdt` for GPST); floats raise ``TypeError``.

    Returns
    -------
    numpy.ndarray
        VTEC [TECU], the broadcast shape of the inputs; not clamped at 0.
    """
    return vtec_rad(alpha, np.deg2rad(np.asarray(lat_deg, dtype=float)),
                    np.deg2rad(np.asarray(lon_deg, dtype=float)), t_bdt)


def bdgim_slant(alpha, lat_deg, lon_deg, az_deg, el_deg, t_bdt) -> dict:
    """Line-of-sight BDGIM (BDS-SIS-ICD-B2b-1.0 §7.7, Eq. 7-6 to 7-16), vectorised (numpy broadcasting).

    Parameters
    ----------
    alpha : array-like, shape (9,) or (..., 9)
        Broadcast coefficients alpha1..alpha9 [TECU]; one set per row is allowed.
    lat_deg, lon_deg : array-like
        Geodetic latitude / longitude of the receiver [deg] (``flh[0]``, ``flh[1]`` in the sessions).
    az_deg, el_deg : array-like
        Azimuth and elevation of the satellite [deg].
    t_bdt : numpy datetime64
        BDT epochs (use :func:`gpst_to_bdt` for GPST); floats raise ``TypeError``.

    Returns
    -------
    dict
        ``lat_ipp``, ``lon_ipp`` [deg] (IPP on the 400 km shell, longitude in [-180, 180)), ``vtec`` [TECU] at the
        IPP, ``mf`` (Eq. 7-16) and ``stec`` = mf * vtec [TECU]. Every output is NaN where the elevation is
        negative (or NaN). VTEC and STEC are not clamped at 0. Metres: ``K / f**2 * stec``
        (:func:`bdgim_delay_icd` for the ICD constant).
    """
    el = np.asarray(el_deg, dtype=float)
    psi, lat_i, lon_i = ipp_rad(np.deg2rad(np.asarray(lat_deg, dtype=float)),
                                np.deg2rad(np.asarray(lon_deg, dtype=float)),
                                np.deg2rad(np.asarray(az_deg, dtype=float)), np.deg2rad(el))
    v = vtec_rad(alpha, lat_i, lon_i, t_bdt)
    mf = mapping_function(np.deg2rad(el))
    bad = ~(el >= 0.0)
    out = dict(lat_ipp=np.rad2deg(lat_i), lon_ipp=np.rad2deg(lon_i), vtec=v, mf=mf, stec=mf * v)
    shape = np.broadcast_shapes(*(np.shape(x) for x in out.values()))
    bad = np.broadcast_to(bad, shape)
    return {k: np.where(bad, np.nan, np.broadcast_to(x, shape)) for k, x in out.items()}


def bdgim_delay_icd(stec_tecu, freq_hz) -> np.ndarray:
    """Slant delay [m] of Eq. 7-5 with the ICD constant: 40.28e16 / f**2 * STEC [TECU] (golden tests only; the
    GNX sessions use 40.3e16)."""
    return K_ICD / np.asarray(freq_hz, dtype=float) ** 2 * np.asarray(stec_tecu, dtype=float)


# ------------------------------------------------------------------------------------------------- grid
def _to_bdt(times: np.ndarray, time_scale: str) -> np.ndarray:
    if time_scale == "BDT":
        return times
    if time_scale == "GPST":
        return times + BDT_MINUS_GPST
    out = np.empty_like(times)                                     # UTC -> GPST (leap seconds) -> BDT
    for k, t in enumerate(times):
        if np.isnat(t):
            out[k] = t
            continue
        utc = t.astype("datetime64[us]").item().replace(tzinfo=timezone.utc)
        out[k] = t + np.timedelta64(_gps_utc_offset_seconds(utc), "s") + BDT_MINUS_GPST
    return out


def compute_bdgim_grid(lat_grid, lon_grid, times, alpha, *, time_scale) -> xr.Dataset:
    """BDGIM VTEC on a (time, lat, lon) grid, the grid nodes taken as IPPs (no line of sight, no mapping).

    Mirrors :func:`gnx_py.ionosphere.models.compute_ntcm_grid` (BDS-SIS-ICD-B2b-1.0 §7.7).

    Parameters
    ----------
    lat_grid, lon_grid : array-like
        1-D geographic latitudes / longitudes of the nodes [deg].
    times : array-like of numpy datetime64
        1-D epochs in ``time_scale``.
    alpha : array-like, shape (9,) or (n_times, 9)
        Broadcast coefficients [TECU]: one set for every epoch or one per epoch.
    time_scale : {"BDT", "GPST", "UTC"}
        Required keyword (a missing one raises ``TypeError``): the scale of ``times``. IONEX map epochs are UTC.
        "UTC" is converted with the GNX leap-second table (UTC -> GPST -> BDT), "GPST" by -14 s, "BDT" is used as
        given. The scale matters: 18 s move t_p across an even BDT hour (up to 2.5 TECU on the COD grid nodes of
        2024-035/131/132, with the same coefficient set).

    Returns
    -------
    xarray.Dataset
        Variable ``vtec`` [TECU], dims (time, lat, lon); ``time`` holds the given epochs, its scale in the
        ``time_scale`` attribute. Not clamped at 0.
    """
    if time_scale not in _TIME_SCALES:
        raise ValueError(f"compute_bdgim_grid: time_scale must be one of {_TIME_SCALES}; got {time_scale!r}")
    lat_grid = np.asarray(lat_grid, dtype=float).ravel()
    lon_grid = np.asarray(lon_grid, dtype=float).ravel()
    t = np.atleast_1d(_as_datetime64_ns(times, "times")).ravel()
    alpha = _as_alpha(alpha)
    if alpha.ndim == 1:
        alpha = np.broadcast_to(alpha, (t.size, _N_ALPHA))
    elif alpha.shape != (t.size, _N_ALPHA):
        raise ValueError(f"compute_bdgim_grid: alpha must have shape (9,) or ({t.size}, 9); got {alpha.shape}")
    t_bdt = _to_bdt(t, time_scale)
    lat2, lon2 = np.meshgrid(np.deg2rad(lat_grid), np.deg2rad(lon_grid), indexing="ij")
    vt = np.empty((t.size, lat_grid.size, lon_grid.size), dtype=float)
    for k in range(t.size):
        vt[k] = vtec_rad(alpha[k], lat2, lon2, t_bdt[k])
    da = xr.DataArray(
        vt,
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": lat_grid, "lon": lon_grid},
        name="vtec",
        attrs={"units": "TECU", "description": "VTEC from BDGIM (BDS-SIS-ICD-B2b-1.0 §7.7)",
               "time_scale": time_scale},
    )
    return xr.Dataset({"vtec": da}, attrs={"model": "BDGIM", "time_scale": time_scale})
