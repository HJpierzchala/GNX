
from math import floor, acos

import numpy as np
import pandas as pd

from ..core.conversion import ecef2geodetic
from .tides import get_sun_ecef  # przyjmujemy, że get_sun_ecef jest już zoptymalizowana i wektoryzowana
from ..core.utils import calculate_distance


#########################################
# Funkcje pomocnicze geometryczne

def rel_path_corr(rsat, rrcv, const=None):
    """Relativistic path range correction (Shapiro delay) [m].

    rsat: N x 3 matrix of satellite positions [m, ECEF]
    rrcv: receiver position (3-element vector) [m, ECEF]
    """
    if const is None:
        mi = 3986004.418 * 10 ** 8  # stała geocentryczna
        c = 299792458
        const = 2 * mi / (c ** 2)
    norm_rcv = np.linalg.norm(rrcv)
    norm_rsat = np.linalg.norm(rsat, axis=1)
    dist = calculate_distance(rsat, rrcv)  # zakładamy, że jest wektoryzowana
    return const * np.log((norm_rsat + norm_rcv + dist) / (norm_rsat + norm_rcv - dist))


def normv3(vec):
    """Normalize a 3D vector; None for a zero vector."""
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else None


def cross3(a, b):
    """3D cross product."""
    return np.cross(a, b)


def dot(a, b, n=3):
    """Dot product of vectors."""
    return np.dot(a[:n], b[:n])


def norm(vec, n=3):
    """Vector norm."""
    return np.linalg.norm(vec[:n])


def ecef2pos(ecef):
    """Convert ECEF to geodetic [lat, lon, h] (WGS84; lat/lon in rad, h in m), as RTKLIB ecef2pos."""
    lat, lon, h = ecef2geodetic(np.asarray(ecef, dtype=float), deg=False)
    return np.array([lat, lon, h])


def xyz2enu(pos):
    """ECEF -> ENU rotation matrix, row-wise: row 0 = east, row 1 = north, row 2 = up.

    RTKLIB stores the same matrix column-major, so its E[1], E[4], E[7] (north)
    is ``E[1, :]`` here and its E[0], E[3], E[6] (east) is ``E[0, :]``.
    """
    lat, lon = pos[:2]
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    E = np.array([
        -sin_lon, cos_lon, 0,
        -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat,
        cos_lat * cos_lon, cos_lat * sin_lon, sin_lat
    ]).reshape(3, 3)
    return E


#########################################
# Wektorowa wersja funkcji sunmoonpos

def sunmoonpos(time):
    """
    ECEF coordinates of the Sun at one epoch [m].
    For a list of epochs the first element is used.
    """
    if isinstance(time, list):
        time = time[0]
    return get_sun_ecef([time]).iloc[0].to_numpy()


#########################################
# Wektorowa wersja obliczenia efektu windup

OMGE = 7.2921151467e-5  # Earth rotation rate [rad/s] (IS-GPS-200, RTKLIB OMGE)


def orbit_normal_frame(times, satellite_positions, velocities=None):
    """Satellite x/y axes for orbit-normal attitude (yaw = 0), e.g. BeiDou GEO satellites.

    x is along-track (orbit normal x radial), y is minus the orbit normal and z = x * y points
    to the Earth centre: RTKLIB 2.4.3 ``sat_yaw`` with yaw = 0. The orbit normal is r x v with
    the inertial velocity v = dr/dt + OMGE x r. dr/dt is ``velocities`` (ECEF [m/s], one row per
    position; e.g. the orbit interpolation's vx/vy/vz) where given and finite, else finite
    differences over ``times``; for a single epoch without velocities dr/dt is taken as 0, which
    is exact to the orbit inclination for a GEO.
    """
    rs = np.asarray(satellite_positions, dtype=float).reshape(-1, 3)
    given = None if velocities is None else np.asarray(velocities, dtype=float).reshape(-1, 3)
    ok = None if given is None else np.isfinite(given).all(axis=1)
    vel = np.zeros_like(rs)
    if (ok is None or not ok.all()) and len(rs) >= 2:
        ti = pd.DatetimeIndex(pd.to_datetime(times))
        t = np.asarray((ti - ti[0]).total_seconds(), dtype=float)
        if np.all(np.diff(t) > 0):
            vel = np.gradient(rs, t, axis=0)
    if given is not None:
        vel[ok] = given[ok]
    vi = vel + np.cross([0.0, 0.0, OMGE], rs)
    nvec = np.cross(rs, vi)
    en = nvec / np.linalg.norm(nvec, axis=1, keepdims=True)
    es = rs / np.linalg.norm(rs, axis=1, keepdims=True)
    return np.cross(en, es), -en


def process_windup_correction_vectorized(times, satellite_positions, receiver_position, rsun_all,
                                         orbit_normal=False, *, sat_axes=None):
    """
    Phase wind-up correction [cycles] of one satellite for many epochs (vectorised).

    The model follows RTKLIB ``windupcorr`` (rtkcmn.c; Wu et al. 1993); see THIRD_PARTY_NOTICES.

    times: array-like of n datetime objects
    satellite_positions: np.array of shape (n, 3), the satellite position at each epoch [m, ECEF]
    receiver_position: np.array of shape (3,), the receiver position [m, ECEF]
    orbit_normal: True for a satellite flying orbit-normal attitude (BeiDou GEO, see
        ``orbit_normal_frame``); False (default) uses nominal yaw steering, as RTKLIB windupcorr.
    sat_axes: ``(x, y)`` satellite body axes in ECEF, (n, 3) each, from an attitude model
        (``DDPreprocessing._attitude``); they replace the frame of
        ``orbit_normal`` / nominal yaw steering, and ``rsun_all`` is not used.

    Returns: np.array of n wind-up corrections [cycles] (float32), unwrapped over the epochs.
    """
    n = len(times)
    if rsun_all is None and not orbit_normal and sat_axes is None:
        rsun_all = get_sun_ecef(times)[["x", "y", "z"]].to_numpy()

    # Obliczamy wektor od odbiornika do satelity
    r = receiver_position - satellite_positions  # shape (n, 3)
    norm_r = np.linalg.norm(r, axis=1, keepdims=True)
    ek = np.where(norm_r > 0, r / norm_r, np.zeros_like(r))  # jednostkowy wektor

    if sat_axes is not None:
        exs = np.asarray(sat_axes[0], dtype=float).reshape(-1, 3)
        eys = np.asarray(sat_axes[1], dtype=float).reshape(-1, 3)
    elif orbit_normal:
        exs, eys = orbit_normal_frame(times, satellite_positions)
    else:
        # Wektory dla anteny satelity: ezs = -rsat/||rsat||
        norm_rsat = np.linalg.norm(satellite_positions, axis=1, keepdims=True)
        ezs = np.where(norm_rsat > 0, -satellite_positions / norm_rsat, np.zeros_like(satellite_positions))

        # ess = (rsun - rsat) / ||rsun - rsat||
        diff = rsun_all - satellite_positions
        norm_diff = np.linalg.norm(diff, axis=1, keepdims=True)
        ess = np.where(norm_diff > 0, diff / norm_diff, np.zeros_like(diff))

        # eys = znormalizowany iloczyn wektorowy ezs i ess
        raw_eys = np.cross(ezs, ess)
        norm_eys = np.linalg.norm(raw_eys, axis=1, keepdims=True)
        eys = np.where(norm_eys > 0, raw_eys / norm_eys, np.zeros_like(raw_eys))

        # exs = cross(eys, ezs)
        exs = np.cross(eys, ezs)

    # Receiver antenna frame (constant: receiver_position is fixed), as RTKLIB
    # windupcorr (rtkcmn.c): dipole in the local horizontal plane, x = north,
    # y = west, from the geodetic latitude. E is row-wise (row 0 east, row 1 north).
    pos = ecef2pos(receiver_position)  # [lat, lon, h], geodetic
    E = xyz2enu(pos)  # macierz rotacji 3x3
    exr = E[1, :]  # unit vector to the north (x)
    eyr_rec = -E[0, :]  # unit vector to the west (y)

    # Dla każdego punktu obliczamy:
    eks = np.cross(ek, eys)  # shape (n,3)
    dot_ek_exs = np.sum(ek * exs, axis=1)  # shape (n,)
    ds = exs - dot_ek_exs[:, None] * ek - eks  # shape (n,3)

    dot_ek_exr = np.sum(ek * exr, axis=1)  # shape (n,)
    ekr = np.cross(ek, eyr_rec)  # shape (n,3), broadcasting exr i eyr_rec
    dr = exr - dot_ek_exr[:, None] * ek + ekr  # shape (n,3)

    ds_norm = np.linalg.norm(ds, axis=1)
    dr_norm = np.linalg.norm(dr, axis=1)
    valid = (ds_norm > 0) & (dr_norm > 0)

    dot_ds_dr = np.sum(ds * dr, axis=1)
    cosp = np.zeros(n)
    cosp[valid] = dot_ds_dr[valid] / (ds_norm[valid] * dr_norm[valid])
    cosp = np.clip(cosp, -1.0, 1.0)

    # Faza w cyklach (nie w radianach)
    ph = np.arccos(cosp) / (2 * np.pi)

    # Korekta znaku fazy – dla każdego punktu, jeśli dot(ek, cross(ds, dr)) < 0, to ph = -ph
    drs = np.cross(ds, dr)  # shape (n,3)
    sign_adjust = np.sum(ek * drs, axis=1) < 0
    ph[sign_adjust] = -ph[sign_adjust]

    # Faza unwrapped – iteracyjne unwrapping można zastąpić funkcją np.unwrap
    # np.unwrap operuje na radianach, więc przeliczamy cykle na radiany
    ph_rad = ph * 2 * np.pi
    phw_rad = np.unwrap(ph_rad, discont=np.pi)  # domyślny próg to pi
    phw = phw_rad / (2 * np.pi)

    return phw.astype(np.float32)


#########################################
# windupcorr: phase wind-up of one epoch, kept as the scalar reference of
# process_windup_correction_vectorized (which the sessions use). Ported from RTKLIB 2.4.2 src/rtkcmn.c, function
# windupcorr (T. TAKASU; BSD 2-clause licence; model of Wu et al. 1993): the vectors, their names and the
# unwrapping phw = ph + floor(prev_phw - ph + 0.5) follow it. process_windup_correction_vectorized repeats the
# arithmetic over epochs (np.unwrap instead of the floor rule; other satellite frames optional): see
# THIRD_PARTY_NOTICES, section 3.

def windupcorr(time, rs, rr, prev_phw=0):
    rsun = sunmoonpos(time)
    r = rr - rs
    ek = normv3(r)
    if ek is None:
        return 0
    ezs = normv3(-rs)
    if ezs is None:
        return 0
    ess = normv3(sunmoonpos(time) - rs)
    if ess is None:
        return 0
    eys = normv3(cross3(ezs, ess))
    if eys is None:
        return 0
    exs = cross3(eys, ezs)
    pos = ecef2pos(rr)
    E = xyz2enu(pos)
    exr = E[1, :]  # north (RTKLIB E[1], E[4], E[7])
    eyr = -E[0, :]  # west (RTKLIB -E[0], -E[3], -E[6])
    eks = cross3(ek, eys)
    ekr = cross3(ek, eyr)
    ek = ek.reshape(-1, 1)
    exs = exs.reshape(-1, 1)
    eks = eks.reshape(-1, 1)
    exr = exr.reshape(-1, 1)
    ds = exs - (np.dot(ek.T, exs)) * ek - eks
    dr = exr - (np.dot(ek.T, exr)) * ek + ekr.reshape(-1, 1)
    ds = ds.flatten()
    dr = dr.flatten()
    cosp = np.dot(ds, dr) / (norm(ds) * norm(dr))
    cosp = max(min(cosp, 1.0), -1.0)
    ph = acos(cosp) / (2 * np.pi)
    drs = cross3(ds, dr)
    if np.dot(ek.flatten(), drs) < 0:
        ph = -ph
    phw = ph + floor(prev_phw - ph + 0.5)
    return phw
