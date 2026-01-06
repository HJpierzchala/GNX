
from math import floor, acos

import numpy as np

from .tides import get_sun_ecef  # przyjmujemy, że get_sun_ecef jest już zoptymalizowana i wektoryzowana
from .utils import calculate_distance


#########################################
# Funkcje pomocnicze geometryczne

def rel_path_corr(rsat, rrcv, const=None):
    """
    rsat: macierz Nx3 pozycji satelity
    rrcv: pozycja odbiornika (wektor 3-elementowy)
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
    """Normalize 3D vector, zwraca None, gdy wektor zerowy."""
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
    """Convert ECEF to [lat, lon, h]."""
    x, y, z = ecef
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p)
    lon = np.arctan2(y, x)
    return np.array([lat, lon, 0])


def xyz2enu(pos):
    """Convert position to ENU rotation matrix."""
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
    Zwraca współrzędne ECEF Słońca dla pojedynczej epoki.
    Jeśli przekazano listę, wykorzystaj pierwszy element.
    """
    if isinstance(time, list):
        time = time[0]
    return get_sun_ecef([time]).iloc[0].to_numpy()


#########################################
# Wektorowa wersja obliczenia efektu windup

def process_windup_correction_vectorized(times, satellite_positions, receiver_position,rsun_all):
    """
    Wektorowo oblicza korekcję efektu windup dla wielu epok.

    times: array-like obiektów datetime o długości n
    satellite_positions: np.array o kształcie (n, 3) – pozycje satelity dla kolejnych epok
    receiver_position: np.array o kształcie (3,) – pozycja odbiornika

    Zwraca: np.array windup_corrections o długości n (typ float32).
    """
    n = len(times)
    if rsun_all is None:
        rsun_all = get_sun_ecef(times)[["x", "y", "z"]].to_numpy()

    # Obliczamy wektor od odbiornika do satelity
    r = receiver_position - satellite_positions  # shape (n, 3)
    norm_r = np.linalg.norm(r, axis=1, keepdims=True)
    ek = np.where(norm_r > 0, r / norm_r, np.zeros_like(r))  # jednostkowy wektor

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

    # Obliczenia dla anteny odbiornika – są stałe, bo receiver_position jest stały
    pos = ecef2pos(receiver_position)  # [lat, lon, h]
    E = xyz2enu(pos)  # macierz rotacji 3x3
    exr = E[0:3, 1]  # jednostkowy wektor skierowany na północ (x)
    eyr_rec = -E[0:3, 0]  # jednostkowy wektor skierowany na zachód (y)

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
# Tradycyjna funkcja, pozostawiona dla porównania
def process_windup_correction(times, satellite_positions, receiver_position):
    n = len(times)
    windup_corrections = np.empty(n, dtype=np.float32)
    prev_phw = 0.0
    for i in range(n):
        windup_corrections[i] = windupcorr(times[i], satellite_positions[i], receiver_position, prev_phw)
        prev_phw = windup_corrections[i]
    return windup_corrections


#########################################
# Funkcja windupcorr – zachowujemy wersję skalarową jako odniesienie,
# ale dla optymalizacji korzystamy z wersji wektorowej w process_windup_correction_vectorized

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
    exr = E[0:3, 1]
    eyr = -E[0:3, 0]
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
