import numpy as np
import pandas as pd

from .conversion import ecef2geodetic


def calculate_distance(points, reference_point):
    """
    This function is used for fast distance calculation
    :param points: points to which distance is calculated - x,y,z
    :param reference_point: point from which distance is calculated (e.g. observer, site) -x, y, z
    :return: distance (float)
    """
    if isinstance(points, list):
        points = np.array(points, dtype=float)
    elif isinstance(points, pd.DataFrame):
        points = points.to_numpy()

    if isinstance(reference_point, list):
        reference_point = np.array(reference_point, dtype=float)
    elif isinstance(reference_point, pd.DataFrame):
        reference_point = reference_point.to_numpy()
    points = np.array(points, dtype=float)
    reference_point = np.array(reference_point, dtype=float)

    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)

    if reference_point.ndim == 1:
        reference_point = np.expand_dims(reference_point, axis=0)

    return np.linalg.norm(points - reference_point, axis=1)


def elevation_azimuth(target_xyz, pos_xyz):
    pos_flh =  ecef2geodetic(ecef=pos_xyz,deg=True)
    fi, la, h = pos_flh[0], pos_flh[1], pos_flh[2]
    e = np.array([
        - np.sin(la),
        np.cos(la),
        0
    ])
    n = np.array([
        -np.cos(la) * (np.sin(fi)),
        -np.sin(la) * np.sin(fi),
        np.cos(fi)
    ])

    u = np.array([
        np.cos(la) * np.cos(fi),
        np.sin(la) * np.cos(fi),
        np.sin(fi)
    ])
    dist = calculate_distance(target_xyz, pos_xyz)
    p = np.array([
        (target_xyz[:, 0] - pos_xyz[0]) / dist,
        (target_xyz[:, 1] - pos_xyz[1]) / dist,
        (target_xyz[:, 2] - pos_xyz[2]) / dist
    ]).T
    ev = np.arcsin(np.dot(p, u))
    az = np.arctan2(np.dot(p, e), np.dot(p, n))
    az = (az + 2 * np.pi) % (2 * np.pi)
    return ev, az


def elevation_azimuth_geocentric(target_geo, pos_geo):
    """
    Oblicza elewację i azymut na podstawie pozycji i macierzy targetów w układzie geocentrycznym.

    Parametry:
    - target_geo: Nx3 macierz (szerokość, długość, promień) dla punktów docelowych
    - pos_geo: (szerokość, długość, promień) dla pozycji obserwatora

    Zwraca:
    - Wektory elewacji i azymutu w radianach dla każdego punktu docelowego
    """
    # Konwersja współrzędnych geocentrycznych do ECEF

    def geocentric_to_ecef(lat, lon, radius):
        """Konwersja współrzędnych geocentrycznych (lat, lon, radius) do ECEF (x, y, z)."""
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        x = radius * np.cos(lat) * np.cos(lon)
        y = radius * np.cos(lat) * np.sin(lon)
        z = radius * np.sin(lat)
        return np.array([x, y, z])

    pos_xyz = geocentric_to_ecef(*pos_geo)
    target_xyz = np.array([geocentric_to_ecef(lat, lon, radius) for lat, lon, radius in target_geo])

    # Obliczenie szerokości, długości i wysokości pozycji
    lat, lon, alt = pos_geo
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    # Wektory lokalnego układu ENU
    e = np.array([-np.sin(lon), np.cos(lon), 0])
    n = np.array([-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)])
    u = np.array([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])

    # Obliczenie odległości i wektora kierunkowego dla każdego punktu targetowego
    dist = np.linalg.norm(target_xyz - pos_xyz, axis=1)
    p = (target_xyz - pos_xyz) / dist[:, np.newaxis]

    # Obliczenie elewacji i azymutu dla każdego punktu targetowego
    ev = np.arcsin(np.dot(p, u))
    az = np.arctan2(np.dot(p, e), np.dot(p, n))
    az = (az + 2 * np.pi) % (2 * np.pi)  # Azymut w zakresie 0-2π

    return ev, az


