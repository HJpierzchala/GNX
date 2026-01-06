import numpy as np

_A  = 6378137.0                       # semi-equatorial axis [m]
_F  = 1 / 298.257223563               # flattening
_E2 = _F * (2.0 - _F)                 # e^{2}
_ONE_MINUS_F2 = (1.0 - _F) ** 2       # (1−f)^{2} = 1−e^{2}

def ecef2geodetic(ecef, deg: bool = True):
    """ ECEF transform (X,Y,Z) -> (φ, λ, h) for a single point or array of points.
    Supports input [X, Y, Z], (N, 3), or (3, N).
    Algorithm: Bowring + 2 Newton iterations (~1 nm precision).
    Parameters ----------
    ecef : list | tuple | np.ndarray Cartesian coordinates [m], single point (3,) or array (N,3)/(3,N).
    deg : bool, default True Whether to return latitude/longitude in degrees.
    Returns -------
    lat, lon, h : np.ndarray Latitude, longitude (rad/deg), height (m).
     Single point → shape (3,), multiple points → shape (N, 3) """
    ecef = np.asarray(ecef, dtype=np.float64)

    if ecef.ndim == 1:
        X = ecef[0]
        Y = ecef[1]
        Z = ecef[2]
        single = True
    else:
        if ecef.shape[0] == 3 and ecef.shape[1] != 3:
            ecef = ecef.T
        if ecef.shape[1] != 3:
            raise ValueError("Dane muszą być w formacie (N,3) lub (3,N) lub (3,)")
        X = ecef[:, 0]
        Y = ecef[:, 1]
        Z = ecef[:, 2]
        single = False

    A = _A
    E2 = _E2
    E2p = E2 / (1.0 - E2)

    lon = np.arctan2(Y, X)
    r2 = X*X + Y*Y
    r = np.sqrt(r2)
    u = np.arctan2(Z * (1 + E2p), r)
    sin_u = np.sin(u)
    cos_u = np.cos(u)
    lat = np.arctan2(Z + E2p * A * sin_u**3,
                     r - E2 * A * cos_u**3)

    for _ in range(2):
        sin_lat = np.sin(lat)
        N = A / np.sqrt(1 - E2 * sin_lat*sin_lat)
        h = r / np.cos(lat) - N
        lat = np.arctan2(Z, r * (1 - E2 * N / (N + h)))

    sin_lat = np.sin(lat)
    N = A / np.sqrt(1 - E2 * sin_lat*sin_lat)
    h = r / np.cos(lat) - N

    if deg:
        lat = np.degrees(lat)
        lon = np.degrees(lon)

    result = np.stack([lat, lon, h], axis=-1)
    if single:
        return result.flatten()
    return result


def geodetic2ecef(geo):
    """
        (lat, lon, h) -> (x, y, z)
        Supports a single point and an array of points.
        Input: [lat, lon, h] in radians and meters!
    """
    geo = np.asarray(geo, dtype=np.float64)
    single = False

    if geo.ndim == 1:
        lat = geo[0]
        lon = geo[1]
        h   = geo[2]
        single = True
    else:
        if geo.shape[0] == 3 and geo.shape[1] != 3:
            geo = geo.T
        if geo.shape[1] != 3:
            raise ValueError("Dane muszą być w formacie (N,3), (3,N) lub (3,)")
        lat = geo[:, 0]
        lon = geo[:, 1]
        h   = geo[:, 2]

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = _A / np.sqrt(1 - _E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - _E2) + h) * sin_lat

    result = np.stack([x, y, z], axis=-1)
    if single:
        return result.flatten()
    return result

def geodetic2spherical(geo, r=None):
    """
    (lat_gd, lon, h) -> (lat_gc, lon, R)
    Supports a single point and an array of points.
    Input: [lat_gd, lon, h] (radians, meters)
    """
    geo = np.asarray(geo, dtype=np.float64)
    single = False

    # Rozpoznaj rozmiar
    if geo.ndim == 1:
        lat_gd = geo[0]
        lon = geo[1]
        h   = geo[2]
        single = True
    else:
        if geo.shape[0] == 3 and geo.shape[1] != 3:
            geo = geo.T
        if geo.shape[1] != 3:
            raise ValueError("Data must be in (N,3), (3,N)  (3,) format")
        lat_gd = geo[:, 0]
        lon = geo[:, 1]
        h   = geo[:, 2]

    sin_lat = np.sin(lat_gd)
    cos_lat = np.cos(lat_gd)

    if r is None:
        N = _A / np.sqrt(1 - _E2 * sin_lat * sin_lat)
        rho = (N + h) * cos_lat
        z   = (N * _ONE_MINUS_F2 + h) * sin_lat
        r   = np.sqrt(rho * rho + z * z)
    else:
        r = np.asarray(r, dtype=np.float64)

    lat_gc = np.arctan2(_ONE_MINUS_F2 * sin_lat, cos_lat)

    result = np.stack([lat_gc, lon, r], axis=-1)
    if single:
        return result.flatten()
    return result


import numpy as np

def ecef2spherical(xyz):
    """
    Parameters
    ----------
    xyz : array_like, shape (3,) lub (N,3)  [m]

    Returns
    -------
    sph : ndarray, shape (3,) lub (N,3)
          [lat_gc_deg, lon_deg, R_m]
          (lat_gc = szerokość geocentryczna)
    """
    arr = np.asarray(xyz, dtype=float)
    was_1d = (arr.ndim == 1)
    if was_1d:
        arr = arr.reshape(1, 3)

    x, y, z = arr.T
    r_xy = np.hypot(x, y)          # sqrt(x^2 + y^2)
    r = np.hypot(r_xy, z)          # sqrt(r_xy^2 + z^2)
    lon = np.arctan2(y, x)
    lat_gc = np.arctan2(z, r_xy)

    out = np.column_stack((np.degrees(lat_gc), np.degrees(lon), r))
    return out[0] if was_1d else out


def spherical2ecef(sph):
    """
    Parameters
    ----------
    sph : array_like, shape (3,) lub (N,3)
          [lat_gc_deg, lon_deg, R_m]

    Returns
    -------
    xyz : ndarray, shape (3,) lub (N,3) → X,Y,Z [m]
    """
    arr = np.asarray(sph, dtype=float)
    was_1d = (arr.ndim == 1)
    if was_1d:
        arr = arr.reshape(1, 3)

    lat_gc = np.radians(arr[:, 0])
    lon    = np.radians(arr[:, 1])
    r      = arr[:, 2]

    cos_lat = np.cos(lat_gc)
    x = r * cos_lat * np.cos(lon)
    y = r * cos_lat * np.sin(lon)
    z = r * np.sin(lat_gc)

    xyz = np.column_stack((x, y, z))
    return xyz[0] if was_1d else xyz



import numpy as np

def ecef_to_enu(dXYZ, flh, degrees=True):
    """
    Converts ECEF displacements (dx, dy, dz) to ENU displacements (E, N, U).

    Parameters
    ----------
    dXYZ : array-like
        Shape (3,) or (N, 3)
    flh : array-like
        Receiver latitude, longitude, height (only lat, lon used)
    degrees : bool
        If True, lat/lon are given in degrees

    Returns
    -------
    np.ndarray
        Shape (3,) or (N, 3) – same shape as input
    """

    dXYZ = np.asarray(dXYZ, dtype=float)
    single_input = dXYZ.ndim == 1

    if single_input:
        dXYZ = dXYZ.reshape(1, 3)

    lat, lon = flh[0], flh[1]
    if degrees:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    dx = dXYZ[:, 0]
    dy = dXYZ[:, 1]
    dz = dXYZ[:, 2]

    E = -sin_lon * dx + cos_lon * dy
    N = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    U =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    enu = np.column_stack((E, N, U))

    return enu[0] if single_input else enu



def enu_to_ecef(dENU, flh, degrees=True):
    """
    Converts displacements in the ENU coordinate sys (E, N, U) to displacements in the ECEF coordinate sys (dx, dy, dz).

    Args:
    Returns:
    np.array: (dx, dy, dz) – displacements in the ECEF coordinate sys.
    """
    # Konwersja kątów na radiany, jeśli podano w stopniach
    E,N,U = dENU
    lat, lon = flh[0], flh[1]
    if degrees:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

    # Reverse transformation: rotation matrix inverse to that of ecef_to_enu
    # dx = -sin(lon)*E - sin(lat)*cos(lon)*N + cos(lat)*cos(lon)*U
    # dy =  cos(lon)*E - sin(lat)*sin(lon)*N + cos(lat)*sin(lon)*U
    # dz =  cos(lat)*N + sin(lat)*U
    dx = -np.sin(lon) * E - np.sin(lat) * np.cos(lon) * N + np.cos(lat) * np.cos(lon) * U
    dy = np.cos(lon) * E - np.sin(lat) * np.sin(lon) * N + np.cos(lat) * np.sin(lon) * U
    dz = np.cos(lat) * N + np.sin(lat) * U
    return np.array([dx, dy, dz])


def geographic2geomagnetic_latitude(fi_pp, la_pp, FGNP=79.74, LGNP=-71.78):
    """Function that computes piercing point geomagnetic latitude based on magnetic dipole model

Args:
    fi_pp (arr): IPP latitude in degrees
    la_pp (arr): IPP longitude in degrees
    FGNP (float, optional): geomagnetic north pole latitude. Defaults to 79.74.
    LGNP (float, optional): geomagnetoic north pole longitude. Defaults to -71.78.

Returns:
    arr: IPP geomagnetic latitude
"""
    FGNP = np.deg2rad(FGNP)
    LGNP = np.deg2rad(LGNP)
    fi_pp = np.deg2rad(fi_pp)
    la_pp = np.deg2rad(la_pp)
    fmag = np.arcsin(np.sin(fi_pp) * np.sin(FGNP) + np.cos(fi_pp) * np.cos(FGNP) * np.cos(la_pp - LGNP))
    return fmag, np.rad2deg(fmag)
