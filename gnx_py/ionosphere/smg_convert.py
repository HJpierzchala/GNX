import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union

# optional acceleration
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

from astropy.time import Time
from astropy.coordinates import get_sun, ITRS
import astropy.units as u
from ..conversion import ecef2geodetic, ecef2spherical, spherical2ecef, geodetic2spherical
ArrayLike = Union[np.ndarray, float]

# ---------------------- low-level numeric utils ----------------------

# --- ZAMIANA TEGO ---
# @njit(cache=True)
# def _unit(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float64)
#     n = np.linalg.norm(v, axis=-1, keepdims=True)
#     n = np.where(n == 0.0, 1.0, n)
#     return v / n

# --- NA TO: ---
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

import numpy as np

@njit(cache=True)
def _unit2d(v2: np.ndarray) -> np.ndarray:
    # v2 shape: (N, 3)
    n = np.sqrt((v2 * v2).sum(axis=1))
    # avoid div by zero
    out = np.empty_like(v2)
    for i in range(v2.shape[0]):
        d = n[i]
        if d == 0.0:
            d = 1.0
        out[i, 0] = v2[i, 0] / d
        out[i, 1] = v2[i, 1] / d
        out[i, 2] = v2[i, 2] / d
    return out

@njit(cache=True)
def _unit1d(v1: np.ndarray) -> np.ndarray:
    # v1 shape: (3,)
    s = 0.0
    for j in range(v1.shape[0]):
        s += v1[j] * v1[j]
    n = np.sqrt(s)
    if n == 0.0:
        n = 1.0
    out = np.empty_like(v1)
    for j in range(v1.shape[0]):
        out[j] = v1[j] / n
    return out

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        return _unit1d(v)
    elif v.ndim == 2:
        return _unit2d(v)
    else:
        # spłaszcz wszystko do (N,3)
        v2 = v.reshape(-1, v.shape[-1])
        u2 = _unit2d(v2)
        return u2.reshape(v.shape)


@njit(cache=True)
def _enu_angles_from_xyz(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherical angles from Cartesian (geocentric):
    lat = asin(z/r), lon = atan2(y, x)
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    r = np.sqrt(x*x + y*y + z*z)
    r = np.where(r == 0.0, 1.0, r)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return lat, lon

@njit(cache=True)
def _xyz_from_angles_r(lat: np.ndarray, lon: np.ndarray, r: np.ndarray) -> np.ndarray:
    cl = np.cos(lat)
    x = r * cl * np.cos(lon)
    y = r * cl * np.sin(lon)
    z = r * np.sin(lat)
    out = np.empty((lat.size, 3), dtype=np.float64)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = z
    return out

# ---------------------- WGS84 geodetic <-> ECEF ----------------------

# WGS84 constants
_A  = 6378137.0
_F  = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)                      # e^2
_B  = _A * (1.0 - _F)

# ZAMIANA ISTNIEJĄCEJ FUNKCJI (możesz usunąć @njit dla prostoty)
def _ecef_from_geodetic(lat: np.ndarray, lon: np.ndarray, h: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    h   = np.asarray(h,   dtype=np.float64)

    # 1) broadcast do wspólnego kształtu
    lat_b, lon_b, h_b = np.broadcast_arrays(lat, lon, h)

    # 2) obliczenia WGS84
    s = np.sin(lat_b); c = np.cos(lat_b)
    N = _A / np.sqrt(1.0 - _E2 * s*s)
    x = (N + h_b) * c * np.cos(lon_b)
    y = (N + h_b) * c * np.sin(lon_b)
    z = (N * (1.0 - _E2) + h_b) * s

    # 3) zwróć w kształcie (..., 3)
    out = np.empty(lat_b.shape + (3,), dtype=np.float64)
    out[..., 0] = x
    out[..., 1] = y
    out[..., 2] = z
    return out


@njit(cache=True)
def _geodetic_from_ecef(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bowring's method (fast, accurate) for geodetic lat, lon, h from ECEF.
    """
    x = xyz[:, 0]; y = xyz[:, 1]; z = xyz[:, 2]
    lon = np.arctan2(y, x)

    # auxiliary values
    r = np.sqrt(x*x + y*y)
    # initial latitude using Bowring formulation
    E2p = (_A*_A - _B*_B) / (_B*_B)    # second eccentricity squared
    theta = np.arctan2(z * _A, r * _B)
    st, ct = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + E2p * _B * st*st*st, r - _E2 * _A * ct*ct*ct)

    s = np.sin(lat); c = np.cos(lat)
    N = _A / np.sqrt(1.0 - _E2 * s*s)
    h  = r / c - N

    return lat, lon, h

# ---------------------- IGRF dipole axis model ----------------------

def _decimal_year(astropy_time: Time) -> np.ndarray:
    # precise fractional year from astropy (TT/UTC ok for our use)
    return astropy_time.decimalyear

@dataclass
class IGRFDipole:
    """
    Simple n=1 IGRF model to get centered-dipole axis m_hat(t) in ITRS/ECEF.

    Dipole axis vector (toward N geomagnetic pole) is:
        m_hat = -[g11, h11, g10]/norm([g11, h11, g10])

    Params are base values at t0_year with secular rates (nT/year).
    Defaults are IGRF-13 main field at 2020.0 and secular variation 2020–2025.
    """
    g10_0: float = -29404.8
    g11_0: float = -1450.9
    h11_0: float =  4652.5
    g10_dot: float =  5.7
    g11_dot: float =  7.0
    h11_dot: float = -25.9
    t0_year: float = 2020.0
    clamp_to_2025: bool = True

    def m_hat(self, t: Time) -> np.ndarray:
        y = _decimal_year(t)
        if self.clamp_to_2025:
            y = np.minimum(y, 2025.0)
        dt = y - self.t0_year

        g10 = self.g10_0 + self.g10_dot * dt
        g11 = self.g11_0 + self.g11_dot * dt
        h11 = self.h11_0 + self.h11_dot * dt

        # shape handling
        g10 = np.asarray(g10, dtype=np.float64)
        g11 = np.asarray(g11, dtype=np.float64)
        h11 = np.asarray(h11, dtype=np.float64)

        vec = np.stack([ -g11, -h11, -g10 ], axis=-1)  # minus -> toward N geomag pole
        n = np.linalg.norm(vec, axis=-1, keepdims=True)
        n = np.where(n == 0.0, 1.0, n)
        return vec / n

# ---------------------- Solar-Geomagnetic transform ----------------------

# def _sm_basis_from_sun_and_m(s_sun: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
#     """
#     Build SM basis (in ITRS) for each epoch.
#       Z = m_hat
#       X = normalized( s_sun - (s·m)m )   (projection of sun vector on plane ⟂ m)
#       Y = Z × X
#     Returns matrix with columns [X, Y, Z] in ITRS coordinates (shape [..., 3, 3]).
#     """
#     s = _unit(s_sun)
#     m = _unit(m_hat)
#
#     # X-axis: projection of sun vector onto plane perpendicular to m
#     dot = np.sum(s * m, axis=-1, keepdims=True)
#     x = s - dot * m
#     x = _unit(x)
#
#     # Y-axis: Z × X
#     y = np.cross(m, x)
#     y = _unit(y)
#
#     # Stack columns
#     # basis[:, :, 0] = X; basis[:, :, 1] = Y; basis[:, :, 2] = Z
#     basis = np.stack([x, y, m], axis=-1)
#     return basis

def _sm_basis_from_sun_and_m(s_sun: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    s = _unit(s_sun)
    m = _unit(m_hat)

    # X-axis: projection of sun vector onto plane perpendicular to m
    dot = np.sum(s * m, axis=-1, keepdims=True)
    x = s - dot * m

    # --- STRAŻNIK DEGENERACJI: s ~ // m ---
    # Jeśli ||x|| ~ 0, wybierz dowolny kierunek prostopadły do m
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    eps = 1e-12
    if x.ndim == 1:
        if n[0] < eps:
            z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            x_alt = np.cross(z, m)
            if np.linalg.norm(x_alt) < eps:
                x_alt = np.cross(np.array([1.0, 0.0, 0.0]), m)
            x = x_alt
    else:
        # wersja wektorowa (broadcast)
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        mask = (n[..., 0] < eps)
        if np.any(mask):
            x_alt = np.cross(np.broadcast_to(z, m.shape)[mask], m[mask])
            bad = (np.linalg.norm(x_alt, axis=-1) < eps)
            if np.any(bad):
                x_alt[bad] = np.cross(np.array([1.0, 0.0, 0.0]), m[mask][bad])
            x[mask] = x_alt

    x = _unit(x)

    # Y-axis: Z × X
    y = _unit(np.cross(m, x))

    basis = np.stack([x, y, m], axis=-1)
    return basis


def _apply_basis(basis_ITRS_cols: np.ndarray, v_ITRS: np.ndarray, forward: bool) -> np.ndarray:
    """
    If forward=True: transform ITRS -> SM: v_SM = B^T * v_ITRS, where B columns are SM axes in ITRS.
    If forward=False: SM -> ITRS: v_ITRS = B * v_SM.
    """
    if forward:
        # v_SM = transpose(B) @ v_ITRS
        return np.einsum('...ji,...j->...i', basis_ITRS_cols, v_ITRS)
    else:
        # v_ITRS = B @ v_SM
        return np.einsum('...ij,...j->...i', basis_ITRS_cols, v_ITRS)

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr

# ---------------------- Public class ----------------------
from dataclasses import dataclass, field


@dataclass
class SolarGeomagneticTransformer:
    """
    Transformer between:
      - geodetic (lat[rad], lon[rad], h[m])  <->  Solar-Geomagnetic geodetic (lat_SM, lon_SM, h)
      - ECEF/ITRS xyz [m]                    <->  SM Cartesian [m]

    SM frame (Knecht & Schumann 1985):
      Z = centered-dipole axis (from IGRF n=1),
      X = projection of Sun direction on plane ⟂ Z,
      Y = Z × X.

    Only 'astropy' is used for Sun and time. Everything else is NumPy/Numba.
    """
    dipole_model: IGRFDipole = field(default_factory=IGRFDipole)
    custom_m_provider: Optional[Callable[[Time], np.ndarray]] = None
    # ---------- helpers ----------
    def _sun_vec_itrs(self, t: Time) -> np.ndarray:
        sun = get_sun(t).transform_to(ITRS(obstime=t))
        x = np.atleast_1d(sun.x.to_value(u.m))
        y = np.atleast_1d(sun.y.to_value(u.m))
        z = np.atleast_1d(sun.z.to_value(u.m))
        v = np.column_stack([x, y, z])  # (N,3) nawet dla skalarnego t -> N=1
        return _unit(v)

    def _m_hat_itrs(self, t: Time) -> np.ndarray:
        if self.custom_m_provider is not None:
            v = np.asarray(self.custom_m_provider(t), dtype=np.float64)
            if v.ndim == 1:
                v = v.reshape(1, 3)
            return _unit(v)
        m = self.dipole_model.m_hat(t)
        if m.ndim == 1:
            m = m.reshape(1, 3)
        return _unit(m)

    def _basis_itrs(self, t: Time) -> np.ndarray:
        s = self._sun_vec_itrs(t)
        m = self._m_hat_itrs(t)
        return _sm_basis_from_sun_and_m(s, m)  # columns [X,Y,Z] in ITRS

    # ---------- ECEF <-> SM Cartesian ----------
    def ecef_to_sm_xyz(self, xyz_m: ArrayLike, obstime: Union[Time, str]) -> np.ndarray:
        """
        xyz_m: (..., 3) ECEF [m]
        obstime: astropy Time or anything Time() accepts. Broadcasts over leading dims:
          - If obstime is scalar and xyz has N vectors -> same basis for all N.
          - If obstime has length N and xyz has N vectors -> per-epoch basis.
        Returns v_SM in meters with same shape as xyz.
        """
        xyz = _ensure_2d(np.asarray(xyz_m, dtype=np.float64))
        t = Time(obstime) if not isinstance(obstime, Time) else obstime
        if t.isscalar:
            B = self._basis_itrs(t)           # (1,3,3)
            B = np.broadcast_to(B, (xyz.shape[0], 3, 3))
        else:
            if len(t) != xyz.shape[0]:
                raise ValueError("Length of obstime must match number of xyz vectors, or be scalar.")
            B = self._basis_itrs(t)           # (N,3,3)
        return _apply_basis(B, xyz, forward=True)

    def sm_xyz_to_ecef(self, v_sm_m: ArrayLike, obstime: Union[Time, str]) -> np.ndarray:
        vsm = _ensure_2d(np.asarray(v_sm_m, dtype=np.float64))
        t = Time(obstime) if not isinstance(obstime, Time) else obstime
        if t.isscalar:
            B = self._basis_itrs(t); B = np.broadcast_to(B, (vsm.shape[0], 3, 3))
        else:
            if len(t) != vsm.shape[0]:
                raise ValueError("Length of obstime must match number of vectors, or be scalar.")
            B = self._basis_itrs(t)
        return _apply_basis(B, vsm, forward=False)

    # ---------- Geodetic <-> SM "geodetic" (spherical) ----------
    def geodetic_to_sm(self, lat_rad: ArrayLike, lon_rad: ArrayLike, h_m: ArrayLike,
                       obstime: Union[Time, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Input: geodetic (ITRS/WGS84) latitude, longitude [rad], height [m]
        Output: SM (spherical/geocentric) latitude, longitude [rad], height [m] (height preserved).
        """
        lat = np.atleast_1d(np.asarray(lat_rad, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon_rad, dtype=np.float64))
        h   = np.atleast_1d(np.asarray(h_m,   dtype=np.float64))
        if not (lat.size == lon.size == h.size):
            raise ValueError("lat, lon, h must have same length")

        # geodetic -> ECEF -> SM Cartesian
        xyz = _ecef_from_geodetic(lat, lon, h)
        vsm = self.ecef_to_sm_xyz(xyz, obstime=obstime)

        # SM Cartesian -> spherical angles (geocentric) + keep radius -> but we keep 'h' from geodetic
        sm_lat, sm_lon = _enu_angles_from_xyz(vsm)
        return sm_lat, sm_lon, h.copy()

    def sm_to_geodetic(self, sm_lat_rad: ArrayLike, sm_lon_rad: ArrayLike, h_m: ArrayLike,
                       obstime: Union[Time, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Input: SM (spherical) latitude, longitude [rad], height [m]
        Output: geodetic (ITRS/WGS84) latitude, longitude [rad], height [m]
        """
        slat = np.atleast_1d(np.asarray(sm_lat_rad, dtype=np.float64))
        slon = np.atleast_1d(np.asarray(sm_lon_rad, dtype=np.float64))
        h    = np.atleast_1d(np.asarray(h_m,         dtype=np.float64))
        if not (slat.size == slon.size == h.size):
            raise ValueError("sm_lat, sm_lon, h must have same length")

        # choose a radius consistent with geodetic height when mapping back:
        # we don't know geocentric radius from (slat, slon, h) directly,
        # so we iterate: start with mean Earth radius and refine using geodetic mapping.
        # However, a simpler and robust approach: map unit vectors and then solve for geodetic lat/lon,h
        r_guess = _A  # start around Earth's equatorial radius
        v_sm = _xyz_from_angles_r(slat, slon, np.full(slat.size, r_guess, dtype=np.float64))
        # rotate SM->ITRS and then "replace" radius by geodetic (via inverse/forward consistency)
        v_ecef_dir = _unit(self.sm_xyz_to_ecef(v_sm, obstime=obstime))

        # find geodetic lat/lon by intersecting line along v_ecef_dir with WGS84 at height h:
        # closed form: latitude depends only on direction; then compute ECEF at given h.
        # Compute geodetic lat/lon from a far-away point along the ray:
        xyz_far = v_ecef_dir * ( _A + 1.0e6 )  # 1000 km above surface in that direction
        glat, glon, _ = _geodetic_from_ecef(xyz_far)
        xyz_final = _ecef_from_geodetic(glat, glon, h)
        # Provide exact geodetic from ECEF (cleanup numerical drift)
        lat, lon, h_out = _geodetic_from_ecef(xyz_final)
        return lat, lon, h_out

    # ---------- Convenience wrappers ----------
    def ecef_to_sm_geodetic(self, xyz_m: ArrayLike, obstime: Union[Time, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyz = _ensure_2d(np.asarray(xyz_m, dtype=np.float64))
        lat, lon, h = _geodetic_from_ecef(xyz)
        return self.geodetic_to_sm(lat, lon, h, obstime=obstime)

    def sm_geodetic_to_ecef(self, sm_lat_rad: ArrayLike, sm_lon_rad: ArrayLike, h_m: ArrayLike,
                            obstime: Union[Time, str]) -> np.ndarray:
        lat, lon, h = self.sm_to_geodetic(sm_lat_rad, sm_lon_rad, h_m, obstime=obstime)
        return _ecef_from_geodetic(lat, lon, h)

    def sm_ll_to_ecef(self,
                  sm_lat_rad: ArrayLike,
                  sm_lon_rad: ArrayLike,
                  r_norm: ArrayLike,
                  when: Union[Time, str],
                  per_point_time: bool = False) -> np.ndarray:
        """
        SM (lat,lon) + r_norm [m]  ->  ECEF xyz [m]
        - sm_lat_rad, sm_lon_rad: skalar / 1D / 2D (np. siatka Ny×Nx)
        - r_norm: skalar lub tablica broadcastowalna do kształtu sm_lat/sm_lon
        - when: epoka (astropy Time lub string), jedna dla wszystkich punktów
        Zwraca xyz o kształcie (..., 3) zgodnym z wejściem.
        """
        slat = np.asarray(sm_lat_rad, dtype=np.float64)
        slon = np.asarray(sm_lon_rad, dtype=np.float64)
        r    = np.asarray(r_norm,     dtype=np.float64)

        # broadcast do wspólnego kształtu
        slat_b, slon_b, r_b = np.broadcast_arrays(slat, slon, r)

        # na wektor 1D do obliczeń
        v_lat = slat_b.ravel()
        v_lon = slon_b.ravel()
        v_r   = r_b.ravel()

        # wektor w SM (kart.) o długości r_norm
        cl = np.cos(v_lat)
        x_sm = v_r * cl * np.cos(v_lon)
        y_sm = v_r * cl * np.sin(v_lon)
        z_sm = v_r * np.sin(v_lat)
        v_sm = np.column_stack([x_sm, y_sm, z_sm])  # (N,3)

        # obrót SM -> ITRS/ECEF
        t = when if isinstance(when, Time) else Time(when)
        xyz = self.sm_xyz_to_ecef(v_sm, obstime=t)  # (N,3)

        # powrót do kształtu wejścia
        return xyz.reshape(slat_b.shape + (3,))

    def ecef_to_geodetic(self, xyz_m: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ECEF (...,3) -> geodetyczne (lat[rad], lon[rad], h[m]) o kształcie czołowym ...
        """
        xyz = np.asarray(xyz_m, dtype=np.float64)
        shp = xyz.shape[:-1]
        xyz2 = xyz.reshape(-1, 3)
        lat, lon, h = _geodetic_from_ecef(xyz2)  # (N,), (N,), (N,)
        return lat.reshape(shp), lon.reshape(shp), h.reshape(shp)
