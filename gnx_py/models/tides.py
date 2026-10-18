import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import G, M_earth, M_sun, R_earth
from astropy.coordinates import get_sun, get_body, ITRS
from astropy.time import Time, TimeDelta

moon_to_earth_mass_ratio = 1 / 81.30059
M_moon = M_earth * moon_to_earth_mass_ratio

# Gravitational Parameter (GM = G * M)
GM_earth = (G * M_earth).to('m3 / s2')
GM_sun = (G * M_sun).to('m3 / s2')
GM_moon = (G * M_moon).to('m3 / s2')

mass_earth = M_earth.to('kg')
mass_sun = M_sun.to('kg')
mass_moon = M_moon.to('kg')

RE = 6378136.6
#: Sun/Earth and Moon/Earth mass ratios of the solid tide (IERS Conventions 2010 TN36, IAU 2009 values, as
#: DEHANTTIDEINEL). The astropy ratios above (IAU 2015 nominal GM: 332946.078; 1/81.30059) differ by 9e-8 and
#: 2.7e-7 relative, about 0.1 micrometre of tide; since v1.0.0 GNX-py uses the IERS values in every solid-tide term.
MASS_RATIO_SUN = 332946.0482
MASS_RATIO_MOON = 0.0123000371


# ------------------- END OF TEST ---------------------
# TAI - GPST is a constant 19 s (IS-GPS-200; RTKLIB gpst2utc goes through the leap-second table).
_TAI_MINUS_GPST_S = 19.0


def _astropy_time(datetimes, time_scale="gpst"):
    """Astropy Time for epoch labels given in ``time_scale``.

    GNX epochs (RINEX observation and orbit epochs) are GPS time labels.
    astropy has no GPST scale for calendar input, so a GPST label t is passed
    as TAI t + 19 s. ``time_scale='utc'`` takes the labels as UTC.
    """
    if time_scale == "gpst":
        return Time(datetimes, scale="tai") + TimeDelta(_TAI_MINUS_GPST_S, format="sec")
    if time_scale == "utc":
        return Time(datetimes, scale="utc")
    raise ValueError(f"time_scale must be 'gpst' or 'utc', got {time_scale!r}")


def _no_epochs(datetimes) -> bool:
    """True for an empty sequence of epochs (a scalar epoch is one epoch)."""
    try:
        return len(datetimes) == 0
    except TypeError:
        return False


def _empty_body_frame() -> pd.DataFrame:
    """The frame of :func:`get_sun_ecef` / :func:`get_moon_ecef` for no epoch."""
    return pd.DataFrame({'x': np.zeros(0), 'y': np.zeros(0), 'z': np.zeros(0)},
                        index=pd.DatetimeIndex([], name='time'))


def get_sun_ecef(datetimes, time_scale="gpst"):
    """Position of the Sun in ECEF (ITRS) [m] for multiple epochs.

    ``datetimes`` are GPS time labels by default (``time_scale='gpst'``);
    pass ``time_scale='utc'`` for UTC labels. The returned index keeps the
    labels as given.
    """
    if _no_epochs(datetimes):
        # astropy cannot build a Time from a zero-size array ("cannot guess format")
        return _empty_body_frame()
    times = _astropy_time(datetimes, time_scale)
    sun_coord = get_sun(times)
    itrs = sun_coord.transform_to(ITRS(obstime=times))

    x = itrs.x.to(u.m).value
    y = itrs.y.to(u.m).value
    z = itrs.z.to(u.m).value

    sun_df = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=pd.to_datetime(datetimes))
    sun_df.index.name = 'time'
    return sun_df

def get_moon_ecef(datetimes, time_scale="gpst"):
    """Position of the Moon in ECEF (ITRS) [m] for multiple epochs (labels as in get_sun_ecef)."""
    if _no_epochs(datetimes):
        return _empty_body_frame()
    times = _astropy_time(datetimes, time_scale)
    moon = get_body('moon', times)
    moon_itrs = moon.transform_to(ITRS(obstime=times))
    x = moon_itrs.x.to(u.m).value
    y = moon_itrs.y.to(u.m).value
    z = moon_itrs.z.to(u.m).value
    moon_df = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=pd.to_datetime(datetimes))
    moon_df.index.name = 'time'
    return moon_df

def calculate_unit_vectors_and_magnitude(df):
    """Calculates the vector modulus and unit vectors for columns x, y, z."""
    df['magnitude'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
    df['unit_x'] = df['x'] / df['magnitude']
    df['unit_y'] = df['y'] / df['magnitude']
    df['unit_z'] = df['z'] / df['magnitude']
    return df


def adjust_tidal_parameters(xyz):
    """
        Calculates the modified h2 and l2 parameters and determines the station's unit vector.
        IERS Conventions 2010 eq. 7.2 with the geocentric latitude of the station (phi in eqs. 7.1-7.2 and in
        DEHANTTIDEINEL; the geodetic latitude used before v1.0.0 moved h2 by up to 3e-6, 1 micrometre).
    """
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    unit_station = xyz / np.linalg.norm(xyz)
    # IERS Conventions 2010 eq. 7.2: P2(sin phi) = (3 sin^2(phi) - 1) / 2, sin phi = z / |r|
    adjustment = (3 * unit_station[2] ** 2 - 1) / 2
    h2 = 0.6078 - 0.0006 * adjustment
    l2 = 0.0847 + 0.0002 * adjustment
    return h2, l2, unit_station


def tidal_displacement(vec, unit_station, h2, l2):
    """
        Calculates the tidal displacement vector for a given unit body vector (vec).
        Uses the scalar product between the body vector and the station vector.
    """
    dot_prod = np.dot(vec, unit_station)
    term1 = h2 * ((3 / 2) * dot_prod ** 2 - 0.5) * unit_station
    term2 = 3 * l2 * dot_prod * (vec - dot_prod * unit_station)
    return term1 + term2


def get_sun_tides(sun_df, xyz):
    h2, l2, unit_station = adjust_tidal_parameters(xyz)

    # numpy arrays
    r = np.sqrt(sun_df["x"].to_numpy()**2 + sun_df["y"].to_numpy()**2 + sun_df["z"].to_numpy()**2)
    ux = sun_df["x"].to_numpy() / r
    uy = sun_df["y"].to_numpy() / r
    uz = sun_df["z"].to_numpy() / r
    U = np.column_stack((ux, uy, uz))           # (N,3)

    # a(N) vectorized
    a = MASS_RATIO_SUN * RE**4 / r**3  # (N,), GM_sun Re^4 / (GM_earth r^3)

    # dot = U · unit_station  (N,)
    dot = U @ unit_station

    # term1, term2 vectorized
    term1 = h2 * ((1.5 * dot**2 - 0.5)[:, None]) * unit_station[None, :]
    term2 = 3.0 * l2 * (dot[:, None]) * (U - dot[:, None] * unit_station[None, :])

    tide = (term1 + term2) * a[:, None]  # (N,3)

    return pd.DataFrame(tide, columns=["sx", "sy", "sz"], index=sun_df.index)

def get_moon_tides(moon_df, xyz):
    h2, l2, unit_station = adjust_tidal_parameters(xyz)

    r = np.sqrt(moon_df["x"].to_numpy()**2 + moon_df["y"].to_numpy()**2 + moon_df["z"].to_numpy()**2)
    U = np.column_stack((moon_df["x"].to_numpy()/r, moon_df["y"].to_numpy()/r, moon_df["z"].to_numpy()/r))

    a = MASS_RATIO_MOON * RE**4 / r**3

    dot = U @ unit_station
    term1 = h2 * ((1.5 * dot**2 - 0.5)[:, None]) * unit_station[None, :]
    term2 = 3.0 * l2 * (dot[:, None]) * (U - dot[:, None] * unit_station[None, :])

    tide = (term1 + term2) * a[:, None]

    return pd.DataFrame(tide, columns=["mx", "my", "mz"], index=moon_df.index)

def get_tides(xyz, datetimes):
    """IERS 2010 step-1 solid tide [m, ECEF] at GPST labels: ``rsol`` (dx, dy, dz) is the degree-2 in-phase part
    of the Sun (``sun_tidal``) and the Moon (``moon_tidal``) plus :func:`solid_tide_step1_extra` (degree 3,
    latitude dependence l(1), out-of-phase parts). Step 2 is :func:`solid_tide_step2`.
    ``radial`` / ``north`` [m]: permanent tide of TN36 eq. 7.14 at the geocentric latitude (sin phi = z / |r|)."""
    sun_df = get_sun_ecef(datetimes)
    moon_df = get_moon_ecef(datetimes)

    sun_tidal = get_sun_tides(sun_df, xyz)      # sx,sy,sz
    moon_tidal = get_moon_tides(moon_df, xyz)   # mx,my,mz
    extra = solid_tide_step1_extra(xyz, sun_df[["x", "y", "z"]].to_numpy(dtype=float),
                                   moon_df[["x", "y", "z"]].to_numpy(dtype=float))

    rsol = pd.DataFrame({
        "dx": sun_tidal["sx"].to_numpy() + moon_tidal["mx"].to_numpy() + extra[:, 0],
        "dy": sun_tidal["sy"].to_numpy() + moon_tidal["my"].to_numpy() + extra[:, 1],
        "dz": sun_tidal["sz"].to_numpy() + moon_tidal["mz"].to_numpy() + extra[:, 2],
    }, index=sun_df.index)

    sinphi, cosphi, _, _, _ = _station_sincos(xyz)
    P2 = (3 * sinphi ** 2 - 1) / 2
    radial = (-0.1206 + 0.0001 * P2) * P2
    north = (-0.0252 + 0.0001 * P2) * (2 * sinphi * cosphi)

    return rsol, radial, north, sun_tidal, moon_tidal



# ============================================================================ bug-fix stage 11c
# IERS Conventions 2010 (TN36) step-1 terms beyond the degree-2 in-phase tide of get_sun_tides / get_moon_tides
# (section 7.1.1, "Step 1" table): degree 3 (eq. 7.6), latitude dependence through l(1) (eqs. 7.8, 7.9) and the
# out-of-phase parts of the diurnal and semidiurnal Love numbers (eqs. 7.10, 7.11). Written from the published
# equations with the body terms in the Cartesian form of eqs. 7.7b/c; DEHANTTIDEINEL / ST1IDIU / ST1ISEM / ST1L1 are
# the external test oracle only. Station latitude phi and longitude lambda are geocentric, as in TN36.

#: nominal step-1 numbers of TN36 section 7.1.1
STEP1_H3, STEP1_L3 = 0.292, 0.015
STEP1_L1_DIURNAL, STEP1_L1_SEMIDIURNAL = 0.0012, 0.0024
STEP1_HI_DIURNAL, STEP1_LI_DIURNAL = -0.0025, -0.0007
STEP1_HI_SEMIDIURNAL, STEP1_LI_SEMIDIURNAL = -0.0022, -0.0007


def _body(body):
    """(N, 3) body positions [m, ECEF] and their distances R_j."""
    b = np.atleast_2d(np.asarray(body, dtype=float))
    return b, np.sqrt(b[:, 0] ** 2 + b[:, 1] ** 2 + b[:, 2] ** 2)


def _rne_to_ecef(dr, dn, de, xyz):
    """Radial / north / east components (N,) [m] at the geocentric station frame of ``xyz`` -> ECEF (N, 3)."""
    sinphi, cosphi, cosla, sinla, _ = _station_sincos(xyz)
    return np.column_stack([dr * cosphi * cosla - dn * sinphi * cosla - de * sinla,
                            dr * cosphi * sinla - dn * sinphi * sinla + de * cosla,
                            dr * sinphi + dn * cosphi])


def _degree3(xyz, body, mass_ratio):
    """Eq. 7.6: GM_j Re^5 / (GM_E R_j^4) [h3 r (5/2 c^3 - 3/2 c) + l3 (15/2 c^2 - 3/2)(R_j - c r)], c = R_j . r."""
    b, rj = _body(body)
    r = np.asarray(xyz, dtype=float).reshape(3)
    rhat = r / np.linalg.norm(r)
    bhat = b / rj[:, None]
    c = bhat @ rhat
    fac3 = mass_ratio * RE * (RE / rj) ** 4
    radial = STEP1_H3 * (2.5 * c ** 3 - 1.5 * c)
    transverse = STEP1_L3 * (7.5 * c ** 2 - 1.5)
    return fac3[:, None] * (radial[:, None] * rhat[None, :]
                            + transverse[:, None] * (bhat - c[:, None] * rhat[None, :]))


def _p21_p22(body):
    """Eqs. 7.7b/c divided by 3: P21(sin Phi_j) cos/sin(lambda_j) and P22(sin Phi_j) cos/sin(2 lambda_j), over 3."""
    b, rj = _body(body)
    x, y, z = b[:, 0], b[:, 1], b[:, 2]
    r2 = rj ** 2
    return x * z / r2, y * z / r2, (x ** 2 - y ** 2) / r2, 2.0 * x * y / r2


def _latitude_l1_diurnal(xyz, body, fac2):
    """Eq. 7.8 (diurnal band, l(1) = 0.0012): transverse only [m, ECEF]."""
    sinphi, cosphi, cosla, sinla, _ = _station_sincos(xyz)
    p21c, p21s, _, _ = _p21_p22(body)
    d_cos = 3.0 * (p21c * cosla + p21s * sinla)       # P21(sin Phi_j) cos(lambda - lambda_j)
    d_sin = 3.0 * (p21c * sinla - p21s * cosla)       # P21(sin Phi_j) sin(lambda - lambda_j)
    dn = -STEP1_L1_DIURNAL * sinphi * fac2 * d_cos * sinphi
    de = STEP1_L1_DIURNAL * sinphi * fac2 * d_sin * (cosphi ** 2 - sinphi ** 2)
    return _rne_to_ecef(np.zeros_like(dn), dn, de, xyz)


def _latitude_l1_semidiurnal(xyz, body, fac2):
    """Eq. 7.9 (semidiurnal band, l(1) = 0.0024): transverse only [m, ECEF]."""
    sinphi, cosphi, cosla, sinla, _ = _station_sincos(xyz)
    _, _, p22c, p22s = _p21_p22(body)
    cos2la, sin2la = cosla ** 2 - sinla ** 2, 2.0 * cosla * sinla
    s_cos = 3.0 * (p22c * cos2la + p22s * sin2la)     # P22(sin Phi_j) cos 2(lambda - lambda_j)
    s_sin = 3.0 * (p22c * sin2la - p22s * cos2la)     # P22(sin Phi_j) sin 2(lambda - lambda_j)
    dn = -0.5 * STEP1_L1_SEMIDIURNAL * sinphi * cosphi * fac2 * s_cos
    de = -0.5 * STEP1_L1_SEMIDIURNAL * sinphi * cosphi * fac2 * s_sin * sinphi
    return _rne_to_ecef(np.zeros_like(dn), dn, de, xyz)


def _latitude_l1(xyz, body, fac2):
    """Eqs. 7.8 + 7.9: the l(1) latitude-dependence terms of both bands [m, ECEF]."""
    return _latitude_l1_diurnal(xyz, body, fac2) + _latitude_l1_semidiurnal(xyz, body, fac2)


def _out_of_phase_diurnal(xyz, body, fac2):
    """Eqs. 7.10a/b (hI = -0.0025, lI = -0.0007) [m, ECEF]; sin 2 Phi_j sin/cos(lambda - lambda_j) from eq. 7.7b."""
    sinphi, cosphi, cosla, sinla, _ = _station_sincos(xyz)
    p21c, p21s, _, _ = _p21_p22(body)
    s2_sin = 2.0 * (p21c * sinla - p21s * cosla)     # sin 2 Phi_j sin(lambda - lambda_j)
    s2_cos = 2.0 * (p21c * cosla + p21s * sinla)     # sin 2 Phi_j cos(lambda - lambda_j)
    dr = -0.75 * STEP1_HI_DIURNAL * fac2 * s2_sin * (2.0 * sinphi * cosphi)
    dn = -1.5 * STEP1_LI_DIURNAL * fac2 * s2_sin * (cosphi ** 2 - sinphi ** 2)
    de = -1.5 * STEP1_LI_DIURNAL * fac2 * s2_cos * sinphi
    return _rne_to_ecef(dr, dn, de, xyz)


def _out_of_phase_semidiurnal(xyz, body, fac2):
    """Eqs. 7.11a/b (hI = -0.0022, lI = -0.0007) [m, ECEF]; cos^2 Phi_j sin/cos 2(lambda - lambda_j) from eq. 7.7c."""
    sinphi, cosphi, cosla, sinla, _ = _station_sincos(xyz)
    _, _, p22c, p22s = _p21_p22(body)
    cos2la, sin2la = cosla ** 2 - sinla ** 2, 2.0 * cosla * sinla
    c2_sin = p22c * sin2la - p22s * cos2la            # cos^2 Phi_j sin 2(lambda - lambda_j)
    c2_cos = p22c * cos2la + p22s * sin2la            # cos^2 Phi_j cos 2(lambda - lambda_j)
    dr = -0.75 * STEP1_HI_SEMIDIURNAL * fac2 * c2_sin * cosphi ** 2
    dn = 0.75 * STEP1_LI_SEMIDIURNAL * fac2 * c2_sin * (2.0 * sinphi * cosphi)
    de = 0.75 * STEP1_LI_SEMIDIURNAL * fac2 * c2_cos * (-2.0 * cosphi)
    return _rne_to_ecef(dr, dn, de, xyz)


def solid_tide_step1_extra(xyz, sun, moon) -> np.ndarray:
    """IERS 2010 step-1 terms beyond the degree-2 in-phase tide [m], ECEF (N, 3).

    ``sun``, ``moon``: (N, 3) ECEF positions [m]. Sum over the Sun and the Moon (as DEHANTTIDEINEL) of degree 3
    (eq. 7.6, radial up to 1.7 mm), the l(1) latitude terms (eqs. 7.8 / 7.9, up to 0.8 / 1.0 mm transverse) and the
    out-of-phase diurnal / semidiurnal terms (eqs. 7.10 / 7.11). Part of ``solid_tides`` (no own switch).
    """
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    out = 0.0
    for body, ratio in ((sun, MASS_RATIO_SUN), (moon, MASS_RATIO_MOON)):
        _, rj = _body(body)
        fac2 = ratio * RE * (RE / rj) ** 3          # GM_j Re^4 / (GM_E R_j^3)
        out = out + (_degree3(xyz, body, ratio) + _latitude_l1(xyz, body, fac2)
                     + _out_of_phase_diurnal(xyz, body, fac2) + _out_of_phase_semidiurnal(xyz, body, fac2))
    return out


# ============================================================================ bug-fix stage 11b
# IERS Conventions 2010 step-2 solid tide (frequency dependence of the Love numbers), the pole tide (7.1.4) and the
# combined site displacement used by the preprocessing (solid tide + step 2 + ocean tide loading + pole tide).

_J2000_TT = np.datetime64("2000-01-01T12:00:00", "ns")
#: TT - GPST [s] (TT - TAI 32.184 s, TAI - GPST 19 s)
_TT_MINUS_GPST_S = 32.184 + _TAI_MINUS_GPST_S


def gpst_minus_utc(times) -> np.ndarray:
    """GPST - UTC [s] at GPST epochs (leap-second table ``gnx_py.time._LEAP_TABLE``; 18 s since 2017)."""
    from ..core.time import _LEAP_TABLE

    t = np.atleast_1d(np.asarray(pd.to_datetime(np.atleast_1d(times)).to_numpy(), dtype="datetime64[ns]"))
    since = np.array([np.datetime64(d.replace(tzinfo=None), "ns") + np.timedelta64(v, "s")
                      for d, v in _LEAP_TABLE], dtype="datetime64[ns]")
    vals = np.array([v for _, v in _LEAP_TABLE], dtype=np.int64)
    k = np.searchsorted(since, t, side="right") - 1
    return np.where(k >= 0, vals[np.clip(k, 0, None)], 0)


def gpst_to_utc(times) -> np.ndarray:
    """UTC epochs (datetime64[ns]) of GPST epoch labels (RTKLIB ``gpst2utc``)."""
    t = np.atleast_1d(np.asarray(pd.to_datetime(np.atleast_1d(times)).to_numpy(), dtype="datetime64[ns]"))
    return t - (gpst_minus_utc(t) * 1_000_000_000).astype("timedelta64[ns]")


def _tt_centuries(times_gpst) -> np.ndarray:
    """TT Julian centuries since J2000 of GPST labels (DEHANTTIDEINEL: T of the UTC date + (TAI-UTC + 32.184) s)."""
    t = np.atleast_1d(np.asarray(pd.to_datetime(np.atleast_1d(times_gpst)).to_numpy(), dtype="datetime64[ns]"))
    sec = (t - _J2000_TT).astype("int64") * 1e-9 + _TT_MINUS_GPST_S
    return sec / (86400.0 * 36525.0)


# Derived work of the IERS Conventions software routines STEP2DIU and STEP2LON (IERS Conventions Software License,
# reproduced in THIRD_PARTY_NOTICES): the tables and fundamental arguments are the routines'; GNX re-implements them
# vectorised in NumPy under other names (step2_diurnal, step2_long_period, solid_tide_step2) and feeds them GPST
# labels converted to the routines' UTC hour and TT centuries. Not IERS Conventions software, and neither
# distributed nor endorsed by the IERS Conventions Center.
# STEP2DIU (IERS 2010, B.E. Stetzler 2010-10-20 version): Doodson multipliers of s, h, p, N', ps and the in-phase /
# out-of-phase radial and tangential corrections [mm] of 31 diurnal constituents (K1 row: 12.00 / -0.80 / -0.67 / -0.03)
_STEP2DIU = np.array([
    [-3, 0, 2, 0, 0, -0.01, 0, 0, 0],
    [-3, 2, 0, 0, 0, -0.01, 0, 0, 0],
    [-2, 0, 1, -1, 0, -0.02, 0, 0, 0],
    [-2, 0, 1, 0, 0, -0.08, 0, -0.01, 0.01],
    [-2, 2, -1, 0, 0, -0.02, 0, 0, 0],
    [-1, 0, 0, -1, 0, -0.10, 0, 0, 0],
    [-1, 0, 0, 0, 0, -0.51, 0, -0.02, 0.03],
    [-1, 2, 0, 0, 0, 0.01, 0, 0, 0],
    [0, -2, 1, 0, 0, 0.01, 0, 0, 0],
    [0, 0, -1, 0, 0, 0.02, 0, 0, 0],
    [0, 0, 1, 0, 0, 0.06, 0, 0, 0],
    [0, 0, 1, 1, 0, 0.01, 0, 0, 0],
    [0, 2, -1, 0, 0, 0.01, 0, 0, 0],
    [1, -3, 0, 0, 1, -0.06, 0, 0, 0],
    [1, -2, 0, -1, 0, 0.01, 0, 0, 0],
    [1, -2, 0, 0, 0, -1.23, -0.07, 0.06, 0.01],
    [1, -1, 0, 0, -1, 0.02, 0, 0, 0],
    [1, -1, 0, 0, 1, 0.04, 0, 0, 0],
    [1, 0, 0, -1, 0, -0.22, 0.01, 0.01, 0],
    [1, 0, 0, 0, 0, 12.00, -0.80, -0.67, -0.03],
    [1, 0, 0, 1, 0, 1.73, -0.12, -0.10, 0],
    [1, 0, 0, 2, 0, -0.04, 0, 0, 0],
    [1, 1, 0, 0, -1, -0.50, -0.01, 0.03, 0],
    [1, 1, 0, 0, 1, 0.01, 0, 0, 0],
    [0, 1, 0, 1, -1, -0.01, 0, 0, 0],
    [1, 2, -2, 0, 0, -0.01, 0, 0, 0],
    [1, 2, 0, 0, 0, -0.11, 0.01, 0.01, 0],
    [2, -2, 1, 0, 0, -0.01, 0, 0, 0],
    [2, 0, -1, 0, 0, -0.02, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 1, 0, 0, 0, 0, 0],
], dtype=float)

# STEP2LON (IERS 2010): 5 long-period constituents, in-phase radial / tangential, out-of-phase radial / tangential [mm]
_STEP2LON = np.array([
    [0, 0, 0, 1, 0, 0.47, 0.23, 0.16, 0.07],
    [0, 2, 0, 0, 0, -0.20, -0.12, -0.11, -0.05],
    [1, 0, -1, 0, 0, -0.11, -0.08, -0.09, -0.04],
    [2, 0, 0, 0, 0, -0.13, -0.11, -0.15, -0.07],
    [2, 0, 0, 1, 0, -0.05, -0.05, -0.06, -0.03],
], dtype=float)


def _fundamental_arguments_deg(T):
    """s, h, p, N', ps [deg] of STEP2DIU / STEP2LON (s with the precession term PR), reduced with Fortran DMOD."""
    s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * T) * T) * T
    pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * T) * T) * T) * T
    h = 280.46645 + (36000.7697489 + (0.00030322222 + (0.000000020 + (-0.00000000654) * T) * T) * T) * T
    p = 83.35324312 + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * T) * T) * T) * T
    zns = 234.95544499 + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * T) * T) * T) * T
    ps = 282.93734098 + (1.71945766667 + (0.00045688889 + (-0.00000001778 + (-0.00000000334) * T) * T) * T) * T
    return s, pr, h, p, zns, ps


def _station_sincos(xyz):
    r = np.asarray(xyz, dtype=float).reshape(3)
    rsta = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    sinphi = r[2] / rsta
    cosphi = np.sqrt(r[0] ** 2 + r[1] ** 2) / rsta
    zla = np.arctan2(r[1], r[0])                 # 0 on the polar axis (x = y = 0), where x / (r cos phi) is 0 / 0
    return sinphi, cosphi, np.cos(zla), np.sin(zla), zla


def step2_diurnal(xyz, fhr, T) -> np.ndarray:
    """IERS 2010 STEP2DIU: diurnal-band frequency-dependence correction [m], ECEF (N, 3).

    ``fhr``: UTC hour of the day; ``T``: TT Julian centuries since J2000 (as DEHANTTIDEINEL passes them).
    """
    fhr = np.atleast_1d(np.asarray(fhr, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    s, pr, h, p, zns, ps = _fundamental_arguments_deg(T)
    tau = fhr * 15.0 + 280.4606184 + (36000.7700536 + (0.00038793 + (-0.0000000258) * T) * T) * T + (-s)
    s = s + pr
    s, tau, h, p, zns, ps = (np.fmod(a, 360.0) for a in (s, tau, h, p, zns, ps))
    sinphi, cosphi, cosla, sinla, zla = _station_sincos(xyz)
    d = _STEP2DIU
    theta = (tau[:, None] + d[None, :, 0] * s[:, None] + d[None, :, 1] * h[:, None] + d[None, :, 2] * p[:, None]
             + d[None, :, 3] * zns[:, None] + d[None, :, 4] * ps[:, None]) * (2.0 * np.pi / 360.0)
    arg = theta + zla
    sa, ca = np.sin(arg), np.cos(arg)
    dr = d[None, :, 5] * 2.0 * sinphi * cosphi * sa + d[None, :, 6] * 2.0 * sinphi * cosphi * ca
    dn = d[None, :, 7] * (cosphi ** 2 - sinphi ** 2) * sa + d[None, :, 8] * (cosphi ** 2 - sinphi ** 2) * ca
    de = d[None, :, 7] * sinphi * ca - d[None, :, 8] * sinphi * sa
    x = dr * cosla * cosphi - de * sinla - dn * sinphi * cosla
    y = dr * sinla * cosphi + de * cosla - dn * sinphi * sinla
    z = dr * sinphi + dn * cosphi
    return np.column_stack([x.sum(axis=1), y.sum(axis=1), z.sum(axis=1)]) / 1000.0


def step2_long_period(xyz, T) -> np.ndarray:
    """IERS 2010 STEP2LON: long-period-band frequency-dependence correction [m], ECEF (N, 3)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    s, pr, h, p, zns, ps = _fundamental_arguments_deg(T)
    s = s + pr
    s, h, p, zns, ps = (np.fmod(a, 360.0) for a in (s, h, p, zns, ps))
    sinphi, cosphi, cosla, sinla, _ = _station_sincos(xyz)
    d = _STEP2LON
    theta = (d[None, :, 0] * s[:, None] + d[None, :, 1] * h[:, None] + d[None, :, 2] * p[:, None]
             + d[None, :, 3] * zns[:, None] + d[None, :, 4] * ps[:, None]) * (2.0 * np.pi / 360.0)
    ct, st = np.cos(theta), np.sin(theta)
    p2 = (3.0 * sinphi ** 2 - 1.0) / 2.0
    dr = d[None, :, 5] * p2 * ct + d[None, :, 7] * p2 * st
    dn = d[None, :, 6] * (cosphi * sinphi * 2.0) * ct + d[None, :, 8] * (cosphi * sinphi * 2.0) * st
    x = dr * cosla * cosphi - dn * sinphi * cosla
    y = dr * sinla * cosphi - dn * sinphi * sinla
    z = dr * sinphi + dn * cosphi
    return np.column_stack([x.sum(axis=1), y.sum(axis=1), z.sum(axis=1)]) / 1000.0


def solid_tide_step2(xyz, times_gpst) -> np.ndarray:
    """IERS 2010 step-2 solid-tide correction (STEP2DIU + STEP2LON) [m], ECEF (N, 3), at GPST epoch labels.

    Added to the step-1 displacement of :func:`get_tides`; the K1 radial term reaches 12 mm sin(2 phi)
    (about 12 mm at 45 deg latitude). DEHANTTIDEINEL arguments: FHR = UTC hour, T = TT centuries since J2000.
    """
    t = np.atleast_1d(np.asarray(pd.to_datetime(np.atleast_1d(times_gpst)).to_numpy(), dtype="datetime64[ns]"))
    utc = gpst_to_utc(t)
    fhr = (utc - utc.astype("datetime64[D]").astype("datetime64[ns]")).astype("int64") * 1e-9 / 3600.0
    T = _tt_centuries(t)
    return step2_diurnal(xyz, fhr, T) + step2_long_period(xyz, T)


# ---------------------------------------------------------------------------- pole tide
#: mean pole models of the pole tide: "secular" (IERS Conventions 2010 updated in 2018, eq. 7.25: the secular pole
#: used with IGS20/ITRF2020) and "cubic" (the original IERS 2010 Table 7.7 model, cubic until 2010.0 and linear
#: after, as RTKLIB 2.4.2 iers_mean_pole)
POLE_TIDE_MEAN_POLES = ("secular", "cubic")


def mean_pole_mas(times_ut, model: str = "secular"):
    """Mean pole (x, y) [mas] at UT epochs (years since 2000-01-01 00:00 of 365.25 days, as RTKLIB)."""
    t = np.atleast_1d(np.asarray(pd.to_datetime(np.atleast_1d(times_ut)).to_numpy(), dtype="datetime64[ns]"))
    y = (t - np.datetime64("2000-01-01T00:00:00", "ns")).astype("int64") * 1e-9 / 86400.0 / 365.25
    if model == "secular":
        return 55.0 + 1.677 * y, 320.5 + 3.460 * y
    if model == "cubic":
        y2, y3 = y * y, y * y * y
        before = y < 3653.0 / 365.25
        xp = np.where(before, 55.974 + 1.8243 * y + 0.18413 * y2 + 0.007024 * y3, 23.513 + 7.6141 * y)
        yp = np.where(before, 346.346 + 1.7896 * y - 0.10729 * y2 - 0.000908 * y3, 358.891 - 0.6287 * y)
        return xp, yp
    raise ValueError(f"mean pole model must be one of {list(POLE_TIDE_MEAN_POLES)} (got {model!r})")


def pole_tide_enu(xyz, times_ut, xp_arcsec, yp_arcsec, mean_pole: str = "secular") -> np.ndarray:
    """Solid-Earth pole tide (east, north, up) [m], (N, 3): IERS 2010 eq. 7.26 as RTKLIB ``tide_pole``.

    m1 = xp - xp_mean, m2 = -(yp - yp_mean) [arcsec]; geocentric latitude and longitude of ``xyz``.

    Not modelled: the ocean pole tide loading (IERS 2010 7.1.5, Desai 2002; up to ~2 mm radial and ~0.5 mm
    horizontal, applied by IGS repro3 / IGS20) needs the Desai coefficient grid, which GNX does not ship.
    """
    r = np.asarray(xyz, dtype=float).reshape(3)
    lat = np.arcsin(r[2] / np.linalg.norm(r))
    lon = np.arctan2(r[1], r[0])
    xbar, ybar = mean_pole_mas(times_ut, mean_pole)
    m1 = np.asarray(xp_arcsec, dtype=float) - xbar * 1e-3
    m2 = -np.asarray(yp_arcsec, dtype=float) + ybar * 1e-3
    cosl, sinl = np.cos(lon), np.sin(lon)
    de = 9E-3 * np.sin(lat) * (m1 * sinl - m2 * cosl)
    dn = -9E-3 * np.cos(2.0 * lat) * (m1 * cosl + m2 * sinl)
    du = -33E-3 * np.sin(2.0 * lat) * (m1 * cosl + m2 * sinl)
    return np.column_stack([de, dn, du])


def site_displacement(xyz, times_gpst, *, solid: bool = True, step2: bool = True, blq_station=None,
                      erp=None, mean_pole: str = "secular") -> pd.DataFrame:
    """Total tidal site displacement [m] in ECEF at GPST epoch labels.

    Columns ``dx, dy, dz`` (the sum), and per part ``solid_*`` (step 1 of :func:`get_tides` plus, with ``step2``,
    :func:`solid_tide_step2`), ``otl_e/n/u`` (ocean tide loading, ``gnx_py.ocean_loading``, with the BLQ block
    ``blq_station``; zero when None) and ``pole_e/n/u`` (pole tide with ``erp``, a
    ``gnx_py.erp.ERPData``; zero when None). OTL and pole tide take UT epochs: UTC, or UT1 = UTC + (UT1-UTC) of
    ``erp``. Index: the epochs as given, named 'time'.
    """
    from .ocean_loading import enu_to_ecef, ocean_loading_enu

    t = np.atleast_1d(np.asarray(pd.to_datetime(np.atleast_1d(times_gpst)).to_numpy(), dtype="datetime64[ns]"))
    index = pd.DatetimeIndex(t, name="time")
    n = len(t)
    solid_xyz = np.zeros((n, 3))
    if solid and n:
        rsol, _, _, _, _ = get_tides(xyz=np.asarray(xyz, dtype=float), datetimes=list(pd.DatetimeIndex(t)))
        solid_xyz = rsol[["dx", "dy", "dz"]].to_numpy(dtype=float)
        if step2:
            solid_xyz = solid_xyz + solid_tide_step2(xyz, t)
    utc = gpst_to_utc(t) if n else t
    ut = utc
    erp_values = None
    if erp is not None and n:
        erp_values = erp.at(utc)
        ut = utc + np.round(erp_values["ut1_utc"] * 1e9).astype("int64").astype("timedelta64[ns]")
    otl = np.zeros((n, 3))
    if blq_station is not None and n:
        otl = ocean_loading_enu(blq_station, ut)
    pole = np.zeros((n, 3))
    if erp_values is not None:
        pole = pole_tide_enu(xyz, ut, erp_values["xp"], erp_values["yp"], mean_pole=mean_pole)
    total = solid_xyz + enu_to_ecef(xyz, otl) + enu_to_ecef(xyz, pole)
    out = pd.DataFrame({"dx": total[:, 0], "dy": total[:, 1], "dz": total[:, 2],
                        "solid_x": solid_xyz[:, 0], "solid_y": solid_xyz[:, 1], "solid_z": solid_xyz[:, 2],
                        "otl_e": otl[:, 0], "otl_n": otl[:, 1], "otl_u": otl[:, 2],
                        "pole_e": pole[:, 0], "pole_n": pole[:, 1], "pole_u": pole[:, 2]}, index=index)
    return out
