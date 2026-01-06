import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import G, M_earth, M_sun, R_earth
from astropy.coordinates import get_sun, get_body, ITRS
from astropy.time import Time

from .utils import ecef2geodetic

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


# ------------------- END OF TEST ---------------------
def get_sun_ecef(datetimes):
    """Downloading the position of the Sun in the ECEF coordinate sys for multiple epochs."""
    times = Time(datetimes)
    sun_coord = get_sun(times)
    itrs = sun_coord.transform_to(ITRS(obstime=times))

    x = itrs.x.to(u.m).value
    y = itrs.y.to(u.m).value
    z = itrs.z.to(u.m).value

    sun_df = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=pd.to_datetime(datetimes))
    sun_df.index.name = 'time'
    return sun_df

def get_moon_ecef(datetimes):
    """Downloading the Moon's position in the ECEF coordinate sys for multiple epochs"""
    times = Time(datetimes)
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
        Uses the ecef2geodetic function to obtain the geodetic latitude (fi).
    """
    unit_station = xyz / np.linalg.norm(xyz)
    geodetic = ecef2geodetic(xyz)
    fi, lam, alt = geodetic
    # Obliczamy poprawny czynnik: (3*sin^2(fi) - 1)/2
    adjustment = (3 * (np.sin(np.deg2rad(fi)) ** 2 - 1)) / 2
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
    a = (GM_sun.value * RE**4) / (GM_earth.value * r**3)  # (N,)

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

    a = (GM_moon.value * RE**4) / (GM_earth.value * r**3)

    dot = U @ unit_station
    term1 = h2 * ((1.5 * dot**2 - 0.5)[:, None]) * unit_station[None, :]
    term2 = 3.0 * l2 * (dot[:, None]) * (U - dot[:, None] * unit_station[None, :])

    tide = (term1 + term2) * a[:, None]

    return pd.DataFrame(tide, columns=["mx", "my", "mz"], index=moon_df.index)

def get_tides(xyz, datetimes):
    fi, lam, alt = ecef2geodetic(xyz)

    sun_df = get_sun_ecef(datetimes)
    moon_df = get_moon_ecef(datetimes)

    sun_tidal = get_sun_tides(sun_df, xyz)      # sx,sy,sz
    moon_tidal = get_moon_tides(moon_df, xyz)   # mx,my,mz

    rsol = pd.DataFrame({
        "dx": sun_tidal["sx"].to_numpy() + moon_tidal["mx"].to_numpy(),
        "dy": sun_tidal["sy"].to_numpy() + moon_tidal["my"].to_numpy(),
        "dz": sun_tidal["sz"].to_numpy() + moon_tidal["mz"].to_numpy(),
    }, index=sun_df.index)

    P2 = (3 * np.sin(np.deg2rad(fi)) ** 2 - 1) / 2
    radial = (-0.1206 + 0.0001 * P2) * P2
    north = (-0.0252 + 0.0001 * P2) * np.sin(2 * np.deg2rad(fi))

    return rsol, radial, north, sun_tidal, moon_tidal

