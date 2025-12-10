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


def to_ecef(coord):
    """Transformation of coordinates from the GCRS system to the ITRS (ECEF)."""
    gcrs = coord.transform_to('gcrs')
    itrs = gcrs.transform_to('itrs')

    # Konwersja na metry
    ecef_x = itrs.x.to(u.m).value
    ecef_y = itrs.y.to(u.m).value
    ecef_z = itrs.z.to(u.m).value
    return ecef_x, ecef_y, ecef_z


def get_sun_ecef(datetimes):
    """Downloading the position of the Sun in the ECEF coordinate system for multiple epochs."""
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
    """Downloading the Moon's position in the ECEF coordinate system for multiple epochs"""
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
    """
        Calculates the tidal effect of the Sun on the station.
        Uses an improved version of the calculation, in which element-by-element multiplication is replaced by scalar multiplication.
        """
    h2, l2, unit_station = adjust_tidal_parameters(xyz)
    sun_df = calculate_unit_vectors_and_magnitude(sun_df)
    sun_df['a'] = sun_df['magnitude'].apply(lambda r: (GM_sun.value * RE ** 4) / (GM_earth.value * r ** 3))

    def compute_tide(row):
        vec = np.array([row['unit_x'], row['unit_y'], row['unit_z']])
        return tidal_displacement(vec, unit_station, h2, l2)

    sun_df['tidal_vector'] = sun_df.apply(compute_tide, axis=1)
    sun_df['tidal_vector'] = sun_df.apply(lambda row: row['a'] * row['tidal_vector'], axis=1)
    tidal_vectors = np.vstack(sun_df['tidal_vector'].values)
    sun_tidal = pd.DataFrame(tidal_vectors, columns=['unit_x', 'unit_y', 'unit_z'], index=sun_df.index)
    return sun_tidal


def get_moon_tides(moon_df, xyz):
    """
        Calculates the tidal effect of the Moon on the station.
        Similar to get_sun_tides, it uses improved scalar product calculations.
        """
    h2, l2, unit_station = adjust_tidal_parameters(xyz)
    moon_df = calculate_unit_vectors_and_magnitude(moon_df)
    moon_df['a'] = moon_df['magnitude'].apply(lambda r: (GM_moon.value * RE ** 4) / (GM_earth.value * r ** 3))

    def compute_tide(row):
        vec = np.array([row['unit_x'], row['unit_y'], row['unit_z']])
        return tidal_displacement(vec, unit_station, h2, l2)

    moon_df['tidal_vector'] = moon_df.apply(compute_tide, axis=1)
    moon_df['tidal_vector'] = moon_df.apply(lambda row: row['a'] * row['tidal_vector'], axis=1)
    tidal_vectors = np.vstack(moon_df['tidal_vector'].values)
    moon_tidal = pd.DataFrame(tidal_vectors, columns=['unit_x', 'unit_y', 'unit_z'], index=moon_df.index)
    return moon_tidal


def get_tides(xyz, datetimes):
    """
        Combines tidal effects from the Sun and Moon for a given station position (xyz) and list of epochs (datetimes).
        We use the ecef2geodetic function to obtain the geodetic latitude.
        """
    geodetic = ecef2geodetic(xyz)
    fi, lam, alt = geodetic

    sun_df = get_sun_ecef(datetimes)
    moon_df = get_moon_ecef(datetimes)
    moon_tidal = get_moon_tides(moon_df, xyz)
    sun_tidal = get_sun_tides(sun_df, xyz)

    moon_tidal = moon_tidal.rename(columns={'unit_x': 'mx', 'unit_y': 'my', 'unit_z': 'mz'})
    sun_tidal = sun_tidal.rename(columns={'unit_x': 'sx', 'unit_y': 'sy', 'unit_z': 'sz'})

    tidals = pd.concat([moon_tidal, sun_tidal], axis=1)
    tidals['dx'] = tidals['mx'] + tidals['sx']
    tidals['dy'] = tidals['my'] + tidals['sy']
    tidals['dz'] = tidals['mz'] + tidals['sz']
    rsol = tidals[['dx', 'dy', 'dz']]

    P2 = (3 * np.sin(np.deg2rad(fi)) ** 2 - 1) / 2
    radial = (-0.1206 + 0.0001 * P2) * P2
    north = (-0.0252 + 0.0001 * P2) * np.sin(2 * np.deg2rad(fi))
    return rsol, radial, north, sun_tidal, moon_tidal
