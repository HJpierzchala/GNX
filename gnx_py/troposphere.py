import numpy as np

def saastamoinen(ev, h):
    """
    This function computes tropospheric delay based on Saastamoinen model.
    :param ev: satellite elevation (float, int) [deg]
    :param h: station height (float, int) [m]
    :return: dTrop -> tropospheric delay [m]
    """
    # Conversion to radians
    el = np.abs(ev) * np.pi / 180

    # Standard atmosphere - Berg, 1948 (Bernese)
    # Pressure [mbar]
    Pr = 1013.25
    # Temperature [K]
    Tr = 291.15
    # Numerical constants for the algorithm [-] [m] [mbar]
    Hr = 50.0

    P = Pr * (1 - 0.0000226 * h) ** 5.225
    T = Tr - 0.0065 * h
    H = Hr * np.exp(-0.0006396 * h)

    # Linear interpolation
    h_a = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000])
    B_a = np.array([1.156, 1.079, 1.006, 0.938, 0.874, 0.813, 0.757, 0.654, 0.563])

    # Interpolate value of B based on h
    B = np.interp(h, h_a, B_a)

    e = 0.01 * H * np.exp(-37.2465 + 0.213166 * T - 0.000256908 * T ** 2)

    # Tropospheric error
    dTrop = ((0.002277 / np.sin(el)) * (P - (B / (np.tan(el)) ** 2)) +
            (0.002277 / np.sin(el)) * (1255 / T + 0.05) * e)
    return dTrop


def get_hydro(lat, lon, h, ev, doy):
    """Get hydrostatic part of Niell troposphere model

    Args:
        lat (_type_): _description_
        lon (_type_): _description_
        ev (_type_): _description_
        time (_type_): _description_
    """
    alpha = 2.3
    beta = 0.116e-03
    e = np.deg2rad(ev)
    t_factor = np.cos(2 * np.pi * ((doy - 28) / 365.25))
    trz_dry = alpha * np.exp(-beta * h)
    # e^x = np.exp(x)
    lat_table = np.array([15, 30, 45, 60, 75])
    a_av = np.array([1.2769934e-03, 1.2683230e-03, 1.2465397e-03, 1.2196049e-03, 1.2045996e-03])
    b_av = np.array([2.9153695e-03, 2.9152299e-03, 2.9288445e-03, 2.9022565e-03, 2.9024912e-03])
    c_av = np.array([62.610505e-03, 62.837393e-03, 63.721774e-03, 63.824265e-03, 64.258455e-03])
    a_amp = np.array([0.0, 1.2709626e-05, 2.6523662e-05, 3.4000425e-05, 4.1202191e-05])
    b_amp = np.array([0.0, 2.1414979e-05, 3.0160779e-05, 7.2562722e-05, 11.723375e-05])
    c_amp = np.array([0.0, 9.0128400e-05, 4.3497037e-05, 84.795348e-05, 170.37206e-05])
    h_corr = np.array([2.53e-05, 5.49e-03, 1.14e-03])

    aav = np.interp(lat, lat_table, a_av)
    aamp = np.interp(lat, lat_table, a_amp)
    a = aav - aamp * t_factor  # wyinterpolowane ad

    bav = np.interp(lat, lat_table, b_av)
    bamp = np.interp(lat, lat_table, b_amp)
    b = bav - bamp * t_factor  # wyinterpolowane bd
    cav = np.interp(lat, lat_table, c_av)
    camp = np.interp(lat, lat_table, c_amp)
    c = cav - camp * t_factor  # wyinterpolowane cd
    delta_m = ((1 / np.sin(e)) - marini(ev, h_corr[0], h_corr[1], h_corr[2])) * (h/1000)  # delta_m komponent
    m_dry = marini(elev=ev, a=a, b=b, c=c)  # marini(ev,ad,bd,cd)
    tro = trz_dry * m_dry + delta_m  # zenith*mapping
    return tro  # dry tropo correction


def get_wet(ev, lat):
    """Get wet part of Niell troposphere model

    Args:
        ev (_type_): _description_
        doy (_type_): _description_
        lat (_type_): _description_
    """
    trz_wet = 0.1
    lat_table = np.array([15, 30, 45, 60, 75])
    a_table = np.array([5.821897e-04, 5.6794847e-04, 5.8118019e-04, 5.9727542e-04, 6.1641693e-04])
    b_table = np.array([1.4275269e-03, 1.5138625e-03, 1.4572752e-03, 15.5007428e-03, 1.7599082e-03])
    c_table = np.array([4.3472961e-02, 4.6729510e-02, 4.3908931e-02, 4.4626982e-02, 5.4736038e-02])
    aw = np.interp(lat, lat_table, a_table)
    bw = np.interp(lat, lat_table, b_table)
    cw = np.interp(lat, lat_table, c_table)
    me = marini(ev, aw, bw, cw)
    return me, trz_wet * me  # MF and wet mapped


def niell(ev, lat, lon, h, doy):

    dry = get_hydro(lat, lon, h, ev, doy)
    me_wet, wet = get_wet(ev, lat)
    return dry + wet, dry, wet, me_wet


def marini(elev, a, b, c):
    e = np.deg2rad(elev)
    licznik = 1 + (a / (1 + (b / (1 + c))))
    mianownik = np.sin(e) + (a / (np.sin(e) + (b / (np.sin(e) + c))))
    return licznik / mianownik


def tropospheric_delay(f, h, elevation,doy):
    """
    Calculates tropospheric delay using Colins(1999) method
    Input:
        Cartesian coordinates of receiver in ECEF frame (x,y,z)
        Elevation Angle [unit: degree] of satellite vehicle
    Output:
        Tropospheric delay [unit: m]

    Reference:
    Collins, J. P. (1999). Assessment and Development of a Tropospheric Delay Model for
    Aircraft Users of the Global Positioning System. M.Sc.E. thesis, Department of
    Geodesy and Geomatics Engineering Technical Report No. 203, University of
    New Brunswick, Fredericton, New Brunswick, Canada, 174 pp
    """
    # --------------------
    # constants
    k1 = 77.604  # K/mbar
    k2 = 382000  # K^2/mbar
    Rd = 287.054  # J/Kg/K
    g = 9.80665  # m/s^2
    gm = 9.784  # m/s^2
    # --------------------
    # linear interpolation of meteorological values
    # Average values
    ave_params = np.array([
        [1013.25, 299.65, 26.31, 6.30e-3, 2.77],
        [1017.25, 294.15, 21.79, 6.05e-3, 3.15],
        [1015.75, 283.15, 11.66, 5.58e-3, 2.57],
        [1011.75, 272.15, 6.78, 5.39e-3, 1.81],
        [1013.00, 263.65, 4.11, 4.53e-3, 1.55]
    ])
    # seasonal variations
    sea_params = np.array([
        [0.00, 0.00, 0.00, 0.00e-3, 0.00],
        [-3.75, 7.00, 8.85, 0.25e-3, 0.33],
        [-2.25, 11.00, 7.24, 0.32e-3, 0.46],
        [-1.75, 15.00, 5.36, 0.81e-3, 0.74],
        [-0.50, 14.50, 3.39, 0.62e-3, 0.30]
    ])
    # Latitude index
    Latitude = np.linspace(15, 75, 5)
    if abs(f) <= 15.0:
        indexLat = 0
    elif 15 < abs(f) <= 30:
        indexLat = 1
    elif 30 < abs(f) <= 45:
        indexLat = 2
    elif 45 < abs(f) <= 60:
        indexLat = 3
    elif 60 < abs(f) < 75:
        indexLat = 4
    elif 75 <= abs(f):
        indexLat = 5
    # ----------------
    if indexLat == 0:
        ave_meteo = ave_params[indexLat, :]
        svar_meteo = sea_params[indexLat - 1, :]
    elif indexLat == 5:
        ave_meteo = ave_params[indexLat - 1, :]
        svar_meteo = sea_params[indexLat - 1, :]
    else:
        ave_meteo = ave_params[indexLat - 1, :] + (ave_params[indexLat, :] - ave_params[indexLat - 1, :]) * (
                    abs(f) - Latitude[indexLat - 1]) / (Latitude[indexLat] - Latitude[indexLat - 1])
        svar_meteo = sea_params[indexLat - 1, :] + (sea_params[indexLat, :] - sea_params[indexLat - 1, :]) * (
                    abs(f) - Latitude[indexLat - 1]) / (Latitude[indexLat] - Latitude[indexLat - 1])
    #
    if f >= 0.0:  # northern hemisphere
        doy_min = 28
    else:  # southern latitudes
        doy_min = 211
    param_meteo = ave_meteo - svar_meteo * np.cos((2 * np.pi * (doy - doy_min)) / 365.25)
    pressure, temperature, e, beta, lamda = param_meteo[0], param_meteo[1], param_meteo[2], param_meteo[3], param_meteo[
        4]
    # --------------------
    ave_dry = 1e-6 * k1 * Rd * pressure / gm
    ave_wet = 1e-6 * k2 * Rd / (gm * (lamda + 1) - beta * Rd) * e / temperature
    d_dry = ave_dry * (1 - beta * h / temperature) ** (g / Rd / beta)
    d_wet = ave_wet * (1 - beta * h / temperature) ** (((lamda + 1) * g / Rd / beta) - 1)
    m_elev = 1.001 / np.sqrt(0.002001 + np.sin(np.deg2rad(elevation)) ** 2)
    dtropo = (d_dry + d_wet) * m_elev
    return dtropo



