import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# Styl publikacyjny
sns.set_context("paper", font_scale=1.3)
sns.set_style("ticks")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

def plot_stec_change(
    rx_xy=(0.0, 0.0),
    sat_xy_t0=(-5.0, 6.0),
    sat_xy_t1=(5.0, 6),
    stec_t0=23.4,      # TECU
    stec_t1=27.9,      # TECU
    annotate=True,
    scale=1.1,         # nowy parametr: powiększenie sceny
    show_iono_arc=True # rysuj łuk powłoki jonosferycznej
):
    """
    Wizualizacja zmiany STEC między dwiema epokami:
      - kropka: odbiornik (rx)
      - gwiazdki: satelita w t0 i t1
      - linie: tor sygnału rx→sat dla t0 i t1
      - łuk: symboliczna powłoka jonosferyczna
      - adnotacje: STEC(t0), STEC(t1), ΔSTEC
    """
    rx = np.array(rx_xy, dtype=float)
    s0 = np.array(sat_xy_t0, dtype=float)
    s1 = np.array(sat_xy_t1, dtype=float)
    dSTEC = stec_t1 - stec_t0

    fig, ax = plt.subplots(figsize=(7.5 * scale, 6 * scale))

    # --- odbiornik ---
    ax.scatter(*rx, s=60, marker='o', label='Receiver', zorder=3, color='black')

    # --- satelity ---
    ax.scatter(*s0, s=120, marker='*', label='Satellite @t0', zorder=3, color='royalblue')
    ax.scatter(*s1, s=120, marker='*', label='Satellite @t1', zorder=3, color='deepskyblue')

    # --- linie sygnału ---
    ax.plot([rx[0], s0[0]], [rx[1], s0[1]], linewidth=2, label='Link @ t0', color='royalblue')
    ax.plot([rx[0], s1[0]], [rx[1], s1[1]], linewidth=2, linestyle='--', label='Link @ t1', color='deepskyblue')

    # --- łuk powłoki jonosferycznej ---
    if show_iono_arc:
        R = np.linalg.norm([s0[0]*0.8, s0[1]*0.8]) * 1.1  # promień łuku (symboliczny)
        theta = np.linspace(np.pi/8, np.pi - np.pi/8, 200)
        x_arc = R * np.cos(theta)
        y_arc = R * np.sin(theta)
        ax.plot(x_arc, y_arc, color='lightblue', linewidth=2, linestyle='-', alpha=0.6,
                label='Ionospheric shell')
        ax.text(x_arc[len(x_arc)//2], y_arc[len(y_arc)//2]+0.5,
                'Ionospheric shell (~450 km)', color='steelblue',
                ha='center', fontsize=10)

    # --- adnotacje ---
    if annotate:
        # opisy przy satelitach
        ax.text(s0[0], s0[1], f"  STEC(t0)={stec_t0:.2f} TECU", va='bottom', ha='left', fontsize=11)
        ax.text(s1[0], s1[1], f"  STEC(t1)={stec_t1:.2f} TECU", va='bottom', ha='left', fontsize=11)

        # opis ΔSTEC między torami
        mid = 0.5 * ((rx + s0) + (rx + s1)) / 2.0
        ax.text(mid[0], mid[1], f"ΔSTEC={dSTEC:+.2f} TECU",
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7"),
                fontsize=11)

    # --- styl ---
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Horizontal distance (a.u.)')
    ax.set_ylabel('Height / direction (a.u.)')
    ax.legend(loc='best', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal', adjustable='box')

    # skalowanie osi — zależne od parametru scale
    max_extent = max(np.abs(s0).max(), np.abs(s1).max()) * 1.2
    ax.set_xlim(-max_extent * scale, max_extent * scale)
    ax.set_ylim(-max_extent * 0.1, max_extent * scale)
    ax.set_axis_off()

    plt.figtext(
        0.5, -0.05,
        "Figure 1. Visualization of STEC change between t0 and t1.",
        ha='center', fontsize=11
    )
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# --- Stałe geodezyjne (symbolicznie) ---
R_E_KM = 6371.0      # promień Ziemi [km]
H_KM   = 450.0       # wysokość powłoki (SLM) [km] – można zmienić

def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_km):
    """Odległość po łuku wielkiego koła (haversine) na sferze o promieniu radius_km."""
    phi1, lam1, phi2, lam2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dphi, dlam = phi2 - phi1, lam2 - lam1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return radius_km * c

def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Azymut geodezyjny od punktu 1 → 2 (deg, 0°=N, 90°=E)."""
    phi1, lam1, phi2, lam2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlam = lam2 - lam1
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    th = np.arctan2(y, x)
    return (np.rad2deg(th) + 360) % 360

import numpy as np
import matplotlib.pyplot as plt

R_E_KM = 6371.0
H_KM   = 450.0

def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_km):
    phi1, lam1, phi2, lam2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dphi, dlam = phi2 - phi1, lam2 - lam1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return radius_km * c

def initial_bearing_deg(lat1, lon1, lat2, lon2):
    phi1, lam1, phi2, lam2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlam = lam2 - lam1
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    th = np.arctan2(y, x)
    return (np.rad2deg(th) + 360) % 360

def gc_path(lat1, lon1, lat2, lon2, n=100):
    """Współrzędne łuku wielkiego koła (linia 'po sferze')."""
    # do kartezjańskich (jednostkowa sfera)
    def sph2cart(lat, lon):
        latr, lonr = np.deg2rad(lat), np.deg2rad(lon)
        x = np.cos(latr)*np.cos(lonr)
        y = np.cos(latr)*np.sin(lonr)
        z = np.sin(latr)
        return np.array([x, y, z])
    def cart2sph(v):
        x, y, z = v
        lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
        lon = np.rad2deg(np.arctan2(y, x))
        return lat, lon

    a = sph2cart(lat1, lon1)
    b = sph2cart(lat2, lon2)
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if omega < 1e-9:
        lats = np.linspace(lat1, lat2, n)
        lons = np.linspace(lon1, lon2, n)
        return lats, lons
    t = np.linspace(0, 1, n)
    s1 = np.sin((1-t)*omega) / np.sin(omega)
    s2 = np.sin(t*omega)     / np.sin(omega)
    pts = (a[:,None]*s1 + b[:,None]*s2)
    pts = pts/np.linalg.norm(pts, axis=0, keepdims=True)
    lats, lons = cart2sph(pts)
    # zawijanie lon do [-180,180]
    lons = (lons + 540) % 360 - 180
    return lats, lons

import numpy as np
import matplotlib.pyplot as plt

R_E_KM = 6371.0
H_KM   = 450.0

def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_km):
    phi1, lam1, phi2, lam2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dphi, dlam = phi2 - phi1, lam2 - lam1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return radius_km * c

def initial_bearing_deg(lat1, lon1, lat2, lon2):
    phi1, lam1, phi2, lam2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlam = lam2 - lam1
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    th = np.arctan2(y, x)
    return (np.rad2deg(th) + 360) % 360

def spherical_midpoint(lat1, lon1, lat2, lon2):
    """Midpoint na sferze (SLERP t=0.5), wynik w stopniach, lon w [-180,180]."""
    def sph2cart(lat, lon):
        latr, lonr = np.deg2rad(lat), np.deg2rad(lon)
        x = np.cos(latr)*np.cos(lonr)
        y = np.cos(latr)*np.sin(lonr)
        z = np.sin(latr)
        return np.array([x, y, z])
    def cart2sph(v):
        x, y, z = v
        lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
        lon = (np.rad2deg(np.arctan2(y, x)) + 540) % 360 - 180
        return lat, lon
    a = sph2cart(lat1, lon1); b = sph2cart(lat2, lon2)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if omega < 1e-12:
        return (lat1+lat2)/2, (lon1+lon2)/2
    s1 = np.sin(0.5*omega)/np.sin(omega)
    s2 = s1
    m = a*s1 + b*s2
    m /= np.linalg.norm(m)
    return cart2sph(m)

def gc_path(lat1, lon1, lat2, lon2, n=100):
    """Współrzędne łuku wielkiego koła (linia 'po sferze')."""
    def sph2cart(lat, lon):
        latr, lonr = np.deg2rad(lat), np.deg2rad(lon)
        x = np.cos(latr)*np.cos(lonr)
        y = np.cos(latr)*np.sin(lonr)
        z = np.sin(latr)
        return np.array([x, y, z])
    def cart2sph(v):
        x, y, z = v
        lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
        lon = (np.rad2deg(np.arctan2(y, x)) + 540) % 360 - 180
        return lat, lon

    a = sph2cart(lat1, lon1); b = sph2cart(lat2, lon2)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if omega < 1e-9:
        return np.linspace(lat1, lat2, n), np.linspace(lon1, lon2, n)
    t = np.linspace(0, 1, n)
    s1 = np.sin((1-t)*omega) / np.sin(omega)
    s2 = np.sin(t*omega)     / np.sin(omega)
    pts = a[:,None]*s1 + b[:,None]*s2
    pts = pts/np.linalg.norm(pts, axis=0, keepdims=True)
    lats, lons = cart2sph(pts)
    return lats, lons

def plot_gix_two_points(
        lat_i=52.0, lon_i=16.0,  vtec_i=12.5,
        lat_j=50.5, lon_j=12.0,  vtec_j=10.9,
        iono_height_km=H_KM,
    title: str = "GIX for two IPPs (with CP)",
    draw_gc_arc: bool = True,
    arrow_scale: float = 0.55,
    show_cp_arrow: bool = True,   # mała strzałka gradientu w CP w kierunku azymutu j→i
    annotate: bool = True,
    dpi: int = 120
):
    R_shell = R_E_KM + float(iono_height_km)

    # Geo-geometria pary
    ds_km = great_circle_distance_km(lat_j, lon_j, lat_i, lon_i, R_shell)
    az_deg = initial_bearing_deg(lat_j, lon_j, lat_i, lon_i)
    az_r   = np.deg2rad(az_deg)

    # Gradient
    dVTEC = float(vtec_i) - float(vtec_j)          # [TECU]
    g     = dVTEC / ds_km if ds_km > 0 else np.nan # [TECU/km]
    gx, gy = g*np.sin(az_r), g*np.cos(az_r)

    # CP (midpoint na GC)
    cp_lat, cp_lon = spherical_midpoint(lat_j, lon_j, lat_i, lon_i)

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=dpi)

    # Łuk GC lub prosta
    if draw_gc_arc:
        lat_gc, lon_gc = gc_path(lat_j, lon_j, lat_i, lon_i, n=200)
        ax.plot(lon_gc, lat_gc, '-', color='tab:blue', lw=2, label='Pair (j→i, great-circle)')
    else:
        ax.plot([lon_j, lon_i], [lat_j, lat_i], '-', color='tab:blue', lw=2, label='Pair (j→i)')

    # Punkty i CP
    ax.scatter([lon_j], [lat_j], s=80, c='tab:orange', marker='o', zorder=3, label='IPP j')
    ax.scatter([lon_i], [lat_i], s=90, c='tab:green',  marker='^', zorder=3, label='IPP i')
    ax.scatter([cp_lon], [cp_lat], s=70, c='mediumpurple', marker='s', zorder=4, label=r'$CP_{i,j}$')

    # Strzałka azymutu przy j
    base = max(0.5, min(2.0, ds_km/400.0)) * arrow_scale
    dlon = base * np.sin(az_r) / np.cos(np.deg2rad(lat_j))
    dlat = base * np.cos(az_r)
    ax.arrow(lon_j, lat_j, dlon, dlat,
             width=0.01, head_width=0.1, head_length=0.2,
             color='tab:red', alpha=0.95, length_includes_head=True, zorder=4,
             label='Azimuth j→i')

    # Mała strzałka w CP w kierunku azymutu (miejsce „przypisania” gradientu)
    if show_cp_arrow:
        base_cp = base * 0.6
        dlon_cp = base_cp * np.sin(az_r) / np.cos(np.deg2rad(cp_lat))
        dlat_cp = base_cp * np.cos(az_r)
        ax.arrow(cp_lon, cp_lat, dlon_cp, dlat_cp,
                 width=0.008, head_width=0.1, head_length=0.2,
                 color='mediumpurple', alpha=0.9, length_includes_head=True, zorder=5)

    # Adnotacje
    if annotate:
        ax.text(lon_j, lat_j, f"  j\nVTEC={vtec_j:.2f} TECU",
                va='top', ha='left', fontsize=10, color='tab:orange')
        ax.text(lon_i, lat_i, f"  i\nVTEC={vtec_i:.2f} TECU",
                va='bottom', ha='left', fontsize=10, color='tab:green')
        ax.text(cp_lon+0.7, cp_lat-0.5,
                r"$CP_{i,j}$"+"\n"
                "Δs = {:.1f} km\n"
                "lat: {:.2f} lon: {:.2f}".format(ds_km,cp_lat, cp_lon),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6'),
                ha='left', va='bottom', fontsize=10, color='mediumpurple')

        # dodatkowy box z pełnym zestawem
        mid_lon, mid_lat = (lon_i+lon_j)/2, (lat_i+lat_j)/2
        # przesunięcie prostopadle do kierunku
        n_dlon = -np.cos(az_r) / max(1e-6, np.cos(np.deg2rad(mid_lat)))
        n_dlat =  np.sin(az_r)
        n = np.hypot(n_dlon, n_dlat); n_dlon /= n; n_dlat /= n
        box_lon = mid_lon - 2*n_dlon
        box_lat = mid_lat - 2*n_dlat
        ax.text(box_lon, box_lat,
                "az(j→i) = {:.1f}°\nΔVTEC = {:+.2f} TECU\n"
                "gₓ = {:+.4f}, gᵧ = {:+.4f} TECU/km".format(az_deg, dVTEC, gx, gy),
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.6'),
                ha='center', va='center', fontsize=10)

    # Estetyka
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    # ax.set_title(f"{title}\nShell height H={iono_height_km:.0f} km,  R_shell={R_shell:.0f} km")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', frameon=True)
    pad_lon = max(1.0, 0.12*abs(lon_i - lon_j) + 1.0)
    pad_lat = max(1.0, 0.12*abs(lat_i - lat_j) + 1.0)
    ax.set_xlim(min(lon_i, lon_j) - pad_lon, max(lon_i, lon_j) + pad_lon)
    ax.set_ylim(min(lat_i, lat_j) - pad_lat, max(lat_i, lat_j) + pad_lat)
    plt.tight_layout()
    plt.figtext(
        0.5, -0.05,
        "Figure 2. Visualization of TEC gradient between pair of IPPs.",
        ha='center', fontsize=11
    )
    plt.show()
