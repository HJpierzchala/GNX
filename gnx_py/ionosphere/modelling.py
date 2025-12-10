from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable, Optional, Dict, Any, Tuple, Literal,Iterable
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree, distance
from .kriging import OrdinaryKrigingGeo, UniversalKrigingGeo


KrigingMode = Literal["OK", "UK", "WAAS"]
CoordFrame  = Literal["GEO", "SM"]
EARTH_RADIUS_KM = 6371.0
IONO_SHELL_H_KM = 450.0  # WAAS: 350 km

def load_stec_folder(
    folder: os.PathLike | str,
    *,
    file_suffix: str = "STEC.parquet.gzip",
    min_elev_deg: float = 10.0,
    station_name_len: int = 6,
    unique_id_len: int = 4,
    network_filter: Optional[Iterable[str]] = None,
    required_columns: Tuple[str, str, str, str] = ("lat_ipp", "lon_ipp", "leveled_tec", "ev","sv","time"),
    quiet: bool = False,
    skip_negative:bool=True
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Load and validate STEC parquet files from a folder, filtering by elevation
    and rejecting files that contain negative leveled TEC values.

    The function iterates over files ending with ``file_suffix`` inside ``folder``.
    For each file:
      1) Extracts a station name (first ``station_name_len`` characters of the filename)
         and a short unique ID (first ``unique_id_len`` characters).
      2) Reads the parquet into a DataFrame.
      3) Keeps only rows with elevation ``ev > min_elev_deg`` and only the columns
         specified by ``required_columns``.
      4) Rejects the entire file if any ``leveled_tec`` is negative.
      5) Appends an extra column ``name`` with the station name and collects results.

    Parameters
    ----------
    folder : os.PathLike | str
        Path to the directory containing STEC parquet files.
    file_suffix : str, optional
        File name suffix to match (default: ``"STEC.parquet.gzip"``).
    min_elev_deg : float, optional
        Minimum elevation threshold in degrees; rows with ``ev <= min_elev_deg`` are dropped.
    station_name_len : int, optional
        Number of leading characters used as the station name (default: 6).
        Assumes filenames begin with a station identifier (e.g., ``"ABCD01…"``).
    unique_id_len : int, optional
        Number of leading characters used as the short unique station ID (default: 4).
    network_filter : Iterable[str] | None, optional
        If provided, keep only files whose short unique ID is in this set/list.
        (Comparison is case-sensitive.)
    required_columns : tuple[str, str, str, str], optional
        Columns expected in each parquet file in order: (lat, lon, tec, elevation).
        Default: ``("lat_ipp", "lon_ipp", "leveled_tec", "ev")``.
    quiet : bool, optional
        If True, suppress per-file log messages.

    Returns
    -------
    df_all : pandas.DataFrame
        Concatenated DataFrame of accepted files with columns:
        ``lat_ipp, lon_ipp, leveled_tec, ev, name``.
        If no files are accepted, an empty DataFrame with those columns is returned.
    summary : dict
        A dictionary with processing metadata:
        - ``files_scanned`` (int): total files that matched the suffix
        - ``files_accepted`` (int): files included in the final DataFrame
        - ``files_rejected`` (int): files rejected (negative TEC or errors)
        - ``stations_unique`` (List[str]): unique short IDs encountered (accepted or rejected)
        - ``stations_accepted`` (List[str]): unique short IDs among accepted files
        - ``stations_rejected`` (List[str]): unique short IDs among rejected files
        - ``rejected_files`` (List[str]): basenames of rejected files
        - ``errors`` (List[Tuple[str, str]]): (filename, error message) pairs for exceptions

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist or is not a directory.

    Notes
    -----
    * Behavior mirrors the reference snippet: any file with *any* negative
      ``leveled_tec`` is fully skipped.
    * Unlike the original code, this function **does not** divide counters by two.
      Instead, it reports counts directly and also lists unique short IDs so you
      can infer pairing at the station level without guessing.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    lat_col, lon_col, tec_col, ev_col, sv_col, time_col = required_columns
    dfs: List[pd.DataFrame] = []

    files = sorted(p for p in folder.iterdir() if p.name.endswith(file_suffix))
    files_scanned = len(files)

    stations_unique: set[str] = set()
    stations_accepted: set[str] = set()
    stations_rejected: set[str] = set()

    rejected_files: List[str] = []
    errors: List[Tuple[str, str]] = []

    for fpath in files:
        try:
            # Names as in your original logic
            name = fpath.name[:station_name_len]
            unique_id = name[:unique_id_len]
            stations_unique.add(unique_id)

            if network_filter is not None and unique_id not in network_filter:
                if not quiet:
                    print(f"[SKIP-NET] {fpath.name} (unique_id {unique_id} not in network_filter)")
                continue

            df = pd.read_parquet(fpath)

            # Validate required columns early
            missing = [c for c in (lat_col, lon_col, tec_col, ev_col) if c not in df.columns]
            if missing:
                raise KeyError(f"Missing required columns {missing} in {fpath.name}")

            # Elevation filtering and column selection
            df = df[df[ev_col] > float(min_elev_deg)]#[[lat_col, lon_col, tec_col, ev_col]]

            # Reject entire file if any negative TEC
            if skip_negative:
                if (df[tec_col] < 0).any():
                    if not quiet:
                        print(f"[REJECT-NEG] {fpath.name} — negative TEC detected")
                    rejected_files.append(fpath.name)
                    stations_rejected.add(unique_id)
                    continue

            if df.empty:
                if not quiet:
                    print(f"[SKIP-EMPTY] {fpath.name} — no rows after filtering")
                # Treat as rejected but without negative TEC
                rejected_files.append(fpath.name)
                stations_rejected.add(unique_id)
                continue
            df[tec_col]/=1e16
            # Add station name column and collect
            df = df.copy()
            df["name"] = name

            dfs.append(df)
            stations_accepted.add(unique_id)
            if not quiet:
                print(f"[ACCEPT] {fpath.name} → rows: {len(df)}")

        except Exception as e:
            errors.append((fpath.name, str(e)))
            rejected_files.append(fpath.name)
            stations_rejected.add(fpath.name[:unique_id_len])
            if not quiet:
                print(f"[ERROR] {fpath.name}: {e}")

    # Build final DataFrame
    if dfs:
        df_all = pd.concat(dfs, axis=0)
        #df_all = df_all[[lat_col, lon_col, tec_col, ev_col, "name"]]
        df_all = df_all.reset_index()
        df_all['time'] = pd.to_datetime(df_all['time'])
        df_all.set_index(['sv', 'time'], inplace=True)
    # else:
    #     df_all = pd.DataFrame(columns=[lat_col, lon_col, tec_col, ev_col, "name"])

    summary = {
        "files_scanned": files_scanned,
        "files_accepted": len(dfs),
        "files_rejected": len(rejected_files),
        "stations_unique": sorted(stations_unique),
        "stations_accepted": sorted(stations_accepted),
        "stations_rejected": sorted(stations_rejected),
        "rejected_files": rejected_files,
        "errors": errors,
    }
    return df_all, summary


# ------------------------------------------
# Variogram/Covariance (Sparks/WAAS model)
# ------------------------------------------
@runtime_checkable
class VariogramLike(Protocol):
    """Minimalny interfejs wariogramu na potrzeby krigingu:
    - cov_func(d_km): kowariancja C(d) rezyduów [VTEC^2] dla odległości w [km]
    - sill_var: wariancja 'sill' = sigma_total^2 (float)
    """
    def cov_func(self, d_km: np.ndarray) -> np.ndarray: ...
    @property
    def sill_var(self) -> float: ...

# =========================
# 2) Sparks (jak u Ciebie)
# =========================
@dataclass
class SparksVariogram:
    """Sparks/WAAS (2011):
    C(d) = sigma_total^2                         , d=0
         = (sigma_total^2 - sigma_nominal^2) * exp(-d/ddecorr) , d>0
    Jednostki: VTEC->TECU, kowariancje -> (TECU)^2.
    """
    sigma_nominal: float = 0.5    # TECU
    sigma_total: float = 1.5      # TECU
    ddecorr_km: float = 800.0     # km

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total ** 2
        cN = self.sigma_nominal ** 2
        out = (c0 - cN) * np.exp(-np.maximum(d_km, 0.0) / max(self.ddecorr_km, 1e-6))
        out = np.where(d_km <= 1e-12, c0, out)
        return out

    @property
    def sill_var(self) -> float:
        return self.sigma_total ** 2


# =========================
# 3) Inne modele wariogramów
# =========================
@dataclass
class SphericalVariogram:
    """Model sferyczny (klasyka geostatystyki).
    Parametry:
      - sigma_nominal: √nugget (TECU)
      - sigma_total:   √sill   (TECU)
      - range_km:      zasięg a [km], po którym C(d>=a)=0
    Kowariancja rezyduów:
      C(0)=sill; dla 0<d<a: (sill-nugget)*(1 - 1.5 h + 0.5 h^3), h=d/a; dla d>=a: 0
    """
    sigma_nominal: float = 0.5
    sigma_total: float = 1.5
    range_km: float = 1200.0

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total**2
        cN = self.sigma_nominal**2
        a = max(self.range_km, 1e-6)
        h = np.clip(np.asarray(d_km, float) / a, 0.0, np.inf)
        core = (1.0 - 1.5*h + 0.5*h**3)
        core = np.where(h < 1.0, core, 0.0)
        out = (c0 - cN) * core
        out = np.where(h <= 1e-12, c0, out)
        return out

    @property
    def sill_var(self) -> float:
        return self.sigma_total**2

@dataclass
class ExponentialVariogram:
    """Model wykładniczy:
       C(d) = (sill - nugget) * exp(-d / L), C(0)=sill
    gdzie L ~ długość skali/dekoherencji [km].
    """
    sigma_nominal: float = 0.5
    sigma_total: float = 1.5
    L_km: float = 800.0

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total**2
        cN = self.sigma_nominal**2
        L = max(self.L_km, 1e-6)
        out = (c0 - cN) * np.exp(-np.maximum(d_km, 0.0)/L)
        out = np.where(d_km <= 1e-12, c0, out)
        return out

    @property
    def sill_var(self) -> float:
        return self.sigma_total**2

@dataclass
class GaussianVariogram:
    """Model gaussowski:
       C(d) = (sill - nugget) * exp(-(d/L)^2), C(0)=sill
    Bardziej „gładki” przy d~0 niż wykładniczy.
    """
    sigma_nominal: float = 0.5
    sigma_total: float = 1.5
    L_km: float = 900.0

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total**2
        cN = self.sigma_nominal**2
        L2 = max(self.L_km, 1e-6)**2
        out = (c0 - cN) * np.exp(- (np.maximum(d_km, 0.0)**2) / L2)
        out = np.where(d_km <= 1e-12, c0, out)
        return out

    @property
    def sill_var(self) -> float:
        return self.sigma_total**2

# =========================
# 4) Uniwersalny wrapper (opcjonalnie)
# =========================
@dataclass
class GenericVariogram:
    """Uniwersalny wrapper: model ∈ {'spherical','exponential','gaussian','sparks'}.
    Przydatny gdy chcesz wybierać modelem po nazwie (np. z pliku YAML/JSON).
    """
    model: Literal["spherical", "exponential", "gaussian", "sparks"] = "exponential"
    params: Dict[str, float] = field(default_factory=dict)

    # lazy konstrukcja konkretnego modelu:
    def _impl(self) -> VariogramLike:
        m = self.model.lower()
        if m == "spherical":
            return SphericalVariogram(
                sigma_nominal=float(self.params.get("sigma_nominal", 0.5)),
                sigma_total=float(self.params.get("sigma_total", 1.5)),
                range_km=float(self.params.get("range_km", 1200.0)),
            )
        if m == "gaussian":
            return GaussianVariogram(
                sigma_nominal=float(self.params.get("sigma_nominal", 0.5)),
                sigma_total=float(self.params.get("sigma_total", 1.5)),
                L_km=float(self.params.get("L_km", 900.0)),
            )
        if m == "sparks":
            return SparksVariogram(
                sigma_nominal=float(self.params.get("sigma_nominal", 0.5)),
                sigma_total=float(self.params.get("sigma_total", 1.5)),
                ddecorr_km=float(self.params.get("ddecorr_km", 800.0)),
            )
        # default: exponential
        return ExponentialVariogram(
            sigma_nominal=float(self.params.get("sigma_nominal", 0.5)),
            sigma_total=float(self.params.get("sigma_total", 1.5)),
            L_km=float(self.params.get("L_km", 800.0)),
        )

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        return self._impl().cov_func(d_km)

    @property
    def sill_var(self) -> float:
        return self._impl().sill_var

assert isinstance(GenericVariogram(), VariogramLike)


def llh_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_km: float = IONO_SHELL_H_KM) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    R = EARTH_RADIUS_KM + h_km
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.stack([x, y, z], axis=-1)   # <— BYŁO axis=1, MA BYĆ axis=-1



def enu_basis_at(lat0_deg: float, lon0_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zwraca wektory jednostkowe (up, east, north) w ECEF (zgodnie z Sparks 2011 eq. 7)."""

    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    eup = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    ee = np.array([-np.sin(lon0), np.cos(lon0), 0.0])  # east
    en = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])  # north
    return eup, ee, en


def ecef_distance_km(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Odległość euklidesowa w ECEF w km. P: (...,3), Q: (...,3)."""
    return np.linalg.norm(P - Q, axis=-1)

def build_G(dx_e: np.ndarray, dx_n: np.ndarray) -> np.ndarray:
    """Zwraca macierz G (N x 3): [1, dx·east, dx·north]."""
    ones = np.ones_like(dx_e)
    return np.stack([ones, dx_e, dx_n], axis=1)

def waas_kriging_point_fast(
    ipp_xyz: np.ndarray,          # (N,3) IPP ECEF
    ipp_vals: np.ndarray,         # (N,)
    igp_xyz: np.ndarray,          # (3,)
    igp_latlon: Tuple[float, float],
    vario: VariogramLike,
    C_full: np.ndarray,           # (N,N) = vario.cov_func(D), precomputed for epoch
    M_full: Optional[np.ndarray] = None,  # (N,N) pomiarowe/biasy, precomputed for epoch (może być None)
    idx: Optional[np.ndarray] = None,     # sąsiedzi użyci do estymacji (1D indeksy)
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Szybki solver: używa tylko podmacierzy A11 = (C+M)[idx][:,idx] i Schura.
    Zwraca (I_hat, var_hat, info). var_hat korzysta z formuły wariancji krigingu
    przez mnożniki Lagrange’a: σ^2 = sill - (c^T w + s^T λ).
    """
    if idx is None:
        idx = np.arange(ipp_xyz.shape[0], dtype=np.int64)
    N = idx.size
    if N < 3:
        return np.nan, np.nan, {"used": int(N)}

    # ENU w IGP i G

    eup, ee, en = enu_basis_at(*igp_latlon)
    dxyz = ipp_xyz[idx] - igp_xyz[None, :]
    dx_e = dxyz @ ee
    dx_n = dxyz @ en
    G = build_G(dx_e, dx_n)             # (N,3)
    s = np.array([1.0, 0.0, 0.0])       # (3,)

    # A11 = C + M (podmacierz na idx)
    A11 = C_full[np.ix_(idx, idx)].copy()
    if M_full is not None:
        A11 += M_full[np.ix_(idx, idx)]

    # wektor kowariancji c (N,)
    d_igp = ecef_distance_km(ipp_xyz[idx], igp_xyz[None, :])
    c = vario.cov_func(d_igp)

    # Cholesky A11
    try:
        cf = cho_factor(A11, overwrite_a=False, check_finite=False)
    except Exception:
        # minimalna regularizacja, gdyby A11 było osobliwe
        eps = 1e-8 * (np.trace(A11)/max(N,1))
        cf = cho_factor(A11 + eps*np.eye(N), overwrite_a=False, check_finite=False)

    # Solve’y przez A11^{-1} (używamy cho_solve, bez jawnego odwracania)
    y_c = cho_solve(cf, c, check_finite=False)        # A11^{-1} c     (N,)
    Y_G = cho_solve(cf, G, check_finite=False)        # A11^{-1} G     (N,3)

    # Schur: (G^T A11^{-1} G) λ = (G^T A11^{-1} c) - s
    S = G.T @ Y_G                                     # (3,3)
    rhs = G.T @ y_c - s                               # (3,)
    try:
        lam = np.linalg.solve(S, rhs)                 # λ
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(S, rhs, rcond=None)[0]

    # w = A11^{-1}(c - G λ) = y_c - Y_G λ
    w = y_c - (Y_G @ lam)

    # Estymata i wariancja
    I_hat = float(w @ ipp_vals[idx])
    # σ^2 = sill - (c^T w + s^T λ)  (kriging variance z mnożnikami Lagrange’a)
    var_hat = float(vario.sill_var - (c @ w + s @ lam))
    if not np.isfinite(var_hat):
        # fallback kompatybilny z Twoją starą formułą
        C = C_full[np.ix_(idx, idx)]
        var_hat = float(w @ (C @ w) - 2.0*(w @ c) + vario.sill_var)
    var_hat = max(var_hat, 0.0)

    info = {"used": int(N), "condS": float(np.linalg.cond(S))}
    return I_hat, var_hat, info


def _sm_ll_to_ecef_km(tr, lat_deg, lon_deg, r_norm_km, obstime):
    """SM(lat,lon,r_norm) -> ECEF w [km]. lat/lon deg; r_norm_km skalar lub 2D."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    r_m = np.asarray(r_norm_km, float) * 1e3
    xyz_m = tr.sm_ll_to_ecef(lat_rad, lon_rad, r_norm=r_m, when=obstime)  # (...,3) [m]
    return np.asarray(xyz_m, float) / 1e3  # [km]


def _default_rnorm_km(shape, base=EARTH_RADIUS_KM, h=IONO_SHELL_H_KM):
    """Stały promień geocentryczny (sfera) w [km]."""
    return np.full(shape, base + h, dtype=float)

@dataclass
class IonoKrigingMonitor:
    # siatka, na której chcemy mapy (deg) – może być GEO lub SM
    grid_lon: np.ndarray
    grid_lat: np.ndarray

    # w jakim układzie podaję grid oraz IPP (lat/lon w deg)?
    coord_frame: CoordFrame = "GEO"           # "GEO" (geodetyczne) lub "SM"

    # kontekst SM (wymagany jeśli coord_frame="SM")
    sm_transformer: Optional[Any] = None      # np. SolarGeomagneticTransformer
    sm_obstime:     Optional[Any] = None      # astropy.Time (skalar)
    grid_rnorm_km:  Optional[np.ndarray] = None  # (Ny,Nx) promień geocentryczny; jeśli None -> sfera

    # tryb krigingu
    kriging_mode: KrigingMode = "WAAS"

    # OK/UK fallback
    variogram_model: Literal["spherical","exponential","gaussian","linear","power","hole-effect"] = "spherical"
    variogram_parameters: Optional[Dict[str, float]] = None
    n_closest_points: int = 60
    backend: Literal["C", "loop", "vectorized"] = "C"
    use_external_drift: bool = False

    # WAAS/Sparks: parametry selekcji sąsiedztwa
    Rmin_km: float = 800.0
    Rmax_km: float = 2100.0
    Ntarget: int = 30
    Nmin: int = 10

    # WAAS/Sparks: model variogramu
    variogram: VariogramLike = field(default_factory=SparksVariogram)

    # WAAS/Sparks: wariancje szumów (opcjonalnie w df)
    default_meas_var: float = 0.0
    bias_var_rec: float = 0.0
    bias_var_sat: float = 0.0

    # opcjonalne nadpisania (użyteczne gdy chcesz sam policzyć ECEF poza klasą)
    grid_xyz_override: Optional[np.ndarray] = None   # (Ny,Nx,3) w km
    ipp_xyz_override:  Optional[np.ndarray] = None   # (N,3)    w km

    def __repr__(self):
        args = ", ".join(f"{k}={v}\n" for k, v in self.__dict__.items())
        return f"IonoKrigingMonitor: ({args})"

    def __post_init__(self):
        self.grid_lon = np.asarray(self.grid_lon, float)
        self.grid_lat = np.asarray(self.grid_lat, float)

        # 1D -> 2D albo 2D -> jak jest
        if self.grid_lon.ndim == 2 and self.grid_lat.ndim == 2:
            Lon, Lat = self.grid_lon, self.grid_lat
        elif self.grid_lon.ndim == 1 and self.grid_lat.ndim == 1:
            Lat, Lon = np.meshgrid(self.grid_lat, self.grid_lon, indexing='ij')
        else:
            raise ValueError("grid_lat/grid_lon muszą być oba 1D albo oba 2D o zgodnych kształtach.")

        self._grid_lat = Lat
        self._grid_lon = Lon
        self._grid_shape = Lat.shape  # (Ny, Nx)

        # --- gdzie pracujemy: GEO czy SM? ---
        if self.grid_xyz_override is not None:
            # dostaliśmy gotowe ECEF [km]
            if self.grid_xyz_override.shape != (Lat.shape[0], Lat.shape[1], 3):
                raise ValueError("grid_xyz_override musi mieć kształt (Ny,Nx,3).")
            self._grid_xyz = np.asarray(self.grid_xyz_override, float)
        elif self.coord_frame == "GEO":
            # bezpiecznie: sfera (jak dotąd)
            self._grid_xyz = llh_to_ecef(Lat, Lon)     # [km]
        elif self.coord_frame == "SM":
            if (self.sm_transformer is None) or (self.sm_obstime is None):
                raise ValueError("Dla coord_frame='SM' podaj sm_transformer i sm_obstime.")
            # promień geocentryczny dla siatki
            rnorm = self.grid_rnorm_km
            if rnorm is None:
                rnorm = _default_rnorm_km(Lat.shape)
            else:
                rnorm = np.asarray(rnorm, float)
                if rnorm.shape != Lat.shape:
                    raise ValueError("grid_rnorm_km musi mieć kształt (Ny,Nx).")
            # SM(lat,lon,r_norm) -> ECEF [km]
            self._grid_xyz = _sm_ll_to_ecef_km(self.sm_transformer, Lat, Lon, rnorm, self.sm_obstime)
        else:
            raise ValueError("coord_frame musi być 'GEO' albo 'SM'.")
        if not isinstance(self.variogram, VariogramLike):
            raise TypeError("variogram musi implementować VariogramLike (cov_func & sill_var).")

    # -------------- PYKRIGE (bez zmian) --------------
    def _prepare_variogram_pykrige(self) -> dict[str, float] | None:
        if self.variogram_parameters is not None:
            return self.variogram_parameters
        return None

    def _build_kriger(self, lon, lat, vtec, drift_data=None):
        params = self._prepare_variogram_pykrige()

        if self.kriging_mode == "OK":
            return OrdinaryKrigingGeo(
                lon, lat, vtec,
                variogram_model=self.variogram_model,
                variogram_parameters=([params["sill"], params["range"], params["nugget"]]),
            )
        elif self.kriging_mode == "UK":
            return UniversalKrigingGeo(
                    lon,
                    lat,
                    vtec,
                    variogram_model=self.variogram_model,
                    variogram_parameters=([params["sill"], params["range"], params["nugget"]]),   # dokładnie te same,
                    coordinates_type="euclidean",
                    drift_terms=["regional_linear"],
                )


    # -------------- WAAS/Sparks: sąsiedztwo --------------
    def _neighbors_for_igp(self, igp_xyz: np.ndarray, ipp_xyz: np.ndarray, Nmin: int, Ntarget: int) -> np.ndarray:
        d = ecef_distance_km(ipp_xyz, igp_xyz[None, :])
        in_Rmin = np.where(d <= self.Rmin_km)[0]
        if in_Rmin.size >= Nmin:
            return in_Rmin
        order = np.argsort(d)
        if order.size >= Ntarget:
            Rfit = min(d[order[Ntarget - 1]], self.Rmax_km)
        else:
            Rfit = self.Rmax_km
        within = np.where(d <= Rfit)[0]
        if within.size >= Nmin:
            return within
        return within

    # -------------- KRIGING EPOKI --------------
    def krige_epoch(
        self,
        df_epoch: pd.DataFrame,
        epoch_time: Optional[pd.Timestamp] = None,
        ipp_xyz_override: Optional[np.ndarray] = None,   # (N,3) w km
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

        dfe = df_epoch.dropna(subset=["lat_ipp", "lon_ipp", "vtec"]).copy()
        lon = dfe["lon_ipp"].to_numpy(float)
        lat = dfe["lat_ipp"].to_numpy(float)
        vtec = dfe["vtec"].to_numpy(float)

        Ny, Nx = self._grid_shape
        Vgrid = np.full((Ny, Nx), np.nan, float)
        SS    = np.full((Ny, Nx), np.nan, float)

        if len(vtec) == 0:
            return (Vgrid, SS, {"n_points": 0}, None)

        # --- IPP -> ECEF [km] ---
        if ipp_xyz_override is not None:
            ipp_xyz = np.asarray(ipp_xyz_override, float)
            if ipp_xyz.ndim != 2 or ipp_xyz.shape[1] != 3:
                raise ValueError("ipp_xyz_override musi mieć kształt (N,3) w km.")
        elif self.ipp_xyz_override is not None:
            ipp_xyz = np.asarray(self.ipp_xyz_override, float)
            if ipp_xyz.ndim != 2 or ipp_xyz.shape[1] != 3:
                raise ValueError("ipp_xyz_override musi mieć kształt (N,3) w km.")
        else:
            if self.coord_frame == "GEO":
                ipp_xyz = llh_to_ecef(lat, lon, IONO_SHELL_H_KM)
            elif self.coord_frame == "SM":
                if (self.sm_transformer is None) or (self.sm_obstime is None):
                    raise ValueError("Dla coord_frame='SM' potrzebne sm_transformer oraz sm_obstime.")
                # r_norm IPP: jeśli kolumna 'r_norm_km' jest w df -> użyj; inaczej sfera
                if "r_norm_km" in dfe.columns:
                    rnorm_km = dfe["r_norm_km"].to_numpy(float)
                else:
                    rnorm_km = EARTH_RADIUS_KM + IONO_SHELL_H_KM
                ipp_xyz = _sm_ll_to_ecef_km(self.sm_transformer, lat, lon, rnorm_km, self.sm_obstime)
            else:
                raise ValueError("coord_frame musi być 'GEO' albo 'SM'.")

        meas_var = dfe["meas_var"].to_numpy(float) if "meas_var" in dfe.columns else None
        rec_id = dfe["rec_id"].to_numpy(object) if "rec_id" in dfe.columns else None
        sat_id = dfe["sat_id"].to_numpy(object) if "sat_id" in dfe.columns else None
        if meas_var is None and self.default_meas_var > 0:
            meas_var = np.full(len(vtec), float(self.default_meas_var), float)

        if self.kriging_mode in ["OK","UK"]:
            # --- Ordinary Kriging z PyKrige, współrzędne geograficzne ---
            kriger = self._build_kriger(lon, lat, vtec)

            # Grid z klasy: 2D -> wymagane 1D wektory dla 'grid' w PyKrige
            gx = self._grid_lon[0, :].astype(float)  # Nx
            gy = self._grid_lat[:, 0].astype(float)  # Ny

            # PyKrige wymaga rosnących wektorów — posortuj jeśli trzeba i zapamiętaj kolejność
            sort_x = np.argsort(gx)
            sort_y = np.argsort(gy)
            gx_s = gx[sort_x]
            gy_s = gy[sort_y]

            z_s, ss_s = kriger.execute(
                "grid",
                gx_s,  # 1D longitudes
                gy_s,  # 1D latitudes
            )
            # z_s, ss_s mają shape (len(gy_s), len(gx_s)) == (Ny, Nx)

            # Odwróć sortowania do oryginalnego ułożenia siatki
            inv_x = np.argsort(sort_x)
            inv_y = np.argsort(sort_y)
            Vgrid = np.asarray(z_s)[inv_y][:, inv_x].astype(float)
            SS = np.asarray(ss_s)[inv_y][:, inv_x].astype(float)

            # meta
            par = self._prepare_variogram_pykrige()
            info = {
                "mode": self.kriging_mode,
                "coord_frame": self.coord_frame,
                "n_points": int(len(vtec)),
                "variogram_model": self.variogram_model,
                "variogram_parameters": {
                    "sill": float(getattr(par,"sill",9999)),
                    "range": float(getattr(par,"range",9999)),
                    "nugget": float(getattr(par,"nugget",9999)),
                },
                "Rmin_km": float(self.Rmin_km),
                "Rmax_km": float(self.Rmax_km),
                "Ntarget": int(self.Ntarget),
                "Nmin": int(self.Nmin),
            }
            # zapis stanu czasowego (zgodnie z Twoją logiką ROT)

            return Vgrid, SS, info
        elif self.kriging_mode == "WAAS":
            # PRECOMPUTE
            D = distance.cdist(ipp_xyz, ipp_xyz, metric="euclidean")  # [km]
            C_full = self.variogram.cov_func(D)
            M_full = None
            if (meas_var is not None) or (self.bias_var_rec > 0) or (self.bias_var_sat > 0):
                M_full = np.zeros_like(C_full)
                if meas_var is not None:
                    np.fill_diagonal(M_full, meas_var.astype(float))
                if (rec_id is not None) and (self.bias_var_rec > 0):
                    same_r = rec_id.reshape(-1,1) == rec_id.reshape(1,-1)
                    M_full += np.where(same_r, float(self.bias_var_rec), 0.0)
                    if meas_var is not None:
                        np.fill_diagonal(M_full, meas_var.astype(float))
                if (sat_id is not None) and (self.bias_var_sat > 0):
                    same_s = sat_id.reshape(-1,1) == sat_id.reshape(1,-1)
                    M_full += np.where(same_s, float(self.bias_var_sat), 0.0)
                    if meas_var is not None:
                        np.fill_diagonal(M_full, meas_var.astype(float))

            tree = cKDTree(ipp_xyz)
            Kcand = max(self.Ntarget * 2, self.Nmin + 8)

            for iy in range(Ny):
                for ix in range(Nx):
                    igp_xyz  = self._grid_xyz[iy, ix]            # (3,) km
                    igp_lat  = self._grid_lat[iy, ix]
                    igp_lon  = self._grid_lon[iy, ix]
                    igp_latlon = (igp_lat, igp_lon)

                    dist, cand = tree.query(igp_xyz, k=min(Kcand, ipp_xyz.shape[0]))
                    cand = np.atleast_1d(cand); dist = np.atleast_1d(dist)

                    sel = cand[dist <= self.Rmax_km]
                    if sel.size < self.Nmin:
                        sel = cand[:max(self.Nmin, min(self.Ntarget, cand.size))]
                    if sel.size < self.Nmin:
                        continue

                    I_hat, var_hat, _info = waas_kriging_point_fast(
                        ipp_xyz, vtec, igp_xyz, igp_latlon, self.variogram,
                        C_full=C_full, M_full=M_full, idx=sel
                    )
                    Vgrid[iy, ix] = I_hat
                    SS[iy, ix] = var_hat
        else:
            raise ValueError(f"Nieznany tryb krigingu: {self.kriging_mode}")




        info = {
            "mode": self.kriging_mode,
            "coord_frame": self.coord_frame,
            "n_points": int(len(vtec)),
            "sparks_vario": {
                "sigma_nominal": float(getattr(self.variogram,'sigma_nominal',0.0)),
                "sigma_total": float(getattr(self.variogram,'sigma_total',0.0)),
                "ddecorr_km": float(getattr(self.variogram,'ddecorr_km',0.0)),
            },
            "Rmin_km": float(self.Rmin_km),
            "Rmax_km": float(self.Rmax_km),
            "Ntarget": int(self.Ntarget),
            "Nmin": int(self.Nmin),
        }
        return Vgrid, SS, info


import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Optional, Union, Sequence

def save_grids_to_netcdf(
    times: Sequence[Union[np.datetime64, pd.Timestamp, str]],
    V_all: Union[List[np.ndarray], np.ndarray],
    SS_All: Union[List[np.ndarray], np.ndarray],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    out_path: str = "grids.nc",
    out_name:str="vtec",
    var_names=("V", "SS"),
    chunks: Optional[dict] = None,
    compress_level: int = 4,
) -> xr.Dataset:
    """
    Zapisuje pola na siatce do NetCDF:
      - rectilinear: lat(Ny,), lon(Nx,), dane (T, Ny, Nx)
      - curvilinear: lat(Ny,Nx), lon(Ny,Nx), dane (T, Ny, Nx)
    V_all/SS_All mogą być listami 2D lub tablicami 2D/3D.
    """

    # ---------- przygotowanie czasu ----------
    # Rzut do pandas.DatetimeIndex → numpy datetime64[ns]
    time_index = pd.to_datetime(times)
    if time_index.isna().any():
        raise ValueError("W 'times' wykryto niepoprawne wartości daty/czasu.")
    time_vals = time_index.to_numpy(dtype="datetime64[ns]")
    T = time_vals.shape[0]

    # ---------- siatka ----------
    # --- detekcja siatki (jak u Ciebie) ---
    lat_grid = np.asarray(lat_grid)
    lon_grid = np.asarray(lon_grid)
    if lat_grid.ndim == 1 and lon_grid.ndim == 1:
        nlat, nlon = lat_grid.size, lon_grid.size
        grid_kind = "rect"
    elif lat_grid.ndim == 2 and lon_grid.ndim == 2:
        nlat, nlon = lat_grid.shape
        if lon_grid.shape != (nlat, nlon):
            raise ValueError(f"lon_grid.shape {lon_grid.shape} != lat_grid.shape {(nlat, nlon)}")
        grid_kind = "curvi"
    else:
        raise ValueError("lat_grid/lon_grid muszą być oba 1D albo oba 2D.")

    # --- NORMALIZACJA DANYCH: dopuszczamy listę [ (1,Ny,Nx) ] lub ndarray (T,1,Ny,Nx) ---
    def _to_3d(A, name):
        A = np.asarray(A, dtype=np.float32)

        if isinstance(A, list):
            normed = []
            for i, a in enumerate(A):
                a = np.asarray(a, dtype=np.float32)
                # zgnieć osie o długości 1 (np. (1,Ny,Nx) -> (Ny,Nx))
                a = np.squeeze(a)
                if a.ndim != 2:
                    raise ValueError(f"{name}[{i}] po squeeze ma {a.ndim} wymiary; oczekiwano 2D.")
                if a.shape == (nlat, nlon):
                    normed.append(a)
                elif a.shape == (nlon, nlat):
                    normed.append(a.T)
                else:
                    raise ValueError(f"{name}[{i}] shape {a.shape} nie pasuje do (Ny,Nx)=({nlat},{nlon}).")
            A = np.stack(normed, axis=0)  # (T,Ny,Nx)
        else:
            # ndarray: może być 2D, 3D, czasem 4D z jedynkami
            if A.ndim == 4:
                # typowo (T,1,Ny,Nx) lub (T,Ny,1,Nx) etc. -> zgnieć jedynki
                A = np.squeeze(A)
            if A.ndim == 2:
                if A.shape == (nlat, nlon):
                    A = A[None, ...]
                elif A.shape == (nlon, nlat):
                    A = A.T[None, ...]
                else:
                    raise ValueError(f"{name} shape {A.shape} nie pasuje do (Ny,Nx)=({nlat},{nlon}).")
            elif A.ndim == 3:
                sh = A.shape[1:]
                if sh == (nlat, nlon):
                    pass
                elif sh == (nlon, nlat):
                    A = A.transpose(0, 2, 1)
                else:
                    # spróbuj jeszcze squeeze na wypadek nietypowych jedynek
                    A2 = np.squeeze(A)
                    if A2.ndim == 2:
                        # traktuj jak „pojedynczy czas”
                        if A2.shape == (nlat, nlon):
                            A = A2[None, ...]
                        elif A2.shape == (nlon, nlat):
                            A = A2.T[None, ...]
                        else:
                            raise ValueError(f"{name} shape {A.shape}->{A2.shape} nie pasuje do (Ny,Nx).")
                    elif A2.ndim == 3 and A2.shape[1:] in [(nlat, nlon), (nlon, nlat)]:
                        if A2.shape[1:] == (nlon, nlat):
                            A = A2.transpose(0, 2, 1)
                        else:
                            A = A2
                    else:
                        raise ValueError(f"{name} shape {A.shape} nie pasuje do (T,Ny,Nx) po squeeze.")
            else:
                raise ValueError(f"{name} ma {A.ndim} wymiary; oczekiwano 2D–4D.")
        # sprawdź długość czasu
        if A.shape[0] != len(time_vals):
            raise ValueError(f"Długość czasu T={len(time_vals)}, a {name}.shape[0]={A.shape[0]}.")
        return A.astype("float32", copy=False)


    V_3d  = _to_3d(V_all,  "V_all")   # (T, Ny, Nx)
    SS_3d = _to_3d(SS_All, "SS_All")  # (T, Ny, Nx)

    V_name, SS_name = var_names

    # ---------- budowa Datasetu ----------
    if grid_kind == "rect":
        lat_da = xr.DataArray(lat_grid.astype("float32"),
                              dims=("lat",),
                              attrs=dict(standard_name="latitude", long_name="latitude", units="degrees_north"))
        lon_da = xr.DataArray(lon_grid.astype("float32"),
                              dims=("lon",),
                              attrs=dict(standard_name="longitude", long_name="longitude", units="degrees_east"))
        coords = dict(time=("time", time_vals),
                      lat=lat_da, lon=lon_da)
        dims = ("time", "lat", "lon")
    else:
        # curvilinear: dane (time,y,x), lat(y,x), lon(y,x)
        y_da = xr.DataArray(np.arange(nlat, dtype="int32"), dims=("y",), attrs=dict(long_name="grid_y_index"))
        x_da = xr.DataArray(np.arange(nlon, dtype="int32"), dims=("x",), attrs=dict(long_name="grid_x_index"))
        lat_da = xr.DataArray(lat_grid.astype("float32"),
                              dims=("y", "x"),
                              attrs=dict(standard_name="latitude", long_name="latitude", units="degrees_north"))
        lon_da = xr.DataArray(lon_grid.astype("float32"),
                              dims=("y", "x"),
                              attrs=dict(standard_name="longitude", long_name="longitude", units="degrees_east"))
        coords = dict(time=("time", time_vals), y=y_da, x=x_da, lat=lat_da, lon=lon_da)
        dims = ("time", "y", "x")

    ds = xr.Dataset(
        data_vars={
            V_name: (dims, V_3d,  {"long_name": "V field",  "coordinates": "lat lon" if grid_kind=="curvi" else ""}),
            SS_name:(dims, SS_3d, {"long_name": "SS field", "coordinates": "lat lon" if grid_kind=="curvi" else ""}),
        },
        coords=coords,
        attrs=dict(
            title="Gridded fields",
            Conventions="CF-1.8",
            history=f"Created with xarray; compress_level={compress_level}",
            grid_type=("rectilinear" if grid_kind=="rect" else "curvilinear"),
            geospatial_lat_min=float(lat_grid.min()),
            geospatial_lat_max=float(lat_grid.max()),
            geospatial_lon_min=float(lon_grid.min()),
            geospatial_lon_max=float(lon_grid.max()),
        ),
    )

    # ---------- encoding / kompresja ----------
    enc = {
        "time": {"units": "seconds since 1970-01-01 00:00:00", "calendar": "standard"},
        V_name: {"zlib": True, "complevel": int(compress_level), "dtype": "float32", "_FillValue": np.float32(np.nan)},
        SS_name:{"zlib": True, "complevel": int(compress_level), "dtype": "float32", "_FillValue": np.float32(np.nan)},
    }
    if grid_kind == "rect":
        enc["lat"] = {"dtype": "float32"}
        enc["lon"] = {"dtype": "float32"}

    if chunks:
        ds = ds.chunk(chunks)

    # _FillValue trzymamy w encoding (CF), więc usuń z attrs jeśli się pojawił
    for vn in (V_name, SS_name):
        ds[vn].attrs.pop("_FillValue", None)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f'{out_name}.nc'
    ds.to_netcdf(outfile,mode="w", format="NETCDF4", encoding=enc)
    return ds