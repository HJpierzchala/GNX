import numpy as np
from scipy.linalg import cho_factor, cho_solve
from typing import Protocol, runtime_checkable, Optional, Dict, Any, Tuple, Literal, Iterable
from dataclasses import dataclass, field
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree, distance
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Optional, Union, Sequence

KrigingMode = Literal["OK", "UK", "WAAS"]
CoordFrame  = Literal["GEO", "SM"]
EARTH_RADIUS_KM = 6371.0
IONO_SHELL_H_KM = 450.0  # WAAS: 350 km

# ------------------------------------------
# Variogram/Covariance models
# ------------------------------------------
@runtime_checkable
class VariogramLike(Protocol):
    """Minimum variogram interface for kriging:
        - cov_func(d_km): covariance C(d) of residuals [VTEC^2] for distances in [km]
        - sill_var: variance 'sill' = sigma_total^2 (float)
        """
    def cov_func(self, d_km: np.ndarray) -> np.ndarray: ...
    @property
    def sill_var(self) -> float: ...

# =========================
# 2) Sparks model
# =========================
@dataclass
class SparksVariogram:
    """Sparks/WAAS (2011):
        C(d) = sigma_total^2                         , d=0
             = (sigma_total^2 - sigma_nominal^2) * exp(-d/ddecorr) , d>0
        Units: VTEC->TECU, covariances -> (TECU)^2.
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
# 3) Other models
# =========================
@dataclass
class SphericalVariogram:
    """Spherical model (classic geostatistics).
        Parameters:
          - sigma_nominal: √nugget (TECU)
          - sigma_total:   √sill   (TECU)
          - range_km:      range a [km], after which C(d>=a)=0
        Covariance of residuals:
          C(0)=sill; for 0<d<a: (sill-nugget)*(1 - 1.5 h + 0.5 h^3), h=d/a; for d>=a: 0
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
    """Exponential model:
           C(d) = (sill - nugget) * exp(-d / L), C(0)=sill
        where L ~ scale/decoherence length [km].
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
    """Gaussian model:
           C(d) = (sill - nugget) * exp(-(d/L)^2), C(0)=sill
        More "smooth" at d~0 than exponential.
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
# 4) Universal wrapper
# =========================
@dataclass
class GenericVariogram:
    """Universal wrapper: model ∈ {'spherical','exponential','gaussian','sparks'}.
        Useful when you want to select a model by name
        """
    model: Literal["spherical", "exponential", "gaussian", "sparks"] = "exponential"
    params: Dict[str, float] = field(default_factory=dict)

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


def llh_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_km: float = IONO_SHELL_H_KM) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    R = EARTH_RADIUS_KM + h_km
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.stack([x, y, z], axis=-1)



def enu_basis_at(lat0_deg: float, lon0_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns unit vectors (up, east, north) in ECEF (according to Sparks 2011)"""

    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    eup = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    ee = np.array([-np.sin(lon0), np.cos(lon0), 0.0])  # east
    en = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])  # north
    return eup, ee, en


def ecef_distance_km(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Euclidean distance in ECEF in kilometers. P: (...,3), Q: (...,3)."""
    return np.linalg.norm(P - Q, axis=-1)

def build_G(dx_e: np.ndarray, dx_n: np.ndarray) -> np.ndarray:
    """Returns matrix G (N x 3): [1, dx·east, dx·north]."""
    ones = np.ones_like(dx_e)
    return np.stack([ones, dx_e, dx_n], axis=1)

def waas_kriging_point_fast(
    ipp_xyz: np.ndarray,          # (N,3) IPP ECEF
    ipp_vals: np.ndarray,         # (N,)
    igp_xyz: np.ndarray,          # (3,)
    igp_latlon: Tuple[float, float],
    vario: VariogramLike,
    C_full: np.ndarray,           # (N,N) = vario.cov_func(D), precomputed for epoch
    M_full: Optional[np.ndarray] = None,  # (N,N)  precomputed for epoch (może być None)
    idx: Optional[np.ndarray] = None,     # neighbors used for estimation (1D indeksy)
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Fast solver: uses only submatrices A11 = (C+M)[idx][:,idx] and Schur.
    Returns (I_hat, var_hat, info). var_hat uses the kriging variance formula
    via Lagrange multipliers: σ^2 = sill - (c^T w + s^T λ).
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

    # A11 = C + M (submatrix on idx)
    A11 = C_full[np.ix_(idx, idx)].copy()
    if M_full is not None:
        A11 += M_full[np.ix_(idx, idx)]

    # cov vector c (N,)
    d_igp = ecef_distance_km(ipp_xyz[idx], igp_xyz[None, :])
    c = vario.cov_func(d_igp)

    # Cholesky A11
    try:
        cf = cho_factor(A11, overwrite_a=False, check_finite=False)
    except Exception:
        # minimal regularization, if A11 were singular
        eps = 1e-8 * (np.trace(A11)/max(N,1))
        cf = cho_factor(A11 + eps*np.eye(N), overwrite_a=False, check_finite=False)

    # Solve’y with A11^{-1} (we use cho_solve, without explicit inversion)
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

    # Estimate and variance
    I_hat = float(w @ ipp_vals[idx])
    # σ^2 = sill - (c^T w + s^T λ)  (kriging variance with Lagrange multipliers)
    var_hat = float(vario.sill_var - (c @ w + s @ lam))
    if not np.isfinite(var_hat):

        C = C_full[np.ix_(idx, idx)]
        var_hat = float(w @ (C @ w) - 2.0*(w @ c) + vario.sill_var)
    var_hat = max(var_hat, 0.0)

    info = {"used": int(N), "condS": float(np.linalg.cond(S))}
    return I_hat, var_hat, info


def _sm_ll_to_ecef_km(tr, lat_deg, lon_deg, r_norm_km, obstime):
    """SM(lat,lon,r_norm) -> ECEF in [km]. lat/lon deg; r_norm_km skalar or 2D."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    r_m = np.asarray(r_norm_km, float) * 1e3
    xyz_m = tr.sm_ll_to_ecef(lat_rad, lon_rad, r_norm=r_m, when=obstime)  # (...,3) [m]
    return np.asarray(xyz_m, float) / 1e3  # [km]


def _default_rnorm_km(shape, base=EARTH_RADIUS_KM, h=IONO_SHELL_H_KM):
    """Constant geocentric radius (sphere) in [km]."""
    return np.full(shape, base + h, dtype=float)

@dataclass
class IonoKrigingMonitor:
    # grid on which we want maps (deg) – can be GEO or SM
    grid_lon: np.ndarray
    grid_lat: np.ndarray

    # In what format do I provide the grid and IPP (lat/lon in degrees)?
    coord_frame: CoordFrame = "GEO"           # "GEO" (geodetic) or "SM"

    # SM context (required if coord_frame="SM"))
    sm_transformer: Optional[Any] = None      # gnx.SolarGeomagneticTransformer
    sm_obstime:     Optional[Any] = None      # astropy.Time (skalar)
    grid_rnorm_km:  Optional[np.ndarray] = None  # (Ny,Nx) geocentric radius; if None -> sphere

    # kriging mode
    kriging_mode: KrigingMode = "WAAS"

    # OK/UK fallback
    variogram_model: Literal["spherical","exponential","gaussian","linear","power","hole-effect"] = "spherical"
    variogram_parameters: Optional[Dict[str, float]] = None
    n_closest_points: int = 60
    backend: Literal["C", "loop", "vectorized"] = "C"
    use_external_drift: bool = False

    # WAAS/Sparks: neighborhood selection parameters
    Rmin_km: float = 800.0
    Rmax_km: float = 2100.0
    Ntarget: int = 30
    Nmin: int = 10

    # WAAS/Sparks: variogram model
    variogram: VariogramLike = field(default_factory=SparksVariogram)

    # WAAS/Sparks:noise variations (optional in df)
    default_meas_var: float = 0.0
    bias_var_rec: float = 0.0
    bias_var_sat: float = 0.0

    # optional overrides (useful when you want to calculate ECEF yourself outside of class)
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
            raise ValueError("grid_lat/grid_lon must both be 1D or both 2D with compatible shapes.")

        self._grid_lat = Lat
        self._grid_lon = Lon
        self._grid_shape = Lat.shape  # (Ny, Nx)

        # --- gdzie pracujemy: GEO czy SM? ---
        if self.grid_xyz_override is not None:
            # dostaliśmy gotowe ECEF [km]
            if self.grid_xyz_override.shape != (Lat.shape[0], Lat.shape[1], 3):
                raise ValueError("grid_xyz_override must have the shape (Ny,Nx,3).")
            self._grid_xyz = np.asarray(self.grid_xyz_override, float)
        elif self.coord_frame == "GEO":
            # bezpiecznie: sfera (jak dotąd)
            self._grid_xyz = llh_to_ecef(Lat, Lon)     # [km]
        elif self.coord_frame == "SM":
            if (self.sm_transformer is None) or (self.sm_obstime is None):
                raise ValueError("For coord_frame='SM', specify sm_transformer and sm_obstime.")
            # promień geocentryczny dla siatki
            rnorm = self.grid_rnorm_km
            if rnorm is None:
                rnorm = _default_rnorm_km(Lat.shape)
            else:
                rnorm = np.asarray(rnorm, float)
                if rnorm.shape != Lat.shape:
                    raise ValueError("grid_rnorm_km must have the shape (Ny,Nx).")
            # SM(lat,lon,r_norm) -> ECEF [km]
            self._grid_xyz = _sm_ll_to_ecef_km(self.sm_transformer, Lat, Lon, rnorm, self.sm_obstime)
        else:
            raise ValueError("coord_frame must be 'GEO' or 'SM'.")
        if not isinstance(self.variogram, VariogramLike):
            raise TypeError("The variogram must implement VariogramLike (cov_func & sill_var).")

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


    # -------------- WAAS/Sparks --------------
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

    # -------------- SINGLE EPOCH KRIGING --------------
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
                raise ValueError("ipp_xyz_override must have shape (N,3) in km.")
        elif self.ipp_xyz_override is not None:
            ipp_xyz = np.asarray(self.ipp_xyz_override, float)
            if ipp_xyz.ndim != 2 or ipp_xyz.shape[1] != 3:
                raise ValueError("ipp_xyz_override must have shape (N,3) in km.")
        else:
            if self.coord_frame == "GEO":
                ipp_xyz = llh_to_ecef(lat, lon, IONO_SHELL_H_KM)
            elif self.coord_frame == "SM":
                if (self.sm_transformer is None) or (self.sm_obstime is None):
                    raise ValueError("For coord_frame='SM', sm_transformer and sm_obstime are required.")
                if "r_norm_km" in dfe.columns:
                    rnorm_km = dfe["r_norm_km"].to_numpy(float)
                else:
                    rnorm_km = EARTH_RADIUS_KM + IONO_SHELL_H_KM
                ipp_xyz = _sm_ll_to_ecef_km(self.sm_transformer, lat, lon, rnorm_km, self.sm_obstime)
            else:
                raise ValueError("coord_frame must be 'GEO' or 'SM'.")

        meas_var = dfe["meas_var"].to_numpy(float) if "meas_var" in dfe.columns else None
        rec_id = dfe["rec_id"].to_numpy(object) if "rec_id" in dfe.columns else None
        sat_id = dfe["sat_id"].to_numpy(object) if "sat_id" in dfe.columns else None
        if meas_var is None and self.default_meas_var > 0:
            meas_var = np.full(len(vtec), float(self.default_meas_var), float)

        if self.kriging_mode in ["OK","UK"]:
            kriger = self._build_kriger(lon, lat, vtec)

            gx = self._grid_lon[0, :].astype(float)  # Nx
            gy = self._grid_lat[:, 0].astype(float)  # Ny

            sort_x = np.argsort(gx)
            sort_y = np.argsort(gy)
            gx_s = gx[sort_x]
            gy_s = gy[sort_y]

            z_s, ss_s = kriger.execute(
                "grid",
                gx_s,  # 1D longitudes
                gy_s,  # 1D latitudes
            )
            inv_x = np.argsort(sort_x)
            inv_y = np.argsort(sort_y)
            Vgrid = np.asarray(z_s)[inv_y][:, inv_x].astype(float)
            SS = np.asarray(ss_s)[inv_y][:, inv_x].astype(float)
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
            raise ValueError(f"Unknown kriging mode:  {self.kriging_mode}")




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
     Saves fields on a grid to NetCDF:
  - rectilinear: lat(Ny,), lon(Nx,), data (T, Ny, Nx)
  - curvilinear: lat(Ny,Nx), lon(Ny,Nx), data (T, Ny, Nx)
V_all/SS_All can be 2D lists or 2D/3D arrays.
        """

    # ---------- przygotowanie czasu ----------
    # Rzut do pandas.DatetimeIndex → numpy datetime64[ns]
    time_index = pd.to_datetime(times)
    if time_index.isna().any():
        raise ValueError("Incorrect date/time values detected in 'times'.")
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
        raise ValueError("lat_grid/lon_grid must both be 1D or both 2D.")

    def _to_3d(A, name):
        A = np.asarray(A, dtype=np.float32)

        if isinstance(A, list):
            normed = []
            for i, a in enumerate(A):
                a = np.asarray(a, dtype=np.float32)
                a = np.squeeze(a)
                if a.ndim != 2:
                    raise ValueError(f"{name}[{i}] after squeeze has {a.ndim} dimensions; 2D was expected.")
                if a.shape == (nlat, nlon):
                    normed.append(a)
                elif a.shape == (nlon, nlat):
                    normed.append(a.T)
                else:
                    raise ValueError(f"{name}[{i}] shape {a.shape} not in match with (Ny,Nx)=({nlat},{nlon}).")
            A = np.stack(normed, axis=0)  # (T,Ny,Nx)
        else:
            if A.ndim == 4:
                A = np.squeeze(A)
            if A.ndim == 2:
                if A.shape == (nlat, nlon):
                    A = A[None, ...]
                elif A.shape == (nlon, nlat):
                    A = A.T[None, ...]
                else:
                    raise ValueError(f"{name} shape {A.shape} not in match with (Ny,Nx)=({nlat},{nlon}).")
            elif A.ndim == 3:
                sh = A.shape[1:]
                if sh == (nlat, nlon):
                    pass
                elif sh == (nlon, nlat):
                    A = A.transpose(0, 2, 1)
                else:
                    A2 = np.squeeze(A)
                    if A2.ndim == 2:
                        # traktuj jak „pojedynczy czas”
                        if A2.shape == (nlat, nlon):
                            A = A2[None, ...]
                        elif A2.shape == (nlon, nlat):
                            A = A2.T[None, ...]
                        else:
                            raise ValueError(f"{name} shape {A.shape}->{A2.shape} not in match with (Ny,Nx).")
                    elif A2.ndim == 3 and A2.shape[1:] in [(nlat, nlon), (nlon, nlat)]:
                        if A2.shape[1:] == (nlon, nlat):
                            A = A2.transpose(0, 2, 1)
                        else:
                            A = A2
                    else:
                        raise ValueError(f"{name} shape {A.shape} not in match with (T,Ny,Nx) after squeeze.")
            else:
                raise ValueError(f"{name} has {A.ndim} dimensions; 2D–4D was expected.")
        # sprawdź długość czasu
        if A.shape[0] != len(time_vals):
            raise ValueError(f"The length of time T={len(time_vals)}, and {name}.shape[0]={A.shape[0]}.")
        return A.astype("float32", copy=False)


    V_3d  = _to_3d(V_all,  "V_all")
    SS_3d = _to_3d(SS_All, "SS_All")

    V_name, SS_name = var_names

    # ---------- Datasetu building----------
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
        # curvilinear: data (time,y,x), lat(y,x), lon(y,x)
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

    for vn in (V_name, SS_name):
        ds[vn].attrs.pop("_FillValue", None)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f'{out_name}.nc'
    ds.to_netcdf(outfile,mode="w", format="NETCDF4", encoding=enc)
    return ds

def _dist_matrix(lon, lat, coordinates_type="euclidean"):
    """
        The distance matrix d_ij between data points.

        coordinates_type:
            'euclidean'  -> Euclidean distance in degrees
            'geographic' -> angular distance on the sphere [in radians]
        """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)

    if coordinates_type == "euclidean":
        dx = lon[:, None] - lon[None, :]
        dy = lat[:, None] - lat[None, :]
        return np.sqrt(dx * dx + dy * dy)

    elif coordinates_type == "geographic":
        lon_r = np.deg2rad(lon)
        lat_r = np.deg2rad(lat)
        lon1 = lon_r[:, None]
        lon2 = lon_r[None, :]
        lat1 = lat_r[:, None]
        lat2 = lat_r[None, :]

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        return c

    else:
        raise ValueError(f"Unknown coordinates_type={coordinates_type!r}")


def _dist_data_to_points(lon_data, lat_data, lon_points, lat_points, coordinates_type="euclidean"):
    """
        Distances between data points (n) and target points (m).

        Returns D with shape (n, m).
        """
    lon_data = np.asarray(lon_data, float)
    lat_data = np.asarray(lat_data, float)
    lon_points = np.asarray(lon_points, float)
    lat_points = np.asarray(lat_points, float)

    if coordinates_type == "euclidean":
        dx = lon_data[:, None] - lon_points[None, :]
        dy = lat_data[:, None] - lat_points[None, :]
        return np.sqrt(dx * dx + dy * dy)

    elif coordinates_type == "geographic":
        lon_d = np.deg2rad(lon_data)[:, None]
        lat_d = np.deg2rad(lat_data)[:, None]
        lon_p = np.deg2rad(lon_points)[None, :]
        lat_p = np.deg2rad(lat_points)[None, :]

        dlon = lon_p - lon_d
        dlat = lat_p - lat_d
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_d) * np.cos(lat_p) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        return c

    else:
        raise ValueError(f"Unknown coordinates_type={coordinates_type!r}")


# ---------- wariogram jak w PyKrige: gamma(d) ----------

def _build_variogram_function(variogram_model, variogram_parameters):
    """
        Variogram models mapping

        variogram_model : {'spherical','exponential','gaussian','linear','power','hole-effect'}
        variogram_parameters :
            - dict: {'sill': s, 'range': r, 'nugget': n}  or {'psill': p, 'range': r, 'nugget': n}
            - list: [sill, range, nugget]  (sill = full sill, NOT partial)

        Returns:
            gamma(d)  -> semi-variogram
            psill, nugget, var_range
        """
    if variogram_parameters is None:
        raise ValueError("variogram_parameters must be specified (sill/psill, range, nugget).")

    if isinstance(variogram_parameters, dict):
        if "psill" in variogram_parameters:
            psill = float(variogram_parameters["psill"])
            var_range = float(variogram_parameters["range"])
            nugget = float(variogram_parameters.get("nugget", 0.0))
        else:
            sill = float(variogram_parameters["sill"])
            var_range = float(variogram_parameters["range"])
            nugget = float(variogram_parameters.get("nugget", 0.0))
            psill = sill - nugget
    else:
        sill, var_range, nugget = map(float, variogram_parameters)
        psill = sill - nugget

    model = variogram_model.lower()

    def gamma(d):
        d = np.asarray(d, float)
        if model == "spherical":
            hr = d / var_range
            g = np.empty_like(hr)
            inside = hr <= 1.0
            g[inside] = psill * (1.5 * hr[inside] - 0.5 * hr[inside] ** 3) + nugget
            g[~inside] = psill + nugget
            return g

        elif model == "exponential":
            return psill * (1.0 - np.exp(-d / (var_range / 3.0))) + nugget

        elif model == "gaussian":
            denom = (4.0 / 7.0 * var_range) ** 2
            return psill * (1.0 - np.exp(-d**2 / denom)) + nugget

        elif model == "hole-effect":
            a = d / (var_range / 3.0)
            return psill * (1.0 - (1.0 - a) * np.exp(-a)) + nugget

        elif model == "linear":
            slope = psill / var_range
            return slope * d + nugget

        elif model == "power":
            exponent = 1.0
            scale = psill / (var_range**exponent)
            return scale * d**exponent + nugget

        else:
            raise ValueError(f"Unsupported variogram_model={variogram_model!r}")

    return gamma, psill, nugget, var_range


# ---------- baza dla OK/UK ----------

class _BaseKrigingGeo:
    """
        Base for OK/UK on geographic coordinates (lon, lat).

        - works in VARIOGRAM space: gamma(d)
        - kriging matrix has gamma(0)=0 on the diagonal (exact values)
        - distances: 'euclidean' by default
    """

    def __init__(
        self,
        lon,
        lat,
        values,
        *,
        variogram_model="spherical",
        variogram_parameters=None,
        coordinates_type="euclidean",
    ):
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        values = np.asarray(values, float)
        if lon.shape != lat.shape or lon.shape != values.shape:
            raise ValueError("lon, lat, values must have the same 1D shape.")

        self.lon = lon
        self.lat = lat
        self.values = values
        self.n = lon.size
        self.coordinates_type = coordinates_type

        self.gamma, self.psill, self.nugget, self.var_range = _build_variogram_function(
            variogram_model, variogram_parameters
        )
        self.variogram_model = variogram_model
        self.variogram_parameters = variogram_parameters

        D = _dist_matrix(lon, lat, coordinates_type=self.coordinates_type)
        Gamma = self.gamma(D)

        np.fill_diagonal(Gamma, 0.0)

        self.Gamma = Gamma  # (n, n)
        self._cho = None
        self._A_pinv = None
        self._n_aug = None


# ---------- Ordinary Kriging ----------

class OrdinaryKrigingGeo(_BaseKrigingGeo):
    """
        Ordinary Kriging.

            OK = OrdinaryKrigingGeo(
                lon, lat, vtec,
                variogram_model="spherical",
                variogram_parameters={"sill": ..., "range": ..., "nugget": ...},
                coordinates_type="euclidean",
            )
            z, ss = OK.execute("grid", gx, gy)
        """

    def __init__(self, lon, lat, values, **kwargs):
        super().__init__(lon, lat, values, **kwargs)
        self._build_system()

    def _build_system(self):
        n = self.n
        A = np.empty((n + 1, n + 1), float)
        A[:n, :n] = self.Gamma
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        A[n, n] = 0.0

        try:
            self._cho = cho_factor(A, overwrite_a=False, check_finite=False)
            self._A_pinv = None
        except Exception:
            self._cho = None
            self._A_pinv = np.linalg.pinv(A)

        self._n_aug = n + 1

    def _solve_weights(self, gamma_vec):
        """
        We solve:
            [Γ 1; 1^T 0] [λ; μ] = [γ*; 1]
        """
        n = self.n
        b = np.empty(self._n_aug, float)
        b[:n] = gamma_vec
        b[n] = 1.0

        if self._cho is not None:
            x = cho_solve(self._cho, b, check_finite=False)
        else:
            x = self._A_pinv @ b

        w = x[:n]
        mu = x[n]
        return w, mu

    def execute(self, style, xpoints, ypoints):
        if style != "grid":
            raise NotImplementedError("OrdinaryKrigingGeo.execute only supports style='grid'.")

        gx = np.asarray(xpoints, float)
        gy = np.asarray(ypoints, float)
        Nx = gx.size
        Ny = gy.size

        z_out = np.empty((Ny, Nx), float)
        ss_out = np.empty((Ny, Nx), float)

        for j, lat_row in enumerate(gy):
            D = _dist_data_to_points(
                self.lon, self.lat,
                gx, np.full_like(gx, lat_row),
                coordinates_type=self.coordinates_type,
            )
            Gamma_star = self.gamma(D)  # (n, Nx)

            for i in range(Nx):
                gamma_vec = Gamma_star[:, i]
                w, mu = self._solve_weights(gamma_vec)
                z_hat = float(w @ self.values)
                var_hat = float(w @ gamma_vec + mu)  # σ² = Σ λ_i γ_i0 + μ
                z_out[j, i] = z_hat
                ss_out[j, i] = max(var_hat, 0.0)

        return z_out, ss_out


# ---------- Universal Kriging ----------

class UniversalKrigingGeo(_BaseKrigingGeo):
    """
    Universal Kriging with drift_terms=['regional_linear'] -> baza [1, lon, lat].

    UK = UniversalKrigingGeo(
        lon, lat, vtec,
        variogram_model="spherical",
        variogram_parameters={"sill": ..., "range": ..., "nugget": ...},
        coordinates_type="euclidean",
        drift_terms=["regional_linear"],
    )
    z, ss = UK.execute("grid", gx, gy)
    """

    def __init__(self, lon, lat, values, *, drift_terms=None, **kwargs):
        self.drift_terms = drift_terms or ["regional_linear"]
        super().__init__(lon, lat, values, **kwargs)
        self._build_drift()
        self._build_system()

    def _build_drift(self):
        lon = self.lon
        lat = self.lat

        if self.drift_terms == ["regional_linear"]:
            F = np.column_stack([np.ones_like(lon), lon, lat])
        else:
            raise NotImplementedError(
                f"UniversalKrigingGeo currently only supports drift_terms=['regional_linear'], "
                f"received {self.drift_terms!r}"
            )
        self.F = F
        self.p = F.shape[1]

    def _build_system(self):
        n = self.n
        p = self.p
        Gamma = self.Gamma
        F = self.F

        A = np.empty((n + p, n + p), float)
        A[:n, :n] = Gamma
        A[:n, n:] = F
        A[n:, :n] = F.T
        A[n:, n:] = 0.0

        try:
            self._cho = cho_factor(A, overwrite_a=False, check_finite=False)
            self._A_pinv = None
        except Exception:
            self._cho = None
            self._A_pinv = np.linalg.pinv(A)

        self._n_aug = n + p

    def _solve_weights(self, gamma_vec, f0):
        """
        We solve:
            [Γ F; F^T 0] [λ; α] = [γ*; f0]
        """
        n = self.n
        b = np.empty(self._n_aug, float)
        b[:n] = gamma_vec
        b[n:] = f0

        if self._cho is not None:
            x = cho_solve(self._cho, b, check_finite=False)
        else:
            x = self._A_pinv @ b

        w = x[:n]
        alpha = x[n:]
        return w, alpha

    def _drift_vector(self, lon_p, lat_p):
        if self.drift_terms == ["regional_linear"]:
            return np.array([1.0, float(lon_p), float(lat_p)], dtype=float)
        else:
            raise NotImplementedError

    def execute(self, style, xpoints, ypoints):
        if style != "grid":
            raise NotImplementedError("UniversalKrigingGeo.execute only supports style='grid'.")

        gx = np.asarray(xpoints, float)
        gy = np.asarray(ypoints, float)
        Nx = gx.size
        Ny = gy.size

        z_out = np.empty((Ny, Nx), float)
        ss_out = np.empty((Ny, Nx), float)

        for j, lat_row in enumerate(gy):
            D = _dist_data_to_points(
                self.lon, self.lat,
                gx, np.full_like(gx, lat_row),
                coordinates_type=self.coordinates_type,
            )
            Gamma_star = self.gamma(D)  # (n, Nx)

            for i in range(Nx):
                gamma_vec = Gamma_star[:, i]
                f0 = self._drift_vector(gx[i], lat_row)
                w, alpha = self._solve_weights(gamma_vec, f0)
                z_hat = float(w @ self.values)
                var_hat = float(w @ gamma_vec + alpha @ f0)
                z_out[j, i] = z_hat
                ss_out[j, i] = max(var_hat, 0.0)

        return z_out, ss_out
