import dataclasses
import warnings
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

from .utils import _check_choice, _check_number

KrigingMode = Literal["OK", "UK", "WAAS"]
CoordFrame  = Literal["GEO", "SM"]
EARTH_RADIUS_KM = 6371.0
IONO_SHELL_H_KM = 450.0  # WAAS: 350 km
KRIGING_MODES = ("OK", "UK", "WAAS")
COORD_FRAMES = ("GEO", "SM")
OKUK_COORDINATES_TYPES = ("geographic", "euclidean")
GENERIC_VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "sparks")
_GENERIC_RANGE_KEY = {"spherical": "range_km", "exponential": "L_km", "gaussian": "L_km", "sparks": "ddecorr_km"}
OKUK_VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "linear", "power", "hole-effect")
WAAS_MIN_POINTS = 3   # the WAAS planar trend [1, e, n] has 3 unknowns


def _wrap180(lon_deg):
    """Longitude (or longitude difference) wrapped to [-180, 180) degrees."""
    return (np.asarray(lon_deg, float) + 180.0) % 360.0 - 180.0


def _check_sigmas(owner: str, sigma_nominal, sigma_total) -> None:
    """0 <= sigma_nominal <= sigma_total (TECU): the covariance C(d > 0) must not be negative."""
    _check_number(f"{owner}.sigma_nominal", sigma_nominal, ge=0.0, unit="TECU")
    _check_number(f"{owner}.sigma_total", sigma_total, ge=0.0, unit="TECU")
    if sigma_nominal > sigma_total:
        raise ValueError(f"{owner}: sigma_nominal must be <= sigma_total (got sigma_nominal={sigma_nominal!r}, "
                         f"sigma_total={sigma_total!r}); C(d>0) = sigma_total^2 - sigma_nominal^2 would be negative")

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
        Status: experimental/research helper for ionosphere kriging.

        C(d) = sigma_total^2                         , d=0
             = (sigma_total^2 - sigma_nominal^2) * exp(-d/ddecorr) , d>0
        Units: VTEC->TECU, covariances -> (TECU)^2.
        Validation: 0 <= sigma_nominal <= sigma_total and ddecorr_km > 0 (ValueError otherwise).
        """
    sigma_nominal: float = 0.5    # TECU
    sigma_total: float = 1.5      # TECU
    ddecorr_km: float = 800.0     # km

    def __post_init__(self):
        _check_sigmas("SparksVariogram", self.sigma_nominal, self.sigma_total)
        _check_number("SparksVariogram.ddecorr_km", self.ddecorr_km, gt=0.0, unit="km")

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total ** 2
        cN = self.sigma_nominal ** 2
        out = (c0 - cN) * np.exp(-np.maximum(d_km, 0.0) / self.ddecorr_km)
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
        Validation: 0 <= sigma_nominal <= sigma_total and range_km > 0 (ValueError otherwise).
        """
    sigma_nominal: float = 0.5
    sigma_total: float = 1.5
    range_km: float = 1200.0

    def __post_init__(self):
        _check_sigmas("SphericalVariogram", self.sigma_nominal, self.sigma_total)
        _check_number("SphericalVariogram.range_km", self.range_km, gt=0.0, unit="km")

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total**2
        cN = self.sigma_nominal**2
        a = self.range_km
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
        Validation: 0 <= sigma_nominal <= sigma_total and L_km > 0 (ValueError otherwise).
        """
    sigma_nominal: float = 0.5
    sigma_total: float = 1.5
    L_km: float = 800.0

    def __post_init__(self):
        _check_sigmas("ExponentialVariogram", self.sigma_nominal, self.sigma_total)
        _check_number("ExponentialVariogram.L_km", self.L_km, gt=0.0, unit="km")

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total**2
        cN = self.sigma_nominal**2
        L = self.L_km
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
        Validation: 0 <= sigma_nominal <= sigma_total and L_km > 0 (ValueError otherwise).
        """
    sigma_nominal: float = 0.5
    sigma_total: float = 1.5
    L_km: float = 900.0

    def __post_init__(self):
        _check_sigmas("GaussianVariogram", self.sigma_nominal, self.sigma_total)
        _check_number("GaussianVariogram.L_km", self.L_km, gt=0.0, unit="km")

    def cov_func(self, d_km: np.ndarray) -> np.ndarray:
        c0 = self.sigma_total**2
        cN = self.sigma_nominal**2
        L2 = self.L_km**2
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
    """Universal wrapper: model ∈ {'spherical','exponential','gaussian','sparks'} (case-insensitive).
        Useful when you want to select a model by name. ``params`` are the keyword arguments of the
        selected class (sigma_nominal, sigma_total and range_km / L_km / ddecorr_km); missing ones take
        that class's defaults. An unknown model or invalid parameter values raise ValueError at construction.
        """
    model: Literal["spherical", "exponential", "gaussian", "sparks"] = "exponential"
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not (isinstance(self.model, str) and self.model.lower() in GENERIC_VARIOGRAM_MODELS):
            _check_choice("GenericVariogram.model", self.model, GENERIC_VARIOGRAM_MODELS)
        if not isinstance(self.params, dict):
            raise ValueError(f"GenericVariogram.params must be a dict (got {type(self.params).__name__})")
        allowed = ("sigma_nominal", "sigma_total", _GENERIC_RANGE_KEY[self.model.lower()])
        foreign = sorted(str(k) for k in self.params if k not in allowed)
        if foreign:
            raise ValueError(f"GenericVariogram.params has keys {foreign} that model {self.model!r} does not use; "
                             f"expected a subset of {list(allowed)}")
        self._impl()   # validates the parameter values of the selected model

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
        # m == "exponential" (the model is validated in __post_init__)
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


_VARIOGRAM_NAMES = {SparksVariogram: "sparks", SphericalVariogram: "spherical",
                    ExponentialVariogram: "exponential", GaussianVariogram: "gaussian"}


def _variogram_info(vg) -> Dict[str, Any]:
    """Model name and actual parameters of a WAAS variogram (GenericVariogram resolved to its model)."""
    impl = vg._impl() if isinstance(vg, GenericVariogram) else vg
    out: Dict[str, Any] = {"model": _VARIOGRAM_NAMES.get(type(impl), type(impl).__name__)}
    if dataclasses.is_dataclass(impl) and not isinstance(impl, type):
        out.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in dataclasses.asdict(impl).items()})
    out["sill_var"] = float(impl.sill_var)
    return out


_OKUK_PARAM_KEYS = ("sill", "psill", "range", "nugget")


def _okuk_params_dict(variogram_parameters, *, owner: str = "variogram_parameters") -> Dict[str, float]:
    """OK/UK variogram parameters as a dict of floats: {'sill' | 'psill', 'range', 'nugget'}.

    Accepts the dict forms {'sill', 'range', 'nugget'} / {'psill', 'range', 'nugget'} or a
    sequence [sill, range, nugget] (sill = full sill). ``range`` is in degrees of great-circle
    distance ('geographic') or of planar lon/lat ('euclidean'). Raises ValueError naming the
    missing or invalid entry.
    """
    form = "{'sill' (or 'psill'), 'range', 'nugget'} or [sill, range, nugget]"
    if variogram_parameters is None:
        raise ValueError(f"{owner} must be given for OK/UK kriging: {form} (got None)")
    if isinstance(variogram_parameters, dict):
        p = dict(variogram_parameters)
        unknown = sorted(str(k) for k in p if k not in _OKUK_PARAM_KEYS)
        if unknown:
            raise ValueError(f"{owner} has unknown keys {unknown}; expected {form}")
        missing = ([] if ("sill" in p or "psill" in p) else ["sill"]) + [k for k in ("range", "nugget") if k not in p]
        if missing:
            raise ValueError(f"{owner} is missing {missing}; expected {form} (got keys {sorted(map(str, p))})")
        if "sill" in p and "psill" in p:
            raise ValueError(f"{owner}: give either 'sill' or 'psill', not both")
    elif isinstance(variogram_parameters, (list, tuple, np.ndarray)) and len(variogram_parameters) == 3:
        p = dict(zip(("sill", "range", "nugget"), variogram_parameters))
    else:
        raise ValueError(f"{owner} must be {form} (got {variogram_parameters!r})")
    p = {k: (v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v) for k, v in p.items()}  # 0-d arrays
    out = {k: _check_number(f"{owner}['{k}']", v) for k, v in p.items()}
    out = {k: float(v) for k, v in out.items()}
    _check_number(f"{owner}['range']", out["range"], gt=0.0, unit="deg")
    _check_number(f"{owner}['nugget']", out["nugget"], ge=0.0)
    psill = out["psill"] if "psill" in out else out["sill"] - out["nugget"]
    if psill < 0:
        raise ValueError(f"{owner}: the sill must be >= the nugget (got {p})")
    return out


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
    if N < WAAS_MIN_POINTS:
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
    """Generate gridded VTEC estimates from IPP observations by kriging.

    Status:
        Experimental public API. It supports WAAS/Sparks-style kriging and
        lightweight OK/UK implementations, but should be validated on each
        network, coordinate frame and variogram choice before operational use.

    Inputs:
        ``grid_lon``/``grid_lat`` define the output grid in degrees: both 1D
        (lon of length Nx, lat of length Ny) or both 2D of shape (Ny, Nx). OK/UK
        need a rectilinear 2D grid with lon varying along axis 1 and lat along
        axis 0 (``np.meshgrid(lon, lat)``, indexing='xy'); another orientation
        raises ValueError. Input epoch data passed to ``krige_epoch`` must
        contain IPP coordinates and ``vtec``. SM mode additionally requires
        ``SolarGeomagneticTransformer`` and an observation time.

    SM frame (``coord_frame='SM'``): the transformation time is the
    ``sm_obstime`` of the constructor and the SM grid is built for it once; the
    ``epoch_time`` of ``krige_epoch`` does not update it, so build one monitor per
    epoch for SM runs over several epochs. In SM mode the ``lat_ipp`` /
    ``lon_ipp`` columns given to ``krige_epoch`` hold SM coordinates (the
    ``lat_sm`` / ``lon_sm`` columns of ``compute_roti_links(coord_mode='SM')``,
    renamed). Mixed systems may share one call: each IPP may carry
    ``rec_id`` and ``sat_id`` for the receiver and satellite bias blocks of the
    WAAS noise covariance.

    Options (validated at construction, ValueError naming the option):
        ``kriging_mode`` in {'OK', 'UK', 'WAAS'}; ``coord_frame`` in {'GEO', 'SM'};
        OK/UK require ``variogram_parameters`` {'sill' (or 'psill'), 'range',
        'nugget'}, and ``variogram_model`` is one of the six OK/UK models;
        ``0 < Rmin_km <= Rmax_km`` (the default Rmin_km = 800 km, so set it too when
        Rmax_km < 800); ``1 <= Nmin <= Ntarget``, and ``Nmin >= 3`` for WAAS (planar trend
        with 3 unknowns); ``default_meas_var``, ``bias_var_rec``, ``bias_var_sat`` >= 0
        [TECU^2]. ``krige_epoch`` re-validates them, so options changed after construction
        take effect (or raise) at the next epoch. ``n_closest_points``, ``backend`` and
        ``use_external_drift`` are not used (OK/UK solve one system over every IPP given to
        ``krige_epoch``, UK with a regional linear drift): a value other than the default gives
        one UserWarning at construction; they are removed after v1.

    OK/UK distances (``coordinates_type``):
        'geographic' (default): great-circle distance in degrees of arc.
        'euclidean': planar distance in degrees of lon/lat, with the longitude
        difference wrapped to [-180, 180). In both cases the variogram
        ``range`` is in degrees. Longitudes are normalised to [-180, 180).
        On great-circle distance only the exponential and spherical models are
        guaranteed positive definite on the sphere (Gneiting 2013); gaussian,
        hole-effect, linear and power are not. The OK/UK systems are solved with a
        pseudo-inverse fallback when the Cholesky factorisation fails.

    WAAS neighbourhood (RTCA DO-229, Sparks et al. 2011):
        all IPPs within ``Rmin_km`` of a grid node; if fewer than ``Ntarget``,
        the ``Ntarget`` nearest IPPs within ``Rmax_km``; with fewer than
        ``Nmin`` the node is not estimated (NaN). Distances are straight-line
        (ECEF) distances between points of the ionospheric shell, in km.
        The Rmin set has no size cap: with dense data (e.g. many epochs pooled
        into one call) every IPP within Rmin enters the node's solve, which costs
        O(n^3) per node (measured: 9205 pooled IPPs, 357 nodes, 31 s).
        The noise covariance is ``M = diag(meas_var) + bias_var_rec*[same
        receiver] + bias_var_sat*[same satellite]`` (``rec_id`` / ``sat_id``
        columns; a missing id matches no other row); ``meas_var`` must be finite
        and >= 0 [TECU^2].

    ``ipp_xyz_override`` (N,3) km has one row per row of ``df_epoch`` (the NaN
    lat/lon/vtec filter is applied to it as well) or one per row kept by that filter.

    Outputs:
        ``krige_epoch`` returns (VTEC grid, variance grid, info dict), also for
        an epoch without data (all-NaN grids, ``info['n_points'] == 0``).

    Limitations:
        The class mixes research workflows and tutorial utilities. Do not change
        neighborhood selection, covariance handling or coordinate transforms
        without numerical comparisons.
    """

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
    # not used (bug-fix stage 13d, Q110; author decision D5 of stage 13): OK/UK solve one system over every IPP
    # given to krige_epoch (no moving neighbourhood, no backend choice, no external drift). A value other than the
    # default gives one UserWarning; the three fields are removed after v1
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

    # OK/UK distance metric: great-circle degrees (default) or planar lon/lat degrees
    coordinates_type: Literal["geographic", "euclidean"] = "geographic"

    def __repr__(self):
        args = ", ".join(f"{k}={v}\n" for k, v in self.__dict__.items())
        return f"IonoKrigingMonitor: ({args})"

    def _validate_options(self):
        _check_choice("kriging_mode", self.kriging_mode, KRIGING_MODES)
        _check_choice("coord_frame", self.coord_frame, COORD_FRAMES)
        _check_choice("coordinates_type", self.coordinates_type, OKUK_COORDINATES_TYPES)
        _check_number("Rmin_km", self.Rmin_km, gt=0.0, unit="km")
        _check_number("Rmax_km", self.Rmax_km, gt=0.0, unit="km")
        if self.Rmin_km > self.Rmax_km:
            raise ValueError(f"Rmin_km must be <= Rmax_km (got Rmin_km={self.Rmin_km!r}, Rmax_km={self.Rmax_km!r}); "
                             "the default Rmin_km is 800 km, so with Rmax_km < 800 set Rmin_km as well")
        _check_number("Ntarget", self.Ntarget, ge=1, integer=True)
        _check_number("Nmin", self.Nmin, ge=1, integer=True)
        if self.Nmin > self.Ntarget:
            raise ValueError(f"Nmin must be <= Ntarget (got Nmin={self.Nmin!r}, Ntarget={self.Ntarget!r})")
        if self.kriging_mode == "WAAS" and self.Nmin < WAAS_MIN_POINTS:
            raise ValueError(f"Nmin must be >= {WAAS_MIN_POINTS} for WAAS kriging, whose planar trend [1, e, n] "
                             f"has 3 unknowns (got Nmin={self.Nmin!r}); nodes with fewer IPPs are never estimated")
        for name in ("default_meas_var", "bias_var_rec", "bias_var_sat"):
            _check_number(name, getattr(self, name), ge=0.0, unit="TECU^2")
        for name in ("grid_xyz_override", "ipp_xyz_override"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, np.ndarray):
                raise TypeError(f"{name} must be a numpy array in km (got {type(v).__name__})")
        if self.kriging_mode in ("OK", "UK"):
            if not (isinstance(self.variogram_model, str) and self.variogram_model.lower() in OKUK_VARIOGRAM_MODELS):
                _check_choice("variogram_model", self.variogram_model, OKUK_VARIOGRAM_MODELS)
            _okuk_params_dict(self.variogram_parameters)   # validated here, re-read from the field at use

    #: fields read by nothing: name -> default
    _UNUSED_FIELDS = {"n_closest_points": 60, "backend": "C", "use_external_drift": False}

    def _warn_unused_fields(self):
        for name, default in self._UNUSED_FIELDS.items():
            value = getattr(self, name)
            try:
                same = bool(value == default)
            except Exception:  # noqa: BLE001 - not comparable: not the default
                same = False
            if not same:
                warnings.warn(f"IonoKrigingMonitor {name}={value!r} is not used: OK/UK kriging solves one system "
                              f"over every IPP given to krige_epoch (no moving neighbourhood of n_closest_points, "
                              f"no backend choice, no external drift). The setting has no effect and will be "
                              f"removed after GNX-py v1.", UserWarning, stacklevel=4)

    def __post_init__(self):
        self._validate_options()
        self._warn_unused_fields()
        self.grid_lon = np.asarray(self.grid_lon, float)
        self.grid_lat = np.asarray(self.grid_lat, float)

        # 1D -> 2D albo 2D -> jak jest
        if self.grid_lon.ndim == 2 and self.grid_lat.ndim == 2:
            if self.grid_lon.shape != self.grid_lat.shape:
                raise ValueError(f"2D grid_lon {self.grid_lon.shape} and grid_lat {self.grid_lat.shape} "
                                 "must have the same shape (Ny, Nx).")
            Lon, Lat = self.grid_lon, self.grid_lat
            if self.kriging_mode in ("OK", "UK") and not (
                    np.all(Lon == Lon[0:1, :]) and np.all(Lat == Lat[:, 0:1])):
                raise ValueError("OK/UK need a rectilinear grid of shape (Ny, Nx) with grid_lon varying along "
                                 "axis 1 only and grid_lat along axis 0 only (np.meshgrid(lon, lat), "
                                 "indexing='xy'); pass 1D grid_lon/grid_lat or fix the grid orientation.")
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
                raise ValueError(f"grid_xyz_override must have the shape (Ny,Nx,3) = "
                                 f"{(Lat.shape[0], Lat.shape[1], 3)} (got {self.grid_xyz_override.shape}).")
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
        params = _okuk_params_dict(self.variogram_parameters)   # current field value, not a construction copy

        if self.kriging_mode == "OK":
            return OrdinaryKrigingGeo(
                lon, lat, vtec,
                variogram_model=self.variogram_model,
                variogram_parameters=params,
                coordinates_type=self.coordinates_type,
            )
        elif self.kriging_mode == "UK":
            return UniversalKrigingGeo(
                    lon,
                    lat,
                    vtec,
                    variogram_model=self.variogram_model,
                    variogram_parameters=params,
                    coordinates_type=self.coordinates_type,
                    drift_terms=["regional_linear"],
                )


    # -------------- WAAS/Sparks --------------
    def _neighbors_for_igp(self, tree: cKDTree, igp_xyz: np.ndarray, n_ipp: int) -> List[Optional[np.ndarray]]:
        """WAAS fit neighbourhood of grid nodes (RTCA DO-229; Sparks et al. 2011).

        ``igp_xyz`` (K,3) km. Per node: every IPP within ``Rmin_km``; if that is fewer than
        ``Ntarget``, the ``Ntarget`` nearest IPPs within ``Rmax_km``. Returns a list of index
        arrays, ``None`` where fewer than ``Nmin`` IPPs qualify (the node stays NaN).
        """
        igp_xyz = np.atleast_2d(np.asarray(igp_xyz, float))
        within = tree.query_ball_point(igp_xyz, r=self.Rmin_km)
        k = int(min(self.Ntarget, n_ipp))
        dist, idx = tree.query(igp_xyz, k=k, distance_upper_bound=self.Rmax_km)
        dist = np.asarray(dist, float).reshape(len(igp_xyz), k)
        idx = np.asarray(idx).reshape(len(igp_xyz), k)
        out: List[Optional[np.ndarray]] = []
        for i in range(len(igp_xyz)):
            if len(within[i]) >= self.Ntarget:
                sel = np.sort(np.asarray(within[i], dtype=np.int64))
            else:
                sel = idx[i][np.isfinite(dist[i])].astype(np.int64)
            out.append(sel if sel.size >= self.Nmin else None)
        return out

    def _info(self, n_points: int, Vgrid: np.ndarray) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "mode": self.kriging_mode,
            "coord_frame": self.coord_frame,
            "n_points": int(n_points),
            "n_estimated": int(np.isfinite(Vgrid).sum()),
        }
        if self.kriging_mode in ("OK", "UK"):
            info["variogram_model"] = self.variogram_model
            info["variogram_parameters"] = _okuk_params_dict(self.variogram_parameters)
            info["coordinates_type"] = self.coordinates_type
        else:
            vario = _variogram_info(self.variogram)
            info["variogram"] = vario
            info["sparks_vario"] = dict(vario)   # legacy key: same content as 'variogram'
        info.update({
            "Rmin_km": float(self.Rmin_km),
            "Rmax_km": float(self.Rmax_km),
            "Ntarget": int(self.Ntarget),
            "Nmin": int(self.Nmin),
        })
        return info

    # -------------- SINGLE EPOCH KRIGING --------------
    def krige_epoch(
        self,
        df_epoch: pd.DataFrame,
        epoch_time: Optional[pd.Timestamp] = None,
        ipp_xyz_override: Optional[np.ndarray] = None,   # (N,3) w km
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

        self._validate_options()   # options may have been changed after construction
        keep = df_epoch[["lat_ipp", "lon_ipp", "vtec"]].notna().all(axis=1).to_numpy()
        dfe = df_epoch.loc[keep].copy()
        lon = dfe["lon_ipp"].to_numpy(float)
        lat = dfe["lat_ipp"].to_numpy(float)
        vtec = dfe["vtec"].to_numpy(float)

        Ny, Nx = self._grid_shape
        Vgrid = np.full((Ny, Nx), np.nan, float)
        SS    = np.full((Ny, Nx), np.nan, float)

        if len(vtec) == 0:
            return Vgrid, SS, self._info(0, Vgrid)

        # --- IPP -> ECEF [km] ---
        override = ipp_xyz_override if ipp_xyz_override is not None else self.ipp_xyz_override
        if override is not None:
            # rows of df_epoch (len(df_epoch)) or rows left after the NaN lat/lon/vtec filter (len(vtec))
            ipp_xyz = np.asarray(override, float)
            if ipp_xyz.ndim != 2 or ipp_xyz.shape[1] != 3:
                raise ValueError("ipp_xyz_override must have shape (N,3) in km.")
            if ipp_xyz.shape[0] == len(df_epoch):
                ipp_xyz = ipp_xyz[keep]
            elif ipp_xyz.shape[0] != len(vtec):
                raise ValueError(f"ipp_xyz_override has {ipp_xyz.shape[0]} rows; expected one per row of df_epoch "
                                 f"({len(df_epoch)}) or per row with finite lat_ipp/lon_ipp/vtec ({len(vtec)})")
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
            return Vgrid, SS, self._info(len(vtec), Vgrid)
        elif self.kriging_mode == "WAAS":
            if meas_var is not None:
                bad = ~np.isfinite(meas_var) | (meas_var < 0)
                if bad.any():
                    raise ValueError(f"meas_var must be finite and >= 0 TECU^2: {int(bad.sum())} of {bad.size} "
                                     "rows are NaN, infinite or negative (fill or drop them before kriging)")
            # PRECOMPUTE
            D = distance.cdist(ipp_xyz, ipp_xyz, metric="euclidean")  # [km]
            C_full = self.variogram.cov_func(D)
            # noise covariance M = diag(meas_var) + s2_rec*[same receiver] + s2_sat*[same satellite]
            # (Sparks et al. 2011; Blanch 2002): the diagonal carries all three terms
            use_rec = (rec_id is not None) and (self.bias_var_rec > 0)
            use_sat = (sat_id is not None) and (self.bias_var_sat > 0)
            M_full = None
            if (meas_var is not None) or use_rec or use_sat:
                M_full = np.diag(meas_var) if meas_var is not None else np.zeros_like(C_full)
                for use, ids, var in ((use_rec, rec_id, self.bias_var_rec), (use_sat, sat_id, self.bias_var_sat)):
                    if use:
                        same = ids.reshape(-1, 1) == ids.reshape(1, -1)
                        na = pd.isna(ids)                  # a missing id (None/NaN) matches no other row
                        same &= ~(na[:, None] | na[None, :])
                        np.fill_diagonal(same, True)   # a row shares its own bias even with a missing id
                        M_full += float(var) * same

            tree = cKDTree(ipp_xyz)
            nodes = self._neighbors_for_igp(tree, self._grid_xyz.reshape(-1, 3), ipp_xyz.shape[0])
            for k, sel in enumerate(nodes):
                if sel is None:
                    continue   # fewer than Nmin IPPs within the WAAS neighbourhood: not estimated
                iy, ix = divmod(k, Nx)
                igp_xyz = self._grid_xyz[iy, ix]            # (3,) km
                igp_latlon = (self._grid_lat[iy, ix], self._grid_lon[iy, ix])
                I_hat, var_hat, _info = waas_kriging_point_fast(
                    ipp_xyz, vtec, igp_xyz, igp_latlon, self.variogram,
                    C_full=C_full, M_full=M_full, idx=sel
                )
                Vgrid[iy, ix] = I_hat
                SS[iy, ix] = var_hat
        else:
            raise ValueError(f"Unknown kriging mode:  {self.kriging_mode}")

        return Vgrid, SS, self._info(len(vtec), Vgrid)




def _times_to_gpst_ns(times, *, who: str) -> np.ndarray:
    """Epochs as naive GPST datetime64[ns]. Naive input is taken as GPST (the GNX convention);
    timezone-aware input is converted to UTC and then to GPST (+ GPS-UTC leap seconds), with one
    UserWarning. NaT raises ValueError; numbers raise TypeError (they would become 1970 epochs),
    as in ``monitoring._to_gpst``."""
    arr = np.asarray(times) if not isinstance(times, (pd.Index, pd.Series)) else times
    if (arr.dtype.kind in "biuf" and arr.size) or (arr.dtype == object and arr.size and any(
            isinstance(t, (int, float, np.integer, np.floating)) and not isinstance(t, (bool, np.bool_))
            and not pd.isna(t) for t in np.ravel(arr))):
        raise TypeError(f"{who}: 'times' are numeric ({arr.dtype}), not datetimes; pass the epochs as naive "
                        "GPST datetimes")
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(times))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{who}: 'times' must be date/time values ({exc})") from None
    if idx.isna().any():
        raise ValueError(f"{who}: incorrect date/time values (NaT) detected in 'times'.")
    if idx.tz is not None:
        from ..core.time import _gps_utc_offset_seconds

        utc = idx.tz_convert("UTC")
        leap = np.array([_gps_utc_offset_seconds(t.to_pydatetime()) for t in utc], dtype=float)
        warnings.warn(f"{who}: timezone-aware 'times' converted to GPST (UTC + "
                      f"{'/'.join(str(int(v)) for v in sorted(set(leap)))} s); GNX epochs are naive GPS time.",
                      UserWarning, stacklevel=3)
        idx = utc.tz_localize(None) + pd.to_timedelta(leap, unit="s")
    return idx.to_numpy(dtype="datetime64[ns]")


def save_grids_to_netcdf(
    times: Sequence[Union[np.datetime64, pd.Timestamp, str]],
    V_all: Union[List[np.ndarray], np.ndarray],
    SS_All: Union[List[np.ndarray], np.ndarray],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    out_path: str = ".",
    out_name:str="vtec",
    var_names=("V", "SS"),
    chunks: Optional[dict] = None,
    compress_level: int = 4,
    dims: Optional[Literal["yx", "xy"]] = None,
) -> xr.Dataset:
    """Save kriged VTEC grids and their variances to ``<out_path>/<out_name>.nc`` (NetCDF4).

    Layout:
      - rectilinear: lat(Ny,), lon(Nx,), data (time, lat, lon)
      - curvilinear: lat(Ny,Nx), lon(Ny,Nx), data (time, y, x)

    ``V_all`` / ``SS_All``: a list of 2D grids, a 2D grid (one epoch), a 3D array (T, ., .)
    or a 4D array with one unit (singleton) dimension. ``dims`` gives the order of the two
    horizontal axes of the input: "yx" = (Ny, Nx), "xy" = (Nx, Ny) (transposed on write).
    When omitted it is inferred from the shape; for a square grid (Ny == Nx) the orientation
    is ambiguous and ``dims`` is required (ValueError otherwise).

    Units: the first variable is VTEC [TECU], the second its kriging variance [TECU^2].
    Time: GNX epochs are naive GPS time; the ``time`` variable carries ``time_scale="GPST"``.
    Timezone-aware ``times`` are converted to UTC and then to GPST (+ GPS-UTC leap seconds),
    with one UserWarning.

    ``out_path`` is the output directory (created if needed); ``compress_level`` is the zlib
    level 0..9; ``var_names`` are two distinct variable names.
    """
    # ---------- options ----------
    if (not isinstance(compress_level, (int, np.integer)) or isinstance(compress_level, (bool, np.bool_))
            or not 0 <= int(compress_level) <= 9):
        raise ValueError(f"compress_level must be an integer zlib compression level in 0..9 "
                         f"(got {compress_level!r})")
    if (not isinstance(var_names, (tuple, list)) or len(var_names) != 2
            or not all(isinstance(v, str) and v for v in var_names)):
        raise ValueError(f"var_names must be two non-empty strings (got {var_names!r})")
    if var_names[0] == var_names[1]:
        raise ValueError(f"var_names must be two different names (got {tuple(var_names)!r}); "
                         "the second variable would overwrite the first")
    if dims is not None:
        _check_choice("dims", dims, ("yx", "xy"))

    # ---------- przygotowanie czasu ----------
    time_vals = _times_to_gpst_ns(times, who="save_grids_to_netcdf")
    T = time_vals.shape[0]

    # ---------- siatka ----------
    # --- grid detection ---
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
        if A.ndim == 4:
            # one unit (singleton) dimension is dropped, e.g. (T, 1, Ny, Nx) or (1, T, Ny, Nx)
            unit = [ax for ax in (1, 0, 2, 3) if A.shape[ax] == 1]
            if not unit:
                raise ValueError(f"{name} is 4D with shape {A.shape}; a 4D input needs one unit (singleton) "
                                 "dimension, e.g. (T, 1, Ny, Nx)")
            A = np.take(A, 0, axis=unit[0])
        if A.ndim == 2:
            A = A[None, ...]
        if A.ndim != 3:
            raise ValueError(f"{name} has {A.ndim} dimensions; expected (T, Ny, Nx), (Ny, Nx) or 4D with one "
                             "unit dimension")
        sh = A.shape[1:]
        order = dims
        if order is None:
            if nlat == nlon:
                if sh == (nlat, nlon):
                    raise ValueError(f"{name}: square grid (Ny == Nx == {nlat}) makes the axis order ambiguous; "
                                     "pass dims='yx' for (..., Ny, Nx) or dims='xy' for (..., Nx, Ny)")
            elif sh == (nlat, nlon):
                order = "yx"
            elif sh == (nlon, nlat):
                order = "xy"
        expected = (nlat, nlon) if order != "xy" else (nlon, nlat)
        if order is None or sh != expected:
            raise ValueError(f"{name} shape {A.shape} does not match (T, Ny, Nx) = (T, {nlat}, {nlon})"
                             + (f" with dims={dims!r}" if dims is not None else ""))
        if order == "xy":
            A = A.transpose(0, 2, 1)
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
        data_dims = ("time", "lat", "lon")
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
        data_dims = ("time", "y", "x")

    ds = xr.Dataset(
        data_vars={
            V_name: (data_dims, V_3d, {"long_name": "vertical total electron content (kriging estimate)",
                                       "units": "TECU",
                                       **({"coordinates": "lat lon"} if grid_kind == "curvi" else {})}),
            SS_name: (data_dims, SS_3d, {"long_name": "kriging variance of VTEC", "units": "TECU^2",
                                         **({"coordinates": "lat lon"} if grid_kind == "curvi" else {})}),
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

    ds["time"].attrs.update(time_scale="GPST", long_name="epoch (GPS time)")

    if chunks:
        ds = ds.chunk(chunks)

    for vn in (V_name, SS_name):
        ds[vn].attrs.pop("_FillValue", None)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f'{out_name}.nc'
    ds.to_netcdf(outfile,mode="w", format="NETCDF4", encoding=enc)
    return ds

def _pair_distance_deg(lon1, lat1, lon2, lat2, coordinates_type):
    """Distance in degrees between broadcast lon/lat arrays (degrees).

    'geographic': great-circle angle (haversine), in degrees of arc.
    'euclidean':  planar sqrt(dlon^2 + dlat^2) in degrees, dlon wrapped to [-180, 180).
    """
    _check_choice("coordinates_type", coordinates_type, OKUK_COORDINATES_TYPES)
    if coordinates_type == "euclidean":
        dx = _wrap180(lon2 - lon1)
        dy = lat2 - lat1
        return np.sqrt(dx * dx + dy * dy)
    lon1, lat1, lon2, lat2 = (np.deg2rad(x) for x in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return np.rad2deg(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a))))


def _dist_matrix(lon, lat, coordinates_type="geographic"):
    """
        The distance matrix d_ij between data points, in degrees.

        coordinates_type:
            'geographic' -> great-circle (angular) distance on the sphere, in degrees (default)
            'euclidean'  -> planar distance in degrees of lon/lat, longitude difference wrapped
                            to [-180, 180)
        """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    return _pair_distance_deg(lon[:, None], lat[:, None], lon[None, :], lat[None, :], coordinates_type)


def _dist_data_to_points(lon_data, lat_data, lon_points, lat_points, coordinates_type="geographic"):
    """
        Distances between data points (n) and target points (m), in degrees
        (see ``_dist_matrix`` for the two coordinate types).

        Returns D with shape (n, m).
        """
    lon_data = np.asarray(lon_data, float)
    lat_data = np.asarray(lat_data, float)
    lon_points = np.asarray(lon_points, float)
    lat_points = np.asarray(lat_points, float)
    return _pair_distance_deg(lon_data[:, None], lat_data[:, None], lon_points[None, :], lat_points[None, :],
                              coordinates_type)


# ---------- variogram gamma(d) ----------
# The variogram models and their parameter names follow PyKrige (BSD 3-Clause licence); PyKrige is not
# included in GNX-py, and this code is written for GNX-py.

def _build_variogram_function(variogram_model, variogram_parameters):
    """
        Variogram models mapping

        variogram_model : {'spherical','exponential','gaussian','linear','power','hole-effect'}
        variogram_parameters :
            - dict: {'sill': s, 'range': r, 'nugget': n}  or {'psill': p, 'range': r, 'nugget': n}
            - list: [sill, range, nugget]  (sill = full sill, NOT partial)
            ``range`` is in degrees (the OK/UK distances are in degrees for both coordinate types).
            A missing 'sill'/'psill' or 'range', or invalid values, raise ValueError.

        Returns:
            gamma(d)  -> semi-variogram
            psill, nugget, var_range
        """
    if isinstance(variogram_parameters, dict) and "nugget" not in variogram_parameters:
        variogram_parameters = {**variogram_parameters, "nugget": 0.0}   # classes: nugget defaults to 0
    p = _okuk_params_dict(variogram_parameters)
    var_range = p["range"]
    nugget = p["nugget"]
    psill = p["psill"] if "psill" in p else p["sill"] - nugget

    models = ("spherical", "exponential", "gaussian", "linear", "power", "hole-effect")
    if not (isinstance(variogram_model, str) and variogram_model.lower() in models):
        raise ValueError(f"Unsupported variogram_model={variogram_model!r}; expected one of {list(models)}")
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
        Base for OK/UK on geographic coordinates (lon, lat) in degrees.

        - works in VARIOGRAM space: gamma(d)
        - kriging matrix has gamma(0)=0 on the diagonal (exact values)
        - distances, always in degrees, so the variogram ``range`` is in degrees:
            'geographic' (default): great-circle distance in degrees of arc;
            'euclidean': planar distance in degrees of lon/lat, with the longitude
            difference wrapped to [-180, 180) (no cos(lat) factor).
        - longitudes are normalised to [-180, 180) on entry (data and target points).
    """

    def __init__(
        self,
        lon,
        lat,
        values,
        *,
        variogram_model="spherical",
        variogram_parameters=None,
        coordinates_type="geographic",
    ):
        """lon/lat in degrees, values (e.g. VTEC in TECU) of the same 1D shape.
        ``variogram_parameters``: {'sill' | 'psill', 'range' [deg], 'nugget'} or [sill, range, nugget].
        ``coordinates_type``: 'geographic' (great-circle degrees, default) or 'euclidean' (planar degrees)."""
        _check_choice("coordinates_type", coordinates_type, OKUK_COORDINATES_TYPES)
        lon = _wrap180(lon)
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

            Status: experimental compatibility implementation used by
            ``IonoKrigingMonitor`` when ``kriging_mode="OK"``.

            OK = OrdinaryKrigingGeo(
                lon, lat, vtec,
                variogram_model="spherical",
                variogram_parameters={"sill": ..., "range": ..., "nugget": ...},
                coordinates_type="geographic",
            )
            z, ss = OK.execute("grid", gx, gy)

            Units: lon/lat and the variogram ``range`` in degrees. 'geographic'
            (default) measures great-circle distance in degrees of arc,
            'euclidean' planar lon/lat degrees with the longitude difference
            wrapped to [-180, 180). Longitudes are normalised to [-180, 180).
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
    Universal Kriging with drift_terms=['regional_linear'] -> basis [1, x, y].

    Status:
        Experimental compatibility implementation used by
        ``IonoKrigingMonitor`` when ``kriging_mode="UK"``. Only regional
        linear drift is supported.

    UK = UniversalKrigingGeo(
        lon, lat, vtec,
        variogram_model="spherical",
        variogram_parameters={"sill": ..., "range": ..., "nugget": ...},
        coordinates_type="geographic",
        drift_terms=["regional_linear"],
    )
    z, ss = UK.execute("grid", gx, gy)

    Units: lon/lat and the variogram ``range`` in degrees ('geographic', the
    default: great-circle degrees; 'euclidean': planar lon/lat degrees with the
    longitude difference wrapped to [-180, 180)).

    Drift: x is the longitude unwrapped around the circular mean of the data
    longitudes and centred on the data mean, y the latitude centred on the data
    mean, so the linear drift is continuous across the antimeridian.
    """

    def __init__(self, lon, lat, values, *, drift_terms=None, **kwargs):
        if drift_terms is None:
            drift_terms = ["regional_linear"]
        elif isinstance(drift_terms, str):
            drift_terms = [drift_terms]
        else:
            drift_terms = list(drift_terms)
        if not drift_terms:
            raise ValueError("drift_terms must not be empty; use ['regional_linear'] (or None for the default)")
        if drift_terms != ["regional_linear"]:
            raise NotImplementedError(
                f"UniversalKrigingGeo currently only supports drift_terms=['regional_linear'], "
                f"received {drift_terms!r}"
            )
        self.drift_terms = drift_terms
        super().__init__(lon, lat, values, **kwargs)
        self._build_drift()
        self._build_system()

    def _drift_xy(self, lon, lat):
        """Drift coordinates: unwrapped, centred longitude and centred latitude [deg]."""
        x = _wrap180(np.asarray(lon, float) - self._lon_ref) - self._x0
        y = np.asarray(lat, float) - self._y0
        return x, y

    def _build_drift(self):
        lon = self.lon
        lat = self.lat
        # reference longitude: circular mean of the data (continuous across +-180 deg)
        lr = np.deg2rad(lon)
        self._lon_ref = float(np.rad2deg(np.arctan2(np.mean(np.sin(lr)), np.mean(np.cos(lr))))) if lon.size else 0.0
        self._x0 = 0.0
        self._y0 = 0.0
        x, y = self._drift_xy(lon, lat)
        self._x0 = float(np.mean(x)) if x.size else 0.0
        self._y0 = float(np.mean(y)) if y.size else 0.0
        x, y = self._drift_xy(lon, lat)

        F = np.column_stack([np.ones_like(lon), x, y])
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
        x, y = self._drift_xy(float(lon_p), float(lat_p))
        return np.array([1.0, float(x), float(y)], dtype=float)

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
