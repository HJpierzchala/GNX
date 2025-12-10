import numpy as np
from scipy.linalg import cho_factor, cho_solve


# ---------- odległości jak w PyKrige (domyślnie "euclidean") ----------

def _dist_matrix(lon, lat, coordinates_type="euclidean"):
    """
    Macierz odległości d_ij między punktami danych.

    coordinates_type:
        'euclidean'  -> jak PyKrige domyślnie: odległość euklidesowa w stopniach
        'geographic' -> odległość kątowa na sferze [w radianach]
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)

    if coordinates_type == "euclidean":
        dx = lon[:, None] - lon[None, :]
        dy = lat[:, None] - lat[None, :]
        return np.sqrt(dx * dx + dy * dy)

    elif coordinates_type == "geographic":
        # central angle (rad), R=1 – zgodnie z krigingiem „geometric”
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
        raise ValueError(f"Nieznany coordinates_type={coordinates_type!r}")


def _dist_data_to_points(lon_data, lat_data, lon_points, lat_points, coordinates_type="euclidean"):
    """
    Odległości między punktami danych (n) i punktami celu (m).

    Zwraca D o kształcie (n, m).
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
        raise ValueError(f"Nieznany coordinates_type={coordinates_type!r}")


# ---------- wariogram jak w PyKrige: gamma(d) ----------

def _build_variogram_function(variogram_model, variogram_parameters):
    """
    Dokładne odwzorowanie modeli wariogramu z dokumentacji PyKrige.

    variogram_model : {'spherical','exponential','gaussian','linear','power','hole-effect'}
    variogram_parameters :
        - dict: {'sill': s, 'range': r, 'nugget': n}  lub {'psill': p, 'range': r, 'nugget': n}
        - list: [sill, range, nugget]  (sill = pełny sill, NIE partial)

    Zwraca:
        gamma(d)  -> semiwariogram
        psill, nugget, var_range
    """
    if variogram_parameters is None:
        raise ValueError("variogram_parameters musi być podane (sill/psill, range, nugget).")

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
            raise ValueError(f"Nieobsługiwany variogram_model={variogram_model!r}")

    return gamma, psill, nugget, var_range


# ---------- baza dla OK/UK ----------

class _BaseKrigingGeo:
    """
    Baza dla OK/UK na współrzędnych geograficznych (lon, lat) w stylu PyKrige.

    - pracuje w przestrzeni WARIOGRAMU: gamma(d)
    - macierz krigingowa ma na przekątnej gamma(0)=0 (dokładne wartości)
    - distances: domyślnie 'euclidean' (jak PyKrige)
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
            raise ValueError("lon, lat, values muszą mieć ten sam kształt 1D.")

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

        # Macierz semiwariogramu Γ
        D = _dist_matrix(lon, lat, coordinates_type=self.coordinates_type)
        Gamma = self.gamma(D)

        # PyKrige: diagonalę kriging matrix wymusza na 0 -> exact values
        np.fill_diagonal(Gamma, 0.0)

        self.Gamma = Gamma  # (n, n)
        self._cho = None
        self._A_pinv = None
        self._n_aug = None


# ---------- Ordinary Kriging ----------

class OrdinaryKrigingGeo(_BaseKrigingGeo):
    """
    Ordinary Kriging w wariancie jak PyKrige.ok.OrdinaryKriging (2D, grid).

        OK = OrdinaryKrigingGeo(
            lon, lat, vtec,
            variogram_model="spherical",
            variogram_parameters={"sill": ..., "range": ..., "nugget": ...},
            coordinates_type="euclidean",   # ważne dla zgodności z PyKrige
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
        Rozwiązujemy:
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
            raise NotImplementedError("OrdinaryKrigingGeo.execute wspiera tylko style='grid'.")

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
    Universal Kriging w stylu PyKrige.uk.UniversalKriging
    z drift_terms=['regional_linear'] -> baza [1, lon, lat].

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
                f"UniversalKrigingGeo aktualnie wspiera tylko drift_terms=['regional_linear'], "
                f"otrzymano {self.drift_terms!r}"
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
        Rozwiązujemy:
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
            raise NotImplementedError("UniversalKrigingGeo.execute wspiera tylko style='grid'.")

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
