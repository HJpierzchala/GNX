"""Non-plotting helpers for the ionosphere tutorial notebooks.

Plots and table previews in the tutorial come from :mod:`gnx_py.viz`. This
module only keeps the computational helper that GNX-py does not ship yet:
:class:`VTECZenithEstimator`, a rough zenith-VTEC estimate above a station.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


@dataclass
class VTECZenithEstimator:
    """
    Simple estimate of the VTEC above a station (zenith) from STEC observations.

    Required columns of df:
      - time : datetime64[ns]
      - ev   : elewacja satelity [deg]
      - leveled_tec : STEC [TECU] (after the corrections, DCB and levelling)

    Optional:
      - sv : satellite identifier (useful for diagnostics)
      - az : azimuth [deg] (not used here)

    Steps:
      1) VTEC = STEC / M(e), M(e) the obliquity factor for h_ion_m.
      2) At each epoch, the satellite with max(ev) >= min_elev_deg.
      3) The VTEC_zenith time series, linearly interpolated in time and smoothed.
    """
    df: pd.DataFrame
    time_col: str = "time"
    elev_col: str = "ev"
    stec_col: str = "leveled_tec"
    sv_col: Optional[str] = "sv"
    h_ion_m: float = 450e3
    R_earth_m: float = 6371e3
    min_elev_deg: float = 30.0  # very low paths are ignored
    resample_freq: Optional[str] = None  # e.g. '30S'; None interpolates on the irregular epochs
    rolling_median_win: Optional[int] = None  # e.g. 3; light smoothing after the interpolation
    # smooth_method: 'median' | 'ema' | 'savgol' | 'whittaker'

    smooth_method: str = "whittaker"
    smooth_params: dict = field(default_factory=dict)

    # ----------------------------- core geometry -----------------------------
    def _obliquity_factor(self, elev_deg: np.ndarray) -> np.ndarray:
        """
        M(e) = 1 / cos(z'), cos(z') = sqrt(1 - (Re/(Re+h))^2 * cos^2(e))
        """
        e_rad = np.deg2rad(elev_deg)
        rho = self.R_earth_m / (self.R_earth_m + self.h_ion_m)
        cosz_prime_sq = 1.0 - (rho ** 2) * (np.cos(e_rad) ** 2)
        # numerical safeguard
        cosz_prime_sq = np.clip(cosz_prime_sq, 1e-8, None)
        M = 1.0 / np.sqrt(cosz_prime_sq)
        return M

    # ----------------------------- pipeline ---------------------------------
    def compute_vtec_per_obs(self) -> pd.DataFrame:
        """Add the columns 'M' and 'vtec' to a copy of the input frame."""
        df = self.df.copy()
        # mask of the valid values
        m = np.isfinite(df[self.stec_col]) & np.isfinite(df[self.elev_col])
        df.loc[~m, [self.stec_col, self.elev_col]] = np.nan

        M = self._obliquity_factor(df[self.elev_col].to_numpy())
        df["M"] = M
        df["vtec"] = df[self.stec_col] / M  # [TECU]
        return df

    def pick_nearest_zenith(self, df_v: pd.DataFrame) -> pd.DataFrame:
        """
        At each epoch, pick the record with the highest elevation (>= min_elev_deg).
        Works for:
          - a MultiIndex with a 'time' level (e.g. index = ['sv','time'])
          - a plain index with a 'time' column
        """
        elev = self.elev_col
        tcol = self.time_col

        # --- MultiIndex with a 'time' level ---
        if isinstance(df_v.index, pd.MultiIndex) and (tcol in df_v.index.names):
            # index labels of the rows with the highest elevation at each epoch
            idx = (
                df_v[df_v[elev].notna()]
                .groupby(level=tcol)[elev]
                .idxmax()
                .dropna()
            )
            # the list of tuples serves as the .loc labels
            chosen = df_v.loc[list(idx)].copy().reset_index()  # 'sv' and 'time' become columns again

            # number of satellites above the threshold per epoch (MultiIndex: grouped by level=time)
            cnt = (
                df_v.loc[df_v[elev] >= self.min_elev_deg]
                .groupby(level=tcol)[elev]
                .size()
                .rename("n_sat_ge_thr").astype(int)
            )

            chosen = chosen.merge(cnt, left_on=tcol, right_index=True, how="left")
            chosen = chosen.rename(columns={"n_sat_ge_thr": "nsat"})

        # --- Plain DataFrame (time as a column) ---
        else:
            # make sure that 'time' is a column
            if tcol not in df_v.columns:
                if df_v.index.name == tcol:
                    df_v = df_v.reset_index()
                else:
                    raise ValueError(f"No column or index level '{tcol}' in df_v.")
            tmp = (
                df_v[df_v[elev].notna()]
                .sort_values([tcol, elev], ascending=[True, False])
                .groupby(tcol, as_index=False)
                .head(1)
                .copy()
                .reset_index(drop=True)
            )
            cnt = (
                df_v.loc[df_v[elev] >= self.min_elev_deg]
                .groupby(tcol)[elev]
                .size()
                .rename("nsat").astype(int)
            )
            chosen = tmp.merge(cnt, left_on=tcol, right_index=True, how="left")

        # elevation threshold: a sample below min_elev_deg is not trusted
        chosen.loc[chosen[elev] < self.min_elev_deg, "vtec"] = np.nan

        # key columns
        cols = [self.time_col, "vtec", self.elev_col, "M", "nsat"]
        if self.sv_col and self.sv_col in chosen.columns:
            cols.insert(1, self.sv_col)

        # sort by time
        chosen[self.time_col] = pd.to_datetime(chosen[self.time_col], errors="coerce")
        return chosen[cols].sort_values(self.time_col)


    def interpolate_timeseries(self, ts: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolation in time and smoothing by self.smooth_method.
        """
        out = ts.copy()

        # make sure that 'time' is a column
        if self.time_col not in out.columns:
            if isinstance(out.index, pd.MultiIndex) and self.time_col in out.index.names:
                out = out.reset_index(self.time_col)
            elif out.index.name == self.time_col:
                out = out.reset_index()
            else:
                raise ValueError(f"No column '{self.time_col}' in ts.")

        out[self.time_col] = pd.to_datetime(out[self.time_col], errors="coerce")
        out = out.set_index(self.time_col).sort_index()

        # regular time grid (optional)
        if self.resample_freq:
            num_cols = out.select_dtypes(include="number").columns
            out = out[num_cols].resample(self.resample_freq).mean()

        # linear interpolation over the gaps (in time)
        if "vtec" in out:
            out["vtec"] = out["vtec"].interpolate(method="time", limit_direction="both")

        # --- smoothing ---
        y = out["vtec"].to_numpy()
        y_s = self._apply_smoothing(out.index, y, method=getattr(self, "smooth_method", "savgol"),
                                    params=getattr(self, "smooth_params", {}))
        out["vtec_smooth"] = y_s

        return out.reset_index()

    def _apply_smoothing(self, t_index: pd.Index, y: np.ndarray,
                         method: str = "savgol", params: dict = None) -> np.ndarray:
        """
        Common entry point of the smoothing methods. NaNs are filled first.
        Methods:
          - 'median'    : window of N samples (param 'window')
          - 'ema'       : two-sided EWM (param 'alpha', 'span' or 'halflife')
          - 'savgol'    : Savitzky-Golay (params 'window', 'polyorder')
          - 'whittaker' : Eilers' smoother (params 'lam', 'order' [1 or 2])
        """
        params = params or {}
        y = y.astype(float)
        # fill any NaNs by simple interpolation
        if np.isnan(y).any():
            s = pd.Series(y, index=t_index).interpolate(method="time", limit_direction="both")
            y = s.to_numpy()

        n = len(y)
        if n < 5:
            return y  # too few samples: unchanged

        # time step estimate [s], used to choose a window
        try:
            t_sec = pd.to_datetime(t_index).view("int64") / 1e9
            dt = np.median(np.diff(t_sec))
            if not np.isfinite(dt) or dt <= 0:
                dt = None
        except Exception:
            dt = None

        method = (method or "savgol").lower()

        if method == "median":
            win = int(params.get("window", 7))
            win = max(1, win)
            s = pd.Series(y).rolling(win, center=True, min_periods=1).median().to_numpy()
            return s

        elif method == "ema":
            # two-sided EMA (forward + backward) reduces the phase shift
            if "alpha" in params:
                alpha = float(params["alpha"])
            elif "span" in params:
                alpha = 2.0 / (float(params["span"]) + 1.0)
            elif "halflife" in params:
                hl = float(params["halflife"])
                alpha = 1 - np.exp(-np.log(2) / hl)
            else:
                # default: a time constant of about 10 min when dt is known
                if dt is not None:
                    tau = params.get("tau_seconds", 600.0)
                    alpha = 1 - np.exp(-dt / float(tau))
                else:
                    alpha = 0.2  # dt unknown
            # forward
            yf = np.copy(y)
            for i in range(1, n):
                yf[i] = alpha * y[i] + (1 - alpha) * yf[i - 1]
            # backward
            yb = np.copy(y)
            for i in range(n - 2, -1, -1):
                yb[i] = alpha * yb[i] + (1 - alpha) * yb[i + 1]
            return 0.5 * (yf + yb)

        elif method == "savgol":
            # window: when not given, about 10 min / dt (or 11 as a safe value)
            poly = int(params.get("polyorder", 3))
            if "window" in params:
                win = int(params["window"])
            else:
                if dt is not None:
                    target_sec = float(params.get("window_seconds", 600.0))  # 10 min
                    win = max(5, int(round(target_sec / dt)))
                else:
                    win = 11
            # the window must be odd and <= n
            if win % 2 == 0:
                win += 1
            if win > n:
                win = n - (1 - n % 2)  # largest odd number <= n
            win = max(win, poly + 2 | 1)  # > polyorder and odd
            return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")


        elif method == "whittaker":

            order = int(params.get("order", 2))

            order = max(1, min(order, len(y) - 1))

            lam = float(params.get("lam", 2000.0))

            return self._whittaker(y, lam=lam, order=order)


        else:
            # unknown method: unchanged
            return y

    def _whittaker(self, y: np.ndarray, lam: float = 1000.0, order: int = 2) -> np.ndarray:
        """
        Whittaker–Eilers smoother: min ||y - z||^2 + lam * ||D^d z||^2
        D (the d-th difference) is built step by step with decreasing dimensions.
        """
        n = len(y)
        if n < 3:
            return y

        # valid range of the order
        order = max(1, min(int(order), n - 1))

        E = sparse.eye(n, format="csc")

        # first difference: (n-1) x n
        D = sparse.diags([1, -1], [0, 1], shape=(n - 1, n), format="csc", dtype=float)

        # higher orders: each difference acts on the previous one
        for k in range(1, order):
            rows, cols = D.shape  # after the k-th iteration D is (n-k, n)
            Di = sparse.diags([1, -1], [0, 1], shape=(rows - 1, rows), format="csc", dtype=float)
            D = Di @ D  # finally D is (n - order, n)

        A = E + lam * (D.T @ D)
        z = spsolve(A, y)
        return np.asarray(z, dtype=float)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          df_v : frame with the VTEC of every observation (columns ..., 'M', 'vtec')
          zen  : VTEC time series above the station (satellite nearest the zenith, interpolated)
        """
        df_v = self.compute_vtec_per_obs()
        zen_raw = self.pick_nearest_zenith(df_v)
        zen = self.interpolate_timeseries(zen_raw)
        return df_v, zen
