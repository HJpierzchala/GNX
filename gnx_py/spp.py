from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Literal, Set, Final, Any
from pathlib import Path
from datetime import datetime
from .configuration import Config
import pandas as pd
import logging
import numpy as np
from .utils import calculate_distance
from .conversion import ecef_to_enu, ecef2geodetic
from .coordinates import BroadcastInterp, lagrange_emission_interp, lagrange_reception_interp, CustomWrapper, make_ionofree
from .io import GNSSDataProcessor2, read_sp3, parse_sinex
from . import DDPreprocessing
from .session_errors import guarded_session_run


@dataclass(slots=True)
class SPPResult:
    """Container holding PPP solution and auxiliary data."""
    solution: Union["pd.DataFrame",None]  =None
    residuals_gps: Union["pd.DataFrame",None] = None
    residuals_gal: Union["pd.DataFrame",None] = None
    covariance_info: Union["pd.DataFrame",None] = None

    # Convenience exporters --------------------------------------------------
    def to_netcdf(self, path: Path, **kwargs: Any) -> None:
        if not path.suffix:
            path = path.with_suffix(".nc")
        self.solution.to_xarray().to_netcdf(path, **kwargs)

@dataclass()
class SPPConfig(Config):
    trace_filter: Optional[bool] = False
    solver: Literal["LS"] ="LS"



class SinglePointPositioning:

    def __init__(self, config, gps_obs: pd.DataFrame, gal_obs: Union[pd.DataFrame, None], gps_mode, gal_mode,
                 solver='LSQ', xyz_apr=np.zeros(3)):
        self.config = config
        self.gps_obs = gps_obs
        self.gal_obs = gal_obs
        self.gps_mode = gps_mode
        self.gal_mode = gal_mode
        self.solver = solver
        self.FREQ_DICT = {'L1': 1575.42e06, 'E1': 1575.42e06,
                          'L2': 1227.60e06, 'E5a': 1176.45e06,
                          'L5': 1176.450e06, 'E5b': 1207.14e06}
        self.xyz_apr = xyz_apr
        self._if_coeff = {
            'L1L2': self._iono_free_coeff('L1', 'L2'),
            'L1L5': self._iono_free_coeff('L1', 'L5'),
            'L2L5': self._iono_free_coeff('L2', 'L5'),
            'E1E5a': self._iono_free_coeff('E1', 'E5a'),
            'E1E5b': self._iono_free_coeff('E1', 'E5b'),
            'E5aE5b': self._iono_free_coeff('E5a', 'E5b'),
        }

    def _iono_free_coeff(self, f1_name: str, f2_name: str) -> Tuple[float, float]:
        f1 = self.FREQ_DICT[f1_name]
        f2 = self.FREQ_DICT[f2_name]
        den = (f1 ** 2 - f2 ** 2)
        return (f1 ** 2) / den, (f2 ** 2) / den

    @staticmethod
    def _first_matching_col(columns: pd.Index, prefixes: List[str]) -> Optional[str]:
        for p in prefixes:
            for c in columns:
                if c.startswith(p):
                    return c
        return None

    @staticmethod
    def _series_or_zeros(df: pd.DataFrame, name: str, n: int) -> np.ndarray:
        if name in df.columns:
            return df[name].to_numpy(dtype='float64', copy=False)
        return np.zeros(n, dtype='float64')

    @staticmethod
    def _first_existing_or_zeros(df: pd.DataFrame, names: List[str], n: int) -> np.ndarray:
        for ncol in names:
            if ncol in df.columns:
                return df[ncol].to_numpy(dtype='float64', copy=False)
        return np.zeros(n, dtype='float64')

    def _build_obs_base(self, epoch: Optional[pd.DataFrame], system: str, mode: Optional[str]) -> Optional[np.ndarray]:
        if epoch is None or len(epoch) == 0 or mode is None:
            return None

        C = 299_792_458.0
        n = len(epoch)
        cols = epoch.columns

        if system == 'G':
            if mode == 'L1':
                c1 = self._first_matching_col(cols, ['C1C', 'C1W', 'C1', 'C1P'])
                if c1 is None:
                    raise ValueError("GPS L1: column C1* missing in gps_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                TGD_s = self._first_existing_or_zeros(epoch, ['TGD'], n)
                pco_l1 = self._first_existing_or_zeros(epoch, ['pco_los_l1'], n)
                return P - tro - ion - dprel + C * (clk_s - TGD_s) + pco_l1
            if mode == 'L2':
                c1 = self._first_matching_col(cols, ['C2W', 'C2X', 'C2', 'C2P'])
                if c1 is None:
                    raise ValueError("GPS L2: column C2* missing in gps_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                TGD_s = self._first_existing_or_zeros(epoch, ['TGD'], n)
                pco_l2 = self._first_existing_or_zeros(epoch, ['pco_los_l2'], n)
                return P - tro - ion - dprel + C * (clk_s - TGD_s) + pco_l2
            if mode == 'L5':
                c1 = self._first_matching_col(cols, ['C5X', 'C5'])
                if c1 is None:
                    raise ValueError("GPS L5: column C5* missing in gps_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                TGD_s = self._first_existing_or_zeros(epoch, ['TGD'], n)
                pco_l5 = self._first_existing_or_zeros(epoch, ['pco_los_l5'], n)
                return P - tro - ion - dprel + C * (clk_s - TGD_s) + pco_l5
            if mode == 'L1L2':
                c1 = self._first_matching_col(cols, ['C1C', 'C1W', 'C1', 'C1P'])
                c2 = self._first_matching_col(cols, ['C2W', 'C2L', 'C2P', 'C2'])
                if c1 is None or c2 is None:
                    raise ValueError("GPS L1L2: no C1* or C2* columns in gps_epoch.")
                k1, k2 = self._if_coeff['L1L2']
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P2 = epoch[c2].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                pco_l1 = self._first_existing_or_zeros(epoch, ['pco_los_l1'], n)
                pco_l2 = self._first_existing_or_zeros(epoch, ['pco_los_l2'], n)
                pco = (k1 * pco_l1 - k2 * pco_l2)
                return (k1 * P1 - k2 * P2) + C * clk_s - tro - dprel + pco
            if mode == 'L1L5':
                c1 = self._first_matching_col(cols, ['C1'])
                c2 = self._first_matching_col(cols, ['C5'])
                if c1 is None or c2 is None:
                    raise ValueError("GPS L1L5: no C1* or C5* columns in gps_epoch.")
                k1, k2 = self._if_coeff['L1L5']
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P2 = epoch[c2].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                pco_l1 = self._first_existing_or_zeros(epoch, ['pco_los_l1'], n)
                pco_l5 = self._first_existing_or_zeros(epoch, ['pco_los_l5'], n)
                pco = (k1 * pco_l1 - k2 * pco_l5)
                return (k1 * P1 - k2 * P2) + C * clk_s - tro - dprel + pco
            if mode == 'L2L5':
                c1 = self._first_matching_col(cols, ['C2'])
                c2 = self._first_matching_col(cols, ['C5'])
                if c1 is None or c2 is None:
                    raise ValueError("GPS L2L5: no C2* or C5* columns in gps_epoch.")
                k1, k2 = self._if_coeff['L2L5']
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P2 = epoch[c2].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                pco_l2 = self._first_existing_or_zeros(epoch, ['pco_los_l2'], n)
                pco_l5 = self._first_existing_or_zeros(epoch, ['pco_los_l5'], n)
                pco = (k1 * pco_l2 - k2 * pco_l5)
                return (k1 * P1 - k2 * P2) + C * clk_s - tro - dprel + pco
            raise ValueError(f"Unknown gps_mode: {mode}")

        if system == 'E':
            if mode == 'E1':
                c1 = self._first_matching_col(cols, ['C1C', 'C1X', 'C1A', 'C1'])
                if c1 is None:
                    c1 = self._first_matching_col(cols, ['C1B'])
                if c1 is None:
                    raise ValueError("GAL E1: column C1* missing in gal_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                BGD_s = self._first_existing_or_zeros(
                    epoch, ['BGD', 'BGDe5a', 'BGDe5b', 'BGD_E1E5a', 'BGD_E1E5b'], n
                )
                return P - tro - ion - dprel + C * (clk_s - BGD_s)
            if mode == 'E5a':
                c1 = self._first_matching_col(cols, ['C5Q', 'C5X', 'C5'])
                if c1 is None:
                    raise ValueError("GAL E5a: column C5* missing in gal_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                BGD_s = self._first_existing_or_zeros(epoch, ['BGDe5a'], n)
                return P - tro - ion - dprel + C * (clk_s - BGD_s)
            if mode == 'E5b':
                c1 = self._first_matching_col(cols, ['C7'])
                if c1 is None:
                    raise ValueError("GAL E5b: column C7* missing in gal_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                BGD_s = self._first_existing_or_zeros(epoch, ['BGDe5b'], n)
                return P - tro - ion - dprel + C * (clk_s - BGD_s)
            if mode == 'E1E5a':
                c1 = self._first_matching_col(cols, ['C1C', 'C1X', 'C1W', 'C1'])
                c5 = self._first_matching_col(cols, ['C5Q', 'C5X', 'C5', 'C5A'])
                if c1 is None or c5 is None:
                    raise ValueError("GAL E1E5a: columns C1* or C5* are missing in gal_epoch.")
                k1, k2 = self._if_coeff['E1E5a']
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P5 = epoch[c5].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                return (k1 * P1 - k2 * P5) + C * clk_s - tro - dprel
            if mode == 'E1E5b':
                c1 = self._first_matching_col(cols, ['C1C', 'C1X', 'C1W', 'C1'])
                c5 = self._first_matching_col(cols, ['C7Q', 'C7X', 'C7'])
                if c1 is None or c5 is None:
                    raise ValueError("GAL E1E5b: columns C1* or C7* are missing in gal_epoch.")
                k1, k2 = self._if_coeff['E1E5b']
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P5 = epoch[c5].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                return (k1 * P1 - k2 * P5) + C * clk_s - tro - dprel
            if mode == 'E5aE5b':
                c1 = self._first_matching_col(cols, ['C5'])
                c5 = self._first_matching_col(cols, ['C7'])
                if c1 is None or c5 is None:
                    raise ValueError("GAL E5aE5b: columns C5* or C7* are missing in gal_epoch.")
                k1, k2 = self._if_coeff['E5aE5b']
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P5 = epoch[c5].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                return (k1 * P1 - k2 * P5) + C * clk_s - tro - dprel
            raise ValueError(f"Unknown gal_mode: {mode}")

        raise ValueError(f"Unknown system: {system}")


    def hjacobian(self, x, gps_sats=None, gal_sats=None):
        """
        Jacobian for SPP LSQ.

        State convention:
          - single-system: [x, y, z, dtr]              -> shape (N, 4)
          - dual-system  : [x, y, z, dtr_gps, dtr_gal] -> shape (N_gps+N_gal, 5)

        Row ordering in the dual-system case: GPS rows first, then Galileo rows.
        """

        has_gps = gps_sats is not None and len(gps_sats) > 0
        has_gal = gal_sats is not None and len(gal_sats) > 0

        if not has_gps and not has_gal:
            return np.zeros((0, 4), dtype="float64")

        # Decide parameter dimension
        n_params = 5 if (has_gps and has_gal) else 4
        n_rows = (len(gps_sats) if has_gps else 0) + (len(gal_sats) if has_gal else 0)

        A = np.zeros((n_rows, n_params), dtype="float64")

        row0 = 0

        # --- GPS block
        if has_gps:
            dxyz = x[:3] - gps_sats  # (N,3)
            dnorm = np.linalg.norm(dxyz, axis=1)  # (N,)
            A[row0:row0 + len(gps_sats), 0:3] = dxyz / dnorm[:, None]

            if n_params == 4:
                A[row0:row0 + len(gps_sats), 3] = 1.0  # dtr
            else:
                A[row0:row0 + len(gps_sats), 3] = 1.0  # dtr_gps
                # A[:,4] stays 0 for GPS rows

            row0 += len(gps_sats)

        # --- Galileo block
        if has_gal:
            dxyz = x[:3] - gal_sats
            dnorm = np.linalg.norm(dxyz, axis=1)
            A[row0:row0 + len(gal_sats), 0:3] = dxyz / dnorm[:, None]

            if n_params == 4:
                A[row0:row0 + len(gal_sats), 3] = 1.0  # dtr (single-system)
            else:
                # A[:,3] stays 0 for GAL rows
                A[row0:row0 + len(gal_sats), 4] = 1.0  # dtr_gal

        return A

    def hx(self, x, gps_sats=None, gal_sats=None):
        """
        Modelled pseudorange (geometry + receiver clock term(s)).

        State convention:
          - single-system: [x, y, z, dtr]
          - dual-system  : [x, y, z, dtr_gps, dtr_gal]

        Output ordering in dual-system: GPS first then Galileo (must match hjacobian()).
        """

        has_gps = gps_sats is not None and len(gps_sats) > 0
        has_gal = gal_sats is not None and len(gal_sats) > 0

        if not has_gps and not has_gal:
            return np.zeros((0,), dtype="float64")

        dual = has_gps and has_gal

        out = []

        # --- GPS
        if has_gps:
            dxyz = x[:3] - gps_sats
            dist = np.linalg.norm(dxyz, axis=1)

            if dual:
                dist = dist + x[3]  # dtr_gps
            else:
                dist = dist + x[3]  # dtr (single-system)

            out.append(dist)

        # --- GAL
        if has_gal:
            dxyz = x[:3] - gal_sats
            dist = np.linalg.norm(dxyz, axis=1)

            if dual:
                dist = dist + x[4]  # dtr_gal
            else:
                dist = dist + x[3]  # dtr (single-system)

            out.append(dist)

        return np.concatenate(out) if len(out) > 1 else out[0]

    @staticmethod
    def _weighted_lsq_solution(A: np.ndarray, L: np.ndarray, err: np.ndarray) -> np.ndarray:
        """
        Weighted least squares solution with diagonal weights stored as variances in err.
        Returns X for the system (A^T W A) X = A^T W L without forming W.
        """
        if A.size == 0:
            return np.zeros(0)
        w_sqrt = 1.0 / np.sqrt(err)
        Aw = A * w_sqrt[:, None]
        Lw = L * w_sqrt
        return np.linalg.lstsq(Aw, Lw, rcond=None)[0]

    @staticmethod
    def _weighted_lsq_covariance(A: np.ndarray, err: np.ndarray) -> np.ndarray:
        """
        Weighted least squares covariance for diagonal weights stored as variances in err.
        Returns Q = (A^T W A)^-1 without forming W.
        """
        if A.size == 0:
            return np.zeros((0, 0))
        w_sqrt = 1.0 / np.sqrt(err)
        Aw = A * w_sqrt[:, None]
        return np.linalg.pinv(Aw.T @ Aw)

    def code_screening(self, omc, err, n_sigma=2):
        md = np.median(omc)
        mask = np.where(np.abs(omc) > n_sigma * md)
        if len(mask[0]) >= len(omc) / 2:
            for ind in mask:
                err[ind] *= 3
        return err

    def observed(self, gps_epoch, gal_epoch):
        """
                Builds 'observed' vectors (metres) for GPS and Galileo depending on the mode.
                Assumptions regarding column units:
                  - C1*/C2*/C5* pseudo-ranges: metres
                  - tro, ion, dprel: metres (if you have any in seconds, convert them beforehand)
                  - clk, TGD, BGD*: seconds  (clk understood as dt_r - dt_s)
                Clock convention:
                  - we assume clk = dt_r - dt_s  ⇒  we add +c*(clk - TGD/BGD)
        """
        C = 299_792_458.0

        def first_col(df, prefixes):
            """Return the name of the first existing column starting with a given prefix."""
            for p in prefixes:
                for c in df.columns:
                    if c.startswith(p):
                        return c
            return None

        def get_arr(df, name, default=0.0, dtype='float64'):
            """Get the column as an np.ndarray; if missing, fill with a constant."""
            if name in df.columns:
                return np.asarray(df[name].values, dtype=dtype)
            # brak kolumny: zwróć stałą o długości N
            N = len(df)
            return np.full(N, default, dtype=dtype)

        def get_any(df, names, default=0.0):
            """Get the first one from the list of possible columns; if there are none, return 0."""
            for n in names:
                if n in df.columns:
                    return np.asarray(df[n].values, dtype='float64')
            return np.full(len(df), default, dtype='float64')

        # ---------- GPS ----------
        gps_obs = None
        if gps_epoch is not None and len(gps_epoch) > 0:
            if self.gps_mode == 'L1':
                c1 = first_col(gps_epoch, ['C1C', 'C1W', 'C1', 'C1P'])
                if c1 is None:
                    raise ValueError("GPS L1: column C1* missing in gps_epoch.")
                P = get_arr(gps_epoch, c1)

                tro = get_any(gps_epoch, ['tro'], 0.0)
                ion = get_any(gps_epoch, ['ion'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                TGD_s = get_any(gps_epoch, ['TGD'], 0.0)  # s
                dtr = get_any(gps_epoch, ['dtr'], 0.0)
                pco_l1 = get_any(gps_epoch, ['pco_los_l1'], 0.0)


                clk_term_m = C * (clk_s - TGD_s)
                gps_obs = P - tro - ion - dprel + clk_term_m - dtr + pco_l1
            elif self.gps_mode == 'L2':
                c1 = first_col(gps_epoch, ['C2W', 'C2X', 'C2', 'C2P'])
                if c1 is None:
                    raise ValueError("GPS L1: column C1* missing in gps_epoch.")
                P = get_arr(gps_epoch, c1)

                tro = get_any(gps_epoch, ['tro'], 0.0)
                ion = get_any(gps_epoch, ['ion'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                TGD_s = get_any(gps_epoch, ['TGD'], 0.0)  # s
                dtr = get_any(gps_epoch, ['dtr'], 0.0)
                pco_l1 = get_any(gps_epoch, ['pco_los_l2'], 0.0)


                clk_term_m = C * (clk_s - TGD_s)
                gps_obs = P - tro - ion - dprel + clk_term_m - dtr + pco_l1

            elif self.gps_mode == 'L5':
                c1 = first_col(gps_epoch, ['C5X', 'C5'])
                if c1 is None:
                    raise ValueError("GPS L1: column C1* missing in gps_epoch.")
                P = get_arr(gps_epoch, c1)

                tro = get_any(gps_epoch, ['tro'], 0.0)
                ion = get_any(gps_epoch, ['ion'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                TGD_s = get_any(gps_epoch, ['TGD'], 0.0)  # s
                dtr = get_any(gps_epoch, ['dtr'], 0.0)
                pco_l1 = get_any(gps_epoch, ['pco_los_l5'], 0.0)

                clk_term_m = C * (clk_s - TGD_s)
                gps_obs = P - tro - ion - dprel + clk_term_m - dtr + pco_l1


            elif self.gps_mode == 'L1L2':
                # IF(L1,L2): k1*C1 - k2*C2  + c*clk  - tro  - dprel
                c1 = first_col(gps_epoch, ['C1C', 'C1W', 'C1', 'C1P'])
                c2 = first_col(gps_epoch, ['C2W', 'C2L', 'C2P', 'C2'])
                if c1 is None or c2 is None:
                    raise ValueError("GPS L1L2: no C1* or C2* columns in gps_epoch.")

                f1 = self.FREQ_DICT['L1']
                f2 = self.FREQ_DICT['L2']
                k1 = (f1 ** 2) / (f1 ** 2 - f2 ** 2)
                k2 = (f2 ** 2) / (f1 ** 2 - f2 ** 2)

                P1 = get_arr(gps_epoch, c1)
                P2 = get_arr(gps_epoch, c2)
                tro = get_any(gps_epoch, ['tro'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                pco_l1 = get_any(gps_epoch, ['pco_los_l1'], 0.0)
                pco_l2 = get_any(gps_epoch, ['pco_los_l2'], 0.0)
                pco = (k1*pco_l1 - k2*pco_l2)
                dtr = get_any(gps_epoch, ['dtr'], 0.0)
                gps_obs = (k1 * P1 - k2 * P2) + C * clk_s - tro - dprel + pco - dtr

            elif self.gps_mode == 'L1L5':
                # IF(L1,L2): k1*C1 - k2*C2  + c*clk  - tro  - dprel
                c1 = first_col(gps_epoch, ['C1'])
                c2 = first_col(gps_epoch, ['C5'])
                if c1 is None or c2 is None:
                    raise ValueError("GPS L1L2: no C1* or C2* columns in gps_epoch.")

                f1 = self.FREQ_DICT['L1']
                f2 = self.FREQ_DICT['L5']
                k1 = (f1 ** 2) / (f1 ** 2 - f2 ** 2)
                k2 = (f2 ** 2) / (f1 ** 2 - f2 ** 2)

                P1 = get_arr(gps_epoch, c1)
                P2 = get_arr(gps_epoch, c2)
                tro = get_any(gps_epoch, ['tro'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                pco_l1 = get_any(gps_epoch, ['pco_los_l1'], 0.0)
                pco_l2 = get_any(gps_epoch, ['pco_los_l5'], 0.0)
                pco = (k1*pco_l1 - k2*pco_l2)
                dtr = get_any(gps_epoch, ['dtr'], 0.0)
                gps_obs = (k1 * P1 - k2 * P2) + C * clk_s - tro - dprel + pco - dtr

            elif self.gps_mode == 'L2L5':
                # IF(L1,L2): k1*C1 - k2*C2  + c*clk  - tro  - dprel
                c1 = first_col(gps_epoch, ['C2'])
                c2 = first_col(gps_epoch, ['C5'])
                if c1 is None or c2 is None:
                    raise ValueError("GPS L1L2: no C2* or C5* columns in gps_epoch.")

                f1 = self.FREQ_DICT['L2']
                f2 = self.FREQ_DICT['L5']
                k1 = (f1 ** 2) / (f1 ** 2 - f2 ** 2)
                k2 = (f2 ** 2) / (f1 ** 2 - f2 ** 2)

                P1 = get_arr(gps_epoch, c1)
                P2 = get_arr(gps_epoch, c2)
                tro = get_any(gps_epoch, ['tro'], 0.0)
                dprel = get_any(gps_epoch, ['dprel'], 0.0)
                clk_s = get_any(gps_epoch, ['clk'], 0.0)  # s
                pco_l1 = get_any(gps_epoch, ['pco_los_l2'], 0.0)
                pco_l2 = get_any(gps_epoch, ['pco_los_l5'], 0.0)
                pco = (k1*pco_l1 - k2*pco_l2)
                dtr = get_any(gps_epoch, ['dtr'], 0.0)
                gps_obs = (k1 * P1 - k2 * P2) + C * clk_s - tro - dprel + pco - dtr

            else:
                raise ValueError(f"Unknown gps_mode: {self.gps_mode}")

        # ---------- Galileo ----------
        gal_obs = None
        if gal_epoch is not None and len(gal_epoch) > 0 and getattr(self, 'gal_mode', None):

            if self.gal_mode == 'E1':
                # E1 single: P_E1 - tro - ion - dprel + c*(clk - BGD_E1*)
                c1 = first_col(gal_epoch, ['C1C', 'C1X', 'C1A', 'C1'])
                if c1 is None:
                    c1 = first_col(gal_epoch, ['C1B'])
                if c1 is None:
                    raise ValueError("GAL E1: column C1* missing in gal_epoch.")

                P = get_arr(gal_epoch, c1)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                ion = get_any(gal_epoch, ['ion'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)
                dtr = get_any(gal_epoch, ['dtr_gal'], 0.0)

                # BGD (sekundy) – próbujemy kilka popularnych nazw
                BGD_s = get_any(
                    gal_epoch,
                    ['BGD', 'BGDe5a', 'BGDe5b', 'BGD_E1E5a', 'BGD_E1E5b'],
                    0.0
                )

                clk_term_m = C * (clk_s - BGD_s)
                gal_obs = P - tro - ion - dprel + clk_term_m - dtr
            elif self.gal_mode == 'E5a':
                # E1 single: P_E1 - tro - ion - dprel + c*(clk - BGD_E1*)
                c1 = first_col(gal_epoch, ['C5Q', 'C5X', 'C5'])


                P = get_arr(gal_epoch, c1)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                ion = get_any(gal_epoch, ['ion'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)
                dtr = get_any(gal_epoch, ['dtr_gal'], 0.0)

                # BGD (sekundy) – próbujemy kilka popularnych nazw
                BGD_s = get_any(
                    gal_epoch,
                    ['BGDe5a'],
                    0.0
                )

                clk_term_m = C * (clk_s - BGD_s)
                gal_obs = P - tro - ion - dprel + clk_term_m - dtr

            elif self.gal_mode == 'E5b':
                # E1 single: P_E1 - tro - ion - dprel + c*(clk - BGD_E1*)
                c1 = first_col(gal_epoch, ['C7'])

                P = get_arr(gal_epoch, c1)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                ion = get_any(gal_epoch, ['ion'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)
                dtr = get_any(gal_epoch, ['dtr_gal'], 0.0)

                # BGD (sekundy) – próbujemy kilka popularnych nazw
                BGD_s = get_any(
                    gal_epoch,
                    ['BGDe5b'],
                    0.0
                )

                clk_term_m = C * (clk_s - BGD_s)
                gal_obs = P - tro - ion - dprel + clk_term_m - dtr

            elif self.gal_mode == 'E1E5a':
                # IF(E1,E5a): k1*C1 - k2*C5  + c*clk  - tro  - dprel
                c1 = first_col(gal_epoch, ['C1C', 'C1X', 'C1W', 'C1'])
                c5 = first_col(gal_epoch, ['C5Q', 'C5X', 'C5', 'C5A'])
                if c1 is None or c5 is None:
                    raise ValueError("GAL E1E5a: columns C1* or C5* are missing in gal_epoch.")

                f1 = self.FREQ_DICT['E1']
                f5a = self.FREQ_DICT['E5a']
                k1 = (f1 ** 2) / (f1 ** 2 - f5a ** 2)
                k2 = (f5a ** 2) / (f1 ** 2 - f5a ** 2)

                P1 = get_arr(gal_epoch, c1)
                P5 = get_arr(gal_epoch, c5)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)
                dtr = get_any(gal_epoch, ['dtr_gal'], 0.0)

                gal_obs = (k1 * P1 - k2 * P5) + C * clk_s - tro - dprel - dtr

            elif self.gal_mode == 'E1E5b':
                # IF(E1,E5a): k1*C1 - k2*C5  + c*clk  - tro  - dprel
                c1 = first_col(gal_epoch, ['C1C', 'C1X', 'C1W', 'C1'])
                c5 = first_col(gal_epoch, ['C7Q', 'C7X', 'C7'])
                if c1 is None or c5 is None:
                    raise ValueError("GAL E1E5a: columns C1* or C7* are missing in gal_epoch.")

                f1 = self.FREQ_DICT['E1']
                f5a = self.FREQ_DICT['E5b']
                k1 = (f1 ** 2) / (f1 ** 2 - f5a ** 2)
                k2 = (f5a ** 2) / (f1 ** 2 - f5a ** 2)

                P1 = get_arr(gal_epoch, c1)
                P5 = get_arr(gal_epoch, c5)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)
                dtr = get_any(gal_epoch, ['dtr_gal'], 0.0)

                gal_obs = (k1 * P1 - k2 * P5) + C * clk_s - tro - dprel - dtr

            elif self.gal_mode == 'E5aE5b':
                # IF(E1,E5a): k1*C1 - k2*C5  + c*clk  - tro  - dprel
                c1 = first_col(gal_epoch, ['C5'])
                c5 = first_col(gal_epoch, ['C7'])
                if c1 is None or c5 is None:
                    raise ValueError("GAL E1E5a: columns C1* or C7* are missing in gal_epoch.")

                f1 = self.FREQ_DICT['E5a']
                f5a = self.FREQ_DICT['E5b']
                k1 = (f1 ** 2) / (f1 ** 2 - f5a ** 2)
                k2 = (f5a ** 2) / (f1 ** 2 - f5a ** 2)

                P1 = get_arr(gal_epoch, c1)
                P5 = get_arr(gal_epoch, c5)
                tro = get_any(gal_epoch, ['tro'], 0.0)
                dprel = get_any(gal_epoch, ['dprel'], 0.0)
                clk_s = get_any(gal_epoch, ['clk'], 0.0)
                dtr = get_any(gal_epoch, ['dtr_gal'], 0.0)

                gal_obs = (k1 * P1 - k2 * P5) + C * clk_s - tro - dprel - dtr

            else:
                raise ValueError(f"Unknown gal_mode: {self.gal_mode}")

        # ---------- Zwracanie zgodnie z Twoją konwencją ----------

        if gps_obs is None and gal_obs is None:
            return None, None, None

        if gps_obs is None:
            return gal_obs, None, gal_obs

        if gal_obs is None:
            return gps_obs, gps_obs, None

        all_obs = np.concatenate((gps_obs, gal_obs))
        return all_obs, gps_obs, gal_obs

    def select_epoch(self, df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
        idx = df.index

        if not isinstance(idx, pd.MultiIndex):
            raise ValueError("Expected MultiIndex")

        if 'time' in idx.names:
            lvl = idx.names.index('time')
            return df.xs(t, level=lvl)

        # fallback: po typie danych
        for i in range(idx.nlevels):
            if isinstance(idx.levels[i][0], pd.Timestamp):
                return df.xs(t, level=i)

        raise KeyError("No datetime level found in index")

    def lsq(self, ref=None):
        results_rows = []

        xyz_apr = self.xyz_apr.copy()

        has_gps = self.gps_obs is not None
        has_gal = self.gal_obs is not None

        if not has_gps and not has_gal:
            raise ValueError("No observations: both gps_obs and gal_obs are None.")

        # --- epochs: UNION (nie intersection)
        gps_epochs = {}
        gal_epochs = {}
        epoch_index = None
        if has_gps:
            gps_times = self.gps_obs.index.get_level_values("time").unique()
            epoch_index = gps_times if epoch_index is None else epoch_index.union(gps_times)
            gps_epochs = {t: df for t, df in self.gps_obs.groupby(level="time", sort=False)}
        if has_gal:
            gal_times = self.gal_obs.index.get_level_values("time").unique()
            epoch_index = gal_times if epoch_index is None else epoch_index.union(gal_times)
            gal_epochs = {t: df for t, df in self.gal_obs.groupby(level="time", sort=False)}
        epochs = epoch_index.sort_values() if epoch_index is not None else []


        tol = 1e-2
        max_iter = 5

        for t in epochs:
            # --- fetch epoch slices if present
            gps_epoch = None
            gal_epoch = None
            gps_sats = None
            gal_sats = None

            if has_gps:
                gps_epoch = gps_epochs.get(t)

                if gps_epoch is not None and len(gps_epoch) > 0:
                    gps_sats = gps_epoch[["xe", "ye", "ze"]].to_numpy(copy=False)

            if has_gal:
                gal_epoch = gal_epochs.get(t)

                if gal_epoch is not None and len(gal_epoch) > 0:
                    gal_sats = gal_epoch[["xe", "ye", "ze"]].to_numpy(copy=False)

            n_gps = 0 if gps_sats is None else len(gps_sats)
            n_gal = 0 if gal_sats is None else len(gal_sats)

            # --- minimal observations check (4 params for single-system, 5 for dual)
            n_params = 0
            if n_gps > 0 and n_gal > 0:
                n_params = 5
            elif n_gps > 0 or n_gal > 0:
                n_params = 4
            else:
                continue

            if (n_gps + n_gal) < n_params:
                continue

            # --- design matrix
            A = self.hjacobian(x=xyz_apr, gps_sats=gps_sats, gal_sats=gal_sats)

            # --- observed vectors
            gps_base = self._build_obs_base(gps_epoch, 'G', self.gps_mode) if n_gps > 0 else None
            gal_base = self._build_obs_base(gal_epoch, 'E', self.gal_mode) if n_gal > 0 else None
            if gps_base is None and gal_base is None:
                continue

            # --- computed distances + L
            L_parts = []
            err_parts = []

            if n_gps > 0:
                gps_clock_init = (
                    gps_epoch["dtr"].to_numpy(dtype='float64', copy=False)
                    if "dtr" in gps_epoch.columns else 0.0
                )
                observed_gps = gps_base - gps_clock_init
                computed_gps = calculate_distance(gps_sats, xyz_apr[:3])
                L_gps = observed_gps - computed_gps
                L_parts.append(L_gps)

                ev_gps = gps_epoch["ev"].to_numpy(copy=False)
                sin_el_gps = np.sin(np.deg2rad(ev_gps))
                err_gps = 0.5 / (sin_el_gps ** 2)
                err_gps = self.code_screening(L_gps, err_gps)
                err_parts.append(err_gps)
            else:
                L_gps = None

            if n_gal > 0:
                gal_clock_init = (
                    gal_epoch["dtr_gal"].to_numpy(dtype='float64', copy=False)
                    if "dtr_gal" in gal_epoch.columns else 0.0
                )
                observed_gal = gal_base - gal_clock_init
                computed_gal = calculate_distance(gal_sats, xyz_apr[:3])
                L_gal = observed_gal - computed_gal
                L_parts.append(L_gal)

                ev_gal = gal_epoch["ev"].to_numpy(copy=False)
                sin_el_gal = np.sin(np.deg2rad(ev_gal))
                err_gal = 0.5 / (sin_el_gal ** 2)
                base_err_gal = err_gal.copy()
                err_gal = self.code_screening(L_gal, err_gal)
                err_parts.append(err_gal)
            else:
                L_gal = None

            L_all = np.concatenate(L_parts) if len(L_parts) > 1 else L_parts[0]
            err = np.concatenate(err_parts) if len(err_parts) > 1 else err_parts[0]

            # --- LSQ
            X = self._weighted_lsq_solution(A, L_all, err)

            # --- init update
            obl = xyz_apr[:3] + X[:3]
            gps_clock = 0.0
            gal_clock = 0.0

            # indeksy zegarów w wektorze X:
            # single-system: X[3] = dtr (dla tego systemu)
            # dual-system:   X[3] = dtr_gps, X[4] = dtr_gal
            if n_gps > 0 and n_gal > 0:
                gps_clock = float(X[3])
                gal_clock = float(X[4])
            elif n_gps > 0:
                gps_clock = float(X[3])
            elif n_gal > 0:
                gal_clock = float(X[3])

            # --- iterations
            it = 0
            while True:
                x_prev = X.copy()

                A = self.hjacobian(x=obl, gps_sats=gps_sats, gal_sats=gal_sats)

                L_parts = []
                err_parts = []

                if n_gps > 0:
                    observed_gps = gps_base - gps_clock
                    computed_gps = calculate_distance(gps_sats, obl)
                    L_gps = observed_gps - computed_gps
                    err_gps = self.code_screening(L_gps, err_gps)
                    L_parts.append(L_gps)
                    err_parts.append(err_gps)

                if n_gal > 0:
                    observed_gal = gal_base - gal_clock
                    computed_gal = calculate_distance(gal_sats, obl)
                    L_gal = observed_gal - computed_gal
                    err_gal = base_err_gal.copy()
                    err_gal = self.code_screening(L_gal, err_gal)
                    L_parts.append(L_gal)
                    err_parts.append(err_gal)

                L_all = np.concatenate(L_parts) if len(L_parts) > 1 else L_parts[0]
                err = np.concatenate(err_parts) if len(err_parts) > 1 else err_parts[0]

                X = self._weighted_lsq_solution(A, L_all, err)

                obl = obl + X[:3]

                # update clocks
                if n_gps > 0 and n_gal > 0:
                    gps_clock = float(X[3])
                    gal_clock = float(X[4])
                elif n_gps > 0:
                    gps_clock = float(X[3])
                elif n_gal > 0:
                    gal_clock = float(X[3])

                it += 1
                if np.max(np.abs(X - x_prev)) < tol:
                    break
                if it >= max_iter:
                    break

            # --- residuals
            # zbuduj L_all jeszcze raz w końcowym punkcie (obl i zegary już aktualne)
            A = self.hjacobian(x=obl, gps_sats=gps_sats, gal_sats=gal_sats)

            L_parts = []
            if n_gps > 0:
                observed_gps = gps_base - gps_clock
                computed_gps = calculate_distance(gps_sats, obl)
                L_gps = observed_gps - computed_gps
                L_parts.append(L_gps)
            if n_gal > 0:
                observed_gal = gal_base - gal_clock
                computed_gal = calculate_distance(gal_sats, obl)
                L_gal = observed_gal - computed_gal
                L_parts.append(L_gal)

            L_all = np.concatenate(L_parts) if len(L_parts) > 1 else L_parts[0]
            V_all = A @ X - L_all
            rms_v = float(np.sqrt(np.mean(V_all ** 2))) if V_all.size else np.nan

            # wpisz V do źródłowych df (tylko jeśli istnieją)
            cursor = 0
            if n_gps > 0 and has_gps:
                self.gps_obs.loc[(
                    (slice(None), t) if np.issubdtype(self.gps_obs.index.levels[1].dtype, np.datetime64) else (t, slice(
                        None))), "V"] = V_all[cursor:cursor + n_gps]
                cursor += n_gps

            if n_gal > 0 and has_gal:
                self.gal_obs.loc[(
                    (slice(None), t) if np.issubdtype(self.gal_obs.index.levels[1].dtype, np.datetime64) else (t, slice(
                        None))), "V"] = V_all[cursor:cursor + n_gal]
                cursor += n_gal

            Q = self._weighted_lsq_covariance(A, err)
            sig = np.sqrt(np.clip(np.diag(Q), 0.0, np.inf))
            sig_x, sig_y, sig_z = sig[0], sig[1], sig[2]
            sig_dtr = sig[3] if len(sig) > 3 else np.nan

            # zegary do raportu
            dtr_gps = None
            dtr_gal = None
            if n_gps > 0 and n_gal > 0:
                dtr_gps = float(X[3])
                dtr_gal = float(X[4])
            elif n_gps > 0:
                dtr_gps = float(X[3])
            elif n_gal > 0:
                dtr_gal = float(X[3])

            row = {
                "time": t,
                "x": float(obl[0]),
                "y": float(obl[1]),
                "z": float(obl[2]),
                "dtr_gps": dtr_gps,
                "dtr_gal": dtr_gal,
                "sig_x": float(sig_x),
                "sig_y": float(sig_y),
                "sig_z": float(sig_z),
                "sig_dtr": float(sig_dtr),
                "n_gps": int(n_gps),
                "n_gal": int(n_gal),
                "iters": int(it),
                "rms_v": rms_v,
            }

            xyz_apr[:3] = obl

            if ref is not None:
                dif = ref - obl
                ref_flh = ecef2geodetic(ref)
                denu = ecef_to_enu(dif, flh=ref_flh, degrees=True)
                row.update({"de": float(denu[0]), "dn": float(denu[1]), "du": float(denu[2])})
                if getattr(self.config, "trace_filter", False):
                    print(f"Epoch: {t} error de: {denu[0]} dn: {denu[1]} du: {denu[2]}")
                    if n_gps > 0:
                        print(V_all[:n_gps])
                    if n_gal > 0:
                        print(V_all[n_gps:n_gps + n_gal])
                    print("====" * 30, "\n")

            results_rows.append(row)

        results_df = pd.DataFrame(results_rows)
        return results_df, (self.gps_obs if has_gps else None), (self.gal_obs if has_gal else None)


class SPPSession:  # noqa: D101 – docstring below
    """High‑level orchestrator for PPP processing session.

    The controller is intentionally *thin*: heavy lifting is delegated to helper
    objects from the underlying GNSS library – injected via the constructor or
    at call‑time so they can be mocked in unit‑tests.
    """


    def __init__(
        self,
        config: SPPConfig,
        logger: Optional[logging.Logger] = None,

    ) -> None:
        self.config: SPPConfig = config
        self.FREQ_DICT = {'L1': 1575.42e06, 'L2': 1227.60e06,
                          'E1': 1575.42e06, 'E5a': 1176.45e06}


    # ‑‑‑ public API ---------------------------------------------------------
    @guarded_session_run("SPPSession")
    def run(self) -> SPPResult:  # noqa: C901 – orchestration, acceptable
        processor = GNSSDataProcessor2(
            atx_path=self.config.atx_path,
            obs_path=self.config.obs_path,
            dcb_path=self.config.dcb_path,
            nav_path=self.config.nav_path,
            mode=self.config.gps_freq,
            sys=self.config.sys,
            galileo_modes=self.config.gal_freq,
            use_gfz=self.config.use_gfz,
        )

        # --------- Load observations
        obs_data = self._load_obs_data(processor)

        # --------- Validate selected systems and availability
        selected = set(self.config.sys)
        allowed = {"G", "E"}
        unknown = selected - allowed
        if unknown:
            raise ValueError(f"Unsupported systems in config.sys: {unknown}. Allowed: {allowed}")

        obs_by_sys = {"G": obs_data.gps, "E": obs_data.gal}
        missing = [s for s in selected if obs_by_sys.get(s) is None]
        if missing:
            raise ValueError(
                f"Selected systems {missing} but no observations found in {self.config.obs_path}"
            )

        # --------- Approximate coordinates from the header
        xyz, flh = obs_data.meta[4], obs_data.meta[5]

        # --------- Load broadcast data (optional) + screening (optional)
        broadcast = None
        if self.config.nav_path:
            broadcast = self._load_nav_data(processor)

            if self.config.screen:
                gps_obs, gal_obs = self._screen_data(
                    processor, obs_by_sys["G"], obs_by_sys["E"], broadcast
                )
                obs_by_sys["G"], obs_by_sys["E"] = gps_obs, gal_obs

        # --------- Modes per system
        mode_by_sys = {"G": self.config.gps_freq, "E": self.config.gal_freq}

        # --------- Interpolation + elevation mask
        crd_by_sys = {}
        for sys in selected:
            obs = obs_by_sys[sys]
            mode = mode_by_sys[sys]

            if self.config.orbit_type == "broadcast":
                crd = self._interpolate_broadcast(
                    obs_df=obs,
                    xyz=xyz.copy(),
                    flh=flh.copy(),
                    sys=sys,
                    mode=mode,
                    orbit=broadcast,
                    tolerance=self.config.broadcast_tolerance,
                )
            elif self.config.orbit_type == "precise":
                crd = self._interpolate_lgr(
                    obs=obs,
                    xyz=xyz.copy(),
                    flh=flh.copy(),
                    mode=mode,
                    degree=self.config.interpolation_degree,
                )
            else:
                raise ValueError(f"Unsupported orbit_type: {self.config.orbit_type}")

            if "ev" in crd.columns:
                crd = crd[crd["ev"] > self.config.ev_mask]

            crd_by_sys[sys] = crd

        # --------- Preprocessing
        for sys in selected:
            crd_by_sys[sys] = self._preprocess(
                obs=crd_by_sys[sys],
                flh=flh.copy(),
                xyz=xyz.copy(),
                broadcast=broadcast,
                phase_shift=obs_data.meta[-1],
                sat_pco=obs_data.sat_pco,
                rec_pco=obs_data.rec_pco,
                antenna_h=obs_data.meta[3],
                system=sys,
            )

        # --------- Run filter (keep current signature)
        obs_gps_crd = crd_by_sys.get("G")
        obs_gal_crd = crd_by_sys.get("E")

        result = self._run_filter(
            obs_data=obs_data,
            obs_gps_crd=obs_gps_crd,
            obs_gal_crd=obs_gal_crd,
            interval=obs_data.interval,
        )
        return result

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, interval)->SPPResult:
        solver = SinglePointPositioning(config=self.config,gps_obs=obs_gps_crd,gal_obs=obs_gal_crd,
                                        gps_mode=self.config.gps_freq,gal_mode=self.config.gal_freq,xyz_apr=obs_data.meta[4])
        if self.config.sinex_path:
            sta_name = obs_data.meta[0][:4].upper()
            snx = parse_sinex(self.config.sinex_path)
            if sta_name in snx.index:
                ref = snx.loc[sta_name].to_numpy()
                result, gps_result, gal_result = solver.lsq(ref=ref)
            else:
                result, gps_result, gal_result = solver.lsq()
        else:
            result, gps_result, gal_result = solver.lsq()
        return SPPResult(solution=result,residuals_gps=gps_result,residuals_gal=gal_result)


        # return result
    def _load_nav_data(self, processor):
        """ load broadcast navigation messages for screening"""
        broadcast = processor.load_broadcast_orbit()
        return broadcast

    # ‑‑‑ private helpers ----------------------------------------------------
    def _load_obs_data(self, processor):  # noqa: D401 – simple phrase OK
        """Read observation & navigation files, apply basic screenings."""
        # Actual implementation should call the GNSS library. Here we only log.
        obs = processor.load_obs_data()

        return obs

    def _screen_data(self, processor, gps_obs, gal_obs, broadcast):
        if self.config.system_includes('G') and gps_obs is not None:
            gps_nav = processor.screen_navigation_message(broadcast.gps_orb, 'G')
            gps_obs = processor.mark_outages(gps_obs, 'G', gps_nav)
            gps_obs = gps_obs[gps_obs['gps_outage'] == False]
            gps_obs = gps_obs[(gps_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        if self.config.system_includes('E') and gal_obs is not None:
            gal_nav = processor.screen_navigation_message(broadcast.gal_orb, 'E')
            gal_obs = processor.mark_outages(gal_obs, 'E', gal_nav)
            gal_obs = gal_obs[(gal_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        return gps_obs, gal_obs


    def _interpolate_broadcast(self, obs_df, xyz, flh,sys,mode, orbit, tolerance):
        make_ionofree(obs_df,sys,mode)
        if sys == 'G':
            orb = orbit.gps_orb
        elif sys == 'E':
            orb = orbit.gal_orb
        else:
            raise ValueError("Invalid sys: %s", sys)

        interpolator =BroadcastInterp(obs=obs_df,mode=mode,sys=sys,nav=orb,emission_time=True,tolerance=tolerance)
        obs_crd = interpolator.interpolate()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _interpolate_lgr(self, obs, flh, xyz, mode,degree):
        obs = obs.swaplevel('sv','time')
        start = datetime.now()
        sp3 = [read_sp3(f) for f in self.config.sp3_path]
        sp3_df = pd.concat(sp3)
        sp3_df = sp3_df.reset_index().drop_duplicates()
        sp3_df = sp3_df.set_index(['time', 'sv'])
        sp3_df[['x','y','z']]=sp3_df[['x','y','z']].apply(lambda x: x*1e3)
        sp3_df['clk'] = sp3_df['clk'].apply(lambda x: x * 1e-6)
        _, positions = lagrange_reception_interp(obs=obs,sp3_df=sp3_df,degree=degree)
        obs_crd = lagrange_emission_interp(obs=obs,positions=positions,sp3_df=sp3_df,degree=degree)
        end=datetime.now()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _preprocess(self, obs, flh, xyz,
                    broadcast,phase_shift,sat_pco, rec_pco, antenna_h, system):

        if broadcast is None:
            gpsa, gpsb, gala = None, None, None
        else:
            gpsa, gpsb, gala = broadcast.gpsa, broadcast.gpsb, broadcast.gala
        prc =  DDPreprocessing(df=obs, flh=flh.copy(), xyz=xyz.copy(), config=self.config, gpsa=gpsa,
                                  gpsb=gpsb, gala=gala, phase_shift_dict=phase_shift, sat_pco=sat_pco, rec_pco=rec_pco,
                                  antenna_h=antenna_h, system=system)
        obs_preprocessed = prc.run()
        return obs_preprocessed
