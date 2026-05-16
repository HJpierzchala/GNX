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
from .biases import bias_column_m, osb_m
from .gnss import (
    CLIGHT,
    frequency_hz,
    mode_ionosphere_free_coefficients,
    mode_signals,
    signal_spec,
)


@dataclass(slots=True)
class SPPResult:
    """Container holding PPP solution and auxiliary data."""
    solution: Union["pd.DataFrame",None]  =None
    residuals_gps: Union["pd.DataFrame",None] = None
    residuals_gal: Union["pd.DataFrame",None] = None
    residuals_bds: Union["pd.DataFrame",None] = None
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
    spp_residual_rejection: bool = True
    spp_residual_rejection_threshold_m: float = 8.0
    spp_residual_rejection_max_iter: int = 3
    spp_bds_low_redundancy_max_jump_m: Optional[float] = 30.0



class SinglePointPositioning:

    def __init__(self, config, gps_obs: pd.DataFrame, gal_obs: Union[pd.DataFrame, None], gps_mode, gal_mode,
                 solver='LSQ', xyz_apr=np.zeros(3), bds_obs: Union[pd.DataFrame, None] = None,
                 bds_mode: Optional[str] = None):
        self.config = config
        self.gps_obs = gps_obs
        self.gal_obs = gal_obs
        self.bds_obs = bds_obs
        self.gps_mode = gps_mode
        self.gal_mode = gal_mode
        self.bds_mode = bds_mode
        self.solver = solver
        self.FREQ_DICT = {
            signal: frequency_hz(signal)
            for signal in ('L1', 'L2', 'L5', 'E1', 'E5a', 'E5b', 'B1I', 'B1C', 'B2a', 'B2I', 'B2b', 'B3I')
        }
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

    @staticmethod
    def _preferred_signal_col(columns: pd.Index, signal: str, kind: Literal["code", "phase"]) -> Optional[str]:
        spec = signal_spec(signal)
        prefix = spec.code_prefix if kind == "code" else spec.phase_prefix
        for suffix in spec.suffix_priority:
            col = f"{prefix}{suffix}"
            if col in columns:
                return col
        for col in columns:
            if col.startswith(prefix):
                return col
        return None

    @staticmethod
    def _bds_tgd_column(signal: str) -> Optional[str]:
        if signal == "B1I":
            return "TGD1"
        if signal == "B2I":
            return "TGD2"
        return None

    @staticmethod
    def _bias_col_m(epoch: pd.DataFrame, column: str, n: int) -> Optional[np.ndarray]:
        return bias_column_m(epoch, column, n)

    def _code_osb_correction_m(self, epoch: pd.DataFrame, code_col: str, n: int) -> np.ndarray:
        if getattr(self.config, "orbit_type", None) != "precise":
            return np.zeros(n, dtype='float64')
        return -osb_m(epoch, code_col, n)

    @classmethod
    def _dcb_between_m(cls, epoch: pd.DataFrame, obs_a: str, obs_b: str, n: int) -> Optional[np.ndarray]:
        if obs_a == obs_b:
            return np.zeros(n, dtype='float64')

        direct = cls._bias_col_m(epoch, f"BIAS_{obs_a}_{obs_b}", n)
        if direct is not None:
            return direct

        reverse = cls._bias_col_m(epoch, f"BIAS_{obs_b}_{obs_a}", n)
        if reverse is not None:
            return -reverse

        return None

    @staticmethod
    def _bds_reference_codes(code_col: str) -> Optional[tuple[str, str]]:
        if code_col in {"C2I", "C6I", "C7I"}:
            return "C2I", "C6I"
        if code_col.startswith("C1"):
            suffix = code_col[-1]
            return code_col, f"C5{suffix}"
        if code_col.startswith("C5"):
            suffix = code_col[-1]
            return f"C1{suffix}", code_col
        if code_col.startswith("C7"):
            suffix = code_col[-1]
            if suffix == "Z":
                return "C1X", code_col
            return f"C1{suffix}", f"C5{suffix}"
        return None

    @classmethod
    def _bds_precise_clock_bias_correction_m(
        cls,
        epoch: pd.DataFrame,
        weighted_codes: list[tuple[str, float]],
        n: int,
    ) -> np.ndarray:
        """Return code-bias correction needed when precise BDS clocks are used.

        CODE/CAS MGEX BDS precise clocks are tied to a reference code combination.
        Raw single-frequency BDS codes must be shifted to that reference. When
        absolute OSBs for the selected observation are available they are used
        directly; otherwise DSBs are used to translate the selected BDS code
        combination to the clock reference combination.
        """
        if not weighted_codes:
            return np.zeros(n, dtype='float64')

        osb_terms = []
        all_osb_available = True
        for code_col, coeff in weighted_codes:
            osb = cls._bias_col_m(epoch, f"OSB_{code_col}", n)
            if osb is None:
                all_osb_available = False
                break
            osb_terms.append(coeff * osb)
        if all_osb_available:
            return -np.sum(osb_terms, axis=0)

        ref = cls._bds_reference_codes(weighted_codes[0][0])
        if ref is None:
            return np.zeros(n, dtype='float64')

        anchor, ref_second = ref
        ref_sig1 = signal_spec("B1C" if anchor.startswith("C1") else "B1I").name
        ref_sig2 = signal_spec("B2a" if ref_second.startswith("C5") else "B3I").name
        _, k_ref_second = mode_ionosphere_free_coefficients(ref_sig1 + ref_sig2)

        ref_dcb = cls._dcb_between_m(epoch, anchor, ref_second, n)
        if ref_dcb is None:
            return np.zeros(n, dtype='float64')

        corr = k_ref_second * ref_dcb
        for code_col, coeff in weighted_codes:
            dcb = cls._dcb_between_m(epoch, anchor, code_col, n)
            if dcb is not None:
                corr = corr + coeff * dcb
        return corr

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
                bias_corr = self._code_osb_correction_m(epoch, c1, n)
                return P + bias_corr - tro - ion - dprel + C * (clk_s - TGD_s) + pco_l1
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
                bias_corr = self._code_osb_correction_m(epoch, c1, n)
                return P + bias_corr - tro - ion - dprel + C * (clk_s - TGD_s) + pco_l2
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
                bias_corr = self._code_osb_correction_m(epoch, c1, n)
                return P + bias_corr - tro - ion - dprel + C * (clk_s - TGD_s) + pco_l5
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
                bias_corr = k1 * self._code_osb_correction_m(epoch, c1, n) - k2 * self._code_osb_correction_m(epoch, c2, n)
                return (k1 * P1 - k2 * P2) + bias_corr + C * clk_s - tro - dprel + pco
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
                bias_corr = k1 * self._code_osb_correction_m(epoch, c1, n) - k2 * self._code_osb_correction_m(epoch, c2, n)
                return (k1 * P1 - k2 * P2) + bias_corr + C * clk_s - tro - dprel + pco
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
                bias_corr = k1 * self._code_osb_correction_m(epoch, c1, n) - k2 * self._code_osb_correction_m(epoch, c2, n)
                return (k1 * P1 - k2 * P2) + bias_corr + C * clk_s - tro - dprel + pco
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
                bias_corr = self._code_osb_correction_m(epoch, c1, n)
                return P + bias_corr - tro - ion - dprel + C * (clk_s - BGD_s)
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
                bias_corr = self._code_osb_correction_m(epoch, c1, n)
                return P + bias_corr - tro - ion - dprel + C * (clk_s - BGD_s)
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
                bias_corr = self._code_osb_correction_m(epoch, c1, n)
                return P + bias_corr - tro - ion - dprel + C * (clk_s - BGD_s)
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
                bias_corr = k1 * self._code_osb_correction_m(epoch, c1, n) - k2 * self._code_osb_correction_m(epoch, c5, n)
                return (k1 * P1 - k2 * P5) + bias_corr + C * clk_s - tro - dprel
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
                bias_corr = k1 * self._code_osb_correction_m(epoch, c1, n) - k2 * self._code_osb_correction_m(epoch, c5, n)
                return (k1 * P1 - k2 * P5) + bias_corr + C * clk_s - tro - dprel
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
                bias_corr = k1 * self._code_osb_correction_m(epoch, c1, n) - k2 * self._code_osb_correction_m(epoch, c5, n)
                return (k1 * P1 - k2 * P5) + bias_corr + C * clk_s - tro - dprel
            raise ValueError(f"Unknown gal_mode: {mode}")

        if system == 'C':
            signals = mode_signals(mode)
            tro = self._first_existing_or_zeros(epoch, ['tro'], n)
            ion = self._first_existing_or_zeros(epoch, ['ion'], n)
            dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
            clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)

            if len(signals) == 1:
                signal = signals[0]
                code_col = self._preferred_signal_col(cols, signal, "code")
                if code_col is None:
                    spec = signal_spec(signal)
                    raise ValueError(f"BeiDou {mode}: column {spec.code_prefix}* missing in bds_epoch.")
                P = epoch[code_col].to_numpy(dtype='float64', copy=False)
                tgd_col = self._bds_tgd_column(signal)
                tgd_s = self._first_existing_or_zeros(epoch, [tgd_col], n) if tgd_col else np.zeros(n)
                pco = self._first_existing_or_zeros(epoch, ['pco_los'], n)
                bias_corr = np.zeros(n, dtype='float64')
                if getattr(self.config, "orbit_type", None) == "precise":
                    bias_corr = self._bds_precise_clock_bias_correction_m(
                        epoch=epoch,
                        weighted_codes=[(code_col, 1.0)],
                        n=n,
                    )
                    tgd_s = np.zeros(n, dtype='float64')
                return P + bias_corr - tro - ion - dprel + C * (clk_s - tgd_s) + pco

            if len(signals) == 2:
                sig1, sig2 = signals
                c1 = self._preferred_signal_col(cols, sig1, "code")
                c2 = self._preferred_signal_col(cols, sig2, "code")
                if c1 is None or c2 is None:
                    raise ValueError(f"BeiDou {mode}: required code columns are missing in bds_epoch.")
                k1, k2 = mode_ionosphere_free_coefficients(mode)
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P2 = epoch[c2].to_numpy(dtype='float64', copy=False)
                pco_l1 = self._first_existing_or_zeros(epoch, ['pco_los_l1'], n)
                pco_l2 = self._first_existing_or_zeros(epoch, ['pco_los_l2'], n)
                pco = k1 * pco_l1 - k2 * pco_l2
                tgd1_col = self._bds_tgd_column(sig1)
                tgd2_col = self._bds_tgd_column(sig2)
                tgd1_s = self._first_existing_or_zeros(epoch, [tgd1_col], n) if tgd1_col else np.zeros(n)
                tgd2_s = self._first_existing_or_zeros(epoch, [tgd2_col], n) if tgd2_col else np.zeros(n)
                tgd_if = k1 * tgd1_s - k2 * tgd2_s
                bias_corr = np.zeros(n, dtype='float64')
                if getattr(self.config, "orbit_type", None) == "precise":
                    bias_corr = self._bds_precise_clock_bias_correction_m(
                        epoch=epoch,
                        weighted_codes=[(c1, k1), (c2, -k2)],
                        n=n,
                    )
                    tgd_if = np.zeros(n, dtype='float64')
                return (k1 * P1 - k2 * P2) + bias_corr + C * (clk_s - tgd_if) - tro - dprel + pco

            raise ValueError(f"Unknown bds_mode: {mode}")

        raise ValueError(f"Unknown system: {system}")


    def hjacobian(self, x, gps_sats=None, gal_sats=None, bds_sats=None):
        """
        Jacobian for SPP LSQ.

        State convention:
          - single-system: [x, y, z, dtr]              -> shape (N, 4)
          - dual-system  : [x, y, z, dtr_gps, dtr_gal] -> shape (N_gps+N_gal, 5)

        Row ordering in the dual-system case: GPS rows first, then Galileo rows.
        """

        active = [
            ("G", gps_sats),
            ("E", gal_sats),
            ("C", bds_sats),
        ]
        active = [(system, sats) for system, sats in active if sats is not None and len(sats) > 0]

        if not active:
            return np.zeros((0, 4), dtype="float64")

        n_params = 3 + len(active)
        n_rows = sum(len(sats) for _, sats in active)
        A = np.zeros((n_rows, n_params), dtype="float64")

        row0 = 0
        for clock_idx, (_, sats) in enumerate(active, start=3):
            dxyz = x[:3] - sats
            dnorm = np.linalg.norm(dxyz, axis=1)  # (N,)
            A[row0:row0 + len(sats), 0:3] = dxyz / dnorm[:, None]
            A[row0:row0 + len(sats), clock_idx] = 1.0
            row0 += len(sats)

        return A

    def hx(self, x, gps_sats=None, gal_sats=None, bds_sats=None):
        """
        Modelled pseudorange (geometry + receiver clock term(s)).

        State convention:
          - single-system: [x, y, z, dtr]
          - dual-system  : [x, y, z, dtr_gps, dtr_gal]

        Output ordering in dual-system: GPS first then Galileo (must match hjacobian()).
        """

        active = [
            ("G", gps_sats),
            ("E", gal_sats),
            ("C", bds_sats),
        ]
        active = [(system, sats) for system, sats in active if sats is not None and len(sats) > 0]

        if not active:
            return np.zeros((0,), dtype="float64")

        out = []
        for clock_idx, (_, sats) in enumerate(active, start=3):
            dxyz = x[:3] - sats
            dist = np.linalg.norm(dxyz, axis=1)
            dist = dist + x[clock_idx]
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

    @staticmethod
    def _weighted_design_condition(A: np.ndarray, err: np.ndarray) -> float:
        if A.size == 0:
            return np.nan
        w_sqrt = 1.0 / np.sqrt(err)
        Aw = A * w_sqrt[:, None]
        try:
            return float(np.linalg.cond(Aw))
        except np.linalg.LinAlgError:
            return np.inf

    def code_screening(self, omc, err, n_sigma=2):
        scale = np.median(np.abs(omc - np.median(omc)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = max(np.median(np.abs(omc)), 1.0)
        mask = np.abs(omc - np.median(omc)) > n_sigma * scale
        if np.count_nonzero(mask) < len(omc) / 2:
            err[mask] *= 3
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
        xyz_origin = self.xyz_apr[:3].copy()

        obs_by_sys = {"G": self.gps_obs, "E": self.gal_obs, "C": self.bds_obs}
        mode_by_sys = {"G": self.gps_mode, "E": self.gal_mode, "C": self.bds_mode}
        obs_by_sys = {s: obs for s, obs in obs_by_sys.items() if obs is not None}

        if not obs_by_sys:
            raise ValueError("No observations: GPS, Galileo and BeiDou observations are all None.")

        for obs in obs_by_sys.values():
            if "V" not in obs.columns:
                obs.loc[:, "V"] = np.nan
            obs.loc[:, "spp_rejected"] = False
            obs.loc[:, "spp_reject_reason"] = ""

        epoch_groups = {}
        epoch_index = None
        for system, obs in obs_by_sys.items():
            times = obs.index.get_level_values("time").unique()
            epoch_index = times if epoch_index is None else epoch_index.union(times)
            epoch_groups[system] = {t: df for t, df in obs.groupby(level="time", sort=False)}
        epochs = epoch_index.sort_values() if epoch_index is not None else []

        def _assign_v(system: str, t, values: np.ndarray) -> None:
            obs = obs_by_sys[system]
            if not isinstance(obs.index, pd.MultiIndex):
                return
            idx = [slice(None)] * obs.index.nlevels
            idx[obs.index.names.index("time")] = t
            obs.loc[tuple(idx), "V"] = values

        def _mark_epoch_rejected(system: str, t, reason: str) -> None:
            obs = obs_by_sys[system]
            if not isinstance(obs.index, pd.MultiIndex):
                return
            idx = [slice(None)] * obs.index.nlevels
            idx[obs.index.names.index("time")] = t
            obs.loc[tuple(idx), "spp_rejected"] = True
            obs.loc[tuple(idx), "spp_reject_reason"] = reason

        def _mark_observation_rejected(system: str, row_key, reason: str) -> None:
            obs = obs_by_sys[system]
            obs.loc[row_key, "spp_rejected"] = True
            obs.loc[row_key, "spp_reject_reason"] = reason

        def _subset_system_dict(values_by_sys: dict[str, np.ndarray], keep_by_sys: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {
                system: values[keep_by_sys[system]]
                for system, values in values_by_sys.items()
                if system in keep_by_sys and np.count_nonzero(keep_by_sys[system]) > 0
            }

        def _locate_global_row(
            active_systems: list[str],
            keep_by_sys: dict[str, np.ndarray],
            global_idx: int,
        ) -> tuple[str, int]:
            cursor = 0
            for system in active_systems:
                kept = np.flatnonzero(keep_by_sys[system])
                n_sys = len(kept)
                if cursor <= global_idx < cursor + n_sys:
                    return system, int(kept[global_idx - cursor])
                cursor += n_sys
            raise IndexError("Residual row is outside the active SPP design matrix.")

        tol = 1e-2
        max_iter = 5
        residual_rejection = bool(getattr(self.config, "spp_residual_rejection", True))
        residual_threshold = float(getattr(self.config, "spp_residual_rejection_threshold_m", 8.0))
        residual_max_iter = int(getattr(self.config, "spp_residual_rejection_max_iter", 3))
        low_redundancy_jump_limit = getattr(
            self.config,
            "spp_bds_low_redundancy_max_jump_m",
            getattr(self.config, "spp_exact_solution_max_jump_m", 30.0),
        )

        for t in epochs:
            epoch_by_sys = {}
            sats_by_sys = {}
            base_by_sys = {}
            err_base_by_sys = {}

            for system, groups in epoch_groups.items():
                epoch = groups.get(t)
                if epoch is None or len(epoch) == 0:
                    continue
                sats = epoch[["xe", "ye", "ze"]].to_numpy(copy=False)
                if len(sats) == 0:
                    continue
                epoch_by_sys[system] = epoch
                sats_by_sys[system] = sats

            active = [s for s in ("G", "E", "C") if s in sats_by_sys]
            n_obs = sum(len(sats_by_sys[s]) for s in active)
            n_params = 3 + len(active)
            if not active or n_obs < n_params:
                continue

            for system in active:
                base_by_sys[system] = self._build_obs_base(epoch_by_sys[system], system, mode_by_sys[system])
                ev = epoch_by_sys[system]["ev"].to_numpy(copy=False)
                sin_el = np.sin(np.deg2rad(ev))
                err_base_by_sys[system] = 0.5 / (sin_el ** 2)

            keep_by_sys = {
                system: np.ones(len(sats_by_sys[system]), dtype=bool)
                for system in active
            }
            rejected_obs = 0
            skip_epoch = False
            reject_reason = ""

            for rejection_iter in range(residual_max_iter + 1):
                active_now = [s for s in active if np.count_nonzero(keep_by_sys[s]) > 0]
                sats_now = _subset_system_dict(sats_by_sys, keep_by_sys)
                base_now = _subset_system_dict(base_by_sys, keep_by_sys)
                err_base_now = _subset_system_dict(err_base_by_sys, keep_by_sys)
                n_obs_now = sum(len(sats_now[s]) for s in active_now)
                n_params_now = 3 + len(active_now)

                if not active_now or n_obs_now < n_params_now:
                    skip_epoch = True
                    reject_reason = "insufficient_observations_after_spp_rejection"
                    break

                clocks = {system: 0.0 for system in active_now}
                obl = xyz_apr[:3].copy()
                X = np.zeros(n_params_now, dtype="float64")
                err = np.ones(n_obs_now, dtype="float64")
                it = 0

                for it in range(max_iter):
                    A = self.hjacobian(
                        x=obl,
                        gps_sats=sats_now.get("G"),
                        gal_sats=sats_now.get("E"),
                        bds_sats=sats_now.get("C"),
                    )
                    L_parts = []
                    err_parts = []
                    for system in active_now:
                        observed = base_now[system] - clocks[system]
                        computed = calculate_distance(sats_now[system], obl)
                        L = observed - computed
                        err_sys = self.code_screening(L, err_base_now[system].copy())
                        L_parts.append(L)
                        err_parts.append(err_sys)

                    L_all = np.concatenate(L_parts) if len(L_parts) > 1 else L_parts[0]
                    err = np.concatenate(err_parts) if len(err_parts) > 1 else err_parts[0]
                    X = self._weighted_lsq_solution(A, L_all, err)
                    obl = obl + X[:3]
                    for offset, system in enumerate(active_now, start=3):
                        clocks[system] += float(X[offset])

                    if np.max(np.abs(X)) < tol:
                        break

                A = self.hjacobian(
                    x=obl,
                    gps_sats=sats_now.get("G"),
                    gal_sats=sats_now.get("E"),
                    bds_sats=sats_now.get("C"),
                )
                L_parts = []
                for system in active_now:
                    observed = base_now[system] - clocks[system]
                    computed = calculate_distance(sats_now[system], obl)
                    L_parts.append(observed - computed)
                L_all = np.concatenate(L_parts) if len(L_parts) > 1 else L_parts[0]
                V_all = A @ X - L_all

                can_reject = (
                    residual_rejection
                    and rejection_iter < residual_max_iter
                    and n_obs_now > n_params_now
                    and V_all.size > 0
                    and np.isfinite(V_all).any()
                )
                if can_reject:
                    worst = int(np.nanargmax(np.abs(V_all)))
                    worst_abs = float(abs(V_all[worst]))
                    if worst_abs > residual_threshold and n_obs_now - 1 >= n_params_now:
                        worst_system, worst_local = _locate_global_row(active_now, keep_by_sys, worst)
                        if np.count_nonzero(keep_by_sys[worst_system]) > 1:
                            keep_by_sys[worst_system][worst_local] = False
                            row_key = epoch_by_sys[worst_system].index[worst_local]
                            _mark_observation_rejected(worst_system, row_key, "postfit_residual")
                            rejected_obs += 1
                            continue

                break

            if skip_epoch:
                for system in active:
                    _mark_epoch_rejected(system, t, reject_reason)
                continue

            rms_v = float(np.sqrt(np.mean(V_all ** 2))) if V_all.size else np.nan

            for system in active:
                full_v = np.full(len(sats_by_sys[system]), np.nan, dtype="float64")
                if system in active_now:
                    cursor = 0
                    for s2 in active_now:
                        n_sys = len(sats_now[s2])
                        if s2 == system:
                            full_v[keep_by_sys[system]] = V_all[cursor:cursor + n_sys]
                            break
                        cursor += n_sys
                _assign_v(system, t, full_v)

            Q = self._weighted_lsq_covariance(A, err)
            cond = self._weighted_design_condition(A, err)
            sig = np.sqrt(np.clip(np.diag(Q), 0.0, np.inf))
            sig_x, sig_y, sig_z = sig[0], sig[1], sig[2]
            sig_dtr = sig[3] if len(sig) > 3 else np.nan

            if (
                active_now == ["C"]
                and n_obs_now <= n_params_now + 1
                and low_redundancy_jump_limit is not None
            ):
                epoch_step = float(np.linalg.norm(obl - xyz_apr[:3]))
                origin_step = float(np.linalg.norm(obl - xyz_origin))
                if (
                    epoch_step > float(low_redundancy_jump_limit)
                    or origin_step > float(low_redundancy_jump_limit)
                ):
                    _mark_epoch_rejected("C", t, "low_redundancy_bds_geometry_jump")
                    continue

            row = {
                "time": t,
                "x": float(obl[0]),
                "y": float(obl[1]),
                "z": float(obl[2]),
                "dtr_gps": clocks.get("G"),
                "dtr_gal": clocks.get("E"),
                "dtr_bds": clocks.get("C"),
                "sig_x": float(sig_x),
                "sig_y": float(sig_y),
                "sig_z": float(sig_z),
                "sig_dtr": float(sig_dtr),
                "n_gps": int(np.count_nonzero(keep_by_sys.get("G", []))),
                "n_gal": int(np.count_nonzero(keep_by_sys.get("E", []))),
                "n_bds": int(np.count_nonzero(keep_by_sys.get("C", []))),
                "spp_rejected_obs": int(rejected_obs),
                "spp_design_cond": cond,
                "iters": int(it + 1),
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
                    print(V_all)
                    print("====" * 30, "\n")

            results_rows.append(row)

        results_df = pd.DataFrame(results_rows)
        return (
            results_df,
            obs_by_sys.get("G"),
            obs_by_sys.get("E"),
            obs_by_sys.get("C"),
        )


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
            beidou_modes=self.config.bds_freq,
            use_gfz=self.config.use_gfz,
            station_name=self.config.station_name,
        )

        # --------- Load observations
        obs_data = self._load_obs_data(processor)

        # --------- Validate selected systems and availability
        selected = set(self.config.sys)
        allowed = {"G", "E", "C"}
        unknown = selected - allowed
        if unknown:
            raise ValueError(f"Unsupported systems in config.sys: {unknown}. Allowed: {allowed}")

        obs_by_sys = {"G": obs_data.gps, "E": obs_data.gal, "C": obs_data.bds}
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
                gps_obs, gal_obs, bds_obs = self._screen_data(
                    processor, obs_by_sys["G"], obs_by_sys["E"], obs_by_sys["C"], broadcast
                )
                obs_by_sys["G"], obs_by_sys["E"], obs_by_sys["C"] = gps_obs, gal_obs, bds_obs

        # --------- Modes per system
        mode_by_sys = {"G": self.config.gps_freq, "E": self.config.gal_freq, "C": self.config.bds_freq}

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
                    sys=sys,
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
        obs_bds_crd = crd_by_sys.get("C")

        result = self._run_filter(
            obs_data=obs_data,
            obs_gps_crd=obs_gps_crd,
            obs_gal_crd=obs_gal_crd,
            obs_bds_crd=obs_bds_crd,
            interval=obs_data.interval,
        )
        return result

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, obs_bds_crd, interval)->SPPResult:
        solver = SinglePointPositioning(config=self.config,gps_obs=obs_gps_crd,gal_obs=obs_gal_crd,
                                        gps_mode=self.config.gps_freq,gal_mode=self.config.gal_freq,
                                        bds_obs=obs_bds_crd, bds_mode=self.config.bds_freq,
                                        xyz_apr=obs_data.meta[4])
        if self.config.sinex_path:
            sta_name = obs_data.meta[0][:4].upper()
            snx = parse_sinex(self.config.sinex_path)
            if sta_name in snx.index:
                ref = snx.loc[sta_name].to_numpy()
                result, gps_result, gal_result, bds_result = solver.lsq(ref=ref)
            else:
                result, gps_result, gal_result, bds_result = solver.lsq()
        else:
            result, gps_result, gal_result, bds_result = solver.lsq()
        return SPPResult(solution=result,residuals_gps=gps_result,residuals_gal=gal_result,residuals_bds=bds_result)


        # return result
    def _load_nav_data(self, processor):
        """ load broadcast navigation messages for screening"""
        broadcast = processor.load_broadcast_orbit(tlim=self.config.time_limit)
        return broadcast

    # ‑‑‑ private helpers ----------------------------------------------------
    def _load_obs_data(self, processor):  # noqa: D401 – simple phrase OK
        """Read observation & navigation files, apply basic screenings."""
        # Actual implementation should call the GNSS library. Here we only log.
        obs = processor.load_obs_data(tlim=self.config.time_limit)

        return obs

    def _screen_data(self, processor, gps_obs, gal_obs, bds_obs, broadcast):
        if self.config.system_includes('G') and gps_obs is not None:
            gps_nav = processor.screen_navigation_message(broadcast.gps_orb, 'G')
            gps_obs = processor.mark_outages(gps_obs, 'G', gps_nav)
            gps_obs = gps_obs[gps_obs['gps_outage'] == False]
            gps_obs = gps_obs[(gps_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        if self.config.system_includes('E') and gal_obs is not None:
            gal_nav = processor.screen_navigation_message(broadcast.gal_orb, 'E')
            gal_obs = processor.mark_outages(gal_obs, 'E', gal_nav)
            gal_obs = gal_obs[(gal_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        if self.config.system_includes('C') and bds_obs is not None:
            bds_nav = processor.screen_navigation_message(broadcast.bds_orb, 'C')
            bds_obs = processor.mark_outages(bds_obs, 'C', bds_nav)
            bds_obs = bds_obs[bds_obs['bds_outage'] == False]
            bds_obs = bds_obs[(bds_obs.filter(regex=r'^L').gt(0)).all(axis=1)]

        return gps_obs, gal_obs, bds_obs


    def _interpolate_broadcast(self, obs_df, xyz, flh,sys,mode, orbit, tolerance):
        make_ionofree(obs_df=obs_df, mode=mode, sys=sys)
        if sys == 'G':
            orb = orbit.gps_orb
        elif sys == 'E':
            orb = orbit.gal_orb
        elif sys == 'C':
            orb = orbit.bds_orb
        else:
            raise ValueError("Invalid sys: %s", sys)

        interpolator =BroadcastInterp(obs=obs_df,mode=mode,sys=sys,nav=orb,emission_time=True,tolerance=tolerance)
        obs_crd = interpolator.interpolate()
        wrapper = CustomWrapper(obs=obs_crd, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(),
                                mode=mode)
        obs_crd = wrapper.run()
        return obs_crd

    def _interpolate_lgr(self, obs, flh, xyz, mode, degree, sys=None):
        obs = obs.swaplevel('sv','time')
        obs.attrs["mode"] = mode
        start = datetime.now()
        sp3_systems = (sys,) if sys is not None else ('G', 'E', 'C')
        sp3 = [read_sp3(f, sys=sp3_systems) for f in self.config.sp3_path]
        sp3 = [df for df in sp3 if not df.empty]
        if not sp3:
            raise ValueError(f"No precise SP3 records found for system {sys}")
        sp3_df = pd.concat(sp3)
        sp3_df = sp3_df.reset_index().drop_duplicates(subset=['time', 'sv'], keep='first')
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
