from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Literal, Set, Final, Any
from pathlib import Path
from ..core.configuration import Config
from ..core.validation import check_choice, check_number
import copy
import pandas as pd
import logging
import warnings
import numpy as np
from ..core.utils import calculate_distance
from ..core.conversion import ecef_to_enu, ecef2geodetic
from ..core.coordinates import BroadcastInterp, lagrange_emission_interp, lagrange_reception_interp, CustomWrapper, make_ionofree
from ..io import GNSSDataProcessor2, parse_sinex
from ..io import _first_epoch, _sinex_station_reference, _solution_to_netcdf
# stage-10 helpers under private names: ``gnx_py/__init__`` star-imports this module
from ..models.apriori import resolve_position as _resolve_position
from ..io.rinex_clk import apply_clk_clocks as _apply_clk_clocks
from ..core import session_guards as _guards
from ..core.tools import DDPreprocessing
from ..core.tools import RunNotes as _RunNotes
from ..core.session_errors import guarded_session_run
from ..core import _notes
from ..ppp.ppp_helpers import trace_message
from ..ppp.ppp_uduc import _align_sf_iono_model_to_code_biases
from ..models.biases import (attach_broadcast_group_delays, code_group_delay_correction, missing_correction_hint,
                     unmoved_code_variance_m2)
from ..core.gnss import (
    CLIGHT,
    frequency_hz,
    ionosphere_free_coefficients,
    mode_ionosphere_free_coefficients,
    mode_layout,
    mode_signals,
    signal_spec,
    system_name,
)


@dataclass(slots=True)
class SPPResult:
    """Result of an SPP run (:meth:`SPPSession.run`): the solution and the per-system residuals.

    Attributes
    ----------
    solution:
        One row per solved epoch: ``time``, the ECEF position ``x``/``y``/``z`` [m], the receiver clock per
        system ``dtr_gps``/``dtr_gal``/``dtr_bds`` [m], formal sigmas ``sig_x``/``sig_y``/``sig_z``/``sig_dtr``,
        the satellites used per system ``n_gps``/``n_gal``/``n_bds``, ``spp_rejected_obs``,
        ``spp_design_cond``, ``gdop``/``pdop``, ``redundancy``, ``iters``, ``rms_v`` [m] and, with
        ``sinex_path``, the errors ``de``/``dn``/``du`` [m]. Epochs that were not solved have no row. A
        satellite-epoch without an SP3 clock (outside the clock nodes, across a missing node, or in an interval
        that fails the clock-continuity test) is left out of its epoch with one warning naming the span, so no NaN
        reaches a solved position (the step PPP shares).

        * ``sig_x``/``sig_y``/``sig_z`` [m, ECEF] and ``sig_dtr`` [m] are formal sigmas: the square roots of the
          diagonal of ``(A^T W A)^-1`` with the weights of the final solve (variance 0.5 m^2 / sin^2(E), times 3
          for an observation the 2-MAD code screening marks), without an a-posteriori variance factor, so they
          do not scale with the residuals. ``sig_dtr`` is the sigma of the first receiver clock of the epoch in the
          order GPS, Galileo, BeiDou (the clock of the first system with observations at that epoch).
        * ``de``/``dn``/``du`` = reference minus estimate, rotated to east/north/up at the reference: the
          SOLUTION/ESTIMATE coordinates of the station (first 4 characters of the RINEX MARKER NAME) in
          ``sinex_path``, the solution valid at the first epoch (:func:`gnx_py.io.parse_sinex`). The same
          convention as PPP (``PPPResult``). A station that is not in the SINEX gives NaN columns and one
          UserWarning naming the station and the file (before, silently no columns).
          Without ``sinex_path`` the columns are absent.
    residuals_gps, residuals_gal, residuals_bds:
        The preprocessed observations of each selected system (None for a system not processed), indexed by
        ``(sv, time)``, with the columns the adjustment writes:

        * ``V`` [m]: post-fit pseudorange residual of an observation used in the epoch's solution, computed
          minus observed (``A dx - L``; an observation 25 m too long gets about -25 m). ``V`` is finite
          exactly for the observations used; every other row is NaN.
        * ``V_rejected`` [m]: residual of an observation the residual-rejection loop dropped
          (``spp_reject_reason='postfit_residual'``), against the epoch's final solution (the solution it
          was excluded from), with the sign convention of ``V``. NaN for every other row, including all
          rows of an epoch that was rejected as a whole and ``nonfinite_correction`` rows.
          :func:`gnx_py.viz.plot_spp_residuals` draws these values as the rejected observations.
        * ``spp_rejected`` (bool) and ``spp_reject_reason`` (str): why an observation was not used, either
          for itself (``postfit_residual``, ``nonfinite_correction``) or because its whole epoch was rejected
          (for example ``lsq_not_converged``, ``gdop_above_limit``, and ``insufficient_observations`` for an
          epoch with fewer observations than parameters, 3 + one clock per system, which never enters the
          adjustment; before v1.0.0 those rows were left unflagged). ``False`` / ``""`` for a used
          observation. A finite ``V`` remains the test for "used".

    The former ``covariance_info`` field, never filled, was removed in v1.0.0.
    """
    solution: Union["pd.DataFrame",None]  =None
    residuals_gps: Union["pd.DataFrame",None] = None
    residuals_gal: Union["pd.DataFrame",None] = None
    residuals_bds: Union["pd.DataFrame",None] = None

    # Convenience exporters --------------------------------------------------
    def to_netcdf(self, path: Union[str, Path], **kwargs: Any) -> None:
        """Write the solution to a netCDF file (``xarray``); ``path`` is a str or a Path, ``.nc`` is added when it
        has no suffix (before v1.0.0 a str raised AttributeError)."""
        _solution_to_netcdf(self.solution, path, **kwargs)

@dataclass()
class SPPConfig(Config):
    # True: one trace line per epoch on stdout through the shared PPP trace sink
    # (``ppp_helpers.trace_message``; prefix "[PPP TRACE] SinglePointPositioning")
    trace_filter: Optional[bool] = False
    solver: Literal["LS"] ="LS"
    spp_residual_rejection: bool = True
    spp_residual_rejection_threshold_m: float = 8.0
    spp_residual_rejection_max_iter: int = 3
    spp_bds_low_redundancy_max_jump_m: Optional[float] = 30.0
    # epochs whose geometry gives GDOP above this limit are not output (RTKLIB pntpos
    # maxgdop, default 30); None disables the test. The GDOP (solution column 'gdop') has one
    # receiver clock per system, so for multi-GNSS it is ~1.3x RTKLIB's single-clock GDOP
    # (identical for one system)
    spp_max_gdop: Optional[float] = 30.0
    # the site-displacement settings of Config (solid_tides, solid_tide_step2, ocean_loading_path, erp_path) only
    # fill the preprocessing columns dx/dy/dz and tides_los: the SPP code model does not apply them (bug-fix stage
    # 11b; a tide of a few cm is far below the SPP noise)
    # antenna_h (Config, default True, as in PPP): the header's ANTENNA: DELTA H/E/N is removed along the line of
    # sight, so the position refers to the marker; False: the antenna reference point (_antenna_m)

    #: satellite PCO modes of SPP (as PPP and TEC): only the line-of-sight correction is read
    SPP_SAT_PCO_MODES = ("los", False)

    def _field_checks(self):
        checks = super()._field_checks()
        checks.update({
            "solver": lambda v: check_choice("solver", v, ("LS",)),
            # troposphere_model: niell, saastamoinen, collins (UNB3, written to the 'tro' column SPP reads since
            # bug-fix stage 11) or False, as checked by Config
            # SPP corrects the code along the line of sight (sat_pco_los_<signal>); 'crd' writes satellite
            # coordinates (x_apc) that SPP never reads, so it would silently apply no satellite PCO
            "sat_pco": lambda v: check_choice("sat_pco", v, self.SPP_SAT_PCO_MODES, notes={
                "crd": "SPP applies the satellite PCO along the line of sight ('los'); 'crd' shifts satellite "
                       "coordinates that SPP does not read (no satellite PCO would be applied)"}),
            "spp_residual_rejection_threshold_m": lambda v: check_number(
                "spp_residual_rejection_threshold_m", v, gt=0, unit="m"),
            "spp_residual_rejection_max_iter": lambda v: check_number(
                "spp_residual_rejection_max_iter", v, ge=0, integer=True),
            "spp_bds_low_redundancy_max_jump_m": lambda v: check_number(
                "spp_bds_low_redundancy_max_jump_m", v, ge=0, unit="m", allow_none=True),
            "spp_max_gdop": lambda v: check_number("spp_max_gdop", v, gt=0, allow_none=True),
        })
        return checks


def _spp_satellite_antenna_term(system: str, mode: str, orbit_type) -> bool:
    """Whether SPP applies the satellite antenna term (``sat_pco_los_<signal>``) of ``system``/``mode``.

    GPS and Galileo: always (with broadcast orbits preprocessing writes 0). BeiDou: only dual-frequency modes with
    precise orbits.

    - Broadcast BeiDou orbits: no satellite term, as for GPS and Galileo. The
      broadcast orbit and clock refer to the B3I APC (``antex.broadcast_apc_reference``) and the message group
      delays (TGD relative to B3I) map it to the signal. Measured (rms3d [m], 24 h, none / PCO(signal) - PCO(B3I) /
      full signal PCO): BRUX 2024-035 B1I 1.914 / 2.157 / 2.793, B1C 1.427 / 1.591 / 1.657, B1IB3I 3.372 /
      3.984 / 5.138; over 18 station-day rows none is better than PCO(signal) - PCO(B3I) on 11 and worse on 4.
    - Precise single-frequency BeiDou (measured; the broadcast-orbit rule does not apply): no term. BRUX 2024-035, 24 h, GRG + CAS1 DCB, IGS20_2317 (rms3d [m] with / without the term):
      B1IB3I 1.977 / 2.057, B1CB2a 1.171 / 1.225, but B1I 1.598 / 1.356, B3I 1.770 / 1.560, B1C 1.368 / 1.233;
      the PCO of the IF(B1I,B3I) clock reference instead is worse still (B1I 2.135, B3I 2.663, B1C 1.635), while
      GPS L1 (1.273 / 1.645) and Galileo E1 (1.257 / 1.282) gain from the signal term.

    When the term is not applied, SPP preprocesses BeiDou with ``sat_pco=False``, so a satellite without an ANTEX
    entry is not excluded for a term SPP does not use.
    """
    if system != "C":
        return True
    return orbit_type != "broadcast" and len(mode_signals(mode)) > 1


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
        # (system, mode, missing correction) already reported by a warning in this run
        self._group_delay_reported: Set[tuple] = set()
        # (system, mode, clock, template) -> satellites, reported at the end of the run
        self._group_delay_pending: dict = {}

    def _iono_free_coeff(self, f1_name: str, f2_name: str) -> Tuple[float, float]:
        return ionosphere_free_coefficients(f1_name, f2_name)

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

    def _group_delay_m(self, epoch: pd.DataFrame, system: str, mode: str, codes: tuple,
                       weights: tuple) -> np.ndarray:
        """Satellite group-delay correction [m] of the mode's code observable(s), added to the code.

        One convention for every signal and clock source (``biases.code_group_delay_correction``):
        broadcast clocks take the ICD group delays of the navigation message the mode uses (GPS
        LNAV TGD; Galileo I/NAV for E1/E5b/E1E5b/E5aE5b and F/NAV for E5a/E1E5a, as selected by
        ``BroadcastInterp``; BeiDou TGD1/TGD2 from D1/D2 and TGD_B1Cp/TGD_B2ap/TGD_B2bI from B-CNAV);
        precise clocks take the product OSB/DSB, else the broadcast group delay moved to the
        precise clock datum (per satellite: one the product does not give takes the next source).
        A correction that is not available is taken as 0 (or replaced as the message says) and
        reported by one warning per run, system, mode and missing quantity; the satellites named
        by the helper are collected over the run and named in one warning at its end.
        """
        precise = getattr(self.config, "orbit_type", None) == "precise"
        clock = "precise" if precise else "broadcast"
        nav_message = None
        if system == "E" and not precise and not self._has_clock_message(epoch):
            # the message BroadcastInterp selects for the mode; rows with DataSrc use their record's
            nav_message = "F/NAV" if mode in ("E5a", "E1E5a") else "I/NAV"
        # precise BeiDou clocks: the IF datum of each satellite's generation, resolved once per session run
        # (session_guards.resolve_precise_clock_datum, bug-fix stage 13b SPP-080; None: IF(C2I,C6I)); broadcast
        # clocks: the satellite generations of the run, B2I or B2b-I of a band-7 code (13d, NQ-SPP-32)
        datum = _notes.run_memo(self.config, "precise_clock_datum" if precise else "bds_generations").get(system)
        result = code_group_delay_correction(
            epoch, system, codes, weights, clock=clock, nav_message=nav_message, include_station=True,
            zero_missing=True, clock_datum=datum,
        )
        correction = result.correction_m
        single = {template.format(satellites=name): (template, name) for template, name in result.per_satellite}
        for missing in result.missing:
            if missing in single:
                # named per satellite and epoch: one warning for all satellites at the end of the run
                template, name = single[missing]
                self._group_delay_pending.setdefault((system, mode, clock, template), set()).add(name)
                continue
            self._warn_group_delay(system, mode, clock, missing)
        return np.where(np.isfinite(correction), correction, 0.0)

    def _warn_group_delay(self, system: str, mode: str, clock: str, missing: str) -> None:
        key = (system, mode, missing)
        if key in self._group_delay_reported:
            return
        self._group_delay_reported.add(key)
        # replaced by another source (e.g. "...; that of C1W is used"): not taken as 0
        replaced = "is used" in missing or ("broadcast group delay" in missing and "takes" in missing) \
            or "is taken as" in missing
        consequence = "" if replaced else " A missing group delay is taken as 0."
        warnings.warn(
            f"SPP {system_name(system)} {mode} ({clock} clocks), satellite group delay not available: "
            f"{missing}.{consequence} Typical size: {missing_correction_hint(system, missing)}; a "
            f"Bias-SINEX product with these codes in dcb_path (e.g. CAS DSB) provides it.",
            RuntimeWarning,
            stacklevel=4,
        )

    def _flush_group_delay_warnings(self) -> None:
        """One warning per system, mode and missing quantity naming every satellite it concerned."""
        pending, self._group_delay_pending = self._group_delay_pending, {}
        for (system, mode, clock, template), names in pending.items():
            self._warn_group_delay(system, mode, clock, template.format(satellites=", ".join(sorted(names))))

    @staticmethod
    def _has_clock_message(epoch: pd.DataFrame) -> bool:
        """True when every Galileo row carries the clock message of its record (RINEX 3.04 DataSrc
        bit 9 I/NAV E5b,E1 or bit 8 F/NAV E5a,E1)."""
        if "DataSrc" not in epoch.columns:
            return False
        src = np.nan_to_num(epoch["DataSrc"].to_numpy(dtype=float)).astype(np.int64)
        return bool(np.all((src & 768) > 0))

    def _antenna_m(self, epoch: pd.DataFrame, system: str, mode: str, n: int) -> np.ndarray:
        """Antenna term [m] added to the corrected code: - antenna height - receiver PCO/PCV + satellite PCO/PCV.

        Preprocessing (``DDPreprocessing``) models the code as P = rho(ARP, CoM) + pco_los - sat_pco_los,
        the receiver term in the per-band columns of ``gnss.mode_layout`` (``pco_los`` for one signal,
        ``pco_los_l1`` / ``pco_los_l2`` or ``pco_los_l5`` for two) and the satellite term in
        ``sat_pco_los_<signal>``, as PPP (ppp_gnss.py) and TEC subtract them. The model of SPP is rho + c*dtr,
        so both are removed from the observation; ionosphere-free modes take the IF combination of the two
        bands. Before v1.0.0 SPP added the receiver term (wrong sign), read columns preprocessing
        does not write for GPS L1/L2/L5/L1L5 and Galileo, and ignored the satellite term.

        Broadcast orbits, every system: no satellite term (preprocessing writes 0; the broadcast group delays
        map the message's clock-reference APC, GPS IF(L1,L2), BeiDou B3I, to the signal, as in RTKLIB/gLAB). Precise BeiDou: only dual-frequency modes take the satellite term (single-frequency
        measured, evidence in :func:`_spp_satellite_antenna_term`).

        Antenna height (``antenna_h``, as in PPP): preprocessing projects the header's ANTENNA: DELTA H/E/N (marker
        -> ARP) on the line of sight (``ah_los``, P = rho(marker) + ah_los); SPP removes it, so the position refers
        to the marker like PPP. A geometric term, the same on every band (the IF coefficients sum to 1).
        ``antenna_h=False``: not applied, the position refers to the ARP (the behaviour before v1.0.0).
        """
        layout = mode_layout(mode)
        if len(layout) == 1:
            coefficients = (1.0,)
        else:
            k1, k2 = mode_ionosphere_free_coefficients(mode)
            coefficients = (k1, -k2)
        satellite = _spp_satellite_antenna_term(system, mode, getattr(self.config, "orbit_type", None))
        term = np.zeros(n, dtype="float64")
        if getattr(self.config, "antenna_h", False):
            term -= self._series_or_zeros(epoch, "ah_los", n)
        for c, band in zip(coefficients, layout):
            term -= c * self._series_or_zeros(epoch, band["rec_pco_col"], n)
            if satellite:
                term += c * self._series_or_zeros(epoch, f"sat_pco_los_{band['signal']}", n)
        return term

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
                gd = self._group_delay_m(epoch, system, mode, (c1,), (1.0,))
                return P + gd - tro - ion - dprel + C * clk_s + self._antenna_m(epoch, system, mode, n)
            if mode == 'L2':
                c1 = self._first_matching_col(cols, ['C2W', 'C2X', 'C2', 'C2P'])
                if c1 is None:
                    raise ValueError("GPS L2: column C2* missing in gps_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                # IS-GPS-200 20.3.3.3.3.2: L2 P(Y) takes gamma12*TGD (was TGD)
                gd = self._group_delay_m(epoch, system, mode, (c1,), (1.0,))
                return P + gd - tro - ion - dprel + C * clk_s + self._antenna_m(epoch, system, mode, n)
            if mode == 'L5':
                c1 = self._first_matching_col(cols, ['C5X', 'C5'])
                if c1 is None:
                    raise ValueError("GPS L5: column C5* missing in gps_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                # IS-GPS-705 20.3.3.3.1.2: TGD - ISC_L5, ISC_L5 from a DSB/OSB product (SPP-063)
                gd = self._group_delay_m(epoch, system, mode, (c1,), (1.0,))
                return P + gd - tro - ion - dprel + C * clk_s + self._antenna_m(epoch, system, mode, n)
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
                gd = self._group_delay_m(epoch, system, mode, (c1, c2), (k1, -k2))
                return (k1 * P1 - k2 * P2) + gd + C * clk_s - tro - dprel + self._antenna_m(epoch, system, mode, n)
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
                # IF(L1,L5) on the IF(L1P,L2P) clock: c*TGD + k2*c*ISC_L5 (SPP-063)
                gd = self._group_delay_m(epoch, system, mode, (c1, c2), (k1, -k2))
                return (k1 * P1 - k2 * P2) + gd + C * clk_s - tro - dprel + self._antenna_m(epoch, system, mode, n)
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
                # IF(L2,L5) on the IF(L1P,L2P) clock: c*TGD*(k1*gamma12 - k2) + k2*c*ISC_L5 (SPP-063)
                gd = self._group_delay_m(epoch, system, mode, (c1, c2), (k1, -k2))
                return (k1 * P1 - k2 * P2) + gd + C * clk_s - tro - dprel + self._antenna_m(epoch, system, mode, n)
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
                # Galileo OS SIS ICD 5.1.5, I/NAV clock (E1/E5b): dt - BGD(E1,E5b) (was BGD(E1,E5a), SPP-065)
                gd = self._group_delay_m(epoch, system, mode, (c1,), (1.0,))
                return P + gd - tro - ion - dprel + C * clk_s + self._antenna_m(epoch, system, mode, n)
            if mode == 'E5a':
                c1 = self._first_matching_col(cols, ['C5Q', 'C5X', 'C5'])
                if c1 is None:
                    raise ValueError("GAL E5a: column C5* missing in gal_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                # F/NAV clock (E1/E5a): dt - (fE1/fE5a)^2 BGD(E1,E5a) (the factor was missing, SPP-065)
                gd = self._group_delay_m(epoch, system, mode, (c1,), (1.0,))
                return P + gd - tro - ion - dprel + C * clk_s + self._antenna_m(epoch, system, mode, n)
            if mode == 'E5b':
                c1 = self._first_matching_col(cols, ['C7'])
                if c1 is None:
                    raise ValueError("GAL E5b: column C7* missing in gal_epoch.")
                P = epoch[c1].to_numpy(dtype='float64', copy=False)
                tro = self._first_existing_or_zeros(epoch, ['tro'], n)
                ion = self._first_existing_or_zeros(epoch, ['ion'], n)
                dprel = self._first_existing_or_zeros(epoch, ['dprel'], n)
                clk_s = self._first_existing_or_zeros(epoch, ['clk'], n)
                # I/NAV clock (E1/E5b): dt - (fE1/fE5b)^2 BGD(E1,E5b) (the factor was missing, SPP-065)
                gd = self._group_delay_m(epoch, system, mode, (c1,), (1.0,))
                return P + gd - tro - ion - dprel + C * clk_s + self._antenna_m(epoch, system, mode, n)
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
                gd = self._group_delay_m(epoch, system, mode, (c1, c5), (k1, -k2))
                return (k1 * P1 - k2 * P5) + gd + C * clk_s - tro - dprel + self._antenna_m(epoch, system, mode, n)
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
                gd = self._group_delay_m(epoch, system, mode, (c1, c5), (k1, -k2))
                return (k1 * P1 - k2 * P5) + gd + C * clk_s - tro - dprel + self._antenna_m(epoch, system, mode, n)
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
                # I/NAV clock refers to IF(E1,E5b): IF(E5a,E5b) takes -K*c*[BGD(E1,E5b) - BGD(E1,E5a)],
                # K = (fE1^2 - fE5a^2)/(fE5b^2 - fE5a^2) = 15.009 (SPP-066)
                gd = self._group_delay_m(epoch, system, mode, (c1, c5), (k1, -k2))
                return (k1 * P1 - k2 * P5) + gd + C * clk_s - tro - dprel + self._antenna_m(epoch, system, mode, n)
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
                # B1C/B2a/B2b: B-CNAV TGD_B1Cp/TGD_B2ap/TGD_B2bI (SPP-061); precise clocks without a
                # product bias: broadcast group delay moved to the IF(B1I,B3I) datum (SPP-062)
                gd = self._group_delay_m(epoch, system, mode, (code_col,), (1.0,))
                return P + gd - tro - ion - dprel + C * clk_s + self._antenna_m(epoch, system, mode, n)

            if len(signals) == 2:
                sig1, sig2 = signals
                c1 = self._preferred_signal_col(cols, sig1, "code")
                c2 = self._preferred_signal_col(cols, sig2, "code")
                if c1 is None or c2 is None:
                    raise ValueError(f"BeiDou {mode}: required code columns are missing in bds_epoch.")
                k1, k2 = mode_ionosphere_free_coefficients(mode)
                P1 = epoch[c1].to_numpy(dtype='float64', copy=False)
                P2 = epoch[c2].to_numpy(dtype='float64', copy=False)
                gd = self._group_delay_m(epoch, system, mode, (c1, c2), (k1, -k2))
                return (k1 * P1 - k2 * P2) + gd + C * clk_s - tro - dprel + self._antenna_m(epoch, system, mode, n)

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

    @staticmethod
    def _dops(A: np.ndarray) -> Tuple[float, float]:
        """GDOP and PDOP of the unweighted design matrix, Q = (A^T A)^-1 (Hofmann-Wellenhof et al. 2008).

        Every receiver clock column of a multi-GNSS design enters the GDOP trace, so a
        system observed by one or two satellites only (its clock absorbs them) raises it.
        Returns (inf, inf) for a singular geometry.
        """
        if A.size == 0 or A.shape[0] < A.shape[1]:
            return np.inf, np.inf
        try:
            q = np.diag(np.linalg.inv(A.T @ A))
        except np.linalg.LinAlgError:
            return np.inf, np.inf
        if not np.all(np.isfinite(q)) or np.any(q < 0.0):
            return np.inf, np.inf
        return float(np.sqrt(np.sum(q))), float(np.sqrt(np.sum(q[:3])))

    @staticmethod
    def _validated_max_gdop(config) -> Optional[float]:
        """``config.spp_max_gdop`` as a float (None = no geometry gate); ValueError if not positive."""
        max_gdop = getattr(config, "spp_max_gdop", 30.0)
        if max_gdop is None:
            return None
        try:
            value = float(max_gdop)
        except (TypeError, ValueError):
            value = np.nan
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"spp_max_gdop must be a positive number or None, got {max_gdop!r}")
        return value

    def code_screening(self, omc, err, n_sigma=2):
        scale = np.median(np.abs(omc - np.median(omc)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = max(np.median(np.abs(omc)), 1.0)
        mask = np.abs(omc - np.median(omc)) > n_sigma * scale
        if np.count_nonzero(mask) < len(omc) / 2:
            err[mask] *= 3
        return err

    def lsq(self, ref=None):
        try:
            return self._lsq(ref)
        finally:
            # the satellites without a group delay are named also when the run stops with an error
            self._flush_group_delay_warnings()

    def _lsq(self, ref=None):
        results_rows = []
        xyz_apr = self.xyz_apr.copy()
        xyz_origin = self.xyz_apr[:3].copy()

        obs_by_sys = {"G": self.gps_obs, "E": self.gal_obs, "C": self.bds_obs}
        mode_by_sys = {"G": self.gps_mode, "E": self.gal_mode, "C": self.bds_mode}
        obs_by_sys = {s: obs for s, obs in obs_by_sys.items() if obs is not None}

        if not obs_by_sys:
            raise ValueError("No observations: GPS, Galileo and BeiDou observations are all None.")

        for system, obs in obs_by_sys.items():
            if len(obs) == 0:
                # a selected system with no observation left keeps an empty residual frame
                obs_by_sys[system] = obs.assign(V=np.nan, spp_rejected=False, spp_reject_reason="",
                                                V_rejected=np.nan)
                continue
            if "V" not in obs.columns:
                obs.loc[:, "V"] = np.nan
            obs.loc[:, "spp_rejected"] = False
            obs.loc[:, "spp_reject_reason"] = ""
            # residual of an observation dropped by the rejection loop (V stays NaN for it: finite V means
            # "used"); filled per epoch below, NaN everywhere else
            obs.loc[:, "V_rejected"] = np.nan

        epoch_groups = {}
        epoch_index = None
        for system, obs in obs_by_sys.items():
            times = obs.index.get_level_values("time").unique()
            epoch_index = times if epoch_index is None else epoch_index.union(times)
            epoch_groups[system] = {t: df for t, df in obs.groupby(level="time", sort=False)}
        epochs = epoch_index.sort_values() if epoch_index is not None else []
        epoch_rejections: dict[str, set] = {}

        def _assign_v(system: str, t, values: np.ndarray, column: str = "V") -> None:
            obs = obs_by_sys[system]
            if not isinstance(obs.index, pd.MultiIndex):
                return
            idx = [slice(None)] * obs.index.nlevels
            idx[obs.index.names.index("time")] = t
            obs.loc[tuple(idx), column] = values

        def _mark_epoch_rejected(system: str, t, reason: str) -> None:
            obs = obs_by_sys[system]
            if not isinstance(obs.index, pd.MultiIndex):
                return
            idx = [slice(None)] * obs.index.nlevels
            idx[obs.index.names.index("time")] = t
            obs.loc[tuple(idx), "spp_rejected"] = True
            # an observation already rejected on its own (postfit_residual, nonfinite_correction)
            # keeps that reason; the epoch reason goes to the rows that were still in the adjustment
            reasons = obs.loc[tuple(idx), "spp_reject_reason"].to_numpy(dtype=object)
            obs.loc[tuple(idx), "spp_reject_reason"] = np.where(reasons != "", reasons, reason)
            epoch_rejections.setdefault(reason, set()).add(t)

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

        def _sv_name(system: str, row_key) -> str:
            names = obs_by_sys[system].index.names
            if isinstance(row_key, tuple) and "sv" in names:
                return str(row_key[names.index("sv")])
            return str(row_key)

        def _omc(sats_now, base_now, active_now, x, clocks) -> list[np.ndarray]:
            # observed minus computed pseudorange per system (receiver clock of the system removed)
            return [base_now[s] - clocks[s] - calculate_distance(sats_now[s], x) for s in active_now]

        def _gauss_newton(sats_now, base_now, active_now, err, x0, clocks0, n_iter):
            """Gauss-Newton on the linearised pseudorange model with FIXED weights (variances ``err``).

            Converged when the whole position/clock step is below ``tol``; returns
            (position, clocks, last step, iterations, converged).
            """
            x = x0.copy()
            clocks = {s: float(clocks0.get(s, 0.0)) for s in active_now}
            step = np.zeros(3 + len(active_now), dtype="float64")
            for k in range(n_iter):
                A_k = self.hjacobian(
                    x=x,
                    gps_sats=sats_now.get("G"),
                    gal_sats=sats_now.get("E"),
                    bds_sats=sats_now.get("C"),
                )
                L_k = np.concatenate(_omc(sats_now, base_now, active_now, x, clocks))
                step = self._weighted_lsq_solution(A_k, L_k, err)
                x = x + step[:3]
                for offset, s in enumerate(active_now, start=3):
                    clocks[s] += float(step[offset])
                if np.max(np.abs(step)) < tol:
                    return x, clocks, step, k + 1, True
            return x, clocks, step, n_iter, False

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
        max_gdop = self._validated_max_gdop(self.config)
        nonfinite = {"count": 0, "sats": set(), "columns": set(), "first": None}

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
                # fewer observations than parameters (3 + one clock per system): the epoch never enters the
                # adjustment; its rows are flagged (bug-fix stage 13d, SPP-044: they were left unflagged)
                for system in active:
                    _mark_epoch_rejected(system, t, "insufficient_observations")
                continue

            keep_by_sys = {}
            # rows that may enter the adjustment; keep_by_sys starts as a copy and loses the rows the
            # residual-rejection loop drops, so finite & ~keep marks exactly those rows
            finite_by_sys = {}
            for system in active:
                epoch = epoch_by_sys[system]
                base = self._build_obs_base(epoch, system, mode_by_sys[system])
                ev = epoch["ev"].to_numpy(dtype="float64", copy=False)
                sin_el = np.sin(np.deg2rad(ev))
                base_by_sys[system] = base
                # BDS-3 satellites whose codes cannot be moved to the precise clock datum: their variance grows by
                # the size of the missing group delay (author decision of 2026-10-15, NQ-SPP-33; 0 for the others)
                err_base_by_sys[system] = 0.5 / (sin_el ** 2) + unmoved_code_variance_m2(self.config, system, epoch)
                # an observation whose correction or satellite state is not finite (e.g. no GIM value:
                # IPP outside the IONEX grid near the pole) never reaches the adjustment; one NaN row
                # made np.linalg.lstsq fail for the whole session
                finite = (
                    np.isfinite(base)
                    & np.isfinite(sats_by_sys[system]).all(axis=1)
                    & np.isfinite(err_base_by_sys[system])
                )
                keep_by_sys[system] = finite.copy()
                finite_by_sys[system] = finite
                if not finite.all():
                    bad = np.flatnonzero(~finite)
                    for local in bad:
                        row_key = epoch.index[local]
                        _mark_observation_rejected(system, row_key, "nonfinite_correction")
                        nonfinite["sats"].add(_sv_name(system, row_key))
                    nonfinite["count"] += len(bad)
                    if nonfinite["first"] is None:
                        nonfinite["first"] = t
                    nonfinite["columns"].update(
                        c for c in ("ion", "tro", "clk", "TGD", "dprel", "xe", "ye", "ze", "ev",
                                    *(k for k in epoch.columns if str(k).startswith(("pco_los", "sat_pco_los_"))))
                        if c in epoch.columns and epoch[c].iloc[bad].isna().any()
                    )

            rejected_obs = 0
            skip_epoch = False
            reject_reason = ""
            x_start = xyz_apr[:3].copy()
            clocks_start: dict[str, float] = {}

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

                # 1) elevation-weighted solution from the a-priori position (previous epoch, or the
                #    header position at the first epoch, which may need more than max_iter steps)
                err_el = np.concatenate([err_base_now[s] for s in active_now])
                obl, clocks, X, it, converged = _gauss_newton(
                    sats_now, base_now, active_now, err_el, x_start, clocks_start, 2 * max_iter
                )
                # 2) code screening once per epoch, on the residuals of that solution; the weights are
                #    then frozen. Re-weighting inside the Gauss-Newton loop changed the cost function
                #    at every step, so the step never settled below tol (SPP-001).
                #    With no redundancy the residuals are zero up to rounding (~1e-8 m) and carry no
                #    information: screening them would down-weight a satellite picked by the noise.
                if n_obs_now > n_params_now:
                    err = np.concatenate([
                        self.code_screening(L, err_base_now[s].copy())
                        for s, L in zip(active_now, _omc(sats_now, base_now, active_now, obl, clocks))
                    ])
                else:
                    err = err_el
                if not converged or it > max_iter or np.any(err != err_el):
                    obl, clocks, X, it, converged = _gauss_newton(
                        sats_now, base_now, active_now, err, obl, clocks, max_iter
                    )

                A = self.hjacobian(
                    x=obl,
                    gps_sats=sats_now.get("G"),
                    gal_sats=sats_now.get("E"),
                    bds_sats=sats_now.get("C"),
                )
                L_all = np.concatenate(_omc(sats_now, base_now, active_now, obl, clocks))
                V_all = A @ X - L_all

                # 3) outlier rejection outside the convergence loop: drop the largest post-fit
                #    residual and re-solve, but never down to n_obs == n_params (no redundancy left
                #    to check the result: a wrong code would then fit exactly, SPP-059)
                can_reject = (
                    residual_rejection
                    and rejection_iter < residual_max_iter
                    and n_obs_now - 1 > n_params_now
                    and V_all.size > 0
                    and np.isfinite(V_all).any()
                )
                if can_reject:
                    worst = int(np.nanargmax(np.abs(V_all)))
                    worst_abs = float(abs(V_all[worst]))
                    if worst_abs > residual_threshold:
                        worst_system, worst_local = _locate_global_row(active_now, keep_by_sys, worst)
                        if np.count_nonzero(keep_by_sys[worst_system]) > 1:
                            keep_by_sys[worst_system][worst_local] = False
                            row_key = epoch_by_sys[worst_system].index[worst_local]
                            _mark_observation_rejected(worst_system, row_key, "postfit_residual")
                            rejected_obs += 1
                            x_start, clocks_start = obl, clocks
                            continue

                break

            if skip_epoch:
                for system in active:
                    _mark_epoch_rejected(system, t, reject_reason)
                continue

            gdop, pdop = self._dops(A)

            # epoch-level gates first: a rejected epoch gets no residuals V (NaN), only its reason
            if not converged:
                # the position/clock step did not settle within max_iter (RTKLIB estpos: no solution)
                for system in active:
                    _mark_epoch_rejected(system, t, "lsq_not_converged")
                continue

            if max_gdop is not None and not gdop <= float(max_gdop):
                # geometry gate for every system (RTKLIB valsol: GDOP > maxgdop); e.g. four Galileo
                # satellites with PDOP in the hundreds gave km-level positions
                for system in active:
                    _mark_epoch_rejected(system, t, "gdop_above_limit")
                continue

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
                # observations dropped by the rejection loop: residual against this final solution, with
                # the sign of V (V_all = A X - L, X ~ 0 at convergence). A system with no row left in the
                # adjustment has no clock in this solution and keeps NaN.
                dropped = finite_by_sys[system] & ~keep_by_sys[system]
                if dropped.any() and system in clocks:
                    v_rejected = np.full(len(sats_by_sys[system]), np.nan, dtype="float64")
                    v_rejected[dropped] = -(
                        base_by_sys[system][dropped]
                        - clocks[system]
                        - calculate_distance(sats_by_sys[system][dropped], obl)
                    )
                    _assign_v(system, t, v_rejected, column="V_rejected")

            Q = self._weighted_lsq_covariance(A, err)
            cond = self._weighted_design_condition(A, err)
            sig = np.sqrt(np.clip(np.diag(Q), 0.0, np.inf))
            sig_x, sig_y, sig_z = sig[0], sig[1], sig[2]
            sig_dtr = sig[3] if len(sig) > 3 else np.nan

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
                "gdop": gdop,
                "pdop": pdop,
                # observations minus parameters: 0 = exact fit, no check of the result is possible;
                # 1 = an outlier shows in V but cannot be identified, so it is not rejected
                "redundancy": int(n_obs_now - n_params_now),
                # iterations of the final fixed-weight solve (an epoch that did not converge in it is
                # not output: spp_reject_reason='lsq_not_converged')
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
                    # the documented trace sink (one line per epoch), as in PPP
                    v_abs = np.abs(V_all[np.isfinite(V_all)]) if V_all.size else np.zeros(0)
                    trace_message(True, "SinglePointPositioning", "epoch_solution", time=t,
                                  de=float(denu[0]), dn=float(denu[1]), du=float(denu[2]),
                                  n_obs=int(V_all.size), rms_v=rms_v,
                                  max_abs_v=float(v_abs.max()) if v_abs.size else None)

            results_rows.append(row)

        self._flush_group_delay_warnings()
        if nonfinite["count"]:
            cols = ", ".join(sorted(nonfinite["columns"])) or "code / bias"
            warnings.warn(
                f"SPP: {nonfinite['count']} observation(s) left out of the adjustment because a correction "
                f"or the satellite state is not finite (NaN in: {cols}; first at epoch {nonfinite['first']}); "
                f"satellites: {', '.join(sorted(nonfinite['sats']))}. They are marked "
                f"spp_reject_reason='nonfinite_correction' in the residual frames.",
                RuntimeWarning,
                stacklevel=2,
            )

        if results_rows:
            results_df = pd.DataFrame(results_rows)
        else:
            # keep the solution columns when no epoch passed, so sol["x"] etc. still work
            cols = ["time", "x", "y", "z", "dtr_gps", "dtr_gal", "dtr_bds", "sig_x", "sig_y", "sig_z",
                    "sig_dtr", "n_gps", "n_gal", "n_bds", "spp_rejected_obs", "spp_design_cond", "gdop",
                    "pdop", "redundancy", "iters", "rms_v"] + (["de", "dn", "du"] if ref is not None else [])
            results_df = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
            results_df["time"] = pd.Series(dtype="datetime64[ns]")
            counts = [f"{r}: {len(ts)}" for r, ts in sorted(epoch_rejections.items())]
            n_few = len(epochs) - len(set().union(*epoch_rejections.values()))
            if n_few:
                counts.append(f"fewer satellites than parameters: {n_few}")
            warnings.warn(
                f"SPP: no epoch solved out of {len(epochs)} (epochs by reason: {', '.join(counts) or 'none'}); "
                f"the solution is empty.",
                RuntimeWarning,
                stacklevel=2,
            )
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
        # every given input file must exist before the session starts
        self.config.check_input_files()
        self.FREQ_DICT = {'L1': 1575.42e06, 'L2': 1227.60e06,
                          'E1': 1575.42e06, 'E5a': 1176.45e06}


    # ‑‑‑ public API ---------------------------------------------------------
    @guarded_session_run("SPPSession")
    def run(self) -> SPPResult:  # noqa: C901 – orchestration, acceptable
        # warn-once keys are per run (bug-fix stage 13b): a configuration reused for another run warns again
        _notes.clear_run_notes(self.config)
        # settings changed after construction, obs_path, model/path consistency and input files,
        # all before reading any file
        self.config.validate_for_run()
        SinglePointPositioning._validated_max_gdop(self.config)  # fail before reading any file
        processor = GNSSDataProcessor2(
            # the ANTEX is read by gnx_py.antex (session_antenna_model below), not by io (bug-fix stage 11)
            atx_path=None,
            obs_path=self.config.obs_path,
            dcb_path=self.config.dcb_path,
            nav_path=self.config.nav_path,
            mode=self.config.gps_freq,
            sys=self.config.sys,
            galileo_modes=self.config.gal_freq,
            beidou_modes=self.config.bds_freq,
            use_gfz=self.config.use_gfz,
            station_name=self.config.station_name,
            rinex2_tracking=self.config.rinex2_tracking,
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
        span = _guards.file_span(obs_data, obs_by_sys)
        if missing:
            raise _guards.no_observations_error(f"Selected systems {missing} but no observations found in",
                                                missing, self.config.obs_path, self.config.time_limit, span)

        # --------- Antenna model (receiver and satellite PCO/PCV), once per session
        antenna = self.config.session_antenna_model(processor, [obs_by_sys[s] for s in selected], selected)

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

        # --------- SP3 once per session, coverage checked per system (bug-fix stage 10; NQ-SPP-14)
        sp3_df = None
        sp3_lost: set = set()
        n_obs = {s: len(obs_by_sys[s]) for s in selected}
        if self.config.orbit_type == "precise":
            sp3_df, kept = _guards.load_sp3(self.config.sp3_path, {s: obs_by_sys[s] for s in selected},
                                            time_limit=self.config.time_limit, session="SPP",
                                            obs_path=self.config.obs_path, span=span)
            # a system the SP3 does not serve keeps an empty frame (empty residual frame, as for the mask)
            sp3_lost = selected - set(kept)
            for s in sorted(sp3_lost):
                obs_by_sys[s] = obs_by_sys[s].iloc[0:0]
        elif self.config.orbit_type != "broadcast":
            raise ValueError(f"Unsupported orbit_type: {self.config.orbit_type}")
        # 30 s clocks of clk_path (D10): the SP3's analysis centre only
        self._clk_files = _guards.session_clk_files(self.config, {s: obs_by_sys[s] for s in selected},
                                                    session="SPP")
        # IF datum of the precise BeiDou clocks per satellite generation (bug-fix stage 13b, SPP-080)
        _guards.resolve_precise_clock_datum(self.config, sp3_df=sp3_df, nav=getattr(broadcast, "bds_orb", None),
                                            obs=obs_by_sys.get("C") if "C" in selected else None, antenna=antenna,
                                            session="SPP")

        # --------- Satellite states, then the a-priori position (header, approx_position or code, D3)
        states = {}
        for sys in sorted(selected):
            if len(obs_by_sys[sys]) == 0:
                continue
            states[sys] = self._satellite_states(obs=obs_by_sys[sys], sys=sys, mode=mode_by_sys[sys],
                                                 orbit=broadcast, sp3_df=sp3_df)
        xyz, flh = self._resolve_apriori(obs_data, states, mode_by_sys)

        # --------- Geometry + elevation mask
        crd_by_sys = {}
        counts = {}
        for sys in sorted(selected):
            obs = obs_by_sys[sys]
            mode = mode_by_sys[sys]
            if sys not in states:
                # no observation row (time window, screening) or no SP3 record of the system (load_sp3)
                crd_by_sys[sys] = obs.iloc[0:0]
                counts[sys] = {"obs": n_obs[sys], "orbit": 0, "reported": sys in sp3_lost}
                continue
            crd = self._wrap(states[sys], xyz=xyz.copy(), flh=flh.copy(), mode=mode)
            if self.config.orbit_type == "broadcast":
                # e.g. B1CB2b takes CNV3 records (TGD_B2bI only): TGD_B1Cp comes from the CNV1/2
                # record of the satellite; all B-CNAV clocks refer to B3I (SPP-061)
                crd = self._attach_broadcast_group_delays(crd, broadcast, sys, only_missing=True)
            else:
                # a code without a product bias falls back to the broadcast group delay (SPP-062)
                crd = self._attach_broadcast_group_delays(crd, broadcast, sys)

            pos_cols = [c for c in ("xe", "ye", "ze") if c in crd.columns]
            counts[sys] = {"obs": len(obs),
                           "orbit": int(crd[pos_cols].notna().all(axis=1).sum()) if pos_cols else len(crd)}
            if "ev" in crd.columns:
                crd = crd[crd["ev"] > self.config.ev_mask]
            if self.config.orbit_type == "precise":
                # a satellite-epoch the SP3 nodes do not cover (no extrapolation, reported by the
                # interpolator) has no position or no clock and is left out of the solution
                crd = crd.dropna(subset=[c for c in ("xe", "ye", "ze", "clk") if c in crd.columns])

            crd_by_sys[sys] = crd
        del states

        # one system left without observations (e.g. BDS under a high mask) is solved without; its empty frame
        # skips the preprocessing (sun position etc. fail on zero-size input); none left: readable error
        products = self.config.sp3_path if self.config.orbit_type == "precise" else self.config.nav_path
        _guards.drop_empty_systems(crd_by_sys, counts, "SPP", self.config.obs_path, self.config.ev_mask,
                                   orbit_type=self.config.orbit_type, products=products)

        # --------- Preprocessing
        # run-wide notes shared by the systems of this run: one warning per run, the broadcast ionosphere report
        # per system (run_notes.iono_report; bug-fix stage 10b); the reader warned for the systems of config.sys
        self.run_notes = _RunNotes(nav_systems=self.config.sys)
        for sys in selected:
            if len(crd_by_sys[sys]) == 0:
                continue
            crd_by_sys[sys] = self._preprocess(
                obs=crd_by_sys[sys],
                flh=flh.copy(),
                xyz=xyz.copy(),
                broadcast=broadcast,
                # SPP uses no phase: SYS / PHASE SHIFT plays no part (DDPreprocessing.apply_phase_shift)
                phase_shift=None,
                sat_pco=obs_data.sat_pco,
                rec_pco=obs_data.rec_pco,
                antenna_h=obs_data.meta[3],
                system=sys,
                antenna=antenna,
            )

        # --------- Single-frequency GIM correction in the satellite code-bias datum of the product (bug-fix stage 13d
        # integration review, NQ-SPP-34): with the IONEX DCBs of the system, ion + (f1/fs)^2 beta (D_used - D_GIM)
        # per satellite differing by >= 1 ns from the median (the rule of the dual-frequency constrained PPP routes)
        self.iono_dsb_alignment = {}
        for sys in sorted(selected):
            signals = mode_signals(mode_by_sys[sys]) if mode_by_sys[sys] else ()
            if len(crd_by_sys[sys]) and len(signals) == 1:
                crd_by_sys[sys], self.iono_dsb_alignment[sys] = _align_sf_iono_model_to_code_biases(
                    crd_by_sys[sys], signals[0], sys, self.config)

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

    def _session_apriori(self, obs_data):
        """The a-priori position of run() (header, approx_position or code observations), else the header's."""
        xyz = getattr(self, "_apriori_xyz", None)
        return np.asarray(obs_data.meta[4], dtype=float) if xyz is None else xyz.copy()

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, obs_bds_crd, interval)->SPPResult:
        solver = SinglePointPositioning(config=self.config,gps_obs=obs_gps_crd,gal_obs=obs_gal_crd,
                                        gps_mode=self.config.gps_freq,gal_mode=self.config.gal_freq,
                                        bds_obs=obs_bds_crd, bds_mode=self.config.bds_freq,
                                        xyz_apr=self._session_apriori(obs_data))
        ref = None
        if self.config.sinex_path:
            # SOLUTION/ESTIMATE of the station (MARKER NAME[:4]) valid at the first epoch (bug-fix stage 13d,
            # SPP-031 / SPP-081); a station not in the file: one warning and NaN de/dn/du (it was silent)
            ref, missing = _sinex_station_reference(
                self.config.sinex_path, obs_data.meta[0],
                _first_epoch(obs_gps_crd, obs_gal_crd, obs_bds_crd), "SPP", self.config.obs_path)
            if missing:
                _notes.warn_once(self.config, ("sinex_station", missing), missing, UserWarning, stacklevel=3)
        if ref is not None:
            result, gps_result, gal_result, bds_result = solver.lsq(ref=ref)
        else:
            result, gps_result, gal_result, bds_result = solver.lsq()
            if self.config.sinex_path and isinstance(result, pd.DataFrame):
                for col in ("de", "dn", "du"):
                    result[col] = np.nan
        datum = _notes.run_memo(self.config, "precise_clock_datum").get("C")
        if datum is not None and isinstance(result, pd.DataFrame):
            # the BeiDou precise clock datum the code biases used (SPP-080)
            result.attrs["precise_clock_datum"] = {"C": datum.as_dict()}
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
        """``screen=True``: outage flags in force (Galileo: the DVS/SHS flags of the mode's signals) and
        the presence of the mode's phases, negative phases included (``session_guards.screen_observations``)."""
        modes = {'G': self.config.gps_freq, 'E': self.config.gal_freq, 'C': self.config.bds_freq}
        systems = [s for s in ('G', 'E', 'C') if self.config.system_includes(s)]
        out = _guards.screen_observations(processor, {'G': gps_obs, 'E': gal_obs, 'C': bds_obs}, broadcast, modes,
                                          systems)
        return out['G'], out['E'], out['C']

    def _satellite_states(self, obs, sys, mode, orbit=None, sp3_df=None):
        """Satellite positions and clocks at emission (broadcast or precise orbits; with ``clk_path`` the 30 s
        clocks of :mod:`gnx_py.rinex_clk`), before any use of the receiver position."""
        if self.config.orbit_type == "broadcast":
            make_ionofree(obs_df=obs, mode=mode, sys=sys)
            if sys not in ('G', 'E', 'C'):
                raise ValueError("Invalid sys: %s", sys)
            orb = {'G': orbit.gps_orb, 'E': orbit.gal_orb, 'C': orbit.bds_orb}[sys]
            # broadcast_tolerance is not passed: BroadcastInterp never reads its tolerance argument (PPP-028, Q16)
            interpolator = BroadcastInterp(obs=obs, mode=mode, sys=sys, nav=orb, emission_time=True)
            return interpolator.interpolate()
        obs = obs.swaplevel('sv', 'time')
        obs.attrs["mode"] = mode
        # the session's SP3 records of this system (read once, session_guards.load_sp3)
        sp3_sys = sp3_df[sp3_df.index.get_level_values('sv').astype(str).str.startswith(sys)]
        degree = self.config.interpolation_degree
        # the run's configuration keeps the warn-once state of the SP3 node check (once per run, not per call)
        _, positions = lagrange_reception_interp(obs=obs, sp3_df=sp3_sys, degree=degree, owner=self.config)
        obs_crd = lagrange_emission_interp(obs=obs, positions=positions, sp3_df=sp3_sys, degree=degree, mode=mode,
                                           owner=self.config)
        clk_files = getattr(self, "_clk_files", None)
        if clk_files:
            # rows inside a gap of the clock files: left out (clk_gap_policy 'drop', RTKLIB pephclk) or SP3 clock
            obs_crd = _apply_clk_clocks(obs_crd, clk_files, gap_policy=self.config.clk_gap_policy)
        return obs_crd

    @staticmethod
    def _wrap(states, xyz, flh, mode):
        """Earth rotation during the signal travel, elevation and azimuth at the a-priori position."""
        wrapper = CustomWrapper(obs=states, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(), mode=mode)
        return wrapper.run()

    def _resolve_apriori(self, obs_data, states, mode_by_sys):
        """A-priori position (``apriori.resolve_position``): ``approx_position``, else the header position
        when plausible and within 100 m of the marker position from code observations (antenna minus ANTENNA:
        DELTA H/E/N; a moving receiver: its start), else the code position (with a warning; formerly a ValueError
        without the header record)."""
        meta = obs_data.meta
        xyz, flh, source = _resolve_position(
            meta[4], meta[5], states, mode_by_sys, floor_m=100.0,
            approx_position=getattr(self.config, "approx_position", None), obs_path=self.config.obs_path,
            session="SPP", delta_hen=meta[3], kinematic=None)
        self._apriori_xyz = np.asarray(xyz, dtype=float)
        self._apriori_source = source
        return np.asarray(xyz, dtype=float), np.asarray(flh, dtype=float)

    def _preprocess(self, obs, flh, xyz,
                    broadcast,phase_shift,sat_pco, rec_pco, antenna_h, system, antenna=None):

        if broadcast is None:
            gpsa, gpsb, gala = None, None, None
        else:
            gpsa, gpsb, gala = broadcast.gpsa, broadcast.gpsb, broadcast.gala
        config = self.config
        if system == "C" and config.sat_pco and not _spp_satellite_antenna_term(
                system, config.bds_freq, config.orbit_type):
            # no satellite term for this BeiDou mode: no missing-satellite exclusion for it either
            config = copy.copy(config)
            config.sat_pco = False
        prc =  DDPreprocessing(df=obs, flh=flh.copy(), xyz=xyz.copy(), config=config, gpsa=gpsa,
                                  gpsb=gpsb, gala=gala, phase_shift_dict=phase_shift, sat_pco=sat_pco, rec_pco=rec_pco,
                                  antenna_h=antenna_h, system=system, antenna=antenna,
                                  iono=getattr(broadcast, 'iono', None), run_notes=getattr(self, 'run_notes', None))
        obs_preprocessed = prc.run()
        return obs_preprocessed

    @staticmethod
    def _attach_broadcast_group_delays(crd: pd.DataFrame, broadcast, system: str,
                                       only_missing: bool = False) -> pd.DataFrame:
        """Navigation-message group delays of the nearest record (``biases.attach_broadcast_group_delays``).

        Precise orbits: used only for a code without a product OSB/DSB, moved to the precise clock
        datum (e.g. BDS B1I: -k3*c*TGD1 on IF(B1I,B3I)). ``only_missing`` (broadcast orbits): only
        the fields the selected records lack.
        """
        nav = None
        if broadcast is not None:
            nav = {"G": broadcast.gps_orb, "E": broadcast.gal_orb, "C": broadcast.bds_orb}.get(system)
        return attach_broadcast_group_delays(crd, nav, system, only_missing=only_missing)
