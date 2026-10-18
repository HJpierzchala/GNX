"""PPP session orchestration and filter selection.

This module contains the in-package ``PPPSession`` class, which is the main
runtime coordinator for PPP processing. This module loads/preprocesses data and
routes a ``PPPConfig`` instance to the concrete combined, uncombined,
single-system, mixed-system, and ionosphere-constrained filter classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Set, Union, Literal, Final, Generic

import pandas as pd

from .ppp_gnss import PPPDualFreqMultiGNSS
from .ppp_single import PPPDualFreqSingleGNSS,PPPSingleFreqSingleGNSS
from .ppp_uduc import (
    PPPUdGenericMixedGNSS,
    PPPUdSingleGNSS,
    PPPUdMixedGNSS,
    PPPUducSFSingleGNSS,
    PPPFilterMultiGNSSIonConst,
    PPPFilterMultiGNSSIonConstGEC,
)
from ..core.conversion import ecef2geodetic
from ..core.coordinates import SP3InterpolatorOptimized, make_ionofree, CustomWrapper, BroadcastInterp, lagrange_reception_interp, lagrange_emission_interp
from ..io import read_sp3, parse_sinex, GNSSDataProcessor2  # read_sp3: re-exported by gnx_py.ppp (__all__)
from ..io import _first_epoch, _sinex_station_reference, _solution_to_netcdf
from ..ionosphere.config import TECConfig, STECMonitor
from ..ionosphere.ionex import GIMReader
from ..core.time import arange_datetime
from ..core.tools import CSDetector, DDPreprocessing, RunNotes, observation_phase_shift_attrs
from ..core.configuration import MISSING, PPPConfig
from ..core.session_errors import guarded_session_run
from ..core import _notes
from ..models.apriori import resolve_position
from ..io.rinex_clk import apply_clk_clocks
from ..core.session_guards import (drop_empty_systems, file_span, load_sp3, no_observations_error, require_phases,
                              resolve_precise_clock_datum, screen_observations, session_clk_files)
from ..core.gnss import code_signal, frequency_hz, ionosphere_free_coefficients, mode_signals, system_name
from .pppar import AR_MIXED_PRODUCTS_COLUMN, AR_PHASE_BIAS_COLUMN, combined_if_pppar_enabled
from ..models.biases import (
    BROADCAST_GROUP_DELAY_FIELDS,
    NS_TO_M,
    attach_broadcast_group_delays,
    bias_column_ns,
    code_group_delay_correction,
    group_satellite_messages,
    has_satellite_osb,
    missing_correction_hint,
    satellite_code_bias_m,
)
import numpy as np
import re
import warnings

#: Noise gain sqrt(a^2 + b^2) of an ionosphere-free code above which ``_resolve_apriori`` uses the 100 m floor:
#: L1L2 2.98, L1L5/E1E5a/B1CB2a 2.59, E1E5b 2.81, B1IB2I 2.90, B1IB3I 3.53 stay at 10 m; L2L5 16.6 and
#: E5aE5b/B2aB2b 27.5 get 100 m (since v1.0.0)
CLOSE_PAIR_IF_GAIN = 5.0

SINGLE_FREQUENCY_MODES: Final[set[str]] = {"L1", "L2", "L5", "E1", "E5a", "E5b", "B1I", "B1C", "B2a", "B2I", "B2b", "B3I"}
DUAL_FREQUENCY_MODES: Final[set[str]] = {
    "L1L2",
    "L1L5",
    "L2L5",
    "E1E5a",
    "E1E5b",
    "E5aE5b",
    "B1IB2I",
    "B1IB3I",
    "B1CB2a",
    "B1CB2b",
    "B1CB3I",
    "B2aB2b",
}


def _blank_enu_without_reference(solution) -> None:
    """de/dn/du of a PPP run without reference coordinates (since v1.0.0):
    the filters write 0.0, which reads as a perfect solution. A station that is not in the given SINEX and a run
    without ``sinex_path`` get NaN ("not computed")."""
    if not isinstance(solution, pd.DataFrame):
        return
    for col in ("de", "dn", "du"):
        if col in solution.columns:
            solution[col] = np.nan


@dataclass(slots=True)
class PPPResult:
    """Container for PPP solution products and diagnostics.

    Status:
        Active public result container returned by ``PPPSession.run``.

    Fields:
        ``solution`` is the epoch-indexed position/filter output produced by a
        PPP filter. Typical columns include ENU errors, receiver clock, ZTD,
        ISB/DCB estimates, satellite counts, residual summaries, and PPP-AR
        diagnostic columns when ambiguity resolution is enabled.

        ``de``/``dn``/``du`` [m] = reference minus estimate, rotated to
        east/north/up at the reference (the convention of SPP as well,
        ``SPPResult``): the reference is the SOLUTION/ESTIMATE position of
        the station (first 4 characters of the RINEX MARKER NAME) in
        ``sinex_path``, the solution valid at the first epoch
        (:func:`gnx_py.io.parse_sinex`). A station that is not in the
        file gives one UserWarning naming the station and the file and NaN
        columns (they were 0.0, which reads as
        a perfect solution). Without ``sinex_path`` the columns are NaN as
        well (they were 0.0 before v1.0.0). SPP without
        ``sinex_path`` has no de/dn/du columns.

        Every route, an empty solution included, has the columns ``x``,
        ``y``, ``z`` [m, ECEF] and ``ct_min`` [min, float; NaN when the
        position did not converge] (before v1.0.0 ``ct_min`` held hours, the single-system IF and R4c routes
        (``PPPConfig`` explains the route labels) had
        no ``ct_min`` and R4c named the position ``xr``, ``yr``, ``zr``).
        An empty solution (no epoch solved) is a frame with these columns
        and zero rows. ``solution.attrs['precise_clock_datum']``: the IF
        datum of the precise BeiDou clocks the code biases used.

        ``residuals_gps``, ``residuals_gal`` and ``residuals_bds`` hold the
        per-system residual frames, routed by system letter: every filter returns ``{system letter: residual
        frame}`` of the systems it processed, and a processed system without
        residual rows gets an empty frame with the usual columns. A system
        that was not processed keeps None (a GPS-only R7 run leaves
        ``residuals_gal`` None; a Galileo-only run fills ``residuals_gal``).
        On the IF, GRAPHIC and R4c routes the prefit (innovation) columns are
        taken after the epoch's code-derived receiver clock prior
        (``PPPConfig.clock_prior_sigma_m``); the uncombined R1, R2, R7 and
        single-frequency constrained routes keep ``z - h(previous state)``.

        ``convergence`` is the convergence time reported by the filter in
        hours, or ``None`` when the filter did not declare convergence
        (``solution['ct_min']`` holds the same time in minutes).
    """
    solution: Union["pd.DataFrame",None]  =None
    residuals_gps: Union["pd.DataFrame",None] = None
    residuals_gal: Union["pd.DataFrame",None] = None
    residuals_bds: Union["pd.DataFrame",None] = None
    convergence: Union[float, int, None] = None

    #: columns every solution has (since v1.0.0): ECEF position [m] and the convergence time
    #: [min, NaN when not converged]
    SOLUTION_COLUMNS = ("x", "y", "z", "ct_min")

    def __post_init__(self) -> None:
        # an empty solution (no epoch solved) is a frame with the C2 columns and zero rows, never None or a
        # column-less frame (contract C2: no KeyError downstream)
        sol = self.solution
        if sol is None or (isinstance(sol, pd.DataFrame) and len(sol) == 0):
            frame = pd.DataFrame() if sol is None else sol.copy()
            for col in self.SOLUTION_COLUMNS:
                if col not in frame.columns:
                    frame[col] = pd.Series(dtype=float)
            self.solution = frame

    # Convenience exporters --------------------------------------------------
    def to_netcdf(self, path: Union[str, Path], **kwargs: Any) -> None:
        """Write the solution to a netCDF file (``xarray``); ``path`` is a str or a Path, ``.nc`` is added when it
        has no suffix (before v1.0.0 a str raised AttributeError); ``solution.attrs`` values netCDF
        cannot hold (dicts) are written as text."""
        _solution_to_netcdf(self.solution, path, **kwargs)


class PPPSession:  # noqa: D101 – docstring below
    """High-level orchestrator for a PPP processing session.

    Purpose:
        Drives the in-package PPP workflow: observation loading, optional
        broadcast navigation loading, precise/broadcast orbit interpolation,
        preprocessing, cycle-slip detection, optional STEC/GIM preparation,
        filter selection, filter execution, and normalization into
        ``PPPResult``.

    Status:
        Active session/router class. It is the supported in-package entry point
        for PPP execution and the authoritative location for filter routing.

    Config routing:
        The session reads ``PPPConfig`` but does not define configuration
        defaults itself. ``positioning_mode='combined'`` routes to ionosphere-
        free combined filters. ``positioning_mode='uncombined'`` routes by
        active systems, signal mode cardinality, and ``use_iono_constr``.
        ``positioning_mode='single'`` is treated as a deprecated alias for
        uncombined single-frequency routing.

    Supported systems:
        GPS (``G``), Galileo (``E``) and BeiDou (``C``), depending on available
        observations and configured modes.

    Warnings:
        This class is a routing/orchestration layer, not a mathematical PPP
        model. Numerical behavior lives in the selected filter class.
    """


    def __init__(
        self,
        config: PPPConfig,
    ) -> None:
        self.config: PPPConfig = config
        # every given input file must exist before the session starts
        self.config.check_input_files()
        self.FREQ_DICT = {
            signal: frequency_hz(signal)
            for signal in ("L1", "L2", "L5", "E1", "E5a", "E5b", "B1I", "B1C", "B2a", "B2I", "B2b", "B3I")
        }
        # (system, mode, missing correction) already reported by a warning in this run
        self._group_delay_reported: Set[tuple] = set()

    @guarded_session_run("PPPSession")
    def run(self):
        # warn-once keys are per run (bug-fix stage 13b): a configuration reused for another run warns again
        _notes.clear_run_notes(self.config)
        # settings changed after construction, obs_path, model/path consistency and input files,
        # all before reading any file
        self.config.validate_for_run()

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
                use_gfz = self.config.use_gfz,
                station_name=self.config.station_name,
                rinex2_tracking=self.config.rinex2_tracking,
                # single-band GPS PPP: the code of the phase's tracking (C1C + L1C; NQ-PPP-28, bug-fix stage 13b)
                gps_sf_same_tracking=self.config.gps_freq in ("L1", "L2", "L5"),
            )

        selected = set(self.config.sys)
        allowed = {'G', 'E', 'C'}
        unknown = selected - allowed
        if unknown:
            raise ValueError(f"Unsupported systems in config.sys: {unknown}. Allowed: {allowed}")
        obs_data = self._load_obs_data(processor)

        obs_by_sys = {'G': obs_data.gps, 'E': obs_data.gal, 'C': obs_data.bds}

        missing = [s for s in selected if obs_by_sys.get(s) is None]
        span = file_span(obs_data, obs_by_sys)
        if missing:
            raise no_observations_error(f"Selected systems {missing} but no observations found in", missing,
                                        self.config.obs_path, self.config.time_limit, span)
        # carrier phases of every mode signal (a code-only file is read for SPP only; stage-10 revision)
        require_phases(obs_by_sys, {'G': self.config.gps_freq, 'E': self.config.gal_freq, 'C': self.config.bds_freq},
                       sorted(selected), self.config.obs_path, "PPP")
        # antenna model (receiver and satellite PCO/PCV), once per session
        antenna = self.config.session_antenna_model(processor, [obs_by_sys[s] for s in selected], selected)
        # RINEX 2 phases of unknown tracking (bug-fix stage 10, D11): PPP-AR off (_warn_ar_phase_bias)
        self._rinex2_ar_unknown = self._rinex2_unknown_phase_tracking(processor, obs_by_sys, selected)

        broadcast = None

        if self.config.nav_path:
            broadcast = self._load_nav_data(processor)

        mode_by_sys = {'G': self.config.gps_freq, 'E': self.config.gal_freq, 'C': self.config.bds_freq}
        if self.config.nav_path and self.config.screen:
            gps_obs, gal_obs, bds_obs = self._screen_data(processor, obs_by_sys['G'], obs_by_sys['E'], obs_by_sys['C'], broadcast)
            obs_by_sys['G'], obs_by_sys['E'], obs_by_sys['C'] = gps_obs, gal_obs, bds_obs
        # the SP3 files once, with their coverage of every observed system checked (PPP-030)
        sp3_df = None
        if self.config.orbit_type == "precise":
            sp3_df, kept = load_sp3(self.config.sp3_path, {s: obs_by_sys[s] for s in selected},
                                    time_limit=self.config.time_limit, session="PPP",
                                    obs_path=self.config.obs_path, span=span)
            selected = {s for s in selected if s in kept}
        # 30 s clocks of clk_path for the filter (D10): the SP3's analysis centre only
        self._clk_files = session_clk_files(self.config, {s: obs_by_sys[s] for s in selected}, session="PPP")
        # IF datum of the precise BeiDou clocks per satellite generation, once per session (bug-fix stage 13b,
        # SPP-080): declared by the bias product, tested on the broadcast clocks, or assumed with a warning
        self._clock_datum = resolve_precise_clock_datum(
            self.config, sp3_df=sp3_df, nav=getattr(broadcast, "bds_orb", None),
            obs=obs_by_sys.get('C') if 'C' in selected else None, antenna=antenna, session="PPP")

        # satellite states of every system first (they do not depend on the receiver position), then the
        # a-priori position (header, approx_position or code observations, D3), then the geometry
        states = {}
        for sys in sorted(selected):
            if len(obs_by_sys[sys]):
                states[sys] = self._satellite_states(obs=obs_by_sys[sys], sys=sys, mode=mode_by_sys[sys],
                                                     orbit=broadcast, sp3_df=sp3_df)
        xyz, flh = self._resolve_apriori(obs_data, states, mode_by_sys)

        crd_by_sys = {}
        counts = {}
        self._group_delay_reported = set()
        for sys in sorted(selected):
            if sys not in states:
                # no observation row left (time window, screening): named by drop_empty_systems below
                crd_by_sys[sys], counts[sys] = obs_by_sys[sys], {"obs": 0}
                continue
            crd = self._wrap(states[sys], xyz=xyz.copy(), flh=flh.copy(), mode=mode_by_sys[sys])
            counts[sys] = {"obs": len(obs_by_sys[sys]),
                           "orbit": int(crd['ev'].notna().sum()) if 'ev' in crd.columns else len(crd)}
            if 'ev' in crd.columns:
                crd = crd[crd['ev'] > self.config.ev_mask]
            crd_by_sys[sys] = crd
        del states
        # a system left without observations is dropped with a warning; none left: readable error (PPP-042)
        products = self.config.sp3_path if self.config.orbit_type == "precise" else self.config.nav_path
        crd_by_sys = drop_empty_systems(crd_by_sys, counts, "PPP", self.config.obs_path, self.config.ev_mask,
                                        orbit_type=self.config.orbit_type, products=products)
        selected = set(crd_by_sys)
        self._warn_l5_ifcb(crd_by_sys.get('G'), mode_by_sys['G'], antenna)
        for sys in sorted(selected):
            mode = mode_by_sys[sys]
            crd = crd_by_sys[sys]

            # satellite code group delays as OSB_<code> relative to the clock datum: broadcast
            # TGD/BGD/ISC (PPP-058 and the PPP counterparts of SPP-061..066), precise orbits
            # without a bias product: broadcast group delay moved to the precise clock datum
            # (SPP-062)
            crd = self._attach_code_group_delays(crd, broadcast, sys, mode)
            # PPP-AR candidates: satellites with product phase biases (before the gaps are filled with 0)
            crd = self._mark_ar_phase_bias(crd, sys, mode)
            if self.config.dcb_path:
                crd = self._fill_phase_osb_gaps(crd, sys, mode)
                crd = self._drop_partial_receiver_osb(crd, sys, mode)
            crd_by_sys[sys] = crd
        self._warn_ar_phase_bias(crd_by_sys)

        # run-wide notes shared by the systems of this run: one warning per run, the broadcast ionosphere report
        # per system (run_notes.iono_report; bug-fix stage 10b); the reader warned for the systems of config.sys
        self.run_notes = RunNotes(nav_systems=self.config.sys)
        for sys in selected:
            crd_by_sys[sys] = self._preprocess(
                obs=crd_by_sys[sys],
                flh=flh.copy(),
                xyz=xyz.copy(),
                broadcast=broadcast,
                # SYS / PHASE SHIFT: the phases of the file are used as written (RINEX 3.0x applies the correction;
                # HEAD added the header values a second time); the reader's in-data records (contract C4) are
                # reported, the arcs kept (bug-fix stage 13b, NQ-PPP-38)
                phase_shift=None,
                sat_pco=obs_data.sat_pco,
                rec_pco=obs_data.rec_pco,
                antenna_h=obs_data.meta[3],
                system=sys,
                antenna=antenna,
                phase_shift_attrs=observation_phase_shift_attrs(obs_data, sys),
            )

        # RINEX LLI bit 1 (half-cycle ambiguity) of the mode's phases per observation (contract C3: PPP-AR keeps
        # these ambiguities out of fixing, the slip layer starts a new arc when the flag clears)
        for sys in selected:
            crd_by_sys[sys] = self._mark_half_cycle(crd_by_sys[sys], sys, mode_by_sys[sys])

        crd_by_sys = self._detect_all_cycle_slips(crd_by_sys, mode_by_sys, obs_data.interval, xyz)

        if self.config.ionosphere_model == "measure STEC":
            K = 40.3e16
            for sys in selected:
                mode = mode_by_sys[sys]
                f = self.FREQ_DICT[mode_signals(mode)[0]]
                tec2m = K / (f ** 2)

                df = self._measure_STEC(df=crd_by_sys[sys], system=sys)
                df.loc[:, 'ion'] = (df['leveled_tec'] / 1e16) * tec2m
                crd_by_sys[sys] = df

        for sys in selected:
            crd_by_sys[sys] = self._sanitize(df=crd_by_sys[sys])

        obs_gps_crd = crd_by_sys.get('G')
        obs_gal_crd = crd_by_sys.get('E')
        obs_bds_crd = crd_by_sys.get('C')
        result = self._run_filter(
            obs_data=obs_data,
            obs_gps_crd=obs_gps_crd,
            obs_gal_crd=obs_gal_crd,
            obs_bds_crd=obs_bds_crd,
            interval=obs_data.interval
        )
        return result

    def _sanitize(self, df):
        cols = ['xe', 'ye', 'ze', 'clk']
        return df.dropna(subset=cols)

    @staticmethod
    def _mode_code_columns(obs: pd.DataFrame, system: str, mode: str) -> List[List[str]]:
        """RINEX code columns of each signal of the mode, e.g. [['C1W'], ['C2W']] for GPS L1L2."""
        out = []
        for signal in mode_signals(mode):
            cols = []
            for col in obs.columns:
                if not (isinstance(col, str) and len(col) in (2, 3) and col[0] == "C" and col[1].isdigit()):
                    continue
                try:
                    if code_signal(system, col) == signal:
                        cols.append(col)
                except ValueError:
                    continue
            out.append(cols)
        return out

    def _attach_code_group_delays(self, crd: pd.DataFrame, broadcast, system: str, mode: str) -> pd.DataFrame:
        """Satellite code group delays of the mode's codes as ``OSB_<code>`` [ns].

        Every PPP filter subtracts ``OSB_<code>`` from the code observable (``biases.osb_m``), i.e.
        it takes the column as the satellite code bias relative to the satellite clock datum; the
        convention is that of ``biases.code_group_delay_correction`` (one place for SPP and PPP):

        * broadcast orbits: the navigation-message group delays of the record the orbit uses. GPS
          LNAV clock IF(L1P,L2P): L1 c*TGD, L2 c*gamma12*TGD, L5 c*(TGD - ISC_L5) with ISC_L5 from a
          product DSB/OSB when given (IS-GPS-200 20.3.3.3.3.2, IS-GPS-705 20.3.3.3.1.2). Galileo
          I/NAV clock IF(E1,E5b) (E1, E5b, E1E5b, E5aE5b), F/NAV IF(E1,E5a) (E5a, E1E5a): E1 c*BGD_y,
          Ey c*(fE1/fEy)^2*BGD_y, the other E5 signal c*[BGD_y + ((fE1/fEz)^2 - 1)*BGD_z] (OS SIS ICD
          5.1.5; IF(E5a,E5b) on the I/NAV clock gets -K*c*[BGD(E1,E5b) - BGD(E1,E5a)], K = 15.009). BeiDou clock B3I: B1I c*TGD1, B2I c*TGD2 (D1/D2), B1C c*TGD_B1Cp, B2a c*TGD_B2ap,
          B2b c*TGD_B2bI (B-CNAV). They replace product code OSBs of these codes, which refer to the
          precise clock datum (as in SPP: broadcast group delay first).
        * precise orbits without a bias product (no dcb_path): the broadcast group delays moved to
          the precise clock datum IF(C1W,C2W) / IF(C1C,C5Q) / IF(C2I,C6I), b - [k1*b(ref1) - k2*b(ref2)]
          (needs nav_path). The IF filters keep 0 for the datum combination itself. BeiDou: the
          datum of each satellite's generation resolved for the session (``_clock_datum``,
          ``session_guards.resolve_precise_clock_datum``; IF(B1C,B2a) for BDS-3 when the products refer to
          it).
        * precise orbits with a bias product (dcb_path): :meth:`_attach_product_code_biases`, the
          product OSB, else its DSBs expressed in the clock datum, else per satellite the next
          source.

        A correction that is not available is taken as 0 and reported by one warning per run,
        system, mode and missing quantity; on a row where one code of the mode has none, every code
        of the mode gets 0 (correcting one code of a pair alone shifts the combination by its whole
        k*b). A receiver DSB of the product is kept as receiver OSBs ``STA_OSB_<code>`` with the
        filters' DSB split, which the satellite OSB columns switch off.
        """
        if crd is None or len(crd) == 0:
            return crd
        by_signal = self._mode_code_columns(crd, system, mode)
        if not by_signal or any(not cols for cols in by_signal):
            return crd
        codes = [code for cols in by_signal for code in cols]
        precise = self.config.orbit_type == "precise"
        nav = None
        if broadcast is not None:
            nav = {"G": broadcast.gps_orb, "E": broadcast.gal_orb, "C": broadcast.bds_orb}.get(system)
        if precise and self.config.dcb_path:
            return self._attach_product_code_biases(crd, nav, system, mode, by_signal)
        # broadcast orbits: only the fields the selected records lack (e.g. B1CB2b takes CNV3 records
        # with TGD_B2bI only; TGD_B1Cp comes from the CNV1/2 record, all B-CNAV clocks refer to B3I)
        given = crd
        crd = attach_broadcast_group_delays(crd, nav, system, only_missing=not precise)
        clock = "precise" if precise else "broadcast"
        nav_message = None
        if system == "E":
            # rows with DataSrc (RINEX 3.04 bit 9: clock E5b,E1 = I/NAV, bit 8: E5a,E1 = F/NAV) use their
            # record's message, else the one BroadcastInterp selects for the mode
            src = crd["DataSrc"].to_numpy(dtype=float) if "DataSrc" in crd.columns else np.zeros(len(crd))
            if not np.all((np.nan_to_num(src).astype(np.int64) & 768) > 0):
                nav_message = "F/NAV" if mode in ("E5a", "E1E5a") else "I/NAV"

        # precise BeiDou clocks: the IF datum of each satellite's generation (SPP-080; None: IF(C2I,C6I)); broadcast
        # clocks: the satellite generations of the run, B2I or B2b-I of a band-7 code (bug-fix stage 13d, NQ-SPP-32)
        bds_datum = getattr(self, "_clock_datum", None) if precise \
            else _notes.run_memo(self.config, "bds_generations").get(system)
        per_code = {
            code: code_group_delay_correction(crd, system, (code,), (1.0,), clock=clock, nav_message=nav_message,
                                              clock_datum=bds_datum)
            for code in codes
        }
        missing = group_satellite_messages([m for result in per_code.values() for m in result.missing],
                                           [p for result in per_code.values() for p in result.per_satellite])
        first = [cols[0] for cols in by_signal]
        if precise:
            # the codes of the combination take their biases from one source (biases helper): no
            # product -> the datum combination needs none (0), any other all broadcast group delays
            weights = (1.0,)
            if len(first) == 2:
                k1, k2 = ionosphere_free_coefficients(*mode_signals(mode))
                weights = (k1, -k2)
            combo = code_group_delay_correction(crd, system, first, weights, clock="precise",
                                                nav_message=nav_message, clock_datum=bds_datum)
            labels = set(combo.source.tolist()) - {"none"}
            per_code_available = all(np.isfinite(r.correction_m).any() for r in per_code.values())
            nav_fields = any(f in crd.columns for f in BROADCAST_GROUP_DELAY_FIELDS[system])
            datum_if = labels == {"datum"} and self.config.positioning_mode == "combined"
            if not labels or not (per_code_available and nav_fields) or datum_if:
                # no correction, or the datum combination without navigation data or in the IF filters:
                # 0 (as before)
                combo_missing = group_satellite_messages(combo.missing, combo.per_satellite)
                if labels == {"datum"} and not datum_if and self.config.use_iono_constr:
                    # uncombined filters with an ionosphere constraint: the datum combination needs
                    # no group delay, its codes do (satellite DCB of the pair, ionosphere-like)
                    combo_missing.append(
                        f"satellite DCB {first[0]}-{first[1]} of the uncombined code observations "
                        f"(ionosphere-like, biases the constrained ionosphere; the broadcast group "
                        f"delays of nav_path provide it as well)"
                    )
                self._warn_group_delay(system, mode, clock, combo_missing)
                return given
            # per code (uncombined filters): the datum combination with navigation data gets the
            # broadcast group delays moved to the datum as well; their IF combination is 0
        # a row without the bias of one code of the mode gets none on any code, as the combination
        # in the helper (NaN there): one code of a pair corrected alone shifts the IF combination by
        # its whole k*b (BDS-3 B1IB2I with navigation data without TGD_B2bI: C2I alone, TGD1 = -40 ns,
        # moves IF(B1I,B2I) by -58 m)
        incomplete = np.zeros(len(crd), dtype=bool)
        for result in per_code.values():
            incomplete |= ~np.isfinite(result.correction_m)
        self._warn_group_delay(system, mode, clock, missing,
                               all_codes=len(per_code) > 1 and bool(incomplete.any()))
        crd = crd.copy()
        for code, result in per_code.items():
            # OSB_<code> = b(code) [ns]; the correction of the code is -b
            crd[f"OSB_{code}"] = np.where(incomplete, 0.0, -result.correction_m) / NS_TO_M
        self._receiver_dsb_as_osb(crd, given, first, mode)
        return crd

    @staticmethod
    def _receiver_dsb_as_osb(crd: pd.DataFrame, given: pd.DataFrame, first: List[str], mode: str) -> None:
        """Keep a product receiver DSB of the mode's pair as receiver OSBs (in place)."""
        if len(first) == 2 and not has_satellite_osb(given, first) \
                and not any(f"STA_OSB_{code}" in given.columns for code in first):
            # the satellite OSB columns switch off the filters' split of a product DSB
            # (biases.split_dual_code_dsb_corrections_m), its receiver part as well: keep a receiver
            # DSB D = B(code1) - B(code2) as receiver OSBs with the same split, -k2*D and -k1*D (IF 0)
            c1, c2 = first
            d_rcv = bias_column_ns(given, f"STA_BIAS_{c1}_{c2}", len(given))
            if d_rcv is None:
                d_rcv = bias_column_ns(given, f"STA_BIAS_{c2}_{c1}", len(given))
                d_rcv = None if d_rcv is None else -d_rcv
            if d_rcv is not None:
                k1, k2 = ionosphere_free_coefficients(*mode_signals(mode))
                crd[f"STA_OSB_{c1}"] = -k2 * d_rcv
                crd[f"STA_OSB_{c2}"] = -k1 * d_rcv

    def _attach_product_code_biases(self, crd: pd.DataFrame, nav, system: str, mode: str,
                                    by_signal: List[List[str]]) -> pd.DataFrame:
        """Precise orbits with a bias product: ``OSB_<code>`` of every code of the mode [ns] from the
        shared helper (``biases.code_group_delay_correction``, as SPP), for every filter.

        The product OSB as given; a DSB-only product (e.g. CAS) through its DSBs expressed in the
        precise clock datum, ``b = k1 D(code,ref1) - k2 D(code,ref2)`` (before v1.0.0 the filters
        split the DSB of the mode's pair so that its IF stayed 0, right for the datum pair only, so
        L1L5, E1E5b, B1CB2a ... got no satellite bias; for the datum pair the values are those of the
        split, b1 = -k2 D, b2 = -k1 D). A code the product does not give, or a satellite it does not
        give, takes the next source for all codes of the combination (broadcast group delays moved to
        the datum, needs nav_path; 0 for the datum combination itself); what is left is taken as 0 on
        all codes of the mode for that satellite and epoch and named in one warning (never a
        silent 0). Product phase OSBs of satellites the product does not give are taken as 0 (absorbed
        by the float ambiguity) with a warning; receiver OSBs (``STA_OSB_``) are kept and a receiver
        DSB of the pair becomes receiver OSBs, as without a product.
        """
        given = crd
        crd = attach_broadcast_group_delays(crd, nav, system)
        first = [cols[0] for cols in by_signal]
        weights = (1.0,)
        if len(first) == 2:
            k1, k2 = ionosphere_free_coefficients(*mode_signals(mode))
            weights = (k1, -k2)

        bds_datum = getattr(self, "_clock_datum", None)

        def combination(codes):
            return code_group_delay_correction(crd, system, tuple(codes), weights, clock="precise",
                                               clock_datum=bds_datum)

        # every code of the mode in the combination with the first code of the other signal, so that
        # the codes the filters use (the first of each signal) take one source per satellite and epoch
        results = [combination(first)]
        biases = dict(zip(first, results[0].biases_m))
        for i, cols in enumerate(by_signal):
            for code in cols[1:]:
                codes = list(first)
                codes[i] = code
                results.append(combination(codes))
                biases[code] = results[-1].biases_m[i]
        # reported for the codes the filters use (the other codes of a signal are written for completeness)
        missing: List[str] = list(results[0].missing)
        per_satellite: List[tuple] = list(results[0].per_satellite)

        values = np.vstack([biases[code] for code in first])
        datum = results[0].source == "datum"
        if datum.any() and self.config.positioning_mode != "combined":
            # the clock-datum combination without product biases has the group delay 0; for the
            # uncombined filters each code takes the broadcast group delay moved to the datum when
            # available (keeps the satellite DCB of the pair, ionosphere-like), as without a product
            # (the helper already gives them to the satellites the product lacks)
            zero = datum & (values == 0.0).all(axis=0)
            broadcast = {code: satellite_code_bias_m(crd, system, code, clock="precise", use_product=False,
                                                     clock_datum=bds_datum)[0]
                         for cols in by_signal for code in cols}
            rows = zero & np.isfinite(np.vstack([broadcast[code] for code in first])).all(axis=0)
            for code in biases:
                biases[code] = np.where(rows & np.isfinite(broadcast[code]), broadcast[code], biases[code])
            dcb = (f"the satellite DCB of the pair (ionosphere-like) is taken as that of the broadcast group delays "
                   f"moved to the precise clock datum; their IF combination, the clock datum, needs none")
            given_rows = datum & np.isfinite(np.vstack([biases[code] for code in first])).all(axis=0) \
                & (np.vstack([biases[code] for code in first]) != 0.0).any(axis=0)
            sv = (crd.index.get_level_values("sv") if "sv" in crd.index.names else crd["sv"])
            sv = pd.Index(sv).astype(str).str[:3].to_numpy()

            def name_satellites(rows, template):
                names = sorted(set(sv[rows]))
                per_satellite.extend((template, name) for name in names)
                missing.extend(template.format(satellites=name) for name in names)

            if given_rows.any() and zero.all():
                # the product gives neither code (e.g. COD0OPSFIN has no BeiDou)
                missing.append(f"product bias of {first[0]}, {first[1]} (not in the bias product): {dcb}")
            elif given_rows.any():
                name_satellites(given_rows, f"product bias of {first[0]}, {first[1]} for {{satellites}} (not in the "
                                            f"bias product): {dcb}")
            if (zero & ~rows).any() and self.config.use_iono_constr:
                name_satellites(zero & ~rows,
                                f"satellite DCB {first[0]}-{first[1]} of the uncombined code observations for "
                                f"{{satellites}} (no product bias, no broadcast group delay; ionosphere-like, biases "
                                f"the constrained ionosphere)")
        # a row without the bias of one code the filters use gets none on any code (one code of a pair
        # corrected alone shifts the IF combination by its whole k*b)
        incomplete = ~np.isfinite(np.vstack([biases[code] for code in first])).all(axis=0)
        self._warn_group_delay(system, mode, "precise", group_satellite_messages(missing, per_satellite),
                               all_codes=len(first) > 1 and bool(incomplete.any()))
        crd = crd.copy()
        for code, b in biases.items():
            # OSB_<code> = b(code) [ns]; the correction of the code is -b
            crd[f"OSB_{code}"] = np.where(incomplete | ~np.isfinite(b), 0.0, b) / NS_TO_M
        self._receiver_dsb_as_osb(crd, given, first, mode)
        return crd

    def _drop_partial_receiver_osb(self, crd: pd.DataFrame, system: str, mode: str) -> pd.DataFrame:
        """Receiver code OSBs (``STA_OSB_``) of a dual-frequency mode given for one code of the pair only
        are dropped on those rows (the filters take a missing receiver OSB as 0: one code corrected alone
        is a receiver DCB of that whole OSB, which biases the ionosphere of the uncombined filters; the
        IF combination's receiver part is absorbed by the receiver clock either way), with a warning."""
        if crd is None or len(crd) == 0:
            return crd
        by_signal = self._mode_code_columns(crd, system, mode)
        if len(by_signal) != 2 or any(not cols for cols in by_signal):
            return crd
        first = [cols[0] for cols in by_signal]
        columns = [f"STA_OSB_{code}" for code in first]
        given = [crd[col].notna().to_numpy() if col in crd.columns else np.zeros(len(crd), dtype=bool)
                 for col in columns]
        partial = given[0] ^ given[1]
        if not partial.any():
            return crd
        lacking = [code for code, has in zip(first, given) if (partial & ~has).any()]
        warnings.warn(
            f"PPP {system_name(system)} {mode}: receiver code OSB of {', '.join(lacking)} not in the bias product "
            f"for station {self.config.station_name!r} while that of the other code of the pair is: no receiver "
            f"OSB is applied to the pair ({int(partial.sum())} observations).",
            RuntimeWarning,
            stacklevel=4,
        )
        crd = crd.copy()
        for col in columns:
            if col in crd.columns:
                crd[col] = np.where(partial, np.nan, crd[col].to_numpy(dtype=float))
        return crd

    def _fill_phase_osb_gaps(self, crd: pd.DataFrame, system: str, mode: str) -> pd.DataFrame:
        """Product phase OSBs of the mode's phase observables left NaN (a satellite, or epochs, the
        product does not give; the IF filters read the columns as they are, NaN stopped them).

        A satellite the product gives at other epochs keeps its nearest value (a phase bias set to 0
        within an arc would jump, which a constant float ambiguity does not absorb); one it never gives
        takes 0 (a constant phase bias is absorbed by the float ambiguity, but its ambiguity is not an
        integer). Both are named in one warning per observable.
        """
        if crd is None or len(crd) == 0:
            return crd
        signals = set(mode_signals(mode))
        sv = crd.index.get_level_values("sv") if "sv" in crd.index.names else None
        copied = False
        # the phase observables the filters use (one per signal of the mode); other OSB_L columns are not read
        for obs_col in [c for c in crd.columns if isinstance(c, str) and len(c) in (2, 3) and c[0] == "L"
                        and c[1].isdigit()]:
            col = f"OSB_{obs_col}"
            try:
                if col not in crd.columns or code_signal(system, "C" + obs_col[1:]) not in signals:
                    continue
            except ValueError:
                continue
            values = crd[col].to_numpy(dtype=float)
            gaps = ~np.isfinite(values)
            if not gaps.any():
                continue
            filled = np.where(gaps, 0.0, values)
            if sv is not None and np.isfinite(values).any():
                order = np.argsort(crd.index.get_level_values("time"), kind="stable") \
                    if "time" in crd.index.names else np.arange(len(crd))
                in_time = pd.Series(values[order], index=pd.Index(np.asarray(sv)[order], name="sv"))
                in_time = in_time.groupby(level="sv").ffill().groupby(level="sv").bfill()
                nearest = np.empty(len(values))
                nearest[order] = in_time.to_numpy(dtype=float)
                held = gaps & np.isfinite(nearest)
                filled = np.where(held, nearest, filled)
                names = pd.Index(sv).astype(str).str[:3].to_numpy()
                parts = []
                if (gaps & ~held).any():
                    ar = " They are excluded from PPP-AR (pppar_enabled)." \
                        if getattr(self.config, "pppar_enabled", False) else ""
                    parts.append(f"for {', '.join(sorted(set(names[gaps & ~held])))} not in the bias product, "
                                 f"taken as 0 (absorbed by the float ambiguity; the ambiguities of these satellites "
                                 f"are not integer).{ar}")
                if held.any():
                    parts.append(f"for {', '.join(sorted(set(names[held])))} missing at some epochs, their value "
                                 f"at the nearest epoch the product gives is used.")
                warnings.warn(
                    f"PPP {system_name(system)} {mode}: phase OSB of {obs_col} " + " Phase OSB ".join(parts),
                    RuntimeWarning,
                    stacklevel=4,
                )
            if not copied:
                crd, copied = crd.copy(), True
            crd[col] = filled
        return crd

    @staticmethod
    def _mode_phase_observables(crd: pd.DataFrame, system: str, mode: str) -> List[str]:
        """Phase observables of the mode present in ``crd`` (one per signal), e.g. ['L1C', 'L2W']."""
        signals = set(mode_signals(mode))
        found = []
        for obs_col in [c for c in crd.columns if isinstance(c, str) and len(c) in (2, 3) and c[0] == "L"
                        and c[1].isdigit()]:
            try:
                if code_signal(system, "C" + obs_col[1:]) in signals:
                    found.append(obs_col)
            except ValueError:
                continue
        return found

    #: per-observation flag (since v1.0.0): RINEX LLI bit 1 on a phase of the mode
    HALF_CYCLE_COLUMN = "lli_half_cycle"

    def _mark_half_cycle(self, crd: pd.DataFrame, system: str, mode: str) -> pd.DataFrame:
        """Column ``lli_half_cycle``: True where the RINEX loss-of-lock indicator of a phase of the
        mode has bit 1 set, "half-cycle ambiguity/slip possible" (RINEX 3.03 §5.4 and Appendix Table A3, RINEX 4.00
        §5.2.13 and §6.7.1; ``rinex_obs.lli_half_cycle``, bit ``rinex_obs.LLI_HALF_CYCLE``). Read from the
        reader's LLI frames (``rinex_obs.lli_for``, the parse the session already made); False where the file gives
        no LLI (blank indicator, RINEX 2 phases the reader names differently) or cannot be read. The float solution
        does not use it; PPP-AR (``pppar``) excludes the flagged ambiguities from fixing and the slip layer starts a
        new arc when the flag sets or clears. This is RTKLIB's relative-positioning rule (2.4.2 p13 rtkpos.c:592;
        2.5.1 ex rtkpos.c:708-720, 836-837); RTKLIB PPP instead resets the ambiguity at every epoch while bit 0 or
        bit 1 is set (2.4.2 p13 ppp.c:715 ``detslp_ll``, ``LLI & 3``; 2.5.1 ex ppp.c:462)."""
        if crd is None or len(crd) == 0:
            return crd
        flag = np.zeros(len(crd), dtype=bool)
        phases = self._mode_phase_observables(crd, system, mode)
        names = list(getattr(crd.index, "names", None) or [])
        if phases and self.config.obs_path and {"sv", "time"} <= set(names):
            from ..io.rinex_obs import lli_for, lli_half_cycle

            try:
                lli = lli_for(self.config.obs_path, system, phases, tlim=getattr(self.config, "time_limit", None))
            except (OSError, ValueError):
                lli = None
            if lli is not None and len(lli):
                half = lli_half_cycle(lli).any(axis=1)
                half = half[~half.index.duplicated()]
                keys = pd.MultiIndex.from_arrays(
                    [pd.Index(crd.index.get_level_values("sv")).astype(str).str[:3],
                     pd.DatetimeIndex(crd.index.get_level_values("time"))], names=["sv", "time"])
                flag = half.reindex(keys, fill_value=False).to_numpy(dtype=bool)
        crd = crd.copy()
        crd[self.HALF_CYCLE_COLUMN] = flag
        return crd

    def _warn_l5_ifcb(self, crd: Optional[pd.DataFrame], mode: Optional[str], antenna) -> None:
        """One warning per run (a warning and a documented limitation in v1): GPS L1L5 / L2L5 / L5 with precise clocks and Block IIF satellites in the data. The IIF satellites carry
        a time-varying L5 inter-frequency clock bias (IFCB) that the IF(L1,L2) precise clocks do not contain
        (Montenbruck et al. 2012, GPS Solutions 16:303-313): d(t) = IF(L1,L2) - IF(L1,L5) of the phases varies by
        up to ~20 cm peak to peak within an arc (BRUX 2024-035, 10-min smoothing: G09 19.8 cm, G06 13.9 cm, IIF
        median 7.8 cm; GPS III 0.1-0.4 cm), the same at BRUX, NYA2 and NKLG (a satellite effect). An L5-only phase
        delay b enters IF(L1,L5) as -1.261 b and IF(L2,L5) as -11.255 b (8.9 x d): BRUX 2024-035 6-h static IF
        0.14 m (L1L5) and 1.3 m (L2L5) against 0.016 m (L1L2), whatever the bias product. The single-frequency L5
        mode takes b itself (d / 1.261, up to ~16 cm peak to peak; not measured)."""
        if crd is None or len(crd) == 0 or mode not in ("L1L5", "L2L5", "L5") or self.config.orbit_type != "precise":
            return
        names = list(getattr(crd.index, "names", None) or [])
        if "sv" not in names:
            return
        svs = sorted(set(pd.Index(crd.index.get_level_values("sv")).astype(str).str[:3]))
        atx = getattr(antenna, "atx", None)
        if atx is None and self.config.atx_path:
            from ..models.antex import load_antex

            try:
                atx = load_antex(self.config.atx_path)
            except (OSError, ValueError):
                atx = None
        if atx is not None:
            epoch = pd.DatetimeIndex(crd.index.get_level_values("time")).min().to_pydatetime() \
                if "time" in names else None
            iif = [sv for sv in svs
                   if str(getattr(atx.satellite(sv, epoch), "type", "") or "").strip().upper().startswith("BLOCK IIF")]
            if not iif:
                return
            which = f"the Block IIF satellites {', '.join(iif)}"
        else:
            which = "Block IIF satellites (no ANTEX to name them; IIF satellites are in orbit since 2010)"
        _notes.warn_once(
            self.config, ("gps_l5_ifcb", mode),
            f"PPP GPS {mode} with precise clocks: {which} carry a time-varying inter-frequency clock bias of L5 "
            f"(IFCB) that the IF(L1,L2) precise clocks do not contain: up to ~20 cm peak to peak within an arc in "
            f"IF(L1,L5), 8.9 times that in IF(L2,L5), 1/1.261 of it on L5 alone (BRUX 2024-035 6-h static: 0.14 m "
            f"L1L5, 1.3 m L2L5, 0.016 m L1L2). Expect dm-level (L1L5, L5) to m-level (L2L5) position errors; GNX "
            f"has no IFCB model (use L1L2 with precise products)", UserWarning, stacklevel=3)

    def _mark_ar_phase_bias(self, crd: pd.DataFrame, system: str, mode: str) -> pd.DataFrame:
        """Column ``AR_PHASE_BIAS_COLUMN``: the satellite has product phase OSBs for every phase observable
        of the mode, so its ambiguities can be integer; the AR engine takes only these satellites as
        candidates (a satellite without phase OSB is excluded, not used with 0).

        Broadcast orbits and clocks, a run without ``dcb_path`` and a product without phase OSBs
        (DCB-only, e.g. CAS) give no candidate at all. A satellite the product gives at some
        epochs only keeps its candidacy: its value at the nearest epoch is used.
        """
        if crd is None or len(crd) == 0 or not getattr(self.config, "pppar_enabled", False):
            return crd
        eligible = np.zeros(len(crd), dtype=bool)
        phase_obs = self._mode_phase_observables(crd, system, mode)
        # phase observables of the mode the product gives for no satellite (named in the warning)
        missing = [obs for obs in phase_obs if f"OSB_{obs}" not in crd.columns
                   or not np.isfinite(crd[f"OSB_{obs}"].to_numpy(dtype=float)).any()]
        if not hasattr(self, "_ar_phase_missing"):
            self._ar_phase_missing = {}
        self._ar_phase_missing[system] = missing if phase_obs else ["(none of the mode)"]
        if self.config.orbit_type == "precise" and self.config.dcb_path and phase_obs and not missing:
            eligible[:] = True
            sv = crd.index.get_level_values("sv") if "sv" in crd.index.names else pd.RangeIndex(len(crd))
            for obs in phase_obs:
                finite = pd.Series(np.isfinite(crd[f"OSB_{obs}"].to_numpy(dtype=float)), index=sv)
                eligible &= finite.groupby(level=0).transform("any").to_numpy(dtype=bool)
        crd = crd.copy()
        crd[AR_PHASE_BIAS_COLUMN] = eligible
        return crd

    def _rinex2_unknown_phase_tracking(self, processor, obs_by_sys: dict, selected) -> dict:
        """RINEX 2 observations: ``{system: [RINEX 2 phase observables]}`` of the selected
        systems whose tracking mode neither ``rinex2_tracking`` nor the converter's header block gives ({} for
        RINEX 3/4 files). An L2 phase tracked on L2C can be a quarter cycle off L2W, so such ambiguities are not
        integer with the product's phase biases."""
        if not getattr(self.config, "pppar_enabled", False):
            return {}
        frames = {s: obs_by_sys.get(s) for s in selected}
        adapters = {s: (f.attrs.get("gnx_adapter") or {}) for s, f in frames.items() if f is not None}
        if not any(a.get("source_format") == "RINEX2" for a in adapters.values()):
            return {}
        block = {}
        getter = getattr(processor, "_rinex2_header_tracking", None)
        if getter is not None:
            try:
                block = getattr(getter(), "entries", {}) or {}
            except (OSError, ValueError, AttributeError):  # no readable header block: the tracking is unknown
                block = {}
        setting = getattr(self.config, "rinex2_tracking", None) or {}
        unknown = {}
        for s, a in adapters.items():
            if a.get("source_format") != "RINEX2":
                continue
            known = set(block.get(s, {})) | set(setting.get(s, {}))
            phases = sorted({str(v.get("source")) for k, v in (a.get("column_map") or {}).items()
                             if str(k).startswith("L") and isinstance(v, dict) and v.get("source")})
            lacking = [p for p in phases if p not in known]
            if lacking or not phases:
                unknown[s] = lacking or ["(phases)"]
        return unknown

    def _warn_rinex2_ar(self, crd_by_sys: dict) -> bool:
        """PPP-AR with RINEX 2 phases of unknown tracking: one warning and AR off for the run. True when
        AR was switched off here."""
        unknown = getattr(self, "_rinex2_ar_unknown", None) or {}
        if not unknown:
            return False
        for s, crd in crd_by_sys.items():
            if crd is not None and len(crd) and AR_PHASE_BIAS_COLUMN in crd.columns:
                crd = crd.copy()
                crd[AR_PHASE_BIAS_COLUMN] = False
                crd_by_sys[s] = crd
        what = "; ".join(f"{system_name(s)} {', '.join(p)}" for s, p in sorted(unknown.items()))
        warnings.warn(
            f"PPP-AR (pppar_enabled) is switched off for this RINEX 2 observation file: the tracking mode of the "
            f"phase observables ({what}) is not known (RINEX 2 has no attribute), and an L2 phase tracked on L2C "
            f"can be a quarter cycle off L2W, so the ambiguities are not integer with the product's phase biases. "
            f"The result is the float solution. rinex2_tracking naming every phase of the file (e.g. "
            f"{{'G': {{'L1': 'L1C', 'L2': 'L2W'}}}}) or a converter header block naming the phases enables it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return True

    def _warn_ar_phase_bias(self, crd_by_sys: dict) -> None:
        """One warning when PPP-AR has no candidate in a system: no satellite phase biases."""
        if not getattr(self.config, "pppar_enabled", False):
            return
        if self.config.positioning_mode == "combined" and not combined_if_pppar_enabled(self.config):
            return  # the route warning (PPP-015) says that AR is off
        if self._warn_rinex2_ar(crd_by_sys):
            return  # RINEX 2 phases of unknown tracking (D11), said by that warning
        if self._warn_ar_product_mismatch(crd_by_sys):
            return  # AR disabled for mismatched products (NQ-PPP-24), said by that warning
        lacking = [s for s, crd in crd_by_sys.items()
                   if crd is not None and len(crd) and AR_PHASE_BIAS_COLUMN in crd.columns
                   and not crd[AR_PHASE_BIAS_COLUMN].any()]
        if not lacking:
            return
        if self.config.orbit_type != "precise":
            why = "broadcast orbits and clocks have no satellite phase biases"
        elif not self.config.dcb_path:
            why = "no bias product is given (dcb_path), so there are no satellite phase biases"
        else:
            missing = getattr(self, "_ar_phase_missing", {})
            obs = ", ".join(f"{system_name(s)} {'/'.join(missing.get(s) or ['?'])}" for s in sorted(lacking))
            why = f"the bias product in dcb_path gives no phase OSB for {obs} (a DCB-only product has none; " \
                  f"a phase OSB of another tracking mode of the band is not substituted)"
        names = ", ".join(system_name(s) for s in sorted(lacking))
        others = sorted(set(s for s, crd in crd_by_sys.items() if crd is not None and len(crd)) - set(lacking))
        scope = "PPP-AR is disabled and the result is the float solution" if not others else \
            f"PPP-AR is disabled for {names} (float ambiguities); only " \
            f"{', '.join(system_name(s) for s in others)} ambiguities can be fixed"
        warnings.warn(
            f"PPP-AR (pppar_enabled) needs satellite phase biases (phase OSB) consistent with the clocks: for "
            f"{names} {why}. {scope}. A precise product with phase OSBs (e.g. GRG0MGXFIN or COD0MGXFIN OSB) "
            f"in dcb_path, with its own orbits and clocks, enables it.",
            RuntimeWarning,
            stacklevel=3,
        )

    #: solution identifier of an IGS long product name, e.g. GRG0MGXFIN in GRG0MGXFIN_20240350000_01D_01D_OSB.BIA
    _PRODUCT_ID = re.compile(r"^([A-Z0-9]{3}[0-9][A-Z0-9]{3}[A-Z]{3})_[0-9]{11}_")

    def _warn_ar_product_mismatch(self, crd_by_sys: dict) -> bool:
        """PPP-AR with the phase OSBs of another solution than the orbits and clocks: one warning, and AR
        disabled unless ``pppar_allow_mixed_products``. Returns True when AR
        was disabled here (every satellite loses its AR candidacy, the result is the float solution).

        Satellite phase biases are estimated together with the satellite clocks; with the clocks of another
        analysis centre or campaign the ambiguities are not integer and fixes can be wrong. Products are
        compared by the solution identifier of their IGS long names (``*_OSB.BIA`` in ``dcb_path`` against
        ``*_ORB.SP3`` / ``*_CLK.CLK`` in ``sp3_path``, and against ``*_CLK.CLK`` in ``clk_path`` when the
        filter uses those clocks); other file names are not compared. They are consistent
        only when every compared file is of the one OSB solution (before v1.0.0 the OSB was
        compared with the union of the SP3 solutions, so COD, COD, GRG SP3 files with the GRG OSB kept AR on
        although the clocks of the processed day were COD's). The warning names each solution with the days
        of its files.
        """
        if self.config.orbit_type != "precise" or self.config.positioning_mode == "combined":
            return False
        if not any(crd is not None and len(crd) and AR_PHASE_BIAS_COLUMN in crd.columns
                   and crd[AR_PHASE_BIAS_COLUMN].any() for crd in crd_by_sys.values()):
            return False
        from ..core.session_guards import describe_solutions, product_solutions, solution_years

        bias = product_solutions(self.config.dcb_path, ("OSB",), self._PRODUCT_ID)
        orbits = product_solutions(self.config.sp3_path, ("ORB", "CLK"), self._PRODUCT_ID)
        # the 30 s clocks of clk_path are the filter's clocks when used (bug-fix stage 10): they must match too
        clocks = (product_solutions(self.config.clk_path, ("CLK",), self._PRODUCT_ID)
                  if getattr(self, "_clk_files", None) else {})
        if not bias or not (orbits or clocks) or len(set(bias) | set(orbits) | set(clocks)) == 1:
            return False
        allow = bool(getattr(self.config, "pppar_allow_mixed_products", False))
        if allow:
            outcome = ("pppar_allow_mixed_products=True keeps PPP-AR on: fixes can be wrong (the validation "
                       "rejects most of them)")
        else:
            outcome = ("PPP-AR is disabled and the result is the float solution; pppar_allow_mixed_products=True "
                       "forces it")
            for s, crd in crd_by_sys.items():
                if crd is not None and len(crd) and AR_PHASE_BIAS_COLUMN in crd.columns:
                    crd = crd.copy()
                    crd[AR_PHASE_BIAS_COLUMN] = False
                    # reported as AR reason mixed_products, not no_phase_bias
                    crd[AR_MIXED_PRODUCTS_COLUMN] = True
                    crd_by_sys[s] = crd
        years = solution_years(bias, orbits, clocks)       # one day format for the three products
        sources = ([f"the orbits and clocks in sp3_path from {describe_solutions(orbits, years)}"] if orbits
                   else []) + ([f"the clocks in clk_path from {describe_solutions(clocks, years)}"] if clocks else [])
        warnings.warn(
            f"PPP-AR (pppar_enabled): the phase OSBs in dcb_path come from {describe_solutions(bias, years)}, "
            f"{' and '.join(sources)}. Satellite phase biases are "
            f"consistent only with the clocks estimated with them: with another solution's clocks the "
            f"ambiguities are not integer. {outcome}. Use the OSB product of the orbit/clock solution"
            f"{' and one solution for every SP3 day' if len(orbits) > 1 else ''}.",
            RuntimeWarning,
            stacklevel=4,
        )
        return not allow

    def _warn_ar_route(self, ppp_filter) -> None:
        """PPP-AR settings the selected route cannot honour, or honours only experimentally."""
        if not getattr(self.config, "pppar_enabled", False):
            return
        # by class name: the filter classes of this module may be wrapped (spies in the test harness)
        name = type(ppp_filter).__name__
        if name == "PPPSingleFreqSingleGNSS":
            warnings.warn(
                "pppar_enabled=True is ignored by single-frequency PPP without ionospheric constraints "
                "(use_iono_constr=False): its code/phase average (GRAPHIC) has an ambiguity of half a "
                "carrier ambiguity plus code and phase biases, which is not an integer, so the result is "
                "the float solution. Single-frequency PPP-AR needs use_iono_constr=True.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        if self.config.positioning_mode != "combined":
            return
        if not combined_if_pppar_enabled(self.config):
            warnings.warn(
                "pppar_enabled=True is ignored by combined (ionosphere-free) PPP: the result is the float "
                "solution. The ionosphere-free ambiguity is an integer only together with the wide lane "
                "(N1 + f2/(f1 - f2) * N_WL in narrow-lane cycles), which this route does not separate. Use "
                "positioning_mode='uncombined' for PPP-AR.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        if set(self.config.sys) == {"C"}:
            return  # the BeiDou filter says that its AR is off
        warnings.warn(
            f"Combined (ionosphere-free) PPP-AR (pppar_combined_if_ar_enabled=True) fixes no ambiguity "
            f"(systems {''.join(sorted(self.config.sys))}): the result is the float solution. The "
            f"ionosphere-free ambiguity is an integer only together with the wide lane (N1 + f2/(f1 - f2) * "
            f"N_WL in narrow-lane cycles), which this route does not separate; fixing it directly gave wrong "
            f"integers. Use positioning_mode='uncombined' for wide-lane/narrow-lane PPP-AR.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _warn_group_delay(self, system: str, mode: str, clock: str, missing, all_codes: bool = False) -> None:
        for item in missing:
            key = (system, mode, item)
            if key in self._group_delay_reported:
                continue
            self._group_delay_reported.add(key)
            replaced = "is used" in item or ("broadcast group delay" in item and "takes" in item) \
                or "is taken as" in item or "no satellite group delay is applied" in item
            consequence = "" if replaced else " A missing group delay is taken as 0."
            if all_codes and not replaced:
                consequence = (f" A missing group delay is taken as 0, and so are those of the other codes of "
                               f"{mode} on the same satellite and epoch.")
            if clock == "broadcast" and item.startswith("broadcast"):
                remedy = "navigation records with these fields in nav_path provide it"
            else:
                remedy = "a Bias-SINEX product with these codes in dcb_path (e.g. CAS DSB) provides it"
            warnings.warn(
                f"PPP {system_name(system)} {mode} ({clock} clocks), satellite group delay not available: "
                f"{item}.{consequence} Typical size: {missing_correction_hint(system, item)}; {remedy}.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _measure_STEC(self, df:pd.DataFrame, system:str):
        # DOWNLOAD DCB FROM GIM
        parser = GIMReader(tec_path=self.config.gim_path,dcb_path=self.config.gim_path)
        data= parser.read()
        if data.dcb is not None:
            sat_dcb = data.dcb[(data.dcb['entry_type'] == 'satellite') &
                               data.dcb['sv'].str.startswith(system)]
            sat_dcb = sat_dcb.set_index('sv')
            df['sv_main'] = df.index.get_level_values('sv')
            df['sv_main'] = df['sv_main'].apply(lambda x: x[:3])
            sat_dcb = sat_dcb.reset_index()
            merged = df.merge(sat_dcb, left_on='sv_main',right_on='sv',how='left')
            merged = merged.set_index(df.index)
            merged = merged.drop(columns='sv')
            df=merged.copy()
            if 'bias' in df.columns:
                df['bias'] = df['bias'].fillna(0.0)

            name = self.config.station_name
            station = (data.dcb[(data.dcb['entry_type'] == 'station') &
                                (data.dcb['prn_or_site'].str.startswith(name))])
            sta_bias = station[station['sys'] == system]
            if not sta_bias.empty:
                df['sta_bias'] = sta_bias['bias'].values[0]
            else:
                df['sta_bias'] = 0.0
        # MEASURE STEC
        ## create a config
        tec_config =  TECConfig(station_name = self.config.station_name,day_of_year = self.config.day_of_year)
        ## create a monitor
        if system =='G':
            mode = self.config.gps_freq
        elif system =='E':
            mode = self.config.gal_freq
        elif system == 'C':
            mode = self.config.bds_freq
        else:
            raise ValueError(f"Unsupported system for STEC measurement: {system}")
        monitor =  STECMonitor(config=tec_config,sys=system, mode=mode,obs=df)
        out = monitor.run()
        return out

    def _load_nav_data(self, processor):
        """ load broadcast navigation messages for screening"""
        broadcast = processor.load_broadcast_orbit(tlim=self.config.time_limit)
        return broadcast

    # ‑‑‑ private helpers ----------------------------------------------------
    def _load_obs_data(self, processor):  # noqa: D401 – simple phrase OK
        """Read observation & navigation files, apply basic screenings."""
        obs = processor.load_obs_data(tlim=self.config.time_limit)

        return obs

    def _screen_data(self, processor, gps_obs, gal_obs, bds_obs, broadcast):
        """``screen=True``: outage flags in force (Galileo: the DVS/SHS flags of the mode's signals) and
        the presence of the mode's phases (``session_guards.screen_observations``)."""
        modes = {'G': self.config.gps_freq, 'E': self.config.gal_freq, 'C': self.config.bds_freq}
        systems = [s for s in ('G', 'E', 'C') if self.config.system_includes(s)]
        out = screen_observations(processor, {'G': gps_obs, 'E': gal_obs, 'C': bds_obs}, broadcast, modes, systems)
        return out['G'], out['E'], out['C']

    def _satellite_states(self, obs, sys, mode, orbit=None, sp3_df=None):
        """Satellite positions and clocks at emission of the observation frame ``obs`` (broadcast or precise
        orbits; with ``clk_path``, the 30 s clocks of :mod:`gnx_py.rinex_clk`), before any use of the receiver
        position. Precise orbits give the index (time, sv), broadcast orbits (sv, time), as before."""
        if self.config.orbit_type == "broadcast":
            make_ionofree(obs_df=obs, mode=mode, sys=sys)
            orb = {'G': orbit.gps_orb, 'E': orbit.gal_orb, 'C': orbit.bds_orb}.get(sys)
            if sys not in ('G', 'E', 'C'):
                raise ValueError("Invalid sys: %s", sys)
            # broadcast_tolerance is not passed: BroadcastInterp never reads its tolerance argument (PPP-028, Q16)
            interpolator = BroadcastInterp(obs=obs, mode=mode, sys=sys, nav=orb, emission_time=True)
            return interpolator.interpolate()
        make_ionofree(obs_df=obs, mode=mode)
        obs = obs.swaplevel('sv', 'time')
        degree = self.config.interpolation_degree
        # the run's configuration keeps the warn-once state of the SP3 node check (once per run, not per call)
        _, positions = lagrange_reception_interp(obs=obs, sp3_df=sp3_df, degree=degree, owner=self.config)
        obs_crd = lagrange_emission_interp(obs=obs, positions=positions, sp3_df=sp3_df, degree=degree, mode=mode,
                                           owner=self.config)
        clk_files = getattr(self, "_clk_files", None)
        if clk_files:
            # rows inside a gap of the clock files: left out (clk_gap_policy 'drop', RTKLIB pephclk) or SP3 clock
            obs_crd = apply_clk_clocks(obs_crd, clk_files, gap_policy=self.config.clk_gap_policy)
        return obs_crd

    @staticmethod
    def _wrap(states, xyz, flh, mode):
        """Earth rotation during the signal travel, elevation and azimuth at the a-priori position."""
        wrapper = CustomWrapper(obs=states, epochs=None, flh=flh.copy(), xyz_a=xyz.copy(), mode=mode)
        return wrapper.run()

    def _resolve_apriori(self, obs_data, states, mode_by_sys):
        """A-priori position (``apriori.resolve_position``): ``approx_position``, else the header position when
        plausible and within max(10 m, 3 sqrt(p_crd), 4 spread) of the marker position from code observations
        (antenna minus ANTENNA: DELTA H/E/N), else the code position. With ``q_crd > 0`` the code position is the
        receiver's start. The 10 m floor needs a code position free of metre-level biases: static, dual-frequency,
        precise orbits and clocks. Otherwise the floor is 100 m, as in SPP and TEC: a single-frequency code carries
        the ionosphere; broadcast clocks and group delays leave metres (BeiDou: the B3I clock datum without TGD);
        a moving receiver's start rests on 5 epochs (measured: C/GC/EC/GEC BRDC kinematic rows at
        BRUX 035 had their header replaced by start positions 10.7-25.1 m away); a close-pair ionosphere-free code
        (L2L5, E5aE5b, B2aB2b: noise and code biases amplified 16.6-27.5 times, :data:`CLOSE_PAIR_IF_GAIN`) put
        the static precise code position 14.0 m (BRUX 035) and 14.2 m (NYA2 035) from the marker (v1.0.0
        revision; under the 10 m floor those headers stayed only through the 4 x spread term)."""
        meta = obs_data.meta
        p_crd = None
        if hasattr(self.config, "effective"):
            try:
                p_crd = float(self.config.effective().get("p_crd"))
            except (TypeError, ValueError):
                p_crd = None
        single = any(len(mode_signals(mode_by_sys[s])) < 2 for s in states)
        q_crd = self.config.effective().get("q_crd", 0) if hasattr(self.config, "effective") else 0
        try:
            moving = bool(q_crd) and float(q_crd) > 0
        except (TypeError, ValueError):
            moving = False
        precise = str(getattr(self.config, "orbit_type", "precise")).lower() == "precise"
        close_pair = not single and any(
            float(np.hypot(*ionosphere_free_coefficients(*mode_signals(mode_by_sys[s])))) > CLOSE_PAIR_IF_GAIN
            for s in states)
        floor = 10.0 if (precise and not single and not moving and not close_pair) else 100.0
        xyz, flh, source = resolve_position(
            meta[4], meta[5], states, mode_by_sys, floor_m=floor, p_crd=p_crd,
            approx_position=getattr(self.config, "approx_position", None), obs_path=self.config.obs_path,
            session="PPP", delta_hen=meta[3], kinematic=moving)
        self._apriori_xyz = np.asarray(xyz, dtype=float)
        self._apriori_source = source
        return np.asarray(xyz, dtype=float), np.asarray(flh, dtype=float)

    def _preprocess(self, obs, flh, xyz,
                    broadcast,phase_shift,sat_pco, rec_pco, antenna_h, system, antenna=None, phase_shift_attrs=None):

        if broadcast is None:
            gpsa, gpsb, gala = None, None, None
        else:
            gpsa, gpsb, gala = broadcast.gpsa, broadcast.gpsb, broadcast.gala
        prc =  DDPreprocessing(df=obs, flh=flh.copy(), xyz=xyz.copy(), config=self.config, gpsa=gpsa,
                                  gpsb=gpsb, gala=gala, phase_shift_dict=phase_shift, sat_pco=sat_pco, rec_pco=rec_pco,
                                  antenna_h=antenna_h, system=system, antenna=antenna,
                                  iono=getattr(broadcast, 'iono', None), run_notes=getattr(self, 'run_notes', None),
                                  phase_shift_attrs=phase_shift_attrs)
        obs_preprocessed = prc.run()
        return obs_preprocessed

    def _detect_all_cycle_slips(self, crd_by_sys, mode_by_sys, interval, xyz):
        """Arcs of continuous phase of every selected system (``interval`` in minutes, ``xyz`` the a-priori
        receiver position [m]).

        The slip context (``gnx_py.cycle_slips.build_slip_context``: LLI and the third carrier from the RINEX
        file, the ``cycle_slips_detection`` master switch) is built from exactly the frames received here, then
        :meth:`_detect_cycle_slips` runs per system.
        """
        from ..core.cycle_slips import build_slip_context

        q_crd = self.config.effective().get('q_crd', 0) if hasattr(self.config, 'effective') else 0
        self._slip_context = build_slip_context(
            self.config, crd_by_sys, mode_by_sys, xyz=xyz, interval_s=float(interval) * 60.0,
            windup_applied=False, kinematic=bool(q_crd and float(q_crd) > 0))
        out = dict(crd_by_sys)
        try:
            for sys in crd_by_sys:
                out[sys] = self._detect_cycle_slips(obs_df=crd_by_sys[sys], mode=mode_by_sys[sys], system=sys,
                                                    interval=interval)
        finally:
            # the context belongs to these frames: a later direct call must not align its third carrier
            self._slip_context = None
        return out

    def _detect_cycle_slips(self, obs_df, mode, system, interval):
        """Arcs of continuous phase (``gnx_py.tools.CSDetector``; ``interval`` in minutes).

        The detector uses the system's slip context of :meth:`_detect_all_cycle_slips` when there is one
        (``gnx_py.cycle_slips``; layers and measured rates in the CSDetector docstring):

        * RINEX loss-of-lock indicators of every phase (``cycle_slip_aux_carrier``);
        * the third-carrier tests (L1 for L2L5, E1 for E5aE5b, B1C for B2aB2b, L5 for L1L2, E5b for E1E5a,
          L2W / E5a / B3I for single-frequency L1 / E1 / B1I; every sampling interval since v1.0.0),
          which find equal slips on a close pair and one-cycle slips on a single frequency without products;
        * the geometry layer (``cycle_slip_geometry``; precise orbits and 30 s sampling only), shared by all
          systems of the run: one epoch-differenced test of every carrier with one receiver-clock change
          per epoch, which also finds a slip on all three carriers at once and equal slips under the polar
          ionosphere; with ``clk_path`` it uses 30 s clocks of the SP3's analysis centre (SP3 clocks where
          they do not cover the run), the clocks the filter uses as well since v1.0.0
          (``rinex_clk.apply_clk_clocks`` in :meth:`_satellite_states`). With ``q_crd > 0`` the receiver
          keeps this static model while it does not move; a moving receiver is tracked at its integrated
          position, or the layer is off for the run;
        * the fusion: a flag that only one layer raises must show a persistent ionosphere-free step
          (7 sigma, 7 cm), and a slip on the third carrier alone does not break the arc.

        Without products and third carrier (broadcast orbits and other modes, RINEX 2,
        ``cycle_slip_aux_carrier=False``) the limits of the combinations stay: equal slips on a
        close pair leave Melbourne-Wuebbena unchanged and move the geometry-free phase only by
        n*|lambda1 - lambda2| (1.06 / 0.65 cm per cycle); single-frequency code minus phase finds slips of
        several cycles only. ``cycle_slips_detection=False``: arcs end at data gaps and LLI only.
        """
        context = (getattr(self, '_slip_context', None) or {}).get(system)
        detector =  CSDetector(obs=obs_df,phase_shift_dict=None,dcb=None,sys=system,mode=mode,
                                     interval=interval*60, context=context)
        obs_df = detector.run(min_arc_lenth=self.config.min_arc_len)
        obs_df = obs_df.reset_index()
        obs_df = obs_df.set_index(['arc', 'time'])
        obs_df['arc']=obs_df.index.get_level_values('arc')
        obs_df.index.names = ['sv', 'time']
        obs_df = obs_df.drop(columns=['sv'])
        return obs_df

    def create_if_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        """Create a combined ionosphere-free PPP filter from active systems.

        Mixed-system input routes to ``PPPDualFreqMultiGNSS`` with a reference
        constellation clock and per-non-reference ISB states. Single-system
        input routes to ``PPPDualFreqSingleGNSS`` for GPS, Galileo or BeiDou.
        All branches require dual-frequency modes because the observables are
        formed as ionosphere-free combinations before filtering.

        Status:
            Active routing helper. It should not be changed without checking
            combined single-system and mixed-system regression tests.
        """
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq
        bds_mode = self.config.bds_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty
        has_bds = isinstance(obs_bds_crd, pd.DataFrame) and not obs_bds_crd.empty

        if sum((has_gps, has_gal, has_bds)) >= 2:
            return PPPDualFreqMultiGNSS(
                gps_obs=obs_gps_crd.copy() if has_gps else None,
                gal_obs=obs_gal_crd.copy() if has_gal else None,
                bds_obs=obs_bds_crd.copy() if has_bds else None,
                gps_mode=gps_mode,
                gal_mode=gal_mode,
                bds_mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
            )
        elif has_gps:
            # Single-GNSS / GPS
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gps_crd.copy(),
                gps_mode=gps_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
                system="G",
            )
        elif has_gal:
            # Single-GNSS na Galileo (we pass the GAL mode as gps_mode)
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_gal_crd.copy(),
                gps_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
                system="E",
            )
        elif has_bds:
            return PPPDualFreqSingleGNSS(
                gps_obs=obs_bds_crd.copy(),
                gps_mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
                config=self.config,
                system="C",
            )
        else:
            raise ValueError("No GPS, Galileo or BeiDou observations after preprocessing (None/empty).")

    def _frequency_kind(self, mode: str) -> str:
        if mode in SINGLE_FREQUENCY_MODES:
            return "single"
        if mode in DUAL_FREQUENCY_MODES:
            return "dual"
        raise ValueError(f"Unsupported uncombined frequency mode: {mode}")

    def _create_uncombined_single_system_filter(self, obs, mode, ekf, pos0, interval):
        """Create the single-system uncombined filter for one signal mode.

        Single-frequency modes route either to ``PPPUducSFSingleGNSS`` when
        ionospheric constraints are enabled, or to ``PPPSingleFreqSingleGNSS``
        without constraints. Dual-frequency modes route to ``PPPUdSingleGNSS``.

        Status:
            Active routing helper for single-system uncombined PPP.
        """
        kind = self._frequency_kind(mode)
        use_iono = bool(self.config.use_iono_constr)

        if kind == "single":
            if use_iono:
                return PPPUducSFSingleGNSS(
                    obs=obs,
                    mode=mode,
                    ekf=ekf,
                    pos0=pos0,
                    tro=True,
                    interval=interval,
                    use_iono_rms=self.config.use_iono_rms,
                    sigma_iono_0=self.config.sigma_iono_0,
                    sigma_iono_end=self.config.sigma_iono_end,
                    t_end=self.config.t_end,
                    config=self.config,
                )
            return PPPSingleFreqSingleGNSS(
                obs=obs,
                mode=mode,
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        return PPPUdSingleGNSS(
            obs=obs,
            mode=mode,
            ekf=ekf,
            pos0=pos0,
            tro=True,
            interval=interval,
            config=self.config,
        )

    def create_uncombined_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        """Create the uncombined PPP filter selected by systems and constraints.

        Routing ("active" = the systems with observations left at this point;
        ``use_iono_constr=None`` counts as False):

        ==========================================================  ==================================
        active systems and settings                                 filter class
        ==========================================================  ==================================
        >= 2 systems, no ionospheric constraints                    ``PPPUdGenericMixedGNSS``
        >= 2 systems, constraints, BeiDou active or
        ``uncombined_iono_constraints_model='gec'`` or broadcast    ``PPPFilterMultiGNSSIonConstGEC``
        G+E, constraints, 'legacy' model, precise, both DF          ``PPPFilterMultiGNSSIonConst``
        G+E, constraints, 'legacy' model, precise, >= 1 SF          ``PPPUdMixedGNSS``
        1 system, dual frequency                                    ``PPPUdSingleGNSS``
        1 system, single frequency, constraints                     ``PPPUducSFSingleGNSS``
        1 system, single frequency, no constraints                  ``PPPSingleFreqSingleGNSS``
        ==========================================================  ==================================

        Every class listed is reachable and no other class is (since v1.0.0;
        the unreachable G+C / E+C and unconstrained G+E
        branches and their classes ``PPPUducSFMultiGNSS``, ``PPPUdMultiGNSS``
        were removed). ``PPPFilterMultiGNSSIonConst`` is the reference
        implementation of the G/E dual-frequency constrained model.
        """
        gps_mode = self.config.gps_freq
        gal_mode = self.config.gal_freq
        bds_mode = self.config.bds_freq

        has_gps = isinstance(obs_gps_crd, pd.DataFrame) and not obs_gps_crd.empty
        has_gal = isinstance(obs_gal_crd, pd.DataFrame) and not obs_gal_crd.empty
        has_bds = isinstance(obs_bds_crd, pd.DataFrame) and not obs_bds_crd.empty
        use_iono = bool(self.config.use_iono_constr)
        iono_constraints_model = getattr(self.config, "uncombined_iono_constraints_model", "legacy")

        active = [s for s, present in (('G', has_gps), ('E', has_gal), ('C', has_bds)) if present]
        if len(active) > 1 and not use_iono:
            obs_by_system = {
                "G": obs_gps_crd,
                "E": obs_gal_crd,
                "C": obs_bds_crd,
            }
            mode_by_system = {
                "G": gps_mode,
                "E": gal_mode,
                "C": bds_mode,
            }
            return PPPUdGenericMixedGNSS(
                obs_by_system={system: obs_by_system[system] for system in active},
                mode_by_system={system: mode_by_system[system] for system in active},
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if use_iono and len(active) > 1 and (
            has_bds
            or iono_constraints_model == "gec"
            or self.config.orbit_type == "broadcast"
        ):
            obs_by_system = {
                "G": obs_gps_crd,
                "E": obs_gal_crd,
                "C": obs_bds_crd,
            }
            mode_by_system = {
                "G": gps_mode,
                "E": gal_mode,
                "C": bds_mode,
            }
            return PPPFilterMultiGNSSIonConstGEC(
                obs_by_system={system: obs_by_system[system] for system in active},
                mode_by_system={system: mode_by_system[system] for system in active},
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if has_gps and has_gal:
            # past the two checks above: constraints on, no BeiDou, the 'legacy' constraint model and precise
            # orbits (R4g, bug-fix stage 13b); _frequency_kind on both modes keeps the ValueError of an unknown mode
            gps_kind = self._frequency_kind(gps_mode)
            gal_kind = self._frequency_kind(gal_mode)
            if gps_kind == "dual" and gal_kind == "dual":
                return PPPFilterMultiGNSSIonConst(
                    gps_obs=obs_gps_crd,
                    gps_mode=gps_mode,
                    gal_obs=obs_gal_crd,
                    gal_mode=gal_mode,
                    ekf=ekf,
                    pos0=pos0,
                    tro=True,
                    est_dcb=True,
                    interval=interval,
                    use_iono_rms=self.config.use_iono_rms,
                    sigma_iono_0=self.config.sigma_iono_0,
                    sigma_iono_end=self.config.sigma_iono_end,
                    t_end=self.config.t_end,
                    config=self.config,
                )
            return PPPUdMixedGNSS(
                gps_obs=obs_gps_crd,
                gps_mode=gps_mode,
                gal_obs=obs_gal_crd,
                gal_mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                tro=True,
                interval=interval,
                config=self.config,
            )

        if has_gps:
            return self._create_uncombined_single_system_filter(
                obs=obs_gps_crd,
                mode=gps_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
            )

        if has_gal:
            return self._create_uncombined_single_system_filter(
                obs=obs_gal_crd,
                mode=gal_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
            )

        if has_bds:
            return self._create_uncombined_single_system_filter(
                obs=obs_bds_crd,
                mode=bds_mode,
                ekf=ekf,
                pos0=pos0,
                interval=interval,
            )

        raise ValueError("No GPS, Galileo or BeiDou observations after preprocessing (None/empty).")

    def create_uduc_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        return self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)

    def create_sf_filter(self, obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=None):
        return self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)

    #: uncombined settings of the no-constraint and of the constrained filters (PPPConfig docstring)
    _NO_CONSTRAINT_SETTINGS = (
        "uncombined_code_var", "uncombined_phase_var", "uncombined_code_prefit_threshold",
        "uncombined_phase_screen", "uncombined_phase_screen_threshold", "uncombined_phase_screen_warmup_epochs",
    )
    _CONSTRAINT_SETTINGS = (
        "uncombined_est_rx_dcb", "uncombined_iono_clock_identity", "uncombined_iono_code_prefit_threshold",
        "uncombined_iono_phase_screen", "uncombined_iono_phase_screen_threshold",
        "uncombined_iono_phase_screen_warmup_epochs", "uncombined_iono_phase_screen_legacy_source",
        "uncombined_iono_legacy_weighting", "uncombined_iono_code_var_by_system", "uncombined_iono_code_var_base",
        "uncombined_iono_code_var_elev_coeff", "uncombined_iono_phase_var_base", "uncombined_iono_phase_var_elev_coeff",
    )
    #: settings a filter class does not read: name -> (value the filter works with, or None when it has
    #: its own fixed model; what it does instead). Reported when set explicitly to another value.
    _FIXED_SETTINGS = {
        "PPPUdGenericMixedGNSS": {
            "q_dcb": (0.0, "the receiver signal-bias states of mixed PPP without constraints are constant"),
        },
        "PPPFilterMultiGNSSIonConstGEC": {
            "uncombined_est_rx_dcb": (True, "the receiver code-bias states are always estimated"),
        },
        "PPPFilterMultiGNSSIonConst": {
            "uncombined_est_rx_dcb": (True, "the receiver code-bias states are always estimated"),
            "uncombined_iono_phase_screen_legacy_source": (
                True, "the phase-jump screening always uses phase minus geometric range"),
        },
        "PPPUdMixedGNSS": {
            **{k: (None, "fixed weights 1/sin(el) m^2 (code) and 0.001/sin(el) m^2 (phase), a 30 m code gate, "
                         "no phase-jump screening, the clock re-estimated every epoch")
               for k in _NO_CONSTRAINT_SETTINGS + _CONSTRAINT_SETTINGS},
        },
        "PPPUducSFSingleGNSS": {
            **{k: (None, "fixed weights 1/sin(el) m^2 (code) and 0.001/sin(el) m^2 (phase), a 30 m code gate, "
                         "no phase-jump screening, the clock re-estimated every epoch")
               for k in _NO_CONSTRAINT_SETTINGS + _CONSTRAINT_SETTINGS},
        },
        "PPPSingleFreqSingleGNSS": {
            # GRAPHIC (bug-fix stage 8b): weighted with uncombined_code_var / uncombined_phase_var
            **{k: (None, "the code/phase average (GRAPHIC) is gated on its own prefit residual "
                         "(max(3 m, 5 sigma)) and has no phase-jump screening")
               for k in _NO_CONSTRAINT_SETTINGS[2:]},
        },
    }
    #: ionospheric-constraint settings outside the uncombined_* group
    _IONO_CONSTRAINT_SETTINGS = ("use_iono_rms", "sigma_iono_0", "sigma_iono_end", "t_end")
    #: filters that always apply ionospheric constraints
    _CONSTRAINED_FILTERS = ("PPPFilterMultiGNSSIonConstGEC", "PPPFilterMultiGNSSIonConst", "PPPUducSFSingleGNSS")

    def _warn_unused_settings(self, ppp_filter) -> None:
        """Warn about explicitly set PPP settings the selected filter does not use,
        and about the ionospheric constraint sigma of a model without RMS maps."""
        cfg = self.config
        name = type(ppp_filter).__name__
        fields = getattr(type(cfg), "__dataclass_fields__", {})
        defaults = cfg._PRESETS.get("uncombined", {})
        unused: dict[str, list[str]] = {}
        fixed = dict(self._FIXED_SETTINGS.get(name, {}))
        constrained = cfg.positioning_mode != "combined" and (
            name in self._CONSTRAINED_FILTERS or bool(getattr(ppp_filter, "use_iono_constr", False)))
        if cfg.positioning_mode == "combined":
            for key in self._NO_CONSTRAINT_SETTINGS + self._CONSTRAINT_SETTINGS + self._IONO_CONSTRAINT_SETTINGS:
                fixed.setdefault(key, (None, "used only by uncombined PPP; the ionosphere-free combination "
                                             "has its own weights and no ionospheric constraints"))
        elif constrained:
            # the constrained filters gate and screen with the uncombined_iono_* settings
            for key in self._NO_CONSTRAINT_SETTINGS[2:]:
                fixed.setdefault(key, (None, "used only without ionospheric constraints; with constraints the "
                                             "uncombined_iono_* gate and phase-jump screening apply"))
            if cfg.uncombined_iono_legacy_weighting:
                for key in self._NO_CONSTRAINT_SETTINGS[:2] + ("uncombined_iono_code_var_by_system",):
                    fixed.setdefault(key, (None, "used with ionospheric constraints only when "
                                                 "uncombined_iono_legacy_weighting=False; the observations are "
                                                 "weighted with the uncombined_iono_*_var_* settings"))
            else:
                # bug-fix stage 13b (PPP-074): code c_sys/sin(E) from uncombined_iono_code_var_by_system, phase
                # uncombined_phase_var/sin(E); the legacy elevation model and uncombined_code_var are not read
                for key in ("uncombined_iono_code_var_base", "uncombined_iono_code_var_elev_coeff",
                            "uncombined_iono_phase_var_base", "uncombined_iono_phase_var_elev_coeff"):
                    fixed.setdefault(key, (None, "used only with uncombined_iono_legacy_weighting=True; the "
                                                 "observations are weighted with uncombined_iono_code_var_by_system "
                                                 "(code) and uncombined_phase_var (phase)"))
                fixed.setdefault("uncombined_code_var", (None, "used only without ionospheric constraints; "
                                                               "with constraints the code is weighted with "
                                                               "uncombined_iono_code_var_by_system"))
        else:
            for key in self._CONSTRAINT_SETTINGS + self._IONO_CONSTRAINT_SETTINGS:
                fixed.setdefault(key, (None, "used only with ionospheric constraints (use_iono_constr=True)"))
        if name != "PPPFilterMultiGNSSIonConst":
            # bug-fix stage 13d (PPP-028): read only by the legacy G+E constrained filter (R4c, PPP-061); every
            # other route ignored it without a word
            fixed.setdefault("uncombined_iono_signal_bias_phase_jacobian", (
                None, "read only by the legacy GPS+Galileo constrained filter PPPFilterMultiGNSSIonConst, whose "
                      "receiver signal-bias states also enter the phase rows"))
        for key, (value, instead) in fixed.items():
            given = getattr(cfg, key)
            # set by the user: an EKF parameter overridden explicitly, an uncombined_* setting
            # (always registered with its default) or a constraint setting given with another
            # value than the default
            if key in fields and fields[key].default is MISSING:
                if not cfg.is_overridden(key):
                    continue
            elif key in defaults:
                if given == defaults[key]:
                    continue
            elif key not in fields or given == fields[key].default:
                continue
            if value is not None and given == value:
                continue
            unused.setdefault(instead, []).append(f"{key}={given!r}")
        if unused:
            parts = "; ".join(f"{', '.join(keys)} ({instead})" for instead, keys in unused.items())
            warnings.warn(
                f"PPP filter {name} does not use {parts}: these settings have no effect on this route.",
                UserWarning,
                stacklevel=3,
            )
        if constrained and cfg.use_iono_rms and cfg.ionosphere_model in ("klobuchar", "ntcm", "bdgim"):
            # information (use_iono_rms is on by default): the constraint sigma is modelled, not read
            # bug-fix stage 10b (D3, BDGIM) and 13d (NQ-PPP-45, Klobuchar and NTCM): a floor, so a model value near 0
            # does not make the constraint rigid
            sigma = ("hypot(50 % of the modelled delay, 5 TECU x mapping function) as ion_rms (the "
                     "broadcast-model error of RTKLIB with a 5 TECU zenith floor"
                     + (", from the unclamped model)" if cfg.ionosphere_model == "bdgim" else ")"))
            warnings.warn(
                f"ionosphere_model={cfg.ionosphere_model!r} has no RMS map: with use_iono_rms=True the "
                f"ionospheric constraint takes {sigma} instead of a GIM RMS; use_iono_rms=False uses the "
                f"sigma_iono_0/sigma_iono_end ramp.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _residuals_by_system(ppp_filter, first, second, processed: List[str]) -> dict:
        """``{system letter: residual frame}`` of a filter's return.

        Every PPP filter returns ``(solution, {system letter: residual frame}, None, hours)`` with exactly the
        systems it processed, and the frames are routed by key only. Any other shape, the
        positional pre-C1 tuple ``(frame, frame | None)`` included, raises TypeError naming the filter, so a residual
        frame is never dropped or put in another system's field silently (before v1.0.0 a Galileo-only IF run landed
        in ``residuals_gps`` and a GPS R7 run filled ``residuals_gal`` with a copy of its frame)."""
        if isinstance(first, dict):
            return first
        raise TypeError(
            f"PPP filter {type(ppp_filter).__name__}.run_filter returned residuals of type "
            f"{type(first).__name__}/{type(second).__name__} for the systems {processed}: expected "
            f"{{system letter: residual frame}} (contract C1)")

    def _run_filter(self, obs_data, obs_gps_crd, obs_gal_crd, obs_bds_crd, interval) -> PPPResult:  # noqa: D401 – simple phrase
        """Select, run, and normalize the configured PPP filter."""
        from .ppp_helpers import GNXExtendedKalmanFilter as ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(dim_x=1,dim_z=1)
        # the a-priori position of run() (header, approx_position or code observations, bug-fix stage 10)
        pos0 = getattr(self, "_apriori_xyz", None)
        if pos0 is None:
            pos0 = obs_data.meta[4]

        if self.config.positioning_mode == 'combined':
            ppp_filter = self.create_if_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)
        elif self.config.positioning_mode == 'uncombined':
            ppp_filter = self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)
        elif self.config.positioning_mode == 'single':
            warnings.warn(
                "positioning_mode='single' is deprecated; use positioning_mode='uncombined' with a single-frequency mode.",
                DeprecationWarning,
                stacklevel=2,
            )
            ppp_filter = self.create_uncombined_filter(obs_gps_crd, obs_gal_crd, ekf, pos0, interval, obs_bds_crd=obs_bds_crd)
        else:
            raise ValueError(f"Unsupported positioning_mode: {self.config.positioning_mode}")
        self._warn_unused_settings(ppp_filter)
        self._warn_ar_route(ppp_filter)


        ref = None
        if self.config.sinex_path:
            # SOLUTION/ESTIMATE of the station (MARKER NAME[:4]) valid at the first epoch (bug-fix stage 13d,
            # PPP-031 / SPP-081); a station not in the file: one warning naming it and the file
            ref, missing = _sinex_station_reference(
                self.config.sinex_path, obs_data.meta[0],
                _first_epoch(obs_gps_crd, obs_gal_crd, obs_bds_crd), "PPP", self.config.obs_path)
            if missing:
                _notes.warn_once(self.config, ("sinex_station", missing), missing, UserWarning, stacklevel=3)
        if ref is not None:
            flh = ecef2geodetic(ecef=ref, deg=True)
            ppp_result, ppp_gps, ppp_gal, conv_time = ppp_filter.run_filter(ref=ref.copy(), flh=flh.copy(),
                                                                            reset_every=self.config.reset_every,
                                                                            trace_filter=self.config.trace_filter)
        else:
            ppp_result, ppp_gps, ppp_gal, conv_time = ppp_filter.run_filter(reset_every=self.config.reset_every,
                                                                            trace_filter=self.config.trace_filter)
            _blank_enu_without_reference(ppp_result)



        # contract C1 (bug-fix stage 13b, PPP-033): every filter returns {system letter: residual frame} of the
        # systems it processed; the residuals are routed by key only (no positional mapping)
        processed = [s for s, frame in (("G", obs_gps_crd), ("E", obs_gal_crd), ("C", obs_bds_crd))
                     if isinstance(frame, pd.DataFrame) and not frame.empty]
        residuals = self._residuals_by_system(ppp_filter, ppp_gps, ppp_gal, processed)
        result = PPPResult(solution=ppp_result, residuals_gps=residuals.get('G'), residuals_gal=residuals.get('E'),
                           residuals_bds=residuals.get('C'), convergence=conv_time)
        datum = getattr(self, "_clock_datum", None)
        if datum is not None and isinstance(result.solution, pd.DataFrame):
            # the BeiDou precise clock datum the code biases used (SPP-080)
            result.solution.attrs["precise_clock_datum"] = {"C": datum.as_dict()}
        return result
