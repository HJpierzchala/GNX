"""Configuration and high-level control for GNSS orbit/clock/DCB processing.

This module contains:
- SISConfig: a configuration container for file paths, modes, and processing flags.
- SISController: an orchestrator that runs the end-to-end pipeline (file classification,
  orbit handling, eclipse detection, antenna PCO/PCV application, DCB handling,
  and merging of outputs).

Conventions:
- Time is UTC unless stated otherwise.
- Units follow SI where applicable (meters, seconds), unless specified.
- Array-like inputs follow NumPy broadcasting semantics when relevant.

Notes:
- External resources (SP3, ATX, DCB/OSB) are provided via SISConfig fields.
- Controller methods document behavior and expected outputs without changing any logic.
"""

import re
import warnings
import logging as _logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd

from ..io import GNSSDataProcessor2, read_sp3
from ..time import arange_datetime
from ..tools import DDPreprocessing, DefaultConfig
from ..coordinates import BrdcGenerator, SP3InterpolatorOptimized
from ..session_errors import guarded_session_run
from ..gnss import bds_orbit_type, is_bds_geo, mode_ionosphere_free_coefficients, mode_signals, signal_spec


_logger = _logging.getLogger(__name__)


DEFAULT_SISRE_COEFFICIENTS = {
    "kr": 0.9604,
    "kb": 1.0,
    "kac": 0.019881,
    "krb": -1.960,
}

BDS_SISRE_ALPHA_BETA = {
    "MEO": (0.98, 54.0),
    "IGSO": (0.99, 127.0),
}

BDS_LEGACY_BROADCAST_TGD_SIGNALS = {"B1I", "B2I", "B3I"}
BDS_MODERN_BROADCAST_SIGNALS = {"B1C", "B2a", "B2b"}
BDS_LEGACY_IF_PRECISE_CLOCK_MODE = "B1IB3I"

BDS_CLOCK_CONVENTION_NOTE = (
    "BDS precise clocks in the sample MGEX stacks are tied to a code/IF clock "
    "reference, commonly C2I/C6I for legacy products. Broadcast TGD is a user "
    "correction for a selected signal or IF combination, not a generic DCB/OSB "
    "replacement. SIS clock errors are directly comparable only after both clocks "
    "are expressed in the same datum."
)


@dataclass
class SISConfig:
    """
    Configuration container for signal-in-space orbit and clock comparison.

    Status:
        Active public API. `SISConfig` is the recommended entry point for
        programmatic SIS/SISRE comparisons. It is intentionally lightweight and
        mostly stores paths and switches consumed by `SISController`.

    Purpose:
        The object describes one comparison run: two orbit/clock inputs, the
        GNSS system, signal/frequency mode, optional bias products, time window,
        and modeling switches such as eclipse masking and satellite PCO.

    Supported modes:
        GPS (`system="G"`) and Galileo (`system="E"`) support broadcast-vs-SP3
        and SP3-vs-SP3 style comparisons when the requested signal mode matches
        the available navigation and bias products. BeiDou (`system="C"`) is
        supported for legacy `B1IB3I` MEO/IGSO validation and guarded modern
        BDS experiments. BeiDou GEO is deliberately excluded from the SIS/SISRE
        comparison path until GEO-specific validation is added.

    Main parameters:
        orb_path_0, orb_path_1:
            Input orbit products. RINEX navigation files are treated as
            broadcast ephemerides; `.sp3`/`.sp3.gz` files are treated as precise
            orbit/clock products.
        interval:
            Target processing interval in seconds. The controller converts this
            to minutes for epoch generation and interpolation.
        system:
            GNSS constellation code: `"G"` for GPS, `"E"` for Galileo, `"C"`
            for BeiDou.
        gps_mode, gal_mode, bds_mode:
            Signal or ionosphere-free mode used by broadcast generation and
            group-delay conversion. Recommended BDS validation mode is
            `bds_mode="B1IB3I"`. Single-signal BDS modes such as `"B1I"` and
            modern BDS modes such as `"B1C"`, `"B2a"` or `"B2b"` are reported
            with warning/requires-validation clock-convention status unless the
            clock datum and OSB/DSB stack are explicitly consistent.
        dcb_path_0, dcb_path_1, compare_dcb:
            Optional DCB/OSB/BIA products. When `compare_dcb` is false, GPS and
            Galileo receive zero TGD placeholders, while BeiDou broadcast NAV
            inputs use the broadcast group-delay fields available in the NAV
            data. When `compare_dcb` is true, matching DSB/OSB products are read
            and converted to the internal `TGD` column.
        atx_path, apply_satellite_pco:
            Optional antenna model path and switch for satellite phase-center
            correction of precise SP3 coordinates.
        prev_sp3_0, next_sp3_0, prev_sp3, next_sp3:
            Adjacent-day SP3 files used to stabilize Chebyshev interpolation at
            day boundaries.
        clock_bias, clock_bias_function:
            Controls removal of a common per-epoch clock datum before SISRE
            computation. The function is passed to pandas groupby transform,
            typically `"mean"` or another reducer understood by pandas.
        apply_eclipse, extend_eclipse, extension_time:
            Controls satellite eclipse flagging and optional post-eclipse mask
            extension. Units for `extension_time` are minutes.
        tlim:
            Optional `[start_datetime, end_datetime]` processing window.

    Warnings:
        A large BDS clock/SIS component can indicate a clock-datum mismatch
        between broadcast corrections and precise clock products rather than a
        physical orbit/clock error. Inspect `clock_convention_status` and
        `clock_convention_note` on the output before interpreting BDS results.
    """

    def __init__(self, orb_path_0, orb_path_1, interval, prev_sp3=None, next_sp3=None, dcb_path_0=None, dcb_path_1=None, gps_mode=None ,gal_mode=None, bds_mode=None, atx_path=None, system='G',
                 prev_sp3_0=None, next_sp3_0=None, tlim=None,compare_dcb=False, clock_bias=True, clock_bias_function='mean',apply_eclipse=True,apply_satellite_pco=True,extend_eclipse = True, extension_time=30):
        """
                Initialize SIS configuration.

                The constructor wires runtime options, file paths, and modeling flags
                (e.g., orbit/DCB/ATX sources, interval/time limits, eclipse/PCO switches).

                Notes:
                    - Required fields and types are validated either here or within SISController.
                    - Paths should point to existing files in supported formats.

                Raises:
                    ValueError: If mutually incompatible options are provided.
                    FileNotFoundError: If a required resource is missing (when validated here).
                """

        self.orb_path_0:Union[str,Path] = orb_path_0
        self.orb_path_1: Union[str, Path] = orb_path_1

        self.compare_dcb = compare_dcb
        self.dcb_path_0: Optional[Union[str, Path, None]] = dcb_path_0
        self.dcb_path_1: Optional[Union[str, Path, None]] = dcb_path_1

        self.atx_path: Union[str, Path] =atx_path
        self.system: str = system
        self.interval: Union[float, int] = interval

        self.gps_mode: Union[str, None] = gps_mode
        self.gal_mode:Union[str, None] = gal_mode
        self.bds_mode: Union[str, None] = bds_mode or "B1IB3I"

        self.apply_eclipse: bool = apply_eclipse
        self.apply_satellite_pco:bool=apply_satellite_pco

        self.prev_sp3_0: Optional[Union[str, Path, None]] = prev_sp3_0
        self.next_sp3_0: Optional[Union[str, Path, None]] = next_sp3_0
        self.prev_sp3: Optional[Union[str, Path, None]] = prev_sp3
        self.next_sp3: Optional[Union[str, Path, None]] = next_sp3

        self.clock_bias:bool=clock_bias
        self.clock_bias_function:str=clock_bias_function

        if self.apply_eclipse:
            self.extend_eclipse:bool = extend_eclipse
            self.extension_time : Union[int, float] = extension_time
        self.tlim: Union[List[datetime, datetime], None] = tlim


class SISController:
    """
    High-level controller for SIS/SISRE orbit and clock comparisons.

    Status:
        Active public API. The class is the main orchestration layer for the
        `gnx_py.orbits` package. It also contains guarded BDS helpers that
        require caution because they encode clock datum, TGD, OSB and DSB
        assumptions.

    Pipeline:
        1. Classify both orbit inputs as RINEX navigation, SP3, observation or
           unknown files.
        2. Generate broadcast coordinates/clocks or interpolate precise SP3
           products onto the requested epoch grid.
        3. Optionally apply satellite PCO corrections to precise products.
        4. Merge broadcast TGD or external DCB/OSB/DSB products into a common
           `TGD` representation in meters.
        5. Align both products to common satellite/epoch pairs.
        6. Compute coordinate differences, clock differences, RAC components
           and SISRE/SIS-style metrics.
        7. Annotate BDS outputs with clock-convention status when relevant.

    Outputs:
        `run()` returns a pandas DataFrame indexed by satellite and epoch. Core
        columns include position differences (`dx`, `dy`, `dz`), RAC components
        (`dR`, `dA`, `dC`), clock error in meters (`dt`), group delay difference
        (`dTGD`) and SISRE variants (`sisre`, `sisre_orb`, `sisre_notgd`).

    Limitations:
        BeiDou GEO is unsupported in this path. Legacy BDS `B1IB3I` MEO/IGSO is
        the validated broadcast-vs-precise clock comparison mode. Modern BDS
        modes and single-signal legacy modes are exposed for guarded validation
        and should be interpreted through the clock-convention metadata.
    """

    def __init__(self, config: SISConfig):
        """
                Initialize controller state and bind configuration.

                Notes:
                    - Binds constants (e.g., speed of light) and establishes internal state.
                    - Does not perform I/O; data loading happens in dedicated methods.
                """

        self.config=config
        self.system = self.config.system
        self.clight = 299792458
        self.output_cols = ['dR','dA','dC','dt','dt_mean','dTGD','dTGD_mean','sisre','sisre_orb','sisre_notgd', 'dx', 'dy',
       'dz']


    def classify_file(self, path: str | Path) -> str:
        """
        Classify an input file and determine its handling route.

        Returns:
            dict: A lightweight descriptor with keys such as:
                - "type": e.g., "broadcast", "precise_orbit", "dcb", "atx".
                - "sys": GNSS sys code, if determinable.
                - Additional fields required by downstream steps.

        Raises:
            ValueError: If the file type is unsupported or cannot be classified.
        """

        # -- stałe, kompilujemy raz -----------------------------------------------
        _SP3_RE = re.compile(r'\.sp3(?:\.gz)?$', re.IGNORECASE)

        # RINEX 3:  …_<CONST><TYPE>.(rnx|crx)[.gz]
        # np. …_MN.rnx      (mixed-constellation navigation)
        #     …_GO.crx.gz   (GPS observations, Hatanaka + gzip)
        _RINEX3_RE = re.compile(r'_(?P<const>[A-Z])(?P<dtype>[ONM])\.(?:rnx|crx)(?:\.gz)?$', re.IGNORECASE)

        # RINEX 2 nawigacyjny: *.YYn[.gz]  (np. brdc2400.24n.gz)
        _RINEX2_NAV_RE = re.compile(r'\.\d{2}n(?:\.gz)?$', re.IGNORECASE)

        # RINEX 2 obserwacyjny: *.rnx|*.crx[.gz]
        _RINEX2_OBS_RE = re.compile(r'\.(rnx|crx)(?:\.gz)?$', re.IGNORECASE)

        name = Path(path).name  # osobno, bez katalogu; wielkość liter zachowujemy do RINEX3
        low = name.lower()  # wersja małoliterowa – szybciej porównywać rozszerzenia

        # -- 1. SP3 ----------------------------------------------------------------
        if _SP3_RE.search(low):
            return "SP3"

        # -- 2. RINEX 3 ------------------------------------------------------------
        m3 = _RINEX3_RE.search(name)  # tu potrzebujemy oryginalnego (wielkie litery)
        if m3:
            dtype = m3.group('dtype').upper()  # O/N/M
            match dtype:
                case 'N':
                    return "RINEX_NAV"
                case 'O':
                    return "RINEX_OBS"
                case 'M':
                    return "RINEX_MET"
            # ewentualnie inne litery w przyszłości
            return "RINEX_UNKNOWN"

        # -- 3. RINEX 2 ------------------------------------------------------------
        if _RINEX2_NAV_RE.search(low):
            return "RINEX_NAV"
        if _RINEX2_OBS_RE.search(low):
            return "RINEX_OBS"

        # -- 4. Nic nie pasuje -----------------------------------------------------
        return "UNKNOWN"

    def process_broadcast_orbit(self, path: str | Path):
        """
        Process broadcast ephemerides and align them to the target epoch grid.

        Status:
            Active. Supports GPS, Galileo and BeiDou broadcast branches through
            `BrdcGenerator` and the configured signal mode.

        Returns:
            pandas.DataFrame: Per-satellite broadcast coordinates, clocks and
            available navigation delay fields aligned to the controller epochs.

        Raises:
            ValueError: If the configured system is unsupported or no broadcast
            data are available for that system.
        """

        prc =   GNSSDataProcessor2(nav_path=path, sys=self.config.system)
        if self.config.tlim is not None:
            nav = prc.load_broadcast_orbit(tlim=self.config.tlim)
        else:
            nav = prc.load_broadcast_orbit()
        if self.system =='G':
            orbit = nav.gps_orb
            mode = self.config.gps_mode
            tol='4H'
        elif self.system =='E':
            orbit = nav.gal_orb
            mode = self.config.gal_mode
            tol = '45T'
        elif self.system == 'C':
            orbit = nav.bds_orb
            mode = self.config.bds_mode
            tol = '2H'
        else:
            raise ValueError(f"Unsupported sys: {self.system}")
        if orbit is None or orbit.empty:
            raise ValueError(f"No broadcast orbit data found for system {self.system}.")

        broadcast_interpolator = BrdcGenerator(system=self.system,
                                               interval=self.config.interval,
                                               mode=mode,
                                               nav=orbit,tolerance=tol)
        crd = broadcast_interpolator.generate()
        crd[['x_apc','y_apc','z_apc']] = crd[['x','y','z']]
        return crd

    def process_precise_orbit(self, path: str | Path, prev_path: str|Path, next_path:str|Path):
        """
        Process precise SP3 orbits and align them to the target epoch grid.

        Status:
            Active. For BDS, GEO satellites are filtered out before
            interpolation because the current SISRE path is validated for
            MEO/IGSO only.

        Args:
            path: Nominal-day SP3 file.
            prev_path: Previous-day SP3 file used for boundary interpolation.
            next_path: Next-day SP3 file used for boundary interpolation.

        Returns:
            pandas.DataFrame: Interpolated satellite positions and clocks,
            indexed by satellite and time.

        Notes:
            The method uses Chebyshev interpolation and optionally adjacent-day
            SP3 products to reduce edge effects.
        """

        sp3 =   read_sp3(path=path, sys=self.system)
        sp3 = self._filter_precise_bds_geo(sp3, path)
        if sp3.empty:
            raise ValueError(f"No precise SP3 orbit data found for system {self.system}: {path}")
        if self.config.tlim is not None:

            epochs =    arange_datetime(start_datetime=self.config.tlim[0], end_datetime=self.config.tlim[1],
                                                  step_minutes=self.config.interval/60)
        else:
            sp3_epochs = sorted(sp3.index.get_level_values('time').tolist())
            epochs =    arange_datetime(start_datetime=sp3_epochs[0], end_datetime=sp3_epochs[-1],
                                                  step_minutes=self.config.interval/60)[:-1]

        precise_interp = SP3InterpolatorOptimized(sp3_dataframe=sp3)
        if prev_path is not None:
            prev_sp3 =   read_sp3(path=self.config.prev_sp3, sys=self.system)
        else:
            prev_sp3 = None
        if next_path is not None:
            next_sp3 =   read_sp3(path=self.config.next_sp3, sys=self.system)
        else:
            next_sp3 = None
        precise_interp.include_adjacent_data(prev_sp3_df=prev_sp3,next_sp3_df=next_sp3)
        interpolated = precise_interp.run(epochs=epochs, method='chebyshev')
        interpolated.index.names = ['time', 'sv']
        interpolated = interpolated.swaplevel()
        return interpolated

    def _filter_precise_bds_geo(self, sp3: pd.DataFrame, path: str | Path) -> pd.DataFrame:
        """
        Remove BeiDou GEO satellites from precise-orbit SIS processing.

        Status:
            Active guard rail / unsupported-case handling.

        The current SISRE coefficient and validation stack covers BDS MEO and
        IGSO satellites. GEO satellites are therefore skipped with a runtime
        warning instead of being allowed to enter RAC/SISRE calculations with
        coefficients that have not been validated for GEO geometry.
        """

        if self.system != "C" or sp3.empty:
            return sp3
        sv_index = sp3.index.get_level_values("sv")
        geo_sats = sorted({sv for sv in sv_index if is_bds_geo(sv)})
        if not geo_sats:
            return sp3
        warnings.warn(
            "BeiDou SIS/SISRE precise-orbit processing currently supports BDS MEO/IGSO only; "
            f"skipping GEO satellites from {path}: {', '.join(geo_sats)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        out = sp3.loc[~sv_index.map(is_bds_geo)].copy()
        return out

    def set_common_epochs(self, orbit_0:pd.DataFrame, orbit_1:pd.DataFrame):
        """
                Align two time series to a common epoch grid.

                Returns:
                    tuple(pd.DataFrame, pd.DataFrame): Both inputs aligned to the same epoch grid,
                    following the controller's alignment rules.

                Notes:
                    - The specific return structure follows the internal data model in use.
                """

        cmn = orbit_0.index.intersection(orbit_1.index)
        return orbit_0.loc[cmn].copy(), orbit_1.loc[cmn].copy()

    def _ensure_apc_columns(self, orbit: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that orbit coordinates expose antenna-phase-center columns.

        Status:
            Active compatibility helper.

        Broadcast products generally carry geometric coordinates as `x`, `y`,
        `z`; precise products may later be corrected to `x_apc`, `y_apc`,
        `z_apc`. This helper keeps both branches compatible without changing
        coordinate values when APC-specific columns are absent.
        """

        out = orbit.copy()
        for raw_col, apc_col in (("x", "x_apc"), ("y", "y_apc"), ("z", "z_apc")):
            if apc_col not in out.columns:
                if raw_col not in out.columns:
                    raise KeyError(f"Orbit table is missing both {apc_col!r} and {raw_col!r}.")
                out[apc_col] = out[raw_col]
        return out

    def _bds_clock_convention_policy(self, orbit_0_type: str, orbit_1_type: str) -> dict[str, object]:
        """
        Classify BDS clock-datum comparability for the current run.

        Status:
            Active, requires caution.

        Returns:
            dict: Metadata describing whether the selected BDS mode is directly
            comparable, warning-only, requires validation, or unsupported.

        Notes:
            Broadcast BDS TGD/ISC fields are user corrections for a selected
            signal or ionosphere-free combination. Precise SP3 clocks can be
            tied to a different code or IF datum. This method does not alter
            the numerical result; it records whether the result can be
            interpreted as a direct SIS clock error.
        """

        mode = self.config.bds_mode
        policy = {
            "system": self.system,
            "mode": mode,
            "status": "ok",
            "directly_comparable": True,
            "note": "",
        }
        if self.system != "C":
            return policy

        try:
            signals = mode_signals(mode)
        except ValueError as exc:
            return {
                **policy,
                "status": "unsupported",
                "directly_comparable": False,
                "note": str(exc),
            }

        compares_broadcast_to_precise = {"RINEX_NAV", "SP3"}.issubset({orbit_0_type, orbit_1_type})
        if not compares_broadcast_to_precise:
            return {
                **policy,
                "note": "No broadcast-vs-precise BDS clock datum comparison is performed in this run.",
            }

        has_external_bias = bool(
            self.config.compare_dcb and (self.config.dcb_path_0 is not None or self.config.dcb_path_1 is not None)
        )
        signal_set = set(signals)

        if signal_set.intersection(BDS_MODERN_BROADCAST_SIGNALS):
            return {
                **policy,
                "status": "requires_validation",
                "directly_comparable": False,
                "note": (
                    f"BeiDou mode {mode} contains modern signals {sorted(signal_set.intersection(BDS_MODERN_BROADCAST_SIGNALS))}. "
                    "Broadcast-vs-precise SIS clock comparison requires RINEX 4 CNAV ISC/TGD fields and a clock-reference "
                    "consistent OSB/DSB stack for the selected observables. "
                    + BDS_CLOCK_CONVENTION_NOTE
                ),
            }

        if mode == BDS_LEGACY_IF_PRECISE_CLOCK_MODE:
            return {
                **policy,
                "note": (
                    "BeiDou B1I/B3I ionosphere-free mode uses the legacy IF broadcast datum correction "
                    "a*TGD1 and is the validated legacy mode for precise C2I/C6I-referenced clocks."
                ),
            }

        if len(signals) == 1:
            if has_external_bias:
                return {
                    **policy,
                    "status": "warning",
                    "directly_comparable": False,
                    "note": (
                        f"BeiDou single-signal mode {mode} uses a signal-specific broadcast group delay, while the "
                        "precise clock may be tied to an IF reference such as C2I/C6I. The external bias product is "
                        "used where possible, but clock-stack consistency must be verified before interpreting the "
                        "SIS clock component as a true signal clock error. "
                        + BDS_CLOCK_CONVENTION_NOTE
                    ),
                }
            return {
                **policy,
                "status": "requires_validation",
                "directly_comparable": False,
                "note": (
                    f"BeiDou single-signal mode {mode} is not directly comparable with a precise IF clock without "
                    "an explicit clock-datum conversion from a matching OSB/DSB product. Use B1IB3I for legacy "
                    "BDS SIS clock validation, or enable compare_dcb with a clock-reference consistent product. "
                    + BDS_CLOCK_CONVENTION_NOTE
                ),
            }

        return {
            **policy,
            "status": "requires_validation",
            "directly_comparable": False,
            "note": (
                f"BeiDou mode {mode} is not the validated legacy precise-clock comparison mode. Its broadcast "
                "TGD combination may be valid for user corrections, but SIS clock comparison needs a documented "
                "precise-clock datum conversion for the selected observables. "
                + BDS_CLOCK_CONVENTION_NOTE
            ),
        }

    def _annotate_clock_convention(
        self,
        compared: pd.DataFrame,
        orbit_0_type: str,
        orbit_1_type: str,
    ) -> pd.DataFrame:
        """
        Attach BDS clock-convention metadata to a comparison result.

        Status:
            Active output annotation.

        The method writes both DataFrame attributes and explicit columns so that
        downstream code, CSV exports and tests can see the same status. It does
        not modify orbit, clock, TGD or SISRE values.
        """

        policy = self._bds_clock_convention_policy(orbit_0_type, orbit_1_type)
        if self.system != "C":
            return compared

        out = compared.copy()
        for key, value in policy.items():
            out.attrs[f"clock_convention_{key}"] = value
        out["clock_convention_status"] = policy["status"]
        out["clock_convention_mode"] = policy["mode"]
        out["clock_convention_directly_comparable"] = bool(policy["directly_comparable"])
        out["clock_convention_note"] = policy["note"]

        if policy["status"] != "ok":
            warnings.warn(
                f"BeiDou SIS clock convention status {policy['status']} for mode {policy['mode']}: {policy['note']}",
                RuntimeWarning,
                stacklevel=2,
            )
        return out

    def eclipse_periods(self, orbit:pd.DataFrame):
        """
                Identify satellite eclipse periods and optionally extend masks around ingress/egress.

                Returns:
                    pd.DataFrame: df with Per-satellite, per-epoch eclipse flags/masks usable
                    for filtering or modeling.

                Notes:
                    - The exact umbra/penumbra criteria follow project definitions.
                """

        # --- stałe ---------------------------------------------------------
        A_E = 6_378_137.0  # równikowy promień Ziemi [m]

        # --- wektory -------------------------------------------------------
        r_sat = orbit[['x', 'y', 'z']].to_numpy()  # (N,3)
        r_sun = orbit[['xs', 'ys', 'zs']].to_numpy()  # (N,3)

        norm_sat = np.linalg.norm(r_sat, axis=1)
        norm_sun = np.linalg.norm(r_sun, axis=1)

        cos_phi = (r_sat * r_sun).sum(1) / (norm_sat * norm_sun)

        # --- warunki Navipedia --------------------------------------------
        cond1 = cos_phi < 0  # (za Ziemią)
        cond2 = norm_sat * np.sqrt(1 - cos_phi ** 2) < A_E  # (w cylindrze)

        orbit['eclipse_raw'] = cond1 & cond2  # flaga "czysty" cień

        def extend_post_eclipse(group, minutes=30):
            """Extend a per-satellite eclipse mask for a fixed number of minutes."""

            t = group.index.get_level_values('time')
            eclipse = group['eclipse_raw'].to_numpy()
            shadow = eclipse.copy()

            last_exit = None
            for i, is_ecl in enumerate(eclipse):
                if is_ecl:
                    last_exit = t[i]  # cały czas w cieniu
                elif last_exit is not None and (t[i] - last_exit) <= pd.Timedelta(minutes=minutes):
                    shadow[i] = True  # +30 min po wyjściu
                else:
                    last_exit = None

            group['eclipse'] = shadow
            return group
        if self.config.extend_eclipse:
            orbit = orbit.groupby(level='sv', group_keys=False).apply(extend_post_eclipse,minutes=self.config.extension_time)
        return orbit

    def apply_satellite_pco(self, precise_orbit:pd.DataFrame):
        """
                Apply satellite antenna PCO/PCV corrections to orbit coordinates.

                Returns:
                    pd.DataFrame: Orbit data with PCO/PCV corrections applied per satellite and epoch.

                Raises:
                    ValueError: If required antenna model entries are missing for a satellite.
                """

        # sat_pco =   GNSSDataProcessor2(atx_path=self.config.atx_pat).satellite_pco(
        #     sats=precise_orbit.index.get_level_values('sv').unique().tolist())
        sat_pco =   GNSSDataProcessor2(atx_path=self.config.atx_path).read_pco_antex(system_code=self.system,date=precise_orbit.index.get_level_values('time').unique().tolist()[0])
        cfg =  DefaultConfig(gps_freq=self.config.gps_mode, gal_freq=self.config.gal_mode, bds_freq=self.config.bds_mode)
        processor = DDPreprocessing(df=precise_orbit, flh=None, xyz=np.array([1e7, 1e7, 1e7]),
                                        phase_shift_dict=None, sat_pco=sat_pco, rec_pco=None, antenna_h=None,
                                        system=self.system, config=cfg)
        precise_orbit[['xe', 'ye', 'ze']] = precise_orbit[['x', 'y', 'z']]
        processor.sat_fixed_system()
        precise_apc = processor.df.copy().swaplevel()
        return precise_apc


    def compare_orbits(self, orbit_0:pd.DataFrame, orbit_1:pd.DataFrame, kr, kb, kac, krb, clock_bias=True, clock_bias_function='mean'):
        """
        Compare two orbit/clock solutions on shared satellite/epoch pairs.

        Status:
            Active numerical core. Do not change the RAC/SISRE equations without
            cross-system regression tests.

        Inputs:
            orbit_0, orbit_1:
                DataFrames indexed by `sv,time`, containing APC coordinates
                (`x_apc`, `y_apc`, `z_apc`), raw coordinates (`x`, `y`, `z`),
                velocity (`vx`, `vy`, `vz`), clock (`clk`) and group delay
                (`TGD`). `orbit_1` or `orbit_0` should also carry Sun
                coordinates (`xs`, `ys`, `zs`) when eclipse flags are requested.
            kr, kb, kac, krb:
                SISRE weighting coefficients. For BDS defaults, MEO/IGSO
                coefficients are attached per satellite.
            clock_bias:
                If true, remove a common epoch clock and TGD datum before SISRE
                calculation.

        Outputs:
            pandas.DataFrame: Original comparison table plus `dx`, `dy`, `dz`,
            clock error `dt` in meters, group-delay error `dTGD`, RAC components
            (`dR`, `dA`, `dC`), `sisre`, `sisre_orb` and `sisre_notgd`.

        Units:
            Positions and RAC components are meters. Clock differences are
            converted from seconds to meters with the speed of light.
        """
        _logger.debug("SIS compare orbit_0 columns=%s", list(orbit_0.columns))
        _logger.debug("SIS compare orbit_1 columns=%s", list(orbit_1.columns))
        orbit_0[['dx', 'dy', 'dz', 'dt','dTGD']] = (
                    orbit_0[['x_apc', 'y_apc', 'z_apc', 'clk','TGD']] - orbit_1[['x_apc', 'y_apc', 'z_apc', 'clk','TGD']]).values
        orbit_0['clk_0'] = orbit_0['clk']
        orbit_0['clk_1'] = orbit_1['clk']
        compare = orbit_0.copy()
        compare['dt']*=self.clight
        compare = self._attach_sisre_coefficients(compare, kr=kr, kb=kb, kac=kac, krb=krb)
        if all(col in orbit_1.columns for col in ['xs', 'ys', 'zs']):
            compare[['xs', 'ys', 'zs']] = orbit_1[['xs', 'ys', 'zs']].copy()
        elif all(col in orbit_0.columns for col in ['xs', 'ys', 'zs']):
            compare[['xs', 'ys', 'zs']] = orbit_0[['xs', 'ys', 'zs']].copy()

        for ind, group in compare.groupby('sv'):
            r = group[['x_apc', 'y_apc', 'z_apc']].to_numpy()
            v = group[['vx', 'vy', 'vz']].to_numpy()

            # e1 - jednostkowy wektor wzdłuż r
            e1 = np.apply_along_axis(lambda row: -row / np.linalg.norm(row), axis=1, arr=r)

            # e3 - jednostkowy wektor poprzeczny do r i v
            cross_rv = np.cross(r, v)
            e3 = -np.apply_along_axis(lambda row: row / np.linalg.norm(row), axis=1, arr=cross_rv)

            # e2 - wektor prostopadły do e1 i e3
            e2 = np.cross(e3, e1)

            # Przemieszczenia
            dxyz = group[['dx', 'dy', 'dz']].to_numpy()

            # Składowe przemieszczeń
            dR = np.sum(e1 * dxyz, axis=1)
            dA = np.sum(e2 * dxyz, axis=1)
            dC = np.sum(e3 * dxyz, axis=1)

            # Przypisanie wyników do DataFrame
            compare.loc[group.index, 'dR'] = dR
            compare.loc[group.index, 'dA'] = dA
            compare.loc[group.index, 'dC'] = dC
        if clock_bias:
            grp = compare.groupby('time')['dt']
            compare['dt_mean'] = grp.transform(clock_bias_function)

            grp = compare.groupby('time')['dTGD']
            compare['dTGD_mean'] = grp.transform(clock_bias_function)
        else:
            compare['dt_mean'] = 0.0
            compare['dTGD_mean'] = 0.0

        for ind, group in compare.groupby('time'):
            dR = group['dR']
            dA = group['dA']
            dC = group['dC']
            dt = (group['dt'] - group['dt_mean'])  # *clight
            dtgd = (group['dTGD'] - group['dTGD_mean'])
            ac = np.sqrt(dA ** 2 + dC ** 2)
            kr_i = group['_sisre_kr']
            kb_i = group['_sisre_kb']
            kac_i = group['_sisre_kac']
            krb_i = group['_sisre_krb']
            ure_orbit = np.sqrt(
                kr_i * dR ** 2 + kac_i * ac)
            ure_sv_ga = np.sqrt(
                kr_i * dR ** 2 + kb_i * (dt-dtgd) ** 2 + kac_i * ac + krb_i * dR * (dt-dtgd))
            ure_sv_ga_notgd = np.sqrt(
                kr_i * dR ** 2 + kb_i * dt ** 2 + kac_i * ac + krb_i * dR * dt)
            compare.loc[group.index, 'sisre'] = ure_sv_ga
            compare.loc[group.index, 'sisre_orb'] = ure_orbit
            compare.loc[group.index, 'sisre_notgd'] = ure_sv_ga_notgd
        return compare

    def _attach_sisre_coefficients(self, compare: pd.DataFrame, kr, kb, kac, krb) -> pd.DataFrame:
        """
        Attach SISRE weighting coefficients to each comparison row.

        Status:
            Active. For GPS/Galileo and custom coefficients, the scalar inputs
            are copied to every row. For default BeiDou coefficients, validated
            MEO/IGSO alpha/beta values are converted into row-level weights.

        Raises:
            ValueError: If a BDS GEO satellite reaches this method, because GEO
            SISRE coefficients are not validated in this implementation.
        """

        compare['_sisre_kr'] = kr
        compare['_sisre_kb'] = kb
        compare['_sisre_kac'] = kac
        compare['_sisre_krb'] = krb
        if self.system != "C":
            return compare

        uses_default = (
            kr == DEFAULT_SISRE_COEFFICIENTS["kr"]
            and kb == DEFAULT_SISRE_COEFFICIENTS["kb"]
            and kac == DEFAULT_SISRE_COEFFICIENTS["kac"]
            and krb == DEFAULT_SISRE_COEFFICIENTS["krb"]
        )
        if not uses_default:
            return compare

        for sv in compare.index.get_level_values("sv").unique():
            orbit_type = bds_orbit_type(sv)
            if orbit_type == "GEO":
                raise ValueError(
                    f"BeiDou GEO satellite {sv} is not supported by the SIS/SISRE orbit comparison path."
                )
            alpha, beta = BDS_SISRE_ALPHA_BETA[orbit_type]
            mask = compare.index.get_level_values("sv") == sv
            compare.loc[mask, '_sisre_kr'] = alpha ** 2
            compare.loc[mask, '_sisre_kb'] = 1.0
            compare.loc[mask, '_sisre_kac'] = 1.0 / beta
            compare.loc[mask, '_sisre_krb'] = -2.0 * alpha
        return compare

        # --- GŁÓWNA METODA PUBLICZNA ---
    def process_dcb_and_merge(self, orbit_0, orbit_1, orbit_0_type, orbit_1_type):
        """
        Ingest DCB/OSB resources, convert to appropriate group delays if needed,
        and merge them with the orbit/measurement data.

        Status:
            Active bias/clock-datum policy. Requires caution for BDS because the
            selected signal mode must match the broadcast fields and any
            external OSB/DSB product.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame): Orbit/measurement table with per-satellite delay fields merged.

        Raises:
            RuntimeError: If DCB parsing or conversion fails.
        """

        if not self.config.compare_dcb:
            if self.system == 'C':
                orbit_0 = self._merge_bds_broadcast_tgd_without_external_dcb(orbit_0, orbit_0_type)
                orbit_1 = self._merge_bds_broadcast_tgd_without_external_dcb(orbit_1, orbit_1_type)
                return orbit_0, orbit_1
            orbit_0['TGD'] = 0.0
            orbit_1['TGD'] = 0.0
            return orbit_0, orbit_1
        if self.config.dcb_path_0 is not None and self.config.dcb_path_1 is not None:
            if self.system == 'G':
                dcb_0, type_dcb_0 = self._get_gps_dcb(orbit_0, orbit_0_type, self.config.dcb_path_0)
                dcb_1, type_dcb_1 = self._get_gps_dcb(orbit_1, orbit_1_type, self.config.dcb_path_1)
                dcb_0 = self._convert_gps_dcb_to_tgd(dcb_0, type_dcb_0)
                dcb_1 = self._convert_gps_dcb_to_tgd(dcb_1, type_dcb_1)
                orbit_0 = self._merge_tgd_with_orbit(orbit_0, dcb_0)
                orbit_1 = self._merge_tgd_with_orbit(orbit_1, dcb_1)
            elif self.system == 'E':
                gal_field = self._get_gal_field()
                dcb_0, type_dcb_0 = self._get_gal_dcb(orbit_0, orbit_0_type, self.config.dcb_path_0, gal_field)
                dcb_1, type_dcb_1 = self._get_gal_dcb(orbit_1, orbit_1_type, self.config.dcb_path_1, gal_field)
                dcb_0_out = self._convert_gal_dcb_to_tgd(dcb_0, type_dcb_0)
                dcb_1_out = self._convert_gal_dcb_to_tgd(dcb_1, type_dcb_1)
                orbit_0 = self._merge_gal_tgd_with_orbit(orbit_0, dcb_0_out, type_dcb_0)
                orbit_1 = self._merge_gal_tgd_with_orbit(orbit_1, dcb_1_out, type_dcb_1)
            elif self.system == 'C':
                bds_field = self._get_bds_field()
                dcb_0, type_dcb_0 = self._get_bds_dcb(orbit_0, orbit_0_type, self.config.dcb_path_0, bds_field)
                dcb_1, type_dcb_1 = self._get_bds_dcb(orbit_1, orbit_1_type, self.config.dcb_path_1, bds_field)
                dcb_0_out = self._convert_bds_dcb_to_tgd(dcb_0, type_dcb_0)
                dcb_1_out = self._convert_bds_dcb_to_tgd(dcb_1, type_dcb_1)
                orbit_0 = self._merge_gal_tgd_with_orbit(orbit_0, dcb_0_out, type_dcb_0)
                _logger.debug("SIS BDS orbit_0 columns after TGD merge=%s", list(orbit_0.columns))
                orbit_1 = self._merge_gal_tgd_with_orbit(orbit_1, dcb_1_out, type_dcb_1)
                _logger.debug("SIS BDS orbit_1 columns after TGD merge=%s", list(orbit_1.columns))
            else:
                raise ValueError(f"Unsupported sys: {self.system}")
        return orbit_0, orbit_1

    # --- POMOCNICZE METODY DLA GPS ---
    def _get_gps_dcb(self, orbit, orbit_type, dcb_path):
        """
        Load or resolve GPS DCB/OSB resource from the provided source.

        Returns:
            Any: Parsed DCB/OSB structure suitable for conversion/merging.
        """

        if dcb_path is not None and orbit_type != 'RINEX_NAV':
            return   GNSSDataProcessor2().read_bia(path=dcb_path)
        elif orbit_type == 'RINEX_NAV':
            dcb, = orbit.pop('TGD').to_frame() * self.clight,
            return dcb, 'brdc'
        else:
            return None, None

    def _convert_gps_dcb_to_tgd(self, dcb, type_dcb):
        """
        Convert GPS DCB/OSB data to per-satellite group delay (e.g., TGD).

        Returns:
            table-like or dict: Per-satellite delay values aligned with the internal data model.
        """

        k2 = -1.1545
        if type_dcb == 'DSB':
            dcb = k2 * dcb[['BIAS_C1W_C2W']] * 1e-09 * self.clight
            dcb = dcb.rename(columns={'BIAS_C1W_C2W': 'TGD'})
        elif type_dcb == 'OSB':
            dcb = k2 * (dcb[['OSB_C1W']] - dcb[['OSB_C2W']].values) * 1e-09 * self.clight
            dcb = dcb.rename(columns={'OSB_C1W': 'TGD'})
        # brdc or None -> return as is
        return dcb

    # --- POMOCNICZE METODY DLA GALILEO ---
    def _get_gal_field(self):
        """
        Resolve a Galileo-specific configuration or data field.

        Returns:
            Any: Field value according to controller/config state.
        """

        gal_mode_map = {
            'E1': 'BGDe5b',
            'E5b': 'BGDe5b',
            'E1E5b': 'BGDe5b',
            'E5a': 'BGDe5a',
            'E1E5a': 'BGDe5a'
        }
        gal_field = gal_mode_map.get(self.config.gal_mode)
        if gal_field is None:
            raise ValueError(f"Unsupported gal_mode: {self.config.gal_mode}")
        return gal_field

    def _get_gal_dcb(self, orbit, orbit_type, dcb_path, gal_field):
        """
        Load or resolve Galileo DCB/OSB resource from the provided source.

        Returns:
            Any: Parsed Galileo DCB/OSB structure.
        """

        if dcb_path is not None and orbit_type != 'RINEX_NAV':
            return   GNSSDataProcessor2().read_bia(path=dcb_path)
        elif orbit_type == 'RINEX_NAV':
            dcb = orbit.pop(gal_field).to_frame() * self.clight
            dcb = dcb.rename(columns={gal_field: 'TGD'})
            return dcb, 'brdc'
        else:
            return None, None

    def _convert_gal_dcb_to_tgd(self, dcb, type_dcb):
        """
        Convert Galileo DCB/OSB data to per-satellite group delay.

        Returns:
            table-like or dict: Per-satellite delay values aligned with the internal data model.
        """

        if self.config.gal_mode in ['E1', 'E5b']:
            k2 = -2.26
            col = '7Q'
        elif self.config.gal_mode in ['E1E5a', 'E5a']:
            k2 = -1.26
            col = '5Q'
        else:
            raise ValueError(f"Nieobsługiwany gal_mode: {self.config.gal_mode}")

        dsb_col = f'BIAS_C1C_C{col}'
        osb_c1c = f'OSB_C1C'
        osb_cxq = f'OSB_C{col}'

        if type_dcb == 'DSB' and dcb is not None and dsb_col in dcb.columns:
            out = k2 * dcb[[dsb_col]] * 1e-9 * self.clight
            out = out.rename(columns={dsb_col: 'TGD'})
            return out
        elif type_dcb == 'OSB' and dcb is not None and osb_c1c in dcb.columns and osb_cxq in dcb.columns:
            out = k2 * (dcb[[osb_c1c]] - dcb[[osb_cxq]].values) * 1e-9 * self.clight
            out = out.rename(columns={osb_c1c: 'TGD'})
            return out
        elif type_dcb == 'brdc':
            return dcb  # BRDC, już TGD
        else:
            return None

    # --- POMOCNICZE METODY DLA BEIDOU ---
    def _get_bds_field(self):
        """
        Resolve the legacy BDS broadcast TGD field for the configured mode.

        Status:
            Active BDS compatibility helper.

        Modern BDS CNAV modes do not map to the legacy `TGD1`/`TGD2` fields, so
        this returns `None` for modern signals and lets the guarded RINEX 4 /
        OSB/DSB path handle them.
        """

        signals = mode_signals(self.config.bds_mode)
        if any(signal in BDS_MODERN_BROADCAST_SIGNALS for signal in signals):
            return None
        if signals == ("B3I",):
            return None
        if "B3I" in signals and "B1I" in signals:
            return "TGD1"
        if "B3I" in signals and "B2I" in signals:
            return "TGD2"
        if "B1I" in signals:
            return "TGD1"
        if "B2I" in signals:
            return "TGD2"
        raise ValueError(f"Unsupported bds_mode for broadcast group delay: {self.config.bds_mode}")

    def _bds_signal_tgd_seconds(self, orbit: pd.DataFrame, signal: str) -> pd.Series:
        """
        Read a signal-specific BDS broadcast group delay in seconds.

        Status:
            Active, requires validation for modern BDS.

        `B3I` is the legacy reference signal in the B1I/B3I datum and therefore
        contributes zero delay here. Modern signals require RINEX 4 CNAV
        signal-specific fields; missing or empty fields raise immediately.
        """

        zero = pd.Series(0.0, index=orbit.index)
        if signal == "B3I":
            return zero
        if signal == "B1I":
            field = "TGD1"
        elif signal == "B2I":
            field = "TGD2"
        elif signal == "B1C":
            field = "TGD_B1Cp"
        elif signal == "B2a":
            field = "TGD_B2ap"
        elif signal == "B2b":
            field = "TGD_B2bI"
        else:
            raise ValueError(
                f"Unsupported BeiDou broadcast group delay for signal {signal}: "
                "modern B1C/B2a/B2b need RINEX 4 CNAV ISC/TGD fields or validated OSB handling."
            )
        if field not in orbit.columns:
            raise ValueError(f"BeiDou broadcast orbit does not contain required {field} field.")
        values = orbit[field]
        if values.isna().all():
            raise ValueError(
                f"BeiDou broadcast orbit contains {field}, but it is empty for mode {self.config.bds_mode}. "
                "This mode needs fields from multiple RINEX 4 CNAV message types or a validated OSB/DSB product."
            )
        return values

    def _bds_broadcast_group_delay_to_clock_datum(self, orbit: pd.DataFrame) -> pd.DataFrame:
        """
        Convert BDS broadcast group delays to the internal meter-level TGD.

        Status:
            Active clock-datum helper, requires caution.

        The output is a DataFrame named `TGD` in meters. Single-signal modes use
        that signal's broadcast delay. Two-signal modes use the
        ionosphere-free coefficients from `mode_ionosphere_free_coefficients`.
        Modern modes are only accepted when the required RINEX 4 fields are
        present; otherwise the method raises instead of silently producing a
        misleading clock correction.
        """

        signals = mode_signals(self.config.bds_mode)
        unsupported_signals = set(signals) - BDS_LEGACY_BROADCAST_TGD_SIGNALS
        has_modern_broadcast_fields = bool({"TGD_B1Cp", "TGD_B2ap", "TGD_B2bI"}.intersection(orbit.columns))
        if unsupported_signals and not has_modern_broadcast_fields:
            raise ValueError(
                "Unsupported BeiDou broadcast clock/TGD mode "
                f"{self.config.bds_mode!r}: signals {sorted(unsupported_signals)} require "
                "RINEX 4 BDS CNAV signal-specific ISC/TGD fields or validated OSB handling."
            )
        if len(signals) == 1:
            tgd_seconds = self._bds_signal_tgd_seconds(orbit, signals[0])
        elif len(signals) == 2:
            a, b = mode_ionosphere_free_coefficients(self.config.bds_mode)
            tgd_seconds = (
                a * self._bds_signal_tgd_seconds(orbit, signals[0])
                - b * self._bds_signal_tgd_seconds(orbit, signals[1])
            )
        else:
            raise ValueError(f"Unsupported BeiDou mode for broadcast group delay: {self.config.bds_mode}")
        return tgd_seconds.to_frame(name="TGD") * self.clight

    def _merge_bds_broadcast_tgd_without_external_dcb(self, orbit: pd.DataFrame, orbit_type: str) -> pd.DataFrame:
        """
        Merge broadcast BDS TGD when no external bias product is configured.

        Status:
            Active fallback policy.

        Broadcast NAV inputs use the BDS TGD/ISC fields available in the orbit
        table. Precise products receive zero TGD so that later comparison code
        can operate on a uniform schema.
        """

        if orbit_type == 'RINEX_NAV':
            tgd = self._bds_broadcast_group_delay_to_clock_datum(orbit)
            return orbit.join(tgd)
        orbit['TGD'] = 0.0
        return orbit

    def _get_bds_dcb(self, orbit, orbit_type, dcb_path, bds_field):
        """
        Load or derive a BDS bias source for the selected orbit product.

        Status:
            Active BDS bias policy helper.

        External precise products read BIA/OSB/DSB data. Broadcast navigation
        inputs derive `TGD` from the NAV fields so they share the same meter
        convention as external products.
        """

        if dcb_path is not None and orbit_type != 'RINEX_NAV':
            return GNSSDataProcessor2().read_bia(path=dcb_path)
        if orbit_type == 'RINEX_NAV':
            return self._bds_broadcast_group_delay_to_clock_datum(orbit), 'brdc'
        return None, None

    def _bds_observation_candidates(self, signal: str) -> list[str]:
        """
        Return prioritized BDS observable codes for a logical signal name.

        Status:
            Active helper used by OSB/DSB matching.
        """

        spec = signal_spec(signal)
        return [f"{spec.code_prefix}{suffix}" for suffix in spec.suffix_priority]

    @staticmethod
    def _available_bds_bias_observables(dcb: pd.DataFrame) -> set[str]:
        """
        Read advertised BDS OSB observables from a parsed BIA product.

        Status:
            Active metadata helper. Empty metadata is treated as unknown rather
            than as a hard failure, because older parsers may not populate attrs.
        """

        availability = dcb.attrs.get("bias_observables_by_system", {})
        values = availability.get("C", []) if isinstance(availability, dict) else []
        return set(values)

    @staticmethod
    def _available_bds_bias_pairs(dcb: pd.DataFrame) -> set[str]:
        """
        Read advertised BDS DSB observable pairs from a parsed BIA product.

        Status:
            Active metadata helper for pair selection and error reporting.
        """

        availability = dcb.attrs.get("bias_pairs_by_system", {})
        values = availability.get("C", []) if isinstance(availability, dict) else []
        return set(values)

    def _resolve_bds_osb_observable(self, dcb: pd.DataFrame, signal: str) -> str:
        """
        Select the best OSB observable column for a BDS signal.

        Status:
            Active, requires caution for mixed bias products.

        The resolver follows the observable suffix priority from `gnx_py.gnss`
        and checks both DataFrame columns and optional parser metadata. It raises
        a descriptive error when the requested signal is not represented.
        """

        available = self._available_bds_bias_observables(dcb)
        candidates = self._bds_observation_candidates(signal)
        for obs in candidates:
            if f"OSB_{obs}" in dcb.columns and (not available or obs in available):
                return obs
        raise ValueError(
            f"No BDS OSB observable found for signal {signal} in DCB/OSB product. "
            f"Tried {candidates}; available BDS observables: {sorted(available) or 'unknown'}."
        )

    def _resolve_bds_dsb_pair(self, dcb: pd.DataFrame, signal1: str, signal2: str) -> tuple[str, str, str]:
        """
        Select a DSB observable pair for a two-signal BDS mode.

        Status:
            Active, requires caution.

        Returns:
            tuple: `(obs1, obs2, direction)`, where direction is `"direct"` or
            `"reverse"` depending on how the BIA product stores the pair.
        """

        available = self._available_bds_bias_pairs(dcb)
        candidates1 = self._bds_observation_candidates(signal1)
        candidates2 = self._bds_observation_candidates(signal2)
        for obs1 in candidates1:
            for obs2 in candidates2:
                direct = f"{obs1}_{obs2}"
                reverse = f"{obs2}_{obs1}"
                if f"BIAS_{direct}" in dcb.columns and (not available or direct in available):
                    return obs1, obs2, "direct"
                if f"BIAS_{reverse}" in dcb.columns and (not available or reverse in available):
                    return obs1, obs2, "reverse"
        raise ValueError(
            f"No BDS DSB pair found for mode {self.config.bds_mode} in DCB product. "
            f"Tried {candidates1} x {candidates2}; available BDS pairs: {sorted(available) or 'unknown'}."
        )

    def _bds_bias_columns(self, dcb: pd.DataFrame) -> tuple[str, str]:
        """
        Return OSB observables used by a two-signal BDS mode.

        Status:
            Active helper retained for diagnostics and tests. Single-signal
            modes return empty strings because they do not form an IF pair.
        """

        signals = mode_signals(self.config.bds_mode)
        if len(signals) == 1:
            return "", ""
        return (
            self._resolve_bds_osb_observable(dcb, signals[0]),
            self._resolve_bds_osb_observable(dcb, signals[1]),
        )

    def _convert_bds_dcb_to_tgd(self, dcb, type_dcb):
        """
        Convert BDS OSB/DSB/BRDC bias data to the internal TGD convention.

        Status:
            Active BDS clock/bias conversion helper, requires caution.

        OSB inputs are converted from nanoseconds to meters for either a
        single-signal datum or an ionosphere-free combination. DSB inputs are
        supported only for two-signal modes because a DSB pair cannot define a
        single-signal clock datum. Broadcast (`brdc`) data are already in the
        internal representation and are returned unchanged.
        """

        if type_dcb in {'brdc', None} or dcb is None:
            return dcb
        signals = mode_signals(self.config.bds_mode)
        if type_dcb == 'OSB':
            if len(signals) == 1:
                obs = self._resolve_bds_osb_observable(dcb, signals[0])
                col = f"OSB_{obs}"
                out = dcb[[col]] * 1e-9 * self.clight
                return out.rename(columns={col: 'TGD'})
            if len(signals) == 2:
                obs1 = self._resolve_bds_osb_observable(dcb, signals[0])
                obs2 = self._resolve_bds_osb_observable(dcb, signals[1])
                col1, col2 = f"OSB_{obs1}", f"OSB_{obs2}"
                a, b = mode_ionosphere_free_coefficients(self.config.bds_mode)
                out = (a * dcb[[col1]] - b * dcb[[col2]].values) * 1e-9 * self.clight
                return out.rename(columns={col1: 'TGD'})
        if type_dcb == 'DSB':
            if len(signals) != 2:
                raise ValueError(
                    f"BDS DSB products cannot define a single-signal clock datum for {self.config.bds_mode}; "
                    "use a matching OSB product instead."
                )
            obs1, obs2, direction = self._resolve_bds_dsb_pair(dcb, signals[0], signals[1])
            _, b = mode_ionosphere_free_coefficients(self.config.bds_mode)
            direct = f"BIAS_{obs1}_{obs2}"
            reverse = f"BIAS_{obs2}_{obs1}"
            if direction == "direct":
                out = -b * dcb[[direct]] * 1e-9 * self.clight
                return out.rename(columns={direct: 'TGD'})
            out = b * dcb[[reverse]] * 1e-9 * self.clight
            return out.rename(columns={reverse: 'TGD'})
        return None

    # --- ŁĄCZENIE TGD Z ORBITAMI (merge_asof z indexowaniem) ---
    def _merge_tgd_with_orbit(self, orbit, dcb):
        """
        Merge GPS per-satellite group delays with orbit/measurement data.

        Returns:
            table-like: Merged structure with delay columns appended.
        """

        if dcb is not None:
            orbit = pd.merge_asof(
                orbit.sort_values('time'), dcb.sort_values('time'),
                left_on='time', right_on='time', by='sv', direction='nearest'
            )
            orbit = orbit.set_index(['sv', 'time'])
        return orbit

    def _merge_gal_tgd_with_orbit(self, orbit, dcb_out, type_dcb):
        """
        Merge Galileo per-satellite group delays with orbit/measurement data.

        Returns:
            table-like: Merged structure with delay columns appended.
        """

        orbit = orbit.sort_values('time')
        if dcb_out is not None:
            orbit = pd.merge_asof(
                orbit, dcb_out.sort_values('time'),
                left_on='time', right_on='time', by='sv', direction='nearest'
            )
            orbit = orbit.set_index(['sv', 'time'])
        else:
            if type_dcb != 'brdc':
                orbit['TGD'] = 0.0
        return orbit

    @guarded_session_run("SISController")
    def run(self,kr = 0.9604, kb = 1.0, kac = 0.019881, krb = -1.960):
        """
        Execute the configured SIS processing pipeline.

        Status:
            Active public method. This method is intentionally procedural so the
            routing remains explicit for broadcast-vs-precise, SP3-vs-SP3 and
            bias-product comparisons.

        Steps (high level):
            1) Input classification
            2) Orbit processing (broadcast or precise)
            3) Optional eclipse masking
            4) Optional satellite PCO/PCV application
            5) DCB ingestion and conversion to group delays
            6) Final merging and output preparation

        Returns:
            table-like: Final product of the pipeline (e.g., DataFrame with orbit
            coordinates, clocks, and per-satellite delays).

        Raises:
            Exception: Propagates critical errors encountered during processing.
        """

        # ---- orbit interpolation
        orbit_0_type = self.classify_file(self.config.orb_path_0)
        orbit_1_type = self.classify_file(self.config.orb_path_1)
        _logger.debug("SIS pipeline event=classified orbit_0_type=%s orbit_1_type=%s", orbit_0_type, orbit_1_type)
        if orbit_0_type == 'RINEX_NAV':
            orbit_0 = self.process_broadcast_orbit(path=self.config.orb_path_0)
        elif orbit_0_type =='SP3':
            orbit_0 = self.process_precise_orbit(path=self.config.orb_path_0,prev_path=self.config.prev_sp3_0,
                                                 next_path=self.config.next_sp3_0)
            if self.config.apply_satellite_pco:
                orbit_0 = self.apply_satellite_pco(precise_orbit=orbit_0)

        if orbit_1_type == 'RINEX_NAV':
            orbit_1 = self.process_broadcast_orbit(path=self.config.orb_path_1)
        elif orbit_1_type == 'SP3':
            orbit_1 = self.process_precise_orbit(path=self.config.orb_path_1,prev_path=self.config.prev_sp3,
                                                 next_path=self.config.next_sp3)
            if self.config.apply_satellite_pco:
                orbit_1 = self.apply_satellite_pco(precise_orbit=orbit_1)
        _logger.debug("SIS pipeline event=orbits-processed")
        # ---- DCB comparison
        orbit_0, orbit_1 = self.process_dcb_and_merge(orbit_0, orbit_1, orbit_0_type, orbit_1_type)
        orbit_0 = self._ensure_apc_columns(orbit_0)
        orbit_1 = self._ensure_apc_columns(orbit_1)
        _logger.debug("SIS pipeline event=dcb-processed")

        # ---- SISRE computation
        orbit_0, orbit_1 = self.set_common_epochs(orbit_0, orbit_1)
        _logger.debug("SIS pipeline event=common-epochs-processed epochs=%d", len(orbit_0))
        compared_orbits = self.compare_orbits(orbit_0=orbit_0, orbit_1=orbit_1,
                                              clock_bias=self.config.clock_bias,
                                              clock_bias_function=self.config.clock_bias_function,
                                              kr=kr, krb=krb, kac=kac,kb=kb)
        if self.config.apply_eclipse:
            compared_orbits = self.eclipse_periods(orbit=compared_orbits)
            self.output_cols += ['eclipse_raw', 'eclipse']
        compared_orbits = self._annotate_clock_convention(compared_orbits, orbit_0_type, orbit_1_type)
        return compared_orbits#[self.output_cols]
