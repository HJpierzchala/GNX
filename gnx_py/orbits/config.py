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


@dataclass
class SISConfig:
    """
    Container for SIS processing configuration.

    This class centralizes resource paths, runtime options, and modeling flags
    used by the SIS pipeline (e.g., SP3 paths, ATX path, DCB handling, time
    limits and interpolation interval, eclipse modeling, and satellite PCO).

    Attributes (selection):
        systems: GNSS systems to process (e.g., {"G", "E"}).
        gps_mode: Processing mode or frequency pair for GPS.
        gal_mode: Processing mode or frequency pair for Galileo.
        interval: Interpolation/processing interval (in seconds or minutes, as used by the controller).
        orb_path_0, orb_path_1: Paths to precise orbit products (e.g., SP3).
        dcb_path_0, dcb_path_1: Paths to DCB/OSB resources, if used.
        compare_dcb: Whether to compare multiple DCB sources.
        atx_pat: Path to antenna phase center model (ATX).
        apply_eclipse: Whether to detect/apply eclipse masking.
        extend_eclipse: Whether to extend eclipse masks around ingress/egress.
        apply_satellite_pco: Whether to apply satellite PCO/PCV corrections.
        prev_sp3_0, next_sp3_0: Paths to adjacent-day SP3 files for boundary stabilization.
        prev_sp3, next_sp3: Cached/parsed adjacent SP3 references if applicable.
        clock_bias, clock_bias_function: Clock handling mode and optional function.
        extension_time: Time extension around product boundaries for safe interpolation.
        tlim: Optional time window (start, end) for processing.

    Notes:
        - File path attributes should reference accessible resources in supported formats.
        - This object is typically treated as immutable during a single run.
    """

    def __init__(self, orb_path_0, orb_path_1, interval, prev_sp3=None, next_sp3=None, dcb_path_0=None, dcb_path_1=None, gps_mode=None ,gal_mode=None, atx_path=None, system='G',
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
        Orchestrates SIS data processing driven by a SISConfig instance.

        Responsibilities:
            - Classify inputs and route them to broadcast or precise-orbit handlers.
            - Process orbits (interpolation, alignment, sys selection).
            - Detect eclipse periods and optionally extend masks.
            - Apply satellite antenna PCO/PCV when enabled by configuration.
            - Ingest DCB/OSB, convert to appropriate group delays (e.g., TGD) when needed,
              and merge them with orbit/measurement data.
            - Produce a table-like result aligned to the chosen epoch grid.

        Attributes:
            config: Configuration object controlling file paths, modes, and flags.
            system: GNSS sys code processed by this controller instance.
            clight: Speed of light in vacuum [m/s].
            output_cols: Expected output columns of the final table, if defined.

        Notes:
            - The controller is stateful during a run; create a new instance for independent sessions.
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

                The method interpolates satellite coordinates (and related fields as supported)
                from broadcast navigation messages and aligns them to the working epochs.

                Returns:
                    table-like: A structure (e.g., pandas.DataFrame) holding per-satellite
                    coordinates and ancillary fields aligned to the controller's epochs.

                Raises:
                    RuntimeError: If interpolation fails or input data are insufficient.
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

        broadcast_interpolator = BrdcGenerator(system=self.system,
                                               interval=self.config.interval,
                                               mode=mode,
                                               nav=orbit,tolerance=tol)
        crd = broadcast_interpolator.generate()
        crd[['x_apc','y_apc','z_apc']] = crd[['x','y','z']]
        return crd

    def process_precise_orbit(self, path: str | Path, prev_path: str|Path, next_path:str|Path):
        """
                Process precise orbits (SP3) and align them to an internally determined epoch grid.

                The method reads the nominal-day SP3 file and may include adjacent-day SP3
                files to stabilize boundary interpolation. Epochs are derived from the
                configuration time window (tlim and interval) when provided; otherwise
                they are inferred from the SP3 time span.

                Args:
                    path (str | pathlib.Path): Path to the nominal-day SP3 file.
                    prev_path (str | pathlib.Path): Path to the previous-day SP3 file (may be unused if None-like).
                    next_path (str | pathlib.Path): Path to the next-day SP3 file (may be unused if None-like).

                Returns:
                    pandas.DataFrame: Satellite positions (and clocks if available), indexed by
                    satellite and time, aligned to the computed epoch grid.

                Raises:
                    RuntimeError: If interpolation or alignment fails due to insufficient or invalid data.

                Notes:
                    - The epoch step is derived from the configuration interval.
                    - Adjacent-day files, when present, improve interpolation at the edges.
                """

        sp3 =   read_sp3(path=path, sys=self.system)
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
        cfg =  DefaultConfig(gps_freq=self.config.gps_mode, gal_freq=self.config.gal_mode)
        processor = DDPreprocessing(df=precise_orbit, flh=None, xyz=np.array([1e7, 1e7, 1e7]),
                                        phase_shift_dict=None, sat_pco=sat_pco, rec_pco=None, antenna_h=None,
                                        system=self.system)
        precise_orbit[['xe', 'ye', 'ze']] = precise_orbit[['x', 'y', 'z']]
        processor.sat_fixed_system()
        precise_apc = processor.df.copy().swaplevel()
        return precise_apc


    def compare_orbits(self, orbit_0:pd.DataFrame, orbit_1:pd.DataFrame, kr, kb, kac, krb, clock_bias=True, clock_bias_function='mean'):
        """
         Compare two orbit solutions on a shared epoch/satellite set.

         Returns:
             pd.DataFrame: Summary metrics and/or per-satellite time series suitable
             for analysis and plotting.
         """
        orbit_0[['dx', 'dy', 'dz', 'dt','dTGD']] = (
                    orbit_0[['x_apc', 'y_apc', 'z_apc', 'clk','TGD']] - orbit_1[['x_apc', 'y_apc', 'z_apc', 'clk','TGD']]).values
        compare = orbit_0.copy()
        compare['dt']*=self.clight
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
            ure_orbit = np.sqrt(
                kr * dR ** 2 + kac * ac)
            ure_sv_ga = np.sqrt(
                kr * dR ** 2 + kb * (dt-dtgd) ** 2 + kac * ac + krb * dR * (dt-dtgd))
            ure_sv_ga_notgd = np.sqrt(
                kr * dR ** 2 + kb * dt ** 2 + kac * ac + krb * dR * dt)
            compare.loc[group.index, 'sisre'] = ure_sv_ga
            compare.loc[group.index, 'sisre_orb'] = ure_orbit
            compare.loc[group.index, 'sisre_notgd'] = ure_sv_ga_notgd
        return compare

        # --- GŁÓWNA METODA PUBLICZNA ---
    def process_dcb_and_merge(self, orbit_0, orbit_1, orbit_0_type, orbit_1_type):
        """
        Ingest DCB/OSB resources, convert to appropriate group delays if needed,
        and merge them with the orbit/measurement data.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame): Orbit/measurement table with per-satellite delay fields merged.

        Raises:
            RuntimeError: If DCB parsing or conversion fails.
        """

        if not self.config.compare_dcb:
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
            print('DCB OUT is None')
            if type_dcb != 'brdc':
                orbit['TGD'] = 0.0
        return orbit

    def run(self,kr = 0.9604, kb = 1.0, kac = 0.019881, krb = -1.960):
        """
        Execute the configured SIS processing pipeline.

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

        # ---- DCB comparison
        orbit_0, orbit_1 = self.process_dcb_and_merge(orbit_0, orbit_1, orbit_0_type, orbit_1_type)

        # ---- SISRE computation
        orbit_0, orbit_1 = self.set_common_epochs(orbit_0, orbit_1)
        compared_orbits = self.compare_orbits(orbit_0=orbit_0, orbit_1=orbit_1,
                                              clock_bias=self.config.clock_bias,
                                              clock_bias_function=self.config.clock_bias_function,
                                              kr=kr, krb=krb, kac=kac,kb=kb)
        if self.config.apply_eclipse:
            compared_orbits = self.eclipse_periods(orbit=compared_orbits)
            self.output_cols += ['eclipse_raw', 'eclipse']
        return compared_orbits[self.output_cols]
