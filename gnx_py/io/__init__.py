import math
import re
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass, replace as _dataclass_replace
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional
from typing import List
import numpy as np
import pandas as pd

from ..core.conversion import ecef2geodetic
# the readers of bug-fix stage 10, imported as modules (``from .io import *`` in the package exports no new name)
from . import compression as _compression
from . import rinex2_adapter as _rinex2
from . import rinex_nav as _rinex_nav
from . import rinex_obs as _rinex_obs
from . import sp3_reader as _sp3_reader
from ..core.gnss import (
    DEFAULT_MODE_BY_SYSTEM,
    mode_signals,
    signal_spec,
    validate_mode_for_system,
)
from .rinex2_adapter import (  # adapt_rinex2_gps_nav, resolve_rinex_version: kept in the package namespace
    RINEX2_OBS_SYSTEMS,
    adapt_rinex2_gps_nav,
    adapt_rinex2_obs,
    is_supported_rinex2_version,
    resolve_rinex_version,
)
from ..core.time import doy_to_datetime, datetime2toc  # doy_to_datetime: kept in the package namespace

def parse_phase_shift(fields: str) -> dict:
    """
        Parses the string from the "SYS / PHASE SHIFT" header in the RINEX file.
        For each observation (e.g. "G L1C"), checks whether there is a numerical value.
        If not, returns 0.0; if so, converts the value to float.

        The returned dictionary has the following form:
          {
             'G L1C': 0.0,
             'G L1L': 0.25000,
             ...
          }
        """
    result = {}
    pattern = re.compile(r"([A-Z])\s+([A-Z0-9]+)(?:\s+([+-]?\d+\.\d+))?")

    for match in pattern.finditer(fields):
        system, obs, value = match.groups()
        key = f"{system} {obs}"
        result[key] = float(value) if value else 0.0
    return result


@dataclass
class OrbitData:
    """
    Nav data container

    ``gps_orb``/``gal_orb``/``bds_orb``: broadcast ephemerides, index (sv, time = toc). ``gpsa``, ``gpsb``,
    ``gala``, ``bdsa``, ``bdsb``: one broadcast ionosphere coefficient set per model (lists of floats or None).
    ``iono``: every broadcast ionosphere set of the file with its source, transmission time and plausibility
    (``rinex_nav.IONO_COLUMNS``), ``nav_info``: version, file type and systems of the file
    (``rinex_nav.NavInfo``); both None when not read.
    """
    gps_orb: Optional[pd.DataFrame] = None
    gal_orb: Optional[pd.DataFrame] = None
    bds_orb: Optional[pd.DataFrame] = None
    gpsa: float | None = None
    gpsb: float | None = None
    gala: float | None = None
    bdsa: float | None = None
    bdsb: float | None = None
    iono: Optional[pd.DataFrame] = None
    nav_info: Optional[object] = None


@dataclass
class ObsData:
    """
    Obs data container

    ``gps``/``gal``/``bds``: one frame per system, index ['sv', 'time'] (time in GPST), one column per observation
    type selected for the mode. ``meta``: the 12 header items of
    ``process_observations2`` (marker name, antenna type, radome, antenna DELTA H/E/N, approx. position XYZ and
    its geodetic position - both None without a usable APPROX POSITION XYZ -, interval [s], GPS and Galileo
    observation types, first and last epoch, phase shifts). ``interval``: [min]. ``header``: the
    ``rinex_obs.ObsHeader`` of the file. Every frame carries ``attrs['phase_shift']`` and
    ``attrs['phase_shift_events']`` of ``rinex_obs.read_obs``: the SYS / PHASE SHIFT records of the header with
    their satellite lists and those inside the data (empty for RINEX 2), stored
    read-only and JSON-safe; ``rinex_obs.phase_shift_of(frame)`` / ``rinex_obs.phase_shift_events_of(frame)`` give
    them in the contract form. meta[11] keeps the earlier text form (``parse_phase_shift``: one value per code, no
    satellite lists).
    """
    gps: pd.DataFrame = None
    gal: pd.DataFrame = None
    bds: pd.DataFrame = None
    sat_pco: defaultdict = None
    rec_pco: defaultdict = None
    meta: tuple= None
    interval: float = None
    header: Optional[object] = None


#: A navigation file that cannot give the requested systems (``rinex_nav.RinexNavError``, a ValueError naming the
#: file, its version and the systems; since v1.0.0). The former RINEX 4 error class is this one.
Rinex4NavReadError = _rinex_nav.RinexNavError


def _check_time_limit(tlim, name: str = "tlim") -> tuple[pd.Timestamp, pd.Timestamp]:
    """``tlim`` as ``(start, end)`` Timestamps; ValueError naming the time limit otherwise."""
    from ..core.time import check_time_limit

    return check_time_limit(tlim, name)


#: Since v1.0.0 a single-frequency mode (GPS L1, L2, L5; Galileo E1, E5a, E5b) selects and
#: requires only the observables of its own band. True restores the earlier rule, under which the GPS and Galileo
#: selectors also took a second band (L1 and L2: L1+L2, L5: L1+L5, E1 and E5a: E1+E5a, E5b: E1+E5b) and the row
#: filter required it, dropping the epochs where only the second band was missing (BRUX 2024-035: L5 2.7 % of
#: the rows, E5a 1.5 %, E5b 1.4 %, E1 0.26 %). BeiDou selects per signal of its mode in both cases.
SINGLE_FREQUENCY_NEEDS_SECOND_BAND = False

_SYSTEM_NAME = {"G": "GPS", "E": "Galileo", "C": "BeiDou", "R": "GLONASS", "J": "QZSS", "I": "NavIC", "S": "SBAS"}

#: Tracking attributes of a BeiDou signal in the strict pick where ``gnss.SignalSpec.suffix_priority`` is wider than
#: the signal: RINEX 3.05/4.0x band 1 S/L/Z is B1A, not B1C (B1C is D/P/X).
_BDS_PICK_ATTRIBUTES = {"B1C": ("X", "P", "D")}


def _bds_attributes(signal: str) -> tuple[str, ...]:
    """RINEX tracking attributes of the BeiDou ``signal`` in priority order (strict pick)."""
    return _BDS_PICK_ATTRIBUTES.get(signal, signal_spec(signal).suffix_priority)

# validity bounds of a Bias-SINEX record without a (parseable) BIAS_START / BIAS_END
_BIAS_OPEN_START = pd.Timestamp("1980-01-06")
_BIAS_OPEN_END = pd.Timestamp("2200-01-01")
# BIAS_END is given to the second: a record holds until the end of its last second, so a daily record
# ending at 23:59:59 (GRG) closes the day like one ending at 24:00 (CODE, CAS)
_BIAS_END_RESOLUTION = pd.Timedelta(seconds=1)


class GNSSDataProcessor2:
    """
    GNSS data processor. Supports:
    - RINEX 2/3/4 observation files (``gnx_py.rinex_obs``)
    - RINEX 2/3/4 navigation files
    - DCB .BIA and .BSX files
    - ANTEX files
    Main class for opening basic GNSS input files

    """

    def __init__(self, atx_path=None, obs_path=None, dcb_path=None, nav_path=None, mode='L1', sys='G',
                 galileo_modes=None, beidou_modes=None, use_gfz=False, station_name=None,
                 rinex2_tracking=None, gps_sf_same_tracking=False):
        """
        :param atx_path: path to .atx file
        :param obs_path: path to .obs file
        :param dcb_path: path to .dcb file
        :param nav_path: path to nav file
        :param use_gfz: accepted, no effect since v1.0.0: every observation file (RINEX 2/3/4, plain,
            gzip, .Z, bzip2, zip, CRINEX) is read by ``gnx_py.rinex_obs`` in memory

        :param sys: str, set
        :param mode: GPS mode (L1, L1L2 eec)
        :param galileo_modes: Galileo modes (E1, E1E5a etc)
        :param rinex2_tracking: RINEX 2 observations: {system: {RINEX 2 observable: RINEX 3 observable}}
            whose product biases they take, before the header and the defaults (``Config.rinex2_tracking``,
            ``biases.resolve_rinex2_tracking``)
        :param gps_sf_same_tracking: a single-band GPS mode (L1, L2, L5) takes the code of the phase's tracking
            (C1C with L1C, not the semi-codeless C1W) when the file has such a pair (:meth:`select_obs_types_gps`). Default False: SPP, TEC and every dual-frequency mode keep
            the selection of earlier stages; ``PPPSession`` sets it for its single-band GPS modes
        """
        self.atx_path = atx_path
        self.obs_path = obs_path
        self.dcb_path = dcb_path
        self.nav_path = nav_path
        self.mode = mode
        self.use_gfz=use_gfz

        if isinstance(sys, str):
            self.sys = {sys}
        else:
            self.sys = set(sys)
        self.galileo_modes = galileo_modes
        self.beidou_modes = beidou_modes or DEFAULT_MODE_BY_SYSTEM["C"]
        self.station_name = station_name
        self.rinex2_tracking = rinex2_tracking
        self.gps_sf_same_tracking = bool(gps_sf_same_tracking)

        self.dcb_paths = self._normalize_dcb_paths(dcb_path)
        # sampling token of the first bias file name (IGS long name, e.g. '01D' or '30S'), kept for
        # information only: biases are assigned to epochs by the validity of each record
        # (BIAS_START/BIAS_END), whatever the file is called
        self.dcb_type = None
        if self.dcb_paths:
            name_tokens = Path(str(self.dcb_paths[0])).name.split('_')
            self.dcb_type = name_tokens[3] if len(name_tokens) > 3 else None
            self.add_dcb = True
        else:
            self.add_dcb = False

    @staticmethod
    def _normalize_dcb_paths(dcb_path) -> list[str]:
        if dcb_path is None:
            return []
        if isinstance(dcb_path, (list, tuple, set)):
            return [str(path) for path in dcb_path if path]
        return [str(dcb_path)]

    def _load_bias_products(self) -> pd.DataFrame | None:
        """Read every file of ``dcb_path`` (Bias-SINEX OSB/DSB, also gzip-compressed).

        Returns the records of all products (``read_bia`` frames stacked, in the order given; records of
        several products for the same satellite and start are all kept) with ``attrs["bias_products"]``:
        the frame of each product, from which ``_attach_dcb_to_observations`` takes, per satellite and
        epoch, the biases of the first product that has a valid record (products are not merged value by
        value: OSBs of different analysis centres refer to different clock datums).
        """
        merged = None
        products = [self.read_bia(path=path)[0] for path in self.dcb_paths]
        if len(products) == 1:
            merged = products[0].copy()
        elif products:
            def records(frame: pd.DataFrame) -> pd.DataFrame:  # without attrs (concat compares them)
                frame = frame.copy(deep=False)
                frame.attrs = {}
                return frame

            merged = pd.concat([records(p) for p in products], sort=False)
            merged.attrs = dict(products[0].attrs)
            station_parts = [records(p.attrs["station_bias_by_system"]) for p in products
                             if isinstance(p.attrs.get("station_bias_by_system"), pd.DataFrame)]
            if station_parts:
                merged.attrs["station_bias_by_system"] = pd.concat(station_parts, sort=False)
        if merged is not None:
            merged.attrs["bias_products"] = products

        return merged

    def _obs_header(self):
        """The ``rinex_obs.ObsHeader`` of ``obs_path``, read once per path (every record by column, the true
        RINEX version also for CRINEX). ``rinex_obs.RinexObsError`` (a ValueError naming the file) for a file
        that is not a RINEX observation file."""
        cached = self.__dict__.get("_obs_header_cache")
        if cached is None or cached[0] != self.obs_path:
            cached = self._obs_header_cache = (self.obs_path, _rinex_obs.read_header(self.obs_path))
        return cached[1]

    def _get_obs_rinex_info(self) -> dict:
        """Version, file type and systems of the observation file, as ``georinex.rinexinfo`` gave them but with
        the true RINEX version (``rinexinfo`` reported the CRINEX version, 1.0 over RINEX 2 and 3.0 over RINEX
        3, so a Hatanaka-compressed RINEX 2 file skipped the adapter). Empty without ``obs_path``."""
        if not self.obs_path:
            return {}
        header = self._obs_header()
        return {"version": header.version, "filetype": header.file_type, "rinextype": "obs",
                "systems": header.system}

    def _get_obs_header(self) -> dict:
        """The observation header as the dict ``georinex.rinexheader`` gave (``ObsHeader.georinex_view``: every
        label with its line[:60] payloads, plus 'version', 'fields', 't0', 'interval', 'position' ...), read by
        ``rinex_obs`` (antex, configuration and the TEC session read it). Empty without ``obs_path``."""
        if not self.obs_path:
            return {}
        cached = self.__dict__.get("_obs_header_view")
        if cached is None or cached[0] != self.obs_path:
            cached = self._obs_header_view = (self.obs_path, self._obs_header().georinex_view())
        return cached[1]

    def _is_rinex2_obs(self) -> bool:
        return is_supported_rinex2_version(self._get_obs_rinex_info().get("version"))

    @staticmethod
    def _adapter_report(df: pd.DataFrame | None) -> dict:
        if isinstance(df, pd.DataFrame):
            return df.attrs.get("gnx_adapter", {})
        return {}

    @staticmethod
    def _infer_interval_from_df(df: pd.DataFrame | None) -> float | None:
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.MultiIndex):
            return None
        if "time" not in df.index.names:
            return None
        times = pd.Index(df.index.get_level_values("time")).drop_duplicates().sort_values()
        if len(times) < 2:
            return None
        diffs = times.to_series().diff().dt.total_seconds().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return None
        return float(diffs.median()) / 60.0

    def _last_obs_read(self):
        """The ``rinex_obs.ObsRead`` (without frames) of the last load of ``obs_path``; None before any load or
        after ``obs_path`` changed."""
        cached = self.__dict__.get("_obs_read")
        return cached[1] if cached is not None and cached[0] == self.obs_path else None

    def _obs_interval_s(self) -> float | None:
        """Observation interval [s]: the median spacing of the epochs read
        (``rinex_obs.read_obs``), with fewer than 2 epochs the header INTERVAL, then the interval token of the file
        name with its unit (C = 100 Hz, Z = Hz, S, M, H, D); before any read the header INTERVAL. A header
        INTERVAL more than 1 % off the data gives one UserWarning (the data is used). None when nothing gives an
        interval."""
        read = self._last_obs_read()
        if read is None:
            return self._obs_header().interval_header if self.obs_path else None
        seconds = read.interval_s
        header_s = read.header.interval_header
        if (read.interval_source == "data" and seconds and header_s is not None
                and abs(header_s - seconds) > 0.01 * seconds):
            notes = self.__dict__.setdefault("_obs_notes", set())
            if ("interval", self.obs_path) not in notes:
                notes.add(("interval", self.obs_path))
                warnings.warn(f"{self.obs_path}: header INTERVAL {header_s:g} s differs from the spacing of the "
                              f"observation epochs ({seconds:g} s); the interval of the data is used",
                              UserWarning, stacklevel=3)
        return seconds

    def _resolve_obs_interval_minutes(
        self,
        gps: pd.DataFrame | None,
        gal: pd.DataFrame | None,
        bds: pd.DataFrame | None = None,
    ) -> float | None:
        """Observation interval [min]: :meth:`_obs_interval_s` (data first), else the median epoch spacing of the
        frames; 0.5 min (30 s) with a UserWarning when nothing gives one."""
        seconds = self._obs_interval_s()
        if seconds:
            return float(seconds) / 60.0
        inferred = self._infer_interval_from_df(gps) or self._infer_interval_from_df(gal) or self._infer_interval_from_df(bds)
        if inferred is not None:
            return inferred
        warnings.warn(f"{self.obs_path}: the observation interval is unknown (fewer than 2 epochs, no usable "
                      f"INTERVAL record, no interval token in the file name); 30 s is assumed",
                      UserWarning, stacklevel=3)
        return 0.5

    def _apply_obs_meta_overrides(self, gps_info: tuple, gps: pd.DataFrame | None, gal: pd.DataFrame | None) -> tuple:
        """The RINEX 2 adapter's observable lists (and no phase shifts) in the header items; first/last epoch from
        the frames when nothing was read (the interval and the epochs otherwise come from the data read)."""
        info = list(gps_info)
        # A mixed RINEX 2 file is adapted once per constellation, so each frame
        # carries its own report and only knows its own observable list. Apply
        # them in turn and let each fill in the slot it actually owns.
        reports = [report for report in (self._adapter_report(gps), self._adapter_report(gal)) if report]
        if reports and any(report.get("meta_override", {}).get("phase_shift_dict") is None
                           for report in reports):
            # RINEX 2 has no SYS / PHASE SHIFT block at all.
            info[11] = None

        for report in reports:
            override = report.get("meta_override", {})
            if override.get("gps_obs") is not None:
                info[7] = override["gps_obs"]
            if override.get("gal_obs") is not None:
                info[8] = override["gal_obs"]

        if info[9] is None:
            for df in (gps, gal):
                if isinstance(df, pd.DataFrame) and len(df):
                    info[9] = df.index.get_level_values("time").min()
                    break
        if info[10] is None:
            for df in (gps, gal):
                if isinstance(df, pd.DataFrame) and len(df):
                    info[10] = df.index.get_level_values("time").max()
                    break

        return tuple(info)

    def _colspecs_from_header(self,header: str):
        colspecs, in_field = [], False
        for i, ch in enumerate(header):
            if ch != " " and not in_field:
                start, in_field = i, True
            elif ch == " " and in_field:
                colspecs.append((start, i))
                in_field = False
        if in_field:
            colspecs.append((start, len(header)))
        return colspecs

    def _to_datetime_gps(self,text: str):
        """YYYY:DOY:SSSSS  →  pandas.Timestamp (naive, GPS epoch)."""
        if pd.isna(text) or not text.strip():
            return pd.NaT
        try:
            year, doy, sec = text.split(":")
            base = datetime.strptime(f"{year} {doy}", "%Y %j")
            return pd.Timestamp(base + timedelta(seconds=int(sec)))
        except Exception:
            return pd.NaT

    def read_bia(self, path, *, encodings=("utf-8", "cp1250", "iso-8859-2", "latin-1"), parse_time=True) -> tuple[
        pd.DataFrame, str]:
        """Read a Bias-SINEX file (OSB and/or DSB records; ``.BIA``, ``.BSX``, plain or compressed: gzip, Unix
        compress, bzip2, zip).

        Returns ``(frame, file_type)``. ``frame``: satellite biases [ns], one row per satellite and
        validity record, index (``sv``, ``time`` = BIAS_START), column ``time_end`` (BIAS_END) and
        ``OSB_<obs>`` (OSB records) / ``BIAS_<obs1>_<obs2>`` (DSB records). An observable or pair the
        product does not give for a satellite is NaN (never 0.0: a 0.0 is a value of the file), and so is
        a code bias given as exactly 0.0 for every satellite of a system (at least two; a placeholder of a
        bias the product does not estimate, e.g. COD MGX BeiDou C2I/C6I), listed in
        ``attrs["placeholder_biases_by_system"]`` (the consumers warn for the system they process).
        ESTIMATED_SLOPE [ns/s]: where a record has a non-zero slope, ``SLOPE_<column>`` holds it and
        ``_bias_values_at`` gives value + slope * (t - t_ref), t_ref the middle of BIAS_START..BIAS_END
        (BIAS_START when the end is open, BIAS_END when the start is; Bias-SINEX 1.00, section 5.1).
        ``attrs["station_bias_by_system"]``: receiver biases, index (``station``, ``system``, ``time``)
        with ``time_end``, same columns. ``attrs["bias_observables_by_system"]`` /
        ``attrs["bias_pairs_by_system"]``: observables/pairs given per system. ``file_type``: ``'OSB'``
        or ``'DSB'`` (the type of the first record). A record is valid from BIAS_START to BIAS_END; the
        sampling token of the file name plays no role. An empty or zero BIAS_END is open-ended (an unreadable
        date as well, with a warning). Records in a unit other than ns are left out (warning).
        ``attrs["clock_reference_observables"]``: SATELLITE_CLOCK_REFERENCE_OBSERVABLES of the description
        block per system (``{'C': ['C2I', 'C6I']}``; GRG gives phase codes), empty when not declared.
        """
        # -- 1. Dekodowanie całego pliku ---------------------------------------
        # plain, gzip (.BIA.gz / .BSX.gz, as the downloader stores them), Unix compress (.Z), bzip2 or zip, in
        # memory (gnx_py.compression; a damaged archive raises CompressedFileError naming the file)
        raw = _compression.read_bytes(path, what="Bias file")
        for enc in encodings:
            try:
                text, used_enc = raw.decode(enc), enc
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Bias file {path}: not decodable with the encodings {list(encodings)}")

        lines = text.splitlines()
        # the record header line of the +BIAS/SOLUTION block (files without the block marker: the first one)
        block = next((i for i, ln in enumerate(lines) if ln.startswith("+BIAS/SOLUTION")), -1)
        idx = None
        try:
            idx = next(i for i in range(block + 1, len(lines)) if lines[i].lstrip().startswith("*BIAS"))
        except StopIteration:
            raise ValueError(f"No header matching *BIAS was found in {path}: not a Bias-SINEX file "
                             f"(+BIAS/SOLUTION block with a '*BIAS SVN_ PRN ...' header line)")
        end = next((i for i in range(idx + 1, len(lines)) if lines[i].startswith("-BIAS")), len(lines))
        # +BIAS/DESCRIPTION: observables defining the satellite clocks of the product per system (their
        # ionosphere-free combination carries no satellite bias), e.g. {'G': ['C1W', 'C2W']}
        clock_reference: dict[str, list[str]] = {}
        for line in lines[:idx]:
            tokens = line.split()
            if not line.startswith("*") and "SATELLITE_CLOCK_REFERENCE_OBSERVABLES" in tokens:
                given = tokens[tokens.index("SATELLITE_CLOCK_REFERENCE_OBSERVABLES") + 1:]
                if len(given) > 1:
                    clock_reference[given[0]] = given[1:]
        lines = lines[:end]

        file_type = None
        for line in lines[idx + 1:]:
            if line.strip() and not line.startswith("*"):  # Pomijamy puste linie i komentarze
                first_col = line.split()[0]
                if first_col in ["OSB", "DSB"]:
                    file_type = first_col
                    break
        if file_type is None:
            raise ValueError(f"Bias file {path}: no OSB or DSB record in its BIAS/SOLUTION block.")

        header = lines[idx]
        colspecs = self._colspecs_from_header(header)

        raw_names = [header[s:e].replace("*", "").strip().replace("_", "") for s, e in colspecs]
        counter, names = Counter(), []
        for n in raw_names:
            counter[n] += 1
            names.append(f"{n}_{counter[n]}" if counter[n] > 1 else n)

        df = pd.read_fwf(
            StringIO("\n".join(lines[idx + 1:])),
            colspecs=colspecs,
            names=names,
            comment="*",
            dtype=str,
        ).map(lambda x: x.strip() if isinstance(x, str) else x)

        def safe_to_numeric(col):
            try:
                return pd.to_numeric(col)
            except Exception:
                return col

        df = df.apply(safe_to_numeric)

        df.attrs["encoding"] = used_enc

        if parse_time and {"BIASSTART", "BIASEND"} <= set(df.columns):
            df["BIASSTART_dt"] = df["BIASSTART"].apply(self._to_datetime_gps)
            df["BIASEND_dt"] = df["BIASEND"].apply(self._to_datetime_gps)
            # blank or zero (0000:000:00000) is open-ended; any other unreadable date is reported
            unreadable = set()
            for raw_col, parsed_col in (("BIASSTART", "BIASSTART_dt"), ("BIASEND", "BIASEND_dt")):
                given = df[raw_col].astype(str).str.strip()
                bad = df[parsed_col].isna() & ~given.str.fullmatch(r"(0+:0+:0+|nan|)")
                unreadable.update(given[bad])
            if unreadable:
                warnings.warn(f"Bias file {path}: unreadable BIAS_START/BIAS_END {sorted(unreadable)[:5]} "
                              f"(YYYY:DOY:SSSSS expected); those records are taken as valid without limit",
                              UserWarning, stacklevel=2)

        df = df.rename(columns={'BIASSTART_dt': 'time', 'BIASEND_dt': 'time_end'})
        if "time" in df.columns:
            # validity of each record: a missing (or zero, 0000:000:00000) BIAS_END is open-ended
            df["time"] = pd.to_datetime(df["time"]).fillna(_BIAS_OPEN_START).astype("datetime64[ns]")
            ends = pd.to_datetime(df["time_end"]) if "time_end" in df.columns else pd.Series(pd.NaT, index=df.index)
            df["time_end"] = ends.fillna(_BIAS_OPEN_END).astype("datetime64[ns]")
        if "PRN" in df.columns:
            df["_BIAS_SYSTEM"] = df["PRN"].astype(str).str.strip().str[:1]
        else:
            df["_BIAS_SYSTEM"] = ""

        if "STATION" in df.columns:
            station = df["STATION"].replace("", np.nan)
            station_mask = station.notna()
            df["sv"] = df["PRN"]
            df["_STATION_NAME"] = station
        else:
            station_mask = pd.Series(False, index=df.index)
            df = df.rename(columns={'PRN': 'sv'})
            df["_STATION_NAME"] = np.nan

        # OSB records -> OSB_<OBS1>, DSB records -> BIAS_<OBS1>_<OBS2> (a file may hold both; other record
        # types, e.g. ISB, are not used)
        kind = df["BIAS"].astype(str).str.strip() if "BIAS" in df.columns else pd.Series(file_type, index=df.index)
        obs2 = df["OBS2"].astype(str).str.strip() if "OBS2" in df.columns else pd.Series("", index=df.index)
        df["_BIAS_COLUMN"] = np.where(kind.eq("DSB"), "BIAS_" + df["OBS1"].astype(str) + "_" + obs2,
                                      "OSB_" + df["OBS1"].astype(str))
        used = kind.isin(["OSB", "DSB"]).to_numpy()
        if "UNIT" in df.columns:
            # values are used in ns; a record in another unit (cyc for a phase bias) is left out
            unit = df["UNIT"].astype(str).str.strip().str.lower()
            other = used & ~unit.eq("ns").to_numpy()
            if other.any():
                warnings.warn(f"Bias file {path}: {int(other.sum())} records in {sorted(set(unit[other]))} "
                              f"instead of ns are not used", UserWarning, stacklevel=2)
                used &= ~other
        # ESTIMATED_SLOPE [ns/s for ns records] (Bias-SINEX 1.00, 5.1): with a slope the value refers to the
        # middle of BIAS_START..BIAS_END (to BIAS_START when the end is open, to BIAS_END when the start
        # is), applied by _bias_values_at from SLOPE_<column> (only when a record has a non-zero slope).
        # Only ns records are kept (above), so a slope is ns/s (the cyc/s of cyc phase records never enter)
        slope = pd.to_numeric(df["ESTIMATEDSLOPE"], errors="coerce") if "ESTIMATEDSLOPE" in df.columns else None
        with_slope = slope is not None and bool((slope[used].fillna(0.0) != 0.0).any())
        if with_slope:
            df["ESTIMATEDSLOPE"] = slope.fillna(0.0)
            unbounded = used & (slope.fillna(0.0) != 0.0).to_numpy()
            if "time" in df.columns:
                unbounded &= (df["time"].eq(_BIAS_OPEN_START) & df["time_end"].eq(_BIAS_OPEN_END)).to_numpy()
                reason = "records without BIAS_START and BIAS_END is not applied (no reference epoch)"
            else:
                reason = "records is not applied (parse_time=False: the validity times are not read)"
            if unbounded.any():
                warnings.warn(f"Bias file {path}: ESTIMATED_SLOPE of {int(unbounded.sum())} {reason}",
                              UserWarning, stacklevel=2)
                df.loc[unbounded, "ESTIMATEDSLOPE"] = 0.0
        df, station_mask = df[used], station_mask[used]

        def to_wide(records: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
            if records.empty:
                return pd.DataFrame({"time_end": pd.Series(dtype="datetime64[ns]")},
                                    index=pd.MultiIndex.from_arrays([[]] * len(keys), names=keys))
            wide_records = records.pivot_table(index=keys + ["time_end"], columns="_BIAS_COLUMN",
                                               values="ESTIMATEDVALUE", aggfunc="first")
            wide_records.columns.name = None
            if with_slope:
                slopes = records.pivot_table(index=keys + ["time_end"], columns="_BIAS_COLUMN",
                                             values="ESTIMATEDSLOPE", aggfunc="first")
                slopes = slopes.loc[:, (slopes.fillna(0.0) != 0.0).any()]
                slopes.columns = ["SLOPE_" + str(col) for col in slopes.columns]
                wide_records = wide_records.join(slopes)
            return wide_records.reset_index(level="time_end")

        wide = to_wide(df[~station_mask], ["sv", "time"])
        station_wide = None
        if station_mask.any():
            station_wide = to_wide(df[station_mask], ["_STATION_NAME", "_BIAS_SYSTEM", "time"])

        valid_systems = ('G', 'E', 'C')  # GPS, Galileo, BeiDou
        sv_index = wide.index.get_level_values('sv').astype(str)
        keep = sv_index.str.startswith(valid_systems)
        wide = wide[keep].copy()
        bias_availability: dict[str, set[str]] = {}
        bias_pair_availability: dict[str, set[str]] = {}
        placeholders: dict[str, list[str]] = {}
        sv_index = wide.index.get_level_values('sv').astype(str)
        for system in valid_systems:
            rows = sv_index.str.startswith(system)
            system_wide = wide[rows]
            if system_wide.empty:
                continue
            # a satellite code bias that is exactly 0.0 for every satellite of the system (two or more) is a
            # placeholder for a bias the product does not estimate (COD MGX: BeiDou C2I/C6I 0.0000 +- 0.0000),
            # not a value: absent, so that SPP, PPP, STEC and the orbit comparison all take their next source.
            # Phase biases are left as given (an all-zero phase OSB may be a product's datum convention)
            zero = [col for col in system_wide.columns if str(col).startswith(("OSB_C", "BIAS_C"))
                    and system_wide.loc[system_wide[col].notna()].index.get_level_values("sv").nunique() >= 2
                    and (system_wide[col].dropna() == 0.0).all()
                    and not (system_wide.get(f"SLOPE_{col}", pd.Series(dtype=float)).fillna(0.0) != 0.0).any()]
            if zero:
                wide.loc[rows, zero] = np.nan
                placeholders[system] = [col.split("_", 1)[1].replace("_", "-") for col in zero]
                system_wide = wide[rows]
            given = {col for col in system_wide.columns if system_wide[col].notna().any()}
            if any(col.startswith("OSB_") for col in given):
                bias_availability[system] = {col[4:] for col in given if col.startswith("OSB_")}
            if any(col.startswith("BIAS_") for col in given):
                bias_pair_availability[system] = {col[5:] for col in given if col.startswith("BIAS_")}
        if station_wide is not None:
            station_wide.index = station_wide.index.set_names(["station", "system", "time"])
            # NaN: observable not given for that station (as for satellites)
            wide.attrs["station_bias_by_system"] = station_wide

        out = wide
        out.attrs["bias_observables_by_system"] = {
            system: sorted(values) for system, values in bias_availability.items()
        }
        out.attrs["bias_pairs_by_system"] = {
            system: sorted(values) for system, values in bias_pair_availability.items()
        }
        out.attrs["bias_path"] = str(path)
        out.attrs["clock_reference_observables"] = clock_reference
        # observables / pairs given as 0.0 for every satellite of a system, taken as absent (see above); the
        # warning comes from the consumer of the system (_attach_dcb_to_observations, the orbit comparison)
        out.attrs["placeholder_biases_by_system"] = placeholders
        return out, file_type

    @staticmethod
    def _placeholder_bias_message(path, system: str, codes, next_source: str) -> str:
        """Warning text for the placeholder biases ``codes`` of ``system`` in the bias file ``path``."""
        name = {"G": "GPS", "E": "Galileo", "C": "BeiDou"}.get(system, system)
        return (f"Bias file {Path(str(path)).name}: {name} {', '.join(codes)} biases are 0.0 for every satellite "
                f"of the system (placeholders for biases the product does not estimate) and are taken as "
                f"absent; {next_source}")

    def process_observations2(self):
        (sta_name, ant_type, ant_cover, ant_h, approx_pos, pos_geod, interval, gps_obs, gal_obs,
         time_of_first_obs, time_of_last_obs, phase_shift_dict, num_obs_dict) = self.obs_header_reader()

        info = (sta_name, ant_type, ant_cover, ant_h, approx_pos, pos_geod, interval, gps_obs, gal_obs,
                time_of_first_obs, time_of_last_obs, phase_shift_dict)

        return info

    @staticmethod
    def _canonical_bias_sv(sv: object) -> str:
        """Return the PRN used by product bias files, dropping adapter suffixes."""
        return str(sv).split("_", 1)[0]

    @staticmethod
    def _infer_observation_system(df: pd.DataFrame) -> str | None:
        if not isinstance(df.index, pd.MultiIndex) or "sv" not in df.index.names:
            return None
        for sv in df.index.get_level_values("sv").astype(str):
            system = GNSSDataProcessor2._canonical_bias_sv(sv)[:1]
            if system in {"G", "E", "C"}:
                return system
        return None

    @staticmethod
    def _bias_values_at(keys: np.ndarray, times: np.ndarray, records: pd.DataFrame, key: str
                        ) -> dict[str, np.ndarray]:
        """Bias columns of ``records`` at the rows (``keys``, ``times``): the value of the record of the row's
        key (``records[key]``) valid at its epoch, NaN where none is.

        A record is valid from ``time`` (BIAS_START) to ``time_end`` (BIAS_END) plus its 1 s resolution, both
        inclusive (GRG ends a daily record at 23:59:59, CODE and CAS at 24:00: both hold at the 24:00 epoch,
        where the record of the next day, when given, applies); where several records of a key are valid at an
        epoch, the one that starts last applies. Columns with the same validity records are joined together (one join for a daily
        product). A column with ESTIMATED_SLOPE (``SLOPE_<column>`` [ns/s], ``read_bia``) gives
        value + slope * (t - t_ref), t_ref the middle of the record's BIAS_START..BIAS_END (its start when the
        end is open, its end when the start is: Bias-SINEX 1.00, 5.1).
        """
        cols = [c for c in records.columns if isinstance(c, str) and c.startswith(("OSB_", "BIAS_"))]
        if not cols or len(keys) == 0 or records.empty:
            return {}
        rec = records.rename(columns={key: "_key"})
        rec["_key"] = rec["_key"].astype(str)
        rec["time"] = pd.to_datetime(rec["time"]).astype("datetime64[ns]")
        end = rec["time_end"] if "time_end" in rec.columns else pd.Series(_BIAS_OPEN_END, index=rec.index)
        end = pd.to_datetime(end).fillna(_BIAS_OPEN_END).astype("datetime64[ns]")
        sloped = {c: f"SLOPE_{c}" for c in cols if f"SLOPE_{c}" in rec.columns}
        if sloped:
            start_open, end_open = rec["time"].eq(_BIAS_OPEN_START), end.eq(_BIAS_OPEN_END)
            middle = rec["time"] + (end - rec["time"]) / 2
            rec["_slope_ref"] = middle.where(~end_open, rec["time"]).where(~start_open | end_open, end)
        rec["time_end"] = (end + _BIAS_END_RESOLUTION).astype("datetime64[ns]")
        left = pd.DataFrame({"_key": np.asarray(keys, dtype=str), "time": np.asarray(times, dtype="datetime64[ns]"),
                             "_row": np.arange(len(keys))}).sort_values("time", kind="stable")
        interval, _ = pd.MultiIndex.from_arrays([rec["time"], rec["time_end"]]).factorize()
        given = rec[cols].notna().to_numpy()
        groups: dict[bytes, list[int]] = {}
        for j in range(len(cols)):
            groups.setdefault(np.unique(interval[given[:, j]]).tobytes(), []).append(j)
        out: dict[str, np.ndarray] = {}
        for members in groups.values():
            group = [cols[j] for j in members]
            rows = given[:, members].any(axis=1)
            if not rows.any():
                continue
            slopes = [sloped[col] for col in group if col in sloped]
            extra = [*slopes, "_slope_ref"] if slopes else []
            right = rec.loc[rows, ["_key", "time", "time_end", *group, *extra]].sort_values("time", kind="stable")
            merged = pd.merge_asof(left, right, on="time", by="_key", direction="backward")
            valid = (merged["time"] <= merged["time_end"]).to_numpy()
            position = merged["_row"].to_numpy()

            def value_at(frame: pd.DataFrame, col: str, at: np.ndarray) -> np.ndarray:
                """Record value of ``col`` at the epochs ``at`` (with its slope, where it has one)."""
                value = frame[col].to_numpy(dtype=float)
                if col in sloped:
                    dt = (at - frame["_slope_ref"].to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "s")
                    value = value + np.nan_to_num(frame[sloped[col]].to_numpy(dtype=float)) * dt
                return value

            epochs = merged["time"].to_numpy(dtype="datetime64[ns]")
            found = {col: np.where(valid, value_at(merged, col, epochs), np.nan) for col in group}
            # overlapping records of a key: where the last-starting record has expired, an earlier one that
            # still covers the epoch applies (the latest-starting of those)
            by_key = right.sort_values(["_key", "time"], kind="stable")
            earlier_end = by_key.groupby("_key")["time_end"].cummax().groupby(by_key["_key"]).shift()
            for overlap_key in by_key.loc[by_key["time"] <= earlier_end, "_key"].unique():
                retry = np.flatnonzero(~valid & (merged["_key"] == overlap_key).to_numpy())
                if not retry.size:
                    continue
                candidates = by_key[by_key["_key"] == overlap_key]
                t = merged["time"].to_numpy()[retry][:, None]
                covers = (candidates["time"].to_numpy()[None, :] <= t) & (t <= candidates["time_end"].to_numpy()[None, :])
                last = covers.shape[1] - 1 - np.argmax(covers[:, ::-1], axis=1)
                has = covers.any(axis=1)
                for col in group:
                    chosen = candidates.iloc[last[has]]
                    found[col][retry[has]] = value_at(chosen, col, epochs[retry[has]])
            for col in group:
                values = np.full(len(keys), np.nan)
                values[position] = found[col]
                out[col] = values
        return out

    def _rinex2_header_tracking(self):
        """What the RINEX 2 observation header says about its tracking modes (``biases.Rinex2HeaderTracking``):
        the converter's COMMENT block and REC # / TYPE, from the raw header lines (also Hatanaka, gzip),
        else from the georinex header. Read once per observation file."""
        cached = self.__dict__.get("_rinex2_tracking_info")
        if cached is None or cached[0] != self.obs_path:
            from ..models.biases import read_rinex_header_lines, rinex2_header_tracking

            lines = read_rinex_header_lines(self.obs_path)
            if lines:
                info = rinex2_header_tracking(lines)
            else:
                try:
                    header = self._get_obs_header()
                except Exception:  # noqa: BLE001 - no header: the receiver table and the default decide
                    header = {}
                info = rinex2_header_tracking(header=header)
            cached = self._rinex2_tracking_info = (self.obs_path, info)
        return cached[1]

    def _rinex2_bias_aliases(self, system: str, report: dict, available: set, station_codes=(),
                             station_source: str | None = None):
        """Product code -> RINEX 2 observable of the frame (``C1W`` -> ``C1`` for C1 read from P1).

        The RINEX 3 code of each code and phase observable of the frame is resolved once per file
        (``biases.resolve_rinex2_tracking``: the ``rinex2_tracking`` setting, the fixed codes, the converter's
        header block, the station biases ``station_codes`` of ``station_source``, the receiver type, the
        default pilot tracking); a phase without its own entry takes the attribute of its band's code. A
        product without that code gives another attribute of the band (``biases.rinex2_fallback_codes``).
        Returns (alias, [(observable, RINEX 3 code, source)], fallbacks, missing) with the texts of the
        warnings.
        """
        from ..models.biases import resolve_rinex2_tracking, rinex2_fallback_codes

        column_map = report.get("column_map", {}) or {}
        frame = {str(target): str(entry.get("source")) for target, entry in column_map.items()
                 if (entry or {}).get("source") and str(target)[:1] in ("C", "L")}
        resolved = resolve_rinex2_tracking(system, sorted(set(frame.values())), info=self._rinex2_header_tracking(),
                                           override=(self.rinex2_tracking or {}), station_codes=station_codes,
                                           station_source=station_source)
        unused = sorted(set((self.rinex2_tracking or {}).get(system, {})) - set(frame.values()))
        notes_given = self.__dict__.setdefault("_bias_notes", set())
        if unused and ("rinex2_tracking_unused", system) not in notes_given:
            notes_given.add(("rinex2_tracking_unused", system))
            warnings.warn(
                f"rinex2_tracking[{system!r}] {unused} not used: the {system} observations read from "
                f"{self.obs_path} for this run have {sorted(set(frame.values()))}", UserWarning, stacklevel=4)
        # phase biases are named only when the product gives phase biases of the system
        phases = any(code.startswith("L") for code in available)
        alias: dict[str, str] = {}
        chosen: list[tuple[str, str, str]] = []
        fallbacks: list[str] = []
        missing: list[str] = []
        for name in sorted(frame, key=lambda target: (target[1:], target[0])):
            source = frame[name]
            code, how = resolved.get(source, (None, None))
            if code is None and name.startswith("L"):
                code_source = frame.get("C" + name[1:])
                base = resolved.get(code_source) if code_source else None
                if base is not None:
                    code, how = "L" + base[0][1:], f"attribute of {code_source}"
            if code is None:
                continue
            label = name if name == source else f"{name} ({source})"
            if name.startswith("C") or phases:
                chosen.append((label, code, how))
            others = rinex2_fallback_codes(system, source, code)
            product_code = code if code in available else next((alt for alt in others if alt in available), None)
            if product_code is None:
                if name.startswith("C") or phases:
                    missing.append(f"{label}: none of {'/'.join((code, *others))} in the product")
                continue
            if product_code != code and name.startswith("C"):
                # (the phases of one band are aligned: another attribute gives the same bias, no warning)
                fallbacks.append(f"{label} {code} -> {product_code}")
            alias[product_code] = name
        return alias, chosen, fallbacks, missing

    @staticmethod
    def _alias_bias_column(column: str, alias: dict[str, str]) -> str | None:
        """``OSB_<code>`` / ``BIAS_<a>_<b>`` with the codes of ``alias`` replaced (None when none is)."""
        for prefix in ("STA_OSB_", "STA_BIAS_", "OSB_", "BIAS_"):
            if column.startswith(prefix):
                codes = column[len(prefix):].split("_")
                renamed = [alias.get(code, code) for code in codes]
                return prefix + "_".join(renamed) if renamed != codes else None
        return None

    def _attach_dcb_to_observations(self, df: pd.DataFrame | None, dcb: pd.DataFrame | None) -> pd.DataFrame | None:
        """Attach the biases of the bias products (``dcb_path``) to the observation rows of one system [ns].

        Satellite biases ``OSB_<obs>`` / ``BIAS_<obs1>_<obs2>`` (DSB) and receiver biases of ``station_name``
        ``STA_OSB_<obs>`` / ``STA_BIAS_<obs1>_<obs2>``, matching suffixed observation SV ids to product PRNs.

        * Validity: a row takes the records of its satellite valid at its epoch (BIAS_START <= t <= BIAS_END
          of each record; where several are, the one that starts last). The sampling token of the file name
          (``30S``, ``01D``, ``07D``) plays no role. One warning when no product covers some epochs.
        * Several products: the OSBs of a row all come from the first product of ``dcb_path`` that has an
          OSB of that satellite at that epoch, the DSBs likewise (and the receiver ``STA_OSB_`` /
          ``STA_BIAS_`` of the station); values of different products are not combined for one satellite
          (OSBs of different analysis centres refer to different clock datums), with one warning naming
          what a later product would have given. Adjacent-day products thus cover each their own day.
        * Missing values: a column is added only when a product gives it for some row of the frame; NaN in a
          column means no product has a valid record of that observable for that satellite at that epoch
          (satellite missing from the product, epoch outside its records). A missing bias is never 0.0 (0.0 is
          a product value); consumers decide what it means (next source, warning, missing-bias policy).
          ``attrs["bias_clock_reference_observables"]``: per product path whose OSBs were attached, the
          SATELLITE_CLOCK_REFERENCE_OBSERVABLES it declares for the system (their ionosphere-free combination
          has no satellite bias in that product's clocks).
        * Receiver biases: the records of the frame's system for ``station_name``, matched as a 9-character id
          or a 4-character site code, case-insensitive (exact match first, ``biases.match_station``). A station
          the product does not list gets none (no message; consumers that need it report it), except one
          warning when the product lists another station id of the same site (``BRUX01BEL`` for
          ``BRUX00BEL``).
        * RINEX 2 frames (RINEX 2 adapter): the RINEX 2 codes carry no tracking mode, so each code and phase
          observable takes the biases of one RINEX 3 code resolved once per file (``P1`` -> ``C1W``, ``C5`` ->
          ``C5Q`` from the gfzrnx conversion block ...: ``biases.resolve_rinex2_tracking``, the
          ``rinex2_tracking`` setting first), attached also under the frame's names (``OSB_C1`` = ``OSB_C1W``;
          DSB pairs with the codes replaced, ``BIAS_C1C_C1``), with one warning per system naming each code
          and its source (and one for a product that gives only another attribute of the band).
        """
        if not isinstance(df, pd.DataFrame) or not isinstance(dcb, pd.DataFrame):
            return df
        report = df.attrs.get("gnx_adapter", {}) or {}
        rinex2 = report.get("source_format") == "RINEX2"
        if (not rinex2 and self.rinex2_tracking and ("rinex2_tracking_rinex3",) not in self.__dict__.setdefault(
                "_bias_notes", set()) and not self._is_rinex2_obs()):
            self._bias_notes.add(("rinex2_tracking_rinex3",))
            warnings.warn(f"rinex2_tracking is not used: the observations ({self.obs_path}) are not RINEX 2 "
                          f"(their RINEX 3 codes name the tracking mode)", UserWarning, stacklevel=3)
        bias_ok = report.get("bias_eligibility", {}).get("attach_bia_in_load_obs_data", True)
        if not bias_ok and not rinex2:
            return df
        from ..models.biases import match_station

        notes_given = self.__dict__.setdefault("_bias_notes", set())
        products = dcb.attrs.get("bias_products") or [dcb]
        left = df.reset_index()
        keys = left["sv"].map(self._canonical_bias_sv).astype(str).to_numpy()
        times = pd.to_datetime(left["time"]).to_numpy(dtype="datetime64[ns]")
        system = self._infer_observation_system(df)

        # biases of each product at the rows; spans of the products' records of the system
        joined_by_product: list[dict[str, np.ndarray]] = []
        covered, spans = np.zeros(len(left), dtype=bool), []
        for product in products:
            # placeholder biases of this system (read_bia: 0.0 for every satellite, taken as absent), once
            codes = (product.attrs.get("placeholder_biases_by_system") or {}).get(system or "?")
            key = ("placeholder", product.attrs.get("bias_path"), system)
            if codes and key not in notes_given:
                notes_given.add(key)
                warnings.warn(self._placeholder_bias_message(
                    product.attrs.get("bias_path", ""), system, codes,
                    "the next source applies (another product in dcb_path, the broadcast group delay)"),
                    UserWarning, stacklevel=3)
            records = product.reset_index()
            joined = self._bias_values_at(keys, times, records, "sv")
            records = records[records["sv"].astype(str).str.startswith(system or "?")]
            if len(records):
                start, end = records["time"].min(), records.get("time_end", records["time"]).max()
                covered |= (times >= np.datetime64(start)) & (times <= np.datetime64(end + _BIAS_END_RESOLUTION))
                spans.append(f"{Path(str(product.attrs.get('bias_path', ''))).name} {start} to {end}")
            station_bias = product.attrs.get("station_bias_by_system")
            if self.station_name and system is not None and isinstance(station_bias, pd.DataFrame):
                station_bias = station_bias.reset_index()
                station_bias = station_bias[station_bias["system"].astype(str).eq(system)]
                listed = station_bias["station"].astype(str).unique()
                names = match_station(self.station_name, listed)
                if names:
                    station_bias = station_bias[station_bias["station"].astype(str).eq(names[0])]
                    station_joined = self._bias_values_at(np.full(len(left), "station"), times,
                                                          station_bias.assign(_station="station"), "_station")
                    joined.update({"STA_" + col: column_values for col, column_values in station_joined.items()})
                else:
                    # another station id of the same site (BRUX01BEL for BRUX00BEL): likely a naming mismatch
                    same_site = sorted({name for name in listed
                                        if name.strip().upper()[:4] == str(self.station_name).strip().upper()[:4]})
                    key = ("station", product.attrs.get("bias_path"), system)
                    if same_site and key not in notes_given:
                        notes_given.add(key)
                        warnings.warn(
                            f"bias product {product.attrs.get('bias_path', '')} has {system} receiver biases for "
                            f"{same_site} but not for station {self.station_name!r} (a different 9-character id "
                            f"of the site): no receiver bias of the product is applied", UserWarning, stacklevel=3)
            joined_by_product.append(joined)
        if spans and not covered.all() and ("validity", system) not in notes_given:
            notes_given.add(("validity", system))
            warnings.warn(
                f"bias products ({'; '.join(spans)}): {int((~covered).sum())} of {covered.size} {system} "
                f"observations lie outside the validity of every product's {system} records (their biases are "
                f"missing)", UserWarning, stacklevel=3)

        # OSBs (DSBs, receiver OSBs, receiver DSBs) of a row from one product: the first that has one there
        def family(col: str) -> str:
            return col.split("_", 2)[1] if col.startswith("STA_") else "SAT_" + col.split("_", 1)[0]

        values: dict[str, np.ndarray] = {}
        not_combined: dict[str, set] = {}
        used_products: dict[str, set] = {}
        for fam in sorted({family(col) for joined in joined_by_product for col in joined}):
            chosen = np.full(len(left), -1)
            for i, joined in enumerate(joined_by_product):
                cols = [col for col in joined if family(col) == fam]
                if cols:
                    has = np.any([np.isfinite(joined[col]) for col in cols], axis=0)
                    chosen[(chosen < 0) & has] = i
            used_products[fam] = set(chosen[chosen >= 0].tolist())
            for col in sorted({col for joined in joined_by_product for col in joined if family(col) == fam}):
                column_values = np.full(len(left), np.nan)
                for i, joined in enumerate(joined_by_product):
                    if col in joined:
                        column_values[chosen == i] = joined[col][chosen == i]
                for i, joined in enumerate(joined_by_product):
                    if col in joined:
                        lost = np.isnan(column_values) & np.isfinite(joined[col]) & (chosen >= 0)
                        if lost.any():
                            names = sorted(set(keys[lost])) if not col.startswith("STA_") else [self.station_name]
                            not_combined.setdefault(col, set()).update(map(str, names))
                if np.isfinite(column_values).any():
                    values[col] = column_values
        if not_combined and ("not_combined", system) not in notes_given:
            notes_given.add(("not_combined", system))
            listed = "; ".join(f"{col[col.index('_') + 1:] if not col.startswith('STA_') else col}: "
                               f"{', '.join(sorted(svs)[:12])}{' ...' if len(svs) > 12 else ''}"
                               for col, svs in sorted(not_combined.items()))
            warnings.warn(
                f"bias products {[Path(str(p.attrs.get('bias_path', ''))).name for p in products]}: the {system} "
                f"biases of a satellite are taken from one product, the first that has it (biases of different "
                f"analysis centres refer to different clock datums); a later product gives what that product "
                f"lacks, left missing: {listed}",
                UserWarning, stacklevel=3)
        references = {}
        for i in sorted(used_products.get("SAT_OSB", ())):
            reference = (products[i].attrs.get("clock_reference_observables") or {}).get(system)
            if reference:
                references[str(products[i].attrs.get("bias_path", ""))] = list(reference)

        if rinex2 and system is not None and values:
            given = [col.split("_", 1)[1] for col in values if not col.startswith("STA_")]
            available = {code for col in given for code in col.split("_")}
            # receiver biases of the station: evidence of its tracking modes (one attribute per band)
            station_codes = {code for col in values if col.startswith("STA_")
                             for code in col.split("_", 2)[2].split("_")}
            station_source = ", ".join(
                Path(str(products[i].attrs.get("bias_path", ""))).name
                for i, joined in enumerate(joined_by_product) if any(col.startswith("STA_") for col in joined))
            alias, chosen, fallbacks, missing = self._rinex2_bias_aliases(
                system, report, available, station_codes, f"{station_source} ({self.station_name})")
            for col in list(values):
                renamed = self._alias_bias_column(col, alias)
                if renamed is not None and renamed not in values:
                    values[renamed] = values[col]
            key = ("rinex2", system)
            if (chosen or missing) and key not in notes_given:
                notes_given.add(key)
                by_source: dict[str, list[str]] = {}
                for label, code, how in chosen:
                    by_source.setdefault(how, []).append(f"{label} as {code}")
                taken = "; ".join(f"{', '.join(items)} ({how})" for how, items in by_source.items())
                warnings.warn(
                    f"RINEX 2 observation codes carry no tracking mode ({self.obs_path}): the {system} biases of the "
                    f"bias product are taken for {taken}{''.join('; ' + text for text in missing)}; set "
                    f"rinex2_tracking to override (gnx_py.biases.resolve_rinex2_tracking)",
                    UserWarning, stacklevel=3)
                if fallbacks:
                    warnings.warn(
                        f"RINEX 2 observations ({self.obs_path}): the bias product lacks the {system} code resolved "
                        f"for an observable and gives the bias of another attribute of the same band instead: "
                        f"{', '.join(fallbacks)} (the biases of the tracking modes of a band differ by typ. "
                        f"0.1-0.5 ns, up to ~1.6 ns = 0.5 m, amplified 1.26x in IF(L1,L5) and 11.3x in "
                        f"IF(L2,L5))", UserWarning, stacklevel=3)
                info = self._rinex2_header_tracking()
                if info.issues and ("rinex2_block",) not in notes_given:
                    notes_given.add(("rinex2_block",))
                    warnings.warn(
                        f"RINEX 2 observations ({self.obs_path}): {info.converter or 'conversion block'} entries "
                        f"not used: {'; '.join(info.issues)}", UserWarning, stacklevel=3)

        out = left.drop(columns=[col for col in values if col in left.columns])
        out = pd.concat([out, pd.DataFrame(values, index=out.index)], axis=1)
        out.attrs = dict(df.attrs)
        if references:
            out.attrs["bias_clock_reference_observables"] = references
        return out.set_index(["sv", "time"])

    def _read_observations(self, tlim=None):
        """``rinex_obs.read_obs`` of ``obs_path`` for the processor's systems within ``tlim`` (code, P code, phase
        and SNR; Doppler is never selected). The header, the interval and the first/last epoch of the read are
        kept with the path for :meth:`obs_header_reader` (the frames are not)."""
        read = _rinex_obs.read_obs(self.obs_path, systems="".join(sorted(self.sys)), tlim=tlim,
                                   kinds=("C", "P", "L", "S"))
        self._obs_read = (self.obs_path, _dataclass_replace(read, frames={}, lli={}))
        if self.__dict__.get("_obs_header_cache", (None,))[0] != self.obs_path:
            self._obs_header_cache = (self.obs_path, read.header)
        return read

    def _warn_missing_system(self, system: str, read, tlim) -> None:
        """One UserWarning for a requested system without observations, naming the file and its systems."""
        header = read.header
        name = _SYSTEM_NAME.get(system, system)
        listed = ", ".join(sorted(header.obs_types)) or "none"
        if system not in header.obs_types:
            msg = (f"{self.obs_path} has no {name} observations (systems with observation types in the header: "
                   f"{listed}); {name} is left out")
        elif tlim is not None:
            msg = (f"{self.obs_path}: no {name} observation within the time limit {tlim[0]} .. {tlim[1]} "
                   f"(systems with observation types in the header: {listed}); {name} is left out")
        else:
            msg = (f"{self.obs_path}: the header lists {name} observation types but the file holds no {name} "
                   f"observation; {name} is left out")
        warnings.warn(msg, UserWarning, stacklevel=3)

    def load_and_filter(self, tlim=None, version=3.0):
        """
        Observations of the processor's systems, with the observation types of their modes.

        Every file (RINEX 2.10/2.11, 3.0x, 4.0x; plain, gzip, .Z, bzip2, zip, CRINEX 1/3) is read by
        :mod:`gnx_py.rinex_obs` in memory, with nothing written next to the data (``use_gfz``
        has no effect): ``tlim`` ``[start, end]`` (inclusive, GPST) is always applied, BDT time tags are moved
        to GPST, blank and 0.0 values are missing, and the records of epoch flags 2-6 are skipped. A RINEX 2
        frame goes through the band adapter (``rinex2_adapter.adapt_rinex2_obs``). Each system keeps the
        observation types its mode selects (``select_obs_types_*``; BeiDou strict per signal) and the rows
        where every selected code and phase has a value.

        Returns {'G', 'E', 'C': frame or None}; frames have the index ['sv', 'time'] lexsorted. A
        requested system without observations in the file (or within ``tlim``) is None, with one UserWarning
        naming the file and the systems it holds. BeiDou cannot be read from RINEX 2 (ValueError).
        ``version`` is not used (kept for compatibility).
        """
        results = {"G": None, "E": None, "C": None}
        if tlim is not None:
            tlim = _check_time_limit(tlim)
        read = self._read_observations(tlim)
        header = read.header
        obs_rinex2 = is_supported_rinex2_version(header.version)
        if obs_rinex2:
            unmapped = sorted({sys_char for sys_char in self.sys} - RINEX2_OBS_SYSTEMS)
            if unmapped:
                raise ValueError(
                    f"{self.obs_path} (RINEX {header.version:.2f}, systems {header.system!r}): "
                    + "; ".join(_rinex2.unmapped_system_message(sys_char) for sys_char in unmapped))
        obs_info = self._get_obs_rinex_info()
        obs_header = self._get_obs_header() if obs_rinex2 else None

        # --- Helper: fast filtering without dropna double-copy ---
        def _filter_df(df, obs_types, *, fast_mask=True):
            """
            obs_types: list[str] columns to keep.
            fast_mask:
              - if True: compute mask on minimal critical columns (codes+phases only)
              - else: compute mask on all obs_types
            """
            if df is None or not obs_types:
                return None

            # keep only existing columns (robust to missing in some RINEX)
            obs_types = [c for c in obs_types if c in df.columns]
            if not obs_types:
                return None

            sub = df.loc[:, obs_types]

            # Decide which columns define "valid row".
            # You were doing dropna(how="any") => all selected columns must be non-NaN.
            # Here we reproduce that, but optionally compute the mask on a smaller set
            # to reduce cost (then apply mask to full sub).
            if fast_mask:
                # Minimal set: keep only Code + Phase columns among selected (C* and L*)
                crit = [c for c in obs_types if c and (c[0] == "C" or c[0] == "L")]
                # If somehow nothing matches, fall back to all columns
                mask_cols = crit if crit else obs_types
            else:
                mask_cols = obs_types

            # one mask, then one slicing (usually faster + fewer copies than dropna)
            mask = sub.loc[:, mask_cols].notna().all(axis=1)
            # If mask is all False (weird files), return empty df with right columns
            filtered = sub.loc[mask]
            filtered.attrs = dict(df.attrs)
            return filtered

        def _adapt(df, sys_char: str, mode: str):
            """Route a RINEX 2 frame through the band adapter; pass RINEX 3 through."""
            if not obs_rinex2 or df is None or df.empty:
                return df
            return adapt_rinex2_obs(
                df,
                header=obs_header,
                rinex_info=obs_info,
                system=sys_char,
                mode=mode,
                strict=True,
                bias_policy="safe",
                obs_path=self.obs_path,
            )

        modes = {"G": self.mode, "E": self.galileo_modes, "C": self.beidou_modes}
        selectors = {
            "G": lambda frame: self.select_obs_types_gps(frame),
            "E": lambda frame: self.select_obs_types_galileo(frame, self.galileo_modes),
            "C": lambda frame: self.select_obs_types_beidou(frame, self.beidou_modes),
        }
        for sys_char in ("G", "E", "C"):
            if sys_char not in self.sys:
                continue
            frame = read.frames.get(sys_char)
            if frame is None:
                self._warn_missing_system(sys_char, read, tlim)
                continue
            if obs_rinex2 and sys_char == "G":
                self._warn_rinex2_wavelength_factors(header, modes[sys_char])
            frame = _adapt(frame, sys_char, modes[sys_char])
            selected = selectors[sys_char](frame)
            filtered = _filter_df(frame, selected, fast_mask=True)
            if filtered is not None:
                # contract C4 (bug-fix stage 13b): the SYS / PHASE SHIFT records of the reader on every frame
                # (read-only, JSON-safe; rinex_obs.phase_shift_of / phase_shift_events_of give the contract form);
                # empty for RINEX 2 (no such record; the band adapter builds new attrs)
                for key, value in _rinex_obs._phase_shift_attrs().items():
                    filtered.attrs.setdefault(key, value)
            if filtered is None or filtered.empty:
                name = _SYSTEM_NAME.get(sys_char, sys_char)
                warnings.warn(
                    f"{self.obs_path}: no {name} epoch has every code and phase of mode {modes[sys_char]!r} "
                    f"(observables selected: {', '.join(selected) if selected else 'none'}; {len(frame)} rows "
                    f"read, 0 left); {name} is left out", UserWarning, stacklevel=2)
                filtered = None
            results[sys_char] = filtered

        return results

    def _warn_rinex2_wavelength_factors(self, header, mode: str) -> None:
        """One UserWarning per file for a RINEX 2 WAVELENGTH FACT L1/2 record (RINEX 2.11 Table A1, GPS only) whose
        factor is not 1 on a band of the GPS ``mode``: 2 = half-cycle ambiguities (squaring receivers), 0 on L2 =
        single-frequency receiver. GNX takes RINEX 2 phase as full-cycle and does not correct it."""
        try:
            bands = {signal_spec(signal).phase_prefix[1:] for signal in mode_signals(mode)}
        except ValueError:
            return
        found = []
        for payload in header.raw.get("WAVELENGTH FACT L1/2", []):
            payload = payload.ljust(60)
            try:
                factors = {"1": int(payload[0:6]), "2": int(payload[6:12] or 0)}
                n_sat = int(payload[12:18] or 0)
            except ValueError:
                continue
            prns = [payload[18 + 6 * k + 3:18 + 6 * k + 6].strip() for k in range(min(n_sat, 7))]
            scope = "all satellites" if n_sat == 0 else ", ".join(p for p in prns if p) or "listed satellites"
            for band in sorted(bands & set(factors)):
                if factors[band] != 1:
                    meaning = ("half-cycle ambiguities" if factors[band] == 2 else
                               "single-frequency receiver" if factors[band] == 0 else "undefined factor")
                    found.append(f"L{band} factor {factors[band]} ({meaning}) for {scope}")
        notes = self.__dict__.setdefault("_obs_notes", set())
        if found and ("wavelength", self.obs_path) not in notes:
            notes.add(("wavelength", self.obs_path))
            warnings.warn(f"RINEX 2 observations {self.obs_path}: WAVELENGTH FACT L1/2 {'; '.join(found)}: the GPS "
                          f"phase is taken as full-cycle (factor 1) and not corrected; half-cycle slips are not "
                          f"repaired", UserWarning, stacklevel=3)

    def select_obs_types_gps(self, dataframe, *, consistent_suffix: bool = True) -> List[str]:
        """
        GPS observation types of the mode: per kind (code C1, phase L1, SNR S1, ... of the mode's bands) one
        column with data, the canonical two-character slot of a RINEX 2 frame first, then a tracking attribute
        already chosen for another kind, then the priority W, C, P, Y, L, X, Q (so a Septentrio file gives the
        semi-codeless P(Y) code C1W, and the phase L1C when the file has no L1W).

        ``gps_sf_same_tracking``: in a single-band mode (L1, L2, L5) the code
        and the phase of the band are taken as a pair of one tracking attribute, the first attribute of the
        priority with both a code and a phase holding data (NYA2 / BRUX / NKLG: C1C + L1C instead of C1W + L1C;
        same-tracking code and carrier, RINEX 3.05 Table 4 attributes). A band without such a pair, a RINEX 2
        frame (two-character slots) and dual-frequency modes keep the rule above. The SNR then follows the pair's
        attribute. Measured (GRAPHIC / GIM-constrained L1, 6 h): NYA2 2024-035 0.2612 -> 0.2169 m, better in
        15 of 18 static station-days (the 3 others by at most 1.6 mm); the code group delay is the OSB of the
        chosen code (C1C) or, with broadcast clocks, c TGD with ISC_L1CA.
        """

        cols = list(map(str, dataframe.columns))

        # --- 1) Candidate columns by mode: the bands of the mode's signals (Q196: a single-frequency
        # mode takes a second band only with SINGLE_FREQUENCY_NEEDS_SECOND_BAND) ---
        two_bands = SINGLE_FREQUENCY_NEEDS_SECOND_BAND
        if self.mode == "L1L2" or (two_bands and self.mode in ("L1", "L2")):
            wanted_freqs = ("1", "2")
        elif self.mode == "L1L5" or (two_bands and self.mode == "L5"):
            wanted_freqs = ("1", "5")
        elif self.mode == "L2L5":
            wanted_freqs = ("2", "5")
        elif self.mode in ("L1", "L2", "L5"):
            wanted_freqs = (self.mode[1],)
        else:
            return []

        # keep same "exclude Doppler + 'li' suffix" policy
        selected_columns = [
            c for c in cols
            if any(f in c for f in wanted_freqs)
               and not (c.startswith("D") or c.endswith("li"))
               and len(c) >= 2
        ]
        if not selected_columns:
            return []

        # --- 2) Precompute non-NaN counts ONCE ---
        # This is the big win: avoid dataframe[candidates].notna().sum() inside loops.
        nn = dataframe[selected_columns].notna().sum(axis=0)  # Series: col -> count

        # --- 3) Group by kind = col[:-1] ---
        by_kind = defaultdict(list)
        for c in selected_columns:
            kind = c if len(c) == 2 else c[:-1]
            by_kind[kind].append(c)

        # Determine kind ordering (same as your freq_key idea)
        def freq_key(kind: str) -> int:
            m = re.search(r"(\d+)$", kind)
            return int(m.group(1)) if m else math.inf

        kinds = sorted(by_kind.keys(), key=freq_key)

        priority_order = ["W", "C", "P", "Y", "L", "X", "Q"]
        selected_types: List[str] = []
        chosen_suffixes: List[str] = []

        # NQ-PPP-28 (bug-fix stage 13b): code and phase of one tracking attribute in a single-band mode
        paired: dict = {}
        if self.gps_sf_same_tracking and self.mode in ("L1", "L2", "L5"):
            band = self.mode[1]
            code_kind, phase_kind = f"C{band}", f"L{band}"
            with_data = {kind: {c[-1] for c in by_kind.get(kind, []) if len(c) == 3 and int(nn[c]) > 0}
                         for kind in (code_kind, phase_kind)}
            canonical = any(kind in by_kind.get(kind, ()) for kind in (code_kind, phase_kind))  # RINEX 2 C1 / L1
            if not canonical:
                for suf in priority_order:
                    if suf in with_data[code_kind] and suf in with_data[phase_kind]:
                        paired = {code_kind: code_kind + suf, phase_kind: phase_kind + suf}
                        chosen_suffixes.append(suf)
                        break

        # --- 4) Choose best column per kind (only columns with data: a header-listed column that stays
        # empty for the file or the tlim window would drop every row) ---
        for kind in kinds:
            kind_cols = [c for c in by_kind[kind] if int(nn[c]) > 0]
            if not kind_cols:
                continue

            chosen_col = paired.get(kind)

            # Canonical internal slots like C1/L1 should win over suffixed variants.
            if chosen_col is None and kind in kind_cols:
                chosen_col = kind

            if chosen_col is None and consistent_suffix and chosen_suffixes:
                # Try already-chosen suffixes first, pick the candidate with max nn in that suffix.
                for suf in chosen_suffixes:
                    candidates = [c for c in kind_cols if c.endswith(suf)]
                    if candidates:
                        # choose candidate with max non-NaN count (tie: lexicographic)
                        best_count = nn[candidates].max()
                        best = sorted([c for c in candidates if nn[c] == best_count])
                        chosen_col = best[0]
                        break

            if chosen_col is None:
                # Fall back to base priority order of suffixes
                for suf in priority_order:
                    candidates = [c for c in kind_cols if c.endswith(suf)]
                    if candidates:
                        # mimic your deterministic pick: smallest lexicographically
                        chosen_col = sorted(candidates)[0]
                        break

            if chosen_col:
                selected_types.append(chosen_col)
                suf = "" if len(chosen_col) == 2 else chosen_col[-1]
                if suf and suf not in chosen_suffixes:
                    chosen_suffixes.append(suf)

        self._require_mode_observables("G", self.mode, selected_types, dataframe)
        return selected_types

    def _require_mode_observables(self, system: str, mode: str, selected: List[str], dataframe) -> None:
        """ValueError when a signal of the GPS or Galileo ``mode`` has no code or no phase observable with data
        among ``selected`` (the phase only when the frame has phase observables with data: a code-only file is
        read for SPP) (before v1.0.0 a dual-frequency mode without one of its bands came back as a
        single-band frame), naming the mode, the observables looked for and the band's observables of the file;
        the rule of the BeiDou selection. An unknown mode is left to the caller (nothing is selected)."""
        try:
            signals = mode_signals(mode)
        except ValueError:
            return
        name = _SYSTEM_NAME.get(system, system)
        cols = [str(c) for c in dataframe.columns]
        # a code-only file (no phase observable with data at all) stays readable: SPP needs codes only (HEAD
        # behaviour, SPP N66); a mode band without phase in a file that has phases raises
        has_phase = any(c.startswith("L") and dataframe[c].notna().any() for c in cols)
        for signal in signals:
            spec = signal_spec(signal)
            for kind, prefix in (("code", spec.code_prefix), ("phase", spec.phase_prefix)):
                if kind == "phase" and not has_phase:
                    continue
                if any(c.startswith(prefix) and dataframe[c].notna().any() for c in selected):
                    continue
                present = [c for c in cols if c.startswith(prefix)]
                found = ", ".join(f"{c}{'' if dataframe[c].notna().any() else ' (listed but empty)'}"
                                  for c in present) if present else "none"
                raise ValueError(
                    f"{name} mode {mode!r}: no {signal} {kind} observable ({prefix}x) with data in "
                    f"{self.obs_path}; band-{prefix[1:]} {kind} observables present: {found}")

    def select_obs_types_galileo(self, dataframe, mode) -> List[str]:
        """
        Faster + corrected Galileo observation-type selection.

        Fixes vs your original:
        - uses notna().sum(axis=0) instead of sum(axis=0) (you want availability, not sum of values)
        - precomputes availability counts once per band (E1/E5a/E5b) and groups by kind once
        - removes list(set(...)) (order-preserving unique)
        """

        cols = list(map(str, dataframe.columns))

        def order_preserving_unique(seq: List[str]) -> List[str]:
            return list(dict.fromkeys(seq))

        def pick_best_for_band(
                band_cols: List[str],
                priority_order: List[str],
        ) -> List[str]:
            """
            For each kind (col[:-1]) choose:
            - among columns with max availability (non-NaN count), pick best suffix by priority_order
            - if tie remains, pick lexicographically smallest
            """
            if not band_cols:
                return []

            # availability counts once
            nn = dataframe[band_cols].notna().sum(axis=0)

            # group by kind. A RINEX 2 file adapted to the internal contract has
            # two-character names ('C1'), which are the kind itself; a RINEX 3
            # file has a signal attribute to strip ('C1C' -> 'C1').
            by_kind = defaultdict(list)
            for c in band_cols:
                by_kind[c if len(c) == 2 else c[:-1]].append(c)

            out = []
            for kind, kind_cols in by_kind.items():
                # A canonical unsuffixed slot is the only candidate there is.
                if kind in kind_cols:
                    out.append(kind)
                    continue

                # choose max availability within this kind
                best_count = nn[kind_cols].max()
                max_cols = [c for c in kind_cols if nn[c] == best_count]

                # suffix priority
                def pri(c: str) -> int:
                    try:
                        return priority_order.index(c[-1])
                    except ValueError:
                        return 10 ** 9

                # choose best by (priority, lexicographic)
                max_cols_sorted = sorted(max_cols, key=lambda c: (pri(c), c))
                out.append(max_cols_sorted[0])

            return out

        selected_types: List[str] = []

        def band_filter(freq_char: str) -> List[str]:
            # Keep your original filtering spirit, including dropping 'D*' and '*li'.
            # len >= 2 admits the two-character names a RINEX 2 file adapts to.
            return [
                c for c in cols
                if len(c) >= 2
                   and c[1] == freq_char
                   and not (c.startswith("D") or c.endswith("li"))
            ]

        # Q196: a single-frequency mode takes the second band only with SINGLE_FREQUENCY_NEEDS_SECOND_BAND
        two_bands = SINGLE_FREQUENCY_NEEDS_SECOND_BAND
        if mode == "E1E5a" or (two_bands and mode in ["E1", "E5a"]):
            # E1 = '1', E5a = '5'
            e1_cols = band_filter("1")
            e5a_cols = band_filter("5")
            selected_types.extend(pick_best_for_band(e1_cols, ["C", "X", "Z", "B"]))
            selected_types.extend(pick_best_for_band(e5a_cols, ["Q", "I", "X"]))
        elif mode == "E1":
            selected_types.extend(pick_best_for_band(band_filter("1"), ["C", "X", "Z", "B"]))
        elif mode == "E5a":
            selected_types.extend(pick_best_for_band(band_filter("5"), ["Q", "I", "X"]))

        if mode == "E1E5b" or (two_bands and mode == "E5b"):
            # E1 = '1', E5b = '7'
            e1_cols = band_filter("1")
            e5b_cols = band_filter("7")
            selected_types.extend(pick_best_for_band(e1_cols, ["C", "X", "Z", "B"]))
            selected_types.extend(pick_best_for_band(e5b_cols, ["Q", "I", "X"]))
        elif mode == "E5b":
            selected_types.extend(pick_best_for_band(band_filter("7"), ["Q", "I", "X"]))

        if mode in ["E5aE5b"]:
            # E5a = '5', E5b = '7'
            e5a_cols = band_filter("5")
            e5b_cols = band_filter("7")
            selected_types.extend(pick_best_for_band(e5a_cols, ["Q", "I", "X"]))
            selected_types.extend(pick_best_for_band(e5b_cols, ["Q", "I", "X"]))

        selected_types = order_preserving_unique(selected_types)
        self._require_mode_observables("E", mode, selected_types, dataframe)
        return selected_types

    def select_obs_types_beidou(self, dataframe, mode) -> List[str]:
        """Select BeiDou RINEX 3/4 observables for a configured BDS signal mode.

        BeiDou RINEX bands are intentionally handled through signal metadata:
        B1I is C2/L2, B1C is C1/L1, B2a is C5/L5, B2I/B2b is C7/L7,
        and B3I is C6/L6.

        Strict per signal: an observable is taken only with a tracking
        attribute of the signal's ``suffix_priority`` (B2I: I/Q/X, B2b: Z/P/D; B1C: X/P/D, band-1 S/L/Z being
        B1A; first attribute present, then the column with the most values), or as the bare 2-character code (a
        RINEX 2 style column equal to the prefix), and only when it has data (a header-listed column left empty
        is never chosen). Band 7 carries both B2I and B2b, so another attribute of the band is never taken in
        place of the signal's own. A signal of the mode without a code or a phase observable of its own with
        data raises a ValueError (the phase only when the file has phase observables with data: a code-only file
        is read for SPP) naming the mode, the observables looked for and the band's observables the
        file has ('listed but empty' for an empty one); an SNR observable is optional. GPS and Galileo follow the
        same rule for code and phase (:meth:`_require_mode_observables`).
        """
        if dataframe is None:
            return []
        validate_mode_for_system(mode, "C")

        cols = list(map(str, dataframe.columns))
        selected: list[str] = []

        def candidates(prefix: str) -> list[str]:
            return [
                c for c in cols
                if c.startswith(prefix)
                   and not c.startswith("D")
                   and not c.endswith("li")
            ]

        nn_all = dataframe.notna().sum(axis=0)

        def pick(prefix: str, priority: tuple[str, ...]) -> str | None:
            # only observables with data: a header-listed column that stays empty (B2I of a BDS-3-only window,
            # a tlim window) would drop every row of the system
            cand = [c for c in candidates(prefix) if int(nn_all[c]) > 0]
            if not cand:
                return None
            if prefix in cand:  # bare RINEX 2 style code: the only candidate there is
                return prefix
            for suffix in priority:
                by_suffix = [c for c in cand if len(c) == len(prefix) + 1 and c[-1] == suffix]
                if by_suffix:
                    return max(by_suffix, key=lambda c: (int(nn_all[c]), c))
            return None

        def signals_of(code: str) -> str:
            """BeiDou signals whose tracking attributes include ``code``'s ('C7I' -> 'B2I')."""
            names = [name for name in ("B1I", "B1C", "B2a", "B2I", "B2b", "B3I")
                     if signal_spec(name).code_prefix[1:] == code[1:2]
                     and code[2:3] in _bds_attributes(name)]
            label = "/".join(names) or "no BeiDou signal of GNX"
            return label if int(nn_all[code]) > 0 else f"{label}, listed but empty"

        # a code-only file (no phase observable with data at all) stays readable (SPP); see
        # _require_mode_observables
        has_phase = any(c.startswith("L") and int(nn_all[c]) > 0 for c in cols)
        for signal in mode_signals(mode):
            spec = signal_spec(signal)
            attributes = _bds_attributes(signal)
            for kind, prefix in (("code", spec.code_prefix), ("phase", spec.phase_prefix),
                                 ("snr", "S" + spec.code_prefix[1:])):
                chosen = pick(prefix, attributes)
                if chosen is None and kind != "snr" and not (kind == "phase" and not has_phase):
                    wanted = "/".join(prefix + suffix for suffix in attributes)
                    present = [c for c in candidates(prefix) if len(c) == len(prefix) + 1 or c == prefix]
                    found = (", ".join(f"{c} ({signals_of(c)})" for c in present) if present else "none")
                    raise ValueError(
                        f"BeiDou mode {mode!r}: no {signal} {kind} observable ({wanted}) with data in "
                        f"{self.obs_path}; band-{spec.code_prefix[1:]} {kind} observables present: {found}")
                if chosen is not None and chosen not in selected:
                    selected.append(chosen)

        return selected

    def obs_header_reader(self):
        """Header items of the observation file: (marker name, antenna type, radome, antenna DELTA H/E/N [m],
        approx. position XYZ [m], its geodetic position, interval [s], GPS types, Galileo types, first epoch, last
        epoch, phase shifts, number of observations).

        Every record is read by column (``rinex_obs.read_header``):

        * ANT # / TYPE: the type from columns 21-36 and the radome from 37-40 (blank: 'NONE'), the rule of
          ``antex.receiver_antenna_from_header``; a type-only line is valid, a serial with blanks no longer
          shifts the type. A blank type gives one UserWarning naming the fields found, a missing record one
          UserWarning; the antenna is then unknown (None, None).
        * APPROX POSITION XYZ missing, zero, unreadable or off the Earth's surface: the position and the
          geodetic position are None (the session decides; nothing is computed from zeros).
        * ANTENNA: DELTA H/E/N missing or unreadable: zeros, with one UserWarning.
        * interval: :meth:`_obs_interval_s` (the data first; None when nothing gives one - ``load_obs_data``
          puts the resolved interval of ``ObsData.interval`` in meta[6]); first and last epoch: the first and last epoch
          read (GPST), the header TIME OF FIRST/LAST OBS only before any read (moved to GPST for BDT).
        * number of observations per satellite: None (not read).
        """
        header = self._obs_header()
        path = self.obs_path
        sta_name = header.marker_name or 'None'

        ant_type, ant_cover = header.antenna_type, header.antenna_radome
        if ant_type is None:
            ant_cover = None
            if header.raw.get("ANT # / TYPE"):
                payload = header.raw["ANT # / TYPE"][0].ljust(60)
                warnings.warn(
                    f"ANT # / TYPE of {path} has no antenna type (columns 21-36 blank; found serial "
                    f"{payload[0:20].strip()!r}, radome {payload[36:40].strip()!r}, "
                    f"line {payload.rstrip()!r}): the receiver antenna is unknown (no receiver PCO/PCV)",
                    UserWarning, stacklevel=2)
            else:
                warnings.warn(f"ANT # / TYPE missing in {path}: the receiver antenna is unknown (no receiver "
                              f"PCO/PCV)", UserWarning, stacklevel=2)

        if header.approx_position is not None:
            approx_pos = np.array(header.approx_position, dtype=float)
            pos_geod = ecef2geodetic(ecef=approx_pos)
        else:
            approx_pos = pos_geod = None

        interval = self._obs_interval_s()

        gps_obs = header.obs_types.get('G')
        gal_obs = header.obs_types.get('E')
        bds_obs = header.obs_types.get('C')
        self._obs_fields_by_system = {"G": gps_obs, "E": gal_obs, "C": bds_obs}

        read = self._last_obs_read()
        if read is not None:
            time_of_first_obs, time_of_last_obs = read.t_first, read.t_last
        else:
            shift = pd.Timedelta(seconds=_rinex_obs.BDT_TO_GPST_S if header.time_system == "BDT" else 0)
            time_of_first_obs = header.time_of_first_obs + shift if header.time_of_first_obs is not None else None
            time_of_last_obs = header.time_of_last_obs + shift if header.time_of_last_obs is not None else None

        phase_shift_dict = parse_phase_shift(fields=header.phase_shift_text) if header.phase_shift_text else None

        if header.antenna_delta_hen is not None:
            ant_h = np.array(header.antenna_delta_hen, dtype=float)
        else:
            ant_h = np.zeros(3)
            given = header.raw.get("ANTENNA: DELTA H/E/N")
            state = f"unreadable ({given[0].rstrip()!r})" if given else "missing"
            warnings.warn(f"ANTENNA: DELTA H/E/N {state} in {path}: antenna offsets taken as 0",
                          UserWarning, stacklevel=2)

        return (sta_name, ant_type, ant_cover, ant_h, approx_pos, pos_geod, interval, gps_obs, gal_obs,
                time_of_first_obs, time_of_last_obs, phase_shift_dict, None)

    def _nav_iono(self) -> dict:
        """``rinex_nav.legacy_iono`` of the ionosphere table of ``nav_path`` (``rinex_nav.read_iono_table``, cached per
        file): one set per model, lists of floats or None. Rejected records warn only for the configured systems
        and GPS, as in ``read_navigation``."""
        return _rinex_nav.legacy_iono(_rinex_nav.read_iono_table(self.nav_path, systems=sorted(self.sys)))

    def nav_header_reader(self):
        """GPS and Galileo broadcast ionosphere coefficients of ``nav_path``: (gpsa, gpsb, gala), lists of floats
        (4, 4, 3 values) or None.

        From the ionosphere table of ``rinex_nav``: RINEX 2 ION ALPHA / ION BETA read as
        floats (they came back as raw strings), RINEX 3 IONOSPHERIC CORR records and RINEX 4 '> ION'
        records, each set screened for plausibility; the first plausible set, header first, then LNAV,
        CNVX, IFNV. None for a model the file does not give (no message: the models that need the
        coefficients raise a readable error)."""
        legacy = self._nav_iono()
        return legacy["gpsa"], legacy["gpsb"], legacy["gala"]

    def nav_header_reader_bds(self):
        """BeiDou broadcast Klobuchar coefficients of ``nav_path``: (bdsa, bdsb), 4 floats each or None; one
        plausible set, GEO first, then IGSO, then MEO (RINEX 3.05 Table A5 note 2; ``rinex_nav.legacy_iono``)."""
        legacy = self._nav_iono()
        return legacy["bdsa"], legacy["bdsb"]

    def receiver_pco(self, name, type):
        """PCO of the receiver antenna ``name`` with radome ``type`` from ``atx_path``: {(antenna, radome, freq):
        [n, e, u] mm}, empty when the antenna (with exactly that radome) has no calibration.

        Read by :mod:`gnx_py.antex` (``Antex.legacy_rec_table``; the earlier line scan of
        this module is gone). The sessions build their antenna model with ``gnx_py.antex`` directly."""
        from ..models.antex import load_antex

        return defaultdict(dict, load_antex(self.atx_path).legacy_rec_table(name, type))

    def read_pco_antex(self, system_code: str, date: datetime):
        """Satellite PCOs of ``system_code`` ('G', 'E', 'C') valid at ``date`` from ``atx_path``:
        {(PRN, freq): [x, y, z] mm}.

        Read by :mod:`gnx_py.antex` (``Antex.legacy_sat_table``): an entry is valid
        from VALID FROM to VALID UNTIL; the earlier line scan of this module failed on a VALID UNTIL of
        23:59:59.9999999 and kept such entries without limit (expired E51, E52 and C47 on 2024-035). Several
        valid entries of a PRN give the one with the latest VALID FROM. ``date`` is required (``load_obs_data``
        passes the first observation epoch); None raises ValueError.
        """
        if date is None:
            raise ValueError(f"read_pco_antex({system_code!r}): the epoch is required (the satellite entries of "
                             f"{self.atx_path} valid at that epoch are returned); load_obs_data passes the first "
                             f"observation epoch")
        from ..models.antex import load_antex

        epoch = pd.Timestamp(date).floor("us").to_pydatetime()
        return defaultdict(list, load_antex(self.atx_path).legacy_sat_table(epoch, system_code))

    def load_obs_data(self, tlim=None):
        import pandas as pd

        # --- helpers -------------------------------------------------------------
        def _sv_list(df: pd.DataFrame | None) -> list[str]:
            if isinstance(df, pd.DataFrame):
                return df.index.get_level_values('sv').unique().tolist()
            return []

        def _assert_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
            """Clears the 'time' column, if any, and checks the index."""
            if isinstance(df, pd.DataFrame):
                df = df.drop(columns='time', errors='ignore')
                assert set(df.index.names) == {'sv', 'time'}
            return df

        # ------------------------------------------------------------------------

        obs_dict = self.load_and_filter(tlim=tlim)
        gps, gal, bds = obs_dict.get('G'), obs_dict.get('E'), obs_dict.get('C')
        interval = self._resolve_obs_interval_minutes(gps, gal, bds)

        gps_info = self._apply_obs_meta_overrides(self.process_observations2(), gps, gal)
        # meta[6]: the interval resolved for ObsData.interval [s] (data, header, file name, else 30 s)
        gps_info = gps_info[:6] + (interval * 60.0,) + gps_info[7:]
        time_of_first_obs = gps_info[-3]
        sta_name, sta_type, sta_cover = gps_info[0], gps_info[1], gps_info[2]
        xyz, flh = gps_info[4], gps_info[5]

        # Receiver PCO
        reciever_pco = None
        if self.atx_path is not None:
            reciever_pco = self.receiver_pco(name=sta_type, type=sta_cover)
        else:
            reciever_pco=None

        # Satellite PCO: the entries valid at the first observation epoch (none without observations)
        if self.atx_path is not None:
            satellite_pco = {}
            if time_of_first_obs is not None:
                satellite_pco_g = self.read_pco_antex(system_code='G', date=time_of_first_obs)
                satellite_pco_e = self.read_pco_antex(system_code='E', date=time_of_first_obs)
                satellite_pco_c = self.read_pco_antex(system_code='C', date=time_of_first_obs)
                satellite_pco = {**satellite_pco_e, **satellite_pco_g, **satellite_pco_c}
        else:
            satellite_pco = None
        # Satellite DCB
        if self.add_dcb:
            dcb_df = self._load_bias_products()
        else:
            dcb_df = None
        gps = self._attach_dcb_to_observations(gps, dcb_df)
        gal = self._attach_dcb_to_observations(gal, dcb_df)
        bds = self._attach_dcb_to_observations(bds, dcb_df)

        gps = _assert_index(gps)
        gal = _assert_index(gal)
        bds = _assert_index(bds)

        return ObsData(
            gps=gps,
            gal=gal,
            bds=bds,
            sat_pco=satellite_pco,
            rec_pco=reciever_pco,
            meta=gps_info,
            interval=interval,
            header=self._obs_header(),
        )

    def interpret_data_src(self, df):
        """The DataSrc bits of a Galileo navigation frame as boolean columns 'I/NAV_E1-B', 'F/NAV_E5a-I',
        'I/NAV_E5b-I', 'dt_E5a_E1', 'dt_E5b_E1' (``rinex_nav.galileo_data_source_columns``)."""
        return _rinex_nav.galileo_data_source_columns(df)

    def add_galileo_health_cols(self, df: pd.DataFrame,
                                health_col: str = "health",
                                prefix: str = "") -> pd.DataFrame:
        """The Galileo health word decoded into E1B_DVS, E1B_SHS, E5a_DVS, E5a_SHS, E5b_DVS, E5b_SHS (RINEX 3.05
        Table A8; ``rinex_nav.galileo_health_columns``; a word that cannot be decoded is DVS = 1 everywhere)."""
        return _rinex_nav.galileo_health_columns(df, health_col=health_col, prefix=prefix)

    def load_broadcast_orbit(self, tlim=None):
        """Read the broadcast ephemerides of the configured systems from ``nav_path`` (``rinex_nav``).

        RINEX 2 (GPS), RINEX 3 and RINEX 4 files, plain or compressed (gzip, .Z, bzip2, zip): RINEX 3 GPS,
        Galileo and BeiDou records are read from their fixed slots, RINEX 4 GPS LNAV (with the CNAV group delays
        ISC_L1CA, ISC_L2C, ISC_L5I5, ISC_L5Q5, TGD_CNAV), Galileo I/NAV and F/NAV and BeiDou D1/D2/CNV1-3 from
        their '>' records (``nav_message`` column). A file that cannot give a configured system raises
        ``rinex_nav.RinexNavError`` (a ValueError naming the file, its version and the systems in it) before
        any record is read where the header tells: a RINEX 2 navigation file (GPS only) for Galileo or
        BeiDou, a GLONASS/SBAS RINEX 2 file, an unknown version; after the read for a RINEX 3/4 file without
        records of the system. Nothing is printed.

        ``tlim`` (``[start, end]``) limits the epochs a caller processes, not the
        ephemerides: every record of the file is returned. The record in use at an
        epoch of a short window may have been issued before the window (Galileo
        ephemerides are used up to 4 h from Toe, GPS and BeiDou 2 h), so filtering the
        file by ``tlim`` would make the selector take another record than in the
        daily run at the same epoch. ``tlim`` is only checked here.

        Returns ``OrbitData``: the frames (index (sv, time = toc)), one ionosphere set per model (gpsa, gpsb,
        gala; bdsa, bdsb when BeiDou is configured), the ionosphere table ``iono`` and ``nav_info``.
        """
        if tlim is not None:
            _check_time_limit(tlim)
        nav = _rinex_nav.read_navigation(self.nav_path, "".join(sorted(self.sys)))
        legacy = _rinex_nav.legacy_iono(nav.iono)
        bdsa, bdsb = (legacy["bdsa"], legacy["bdsb"]) if 'C' in self.sys else (None, None)
        return OrbitData(gps_orb=nav.gps if 'G' in self.sys else None,
                         gal_orb=nav.gal if 'E' in self.sys else None,
                         bds_orb=nav.bds if 'C' in self.sys else None,
                         gpsa=legacy["gpsa"], gpsb=legacy["gpsb"], gala=legacy["gala"],
                         bdsa=bdsa, bdsb=bdsb, iono=nav.iono, nav_info=nav.info)

    def screen_navigation_message(self, broadcast_df: pd.DataFrame, system: str):
        """Broadcast health of ``system`` ('G', 'E' or 'C') as a ``rinex_nav.HealthTable``: one flag series per base satellite and message type (Galileo: E1-B and E5b flags from
        the I/NAV records, E5a flags from the F/NAV records; GPS health != 0; BeiDou SatH1 != 0), each record at
        its transmission time. :meth:`mark_outages` applies it: a flag stays in force until a newer record of the
        same satellite and message type, at most 4 h (GPS, Galileo) or 2 h (BeiDou). An empty table for None.

        The former result (a list of (sv, epochs) for G/C, {sv: {flag: epochs}} for E) kept one record copy per
        satellite (the last copy overwrote the others, so the Galileo E5a flags were lost); that form is
        still accepted by :meth:`mark_outages`.
        """
        return _rinex_nav.health_table(broadcast_df, system)

    @staticmethod
    def mark_outages(obs_df: pd.DataFrame,
                     system: str,
                     outage_info,
                     gps_span: pd.Timedelta = pd.Timedelta('2h'),
                     bds_span: pd.Timedelta = pd.Timedelta('2h'),
                     gal_span: pd.Timedelta = pd.Timedelta('10min')) -> pd.DataFrame:
        """
                Returns a copy of obs_df with additional outage columns (``rinex_nav.HEALTH_COLUMNS``):
                  • GPS : 'gps_outage', BeiDou: 'bds_outage'  (bool)
                  • GAL : 'E1B_DVS_outage', 'E1B_SHS_outage', 'E5a_DVS_outage', 'E5a_SHS_outage',
                          'E5b_DVS_outage', 'E5b_SHS_outage'

                obs_df      – MultiIndex over (sv, time) or (time, sv), in any row order
                outage_info – a ``rinex_nav.HealthTable`` (``screen_navigation_message``):
                              the broadcast health in force at each row (the latest record of the satellite and
                              message type at or before the epoch, at most ``HealthTable.max_age`` old; a
                              healthy record clears an earlier flag), or the legacy form: a list of
                              (sv, [epochs]) for G/C, a dict {sv: {flag: [epochs]}} for E, each flagged epoch
                              holding for ``gps_span`` / ``bds_span`` / ``gal_span`` after it.

                ``merge_asof`` needs its left frame sorted by time, but the
                result is written back to ``obs_df`` in ``obs_df``'s own row
                order. Sorting therefore has to be undone before the flags are
                assigned, otherwise a frame that is not already time-sorted -
                a RINEX 2 frame comes out grouped by satellite - has its outage
                flags scattered onto the wrong epochs.
                """
        if isinstance(outage_info, _rinex_nav.HealthTable):
            if outage_info.system.upper() != system.upper():
                raise ValueError(f"mark_outages: health table of system {outage_info.system!r} given for the "
                                 f"{system!r} observations")
            flags = _rinex_nav.health_flags_at(outage_info, obs_df.index)
            obs_df = obs_df.copy()
            for column in outage_info.flag_columns:
                obs_df[column] = flags[column].to_numpy(dtype=bool)
            return obs_df

        obs = obs_df.reset_index()
        obs['__row'] = np.arange(len(obs), dtype=np.int64)
        obs = obs.sort_values('time', kind='stable')

        def _scatter(merged: pd.DataFrame) -> np.ndarray:
            """Return a flag array in ``obs_df`` row order."""
            flags = np.zeros(len(obs_df), dtype=bool)
            flags[merged['__row'].to_numpy()] = merged['dummy'].notna().to_numpy()
            return flags

        # ------------------------------------------------------------------ GPS
        if system.upper() in {'G', 'C'}:
            outage_col = 'gps_outage' if system.upper() == 'G' else 'bds_outage'
            span = gps_span if system.upper() == 'G' else bds_span
            # --> long‑form: time, sv
            rows = [{'time': t, 'sv': sv} for sv, ts in outage_info for t in ts]
            if not rows:
                obs_df = obs_df.copy()
                obs_df[outage_col] = False
                return obs_df
            bad = (pd.DataFrame(rows)
                   .sort_values('time')
                   .assign(dummy=True))

            merged = pd.merge_asof(obs, bad, on='time', by='sv',
                                   direction='backward', tolerance=span)
            obs_df = obs_df.copy()
            obs_df[outage_col] = _scatter(merged)
            return obs_df

        # -------------------------------------------------------------- GALILEO
        if system.upper() == 'E':
            flag_cols = ['E1B_DVS', 'E1B_SHS', 'E5a_DVS', 'E5a_SHS', 'E5b_DVS', 'E5b_SHS']
            # long‑form: time, sv, flag
            flat = [{'time': t, 'sv': sv, 'flag': f}
                    for sv, flags in outage_info.items()
                    for f, ts in flags.items() for t in ts]
            if not flat:
                obs_df = obs_df.copy()
                for f in flag_cols:
                    obs_df[f + '_outage'] = False
                return obs_df

            bad = (pd.DataFrame(flat)
                   .sort_values('time')
                   .assign(dummy=True))

            obs_df = obs_df.copy()
            for fl in flag_cols:
                sub = bad[bad['flag'] == fl].drop('flag', axis=1)
                if sub.empty:
                    obs_df[fl + '_outage'] = False
                    continue
                sub = sub.sort_values('time')
                m = pd.merge_asof(obs, sub, on='time', by='sv',
                                  direction='backward', tolerance=gal_span)
                obs_df[fl + '_outage'] = _scatter(m)
            return obs_df

        # ------------------------------------------------------------
        raise ValueError(f"Unknown sys: {system}")


#: SINEX coordinate and velocity parameter types of SOLUTION/ESTIMATE read by :func:`parse_sinex`
_SINEX_COORDINATES = ("STAX", "STAY", "STAZ")
_SINEX_VELOCITIES = ("VELX", "VELY", "VELZ")


def _sinex_epoch(text: str) -> Optional[pd.Timestamp]:
    """A SINEX time ``YY:DDD:SSSSS`` (SINEX 2.02: YY <= 50 is 20YY, else 19YY) or ``YYYY:DDD:SSSSS``; None for
    ``00:000:00000`` (not given / open end) and for a field that is not a time."""
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        year, doy, sod = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if doy == 0 and sod == 0 and year == 0:
        return None
    if len(parts[0]) <= 2:
        year += 2000 if year <= 50 else 1900
    if doy < 1:
        return None
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1, seconds=sod)


def _sinex_solutions(text: str) -> dict:
    """``{(code, point, soln): record}`` of the SOLUTION/ESTIMATE station coordinates (and velocities) of a SINEX
    file, with the data span of SOLUTION/EPOCHS. ``record``: ``xyz`` [m], ``vel`` [m/y] (None without VELX/Y/Z),
    ``ref_epoch`` (of the STAX line), ``start``/``end`` (None: not given or open)."""
    sols: dict = {}
    spans: dict = {}
    block = None
    for line in text.splitlines():
        if line.startswith("+"):
            block = line[1:].split()[0] if line[1:].split() else ""
            continue
        if line.startswith("-"):
            block = None
            continue
        if block not in ("SOLUTION/ESTIMATE", "SOLUTION/EPOCHS") or line.startswith("*"):
            continue
        parts = line.split()
        if block == "SOLUTION/ESTIMATE" and len(parts) >= 9:
            kind = parts[1]
            if kind not in _SINEX_COORDINATES and kind not in _SINEX_VELOCITIES:
                continue
            try:
                value = float(parts[8])
            except ValueError:
                continue
            rec = sols.setdefault((parts[2], parts[3], parts[4]),
                                  {"xyz": [np.nan] * 3, "vel": [np.nan] * 3, "ref_epoch": None})
            if kind in _SINEX_COORDINATES:
                rec["xyz"][_SINEX_COORDINATES.index(kind)] = value
                rec["ref_epoch"] = _sinex_epoch(parts[5])
            else:
                rec["vel"][_SINEX_VELOCITIES.index(kind)] = value
        elif block == "SOLUTION/EPOCHS" and len(parts) >= 6:
            spans[(parts[0], parts[1], parts[2])] = (_sinex_epoch(parts[4]), _sinex_epoch(parts[5]))
    out = {}
    for key, rec in sols.items():
        xyz = np.asarray(rec["xyz"], dtype=float)
        if not np.all(np.isfinite(xyz)):
            continue
        vel = np.asarray(rec["vel"], dtype=float)
        start, end = spans.get(key, (None, None))
        out[key] = {"xyz": xyz, "vel": vel if np.all(np.isfinite(vel)) else None, "ref_epoch": rec["ref_epoch"],
                    "start": start, "end": end}
    return out


def _sinex_select(code: str, candidates: list, epoch: Optional[pd.Timestamp]) -> tuple:
    """(key, record, problem) of the solution of one site; ``problem`` is '' or the reason for a warning."""
    def label(c):
        (_, pt, soln), rec = c
        span = f"{rec['start'] or '?'} .. {rec['end'] or 'open'}"
        return f"point {pt} solution {soln} ({span})"

    def latest(cands):
        # SOLN compared as a number (SOLN 10 is later than SOLN 9), then as text; then the point code
        return max(cands, key=lambda c: (c[1]["start"] or pd.Timestamp.min, _sinex_soln_order(c[0][2]), c[0][1]))

    if len(candidates) == 1:
        return candidates[0] + ("",)
    if epoch is None:
        chosen = latest(candidates)
        return chosen + (f"{code}: {len(candidates)} solutions and no epoch given ("
                         + "; ".join(label(c) for c in candidates) + f"), the latest taken: {label(chosen)}",)
    valid = [c for c in candidates
             if (c[1]["start"] is None or c[1]["start"] <= epoch) and (c[1]["end"] is None or epoch <= c[1]["end"])]
    if len(valid) == 1:
        return valid[0] + ("",)
    if valid:
        chosen = latest(valid)
        return chosen + (f"{code}: {len(valid)} solutions valid at {epoch} ("
                         + "; ".join(label(c) for c in valid) + f"), the latest taken: {label(chosen)}",)

    def distance(c):
        start, end = c[1]["start"], c[1]["end"]
        if start is not None and epoch < start:
            return start - epoch
        return epoch - end if end is not None else pd.Timedelta(0)

    chosen = min(candidates, key=distance)
    # extrapolation outside the span of every solution (after the data end of the latest, before the data start of
    # the earliest) is the normal use of a multi-year solution: the outermost solution is taken without a warning;
    # an epoch in a gap between two solutions is ambiguous and warns
    ends = [c[1]["end"] for c in candidates]
    starts = [c[1]["start"] for c in candidates]
    after_all = all(e is not None and epoch > e for e in ends)
    before_all = all(s is not None and epoch < s for s in starts)
    if (after_all and chosen[1]["end"] == max(ends)) or (before_all and chosen[1]["start"] == min(starts)):
        return chosen + ("",)
    return chosen + (f"{code}: no solution valid at {epoch} (" + "; ".join(label(c) for c in candidates)
                     + f"), the nearest taken: {label(chosen)}",)


def _sinex_soln_order(soln: str) -> tuple:
    """Sort key of a SINEX solution number SOLN: numeric when it is a number (``'10'`` after ``'9'``)."""
    text = str(soln).strip()
    return (1, int(text), text) if text.isdigit() else (0, 0, text)


def parse_sinex(sinex_path, epoch=None, sites=None):
    """Station coordinates of a SINEX file: ``SOLUTION/ESTIMATE`` STAX/STAY/STAZ [m].

    The file may be plain or compressed (gzip ``.SNX.gz``, Unix compress ``.Z``, bzip2, zip; read in memory).
    Only the ``SOLUTION/ESTIMATE`` block is read: the a-priori values of ``SOLUTION/APRIORI`` and every other
    block are ignored (before v1.0.0 every line holding STAX/STAY/STAZ was read and the last one won, so a file with
    ``SOLUTION/APRIORI`` after ``SOLUTION/ESTIMATE`` gave the a-priori coordinates).

    One row per 4-character site code (index, sorted) with the columns ``STAX``, ``STAY``, ``STAZ``. A site with
    one solution takes it whatever ``epoch`` (its data span is not checked: it is the only coordinate set of the
    site). A site with several (several solution numbers SOLN after a discontinuity, or several point codes) takes,
    with ``epoch`` given, the one whose ``SOLUTION/EPOCHS`` data span contains the epoch (a solution without a span
    counts as valid), else the latest (latest data start, then the highest SOLN, numerically). An epoch after the
    data end of every solution takes the latest one and an epoch before the data start of every solution the
    earliest one, without a warning (extrapolation, the normal use of a multi-year solution). Several valid
    solutions, an epoch in a gap between solutions (the nearest is taken) or several solutions without ``epoch``
    give one UserWarning per call naming the sites and the solutions. With ``epoch`` given and velocities
    (VELX/VELY/VELZ [m/y]) for the chosen solution, the coordinates are moved from the reference epoch of the
    solution to ``epoch`` (years of 365.25 days); daily IGS solutions have no velocities.

    :param sinex_path: path of the SINEX file
    :param epoch: time the coordinates are wanted for (datetime-like) or None
    :param sites: iterable of 4-character site codes to read (upper case) or None for every site
    :return: pandas.DataFrame indexed by site code with the columns STAX, STAY, STAZ [m]. ``attrs['solutions']``:
        ``{site: {'point', 'soln', 'ref_epoch', 'start', 'end', 'velocity_applied'}}`` of the chosen solutions
    """
    text = _compression.open_text(sinex_path, what="SINEX file").read()
    epoch = None if epoch is None else pd.Timestamp(epoch)
    if epoch is not None and epoch.tzinfo is not None:
        epoch = epoch.tz_convert("UTC").tz_localize(None)
    wanted = None if sites is None else {str(s) for s in sites}
    by_site: dict = {}
    for key, rec in _sinex_solutions(text).items():
        if wanted is None or key[0] in wanted:
            by_site.setdefault(key[0], []).append((key, rec))
    rows, meta, problems = {}, {}, []
    for code in sorted(by_site):
        (_, pt, soln), rec, problem = _sinex_select(code, by_site[code], epoch)
        if problem:
            problems.append(problem)
        xyz = rec["xyz"].copy()
        moved = False
        if epoch is not None and rec["vel"] is not None and rec["ref_epoch"] is not None:
            years = (epoch - rec["ref_epoch"]) / pd.Timedelta(days=365.25)
            xyz = xyz + rec["vel"] * years
            moved = True
        rows[code] = xyz
        meta[code] = {"point": pt, "soln": soln, "ref_epoch": rec["ref_epoch"], "start": rec["start"],
                      "end": rec["end"], "velocity_applied": moved}
    if problems:
        warnings.warn(f"SINEX {Path(str(sinex_path)).name}: " + " | ".join(problems[:10])
                      + (f" | ... ({len(problems)} sites)" if len(problems) > 10 else ""), UserWarning, stacklevel=2)
    sinex = pd.DataFrame.from_dict(rows, orient="index", columns=list(_SINEX_COORDINATES), dtype=float)
    sinex.attrs["solutions"] = meta
    return sinex


def _sinex_station_reference(sinex_path, marker, epoch, session: str, obs_path=None):
    """(ECEF reference [m] or None, warning text or '') of the SPP/PPP station in ``sinex_path``: the site code is the first 4 characters of the RINEX MARKER NAME (upper case),
    the solution the one valid at ``epoch`` (:func:`parse_sinex`). A station that is not in the file gives None
    and the text of the session's warning (de/dn/du are then NaN)."""
    key = str(marker or "").strip()[:4].upper()
    name = Path(str(sinex_path)).name
    if not key or key == "NONE":
        return None, (f"{session}: the observation file{_obs_name(obs_path)} has no MARKER NAME, so its station "
                      f"cannot be found in the SINEX {name}: de/dn/du are NaN (no reference coordinates)")
    snx = parse_sinex(sinex_path, epoch=epoch, sites=[key])
    if key in snx.index:
        return snx.loc[key].to_numpy(dtype=float), ""
    return None, (f"{session}: station {key!r} (first 4 characters of the MARKER NAME{_obs_name(obs_path)}) is not "
                  f"in SOLUTION/ESTIMATE of the SINEX {name}: de/dn/du are NaN (no reference coordinates)")


def _obs_name(obs_path) -> str:
    return f" of {Path(str(obs_path)).name}" if obs_path else ""


def _first_epoch(*frames) -> Optional[pd.Timestamp]:
    """Earliest ``time`` index value of the given observation frames (None for frames without one)."""
    first = None
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "time" not in (frame.index.names or ()):
            continue
        t = pd.Timestamp(frame.index.get_level_values("time").min())
        first = t if first is None or t < first else first
    return first


# SP3 clock values from 999999 up mark a bad/absent clock (spec value 999999.999999)
_SP3_BAD_CLOCK = _sp3_reader.BAD_CLOCK
_sp3_sv = _sp3_reader.sp3_sv


def _sp3_facts(path) -> tuple[Optional[str], bool]:
    """(time system, velocity records) of an SP3 file, after the checks of :func:`gnx_py.sp3_reader.read_sp3_file`.

    The time system: columns 10-12 of the first '%c' line of SP3-c/d ('GPS', 'GAL', 'GLO', 'UTC', 'TAI', 'BDT',
    'QZS', 'IRN'); None for SP3-a/b (no '%c' line before the first epoch) or a blank/'ccc' field. Velocity
    records: whether the file has 'V' records (clock rates). The reader checks that every epoch lists exactly the
    header satellites in header order, with a clock field (ValueError naming the file, the line and the
    difference otherwise)."""
    sp3 = _sp3_reader.read_sp3_file(path)
    return sp3.time_system, sp3.velocity


def read_sp3(path: str, sys=('G','E','C')) -> pd.DataFrame:
    """
        Input
        -------
        path to sp3 (plain or compressed)

        Output
        -------
        MultiIndex (time, sv), rows in file order (epochs ascending, satellites in header order)
        columns:   x, y, z [km], clk [microseconds], dclk [1e-4 microseconds/s], sat, toc, epoch
        x, y, z are NaN for an absent position (0.000000) and clk, dclk for a bad/absent
        clock (999999.999999); a record missing both is dropped.

        Read by GNX's own SP3 reader (:mod:`gnx_py.sp3_reader`, SP3-a/b/c/d, replaces
        ``georinex.load_sp3`` with identical frames on the validation data and a corpus of real files). A file that is not a readable
        SP3 file raises ValueError naming the file and the line; a file without EOF that ends before the epochs its
        line 1 declares (a cut file) is read with a ``sp3_reader.SP3Warning`` naming the file.
        The epochs are taken as GPS time: the time system of the '%c' line (SP3-c/d) must be GPS (or missing,
        SP3-a/b); GAL (Galileo System Time, steered to GPS time within a few ns) is accepted with a warning; any
        other (GLO/UTC, TAI, BDT, ...) raises ValueError naming the file and the system.
        dclk is NaN for a record without a velocity ('V') record, and for a bad clock rate (999999.999999).
        """
    empty = pd.DataFrame(
        columns=['x', 'y', 'z', 'clk', 'dclk', 'sat', 'toc', 'epoch'],
        index=pd.MultiIndex.from_arrays([[], []], names=['time', 'sv']),
    )

    sp3 = _sp3_reader.read_sp3_file(path)
    time_system = sp3.time_system
    if time_system not in (None, "GPS", "GAL"):
        raise ValueError(f"SP3 file {path}: time system {time_system!r} (first '%c' line, columns 10-12); GNX "
                         f"reads SP3 epochs as GPS time and accepts GPS (or GAL) products only - use the product "
                         f"in GPS time")
    if time_system == "GAL":
        warnings.warn(f"SP3 file {path}: time system GAL (Galileo System Time) taken as GPS time (GST is steered "
                      f"to GPS time within a few ns)", UserWarning, stacklevel=2)

    rec = sp3.records
    out = rec[['x', 'y', 'z', 'clk', 'dclk']].copy()
    out['sat'] = rec.index.get_level_values('sv').tolist()
    out = out[out['sat'].str.startswith(sys)]
    if out.empty:
        return empty
    out['toc'] = datetime2toc(out.index.get_level_values('time').tolist())
    out['epoch'] = out.index.get_level_values('time')
    # SP3-c/d "no value" markers are per quantity: a bad/absent clock is 999999.999999 and an
    # absent position is 0.000000 0.000000 0.000000. A record may miss only one of them (CODE
    # finals give the 24:00 node a valid position with a 999999 clock), so each is set to NaN on
    # its own; the interpolators then take positions and clocks from their own valid nodes.
    bad_clk = out['clk'].to_numpy(dtype=float) >= _SP3_BAD_CLOCK
    out.loc[bad_clk, ['clk', 'dclk']] = np.nan
    out.loc[out['dclk'].to_numpy(dtype=float) >= _SP3_BAD_CLOCK, 'dclk'] = np.nan
    bad_pos = (out[['x', 'y', 'z']].to_numpy(dtype=float) == 0.0).all(axis=1)
    out.loc[bad_pos, ['x', 'y', 'z']] = np.nan
    # a record with neither a position nor a clock carries nothing
    out = out[~(bad_clk & bad_pos)]
    return out


def _solution_to_netcdf(solution: pd.DataFrame, path, **kwargs) -> None:
    """``solution.to_xarray().to_netcdf(path)`` for SPP and PPP results: ``path`` as str or Path (``.nc`` added
    without a suffix); the frame's ``attrs`` that netCDF cannot store (dicts such as ``precise_clock_datum``) are
    written as their text."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".nc")
    ds = solution.to_xarray()
    ds.attrs = {str(k): (v if isinstance(v, (str, int, float, np.number)) else str(v))
                for k, v in (solution.attrs or {}).items()}
    ds.to_netcdf(path, **kwargs)


# ``from gnx_py.io import *`` exports what it exported when gnx_py.io was the module io.py: the public names of this
# file, not the modules of the package (rinex_obs, sp3_reader, ... are package attributes once imported).
__all__ = [name for name, value in globals().items()
           if not name.startswith("_")
           and not (isinstance(value, type(math)) and value.__name__.startswith(__name__ + "."))]
