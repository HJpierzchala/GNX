import gzip
import math
import re
import shutil
import subprocess
import traceback
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional
import math
import re
from collections import defaultdict
from typing import List
import georinex as gr
# from geodezyx.files_rw import read_rinex_obs
import numpy as np
import pandas as pd

from . import ecef2geodetic
from .gnss import (
    DEFAULT_MODE_BY_SYSTEM,
    mode_signals,
    signal_spec,
    validate_mode_for_system,
)
from .rinex2_adapter import adapt_rinex2_gps_nav, adapt_rinex2_gps_obs, is_supported_rinex2_version
from .time import doy_to_datetime, datetime2toc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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


class GFZRNX2:
    """Light wrapper to gfzrnx → pandas.DataFrame"""
    def __init__(
        self,
        exe: str | None = None,
        crx2rnx: str | None = None,
        drop_sentinel: float = 9999999999.999,
        keep_tmp: bool = False,
    ):
        import shutil as _shutil

        self.exe = exe or _shutil.which("gfzrnx")
        if not self.exe:
            raise FileNotFoundError("gfzrnx not found in PATH")

        self.crx2rnx = crx2rnx or _shutil.which("crx2rnx")
        self.drop_sentinel = drop_sentinel
        self.keep_tmp = keep_tmp

    # ---------- helpers ------------------------------------------------ #
    def _run(self, cmd: list[str]) -> None:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Command failed\n"
                f"$ {' '.join(cmd)}\n\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
            ) from None

    def _crx_to_rnx(self, crx: Path) -> Path:
        rnx = crx.with_suffix(".rnx")
        if self.crx2rnx:
            self._run([self.crx2rnx, "-f", str(crx)])
            if not rnx.exists():
                raise FileNotFoundError(f"{rnx} was not created after crx2rnx")
        else:
            self._run(
                [self.exe, "-crx2rnx", "-finp", str(crx), "-fout", str(rnx), "-f"]
            )
        return rnx

    # ---------- API ---------------------------------------------------- #
    def tabulate(
        self,
        rinex: str | Path,
        out: str | Path | None = None,
        satsys: str = "G",
        parse: bool = True,
    ):
        rinex = Path(rinex)
        tmp_to_delete: list[Path] = []

        # ── 1. prepare clean .rnx ────────────────────────────────── #
        if rinex.suffixes[-2:] == [".crx", ".gz"]:          # *.crx.gz
            ungz = rinex.with_suffix("")                    # …/file.crx
            with gzip.open(rinex, "rb") as fin, open(ungz, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            tmp_to_delete.append(ungz)
            rinex_obs = self._crx_to_rnx(ungz)
            tmp_to_delete.append(rinex_obs)                #delete after

        elif rinex.suffix == ".crx":                        # *.crx
            rinex_obs = self._crx_to_rnx(rinex)

        else:                                               # *.rnx | *.rnx.gz
            if rinex.suffix == ".gz":
                ungz = rinex.with_suffix("")                # …/file.rnx
                with gzip.open(rinex, "rb") as fin, open(ungz, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                tmp_to_delete.append(ungz)
                rinex_obs = ungz
            else:
                rinex_obs = rinex

        # ── 2. tabulate GFZRNX ─────────────────────────────────────── #
        out = Path(out) if out else rinex_obs.with_suffix(".tab")
        self._run(
            [
                self.exe,
                "-finp",
                str(rinex_obs),
                "-fout",
                str(out),
                "-tab",
                "-satsys",
                satsys,
                "-f",
            ]
        )

        if not self.keep_tmp:
            for p in tmp_to_delete:
                try:
                    p.unlink(missing_ok=True)
                except PermissionError:
                    pass

        if not parse:
            return out

        # ── 3. pandas post-processing ───────────────────────────────── #
        df = (
            pd.read_csv(out, sep=r"\s+")
            .assign(time=lambda x: pd.to_datetime(x["DATE"] + " " + x["TIME"]))
            .rename(columns={"PRN": "sv"})
            .replace(self.drop_sentinel, pd.NA)
        )
        df.columns = df.columns.str.lstrip("# ").str.strip()

        df = (
            df.set_index(["sv", "time"])
            .filter(regex=r"^[CLS]\d")
            .sort_index()
        )
        df = df.apply(pd.to_numeric, errors="coerce").replace(self.drop_sentinel, pd.NA)
        mask = (df[[c for c in df.columns if c.startswith(('C', 'L'))]] == 9999999999.999).any(axis=1)
        df = df[~mask]
        #
        out.unlink(missing_ok=True)
        return df


@dataclass
class OrbitData:
    """
    Nav data container
    """
    gps_orb: Optional[pd.DataFrame] = None
    gal_orb: Optional[pd.DataFrame] = None
    bds_orb: Optional[pd.DataFrame] = None
    gpsa: float | None = None
    gpsb: float | None = None
    gala: float | None = None
    bdsa: float | None = None
    bdsb: float | None = None


@dataclass
class ObsData:
    """
    Obs data container
    """
    gps: pd.DataFrame = None
    gal: pd.DataFrame = None
    bds: pd.DataFrame = None
    sat_pco: defaultdict = None
    rec_pco: defaultdict = None
    meta: tuple= None
    interval: float = None

class GNSSDataProcessor2:
    """
    GNSS data processor. Supports:
    - RINEX 3 observation files
    - RINEX 3 navigation files
    - DCB .BIA and .BSX files
    - ANTEX files
    Main class for opening basic GNSS input files

    """

    def __init__(self, atx_path=None, obs_path=None, dcb_path=None, nav_path=None, mode='L1', sys='G',
                 galileo_modes=None, beidou_modes=None, use_gfz=False, station_name=None):
        """
        :param atx_path: path to .atx file
        :param obs_path: path to .obs file
        :param dcb_path: path to .dcb file
        :param nav_path: path to nav file
        :param use_gfz: flag to use GFZRNX

        :param sys: str, set
        :param mode: GPS mode (L1, L1L2 eec)
        :param galileo_modes: Galileo modes (E1, E1E5a etc)
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

        self.dcb_paths = self._normalize_dcb_paths(dcb_path)
        self.dcb_type = None
        if self.dcb_paths:
            first_bias_path = str(self.dcb_paths[0]).split('/')[-1].split('_')
            self.dcb_type = first_bias_path[3]
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
        merged = None
        station_bias_parts: list[pd.DataFrame] = []

        def combine_first_aligned(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
            left_aligned = left.copy()
            right_aligned = right.copy()

            missing_in_left = [col for col in right_aligned.columns if col not in left_aligned.columns]
            for col in missing_in_left:
                left_aligned[col] = np.nan

            missing_in_right = [col for col in left_aligned.columns if col not in right_aligned.columns]
            for col in missing_in_right:
                right_aligned[col] = np.nan

            left_aligned = left_aligned.sort_index(axis=1)
            right_aligned = right_aligned.reindex(columns=left_aligned.columns)
            return left_aligned.combine_first(right_aligned)

        for path in self.dcb_paths:
            bias_df, _ = self.read_bia(path=path)
            station_bias = bias_df.attrs.get("station_bias_by_system")
            if isinstance(station_bias, pd.DataFrame) and not station_bias.empty:
                station_bias_parts.append(station_bias)
            if merged is None:
                merged = bias_df
                continue

            merged = combine_first_aligned(merged, bias_df)

        if merged is not None and station_bias_parts:
            station_merged = station_bias_parts[0]
            for station_bias in station_bias_parts[1:]:
                station_merged = combine_first_aligned(station_merged, station_bias)
            merged.attrs["station_bias_by_system"] = station_merged

        return merged

    def _get_obs_rinex_info(self) -> dict:
        if not hasattr(self, "_obs_rinex_info_cache"):
            self._obs_rinex_info_cache = gr.rinexinfo(self.obs_path) if self.obs_path else {}
        return self._obs_rinex_info_cache

    def _get_nav_rinex_info(self) -> dict:
        if not hasattr(self, "_nav_rinex_info_cache"):
            self._nav_rinex_info_cache = gr.rinexinfo(self.nav_path) if self.nav_path else {}
        return self._nav_rinex_info_cache

    def _get_obs_header(self) -> dict:
        if not hasattr(self, "_obs_header_cache"):
            self._obs_header_cache = gr.rinexheader(self.obs_path) if self.obs_path else {}
        return self._obs_header_cache

    def _get_nav_header(self) -> dict:
        if not hasattr(self, "_nav_header_cache"):
            if not self.nav_path:
                self._nav_header_cache = {}
            else:
                try:
                    self._nav_header_cache = gr.rinexheader(self.nav_path)
                except ValueError:
                    self._nav_header_cache = self._read_basic_rinex_header(self.nav_path)
        return self._nav_header_cache

    @staticmethod
    def _read_basic_rinex_header(path: str | Path) -> dict:
        header: dict[str, object] = {"IONOSPHERIC CORR": {}}
        with open(path, "r", encoding="ascii", errors="ignore") as stream:
            for line in stream:
                label = line[60:].strip() if len(line) >= 60 else ""
                payload = line[:60]
                if label == "RINEX VERSION / TYPE":
                    try:
                        header["version"] = float(payload[:9])
                    except ValueError:
                        pass
                    header["filetype"] = payload[20:21].strip()
                    header["systems"] = payload[40:41].strip()
                elif label == "IONOSPHERIC CORR":
                    parts = payload.split()
                    if parts:
                        key = parts[0]
                        values = []
                        for value in parts[1:]:
                            try:
                                values.append(float(value.replace("D", "E").replace("d", "e")))
                            except ValueError:
                                pass
                        header["IONOSPHERIC CORR"][key] = values
                elif label == "LEAP SECONDS":
                    parts = payload.split()
                    if parts:
                        try:
                            header["LEAP SECONDS"] = int(parts[0])
                        except ValueError:
                            pass
                elif label == "END OF HEADER" or "END OF HEADER" in line:
                    break
        return header

    def _is_rinex2_obs(self) -> bool:
        return is_supported_rinex2_version(self._get_obs_rinex_info().get("version"))

    def _is_rinex2_nav(self) -> bool:
        return is_supported_rinex2_version(self._get_nav_rinex_info().get("version"))

    def _is_rinex4_nav(self) -> bool:
        try:
            return float(self._get_nav_rinex_info().get("version", 0.0)) >= 4.0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _adapter_report(df: pd.DataFrame | None) -> dict:
        if isinstance(df, pd.DataFrame):
            return df.attrs.get("gnx_adapter", {})
        return {}

    def _parse_obs_start_from_filename(self) -> datetime | None:
        name = Path(self.obs_path).name
        parts = name.split("_")
        if len(parts) < 4:
            return None
        token = parts[-4]
        if len(token) < 7 or not token[:7].isdigit():
            return None
        yr = int(token[:4])
        doy = int(token[4:7])
        return doy_to_datetime(year=yr, doy=doy)

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

    def _resolve_obs_interval_minutes(
        self,
        gps: pd.DataFrame | None,
        gal: pd.DataFrame | None,
        bds: pd.DataFrame | None = None,
    ) -> float | None:
        for df in (gps, gal, bds):
            report = self._adapter_report(df)
            interval = report.get("meta_override", {}).get("interval")
            if interval is not None:
                return float(interval) / 60.0

        hdr = self._get_obs_header()
        if "INTERVAL" in hdr:
            try:
                return float(str(hdr["INTERVAL"]).strip()) / 60.0
            except ValueError:
                pass

        inferred = self._infer_interval_from_df(gps) or self._infer_interval_from_df(gal) or self._infer_interval_from_df(bds)
        if inferred is not None:
            return inferred

        parts = Path(self.obs_path).name.split("_")
        if len(parts) >= 2:
            token = parts[-2]
            if len(token) >= 2 and token[:2].isdigit():
                return float(token[:2]) / 60.0
        return 0.5

    def _apply_obs_meta_overrides(self, gps_info: tuple, gps: pd.DataFrame | None, gal: pd.DataFrame | None) -> tuple:
        info = list(gps_info)
        report = self._adapter_report(gps) or self._adapter_report(gal)
        override = report.get("meta_override", {})

        if "interval" in override and override["interval"] is not None:
            info[6] = override["interval"]
        if "gps_obs" in override and override["gps_obs"] is not None:
            info[7] = override["gps_obs"]
        if "gal_obs" in override:
            info[8] = override["gal_obs"]
        if "time_of_first_obs" in override and override["time_of_first_obs"] is not None:
            info[9] = override["time_of_first_obs"]
        if "time_of_last_obs" in override and override["time_of_last_obs"] is not None:
            info[10] = override["time_of_last_obs"]
        if "phase_shift_dict" in override:
            info[11] = override["phase_shift_dict"]

        if info[9] is None:
            for df in (gps, gal):
                if isinstance(df, pd.DataFrame):
                    info[9] = df.index.get_level_values("time").min()
                    break
        if info[10] is None:
            for df in (gps, gal):
                if isinstance(df, pd.DataFrame):
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
        """
                Reads .BIA files (OSB, DSB, etc.) into a DataFrame.
                When parse_time=True, adds the BIASSTART_dt and BIASEND_dt columns.
                For OSB, generates columns in the BIAS_<OBS1> format, for DSB in the BIAS_<OBS1>_<OBS2> format.
                """
        # -- 1. Dekodowanie całego pliku ---------------------------------------
        raw = Path(path).read_bytes()
        for enc in encodings:
            try:
                text, used_enc = raw.decode(enc), enc
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("The file does not match the specified encodings.")

        lines = text.splitlines()
        idx = None
        try:
            idx = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("*BIAS"))
        except StopIteration:
            raise ValueError("No header matching *BIAS was found")

        file_type = None
        for line in lines[idx + 1:]:
            if line.strip() and not line.startswith("*"):  # Pomijamy puste linie i komentarze
                first_col = line.split()[0]
                if first_col in ["OSB", "DSB"]:
                    file_type = first_col
                    break
        if file_type is None:
            raise ValueError("The file type (OSB or DSB) cannot be determined based on the data.")

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

        df = df.rename(columns={'BIASSTART_dt': 'time'})
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

        sat_df = df[~station_mask].copy()
        station_df = df[station_mask].copy()

        station_wide = None
        if file_type == "OSB":
            wide = (
                sat_df.pivot_table(
                    index=["sv", "time"],
                    columns="OBS1",
                    values="ESTIMATEDVALUE",
                    aggfunc="first"
                )
                .add_prefix("OSB_")
                .reset_index()
            )
            if not station_df.empty:
                station_wide = (
                    station_df.pivot_table(
                        index=["_STATION_NAME", "_BIAS_SYSTEM", "time"],
                        columns="OBS1",
                        values="ESTIMATEDVALUE",
                        aggfunc="first",
                    )
                    .add_prefix("OSB_")
                )
            file_type = 'OSB'
        else:  # file_type == "DSB"
            # Tworzenie nowej kolumny z połączonymi OBS1 i OBS2
            df["OBS_PAIR"] = df["OBS1"] + "_" + df["OBS2"]
            sat_df = df[~station_mask].copy()
            station_df = df[station_mask].copy()
            wide = (
                sat_df.pivot_table(
                    index=["sv", "time"],
                    columns="OBS_PAIR",
                    values="ESTIMATEDVALUE",
                    aggfunc="first"
                )
                .add_prefix("BIAS_")
                .reset_index()
            )
            if not station_df.empty:
                station_wide = (
                    station_df.pivot_table(
                        index=["_STATION_NAME", "_BIAS_SYSTEM", "time"],
                        columns="OBS_PAIR",
                        values="ESTIMATEDVALUE",
                        aggfunc="first",
                    )
                    .add_prefix("BIAS_")
                )
            file_type = 'DSB'

        wide = wide.set_index(['sv', 'time'])
        valid_systems = ('G', 'E', 'C')  # GPS, Galileo, BeiDou
        sv_index = wide.index.get_level_values('sv').astype(str)
        keep = sv_index.str.startswith(valid_systems)
        wide = wide[keep]
        bias_availability: dict[str, set[str]] = {}
        bias_pair_availability: dict[str, set[str]] = {}
        sv_index = wide.index.get_level_values('sv').astype(str)
        for system in valid_systems:
            system_wide = wide[sv_index.str.startswith(system)]
            if system_wide.empty:
                continue
            if file_type == "OSB":
                bias_availability[system] = {
                    col[4:]
                    for col in system_wide.columns
                    if col.startswith("OSB_") and system_wide[col].notna().any()
                }
            else:
                bias_pair_availability[system] = {
                    col[5:]
                    for col in system_wide.columns
                    if col.startswith("BIAS_") and system_wide[col].notna().any()
                }
        wide.columns.name = None
        if station_wide is not None:
            station_wide.index = station_wide.index.set_names(["station", "system", "time"])
            station_wide.columns.name = None
            wide.attrs["station_bias_by_system"] = station_wide.fillna(0.0)

        out = wide.fillna(0.0)
        out.attrs.update(wide.attrs)
        out.attrs["bias_observables_by_system"] = {
            system: sorted(values) for system, values in bias_availability.items()
        }
        out.attrs["bias_pairs_by_system"] = {
            system: sorted(values) for system, values in bias_pair_availability.items()
        }
        return out, file_type

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

    def _attach_dcb_to_observations(self, df: pd.DataFrame | None, dcb: pd.DataFrame | None) -> pd.DataFrame | None:
        """Attach OSB/DSB columns, matching suffixed observation SV ids to product PRNs."""
        if not isinstance(df, pd.DataFrame) or not isinstance(dcb, pd.DataFrame):
            return df
        report = df.attrs.get("gnx_adapter", {})
        bias_ok = report.get("bias_eligibility", {}).get("attach_bia_in_load_obs_data", True)
        if not bias_ok:
            return df

        left = df.reset_index()
        left["_bias_sv"] = left["sv"].map(self._canonical_bias_sv)

        if self.dcb_type == '30S':
            right = dcb.reset_index().rename(columns={"sv": "_bias_sv"})
            out = left.merge(right, on=["_bias_sv", "time"], how="left", sort=False)
        elif self.dcb_type == '01D':
            right = dcb.reset_index().rename(columns={"sv": "_bias_sv"}).drop(columns="time", errors="ignore")
            right = right.drop_duplicates(subset=["_bias_sv"], keep="first")
            out = left.merge(right, on="_bias_sv", how="left", sort=False)
        else:
            return df

        out = out.drop(columns="_bias_sv", errors="ignore").set_index(["sv", "time"])
        bias_cols = out.columns[out.columns.str.startswith(("OSB", "BIAS"))]
        if len(bias_cols):
            out.loc[:, bias_cols] = out.loc[:, bias_cols].fillna(0)
        if self.station_name:
            target_system = self._infer_observation_system(df)
            station_bias_by_system = dcb.attrs.get("station_bias_by_system")
            if isinstance(station_bias_by_system, pd.DataFrame) and not station_bias_by_system.empty:
                station_bias = station_bias_by_system.reset_index()
                station_bias = station_bias[
                    station_bias["station"].astype(str).str.upper().eq(str(self.station_name).upper())
                ]
                if target_system is not None and "system" in station_bias.columns:
                    station_bias = station_bias[station_bias["system"].astype(str).eq(target_system)]
            else:
                station_bias = dcb.reset_index()
                station_bias = station_bias[
                    station_bias["sv"].astype(str).str.upper().str.startswith(str(self.station_name).upper())
                ]
                if target_system is not None and "system" in station_bias.columns:
                    station_bias = station_bias[station_bias["system"].astype(str).eq(target_system)]
            station_bias_cols = [
                col for col in station_bias.columns
                if col.startswith(("OSB", "BIAS"))
            ]
            if not station_bias.empty and station_bias_cols:
                station_values = pd.to_numeric(
                    station_bias.iloc[0][station_bias_cols],
                    errors="coerce",
                ).fillna(0.0)
                for col, value in station_values.items():
                    out[f"STA_{col}"] = value
        return out

    from pathlib import Path

    def load_and_filter(self, tlim=None, version=3.0):
        """
        Faster load + filter for GPS/Galileo/BeiDou.
        Main speedups:
        - avoid df[cols].dropna(...) (double-copy); use one boolean mask
        - optionally build mask on minimal "critical" columns only
        - instantiate GFZRNX2 once
        """
        results = {"G": None, "E": None, "C": None}
        obs_rinex2 = self._is_rinex2_obs()
        obs_info = self._get_obs_rinex_info()
        obs_header = self._get_obs_header() if obs_rinex2 else None

        if obs_rinex2 and any(sys_char != "G" for sys_char in self.sys):
            raise ValueError("RINEX 2 MVP currently supports GPS-only observations.")

        # --- Helper: build deterministic output tab path ---
        def _tab_out_path(sys_char: str) -> str:
            name = Path(self.obs_path).name
            for sfx in [".crx", ".crx.gz", "rnx"]:
                if name.endswith(sfx):
                    name = name.replace(sfx, ".tab")
            return f"{Path(self.obs_path).parent}/{sys_char}_{name}"

        # --- Helper: load dataframe for a given system ---
        gfz = None
        if getattr(self, "use_gfz", False):
            try:
                gfz = GFZRNX2()
            except Exception:
                gfz = None

        def _load_sys(sys_char: str):
            if obs_rinex2:
                return gr.load(self.obs_path, use={sys_char}, tlim=tlim, fast=True).to_dataframe()
            # 1) try gfzrnx if requested and available
            if gfz is not None:
                try:
                    out = _tab_out_path(sys_char)
                    return gfz.tabulate(rinex=self.obs_path, out=out, satsys=sys_char)
                except (FileNotFoundError, RuntimeError):
                    pass
            # 2) fallback georinex
            return gr.load(self.obs_path, use={sys_char}, tlim=tlim, fast=True).to_dataframe()

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

        # --- GPS ---
        if "G" in self.sys:
            try:
                gps_df = _load_sys("G")
                if obs_rinex2 and gps_df is not None and not gps_df.empty:
                    gps_df = adapt_rinex2_gps_obs(
                        gps_df,
                        header=obs_header,
                        rinex_info=obs_info,
                        system="G",
                        mode=self.mode,
                        strict=True,
                        bias_policy="safe",
                        obs_path=self.obs_path,
                    )
                gps_obs_types = self.select_obs_types_gps(gps_df)
                # fast_mask=True is usually faster and matches your "avoid NaN columns" intent
                gps_df_filtered = _filter_df(gps_df, gps_obs_types, fast_mask=True)
                results["G"] = gps_df_filtered
            except KeyError:
                results["G"] = None

        # --- Galileo ---
        if "E" in self.sys:
            try:
                gal_df = _load_sys("E")
                gal_obs_types = self.select_obs_types_galileo(gal_df, self.galileo_modes)
                gal_df_filtered = _filter_df(gal_df, gal_obs_types, fast_mask=True)
                results["E"] = gal_df_filtered
            except KeyError:
                results["E"] = None

        # --- BeiDou / BDS ---
        if "C" in self.sys:
            try:
                bds_df = _load_sys("C")
                bds_obs_types = self.select_obs_types_beidou(bds_df, self.beidou_modes)
                bds_df_filtered = _filter_df(bds_df, bds_obs_types, fast_mask=True)
                results["C"] = bds_df_filtered
            except KeyError:
                results["C"] = None

        return results

    def select_obs_types_gps(self, dataframe, *, consistent_suffix: bool = True) -> List[str]:
        """
        Faster GPS observation-type selection.

        Key speedups vs original:
        - build candidate column list once
        - compute non-NaN counts ONCE for all candidates
        - group columns by 'kind' (col[:-1]) once (no repeated startswith scanning)
        """

        cols = list(map(str, dataframe.columns))

        # --- 1) Candidate columns by mode (same intent as your code) ---
        if self.mode in ["L1L2", "L1", "L2"]:
            wanted_freqs = ("1", "2")
        elif self.mode in ["L1L5", "L5"]:
            wanted_freqs = ("1", "5")
        elif self.mode in ["L2L5"]:
            wanted_freqs = ("2", "5")
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

        # --- 4) Choose best column per kind ---
        for kind in kinds:
            kind_cols = by_kind[kind]
            if not kind_cols:
                continue

            chosen_col = None

            # Canonical internal slots like C1/L1 should win over suffixed variants.
            if kind in kind_cols:
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

        return selected_types

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

            # group by kind
            by_kind = defaultdict(list)
            for c in band_cols:
                by_kind[c[:-1]].append(c)

            out = []
            for kind, kind_cols in by_kind.items():
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
            # Keep your original filtering spirit, including dropping 'D*' and '*li'
            return [
                c for c in cols
                if len(c) >= 3
                   and c[1] == freq_char
                   and not (c.startswith("D") or c.endswith("li"))
            ]

        if mode in ["E1E5a", "E1", "E5a"]:
            # E1 = '1', E5a = '5'
            e1_cols = band_filter("1")
            e5a_cols = band_filter("5")
            selected_types.extend(pick_best_for_band(e1_cols, ["C", "X", "Z", "B"]))
            selected_types.extend(pick_best_for_band(e5a_cols, ["Q", "I", "X"]))

        if mode in ["E1E5b", "E5b"]:
            # E1 = '1', E5b = '7'
            e1_cols = band_filter("1")
            e5b_cols = band_filter("7")
            selected_types.extend(pick_best_for_band(e1_cols, ["C", "X", "Z", "B"]))
            selected_types.extend(pick_best_for_band(e5b_cols, ["Q", "I", "X"]))

        if mode in ["E5aE5b"]:
            # E5a = '5', E5b = '7'
            e5a_cols = band_filter("5")
            e5b_cols = band_filter("7")
            selected_types.extend(pick_best_for_band(e5a_cols, ["Q", "I", "X"]))
            selected_types.extend(pick_best_for_band(e5b_cols, ["Q", "I", "X"]))

        return order_preserving_unique(selected_types)

    def select_obs_types_beidou(self, dataframe, mode) -> List[str]:
        """Select BeiDou RINEX 3/4 observables for a configured BDS signal mode.

        BeiDou RINEX bands are intentionally handled through signal metadata:
        B1I is C2/L2, B1C is C1/L1, B2a is C5/L5, B2I/B2b is C7/L7,
        and B3I is C6/L6.
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

        def pick(prefix: str, priority: tuple[str, ...]) -> str | None:
            cand = candidates(prefix)
            if not cand:
                return None
            nn = dataframe[cand].notna().sum(axis=0)
            for suffix in priority:
                by_suffix = [c for c in cand if len(c) > len(prefix) and c[-1] == suffix]
                if by_suffix:
                    return max(by_suffix, key=lambda c: (int(nn[c]), c))
            return max(cand, key=lambda c: (int(nn[c]), c))

        for signal in mode_signals(mode):
            spec = signal_spec(signal)
            for prefix in (spec.code_prefix, spec.phase_prefix, "S" + spec.code_prefix[1:]):
                chosen = pick(prefix, spec.suffix_priority)
                if chosen is not None and chosen not in selected:
                    selected.append(chosen)

        return selected

    def obs_header_reader(self):
        path = self.obs_path
        hdr = self._get_obs_header()
        if 'MARKER NAME' in hdr.keys():
            sta_name = hdr['MARKER NAME'].strip()
        else:
            sta_name = 'None'

        if 'ANT # / TYPE' in hdr.keys():
            # POPRAWA
            txt = hdr['ANT # / TYPE'].strip().split()
            if len(txt) == 3:
                ant_type = hdr['ANT # / TYPE'].strip().split()[1]
                ant_cover = hdr['ANT # / TYPE'].strip().split()[-1]
            else:
                ant_type = hdr['ANT # / TYPE'].strip().split()[0]
                ant_cover = hdr['ANT # / TYPE'].strip().split()[1]
        else:
            ant_type = None
            ant_cover = None

        if 'position' in hdr.keys():
            approx_pos = np.array(hdr['position'])
        elif 'APPROX POSITION XYZ' in hdr.keys():
            pos_str = hdr['APPROX POSITION XYZ'].strip().split()
            approx_pos = np.array([float(i) for i in pos_str])
        else:
            approx_pos = np.array([0.0, 0.0, 0.0])

        if 'position_geodetic' in hdr.keys():
            pos_geod = np.array(hdr['position_geodetic'])
        else:
            pos_geod = ecef2geodetic(ecef=approx_pos)

        if 'INTERVAL' in hdr.keys():
            interval = float(hdr['INTERVAL'].strip())
        else:
            interval = None

        if 'fields' in hdr.keys():

            if isinstance(hdr['fields'], list):
                if hdr.get('systems') == 'G':
                    gps_obs, gal_obs, bds_obs = hdr['fields'], None, None
                elif hdr.get('systems') == 'E':
                    gps_obs, gal_obs, bds_obs = None, hdr['fields'], None
                elif hdr.get('systems') == 'C':
                    gps_obs, gal_obs, bds_obs = None, None, hdr['fields']
                else:
                    gps_obs = gal_obs = bds_obs = hdr['fields']
            else:
                if 'G' in hdr['fields'].keys():
                    gps_obs = hdr['fields']['G']
                else:
                    gps_obs = None
                if 'E' in hdr['fields'].keys():
                    gal_obs = hdr['fields']['E']
                else:
                    gal_obs = None
                if 'C' in hdr['fields'].keys():
                    bds_obs = hdr['fields']['C']
                else:
                    bds_obs = None
        else:
            gps_obs = None
            gal_obs = None
            bds_obs = None
        self._obs_fields_by_system = {"G": gps_obs, "E": gal_obs, "C": bds_obs}

        if 't0' in hdr.keys():
            time_of_first_obs = hdr['t0']
        elif 'TIME OF FIRST OBS3' in hdr.keys():
            date_list = hdr['TIME OF FIRST OBS3'].strip().split()[:-1]
            year, month, day, hour, minute, second = map(float, date_list)
            time_of_first_obs = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        else:
            time_of_first_obs = None

        if 't1' in hdr.keys():
            time_of_last_obs = hdr['t1']
        elif 'TIME OF LAST OBS3' in hdr.keys():
            date_list = hdr['TIME OF LAST OBS3'].strip().split()[:-1]
            year, month, day, hour, minute, second = map(float, date_list)
            time_of_last_obs = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        else:
            time_of_last_obs = None
        if time_of_first_obs is None:
            time_of_first_obs = self._parse_obs_start_from_filename()
        if time_of_last_obs is None and time_of_first_obs is not None:
            time_of_last_obs = time_of_first_obs + timedelta(hours=24)
        if 'PRN / # OF OBS3' in hdr.keys():
            raw_data = hdr['PRN / # OF OBS3']
            matches = re.findall(r'([A-Z]\d{2})\s+((?:\d+\s+)+)', raw_data)
            satellite_data = {}
            for match in matches:
                sat_id = match[0]
                values = list(map(int, match[1].split()))
                satellite_data[sat_id] = values

            num_obs_dict = {}
            for sat_id, values in satellite_data.items():
                if sat_id.startswith(('G', 'E', 'C')):
                    num_obs_dict[sat_id] = {}
                    num_obs_dict[sat_id] = {f"{hdr['fields'][sat_id[:1]][i]}": values[i] for i in range(len(values))}
        else:
            num_obs_dict = None

        if 'SYS / PHASE SHIFT' in hdr.keys():
            text = hdr['SYS / PHASE SHIFT']
            phase_shift_dict = parse_phase_shift(fields=text)
        else:
            phase_shift_dict = None

        if 'ANTENNA: DELTA H/E/N' in hdr.keys():
            ant_h = np.array([float(i) for i in hdr['ANTENNA: DELTA H/E/N'].strip().split()])
        else:
            ant_h = None

        return sta_name, ant_type, ant_cover, ant_h, approx_pos, pos_geod, interval, gps_obs, gal_obs, time_of_first_obs, time_of_last_obs, phase_shift_dict, num_obs_dict

    def nav_header_reader(self):
        hdr = self._get_nav_header()
        if {'ION ALPHA', 'ION BETA'}.issubset(hdr.keys()):
            return hdr['ION ALPHA'], hdr['ION BETA'], None
        if 'IONOSPHERIC CORR' in hdr.keys():
            ion_corr = hdr['IONOSPHERIC CORR']
            ion_keys = set(ion_corr.keys())
            if {'GPSA', 'GPSB'}.issubset(ion_keys):
                gpsa = ion_corr['GPSA']
                gpsb = ion_corr['GPSB']
            elif {'ION ALPHA', 'ION BETA'}.issubset(ion_keys):
                gpsa, gpsb = ion_corr['ION ALPHA'], ion_corr['ION BETA']
            else:
                gpsa = None
                gpsb = None

            if 'GAL' in ion_keys:
                gala = ion_corr['GAL']
            else:
                gala = None
            return gpsa, gpsb, gala
        else:
            print('No ionospheric coefficients for GPS and Galileo found in navigation file')
            return None, None, None

    def nav_header_reader_bds(self):
        """Read BeiDou broadcast ionospheric coefficients when present."""
        hdr = self._get_nav_header()
        ion_corr = hdr.get('IONOSPHERIC CORR', {})
        if {'BDSA', 'BDSB'}.issubset(ion_corr.keys()):
            return ion_corr['BDSA'], ion_corr['BDSB']
        return None, None

    def receiver_pco(self, name, type):
        my_dict = defaultdict(dict)
        with open(self.atx_path, 'r') as plik:
            lines = plik.readlines()
            for i, line in enumerate(lines):
                if (name in line.split()) and (type in line.split()) and ('COMMENT' not in line.split()) and (
                        'TYPE / SERIAL NO' in line):
                    j = i
                    for j in range(j, len(lines)):
                        if 'START OF FREQUENCY' in lines[j]:
                            FRE = lines[j].split()[0]
                        if 'NORTH / EAST / UP' in lines[j]:
                            pco = [float(x) for x in lines[j].split()[:3]]
                            my_dict[(name, type, FRE)] = pco
                        if 'END OF ANTENNA' in lines[j]:
                            break
        return my_dict

    def read_pco_antex(self, system_code: str, date: datetime):
        """
                Loads satellite PCO offsets from the ANTEX file.

                Parameters
                ---------
                antex_path : str
                    Full path to the ANTEX file (.atx).
                system_code : str
                    System code ('G' for GPS, 'E' for Galileo, etc.).
                date : datetime
                    Date and time for which we are downloading calibrations.  The function
                    selects entries where 'VALID FROM' <= date <= 'VALID UNTIL'.

                Returns
        -------
        defaultdict(list)
        Dictionary where the key is (PRN, FRQ) and the value is a three-element list
         with the PCO vector in millimetres.
        """
        antex_path = self.atx_path
        results = defaultdict(list)
        with open(antex_path, 'r') as f:
            lines = f.readlines()
        n = len(lines)
        i = 0
        while i < n:
            if 'START OF ANTENNA' in lines[i]:
                prn_code = None
                valid_from = datetime.min
                valid_until = datetime.max
                freq_pco = {}
                i += 1
                while i < n and 'END OF ANTENNA' not in lines[i]:
                    line = lines[i]
                    if 'TYPE / SERIAL NO' in line:
                        m = re.search(r'\b([GREJCS][0-9]{2,3})\b', line)
                        if m:
                            prn_code = m.group(1)
                    elif 'VALID FROM' in line:
                        parts = line[:60].split()
                        if len(parts) >= 6:
                            try:
                                y, mth, d, hh, mm = map(int, parts[:5])
                                sec = float(parts[5])
                                sec_int = int(sec)
                                micro = int(round((sec - sec_int) * 1e6))
                                valid_from = datetime(y, mth, d, hh, mm, sec_int, micro)
                            except:
                                pass
                    elif 'VALID UNTIL' in line:
                        parts = line[:60].split()
                        if len(parts) >= 6:
                            ints = [int(float(p)) for p in parts[:6]]
                            if any(ints):
                                try:
                                    y, mth, d, hh, mm = map(int, parts[:5])
                                    sec = float(parts[5])
                                    sec_int = int(sec)
                                    micro = int(round((sec - sec_int) * 1e6))
                                    valid_until = datetime(y, mth, d, hh, mm, sec_int, micro)
                                except:
                                    pass
                            else:
                                valid_until = datetime.max
                    elif 'START OF FREQUENCY' in line:
                        m = re.search(r'\b([GREJCS][0-9]{2})\b', line)
                        freq_code = m.group(1) if m else line.strip().split()[0]
                        i += 1
                        while i < n and 'END OF FREQUENCY' not in lines[i]:
                            line = lines[i]
                            if 'NORTH / EAST / UP' in line:
                                try:
                                    coords = [float(line[j:j + 10]) for j in range(0, 30, 10)]
                                    freq_pco[freq_code] = coords
                                except:
                                    pass
                            i += 1
                    i += 1
                if prn_code and prn_code[0] == system_code and date >= valid_from and date <= valid_until:
                    for f, coords in freq_pco.items():
                        results[(prn_code, f)] = coords
            else:
                i += 1
        return results


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
        time_of_first_obs = gps_info[-3]
        sta_name, sta_type, sta_cover = gps_info[0], gps_info[1], gps_info[2]
        xyz, flh = gps_info[4], gps_info[5]

        # Receiver PCO
        reciever_pco = None
        if self.atx_path is not None:
            reciever_pco = self.receiver_pco(name=sta_type, type=sta_cover)
        else:
            reciever_pco=None

        # Satellite PCO
        if self.atx_path is not None:
            satellite_pco_g = self.read_pco_antex(system_code='G',date=time_of_first_obs)
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
            interval=interval
        )

    def interpret_data_src(self, df):
        """
                Interprets the 'DataSrc' column in DataFrame to extract information about Galileo orbit types.
                Works on a single satellite orbit segment.

                Parameters:
                df (pandas.DataFrame): DataFrame containing the 'DataSrc' column with integer values.

                Returns:
                pandas.DataFrame: The original DataFrame with additional columns indicating data sources and orbit types.
                """
        bit_meanings = {
            0: 'I/NAV_E1-B',
            1: 'F/NAV_E5a-I',
            2: 'I/NAV_E5b-I',
            # Bity 3 i 4 są zarezerwowane
            8: 'dt_E5a_E1',
            9: 'dt_E5b_E1'
        }

        def interpret_bits(data_src_value):
            data_src_value = int(data_src_value)
            bits = bin(data_src_value)[2:].zfill(10)
            bits = bits[::-1]
            result = {}
            for bit_position, meaning in bit_meanings.items():
                if bit_position < len(bits):
                    bit_value = int(bits[bit_position])
                    result[meaning] = bool(bit_value)
                else:
                    result[meaning] = False
            return pd.Series(result)

        df_bits = df['DataSrc'].apply(interpret_bits)

        df = pd.concat([df, df_bits], axis=1)

        return df

    def add_galileo_health_cols(self, df: pd.DataFrame,
                                health_col: str = "health",
                                prefix: str = "") -> pd.DataFrame:
        """
                Decoding Signal Health Status and Data Validity Status for Galileo.
                Works on a single satellite orbit segment.
        Map:
        SHS: 0 - OK
        1 - out of service
        2 - Extended operation
        3 - in test
        DVS: 0 - navigation data valid
        1 - working without guarantee
        """

        _DVS_TEXT = np.array([0, 1])
        _HS_TEXT = np.array([0,
                             1,
                             2,
                             3])
        word = (df[health_col].round(12) * (2 ** 30)).astype("int64") & 0x3F
        w = word.to_numpy()

        e5a_dvs = w & 0b001
        e5a_hs = np.bitwise_and(np.right_shift(w, 1), 0b011)
        e5b_dvs = np.bitwise_and(np.right_shift(w, 3), 0b001)
        e5b_hs = np.bitwise_and(np.right_shift(w, 4), 0b011)

        df[prefix + "E5a_DVS"] = _DVS_TEXT[e5a_dvs]
        df[prefix + "E5a_SHS"] = _HS_TEXT[e5a_hs]
        df[prefix + "E5b_DVS"] = _DVS_TEXT[e5b_dvs]
        df[prefix + "E5b_SHS"] = _HS_TEXT[e5b_hs]

        cat = pd.CategoricalDtype(_HS_TEXT, ordered=False)
        df[prefix + "E5a_SHS"] = df[prefix + "E5a_SHS"].astype(cat)
        df[prefix + "E5b_SHS"] = df[prefix + "E5b_SHS"].astype(cat)
        return df

    @staticmethod
    def _repair_galileo_bgd_transtime(data: pd.DataFrame, threshold: float = 1e-1) -> pd.DataFrame:
        if data.empty or not {'BGDe5a', 'BGDe5b', 'TransTime'}.issubset(data.columns):
            return data

        repaired = data.copy()
        invalid_e5a = repaired['BGDe5a'].abs() > threshold
        invalid_e5b = repaired['BGDe5b'].abs() > threshold

        # When exactly one BGD column carries a transmit-time-scale value, swap it back.
        # Rows with both BGDs above threshold are ambiguous and should not be altered silently.
        swap_e5a = invalid_e5a & ~invalid_e5b
        swap_e5b = invalid_e5b & ~invalid_e5a

        if swap_e5a.any():
            original_trans = repaired.loc[swap_e5a, 'TransTime'].copy()
            repaired.loc[swap_e5a, 'TransTime'] = repaired.loc[swap_e5a, 'BGDe5a'].to_numpy()
            repaired.loc[swap_e5a, 'BGDe5a'] = original_trans.to_numpy()

        if swap_e5b.any():
            original_trans = repaired.loc[swap_e5b, 'TransTime'].copy()
            repaired.loc[swap_e5b, 'TransTime'] = repaired.loc[swap_e5b, 'BGDe5b'].to_numpy()
            repaired.loc[swap_e5b, 'BGDe5b'] = original_trans.to_numpy()

        return repaired

    @staticmethod
    def _repair_beidou_nav_tail(data: pd.DataFrame) -> pd.DataFrame:
        """Repair georinex BDS NAV tail shifts around SVacc/SatH1/TGD/TransTime.

        Some mixed RINEX 3.04 BDS records are parsed with one optional spare
        field present for only part of the constellation. georinex then exposes
        rows where the tail fields are shifted, e.g. TGD2 contains TransTime or
        AODC contains TransTime. That makes healthy messages look unhealthy and
        causes SPP to select stale/future ephemerides.
        """
        needed = {"SVacc", "SatH1", "TGD1", "TGD2", "TransTime", "AODC"}
        if data.empty or not needed.issubset(data.columns):
            return data

        repaired = data.copy()
        small = 1e-5
        sow_like = 1_000.0

        # Pattern A: an extra spare1 value is present and the useful tail is
        # shifted left from SVacc onward:
        # spare1->SVacc, SVacc->SatH1, SatH1->TGD1, TGD1->TGD2,
        # TGD2->TransTime, TransTime->AODC.
        if "spare1" in repaired.columns:
            shift_from_spare1 = (
                repaired["spare1"].notna()
                & repaired["TGD2"].abs().gt(sow_like)
                & repaired["SatH1"].abs().lt(small)
                & repaired["TGD1"].abs().lt(small)
            )
            if shift_from_spare1.any():
                old = repaired.loc[
                    shift_from_spare1,
                    ["spare1", "SVacc", "SatH1", "TGD1", "TGD2", "TransTime"],
                ].copy()
                repaired.loc[shift_from_spare1, "SVacc"] = old["spare1"].to_numpy()
                repaired.loc[shift_from_spare1, "SatH1"] = old["SVacc"].to_numpy()
                repaired.loc[shift_from_spare1, "TGD1"] = old["SatH1"].to_numpy()
                repaired.loc[shift_from_spare1, "TGD2"] = old["TGD1"].to_numpy()
                repaired.loc[shift_from_spare1, "TransTime"] = old["TGD2"].to_numpy()
                repaired.loc[shift_from_spare1, "AODC"] = old["TransTime"].to_numpy()

        # Pattern B: no spare1 value was parsed, so the tail is shifted right:
        # SatH1->SVacc, TGD1->SatH1, TGD2->TGD1, TransTime->TGD2,
        # AODC->TransTime. AODC itself is absent in these rows and is not used
        # in SPP, so set it to zero after moving TransTime back.
        shift_from_aodc = (
            repaired["AODC"].abs().gt(sow_like)
            & repaired["TransTime"].abs().lt(small)
            & repaired["TGD2"].abs().lt(small)
        )
        if shift_from_aodc.any():
            old = repaired.loc[
                shift_from_aodc,
                ["SatH1", "TGD1", "TGD2", "TransTime", "AODC"],
            ].copy()
            repaired.loc[shift_from_aodc, "SVacc"] = old["SatH1"].to_numpy()
            repaired.loc[shift_from_aodc, "SatH1"] = old["TGD1"].to_numpy()
            repaired.loc[shift_from_aodc, "TGD1"] = old["TGD2"].to_numpy()
            repaired.loc[shift_from_aodc, "TGD2"] = old["TransTime"].to_numpy()
            repaired.loc[shift_from_aodc, "TransTime"] = old["AODC"].to_numpy()
            repaired.loc[shift_from_aodc, "AODC"] = 0.0

        return repaired

    @staticmethod
    def _rinex_nav_float_fields(line: str, start: int = 4, count: int = 4) -> list[float]:
        out: list[float] = []
        padded = line.rstrip("\n").ljust(start + 19 * count)
        for i in range(count):
            field = padded[start + 19 * i:start + 19 * (i + 1)].strip()
            if not field:
                out.append(np.nan)
                continue
            try:
                out.append(float(field.replace("D", "E").replace("d", "e")))
            except ValueError:
                out.append(np.nan)
        return out

    @staticmethod
    def _rinex4_clock_line(line: str) -> tuple[str, pd.Timestamp, list[float]]:
        sv = line[:3].strip()
        year = int(line[4:8])
        month = int(line[9:11])
        day = int(line[12:14])
        hour = int(line[15:17])
        minute = int(line[18:20])
        second = float(line[21:23])
        sec_int = int(second)
        microsecond = int(round((second - sec_int) * 1_000_000))
        epoch = pd.Timestamp(datetime(year, month, day, hour, minute, sec_int, microsecond))
        return sv, epoch, GNSSDataProcessor2._rinex_nav_float_fields(line, start=23, count=3)

    @staticmethod
    def _rinex4_record_lines(path: str | Path):
        in_header = True
        current_header: str | None = None
        current_lines: list[str] = []
        with open(path, "r", encoding="ascii", errors="ignore") as stream:
            for line in stream:
                if in_header:
                    if "END OF HEADER" in line:
                        in_header = False
                    continue
                if line.startswith(">"):
                    if current_header is not None:
                        yield current_header, current_lines
                    current_header = line.rstrip("\n")
                    current_lines = []
                else:
                    current_lines.append(line.rstrip("\n"))
        if current_header is not None:
            yield current_header, current_lines

    def _parse_rinex4_bds_eph_record(self, header: str, lines: list[str]) -> dict | None:
        parts = header.split()
        if len(parts) < 4 or parts[1] != "EPH" or not parts[2].startswith("C"):
            return None
        message_type = parts[3]
        if message_type not in {"D1", "D2", "CNV1", "CNV2", "CNV3"} or not lines:
            return None

        sv, epoch, clock = self._rinex4_clock_line(lines[0])
        data_lines = [self._rinex_nav_float_fields(line) for line in lines[1:]]
        row: dict[str, object] = {
            "sv": sv,
            "time": epoch,
            "nav_message": message_type,
            "SVclockBias": clock[0],
            "SVclockDrift": clock[1],
            "SVclockDriftRate": clock[2],
        }

        if message_type in {"D1", "D2"}:
            vals = [value for fields in data_lines for value in fields]
            names = [
                "AODE", "Crs", "DeltaN", "M0",
                "Cuc", "Eccentricity", "Cus", "sqrtA",
                "Toe", "Cic", "Omega0", "Cis",
                "Io", "Crc", "omega", "OmegaDot",
                "IDOT", "spare0", "BDTWeek", "spare1",
                "SVacc", "SatH1", "TGD1", "TGD2",
                "TransTime", "AODC", "spare2", "spare3",
            ]
            row.update({name: vals[i] if i < len(vals) else np.nan for i, name in enumerate(names)})
            row["health"] = row.get("SatH1")
            return row

        if len(data_lines) < 7:
            return None
        orbit1, orbit2, orbit3, orbit4, orbit5, orbit6 = data_lines[:6]
        row.update(
            {
                "ADot": orbit1[0],
                "Crs": orbit1[1],
                "DeltaN": orbit1[2],
                "M0": orbit1[3],
                "Cuc": orbit2[0],
                "Eccentricity": orbit2[1],
                "Cus": orbit2[2],
                "sqrtA": orbit2[3],
                "Toe": orbit3[0],
                "Cic": orbit3[1],
                "Omega0": orbit3[2],
                "Cis": orbit3[3],
                "Io": orbit4[0],
                "Crc": orbit4[1],
                "omega": orbit4[2],
                "OmegaDot": orbit4[3],
                "IDOT": orbit5[0],
                "DeltaNdot": orbit5[1],
                "SatType": orbit5[2],
                "Top": orbit5[3],
                "SISAI_oe": orbit6[0],
                "SISAI_ocb": orbit6[1],
                "SISAI_oc1": orbit6[2],
                "SISAI_oc2": orbit6[3],
                # Compatibility fields used by the existing BDS propagator.
                "AODE": 0.0,
                "spare0": orbit5[1],
                "BDTWeek": 0.0,
                "SVacc": orbit6[0],
                "TGD1": 0.0,
                "TGD2": 0.0,
            }
        )

        if message_type in {"CNV1", "CNV2"}:
            if len(data_lines) < 9:
                return None
            orbit7, orbit8, orbit9 = data_lines[6], data_lines[7], data_lines[8]
            if message_type == "CNV1":
                row["ISC_B1Cd"] = orbit7[0]
                row["TGD_B1Cp"] = orbit7[2]
                row["TGD_B2ap"] = orbit7[3]
            else:
                row["ISC_B2ad"] = orbit7[1]
                row["TGD_B1Cp"] = orbit7[2]
                row["TGD_B2ap"] = orbit7[3]
            row.update(
                {
                    "SISMAI": orbit8[0],
                    "SatH1": orbit8[1],
                    "IntegrityFlags": orbit8[2],
                    "AODC": orbit8[3],
                    "IODC": orbit8[3],
                    "TransTime": orbit9[0],
                    "IODE": orbit9[3],
                    "AODE": orbit9[3],
                    "health": orbit8[1],
                }
            )
            return row

        orbit7 = data_lines[6]
        orbit8 = data_lines[7] if len(data_lines) > 7 else [np.nan, np.nan, np.nan, np.nan]
        row.update(
            {
                "SISMAI": orbit7[0],
                "SatH1": orbit7[1],
                "IntegrityFlags": orbit7[2],
                "TGD_B2bI": orbit7[3],
                "TransTime": orbit8[0],
                "AODC": 0.0,
                "IODC": 0.0,
                "IODE": 0.0,
                "AODE": 0.0,
                "health": orbit7[1],
            }
        )
        return row

    def _load_rinex4_bds_nav(self, tlim=None) -> pd.DataFrame:
        rows: list[dict] = []
        for header, lines in self._rinex4_record_lines(self.nav_path):
            row = self._parse_rinex4_bds_eph_record(header, lines)
            if row is not None:
                rows.append(row)
        if not rows:
            return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["sv", "time"]))

        out = pd.DataFrame(rows)
        if tlim is not None:
            start, end = pd.Timestamp(tlim[0]), pd.Timestamp(tlim[1])
            out = out[(out["time"] >= start) & (out["time"] <= end)]
        if out.empty:
            return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["sv", "time"]))

        out = out.set_index(["sv", "time"]).sort_index()
        out.attrs["rinex_version"] = self._get_nav_rinex_info().get("version")
        out.attrs["rinex4_bds_message_types"] = sorted(out["nav_message"].dropna().unique().tolist())
        return out

    def load_broadcast_orbit(self, tlim=None):
        gpsa, gpsb, gala = self.nav_header_reader()
        bdsa, bdsb = self.nav_header_reader_bds() if 'C' in self.sys else (None, None)
        nav_rinex2 = self._is_rinex2_nav() if 'G' in self.sys else False
        nav_info = self._get_nav_rinex_info() if nav_rinex2 else {}
        nav_header = self._get_nav_header() if nav_rinex2 else None
        if nav_rinex2 and any(sys_char != "G" for sys_char in self.sys):
            raise ValueError("RINEX 2 MVP currently supports GPS-only navigation.")
        if 'G' in self.sys:
            gps_orb = gr.load(rinexfn=self.nav_path,
                              use={'G'},
                              tlim=tlim).dropna(dim='time', how='all').to_dataframe().dropna(how='all')
            if nav_rinex2 and not gps_orb.empty:
                gps_orb = adapt_rinex2_gps_nav(
                    gps_orb,
                    header=nav_header,
                    rinex_info=nav_info,
                    system='G',
                    strict=True,
                    nav_path=self.nav_path,
                )
            else:
                gps_orb = gps_orb.swaplevel()
        else:
            gps_orb = None
        if 'E' in self.sys:
            gal_orb = None
            try:
                gal_orb = gr.load(rinexfn=self.nav_path,
                                  use={'E'},
                                  tlim=tlim).dropna(dim='time', how='all').to_dataframe().dropna(how='all')
                gal_orb = self._repair_galileo_bgd_transtime(gal_orb)
                gal_orb = self.interpret_data_src(df=gal_orb)
                gal_orb = self.add_galileo_health_cols(df=gal_orb)
                gal_orb = gal_orb.swaplevel()
            except Exception as e:
                traceback.print_exc()
        else:
            gal_orb = None
        if 'C' in self.sys:
            bds_orb = None
            try:
                if self._is_rinex4_nav():
                    bds_orb = self._load_rinex4_bds_nav(tlim=tlim)
                else:
                    bds_orb = gr.load(rinexfn=self.nav_path,
                                      use={'C'},
                                      tlim=tlim).dropna(dim='time', how='all').to_dataframe().dropna(how='all')
                if not bds_orb.empty:
                    if not self._is_rinex4_nav():
                        bds_orb = self._repair_beidou_nav_tail(bds_orb)
                        bds_orb = bds_orb.swaplevel()
                    if 'SatH1' in bds_orb.columns and 'health' not in bds_orb.columns:
                        bds_orb['health'] = bds_orb['SatH1']
            except Exception:
                traceback.print_exc()
        else:
            bds_orb = None
        return OrbitData(gps_orb=gps_orb,
                         gal_orb=gal_orb,
                         bds_orb=bds_orb,
                         gpsa=gpsa, gpsb=gpsb, gala=gala,
                         bdsa=bdsa, bdsb=bdsb)

    def screen_navigation_message(self, broadcast_df: pd.DataFrame, system: str):
        """
                Returns a list of tuples (sv, epoch_list) for satellites
                that have at least one navigation message marked as 'uncertain'
                or 'faulty'.

                broadcast_df – MultiIndex (time, sv) + columns E5a_DVS, E5a_SHS, E5b_DVS, E5b_SHS
                sys       – 'G' (GPS) or 'E' (Galileo)
                """
        if system in {'G', 'C'}: # GPS / BeiDou
            unhealthy = []
            if broadcast_df is not None:
                for sv, g in broadcast_df.groupby('sv'):
                    if 'health' in g.columns and np.any(g['health'] != 0):
                        unhealthy.append(
                            (sv[:3] if len(sv) > 3 else sv,
                             g.loc[g['health'] != 0].index.get_level_values('time').tolist())
                        )
            return unhealthy

            # ---------------  GALILEO  ---------------
        elif system == 'E':
            result = {}
            if broadcast_df is not None:
                required = ['E5a_DVS', 'E5a_SHS', 'E5b_DVS', 'E5b_SHS']
                if not set(required).issubset(broadcast_df.columns):
                    raise KeyError(f"Columns missing: {set(required) - set(broadcast_df.columns)}")

                mask = pd.DataFrame({
                    'E5a_DVS': broadcast_df['E5a_DVS'] == 1,
                    'E5a_SHS': broadcast_df['E5a_SHS'].isin([1, 3]),
                    'E5b_DVS': broadcast_df['E5b_DVS'] == 1,
                    'E5b_SHS': broadcast_df['E5b_SHS'].isin([1, 3]),
                }, index=broadcast_df.index)


                for sv, g_mask in mask.groupby(level='sv'):
                    flags_with_errors = {}
                    for flag in required:
                        if g_mask[flag].any():
                            epochs = g_mask.index[g_mask[flag]].get_level_values('time').tolist()
                            flags_with_errors[flag] = epochs
                    if flags_with_errors:
                        result[sv.split('_')[0]] = flags_with_errors  # 'E07_1' → 'E07'
            return result

    @staticmethod
    def mark_outages(obs_df: pd.DataFrame,
                     system: str,
                     outage_info,
                     gps_span: pd.Timedelta = pd.Timedelta('2h'),
                     bds_span: pd.Timedelta = pd.Timedelta('2h'),
                     gal_span: pd.Timedelta = pd.Timedelta('10min')) -> pd.DataFrame:
        """
                Returns a copy of obs_df with additional outage columns.
                  • GPS : 'gps_outage'  (bool)
                  • GAL : 'E5a_DVS_outage', 'E5a_SHS_outage', 'E5b_DVS_outage', 'E5b_SHS_outage'

                obs_df      – MultiIndex(time, sv) sorted in ascending order
                outage_info – see description in the question
                """
        obs = obs_df.reset_index().sort_values('time')

        # ------------------------------------------------------------------ GPS
        if system.upper() in {'G', 'C'}:
            outage_col = 'gps_outage' if system.upper() == 'G' else 'bds_outage'
            span = gps_span if system.upper() == 'G' else bds_span
            # --> long‑form: time, sv
            rows = [{'time': t, 'sv': sv} for sv, ts in outage_info for t in ts]
            if not rows:
                obs_df[outage_col] = False
                return obs_df
            bad = (pd.DataFrame(rows)
                   .sort_values('time')
                   .assign(dummy=True))

            merged = pd.merge_asof(obs, bad, on='time', by='sv',
                                   direction='backward', tolerance=span)
            obs_df = obs_df.copy()
            obs_df[outage_col] = merged['dummy'].notna().to_numpy()
            return obs_df

        # -------------------------------------------------------------- GALILEO
        if system.upper() == 'E':
            flag_cols = ['E5a_DVS', 'E5a_SHS', 'E5b_DVS', 'E5b_SHS']
            # long‑form: time, sv, flag
            flat = [{'time': t, 'sv': sv, 'flag': f}
                    for sv, flags in outage_info.items()
                    for f, ts in flags.items() for t in ts]
            if not flat:
                for f in flag_cols:
                    obs_df[f + '_outage'] = False
                return obs_df

            bad = (pd.DataFrame(flat)
                   .sort_values('time')
                   .assign(dummy=True))

            obs_out = obs.copy()
            for fl in flag_cols:
                sub = bad[bad['flag'] == fl].drop('flag', axis=1)
                if sub.empty:
                    obs_out[fl + '_outage'] = False
                    continue
                sub = sub.sort_values('time')
                m = pd.merge_asof(obs, sub, on='time', by='sv',
                                  direction='backward', tolerance=gal_span)
                obs_out[fl + '_outage'] = m['dummy'].notna().to_numpy()

            cols_to_add = [c for c in obs_out.columns if c.endswith('_outage')]
            obs_df = obs_df.copy()
            obs_df[cols_to_add] = obs_out[cols_to_add].values
            return obs_df

        # ------------------------------------------------------------
        raise ValueError(f"Unknown sys: {system}")


def parse_sinex(sinex_path):
    """
    Reading SINEX files
    :param sinex_path: path to sinex file
    :return: pandas.DataFrame with SINEX data
    """
    with open(sinex_path,'r') as file:
        lines=file.readlines()
        result = {}
        for line in lines:
            if 'STAX' in line or 'STAY' in line or 'STAZ' in line:
                lsp = line.split()
                result[(lsp[2] ,lsp[1])] = np.float64(lsp[-2])
    s = pd.Series(result)
    sinex = s.unstack()
    return sinex


def read_sp3(path: str, sys=('G','E','C')) -> pd.DataFrame:
    """
        Input
        -------
        path to sp3

        Output
        -------
        MultiIndex (time, sv)
        columns:   x, y, z, clock, dclock, sat, toc, epoch
        """
    empty = pd.DataFrame(
        columns=['x', 'y', 'z', 'clk', 'dclk', 'sat', 'toc', 'epoch'],
        index=pd.MultiIndex.from_arrays([[], []], names=['time', 'sv']),
    )

    df = gr.load_sp3(path, outfn=None).to_dataframe()
    df['sat'] = df.index.get_level_values('sv').tolist()
    df = df[df['sat'].str.startswith(sys)]
    if df.empty:
        return empty
    df['epoch'] = df.index.get_level_values('time')
    df['toc'] = datetime2toc(df['epoch'].tolist())
    # --- 1.   (x, y, z)  -------------------------------
    xyz = df['position'].unstack('ECEF')          # kolumny x,y,z

    # --- 2.  clock & dclock  ----
    rest = df.xs('x', level='ECEF')[['clock', 'dclock','sat','toc','epoch']]

    # --- 3.  merge data  ----------------------------------------
    out = pd.concat([xyz, rest], axis=1)


    out = out[['x', 'y', 'z', 'clock', 'dclock','sat','toc','epoch']]
    out = out.rename(columns={'clock':'clk','dclock':'dclk'})
    out = out[out['clk']!=999999.999999]
    return out
