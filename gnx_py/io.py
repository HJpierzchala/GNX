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
        df = df.apply(lambda x: pd.to_numeric(x)).replace(self.drop_sentinel, pd.NA)
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
    gpsa: float | None = None
    gpsb: float | None = None
    gala: float | None = None


@dataclass
class ObsData:
    """
    Obs data container
    """
    gps: pd.DataFrame = None
    gal: pd.DataFrame = None
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
                 galileo_modes=None, use_gfz=False):
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
            self.sys = sys
        self.galileo_modes = galileo_modes

        self.dcb_type = None
        if self.dcb_path is not None:
            osb_path = self.dcb_path.split('/')[-1].split('_')

            self.dcb_type = osb_path[3]
            self.add_dcb = True
        else:
            self.add_dcb = False

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

        df = df.rename(columns={'BIASSTART_dt': 'time', 'PRN': 'sv'})

        if file_type == "OSB":
            wide = (
                df.pivot_table(
                    index=["sv", "time"],
                    columns="OBS1",
                    values="ESTIMATEDVALUE",
                    aggfunc="first"
                )
                .add_prefix("OSB_")  # BIAS_C1C, BIAS_C1W, ...
                .reset_index()
            )
            file_type = 'OSB'
        else:  # file_type == "DSB"
            # Tworzenie nowej kolumny z połączonymi OBS1 i OBS2
            df["OBS_PAIR"] = df["OBS1"] + "_" + df["OBS2"]
            wide = (
                df.pivot_table(
                    index=["sv", "time"],
                    columns="OBS_PAIR",
                    values="ESTIMATEDVALUE",
                    aggfunc="first"
                )
                .add_prefix("BIAS_")  # BIAS_C1C_C1W, BIAS_C1C_C2W, ...
                .reset_index()
            )
            file_type = 'DSB'

        wide = wide.set_index(['sv', 'time'])
        valid_systems = ('G', 'E')  # GPS, Galileo
        wide = wide[wide.index.get_level_values('sv').str.startswith(valid_systems)]
        wide.columns.name = None

        return wide.fillna(0.0), file_type

    def process_observations2(self):
        (sta_name, ant_type, ant_cover, ant_h, approx_pos, pos_geod, interval, gps_obs, gal_obs,
         time_of_first_obs, time_of_last_obs, phase_shift_dict, num_obs_dict) = self.obs_header_reader()

        info = (sta_name, ant_type, ant_cover, ant_h, approx_pos, pos_geod, interval, gps_obs, gal_obs,
                time_of_first_obs, time_of_last_obs, phase_shift_dict)

        return info

    from pathlib import Path

    def load_and_filter(self, tlim=None, version=3.0):
        """
        Faster load + filter for GPS/Galileo.
        Main speedups:
        - avoid df[cols].dropna(...) (double-copy); use one boolean mask
        - optionally build mask on minimal "critical" columns only
        - instantiate GFZRNX2 once
        """
        results = {"G": None, "E": None}

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
            return sub.loc[mask]

        # --- GPS ---
        if "G" in self.sys:
            try:
                gps_df = _load_sys("G")
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
               and len(c) >= 3
        ]
        if not selected_columns:
            return []

        # --- 2) Precompute non-NaN counts ONCE ---
        # This is the big win: avoid dataframe[candidates].notna().sum() inside loops.
        nn = dataframe[selected_columns].notna().sum(axis=0)  # Series: col -> count

        # --- 3) Group by kind = col[:-1] ---
        by_kind = defaultdict(list)
        for c in selected_columns:
            by_kind[c[:-1]].append(c)

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

            if consistent_suffix and chosen_suffixes:
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
                suf = chosen_col[-1]
                if suf not in chosen_suffixes:
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

    def obs_header_reader(self):
        path = self.obs_path
        hdr = gr.rinexheader(path)
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
                gps_obs=gal_obs = hdr['fields']
            else:
                if 'G' in hdr['fields'].keys():
                    gps_obs = hdr['fields']['G']
                else:
                    gps_obs = None
                if 'E' in hdr['fields'].keys():
                    gal_obs = hdr['fields']['E']
                else:
                    gal_obs = None
        else:
            gps_obs = None
            gal_obs = None

        if 't0' in hdr.keys():
            time_of_first_obs = hdr['t0']
        elif 'TIME OF FIRST OBS3' in hdr.keys():
            date_list = hdr['TIME OF FIRST OBS3'].strip().split()[:-1]
            year, month, day, hour, minute, second = map(float, date_list)
            time_of_first_obs = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        else:
            time_of_first_obs = None

        if 'TIME OF LAST OBS3' in hdr.keys():
            date_list = hdr['TIME OF LAST OBS3'].strip().split()[:-1]
            year, month, day, hour, minute, second = map(float, date_list)
            time_of_last_obs = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        else:
            time_of_last_obs = None
        if time_of_first_obs is None:
            yr = int(self.obs_path.split('_')[-4][:4])
            doy = int(self.obs_path.split('_')[-4][4:7])
            time_of_first_obs = doy_to_datetime(year=yr,doy=doy)
        if time_of_last_obs is None:
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
                if sat_id.startswith(('G', 'E')):
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
        hdr = gr.rinexheader(self.nav_path)
        if 'IONOSPHERIC CORR' in hdr.keys():
            if 'GPSA' and 'GPSB' in hdr['IONOSPHERIC CORR'].keys():
                gpsa = hdr['IONOSPHERIC CORR']['GPSA']
                gpsb = hdr['IONOSPHERIC CORR']['GPSB']
            elif 'ION ALPHA' and 'ION BETA' in hdr['IONOSPHERIC CORR'].keys():
                gpsa, gpsb = hdr['IONOSPHERIC CORR']['ION ALPHA'], hdr['IONOSPHERIC CORR']['ION BETA']
            else:
                gpsa = None
                gpsb = None

            if 'GAL' in hdr['IONOSPHERIC CORR'].keys():
                gala = hdr['IONOSPHERIC CORR']['GAL']
            else:
                gala = None
            return gpsa, gpsb, gala
        else:
            print('No ionospheric coefficients for GPS and Galileo found in navigation file')
            return None, None, None

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

        def _attach_dcb(df: pd.DataFrame | None, dcb: pd.DataFrame | None) -> pd.DataFrame | None:
            """Adds DCB (OSB) to df according to self.dcb_type and fills missing OSBs with zeros."""
            if not isinstance(df, pd.DataFrame) or not isinstance(dcb, pd.DataFrame):
                return df

            if self.dcb_type == '30S':
                out = df.join(dcb, on=['sv', 'time'])
            elif self.dcb_type == '01D':
                tmp = dcb.reset_index().set_index('sv')
                out = df.join(tmp, on='sv')
            else:
                out = df

            osb_cols = out.columns[out.columns.str.startswith("OSB")]
            if len(osb_cols):
                out.loc[:, osb_cols] = out.loc[:, osb_cols].fillna(0)
            return out

        def _assert_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
            """Clears the 'time' column, if any, and checks the index."""
            if isinstance(df, pd.DataFrame):
                df = df.drop(columns='time', errors='ignore')
                assert set(df.index.names) == {'sv', 'time'}
            return df

        # ------------------------------------------------------------------------

        rinex_info = gr.rinexinfo(self.obs_path)
        rinex_version = rinex_info['version']

        try:
            interval = float(gr.rinexheader(self.obs_path)['INTERVAL'].strip()) / 60
        except KeyError:
            interval = float(self.obs_path.split('_')[-2][:2]) / 60

        obs_dict = self.load_and_filter(tlim=tlim)
        gps, gal = obs_dict.get('G'), obs_dict.get('E')

        gps_info  = self.process_observations2()
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
            satellite_pco = {**satellite_pco_e, **satellite_pco_g}
        else:
            satellite_pco = None
        # Satellite DCB
        if self.add_dcb:
            dcb_df, _ = self.read_bia(path=self.dcb_path)
        else:
            dcb_df = None
        gps = _attach_dcb(gps, dcb_df)
        gal = _attach_dcb(gal, dcb_df)

        gps = _assert_index(gps)
        gal = _assert_index(gal)

        return ObsData(
            gps=gps,
            gal=gal,
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

    def load_broadcast_orbit(self, tlim=None):
        gpsa, gpsb, gala = self.nav_header_reader()
        if 'G' in self.sys:
            gps_orb = gr.load(rinexfn=self.nav_path,
                              use={'G'},
                              tlim=tlim).dropna(dim='time', how='all').to_dataframe().dropna(how='all')
            gps_orb = gps_orb.swaplevel()
        else:
            gps_orb = None
        if 'E' in self.sys:
            try:
                gal_orb = gr.load(rinexfn=self.nav_path,
                                  use={'E'},
                                  tlim=tlim).dropna(dim='time', how='all').to_dataframe().dropna(how='all')
                # mixed columns between BGD and trans time - seems like georinex error
                # --- TEMP FIX:
                data = gal_orb.copy()
                threshold = 1e-1
                swapped_rows = (data['BGDe5a'] > threshold) | (data['BGDe5b'] > threshold )

                for idx in data.index[swapped_rows]:
                    if data.loc[idx, 'BGDe5a'] > threshold:
                        # Swap BGDe5a with TransTime
                        data.loc[idx, 'TransTime'], data.loc[idx, 'BGDe5a'] = data.loc[idx, 'BGDe5a'], data.loc[
                            idx, 'TransTime']
                    if data.loc[idx, 'BGDe5b'] > threshold:
                        # Swap BGDe5b with TransTime
                        data.loc[idx, 'TransTime'], data.loc[idx, 'BGDe5b'] = data.loc[idx, 'BGDe5b'], data.loc[
                            idx, 'TransTime']
                gal_orb = data.copy()
                # --- END OF TEMP FIX

                gal_orb = self.interpret_data_src(df=gal_orb)
                gal_orb = self.add_galileo_health_cols(df=gal_orb)
                gal_orb = gal_orb.swaplevel()
            except Exception as e:
                traceback.print_exc()
        else:
            gal_orb = None
        return OrbitData(gps_orb=gps_orb,
                         gal_orb=gal_orb,
                         gpsa=gpsa, gpsb=gpsb, gala=gala)

    def screen_navigation_message(self, broadcast_df: pd.DataFrame, system: str):
        """
                Returns a list of tuples (sv, epoch_list) for satellites
                that have at least one navigation message marked as 'uncertain'
                or 'faulty'.

                broadcast_df – MultiIndex (time, sv) + columns E5a_DVS, E5a_SHS, E5b_DVS, E5b_SHS
                sys       – 'G' (GPS) or 'E' (Galileo)
                """
        if system == 'G': # GPS
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
        if system.upper() == 'G':
            # --> long‑form: time, sv
            rows = [{'time': t, 'sv': sv} for sv, ts in outage_info for t in ts]
            if not rows:
                obs_df['gps_outage'] = False
                return obs_df
            bad = (pd.DataFrame(rows)
                   .sort_values('time')
                   .assign(dummy=True))

            merged = pd.merge_asof(obs, bad, on='time', by='sv',
                                   direction='backward', tolerance=gps_span)
            obs_df = obs_df.copy()
            obs_df['gps_outage'] = merged['dummy'].notna().to_numpy()
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


def read_sp3(path: str, sys=('G','E')) -> pd.DataFrame:
    """
        Input
        -------
        path to sp3

        Output
        -------
        MultiIndex (time, sv)
        columns:   x, y, z, clock, dclock, sat, toc, epoch
        """
    df = gr.load_sp3(path, outfn=None).to_dataframe()
    df['sat'] = df.index.get_level_values('sv').tolist()
    df = df[df['sat'].str.startswith(sys)]
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
