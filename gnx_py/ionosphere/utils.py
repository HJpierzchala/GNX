"""Ionospheric models and utilities for TEC estimation and related GNSS workflows.

This module aggregates model implementations and helpers used across
ionosphere-related processing, including thin-shell assumptions, mapping
functions, and empirical/analytic formulations employed in TEC computation.
The focus is on clarity and composability so that higher-level pipelines
can select and compare different modeling approaches without modifying logic.

Notes:
- All public functions/classes should document units explicitly (e.g., meters,
  kilometers, degrees, radians) and the expected array shapes or broadcasting.
- Where multiple model variants exist, prefer a single entry point with a
  parameter switch while keeping mathematical equivalence clear in the docs.
"""
import os
from typing import Optional, Iterable, Tuple, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
C = 299792458
PI = 3.1415296535898

def get_ipp(ev, az, lat, lon, ish=450.0, R=6371.0):
    """Compute ionospheric pierce point (IPP) geographic coordinates.

        Computes the latitude and longitude of the ionospheric pierce point (IPP)
        at a specified thin-shell ionosphere height using the receiver geodetic
        position and line-of-sight defined by elevation and azimuth.

        Args:
            ev (array-like of float): Elevation angles in degrees.
            az (array-like of float): Azimuth angles in degrees.
            lat (array-like of float): Receiver geodetic latitude in degrees.
            lon (array-like of float): Receiver geodetic longitude in degrees.
            ish (float, optional): Ionospheric shell height in kilometers.
                Defaults to 450.0.
            R (float, optional): Mean Earth radius in kilometers. Defaults to 6371.0.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple (lat_ipp_deg, lon_ipp_deg) where:
                - lat_ipp_deg is IPP latitude in degrees, clipped to [-90, 90].
                - lon_ipp_deg is IPP longitude in degrees, normalized to [-180, 180].

        Notes:
            - Inputs are broadcast using NumPy semantics; the result shapes follow
              NumPy broadcasting rules based on the inputs.
            - Elevation and azimuth are assumed to follow the local topocentric
              frame at the receiver location.
        """

    ev = np.array(ev, dtype=float)  # Upewnij się, że `ev` jest tablicą typu float
    az = np.array(az, dtype=float)  # Upewnij się, że `az` jest tablicą typu float
    ev = np.deg2rad(ev)
    az = np.deg2rad(az)

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    wi_pp = (PI / 2) - ev - np.arcsin((R / (R + ish)) * np.cos(ev))
    fi_pp = np.arcsin(np.sin(lat) * np.cos(wi_pp) + np.cos(lat) * np.sin(wi_pp) * np.cos(az))
    la_pp = lon + np.arcsin(((np.sin(wi_pp) * np.sin(az)) / np.cos(fi_pp)))

    # Konwersja wyników na stopnie
    fi_pp = np.rad2deg(fi_pp)
    la_pp = np.rad2deg(la_pp)
    # Normalize latitude (-90 to 90)
    fi_pp = np.clip(fi_pp, -90, 90)

    # Normalize longitude (-180 to 180)
    la_pp = (la_pp + 180) % 360 - 180

    return fi_pp, la_pp


def get_local_time(la_pp, ut):
    """Compute local time (solar) from longitude and UTC time.

        Converts a UTC timestamp to local solar time at the specified longitude.

        Args:
            la_pp (float or array-like of float): Longitude in degrees (east positive).
            ut: UTC timestamp-like object exposing hour, minute, and second attributes
                (e.g., datetime.datetime in UTC).

        Returns:
            float or np.ndarray: Local time in decimal hours. The value is not wrapped
            to [0, 24); apply modulo 24 externally if needed.

        Notes:
            - The conversion uses 15 degrees per hour (360°/24 h).
            - If an array of longitudes is provided, NumPy broadcasting applies.
        """

    # Convert longitude from radians to degrees and compute local time
    uth = ut.hour + ut.minute / 60 + ut.second / 3600
    lt = uth + la_pp / 15
    return lt


def stec_mf(ev, ish=450e03, R=6371e03, no=1):
    """Compute slant TEC mapping function (thin-shell ionosphere).

    Implements several mapping function (MF) formulations to convert between
    vertical and slant TEC under a single-layer ionosphere assumption.

    Args:
        ev (array-like of float): Elevation angles in degrees.
        ish (float, optional): Ionospheric shell height in meters. Defaults to 450e03.
        R (float, optional): Mean Earth radius in meters. Defaults to 6371e03.
        no (int, optional): Mapping function variant selector. Must be one of:
            1: Empirical variant using sin(z') with scale 0.9782.
            2: Standard thin-shell geometry using zenith angle at IPP.
            3: Alternative form based on cosine of elevation.
            4: Secant of zenith angle at receiver approximation (1 / cos(z)).

    Returns:
        np.ndarray: Mapping function values (dimensionless), broadcast to the shape
        of the input elevation array.

    Raises:
        AssertionError: If `no` is not in {1, 2, 3, 4}.

    Notes:
        - The mapping function (MF) is typically multiplied by vertical TEC (VTEC)
          to obtain slant TEC (STEC), i.e., STEC ≈ MF · VTEC.
        - Units must be consistent: this function expects `ish` and `R` in meters.
        - Input elevation is converted to radians internally.
    """

    ev = np.array(ev, dtype=float)
    ev = np.deg2rad(ev)
    assert no in [1, 2, 3, 4]
    if no == 1:
        sinz = (R / (R + ish)) * np.sin(0.9782 * ((PI / 2) - ev))
        mf = 1 / (np.sqrt(1 - (sinz ** 2)))
    elif no == 2:
        sinzip = R / (R + ish) * np.sin(np.pi / 2 - ev)
        zips = np.arcsin(sinzip)
        mf = 1 / np.cos(zips)
    elif no == 3:
        mf = (1 - ((R * np.cos(ev) / (R + ish)) ** 2)) ** (-0.5)
    elif no == 4:
        mf = 1 / np.cos(np.pi / 2 - ev)

    return mf

def load_stec_folder(
    folder: os.PathLike | str,
    *,
    file_suffix: str = "STEC.parquet.gzip",
    min_elev_deg: float = 10.0,
    station_name_len: int = 6,
    unique_id_len: int = 4,
    network_filter: Optional[Iterable[str]] = None,
    required_columns: Tuple[str, str, str, str] = ("lat_ipp", "lon_ipp", "leveled_tec", "ev","sv","time"),
    quiet: bool = False,
    skip_negative:bool=True
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Load and validate STEC parquet files from a folder, filtering by elevation
    and rejecting files that contain negative leveled TEC values.

    The function iterates over files ending with ``file_suffix`` inside ``folder``.
    For each file:
      1) Extracts a station name (first ``station_name_len`` characters of the filename)
         and a short unique ID (first ``unique_id_len`` characters).
      2) Reads the parquet into a DataFrame.
      3) Keeps only rows with elevation ``ev > min_elev_deg`` and only the columns
         specified by ``required_columns``.
      4) Rejects the entire file if any ``leveled_tec`` is negative.
      5) Appends an extra column ``name`` with the station name and collects results.

    Parameters
    ----------
    folder : os.PathLike | str
        Path to the directory containing STEC parquet files.
    file_suffix : str, optional
        File name suffix to match (default: ``"STEC.parquet.gzip"``).
    min_elev_deg : float, optional
        Minimum elevation threshold in degrees; rows with ``ev <= min_elev_deg`` are dropped.
    station_name_len : int, optional
        Number of leading characters used as the station name (default: 6).
        Assumes filenames begin with a station identifier (e.g., ``"ABCD01…"``).
    unique_id_len : int, optional
        Number of leading characters used as the short unique station ID (default: 4).
    network_filter : Iterable[str] | None, optional
        If provided, keep only files whose short unique ID is in this set/list.
        (Comparison is case-sensitive.)
    required_columns : tuple[str, str, str, str], optional
        Columns expected in each parquet file in order: (lat, lon, tec, elevation).
        Default: ``("lat_ipp", "lon_ipp", "leveled_tec", "ev")``.
    quiet : bool, optional
        If True, suppress per-file log messages.

    Returns
    -------
    df_all : pandas.DataFrame
        Concatenated DataFrame of accepted files with columns:
        ``lat_ipp, lon_ipp, leveled_tec, ev, name``.
        If no files are accepted, an empty DataFrame with those columns is returned.
    summary : dict
        A dictionary with processing metadata:
        - ``files_scanned`` (int): total files that matched the suffix
        - ``files_accepted`` (int): files included in the final DataFrame
        - ``files_rejected`` (int): files rejected (negative TEC or errors)
        - ``stations_unique`` (List[str]): unique short IDs encountered (accepted or rejected)
        - ``stations_accepted`` (List[str]): unique short IDs among accepted files
        - ``stations_rejected`` (List[str]): unique short IDs among rejected files
        - ``rejected_files`` (List[str]): basenames of rejected files
        - ``errors`` (List[Tuple[str, str]]): (filename, error message) pairs for exceptions

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist or is not a directory.

    Notes
    -----
    * Behavior mirrors the reference snippet: any file with *any* negative
      ``leveled_tec`` is fully skipped.
    * Unlike the original code, this function **does not** divide counters by two.
      Instead, it reports counts directly and also lists unique short IDs so you
      can infer pairing at the station level without guessing.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    lat_col, lon_col, tec_col, ev_col, sv_col, time_col = required_columns
    dfs: List[pd.DataFrame] = []

    files = sorted(p for p in folder.iterdir() if p.name.endswith(file_suffix))
    files_scanned = len(files)

    stations_unique: set[str] = set()
    stations_accepted: set[str] = set()
    stations_rejected: set[str] = set()

    rejected_files: List[str] = []
    errors: List[Tuple[str, str]] = []

    for fpath in files:
        try:
            # Names as in your original logic
            name = fpath.name[:station_name_len]
            unique_id = name[:unique_id_len]
            stations_unique.add(unique_id)

            if network_filter is not None and unique_id not in network_filter:
                if not quiet:
                    print(f"[SKIP-NET] {fpath.name} (unique_id {unique_id} not in network_filter)")
                continue

            df = pd.read_parquet(fpath)

            # Validate required columns early
            missing = [c for c in (lat_col, lon_col, tec_col, ev_col) if c not in df.columns]
            if missing:
                raise KeyError(f"Missing required columns {missing} in {fpath.name}")

            # Elevation filtering and column selection
            df = df[df[ev_col] > float(min_elev_deg)]#[[lat_col, lon_col, tec_col, ev_col]]

            # Reject entire file if any negative TEC
            if skip_negative:
                if (df[tec_col] < 0).any():
                    if not quiet:
                        print(f"[REJECT-NEG] {fpath.name} — negative TEC detected")
                    rejected_files.append(fpath.name)
                    stations_rejected.add(unique_id)
                    continue

            if df.empty:
                if not quiet:
                    print(f"[SKIP-EMPTY] {fpath.name} — no rows after filtering")
                # Treat as rejected but without negative TEC
                rejected_files.append(fpath.name)
                stations_rejected.add(unique_id)
                continue
            df[tec_col]/=1e16
            # Add station name column and collect
            df = df.copy()
            df["name"] = name

            dfs.append(df)
            stations_accepted.add(unique_id)
            if not quiet:
                print(f"[ACCEPT] {fpath.name} → rows: {len(df)}")

        except Exception as e:
            errors.append((fpath.name, str(e)))
            rejected_files.append(fpath.name)
            stations_rejected.add(fpath.name[:unique_id_len])
            if not quiet:
                print(f"[ERROR] {fpath.name}: {e}")

    # Build final DataFrame
    if dfs:
        df_all = pd.concat(dfs, axis=0)
        #df_all = df_all[[lat_col, lon_col, tec_col, ev_col, "name"]]
        df_all = df_all.reset_index()
        df_all['time'] = pd.to_datetime(df_all['time'])
        df_all.set_index(['sv', 'time'], inplace=True)
    # else:
    #     df_all = pd.DataFrame(columns=[lat_col, lon_col, tec_col, ev_col, "name"])

    summary = {
        "files_scanned": files_scanned,
        "files_accepted": len(dfs),
        "files_rejected": len(rejected_files),
        "stations_unique": sorted(stations_unique),
        "stations_accepted": sorted(stations_accepted),
        "stations_rejected": sorted(stations_rejected),
        "rejected_files": rejected_files,
        "errors": errors,
    }
    return df_all, summary
