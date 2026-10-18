"""Run a Single Point Positioning (SPP) example on the repository sample data.

This script is designed to be opened and edited in an IDE. Change the values in
USER SETTINGS to select another constellation set, orbit source, signal mode or
time window. The default run processes one hour of station BRUX on the sample
day (2024-02-04, 30 s) with GPS + Galileo + BeiDou (ionosphere-free code of
L1/L2, E1/E5a and B1I/B3I, broadcast orbits) and writes the solution to
`examples/output/`.

Run from any directory:

    python examples/run_spp_example.py          # one hour (default)
    python examples/run_spp_example.py --full   # the whole sample day

`FULL_DAY = True` in USER SETTINGS does the same as `--full`.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from gnx_py.spp import SPPConfig, SPPSession


# =============================================================================
# USER SETTINGS
# =============================================================================

STATION = "BRUX"
SYSTEMS = {"G", "E", "C"}  # Examples: {"G"}, {"E"}, {"C"}, {"G", "E"}, {"G", "E", "C"}
ORBIT_TYPE = "broadcast"  # "broadcast" is the simplest SPP path; "precise" uses the SP3 files below.

GPS_FREQ = "L1L2"  # GPS signals: "L1L2" (ionosphere-free) or one frequency, e.g. "L1"
GAL_FREQ = "E1E5a"  # Galileo signals: "E1E5a" (ionosphere-free) or one frequency, e.g. "E1"
BDS_FREQ = "B1IB3I"  # BeiDou signals: "B1IB3I" (ionosphere-free) or one frequency, e.g. "B1I"

FULL_DAY = False  # False: the one-hour window; True (or --full): the whole sample day
TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 1, 0)]  # the default one-hour window
FULL_DAY_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 5, 0, 0)]  # the whole sample day

# Used only by single-frequency modes (for example GPS_FREQ = "L1", "E1", "B1I"); the ionosphere-free
# default modes (L1L2, E1E5a, B1IB3I) do not use it. "klobuchar", "gim", "ntcm", "bdgim" or False;
# "bdgim" needs a RINEX 4 navigation file:
# NAV_PATH = SAMPLE_DATA / "BRD400DLR_S_20240350000_01D_MN.rnx".
IONOSPHERE_MODEL = "klobuchar"
TROPOSPHERE_MODEL = "saastamoinen"  # "saastamoinen", "collins" or "niell".
ELEVATION_MASK_DEG = 10  # elevation mask [deg]
TRACE_FILTER = False  # True prints one diagnostic line per epoch (errors, residual RMS)

MAKE_PLOT = False  # ENU error plot (PNG) next to the CSV


# =============================================================================
# PATHS (relative to this script, so the example runs from any directory)
# =============================================================================

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

OBS_PATH = SAMPLE_DATA / f"{STATION}00BEL_R_20240350000_01D_30S_MO.crx.gz"
NAV_PATH = SAMPLE_DATA / "BRDC00IGS_R_20240350000_01D_MN.rnx"
SP3_PATHS = [
    SAMPLE_DATA / "COD0MGXFIN_20240340000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "COD0MGXFIN_20240350000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "COD0MGXFIN_20240360000_01D_05M_ORB.SP3",
]
ATX_PATH = SAMPLE_DATA / "igs20.atx"
DCB_PATH = SAMPLE_DATA / "CAS1OPSRAP_20240350000_01D_01D_DCB.BIA"
GIM_PATH = SAMPLE_DATA / "COD0OPSFIN_20240350000_01D_01H_GIM.INX"
SINEX_PATH = SAMPLE_DATA / "IGS0OPSSNX_20240350000_01D_01D_CRD.SNX"


# =============================================================================
# CONFIG
# =============================================================================


def full_day() -> bool:
    """FULL_DAY or the --full flag (--quick, the default one-hour window, is accepted as well)."""
    parser = argparse.ArgumentParser(description="GNX-py SPP example on the sample data.")
    parser.add_argument("--full", action="store_true", help="process the whole sample day (default: one hour)")
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return FULL_DAY or args.full


def build_config(time_limit) -> SPPConfig:
    """Build a user-editable SPP configuration."""

    return SPPConfig(
        obs_path=OBS_PATH,
        nav_path=NAV_PATH,
        sp3_path=SP3_PATHS,
        atx_path=ATX_PATH,
        dcb_path=DCB_PATH,
        gim_path=GIM_PATH,
        sinex_path=SINEX_PATH,
        sys=SYSTEMS,
        gps_freq=GPS_FREQ,
        gal_freq=GAL_FREQ,
        bds_freq=BDS_FREQ,
        orbit_type=ORBIT_TYPE,
        ionosphere_model=IONOSPHERE_MODEL,
        troposphere_model=TROPOSPHERE_MODEL,
        time_limit=time_limit,
        day_of_year=35,
        station_name=STATION,
        screen=False,
        ev_mask=ELEVATION_MASK_DEG,
        # satellite PCO along the line of sight; with broadcast orbits no satellite PCO is applied
        sat_pco="los",
        rec_pco=True,
        trace_filter=TRACE_FILTER,
    )


# =============================================================================
# RUN
# =============================================================================


def main() -> None:
    full = full_day()
    time_limit = FULL_DAY_TIME_LIMIT if full else TIME_LIMIT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result = SPPSession(build_config(time_limit)).run()
    elapsed = time.perf_counter() - started

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    systems = "".join(s for s in "GEC" if s in SYSTEMS)  # e.g. "GEC"
    label = f"spp_{STATION}_{systems}_{ORBIT_TYPE}_{'24h' if full else '1h'}"
    solution_path = OUTPUT_DIR / f"{label}.csv"
    solution = result.solution
    solution.to_csv(solution_path, index=False)

    print(f"SPP example ({'whole day' if full else 'one hour'}) complete in {elapsed:.1f} s: {solution_path}")
    print(f"epochs={len(solution)} systems={systems} orbit_type={ORBIT_TYPE} "
          f"ionosphere={IONOSPHERE_MODEL} troposphere={TROPOSPHERE_MODEL}")

    enu_cols = [col for col in ("de", "dn", "du") if col in solution.columns]
    if enu_cols and len(solution):
        rms = np.sqrt((solution[enu_cols] ** 2).mean())
        rms_3d = float(np.sqrt((rms ** 2).sum()))
        print("RMS of the errors against the SINEX coordinates [m]: "
              + " ".join(f"{col}={rms[col]:.3f}" for col in enu_cols) + f" 3D={rms_3d:.3f}")

    for name, residuals in {
        "gps": result.residuals_gps,
        "gal": result.residuals_gal,
        "bds": result.residuals_bds,
    }.items():
        if residuals is not None and not residuals.empty:
            rejected = int(residuals["spp_rejected"].sum()) if "spp_rejected" in residuals else 0
            print(f"{name}: residual_rows={len(residuals)} not_used={rejected}")

    if MAKE_PLOT and enu_cols and len(solution):
        from gnx_py import viz

        plot_path = OUTPUT_DIR / f"{label}_enu.png"
        viz.plot_spp_positioning(
            solution,
            columns=enu_cols,
            show_satellites=False,
            title=f"SPP ENU errors: {STATION}",
            save_path=plot_path,
            close=True,
        )
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
