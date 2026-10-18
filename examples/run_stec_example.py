"""Run a slant TEC (STEC) example on the repository sample data.

This script computes the levelled slant TEC of one station for GPS (`L1L2`),
Galileo (`E1E5a`) and BeiDou (`B1IB3I`), one constellation after the other
(`TECSession` processes one system per run). The default run uses one hour of
the BRUX observations of the sample day with broadcast orbits, the CODE GIM and
an explicit bias policy, and writes one CSV per system.

STEC columns are in electrons/m^2 (divide by 1e16 for TECU); columns ending in
`_tecu` are in TECU.

Run from any directory:

    python examples/run_stec_example.py          # one hour (default)
    python examples/run_stec_example.py --full   # the whole sample day

`FULL_DAY = True` in USER SETTINGS does the same as `--full`.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from gnx_py.ionosphere import TECConfig, TECSession


# =============================================================================
# USER SETTINGS
# =============================================================================

STATION = "BRUX"
SYSTEMS = ("G", "E", "C")  # processed one after the other: "G" GPS, "E" Galileo, "C" BeiDou

GPS_FREQ = "L1L2"  # GPS signal pair of the geometry-free combinations
GAL_FREQ = "E1E5a"  # Galileo signal pair
BDS_FREQ = "B1IB3I"  # the BeiDou pair of the 2024 precise clocks and bias products

ORBIT_TYPE = "broadcast"  # "broadcast" or "precise" (the SP3 files below)
IONOSPHERE_MODEL = "gim"  # background model (the "ion" column); the GIM receiver DCB needs "gim"
ELEVATION_MASK_DEG = 30  # elevation mask [deg]

# Fallback order of the satellite/receiver biases. With RECEIVER_DCB_SOURCE = "gim" the satellite chain
# takes "gim" first wherever the IONEX receiver DCB applies (one datum per arc): the GPS and Galileo arcs use
# the GIM DCBs; the GIM has no BeiDou receiver DCB of BRUX, so BeiDou takes the satellite and station DSBs of
# the CAS product in DCB_PATH (with one warning).
BIAS_SOURCES = ("product", "gim", "config", "zero")
MISSING_BIAS_POLICY = "warn_zero"  # no bias found: "warn_zero" (warn, use zero), "zero" or "raise"
ADD_SATELLITE_DCB = True  # remove the satellite code biases
ADD_STATION_DCB = True  # remove the receiver (station) code bias
RECEIVER_DCB_SOURCE = "gim"  # receiver DCB: "gim" (IONEX), "calibrate" (against the GIM), "defined" or "none"
WRITE_FULL_OUTPUT = False  # True writes every column of the session result instead of the main ones

FULL_DAY = False  # False: the one-hour window; True (or --full): the whole sample day
TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 1, 0)]  # the default one-hour window
FULL_DAY_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 5, 0, 0)]  # the whole sample day

TECU = 1e16  # electrons/m^2


# =============================================================================
# PATHS (relative to this script, so the example runs from any directory)
# =============================================================================

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

OBS_PATH = SAMPLE_DATA / f"{STATION}00BEL_R_20240350000_01D_30S_MO.crx.gz"
NAV_PATH = SAMPLE_DATA / "BRDC00IGS_R_20240350000_01D_MN.rnx"
SP3_PATHS = [
    SAMPLE_DATA / "GRG0MGXFIN_20240340000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "GRG0MGXFIN_20240350000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "GRG0MGXFIN_20240360000_01D_05M_ORB.SP3",
]
ATX_PATH = SAMPLE_DATA / "igs20.atx"
DCB_PATH = SAMPLE_DATA / "CAS1OPSRAP_20240350000_01D_01D_DCB.BIA"
GIM_PATH = SAMPLE_DATA / "COD0OPSFIN_20240350000_01D_01H_GIM.INX"


# =============================================================================
# CONFIG
# =============================================================================


def full_day() -> bool:
    """FULL_DAY or the --full flag (--quick, the default one-hour window, is accepted as well)."""
    parser = argparse.ArgumentParser(description="GNX-py STEC example on the sample data.")
    parser.add_argument("--full", action="store_true", help="process the whole sample day (default: one hour)")
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return FULL_DAY or args.full


def build_config(system: str, time_limit) -> TECConfig:
    """Build a user-editable STEC/TEC configuration."""

    return TECConfig(
        obs_path=OBS_PATH,
        nav_path=NAV_PATH,
        sp3_path=SP3_PATHS,
        atx_path=ATX_PATH,
        dcb_path=DCB_PATH,
        gim_path=GIM_PATH,
        sys=system,
        gps_freq=GPS_FREQ,
        gal_freq=GAL_FREQ,
        bds_freq=BDS_FREQ,
        orbit_type=ORBIT_TYPE,
        ionosphere_model=IONOSPHERE_MODEL,
        time_limit=time_limit,
        day_of_year=35,
        station_name=STATION,
        screen=False,
        ev_mask=ELEVATION_MASK_DEG,
        windup=True,
        rel_path=True,
        sat_pco="los",  # satellite antenna offsets along the line of sight (as in the ionosphere tutorial)
        rec_pco=True,
        add_dcb=ADD_SATELLITE_DCB,
        add_sta_dcb=ADD_STATION_DCB,
        rcv_dcb_source=RECEIVER_DCB_SOURCE,
        stec_bias_sources=BIAS_SOURCES,
        stec_missing_bias=MISSING_BIAS_POLICY,
        compare_models=False,
        leveling_ws=None,  # None levels each continuous arc as a whole; an integer gives a window of samples
        median_leveling_ws=None,
        min_arc_len=2,
    )


# =============================================================================
# RUN
# =============================================================================


def main() -> None:
    full = full_day()
    time_limit = FULL_DAY_TIME_LIMIT if full else TIME_LIMIT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for system in SYSTEMS:
        started = time.perf_counter()
        tec = TECSession(config=build_config(system, time_limit), use_sys=system).run()
        elapsed = time.perf_counter() - started

        # ---------------------------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------------------------

        output_path = OUTPUT_DIR / f"stec_{STATION}_{system}_{'24h' if full else '1h'}.csv"
        output_cols = [
            col
            for col in (
                "leveled_tec",
                "median_leveled_tec",
                "code_tec",
                "leveling_std_tecu",
                "ev",
                "az",
                "lat_ipp",
                "lon_ipp",
                "stec_sat_bias_source",
                "stec_station_bias_source",
                "stec_sat_bias_m",
                "stec_station_bias_m",
            )
            if col in tec.columns
        ]
        output_frame = tec.reset_index() if WRITE_FULL_OUTPUT else tec.reset_index()[["sv", "time", *output_cols]]
        output_frame.to_csv(output_path, index=False)

        print(f"\nSTEC {system} ({'whole day' if full else 'one hour'}) complete in {elapsed:.1f} s: {output_path}")
        print(f"rows={len(tec)} system={system} satellites={tec.index.get_level_values('sv').nunique()}")

        if "leveled_tec" in tec.columns:
            stec = tec["leveled_tec"] / TECU
            print(f"levelled STEC [TECU]: min={stec.min():.1f} median={stec.median():.1f} max={stec.max():.1f}")
        if "leveling_std_tecu" in tec.columns:
            print(f"levelling spread per arc [TECU]: median={tec['leveling_std_tecu'].median():.2f}")

        for source_col in ("stec_sat_bias_source", "stec_station_bias_source"):
            if source_col in tec.columns:
                counts = tec[source_col].value_counts(dropna=False)
                print(f"{source_col}: " + ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
