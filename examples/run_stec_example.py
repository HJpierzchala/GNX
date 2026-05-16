"""Run a small STEC / ionosphere example on sample data.

This script computes slant TEC for one station and one constellation at a time.
The default smoke run uses BRUX GPS observations with broadcast orbits, GIM
support and a conservative bias fallback policy. Change USER SETTINGS to switch
to Galileo (`SYSTEM = "E"`) or BeiDou (`SYSTEM = "C"`, usually with
`BDS_FREQ = "B1IB3I"`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from gnx_py.configuration import TECConfig
from gnx_py.ionosphere.config import TECSession


# =============================================================================
# USER SETTINGS
# =============================================================================

STATION = "BRUX"
SYSTEM = "G"  # Choose one: "G", "E", or "C". TECSession processes one system.

GPS_FREQ = "L1L2"
GAL_FREQ = "E1E5a"
BDS_FREQ = "B1IB3I"

ORBIT_TYPE = "broadcast"
IONOSPHERE_MODEL = "gim"
ELEVATION_MASK_DEG = 30

BIAS_SOURCES = ("product", "gim", "config", "zero")
MISSING_BIAS_POLICY = "warn_zero"
ADD_SATELLITE_DCB = True
ADD_STATION_DCB = True
RECEIVER_DCB_SOURCE = "gim"
WRITE_FULL_OUTPUT = False

RUN_SMOKE = True
SMOKE_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 5, 0, 0)]


# =============================================================================
# PATHS
# =============================================================================

SAMPLE_DATA = REPO_ROOT / "sample_data"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"

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


def build_config() -> TECConfig:
    """Build a user-editable STEC/TEC configuration."""

    return TECConfig(
        obs_path=OBS_PATH,
        nav_path=NAV_PATH,
        sp3_path=SP3_PATHS,
        atx_path=ATX_PATH,
        dcb_path=DCB_PATH,
        gim_path=GIM_PATH,
        sys=SYSTEM,
        gps_freq=GPS_FREQ,
        gal_freq=GAL_FREQ,
        bds_freq=BDS_FREQ,
        orbit_type=ORBIT_TYPE,
        ionosphere_model=IONOSPHERE_MODEL,
        time_limit=SMOKE_TIME_LIMIT if RUN_SMOKE else None,
        day_of_year=35,
        station_name=STATION,
        screen=False,
        ev_mask=ELEVATION_MASK_DEG,
        use_gfz=True,
        windup=True,
        rel_path=True,
        sat_pco=False,
        rec_pco=True,
        add_dcb=ADD_SATELLITE_DCB,
        add_sta_dcb=ADD_STATION_DCB,
        rcv_dcb_source=RECEIVER_DCB_SOURCE,
        stec_bias_sources=BIAS_SOURCES,
        stec_missing_bias=MISSING_BIAS_POLICY,
        compare_models=False,
        leveling_ws=20,
        median_leveling_ws=20,
        min_arc_len=2,
    )


# =============================================================================
# RUN
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config()
    tec = TECSession(config=config, use_sys=SYSTEM).run()

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    output_path = OUTPUT_DIR / f"stec_{STATION}_{SYSTEM}.csv"
    output_cols = [
        col
        for col in (
            "leveled_tec",
            "median_leveled_tec",
            "code_tec",
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

    print(f"STEC example complete: {output_path}")
    print(f"rows={len(tec)} system={SYSTEM} satellites={tec.index.get_level_values('sv').nunique()}")

    stat_cols = [col for col in ("leveled_tec", "code_tec", "ev", "lat_ipp", "lon_ipp") if col in tec.columns]
    if stat_cols:
        print("STEC statistics:")
        print(tec[stat_cols].describe().to_string())

    for source_col in ("stec_sat_bias_source", "stec_station_bias_source"):
        if source_col in tec.columns:
            print(f"{source_col}:")
            print(tec[source_col].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
