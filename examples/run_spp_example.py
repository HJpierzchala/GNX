"""Run a small Single Point Positioning (SPP) example on sample data.

This script is designed to be opened and edited in an IDE. Change the values in
USER SETTINGS to select another station, constellation set, orbit source or
signal mode. The default smoke run uses BRUX GPS broadcast data from
`sample_data` and writes a compact CSV to `examples/output`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from gnx_py.spp import SPPConfig, SPPSession


# =============================================================================
# USER SETTINGS
# =============================================================================

STATION = "BRUX"
SYSTEMS = {"G"}  # Examples: {"G"}, {"E"}, {"C"}, {"G", "E"}, {"G", "E", "C"}
ORBIT_TYPE = "broadcast"  # "broadcast" is the simplest SPP path; "precise" needs SP3 files.

GPS_FREQ = "L1L2"
GAL_FREQ = "E1E5a"
BDS_FREQ = "B1IB3I"

RUN_SMOKE = True
SMOKE_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 5, 0, 0)]

IONOSPHERE_MODEL = "klobuchar"  # "klobuchar", "gim", "ntcm", or False.
TROPOSPHERE_MODEL = "saastamoinen"
ELEVATION_MASK_DEG = 10
TRACE_FILTER = False

MAKE_PLOT = False


# =============================================================================
# PATHS
# =============================================================================

SAMPLE_DATA = REPO_ROOT / "sample_data"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"

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


def build_config() -> SPPConfig:
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
        time_limit=SMOKE_TIME_LIMIT if RUN_SMOKE else None,
        day_of_year=35,
        station_name=STATION,
        screen=False,
        ev_mask=ELEVATION_MASK_DEG,
        use_gfz=True,
        sat_pco=False,
        rec_pco=True,
        trace_filter=TRACE_FILTER,
    )


# =============================================================================
# RUN
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config()
    result = SPPSession(config).run()

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    solution_path = OUTPUT_DIR / f"spp_{STATION}_{''.join(sorted(SYSTEMS))}_{ORBIT_TYPE}.csv"
    result.solution.to_csv(solution_path, index=False)

    print(f"SPP example complete: {solution_path}")
    print(f"solution_rows={len(result.solution)} systems={sorted(SYSTEMS)} orbit_type={ORBIT_TYPE}")

    enu_cols = [col for col in ("de", "dn", "du") if col in result.solution.columns]
    if enu_cols:
        print("Last ENU rows:")
        print(result.solution[enu_cols].tail().to_string(index=False))

    for name, residuals in {
        "gps": result.residuals_gps,
        "gal": result.residuals_gal,
        "bds": result.residuals_bds,
    }.items():
        if residuals is not None and not residuals.empty:
            print(f"{name}_residual_rows={len(residuals)}")

    if MAKE_PLOT and enu_cols:
        import matplotlib.pyplot as plt

        ax = result.solution[enu_cols].plot(title=f"SPP ENU errors: {STATION}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("meters")
        plot_path = OUTPUT_DIR / f"spp_{STATION}_{''.join(sorted(SYSTEMS))}_enu.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
