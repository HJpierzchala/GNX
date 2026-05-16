"""Run a small Precise Point Positioning (PPP) example on sample data.

This script shows a stable, conservative PPP setup using precise products from
`sample_data`. Edit USER SETTINGS to change station, constellations, combined
vs uncombined routing, ionosphere constraints, signal modes or PPP-AR settings.
PPP-AR is intentionally disabled by default because it is an advanced /
experimental validation mode in parts of the codebase.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]

from gnx_py.configuration import PPPConfig
from gnx_py.ppp.config import PPPSession


# =============================================================================
# USER SETTINGS
# =============================================================================

STATION = "BRUX00BEL"
SYSTEMS = {"G", "E"}  # Use {"C"} for BDS-only, or {"G", "E", "C"} for mixed G/E/C.
POSITIONING_MODE = "combined"  # "combined" or "uncombined".
USE_IONOSPHERIC_CONSTRAINTS = False

GPS_FREQ = "L1L2"
GAL_FREQ = "E1E5a"
BDS_FREQ = "B1IB3I"

ORBIT_TYPE = "precise"
IONOSPHERE_MODEL = "gim"
TROPOSPHERE_MODEL = "niell"
ELEVATION_MASK_DEG = 10

PPP_AR_ENABLED = False
PPP_AR_COMBINED_IF_ENABLED = False  # Advanced/experimental; keep False by default.
TRACE_FILTER = False
WRITE_RESIDUAL_TAILS = False

RUN_SMOKE = True
SMOKE_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 0, 30)]


# =============================================================================
# PATHS
# =============================================================================

SAMPLE_DATA = REPO_ROOT / "sample_data"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"

OBS_PATH = SAMPLE_DATA / f"{STATION}_R_20240350000_01D_30S_MO.crx.gz"
NAV_PATH = SAMPLE_DATA / "BRDC00IGS_R_20240350000_01D_MN.rnx"
SP3_PATHS = [
    SAMPLE_DATA / "COD0MGXFIN_20240340000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "COD0MGXFIN_20240350000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "COD0MGXFIN_20240360000_01D_05M_ORB.SP3",
]
ATX_PATH = SAMPLE_DATA / "igs20.atx"
BIAS_PATH = SAMPLE_DATA / "COD0OPSFIN_20240350000_01D_01D_OSB.BIA"
GIM_PATH = SAMPLE_DATA / "COD0OPSFIN_20240350000_01D_01H_GIM.INX"
SINEX_PATH = SAMPLE_DATA / "IGS0OPSSNX_20240350000_01D_01D_CRD.SNX"


# =============================================================================
# CONFIG
# =============================================================================


def build_config() -> PPPConfig:
    """Build a user-editable PPP configuration."""

    return PPPConfig(
        obs_path=OBS_PATH,
        nav_path=NAV_PATH,
        sp3_path=SP3_PATHS,
        atx_path=ATX_PATH,
        dcb_path=BIAS_PATH,
        gim_path=GIM_PATH,
        sinex_path=SINEX_PATH,
        sys=SYSTEMS,
        gps_freq=GPS_FREQ,
        gal_freq=GAL_FREQ,
        bds_freq=BDS_FREQ,
        orbit_type=ORBIT_TYPE,
        positioning_mode=POSITIONING_MODE,
        ionosphere_model=IONOSPHERE_MODEL,
        troposphere_model=TROPOSPHERE_MODEL,
        use_iono_constr=USE_IONOSPHERIC_CONSTRAINTS,
        use_iono_rms=False,
        time_limit=SMOKE_TIME_LIMIT if RUN_SMOKE else None,
        day_of_year=35,
        station_name=STATION,
        screen=False,
        ev_mask=ELEVATION_MASK_DEG,
        use_gfz=True,
        sat_pco="los",
        rec_pco=True,
        min_arc_len=10,
        trace_filter=TRACE_FILTER,
        pppar_enabled=PPP_AR_ENABLED,
        pppar_combined_if_ar_enabled=PPP_AR_COMBINED_IF_ENABLED,
    )


# =============================================================================
# RUN
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config()
    result = PPPSession(config).run()

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    system_label = "".join(sorted(SYSTEMS))
    solution_path = OUTPUT_DIR / f"ppp_{STATION}_{system_label}_{POSITIONING_MODE}.csv"
    result.solution.to_csv(solution_path)

    print(f"PPP example complete: {solution_path}")
    print(
        f"solution_rows={len(result.solution)} systems={sorted(SYSTEMS)} "
        f"mode={POSITIONING_MODE} convergence={result.convergence}"
    )

    summary_cols = [
        col
        for col in ("de", "dn", "du", "dtr", "ztd", "n_sats", "n_sats_G", "n_sats_E", "n_sats_C")
        if col in result.solution.columns
    ]
    if summary_cols:
        print("Last solution rows:")
        print(result.solution[summary_cols].tail().to_string())

    for name, residuals in {
        "gps": result.residuals_gps,
        "gal": result.residuals_gal,
        "bds": result.residuals_bds,
    }.items():
        if residuals is not None and not residuals.empty:
            message = f"{name}_residual_rows={len(residuals)}"
            if WRITE_RESIDUAL_TAILS:
                residual_path = OUTPUT_DIR / f"ppp_{STATION}_{system_label}_{name}_residuals_tail.csv"
                residuals.tail(100).to_csv(residual_path)
                message += f" tail_csv={residual_path}"
            print(message)


if __name__ == "__main__":
    main()
