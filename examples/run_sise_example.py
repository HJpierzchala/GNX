"""Run a signal-in-space error (SISE/SISRE) orbit comparison example.

The default configuration compares BeiDou broadcast navigation against a
precise SP3 product on a short smoke window. It uses the conservative legacy
BDS `B1IB3I` mode. Modern BDS modes such as `B1C`, `B2a` or `B2b` require
matching RINEX 4 and clock-consistent bias products, so they are left as
advanced user edits.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from gnx_py.orbits import SISConfig, SISController


# =============================================================================
# USER SETTINGS
# =============================================================================

SYSTEM = "G"  # "G" for GPS, "E" for Galileo, "C" for BeiDou.
GPS_MODE = "L1L2"
GAL_MODE = "E1E5a"
BDS_MODE = "B1IB3I"  # Recommended legacy BDS MEO/IGSO validation mode.

INTERVAL_SECONDS = 300
COMPARE_DCB = False
APPLY_SATELLITE_PCO = True
APPLY_ECLIPSE = True
WRITE_DETAIL_CSV = False

RUN_SMOKE = True
SMOKE_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 2, 0)]


# =============================================================================
# PATHS
# =============================================================================

SAMPLE_DATA = REPO_ROOT / "sample_data"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"

NAV_PATH = SAMPLE_DATA / "BRDC00IGS_R_20240350000_01D_MN.rnx"
SP3_PATH = SAMPLE_DATA / "COD0MGXFIN_20240350000_01D_05M_ORB.SP3"
PREV_SP3_PATH = SAMPLE_DATA / "COD0MGXFIN_20240340000_01D_05M_ORB.SP3"
NEXT_SP3_PATH = SAMPLE_DATA / "COD0MGXFIN_20240360000_01D_05M_ORB.SP3"
ATX_PATH = SAMPLE_DATA / "igs20.atx"
DCB_PATH = SAMPLE_DATA / "CAS1OPSRAP_20240350000_01D_01D_DCB.BIA"


# =============================================================================
# CONFIG
# =============================================================================


def build_config() -> SISConfig:
    """Build a user-editable SISE/SISRE comparison configuration."""

    return SISConfig(
        orb_path_0=NAV_PATH,
        orb_path_1=SP3_PATH,
        interval=INTERVAL_SECONDS,
        prev_sp3=PREV_SP3_PATH,
        next_sp3=NEXT_SP3_PATH,
        dcb_path_0=DCB_PATH if COMPARE_DCB else None,
        dcb_path_1=DCB_PATH if COMPARE_DCB else None,
        gps_mode=GPS_MODE,
        gal_mode=GAL_MODE,
        bds_mode=BDS_MODE,
        atx_path=ATX_PATH,
        system=SYSTEM,
        tlim=SMOKE_TIME_LIMIT if RUN_SMOKE else None,
        compare_dcb=COMPARE_DCB,
        clock_bias=True,
        clock_bias_function="mean",
        apply_eclipse=APPLY_ECLIPSE,
        apply_satellite_pco=APPLY_SATELLITE_PCO,
    )


# =============================================================================
# RUN
# =============================================================================


def _summarize_by_satellite(compared: pd.DataFrame) -> pd.DataFrame:
    """Create a compact per-satellite summary for CSV output."""

    summary = compared.groupby(level="sv").agg(
        epochs=("sisre", "count"),
        dR_rms_m=("dR", lambda s: float((s.pow(2).mean()) ** 0.5)),
        dA_rms_m=("dA", lambda s: float((s.pow(2).mean()) ** 0.5)),
        dC_rms_m=("dC", lambda s: float((s.pow(2).mean()) ** 0.5)),
        clock_abs_p95_m=("dt", lambda s: float(s.abs().quantile(0.95))),
        sisre_rms_m=("sisre", lambda s: float((s.pow(2).mean()) ** 0.5)),
    )
    return summary.reset_index()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config()
    compared = SISController(config).run()
    summary = _summarize_by_satellite(compared)

    summary_path = OUTPUT_DIR / f"sise_{SYSTEM}_{BDS_MODE}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"SISE/SISRE example complete: {summary_path}")
    print(f"rows={len(compared)} satellites={compared.index.get_level_values('sv').nunique()}")
    print(summary.head(12).to_string(index=False))

    if WRITE_DETAIL_CSV:
        detail_path = OUTPUT_DIR / f"sise_{SYSTEM}_{BDS_MODE}_detail.csv"
        detail_cols = [
            col for col in ("sv", "time", "dR", "dA", "dC", "dt", "dTGD", "sisre", "sisre_orb")
            if col in compared.reset_index().columns
        ]
        compared.reset_index()[detail_cols].to_csv(detail_path, index=False)
        print(f"detail_csv={detail_path}")

    status = compared.attrs.get("clock_convention_status")
    note = compared.attrs.get("clock_convention_note")
    if status:
        print(f"clock_convention_status={status}")
        print(f"clock_convention_note={note}")


if __name__ == "__main__":
    main()
