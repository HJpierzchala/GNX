"""Run a signal-in-space error (SISE/SISRE) orbit and clock comparison example.

The default configuration compares the broadcast navigation of GPS (`L1L2`),
Galileo (`E1E5a`) and BeiDou (`B1IB3I`) with the CODE MGEX precise SP3 product
over one hour of the sample day (2024-02-04), every 300 s, one system after the
other, and writes a per-satellite summary of each system to `examples/output/`.
`B1IB3I` is the conservative BeiDou mode; BeiDou GEO satellites are skipped.
Modern BDS modes such as `B1C`, `B2a` or `B2b` require RINEX 4 navigation and
clock-consistent bias products, so they are left as advanced user edits.

Run from any directory:

    python examples/run_sise_example.py          # one hour (default)
    python examples/run_sise_example.py --full   # the whole sample day

`FULL_DAY = True` in USER SETTINGS does the same as `--full`.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from gnx_py.orbits import SISConfig, SISSession


# =============================================================================
# USER SETTINGS
# =============================================================================

SYSTEMS = ("G", "E", "C")  # compared one after the other: "G" GPS, "E" Galileo, "C" BeiDou
GPS_MODE = "L1L2"  # GPS signal mode of the comparison
GAL_MODE = "E1E5a"  # Galileo signal mode
BDS_MODE = "B1IB3I"  # Recommended legacy BDS MEO/IGSO validation mode.

INTERVAL_SECONDS = 300  # comparison step [s]
COMPARE_DCB = False  # True compares the group delays (TGD) with the bias product of DCB_PATH
APPLY_SATELLITE_PCO = True  # move the SP3 centre-of-mass orbits to the antenna phase centre (ATX_PATH)
APPLY_ECLIPSE = True  # flag the satellites in eclipse
WRITE_DETAIL_CSV = False  # True also writes every satellite/epoch comparison row

FULL_DAY = False  # False: the one-hour window; True (or --full): the whole sample day
TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 1, 0)]  # the default one-hour window
FULL_DAY_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 5, 0, 0)]  # the whole sample day


# =============================================================================
# PATHS (relative to this script, so the example runs from any directory)
# =============================================================================

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

NAV_PATH = SAMPLE_DATA / "BRDC00IGS_R_20240350000_01D_MN.rnx"
SP3_PATH = SAMPLE_DATA / "COD0MGXFIN_20240350000_01D_05M_ORB.SP3"
PREV_SP3_PATH = SAMPLE_DATA / "COD0MGXFIN_20240340000_01D_05M_ORB.SP3"
NEXT_SP3_PATH = SAMPLE_DATA / "COD0MGXFIN_20240360000_01D_05M_ORB.SP3"
ATX_PATH = SAMPLE_DATA / "igs20.atx"
DCB_PATH = SAMPLE_DATA / "CAS1OPSRAP_20240350000_01D_01D_DCB.BIA"

# signal mode of each system (used by the configuration and the output file names)
MODES = {"G": GPS_MODE, "E": GAL_MODE, "C": BDS_MODE}


# =============================================================================
# CONFIG
# =============================================================================


def full_day() -> bool:
    """FULL_DAY or the --full flag (--quick, the default one-hour window, is accepted as well)."""
    parser = argparse.ArgumentParser(description="GNX-py SISE/SISRE example on the sample data.")
    parser.add_argument("--full", action="store_true", help="compare the whole sample day (default: one hour)")
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return FULL_DAY or args.full


def build_config(system: str, time_limit) -> SISConfig:
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
        system=system,
        tlim=time_limit,
        compare_dcb=COMPARE_DCB,
        clock_bias=True,
        clock_bias_function="mean",
        apply_eclipse=APPLY_ECLIPSE,
        apply_satellite_pco=APPLY_SATELLITE_PCO,
    )


# =============================================================================
# RUN
# =============================================================================


def _rms(values: pd.Series) -> float:
    return float((values.pow(2).mean()) ** 0.5)


def _summarize_by_satellite(compared: pd.DataFrame) -> pd.DataFrame:
    """Create a compact per-satellite summary for CSV output.

    The clock column is the clock error that enters SISRE, after removing the epoch clock and group-delay
    datums: (dt - dt_mean) - (dTGD - dTGD_mean). The raw `dt` also holds the datum offset between the products.
    """

    clock_m = (compared["dt"] - compared["dt_mean"]) - (compared["dTGD"] - compared["dTGD_mean"])
    compared = compared.assign(clock_m=clock_m)
    summary = compared.groupby(level="sv").agg(
        epochs=("sisre", "count"),
        dR_rms_m=("dR", _rms),
        dA_rms_m=("dA", _rms),
        dC_rms_m=("dC", _rms),
        clock_abs_p95_m=("clock_m", lambda s: float(s.abs().quantile(0.95))),
        sisre_rms_m=("sisre", _rms),
    )
    return summary.reset_index()


def main() -> None:
    full = full_day()
    time_limit = FULL_DAY_TIME_LIMIT if full else TIME_LIMIT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for system in SYSTEMS:
        mode = MODES[system]
        started = time.perf_counter()
        compared = SISSession(build_config(system, time_limit)).run()
        elapsed = time.perf_counter() - started
        summary = _summarize_by_satellite(compared)

        label = f"sise_{system}_{mode}_{'24h' if full else '1h'}"
        summary_path = OUTPUT_DIR / f"{label}_summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\nSISE/SISRE {system} ({'whole day' if full else 'one hour'}) complete in {elapsed:.1f} s: "
              f"{summary_path}")
        print(f"rows={len(compared)} satellites={compared.index.get_level_values('sv').nunique()} "
              f"system={system} mode={mode} interval={INTERVAL_SECONDS} s")
        print(f"SISRE RMS over all satellites: {_rms(compared['sisre']):.3f} m "
              f"(orbit only: {_rms(compared['sisre_orb']):.3f} m)")
        print(summary.head(12).to_string(index=False))

        if WRITE_DETAIL_CSV:
            detail_path = OUTPUT_DIR / f"{label}_detail.csv"
            detail_cols = [
                col
                for col in ("sv", "time", "dR", "dA", "dC", "dt", "dt_mean", "dTGD", "dTGD_mean", "sisre", "sisre_orb")
                if col in compared.reset_index().columns
            ]
            compared.reset_index()[detail_cols].to_csv(detail_path, index=False)
            print(f"detail_csv={detail_path}")

        status = compared.attrs.get("clock_convention_status")  # BeiDou only
        note = compared.attrs.get("clock_convention_note")
        if status:
            print(f"clock_convention_status={status}")
            print(f"clock_convention_note={note}")


if __name__ == "__main__":
    main()
