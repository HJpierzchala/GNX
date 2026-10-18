"""Run a Precise Point Positioning (PPP) example on the repository sample data.

This script is designed to be opened and edited in an IDE. The default run
processes one hour of station BRUX on the sample day (2024-02-04, 30 s) with
GPS + Galileo + BeiDou: uncombined PPP (one code and one phase per frequency)
with the GIM as ionospheric constraint, static coordinates and PPP ambiguity
resolution (PPP-AR). Orbits, clocks and code/phase biases (OSB) come from one
solution, CNES/CLS GRG, so the phase biases fit the clocks and ambiguities can
be fixed. The solution and the residuals are written to `examples/output/`.

Run from any directory:

    python examples/run_ppp_example.py          # one hour (default)
    python examples/run_ppp_example.py --full   # the whole sample day

`FULL_DAY = True` in USER SETTINGS does the same as `--full`.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from gnx_py.ppp import PPPConfig, PPPSession


# =============================================================================
# USER SETTINGS
# =============================================================================

STATION = "BRUX"
SYSTEMS = {"G", "E", "C"}  # Examples: {"G"}, {"G", "E"}, {"G", "E", "C"}
POSITIONING_MODE = "uncombined"  # "uncombined" (frequency-specific observations) or "combined" (ionosphere-free)
USE_IONOSPHERIC_CONSTRAINTS = True  # uncombined only: the GIM slant ionosphere as a pseudo-observation

GPS_FREQ = "L1L2"  # GPS signal pair
GAL_FREQ = "E1E5a"  # Galileo signal pair
BDS_FREQ = "B1IB3I"  # the BeiDou mode of the precise clocks and the GRG biases of 2024

# Model of the ionospheric constraints: "gim", "klobuchar", "ntcm" or "bdgim"; "bdgim" needs a RINEX 4 navigation
# file: NAV_PATH = SAMPLE_DATA / "BRD400DLR_S_20240350000_01D_MN.rnx". (The troposphere is always Niell in PPP.)
IONOSPHERE_MODEL = "gim"
ELEVATION_MASK_DEG = 10  # elevation mask [deg]

PPP_AR_ENABLED = True  # uncombined routes only; needs the phase OSBs of the orbit/clock solution (BIAS_PATH)
TRACE_FILTER = False  # True prints compact [PPP TRACE] diagnostics of the filter
WRITE_RESIDUALS = True  # one residual CSV per system
MAKE_PLOT = False  # PNG of the solution time series (gnx_py.viz) next to the CSV

FULL_DAY = False  # False: the one-hour window; True (or --full): the whole sample day
TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 1, 0)]  # the default one-hour window
FULL_DAY_TIME_LIMIT = [datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 5, 0, 0)]  # the whole sample day


# =============================================================================
# PATHS (relative to this script, so the example runs from any directory)
# =============================================================================

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

OBS_PATH = SAMPLE_DATA / "BRUX00BEL_R_20240350000_01D_30S_MO.crx.gz"
NAV_PATH = SAMPLE_DATA / "BRDC00IGS_R_20240350000_01D_MN.rnx"
# orbits and clocks of the previous, the processed and the next day (interpolation at the day boundaries)
SP3_PATHS = [
    SAMPLE_DATA / "GRG0MGXFIN_20240340000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "GRG0MGXFIN_20240350000_01D_05M_ORB.SP3",
    SAMPLE_DATA / "GRG0MGXFIN_20240360000_01D_05M_ORB.SP3",
]
# Bias products (one path or a list). The GRG0MGXFIN OSB holds the code and phase biases of G, E and C of the
# same solution as the GRG orbits and clocks, which PPP-AR needs (sample_data has no COD MGEX OSB; COD0OPSFIN OSB
# has no BeiDou and is another solution than COD0MGXFIN, so PPP-AR would be switched off). The CAS DSB adds the
# BeiDou C2I-C6I code bias that refers the GIM constraint to the code-bias datum (without it: a warning).
BIAS_PATH = [
    SAMPLE_DATA / "GRG0MGXFIN_20240350000_01D_01D_OSB.BIA",
    SAMPLE_DATA / "CAS1OPSRAP_20240350000_01D_01D_DCB.BIA",
]
GIM_PATH = SAMPLE_DATA / "COD0OPSFIN_20240350000_01D_01H_GIM.INX"  # ionospheric constraints
ATX_PATH = SAMPLE_DATA / "igs20.atx"
SINEX_PATH = SAMPLE_DATA / "IGS0OPSSNX_20240350000_01D_01D_CRD.SNX"  # reference coordinates for de/dn/du


# =============================================================================
# CONFIG
# =============================================================================


def full_day() -> bool:
    """FULL_DAY or the --full flag (--quick, the default one-hour window, is accepted as well)."""
    parser = argparse.ArgumentParser(description="GNX-py PPP example on the sample data.")
    parser.add_argument("--full", action="store_true", help="process the whole sample day (default: one hour)")
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return FULL_DAY or args.full


def build_config(time_limit) -> PPPConfig:
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
        orbit_type="precise",
        positioning_mode=POSITIONING_MODE,
        use_iono_constr=USE_IONOSPHERIC_CONSTRAINTS,
        ionosphere_model=IONOSPHERE_MODEL,
        time_limit=time_limit,
        day_of_year=35,
        station_name=STATION,
        screen=False,
        ev_mask=ELEVATION_MASK_DEG,
        sat_pco="los",  # satellite antenna offsets along the line of sight
        rec_pco=True,
        min_arc_len=10,
        pppar_enabled=PPP_AR_ENABLED,
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
    result = PPPSession(build_config(time_limit)).run()
    elapsed = time.perf_counter() - started

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    systems = "".join(s for s in "GEC" if s in SYSTEMS)  # e.g. "GEC"
    label = f"ppp_{STATION}_{systems}_{POSITIONING_MODE}_{'24h' if full else '1h'}"
    solution = result.solution
    solution_path = OUTPUT_DIR / f"{label}.csv"
    solution.to_csv(solution_path)  # epochs in the index; x, y, z [m, ECEF], de, dn, du [m], ct_min [min], ...

    print(f"PPP example ({'whole day' if full else 'one hour'}) complete in {elapsed:.1f} s: {solution_path}")
    uncombined = POSITIONING_MODE == "uncombined"
    print(f"epochs={len(solution)} systems={systems} mode={POSITIONING_MODE} "
          f"ionospheric_constraints={USE_IONOSPHERIC_CONSTRAINTS and uncombined} "
          f"ppp_ar={PPP_AR_ENABLED and uncombined}")

    enu_cols = ["de", "dn", "du"]
    ct_min = float(solution["ct_min"].iloc[-1]) if len(solution) else float("nan")
    if np.isfinite(ct_min):
        print(f"convergence time: {ct_min:.1f} min")
    else:
        print("convergence time: not converged in this window")
    if len(solution) and solution[enu_cols].notna().all(axis=None):
        err3d = np.sqrt((solution[enu_cols] ** 2).sum(axis=1))
        final = solution[enu_cols].iloc[-1]
        print("final error against the SINEX coordinates [m]: "
              + " ".join(f"{col}={final[col]:.3f}" for col in enu_cols) + f" 3D={err3d.iloc[-1]:.3f}")
        if np.isfinite(ct_min):
            after = err3d[err3d.index >= solution.index[0] + np.timedelta64(int(round(ct_min * 60)), "s")]
            print(f"3D RMS after convergence: {np.sqrt((after ** 2).mean()):.3f} m ({len(after)} epochs)")
    if PPP_AR_ENABLED and uncombined and "ar_fixed" in solution.columns:
        fixed = solution.index[solution["ar_fixed"] > 0]
        first_fix = f", first after {(fixed[0] - solution.index[0]).total_seconds() / 60:.1f} min" if len(fixed) else ""
        print(f"epochs with fixed ambiguities: {len(fixed)} of {len(solution)}{first_fix}")

    for name, residuals in {
        "gps": result.residuals_gps,
        "gal": result.residuals_gal,
        "bds": result.residuals_bds,
    }.items():
        if residuals is not None and not residuals.empty:
            message = f"{name}: residual_rows={len(residuals)}"
            if WRITE_RESIDUALS:
                residual_path = OUTPUT_DIR / f"{label}_{name}_residuals.csv"
                residuals.to_csv(residual_path)
                message += f" csv={residual_path.name}"
            print(message)

    if MAKE_PLOT and len(solution):
        from gnx_py import viz

        plot_path = OUTPUT_DIR / f"{label}.png"
        viz.plot_positioning_series(
            solution,
            convergence_time=ct_min if np.isfinite(ct_min) else None,
            title=f"PPP: {STATION} {systems}",
            save_path=str(plot_path),
            close=True,
        )
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
