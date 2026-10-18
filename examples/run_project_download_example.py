"""Download the input files of a GNSS processing project into one folder.

This script is designed to be opened and edited in an IDE. It uses the download
API of GNX-py, `DownloadConfig` + `DownloadSession` from `gnx_py.downloaders`:
a project name, the dates, the stations and the product families give one
project folder with everything a GPS + Galileo + BeiDou run needs, for the
sample day of this repository (2024-02-04, station BRUX) by default:

- observations (RINEX 3/4 daily files; the EUREF archive for EPN stations);
- broadcast navigation of all systems;
- precise orbits (SP3, with the day before and after for the interpolation),
  30 s clocks and code/phase biases (OSB) of one analysis-centre solution, so
  PPP-AR gets consistent products;
- code biases (DSB, e.g. CAS) and global ionosphere maps (IONEX/GIM).

Every file is checked (whole archive, size, content header) before it is kept,
a valid file already on disk is not downloaded again, and files stay compressed
as in the archive (GNX-py reads them directly).

By default the script is a dry run without network access: it lists the files
it would download (and creates only the empty project folder under
`examples/output/projects/`, which is not committed). Set
`CHECK_AVAILABILITY = True` to ask the archives which files exist, then
`DRY_RUN = False` to download. A `CREDENTIALS_FILE` is exchanged for a token at
NASA Earthdata in every run, a dry run included (one network request).

NASA CDDIS needs an Earthdata token. Never write it into this script: set the
environment variable `EARTHDATA_TOKEN` (for example in your shell profile or the
run configuration of your IDE), or point `CREDENTIALS_FILE` to a private,
git-ignored text file with `EARTHDATA_USERNAME=...` and `EARTHDATA_PASSWORD=...`
lines, which the library exchanges for a token. Without a token the CDDIS files
are reported as `auth_missing` and the next source (BKG, CODE) is tried.

Run from any directory:

    python examples/run_project_download_example.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from gnx_py.downloaders import DownloadConfig, DownloadSession


# =============================================================================
# USER SETTINGS
# =============================================================================

PROJECT_NAME = "brux_2024_035"  # name of the project folder under OUTPUT_ROOT
DATES = [date(2024, 2, 4)]  # one date or a list of dates (calendar days, UTC)

# 4-character codes or 9-character RINEX ids. A 4-character code needs its 9-character id for the RINEX 3/4 file
# names: "BRUX" -> "BRUX00BEL" is built into the library, other stations go to STATION_ALIASES.
STATIONS = ("BRUX",)
STATION_ALIASES: dict = {}  # e.g. {"WTZR": "WTZR00DEU"}

# Product families (see gnx_py.downloaders.PRODUCT_TYPE_ALIASES for other spellings):
# obs observations, nav broadcast navigation, sp3 orbits, clk 30 s clocks, bia OSB code/phase biases,
# dcb DSB code biases, ionex global ionosphere maps.
PRODUCT_TYPES = ("obs", "nav", "sp3", "clk", "bia", "dcb", "ionex")

# Archives in priority order: "cddis" (NASA CDDIS, every product type, needs an Earthdata token), "bkg" (BKG, no
# IONEX), "code" (CODE's own products) and "euref" (EUREF Permanent GNSS Network observations, for European
# stations that are not in the IGS archives).
SOURCES = ("cddis", "bkg", "code", "euref")

# Products to try first, per type; empty: the library defaults (COD0MGXFIN, then GRG0MGXFIN, ... for orbits, clocks
# and OSB). E.g. {"sp3": ("GRG0MGXFIN",)} takes the GRG orbits, clocks and OSB of the PPP example, and
# {"nav": ("BRD400DLR",)} the RINEX 4 navigation with the BeiDou-3 BDGIM parameters.
PRODUCT_PREFERENCES: dict = {}

# None: the EARTHDATA_TOKEN environment variable; or a path to a git-ignored credentials file (exchanged for a
# token at NASA Earthdata in every run, a dry run included)
CREDENTIALS_FILE = None

DRY_RUN = True  # True: download nothing, list the files that would be downloaded
CHECK_AVAILABILITY = False  # dry run only; True asks the archives which files exist (network access)
OVERWRITE = False  # True downloads again files already valid on disk
VERBOSE = False  # True shows the INFO messages of the downloader (sources tried, product set)


# =============================================================================
# PATHS (relative to this script, so the example runs from any directory)
# =============================================================================

OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "projects"


# =============================================================================
# CONFIG
# =============================================================================


def build_config() -> DownloadConfig:
    """Build a user-editable download session configuration."""

    return DownloadConfig(
        project_name=PROJECT_NAME,
        output_root=OUTPUT_ROOT,
        dates=DATES,
        stations=STATIONS,
        station_aliases=STATION_ALIASES,
        product_types=PRODUCT_TYPES,
        sources=SOURCES,
        product_preferences=PRODUCT_PREFERENCES,
        credentials_file=CREDENTIALS_FILE,
        dry_run=DRY_RUN,
        check_availability=CHECK_AVAILABILITY,
        overwrite=OVERWRITE,
        verbose=VERBOSE,
    )


# =============================================================================
# RUN
# =============================================================================

ATTENTION = ("missing", "error", "auth_missing", "unavailable")


def main() -> None:
    config = build_config()
    report = DownloadSession(config).run()

    mode = "download" if not config.dry_run else (
        "dry run with availability checks" if config.check_availability else "dry run, no archive queries")
    print(f"Project {report.project_name} ({mode}): {report.output_dir}")
    print("Summary:", report.summary())

    choice = report.product_sets
    if choice is not None and choice.per_item:
        checked = "checked" if choice.checked else "not checked"
        print(f"Orbit/clock/bias solution: {choice.solution or 'per day'} (availability {checked})")
        if report.mixed_products:
            print("  mixed products:", [(t, str(d), p) for t, d, p in choice.mixed])

    print("\nRequests:")
    for o in report.outcomes():
        label = f"{o.data_type} {o.day}" + (f" {o.station}" if o.station else "")
        name = Path(o.path).name if o.path else ""
        print(f"  {o.status:16s} {label:26s} {o.source or '':6s} {name}")

    attention = [o for o in report.outcomes() if o.status in ATTENTION]
    if attention:
        print("\nNeeds attention:")
        for o in attention:
            label = f"{o.data_type} {o.day}" + (f" {o.station}" if o.station else "")
            print(f"  {o.status:16s} {label:26s} {o.message}")

    print("\nPaths for PPPConfig:")
    if config.dry_run:
        print("  (dry run: nothing on disk; set DRY_RUN = False to download)")
    else:
        for key, data_types in (("obs_path", ("obs",)), ("nav_path", ("nav",)), ("sp3_path", ("sp3",)),
                                ("clk_path", ("clk",)), ("dcb_path", ("bia", "dcb")), ("gim_path", ("ionex",))):
            paths = [str(p) for t in data_types if t in config.product_types for p in report.files(t)]
            print(f"  {key:9s} = {paths}")


if __name__ == "__main__":
    main()
