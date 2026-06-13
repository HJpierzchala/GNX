"""Dry-run GNSS archive download example.

Edit the USER SETTINGS block, then run from the repository root:

    python examples/run_download_gnss_data_example.py
"""

from pathlib import Path

from gnx_py.downloaders import GNSSDownloadConfig, GNSSDownloader


# ----------------------------- USER SETTINGS ----------------------------- #
YEAR = 2024
DOY_START = 35
DOY_END = 35
STATIONS = ("BRUX",)
OUTPUT_DIR = Path("examples/output/gnss_download")
DATA_TYPES = ("obs", "nav", "sp3")
SOURCES = ("bkg", "cddis", "code")

# Keep the example safe by default: generate candidate URLs, but do not touch
# the network and do not download files.
DRY_RUN = True
CHECK_AVAILABILITY = False

# Add aliases for 4-character station codes that need RINEX3/4 long filenames.
STATION_ALIASES = {
    "BRUX": "BRUX00BEL",
}
# -------------------------------------------------------------------------- #


def main() -> None:
    config = GNSSDownloadConfig(
        year=YEAR,
        doy_start=DOY_START,
        doy_end=DOY_END,
        output_dir=OUTPUT_DIR,
        stations=STATIONS,
        data_types=DATA_TYPES,
        sources=SOURCES,
        station_aliases=STATION_ALIASES,
        dry_run=DRY_RUN,
        check_availability=CHECK_AVAILABILITY,
        verbose=True,
    )

    report = GNSSDownloader(config).download()
    print("Summary:", report.summary())
    for result in report.results:
        print(f"{result.status:16s} {result.url}")
        print(f"{'':16s} -> {result.path}")

    print("\nGenerated candidates:")
    for candidate in report.candidates[:20]:
        print(f"{candidate.source:5s} {candidate.data_type:3s} {candidate.url}")
    if len(report.candidates) > 20:
        print(f"... {len(report.candidates) - 20} more candidates")


if __name__ == "__main__":
    main()
