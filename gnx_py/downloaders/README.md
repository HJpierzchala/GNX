# GNSS Downloaders

`gnx_py.downloaders` provides a conservative GNSS archive downloader for common
daily RINEX observations, broadcast navigation files, and precise products.

The module is intentionally small and uses only the Python standard library.
It generates known archive candidates, checks them with lightweight HTTP
requests, and downloads the first available candidate for each requested logical
file group.

## Basic Use

```python
from gnx_py.downloaders import GNSSDownloadConfig, GNSSDownloader

config = GNSSDownloadConfig(
    year=2024,
    doy_start=35,
    doy_end=35,
    output_dir="data/gnss",
    stations=("BRUX",),
    data_types=("obs", "nav", "sp3"),
    sources=("bkg", "cddis", "code"),
    rinex_version_preference=("3", "4", "2"),
    dry_run=True,
    check_availability=False,
)

report = GNSSDownloader(config).download()
print(report.summary())
for result in report.results:
    print(result.status, result.url)
```

You can also import the same API from `gnx_py.download`.

## Output Layout

Files are stored under:

```text
output_dir/YYYY/DDD/source/type/filename
```

For example:

```text
data/gnss/2024/035/bkg/nav/BRDC00IGS_R_20240350000_01D_MN.rnx.gz
```

Existing files are skipped by default. Set `overwrite=True` to replace them.
Set `max_workers` to a value greater than 1 to check/download independent
request groups concurrently.

## Sources

Supported source keys:

- `bkg`: BKG IGS archive under `https://igs.bkg.bund.de/root_ftp/IGS`
- `cddis`: NASA CDDIS GNSS archive under `https://cddis.nasa.gov/archive/gnss`
- `code`: CODE/AIUB product archive under `https://ftp.aiub.unibe.ch/CODE`
  and the CODE MGEX product tree

CODE is treated as a product source. The downloader does not generate station
observation candidates for CODE.

## Data Types

Implemented data type keys:

- `obs`: daily station observation files, preferring RINEX3/4 long-name
  Hatanaka `crx.gz` candidates where a 9-character station id is known
- `nav`: daily broadcast navigation candidates such as `BRDC00IGS` and
  `BRD400DLR`
- `sp3`: precise orbit SP3 products
- `clk`, `erp`, `bia`, `dcb`: conservative product candidates for future and
  advanced workflows

The product support is deliberately conservative. Historical products and
analysis-center naming can differ by year and data center.

## Station Names

RINEX2 candidates use the 4-character legacy station code.

RINEX3 and RINEX4 long filenames require a 9-character station id such as
`BRUX00BEL`. If you pass a 9-character id, the downloader uses it directly. If
you pass a 4-character id, only known aliases are expanded. A small built-in map
contains `BRUX -> BRUX00BEL`; pass `station_aliases` for your own network:

```python
config = GNSSDownloadConfig(
    year=2024,
    doy_start=35,
    doy_end=35,
    output_dir="data/gnss",
    stations=("WTZR",),
    station_aliases={"WTZR": "WTZR00DEU"},
)
```

The downloader does not invent country codes or monument numbers.

## CDDIS / Earthdata Login

CDDIS archive access requires NASA Earthdata Login credentials. GNX never
hardcodes or stores passwords in the repository. Provide credentials by one of
these methods:

Environment variables:

```bash
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"
```

Or a `~/.netrc` entry:

```text
machine urs.earthdata.nasa.gov login your_username password your_password
```

The `.netrc` file should be readable only by your user account.

You can also pass runtime credentials:

```python
from gnx_py.downloaders import CDDISCredentials, GNSSDownloadConfig

config = GNSSDownloadConfig(
    year=2024,
    doy_start=35,
    doy_end=35,
    output_dir="data/gnss",
    sources=("cddis",),
    credentials=CDDISCredentials(username="...", password="..."),
)
```

If credentials are missing, CDDIS candidates return `auth_missing` results
instead of aborting the whole run.

## Dry Run

`dry_run=True` prevents file downloads. With the default
`check_availability=True`, the downloader still performs lightweight URL checks
and reports which remote file would be downloaded. Set
`check_availability=False` to only generate candidate URLs with no network I/O.

## RINEX Priority

`rinex_version_preference=("3", "4", "2")` means:

1. Prefer long-name RINEX3/4 style station observation candidates.
2. Prefer `BRDC00IGS` mixed daily navigation over `BRD400DLR`.
3. Fall back to RINEX2 short-name candidates.

For observation files, RINEX3 and RINEX4 share the long filename convention, so
the downloader cannot prove the internal RINEX version without opening the file
header. It treats them as long-name candidates and reports the requested
priority label.

## Limitations

- Archive layouts differ by data center and have changed over time.
- Not every station publishes RINEX3/4 daily files.
- A 4-character station code is not enough to safely infer a 9-character
  RINEX3/4 station id.
- CDDIS requires Earthdata Login.
- CODE/AIUB is mainly a product source, not a general station observation
  archive.
- Historical SP3, CLK, ERP, BIA, BSX, and DCB products may use other names.
- The downloader is extensible and conservative; it is not a broad web scraper.
