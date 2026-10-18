# GNSS Downloaders

`gnx_py.downloaders` downloads daily RINEX observations, broadcast navigation
files and precise products (SP3 orbits, clocks, ERP, OSB/DCB biases, IONEX
maps and ORBEX satellite attitude) from the BKG, CDDIS and CODE archives,
ready to be passed to the GNX processing sessions.

For each requested item the module generates the archive candidates (URL and
local path), downloads the first one that is available and valid, and records
what happened. It uses `requests` for HTTPS and an Earthdata bearer token for
CDDIS.

Two layers are available:

- `DownloadConfig` / `DownloadSession` (recommended): the public entry point,
  following the same config + session + `run()` pattern as
  `PPPConfig`/`PPPSession`, `TECConfig`/`TECSession` and
  `SISConfig`/`SISSession`. It works with calendar dates, station ids and
  product families, keeps all files under one project folder, chooses one
  consistent product set for orbits, clocks and biases, and returns a
  `DownloadReport`. See "Quick Start".
- `GNSSDownloadConfig` / `GNSSDownloader`: the low-level engine, working in
  year/day-of-year, data-type keys and source keys, and returning an
  `ArchiveDownloadReport`. `DownloadSession` runs it once per day. See
  "Low-Level Engine".

Both layers are importable from `gnx_py.downloaders` and from
`gnx_py.download`. The older `ProjectDownload*` names still work but are
deprecated; see "Deprecated Names".

## Quick Start

```python
from datetime import date

from gnx_py.downloaders import DownloadConfig, DownloadSession

config = DownloadConfig(
    project_name="brux_2024_035",
    output_root="data",                        # -> data/brux_2024_035/...
    dates=date(2024, 2, 4),
    stations="BRUX00BEL",                      # 9-character id (or "BRUX")
    product_types=("obs", "nav", "sp3", "clk", "bia"),
)                                              # CDDIS: EARTHDATA_TOKEN in the environment

report = DownloadSession(config).run()         # a DownloadReport
print(report.summary())
print(report.product_sets.solution)            # e.g. "COD0MGXFIN": one centre for SP3/CLK/OSB

for outcome in report.outcomes():              # one line per requested (type, day[, station])
    print(outcome.data_type, outcome.day, outcome.station, outcome.status, outcome.path)
```

The report gives the paths a processing session needs. GNX reads the
compressed files (`.gz`, `.Z`, `.crx.gz`) directly, so nothing has to be
unpacked:

```python
from gnx_py.ppp import PPPConfig

ppp = PPPConfig(
    obs_path=report.files("obs")[0],
    nav_path=report.files("nav")[0],
    sp3_path=report.files("sp3"),              # d-1, d, d+1 of one solution
    clk_path=report.files("clk"),
    dcb_path=report.files("bia")[0],           # OSB of the same solution
    # ... atx_path and the processing settings
)
```

A dry run (`dry_run=True`) downloads nothing; with `check_availability=False`
it makes no archive request at all and only lists the candidates (their
would-be paths are in `outcomes()`).

## Configuration

`DownloadConfig` holds every session setting and checks all of them when it
is built, before any file, folder or network access. A wrong type raises
`TypeError`, a wrong value `ValueError`, and the message names the setting and
the offending item. `DownloadSession` defines no defaults of its own.

- **Project folder**: `config.output_dir` is `output_root / project_name`
  (read-only: assigning it raises `AttributeError`; set `output_root` and
  `project_name`). `project_name` must be a relative name such as
  `"brux_2024"` or `"campaign/brux"`; absolute paths, drive letters, `.`/`..`
  parts and `None` are refused. The folder is created by `run()`, also in a
  dry run. `output_root` defaults to the current directory.
- **Dates**: a `date`/`datetime` (pandas `Timestamp` included) or an iterable
  of them; strings and numbers are refused. Dates are sorted and
  de-duplicated. A `datetime` with a time zone is converted to UTC before the
  date is taken; a naive `datetime` is taken as UTC (GNSS file days are
  UTC/GPS days).
- **Product types**: friendly names, case-insensitive: `"sp3"`/`"orbit"`,
  `"broadcast"`/`"nav"`, `"ionex"`/`"gim"`, `"bia"`/`"osb"`,
  `"att"`/`"orbex"`/`"attitude"`, `"clk"`/`"clock"`, plus `"obs"`, `"dcb"` and
  `"erp"`. See `PRODUCT_TYPE_ALIASES` for the full mapping.
- **Stations**: 4-character codes (`BRUX`) or 9-character RINEX ids
  (`BRUX00BEL`), stripped and upper-cased; required when `"obs"` is
  requested. See "Station Names".
- **SP3 boundary days**: whenever `"sp3"` is requested, SP3 is also requested
  for `sp3_adjacent_days` days before and after every date (an `int >= 0`,
  default 1), since precise-orbit interpolation needs the surrounding data.
- **Sources**: archives tried in priority order, by default
  `("cddis", "bkg", "code")`; `"euref"` (the BKG EUREF/EPN archive, station
  observations only) is added on request; case-insensitive, duplicates
  dropped. See "Sources".
- **Credentials**: at most one of `token`, `username` + `password` (always
  together) or `credentials_file`. See "CDDIS / Earthdata Login".
- **Product preferences**: `product_preferences` maps a product type (friendly
  names accepted, e.g. `"orbit"`, `"osb"`, `"broadcast"`) to products in
  priority order: a 10-character product id (`COD0MGXFIN`) for
  sp3/clk/erp/bia/dcb/att; a 3-letter analysis centre (`COD`) or a
  10-character id for ionex; one of `BRDC00IGS`, `BRD400DLR`, `BRDM00DLR`,
  `BRDC` for nav. A single string is a list of one; `"obs"` and unknown keys
  are refused. Without an entry the engine defaults apply.
- **RINEX preference**: `rinex_version_preference` (`"3"`, `"4"`, `"2"`,
  default `("3", "4", "2")`) orders the observation candidates and, without a
  `"nav"` product preference, the navigation candidates.
- **Data-source letter**: `rinex_data_source_preference` (`"R"` receiver,
  `"S"` data stream, `"U"` unknown; default `("R", "S")`) orders the letters
  of the RINEX 3/4 long observation names (`BRUX00BEL_R_...`,
  `KATO00POL_S_...`). See "Station Names".
- **Execution switches**: `overwrite`, `dry_run`, `check_availability`,
  `max_workers` (`None` or an `int >= 1`, default 4), `timeout` (seconds, a
  finite number `> 0`, or `None` to wait without limit; default 30),
  `retries` (an `int >= 0`, default 2; see "Errors") and `verbose`.

**Feasibility.** Construction also checks that the request can be served:

- `ValueError` for input errors: `"obs"` without stations, a 4-character
  station with no 9-character id (`station_aliases`) while `"2"` is not in
  the RINEX preference, an empty source, RINEX or product list, or nothing
  requested that the selected sources can provide;
- one `UserWarning` when only part of the request cannot be served (IONEX or
  ORBEX with `sources=("bkg",)`, observations from `"code"`): those items
  become outcomes with status `"unavailable"` and the reason;
- one `UserWarning` when credentials are given but `"cddis"` is not a source.

**Assignment.** As in the other configurations, assigning a setting after
construction checks and normalises that setting at once (a rejected
assignment leaves the object unchanged), and a misspelled setting name raises
`AttributeError` with a "did you mean" hint. Checks of a combination
(`username` with `password`, stations with `"obs"`) are not raised on
assignment, because such settings change in two steps; `validate_for_run()`,
which `DownloadSession.run()` calls first, raises them.

## Reading the Report

`DownloadSession(config).run()` returns a `DownloadReport`:

- `outcomes()`: one `RequestOutcome` per requested item (data type, day, and
  station for observations) with `status`, `path`, `product`, `source`,
  `url`, `message` and `tried` (every engine row of that request). The status
  is one of the engine statuses below or `"unavailable"`.
- `files(data_type, day=None, station=None)`: the local paths of satisfied
  requests (`downloaded` or `skipped_existing`) in day order; friendly type
  names are accepted (`files("orbit")`). `files("sp3", day=d)` gives the SP3
  window of day `d` (d−A … d+A) from one solution.
- `files_by_type()`: the same for every type.
- `unavailable`: the `UnavailableRequest` entries (data type, day, station,
  reason) of items no selected source provides.
- `product_sets`: the `ProductSetChoice` of the session (see "Consistent
  Product Sets"); `mixed_products` is `True` when linked products come from
  more than one solution.
- `reports`: one `ArchiveDownloadReport` per calendar day sent to the engine,
  in chronological order, including the SP3 boundary days.
- `all_results()`: every engine row (`DownloadResult`, one per candidate
  tried); `by_status(status)`, `downloaded`, `skipped_existing`, `dry_runs`,
  `missing` and `errors` (`error` and `auth_missing`) filter them.
- `summary()`: engine status counts, the number of generated `candidates`,
  plus `requests` (number of outcomes) and `unavailable`.

```python
report = DownloadSession(config).run()

attention = [o for o in report.outcomes()
             if o.status in ("missing", "error", "auth_missing", "unavailable")]
for o in attention:
    print(o.data_type, o.day, o.station, o.status, o.message)

sp3 = report.files("sp3")
```

Engine statuses:

- `downloaded`: the file was downloaded and passed the checks of "File
  Integrity".
- `skipped_existing`: a valid local file (of any candidate of the request)
  was already on disk and `overwrite=False`; no request was made.
- `dry_run`: `dry_run=True`; the candidate was generated (and, with
  `check_availability=True`, found on the server) but not downloaded.
- `missing`: the remote file does not exist (HTTP 404).
- `auth_missing`: CDDIS needs a token and none was available, or CDDIS
  refused it (HTTP 401/403 or an Earthdata login page).
- `error`: any other failure for that candidate (another HTTP status, an
  invalid file, retries used up, an unreachable host).

Engine rows are per candidate: a request that fell back from CDDIS to BKG has
an `auth_missing` row and a `downloaded` row, so `errors` lists the first even
though the request was satisfied. `outcomes()` shows what needs attention;
its `message` names earlier failed sources ("also tried: ...").

## Consistent Product Sets

Orbits, clocks, OSB biases and ORBEX attitude (`sp3`, `clk`, `bia`, `att`)
come from one analysis-centre solution for the whole project, the SP3
boundary days included. Mixing solutions breaks PPP: the clocks of different
centres disagree at day boundaries, and PPP-AR needs the OSB of the same
solution as the orbits and clocks (`PPPSession` disables PPP-AR otherwise).

- **Effective lists.** A linked type without its own preference starts with
  the preference of the lead type (sp3, else clk, else bia, else att), then
  its defaults. So `{"sp3": ("WUM0MGXFIN",)}` asks for WUM clocks and OSB
  first too.
- **Tier 1.** The first solution with every file available is used for every
  file. The product decides before the archive: `COD0MGXFIN` at BKG is
  preferred to `GRG0MGXFIN` at CDDIS.
- **Tier 2.** Otherwise the first solution with every SP3 and CLK file of the
  requested days is used; boundary days and ORBEX it lacks are reported
  missing (never taken from another centre), and a missing OSB comes from
  the next product of the bia list with a `UserWarning` (PPP-AR off).
- **Tier 3.** Otherwise the choice is made per day, as a last resort, with one
  `UserWarning` and `report.mixed_products=True`. An SP3 day in the windows
  of two days with different products is fetched once per product, so
  `files("sp3", day=d)` still gives one solution.
- **Availability** comes from a valid local file, or one probe request per
  linked file and source (newest day first; a product that lacks a file stops
  being probed). Only HTTP 404 (or a refusal) makes a file absent: a probe
  without a definite answer (timeout, unreachable host, 5xx after the
  retries) counts as available, is logged as a WARNING and is kept in
  `report.product_sets.probe_failures`, so a passing network fault never
  switches the centre. A dry run without availability checks probes nothing
  (`product_sets.checked` is `False`).
- **The choice is final**: a chosen file that then fails at every source is an
  error row for that file; the session never switches products for it.
- `report.product_sets` holds `solution`, `tier`, `per_item` (the product per
  (type, day)), `missing`, `mixed`, `extra` (tier-3 SP3 days fetched for a
  second product), `checked` and `probe_failures`.
- ERP and navigation are not linked. For ERP the product of the solution is
  tried first when it publishes ERP. For the other types the source order
  decides first, then the product order within each source.

## Errors

Two kinds of problems are handled differently:

- **Configuration errors** are raised by `DownloadConfig(...)` (and by
  `validate_for_run()` at the start of `run()`), before any request.
- **Session failures** abort `run()`: an unexpected exception inside
  `DownloadSession.run()` is re-raised as
  `gnx_py.core.session_errors.SessionExecutionError` (`session` is
  `"DownloadSession"`, `context` holds `project_name`, `output_dir`,
  `product_types`, `sources` and `dates`, long lists shortened, and `cause`
  is the original exception). Typical causes: a missing or unreadable
  credentials file, a failed Earthdata token exchange (`CDDISAuthError`).
- **Per-file problems** do not abort the session; they are engine rows and
  outcomes: a missing remote file, missing or rejected CDDIS credentials, an
  invalid payload, a network error, and also an exception raised by a custom
  transport's `exists()` for one candidate.

HTTP rules (the availability check and the download follow the same rules):

- 404 → `missing`.
- 401/403, or an HTML/XML page instead of the file → `auth_missing` on CDDIS
  (with a hint to check `EARTHDATA_TOKEN`), and `error` with the real HTTP
  status on BKG and CODE.
- Timeouts, broken connections, cut transfers, HTTP 5xx and 429 are retried
  `retries` times (default 2, waits of 1 s then 2 s; a `Retry-After` of a 429
  or 503 is honoured up to 60 s) before the next candidate is tried. Other
  4xx answers, HTML pages, invalid files and failed TLS certificate checks
  are not retried. `DownloadResult.attempts` records the count.
- A host that refuses two connections in a row (refused, name not resolved,
  connect timeout) is marked unreachable for the rest of the session: later
  candidates there become `error` ("<host> unreachable") without a request,
  and the next source is tried. A connect failure is therefore retried once
  before the host is skipped. Read timeouts, dropped connections and 5xx
  answers do not count.
- A redirect from HTTPS to plain HTTP is refused. The bearer token is never
  sent to another host after a redirect.

```python
from gnx_py.core.session_errors import SessionExecutionError

try:
    report = DownloadSession(config).run()
except SessionExecutionError as exc:
    print(exc.session, exc.context)
    print("caused by:", repr(exc.cause))
    raise
```

## File Integrity

A file is accepted only after these checks (the default transport):

- one GET per file; the bytes written must equal `Content-Length` (when the
  server sent it without a `Content-Encoding`);
- at least 100 bytes, no HTML/XML page;
- the container matches the name: a `.gz` file is decompressed in full (all
  members, CRC and length; trailing bytes after the last member are tolerated
  like `gunzip` does), a `.Z` file needs a valid LZW header, and compressed
  bytes under an uncompressed name (`.SP3`, `.rnx`, ...) are refused;
- the content header matches the type: SP3 `#a`…`#d`, RINEX/CRINEX
  observation, navigation and clock headers, IONEX, Bias-SINEX (`%=BIA`) and
  ORBEX (`%=ORBEX`); ERP has no check;
- compressed names (`.gz`, `.Z`, `.bz2`, `.zip`) are written as the raw bytes
  of the archive (a `Content-Encoding: gzip` is not undone); `sha256` of the
  written bytes is in `DownloadResult.sha256`.

Unix compress (`.Z`) has no checksum and no end marker, so a cut `.Z` file is
detected only through `Content-Length`.

## Custom Transport and Logger

`DownloadSession(config, transport=None, logger=None)` performs no I/O in its
constructor. Credential resolution, project-folder creation and transport
construction happen at the start of each `run()`.

- `transport`: any object implementing the `DownloadTransport` protocol
  (`has_credentials(candidate)`, `exists(candidate) -> URLAvailability`,
  `download(candidate, destination) -> int`). It is shared by all days, used
  as-is and not closed. Explicit credentials of `DownloadConfig` are not used
  with an injected transport (a `UserWarning` says so); configure the
  transport's own credentials instead. To report a missing file, `download()`
  raises `RemoteFileMissing` (status `missing`); any other exception gives
  `error`. An injected transport does its own response validation. Without an
  injected transport, `run()` builds one `RequestsDownloadTransport` from the
  credentials, `timeout` and `retries`, shares it across all days and closes
  it at the end.
- `logger`: passed to every engine run. Defaults to the module logger
  `gnx_py.downloaders.core`. `verbose=True` adds one INFO `StreamHandler` to
  the `gnx_py.downloaders` logger (if it has none) and never configures the
  root logger. The handler and level stay for the rest of the process; lines
  print twice if the application also has a root handler.

## Output Layout

Files are stored under:

```text
output_dir/YYYY/DDD/source/type/filename
```

`output_dir` is `output_root/project_name` for `DownloadSession` and
`GNSSDownloadConfig.output_dir` for the engine. For example, the broadcast
navigation of 2024-02-04 from BKG in the Quick Start project:

```text
data/brux_2024_035/2024/035/bkg/nav/BRDC00IGS_R_20240350000_01D_MN.rnx.gz
```

- A file covering several days is stored under the day of its stamp: the
  weekly ERP of IGS0OPSFIN, COD0OPSFIN and GRG0MGXFIN
  (`..._07D_01D_ERP.ERP.gz`) under the GPS-week start day, the quarterly DLR
  biases (`..._03L_01D_...`) under the first day of the quarter. The next day of the same week finds it on disk.
- A valid file already on disk satisfies its request with no request, also
  when it is the file of another candidate (another archive directory or
  name). An invalid file (empty, an HTML page, a cut archive) is logged as a
  WARNING and downloaded again. `overwrite=True` downloads every file again.
- `output_dir/.gnx_downloads.json` records size, modification time and
  sha256 of every checked file, so a rerun checks unchanged files quickly;
  files not in it are checked in full once. Deleting it only costs one full
  check per file.
- Downloads go to a unique temporary file next to the destination
  (`.<name>.<random>.part`), which is renamed when the file is complete and
  removed on any failure, `Ctrl-C` included. `*.part` files left by older
  versions are safe to delete when no session is running.
- `max_workers` > 1 processes the request groups of one day concurrently;
  groups that may end at the same file run in one worker, in order.
  `DownloadConfig` defaults to 4 workers, `GNSSDownloadConfig` to sequential
  processing.

## Sources

| Source | Archive | Serves |
| --- | --- | --- |
| `cddis` | NASA CDDIS, `https://cddis.nasa.gov/archive/gnss` (Earthdata Login) | obs, nav, sp3, clk, erp, bia, dcb, ionex, att |
| `bkg` | BKG, `https://igs.bkg.bund.de/root_ftp/IGS` and `.../IGSac` | obs, nav, sp3, clk, erp, bia, dcb (no IONEX, no ORBEX) |
| `code` | CODE/AIUB, `https://ftp.aiub.unibe.ch/CODE` and `CODE_MGEX` | CODE's own `COD` products: sp3, clk, erp, bia, ionex, att |
| `euref` | BKG as the EUREF Permanent GNSS Network (EPN) data centre, `https://igs.bkg.bund.de/root_ftp/EUREF` | obs (EPN stations; not a default source) |

- Sources are tried in the given order (but see "Consistent Product Sets" for
  the linked types). `DownloadConfig` defaults to `("cddis", "bkg", "code")`,
  `GNSSDownloadConfig` to `("bkg", "cddis", "code")`.
- CDDIS product directories: `products/WWWW/` (4-digit GPS week); for GPS
  weeks before 2238 also `products/mgex/WWWW/` for MGEX campaign products
  (`...MGX...`; from week 2238 they are in `products/WWWW/` and the `mgex/`
  week directory does not exist); then `products/bias/YYYY/` for every DCB
  product and for CAS and DLR OSB. IONEX:
  `products/ionex/YYYY/DDD/`. Observations: `.crx.gz` and `.YYd` files in
  `YYd/`, `.rnx.gz` and `.YYo` files in `YYo/`.
- BKG: `IGS/products/WWWW/`, then the analysis-centre products under
  `IGSac/products/WWWW/` (for example `COD0OPSFIN`).
- CODE serves only CODE products; preferences of other centres are skipped
  there with one INFO line. Observations and navigation are not served.
- EUREF (source `euref`): daily observations in
  `EUREF/obs/YYYY/DDD/`, no Earthdata token, with the names of BKG
  `IGS/obs/`. The long `.crx.gz` names were checked against the archive URLs
  of 38 EPN stations and the EUREF directory page of 2024-035 (captured in
  October 2026): 278 daily `_01D_30S_MO.crx.gz` files
  (`_R_` and `_S_`), RINEX 2 short names (`rove0350.24d.gz`) for 7 stations,
  and no `_MO.rnx.gz` observation file (the `.rnx.gz` candidates find nothing
  there; they cost a request only for a station without a `.crx.gz` file). Many
  EPN stations are only there, e.g. the Polish ASG-EUPOS stations, which
  publish stream files (`KATO00POL_S_...`). It is not a default source: add
  it for EPN stations,
  e.g. `sources=("bkg", "euref")` (the IGS observation directory first). The
  EUREF archive is modelled for observations only; other types requested
  with `sources=("euref",)` are reported as unavailable.
- `core.SOURCE_CAPABILITIES` and `core.unavailable_reason()` hold the answers
  the configuration checks use.

## Data Types

| Key | Content | Defaults (priority order) |
| --- | --- | --- |
| `obs` | daily station observations (RINEX 3/4 `crx.gz`/`rnx.gz`, RINEX 2) | — |
| `nav` | daily broadcast navigation | from `rinex_version_preference`: BRDC00IGS (3), BRD400DLR (4), BRDC (2) |
| `sp3` | precise orbits | COD0MGXFIN, GRG0MGXFIN, GFZ0MGXRAP, COD0OPSFIN, IGS0OPSFIN |
| `clk` | precise clocks (30 s) | as sp3 |
| `erp` | Earth rotation parameters | COD0MGXFIN, COD0OPSFIN, IGS0OPSFIN, GFZ0MGXRAP |
| `bia` | observable-specific biases (OSB) | COD0MGXFIN, GRG0MGXFIN, GFZ0MGXRAP, COD0OPSFIN, CAS0MGXRAP |
| `dcb` | differential code biases | CAS0OPSRAP, CAS1OPSRAP, GFZ0OPSRAP, CAS0MGXRAP, DLR0OPSFIN |
| `ionex` | GIM/TEC maps | COD, IGS, ESA, JPL, UPC |
| `att` | ORBEX satellite attitude | COD0MGXFIN, GRG0MGXFIN, GFZ0MGXRAP |

- **Navigation vocabulary** (`product_preferences["nav"]`): `BRDC00IGS`
  (RINEX 3, mixed), `BRD400DLR` (RINEX 4; carries the BeiDou-3 BDGIM
  parameters needed by `ionosphere_model="bdgim"`), `BRDM00DLR` (RINEX 3),
  `BRDC` (GPS-only RINEX 2 `brdcDDD0.YYn`). GLONASS RINEX 2 navigation files
  (`.YYg`) are not fetched; GNX processes G, E and C.
- **Product catalogue** (`gnx_py.downloaders.catalog`): what each known
  product publishes and its file-name forms (IGS0OPSFIN SP3 every 15 min, CLK
  30 s, weekly ERP; COD/GRG/GFZ MGEX products; COD0OPSFIN and GRG0MGXFIN
  with the weekly ERP only; CAS, GFZ and DLR biases: the daily rapid
  `CAS0OPSRAP`/`GFZ0OPSRAP`/`CAS1OPSRAP` `..._01D_01D_DCB.BIA` files and the
  quarterly `DLR0OPSFIN_..._03L_01D_DCB.BIA.gz`), with a `verified` note per entry
  (most of them checked against the 2024 directory listings of BKG and
  CDDIS). An unknown product id gets the generic names
  (SP3 05M, then 15M).
- **Default `dcb` list** (since v1.0.0): the rapid `DCB.BIA` products
  first. In the 2024 CDDIS listing
  (`products/bias/2024/`) CAS0OPSRAP covers 346 days, CAS1OPSRAP 332,
  GFZ0OPSRAP 364 (all but 021-022) and CAS0OPSRAP with GFZ0OPSRAP every day of
  2024: with the default list a 2024 day takes CAS0OPSRAP on 346 days,
  CAS1OPSRAP on 9 and GFZ0OPSRAP on 11; CAS0MGXRAP (`DCB.BSX`) exists only for
  days 021-048, and the quarterly DLR0OPSFIN file covers each quarter. BKG
  `IGS/products/WWWW/` holds the CAS0OPSRAP and CAS1OPSRAP files as well. The
  list before v1.0.0 (CAS0MGXRAP, DLR0MGXFIN, GFZ0MGXRAP, CAS1MGXRAP) found
  a 2024 DCB for days 021-048 only. GNX-py reads the CAS files (`sample_data`
  and the validation data) and a GFZ0OPSRAP file (2024-085 from CDDIS: one DSB
  pair per system, see "Limitations").
- **IONEX**: a 3-letter centre expands to its final and rapid long names
  (`COD0OPSFIN_..._01D_01H_GIM.INX.gz`, `..._02H_...`, `COD0OPSRAP_...`) plus
  the legacy short name (`codgDDD0.YYi.Z` and `.gz` at CDDIS, `CODGDDD0.YYI.Z`
  at CODE); a 10-character id gives its exact names only.
- Product ids must be letters and digits; anything else is refused.

`DownloadConfig.product_types` accepts these keys and the aliases of
`PRODUCT_TYPE_ALIASES`: `observation`/`observations` for `obs`;
`broadcast`/`navigation`/`brdc` for `nav`; `orbit`/`orbits`/`precise_orbit`
for `sp3`; `gim`/`tec` for `ionex`; `osb`/`bias` for `bia`; `clock` for `clk`;
`orbex`/`attitude` for `att`.

## Station Names

RINEX 3/4 long file names need a 9-character id such as `BRUX00BEL`; RINEX 2
short names use the 4-character code.

- A 9-character id is used directly. It gets RINEX 2 names only when its
  monument/receiver digits are `00` (`BRUX00BEL` → `brux...`); for
  `ABCD01POL` no RINEX 2 name exists, because the short name belongs to the
  `00` receiver.
- A 4-character code is expanded only through known aliases. A small built-in
  map holds `BRUX -> BRUX00BEL`; pass `station_aliases` for your network
  (every id must start with its key). Without an alias, a 4-character code
  needs `"2"` in `rinex_version_preference`, otherwise `DownloadConfig` raises
  `ValueError`. A 4-character request gets the RINEX 2 names.
- Long names carry a data-source letter after the id: `R` for a file from a
  receiver, `S` for one built from a data stream (for example the EPN stations
  of the Polish ASG-EUPOS network: `KATO00POL_S_20240350000_01D_30S_MO.crx.gz`),
  `U` unknown. `rinex_data_source_preference` (default `("R", "S")`) gives
  the letters tried for every id, in order, each with `.crx.gz` then
  `.rnx.gz`; a station that publishes an `R` file is found by the same first
  request as before. A station that a source does not have costs two more
  requests there per day with the default (and two more availability checks
  with `check_availability`) before the RINEX 2 names. Pass `("R",)` to try
  receiver files only. RINEX 2 short names have no letter. Only the letter
  varies: the rest of the long name is fixed to a daily 30-s mixed file
  (`01D_30S_MO`), see "Limitations".
- Every requested station has its own request and report row, also two
  receivers of one site (`ABCD00POL`, `ABCD01POL`). Requests that resolve to
  the same RINEX id (`"BRUX"` with its alias and `"BRUX00BEL"`) are kept
  once, under the first spelling, with an INFO line.

```python
from datetime import date

from gnx_py.downloaders import DownloadConfig

config = DownloadConfig(
    project_name="my_gnss_project",
    dates=date(2024, 2, 4),
    product_types="obs",
    stations=("WTZR",),
    station_aliases={"WTZR": "WTZR00DEU"},
)
```

The downloader does not invent country codes or monument numbers.

EPN stations outside the IGS network are in the EUREF archive:

```python
config = DownloadConfig(
    project_name="epn_2024_035",
    dates=date(2024, 2, 4),
    product_types="obs",
    stations=("KATO00POL", "LAMA00POL"),
    sources=("bkg", "euref"),
)
```

`LAMA00POL` is found in BKG `IGS/obs/` (`_R_`), `KATO00POL` in
`EUREF/obs/` (`_S_`).

## CDDIS / Earthdata Login

CDDIS has required NASA Earthdata Login since 2020. The default transport,
`RequestsDownloadTransport`, authenticates CDDIS requests with a bearer token
only (`Authorization: Bearer <token>`); it never sends a username or password
to CDDIS. GNX never stores credentials.

### With `DownloadConfig`

Provide at most one credential style:

- `token`: an already-issued Earthdata bearer token.
- `username` + `password`: exchanged for a bearer token.
- `credentials_file`: a small text file read by `read_credentials_file` when
  the session runs (not when the configuration is built), then exchanged the
  same way. Accepted formats (a UTF-8 byte-order mark is ignored, values are
  stripped):
  - `EARTHDATA_USERNAME=...` / `EARTHDATA_PASSWORD=...` lines; this format
    takes passwords containing `=` or starting with `#` (the line is split at
    the first `=`);
  - two lines: login on line 1, password on line 2 (neither may contain `=`);
  - one line `login,password` (also `:`, tab or blank, tried in that order:
    `user:pa,ss` gives the login `user:pa`; a password containing one of these
    characters needs another format).

The credentials are used only when they can be: when `"cddis"` is a source
and no transport was injected. Then a username/password or a credentials
file costs one token exchange per `run()` (also with `dry_run=True` and
`check_availability=False`), and the token is shared by all days. Otherwise
nothing is exchanged and a `UserWarning` says why.

The exchange reuses a valid token of the account before creating a new one:
`POST /api/users/find_or_create_token`, then (if that endpoint is not
offered) the token list `GET /api/users/tokens`, taking the first token whose
`expiration_date` is in the future (a token that expires today is not used),
and `POST /api/users/token` only when none is usable. All requests use HTTP
Basic auth to `urs.earthdata.nasa.gov`. Network errors, HTTP errors and
invalid answers raise `CDDISAuthError` naming the endpoint and the status,
never the user name or password; Earthdata answering 429 or 5xx stops the
exchange as "temporarily unavailable".

Without credentials, the default transport falls back to the `EARTHDATA_TOKEN`
environment variable, and to nothing else:

```bash
export EARTHDATA_TOKEN="your_bearer_token"
```

The `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` variables and `~/.netrc` are
not used by the default transport. Keep credentials out of scripts that may be
shared or committed: prefer `EARTHDATA_TOKEN` or a `credentials_file` kept out
of version control. `token` and `password` are excluded from the
configuration's `repr` and from every message.

### With the Low-Level Engine

```python
from gnx_py.downloaders import CDDISCredentials, GNSSDownloadConfig

config = GNSSDownloadConfig(
    year=2024,
    doy_start=35,
    doy_end=35,
    output_dir="data/gnss",
    sources=("cddis",),
    data_types=("sp3",),
    credentials=CDDISCredentials(token="your_bearer_token"),
)
```

Without explicit credentials the engine's default transport also falls back
to `EARTHDATA_TOKEN` only. `CDDISCredentials(username=..., password=...)`
alone is not used by the default transport (it warns); exchange the pair
first, e.g. with `CDDISCredentials.from_password_exchange(username,
password)`. `CDDISCredentials(netrc_path=...)` reads an explicit `.netrc`; a
missing file raises `FileNotFoundError` and an unreadable one `ValueError`,
both naming the file.

### Exchanging a Username/Password Yourself

```python
from gnx_py.downloaders import fetch_earthdata_token

token = fetch_earthdata_token("your_username", "your_password")
```

`fetch_earthdata_token` lives in `gnx_py.downloaders.earthdata_auth` and is
loaded lazily on first access, so importing `gnx_py.downloaders` (or
`gnx_py`) never pulls in credential-exchange code unless you ask for it. It
is not part of wildcard imports; attribute access such as
`gnx_py.fetch_earthdata_token` still resolves it.

### Missing Credentials

Without a token, CDDIS candidates that are checked or downloaded give
`auth_missing` and the next source is tried, instead of aborting the run. With
`dry_run=True` and `check_availability=False`, credentials are not checked,
so CDDIS candidates are listed as `dry_run`.

## Dry Run

`dry_run=True` never downloads files. With the default
`check_availability=True` the session still asks the archives which files
exist (one request per candidate tried, plus the product-set probes) and
reports which file would be downloaded. With `check_availability=False` it
only generates candidates and makes no archive request. A
`username`/`password` or `credentials_file` with `"cddis"` among the sources
still costs one token exchange; use `token`, `EARTHDATA_TOKEN` or no
credentials for a fully offline dry run. `run()` creates the project folder in
every mode.

## RINEX Priority

`rinex_version_preference=("3", "4", "2")` means:

1. Prefer long-name RINEX 3/4 observation candidates.
2. Without a `"nav"` product preference, prefer `BRDC00IGS` (RINEX 3) over
   `BRD400DLR` (RINEX 4), then the GPS RINEX 2 `brdc` file.
3. Fall back to RINEX 2 short names.

The navigation file can be chosen independently of the observations:
`product_preferences={"nav": ("BRD400DLR", "BRDC00IGS")}` with
`rinex_version_preference=("3",)` gives RINEX 3 observations and the RINEX 4
navigation file with the BDGIM parameters.

For observations, RINEX 3 and 4 share the long file name, so the downloader
cannot tell the internal version without opening the file; it reports the
requested priority label.

## Low-Level Engine

`GNSSDownloadConfig` / `GNSSDownloader` are the engine underneath
`DownloadSession`. Use them directly for a single year/day-of-year run or to
inspect candidates with `build_candidates()`; otherwise prefer
`DownloadConfig` + `DownloadSession`.

```python
from gnx_py.downloaders import GNSSDownloadConfig, GNSSDownloader

config = GNSSDownloadConfig(
    year=2024,
    doy_start=35,
    doy_end=35,
    output_dir="data/gnss",
    stations=("BRUX00BEL",),
    data_types=("obs", "nav", "sp3"),
    sources=("bkg", "cddis", "code"),
    rinex_version_preference=("3", "4", "2"),
    dry_run=True,
    check_availability=False,
)

report = GNSSDownloader(config).download()   # also available as .run()
print(report.summary())
for result in report.results:
    print(result.status, result.url)
```

Differences from the session API:

- One request is an inclusive day-of-year range within one calendar year, with
  raw data-type keys (`obs`, `nav`, `sp3`, `clk`, `erp`, `bia`, `dcb`,
  `ionex`, `att`) and source keys. There are no friendly aliases, no SP3
  boundary days and no product-set choice: each type and day falls back
  through its product list on its own.
- `GNSSDownloadConfig` uses the same checks as `DownloadConfig` for the
  values it shares. Its default `data_types` include `"obs"`, so it needs
  `stations`, or `data_types` without `"obs"` (e.g. `("nav", "sp3")`).
- `download()` returns an `ArchiveDownloadReport` with `candidates`,
  `results`, `unavailable` (requested items with no candidate, with the
  reason), `by_status()`, `downloaded`, `skipped_existing`, `dry_runs`,
  `missing`, `errors` and `summary()`.
- It is not wrapped in `SessionExecutionError`: unexpected exceptions
  propagate unchanged.
- `GNSSDownloader(config, transport=None, logger=None)` builds a
  `RequestsDownloadTransport` from `config.credentials`, `config.timeout` and
  `config.retries` when no transport is given, and closes it at the end of
  `download()`.

## Legacy Transport

`UrllibDownloadTransport` was removed before the first release (it was never
exported, and `requests` is a dependency). It wrote files without validating
them, so an HTML page could be saved as a product. Use
`RequestsDownloadTransport` or your own `DownloadTransport`.

## Deprecated Names

These names still work, but emit a `DeprecationWarning` when accessed. The
`ProjectDownload*` names are resolved on attribute access from
`gnx_py.downloaders`, `gnx_py.download` and the top-level `gnx_py` package,
and are not part of wildcard imports, so `import gnx_py` stays warning-free.

| Deprecated name | Use instead |
| --- | --- |
| `ProjectDownloadConfig` | `DownloadConfig` |
| `ProjectDownloader` | `DownloadSession` |
| `ProjectDownloadReport` | `DownloadReport` |
| `gnx_py.downloaders.core.DownloadReport` | `ArchiveDownloadReport` |

When migrating, note:

- `DownloadReport` (as `gnx_py.downloaders.DownloadReport`,
  `gnx_py.download.DownloadReport` and `gnx_py.DownloadReport`) now means the
  session-level report returned by `DownloadSession.run()`. This change emits
  no `DeprecationWarning`. Code that annotates, `isinstance`-checks or
  constructs `GNSSDownloader.download()` results through these names must use
  `ArchiveDownloadReport` instead.
- `DownloadSession.run()` wraps unexpected exceptions in
  `SessionExecutionError`; catch it and inspect `exc.cause`.
- `fetch_earthdata_token` is no longer part of wildcard imports; import it
  explicitly with `from gnx_py.downloaders import fetch_earthdata_token`.

## Limitations

- Archive layouts differ by data center and change over time. The catalogue
  entries carry a `verified` note; entries marked "assumed" (for example
  DLR0MGXFIN and CAS1MGXRAP) follow documentation, not a seen file. The 2024
  CDDIS listing (`products/bias/2024/`) holds neither of them (DLR publishes
  `DLR0OPSFIN`, CAS `CAS0OPSRAP`/`CAS1OPSRAP`), and no CAS0MGXRAP OSB. At
  CDDIS the DCB files sit in `products/bias/YYYY/` only, after a first try in
  `products/WWWW/`.
- GFZ0OPSRAP `DCB.BIA` (third in the default `dcb` list) is served by CDDIS
  only (the BKG pages of 2024 and the rapid and ionosphere product directories
  of the GFZ data centre ISDC hold no GFZ DCB); with the default list it is
  used only on days without a CAS0OPSRAP and a CAS1OPSRAP file (11 days of
  2024: 196-199, 302, 315, 362-366). The file of
  2024-085, read through GNX on 2026-10-16 (Bias-SINEX, `BIAS_MODE`
  RELATIVE, GFZ clock and ionosphere analysis), holds one satellite DSB pair
  per system: GPS C1W-C2W (32 satellites), Galileo C1C-C5Q (25), BeiDou
  C2I-C6I (45), QZSS C1C-C2C and GLONASS C1P-C2P, plus receiver DSBs. A run
  with other signals (BeiDou-3 B1C / B2a / B2b, Galileo E5b, GPS L5, X
  tracking) gets no product DSB from it and falls back to the broadcast group
  delays with a warning. Before v1.0.0 GFZ0OPSRAP came before
  CAS1OPSRAP (multi-GNSS, many pairs), which the default list now takes on
  the 9 days of 2024 without a CAS0OPSRAP file that have one (085, 181, 182,
  210, 259, 267, 274, 308, 318).
- Long observation names are generated for daily 30-s mixed files only
  (`..._01D_30S_MO.crx.gz`/`.rnx.gz`, any data-source letter). A station that
  publishes only another sampling or type is not named: in the 2024-035
  listings METS00FIN and WTZA00DEU at BKG (`_01D_30S_GO` only), and
  at CDDIS 16 stations with `_01D_15S_MO` only (e.g. AC2300USA, ANTF00CHL)
  and 2 with `GO` only. The RINEX 2 short names still find some of them
  (`wtza0350.24d.gz`); METS00FIN has no generated name at either archive. The
  EUREF page of 2024-035 adds JOEN00FIN, SODA00FIN and VAAS00FIN
  (`_01D_30S_GO` only). A sampling/type preference would be a new setting
  (planned after v1).
- The EUREF archive is modelled for observations only (its navigation and
  product directories are not), and only at BKG; other EPN data centres are
  not supported. Its layout was checked against the archive URLs of 38 EPN
  observation files (downloaded on 2026-10-15) and the EUREF observation page
  of 2024-035.
- The CODE archive layout (`ftp.aiub.unibe.ch`) is modelled from its
  documentation and the CODE products seen at CDDIS / BKG: the server did not
  answer in any network run (2026-10-14, 2026-10-15 19:57-20:17 UTC, 2026-10-16
  04:23-04:24 UTC), so the CODE cells are not verified live.
- OPS long product names (`IGS0OPSFIN_...`) exist from GPS week 2238
  (27 November 2022); older products under short names (`igsWWWWD.sp3.Z`)
  are not generated. MGEX campaign ids exist earlier, under
  `products/mgex/WWWW/` (tried for weeks before 2238 only; assumed from the
  CDDIS documentation, the run of 2026-10-14 saw week 2300 only) and
  CODE_MGEX.
- At CDDIS the daily `.Z` name is tried before `.gz` (the 2024 CDDIS daily
  directories hold `.gz` RINEX 2 files only, so a RINEX 2 request costs one
  404 first); the BKG `MGEX/`
  directory is not modelled; BKG IONEX and ORBEX are not modelled although BKG
  IGSac also serves the CODE GIM and ORBEX (`COD0OPSFIN`).
  BKG `IGS/products/` and `IGSac/products/` list no MGEX campaign products
  (2024 pages), yet MGX ids are still tried there: with BKG as a source the
  default lists cost a few requests that cannot succeed before `COD0OPSFIN`
  or `IGS0OPSFIN` is chosen.
- The IGN and WHU mirrors are not supported; ATX, SINEX and BLQ files are not
  downloaded.
- A cut `.Z` file is detected only through `Content-Length` (see "File
  Integrity").
- Not every station publishes RINEX 3/4 daily files, and a 4-character code
  is not enough to infer a 9-character id.
- `DownloadSession` processes days sequentially; `max_workers` parallelizes
  the probes and request groups within one day.
- A chosen linked product that then fails to download at every source is an
  error for that file; the session never switches products for it.
- The downloader generates known candidates; it is not a web scraper.
