# GNX-py User Examples

This directory contains small, editable Python scripts that demonstrate the
main GNX-py workflows. Four of them process the repository `sample_data`
(station BRUX, 2024-02-04, 30 s observations, with broadcast navigation, precise
orbits, biases, global ionosphere maps and station coordinates of the same day);
the fifth downloads the inputs of your own project.

The examples are meant to be opened in an IDE and modified. Each file has a
`USER SETTINGS` block near the top where you can change constellations, signal
modes, products, time windows and output options, and a `PATHS` block with the
input files.

## Running

Install GNX-py into the active environment first, from the repository root:

```bash
python -m pip install -e .
```

Then run the examples:

```bash
python examples/run_spp_example.py
python examples/run_ppp_example.py
python examples/run_stec_example.py
python examples/run_sise_example.py
python examples/run_project_download_example.py
```

The scripts import the installed `gnx_py` package and find `sample_data` and
`examples/output/` relative to their own location, so they can be started from
any directory. They do not modify `sys.path`; if an import fails, check that the
virtual environment used for installation is also the active interpreter.

## Defaults: One Hour, GPS + Galileo + BeiDou

The four processing examples use GPS (`L1L2`), Galileo (`E1E5a`) and BeiDou
(`B1IB3I`) and process the first hour of the sample day (00:00-01:00 GPST), so
each run takes a few seconds. To process the whole sample day, add `--full` or
set `FULL_DAY = True` in `USER SETTINGS`:

```bash
python examples/run_ppp_example.py --full
```

`--quick`, the flag of earlier versions, is still accepted and gives the default
one-hour window. Output file names end in `_1h` or `_24h`.

Typical run times on a laptop (Apple M-series, Python 3.12, including the
start of the interpreter):

| Script | One hour (default) | Whole day (`--full`) |
|---|---|---|
| `run_spp_example.py` | about 6 s | about 16 s |
| `run_ppp_example.py` | about 9 s | about 55 s |
| `run_stec_example.py` (three systems) | about 8 s | about 17 s |
| `run_sise_example.py` (three systems) | about 7 s | about 12 s |
| `run_project_download_example.py` (dry run) | about 3 s | - |

## SPP: `run_spp_example.py`

Single Point Positioning with `SPPConfig` + `SPPSession`: ionosphere-free code
of the three systems with broadcast orbits and clocks and the Saastamoinen
troposphere, one receiver clock per constellation. `ORBIT_TYPE = "precise"`
uses the COD MGEX SP3 files instead. `IONOSPHERE_MODEL` (Klobuchar by default)
is used only by single-frequency modes such as `L1`, `E1` or `B1I`.

Output: the epoch solution `spp_BRUX_GEC_broadcast_1h.csv` with the ENU errors
against the SINEX coordinates; `MAKE_PLOT = True` adds a PNG of the ENU errors
(`gnx_py.viz`). The summary prints the RMS of the errors and the residual rows
per system (one hour: 3D RMS about 2.7 m; whole day: about 1.5 m).

## PPP: `run_ppp_example.py`

Precise Point Positioning with `PPPConfig` + `PPPSession`: uncombined PPP (one
code and one phase per frequency) of the three systems, the CODE GIM as
ionospheric constraint, static coordinates and PPP ambiguity resolution
(PPP-AR).

- Orbits, clocks and code/phase biases come from one solution, CNES/CLS GRG
  (`GRG0MGXFIN` SP3 and OSB), as PPP-AR needs: phase biases of another solution
  than the clocks do not give integer ambiguities, and GNX-py then keeps the
  float solution with a warning. `sample_data` has no COD MGEX OSB, and the
  `COD0OPSFIN` OSB has no BeiDou.
- `BIAS_PATH` is a list: the GRG OSB and the CAS DSB, which gives the BeiDou
  C2I-C6I code bias that refers the GIM constraint to the code-bias datum.
- `POSITIONING_MODE = "combined"` gives the ionosphere-free filters (they fix no
  ambiguities); `PPP_AR_ENABLED = False` the float solution.

Output: the epoch solution `ppp_BRUX_GEC_uncombined_1h.csv` with ECEF `x`, `y`,
`z`, the ENU errors `de`, `dn`, `du` against the SINEX coordinates, the
convergence time `ct_min` [min] and the PPP-AR diagnostics (`ar_fixed`, ...);
with `WRITE_RESIDUALS = True` (default) one residual CSV per system;
`MAKE_PLOT = True` adds a PNG of the solution time series (`gnx_py.viz`). The
summary prints the convergence time, the final 3D error, the 3D RMS after
convergence and the epochs with fixed ambiguities. On the sample data:

| Window | Convergence | Final 3D error | 3D RMS after convergence | Fixed epochs |
|---|---|---|---|---|
| one hour | 33 min | 1.7 cm | 3.0 cm | 1 of 121 |
| whole day | 33 min | 0.6 cm | 0.8 cm | 2619 of 2880 |

PPP-AR needs time: in the one-hour window the filter has only just converged
and most fix attempts are still rejected by the ratio test (one fixed epoch);
in the whole-day run 91 % of the epochs are fixed.

## STEC: `run_stec_example.py`

Slant TEC with `TECConfig` + `TECSession`, which processes one constellation per
run: the script runs GPS, Galileo and BeiDou one after the other (`SYSTEMS`),
with broadcast orbits, the CODE GIM and an explicit bias policy
(`BIAS_SOURCES`, `RECEIVER_DCB_SOURCE`). With `RECEIVER_DCB_SOURCE = "gim"` the
GPS and Galileo arcs take the satellite and receiver DCBs of the GIM (one datum
per arc); the GIM has no BeiDou receiver DCB of BRUX, so BeiDou takes the DSBs
of the CAS product in `DCB_PATH` (with one warning). The phase is levelled to
the code over each whole continuous arc (`leveling_ws=None`) and the satellite
antenna offsets are applied along the line of sight (`sat_pco="los"`), as in
the ionosphere tutorial.

Output: levelled STEC per satellite and epoch, one file per system
(`stec_BRUX_G_1h.csv`, `stec_BRUX_E_1h.csv`, `stec_BRUX_C_1h.csv`), in
electrons/m² (divide by 1e16 for TECU), with the bias sources used;
`WRITE_FULL_OUTPUT = True` writes every column. The summary prints the STEC
range and the levelling spread per system.

## SISE/SISRE: `run_sise_example.py`

Signal-in-space error with `SISConfig` + `SISSession`: the broadcast orbits and
clocks of GPS, Galileo and BeiDou against the CODE MGEX SP3, every 300 s, one
system after the other (`SYSTEMS`). BeiDou GEO satellites are skipped; for
BeiDou the script prints the clock-convention metadata
(`clock_convention_status`, `clock_convention_note`).

Output: one per-satellite summary per system
(`sise_<system>_<mode>_1h_summary.csv`) of the radial, along-track and
cross-track orbit errors, the clock error as it enters the SISRE (95th
percentile of `|(dt - dt_mean) - (dTGD - dTGD_mean)|`, after removing the epoch
clock and group-delay datums) and the SISRE; `WRITE_DETAIL_CSV = True` adds the
per-epoch rows (with the raw `dt` and the datums). The summary prints the SISRE
RMS per system (whole day: GPS 0.51 m, Galileo 0.21 m, BeiDou 1.25 m).

## Download: `run_project_download_example.py`

Downloads the inputs of a processing project with `DownloadConfig` +
`DownloadSession` into one folder, `examples/output/projects/<PROJECT_NAME>/`.
The defaults request the sample day (2024-02-04) of station BRUX (`"BRUX"` is
mapped to its RINEX 3 id `BRUX00BEL` by the library; other 4-character codes go
to `STATION_ALIASES`) with everything a G/E/C run needs: observations,
broadcast navigation, SP3 orbits with the day before and after, 30 s clocks,
OSB biases, DSB biases and IONEX maps. Orbits, clocks and OSB come from one
analysis-centre solution (`COD0MGXFIN` by default; `PRODUCT_PREFERENCES =
{"sp3": ("GRG0MGXFIN",)}` gives the GRG set of the PPP example). The sources are
NASA CDDIS, BKG, CODE and the EUREF archive (observations of EPN stations).

- `DRY_RUN = True` (default) downloads nothing and lists the files that would
  be downloaded; with `CHECK_AVAILABILITY = False` and `CREDENTIALS_FILE = None`
  (the defaults) the run makes no network request (a `CREDENTIALS_FILE` is
  exchanged for a token at NASA Earthdata in every run, a dry run included).
  `CHECK_AVAILABILITY = True` asks the archives which files exist;
  `DRY_RUN = False` downloads.
- CDDIS needs a NASA Earthdata token. Do not write it into the script: set the
  environment variable `EARTHDATA_TOKEN`, or point `CREDENTIALS_FILE` to a
  private, git-ignored file with `EARTHDATA_USERNAME=...` and
  `EARTHDATA_PASSWORD=...` lines, which the library exchanges for a token.
  Without a token CDDIS files are reported as `auth_missing` and the next source
  is tried.

The script prints the report summary, the chosen product solution, one line per
requested file and, after a download, the paths for `PPPConfig`. The time
window of the processing examples does not apply here: the archives hold daily
files.

## Warnings You Will See

The scripts do not silence warnings: they report conditions of the data and
products that matter for the results, once per run.

- All processing examples: a `RinexWarning` that a BeiDou Klobuchar record
  (`BDSA C50`) of the navigation file is rejected (the other BeiDou records are
  used).
- SPP, STEC and SISE (broadcast orbits): `G01 (unhealthy, health=63)` - the
  broadcast navigation marks G01 unhealthy on this day, so it is left out;
  BeiDou GEO satellites are skipped; SPP and STEC: Galileo E18 (and E14 over
  the whole day) unhealthy for `E1E5a`.
- PPP: `AntexVersionWarning` - the `igs20.atx` file of `sample_data` (IGS20 week
  2437) is newer than the antenna model the GRG products were computed with
  (IGS20_2290); the attitude turn guard drops a few BeiDou observations near
  noon/midnight turns (C28 in the first hour).
- STEC: the GIM has no BeiDou receiver DCB of BRUX (see above).
- SISE: the COD SP3 files disagree on the C11 position at midnight (that node is
  not used).

## Choosing Systems and Signals

Common system selections (`SYSTEMS`):

- GPS: `{"G"}`; Galileo: `{"E"}`; BeiDou: `{"C"}`
- Mixed SPP/PPP: `{"G", "E"}`, `{"G", "C"}`, `{"G", "E", "C"}`

STEC and SISE take a tuple of systems processed one after the other, e.g.
`SYSTEMS = ("G",)`.

Common signal modes:

- GPS: `L1L2`
- Galileo: `E1E5a`
- BeiDou: `B1IB3I`

`B1IB3I` is the BeiDou pair of the 2024 precise clocks and bias products and
the safest default. Modern BDS modes such as `B1C`, `B2a` and `B2b` require
RINEX 4 navigation files and matching clock and OSB/DSB products and should be
treated as validation/advanced workflows. `sample_data` contains one RINEX 4
navigation file, `BRD400DLR_S_20240350000_01D_MN.rnx`, which also carries the
BDGIM ionosphere coefficients (`IONOSPHERE_MODEL = "bdgim"`).

## Advanced / Experimental Options

- PPP-AR works on the uncombined routes (dual frequency: wide lane, then narrow
  lane; single frequency only with ionospheric constraints). The combined
  ionosphere-free filters and single-frequency PPP without constraints fix no
  ambiguity (with `pppar_enabled=True` they warn and return the float
  solution). BeiDou AR needs careful validation of the phase biases.
- `TRACE_FILTER = True` in the SPP and PPP examples prints compact filter
  diagnostics.
- Modern BDS SIS clock comparisons can show large clock errors when the
  broadcast and precise clock datums do not match. Inspect
  `clock_convention_status` before interpreting BDS clock metrics.

## Your Own Data

To process other stations or days, download the observations and a consistent
product set with `run_project_download_example.py` (see also
`gnx_py/downloaders/README.md`) and point the paths of the processing examples
to the downloaded files. GNX-py reads compressed files (`.gz`, `.Z`, `.crx.gz`)
directly.

## Outputs

All scripts write under `examples/output/`, a directory for local generated
files that is not part of the repository.
