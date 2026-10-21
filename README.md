# GNX-py

<p align="center">
  <img src="assets/GNX_logo_v2.png" width="220" alt="GNX-py logo"/>
</p>

*A modular Python library for GNSS positioning, signal-in-space analysis, and ionosphere research workflows.*

---

## Overview

**GNX-py** is a research and engineering Python library for processing Global Navigation Satellite System (GNSS) data. It is designed for experiments with positioning algorithms, multi-constellation products, orbit and clock quality analysis, and ionosphere/TEC processing.

The library currently focuses on:

- **Precise Point Positioning (PPP)** in combined, uncombined, single-system, and selected mixed-system modes.
- **Orbit and signal-in-space analysis** including broadcast-vs-precise and SP3-vs-SP3 comparisons.
- **STEC/TEC and ionosphere processing** using RINEX observations, GIM/IONEX products, empirical models, and bias products.
- **Modern GNSS data workflows** using RINEX, SP3, ANTEX, SINEX, OSB/DCB/DSB/BIA, and GIM/IONEX-style products.

GNX-py supports **GPS**, **Galileo**, and **BeiDou/BDS**. It is actively developed as open-source research software, so the codebase intentionally exposes both mature workflows and newer validation branches. The README below describes that status explicitly.

### New in GNX-py v1

Version 1.0.0 is the first stable release. Besides many fixes (see [`CHANGELOG.md`](CHANGELOG.md)) it brings:

- **BDGIM**, the BeiDou-3 global ionosphere model of the B-CNAV message (BDS-SIS-ICD-B2b-1.0 §7.7), as `ionosphere_model="bdgim"` next to Klobuchar, NTCM and GIM;
- **ocean tide loading** from BLQ files (`ocean_loading_path`), the second step of the IERS solid Earth tide and the pole tide (`erp_path`);
- **satellite attitude**: published yaw-steering laws per satellite block and ORBEX attitude files (`attitude_path`), one attitude frame for the satellite PCO, PCV and phase wind-up, and a PPP guard for satellite turns (`attitude_turn_guard`);
- **own file readers** for RINEX 2/3/4 observations and navigation, SP3, RINEX clock and ANTEX files, validated on a corpus of real files (the library no longer needs gfzrnx or georinex);
- **downloads** organised like the processing sessions, `DownloadConfig` -> `DownloadSession` -> `DownloadReport`, with BKG, NASA CDDIS, CODE and the EUREF archive at BKG;
- **`SISSession`**, the signal-in-space session (formerly `SISController`);
- a **validation** of every processing branch over a matrix of settings, products, stations and days, with the positioning accuracy limits referred to RTKLIB on the same data (see [Validation](#validation)).

Version 1.0.1 (2026-10-21) is a documentation release: the tutorial and `GNX_manual.pdf` cite the algorithms, models and formats they use.

---

## Current Capabilities

### Supported GNSS systems

- **GPS (`G`)**: mature support across SPP, PPP, STEC, and SIS/SISRE workflows.
- **Galileo (`E`)**: mature support across GPS/Galileo PPP, SPP, STEC, and orbit analysis workflows.
- **BeiDou/BDS (`C`)**: active support through a shared signal metadata layer for RINEX observation selection, broadcast navigation, SPP, PPP, STEC, and SIS/SISRE analysis.

The default BeiDou dual-frequency mode is **`B1IB3I`**. This is the safest current BDS mode in the examples and validation tests. Modern BDS modes such as `B1C`, `B2a`, and `B2b` are represented in the signal metadata and selected processing branches, but they require matching RINEX 4 navigation fields and clock-consistent OSB/DSB products before being treated as production-stable.

### SPP

GNX-py includes Single Point Positioning through `gnx_py.spp`:

- broadcast-orbit SPP as the simplest workflow;
- precise-orbit SPP when SP3 products are supplied;
- GPS, Galileo, BeiDou, and mixed `G/E/C` system selections;
- one receiver-clock state per active constellation in mixed runs;
- configurable ionosphere models (`klobuchar`, `gim`, `ntcm`, `bdgim`, or disabled) and troposphere models;
- an optional GDOP gate per epoch (`spp_max_gdop`, default 30 as in RTKLIB).

Start with:

```bash
python examples/run_spp_example.py
```

### PPP

The `gnx_py.ppp` package implements Precise Point Positioning workflows around `PPPConfig` and `PPPSession`:

- ionosphere-free **combined PPP**;
- **uncombined PPP** with frequency-specific code and phase observations;
- single-system and mixed-system filtering;
- GPS, Galileo, and BeiDou/BDS support;
- precise SP3 orbit/clock workflows and broadcast-orbit validation paths;
- optional ionospheric constraints for uncombined PPP;
- OSB/DCB/DSB-style bias product ingestion;
- configurable EKF presets, trace diagnostics, cycle-slip preprocessing, antenna corrections (PCO and PCV from ANTEX), satellite attitude, solid Earth, ocean-loading and pole tides, relativity, and troposphere handling.

Recommended starting points are GPS/Galileo combined PPP and BDS-only `B1IB3I` PPP. Mixed uncombined modes are powerful but more sensitive to product and bias consistency.

### PPP ambiguity resolution

PPP ambiguity resolution is available as an advanced feature in selected PPP branches:

- enabled by `pppar_enabled=True`;
- LAMBDA integer least-squares search;
- ratio-test screening;
- lock-time / arc-age gating;
- optional elevation and candidate-count gates;
- optional partial fixing;
- soft hold constraints after accepted fixes;
- diagnostic columns for accepted/rejected ambiguity groups.

PPP-AR is **disabled by default**. Combined ionosphere-free AR is additionally guarded by `pppar_combined_if_ar_enabled=True` and should be treated as experimental. Use PPP-AR only after validating the float PPP solution, the phase-bias products, and the selected constellation/signal combination.

### Ionosphere and STEC

The `gnx_py.ionosphere` package provides STEC and ionosphere tools around `TECConfig` and `TECSession`:

- slant TEC measurement from dual-frequency geometry-free observables;
- GPS, Galileo, and BeiDou processing through shared GNSS signal metadata;
- GIM/IONEX reading and interpolation;
- bias policy controlled by `TECConfig.stec_bias_sources`;
- bias sources from products, GIM DCB blocks, manual configuration, or zero fallback;
- OSB, DCB, and DSB-style bias handling;
- Klobuchar, NTCM-style, BDGIM (BeiDou-3), and GIM-based model support;
- IPP geometry, mapping functions, and TEC/VTEC utilities;
- ionospheric activity indices such as SIDX, GIX-family diagnostics, ROT, and ROTI;
- experimental VTEC kriging in geographic or solar-geomagnetic frames.

`TECSession` currently processes one constellation at a time. For BeiDou STEC, `B1IB3I` is the best-covered current mode.

### Orbits, SIS, and SISRE

The `gnx_py.orbits` package provides signal-in-space analysis through `SISConfig` and `SISSession` (formerly `SISController`, which still works as a deprecated alias and emits a `DeprecationWarning` when instantiated):

- broadcast-vs-precise orbit and clock comparison;
- SP3-vs-SP3 comparison on a common epoch grid;
- radial, along-track, and cross-track decomposition;
- SISRE/SIS-style metrics;
- optional satellite PCO correction with ANTEX;
- optional eclipse flagging and post-eclipse extension;
- DCB/OSB/DSB product ingestion where appropriate;
- GPS, Galileo, and BeiDou processing branches.

For BeiDou SIS/SISRE work, `B1IB3I` is the validated/stable starting mode for MEO/IGSO broadcast-vs-SP3 comparisons. BDS clock interpretation depends strongly on the selected signal, broadcast TGD/ISC convention, precise clock datum, and external bias products. Inspect `clock_convention_status` and related metadata before interpreting BDS clock metrics. BeiDou GEO is not supported in the current SIS/SISRE path and is skipped or guarded.

### GNSS data downloads

The `gnx_py.downloaders` package downloads the inputs of a processing run through `DownloadConfig` and `DownloadSession`, and returns a `DownloadReport` with the paths per product type:

```python
from datetime import date
from gnx_py.downloaders import DownloadConfig, DownloadSession
from gnx_py.ppp import PPPConfig

config = DownloadConfig(project_name="brux_2024_035", dates=date(2024, 2, 4),
                        product_types=("obs", "nav", "sp3", "clk", "bia"),
                        stations="BRUX00BEL", output_root="data")
report = DownloadSession(config).run()      # DownloadReport
print(report.summary())                     # one analysis centre for SP3/CLK/OSB
ppp = PPPConfig(obs_path=report.files("obs")[0], nav_path=report.files("nav")[0],
                sp3_path=report.files("sp3"), dcb_path=report.files("bia")[0], ...)
```

- sources: BKG (observations, navigation, products incl. the analysis-centre `IGSac` tree; no IONEX or ORBEX), NASA CDDIS (every type; needs an Earthdata token), CODE (CODE's own products, incl. GIM and ORBEX) and `"euref"`, the EUREF Permanent GNSS Network archive at BKG (observations only, not a default source: e.g. `sources=("bkg", "euref")`);
- RINEX 3/4 long observation names are tried with the data-source letters of `rinex_data_source_preference` (default `("R", "S")`, receiver then stream);
- CDDIS uses an Earthdata bearer token (`token=` or `EARTHDATA_TOKEN`); a username/password or credentials file is exchanged for a token once per run, only when CDDIS is a source, reusing a valid token of the account;
- orbits, clocks, OSB and ORBEX come from one analysis-centre solution for the whole project, the SP3 boundary days included, so PPP-AR gets consistent products; mixing is a reported last resort;
- `product_preferences={"nav": ("BRD400DLR", "BRDC00IGS")}` fetches the RINEX 4 navigation file with the BeiDou-3 BDGIM parameters (`ionosphere_model="bdgim"`), independently of the observation RINEX version;
- files stay compressed as in the archive (`.gz`, `.Z`, `.crx.gz`); GNX reads them directly (see [Reading files](#reading-files)). Every file is checked (whole gzip stream, size, content header) before it is kept or reused;
- `report.outcomes()` gives one line per requested item, including items no selected source provides (`"unavailable"`, with the reason).

See `examples/run_project_download_example.py`, [`gnx_py/downloaders/README.md`](gnx_py/downloaders/README.md) and the section *Downloading data for your own project* of the introduction tutorial.

### Reading files

GNX-py reads its inputs with its own readers: RINEX 2/3/4 observations (`gnx_py.io.rinex_obs`), RINEX 2/3/4 navigation (`gnx_py.io.rinex_nav`), RINEX clock files (`gnx_py.io.rinex_clk`), SP3 (`gnx_py.io.sp3_reader`), ANTEX (`gnx_py.models.antex`), Bias-SINEX, SINEX, IONEX, BLQ and ORBEX. The readers were checked on a corpus of several hundred real files with very different headers.

- Compressed files (gzip, Unix `.Z`, bzip2, zip) are decompressed in memory, without temporary files, so read-only directories and parallel reads of one file are safe.
- Hatanaka-compressed observations (CRINEX 1 and 3, also inside gzip or `.Z`) are decompressed in memory by the RNXCMP `crx2rnx` program that the `hatanaka` package installs. GNX starts it through its internal helper `gnx_py.core._proc.run`, which uses `posix_spawn` instead of `fork` on macOS and glibc Linux, so reading is safe in processes that have loaded PROJ (cartopy, pyproj).
- gfzrnx is not used, and georinex is not a runtime dependency.

---

## Installation

GNX-py is packaged as `gnx_py`. The project metadata currently targets **Python 3.12**:

```text
requires-python = ">=3.12,<3.13"
```

### macOS / Linux editable install

```bash
git clone https://github.com/HJpierzchala/GNX.git
cd GNX

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Windows PowerShell editable install

```powershell
git clone https://github.com/HJpierzchala/GNX.git
cd GNX

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell refuses to run `Activate.ps1` (the default execution policy blocks scripts), allow it for the current window with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, or use the Command Prompt steps of [`INSTALL_WINDOWS_PIP.md`](INSTALL_WINDOWS_PIP.md).

### Optional extras

`pyproject.toml` declares optional dependency groups:

```bash
python -m pip install -e ".[tutorial]"   # Jupyter notebooks of tutorial/ and the manual build
python -m pip install -e ".[dask]"       # dask, for chunked NetCDF output (save_grids_to_netcdf(..., chunks=...))
python -m pip install -e ".[dev]"        # developer tools (pytest, coverage, linters)
```

### Requirements files

The canonical dependency declaration is `pyproject.toml` (runtime dependencies and the `tutorial`, `dask`, `test`, and `dev` extras). The requirements files mirror it for users who prefer requirements-file workflows:

- `requirements-runtime.txt`: the runtime dependencies of the package metadata.
- `requirements-tutorial.txt`: the `tutorial` extra.

### Verify the import

With the intended environment active:

```bash
python -c "import sys, gnx_py; print(sys.executable); print(gnx_py.__file__, gnx_py.__version__)"
```

For an editable install, `gnx_py.__file__` should point to this repository.

Switching an environment from a regular install (`pip install .`) to the editable install: run `python -m pip uninstall gnx_py` first and delete a `gnx_py` folder that may be left in `site-packages`. Numba writes its cache files next to the installed package, and such a leftover folder hides the editable install (`import gnx_py` then finds a namespace package without `__version__`).

---

## Quick Start

Install GNX-py, then run the examples from the repository root:

```bash
python examples/run_spp_example.py
python examples/run_ppp_example.py
python examples/run_sise_example.py
python examples/run_stec_example.py
python examples/run_project_download_example.py
```

The scripts are intentionally small and editable. Each has a `USER SETTINGS` block near the top where you can change the station, constellation set, signal mode, product paths, time window, and output options.

By default, the processing examples run GPS + Galileo + BeiDou over one hour of the sample day of `sample_data/` (each in a few seconds) and write generated files under `examples/output/`; `--full` (or `FULL_DAY = True` in the script) processes the whole day (see `examples/README.md`). The download example is a dry run without network access by default. Some workflows require matching observation, navigation, SP3, ANTEX, GIM, SINEX, and bias files; check the path block in each script before running with your own data.

Common signal-mode defaults:

- GPS: `L1L2`
- Galileo: `E1E5a`
- BeiDou: `B1IB3I`

---

## Basic Python Usage

```python
from gnx_py.spp import SPPConfig, SPPSession
from gnx_py.ppp import PPPConfig, PPPSession
from gnx_py.ionosphere import TECConfig, TECSession
from gnx_py.orbits import SISConfig, SISSession
from gnx_py.downloaders import DownloadConfig, DownloadSession
```

The example scripts are the recommended entry point before building custom workflows.

### Plotting results

`gnx_py.viz` plots the results of each workflow. Every plot returns a
`matplotlib.figure.Figure`, nothing calls `plt.show()`, and single-axes plots
accept an `ax=` argument so they compose into larger layouts:

```python
from gnx_py.viz import apply_publication_style, plot_positioning_series, plot_spp_positioning

apply_publication_style()

# PPP
fig = plot_positioning_series(ppp_result.solution, convergence_time=15, title="PPP: BRUX")
fig.savefig("ppp_convergence.png", dpi=300)

# SPP: position errors plus the per-constellation satellite count
fig = plot_spp_positioning(spp_result.solution, title="SPP: BRUX")
```

See `gnx_py/viz/README.md` for the map, ionosphere and comparison plots.

---

## Project Structure

```text
GNX/
├─ gnx_py/             # Python package: SPP, PPP, ionosphere, orbits, I/O, corrections
│  ├─ core/            # Settings, time and coordinate conversions, signal tables, preprocessing, cycle slips
│  ├─ models/          # Tides, ocean loading, troposphere, wind-up, attitude, ANTEX, a priori position, biases
│  ├─ io/              # Readers: RINEX observations, navigation and clocks, SP3, SINEX, ORBEX, ERP, compression
│  ├─ spp/             # Single Point Positioning: SPPConfig/SPPSession
│  ├─ ppp/             # PPP filters, PPP session routing, PPP-AR helpers
│  ├─ ionosphere/      # STEC/TEC processing, GIM/IONEX, models, monitoring, kriging
│  ├─ orbits/          # SIS/SISRE orbit and clock comparison tools
│  ├─ downloaders/     # GNSS archive downloads: DownloadConfig/DownloadSession (CDDIS, BKG, CODE, EUREF)
│  └─ viz/             # Plotting: PPP, orbits, ionosphere, maps, diagrams
├─ examples/           # Small editable user examples using sample_data
├─ sample_data/        # One day of example data and products (about 110 MB) for the examples and tutorials
├─ tutorial/           # Jupyter notebooks and manual build tooling
├─ assets/             # Static repository assets
├─ GNX_manual.pdf      # Generated manual/tutorial PDF snapshot
├─ pyproject.toml      # Package metadata, dependencies, optional extras
├─ requirements-*.txt  # Mirrors of the runtime dependencies and of the tutorial extra
├─ CHANGELOG.md        # Release notes
├─ CITATION.cff        # Citation metadata
├─ LICENSE             # MIT license
├─ THIRD_PARTY_NOTICES # Third-party code and data used by the repository
├─ .github/workflows/  # Continuous integration: installation check on Linux, macOS and Windows
└─ README.md
```

The modules that were directly under `gnx_py` before v1.0.0 are in `core`, `models`, `io` and `spp` (and `download` in `downloaders`); their old import paths (`gnx_py.rinex_obs`, `gnx_py.tools`, ...) still work and give the same modules.

---

## Documentation and Tutorials

- `examples/` contains small, editable scripts for SPP, PPP, SISE/SISRE, and STEC workflows on `sample_data` and for downloading the inputs of a project (see `examples/README.md`).
- `gnx_py/ppp/README.md`, `gnx_py/ionosphere/README.md`, `gnx_py/orbits/README.md`, and `gnx_py/downloaders/README.md` provide module-level notes for the main processing domains and the downloads.
- `tutorial/` contains Jupyter notebooks that form a longer guided manual.
- `GNX_manual.pdf` is a generated snapshot of the manual/tutorial material.

The tutorial notebooks are the best place for longer exploratory workflows. The top-level README is intentionally a project overview, not a full API reference.

---

## Validation

Before the 1.0.0 release every processing branch (SPP, PPP, ionosphere, orbits, downloads, plots) was run over a matrix of settings, products, stations, and days, and every graded number was compared with an independent reference. The data are public GNSS files of the stations BRUX, NKLG, and NYA2 on 2024-035 (quiet), 2024-131 (storm onset), and 2024-132 (storm). The positioning accuracy limits are referred to RTKLIB 2.4.2 on the same data, `min(2 x design limit, max(1.5 R, R + a))` with the RTKLIB value `R` and a floor `a` fixed before any comparison with GNX-py (the sensitivity of RTKLIB itself to equally valid settings). Configurations that stay outside their limit for an established cause that is not a GNX-py defect are listed as known limitations in [`CHANGELOG.md`](CHANGELOG.md) instead of loosening the limit.

The continuous-integration workflow `.github/workflows/ci.yml` installs the package on Linux, macOS, and Windows with Python 3.12, imports it, and runs `examples/run_spp_example.py` (one hour, GPS + Galileo + BeiDou) on `sample_data/`.

---

## Known Limitations and Validation Status

GNX-py is actively developed research software. Its important branches are validated and have example workflows, but not every constellation, receiver, station, product source, signal mode, and arc condition has the same validation depth.

Current limitations to keep in mind:

- GPS and Galileo workflows are generally the most mature.
- BeiDou support is active, with `B1IB3I` as the safest current default for SPP/PPP/STEC/SIS examples.
- Modern BDS `B1C`, `B2a`, and `B2b` paths require further validation in some processing branches, especially when precise clock datum and OSB/DSB consistency matter.
- BeiDou GEO is unsupported or skipped in current broadcast/SIS/SISRE paths.
- PPP-AR is advanced and disabled by default; it should be enabled only for controlled validation.
- Combined ionosphere-free PPP-AR is experimental and additionally disabled unless `pppar_combined_if_ar_enabled=True`.
- Mixed uncombined PPP, especially with BeiDou and no ionospheric constraints, is sensitive to product and bias consistency.
- Some public classes and exports are retained for legacy/reference compatibility and are not all ideal starting points for new workflows.
- Kriging, some model-comparison paths, and some ionosphere monitoring utilities are research-facing and should be validated for the selected dataset.
- `sample_data/` is meant for smoke tests, tutorials, and regression-style checks. It is not a full scientific validation dataset for every mode.
- GLONASS and QZSS are supported only in SP3-vs-SP3 orbit comparisons (no antenna offsets or group delays).
- Downloads cover BKG, CDDIS, CODE, and the EUREF archive at BKG (no IGN/WHU mirrors, no ATX/SINEX/BLQ files); OPS long product names are generated from GPS week 2238 (November 2022), older short product names are not, long observation names are generated for daily 30 s mixed files (`01D_30S_MO`) only, and archive layouts not confirmed by a listing are marked "assumed" in the product catalogue.
- The validation configurations that stay outside their accuracy limits for an established cause that is not a GNX-py defect are listed as known limitations in [`CHANGELOG.md`](CHANGELOG.md).

These limitations are intentional status markers. They make it easier to use stable paths confidently while keeping experimental GNSS research workflows accessible for validation.

---

## Contributing

Pull requests are welcome, especially for:

- additional validation datasets and reproducible examples;
- documentation and tutorial improvements;
- validation cases for new constellation/signal/product combinations (the validation suite is not part of this repository: describe the data, the settings and the expected result in the pull request);
- performance improvements that preserve numerical behavior;
- careful API cleanup with compatibility notes.

For ideas, bugs, or questions, open a GitHub Issue.

---

## Citation

If you use GNX-py in research, please cite it with the metadata of [`CITATION.cff`](CITATION.cff) (GitHub shows it as "Cite this repository"), together with the GNSS products, standards, and scientific methods used in your workflow.

---

## License

GNX-py is released under the **MIT** license; see [`LICENSE`](LICENSE). Third-party code and data included in the repository keep their own licences and attributions; they (the filterpy Kalman update, the IERS tide routines, the RTKLIB ocean-loading table, phase wind-up and LAMBDA search, the sample data) are listed in [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES). The GNSS data in `sample_data/` and the station observations behind the tutorial TEC files are not under the MIT license: they keep their providers' terms (the IGS terms of use for most products, CC BY 4.0 for the EPN/ROB station observations, CC BY-NC 4.0 for the GFZ bias file), and their producers should be cited as listed there.

---

## Acknowledgements

GNX-py workflows align with GNSS standards and products from the broader geodesy community, including RINEX, SP3, ANTEX, SINEX, IONEX/GIM, IGS/MGEX-style product conventions, and classical PPP and ionospheric modelling literature.

Compact RINEX (Hatanaka-compressed) observation files are decompressed in memory with the RNXCMP `crx2rnx` program of the Geospatial Information Authority of Japan, which the `hatanaka` Python package installs (MIT; it bundles RNXCMP under the GSI terms); GNX starts it through its internal `posix_spawn` helper (`gnx_py.core._proc.run`). The format and the tools are described in:

- Hatanaka, Y. (2008). A Compression Format and Tools for GNSS Observation Data. *Bulletin of the Geographical Survey Institute*, 55, 21–30.
