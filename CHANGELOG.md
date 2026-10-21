# Changelog

All notable changes to GNX-py are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-10-21

Documentation release.

### Changed

- Documentation: the tutorial and the manual cite the algorithms, models and formats they use (67 new references,
  fixes of existing ones). All seven tutorial notebooks cite their sources where the text names an algorithm, a
  model, a file format, a product or an interface document: the observation formats (RINEX 2.11, 3.05 and 4.02,
  RINEX clock, SP3-d, ANTEX, SINEX, Bias-SINEX, IONEX, ORBEX), the GPS, Galileo and BeiDou interface documents,
  the broadcast ionosphere models (Klobuchar, NTCM-G, BDGIM), the troposphere models, the IERS 2010 site
  displacement, the satellite attitude laws, the phase wind-up, the Kalman filter, the PPP ambiguity resolution
  (LAMBDA, MLAMBDA, ratio test, partial fixing), the analysis-centre products, the ionospheric activity indexes
  and kriging. The SISE definition is now cited from the Galileo OS Service Definition Document, and the eclipse
  condition from Montenbruck and Gill (2000).
- Documentation: the RTKLIB 2.4.2 manual (Takasu 2013) is cited where GNX-py follows RTKLIB: the routines it ports
  (ocean tide loading, phase wind-up, LAMBDA/MLAMBDA), the GDOP limit of SPP, the ZTD process noise, the
  fix-and-hold constraint, the receiver and satellite antenna offsets and the ionospheric pierce point. The
  activity-index chapter no longer calls the IPP coordinates geocentric.
- Bibliography (`tutorial/bibliography.bib`): corrected authors (Mannucci et al. 1998, Carmo et al. 2021), titles
  (Klobuchar 1987, IS-GPS-200N, IONEX Version 1, NTCM-G), institution names (UNB TR 203), years and entry types
  (IS-GPS-200N, DSP Guide, Glaner 2022), added DOIs (Springer Handbook, Kouba and Héroux 2001, Glaner 2022) and
  issue numbers, and replaced links that no longer answer (IS-GPS-200N, Galileo OS SIS ICD, IONEX, Schaer 1999) or
  were missing (UNB TR 203, Hatanaka 2008).

## [1.0.0] - 2026-10-17

GNX-py 1.0.0 is the first stable release. Before the release every processing branch (SPP, PPP, ionosphere,
orbits, downloads, plots) was run over a matrix of settings, products, stations and days (BRUX, NKLG and NYA2 on
2024-035, 2024-131 and 2024-132) and every graded number was compared with an independent reference; the
positioning accuracy limits are referred to RTKLIB 2.4.2 on the same data. The defects found were fixed and the
configurations that stay outside their limits for an established cause are listed under "Known limitations".
Results on the same inputs differ from version 0.1.0: "Output changes" under Changed lists what changes numbers,
columns or result schemas, followed by the changed defaults and the new warnings.

### Added

Readers and data handling

- Own readers for every input format, validated on a corpus of 564 real files: RINEX 2/3/4 observations
  (`gnx_py.io.rinex_obs`), RINEX 2/3/4 navigation (`gnx_py.io.rinex_nav`), RINEX clock files (`gnx_py.io.rinex_clk`),
  SP3 (`gnx_py.io.sp3_reader`) and ANTEX (`gnx_py.models.antex`). Compressed files (gzip, Unix `.Z`, bzip2, zip) are read
  in memory (`gnx_py.io.compression`); Hatanaka-compressed observations (CRINEX 1/3) are decompressed in memory by
  the RNXCMP `crx2rnx` program bundled with the `hatanaka` package. gfzrnx is no longer used, and georinex is
  no longer a runtime dependency.
- Every program the library starts goes through `gnx_py.core._proc.run`, which uses `posix_spawn` instead of
  `fork` on macOS and glibc Linux (a forked child of a process that had loaded PROJ crashed before `exec`).
- Settings validation (`gnx_py.core.validation`): an unknown value is rejected with the list of valid values and a
  "did you mean ...?" hint instead of being replaced silently.

Models

- BDGIM, the BeiDou-3 global ionosphere model of BDS-SIS-ICD-B2b-1.0 section 7.7
  (`gnx_py.ionosphere.bdgim`; `ionosphere_model="bdgim"` in SPP, PPP and TEC model comparisons), read from the
  B-CNAV ionosphere records of RINEX 4 navigation files.
- Ocean tide loading from BLQ files of the Onsala service (`ocean_loading_path`, `gnx_py.models.ocean_loading`), the
  second step of the IERS solid Earth tide (`solid_tide_step2`) and the pole tide with an IGS ERP file
  (`erp_path`, `pole_tide_mean_pole`; `gnx_py.io.erp`).
- Satellite attitude: published yaw-steering laws per satellite block (`gnx_py.models.attitude`), ORBEX attitude files
  (`attitude_path`, `gnx_py.io.orbex`), one attitude frame for the satellite PCO, the PCV azimuth and the phase
  wind-up (`satellite_attitude`, `attitude_laws`), and a PPP turn guard for products whose attitude differs
  from the model (`PPPConfig.attitude_turn_guard`, `"auto"` / `"all"` / `"off"`).
- Receiver and satellite PCV, PCO of every signal from the signal table, and an `AntexVersionWarning` when the
  ANTEX week differs from the PCV label of the SP3 file.
- Cycle-slip detection layers in a new module `gnx_py.core.cycle_slips` (LLI flags of every phase, receiver clock
  jump guard, close-pair and single-frequency tests, a geometry layer); the filters only see arc labels.
- `SPPConfig.spp_max_gdop` (default 30, as RTKLIB): SPP epochs with a larger GDOP are not reported.
- New settings: `Config.clk_gap_policy` (`"drop"` default, `"sp3"`;
  `configuration.CLK_GAP_POLICIES`), `Config.bds_precise_clock_datum` (`"auto"`, `"B1IB3I"`, `"B1CB2a"`;
  `configuration.BDS_PRECISE_CLOCK_DATUMS`), `PPPConfig.clock_prior_sigma_m` (100 m, RTKLIB `VAR_CLK`) and
  `PPPConfig.uncombined_iono_code_var_by_system` (`{"G": 0.3, "E": 0.3, "C": 1.0}` m² at the zenith). New
  dataclass fields in the middle of `Config` / `PPPConfig` shift the positional argument order: create
  configurations with keyword arguments.
- BeiDou clock datum tools in `gnx_py.models.biases` (`PreciseClockDatum`, `detect_bds_clock_datum`,
  `bds_satellite_generations`, `unmoved_code_variance_m2`, `UNMOVED_CODE_SIGMA_FLOOR_M`, ...) and
  `solution.attrs["precise_clock_datum"]` on SPP and PPP results.
- PPP-AR diagnostics: output columns `ar_excluded_half_cycle`, `ar_phase_variance_factor`,
  `ar_reason_half_cycle`; the per-observation column `lli_half_cycle` (`PPPSession.HALF_CYCLE_COLUMN`).
- Observation reader exports for the RINEX `SYS / PHASE SHIFT` records (`parse_phase_shift_records`,
  `phase_shift_map`, `phase_shift_of`, `phase_shift_events_of`) and LLI constants (`LLI_SLIP`, `LLI_HALF_CYCLE`,
  `LLI_BOC`); every observation frame carries `attrs["phase_shift"]` and `attrs["phase_shift_events"]`.
- `SessionExecutionError.message`, and the exception can be pickled.
- `gnx_py.__version__` and `CITATION.cff`.

Downloads

- The downloads package `gnx_py.downloaders` is organised as `DownloadConfig` -> `DownloadSession` ->
  `DownloadReport`, like the other sessions: validation of the whole configuration
  (`gnx_py.downloaders.validation`), a product catalogue (`catalog`), one analysis-centre product set for a
  whole project (`product_sets`), integrity checks of every file before it is kept or reused, and
  `report.outcomes()` with one line per requested item.
- Source `"euref"`: the EUREF / EPN observation archive at BKG (observations only, no token, not a default
  source; e.g. `sources=("bkg", "euref")`).
- `DownloadConfig.rinex_data_source_preference` (default `("R", "S")`): the data-source letter of RINEX 3/4 long
  observation names; a station that publishes only a stream (`_S_`) file is found.
- Catalogue entries `CAS0OPSRAP`, `GFZ0OPSRAP`, `CAS1OPSRAP` (daily DCB; `GFZ0OPSRAP` also SP3, CLK and ERP).

Plots

- `gnx_py.viz`: every public plotting function closes the figures it created when it raises
  (`no_figure_leak`), and inputs are validated before a figure is created.

Project files and packaging

- `LICENSE` (MIT) and `THIRD_PARTY_NOTICES` (filterpy, the IERS tide routines, the RTKLIB ocean-loading table
  in `gnx_py/models/ocean_loading.py`, the attribution of the sample data and of the tutorial STEC network), both
  declared as licence files of the package (PEP 639).
- Optional extras `tutorial` (the notebooks and the manual build), `dask` (chunked NetCDF output of the kriging
  grids), `test` and `dev` (development tools).
- The sample data the examples and tutorials need are in the repository (COD0MGXFIN SP3 of 2024-034..036,
  GRG0MGXFIN SP3 of 2024-034 and 036, the BRD400DLR RINEX 4 navigation file of 2024-035).
- Continuous integration (`.github/workflows/ci.yml`): installation on Ubuntu, macOS and Windows with Python
  3.12, `import gnx_py` with the version check, and the SPP example (its default one-hour run) on `sample_data`.

### Changed

Output changes (numbers, columns, schemas)

- **PPP solution schema** (`PPPResult`): `solution` always has `x`, `y`, `z` [m] and `ct_min` on every route,
  also when it is empty (`PPPResult.SOLUTION_COLUMNS`; `solution` is never None). `ct_min` is the convergence
  time in **minutes** (NaN when not converged); it held hours before, and the single-system ionosphere-free filters
  and the GPS + Galileo uncombined filter with ionospheric constraints (`PPPFilterMultiGNSSIonConst`) had no
  `ct_min`; the latter named the position columns `xr`, `yr`, `zr`.
- **PPP residuals by system**: `residuals_gps`, `residuals_gal` and `residuals_bds` hold the residuals of their
  own system; a Galileo-only IF run fills `residuals_gal` (it filled `residuals_gps`), a GPS-only uncombined run leaves
  `residuals_gal` None (it was a copy of the GPS frame), and a processed system without rows gets an empty frame
  with the usual columns. The internal `run_filter` of every filter returns
  `(solution, {system letter: residual frame}, None, hours)`.
- PPP prefit (innovation) residuals of the ionosphere-free filters, the single-frequency GRAPHIC filter (the
  code/phase average) and `PPPFilterMultiGNSSIonConst` are taken after the clock prior:
  metre-level instead of the full receiver clock offset (plots of `vc1` change).
- **SPP column `V_rejected`**: the residual of an observation removed by the outlier loop, relative to the final
  solution of its epoch. `V` keeps its meaning (finite value = observation used). `plot_spp_residuals` draws
  `V_rejected` as black crosses.
- **SPP applies the antenna height** (`antenna_h=True`, the default shared with PPP): the header's ANTENNA:
  DELTA H/E/N is removed along the line of sight as in the PPP filters, so SPP positions refer to the marker and
  `de`, `dn`, `du` against the SINEX marker no longer contain the antenna height (2024-035, GPS L1L2, broadcast
  orbits: BRUX mean up error +0.773 -> +0.304 m, 3D RMS 2.427 -> 2.321 m; NKLG, DELTA H 3.043 m: +4.559 -> +1.516 m,
  5.128 -> 2.794 m). `antenna_h=False` gives the antenna reference point solution (the point SPP referred to in 0.1.0).
- SPP residual rows of an epoch with fewer observations than parameters are flagged `spp_rejected=True`,
  `spp_reject_reason="insufficient_observations"`.
- PPP and SPP with a station that is not in the SINEX file: one UserWarning and NaN `de`, `dn`, `du` (PPP gave
  0.0, SPP had no such columns). PPP without `sinex_path` gives NaN `de`, `dn`, `du` too (0.0 before).
- SPP and PPP use the SINEX solution valid at the first observation epoch, propagated with its velocities:
  `gnx_py.io.parse_sinex(sinex_path, epoch=None, sites=None)` has the new `epoch` and `sites` arguments and
  reads SOLUTION/ESTIMATE only (a file with APRIORI after ESTIMATE gave the a-priori values).
- `SPPResult.to_netcdf` / `PPPResult.to_netcdf` accept a str path; dict-valued attributes are written as text.
- `gnx_py.orbits.SISConfig` is no longer a dataclass: `==` and `repr` cover its 26 settings;
  `dataclasses.fields` / `asdict` raise TypeError.
- BeiDou SIS: `sisre`, `sisre_notgd`, `dt_mean`, `dTGD_mean` per satellite generation (BDS-2, BDS-3); precise
  BDS-3 clocks referred to IF(B1C,B2a) are converted for B1IB3I with the B-CNAV group delays of a RINEX 4
  navigation file and reported "ok" (with `compare_dcb=True` it stays "warning"; without a RINEX 4 navigation
  file it reports "warning", directly comparable False); without an ANTEX file the SISRE weight class comes from
  the orbit; the bias-product observable of B1C / B2a is taken pilot first (P, X, D).
- `gnx_py.core.gnss.SIGNALS["B1C"].suffix_priority` is `("X", "P", "D")`; the B1A attributes S / L / Z were removed.
- **Single-frequency GPS PPP** (L1, L2, L5) takes the code and the phase of one tracking (e.g. C1C with L1C and
  S1C instead of C1W / S1W on a Septentrio receiver; `GNSSDataProcessor2(gps_sf_same_tracking=...)`, turned on
  by `PPPSession`): NYA2 GPS L1 0.261 -> 0.217 m.
- **One bias source per ionosphere-free pair**: a mix of OSB and DSB products takes both codes of a pair from the
  DSB product (SPP two-code combinations too); before, the two codes could come from different products (Galileo
  E5aE5b PPP 2.21 -> 1.06 cm). The OSB + DSB warning below says which source was used.
- The STEC phase-to-code levelling uses the whole arc (`leveled_tec`; new column `leveling_std_tecu`).
- Plots: `skip_obs=k` removes exactly k epochs per arc (it removed k+1); `df_tail(nrows=0)` shows the header only;
  a single column name given as a string is one column (`columns="de"`, `model_cols="ion"`); naive epochs are
  labelled "Time [GPST]".

Changed defaults and behaviour

- PPP: phase standard deviation 1 cm in uncombined PPP (it was 1 m); diagonal coordinate process noise
  `q_crd`; the process noises `q_tro`, `q_iono`, `q_isb`, `q_dcb` and `q_amb` are random-walk rates [m²/s]
  scaled by the epoch interval (an explicit value means something else than before). `q_tro` defaults to 1e-8
  m²/s as in RTKLIB (the former per-epoch value was about 800 times larger in the combined and 80 000 times in
  the uncombined filters); the other defaults give the former values at 30 s.
- PPP: `uncombined_iono_legacy_weighting` defaults to False: the GIM-constrained routes weight the code with
  `c_sys / sin E` (`uncombined_iono_code_var_by_system`) and the phase with `uncombined_phase_var / sin E`;
  True restores the former model. The single-frequency constrained route uses the same per-system code weights.
- PPP: the receiver clock (and an inter-system bias started from `p_dt`) is re-initialised from the code with
  variance `clock_prior_sigma_m`² at every epoch where its predicted variance exceeds it, so `q_dt` above 1e4 m²
  no longer has any effect and `p_dt` has none with the white-noise `q_dt` of the presets (1e9 / 9e9 m²);
  `q_amb` acts on every ambiguity state of every route; rows with R >= 1e10 m² are left out of the Kalman update.
- PPP: the GIM ionospheric constraint is referred to the satellite code-bias datum of the bias product (IONEX
  satellite DCBs of the same signals, else a DSB product in `dcb_path`); SPP and single-frequency PPP apply the
  same rule to the GIM correction. The Klobuchar / NTCM constraint sigma is hypot(50 % of the delay, 5 TECU x the
  mapping function). `use_iono_rms=True` adds ion_rms² to the constraint ramp (it multiplied the ramp).
- GPS Klobuchar and Galileo NTCM use the broadcast coefficient set in force at each epoch (RINEX 4 `> ION`
  records), not one set per run.
- BeiDou: precise BeiDou runs move the codes to the datum of the precise clocks (`bds_precise_clock_datum`);
  BDS-3 satellites whose codes cannot be moved are down-weighted (at least 3 m added to the code sigma) in SPP and
  PPP; the band-7 code with broadcast clocks follows the satellite generation, not the PRN.
- An OSB bias product is recognised by its Bias-SINEX header, not by "OSB" in the file name.
- Positioning with broadcast orbits applies no satellite antenna offset for any system (TEC keeps it); GPS SIS
  compares at the broadcast reference point; BeiDou precise orbits are moved to the B3I antenna phase centre
  against broadcast orbits.
- Galileo broadcast validity 3 h (was 4 h); a record already transmitted at the epoch is preferred.
- Downloads: an explicit `GFZ0OPSRAP` preference generates SP3 05M only (was 05M then 15M), CLK 05M only (was
  30S only) and ERP 01D_01D only (was 12H then 01D_01D).
- `attitude_turn_guard="auto"` also acts in dual-frequency kinematic PPP.
- PPP-AR: the LAMBDA integer least-squares functions `ldldecom`, `decorrel` and `ssearch` are a port of
  RTKLIB's `lambda.c` (LAMBDA reduction and MLAMBDA search; BSD 2-clause, see THIRD_PARTY_NOTICES) with the
  same signatures and bit-identical results (fixed ambiguities, partial-AR subsets, ratios).
- RINEX 3 `SYS / PHASE SHIFT` header values are no longer added to the phases (they were applied twice).
- PRN reassignment inside a run: a PRN whose satellite has no ANTEX entry from an epoch inside the run on gives
  one UserWarning and its rows after that epoch are excluded.
- TEC: the geometry layer of the cycle-slip detector also runs with broadcast orbits.
- `TECSession.run` raises a ValueError (inside `SessionExecutionError`) when no arc is left.
- SIS SP3 interpolation runs in the calling process without a process pool (about 1.8 times slower; no
  "fell back to sequential execution" warning).
- Downloads: the default DCB preference is `("CAS0OPSRAP", "CAS1OPSRAP", "GFZ0OPSRAP", "CAS0MGXRAP",
  "DLR0OPSFIN")` (the GFZ0OPSRAP files hold one classic DSB pair per system); default observation candidates
  include the `_S_` names after the `_R_` ones; files stay compressed as served and are read directly.
- Warnings are given once per run (they were once per configuration object or per process); `crx2rnx` warnings
  are attributed to `gnx_py`. The SP3 node check of the precise-orbit interpolation follows the same rule: the
  sessions pass their configuration to `lagrange_reception_interp` / `lagrange_emission_interp` (new `owner`
  argument), so a G+E PPP run reports a disagreeing day-boundary node once instead of four times; a direct call
  without `owner` warns at every call.

Package layout

- The modules that were directly under `gnx_py` are grouped into subpackages: `gnx_py.io` (the readers `rinex_obs`,
  `rinex_nav`, `rinex_clk`, `rinex2_adapter`, `sp3_reader`, `orbex`, `erp` and the in-memory decompression
  `compression`; the package itself is the former module `gnx_py.io`), `gnx_py.models` (`tides`, `ocean_loading`,
  `troposphere`, `corrections`, `attitude`, `antex`, `apriori`, `biases`), `gnx_py.spp` (the former module
  `gnx_py.spp`) and `gnx_py.core` (`configuration`, `session_errors`, `session_guards`, `validation`, `time`,
  `gnss`, `conversion`, `coordinates`, `utils`, `tools`, `cycle_slips` and the internal `_notes` and `_proc`);
  `download` moved to `gnx_py.downloaders`. The old paths keep working: `gnx_py.rinex_obs`, `gnx_py.tools`,
  `gnx_py.download`, ... are the same module objects as the new paths (private names and monkeypatching included),
  and `import gnx_py` gives the same names as before plus `core`. The name `gnx_py.models` is the new subpackage;
  before, it was bound to the module `gnx_py.ionosphere.models`, which keeps its path and whose names still resolve
  as attributes of `gnx_py.models` (`gnx_py.models.klobuchar`). Results are unchanged.
- Module, class and logger names follow the new paths (for example the loggers `gnx_py.core.cycle_slips` and
  `gnx_py.io.rinex_clk`); the `gnx_py` logger covers them as before. Warnings raised in a moved module are
  attributed to its new path, so a logger setting or a warnings filter that names an old module path
  (`getLogger("gnx_py.cycle_slips")`, `-W ignore::UserWarning:gnx_py.antex`) has to use the new one
  (`gnx_py.core.cycle_slips`, `gnx_py.models.antex`).

Packaging

- `pyproject.toml` is the single source of the dependencies; `requirements-runtime.txt` and
  `requirements-tutorial.txt` mirror it. The xarray bound `<2026.0` is now in the package metadata (it was only in
  the requirements files), the `tutorial` extra allows `ipykernel` 7, and the licence is an SPDX expression
  (`license = "MIT"`; the build needs setuptools >= 77).
- Version 1.0.0, development status "5 - Production/Stable", Python 3.12 only (`requires-python =
  ">=3.12,<3.13"`).

New warnings and errors

- New warnings: BeiDou clock datum assumed or disagreeing with the products; BDS-2 / BDS-3 GEO taken in
  IF(B1I,B3I); BDS-3 satellites whose codes cannot be moved to the clock datum; OSB re-referenced to the datum;
  GPS Block IIF L5 inter-frequency clock bias (L1L5 / L2L5 / L5 with precise clocks); OSB + DSB source of one
  pair; `SYS / PHASE SHIFT` changes inside the data; a GIM-constrained run with OSB code biases and no reference
  DSB for a system; a PRN reassigned inside the run; a station missing from the SINEX or BLQ file; a BeiDou
  generation with fewer than 3 satellites at an epoch (SIS).
- UserWarning "`<setting>=<value>` is not used ... removed after GNX-py v1" for settings that change nothing:
  `interpolation_method`, `broadcast_tolerance`, `output_dir` (every session config, at a non-default value),
  `clock_process`, `p_N`, `q_N` (PPPConfig, at a value other than the preset of the mode), and the
  `IonoKrigingMonitor` settings `n_closest_points`, `backend`, `use_external_drift`.
- New ValueErrors with readable messages, among them: precise BeiDou runs whose clocks cannot be matched to the
  observation codes (they ran about 11 m off), plot inputs checked before drawing (negative `cut_init_time` /
  `skip_obs`, conflicting colour limits, map extents outside the projection domain, grids without a common node;
  `make_roti_snapshots` checks `out_dir` and every frame file name before the first frame),
  and unknown setting values everywhere.

Documentation, tutorials and examples

- The seven tutorial notebooks (introduction, PPP, orbit and SIS analysis, ionosphere introduction, calibration,
  activity indexes, kriging) describe the v1 library: the own RINEX / SP3 / clock readers instead of gfzrnx and
  georinex, `DownloadSession` (shown, not executed), `SISSession`, broadcast record selection, satellite attitude
  laws and ORBEX, ocean tide loading, BDGIM, the BeiDou precise-clock datum, the PPP result schema (`x`, `y`, `z`,
  `ct_min` in minutes, residuals by system, NaN `de`/`dn`/`du` without SINEX), the observation variances of each
  PPP filter class as in the code, the PPP-AR scope, the SPP `V_rejected` column and naive GPST epochs with GPST
  labels. No tutorial filters warnings: the warnings a run gives are printed and explained. The notebooks are
  stored without outputs.
- `GNX_manual.pdf` is rebuilt from all seven chapters executed on the sample data (`python tutorial/build_manual.py
  --clean --execute`); the build shows absolute paths of the build machine in notebook outputs as `<repo>` and `~`.
- The STEC network of the activity-index and kriging tutorials (`tutorial/ionosphere/network/`, 38 EPN stations,
  2024-02-04) is now in the repository (the files were never tracked, so both notebooks failed on a fresh clone)
  and is recomputed with this version, with the receiver DCB calibrated against the CODE GIM: 75 files (LAMA has
  no Galileo file: almost every Galileo E5a phase is flagged with a loss of lock). The four
  `sample_data/output/*_STEC.parquet.gzip` files are recomputed as well.
- Examples: five scripts. `run_spp_example.py`, `run_ppp_example.py`, `run_stec_example.py` and
  `run_sise_example.py` run on `sample_data` from any directory with GPS + Galileo + BeiDou over one hour by
  default (`--full` or `FULL_DAY = True`: the whole sample day; `--quick` is still accepted and gives the default
  window). The PPP example runs uncombined PPP with the GIM ionospheric constraint and PPP-AR on the GRG orbits,
  clocks and OSB of one solution (plus the CAS DSB for the BeiDou constraint datum); the SIS and STEC examples
  process the three systems one after the other; the STEC example levels whole arcs and applies the satellite
  antenna offsets along the line of sight, like the ionosphere tutorial. `run_project_download_example.py` shows
  `DownloadConfig` + `DownloadSession` for the inputs of a G/E/C project (a dry run without network access by
  default; the Earthdata token from `EARTHDATA_TOKEN` or a credentials file, never in the script).

### Deprecated

- `SISController`: former name of `SISSession`; still works and emits a DeprecationWarning when instantiated.
- `ProjectDownloadConfig`, `ProjectDownloader`, `ProjectDownloadReport`: aliases of `DownloadConfig`,
  `DownloadSession`, `DownloadReport` with a DeprecationWarning.
- `use_gfz` is accepted and has no effect (gfzrnx is no longer run).
- The settings listed under "New warnings" (`interpolation_method`, `broadcast_tolerance`, `output_dir`,
  `clock_process`, `p_N`, `q_N`, the three kriging settings) will be removed after v1; the never-read
  `tolerance` argument of `BroadcastInterp` / `BrdcGenerator` as well.

### Removed

- PPP filter classes the session never selected: `PPPUducSFMultiGNSS`, `PPPUdMultiGNSS` (from `gnx_py` and
  `gnx_py.ppp`), `PPPSingleFreqMultiGNSS`, `_LegacyPPPDualFreqMultiGNSS`, and the unreachable router branches.
- `emission_interp` (`gnx_py`, `gnx_py.ppp`, `gnx_py.ionosphere`, `gnx_py.core.coordinates`),
  `SP3CustomInterpolatorOptimized` with `interpolate_for_tuples`, `CrdWrapper.get_emission_epochs`, `DT_VEL`,
  `process_windup_correction`, `DMIN_KM` / `DMAX_KM`, `SinglePointPositioning.observed` / `select_epoch`,
  `SPPResult.covariance_info` (never filled), `TECSession._interpolate_emission`.
- gfzrnx support (`GFZRNX2` and its helpers) and georinex as a runtime dependency.
- From the repository: `setup.py` (`pyproject.toml` alone describes the package), `requirements-dev.txt`, the
  `constraints-*.txt` pin files, the second logo `assets/GNX-py-logo.png`, the former download example
  `examples/run_download_gnss_data_example.py` (replaced by `examples/run_project_download_example.py`) and the
  tutorial plotting helpers `tutorial/introduction/utils.py`, `tutorial/ionosphere/helpers.py`,
  `tutorial/ionosphere/vtec_viz.py`, `tutorial/orbits/plotting_tools.py` and `tutorial/ppp/plotting_tools.py` (the
  notebooks plot with `gnx_py.viz`).

### Fixed

Relative to 0.1.0, by area (measured effects in parentheses):

- PPP filter: rejected observations are left out of the Kalman update (the 1e12 m² marking left them about
  0.9 % weight and condition numbers up to 3e15); `q_amb` acts on every ambiguity state of every route; an
  epoch skipped for a non-finite observation no longer breaks the next one.
- PPP-AR: the arc age starts at 0 on the first epoch of every arc; half-cycle ambiguities (LLI bit 1) are excluded
  from fixing; the validation uses the a-posteriori variance factor of the float phase residuals.
- PPP ionospheric constraint: the GIM constraint was not in the code-bias datum of the bias product (static GPS
  GRG convergence to 10 cm: 41 -> 15.5 min); the Klobuchar / NTCM constraint was too tight at night.
- SPP with a GIM and a bias product: the GIM correction was not in the product's DSB datum (BRUX 2024-035, GPS L1:
  3D RMS 1.444 -> 0.955 m).
- BeiDou: precise clocks referred to IF(B1C,B2a) with B1I/B3I codes (about 11 m off); the SIS clock datum per
  satellite generation; the band-7 code chosen by a PRN rule that is wrong since 2023 for reassigned PRNs
  (C01-C04, C06-C08, C11-C14).
- Warnings: two TEC sessions built on one configuration no longer silence each other's warnings (a warning is
  given once per run).
- Cycle slips: unconfirmed third-carrier flags, half-cycle arcs, and a NaN epoch when a satellite had no
  available slot in the geometry layer.
- SINEX: coordinates of the solution valid at the epoch; stations missing from the SINEX; netCDF export;
  SISConfig equality.
- Downloads: EUREF archive, the data-source letter of long observation names, the default DCB list that covered
  only 2024 days 021-048.
- Plots: 33 visualisation defects: figures left open after an error, inputs validated only after the figure
  existed, rejected SPP observations never drawn.
- SIS comparison hung on a non-positive interval; downloads never send `~/.netrc` credentials and drop the token
  on a cross-host redirect.
- Covariance carry-over of the uncombined PPP filters when the satellite set changes; all-zero CODE BeiDou OSB
  columns are treated as missing.
- Galileo RINEX 3 navigation records parsed by fixed field positions; Galileo health bits; BeiDou CGCS2000 Earth
  rotation rate and GM.
- SISRE formula and literature weights; SP3 nodes without a clock keep their position; broadcast ephemeris
  selection by the nearest ephemeris time.
- SPP weighting and convergence, the geometry gate for every system; observations with a missing correction never
  enter the adjustment; SPP computed the antenna height correction but did not apply it (see Output changes).
- Settings that silently changed the processing (unknown model names gave no correction).
- Satellite group delays (TGD, BGD, ISC, DCB/DSB/OSB) per signal and clock source in one place
  (`biases.code_group_delay_correction`); Bias-SINEX records by validity period, any file name, gzip; a missing
  bias is a missing value, not zero.
- PPP observation weights, process-noise units, the phase Jacobian of the G+E constrained filter, kinematic mode in
  the uncombined models, single-frequency PPP, and the new cycle-slip detection.
- STEC levelling, GIM mapping function and time interpolation, models, kriging, monitoring indices and plots.
- Antennas (PCO, PCV, ANTEX reader), Niell troposphere, phase wind-up, solid Earth, ocean and pole tides,
  satellite attitude (with the own readers and BDGIM listed under Added).
- Downloads: configuration validation, file integrity, archive layouts, consistent product sets.

### Known limitations

Configurations of the v1.0.0 validation (settings, products, stations BRUX, NKLG, NYA2 and days 2024-035,
2024-131, 2024-132) that stay outside their accuracy limit with an established cause that is not a GNX-py defect.
The limits were not loosened; each case is listed with its cause. A configuration of the validation is called a
row; the labels follow `gnx_py/ppp/README.md` ("Labels Used in the Measured Values").

| Id | Rows | Cause |
|---|---|---|
| LIM-ORB-01 | 4 SIS rows, COD, BeiDou IGSO SISRE p95 | the reference product: the C06 clock of COD MGX 2024-035 is 4.02 m off GRG's |
| LIM-PPP-01 | PPP IF GPS L2L5 | variable L5 inter-frequency clock bias of GPS Block IIF, amplified in IF(L2,L5); model after v1 |
| LIM-PPP-02 | daily PPP IF GPS with broadcast orbits, BRUX and NKLG 2024-132 (storm) | broadcast ephemeris error on that day |
| LIM-PPP-03 | Galileo-only kinematic PPP with GIM constraint, GRG, BRUX 2024-035 | weak Galileo-only kinematic solution driven by one satellite's constraint (E19, GRG DSB 1.2-1.5 ns off other sources) |
| LIM-PPP-04 | two PPP rows with the legacy code weights (`uncombined_iono_legacy_weighting=True`), BeiDou | the final error comes from the code with the legacy weights |
| LIM-PPP-05 | BeiDou kinematic PPP with GIM constraint, GRG (p95 relation to the unconstrained twin) | per-arc GIM error treated as white noise in the constraint |
| LIM-PPP-06 | single-frequency B1C PPP with BDGIM, NKLG 2024-035 | one 92-min low-elevation arc (C32, 10.0-10.6 deg) |
| LIM-SPP-01 | 5 BeiDou-2-only rows | 4-5 BDS-2 satellites; no valid RTKLIB reference |
| LIM-SPP-02 | BeiDou-2 B2I with Klobuchar, raw availability | BeiDou geometry-jump gate (30 m) with low redundancy |
| LIM-SPP-04 | 8 GPS rows with broadcast orbits (p95) | elevation weights 0.5/sin²E and a 3.1 m G08 orbit/clock error at 70 deg |
| LIM-SPP-05 | GPS + Galileo L1/E1 with the CODE GIM, 1 h | GNX maps the CODE GIM with its MSLM mapping function, RTKLIB with SLM |
| LIM-SPP-06 | relativistic path term on / off comparison | the hard 2-MAD screening switches 16 of 2880 epochs |
| LIM-SPP-07 | 2 rows without troposphere and mixed SF/DF modes | no like-for-like RTKLIB reference |
| LIM-SPP-08 | 11 NTCM rows | NTCM-G model error against the CODE GIM (up to about 21 TECU in the 2024-132 storm) |
| LIM-SPP-09 | 5 BDGIM rows | BDGIM model error against the CODE GIM (up to about 20 TECU) |

Other limitations of this release:

- GLONASS and QZSS are supported only in SP3-vs-SP3 orbit comparisons (no antenna offsets or group delays);
  BeiDou GEO satellites are skipped with broadcast orbits and in SIS/SISRE (with precise orbits they are
  processed).
- Modern BeiDou signals (B1C, B2a, B2b) need RINEX 4 navigation and clock-consistent OSB/DSB products; the B2b
  observable for B1CB2b (C7Z or C7D) is not settled.
- GPS Block IIF L5 inter-frequency clock bias is not modelled (warning only).
- BDS-3 satellites whose codes cannot be moved to the precise clock datum are down-weighted, not excluded or
  estimated.
- The single-frequency BeiDou GIM correction has no reference DSB when the GIM carries no BeiDou DCBs (CODE and
  IGS GIMs).
- Downloads: sources BKG, CDDIS, CODE and EUREF (no IGN or WHU mirrors, no ATX/SINEX/BLQ files); long observation
  names are generated for `01D_30S_MO` only, so stations that publish only 15 s or GPS-only files are not found by
  the long name; the CODE archive layout was not confirmed live (the server did not answer during the validation).
- The SP3 interpolation of SIS runs in one process (no parallel workers).
