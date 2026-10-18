# GNX-py Ionosphere Module

`gnx_py.ionosphere` provides the ionospheric side of GNX-py: STEC
measurement, GIM/IONEX reading, empirical ionosphere models, activity
monitoring indices, and experimental VTEC kriging.

The module is designed for research and validation workflows where the user
needs access to both the processed TEC products and the intermediate diagnostic
columns that explain how each satellite arc was corrected.

## What It Supports

- Slant TEC (STEC) measurement from dual-frequency geometry-free observables.
- TEC/VTEC utilities using thin-shell IPP geometry and mapping functions.
- GPS, Galileo and BeiDou processing through the shared GNSS signal metadata.
- GIM/IONEX TEC map reading, interpolation and DCB parsing.
- Bias handling from OSB, DCB/DSB, GIM DCB blocks, manual config values, or a
  zero fallback with warning.
- Broadcast ionosphere models: Klobuchar, NTCM-style helpers and BDGIM
  (BeiDou-3 global model, BDS-SIS-ICD-B2b-1.0 §7.7).
- Ionospheric activity indices: SIDX, GIX/GIX-family and ROTI.
- Experimental kriging of VTEC grids in geographic or solar-geomagnetic frames.

## Recommended Entry Points

The main user workflow is:

```python
from datetime import datetime

from gnx_py.ionosphere import TECConfig, TECSession

cfg = TECConfig(
    obs_path="sample_data/BRUX00BEL_R_20240350000_01D_30S_MO.crx.gz",
    nav_path="sample_data/BRDC00IGS_R_20240350000_01D_MN.rnx",
    orbit_type="broadcast",
    ionosphere_model=False,
    troposphere_model=False,
    sat_pco=False,
    rec_pco=False,
    windup=False,
    sys="C",
    bds_freq="B1IB3I",
    station_name="BRUX",
    time_limit=[datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 0, 10)],
)

stec = TECSession(cfg, use_sys="C").run()
```

`TECSession` currently processes one constellation at a time. Run separate
sessions for `use_sys="G"`, `"E"` and `"C"` when comparing systems.

## Core STEC Workflow

1. `TECConfig` selects the system, signal pair, products, correction switches
   and bias policy.
2. `TECSession` loads RINEX observations, optional NAV/SP3/ANTEX/bias products,
   interpolates satellite coordinates, applies preprocessing corrections and
   detects cycle slips with the layered detector shared with PPP (RINEX LLI,
   a third carrier from the RINEX file, and at 30 s the geometry layer, with
   precise orbits and, since v1.0.0, with broadcast orbits too: a
   broadcast record switch starts a new piece of the layer, NYA2 2024-035
   GPS L1/L2 broadcast run loses 0.39 % of the satellite-epochs instead of
   4.01 %; PPP keeps it off with broadcast orbits; see "Cycle Slips" in
   `gnx_py/ppp/README.md`). An undetected
   equal slip moves the levelled STEC of its arc by 0.45 TECU per cycle on
   L2L5 / E5aE5b and 0.51 TECU on L1L2. TEC runs one system, so the geometry
   layer estimates the receiver clock from that system alone. Without the
   geometry layer (sampling other than 30 s, broadcast frames without their
   records) TEC arcs no
   longer split at non-integer third-carrier steps since v1.0.0: such a flag breaks the arc only when the joint integer estimate of
   the cross pairs cannot exclude a slip on the processed signals (the Block IIF
   G27 L5 step of 4.4-4.7 mm at its 18:00 broadcast record switch, BRUX
   2024-035, stays in the arc; GPS L2L5 broadcast minus precise STEC p99 0.64
   -> 0.0000005 TECU). A change (set or clear) of the RINEX half-cycle flag
   (LLI bit 1) starts a new arc. `SYS / PHASE SHIFT`: the phases are used as
   written (RINEX 3.0x applies the correction in the file; before v1.0.0 the header values were added a second time to every satellite of the
   system, a constant per arc that the levelling absorbed); a change of the
   correction inside the data is reported with one warning and keeps the arc
   (the phases stay aligned on both sides of the record).
   Observations are read by `gnx_py.io.rinex_obs` (RINEX 2/3/4, every compression
   format, in memory; `time_limit` always applied; `use_gfz` has no effect
   since v1.0.0). Without a usable APPROX POSITION XYZ the station
   position comes from the code observations, with a warning. A file without
   the phases of the mode (a code-only file) raises a ValueError naming the
   missing phases and the file. With
   `screen=True` the broadcast health in force is applied per satellite and
   message type, and Galileo rows are screened on the DVS/SHS flags of the
   mode's own signals (E1 -> E1-B from I/NAV, E5a -> E5a from F/NAV, E5b ->
   E5b from I/NAV): E14 and E18 (out of service in every 2024 record) leave the
   screened rows; TEC E1E5a rows lose 1,555-2,347 epochs per day this way. No
   observations within `time_limit` raises a ValueError naming the window and
   the file's span (it was a FileNotFoundError). When no arc is left (every
   arc cut by the slip detection, or no observation reaching it) the run ends
   with a readable ValueError (inside `SessionExecutionError`) that tells the
   two cases apart (it was an empty frame or a KeyError).
3. `STECMonitor` groups observations by satellite arc, resolves bias terms,
   levels phase geometry-free observations to code geometry-free observations,
   and returns diagnostic STEC columns.

Important output columns include:

- `leveled_tec`, `median_leveled_tec`, `code_tec`, `phase_tec` (K x raw L4),
  `leveled_tec_sg`, `leveled_tec_cheby` (filters of `leveled_tec`)
- `stec_uncal`, `stec_sat`, `stec_rcv`, `stec_full` (= `leveled_tec`): the
  levelled STEC without biases, with the satellite bias, the receiver bias or
  both (el/m^2)
- `stec_sat_bias_m`, `stec_station_bias_m`, `stec_total_bias_m`,
  `stec_sat_bias_tecu`, `stec_rcv_bias_tecu` (one constant per arc)
- `stec_sat_bias_source`, `stec_station_bias_source`,
  `stec_sat_bias_coverage`, `stec_station_bias_coverage`, `stec_bias_datum`
- `stec_bias_code_1`, `stec_bias_code_2`
- `leveling_std_tecu`: per-arc spread of code minus levelled phase (TECU)

Units: every STEC value column above (`leveled_tec`,
`median_leveled_tec`, `code_tec`, `phase_tec`, the filtered `leveled_tec_*`
and the `stec_uncal` / `stec_sat` / `stec_rcv` / `stec_full` family) is in
electrons/m² (divide by 1e16 for TECU); columns ending in `_tecu` are TECU and
columns ending in `_m` metres. The conversion factor K = (1/40.3) f1² f2² /
(f1² - f2²) maps the geometry-free combination L4 [m] to electrons/m²
(`STECMonitor._get_factor`).

Levelling is over the whole continuous arc by default (`leveling_ws=None`); see
`STEC_BIAS_USAGE.md` for the bias sources, the datum rule and these columns.

Short arcs: `min_arc_len` is in minutes (default 10). The cycle-slip step
keeps an arc only when it has more than int(`min_arc_len` x 60 / interval)
samples, i.e. more than 20 at 30 s, so a 21-sample arc spanning exactly
10 min is kept (`tools.CSDetector`, the same rule in PPP). The default stays:
the carrier-to-code levelling error, measured on 10-minute blocks of the arcs
of the validation data, has a median of 0.20 TECU and a p95 of 0.67 TECU; an
earlier measurement gives 1.03 TECU for arcs shorter than 30 min (0.4 % of the
samples) without naming the statistic. A longer `min_arc_len`
trades those samples for a smaller levelling error; `leveling_std_tecu` is the
per-arc diagnostic to screen on.

Satellite antenna: STEC does not depend on the satellite antenna reference
point. TEC keeps the per-signal PCO and PCV; the BeiDou B3I broadcast reference
of v1.0.0 changes STEC by at most 1.6e-8 TECU.

## Systems And Signals

GPS and Galileo are the historical paths. BeiDou is supported through the shared
signal registry. `B1IB3I` is the pair of the precise BeiDou clocks of the 2024
products and the most widely tested one. The modern BeiDou-3 signals (`B1C`,
`B2a`, `B2b` and the pairs `B1CB2a`, `B1CB2b`, `B1CB3I`, `B2aB2b`) are
supported when the inputs give their biases: an OSB product with the codes of
the pair (e.g. C1P, C5P, C7D), a DSB product reaching them through a chain, or
the B-CNAV group delays of a RINEX 4 navigation file (the 73 TEC
validation rows of these four pairs, broadcast and precise orbits, pass on the
2024 validation data). Without such inputs the bias policy warns or refuses as configured
(`stec_missing_bias`).

## GIM And IONEX

Use `gnx_py.ionosphere.ionex.GIMReader` to read IONEX/INX Global Ionosphere
Maps and optional DCB blocks. `TECInterpolator` evaluates map values at IPPs
using a thin-shell geometry and elevation mapping. GIM products are also used as
background data for station DCB calibration and model comparison.

Two settings of the base `Config` (so SPP, PPP and TEC alike) control how a GIM
becomes a slant delay:

- `gim_mapping_function` (`"auto"`, `"slm"`, `"mslm"`): `"auto"` follows the
  IONEX header. A DESCRIPTION stating a modified single-layer model (CODE) gives
  the MSLM (H = 506.7 km, alpha = 0.9782, `ionex.mapping_mslm`); `COSZ` (IGS,
  CAS) gives the single layer (`ionex.mapping_slm`); anything else gives the
  single layer with one warning. The IPP stays on the IONEX `HGT1` shell.
- `gim_time_interpolation` (`"rotated"`, `"earth_fixed"`): rotated maps (IONEX
  1.0 eq. 3) by default. Rotation is applied only to grids that are global in
  longitude; a regional grid falls back to earth-fixed interpolation with a
  warning.

Both defaults were chosen in the slant domain.
Against carrier-to-code levelled STEC, mean of 9 station-days (BRUX/NKLG/NYA2
x 2024-035/131/132):

| Product | metric | former defaults | v1.0.0 defaults |
|---|---|---|---|
| COD G (MSLM by header) | mean residual 10-15 deg [TECU] | -3.30 | 0.00 |
| COD G (MSLM by header) | k_ep | 0.879 | 0.973 |
| COD G (MSLM by header) | shape [TECU] | 4.27 | 4.01 |
| IGS G (SLM, rotation only) | shape [TECU] | 4.726 | 4.631 |

They cost in the position domain (measured against the former GIM handling
with the former constrained PPP weighting; see below):

- SPP with a GIM (157 rows): median rms_3d +0.175 m, 117 rows worse and 37
  better. Mostly BRUX (2024-035 median 1.33 -> 1.71 m, +0.39 m); NKLG and
  NYA2 2024-035 get better (3.25 -> 2.68 m, 2.90 -> 2.72 m). Almost all of it
  comes from the COD MSLM. Opt-out: `gim_mapping_function="slm"`.
- PPP with a GIM constraint at BRUX 2024-035 (row labels, `UCi`, `R4c`, `t_c10` / `t_c20`: see "Labels Used
  in the Measured Values" in `gnx_py/ppp/README.md`): rotation slows convergence
  (GC static UCi t_c10 50.5 -> 80 min, C static UCi t_c10 51 -> 91.5 min,
  G kinematic UCi t_c20 81 -> 103 min, C kinematic UCi RMS 0.148 -> 0.157 m,
  R4c kinematic RMS 12.03 -> 12.70 cm with rotation alone, 12.56 cm with both
  defaults). The CODE MSLM raises NKLG 2024-035 G UCi t_c10 48.5 -> 113.5 min;
  both defaults together raise NKLG R4c GE 24 h t_c10 48 -> 93 min (49.5 / 49
  min with either one alone). On the Galileo-only E UCi kinematic row at BRUX
  (failing either way) the MSLM brings t_c20 from 321.5 to 13 min,
  but its statistics then start at 13 min: RMS 0.147 -> 0.294 m. Final
  positions change by at most 9 mm. Opt-out:
  `gim_time_interpolation="earth_fixed"`.

This is not a GIM-handling artefact. In SPP (63 full-day GIM rows) the single layer is better than the
MSLM by a median 0.069 m of rms_3d (36 rows better, 27 worse): the MSLM raises
the single-frequency height by 0.4-1.4 m at every station (mean Up MSLM / SLM:
BRUX 2024-035 +0.629 / +0.028 m, NKLG 2024-035 +0.171 / -1.242 m, NYA2 2024-035
-0.907 / -1.376 m), which helps or hurts with the station's other height
errors; the rotation costs 0.010 m and adding the GIM RMS to the SPP weights
0.001 m. With a GIM constraint (PPP) the four combinations move the static
UCi rows by at most 6.5 min (rotation equal or faster on the four COD rows);
the slow convergence above was the legacy constrained code weighting
(fixed in v1.0.0: GC static UCi t_c10 65.5 -> 20 min). The defaults stay.

IONEX map epochs are UTC. Sessions pass GPS-time epochs through
`tools.gim_slant_tec`, which converts GPST to UTC (an epoch up to 19 s before
the first map takes that map). A direct `TECInterpolator.vtec` call takes UTC
times and defaults to earth-fixed interpolation (`rotate=False`).

## Bias Policy

STEC uses the geometry-free code convention:

```text
P4 = P2 - P1
```

Bias sources are controlled by `TECConfig.stec_bias_sources`:

- `product`: OSB columns or DCB/DSB pair columns attached during product loading.
- `gim`: satellite/station DCB entries parsed from GIM/IONEX auxiliary data.
- `config`: manual nanosecond values from `define_satellite_dcb` or
  `define_station_dcb`.
- `zero`: zero correction fallback.

OSB values are differenced as `OSB(P1) - OSB(P2)`. DCB/DSB values are treated as
the differential bias for the selected code pair. Missing bias behavior is
controlled by `stec_missing_bias`: `warn_zero`, `zero` or `raise`.

More examples are in [STEC_BIAS_USAGE.md](STEC_BIAS_USAGE.md).

## Ionosphere Models

- `gim`: external GIM/IONEX products, typically the most stable background for
  calibration and comparison.
- `klobuchar`: GPS broadcast Klobuchar delay model. For BeiDou the setting
  `bds_klobuchar` of the base `Config` chooses the coefficients (background
  model and `compare_models` alike):
  - `"gps"` (default): the GPS coefficients (GPSA/GPSB) scaled to the BeiDou
    signal, as before v1.0.0;
  - `"icd"` (opt-in): the BDS model of BDS-SIS-ICD-B1I 3.0 §5.2.4.7 (375 km
    shell, BDSA/BDSB header coefficients, `models.klobuchar_bds`). Without
    BDSA/BDSB in the navigation header it falls back to `"gps"` with a warning.

  Why the GPS coefficients are the default: the BDS-2 broadcast coefficients
  are fitted for the Asia-Pacific service area. Slant RMS against the COD GIM
  at B1I (the GIM evaluated with the v1.0.0 defaults, MSLM from the COD header and rotated
  maps, `tools.gim_slant_tec`; synthetic geometry, elevations 15-75 deg,
  every 5 min):

  | Station | 2024-035 | 2024-131 | 2024-132 |
  |---|---|---|---|
  | BRUX | 7.41 / 9.60 | 16.44 / 8.82 | 28.70 / 28.71 |
  | NKLG | 45.99 / 30.04 | 17.51 / 25.38 | 13.80 / 29.27 |
  | NYA2 | 14.87 / 10.63 | 11.14 / 13.15 | 5.73 / 20.59 |

  (GPS-scaled / BDS, TECU.) Mean 17.95 / 19.58 TECU; BDS is worse on 5 of 9
  station-days, ties on one, better on 3. In SPP 29 BeiDou Klobuchar rows got
  worse and 22 better with the BDS model; NKLG 2024-132 C-only rms_3d
  3.08 -> 6.75 m.

  Since v1.0.0 the coefficients come from `gnx_py.io.rinex_nav`: every
  IONOSPHERIC CORR header record (with its time mark and SV id), RINEX 2
  ION ALPHA/ION BETA, and RINEX 4 '> ION' records. Each set is screened per
  model (Klobuchar amplitude over the geomagnetic latitude band, BeiDou A2 > 0,
  all-equal raw counts such as the C48 filler record rejected; NeQuick ai0 > 0),
  one warning per rejected record. `"icd"` takes one GEO-first BeiDou set (GEO
  C01-C05/C59-C62, then IGSO, then MEO); with a RINEX 4 file the latest
  plausible record at or before each epoch (`rinex_nav.broadcast_iono_at`).
  Against the COD GIM: GEO-first 19.45 TECU, latest record 22.15 TECU,
  GPS-scaled 16.53 TECU; all three validation stations are outside the BDS-2 service
  area. The '> ION Cxx CNVX' records of RINEX 4 belong to the separate global
  model `bdgim` below, which has its own set rule (any BeiDou class).
- `ntcm`: NTCM-G, mapped with its own MSLM (alpha = 0.9782 at 450 km).
  `compute_ntcm_grid` evaluates the model at each grid node. Verified against
  Annex D of the NTCM-G description of the European GNSS (Galileo) Open
  Service (38 rows, 4.9e-5 TECU) and an independent implementation. Model
  accuracy on the validation data (slant TEC against the
  COD GIM, NTCM / GPS Klobuchar on the same lines of sight, TECU mean / RMS):
  BRUX 2024-035 10-11 h -17.9 / 18.4 against -4.7 / 6.2, NKLG 2024-132 +20.9 /
  28.7 against -2.6 / 14.4, NYA2 2024-132 +17.8 / 18.8 against +0.6 / 6.3: on
  these station-days NTCM is the weakest broadcast model (known limitation
  LIM-SPP-08 in `CHANGELOG.md`). The pierce-point longitude is the description's (folded beyond
  the pole, as Annex D row 6; at NYA2 1.3 % of a day's GPS L1 rows, RMS against
  the GIM the same either way).

  GPS Klobuchar and NTCM take the broadcast set in force at each epoch: the
  latest RINEX 4 '> ION' record transmitted before it (since v1.0.0; before, one set per run, the header or the first plausible record).
  A RINEX 3 / 2 file has one header set and gives the same numbers as before.
  BRD400DLR changes the Galileo set at 14:11 on 2024-035 and 17:21 on 2024-132,
  the GPS set at 22:57 on 2024-035. The RINEX 3 header GPS set of 2024-035
  (BRDC00IGS) is none of the sets BRD400DLR broadcast that day, which explains
  about 13 TECU between the two Klobuchar runs at BRUX 10-11 h (a property of
  the file).
- `bdgim`: BDGIM, the BeiDou-3 global broadcast ionosphere model, see the next
  subsection.

`compare_models` on `TECConfig` adds model columns (`klobuchar`, `ntcm`, `bdgim`,
`gim`) in slant metres at the first signal of the mode, plus a `<model>_tecu`
column in TECU. The `gim` column uses the same mapping and time rules as
`ionosphere_model='gim'`; the `bdgim` column is the same clamped value as the
background of `ionosphere_model='bdgim'`.

### BDGIM (BeiDou-3 global broadcast model)

`ionosphere_model='bdgim'` (SPP, PPP and TEC; GPS, Galileo and BeiDou alike,
the model is global) and `compare_models` 'bdgim' since v1.0.0. The
model is `gnx_py.ionosphere.bdgim`, written from the published algorithm of
BDS-SIS-ICD-B2b-1.0 §7.7 (CSNO, July 2020; Eq. 7-5 to 7-16, Tables 7-8 to
7-10). The ICD is cited, not distributed with GNX. The module is imported by
its path (it is not re-exported by `gnx_py.ionosphere`).

- Inputs: a RINEX 4 navigation file as `nav_path` with '> ION Cxx CNVX'
  records (B-CNAV; nine broadcast coefficients alpha1..alpha9 in TECU), e.g.
  the IGS/DLR merged file BRD400DLR, which also carries the G/E/C ephemerides
  and can serve broadcast orbits too. The 17 predicted (non-broadcast)
  coefficients come from Table 7-10 and are built in. RINEX 2 and 3 files
  have no BDGIM records.
- Set selection: per epoch the latest plausible record with a transmission
  time (GPST) at or before the epoch, from any BeiDou satellite, without the
  GEO/IGSO/MEO priority of the BeiDou Klobuchar; epochs before the first set
  take the earliest one. The model is evaluated in BDT = GPST - 14 s.
- Screens (`gnx_py.io.rinex_nav`): every record is checked against Table 7-8:
  each coefficient on the 1/8 TECU grid and inside its range (alpha5 has a
  negative scale factor, so its range is -31.875..0 TECU), and not a filler
  (all nine raw words equal modulo 256). A rejected record gives one warning
  that names the first failing coefficient. On the validation days only the C48 record of
  2024-035 04:24 is rejected (raw words 0x2AA, 0xAA x 8; with it in force the
  035 grid RMS was 17.05 instead of 12.44 TECU); on 38 distinct BRD400DLR
  files the filler is the only one of 438 records rejected (all 437 real
  sets pass). An all-zero set is a filler too (reported as raw word 0x00).
- Warnings, once per run: the set in force is more than 6 h old (it is still
  used; the age counts from the latest transmission of the same coefficients
  by any satellite, so a re-broadcast of an unchanged set resets it); epochs
  precede the first set of the file by more than 6 h (is `nav_path` from the
  observation day?); rejected BDGIM records inside the run
  window when the navigation reader did not already warn for BeiDou.
- Errors: `validate_for_run` (when the run starts, before any observation is
  read) raises a ValueError naming `nav_path` and its RINEX version when the
  file has no usable BDGIM set: no records, or all of them rejected (how many
  and why). The message suggests a RINEX 4 file such as BRD400DLR, or
  `ionosphere_model='klobuchar'` / `'gim'`. There is no silent fallback to
  another model.
- Conventions where the ICD leaves a choice: atan2 for the arctan of Eq. 7-7,
  7-8 and 7-9 (the literal one-argument arctan puts pierce points beyond the
  pole in the wrong quadrant: grid RMS 22.1 instead of 12.4 TECU, up to
  6.7 TECU slant error at NYA2); the period T3 = 0.33 day as printed; the
  prediction epoch t_p is the odd hour of the same BDT day nearest to the
  epoch, an exact even hour takes the later one (integer-nanosecond
  arithmetic, no rounding flip).
- Units: the model returns TECU and is not clamped (negative VTEC occurs, down
  to about -4 TECU on the storm days of the validation data). The sessions convert TECU to metres
  at the first signal of the mode with GNX's 40.3e16, as for GIM, Klobuchar and
  NTCM (the ICD's 40.28e16 is used only by `bdgim_delay_icd`; the difference
  is 0.05 %, at most 1 cm at 20 m), and clamp 'ion' at 0.
- PPP constraint sigma ('ion_rms'): 40.3e16 / f^2 x hypot(0.5 |STEC|,
  5 TECU x MF) from the unclamped model; 5 TECU at the zenith is 0.81 m
  (2.7 ns) on L1. Since v1.0.0 Klobuchar and NTCM take the same rule with their own mapping
  function (they kept 50 % of the delay before; see `gnx_py/ppp/README.md`).
- SPP rows (known limitation LIM-SPP-09 in `CHANGELOG.md`): BDGIM slant
  error against the COD GIM +13.6 TECU mean at NYA2 on 2024-132 (BRD400DLR
  Klobuchar -2.0), +20 TECU at NKLG at 20-24 h of 2024-132 (post-sunset
  equatorial storm ionosphere; Klobuchar +6.2), -15.7 TECU at BRUX 2024-035
  10-11 h (Klobuchar +8.5).
- Run report: after `run()`, `session.run_notes.iono_report[sys]` (SPP, PPP
  and TEC) holds the sets used, the maximum age and lead, the rows clamped at
  0 and the rejected records inside the window.
- Maps: `compute_bdgim_grid(lat, lon, times, alpha, time_scale=...)`; the time
  scale is required ("BDT", "GPST" or "UTC"; IONEX epochs are UTC), because an
  18 s error moves t_p at even hours (up to 2.5 TECU).

Validation data (2024-035 quiet, 131 storm onset, 132 storm; TECU):

| Comparison | BDGIM | GPS Klobuchar |
|---|---|---|
| global grid vs COD GIM, 035: RMS / bias | 12.44 / -7.82 (72.7 % corrected) | |
| global grid vs COD GIM, 131 / 132: RMS | 18.02 / 18.23 (67.1 / 57.2 %) | |
| slant TEC, 9 station-days (BRUX, NKLG, NYA2) vs COD: RMS / bias | 14.45 / -0.26 | 22.23 / -1.74 |
| same vs IGS: RMS | 15.15 | 22.88 |
| TEC session, BRUX 035, VTEC at the 400 km IPP vs IGS: bias / RMS | -7.5..-7.9 / 8.0..8.7 | |

BDGIM's global mean equals the broadcast alpha1, which is 7-8 TECU below the
COD GIM on 035: the negative bias is a property of the broadcast
coefficients. BDGIM is worse than the GPS Klobuchar on 3 of the 9
station-days (2024-131 NKLG, 132 NKLG and 132 NYA2). The literature reports
about 77.6 % of the ionospheric delay corrected globally (Yuan, Wang, Li and
Huo 2019, NAVIGATION 66(1):55-69, doi:10.1002/navi.292).

## Monitoring

`monitoring.py` provides:

- `compute_sid`: SIDX from STEC temporal gradients in TECU/min.
- `compute_gix`: GIX and GIX-family spatial gradient diagnostics.
- `compute_roti_links`: ROT/ROTI per station-satellite link.
- Plotting helpers for ROTI and GIX.

`load_stec_folder` loads processed STEC parquet outputs for network monitoring,
including `*_C_STEC.parquet.gzip` BeiDou products generated with `sv/time`
stored either as columns or as parquet index levels.

Time scale: monitoring inputs are taken as naive GPS time (GPST). A tz-aware
input is converted to UTC and then to GPST (+18 s in 2024) with one warning;
outputs are naive GPST with `attrs["time_scale"] = "GPST"`. The SIDX day split
is a GPST midnight. The viz helpers (`plot_roti_series`, `plot_sidx_series`,
`plot_gix_family`) take `time_scale=None` and label the axis from the frame's
`attrs`, else "UTC" for tz-aware and "GPST" for naive epochs.

Mapping function of SIDX/GIX: `compute_sid` and `compute_gix` compute the single
layer at `iono_h_m` by default (`mapping=None`) and accept a supplied `M` column
when it equals the single layer or the MSLM at `iono_h_m` (or the CODE MSLM at
450 km); any other `M` column warns and is recomputed. An explicit `mapping`
enforces that function, and `use_M_column=True` keeps the column as given.

`compute_roti_links` differences only inside contiguous segments of a link
(`sv` must identify an arc), normalises longitudes to [-180, 180) and rejects
NaN link keys. With `detrend_5min=True` (default) the TEC of each segment is
reduced by its trailing 5-min mean (window [t - 5 min, t], past samples only)
before the ROT is formed; the detrend window does not follow `window_min`
(kept trailing on purpose, so a sample never uses later data; a centred
window would use later samples).

GIX is the projected mean of the pair gradients (about |grad|/2 for pairs
spread in two dimensions); the columns `GIX_LS_mtecu_km`,
`GIX_LSx_mtecu_km` and `GIX_LSy_mtecu_km` give the least-squares gradient,
exact for any pair geometry (NaN when the pair directions are nearly
collinear). `load_stec_folder(tec_units=...)` converts every TEC column.

The monitoring and kriging functions operate on calibrated STEC/VTEC point
columns and station-satellite link identifiers, so GPS/Galileo/BeiDou data are
structurally supported after product loading. Mixed systems are allowed in
one `compute_gix` or `krige_epoch` call. Pairing rules of
`compute_gix`: every two IPPs of an epoch form a candidate pair, whatever their
station, satellite or system (same-station pairs of two satellites and
same-satellite pairs of two stations included); a pair is kept when its dipole
length lies in the preset (or `dmin_km`..`dmax_km`), `|Mi - Mj| <=
max_delta_M` and its centre lies in `region`, then thinned to
`max_pairs_per_epoch`. To restrict the pairs to one system, filter the frame by
`sv` before the call. Mixed `G/E/C` activity-index and kriging runs should
still be treated as requiring scientific validation: bias policy, signal pair,
receiver network and variogram assumptions determine whether pooled results
are physically comparable.

## Kriging

`IonoKrigingMonitor` supports experimental VTEC gridding with WAAS/Sparks,
ordinary kriging and universal kriging. It can work in geographic coordinates or
with `SolarGeomagneticTransformer` for solar-geomagnetic coordinates. The fields
`n_closest_points`, `backend` and `use_external_drift` are not used (ordinary
and universal kriging solve one system over every IPP of the call, universal
kriging with a regional linear drift): a value other than the default gives one
UserWarning at construction; they are removed after v1.

Ordinary and universal kriging measure great-circle distances in degrees for
both coordinate types, so the variogram range is in degrees. The WAAS mode
follows the MOPS neighbourhood rule: all IPPs within `Rmin_km` (default
800 km), grown to the `Ntarget` nearest up to `Rmax_km`, NaN below `Nmin`
(`Nmin >= 3`; with the default `Rmin_km`, an `Rmax_km` below 800 km needs an
explicit `Rmin_km`). The measurement-noise covariance is `diag(meas_var)` plus
the receiver and satellite bias blocks. Options are validated when the monitor
is built and again at every epoch. `save_grids_to_netcdf` writes GPS-time
epochs.

Solar-geomagnetic frame (`coord_frame="SM"`): the transformation
time is the `sm_obstime` given to the constructor, and the SM grid is built
for it once; the `epoch_time` argument of `krige_epoch` does not update it.
Build one `IonoKrigingMonitor` per epoch for SM runs over several epochs
(N-P51). In SM mode the `lat_ipp` / `lon_ipp` columns passed to `krige_epoch`
must hold SM coordinates (from `compute_roti_links(coord_mode="SM")` they are
the `lat_sm` / `lon_sm` columns, renamed). Each IPP may carry `rec_id` and
`sat_id` for the receiver and satellite bias blocks, so systems can be mixed
in one call.

This is a research-facing API. Validate variogram parameters, neighborhood
selection and coordinate-frame assumptions before operational use.

## Known Limitations

- The package-level public API is intentionally broad for backward
  compatibility and currently exports some accidental symbols such as `np`,
  `pd` and imported helper classes.
- `Calibration` is still used for receiver DCB estimation but is model-dependent
  and should be changed only with numerical regression tests.
- GIM/IONEX DCB parsing supports the GPS, Galileo and BeiDou records of the
  CODE and IGS products used in the validation, but product format variations
  may require additional parsers.
- BDS support depends heavily on product bias coverage for the selected code
  pair.
- Kriging, `compare_models` and the NTCM and BDGIM grid generation are
  experimental.
- Solar-geomagnetic coordinates use a centred IGRF-13 dipole whose axis is
  frozen at 2025.0, the end of the IGRF-13 secular-variation interval
  (`smg.IGRFDipole.clamp_to_2025=True`); IGRF-14 is not in v1, so for 2026
  data the axis is the 2025.0 one.

## Developer Notes

- Active core: `TECConfig`, `TECSession`, `STECMonitor`,
  `apply_stec_bias_policy`, `GIMReader`, `TECInterpolator`, `get_ipp`,
  `stec_mf`, `klobuchar`, `ntcm_vtec`, `bdgim.bdgim_slant`.
- Legacy/compatibility: `Calibration` and parts of older `gnx_py.core.tools`
  preprocessing paths (the former `STECSession.py` driver script is not
  part of the repository).
- Requires caution: `STECMonitor._process_arc`, `GIMReader._parse_dcb`, BDS bias
  handling. PPP's `measure STEC` bridge (`ionosphere_model="measure STEC"`)
  is disabled: `PPPConfig` rejects the value (it always failed).
- Experimental: kriging classes, `compare_models`, `compute_ntcm_grid`,
  `bdgim.compute_bdgim_grid`.

## Validation

The module was validated for v1.0.0 over a STEC configuration matrix (systems,
signal pairs, orbits, bias policies, GIM and model backgrounds, station classes
and days) against independent references, together with the network products
(SIDX, GIX, ROTI, kriging, variograms and the SM frame), the Klobuchar, NTCM,
BDGIM and GIM handling, the IONEX and DCB parsing and the readable errors.

Coverage: the station classes MID (BRUX), EQ (NKLG) and HIGH (NYA2) on the
quiet day 2024-035, the storm onset 2024-131 and the storm 2024-132. The
network products (GIX, kriging) were graded on the quiet day only, with a
38-station EPN network of 2024-035 computed by the v1.0.0 library (receiver DCB
calibrated against the COD GIM); on the storm day only the per-station ROTI of
the three stations was graded. The STEC files shipped with the tutorials
(`tutorial/ionosphere/network`, 75 files: the same network; LAMA has no
Galileo file) and under `sample_data/output` (BOR1 and BRUX, receiver DCB from
the COD GIM header) were computed by the v1.0.0 library; executed tutorial
notebooks are illustrations, not validation references.

The validation covers library behaviour, but it does not certify every possible
product/system/signal combination.
