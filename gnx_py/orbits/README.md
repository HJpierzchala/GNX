# GNX Orbits and Signal-In-Space Tools

`gnx_py.orbits` contains the GNX pipeline for comparing GNSS broadcast and
precise orbit/clock products. It is focused on signal-in-space (SIS/SISRE)
validation: reading broadcast navigation and SP3 products, aligning them on a
common epoch grid, applying optional satellite antenna corrections, merging
clock/bias terms, and reporting orbit and clock differences in physically
interpretable components.

The module is compact, but it sits on top of several important GNX building
blocks: RINEX navigation parsing, SP3 interpolation, antenna phase-center
handling, BIA/OSB/DSB products, GNSS signal metadata and coordinate geometry.

## What It Supports

- Broadcast-vs-precise orbit and clock comparison.
- SP3-vs-SP3 comparison through the same common-epoch machinery.
- SISRE-style metrics with radial, along-track and cross-track decomposition.
- GPS, Galileo and BeiDou processing branches: broadcast navigation input,
  satellite PCO (`apply_satellite_pco`) and `compare_dcb` support these three
  systems only. GLONASS (`system="R"`) and QZSS (`"J"`) are accepted for SP3 x
  SP3 comparisons with `apply_satellite_pco=False` and `compare_dcb=False`
  (eclipse flagging works); any other combination raises a ValueError naming
  the setting to change.
- BeiDou legacy `B1IB3I` MEO/IGSO validation.
- Guarded BeiDou modern signal experiments (`B1C`, `B2a`, `B2b`) when matching
  RINEX 4 navigation fields and clock-consistent bias products are available.
- Broadcast navigation in RINEX 2 (GPS), RINEX 3 and RINEX 4 (since v1.0.0: GPS LNAV with the CNAV group delays, Galileo I/NAV and F/NAV,
  BeiDou D1/D2 and B-CNAV), plain or compressed (gzip, `.Z`, bzip2, zip).
- Optional DCB/OSB/DSB product ingestion.
- Optional satellite PCO correction using ANTEX.
- Optional eclipse flagging and post-eclipse mask extension.
- Compatibility exports for older `from gnx_py.orbits import *` users.

## Main API

The recommended programmatic entry points are:

- `SISConfig`: configuration object for one SIS comparison run.
- `SISSession`: high-level session that executes the configured pipeline.
- `SISController`: deprecated alias of `SISSession` (see "Deprecated Names").

Example:

```python
from datetime import datetime

from gnx_py.orbits import SISConfig, SISSession

config = SISConfig(
    orb_path_0="sample_data/BRDC00IGS_R_20240350000_01D_MN.rnx",
    orb_path_1="sample_data/COD0MGXFIN_20240350000_01D_05M_ORB.SP3",
    interval=300,
    system="C",
    bds_mode="B1IB3I",
    atx_path="sample_data/igs20.atx",
    tlim=[datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 23, 55)],
    compare_dcb=False,
    clock_bias=True,
    clock_bias_function="mean",
    apply_satellite_pco=True,
    apply_eclipse=True,
)

result = SISSession(config).run()
print(result[["dR", "dA", "dC", "dt", "dTGD", "sisre"]].head())
print(result.attrs.get("clock_convention_status"))
```

## Deprecated Names

`SISController` was renamed to `SISSession`, consistent with
`PPPConfig`/`PPPSession` and `TECConfig`/`TECSession`; `SISConfig` keeps its
name.

| Deprecated name | Use instead |
| --- | --- |
| `SISController` | `SISSession` |

The old name still works: it is importable from `gnx_py.orbits`,
`gnx_py.orbits.config` and the top-level `gnx_py` package (also through
wildcard imports), takes the same arguments and runs the same pipeline.
Importing it is silent; each instantiation emits one `DeprecationWarning`
naming `SISSession`. Old `isinstance`/`issubclass` checks keep working in both
directions: an object built with `SISController(config)` is a `SISSession`,
and `isinstance(SISSession(config), SISController)` is also True. A
`SessionExecutionError` raised by `run()` names the session `"SISSession"`
for both spellings. `SISController` will be removed only in a future major
release; replace `SISController(config).run()` with
`SISSession(config).run()`.

## Processing Model

`SISSession.run()` performs these steps:

1. Classify each input file as RINEX navigation, SP3, observation, meteorology or
   unknown. Navigation files are read by `gnx_py.io.rinex_nav` (GNX's own
   fixed-column readers; georinex is no longer used): a file that cannot give
   the requested system raises a ValueError naming the file, its version and
   its systems (a RINEX 2 navigation file carries GPS only). RINEX 4 GPS and
   Galileo give the same SIS statistics as the RINEX 3 file of the same day
   (2024-035, 2 h, |RMS difference| <= 1 mm radial and 3D). An exact copy of a
   RINEX 2 record is dropped with a note; a conflicting copy raises R2OBSE04.
   SP3 files are read by `gnx_py.io.sp3_reader` (SP3-a/b/c/d): every epoch must
   list the header satellites in header order, and a damaged or cut file
   raises an error naming the line. Time scale: SP3 epochs and the output
   `time` are GPS time. Observation epochs tagged in BDT are moved to GPST
   (BDT = GPST - 14 s) when they are read; BeiDou navigation records keep
   their Toc/Toe (and the frame time index of `NavRead.bds`) in BDT as
   written, and the broadcast evaluation converts the GPST epoch to BDT
   before it uses them. Galileo System Time is taken as GPST (the GST - GPST
   offset, a few nanoseconds, is not applied).
2. Generate broadcast coordinates and clocks, or interpolate SP3 positions and
   clocks onto the requested epoch grid. Broadcast record per satellite and
   epoch (post-processing rule, RTKLIB `seleph`): the nearest Toe within the
   validity of the system, GPS 2 h (then the nearest record of any age),
   Galileo 3 h (RTKLIB `MAXDTOE_GAL`, 4 h before v1.0.0), BeiDou 2 h;
   for Galileo a record already transmitted at the epoch (TransTime <= epoch,
   an unknown TransTime counted from Toe) is preferred within the window
   (chosen among the measured variants:
   a Galileo record is sent about 11 min after its Toe, so the rule picks
   records 11-21 min old; BRUX 2024-035 E-BRDC PPP static IF 10.7 -> 17.3 cm,
   kinematic p95 1.106 -> 0.603 m; Galileo SIS A_rms / clk_rms medians +1.4 cm,
   no verdict changes). The choice depends on the records the file holds: until
   a satellite's first record of the day is transmitted (about 00:11) a daily
   file with the previous day's last records (BRDC00IGS) gives those, a file
   that starts at 00:00 (BRD400DLR) the nearest Toe.
3. Optionally apply satellite antenna PCO correction to precise products, in the
   body frame of the satellite attitude (see Satellite Attitude below). Against
   broadcast orbits the precise orbit is moved to the antenna phase centre of
   the message's clock-reference signal (since v1.0.0):
   GPS IF(L1,L2) (since v1.0.0), BeiDou B3I (new; the BDS ICDs refer the
   group delays to B3I; a BeiDou satellite needs a C06 ANTEX entry only), and
   `dR_apc_signal` reports the radial offset of the mode's APC; Galileo stays
   at the APC of the user signal (E5aE5b at IF(E1,E5b)), single-frequency E1
   and E5a included: at the IF reference of their message the radial error
   would grow (dR RMS 0.147 -> 0.208 m and 0.158 -> 0.240 m; informational, every mode passes). BDS-3 broadcast orbits
   sit at the B3I APC within -0.085..+0.046 m (mean radial offset, 3 days x COD
   / GRG); BDS-2 fits no reference point with the igs20 values (B3I
   +0.34..+1.72 m by day), so the validation grades the BDS-2 radial with its
   own limits (MEO 1.0 m, IGSO 2.0 m): measured class medians (up to
   0.94 / 1.82 m), looser than the 0.6 / 0.8 m BDS-3 keeps, with 6 % / 9 %
   headroom on the graded rows. BeiDou median R_rms MEO 0.717 -> 0.077 m.
4. Merge broadcast TGD or external DCB/OSB/DSB products into a common `TGD`
   column in meters.
5. Align both products on common satellite/epoch pairs.
6. Compute `dx`, `dy`, `dz`, `dt`, `dTGD`, radial/along/cross components
   (`dR`, `dA`, `dC`) and SISRE variants.
7. Optionally flag eclipse intervals.
8. Annotate BeiDou outputs with clock-datum status where relevant.

Clock differences are converted to meters using the speed of light. Position and
RAC differences are also reported in meters.

## Satellite Attitude

Since v1.0.0 the PCO step uses the body frame of `gnx_py.models.attitude`,
the same provider as PPP, SPP and TEC (`SISConfig.satellite_attitude`,
`attitude_laws`, `attitude_path`, as in `gnx_py.core.configuration.Config`):

- `None` / `"model"` (default): the published attitude law of the ANTEX block
  through the noon/midnight turns (GPS IIR/IIF/III, Galileo IOV/FOC, BeiDou
  CAST/SECM/BDS-2), or the ORBEX attitude of `attitude_path` where its
  analysis centre is the one of the SP3 input; yaw steering elsewhere.
- `"nominal"`: yaw steering everywhere (RTKLIB `satantoff`).

During a turn the horizontal PCO turns with the body. Broadcast vs GRG0MGXFIN
or GFZ0MGXRAP 2024-035 at 300 s, laws against nominal yaw: `dA` / `dC` up to
0.377 / 0.249 m (G32, GPS IIF x PCO 0.394 m), 0.217 / 0.209 m (G09), 0.138 /
0.122 m (E19), 0.061 / 0.080 m (C34); `dR` unchanged and `sisre` within
0.025 m. The output column `att_regime` gives the attitude regime per
satellite-epoch (0 yaw steering, 1-6 turn laws, 7 orbit-normal, 8 ORBEX, 9
nominal fallback). The BeiDou orbit type of the SISRE weights follows the
ANTEX block of the satellite (`gnss.bds_orbit_type(sv, block)`; the
PRN table was wrong for C31/C56 and after the PRN reassignment of
2026-04-17..21); without `atx_path` the class comes from the orbit itself
since v1.0.0 (median radius above 35,000 km is geosynchronous, a
geosynchronous satellite moving faster than 500 m/s in the Earth-fixed frame
is IGSO, else GEO; below that radius MEO; `SISSession.BDS_GEOSYNC_RADIUS_M`,
`BDS_IGSO_MIN_ECEF_SPEED`); the 2024 PRN table only for a frame without
positions and velocities. It equals the PRN table on every 2024 validation row.

PRN reassignment: a PRN whose satellite (SVN)
has no ANTEX entry from an epoch inside the run on (a PRN reassigned to another
satellite, e.g. igs20 C12: SVN C013 until 2026-04-21 04:00, C233 from 08:00,
or a gap without an entry) gives one UserWarning per run naming the PRN, the
SVNs and the change epoch, and its rows after that epoch are excluded. Entries
of the same SVN that follow each other within 60 s (`antex.PRN_CHANGE_CONTIGUITY`,
calibration splits such as C35 C215 2022-10-10 / 2023-01-01) are no change.
The rule acts in every session that reads the ANTEX: SIS with `atx_path`
(with or without the satellite PCO), PPP / SPP / TEC with `atx_path` and
`sat_pco` or `rec_pco`. Without an ANTEX nothing is checked (orbit products
carry no SVN). After a same-SVN gap the PRN is not restarted when its
satellite gets an entry again (conservative; no validation day has a PRN change).

## GPS and Galileo

GPS and Galileo are the standard active branches for broadcast-vs-precise and
SP3-vs-SP3 comparisons. The signal mode should match the navigation message and
the bias product used for the run:

- GPS examples: `L1L2` style ionosphere-free processing.
- Galileo examples: `E1E5a`, `E1E5b`, `E5a`, `E5b`.

When `compare_dcb=False`, non-BDS runs use a zero `TGD` placeholder so the
comparison table has a uniform schema. When `compare_dcb=True`, matching DCB,
DSB or OSB products are read and converted into the internal delay convention.
Each bias record applies from its BIAS_START to its BIAS_END (to the end of that
second, so a daily record ending at 23:59:59 still holds at 24:00); for a
comparison across midnight give the products of both days as a list
(`dcb_path_1=[day1, day2]`). A product without
the system, without the codes of the signal mode (e.g. Galileo `E1` needs C1C
and C7Q), or without a record valid at the compared epochs raises `ValueError`;
satellites the product does not cover get NaN `TGD`/`dTGD`/`sisre` and one
`RuntimeWarning` naming them (`sisre_notgd` is kept).

## BeiDou Status

BeiDou support is deliberately guarded because BDS SIS clock interpretation
depends strongly on the clock datum and the signal-specific group-delay model.

Recommended validated path:

- `system="C"`
- `bds_mode="B1IB3I"`
- MEO/IGSO satellites
- broadcast NAV vs precise SP3 clocks whose BDS-3 satellites refer to the
  IF(B1I,B3I) datum (C2I/C6I; the 2024 MGEX products of COD, GRG and GFZ).
  Since v1.0.0 the datum of the precise clocks is
  resolved from the files (a bias product of the SP3's analysis centre that
  declares it, else a test on the broadcast clocks): with BDS-3 clocks in
  IF(B1C,B2a) (COD MGEX 2026) mode B1IB3I converts them since v1.0.0: with a RINEX 4 navigation file the BDS-3 precise clocks get
  TGD_precise = c[k1 TGD1 - (k1' TGD_B1Cp - k2' TGD_B2ap)] (k1 = 2.944,
  k1' = 2.261, k2' = 1.261; the B-CNAV values are the medians of the
  satellite's CNV1 / CNV2 records of the file) and the status is "ok". BDS-2
  and BDS-3 GEO keep 0; a BDS-3 satellite without B-CNAV values gets NaN
  `sisre` and one warning; a RINEX 3 navigation file (no B-CNAV) converts
  nothing and keeps the "warning" with the remedy. With `compare_dcb=True` a
  product of the SP3's centre that declares the IF(B1C,B2a) datum should
  already give IF(B1I,B3I), but this path is not validated on data, so the
  status stays "warning" (as before v1.0.0). BRUX 2026-213 (COD MGEX 2026) BRD400DLR
  against COD0MGXFIN: clock RMS class median MEO / IGSO 4.00 / 2.39 -> 0.53 /
  0.82 m, SISRE 4.09 / 2.46 -> 0.49 / 0.89 m; the same clock statistics
  without an ANTEX.

The satellite generation (BDS-2 or BDS-3) that the clock datum, the
conversion and the band-7 code use comes from the ANTEX block of `atx_path` at
the first epoch, else from the B-CNAV records of the whole navigation file,
else from the PRN.

Requires caution:

- `B1I`, `B2I`, `B3I` single-signal comparisons against precise IF clocks.
- Any run where the precise clock datum is not known.
- Runs with external OSB/DSB products that are not demonstrably clock-reference
  consistent with the precise orbit/clock product.

Requires validation:

- Modern BDS modes involving `B1C`, `B2a` or `B2b`. The bias-product
  observable of B1C and B2a is taken pilot first, P, X, D
  (`BDS_PRODUCT_OBSERVABLE_ORDER`; the
  broadcast group delays are the pilot TGD_B1Cp / TGD_B2ap); with B2b the B2b
  priority comes first (B1CB2b keeps the CAS1 pair C1X-C7Z; TGD_B2bI refers
  to the B2b-I data component 7D, so C7D may be the better B2b observable:
  not settled).
- Modern RINEX 4 CNAV fields without a matching validated OSB/DSB stack.
- Any mixed clock/bias stack assembled from different analysis centers.

Unsupported in the SIS/SISRE path:

- BeiDou GEO satellites. They are skipped from precise BDS processing and raise
  if they reach the BDS SISRE coefficient path.

## Clock Datum, TGD, DCB, OSB and DSB

The most common interpretation trap is a large BDS clock error. It may not mean
that the orbit or clock product is wrong. It can simply mean that the broadcast
clock correction and the precise clock are expressed in different datums.

Important terms:

- `TGD`: broadcast group-delay correction represented internally in meters.
- `DCB` / `DSB`: differential code bias between two observables.
- `OSB`: observable-specific bias for one signal/code observable.
- `clock_bias`: optional per-epoch common clock datum removal before SISRE.
- `clock_convention_status`: output metadata describing whether the BDS clock
  component is directly comparable, warning-only, requires validation or
  unsupported.

For BDS:

- `B1IB3I` uses a legacy ionosphere-free broadcast datum correction and is the
  recommended validation mode.
- Single-signal modes can use a broadcast group delay, but this is not the same
  as proving comparability with a precise IF clock.
- DSB products define a pairwise bias and cannot define a single-signal datum by
  themselves.
- OSB products can support single-signal or two-signal conversion only when the
  observable stack matches the selected mode and precise-clock convention.

`compare_dcb` modes (`SIS_COMPARE_DCB_MODES`):

| System | Supported with `compare_dcb=True` | Kept as a readable `ValueError` |
|---|---|---|
| GPS | every `gps_mode` | - |
| Galileo | `E1`, `E5b`, `E5a`, `E1E5a` | `E1E5b`, `E5aE5b` |
| BeiDou | two-signal modes with an OSB product or the mode's own DSB pair; single-signal modes with an OSB product | single-signal modes with a DSB product; pairs reachable only through DSB chains (e.g. `B2aB2b`) |

The `ValueError` names the supported modes; enabling the others is a post-v1
item.

Always inspect:

```python
result.attrs["clock_convention_status"]
result.attrs["clock_convention_note"]
```

or the per-row `clock_convention_*` columns before interpreting BDS clock/SIS
statistics.

## Output Columns

`SISSession.run()` returns one row per compared satellite and epoch, index
`['sv', 'time']` (`time` in GPST). The frame is the processed `orb_path_0`
table with the comparison columns added; no column selection is applied (the
commented-out `output_cols` selection of earlier versions is not restored). The contract is:

- Always present: `dx`, `dy`, `dz`, `dt`, `dTGD`, `dR`, `dA`, `dC`,
  `dt_mean`, `dTGD_mean`, `sisre`, `sisre_orb`, `sisre_notgd` (meters, see
  below); `clk_0`, `clk_1` (the clocks of the two inputs, seconds); `TGD`
  (the group delay of `orb_path_0` in the comparison convention, meters);
  `x`, `y`, `z`, `vx`, `vy`, `vz`, `clk` and `x_apc`, `y_apc`, `z_apc` of
  `orb_path_0` (the position the differences use is the `*_apc` one: with
  `apply_satellite_pco=True` the antenna phase centre of the comparison
  signal; otherwise the input's own reference point, i.e. the antenna phase
  centre the message refers to for a broadcast input, where `x_apc = x`,
  and the centre of mass for an SP3 input).
- `xs`, `ys`, `zs` (the Sun): only when the eclipse or the satellite PCO
  step ran (`apply_eclipse=True` or `apply_satellite_pco=True`); absent when
  both are off.
- `eclipse_raw`, `eclipse`: with `apply_eclipse=True` (the default).
- `att_regime`: when a precise input went through the PCO step (attitude
  regime, see Satellite Attitude).
- `dR_apc_signal`: broadcast against precise with satellite PCO when the
  precise orbit is moved to a reference APC other than the mode's own (GPS
  modes other than L1L2, BeiDou modes other than B3I, Galileo E5aE5b): the
  radial offset of the mode's APC from that reference, reported only (it
  enters neither `dR` nor `sisre`); `attrs["apc_reference"]` and
  `attrs["apc_signal"]` name both.
- `clock_convention_status`, `clock_convention_mode`,
  `clock_convention_directly_comparable`, `clock_convention_note`: BeiDou runs
  (also in `attrs`, with `clock_convention_system`).
- Pass-through columns of the inputs: the navigation-record fields of a
  broadcast `orb_path_0` (system-specific, e.g. `Toe`, `IODE`, `TGD1`,
  `TransTime`, the Galileo health flags) and `dt_rel` / `clk_raw` of an SP3
  `orb_path_0`. They depend on the input type and are not part of the
  contract; neither are the internal `_sisre_kr`, `_sisre_kb`, `_sisre_kac`,
  `_sisre_krb` (the per-row SISRE coefficients).

## SISRE and RAC Components

The output includes:

- `dx`, `dy`, `dz`: Cartesian position differences.
- `dt`: clock difference in meters.
- `dTGD`: group-delay difference in meters.
- `dR`, `dA`, `dC`: radial (positive outward), along-track and cross-track
  (along the normal of the inertial orbit plane, r x (v + w_E x r) with the
  Earth-fixed velocity v) components.
- `dt_mean`, `dTGD_mean`: epoch datums; `dt'' = (dt - dt_mean) - (dTGD - dTGD_mean)`.
  `dt_mean` is taken over all satellites with a clock; the datum of `dt''` over
  the satellites having both a clock and a group delay. For BeiDou the datums
  are taken per satellite generation (BDS-2, BDS-3): the two clock ensembles
  have independent datums; a
  generation with fewer than 3 satellites at an epoch
  (`SISSession.BDS_CLOCK_DATUM_MIN_SATS`) gives NaN `sisre` / `sisre_notgd`
  there, with one `RuntimeWarning`.
- `sisre`: SISRE including clock and group-delay difference.
- `sisre_orb`: orbit-only SISRE component.
- `sisre_notgd`: SISRE-like metric without TGD correction.

All differences are `orb_path_0` minus `orb_path_1`. SISRE follows Montenbruck et
al. (2015, GPS Solut 19:321-333; 2018, Adv Space Res 61:3020-3038):

    sisre     = sqrt((w_R*dR - dt'')^2 + w_AC^2*(dA^2 + dC^2))
    sisre_orb = sqrt(w_R^2*dR^2 + w_AC^2*(dA^2 + dC^2))

with `dt''` the clock difference after removing the epoch datum and the TGD
difference (`sisre_notgd` uses the clock difference without TGD).

Meaning of `TGD`, `sisre` and `sisre_notgd` per signal mode (a convention): GPS and Galileo keep the group delay of the message for every mode
(the clock minus TGD/BGD is the L1 P(Y) or E1 user's clock), so for GPS L2, L5
and the IF modes and for Galileo E5a/E5b `sisre` is the error seen by the L1
P(Y) / E1 user, not by a user of the mode's own signal. BeiDou applies the
group delay of the mode's signal or IF combination. BeiDou `sisre_notgd`
compares the B3I-referenced broadcast clock with the IF precise clock (about
8 m median) and is informational only.

Default weights
`(w_R, w_AC^2)`: GPS (0.98, 1/49), GLONASS (0.98, 1/45), Galileo (0.98, 1/61),
BDS MEO (0.98, 1/54), BDS IGSO (0.99, 1/126), QZSS (0.99, 1/126). BDS GEO is not
supported. `run(kr, kb, kac, krb)` overrides the coefficients of the expanded form
`kr*dR^2 + kb*dt''^2 + kac*(dA^2 + dC^2) + krb*dR*dt''` (literature:
`w_R^2, 1, w_AC^2, -2*w_R`); a coefficient left as `None` keeps its default.
The former defaults `(0.9604, 1.0, 0.019881, -1.960)` passed together select the
literature weights as well, with a `FutureWarning`.

Neighbour-day files: `prev_sp3_0`/`next_sp3_0` extend `orb_path_0`,
`prev_sp3`/`next_sp3` extend `orb_path_1`. A neighbour must adjoin its nominal day
(gap at most 1 h, otherwise `ValueError`); a neighbour from another agency (SP3
header) gives a `RuntimeWarning`, since its clocks refer to another datum.

## Validation

The module was validated for v1.0.0 over a configuration matrix (broadcast
against precise, SP3 x SP3, PCO, eclipse, `compare_dcb`, BeiDou clock
conventions) against independent references, with negative cases (readable
errors), the `SISController` alias, the broadcast APC reference, the Galileo
record selection, the attitude used by the PCO step and the navigation and SP3
readers. Configurations that stay outside their limits for an established
cause are listed as known limitations in `CHANGELOG.md`.

Executed tutorial notebooks and the scripts under `examples/` are
illustrations, not validation references.

## Developer Notes

- `SISConfig` and `SISSession` are active public API.
- `SISController` is a deprecated alias of `SISSession` (warning on
  instantiation), kept until a future major release.
- `PPP`, `SPP` and ionosphere code paths are not owned by this module.
- `gnx_py.orbits.__all__` is explicit but compatibility-preserving. It still
  includes historical accidental exports such as `np`, `pd`, `re`, `warnings`
  and imported helper classes because older wildcard imports may depend on them.
- Future API cleanup should first add deprecation notes and import tests, then
  remove accidental exports only in a planned compatibility-breaking release.
- BDS helpers in `config.py` are intentionally explicit rather than deeply
  abstracted. The clock-datum logic is mathematically sensitive and should be
  changed only with regression data across legacy and modern BDS products.

## Known Limitations

- BDS GEO SIS/SISRE comparison is unsupported.
- Modern BDS modes require more validation before they should be used as
  production clock-error metrics.
- Large BDS clock errors may indicate clock-datum mismatch rather than product
  failure.
- Bias products must match the selected signal mode and precise-clock datum.
- SP3 interpolation near product boundaries benefits from adjacent-day files.
- BeiDou B1IB3I against precise clocks in IF(B1C,B2a) (COD MGEX 2026) is
  converted with the B-CNAV group delays of a RINEX 4 navigation file (since v1.0.0); with a RINEX 3 file, or with `compare_dcb=True` and a
  product declaring that datum, the status stays "warning" (not validated on
  data; when `dcb_path` is a list, the product used is not checked to be the
  declaring one). The datum detection reads the SP3 and navigation files once
  more per run.
- BDS-2 broadcast orbits fit no antenna reference point with the igs20 PCO
  values; their radial is graded with the measured limits (MEO 1.0 m, IGSO
  2.0 m).
- Product properties that show in the comparisons and are not library
  defects: the GRG GPS OSBs do not satisfy the IF(C1W,C2W) clock-datum
  condition a*OSB_C1W - b*OSB_C2W = 0 (2024-035: median 0.05 m, up to 0.38 m
  for G30/G10; COD and GFZ: 0), which shows in SP3 x SP3 group-delay
  comparisons; the COD0MGXFIN 2024-035 clock of C06 is +4.02 m off
  GRG, so the BeiDou IGSO SISRE p95 against COD is 3.28 m (1.36 m without
  C06; known limitation LIM-ORB-01 in `CHANGELOG.md`).
- Public wildcard exports are intentionally broad for compatibility and are not
  a clean API boundary yet.

## Recommended Workflow

1. Start with GPS or Galileo if validating the mechanical orbit comparison.
2. For BDS, start with `B1IB3I` MEO/IGSO.
3. Inspect `clock_convention_status` before interpreting BDS clock metrics.
4. Add external OSB/DSB products only when the product stack is known to be
   clock-datum consistent.
5. Use modern BDS modes for validation experiments, not default production
   reporting, until the full RINEX 4 plus bias stack is validated.
