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
- GPS, Galileo and BeiDou processing branches.
- BeiDou legacy `B1IB3I` MEO/IGSO validation.
- Guarded BeiDou modern signal experiments (`B1C`, `B2a`, `B2b`) when matching
  RINEX 4 navigation fields and clock-consistent bias products are available.
- Optional DCB/OSB/DSB product ingestion.
- Optional satellite PCO correction using ANTEX.
- Optional eclipse flagging and post-eclipse mask extension.
- Compatibility exports for older `from gnx_py.orbits import *` users.

## Main API

The recommended programmatic entry points are:

- `SISConfig`: configuration object for one SIS comparison run.
- `SISController`: high-level controller that executes the configured pipeline.

Example:

```python
from datetime import datetime

from gnx_py.orbits import SISConfig, SISController

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

result = SISController(config).run()
print(result[["dR", "dA", "dC", "dt", "dTGD", "sisre"]].head())
print(result.attrs.get("clock_convention_status"))
```

## Processing Model

`SISController.run()` performs these steps:

1. Classify each input file as RINEX navigation, SP3, observation, meteorology or
   unknown.
2. Generate broadcast coordinates and clocks, or interpolate SP3 positions and
   clocks onto the requested epoch grid.
3. Optionally apply satellite antenna PCO correction to precise products.
4. Merge broadcast TGD or external DCB/OSB/DSB products into a common `TGD`
   column in meters.
5. Align both products on common satellite/epoch pairs.
6. Compute `dx`, `dy`, `dz`, `dt`, `dTGD`, radial/along/cross components
   (`dR`, `dA`, `dC`) and SISRE variants.
7. Optionally flag eclipse intervals.
8. Annotate BeiDou outputs with clock-datum status where relevant.

Clock differences are converted to meters using the speed of light. Position and
RAC differences are also reported in meters.

## GPS and Galileo

GPS and Galileo are the standard active branches for broadcast-vs-precise and
SP3-vs-SP3 comparisons. The signal mode should match the navigation message and
the bias product used for the run:

- GPS examples: `L1L2` style ionosphere-free processing.
- Galileo examples: `E1E5a`, `E1E5b`, `E5a`, `E5b`.

When `compare_dcb=False`, non-BDS runs use a zero `TGD` placeholder so the
comparison table has a uniform schema. When `compare_dcb=True`, matching DCB,
DSB or OSB products are read and converted into the internal delay convention.

## BeiDou Status

BeiDou support is deliberately guarded because BDS SIS clock interpretation
depends strongly on the clock datum and the signal-specific group-delay model.

Recommended validated path:

- `system="C"`
- `bds_mode="B1IB3I"`
- MEO/IGSO satellites
- broadcast NAV vs precise SP3 clocks tied to the legacy C2I/C6I-style
  ionosphere-free datum

Requires caution:

- `B1I`, `B2I`, `B3I` single-signal comparisons against precise IF clocks.
- Any run where the precise clock datum is not known.
- Runs with external OSB/DSB products that are not demonstrably clock-reference
  consistent with the precise orbit/clock product.

Requires validation:

- Modern BDS modes involving `B1C`, `B2a` or `B2b`.
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

Always inspect:

```python
result.attrs["clock_convention_status"]
result.attrs["clock_convention_note"]
```

or the per-row `clock_convention_*` columns before interpreting BDS clock/SIS
statistics.

## SISRE and RAC Components

The output includes:

- `dx`, `dy`, `dz`: Cartesian position differences.
- `dt`: clock difference in meters.
- `dTGD`: group-delay difference in meters.
- `dR`, `dA`, `dC`: radial, along-track and cross-track components.
- `sisre`: SISRE including clock and group-delay difference.
- `sisre_orb`: orbit-only SISRE component.
- `sisre_notgd`: SISRE-like metric without TGD correction.

For BDS default coefficients, MEO and IGSO satellites use separate validated
alpha/beta coefficient mappings. GEO coefficients are not implemented.

## Validation and Runner Scripts

The repository includes focused tests and validation scripts for:

- BDS broadcast orbit generation and GEO skipping.
- BDS legacy and modern clock-convention policy.
- BDS TGD, OSB and DSB conversion.
- RINEX 4 modern BDS field preservation.
- Public API compatibility for `gnx_py.orbits`.

Example runner scripts under `scripts/` produce compact summaries in `/private`
temporary output directories. They are diagnostic tools, not the stable public
API.

## Developer Notes

- `SISConfig` and `SISController` are active public API.
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
