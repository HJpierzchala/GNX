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
- Broadcast ionosphere models: Klobuchar and NTCM-style helpers.
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
   detects cycle slips.
3. `STECMonitor` groups observations by satellite arc, resolves bias terms,
   levels phase geometry-free observations to code geometry-free observations,
   and returns diagnostic STEC columns.

Important output columns include:

- `leveled_tec`, `median_leveled_tec`, `code_tec`, `phase_tec`
- `stec_sat_bias_m`, `stec_station_bias_m`, `stec_total_bias_m`
- `stec_sat_bias_source`, `stec_station_bias_source`
- `stec_bias_code_1`, `stec_bias_code_2`

## Systems And Signals

GPS and Galileo are the historical paths. BeiDou is supported through the shared
signal registry, with `B1IB3I` currently the best-tested BDS STEC pair in the
repository. Modern BDS pairs can be configured when the observation and bias
products contain matching code observables, but should be validated before
scientific use.

## GIM And IONEX

Use `gnx_py.ionosphere.ionex.GIMReader` to read IONEX/INX Global Ionosphere
Maps and optional DCB blocks. `TECInterpolator` evaluates map values at IPPs
using a thin-shell geometry and elevation mapping. GIM products are also used as
background data for station DCB calibration and model comparison.

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
- `klobuchar`: GPS broadcast Klobuchar delay model.
- `ntcm`: NTCM-style helper functions, primarily used for Galileo-style model
  comparisons and experimental grid generation.

`compare_models` on `TECConfig` can add model comparison columns, but this path
is experimental and should be validated for the selected system and product set.

## Monitoring

`monitoring.py` provides:

- `compute_sid`: SIDX from STEC temporal gradients in TECU/min.
- `compute_gix`: GIX and GIX-family spatial gradient diagnostics.
- `compute_roti_links`: ROT/ROTI per station-satellite link.
- Plotting helpers for ROTI and GIX.

`load_stec_folder` loads processed STEC parquet outputs for network monitoring.

## Kriging

`IonoKrigingMonitor` supports experimental VTEC gridding with WAAS/Sparks,
ordinary kriging and universal kriging. It can work in geographic coordinates or
with `SolarGeomagneticTransformer` for solar-geomagnetic coordinates.

This is a research-facing API. Validate variogram parameters, neighborhood
selection and coordinate-frame assumptions before operational use.

## Known Limitations

- The package-level public API is intentionally broad for backward
  compatibility and currently exports some accidental symbols such as `np`,
  `pd` and imported helper classes.
- `Calibration` is still used for receiver DCB estimation but is model-dependent
  and should be changed only with numerical regression tests.
- GIM/IONEX DCB parsing supports GPS, Galileo and BeiDou records used by current
  tests, but product format variations may require additional parsers.
- BDS support depends heavily on product bias coverage for the selected code
  pair.
- Kriging, `compare_models` and NTCM grid generation are experimental.

## Developer Notes

- Active core: `TECConfig`, `TECSession`, `STECMonitor`,
  `apply_stec_bias_policy`, `GIMReader`, `TECInterpolator`, `get_ipp`,
  `stec_mf`, `klobuchar`, `ntcm_vtec`.
- Legacy/compatibility: `Calibration`, root-level `STECSession.py`, and parts of
  older `gnx_py.tools` preprocessing paths.
- Requires caution: `STECMonitor._process_arc`, `GIMReader._parse_dcb`, BDS bias
  handling and PPP's `measure STEC` bridge.
- Experimental: kriging classes, `compare_models`, `compute_ntcm_grid`.

## Validation

The module has focused tests for IPP geometry, STEC bias fallback behavior,
BeiDou B1I/B3I routing, GIM DCB parsing and package public API compatibility.
Those tests protect library behavior, but they do not certify every possible
product/system/signal combination.
