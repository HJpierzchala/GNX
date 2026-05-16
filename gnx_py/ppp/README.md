# GNX-py PPP

The `gnx_py.ppp` package implements Precise Point Positioning workflows for
research, validation, and engineering experiments with high-precision GNSS
data. It combines RINEX observation loading, orbit/clock interpolation,
antenna and propagation corrections, cycle-slip preprocessing, extended Kalman
filter PPP models, diagnostics, and optional ambiguity-resolution logic.

The module is intentionally broad: it supports both mature reference paths and
newer multi-GNSS development paths. The public API is currently preserved for
compatibility, so some legacy classes remain importable even when they are not
recommended for new routing.

## Highlights

- Combined ionosphere-free PPP for single-system and mixed-system runs.
- Uncombined PPP with frequency-specific code/phase observations.
- Single-system and mixed-system filtering.
- GPS, Galileo, and BeiDou/BDS support.
- Precise SP3 orbit/clock workflows and broadcast-orbit validation workflows.
- Ionospheric constraints from GIM/STEC-style preprocessing.
- Bias product support through OSB/DCB/DSB-style corrections.
- PPP-AR helpers with ratio testing, arc-age gates, partial fixing, and soft
  hold constraints.
- Controlled trace diagnostics via `PPPConfig.trace_filter`.

## Main Entry Points

Most users should start with:

- `gnx_py.PPPConfig` or `gnx_py.ppp.PPPConfig`: main configuration object.
- `gnx_py.PPPSession` or `gnx_py.ppp.PPPSession`: in-package PPP session
  orchestrator.
- `PPPSession.py`: repository-level runnable driver for sample and validation
  workflows.

`PPPSession.py` is a script/driver, not a configuration class. It builds
`PPPConfig`, loads product paths from environment variables, runs the session,
prints summaries, and writes CSV outputs.

## Supported Systems

| System | Code | Status |
|---|---|---|
| GPS | `G` | Mature across combined and uncombined workflows. |
| Galileo | `E` | Mature across combined and uncombined GPS/Galileo workflows. |
| BeiDou/BDS | `C` | Supported and tested, best starting point is `B1IB3I`; mixed and AR paths need careful validation. |

The safest BDS defaults are:

- `sys="C"` or mixed sets containing `"C"` only when matching products are
  available.
- `bds_freq="B1IB3I"` for dual-frequency PPP.
- Combined PPP or BDS-only uncombined PPP before attempting advanced mixed
  no-constraint or AR experiments.

## PPP Modes

### Combined PPP

Use `positioning_mode="combined"` for ionosphere-free code and phase
combinations. This mode is usually the easiest high-precision PPP starting
point when reliable dual-frequency observations and precise products are
available.

Routing:

- One active system: `PPPDualFreqSingleGNSS`.
- Multiple active systems: `PPPDualFreqMultiGNSS`.

Combined PPP does not use ionospheric constraints because the first-order
ionosphere term is removed by the ionosphere-free observable combination.

### Uncombined PPP

Use `positioning_mode="uncombined"` to keep frequency-specific observables in
the filter. This is more flexible and exposes ionosphere/bias states directly,
but it is also more sensitive to product consistency and stochastic modeling.

Routing highlights:

- Single-system dual-frequency: `PPPUdSingleGNSS`.
- Single-system single-frequency with ionospheric constraints:
  `PPPUducSFSingleGNSS`.
- Mixed no-constraints: `PPPUdGenericMixedGNSS`.
- GPS/Galileo dual-frequency with ionospheric constraints:
  `PPPFilterMultiGNSSIonConst` reference branch.
- Generic G/E/C ionospheric constraints:
  `PPPFilterMultiGNSSIonConstGEC`.

`positioning_mode="single"` is a legacy compatibility alias. Prefer
`positioning_mode="uncombined"` with a single-frequency signal mode such as
`L1`, `E1`, or `B1I`.

## Recommended Configurations

### Stable GPS/Galileo Combined PPP

```python
from gnx_py.ppp import PPPConfig, PPPSession

cfg = PPPConfig(
    obs_path="sample_data/STATION.crx.gz",
    sp3_path=[
        "sample_data/COD0OPSFIN_20240340000_01D_05M_ORB.SP3",
        "sample_data/COD0OPSFIN_20240350000_01D_05M_ORB.SP3",
    ],
    atx_path="sample_data/igs20.atx",
    dcb_path="sample_data/COD0OPSFIN_20240350000_01D_01D_OSB.BIA",
    gim_path="sample_data/COD0OPSFIN_20240350000_01D_01H_GIM.INX",
    sinex_path="sample_data/IGS0OPSSNX_20240350000_01D_01D_CRD.SNX",
    sys={"G", "E"},
    gps_freq="L1L2",
    gal_freq="E1E5a",
    positioning_mode="combined",
    orbit_type="precise",
)

result = PPPSession(cfg).run()
print(result.solution.tail())
```

### BDS Single-System PPP

```python
cfg = PPPConfig(
    obs_path="sample_data/STATION.crx.gz",
    sp3_path=["sample_data/COD0MGXFIN_20240350000_01D_05M_ORB.SP3"],
    atx_path="sample_data/igs20.atx",
    dcb_path="sample_data/CAS0OPSRAP_20240350000_01D_01D_DCB.BIA",
    gim_path="sample_data/COD0OPSFIN_20240350000_01D_01H_GIM.INX",
    sys="C",
    bds_freq="B1IB3I",
    positioning_mode="combined",
    orbit_type="precise",
)
```

### Mixed G/E/C Uncombined Without Constraints

```python
cfg = PPPConfig(
    obs_path="sample_data/STATION.crx.gz",
    sp3_path=[
        "sample_data/COD0OPSFIN_20240350000_01D_05M_ORB.SP3",
        "sample_data/COD0MGXFIN_20240350000_01D_05M_ORB.SP3",
    ],
    atx_path="sample_data/igs20.atx",
    dcb_path=[
        "sample_data/COD0OPSFIN_20240350000_01D_01D_OSB.BIA",
        "sample_data/CAS0OPSRAP_20240350000_01D_01D_DCB.BIA",
    ],
    gim_path="sample_data/COD0OPSFIN_20240350000_01D_01H_GIM.INX",
    sys={"G", "E", "C"},
    gps_freq="L1L2",
    gal_freq="E1E5a",
    bds_freq="B1IB3I",
    positioning_mode="uncombined",
    use_iono_constr=False,
)
```

Mixed no-constraint uncombined PPP is powerful, but it is product- and
bias-sensitive. Use it with trace diagnostics and numerical comparisons before
treating new signal/product combinations as operational.

### Uncombined PPP With Ionospheric Constraints

```python
cfg = PPPConfig(
    obs_path="sample_data/STATION.crx.gz",
    sp3_path=["sample_data/COD0OPSFIN_20240350000_01D_05M_ORB.SP3"],
    atx_path="sample_data/igs20.atx",
    dcb_path="sample_data/COD0OPSFIN_20240350000_01D_01D_OSB.BIA",
    gim_path="sample_data/COD0OPSFIN_20240350000_01D_01H_GIM.INX",
    sys={"G", "E"},
    positioning_mode="uncombined",
    use_iono_constr=True,
    use_iono_rms=True,
)
```

Use constraints when you have a trusted ionosphere source and want the filter
to estimate slant ionosphere states while being guided by external information.

## Important `PPPConfig` Fields

- `sys`: active constellations: `"G"`, `"E"`, `"C"` or a set.
- `positioning_mode`: `"combined"` or `"uncombined"`; `"single"` is legacy.
- `gps_freq`, `gal_freq`, `bds_freq`: signal modes that determine
  single-frequency vs dual-frequency routing.
- `orbit_type`: `"precise"` for SP3 products or `"broadcast"` for NAV products.
- `sp3_path`, `nav_path`, `atx_path`, `dcb_path`, `gim_path`, `sinex_path`:
  external product paths.
- `use_iono_constr`: selects constrained vs no-constraint uncombined branches.
- `use_iono_rms`, `sigma_iono_0`, `sigma_iono_end`, `t_end`: stochastic model
  for ionospheric constraints.
- `pppar_enabled`: enables PPP-AR attempts where supported.
- `pppar_combined_if_ar_enabled`: additional experimental gate for combined
  ionosphere-free AR.
- `trace_filter`: enables compact `[PPP TRACE]` diagnostics.

See the `PPPConfig` docstring in `gnx_py/configuration.py` for field-by-field
details.

## Bias Products

Bias handling is central to uncombined and multi-GNSS PPP.

- OSB/BIA products are useful for precise code/phase bias consistency.
- DCB/DSB-style products are used by ionosphere and BDS/STEC workflows.
- Mixed-system uncombined PPP is especially sensitive to whether GPS, Galileo,
  and BDS products share compatible datums.
- BDS workflows may require CAS/CODE product choices depending on the branch
  under validation.

If results look unstable in mixed uncombined mode, verify bias products before
tuning the Kalman filter.

## PPP-AR Status

PPP ambiguity resolution is available through `gnx_py.ppp.pppar` helpers and
selected filter branches. The implementation includes:

- LAMBDA integer least-squares search.
- Ratio testing.
- Arc-age / lock-time gating.
- Elevation-based candidate filtering.
- Optional partial fixing.
- Soft hold constraints after accepted fixes.

Recommendations:

- Keep `pppar_enabled=False` for baseline PPP validation.
- Enable AR only after float PPP is stable.
- Treat combined ionosphere-free AR as experimental; it requires both
  `pppar_enabled=True` and `pppar_combined_if_ar_enabled=True`.
- Treat BDS AR as requiring additional phase-bias validation.

## BeiDou Notes

BeiDou support is active in the PPP package:

- RINEX loading and preprocessing support BDS signals.
- BDS single-system combined and uncombined PPP are tested.
- `B1IB3I` is the safest current dual-frequency default.
- Mixed G/E/C routing exists for combined and uncombined workflows.

Known caution areas:

- BDS bias products and signal datums must match the selected branch.
- Mixed no-constraint uncombined PPP can merge ionosphere and receiver-bias
  effects; compare carefully before interpreting absolute states.
- PPP-AR with BDS is not treated as production-stable without additional
  phase-bias validation.
- BDS GEO/broadcast edge cases should be validated separately from MEO/IGSO
  workflows.

## Developer Notes

Active primary classes:

- `PPPDualFreqSingleGNSS`
- `PPPDualFreqMultiGNSS`
- `PPPUducSFSingleGNSS`
- `PPPUdSingleGNSS`
- `PPPUdGenericMixedGNSS`
- `PPPFilterMultiGNSSIonConstGEC`

Reference / legacy classes:

- `PPPFilterMultiGNSSIonConst`: GPS/Galileo constrained reference baseline.
- `_LegacyPPPDualFreqMultiGNSS`: legacy mixed combined reference.
- `PPPUdMultiGNSS`, `PPPUducSFMultiGNSS`, `PPPUdMixedGNSS`:
  compatibility paths that remain tested or routed in selected cases.
- `PPPSingleFreqMultiGNSS` and `PPPSingleFreqSingleGNSS`:
  legacy/compatibility single-frequency paths.

Do not remove or reroute legacy classes only because a generic class exists.
Several paths are kept for reference comparisons, public API compatibility, or
special routing cases.

`gnx_py.ppp.__all__` currently preserves a broad compatibility surface. Some
exports are historical accidents and may be narrowed only through a future API
migration.

## Diagnostics and Trace

Set `trace_filter=True` to enable controlled trace output. Trace lines use a
compact format such as:

```text
[PPP TRACE] PPPUdSingleGNSS epoch=123 time=... event=epoch-summary n_sats=...
```

Trace mode is meant for debugging filter events, resets, convergence, and
screening outcomes. It avoids large DataFrame dumps in normal library use.

## Validation Status

The PPP package is covered by unit tests for:

- PPP-AR helper behavior.
- Public API import compatibility.
- GPS/Galileo constrained and unconstrained routing.
- BeiDou signal metadata, PPP routing, bias handling and broadcast cross-branch
  behavior.
- Selected numerical helper behavior.

The test suite protects important branches, but it is not a formal accuracy
guarantee for every possible constellation, product source, station, signal
mode, receiver, and arc condition. New scientific uses should still validate
against independent products, known coordinates, and representative datasets.

## Known Limitations

- Combined IF PPP-AR is experimental and disabled by default.
- BDS AR requires additional phase-bias validation.
- Mixed uncombined no-constraint PPP is sensitive to bias/product consistency.
- Some legacy classes remain public for compatibility and should not be used as
  the starting point for new workflows unless validating historical behavior.
- Rerouting or consolidating classes requires before/after numerical
  regression tests.
- The broad public API is currently compatibility-preserving and includes
  symbols that are not intended long-term PPP API.

