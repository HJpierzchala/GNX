# STEC Bias Policy

This note documents how STEC satellite and station biases are selected in
`gnx_py.ionosphere`.

`TECSession` processes one constellation at a time. Use `use_sys="G"`,
`use_sys="E"` or `use_sys="C"` and choose the matching signal pair through
`gps_freq`, `gal_freq` or `bds_freq`.

## Minimal Example

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
    dcb_path="sample_data/bias_data/GRG/GRG0MGXFIN_20240350000_01D_01D_OSB.BIA",
    add_dcb=True,
    add_sta_dcb=False,
    rcv_dcb_source="none",
    stec_bias_sources=("product", "gim", "config", "zero"),
    stec_missing_bias="warn_zero",
    time_limit=[datetime(2024, 2, 4, 0, 0), datetime(2024, 2, 4, 0, 10)],
)

out = TECSession(cfg, use_sys="C").run()
```

## Geometry-Free Convention

STEC uses:

```text
P4 = P2 - P1
```

The concrete code columns are selected from the configured signal mode. For
BeiDou `B1IB3I`, the current BDS tests verify `C2I/C6I`.

## OSB vs DCB/DSB

- OSB products are observable-specific. The STEC policy applies them as
  `OSB(P1) - OSB(P2)`.
- DCB/DSB products are already differential for a code pair. A direct
  `BIAS_code1_code2` column is used as-is; a reversed pair is used with the sign
  flipped.
- Biases are applied to the code geometry-free term, not to phase.
- Product values are converted to meters through the common bias helpers.

## Source Priority

`TECConfig.stec_bias_sources` defines fallback order. The default style is:

```python
stec_bias_sources=("product", "gim", "config", "zero")
```

Supported source names:

- `product`: OSB or DCB/DSB product columns attached to observations.
- `gim`: GIM/IONEX DCB values in columns such as `bias` or `sta_bias`.
- `config`: manual nanosecond values from `define_satellite_dcb` and
  `define_station_dcb`.
- `zero`: zero correction.

If `zero` is omitted, the policy appends it internally so the fallback is
well-defined unless `stec_missing_bias="raise"` is selected.

## Missing Bias Behavior

`TECConfig.stec_missing_bias` controls missing sources:

- `warn_zero`: emit `UserWarning` and logger warning, then use zero.
- `zero`: use zero without warning.
- `raise`: raise `KeyError`.

Use `raise` for validation runs where silently missing bias products would hide
a configuration problem.

## Satellite Bias

Satellite bias is used when:

```python
stec_bias_enabled=True
add_dcb=True
```

Manual values are in nanoseconds:

```python
define_satellite_dcb={"C06": 1.2, "C": 0.5, "default": 0.0}
```

Dictionary lookup checks the concrete SV, base SV, system and `default`.

## Station Bias

Station bias is used when:

```python
add_sta_dcb=True
rcv_dcb_source != "none"
```

Supported station paths:

- `rcv_dcb_source="calibrate"`: use receiver DCB estimated by `Calibration`
  when available.
- `rcv_dcb_source="gim"`: use GIM station DCB.
- `rcv_dcb_source="defined"`: use product/config fallback.
- `rcv_dcb_source="none"`: disable station bias.

Manual station values are in nanoseconds:

```python
define_station_dcb={"BRUX": 2.0, "C": 1.0, "default": 0.0}
```

## Common Configurations

Satellite and station from products:

```python
TECConfig(
    dcb_path="sample_data/bias_data/GRG/GRG0MGXFIN_20240350000_01D_01D_OSB.BIA",
    add_dcb=True,
    add_sta_dcb=True,
    rcv_dcb_source="defined",
    stec_bias_sources=("product", "zero"),
)
```

Satellite only from products:

```python
TECConfig(
    dcb_path="sample_data/bias_data/GRG/GRG0MGXFIN_20240350000_01D_01D_OSB.BIA",
    add_dcb=True,
    add_sta_dcb=False,
    rcv_dcb_source="none",
    stec_bias_sources=("product", "zero"),
)
```

GIM fallback:

```python
TECConfig(
    gim_path="sample_data/IGS0OPSFIN_20240350000_01D_02H_GIM.INX",
    ionosphere_model="gim",
    add_dcb=True,
    add_sta_dcb=True,
    rcv_dcb_source="gim",
    stec_bias_sources=("gim", "zero"),
)
```

No bias correction:

```python
TECConfig(
    stec_bias_enabled=False,
    add_dcb=False,
    add_sta_dcb=False,
    rcv_dcb_source="none",
)
```

## Diagnostics

`STECMonitor` adds:

- `stec_sat_bias_m`
- `stec_station_bias_m`
- `stec_total_bias_m`
- `stec_sat_bias_source`
- `stec_station_bias_source`
- `stec_bias_code_1`
- `stec_bias_code_2`

These columns should be inspected whenever changing product source, signal pair
or constellation.

## BeiDou Notes

BeiDou STEC support is active, with `B1IB3I` currently the best-covered pair in
tests. Modern BDS pairs depend on available code observations and matching bias
products. If a product lacks the selected BDS pair, prefer `stec_missing_bias`
set to `raise` during validation.
