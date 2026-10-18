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
    dcb_path="sample_data/GRG0MGXFIN_20240350000_01D_01D_OSB.BIA",
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
BeiDou `B1IB3I` the validation checks `C2I/C6I`; the modern pairs `B1CB2a`,
`B1CB2b`, `B1CB3I` and `B2aB2b` are checked too (see
`README.md`, Systems And Signals).

## OSB vs DCB/DSB

- OSB products are observable-specific. The STEC policy applies them as
  `OSB(P1) - OSB(P2)`.
- DCB/DSB products are already differential for a code pair. A direct
  `BIAS_code1_code2` column is used as-is; a reversed pair is used with the sign
  flipped.
- A product without the pair's DSB takes the shortest chain of its DSBs, e.g.
  GPS `C1W-C5Q = (C1W-C2W) + (C2W-C1C) + (C1C-C5Q)` when CAS1 gives no C1W-C5Q
  DSB (label `product_dsb_chain`). A link without a value leaves the sample
  without a value; a station chain needs every receiver (`STA_BIAS_`) link.
- Each arc is levelled once without biases; the bias constants of the arc are
  then added to the levelled phase (`leveled_tec = stec_full`, see
  Diagnostics). Adding a constant before or after carrier-to-code levelling is
  the same; the code geometry-free term `P4` itself is left as observed.
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

A source that gives no value for a sample is missing, never zero: an OSB pair
needs both OSBs (a satellite without `OSB(P2)` gets no `OSB(P1) - 0`), a GIM
without a DCB for the satellite or system and a calibration without usable data
are missing as well.

One source gives the bias of a whole arc: the first source of the chain with a
value on any sample of the arc. Its values become one constant, the
levelling-weighted (`sin^4 E`) mean over the samples it covers, applied to every
sample of the arc; the share of covered samples is in
`stec_sat_bias_coverage` / `stec_station_bias_coverage`. A bias that changed
inside a continuous phase arc (two daily products across midnight: up to 1.4 ns,
3.2 TECU on Galileo E1/E5a) would otherwise put a step into the levelled STEC.
A source without a value on any sample leaves the arc to the next source; an arc
that reaches `zero` follows `TECConfig.stec_missing_bias`:

- `warn_zero`: emit `UserWarning` and logger warning (once per run and
  satellite or station, for the first arc that needs it; further arcs of the
  same satellite or station go to the log at INFO level), then use zero.
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

`rcv_dcb_source` chooses the first station bias source; the station bias then
goes on with `stec_bias_sources` (without the chosen one):

- `rcv_dcb_source="calibrate"`: receiver DCB estimated by `Calibration` against
  the GIM (needs `ionosphere_model="gim"`). It uses the satellite biases the
  chain applies (any source of `stec_bias_sources`, not only the GIM DCBs), so
  the receiver DCB is in their datum; elevation cutoff
  `calibration_min_elev` (30 deg).
- `rcv_dcb_source="gim"`: GIM/IONEX receiver DCB of `station_name`, only for
  exactly its own signals and (GPS) tracking modes: an E1/E5a receiver DCB is
  not used on E1/E5b (the receiver E5a-E5b bias is several ns), a P1-P2 one not
  on C1C-C2W. The satellite chain then tries `gim` first (see Datum).
- `rcv_dcb_source="defined"`: `define_station_dcb`.
- `rcv_dcb_source=None`: no preferred source, `stec_bias_sources` order only.
- `rcv_dcb_source="none"`: disable station bias.

When the chosen source gives no value (no calibration, station not in the IONEX
file, `define_station_dcb` not set for the station/system) one warning per run
says so and names the sources that follow. It goes to the log only (INFO) when
the station bias falls to `zero` anyway (the `zero` warning covers it) or when
the calibration was not done for a reason already warned about (calibration
without usable data, GIM DCBs of another signal pair).

Manual station values are in nanoseconds:

```python
define_station_dcb={"BRUX": 2.0, "C": 1.0, "default": 0.0}
```

`station_name=None` takes the RINEX MARKER NAME; a `station_name` whose first
4 characters differ from the MARKER NAME gives a warning.

## Datum

Satellite and receiver biases are estimated together under a zero-mean
condition on the satellite biases, so the two must come from one estimation
(one datum) for their sum to be right. The datum of an arc's satellite biases
is in the column `stec_bias_datum`: `product:<file>`, `ionex:<file>`, `user`
or `none`.

- A product station bias is used only with product satellite biases, the IONEX
  receiver DCB only with the IONEX satellite DCBs. A station source in another
  datum is skipped with one message and the next station source applies.
- When no station source of the satellites' datum has a value, the first
  skipped one is used rather than 0, with one warning: a datum mismatch is
  about 1 TECU (BRUX / NKLG 2024-035, GPS L1/L2: 0.6 / 1.2 TECU), a missing
  receiver DCB 8-22 TECU.
- `rcv_dcb_source="gim"` moves `gim` to the front of the satellite sources when
  the IONEX receiver DCB applies to the arc's pair and `gim` is in
  `stec_bias_sources` (one log line), so the chosen receiver DCB meets its own
  satellite DCBs.
- `define_station_dcb` with product satellite biases gives one warning: the
  value must refer to that product's datum.
- `calibrate` is always in the datum of the satellite biases the chain applies.

## Common Configurations

Satellite and station from products:

```python
TECConfig(
    dcb_path="sample_data/GRG0MGXFIN_20240350000_01D_01D_OSB.BIA",
    add_dcb=True,
    add_sta_dcb=True,
    rcv_dcb_source=None,
    stec_bias_sources=("product", "zero"),
)
```

Satellite only from products:

```python
TECConfig(
    dcb_path="sample_data/GRG0MGXFIN_20240350000_01D_01D_OSB.BIA",
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

- `stec_sat_bias_m`, `stec_station_bias_m`, `stec_total_bias_m` (meters, one
  constant per arc)
- `stec_sat_bias_source`, `stec_station_bias_source`: `product_osb`,
  `product_dsb`, `product_dsb_chain` (joined with `+` when one product mixes
  them in an arc), `gim`, `calibrate` (station), `config`, `zero` or `disabled`
- `stec_sat_bias_coverage`, `stec_station_bias_coverage`: share of the arc's
  samples the chosen source covers
- `stec_bias_datum`: datum of the satellite biases (see Datum)
- `stec_bias_code_1`, `stec_bias_code_2`
- `stec_sat_bias_tecu`, `stec_rcv_bias_tecu`: the arc constants in TECU
- four calibration levels of the levelled STEC (el/m^2): `stec_uncal` (no
  bias), `stec_sat` (satellite bias), `stec_rcv` (receiver bias) and
  `stec_full` (both; equal to `leveled_tec`)
- `leveling_std_tecu`: per arc, the `sin^4 E`-weighted std of the code minus
  the levelled phase in TECU (code noise and multipath; not a formal error)

These columns should be inspected whenever changing product source, signal pair
or constellation.

## BeiDou Notes

BeiDou STEC support is active. `B1IB3I` is the pair of the precise BeiDou
clocks of the 2024 products and the most widely tested one; the modern pairs
`B1CB2a`, `B1CB2b`, `B1CB3I` and `B2aB2b` pass 73 TEC validation rows on the 2024
validation data, broadcast and precise orbits. Modern BDS pairs depend on
available code observations and matching bias products. If a product lacks the selected BDS pair, prefer `stec_missing_bias`
set to `raise` during validation.
