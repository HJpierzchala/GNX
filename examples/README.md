# GNX-py User Examples

This directory contains small, editable Python scripts that demonstrate common
GNX-py workflows on the repository `sample_data`.

The examples are meant to be opened in an IDE and modified. Each file has a
`USER SETTINGS` block near the top where you can change stations, constellations,
signal modes, product paths, time windows and output options.

## Scripts

- `run_spp_example.py`: Single Point Positioning (SPP) with broadcast or precise
  orbit options.
- `run_ppp_example.py`: Precise Point Positioning (PPP) with combined or
  uncombined routing and precise products.
- `run_sise_example.py`: signal-in-space error / SISRE / orbit comparison.
- `run_stec_example.py`: STEC / ionosphere processing with bias policy controls.

## Running

Install GNX-py into the active environment first:

```bash
python -m pip install -e .
```

Then run examples from the repository root:

```bash
python examples/run_spp_example.py
python examples/run_ppp_example.py
python examples/run_sise_example.py
python examples/run_stec_example.py
```

The scripts import the installed `gnx_py` package. They do not modify
`sys.path`; if an import fails, check that the virtual environment used for
installation is also the active interpreter.

The scripts write outputs under `examples/output/`. That directory is intended
for local generated files and should not be committed.

By default, examples write compact CSV files. Fuller diagnostic outputs, such as
PPP residual tails or SISE per-epoch detail rows, are controlled by explicit
flags in each script.

## Choosing Systems and Signals

Common system selections:

- GPS: `"G"` or `{"G"}`
- Galileo: `"E"` or `{"E"}`
- BeiDou: `"C"` or `{"C"}`
- Mixed SPP/PPP: `{"G", "E"}`, `{"G", "C"}`, or `{"G", "E", "C"}`

Common signal modes:

- GPS: `L1L2`
- Galileo: `E1E5a`
- BeiDou: `B1IB3I`

`B1IB3I` is the safest default for current BeiDou examples. Modern BDS modes
such as `B1C`, `B2a` and `B2b` require matching navigation, clock and OSB/DSB
products and should be treated as validation/advanced workflows.

## Smoke Windows

The scripts default to `RUN_SMOKE = True`, which processes a short time window
so users can quickly verify that paths and dependencies are working. Set
`RUN_SMOKE = False` to process the full sample day.

## Product Notes

- SPP defaults to broadcast orbits and Klobuchar ionosphere correction.
- PPP defaults to precise SP3 products, GIM ionosphere support and PPP-AR off.
- SISE/SISRE defaults to BDS legacy `B1IB3I` broadcast-vs-SP3 comparison and
  reports clock-convention metadata.
- STEC defaults to one constellation at a time. Change `SYSTEM` to `G`, `E` or
  `C`; keep the bias policy visible when changing products.

## Advanced / Experimental Options

- PPP-AR is disabled by default. Enable it only when you are validating AR
  behavior and understand the selected filter path.
- Combined ionosphere-free PPP-AR is additionally gated and should remain off
  unless explicitly validating that branch.
- BeiDou GEO is not supported in the SIS/SISRE comparison path.
- Modern BDS SIS clock comparisons can show large clock errors when the
  broadcast and precise clock datums do not match. Inspect
  `clock_convention_status` before interpreting BDS clock metrics.
