# GNX-py PPP

The `gnx_py.ppp` package implements Precise Point Positioning workflows for
research, validation, and engineering experiments with high-precision GNSS
data. It combines RINEX observation loading, orbit/clock interpolation,
antenna and propagation corrections, cycle-slip preprocessing, extended Kalman
filter PPP models, diagnostics, and optional ambiguity-resolution logic.

The module is intentionally broad: it supports both mature reference paths and
newer multi-GNSS development paths. Every PPP filter class is selected by the
session for some configuration (Developer Notes); the classes it never selected
were removed in v1.0.0.

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

## Labels Used in the Measured Values

The measured values in this document come from the v1.0.0 validation: public
data of the stations BRUX, NKLG and NYA2 on 2024-035 (quiet), 2024-131 and
2024-132 (storm), with CODE (COD) and CNES/CLS (GRG) precise products. A
configuration of that validation is called a row; its label combines the
systems (G GPS, E Galileo, C BeiDou), the orbit product (`SP3cod`, `SP3grg`,
broadcast), static or kinematic processing (`kin`) and the observation model:

- `IF`: the combined ionosphere-free filters; `GRAPHIC`: the
  single-frequency code/phase average `(P + L) / 2` (Single-Frequency PPP).
- `UC0` / `UCi`: uncombined PPP without / with ionospheric constraints.
- Filter routes of the session for uncombined PPP: R1 = two or more systems
  without ionospheric constraints (`PPPUdGenericMixedGNSS`); R2 = with
  constraints, two or more systems with BeiDou, the `"gec"` model or
  broadcast orbits (`PPPFilterMultiGNSSIonConstGEC`); R4c = with
  constraints, GPS + Galileo, precise orbits, the `"legacy"` constraints
  model (`PPPFilterMultiGNSSIonConst`); R7 = one system
  (`PPPUdSingleGNSS`). `R7-G` is a GPS-only R7 run.
- `t_c10` / `t_c20`: minutes from the start until the 3D error against the
  reference coordinates stays below 10 / 20 cm for 20 consecutive epochs;
  `err3d_final`: median 3D error of the last 30 minutes; `rms3d_post`: RMS of
  the 3D error from 30 minutes after `t_c10`; p95: 95th percentile.
- "Relation" compares a constrained row with its unconstrained (UC0) twin.

## Main Entry Points

Most users should start with:

- `gnx_py.PPPConfig` or `gnx_py.ppp.PPPConfig`: main configuration object.
- `gnx_py.PPPSession` or `gnx_py.ppp.PPPSession`: in-package PPP session
  orchestrator.

`examples/run_ppp_example.py` shows a complete run on local files (an
illustration, not a test reference).

## Results

`PPPSession.run()` returns a `PPPResult` (an output change of v1.0.0):

- `solution`: every route, an empty solution included, has the columns `x`,
  `y`, `z` [m, ECEF] and `ct_min` [min, NaN when the position did not
  converge]; before v1.0.0 `ct_min` held hours, the single-system IF and R4c
  routes had none and R4c named the position `xr`, `yr`, `zr`. An empty
  solution is a frame with these columns and no rows.
  `solution.attrs["precise_clock_datum"]` names the datum of the precise
  BeiDou clocks the code biases used (BeiDou Notes). `de` / `dn` / `du` are
  the reference minus the estimate in east / north / up at the reference: the
  SOLUTION/ESTIMATE coordinates of the station in `sinex_path`, the solution
  valid at the first epoch (`gnx_py.io.parse_sinex`). A station that is not
  in the SINEX gives NaN with one UserWarning naming the station and the file
  (they were 0.0, which reads like an error-free solution), and so does a run
  without `sinex_path` (SPP without `sinex_path` has no de/dn/du columns).
  `to_netcdf` takes a str path.
- `residuals_gps`, `residuals_gal`, `residuals_bds`: the residual frames of
  the systems the filter processed, routed by system letter (every filter
  returns `{system: frame}`; a processed system without rows gets an empty
  frame with the usual columns; a system not processed keeps None). Before
  v1.0.0 a Galileo-only IF run put its residuals in `residuals_gps` and a GPS-only R7
  run filled `residuals_gal` with a copy of the GPS frame.
- Residual columns: on the IF, GRAPHIC and R4c routes the prefit
  (innovation) columns (`prefit_*`, GRAPHIC `res`, R4c `vc1`, `vl1`, `vc2`,
  `vl2`) are taken after the epoch's code-derived receiver clock prior, so they
  are metre-level instead of carrying the full receiver clock offset (IF-G
  BRUX 1 h: median |prefit code residual| 60.6 -> 0.35 m; R4c `vc1` max 58.7 ->
  1.9 m; `gnx_py.viz.ppp` plots of `vc1` change accordingly). The uncombined
  R1, R2, R7 and single-frequency constrained routes keep `z - h(previous
  state)`, which carries the previous epoch's clock. Postfit residuals are
  unchanged.
- `convergence`: the convergence time in hours (None when not converged);
  `solution["ct_min"]` gives it in minutes.

Filter core (`GNXExtendedKalmanFilter`): observations
the filters reject (variance 1e12 m², non-finite variance) are left out of the
Kalman update instead of being weighted with 1e12 m² (RTKLIB `res_ppp`): the
innovation `y` stays full length, the gain has zero columns for the dropped
rows, `S`, `mahalanobis` and `log_likelihood` describe the rows used
(`rows_used`); cond(S) on IF-GE drops from 2.5e15 to 9.3e8. The receiver clock
(and an inter-system bias that starts from `p_dt`) is re-initialised at every
epoch from the code with the variance `clock_prior_sigma_m`² (100 m, RTKLIB
`VAR_CLK`) when its predicted variance exceeds that (`state_priors`,
`priors_applied`): on 149 validation rows the metrics move by at most 1.1 mm
and one epoch of t_c10.

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

The multi-system filters, the combined IF one and the mixed uncombined one
alike, process only the epochs common to all active systems: an epoch in
which one system has no observation left is skipped as a whole, not
processed with the other systems (a route property).

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

### Single-Frequency PPP

A single-frequency mode (`gps_freq="L1"`, `gal_freq="E1"`, `bds_freq="B1I"`,
...) with one system routes to:

- `use_iono_constr=False`: `PPPSingleFreqSingleGNSS`, the code/phase average
  GRAPHIC, `(P + L) / 2`, free of the first-order ionosphere (code delay and
  phase advance cancel), with a float ambiguity per arc. It needs no
  ionosphere model (`ionosphere_model=False` works; an `ion` column is not
  used). The noise is half the code noise, so it converges in hours.
- `use_iono_constr=True`: `PPPUducSFSingleGNSS`, code and phase with a slant
  ionosphere state per satellite constrained by the model (GIM, Klobuchar,
  NTCM, BDGIM). The code is weighted with c_sys / sin E from
  `uncombined_iono_code_var_by_system` (G 0.3, E 0.3, C 1.0 m²) since v1.0.0 (before, 1.0 m² for every system, which `{"G": 1.0,
  "E": 1.0, "C": 1.0}` gives back): BRUX 2024-035 6 h static GIM final 3D G L1
  6.7 -> 4.1 cm, E1 4.8 -> 3.0 cm, NKLG G L1 t_c20 93 -> 53.5 min, BRUX
  2024-132 E1 t_c20 201.5 -> 71 min; NYA2 gets worse (G L1 10.2 -> 18.0 cm, E1
  7.1 -> 11.6 cm, limit 25 cm): the station's code error that rises with
  elevation (JAVRINGANT_G5T, no group-delay-variation calibration) enters with
  the stronger code weight. With one frequency the GIM error (1-3 TECU,
  correlated in time and with elevation) cannot be separated from the position
  and, kept at full weight, it stayed in the solution. The route default
  `sigma_iono_end` 1e4 makes the GIM a prior on the
  satellites in view at the start: the ramp is linear in the variance (1.1 m² at the first epoch, about
  170 m² at the second with 30 s data), so satellites that rise later and new
  arcs get practically no GIM information and the solution then rests on the
  code/phase information, as GRAPHIC. A GIM prior for every new arc (the GIM
  start value and the same ramp from the arc's own start,
  `PPPUducSFSingleGNSS.SF_GIM_PRIOR_NEW_ARCS`) is off: on the
  12 static cases the median improves (final L1 / E1 / B1I 0.090 → 0.083,
  0.105 → 0.100, 0.071 → 0.068 m; first 15 min 1.52 → 1.32, 1.84 → 1.84,
  1.60 → 1.48 m), but every NYA2 final gets worse (polar GIM; worst E1 0.129
  → 0.182 m, B1I 0.201 → 0.254 m) and the kinematic L1 / E1 runs too (median
  RMS after 1 h 0.270 → 0.364, 0.329 → 0.384 m; B1I 0.395 → 0.264 m). The 1e4 route is better than GRAPHIC in the first
  15 min in some cases (3-5 of 12 per signal; BRUX 2024-035 E1 RMS 0.38
  against 1.23 m), not on the median (L1/E1/B1I 1.87/1.84/1.60 against
  1.33/0.92/1.36 m), and worse after 1 h (table below).

Single-frequency PPP, measured during the v1.0.0 development (before the later
weighting changes), precise products (GRG), 30 s, 10° mask, float; 3D error,
median over 12 cases (BRUX/NKLG/NYA2, 2024-035 and 2024-132, 6 h windows from
00:00 and 12:00), worst case in brackets:

| Route | after 1 h | after 6 h | former code, 6 h |
|---|---|---|---|
| GRAPHIC L1 | 0.19 m | 0.08 m (0.26) | 0.98 m (4.8) |
| GRAPHIC E1 | 0.15 m | 0.10 m (0.19) | 1.10 m (3.1) |
| GRAPHIC B1I | 0.17 m | 0.08 m (0.27) | 0.87 m (4.5) |
| GIM-constrained L1 | 0.25 m | 0.09 m (0.27) | 0.17 m (0.59) |
| GIM-constrained E1 | 0.42 m | 0.11 m (0.13) | 0.27 m (0.44) |
| GIM-constrained B1I | 0.42 m | 0.08 m (0.20) | 0.19 m (0.56) |

Kinematic (BRUX/NKLG/NYA2 2024-035, RMS 3D after the first hour, median):
GRAPHIC L1 / E1 / B1I 0.31 / 0.23 / 0.44 m (before: 2.7 / 2.6 / 2.4 m);
GIM-constrained 0.27 / 0.32 / 0.40 m (before: 0.64 / 0.57 / 1.05 m).
Literature: GRAPHIC static solutions reach the decimetre level after a few
hours with precise products (e.g. Cai et al. 2013, J. Navigation, GPS+GLONASS: 6 / 3 / 12 cm
E/N/U). The former model of `PPPSingleFreqSingleGNSS` subtracted the model
ionosphere from code and phase and weighted the phase at 3 cm, so the model
error moved the position by decimetres to metres, as does RTKLIB's
single-frequency PPP with `ionoopt=ionex-tec` on the same data (BRUX L1 6 h:
0.42 m, 0.93 m after 1 h).

Undetected small cycle slips (single-frequency detection is code-noise
limited, see Cycle Slips) move the GRAPHIC observation by half a wavelength
per cycle (9.5 cm on L1): they are not caught by its prefit gate (3 m or
5 sigma) and are diluted over the arc.

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
  for ionospheric constraints. `sigma_iono_end=None` (default) is the route's
  default: 1e4 for single-frequency PPP with constraints
  (`PPPUducSFSingleGNSS`: in practice a GIM prior on the satellites in view
  at the start, see Single-Frequency PPP), 3.0 for the other constrained
  filters. The legacy G+E constrained filter (`PPPFilterMultiGNSSIonConst`)
  took 2.3 before v1.0.0 (the 2.3 tuning compensated an error of the phase
  Jacobian of that filter, fixed in v1.0.0). Measured during the v1.0.0 development,
  GRG, float, 2.3 → 3.0, with the v1.0.0 GIM handling, t_c10 / final 3D /
  RMS 0–30 min: BRUX 6 h 46.0 → 41.0 min / 2.37 → 2.34 cm / 0.34 → 0.33 m,
  NKLG 96.5 → 93.0 min / 2.28 → 2.22 cm / 1.05 → 0.99 m, NYA2 91.0 →
  82.5 min / 3.95 → 3.96 cm / 0.85 → 0.84 m, BRUX storm day 62.5 →
  53.0 min / 4.31 → 4.26 cm / 0.50 → 0.48 m, BRUX 24 h 46.0 → 41.0 min /
  2.44 → 2.47 cm; BRUX kinematic RMS / p95 12.30 / 19.07 → 12.56 / 19.12 cm,
  the one row where 3.0 is worse (with the former GIM handling: static the
  same pattern, kinematic 12.69 / 19.27 → 12.03 / 18.45 cm).
  The ramp values are variances [m²] of the ionospheric pseudo-observation,
  used as given (`sigma_iono_0` / `sigma_iono_end` are not squared);
  `t_end` is the length of the ramp in minutes, counted from the first epoch
  (a filter reset does not restart it); `reset_every` is in minutes too
  and `ct_min` counts from the latest reset.
  Since v1.0.0 `use_iono_rms=True` adds the squared GIM (or
  modelled) RMS to the ramp (constraint variance = ramp + ion_rms²; it
  multiplied the ramp, metres times m²): C COD static UCi
  t_c10 224.5 -> 38.5 min and G GRG kinematic UCi t_c20 134.5 -> 10 min with
  `use_iono_rms=True`, equal to the `use_iono_rms=False` runs within 0.5 min
  and 0.2 cm.
- GIM constraint datum (a defect fixed in v1.0.0): the
  slant ionosphere state of a satellite in an uncombined filter with product
  code biases is I + beta (D_product - D_true), beta = f2² / (f1² - f2²), D the
  satellite DSB of the pair: the state is in the DSB datum of the bias product,
  while a GIM gives I in the datum of its own satellite DCBs. The
  dual-frequency constrained routes now refer the model ionosphere (`ion`) to
  the code-bias datum of the bias product: the IONEX satellite DCBs of the same
  two signals are the reference (GPS C1W-C2W, Galileo E1 / E5a in the CODE
  GIMs), else a DSB product of the pair in `dcb_path`; satellites whose DSB
  differs by 1 ns or more from the median difference are corrected
  (`ppp_uduc._IONO_DSB_ALIGN_MIN_NS`; two DSB estimates of one satellite agree
  to 0.3-0.5 ns), the summary per system is the filter's
  `iono_dsb_alignment`. BRUX 2024-035 with GRG OSB: GPS III 1.0-1.6 m on L1,
  BeiDou-2 IGSO / BeiDou-3 MEO up to 1.6 m on B1I (CAS DSB), Galileo E19 0.5
  m; G GRG static UCi t_c10 41 -> 15.5 min, E GRG 23 -> 13.5 min, C GRG
  kinematic UCi RMS 16.25 -> 13.7 cm (with the CAS DSB), GEC kinematic UCi
  relation to UC0 5.20 -> 4.29 cm. The COD and IGS GIMs give no BeiDou DCBs:
  add a DSB product (CAS) to `dcb_path`, else one UserWarning per run and
  system says the constraint is not referred. With the CAS DSB the
  BeiDou-constrained GRG rows converge about twice as fast (C / EC / GC / GEC
  static UCi t_c10 47 / 36.5 / 35.5 / 29 -> 22.5 / 14 / 15 / 13 min) while
  their final error grows a little: 20 of the 26 final values moved by it get
  worse, median x1.14 (GEC dual-frequency rows 9.5 -> 14.0 mm, the legacy
  weighting row pw_r7iono-C 3.8 -> 6.7 cm, known limitation LIM-PPP-04 in
  `CHANGELOG.md`).
  Single-frequency PPP (the GIM prior of `PPPUducSFSingleGNSS` and the
  single-frequency modes of the mixed constrained routes) and single-frequency
  SPP take the same rule (only the IONEX DCBs as reference: GPS L1 / L2,
  Galileo E1 / E5a). On the validation data 112 reference values move (89 SPP,
  23 PPP), none worse by more than 5 %: GPS L1
  SPP BRUX 2024-035 rms 3D 1.444 -> 0.955 m, L2 2.259 -> 1.354 m;
  single-frequency PPP E1 kinematic p95 67.9 -> 48.7 cm, N05 Galileo E1 /
  E5a 15.4 / 20.5 -> 7.9 / 6.9 cm, the static rows within 3 min and 0.3 cm.
  Known cost: the Galileo-only kinematic GRG row moves 9.6 -> 16.2 cm (p95
  14.6 -> 31.2 cm, t_c20 10.5 -> 114.5 min), because GRG's E19 DSB is 1.2-1.5
  ns off three independent sources and that one satellite's first-hour
  constraint decides the E-only kinematic solution, whose unconstrained twin
  itself gives 12.4 cm; a looser constraint (x3: 13.4 cm) or one released after
  the start (`sigma_iono_end` 1e4: 11.6 cm) passes (known limitation
  LIM-PPP-03 in `CHANGELOG.md`; the constraint sigma stays).
- `uncombined_iono_code_var_by_system` (default
  `{"G": 0.3, "E": 0.3, "C": 1.0}` m² at the zenith) and
  `uncombined_iono_legacy_weighting` (default `False` since v1.0.0): the
  constrained routes (R7, R2) weight the code with c_sys / sin E and the
  phase with `uncombined_phase_var` / sin E. The legacy weighting (opt-in)
  takes the `uncombined_iono_*_var_*` settings, 0.3 + 0.0025 / sin E m² for the
  code, practically independent of the elevation and 3.3-18 times stronger than
  the unconstrained routes' 1 / sin E: with the slant ionosphere pinned by the
  GIM every code acts on the geometry, so low-elevation code errors and product
  code-datum errors (COD C06 +4.9 m) biased the float ambiguities. BRUX
  2024-035 6 h: C / EC / GC / GEC COD static UCi t_c10 157.5 / 130 / 65.5 /
  48.5 -> 38.5 / 29 / 20 / 18 min (C err3d 5.20 -> 0.74 cm), G GRG kinematic
  UCi t_c20 103.5 -> 10.5 min (RMS 10.10 -> 8.64 cm), GE GRG kinematic RMS
  8.87 -> 5.01 cm; all 9 daily UCi float rows converge faster or end closer
  (NKLG R7-G t_c10 98.5 -> 51.5 min); slower but within their limits: E GRG
  static UCi 13 -> 23 min, G COD static UCi 17.5 -> 24 min, C GRG kinematic UCi
  RMS 15.18 -> 16.25 cm (above the original 15 cm target, within the limit
  referred to RTKLIB on the same data). With PPP-AR the 24 h R7-G rows fix
  fewer epochs than with the former weighting (with the v1.0.0 AR changes;
  BRUX 2024-035 / 132, NKLG 035 / 132, NYA2 035:
  2235 / 2131 / 425 / 469 / 1643 -> 1741 / 1424 / 510 / 290 / 1836 fixed
  epochs after warm-up, -16 % in total) while the final 3D error improves on
  all five days (5.8 / 7.6 / 12.4 / 21.4 / 31.6 -> 4.5 / 5.7 / 10.8 / 16.5 /
  15.4 mm); `uncombined_iono_legacy_weighting=True` keeps the former counts
  (2240 / 2135 / 426 / 471 / 1641). One value cannot
  serve every system: 1.0 for all slows Galileo (E GRG static 13 -> 44 min),
  0.3 for all leaves BeiDou slow. With the legacy weighting on, the new
  setting is named in the "does not use" warning; with it off,
  `uncombined_code_var` and the `uncombined_iono_*_var_*` settings are.
- `gim_mapping_function` (`"auto"`, `"slm"`, `"mslm"`) and
  `gim_time_interpolation` (`"rotated"`, `"earth_fixed"`): how the GIM
  constraint becomes a slant delay (see `gnx_py/ionosphere/README.md`). The
  defaults since v1.0.0 (the CODE MSLM by header, rotated maps) fit the
  levelled STEC better (COD 10-15° bin −3.30 → 0.00
  TECU, k_ep 0.879 → 0.973; IGS shape 4.726 → 4.631 TECU with rotation), but
  cost convergence with a GIM constraint, measured against the former GIM
  handling (SLM, earth-fixed maps; GRG, float, 2024-035) with the former
  constrained weighting:
  - rotation, BRUX: GC static UCi t_c10 50.5 → 80 min, G kinematic UCi t_c20
    81 → 103 min, C kinematic UCi RMS 0.148 → 0.157 m (the GC float/fixed, G
    and C rows miss their limits), C static UCi t_c10 51 → 91.5 min
    (within its 120 min limit), R4c kinematic RMS 12.03 → 12.70 cm with
    rotation alone (12.56 cm with both defaults);
  - MSLM, NKLG: G UCi t_c10 48.5 → 113.5 min;
  - both together, NKLG: R4c GE 24 h t_c10 48 → 93 min (49.5 / 49 min with
    either one alone);
  - MSLM, BRUX Galileo-only E UCi kinematic (fails either way):
    t_c20 321.5 → 13 min, but the kinematic statistics then start at 13 min,
    so RMS / p95 0.147 / 0.191 → 0.294 / 0.394 m;
  - final positions change by at most 9 mm.

  This is not a GIM-handling artefact: with the corrected constrained
  weighting of v1.0.0 the four
  mapping / interpolation combinations move the static UCi rows by at most
  6.5 min, rotation equal or faster on the four COD rows; the slowness above
  was the weighting. Opt-outs: `gim_time_interpolation="earth_fixed"` and
  `gim_mapping_function="slm"`.
- `bds_klobuchar` (`"gps"` default, `"icd"`): BeiDou Klobuchar with
  `ionosphere_model="klobuchar"`. `"gps"` takes the GPS coefficients scaled to
  the BeiDou signal; `"icd"` (opt-in) the BDS-SIS-ICD-B1I model with BDSA/BDSB,
  falling back to `"gps"` with a warning when the header has none. The BDS-2
  coefficients are regional (Asia-Pacific): slant RMS against the COD GIM at
  B1I over 9 station-days 17.95 TECU GPS-scaled against 19.58 TECU, BDS worse
  on 5 of 9. Since v1.0.0 `"icd"` takes one GEO-first set (GEO, then
  IGSO, then MEO records; implausible sets such as all-equal filler counts or
  A2 <= 0 rejected with a warning) and, with a RINEX 4 file, the latest
  plausible '> ION' record at or before each epoch. The BDS-3 global model is
  a separate setting, `ionosphere_model="bdgim"` (below).
- `ionosphere_model="bdgim"`: BDGIM, the BeiDou-3 global
  broadcast model of BDS-SIS-ICD-B2b-1.0 §7.7, for G, E and C, as the
  ionosphere constraint of the constrained routes (`use_iono_constr=True`,
  single and dual frequency), like `"klobuchar"` and `"ntcm"`. It needs a RINEX 4 `nav_path` with
  '> ION Cxx CNVX' records (e.g. BRD400DLR, which also serves the broadcast
  ephemerides; with SP3 orbits it is read for its '> ION' records); a file
  without a usable BDGIM set raises a ValueError naming the file and its RINEX
  version when the run starts. With `use_iono_rms=True` the constraint sigma
  is 40.3e16 / f^2 x hypot(0.5 |STEC|, 5 TECU x MF) from the unclamped model
  (5 TECU at the zenith is 0.81 m on L1), so a model value clamped at 0 never
  makes the constraint rigid. Since v1.0.0 Klobuchar and NTCM take the
  same rule: hypot(50 % of the delay, 5 TECU x the model's mapping function;
  GPS Klobuchar its obliquity factor, BeiDou Klobuchar with
  `bds_klobuchar='icd'` its own 375 km shell, NTCM its MSLM;
  `DDPreprocessing.BROADCAST_SIGMA_FLOOR_TECU`). With 50 % alone the constraint
  at night (Klobuchar is the coefficient-free 5 ns term there) was so tight
  that the constrained rows converged slower than without it: R7-G BRUX t_c10
  50.5 -> 46.0 min (UC0 41.5), R4c-GE relation 40 -> 29 min (UC0 18). Set
  selection, screens and conventions are in `gnx_py/ionosphere/README.md`
  (BDGIM); the run report is `session.run_notes.iono_report` after `run()`.
  Validation rows (SP3-GRG + BRD400DLR, 2024-035, float, BRUX / NKLG), with
  the former constrained weighting: uncombined constrained R7 G t_c10 49 /
  90.5 min with BDGIM against 79.5 / 117.5 min with Klobuchar, R4c GE 48 / 48
  against 73.5 / 108 min; with the v1.0.0 weighting and the additive RMS
  R7 G 41 / 51.5 against 50.5 / 44 min, R4c GE 21.5 / 35.5 against
  40 / 48.5 min (all 17 rows pass).
- `q_tro`, `q_iono`, `q_isb`, `q_dcb`, `q_amb`: random-walk process noise in
  m²/s, multiplied by the epoch interval (`q_tro` default 1e-8 m²/s, the
  RTKLIB value, 6 mm/√h; before v1.0.0 it was a variance per epoch,
  2.5e-4 m² combined and 0.025 m² uncombined). After a data gap the filters
  add `q * dt` with the real time since the previous epoch. `q_iono`
  (10 m²/h) is an empirical value. `q_crd` and `q_dt` are variances per epoch
  (0 = static coordinates; 1e2 = kinematic). `q_amb` acts on every ambiguity
  state of every route, initial, rebuilt and restarted ones included, since v1.0.0 (R7 and R4c ignored it, the single-system IF
  filter applied it to rising satellites only).
- `clock_prior_sigma_m` (100 m, RTKLIB `VAR_CLK`; read since v1.0.0): a receiver clock (or an inter-system bias starting from `p_dt`) whose
  predicted variance exceeds its square is re-initialised from the code at
  every epoch, so `p_dt` and `q_dt` above 1e4 m² have no further effect.
- `pppar_enabled`: enables PPP-AR attempts where supported.
- `pppar_allow_mixed_products`: forces PPP-AR when the phase OSBs come from
  another solution than the orbits and clocks (off by default: AR disabled).
- `pppar_combined_if_ar_enabled`: additional gate for combined
  ionosphere-free AR, which fixes no ambiguity (see PPP-AR Status).
- `trace_filter`: enables compact `[PPP TRACE]` diagnostics.
- `satellite_attitude`, `attitude_laws`, `attitude_path`, `attitude_turn_guard`,
  `attitude_allow_mixed_products`: satellite attitude at noon/midnight turns
  (see Satellite Attitude below).

See the `PPPConfig` docstring in `gnx_py/core/configuration.py` for field-by-field
details.

## Input Files and Session Checks

Since v1.0.0 GNX reads its input files with its own readers, in
memory, with nothing written next to the data (read-only directories and
parallel reads of one file are safe):

- Observations: `gnx_py.io.rinex_obs` reads RINEX 2.10/2.11, 3.0x and 4.0x, plain
  or compressed (gzip, Unix compress `.Z`, bzip2, zip, Hatanaka CRINEX 1/3);
  `time_limit` is always applied (inclusive, GPST), BeiDou time tags are moved
  to GPST (BDT = GPST - 14 s), Galileo time tags are taken as GPST (the GST -
  GPST offset, a few nanoseconds, is not applied), blank and 0.000
  values are missing, LLI is read for every version.
  `use_gfz` is accepted and has no effect (gfzrnx is no longer run).
- `SYS / PHASE SHIFT`: the phases are used as
  written. RINEX 3.01+ phases are already aligned and the record reports the
  correction applied (RINEX 3.03 section 9.1, Table 25); before v1.0.0 PPP and TEC
  added the header values a second time to every satellite of the system
  (files with non-zero values such as WUH2, SAVU, USN9 change, PPP-AR
  with OSB phase biases most). A record inside the data (event-flag block) that
  changes the correction in force for a processed phase is reported with one
  warning per run and system; the arc is kept, as in RTKLIB, which ignores the
  record: the phases stay aligned on both sides of it (WUH200CHN 2024-141:
  geometry-free step 0.2 cm median at 28 measurable changes, against 6.2-6.4 cm
  for a quarter cycle), and a real step is left to the slip detectors. The
  reader keeps the satellite lists and per-satellite values
  (`rinex_obs.phase_shift_of` / `phase_shift_events_of`).
- LLI bit 1 (half-cycle ambiguity possible) gives the column `lli_half_cycle`
  of the mode's phases (PPP-AR and Cycle Slips).
- Single-frequency GPS (L1, L2, L5) takes the code of the phase's tracking,
  e.g. C1C with L1C instead of the semi-codeless C1W: NYA2 2024-035 L1 0.261 -> 0.217 m, BRUX / NKLG +-3 mm. SPP, TEC
  and dual-frequency PPP keep their selection.
- One column per observable kind and band: the reader keeps one column per
  observation type, and the session passes the filters one code, one phase
  and one signal-strength column per band of the mode (e.g. GPS L1L2 at BRUX:
  `C1W`, `C2W`, `L1C`, `L2W`), so a filter that looks a column up by its
  two-character prefix (`C1`, `L2`, ...) finds exactly one, and taking the
  first or the last match is the same.
- Navigation: `gnx_py.io.rinex_nav` reads RINEX 2 (GPS), 3 and 4 (GPS LNAV with
  the CNAV group delays, Galileo I/NAV and F/NAV, BeiDou D1/D2 and B-CNAV) and
  the ionosphere coefficients of the header and of RINEX 4 '> ION' records. A
  file that cannot give a requested system raises a ValueError naming the
  file, its version and its systems.
- SP3 (`gnx_py.io.sp3_reader`), RINEX clock (`gnx_py.io.rinex_clk`) and every other
  product reader accept the same compression formats.
- Header: `ANT # / TYPE` is read by columns (a type-only line is valid; a blank
  type warns and leaves the receiver antenna unknown); `ANTENNA: DELTA H/E/N`
  missing gives zero offsets with a warning.
- Phases: a file without a phase observable with data for a signal of the mode
  (a code-only file, which SPP processes) raises a ValueError naming the
  system, the mode, the missing phases and the file (it ended in "Lack of dual
  frequency data!").
- A-priori position: `approx_position` (ECEF metres) overrides everything;
  otherwise the header position is used when it is plausible and within
  max(10 m, 3·sqrt(p_crd), 4·spread) of a position from the code observations
  (least squares from the geocentre on up to 20 epochs, antenna offset
  removed; 100 m instead of 10 m for single-frequency modes, the close-pair
  modes L2L5/E5aE5b/B2aB2b (ionosphere-free noise gain 16.6-27.5), broadcast
  orbits and kinematic runs, whose code positions carry metre-level biases);
  otherwise the code position, with a warning (no APPROX POSITION XYZ no
  longer means a station at the geocentre).
- SP3 coverage is checked per system before the run: a system the SP3 files
  do not cover is dropped with a warning, or the run raises a ValueError naming
  the SP3 files, their spans, the observation span and `time_limit`.
- A satellite-epoch without an SP3 clock (outside the clock nodes, across a
  missing node that cannot be checked, or inside an interval that fails the
  clock-continuity test) gets no clock and is left out of that epoch, with one
  warning naming the span; no NaN reaches the solved position (SPP and PPP
  share this step).
- A GIM that covers only part of the session (maps ending before the
  observations, a session across midnight with one IONEX): the uncovered
  epochs get a NaN `ion` with one RuntimeWarning naming the span
  (`tools.check_gim_coverage`), and the GIM-constrained routes drop them, so
  the solution ends at the last map; the IF routes are unaffected; a GIM of
  another day raises a ValueError. This is RTKLIB's rule as well (`ppp.c`
  `corrmeas`: outside the map period the ionosphere correction fails and the
  measurement is not used).
- A system with no observation left after preprocessing is dropped with a
  warning naming the reason; with none left the run raises "No GPS, Galileo or
  BeiDou observations after preprocessing: ..." with the reason per system.
- `screen=True` applies the broadcast health in force (the latest record of
  each message type at or before the epoch) and, for Galileo, the DVS/SHS
  flags of the mode's own signals (E1 -> E1-B, E5a -> E5a, E5b -> E5b). E14
  and E18 (out of service in every 2024 record) leave screened runs.
- `clk_path`: 30 s RINEX clocks of the SP3's analysis centre replace the SP3
  clocks in the filter (not only in the slip detector); each file is its own
  series, interpolation only between nodes at most 1.5 intervals apart, rows
  the files do not cover keep the SP3 clock (`clk_src` column). Clock files of
  another centre, or unreadable ones, are ignored with a warning. Kit example
  (COD IF-G, BRUX 2024-035): kinematic full-day RMS 6.05 -> 2.89 cm, static
  2 h t_c10 25 -> 7.5 min. Runs without `clk_path` do not change. Rows inside
  a gap of the clock files (between two CLK nodes more than 1.5 intervals
  apart, inside the files' span; column `clk_gap`) are left out like an
  ordinary data gap with the default `clk_gap_policy="drop"` (RTKLIB `pephclk`); `"sp3"` keeps them with the SP3 clock (the behaviour
  before; BRUX 2024-132 drops one C10 row, metrics equal to 0.1 mm).
- RINEX 2 observations with PPP-AR: AR is switched off with a warning unless
  the phase tracking is known (`rinex2_tracking` or the converter's header
  block): an L2C-tracked L2 phase can be a quarter cycle off L2W.

## Satellite Attitude

Since v1.0.0 one attitude provider (`gnx_py.models.attitude`, called by
`tools.DDPreprocessing._attitude`) gives the satellite body axes of every
satellite-epoch; the same frame serves the satellite PCO, the azimuth of the
satellite PCV (Galileo entries with an azimuth grid) and the phase wind-up.

- `satellite_attitude=None` / `"model"` (default): the ORBEX attitude of
  `attitude_path` (`gnx_py.io.orbex`) where it covers a satellite's whole arc,
  else the published attitude law of the ANTEX block: GPS IIR catch-up
  (Kouba 2009), GPS IIF noon catch-up and midnight shadow turn, GPS III
  (Montenbruck et al. 2026), Galileo IOV/FOC (GSC metadata), BeiDou-3 CAST
  (Wang et al. 2018) and SECM (CSNO with the Yang et al. 2023 sign latch),
  BeiDou-2 continuous yaw (Dilssner 2017); BeiDou GEO orbit-normal.
  `attitude_laws` overrides a family, e.g. `{"gps_iii": "iir"}` (what CODE
  flies) or `{"bds3_secm": "dilssner2017"}` (COD's ORBEX of 2024-035);
  `gnx_py.models.attitude.LAWS` lists them. BeiDou-2 IGSO/MEO stays yaw steering by
  default (`bds2_ys_on: "nominal"`): GRG0MGXFIN flies
  C11/C12 in yaw steering (ORBEX within 0.012 deg), COD0MGXFIN orbit-normal
  (C12 all day, C11 from 07:54 on 2024-035); the opt-in `"ys_on"` (Li et al.
  2018) matches COD's C12 but misses COD's C11 switch by about 5 h. With COD
  products and no ORBEX the default is wrong by up to 177 deg on half of each
  orbit of C11/C12 (0.44 cycle of wind-up, up to 0.22 m of PCO on the line of
  sight); give COD's ORBEX file for such runs.
- `"nominal"`: yaw steering everywhere and BeiDou GEO orbit-normal, the
  attitude of RTKLIB 2.4.2 (`satantoff`, `windupcorr`) and of earlier GNX versions;
  no ORBEX is read (a UserWarning when `attitude_path` is set) and no row is
  guarded. Use it, with `attitude_turn_guard="off"`, for like-for-like
  comparisons with RTKLIB (the validation does so for its RTKLIB rows).

ORBEX (`attitude_path`, `*_ATT.OBX[.gz]`, ORBEX 0.09, TIME_SYSTEM GPS,
FRAME_TYPE ECEF): quaternions with the scalar first, body axes = rows of the
rotation matrix (IGS convention; checked against the orbit: z within 1 deg of
nadir, x on the Sun side outside turns, per satellite), slerp between records
at most two intervals apart and rotated by at most 30 deg. An arc not fully
covered takes the law (one UserWarning). A file of another analysis centre than
`sp3_path` is not used (UserWarning naming both; `attitude_allow_mixed_products`
overrides): the clocks of a centre hold the wind-up of its own attitude.

Turn guard (`attitude_turn_guard`, PPP only). Analysis centres fly some turns
the other way round from the published laws (one wind-up cycle, about 10.7 cm
of the ionosphere-free phase, plus the horizontal PCO), and their clocks
absorb it. Measured on the validation data (clock steps across 388 turns of
2024-034/035/036/130-133, the ORBEX files of 2024-035): GRG reverses every
BeiDou-3 SECM turn at |beta| < 3 deg (CSNO law with the beta sign reversed);
COD's clocks show one-cycle steps at Galileo turns with 0 <= beta <= 0.3 deg
that COD's own ORBEX (which follows the GSC laws within 0.4 deg) does not
explain. The guard drops a satellite's rows within 35 min of a noon/midnight
passage when |beta| there is below 0.5 deg (GPS, Galileo), 3 deg (BeiDou-3) or
4 deg (BeiDou-2), never GEO; the next row opens a new arc.

- `"auto"` (default): static PPP (effective `q_crd == 0`)
  with precise orbits; only the classes where the product's centre is not known
  to agree with the laws (`gnx_py.models.attitude.AC_GUARD_EXEMPT`: GRG GPS and
  Galileo, plus BeiDou-2 while its law is `"nominal"`; COD BeiDou-3 SECM; GFZ
  GPS, Galileo and BeiDou-3, its CAST entries unverified on the validation data and taken
  from the literature); rows with the ORBEX attitude of the product's own
  centre are not guarded, except COD Galileo (`ORBEX_GUARDED`). Both tables
  rest on the 2024 products; a centre that changes its convention later is
  exempted silently until they are re-checked. Since v1.0.0
  dual-frequency kinematic PPP is guarded as static PPP: GRG flies the
  BeiDou-3 SECM turns the other way round and its clocks hold the one-cycle
  wind-up step (about 10.6 cm of the B1I/B3I ionosphere-free phase), which
  entered the kinematic position directly. BRUX 2024-035 6 h, RMS from 60 min:
  C GRG kinematic IF 14.8 -> 5.9 cm, C UCi 13.7 -> 8.0 cm, EC IF 11.0 -> 1.5
  cm, GC IF 5.9 -> 4.3 cm, GEC IF 5.0 -> 2.8 cm, GEC UCi 4.29 -> 2.74 cm; the C
  GRG kinematic IF row converges later (t_c20 42.5 -> 64 min: the guarded
  satellites restart their ambiguities); the C UCi row misses its p95 relation
  to the unconstrained twin (13.8 against 11.2 cm: the twin gains more, and the
  constraint's per-arc GIM error stays; x3 sigma 9.8 cm, released 7.4 cm;
  known limitation LIM-PPP-05 in `CHANGELOG.md`). A product whose
  centre is not in `AC_GUARD_EXEMPT` is guarded in every class; only GRG and
  COD are measured in dual-frequency kinematic PPP. Single-frequency kinematic
  PPP is guarded for the `ORBEX_GUARDED` families only (since v1.0.0; today
  COD Galileo IOV / FOC, whose clocks step
  at the turns): E-SP3cod-kin-IF rms3d_kin / p95 0.1848 / 0.3181 -> 0.1078 /
  0.2445 m (388 rows of E03, E05, E09 guarded); GE / EC / GEC COD kinematic RMS
  6.20 / 8.19 / 5.25 -> 4.35 / 4.73 / 3.88 cm (measured before the
  dual-frequency extension).
- `"all"`: PPP with precise orbits, every class, static or kinematic.
- `"off"`: no guard. Broadcast orbits are never guarded.

The guard costs data (the guarded satellite restarts its ambiguity after the
turn); it helps static solutions and hurts weak kinematic ones, so `"auto"`
leaves the other classes of single-frequency kinematic runs alone (guarding
every class breaks the single-frequency BeiDou rows: p95 0.59 -> 1.87 m).
GRG's GPS IIF shadow turns and GPS IIIA turns differ from the laws by up to
60-78 deg in the same direction (0.13-0.22 cycle), which the guard does not
address; with GRG products the GRG ORBEX
file makes the static GPS-only solution slightly worse than the laws
(G-SP3grg-static-IF err3d / rms3d_post 1.05 / 2.00 -> 1.54 / 2.42 cm, t_c10
41.5 -> 24.5 min), so ORBEX is not a cure for everything.

Cost: one closed-form attitude evaluation per satellite-epoch (0.4-0.7 s for a
day of 94 satellites at 30 s), well below 1 % of a PPP run; reading an ORBEX
file once per process (0.5-1.7 s, cached).

Measured (BRUX 2024-035, 6 h, SP3/OSB of the named
centre; static err3d_final / rms3d_post [cm] / t_c10 [min], kinematic
rms3d_kin / p95 [cm] / t_c20 [min]; F = misses the row's limit):

| Row | nominal | laws | auto (default) | all | ORBEX |
|---|---|---|---|---|---|
| C GRG static UCi | 6.30/7.56/135 F | 6.32/7.56/134 F | 2.92/4.03/48 | 2.92/4.03/48 | 2.87/3.48/50.5 |
| C GRG static IF | 5.06/7.71/27.5 | 5.11/7.66/27.5 | 1.09/1.88/27.5 | 1.09/1.88/27.5 | 1.00/2.00/27.5 |
| GC COD static UCi | 1.53/3.09/68 F | 1.55/3.41/65.5 F | 1.55/3.41/65.5 F | 1.58/3.39/61 F | 1.54/3.45/62 F |
| EC COD static UC0 | 6.33/15.82/23 F | 6.26/15.32/23 F | 0.54/1.78/22.5 | 0.54/1.79/22.5 | 0.54/1.78/22.5 |
| E COD static IF | 6.42/15.25/22 | 6.33/14.79/22 | 0.60/1.58/21.5 | 0.60/1.58/21.5 | 0.60/1.58/21.5 |
| E GRG static IF | 0.44/1.77/20 | 0.44/1.68/20 | 0.44/1.68/20 | 0.41/1.86/18.5 | 0.42/1.67/20 |
| GE GRG static IF | 0.44/1.41/12 | 0.48/1.47/14 | 0.48/1.47/14 | 0.74/1.72/14 | 0.68/1.47/13.5 |
| G GRG static IF | 1.16/2.60/46 | 1.05/2.00/41.5 | 1.05/2.00/41.5 | 1.05/2.00/41.5 | 1.54/2.42/24.5 |
| GEC COD static IF | 2.50/4.01/12 | 2.24/3.44/13.5 | 0.71/1.38/13.5 | 0.71/1.43/13.5 | 0.69/1.31/14 |
| G GRG kin UCi | 9.30/16.01/132.5 F | 10.10/17.97/103.5 F | 10.10/17.97/103.5 F | 10.10/17.97/103.5 F | 13.99/23.42/52 |
| GEC GRG kin UCi | 8.03/14.97/41.5 | 7.83/14.41/38 | 7.83/14.41/38 | 6.75/13.05/38 | 7.69/13.72/38 |
| EC COD kin IF | 8.24/14.47/8.5 | 8.19/14.09/8.5 | 8.19/14.09/8.5 | 5.41/12.16/8.5 | 8.16/13.84/8.5 |

GC COD static UCi, which misses its limit in every attitude variant above,
passes with the constrained code weighting of v1.0.0 (t_c10 65.5 -> 20 min):
its slowness came from the weighting, not from the attitude.

## Cycle Slips

Before the filter, `gnx_py.core.tools.CSDetector` splits every satellite's data
into arcs of continuous phase (each arc gets its own ambiguity). The filters
only see the arc labels. Layers (`gnx_py.core.cycle_slips`):

- **L0, metadata**: data gaps, the RINEX loss-of-lock indicators (LLI) of
  every phase read from the RINEX 3 file, and a receiver millisecond
  clock-jump guard (on with `cycle_slips_detection`, whatever the two
  settings below). A change (set or clear) of the half-cycle flag (LLI bit 1,
  column `lli_half_cycle`) starts a new arc, also with
  `cycle_slips_detection=False`; a change of
  `SYS / PHASE SHIFT` inside the data does not (it is reported only; the
  phases stay aligned).
- **Today's tests**: level shifts of the geometry-free phase and of
  Melbourne-Wübbena (dual frequency) or of code minus phase (single
  frequency), with thresholds scaled by the noise measured on the data.
- **L1, third carrier** (`cycle_slip_aux_carrier`, no products, any
  sampling interval since v1.0.0; cost above 60 s below): another carrier of the satellite from the RINEX file (L1 for
  L2L5, E1 for E5aE5b, B1C for B2aB2b, L5 for L1L2, E5b for E1E5a, L2W / E5a /
  B3I for single-frequency L1 / E1 / B1I) forms cross pairs with each
  processed signal, so an equal slip (n, n) on a close pair changes their
  Melbourne-Wübbena combination; a common-mode guard ignores ionospheric
  events that hit most satellites at once (solar flares).
- **L2, geometry** (`cycle_slip_geometry`, precise orbits and 30 s sampling
  only in PPP; the TEC session runs it with broadcast orbits too since v1.0.0, a broadcast record switch starting a new piece): an
  epoch-differenced test of every carrier of every satellite of all
  selected systems against the orbits and clocks of the frame, with one
  receiver-clock change per epoch. It finds slips on all carriers at once
  (n, n, n), equal slips on L1/L2 under the polar ionosphere and one-cycle
  slips on a single frequency. The 5-min SP3 clocks limit it at high
  latitude; `clk_path` gives it 30 s RINEX clocks of the SP3's analysis
  centre (another centre is ignored with a warning; rows the files do not
  cover keep the SP3 clocks). An a-priori position more than max(5 m,
  4 sigma) from the ionosphere-free code solution is replaced by it; a static
  run more than 100 m off has the layer off. Kinematic runs (`q_crd > 0`)
  keep the static model while the receiver does not move; a moving receiver
  is tracked at its integrated position, or the layer is off for the run.
- **L3, fusion**: a flag that only one layer raises must show a persistent
  ionosphere-free step (7 sigma, 7 cm); a slip on the third carrier alone
  does not break the arc. No integer repair: every slip starts a new arc.
  Without the geometry layer (broadcast orbits, sampling other than 30 s, or
  no result of the layer) a third-carrier flag that no other test, LLI or
  common-mode event raises breaks the arc only when the joint integer estimate
  of the cross pairs cannot exclude a slip on the processed signals
  (Melbourne-Wübbena within 0.2 cycle, geometry-free within 3 sigma, 6-sigma
  chi2 margin): a non-integer carrier step, such as the 4.4-4.7 mm L5 step of
  the Block IIF G27 at its 18:00 broadcast record switch (BRUX 2024-035),
  keeps the arc (GPS L2L5 broadcast STEC p99
  against the precise run 0.64 -> 0.0000005 TECU).

`cycle_slips_detection=False` keeps only the data gaps and LLI.

Found at the slip epoch (library bench, 30 s data, 30 s clocks; BRUX / NKLG /
NYA2 2024-035 / NYA2 2024-132, held-out BRUX / NKLG / NYA2 2024-131; today in
brackets):

| Mode, slip | All layers | Today |
|---|---|---|
| L2/L5 (1, 1) | 100 % at all seven | 58.7 / 32.7 / 2.2 / 7.0; 47.3 / 30.6 / 2.1 % |
| L2/L5 (1, 1, 1) | 100 / 99.7 / 96.1 / 97.4; 99.4 / 98.9 / 97.1 % | 1.6-57.6 % |
| E5a/E5b (1, 1) | 100 % at all seven | 1.6-42.7 % |
| B2a/B2b (1, 1) | 100 % (BRUX, NKLG 035, BRUX 131) | 19.5-38.0 % |
| L1/L2 (1, 1) | 100 / 100 / 99.6 / 99.6; 100 / 98.8 / 98.7 % | 95.4 / 84.4 / 8.8 / 21.5; 86.5 / 71.0 / 11.8 % |
| L1 single frequency, 1 cycle (L2W in the file) | 99.6-100 % | 0 % |

- New ambiguities per satellite-hour on clean data: at most +0.016 over
  today (L2/L5 at NYA2 2024-035: 0.104 → 0.118).
- Epochs dropped as unverifiable on L1/L2 at NYA2: 4.44 / 3.86 % →
  0.26 / 0.80 %.
- A slip on the third carrier alone breaks the arc in 0–3.9 % of the cases.
- Converged PPP (SP3cod, 6 h static): a missed (1, 1) slip costs 4.93 cm
  (BRUX IF-E E5a/E5b) and 2.64 cm (NYA2 IF-G L1/L2) of final 3D error; with
  all layers the slip starts a new arc and the final error changes by 0.09 /
  0.42 cm.

Fallbacks:

- Broadcast orbits, sampling other than 30 s: L0 + L1 only. Close-pair
  (1, 1) 95.6–99.7 %, but (1, 1, 1) at NYA2 5.3–11.7 % and L1/L2 (1, 1) at
  NYA2 59–61 %.
- 5-min SP3 clocks (no `clk_path`), NYA2 2024-035: (1, 1, 1) on L2/L5
  85.9 %, L1/L2 (1, 1) 62.6 %; correlated L1 + L2W slips on single-frequency
  L1 45.1 % (BRUX 3 h with GRG clocks: 87.5 % against 91.7–95.8 % with the
  third carrier alone).
- Kinematic (`q_crd > 0`): a static receiver keeps the static results (NYA2
  2024-035 L2/L5 (1, 1, 1) 96.1 %); a receiver on a simulated 2 km circle,
  tracked: (1, 1, 1) 100 % at BRUX, 94.7 % at NYA2 (0–8 h), no new
  ambiguity over today; GPS only on that circle, or single frequency
  without a second band: motion guard (third carrier and today's tests).
- Close pair without a third carrier: (1, 1) on L2/L5 98.8 / 87.5 / 17.2 /
  32.6 %. Single frequency without a second band, one cycle on L1: 99.2 /
  90.6 / 20.9 / 38.8 %.
- Neither products nor a third carrier (other modes, RINEX 2,
  `cycle_slip_aux_carrier=False`): today's tests, with their limits.
  On a close pair an equal slip (n, n) leaves Melbourne-Wübbena
  unchanged and moves the geometry-free phase by only n·|λ1 − λ2| (1.06 cm
  per cycle for L2/L5, 0.65 cm for E5a/E5b): (1, 1) found in 75 % (L2/L5) /
  35 % (E5a/E5b) of the cases at BRUX. Single-frequency code minus phase is
  limited by the code noise: 5-cycle slips 28–55 %, 20 cycles 95–100 %. At
  120–300 s sampling the linear geometry-free model leaves ionospheric
  curvature: about 0.14 spurious slips per satellite-hour at 300 s.
- Sampling above 60 s (no geometry layer; the third carrier runs at every
  sampling interval since v1.0.0).
  BRUX / NKLG / NYA2, quiet day 2024-035 and storm day 2024-132, 24 h
  thinned from 30 s, without → with the third carrier:

  | Interval, day | (1, 1) L2/L5, E5a/E5b | 1 cycle L1/E1 single frequency | (1, 1, 1) | Arc broken by a slip on the third carrier alone | New ambiguities per satellite-hour | Epochs lost |
  |---|---|---|---|---|---|---|
  | 120 s, quiet | 0.0–27.4 → 81.0–97.2 % | 0.0–1.1 → 63.9–84.2 % | 0.8–22.1 → 0.8–52.7 % | 0–2.3 → 25.1–93.9 % | +0.000 to +0.016 | +0.0 to +0.6 points |
  | 300 s, quiet | 0.4–3.5 → 42.1–98.7 % | 0.0–0.6 → 6.2–58.8 % | 0.6–2.0 → 0.6–9.4 % | 0–2.8 → 4.8–96.0 % | +0.000 to +0.033 (BRUX L2/L5 0.074 → 0.107) | +0.0 to +1.9 points (NYA2 L2/L5 0.75 → 2.60 %) |
  | 120 s, storm | 0.7–12.1 → 85.6–95.9 % | 0.0 → 58.9–78.7 % | 0.0–8.5 → 0.4–22.5 % | 0–2.5 → 33.9–94.8 % | +0.000 to +0.032 (BRUX L1 0.000 → 0.032) | up to +1.1 points (NYA2 E5a/E5b 1.78 → 2.90 %) |
  | 300 s, storm | 0.0–3.6 → 69.1–98.5 % | 0.0–0.7 → 6.0–55.8 % | 0.0–2.5 → 0.0–2.5 % | 0–4.4 → 3.5–97.0 % | +0.000 to +0.024 (NYA2 E5a/E5b 0.107 → 0.131) | up to +1.5 points (NYA2 L2/L5 2.27 → 3.80 %) |

  Default modes L1/L2 and E1/E5a (third carrier L5 / E5b), both days:
  (1, 1) 0.6–65.1 → 54.0–100 % at 120 s and 0.6–15.5 → 52.5–100 % at
  300 s; one cycle on one signal at 300 s 5.1–67.9 → 24.5–98.1 % (NYA2
  E1/E5a quiet 5.8 → 95.4 %); (1, 1, 1) unchanged; a slip on the third
  carrier alone breaks the arc in 29.8–99.7 % (120 s) and 53.4–100 %
  (300 s) of the cases; new ambiguities per satellite-hour +0.000 to +0.020
  (NYA2 E1/E5a storm 120 s 0.170 → 0.190); epochs lost up to +0.9 points
  (NYA2 E1/E5a storm 300 s 2.23 → 3.17 %).
  At 300 s on single frequency the layer cannot tell which carrier slipped:
  a flag is about as likely to come from a slip on the third carrier as on
  the processed signal (found vs broken, quiet day: NKLG L1 16.5 vs 17.0 %,
  E1 58.8 vs 54.2 %; NYA2 L1 6.2 vs 4.8 %). TEC sessions pass the same slip
  context, so a STEC run above 60 s gets the layer too (not benched).

  The 60 s rows were already on: (1, 1) 92.9–100 %, a slip on the
  third carrier alone breaks the arc in 9.0–93.6 % (NYA2 highest). A finite
  `gnx_py.core.cycle_slips.AUX_MAX_INTERVAL_S` (e.g. 60) switches the layer off
  above it.

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

DSB chains (PPP and SPP alike; `gnx_py.models.biases`): a code-bias difference that a
DSB product gives only indirectly follows the shortest chain of satellite DSBs
between the two codes, and the receiver part adds the station DSB of every
link of that same chain (the product's `STA_BIAS_<a>_<b>` record; a link
without a station value counts 0), not only of the first code pair. The
receiver part is a code difference (a receiver DCB), not a
bias common to the codes: on the ionosphere-free and single-code routes it is
absorbed by the receiver clock and the inter-system bias and does not move the
position; on uncombined routes without an ionospheric constraint it goes into
the slant-ionosphere states; on GIM-constrained uncombined routes
(`use_iono_constr=True`, `est_dcb=False`) it biases the ionosphere states
against the GIM and can move the solution. The products of the validation
data list no receiver DSB, so none of its rows is affected.

Since v1.0.0:

- One bias source per ionosphere-free pair (PPP and SPP): with
  `dcb_path=[OSB, DSB]` a pair whose second code has no OSB (GPS C5Q,
  Galileo C7Q in the COD / GRG / GFZ MGEX OSB) takes both codes from the DSB
  product (else the partner OSB plus the DSB difference, else the broadcast
  path), with a warning naming the code and the source. Mixing the sources
  amplified the OSB / DSB disagreement 31x (L2L5) / 43x (E5aE5b): BRUX
  2024-035 6 h static IF E5aE5b COD+CAS1 2.21 -> 1.06 cm, t_c10 229.5 -> 55.5
  min.
- GPS Block IIF L5 inter-frequency clock bias (IFCB): the L5 phase of the IIF
  satellites carries a time-varying bias (IF(L1,L2) - IF(L1,L5) up to 6.1 cm
  std, about 20 cm peak to peak within an arc) that IF(L1,L2) precise clocks do
  not contain; IF(L2,L5) amplifies it 8.9x. L1L5, L2L5 and L5 with precise
  clocks give one warning per run when IIF satellites are present; this is a
  documented v1 limitation (BRUX 2024-035 6 h static IF L2L5 1.34 m, never
  converges; L1L5 0.14 m); an IFCB model is planned after v1.

## PPP-AR Status

PPP ambiguity resolution is available through `gnx_py.ppp.pppar` helpers and
selected filter branches. The implementation includes:

- Candidates: satellites with product phase biases (phase OSB) for the
  phase observables of the mode. Broadcast orbits, runs without `dcb_path`
  and DCB-only products give none: AR is then off with a warning and the
  result is the float solution. A phase flagged with LLI bit 1 (half-cycle
  ambiguity possible) is not fixed while the flag is set (reason
  `half_cycle`, column `ar_excluded_half_cycle`, RTKLIB's relative-positioning rule); the float solution is
  unaffected.
- Uncombined dual-frequency AR in two steps: wide lane N1 - N2 by rounding
  where that is safe (fraction within `pppar_wide_lane_max_frac_cycles`,
  sigma below 0.15 cycles; satellites with a receiver-specific wide-lane
  bias, e.g. GPS III or Galileo IOV on some receivers, are left out), then
  N1 conditioned on it (narrow lane) with LAMBDA integer least squares.
- Partial fixing (`pppar_partial_fixing_enabled`, default on): young arcs
  wait, and the subset of decorrelated ambiguities with a bootstrapped
  success rate of at least 0.999 is fixed when the full set is not.
- Validation of every fix: ratio test, success rate, narrow-lane gate (each
  fixed combination within 0.25 cycles of its float value), chi-square test
  of the float-vs-fixed ambiguity distance, fixed-vs-float position check.
  Since v1.0.0 the wide-lane sigma
  gate, the success rate, the chi-square distance and the position check use
  the float covariance times the a-posteriori variance factor s0² = v'R⁻¹v /
  dof of the float phase residuals of the epoch (dof = the sum of their
  redundancy numbers, `pppar.update_redundancy`), floored at 1, per system;
  the LAMBDA search, the ratio and the narrow-lane gate do not change with a
  scale. R07 days: factor > 1 on 179-318 of 2820 AR epochs (max 7.3), +11 to
  +41 fixed epochs per day, fixed-minus-float unchanged. On the validation data
  (BRUX 2024-035, GRG) the first GPS-only fix comes after about 2.5 h,
  Galileo after about 50 min, G+E after about 35 min.
- The narrow-lane search is conditioned on the rounded wide-lane integers
  with an internal 1e-3 cycles; `pppar_constraint_sigma_cycles` (and its
  floor) weight only the constraint applied to the filter state (before
  v1.0.0, 0.1 cycles there fixed nothing).
- Warm-up: `pppar_warmup_epochs` counts epochs from the session start (the
  first epoch of the run), not from the last filter reset (`reset_every`); a
  reset restarts every arc age instead (see the arc age below).
- Arc-age / lock-time gating and elevation-based candidate filtering. The arc
  age is the number of epochs of the arc before the current one, 0 at the
  first epoch of every arc (new satellite, slip, filter reset): with
  `pppar_min_lock_epochs=N` an ambiguity is fixable from the (N+1)-th epoch of
  its arc.
- By default the filter stays float and every epoch is fixed anew from it,
  so a wrong fix cannot outlive its epoch; `pppar_fix_and_hold=True` feeds
  the fixes back into the filter.

Recommendations:

- Keep `pppar_enabled=False` for baseline PPP validation.
- Enable AR only after float PPP is stable, with an orbit/clock product and
  its own phase-bias (OSB) product. When the OSB file comes from another
  solution than the orbits and clocks (IGS long names, e.g. COD0OPSFIN OSB
  with COD0MGXFIN orbits) the session warns and disables AR (float result;
  AR diagnostics reason `mixed_products`, `ar_excluded_mixed_products`);
  `pppar_allow_mixed_products=True` forces it, with most fixes then rejected
  by the validation.
- Combined ionosphere-free filters fix no ambiguity: the ionosphere-free
  ambiguity is an integer only together with the wide lane, which they do
  not separate. With `pppar_enabled=True` they warn and return the float
  solution, also with `pppar_combined_if_ar_enabled=True`.
- Single-frequency PPP without ionospheric constraints ignores
  `pppar_enabled` (the GRAPHIC ambiguity is half a carrier ambiguity plus
  biases, not an integer).
- Treat BDS AR as requiring additional phase-bias validation.
- AR mainly improves the horizontal position. The positive +1 to +3 cm
  height offset of the fixed solution reported before v1.0.0
  (antenna PCV and ocean tide loading not modelled then) is gone: on the 24 h
  R07 days of the validation data the fixed-minus-float 3D RMS on the fixed
  epochs is -11.3 to +1.1 mm, and no validation run is worse fixed than float
  in 3D by more than 1 cm (the validation grades fixed <= float + 1 cm in 3D;
  the height is not graded separately). A small negative
  fixed-minus-float height remains at the high-latitude station: NYA2 R7-G
  24 h 2024-035, -0.27 cm without and -0.38 cm with ocean tide loading
  (recorded with the station).

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

Precise clock datum (PPP and SPP alike). Precise
BeiDou clocks refer to an ionosphere-free pair, and products differ: the 2024
MGEX products refer to IF(B1I,B3I), COD MGEX 2026 refers its BDS-3 clocks to
IF(B1C,B2a). The session resolves the datum once
(`session_guards.resolve_precise_clock_datum`; `solution.attrs
["precise_clock_datum"]`):

1. the setting `bds_precise_clock_datum` (`"auto"` default, `"B1IB3I"`,
   `"B1CB2a"`);
2. the declaration of a bias product of the clocks' analysis centre (its
   SATELLITE_CLOCK_REFERENCE_OBSERVABLES; another centre's declaration is
   evidence only);
3. a test on the broadcast clocks (precise minus broadcast clock per BDS-3
   satellite against the broadcast group delays; accepted at a
   between-satellite scatter <= 1.5 m and 2.5 times below the other datum;
   COD 2026-213: 9.07 m against 0.57 m);
4. otherwise IF(B1I,B3I) with a warning naming the risk (about 10 m).

The datum applies per satellite generation, taken from the ANTEX block valid
at the first observation (then B-CNAV records, then the PRN rule): BDS-2 and
BDS-3 GEO stay in IF(B1I,B3I) (no open-service B1C/B2a; not verified with a
2026 GEO product). By 2026 PRNs C01-C04 are BDS-3 GEO and C06-C08, C11-C14
BDS-3, so a "PRN below 19 is BDS-2" rule is wrong there. Codes that cannot be
moved to the datum (a RINEX 3 navigation file without B-CNAV group delays and
no bias product in that datum) make the run refuse with a ValueError naming
the remedies (`bds_freq="B1CB2a"`, a RINEX 4 file such as BRD400DLR, the
centre's OSB product); some satellites without B-CNAV records are named in one
warning. An OSB product is applied as given only when every OSB product of
`dcb_path` declares the session's datum, otherwise it is re-referenced to it.
BRUX 2026-213 (COD MGEX 2026), COD + BRD400DLR: PPP C kinematic p95 after 1 h
6.63 -> 0.49 m (SPP C B1IB3I 11.43 -> 2.36 m rms).

Broadcast orbits: positioning applies no
satellite antenna term for any system; the message group delays map the
clock's reference signal (BeiDou: B3I) to the user's signal, as RTKLIB does.
Broadcast BeiDou PPP took the full signal PCO before: C-BRDC-kin-IF p95
2.210 -> 1.668 m (BRUX), 2.577 -> 0.932 m (NKLG); static C IF 0.729 -> 0.657
m. With broadcast clocks the band-7 code of a BeiDou satellite is B2b-I
(TGD_B2bI) or B2I (TGD2) by the satellite generation of the run (ANTEX block,
then B-CNAV records, then the PRN), as with precise clocks (before v1.0.0,
the PRN rule, wrong after 2023 for the reassigned PRNs).

BDS-3 satellites whose codes cannot be moved to the precise clock datum
(IF(B1C,B2a) clocks, no B-CNAV TGD_B1Cp / TGD_B2ap of the satellite and no
product bias in that datum) keep the warning and a group delay of 0; since
v1.0.0 SPP and every PPP filter down-weight their code observations: the size of the missing term
is added in quadrature to the code sigma, the RMS over the movable BDS-3
satellites of the run of the correction of the observable the filter uses
(the mode's combination for SPP and the IF / GRAPHIC filters,
`PreciseClockDatum.unmoved_sigma_m`; a single code for the uncombined filters,
`unmoved_code_sigma_m`), at least 3 m (`biases.UNMOVED_CODE_SIGMA_FLOOR_M`).
Only satellites with SP3 records are named. BRUX 2026-213 with the B-CNAV
records of three satellites removed: SPP C B1IB3I rms3d 4.68 -> 2.75 m
(2.36 m with every satellite movable), PPP C B1IB3I IF static RMS after 1 h
0.205 -> 0.115 m. An OSB product is recognised by its Bias-SINEX header
(`BIAS_MODE`, else the bias-mode letter of the `%=BIA` line;
`biases.bias_product_mode`), not by "OSB" in the file name.

## Developer Notes

Active primary classes:

- `PPPDualFreqSingleGNSS`
- `PPPDualFreqMultiGNSS`
- `PPPUducSFSingleGNSS`
- `PPPUdSingleGNSS`
- `PPPUdGenericMixedGNSS`
- `PPPFilterMultiGNSSIonConstGEC`

Other routed classes (each is selected by `PPPSession` for the inputs named):

- `PPPFilterMultiGNSSIonConst`: GPS+Galileo, both dual-frequency, ionospheric
  constraints, `uncombined_iono_constraints_model="legacy"`, precise orbits; the
  GPS/Galileo constrained reference baseline.
- `PPPUdMixedGNSS`: GPS+Galileo with ionospheric constraints, the 'legacy'
  constraint model and precise orbits, when at least one of the two systems uses
  a single-frequency mode.
- `PPPSingleFreqSingleGNSS`: one system, single frequency, no ionospheric
  constraints.

The session selects exactly the nine classes of these two lists, and no other
(see the `create_uncombined_filter` docstring). The classes it never selected
were removed in v1.0.0:
`PPPUducSFMultiGNSS` (broken: `KeyError 'P4'` when forced), `PPPUdMultiGNSS`
(duplicated `PPPUdGenericMixedGNSS`), `PPPSingleFreqMultiGNSS` and
`_LegacyPPPDualFreqMultiGNSS`, together with the unreachable G+C / E+C and
unconstrained G+E router branches. Reroute a configuration only with numerical
regression coverage of the routes involved.

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

The PPP package was validated for v1.0.0 over a configuration matrix (routes,
systems, products, static and kinematic, AR, single frequency, daily rows) at
the stations BRUX, NKLG and NYA2 on 2024-035 (quiet), 2024-131 (storm onset)
and 2024-132 (storm), against independent references, with the accuracy
limits referred to RTKLIB on the same data. Readable errors and warnings, the
router table (the nine classes of Developer Notes), the filter core and
weights, PPP-AR, the bias handling, the BeiDou clock datum, the attitude,
ORBEX and frames, and the tides were checked with focused tests. Configurations
that stay outside their limits for an established cause are listed as known
limitations in `CHANGELOG.md`.

Executed tutorial notebooks and the scripts under `examples/` are
illustrations, not validation references.

The validation covers important branches, but it is not a formal accuracy
guarantee for every possible constellation, product source, station, signal
mode, receiver, and arc condition. New scientific uses should still validate
against independent products, known coordinates, and representative datasets.

## Known Limitations

- Combined IF PPP-AR fixes no ambiguity (no wide-lane / narrow-lane split).
- BDS AR requires additional phase-bias validation.
- Mixed uncombined no-constraint PPP is sensitive to bias/product consistency.
- GPS Block IIF L5 inter-frequency clock bias: L1L5, L2L5 and L5 with
  precise clocks warn and stay a documented v1 limitation (Bias Products; a
  model is planned after v1).
- Single-frequency positioning at a station whose code delay varies with
  elevation keeps a height error: NYA2 (JAVRINGANT_G5T) L1 about -0.21 m, the
  code about 10 cm longer at 50-55 deg than at 30 deg relative to the phase,
  for G, E and C; igs20.atx has no code calibration for it (no
  group-delay-variation model in v1).
- Ocean tide loading at NYA2 costs 17 % of the fixed epochs of the 24 h
  R7-G row (1836 -> 1518, rejected by the chi-square float-versus-fixed test);
  a station limitation.
- Without the ORBEX file of the product's analysis centre the satellite attitude
  at noon/midnight turns is the published law, which some centres do not fly
  (see Satellite Attitude); the `"auto"` turn guard covers the measured
  classes of GRG, COD and GFZ only.
- The reference classes (`PPPFilterMultiGNSSIonConst`, `PPPUdMixedGNSS`) serve
  only the G+E configurations listed in Developer Notes; new workflows start from
  the active primary classes.
- Rerouting or consolidating classes requires before/after numerical
  regression tests.
- Broadcast G+E uncombined PPP with the GIM constraint: applying the
  satellite group delays made it converge faster but end
  slightly worse (values measured during the v1.0.0 development, before the
  weighting, PCV, ocean tide loading and attitude changes; 24 h: E1E5b RMS after 2 h
  0.135 -> 0.150 m, E5aE5b 0.170 -> 0.218 m; 12-24 h median over 24 runs
  0.171 -> 0.191 m; NYA2 B-C-IF 0.100 -> 0.272 m), while G-only is unchanged
  and E-only improves. The cause is not established (suspected: no receiver
  DCB estimate against the GIM on this route, see DSB chains above). The rows
  pass their current limits (a receiver-DCB estimate on this route
  is not scheduled).
- The broad public API is currently compatibility-preserving and includes
  symbols that are not intended long-term PPP API.

