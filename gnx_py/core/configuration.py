from __future__ import annotations
import difflib as _difflib
import os as _os
import re as _re
from typing import List, Set,Optional, Union, Literal, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, fields as _dataclass_fields
from .gnss import SUPPORTED_MODES_BY_SYSTEM, SYSTEM_NAMES, normalize_systems
from .time import check_time_limit
from .validation import as_path_list, check_choice, check_flag, check_number, check_paths_exist

#: allowed values of the model settings shared by every configuration
ORBIT_TYPES = ("broadcast", "precise")
IONOSPHERE_MODELS = ("gim", "klobuchar", "ntcm", "bdgim", False)
#: GIM mapping function: by the IONEX header ("auto"), single layer, CODE modified single layer
GIM_MAPPING_FUNCTIONS = ("auto", "slm", "mslm")
#: GIM time interpolation: rotated maps (IONEX 1.0 eq. 3) or earth-fixed maps (eq. 2)
GIM_TIME_INTERPOLATIONS = ("rotated", "earth_fixed")
#: BeiDou Klobuchar: GPS coefficients scaled to the BDS signal ("gps", default) or the BDS-ICD model ("icd")
BDS_KLOBUCHAR_MODELS = ("gps", "icd")
TROPOSPHERE_MODELS = ("niell", "saastamoinen", "collins", False)
SAT_PCO_MODES = ("los", "crd", False)
#: what a session does with a satellite that has no valid ANTEX entry for the frequencies it needs
#: (gnx_py.antex): drop its observations with one warning naming them, or raise ValueError
MISSING_SAT_ANTENNA_POLICIES = ("exclude", "raise")
#: input-file settings checked for existence before a session starts
INPUT_PATH_SETTINGS = ("obs_path", "nav_path", "sp3_path", "atx_path", "dcb_path", "sinex_path", "gim_path",
                       "clk_path", "ocean_loading_path", "erp_path", "attitude_path")
#: mean pole of the pole tide (gnx_py.tides.mean_pole_mas)
POLE_TIDE_MEAN_POLES = ("secular", "cubic")
#: satellite attitude: attitude laws / ORBEX, or nominal yaw steering (BeiDou GEO orbit-normal)
SATELLITE_ATTITUDE_MODES = ("model", "nominal")
#: PPP turn guard (DDPreprocessing._apply_turn_guard)
ATTITUDE_TURN_GUARDS = ("auto", "all", "off")
#: rows inside a gap of the 30 s clock files (clk_path): left out ("drop", RTKLIB pephclk) or given the SP3
#: clock ("sp3"); see rinex_clk.apply_clk_clocks
CLK_GAP_POLICIES = ("drop", "sp3")
#: IF datum of the precise BeiDou clocks of BDS-3 satellites: "auto" (the bias
#: product's declaration, else the broadcast-clock test, else IF(B1I,B3I) with a warning) or the pair itself
BDS_PRECISE_CLOCK_DATUMS = ("auto", "B1IB3I", "B1CB2a")
_MODEL_NOTES = {
    "nequick": "NeQuick is not implemented in GNX-py",
    "nequick-g": "NeQuick is not implemented in GNX-py",
    "nequickg": "NeQuick is not implemented in GNX-py",
    "nequick2": "NeQuick is not implemented in GNX-py",
}
_FREQ_SETTING = {"G": "gps_freq", "E": "gal_freq", "C": "bds_freq"}
_UNSET = object()
#: frames inside gnx_py (and the generated dataclass ``__init__``) are skipped when a settings warning is
#: attributed, so it points at the caller's line
_PACKAGE_PREFIX = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))) + _os.sep


def _warn_unused_setting(name: str, value: Any, instead: str) -> None:
    """UserWarning for a setting that is accepted but read by nothing (a warning at a non-default value; the
    setting is removed after GNX v1)."""
    import sys as _sys
    import warnings as _warnings

    # attributed to the first frame outside gnx_py and the generated dataclass __init__ (the user's line)
    level, frame = 2, _sys._getframe(1)
    while frame is not None and (frame.f_code.co_filename.startswith(_PACKAGE_PREFIX)
                                 or (frame.f_code.co_filename == "<string>" and frame.f_code.co_name == "__init__")):
        frame, level = frame.f_back, level + 1
    _warnings.warn(f"{name}={value!r} is not used: {instead}. The setting has no effect and will be removed after "
                   f"GNX-py v1.", UserWarning, stacklevel=level)


_FLAG_FIELDS: Dict[type, tuple] = {}


def _flag_fields(cls: type) -> tuple:
    """``(name, optional)`` of the dataclass fields annotated ``bool`` or ``Optional[bool]`` (cached per class)."""
    cached = _FLAG_FIELDS.get(cls)
    if cached is None:
        flags = []
        for f in _dataclass_fields(cls):
            t = str(f.type).replace("typing.", "")
            if t in ("bool", "<class 'bool'>"):
                flags.append((f.name, False))
            elif t in ("Optional[bool]", "Union[bool, None]", "bool | None"):
                flags.append((f.name, True))
        cached = _FLAG_FIELDS[cls] = tuple(flags)
    return cached


def _same_value(value: Any, default: Any) -> bool:
    try:
        return bool(value == default)
    except Exception:  # noqa: BLE001 - e.g. an array compared with a string: not the default
        return False


def _choice(name: str, choices, **kw) -> Callable[[Any], Any]:
    return lambda value: check_choice(name, value, choices, **kw)


def _number(name: str, **kw) -> Callable[[Any], Any]:
    return lambda value: check_number(name, value, **kw)


def _check_attitude_laws(value: Any) -> Optional[Dict[str, str]]:
    """``attitude_laws``: None or {law family: law} of ``gnx_py.attitude`` (ValueError with a "did you mean")."""
    if value is None:
        return None
    from collections.abc import Mapping

    if not isinstance(value, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"attitude_laws must be None or a dict {{law family: law}}, e.g. {{'gps_iii': 'iir'}} "
                         f"(got {value!r})")
    from ..models.attitude import check_laws

    check_laws(dict(value))
    return dict(value)


def _optional_path(name: str, value: Any) -> Any:
    """One input file path (``str``/``Path``) or None; ``TypeError`` naming the setting otherwise."""
    if value is None:
        return None
    import os as _os

    if isinstance(value, (str, _os.PathLike)) and str(value).strip():
        return value
    raise TypeError(f"{name} must be a file path (str or pathlib.Path) or None (got {value!r})")


def _check_approx_position(value: Any) -> Optional[List[float]]:
    """``approx_position``: None or three finite ECEF coordinates [m] of a point on the Earth's surface
    (geocentric distance within ``rinex_obs.APPROX_XYZ_MIN_M`` .. ``APPROX_XYZ_MAX_M``)."""
    if value is None:
        return None
    import numpy as _np
    from ..io.rinex_obs import APPROX_XYZ_MAX_M, APPROX_XYZ_MIN_M

    try:
        xyz = _np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        xyz = _np.zeros(0)
    if isinstance(value, (str, bytes)) or xyz.size != 3 or not _np.all(_np.isfinite(xyz)):
        raise ValueError(f"approx_position must be None or [X, Y, Z] in ECEF metres (got {value!r})")
    r = float(_np.linalg.norm(xyz))
    if not APPROX_XYZ_MIN_M <= r <= APPROX_XYZ_MAX_M:
        raise ValueError(f"approx_position {value!r} is {r:.0f} m from the geocentre: an ECEF position on the "
                         f"Earth's surface ({APPROX_XYZ_MIN_M:.3g} - {APPROX_XYZ_MAX_M:.3g} m) in metres is needed")
    return [float(v) for v in xyz]


def _check_systems(value: Any, name: str = "sys") -> set:
    """``sys``/``use_sys`` as a non-empty set of supported system codes (ValueError naming ``name``)."""
    allowed = sorted(SYSTEM_NAMES)
    if isinstance(value, str):
        systems = {value}
    else:
        try:
            systems = set(value)
        except TypeError:
            raise ValueError(f"{name} must be one of {allowed} or a set of them (got {value!r})") from None
    if not systems:
        raise ValueError(f"{name} must name at least one of {allowed} (got {value!r})")
    try:
        return normalize_systems(systems)
    except ValueError:
        unknown = sorted(str(s) for s in systems - set(SYSTEM_NAMES))
        msg = f"{name} must be one of {allowed} or a set of them (got {value!r}); unsupported: {unknown}"
        upper = {s.upper() for s in systems if isinstance(s, str)}
        if len(upper) == len(systems) and upper <= set(SYSTEM_NAMES):
            fixed = next(iter(upper)) if len(upper) == 1 and isinstance(value, str) else sorted(upper)
            msg += f"; did you mean {fixed!r}?"
        elif isinstance(value, str) and len(value) > 1 and set(value) <= set(SYSTEM_NAMES):
            msg += f"; to select several systems pass a set, e.g. {sorted(set(value))}"
        elif any(s in ("R", "GLONASS") for s in systems):
            msg += " (GLONASS is not supported)"
        raise ValueError(msg) from None

@dataclass()
class Config:
    """Default GNX-py Configuration class.

    Every setting is validated at construction and on every later assignment: an
    unknown model name, an out-of-range number or an empty system set raises a
    ``ValueError`` naming the setting, the allowed values or range and the given
    value (with a "did you mean" hint for case/spelling variants). ``sp3_path``
    accepts one path or a list of paths. ``rinex2_tracking`` (RINEX 2 observations
    only) names the tracking mode of a RINEX 2 observable for the bias lookup, e.g.
    ``{"G": {"C5": "C5X"}}``; a code it names also sets the tracking family of the
    other codes of that system the converter's header block does not give (name
    the codes of the other system too: ``{"G": {"C5": "C5X"}, "E": {"C1": "C1X"}}``).
    Sessions call ``validate_for_run`` before
    reading any file (``obs_path`` required, model/path consistency, every given
    input file must exist).

    Ionosphere (shared by ``SPPConfig``, ``PPPConfig`` and ``TECConfig``):
    ``ionosphere_model`` is ``"gim"`` (``gim_path``), ``"klobuchar"``,
    ``"ntcm"`` or ``"bdgim"`` (``nav_path``) or ``False``. The 'ion' column is
    the slant delay of the first signal of the mode.

    - ``"bdgim"``: the BeiDou-3 global broadcast model
      (BDS-SIS-ICD-B2b-1.0 §7.7, ``gnx_py.ionosphere.bdgim``), global, for G,
      E and C. It needs a RINEX 4 ``nav_path`` with '> ION Cxx CNVX' records
      (e.g. the IGS/DLR BRD400DLR merged file, which also carries the G/E/C
      ephemerides); RINEX 2/3 files have none and raise a ValueError naming
      the file and its version when the run starts (``validate_for_run``).
      Per epoch the latest plausible set at or before the epoch (GPST) is
      used, from any BeiDou satellite; the model is evaluated in BDT
      (GPST - 14 s). A set older than 6 h, or epochs more than 6 h before the
      first set of the file (they take that set), give one warning per run
      and the set is used. TEC -> metres with GNX's 40.3e16 (the ICD's
      40.28e16 differs by 0.05 %); a negative model STEC gives 'ion' = 0. The
      run report (sets used, ages, clamped rows, rejected records in the
      window) is ``session.run_notes.iono_report`` per system after
      ``run()``. PPP constraint sigma ('ion_rms'): hypot(50 % of the model
      delay, 5 TECU x mapping function).

    - ``bds_klobuchar`` (``"gps"`` default, ``"icd"``): the BeiDou Klobuchar.
      ``"gps"`` takes the GPS coefficients (GPSA/GPSB) scaled to the BeiDou
      signal, as before v1.0.0. ``"icd"`` is an opt-in: the BDS model
      of BDS-SIS-ICD-B1I 3.0 5.2.4.7 (``ionosphere.models.klobuchar_bds``)
      with, per epoch, the broadcast set in force (RINEX 4
      '> ION' records, GEO first, the latest at or before the epoch; RINEX 3
      one GEO-first header set); without any it falls back to ``"gps"`` with a
      warning. Measured against the COD GIM: GEO-first 19.45 TECU, latest
      record 22.15, GPS-scaled 16.53. The BDS-2 broadcast coefficients
      are fitted for the Asia-Pacific service area: over 9 station-days
      (BRUX/NKLG/NYA2 x 2024-035/131/132) the slant RMS against the COD GIM at
      B1I is 17.95 TECU GPS-scaled and 19.58 TECU with the BDS model, which is
      worse on 5 of 9 days (1 tie). In SPP 29 BeiDou rows get worse and 22
      better with it (NKLG 2024-132 C-only rms_3d 3.08 -> 6.75 m). The global
      BDS-3 model is ``ionosphere_model="bdgim"``.
    - ``gim_mapping_function`` (``"auto"`` default: MSLM for CODE, single layer
      for COSZ) and ``gim_time_interpolation`` (``"rotated"`` default) improve
      the GIM slant TEC (COD 10-15 deg bin -3.30 -> 0.00 TECU, k_ep 0.879 ->
      0.973). Position domain (measured for v1.0.0; no GIM-handling
      artefact): SPP with a GIM (63
      full-day rows) loses a median 0.069 m of rms_3d with the MSLM against the
      single layer (36 rows better with the single layer, 27 worse): the MSLM
      raises the single-frequency height by 0.4-1.4 m at every station, which
      helps or hurts with the station's other height errors (mean Up BRUX
      2024-035 +0.63 m MSLM / +0.03 m SLM, NKLG 2024-035 +0.17 / -1.24 m);
      the rotation costs 0.010 m. With a GIM constraint (PPP) the four
      combinations move the static UCi rows by at most 6.5 min (rotation equal
      or faster on the COD rows); the slow convergence measured earlier was
      the constrained code weighting (fixed in v1.0.0: GC static UCi
      t_c10 65.5 -> 20 min). Opt-outs: ``gim_mapping_function="slm"``,
      ``gim_time_interpolation="earth_fixed"``.
    - GIM and product code biases: a GIM and its
      satellite DCBs (IONEX ``PRN / BIAS / RMS``) are one consistent pair, the
      code bias of a product (OSB / DSB in ``dcb_path``) has the product's
      satellite DSB datum. Per satellite whose DSB differs from the GIM's by
      1 ns or more from the median difference, 'ion' takes
      (f1/fs)^2 beta (D_product - D_GIM) on signal s (beta = f2^2 / (f1^2 -
      f2^2) of the IONEX code pair): in the dual-frequency constrained PPP
      routes and in
      single-frequency SPP and the single-frequency constrained PPP routes
      (only a signal of the IONEX pair's frequencies, e.g. GPS L1 / L2,
      Galileo E1 / E5a; the CODE and IGS GIMs give no BeiDou DCBs). BRUX
      2024-035, GRG OSB with the COD GIM: the six GPS III satellites 1.25-1.60 m
      on L1; GPS L1 SPP rms 3D 1.444 -> 0.955 m (RTKLIB 0.855 m, which applies
      the IONEX DCBs itself). ``ppp_uduc._align_iono_model_to_code_biases``.

    Satellite attitude (SPP, PPP, TEC; one frame for the
    satellite PCO, the azimuth of the satellite PCV and the phase wind-up):

    - ``satellite_attitude``: ``None`` (default) or ``"model"`` - the ORBEX
      attitude of ``attitude_path`` where it covers a satellite's arc, else
      the published attitude law of the satellite block (``gnx_py.attitude``:
      GPS IIR/IIF/III, Galileo IOV/FOC, BeiDou noon/midnight turns and GEO
      orbit-normal; block from the ANTEX entry, so the laws need
      ``atx_path``; without one the ORBEX attitude still applies where it
      covers an arc and the nominal attitude elsewhere, INFO for None, a
      UserWarning for an explicit "model"). ``"nominal"``: yaw steering
      everywhere and BeiDou GEO orbit-normal, as RTKLIB, for like-for-like
      comparisons with RTKLIB; it reads no ORBEX file (a UserWarning when
      ``attitude_path`` is set) and has no PPP turn guard.
    - ``attitude_laws``: ``{law family: law}`` overrides of the default law,
      e.g. ``{"gps_iii": "iir"}`` (what CODE uses for GPS III) or
      ``{"bds3_cast_meo": "dilssner2017"}`` (``gnx_py.attitude.LAWS``).
    - ``attitude_path``: ORBEX file(s) (``*_ATT.OBX[.gz]``). Used only when
      their analysis centre is the one of ``sp3_path``: centres fly some
      turns the other way round and their clocks absorb the wind-up of their
      own attitude (one cycle, about 10.7 cm of the ionosphere-free phase).
      A file of another centre is not used, with a UserWarning; the rule
      is ``gnx_py.orbex.check_product``. Only PPP can override it
      (``PPPConfig.attitude_allow_mixed_products``): SPP, TEC and SIS
      sessions never use the ORBEX file of another centre. The cost
      is the attitude evaluation of every satellite-epoch (well below 1 % of
      a PPP run) plus about the read time of the ORBEX file once per process.
    """
    # --- I/O parameters
    obs_path: Union[str, Path, None] = None
    nav_path: Optional[Union[str, Path, None]] = None
    sp3_path: Optional[Union[List,None]] = None
    atx_path: Optional[Union[str, Path, None]] = None
    dcb_path: Optional[Union[str, Path, None]] = None
    sinex_path: Optional[Union[str, Path, None]] = None
    gim_path: Optional[Union[str, Path, None]] = None
    # accepted for compatibility and without effect since bug-fix stage 10: every observation file is read by
    # gnx_py.rinex_obs (RINEX 2/3/4, CRINEX, gzip/.Z/bzip2/zip in memory); gfzrnx is no longer run
    use_gfz: Optional[bool]=True
    output_dir: Optional[Union[str, Path]] = None

    # --- systems and signals
    sys: Union[str, Set[str]] = "G"
    gps_freq: Literal["L1", "L2", "L5", "L1L2", "L1L5", "L2L5"] = "L1L2"
    gal_freq: Optional[Literal["E1", "E5a", "E5b", "E1E5a", "E1E5b", "E5aE5b"]] = "E1E5a"
    bds_freq: Optional[
        Literal[
            "B1I",
            "B1C",
            "B2a",
            "B2I",
            "B2b",
            "B3I",
            "B1IB2I",
            "B1IB3I",
            "B1CB2a",
            "B1CB2b",
            "B1CB3I",
            "B2aB2b",
        ]
    ] = "B1IB3I"

    # --- orbits
    screen: Optional[bool] = False
    orbit_type: Literal["broadcast", "precise"] = "precise"
    interpolation_method: str = "lagrange"
    interpolation_degree: Optional[int] = 12
    broadcast_tolerance: Optional[str] = '2H'

    # --- correction models
    # - ionosphere
    ionosphere_model: Union[Literal['gim','klobuchar','ntcm','bdgim'], False] = 'gim'
    # GIM mapping function STEC/VTEC (SPP, PPP and TEC): "auto" follows the IONEX header - the CODE modified
    # single-layer model (H 506.7 km, alpha 0.9782) when DESCRIPTION states it, the single layer at HGT1 for
    # COSZ (IGS, CAS); "slm" / "mslm" force one (gnx_py.ionosphere.ionex.mapping_slm / mapping_mslm)
    # Default since bug-fix stage 9 (S9-D14), chosen in the slant domain: against carrier-to-code levelled STEC,
    # COD, mean of 9 station-days (BRUX/NKLG/NYA2 x 2024-035/131/132), the 10-15 deg bin -3.30 -> 0.00 TECU,
    # k_ep 0.879 -> 0.973, shape 4.27 -> 4.01 TECU. Position-domain cost (stage 9, against the pre-stage-9 GIM
    # handling): SPP with a GIM, 157 rows, median rms_3d +0.175 m, mostly BRUX, almost all from the COD MSLM.
    # Bug-fix stage 13b (R4a, NEW-ION-30 resolved): the MSLM raises the single-frequency SPP height by 0.4-1.4 m
    # at every station, which helps or hurts with the station's other height errors (63 full-day rows: median
    # rms_3d 0.069 m better with "slm", 36 rows better / 27 worse; adding the GIM RMS to the SPP weights changes
    # 0.001 m); no GIM-handling artefact. The slow GIM-constrained PPP convergence of stage 9 (NKLG G UCi t_c10
    # 48.5 -> 113.5 min) was the constrained code weighting (PPP-074, fixed in 13b); the mapping and rotation
    # move those rows by <= 6.5 min. Opt-out: gim_mapping_function="slm".
    gim_mapping_function: Literal["auto", "slm", "mslm"] = "auto"
    # GIM time interpolation between maps: "rotated" (IONEX 1.0 eq. 3, recommended there) or "earth_fixed" (eq. 2)
    # Default since bug-fix stage 9 (S9-D15); applied to grids global in longitude only. Slant domain: IGS shape
    # 4.726 -> 4.631 TECU with rotation (COD -0.005 TECU). Position-domain cost: GIM-constrained PPP at BRUX
    # converges more slowly (GC static UCi t_c10 50.5 -> 80 min, C static UCi t_c10 51 -> 91.5 min, G kinematic
    # UCi t_c20 81 -> 103 min, C kinematic UCi RMS 0.148 -> 0.157 m; R4c kinematic RMS 12.03 -> 12.70 cm with
    # rotation alone, 12.56 cm with both defaults); final positions change by <= 9 mm. Bug-fix stage 13b (R4a,
    # NQ-PPP-32 resolved): with the corrected constrained weighting (PPP-074) the rotation moves the static UCi
    # rows by <= 6.5 min and is equal or faster on the four COD rows; SPP -0.010 m. Opt-out:
    # gim_time_interpolation="earth_fixed".
    gim_time_interpolation: Literal["rotated", "earth_fixed"] = "rotated"
    # BeiDou Klobuchar (ionosphere_model='klobuchar' and TEC compare_models 'klobuchar'; SPP, PPP and TEC):
    # - "gps" (default): the GPS Klobuchar coefficients (GPSA/GPSB, IS-GPS-200) scaled to the BeiDou signal,
    #   the behaviour before bug-fix stage 9;
    # - "icd" (opt-in): the BDS model of BDS-SIS-ICD-B1I 3.0 5.2.4.7 (gnx_py.ionosphere.models.klobuchar_bds)
    #   with, per epoch, the broadcast set in force (RINEX 4 '> ION' records GEO first, the
    #   latest at or before the epoch; RINEX 3 one GEO-first header set); without any it falls back to "gps"
    #   with a warning.
    # Why "gps" is the default (author's decision 2026-10-09, NEW-ION-21): the BDS-2 broadcast coefficients are
    # fitted for the Asia-Pacific service area. Slant RMS against the COD GIM at B1I over 9 station-days
    # (BRUX/NKLG/NYA2 x 2024-035/131/132): GPS-scaled 17.95, BDS 19.58 TECU; BDS worse on 5 of 9 (1 tie, 3
    # better). SPP BeiDou rows: 29 worse, 22 better; NKLG 2024-132 C-only rms_3d 3.08 -> 6.75 m. The global
    # BDS-3 model (BDGIM, BDS-SIS-ICD-B2b-1.0 §7.7) is ionosphere_model='bdgim' (bug-fix stage 10b); this
    # setting applies to the Klobuchar only.
    bds_klobuchar: Literal["gps", "icd"] = "gps"
    # - troposphere
    troposphere_model: Union[Literal['niell','saastamoinen','collins'], False] = 'niell'
    # - satellite PCO
    sat_pco: Union[Literal['los','crd'],False]='los'
    # - reciever PCO
    rec_pco:bool =True
    # satellite without a valid ANTEX entry (PRN within VALID FROM/UNTIL) for a frequency the mode needs
    # (bug-fix stage 11; satellite PCO and PCV, receiver PCV tied to sat_pco / rec_pco): "exclude" (default)
    # drops its observations with one UserWarning naming the satellites, the ANTEX file and the date (e.g.
    # E28/E32 in igs20_2388 on 2026 data); "raise" stops with a ValueError. Only used when sat_pco is enabled
    missing_sat_antenna: Literal["exclude", "raise"] = "exclude"
    # - other corrections:
    windup:bool = True
    rel_path: bool = True
    solid_tides: bool = True
    antenna_h: bool = True
    ev_mask: int = 10
    # cycle slips (bug-fix stage 8c, gnx_py.cycle_slips): False - arcs split at data gaps, and at RINEX
    # loss-of-lock indicators when cycle_slip_aux_carrier is True; no slip tests. True includes the receiver
    # millisecond clock-jump guard whatever the two settings below
    cycle_slips_detection: bool = True
    # LLI and a third carrier of the satellite read from the RINEX 3 file for slip detection (gnss.AUX_SIGNALS;
    # the third carrier at every sampling interval since bug-fix stage 9). False: no LLI, no third carrier
    cycle_slip_aux_carrier: bool = True
    # geometry layer of the slip detection (bug-fix stage 8c, gnx_py.cycle_slips.GeometrySlipDetector): an
    # epoch-differenced test of all satellites with the orbits and clocks of the frame; on with 30 s sampling and
    # orbit_type='precise' (TEC since bug-fix stage 13d also with broadcast orbits, a record switch starting a new
    # piece; otherwise off, one INFO line)
    cycle_slip_geometry: bool = True
    # 30 s RINEX clock files used by the geometry layer only (the PPP clocks stay those of sp3_path); used only
    # when their analysis centre is the one of the SP3 files, otherwise ignored with a WARNING; rows outside their
    # span keep the SP3 clocks, rows inside a gap of a file follow clk_gap_policy (bug-fix stage 13b)
    clk_path: Optional[List] = None
    # rows of a precise-orbit run inside a gap of the clk_path clock files: "drop" (default; RTKLIB pephclk, the
    # rows are left out as an ordinary data gap) or "sp3" (they keep the SP3 clock, the behaviour before bug-fix
    # stage 13b)
    clk_gap_policy: Literal["drop", "sp3"] = "drop"
    # IF datum of the precise BeiDou clocks of BDS-3 satellites: "auto" resolves it
    # once per session (bias product declaration of the clocks' analysis centre, else the broadcast-clock test,
    # else IF(B1I,B3I) with a warning); "B1IB3I" / "B1CB2a" force it. BDS-2 and BDS-3 GEO satellites always use
    # IF(B1I,B3I)
    bds_precise_clock_datum: Literal["auto", "B1IB3I", "B1CB2a"] = "auto"
    min_arc_len:int=10
    # a-priori receiver position [X, Y, Z] ECEF metres (bug-fix stage 10, D3). None: the RINEX header's APPROX
    # POSITION XYZ when it is plausible and agrees with a position from code observations (PPP: within max(10 m,
    # 3 sqrt(p_crd), 4 x spread) static dual-frequency on precise orbits except the close pairs L2L5/E5aE5b/B2aB2b,
    # else 100 m; SPP and TEC: 100 m), else that
    # code position with a warning (a missing or zero header record no longer stops the session); the code position
    # is the marker (antenna minus DELTA H/E/N), for kinematic PPP / moving SPP the start position. Given: used as it is
    approx_position: Optional[List[float]] = None

    # additional parameters
    station_name: Optional[str] = None
    day_of_year: Optional[int] = None
    time_limit:Optional[List[datetime]] = None
    # RINEX 2 observations carry no tracking mode: the RINEX 3 code whose product biases a RINEX 2
    # observable takes, {system: {RINEX 2 observable: RINEX 3 observable}}, e.g. {"G": {"C5": "C5X"}}.
    # It comes before the converter's header block, the station biases, the receiver type and the
    # default pilot tracking (gnx_py.biases.resolve_rinex2_tracking); None: those decide
    rinex2_tracking: Optional[Dict[str, Dict[str, str]]] = None

    # --- site displacement beyond the solid tide (bug-fix stage 11b; added to dx/dy/dz with the solid tide, read
    # by the PPP observation models through tides_los; SPP and TEC compute the columns but do not use them)
    # ocean tide loading: BLQ file of the Onsala service (several stations per file; the station is found by
    # station_name, else the RINEX MARKER NAME, else the observation file name, on the 4-character site code).
    # RTKLIB tide_oload model: 11 constituents, UT arguments. None: off. A station missing from the file: one
    # UserWarning and no ocean tide loading
    ocean_loading_path: Optional[Union[str, Path]] = None
    # IERS 2010 step-2 solid-tide terms (STEP2DIU 31 diurnal + STEP2LON 5 long-period constituents of
    # DEHANTTIDEINEL; K1 radial up to 12 mm). Added to the step-1 tide when solid_tides is True
    solid_tide_step2: bool = True
    # IGS ERP (version 2) file: enables the pole tide (IERS 2010 7.1.4) and UT1 for the ocean-loading arguments.
    # None: no pole tide (UT = UTC)
    erp_path: Optional[Union[str, Path]] = None
    # mean pole of the pole tide: "secular" (IERS 2010 update 2018, used with IGS20 / ITRF2020) or "cubic" (the
    # original IERS 2010 model, as RTKLIB 2.4.2). Used only with erp_path
    pole_tide_mean_pole: Literal["secular", "cubic"] = "secular"

    # --- satellite attitude (bug-fix stage 11c; see the class docstring). One frame per satellite-epoch serves
    # the satellite PCO, the azimuth of the satellite PCV and the wind-up. None = "model": ORBEX (attitude_path, same
    # analysis centre as sp3_path) where it covers an arc, else the attitude law of the ANTEX block; "nominal": yaw
    # steering, BeiDou GEO orbit-normal (RTKLIB)
    satellite_attitude: Optional[Literal["model", "nominal"]] = None
    # {law family: law} overrides of gnx_py.attitude (e.g. {"gps_iii": "iir"}); None: the default laws
    attitude_laws: Optional[Dict[str, str]] = None
    # ORBEX attitude file(s), one path or a list; None: attitude laws only
    attitude_path: Optional[Union[str, Path, List]] = None

    #: settings accepted without effect (since v1.0.0):
    #: name -> (default, what the library does instead). A value other than the default gives one UserWarning at
    #: construction and at assignment; the settings are removed after v1
    _UNUSED_SETTINGS = {
        "interpolation_method": (
            "lagrange", "precise orbits are always interpolated with Lagrange polynomials through "
                        "interpolation_degree SP3 nodes, the SP3 clocks linearly"),
        "broadcast_tolerance": (
            "2H", "a broadcast record is taken by its Toe nearest to the epoch within the fixed validity window "
                  "of the system (GPS 2 h, Galileo 3 h, BeiDou 2 h; Galileo prefers the records already "
                  "transmitted at the epoch)"),
        "output_dir": (
            None, "the sessions write no files; they return their results (e.g. PPPResult.to_netcdf, "
                  "DataFrame.to_csv)"),
    }

    def _warn_if_unused(self, name: str, value: Any) -> None:
        spec = self._UNUSED_SETTINGS.get(name)
        if spec is not None and not _same_value(value, spec[0]):
            _warn_unused_setting(name, value, spec[1])

    def __post_init__(self) -> None:  # noqa: D401 – simple phrase is OK
        # every setting is checked (and sp3_path/sys normalised) once here; afterwards
        # __setattr__ applies the same checks to every assignment
        self._validate_fields()
        self._validate_cross()
        for name in self._UNUSED_SETTINGS:
            if hasattr(self, name):
                self._warn_if_unused(name, getattr(self, name))

        has_processing_inputs = any(
            value
            for value in (
                self.obs_path,
                self.nav_path,
                self.sp3_path,
                self.atx_path,
                self.gim_path,
                self.sinex_path,
            )
        )
        if has_processing_inputs:
            self._check_consistency()
        object.__setattr__(self, "_gnx_checked", True)

    # -------- validation --------
    def _field_checks(self) -> Dict[str, Callable[[Any], Any]]:
        """Checker per setting: takes the value, returns the (normalised) value or raises ValueError."""
        modes = SUPPORTED_MODES_BY_SYSTEM
        # on/off switches (``bool`` / ``Optional[bool]`` fields): a string such as 'False' is truthy
        checks = {name: (lambda v, n=name, o=optional: check_flag(n, v, allow_none=o))
                  for name, optional in _flag_fields(type(self))}
        checks.update(self._model_checks(modes))
        return checks

    def _model_checks(self, modes) -> Dict[str, Callable[[Any], Any]]:
        return {
            "sp3_path": lambda v: as_path_list("sp3_path", v),
            "clk_path": lambda v: as_path_list("clk_path", v),
            "clk_gap_policy": _choice("clk_gap_policy", CLK_GAP_POLICIES),
            "bds_precise_clock_datum": _choice("bds_precise_clock_datum", BDS_PRECISE_CLOCK_DATUMS),
            "approx_position": _check_approx_position,
            "sys": self._check_sys,
            "gps_freq": lambda v: v if v is None else check_choice("gps_freq", v, modes["G"]),
            "gal_freq": lambda v: v if v is None else check_choice("gal_freq", v, modes["E"]),
            "bds_freq": lambda v: v if v is None else check_choice("bds_freq", v, modes["C"]),
            "orbit_type": _choice("orbit_type", ORBIT_TYPES),
            "interpolation_degree": _number("interpolation_degree", ge=2, integer=True),
            "ionosphere_model": _choice("ionosphere_model", IONOSPHERE_MODELS, notes=_MODEL_NOTES),
            "gim_mapping_function": _choice("gim_mapping_function", GIM_MAPPING_FUNCTIONS),
            "gim_time_interpolation": _choice("gim_time_interpolation", GIM_TIME_INTERPOLATIONS),
            "bds_klobuchar": _choice("bds_klobuchar", BDS_KLOBUCHAR_MODELS),
            "troposphere_model": _choice("troposphere_model", TROPOSPHERE_MODELS),
            "sat_pco": _choice("sat_pco", SAT_PCO_MODES),
            "missing_sat_antenna": _choice("missing_sat_antenna", MISSING_SAT_ANTENNA_POLICIES),
            "ev_mask": _number("ev_mask", ge=0, le=90, unit="deg"),
            "min_arc_len": _number("min_arc_len", ge=0),
            "day_of_year": _number("day_of_year", ge=1, le=366, integer=True, allow_none=True),
            "time_limit": self._check_time_limit,
            "rinex2_tracking": self._check_rinex2_tracking,
            "ocean_loading_path": lambda v: _optional_path("ocean_loading_path", v),
            "erp_path": lambda v: _optional_path("erp_path", v),
            "pole_tide_mean_pole": _choice("pole_tide_mean_pole", POLE_TIDE_MEAN_POLES),
            "satellite_attitude": lambda v: v if v is None else check_choice("satellite_attitude", v,
                                                                             SATELLITE_ATTITUDE_MODES),
            "attitude_laws": _check_attitude_laws,
            "attitude_path": lambda v: as_path_list("attitude_path", v),
        }

    @staticmethod
    def _check_rinex2_tracking(value: Any) -> Optional[Dict[str, Dict[str, str]]]:
        """``rinex2_tracking``: None or {'G'|'E': {RINEX 2 observable: RINEX 3 observable of its band}}."""
        if value is None:
            return None
        from collections.abc import Mapping
        from ..models.biases import rinex2_setting_problem

        shape = "a dict {system: {RINEX 2 observable: RINEX 3 observable}}, e.g. {'G': {'C5': 'C5X'}}"
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"rinex2_tracking must be None or {shape} (got {value!r})")
        checked: Dict[str, Dict[str, str]] = {}
        for system, codes in value.items():
            check_choice("rinex2_tracking system", system, ("G", "E"),
                         notes={"c": "BeiDou has no RINEX 2 observations (RINEX 3.02 onwards)"})
            if not isinstance(codes, Mapping) or not codes:
                raise ValueError(f"rinex2_tracking[{system!r}] must be a non-empty dict {{RINEX 2 observable: "
                                 f"RINEX 3 observable}}, e.g. {{'C5': 'C5X'}} (got {codes!r})")
            for rinex2, rinex3 in codes.items():
                problem = rinex2_setting_problem(system, rinex2, rinex3)
                if problem:
                    msg = f"rinex2_tracking[{system!r}][{rinex2!r}] = {rinex3!r}: {problem}"
                    if (isinstance(rinex2, str) and isinstance(rinex3, str)
                            and not rinex2_setting_problem(system, rinex2.upper(), rinex3.upper())):
                        msg += f"; did you mean {{{rinex2.upper()!r}: {rinex3.upper()!r}}}?"
                    raise ValueError(msg)
            checked[system] = dict(codes)
        return checked

    @staticmethod
    def _check_sys(value: Any) -> Union[str, Set[str]]:
        selected = _check_systems(value, "sys")
        return selected if not isinstance(value, str) or len(selected) > 1 else next(iter(selected))

    @staticmethod
    def _check_time_limit(value: Any) -> Optional[List[datetime]]:
        if value is None:
            return None
        # [start, end] with start <= end, kept as datetimes (strings are accepted)
        return [t.to_pydatetime() for t in check_time_limit(value, "time_limit")]

    def _validate_fields(self) -> None:
        for name, check in self._field_checks().items():
            if hasattr(self, name):
                object.__setattr__(self, name, check(getattr(self, name)))

    def _selected_systems(self) -> set:
        return {self.sys} if isinstance(self.sys, str) else set(self.sys)

    def _validate_cross(self, changed: Optional[str] = None) -> None:
        """Checks involving several settings; ``changed`` is the setting just assigned (None: all)."""
        if changed in (None, "sys", "gps_freq", "gal_freq", "bds_freq"):
            for system in sorted(self._selected_systems()):
                name = _FREQ_SETTING[system]
                mode = getattr(self, name)
                check_choice(name, mode, SUPPORTED_MODES_BY_SYSTEM[system], suffix=f" for system {system!r}")

    #: settings whose assignment re-runs ``_validate_cross``
    _CROSS_SETTINGS = frozenset({"sys", "gps_freq", "gal_freq", "bds_freq"})

    def _check_unknown_attribute(self, name: str) -> None:
        """Hook for configurations that reject misspelled attribute names (see TECConfig)."""

    def __setattr__(self, name: str, value: Any) -> None:
        # plain assignment while the dataclass __init__ runs; checked afterwards
        if not self.__dict__.get("_gnx_checked", False):
            object.__setattr__(self, name, value)
            return
        self._set_checked(name, value)

    def _set_checked(self, name: str, value: Any) -> None:
        check = self._field_checks().get(name)
        if check is not None:
            value = check(value)
        else:
            self._check_unknown_attribute(name)
        self._warn_if_unused(name, value)
        old = self.__dict__.get(name, _UNSET)
        object.__setattr__(self, name, value)
        if name in self._CROSS_SETTINGS:
            try:
                self._validate_cross(changed=name)
            except Exception:
                if old is _UNSET:
                    object.__delattr__(self, name)
                else:
                    object.__setattr__(self, name, old)
                raise

    def check_input_files(self) -> None:
        """FileNotFoundError ``"<setting>: file not found: <path>"`` when a given input path does not exist."""
        check_paths_exist({name: getattr(self, name, None) for name in INPUT_PATH_SETTINGS})

    def validate_for_run(self, required: tuple = ("obs_path",)) -> None:
        """Complete check before a session reads any file.

        Re-checks every setting (also those changed after construction), requires the
        ``required`` paths, runs the model/orbit/PCO consistency checks (also when no
        path was given at construction) and checks that every given input file exists.
        """
        self._validate_fields()
        self._validate_cross()
        for name in required:
            if not getattr(self, name, None):
                raise ValueError(f"{name} is required to run a session (got {getattr(self, name, None)!r})")
        self._check_consistency()
        self.check_input_files()
        self._check_tide_files()
        self._check_attitude_files()
        self._check_bdgim_nav()
        if self.rinex2_tracking and not getattr(self, "dcb_path", None):
            import warnings

            warnings.warn("rinex2_tracking is not used: it selects the product biases of RINEX 2 observables "
                          "and no bias product is given (dcb_path)", UserWarning, stacklevel=3)

    def _check_bdgim_nav(self) -> None:
        """BDGIM (``ionosphere_model='bdgim'`` or ``compare_models`` 'bdgim'): the
        ionosphere table of ``nav_path`` must hold a plausible BDGIM set ('> ION Cxx CNVX', RINEX 4), checked before
        the session reads the observations, so a RINEX 2/3 file or a file with only rejected records raises the
        ValueError (the file, its RINEX version, the rejected records and the hint) at once. The table is cached:
        the session does not read it again. An unreadable file is left to the reader of the session."""
        model = getattr(self, "ionosphere_model", None) == "bdgim"
        compare = "bdgim" in (getattr(self, "compare_models", None) or ())
        nav_path = getattr(self, "nav_path", None)
        if not (model or compare) or not nav_path or not isinstance(nav_path, (str, Path)):
            return
        import numpy as np

        from ..io.rinex_nav import RinexNavError, _iono_cached
        from .tools import _bdgim_rows, bdgim_missing_message

        try:
            table = _iono_cached(nav_path)[0]
        except (OSError, RinexNavError):
            return
        alpha, info = _bdgim_rows(table, np.array([], dtype="datetime64[ns]"))
        if alpha is not None:
            return
        if model:
            what, alternatives = "ionosphere_model='bdgim'", "ionosphere_model='klobuchar' / 'gim'"
        else:
            what, alternatives = "compare_models 'bdgim'", "drop 'bdgim' from compare_models"
        raise ValueError(bdgim_missing_message(nav_path, info, what, alternatives))

    def _check_tide_files(self) -> None:
        """Parse the BLQ and ERP files before the session reads anything else: a malformed
        file raises ValueError naming it and the line. Cached, so the preprocessing does not read them again.
        The station is looked up here too (``station_name``, else the RINEX MARKER NAME, else the observation
        file name): a name that matches several BLQ blocks raises ValueError, and a station missing from the file
        gives the one UserWarning (the preprocessing then applies no ocean tide loading without warning again)."""
        if getattr(self, "ocean_loading_path", None):
            from ..models.ocean_loading import resolve_station, warn_missing_station_once

            station, blq, names = resolve_station(self)
            # no station_name and no observation file yet (TECSession requires obs_path only at the start of
            # run()): the preprocessing looks the station up
            if station is None and names:
                warn_missing_station_once(self, self.ocean_loading_path, blq, names, stacklevel=4)
        if getattr(self, "erp_path", None):
            from ..io.erp import read_erp

            read_erp(self.erp_path)

    def _check_attitude_files(self) -> None:
        """Read the ORBEX file(s) of ``attitude_path`` before the session reads anything else:
        a malformed file, another time system or an inertial frame raises ValueError naming it; a file of another
        analysis centre than ``sp3_path`` gives the one UserWarning (it is then not used). Cached, so the
        preprocessing does not read it again. With ``satellite_attitude='nominal'`` the file is not read (one
        UserWarning says it is unused)."""
        if not getattr(self, "attitude_path", None):
            return
        if getattr(self, "satellite_attitude", None) == "nominal":
            from .tools import _warn_attitude_path_unused

            _warn_attitude_path_unused(self, stacklevel=4)
            return
        from .tools import orbex_for_config

        orbex_for_config(self, stacklevel=4)

    def _check_consistency(self) -> None:
        """Paths required by the selected models/orbits (ValueError naming the missing path)."""
        def missing(path: str, reason: str) -> ValueError:
            return ValueError(f"{path} is required for {reason} (got {getattr(self, path)!r})")

        if self.ionosphere_model == "gim" and not self.gim_path:
            raise missing("gim_path", "ionosphere_model='gim' (IONEX file)")
        if self.ionosphere_model in {"klobuchar", "ntcm", "bdgim"} and not self.nav_path:
            raise missing("nav_path", f"ionosphere_model={self.ionosphere_model!r} (navigation file)")

        if self.orbit_type == "broadcast" and not self.nav_path:
            raise missing("nav_path", "orbit_type='broadcast' (navigation file)")

        if self.screen and not self.nav_path:
            raise missing("nav_path", "screen=True (orbit screening needs the navigation file)")
        if (self.sat_pco or self.rec_pco) and not self.atx_path:
            reason = f"sat_pco={self.sat_pco!r}" if self.sat_pco else "rec_pco=True"
            raise missing("atx_path", f"{reason} (ANTEX file)")

        if self.orbit_type =='precise' and not self.sp3_path:
            raise missing("sp3_path", "orbit_type='precise' (SP3 file)")
    def system_includes(self, code: str) -> bool:
        """Return *True* if the chosen constellation set includes *code* ("G"/"E"/"C")."""
        return code in self.sys if isinstance(self.sys, set) else self.sys == code

    def session_antenna_model(self, processor, obs_frames, systems):
        """Antenna model of one session (``gnx_py.antex.AntennaModel``), or None.

        Built once from ``atx_path`` with the receiver antenna of the observation header (``ANT # / TYPE``
        read by column: the processor's cached header, ``rinex_obs.read_header(obs_path).georinex_view()`` as a
        fallback),
        the satellite entries valid at the first observation epoch (minimum of the observation frames
        ``obs_frames`` of ``systems``, consistent with the RINEX 2 header overrides; not TIME OF FIRST OBS) and
        the PRN changes before the last one (``prn_changes``),
        the policy ``missing_sat_antenna`` and, for precise orbits only, the SP3 'PCV:' label for the
        version check. None without ``atx_path`` or with ``sat_pco`` and ``rec_pco`` both off (the
        preprocessing then applies no antenna term).
        """
        if not self.atx_path or not (self.sat_pco or self.rec_pco):
            return None
        from ..models.antex import build_antenna_model

        header = None
        try:
            header = processor._get_obs_header() if processor is not None else None
        except Exception:  # noqa: BLE001 - fall back to the observation reader's header below
            header = None
        if not header and self.obs_path:
            try:
                from ..io.rinex_obs import read_header

                # the georinex-compatible view of the header (true version, ANT # / TYPE by column; bug-fix
                # stage 10: no gr.rinexheader, which reports CRINEX files as version 3.0 with no system)
                header = read_header(self.obs_path).georinex_view()
            except Exception:  # noqa: BLE001 - no receiver antenna: PCO/PCV 0 with a warning
                header = {}
        epochs, ends = [], []
        for frame in obs_frames or ():
            names = getattr(getattr(frame, "index", None), "names", None) or []
            if "time" in names and len(frame):
                epochs.append(frame.index.get_level_values("time").min())
                ends.append(frame.index.get_level_values("time").max())
        epoch0 = min(epochs) if epochs else None
        sp3 = self.sp3_path if self.orbit_type == "precise" else None
        # the session end: a PRN whose satellite has no entry from an epoch before it on (reassigned during the
        # session) is warned once per run and its later observations are dropped by every system's preprocessing
        # (DDPreprocessing.apply_prn_change_rule; decision D9 of bug-fix stage 13, gnx_py.antex)
        return build_antenna_model(self.atx_path, header if header is not None else {}, epoch0, sorted(systems),
                                   missing_policy=self.missing_sat_antenna, sp3_path=sp3,
                                   epoch_end=max(ends) if ends else None, notes_owner=self)



@dataclass()
class TECConfig(Config):
    """Configuration object for ionosphere STEC/TEC processing.

    Status:
        Active public configuration class for ``gnx_py.ionosphere``. It is the
        recommended user-facing entry point for ``TECSession`` and
        ``STECMonitor`` workflows.

    Purpose:
        ``TECConfig`` extends the shared GNSS ``Config`` with settings specific
        to slant TEC measurement, receiver/satellite bias handling, optional
        model comparison and GIM-based calibration. ``TECSession`` currently
        processes one constellation at a time; select the constellation with
        ``sys`` or pass ``use_sys`` to ``TECSession``.

    Main inherited routing fields:
        ``sys`` selects GPS (``"G"``), Galileo (``"E"``) or BeiDou
        (``"C"``). ``gps_freq``, ``gal_freq`` and ``bds_freq`` select the
        code/phase pair used to form geometry-free ``P4/L4`` observables.
        ``orbit_type`` chooses precise SP3 interpolation or broadcast NAV
        interpolation. ``ionosphere_model`` selects the background model
        (``"gim"``, ``"klobuchar"``, ``"ntcm"``, ``"bdgim"`` or ``False``;
        ``"bdgim"`` needs a RINEX 4 ``nav_path`` with '> ION Cxx CNVX'
        records, see ``Config``). For BeiDou,
        ``bds_klobuchar`` selects the Klobuchar of the background model and of
        ``compare_models`` 'klobuchar': ``"gps"`` (default, GPS coefficients
        scaled to the BeiDou signal) or ``"icd"`` (opt-in, BDS-SIS-ICD-B1I
        model with BDSA/BDSB; the regional BDS-2 coefficients were worse than
        GPS-scaled on 5 of 9 station-days, see ``Config``).
        ``gim_mapping_function`` / ``gim_time_interpolation`` (defaults
        ``"auto"`` / ``"rotated"``) set the GIM slant TEC of the background,
        the receiver DCB calibration and ``compare_models`` 'gim'; their
        position-domain cost in SPP/PPP is described in ``Config``.

    STEC parameters:
        Carrier-to-code levelling (``STECMonitor``): the levelled phase is
        ``L4 + sum w (P4 - L4) / sum w`` with weights ``sin^4(E)``
        (``gnx_py.ionosphere.config.leveling_weights``). ``leveling_ws=None``
        (default) levels over the whole continuous arc (one offset per arc,
        Mannucci et al. 1998; Ciraolo et al. 2007); an integer levels over a
        centred window of that many samples (opt-in: a 15-minute window keeps
        about 0.9 TECU of code multipath on GPS L1/L2). ``median_leveling_ws``
        does the same for the median variant ``L4 + median(P4 - L4)``.
        ``tec_zenith_err`` is deprecated: a constant factor of the weights,
        it cancels in the levelling. ``sg_window`` / ``sg_polyorder`` and
        ``cheby_*`` set the Savitzky-Golay and Chebyshev filters of the
        levelled TEC (columns ``leveled_tec_sg`` / ``leveled_tec_cheby``).
        ``calibration_min_elev`` is the elevation cutoff of the receiver DCB
        calibration against the GIM. ``irls_timescale`` is a
        reserved/advanced timescale parameter; validate before relying on it in
        new workflows.

    Bias policy:
        ``stec_bias_enabled`` enables the common STEC bias policy.
        ``stec_bias_sources`` defines fallback order using ``"product"``
        (OSB/DSB columns), ``"gim"`` (IONEX DCB columns), ``"config"``
        (manual ns values) and ``"zero"``. ``stec_missing_bias`` controls
        missing-bias behavior: warn and use zero, silently use zero, or raise.
        ``add_dcb`` enables satellite bias correction. ``add_sta_dcb`` enables
        receiver/station bias correction. ``rcv_dcb_source`` selects station
        bias source policy: ``"gim"``, ``"calibrate"``, ``"defined"`` or
        ``"none"``. ``define_satellite_dcb`` and ``define_station_dcb`` accept
        scalar ns values or dictionaries keyed by SV/system/station/default.
        The satellite and station biases of an arc refer to one datum: a
        product station bias goes only with product satellite biases, the
        IONEX receiver DCB only with the IONEX satellite DCBs (otherwise the
        next station source applies). ``rcv_dcb_source="gim"`` therefore
        tries the IONEX satellite DCBs first when its receiver DCB applies to
        the pair; ``"calibrate"`` uses the satellite biases of the chain.
        See ``gnx_py/ionosphere/STEC_BIAS_USAGE.md``.

    Model and diagnostics:
        ``compare_models`` may request additional ``klobuchar``/``ntcm``/``bdgim``/``gim``
        comparison columns (``gim`` needs ``gim_path``, the others ``nav_path``;
        ``bdgim`` a RINEX 4 one with '> ION Cxx CNVX' records). This path is experimental and should be validated
        against known products before scientific use. ``use_iono_rms`` controls
        use of GIM RMS where available. ``skip_sat`` may exclude satellites from
        TEC processing.

    Antenna model:
        With ``sat_pco`` enabled, a satellite without a valid ANTEX entry for
        both frequencies of the pair follows ``missing_sat_antenna`` as in
        PPP/SPP: ``"exclude"`` (default) drops its whole arcs from the TEC
        output with one UserWarning naming the satellites, ``"raise"`` stops.
        TEC sees the satellite term through the geometry-free difference
        of the two frequencies (line-of-sight PCO and PCV), which is not
        small for every block. IGS20 z-offset differences: GPS IIR/IIF 0,
        GPS III 0.48-0.51 m (about 4.8 TECU on L1L2), Galileo E1-E5a FOC
        0.04-0.18 m (median 0.12 m, about 0.9 TECU), BeiDou B1I-B3I -0.18
        to 0.73 m (BDS-3 CAST median 0.20 m, about 2.4 TECU; BDS-2 IGSO
        0.73 m, about 8.6 TECU). With an ANTEX older than the data whole
        satellites may be lost (paper day BRUX 2026-213, E1E5a,
        IGS20_2388: E28/E32 dropped, 30 -> 28 satellites, -6.9 % rows).
        Use an ANTEX that covers the data; ``sat_pco=False`` keeps every
        satellite but leaves the satellite term in the STEC of all of
        them (the earlier text gave 0.03 TECU).

    Cycle slips:
        Arcs come from the layered detector shared with PPP
        (``gnx_py.tools.CSDetector``): ``cycle_slips_detection`` (master
        switch), ``cycle_slip_aux_carrier`` (RINEX LLI and a third carrier),
        ``cycle_slip_geometry`` and ``clk_path`` (geometry layer, 30 s
        sampling, precise orbits or, since v1.0.0, broadcast
        orbits with a new piece at each record switch; the frame
        phases already carry the
        wind-up, so the layer does not apply it again). TEC runs one
        system, so its geometry clock has no other system to help it. An
        undetected equal slip moves L4 by 0.45 TECU per cycle on L2L5 /
        E5aE5b and 0.51 TECU on L1L2. Since v1.0.0 the third-carrier
        layer runs at every sampling interval, so TEC sessions sampled above
        60 s use it too.

    Supported systems:
        GPS and Galileo are the historical paths. BeiDou is supported for the
        configured BDS modes, with ``B1IB3I`` currently the best-tested STEC
        pair in the repository tests. Bias-product availability remains the
        limiting factor for some BDS products.

    Limitations:
        Changing leveling windows, bias fallback order or station calibration
        changes the measurement model and can affect STEC values. Treat those
        fields as numerical configuration, not cosmetic options.
        The site-displacement settings inherited from ``Config``
        (``solid_tides``, ``solid_tide_step2``, ``ocean_loading_path``,
        ``erp_path``) only fill the preprocessing columns dx/dy/dz and
        ``tides_los``; the geometry-free STEC does not use them.
    """
    leveling_ws: Optional[int] = None
    median_leveling_ws: Optional[int] = None
    tec_zenith_err: Optional[Union[float, int]] = None
    irls_timescale: Optional[Union[float, int]] = 900
    stec_bias_enabled: bool = True
    stec_bias_sources: tuple[Literal["product", "gim", "config", "zero"], ...] = (
        "product",
        "gim",
        "config",
        "zero",
    )
    stec_missing_bias: Literal["warn_zero", "zero", "raise"] = "warn_zero"
    add_dcb: bool = True
    add_sta_dcb: bool = False
    rcv_dcb_source: Optional[Literal["gim", "calibrate", "defined", "none"]] = "calibrate"
    define_satellite_dcb: Optional[Union[float, Dict[str, float]]] = None
    define_station_dcb: Optional[Union[float, Dict[str, float]]] = None
    compare_models: Optional[Union[bool, list[str]]] = False
    use_iono_rms: Optional[bool] = True
    skip_sat: Optional[Union[bool, list[str]]] = False
    # filters of the levelled TEC (STECMonitor columns leveled_tec_sg / leveled_tec_cheby); sg_window in
    # samples (odd; 31 = about 15 min at 30 s)
    sg_window: int = 31
    sg_polyorder: int = 1
    cheby_cutoff_s: float = 1800.0
    cheby_order: int = 2
    cheby_ripple_db: float = 0.5
    # elevation cutoff [deg] of the receiver DCB calibration against the GIM (rcv_dcb_source='calibrate')
    calibration_min_elev: float = 30.0
    # observation interval set by TECSession.run (minutes)
    interval: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    #: models available for compare_models ('gim': the IONEX file gim_path)
    COMPARE_MODELS = ("klobuchar", "ntcm", "bdgim", "gim")
    STEC_BIAS_SOURCES = ("product", "gim", "config", "zero")
    #: STEC applies the satellite PCO along the line of sight only
    TEC_SAT_PCO_MODES = ("los", False)

    @staticmethod
    def _check_tec_zenith_err(value: Any) -> Any:
        """Deprecated: the levelling weights ``1/(err sin^2 E)^2`` are ``sin^4 E`` up to the constant ``err^-2``,
        which cancels in the weighted mean; a given value is still range-checked."""
        if value is None:
            return None
        value = check_number("tec_zenith_err", value, gt=0)
        import warnings as _warnings
        _warnings.warn("tec_zenith_err is deprecated and has no effect: the levelling weights are sin^4(E) "
                       "(gnx_py.ionosphere.config.leveling_weights); a constant factor cancels in the weighted mean",
                       DeprecationWarning, stacklevel=4)
        return value

    def _field_checks(self) -> Dict[str, Callable[[Any], Any]]:
        checks = super()._field_checks()
        checks.update({
            "min_arc_len": _number("min_arc_len", gt=0),
            # None: the whole continuous arc (default); an integer: a window of that many samples
            "leveling_ws": _number("leveling_ws", ge=1, integer=True, allow_none=True),
            "median_leveling_ws": _number("median_leveling_ws", ge=1, integer=True, allow_none=True),
            "tec_zenith_err": self._check_tec_zenith_err,
            # STEC is the geometry-free combination: only the line-of-sight difference of the two satellite
            # PCOs enters it, K * dPCO_LOS ('los'); shifting the satellite coordinates ('crd') leaves L4
            # unchanged (ION-008), as in PPP
            "sat_pco": _choice("sat_pco", self.TEC_SAT_PCO_MODES, notes={
                "crd": "STEC is the geometry-free combination: only the line-of-sight difference of the two "
                       "satellite PCOs enters it ('los'); shifting the satellite coordinates ('crd') leaves L4 "
                       "unchanged"}),
            "sg_window": _number("sg_window", ge=1, integer=True),
            "calibration_min_elev": _number("calibration_min_elev", ge=0, lt=90, unit="deg"),
            "irls_timescale": _number("irls_timescale", gt=0, allow_none=True),
            "stec_bias_sources": self._check_bias_sources,
            "stec_missing_bias": _choice("stec_missing_bias", ("warn_zero", "zero", "raise")),
            "rcv_dcb_source": _choice("rcv_dcb_source", ("gim", "calibrate", "defined", "none", None)),
            "define_satellite_dcb": lambda v: self._check_dcb_values("define_satellite_dcb", v),
            "define_station_dcb": lambda v: self._check_dcb_values("define_station_dcb", v),
            "compare_models": self._check_compare_models,
            "skip_sat": self._check_skip_sat,
            "sg_polyorder": _number("sg_polyorder", ge=0, integer=True),
            "cheby_cutoff_s": _number("cheby_cutoff_s", gt=0, unit="s"),
            "cheby_order": _number("cheby_order", ge=1, integer=True),
            "cheby_ripple_db": _number("cheby_ripple_db", gt=0, unit="dB"),
        })
        return checks

    @classmethod
    def _check_bias_sources(cls, value: Any) -> tuple:
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError(f"stec_bias_sources must be a tuple of sources from {list(cls.STEC_BIAS_SOURCES)}, "
                             f"e.g. ('product', 'gim', 'zero') (got {value!r})")
        for source in value:
            check_choice("stec_bias_sources entries", source, cls.STEC_BIAS_SOURCES)
        return tuple(value)

    @staticmethod
    def _check_dcb_values(name: str, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            for key, v in value.items():
                check_number(f"{name}[{key!r}]", v, unit="ns", allow_none=True)
            return value
        return check_number(name, value, unit="ns")

    @classmethod
    def _check_compare_models(cls, value: Any) -> Any:
        if value is None or value is False:
            return value
        if value is True:
            raise ValueError(f"compare_models must be False or a list of models from {list(cls.COMPARE_MODELS)} "
                             f"(got True)")
        models = [value] if isinstance(value, str) else value
        if not isinstance(models, (list, tuple)):
            raise ValueError(f"compare_models must be False or a list of models from {list(cls.COMPARE_MODELS)} "
                             f"(got {value!r})")
        for model in models:
            check_choice("compare_models entries", model, cls.COMPARE_MODELS, notes=_MODEL_NOTES)
        return list(models)

    @staticmethod
    def _check_skip_sat(value: Any) -> Any:
        if value is None or value is False:
            return value
        example = "e.g. ['G05', 'G12']"
        if value is True or isinstance(value, str) or not isinstance(value, (list, tuple, set)):
            raise ValueError(f"skip_sat must be False or a list of satellite PRNs, {example} (got {value!r})")
        for sv in value:
            if not isinstance(sv, str):
                raise ValueError(f"skip_sat must hold satellite PRNs such as 'G05' (got {sv!r})")
            if "_" in sv:
                raise ValueError(f"skip_sat must hold satellite PRNs, not arc labels (got {sv!r}); "
                                 f"use {sv.split('_')[0]!r} to skip the satellite")
            if not _re.fullmatch(r"[A-Z]\d{2}", sv):
                msg = f"skip_sat must hold satellite PRNs such as 'G05' (got {sv!r})"
                if _re.fullmatch(r"[A-Z]\d{2}", sv.strip().upper()):
                    msg += f"; did you mean {sv.strip().upper()!r}?"
                raise ValueError(msg)
        return list(value)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        # klobuchar/ntcm/bdgim comparison columns are computed from the broadcast ionosphere coefficients, the
        # 'gim' column from the IONEX file
        models = set(self.compare_models or ())
        if models & {"klobuchar", "ntcm", "bdgim"} and not self.nav_path:
            raise ValueError(f"nav_path is required for compare_models={list(self.compare_models)!r} "
                             f"(broadcast ionosphere coefficients) (got {self.nav_path!r})")
        if "gim" in models and not self.gim_path:
            raise ValueError(f"gim_path is required for compare_models={list(self.compare_models)!r} "
                             f"(IONEX file) (got {self.gim_path!r})")

    def _check_unknown_attribute(self, name: str) -> None:
        if name.startswith("_"):
            return
        known = [f.name for f in _dataclass_fields(self)]
        if name not in known:
            hint = _difflib.get_close_matches(name, known, n=1)
            msg = f"{type(self).__name__} has no setting {name!r}"
            if hint:
                msg += f"; did you mean {hint[0]!r}?"
            raise AttributeError(msg)



class _MissingType:
    """Marker of a preset-managed PPP parameter the user did not set.

    A single instance that survives ``copy``/``deepcopy`` and pickling, so a copied
    configuration still recognises its unset parameters.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __reduce__(self):
        return "MISSING"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


MISSING = _MissingType()
def _merge_preset_with_overrides(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(preset)
    out.update(overrides)
    return out


#: default zenith troposphere random walk of PPP [m^2/s]: RTKLIB stats-prntrop 1e-4 m/sqrt(s),
#: i.e. 6 mm/sqrt(h) (geodetic PPP uses 5-10 mm/sqrt(h)); 3e-7 m^2 per 30 s epoch
Q_TRO_DEFAULT = 1e-8


@dataclass
class PPPConfig(Config):
    """Main configuration object for Precise Point Positioning (PPP).

    Overview:
        ``PPPConfig`` is the user-facing control object for PPP processing in
        GNX-py. It is passed to ``gnx_py.ppp.config.PPPSession``, which uses it
        to load products, preprocess observations, select the concrete PPP
        filter class, run the EKF, and return a ``PPPResult``.

        The class extends ``Config``. That means the constructor accepts both
        general GNSS processing fields (input paths, systems, signal modes,
        orbit and correction models) and PPP-specific fields (combined vs
        uncombined routing, ionospheric constraints, PPP-AR, EKF presets,
        trace output and advanced uncombined tuning).

    Status:
        Active public PPP configuration class. It is the recommended way to
        configure PPP sessions. It is not a legacy wrapper.

    Labels used in these notes:
        Filter routes of the session: R1 = uncombined, two or more systems,
        no ionospheric constraints (``PPPUdGenericMixedGNSS``); R2 =
        uncombined with ionospheric constraints, two or more systems with
        BeiDou, the ``"gec"`` model or broadcast orbits
        (``PPPFilterMultiGNSSIonConstGEC``); R4c = uncombined with
        ionospheric constraints, GPS + Galileo, precise orbits, the
        ``"legacy"`` constraints model (``PPPFilterMultiGNSSIonConst``); R7 =
        uncombined, one system (``PPPUdSingleGNSS``); IF = the combined
        ionosphere-free filters; GRAPHIC = the single-frequency code/phase
        average. UC0 / UCi: uncombined without / with ionospheric
        constraints. ``t_c10`` / ``t_c20``: minutes until the 3D error stays
        below 10 / 20 cm for 20 consecutive epochs. Measured values refer to
        the v1.0.0 validation (stations BRUX, NKLG and NYA2; G GPS, E
        Galileo, C BeiDou; COD / GRG products).

    Common examples:
        Stable GPS/Galileo combined PPP typically uses
        ``positioning_mode="combined"``, ``sys={"G", "E"}``,
        ``gps_freq="L1L2"``, ``gal_freq="E1E5a"``, ``orbit_type="precise"``
        and precise SP3/bias/ANTEX products.

        Single-system BeiDou PPP commonly uses ``sys="C"`` with
        ``bds_freq="B1IB3I"``. Combined and BDS-only uncombined paths are the
        safest starting points; mixed no-constraint and AR cases require extra
        validation.

        Uncombined PPP with ionospheric constraints uses
        ``positioning_mode="uncombined"`` and ``use_iono_constr=True``. The
        observations must contain the ionospheric columns prepared by the
        selected ionosphere workflow, usually based on GIM/STEC processing.

    Parameters inherited from ``Config``:
        obs_path (str | Path | None, optional):
            Observation RINEX/CRX path or an input directory used by the driver
            script. Required for full session execution.

        nav_path (str | Path | None, optional):
            Broadcast navigation file. Required for ``orbit_type="broadcast"``,
            broadcast-based screening, and ionosphere models such as
            ``"klobuchar"``, ``"ntcm"`` or ``"bdgim"`` (RINEX 4 with
            '> ION Cxx CNVX' records, e.g. BRD400DLR).

        sp3_path (list[str | Path] | None, optional):
            One or more precise orbit/clock SP3 product paths. Required when
            ``orbit_type="precise"`` and processing inputs are supplied.

        atx_path (str | Path | None, optional):
            ANTEX file used for satellite/receiver antenna phase-center
            corrections. Required when ``sat_pco`` or ``rec_pco`` is enabled.

        dcb_path (str | Path | list[str | Path] | None, optional):
            Bias product path(s), for example OSB/DCB/BIA files. Recommended
            for uncombined PPP and BeiDou work. Mixed-system and uncombined
            paths are sensitive to consistent bias products.

        sinex_path (str | Path | None, optional):
            SINEX station-coordinate file. When present and the station is
            found, PPP results can be reported against the reference position.

        gim_path (str | Path | None, optional):
            Global ionosphere map or related ionosphere product. Required by
            ``ionosphere_model="gim"`` and by workflows that generate
            ionospheric constraints.

        use_gfz (bool, optional):
            Accepted for compatibility; it has no effect since v1.0.0. Every observation file (RINEX 2/3/4, CRINEX, gzip, .Z,
            bzip2, zip) is read by ``gnx_py.rinex_obs`` in memory, with LLI,
            ``time_limit`` always applied and nothing written next to the
            data; gfzrnx is no longer run.

        approx_position (list[float] | None, optional):
            A-priori receiver position [X, Y, Z] in ECEF metres. None (default): the APPROX POSITION XYZ of the RINEX
            header when it lies on the Earth's surface and agrees with a
            position computed from the code observations (within max(10 m,
            3 sqrt(p_crd), 4 x the spread of the epoch solutions) in static
            dual-frequency PPP on precise orbits; 100 m for single-frequency
            modes, the close-pair modes L2L5/E5aE5b/B2aB2b, broadcast
            orbits, kinematic PPP, SPP and TEC, whose code positions carry
            metre-level biases), else
            the code position with a warning. A missing or zero header
            record therefore no longer stops the run. The code
            position is moved from the antenna to the marker by ANTENNA:
            DELTA H/E/N. A static session takes the median of epoch
            solutions spread over the session (more than 100 m spread is an
            error that names a moving receiver); kinematic PPP (q_crd > 0)
            and SPP of a moving receiver take the start position (first 5
            epoch solutions). Shared by ``SPPConfig`` and ``TECConfig``
            (threshold 100 m there). Set it for a moving receiver whose
            start differs from the header, or when the code solution is
            reported weak.

        output_dir (str | Path | None, optional):
            Not used: the sessions write no files, they return their results
            (``PPPResult.to_netcdf``, ``SPPResult.to_netcdf``, DataFrame
            methods). A value other than None gives a UserWarning; removed
            after v1.

        sys (str | set[str], recommended):
            Active constellations. Use ``"G"`` for GPS, ``"E"`` for Galileo,
            ``"C"`` for BeiDou, or a set such as ``{"G", "E", "C"}``.
            This field is normalized in ``Config.__post_init__`` and is one of
            the main routing inputs.

        gps_freq (str):
            GPS signal mode. Typical values are ``"L1L2"`` for dual-frequency
            PPP or ``"L1"`` for single-frequency. The mode controls whether
            the uncombined router selects single- or dual-frequency filters.

        gal_freq (str | None):
            Galileo signal mode, typically ``"E1E5a"`` for dual-frequency PPP
            or ``"E1"`` for single-frequency. Must be compatible with Galileo
            when ``"E"`` is active.

        bds_freq (str | None):
            BeiDou signal mode. ``"B1IB3I"`` is the current default and the
            best-supported dual-frequency BDS PPP starting point. Other BDS
            modes are accepted when present in GNX-py signal metadata and input
            observations, but may require extra validation.

        screen (bool, optional):
            Enables broadcast-assisted screening before filtering. Requires
            ``nav_path``.

        orbit_type ({"precise", "broadcast"}):
            Selects orbit/clock source. ``"precise"`` uses SP3 interpolation
            and is recommended for high-accuracy PPP. ``"broadcast"`` uses
            navigation data and is useful for validation and lower-dependency
            runs.

        interpolation_method (str):
            Not used: precise orbits are always interpolated with Lagrange
            polynomials (``interpolation_degree`` nodes), the SP3 clocks
            linearly between the two nodes around the epoch. A value other
            than ``"lagrange"`` gives a UserWarning; removed after v1.

        interpolation_degree (int | None):
            Number of SP3 nodes of the Lagrange interpolation of precise
            orbits (the polynomial degree is one less; default 12 nodes, at
            least 2), the window centred on the epoch and moved inward at the
            ends of the arc; the clocks are interpolated linearly. Changing it changes the numerical
            results.

        broadcast_tolerance (str | None):
            Not used: a broadcast record is taken by its Toe nearest to the
            epoch within the fixed validity window of the system (GPS 2 h,
            Galileo 3 h, BeiDou 2 h); for Galileo the nearest Toe among the
            records already transmitted at the epoch is preferred when one
            lies within the window. A value
            other than ``"2H"`` gives a
            UserWarning; removed after v1.

        ionosphere_model ({"gim", "klobuchar", "ntcm", "bdgim"} | False):
            Ionosphere source used during preprocessing. ``"gim"`` requires
            ``gim_path``; ``"klobuchar"``, ``"ntcm"`` and ``"bdgim"`` require
            ``nav_path``. ``"bdgim"``: BeiDou-3 global broadcast model
            (BDS-SIS-ICD-B2b-1.0 §7.7), RINEX 4 ``nav_path`` with '> ION Cxx
            CNVX' records; global, for G/E/C (see ``Config``).
            Set to ``False`` only for workflows that intentionally skip this
            correction/constraint preparation.

        bds_klobuchar ({"gps", "icd"}):
            BeiDou Klobuchar with ``ionosphere_model="klobuchar"``. ``"gps"``
            (default): GPS coefficients scaled to the BeiDou signal, as before v1.0.0. ``"icd"`` (opt-in): BDS-SIS-ICD-B1I 3.0 5.2.4.7
            with the broadcast set in force at each epoch (RINEX 4 '> ION'
            records GEO first; RINEX 3 one GEO-first header set); without any it falls back to ``"gps"`` with a warning. The BDS-2 coefficients are regional
            (Asia-Pacific): slant RMS against the COD GIM at B1I over 9
            station-days 17.95 (GPS-scaled) against 19.58 TECU (BDS), BDS
            worse on 5 of 9. The BDS-3 global model is
            ``ionosphere_model="bdgim"``.

        gim_mapping_function ({"auto", "slm", "mslm"}) and
        gim_time_interpolation ({"rotated", "earth_fixed"}):
            GIM slant delay and constraint. The defaults ``"auto"`` (MSLM for
            CODE) and ``"rotated"`` fit the
            levelled STEC better, but with a GIM constraint rotation slows
            PPP convergence at BRUX 2024-035 (GC static UCi t_c10 50.5 ->
            80 min, C static UCi t_c10 51 -> 91.5 min, G kinematic UCi t_c20
            81 -> 103 min, R4c kinematic RMS 12.03 -> 12.70 cm with rotation
            alone, 12.56 cm with both defaults) and the CODE MSLM at NKLG
            2024-035 (G UCi t_c10 48.5 -> 113.5 min; R4c GE 24 h t_c10 48 ->
            93 min with both defaults); final positions change by <= 9 mm.
            That slowness was the constrained code weighting (fixed in v1.0.0:
            GC 65.5 -> 20 min, C 157.5 -> 38.5 min at BRUX with the COD
            products); with it the four mapping / interpolation combinations
            move the static UCi rows by at most 6.5 min, rotation equal or
            faster on the COD rows. In SPP the MSLM raises the
            single-frequency height by 0.4-1.4 m (median rms_3d 0.069 m better
            with ``"slm"`` over 63 full-day rows, 36 better / 27 worse).
            Opt-outs: ``gim_time_interpolation="earth_fixed"``,
            ``gim_mapping_function="slm"``.

        troposphere_model ({"niell"}):
            Troposphere model used during preprocessing. The PPP filters
            estimate the zenith wet delay with the Niell wet mapping function,
            so PPP accepts only ``"niell"`` (other values raise ``ValueError``).

        sat_pco ({"los"} | False):
            Satellite antenna phase-center correction mode. ``"los"`` (line of
            sight) is the PPP path; ``"crd"`` is rejected because no PPP
            filter reads it. Requires ``atx_path`` when enabled.

        rec_pco (bool):
            Enables receiver antenna phase-center correction. Requires
            ``atx_path`` when enabled.

        missing_sat_antenna ({"exclude", "raise"}):
            A satellite without a valid ANTEX entry for a frequency the mode
            needs: ``"exclude"`` (default) drops its observations with one
            UserWarning naming the satellites, the ANTEX file and the date;
            ``"raise"`` stops with a ValueError.

        windup, rel_path, solid_tides, antenna_h (bool):
            Standard PPP correction switches for phase wind-up, relativistic
            path term, solid tides and antenna height handling. Recommended to
            keep enabled for precise PPP unless validating a controlled model
            difference.

        solid_tide_step2 (bool):
            IERS 2010 step-2 solid-tide terms (frequency dependence of the
            Love numbers, DEHANTTIDEINEL STEP2DIU + STEP2LON; K1 radial up to
            12 mm), added to the step-1 tide when ``solid_tides`` is on.
            Default True since v1.0.0.

        ocean_loading_path (str | Path | None):
            BLQ file of the Onsala ocean loading service (several stations
            per file, FES2014b with CMC recommended). The station is found by
            ``station_name``, else the RINEX MARKER NAME, else the observation
            file name, on the 4-character site code (any case; a 9-character
            id matches). RTKLIB ``tide_oload`` model (11 constituents, UT
            arguments, no nodal modulation), added as an ENU displacement to
            the solid tide. A station missing from the file gives one
            UserWarning and no ocean tide loading. None (default): off.

        erp_path (str | Path | None), pole_tide_mean_pole ({"secular", "cubic"}):
            IGS ERP (version 2) file: enables the pole tide (IERS 2010 7.1.4)
            and UT1 for the ocean-loading arguments. ``pole_tide_mean_pole``:
            ``"secular"`` (IERS 2010 update 2018, IGS20 / ITRF2020; default)
            or ``"cubic"`` (original IERS 2010 model, as RTKLIB 2.4.2). None
            (default): no pole tide. The ocean pole tide loading (IERS 2010
            7.1.5, up to ~2 mm radial) is not modelled.

        ev_mask (int | float):
            Elevation cutoff in degrees. Affects preprocessing and accepted
            observations.

        cycle_slips_detection (bool):
            Master switch of the cycle-slip tests before filtering
            (``gnx_py.tools.CSDetector``). ``False``: arcs are split only at
            data gaps and, with ``cycle_slip_aux_carrier=True``, at RINEX
            loss-of-lock indicators (LLI); every combination test is off.
            ``True`` also removes receiver millisecond clock jumps from the
            detector's copies of the codes (L0 guard, independent of the two
            settings below; a 1 ms code jump no longer breaks every arc).
            Shared by ``TECConfig``.

        cycle_slip_aux_carrier (bool):
            Read the RINEX loss-of-lock indicators and a third carrier of the
            satellite from ``obs_path`` for slip detection (``gnx_py.cycle_slips``; default ``True``). The third carrier
            (``gnss.AUX_SIGNALS``: L1 for L2L5, L5 for L1L2, E1 for E5aE5b,
            E5b for E1E5a, B1C for B2aB2b, L2W/E5a/B3I for single-frequency
            L1/E1/B1I) is tested together with each processed signal, so an
            equal slip (n, n) on a close pair (L2/L5, E5a/E5b, B2a/B2b) and a
            one-cycle slip on a single frequency become detectable without
            products, at every sampling interval (since v1.0.0;
            calibrated at 30 s; above 60 s it costs more spurious arc breaks:
            new ambiguities per satellite-hour up to +0.033 (quiet day,
            300 s) and +0.032 (storm day, 120 s), and a slip on the third
            carrier alone breaks the arc in 3.5-100 % of the cases; at 300 s
            on single frequency such a break is about as frequent as a found
            slip, see ``cycle_slips.AUX_MAX_INTERVAL_S``). Other modes and
            satellites or epochs without the carrier keep today's tests plus
            LLI; RINEX 2 files keep today's tests only. Default
            modes are affected too: GPS L1L2 gets L5 and Galileo E1E5a gets
            E5b where the satellite has them. ``False``: today's tests only,
            no LLI. The side read parses the file once per process (about
            0.6 s for a 30 s station-day with all systems, cached by file
            content). Shared by ``TECConfig``.

        cycle_slip_geometry (bool):
            Geometry layer of the slip detection (``gnx_py.cycle_slips.GeometrySlipDetector``; default ``True``):
            an epoch-differenced test of every carrier of every satellite
            of all selected systems with the orbits and clocks of the frame
            (one receiver-clock change per epoch), ionosphere-weighted. It
            finds a slip on all carriers at once (n, n, n), equal slips
            under a strong ionosphere and one-cycle slips on a single
            frequency; a flag that only one layer raises must show a
            persistent ionosphere-free step (7 sigma, 7 cm). On only with
            ``orbit_type='precise'`` (TEC since v1.0.0 also
            broadcast, a record switch starting a new piece), 30 s sampling
            and a plausible
            a-priori position (otherwise off with one INFO line). An
            a-priori position more than max(5 m, 4 sigma) from the code
            solution is replaced by it (INFO); a static run more than
            100 m off has the layer off (WARNING). Kinematic runs
            (``q_crd > 0``) keep the static model while the receiver does
            not move; a moving receiver is tracked at its integrated
            position, or the layer is off for the run (INFO). About 1.5 s
            per receiver-day. Shared by ``TECConfig``.

        clk_path (list[str | Path] | None):
            30 s RINEX clock files (plain, .gz or .Z). Since v1.0.0 the PPP filter (and SPP) uses their satellite clocks instead
            of the 5-min SP3 clocks, and the geometry layer of the slip
            detection as before. Used only with precise orbits and when the
            analysis centre of the files (first three characters of the
            name, or the ``ANALYSIS CENTER`` header) is the one of the SP3
            files; otherwise the SP3 clocks are kept with a warning. Each
            file is its own series (no interpolation from one day's file to
            the next, 1 s tolerance at its edges); between two nodes at most
            1.5 intervals apart the clock is interpolated linearly, so a
            missing node (COD 2024-035 C48 18:22:30-18:26:30) is not
            bridged: the rows inside such a gap follow ``clk_gap_policy``
            (default ``"drop"``: left out). Rows outside
            the files' span keep the SP3 clocks; the covered share is logged
            (WARNING below 90 %; dropped rows count as not covered), and
            files that cover none of the observations are not used (a
            warning).
            Measured (v1.0.0 validation, COD, 50 configurations at BRUX 2024-035):
            G-IF kinematic 4.45 -> 1.15 cm, STORM G-IF 0.88 -> 0.38 cm,
            median final error -0.03 cm; runs without clk_path are
            unchanged. The 5-min SP3 clocks leave interpolation errors of a
            few cm per epoch (GPS p95 6.1 cm): NYA2 2024-035 L1/L2 (1, 1) is
            found in about 65 % of the cases with them against 99 % with 30 s
            clocks. TEC does not use the satellite clock. Shared by
            ``TECConfig`` (geometry layer only).

        clk_gap_policy ({"drop", "sp3"}):
            Rows inside a gap of the ``clk_path`` clock files (between two
            nodes of a satellite more than 1.5 intervals apart, e.g. COD
            2024-035 C48 18:22:30-18:26:30; column ``clk_gap`` of
            ``rinex_clk.apply_clk_clocks``): ``"drop"`` (default) leaves them out before the filter as an ordinary data
            gap, the RTKLIB ``pephclk`` convention ("prec clock outage": no
            SP3 fallback once clock files are loaded); ``"sp3"`` gives them
            the SP3 clock (the behaviour before v1.0.0: the clock source changes
            inside the arc). Rows outside the files' span keep the SP3 clock
            under both; the covered share counts the dropped rows as not
            covered. SPP and PPP.

        bds_precise_clock_datum ({"auto", "B1IB3I", "B1CB2a"}):
            IF datum of the precise BeiDou clocks of BDS-3 satellites. An analysis centre chooses it: IF(B1I,B3I) in the MGEX
            products of 2024, IF(B1C,B2a) for COD MGEX 2026-213. ``"auto"``
            (default), once per session: the
            SATELLITE_CLOCK_REFERENCE_OBSERVABLES of a bias product of the
            clocks' analysis centre in ``dcb_path``, else a test of the
            precise against the broadcast clocks (``nav_path``; two-sided
            with RINEX 4 B-CNAV TGD_B1Cp/TGD_B2ap; BDS-3 between-satellite
            scatter <= 1.5 m and the other datum 2.5x larger), else
            IF(B1I,B3I) with a warning. BDS-2 and BDS-3 GEO satellites (no
            open-service B1C/B2a) always use IF(B1I,B3I); the generation
            comes from the ANTEX block (PRNs C06-C08 and C11-C14 are BDS-3
            satellites, C01-C04 BDS-3 GEO by 2026), else B-CNAV records,
            else the PRN. Under IF(B1C,B2a) a product OSB is used as given
            when every OSB product in ``dcb_path`` declares that datum, else
            referred to it with the product's OSBs of C1P/C5P (else its DSB
            chain, else the broadcast group delays). A datum other than IF(B1I,B3I) that no
            observed BDS-3 satellite's codes can be moved to (no product bias
            in that datum, no B-CNAV records) stops the run with a ValueError
            naming the remedies; satellites that cannot be moved while
            others can are named in a warning. Measured: SPP C IF(B1I,B3I)
            BRUX 2026-213 COD MGEX 11.43 m -> 2.36 m rms; the 2024 rows are
            unchanged.

        min_arc_len (int):
            Minimum arc length [min] accepted by cycle-slip preprocessing (TEC
            and PPP): an arc with at most ``int(min_arc_len x 60 / interval)``
            samples is dropped, so the default 10 min keeps arcs of more than
            20 samples at 30 s (a 21-sample arc spanning exactly 10 min is
            kept). Larger values are more conservative and may
            reject short arcs.

        station_name (str | None), day_of_year (int | None),
        time_limit (list[datetime] | None):
            Optional metadata and run-window controls used by drivers and data
            loaders.

    PPP routing fields:
        positioning_mode ({"combined", "uncombined", "single"}):
            Main PPP mode selector. ``"combined"`` forms ionosphere-free
            code/phase combinations and routes to combined filters.
            ``"uncombined"`` keeps frequency-specific observables and routes
            by active systems, signal modes and ``use_iono_constr``.
            ``"single"`` is a legacy compatibility alias for uncombined
            single-frequency workflows; prefer ``"uncombined"``.

        use_iono_constr (bool | None):
            For uncombined PPP, selects whether ionospheric pseudo-observation
            constraints are used. ``True`` routes to constrained branches when
            available; ``False`` routes to float/no-constraint branches.
            With ``positioning_mode`` ``"uncombined"`` or ``"single"``,
            ``True`` needs an ionosphere model: ``ionosphere_model=False``
            raises ValueError at construction and on assignment (the
            setting is rolled back). The default is ``True``, so on an
            existing config set ``use_iono_constr=False`` before
            ``ionosphere_model=False``, or pass both to the constructor.

        use_iono_rms (bool | None):
            Adds the per-observation ``ion_rms`` [m] (the GIM RMS map, or the
            modelled error of a broadcast model), when available, to the
            ionospheric constraint: variance = ramp value + ``ion_rms**2``
            [m^2], as RTKLIB adds the variance of its ionosphere model. Before v1.0.0 the variance was ``ion_rms`` times the
            ramp value (a metre times a variance), 2-3x tighter at BRUX and
            slower to converge (C COD static UCi t_c10 224.5 min; with the
            sum 38 min, the same as without ``ion_rms``). A broadcast model
            (Klobuchar, NTCM, BDGIM) gives ``ion_rms`` = hypot(50 % of its
            delay, 5 TECU x its mapping function); Klobuchar and NTCM took the
            50 % alone before v1.0.0.

        sigma_iono_0, sigma_iono_end (float | int | None):
            Initial and final values of the ionospheric constraint ramp.
            Despite the names they are variances [m^2] of the ionospheric
            pseudo-observation (used as given on the diagonal of R, not
            squared): 1.1 is about (1.05 m)^2, 3.0 about (1.73 m)^2.
            Advanced tuning knobs; changing them changes the stochastic model.
            ``sigma_iono_end=None`` (default) takes the default of the route:
            1e4 for single-frequency PPP with constraints
            (``PPPUducSFSingleGNSS``): the ramp is linear in the variance, so
            the GIM is in practice a prior on the satellites in view at the
            start (1.1 m^2 at the first epoch, about 170 m^2 at the second
            with 30 s data) and the solution then rests on the
            ionosphere-free code/phase information, because with one
            frequency the time-correlated GIM error does not average out
            (a GIM prior for every new arc,
            ``PPPUducSFSingleGNSS.SF_GIM_PRIOR_NEW_ARCS``, is off: better at
            BRUX/NKLG, worse at NYA2 and in kinematic L1/E1 runs); 3.0 for every other constrained filter, the legacy
            GPS+Galileo constrained filter (``PPPFilterMultiGNSSIonConst``,
            the R4c route: dual-frequency G+E, precise orbits,
            ``use_iono_constr=True`` and
            ``uncombined_iono_constraints_model="legacy"``) included since v1.0.0 (it took 2.3 before: against
            2.3 with the v1.0.0 GIM handling, 6 h t_c10 BRUX / NKLG / NYA2 46.0 /
            96.5 / 91.0 -> 41.0 / 93.0 / 82.5 min, storm day 62.5 -> 53.0
            min, static finals within 0.06 cm; kinematic BRUX RMS 12.30 ->
            12.56 cm, the one row where 3.0 is worse; with the former GIM
            handling 12.69 -> 12.03 cm). An explicit value applies to every route. The ramp values
            are used as variances [m^2] of the ionospheric pseudo-observation
            (plus ``ion_rms**2`` with ``use_iono_rms``).

        t_end (int | None):
            Duration [min] of the ramp from ``sigma_iono_0`` to
            ``sigma_iono_end`` on every constrained route: the filters divide
            it by the sampling interval in minutes and count epochs with data
            from the first epoch of the run (30 = 60 epochs at 30 s; a data
            gap is not counted). A reset (``reset_every``) does not start the
            ramp again.

        trace_filter (bool | None):
            Enables controlled PPP trace output. Normal library use should keep
            it disabled; debugging output is formatted through PPP trace
            helpers.

        reset_every (int | None):
            Periodic filter reset interval [min]: the filters reset at every
            epoch whose index times the sampling interval [min] is a multiple
            of it (the time since the first epoch when the data have no gap);
            0, the default, disables resets. Advanced validation/debug option.
            The convergence time (``ct_min``) is measured from the latest
            reset: the reset epoch starts the clock again and a position that
            moves after the reset is not converged, so ``ct_min`` is the time
            to settle after the last reset that preceded the final settling.

    PPP-AR fields:
        pppar_enabled (bool):
            Enables PPP ambiguity-resolution attempts where the selected filter
            supports AR (wide-lane / narrow-lane on the uncombined routes).
            Disabled by default. A route that fixes no ambiguity gives the
            float solution and a RuntimeWarning saying so: combined
            (ionosphere-free) PPP, and single-frequency PPP without
            ionospheric constraints (``use_iono_constr=False``; its
            code/phase average has a non-integer ambiguity).

        pppar_warmup_epochs (int):
            Minimum warm-up period before AR attempts.

        pppar_min_ambiguities (int):
            Minimum ambiguity count required for a candidate group.

        pppar_ratio_threshold (float):
            LAMBDA ratio-test threshold. Lower values are more permissive;
            changing this directly affects AR acceptance.

        pppar_constraint_sigma_cycles (float):
            Sigma of the constraint applied to the filter state for accepted
            integer fixes [cycles]. Since v1.0.0 it no
            longer conditions the search: the narrow-lane candidates are
            conditioned on the rounded wide-lane integers with an internal
            1e-3 cycles (``pppar.WIDE_LANE_CONDITION_SIGMA_CYCLES``), so 0.1
            cycles fixes as many epochs as the default (GPS-only uncombined, BRUX 3 h:
            0 -> 73 fixed epochs, the default 73).

        pppar_constraint_sigma_floor_cycles (float | None):
            Lower bound of that applied-constraint sigma [cycles]. ``None``
            disables the floor.

        pppar_min_lock_epochs (int | None):
            Minimum arc age before an ambiguity can be fixed. The arc age is
            the number of epochs of the arc before the current one: 0 at its
            first epoch, also after a slip or a filter reset, so ``N`` makes an ambiguity fixable from the
            (N+1)-th epoch of its arc. ``None`` takes
            ``pppar_warmup_epochs``.

        pppar_min_candidate_elevation_deg (float | None):
            Optional elevation cutoff for AR candidates.

        pppar_use_float_ratio_covariance (bool):
            Uses float covariance for the AR ratio test when supported.
            Advanced option; keep enabled unless validating covariance policy.

        pppar_partial_fixing_enabled (bool):
            Allows fixing mature subsets when some candidates are still too
            young, and (uncombined AR) fixing the subset of decorrelated
            ambiguities whose success rate reaches 0.999 when the full set
            cannot be fixed. Default ``True``; ``False`` fixes all candidates
            of a system at once or none.

        pppar_partial_min_ambiguities (int | None):
            Minimum ambiguity count for partial fixing. Defaults to the full
            ``pppar_min_ambiguities`` policy when ``None``.

        pppar_wide_lane_max_frac_cycles (float | None):
            Wide-lane fractional gate used by uncombined dual-frequency AR.

        pppar_combined_if_ar_enabled (bool):
            Gate of the AR step of the combined ionosphere-free filters
            (``pppar_enabled`` must also be ``True``). That step fixes no
            ambiguity: the ionosphere-free ambiguity is an integer only
            together with the wide lane, which these filters do not separate.
            The session warns and the result is the float solution;
            use ``positioning_mode="uncombined"`` for PPP-AR.

        pppar_fix_and_hold (bool):
            ``False`` (default): the filter stays float; every epoch is fixed
            anew from the float state and reported with the fixed solution, so
            a wrong fix cannot outlive its epoch. ``True``: accepted fixes are
            fed back into the filter as constraints (fix-and-hold).

        pppar_allow_mixed_products (bool):
            ``False`` (default): when the phase OSBs in ``dcb_path`` come from
            another solution than the orbits and clocks in ``sp3_path``
            (solution identifiers of the IGS long file names, e.g.
            ``COD0OPSFIN`` OSB with ``COD0MGXFIN`` orbits), PPP-AR is disabled
            with a RuntimeWarning naming both solutions and the result is the
            float solution: phase biases are consistent only with the clocks
            estimated with them; the AR diagnostics report the exclusion as reason
            ``mixed_products`` (``ar_reason_mixed_products``,
            ``ar_excluded_mixed_products``). ``True`` keeps AR on with the
            warning (on the validation data the AR validation then rejects most
            fixes: BRUX 2024-035, 6 h, G+E constrained, COD0MGXFIN orbits with
            GRG0MGXFIN OSB fixed 83 of 720 epochs, with COD0OPSFIN OSB none,
            against 460 with the matching GRG orbits, clocks and OSB). Files
            without IGS long names are not compared.

    Satellite attitude (``satellite_attitude``,
    ``attitude_laws`` and ``attitude_path`` are described in ``Config``):
        attitude_turn_guard ({"auto", "all", "off"}):
            Without the ORBEX attitude of the orbit product's analysis centre
            the clocks may hold another wind-up than the attitude laws at
            noon/midnight turns, a step of one cycle (about 10.7 cm of the
            ionosphere-free phase). On the 2024 validation days GRG reverses the
            BeiDou-3 SECM turns at |beta| < 3 deg; COD clocks show steps at
            Galileo turns at 0 <= beta <= 0.3 deg that COD's own ORBEX does
            not explain (its yaw follows the laws within 0.4 deg); GRG and
            COD differ by one cycle at a few GPS noon turns. (GRG's GPS IIF
            shadow and GPS IIIA turns differ from the laws by up to 0.2 cycle,
            which the guard does not address.) The guard drops a satellite's
            rows within 35 min of such a turn when |beta| is below the class
            limit (GPS and Galileo 0.5 deg, BeiDou-3 3 deg, BeiDou-2 4 deg,
            never GEO); the next row opens a new arc, so the ambiguity
            restarts. One UserWarning per system gives the counts. The guard
            is part of ``satellite_attitude="model"`` (the default); with
            ``"nominal"`` no row is guarded, whatever this field says.
            ``"auto"`` (default): static PPP (effective ``q_crd == 0``) with
            precise orbits, only the block classes where the product's
            analysis centre is not known to agree with the laws
            (``gnx_py.attitude.AC_GUARD_EXEMPT``; BeiDou-2 of GRG, which flies
            it in yaw steering, while its law is "nominal",
            ``AC_GUARD_EXEMPT_NOMINAL``), and not where an ORBEX of the
            product's own centre covers the arc, except COD Galileo
            (``ORBEX_GUARDED``: COD's ORBEX follows the laws, its clocks do
            not; E-SP3cod-static-IF err3d 6.3 cm with COD's ORBEX unguarded,
            0.6 cm guarded). Kinematic PPP with precise orbits:
            dual-frequency as static (GRG
            flies the BeiDou-3 SECM turns reversed and its clocks hold the
            one-cycle wind-up step; C/EC/GEC GRG kinematic IF RMS 14.8 / 11.0
            / 5.0 -> 5.9 / 1.5 / 2.8 cm at BRUX 2024-035); single-frequency
            only the ``ORBEX_GUARDED`` classes of the product's centre (COD
            Galileo), whose clocks are proven to step at the turns (since v1.0.0:
            E-SP3cod-kin-IF BRUX 2024-035 6 h RMS 0.1848 -> 0.1078 m, p95
            0.3181 -> 0.2445 m; guarding every class takes
            sf-C-B1I-UCi/UC0-kin p95 0.59 -> 1.87 m). A product whose centre
            is not in ``AC_GUARD_EXEMPT`` is guarded in every class.
            ``"all"``: PPP with precise orbits, every class.
            ``"off"``: no guard. Default "auto": over the PPP validation rows
            no row fails because of the attitude handling; on 12
            key rows of BRUX 2024-035 nominal / laws / "auto" / "all" / ORBEX
            fail 4 / 4 / 2 / 2 / 1. The guard helps static solutions and hurts
            weak kinematic ones (it removes satellites at every epoch of the
            window: every non-exempt class guarded in kinematic PPP takes
            sf-C-B1I-UCi/UC0-kin p95 0.59 -> 1.87 m, 0.90 -> 2.34 m), so
            "auto" leaves the other classes of single-frequency kinematic PPP
            alone.

        attitude_allow_mixed_products (bool):
            ``False`` (default): an ORBEX file (``attitude_path``) of another
            analysis centre than ``sp3_path`` is not used (UserWarning naming
            both centres); the laws and the turn guard apply. ``True`` uses it.

    EKF preset and override fields:
        clock_process ({"RW", "WN"} | MISSING):
            Not used: every PPP filter re-estimates the receiver clock at each
            epoch from ``q_dt`` and ``clock_prior_sigma_m``. Filled from the
            mode preset (``"WN"`` combined, ``"RW"`` uncombined) for
            compatibility; an explicit value other than the preset value of
            the mode gives a UserWarning and the setting is removed after v1.

        p_crd, p_dt, p_amb, p_tro, p_isb, p_N, p_iono, p_dcb:
            Initial covariance values for coordinates, receiver clock,
            ambiguities, troposphere, inter-system bias, ambiguity/phase state,
            ionosphere state and receiver/signal bias states. Advanced; changes
            affect convergence and numerical behavior. Variances must be finite
            and >= 0. ``p_isb=None`` (the combined preset) starts each
            inter-system bias with ``p_dt``; an explicit value (also 0.0) is used
            as given. Since v1.0.0 a receiver clock or an
            inter-system bias whose variance exceeds ``clock_prior_sigma_m**2``
            is re-initialised from the code residuals (see
            ``clock_prior_sigma_m``), so ``p_dt`` (1e9 / 9e9 in the presets)
            matters only when it is smaller than that.

        q_crd, q_dt, q_tro, q_amb, q_isb, q_N, q_iono, q_dcb:
            Process noise of the corresponding states. Advanced model knobs.
            Defaults come from the combined/uncombined presets. Every PPP
            filter, combined and uncombined, uses them. Two kinds of unit:

            * ``q_crd`` and ``q_dt``: variances added per epoch [m^2]. They
              switch a state between constant (0.0) and practically free at
              every epoch, where the sampling interval does not matter.
              ``p_crd`` and ``q_crd`` are per-axis variances (the X, Y, Z
              coordinates are independent): ``q_crd=0.0`` is static PPP, a
              value > 0 a kinematic random walk (e.g. ``1e2``). ``q_dt``
              (9e9 combined, 1e9 uncombined) re-estimates the receiver clock
              at every epoch. Only 0.0 and practically free values are
              independent of the sampling rate: a small kinematic random walk
              (e.g. ``q_crd=1e-4``) moves more per hour at 1 s than at 30 s.
            * ``q_tro``, ``q_iono``, ``q_isb``, ``q_dcb``, ``q_amb``:
              random-walk rates [m^2/s]. The filters add ``q * interval``
              (interval in seconds) at every epoch, so a state wanders as
              much per hour at 1 s as at 5 min sampling. After a data gap or
              skipped epochs they add ``q * dt`` with the real time ``dt``
              since the previous filter epoch, as RTKLIB does (``q_crd``,
              ``q_dt`` and states with a per-epoch variance >= 1e4 m^2 are
              not scaled).
              ``q_tro`` (zenith troposphere delay): default 1e-8 m^2/s, i.e.
              3e-7 m^2 per 30 s epoch, 6 mm/sqrt(h) - the RTKLIB default
              (``stats-prntrop`` = 1e-4 m/sqrt(s)); geodetic PPP uses
              5-10 mm/sqrt(h). Before v1.0.0 it was a variance per
              epoch (2.5e-4 m^2 combined, 0.025 m^2 uncombined at 30 s, about
              800 and 80 000 times as much), so the troposphere absorbed the
              height.
              ``q_iono`` (slant ionosphere of each satellite, uncombined
              filters): 10/3600 m^2/s, i.e. 10 m^2 per hour (it was given per
              hour before). An empirical value, about 2800 times the RTKLIB
              ``stats-prniono`` default (1e-3 m/sqrt(s), 1e-6 m^2/s); 1e-4 to
              1e-6 m^2/s gave worse results with the GIM constraint (measured
              sweep), to be re-tuned with the single-frequency/GIM-constraint
              work. ``q_isb`` (inter-system bias) and ``q_dcb``
              (receiver code-bias states): 1/30 and 1e-4/30 m^2/s in the
              uncombined preset (the former 1 m^2 and 1e-4 m^2 per 30 s
              epoch), 0.0 combined. ``q_amb``: 0.0 (constant ambiguities).
              An explicit value is a rate as well: ``q_tro=2.5e-4`` now adds
              7.5e-3 m^2 per 30 s epoch.
            * ``q_N`` (and ``p_N``): not read by any filter (the ambiguities
              use ``p_amb`` / ``q_amb``); an explicit value other than the
              preset value (``q_N``: 0.0; ``p_N`` is in no preset) gives a
              UserWarning, removed after v1.

        clock_prior_sigma_m (float):
            Standard deviation [m] of the receiver clock prior, as RTKLIB
            (``VAR_CLK`` = SQR(100.0), ppp.c:55; ``udclk_ppp``, ppp.c:660-678).
            Finite and > 0. At every epoch at which the predicted clock
            variance is larger than ``clock_prior_sigma_m**2`` (with the
            white-noise clock of the presets, ``q_dt`` 1e9 / 9e9 m^2: every
            epoch) every PPP filter re-initialises the receiver clock: value =
            weighted mean of the epoch's code residuals without the clock
            (weights 1/variance; rejected observations left out; RTKLIB takes
            its SPP clock), variance ``clock_prior_sigma_m**2``, correlations
            0. The same applies once to an inter-system bias that starts with
            ``p_dt`` (``p_isb=None``, mixed combined PPP): the mean of its
            system's code residuals minus that clock. Before v1.0.0
            the clock kept the route's prediction: 0 with variance ``q_dt``
            where the transition is 0 (combined IF, GRAPHIC, single-frequency
            and unconstrained uncombined routes; ``q_dt`` 9e9 m^2 combined),
            the previous epoch's clock with variance P + ``q_dt`` on the
            constrained uncombined routes (R7, R2) with
            ``uncombined_iono_clock_identity=True``; with the
            white-noise ``q_dt`` both are practically unconstrained, and
            together with the rejected rows (R = 1e12, now left out of the
            update) cond(S) reached 3e15. A clock modelled as a random walk
            (``q_dt`` < ``clock_prior_sigma_m**2``) keeps its prediction after
            the first epoch.

    Uncombined observation and constraint tuning:
        uncombined_code_var, uncombined_phase_var:
            Zenith variances [m^2] of one undifferenced code and phase
            observation, scaled by 1/sin(elevation), in the uncombined filters
            without ionospheric constraints. With constraints and
            ``uncombined_iono_legacy_weighting=False`` (the default since v1.0.0) the phase takes ``uncombined_phase_var`` and the
            code ``uncombined_iono_code_var_by_system``. Defaults 1.0 (code,
            1 m) and 1e-4 (phase, 1 cm).

        uncombined_code_prefit_threshold:
            Code prefit residual gate for uncombined screening.

        uncombined_phase_screen:
            Enables phase-residual jump screening in generic uncombined paths.

        uncombined_phase_screen_threshold:
            Threshold for phase residual jump screening.

        uncombined_phase_screen_warmup_epochs:
            Warm-up before phase screening becomes active.

        uncombined_est_rx_dcb:
            Enables receiver signal-bias/DCB datum states in constrained
            single-system uncombined PPP. Recommended for constrained UD
            validation; changing it changes the state vector.

        uncombined_iono_clock_identity:
            Receiver clock transition of the constrained uncombined routes
            (R7, R2, R4c and the mixed two-system filter): True predicts the clock as the previous epoch's
            value with variance P + ``q_dt`` (F = 1), False as 0 with variance
            ``q_dt`` (F = 0, as the other routes). With the white-noise
            ``q_dt`` of the presets (1e9 m^2) both predictions are practically
            free and the clock prior (``clock_prior_sigma_m``) re-initialises
            the clock at every epoch either way, so the option matters only
            with a random-walk ``q_dt`` < ``clock_prior_sigma_m**2``. Measured
            already before v1.0.0: True and False give identical results
            (G GRG static UCi, BRUX 2024-035).

        uncombined_iono_code_prefit_threshold:
            Code gate used by constrained uncombined branches.

        uncombined_iono_phase_screen:
            Enables legacy-style phase jump screening for constrained
            uncombined branches.

        uncombined_iono_phase_screen_threshold:
            Phase jump threshold for constrained uncombined screening.

        uncombined_iono_phase_screen_warmup_epochs:
            Warm-up before constrained phase screening is applied.

        uncombined_iono_phase_screen_legacy_source:
            Uses the legacy phase-screening source/policy. Compatibility option
            for preserving historical constrained behavior.

        uncombined_iono_legacy_weighting:
            ``False`` (default since v1.0.0): the filters
            with ionospheric constraints (R7 ``PPPUdSingleGNSS``, R2
            ``PPPFilterMultiGNSSIonConstGEC``, R4c
            ``PPPFilterMultiGNSSIonConst``) weight code with
            ``c_sys / sin(elevation)`` (``uncombined_iono_code_var_by_system``)
            and phase with ``uncombined_phase_var / sin(elevation)``.
            ``True``: the former weighting ``base + elev_coeff /
            sin(elevation)`` of the ``uncombined_iono_*_var_*`` settings (code
            0.3 + 0.0025/sin(E) m^2), kept to reproduce earlier results. It is
            practically independent of the elevation and 3.3-18x stronger
            than the unconstrained routes' 1/sin(E): with the slant
            ionosphere pinned by the GIM every code is direct geometry
            information, so low-elevation and product code errors (COD C06
            +4.9 m on 2024-035, a BeiDou satellite without DSB) entered the
            float ambiguities with zenith weight and slowed the convergence
            2-6x (BRUX 2024-035, 6 h: C/EC/GC/GEC COD static t_c10
            157.5/130/65.5/48.5 -> 38.5/29/20/18 min, G GRG kinematic t_c20
            103.5 -> 10.5 min).

        uncombined_iono_code_var_by_system (dict):
            Zenith code variance [m^2] per system of the constrained filters
            (``uncombined_iono_legacy_weighting=False``), scaled by
            1/sin(elevation): ``{"G": 0.3, "E": 0.3, "C": 1.0}`` by default.
            Keys from G, E, C; values finite and > 0; a system not listed
            keeps its default (``{"C": 2.0}`` changes BeiDou only; the
            setting then reads ``{"G": 0.3, "E": 0.3, "C": 2.0}``). One value
            does not fit every
            system: 1.0 for all slows Galileo (E GRG static UCi 13 -> 44 min),
            0.3 for all leaves BeiDou slow (C COD 58.5 min); 1.0 m^2 for
            BeiDou allows for the elevation-dependent satellite-induced code
            variations of BDS-2 (2.0 is the measured fallback, EC COD 23.5
            min). Since v1.0.0
            the single-frequency route with ionospheric constraints
            (``PPPUducSFSingleGNSS``) takes it too, whatever
            ``uncombined_iono_legacy_weighting`` (it hard-coded 1.0 m^2 for
            every system before; ``{"G": 1.0, "E": 1.0, "C": 1.0}`` gives that
            weighting back).

        uncombined_iono_code_var_base,
        uncombined_iono_code_var_elev_coeff,
        uncombined_iono_phase_var_base,
        uncombined_iono_phase_var_elev_coeff:
            Components of the legacy elevation-dependent observation weighting
            model used in constrained uncombined PPP with
            ``uncombined_iono_legacy_weighting=True``.

        uncombined_iono_signal_bias_phase_jacobian:
            Advanced switch controlling how receiver signal-bias states enter
            constrained phase equations. Use only when validating the bias
            model. Read by the legacy G+E constrained filter
            (``PPPFilterMultiGNSSIonConst``): ``True`` (default) adds the
            receiver code-bias state of each frequency to that frequency's
            phase rows, ``False`` leaves it out (the ambiguities absorb it).
            Both are consistent models (the Jacobian and the predicted
            observations agree) and give the same positions to a few
            mm.

        uncombined_iono_constraints_model ({"legacy", "gec"}):
            Selects the constrained mixed-system branch. ``"legacy"`` keeps the
            GPS/Galileo reference path where routing allows it. ``"gec"``
            forces the newer generic GPS/Galileo/BeiDou constrained branch for
            mixed systems.

        uncombined_mixed_phase_code_gate_systems (str | list[str] | set[str] | None):
            Optional advanced selector for stricter mixed uncombined phase/code
            gating by system. Use only for validation experiments.

    Validation:
        Every setting is checked at construction and again on every later
        assignment (``cfg.ev_mask = 95`` raises at once). PPP accepts only
        ``troposphere_model="niell"`` and ``sat_pco`` in ``{"los", False}``.
        Assigning ``positioning_mode`` re-applies the EKF preset of the new
        mode (as ``set_mode``); a preset-managed parameter assigned directly
        (``cfg.q_crd = 1e2``) is an explicit override and is kept.
        ``PPPSession`` checks that every given input file exists and that
        ``obs_path`` is set before reading any file.

    Internal fields:
        _overrides:
            Stores user overrides for preset-managed EKF fields.

        _PRESETS:
            Combined/uncombined preset table. This is internal model
            configuration; changing values can alter numerical results.

    Notes and limitations:
        Combined and uncombined PPP are different observation models and should
        not be compared as interchangeable filters.

        GPS/Galileo constrained uncombined PPP has a legacy/reference branch
        (``PPPFilterMultiGNSSIonConst``). The generic G/E/C constrained branch
        is active but should still be validated when changing bias or
        ionosphere settings.

        BeiDou is supported in PPP routing and tests, with ``B1IB3I`` as the
        safest default BDS dual-frequency mode. BDS mixed no-constraint and AR
        cases remain sensitive to product/bias consistency.

        PPP-AR is off by default. Uncombined AR has targeted tests; combined
        ionosphere-free filters fix no ambiguity, even with the
        ``pppar_combined_if_ar_enabled`` gate.
    """

    positioning_mode: Literal["combined", "uncombined", "single"] = "combined"

    # iono / processing (unchanged)
    use_iono_constr: Optional[bool] = True
    use_iono_rms: Optional[bool] = True
    sigma_iono_0: Optional[Union[float, int]] = 1.1
    # None: the route's default (1e4 single frequency with constraints, 3.0 the others; NQ-PPP-22, NQ-PPP-23)
    sigma_iono_end: Optional[Union[float, int]] = None
    t_end: Optional[int] = 30

    trace_filter: Optional[bool] = False
    reset_every: Optional[int] = 0

    # PPP-AR options
    pppar_enabled: bool = False
    pppar_warmup_epochs: int = 60
    pppar_min_ambiguities: int = 4
    pppar_ratio_threshold: float = 2.0
    pppar_constraint_sigma_cycles: float = 1e-3
    pppar_constraint_sigma_floor_cycles: Optional[float] = 1e-3
    pppar_min_lock_epochs: Optional[int] = None
    pppar_min_candidate_elevation_deg: Optional[float] = None
    pppar_use_float_ratio_covariance: bool = True
    pppar_partial_fixing_enabled: bool = True
    pppar_partial_min_ambiguities: Optional[int] = None
    pppar_wide_lane_max_frac_cycles: Optional[float] = 0.25
    pppar_combined_if_ar_enabled: bool = False
    pppar_fix_and_hold: bool = False
    pppar_allow_mixed_products: bool = False
    uncombined_mixed_phase_code_gate_systems: Optional[Union[str, List[str], Set[str]]] = None

    # satellite attitude turn guard and ORBEX of another analysis centre (bug-fix stage 11c; see the docstring)
    attitude_turn_guard: Literal["auto", "all", "off"] = "auto"
    attitude_allow_mixed_products: bool = False

    # KF params: ustawiamy MISSING, żeby wykryć override od użytkownika
    clock_process: Any = field(default=MISSING)  # Literal['RW','WN']
    p_crd: Any = field(default=MISSING)
    p_dt: Any = field(default=MISSING)
    p_amb: Any = field(default=MISSING)
    p_tro: Any = field(default=MISSING)

    q_crd: Any = field(default=MISSING)
    q_dt: Any = field(default=MISSING)
    q_tro: Any = field(default=MISSING)
    q_amb: Any = field(default=MISSING)

    p_isb: Any = field(default=MISSING)
    q_isb: Any = field(default=MISSING)
    p_N: Any = field(default=MISSING)
    q_N: Any = field(default=MISSING)

    p_iono: Any = field(default=MISSING)
    q_iono: Any = field(default=MISSING)

    p_dcb: Any = field(default=MISSING)
    q_dcb: Any = field(default=MISSING)

    # receiver clock prior [m] (RTKLIB VAR_CLK), set again at every epoch of a white-noise clock (bug-fix stage 13b)
    clock_prior_sigma_m: float = 100.0

    # PPP uncombined observation weighting/gating knobs.  The generic values
    # below are still used by no-constraint paths and remain available as a
    # fallback for constrained single-GNSS runs.
    # zenith variances [m^2], scaled by 1/sin(elevation): code 1 m, phase 1 cm (PPP-050: a phase
    # variance of 1 m^2 weighted the carrier phase like a pseudorange)
    uncombined_code_var: float = 1.0
    uncombined_phase_var: float = 1e-4
    uncombined_code_prefit_threshold: float = 30.0
    uncombined_phase_screen: bool = False
    uncombined_phase_screen_threshold: float = 10.0
    uncombined_phase_screen_warmup_epochs: int = 60

    # Single-GNSS uncombined + ionospheric constraints should follow the
    # stable legacy multi-GNSS constrained methodology by default: estimate a
    # receiver signal-bias datum, keep the first code datum fixed, use legacy
    # phase/code weighting, and run phase jump screening.
    uncombined_est_rx_dcb: bool = True
    uncombined_iono_clock_identity: bool = True
    uncombined_iono_code_prefit_threshold: float = 10.0
    uncombined_iono_phase_screen: bool = True
    uncombined_iono_phase_screen_threshold: float = 1.0
    uncombined_iono_phase_screen_warmup_epochs: int = 60
    uncombined_iono_phase_screen_legacy_source: bool = True
    # False since bug-fix stage 13b (PPP-074): code c_sys/sin(E) from uncombined_iono_code_var_by_system [m^2 at
    # zenith], phase uncombined_phase_var/sin(E); True: the former base + coeff/sin(E) weighting below
    uncombined_iono_legacy_weighting: bool = False
    uncombined_iono_code_var_by_system: Dict[str, float] = field(
        default_factory=lambda: {"G": 0.3, "E": 0.3, "C": 1.0})
    uncombined_iono_code_var_base: float = 0.3
    uncombined_iono_code_var_elev_coeff: float = 0.0025
    uncombined_iono_phase_var_base: float = 1e-4
    uncombined_iono_phase_var_elev_coeff: float = 0.0003
    uncombined_iono_signal_bias_phase_jacobian: bool = True
    uncombined_iono_constraints_model: Literal["legacy", "gec"] = "legacy"

    _overrides: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # q_crd, q_dt: variances per epoch [m^2]; q_tro, q_iono, q_isb, q_dcb, q_amb: random-walk rates
    # [m^2/s], multiplied by the epoch interval (bug-fix stage 8b, PPP-060). q_tro = Q_TRO_DEFAULT
    # (RTKLIB stats-prntrop 1e-4 m/sqrt(s)); q_iono, q_isb and q_dcb keep their former values at 30 s
    # sampling (10 m^2 per hour, 1 m^2 and 1e-4 m^2 per 30 s epoch)
    _PRESETS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "combined": {
            "clock_process": "WN",
            "p_crd": 10.0, "p_dt": 9e9, "p_amb": 400, "p_tro": 2.0,
            "q_crd": 0.0,  "q_dt": 9e9, "q_tro": Q_TRO_DEFAULT, "q_amb": 0.0,
            # p_isb None: a mixed-system combined run starts the ISB with p_dt (ppp_gnss)
            "p_isb": None, "q_isb": 0.0, "q_N": 0.0,
            "p_iono": 0.0, "q_iono": 0.0, "p_dcb": 0.0, "q_dcb": 0.0,
        },
        "uncombined": {
            "clock_process": "RW",
            "p_crd": 100, "p_dt": 1e9, "p_tro": 2.0, "p_iono": 10,
             "p_isb": 100, "p_dcb": 1e2, "p_amb": 1e6,
            "q_crd": 0.0, "q_dt": 1e9, "q_tro": Q_TRO_DEFAULT, "q_iono": 10.0 / 3600.0,
            "q_N": 0.0, "q_isb": 1.0 / 30.0, "q_dcb": 1e-4 / 30.0, "q_amb": 0.0,
            "uncombined_code_var": 1.0,
            "uncombined_phase_var": 1e-4,
            "uncombined_code_prefit_threshold": 30.0,
            "uncombined_phase_screen": False,
            "uncombined_phase_screen_threshold": 10.0,
            "uncombined_phase_screen_warmup_epochs": 60,
            "uncombined_est_rx_dcb": True,
            "uncombined_iono_clock_identity": True,
            "uncombined_iono_code_prefit_threshold": 10.0,
            "uncombined_iono_phase_screen": True,
            "uncombined_iono_phase_screen_threshold": 1.0,
            "uncombined_iono_phase_screen_warmup_epochs": 60,
            "uncombined_iono_phase_screen_legacy_source": True,
            "uncombined_iono_legacy_weighting": False,
            "uncombined_iono_code_var_by_system": {"G": 0.3, "E": 0.3, "C": 1.0},
            "uncombined_iono_code_var_base": 0.3,
            "uncombined_iono_code_var_elev_coeff": 0.0025,
            "uncombined_iono_phase_var_base": 1e-4,
            "uncombined_iono_phase_var_elev_coeff": 0.0003,
            "uncombined_iono_signal_bias_phase_jacobian": True,
            "uncombined_iono_constraints_model": "legacy",
        },
    }, init=False, repr=False)

    #: EKF preset parameters read by no filter (since v1.0.0): an
    #: explicit value other than the preset value of the mode (constructor, assignment, set_param) gives one
    #: UserWarning; removed after v1
    _UNUSED_PRESET_SETTINGS = {
        "clock_process": "every PPP filter re-estimates the receiver clock at each epoch from q_dt and "
                         "clock_prior_sigma_m (combined: white noise, uncombined: q_dt 1e9 m^2 per epoch)",
        "p_N": "no filter has a separate ambiguity/phase state; the ambiguities start with p_amb",
        "q_N": "no filter has a separate ambiguity/phase state; the ambiguities use q_amb",
    }

    def _warn_if_unused(self, name: str, value: Any) -> None:
        instead = self._UNUSED_PRESET_SETTINGS.get(name)
        if instead is not None:
            # silent for MISSING (back to the preset) and for the preset value of the current mode (e.g.
            # dataclasses.replace passes every preset value explicitly); p_N is in no preset: any value warns
            preset = self._PRESETS.get(self._effective_positioning_mode(), {})
            if value is not MISSING and not (name in preset and _same_value(value, preset[name])):
                _warn_unused_setting(name, value, instead)
            return
        super()._warn_if_unused(name, value)

    def __post_init__(self):
        # 1) najpierw baza (sprawdza też pola PPP; pola presetu MISSING są pomijane)
        super().__post_init__()

        # 2) zbierz override'y przekazane w konstruktorze
        preset_keys = self._preset_keys()
        for k in preset_keys:
            v = getattr(self, k)
            if v is not MISSING:
                self._overrides[k] = v
                self._warn_if_unused(k, v)
        for k in self._UNUSED_PRESET_SETTINGS.keys() - preset_keys:
            # p_N is in no preset: it keeps MISSING unless given
            self._warn_if_unused(k, getattr(self, k))

        # 3) ustaw effective (preset + overrides)
        self._apply_effective_to_self()

    # -------- public API --------
    def set_mode(self, mode: Literal["combined", "uncombined", "single"], keep_overrides: bool = True) -> None:
        mode = check_choice("positioning_mode", mode, self.POSITIONING_MODES)
        if not keep_overrides:
            self._overrides.clear()
        # the assignment re-applies the preset of the new mode (see _set_checked)
        self.positioning_mode = mode

    def set_param(self, **kwargs: Any) -> None:
        for k in kwargs:
            if k not in self._preset_keys():
                raise KeyError(f"Unknown/unsupported param: {k}")
        checks = self._field_checks()
        checked = {k: (checks[k](v) if k in checks else v) for k, v in kwargs.items()}
        for k, v in checked.items():
            self._warn_if_unused(k, v)
        self._overrides.update(checked)
        self._apply_effective_to_self()

    def clear_override(self, *keys: str) -> None:
        for k in keys:
            self._overrides.pop(k, None)
        self._apply_effective_to_self()

    def effective(self) -> Dict[str, Any]:
        preset = self._PRESETS[self._effective_positioning_mode()]
        return _merge_preset_with_overrides(preset, self._overrides)

    def is_overridden(self, key: str) -> bool:
        """True when the preset-managed ``key`` was set explicitly (constructor, assignment or set_param)."""
        return key in self._overrides

    # -------- internals --------
    def _effective_positioning_mode(self) -> str:
        if self.positioning_mode == "single":
            return "uncombined"
        return self.positioning_mode

    def _preset_keys(self) -> set[str]:
        keys = set()
        for p in self._PRESETS.values():
            keys |= set(p.keys())
        return keys

    def _apply_effective_to_self(self) -> None:
        eff = self.effective()
        for k, v in eff.items():
            # preset values are valid and overrides were checked when set: no re-registration
            object.__setattr__(self, k, v)

    # -------- validation --------
    POSITIONING_MODES = ("combined", "uncombined", "single")
    #: PPP needs the Niell wet mapping function and applies the satellite PCO along the line of sight
    PPP_TROPOSPHERE_MODELS = ("niell",)
    PPP_SAT_PCO_MODES = ("los", False)

    def _field_checks(self) -> Dict[str, Callable[[Any], Any]]:
        checks = super()._field_checks()

        # variances and sigmas must be finite (inf turns the EKF covariance into NaN);
        # a residual gate may be inf (no gating)
        def nonneg(name: str, finite: bool = True, **kw) -> Callable[[Any], Any]:
            return lambda v: v if v is MISSING else check_number(name, v, ge=0, finite=finite, **kw)

        def pos(name: str, finite: bool = True, **kw) -> Callable[[Any], Any]:
            return lambda v: v if v is MISSING else check_number(name, v, gt=0, finite=finite, **kw)

        checks.update({
            "positioning_mode": _choice("positioning_mode", self.POSITIONING_MODES, notes={
                m: ("positioning_mode selects the observation model; static or kinematic coordinates are set "
                    "with q_crd (0.0 = static)") for m in ("static", "kinematic")}),
            "ionosphere_model": _choice("ionosphere_model", IONOSPHERE_MODELS, notes={
                **_MODEL_NOTES, "measure stec": "'measure STEC' is not supported by PPPSession"}),
            "troposphere_model": _choice("troposphere_model", self.PPP_TROPOSPHERE_MODELS, notes={
                **{m: f"PPP estimates the zenith wet delay with the Niell mapping function; {m!r} is not supported"
                   for m in ("saastamoinen", "collins")},
                False: "PPP estimates the zenith wet delay with the Niell mapping function; the troposphere "
                       "cannot be disabled"}),
            "sat_pco": _choice("sat_pco", self.PPP_SAT_PCO_MODES, notes={
                "crd": "the PPP filters read only the line-of-sight satellite PCO ('los')"}),
            "uncombined_iono_constraints_model": _choice("uncombined_iono_constraints_model", ("legacy", "gec")),
            "clock_process": lambda v: v if v is MISSING else check_choice("clock_process", v, ("RW", "WN")),
            "sigma_iono_0": pos("sigma_iono_0", allow_none=True),
            "sigma_iono_end": pos("sigma_iono_end", allow_none=True),
            "t_end": _number("t_end", ge=1, allow_none=True),
            "reset_every": _number("reset_every", ge=0, integer=True),
            "pppar_warmup_epochs": _number("pppar_warmup_epochs", ge=0, integer=True),
            "pppar_min_ambiguities": _number("pppar_min_ambiguities", ge=2, integer=True),
            # the ratio second-best/best is always >= 1: a threshold below 1 would switch the test off
            "pppar_ratio_threshold": _number("pppar_ratio_threshold", ge=1),
            "pppar_constraint_sigma_cycles": _number("pppar_constraint_sigma_cycles", gt=0),
            "pppar_constraint_sigma_floor_cycles": _number("pppar_constraint_sigma_floor_cycles", ge=0,
                                                           allow_none=True),
            "pppar_min_lock_epochs": _number("pppar_min_lock_epochs", ge=0, integer=True, allow_none=True),
            "pppar_min_candidate_elevation_deg": _number("pppar_min_candidate_elevation_deg", ge=0, le=90,
                                                         unit="deg", allow_none=True),
            "pppar_partial_min_ambiguities": _number("pppar_partial_min_ambiguities", ge=2, integer=True,
                                                     allow_none=True),
            # |fractional part| never exceeds 0.5 cycles: a larger gate screens nothing (None disables it)
            "pppar_wide_lane_max_frac_cycles": _number("pppar_wide_lane_max_frac_cycles", ge=0, le=0.5,
                                                       allow_none=True),
            "uncombined_mixed_phase_code_gate_systems": self._check_gate_systems,
            "attitude_turn_guard": _choice("attitude_turn_guard", ATTITUDE_TURN_GUARDS),
            "uncombined_code_var": pos("uncombined_code_var"),
            "uncombined_phase_var": pos("uncombined_phase_var"),
            "uncombined_code_prefit_threshold": pos("uncombined_code_prefit_threshold", finite=False),
            "uncombined_phase_screen_threshold": pos("uncombined_phase_screen_threshold", finite=False),
            "uncombined_phase_screen_warmup_epochs": nonneg("uncombined_phase_screen_warmup_epochs", integer=True),
            "uncombined_iono_code_prefit_threshold": pos("uncombined_iono_code_prefit_threshold", finite=False),
            "uncombined_iono_phase_screen_threshold": pos("uncombined_iono_phase_screen_threshold", finite=False),
            "uncombined_iono_phase_screen_warmup_epochs": nonneg("uncombined_iono_phase_screen_warmup_epochs",
                                                                 integer=True),
            "uncombined_iono_code_var_base": pos("uncombined_iono_code_var_base"),
            "uncombined_iono_code_var_elev_coeff": nonneg("uncombined_iono_code_var_elev_coeff"),
            "uncombined_iono_phase_var_base": pos("uncombined_iono_phase_var_base"),
            "uncombined_iono_phase_var_elev_coeff": nonneg("uncombined_iono_phase_var_elev_coeff"),
            "clock_prior_sigma_m": pos("clock_prior_sigma_m"),
            "uncombined_iono_code_var_by_system": self._check_code_var_by_system,
        })
        # initial covariances and process noises of the EKF states: variances, >= 0
        for name in ("p_crd", "p_dt", "p_amb", "p_tro", "p_N", "p_iono", "p_dcb",
                     "q_crd", "q_dt", "q_tro", "q_amb", "q_isb", "q_N", "q_iono", "q_dcb"):
            checks[name] = nonneg(name)
        # p_isb=None (the combined preset): the ISB starts with the receiver clock variance p_dt
        checks["p_isb"] = nonneg("p_isb", allow_none=True)
        return checks

    #: default of uncombined_iono_code_var_by_system [m^2 at zenith]; its keys are the systems the
    #: constrained filters weight
    CODE_VAR_BY_SYSTEM_DEFAULT = {"G": 0.3, "E": 0.3, "C": 1.0}
    CODE_VAR_SYSTEMS = tuple(CODE_VAR_BY_SYSTEM_DEFAULT)

    @classmethod
    def _check_code_var_by_system(cls, value: Any) -> Dict[str, float]:
        """``uncombined_iono_code_var_by_system``: {system: zenith code variance [m^2]}, keys from G/E/C, values
        finite and > 0; returned as a new dict of floats with every system, a system not given keeping its default.
       """
        from collections.abc import Mapping

        if not isinstance(value, Mapping):
            raise ValueError(f"uncombined_iono_code_var_by_system must be a dict {{system: variance [m^2]}} with "
                             f"systems from {list(cls.CODE_VAR_SYSTEMS)}, e.g. {{'G': 0.3, 'E': 0.3, 'C': 1.0}} "
                             f"(got {value!r})")
        out: Dict[str, float] = dict(cls.CODE_VAR_BY_SYSTEM_DEFAULT)
        for key, var in value.items():
            if key not in cls.CODE_VAR_SYSTEMS:
                raise ValueError(f"uncombined_iono_code_var_by_system: unknown system {key!r}; the constrained "
                                 f"filters weight {list(cls.CODE_VAR_SYSTEMS)}")
            out[key] = float(check_number(f"uncombined_iono_code_var_by_system[{key!r}]", var, gt=0, finite=True,
                                          unit="m^2"))
        return out

    @staticmethod
    def _check_gate_systems(value: Any) -> Any:
        if value is None:
            return None
        items = value.split(",") if isinstance(value, str) else value
        try:
            items = [s.strip() if isinstance(s, str) else s for s in items]
        except TypeError:
            raise ValueError(f"uncombined_mixed_phase_code_gate_systems must be None, 'G,E' or a list of systems "
                             f"(got {value!r})") from None
        items = [s for s in items if s != ""]
        if items:
            _check_systems(items, "uncombined_mixed_phase_code_gate_systems")
        return value

    def _validate_cross(self, changed: Optional[str] = None) -> None:
        super()._validate_cross(changed)
        if changed in (None, "pppar_partial_min_ambiguities", "pppar_min_ambiguities"):
            partial, minimum = self.pppar_partial_min_ambiguities, self.pppar_min_ambiguities
            if partial is not None and partial < minimum:
                raise ValueError(f"pppar_partial_min_ambiguities must be >= pppar_min_ambiguities={minimum} "
                                 f"or None (got {partial})")
        if changed in (None, "positioning_mode", "use_iono_constr", "ionosphere_model"):
            # every constrained uncombined route (R4c, R7, R2, single frequency) constrains the slant ionosphere
            # with the model value ('ion' column); without a model R4c failed with a KeyError (PPP-017), R7/R2
            # only inside the filter, and single-frequency PPP was silently constrained to 0 (bug-fix stage 11)
            if (self.positioning_mode in ("uncombined", "single") and bool(self.use_iono_constr)
                    and self.ionosphere_model is False):
                raise ValueError(
                    f"use_iono_constr=True with ionosphere_model=False (positioning_mode="
                    f"{self.positioning_mode!r}): the ionospheric constraints of uncombined PPP need a model "
                    f"value; set ionosphere_model to 'gim', 'klobuchar', 'ntcm' or 'bdgim', or use_iono_constr=False")

    _CROSS_SETTINGS = Config._CROSS_SETTINGS | {"pppar_partial_min_ambiguities", "pppar_min_ambiguities",
                                                "positioning_mode", "use_iono_constr", "ionosphere_model"}

    def _set_checked(self, name: str, value: Any) -> None:
        if name == "positioning_mode":
            # a new mode re-applies its EKF preset; explicit overrides are kept (as set_mode does)
            super()._set_checked(name, value)
            self._apply_effective_to_self()
            return
        if name in self._preset_keys():
            if value is MISSING:
                self.clear_override(name)
                return
            check = self._field_checks().get(name)
            value = check(value) if check is not None else value
            self._warn_if_unused(name, value)
            # an assigned EKF parameter is an explicit override: set_mode keeps it
            self._overrides[name] = value
            object.__setattr__(self, name, value)
            return
        super()._set_checked(name, value)
