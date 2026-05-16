from __future__ import annotations
from typing import List, Set,Optional, Union, Literal, Dict, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from .gnss import SUPPORTED_MODES_BY_SYSTEM, normalize_systems

@dataclass()
class Config:
    """Default GNX-py Configuration class"""
    # --- I/O parameters
    obs_path: Union[str, Path, None] = None
    nav_path: Optional[Union[str, Path, None]] = None
    sp3_path: Optional[Union[List,None]] = None
    atx_path: Optional[Union[str, Path, None]] = None
    dcb_path: Optional[Union[str, Path, None]] = None
    sinex_path: Optional[Union[str, Path, None]] = None
    gim_path: Optional[Union[str, Path, None]] = None
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
    ionosphere_model: Union[Literal['gim','klobuchar','ntcm'], False] = 'gim'
    # - troposphere
    troposphere_model: Union[Literal['niell','saastamoinen','collins'], False] = 'niell'
    # - satellite PCO
    sat_pco: Union[Literal['los','crd'],False]='los'
    # - reciever PCO
    rec_pco:bool =True
    # - other corrections:
    windup:bool = True
    rel_path: bool = True
    solid_tides: bool = True
    antenna_h: bool = True
    ev_mask: int = 10
    cycle_slips_detection: bool = True
    min_arc_len:int=10

    # additional parameters
    station_name: Optional[str] = None
    day_of_year: Optional[int] = None
    time_limit:Optional[List[datetime]] = None

    def __post_init__(self) -> None:  # noqa: D401 – simple phrase is OK
        selected = normalize_systems(self.sys)
        self.sys = selected if not isinstance(self.sys, str) or len(selected) > 1 else next(iter(selected))

        mode_by_system = {
            "G": self.gps_freq,
            "E": self.gal_freq,
            "C": self.bds_freq,
        }
        for system in selected:
            mode = mode_by_system[system]
            if mode not in SUPPORTED_MODES_BY_SYSTEM[system]:
                raise ValueError(
                    f"Unsupported mode {mode!r} for system {system!r}. "
                    f"Supported: {SUPPORTED_MODES_BY_SYSTEM[system]}"
                )

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
        if not has_processing_inputs:
            return

        # Consistency checks --------------------------------------------------
        if self.ionosphere_model == "gim" and not self.gim_path:
            raise ValueError("Ionosphere model 'gim' requires `gim_path`." )
        if self.ionosphere_model in {"klobuchar", "ntcm"} and not self.nav_path:
            raise ValueError(
                f"Ionosphere model '{self.ionosphere_model}' requires ʻnav_pathʼ (navigation file missing)."
            )

        if self.orbit_type == "broadcast" and not self.nav_path:
            raise ValueError("Broadcast orbits demand `nav_path`. (navigation file missing).")

        if self.screen and not self.nav_path:
            raise ValueError("Screening orbits demand `nav_path`. (navigation file missing).")
        if (self.sat_pco or self.rec_pco) and not self.atx_path:
            raise ValueError('Applying satellite or reciever PCO requires `atx_path`. (ANTEX file missing)')

        if self.orbit_type =='precise' and not self.sp3_path:
            raise ValueError('Precise interpolation method requires `sp3_path`. (SP3 file missing).')
    def system_includes(self, code: str) -> bool:
        """Return *True* if the chosen constellation set includes *code* ("G"/"E"/"C")."""
        return code in self.sys if isinstance(self.sys, set) else self.sys == code



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
        (``"gim"``, ``"klobuchar"``, ``"ntcm"`` or ``False``).

    STEC parameters:
        ``leveling_ws`` and ``median_leveling_ws`` control phase-to-code
        leveling windows in samples. ``tec_zenith_err`` is the zenith TEC
        weighting parameter used during leveling. ``irls_timescale`` is a
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

    Model and diagnostics:
        ``compare_models`` may request additional ``klobuchar``/``ntcm``/``gim``
        comparison columns. This path is experimental and should be validated
        against known products before scientific use. ``use_iono_rms`` controls
        use of GIM RMS where available. ``skip_sat`` may exclude satellites from
        TEC processing.

    Supported systems:
        GPS and Galileo are the historical paths. BeiDou is supported for the
        configured BDS modes, with ``B1IB3I`` currently the best-tested STEC
        pair in the repository tests. Bias-product availability remains the
        limiting factor for some BDS products.

    Limitations:
        Changing leveling windows, bias fallback order or station calibration
        changes the measurement model and can affect STEC values. Treat those
        fields as numerical configuration, not cosmetic options.
    """
    leveling_ws: int = 30
    median_leveling_ws: int = 30
    tec_zenith_err: Union[float, int] = 4.5
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



MISSING = object()
def _merge_preset_with_overrides(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(preset)
    out.update(overrides)
    return out


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
            ``"klobuchar"`` or ``"ntcm"``.

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
            Controls the observation loader convention for some RINEX/GFZ style
            products. Recommended to leave at the session/default value unless
            validating data-source differences.

        output_dir (str | Path | None, optional):
            Optional output destination for drivers or downstream code. The
            core PPP session returns data frames and does not require this.

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
            Orbit interpolation method name. The active PPP path uses Lagrange
            interpolation for precise products.

        interpolation_degree (int | None):
            Degree used by precise orbit interpolation. The default is tuned
            for the existing SP3 interpolation path; changing it can affect
            numerical results.

        broadcast_tolerance (str | None):
            Time tolerance for broadcast interpolation, for example ``"2H"``.

        ionosphere_model ({"gim", "klobuchar", "ntcm"} | False):
            Ionosphere source used during preprocessing. ``"gim"`` requires
            ``gim_path``; ``"klobuchar"`` and ``"ntcm"`` require ``nav_path``.
            Set to ``False`` only for workflows that intentionally skip this
            correction/constraint preparation.

        troposphere_model ({"niell", "saastamoinen", "collins"} | False):
            Troposphere model used during preprocessing. PPP filters generally
            estimate ZTD; this field controls model terms prepared upstream.

        sat_pco ({"los", "crd"} | False):
            Satellite antenna phase-center correction mode. ``"los"`` is the
            usual PPP path. Requires ``atx_path`` when enabled.

        rec_pco (bool):
            Enables receiver antenna phase-center correction. Requires
            ``atx_path`` when enabled.

        windup, rel_path, solid_tides, antenna_h (bool):
            Standard PPP correction switches for phase wind-up, relativistic
            path term, solid tides and antenna height handling. Recommended to
            keep enabled for precise PPP unless validating a controlled model
            difference.

        ev_mask (int | float):
            Elevation cutoff in degrees. Affects preprocessing and accepted
            observations.

        cycle_slips_detection (bool):
            Enables cycle-slip detection before filtering.

        min_arc_len (int):
            Minimum arc length accepted by cycle-slip preprocessing. Larger
            values are more conservative and may reject short arcs.

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

        use_iono_rms (bool | None):
            Uses per-observation ``ion_rms`` values, when available, to scale
            ionospheric constraint variance.

        sigma_iono_0, sigma_iono_end (float | int | None):
            Initial and final sigma values for the ionospheric constraint ramp.
            Advanced tuning knobs; changing them changes the stochastic model.

        t_end (int | None):
            Time/epoch horizon over which the ionospheric constraint sigma ramps
            toward ``sigma_iono_end``.

        trace_filter (bool | None):
            Enables controlled PPP trace output. Normal library use should keep
            it disabled; debugging output is formatted through PPP trace
            helpers.

        reset_every (int | None):
            Optional periodic filter reset interval. ``0`` disables periodic
            resets. Advanced validation/debug option.

    PPP-AR fields:
        pppar_enabled (bool):
            Enables PPP ambiguity-resolution attempts where the selected filter
            supports AR. Disabled by default.

        pppar_warmup_epochs (int):
            Minimum warm-up period before AR attempts.

        pppar_min_ambiguities (int):
            Minimum ambiguity count required for a candidate group.

        pppar_ratio_threshold (float):
            LAMBDA ratio-test threshold. Lower values are more permissive;
            changing this directly affects AR acceptance.

        pppar_constraint_sigma_cycles (float):
            Soft-constraint sigma applied to accepted integer fixes, in cycles.

        pppar_constraint_sigma_floor_cycles (float | None):
            Lower bound for the AR hold sigma. ``None`` disables the floor.

        pppar_min_lock_epochs (int | None):
            Optional minimum arc age before an ambiguity can be fixed. If
            omitted, the warm-up setting is used as the effective lock gate.

        pppar_min_candidate_elevation_deg (float | None):
            Optional elevation cutoff for AR candidates.

        pppar_use_float_ratio_covariance (bool):
            Uses float covariance for the AR ratio test when supported.
            Advanced option; keep enabled unless validating covariance policy.

        pppar_partial_fixing_enabled (bool):
            Allows fixing mature subsets when some candidates are still too
            young. Advanced AR option.

        pppar_partial_min_ambiguities (int | None):
            Minimum ambiguity count for partial fixing. Defaults to the full
            ``pppar_min_ambiguities`` policy when ``None``.

        pppar_wide_lane_max_frac_cycles (float | None):
            Wide-lane fractional gate used by uncombined dual-frequency AR.

        pppar_combined_if_ar_enabled (bool):
            Experimental safety gate for combined ionosphere-free PPP-AR.
            ``pppar_enabled`` alone is not enough; this must also be ``True``.
            Leave disabled unless explicitly validating combined IF AR.

    EKF preset and override fields:
        clock_process ({"RW", "WN"} | MISSING):
            Receiver clock process model. Filled from the selected mode preset
            unless explicitly overridden.

        p_crd, p_dt, p_amb, p_tro, p_isb, p_N, p_iono, p_dcb:
            Initial covariance values for coordinates, receiver clock,
            ambiguities, troposphere, inter-system bias, ambiguity/phase state,
            ionosphere state and receiver/signal bias states. Advanced; changes
            affect convergence and numerical behavior.

        q_crd, q_dt, q_tro, q_amb, q_isb, q_N, q_iono, q_dcb:
            Process-noise values for the corresponding states. Advanced model
            knobs. Defaults come from the combined/uncombined presets.

    Uncombined observation and constraint tuning:
        uncombined_code_var, uncombined_phase_var:
            Base observation variances for generic uncombined no-constraint
            paths.

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
            Advanced constrained-model option that controls the relationship
            between ionosphere and clock/bias identities.

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
            Uses legacy constrained code/phase weighting. Recommended for
            comparisons against the GPS/Galileo reference branch.

        uncombined_iono_code_var_base,
        uncombined_iono_code_var_elev_coeff,
        uncombined_iono_phase_var_base,
        uncombined_iono_phase_var_elev_coeff:
            Components of the legacy elevation-dependent observation weighting
            model used in constrained uncombined PPP.

        uncombined_iono_signal_bias_phase_jacobian:
            Advanced switch controlling how receiver signal-bias states enter
            constrained phase equations. Use only when validating the bias
            model.

        uncombined_iono_constraints_model ({"legacy", "gec"}):
            Selects the constrained mixed-system branch. ``"legacy"`` keeps the
            GPS/Galileo reference path where routing allows it. ``"gec"``
            forces the newer generic GPS/Galileo/BeiDou constrained branch for
            mixed systems.

        uncombined_mixed_phase_code_gate_systems (str | list[str] | set[str] | None):
            Optional advanced selector for stricter mixed uncombined phase/code
            gating by system. Use only for validation experiments.

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
        ionosphere-free AR is experimental and requires the explicit
        ``pppar_combined_if_ar_enabled`` gate.
    """

    positioning_mode: Literal["combined", "uncombined", "single"] = "combined"

    # iono / processing (zostają jak u Ciebie)
    use_iono_constr: Optional[bool] = True
    use_iono_rms: Optional[bool] = True
    sigma_iono_0: Optional[Union[float, int]] = 1.1
    sigma_iono_end: Optional[Union[float, int]] = 3.0
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
    pppar_partial_fixing_enabled: bool = False
    pppar_partial_min_ambiguities: Optional[int] = None
    pppar_wide_lane_max_frac_cycles: Optional[float] = 0.25
    pppar_combined_if_ar_enabled: bool = False
    uncombined_mixed_phase_code_gate_systems: Optional[Union[str, List[str], Set[str]]] = None

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

    # PPP uncombined observation weighting/gating knobs.  The generic values
    # below are still used by no-constraint paths and remain available as a
    # fallback for constrained single-GNSS runs.
    uncombined_code_var: float = 1.0
    uncombined_phase_var: float = 1.0
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
    uncombined_iono_legacy_weighting: bool = True
    uncombined_iono_code_var_base: float = 0.3
    uncombined_iono_code_var_elev_coeff: float = 0.0025
    uncombined_iono_phase_var_base: float = 1e-4
    uncombined_iono_phase_var_elev_coeff: float = 0.0003
    uncombined_iono_signal_bias_phase_jacobian: bool = True
    uncombined_iono_constraints_model: Literal["legacy", "gec"] = "legacy"

    _overrides: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    _PRESETS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "combined": {
            "clock_process": "WN",
            "p_crd": 10.0, "p_dt": 9e9, "p_amb": 400, "p_tro": 2.0,
            "q_crd": 0.0,  "q_dt": 9e9, "q_tro": 0.00025, "q_amb": 0.0,
            "p_isb": 0.0, "q_isb": 0.0, "q_N": 0.0,
            "p_iono": 0.0, "q_iono": 0.0, "p_dcb": 0.0, "q_dcb": 0.0,
        },
        "uncombined": {
            "clock_process": "RW",
            "p_crd": 100, "p_dt": 1e9, "p_tro": 2.0, "p_iono": 10,
             "p_isb": 100, "p_dcb": 1e2, "p_amb": 1e6,
            "q_crd": 0.0, "q_dt": 1e9, "q_tro": 0.025, "q_iono": 10.0,
            "q_N": 0.0, "q_isb": 1.0, "q_dcb": 1e-4, "q_amb": 0.0,
            "uncombined_code_var": 1.0,
            "uncombined_phase_var": 1.0,
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
            "uncombined_iono_legacy_weighting": True,
            "uncombined_iono_code_var_base": 0.3,
            "uncombined_iono_code_var_elev_coeff": 0.0025,
            "uncombined_iono_phase_var_base": 1e-4,
            "uncombined_iono_phase_var_elev_coeff": 0.0003,
            "uncombined_iono_signal_bias_phase_jacobian": True,
            "uncombined_iono_constraints_model": "legacy",
        },
    }, init=False, repr=False)

    def __post_init__(self):
        # 1) najpierw baza
        super().__post_init__()

        # 2) zbierz override'y przekazane w konstruktorze
        for k in self._preset_keys():
            v = getattr(self, k)
            if v is not MISSING:
                self._overrides[k] = v

        # 3) ustaw effective (preset + overrides)
        self._apply_effective_to_self()

    # -------- public API --------
    def set_mode(self, mode: Literal["combined", "uncombined", "single"], keep_overrides: bool = True) -> None:
        self.positioning_mode = mode
        if not keep_overrides:
            self._overrides.clear()
        self._apply_effective_to_self()

    def set_param(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k not in self._preset_keys():
                raise KeyError(f"Unknown/unsupported param: {k}")
            self._overrides[k] = v
        self._apply_effective_to_self()

    def clear_override(self, *keys: str) -> None:
        for k in keys:
            self._overrides.pop(k, None)
        self._apply_effective_to_self()

    def effective(self) -> Dict[str, Any]:
        preset = self._PRESETS[self._effective_positioning_mode()]
        return _merge_preset_with_overrides(preset, self._overrides)

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
            setattr(self, k, v)
