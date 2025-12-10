from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Literal, Set
from pathlib import Path
from datetime import datetime

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
    gps_freq: Literal["L1", "L2", "L1L2", "L1L5", "L5"] = "L1L2"
    gal_freq: Optional[Literal["E1", "E5a", "E5b", "E1E5a", "E1E5b"]] = "E1E5a"

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
        """Return *True* if the chosen constellation set includes *code* ("G"/"E")."""
        return code in self.sys if isinstance(self.sys, set) else self.sys == code



@dataclass()
class TECConfig(Config):
    """Default TEC GNX-py Configuration class"""
    leveling_ws: int = 30
    median_leveling_ws: int = 30
    tec_zenith_err: Union[float, int] = 4.5
    irls_timescale: Optional[Union[float, int]] = 900
    add_dcb: bool = True
    add_sta_dcb: bool = False
    rcv_dcb_source: Optional[Literal["gim", "calibrate", "defined", "none"]] = "calibrate"
    define_station_dcb: Optional[float] = None
    compare_models: Optional[Union[bool, list[str]]] = False
    use_iono_rms: Optional[bool] = True,
    skip_sat: Optional[Union[bool, list[str]]] = False

@dataclass
class PPPConfig(Config):
    """GNX-py PPP Configuration class"""
    # --- positioning parameters
    positioning_mode:Optional[Literal['combined','uncombined','single']]= 'combined'

    # - ionosphere parameters
    use_iono_constr: Optional[bool] = True
    use_iono_rms: Optional[bool] = True
    sigma_iono_0: Optional[Union[float, int]] = 1.1
    sigma_iono_end: Optional[Union[float, int]] = 3.0
    t_end: Optional[int] = 30

    # --- processing parameters
    trace_filter: Optional[bool] = False
    reset_every: Optional[int] = 0

    # --- Kalman Filter params
    # clock process
    clock_process: Optional[Literal['RW','WN']]='WN'
    # P - prior variances
    p_crd: Union[float, int] = 10.0
    p_dt: Union[float, int] = 9e9
    p_amb: Union[float, int] = 400
    p_tro: Union[float, int] = 0.0
    # Q - process noise
    q_crd: Union[float, int] = 0.0
    q_dt: Union[float, int] = 9e9
    q_tro: Union[float, int] = 0.00025
    q_amb: Union[float, int] = 0.0

    # uduc params
    p_isb: Union[float, int] = 0.0
    q_isb: Union[float, int] = 0.0
    p_N: Union[float, int] = 0.0
    q_N: Union[float, int] = 0.0

    p_iono: Union[float, int] = 0.0
    q_iono: Union[float, int] = 0.0

    p_dcb: Union[float, int] = 0.0
    q_dcb: Union[float, int] = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.positioning_mode == 'uncombined':
            self.p_crd = 100
            self.p_dt =1e9
            self.p_tro = 2.0
            self.p_iono = 1.5
            self.p_N = 1e6
            self.p_isb = 100
            self.p_dcb = 1e2

            self.clock_process ='RW'
            self.q_dt = 1e9
            self.q_tro  = 0.025
            self.q_iono = 2.0 # RW model
            self.q_N = 0.0
            self.q_isb = 1.0
            self.q_dcb = 1e-4
        if self.positioning_mode == 'single':
            self.p_crd = 100
            self.p_dt =1e9
            self.p_tro = 2.0
            self.p_iono = 100
            self.p_N = 1e6

            self.clock_process ='WN'
            self.q_dt = 9e9
            self.q_tro  = 0.025
            self.q_iono = 25
            self.q_N = 0.0












