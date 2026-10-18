"""Satellite attitude (yaw) of GPS, Galileo and BeiDou satellites.

One provider for the satellite antenna offset (PCO), the satellite PCV azimuth and the phase wind-up: the IGS body
axes in ECEF per observation row, from the published attitude law of the satellite block, from an ORBEX file of
the analysis centre (:mod:`gnx_py.orbex`) where it covers a satellite's arc, or nominal yaw steering.

Conventions (Montenbruck et al. 2015, ASR 56:1015, eqs. 1, 4-6; Kouba 2009, GPS Solut 13:1):

* inertial velocity v_i = v_ecef + w_E x r; e_R = r/|r|, e_N = unit(r x v_i), e_T = e_N x e_R;
* s = unit(r_sun) (geocentric Sun); beta = asin(s . e_N), the Sun elevation above the orbital plane;
* mu, the orbit angle counted from orbit midnight in the direction of motion (noon at 180 deg):
  mu = atan2(s . e_T, -s . e_R); mu_dot = |r x v_i| / |r|^2;
* yaw psi: angle from e_T to the body x axis, right-handed about z_b = -e_R:
  x_b = cos(psi) e_T - sin(psi) e_N, y_b = -sin(psi) e_T - cos(psi) e_N, z_b = -e_R.
  psi = 0 is orbit-normal attitude (RTKLIB 2.4.3 ``sat_yaw`` with yaw 0, ``corrections.orbit_normal_frame``,
  CSNO "zero-bias attitude");
* nominal yaw steering psi_n = atan2(-tan(beta), sin(mu)): the body frame of RTKLIB ``satantoff``/``windupcorr``
  (y = unit(s x r), x toward the Sun side), which this module computes with the same arithmetic as GNX before v1.0.0 for every row in yaw steering.

Turn laws (``LAWS``; the first law of a family is the default of mode 'model'):

* GPS IIR (``kouba2009``): catch-up yaw steering at 0.20 deg/s at noon and midnight (Kouba 2009; Orekit
  GPSBlockIIR, Apache-2.0, for the conventions): beta0 = atan(mu_dot/R), start sqrt(|beta|(beta0 - |beta|)) rad
  before the turn centre, linear yaw until it meets the nominal yaw again;
* GPS IIF (``iif``): noon catch-up at 0.11 deg/s, reversed for -0.7 < beta < 0 deg (Kuang, Desai, Sibois 2017,
  GPS Solut 21:739); midnight: constant yaw rate across the Earth's shadow (cone of 13.25 deg, Kouba eclips),
  from the nominal yaw at shadow entry to the one at exit by the short route;
* GPS III (``montenbruck2026``): auxiliary Sun vector with c_x = sin 15 deg, c_y = sin 5.8 deg (Montenbruck,
  Duan, Steigenberger, Hugentobler 2026, ASR 77:4901, eqs. 11-13, 17-18); ``iir``/``iif`` substitute the GPS IIR
  or IIF law (COD0OPSFIN uses the IIR law for GPS III, ibid. section 2);
* Galileo IOV (``gsc_iov``): GSC Galileo satellite metadata 3.1.1 in the IGS form of 3.2 (c_y = sin 2 deg);
* Galileo FOC (``gsc_foc``): GSC 3.1.2: for |beta| < 4.1 deg, 10 deg (in-plane angle) before noon/midnight,
  psi = 90 S + (psi_init - 90 S) cos(2 pi t/5656 s) for 2828 s (the end at the mirror yaw is inferred);
* BeiDou GEO (``orbit_normal``): psi = 0 in every mode;
* BDS-2 IGSO/MEO with yaw-steering/orbit-normal switching (``ys_on``, opt-in; default nominal): orbit-normal
  from the first |psi_n| <= 5 deg pass after |beta| < 4 deg, back at the first such pass after |beta| > 4 deg
  (Li et al. 2018, via Zhao et al. 2022, Satellite Navigation 3:2);
* BDS-2 continuous yaw steering (``dilssner2017``: SVN C005 from 2017-03-01, C015 from 2016-10-01, C017, C019):
  beta_d = beta + f (sign(beta) 2.8 deg - beta), f = 1/(1 + 80000 sin^4 mu) for |beta| < 2.8 deg (Dilssner 2017,
  ESA note on BeiDou IGSO-6, eq. 4 with its two cases in the order of its text);
* BDS-3 CAST MEO/IGSO (``wang2018``): psi = 90 S + (psi_s - 90 S) cos(2 pi (t - t_s)/t_max) for |beta| <= 3 deg
  from 6 deg before noon/midnight, t_max 3090 s (MEO) and 5740 s (IGSO) (Wang et al. 2018, GPS Solut 22:117, as
  Yang, Guo, Zhao 2023, J Geod 97:6, eq. 1); ``dilssner2017`` is the law COD, IAC and SHAO use (Steigenberger
  et al. 2023, ASR 71:355, Table 2);
* BDS-3 SECM (``csno_yang2023``): psi = atan2(-tan(sigma 3 deg), sin mu) for |beta| < 3 deg (CSNO BD
  420025-2019 5.4); sigma takes the new sign of beta only at the first epoch after the beta zero crossing where
  |psi| < 5 deg, i.e. mu = 36.80 deg (Yang et al. 2023, Table 1); ``csno``: sigma = sign(beta); ``dilssner2017``:
  the smoothed law of COD (COD0MGXFIN ORBEX of 2024-035: 0.02 deg RMS on the 10 SECM satellites).

The ORBEX files of 2024-035 against the laws, over the turn rows of the default law and the rows where the file
leaves it by more than 1 deg (yaw RMS / maximum): Galileo FOC GRG 0.94/2.3 deg, COD
0.14/0.3 deg; Galileo IOV 0.03/0.1 deg (both); GPS IIF COD 0.84/2.1 deg, GRG 17.5/60.0 deg (a later, faster
shadow turn in the same direction); GPS III COD flies the IIR law (``iir``: 0.45/1.9 deg), GRG a rate-limited turn
in the same direction (32 deg RMS against ``montenbruck2026``); BDS-3 SECM COD flies ``dilssner2017`` (0.02/0.1
deg), GRG the CSNO law with sigma = -sign(beta), i.e. every turn the other way; BDS-2 C11/C12 COD flies YS/ON
(``ys_on``: C12 orbit-normal all day, 0.00 deg; C11 switched before 08:30, earlier than the rule), GRG yaw steering.

Every law is evaluated in closed form per row from (r, v, Sun): the turn centre, the event start (t_s, psi_s, the
entry sign, the shadow entry/exit, the SECM sign, the BDS-2 mode) follow from the geometry of the row,
propagated with the two-body orbit (Kepler's equation from the osculating elements of the row) and a linear
beta(t) whose rate holds the Sun's annual motion and the J2 nodal regression of the orbit. There is no state
across rows, so elevation-masked frames with gaps, broadcast orbits and a first epoch inside a turn all work
the same way. The osculating beta of real orbits wobbles by about 0.001 deg (J2 short-period terms), which moves
the predicted beta zero crossing by minutes; so the discrete choices of one event are made once for all rows of
the call that belong to it (one satellite, one noon/midnight passage or one beta crossing): the beta tests of the
turn laws and of the guard on the event's median beta, the turn direction on its sign, the SECM sign switch and
the BDS-2 mode switch at the median crossing time of the rows within 1.05 periods of it. No numpy error state is
changed: undefined cases (atan2(0, 0), |j| = 0) are masked explicitly and at exactly beta = 0 the turn-direction
rule takes the positive branch (catch-up and IIF shadow turns alike).

Limit: the Galileo FOC profile has a fixed length of 2828 s from the switch 10 deg before noon/midnight. On the
eccentric E14/E18 (e = 0.16) mu_dot varies by about +-30 % around the orbit, so the profile can end up to about 10
deg from the nominal yaw (one step of a few mm of wind-up and PCO); the circular orbits end within 0.3 deg.

Turn guard (``guard_mask``): without the attitude of the analysis centre (AC) whose orbits and clocks a PPP
session uses, the clocks hold the AC's own wind-up of every noon/midnight turn, and ACs fly some turns the
other way round (one wind-up cycle, about 10.6-10.9 cm of the ionosphere-free phase). Rows within
``GUARD_HALF_WINDOW_S`` of a turn whose |beta| is below ``GUARD_BETA_DEG`` of the satellite class can be
dropped; ``AC_GUARD_EXEMPT`` lists the classes in which an AC agrees with the laws here
(``AC_GUARD_EXEMPT_NOMINAL`` those it flies in nominal yaw steering, exempt while the law here is 'nominal').
The ORBEX attitude of the product's own AC replaces the guard, except for ``ORBEX_GUARDED`` (COD Galileo: its
ORBEX turns as the laws, its clocks do not). The guard 'auto' acts on static and dual-frequency kinematic PPP
 and, for the ``ORBEX_GUARDED`` families only, on single-frequency kinematic PPP.

No GPL code was used: the laws are written from the cited papers and operator metadata; Orekit
(org.orekit.gnss.attitude, Apache-2.0), Ginan (attitude.cpp, Apache-2.0) and RTKLIB (BSD-2) were read for
conventions only.
"""
from __future__ import annotations

import difflib
import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "LAWS",
    "LAW_FAMILIES",
    "DEFAULT_LAWS",
    "REGIMES",
    "SOURCES",
    "GUARD_BETA_DEG",
    "GUARD_HALF_WINDOW_S",
    "GUARD_MODES",
    "SHADOW_HALF_ANGLE_DEG",
    "AC_GUARD_EXEMPT",
    "AC_GUARD_EXEMPT_NOMINAL",
    "ORBEX_GUARDED",
    "Attitude",
    "law_family",
    "check_laws",
    "guard_class",
    "guard_mask",
    "nominal_yaw",
    "nominal_yaw_rate",
    "orbit_frame",
    "body_axes",
    "yaw_of_axes",
    "satellite_attitude",
]

OMGE = 7.2921151467e-5      # Earth rotation rate [rad/s] (IS-GPS-200, as corrections.OMGE)
GM = 3.986004418e14         # [m^3/s^2] (IERS 2010, WGS84, CGCS2000, GTRF)
J2 = 1.0826267e-3           # (EGM2008)
RE = 6378137.0              # [m]
GPS_EPOCH = datetime(1980, 1, 6)
_D2R = math.pi / 180.0

#: attitude laws per family; the first one is the default of mode 'model'
LAWS: Dict[str, Tuple[str, ...]] = {
    "gps_iir": ("kouba2009", "nominal"),
    "gps_iif": ("iif", "nominal"),
    "gps_iii": ("montenbruck2026", "iir", "iif", "nominal"),
    "gal_iov": ("gsc_iov", "nominal"),
    "gal_foc": ("gsc_foc", "nominal"),
    "bds_geo": ("orbit_normal",),
    "bds2_ys_on": ("nominal", "ys_on"),
    "bds2_cys": ("dilssner2017", "nominal"),
    "bds3_cast_meo": ("wang2018", "dilssner2017", "nominal"),
    "bds3_cast_igso": ("wang2018", "dilssner2017", "nominal"),
    "bds3_secm": ("csno_yang2023", "csno", "dilssner2017", "nominal"),
    "unknown": ("nominal",),
}
LAW_FAMILIES: Tuple[str, ...] = tuple(LAWS)
DEFAULT_LAWS: Dict[str, str] = {f: laws[0] for f, laws in LAWS.items()}

#: ``Attitude.regime`` codes
REGIMES: Dict[int, str] = {
    0: "yaw steering",
    1: "catch-up",
    2: "shadow",
    3: "auxiliary Sun",
    4: "cosine",
    5: "SECM +-3 deg",
    6: "BDS-2 smoothed",
    7: "orbit-normal",
    8: "ORBEX",
    9: "nominal fallback",
}
#: ``Attitude.source`` codes
SOURCES: Dict[int, str] = {0: "nominal", 1: "law", 2: "ORBEX"}

#: turn guard: |beta| at the turn [deg] below which a class is guarded (GEO never)
GUARD_BETA_DEG: Dict[str, float] = {"G": 0.5, "E": 0.5, "BDS-3": 3.0, "BDS-2": 4.0}
#: turn guard: rows within this time of the noon/midnight passage [s]
GUARD_HALF_WINDOW_S = 2100.0
#: accepted guard modes (None = 'off')
GUARD_MODES = ("auto", "all", "off")
#: Earth shadow half-angle seen from the geocentre for the GPS IIF midnight turn [deg] (Kouba eclips; Orekit)
SHADOW_HALF_ANGLE_DEG = 13.25
#: rows of one satellite whose predicted noon/midnight passages lie within this time of each other belong to one
#: passage [s] (the predictions of one passage agree to seconds; passages are at least 5.6 h apart)
EVENT_GAP_S = 3600.0
#: ... and the predicted beta crossings (zero, or |beta| = 4 deg) within this time to one crossing [s] (minutes of
#: scatter; crossings of one satellite are days apart)
CROSSING_GAP_S = 43200.0
#: the beta of a sign-sensitive nominal yaw takes the sign of its event; |beta| at least this [rad]
_TINY_BETA = 1e-9

#: Families whose noon/midnight turns an analysis centre flies in the direction of the laws here (no whole wind-up
#: cycle between its clocks and the laws), so the turn guard 'auto' leaves them alone. Evidence (v1.0.0
#: validation, days 2024-034/035/036/130-133): steps of one ionosphere-free cycle in the GRG0MGXFIN - COD0MGXFIN user
#: range across 388 turns with |beta| < 4.5 deg (BDS-3 SECM 77 of 83 turns; Galileo 17 of 157, all at
#: 0 <= beta <= 0.3 deg; GPS 4 of 132 noon turns at beta +0.1/+0.2 deg), GFZ0MGXRAP as the tie-breaker (GFZ agrees
#: with COD on SECM and with GRG on Galileo and GPS), the geometry- and ionosphere-free three-frequency phase at
#: BRUX (Galileo E09/E03/E05 turn in the direction of the GSC law), and the ORBEX files of 2024-035 (GRG GPS IIF/III
#: and Galileo turns in the direction of the laws; GRG SECM reversed; COD SECM the Dilssner law, same direction).
#: GRG agrees for GPS and Galileo; COD for BDS-3 SECM; GFZ for GPS, Galileo and BDS-3. GFZ's BDS-3 CAST entries
#: (bds3_cast_meo, bds3_cast_igso) are unverified on the validation data: no CAST turn at small |beta| on those days and no
#: GFZ ORBEX; they rest on the literature (GFZ flies the Wang et al. 2018 CAST law, the default law here,
#: Steigenberger et al. 2023). Valid for the 2024 products: a centre that changes its attitude convention later is
#: exempted silently, so re-check the table against newer ORBEX files. A missing entry costs data (the guard
#: drops the rows), never consistency.
AC_GUARD_EXEMPT: Dict[str, frozenset] = {
    "GRG": frozenset({"gps_iir", "gps_iif", "gps_iii", "gal_iov", "gal_foc"}),
    "COD": frozenset({"bds3_secm"}),
    "GFZ": frozenset({"gps_iir", "gps_iif", "gps_iii", "gal_iov", "gal_foc", "bds3_secm", "bds3_cast_meo",
                      "bds3_cast_igso"}),
}
#: Families an analysis centre flies in nominal yaw steering through the turns, exempt from the guard 'auto' only
#: while their law here is 'nominal' (the default of ``bds2_ys_on``): GRG0MGXFIN ORBEX of 2024-035 keeps C11/C12
#: within 0.012 deg of the nominal yaw all day (COD flies them orbit-normal). With ``attitude_laws={'bds2_ys_on':
#: 'ys_on'}`` the laws and GRG disagree, so the guard applies again.
AC_GUARD_EXEMPT_NOMINAL: Dict[str, frozenset] = {
    "GRG": frozenset({"bds2_ys_on"}),
}
#: Families whose turns stay guarded even with the ORBEX attitude of the product's own analysis centre: COD's ORBEX
#: of 2024-035 flies the Galileo turns in the direction of the GSC laws (all 24 windows, within 0.4 deg), yet COD's
#: clocks hold one-cycle steps at the Galileo turns with 0 <= beta <= 0.3 deg (17 of 157 turns on the validation days):
#: with COD's ORBEX and no guard EC-SP3cod-static-UC0 (BRUX 2024-035, 6 h) ends at 6.26 cm, exactly as with the
#: laws alone, with the guard on the Galileo turns at 0.54 cm (E-SP3cod-static-IF 6.33 -> 0.60 cm). Valid for the
#: 2024 products. Since v1.0.0 these are also the only families the guard 'auto' acts on in
#: kinematic PPP: E-SP3cod-kin-IF BRUX 2024-035 6 h RMS 0.1848 -> 0.1078 m, p95 0.3181 -> 0.2445 m.
ORBEX_GUARDED: Dict[str, frozenset] = {
    "COD": frozenset({"gal_iov", "gal_foc"}),
}

# ----------------------------------------------------------------------------------------------- law constants
IIR_RATE_DEG_S = 0.20               # Kouba 2009 / eclips: every GPS IIR
IIF_RATE_DEG_S = 0.11               # Dilssner 2010 / eclips
IIF_NOON_REVERSAL_DEG = 0.7         # IIF noon turns reverse for -0.7 < beta < 0 deg (Kuang et al. 2017)
GPS3_AUX_DEG = (15.0, 5.8)          # Montenbruck et al. 2026, c_x = sin 15, c_y = sin 5.8
IOV_AUX_DEG = (15.0, 2.0)           # GSC 3.1.1: beta_x = 15, beta_y = 2
FOC_BETA_DEG = 4.1                  # GSC 3.1.2
FOC_SWITCH_DEG = 10.0
FOC_PERIOD_S = 5656.0
CAST_BETA_DEG = 3.0                 # Wang et al. 2018 (Yang et al. 2023 eq. 1)
CAST_SWITCH_DEG = 6.0
CAST_PERIOD_S = {"bds3_cast_meo": 3090.0, "bds3_cast_igso": 5740.0}
SECM_BETA_DEG = 3.0                 # CSNO 2019 5.4
SECM_SWITCH_YAW_DEG = 5.0           # Yang et al. 2023
YSON_BETA_DEG = 4.0                 # Li et al. 2018
YSON_SWITCH_YAW_DEG = 5.0
DILSSNER_BETA_DEG = 2.8             # Dilssner 2017
DILSSNER_D = 80000.0
#: BDS-2 satellites in continuous yaw steering (SVN: since; None = since launch) (Dilssner 2017; Zhao et al. 2022)
BDS2_CONTINUOUS_YAW: Dict[str, Optional[date]] = {
    "C005": date(2017, 3, 1), "C015": date(2016, 10, 1), "C017": None, "C019": None}

#: law -> algorithm of ``satellite_attitude`` (the family adds its own constants: the CAST period)
_ALGORITHMS = {
    "nominal": "nominal", "kouba2009": "catchup_iir", "iir": "catchup_iir", "iif": "iif",
    "montenbruck2026": "aux_gps3", "gsc_iov": "aux_iov", "gsc_foc": "foc", "orbit_normal": "orbit_normal",
    "ys_on": "ys_on", "dilssner2017": "dilssner", "wang2018": "cast", "csno_yang2023": "secm_yang",
    "csno": "secm_csno",
}


# ============================================================================================== blocks and laws
def _epoch_date(epoch) -> Optional[date]:
    """Calendar date of ``epoch`` (datetime, pandas/numpy timestamp, ISO string or GPST seconds), or None."""
    if epoch is None:
        return None
    if isinstance(epoch, datetime):
        return epoch.date()
    if isinstance(epoch, date):
        return epoch
    if isinstance(epoch, (int, float, np.integer, np.floating)):
        days = int(float(epoch) // 86400.0)
        return date.fromordinal(GPS_EPOCH.toordinal() + days)
    try:
        ts = np.datetime64(epoch, "D")
        return date.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


def law_family(block_type, svn=None, prn=None, epoch=None) -> str:
    """Attitude family of an ANTEX satellite entry (TYPE, columns 1-20; SVN, columns 41-50).

    BLOCK IIR-A/-B/-M -> gps_iir; BLOCK IIF -> gps_iif; BLOCK IIIA -> gps_iii (BLOCK IIIF too, with a warning);
    GALILEO-1 -> gal_iov; GALILEO-2 -> gal_foc; BEIDOU-2G and BEIDOU-3G-* -> bds_geo; BEIDOU-2I/2M ->
    bds2_cys for the continuous-yaw SVNs of ``BDS2_CONTINUOUS_YAW`` on or after their date (``epoch``; None =
    today's status), else bds2_ys_on; BEIDOU-3M-CAST and 3SM-CAST -> bds3_cast_meo; BEIDOU-3I and 3SI-CAST ->
    bds3_cast_igso; BEIDOU-3M-SECM and 3SI-SECM -> bds3_secm. Any other type -> 'unknown' (nominal attitude)
    with one UserWarning naming ``prn``.
    """
    b = " ".join(str(block_type or "").upper().split())
    who = f" of {prn}" if prn else ""
    if b in ("BLOCK IIR-A", "BLOCK IIR-B", "BLOCK IIR-M", "BLOCK IIR"):
        return "gps_iir"
    if b == "BLOCK IIF":
        return "gps_iif"
    if b == "BLOCK IIIA":
        return "gps_iii"
    if b == "BLOCK IIIF":
        warnings.warn(f"satellite attitude: ANTEX block {b}{who} has no published attitude law; the GPS III law "
                      f"(Montenbruck et al. 2026) is used", UserWarning, stacklevel=2)
        return "gps_iii"
    if b == "GALILEO-1":
        return "gal_iov"
    if b == "GALILEO-2":
        return "gal_foc"
    if b.startswith(("BEIDOU-2G", "BEIDOU-3G")):
        return "bds_geo"
    if b.startswith(("BEIDOU-2I", "BEIDOU-2M")):
        key = str(svn or "").strip().upper()
        if key in BDS2_CONTINUOUS_YAW:
            since = BDS2_CONTINUOUS_YAW[key]
            day = _epoch_date(epoch)
            if since is None or day is None or day >= since:
                return "bds2_cys"
        return "bds2_ys_on"
    if b.startswith("BEIDOU-3") and "SECM" in b:
        return "bds3_secm"
    if b in ("BEIDOU-3M-CAST", "BEIDOU-3SM-CAST"):
        return "bds3_cast_meo"
    if b in ("BEIDOU-3I", "BEIDOU-3SI-CAST"):
        return "bds3_cast_igso"
    warnings.warn(f"satellite attitude: ANTEX block {str(block_type).strip()!r}{who} has no attitude law in GNX; "
                  f"the nominal attitude (yaw steering) is used", UserWarning, stacklevel=2)
    return "unknown"


def _did_you_mean(word: str, choices) -> str:
    close = difflib.get_close_matches(str(word), list(choices), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def check_laws(mapping: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Validated ``attitude_laws`` override {family: law}; ValueError for an unknown family or a law the family
    does not have, with a "did you mean" hint. None -> {}."""
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"attitude_laws must be a mapping {{family: law}}, got {type(mapping).__name__}")
    out = {}
    for fam, law in mapping.items():
        f = str(fam).strip().lower()
        if f not in LAWS:
            raise ValueError(f"attitude_laws: unknown attitude family {fam!r}{_did_you_mean(f, LAWS)}; families: "
                             f"{', '.join(LAW_FAMILIES)}")
        lw = str(law).strip().lower()
        if lw not in LAWS[f]:
            raise ValueError(f"attitude_laws: family {f!r} has no law {law!r}{_did_you_mean(lw, LAWS[f])}; laws: "
                             f"{', '.join(LAWS[f])}")
        out[f] = lw
    return out


# ============================================================================================== guard
def guard_class(family: str, sv: Optional[str] = None) -> Optional[str]:
    """Class of ``GUARD_BETA_DEG`` for an attitude family ('G', 'E', 'BDS-3', 'BDS-2'), None for GEO and for a
    BeiDou satellite of unknown block (never guarded)."""
    f = str(family)
    if f.startswith("gps_"):
        return "G"
    if f.startswith("gal_"):
        return "E"
    if f.startswith("bds3_"):
        return "BDS-3"
    if f.startswith("bds2_"):
        return "BDS-2"
    if f == "unknown" and sv:
        return {"G": "G", "E": "E"}.get(str(sv)[:1])
    return None


def _ac3(product_ac) -> Optional[str]:
    if product_ac is None:
        return None
    s = str(product_ac).strip()
    return s[:3].upper() if s else None


def guard_mask(families, beta_c_deg, dt_c_s, *, guard="auto", product_ac=None, static=True, broadcast=False,
               consistent=None, svs=None, laws=None, single_frequency=False) -> np.ndarray:
    """Turn-guard mask per row.

    ``families``: family per row (or one for all rows); ``beta_c_deg``: beta at the nearest noon/midnight passage;
    ``dt_c_s``: time from the row to that passage. ``guard``: 'auto' (static PPP with precise orbits: classes not
    exempt for the product's analysis centre, ``AC_GUARD_EXEMPT``, plus ``AC_GUARD_EXEMPT_NOMINAL`` while the
    family's law in ``laws`` (default ``DEFAULT_LAWS``) is 'nominal'; unknown centre -> no exemption; kinematic
    dual-frequency PPP: the same classes as static; kinematic single-frequency PPP (``single_frequency``): only
    the families of ``ORBEX_GUARDED`` of the product's centre, the classes whose clocks are proven to step at the
    turns, COD Galileo), 'all' (precise orbits, every class, static or kinematic) or 'off'/None.
    ``consistent``: rows whose attitude is the ORBEX of the product's centre, never guarded. Broadcast orbits are
    never guarded (their clocks hold no AC attitude).

    The kinematic part of 'auto' (since v1.0.0):
    E-SP3cod-kin-IF BRUX 2024-035 6 h RMS / p95 0.1848 / 0.3181 -> 0.1078 / 0.2445 m (388 rows of E03, E05, E09
    guarded; t_c20 15.5 min unchanged), GE/EC/GEC COD kinematic improve as well, no COD kinematic row worse. The
    guard of every non-exempt class is not used in kinematic single-frequency PPP: it removes satellites at every
    epoch of a turn window and broke the weak single-frequency BeiDou rows (sf-C-B1I-UCi/UC0-kin p95 0.59 -> 1.87
    m, 0.90 -> 2.34 m). Kinematic dual-frequency PPP guards the static classes: GRG
    flies the BeiDou-3 SECM turns the other way round (checked against the ORBEX files), so without the guard its clocks'
    wind-up step enters the kinematic position (BRUX 2024-035 6 h, 435 BeiDou-3 rows guarded, RMS from 60 min with
    the GIM constraint in the code-bias datum: C-SP3grg-kin-IF 0.1477 -> 0.0585 m, C-SP3grg-kin-UCi 0.1368 -> 0.0799
    m, EC-SP3grg-kin-IF 0.1100 -> 0.0155 m, GEC-SP3grg-kin-UCi 0.0429 -> 0.0274 m).
    True = the row lies within ``GUARD_HALF_WINDOW_S`` of a turn with |beta| below ``GUARD_BETA_DEG`` of its class.
    """
    beta_c = np.atleast_1d(np.asarray(beta_c_deg, dtype=float))
    dt_c = np.atleast_1d(np.asarray(dt_c_s, dtype=float))
    n = beta_c.size
    if guard is None or guard == "off":
        return np.zeros(n, dtype=bool)
    if guard not in GUARD_MODES:
        raise ValueError(f"attitude_turn_guard must be one of {GUARD_MODES}, got {guard!r}"
                         f"{_did_you_mean(str(guard), GUARD_MODES)}")
    kinematic_auto = guard == "auto" and not static and bool(single_frequency)
    only = ORBEX_GUARDED.get(_ac3(product_ac) or "", frozenset()) if kinematic_auto else frozenset()
    if broadcast or (kinematic_auto and not only):
        return np.zeros(n, dtype=bool)
    fam = np.broadcast_to(np.asarray(families).astype(str), (n,))
    system = np.broadcast_to(np.asarray(svs if svs is not None else "").astype(str).astype("<U1"), (n,))
    exempt = frozenset()
    if guard == "auto":
        ac = _ac3(product_ac) or ""
        chosen = dict(DEFAULT_LAWS)
        chosen.update(laws or {})
        exempt = AC_GUARD_EXEMPT.get(ac, frozenset()) | frozenset(
            f for f in AC_GUARD_EXEMPT_NOMINAL.get(ac, frozenset()) if chosen.get(f) == "nominal")
    u_fam, i_fam = np.unique(fam, return_inverse=True)
    u_sys, i_sys = np.unique(system, return_inverse=True)
    table = np.full((u_fam.size, u_sys.size), -1.0)
    for a, f in enumerate(u_fam.tolist()):
        for b, s in enumerate(u_sys.tolist()):
            cls = guard_class(f, s)
            if cls is not None and f not in exempt:
                table[a, b] = GUARD_BETA_DEG[cls]
    limit = table[i_fam.ravel(), i_sys.ravel()]
    ok = np.isfinite(beta_c) & np.isfinite(dt_c)
    out = np.zeros(n, dtype=bool)
    out[ok] = (np.abs(beta_c[ok]) < limit[ok]) & (np.abs(dt_c[ok]) <= GUARD_HALF_WINDOW_S)
    if consistent is not None:
        out &= ~np.broadcast_to(np.asarray(consistent, dtype=bool), (n,))
    if kinematic_auto:
        out &= np.isin(fam, sorted(only))
    return out


# ============================================================================================== geometry
def _wrap(a):
    """Angle in [-pi, pi)."""
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def _sgn0(x):
    """sign with sgn0(0) = +1 (the turn-direction rule at exactly beta = 0)."""
    return np.where(np.asarray(x, dtype=float) >= 0.0, 1.0, -1.0)


def _signed(b, ref):
    """|b| (at least ``_TINY_BETA``) with the sign of ``ref`` (sgn0): the beta of one event's nominal yaw takes the
    event's sign, so its rows turn the same way."""
    return _sgn0(ref) * np.maximum(np.abs(np.asarray(b, dtype=float)), _TINY_BETA)


def _clusters(key, t_event, gap) -> np.ndarray:
    """Event label per row: rows of one ``key`` (satellite) whose predicted event times ``t_event`` [s] form a chain
    with steps of at most ``gap``; a row without a finite time is an event of its own."""
    key = np.asarray(key)
    t_event = np.asarray(t_event, dtype=float)
    n = t_event.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    fin = np.isfinite(t_event)
    te = np.where(fin, t_event, 0.0)
    order = np.lexsort((te, key))
    k, ts, f = key[order], te[order], fin[order]
    new = np.ones(n, dtype=bool)
    new[1:] = (k[1:] != k[:-1]) | (np.diff(ts) > gap) | ~f[1:] | ~f[:-1]
    lab = np.empty(n, dtype=np.int64)
    lab[order] = np.cumsum(new) - 1
    return lab


def _event_median(lab, values, use=None) -> np.ndarray:
    """Median of ``values`` over the rows (``use``, finite) of each event label, per row; rows of an event without
    such rows keep their own value."""
    values = np.asarray(values, dtype=float)
    out = values.copy()
    m = np.isfinite(values) if use is None else (np.asarray(use, dtype=bool) & np.isfinite(values))
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return out
    lab_m, val = lab[idx], values[idx]
    order = np.lexsort((val, lab_m))
    lab_m, val = lab_m[order], val[order]
    starts = np.flatnonzero(np.r_[True, lab_m[1:] != lab_m[:-1]])
    counts = np.diff(np.r_[starts, lab_m.size])
    med = 0.5 * (val[starts + (counts - 1) // 2] + val[starts + counts // 2])
    table = np.full(int(lab.max()) + 1, np.nan)
    table[lab_m[starts]] = med
    got = table[lab]
    has = np.isfinite(got)
    out[has] = got[has]
    return out


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _unit(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    ok = n > 0.0
    return np.where(ok, a / np.where(ok, n, 1.0), 0.0)


def nominal_yaw(beta, mu):
    """Nominal yaw-steering angle psi_n = atan2(-tan(beta), sin(mu)) [rad] (Kouba 2009 eq. 1), as
    atan2(-sin(beta), cos(beta) sin(mu)), which is the same for |beta| < 90 deg."""
    beta = np.asarray(beta, dtype=float)
    return np.arctan2(-np.sin(beta), np.cos(beta) * np.sin(np.asarray(mu, dtype=float)))


def nominal_yaw_rate(beta, mu, mu_dot):
    """d psi_n/dt = mu_dot tan(beta) cos(mu) / (sin^2(mu) + tan^2(beta)) [rad/s] (Kouba 2009 eq. 2; Montenbruck
    et al. 2026 eq. 2); 0 where undefined (beta = 0 at noon/midnight)."""
    tb = np.tan(np.asarray(beta, dtype=float))
    mu = np.asarray(mu, dtype=float)
    den = np.sin(mu) ** 2 + tb ** 2
    ok = den > 0.0
    return np.where(ok, np.asarray(mu_dot, dtype=float) * tb * np.cos(mu) / np.where(ok, den, 1.0), 0.0)


def orbit_frame(r, v_ecef):
    """(e_R, e_T, e_N) per row from ECEF position [m] and ECEF velocity [m/s] (inertial velocity v + w_E x r)."""
    r = np.atleast_2d(np.asarray(r, dtype=float))
    vi = np.atleast_2d(np.asarray(v_ecef, dtype=float)) + np.cross([0.0, 0.0, OMGE], r)
    e_r = _unit(r)
    e_n = _unit(np.cross(r, vi))
    return e_r, np.cross(e_n, e_r), e_n


def body_axes(yaw, e_t, e_n, e_r):
    """IGS body axes (x, y, z) in ECEF from the yaw [rad] and the orbit frame (module docstring)."""
    c = np.cos(np.asarray(yaw, dtype=float))[:, None]
    s = np.sin(np.asarray(yaw, dtype=float))[:, None]
    return c * e_t - s * e_n, -s * e_t - c * e_n, -e_r


def yaw_of_axes(x_b, e_t, e_n):
    """Yaw [rad] of body x axes in the orbit frame: atan2(-x_b . e_N, x_b . e_T)."""
    return np.arctan2(-_dot(x_b, e_n), _dot(x_b, e_t))


def _nominal_axes(r, r_sun):
    """Nominal yaw-steering axes (RTKLIB satantoff), with the arithmetic of GNX before v1.0.0
    (``tools._satellite_body_frame``): bit-identical rows; where the Sun is on the z axis, x = y = 0."""
    k = -r / np.linalg.norm(r, axis=1, keepdims=True)
    e = r_sun - r
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    j = np.cross(k, e)
    nj = np.linalg.norm(j, axis=1, keepdims=True)
    ok = nj > 1e-12
    j = np.where(ok, j / np.where(ok, nj, 1.0), 0.0)
    i = np.cross(j, k)
    return i, j, k


# ---------------------------------------------------------------------------------------------- Sun motion
def _sun_motion(t_gpst_s, s, e_n, beta, n_mean, a, ecc):
    """(beta_dot, omega_m) [rad/s]: the rate of beta, and the rate at which the orbit-midnight direction turns
    about e_N (in the direction of motion), from the Sun's annual motion and the J2 nodal regression.

    Sun: ecliptic longitude rate of the low-precision solar coordinates of the Astronomical Almanac (mean motion
    plus the equation of the centre), about the ecliptic pole of date rotated into ECEF by the Greenwich mean
    sidereal time (Meeus, Astronomical Algorithms, eq. 12.4). Orbit normal: d e_N/dt = Omega_dot z x e_N with
    Omega_dot = -1.5 n J2 (R_E/a)^2 cos(i)/(1 - e^2)^2. Earth rotation turns the Sun and e_N alike in ECEF and
    drops out. Equivalent to differencing beta at t +- 600 s with e_N rotated about z by -(w_E - Omega_dot) dt.
    """
    d = (np.asarray(t_gpst_s, dtype=float) - 18.0) / 86400.0 + (2444244.5 - 2451545.0)   # days from J2000 (UT1)
    g = (357.528 + 0.9856003 * d) * _D2R
    lam_dot = (0.9856474 + (1.915 * np.cos(g) + 0.040 * np.cos(2.0 * g)) * 0.9856003 * _D2R) * _D2R / 86400.0
    eps = (23.439291 - 0.0130042 * d / 36525.0) * _D2R
    theta = (280.46061837 + 360.98564736629 * d) * _D2R
    k = np.column_stack([-np.sin(theta) * np.sin(eps), -np.cos(theta) * np.sin(eps), np.cos(eps)])
    s_dot = lam_dot[:, None] * np.cross(k, s)
    cos_i = e_n[:, 2]
    node_dot = -1.5 * n_mean * J2 * (RE / a) ** 2 * cos_i / (1.0 - ecc ** 2) ** 2
    en_dot = node_dot[:, None] * np.cross([0.0, 0.0, 1.0], e_n)
    cb = np.maximum(np.cos(beta), 1e-12)                     # |beta| < 90 deg for every GNSS orbit
    beta_dot = (_dot(s_dot, e_n) + _dot(s, en_dot)) / cb
    m_hat = -_unit(s - _dot(s, e_n)[:, None] * e_n)
    u = np.cross(e_n, m_hat)
    omega_m = -(_dot(s_dot, u) - np.sin(beta) * _dot(en_dot, u)) / cb
    return beta_dot, omega_m


# ---------------------------------------------------------------------------------------------- two-body orbit
def _kepler_ecc(M, e):
    """Eccentric anomaly from the mean anomaly M in [-pi, pi) (Newton, e < 0.3)."""
    E = M + e * np.sin(M)
    for _ in range(12):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def _true_from_ecc(E, e):
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(0.5 * E), np.sqrt(1.0 - e) * np.cos(0.5 * E))


def _ecc_from_true(nu, e):
    return 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(0.5 * nu), np.sqrt(1.0 + e) * np.cos(0.5 * nu))


class _Orbit:
    """Per-row propagation of the orbit angle mu (two-body Kepler motion in the orbital plane minus the turn of the
    midnight direction), beta (linear), the radius and the nominal yaw. Arrays of shape (k,); time offsets dt of
    shape (k,) or (k, m) [s] from the row time."""

    _FIELDS = ("n", "e", "M", "nu", "C", "a", "h", "mu", "beta", "beta_dot", "omega_m")

    def __init__(self, **kw):
        for f in self._FIELDS:
            setattr(self, f, np.asarray(kw[f], dtype=float))

    def take(self, idx) -> "_Orbit":
        return _Orbit(**{f: getattr(self, f)[idx] for f in self._FIELDS})

    @staticmethod
    def _c(x, like):
        return x[:, None] if np.ndim(like) == 2 else x

    def _dnu(self, dt):
        """Unwrapped true-anomaly advance after dt."""
        n, e, M, C = (self._c(getattr(self, f), dt) for f in ("n", "e", "M", "C"))
        Mt = _wrap(M + n * dt)
        nut = _true_from_ecc(_kepler_ecc(Mt, e), e)
        return n * dt + _wrap(nut - Mt) - C

    def advance(self, dt):
        """Unwrapped mu advance [rad] after dt [s]."""
        dt = np.asarray(dt, dtype=float)
        return self._dnu(dt) - self._c(self.omega_m, dt) * dt

    def time_to(self, dmu):
        """Time [s] after which mu has advanced by the unwrapped angle dmu [rad] (negative: before)."""
        dmu = np.asarray(dmu, dtype=float)
        n, e, nu, C, om = (self._c(getattr(self, f), dmu) for f in ("n", "e", "nu", "C", "omega_m"))
        dt = np.zeros_like(dmu)
        for _ in range(3):
            dnu = dmu + om * dt
            nut = _wrap(nu + dnu)
            Et = _ecc_from_true(nut, e)
            Ct = _wrap(nut - (Et - e * np.sin(Et)))
            dt = (dnu - Ct + C) / n
        return dt

    def beta_at(self, dt):
        return self._c(self.beta, dt) + self._c(self.beta_dot, dt) * np.asarray(dt, dtype=float)

    def mu_at(self, dt):
        return _wrap(self._c(self.mu, dt) + self.advance(dt))

    def mu_dot_at(self, dt):
        """|r x v|/|r|^2 [rad/s] at t + dt (two-body radius)."""
        n, e, M, a, h = (self._c(getattr(self, f), dt) for f in ("n", "e", "M", "a", "h"))
        E = _kepler_ecc(_wrap(M + n * np.asarray(dt, dtype=float)), e)
        rr = a * (1.0 - e * np.cos(E))
        return h / rr ** 2

    def psi_n_at(self, dt):
        return nominal_yaw(self.beta_at(dt), self.mu_at(dt))


@dataclass
class _Geometry:
    """Per-row geometry of the rows with a valid position, velocity and Sun (index ``rows`` of the input)."""

    rows: np.ndarray
    e_r: np.ndarray
    e_t: np.ndarray
    e_n: np.ndarray
    s: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    mu_dot: np.ndarray
    orbit: _Orbit
    kepler_ok: np.ndarray


def _geometry(t, r, v, r_sun, rows) -> _Geometry:
    r = r[rows]
    vi = v[rows] + np.cross([0.0, 0.0, OMGE], r)
    rn = np.linalg.norm(r, axis=1)
    e_r = r / rn[:, None]
    hv = np.cross(r, vi)
    h = np.linalg.norm(hv, axis=1)
    e_n = hv / h[:, None]
    e_t = np.cross(e_n, e_r)
    s = r_sun[rows] / np.linalg.norm(r_sun[rows], axis=1, keepdims=True)
    s_n = np.clip(_dot(s, e_n), -1.0, 1.0)
    beta = np.arcsin(s_n)
    mu = np.arctan2(_dot(s, e_t), -_dot(s, e_r))
    mu_dot = h / rn ** 2
    # osculating two-body elements in the ECEF-aligned inertial frame of the row
    ev = np.cross(vi, hv) / GM - e_r
    ecc = np.linalg.norm(ev, axis=1)
    inv_a = 2.0 / rn - _dot(vi, vi) / GM
    kepler_ok = (inv_a > 0.0) & (ecc < 0.5)
    a = np.where(kepler_ok, 1.0 / np.where(kepler_ok, inv_a, 1.0), rn)
    ecc = np.where(kepler_ok, ecc, 0.0)
    n_mean = np.where(kepler_ok, np.sqrt(GM / a ** 3), mu_dot)
    nu = np.arctan2(_dot(np.cross(ev, e_r), e_n), _dot(ev, e_r))
    nu = np.where(kepler_ok, nu, 0.0)
    E = _ecc_from_true(nu, ecc)
    M = E - ecc * np.sin(E)
    beta_dot, omega_m = _sun_motion(t[rows], s, e_n, beta, n_mean, a, ecc)
    orbit = _Orbit(n=n_mean, e=ecc, M=M, nu=nu, C=_wrap(nu - M), a=a, h=h, mu=mu, beta=beta, beta_dot=beta_dot,
                   omega_m=omega_m)
    return _Geometry(rows=rows, e_r=e_r, e_t=e_t, e_n=e_n, s=s, beta=beta, mu=mu, mu_dot=mu_dot, orbit=orbit,
                     kepler_ok=kepler_ok)


def _nearest_turn(g: _Geometry):
    """(noon, delta_c, dt_c): the nearer of noon and midnight, the unwrapped mu advance to it and the time to it."""
    d_mid = _wrap(-g.mu)
    d_noon = _wrap(math.pi - g.mu)
    noon = np.abs(d_noon) < np.abs(d_mid)
    delta_c = np.where(noon, d_noon, d_mid)
    return noon, delta_c, g.orbit.time_to(delta_c)


# ============================================================================================== laws
def _law_catchup(o: _Orbit, noon, delta_c, dt_c, rate_deg_s, noon_reversal_deg=0.0, midnight=True, ev=None):
    """Catch-up yaw steering (Kouba 2009). Returns (active, psi) for the rows of ``o``. ``ev``: passage label per
    row (``_clusters``; None: every row its own): the beta test and the direction are made once per passage, on
    the median beta at the centre, and each row's beta line passes through it."""
    k = o.beta.size
    psi = np.zeros(k)
    active = np.zeros(k, dtype=bool)
    ev = np.arange(k) if ev is None else ev
    R = rate_deg_s * _D2R
    mu_dot_c = o.mu_dot_at(dt_c)
    beta0 = np.arctan(mu_dot_c / R)
    beta_c = o.beta_at(dt_c)
    beta_e = _event_median(ev, beta_c)
    cand = _event_median(ev, np.abs(beta_c) - beta0) < 0.0
    if not midnight:
        cand &= noon
    # rows from the turn start (at most beta0/2 before the centre) to its end (at most 360 deg / R after it)
    cand &= (-dt_c <= 2.0 * math.pi / R + 600.0) & (dt_c <= 0.5 * beta0 / mu_dot_c + 600.0)
    if not cand.any():
        return active, psi
    idx = np.flatnonzero(cand)
    q = o.take(idx)
    be = beta_e[idx]
    q.beta = q.beta + (be - beta_c[idx])            # beta(centre) = the passage's beta
    dc, tc, b0 = delta_c[idx], dt_c[idx], beta0[idx]
    bs = be
    for _ in range(2):    # start angle from beta at the start (beta varies by < 1e-3 deg over a turn)
        h = np.sqrt(np.clip(np.abs(bs) * (b0 - np.abs(bs)), 0.0, None))
        ts = q.time_to(dc - h)
        bs = _signed(q.beta_at(ts), be)
    mu_c = np.where(noon[idx], math.pi, 0.0)
    psi_s = nominal_yaw(bs, mu_c - h)
    direction = np.where(noon[idx], -_sgn0(be + noon_reversal_deg * _D2R), _sgn0(be))
    tau = -ts                                       # time since the start
    started = tau >= 0.0
    before_centre = tc > 0.0
    inside = started & before_centre
    # after the centre: the turn ends at the first meeting with the nominal yaw (mod 360 deg)
    late = np.flatnonzero(started & ~before_centre)
    if late.size:
        ql = q.take(late)
        t0 = tc[late]
        t1 = ts[late] + 2.0 * math.pi / R + 120.0
        m = 97
        grid = t0[:, None] + (t1 - t0)[:, None] * np.linspace(0.0, 1.0, m)[None, :]
        lin = psi_s[late][:, None] + direction[late][:, None] * R * (grid - ts[late][:, None])
        dd = _wrap(ql.psi_n_at(grid) - lin) * direction[late][:, None]
        meet = (dd[:, :-1] > 0.0) & (dd[:, 1:] <= 0.0) & (np.abs(dd[:, :-1]) < 0.5 * math.pi) & \
               (np.abs(dd[:, 1:]) < 0.5 * math.pi)
        has = meet.any(axis=1)
        j = np.argmax(meet, axis=1)
        lo = grid[np.arange(late.size), j]
        hi = grid[np.arange(late.size), np.minimum(j + 1, m - 1)]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lin_m = psi_s[late] + direction[late] * R * (mid - ts[late])
            d_m = _wrap(ql.psi_n_at(mid) - lin_m) * direction[late]
            pos = d_m > 0.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        t_end = np.where(has, 0.5 * (lo + hi), t1)
        inside[late] = 0.0 < t_end      # the row (dt = 0) lies before the end of the turn
    sel = idx[inside]
    active[sel] = True
    psi[sel] = _wrap(psi_s[inside] + direction[inside] * R * tau[inside])
    return active, psi


def _law_iif_shadow(o: _Orbit, noon, delta_c, dt_c, ev=None):
    """GPS IIF midnight: constant yaw rate across the shadow cone (Kuang et al. 2017; Kouba eclips). ``ev`` as for
    ``_law_catchup``: the direction is the sign of the passage's beta (positive at exactly 0)."""
    k = o.beta.size
    psi = np.zeros(k)
    active = np.zeros(k, dtype=bool)
    ev = np.arange(k) if ev is None else ev
    beta_c = o.beta_at(dt_c)
    beta_e = _event_median(ev, beta_c)
    ratio = math.cos(SHADOW_HALF_ANGLE_DEG * _D2R) / np.cos(beta_e)
    cand = ~noon & (ratio < 1.0) & (np.abs(dt_c) < 3600.0)
    if not cand.any():
        return active, psi
    idx = np.flatnonzero(cand)
    q = o.take(idx)
    be = beta_e[idx]
    q.beta = q.beta + (be - beta_c[idx])
    h = np.arccos(ratio[idx])
    t_in = q.time_to(delta_c[idx] - h)
    t_out = q.time_to(delta_c[idx] + h)
    inside = (t_in <= 0.0) & (t_out >= 0.0) & (t_out > t_in)
    if not inside.any():
        return active, psi
    # nominal yaw at entry and exit with the passage's sign of beta: the short route turns by +sign(beta)
    p_in = nominal_yaw(_signed(q.beta_at(t_in), be), -h)
    p_out = nominal_yaw(_signed(q.beta_at(t_out), be), h)
    rate = _wrap(p_out - p_in) / np.where(t_out > t_in, t_out - t_in, 1.0)
    sel = idx[inside]
    active[sel] = True
    psi[sel] = _wrap(p_in[inside] + rate[inside] * (-t_in[inside]))
    return active, psi


def _law_aux_sun(o: _Orbit, delta_c, cx_deg, cy_deg, ev=None):
    """Auxiliary Sun vector: GSC 3.1.1 (Galileo IOV) / Montenbruck et al. 2026 eqs. 17-18 (GPS III). ``ev`` as for
    ``_law_catchup``: the entry sign of one passage from the median beta at its entry."""
    cx, cy = math.sin(cx_deg * _D2R), math.sin(cy_deg * _D2R)
    sx = np.cos(o.beta) * np.sin(o.mu)
    sy = -np.sin(o.beta)
    active = (np.abs(sx) < cx) & (np.abs(sy) < cy)
    psi = np.zeros(o.beta.size)
    if not active.any():
        return active, psi
    idx = np.flatnonzero(active)
    q = o.take(idx)
    # entry into the region (|s_x| = sin cx before the turn centre): the sign of s_y there is kept
    d_e = np.arcsin(np.clip(cx / np.cos(q.beta), 0.0, 1.0))
    t_e = q.time_to(delta_c[idx] - d_e)
    b_entry = q.beta_at(np.minimum(t_e, 0.0))
    if ev is not None:
        b_entry = _event_median(ev[idx], b_entry)
    gamma = _sgn0(-np.sin(b_entry))
    g = np.cos(math.pi * np.abs(sx[idx]) / cx)
    sy_star = 0.5 * (1.0 + g) * gamma * cy + 0.5 * (1.0 - g) * sy[idx]
    psi[idx] = np.arctan2(sy_star, sx[idx])
    return active, psi


def _law_cosine(o: _Orbit, noon, delta_c, beta_lim_deg, switch_deg, period_s, beta_strict=True, ev=None):
    """psi = 90 S + (psi_s - 90 S) cos(2 pi (t - t_s)/T) for T/2 from ``switch_deg`` before noon/midnight:
    Galileo FOC (GSC 3.1.2) and BDS-3 CAST (Wang et al. 2018). ``ev`` as for ``_law_catchup``: the beta test and
    the sign S of one passage from the median beta at its start."""
    k = o.beta.size
    psi = np.zeros(k)
    ev = np.arange(k) if ev is None else ev
    sw = switch_deg * _D2R
    ts = o.time_to(delta_c - sw)
    bs = o.beta_at(ts)
    lim = beta_lim_deg * _D2R
    margin = _event_median(ev, np.abs(bs) - lim)
    ok_beta = margin < 0.0 if beta_strict else margin <= 0.0
    tau = -ts
    active = ok_beta & (tau >= 0.0) & (tau <= 0.5 * period_s)
    if active.any():
        mu_s = np.where(noon, math.pi, 0.0) - sw
        p0 = nominal_yaw(_signed(bs, _event_median(ev, bs)), mu_s)
        S = _sgn0(p0) * (0.5 * math.pi)
        p = S + (p0 - S) * np.cos(2.0 * math.pi * tau / period_s)
        psi[active] = _wrap(p[active])
    return active, psi


def _law_dilssner(o: _Orbit):
    """Dilssner (2017) smoothed yaw steering of BDS-2 continuous-yaw satellites (and COD's BDS-3 CAST law)."""
    b0 = DILSSNER_BETA_DEG * _D2R
    active = np.abs(o.beta) < b0
    f = 1.0 / (1.0 + DILSSNER_D * np.sin(o.mu) ** 4)
    beta_d = np.where(active, o.beta + f * (_sgn0(o.beta) * b0 - o.beta), o.beta)
    return active, nominal_yaw(beta_d, o.mu)


def _accumulated(o: _Orbit, dt_back):
    """mu at t + dt_back (dt_back <= 0) and the unwrapped mu advance from there to the row."""
    adv = -o.advance(dt_back)
    return _wrap(o.mu - adv), adv


def _first_window_pass(o: _Orbit, dt_event, yaw_limit_deg, beta_deg=None):
    """True where, since the event at t + dt_event (<= 0), the satellite has entered the window where the yaw
    atan2(-tan(b), sin(mu)) is within +-yaw_limit_deg, i.e. sin(mu) > tan|b|/tan(yaw_limit) - or was inside it at
    the event. b: ``beta_deg`` (SECM: the fixed 3 deg), else the beta at the window entry (BDS-2 YS/ON)."""
    tan_lim = math.tan(yaw_limit_deg * _D2R)
    mu0, adv = _accumulated(o, dt_event)

    def edge(b):
        return np.clip(np.tan(np.abs(b)) / tan_lim, 0.0, 1.0)

    b_fixed = None if beta_deg is None else beta_deg * _D2R
    inside0 = np.sin(mu0) > edge(o.beta_at(dt_event) if b_fixed is None else b_fixed)
    entry = np.arcsin(edge(o.beta_at(dt_event) if b_fixed is None else b_fixed))
    need = (entry - mu0) % (2.0 * math.pi)
    if b_fixed is None:      # the window edge moves with beta: take beta at the entry (two iterations)
        for _ in range(2):
            dt_w = o.time_to(need - adv)
            entry = np.arcsin(edge(o.beta_at(dt_w)))
            need = (entry - mu0) % (2.0 * math.pi)
    return inside0 | (adv >= need)


def _crossing_dt(o: _Orbit, f, f_dot, key=None, t=None):
    """Time [s] from each row to the crossing of f = 0 (f linear in time at the rate f_dot; NaN where f_dot = 0;
    negative: in the past). With ``key`` (satellite and crossing direction per row) and ``t`` (row times [s]): the
    median crossing time over the rows of one crossing that lie within 1.05 periods of it, so the rows of one
    crossing switch at one time."""
    moving = f_dot != 0.0
    dt = np.where(moving, -f / np.where(moving, f_dot, 1.0), np.nan)
    if key is None or t is None or dt.size == 0:
        return dt
    t_x = t + dt
    lab = _clusters(key, t_x, CROSSING_GAP_S)
    near = np.abs(dt) <= 1.05 * 2.0 * math.pi / o.n
    return _event_median(lab, t_x, use=near) - t


def _law_secm(o: _Orbit, yang=True, sv=None, t=None):
    """BDS-3 SECM: CSNO 2019 5.4 with the sign latch of Yang et al. (2023). ``sv``/``t``: satellite code and time
    per row; the rows of one beta zero crossing take its median time (``_crossing_dt``)."""
    b0 = SECM_BETA_DEG * _D2R
    active = np.abs(o.beta) < b0
    sigma = _sgn0(o.beta)
    if yang and active.any():
        idx = np.flatnonzero(active)
        q = o.take(idx)
        dt0 = _crossing_dt(q, q.beta, q.beta_dot, None if sv is None else sv[idx], None if t is None else t[idx])
        new = _sgn0(q.beta_dot)                   # the sign of beta after the crossing
        crossed = dt0 <= 0.0
        sig = np.where(np.isnan(dt0), _sgn0(q.beta), np.where(crossed, new, -new))
        recent = crossed & (-dt0 <= 1.05 * 2.0 * math.pi / q.n)
        if recent.any():
            r_idx = np.flatnonzero(recent)
            switched = _first_window_pass(q.take(r_idx), dt0[r_idx], SECM_SWITCH_YAW_DEG, beta_deg=SECM_BETA_DEG)
            sig[r_idx] = np.where(switched, new[r_idx], -new[r_idx])
        sigma[idx] = sig
    psi = nominal_yaw(sigma * b0, o.mu)
    return active, psi


def _law_ys_on(o: _Orbit, sv=None, t=None):
    """BDS-2 YS/ON switching (Li et al. 2018): orbit-normal from the first |psi_n| <= 5 deg pass after |beta|
    falls below 4 deg, yaw steering from the first such pass after |beta| exceeds 4 deg. Returns (on, psi).
    ``sv``/``t`` as for ``_law_secm`` (the rows of one crossing of |beta| = 4 deg take its median time)."""
    lim = YSON_BETA_DEG * _D2R
    b = np.abs(o.beta)
    b_dot = _sgn0(o.beta) * o.beta_dot
    period = 2.0 * math.pi / o.n
    down = b_dot < 0.0                            # the nearest crossing of |beta| = 4 deg is a downward one
    key = None if sv is None else 2 * np.asarray(sv, dtype=np.int64) + down
    dt4 = _crossing_dt(o, b - lim, b_dot, key, t)
    crossed = dt4 <= 0.0
    # before a downward crossing (and long after an upward one): yaw steering; else orbit-normal
    on = np.where(np.isnan(dt4), b < lim, np.where(crossed, down, ~down))
    recent = crossed & (-dt4 <= 1.05 * period)
    if recent.any():
        idx = np.flatnonzero(recent)
        switched = _first_window_pass(o.take(idx), dt4[idx], YSON_SWITCH_YAW_DEG)
        on[idx] = np.where(down[idx], switched, ~switched)
    return on, np.zeros(b.size)


# ============================================================================================== attitude
@dataclass
class Attitude:
    """Satellite attitude per row (``satellite_attitude``).

    x, y, z: (n, 3) IGS body axes in ECEF; yaw_deg, yaw_nominal_deg: yaw of the attitude and of nominal yaw
    steering (NaN without a velocity); regime: ``REGIMES`` code (int8); source: ``SOURCES`` code (int8); guard:
    turn-guard rows (``guard_mask``); beta_deg, mu_deg: Sun elevation above the orbital plane and orbit angle
    from midnight (NaN without a velocity).
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    yaw_deg: np.ndarray
    yaw_nominal_deg: np.ndarray
    regime: np.ndarray
    source: np.ndarray
    guard: np.ndarray
    beta_deg: Optional[np.ndarray] = None
    mu_deg: Optional[np.ndarray] = None


def _gpst_seconds(t) -> np.ndarray:
    """GPS seconds since 1980-01-06 00:00:00 GPST from seconds or datetime-like values."""
    arr = np.asarray(t)
    if arr.dtype.kind in "iuf":
        return np.atleast_1d(arr.astype(float))
    if arr.dtype.kind == "M":
        return np.atleast_1d((arr - np.datetime64("1980-01-06T00:00:00")) / np.timedelta64(1, "ns") * 1e-9)
    vals = np.atleast_1d(arr).astype("datetime64[ns]")
    return (vals - np.datetime64("1980-01-06T00:00:00")) / np.timedelta64(1, "ns") * 1e-9


def _per_row(values, n, name) -> np.ndarray:
    """String array of length n from one value or one value per row."""
    if isinstance(values, str) or np.ndim(values) == 0:
        return np.full(n, str(values))
    out = np.asarray(values).astype(str).ravel()
    if out.size != n:
        raise ValueError(f"satellite_attitude: {name} has {out.size} values for {n} rows")
    return out


def _row_families(svs, families) -> np.ndarray:
    n = svs.size
    if isinstance(families, Mapping):
        u, inv = np.unique(svs, return_inverse=True)
        fam_u = [families.get(s, families.get(s.split("_", 1)[0])) for s in u.tolist()]
        missing = [s for s, f in zip(u.tolist(), fam_u) if f is None]
        if missing:
            warnings.warn(f"satellite_attitude: no attitude family for {', '.join(missing)}; the nominal attitude "
                          f"(yaw steering) is used", UserWarning, stacklevel=3)
        fam_u = ["unknown" if f is None else f for f in fam_u]
        out = np.asarray(fam_u, dtype=str)[inv.ravel()] if n else np.zeros(0, dtype=str)
    else:
        out = _per_row(families, n, "families")
    for f in np.unique(out).tolist():
        if f not in LAWS:
            raise ValueError(f"satellite_attitude: unknown attitude family {f!r}{_did_you_mean(f, LAWS)}")
    return out


def _geo_velocity(svs, t, r, v, geo) -> np.ndarray:
    """``v`` with the rows ``geo`` (BeiDou GEO with a position) that have no finite velocity filled: the finite
    difference over the satellite's rows in time order (numpy.gradient, as ``corrections.orbit_normal_frame``,
    so a caller's own GEO frame and this one agree), v = 0 for a single row or repeated times."""
    need = geo & ~np.isfinite(v).all(axis=1)
    if not need.any():
        return v
    v = v.copy()
    for s in np.unique(svs[need]).tolist():
        k = np.flatnonzero(geo & (svs == s))
        k = k[np.argsort(t[k], kind="stable")]
        vel = np.zeros((k.size, 3))
        if k.size >= 2 and np.all(np.diff(t[k]) > 0):
            vel = np.gradient(r[k], t[k], axis=0)
        miss = ~np.isfinite(v[k]).all(axis=1)
        v[k[miss]] = vel[miss]
    return v


def _arcs(t: np.ndarray):
    """Index arrays of the continuous arcs of one satellite's sorted times (gap > 1.5 x the median step)."""
    if t.size < 2:
        return [np.arange(t.size)]
    step = np.diff(t)
    pos = step[step > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    cut = np.flatnonzero(step > 1.5 * med) + 1 if med > 0 else np.zeros(0, dtype=int)
    return np.split(np.arange(t.size), cut)


def satellite_attitude(sv, t_gpst_s, r, v_ecef, r_sun, families, *, mode="model", laws=None, orbex=None,
                       guard=None, product_ac=None, static=True, broadcast=False,
                       single_frequency=False) -> Attitude:
    """Body axes of satellites at observation rows (module docstring).

    ``sv``: satellite id per row or one id for all rows; ``t_gpst_s``: GPS seconds since 1980-01-06 (or
    datetime-like, GPST); ``r``, ``v_ecef``, ``r_sun``: (n, 3) ECEF position [m], velocity [m/s] and Sun [m];
    ``families``: {sv: family} (``law_family``), one family, or one per row.

    ``mode``: 'model' (the law of each family, ``laws`` overrides the defaults; ``check_laws``) or 'nominal' (yaw
    steering everywhere; BeiDou GEO orbit-normal in both modes). ``orbex``: an :class:`gnx_py.orbex.OrbexAttitude`
    used per satellite and continuous arc where it covers the whole arc (the law elsewhere, one UserWarning
    naming the satellites). ``guard``/``product_ac``/``static``/``broadcast``/``single_frequency``: the turn guard
    (``guard_mask``);
    ORBEX rows of the product's analysis centre are not guarded, except the families of ``ORBEX_GUARDED``.

    Rows without a finite velocity get the nominal frame from the position and the Sun (regime 9, NaN yaw); the
    caller reports them. A GEO row without a velocity takes the finite difference over its satellite's rows of the
    call (``_geo_velocity``, the arithmetic of ``corrections.orbit_normal_frame``; v = 0 for a single row, exact to
    the orbit inclination). Rows sorted by time per satellite are not required.
    """
    if mode not in ("model", "nominal"):
        raise ValueError(f"satellite_attitude: mode must be 'model' or 'nominal', got {mode!r}"
                         f"{_did_you_mean(str(mode), ('model', 'nominal'))}")
    if guard is not None and guard not in GUARD_MODES:
        raise ValueError(f"attitude_turn_guard must be one of {GUARD_MODES}, got {guard!r}"
                         f"{_did_you_mean(str(guard), GUARD_MODES)}")
    r = np.asarray(r, dtype=float).reshape(-1, 3)
    n = r.shape[0]
    t = _gpst_seconds(t_gpst_s)
    if t.size == 1 and n > 1:
        t = np.full(n, t[0])
    if t.size != n:
        raise ValueError(f"satellite_attitude: {t.size} times for {n} rows")
    v = np.full((n, 3), np.nan) if v_ecef is None else np.asarray(v_ecef, dtype=float).reshape(-1, 3)
    rs = np.asarray(r_sun, dtype=float).reshape(-1, 3)
    if v.shape[0] != n or rs.shape[0] != n:
        raise ValueError("satellite_attitude: r, v_ecef and r_sun must have the same number of rows")
    svs = _per_row(sv, n, "sv")
    fam = _row_families(svs, families)
    chosen = dict(DEFAULT_LAWS)
    chosen.update(check_laws(laws))

    x = np.full((n, 3), np.nan)
    y = np.full((n, 3), np.nan)
    z = np.full((n, 3), np.nan)
    yaw = np.full(n, np.nan)
    yaw_n = np.full(n, np.nan)
    beta_deg = np.full(n, np.nan)
    mu_deg = np.full(n, np.nan)
    regime = np.full(n, 9, dtype=np.int8)
    source = np.zeros(n, dtype=np.int8)
    guard_out = np.zeros(n, dtype=bool)
    if n == 0:
        return Attitude(x, y, z, yaw, yaw_n, regime, source, guard_out, beta_deg, mu_deg)

    pos_ok = np.isfinite(r).all(axis=1) & np.isfinite(rs).all(axis=1) & (np.linalg.norm(np.nan_to_num(r), axis=1) > 0)
    v = _geo_velocity(svs, t, r, v, pos_ok & (fam == "bds_geo"))
    vel_ok = pos_ok & np.isfinite(v).all(axis=1)
    hv = np.cross(np.nan_to_num(r), np.nan_to_num(v) + np.cross([0.0, 0.0, OMGE], np.nan_to_num(r)))
    vel_ok &= np.linalg.norm(hv, axis=1) > 0.0
    if pos_ok.any():                                   # nominal frame first (bit-identical to GNX before 11c)
        ii, jj, kk = _nominal_axes(r[pos_ok], rs[pos_ok])
        x[pos_ok], y[pos_ok], z[pos_ok] = ii, jj, kk
    rows = np.flatnonzero(vel_ok)
    if rows.size == 0:
        return Attitude(x, y, z, yaw, yaw_n, regime, source, guard_out, beta_deg, mu_deg)

    g = _geometry(t, r, v, rs, rows)
    o = g.orbit
    psi_nom = nominal_yaw(g.beta, g.mu)
    psi = psi_nom.copy()
    reg = np.zeros(rows.size, dtype=np.int8)
    src = np.zeros(rows.size, dtype=np.int8)
    noon, delta_c, dt_c = _nearest_turn(g)
    fr = fam[rows]
    t_rows = t[rows]
    sv_code = np.unique(svs[rows], return_inverse=True)[1].ravel()
    ev = _clusters(sv_code, t_rows + dt_c, EVENT_GAP_S)       # noon/midnight passage of each row
    for f in np.unique(fr).tolist():
        sel = np.flatnonzero(fr == f)
        law = "orbit_normal" if f == "bds_geo" else ("nominal" if mode == "nominal" else chosen[f])
        algo = _ALGORITHMS[law]
        if algo == "nominal":
            if f == "unknown" and mode == "model":
                reg[sel] = 9
            continue
        if algo == "orbit_normal":
            psi[sel] = 0.0
            reg[sel] = 7
            src[sel] = 1 if mode == "model" else 0
            continue
        src[sel] = 1
        q = o.take(sel)
        args = (noon[sel], delta_c[sel], dt_c[sel])
        if algo == "catchup_iir":
            act, p = _law_catchup(q, *args, IIR_RATE_DEG_S, ev=ev[sel])
            codes = [(act, p, 1)]
        elif algo == "iif":
            act, p = _law_catchup(q, *args, IIF_RATE_DEG_S, noon_reversal_deg=IIF_NOON_REVERSAL_DEG,
                                  midnight=False, ev=ev[sel])
            act2, p2 = _law_iif_shadow(q, *args, ev=ev[sel])
            codes = [(act, p, 1), (act2 & ~act, p2, 2)]
        elif algo in ("aux_gps3", "aux_iov"):
            cx, cy = GPS3_AUX_DEG if algo == "aux_gps3" else IOV_AUX_DEG
            act, p = _law_aux_sun(q, delta_c[sel], cx, cy, ev=ev[sel])
            codes = [(act, p, 3)]
        elif algo == "foc":
            act, p = _law_cosine(q, noon[sel], delta_c[sel], FOC_BETA_DEG, FOC_SWITCH_DEG, FOC_PERIOD_S,
                                 ev=ev[sel])
            codes = [(act, p, 4)]
        elif algo == "cast":
            act, p = _law_cosine(q, noon[sel], delta_c[sel], CAST_BETA_DEG, CAST_SWITCH_DEG, CAST_PERIOD_S[f],
                                 beta_strict=False, ev=ev[sel])
            codes = [(act, p, 4)]
        elif algo == "dilssner":
            act, p = _law_dilssner(q)
            codes = [(act, p, 6)]
        elif algo in ("secm_yang", "secm_csno"):
            act, p = _law_secm(q, yang=algo == "secm_yang", sv=sv_code[sel], t=t_rows[sel])
            codes = [(act, p, 5)]
        elif algo == "ys_on":
            act, p = _law_ys_on(q, sv=sv_code[sel], t=t_rows[sel])
            codes = [(act, p, 7)]
        else:  # pragma: no cover - _ALGORITHMS and the branches above are kept in step
            raise AssertionError(algo)
        for act, p, code in codes:
            psi[sel[act]] = p[act]
            reg[sel[act]] = code

    # axes: law rows from the yaw; yaw-steering rows keep the nominal frame computed above
    law_rows = reg != 0
    law_rows &= reg != 9
    if law_rows.any():
        xb, yb, zb = body_axes(psi[law_rows], g.e_t[law_rows], g.e_n[law_rows], g.e_r[law_rows])
        x[rows[law_rows]], y[rows[law_rows]], z[rows[law_rows]] = xb, yb, zb

    consistent = np.zeros(rows.size, dtype=bool)
    if orbex is not None:
        _apply_orbex(orbex, svs, t, r, rows, fr, g, psi, reg, src, x, y, z, consistent, product_ac)

    yaw[rows] = np.degrees(_wrap(psi))
    yaw_n[rows] = np.degrees(psi_nom)
    beta_deg[rows] = np.degrees(g.beta)
    mu_deg[rows] = np.degrees(g.mu)
    regime[rows] = reg
    source[rows] = src
    beta_c = np.degrees(_event_median(ev, o.beta_at(dt_c)))     # one guard decision per passage
    guard_out[rows] = guard_mask(fr, beta_c, dt_c, guard=guard, product_ac=product_ac, static=static,
                                 broadcast=broadcast, consistent=consistent, svs=svs[rows],
                                 single_frequency=single_frequency,
                                 laws=chosen if mode == "model" else {f: "nominal" for f in chosen})
    return Attitude(x, y, z, yaw, yaw_n, regime, source, guard_out, beta_deg, mu_deg)


# ============================================================================================== ORBEX
def _apply_orbex(orbex, svs, t, r, rows, fr, g, psi, reg, src, x, y, z, consistent, product_ac) -> None:
    """ORBEX attitude per satellite and continuous arc where it covers the whole arc (in place)."""
    ac = _ac3(getattr(orbex, "ac", None))
    prod = _ac3(product_ac)
    same_ac = prod is None or ac is None or ac == prod
    sv_rows = svs[rows]
    fallback = []
    flipped = []
    rejected = []
    decide = getattr(orbex, "_x_axis_decision", None)
    for sv in np.unique(sv_rows).tolist():
        k = np.flatnonzero(sv_rows == sv)
        k = k[np.argsort(t[rows[k]], kind="stable")]
        xo, yo, zo, cov = orbex.axes(sv, t[rows[k]], r[rows[k]])
        use = np.zeros(k.size, dtype=bool)
        for arc in _arcs(t[rows[k]]):
            if arc.size and cov[arc].all():
                use[arc] = True
        if not use.all():
            fallback.append(sv)
        if not use.any():
            continue
        ku = k[use]
        # x on the side of the law's x outside turns (yaw steering, or orbit-normal for a GEO), per satellite: a
        # frame turned by 180 deg about z (the JPL GPS IIR manufacturer axes, Montenbruck et al. 2015 Table 3) is
        # turned back, an inconsistent one is not used (OrbexAttitude._x_axis_decision)
        clean = (np.abs(g.beta[ku]) > 6.0 * _D2R) & (np.minimum(np.abs(_wrap(g.mu[ku])),
                                                                np.abs(_wrap(g.mu[ku] - math.pi))) > 15.0 * _D2R)
        clean &= (reg[ku] == 0) | (reg[ku] == 7)
        ref = body_axes(psi[ku], g.e_t[ku], g.e_n[ku], g.e_r[ku])[0]
        flip, pos = decide(fr[ku[0]], _dot(xo[use][clean], ref[clean]) if clean.any() else None) if decide \
            else (False, float("nan"))
        if flip is None:
            rejected.append(f"{sv} ({100 * pos:.0f} %)")
            continue
        if flip:
            xo, yo = -xo, -yo
            flipped.append(f"{sv} ({fr[ku[0]]})")
        rk = rows[ku]
        x[rk], y[rk], z[rk] = xo[use], yo[use], zo[use]
        psi[ku] = yaw_of_axes(xo[use], g.e_t[ku], g.e_n[ku])
        reg[ku] = 8
        src[ku] = 2
        consistent[ku] = same_ac and fr[ku[0]] not in ORBEX_GUARDED.get(ac or "", frozenset())
    name = getattr(orbex, "label", None) or getattr(orbex, "product", "ORBEX")
    if flipped:
        warnings.warn(f"attitude_path {name}: the body x axis of {', '.join(flipped)} points away from the Sun side "
                      f"outside turns (a frame turned by 180 deg about z, e.g. the manufacturer axes of GPS IIR); x "
                      f"and y are turned back to the IGS frame", UserWarning, stacklevel=3)
    if rejected:
        from ..io.orbex import SUN_SIDE_FRACTION

        warnings.warn(f"attitude_path {name}: the body x axis of {', '.join(rejected)} is not on the Sun side outside "
                      f"turns (in parentheses: share of the rows outside turns with x on the Sun side; the IGS frame "
                      f"needs at least {100 * SUN_SIDE_FRACTION:.0f} %, a frame turned by 180 deg at most "
                      f"{100 * (1 - SUN_SIDE_FRACTION):.0f} %, orbex.SUN_SIDE_FRACTION); their ORBEX attitude is "
                      f"not used, the attitude law is", UserWarning, stacklevel=3)
    if fallback:
        warnings.warn(f"attitude_path: the ORBEX attitude of {name} does not cover every arc of "
                      f"{', '.join(sorted(set(fallback)))}; the attitude law is used for those arcs",
                      UserWarning, stacklevel=3)
