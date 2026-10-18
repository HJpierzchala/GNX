"""ANTEX 1.4 reader and per-session antenna model (receiver and satellite PCO/PCV).

Format: ANTEX 1.4 (Rothacher & Schmid, IGS format description). Every antenna block holds, per
frequency, the phase-centre offset (PCO, line ``NORTH / EAST / UP``) and the phase-centre
variations (PCV: the ``NOAZI`` row on the zenith/nadir grid and, when ``DAZI`` > 0, one row per
azimuth). PCO and PCV of one frequency form one set and are applied together.

Sign conventions (ANTEX 1.4; RTKLIB ``antmodel``/``satantpcv``; gLAB):

* receiver: observed range = rho(ARP) - PCO_enu . u + PCV(zenith, azimuth), ``u`` receiver -> satellite;
  the PCO line holds North, East, Up [mm];
* satellite: observed range = rho(CoM) - (R_body PCO_xyz) . u + PCV(nadir); the PCO line holds the
  body-frame X, Y, Z [mm] (IGS satellite body frame: z to the Earth, y = z x sun, x completes it).

Units: mm in the file, metres from every accessor.

Lookups:

* satellites by PRN within VALID FROM / VALID UNTIL (a VALID UNTIL second of ``59.9999999`` is read as
  a float and rounds to the next minute); where two entries are valid (the hand-over instant, or no
  epoch given) the one with the latest VALID FROM is used;
* receivers by antenna type and radome; a radome without its own calibration takes the radome ``NONE``
  values with a warning (IGS ATX header hint);
* receiver frequencies without a calibration take the nearest calibrated frequency (same frequency in
  another system first, ties broken by the same system), with a warning (RTKLIB/gLAB practice);
* satellite G05 missing: G01 for a frequency-independent entry (igs20: "L1 and L2 set to the results for
  the ionosphere-free linear combination"), with a warning; otherwise the satellite has no value;
* a PRN whose satellite (SVN) of the first epoch has no entry from an epoch before the session end on, e.g. a
  BeiDou PRN reassigned to another satellite during the session (contiguous entries of the same SVN are one
  validity, not a change): one UserWarning naming the PRN, the SVNs and the change epoch, and every session that
  reads the ANTEX excludes the observations (orbit rows) of that PRN after that epoch
  (:attr:`AntennaModel.prn_changes`, :meth:`AntennaModel.drop_after_prn_change`).

Satellite PCV is nadir-only (``NOAZI``) as RTKLIB ``satantpcv``; igs20 gives azimuth grids for some
Galileo satellites (grid - NOAZI <= 8.2 mm within nadir 14.5 deg, median about 1 mm).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core import _notes
from ..io.compression import head_lines, read_bytes

__all__ = [
    "ANTEX_FREQUENCY_HZ",
    "AntexVersionWarning",
    "AntexFrequency",
    "AntexAntenna",
    "Antex",
    "AntennaModel",
    "load_antex",
    "receiver_frequency",
    "satellite_frequency",
    "sp3_pcv_label",
    "check_version",
    "receiver_antenna_from_header",
    "build_antenna_model",
    "PrnChange",
    "broadcast_apc_reference",
    "missing_satellite_antennas",
    "legacy_sat_table",
    "legacy_rec_table",
]

#: carrier frequency [Hz] of the ANTEX frequency codes (ANTEX 1.4 table; GLONASS FDMA codes are not used
#: for substitution)
ANTEX_FREQUENCY_HZ: Dict[str, float] = {
    "G01": 1575.42e6, "G02": 1227.60e6, "G05": 1176.45e6,
    "E01": 1575.42e6, "E05": 1176.45e6, "E07": 1207.14e6, "E08": 1191.795e6, "E06": 1278.75e6,
    "C01": 1575.42e6, "C02": 1561.098e6, "C05": 1176.45e6, "C06": 1268.52e6, "C07": 1207.14e6,
    "C08": 1191.795e6,
    "J01": 1575.42e6, "J02": 1227.60e6, "J05": 1176.45e6, "J06": 1278.75e6,
}

#: systems whose frequencies may stand in for a missing receiver calibration (order = tie-break order
#: after the system of the requested frequency)
_SUBSTITUTE_SYSTEMS = ("G", "E", "C", "J")

_LABELS = {
    "TYPE / SERIAL NO", "METH / BY / # / DATE", "DAZI", "ZEN1 / ZEN2 / DZEN", "# OF FREQUENCIES",
    "VALID FROM", "VALID UNTIL", "SINEX CODE", "COMMENT", "START OF FREQUENCY", "NORTH / EAST / UP",
    "END OF FREQUENCY", "START OF FREQ RMS", "END OF FREQ RMS", "START OF ANTENNA", "END OF ANTENNA",
}


class AntexVersionWarning(UserWarning):
    """The ANTEX differs from the antenna model of the orbit/clock product (week or frame)."""


# --------------------------------------------------------------------------- data structures
@dataclass
class AntexFrequency:
    """One frequency block: PCO [mm] and PCV [mm] on the zenith (receiver) / nadir (satellite) grid."""

    pco_mm: np.ndarray
    noazi_mm: Optional[np.ndarray] = None
    #: (n_azimuth, n_zenith), azimuth 0..360 deg in steps of DAZI (only when DAZI > 0)
    grid_mm: Optional[np.ndarray] = None


@dataclass
class AntexAntenna:
    """One ANTEX antenna block (receiver type calibration or one satellite SVN)."""

    type: str            # columns 1-20: antenna (1-16) + radome (17-20); satellites: block name
    serial: str          # columns 21-40: receivers serial (blank = type mean); satellites the PRN 'sNN'
    svn: str = ""        # columns 41-50 (satellites)
    cospar: str = ""     # columns 51-60 (satellites)
    dazi: float = 0.0
    zen1: float = 0.0
    zen2: float = 90.0
    dzen: float = 5.0
    valid_from: datetime = datetime.min
    valid_until: datetime = datetime.max
    sinex_code: str = ""
    freqs: Dict[str, AntexFrequency] = field(default_factory=dict)
    line_no: int = 0

    # ------------------------------------------------------------------ identity
    @property
    def is_satellite(self) -> bool:
        s = self.serial.strip()
        return bool(self.svn.strip()) and len(s) == 3 and s[0] in "GRECJSI" and s[1:].isdigit()

    @property
    def prn(self) -> str:
        return self.serial.strip()

    @property
    def antenna(self) -> str:
        return self.type[:16].strip()

    @property
    def radome(self) -> str:
        return self.type[16:20].strip() or "NONE"

    def zen_grid(self) -> np.ndarray:
        n = int(round((self.zen2 - self.zen1) / self.dzen)) + 1 if self.dzen > 0 else 1
        return self.zen1 + self.dzen * np.arange(n)

    def valid_at(self, epoch: Optional[datetime]) -> bool:
        return epoch is None or self.valid_from <= epoch <= self.valid_until

    # ------------------------------------------------------------------ values
    def has(self, freq: str) -> bool:
        return freq in self.freqs

    def pco_neu_m(self, freq: str) -> Optional[np.ndarray]:
        """PCO [m] as in the file: receivers North, East, Up; satellites body-frame X, Y, Z."""
        f = self.freqs.get(freq)
        return None if f is None else np.asarray(f.pco_mm, dtype=float) / 1000.0

    def pco_enu_m(self, freq: str) -> Optional[np.ndarray]:
        """Receiver PCO [m] as East, North, Up (the order of a local ENU -> ECEF rotation)."""
        neu = self.pco_neu_m(freq)
        return None if neu is None else neu[[1, 0, 2]]

    def pco_xyz_m(self, freq: str) -> Optional[np.ndarray]:
        """Satellite PCO [m] in the body frame X, Y, Z (same numbers as :meth:`pco_neu_m`)."""
        return self.pco_neu_m(freq)

    def pcv_m(self, freq: str, zen_deg, az_deg=None) -> np.ndarray:
        """PCV [m] at zenith (receiver) or nadir (satellite) angles [deg].

        Bilinear on the azimuth grid when DAZI > 0 and ``az_deg`` is given (azimuth clockwise from north),
        else linear on the NOAZI row. Angles outside the grid take the edge value (RTKLIB ``interpvar``).
        A frequency without PCV rows (or unknown) gives zeros.
        """
        zen = np.asarray(zen_deg, dtype=float)
        f = self.freqs.get(freq)
        if f is None or f.noazi_mm is None:
            return np.zeros_like(zen, dtype=float)
        zg = self.zen_grid()
        noazi = np.asarray(f.noazi_mm, dtype=float)
        if noazi.size != zg.size:                       # inconsistent block: use the shorter common part
            n = min(noazi.size, zg.size)
            zg, noazi = zg[:n], noazi[:n]
        z = np.clip(zen, zg[0], zg[-1])
        if az_deg is not None and self.dazi > 0 and f.grid_mm is not None and f.grid_mm.shape[0] >= 2 \
                and f.grid_mm.shape[1] == zg.size and zg.size >= 2:
            V = np.asarray(f.grid_mm, dtype=float)
            a, zb = np.broadcast_arrays(np.mod(np.asarray(az_deg, dtype=float), 360.0), z)
            out = np.full(a.shape, np.nan)
            ok = np.isfinite(a) & np.isfinite(zb)       # NaN angles (no position) give NaN, as the NOAZI row
            a, zk = a[ok], zb[ok]
            ia = np.clip(np.floor(a / self.dazi).astype(int), 0, V.shape[0] - 2)
            wa = (a - ia * self.dazi) / self.dazi
            iz = np.clip(np.searchsorted(zg, zk, side="right") - 1, 0, zg.size - 2)
            wz = (zk - zg[iz]) / (zg[iz + 1] - zg[iz])
            v = ((1 - wa) * ((1 - wz) * V[ia, iz] + wz * V[ia, iz + 1])
                 + wa * ((1 - wz) * V[ia + 1, iz] + wz * V[ia + 1, iz + 1]))
            out[ok] = v / 1000.0
            return out
        if zg.size == 1:
            return np.full_like(z, noazi[0] / 1000.0, dtype=float)
        return np.interp(z, zg, noazi) / 1000.0

    def frequency_independent(self) -> bool:
        """True when every frequency has the same PCO and the same PCV (NOAZI and grid)."""
        vals = list(self.freqs.values())
        if not vals:
            return False
        ref = vals[0]
        for f in vals[1:]:
            if not np.allclose(f.pco_mm, ref.pco_mm, atol=1e-9):
                return False
            for a, b in ((f.noazi_mm, ref.noazi_mm), (f.grid_mm, ref.grid_mm)):
                if (a is None) != (b is None):
                    return False
                if a is not None and (np.shape(a) != np.shape(b) or not np.allclose(a, b, atol=1e-9)):
                    return False
        return True


# --------------------------------------------------------------------------- parsing
def _antex_datetime(line: str) -> datetime:
    """VALID FROM/UNTIL: year month day hour minute + float seconds (59.9999999 -> next minute)."""
    p = line[:43].split()
    y, mo, d, h, mi = (int(x) for x in p[:5])
    sec = float(p[5]) if len(p) > 5 else 0.0
    return datetime(y, mo, d, h, mi) + timedelta(seconds=sec)


def _floats(s: str, width: int = 8) -> List[float]:
    s = s.rstrip()
    out = []
    for k in range(0, len(s), width):
        t = s[k:k + width].strip()
        if t:
            out.append(float(t))
    return out


def _open_text(path: Path) -> List[str]:
    # gzip, Unix compress (.Z), bzip2 and zip by content, in memory (gnx_py.compression, bug-fix stage 10)
    return read_bytes(path, what="ANTEX file").decode("latin-1").splitlines()


class Antex:
    """Parsed ANTEX file. Use :func:`load_antex` (cached) rather than the constructor."""

    def __init__(self, path):
        self.path = Path(path)
        self.version: Optional[float] = None
        self.system: Optional[str] = None
        self.pcv_type: Optional[str] = None
        self.antennas: List[AntexAntenna] = []
        self._parse()
        codes = [a.sinex_code for a in self.antennas if a.sinex_code]
        #: most frequent SINEX CODE of the antenna blocks, e.g. 'IGS20_2437'
        self.sinex_code = max(set(codes), key=codes.count) if codes else ""
        self._by_prn: Dict[str, List[AntexAntenna]] = {}
        for a in self.antennas:
            if a.is_satellite:
                self._by_prn.setdefault(a.prn, []).append(a)

    @property
    def name(self) -> str:
        return self.path.name

    def _parse(self) -> None:
        lines = _open_text(self.path)
        i, n = 0, len(lines)
        while i < n:
            lab = lines[i][60:80].strip()
            if lab == "ANTEX VERSION / SYST":
                try:
                    self.version = float(lines[i][:8])
                except ValueError:
                    self.version = None
                self.system = lines[i][20:21].strip() or None
            elif lab == "PCV TYPE / REFANT":
                self.pcv_type = lines[i][:1].strip() or None
            elif lab == "END OF HEADER":
                i += 1
                break
            i += 1
        if self.version is None or self.version < 1.4 - 1e-9:
            warnings.warn(f"{self.path.name}: ANTEX version {self.version} (the reader expects 1.4)",
                          UserWarning, stacklevel=3)
        if self.pcv_type not in ("A", None):
            warnings.warn(f"{self.path.name}: PCV type {self.pcv_type!r} is not absolute ('A')",
                          UserWarning, stacklevel=3)
        while i < n:
            if lines[i][60:80].strip() != "START OF ANTENNA":
                i += 1
                continue
            start = i
            ant: Optional[AntexAntenna] = None
            cur: Optional[str] = None
            rows: List[np.ndarray] = []
            i += 1
            while i < n and lines[i][60:80].strip() != "END OF ANTENNA":
                line = lines[i]
                # labels sit in columns 61-80; data rows are longer or carry numbers there
                lab = line[60:80].strip()
                if lab not in _LABELS:
                    lab = ""
                if lab == "TYPE / SERIAL NO":
                    ant = AntexAntenna(type=line[0:20].ljust(20), serial=line[20:40], svn=line[40:50],
                                       cospar=line[50:60], line_no=start + 1)
                elif ant is None:
                    pass
                elif lab == "DAZI":
                    ant.dazi = float(line[2:8])
                elif lab == "ZEN1 / ZEN2 / DZEN":
                    ant.zen1, ant.zen2, ant.dzen = float(line[2:8]), float(line[8:14]), float(line[14:20])
                elif lab == "VALID FROM":
                    ant.valid_from = _antex_datetime(line)
                elif lab == "VALID UNTIL":
                    ant.valid_until = _antex_datetime(line)
                elif lab == "SINEX CODE":
                    ant.sinex_code = line[:10].strip()
                elif lab == "START OF FREQUENCY":
                    cur = line[3:6]
                    rows = []
                    ant.freqs[cur] = AntexFrequency(pco_mm=np.zeros(3))
                elif lab == "NORTH / EAST / UP" and cur:
                    ant.freqs[cur].pco_mm = np.array([float(line[k:k + 10]) for k in (0, 10, 20)])
                elif lab == "END OF FREQUENCY":
                    if cur and rows:
                        ant.freqs[cur].grid_mm = np.vstack(rows)
                    cur = None
                elif lab == "START OF FREQ RMS":
                    while i < n and lines[i][60:80].strip() != "END OF FREQ RMS":
                        i += 1
                elif cur and lab == "" and line[3:8] == "NOAZI":
                    ant.freqs[cur].noazi_mm = np.array(_floats(line[8:]))
                elif cur and lab == "" and line[:8].strip():
                    rows.append(np.array(_floats(line[8:])))
                i += 1
            if ant is not None:
                self.antennas.append(ant)
            i += 1

    # ------------------------------------------------------------------ lookups
    @staticmethod
    def _current(entries: Iterable[AntexAntenna], epoch: Optional[datetime]) -> Optional[AntexAntenna]:
        """Valid entry with the latest VALID FROM (first in the file on a tie).

        A VALID UNTIL of 23:59:59.9999999 rounds to the next 00:00, so at a hand-over instant both the
        expiring and the new entry are valid: the new one (later VALID FROM) is the one in use. With no
        epoch every entry is valid and the newest one is returned.
        """
        best = None
        for a in entries:
            if a.valid_at(epoch) and (best is None or a.valid_from > best.valid_from):
                best = a
        return best

    def satellite(self, prn: str, epoch: Optional[datetime]) -> Optional[AntexAntenna]:
        """Entry of ``prn`` valid at ``epoch`` (VALID FROM <= epoch <= VALID UNTIL), or None.

        Several valid entries (hand-over instant, or ``epoch`` None): the one with the latest VALID FROM.
        """
        return self._current(self._by_prn.get(str(prn).strip(), ()), epoch)

    def satellites(self, epoch: Optional[datetime], systems: Iterable[str] = "GEC") -> Dict[str, AntexAntenna]:
        """{PRN: entry} of every satellite of ``systems`` valid at ``epoch``."""
        systems = set(systems)
        out = {}
        for prn, entries in self._by_prn.items():
            if prn[0] not in systems:
                continue
            a = self._current(entries, epoch)
            if a is not None:
                out[prn] = a
        return out

    def receiver(self, ant_type: str, radome: Optional[str] = "NONE", *, epoch: Optional[datetime] = None,
                 radome_fallback: bool = True, warn: bool = True) -> Tuple[Optional[AntexAntenna], Optional[str]]:
        """(entry, note) of a receiver antenna.

        ``note`` is None for an exact match. A radome without its own calibration takes the radome NONE
        values (IGS convention) with a warning; an antenna type without any calibration gives
        ``(None, note)``.
        """
        ant_type = (ant_type or "").strip()
        radome = (radome or "").strip() or "NONE"
        cands = [a for a in self.antennas if not a.is_satellite and a.antenna == ant_type and a.valid_at(epoch)]
        cands.sort(key=lambda a: bool(a.serial.strip()))         # type means before individual calibrations
        exact = [a for a in cands if a.radome == radome]
        if exact:
            return exact[0], None
        if radome_fallback and radome != "NONE":
            none = [a for a in cands if a.radome == "NONE"]
            if none:
                note = (f"receiver antenna {ant_type} with radome {radome} has no calibration in {self.name}: "
                        f"the radome NONE values are used (IGS convention)")
                if warn:
                    warnings.warn(note, UserWarning, stacklevel=2)
                return none[0], note
        return None, f"receiver antenna {ant_type} {radome} not in {self.name}"

    # ------------------------------------------------------------------ legacy-compatible tables
    def legacy_sat_table(self, epoch: Optional[datetime], systems: Iterable[str] = "GEC") -> dict:
        """{(PRN, freq): [x, y, z] mm}, the shape of ``io.GNSSDataProcessor2.read_pco_antex``."""
        out = {}
        for prn, a in self.satellites(epoch, systems).items():
            for f, fr in a.freqs.items():
                out[(prn, f)] = [float(v) for v in fr.pco_mm]
        return out

    def legacy_rec_table(self, ant_type: str, radome: str) -> dict:
        """{(antenna, radome, freq): [n, e, u] mm}, the shape of ``io.GNSSDataProcessor2.receiver_pco``."""
        a, _ = self.receiver(ant_type, radome, radome_fallback=False, warn=False)
        if a is None:
            return {}
        return {(ant_type.strip(), (radome or "NONE").strip() or "NONE", f): [float(v) for v in fr.pco_mm]
                for f, fr in a.freqs.items()}


_ANTEX_CACHE: Dict[tuple, Antex] = {}
_ANTEX_CACHE_MAX = 4


def load_antex(path) -> Antex:
    """Parsed ANTEX, cached by (resolved path, mtime, size): about 1 s for 60 MB, once per process."""
    p = Path(path).expanduser().resolve()
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    hit = _ANTEX_CACHE.get(key)
    if hit is None:
        hit = Antex(p)
        if len(_ANTEX_CACHE) >= _ANTEX_CACHE_MAX:
            _ANTEX_CACHE.pop(next(iter(_ANTEX_CACHE)))
        _ANTEX_CACHE[key] = hit
    return hit


# --------------------------------------------------------------------------- frequency choice
def receiver_frequency(antenna: AntexAntenna, freq: str, *, warn: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """(ANTEX frequency used for ``freq``, note) of a receiver entry.

    Order: the code itself; else the calibrated frequency nearest in Hz (so the same frequency of another
    system first, e.g. C07 -> E07 at 1207.14 MHz, C06 -> E06 before G02), ties broken by the same system
    (C02 -> C01 before G01), then G, E, C, J. A substitution is warned (RTKLIB/gLAB practice; the IGS hint
    to drop the observations is not followed). ``(None, note)`` when nothing usable is calibrated.
    """
    if antenna is None:
        return None, "no receiver antenna entry"
    if freq in antenna.freqs:
        return freq, None
    target = ANTEX_FREQUENCY_HZ.get(freq)
    if target is None:
        return None, f"{antenna.antenna} {antenna.radome}: unknown ANTEX frequency {freq}"
    cands = []
    for code in antenna.freqs:
        f_hz = ANTEX_FREQUENCY_HZ.get(code)
        if f_hz is None or code[0] not in _SUBSTITUTE_SYSTEMS:
            continue
        order = 0 if code[0] == freq[0] else 1 + _SUBSTITUTE_SYSTEMS.index(code[0])
        cands.append((round(abs(f_hz - target) / 1e3), order, code))     # kHz resolution for the ties
    if not cands:
        return None, f"receiver antenna {antenna.antenna} {antenna.radome}: no calibration usable for {freq}"
    cands.sort()
    used = cands[0][2]
    note = (f"receiver antenna {antenna.antenna} {antenna.radome} has no {freq} calibration: the {used} values "
            f"are used ({abs(ANTEX_FREQUENCY_HZ[used] - target) / 1e6:.2f} MHz away)")
    if warn:
        warnings.warn(note, UserWarning, stacklevel=2)
    return used, note


def satellite_frequency(antenna: AntexAntenna, freq: str, *, warn: bool = True, notes=None) -> Optional[str]:
    """ANTEX frequency of a satellite entry used for ``freq``.

    The code itself when calibrated; GPS G05 missing: G01 when the entry is frequency independent (igs20:
    L1 = L2 = ionosphere-free estimate), warned once per satellite and run of ``notes`` (``gnx_py._notes``; a
    session's configuration), or once per antenna entry when ``notes`` is None (before,
    once per process); otherwise None. The sessions call it with ``warn=False`` and report the satellites in
    one warning of their own.
    """
    if antenna is None:
        return None
    if freq in antenna.freqs:
        return freq
    if freq == "G05" and "G01" in antenna.freqs and antenna.frequency_independent():
        if warn:
            key = ("sat_g05_fallback", antenna.prn, antenna.svn.strip(), freq)
            _notes.warn_once(antenna if notes is None else notes, key,
                      f"satellite {antenna.prn} ({antenna.svn.strip()}) has no G05 calibration: the "
                      f"frequency-independent G01 values are used", UserWarning, stacklevel=2)
        return "G01"
    return None


# --------------------------------------------------------------------------- broadcast reference
#: Galileo modes whose broadcast clock comes from I/NAV (E1,E5b); the other modes use F/NAV (E1,E5a);
#: same choice as spp.py and ppp/config.py (Galileo OS SIS ICD 5.1.1/5.1.5)
GAL_INAV_MODES = ("E1", "E5b", "E1E5b", "E5aE5b")


def broadcast_apc_reference(system: str, mode: str) -> Optional[Tuple[Tuple[str, str], Tuple[float, float]]]:
    """ANTEX frequencies and IF coefficients of the antenna phase centre broadcast orbits refer to.

    The APC of the message's clock-reference signal:

    - GPS: IF(L1, L2) (IS-GPS-200 20.3.3.4.3).
    - Galileo: the IF pair of the message's clock, F/NAV IF(E1, E5a) for E5a/E1E5a, I/NAV IF(E1, E5b)
      otherwise.
    - BeiDou: B3I for every mode, D1/D2 and B-CNAV1/2/3: the broadcast clock is the B3I clock and every
      message group delay maps it to another signal (dt_B1I = dt - TGD1, dt_B2I = dt - TGD2, B3I none,
      BDS-SIS-ICD-B1I 3.0 5.2.4.10; dt_B1Cp = dt - TGD_B1Cp, BDS-SIS-ICD-B1C 1.0 7.6; dt_B2ap = dt - TGD_B2ap,
      B2a 1.0 7.6; dt_B2bI = dt - TGD_B2bI, B2b 1.0 7.5; ``biases`` table "BeiDou D1/D2, B-CNAV: B3I"), so the
      broadcast orbit and clock refer to the B3I APC. Returned as ``(("C06", "C06"), (1.0, 0.0))``: reference
      PCO = PCO(C06), the ANTEX frequency of B3I.

    Returns ``((f1, f2), (a, b))`` with reference PCO = a*PCO(f1) - b*PCO(f2), None for another system.

    Use in the preprocessing: with broadcast orbits no satellite antenna term is applied for any system (the
    broadcast group delays map the reference to the signal, RTKLIB/gLAB); a precise orbit compared with a
    broadcast one (SIS) is moved to this APC for GPS and BeiDou. Measured (2024-035/131/132, COD and
    GRG, IGS20_2388): BeiDou-3 mean dR within -0.085..+0.046 m at B3I against +0.69..+1.12 m at IF(B1I,B3I) and
    -1.04..-1.68 m at the CoM. For Galileo the measured broadcast orbit APC is not the IF of the ANTEX
    frequencies (FOC mean dR +0.17..+0.25 m at the IF of the message pair, -0.07..+0.10 m at E1/E5a/E5b), so the
    Galileo pair here is the clock reference only and SIS keeps Galileo at the APC of the mode.
    """
    from ..core.gnss import ionosphere_free_coefficients

    if system == "G":
        return ("G01", "G02"), ionosphere_free_coefficients("L1", "L2")
    if system == "E":
        if mode in GAL_INAV_MODES:
            return ("E01", "E07"), ionosphere_free_coefficients("E1", "E5b")
        return ("E01", "E05"), ionosphere_free_coefficients("E1", "E5a")
    if system == "C":
        return ("C06", "C06"), (1.0, 0.0)
    return None


# --------------------------------------------------------------------------- version check
_SP3_PCV = re.compile(r"PCV:\s*(\S+)")


def sp3_pcv_label(sp3_path) -> Optional[str]:
    """Antenna model label of an SP3 header comment ('/* PCV:IGS20_2290 ...'), or None."""
    # any compression of gnx_py.compression (bug-fix stage 10: a .Z file silently gave no label)
    for k, line in enumerate(head_lines(Path(sp3_path), "SP3 file", 20_000)):
        if k > 80:
            break
        if line.startswith("*  ") or line.startswith("* "):
            break                                  # first epoch record: header ended
        m = _SP3_PCV.search(line)
        if line.startswith("/*") and m:
            return m.group(1)
    return None


_VERSION_RE_ATX = re.compile(r"([A-Za-z]+\d+)_(\d{4})")
_VERSION_RE_SP3 = re.compile(r"([A-Za-z]+\d+)(?:_(\d{4}))?")


def check_version(antex: Antex, label: Optional[str], *, warn: bool = True) -> Optional[str]:
    """Compare the ANTEX with the antenna model of the orbit/clock product (SP3 'PCV:' label).

    Warns (``AntexVersionWarning``) when the frames differ (IGS14 vs IGS20) or the GPS weeks differ,
    with stronger wording when the ANTEX is older than the product. A label without a week (COD 2024
    'IGS20') cannot be checked: no warning. Returns the message or None.
    """
    if not label or antex is None:
        return None
    m1 = _VERSION_RE_ATX.match(antex.sinex_code or "")
    m2 = _VERSION_RE_SP3.match(label)
    if not m1 or not m2:
        return None
    msg = None
    if m1.group(1).upper() != m2.group(1).upper():
        msg = (f"ANTEX {antex.sinex_code} ({antex.name}) and the orbit product (PCV:{label}) use different "
               f"antenna frames ({m1.group(1)} vs {m2.group(1)})")
    elif m2.group(2):
        w_atx, w_prod = int(m1.group(2)), int(m2.group(2))
        if w_atx < w_prod:
            msg = (f"ANTEX {antex.sinex_code} ({antex.name}) is OLDER than the antenna model of the orbit "
                   f"product (PCV:{label}): satellites launched or recalibrated since week {w_atx} may be "
                   f"missing or carry other offsets; use an ANTEX of week {w_prod} or later")
        elif w_atx > w_prod:
            msg = (f"ANTEX {antex.sinex_code} ({antex.name}) differs from the antenna model of the orbit "
                   f"product (PCV:{label}): calibrations changed after week {w_prod} are not those the "
                   f"product was computed with")
    if msg and warn:
        warnings.warn(msg, AntexVersionWarning, stacklevel=2)
    return msg


# --------------------------------------------------------------------------- receiver from header
def receiver_antenna_from_header(header: Optional[Mapping]) -> Tuple[Optional[str], Optional[str]]:
    """(antenna type, radome) from the RINEX header line 'ANT # / TYPE', by column.

    Columns 21-36 hold the antenna type and 37-40 the radome (RINEX 2/3/4); the header dict is the one
    ``georinex.rinexheader`` returns (line[:60] kept raw). A blank radome is 'NONE'. ``(None, None)``
    when the line is missing or blank.
    """
    if not header:
        return None, None
    raw = header.get("ANT # / TYPE")
    if raw is None:
        return None, None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    raw = str(raw).ljust(40)
    ant = raw[20:36].strip()
    radome = raw[36:40].strip() or "NONE"
    if not ant:
        return None, None
    return ant, radome


# --------------------------------------------------------------------------- session model
@dataclass
class AntennaModel:
    """Antenna entries of one processing session (built once by :func:`build_antenna_model`).

    ``satellites``: entries valid at the first observation epoch, by PRN; ``receiver``: the receiver entry
    (None when the antenna is not calibrated or unknown; PCO and PCV are then 0 with a warning);
    ``pcv``: False for a model made from the legacy PCO dicts (PCO only).
    """

    atx: Optional[Antex]
    epoch0: Optional[datetime]
    systems: Tuple[str, ...] = ("G", "E", "C")
    satellites: Dict[str, AntexAntenna] = field(default_factory=dict)
    receiver: Optional[AntexAntenna] = None
    receiver_type: Optional[str] = None
    receiver_radome: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    missing_policy: str = "exclude"
    sp3_label: Optional[str] = None
    version_note: Optional[str] = None
    pcv: bool = True
    source: str = ""
    #: PRN -> :class:`PrnChange`: the satellite of the first epoch has no entry from an epoch before the session end
    #: on (since v1.0.0)
    prn_changes: Dict[str, "PrnChange"] = field(default_factory=dict)

    @property
    def atx_name(self) -> str:
        return self.atx.name if self.atx is not None else (self.source or "legacy PCO tables")

    def satellite(self, prn: str) -> Optional[AntexAntenna]:
        """Entry of ``prn`` valid at the first epoch (looked up in the ANTEX for systems not prebuilt)."""
        prn = str(prn).strip()
        hit = self.satellites.get(prn)
        if hit is None and self.atx is not None and prn[:1] not in self.systems:
            hit = self.atx.satellite(prn, self.epoch0)
            if hit is not None:
                self.satellites[prn] = hit
        return hit

    def drop_after_prn_change(self, frame):
        """``frame`` without the rows of a PRN in :attr:`prn_changes` after the end of its satellite's validity
        (time > ``change``; also after a same-SVN gap, the session does not restart the PRN); the frame needs index levels ``sv`` (PRN, first three characters used) and ``time``. The
        same object when nothing is dropped."""
        if not self.prn_changes or frame is None or len(frame) == 0:
            return frame
        names = list(getattr(frame.index, "names", None) or [])
        if "sv" not in names or "time" not in names:
            return frame
        import pandas as pd

        sv = pd.Index(frame.index.get_level_values("sv")).astype(str).str[:3]
        times = pd.DatetimeIndex(frame.index.get_level_values("time"))
        drop = np.zeros(len(frame), dtype=bool)
        for prn, change in self.prn_changes.items():
            drop |= (sv == prn) & (times > pd.Timestamp(change.change))
        return frame[~drop].copy() if drop.any() else frame

    def receiver_label(self) -> str:
        if self.receiver_type:
            return f"{self.receiver_type} {self.receiver_radome or 'NONE'}"
        return "unknown receiver antenna"

    # ------------------------------------------------------------------ legacy tables
    def legacy_sat_table(self, system: Optional[str] = None) -> dict:
        """{(PRN, freq): [x, y, z] mm} of the session's satellites (optionally one system)."""
        return {(prn, f): [float(v) for v in fr.pco_mm]
                for prn, a in self.satellites.items() if system is None or prn[0] == system
                for f, fr in a.freqs.items()}

    def legacy_rec_table(self) -> dict:
        """{(antenna, radome, freq): [n, e, u] mm} of the session's receiver ({} when uncalibrated)."""
        if self.receiver is None:
            return {}
        ant = self.receiver_type or self.receiver.antenna
        rad = self.receiver_radome or self.receiver.radome
        return {(ant, rad, f): [float(v) for v in fr.pco_mm] for f, fr in self.receiver.freqs.items()}

    # ------------------------------------------------------------------ legacy input
    @classmethod
    def from_legacy(cls, sat_pco: Optional[Mapping] = None, rec_pco: Optional[Mapping] = None, *,
                    epoch0: Optional[datetime] = None, missing_policy: str = "exclude") -> "AntennaModel":
        """Model from the PCO-only dicts of ``io.GNSSDataProcessor2`` ({(PRN, freq): xyz mm} and
        {(antenna, radome, freq): neu mm}); PCV is not available (``pcv=False``)."""
        sats: Dict[str, AntexAntenna] = {}
        for key, val in dict(sat_pco or {}).items():
            try:
                prn, freq = key
            except (TypeError, ValueError):
                continue
            if val is None or len(val) != 3:
                continue
            a = sats.setdefault(str(prn), AntexAntenna(type="LEGACY".ljust(20), serial=str(prn), svn="LEGACY"))
            a.freqs[str(freq)] = AntexFrequency(pco_mm=np.asarray(val, dtype=float))
        rec = None
        rtype = rrad = None
        for key, val in dict(rec_pco or {}).items():
            try:
                ant, rad, freq = key
            except (TypeError, ValueError):
                continue
            if val is None or len(val) != 3:
                continue
            if rec is None:
                rtype, rrad = str(ant).strip(), (str(rad).strip() or "NONE")
                rec = AntexAntenna(type=rtype.ljust(16)[:16] + rrad.ljust(4)[:4], serial="")
            rec.freqs[str(freq)] = AntexFrequency(pco_mm=np.asarray(val, dtype=float))
        return cls(atx=None, epoch0=epoch0, systems=tuple(sorted({p[0] for p in sats})), satellites=sats,
                   receiver=rec, receiver_type=rtype, receiver_radome=rrad, missing_policy=missing_policy,
                   pcv=False, source="legacy PCO tables")


@dataclass(frozen=True)
class PrnChange:
    """A PRN whose satellite (SVN) of the first epoch has no entry from ``change`` on within the session: ``svn`` of
    that satellite, ``change`` the VALID UNTIL of its last entry contiguous with the one valid at the first epoch
    (consecutive entries of the same SVN, e.g. igs20 C35 C215 split at 2022-10-10 and 2023-01-01, are one
    validity), ``next_svn`` and ``next_from`` of the PRN's next entry ('' / None when none starts within the
    session; ``next_svn == svn``: the same satellite again after a gap without an entry, e.g. igs20 C31 C101
    2018-04-24 .. 2018-07-10)."""

    prn: str
    svn: str
    change: datetime
    next_svn: str = ""
    next_from: Optional[datetime] = None


#: largest gap between the VALID UNTIL of an entry and the VALID FROM of the next entry of the same SVN that still
#: counts as one validity (igs20: 0 s, or 1 s where VALID UNTIL is hh:mm:59)
PRN_CHANGE_CONTIGUITY = timedelta(seconds=60)


def _prn_changes(atx: "Antex", sats: Mapping[str, AntexAntenna], t_end: Optional[datetime]) -> Dict[str, PrnChange]:
    """PRNs of ``sats`` (entries valid at the first epoch) whose satellite has no entry from an epoch before
    ``t_end`` on: the next entry of the PRN belongs to another SVN (a reassigned PRN), or none follows without a
    gap. A split of one SVN's validity into contiguous entries (a new calibration of the same satellite) is no
    change."""
    out: Dict[str, PrnChange] = {}
    if t_end is None:
        return out
    for prn, entry in sorted(sats.items()):
        entries = list(atx._by_prn.get(prn, ()))
        cur = entry
        while cur.valid_until < t_end:
            later = [a for a in entries if a is not cur and a.valid_from > cur.valid_from]
            nxt = min(later, key=lambda a: a.valid_from) if later else None
            if (nxt is not None and nxt.svn.strip() == cur.svn.strip()
                    and nxt.valid_from <= cur.valid_until + PRN_CHANGE_CONTIGUITY):
                cur = nxt                                     # the same satellite: its validity continues
                continue
            within = nxt is not None and nxt.valid_from <= t_end
            out[prn] = PrnChange(prn=prn, svn=entry.svn.strip(), change=cur.valid_until,
                                 next_svn=nxt.svn.strip() if within else "",
                                 next_from=nxt.valid_from if within else None)
            break
    return out


def prn_change_message(model: "AntennaModel") -> Optional[str]:
    """The warning text of the PRN changes of ``model`` (None without any)."""
    if not model.prn_changes:
        return None
    parts = []
    for ch in model.prn_changes.values():
        if not ch.next_svn:
            after = "no later entry within the session"
        elif ch.next_svn == ch.svn:
            after = f"no entry until {ch.next_from:%Y-%m-%d %H:%M:%S}, then {ch.next_svn} again"
        else:
            after = f"then {ch.next_svn}"
        parts.append(f"{ch.prn} (SVN {ch.svn} valid until {ch.change:%Y-%m-%d %H:%M:%S}, {after})")
    return (f"ANTEX {model.atx_name}: the satellite of PRN {', '.join(parts)} changes or loses its entry during "
            "the session: their observations (orbit rows) after that epoch are excluded (a reassigned PRN belongs "
            "to another satellite with its own antenna, clock and group delays)")


def _first_epoch(epoch0) -> Optional[datetime]:
    if epoch0 is None:
        return None
    try:
        import pandas as pd

        return pd.Timestamp(epoch0).to_pydatetime().replace(tzinfo=None)
    except Exception:  # noqa: BLE001 - plain datetime
        return epoch0


def build_antenna_model(atx_path, obs_header: Optional[Mapping], epoch0, systems: Iterable[str] = "GEC", *,
                        missing_policy: str = "exclude", sp3_path=None, epoch_end=None,
                        notes_owner=None) -> AntennaModel:
    """Antenna model of one session.

    ``obs_header``: RINEX observation header dict (``GNSSDataProcessor2._get_obs_header()`` or
    ``georinex.rinexheader``); None for a session without a receiver (SIS). ``epoch0``: first observation
    epoch (satellite entries valid then). ``epoch_end``: last epoch of the session: a PRN whose satellite (SVN) valid at ``epoch0`` has no entry from an epoch before it on -- the PRN
    reassigned to another satellite during the session, as BeiDou C06-C08 and C11-C14 were, or no entry after a
    VALID UNTIL -- is listed in ``prn_changes`` with one UserWarning naming the PRN, the SVNs and the change
    epoch; contiguous entries of the same SVN (a split of its calibration) are no change. Every session that
    builds the model drops the PRN's observations (orbit rows) after that epoch
    (:meth:`AntennaModel.drop_after_prn_change`: ``DDPreprocessing.run`` for PPP/SPP/TEC, ``SISSession.run``).
    None: not checked (as before). ``notes_owner``: the warning goes through ``gnx_py._notes.warn_once`` of this
    owner (the session's configuration: once per run however often the model is built); None: every build
    warns. ``sp3_path``
    (precise orbits only; a path or a list, the first file is read): its 'PCV:' label is compared with the
    ANTEX (:func:`check_version`, one warning).
    Receiver: radome without calibration -> NONE with a warning; antenna without calibration -> None
    (PCO/PCV 0, warned by the preprocessing).
    """
    if missing_policy not in ("exclude", "raise"):
        raise ValueError(f"missing_policy must be 'exclude' or 'raise', got {missing_policy!r}")
    atx = load_antex(atx_path)
    t0 = _first_epoch(epoch0)
    systems = tuple(dict.fromkeys(s for s in systems))
    sats = atx.satellites(t0, systems)
    changes = _prn_changes(atx, sats, _first_epoch(epoch_end)) if t0 is not None else {}
    notes: List[str] = []
    rtype, rrad = receiver_antenna_from_header(obs_header)
    rec = None
    if obs_header is not None:
        if rtype is None:
            notes.append("the RINEX header has no receiver antenna type (ANT # / TYPE)")
        else:
            rec, note = atx.receiver(rtype, rrad, epoch=t0)
            if note:
                notes.append(note)
    label = None
    vnote = None
    if sp3_path is not None:
        first = sp3_path[0] if isinstance(sp3_path, (list, tuple)) and sp3_path else sp3_path
        if first:
            try:
                label = sp3_pcv_label(first)
            except OSError:
                label = None
            vnote = check_version(atx, label)
            if vnote:
                notes.append(vnote)
    model = AntennaModel(atx=atx, epoch0=t0, systems=systems, satellites=sats, receiver=rec, receiver_type=rtype,
                         receiver_radome=rrad, notes=notes, missing_policy=missing_policy, sp3_label=label,
                         version_note=vnote, pcv=True, source=str(atx.path), prn_changes=changes)
    message = prn_change_message(model)
    if message:
        notes.append(message)
        if notes_owner is None:
            warnings.warn(message, UserWarning, stacklevel=2)
        else:
            _notes.warn_once(notes_owner, ("antex_prn_change", message), message, UserWarning, stacklevel=2)
    return model


# --------------------------------------------------------------------------- coverage
def missing_satellite_antennas(model, prns: Iterable[str], freqs: Sequence[str]) -> List[str]:
    """PRNs without a valid entry for every frequency in ``freqs`` (G05 -> G01 fallback allowed).

    ``model``: an :class:`AntennaModel`, or an :class:`Antex` together with the epoch as ``(antex, epoch)``.
    Used by the preprocessing (missing-satellite rule) and by coverage checks of test datasets.
    """
    if isinstance(model, tuple):
        atx, epoch = model
        lookup = lambda p: atx.satellite(p, _first_epoch(epoch))  # noqa: E731
    else:
        lookup = model.satellite
    out = []
    for prn in sorted({str(p) for p in prns}):
        e = lookup(prn)
        if e is None or any(satellite_frequency(e, f, warn=False) is None for f in freqs):
            out.append(prn)
    return out


def legacy_sat_table(model: AntennaModel, system: Optional[str] = None) -> dict:
    """Module-level form of :meth:`AntennaModel.legacy_sat_table`."""
    return model.legacy_sat_table(system)


def legacy_rec_table(model: AntennaModel) -> dict:
    """Module-level form of :meth:`AntennaModel.legacy_rec_table`."""
    return model.legacy_rec_table()
