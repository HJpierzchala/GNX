"""ORBEX attitude reader: satellite attitude quaternions of an analysis centre.

Format: ORBEX 0.09 (Loyer, Montenbruck, Hilla 2019) as adopted by the IGS (Loyer, Banville, Perosanz, Mercier,
proposal of 30 Apr 2019): header ``+FILE/DESCRIPTION`` (TIME_SYSTEM, EPOCH_INTERVAL, FRAME_TYPE, ...), data
block ``+EPHEMERIS/DATA`` with epoch lines ``## yyyy mm dd hh mm ss.sss n`` and records
``ATT <sv> 4 q0 q1 q2 q3`` (scalar first). Plain or compressed files (gzip, .Z, bzip2, zip; gnx_py.compression).

Convention (IGS proposal 2019; the GRG and COD headers "transformation from terrestrial frame coordinates to
satellite body frame": (0, x_body) = q (0, X_ecef) q*): x_body = M(q) X_ecef with M the rotation matrix of q, so
the IGS body axes in ECEF are the rows of M (Ginan orbex.cpp/attitude.cpp, Apache-2.0, read for the convention).
The body frame is the IGS frame of the ANTEX PCO (Montenbruck et al. 2015, ASR 56:1015).

Checks:

* TIME_SYSTEM must be GPS (missing: GPS assumed, one UserWarning) and FRAME_TYPE ECEF (ValueError naming
  ``attitude_path`` otherwise);
* records with |‖q‖ - 1| > 1e-6 are dropped (one UserWarning with the count); at a day boundary the duplicate
  24:00/00:00 record of the later file is dropped (the first is kept);
* against the orbit of the processing (:meth:`OrbexAttitude.axes`): the body z axis lies within 1 deg of the
  nadir direction for at least 99 % of the rows; when M fails and its transpose passes, the transpose is used
  with one UserWarning naming the file; when neither passes for half of the rows, ValueError (scalar-last
  quaternions?); single rows more than 1 deg off nadir are not covered;
* the body x axis points to the side of the nominal x axis outside turns (``gnx_py.attitude``), per satellite: a
  frame turned by 180 deg about z (the JPL GPS IIR manufacturer axes, Montenbruck et al. 2015 Table 3) is turned
  back with one UserWarning naming the satellites; a satellite whose x axis is on neither side is not used (the
  attitude law instead, one UserWarning per call naming it).

Interpolation: spherical linear interpolation (slerp) between neighbouring records after sign alignment; a time
is covered only when its records are at most 2 x EPOCH_INTERVAL apart and rotated by at most 30 deg.
"""
from __future__ import annotations

import itertools
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .compression import head_lines, read_bytes

__all__ = [
    "OrbexAttitude",
    "read_orbex",
    "read_orbex_header",
    "analysis_centre",
    "product_id",
    "check_product",
    "quaternion_matrix",
    "NORM_TOLERANCE",
    "MAX_GAP_INTERVALS",
    "MAX_ROTATION_DEG",
    "NADIR_TOLERANCE_DEG",
]

GPS_EPOCH = datetime(1980, 1, 6)
#: records whose quaternion norm differs from 1 by more are dropped
NORM_TOLERANCE = 1e-6
#: a time is covered when its records are at most this many EPOCH_INTERVAL apart ...
MAX_GAP_INTERVALS = 2.0
#: ... and rotated by at most this angle [deg]
MAX_ROTATION_DEG = 30.0
#: the body z axis must lie within this angle of the nadir direction [deg] ...
NADIR_TOLERANCE_DEG = 1.0
#: ... for at least this fraction of the rows
NADIR_FRACTION = 0.99
#: x axis toward the nominal side outside turns for at least this fraction of the rows of a block
SUN_SIDE_FRACTION = 0.95

_LONG_NAME = re.compile(r"^([A-Z0-9]{3})(\d)([A-Z]{3})([A-Z]{3})_\d{11}_", re.IGNORECASE)


def analysis_centre(path) -> str:
    """Analysis centre of a product file: the first three characters of its name, upper case (IGS long names
    'GRG0MGXFIN_...', short names 'grg23000.obx')."""
    return os.path.basename(os.fspath(path))[:3].upper()


def product_id(path) -> str:
    """Product of a long file name, e.g. 'GRG0MGXFIN' (centre, version, campaign, type); else the centre."""
    name = os.path.basename(os.fspath(path))
    m = _LONG_NAME.match(name)
    return (m.group(1) + m.group(2) + m.group(3) + m.group(4)).upper() if m else analysis_centre(name)


def _open_lines(path: Path) -> List[str]:
    # gzip, Unix compress (.Z), bzip2 and zip undone in memory (bug-fix stage 10: a .Z file read as "not an
    # ORBEX file")
    return read_bytes(path, what="ORBEX file").decode("latin-1").splitlines()


def _gps_seconds(y, mo, d, h, mi, sec) -> float:
    days = datetime(y, mo, d).toordinal() - GPS_EPOCH.toordinal()
    return days * 86400.0 + h * 3600.0 + mi * 60.0 + sec


def _sv_id(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[1:].strip().isdigit():
        return f"{token[0].upper()}{int(token[1:]):02d}"
    return token.upper()


def read_orbex_header(path) -> Dict[str, str]:
    """``+FILE/DESCRIPTION`` keywords of an ORBEX file ({'TIME_SYSTEM': 'GPS', ...}) and 'VERSION' (only the
    header is read, at most 2000 lines; plain, gzip and bzip2 files are decompressed only that far)."""
    return _description(itertools.islice(iter(head_lines(Path(path), "ORBEX file", 2000 * 128)), 2000))[0]


def quaternion_matrix(q) -> np.ndarray:
    """(n, 3, 3) rotation matrices M of quaternions (n, 4), scalar first: x_body = M X_ecef for ORBEX (rows of M =
    body axes in ECEF)."""
    q = np.asarray(q, dtype=float).reshape(-1, 4)
    w, a, b, c = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (b * b + c * c), 2 * (a * b - w * c), 2 * (a * c + w * b)], -1),
        np.stack([2 * (a * b + w * c), 1 - 2 * (a * a + c * c), 2 * (b * c - w * a)], -1),
        np.stack([2 * (a * c - w * b), 2 * (b * c + w * a), 1 - 2 * (a * a + b * b)], -1)], -2)


@dataclass
class OrbexAttitude:
    """Attitude quaternions of one analysis centre (``read_orbex``).

    ``satellites``: {sv: (t [GPS s since 1980-01-06], q (k, 4) scalar first, unit norm)}, sorted by time.
    """

    ac: str
    product: str
    interval_s: float
    satellites: Dict[str, Tuple[np.ndarray, np.ndarray]]
    paths: Tuple[str, ...] = ()
    coord_system: str = ""
    n_dropped: int = 0
    transposed: Optional[bool] = None
    _flips: Dict[str, List[int]] = field(default_factory=dict, repr=False)   # family -> [keep, flip]

    @property
    def label(self) -> str:
        return ", ".join(os.path.basename(p) for p in self.paths) or self.product

    def quaternions(self, sv, t_gpst_s) -> Tuple[np.ndarray, np.ndarray]:
        """(q (n, 4), covered (n,)): slerp of the neighbouring records (module docstring)."""
        t = np.atleast_1d(np.asarray(t_gpst_s, dtype=float))
        q = np.full((t.size, 4), np.nan)
        covered = np.zeros(t.size, dtype=bool)
        data = self.satellites.get(_sv_id(str(sv).split("_", 1)[0]))
        if data is None or t.size == 0:
            return q, covered
        tk, qk = data
        if tk.size == 0:
            return q, covered
        i = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, tk.size - 1)
        exact = np.abs(tk[i] - t) < 1e-6
        j = np.minimum(i + 1, tk.size - 1)
        inner = (t > tk[i]) & (t < tk[j]) & (j > i)
        q0, q1 = qk[i], qk[j]
        d = np.einsum("ij,ij->i", q0, q1)
        q1 = np.where(d[:, None] < 0.0, -q1, q1)
        d = np.abs(d)
        theta = np.arccos(np.clip(d, -1.0, 1.0))          # half the rotation angle between the records
        ok = inner & (tk[j] - tk[i] <= MAX_GAP_INTERVALS * self.interval_s + 1e-6) & \
            (2.0 * theta <= math.radians(MAX_ROTATION_DEG))
        u = np.where(ok, (t - tk[i]) / np.where(ok, tk[j] - tk[i], 1.0), 0.0)
        small = theta < 1e-8
        st = np.where(small, 1.0, np.sin(theta))
        w0 = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / st)
        w1 = np.where(small, u, np.sin(u * theta) / st)
        qi = w0[:, None] * q0 + w1[:, None] * q1
        qi = qi / np.linalg.norm(qi, axis=1, keepdims=True)
        q[ok] = qi[ok]
        q[exact] = qk[i][exact]
        covered = ok | exact
        return q, covered

    def axes(self, sv, t_gpst_s, r=None):
        """(x, y, z, covered): IGS body axes in ECEF (n, 3) of ``sv`` at ``t_gpst_s`` (NaN where not covered).

        ``r``: the satellite positions (n, 3) of the processing [m]; the first call with at least 10 covered rows
        decides between M and its transpose (module docstring); rows whose z axis is more than 1 deg off nadir
        are then not covered.
        """
        q, covered = self.quaternions(sv, t_gpst_s)
        n = covered.size
        x = np.full((n, 3), np.nan)
        y = np.full((n, 3), np.nan)
        z = np.full((n, 3), np.nan)
        if not covered.any():
            return x, y, z, covered
        M = quaternion_matrix(q[covered])
        transposed = bool(self.transposed)
        if r is not None:
            r = np.asarray(r, dtype=float).reshape(-1, 3)[covered]
            fin = np.isfinite(r).all(axis=1) & (np.abs(r).sum(axis=1) > 0)
            nadir = np.zeros_like(r)
            nadir[fin] = -r[fin] / np.linalg.norm(r[fin], axis=1, keepdims=True)
            lim = math.cos(math.radians(NADIR_TOLERANCE_DEG))
            if self.transposed is None and fin.sum() >= 10:
                transposed = self._decide_convention(M[fin], nadir[fin], lim, sv)
            zz = M[:, :, 2] if transposed else M[:, 2, :]
            good = fin & (np.einsum("ij,ij->i", zz, nadir) > lim)
            idx = np.flatnonzero(covered)
            covered[idx[~good]] = False
            M = M[good]
        if transposed:
            M = np.transpose(M, (0, 2, 1))
        x[covered], y[covered], z[covered] = M[:, 0, :], M[:, 1, :], M[:, 2, :]
        return x, y, z, covered

    def _decide_convention(self, M, nadir, lim, sv) -> bool:
        """M (False) or its transpose (True) by the nadir test on one satellite's rows; decided for the file when one
        convention passes for NADIR_FRACTION of the rows, ValueError when neither passes for half of them."""
        f_rows = float(np.mean(np.einsum("ij,ij->i", M[:, 2, :], nadir) > lim))
        f_cols = float(np.mean(np.einsum("ij,ij->i", M[:, :, 2], nadir) > lim))
        if f_rows >= NADIR_FRACTION:
            self.transposed = False
        elif f_cols >= NADIR_FRACTION:
            self.transposed = True
            warnings.warn(f"attitude_path {self.label}: the quaternions map the body frame to ECEF (the transpose "
                          f"of the IGS ORBEX convention); they are used transposed", UserWarning, stacklevel=4)
        elif max(f_rows, f_cols) < 0.5:
            raise ValueError(
                f"attitude_path {self.label}: the body z axis of {sv} is not within {NADIR_TOLERANCE_DEG:g} deg of "
                f"the nadir direction for either quaternion convention ({100 * f_rows:.0f} % / {100 * f_cols:.0f} % of "
                f"the rows); scalar-last quaternions? ORBEX ATT records hold q0 (scalar) q1 q2 q3")
        # neither decides (a satellite with some bad records): this call uses the better one, the next decides
        return self.transposed if self.transposed is not None else f_cols > f_rows

    def _x_axis_decision(self, family: str, dots) -> Tuple[Optional[bool], float]:
        """(decision, fraction) for one satellite of ``family`` from x_orbex . x_reference on its rows outside
        turns: False (keep: x on the Sun side for SUN_SIDE_FRACTION of the rows), True (turned by 180 deg about z:
        flip x and y), None (neither: its ORBEX attitude is not used); fraction: rows with x on the Sun side. Each
        satellite with at least 10 such rows decides for itself; one with fewer takes the decision most satellites
        of its family took so far in this file (keep without any). No warning: the caller names the satellites."""
        tally = self._flips.setdefault(family, [0, 0])
        if dots is None or np.size(dots) < 10:
            return tally[1] > tally[0], float("nan")
        pos = float(np.mean(np.asarray(dots) > 0.0))
        if pos >= SUN_SIDE_FRACTION:
            tally[0] += 1
            return False, pos
        if pos <= 1.0 - SUN_SIDE_FRACTION:
            tally[1] += 1
            return True, pos
        return None, pos


def _description(lines) -> Tuple[Dict[str, str], int]:
    """(``+FILE/DESCRIPTION`` keywords and 'VERSION', index of the ``+EPHEMERIS/DATA`` line or the number of lines
    read) of the header lines of an ORBEX file."""
    out: Dict[str, str] = {}
    inside = False
    k = -1
    for k, line in enumerate(lines):
        line = line.rstrip("\r\n")
        if line.startswith("%=ORBEX"):
            out["VERSION"] = line[7:].strip()
        elif line.startswith("+FILE/DESCRIPTION"):
            inside = True
        elif line.startswith("-FILE/DESCRIPTION"):
            inside = False
        elif line.startswith("+EPHEMERIS/DATA"):
            return out, k
        elif inside and line[:1] == " ":
            key, val = line[1:21].strip(), line[21:].strip()
            if key:
                out[key] = val
    return out, k + 1


def _parse(path: Path):
    lines = _open_lines(path)
    if not lines or not lines[0].startswith("%=ORBEX"):
        raise ValueError(f"attitude_path: {path.name} is not an ORBEX file (no '%=ORBEX' first line)")
    head, i = _description(lines)
    ts: List[float] = []
    svs: List[str] = []
    qs: List[str] = []
    t = None
    for line in lines[i + 1:]:
        if line.startswith("##"):
            p = line[2:].split()
            t = _gps_seconds(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), float(p[5]))
        elif line.startswith(" ATT") and t is not None:
            p = line.split()
            if len(p) >= 7:
                ts.append(t)
                svs.append(p[1])
                qs.append(" ".join(p[3:7]))
        elif line.startswith("-EPHEMERIS/DATA"):
            break
    q = np.array(" ".join(qs).split(), dtype=float).reshape(-1, 4) if qs else np.zeros((0, 4))
    return head, np.asarray(ts, dtype=float), np.asarray(svs, dtype=object), q


def read_orbex(paths) -> OrbexAttitude:
    """ORBEX attitude of one analysis centre from one file or several (e.g. consecutive days).

    ValueError: not an ORBEX file, TIME_SYSTEM other than GPS, FRAME_TYPE other than ECEF, files of several
    analysis centres. Several product lines of one centre: one UserWarning, all files used.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("attitude_path: no ORBEX file given")
    parts = []
    for p in paths:
        head, ts, svs, q = _parse(p)
        ts_ = head.get("TIME_SYSTEM", "").upper()
        if not ts_:
            warnings.warn(f"attitude_path {p.name}: no TIME_SYSTEM in the header; GPS time is assumed",
                          UserWarning, stacklevel=2)
        elif ts_ != "GPS":
            raise ValueError(f"attitude_path {p.name}: TIME_SYSTEM {ts_} is not supported; GNX needs ORBEX in GPS "
                             f"time (the time system of SP3 and clock products)")
        fr = head.get("FRAME_TYPE", "").upper()
        if fr != "ECEF":
            raise ValueError(f"attitude_path {p.name}: FRAME_TYPE {fr or 'missing'} is not supported; GNX needs the "
                             f"quaternions of the Earth-fixed frame (FRAME_TYPE ECEF)")
        try:
            interval = float(head.get("EPOCH_INTERVAL", "nan"))
        except ValueError:
            interval = float("nan")
        if not math.isfinite(interval) or interval <= 0:
            u = np.unique(ts)
            interval = float(np.median(np.diff(u))) if u.size > 1 else 30.0
        parts.append((p, head, ts, svs, q, interval))
    acs = sorted({analysis_centre(p) for p, *_ in parts})
    if len(acs) > 1:
        raise ValueError(f"attitude_path: ORBEX files of several analysis centres ({', '.join(acs)}); give the files "
                         f"of the centre of the orbits and clocks")
    prods = sorted({product_id(p) for p, *_ in parts})
    if len(prods) > 1:
        warnings.warn(f"attitude_path: ORBEX files of several products of {acs[0]} ({', '.join(prods)}) are combined",
                      UserWarning, stacklevel=2)
    parts.sort(key=lambda x: (x[2].min() if x[2].size else math.inf))
    t_all = np.concatenate([x[2] for x in parts]) if parts else np.zeros(0)
    sv_all = np.concatenate([x[3] for x in parts]) if parts else np.zeros(0, dtype=object)
    q_all = np.concatenate([x[4] for x in parts]) if parts else np.zeros((0, 4))
    norm = np.linalg.norm(q_all, axis=1)
    good = np.abs(norm - 1.0) <= NORM_TOLERANCE
    n_bad = int((~good).sum())
    if n_bad:
        warnings.warn(f"attitude_path: {n_bad} ATT records with |q| differing from 1 by more than {NORM_TOLERANCE:g} "
                      f"are dropped ({', '.join(p.name for p, *_ in parts)})", UserWarning, stacklevel=2)
    sats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ids = np.array([_sv_id(s) for s in sv_all.tolist()], dtype=str)
    sel = np.flatnonzero(good)
    # by satellite, then time, then file order: of two records at one epoch (24:00 of a day = 00:00 of the next)
    # the one of the earlier file is kept
    sel = sel[np.lexsort((sel, t_all[sel], ids[sel]))]
    bounds = np.flatnonzero(ids[sel][1:] != ids[sel][:-1]) + 1
    for k in np.split(sel, bounds):
        if not k.size:
            continue
        keep = np.r_[True, np.diff(t_all[k]) > 1e-6]
        k = k[keep]
        sats[str(ids[k[0]])] = (t_all[k], q_all[k] / norm[k][:, None])
    interval = max(x[5] for x in parts)
    return OrbexAttitude(ac=acs[0], product=prods[0], interval_s=interval, satellites=sats,
                         paths=tuple(str(p) for p, *_ in parts), coord_system=parts[0][1].get("COORD_SYSTEM", ""),
                         n_dropped=n_bad)


def check_product(orbex: OrbexAttitude, product, *, allow_mixed: bool = False, stacklevel: int = 2) -> bool:
    """Whether an ORBEX attitude goes with the orbits and clocks of ``product`` (an SP3/CLK path or a product id).

    Another analysis centre: False with one UserWarning naming both (the clocks hold the wind-up of their own
    centre's attitude); True with ``allow_mixed``. Same centre, another product line (e.g. GRG0OPSFIN with
    GRG0MGXFIN): True with one UserWarning. ``product`` None: True.
    """
    if product is None:
        return True
    name = os.path.basename(os.fspath(product))
    ac = analysis_centre(name)
    if ac != orbex.ac.upper():
        if allow_mixed:
            return True
        warnings.warn(f"attitude_path: the ORBEX attitude of analysis centre {orbex.ac} ({orbex.label}) is not used "
                      f"with the orbits and clocks of {ac}; give the {ac} ORBEX file, or set "
                      f"attitude_allow_mixed_products=True (PPPConfig only) to use it anyway",
                      UserWarning, stacklevel=stacklevel + 1)
        return False
    prod = product_id(name)
    if len(prod) == 10 and len(orbex.product) == 10 and prod != orbex.product:
        warnings.warn(f"attitude_path: ORBEX product {orbex.product} is used with the orbits and clocks of {prod} "
                      f"(same analysis centre, another product line)", UserWarning, stacklevel=stacklevel + 1)
    return True
