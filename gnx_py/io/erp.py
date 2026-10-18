"""IGS ERP (version 2) Earth rotation parameters for the pole tide.

Reads the IGS ERP format 2 (``version 2`` header, then one row per epoch: MJD, Xpole, Ypole [1e-6 arcsec],
UT1-UTC, LOD [1e-7 s], sigmas, Nr Nf Nt, Xrt, Yrt [1e-6 arcsec/day], ...), as RTKLIB ``readerp`` reads it, and
interpolates linearly in UTC MJD, extrapolating outside the file with the rates as RTKLIB ``geterp`` does.

RTKLIB 2.4.2 ``tidedisp`` passes a UTC epoch to ``geterp``, which converts GPST -> UTC once more (18 s early); GNX
interpolates at the UTC epoch (difference below 1e-3 mas of polar motion).
"""
from __future__ import annotations

import functools
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np

from .compression import open_text

PathLike = Union[str, os.PathLike]

_MJD0 = np.datetime64("1858-11-17T00:00:00", "ns")


@dataclass(frozen=True)
class ERPData:
    """Earth rotation parameters per MJD (UTC): xp, yp [arcsec], ut1_utc [s], lod [s/day], xpr, ypr [arcsec/day]."""

    path: str
    mjd: np.ndarray
    xp: np.ndarray
    yp: np.ndarray
    ut1_utc: np.ndarray
    lod: np.ndarray
    xpr: np.ndarray
    ypr: np.ndarray

    def at(self, times_utc, *, warn: bool = True) -> Dict[str, np.ndarray]:
        """Values at UTC epochs: linear interpolation, rate extrapolation outside the file (RTKLIB ``geterp``).

        A ``UserWarning`` names epochs more than 1 day outside the file.
        """
        t = np.atleast_1d(np.asarray(times_utc, dtype="datetime64[ns]"))
        mjd = (t - _MJD0).astype("int64") * 1e-9 / 86400.0
        first, last = self.mjd[0], self.mjd[-1]
        out = {"xp": np.empty(len(t)), "yp": np.empty(len(t)), "ut1_utc": np.empty(len(t)), "lod": np.empty(len(t))}
        lo, hi = mjd <= first, mjd >= last
        mid = ~(lo | hi)
        for mask, k in ((lo, 0), (hi, len(self.mjd) - 1)):
            if mask.any():
                day = mjd[mask] - self.mjd[k]
                out["xp"][mask] = self.xp[k] + self.xpr[k] * day
                out["yp"][mask] = self.yp[k] + self.ypr[k] * day
                out["ut1_utc"][mask] = self.ut1_utc[k] - self.lod[k] * day
                out["lod"][mask] = self.lod[k]
        if mid.any():
            m = mjd[mid]
            j = np.searchsorted(self.mjd, m, side="right") - 1
            j = np.clip(j, 0, len(self.mjd) - 2)
            span = self.mjd[j + 1] - self.mjd[j]
            a = np.where(span > 0, (m - self.mjd[j]) / np.where(span > 0, span, 1.0), 0.5)
            for key in ("xp", "yp", "ut1_utc", "lod"):
                v = getattr(self, key)
                out[key][mid] = (1.0 - a) * v[j] + a * v[j + 1]
        outside = (mjd < first - 1.0) | (mjd > last + 1.0)
        if warn and outside.any():
            warnings.warn(f"erp_path {self.path}: {int(outside.sum())} epochs more than 1 day outside the ERP file "
                          f"(MJD {first:g}-{last:g}); the pole tide extrapolates polar motion with the rates",
                          UserWarning, stacklevel=3)
        return out


def _open_text(path: Path):
    # gzip, Unix compress (.Z), bzip2 and zip by content, in memory (bug-fix stage 10: .Z gave a format error)
    return open_text(path, what="ERP file", encoding="latin-1")


_FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def parse_erp(text: str, path: str = "<string>") -> ERPData:
    """Parse IGS ERP version 2 ``text`` (see :func:`read_erp`)."""
    lines = text.splitlines()
    head = next((ln for ln in lines if ln.strip()), "")
    if not re.match(r"^\s*version\s+2\b", head, re.I):
        raise ValueError(f"erp_path {path}: not an IGS ERP version 2 file (first line {head[:40]!r}, "
                         f"expected 'version 2')")
    rows = []
    for k, ln in enumerate(lines, start=1):
        tok = ln.split()
        if len(tok) < 5 or not all(_FLOAT.match(x) for x in tok[:5]):
            continue
        try:
            v = [float(x) for x in tok]
        except ValueError:
            raise ValueError(f"erp_path {path}: line {k}: non-numeric value in {ln.strip()[:60]!r}") from None
        if not 15000.0 < v[0] < 100000.0:
            raise ValueError(f"erp_path {path}: line {k}: MJD {v[0]:g} out of range")
        rows.append((v[0], v[1], v[2], v[3], v[4], v[12] if len(v) > 13 else 0.0, v[13] if len(v) > 13 else 0.0))
    if not rows:
        raise ValueError(f"erp_path {path}: no ERP data rows (MJD Xpole Ypole UT1-UTC LOD ...)")
    arr = np.array(sorted(rows, key=lambda r: r[0]), dtype=float)
    return ERPData(path=path, mjd=arr[:, 0], xp=arr[:, 1] * 1e-6, yp=arr[:, 2] * 1e-6, ut1_utc=arr[:, 3] * 1e-7,
                   lod=arr[:, 4] * 1e-7, xpr=arr[:, 5] * 1e-6, ypr=arr[:, 6] * 1e-6)


@functools.lru_cache(maxsize=8)
def _read_erp_cached(path: str, mtime_ns: int, size: int) -> ERPData:
    with _open_text(Path(path)) as fh:
        return parse_erp(fh.read(), path)


def read_erp(path: PathLike) -> ERPData:
    """Read an IGS ERP version 2 file (plain, ``.gz``, ``.Z``, bzip2 or zip); cached per (path, modification
    time, size).

    ``FileNotFoundError`` for a missing file, ``ValueError`` naming the file for a malformed one.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"erp_path: file not found: {path}")
    st = p.stat()
    return _read_erp_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)
