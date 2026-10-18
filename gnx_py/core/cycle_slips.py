"""Layered cycle-slip detection inputs and tests.

``CSDetector`` (``gnx_py.tools``) runs these layers before it labels the arcs; the filters only see the
arc labels, so no filter changes:

* **L0, metadata.** Data gaps (``CSDetector.split_obs_dt``), the RINEX loss-of-lock indicators (LLI) of the
  processed phases and of the third carrier, read from the RINEX 2, 3 or 4 file by the observation reader
  :mod:`gnx_py.rinex_obs` (:func:`read_aux_observations`; bit 0, and a change of bit 1, end a run; bit 0 is
  set on every phase of a power-failure epoch, flag 1), and a receiver millisecond clock-jump guard
  (:func:`clock_jump_corrections`).
* **L1, third carrier** (:func:`third_carrier_flags`, no products needed, at every sampling interval since v1.0.0, ``AUX_MAX_INTERVAL_S``; calibrated at 30 s, measured up to 300 s). Where the RINEX file has another
  carrier ``c`` of the satellite (``gnss.AUX_SIGNALS``: L1 for L2L5, E1 for E5aE5b, B1C for B2aB2b, L5 for
  L1L2, E5b for E1E5a, L2W / E5a / B3I for the single-frequency L1 / E1 / B1I), today's pair test
  (``CSDetector._pair_tests``) also runs on the cross pairs (c, a) and (c, b). An equal slip (n, n) on a
  close pair is a slip on one signal of each cross pair, so their Melbourne-Wuebbena combination changes by
  n wide-lane cycles. Cross-pair geometry-free flags count only where the ionosphere is quiet (gate), the
  ionosphere-free geometry-free combination IF(c, a) - IF(c, b) is tested as well (IGF), and a flag that an
  integer slip on the third carrier alone explains does not break the processed arc (attribution). Epochs
  that today's pair cannot verify are kept where every cross pair verifies them. A common-mode guard drops
  the geometry-free-only cross flags at epochs where they hit most satellites at once (a sudden ionospheric
  event such as the X-class flare of 2024-05-10 06:43 UTC at BRUX and NKLG), so equal slips on many
  satellites are still found by Melbourne-Wuebbena.
* **L2, geometry** (:class:`GeometrySlipDetector`, :class:`_GeometryJob`; precise orbits, or since v1.0.0
  in the TEC session broadcast orbits whose frames name the record of each row (a record switch starts a new piece,
  ``GEO_BROADCAST_TEC``; ``GEO_BROADCAST`` for PPP is off), 30 s sampling, ``cycle_slip_geometry``). The frames of all systems of the run (:func:`frame_arrays`) give the
  modelled range from the frame's orbits and clocks (30 s clocks of ``clk_path`` when their analysis centre is the
  SP3's, :func:`read_clk_30s`; the SP3 clocks where the clock files do not cover a row); per epoch one
  receiver-clock change and an ionosphere-weighted chi2 test of every satellite (T1), windowed ionosphere-free level
  shifts for wide pairs (T2), integer attribution of slips on the third carrier alone. An a-priori position more
  than max(5 m, 4 sigma) from the code solution is replaced by it (:func:`apriori_offset`; a static run more than
  100 m off has the layer off). A kinematic run keeps the static model when the receiver does not move
  (:func:`_epoch_motion`, :func:`_static_health`); a moving receiver is tracked causally at its integrated position,
  or the motion guard switches the layer off.
* **L3, fusion** (:func:`fuse_flags`, :func:`confirm_flags`). A flag that only one layer raises must show a
  persistent ionosphere-free level step (:func:`level_step`, 7 sigma and 7 cm); the geometry attribution vetoes the
  cross flags of a slip on the third carrier alone. Without the geometry layer (sampling other than 30 s, broadcast
  orbits in PPP or without their records) a third-carrier flag that no other test raises breaks the arc only when the joint
  integer estimate of the cross pairs cannot exclude a slip on the processed signals (:func:`_no_processed_slip`): a non-integer carrier step (the 4.6 mm L5 step of a Block IIF satellite) keeps the arc. No
  integer repair (a new arc at every slip).

Basis (library bench, BRUX/NKLG/NYA2 2024-035, NYA2 2024-132, held-out BRUX/NKLG/NYA2 2024-131,
30 s, 30 s clocks, injected slips counted at the slip epoch): (1, 1) on L2/L5 100 % at all seven against
2.1-58.7 % today, (1, 1, 1) 96.1-100 %, L1/L2 (1, 1) 98.7-100 % against 8.8-95.4 %, one cycle on single-frequency
L1 99.6-100 % against 0 %; at most +0.016 new ambiguities per satellite-hour over today. Without products (L0 +
L1): close-pair (1, 1) 95.6-99.7 %, (n, n, n) at NYA2 5-12 %, at most +0.026 (BRUX 2024-131 L2/L5, the flare
day). Numbers per mode in the ``CSDetector`` docstring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..io import rinex_clk, rinex_obs
from .gnss import (
    CLIGHT,
    aux_signal,
    aux_suffix_priority,
    frequency_hz,
    mode_signals,
    signal_spec,
)

log = logging.getLogger(__name__)

#: third carrier: shortest run of epochs with finite code and phase and no loss of lock that is tested
AUX_MIN_RUN_EPOCHS = 12
#: third carrier: longest sampling interval [s] at which it is tested (``CSDetector``; LLI is read at any
#: interval). No limit since v1.0.0 (an earlier version gated it at 60 s): a missed equal slip
#: costs more than a spurious new arc (v1.0.0 measurement: a missed (1, 1) +2.35 to +4.93 cm of converged 6 h PPP, a new
#: arc -0.09 to +0.37 cm). Bench: CS-L machinery on 24 h frames of
#: BRUX/NKLG/NYA2 2024-035 (SP3grg) thinned from 30 s, 2 x 4 sites per piece; with the 60 s gate -> without it:
#: * close pairs L2/L5 and E5a/E5b, (1, 1) at the slip epoch: 120 s 0.0-27.4 -> 81.0-97.2 %, 300 s 0.4-3.5 ->
#:   42.1-98.7 %; (1, 1, 1) (no geometry layer above 30 s) 120 s 0.8-22.1 -> 0.8-52.7 %, 300 s 0.6-2.0 ->
#:   0.6-9.4 %; one cycle on one signal 300 s 67.4-99.5 -> 100 %;
#: * single frequency L1 / E1, one cycle: 120 s 0.0-1.1 -> 63.9-84.2 %, 300 s 0.0-0.6 -> 6.2-58.8 %;
#: * cost: a slip on the third carrier alone breaks the processed arc in 25.1-93.9 % (120 s) and 4.8-96.0 %
#:   (300 s) of the cases against 0-2.8 % (60 s with the layer: 9.0-93.6 %; 30 s with the geometry attribution:
#:   0-8.9 %); new ambiguities per satellite-hour on clean data +0.000-0.016 (120 s) and +0.000-0.033 (300 s;
#:   BRUX L2/L5 0.074 -> 0.107, BRUX L1 0.018 -> 0.051, NYA2 E5a/E5b 0.059 -> 0.084), epochs lost +0.0-0.6
#:   (120 s) and +0.0-1.9 points (300 s; NYA2 L2/L5 0.75 -> 2.60 %);
#: * storm day 2024-132, same bench: (1, 1) close pairs 120 s 0.7-12.1 -> 85.6-95.9 %, 300 s 0.0-3.6 ->
#:   69.1-98.5 %; one cycle L1 / E1 120 s 0.0 -> 58.9-78.7 %, 300 s 0.0-0.7 -> 6.0-55.8 %; cost: third
#:   carrier alone breaks the arc 33.9-94.8 % (120 s), 3.5-97.0 % (300 s); new ambiguities per satellite-hour
#:   +0.000-0.032 (120 s; BRUX L1 0.000 -> 0.032) and +0.000-0.024 (300 s; NYA2 E5a/E5b 0.107 -> 0.131),
#:   epochs lost up to +1.1 (120 s) and +1.5 points (300 s; NYA2 L2/L5 2.27 -> 3.80 %);
#: * default modes L1/L2 and E1/E5a, quiet and storm day: (1, 1) 120 s 0.6-65.1 -> 54.0-100 %, 300 s
#:   0.6-15.5 -> 52.5-100 %; one cycle on one signal 300 s 5.1-67.9 -> 24.5-98.1 % (NYA2 E1/E5a 5.8 -> 95.4 %);
#:   (1, 1, 1) unchanged; cost: third carrier alone breaks the arc 29.8-99.7 % (120 s), 53.4-100 % (300 s);
#:   new ambiguities per satellite-hour +0.000-0.020 (NYA2 E1/E5a storm 0.170 -> 0.190), epochs lost up to
#:   +0.9 points (NYA2 E1/E5a storm 300 s 2.23 -> 3.17 %);
#: * at 300 s on single frequency the layer cannot tell which carrier slipped: a flag is about as likely to
#:   come from a slip of the third carrier as of the processed signal (found ~ broken: NKLG L1 16.5 vs
#:   17.0 %, NKLG E1 58.8 vs 54.2 %, NYA2 L1 6.2 vs 4.8 %); at 120 s the two are also close at NYA2 (E1 80.6
#:   vs 86.6 %).
#: TEC/STEC sessions pass the same slip context, so above 60 s they get the layer too (not benched; no
#: ionosphere test samples above 60 s).
#: Set a finite value (e.g. 60.0) to switch the layer off above it (one INFO line per run).
AUX_MAX_INTERVAL_S = float("inf")
#: third carrier: missing epochs (no loss of lock) bridged inside a run, at most this many in a row. Without
#: the bridge a lost epoch starts a new test window with one-sided noise estimates. Bench with 2 % of the
#: third-carrier epochs removed at random (BRUX/NKLG/NYA2 2024-035, 11 mode-days), without -> with the bridge:
#: new ambiguities per satellite-hour above the complete data +0.000-0.020 -> -0.010-+0.010, (1, 1) on the
#: close pairs 85.0-95.3 -> 94.2-100 %, one cycle on L1 86.1 -> 93.2 %
AUX_BRIDGE_MAX_EPOCHS = 2
#: attribution of a cross-pair flag to a slip on the third carrier alone: integer slip vectors searched
#: within +-ATT_MAX cycles per carrier; accepted when the best vector fits (chi2 of the statistics at 99 %,
#: 2 or 3 statistics) and every other vector is at least 6 sigma away (chi2 >= 36)
ATT_MAX = 12
ATT_BEST_CHI2 = {2: 9.21, 3: 11.34}
ATT_ALT_CHI2 = 36.0
#: step sigma floors (wide-lane cycles / m): the local MAD underestimates at high elevation
ATT_SIG_MW_MIN = 0.05
ATT_SIG_GF_MIN = 0.002
#: integer check of an unverified third-carrier flag without the geometry layer
#: (:func:`_no_processed_slip`): the flag breaks the processed arc unless the joint integer estimate of the
#: attribution statistics puts every vector with a processed slip at least INT_EXCL_CHI2 above the best one
#: without (6 sigma, as ATT_ALT_CHI2) and that vector explains each Melbourne-Wuebbena step to within INT_FRAC_CYC
#: cycles. Without the check a non-integer carrier step split the arc: BRUX 2024-035 G27 (Block IIF) L2/L5 at its
#: 18:00 broadcast record switch, a 4.4-4.7 mm (0.018-cycle) step of L5 flagged by IF(L1, L2) - IF(L1, L5) only,
#: levelled STEC -5.71 TECU across the split (with precise orbits the IF-level confirmation already rejects it)
#: INT_FRAC_CYC is a fixed fraction of a cycle (code noise of the MW step), not scaled by the step sigma, and there
#: is no fit bound on chi2_0: G27 has MW -0.177 cycle (3.5 sigma at the ATT_SIG_MW_MIN floor, 0.023 cycle below
#: the gate) and chi2_0 = 17.8 > ATT_BEST_CHI2[3]. Probe of every decision on the broadcast TEC runs of BRUX / NKLG /
#: NYA2 2024-035 and 2024-132 (9 modes, 52 runs): 80 decisions, 6 exclusions, largest excluded |MW| 0.177 cycle;
#: one flag kept by this gate alone (NKLG 2024-035 E1E5a, 0.280 cycle at chi2 margin 39.6)
INT_EXCL_CHI2 = ATT_ALT_CHI2
INT_FRAC_CYC = 0.2
#: ... and the geometry-free step within INT_GF_K sigma of that vector (an ionospheric disturbance keeps the flag:
#: BRUX 2024-05-10 06:43:30 X-class flare, an injected (1, 1) on E5a/E5b of E21 has MW -0.98 / -0.98 cycles and a
#: geometry-free step 11 sigma from both integer candidates; G27 above: MW -0.18 / -0.11 cycles, GF 0.4 sigma)
INT_GF_K = 3.0
#: common-mode (flare) guard: geometry-free-only cross flags (+-1 epoch) on at least CM_MIN_SATS and at least
#: CM_FRAC of the satellites with an active third-carrier run make an ionospheric event epoch
CM_MIN_SATS = 4
CM_FRAC = 0.5
#: event epochs extended by one epoch before and CM_TAIL epochs after (the disturbance decays over minutes;
#: BRUX 2024-05-10 05-08 h broadcast L2L5/E5aE5b: Galileo new ambiguities 0.048 with +1 epoch, 0 = today from +3;
#: +5 loses one (1, 1, 1) site of 29)
CM_TAIL = 3
#: receiver millisecond clock-jump guard: |median epoch difference of P - L| above half a millisecond of range,
#: on at least CLOCK_JUMP_MIN_SATS satellites with CLOCK_JUMP_AGREE of them within CLOCK_JUMP_TOL_M of it
CLOCK_JUMP_MS_M = CLIGHT * 1e-3
CLOCK_JUMP_MIN_SATS = 4
CLOCK_JUMP_AGREE = 0.9
CLOCK_JUMP_TOL_M = 1000.0

#: (path, reason) already logged by the side reader
_AUX_LOGGED: set = set()


# ============================================================================ side reader
def _log_once(path, reason: str, level: int = logging.INFO) -> None:
    key = (str(path), reason)
    if key not in _AUX_LOGGED:
        _AUX_LOGGED.add(key)
        log.log(level, "cycle slips: %s (%s)", reason, path)


def _read_rinex_obs(path, system: str, meas: tuple, tlim) -> Optional[pd.DataFrame]:
    """Observables of one system whose code starts with one of ``meas`` (and ``<code>lli`` for every phase),
    indexed (``sv``, ``time``) within ``tlim``; None without records. Raises for an unreadable file.

    A thin wrapper over :func:`gnx_py.rinex_obs.read_obs` (RINEX 2, 3 and 4 in every compression; values NaN
    where blank or 0.0; LLI 0 where blank, NaN where the value is missing, bit 0 set on the phases of a
    power-failure epoch). The reader keeps the parse of the last file and ``tlim`` in process, keyed by size and
    content digest, so the session's own read and the side read of every system share one parse."""
    res = rinex_obs.read_obs(path, systems=system, tlim=tlim, kinds=("C", "P", "L"), lli=True)
    df = res.frames.get(system)
    if df is None:
        return None
    cols = [c for c in df.columns if c.startswith(meas)]
    if not cols:
        return None
    lli = res.lli[system]
    out = {}
    for c in cols:
        out[c] = df[c].to_numpy(dtype=float)
        if c.startswith("L"):
            out[f"{c}lli"] = lli[c].to_numpy(dtype=float)
    return pd.DataFrame(out, index=df.index)


def _rinex2_codes(signal: str, order) -> list:
    """RINEX 2 code observables of ``signal`` in the priority order of the RINEX 3 tracking attributes
    ``order``: 'P<band>' for the GPS P(Y) trackings (W, P, Y), 'C<band>' for every other (RINEX 2.11 Table A1:
    one code per kind and band, no attribute)."""
    spec = signal_spec(signal)
    band = spec.code_prefix[1:]
    out = []
    for suffix in order:
        code = f"P{band}" if spec.system == "G" and suffix in ("W", "P", "Y") else f"C{band}"
        if code not in out:
            out.append(code)
    return out


def _pick_per_satellite(df: pd.DataFrame, columns: list, chosen: Optional[dict] = None) -> pd.Series:
    """Per satellite the first of ``columns`` (priority order) with any finite value (one tracking per arc);
    ``chosen``: filled with {satellite: column}."""
    out = np.full(len(df), np.nan)
    cols = [c for c in columns if c in df.columns]
    values = [df[c].to_numpy(dtype=float) for c in cols]
    for sv, rows in df.groupby(level="sv", sort=False).indices.items():
        for c, v in zip(cols, values):
            if np.isfinite(v[rows]).any():
                out[rows] = v[rows]
                if chosen is not None:
                    chosen[sv] = c
                break
    return pd.Series(out, index=df.index)


def read_aux_observations(path, system: str, aux: Optional[str], processed_phases=(),
                          tlim=None) -> Optional[pd.DataFrame]:
    """Third carrier and loss-of-lock indicators of one system, read from the RINEX file for slip detection.

    ``aux``: the third carrier (``gnss.AUX_SIGNALS``) or None; ``processed_phases``: the phase observables of the
    processed signals (frame columns, e.g. ``('L2W', 'L5Q')``, RINEX 2 ``('L1', 'L2')``) whose LLI is wanted;
    ``tlim``: the run's time limit. Returns a frame indexed (``sv``, ``time``) with ``aux_P``/``aux_L`` [m] (NaN
    without the carrier), ``aux_lli`` (NaN: no observation) and ``<phase>lli`` of the processed phases (0 where
    the indicator is blank). The attributes ``aux_code``/``aux_phase`` name the observables used (per satellite
    the first tracking of ``gnss.aux_suffix_priority`` with data; a RINEX 2 file has one code per kind and band,
    so its bare codes are taken: 'C1'/'P1', 'C5', 'L5', ...). None for an unreadable file (warning, logged
    once). The file (RINEX 2, 3 or 4; plain, gzip, Unix compress, bzip2, zip or Hatanaka CRINEX) is read by
    :mod:`gnx_py.rinex_obs`, which keeps the parse of all systems of the last file and ``tlim`` in process by
    content (shared with the session's own read): 0.86-0.95 s for a 30 s station-day with all systems and
    observation kinds (BRUX/NYA2 2024-035, NKLG 2024-131, Hatanaka + gzip), 0.016-0.025 s per call after
    that; aux and LLI frames identical to the former parser of this module on these days in all eight
    ``AUX_SIGNALS`` modes; values and LLI of every observation file of the validation identical to that parser and to
    georinex 1.16.2, which reads the LLI of L1*/L2* phases only (RINEX 2: L1/L2/L5; three-way digests). RINEX 2 rows get LLI and the third carrier since v1.0.0 (LLI
    bit 2, the RINEX 2 anti-spoofing flag, is ignored by ``CSDetector._lli_slips``).
    """
    if path is None:
        return None
    meas = []
    if aux is not None:
        spec = signal_spec(aux)
        meas += [spec.code_prefix, spec.phase_prefix, "P" + spec.code_prefix[1:]]  # P<band>: RINEX 2 P code
    meas += [p for p in processed_phases if p]
    meas = tuple(dict.fromkeys(meas))
    if not meas:
        return None
    try:
        df = _read_rinex_obs(path, system, meas, tlim)
    except Exception as exc:  # noqa: BLE001 - the detector falls back to the frame
        _log_once(path, f"observation file not readable for LLI / third carrier: {exc}", logging.WARNING)
        return None
    if df is None or df.empty:
        return None
    out = pd.DataFrame(index=df.index)
    out["aux_P"] = np.nan
    out["aux_L"] = np.nan
    out["aux_lli"] = np.nan
    out.attrs["aux_code"] = out.attrs["aux_phase"] = None
    if aux is not None:
        spec = signal_spec(aux)
        order = aux_suffix_priority(aux)
        # RINEX 3/4 codes, then the bare RINEX 2 codes of the band (a file holds only one of the two kinds)
        codes = [f"{spec.code_prefix}{s}" for s in order] + _rinex2_codes(aux, order)
        phases = [f"{spec.phase_prefix}{s}" for s in order] + [spec.phase_prefix]
        code_sv: dict = {}
        out["aux_P"] = _pick_per_satellite(df, codes, code_sv)
        lam = CLIGHT / spec.frequency_hz
        phase = _pick_per_satellite(df, phases)
        out["aux_L"] = phase * lam
        # LLI of the phase picked for each satellite
        lli = np.full(len(df), np.nan)
        cols = [c for c in phases if c in df.columns]
        values = {c: df[c].to_numpy(dtype=float) for c in cols}
        for rows in df.groupby(level="sv", sort=False).indices.values():
            for c in cols:
                if np.isfinite(values[c][rows]).any():
                    if f"{c}lli" in df.columns:
                        lli[rows] = df[f"{c}lli"].to_numpy(dtype=float)[rows]
                    break
        out["aux_lli"] = lli
        out.attrs["aux_code"] = [c for c in codes if c in df.columns]
        out.attrs["aux_code_sv"] = code_sv
        out.attrs["aux_phase"] = [c for c in phases if c in df.columns]
    for p in processed_phases:
        if p and f"{p}lli" in df.columns:
            out[f"{p}lli"] = df[f"{p}lli"].to_numpy(dtype=float)
    return out


# ============================================================================ context
@dataclass
class SlipContext:
    """Inputs of the layered detector for one system (built by :func:`build_slip_context`).

    ``tests``: the slip tests run (``cycle_slips_detection``; False: gaps and LLI only); ``aux_carrier``:
    LLI and the third carrier are read (``cycle_slip_aux_carrier``); ``windup_applied``: the frame phases
    already carry the wind-up correction (TEC), so the third carrier gets it too. The side read happens on
    first use (:meth:`aux_frame`).
    """

    system: str
    mode: str
    aux_signal: Optional[str] = None
    processed_phases: tuple = ()
    tests: bool = True
    aux_carrier: bool = True
    windup_applied: bool = False
    kinematic: bool = False
    xyz: Optional[np.ndarray] = None
    interval_s: Optional[float] = None
    clk_path: Optional[list] = None
    geometry: object = field(default=None, repr=False)
    _loader: Optional[Callable[[], Optional[pd.DataFrame]]] = field(default=None, repr=False)
    _aux: object = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def geometry_rows(self, prn, times_ns) -> Optional[dict]:
        """Geometry-layer (L2) results per row (satellite ``prn``, epoch ``times_ns`` [ns]), or None when the layer
        is off for the run (:class:`_GeometryJob`, computed on first use for all systems of the receiver)."""
        if self.geometry is None or not self.tests:
            return None
        res = self.geometry.result()
        return None if res is None else res.rows(np.asarray(prn).astype(str), np.asarray(times_ns, dtype=np.int64))

    def aux_frame(self) -> Optional[pd.DataFrame]:
        """Side-read third carrier and LLI (``read_aux_observations``), or None."""
        if not self.aux_carrier:
            return None
        if not self._loaded:
            self._aux = self._loader() if self._loader is not None else None
            self._loaded = True
        return self._aux

    def set_aux_frame(self, df: Optional[pd.DataFrame]) -> None:
        """Use ``df`` (format of :func:`read_aux_observations`) instead of reading the file."""
        self._aux, self._loaded = df, True


def processed_phase_columns(frame: pd.DataFrame, mode: str) -> tuple:
    """Phase column of each signal of ``mode`` in ``frame`` (first ``<prefix>*`` column, as ``CSDetector``)."""
    out = []
    for s in mode_signals(mode):
        pre = signal_spec(s).phase_prefix
        cols = [c for c in frame.columns if str(c).startswith(pre) and not str(c).endswith(("lli", "ssi"))]
        out.append(cols[0] if cols else None)
    return tuple(out)


def build_slip_context(config, frames_by_sys: dict, modes_by_sys: dict, xyz=None, interval_s=None,
                       windup_applied: bool = False, kinematic: bool = False, clk_path=None,
                       broadcast_geometry: Optional[bool] = None) -> dict:
    """``{system: SlipContext}`` for the frames the detector receives.

    ``config``: the session configuration (``obs_path``, ``time_limit``, ``cycle_slips_detection``,
    ``cycle_slip_aux_carrier``, ``cycle_slip_geometry``, ``orbit_type``, ``sp3_path``, ``clk_path``);
    ``frames_by_sys``/``modes_by_sys``: the frames handed to the detector and their modes; ``xyz``: a-priori
    receiver position [m]; ``interval_s``: sampling interval [s]; ``windup_applied``: the frame phases carry the
    wind-up correction (TEC); ``kinematic``: the receiver may move (position change estimated per epoch);
    ``clk_path``: 30 s clock files for the geometry layer (default: ``config.clk_path``); ``broadcast_geometry``:
    the geometry layer may run with broadcast orbits (None: ``GEO_BROADCAST``, the PPP sessions; the TEC session
    passes ``GEO_BROADCAST_TEC``).

    The geometry layer (L2, :class:`GeometrySlipDetector`) is shared by the contexts of all systems (one
    receiver-clock change for all of them) and runs on first use. It is on with the master switch, with
    ``cycle_slip_geometry``, precise orbits or (``broadcast_geometry``) broadcast orbits whose frames name their
    records, a sampling interval of 30 s (+-1 s) and an a-priori position on the Earth's surface; otherwise it is off
    with one INFO line. ``clk_path`` is used only when its analysis centre
    is the one of the SP3 files (:func:`usable_clk_paths`), otherwise ignored with a WARNING.
    """
    tests = bool(getattr(config, "cycle_slips_detection", True))
    use_aux = bool(getattr(config, "cycle_slip_aux_carrier", True))
    obs_path = getattr(config, "obs_path", None)
    tlim = getattr(config, "time_limit", None)
    if clk_path is None:
        clk_path = getattr(config, "clk_path", None)
    out = {}
    for system, frame in frames_by_sys.items():
        if frame is None:
            continue
        mode = modes_by_sys[system]
        phases = processed_phase_columns(frame, mode)
        aux = aux_signal(mode) if tests else None

        def loader(system=system, aux=aux, phases=phases):
            return read_aux_observations(obs_path, system, aux, phases, tlim)

        out[system] = SlipContext(
            system=system, mode=mode, aux_signal=aux, processed_phases=phases, tests=tests,
            aux_carrier=use_aux, windup_applied=bool(windup_applied), kinematic=bool(kinematic),
            xyz=None if xyz is None else np.asarray(xyz, dtype=float), interval_s=interval_s,
            clk_path=clk_path, _loader=loader)
    if out and tests and bool(getattr(config, "cycle_slip_geometry", True)):
        orbit_type = getattr(config, "orbit_type", "precise")
        allow = GEO_BROADCAST if broadcast_geometry is None else bool(broadcast_geometry)
        why = _geometry_off_reason(orbit_type, xyz, interval_s, frames_by_sys, broadcast=allow)
        if why:
            log.info("cycle slips: geometry layer off (%s); third carrier and today's tests", why)
        else:
            clk, why_clk = usable_clk_paths(clk_path, getattr(config, "sp3_path", None))
            if why_clk:
                log.warning("cycle slips: clk_path ignored: %s", why_clk)
            job = _GeometryJob(frames_by_sys, modes_by_sys, out, xyz, interval_s, windup_applied, kinematic, clk,
                               broadcast=orbit_type != "precise")
            for ctx in out.values():
                ctx.geometry = job
    return out


def _geometry_off_reason(orbit_type, xyz, interval_s, frames_by_sys=None, broadcast=None) -> Optional[str]:
    """Why the geometry layer cannot run (None: it can): broadcast orbits when ``broadcast`` (None:
    ``GEO_BROADCAST``) is False or the frames do not name the record of each row (``BROADCAST_RECORD_COLUMNS``),
    a sampling interval other than 30 s
    (calibrated there; the confirmation windows are in epochs) or an a-priori position off the Earth's surface.
    With broadcast orbits a change of the record of a satellite starts a new piece of the layer
    (:func:`frame_arrays`), so the record switch never reaches the epoch-differenced residual (an earlier version
    kept the layer off because the switches moved the model: 0 -> 0.135 false breaks per satellite-hour at BRUX
    E5a/E5b in its prototype)."""
    if orbit_type != "precise":
        if not (GEO_BROADCAST if broadcast is None else broadcast):
            return "broadcast orbits"
        frames = [f for f in (frames_by_sys or {}).values() if f is not None and len(f)]
        if not frames or not all(_record_columns(f) for f in frames):
            return "broadcast orbits without the record of each row (Toe) in the frames"
    if interval_s is None or abs(float(interval_s) - GEO_INTERVAL_S) > GEO_INTERVAL_TOL_S:
        return f"sampling interval {interval_s} s, calibrated at {GEO_INTERVAL_S:g} s"
    r = float(np.linalg.norm(np.asarray(xyz, dtype=float))) if xyz is not None else float("nan")
    if not GEO_XYZ_MIN_M < r < GEO_XYZ_MAX_M:
        return f"a-priori position |XYZ| = {r:.0f} m is not on the Earth's surface"
    return None


# ============================================================================ L0: clock-jump guard
def clock_jump_corrections(epoch: np.ndarray, piece: np.ndarray, p_minus_l: np.ndarray) -> np.ndarray:
    """Receiver millisecond clock jumps: per row the accumulated jump [m] to subtract from every code.

    ``epoch``: integer epoch index of each row; ``piece``: continuous-piece id of each row (rows of one piece in
    time order); ``p_minus_l``: code minus phase of the first processed signal [m]. At an epoch where the median
    epoch difference of P - L over the satellites exceeds half a millisecond of range, with at least
    ``CLOCK_JUMP_MIN_SATS`` satellites of which ``CLOCK_JUMP_AGREE`` lie within ``CLOCK_JUMP_TOL_M`` of the
    median, J = round(m / (c 1 ms)) c 1 ms holds from that epoch on. A code-only jump (and a phase-only jump)
    then no longer moves Melbourne-Wuebbena or code minus phase; the geometry-free phase cancels it anyway.
    """
    n = len(epoch)
    corr = np.zeros(n)
    if n == 0:
        return corr
    d = np.full(n, np.nan)
    same = np.zeros(n, dtype=bool)
    same[1:] = (piece[1:] == piece[:-1]) & (epoch[1:] == epoch[:-1] + 1)
    d[1:] = np.where(same[1:], p_minus_l[1:] - p_minus_l[:-1], np.nan)
    ok = np.isfinite(d)
    if not ok.any():
        return corr
    e, v = epoch[ok], d[ok]
    order = np.argsort(e, kind="stable")
    e, v = e[order], v[order]
    bounds = np.flatnonzero(np.r_[True, e[1:] != e[:-1], True])
    jumps = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        vals = v[a:b]
        if len(vals) < CLOCK_JUMP_MIN_SATS:
            continue
        m = float(np.median(vals))
        if abs(m) <= 0.5 * CLOCK_JUMP_MS_M:
            continue
        if np.mean(np.abs(vals - m) <= CLOCK_JUMP_TOL_M) < CLOCK_JUMP_AGREE:
            continue
        jumps[int(e[a])] = round(m / CLOCK_JUMP_MS_M) * CLOCK_JUMP_MS_M
    if not jumps:
        return corr
    ks = np.array(sorted(jumps))
    cum = np.cumsum([jumps[k] for k in ks])
    pos = np.searchsorted(ks, epoch, side="right")
    corr = np.where(pos > 0, cum[np.maximum(pos - 1, 0)], 0.0)
    log.info("cycle slips: %d receiver millisecond clock jump(s) removed from the detector's codes", len(ks))
    return corr


def half_cycle_breaks(flag) -> np.ndarray:
    """New-arc flags from the half-cycle status of one continuous piece (contract
    C3): ``flag`` is the per-row bool ``lli_half_cycle`` of the observation frame (RINEX LLI bit 1, half-cycle
    ambiguity / slip possible, in the LLI field of the RINEX 3.05 observation data record; missing values count
    as False). A row where the status changes (set or clear) against the previous row starts a new arc: when it
    clears the receiver has resolved the half cycle and may have shifted the phase by half a cycle. This is the
    transition rule of RTKLIB-EX 2.5.1 rtkpos.c:707-720 (RTKLIB 2.4.2 p13 rtkpos.c:591-596) and of
    ``CSDetector._lli_slips`` on the LLI bits; RTKLIB 2.4.2 p13 ppp.c:714-716 (detslp_ll) resets the phase bias
    at every epoch with LLI & 3 instead, which is not followed. While the flag is set the float ambiguity stays
    continuous; the integer fix leaves it out (``gnx_py.ppp.pppar``). The first row is never flagged (an arc
    starts there anyway).
    """
    a = pd.Series(flag, dtype=object)
    a = a.where(a.notna(), False).to_numpy(dtype=bool)
    out = np.zeros(len(a), dtype=bool)
    out[1:] = a[1:] != a[:-1]
    return out


# ============================================================================ L1: third carrier
def _runs(mask: np.ndarray, min_len: int) -> list:
    """[start, stop) runs of True of at least ``min_len``."""
    m = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    d = np.diff(m)
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a >= min_len]


def _dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """``mask`` widened by ``r`` epochs on each side (no wrap-around)."""
    out = np.asarray(mask, dtype=bool).copy()
    for k in range(1, r + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def _step_sigma(y, j, w, linear, lo, hi):
    """Step of ``y`` at epoch ``j`` (``w`` epochs each side within [lo, hi)) and its 1-sigma from the local noise
    of the normalised steps (as ``_level_shifts``; epochs within ``w`` of ``j`` left out, at least 8)."""
    from .tools import _PrefixSums

    n = len(y)
    sums = _PrefixSums(y)
    s, v = sums.step(max(lo, j - w), j, min(hi, j + w), linear)
    jj = np.arange(1, n)
    st, vv = sums.step(np.maximum(0, jj - w), jj, np.minimum(n, jj + w), linear)
    z = np.full(n, np.nan)
    # vv is inf (a side without data) or > 0 (1/n_l + 1/n_r + ...): the normalised step only where it is finite
    fin = np.isfinite(vv)
    z[1:][fin] = st[fin] / np.sqrt(vv[fin])
    half = max(20, 4 * w)
    a, b = max(0, j - half), min(n, j + half + 1)
    zz = z[a:b].copy()
    zz[max(0, j - w + 1 - a):max(0, j + w - a)] = np.nan
    zz = zz[np.isfinite(zz)]
    if len(zz) < 8 or not np.isfinite(v):
        return float(s), np.inf
    noise = 1.4826 * np.median(np.abs(zz - np.median(zz)))
    return float(s), float(noise * np.sqrt(v))


def _slip_grid(m: int, res: list, aux_first: bool, jj: int):
    """Integer slip-vector grid of one tested run at position ``jj`` (the statistics of :func:`_attribute`).

    Returns ``(chi2, dc, dp, stats)``: chi2 of every vector (third-carrier slip ``dc`` and processed slips ``dp``,
    each within +-``ATT_MAX`` cycles) against the Melbourne-Wuebbena steps of every cross pair [cycles] and the
    geometry-free step of the first one [m] with their local sigmas (floors ``ATT_SIG_MW_MIN`` /
    ``ATT_SIG_GF_MIN``); ``stats`` = ``[(kind, pair, step, sigma), ...]``. None when a step or its sigma is not
    finite (a window side without data).
    """
    stats = []
    for r in res:
        s, sg = _step_sigma(r["mw"], jj, r["w_mw"], False, 0, m)
        stats.append(("mw", r, s, max(sg, ATT_SIG_MW_MIN)))
    s, sg = _step_sigma(res[0]["gf"], jj, res[0]["w_gf"], True, 0, m)
    stats.append(("gf", res[0], s, max(sg, ATT_SIG_GF_MIN)))
    if not all(np.isfinite(x[2]) and np.isfinite(x[3]) for x in stats):
        return None
    rng = np.arange(-ATT_MAX, ATT_MAX + 1)
    npr = len(res) if aux_first else 1
    grids = np.meshgrid(*([rng] * (npr + 1)), indexing="ij")
    dc, dp = grids[0], grids[1:]
    chi2 = np.zeros(dc.shape)
    for i, (kind, r, s, sg) in enumerate(stats):
        chi2 += ((s - _slip_model(kind, r, i, dc, dp, aux_first)) / sg) ** 2
    return chi2, dc, dp, stats


def _slip_model(kind: str, r: dict, i: int, dc, dp, aux_first: bool):
    """Model of statistic ``i`` (``kind`` 'mw' [wide-lane cycles] or 'gf' [m], pair ``r``) for the slip vectors
    ``dc`` (third carrier) / ``dp`` (processed signals): MW step of (c, x) = sign(fc - fx) (dc - dx), GF step of
    (c, a) = lambda_c dc - lambda_a da (single frequency: the pair (a, c))."""
    k = i if kind == "mw" else 0
    l1, l2 = r["lam"]
    d1, d2 = (dc, dp[k]) if aux_first else (dp[0], dc)
    sgn = 1.0 if l1 < l2 else -1.0  # sign(f1 - f2)
    return sgn * (d1 - d2) if kind == "mw" else (l1 * d1 - l2 * d2)


def _attribute(cross: list, j: int, epochs: list):
    """Joint integer slip vector of the third carrier and the processed signals at ``epochs`` (positions in
    the tested series of the run; ``j`` the first).

    Statistics: Melbourne-Wuebbena steps of every cross pair [cycles] and the geometry-free step of the first
    one [m], each with its local sigma. Model: MW step of (c, x) = sign(fc - fx) (dc - dx), GF step of
    (c, a) = lambda_c dc - lambda_a da (single frequency: the pair (a, c)). Returns
    ``(accepted, processed slips tuple, third-carrier slip)``.
    """
    best = None
    for m, res, aux_first in cross:
        if not 0 < j < m - 1:
            continue
        for jj in epochs:
            if not 0 < jj < m:
                continue
            g = _slip_grid(m, res, aux_first, jj)
            if g is None:
                continue
            chi2, dc, dp, stats = g
            flat = np.sort(chi2.ravel())
            i0 = np.unravel_index(np.argmin(chi2), chi2.shape)
            cand = (flat[0], flat[1], tuple(int(g[i0]) for g in dp), int(dc[i0]), len(stats))
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        return False, None, None
    ok = best[0] <= ATT_BEST_CHI2[best[4]] and best[1] >= ATT_ALT_CHI2
    return bool(ok), best[2], best[3]


def _no_processed_slip(cross: list, j: int, epochs: list) -> bool:
    """True when the joint integer estimate excludes a slip on the processed signals at every one of ``epochs``
    (an unverified third-carrier flag without the geometry layer).

    At each epoch (grid of :func:`_slip_grid`): chi2_0 = the best vector without a processed slip (dp = 0; the
    third carrier free, the zero vector included), chi2_1 = the best vector with one (dp != 0). Excluded when
    chi2_1 - chi2_0 >= ``INT_EXCL_CHI2`` (a processed slip at least 6 sigma worse, as ``ATT_ALT_CHI2``) and the
    dp = 0 vector explains every step: each Melbourne-Wuebbena step of a cross pair [wide-lane cycles, i.e.
    dc - dx in cycles of the carriers] to within ``INT_FRAC_CYC`` of a cycle (code noise), and the geometry-free
    step of the first pair to within ``INT_GF_K`` of its sigma and ``INT_FRAC_CYC`` wavelengths of its shorter
    carrier. The geometry-free step is the only statistic that tells an equal slip (n, n) on the processed pair
    from a slip -n of the third carrier alone (they differ there by n |lambda_c - lambda_a|, 5.4 cm on L1/L2,
    6.4 cm on E1/E5a, and not at all in the Melbourne-Wuebbena steps; in the ionosphere-free combination of a
    cross pair (c, a) they differ by n lambda_NL(c, a) = n c / (f_c + f_a), 10.70 cm on L1/L2 and 10.89 cm on
    L1/L5, which without the geometry layer cannot be told from geometry and clock, and in IF(c, a) - IF(c, b) by
    n (lambda_NL(c, a) - lambda_NL(c, b)), 1.99 mm for L2/L5 with L1), so an ionospheric disturbance there (a solar
    flare, a gradient) keeps the flag. A step of a fifth of a cycle or
    more that no integer vector explains, an ambiguous estimate, or an epoch without complete statistics keeps the
    flag as well (False), so a real slip is not lost to the check.
    """
    seen = False
    for m, res, aux_first in cross:
        if not 0 < j < m - 1:
            continue
        for jj in epochs:
            if not 0 < jj < m:
                continue
            g = _slip_grid(m, res, aux_first, jj)
            if g is None:
                return False
            chi2, dc, dp, stats = g
            proc = np.zeros(dc.shape, dtype=bool)
            for d in dp:
                proc |= d != 0
            c0 = np.where(proc, np.inf, chi2)
            chi2_0, chi2_1 = float(c0.min()), float(np.where(proc, chi2, np.inf).min())
            if not chi2_1 - chi2_0 >= INT_EXCL_CHI2:
                return False
            i0 = np.unravel_index(np.argmin(c0), chi2.shape)
            v_dc = dc[i0]
            v_dp = [d[i0] for d in dp]
            for i, (kind, r, s, sg) in enumerate(stats):
                tol = INT_FRAC_CYC if kind == "mw" else min(INT_FRAC_CYC * min(r["lam"]), INT_GF_K * sg)
                if not abs(s - float(_slip_model(kind, r, i, v_dc, v_dp, aux_first))) < tol:
                    return False
            seen = True
    return seen


def _aux_runs(lc, pc, lli_c) -> list:
    """Tested series of the third carrier in one piece: ``[(positions, ...)]``, one integer array per run.

    A run holds the epochs with finite code and phase and no loss of lock: LLI bit 0, or a change of bit 1
    (half-cycle status) against the previous epoch with data, ends it (the rule of the processed phases,
    ``CSDetector._lli_slips``). Missing epochs without loss of lock, at most ``AUX_BRIDGE_MAX_EPOCHS`` in a row,
    are bridged: the run continues over them and the tests see the series without them. A run is tested with
    at least ``AUX_MIN_RUN_EPOCHS`` epochs of data.
    """
    n = len(lc)
    fin = np.isfinite(lc) & np.isfinite(pc)
    lock = np.ones(n, dtype=bool)
    if lli_c is not None:
        li = np.nan_to_num(np.asarray(lli_c, dtype=float), nan=0.0).astype(np.int64)
        lock = (li & 1) == 0
        at = np.flatnonzero(fin)
        half = (li[at] & 2) > 0
        lock[at[1:]] &= half[1:] == half[:-1]
    ok = fin & lock
    span = ok.copy()
    for a, b in _runs(~ok, 1):
        if a > 0 and b < n and b - a <= AUX_BRIDGE_MAX_EPOCHS and lock[a:b].all():
            span[a:b] = True
    out = []
    for lo, hi in _runs(span, AUX_MIN_RUN_EPOCHS):
        pos = lo + np.flatnonzero(ok[lo:hi])
        if len(pos) >= AUX_MIN_RUN_EPOCHS:
            out.append(pos)
    return out


def third_carrier_flags(l_proc, p_proc, f_proc, base_slips, lc, pc, fc, lli_c=None, interval=30.0) -> dict:
    """Third-carrier (L1) tests of one continuous piece of the processed signals.

    ``l_proc``/``p_proc``: phases / codes [m] of the processed signals (one or two arrays), ``f_proc`` their
    frequencies [Hz]; ``base_slips``: slip flags of today's test of the processed signals (MW | GF, or code
    minus phase), used by the attribution; ``lc``/``pc``: third-carrier phase / code [m] (NaN where missing),
    ``fc`` its frequency; ``lli_c``: its loss-of-lock indicators (NaN: none).

    Runs of the third carrier (:func:`_aux_runs`: no loss of lock, short data gaps bridged, at least
    ``AUX_MIN_RUN_EPOCHS`` epochs) are tested: the pair test on (c, a) and (c, b) (single frequency: (a, c));
    gated - Melbourne-Wuebbena flags, and geometry-free flags only where their threshold is at its floor
    0.5*|lambda_c - lambda_x| (quiet ionosphere); dual frequency also IGF; a flag that the processed pair does
    not show (+-1 epoch) and that the joint integer estimate explains by a slip on the third carrier alone is
    cleared at j-1..j+1. A flag at the first epoch after a bridged gap also marks the gap epochs (the slip may
    lie in the gap: they become one-epoch arcs).

    Returns ``cross`` (flags after gate, IGF and attribution), ``gf_only`` (gated geometry-free cross flags with
    no Melbourne-Wuebbena or IGF flag within +-1 epoch, before attribution; for the common-mode guard),
    ``verified`` (dual frequency: every cross pair would see a one-cycle slip, MW threshold <= 1 cycle; epochs
    with third-carrier data in a run), ``active`` (epochs with third-carrier data in a tested run) and
    ``unconfirmed`` (the ``cross`` flags away from the processed pair's own flags
    whose joint integer estimate excludes a slip on the processed signals, :func:`_no_processed_slip`; the
    fusion keeps them out of the arc breaks when the geometry layer is off, :func:`fuse_flags`).
    """
    from .tools import CSDetector as K, _level_shifts

    l_proc = [np.asarray(x, dtype=float) for x in l_proc]
    p_proc = [np.asarray(x, dtype=float) for x in p_proc]
    lc, pc = np.asarray(lc, dtype=float), np.asarray(pc, dtype=float)
    n = len(lc)
    dual = len(l_proc) == 2
    out = {"cross": np.zeros(n, bool), "gf_only": np.zeros(n, bool), "verified": np.zeros(n, bool),
           "active": np.zeros(n, bool), "unconfirmed": np.zeros(n, bool)}
    base = np.asarray(base_slips, dtype=bool)
    cbase = np.r_[0, np.cumsum(base)]
    for pos in _aux_runs(lc, pc, lli_c):
        m = len(pos)
        out["active"][pos] = True
        xc, yc = lc[pos], pc[pos]
        igf = None
        if dual:
            res = [K._pair_tests(xc, lx[pos], yc, px[pos], fc, fx, interval)
                   for lx, px, fx in zip(l_proc, p_proc, f_proc)]
            fa, fb = f_proc
            ifa = (fc ** 2 * xc - fa ** 2 * l_proc[0][pos]) / (fc ** 2 - fa ** 2)
            ifb = (fc ** 2 * xc - fb ** 2 * l_proc[1][pos]) / (fc ** 2 - fb ** 2)
            igf, _, _ = _level_shifts(ifa - ifb, linear=True, k=K.GF_K, floor=K.GF_FLOOR_M,
                                      window=K._window(K.DUAL_WINDOW_S, interval, 3))
        else:
            res = [K._pair_tests(l_proc[0][pos], xc, p_proc[0][pos], yc, f_proc[0], fc, interval)]
        run = [(m, res, dual)]
        gated = []
        gf = np.zeros(m, bool)
        mw = np.zeros(m, bool)
        ver = np.ones(m, bool)
        for r in res:
            floor = 0.5 * abs(r["lam"][0] - r["lam"][1])
            g = r["gf_flags"] & (r["gf_thr"] <= floor + 1e-9)
            gated.append(r["mw_flags"] | g)
            gf |= g
            mw |= r["mw_flags"]
            ver &= r["mw_thr"] <= 1.0
        fl = np.zeros(m, bool)
        for g in gated:
            fl |= g
        if igf is not None:
            fl |= igf
            mw |= igf
        gfo = gf & ~_dilate(mw, 1)
        # first two tested epochs after a bridged gap: the linear geometry-free step test sees the missing
        # epoch's ionospheric change as a kink (BRUX/NKLG 2024-035, 2 % loss: 20 / 51 of 22 / 54 new flags
        # were geometry-free only there); Melbourne-Wuebbena and IGF, which have no time term, still count
        after_gap = np.zeros(m, bool)
        after_gap[1:] = np.diff(pos) > 1
        after_gap[1:] |= after_gap[:-1]
        fl &= ~(gfo & after_gap)
        gfo &= ~after_gap
        out["gf_only"][pos] = gfo
        # today's flags of the processed pair from the previous tested epoch (exclusive) to this one
        prev = np.r_[pos[0] - 1, pos[:-1]]
        base_c = (cbase[pos + 1] - cbase[prev + 1]) > 0
        near_base = _dilate(base_c, 1)
        unconf = np.zeros(m, bool)
        for j in np.flatnonzero(fl & ~near_base):
            if j == 0 or not fl[j]:
                continue
            eps = sorted({max(0, j - 1) + int(h) for g in gated
                          for h in np.flatnonzero(g[max(0, j - 1):j + 2])}) or [j]
            acc, dp, dc = _attribute(run, eps[0], eps)
            if acc and not any(dp) and dc:
                fl[max(0, j - 1):j + 2] = False
            elif _no_processed_slip(run, eps[0], eps):
                unconf[j] = True
        fl[0] = False
        hit = np.flatnonzero(fl)
        out["cross"][pos[hit]] = True
        out["unconfirmed"][pos[hit[unconf[hit]]]] = True
        for h in hit:  # a slip after a bridged gap may lie in the gap
            if pos[h] - pos[h - 1] > 1:
                out["cross"][pos[h - 1] + 1:pos[h]] = True
                out["unconfirmed"][pos[h - 1] + 1:pos[h]] = unconf[h]
        if dual:
            out["verified"][pos] = ver
    return out


def common_mode_events(epoch: np.ndarray, gf_only: np.ndarray, active: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """Per row: True at an ionospheric event epoch (-1 to +``CM_TAIL`` epochs) of the common-mode (flare) guard.

    ``epoch``: integer epoch index of each row; ``gf_only``/``active``: per row (``third_carrier_flags``);
    ``piece``: continuous-piece id of each row (rows of one piece in time order). An epoch is an event when
    the geometry-free-only cross flags (+-1 epoch) cover at least ``CM_MIN_SATS`` and at least ``CM_FRAC`` of
    the satellites with an active third-carrier run; the event mask covers one epoch before and ``CM_TAIL``
    epochs after each event epoch.
    """
    n = len(epoch)
    if n == 0 or not gf_only.any():
        return np.zeros(n, dtype=bool)
    # +-1 epoch within the piece
    g = gf_only.copy()
    prev_same = np.zeros(n, bool)
    prev_same[1:] = piece[1:] == piece[:-1]
    g[1:] |= gf_only[:-1] & prev_same[1:]
    g[:-1] |= gf_only[1:] & prev_same[1:]
    e0 = int(epoch.min())
    ne = int(epoch.max()) - e0 + 1
    hits = np.bincount(epoch - e0, weights=(g & active).astype(float), minlength=ne)
    tracked = np.bincount(epoch - e0, weights=active.astype(float), minlength=ne)
    ev = (hits >= CM_MIN_SATS) & (hits >= CM_FRAC * tracked)
    ev = _dilate(ev, 1)
    tail = ev.copy()
    for k in range(2, CM_TAIL + 1):
        tail[k:] |= ev[:-k]
    ev = tail
    if ev.any():
        log.info("cycle slips: %d epoch(s) of common-mode ionospheric events, geometry-free-only third-carrier "
                 "flags dropped there", int(ev.sum()))
    return ev[epoch - e0]


# ============================================================================ L2: geometry
#: geometry layer (L2) enable conditions besides precise orbits: the sampling interval [s] it is calibrated at (the
#: confirmation windows are in epochs) with its tolerance, and the plausible |a-priori position| [m] (one pair of
#: bounds with the observation header reader, ``rinex_obs.APPROX_XYZ_MIN_M/MAX_M``)
GEO_INTERVAL_S = 30.0
GEO_INTERVAL_TOL_S = 1.0
GEO_XYZ_MIN_M = rinex_obs.APPROX_XYZ_MIN_M
GEO_XYZ_MAX_M = rinex_obs.APPROX_XYZ_MAX_M
#: geometry layer with broadcast orbits (since v1.0.0), when the frames carry the broadcast record
#: of every row: a change of the record of a satellite (any of ``BROADCAST_RECORD_COLUMNS``) starts a new piece, as
#: a change of the clock source does, so the record switch (orbit and clock steps of cm to dm) is never tested as a
#: slip. ``GEO_BROADCAST_TEC``: the TEC session (``ionosphere.config``), where the epochs the pair tests cannot
#: verify are dropped; basis (broadcast TEC runs at BRUX / NKLG / NYA2 on 2024-035 and 2024-132, GPS L1/L2,
#: L2/L5, Galileo E1/E5a, BeiDou B1I/B3I, against the layer off): satellite-epochs lost 0.00-4.01 -> 0.00-0.87 %,
#: NYA2 GPS L1/L2 2024-035 4.01 -> 0.39 % (precise orbits 0.30 %), and of its epochs with |ROT| > 2 TECU/min 14.6 ->
#: 2.0 % (precise 2.0 %); new arcs per satellite-hour at most 0.017 above the precise-orbit run of the same data
#: (NYA2 L2/L5 0.089 -> 0.133, precise 0.119); with the record switches inside the pieces (an earlier prototype)
#: NYA2 L1/L2 0.105 -> 0.279. ``GEO_BROADCAST``: the PPP sessions and other callers (False, as before
#: v1.0.0): a PPP run drops no epoch without the layer (the BRUX 2024-035 broadcast rows keep the same
#: satellite-epochs either way), and with it the only change on the validation data is one or two new GPS arcs at non-dispersive
#: range steps of G20 (03:03) and G07 (05:05) that are no cycle slips (-8.7 / +10 cm, equal on L1 and L2; the
#: precise-orbit runs do not break there): G kinematic p95 1.23 -> 1.30 m, GC kinematic 0.71 -> 0.82 m, static
#: -1.1..+2.0 cm
GEO_BROADCAST = False
GEO_BROADCAST_TEC = True
#: frame columns that identify the broadcast record of a row (the selector's merge: GPS IODE, Galileo IODnav, BeiDou
#: AODE / AODC, the clock and orbit reference epochs of every system)
BROADCAST_RECORD_COLUMNS = ("Toe", "nav_toc", "IODE", "IODnav", "IODC", "AODE", "AODC")
#: a-priori position check: robust least squares of ionosphere-free code residuals (codes minus the frame's OSBs,
#: pairs amplifying the code noise at most APRIORI_MAX_AMP times) over at most APRIORI_EPOCHS epochs spread over the
#: run (3 position unknowns, one clock per epoch and system). The model is re-linearised once at the estimate when
#: |dx| > max(APRIORI_RELIN_M, APRIORI_RELIN_SIG sigma_dx); a static run has the layer off when |dx| > APRIORI_OFF_M
#: and > APRIORI_OFF_SIG sigma_dx. Basis (NYA2 2024-035 L2/L5): a static error of 1 / 10 / 100 m
#: gives (1, 1, 1) 98.8 / 97.9 / 90.6 % and 0 / 0 / 14.5 % breaks from slips on the third carrier alone; the code
#: estimate itself is 0.6-1.4 m off the IGS coordinates of BRUX / NYA2, so a correction
#: below 5 m would not bring the model closer
APRIORI_EPOCHS = 100
APRIORI_RELIN_M = 5.0
APRIORI_RELIN_SIG = 4.0
APRIORI_OFF_M = 100.0
APRIORI_OFF_SIG = 5.0
APRIORI_MAX_AMP = 5.0
#: kinematic runs: the receiver counts as static when the per-epoch position change from the ionosphere-free phase
#: epoch differences (:func:`_epoch_motion`) exceeds max(KIN_MOVE_M, min(KIN_MOVE_K sigma, KIN_MOVE_MAX_M)) at no
#: more than KIN_MOVE_SHARE of the epochs and the static model (no position change, run first) fits: in every
#: window of KIN_WINDOW epochs (with at least KIN_WINDOW_MIN_SATS testable satellites per epoch on average) T1
#: tests at least KIN_MIN_TESTED of the testable rows and flags at most KIN_MAX_FLAGGED of the tested ones. A
#: static receiver keeps the static model. Otherwise it moves: with KIN_RELINEARISE the model is re-linearised
#: causally at the tracked position (a-priori position plus the integrated position change, restarted from the
#: code where it is lost) and T1 runs again; without it, or when the position cannot be tracked, the geometry
#: layer is off for the run (motion guard). Basis (library bench and PPP frames, 30 s): static receivers
#: 0-1.0 % of the epochs above the limit with an ionosphere-free pair, 0-2.9 % on single frequency without a second
#: band (NKLG 2024-131), worst windows tested 100 % and flagged <= 5.5 %; a 1 mm/s drift 0.3-0.5 % (the static
#: model keeps it: no new ambiguity); 3 mm/s 8.8 %, a 20 m circle (1 m per epoch), a 2 km circle and 1 m/s 100 %
KIN_MOVE_M = 0.1
KIN_MOVE_K = 5.0
KIN_MOVE_MAX_M = 1.0
KIN_MOVE_SHARE = 0.05
KIN_WINDOW = 20
KIN_WINDOW_MIN_SATS = 4
KIN_MIN_TESTED = 0.9
KIN_MAX_FLAGGED = 0.1
KIN_RELINEARISE = True
#: tracking of a moving receiver: below this share of the testable rows tested the position counts as lost
KIN_TRACK_MIN_TESTED = 0.5
#: tracking of a moving receiver: restarts from the code (epochs whose position change is not estimable) allowed
#: before the motion guard switches the geometry layer off
KIN_MAX_RESTARTS = 10
#: tracking of a moving receiver: gain of the pull of the tracked position towards the code position of each epoch
KIN_CODE_GAIN = 0.05
#: confirmation of an unconfirmed flag (a flag of one layer only): level test of an ionosphere-free combination
#: of the clock-corrected epoch differences. Line fitted to the cumulated level over the CONF_PRE epochs before
#: the flag, median residual over the CONF_POST epochs from it; passes at >= CONF_K sigma and >= CONF_MIN_M.
#: Close pair without a third carrier: sigma_I <= CONF_SIGI_K2 or the IF(a, b) step >= CONF_K_K2 sigma and
#: >= CONF_MIN_M_K2; single frequency without a second band: sigma_I <= CONF_SIGI_K1. Basis (NYA2
#: 2024-035 L2/L5 grid): new ambiguities per satellite-hour 0.125 at 5 sigma / 5 cm, 0.118 from 6 sigma / 6 cm
#: to 10 sigma / 7 cm, (1, 1, 1) 94.7 % up to 7 sigma; held-out day 2024-131 confirms 7 sigma / 7 cm
CONF_PRE = 20
CONF_POST = 10
CONF_K = 7.0
CONF_MIN_M = 0.07
CONF_K_K2 = 5.0
CONF_MIN_M_K2 = 0.05
CONF_SIGI_K2 = 0.02
CONF_SIGI_K1 = 0.03
#: unconfirmed flags of the wide pairs (L1/L2, E1/E5a, B1I/B3I, ...) pass the level test of the processed pair's
#: IF(a, b): T1 flags at >= CONF_K_W sigma and >= CONF_MIN_M_W (CONF_WIDE), third-carrier flags at
#: >= CONF_MIN_M_W (CONF_WIDE_L1: a slip on the third carrier alone leaves IF(a, b) unchanged). Two separate
#: switches. Basis of CONF_WIDE (library bench, L1/L2, seven station-days, 4348 (1, 1) slips): (1, 1)
#: 99.52 % with it, 99.54 % without (per day 98.7-100 against 99.0-100, within the +-0.45 point binomial spread
#: of one day); new ambiguities per satellite-hour 0-0.024 fewer with it (NYA2 2024-035 0.109 against 0.133)
CONF_WIDE = True
CONF_WIDE_L1 = True
CONF_K_W = 5.0
CONF_MIN_M_W = 0.05
#: close pairs: the third carrier joins T1 (K = 3) and T2 is off
CLOSE_PAIRS = ("L2L5", "E5aE5b", "B2aB2b")
#: carrier slots of the geometry arrays: 0 and 1 the processed signals (1 empty for single frequency), 2 the third
SLOT_C = 2
#: satellite categories of the geometry layer
CAT_K3, CAT_H, CAT_W, CAT_SF2, CAT_SF1 = 1, 2, 3, 4, 5
_F_REF = 1575.42e6
#: clock files (clk_path) covering less than this share of the rows: WARNING (the SP3 clocks are used for the rest)
CLK_COVER_WARN = rinex_clk.CLK_COVER_WARN


def _k_to_chi2(k: float, dof) -> np.ndarray:
    """chi2(dof) quantile with the false-alarm rate of a two-sided k-sigma normal test."""
    from scipy.stats import chi2, norm

    return chi2.isf(2 * norm.sf(k), dof)


def _unit_vectors(v: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``v / rho[..., None]`` without a floating-point exception, element for element equal to the plain division
    (explicit masks instead of ``np.errstate``): NaN for 0 / 0, inf / inf and a NaN operand,
    a signed inf for a non-zero component over a zero range (a vector whose norm underflows), the quotient
    elsewhere."""
    r = np.broadcast_to(rho[..., None], v.shape)
    ok = (r > 0) & ~(np.isinf(v) & np.isinf(r))
    out = np.divide(v, r, out=np.full(v.shape, np.nan), where=ok)
    zero = (r == 0) & (v != 0) & ~np.isnan(v)
    out[zero] = np.copysign(np.inf, v[zero])
    return out


def _nanmedian_rows(a: np.ndarray) -> np.ndarray:
    """Median of the finite values of each row of a short 2-D array (NaN for a row without one)."""
    srt = np.sort(a, axis=1)  # NaN last
    k = np.isfinite(a).sum(axis=1)
    lo = np.clip((k - 1) // 2, 0, a.shape[1] - 1)
    hi = np.clip(k // 2, 0, a.shape[1] - 1)
    r = np.arange(len(a))
    return np.where(k > 0, 0.5 * (srt[r, lo] + srt[r, hi]), np.nan)


def _rolling_median(a: np.ndarray, half: int, min_periods: int) -> np.ndarray:
    """Centred rolling median over +-``half`` rows of every column of ``a`` (NaN ignored)."""
    return pd.DataFrame(a).rolling(2 * half + 1, center=True, min_periods=min_periods).median().to_numpy()


def _pieces(ok: np.ndarray, brk: np.ndarray, chain: Optional[np.ndarray] = None) -> list:
    """[a, b) ranges of consecutive ``ok`` epochs without a break (and within one clock chain)."""
    n = len(ok)
    cut = ~ok | brk
    if chain is not None:
        cut = cut | np.r_[False, chain[1:] != chain[:-1]]
    out = []
    a = None
    for t in range(n):
        if not ok[t]:
            if a is not None:
                out.append((a, t))
                a = None
            continue
        if a is None:
            a = t
        elif cut[t]:
            out.append((a, t))
            a = t
    if a is not None:
        out.append((a, n))
    return out


def _integer_candidate(fl: np.ndarray, q_cyc: np.ndarray, search: int = 3):
    """Integer vector nearest to the float slip vector ``fl`` [cycles] with covariance ``q_cyc`` (search +-``search``
    around the rounding): (best vector, ratio second-best / best distance, best distance)."""
    k = len(fl)
    qi = np.linalg.inv(q_cyc)
    base = np.round(fl).astype(int)
    grid = np.array(np.meshgrid(*[range(-search, search + 1)] * k, indexing="ij")).reshape(k, -1).T + base
    dv = grid - fl
    dist = np.einsum("ni,ij,nj->n", dv, qi, dv)
    o = np.argsort(dist)
    return grid[o[0]], dist[o[1]] / max(dist[o[0]], 1e-12), dist[o[0]]


class GeometrySlipDetector:
    """Geometry-based, ionosphere-weighted slip test of all satellites of a receiver (L2).

    Banville & Langley (2010, Navigation 57(4):325-334; 2013, J Geod 87:179-193) style, on the frame of the
    detector. The epoch-differenced geometry residual of carrier k of a satellite,
    d_k = Delta(L_k - lambda_k wu - model), is

        d_k = Delta dtr (- u . Delta x) + Delta G + mu_k Delta I + lambda_k Delta N_k + e_k,   mu_k = (f_L1 / f_k)^2

    with one receiver-clock change Delta dtr for every satellite of every system (the inter-system bias does not
    change over one epoch), a position change Delta x when kinematic, a non-dispersive change Delta G (orbit /
    clock product error, troposphere) constrained to the median of its recent values and an ionosphere change
    Delta I constrained to the previous epoch's estimate. sigma_I and sigma_G come from the satellite's recent
    innovations (first pass causal, second pass centred +-``SIG_HALF`` epochs).

    * **T1** (one epoch): generalised least squares of the clock (and position) change with iterative data
      snooping; per satellite T = v' Q^-1 v ~ chi2(K) divided by a variance factor of its own T series (centred
      median over +-``SCALE_HALF`` epochs), tested at the false-alarm rate of a two-sided ``K_T1`` sigma test. Two
      flags at t and t + 1 whose float vectors cancel are an outlier at t; any other flag is a slip. A carrier
      missing at an epoch (or with a loss-of-lock indicator there) gets variance ``BIG`` and leaves the test.
    * **Attribution**: a flag whose integer slip vector (ratio >= ``ATT_RATIO``, distance <= chi2_K(0.999)) is
      on the third carrier only is dropped (and vetoes the third-carrier cross flags around it).
    * **T2** (windowed; wide pairs only - close pairs without a third carrier use rule H of
      :func:`confirm_flags`, measured without T2): the clock-corrected undifferenced
      ionosphere-free residual tested for level shifts by ``tools._level_shifts``.

    ``freq`` [n_sat, 3]: carrier frequencies of the slots (NaN: none); ``t1_slots`` [n_sat, 3]: slots tested in T1.
    """

    #: epoch-differenced phase noise of one carrier at zenith [m], elevation 1 / sin
    SIG_PHI_TD = 0.0015
    #: non-dispersive change: floor and start value [m]
    SIG_GEO_TD = 0.002
    SIG_G_START = 0.05
    #: ionosphere change (L1 metres): floor and start value
    SIG_I_MIN = 0.002
    SIG_I_START = 0.30
    #: inflation of the innovation sigmas; values remembered; half-widths of the centred sigma map and of the
    #: variance factor [epochs]
    SIG_INFL = 1.5
    HIST = 10
    SIG_HALF = 10
    SCALE_HALF = 20
    #: T1: k sigma (chi2 equivalent); T2: k sigma, window [epochs], floor (fraction of the narrow-lane wavelength)
    K_T1 = 7.0
    K_T2 = 5.0
    T2_WINDOW = 5
    T2_FLOOR_FRAC = 0.5
    #: variance [m^2] of a carrier missing at an epoch
    BIG = 1.0e4
    #: attribution of a flag to a slip on the third carrier alone
    ATT_RATIO = 3.0
    ATT_ALPHA = 1e-3
    ATT_SEARCH = 3

    def __init__(self, freq, t1_slots, kinematic: bool = False):
        from scipy.stats import chi2

        self.freq = np.asarray(freq, dtype=float)
        self.has = np.isfinite(self.freq)
        self.t1 = np.asarray(t1_slots, dtype=bool) & self.has
        f = np.where(self.has, self.freq, _F_REF)
        self.lam = np.where(self.has, CLIGHT / f, np.nan)
        self.mu = np.where(self.has, (_F_REF / f) ** 2, 0.0)
        self.n_sat, self.K = self.freq.shape
        self.kin = bool(kinematic)
        dof = np.arange(1, self.K + 1)
        self.thr_tab = np.r_[np.inf, _k_to_chi2(self.K_T1, dof)]
        self.med_tab = np.r_[np.nan, chi2.median(dof)]

    def _sphi(self, el):
        return self.SIG_PHI_TD / np.maximum(np.sin(np.radians(el)), 0.1)

    # ------------------------------------------------------------------ T1
    def _pass(self, d, av, el, ok, brk, los=None, sig_map=None, track=None) -> dict:
        """One T1 pass over the epoch differences ``d`` [n_ep, n_sat, K] (``av``: slot available at the epoch).

        ``track`` (kinematic, a moving receiver): the epoch differences are formed here, causally, at the position
        tracked through the run - the model of epoch t at the predicted position (previous position plus the
        previous change), of epoch t - 1 at the previous position - so the line-of-sight error stays at the
        tracking error (:func:`_track_diff`); the position is (re)started from the code (:func:`_code_fix`) at the
        start and after an epoch without a solution."""
        n_ep, n_sat = el.shape
        K = self.K
        tdok = np.zeros((n_ep, n_sat), dtype=bool)
        tdok[1:] = ok[1:] & ok[:-1] & ~brk[1:]
        T = np.full((n_ep, n_sat), np.nan)
        dof = np.zeros((n_ep, n_sat), dtype=np.int64)
        clk = np.full(n_ep, np.nan)
        dx = np.full((n_ep, 3), np.nan)
        fl = np.full((n_ep, n_sat, K), np.nan)
        dres = np.full((n_ep, n_sat, K), np.nan)
        mdb1 = np.full((n_ep, n_sat), np.nan)
        inn_i = np.full((n_ep, n_sat), np.nan)
        inn_g = np.full((n_ep, n_sat), np.nan)
        sig_i = np.full((n_ep, n_sat), np.nan)
        sig_g = np.full((n_ep, n_sat), np.nan)
        flag = np.zeros((n_ep, n_sat), dtype=bool)
        di_last = np.full(n_sat, np.nan)
        H_ = self.HIST
        b_g = np.full((n_sat, H_), np.nan)  # recent non-dispersive changes
        b_i = np.full((n_sat, H_), np.nan)  # recent ionosphere innovations
        b_j = np.full((n_sat, H_), np.nan)  # recent non-dispersive innovations
        n_g = np.zeros(n_sat, dtype=np.int64)
        n_i = np.zeros(n_sat, dtype=np.int64)
        n_j = np.zeros(n_sat, dtype=np.int64)
        live = np.zeros(n_sat, dtype=bool)
        p = 4 if self.kin else 1
        idx = np.arange(K)
        if track is not None:
            xhat = np.asarray(track["x0"], dtype=float).copy()
            vel = np.zeros(3)
            need_fix = True
            x_track = np.full((n_ep, 3), np.nan)
            n_fix = 0
        for t in range(1, n_ep):
            rst = ~tdok[t] & live
            if rst.any():
                di_last[rst] = np.nan
                b_g[rst] = b_i[rst] = b_j[rst] = np.nan
                n_g[rst] = n_i[rst] = n_j[rst] = 0
                live[rst] = False
            sv = np.flatnonzero(tdok[t])
            if track is not None:
                if need_fix:
                    xf = _code_fix(track, t - 1, xhat)
                    if xf is None:
                        continue
                    xhat, vel, need_fix = xf, np.zeros(3), False
                    x_track[t - 1] = xhat
                    n_fix += 1
                    # the code position is metres off: every satellite starts again from the loose priors
                    di_last[:] = np.nan
                    b_g[:] = b_i[:] = b_j[:] = np.nan
                    n_g[:] = n_i[:] = n_j[:] = 0
                    live[:] = False
                x_pred = xhat + vel
                d_t, los_t = self._track_diff(track, t, x_pred, xhat)
                av_t = self.t1 & np.isfinite(d_t) & track["avail"][t]
            if len(sv) < p + 2:
                if track is not None:
                    need_fix = True
                continue
            if track is not None:
                d_all, a_, los_sv = d_t[sv], av_t[sv], los_t[sv]
            else:
                d_all, a_ = d[t, sv], av[t, sv]
                los_sv = los[t, sv] if self.kin else None
            d_sv = np.where(self.t1[sv], d_all, np.nan)
            one = a_.astype(float)
            mu = self.mu[sv] * one
            lam = self.lam[sv]
            has_i = np.isfinite(di_last[sv])
            has_g = n_g[sv] >= 3
            p_i = np.where(has_i, di_last[sv], 0.0)
            p_g = np.where(has_g, _nanmedian_rows(b_g[sv]), 0.0)
            c_i = self.SIG_INFL * 1.4826 * _nanmedian_rows(np.abs(b_i[sv]))
            c_g = self.SIG_INFL * 1.4826 * _nanmedian_rows(np.abs(b_j[sv]))
            if sig_map is not None:
                s_i = np.where(np.isfinite(sig_map["I"][t, sv]), sig_map["I"][t, sv], self.SIG_I_START)
                s_g = np.where(np.isfinite(sig_map["G"][t, sv]), sig_map["G"][t, sv], self.SIG_G_START)
            else:
                s_i = np.where(n_i[sv] >= 3, np.maximum(self.SIG_I_MIN, c_i), self.SIG_I_START)
                s_g = np.where(n_j[sv] >= 3, np.maximum(self.SIG_GEO_TD, c_g), self.SIG_G_START)
            s_i = np.where(has_i, s_i, self.SIG_I_START)
            s_g = np.where(has_g, s_g, self.SIG_G_START)
            z = np.where(a_, np.nan_to_num(d_sv - p_i[:, None] * mu - p_g[:, None] * one), 0.0)
            sp = self._sphi(el[t, sv])
            Q = np.zeros((len(sv), K, K))
            Q[:, idx, idx] = np.where(a_, (sp ** 2)[:, None], self.BIG)
            Q += (s_g ** 2)[:, None, None] * np.einsum("ni,nj->nij", one, one)
            Q += (s_i ** 2)[:, None, None] * np.einsum("ni,nj->nij", mu, mu)
            Qi = np.linalg.inv(Q)
            Qi1 = np.einsum("nij,nj->ni", Qi, one)
            w = np.einsum("ni,ni->n", Qi1, one)
            k_s = a_.sum(axis=1)
            # a satellite of the epoch list without an available slot (the kinematic tracking path recomputes the
            # availability) has weight w = 0 and carries no information: it is left out of the epoch solution (13b-2
            # integration; its 0/0 made y, and through it the clock, the position change and every flag of the
            # epoch, NaN); with a slot w = one' Q^-1 one > 0 (Q positive definite)
            has_slot = k_s > 0
            y = np.zeros(len(sv))
            y[has_slot] = np.einsum("ni,ni->n", Qi1[has_slot], z[has_slot]) / w[has_slot]
            A = np.ones((len(sv), p))
            if self.kin:
                A[:, :3] = -los_sv
            thr = self.thr_tab[k_s]
            use = has_slot.copy()
            theta = None
            while use.sum() >= p + 2:
                Aw = A[use] * w[use, None]
                N = Aw.T @ A[use]
                if np.linalg.cond(N) > 1e8:
                    theta = None
                    break
                theta = np.linalg.solve(N, Aw.T @ y[use])
                v = z - (A @ theta)[:, None] * one
                Ts = np.einsum("ni,nij,nj->n", v, Qi, v) / thr
                Tu = np.where(use, Ts, -1.0)
                worst = int(np.argmax(Tu))
                if Tu[worst] > 1.0:
                    use[worst] = False
                    theta = None
                    continue
                break
            if theta is None:
                if track is not None:
                    need_fix = True
                continue
            clk[t] = theta[-1]
            dres[t, sv] = d_all - (A @ theta)[:, None]
            if track is not None:
                x_new = x_pred + theta[:3]
                dx[t] = vel = x_new - xhat
                # the integrated position drifts (stage 8c: up to 16 m a day on a static receiver): pulled towards
                # the code position of the epoch with the gain KIN_CODE_GAIN
                xc = _code_fix(track, t, x_new)
                if xc is not None:
                    x_new = x_new + KIN_CODE_GAIN * (xc - x_new)
                xhat = x_track[t] = x_new
            elif self.kin:
                dx[t] = theta[:3]
            v = z - (A @ theta)[:, None] * one
            Ts = np.einsum("ni,nij,nj->n", v, Qi, v)
            T[t, sv] = Ts
            dof[t, sv] = k_s
            flag[t, sv] = Ts > thr
            fl[t, sv] = np.where(a_, v / np.where(a_, lam, 1.0), np.nan)
            qkk = np.where(a_, np.einsum("nkk->nk", Qi), np.nan)
            # explicit masks: the MDB of every used slot (Qi is positive definite, so qkk > 0 there); NaN elsewhere.
            # A used slot with qkk == 0 keeps the value of the plain division (sqrt(thr / 0) = inf)
            thr_k, lam_k = np.broadcast_to(thr[:, None], qkk.shape), np.broadcast_to(lam, qkk.shape)
            mdb_ok = a_ & (qkk > 0)
            mdb_k = np.full(qkk.shape, np.nan)
            mdb_k[mdb_ok] = np.sqrt(thr_k[mdb_ok] / qkk[mdb_ok]) / lam_k[mdb_ok]
            q_zero = a_ & (qkk == 0)
            mdb_k[q_zero] = np.where(thr_k[q_zero] > 0, np.inf, np.nan) / lam_k[q_zero]
            # a satellite without a used slot keeps NaN (no all-NaN reduction)
            mdb_any = ~np.isnan(mdb_k).all(axis=1)
            mdb_row = np.full(len(sv), np.nan)
            mdb_row[mdb_any] = np.nanmax(mdb_k[mdb_any], axis=1)
            mdb1[t, sv] = mdb_row
            sig_i[t, sv] = s_i
            sig_g[t, sv] = s_g
            # posterior split of the residual into the non-dispersive (G) and ionospheric (I) change
            Ri = np.where(a_, 1.0 / (sp ** 2)[:, None], 0.0)
            H = np.stack([one, mu], axis=2)
            HRH = np.einsum("nki,nk,nkj->nij", H, Ri, H)
            HRv = np.einsum("nki,nk,nk->ni", H, Ri, v)
            HRH_g = HRH.copy()
            HRH_g[:, 0, 0] += 1.0 / s_g ** 2
            HRH_g[:, 1, 1] += 1.0 / s_i ** 2
            xh = np.linalg.solve(HRH_g, HRv[..., None])[..., 0]
            # innovations for the sigma maps: ionosphere with a loose prior (not shrunk by its own small sigma,
            # which would keep sigma_I small under bursts)
            HRH_l = HRH.copy()
            HRH_l[:, 0, 0] += 1.0 / s_g ** 2
            HRH_l[:, 1, 1] += 1.0 / self.SIG_I_START ** 2
            xl = np.linalg.solve(HRH_l, HRv[..., None])[..., 0]
            acc = use & ~flag[t, sv]
            a_s = sv[acc]
            live[a_s] = True
            m = acc & has_i
            s_ = sv[m]
            inn_i[t, s_] = xl[m, 1]
            b_i[s_, n_i[s_] % H_] = xl[m, 1]
            n_i[s_] += 1
            m = acc & has_g
            s_ = sv[m]
            inn_g[t, s_] = xh[m, 0]
            b_j[s_, n_j[s_] % H_] = xh[m, 0]
            n_j[s_] += 1
            di_last[a_s] = p_i[acc] + xh[acc, 1]
            b_g[a_s, n_g[a_s] % H_] = p_g[acc] + xh[acc, 0]
            n_g[a_s] += 1
        out = dict(T=T, dof=dof, flag1=flag, clk=clk, dx=dx, float=fl, mdb1=mdb1, innI=inn_i, innG=inn_g,
                   sigI=sig_i, sigG=sig_g, tdok=tdok, dres=dres)
        if track is not None:
            out.update(xhat=x_track, n_fix=n_fix)
        return out

    def _track_diff(self, track, t, x_pred, x_prev):
        """Epoch differences [n_sat, K] at epoch ``t`` with the model of t at ``x_pred`` and of t - 1 at ``x_prev``,
        and the unit vectors [n_sat, 3] at ``x_pred``."""
        lam0 = np.nan_to_num(self.lam)
        xs, rest, L, wu = track["xs"], track["rest"], track["L"], track["wu"]
        r1 = xs[t] - x_pred
        rho1 = np.linalg.norm(r1, axis=-1)
        rho0 = np.linalg.norm(xs[t - 1] - x_prev, axis=-1)
        g1 = L[t] - lam0 * wu[t][:, None] - (rho1 + rest[t])[:, None]
        g0 = L[t - 1] - lam0 * wu[t - 1][:, None] - (rho0 + rest[t - 1])[:, None]
        d = g1 - g0
        d[~self.has] = np.nan
        # unit vectors (NaN for a satellite without a position)
        los = _unit_vectors(r1, rho1)
        return d, los

    def _sig_maps(self, td) -> dict:
        """Centred robust sigmas of the first pass's innovations (+-``SIG_HALF`` epochs), for the second pass."""
        out = {}
        for key, floor in (("I", self.SIG_I_MIN), ("G", self.SIG_GEO_TD)):
            m = _rolling_median(np.abs(td["inn" + key]), self.SIG_HALF, 3)
            out[key] = np.maximum(floor, self.SIG_INFL * 1.4826 * m)
        return out

    def _scale(self, td) -> dict:
        """Variance factor of each satellite from its own T series (centred median / chi2 median, at least 1)."""
        dof = td["dof"]
        # no floating-point exception is possible here: T is finite or NaN, med_tab[0] is NaN and med_tab[k > 0]
        # > 0, f >= 1 and thr_tab[0] = inf (finite / inf = 0), thr_tab[k > 0] > 0
        tm = td["T"] / self.med_tab[dof]
        med = _rolling_median(tm, self.SCALE_HALF, 5)
        f = np.maximum(1.0, med)
        f = np.where(np.isfinite(f), f, 1.0)
        td["Tn"] = td["T"] / f / self.thr_tab[dof]
        td["f"] = f
        td["flag1"] = np.nan_to_num(td["Tn"], nan=0.0) > 1.0
        td["mdb1"] = td["mdb1"] * np.sqrt(f)
        return td

    @staticmethod
    def _t1_slips(td):
        """T1 flags -> (slips, outlier epochs): flags at t and t+1 whose float vectors cancel are a single
        deviating epoch t; any other flag is a slip at t."""
        f = td["flag1"]
        fl = td["float"]
        slips = f.copy()
        outl = np.zeros_like(f)
        for t, s in zip(*np.nonzero(f[:-1] & f[1:])):
            a, b = fl[t, s], fl[t + 1, s]
            m = np.isfinite(a) & np.isfinite(b)
            if np.all(np.abs(a[m] + b[m]) < 0.5 * np.maximum(np.abs(a[m]), np.abs(b[m])) + 0.15):
                outl[t, s] = True
                slips[t, s] = False
                slips[t + 1, s] = False
        return slips, outl

    def _attribute(self, td, slips, outl, el, av):
        """Flags whose integer slip vector is on the third carrier alone: dropped, returned as a mask."""
        from scipy.stats import chi2

        att = np.zeros_like(slips)
        for t, j in zip(*np.nonzero(slips | outl)):
            if not self.t1[j, SLOT_C]:
                continue
            m = av[t, j]
            if not m[SLOT_C] or m.sum() < 2:
                continue
            flt = td["float"][t, j][m]
            if not np.isfinite(flt).all() or not np.isfinite(td["sigI"][t, j]):
                continue
            sp = self._sphi(np.array([el[t, j]]))[0]
            mu = self.mu[j][m]
            one = np.ones(m.sum())
            Q = (np.eye(m.sum()) * sp ** 2 + td["sigG"][t, j] ** 2 * np.outer(one, one)
                 + td["sigI"][t, j] ** 2 * np.outer(mu, mu)) * td["f"][t, j]
            lam = self.lam[j][m]
            cand, ratio, dist = _integer_candidate(flt, Q / np.outer(lam, lam), self.ATT_SEARCH)
            full = np.zeros(self.K, dtype=int)
            full[np.flatnonzero(m)] = cand
            if (ratio >= self.ATT_RATIO and dist <= chi2.isf(self.ATT_ALPHA, int(m.sum()))
                    and not np.any(full[:SLOT_C]) and full[SLOT_C] != 0):
                slips[t, j] = outl[t, j] = False
                att[t, j] = True
        return att

    # ------------------------------------------------------------------ T2
    def _windowed(self, td, g, ok, brk, los, sats, kin_x=False):
        """T2 slips, outliers and verification of satellites ``sats`` (ionosphere-free of slots 0 and 1)."""
        from .tools import _level_shifts

        n_ep, n_sat = ok.shape
        clk = td["clk"]
        chain = np.cumsum(~np.isfinite(clk))
        cum = np.cumsum(np.nan_to_num(clk))
        G = g - cum[:, None, None]
        if kin_x:
            x = np.cumsum(np.nan_to_num(td["dx"]), axis=0)
            G = G + np.einsum("tsi,ti->ts", np.nan_to_num(los), x)[..., None]
        slip = np.zeros((n_ep, n_sat), dtype=bool)
        outl = np.zeros((n_ep, n_sat), dtype=bool)
        ver = np.zeros((n_ep, n_sat), dtype=bool)
        for s in sats:
            fa, fb = self.freq[s, 0], self.freq[s, 1]
            a = fa ** 2 / (fa ** 2 - fb ** 2)
            y = a * G[:, s, 0] + (1 - a) * G[:, s, 1]
            y[~ok[:, s]] = np.nan
            floor = self.T2_FLOOR_FRAC * CLIGHT / (fa + fb)
            sig1 = min(abs(a) * CLIGHT / fa, abs(1 - a) * CLIGHT / fb)
            for lo, hi in _pieces(ok[:, s], brk[:, s], chain):
                if hi - lo < 2 * self.T2_WINDOW:
                    continue
                f_, thr, o_ = _level_shifts(y[lo:hi], linear=True, k=self.K_T2, floor=floor, window=self.T2_WINDOW)
                slip[lo:hi, s] = f_
                outl[lo:hi, s] = o_
                ver[lo:hi, s] = np.isfinite(thr) & (thr <= sig1)
        return slip, outl, ver

    # ------------------------------------------------------------------ all
    def detect(self, L, model, wu, el, ok, brk, avail, los=None, t2_sats=(), track=None, model_fn=None):
        """Run T1 (two passes, variance factor), the attribution and T2.

        ``L`` [n_ep, n_sat, K] phases [m] (NaN: missing), ``model`` [n_ep, n_sat] modelled range without the
        receiver clock [m], ``wu`` [n_ep, n_sat] wind-up [cycles] (0 where already applied), ``el`` [deg],
        ``ok`` usable epochs (processed carriers and model present), ``brk`` a new continuous piece starts,
        ``avail`` [n_ep, n_sat, K] carrier usable at the epoch (no loss of lock), ``los`` [n_ep, n_sat, 3] unit
        vectors receiver -> satellite (kinematic), ``t2_sats`` satellites tested by T2; ``track``/``model_fn``: a
        moving receiver, tracked causally (:meth:`_pass`; ``model_fn(x [n_ep, 3])`` gives (model, los) at the
        tracked positions for the windowed tests and the confirmation). Returns a dict with
        ``slip``/``outl``/``att``/``ver``/``t2_slip``/``t2_outl`` [n_ep, n_sat], ``dd`` [n_ep, n_sat, K] the
        clock- (and position-) corrected epoch differences, ``sigI``, ``clk`` and ``dx``.
        """
        lam0 = np.nan_to_num(self.lam)

        def diffs(model_):
            g_ = np.asarray(L, dtype=float) - lam0[None] * np.asarray(wu, dtype=float)[..., None] - model_[..., None]
            g_[:, ~self.has] = np.nan
            d_ = np.full_like(g_, np.nan)
            d_[1:] = g_[1:] - g_[:-1]
            return g_, d_

        if track is None:
            g, d = diffs(model)
            av_all = np.isfinite(d) & np.asarray(avail, dtype=bool)
            av = av_all & self.t1[None]
            td1 = self._pass(d, av, el, ok, brk, los)
            td = self._pass(d, av, el, ok, brk, los, sig_map=self._sig_maps(td1))
            kin_x = self.kin
        else:
            td1 = self._pass(None, None, el, ok, brk, track=track)
            td = self._pass(None, None, el, ok, brk, sig_map=self._sig_maps(td1), track=track)
            xl = pd.DataFrame(td["xhat"]).ffill().bfill().to_numpy()
            if not np.isfinite(xl).all():
                xl = np.broadcast_to(np.asarray(track["x0"], dtype=float), xl.shape).copy()
            model, los = model_fn(xl)
            g, d = diffs(model)
            av_all = np.isfinite(d) & np.asarray(avail, dtype=bool)
            av = av_all & self.t1[None]
            kin_x = False
        td = self._scale(td)
        slips, outl = self._t1_slips(td)
        att = self._attribute(td, slips, outl, el, av)
        ver = np.isfinite(td["Tn"]) & (np.nan_to_num(td["mdb1"], nan=np.inf) <= 1.0)
        t2s, t2o, t2v = self._windowed(td, g, ok, brk, los, list(t2_sats), kin_x)
        # epoch differences minus the clock (and position) change of T1, every slot with data (the third carrier
        # of a wide pair too), for the confirmation
        dd = np.where(np.isfinite(td["dres"]) & np.asarray(avail, dtype=bool), td["dres"], np.nan)
        return dict(slip=slips, outl=outl, att=att, ver=ver | t2v, t2_slip=t2s, t2_outl=t2o, dd=dd,
                    sigI=td["sigI"], clk=td["clk"], dx=td["dx"], tested=np.isfinite(td["Tn"]),
                    n_fix=td.get("n_fix"))


# ---------------------------------------------------------------------------- 30 s clocks
# Reading and the analysis-centre check are gnx_py.rinex_clk's (bug-fix stage 10, contract K-CLK): one parser and
# one cache for the sessions and this layer, every compression format, the time system checked.
def usable_clk_paths(clk_path, sp3_path) -> tuple:
    """``(paths, reason)``: the 30 s clock files to use in the geometry layer, or ``(None, why not)``
    (:func:`gnx_py.rinex_clk.usable_clk_paths`: only clocks of the SP3's own analysis centre)."""
    from ..io.rinex_clk import usable_clk_paths as _usable

    return _usable(clk_path, sp3_path)


def read_clk_30s(paths) -> dict:
    """Satellite clocks of RINEX clock files (``AS`` records): ``{sv: (times [ns int64], clock [s])}``, the files
    merged (time-sorted, the first value of a repeated epoch kept).

    Read by :func:`gnx_py.rinex_clk.read_rinex_clk` (plain, gzip, .Z, bzip2 or zip; cached). A file that cannot be
    read (damaged archive, time system other than GPS) is left out with one WARNING; the SP3 clocks are then used
    where it would have applied."""
    from ..io.rinex_clk import read_rinex_clk

    out: dict = {}
    for p in paths:
        try:
            series = read_rinex_clk(p).series
        except (OSError, ValueError) as exc:
            _log_once(p, f"clock file left out, the SP3 clocks are used ({exc})", logging.WARNING)
            continue
        for sv, (t, v) in series.items():
            t = np.asarray(t, dtype=np.int64)
            v = np.asarray(v, dtype=float)
            if sv in out:
                t = np.r_[out[sv][0], t]
                v = np.r_[out[sv][1], v]
                o = np.argsort(t, kind="stable")
                t, v = t[o], v[o]
                keep = np.r_[True, np.diff(t) > 0]
                t, v = t[keep], v[keep]
            out[sv] = (t, v)
    return out


def _interp_clock(t_nodes: np.ndarray, c_nodes: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """Linear interpolation between clock nodes not more than twice their median spacing apart; NaN elsewhere."""
    out = np.full(len(tq), np.nan)
    if len(t_nodes) < 2:
        return out
    step = np.median(np.diff(t_nodes))
    k = np.searchsorted(t_nodes, tq)
    i1 = np.clip(k, 1, len(t_nodes) - 1)
    i0 = i1 - 1
    span = t_nodes[i1] - t_nodes[i0]
    ok = (tq >= t_nodes[i0]) & (tq <= t_nodes[i1]) & (span <= 2 * step) & np.isfinite(c_nodes[i0]) \
        & np.isfinite(c_nodes[i1])
    w = (tq - t_nodes[i0]) / np.where(span > 0, span, 1)
    out[ok] = c_nodes[i0][ok] + w[ok] * (c_nodes[i1][ok] - c_nodes[i0][ok])
    return out


# ---------------------------------------------------------------------------- frame adapter
@dataclass
class GeometryArrays:
    """Arrays [n_ep, n_sat(, 3)] of the detection frames of all systems for the geometry layer
    (:func:`frame_arrays`). Slots: 0 and 1 the processed signals, 2 the third carrier."""

    times: np.ndarray
    prn: np.ndarray
    system: np.ndarray
    cat: np.ndarray
    freq: np.ndarray
    t1: np.ndarray
    L: np.ndarray
    P: np.ndarray
    xs: np.ndarray
    rest: np.ndarray
    wu: np.ndarray
    el: np.ndarray
    ok: np.ndarray
    brk: np.ndarray
    avail: np.ndarray
    #: share of the rows (systems with ``clk_raw``) whose satellite clock comes from the 30 s clock files (None:
    #: no clock files)
    clk_cover: Optional[float] = None


def _frame_level(df: pd.DataFrame, name: str, pos: int) -> pd.Index:
    names = list(df.index.names)
    return df.index.get_level_values(name if name in names else pos)


def _record_columns(df: pd.DataFrame) -> list:
    """The ``BROADCAST_RECORD_COLUMNS`` of a broadcast frame (empty: the frame does not name its records; 'Toe' is
    required)."""
    cols = [c for c in BROADCAST_RECORD_COLUMNS if c in df.columns]
    return cols if "Toe" in cols else []


def _record_changes(df: pd.DataFrame, rows: np.ndarray, prn: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rows (indices into ``rows``) whose broadcast record differs from the previous row of the same satellite."""
    cols = _record_columns(df)
    out = np.zeros(len(rows), dtype=bool)
    if not cols or not len(rows):
        return out
    vals = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[rows] for c in cols])
    p = prn[rows]
    order = np.lexsort((t[rows], p))
    v = vals[order]
    same_sat = np.r_[False, p[order][1:] == p[order][:-1]]
    differ = np.r_[False, np.any(~((v[1:] == v[:-1]) | (np.isnan(v[1:]) & np.isnan(v[:-1]))), axis=1)]
    out[order] = same_sat & differ
    return out


def frame_arrays(frames_by_sys: dict, modes_by_sys: dict, aux_by_sys: Optional[dict] = None,
                 interval_s: Optional[float] = None, windup_applied: bool = False,
                 clocks: Optional[dict] = None, broadcast: bool = False) -> Optional[GeometryArrays]:
    """Geometry-layer arrays of the frames handed to the detector (all systems on one epoch grid).

    Frame columns used: the mode's code and phase columns [m] (codes minus ``OSB_<code>`` and ``STA_OSB_<code>``
    where present, as the PPP filters), ``xe``/``ye``/``ze`` (satellite at emission, rotated), ``clk`` (satellite
    clock + relativity [s]), ``tro`` (or ``tro_co``), ``dprel``, ``tides_los``, ``phw`` (wind-up [cycles]; not used
    where ``windup_applied``), ``ev`` [deg]; a missing correction counts 0. ``aux_by_sys``: side-read third carrier
    and LLI (:func:`read_aux_observations`); ``clocks``: 30 s clocks (:func:`read_clk_30s`) replacing the clock
    (``clk - clk_raw`` + clock at the emission time) where their nodes cover the row; elsewhere the frame's clock
    stays and a change of the source starts a new piece. ``GeometryArrays.clk_cover`` is the covered share.
    ``broadcast``: the rows come from broadcast records; a change of the record of a satellite (``_record_changes``)
    starts a new piece. BeiDou GEO satellites are left out.
    """
    from .gnss import is_bds_geo

    parts = []
    for system in ("G", "E", "C"):
        df = frames_by_sys.get(system)
        if df is None or len(df) == 0:
            continue
        prn = np.asarray(_frame_level(df, "sv", 0)).astype(str)
        t = pd.DatetimeIndex(pd.to_datetime(_frame_level(df, "time", 1))).as_unit("ns").asi8
        parts.append((system, df, prn, t))
    if not parts:
        return None
    times = np.unique(np.concatenate([p[3] for p in parts]))
    sats, sat_sys = [], []
    for system, _, prn, _ in parts:
        for s in sorted(set(prn)):
            if not is_bds_geo(s):
                sats.append(s)
                sat_sys.append(system)
    if not sats:
        return None
    s_index = {s: i for i, s in enumerate(sats)}
    n_ep, n_sat, K = len(times), len(sats), 3
    L = np.full((n_ep, n_sat, K), np.nan)
    P = np.full((n_ep, n_sat, K), np.nan)
    xs = np.full((n_ep, n_sat, 3), np.nan)
    rest = np.full((n_ep, n_sat), np.nan)
    wu = np.zeros((n_ep, n_sat))
    el = np.full((n_ep, n_sat), np.nan)
    lli = np.full((n_ep, n_sat, 2), np.nan)
    lli_c = np.full((n_ep, n_sat), np.nan)
    freq = np.full((n_sat, K), np.nan)
    t1 = np.zeros((n_sat, K), dtype=bool)
    cat = np.zeros(n_sat, dtype=np.int64)
    src = np.full((n_ep, n_sat), -1, dtype=np.int8)  # clock source of the row: 1 clock file, 0 frame (SP3)
    rec_brk = np.zeros((n_ep, n_sat), dtype=bool)  # broadcast record switches
    clk_rows = [0, 0]  # rows with a clock from the clock file, rows of the systems with clk_raw
    for system, df, prn, t in parts:
        mode = modes_by_sys[system]
        sigs = mode_signals(mode)
        aux = aux_signal(mode)
        rows = np.flatnonzero([p in s_index for p in prn])
        if not len(rows):
            continue
        e = np.searchsorted(times, t[rows])
        s = np.array([s_index[p] for p in prn[rows]])

        def col(name, default=0.0):
            if name in df.columns:
                return df[name].to_numpy(dtype=float)[rows]
            return np.full(len(rows), default)

        phases = processed_phase_columns(df, mode)
        codes = []
        for sig in sigs:
            pre = signal_spec(sig).code_prefix
            cc = [c for c in df.columns if str(c).startswith(pre) and not str(c).endswith(("lli", "ssi"))]
            codes.append(cc[0] if cc else None)
        for k, (ph, cd) in enumerate(zip(phases, codes)):
            if ph is not None:
                L[e, s, k] = df[ph].to_numpy(dtype=float)[rows]
                if f"{ph}lli" in df.columns:
                    lli[e, s, k] = df[f"{ph}lli"].to_numpy(dtype=float)[rows]
            if cd is not None:
                # code biases as the PPP filters (biases.osb_m): OSB_<code> + STA_OSB_<code> [ns] subtracted
                osb = sum(np.nan_to_num(col(f"{pre}{cd}")) for pre in ("OSB_", "STA_OSB_"))
                P[e, s, k] = df[cd].to_numpy(dtype=float)[rows] - osb * 1e-9 * CLIGHT
        xs[e, s, 0], xs[e, s, 1], xs[e, s, 2] = col("xe", np.nan), col("ye", np.nan), col("ze", np.nan)
        if broadcast:
            rec_brk[e, s] = _record_changes(df, rows, prn, t)
        clk = col("clk", np.nan)
        if clocks is not None and "clk_raw" in df.columns:
            t_em = col("t_em", 0.0)
            tq = t[rows] - np.rint(np.nan_to_num(t_em) * 1e9).astype(np.int64)
            rel = clk - col("clk_raw", np.nan)
            c30 = np.full(len(rows), np.nan)
            for sv in np.unique(prn[rows]):
                if sv in clocks:
                    m = prn[rows] == sv
                    c30[m] = _interp_clock(clocks[sv][0], clocks[sv][1], tq[m]) + rel[m]
            # where the clock file does not cover a row the frame's (SP3) clock stays; a change of the clock source
            # starts a new piece (a datum step of up to a few cm between the two)
            hit = np.isfinite(c30)
            clk = np.where(hit, c30, clk)
            src[e, s] = np.where(hit, 1, 0)
            clk_rows[0] += int(hit.sum())
            clk_rows[1] += len(rows)
        tro = col("tro") if "tro" in df.columns else col("tro_co")
        # tides_los = LOS . d with LOS from the satellite to the receiver: the range changes by +tides_los (the PPP
        # filters subtract it from the observations, as ah_los and the receiver PCO, and add the satellite PCO);
        # antenna offsets of the first processed signal for every carrier (the code check of the a-priori position
        # then estimates the marker, not the antenna phase centre)
        pco_r = col("pco_los_l1") if "pco_los_l1" in df.columns else col("pco_los")
        pco_s = col(f"sat_pco_los_{sigs[0]}")
        rest[e, s] = -CLIGHT * clk + tro + col("dprel") + col("tides_los") + col("ah_los") + pco_r - pco_s
        phw = col("phw")
        if not windup_applied:
            wu[e, s] = phw
        el[e, s] = col("ev", np.nan)
        fa = [frequency_hz(x) for x in sigs]
        sat_ids = np.unique(s)
        for k, f in enumerate(fa):
            freq[sat_ids, k] = f
        aux_df = (aux_by_sys or {}).get(system)
        has_aux = np.zeros(n_sat, dtype=bool)
        if aux_df is not None and len(aux_df):
            al = aux_df.reindex(pd.MultiIndex.from_arrays([prn[rows], pd.DatetimeIndex(t[rows])],
                                                          names=["sv", "time"]))
            for k, ph in enumerate(phases):
                if ph is not None and f"{ph}lli" in al.columns and f"{ph}lli" not in df.columns:
                    lli[e, s, k] = al[f"{ph}lli"].to_numpy(dtype=float)
            if aux is not None:
                fc = frequency_hz(aux)
                lc = al["aux_L"].to_numpy(dtype=float)
                if windup_applied:
                    lc = lc - CLIGHT / fc * phw
                L[e, s, SLOT_C] = lc
                pc = al["aux_P"].to_numpy(dtype=float)
                # the satellite OSB of the code read for each satellite, where the frame has it (PPP)
                code_sv = aux_df.attrs.get("aux_code_sv") or {}
                for sv_, code_ in code_sv.items():
                    if f"OSB_{code_}" in df.columns:
                        m = prn[rows] == sv_
                        if m.any():
                            pc[m] -= np.nan_to_num(col(f"OSB_{code_}")[m]) * 1e-9 * CLIGHT
                P[e, s, SLOT_C] = pc
                lli_c[e, s] = al["aux_lli"].to_numpy(dtype=float)
                fin = np.isfinite(lc)
                has_aux[np.unique(s[fin])] = True
                freq[has_aux, SLOT_C] = fc
        close = mode in CLOSE_PAIRS
        for j in sat_ids:
            if len(sigs) == 2:
                t1[j, :2] = True
                if close and has_aux[j]:
                    cat[j], t1[j, SLOT_C] = CAT_K3, True
                else:
                    cat[j] = CAT_H if close else CAT_W
            else:
                t1[j, 0] = True
                if has_aux[j]:
                    cat[j], t1[j, SLOT_C] = CAT_SF2, True
                else:
                    cat[j] = CAT_SF1
    single = ~np.isfinite(freq[:, 1])
    ok = np.isfinite(L[..., 0]) & (np.isfinite(L[..., 1]) | single[None]) & np.isfinite(rest) \
        & np.isfinite(xs).all(axis=-1) & np.isfinite(el) & np.isfinite(wu)
    brk = np.zeros((n_ep, n_sat), dtype=bool)
    brk[0] = True
    brk[1:] = ~ok[:-1]
    if interval_s and n_ep > 1:
        gap = np.r_[True, np.diff(times) > (float(interval_s) + 1.0) * 1e9]
        brk |= gap[:, None]
    brk |= rec_brk
    if clocks is not None and n_ep > 1:
        # clock source changes (clock file <-> SP3) between consecutive rows of a satellite
        for j in range(n_sat):
            at = np.flatnonzero(ok[:, j] & (src[:, j] >= 0))
            if len(at) > 1:
                ch = src[at[1:], j] != src[at[:-1], j]
                brk[at[1:][ch], j] = True
    avail = np.ones((n_ep, n_sat, K), dtype=bool)
    from .tools import CSDetector

    for j in range(n_sat):
        at = np.flatnonzero(ok[:, j])
        if len(at) and np.isfinite(lli[at, j]).any():
            brk[at, j] |= CSDetector._lli_slips(lli[at, j], len(at))
        atc = np.flatnonzero(np.isfinite(L[:, j, SLOT_C]))
        if len(atc):
            li = np.nan_to_num(lli_c[atc, j], nan=0.0).astype(np.int64)
            bad = (li & 1) > 0
            half = (li & 2) > 0
            bad[1:] |= half[1:] != half[:-1]
            avail[atc[bad], j, SLOT_C] = False
    return GeometryArrays(times=times, prn=np.array(sats), system=np.array(sat_sys), cat=cat, freq=freq, t1=t1,
                          L=L, P=P, xs=xs, rest=rest, wu=wu, el=el, ok=ok, brk=brk, avail=avail,
                          clk_cover=(clk_rows[0] / clk_rows[1]) if clocks is not None and clk_rows[1] else None)


def _model_at(arr: GeometryArrays, x_lin) -> tuple:
    """Modelled range without the receiver clock [m] and unit vectors receiver -> satellite, at ``x_lin`` ([3] or
    [n_ep, 3])."""
    x = np.asarray(x_lin, dtype=float)
    x = x[None, None, :] if x.ndim == 1 else x[:, None, :]
    v = arr.xs - x
    rho = np.linalg.norm(v, axis=-1)
    # unit vectors (NaN for a satellite-epoch without a position)
    los = _unit_vectors(v, rho)
    return rho + arr.rest, los


def _code_pairs(freq: np.ndarray, max_amp: Optional[float] = None) -> tuple:
    """Per satellite the code pair (slots a, b) and its coefficient alpha: the processed pair when it amplifies the
    code noise at most ``APRIORI_MAX_AMP`` times (its codes carry the frame's OSBs and antenna offsets), otherwise
    the pair of least ionosphere-free noise (a = -1: none - single frequency without a second band, or every pair
    amplifies the code noise more than ``max_amp``)."""
    n_sat = len(freq)
    pa, pb, pal = np.full(n_sat, -1), np.full(n_sat, -1), np.full(n_sat, np.nan)
    for j in range(n_sat):
        f = freq[j]
        best = None
        for a, b in ((0, 1), (0, SLOT_C), (1, SLOT_C)):
            if not (np.isfinite(f[a]) and np.isfinite(f[b])):
                continue
            al = f[a] ** 2 / (f[a] ** 2 - f[b] ** 2)
            amp = np.hypot(al, 1 - al)
            if max_amp is not None and amp > max_amp:
                continue
            if (a, b) == (0, 1) and amp <= APRIORI_MAX_AMP:
                best = (amp, a, b, al)
                break
            if best is None or amp < best[0]:
                best = (amp, a, b, al)
        if best is not None:
            _, pa[j], pb[j], pal[j] = best
    return pa, pb, pal


def _code_if(P: np.ndarray, freq: np.ndarray, max_amp: Optional[float] = None) -> np.ndarray:
    """Ionosphere-free code [..., n_sat] of the pairs of :func:`_code_pairs` (NaN where there is none)."""
    pa, pb, pal = _code_pairs(freq, max_amp)
    j = np.arange(len(freq))
    ok = pa >= 0
    out = np.full(P.shape[:-1], np.nan)
    a, b = np.where(ok, pa, 0), np.where(ok, pb, 0)
    val = pal * P[..., j, a] + (1 - pal) * P[..., j, b]
    out[..., ok] = val[..., ok]
    return out


def _code_fix(track: dict, t: int, x_guess) -> Optional[np.ndarray]:
    """Single-epoch position [m] of epoch ``t`` from the ionosphere-free code (one clock per system; at least
    four satellites more than systems; residuals above max(4 sigma, 5 m) rejected once), or None."""
    pif = track["pif"][t]
    ok = track["ok"][t] & np.isfinite(pif) & np.isfinite(track["rest"][t]) & np.isfinite(track["xs"][t]).all(-1)
    sv = np.flatnonzero(ok)
    sysc = track["sys_code"][sv]
    if len(sv) < 4 + len(np.unique(sysc)):
        return None
    x = np.asarray(x_guess, dtype=float).copy()
    keep = np.ones(len(sv), dtype=bool)
    for it in range(6):
        v = track["xs"][t, sv] - x
        rho = np.linalg.norm(v, axis=1)
        u = v / rho[:, None]
        r = pif[sv] - rho - track["rest"][t, sv]
        k = keep.copy()
        _, inv = np.unique(sysc[k], return_inverse=True)
        cnt = np.bincount(inv)
        rk = r[k] - (np.bincount(inv, weights=r[k]) / cnt)[inv]
        Ak = -u[k] - np.stack([np.bincount(inv, weights=-u[k][:, i]) / cnt for i in range(3)], axis=1)[inv]
        N = Ak.T @ Ak
        if k.sum() < 4 + len(cnt) or np.linalg.cond(N) > 1e8:
            return None
        dx = np.linalg.solve(N, Ak.T @ rk)
        x = x + dx
        if it == 2:
            e = rk - Ak @ dx
            sig = 1.4826 * np.median(np.abs(e - np.median(e)))
            new = keep.copy()
            new[np.flatnonzero(keep)] = np.abs(e) <= max(4.0 * sig, 5.0)
            keep = new
        if np.linalg.norm(dx) < 1e-3 and it > 2:
            break
    return x


def apriori_offset(arr: GeometryArrays, xyz) -> tuple:
    """Position correction [m] of the a-priori ``xyz`` from ionosphere-free code residuals, the observations used
    and the formal 3-D standard deviation of the correction [m] (robust residual scale times sqrt(trace N^-1)):
    robust least squares over at most ``APRIORI_EPOCHS`` epochs spread over the run, three position unknowns and one
    clock per epoch and system. Per satellite the code pair of least noise (processed pair, or a processed signal
    with the third carrier) that amplifies the code noise at most ``APRIORI_MAX_AMP`` times (a close pair alone,
    16.6 times on L2/L5, is not used); codes corrected by the frame's OSBs. (None, 0, nan) when there are too few
    observations."""
    model, los = _model_at(arr, xyz)
    res = _code_if(arr.P, arr.freq, APRIORI_MAX_AMP) - model
    res[~arr.ok] = np.nan
    good = np.flatnonzero(np.isfinite(res).sum(axis=1) >= 5)
    if len(good) < 5:
        return None, 0, float("nan")
    pick = good[np.unique(np.linspace(0, len(good) - 1, min(APRIORI_EPOCHS, len(good))).round().astype(int))]
    tt, ss = np.nonzero(np.isfinite(res[pick]))
    r = res[pick][tt, ss]
    A = -los[pick][tt, ss]
    sys_code = np.unique(arr.system, return_inverse=True)[1]
    grp = pick[tt] * 8 + sys_code[ss]
    keep = np.ones(len(r), dtype=bool)
    dx = np.zeros(3)
    sdx = float("nan")
    for _ in range(6):
        g_ids, inv = np.unique(grp[keep], return_inverse=True)
        cnt = np.bincount(inv)
        rk, Ak = r[keep], A[keep]
        rm = rk - (np.bincount(inv, weights=rk) / cnt)[inv]
        Am = Ak - np.stack([np.bincount(inv, weights=Ak[:, i]) / cnt for i in range(3)], axis=1)[inv]
        use = cnt[inv] >= 2
        if use.sum() < 10:
            return None, 0, float("nan")
        N = Am[use].T @ Am[use]
        if np.linalg.cond(N) > 1e8:
            return None, 0, float("nan")
        dx = np.linalg.solve(N, Am[use].T @ rm[use])
        e = rm - Am @ dx
        sig = 1.4826 * np.median(np.abs(e[use] - np.median(e[use])))
        sdx = float(sig * np.sqrt(np.trace(np.linalg.inv(N))))
        new = keep.copy()
        new[np.flatnonzero(keep)] = np.abs(e) <= max(4.0 * sig, 1.0)
        if (new == keep).all():
            break
        keep = new
    return dx, int(keep.sum()), sdx


@dataclass
class GeometryResult:
    """Geometry-layer flags of a receiver run on the epoch grid ``times`` [ns] and satellites ``prn``."""

    times: np.ndarray
    prn: np.ndarray
    cat: np.ndarray
    freq: np.ndarray
    out: dict

    def rows(self, prn, times_ns) -> dict:
        """Per row (satellite ``prn``, epoch ``times_ns``): the arrays of :meth:`GeometrySlipDetector.detect`
        (False / NaN where the satellite or epoch is not in the run), ``cat`` and ``freq`` per row."""
        index = {s: i for i, s in enumerate(self.prn)}
        s = np.array([index.get(p, -1) for p in prn], dtype=np.int64)
        e = np.searchsorted(self.times, times_ns)
        e = np.clip(e, 0, len(self.times) - 1)
        hit = (s >= 0) & (self.times[e] == times_ns)
        s0, e0 = np.where(hit, s, 0), np.where(hit, e, 0)
        rows = {}
        for k, v in self.out.items():
            a = v[e0, s0]
            if a.dtype == bool:
                a = a & (hit if a.ndim == 1 else hit[:, None])
            else:
                a = np.where(hit if a.ndim == 1 else hit[:, None], a, np.nan)
            rows[k] = a
        rows["cat"] = np.where(hit, self.cat[s0], 0)
        rows["freq"] = np.where(hit[:, None], self.freq[s0], np.nan)
        return rows


def _epoch_motion(arr: GeometryArrays, x0) -> tuple:
    """Per epoch the size of the receiver position change [m] since the previous epoch and its formal standard
    deviation [m], from the epoch differences of the ionosphere-free phase of each satellite (the pair of
    :func:`_code_pairs`; the first carrier alone for single frequency without a second band, weighted as an
    amplification of 10), model at ``x0``:
    weighted least squares (weight 1 / amplification^2) of three position unknowns and one clock change for all
    systems, residuals above max(4 sigma, 2 cm) dropped twice; NaN with fewer than 6 satellites. No state is carried
    between epochs."""
    model, los = _model_at(arr, x0)
    lam = np.nan_to_num(CLIGHT / arr.freq)
    g = arr.L - lam[None] * arr.wu[..., None] - model[..., None]
    pa, pb, pal = _code_pairs(arr.freq)
    j = np.arange(arr.L.shape[1])
    one = pa < 0
    a_, b_ = np.where(one, 0, pa), np.where(one, 0, pb)
    al = np.where(one, 1.0, pal)
    av = arr.avail[:, j, a_] & arr.avail[:, j, b_]
    y_all = al * g[:, j, a_] + (1 - al) * g[:, j, b_]
    y_all[~av] = np.nan
    # single frequency without a second band keeps the ionosphere change: weighted as an amplification of 10
    w_all = 1.0 / np.where(one, 10.0, np.hypot(pal, 1 - pal)) ** 2
    n_ep = len(g)
    out = np.full(n_ep, np.nan)
    sig_out = np.full(n_ep, np.nan)
    for t in range(1, n_ep):
        sv = np.flatnonzero(arr.ok[t] & arr.ok[t - 1] & ~arr.brk[t] & np.isfinite(y_all[t]) & np.isfinite(y_all[t - 1])
                            & np.isfinite(los[t]).all(axis=1))
        if len(sv) < 6:
            continue
        y = y_all[t, sv] - y_all[t - 1, sv]
        A = np.c_[-los[t, sv], np.ones(len(sv))]
        sw = np.sqrt(w_all[sv])
        use = np.ones(len(sv), dtype=bool)
        x = None
        for _ in range(3):
            if use.sum() < 6:
                x = None
                break
            x = np.linalg.lstsq(A[use] * sw[use, None], y[use] * sw[use], rcond=None)[0]
            r = (y - A @ x) * sw
            sig = max(1.4826 * np.median(np.abs(r[use] - np.median(r[use]))), 1e-3)
            new = np.abs(r) <= max(4.0 * sig, 0.02)
            if (new == use).all():
                break
            use = new
        if x is not None:
            Aw = A[use] * sw[use, None]
            out[t] = float(np.linalg.norm(x[:3]))
            sig_out[t] = float(sig * np.sqrt(np.trace(np.linalg.inv(Aw.T @ Aw)[:3, :3])))
    return out, sig_out


def _static_health(out: dict, arr: GeometryArrays) -> dict:
    """How well the T1 run ``out`` (:meth:`GeometrySlipDetector.detect`) fits: share of the testable rows (two
    consecutive usable epochs of one piece) that T1 tested and share of the tested rows it flagged (slip or outlier),
    over the run and in the worst window of ``KIN_WINDOW`` epochs with at least ``KIN_WINDOW_MIN_SATS`` testable
    satellites per epoch on average."""
    cand = np.zeros_like(arr.ok)
    cand[1:] = arr.ok[1:] & arr.ok[:-1] & ~arr.brk[1:]
    tested = np.asarray(out["tested"], dtype=bool) & cand
    flagged = (np.asarray(out["slip"], dtype=bool) | np.asarray(out["outl"], dtype=bool)) & tested
    nc, nt, nf = cand.sum(axis=1), tested.sum(axis=1), flagged.sum(axis=1)
    res = dict(tested=nt.sum() / max(nc.sum(), 1), flagged=nf.sum() / max(nt.sum(), 1), worst_tested=1.0,
               worst_flagged=0.0)
    k = KIN_WINDOW
    if len(nc) >= k:
        box = np.ones(k)
        wc, wt, wf = (np.convolve(a.astype(float), box, "valid") for a in (nc, nt, nf))
        m = wc >= k * KIN_WINDOW_MIN_SATS
        if m.any():
            res["worst_tested"] = float(np.min(wt[m] / wc[m]))
            res["worst_flagged"] = float(np.max(wf[m] / np.maximum(wt[m], 1.0)))
    return {key: float(v) for key, v in res.items()}


class _GeometryJob:
    """Geometry layer of one receiver run, computed on first use for all systems together."""

    def __init__(self, frames_by_sys, modes_by_sys, contexts, xyz, interval_s, windup_applied, kinematic,
                 clk_paths=None, broadcast: bool = False):
        self.frames, self.modes, self.contexts = frames_by_sys, modes_by_sys, contexts
        self.broadcast = bool(broadcast)
        self.xyz = np.asarray(xyz, dtype=float)
        self.interval_s, self.windup_applied, self.kinematic = interval_s, windup_applied, kinematic
        self.clk_paths = clk_paths
        self._result: Optional[GeometryResult] = None
        self._done = False
        self.info: dict = {}

    def result(self) -> Optional[GeometryResult]:
        if not self._done:
            self._done = True
            try:
                self._result = self._compute()
            except Exception as exc:  # noqa: BLE001 - the detector falls back to L0 + L1
                log.warning("cycle slips: geometry layer failed (%s: %s); third carrier and today's tests only",
                            type(exc).__name__, exc)
                self._result = None
        return self._result

    def _compute(self) -> Optional[GeometryResult]:
        import time as _time

        t0 = _time.time()
        aux = {s: c.aux_frame() for s, c in self.contexts.items()}
        clocks = read_clk_30s(self.clk_paths) if self.clk_paths else None
        arr = frame_arrays(self.frames, self.modes, aux, self.interval_s, self.windup_applied, clocks,
                           broadcast=self.broadcast)
        if arr is None or not arr.ok.any():
            log.info("cycle slips: geometry layer off (no row with the processed carriers, the satellite position "
                     "and clock); third carrier and today's tests")
            return None
        if arr.clk_cover is not None:
            self.info["clk_cover"] = arr.clk_cover
            if arr.clk_cover < CLK_COVER_WARN:
                log.warning("cycle slips: clk_path covers %.1f %% of the rows (clock files of another day?); the SP3 "
                            "clocks are used for the rest", 100.0 * arr.clk_cover)
            else:
                log.info("cycle slips: clk_path covers %.1f %% of the rows", 100.0 * arr.clk_cover)
        x0 = self.xyz.copy()
        dx, n_obs, sdx = apriori_offset(arr, x0)
        if dx is not None:
            nd = float(np.linalg.norm(dx))
            self.info.update(apriori_dx=dx, apriori_sig_m=sdx)
            if not self.kinematic and nd > APRIORI_OFF_M and nd > APRIORI_OFF_SIG * sdx:
                log.warning("cycle slips: a-priori position %.0f m (sigma %.1f m) from the code solution (%d "
                            "observations); geometry layer off", nd, sdx, n_obs)
                return None
            if nd > max(APRIORI_RELIN_M, APRIORI_RELIN_SIG * sdx):
                log.info("cycle slips: a-priori position %.1f m (sigma %.1f m) from the code solution (%d "
                         "observations); geometry model re-linearised there", nd, sdx, n_obs)
                x0 = x0 + dx
        # the static model first (kinematic runs too: a receiver that does not move keeps it)
        det = GeometrySlipDetector(arr.freq, arr.t1)
        t2_sats = np.flatnonzero(arr.cat == CAT_W)
        model, _ = _model_at(arr, x0)
        out = det.detect(arr.L, model, arr.wu, arr.el, arr.ok, arr.brk, arr.avail, t2_sats=t2_sats)
        if self.kinematic:
            h = _static_health(out, arr)
            mot, sig_mot = _epoch_motion(arr, x0)
            fin = np.isfinite(mot)
            moved = mot[fin] > np.maximum(KIN_MOVE_M, np.minimum(KIN_MOVE_K * sig_mot[fin], KIN_MOVE_MAX_M))
            h["move_share"] = float(np.mean(moved)) if fin.any() else 0.0
            h["move_median_m"] = float(np.median(mot[fin])) if fin.any() else float("nan")
            self.info.update(h)
            moving = (h["move_share"] > KIN_MOVE_SHARE or h["worst_tested"] < KIN_MIN_TESTED
                      or h["worst_flagged"] > KIN_MAX_FLAGGED)
            self.info["moving"] = moving
            if not moving:
                log.info("cycle slips: kinematic run, the receiver does not move (position change above %.2f m "
                         "and 5 sigma at %.1f %% of the epochs; static model: T1 tests %.1f %% of the rows, flags "
                         "%.2f %% of them); static geometry model", KIN_MOVE_M, 100 * h["move_share"],
                         100 * h["tested"], 100 * h["flagged"])
            elif not KIN_RELINEARISE:
                log.info("cycle slips: receiver moves (position change above %.2f m and 5 sigma at %.1f %% of the "
                         "epochs; static model: %.0f %% tested, %.0f %% flagged in its worst window); geometry layer "
                         "off for this run (motion guard)", KIN_MOVE_M, 100 * h["move_share"],
                         100 * h["worst_tested"], 100 * h["worst_flagged"])
                return None
            else:
                # a moving receiver: re-linearised causally at the tracked position (a-priori position plus the
                # integrated position change), restarted from the code where the position change is not estimable
                detk = GeometrySlipDetector(arr.freq, arr.t1, kinematic=True)
                track = dict(L=arr.L, wu=arr.wu, rest=arr.rest, xs=arr.xs, avail=arr.avail, ok=arr.ok, x0=x0,
                             pif=_code_if(arr.P, arr.freq),
                             sys_code=np.unique(arr.system, return_inverse=True)[1])
                out = detk.detect(arr.L, None, arr.wu, arr.el, arr.ok, arr.brk, arr.avail, t2_sats=t2_sats,
                                  track=track, model_fn=lambda x: _model_at(arr, x))
                n_fix = int(out["n_fix"] or 0)
                tested = _static_health(out, arr)["tested"]
                self.info.update(track_restarts=n_fix - 1, track_tested=tested)
                if n_fix == 0 or n_fix - 1 > KIN_MAX_RESTARTS or tested < KIN_TRACK_MIN_TESTED:
                    why = ("no single-epoch code position (no ionosphere-free code)" if n_fix == 0 else
                           f"position lost at {n_fix - 1} epochs, T1 tests {100 * tested:.0f} % of the rows")
                    log.info("cycle slips: receiver moves and cannot be tracked (%s); geometry layer off for this "
                             "run (motion guard)", why)
                    return None
                log.info("cycle slips: receiver moves; geometry model re-linearised at the tracked position "
                         "(%d restarts from the code)", n_fix - 1)
        keys = ("slip", "outl", "att", "ver", "t2_slip", "t2_outl", "dd", "sigI")
        self.info["runtime_s"] = _time.time() - t0
        log.info("cycle slips: geometry layer on: %d satellites, %d epochs (%.1f s)", arr.ok.any(axis=0).sum(),
                 len(arr.times), self.info["runtime_s"])
        return GeometryResult(times=arr.times, prn=arr.prn, cat=arr.cat, freq=arr.freq,
                              out={k: out[k] for k in keys})


# ---------------------------------------------------------------------------- confirmation and fusion
def level_step(z: np.ndarray, j: int):
    """Level test of the epoch differences ``z`` at position ``j``: (|step| [m], |step| / sigma), or None when the
    window ``j - CONF_PRE .. j + CONF_POST - 1`` is not complete and finite. A line is fitted to the cumulated
    level over the ``CONF_PRE`` epochs before ``j``; sigma = 1.4826 MAD of its residuals (at least 0.1 mm); the
    step is the median residual over the ``CONF_POST`` epochs from ``j``."""
    a, b = j - CONF_PRE, j + CONF_POST
    if a < 0 or b > len(z):
        return None
    w = z[a:b]
    if not np.isfinite(w).all():
        return None
    lev = np.cumsum(w)
    x = np.arange(len(w), dtype=float)
    cf = np.polyfit(x[:CONF_PRE], lev[:CONF_PRE], 1)
    r = lev - np.polyval(cf, x)
    pre = r[:CONF_PRE]
    sp = max(1.4826 * float(np.median(np.abs(pre - np.median(pre)))), 1e-4)
    step = abs(float(np.median(r[CONF_PRE:])))
    return step, step / sp


def _if_series(dd: np.ndarray, freq: np.ndarray, a: int, b: int):
    """Ionosphere-free combination of slots ``a`` and ``b`` of the epoch differences (rows of one satellite)."""
    fa, fb = freq[a], freq[b]
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    al = fa ** 2 / (fa ** 2 - fb ** 2)
    return al * dd[:, a] + (1 - al) * dd[:, b]


def confirm_flags(cand: np.ndarray, geo: dict, source: str = "T1") -> np.ndarray:
    """Unconfirmed flags ``cand`` (rows of one continuous piece of one satellite) that pass the confirmation of
    their satellite's category (``geo``: rows of :meth:`GeometryResult.rows`):

    * close pair or single frequency with the third carrier c: the level test (:func:`level_step`) of
      IF(c, x) for each processed x at >= ``CONF_K`` sigma and >= ``CONF_MIN_M``; kept when no window is complete;
    * wide pair (``CONF_WIDE``), IF(a, b) of the processed pair: a T1 flag (``source`` "T1") passes at
      >= ``CONF_K_W`` sigma and >= ``CONF_MIN_M_W``; a third-carrier flag ("L1") passes with a step of at least
      ``CONF_MIN_M_W`` (half the smallest slip signature, 10.7 cm for (1, 1) on L1/L2): a slip on the third
      carrier alone, which T1 of a wide pair does not see, leaves IF(a, b) unchanged;
    * close pair without the third carrier: sigma_I <= ``CONF_SIGI_K2`` or IF(a, b) >= ``CONF_K_K2`` sigma and
      >= ``CONF_MIN_M_K2``;
    * single frequency without a second band: sigma_I <= ``CONF_SIGI_K1``.
    """
    out = np.zeros(len(cand), dtype=bool)
    if not cand.any():
        return out
    dd, sig_i, cats, freq = geo["dd"], geo["sigI"], geo["cat"], geo["freq"]
    for j in np.flatnonzero(cand):
        cat = int(cats[j])
        f = freq[j]
        if cat == CAT_SF1:
            out[j] = bool(np.isfinite(sig_i[j]) and sig_i[j] <= CONF_SIGI_K1)
            continue
        if cat == CAT_H:
            z = _if_series(dd, f, 0, 1)
            st = level_step(z, j) if z is not None else None
            out[j] = bool((np.isfinite(sig_i[j]) and sig_i[j] <= CONF_SIGI_K2)
                          or (st is not None and st[0] >= CONF_MIN_M_K2 and st[1] >= CONF_K_K2))
            continue
        if cat == CAT_W:
            # the processed pair only: a slip on the third carrier alone (not in T1 of a wide pair, so not
            # attributed there) leaves IF(a, b) unchanged and its third-carrier flags are dropped
            z = _if_series(dd, f, 0, 1)
            use = CONF_WIDE if source == "T1" else CONF_WIDE_L1
            st = level_step(z, j) if (z is not None and use) else None
            out[j] = bool(st is None or (st[0] >= CONF_MIN_M_W and (source == "L1" or st[1] >= CONF_K_W)))
            continue
        combos = []
        if cat == CAT_K3:
            combos += [(SLOT_C, 0), (SLOT_C, 1)]
        elif cat == CAT_SF2:
            combos.append((SLOT_C, 0))
        stats = []
        for a, b in combos:
            z = _if_series(dd, f, a, b)
            st = level_step(z, j) if z is not None else None
            if st is not None:
                stats.append(st)
        out[j] = (not stats) or any(s >= CONF_MIN_M and k >= CONF_K for s, k in stats)
    return out


def fuse_flags(base, lli, outl, unver, l1=None, event=None, geo=None) -> tuple:
    """Fusion of the layers in one continuous piece of one satellite (``CSDetector.split_obs_cs``).

    ``base``/``outl``/``unver``: today's slips, phase outliers and unverifiable epochs of the mode; ``lli``: LLI
    slips of the processed phases; ``l1``: :func:`third_carrier_flags` (or None); ``event``: common-mode event rows;
    ``geo``: rows of :meth:`GeometryResult.rows` (None: geometry layer off). Returns (slip flags before isolation,
    epochs to isolate)::

        without geometry (L1n = L1 'unconfirmed', no processed integer slip):
        flags = base | (L1 cross & ~common-mode GF-only & ~(L1n & ~near(base | LLI | common-mode event)))
        isolate = outliers | (unverifiable & ~ver_L1)

        with geometry:
        L1x = L1 cross & ~veto(geometry attribution, +-1) & ~common-mode GF-only
        T1u = T1 slips & ~near(base | LLI | L1x)                  unconfirmed T1 flags
        L1u = L1x & ~(base | LLI) & ~near(T1 slips | T1 outliers)  unconfirmed L1 flags (with geometry)
        flags = base | (L1x & ~L1u) | confirm(L1u) | (T1 & ~T1u) | confirm(T1u) | T2 slips
        isolate = outliers | T1 outliers | T2 outliers | (unverifiable & ~ver_L1 & ~ver_L2)
    """
    base = np.asarray(base, dtype=bool)
    n = len(base)
    l1x = np.zeros(n, dtype=bool)
    ver1 = np.zeros(n, dtype=bool)
    if l1 is not None:
        gfo = l1["gf_only"] & (event if event is not None else np.zeros(n, dtype=bool))
        l1x = l1["cross"] & ~gfo
        ver1 = l1["verified"]
    if geo is None:
        if l1 is not None and "unconfirmed" in l1:
            # bug-fix stage 13b (ION-073): without the IF-level confirmation of the geometry layer, a third-carrier
            # flag that neither today's tests nor LLI raise (+-1 epoch) breaks the arc only on an integer slip of
            # the processed signals; a flag whose integer estimate excludes one (third_carrier_flags
            # 'unconfirmed') keeps the arc, as an L1u flag that the IF-level confirmation rejects does with the
            # geometry layer (and not at a common-mode ionospheric event, where the geometry-free statistic of the
            # estimate is disturbed)
            anchor = base | np.asarray(lli, dtype=bool)
            if event is not None:
                anchor = anchor | np.asarray(event, dtype=bool)
            l1x = l1x & ~(l1["unconfirmed"] & ~_dilate(anchor, 1))
        return base | l1x, outl | (unver & ~ver1)
    l1x = l1x & ~_dilate(geo["att"], 1)
    t1s, t1o = geo["slip"], geo["outl"]
    anchor = base | np.asarray(lli, dtype=bool)
    t1u = t1s & ~_dilate(anchor | l1x, 1)
    l1u = l1x & ~anchor & ~_dilate(t1s | t1o, 1)
    flags = (base | (l1x & ~l1u) | confirm_flags(l1u, geo, "L1") | (t1s & ~t1u) | confirm_flags(t1u, geo, "T1")
             | geo["t2_slip"])
    iso = outl | t1o | geo["t2_outl"] | (unver & ~ver1 & ~geo["ver"])
    return flags, iso
