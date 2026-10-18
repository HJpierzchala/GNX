"""Validation helpers for GNX-py configuration objects.

The configuration classes (``Config`` and its subclasses ``SPPConfig``,
``PPPConfig`` and ``TECConfig``) use these helpers to reject a setting that would
otherwise silently change the processing: a mistyped model name, a value outside
its physical range or a missing input file. Every error names the setting and the
offending value, in one of these forms::

    <setting> must be one of ['a', 'b'] (got 'X'); did you mean 'a'?
    <setting> must be > 0 (got -1)
    <setting> must be in [0, 90] deg (got 95)
    <setting>: file not found: <path>

Values are never normalised silently: a wrong-case model name is rejected with a
"did you mean" hint instead of being lower-cased.
"""
from __future__ import annotations

import difflib
import math
import numbers
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "did_you_mean",
    "check_choice",
    "check_number",
    "check_flag",
    "as_path_list",
    "iter_paths",
    "check_paths_exist",
]


def _fmt_choices(choices: Sequence[Any]) -> str:
    return "[" + ", ".join(repr(c) for c in choices) + "]"


def _same(a: Any, b: Any) -> bool:
    """Strict equality: ``0``/``1`` are not ``False``/``True`` and ``True`` is not ``'los'``."""
    a, b = _plain_bool(a), _plain_bool(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, str) and isinstance(b, str):
        # str subclasses (numpy.str_, a (str, Enum) member) compare by their text, case-sensitively
        return str.__eq__(a, b) is True
    try:
        return type(a) is type(b) and a == b
    except Exception:  # noqa: BLE001 - exotic objects never match a choice
        return False


def _plain_bool(value: Any) -> Any:
    """``numpy.bool_`` as ``bool``; any other value unchanged."""
    return bool(value) if isinstance(value, np.bool_) else value


def did_you_mean(value: Any, choices: Iterable[Any]) -> Optional[Any]:
    """Closest string choice to ``value`` (case-insensitive equality first, then a close spelling)."""
    if not isinstance(value, str):
        return None
    strs = [c for c in choices if isinstance(c, str)]
    low = value.strip().lower()
    for c in strs:
        if c.lower() == low:
            return c
    lowered = [c.lower() for c in strs]
    match = difflib.get_close_matches(low, lowered, n=1, cutoff=0.6)
    if match:
        return strs[lowered.index(match[0])]
    return None


def check_choice(
    name: str,
    value: Any,
    choices: Sequence[Any],
    *,
    notes: Optional[Mapping[str, str]] = None,
    suffix: str = "",
) -> Any:
    """Return ``value`` when it is one of ``choices`` (strict, case-sensitive); else ``ValueError``.

    ``notes`` maps a lower-case rejected value (or ``False``) to an extra explanation (for
    example ``{"nequick": "NeQuick is not implemented in GNX-py"}``); ``suffix`` is appended to
    the setting name in the message (e.g. ``" for system 'E'"``).
    """
    for c in choices:
        if _same(value, c):
            return value
    msg = f"{name} must be one of {_fmt_choices(choices)}{suffix} (got {value!r})"
    note = None
    if notes and isinstance(value, str):
        note = notes.get(value.strip().lower())
    elif notes and value is False:
        note = notes.get(False)
    if note:
        msg += f"; {note}"
    elif value is None and any(c is False for c in choices):
        msg += "; use False to disable it"
    else:
        hint = did_you_mean(value, choices)
        if hint is not None:
            msg += f"; did you mean {hint!r}?"
    raise ValueError(msg)


def _fmt_num(x: float) -> str:
    if isinstance(x, numbers.Integral):
        return str(int(x))
    return f"{float(x):g}"


def _range_text(ge, gt, le, lt) -> str:
    low = ("[", ge) if ge is not None else (("(", gt) if gt is not None else None)
    high = ("]", le) if le is not None else ((")", lt) if lt is not None else None)
    if low and high:
        return f"in {low[0]}{_fmt_num(low[1])}, {_fmt_num(high[1])}{high[0]}"
    if low:
        return f"{'>=' if low[0] == '[' else '>'} {_fmt_num(low[1])}"
    if high:
        return f"{'<=' if high[0] == ']' else '<'} {_fmt_num(high[1])}"
    return ""


def check_number(
    name: str,
    value: Any,
    *,
    ge: Optional[float] = None,
    gt: Optional[float] = None,
    le: Optional[float] = None,
    lt: Optional[float] = None,
    integer: bool = False,
    allow_none: bool = False,
    unit: str = "",
    finite: bool = False,
) -> Any:
    """Return ``value`` when it is a number within the given bounds; else ``ValueError``.

    ``integer=True`` requires an integral value (``30`` or ``30.0``); ``bool`` is never
    a number here and NaN is always rejected. ``finite=True`` also rejects ``±inf``
    (variances: an infinite value turns the filter covariance into NaN). ``None``
    passes only with ``allow_none``. The value is returned unchanged (no silent conversion).
    """
    if value is None and allow_none:
        return None
    rng = _range_text(ge, gt, le, lt)
    kind = "a finite number" if finite and not integer else ("an integer" if integer else "a number")
    unit_txt = f" {unit}" if unit else ""
    none_txt = " or None" if allow_none else ""

    def _fail() -> None:
        if rng:
            text = f"{name} must be {rng}{unit_txt}"
            if integer or finite or not _is_real(value):
                text = f"{name} must be {kind} {rng}{unit_txt}"
        else:
            text = f"{name} must be {kind}"
        raise ValueError(f"{text}{none_txt} (got {value!r})")

    if not _is_real(value):
        _fail()
    x = float(value)
    if math.isnan(x):
        _fail()
    if integer and not (math.isfinite(x) and x == math.floor(x)):
        _fail()
    if finite and math.isinf(x):
        _fail()
    if math.isinf(x) and ((x > 0 and (le is not None or lt is not None))
                          or (x < 0 and (ge is not None or gt is not None))):
        _fail()
    if ge is not None and not x >= ge:
        _fail()
    if gt is not None and not x > gt:
        _fail()
    if le is not None and not x <= le:
        _fail()
    if lt is not None and not x < lt:
        _fail()
    return value


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def check_flag(name: str, value: Any, *, allow_none: bool = False) -> Any:
    """Return ``value`` when it is ``True``/``False`` (also ``numpy.bool_`` and the integers 0/1); else ``ValueError``.

    A string such as ``'False'`` is rejected: it is truthy and would silently enable the option.
    """
    if value is None and allow_none:
        return None
    plain = _plain_bool(value)
    if isinstance(plain, bool):
        return value
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return value
    msg = f"{name} must be True or False{' or None' if allow_none else ''} (got {value!r})"
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "on", "off"):
        msg += f"; did you mean {value.strip().lower() in ('true', 'yes', 'on')}?"
    raise ValueError(msg)


def _is_path(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


def as_path_list(name: str, value: Any) -> Optional[list]:
    """A single path (``str``/``Path``) or a list/tuple/set of paths as a list; ``None`` stays ``None``.

    A set (unordered) is listed in path order. Raises ``TypeError`` naming the setting for
    anything else (for example a number or a list holding a non-path element).
    """
    if value is None:
        return None
    if _is_path(value):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        bad = [v for v in value if not _is_path(v)]
        if bad:
            raise TypeError(f"{name} must be a path or a list of paths (got element {bad[0]!r})")
        return sorted(value, key=os.fspath) if isinstance(value, (set, frozenset)) else list(value)
    raise TypeError(f"{name} must be a path or a list of paths (got {type(value).__name__} {value!r})")


def iter_paths(value: Any) -> list:
    """The paths held by a path setting (one path, a list/tuple/set of paths, or nothing)."""
    if value is None or value is False:
        return []
    if _is_path(value):
        return [value] if str(value) else []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None and str(v)]
    return [value]


def check_paths_exist(settings: Mapping[str, Any]) -> None:
    """``FileNotFoundError`` ``"<setting>: file not found: <path>"`` for the first missing input path."""
    for name, value in settings.items():
        for p in iter_paths(value):
            if not Path(p).expanduser().exists():
                raise FileNotFoundError(f"{name}: file not found: {p}")
