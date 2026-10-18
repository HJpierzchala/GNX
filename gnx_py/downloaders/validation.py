"""Input validation shared by ``DownloadConfig`` and the download engine.

One function per configuration parameter. ``DownloadConfig`` (its
construction, assignment checks and ``validate_for_run()``) and the low-level
``GNSSDownloadConfig`` use the same functions, so engine-direct users get the
same messages as session users.

Interface:

* every ``normalize_*`` function returns the normalized value in the shape
  stated in its docstring; every ``check_*`` function returns its argument
  when it is valid (numpy integers converted to ``int``);
* every error is a ``TypeError`` (wrong type) or ``ValueError`` (wrong value)
  whose message names the parameter and the offending item;
* nothing here performs I/O, logs credentials or touches the network.
* every function is idempotent: normalizing an already normalized value
  returns an equal value.

Besides these functions this module holds the friendly product-type
vocabulary (:data:`PRODUCT_TYPE_ALIASES`, re-exported by
:mod:`gnx_py.downloaders.project`), the station-id patterns and the
cross-parameter checks of ``DownloadConfig`` (credential styles and the
feasibility of the request).

Layering: this module imports :mod:`gnx_py.downloaders.core` and
:mod:`gnx_py.downloaders.catalog` at module level. ``core`` must therefore
import it only inside functions (e.g. in ``GNSSDownloadConfig.__post_init__``).
Values that other clusters extend (``SUPPORTED_DATA_TYPES``,
``SOURCE_CAPABILITIES``, ``catalog.NAV_PRODUCTS``) are read at call time.
"""
from __future__ import annotations

import logging
import math
import numbers
import os
import re
import unicodedata
from collections.abc import Iterable as _IterableABC
from collections.abc import Mapping as _MappingABC
from datetime import date, datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Iterable, Union

from . import catalog, core

LOGGER = logging.getLogger(__name__)

DateLike = Union[datetime, date]

__all__ = [
    "PRODUCT_TYPE_ALIASES",
    "STATION4_PATTERN",
    "STATION9_PATTERN",
    "check_adjacent_days",
    "check_bool",
    "check_credentials",
    "check_feasibility",
    "check_max_workers",
    "check_output_root",
    "check_retries",
    "check_timeout",
    "normalize_data_types",
    "normalize_dates",
    "normalize_product_preferences",
    "normalize_product_types",
    "normalize_project_name",
    "normalize_rinex_data_source_preference",
    "normalize_rinex_preference",
    "normalize_sources",
    "normalize_station_aliases",
    "normalize_stations",
]

#: Friendly product-type names (lower case) -> engine data-type keys.
#: Re-exported as ``gnx_py.downloaders.PRODUCT_TYPE_ALIASES``.
PRODUCT_TYPE_ALIASES: dict = {
    "obs": "obs",
    "observation": "obs",
    "observations": "obs",
    "sp3": "sp3",
    "orbit": "sp3",
    "orbits": "sp3",
    "precise_orbit": "sp3",
    "broadcast": "nav",
    "nav": "nav",
    "navigation": "nav",
    "brdc": "nav",
    "ionex": "ionex",
    "gim": "ionex",
    "tec": "ionex",
    "bia": "bia",
    "osb": "bia",
    "bias": "bia",
    "dcb": "dcb",
    "clk": "clk",
    "clock": "clk",
    "erp": "erp",
    "att": "att",
    "orbex": "att",
    "attitude": "att",
}

#: Engine data types that take a ``product_preferences`` entry (every type
#: except ``obs``), in the order used by messages.
PREFERENCE_TYPES = ("nav", "sp3", "clk", "erp", "bia", "dcb", "att", "ionex")

#: 4-character station code (RINEX 2 marker name), after ``upper()``.
STATION4_PATTERN = re.compile(r"^[A-Z0-9]{4}$")
#: 9-character RINEX 3/4 station id: code, monument/receiver digits, country.
STATION9_PATTERN = re.compile(r"^[A-Z0-9]{4}[0-9]{2}[A-Z]{3}$")
#: 10-character analysis-centre product id, e.g. COD0MGXFIN or CAS0OPSRAP.
PRODUCT_ID_PATTERN = re.compile(r"^[A-Z0-9]{3}[0-9][A-Z0-9]{3}[A-Z]{3}$")
#: 3-letter IONEX analysis centre, e.g. COD or IGS.
IONEX_AC_PATTERN = re.compile(r"^[A-Z]{3}$")

_STATION_PATTERNS_TEXT = (
    f"'{STATION4_PATTERN.pattern}' (4-character code) or "
    f"'{STATION9_PATTERN.pattern}' (9-character RINEX id such as 'BRUX00BEL')"
)


def _type_name(value) -> str:
    return type(value).__name__


def _is_int(value) -> bool:
    """``int`` or another integral number (numpy included), never ``bool``."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _sequence_items(value, name: str, *, what: str) -> tuple:
    """Items of a bare ``str`` (one item) or of a list/tuple-like iterable.

    Mappings and sets are refused because their order is not meaningful.
    """
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be a str or a list/tuple of {what}, got {_type_name(value)}: {value!r}")
    if isinstance(value, (_MappingABC, set, frozenset)) or not isinstance(value, _IterableABC):
        raise TypeError(f"{name} must be a str or a list/tuple of {what}, got {_type_name(value)}: {value!r}")
    return tuple(value)


def _unique(items) -> tuple:
    return tuple(dict.fromkeys(items))


# --------------------------------------------------------------------------- project and dates
def normalize_project_name(value) -> str:
    """Validate ``project_name`` and return it as a stripped ``str``.

    Rules: a ``str`` (``None`` and other types -> ``TypeError``),
    non-empty after stripping, without control characters (NUL, line
    breaks, tabs), relative in POSIX and Windows terms (no root,
    no drive letter), without ``..``, ``.`` or empty parts when split on
    ``/`` and ``\\``; nested names such as ``"a/b"`` are allowed and become
    nested folders under ``output_root``.
    """
    if not isinstance(value, str):
        raise TypeError(f"project_name must be a str, got {_type_name(value)}: {value!r}")
    name = value.strip()
    if not name:
        raise ValueError("project_name must be a non-empty string.")
    if any(unicodedata.category(char) == "Cc" for char in name):
        raise ValueError(
            f"project_name must not contain control characters (NUL, line breaks, tabs), got {name!r}"
        )
    windows = PureWindowsPath(name)
    if PurePosixPath(name).is_absolute() or windows.is_absolute() or windows.drive or name[0] in "/\\":
        raise ValueError(
            f"project_name must be a relative folder name under output_root, got {name!r} "
            "(an absolute path or a drive letter); set output_root to choose the parent folder"
        )
    bad = [part for part in re.split(r"[/\\]", name) if part in ("", ".", "..")]
    if bad:
        raise ValueError(
            f"project_name {name!r} must not contain empty, '.' or '..' parts "
            "(the project folder must stay under output_root)"
        )
    return name


def _to_day(item, name: str) -> date:
    if isinstance(item, datetime):
        if item.tzinfo is not None and item.utcoffset() is not None:
            item = item.astimezone(timezone.utc)
        day = item.date()
        # pandas.NaT is a datetime subclass whose date() is NaT again.
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError(f"{name} contains a missing date: {item!r}")
        return day
    if isinstance(item, date):
        return item
    raise TypeError(f"{name} items must be date or datetime objects, got {_type_name(item)}: {item!r}")


def normalize_dates(value: Union[DateLike, Iterable[DateLike]]) -> list:
    """Return ``dates`` as a sorted list of unique ``datetime.date`` objects.

    Accepts a ``date``/``datetime`` (pandas ``Timestamp`` included) or an
    iterable of them. ``str``/``bytes`` and non-iterables -> ``TypeError``
    naming ``dates``; an aware ``datetime`` is converted to UTC before
    ``.date()``, a naive one is taken as UTC/GPST; empty ->
    ``ValueError``. Re-exported as ``project._normalize_dates``.
    """
    if isinstance(value, (datetime, date)):
        value = [value]
    elif isinstance(value, (str, bytes, bytearray)) or not isinstance(value, _IterableABC):
        raise TypeError(
            f"dates must be a date/datetime or an iterable of them, got {_type_name(value)}: {value!r}"
        )
    days = {_to_day(item, "dates") for item in value}
    if not days:
        raise ValueError("dates must contain at least one date.")
    return sorted(days)


# --------------------------------------------------------------------------- vocabularies
def normalize_product_types(value) -> tuple:
    """Return ``product_types`` as engine keys in first-seen order.

    A bare ``str`` is one product type. Items are friendly names from
    :data:`PRODUCT_TYPE_ALIASES` (case-insensitive, surrounding blanks
    ignored); unknown -> ``ValueError`` listing the vocabulary; empty ->
    ``ValueError``; duplicates dropped.
    """
    items = _sequence_items(value, "product_types", what="product type names")
    resolved = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"product_types items must be str, got {_type_name(item)}: {item!r}")
        key = item.strip().lower()
        if key not in PRODUCT_TYPE_ALIASES:
            raise ValueError(
                f"Unknown product type {item!r} in product_types. Supported (case-insensitive): "
                f"{sorted(PRODUCT_TYPE_ALIASES)}"
            )
        resolved.append(PRODUCT_TYPE_ALIASES[key])
    if not resolved:
        raise ValueError("At least one product type is required in product_types.")
    return _unique(resolved)


def normalize_data_types(value) -> tuple:
    """Return the engine's ``data_types`` as a tuple of engine keys.

    The low-level counterpart of :func:`normalize_product_types` for
    ``GNSSDownloadConfig``: case-insensitive engine keys from
    ``core.SUPPORTED_DATA_TYPES`` (never stripped); unknown or empty ->
    ``ValueError``; duplicates dropped.
    """
    items = _sequence_items(value, "data_types", what="data type keys")
    out = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"data_types items must be str, got {_type_name(item)}: {item!r}")
        key = item.lower()
        if key not in core.SUPPORTED_DATA_TYPES:
            raise ValueError(
                f"Unsupported data type {item!r} in data_types. Supported values: {core.SUPPORTED_DATA_TYPES}."
            )
        out.append(key)
    if not out:
        raise ValueError(f"data_types must name at least one data type; supported: {core.SUPPORTED_DATA_TYPES}.")
    return _unique(out)


def normalize_sources(value) -> tuple:
    """Return ``sources`` as a tuple of lower-case source keys in input order.

    A bare ``str`` is one source. Case-insensitive but never stripped (so
    ``" cddis"`` is an error); unknown -> ``ValueError`` listing
    ``SUPPORTED_SOURCES``; empty -> ``ValueError``; duplicates dropped in
    order with one INFO line.
    """
    items = _sequence_items(value, "sources", what="source names")
    out = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"sources items must be str, got {_type_name(item)}: {item!r}")
        key = item.lower()
        if key not in core.SUPPORTED_SOURCES:
            raise ValueError(
                f"Unsupported source {item!r} in sources. Supported values: {core.SUPPORTED_SOURCES}."
            )
        out.append(key)
    if not out:
        raise ValueError(f"sources must name at least one source; supported: {core.SUPPORTED_SOURCES}.")
    unique = _unique(out)
    if len(unique) != len(out):
        LOGGER.info("Duplicate entries removed from sources: %r -> %r", tuple(out), unique)
    return unique


def normalize_rinex_preference(value) -> tuple:
    """Return ``rinex_version_preference`` as a tuple of ``"3"``/``"4"``/``"2"``.

    ``str`` or ``int`` items (not ``bool``); a bare ``str`` or ``int`` is one
    item. Items are not stripped; unknown -> ``ValueError``; empty ->
    ``ValueError``; duplicates dropped.
    """
    if _is_int(value):
        value = (value,)
    items = _sequence_items(value, "rinex_version_preference", what="RINEX versions")
    allowed = core.SUPPORTED_RINEX_PREFERENCES
    out = []
    for item in items:
        if not (isinstance(item, str) or _is_int(item)):
            raise TypeError(
                f"rinex_version_preference items must be str or int, got {_type_name(item)}: {item!r}"
            )
        version = str(item)
        if version not in allowed:
            raise ValueError(
                f"Unsupported RINEX version {item!r} in rinex_version_preference. Supported values: {allowed}."
            )
        out.append(version)
    if not out:
        raise ValueError(f"rinex_version_preference must name at least one RINEX version; supported: {allowed}.")
    return _unique(out)


def normalize_rinex_data_source_preference(value) -> tuple:
    """Return ``rinex_data_source_preference`` as a tuple of ``"R"``/``"S"``/``"U"``.

    The data-source letter of a RINEX 3/4 long observation name
    (``BRUX00BEL_R_...``: R receiver, S data stream, U unknown). ``str``
    items, case-insensitive, not stripped; a bare ``str`` is one item;
    unknown -> ``ValueError``; empty -> ``ValueError``; duplicates dropped.
    """
    items = _sequence_items(value, "rinex_data_source_preference", what="data-source letters")
    allowed = core.SUPPORTED_RINEX_DATA_SOURCES
    out = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(
                f"rinex_data_source_preference items must be str, got {_type_name(item)}: {item!r}"
            )
        letter = item.upper()
        if letter not in allowed:
            raise ValueError(
                f"Unsupported data-source letter {item!r} in rinex_data_source_preference. "
                f"Supported values: {allowed} (R receiver, S data stream, U unknown)."
            )
        out.append(letter)
    if not out:
        raise ValueError(
            f"rinex_data_source_preference must name at least one data-source letter; supported: {allowed}."
        )
    return _unique(out)


def _product_item_error(label: str, data_type: str, item: str) -> "ValueError | None":
    if data_type == "nav":
        if item not in catalog.NAV_PRODUCTS:
            return ValueError(
                f"{label}: {item!r} is not a broadcast navigation product; accepted: {tuple(catalog.NAV_PRODUCTS)}"
            )
        return None
    if data_type == "ionex":
        if not (IONEX_AC_PATTERN.match(item) or PRODUCT_ID_PATTERN.match(item)):
            return ValueError(
                f"{label}: {item!r} is neither a 3-letter IONEX analysis centre such as 'COD' nor a "
                "10-character product id such as 'IGS0OPSFIN'"
            )
        return None
    if not PRODUCT_ID_PATTERN.match(item):
        return ValueError(
            f"{label}: {item!r} is not a 10-character product id such as 'COD0MGXFIN' "
            f"(pattern '{PRODUCT_ID_PATTERN.pattern}')"
        )
    return None


def normalize_product_preferences(mapping) -> dict:
    """Return ``product_preferences`` keyed by engine data-type keys.

    Result: ``dict[str, tuple[str, ...]]`` in input order. Keys are friendly
    names mapped through :data:`PRODUCT_TYPE_ALIASES` case-insensitively
    (``"SP3"``/``"orbit"`` -> sp3, ``"broadcast"``/``"brdc"``/``"nav"`` ->
    nav, ``"gim"``/``"tec"`` -> ionex, ``"osb"``/``"bias"`` -> bia,
    ``"clock"`` -> clk, ``"orbex"`` -> att); ``"obs"``, unknown keys and two
    keys of one type -> ``ValueError``. Values: a ``str`` is one product, a
    list/tuple of non-empty ``str`` (order matters, so sets are refused);
    ``None`` or another type -> ``TypeError`` naming
    ``product_preferences['<key>']``; empty -> ``ValueError``. Items are
    stripped and upper-cased, duplicates dropped. Per type: sp3/clk/erp/bia/
    dcb/att a 10-character id (``COD0MGXFIN``); ionex a 3-letter analysis
    centre or a 10-character id; nav one of ``catalog.NAV_PRODUCTS``.
    ``None`` is the same as ``{}``.
    """
    if mapping is None:
        return {}
    if not isinstance(mapping, _MappingABC):
        raise TypeError(
            "product_preferences must be a mapping of product type to product ids, "
            f"got {_type_name(mapping)}: {mapping!r}"
        )
    accepted = sorted(k for k, v in PRODUCT_TYPE_ALIASES.items() if v != "obs")
    out: dict = {}
    source_key: dict = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(
                f"product_preferences keys must be str product types, got {_type_name(key)}: {key!r}"
            )
        data_type = PRODUCT_TYPE_ALIASES.get(key.strip().lower())
        if data_type == "obs":
            raise ValueError(
                f"product_preferences[{key!r}]: observations have no product preference "
                f"(choose stations and rinex_version_preference instead); accepted keys: {accepted}"
            )
        if data_type is None:
            raise ValueError(
                f"Unknown product_preferences key {key!r}; accepted keys (case-insensitive): {accepted}"
            )
        if data_type in out:
            raise ValueError(
                f"product_preferences keys {source_key[data_type]!r} and {key!r} both name the product type "
                f"{data_type!r}; give one of them"
            )
        label = f"product_preferences[{key!r}]"
        if value is None or not (isinstance(value, (str, list, tuple))):
            raise TypeError(
                f"{label} must be a product id or a list/tuple of product ids, got {_type_name(value)}: {value!r}"
            )
        items = (value,) if isinstance(value, str) else tuple(value)
        products = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError(f"{label} items must be str product ids, got {_type_name(item)}: {item!r}")
            product = item.strip().upper()
            if not product:
                raise ValueError(f"{label} contains an empty product id")
            error = _product_item_error(label, data_type, product)
            if error is not None:
                raise error
            products.append(product)
        if not products:
            raise ValueError(
                f"{label} is empty; give at least one product or drop the key to use the defaults"
            )
        out[data_type] = _unique(products)
        source_key[data_type] = key
    return out


# --------------------------------------------------------------------------- stations
def normalize_station_aliases(mapping) -> dict:
    """Return ``station_aliases`` as ``dict[str, tuple[str, ...]]``, upper case.

    Keys are 4-character codes (``'^[A-Z0-9]{4}$'`` after ``upper()``),
    values one 9-character id or a list/tuple of them in priority order
    (``'^[A-Z0-9]{4}[0-9]{2}[A-Z]{3}$'``) whose first four characters are the
    key. A non-``str`` key or id, or a ``set``/``frozenset``/``dict`` value
    -> ``TypeError``; a wrong pattern, an id of another station, an empty
    value or two keys naming one code -> ``ValueError``. ``None`` is
    the same as ``{}``.
    """
    if mapping is None:
        return {}
    if not isinstance(mapping, _MappingABC):
        raise TypeError(
            "station_aliases must be a mapping of 4-character station codes to 9-character ids, "
            f"got {_type_name(mapping)}: {mapping!r}"
        )
    out: dict = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(
                f"station_aliases keys must be str 4-character station codes, got {_type_name(key)}: {key!r}"
            )
        code = key.strip().upper()
        if not STATION4_PATTERN.match(code):
            raise ValueError(
                f"station_aliases key {key!r} is not a 4-character station code "
                f"(pattern '{STATION4_PATTERN.pattern}')"
            )
        if code in out:
            raise ValueError(f"station_aliases names the station code {code!r} twice")
        label = f"station_aliases[{key!r}]"
        if isinstance(value, str):
            items = (value,)
        elif isinstance(value, (list, tuple)):
            items = tuple(value)
        else:
            raise TypeError(
                f"{label} must be a 9-character station id or a list/tuple of them (order matters), "
                f"got {_type_name(value)}: {value!r}"
            )
        ids = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError(f"{label} items must be str station ids, got {_type_name(item)}: {item!r}")
            station9 = item.strip().upper()
            if not STATION9_PATTERN.match(station9):
                raise ValueError(
                    f"{label}: {item!r} is not a 9-character RINEX station id "
                    f"(pattern '{STATION9_PATTERN.pattern}', e.g. 'WTZR00DEU')"
                )
            if station9[:4] != code:
                raise ValueError(
                    f"{label}: {item!r} names another station; a 9-character id starts with its "
                    f"4-character code (e.g. {code + station9[4:]!r})"
                )
            ids.append(station9)
        if not ids:
            raise ValueError(f"{label} is empty; give at least one 9-character id or drop the key")
        out[code] = _unique(ids)
    return out


def normalize_station(value, name: str = "stations") -> str:
    """Return one station id stripped and upper-cased, checking its pattern."""
    if not isinstance(value, str):
        raise TypeError(f"{name} items must be str station ids, got {_type_name(value)}: {value!r}")
    station = value.strip().upper()
    if not (STATION4_PATTERN.match(station) or STATION9_PATTERN.match(station)):
        raise ValueError(f"station {value!r} in {name} must match {_STATION_PATTERNS_TEXT}")
    return station


def _station_identity(station: str, aliases) -> tuple:
    """What a station request resolves to: its 9-character ids, or the bare code."""
    if len(station) == 9:
        return (station,)
    ids = core.station_variants(station, aliases).rinex3_ids
    return ids if ids else ("code", station)


def _station_reach(station: str, identity: tuple) -> set:
    """File names a station request may end at: its 9-character ids and its RINEX 2 short name.

    A 9-character id gets the short name only with the monument/receiver
    digits ``00`` (plan verdict 3); a 4-character request always may.
    """
    reach = {item for item in identity if len(item) == 9}
    if len(station) == 4 or station[4:6] == "00":
        reach.add(("rinex2", station[:4]))
    return reach


def normalize_stations(value, aliases=None) -> tuple:
    """Return ``stations`` as a tuple of station ids.

    ``None`` -> ``()``; a bare ``str`` is one station. Every item a ``str``
    (``TypeError`` naming ``stations`` and the item), stripped and
    upper-cased, matching the 4- or 9-character pattern (``ValueError``
    naming the station and both patterns), de-duplicated by resolved identity
    through ``aliases`` and the built-in aliases (``"BRUX"`` plus
    ``"BRUX00BEL"`` is one request): the first spelling is kept and one INFO
    line names the dropped ones. Requests that differ but may end
    at the same file (``"WTZR"`` and ``"WTZR00DEU"`` through the RINEX 2
    short name, or two alias lists sharing an id) are both kept, with one
    INFO line. ``aliases`` must already be normalized
    (:func:`normalize_station_aliases`). The result depends on ``aliases``:
    ``DownloadConfig`` keeps the stations as given and derives this tuple
    again whenever ``station_aliases`` changes.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    elif isinstance(value, (bytes, bytearray, _MappingABC, set, frozenset)) or not isinstance(value, _IterableABC):
        raise TypeError(
            f"stations must be a station id or a list/tuple of them, got {_type_name(value)}: {value!r}"
        )
    out: list = []
    seen: dict = {}
    reach: dict = {}
    dropped = []
    overlapping = []
    for item in value:
        station = normalize_station(item)
        identity = _station_identity(station, aliases)
        if identity in seen:
            if station != seen[identity]:
                dropped.append((station, seen[identity]))
            continue
        names = _station_reach(station, identity)
        overlapping.extend((kept, station) for kept, kept_names in reach.items() if names & kept_names)
        seen[identity] = station
        reach[station] = names
        out.append(station)
    if dropped:
        LOGGER.info(
            "Duplicate stations removed (same RINEX id): %s",
            ", ".join(f"{station} (kept {kept})" for station, kept in dropped),
        )
    if overlapping:
        LOGGER.info(
            "Stations kept as separate requests that may download the same file "
            "(a shared RINEX id or RINEX 2 short name): %s",
            ", ".join(f"{first} and {second}" for first, second in overlapping),
        )
    return tuple(out)


# --------------------------------------------------------------------------- numbers and switches
def check_max_workers(value):
    """Validate ``max_workers``: ``None`` or an ``int`` (not ``bool``) >= 1."""
    if value is None:
        return value
    if not _is_int(value):
        raise TypeError(f"max_workers must be None or an int >= 1, got {_type_name(value)}: {value!r}")
    if value < 1:
        raise ValueError(f"max_workers must be None or a positive integer, got {value!r}.")
    return int(value)


def check_timeout(value):
    """Validate ``timeout``: ``None`` (wait forever) or a finite real > 0.

    ``bool``, NaN, inf, 0 and negatives are refused.
    """
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"timeout must be None or a number of seconds > 0, got {_type_name(value)}: {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"timeout must be None or a finite number of seconds > 0, got {value!r}.")
    return value


def check_retries(value):
    """Validate ``retries``: an ``int`` (not ``bool``) >= 0 (default 2)."""
    if not _is_int(value):
        raise TypeError(f"retries must be an int >= 0, got {_type_name(value)}: {value!r}")
    if value < 0:
        raise ValueError(f"retries must be >= 0, got {value!r}.")
    return int(value)


def check_adjacent_days(value):
    """Validate ``sp3_adjacent_days``: an ``int`` (not ``bool``, not float) >= 0."""
    if not _is_int(value):
        raise TypeError(f"sp3_adjacent_days must be an int >= 0, got {_type_name(value)}: {value!r}")
    if value < 0:
        raise ValueError(f"sp3_adjacent_days must be >= 0, got {value!r}.")
    return int(value)


def check_bool(value, name: str):
    """Validate an on/off switch: ``bool`` (or a numpy bool); returns ``bool``."""
    if isinstance(value, bool):
        return value
    kind = type(value)
    if kind.__module__ == "numpy" and kind.__name__ in ("bool_", "bool"):
        return bool(value)
    raise TypeError(f"{name} must be True or False, got {_type_name(value)}: {value!r}")


def check_output_root(value):
    """Validate ``output_root``: a ``str`` or ``os.PathLike`` (returned unchanged)."""
    if isinstance(value, str) or isinstance(value, os.PathLike):
        return value
    raise TypeError(f"output_root must be a str or a path, got {_type_name(value)}: {value!r}")


# --------------------------------------------------------------------------- cross-parameter checks
def _given(value) -> bool:
    return value is not None and value != ""


def check_credentials(token, username, password, credentials_file, *, combinations: bool = True) -> "str | None":
    """Check the Earthdata credential options of ``DownloadConfig``.

    A style counts as given when any of its parts is set (``None`` and ``""``
    are unset): ``token``; ``username``/``password``; ``credentials_file``.
    More than one style, or a username without a password (or the reverse),
    -> ``ValueError``. ``token``, ``username`` and ``password`` must be
    ``str``; ``credentials_file`` a ``str`` or ``os.PathLike``. The file is
    never opened or checked here: it is read by ``DownloadSession.run()``.
    With ``combinations=False`` only the types are checked (an assignment
    may complete a pair or switch styles in two steps); the first given
    style is still returned. Returns the given style (``"token"``, ``"username/password"``,
    ``"credentials_file"``) or ``None``. Messages never contain the values.
    """
    for name, value in (("token", token), ("username", username), ("password", password)):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a str or None, got {_type_name(value)}")
    if credentials_file is not None and not isinstance(credentials_file, (str, os.PathLike)):
        raise TypeError(f"credentials_file must be a str or a path, got {_type_name(credentials_file)}")
    styles = [
        name
        for name, given in (
            ("token", _given(token)),
            ("username/password", _given(username) or _given(password)),
            ("credentials_file", _given(credentials_file)),
        )
        if given
    ]
    if not combinations:
        return styles[0] if styles else None
    if len(styles) > 1:
        raise ValueError("Provide only one credential style: " + " or ".join(styles))
    if styles == ["username/password"] and not (_given(username) and _given(password)):
        raise ValueError("username and password must be given together")
    return styles[0] if styles else None


def check_feasibility(
    product_types,
    sources,
    stations,
    station_aliases,
    product_preferences,
    rinex_preference,
) -> list:
    """Decide from the configuration alone what the request can deliver.

    All arguments are normalized values. Input errors raise ``ValueError``:

    * ``"obs"`` requested without stations;
    * a 4-character station without a known 9-character id while ``"2"`` is
      not in ``rinex_preference`` (the message names ``station_aliases``);
    * nothing requested can be served by the selected sources.

    Otherwise the result lists each item the selected sources cannot provide
    as ``(data_type, station or None, reason)``, using the answers of ``unavailable_reason``
    (``core.unavailable_reason`` and ``core.station_reason``); an empty list
    means everything is servable.
    """
    if "obs" in product_types and not stations:
        raise ValueError('product_types includes "obs" but no stations were given.')
    unavailable: list = []
    servable = 0
    for data_type in product_types:
        reason = core.unavailable_reason(data_type, sources, product_preferences)
        if data_type != "obs":
            if reason is None:
                servable += 1
            else:
                unavailable.append((data_type, None, reason))
            continue
        station_errors = []
        station_items = []
        for station in stations:
            variants = core.station_variants(station, station_aliases)
            station_text = core.station_reason(variants, rinex_preference)
            if station_text is None:
                station_items.append((station, None))
            elif not variants.rinex3_ids:
                station_errors.append(station_text)
            else:
                station_items.append((station, station_text))
        if station_errors:
            raise ValueError("; ".join(station_errors))
        if reason is not None:
            unavailable.append(("obs", None, reason))
            continue
        for station, station_text in station_items:
            if station_text is None:
                servable += 1
            else:
                unavailable.append(("obs", station, station_text))
    if not servable:
        details = "; ".join(reason for _, _, reason in unavailable)
        raise ValueError(f"Nothing requested can be provided by the selected sources: {details}")
    return unavailable


def format_unavailable(unavailable) -> str:
    """One line per unavailable item, ``type[ station]: reason``."""
    return "; ".join(
        f"{data_type}{'' if station is None else ' ' + station}: {reason}"
        for data_type, station, reason in unavailable
    )
