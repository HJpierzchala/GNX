"""Consistent product sets for a download session.

Orbits, clocks, observable-specific biases and ORBEX attitude
(:data:`gnx_py.downloaders.catalog.LINKED_TYPES`) must come from one analysis
centre solution for the whole project, SP3 boundary days included: orbits
and clocks of different centres disagree at day boundaries, and PPP-AR needs
OSB of the same solution as the clocks. :func:`choose` picks that solution
before the per-day engine runs; ``DownloadSession.run`` then restricts each
linked type on each day to the chosen product (:func:`day_preferences`).

Interface: ``choose(config, transport, logger)``
returns a :class:`ProductSetChoice`. Its ``per_item`` maps every linked
``(data_type, day)`` the session will request (SP3 boundary days included)
to the chosen product id, or to ``None`` when no product is imposed and the
engine tries that type's effective preference list.

Effective lists:
    The lead type is the first requested of sp3, clk, bia, att. A linked
    type without an explicit ``product_preferences`` entry starts with the
    explicit list of the lead type, then its own default list, so
    ``{"sp3": ("WUM0MGXFIN",)}`` also asks for WUM clocks and OSB first. The
    candidate solutions are the lead list, then the ids that appear only in
    the other linked lists.

Item classes for a solution P:
    core: SP3 and CLK on the requested (main) days. When neither is
    requested, the lead type on the main days is the core.
    same-solution-only: SP3 on the boundary days, ORBEX on the main days
    (ORBEX of another centre is never used).
    with-fallback: OSB on the main days.

Availability of an item for P:
    P is not eligible for a type whose effective list does not contain it,
    nor when :func:`gnx_py.downloaders.catalog.publishes` says ``False``, nor
    when the candidate generator produces no name for it. Otherwise a valid
    local file of any of P's names under any source makes the item available
    without a request (``overwrite=False`` only; a fully checked file is
    recorded in the download manifest, so the engine does not check it
    again); a dry run without availability checks assumes it is available
    (``checked=False``, no request); else ``exists()`` is asked for P's URLs
    in source order until one exists. The session transport caches those
    answers, so the download that follows costs no second probe.

    Only a definite answer makes an item absent: HTTP 404 at every source,
    or a source that refuses the file (another 4xx, an HTML page, or CDDIS
    without usable Earthdata credentials). An answer that is no answer (a
    timeout, a connection failure, an unreachable host, 429 or 5xx after the
    transport's retries, an exception) counts as available, so a passing
    network fault never switches the analysis centre; the download then
    reports the real outcome. Every such answer and every refusal is kept in
    ``ProductSetChoice.probe_failures`` and logged as a WARNING with its URL
    and reason; missing items that CDDIS might hold but could not be asked
    for want of credentials are named as such instead of "no product
    provides".

    Cost: a product's items are probed newest day first (a missing file is
    most often the newest boundary day, not yet published) and the probing
    stops at the first item it lacks, so a product that cannot be used costs
    few requests; with ``max_workers > 1`` the items are probed in batches of
    ``max_workers``.

Tiers:
    1. The first P with every item available: P everywhere.
    2. Else the first P with every core item available: P everywhere. The
       same-solution-only items it lacks are reported "missing"; OSB it
       lacks comes, per day, from the first product of OSB's own list that
       has it, with one ``UserWarning`` (PPPSession then disables PPP-AR
       unless ``pppar_allow_mixed_products``).
    3. Else per main day: the first P with that day's core items (its SP3
       boundary days from the same P only, fetched again from P when another
       main day already uses that SP3 day with another product), or, when
       none has them, each item from the first product of its own list that
       has it. When more than one product is used, one ``UserWarning`` lists
       every mixed item and the SP3 days fetched twice, and ``mixed`` holds
       them; a single product with gaps only has ``missing`` items.

The choice is final: a chosen file that then fails to download at every
source is an error row for that file; the engine never switches to another
product for a linked type. An SP3 day that is also a main day uses the
main-day choice. In tier 3 an SP3 day inside the windows of main days with
different products is fetched once per product, as different files: the
first is in ``per_item``, the others in ``extra``, so every main day keeps
an SP3 window of one solution.

ERP and navigation are not linked. ERP gets the solution first when the
catalogue says that the solution publishes ERP and ``product_preferences``
has no ERP entry.
"""
from __future__ import annotations

import logging
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Iterable, Mapping, NamedTuple, Sequence

from .catalog import LINKED_TYPES, publishes
from .core import (
    DEFAULT_PRODUCT_PREFERENCES,
    GNSSDownloadConfig,
    GNSSDownloader,
    check_local_file,
)

if TYPE_CHECKING:
    from .config import DownloadConfig
    from .core import DownloadCandidate, DownloadTransport

__all__ = [
    "ProbeFailure",
    "ProductSetChoice",
    "choose",
    "day_preferences",
    "effective_lists",
    "linked_items",
]

LOGGER = logging.getLogger(__name__)

# Candidate generation for the availability probes runs the engine's own
# generator once per (product, type, day); its INFO lines would repeat the
# session's, so they go to a silent child logger.
_GENERATOR_LOGGER = logging.getLogger(f"{__name__}.candidates")
_GENERATOR_LOGGER.addHandler(logging.NullHandler())
_GENERATOR_LOGGER.propagate = False

# Item classes (see the module docstring).
_CORE = "core"
_SAME_ONLY = "same-solution-only"
_FALLBACK = "with-fallback"

# Probe answers: available, absent, or no definite answer (counted as available).
_YES = "yes"
_NO = "no"
_UNKNOWN = "unknown"

# Probe failures listed in full in the WARNING line; the rest are counted.
_FAILURES_SHOWN = 5

# Frames inside gnx_py are skipped when a UserWarning is attributed, so it
# points at the caller's line (``DownloadSession.run()`` or ``choose()``).
_PACKAGE_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _user_warning(message: str) -> None:
    """``UserWarning`` attributed to the first frame outside gnx_py."""
    warnings.warn(message, UserWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))


class ProbeFailure(NamedTuple):
    """An availability probe that got no definite "exists"/"404" answer.

    ``kind`` is ``"unknown"`` (timeout, connection failure, unreachable
    host, 429/5xx after retries, an exception: the item counts as
    available), ``"auth"`` (CDDIS without usable Earthdata credentials) or
    ``"refused"`` (another 4xx, or an HTML page instead of the file); the
    last two count as "not there". ``url`` and ``reason`` are those of the
    first such answer for the item.
    """

    data_type: str
    day: date
    product: str
    kind: str
    url: str
    reason: str


@dataclass
class ProductSetChoice:
    """The analysis-centre products chosen for the linked types of a session.

    Fields:
        per_item: ``{(data_type, day): product_id | None}`` for every linked
            type and every day it is requested (SP3 boundary days included).
            ``None`` means no product is imposed for that item.
        solution: the product id used everywhere in tiers 1 and 2, else
            ``None``.
        tier: 1, 2 or 3 (see the module docstring). 1 when no linked type is
            requested.
        mixed: ``[(data_type, day, product_id)]`` items taken from a product
            other than ``solution`` (OSB in tier 2, per-day choices in tier 3).
        missing: ``[(data_type, day)]`` items no eligible product provides
            (reported as "missing", never filled from another centre).
        checked: ``False`` when availability was not checked (dry run with
            ``check_availability=False``), ``True`` otherwise.
        extra: ``[(data_type, day, product_id)]`` further files of tier 3:
            an SP3 day inside the windows of main days with different
            products is fetched once per product; ``per_item`` holds the
            first, ``extra`` the others. Empty in tiers 1 and 2.
        probe_failures: every :class:`ProbeFailure` of the choice (answers
            that were neither "exists" nor HTTP 404), in day order.
    """

    per_item: "dict[tuple[str, date], str | None]" = field(default_factory=dict)
    solution: "str | None" = None
    tier: int = 1
    mixed: "list[tuple[str, date, str]]" = field(default_factory=list)
    missing: "list[tuple[str, date]]" = field(default_factory=list)
    checked: bool = False
    extra: "list[tuple[str, date, str]]" = field(default_factory=list)
    probe_failures: "list[ProbeFailure]" = field(default_factory=list)

    def product_for(self, data_type: str, day: date) -> "str | None":
        """The product imposed for ``(data_type, day)``, or ``None``."""
        return self.per_item.get((data_type, day))


# --- request shape -------------------------------------------------------------


def _engine_preferences(config: "DownloadConfig") -> "dict[str, tuple[str, ...]]":
    """``config.product_preferences`` as ``{engine key: tuple}``.

    ``DownloadConfig`` stores engine keys and tuples already; this only
    guards against a bare string value.
    """
    return {
        str(key): (value,) if isinstance(value, str) else tuple(value)
        for key, value in dict(config.product_preferences or {}).items()
    }


def _unavailable_types(config: "DownloadConfig") -> set:
    """Data types that ``DownloadConfig`` found unavailable as a whole."""
    return {data_type for data_type, station, _reason in getattr(config, "_unavailable", ()) if station is None}


def _linked_types(config: "DownloadConfig") -> "tuple[str, ...]":
    """The requested linked types in lead order, unavailable types left out."""
    dropped = _unavailable_types(config)
    return tuple(t for t in LINKED_TYPES if t in config.product_types and t not in dropped)


def _unique(items: Iterable[str]) -> "tuple[str, ...]":
    return tuple(dict.fromkeys(items))


def effective_lists(config: "DownloadConfig") -> "dict[str, tuple[str, ...]]":
    """The effective product list of every requested linked type.

    A type with an explicit ``product_preferences`` entry keeps it. Any
    other linked type starts with the lead type's explicit list, then its
    own default (``core.DEFAULT_PRODUCT_PREFERENCES``).
    """
    linked = _linked_types(config)
    if not linked:
        return {}
    preferences = _engine_preferences(config)
    lead_explicit = preferences.get(linked[0], ())
    lists: dict = {}
    for data_type in linked:
        if data_type in preferences:
            lists[data_type] = _unique(preferences[data_type])
        else:
            lists[data_type] = _unique(lead_explicit + tuple(DEFAULT_PRODUCT_PREFERENCES.get(data_type, ())))
    return lists


def linked_items(config: "DownloadConfig") -> "list[tuple[str, date]]":
    """Every linked ``(data_type, day)`` a session requests, in day order.

    SP3 is requested on each main day and ``sp3_adjacent_days`` days around
    it; the other linked types only on the main days. Types that
    ``DownloadConfig`` found unavailable as a whole are left out.
    """
    linked = _linked_types(config)
    items: set = set()
    for main_day in config.dates:
        for data_type in linked:
            items.add((data_type, main_day))
            if data_type == "sp3":
                for offset in range(1, config.sp3_adjacent_days + 1):
                    items.add((data_type, main_day - timedelta(days=offset)))
                    items.add((data_type, main_day + timedelta(days=offset)))
    return sorted(items, key=lambda item: (item[1], LINKED_TYPES.index(item[0])))


def _candidate_solutions(lists: Mapping[str, Sequence[str]], linked: Sequence[str]) -> "tuple[str, ...]":
    """The lead list, then ids that appear only in the other linked lists."""
    ordered: list = []
    for data_type in linked:
        ordered.extend(lists.get(data_type, ()))
    return _unique(ordered)


def _core_types(linked: Sequence[str]) -> "tuple[str, ...]":
    core = tuple(t for t in ("sp3", "clk") if t in linked)
    return core or tuple(linked[:1])


# --- availability --------------------------------------------------------------


class _Availability:
    """Answers "does product P have item (type, day)?", with a cache.

    Thread-safe. ``answer()`` gives ``"yes"``, ``"no"`` or ``"unknown"``
    (no definite answer; see the module docstring); calling the object gives
    ``True`` for ``"yes"`` and ``"unknown"``. The failures are collected in
    ``failures``; a local file that passed the full check is recorded in the
    download manifest (saved by :meth:`save_manifest`).
    """

    def __init__(self, config: "DownloadConfig", transport: "DownloadTransport",
                 lists: Mapping[str, Sequence[str]], logger: logging.Logger) -> None:
        self.config = config
        self.transport = transport
        self.lists = lists
        self.logger = logger
        self.checking = not (config.dry_run and not config.check_availability)
        self._lock = threading.Lock()
        self._cache: dict = {}
        self.failures: "dict[tuple[str, tuple[str, date]], ProbeFailure]" = {}
        self._manifest = None
        self._manifest_dirty = False
        if self.checking and not config.overwrite:
            from .manifest import DownloadManifest  # leaf module, imported on use

            self._manifest = DownloadManifest.load(config.output_dir)

    def candidates(self, product: str, data_type: str, day: date) -> "list[DownloadCandidate]":
        """P's candidates for the item, from the engine's own generator."""
        doy = day.timetuple().tm_yday
        engine_config = GNSSDownloadConfig(
            year=day.year,
            doy_start=doy,
            doy_end=doy,
            output_dir=self.config.output_dir,
            stations=(),
            data_types=(data_type,),
            sources=self.config.sources,
            rinex_version_preference=self.config.rinex_version_preference,
            station_aliases=self.config.station_aliases,
            product_preferences={data_type: (product,)},
            timeout=self.config.timeout,
        )
        engine = GNSSDownloader(engine_config, transport=self.transport, logger=_GENERATOR_LOGGER)
        return engine.build_candidates()

    def __call__(self, product: str, item: "tuple[str, date]") -> bool:
        return self.answer(product, item) != _NO

    def answer(self, product: str, item: "tuple[str, date]") -> str:
        key = (product, item)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        answer, failure = self._probe(product, *item)
        with self._lock:
            self._cache[key] = answer
            if failure is not None:
                self.failures.setdefault(key, failure)
        return answer

    def _probe(self, product: str, data_type: str, day: date) -> "tuple[str, ProbeFailure | None]":
        if product not in self.lists.get(data_type, ()):
            return _NO, None
        if publishes(product, data_type) is False:
            return _NO, None
        candidates = self.candidates(product, data_type, day)
        if not candidates:
            return _NO, None
        if not self.checking:
            return _YES, None
        if self._local_file(candidates):
            return _YES, None
        unknown = refused = None

        def failure(kind: str, candidate, reason: str) -> ProbeFailure:
            return ProbeFailure(data_type, day, product, kind, candidate.url, reason)

        for candidate in candidates:
            try:
                if candidate.requires_auth and not self.transport.has_credentials(candidate):
                    refused = refused or failure("auth", candidate, "no Earthdata credentials for CDDIS")
                    continue
                availability = self.transport.exists(candidate)
            except Exception as exc:  # the download reports the real outcome
                unknown = unknown or failure("unknown", candidate, f"exists() failed: {type(exc).__name__}: {exc}")
                continue
            if availability.exists:
                return _YES, None
            status = availability.http_status
            if status == 404:
                continue
            reason = availability.message or (f"HTTP {status}" if status is not None else "no answer")
            if getattr(availability, "auth_failed", False):
                refused = refused or failure("auth", candidate, reason)
            elif status is None or status == 429 or status >= 500:
                unknown = unknown or failure("unknown", candidate, reason)
            else:
                refused = refused or failure("refused", candidate, reason)
        if unknown is not None:
            return _UNKNOWN, unknown
        return _NO, refused

    def _local_file(self, candidates: "Sequence[DownloadCandidate]") -> bool:
        """Whether a valid local file of any candidate exists."""
        if self._manifest is None:
            return False
        for candidate in candidates:
            check = check_local_file(candidate, self._manifest)
            if not check.valid:
                continue
            if check.reason == "checked":
                # The caller records a fully checked file (check_local_file contract),
                # so the engine's own scan of this file is the quick manifest check.
                try:
                    with self._lock:
                        self._manifest.record(candidate.local_path, sha256=check.sha256, check="full")
                        self._manifest_dirty = True
                except (OSError, ValueError) as exc:
                    self.logger.debug("Manifest entry for %s not recorded: %s", candidate.local_path, exc)
            return True
        return False

    def save_manifest(self) -> None:
        """Save the files the probes checked in full (not in a dry run, as the engine)."""
        if self._manifest is None or not self._manifest_dirty or self.config.dry_run:
            return
        try:
            self._manifest.save()
        except OSError as exc:
            self.logger.warning("Could not save the download manifest %s: %s", self._manifest.path, exc)

    def all_available(self, product: str, items: Sequence["tuple[str, date]"]) -> bool:
        """Whether ``product`` has every item.

        Probes run newest day first and stop at the first item the product
        lacks; with ``max_workers > 1`` they run in batches of that size on
        the session's workers.
        """
        if not items:
            return True
        ordered = sorted(items, key=lambda item: (-item[1].toordinal(), LINKED_TYPES.index(item[0])))
        workers = min(self.config.max_workers or 1, len(ordered))
        if not self.checking or workers <= 1:
            return all(self(product, item) for item in ordered)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(ordered), workers):
                batch = ordered[start:start + workers]
                if not all(list(executor.map(lambda item: self(product, item), batch))):
                    return False
        return True

    def first(self, products: Iterable[str], item: "tuple[str, date]", exclude: "str | None" = None) -> "str | None":
        """The first of ``products`` (other than ``exclude``) that has ``item``."""
        for product in products:
            if product != exclude and self(product, item):
                return product
        return None


# --- the choice ----------------------------------------------------------------


def choose(
    config: "DownloadConfig",
    transport: "DownloadTransport",
    logger: "logging.Logger | None" = None,
) -> ProductSetChoice:
    """Choose one analysis-centre solution for the linked types of ``config``.

    Args:
        config: the validated ``DownloadConfig`` of the session.
        transport: the session transport (``DownloadSession`` wraps it with a
            per-URL ``exists()`` cache); used for availability probes only,
            never for downloads.
        logger: logger for INFO and WARNING lines; defaults to this module's
            logger.

    Returns:
        ProductSetChoice. Warnings about mixed products are emitted here
        (``UserWarning``, attributed to the first caller outside gnx_py),
        once per session.
    """
    logger = logger or LOGGER
    items = linked_items(config)
    if not items:
        return ProductSetChoice()
    linked = _linked_types(config)
    lists = effective_lists(config)
    solutions = _candidate_solutions(lists, linked)
    available = _Availability(config, transport, lists, logger)
    main_days = sorted(set(config.dates))
    classes = _classify(items, linked, set(main_days))
    core_items = [item for item in items if classes[item] == _CORE]
    choice = _tiers(config, items, linked, lists, solutions, available, main_days, classes, core_items, logger)
    choice.probe_failures = sorted(
        available.failures.values(),
        key=lambda f: (f.day, LINKED_TYPES.index(f.data_type), f.product),
    )
    _log_problems(choice, logger)
    available.save_manifest()
    return choice


def _tiers(config, items, linked, lists, solutions, available, main_days, classes, core_items, logger):
    checked = available.checking

    # Tier 1: one solution has everything.
    for product in solutions:
        if available.all_available(product, items):
            logger.info(
                "Product set: %s for %s (%s).",
                product,
                ", ".join(linked),
                "every item available" if checked else "availability not checked",
            )
            return ProductSetChoice(per_item={item: product for item in items}, solution=product, tier=1,
                                    checked=checked)

    # Tier 2: one solution has every core item.
    for product in solutions:
        if available.all_available(product, core_items):
            choice = ProductSetChoice(solution=product, tier=2, checked=checked)
            for item in items:
                _assign(choice, item, product, classes[item], lists, available)
            _warn_tier2(choice, logger)
            return choice

    # Tier 3: per main day.
    choice = ProductSetChoice(tier=3, checked=checked)
    adjacent = config.sp3_adjacent_days
    per_day = {
        main_day: next(
            (product for product in solutions
             if available.all_available(product, [item for item in core_items if item[1] == main_day])),
            None,
        )
        for main_day in main_days
    }
    for item in items:
        data_type, day = item
        # The main days whose request needs this item, the item's own day first.
        owners = [day] if day in per_day else []
        if data_type == "sp3":
            owners += [m for m in main_days if m != day and abs((day - m).days) <= adjacent]
        products = [per_day[m] for m in owners]
        primary = products[0] if day in per_day else next((p for p in products if p is not None), None)
        if primary is not None:
            _assign(choice, item, primary, classes[item], lists, available)
        else:
            fallback = available.first(lists.get(data_type, ()), item)
            choice.per_item[item] = fallback
            if fallback is None:
                choice.missing.append(item)
        # Another main day's window needs this SP3 day from its own product:
        # a further file, never a mixed window.
        for other in _unique(p for p in products if p is not None):
            if other == choice.per_item[item]:
                continue
            choice.extra.append((data_type, day, other))
            if not available(other, item) and item not in choice.missing:
                choice.missing.append(item)
    # Mixed only when more than one product is used; a single product with
    # gaps is reported through ``missing``.
    used = {(t, d, p) for (t, d), p in choice.per_item.items() if p is not None} | set(choice.extra)
    order = lambda entry: (entry[1], LINKED_TYPES.index(entry[0]), entry[2])  # noqa: E731
    choice.mixed = sorted(used, key=order) if len({p for _t, _d, p in used}) > 1 else []
    choice.extra.sort(key=order)
    choice.missing.sort(key=lambda entry: (entry[1], LINKED_TYPES.index(entry[0])))
    if choice.mixed:
        shared = ""
        if choice.extra:
            shared = (
                " SP3 days in the windows of main days with different products are fetched once per "
                f"product, as different files: {_mixed_text(choice.extra)}."
            )
        _user_warning(
            f"No single analysis-centre solution provides every {'/'.join(_core_types(linked)).upper()} "
            "file of the session; "
            f"linked products are mixed per day: {_mixed_text(choice.mixed)}.{shared} "
            "Orbits, clocks and OSB of different centres disagree; PPPSession disables PPP-AR "
            "for mixed OSB (pppar_allow_mixed_products)."
        )
    return choice


def _classify(items, linked, main_days) -> dict:
    core_types = _core_types(linked)
    classes: dict = {}
    for data_type, day in items:
        if day in main_days and data_type in core_types:
            classes[(data_type, day)] = _CORE
        elif data_type == "bia" and day in main_days:
            classes[(data_type, day)] = _FALLBACK
        else:  # SP3 boundary days, ORBEX
            classes[(data_type, day)] = _SAME_ONLY
    return classes


def _assign(choice: ProductSetChoice, item, product: str, item_class: str, lists, available: _Availability) -> None:
    """Give ``item`` to ``product``, or apply the rule of its class when it lacks it."""
    if item_class == _CORE or available(product, item):
        choice.per_item[item] = product
        return
    if item_class == _FALLBACK:
        other = available.first(lists.get(item[0], ()), item, exclude=product)
        if other is not None:
            choice.per_item[item] = other
            choice.mixed.append((item[0], item[1], other))
            return
    # Strict within the set: the engine looks for this product only and
    # reports the item missing.
    choice.per_item[item] = product
    choice.missing.append(item)


def _warn_tier2(choice: ProductSetChoice, logger: logging.Logger) -> None:
    if choice.mixed:
        others = sorted({product for _t, _d, product in choice.mixed})
        days = _days_text(d for _t, d, _p in choice.mixed)
        _user_warning(
            f"OSB from {', '.join(others)} with orbits/clocks from {choice.solution} ({days}): "
            "PPPSession disables PPP-AR (pppar_allow_mixed_products)"
        )
    logger.info("Product set: %s (tier 2).", choice.solution)


def _log_problems(choice: ProductSetChoice, logger: logging.Logger) -> None:
    """WARNING lines for missing items and for probes without a definite answer."""
    auth_items = {(f.data_type, f.day) for f in choice.probe_failures if f.kind == "auth"}
    plain = [item for item in choice.missing if item not in auth_items]
    blocked = [item for item in choice.missing if item in auth_items]
    if plain:
        if choice.tier == 2:
            logger.warning(
                "Product set %s lacks %s; reported missing (never taken from another centre).",
                choice.solution,
                _items_text(plain),
            )
        else:
            logger.warning("Product set: no product provides %s.", _items_text(plain))
    if blocked:
        logger.warning(
            "Product set: %s not found at the sources checked; CDDIS was not checked without usable "
            "Earthdata credentials (token, username/password, credentials_file or EARTHDATA_TOKEN).",
            _items_text(blocked),
        )
    shown = [f for f in choice.probe_failures if f.kind != "auth"]
    if shown:
        logger.warning(
            "Product set: %d availability check(s) got no definite answer; 'unknown' counts as "
            "available (the download reports the real outcome), 'refused' as not there: %s%s",
            len(shown),
            "; ".join(f"{f.kind} {f.product} {f.data_type} {f.day.isoformat()} {f.url} ({f.reason})"
                      for f in shown[:_FAILURES_SHOWN]),
            f"; and {len(shown) - _FAILURES_SHOWN} more" if len(shown) > _FAILURES_SHOWN else "",
        )


def _days_text(days) -> str:
    """Dates as compact ranges: ``2024-02-03..2024-02-05, 2024-02-09``."""
    ordered = sorted(set(days))
    runs: list = []
    for day in ordered:
        if runs and (day - runs[-1][1]).days == 1:
            runs[-1][1] = day
        else:
            runs.append([day, day])
    return ", ".join(
        start.isoformat() if start == end else f"{start.isoformat()}..{end.isoformat()}" for start, end in runs
    )


def _items_text(items) -> str:
    grouped: dict = {}
    for data_type, day in items:
        grouped.setdefault(data_type, []).append(day)
    return "; ".join(f"{t} {_days_text(days)}" for t, days in grouped.items())


def _mixed_text(mixed) -> str:
    grouped: dict = {}
    for data_type, day, product in mixed:
        grouped.setdefault(product, {}).setdefault(data_type, []).append(day)
    return "; ".join(
        f"{product}: " + ", ".join(f"{t} {_days_text(days)}" for t, days in by_type.items())
        for product, by_type in grouped.items()
    )


# --- engine preferences per day ------------------------------------------------


def day_preferences(
    config: "DownloadConfig",
    choice: ProductSetChoice,
    day: date,
    data_types: Iterable[str],
) -> "dict[str, tuple[str, ...]]":
    """``product_preferences`` for the per-day engine request of ``day``.

    A linked type gets ``(P,)`` when the choice imposes P, else its
    effective list. ERP without an explicit entry gets the day's solution
    first when the catalogue says that it publishes ERP. Every other entry
    of ``config.product_preferences`` is passed on unchanged.
    """
    preferences = _engine_preferences(config)
    lists = effective_lists(config)
    data_types = tuple(data_types)
    for data_type in data_types:
        if data_type not in lists:
            continue
        product = choice.product_for(data_type, day)
        preferences[data_type] = (product,) if product is not None else lists[data_type]
    if "erp" in data_types and "erp" not in preferences:
        solution = choice.solution or next(
            (choice.product_for(t, day) for t in ("sp3", "clk") if choice.product_for(t, day)), None
        )
        if solution is not None and publishes(solution, "erp") is True:
            preferences["erp"] = _unique((solution,) + tuple(DEFAULT_PRODUCT_PREFERENCES.get("erp", ())))
    return preferences
