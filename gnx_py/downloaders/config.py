"""GNSS download session configuration and orchestration.

This module contains the recommended public entry point for GNSS archive
downloads:

- ``DownloadConfig``: user-facing configuration for one download session
  (project folder, dates, product families, stations, sources, credentials
  and execution switches). It validates every setting before any I/O, again
  on every assignment and at the start of every run.
- ``DownloadSession``: the orchestrator that expands a ``DownloadConfig`` into
  per-day requests for the low-level engine, runs them and aggregates the
  results.
- ``DownloadReport``: the aggregated result returned by
  ``DownloadSession.run``.

The layout follows ``gnx_py.ppp.config`` and ``gnx_py.ionosphere.config``:
configuration and session live here, while the heavy lifting stays in
separate modules. Candidate URL generation, transports, HTTP handling and
response validation live in :mod:`gnx_py.downloaders.core`; the checks of
every setting live in :mod:`gnx_py.downloaders.validation` (shared with the
low-level ``GNSSDownloadConfig``); credential resolution lives in
:mod:`gnx_py.downloaders.project`.
"""
from __future__ import annotations

import difflib
import logging
import os
import warnings
from collections.abc import Iterable as _IterableABC
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

from ..core.session_errors import guarded_session_run
from . import validation as _v
from .core import (
    LOGGER,
    ArchiveDownloadReport,
    CDDISCredentials,
    DownloadResult,
    DownloadTransport,
    GNSSDownloadConfig,
    GNSSDownloader,
    RequestsDownloadTransport,
    UnavailableRequest,
)
from .product_sets import ProductSetChoice
from .project import (
    DEFAULT_SP3_ADJACENT_DAYS,
    PRODUCT_TYPE_ALIASES,
    DateLike,
    _resolve_credentials,
)


@dataclass
class DownloadConfig:
    """User-facing configuration for one GNSS download session.

    Status:
        Active public API. ``DownloadConfig`` together with
        ``DownloadSession`` is the recommended entry point for downloading
        GNSS observations, broadcast navigation and precise products. The
        low-level ``GNSSDownloadConfig``/``GNSSDownloader`` engine in
        :mod:`gnx_py.downloaders.core` remains available for callers that
        want to work directly in year/day-of-year terms. The former
        ``ProjectDownloadConfig`` name is a deprecated alias of this class.

    Purpose:
        The object describes one download session the way a user thinks
        about it: a project name (which becomes the output folder), calendar
        dates, product families, optional station ids, archive sources,
        credentials and execution switches. ``DownloadSession`` reads this
        object but does not define configuration defaults itself. Every
        setting is checked here, before any file, folder or network access:
        a wrong type raises ``TypeError``, a wrong value ``ValueError``, and
        the message names the setting and the offending item.

    Main parameters:
        project_name:
            Name of the project folder under ``output_root``: a relative
            name such as ``"brux_2024"`` or ``"campaign/brux"`` (nested
            folders allowed). Absolute paths, drive letters and ``.``/``..``
            or empty parts are refused, so the folder always stays under
            ``output_root``. Surrounding blanks are removed.
        dates:
            A single ``date``/``datetime`` (pandas ``Timestamp`` included) or
            an iterable of them; strings and numbers are refused. Normalized
            to a sorted, de-duplicated list of ``datetime.date``. A datetime
            with a time zone is converted to UTC first; a naive datetime is
            taken as UTC (GNSS file days are UTC/GPS days).
        product_types:
            One or more friendly product names (case-insensitive), normalized
            to a tuple of engine keys in first-seen order, e.g.
            ``"broadcast"`` becomes ``"nav"``. See "Supported products".
        stations:
            4-character codes (``^[A-Z0-9]{4}$``) or 9-character RINEX ids
            (``^[A-Z0-9]{4}[0-9]{2}[A-Z]{3}$``, e.g. ``"BRUX00BEL"``);
            stripped and upper-cased. Requests that resolve to the same
            RINEX id (``"BRUX"`` with its alias and ``"BRUX00BEL"``) are kept
            once, under the first spelling, with an INFO line. The stations
            as given are kept, so this result is derived again (and dropped
            spellings come back) when ``station_aliases`` changes. Required
            when ``"obs"`` is
            requested; a 4-character code needs a 9-character id
            (``station_aliases``) unless ``"2"`` is in
            ``rinex_version_preference``.
        output_root:
            Parent directory (``str`` or path) of the project folder.
            Defaults to the current directory. A relative root is resolved
            against the working directory when ``DownloadSession.run`` runs.
        output_dir:
            Read-only: ``Path(output_root) / project_name``. Assigning it
            raises ``AttributeError``; set ``output_root`` and
            ``project_name`` instead.
        sources:
            Archive sources tried in priority order: ``"cddis"``, ``"bkg"``,
            ``"code"`` (default: these three in that order) and ``"euref"``
            (the BKG EUREF/EPN archive, opt-in). A single string is one
            source; case does not matter, blanks are not removed; duplicates
            are dropped.
        token, username, password, credentials_file:
            CDDIS/Earthdata credentials. Provide at most one style:

            * ``token``: an already-issued Earthdata bearer token.
            * ``username`` + ``password`` (always together): exchanged for a
              bearer token by ``DownloadSession.run`` via
              :func:`gnx_py.downloaders.earthdata_auth.fetch_earthdata_token`.
            * ``credentials_file``: a small text file parsed by
              :func:`gnx_py.downloaders.read_credentials_file` when the
              session runs (it is not opened here), then exchanged the same
              way.

            If none is given, the default ``RequestsDownloadTransport`` falls
            back to the ``EARTHDATA_TOKEN`` environment variable; without a
            token, availability checks report CDDIS candidates as
            ``auth_missing`` and move on to the next source.
        sp3_adjacent_days:
            Whenever ``"sp3"`` is requested, SP3 is also requested for this
            many days before and after each requested date (default 1: the
            previous day, the day itself and the next day). Other product
            types are never expanded this way. An ``int >= 0``.
        station_aliases:
            Map of 4-character station codes to one 9-character RINEX 3/4 id
            or a list/tuple of them in priority order, e.g.
            ``{"WTZR": "WTZR00DEU"}``, enabling long-name observation
            candidates for 4-character codes. Every id must start with its
            key. Stored upper case as tuples.
        product_preferences:
            Per-product-type list of products tried in priority order, e.g.
            ``{"sp3": "COD0MGXFIN"}`` or ``{"orbit": ("COD0MGXFIN",
            "GRG0MGXFIN")}``. Keys accept the friendly names of
            ``product_types`` except observations (``"SP3"``, ``"orbit"``,
            ``"clock"``, ``"osb"``, ``"gim"``, ``"broadcast"``, ``"orbex"``
            ...) and are stored as engine keys; one key per type. Values: a
            10-character product id (``COD0MGXFIN``, ``CAS0OPSRAP``) for
            sp3/clk/erp/bia/dcb/att; a 3-letter analysis centre (``COD``)
            or a 10-character id for ionex; for nav one of ``BRDC00IGS``
            (RINEX 3), ``BRD400DLR`` (RINEX 4, carries the BeiDou-3 BDGIM
            parameters), ``BRDM00DLR`` (RINEX 3) and ``BRDC`` (GPS-only
            RINEX 2). Without an entry the engine defaults apply.
        rinex_version_preference:
            RINEX version priority (``"3"``, ``"4"``, ``"2"``; ``int`` items
            accepted) for observation candidates and, without a ``"nav"``
            product preference, for navigation candidates.
        rinex_data_source_preference:
            Data-source letters of RINEX 3/4 long observation names, in
            trial order: ``"R"`` (from a receiver), ``"S"`` (from a data
            stream, e.g. the EPN stations of national RTK networks such as
            ``KATO00POL_S_...``), ``"U"`` (unknown). Default ``("R", "S")``:
            the receiver file first, then the stream file. A single string
            is one letter; case does not matter; duplicates are dropped.
            RINEX 2 short names have no such letter.
        overwrite:
            Replace existing local files instead of skipping them.
        dry_run, check_availability:
            ``dry_run=True`` never downloads files. With
            ``check_availability=True`` it still performs lightweight URL
            checks and reports which remote file would be downloaded; with
            ``check_availability=False`` it only generates candidate URLs.
        max_workers:
            Worker threads used by the engine to process independent request
            groups of one day concurrently: ``None`` or an ``int >= 1``;
            ``None`` or ``1`` is sequential.
        timeout:
            HTTP timeout in seconds for archive requests and the token
            exchange: a finite number ``> 0``, or ``None`` to wait without a
            limit (a stalled server then blocks the session).
        verbose:
            Shows the INFO messages of the ``gnx_py.downloaders`` loggers
            (one handler on that logger; the root logger is not changed).
        retries:
            Extra attempts per file after a timeout, a broken connection, a
            cut transfer, a server error (5xx) or 429, before the next
            candidate is tried: an ``int >= 0`` (default 2, waits of 1 s then
            2 s).

    Validation:
        Construction normalizes and checks each setting on its own, then the
        combinations: one credential style and ``username`` with
        ``password``; the request's feasibility (``"obs"`` needs stations, a
        4-character station needs a 9-character id unless ``"2"`` is
        preferred, and ``ValueError`` when nothing requested can be served
        by the selected sources); the project folder under ``output_root``.
        ``validate_for_run()`` (called first by ``DownloadSession.run``)
        runs the same full routine.

        Assigning a setting after construction checks and normalizes that
        setting at once (a rejected assignment leaves the object
        unchanged) and repeats the warnings below when their content
        changes. Errors of a combination are not raised on assignment,
        because paired settings change in two steps (``username`` then
        ``password``, ``token = None`` then ``credentials_file``,
        ``stations`` then ``product_types``): they are raised by
        ``validate_for_run()``, hence when the session runs. Misspelled
        setting names are refused with ``AttributeError`` (with a "did you
        mean" hint). Construction never creates, reads or checks a file or
        folder.

    Supported products:
        ``"sp3"`` (also ``"orbit"``, ``"orbits"``, ``"precise_orbit"``),
        ``"broadcast"`` (also ``"nav"``, ``"navigation"``, ``"brdc"``),
        ``"ionex"`` (also ``"gim"``, ``"tec"``), ``"bia"`` (also ``"osb"``,
        ``"bias"``), ``"obs"`` (also ``"observation"``, ``"observations"``),
        ``"clk"`` (also ``"clock"``), ``"att"`` (also ``"orbex"``,
        ``"attitude"``; ORBEX satellite attitude), ``"dcb"`` and ``"erp"``.
        See :data:`gnx_py.downloaders.PRODUCT_TYPE_ALIASES` for the full
        mapping.

    Supported sources:
        ``"cddis"`` (NASA CDDIS; requires Earthdata Login, which the default
        transport provides as a bearer token; every product type),
        ``"bkg"`` (BKG IGS and IGSac archives; no IONEX and no ORBEX
        attitude), ``"code"`` (CODE/AIUB; only CODE's own ``COD``
        products, no station observations or broadcast navigation; with the
        default product lists e.g. ``dcb`` is unavailable from ``"code"``)
        and ``"euref"`` (BKG as the EUREF Permanent GNSS Network data
        centre, ``root_ftp/EUREF/obs/``: daily observations of the EPN
        stations, many of which are not in the IGS observation directory;
        no products and no navigation). ``"euref"`` is not a default source:
        add it for EPN stations, e.g. ``sources=("bkg", "euref")``.
        :data:`gnx_py.downloaders.core.SOURCE_CAPABILITIES` and
        :func:`gnx_py.downloaders.core.unavailable_reason` hold the answers
        the checks use.

    Product choice:
        Orbits, clocks, OSB biases and ORBEX attitude (``sp3``, ``clk``,
        ``bia``, ``att``) come from one analysis-centre solution for the
        whole project, the SP3 boundary days included: the product decides
        before the archive (``COD0MGXFIN`` at BKG is preferred to
        ``GRG0MGXFIN`` at CDDIS). For the other types the source order
        decides first, then the product order within each source.

    Warnings:
        One ``UserWarning`` at construction (and again when an assignment
        changes it) when:

        * part of the request cannot be served by the selected sources, e.g.
          IONEX with ``sources=("bkg",)``: such items are reported by the
          session as outcomes with status ``"unavailable"`` and the reason;
        * credentials are given while ``"cddis"`` is not a source: they are
          used only for CDDIS.

        ``token`` and ``password`` are excluded from ``repr`` and never
        appear in messages. Prefer ``credentials_file`` or the
        ``EARTHDATA_TOKEN`` environment variable over credentials written
        into scripts that may be shared or committed.

    Limitations:
        Whether a product actually exists for a date and source is only
        known after an availability check or a download. Input that the
        checks cannot see (a missing or unreadable ``credentials_file``, a
        failed token exchange) fails in ``DownloadSession.run``.
    """

    project_name: str
    dates: Union[DateLike, Iterable[DateLike]]
    product_types: Union[str, Iterable[str]]
    stations: Union[str, Iterable[str], None] = None
    output_root: Union[str, Path] = "."
    sources: Sequence[str] = ("cddis", "bkg", "code")
    token: "str | None" = field(default=None, repr=False)
    username: "str | None" = None
    password: "str | None" = field(default=None, repr=False)
    credentials_file: "str | Path | None" = None
    sp3_adjacent_days: int = DEFAULT_SP3_ADJACENT_DAYS
    station_aliases: Mapping[str, str] = field(default_factory=dict)
    product_preferences: Mapping[str, Sequence[str]] = field(default_factory=dict)
    rinex_version_preference: Sequence[str] = ("3", "4", "2")
    overwrite: bool = False
    dry_run: bool = False
    check_availability: bool = True
    max_workers: "int | None" = 4
    timeout: "float | None" = 30.0
    verbose: bool = False
    retries: int = 2
    # Appended in bug-fix stage 13d (after ``retries``) so positional use keeps working.
    rinex_data_source_preference: Sequence[str] = ("R", "S")

    def __post_init__(self) -> None:
        # Requested items no selected source can provide, as
        # (data_type, station or None, reason); turned into "unavailable"
        # outcomes per day by DownloadSession (contract K-OUT).
        self._unavailable: list = []
        self._credentials_unused = False
        # The stations as given (before de-duplication), so that a later
        # change of station_aliases derives ``stations`` again from them.
        self._stations_given = self.stations
        self._apply(self._validated_values(self._raw_values()), stacklevel=4, initial=True)
        self._ready = True

    # ------------------------------------------------------------------ checks
    @property
    def output_dir(self) -> Path:
        """``Path(output_root) / project_name`` (read-only)."""
        return Path(self.output_root) / self.project_name

    def validate_for_run(self) -> None:
        """Re-run every check on the current values (start of ``DownloadSession.run``).

        Raises the combination errors that assignments do not raise (see
        "Validation" in the class docstring) and catches in-place changes
        that bypass the assignment checks, e.g. ``config.dates.append(...)``.
        Normalized values and the unavailable items are written back; a
        warning is repeated only when its content changed since construction
        or the last assignment.
        """
        self._apply(self._validated_values(self._raw_values()), stacklevel=3)

    def __setattr__(self, name: str, value) -> None:
        if name == "output_dir":
            raise AttributeError("output_dir is derived from output_root and project_name; set those")
        if name.startswith("_") or not self.__dict__.get("_ready", False):
            object.__setattr__(self, name, value)
            return
        if name not in _CONFIG_FIELDS:
            hint = difflib.get_close_matches(name, _CONFIG_FIELDS, n=1)
            raise AttributeError(
                f"DownloadConfig has no setting {name!r}"
                + (f"; did you mean {hint[0]!r}?" if hint else f"; settings: {', '.join(_CONFIG_FIELDS)}")
            )
        values = self._raw_values()
        values[name] = value
        self._apply(self._validated_values(values, combinations=False), stacklevel=3)

    def _raw_values(self) -> dict:
        values = {name: getattr(self, name) for name in _CONFIG_FIELDS}
        values["stations"] = self._stations_given
        return values

    @staticmethod
    def _validated_values(values: dict, *, combinations: bool = True) -> dict:
        """Check and normalize a full set of settings; return them plus derived state.

        With ``combinations=False`` (an assignment) the errors of a
        combination of settings are left to ``validate_for_run``; the
        unavailable items are then ``None`` when the combination is not
        valid yet (kept unchanged by ``_apply``).
        """
        out = dict(values)
        stations = values["stations"]
        if isinstance(stations, (list, tuple)) or (
            isinstance(stations, _IterableABC)
            and not isinstance(stations, (str, bytes, bytearray, _MappingABC, set, frozenset))
        ):
            stations = tuple(stations)          # a generator is read once
        out["_stations_given"] = stations
        out["project_name"] = _v.normalize_project_name(values["project_name"])
        out["dates"] = _v.normalize_dates(values["dates"])
        out["product_types"] = _v.normalize_product_types(values["product_types"])
        out["output_root"] = _v.check_output_root(values["output_root"])
        out["sources"] = _v.normalize_sources(values["sources"])
        out["rinex_version_preference"] = _v.normalize_rinex_preference(values["rinex_version_preference"])
        out["rinex_data_source_preference"] = _v.normalize_rinex_data_source_preference(
            values["rinex_data_source_preference"]
        )
        out["station_aliases"] = _v.normalize_station_aliases(values["station_aliases"])
        out["stations"] = _v.normalize_stations(stations, out["station_aliases"])
        out["product_preferences"] = _v.normalize_product_preferences(values["product_preferences"])
        out["sp3_adjacent_days"] = _v.check_adjacent_days(values["sp3_adjacent_days"])
        out["max_workers"] = _v.check_max_workers(values["max_workers"])
        out["timeout"] = _v.check_timeout(values["timeout"])
        out["retries"] = _v.check_retries(values["retries"])
        for name in ("overwrite", "dry_run", "check_availability", "verbose"):
            out[name] = _v.check_bool(values[name], name)
        style = _v.check_credentials(
            values["token"],
            values["username"],
            values["password"],
            values["credentials_file"],
            combinations=combinations,
        )
        _check_project_folder(out["output_root"], out["project_name"])
        try:
            unavailable = _v.check_feasibility(
                out["product_types"],
                out["sources"],
                out["stations"],
                out["station_aliases"],
                out["product_preferences"],
                out["rinex_version_preference"],
            )
        except ValueError:
            if combinations:
                raise
            unavailable = None
        out["_unavailable"] = unavailable
        out["_credentials_unused"] = style is not None and "cddis" not in out["sources"]
        return out

    def _apply(self, values: dict, *, stacklevel: int, initial: bool = False) -> None:
        """Write validated values back and emit the warnings that changed."""
        unavailable = values.pop("_unavailable")
        credentials_unused = values.pop("_credentials_unused")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if unavailable is None:                 # combination not valid yet: decided by validate_for_run
            unavailable = self._unavailable
        elif unavailable and (initial or unavailable != self._unavailable):
            warnings.warn(
                "Part of the request cannot be provided by the selected sources and will be reported "
                "with status 'unavailable': " + _v.format_unavailable(unavailable),
                UserWarning,
                stacklevel=stacklevel,
            )
        if credentials_unused and (initial or not self._credentials_unused):
            warnings.warn(
                "Earthdata credentials are used only for CDDIS; "
                f"sources={self.sources!r} do not include it",
                UserWarning,
                stacklevel=stacklevel,
            )
        object.__setattr__(self, "_unavailable", unavailable)
        object.__setattr__(self, "_credentials_unused", credentials_unused)


#: The settings of ``DownloadConfig`` in field order.
_CONFIG_FIELDS: "tuple[str, ...]" = tuple(item.name for item in fields(DownloadConfig))


def _check_project_folder(output_root, project_name: str) -> None:
    """The project folder must lie under ``output_root`` (lexical check, no I/O)."""
    root = os.path.abspath(os.fspath(output_root))
    folder = os.path.abspath(os.path.join(root, project_name))
    if os.path.commonpath([root, folder]) != root:
        raise ValueError(
            f"project_name {project_name!r} leaves output_root {os.fspath(output_root)!r}; "
            "use a relative folder name"
        )


# Engine statuses that satisfy a request with a file on disk.
SATISFIED_STATUSES = ("downloaded", "skipped_existing")
# Engine statuses that end the candidate search for a request.
_FINAL_STATUSES = ("downloaded", "skipped_existing", "dry_run")


@dataclass(frozen=True)
class RequestOutcome:
    """Outcome of one requested item: (data type, day[, station]).

    ``status`` is an engine status (``downloaded``,
    ``skipped_existing``, ``dry_run``, ``missing``, ``error``,
    ``auth_missing``) or ``"unavailable"`` (no selected source provides the
    item; ``message`` holds the reason). ``path`` is the local file for
    ``downloaded``/``skipped_existing`` and the would-be path for
    ``dry_run``, else ``None``. ``product``, ``source`` and ``url`` describe
    the candidate that decided the outcome (the last one tried when none
    succeeded). ``tried`` holds every engine row of the request in order.
    """

    data_type: str
    day: date
    station: "str | None"
    status: str
    path: "Path | None" = None
    product: "str | None" = None
    source: "str | None" = None
    url: "str | None" = None
    message: str = ""
    tried: "tuple[DownloadResult, ...]" = ()


# Imports of the session region (C4, bug-fix stage 12). The integration step
# moves them into the import block at the top of the module (owned by C1).
import threading  # noqa: E402
from datetime import timezone  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

from .core import RemoteFileMissing  # noqa: E402
from .product_sets import _user_warning  # noqa: E402
from .product_sets import choose as choose_product_sets  # noqa: E402
from .product_sets import day_preferences  # noqa: E402

if TYPE_CHECKING:
    from .core import DownloadCandidate, URLAvailability


@dataclass
class DownloadReport:
    """Aggregated result of one ``DownloadSession.run`` call.

    Status:
        Active public result container returned by ``DownloadSession.run``,
        the download counterpart of ``PPPResult``. The former
        ``ProjectDownloadReport`` name is a deprecated alias of this class.

    Purpose:
        Answers, per requested item, what happened and where the file is:
        ``report.files("sp3")`` gives the paths ``PPPConfig`` needs, and
        ``report.outcomes()`` one line per requested (type, day[, station]).

    Fields:
        ``project_name`` and ``output_dir`` echo the session configuration.
        ``reports`` holds one ``ArchiveDownloadReport`` per calendar day sent
        to the low-level engine, in chronological order, including the extra
        SP3 boundary days; each holds the engine rows (one per candidate
        tried) and the day's ``unavailable`` entries. A day whose SP3 file is
        needed from a further product (``product_sets.extra``, tier 3 only)
        has one more report right after it. ``product_sets`` is the
        :class:`~gnx_py.downloaders.product_sets.ProductSetChoice` of the
        session (``None`` for a report not built by ``DownloadSession``),
        ``mixed_products`` is ``True`` when linked products (SP3, CLK, OSB,
        ORBEX) come from more than one analysis-centre solution, and
        ``sp3_adjacent_days`` is the SP3 window used by ``files("sp3",
        day=...)``.

    Helpers:
        ``outcomes()`` gives one :class:`RequestOutcome` per requested item,
        with status ``downloaded``, ``skipped_existing``, ``dry_run``,
        ``missing``, ``error``, ``auth_missing`` or ``unavailable``.
        ``files(data_type, day=None, station=None)`` and ``files_by_type()``
        give the local paths of satisfied requests (``downloaded`` or
        ``skipped_existing``) in day order; ``files("sp3", day=d)`` gives
        the SP3 window of ``d`` from one solution whenever the session
        downloaded one. ``unavailable`` lists the
        requested items no selected source provides, with the reason.
        ``all_results()`` flattens every engine row; ``by_status()``,
        ``downloaded``, ``errors`` (``"error"`` and ``"auth_missing"``),
        ``missing``, ``skipped_existing`` and ``dry_runs`` filter them.
        ``summary()`` sums the per-day status counts of the engine rows
        together with the number of generated candidates, and adds
        ``"requests"`` (number of outcomes) and ``"unavailable"``.

    Warnings:
        Engine rows are per candidate, so one request that fell back from
        CDDIS to BKG has an ``auth_missing`` row and a ``downloaded`` row;
        ``errors`` lists the first even though the request was satisfied.
        Use ``outcomes()`` to see what needs attention.

    Limitations:
        ``files()`` returns only files on disk; a dry run has none (its
        would-be paths are in ``outcomes()``). In tier 3 of the product-set
        choice an SP3 day shared by the windows of main days with different
        products has two files; ``files("sp3")`` without ``day`` lists both.
    """

    project_name: str
    output_dir: Path
    reports: "list[ArchiveDownloadReport]" = field(default_factory=list)
    # Contract K-OUT: the product-set choice of the session (None when no
    # choice was made), whether linked products had to be mixed, and the SP3
    # boundary days used by files("sp3", day=...).
    product_sets: "ProductSetChoice | None" = None
    mixed_products: bool = False
    sp3_adjacent_days: int = 0

    def all_results(self) -> "list[DownloadResult]":
        return [result for report in self.reports for result in report.results]

    def by_status(self, status: str) -> "list[DownloadResult]":
        return [result for result in self.all_results() if result.status == status]

    @property
    def downloaded(self) -> "list[DownloadResult]":
        return self.by_status("downloaded")

    @property
    def errors(self) -> "list[DownloadResult]":
        return [result for result in self.all_results() if result.status in {"error", "auth_missing"}]

    @property
    def missing(self) -> "list[DownloadResult]":
        return self.by_status("missing")

    @property
    def skipped_existing(self) -> "list[DownloadResult]":
        return self.by_status("skipped_existing")

    @property
    def dry_runs(self) -> "list[DownloadResult]":
        return self.by_status("dry_run")

    @property
    def unavailable(self) -> "list[UnavailableRequest]":
        """Requested items no selected source provides, in day order."""
        return [item for report in self.reports for item in report.unavailable]

    def outcomes(self) -> "list[RequestOutcome]":
        """One :class:`RequestOutcome` per requested (type, day[, station]), in day order.

        Built from the engine request groups (SP3 boundary days included)
        and the ``unavailable`` entries of each day. Within a day the order
        is the engine's group order, then the unavailable entries. An SP3
        day fetched for two products (``product_sets.extra``) has one row
        per product. When no candidate succeeded, the last one tried
        decides; earlier ``error`` and ``auth_missing`` rows are named in
        ``message`` (all rows are in ``tried``).
        """
        rows: "list[RequestOutcome]" = []
        for report in self.reports:
            groups: dict = {}
            for result in report.results:
                candidate = result.candidate
                key = (candidate.data_type, candidate.day, candidate.request or candidate.station)
                groups.setdefault(key, []).append(result)
            for (data_type, day, station), results in groups.items():
                final = next((r for r in results if r.status in _FINAL_STATUSES), results[-1])
                message = final.message
                if final.status not in _FINAL_STATUSES:
                    # The last row decides (K-OUT); an earlier error or missing
                    # authentication is the more useful hint, so it is named too.
                    earlier = [
                        f"{r.candidate.source} {r.status}: {r.message}"
                        for r in results[:-1]
                        if r.status in ("error", "auth_missing")
                    ]
                    if earlier:
                        message = f"{message} (also tried: {'; '.join(earlier)})"
                rows.append(
                    RequestOutcome(
                        data_type=data_type,
                        day=day,
                        station=station,
                        status=final.status,
                        path=final.path if final.status in _FINAL_STATUSES else None,
                        product=final.candidate.product,
                        source=final.candidate.source,
                        url=final.url,
                        message=message,
                        tried=tuple(results),
                    )
                )
            for item in report.unavailable:
                rows.append(
                    RequestOutcome(
                        data_type=item.data_type,
                        day=item.day,
                        station=item.station,
                        status="unavailable",
                        message=item.reason,
                    )
                )
        return sorted(rows, key=lambda row: row.day)

    def files(self, data_type: str, *, day: "DateLike | None" = None, station: "str | None" = None) -> "list[Path]":
        """Local paths of the satisfied requests of one type, in day order.

        Only ``downloaded`` and ``skipped_existing`` outcomes count.
        ``data_type`` accepts the friendly names of ``PRODUCT_TYPE_ALIASES``.
        ``day`` (a ``date`` or ``datetime``; a datetime with a time zone is
        converted to UTC first, a naive one is UTC, as for
        ``DownloadConfig.dates``) selects one day; for ``"sp3"`` it selects
        the window ``day - sp3_adjacent_days ... day + sp3_adjacent_days``
        that orbit interpolation for that day needs. When a day of that
        window has files of two products (tier 3 of the product-set choice),
        the one of the product chosen for ``day`` is returned, so the window
        comes from one solution. ``station`` selects one station
        (case-insensitive) for observations.

        Raises:
            ValueError: unknown ``data_type``.
            TypeError: ``day`` is not a date/datetime, or ``station`` is not
                a string.
        """
        key = PRODUCT_TYPE_ALIASES.get(str(data_type).strip().lower())
        if key is None:
            raise ValueError(
                f"Unknown product type {data_type!r}. Supported (case-insensitive): "
                f"{sorted(set(PRODUCT_TYPE_ALIASES))}"
            )
        if isinstance(day, datetime):
            if day.tzinfo is not None and day.utcoffset() is not None:
                day = day.astimezone(timezone.utc)
            day = day.date()
        elif day is not None and not isinstance(day, date):
            raise TypeError(f"day must be a date or datetime, got {type(day).__name__}: {day!r}")
        if station is not None and not isinstance(station, str):
            raise TypeError(f"station must be a string, got {type(station).__name__}: {station!r}")
        window = self.sp3_adjacent_days if key == "sp3" else 0
        wanted_station = station.strip().upper() if station is not None else None
        selected = []
        for outcome in self.outcomes():
            if outcome.data_type != key or outcome.status not in SATISFIED_STATUSES:
                continue
            if day is not None and abs((outcome.day - day).days) > window:
                continue
            if wanted_station is not None and (outcome.station or "").upper() != wanted_station:
                continue
            selected.append(outcome)
        if day is not None and self.product_sets is not None:
            selected = _one_product_per_day(selected, self.product_sets.product_for(key, day))
        return [outcome.path for outcome in selected]

    def files_by_type(self) -> "dict[str, list[Path]]":
        """``files()`` of every data type that has a satisfied request."""
        result: "dict[str, list[Path]]" = {}
        for outcome in self.outcomes():
            if outcome.status in SATISFIED_STATUSES:
                result.setdefault(outcome.data_type, []).append(outcome.path)
        return result

    def summary(self) -> dict:
        """Status counts of the engine rows, plus ``requests`` and ``unavailable``.

        ``candidates`` is the number of generated candidates; every engine
        status that occurred has its row count; ``requests`` is the number
        of requested items (``len(outcomes())``) and ``unavailable`` the
        number of requested items no selected source provides.
        """
        counts: dict = {"candidates": 0}
        for report in self.reports:
            for key, value in report.summary().items():
                counts[key] = counts.get(key, 0) + value
        counts["requests"] = len(self.outcomes())
        counts["unavailable"] = len(self.unavailable)
        return counts


def _one_product_per_day(outcomes: "list[RequestOutcome]", product: "str | None") -> "list[RequestOutcome]":
    """Keep one outcome per (day, station): the one of ``product`` when a day has several."""
    by_key: dict = {}
    for outcome in outcomes:
        by_key.setdefault((outcome.day, outcome.station), []).append(outcome)
    kept = []
    for outcome in outcomes:
        group = by_key[(outcome.day, outcome.station)]
        if len(group) == 1:
            kept.append(outcome)
            continue
        preferred = next((o for o in group if o.product == product), group[0])
        if outcome is preferred:
            kept.append(outcome)
    return kept


class _SessionTransport:
    """The transport of one ``DownloadSession.run`` call.

    Wraps the injected or built transport: ``exists()`` answers are cached
    per URL (positive ones, and negative ones for HTTP 404), so the
    product-set probes and the engine never ask twice; ``download()`` of a
    URL known to be missing raises ``RemoteFileMissing`` without a request;
    ``has_credentials()`` and any other attribute pass through. Thread-safe.
    """

    def __init__(self, transport: DownloadTransport) -> None:
        self.transport = transport
        self._lock = threading.Lock()
        self._exists: "dict[str, URLAvailability]" = {}

    def has_credentials(self, candidate: DownloadCandidate) -> bool:
        return self.transport.has_credentials(candidate)

    def exists(self, candidate: DownloadCandidate) -> URLAvailability:
        with self._lock:
            cached = self._exists.get(candidate.url)
        if cached is not None:
            return cached
        answer = self.transport.exists(candidate)
        if answer.exists or answer.http_status == 404:
            with self._lock:
                self._exists.setdefault(candidate.url, answer)
        return answer

    def download(self, candidate: DownloadCandidate, destination: Path) -> int:
        with self._lock:
            cached = self._exists.get(candidate.url)
        if cached is not None and not cached.exists:
            raise RemoteFileMissing(
                f"Remote file not found (HTTP 404 in this session's availability check): {candidate.url}"
            )
        return self.transport.download(candidate, destination)

    def __getattr__(self, name: str):
        return getattr(self.transport, name)


class DownloadSession:
    """High-level orchestrator for one GNSS archive download session.

    Status:
        Active public session class and, together with ``DownloadConfig``,
        the recommended entry point for GNSS archive downloads. The former
        ``ProjectDownloader`` name is a deprecated alias of this class.

    Purpose:
        Expands a ``DownloadConfig`` into one low-level ``GNSSDownloadConfig``
        request per calendar day, runs each through ``GNSSDownloader`` with a
        shared transport and aggregates the per-day ``ArchiveDownloadReport``
        objects into a ``DownloadReport``.

    Config routing:
        ``run()`` first calls ``config.validate_for_run()``, so a
        configuration changed after construction is checked again. The
        session reads ``DownloadConfig`` but does not define configuration
        defaults itself. Every requested date receives all requested product
        types; ``"sp3"`` is additionally requested for ``sp3_adjacent_days``
        days before and after each requested date. Items that
        ``DownloadConfig`` found unavailable from the selected sources are
        not sent to the engine; they become ``"unavailable"`` outcomes on
        each requested date. Stations, sources, RINEX preference, overwrite,
        dry-run and availability switches, workers, timeout, retries,
        verbosity and station aliases are forwarded unchanged to every
        per-day engine request; product preferences are forwarded with the
        product-set choice applied (below).

    Transport and credentials:
        Explicit credentials (token, username/password or credentials file)
        are resolved at the start of ``run()`` only when ``"cddis"`` is a
        source and no transport was injected: a username/password or a
        credentials file costs one Earthdata token exchange per ``run()``,
        and none when CDDIS is not used. Without an injected ``transport``,
        ``run()`` builds one ``RequestsDownloadTransport`` from those
        credentials, ``timeout`` and ``retries``, shares it across all days
        and closes it at the end. An injected transport (any object
        implementing ``DownloadTransport``) is used as-is and not closed;
        explicit credentials are then ignored with a ``UserWarning``. Within
        one run, ``exists()`` answers are cached per URL, and a file known
        to be missing (HTTP 404) is not requested again.

    Product sets:
        Orbits, clocks, OSB and ORBEX (sp3, clk, bia, att) come from one
        analysis-centre solution for the whole project, SP3 boundary days
        included (:mod:`gnx_py.downloaders.product_sets`). For
        these types the product beats the archive order: the first product
        of the effective list that has the complete set is used at every
        source, COD at BKG before GRG at CDDIS. If none is complete, the
        first one with every SP3/CLK file is used, the boundary days and
        ORBEX it lacks are reported missing and OSB may come from another
        centre (with a ``UserWarning``: PPP-AR is then disabled by
        ``PPPSession``). Mixing per day is the last resort, with one
        ``UserWarning`` and ``report.mixed_products=True``; an SP3 day in
        the windows of main days with different products is then fetched
        once per product, so each main day keeps an SP3 window of one
        solution. Only HTTP 404 (or a refusal) makes a file absent: a probe
        without a definite answer (timeout, unreachable host, 5xx after the
        retries) counts as available, so a passing network fault never
        switches the centre; it is logged as a WARNING and kept in
        ``report.product_sets.probe_failures``. The choice costs one
        availability probe per linked file and source asked, newest day
        first; a product that lacks a file stops being probed at that file
        (none in a dry run without availability checks, none for a valid
        local file). The download of a probed file needs no second probe.
        ERP and navigation are not linked; for other product types the
        archive order of ``sources`` decides before the product list.

    Warnings:
        ``run()`` creates ``output_dir`` and, unless ``dry_run=True``, writes
        files under it. Unexpected failures are re-raised as
        ``SessionExecutionError`` carrying project, output, product, source
        and date context. Per-file problems (missing remote files,
        authentication failures, invalid payloads) do not abort the session;
        they are reported as ``DownloadResult`` entries and outcomes instead.

    Limitations:
        Days are processed sequentially; ``max_workers`` parallelizes the
        availability probes and the request groups within one day. A chosen
        linked product that then fails to download at every source is an
        error for that file; the session never switches to another product
        for it. An injected transport is responsible for its own response
        validation.
    """

    def __init__(
        self,
        config: DownloadConfig,
        transport: DownloadTransport | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Bind the configuration, optional transport and logger.

        Args:
            config: ``DownloadConfig`` describing the session.
            transport: Optional ``DownloadTransport`` shared by all days. If
                ``None``, ``run()`` builds a ``RequestsDownloadTransport``.
            logger: Optional logger passed to the engine. Defaults to the
                engine's module logger (``gnx_py.downloaders.core``).

        Notes:
            Does not perform I/O; credential resolution, output-folder
            creation and transport construction happen in ``run()``.
        """
        self.config: DownloadConfig = config
        self.transport: DownloadTransport | None = transport
        self.logger: logging.Logger = logger or LOGGER

    @guarded_session_run("DownloadSession")
    def run(self) -> DownloadReport:
        """Execute the configured download session.

        Steps:
            1) ``config.validate_for_run()``.
            2) Resolve explicit credentials, only when ``"cddis"`` is a
               source and no transport was injected.
            3) Create ``output_dir``.
            4) Build the shared transport unless one was injected, and wrap
               it with the per-run ``exists()`` cache.
            5) Choose the product set of the linked types.
            6) Expand the requested dates into per-day data-type sets,
               including the SP3 boundary days, and run one
               ``GNSSDownloader`` per day with the chosen products (plus
               one per further product of a shared SP3 day in tier 3).
            7) Add the unavailable items and aggregate the report; close the
               transport the session built.

        Returns:
            DownloadReport: The per-day engine reports, in chronological
            order, with the product-set choice.

        Raises:
            SessionExecutionError: Wraps any unexpected exception, e.g. a
            configuration rejected by ``validate_for_run()``, a failed token
            exchange, or a transport error raised outside per-file download
            handling.
        """
        config = self.config
        config.validate_for_run()
        credentials = self._resolve_session_credentials()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        built = None
        if self.transport is not None:
            inner = self.transport
        else:
            inner = built = RequestsDownloadTransport(
                timeout=config.timeout,
                credentials=credentials,
                retries=config.retries,
            )
        try:
            transport = _SessionTransport(inner)
            choice = choose_product_sets(config, transport, self.logger)
            unavailable = list(config._unavailable)
            extra_by_date: "dict[date, dict[str, list[str]]]" = {}
            for data_type, day, product in choice.extra:
                extra_by_date.setdefault(day, {}).setdefault(product, []).append(data_type)
            requests_by_date = self._build_date_requests()
            stations = self._available_stations()
            reports = []
            for requested_date in sorted(requests_by_date):
                data_types = tuple(sorted(requests_by_date[requested_date]))
                if data_types:
                    day_config = self._build_day_config(
                        requested_date,
                        data_types=data_types,
                        credentials=credentials,
                        product_preferences=day_preferences(config, choice, requested_date, data_types),
                        stations=stations,
                    )
                    day_report = GNSSDownloader(day_config, transport=transport, logger=self.logger).download()
                else:
                    day_report = ArchiveDownloadReport(candidates=[], results=[])
                if requested_date in config.dates:
                    _add_unavailable(day_report, requested_date, unavailable)
                reports.append(day_report)
                # Tier 3: the same day from a further product (another main
                # day's SP3 window), as different files.
                for product, extra_types in extra_by_date.get(requested_date, {}).items():
                    extra_types = tuple(sorted(extra_types))
                    preferences = day_preferences(config, choice, requested_date, extra_types)
                    preferences.update({data_type: (product,) for data_type in extra_types})
                    extra_config = self._build_day_config(
                        requested_date,
                        data_types=extra_types,
                        credentials=credentials,
                        product_preferences=preferences,
                        stations=(),
                    )
                    reports.append(GNSSDownloader(extra_config, transport=transport, logger=self.logger).download())
        finally:
            if built is not None:
                close = getattr(built, "close", None)
                if callable(close):
                    close()

        report = DownloadReport(
            project_name=config.project_name,
            output_dir=config.output_dir,
            reports=reports,
            product_sets=choice,
            mixed_products=bool(choice.mixed),
            sp3_adjacent_days=config.sp3_adjacent_days,
        )
        self._log_navigation_products(report)
        return report

    def _resolve_session_credentials(self) -> "CDDISCredentials | None":
        """Credentials for the transport the session builds."""
        config = self.config
        explicit = any(
            value is not None and value != ""
            for value in (config.token, config.username, config.password, config.credentials_file)
        )
        if self.transport is not None:
            if explicit:
                _user_warning(
                    "credentials are ignored with an injected transport; the transport "
                    "handles authentication itself"
                )
            return None
        if "cddis" not in {str(source).lower() for source in config.sources}:
            return None
        return _resolve_credentials(config)

    def _available_stations(self) -> "tuple[str, ...]":
        """Requested stations minus those ``DownloadConfig`` found unavailable."""
        dropped = {
            station
            for data_type, station, _reason in self.config._unavailable
            if data_type == "obs" and station is not None
        }
        return tuple(station for station in self.config.stations if station not in dropped)

    def _build_date_requests(self) -> "dict[date, set]":
        """Map each date that needs to be fetched to the data types needed for it.

        SP3 is expanded to +/- ``sp3_adjacent_days`` around every requested
        date; every other product type applies only to the requested dates.
        Types that ``DownloadConfig`` found unavailable as a whole, and
        ``"obs"`` when every station is unavailable, are left out; a
        requested date keeps an (empty) entry so that its unavailable
        outcomes have a report.
        """
        dropped = {
            data_type
            for data_type, station, _reason in self.config._unavailable
            if station is None
        }
        if "obs" in self.config.product_types and not self._available_stations():
            dropped.add("obs")
        product_types = [data_type for data_type in self.config.product_types if data_type not in dropped]
        requests_by_date: "dict[date, set]" = {}
        for requested_date in self.config.dates:
            requests_by_date.setdefault(requested_date, set()).update(product_types)
            if "sp3" in product_types:
                for offset in range(1, self.config.sp3_adjacent_days + 1):
                    requests_by_date.setdefault(requested_date - timedelta(days=offset), set()).add("sp3")
                    requests_by_date.setdefault(requested_date + timedelta(days=offset), set()).add("sp3")
        return requests_by_date

    def _build_day_config(
        self,
        day: date,
        data_types: "tuple[str, ...]",
        credentials: "CDDISCredentials | None",
        product_preferences: "Mapping[str, Sequence[str]] | None" = None,
        stations: "Sequence[str] | None" = None,
    ) -> GNSSDownloadConfig:
        """Build the low-level engine request for one calendar day."""
        doy = day.timetuple().tm_yday
        return GNSSDownloadConfig(
            year=day.year,
            doy_start=doy,
            doy_end=doy,
            output_dir=self.config.output_dir,
            stations=self.config.stations if stations is None else stations,
            data_types=data_types,
            sources=self.config.sources,
            rinex_version_preference=self.config.rinex_version_preference,
            rinex_data_source_preference=self.config.rinex_data_source_preference,
            overwrite=self.config.overwrite,
            dry_run=self.config.dry_run,
            check_availability=self.config.check_availability,
            max_workers=self.config.max_workers,
            credentials=credentials,
            station_aliases=self.config.station_aliases,
            product_preferences=(
                self.config.product_preferences if product_preferences is None else product_preferences
            ),
            timeout=self.config.timeout,
            verbose=self.config.verbose,
            retries=self.config.retries,
        )

    def _log_navigation_products(self, report: DownloadReport) -> None:
        """INFO line when the navigation product differs between days."""
        products: dict = {}
        for outcome in report.outcomes():
            if outcome.data_type == "nav" and outcome.status in _FINAL_STATUSES and outcome.product:
                products[outcome.day] = outcome.product
        if len(set(products.values())) > 1:
            self.logger.info(
                "Navigation products differ between days: %s",
                ", ".join(f"{day.isoformat()} {product}" for day, product in sorted(products.items())),
            )


def _add_unavailable(day_report: ArchiveDownloadReport, day: date, unavailable) -> None:
    """Append the config-level unavailable items of a requested day, once each."""
    seen = {(item.data_type, item.day, item.station) for item in day_report.unavailable}
    for data_type, station, reason in unavailable:
        if (data_type, day, station) in seen:
            continue
        day_report.unavailable.append(UnavailableRequest(data_type=data_type, day=day, station=station, reason=reason))
        seen.add((data_type, day, station))
