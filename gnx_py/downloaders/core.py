"""Low-level GNSS archive download engine.

This module builds archive candidate URLs, checks their availability and
downloads them through a pluggable transport:

- ``GNSSDownloadConfig`` / ``GNSSDownloader``: one engine run in archive terms
  (year, day-of-year range, data-type and source keys), returning an
  ``ArchiveDownloadReport``.
- ``RequestsDownloadTransport``: the default transport implementing the
  ``DownloadTransport`` protocol (the legacy urllib transport was removed in v1.0.0).
- ``CDDISCredentials`` and the CDDIS/Earthdata error types.

It is the layer underneath the recommended public entry point,
``DownloadConfig`` + ``DownloadSession`` in :mod:`gnx_py.downloaders.config`,
which builds one ``GNSSDownloadConfig`` per requested day.
"""
from __future__ import annotations

import email.utils
import hashlib
import http.client
import logging
import netrc
import os
import secrets
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urljoin, urlparse

import requests
import urllib3

from . import catalog

if TYPE_CHECKING:
    from .manifest import DownloadManifest


LOGGER = logging.getLogger(__name__)

BKG_BASE_URL = "https://igs.bkg.bund.de/root_ftp/IGS"
CDDIS_BASE_URL = "https://cddis.nasa.gov/archive/gnss"
CODE_BASE_URL = "https://ftp.aiub.unibe.ch/CODE"
CODE_MGEX_BASE_URL = "https://ftp.aiub.unibe.ch/CODE_MGEX/CODE"
# BKG analysis-centre products (root_ftp/IGSac/products/WWWW/), tried after
# the IGS combined products of BKG_BASE_URL/products/WWWW/ (contract K-URL).
BKG_IGSAC_BASE_URL = "https://igs.bkg.bund.de/root_ftp/IGSac"
# BKG as the EUREF Permanent GNSS Network (EPN) data centre: daily station
# observations under root_ftp/EUREF/obs/YYYY/DDD/ (source "euref"; the EPN
# stations outside the IGS network are only there, bug-fix stage 13d).
EUREF_BASE_URL = "https://igs.bkg.bund.de/root_ftp/EUREF"
# First GPS week whose MGEX campaign products CDDIS keeps in products/WWWW/
# (the week of the switch to long OPS names, 27 November 2022); from this week
# products/mgex/WWWW/ does not exist (HTTP 404 for week 2300 in the network
# run of 2026-10-14, DL-040), so it is tried only for earlier weeks.
CDDIS_MGEX_MERGED_WEEK = 2238

SUPPORTED_SOURCES = ("bkg", "cddis", "code", "euref")
# "att" = ORBEX satellite attitude (``..._ATT.OBX``), tied to the orbit solution.
SUPPORTED_DATA_TYPES = ("obs", "nav", "sp3", "clk", "erp", "bia", "dcb", "ionex", "att")
SUPPORTED_RINEX_PREFERENCES = ("3", "4", "2")
# Data-source letter of RINEX 3/4 long observation names
# (``XXXXMRCCC_K_...``): R from a receiver, S from a data stream (e.g. the
# EPN stations of national RTK networks), U unknown.
SUPPORTED_RINEX_DATA_SOURCES = ("R", "S", "U")
# Letters tried by default, in order: the receiver file first, so a station
# that publishes it is found by the same first request as before stage 13d.
DEFAULT_RINEX_DATA_SOURCE_PREFERENCE = ("R", "S")

# Contract K-CAP: the data types each source can serve with the archive layout
# modelled by ``GNSSDownloader.build_candidates``. It must stay in sync with
# the candidate generator; see also ``unavailable_reason`` and
# ``station_reason``. CODE additionally serves only COD products (and COD
# IONEX), which ``unavailable_reason`` checks against the preference lists.
SOURCE_CAPABILITIES: Mapping[str, frozenset] = MappingProxyType({
    "bkg": frozenset({"obs", "nav", "sp3", "clk", "erp", "bia", "dcb"}),
    "cddis": frozenset(SUPPORTED_DATA_TYPES),
    "code": frozenset({"sp3", "clk", "erp", "bia", "dcb", "ionex", "att"}),
    "euref": frozenset({"obs"}),
})
# Why a (source, data type) pair is absent from SOURCE_CAPABILITIES.
_CAPABILITY_GAPS = {
    ("bkg", "ionex"): "the BKG IONEX layout is not modelled",
    ("bkg", "att"): "no ORBEX attitude directory of BKG is modelled",
    ("code", "obs"): "CODE serves products only, no station observations",
    ("code", "nav"): "CODE serves no broadcast navigation files",
    **{
        ("euref", data_type): "the BKG EUREF archive is modelled for station observations only"
        for data_type in SUPPORTED_DATA_TYPES
        if data_type != "obs"
    },
}
# The only analysis centre whose products the CODE archive serves.
_CODE_ANALYSIS_CENTRE = "COD"

# Response-validation thresholds and file magic bytes (see RequestsDownloadTransport).
MIN_FILE_SIZE_BYTES = 100
GZIP_MAGIC = b"\x1f\x8b"
COMPRESS_MAGIC = b"\x1f\x9d"

# Contract K-TRANS: base of the retry backoff of RequestsDownloadTransport in
# seconds (sleeps of RETRY_BACKOFF_S * 2**k, k = 0, 1, ...); tests patch it to 0.
RETRY_BACKOFF_S = 1.0

# Message shown when a CDDIS candidate is requested without an Earthdata token.
_MISSING_TOKEN_MESSAGE = (
    "CDDIS requires a NASA Earthdata bearer token. Set the EARTHDATA_TOKEN "
    "environment variable, or pass CDDISCredentials(token=...)."
)

# Default analysis centers tried for IONEX/GIM products, in priority order.
DEFAULT_IONEX_PRODUCTS = ("COD", "IGS", "ESA", "JPL", "UPC")

# Default product lists per data type (decision D2: the linked types sp3, clk,
# bia and att share their leading solutions). The file names per product come
# from :mod:`gnx_py.downloaders.catalog`.
DEFAULT_PRODUCT_PREFERENCES = {
    "sp3": ("COD0MGXFIN", "GRG0MGXFIN", "GFZ0MGXRAP", "COD0OPSFIN", "IGS0OPSFIN"),
    "clk": ("COD0MGXFIN", "GRG0MGXFIN", "GFZ0MGXRAP", "COD0OPSFIN", "IGS0OPSFIN"),
    "erp": ("COD0MGXFIN", "COD0OPSFIN", "IGS0OPSFIN", "GFZ0MGXRAP"),
    "bia": ("COD0MGXFIN", "GRG0MGXFIN", "GFZ0MGXRAP", "COD0OPSFIN", "CAS0MGXRAP"),
    # Author decision D9 (bug-fix stage 13d): the rapid DCB.BIA successors of
    # the MGX DCB.BSX files (which end at 2024-048); in the 2024 CDDIS listing
    # CAS0OPSRAP and GFZ0OPSRAP together cover every day. Bug-fix stage 13e
    # (DL-NQ-17): the multi-GNSS CAS1OPSRAP before GFZ0OPSRAP, whose file holds
    # one classic DSB pair per system only.
    "dcb": ("CAS0OPSRAP", "CAS1OPSRAP", "GFZ0OPSRAP", "CAS0MGXRAP", "DLR0OPSFIN"),
    "att": ("COD0MGXFIN", "GRG0MGXFIN", "GFZ0MGXRAP"),
}

# Broadcast navigation products tried when product_preferences has no "nav"
# key, per entry of rinex_version_preference (vocabulary: catalog.NAV_PRODUCTS).
DEFAULT_NAV_BY_RINEX = {"3": "BRDC00IGS", "4": "BRD400DLR", "2": "BRDC"}

# Keep this intentionally tiny. Users can pass station_aliases for their network.
DEFAULT_STATION_ALIASES = {
    "BRUX": ("BRUX00BEL",),
}


class CDDISAuthError(RuntimeError):
    """Raised when CDDIS authentication is missing or invalid.

    This covers a missing Earthdata bearer token, an HTTP 401/403 response, or
    an HTML login-redirect page returned in place of the requested file.
    """


class CDDISDownloadError(RuntimeError):
    """Raised when a file cannot be fetched or the fetched file is not valid.

    Covers HTTP errors other than 404 (:class:`RemoteFileMissing`) and the
    authentication cases (:class:`CDDISAuthError`), an unreachable host,
    connection failures, timeouts and 5xx/429 answers after the last retry,
    a cut transfer (fewer bytes than ``Content-Length``), a redirect from
    HTTPS to plain HTTP, an HTML page in place of the file on a source
    without login, and a file that fails validation: smaller than
    :data:`MIN_FILE_SIZE_BYTES`, a container that does not match the name,
    an incomplete or corrupt gzip/bzip2 stream, or a missing type header.
    Errors raised by :class:`RequestsDownloadTransport` carry
    ``http_status`` and ``attempts`` attributes.
    """


class RemoteFileMissing(CDDISDownloadError):
    """Raised by a transport's ``download()`` when the server answers HTTP 404.

    The engine reports it as status ``"missing"``. A custom
    transport that signals 404 with another exception gets ``"error"``.
    """


# Backwards-compatible alias for the previously public exception name.
CDDISAuthenticationError = CDDISAuthError


@dataclass(frozen=True)
class CDDISCredentials:
    """Earthdata credentials used for CDDIS HTTPS archive access.

    Credentials can be supplied directly at runtime, read from environment
    variables, or discovered from a ``.netrc`` entry for
    ``urs.earthdata.nasa.gov``. Passwords are deliberately excluded from repr.
    """

    username: str | None = None
    password: str | None = field(default=None, repr=False)
    netrc_path: str | Path | None = None
    token: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "CDDISCredentials | None":
        username = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        if username and password:
            return cls(username=username, password=password)
        return None

    @classmethod
    def from_netrc(cls, path: str | Path | None = None) -> "CDDISCredentials | None":
        """Read the ``urs.earthdata.nasa.gov`` entry of a netrc file.

        With an explicit ``path`` a missing file raises
        ``FileNotFoundError`` and a file that cannot be parsed raises
        ``ValueError``, both naming the path. The parser's own error, whose
        text may quote a token of the file (possibly the password), is
        neither shown nor attached as ``__cause__`` or ``__context__``.
        Without ``path`` the
        default ``~/.netrc`` is optional: a missing or unreadable file gives
        ``None`` and one DEBUG line. A file without the Earthdata entry gives
        ``None`` in both cases.
        """
        if path is not None:
            missing_errno = parse_error_line = None
            try:
                auth = netrc.netrc(os.fspath(path))
            except FileNotFoundError as exc:
                missing_errno = exc.errno
            except netrc.NetrcParseError as exc:
                parse_error_line = exc.lineno
            # Raised outside the ``except`` blocks so that no exception
            # context (with the parser's text) is attached.
            if missing_errno is not None:
                raise FileNotFoundError(missing_errno, "netrc file not found", str(path))
            if parse_error_line is not None:
                raise ValueError(
                    f"Cannot parse the netrc file {path} (line {parse_error_line}); expected lines such as "
                    "'machine urs.earthdata.nasa.gov login <user> password <password>'"
                )
        else:
            try:
                auth = netrc.netrc()
            except (FileNotFoundError, netrc.NetrcParseError, OSError) as exc:
                LOGGER.debug("Default ~/.netrc not used (%s).", type(exc).__name__)
                return None

        login = auth.authenticators("urs.earthdata.nasa.gov")
        if not login:
            return None
        username, _, password = login
        if not username or not password:
            return None
        return cls(username=username, password=password, netrc_path=path)

    def resolved(self) -> "CDDISCredentials | None":
        """Return usable credentials: these, an explicit netrc, the environment or ``~/.netrc``.

        A token or a username/password pair returns ``self``. An explicit
        ``netrc_path`` is read with :meth:`from_netrc`, so a missing or
        corrupt file raises instead of returning ``None``. Otherwise the
        ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD`` variables, then the
        optional default ``~/.netrc``, are tried.
        """
        if self.token or (self.username and self.password):
            return self
        if self.netrc_path is not None:
            return self.from_netrc(self.netrc_path)
        return self.from_environment() or self.from_netrc()

    @classmethod
    def from_password_exchange(
        cls,
        username: str,
        password: str,
        *,
        session: "requests.Session | None" = None,
        timeout: float = 30.0,
    ) -> "CDDISCredentials":
        """Exchange an Earthdata username/password for a bearer token.

        This is a thin convenience wrapper around
        :func:`gnx_py.downloaders.earthdata_auth.fetch_earthdata_token`,
        imported lazily here so that everyday use of this module (and of the
        download pipeline in general) does not need the credential-exchange
        code path at all. The password is used only for this one exchange and
        is not retained on the returned object; only the resulting bearer
        token is kept.

        Raises :class:`CDDISAuthError` if no token could be obtained. Never
        logs or includes the password in the exception message.
        """
        from .earthdata_auth import fetch_earthdata_token  # local import by design

        token = fetch_earthdata_token(username, password, session=session, timeout=timeout)
        return cls(username=username, token=token)


@dataclass
class GNSSDownloadConfig:
    """Configuration for one run of the low-level archive download engine.

    Status:
        Low-level engine API. ``GNSSDownloadConfig`` describes a request in
        archive terms and is consumed by ``GNSSDownloader``. The recommended
        public entry point is ``DownloadConfig`` + ``DownloadSession``
        (:mod:`gnx_py.downloaders.config`), which builds one of these objects
        per requested day; use this class directly only when you need
        year/day-of-year control over a single engine run.

    Main parameters:
        year, doy_start, doy_end:
            Inclusive day-of-year range within one calendar year.
        output_dir:
            Root of the local archive tree. Files are stored under
            ``output_dir/YYYY/DDD/source/type/filename``.
        stations, data_types, sources, rinex_version_preference:
            Request vocabulary. Data types: the keys of
            ``SUPPORTED_DATA_TYPES`` (``obs``, ``nav``, ``sp3``, ``clk``,
            ``erp``, ``bia``, ``dcb``, ``att``, ``ionex``). Sources: ``bkg``,
            ``cddis``, ``code``, ``euref`` (the BKG EUREF/EPN archive,
            observations only). A bare string counts as one item; duplicates
            are dropped; unsupported or empty values raise ``ValueError``.
            ``"obs"`` needs at least one station, and a 4-character station
            needs a 9-character id (``station_aliases`` or the built-in
            aliases) unless ``"2"`` is in ``rinex_version_preference``. The
            default ``data_types`` include ``"obs"``: without stations, pass
            ``data_types`` explicitly, e.g. ``("nav", "sp3")``.
        rinex_data_source_preference:
            Data-source letters of the RINEX 3/4 long observation names, in
            trial order: ``"R"`` (receiver), ``"S"`` (data stream), ``"U"``
            (unknown); default ``("R", "S")``. Normalized like
            ``DownloadConfig.rinex_data_source_preference``.
        credentials:
            Optional ``CDDISCredentials`` used when ``GNSSDownloader`` builds
            its default ``RequestsDownloadTransport``; that transport only
            uses the bearer token (falling back to ``EARTHDATA_TOKEN``).
        overwrite, dry_run, check_availability, max_workers, timeout, verbose, retries:
            Execution switches with the same meaning as in ``DownloadConfig``
            (``timeout=None`` waits without a limit).
        station_aliases, product_preferences:
            4-to-9-character station id aliases and per-data-type product
            priority overrides, normalized like in ``DownloadConfig``
            (``product_preferences`` keys become engine keys).

    Validation:
        ``__post_init__`` applies the validators of
        :mod:`gnx_py.downloaders.validation` that ``DownloadConfig`` uses, so
        both raise the same ``TypeError``/``ValueError`` messages, plus the
        year/day-of-year range check. ``output_dir`` stays a plain field.
        Data types that no selected source serves are not an error here; the
        engine lists them in ``ArchiveDownloadReport.unavailable``.
    """

    year: int
    doy_start: int
    doy_end: int
    output_dir: str | Path
    stations: Sequence[str] = ()
    data_types: Sequence[str] = ("obs", "nav", "sp3")
    sources: Sequence[str] = ("bkg", "cddis", "code")
    rinex_version_preference: Sequence[str] = ("3", "4", "2")
    overwrite: bool = False
    dry_run: bool = False
    verbose: bool = False
    max_workers: int | None = None
    credentials: CDDISCredentials | None = field(default=None, repr=False)
    station_aliases: Mapping[str, str | Sequence[str]] = field(default_factory=dict)
    product_preferences: Mapping[str, Sequence[str]] = field(default_factory=dict)
    timeout: float | None = 30.0
    check_availability: bool = True
    # Extra attempts per file for transient failures (decision D3); passed to
    # the RequestsDownloadTransport the engine builds.
    retries: int = 2
    # Data-source letters of RINEX 3/4 long observation names (bug-fix stage
    # 13d): ``_R_`` receiver files, then ``_S_`` stream files.
    rinex_data_source_preference: Sequence[str] = DEFAULT_RINEX_DATA_SOURCE_PREFERENCE

    def __post_init__(self) -> None:
        # The same validators as DownloadConfig (contract K-VAL), imported
        # here because validation imports this module at module level.
        from . import validation as v

        if not isinstance(self.output_dir, (str, os.PathLike)):
            raise TypeError(
                f"output_dir must be a str or a path, got {type(self.output_dir).__name__}: {self.output_dir!r}"
            )
        self.output_dir = Path(self.output_dir)

        # A bare string such as ``sources="cddis"`` or ``data_types="obs"`` is
        # one item, never iterated character by character.
        self.data_types = v.normalize_data_types(self.data_types)
        self.sources = v.normalize_sources(self.sources)
        self.rinex_version_preference = v.normalize_rinex_preference(self.rinex_version_preference)
        self.rinex_data_source_preference = v.normalize_rinex_data_source_preference(
            self.rinex_data_source_preference
        )
        self.station_aliases = v.normalize_station_aliases(self.station_aliases)
        self.stations = v.normalize_stations(self.stations, self.station_aliases)
        self.product_preferences = v.normalize_product_preferences(self.product_preferences)
        for name in ("overwrite", "dry_run", "verbose", "check_availability"):
            setattr(self, name, v.check_bool(getattr(self, name), name))
        self.max_workers = v.check_max_workers(self.max_workers)
        v.check_timeout(self.timeout)
        self.retries = v.check_retries(self.retries)
        if self.credentials is not None and not isinstance(self.credentials, CDDISCredentials):
            raise TypeError(
                f"credentials must be a CDDISCredentials or None, got {type(self.credentials).__name__}"
            )

        # Validate the date range once during config construction.
        list(iter_doy_range(self.year, self.doy_start, self.doy_end))

        # Input errors of the request itself (DL-001): observations need
        # stations, and every station needs at least one file name. Items
        # the selected sources do not serve are reported by the engine as
        # ``ArchiveDownloadReport.unavailable`` instead.
        if "obs" in self.data_types:
            if not self.stations:
                raise ValueError(
                    'data_types includes "obs" but no stations were given (the default data_types are '
                    '("obs", "nav", "sp3")); pass stations, or data_types without "obs", e.g. ("nav", "sp3").'
                )
            for station in self.stations:
                variants = station_variants(station, self.station_aliases)
                reason = station_reason(variants, self.rinex_version_preference)
                if reason is not None and not variants.rinex3_ids:
                    raise ValueError(reason)


@dataclass(frozen=True)
class StationVariants:
    """Normalized station identifiers for legacy and long RINEX filenames."""

    original: str
    legacy4: str
    rinex3_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DownloadCandidate:
    """One archive URL that may satisfy part of a download request."""

    source: str
    data_type: str
    url: str
    local_path: Path
    year: int
    doy: int
    day: date
    priority: int
    station: str | None = None
    rinex_version: str | None = None
    product: str | None = None
    requires_auth: bool = False
    notes: tuple[str, ...] = ()
    # Contract K-CAND: the normalised station id as the user requested it
    # (obs only). Request groups are keyed by ``request or station``.
    request: str | None = None

    @property
    def filename(self) -> str:
        return self.local_path.name

    @property
    def group_key(self) -> tuple[str, int, int, str | None]:
        return (self.data_type, self.year, self.doy, self.request or self.station)


@dataclass(frozen=True)
class URLAvailability:
    """Result of a lightweight URL availability check."""

    exists: bool
    message: str = ""
    http_status: int | None = None
    # The server asked for authentication (401/403 or a login page) on a
    # source that requires it: the engine reports "auth_missing".
    auth_failed: bool = False
    # HTTP attempts made (retries included); 0 when no request was sent.
    attempts: int = 1


@dataclass(frozen=True)
class DownloadResult:
    """Outcome for one candidate that was checked, skipped, or downloaded."""

    candidate: DownloadCandidate
    status: str
    message: str = ""
    path: Path | None = None
    url: str | None = None
    http_status: int | None = None
    bytes_written: int | None = None
    # Contract K-ENG: HTTP attempts made for this candidate (retries
    # included; 0 when no request was sent, e.g. skipped_existing) and the
    # sha256 of the file (downloaded, or known for skipped_existing).
    attempts: int = 1
    sha256: str | None = None


@dataclass(frozen=True)
class UnavailableRequest:
    """A requested item that no selected source can provide, with the reason.

    ``station`` is ``None`` except for observation requests. It is
    collected in ``ArchiveDownloadReport.unavailable`` and reported by the
    session as outcomes with status ``"unavailable"``.
    """

    data_type: str
    day: date
    station: str | None
    reason: str


@dataclass
class ArchiveDownloadReport:
    """Result of one low-level engine run (:meth:`GNSSDownloader.download`).

    Formerly named ``DownloadReport``; that name now refers to the
    session-level report returned by ``DownloadSession.run``, which
    aggregates one ``ArchiveDownloadReport`` per requested day. The old
    ``gnx_py.downloaders.core.DownloadReport`` spelling still resolves to this
    class with a ``DeprecationWarning``.
    """

    candidates: list[DownloadCandidate]
    results: list[DownloadResult]
    # Requested items for which no candidate exists (K-ENG, DL-001).
    unavailable: list[UnavailableRequest] = field(default_factory=list)

    def by_status(self, status: str) -> list[DownloadResult]:
        return [result for result in self.results if result.status == status]

    @property
    def downloaded(self) -> list[DownloadResult]:
        return self.by_status("downloaded")

    @property
    def skipped_existing(self) -> list[DownloadResult]:
        return self.by_status("skipped_existing")

    @property
    def dry_runs(self) -> list[DownloadResult]:
        return self.by_status("dry_run")

    @property
    def missing(self) -> list[DownloadResult]:
        return self.by_status("missing")

    @property
    def errors(self) -> list[DownloadResult]:
        return [
            result
            for result in self.results
            if result.status in {"error", "auth_missing"}
        ]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {"candidates": len(self.candidates)}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts


class DownloadTransport(Protocol):
    def exists(self, candidate: DownloadCandidate) -> URLAvailability:
        ...

    def download(self, candidate: DownloadCandidate, destination: Path) -> int:
        ...

    def has_credentials(self, candidate: DownloadCandidate) -> bool:
        ...


def _resolve_bearer_token(
    token: str | None,
    credentials: CDDISCredentials | None,
) -> str | None:
    """Resolve the Earthdata bearer token from arg, credentials, then env."""

    if token:
        return token
    if credentials is not None and getattr(credentials, "token", None):
        return credentials.token
    return os.environ.get("EARTHDATA_TOKEN")


def _looks_like_markup(head: bytes) -> bool:
    stripped = head.lstrip().lower()
    if stripped.startswith(b"\xef\xbb\xbf"):
        stripped = stripped[3:].lstrip()
    return (
        stripped.startswith(b"<!doctype")
        or stripped.startswith(b"<html")
        or stripped.startswith(b"<?xml")
    )


# File-name suffix -> container format (as :func:`gnx_py.compression.detect_format` names it).
_COMPRESSED_SUFFIXES = {".gz": "gzip", ".z": "compress", ".bz2": "bzip2", ".zip": "zip"}
_FORMAT_LABELS = {"gzip": "gzip", "compress": "Unix compress (.Z)", "bzip2": "bzip2", "zip": "zip", "plain": "plain"}
_MAGIC_TEXT = {"gzip": "1f 8b", "compress": "1f 9d", "bzip2": "42 5a 68 ('BZh')", "zip": "50 4b ('PK')"}

# Content signature per data type (decision D11): (label, markers looked for in the first decompressed KB).
# SP3 is checked on its first line ("#a" ... "#d"); ERP has no signature.
_SIGNATURE_HEAD_BYTES = 1024
_SIGNATURES: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType({
    "sp3": ("SP3", ()),
    "clk": ("RINEX clock", ("RINEX VERSION / TYPE",)),
    "obs": ("RINEX observation", ("RINEX VERSION / TYPE", "CRINEX VERS   / TYPE")),
    "nav": ("RINEX navigation", ("RINEX VERSION / TYPE",)),
    "ionex": ("IONEX", ("IONEX VERSION / TYPE",)),
    "bia": ("Bias-SINEX", ("%=BIA",)),
    "dcb": ("Bias-SINEX", ("%=BIA",)),
    "att": ("ORBEX", ("%=ORBEX",)),
})


def _expected_format(filename: str) -> str:
    """Container format implied by a file name: ``gzip``, ``compress``, ``bzip2``, ``zip`` or ``plain``."""
    low = filename.lower()
    for suffix, fmt in _COMPRESSED_SUFFIXES.items():
        if low.endswith(suffix):
            return fmt
    return "plain"


def _content_head(path: Path, fmt: str, limit: int = _SIGNATURE_HEAD_BYTES) -> bytes:
    """The first ``limit`` decompressed bytes of ``path`` (container already verified)."""
    import bz2
    import zipfile
    import zlib

    with path.open("rb") as handle:
        if fmt == "plain":
            data = handle.read(limit)
        elif fmt in ("gzip", "bzip2"):
            # Feed input until the first ``limit`` bytes come out (a bzip2 block needs up to ~900 KB).
            decoder = zlib.decompressobj(31) if fmt == "gzip" else bz2.BZ2Decompressor()
            data = b""
            while len(data) < limit and not decoder.eof:
                chunk = handle.read(1 << 16)
                if not chunk:
                    break
                data += decoder.decompress(chunk, limit - len(data))
                if fmt == "gzip" and decoder.unconsumed_tail and len(data) >= limit:
                    break
        elif fmt == "compress":
            # LZW has no streaming decoder here; a prefix of the input decodes to a prefix of the output.
            import ncompress

            data = ncompress.decompress(handle.read(1 << 14))[:limit]
        else:  # zip, exactly one member (checked by verify_stream)
            with zipfile.ZipFile(handle) as archive:
                member = [m for m in archive.infolist() if not m.is_dir()][0]
                with archive.open(member) as inner:
                    data = inner.read(limit)
    return data[3:] if data.startswith(b"\xef\xbb\xbf") else data


def _check_signature(path: Path, candidate: "DownloadCandidate", fmt: str) -> None:
    """Raise :class:`CDDISDownloadError` unless the head of ``path`` carries its data type's header."""
    spec = _SIGNATURES.get(candidate.data_type)
    if spec is None:
        return
    label, markers = spec
    try:
        head = _content_head(path, fmt)
    except Exception as exc:  # noqa: BLE001 - any decoder failure means the content is unusable
        raise CDDISDownloadError(
            f"{candidate.filename}: the start of the {_FORMAT_LABELS.get(fmt, fmt)} data cannot be decoded ({exc})"
        ) from exc
    text = head.decode("latin-1")
    lines = text.splitlines()
    first_line = lines[0] if lines else ""
    if candidate.data_type == "sp3":
        ok = len(first_line) >= 2 and first_line[0] == "#" and first_line[1] in "abcd"
    else:
        ok = any(marker in text for marker in markers)
    if not ok:
        raise CDDISDownloadError(
            f"{candidate.filename} is not a {label} file (first line: {first_line[:80]!r}). URL: {candidate.url}"
        )


def _validate_downloaded_file(
    path: Path,
    candidate: "DownloadCandidate",
    http_status: int | None = None,
    *,
    fresh: bool = True,
) -> list[str]:
    """Validate a file for ``candidate``: size, markup, container, integrity and content signature.

    Used for freshly written temp files (``fresh=True``) and for files already on disk
    (:func:`check_local_file`, ``fresh=False``). The checks, in order:

    * at least :data:`MIN_FILE_SIZE_BYTES` bytes;
    * no HTML/XML page: for a source that requires authentication a fresh page raises
      :class:`CDDISAuthError` (an Earthdata login redirect), elsewhere :class:`CDDISDownloadError`
      "server returned an HTML page instead of <file>" with the real HTTP status;
    * a compressed name (``.gz``, ``.Z``, ``.bz2``, ``.zip``) must hold that container, and an uncompressed name
      must not hold compressed bytes;
    * the whole container decompresses (:func:`gnx_py.compression.verify_stream`: every gzip member to its end
      marker with CRC32 and length; ``.Z`` has no end marker, so only its header is checked);
    * the first decompressed KB carries the data type's header (none for ERP).

    Returns the readable notes of the integrity check (stray bytes after the gzip data, which are tolerated).
    Raises :class:`CDDISDownloadError` (or :class:`CDDISAuthError`, see above).
    """
    from ..io.compression import CompressedFileError, detect_format, verify_stream

    name = candidate.filename
    size = path.stat().st_size
    if size < MIN_FILE_SIZE_BYTES:
        what = "Downloaded file" if fresh else f"Local file {path}"
        raise CDDISDownloadError(
            f"{what} is only {size} bytes (< {MIN_FILE_SIZE_BYTES}); it is not a real product. URL: {candidate.url}"
        )

    with path.open("rb") as handle:
        head = handle.read(16)

    if _looks_like_markup(head):
        if not fresh:
            raise CDDISDownloadError(f"Local file {path} holds an HTML/XML page, not {name}.")
        if candidate.requires_auth:
            raise CDDISAuthError(
                "Downloaded content is an HTML/XML page (an Earthdata login redirect) instead of "
                f"{name}. Check EARTHDATA_TOKEN. URL: {candidate.url}"
            )
        raise CDDISDownloadError(
            f"server returned an HTML page instead of {name} (HTTP {http_status}). URL: {candidate.url}"
        )

    expected = _expected_format(name)
    actual = detect_format(head)
    if expected != "plain" and actual != expected:
        raise CDDISDownloadError(
            f"Expected {_FORMAT_LABELS[expected]} magic bytes ({_MAGIC_TEXT[expected]}) for {name!r}, "
            f"got {head[:4]!r}. URL: {candidate.url}"
        )
    if expected == "plain" and actual != "plain":
        raise CDDISDownloadError(
            f"compressed payload under an uncompressed name: {name!r} holds {_FORMAT_LABELS[actual]} data. "
            f"URL: {candidate.url}"
        )

    try:
        _, notes = verify_stream(path, what=name)
    except CompressedFileError as exc:
        label = _FORMAT_LABELS.get(exc.fmt, exc.fmt)
        if isinstance(exc.__cause__, EOFError):
            raise CDDISDownloadError(
                f"Incomplete {label} file {name} (truncated transfer?): the data ends before its end marker. "
                f"URL: {candidate.url}"
            ) from exc
        raise CDDISDownloadError(f"Corrupt {label} file {name}: {exc}. URL: {candidate.url}") from exc

    _check_signature(path, candidate, actual)
    return list(notes)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalFileCheck:
    """Result of :func:`check_local_file`: is a usable file already on disk?

    ``reason`` is ``"absent"``, ``"manifest"`` (quick check: size and ``mtime_ns`` match the manifest entry),
    ``"checked"`` (the full checks passed; the caller records the file in the manifest) or, for an invalid file,
    why it failed. ``sha256`` is known for ``"checked"`` and, when the manifest holds it, for ``"manifest"``.
    """

    exists: bool
    valid: bool
    reason: str = ""
    sha256: str | None = None


def check_local_file(
    candidate: DownloadCandidate,
    manifest: "DownloadManifest | None" = None,
) -> LocalFileCheck:
    """Whether ``candidate.local_path`` already holds a valid copy of the file.

    Rule (also used by the product set choice of the session): quick check against the ``manifest`` entry
    (size and ``mtime_ns``), else the full integrity and content checks of
    :func:`_validate_downloaded_file`, after which the caller records the
    file in the manifest (``reason == "checked"``). Performs no network
    request and never modifies the file.
    """
    path = candidate.local_path
    try:
        if not path.exists():
            return LocalFileCheck(exists=False, valid=False, reason="absent")
        if not path.is_file():
            return LocalFileCheck(exists=True, valid=False, reason=f"{path} is not a regular file")
    except OSError as exc:
        return LocalFileCheck(exists=True, valid=False, reason=f"{path} cannot be read: {exc}")
    if manifest is not None:
        try:
            if manifest.quick_valid(path):
                entry = manifest.entries.get(manifest.key(path))
                return LocalFileCheck(True, True, "manifest", entry.sha256 if entry is not None else None)
        except ValueError:          # outside the manifest root
            pass
    try:
        _validate_downloaded_file(path, candidate, fresh=False)
        digest = _file_sha256(path)
    except (CDDISDownloadError, CDDISAuthError, OSError) as exc:
        return LocalFileCheck(exists=True, valid=False, reason=str(exc))
    return LocalFileCheck(exists=True, valid=True, reason="checked", sha256=digest)


def _no_auth(request):
    """Explicit "no authentication": keeps requests from looking up ``~/.netrc``."""
    return request


class _NoNetrcSession(requests.Session):
    """:class:`requests.Session` that never takes credentials from ``~/.netrc``.

    With ``trust_env`` on, requests adds a matching ``.netrc`` entry as Basic
    auth to every request without explicit auth, and again after every redirect
    (:meth:`rebuild_auth`). The default transport authenticates with a bearer
    token only, so a ``.netrc`` entry would replace the token, reach sources
    that need no login and be re-sent after a cross-host redirect. Proxy and
    CA-bundle environment settings are still honoured.
    """

    def __init__(self) -> None:
        super().__init__()
        self.auth = _no_auth

    def rebuild_auth(self, prepared_request, response) -> None:
        # requests' own rule: drop the Authorization header (the bearer token)
        # when a redirect leaves the original host; never re-add .netrc auth.
        headers = prepared_request.headers
        if "Authorization" in headers and self.should_strip_auth(response.request.url, prepared_request.url):
            del headers["Authorization"]


# Retry policy of RequestsDownloadTransport (decision D3).
RETRY_AFTER_MAX_S = 60.0           # longest Retry-After (429/503) that is honoured
UNREACHABLE_AFTER_FAILURES = 2     # consecutive connection failures that mark a host unreachable
_RETRY_STATUSES = frozenset({429})  # plus every 5xx

# Exceptions raised while reading a response body: the transfer was cut (retried, D3).
_BODY_ERRORS = (
    requests.RequestException,
    urllib3.exceptions.HTTPError,
    http.client.HTTPException,
    ConnectionError,
    TimeoutError,
)
_DECODE_ERRORS = (requests.exceptions.ContentDecodingError, urllib3.exceptions.DecodeError)


def _sleep(seconds: float) -> None:
    """Retry pause (module function so tests can record or skip the sleeps)."""
    if seconds > 0:
        time.sleep(seconds)


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds of a ``Retry-After`` header (delta-seconds or HTTP date), or ``None``."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _host_of(url: str) -> str:
    return (urlparse(url).netloc or url).lower()


def _exception_chain(exc: BaseException) -> list:
    """``exc`` and the errors it wraps (``__cause__``/``__context__``, urllib3 ``reason``, requests ``args[0]``)."""
    chain: list = []
    todo: list = [exc]
    while todo and len(chain) < 20:
        item = todo.pop(0)
        if item is None or any(item is seen for seen in chain):
            continue
        chain.append(item)
        todo.extend([getattr(item, "reason", None), item.__cause__, item.__context__])
        args = getattr(item, "args", ())
        if args and isinstance(args[0], BaseException):
            todo.append(args[0])
    return chain


# Signs in a requests.ConnectionError that a connection was made and then broke (not "no response at all").
_DROPPED_CONNECTION = (
    urllib3.exceptions.ProtocolError,
    http.client.HTTPException,           # RemoteDisconnected, BadStatusLine, ...
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


def _connection_kind(exc: requests.ConnectionError) -> str:
    """Classify a :class:`requests.ConnectionError` for the retry rules of :class:`RequestsDownloadTransport`.

    ``"certificate"``: TLS certificate verification failed (not retried, not counted; it does not go away);
    ``"tls"``: another TLS error (retried, not counted); ``"dropped"``: the host accepted the connection and it
    broke (retried, not counted); ``"connect"``: no connection at all (refused, name not resolved, connect
    timeout, unreachable proxy; retried and counted towards the unreachable-host rule). An error whose cause
    cannot be told counts as ``"connect"``.
    """
    import ssl

    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "connect"
    chain = _exception_chain(exc)
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in chain):
        return "certificate"
    if isinstance(exc, requests.exceptions.SSLError) or any(isinstance(item, ssl.SSLError) for item in chain):
        return "tls"
    if any(isinstance(item, (urllib3.exceptions.NewConnectionError, urllib3.exceptions.ConnectTimeoutError))
           for item in chain):
        return "connect"
    if any(isinstance(item, _DROPPED_CONNECTION) for item in chain):
        return "dropped"
    return "connect"


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _insecure_redirect(candidate_url: str, final_url) -> bool:
    """True when an HTTPS candidate ended on a plain-HTTP URL (loopback hosts excepted, for local test servers)."""
    if not isinstance(final_url, str) or not final_url:
        return False
    if urlparse(candidate_url).scheme.lower() != "https":
        return False
    final = urlparse(final_url)
    if final.scheme.lower() == "https":
        return False
    host = (final.hostname or "").lower()
    return not (host in _LOOPBACK_HOSTS or host.endswith(".localhost"))


def _with_details(exc: BaseException, *, http_status: int | None = None, attempts: int = 1) -> BaseException:
    """Attach ``http_status`` and ``attempts`` to a transport error (read by the engine for its result row)."""
    exc.http_status = http_status
    exc.attempts = attempts
    return exc


class _DownloadSize(int):
    """Byte count returned by :meth:`RequestsDownloadTransport.download`.

    An ``int`` (the ``DownloadTransport`` protocol is unchanged) that also carries ``http_status``, ``sha256`` (of
    the written bytes), ``attempts`` (HTTP attempts, retries included) and ``notes`` (integrity remarks) for the engine.
    """

    http_status: int | None = None
    sha256: str | None = None
    attempts: int = 1
    notes: tuple[str, ...] = ()


class _CutTransfer(Exception):
    """The response body ended early or broke off (retried unless ``retryable`` is false)."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def _create_temp(destination: Path) -> tuple[int, Path]:
    """A new, unique temp file next to ``destination`` (``.<name>.<random>.part``), opened for writing.

    Like :func:`tempfile.mkstemp` (O_EXCL, so two writers never share a name), but created with mode 0o666
    minus the umask, so the final file gets the usual permissions after :func:`os.replace`.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        path = destination.parent / f".{destination.name}.{secrets.token_hex(6)}.part"
        try:
            return os.open(path, flags, 0o666), path
        except FileExistsError:
            continue
    raise FileExistsError(f"no unique temp name next to {destination}")


class RequestsDownloadTransport:
    """HTTPS transport using :mod:`requests` with Earthdata bearer-token auth.

    CDDIS has required NASA Earthdata authentication since 2020. Unauthenticated
    requests receive an HTML login-redirect page instead of the file, so this
    transport:

    * sends ``Authorization: Bearer <token>`` for CDDIS candidates (dropped on
      a redirect to another host or scheme); a redirect from an HTTPS
      candidate to plain HTTP is refused (error, not retried; loopback hosts
      excepted for local test servers);
    * makes ONE GET per downloaded file (``download()``); ``exists()`` is a
      separate GET whose body is not read, used for dry runs with availability
      checks and for the session's product-set probe;
    * streams the body into a unique temp file next to the destination
      (``.<name>.<random>.part``), validates it, then renames it with
      :func:`os.replace`; the temp file is removed on any failure, including
      ``KeyboardInterrupt``. Compressed names (``.gz``, ``.Z``, ``.bz2``,
      ``.zip``) are written as the raw bytes the server sent (a
      ``Content-Encoding`` header is not undone); plain names are decoded;
    * checks the byte count against ``Content-Length`` (when no
      ``Content-Encoding`` was decoded), then the file itself
      (:func:`_validate_downloaded_file`: size, HTML page, container magic,
      whole-stream integrity, content header);
    * retries timeouts, connection errors, cut transfers, HTTP 5xx and 429 up
      to ``retries`` extra times, sleeping ``RETRY_BACKOFF_S * 2**k`` seconds
      (a ``Retry-After`` of a 429/503 is honoured up to
      :data:`RETRY_AFTER_MAX_S`); other 4xx answers, HTML pages, invalid
      files and failed TLS certificate checks are not retried;
    * marks a host unreachable after :data:`UNREACHABLE_AFTER_FAILURES`
      consecutive connect failures (refused, name not resolved, connect
      timeout: no connection at all), counted per attempt, so a connect
      failure is retried once before the host is skipped; later requests to
      it fail at once with "<host> unreachable" for the rest of this
      transport's life (a ``DownloadSession``), and the engine moves on to the
      next source. Any answer resets the count; read timeouts, 5xx answers,
      dropped connections and TLS errors do not count;
    * keeps one HTTP session per thread (``requests.Session`` is not
      guaranteed to be thread-safe); sessions of finished threads are closed
      when a new one is created, and ``close()``, or a ``with`` block, closes
      them all.

    Errors: HTTP 404 raises :class:`RemoteFileMissing` (status ``"missing"``);
    401/403 and login pages raise :class:`CDDISAuthError` for a source that
    requires authentication (``"auth_missing"``) and
    :class:`CDDISDownloadError` elsewhere, with the real HTTP status. Raised
    errors carry ``http_status`` and ``attempts`` attributes; the byte count
    returned by ``download()`` carries ``sha256`` and ``attempts``.

    The bearer token is resolved (in order) from an explicit ``token`` argument,
    ``CDDISCredentials.token``, or ``os.environ["EARTHDATA_TOKEN"]``. The token
    is never written to disk or emitted in logs. A username/password pair is not
    used by this transport (one ``UserWarning``): pass ``token=`` or use
    ``DownloadConfig(username=..., password=...)``, which exchanges the pair.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        credentials: CDDISCredentials | None = None,
        user_agent: str = "gnx-py/0.1 GNSSDownloader",
        token: str | None = None,
        retries: int = 2,
    ) -> None:
        from .validation import check_retries   # validation imports this module at load time

        # Same rule as DownloadConfig/GNSSDownloadConfig (numpy integers accepted), stored as a plain int.
        self.retries = check_retries(retries)
        self.timeout = timeout
        self.user_agent = user_agent
        self.credentials = credentials.resolved() if credentials else None
        self._token = _resolve_bearer_token(token, self.credentials)
        if (
            self.credentials is not None
            and self.credentials.username
            and self.credentials.password
            and not self._token          # a token from token=, credentials or EARTHDATA_TOKEN is used instead
        ):
            warnings.warn(
                "RequestsDownloadTransport: username/password are not used by this transport; pass token= or use "
                "DownloadConfig(username=..., password=...), which exchanges them for an Earthdata token.",
                UserWarning,
                stacklevel=2,
            )
        self._lock = threading.Lock()
        self._local = threading.local()
        # (owning thread, session); sessions of finished threads are closed when the next one is created.
        self._sessions: list[tuple[threading.Thread, requests.Session]] = []
        self._connection_failures: dict[str, int] = {}
        self._unreachable: set[str] = set()

    # -- HTTP sessions (one per thread) -----------------------------------
    @property
    def session(self) -> requests.Session:
        """The HTTP session of the calling thread (created on first use).

        Creating one also closes the sessions of threads that have finished (the engine starts a new thread
        pool for every day, and a long session would otherwise keep their keep-alive sockets open).
        """
        session = getattr(self._local, "session", None)
        if session is None:
            session = _NoNetrcSession()
            session.headers.update({"User-Agent": self.user_agent})
            # Session-level default; individual requests also pass stream=True.
            session.stream = True
            with self._lock:
                finished = [s for t, s in self._sessions if not t.is_alive()]
                self._sessions = [(t, s) for t, s in self._sessions if t.is_alive()]
                self._sessions.append((threading.current_thread(), session))
            self._local.session = session
            for old in finished:
                old.close()
        return session

    def close(self) -> None:
        """Close the HTTP sessions of every thread; later requests open new ones."""
        with self._lock:
            sessions, self._sessions = self._sessions, []
            self._local = threading.local()
        for _, session in sessions:
            session.close()

    def __enter__(self) -> "RequestsDownloadTransport":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- auth helpers -----------------------------------------------------
    def _auth_headers(self, candidate: DownloadCandidate) -> dict:
        if candidate.requires_auth and self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    def has_credentials(self, candidate: DownloadCandidate) -> bool:
        if not candidate.requires_auth:
            return True
        return bool(self._token)

    # -- host bookkeeping -------------------------------------------------
    def _unreachable_message(self, host: str) -> str | None:
        with self._lock:
            if host not in self._unreachable:
                return None
        return f"{host} unreachable (skipped after {UNREACHABLE_AFTER_FAILURES} connection failures)"

    def _connection_failed(self, host: str) -> bool:
        """Count a connection failure; True when the host is now unreachable."""
        with self._lock:
            count = self._connection_failures.get(host, 0) + 1
            self._connection_failures[host] = count
            if count >= UNREACHABLE_AFTER_FAILURES:
                self._unreachable.add(host)
                return True
        return False

    def _host_answered(self, host: str) -> None:
        with self._lock:
            self._connection_failures[host] = 0

    def _connection_error(self, host: str, exc: requests.ConnectionError, attempt: int) -> str | None:
        """Handle a :class:`requests.ConnectionError` of attempt ``attempt``: ``None`` after the retry pause
        (try again), else the final error text.

        Only "no connection at all" (refused, unresolved name, connect timeout, unreachable proxy) counts
        towards :data:`UNREACHABLE_AFTER_FAILURES`; a connection that was accepted and then broke, and TLS
        errors, are retried without counting. A failed certificate verification is not retried.
        """
        kind = _connection_kind(exc)
        if kind == "connect" and self._connection_failed(host):
            return f"{host} unreachable after {attempt} attempt(s): {exc}"
        if kind == "certificate":
            return f"TLS certificate verification failed for {host}: {exc}"
        if attempt <= self.retries:
            _sleep(self._backoff(attempt))
            return None
        return f"connection failed after {attempt} attempt(s): {exc}"

    def _backoff(self, attempt: int, response=None) -> float:
        delay = RETRY_BACKOFF_S * (2 ** (attempt - 1))
        if response is not None and response.status_code in (429, 503):
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            if retry_after is not None:
                delay = max(delay, min(retry_after, RETRY_AFTER_MAX_S))
        return delay

    @staticmethod
    def _insecure_message(candidate: DownloadCandidate, response) -> str | None:
        """Error text when an HTTPS candidate was redirected to plain HTTP (the file would travel unencrypted)."""
        final_url = getattr(response, "url", None)
        if _insecure_redirect(candidate.url, final_url):
            return (f"refused a redirect from HTTPS to plain HTTP ({urlparse(final_url).netloc}) "
                    f"for {candidate.url}")
        return None

    def _get(self, candidate: DownloadCandidate):
        return self.session.get(
            candidate.url,
            headers=self._auth_headers(candidate),
            stream=True,
            timeout=self.timeout,
            allow_redirects=True,
        )

    # -- protocol methods -------------------------------------------------
    def exists(self, candidate: DownloadCandidate) -> URLAvailability:
        """Whether the remote file exists: one GET (retried like downloads) whose body is not read."""
        if candidate.requires_auth and not self.has_credentials(candidate):
            return URLAvailability(False, _MISSING_TOKEN_MESSAGE, 401, auth_failed=True, attempts=0)
        host = _host_of(candidate.url)
        attempt = 0
        while True:
            unreachable = self._unreachable_message(host)
            if unreachable:
                return URLAvailability(False, unreachable, attempts=attempt)
            attempt += 1
            try:
                response = self._get(candidate)
            except requests.ConnectionError as exc:     # connect timeouts included
                failure = self._connection_error(host, exc, attempt)
                if failure is None:
                    continue
                return URLAvailability(False, failure, attempts=attempt)
            except requests.Timeout as exc:             # read timeout: the host answers, slowly
                self._host_answered(host)
                if attempt <= self.retries:
                    _sleep(self._backoff(attempt))
                    continue
                return URLAvailability(False, f"timed out after {attempt} attempt(s): {exc}", attempts=attempt)
            except requests.RequestException as exc:
                return URLAvailability(False, str(exc), attempts=attempt)
            self._host_answered(host)
            with response:
                status = response.status_code
                insecure = self._insecure_message(candidate, response)
                if insecure:
                    return URLAvailability(False, insecure, status, attempts=attempt)
                if status == 404:
                    return URLAvailability(False, "not found", status, attempts=attempt)
                if status in (401, 403):
                    if candidate.requires_auth:
                        return URLAvailability(
                            False, f"CDDIS authentication failed (HTTP {status}); check EARTHDATA_TOKEN.",
                            status, auth_failed=True, attempts=attempt,
                        )
                    return URLAvailability(False, f"HTTP {status} (access denied)", status, attempts=attempt)
                if status in _RETRY_STATUSES or status >= 500:
                    if attempt <= self.retries:
                        _sleep(self._backoff(attempt, response))
                        continue
                    return URLAvailability(False, f"HTTP {status} after {attempt} attempt(s)", status,
                                           attempts=attempt)
                if status >= 400:
                    return URLAvailability(False, f"HTTP {status}", status, attempts=attempt)
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type or "text/xml" in content_type:
                    if candidate.requires_auth:
                        return URLAvailability(
                            False,
                            "Received an HTML login page instead of the file (check EARTHDATA_TOKEN).",
                            status, auth_failed=True, attempts=attempt,
                        )
                    return URLAvailability(
                        False, f"server returned an HTML page instead of {candidate.filename} (HTTP {status})",
                        status, attempts=attempt,
                    )
                return URLAvailability(True, http_status=status, attempts=attempt)

    def download(self, candidate: DownloadCandidate, destination: Path) -> int:
        """Fetch ``candidate`` into ``destination`` with one GET per attempt; returns the byte count.

        See the class docstring for the status rules, retries and validation.
        """
        if candidate.requires_auth and not self.has_credentials(candidate):
            raise _with_details(CDDISAuthError(_MISSING_TOKEN_MESSAGE), attempts=0)
        destination = Path(destination)
        host = _host_of(candidate.url)
        attempt = 0
        while True:
            unreachable = self._unreachable_message(host)
            if unreachable:
                raise _with_details(CDDISDownloadError(f"{unreachable}. URL: {candidate.url}"), attempts=attempt)
            attempt += 1
            try:
                response = self._get(candidate)
            except requests.ConnectionError as exc:     # connect timeouts included
                failure = self._connection_error(host, exc, attempt)
                if failure is None:
                    continue
                raise _with_details(CDDISDownloadError(f"{failure}. URL: {candidate.url}"),
                                    attempts=attempt) from exc
            except requests.Timeout as exc:             # read timeout: the host answers, slowly
                self._host_answered(host)
                if attempt <= self.retries:
                    _sleep(self._backoff(attempt))
                    continue
                raise _with_details(CDDISDownloadError(
                    f"timed out after {attempt} attempt(s): {exc}. URL: {candidate.url}"
                ), attempts=attempt) from exc
            except requests.RequestException as exc:
                raise _with_details(CDDISDownloadError(f"{exc}. URL: {candidate.url}"), attempts=attempt) from exc
            self._host_answered(host)
            with response:
                status = response.status_code
                insecure = self._insecure_message(candidate, response)
                if insecure:
                    raise _with_details(CDDISDownloadError(insecure), http_status=status, attempts=attempt)
                if status in _RETRY_STATUSES or status >= 500:
                    if attempt <= self.retries:
                        _sleep(self._backoff(attempt, response))
                        continue
                    raise _with_details(CDDISDownloadError(
                        f"HTTP {status} while downloading {candidate.url} (after {attempt} attempt(s))"
                    ), http_status=status, attempts=attempt)
                self._raise_for_bad_response(response, candidate, attempt)
                try:
                    return self._write(response, candidate, destination, attempt)
                except _CutTransfer as exc:
                    if exc.retryable and attempt <= self.retries:
                        LOGGER.info("Retrying %s: %s", candidate.url, exc)
                        _sleep(self._backoff(attempt))
                        continue
                    raise _with_details(CDDISDownloadError(
                        f"{exc} (after {attempt} attempt(s)). URL: {candidate.url}"
                    ), http_status=status, attempts=attempt) from exc
                except (CDDISDownloadError, CDDISAuthError) as exc:
                    raise _with_details(exc, http_status=status, attempts=attempt)

    def _write(self, response, candidate: DownloadCandidate, destination: Path, attempt: int) -> int:
        """Stream the body to a unique temp file, check it, rename it to ``destination``."""
        raw = _expected_format(destination.name) != "plain"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = _create_temp(destination)
        renamed = False
        digest = hashlib.sha256()
        written = 0
        try:
            with os.fdopen(fd, "wb") as fout:
                if raw:      # keep the archive's bytes: do not undo a Content-Encoding (DL-016)
                    chunks = response.raw.stream(65536, decode_content=False)
                else:
                    chunks = response.iter_content(chunk_size=65536)
                iterator = iter(chunks)
                while True:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        break
                    except _DECODE_ERRORS as exc:
                        raise _CutTransfer(f"undecodable response body: {exc}", retryable=False) from exc
                    except _BODY_ERRORS as exc:
                        raise _CutTransfer(f"transfer interrupted after {written} bytes: {exc}") from exc
                    if chunk:
                        fout.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
            length = (response.headers.get("Content-Length") or "").strip()
            encoding = (response.headers.get("Content-Encoding") or "").strip().lower()
            if length.isdigit() and (raw or encoding in ("", "identity")):
                expected = int(length)
                if written < expected:
                    raise _CutTransfer(f"truncated transfer: {written} of {expected} bytes")
                if written > expected:
                    raise _CutTransfer(f"size mismatch: {written} bytes received, Content-Length {expected}")
            notes = _validate_downloaded_file(tmp_path, candidate, response.status_code)
            os.replace(tmp_path, destination)
            renamed = True
        finally:
            if not renamed:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
        for note in notes:
            LOGGER.info("%s: %s", candidate.filename, note)
        size = _DownloadSize(written)
        size.http_status = response.status_code
        size.sha256 = digest.hexdigest()
        size.attempts = attempt
        size.notes = tuple(notes)
        return size

    @staticmethod
    def _raise_for_bad_response(response, candidate: DownloadCandidate, attempt: int = 1) -> None:
        status = response.status_code
        if status == 404:
            raise _with_details(RemoteFileMissing(f"Remote file not found (HTTP 404): {candidate.url}"),
                                http_status=404, attempts=attempt)
        if status in (401, 403):
            if candidate.requires_auth:
                raise _with_details(CDDISAuthError(
                    f"CDDIS authentication failed (HTTP {status}). Check EARTHDATA_TOKEN. URL: {candidate.url}"
                ), http_status=status, attempts=attempt)
            raise _with_details(CDDISDownloadError(
                f"HTTP {status} (access denied) while downloading {candidate.url}"
            ), http_status=status, attempts=attempt)
        if status >= 400:
            raise _with_details(CDDISDownloadError(f"HTTP {status} while downloading {candidate.url}"),
                                http_status=status, attempts=attempt)
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type or "text/xml" in content_type:
            if candidate.requires_auth:
                raise _with_details(CDDISAuthError(
                    "Received an HTML/XML page (an Earthdata login redirect) instead of "
                    f"{candidate.filename}. Check EARTHDATA_TOKEN. URL: {candidate.url}"
                ), http_status=status, attempts=attempt)
            raise _with_details(CDDISDownloadError(
                f"server returned an HTML page instead of {candidate.filename} (HTTP {status}). "
                f"URL: {candidate.url}"
            ), http_status=status, attempts=attempt)



def _units_sharing_files(groups: Sequence[Sequence[DownloadCandidate]]) -> list[list[Sequence[DownloadCandidate]]]:
    """Request groups bundled so that groups sharing a local file run in one worker, in order.

    The candidate generator avoids shared files by construction; this is the engine's defence, so that two
    workers never write one file at the same time. Unit order follows the first group of each unit.
    """
    owner: dict[Path, int] = {}
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for index, group in enumerate(groups):
        for candidate in group:
            first = owner.setdefault(candidate.local_path, index)
            if first != index:
                a, b = find(first), find(index)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    units: dict[int, list[Sequence[DownloadCandidate]]] = {}
    for index, group in enumerate(groups):
        units.setdefault(find(index), []).append(group)
    return list(units.values())


_VERBOSE_LOGGER = "gnx_py.downloaders"


def _enable_verbose_logging() -> None:
    """``verbose=True``: one INFO ``StreamHandler`` on the ``gnx_py.downloaders`` logger.

    Added only when that logger has no handler yet (so repeated engine runs add nothing); the root logger and
    ``logging.basicConfig`` are never touched. The logger's level is lowered to INFO only when it is unset or
    higher. Both stay for the rest of the process: later runs with ``verbose=False`` still print INFO lines,
    and when the application also has a handler on the root logger each line appears twice (propagation).
    Remove the handler, or set the level, on ``logging.getLogger("gnx_py.downloaders")`` to change that.
    """
    logger = logging.getLogger(_VERBOSE_LOGGER)
    with _VERBOSE_LOCK:
        if logger.handlers:
            return
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)


_VERBOSE_LOCK = threading.Lock()


class GNSSDownloader:
    """Low-level engine: generate archive candidates, check and download them.

    Status:
        Low-level engine API, used internally by ``DownloadSession`` (one
        instance per requested day). Use it directly when you need a single
        year/day-of-year run or want to inspect candidates with
        ``build_candidates()``; otherwise prefer ``DownloadConfig`` +
        ``DownloadSession``. Both accept a custom ``transport``.

    Behavior:
        ``download()`` (also available as ``run()``) groups candidates by
        data type, day and requested station and tries each group in
        priority order until one candidate is satisfied: a valid existing
        local file (of any candidate of the group), a dry-run hit or a
        successful download; see :meth:`download` for the rules. It returns
        an ``ArchiveDownloadReport``. Unlike ``DownloadSession.run()`` it is
        not wrapped by ``guarded_session_run``; per-candidate transport
        failures become result rows, anything else propagates unchanged.

    Transport:
        Without an injected ``transport`` the engine builds a
        ``RequestsDownloadTransport(timeout, credentials, retries)`` and
        closes it at the end of every ``download()``; an injected transport
        is never closed by the engine.

    Logging:
        Progress is logged at INFO on the ``gnx_py.downloaders.core`` logger
        (or the injected ``logger``). ``verbose=True`` adds one INFO
        ``StreamHandler`` to the ``gnx_py.downloaders`` logger when it has
        none; the root logger is not configured. The handler and the INFO
        level stay for the rest of the process (see
        :func:`_enable_verbose_logging`).
    """

    def __init__(
        self,
        config: GNSSDownloadConfig,
        transport: DownloadTransport | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        # A transport built here is closed at the end of every download(); an injected one is the caller's.
        self._owns_transport = transport is None
        self.transport = transport if transport is not None else RequestsDownloadTransport(
            timeout=config.timeout,
            credentials=config.credentials,
            retries=getattr(config, "retries", 2),
        )
        self.logger = logger or LOGGER
        if config.verbose:
            _enable_verbose_logging()

    def build_candidates(self) -> list[DownloadCandidate]:
        """Every archive candidate of the run, in trial order.

        The order is source (as configured), then data type (as configured),
        then RINEX version or product preference, then file name. The layout
        (directories and names per source, type and day) is described in
        the downloaders README, "Output Layout" and "Sources"; the product
        file names come from :mod:`gnx_py.downloaders.catalog`.

        INFO lines about skipped products and stations are logged once per
        engine. A caller that runs several engines (``DownloadSession``: one
        per day, plus the product-set probes) may set ``engine.layout_notes``
        to one shared ``set`` before the run, so that each line is logged
        once per session.

        Raises ``ValueError`` for a product id that is not letters and digits
        only, or a candidate whose file name, host or local path would leave
        the archive directory or ``output_dir`` (a station id or a product id
        carrying ``/``, ``:`` or ``..``).
        """
        shared_notes = getattr(self, "layout_notes", None)
        self._layout_notes: set[str] = shared_notes if shared_notes is not None else set()
        candidates: list[DownloadCandidate] = []
        for day, doy in iter_doy_range(self.config.year, self.config.doy_start, self.config.doy_end):
            for source in self.config.sources:
                if "obs" in self.config.data_types:
                    candidates.extend(self._obs_candidates(source, day, doy))
                if "nav" in self.config.data_types:
                    candidates.extend(self._nav_candidates(source, day, doy))
                for data_type in catalog.PRODUCT_TYPES:
                    if data_type in self.config.data_types:
                        candidates.extend(self._product_candidates(source, data_type, day, doy))
                if "ionex" in self.config.data_types:
                    candidates.extend(self._ionex_candidates(source, day, doy))
        return sorted(candidates, key=lambda item: item.priority)

    def _layout_note(self, message: str, *args) -> None:
        """Log an INFO line about the candidate layout once per engine (or shared notes set)."""
        notes = getattr(self, "_layout_notes", None)
        if notes is None:
            notes = self._layout_notes = set()
        key = message % args if args else message
        if key in notes:
            return
        notes.add(key)
        self.logger.info(message, *args)

    def download(self) -> ArchiveDownloadReport:
        """Run the engine: build the candidates, process every request group, return the report.

        Groups are keyed by ``(data_type, year, doy, request or station)``.
        Each group is processed in candidate order:

        1. Unless ``overwrite=True``, every candidate's local file is checked
           before any request (:func:`check_local_file`: manifest quick check,
           else the full integrity and content checks). The first valid one
           gives ``skipped_existing`` with no request, also when it lies under
           another source directory or another accepted name. An invalid file
           is logged (WARNING) and replaced by the download; when no source
           delivers a valid file, that candidate's row is ``"error"`` naming
           the stale file. When a later candidate's valid file satisfies the
           request, an invalid one is only logged at INFO and left on disk.
        2. ``dry_run`` without availability checks: ``dry_run`` for the first
           candidate, no request.
        3. A candidate that needs credentials the transport does not have:
           ``auth_missing``, no request.
        4. ``dry_run`` with checks: ``transport.exists()``; HTTP 404 gives
           ``missing``, 401/403 or a login page on a source that requires
           authentication ``auth_missing``, anything else ``error``.
        5. Otherwise one ``transport.download()`` (no separate existence
           check): :class:`RemoteFileMissing` gives ``missing``,
           :class:`CDDISAuthError` ``auth_missing``, any other exception
           ``error`` (the next candidate is then tried).

        Groups whose candidates share a local file run one after the other in
        one worker. Requested items without any candidate are listed in
        ``ArchiveDownloadReport.unavailable`` with the reason
        (:func:`unavailable_reason`, :func:`station_reason`). The manifest
        ``<output_dir>/.gnx_downloads.json`` is read once and, unless
        ``dry_run=True``, saved at the end when an entry changed. A file is
        recorded after a full check or after a download by
        :class:`RequestsDownloadTransport` (which validates it); a file
        written by another transport is not recorded, so the next run checks
        it in full. A transport built by this engine is closed at the end.
        """
        from .manifest import DownloadManifest

        try:
            candidates = self.build_candidates()
            grouped: dict[tuple[str, int, int, str | None], list[DownloadCandidate]] = {}
            for candidate in candidates:
                grouped.setdefault(candidate.group_key, []).append(candidate)
            group_values = list(grouped.values())
            units = _units_sharing_files(group_values)

            self._manifest = DownloadManifest.load(self.config.output_dir)
            self._manifest_lock = threading.Lock()
            self._manifest_dirty = False
            workers = self.config.max_workers
            if workers and workers > 1 and len(units) > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    processed_units = list(executor.map(self._process_unit, units))
            else:
                processed_units = [self._process_unit(unit) for unit in units]
            if not self.config.dry_run and self._manifest_dirty:
                try:
                    self._manifest.save()
                except OSError as exc:
                    self.logger.warning("Could not save the download manifest %s: %s", self._manifest.path, exc)

            by_group = {id(group): outcome for unit, outcomes in zip(units, processed_units)
                        for group, outcome in zip(unit, outcomes)}
            results: list[DownloadResult] = []
            for group in group_values:
                results.extend(by_group[id(group)][0])
            return ArchiveDownloadReport(
                candidates=candidates, results=results, unavailable=self._unavailable(candidates)
            )
        finally:
            if self._owns_transport:
                close = getattr(self.transport, "close", None)
                if callable(close):
                    close()

    # Same ``run()`` verb as the session/controller classes across gnx_py.
    run = download

    def _unavailable(self, candidates: Sequence[DownloadCandidate]) -> list[UnavailableRequest]:
        """Requested (type, day[, station]) items that produced no candidate, with the reason of :func:`unavailable_reason`."""
        covered = {(c.data_type, c.day) for c in candidates if c.data_type != "obs"}
        obs_covered = {(c.day, c.request or c.station) for c in candidates if c.data_type == "obs"}
        obs_legacy = {(c.day, c.station) for c in candidates if c.data_type == "obs" and c.request is None}
        missing: list[UnavailableRequest] = []
        for day, _ in iter_doy_range(self.config.year, self.config.doy_start, self.config.doy_end):
            for data_type in self.config.data_types:
                if data_type == "obs":
                    if not self.config.stations:
                        missing.append(UnavailableRequest("obs", day, None, "no stations requested"))
                    for station in self.config.stations:
                        try:
                            variants = station_variants(station, self.config.station_aliases)
                        except ValueError as exc:
                            missing.append(UnavailableRequest("obs", day, str(station).strip().upper(), str(exc)))
                            continue
                        request = variants.original.strip().upper()     # as DownloadCandidate.request (K-CAND)
                        if (day, request) in obs_covered or (day, variants.legacy4) in obs_legacy:
                            continue
                        reason = (
                            unavailable_reason("obs", self.config.sources, self.config.product_preferences)
                            or station_reason(variants, self.config.rinex_version_preference)
                            or f"no observation file name for station {request!r}"
                        )
                        missing.append(UnavailableRequest("obs", day, request, reason))
                elif (data_type, day) not in covered:
                    reason = (
                        unavailable_reason(data_type, self.config.sources, self.config.product_preferences)
                        or f"no candidate file name for {data_type!r} on {day.isoformat()}"
                    )
                    missing.append(UnavailableRequest(data_type, day, None, reason))
        return missing

    def _process_unit(self, unit: Sequence[Sequence[DownloadCandidate]]) -> list[tuple[list[DownloadResult], bool]]:
        """Process groups that share local files one after the other."""
        return [self._process_group(group) for group in unit]

    def _record(self, path: Path, sha256: str | None, check: str) -> None:
        manifest = getattr(self, "_manifest", None)
        if manifest is None:
            return
        try:
            with self._manifest_lock:
                manifest.record(path, sha256=sha256, check=check)
                self._manifest_dirty = True
        except (OSError, ValueError) as exc:     # vanished file, or a path outside output_dir
            self.logger.debug("Manifest entry for %s not recorded: %s", path, exc)

    def _forget(self, path: Path) -> None:
        manifest = getattr(self, "_manifest", None)
        if manifest is None:
            return
        try:
            with self._manifest_lock:
                manifest.forget(path)
                self._manifest_dirty = True
        except ValueError:
            pass

    def _local_scan(self, candidates: Sequence[DownloadCandidate]) -> tuple[DownloadResult | None, dict[Path, str]]:
        """First valid local file of the group, and the invalid ones with their reason."""
        stale: dict[Path, str] = {}
        seen: set[Path] = set()
        manifest = getattr(self, "_manifest", None)
        for candidate in candidates:
            path = candidate.local_path
            if path in seen:
                continue
            seen.add(path)
            check = check_local_file(candidate, manifest)
            if not check.exists:
                continue
            if check.valid:
                if check.reason == "checked":
                    self._record(path, check.sha256, "full")
                for stale_path, reason in stale.items():
                    # Nothing is downloaded for this request, so the invalid file stays (it is not deleted).
                    self.logger.info("Existing local file %s is invalid and ignored (valid copy at %s): %s",
                                     stale_path, path, reason)
                how = "manifest entry matches" if check.reason == "manifest" else "checked"
                return DownloadResult(
                    candidate=candidate,
                    status="skipped_existing",
                    message=f"Valid local file already exists ({how}) and overwrite=False.",
                    path=path,
                    url=candidate.url,
                    attempts=0,
                    sha256=check.sha256,
                ), stale
            stale[path] = check.reason
            self._forget(path)
        for stale_path, reason in stale.items():
            self.logger.warning("Existing local file %s is invalid and will be downloaded again: %s",
                                stale_path, reason)
        return None, stale

    def _process_group(self, candidates: Sequence[DownloadCandidate]) -> tuple[list[DownloadResult], bool]:
        results: list[DownloadResult] = []
        stale: dict[Path, str] = {}
        if not self.config.overwrite:
            skipped, stale = self._local_scan(candidates)
            if skipped is not None:
                return [skipped], True

        satisfied = False
        for candidate in candidates:
            row = self._try_candidate(candidate)
            results.append(row)
            if row.status in ("downloaded", "dry_run"):
                satisfied = True
                break

        if stale:
            results = [self._stale_row(row, stale[row.candidate.local_path], satisfied)
                       if row.candidate.local_path in stale and row.status not in ("downloaded", "dry_run")
                       else row for row in results]
        return results, satisfied

    @staticmethod
    def _stale_row(row: DownloadResult, reason: str, satisfied: bool) -> DownloadResult:
        note = f"the existing local file {row.candidate.local_path} is invalid ({reason}) and was not replaced"
        status = row.status if satisfied else "error"
        return replace(row, status=status, message=f"{row.message}; {note}" if row.message else note)

    def _row(self, candidate: DownloadCandidate, status: str, message: str, **kw) -> DownloadResult:
        return DownloadResult(candidate=candidate, status=status, message=message, path=candidate.local_path,
                              url=candidate.url, **kw)

    def _try_candidate(self, candidate: DownloadCandidate) -> DownloadResult:
        """One candidate: dry-run row, availability check or download; never raises an ``Exception``."""
        if self.config.dry_run and not self.config.check_availability:
            return self._row(candidate, "dry_run", "Candidate generated; availability not checked.", attempts=0)

        try:
            has_credentials = self.transport.has_credentials(candidate)
        except Exception as exc:  # noqa: BLE001 - a transport failure is a per-candidate error (Q154)
            return self._row(candidate, "error", f"has_credentials() failed: {type(exc).__name__}: {exc}",
                             attempts=0)
        if candidate.requires_auth and not has_credentials:
            return self._row(candidate, "auth_missing", _MISSING_TOKEN_MESSAGE, http_status=401, attempts=0)

        if self.config.dry_run:
            try:
                availability = self.transport.exists(candidate)
            except Exception as exc:  # noqa: BLE001 - Q154
                return self._row(candidate, "error", f"exists() failed: {type(exc).__name__}: {exc}")
            attempts = getattr(availability, "attempts", 1)
            if availability.exists:
                return self._row(candidate, "dry_run", "Remote file exists; dry_run=True so it was not downloaded.",
                                 http_status=availability.http_status, attempts=attempts)
            status = availability.http_status
            if status == 404:
                kind = "missing"
            elif getattr(availability, "auth_failed", False) or (candidate.requires_auth and status in (401, 403)):
                kind = "auth_missing"
            else:
                kind = "error"
            return self._row(candidate, kind, availability.message, http_status=status, attempts=attempts)

        self.logger.info("Downloading %s -> %s", candidate.url, candidate.local_path)
        try:
            bytes_written = self.transport.download(candidate, candidate.local_path)
        except RemoteFileMissing as exc:
            return self._row(candidate, "missing", str(exc), http_status=getattr(exc, "http_status", 404),
                             attempts=getattr(exc, "attempts", 1))
        except CDDISAuthError as exc:
            return self._row(candidate, "auth_missing", str(exc), http_status=getattr(exc, "http_status", None),
                             attempts=getattr(exc, "attempts", 1))
        except Exception as exc:  # noqa: BLE001 - per-candidate error; the next candidate is tried
            return self._row(candidate, "error", str(exc) or type(exc).__name__,
                             http_status=getattr(exc, "http_status", None), attempts=getattr(exc, "attempts", 1))

        sha256 = getattr(bytes_written, "sha256", None)
        # Only RequestsDownloadTransport validates what it writes (its byte count carries the sha256). A file
        # from another transport is not recorded, so the next run checks it in full before trusting it (D5).
        if isinstance(bytes_written, _DownloadSize) and sha256 and candidate.local_path.is_file():
            self._record(candidate.local_path, sha256, "download")
        return self._row(candidate, "downloaded", "Downloaded successfully.",
                         http_status=getattr(bytes_written, "http_status", None),
                         bytes_written=int(bytes_written) if bytes_written is not None else None,
                         attempts=getattr(bytes_written, "attempts", 1), sha256=sha256)

    def _obs_candidates(self, source: str, day: date, doy: int) -> list[DownloadCandidate]:
        if "obs" not in SOURCE_CAPABILITIES.get(source, frozenset()):
            return []

        candidates: list[DownloadCandidate] = []
        # Every requested station gets all of its candidates, so every request
        # has a report row. Two requests share a local file only when both
        # accept that file (the 00 rule keeps another receiver's RINEX 2 file
        # away, DL-033); the engine runs such groups in one worker, in order
        # (C2-10), so the later one finds the file on disk. A request repeated
        # with the same id is one group and is generated once.
        seen_requests: set[str] = set()
        long_versions = any(v in {"3", "4"} for v in self.config.rinex_version_preference)
        for station in self.config.stations:
            variants = station_variants(station, self.config.station_aliases)
            request = variants.original.strip().upper()
            if request in seen_requests:
                continue
            seen_requests.add(request)
            legacy_allowed = _legacy_obs_allowed(variants)
            notes: tuple[str, ...] = ()
            if "2" in self.config.rinex_version_preference and not legacy_allowed:
                notes = (_no_legacy_obs_note(request),)
                self._layout_note("Station %s: %s", request, notes[0])
            for version in _rinex_long_then_legacy(self.config.rinex_version_preference):
                if version in {"3", "4"}:
                    for station9 in variants.rinex3_ids:
                        candidates.extend(
                            self._obs_long_candidates(
                                source, day, doy, variants.legacy4, station9, version,
                                request=request, extra_notes=notes,
                            )
                        )
                elif version == "2" and legacy_allowed:
                    candidates.extend(
                        self._obs_legacy_candidates(source, day, doy, variants, request=request)
                    )

            if not variants.rinex3_ids and long_versions:
                self._layout_note(
                    "Station %s has no known 9-character RINEX station id; "
                    "only RINEX2-style candidates will be generated.",
                    station,
                )
        return candidates

    def _obs_long_candidates(
        self,
        source: str,
        day: date,
        doy: int,
        station4: str,
        station9: str,
        version: str,
        request: str | None = None,
        extra_notes: tuple[str, ...] = (),
    ) -> list[DownloadCandidate]:
        year = day.year
        # One long name per data-source letter (``_R_`` receiver, ``_S_``
        # stream, ``_U_`` unknown; RINEX 3.04 file naming) and compression,
        # in the order of ``rinex_data_source_preference``.
        letters = self.config.rinex_data_source_preference
        filenames = tuple(
            f"{station9}_{letter}_{year}{doy:03d}0000_01D_30S_MO.{extension}"
            for letter in letters
            for extension in ("crx.gz", "rnx.gz")
        )
        return [
            self._candidate(
                source=source,
                data_type="obs",
                day=day,
                doy=doy,
                station=station4,
                filename=filename,
                url=urljoin(_obs_base_url(source, year, doy, filename), filename),
                rinex_version=version,
                priority=self._priority(source, "obs", version, index),
                requires_auth=source == "cddis",
                notes=("RINEX3/4 long-name daily observation candidate.",) + tuple(extra_notes),
                request=request,
            )
            for index, filename in enumerate(filenames)
        ]

    def _obs_legacy_candidates(
        self,
        source: str,
        day: date,
        doy: int,
        variants: StationVariants,
        request: str | None = None,
    ) -> list[DownloadCandidate]:
        year = day.year
        yy = year % 100
        station = variants.legacy4.lower()
        # Hatanaka-compressed "d" files first, then plain "o" files; .Z
        # before .gz (CDDIS moved its daily files to .gz in December 2020).
        filenames = (
            f"{station}{doy:03d}0.{yy:02d}d.Z",
            f"{station}{doy:03d}0.{yy:02d}d.gz",
            f"{station}{doy:03d}0.{yy:02d}o.Z",
            f"{station}{doy:03d}0.{yy:02d}o.gz",
        )
        notes = variants.notes + ("RINEX2 legacy daily observation candidate.",)
        return [
            self._candidate(
                source=source,
                data_type="obs",
                day=day,
                doy=doy,
                station=variants.legacy4,
                filename=filename,
                url=urljoin(_obs_base_url(source, year, doy, filename), filename),
                rinex_version="2",
                priority=self._priority(source, "obs", "2", index + 20),
                requires_auth=source == "cddis",
                notes=notes,
                request=request,
            )
            for index, filename in enumerate(filenames)
        ]

    def _nav_products(self) -> tuple[str, ...]:
        """Broadcast navigation products of the run, in trial order.

        ``product_preferences["nav"]`` (vocabulary
        :data:`gnx_py.downloaders.catalog.NAV_PRODUCTS`) when given, else one
        product per entry of ``rinex_version_preference``
        (:data:`DEFAULT_NAV_BY_RINEX`). The navigation preference is
        independent of the RINEX version of the observation files.
        """
        preferences = self.config.product_preferences or {}
        if "nav" in preferences:
            items = preferences["nav"]
            if isinstance(items, str):
                items = (items,)
            products = tuple(str(item).strip().upper() for item in items)
        else:
            products = tuple(
                DEFAULT_NAV_BY_RINEX[str(version)] for version in self.config.rinex_version_preference
            )
        unknown = [product for product in products if product not in catalog.NAV_PRODUCTS]
        if unknown:
            raise ValueError(
                f"product_preferences['nav'] has unknown navigation products {unknown}; "
                f"supported: {tuple(catalog.NAV_PRODUCTS)}."
            )
        return tuple(dict.fromkeys(products))

    def _nav_candidates(self, source: str, day: date, doy: int) -> list[DownloadCandidate]:
        if "nav" not in SOURCE_CAPABILITIES.get(source, frozenset()):
            return []

        year = day.year
        yy = year % 100
        candidates: list[DownloadCandidate] = []
        for product_index, product in enumerate(self._nav_products()):
            version = catalog.NAV_PRODUCTS[product]
            offset = product_index * 100
            if product == "BRDC":
                # GPS-only RINEX 2 file; GLONASS "g" files are not generated (D7).
                for index, filename in enumerate((
                    f"brdc{doy:03d}0.{yy:02d}n.Z",
                    f"brdc{doy:03d}0.{yy:02d}n.gz",
                )):
                    candidates.append(
                        self._nav_candidate(source, day, doy, filename, version, product, priority_offset=offset)
                    )
                    if source == "cddis":
                        candidates.append(
                            self._nav_candidate(
                                source,
                                day,
                                doy,
                                filename,
                                version,
                                product,
                                cddis_annual_brdc=True,
                                priority_offset=offset + index + 50,
                            )
                        )
            else:
                filename = f"{_NAV_LONG_PREFIXES[product]}_{year}{doy:03d}0000_01D_MN.rnx.gz"
                candidates.append(
                    self._nav_candidate(source, day, doy, filename, version, product, priority_offset=offset)
                )
        return candidates

    def _nav_candidate(
        self,
        source: str,
        day: date,
        doy: int,
        filename: str,
        version: str,
        product: str,
        cddis_annual_brdc: bool = False,
        priority_offset: int = 0,
    ) -> DownloadCandidate:
        year = day.year
        if source == "bkg":
            base_url = f"{BKG_BASE_URL}/BRDC/{year}/{doy:03d}/"
        else:
            if filename.startswith("BRD") or cddis_annual_brdc:
                base_url = f"{CDDIS_BASE_URL}/data/daily/{year}/brdc/"
            else:
                base_url = f"{CDDIS_BASE_URL}/data/daily/{year}/{doy:03d}/{year % 100:02d}n/"
        return self._candidate(
            source=source,
            data_type="nav",
            day=day,
            doy=doy,
            station=None,
            filename=filename,
            url=urljoin(base_url, filename),
            rinex_version=version,
            product=product,
            # Ordered by the navigation product list, not by the RINEX
            # preference of the observation files.
            priority=self._priority(source, "nav", None, priority_offset),
            requires_auth=source == "cddis",
            notes=("Daily broadcast navigation candidate.",),
        )

    def _product_candidates(self, source: str, data_type: str, day: date, doy: int) -> list[DownloadCandidate]:
        if data_type not in SOURCE_CAPABILITIES.get(source, frozenset()):
            return []
        products = self.config.product_preferences.get(
            data_type,
            DEFAULT_PRODUCT_PREFERENCES[data_type],
        )
        if isinstance(products, str):
            products = (products,)
        candidates: list[DownloadCandidate] = []
        skipped: dict[str, list[str]] = {}
        for product_index, item in enumerate(products):
            product = catalog.product_id(item, f"product_preferences[{data_type!r}]")
            reason = _product_skip_reason(source, data_type, product)
            if reason is not None:
                skipped.setdefault(reason, []).append(product)
                continue
            for suffix in catalog.product_suffixes(product, data_type):
                # A weekly (quarterly) file is stamped and stored under its
                # GPS-week (quarter) start day.
                stamp_day = catalog.file_day(day, suffix)
                if stamp_day < _GPS_EPOCH <= day:
                    continue  # a multi-day file starting before GPS week 0 (no GPS products then)
                stamp_doy = stamp_day.timetuple().tm_yday
                filename = f"{product}_{stamp_day.year}{stamp_doy:03d}0000_{suffix}"
                for base_url in _product_base_urls(source, stamp_day, stamp_doy, product, data_type):
                    candidates.append(
                        self._candidate(
                            source=source,
                            data_type=data_type,
                            day=day,
                            doy=doy,
                            station=None,
                            filename=filename,
                            url=urljoin(base_url, filename),
                            product=product,
                            priority=self._priority(source, data_type, None, product_index),
                            requires_auth=source == "cddis",
                            notes=("GNSS product candidate.",),
                            store_day=stamp_day,
                        )
                    )
        self._note_skipped_products(source, data_type, skipped)
        return candidates

    def _note_skipped_products(self, source: str, data_type: str, skipped: Mapping[str, list[str]]) -> None:
        # One INFO line per (source, data type, reason) and run.
        for reason, products in skipped.items():
            self._layout_note(
                "%s %s at %s skipped: %s.", data_type, ", ".join(products), source, reason
            )

    def _ionex_candidates(self, source: str, day: date, doy: int) -> list[DownloadCandidate]:
        # BKG's IONEX layout is not modelled here; use cddis or code for GIM.
        base_urls = _ionex_base_urls(source, day.year, doy)
        if not base_urls:
            return []

        year = day.year
        products = self.config.product_preferences.get("ionex", DEFAULT_IONEX_PRODUCTS)
        if isinstance(products, str):
            products = (products,)
        candidates: list[DownloadCandidate] = []
        skipped: dict[str, list[str]] = {}
        for product_index, item in enumerate(products):
            label = catalog.product_id(item, "product_preferences['ionex']")
            reason = _product_skip_reason(source, "ionex", label)
            if reason is not None:
                skipped.setdefault(reason, []).append(label)
                continue
            for name_index, filename in enumerate(_ionex_filenames(label, year, doy, source)):
                for base_url in base_urls:
                    candidates.append(
                        self._candidate(
                            source=source,
                            data_type="ionex",
                            day=day,
                            doy=doy,
                            station=None,
                            filename=filename,
                            url=urljoin(base_url, filename),
                            product=label,
                            priority=self._priority(
                                source, "ionex", None, product_index * 100 + name_index
                            ),
                            requires_auth=source == "cddis",
                            notes=("IONEX/GIM product candidate.",),
                        )
                    )
        self._note_skipped_products(source, "ionex", skipped)
        return candidates

    def _candidate(
        self,
        source: str,
        data_type: str,
        day: date,
        doy: int,
        station: str | None,
        filename: str,
        url: str,
        priority: int,
        rinex_version: str | None = None,
        product: str | None = None,
        requires_auth: bool = False,
        notes: tuple[str, ...] = (),
        request: str | None = None,
        store_day: date | None = None,
    ) -> DownloadCandidate:
        # ``store_day``: the day directory of the file when it differs from
        # the requested day (a weekly or quarterly product stored under its
        # stamp day, so every day it covers finds it on disk).
        if store_day is None:
            store_year, store_doy = day.year, doy
        else:
            store_year, store_doy = store_day.year, store_day.timetuple().tm_yday
        local_path = (
            self.config.output_dir / f"{store_year:04d}" / f"{store_doy:03d}" / source / data_type / filename
        )
        _check_candidate_target(source, data_type, filename, url, local_path, self.config.output_dir)
        return DownloadCandidate(
            source=source,
            data_type=data_type,
            url=url,
            local_path=local_path,
            year=day.year,
            doy=doy,
            day=day,
            station=station,
            rinex_version=rinex_version,
            product=product,
            requires_auth=requires_auth,
            notes=notes,
            priority=priority,
            request=request,
        )

    def _priority(self, source: str, data_type: str, rinex_version: str | None, index: int) -> int:
        source_rank = self.config.sources.index(source)
        type_rank = self.config.data_types.index(data_type)
        if rinex_version is not None:
            version_rank = self.config.rinex_version_preference.index(rinex_version)
        else:
            version_rank = 0
        return source_rank * 100_000 + type_rank * 10_000 + version_rank * 100 + index


def _check_int(value, name: str) -> int:
    # int and integer-like numbers (numpy included), never bool or float.
    if isinstance(value, bool) or not hasattr(type(value), "__index__"):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    return value.__index__()


def year_doy_to_date(year: int, doy: int) -> date:
    """Convert a year/day-of-year pair to a calendar date.

    ``year`` and ``doy`` must be ``int`` (``TypeError`` otherwise); ``doy``
    must lie within the year (1-365, or 1-366 in a leap year) and ``year``
    within ``datetime.date``'s range (``ValueError`` otherwise).
    """
    year = _check_int(year, "year")
    doy = _check_int(doy, "doy")
    if doy < 1:
        raise ValueError(f"doy must be >= 1, got {doy}.")
    if not date.min.year <= year <= date.max.year:
        raise ValueError(f"year {year} is outside {date.min.year}..{date.max.year}.")
    first_day = date(year, 1, 1)
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    if doy > days_in_year:
        raise ValueError(f"doy {doy} is outside year {year} (1..{days_in_year}).")
    return first_day + timedelta(days=doy - 1)


def iter_doy_range(year: int, doy_start: int, doy_end: int) -> Iterable[tuple[date, int]]:
    """Yield ``(date, doy)`` pairs for an inclusive day-of-year range.

    The arguments are checked before the first pair is produced, so a
    wrong range raises on the first ``next()``.
    """
    year = _check_int(year, "year")
    doy_start = _check_int(doy_start, "doy_start")
    doy_end = _check_int(doy_end, "doy_end")
    if doy_start > doy_end:
        raise ValueError(f"doy_start must be <= doy_end, got {doy_start} > {doy_end}.")
    for name, doy in (("doy_start", doy_start), ("doy_end", doy_end)):
        try:
            year_doy_to_date(year, doy)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
    for doy in range(doy_start, doy_end + 1):
        yield year_doy_to_date(year, doy), doy


def gps_week_day(year: int, doy: int) -> tuple[int, int]:
    """Return GPS week and GPS day-of-week for a calendar year/DOY."""

    gps_epoch = date(1980, 1, 6)
    day = year_doy_to_date(year, doy)
    delta_days = (day - gps_epoch).days
    if delta_days < 0:
        raise ValueError("GPS week is undefined before 1980-01-06.")
    return delta_days // 7, delta_days % 7


def station_variants(
    station: str,
    station_aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> StationVariants:
    """Return the station identifiers used for RINEX 2 and RINEX 3/4 names.

    ``station`` is a 4-character code or a 9-character RINEX id (stripped,
    case-insensitive; ``TypeError`` for a non-string, ``ValueError`` for a
    wrong pattern). A 9-character id is its own long name; a 4-character
    code gets the 9-character ids of ``station_aliases`` (validated like
    ``DownloadConfig.station_aliases``) or, failing that, of
    :data:`DEFAULT_STATION_ALIASES`.
    """
    from .validation import STATION9_PATTERN, normalize_station, normalize_station_aliases

    if not isinstance(station, str):
        raise TypeError(f"station must be a str, got {type(station).__name__}: {station!r}")
    raw = station.strip()
    if not raw:
        raise ValueError("Station name cannot be empty.")
    upper = normalize_station(raw, "station")
    aliases: dict[str, tuple[str, ...]] = {
        key: tuple(value if isinstance(value, (list, tuple)) else (value,))
        for key, value in DEFAULT_STATION_ALIASES.items()
    }
    aliases.update(normalize_station_aliases(station_aliases))

    if STATION9_PATTERN.match(upper):
        return StationVariants(original=raw, legacy4=upper[:4], rinex3_ids=(upper,))

    rinex3_ids = tuple(dict.fromkeys(alias.upper() for alias in aliases.get(upper, ())))
    notes: tuple[str, ...] = ()
    if not rinex3_ids:
        notes = (
            "No 9-character RINEX3/4 station id is known. "
            "Pass station_aliases={'%s': 'XXXX00CCC'} to enable long-name candidates." % upper,
        )
    return StationVariants(original=raw, legacy4=upper, rinex3_ids=rinex3_ids, notes=notes)


def _product_skip_reason(source: str, data_type: str, product: str) -> str | None:
    """Why ``source`` gets no candidate for ``product`` of ``data_type``, or ``None``.

    Shared by the candidate generator and :func:`unavailable_reason`,
    so both give the same answer:

    - the product catalogue lists the product as not publishing the type;
    - CODE serves only COD products and COD IONEX.

    For ``ionex``, ``product`` is a 3-letter analysis-centre code or a
    10-character product id.
    """
    product_id = str(product).strip().upper()
    centre = catalog.analysis_centre(product_id)
    if data_type == "ionex" and len(product_id) != 10:
        publishes = None  # a 3-letter analysis-centre code
    else:
        publishes = catalog.publishes(product_id, data_type)
    if publishes is False:
        return f"the product catalogue lists no {data_type} for {product_id}"
    if source == "code" and centre != _CODE_ANALYSIS_CENTRE:
        return "CODE serves only COD products"
    return None


_GPS_EPOCH = date(1980, 1, 6)


def _source_hosts(source: str) -> frozenset:
    """Hosts of the archive base URLs of ``source`` (read at call time)."""
    bases = {
        "bkg": (BKG_BASE_URL, BKG_IGSAC_BASE_URL),
        "cddis": (CDDIS_BASE_URL,),
        "code": (CODE_BASE_URL, CODE_MGEX_BASE_URL),
        "euref": (EUREF_BASE_URL,),
    }.get(source, ())
    return frozenset(urlparse(base).netloc for base in bases)


def _check_candidate_target(
    source: str, data_type: str, filename: str, url: str, local_path: Path, output_dir: Path
) -> None:
    """Refuse a candidate that could leave the archive or the project folder.

    The file name must be a plain name (no ``/``, ``\\``, ``:``, leading dot
    or ``..``), the URL must stay on the host of the source's archive base
    URL (so a CDDIS bearer token is never sent elsewhere), and the local
    path must stay under ``output_dir``.
    """
    problem = None
    if not catalog.is_plain_file_name(filename):
        problem = "it is not a plain archive file name"
    elif urlparse(url).netloc not in _source_hosts(source):
        problem = f"its URL {url!r} leaves the {source} archive host"
    else:
        root = os.path.normpath(os.path.abspath(output_dir))
        target = os.path.normpath(os.path.abspath(local_path))
        if os.path.commonpath((root, target)) != root:
            problem = "its local path leaves output_dir"
    if problem is not None:
        raise ValueError(
            f"Refused {source} {data_type} candidate {filename!r}: {problem}. "
            "Station ids and product_preferences items must be plain ids (letters and digits)."
        )


def _effective_products(
    data_type: str,
    product_preferences: Mapping[str, Sequence[str]] | None,
) -> tuple[str, ...] | None:
    """The product list the generator uses for ``data_type`` (None: obs/nav)."""
    preferences = product_preferences or {}
    if data_type in preferences:
        products = preferences[data_type]
        if isinstance(products, str):
            products = (products,)
        return tuple(str(product) for product in products)
    if data_type == "ionex":
        return DEFAULT_IONEX_PRODUCTS
    if data_type in DEFAULT_PRODUCT_PREFERENCES:
        return DEFAULT_PRODUCT_PREFERENCES[data_type]
    return None


def unavailable_reason(
    data_type: str,
    sources: Sequence[str],
    product_preferences: Mapping[str, Sequence[str]] | None = None,
) -> str | None:
    """Why no selected source can provide ``data_type``, or ``None`` if one can.

    ``data_type`` is an engine key, ``sources`` the selected
    source keys and ``product_preferences`` the normalised engine-key mapping.
    The answer follows the candidate generator: a type is unavailable when no
    source in ``sources`` serves it (:data:`SOURCE_CAPABILITIES`), when its
    preference list is empty, or when no source has a candidate for any
    product of the list (the catalogue says none publishes the type, or the
    only source is CODE and none is a COD product). Observation stations are
    checked separately by :func:`station_reason`.
    """
    if isinstance(sources, str):
        sources = (sources,)
    sources = tuple(sources)
    serving = [source for source in sources if data_type in SOURCE_CAPABILITIES.get(source, frozenset())]
    if not serving:
        details = "; ".join(
            f"{source}: {_CAPABILITY_GAPS.get((source, data_type), f'{data_type} is not served')}"
            for source in sources
        )
        return f"{data_type!r} is not available from sources={sources!r}" + (f" ({details})" if details else "")
    if product_preferences and data_type in product_preferences:
        listed = product_preferences[data_type]
        if isinstance(listed, str):
            listed = (listed,)
        if not tuple(listed):
            return f"product_preferences[{data_type!r}] is empty"
    products = _effective_products(data_type, product_preferences)
    if products is None:  # obs and nav: every serving source has names
        return None
    details = []
    for source in sources:
        if source not in serving:
            details.append(
                f"{source}: {_CAPABILITY_GAPS.get((source, data_type), f'{data_type} is not served')}"
            )
            continue
        reasons = [_product_skip_reason(source, data_type, product) for product in products]
        if any(reason is None for reason in reasons):
            return None
        details.append(f"{source}: " + "; ".join(dict.fromkeys(reasons)))
    return (
        f"{data_type!r} is not available from sources={sources!r} for products {products!r} "
        f"({'; '.join(details)})"
    )


def _legacy_obs_allowed(variants: StationVariants) -> bool:
    """Whether RINEX 2 short names are generated for a requested station.

    A 4-character request always gets them. A 9-character id gets them only
    when its monument/receiver digits (characters 5-6) are "00": the short
    name ``abcdDDD0.YYd`` carries no monument number, so it normally belongs
    to the 00 receiver.
    """
    requested = variants.original.strip().upper()
    if len(requested) != 9:
        return True
    return requested[4:6] == "00"


def _no_legacy_obs_note(requested: str) -> str:
    return (
        f"no RINEX 2 name exists for {requested}: RINEX 2 short names cover monument/receiver 00 "
        f"only; request the 4-character id to accept the short-name file"
    )


def station_reason(variants: StationVariants, rinex_preference: Sequence[str]) -> str | None:
    """Why no observation file name exists for a station, or ``None``.

    ``variants`` from :func:`station_variants`,
    ``rinex_preference`` the normalised RINEX preference. Follows the
    candidate generator: long names need a 9-character id and "3" or "4";
    RINEX 2 names need "2" and, for a 9-character id, monument/receiver
    digits "00" (:func:`_legacy_obs_allowed`).
    """
    preference = tuple(str(version) for version in rinex_preference)
    if variants.rinex3_ids and any(version in {"3", "4"} for version in preference):
        return None
    legacy_allowed = _legacy_obs_allowed(variants)
    if "2" in preference and legacy_allowed:
        return None
    if not variants.rinex3_ids:
        return (
            f"station {variants.original!r} has no known 9-character RINEX id and '2' is not in "
            f"rinex_version_preference={preference!r}; pass "
            f"station_aliases={{{variants.legacy4!r}: '{variants.legacy4}00CCC'}} or add '2'"
        )
    if "2" in preference and not legacy_allowed:
        return (
            f"{_no_legacy_obs_note(variants.original.strip().upper())}, or add '3' or '4' to "
            f"rinex_version_preference={preference!r}"
        )
    return (
        f"no observation file name for station {variants.original!r} with "
        f"rinex_version_preference={preference!r}"
    )


def _rinex_long_then_legacy(preference: Sequence[str]) -> tuple[str, ...]:
    seen_long = False
    ordered: list[str] = []
    for version in preference:
        if version in {"3", "4"}:
            if not seen_long:
                ordered.append(version)
                seen_long = True
        else:
            ordered.append(version)
    return tuple(ordered)


def _obs_base_url(source: str, year: int, doy: int, filename: str) -> str:
    """Directory of a daily observation file.

    BKG: ``IGS/obs/YYYY/DDD/``; BKG EUREF (source ``euref``):
    ``EUREF/obs/YYYY/DDD/``. CDDIS: ``YYYY/DDD/YYd/`` for Hatanaka files
    (``.crx.gz``, ``.YYd.Z``, ``.YYd.gz``) and ``YYYY/DDD/YYo/`` for plain
    RINEX (``.rnx.gz``, ``.YYo.Z``, ``.YYo.gz``).
    """
    if source == "bkg":
        return f"{BKG_BASE_URL}/obs/{year}/{doy:03d}/"
    if source == "euref":
        return f"{EUREF_BASE_URL}/obs/{year}/{doy:03d}/"
    if source == "cddis":
        yy = year % 100
        stem = filename.rsplit(".", 1)[0] if filename.endswith((".gz", ".Z")) else filename
        kind = "d" if stem.endswith(".crx") or stem.endswith(f".{yy:02d}d") else "o"
        return f"{CDDIS_BASE_URL}/data/daily/{year}/{doy:03d}/{yy:02d}{kind}/"
    raise ValueError(f"Source {source!r} serves no observation files.")


# Long-name prefixes of the broadcast navigation products (catalog.NAV_PRODUCTS).
_NAV_LONG_PREFIXES = {
    "BRDC00IGS": "BRDC00IGS_R",
    "BRD400DLR": "BRD400DLR_S",
    "BRDM00DLR": "BRDM00DLR_S",
}


def _product_base_urls(
    source: str,
    day: date,
    doy: int,
    product: str | None = None,
    data_type: str | None = None,
) -> tuple[str, ...]:
    """Directories of a product file stamped ``day``/``doy``, in trial order.

    - BKG: ``IGS/products/WWWW/`` (IGS combined products), then
      ``IGSac/products/WWWW/`` (analysis-centre products).
    - CDDIS: ``products/WWWW/``; then ``products/mgex/WWWW/`` for MGEX
      campaign ids (``AAA?MGX???``) of GPS weeks before 2238 only (from
      week 2238 the MGEX products sit in ``products/WWWW/`` and
      ``products/mgex/WWWW/`` answers 404); then
      ``products/bias/YYYY/`` for every ``dcb`` product and for
      CAS and DLR ``bia``.
    - CODE: ``CODE_MGEX/CODE/YYYY/``, ``CODE/YYYY/``, ``CODE/``; ORBEX
      attitude (``att``) only in the year directory of its product line:
      ``CODE_MGEX/CODE/YYYY/`` for MGEX campaign ids, ``CODE/YYYY/`` for
      the operational ones (COD0OPSFIN).

    GPS weeks have four digits. OPS long names exist in
    ``products/WWWW/`` only from GPS week 2238 (2022-11-27); earlier days
    have short names there, which are not generated.
    """
    year = day.year
    gps_week, _ = gps_week_day(year, doy)
    week = f"{gps_week:04d}"
    if source == "bkg":
        return (f"{BKG_BASE_URL}/products/{week}/", f"{BKG_IGSAC_BASE_URL}/products/{week}/")
    if source == "cddis":
        urls = [f"{CDDIS_BASE_URL}/products/{week}/"]
        if (
            product is not None
            and catalog.is_mgex_campaign(str(product))
            and gps_week < CDDIS_MGEX_MERGED_WEEK
        ):
            urls.append(f"{CDDIS_BASE_URL}/products/mgex/{week}/")
        if product is not None and (
            data_type == "dcb"
            or (data_type == "bia" and catalog.analysis_centre(str(product)) in ("CAS", "DLR"))
        ):
            urls.append(f"{CDDIS_BASE_URL}/products/bias/{year}/")
        return tuple(urls)
    if source == "code":
        if data_type == "att":  # ORBEX files sit with their product line's year files only
            if product is not None and not catalog.is_mgex_campaign(str(product)):
                return (f"{CODE_BASE_URL}/{year}/",)
            return (f"{CODE_MGEX_BASE_URL}/{year}/",)
        return (
            f"{CODE_MGEX_BASE_URL}/{year}/",
            f"{CODE_BASE_URL}/{year}/",
            f"{CODE_BASE_URL}/",
        )
    raise ValueError(f"Unsupported source {source!r}.")


def _ionex_base_urls(source: str, year: int, doy: int) -> tuple[str, ...]:
    """Return the IONEX/GIM directory URL(s) for a source, or empty if unknown."""

    if source == "cddis":
        return (f"{CDDIS_BASE_URL}/products/ionex/{year}/{doy:03d}/",)
    if source == "code":
        return (f"{CODE_BASE_URL}/{year}/",)
    return ()


def _ionex_filenames(analysis_center: str, year: int, doy: int, source: str = "cddis") -> tuple[str, ...]:
    """Return candidate IONEX filenames for one IONEX preference item.

    - A 10-character product id (e.g. ``COD0OPSFIN``) gives its own long
      names only: ``{ID}_{YYYYDDD}0000_01D_{SMP}_GIM.INX.gz``.
    - A 3-letter analysis-centre code ``AAA`` gives the final, then the
      rapid long names (``AAA0OPSFIN``/``AAA0OPSRAP``), then the legacy short
      name: ``aaagDDD0.YYi.Z`` / ``.gz`` at CDDIS, ``AAAGDDD0.YYI.Z`` at CODE.

    The samplings ``SMP`` are 01H and 02H, plus the high-rate maps of the
    centres listed in the catalogue (UPC 15M, CAS 30M).
    """

    label = str(analysis_center).strip().upper()
    yy = year % 100
    stamp = f"{year}{doy:03d}0000"
    samplings = catalog.ionex_samplings(label)
    if len(label) == 10:
        return tuple(f"{label}_{stamp}_01D_{sampling}_GIM.INX.gz" for sampling in samplings)

    centre = label[:3]
    long_names = tuple(
        f"{centre}0OPS{kind}_{stamp}_01D_{sampling}_GIM.INX.gz"
        for kind in ("FIN", "RAP")
        for sampling in samplings
    )
    if source == "code":
        legacy_names: tuple[str, ...] = (f"{centre}G{doy:03d}0.{yy:02d}I.Z",)
    else:
        legacy_names = (
            f"{centre.lower()}g{doy:03d}0.{yy:02d}i.Z",
            f"{centre.lower()}g{doy:03d}0.{yy:02d}i.gz",
        )
    return long_names + legacy_names


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def parse_directory_listing(html: str, base_url: str | None = None) -> list[str]:
    """Extract file links from a simple Apache-style directory listing.

    Leading and trailing whitespace of an ``href`` is dropped, as a browser
    does (CDDIS pages write ``href="NAME.gz "``); a link that is empty after
    that is skipped.
    """

    parser = _HrefParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.hrefs:
        href = href.strip()
        if not href or href in {"../", "./"}:
            continue
        name = urlparse(href).path.rsplit("/", 1)[-1]
        if not name or name in {"..", "."}:
            continue
        links.append(urljoin(base_url, href) if base_url else href)
    return links


def __getattr__(name: str):
    # ``DownloadReport`` was renamed to ``ArchiveDownloadReport`` to free the
    # name for the session-level report in :mod:`gnx_py.downloaders.config`.
    if name == "DownloadReport":
        warnings.warn(
            "gnx_py.downloaders.core.DownloadReport is deprecated; use "
            "ArchiveDownloadReport instead. gnx_py.downloaders.DownloadReport "
            "now names the session-level report returned by DownloadSession.run().",
            DeprecationWarning,
            stacklevel=2,
        )
        return ArchiveDownloadReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
