from __future__ import annotations

import http.cookiejar
import logging
import netrc
import os
import shutil
import ssl
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import (
    HTTPCookieProcessor,
    HTTPPasswordMgrWithDefaultRealm,
    HTTPBasicAuthHandler,
    Request,
    build_opener,
)


LOGGER = logging.getLogger(__name__)

BKG_BASE_URL = "https://igs.bkg.bund.de/root_ftp/IGS"
CDDIS_BASE_URL = "https://cddis.nasa.gov/archive/gnss"
CODE_BASE_URL = "https://ftp.aiub.unibe.ch/CODE"
CODE_MGEX_BASE_URL = "https://ftp.aiub.unibe.ch/CODE_MGEX/CODE"

SUPPORTED_SOURCES = ("bkg", "cddis", "code")
SUPPORTED_DATA_TYPES = ("obs", "nav", "sp3", "clk", "erp", "bia", "dcb")
SUPPORTED_RINEX_PREFERENCES = ("3", "4", "2")

DEFAULT_PRODUCT_PREFERENCES = {
    "sp3": ("COD0MGXFIN", "GRG0MGXFIN", "GFZ0MGXRAP", "COD0OPSFIN", "IGS0OPSFIN"),
    "clk": ("COD0MGXFIN", "GRG0MGXFIN", "GFZ0MGXRAP", "COD0OPSFIN"),
    "erp": ("COD0MGXFIN", "COD0OPSFIN", "IGS0OPSFIN", "GFZ0MGXRAP"),
    "bia": ("COD0MGXFIN", "GRG0MGXFIN", "GFZ0MGXRAP", "CAS0MGXRAP"),
    "dcb": ("CAS0MGXRAP", "DLR0MGXFIN", "GFZ0MGXRAP", "CAS1MGXRAP"),
}

# Keep this intentionally tiny. Users can pass station_aliases for their network.
DEFAULT_STATION_ALIASES = {
    "BRUX": ("BRUX00BEL",),
}


class CDDISAuthenticationError(RuntimeError):
    """Raised when CDDIS access is requested without Earthdata credentials."""


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

    @classmethod
    def from_environment(cls) -> "CDDISCredentials | None":
        username = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        if username and password:
            return cls(username=username, password=password)
        return None

    @classmethod
    def from_netrc(cls, path: str | Path | None = None) -> "CDDISCredentials | None":
        try:
            auth = netrc.netrc(str(path) if path is not None else None)
        except (FileNotFoundError, netrc.NetrcParseError, OSError):
            return None

        login = auth.authenticators("urs.earthdata.nasa.gov")
        if not login:
            return None
        username, _, password = login
        if not username or not password:
            return None
        return cls(username=username, password=password, netrc_path=path)

    def resolved(self) -> "CDDISCredentials | None":
        if self.username and self.password:
            return self
        if self.netrc_path is not None:
            return self.from_netrc(self.netrc_path)
        return self.from_environment() or self.from_netrc()


@dataclass
class GNSSDownloadConfig:
    """Configuration for archive candidate generation and download execution."""

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
    timeout: float = 30.0
    check_availability: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.data_types = _normalize_requested(self.data_types, SUPPORTED_DATA_TYPES, "data type")
        self.sources = _normalize_requested(self.sources, SUPPORTED_SOURCES, "source")
        self.rinex_version_preference = _normalize_requested(
            tuple(str(value) for value in self.rinex_version_preference),
            SUPPORTED_RINEX_PREFERENCES,
            "RINEX preference",
        )

        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be None or a positive integer.")

        # Validate the date range once during config construction.
        list(iter_doy_range(self.year, self.doy_start, self.doy_end))


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

    @property
    def filename(self) -> str:
        return self.local_path.name

    @property
    def group_key(self) -> tuple[str, int, int, str | None]:
        return (self.data_type, self.year, self.doy, self.station)


@dataclass(frozen=True)
class URLAvailability:
    """Result of a lightweight URL availability check."""

    exists: bool
    message: str = ""
    http_status: int | None = None


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


@dataclass
class DownloadReport:
    """Summary object returned by :meth:`GNSSDownloader.download`."""

    candidates: list[DownloadCandidate]
    results: list[DownloadResult]

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


class UrllibDownloadTransport:
    """Small urllib-based transport with optional Earthdata auth support."""

    def __init__(
        self,
        timeout: float = 30.0,
        credentials: CDDISCredentials | None = None,
        user_agent: str = "gnx-py/0.1 GNSSDownloader",
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.credentials = credentials.resolved() if credentials else (
            CDDISCredentials.from_environment() or CDDISCredentials.from_netrc()
        )
        self._default_opener = build_opener()
        self._cddis_opener = self._build_cddis_opener(self.credentials)

    def has_credentials(self, candidate: DownloadCandidate) -> bool:
        if not candidate.requires_auth:
            return True
        return bool(self.credentials and self.credentials.username and self.credentials.password)

    def exists(self, candidate: DownloadCandidate) -> URLAvailability:
        if candidate.requires_auth and not self.has_credentials(candidate):
            return URLAvailability(
                exists=False,
                message=(
                    "CDDIS requires Earthdata Login credentials. Provide "
                    "EARTHDATA_USERNAME/EARTHDATA_PASSWORD, a .netrc entry for "
                    "urs.earthdata.nasa.gov, or CDDISCredentials at runtime."
                ),
                http_status=401,
            )

        try:
            with self._open(candidate, method="HEAD") as response:
                status = getattr(response, "status", 200)
                return URLAvailability(True, http_status=status)
        except HTTPError as exc:
            if exc.code == 405:
                return self._range_probe(candidate)
            if exc.code in {401, 403} and candidate.requires_auth:
                return URLAvailability(False, "CDDIS authentication failed.", exc.code)
            if exc.code == 404:
                return URLAvailability(False, "not found", exc.code)
            return URLAvailability(False, str(exc), exc.code)
        except (URLError, TimeoutError, OSError) as exc:
            return URLAvailability(False, str(exc))

    def download(self, candidate: DownloadCandidate, destination: Path) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        try:
            with self._open(candidate, method="GET") as response, tmp_path.open("wb") as fout:
                shutil.copyfileobj(response, fout)
            tmp_path.replace(destination)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return destination.stat().st_size

    def _range_probe(self, candidate: DownloadCandidate) -> URLAvailability:
        try:
            with self._open(candidate, method="GET", headers={"Range": "bytes=0-0"}) as response:
                status = getattr(response, "status", 200)
                return URLAvailability(True, http_status=status)
        except HTTPError as exc:
            if exc.code == 404:
                return URLAvailability(False, "not found", exc.code)
            return URLAvailability(False, str(exc), exc.code)
        except (URLError, TimeoutError, OSError) as exc:
            return URLAvailability(False, str(exc))

    def _open(self, candidate: DownloadCandidate, method: str, headers: Mapping[str, str] | None = None):
        all_headers = {"User-Agent": self.user_agent}
        if headers:
            all_headers.update(headers)
        request = Request(candidate.url, method=method, headers=all_headers)
        opener = self._cddis_opener if candidate.requires_auth else self._default_opener
        return opener.open(request, timeout=self.timeout)

    @staticmethod
    def _build_cddis_opener(credentials: CDDISCredentials | None):
        cookie_jar = http.cookiejar.CookieJar()
        handlers = [HTTPCookieProcessor(cookie_jar)]
        if credentials and credentials.username and credentials.password:
            password_mgr = HTTPPasswordMgrWithDefaultRealm()
            for uri in ("https://urs.earthdata.nasa.gov", "https://cddis.nasa.gov"):
                password_mgr.add_password(None, uri, credentials.username, credentials.password)
            handlers.append(HTTPBasicAuthHandler(password_mgr))
        # Default SSL context is intentionally used; keep the import visible for callers
        # inspecting the implementation surface requested in the design notes.
        ssl.create_default_context()
        return build_opener(*handlers)


class GNSSDownloader:
    """Generate archive candidates, check availability, and download GNSS files."""

    def __init__(
        self,
        config: GNSSDownloadConfig,
        transport: DownloadTransport | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibDownloadTransport(
            timeout=config.timeout,
            credentials=config.credentials,
        )
        self.logger = logger or LOGGER
        if config.verbose:
            logging.basicConfig(level=logging.INFO)

    def build_candidates(self) -> list[DownloadCandidate]:
        candidates: list[DownloadCandidate] = []
        for day, doy in iter_doy_range(self.config.year, self.config.doy_start, self.config.doy_end):
            for source in self.config.sources:
                if "obs" in self.config.data_types:
                    candidates.extend(self._obs_candidates(source, day, doy))
                if "nav" in self.config.data_types:
                    candidates.extend(self._nav_candidates(source, day, doy))
                for data_type in ("sp3", "clk", "erp", "bia", "dcb"):
                    if data_type in self.config.data_types:
                        candidates.extend(self._product_candidates(source, data_type, day, doy))
        return sorted(candidates, key=lambda item: item.priority)

    def download(self) -> DownloadReport:
        candidates = self.build_candidates()
        grouped: dict[tuple[str, int, int, str | None], list[DownloadCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.group_key, []).append(candidate)

        results: list[DownloadResult] = []
        group_values = list(grouped.values())
        if self.config.max_workers and self.config.max_workers > 1 and len(group_values) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                processed = list(executor.map(self._process_group, group_values))
        else:
            processed = [self._process_group(group_candidates) for group_candidates in group_values]

        for group_candidates, (group_results, satisfied) in zip(group_values, processed):
            results.extend(group_results)
            if not satisfied and not group_results:
                candidate = group_candidates[0]
                results.append(
                    DownloadResult(
                        candidate=candidate,
                        status="missing",
                        message="No candidate generated for this request group.",
                        url=candidate.url,
                    )
                )
        return DownloadReport(candidates=candidates, results=results)

    def _process_group(self, candidates: Sequence[DownloadCandidate]) -> tuple[list[DownloadResult], bool]:
        results: list[DownloadResult] = []
        for candidate in candidates:
            if candidate.local_path.exists() and not self.config.overwrite:
                return [
                    DownloadResult(
                        candidate=candidate,
                        status="skipped_existing",
                        message="Local file already exists and overwrite=False.",
                        path=candidate.local_path,
                        url=candidate.url,
                    )
                ], True

            if self.config.dry_run and not self.config.check_availability:
                return [
                    DownloadResult(
                        candidate=candidate,
                        status="dry_run",
                        message="Candidate generated; availability not checked.",
                        path=candidate.local_path,
                        url=candidate.url,
                    )
                ], True

            if candidate.requires_auth and not self.transport.has_credentials(candidate):
                results.append(
                    DownloadResult(
                        candidate=candidate,
                        status="auth_missing",
                        message=(
                            "CDDIS requires Earthdata credentials. Use .netrc, "
                            "EARTHDATA_USERNAME/EARTHDATA_PASSWORD, or CDDISCredentials."
                        ),
                        path=candidate.local_path,
                        url=candidate.url,
                        http_status=401,
                    )
                )
                continue

            availability = self.transport.exists(candidate)
            if not availability.exists:
                status = "missing" if availability.http_status == 404 else "error"
                results.append(
                    DownloadResult(
                        candidate=candidate,
                        status=status,
                        message=availability.message,
                        path=candidate.local_path,
                        url=candidate.url,
                        http_status=availability.http_status,
                    )
                )
                continue

            if self.config.dry_run:
                return results + [
                    DownloadResult(
                        candidate=candidate,
                        status="dry_run",
                        message="Remote file exists; dry_run=True so it was not downloaded.",
                        path=candidate.local_path,
                        url=candidate.url,
                        http_status=availability.http_status,
                    )
                ], True

            try:
                self.logger.info("Downloading %s -> %s", candidate.url, candidate.local_path)
                bytes_written = self.transport.download(candidate, candidate.local_path)
            except Exception as exc:  # pragma: no cover - exact urllib errors vary by platform
                results.append(
                    DownloadResult(
                        candidate=candidate,
                        status="error",
                        message=str(exc),
                        path=candidate.local_path,
                        url=candidate.url,
                        http_status=availability.http_status,
                    )
                )
                continue

            return results + [
                DownloadResult(
                    candidate=candidate,
                    status="downloaded",
                    message="Downloaded successfully.",
                    path=candidate.local_path,
                    url=candidate.url,
                    http_status=availability.http_status,
                    bytes_written=bytes_written,
                )
            ], True

        return results, False

    def _obs_candidates(self, source: str, day: date, doy: int) -> list[DownloadCandidate]:
        if source == "code":
            return []

        candidates: list[DownloadCandidate] = []
        for station in self.config.stations:
            variants = station_variants(station, self.config.station_aliases)
            for version in _rinex_long_then_legacy(self.config.rinex_version_preference):
                if version in {"3", "4"}:
                    for station9 in variants.rinex3_ids:
                        candidates.extend(
                            self._obs_long_candidates(source, day, doy, variants.legacy4, station9, version)
                        )
                elif version == "2":
                    candidates.extend(self._obs_legacy_candidates(source, day, doy, variants))

            if not variants.rinex3_ids and any(v in {"3", "4"} for v in self.config.rinex_version_preference):
                self.logger.info(
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
    ) -> list[DownloadCandidate]:
        year = day.year
        prefix = f"{station9}_R_{year}{doy:03d}0000_01D_30S_MO"
        filenames = (f"{prefix}.crx.gz", f"{prefix}.rnx.gz")
        base_url = {
            "bkg": f"{BKG_BASE_URL}/obs/{year}/{doy:03d}/",
            "cddis": f"{CDDIS_BASE_URL}/data/daily/{year}/{doy:03d}/{year % 100:02d}d/",
        }[source]
        return [
            self._candidate(
                source=source,
                data_type="obs",
                day=day,
                doy=doy,
                station=station4,
                filename=filename,
                url=urljoin(base_url, filename),
                rinex_version=version,
                priority=self._priority(source, "obs", version, index),
                requires_auth=source == "cddis",
                notes=("RINEX3/4 long-name daily observation candidate.",),
            )
            for index, filename in enumerate(filenames)
        ]

    def _obs_legacy_candidates(
        self,
        source: str,
        day: date,
        doy: int,
        variants: StationVariants,
    ) -> list[DownloadCandidate]:
        year = day.year
        yy = year % 100
        station = variants.legacy4.lower()
        filenames = (
            f"{station}{doy:03d}0.{yy:02d}d.Z",
            f"{station}{doy:03d}0.{yy:02d}d.gz",
            f"{station}{doy:03d}0.{yy:02d}o.Z",
        )
        base_url = {
            "bkg": f"{BKG_BASE_URL}/obs/{year}/{doy:03d}/",
            "cddis": f"{CDDIS_BASE_URL}/data/daily/{year}/{doy:03d}/{yy:02d}d/",
        }[source]
        notes = variants.notes + ("RINEX2 legacy daily observation candidate.",)
        return [
            self._candidate(
                source=source,
                data_type="obs",
                day=day,
                doy=doy,
                station=variants.legacy4,
                filename=filename,
                url=urljoin(base_url, filename),
                rinex_version="2",
                priority=self._priority(source, "obs", "2", index + 20),
                requires_auth=source == "cddis",
                notes=notes,
            )
            for index, filename in enumerate(filenames)
        ]

    def _nav_candidates(self, source: str, day: date, doy: int) -> list[DownloadCandidate]:
        if source == "code":
            return []

        year = day.year
        yy = year % 100
        candidates: list[DownloadCandidate] = []
        for version in self.config.rinex_version_preference:
            if version == "3":
                filename = f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
                candidates.append(self._nav_candidate(source, day, doy, filename, "3", "BRDC00IGS"))
            elif version == "4":
                filename = f"BRD400DLR_S_{year}{doy:03d}0000_01D_MN.rnx.gz"
                candidates.append(self._nav_candidate(source, day, doy, filename, "4", "BRD400DLR"))
            elif version == "2":
                for index, filename in enumerate((
                    f"brdc{doy:03d}0.{yy:02d}n.Z",
                    f"brdc{doy:03d}0.{yy:02d}n.gz",
                    f"brdc{doy:03d}0.{yy:02d}g.Z",
                    f"brdc{doy:03d}0.{yy:02d}g.gz",
                )):
                    candidates.append(self._nav_candidate(source, day, doy, filename, "2", "BRDC_LEGACY"))
                    if source == "cddis":
                        candidates.append(
                            self._nav_candidate(
                                source,
                                day,
                                doy,
                                filename,
                                "2",
                                "BRDC_LEGACY",
                                cddis_annual_brdc=True,
                                priority_offset=index + 50,
                            )
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
            priority=self._priority(source, "nav", version, priority_offset),
            requires_auth=source == "cddis",
            notes=("Daily broadcast navigation candidate.",),
        )

    def _product_candidates(self, source: str, data_type: str, day: date, doy: int) -> list[DownloadCandidate]:
        year = day.year
        products = self.config.product_preferences.get(
            data_type,
            DEFAULT_PRODUCT_PREFERENCES[data_type],
        )
        candidates: list[DownloadCandidate] = []
        for product_index, product in enumerate(products):
            for filename in _product_filenames(product, data_type, year, doy):
                for base_url in _product_base_urls(source, day, doy):
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
                        )
                    )
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
    ) -> DownloadCandidate:
        local_path = self.config.output_dir / f"{day.year:04d}" / f"{doy:03d}" / source / data_type / filename
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
        )

    def _priority(self, source: str, data_type: str, rinex_version: str | None, index: int) -> int:
        source_rank = self.config.sources.index(source)
        type_rank = self.config.data_types.index(data_type)
        if rinex_version is not None:
            version_rank = self.config.rinex_version_preference.index(rinex_version)
        else:
            version_rank = 0
        return source_rank * 100_000 + type_rank * 10_000 + version_rank * 100 + index


def _normalize_requested(values: Sequence[str], allowed: Sequence[str], label: str) -> tuple[str, ...]:
    normalized = tuple(str(value).lower() for value in values)
    unknown = sorted(set(normalized) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {label}: {unknown}. Supported values: {allowed}.")
    return normalized


def year_doy_to_date(year: int, doy: int) -> date:
    """Convert a year/day-of-year pair to a calendar date."""

    if doy < 1:
        raise ValueError("doy must be >= 1.")
    first_day = date(year, 1, 1)
    result = first_day + timedelta(days=doy - 1)
    if result.year != year:
        raise ValueError(f"doy {doy} is outside year {year}.")
    return result


def iter_doy_range(year: int, doy_start: int, doy_end: int) -> Iterable[tuple[date, int]]:
    """Yield ``(date, doy)`` pairs for an inclusive day-of-year range."""

    if doy_start > doy_end:
        raise ValueError("doy_start must be <= doy_end.")
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
    """Return conservative station variants for RINEX2 and RINEX3/4 names."""

    raw = station.strip()
    if not raw:
        raise ValueError("Station name cannot be empty.")
    upper = raw.upper()
    aliases: dict[str, tuple[str, ...]] = {
        key: tuple(value if isinstance(value, (list, tuple)) else (value,))
        for key, value in DEFAULT_STATION_ALIASES.items()
    }
    if station_aliases:
        for key, value in station_aliases.items():
            aliases[key.upper()] = tuple(value if isinstance(value, (list, tuple)) else (value,))

    if len(upper) == 9:
        return StationVariants(original=raw, legacy4=upper[:4], rinex3_ids=(upper,))

    if len(upper) == 4:
        rinex3_ids = tuple(dict.fromkeys(alias.upper() for alias in aliases.get(upper, ())))
        notes: tuple[str, ...] = ()
        if not rinex3_ids:
            notes = (
                "No 9-character RINEX3/4 station id is known. "
                "Pass station_aliases={'%s': 'XXXX00CCC'} to enable long-name candidates." % upper,
            )
        return StationVariants(original=raw, legacy4=upper, rinex3_ids=rinex3_ids, notes=notes)

    raise ValueError(
        f"Station {station!r} must be a 4-character legacy code or a 9-character RINEX station id."
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


def _product_base_urls(source: str, day: date, doy: int) -> tuple[str, ...]:
    year = day.year
    gps_week, _ = gps_week_day(year, doy)
    if source == "bkg":
        return (f"{BKG_BASE_URL}/products/{gps_week}/",)
    if source == "cddis":
        return (f"{CDDIS_BASE_URL}/products/{gps_week}/",)
    if source == "code":
        return (
            f"{CODE_MGEX_BASE_URL}/{year}/",
            f"{CODE_BASE_URL}/{year}/",
            f"{CODE_BASE_URL}/",
        )
    raise ValueError(f"Unsupported source {source!r}.")


def _product_filenames(product: str, data_type: str, year: int, doy: int) -> tuple[str, ...]:
    stamp = f"{year}{doy:03d}0000"
    if data_type == "sp3":
        suffixes = ("01D_05M_ORB.SP3.gz", "01D_05M_ORB.SP3")
    elif data_type == "clk":
        suffixes = ("01D_30S_CLK.CLK.gz", "01D_30S_CLK.CLK")
    elif data_type == "erp":
        suffixes = ("01D_12H_ERP.ERP.gz", "01D_01D_ERP.ERP.gz", "01D_01D_ERP.ERP")
    elif data_type == "bia":
        suffixes = ("01D_01D_OSB.BIA.gz", "01D_01D_OSB.BIA")
    elif data_type == "dcb":
        suffixes = ("01D_01D_DCB.BSX.gz", "01D_01D_DCB.BSX", "01D_01D_DCB.BIA.gz", "01D_01D_DCB.BIA")
    else:
        raise ValueError(f"Unsupported product data type {data_type!r}.")
    return tuple(f"{product}_{stamp}_{suffix}" for suffix in suffixes)


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
    """Extract file links from a simple Apache-style directory listing."""

    parser = _HrefParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.hrefs:
        if href in {"../", "./"}:
            continue
        name = urlparse(href).path.rsplit("/", 1)[-1]
        if not name or name in {"..", "."}:
            continue
        links.append(urljoin(base_url, href) if base_url else href)
    return links
