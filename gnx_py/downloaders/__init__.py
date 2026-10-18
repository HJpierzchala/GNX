"""GNSS archive download helpers.

Recommended entry point:
    :class:`DownloadConfig` + :class:`DownloadSession` (defined in
    :mod:`gnx_py.downloaders.config`). ``DownloadSession(config).run()``
    returns a :class:`DownloadReport`, following the same config + session +
    ``run()`` pattern as ``PPPConfig``/``PPPSession``,
    ``TECConfig``/``TECSession`` and ``SISConfig``/``SISSession``.
    ``DownloadConfig`` validates every setting at construction and on
    assignment, before any I/O. The report gives one :class:`RequestOutcome`
    per requested (type, day[, station]) through ``outcomes()``, the paths for
    a processing session through ``files("sp3")`` and similar, the
    :class:`UnavailableRequest` entries of data no selected archive provides,
    and the consistent product set (:class:`ProductSetChoice`) chosen for
    orbits, clocks, phase biases and attitude.

Low-level engine:
    :class:`GNSSDownloadConfig` + :class:`GNSSDownloader` (defined in
    :mod:`gnx_py.downloaders.core`) work directly in year/day-of-year and
    archive data-type terms and return an :class:`ArchiveDownloadReport`.
    ``DownloadSession`` builds these internally, one per requested day.

Credential exchange:
    A session exchanges username/password (or a credentials file) for an
    Earthdata bearer token once per ``run()``, and only when "cddis" is a
    source and no transport was injected; otherwise the credentials are not
    used and a ``UserWarning`` says so. An existing valid token of the account
    is reused before a new one is created.
    ``fetch_earthdata_token`` (username/password -> bearer token exchange)
    lives in the separate :mod:`gnx_py.downloaders.earthdata_auth` module and
    is loaded lazily on first access, so importing this package (or
    ``gnx_py`` as a whole) never pulls in the credential-exchange code path
    unless it is actually used. It is deliberately not listed in ``__all__``:
    a wildcard import, such as the one in ``gnx_py/__init__.py``, would
    otherwise load it eagerly. Import it explicitly with
    ``from gnx_py.downloaders import fetch_earthdata_token``.

Deprecated aliases:
    ``ProjectDownloadConfig``, ``ProjectDownloader`` and
    ``ProjectDownloadReport`` resolve to ``DownloadConfig``,
    ``DownloadSession`` and ``DownloadReport`` with a ``DeprecationWarning``,
    here as well as from ``gnx_py.download`` and the top-level ``gnx_py``
    package. They are not listed in ``__all__`` either, so wildcard imports
    stay warning-free.
"""

import warnings

from .config import DownloadConfig, DownloadReport, DownloadSession, RequestOutcome
from .core import (
    ArchiveDownloadReport,
    CDDISAuthenticationError,
    CDDISAuthError,
    CDDISCredentials,
    CDDISDownloadError,
    DownloadCandidate,
    DownloadResult,
    DownloadTransport,
    GNSSDownloadConfig,
    GNSSDownloader,
    RemoteFileMissing,
    RequestsDownloadTransport,
    UnavailableRequest,
    URLAvailability,
    gps_week_day,
    iter_doy_range,
    station_variants,
    year_doy_to_date,
)
from .product_sets import ProbeFailure, ProductSetChoice
from .project import PRODUCT_TYPE_ALIASES, read_credentials_file

__all__ = [
    "ArchiveDownloadReport",
    "CDDISAuthenticationError",
    "CDDISAuthError",
    "CDDISCredentials",
    "CDDISDownloadError",
    "DownloadCandidate",
    "DownloadConfig",
    "DownloadReport",
    "DownloadResult",
    "DownloadSession",
    "DownloadTransport",
    "GNSSDownloadConfig",
    "GNSSDownloader",
    "PRODUCT_TYPE_ALIASES",
    "ProbeFailure",
    "ProductSetChoice",
    "RemoteFileMissing",
    "RequestOutcome",
    "RequestsDownloadTransport",
    "URLAvailability",
    "UnavailableRequest",
    "gps_week_day",
    "iter_doy_range",
    "read_credentials_file",
    "station_variants",
    "year_doy_to_date",
]

# Deprecated name -> current name, resolved on access by ``_lazy_attribute``.
_DEPRECATED_ALIASES = {
    "ProjectDownloadConfig": "DownloadConfig",
    "ProjectDownloader": "DownloadSession",
    "ProjectDownloadReport": "DownloadReport",
}


def _lazy_attribute(module_name: str, name: str):
    """Resolve the names kept out of ``__all__`` on attribute access.

    Shared by this package's ``__getattr__``, the ``gnx_py.download``
    compatibility module and the top-level ``gnx_py`` package, so all of them
    report deprecations at the caller's line.
    """
    if name == "fetch_earthdata_token":
        from .earthdata_auth import fetch_earthdata_token

        return fetch_earthdata_token
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{module_name}.{name} is deprecated; use {new_name} instead.",
            DeprecationWarning,
            stacklevel=3,  # skip this helper and the module-level __getattr__
        )
        return globals()[new_name]
    raise AttributeError(f"module {module_name!r} has no attribute {name!r}")


def __getattr__(name: str):
    return _lazy_attribute(__name__, name)
