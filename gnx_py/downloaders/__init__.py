"""GNSS archive download helpers.

The public entry point is :class:`GNSSDownloader` with a
:class:`GNSSDownloadConfig` configuration object.
"""

from .core import (
    CDDISAuthenticationError,
    CDDISCredentials,
    DownloadCandidate,
    DownloadReport,
    DownloadResult,
    GNSSDownloadConfig,
    GNSSDownloader,
    URLAvailability,
    gps_week_day,
    iter_doy_range,
    station_variants,
    year_doy_to_date,
)

__all__ = [
    "CDDISAuthenticationError",
    "CDDISCredentials",
    "DownloadCandidate",
    "DownloadReport",
    "DownloadResult",
    "GNSSDownloadConfig",
    "GNSSDownloader",
    "URLAvailability",
    "gps_week_day",
    "iter_doy_range",
    "station_variants",
    "year_doy_to_date",
]
