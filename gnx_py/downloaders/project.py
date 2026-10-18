"""Input normalization and credential helpers behind ``DownloadConfig``.

``DownloadConfig`` and ``DownloadSession`` (in :mod:`gnx_py.downloaders.config`)
are the recommended entry point for GNSS downloads. This module holds the
logic they delegate to, so that the configuration/session module stays thin:

* the friendly product-type vocabulary (:data:`PRODUCT_TYPE_ALIASES`) and the
  default number of SP3 boundary days,
* normalization of user input: dates as :class:`datetime.datetime` /
  :class:`datetime.date`, product families, 4- or 9-character station ids,
* credential handling: :func:`read_credentials_file` and the resolution of a
  configuration's credential option into :class:`CDDISCredentials`.

Where :class:`~gnx_py.downloaders.core.GNSSDownloadConfig` speaks in terms of
year/day-of-year and the low-level data-type/source vocabulary, these helpers
translate what a user actually thinks in. SP3 orbit interpolation needs data
from the surrounding days, so ``DownloadSession`` also requests SP3 for the
previous and next day of every requested date (see
``DownloadConfig.sp3_adjacent_days``).

The former ``ProjectDownloadConfig``, ``ProjectDownloader`` and
``ProjectDownloadReport`` classes now live in :mod:`gnx_py.downloaders.config`
as ``DownloadConfig``, ``DownloadSession`` and ``DownloadReport``; the old
names remain importable from :mod:`gnx_py.downloaders` as deprecated aliases.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .core import CDDISCredentials
from .validation import PRODUCT_TYPE_ALIASES  # one vocabulary for configs and reports
from .validation import normalize_dates as _normalize_dates  # K-VAL re-export
from .validation import normalize_product_types as _normalize_product_types
from .validation import normalize_stations as _normalize_stations

if TYPE_CHECKING:
    from .config import DownloadConfig

DateLike = Union[datetime, date]

# Friendly product-type names -> internal core.py data_type keys: the
# PRODUCT_TYPE_ALIASES dict imported above from :mod:`.validation` (one object,
# shared by DownloadConfig, product_preferences keys and DownloadReport.files).

# SP3 always also fetches this many days before and after each requested date.
DEFAULT_SP3_ADJACENT_DAYS = 1


def read_credentials_file(path: Union[str, Path]) -> tuple:
    """Parse a small CDDIS/Earthdata credentials text file.

    Supported formats (checked in this order):

    1. ``KEY=VALUE`` lines using ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD``.
    2. Two non-empty lines: login on line 1, password on line 2. Neither line
       may contain ``=``.
    3. One line containing a ``,``/``:``/tab/space separated ``login`` and
       ``password`` pair.

    Blank lines and lines starting with ``#`` are ignored. The file is read
    as UTF-8; a byte-order mark written by some Windows editors is skipped.
    The file's own permissions are the caller's responsibility; this
    function only reads it.

    Passwords that contain ``=`` or start with ``#`` cannot use formats 2
    and 3 (such a line is taken as a ``KEY=VALUE`` line or a comment). Use
    format 1 for them: each line is split at its first ``=`` only, so
    ``EARTHDATA_PASSWORD=a=b`` gives the password ``a=b`` and
    ``EARTHDATA_PASSWORD=#secret`` gives ``#secret``.

    Format 3 splits at the first separator it finds, trying ``,`` then
    ``:`` then tab then space, so ``user:pa,ss`` gives the login
    ``user:pa``. A password containing ``,``, ``:``, a tab or a space must
    use format 2 or format 1. Every format strips blanks at both ends of
    the login and the password, so a password that begins or ends with a
    blank cannot be stored in this file (give ``DownloadConfig`` the
    ``username``/``password`` pair or a ``token`` instead).

    Returns
    -------
    tuple[str, str]
        ``(username, password)``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError(f"Credentials file {path} is empty.")

    kv = {}
    for ln in lines:
        if "=" in ln:
            key, _, value = ln.partition("=")
            kv[key.strip().upper()] = value.strip()
    if "EARTHDATA_USERNAME" in kv and "EARTHDATA_PASSWORD" in kv:
        return kv["EARTHDATA_USERNAME"], kv["EARTHDATA_PASSWORD"]

    if len(lines) >= 2 and "=" not in lines[0] and "=" not in lines[1]:
        return lines[0], lines[1]

    if len(lines) == 1:
        for separator in (",", ":", "\t", " "):
            if separator in lines[0]:
                login, _, password = lines[0].partition(separator)
                login, password = login.strip(), password.strip()
                if login and password:
                    return login, password

    raise ValueError(
        f"Could not parse credentials from {path}. Expected 'login' and "
        "'password' on two separate lines, 'login,password' on one line, or "
        "EARTHDATA_USERNAME=.../EARTHDATA_PASSWORD=... lines."
    )


def _resolve_credentials(config: "DownloadConfig") -> "CDDISCredentials | None":
    """Resolve a download configuration's credential option into CDDISCredentials.

    * ``token``: used as is, no request.
    * ``credentials_file``: read here (never at construction, so a file
      written after ``DownloadConfig`` was built works; a missing file raises
      ``FileNotFoundError``), then exchanged like a username/password.
    * ``username`` + ``password``: exchanged for a bearer token with one
      request to Earthdata Login, through a local import of
      :mod:`gnx_py.downloaders.earthdata_auth`, kept out of this module's
      top-level namespace by design (see that module's docstring).

    ``DownloadSession.run`` calls this once per run, and only when
    ``"cddis"`` is a source and no transport was injected, so credentials are never exchanged when they cannot be used.
    Returns ``None`` when no explicit credentials were given, so the
    transport falls back to its own defaults (``EARTHDATA_TOKEN`` for
    :class:`~gnx_py.downloaders.core.RequestsDownloadTransport`).
    """
    if config.token:
        return CDDISCredentials(token=config.token)

    if config.credentials_file:
        username, password = read_credentials_file(config.credentials_file)
    else:
        username, password = config.username, config.password

    if username and password:
        from .earthdata_auth import fetch_earthdata_token  # local import by design

        token = fetch_earthdata_token(username, password, timeout=config.timeout)
        return CDDISCredentials(username=username, token=token)

    return None
