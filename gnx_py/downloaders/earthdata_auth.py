"""NASA Earthdata Login (URS) username/password to bearer-token exchange.

This is kept in its own module, separate from :mod:`gnx_py.downloaders.core`,
so that the main download pipeline stays free of embedded-credential-exchange
code. Import this module explicitly when you need to mint or reuse an
Earthdata bearer token from a username/password pair; everyday downloading
only needs an already-issued ``EARTHDATA_TOKEN``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import requests

from .core import CDDISAuthError, _NoNetrcSession

# NASA Earthdata Login (URS) user-token API. POST mints a new token; GET lists
# tokens already issued to the account (URS caps active tokens per user).
URS_TOKEN_ENDPOINT = "https://urs.earthdata.nasa.gov/api/users/token"
URS_TOKEN_LIST_ENDPOINT = "https://urs.earthdata.nasa.gov/api/users/tokens"
# Returns an existing valid token or mints one (contract K-URL); tried first.
URS_FIND_OR_CREATE_TOKEN_ENDPOINT = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"


def fetch_earthdata_token(
    username: str,
    password: str,
    *,
    session: "requests.Session | None" = None,
    timeout: float = 30.0,
) -> str:
    """Exchange Earthdata (URS) username/password for a bearer token, reusing a valid one.

    Requests, all with HTTP Basic auth, in this order:

    1. ``POST {URS_FIND_OR_CREATE_TOKEN_ENDPOINT}``: URS returns a token that
       is still valid or mints one, so repeated runs do not use up the
       per-account token cap. HTTP 401/403 here (wrong credentials) ends the
       exchange at once.
    2. Otherwise (e.g. 404/405 when the endpoint is not offered, or no token in
       the answer): ``GET {URS_TOKEN_LIST_ENDPOINT}`` and the first token whose
       ``expiration_date`` lies in the future. A date without a time
       (``MM/DD/YYYY``, ISO ``YYYY-MM-DD``) counts as expired from the start
       of that UTC day, so a token that ends today is not used.
    3. ``POST {URS_TOKEN_ENDPOINT}`` (mint a new token) only when none is
       usable.
    4. As a last resort, a listed token whose expiration date cannot be read
       (also when step 3 fails with a network error or an HTTP error).

    HTTP 429 or 5xx in step 1 or 2 means URS is unavailable: the exchange
    stops with :class:`CDDISAuthError` instead of minting a new token, which
    would count against the per-account cap.

    An answer whose body is not JSON counts as "no token" and the next step is
    tried; it is named in the final error.

    The module URLs are read at call time, so tests can point them at a local
    server. When ``session`` is ``None`` a session that never reads
    ``~/.netrc`` is used and closed afterwards; an injected session is not
    closed.

    The password is only ever sent as the Basic-auth header of these requests.
    Neither the user name nor the password appears in a log line or an
    exception message.

    Raises
    ------
    CDDISAuthError
        If no usable ``access_token`` is obtained, including network errors
        (``requests.RequestException``) and answers that are not JSON; the
        message names the endpoint and the HTTP status.
    """
    own_session = session is None
    http = _NoNetrcSession() if own_session else session
    secrets_ = tuple(value for value in (password, username) if value)
    attempts: list[str] = []
    try:
        response = _call(http, "POST", URS_FIND_OR_CREATE_TOKEN_ENDPOINT, username, password, timeout,
                         attempts, secrets_)
        if response.status_code == 200:
            token = _token_from(_json(response, attempts))
            if token:
                return token
        else:
            _stop_on(response.status_code, attempts)

        unreadable: list[str] = []
        response = _call(http, "GET", URS_TOKEN_LIST_ENDPOINT, username, password, timeout, attempts, secrets_)
        if response.status_code == 200:
            listed = _json(response, attempts)
            token, unreadable = _first_unexpired(listed)
            if token:
                return token
        else:
            _stop_on(response.status_code, attempts)

        try:
            response = _call(http, "POST", URS_TOKEN_ENDPOINT, username, password, timeout, attempts, secrets_)
        except CDDISAuthError:
            if unreadable:              # last resort (step 4); the network error is not fatal then
                return unreadable[0]
            raise
        if response.status_code == 200:
            token = _token_from(_json(response, attempts))
            if token:
                return token
        if unreadable:
            return unreadable[0]
        if response.status_code in (401, 403):
            raise _failure(attempts, "the user name or password was refused")
        if response.status_code == 429 or response.status_code >= 500:
            raise _failure(attempts, "the Earthdata login service is temporarily unavailable; try again later")
        raise _failure(attempts, "no valid token; delete expired tokens at urs.earthdata.nasa.gov if the "
                                 "account reached its token limit")
    finally:
        if own_session:
            http.close()


def _stop_on(status: int, attempts: list) -> None:
    """Steps 1 and 2: wrong credentials, or URS failing, end the exchange (no new token is minted)."""
    if status in (401, 403):
        raise _failure(attempts, "the user name or password was refused")
    if status == 429 or status >= 500:
        raise _failure(attempts, "the Earthdata login service is temporarily unavailable; try again later")


def _scrub(text: str, secrets_: tuple) -> str:
    for value in secrets_:
        text = text.replace(value, "***")
    return text


def _call(http, method: str, url: str, username: str, password: str, timeout: float,
          attempts: list, secrets_: tuple):
    """One URS request; a network error becomes CDDISAuthError naming the endpoint (never the secrets)."""
    try:
        if method == "POST":
            response = http.post(url, auth=(username, password), timeout=timeout)
        else:
            response = http.get(url, auth=(username, password), timeout=timeout)
    except requests.RequestException as exc:
        detail = _scrub(f"{type(exc).__name__}: {exc}", secrets_)
        attempts.append(f"{method} {url} -> {detail}")
        raise CDDISAuthError(
            "Could not obtain an Earthdata bearer token: " + "; ".join(attempts)
        ) from None
    attempts.append(f"{method} {url} -> HTTP {response.status_code}")
    return response


def _json(response, attempts: list):
    """The JSON body, or ``None`` (noted in ``attempts``) when the body is not JSON: the next step is tried,
    and the final CDDISAuthError names the endpoint, its status and "body is not JSON"."""
    try:
        return response.json()
    except ValueError:
        attempts[-1] += " (body is not JSON)"
        return None


def _token_from(payload) -> str | None:
    if isinstance(payload, dict):
        token = payload.get("access_token")
        if isinstance(token, str) and token:
            return token
    return None


def _expiry(value) -> "datetime | None":
    """Expiration instant of a URS token entry (``MM/DD/YYYY`` or ISO 8601), UTC.

    A date without a time means the start of that UTC day (conservative: URS does not say at what time of
    the day the token ends, and a token that already expired would turn every CDDIS file into auth_missing).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%Y %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc)      # a date alone parses as 00:00
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _first_unexpired(listed) -> tuple:
    """(first token whose expiration date is in the future, tokens whose date cannot be read)."""
    now = datetime.now(timezone.utc)
    unreadable: list[str] = []
    if not isinstance(listed, list):
        return None, unreadable
    for entry in listed:
        token = _token_from(entry)
        if not token:
            continue
        expires = _expiry(entry.get("expiration_date"))
        if expires is None:
            unreadable.append(token)
        elif expires > now:
            return token, unreadable
    return None, unreadable


def _failure(attempts: list, reason: str) -> CDDISAuthError:
    return CDDISAuthError(
        f"Could not obtain an Earthdata bearer token via username/password exchange ({reason}): "
        + "; ".join(attempts)
        + ". Verify EARTHDATA_USERNAME/EARTHDATA_PASSWORD (or the credentials file)."
    )
