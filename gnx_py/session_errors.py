from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Optional


class SessionExecutionError(RuntimeError):
    """Unified error type for top-level Session/Controller execution failures."""

    def __init__(
        self,
        session: str,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.session = session
        self.context = context or {}
        self.cause = cause

        details = []
        for key, value in self.context.items():
            if value is not None:
                details.append(f"{key}={value}")
        details_txt = f" | {'; '.join(details)}" if details else ""
        cause_txt = f" | cause={type(cause).__name__}: {cause}" if cause is not None else ""
        super().__init__(f"[{session}] {message}{details_txt}{cause_txt}")


def _extract_session_context(obj: Any) -> Dict[str, Any]:
    cfg = getattr(obj, "config", None)
    if cfg is None:
        return {}

    keys = (
        "obs_path",
        "nav_path",
        "sp3_path",
        "sys",
        "orbit_type",
        "station_name",
        "system",
        "orb_path_0",
        "orb_path_1",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if hasattr(cfg, k):
            out[k] = getattr(cfg, k)
    return out


def guarded_session_run(session_name: Optional[str] = None) -> Callable:
    """Decorator wrapping run() with a unified, user-facing error type."""

    def _decorator(func: Callable) -> Callable:
        @wraps(func)
        def _wrapped(self, *args, **kwargs):
            name = session_name or self.__class__.__name__
            try:
                return func(self, *args, **kwargs)
            except SessionExecutionError:
                raise
            except Exception as exc:
                raise SessionExecutionError(
                    session=name,
                    message="Session execution failed",
                    context=_extract_session_context(self),
                    cause=exc,
                ) from exc

        return _wrapped

    return _decorator

