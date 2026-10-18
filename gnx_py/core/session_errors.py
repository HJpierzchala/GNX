from __future__ import annotations

from datetime import date
from functools import wraps
from typing import Any, Callable, Dict, Optional


class SessionExecutionError(RuntimeError):
    """Unified error type for top-level Session/Controller execution failures.

    Attributes: ``session`` (name of the session), ``message``, ``context`` (settings of the run) and ``cause``
    (the original exception, also ``__cause__``). The error pickles: raised in a
    ``ProcessPoolExecutor`` worker it reaches the parent with these attributes; before v1.0.0 unpickling raised
    TypeError (the ``message`` argument was missing) and broke the pool. A ``cause`` or context value that does
    not survive a pickle round trip (for example an exception with required constructor arguments) travels as
    its text.
    """

    def __init__(
        self,
        session: str,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.session = session
        self.message = message
        self.context = context or {}
        self.cause = cause

        details = []
        for key, value in self.context.items():
            if value is not None:
                details.append(f"{key}={value}")
        details_txt = f" | {'; '.join(details)}" if details else ""
        cause_txt = f" | cause={type(cause).__name__}: {cause}" if cause is not None else ""
        super().__init__(f"[{session}] {message}{details_txt}{cause_txt}")

    def __reduce__(self):
        cause = self.cause
        if cause is not None and not _picklable(cause):
            cause = RuntimeError(f"{type(cause).__name__}: {cause}")
        context = {k: (v if _picklable(v) else repr(v)) for k, v in self.context.items()}
        return (_rebuild_session_error, (type(self), self.session, self.message, context, cause))


def _picklable(value: Any) -> bool:
    import pickle

    # The round trip, not only dumps: an exception with required constructor arguments (CompressedFileError,
    # SP3Error, ...) dumps fine but raises TypeError on load, which breaks the receiving process pool.
    try:
        pickle.loads(pickle.dumps(value))
    except Exception:  # noqa: BLE001 - any failure means "send the text instead"
        return False
    return True


def _rebuild_session_error(cls, session, message, context, cause):
    """Unpickling of :class:`SessionExecutionError` (and subclasses with the same constructor)."""
    err = cls(session, message, context=context, cause=cause)
    err.__cause__ = cause
    return err


#: Sequences longer than this are summarized in the error context.
_CONTEXT_MAX_ITEMS = 10


def _compact_context_value(value: Any) -> Any:
    """Summarize a long list/tuple so that the error message stays readable.

    A sequence of more than ``_CONTEXT_MAX_ITEMS`` items becomes
    ``"<n> items: a, b, ..., z"``; a sequence of dates becomes
    ``"<n> dates <first> ... <last>"``. Other values are returned unchanged.
    """
    if not isinstance(value, (list, tuple)) or len(value) <= _CONTEXT_MAX_ITEMS:
        return value
    if all(isinstance(item, date) for item in value):
        return f"{len(value)} dates {value[0].isoformat()} … {value[-1].isoformat()}"
    return f"{len(value)} items: {value[0]}, {value[1]}, …, {value[-1]}"


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
        "project_name",
        "output_dir",
        "product_types",
        "sources",
        "dates",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if hasattr(cfg, k):
            out[k] = _compact_context_value(getattr(cfg, k))
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

