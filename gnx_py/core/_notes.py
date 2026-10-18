"""Warn-once state of one session run.

A condition met by several steps of one run (the session validation and the preprocessing of every system) is
warned once per run, not once per configuration object or once per Python process. The keys live on the run's
owner, normally the configuration, in ``owner.__dict__['_gnx_run_notes']``; :func:`clear_run_notes` empties them
and ``PPPSession.run``, ``SPPSession.run``, ``TECSession.run`` and ``SISSession.run`` call it on their configuration
at the start of every run. So a configuration reused for several runs (a loop over days, a notebook running a
session twice) warns again in each run, and sessions sharing one configuration do not silence each other: each
run starts from its own notes. ``TECSession`` validates in its constructor, and that validation belongs to the
session's first run: the constructor keeps a copy of the notes it produced (:func:`snapshot_run_notes`) and the
first ``run()`` starts from that copy (:func:`restore_run_notes`), whatever another session did to the
configuration in between.

"Warns again" means that ``warnings.warn`` is called again. Whether Python shows the repeated warning depends on
the warning filters: under the default filters a ``UserWarning`` with the same text attributed to the same code
line is shown once per Python process (``__warningregistry__`` of that module), so a script that runs a
configuration twice on the same files sees a repeated message once; messages naming a file or a day differ and are
shown again. The "always" filter (``warnings.simplefilter("always")``) shows every call.

Before v1.0.0 the same state was kept per configuration object for the run time of the object
(``ocean_loading`` ``_tide_notes``, ``tools`` ``_attitude_notes`` / ``_attitude_orbex_use``) or in module-level
sets for the whole process (``antex._SAT_FALLBACK_WARNED``, ``ionosphere.ionex._AUTO_MAPPING_WARNED``).

An owner without ``__dict__`` (``None``, a slotted object) keeps no state: every call warns.

The library imports this module as ``from . import _notes`` (never its functions by name), so the helpers do not
become attributes of the public modules and of ``gnx_py`` through their star imports.
"""
from __future__ import annotations

import warnings
from typing import Any, Hashable

__all__ = ["warn_once", "run_memo", "clear_run_notes", "snapshot_run_notes", "restore_run_notes", "NOTES_ATTR"]

#: attribute (in the owner's ``__dict__``, set without ``__setattr__``) holding {namespace: set | dict}
NOTES_ATTR = "_gnx_run_notes"


def _notes(owner: Any):
    d = getattr(owner, "__dict__", None)
    if not isinstance(d, dict):
        return None
    return d.setdefault(NOTES_ATTR, {})


def warn_once(owner: Any, key: Hashable, message: str, category: type = UserWarning, stacklevel: int = 2) -> bool:
    """``warnings.warn(message, category)`` unless ``key`` was already warned for ``owner`` in this run.

    ``stacklevel`` counts as for ``warnings.warn`` called by the caller of this function. Returns True when it
    warned."""
    notes = _notes(owner)
    if notes is not None:
        seen = notes.setdefault("warned", set())
        if key in seen:
            return False
        seen.add(key)
    warnings.warn(message, category, stacklevel=stacklevel + 1)
    return True


def run_memo(owner: Any, namespace: str) -> dict:
    """A dict kept on ``owner`` for the current run (cleared with the warn-once keys), for decisions that are
    made once per run and warn when made; a fresh dict when ``owner`` keeps no state."""
    notes = _notes(owner)
    if notes is None:
        return {}
    return notes.setdefault(("memo", namespace), {})


def clear_run_notes(owner: Any) -> None:
    """Forget every warn-once key and run memo of ``owner`` (start of a session run)."""
    d = getattr(owner, "__dict__", None)
    if isinstance(d, dict):
        d.pop(NOTES_ATTR, None)


def snapshot_run_notes(owner: Any) -> dict | None:
    """A copy of the warn-once keys and run memos of ``owner`` (None when it keeps no state or has no notes)."""
    d = getattr(owner, "__dict__", None)
    if not isinstance(d, dict) or not d.get(NOTES_ATTR):
        return None
    return {k: v.copy() for k, v in d[NOTES_ATTR].items()}


def restore_run_notes(owner: Any, snapshot: dict | None) -> None:
    """Start a run of ``owner`` from ``snapshot`` (:func:`snapshot_run_notes`): the current notes are replaced by
    a copy of it; ``None`` clears them (:func:`clear_run_notes`)."""
    clear_run_notes(owner)
    d = getattr(owner, "__dict__", None)
    if snapshot and isinstance(d, dict):
        d[NOTES_ATTR] = {k: v.copy() for k, v in snapshot.items()}
