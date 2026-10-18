"""Child processes of GNX-py.

Fork safety
-----------
Every program the library starts goes through :func:`run`. It calls :func:`subprocess.run` with the arguments
under which CPython starts the child with ``posix_spawn`` (macOS, glibc Linux) instead of ``fork`` + ``exec``
(CPython 3.12 ``Lib/subprocess.py`` ``_execute_child``): an absolute executable path, ``close_fds=False``,
``cwd=None``, no ``preexec_fn``, no ``pass_fds``, no ``start_new_session``/``process_group`` and no
user/group/umask change, with the standard streams piped (descriptors > 2). A forked child of a process that has
loaded PROJ (pyproj, cartopy) runs PROJ's ``atfork`` handler, which closed the SQLite cache and crashed the child
before ``exec`` (macOS, 2026-10-13: ``_posixsubprocess`` fork -> ``SQLiteHandleCache`` atfork -> SIGSEGV);
``posix_spawn`` runs no ``atfork`` handler. ``close_fds=False`` is safe: Python creates its descriptors
non-inheritable (PEP 446), so the child inherits only the three pipes. Where CPython does not use ``posix_spawn``
(``subprocess._USE_POSIX_SPAWN`` false: Windows, which has no fork; Linux without glibc, where ``_posixsubprocess``
uses ``vfork``; other POSIX systems, where it forks) the call is the ordinary one, so the guarantee holds on macOS
and glibc Linux.

The library starts no process pool: the SP3 interpolation of :class:`gnx_py.coordinates.SP3InterpolatorOptimized`
runs in the calling process (a ``multiprocessing`` 'spawn' pool forks as well on macOS:
``multiprocessing.util.spawnv_passfds`` -> ``_posixsubprocess.fork_exec``).
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Sequence

__all__ = ["run"]


def run(argv: Sequence[str], *, input: Optional[bytes] = None,
        timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run ``argv`` (``argv[0]`` an absolute path to an executable) and return the completed process with
    ``stdout`` and ``stderr`` as bytes. ``input`` is written to its standard input (bytes or None).

    The call meets CPython's conditions for ``posix_spawn`` (module docstring): no ``fork`` of this process.
    The exit status is not checked (``returncode`` is the caller's). ValueError for a relative or empty
    executable path, so a call can never fall back to ``fork`` through a ``PATH`` lookup."""
    argv = [os.fspath(a) for a in argv]
    if not argv or not argv[0] or not os.path.isabs(argv[0]):
        raise ValueError(f"gnx_py._proc.run needs an absolute executable path as argv[0] (got {argv[:1]!r})")
    return subprocess.run(argv, input=input, capture_output=True, check=False, timeout=timeout,
                          close_fds=False, cwd=None, start_new_session=False)

