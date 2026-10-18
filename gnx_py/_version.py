"""Version of GNX-py (``gnx_py.__version__``).

The version is declared once, in ``pyproject.toml`` (``[project] version``). An installed package (wheel)
reports it through its distribution metadata (:mod:`importlib.metadata`). A source tree (a checkout on
``PYTHONPATH``, or an editable install) reads it from the ``pyproject.toml`` next to the package: there the
metadata is a build artifact (``gnx_py.egg-info``, the editable install's ``dist-info``) that keeps the version
of the last install until the package is reinstalled, or it belongs to another copy of the package. Order:

1. the ``pyproject.toml`` of the source tree that holds the package, when it declares ``gnx_py``;
2. the ``gnx_py`` distribution metadata;
3. ``"0+unknown"``.
"""
from __future__ import annotations

from importlib import metadata as _metadata
from pathlib import Path
from typing import Optional

__all__ = ["__version__"]

#: distribution name in ``pyproject.toml``
DISTRIBUTION = "gnx_py"
#: reported when neither a source tree nor the metadata gives a version (PEP 440 local version)
UNKNOWN_VERSION = "0+unknown"


def _metadata_version() -> Optional[str]:
    """Version of the installed ``gnx_py`` distribution, or None."""
    try:
        return _metadata.version(DISTRIBUTION) or None
    except _metadata.PackageNotFoundError:
        return None


def _source_tree_version(pyproject: Path) -> Optional[str]:
    """``[project] version`` of ``pyproject`` when it declares this distribution, else None."""
    if not pyproject.is_file():
        return None
    try:
        import tomllib

        with open(pyproject, "rb") as fh:
            project = tomllib.load(fh).get("project", {})
    except (OSError, ValueError):  # unreadable file, or tomllib.TOMLDecodeError (a ValueError)
        return None
    if not isinstance(project, dict) or project.get("name") != DISTRIBUTION:
        return None
    version = project.get("version")
    return version.strip() if isinstance(version, str) and version.strip() else None


def detect_version(package_init=None) -> str:
    """The version of the ``gnx_py`` package whose ``__init__.py`` is ``package_init`` (default: this one);
    the order is given in the module docstring."""
    init = Path(package_init) if package_init is not None else Path(__file__).with_name("__init__.py")
    source = _source_tree_version(init.resolve().parent.parent / "pyproject.toml")
    if source:
        return source
    return _metadata_version() or UNKNOWN_VERSION


__version__: str = detect_version()
