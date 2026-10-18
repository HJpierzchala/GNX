"""Manifest of verified downloads.

``<output_dir>/.gnx_downloads.json`` maps each file's path relative to
``output_dir`` (POSIX separators) to ``{"size", "mtime_ns", "sha256",
"check"}``. A file whose size and ``mtime_ns`` still match its entry passed
the full integrity check earlier, so a rerun can skip that check (the quick
check). Files without a matching entry get the full check once.

Rules:

* the manifest is read once per engine run (:meth:`DownloadManifest.load`);
* entries are recorded after each successful download or full check
  (:meth:`DownloadManifest.record`);
* :meth:`DownloadManifest.save` writes at the end of the run by an atomic
  replace that merges with the file on disk; concurrent sessions can only
  lose entries (last writer wins), which costs one full check later;
* ``overwrite=True`` ignores the manifest;
* a missing, unreadable or malformed manifest counts as empty.

This module is a leaf (standard library only); ``core`` imports it.

Used by :class:`gnx_py.downloaders.core.GNSSDownloader` (one load and one
save per engine run) and by :func:`gnx_py.downloaders.core.check_local_file`.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["MANIFEST_NAME", "DownloadManifest", "ManifestEntry"]

MANIFEST_NAME = ".gnx_downloads.json"


@dataclass(frozen=True)
class ManifestEntry:
    """One verified file: size and ``st_mtime_ns`` at check time, sha256, check name."""

    size: int
    mtime_ns: int
    sha256: "str | None" = None
    check: str = "full"


@dataclass
class DownloadManifest:
    """The verified-download manifest of one output folder."""

    root: Path
    entries: "dict[str, ManifestEntry]" = field(default_factory=dict)
    # Keys dropped by forget(): removed from the file on save unless recorded again.
    _removed: "set[str]" = field(default_factory=set, repr=False, compare=False)

    @property
    def path(self) -> Path:
        """Location of the manifest file, ``<root>/.gnx_downloads.json``."""
        return Path(self.root) / MANIFEST_NAME

    @classmethod
    def load(cls, root) -> "DownloadManifest":
        """Read the manifest of ``root``; empty when absent or unreadable."""
        manifest = cls(Path(root))
        manifest.entries = _read_entries(manifest.path)
        return manifest

    def key(self, path) -> str:
        """The manifest key of ``path``: relative to ``root``, POSIX separators."""
        return Path(path).relative_to(self.root).as_posix()

    def quick_valid(self, path) -> bool:
        """True when ``path`` exists and its size and ``mtime_ns`` match its entry."""
        entry = self.entries.get(self.key(path))
        if entry is None:
            return False
        try:
            stat = Path(path).stat()
        except OSError:
            return False
        return stat.st_size == entry.size and stat.st_mtime_ns == entry.mtime_ns

    def record(self, path, sha256: "str | None" = None, check: str = "full") -> ManifestEntry:
        """Record ``path`` (as it is on disk now) as verified; returns the entry."""
        stat = Path(path).stat()
        entry = ManifestEntry(stat.st_size, stat.st_mtime_ns, sha256, check)
        key = self.key(path)
        self.entries[key] = entry
        self._removed.discard(key)
        return entry

    def forget(self, path) -> None:
        """Drop the entry of ``path`` (e.g. after the file failed a check)."""
        key = self.key(path)
        self.entries.pop(key, None)
        self._removed.add(key)

    def save(self) -> None:
        """Merge with the manifest on disk and replace it atomically.

        Entries of this object win over entries on disk with the same key;
        keys passed to :meth:`forget` (and not recorded again) are dropped.
        Does nothing when there is nothing to write.
        """
        merged = _read_entries(self.path)
        for key in self._removed:
            merged.pop(key, None)
        merged.update(self.entries)
        if not merged and not self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: {"size": e.size, "mtime_ns": e.mtime_ns, "sha256": e.sha256, "check": e.check}
            for key, e in sorted(merged.items())
        }
        fd, tmp_name = _create_temp(self.path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
            os.replace(tmp_name, self.path)
            tmp_name = None
        finally:
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)


def _create_temp(path: Path) -> "tuple[int, Path]":
    """A new, unique temp file next to ``path``, opened for writing with mode 0o666 minus the umask.

    Not :func:`tempfile.mkstemp` (mode 0o600): in a project folder shared by several users the manifest must
    stay readable, or another user's run would see it as empty and its save would drop the other entries.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        tmp = path.parent / f"{path.name}.{secrets.token_hex(6)}.part"
        try:
            return os.open(tmp, flags, 0o666), tmp
        except FileExistsError:
            continue
    raise FileExistsError(f"no unique temp name next to {path}")


def _read_entries(path: Path) -> "dict[str, ManifestEntry]":
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    entries: dict = {}
    for key, value in raw.items():
        try:
            entries[str(key)] = ManifestEntry(
                size=int(value["size"]),
                mtime_ns=int(value["mtime_ns"]),
                sha256=value.get("sha256"),
                check=str(value.get("check", "full")),
            )
        except (TypeError, KeyError, ValueError, AttributeError):
            continue
    return entries
