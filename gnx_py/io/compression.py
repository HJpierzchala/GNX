"""Decompression of data and product files in memory.

Every GNX file reader opens its input through this module, so a file is accepted plain or compressed alike and
nothing is ever written next to the data (read-only directories and parallel reads of one file are safe).

The format is detected from the first bytes of the file, never from its name:

* gzip (``1f 8b``), multi-member files included (all members are concatenated, as ``gunzip`` does);
* Unix compress, ``.Z`` (``1f 9d``), through the ``ncompress`` package (Unlicense);
* bzip2 (``BZh`` and a block-size digit), multi-stream files included;
* zip (``PK 03 04``, or ``PK 05 06`` for an empty archive) with exactly one member file;
* anything else is plain and returned unchanged (an empty file gives ``b""``).

A UTF-8 byte order mark before the first line (written by some Windows editors; the formats are ASCII) is dropped
by :func:`read_bytes` and :func:`open_text`. Bytes after the last gzip member that are not another member (stray
bytes appended to a download, which ``gunzip`` ignores with "trailing garbage ignored") are ignored with one
:class:`CompressedFileWarning` naming the file; zero padding is ignored silently, as ``gzip`` does.

Hatanaka compression (CRINEX 1/3) is not undone here: that is the observation reader's job
(:mod:`gnx_py.rinex_obs`), because only it knows that the text is an observation file.

A file that has a compression signature but cannot be decompressed raises :class:`CompressedFileError`, a
``ValueError`` whose message names the file, the format and what the caller was reading. A missing or unreadable
file raises the usual ``OSError``. Unix compress has no end marker and no checksum, so a truncated ``.Z`` file
decompresses to its readable prefix without an error; the reader of the text must notice the cut.

Public API::

    read_bytes(path, what="file") -> bytes
    open_text(path, what="file", encoding="ascii", errors="replace") -> io.StringIO
    head_lines(path, what="file", limit=...) -> list[str]   (the head of a file, for header readers)
    detect_format(head) -> "gzip" | "compress" | "bzip2" | "zip" | "plain"
    decompress(raw, fmt) -> (bytes, notes)          (for readers that read the file bytes themselves)
    strip_bom(data) -> bytes
    normalise_line_ends(data) -> bytes
    verify_stream(path, what="file") -> (fmt, notes)   (whole-file integrity check, streamed; for the downloader)
"""
from __future__ import annotations

import bz2
import io
import os
import re
import warnings
import zipfile
import zlib

__all__ = ["BOM", "CompressedFileError", "CompressedFileWarning", "FORMATS", "decompress", "detect_format",
           "head_lines", "normalise_line_ends", "open_text", "read_bytes", "strip_bom", "verify_stream"]

#: The UTF-8 byte order mark.
BOM = b"\xef\xbb\xbf"

#: The formats :func:`detect_format` can return.
FORMATS = ("gzip", "compress", "bzip2", "zip", "plain")

_FORMAT_NAMES = {"gzip": "gzip", "compress": "Unix compress (.Z)", "bzip2": "bzip2", "zip": "zip"}


class CompressedFileError(ValueError):
    """A compressed file that cannot be decompressed; ``path``, ``fmt`` and ``what`` name the file, the format
    and what the caller was reading."""

    def __init__(self, path, fmt: str, what: str, reason: str):
        self.path = os.fspath(path)
        self.fmt = fmt
        self.what = what
        super().__init__(f"{what} {self.path}: {_FORMAT_NAMES.get(fmt, fmt)} data cannot be decompressed "
                         f"({reason})")


class CompressedFileWarning(UserWarning):
    """A readable defect of a compressed file (bytes after the end of its gzip data); the message names the file."""


def detect_format(head: bytes) -> str:
    """Compression format of a file from its first bytes (at least 4 for zip, 4 for bzip2, 2 otherwise)."""
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:2] == b"\x1f\x9d":
        return "compress"
    if head[:3] == b"BZh" and head[3:4].isdigit() and head[3:4] != b"0":
        return "bzip2"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    return "plain"


def _unzip(raw: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) != 1:
            names = ", ".join(m.filename for m in members) or "none"
            raise ValueError(f"a single member file is required, the archive holds {len(members)}: {names}")
        return zf.read(members[0])


def _uncompress(raw: bytes) -> bytes:
    try:
        import ncompress  # Unlicense; installed with hatanaka
    except ImportError as exc:
        raise ValueError("Unix compress needs the 'ncompress' package (pip install ncompress)") from exc
    return ncompress.decompress(raw)


def _gunzip_members(raw: bytes) -> tuple:
    """(data, number of stray bytes after the last member) of gzip data, member after member as ``gunzip`` reads
    them. A member cut before its end marker raises EOFError with the message of :mod:`gzip` ("Compressed file
    ended before the end-of-stream marker was reached"); a CRC or length error raises ``zlib.error``. Bytes after
    a member that do not start another member end the data: zero padding is not counted, anything else is (the
    caller warns)."""
    out = []
    data = raw
    while True:
        d = zlib.decompressobj(31)
        out.append(d.decompress(data))
        out.append(d.flush())
        if not d.eof:
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")
        data = d.unused_data
        if data[:2] != b"\x1f\x8b":
            break
    return b"".join(out), (len(data) if data.strip(b"\x00") else 0)


def _gunzip(raw: bytes) -> bytes:
    return _gunzip_members(raw)[0]


_DECODERS = {"gzip": _gunzip, "compress": _uncompress, "bzip2": bz2.decompress, "zip": _unzip}


def decompress(raw: bytes, fmt: str) -> tuple:
    """(data, notes) of ``raw`` in format ``fmt`` (:func:`detect_format`); ``notes``: readable remarks about the data
    (stray bytes after the gzip data). Raises what the decoder raises (EOFError, OSError, ValueError,
    ``zlib.error``, ``zipfile.BadZipFile``)."""
    if fmt == "plain":
        return raw, []
    if fmt == "gzip":
        data, stray = _gunzip_members(raw)
        notes = [f"{stray} byte(s) after the end of the gzip data ignored (as gunzip does: trailing garbage)"] \
            if stray else []
        return data, notes
    return _DECODERS[fmt](raw), []


def strip_bom(data: bytes) -> bytes:
    """``data`` without a leading UTF-8 byte order mark."""
    return data[len(BOM):] if data.startswith(BOM) else data


_CR_RUN_LF = re.compile(rb"\r+\n")


def normalise_line_ends(data: bytes) -> bytes:
    """``data`` with CR LF, CR CR LF (a CR LF file converted a second time) and bare CR line ends made LF."""
    if b"\r" not in data:
        return data
    data = data.replace(b"\r\n", b"\n")
    if b"\r" in data:                       # bare CR, or what CR CR LF left (CR LF)
        data = _CR_RUN_LF.sub(b"\n", data).replace(b"\r", b"\n")
    return data


def read_bytes(path, what: str = "file") -> bytes:
    """Content of ``path`` with gzip, Unix compress, bzip2 or single-member zip compression undone in memory.

    ``what`` names the file in errors (for example ``"SP3 file"``). Raises :class:`CompressedFileError` when the
    data cannot be decompressed; plain files are returned unchanged, except that a leading UTF-8 byte order mark is
    dropped (also after decompression). Stray bytes after the gzip data give one :class:`CompressedFileWarning`."""
    with open(path, "rb") as fh:
        raw = fh.read()
    fmt = detect_format(raw[:4])
    try:
        data, notes = decompress(raw, fmt)
    except (EOFError, OSError, ValueError, zlib.error, zipfile.BadZipFile) as exc:
        raise CompressedFileError(path, fmt, what, f"{type(exc).__name__}: {exc}") from exc
    for note in notes:
        warnings.warn(f"{what} {os.fspath(path)}: {note}", CompressedFileWarning, stacklevel=3)
    return strip_bom(data)


def open_text(path, what: str = "file", encoding: str = "ascii", errors: str = "replace") -> io.StringIO:
    """Decoded text of ``path`` (see :func:`read_bytes`) as an in-memory text stream.

    Line ends are read as by ``open(path, "rt")``: ``\\r\\n`` and ``\\r`` become ``\\n``; ``\\r\\r\\n`` (a CR LF file
    converted a second time) becomes one ``\\n`` as well, not an empty line after every line. Bytes that do not
    decode are replaced (``errors="replace"``) unless the caller asks otherwise."""
    return io.StringIO(normalise_line_ends(read_bytes(path, what)).decode(encoding, errors), newline=None)


def head_lines(path, what: str = "file", limit: int = 1 << 20) -> list:
    """Complete lines (latin-1) of the first ``limit`` decompressed bytes of ``path``, for readers of a header.

    Plain, gzip and bzip2 files are streamed, so only their head is read and decompressed; Unix compress (.Z) and
    zip have no streaming decoder here and go through :func:`read_bytes`. A leading UTF-8 byte order mark is
    dropped. A line cut by ``limit`` is left out. A damaged archive raises :class:`CompressedFileError`, a missing
    file ``OSError``."""
    import gzip

    with open(path, "rb") as fh:
        fmt = detect_format(fh.read(4))
    if fmt in ("compress", "zip"):
        data = read_bytes(path, what=what)[:limit]
    else:
        opener = {"plain": open, "gzip": gzip.open, "bzip2": bz2.open}[fmt]
        try:
            with opener(path, "rb") as fh:
                data = fh.read(limit)
        except (EOFError, OSError, zlib.error) as exc:
            if fmt == "plain":
                raise
            raise CompressedFileError(path, fmt, what, f"{type(exc).__name__}: {exc}") from exc
        data = strip_bom(data)
    lines = data.decode("latin-1").splitlines()
    if len(data) >= limit and lines and not data.endswith((b"\n", b"\r")):
        lines.pop()                                  # cut by the limit
    return lines


#: Output budget of one decompression step of :func:`verify_stream` (bounds the memory used per file).
_VERIFY_OUT_LIMIT = 1 << 20


def _verify_members(fh, path, what: str, fmt: str, new_decoder, magic: bytes, chunk_size: int) -> list:
    """Stream-check the gzip members (or bzip2 streams) of ``fh``; returns the notes (stray bytes after the data).

    Every member must reach its end-of-stream marker, so a cut file raises; zlib checks each gzip member's CRC32 and
    ISIZE. Bytes after the last member that do not start another member end the data, as ``gunzip`` reads them
    (the rule of :func:`_gunzip_members`): zero padding is ignored silently, anything else gives a note."""
    decoder = new_decoder()
    members = 0
    buf = fh.read(chunk_size)
    while True:
        if decoder is None:                          # between members: another member, or stray bytes
            if not buf:
                buf = fh.read(chunk_size)
            while 0 < len(buf) < len(magic):
                more = fh.read(chunk_size)
                if not more:
                    break
                buf += more
            if not buf:
                break
            if buf[:len(magic)] == magic:
                decoder = new_decoder()
                continue
            stray, nonzero = 0, False
            while buf:
                stray += len(buf)
                nonzero = nonzero or bool(buf.strip(b"\x00"))
                buf = fh.read(chunk_size)
            if nonzero:
                return [f"{stray} byte(s) after the end of the {fmt} data ignored (as gunzip does: trailing garbage)"]
            return []
        if not buf:
            buf = fh.read(chunk_size)
            if not buf:
                break
        try:
            decoder.decompress(buf, _VERIFY_OUT_LIMIT)
            # zlib keeps the input beyond the output budget in unconsumed_tail; bz2 keeps it internally
            while not decoder.eof and (decoder.unconsumed_tail if fmt == "gzip" else not decoder.needs_input):
                decoder.decompress(decoder.unconsumed_tail if fmt == "gzip" else b"", _VERIFY_OUT_LIMIT)
        except (zlib.error, OSError, ValueError) as exc:
            raise CompressedFileError(path, fmt, what, f"corrupt data: {exc}") from exc
        if decoder.eof:
            members += 1
            buf = decoder.unused_data
            decoder = None
        else:
            buf = b""
    if decoder is not None or members == 0:
        cause = EOFError("Compressed file ended before the end-of-stream marker was reached")
        raise CompressedFileError(path, fmt, what, f"truncated: {cause}") from cause
    return []


def verify_stream(path, what: str = "file", chunk_size: int = 1 << 20) -> tuple:
    """Check that the whole of ``path`` decompresses, without keeping the data; returns ``(fmt, notes)``.

    For the downloader's integrity check: the file is read in ``chunk_size`` pieces and
    decompressed with a bounded output budget, so large files cost little memory. ``fmt`` is
    :func:`detect_format` of the first bytes; ``notes`` are readable remarks (stray bytes after the gzip data, which
    are tolerated as ``gunzip`` does).

    * gzip: every member (multi-member files included) must reach its end marker, and zlib checks each member's
      CRC32 and length. A cut member raises :class:`CompressedFileError` whose ``__cause__`` is an ``EOFError``
      ("truncated"); a CRC, length or data error raises it with the ``zlib.error`` as cause ("corrupt data").
    * bzip2: every stream must reach its end; the same two kinds of error (``OSError`` for bad data).
    * Unix compress (.Z): only the header is checked (magic and a valid flags byte: 9 to 16 bits, reserved bits
      clear). The format has no end marker and no checksum, so truncation cannot be detected here; the downloader
      relies on the HTTP Content-Length for that.
    * zip: exactly one member file, whose CRC is checked (``zipfile.testzip``).
    * plain: nothing to check (``("plain", [])``).

    A missing or unreadable file raises the usual ``OSError``."""
    with open(path, "rb") as fh:
        head = fh.read(4)
        fmt = detect_format(head)
        if fmt == "plain":
            return fmt, []
        if fmt == "compress":
            flags = head[2] if len(head) > 2 else None
            if flags is None or not 9 <= (flags & 0x1F) <= 16 or flags & 0x60:
                shown = "missing" if flags is None else f"0x{flags:02x}"
                raise CompressedFileError(path, fmt, what, f"corrupt data: invalid LZW header byte ({shown})")
            return fmt, []
        if fmt == "zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    members = [m for m in zf.infolist() if not m.is_dir()]
                    if len(members) != 1:
                        raise ValueError(f"a single member file is required, the archive holds {len(members)}")
                    bad = zf.testzip()
            except (zipfile.BadZipFile, ValueError, OSError, EOFError, zlib.error) as exc:
                raise CompressedFileError(path, fmt, what, f"corrupt data: {exc}") from exc
            if bad is not None:
                raise CompressedFileError(path, fmt, what, f"corrupt data: CRC error in member {bad!r}")
            return fmt, []
        fh.seek(0)
        if fmt == "gzip":
            notes = _verify_members(fh, path, what, fmt, lambda: zlib.decompressobj(31), b"\x1f\x8b", chunk_size)
        else:
            notes = _verify_members(fh, path, what, fmt, bz2.BZ2Decompressor, b"BZh", chunk_size)
        return fmt, notes
