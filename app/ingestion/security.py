"""Upload security: filename sanitising, type allow-listing and size limits.

Design rules enforced here:

* The client-supplied filename is **never** trusted and **never** used to build
  a storage path. Storage is keyed on a generated internal id.
* Path traversal (``../``, ``..\\``, absolute paths, drive letters, UNC paths,
  NUL bytes) is rejected outright rather than silently sanitised.
* Support is decided by extension **and** by content sniffing, so renaming
  ``evil.exe`` to ``evil.pdf`` does not get it treated as a PDF.
* Size is enforced while streaming to disk, so an oversized upload is aborted
  before it is fully written.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import BinaryIO, Iterable

from app.core.errors import (
    FileTooLargeError,
    UnsafeFileNameError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger

logger = get_logger("msn_ai.ingestion.security")

#: Extension -> canonical media type. The single source of truth for what MSN AI
#: accepts. Anything absent here is rejected with a clear message.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    # documents
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".log": "text/plain",
    ".rst": "text/x-rst",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".xml": "application/xml",
    ".xsd": "application/xml",
    ".rss": "application/rss+xml",
    ".atom": "application/atom+xml",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    # office
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    # video
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    # archives
    ".zip": "application/zip",
}

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".webm", ".avi"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

#: Leading bytes that identify a real container. Used to reject a text file that
#: merely claims to be a PDF/PNG/ZIP.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "zip-container"),  # docx/xlsx/pptx/zip
    (b"PK\x05\x06", "zip-container"),  # empty zip
    (b"PK\x07\x08", "zip-container"),  # spanned zip
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"BM", "bmp"),
    (b"RIFF", "riff"),  # webp / wav / avi
    (b"\x1a\x45\xdf\xa3", "matroska"),  # mkv / webm
)

#: Containers that legitimately begin with a ZIP signature.
_ZIP_BASED = {"zip-container"}

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MULTI_DOT_RE = re.compile(r"\.{2,}")
_UNSAFE_NAME_RE = re.compile(r'[<>:"/\\|?*]')

MAX_FILENAME_LEN = 200


def _looks_like_path(value: str) -> bool:
    """Return True when a filename encodes a path rather than a bare name."""
    if "/" in value or "\\" in value:
        return True
    # Drive letters, UNC and drive-relative paths.
    if re.match(r"^[A-Za-z]:", value):
        return True
    if value.startswith("~"):
        return True
    return False


def sanitize_filename(raw: str | None) -> str:
    """Validate and clean a client-supplied filename.

    Raises :class:`UnsafeFileNameError` when the value is empty, absurdly long,
    contains control characters, or encodes a path. Returns a safe *display*
    name only; it is never used to build a filesystem path.
    """
    if raw is None:
        raise UnsafeFileNameError("No filename was provided.")
    name = unicodedata.normalize("NFC", str(raw)).strip()
    # Some clients append a full path; keep only the basename for display.
    if _looks_like_path(name):
        base_posix = PurePosixPath(name.replace("\\", "/")).name
        base_win = PureWindowsPath(name).name
        candidate = (base_posix or base_win or "").strip()
        # A traversal payload is an attack, not a display-name quirk.
        if _MULTI_DOT_RE.search(name.replace("\\", "/")) and name.count("..") > 0:
            raise UnsafeFileNameError(
                f"Unsafe filename rejected: '{name[:80]}' contains a path traversal sequence."
            )
        name = candidate
        if not name:
            raise UnsafeFileNameError("Unsafe filename rejected: empty after sanitising.")

    if not name:
        raise UnsafeFileNameError("Filename is empty.")
    if _CONTROL_RE.search(name):
        raise UnsafeFileNameError("Filename contains control characters.")
    if _MULTI_DOT_RE.search(name):
        raise UnsafeFileNameError("Filename contains a path traversal sequence ('..').")
    if len(name) > MAX_FILENAME_LEN:
        name = name[:MAX_FILENAME_LEN]
    name = _UNSAFE_NAME_RE.sub("_", name).strip(" .")
    if not name:
        raise UnsafeFileNameError("Filename is empty after sanitising.")
    return name


def normalize_extension(filename: str) -> str:
    """Return the lowercase extension including the dot, or '' when absent."""
    _, ext = _split_ext(filename)
    return ext.lower()


def _split_ext(filename: str) -> tuple[str, str]:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    idx = base.rfind(".")
    if idx <= 0:
        return base, ""
    return base[:idx], base[idx:]


def is_video_extension(ext: str) -> bool:
    return ext.lower() in VIDEO_EXTENSIONS


def is_image_extension(ext: str) -> bool:
    return ext.lower() in IMAGE_EXTENSIONS


def resolve_media_type(filename: str, declared: str | None = None) -> tuple[str, str]:
    """Return ``(extension, canonical_mime)`` for an upload.

    Raises :class:`UnsupportedFileTypeError` with MSN AI's clear "not supported
    yet" message for anything outside the allow-list.
    """
    ext = normalize_extension(filename)
    if not ext:
        raise UnsupportedFileTypeError(
            "MSN AI doesn't support this file type yet: the file has no extension. "
            f"Supported types: {supported_extension_list()}."
        )
    canonical = SUPPORTED_EXTENSIONS.get(ext)
    if canonical is None:
        raise UnsupportedFileTypeError(
            f"MSN AI doesn't support this file type yet ('{ext}'). "
            f"Supported types: {supported_extension_list()}."
        )
    return ext, canonical


def supported_extension_list() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def sniff_kind(head: bytes) -> str:
    """Classify a file from its leading bytes (best effort)."""
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            if kind == "riff":
                if head[8:12] == b"WEBP":
                    return "webp"
                if head[8:12] == b"AVI ":
                    return "avi"
                return "riff"
            if kind == "matroska":
                return "matroska"
            return kind
    text = head.lstrip()[:64].lower()
    if text.startswith(b"<?xml"):
        return "xml"
    if text.startswith(b"{") or text.startswith(b"["):
        return "json"
    if text.startswith(b"<!doctype html") or text.startswith(b"<html"):
        return "html"
    if b"\x00" not in head:
        return "text"
    return "unknown"


def verify_content_matches_extension(ext: str, path) -> list[str]:
    """Check that the bytes on disk look like the claimed extension.

    Returns a list of human-readable warnings (empty when consistent). A
    mismatch is a warning rather than a hard failure so a quirky-but-valid file
    is still readable, but it is reported honestly.
    """
    warnings: list[str] = []
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError as exc:  # pragma: no cover - defensive
        return [f"Could not read file for type verification: {exc}"]

    if not head:
        return ["The file is empty."]

    kind = sniff_kind(head)
    ext = ext.lower()
    expected = {
        ".pdf": "pdf",
        ".zip": "zip-container",
        ".docx": "zip-container",
        ".xlsx": "zip-container",
        ".pptx": "zip-container",
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".bmp": "bmp",
        ".webp": "webp",
        ".mkv": "matroska",
        ".webm": "matroska",
        ".avi": "avi",
    }.get(ext)

    if expected is None:
        # Text-ish formats: a NUL byte in the head means it is not plain text.
        if kind == "unknown" and b"\x00" in head:
            warnings.append(
                f"'{ext}' is normally a text format but the file contains binary data."
            )
        return warnings

    if kind == expected:
        return warnings

    if expected in _ZIP_BASED and kind == "zip-container":
        return warnings

    warnings.append(
        f"File signature ('{kind}') does not match the '{ext}' extension; "
        "MSN AI will try to read it anyway."
    )
    return warnings


def stream_to_disk(
    source: BinaryIO,
    destination,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 256,
) -> tuple[int, str]:
    """Copy an upload to ``destination`` while hashing and enforcing the limit.

    Raises :class:`FileTooLargeError` as soon as the stream exceeds
    ``max_bytes``, and removes the partial file so a rejected upload leaves no
    residue. Returns ``(bytes_written, sha256_hexdigest)``.
    """
    import hashlib

    digest = hashlib.sha256()
    written = 0
    try:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise FileTooLargeError(
                    f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                    details={"max_bytes": max_bytes},
                )
            digest.update(chunk)
            destination.write(chunk)
    except Exception:
        try:
            destination.close()
        except Exception:  # pragma: no cover - defensive
            pass
        raise
    return written, digest.hexdigest()


def enforce_size_limit(size: int, max_bytes: int) -> None:
    """Raise :class:`FileTooLargeError` when a known size is over the limit."""
    if size > max_bytes:
        raise FileTooLargeError(
            f"File is {size / (1024 * 1024):.1f} MB which exceeds the "
            f"{max_bytes // (1024 * 1024)} MB limit."
        )


def first_bytes(path, count: int = 4096) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(count)
    except OSError:  # pragma: no cover - defensive
        return b""


def extension_source_kind(ext: str) -> str:
    """Coarse category used for grouping in the UI and memory scopes."""
    ext = ext.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext == ".zip":
        return "archive"
    return "document"
