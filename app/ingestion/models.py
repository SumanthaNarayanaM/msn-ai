"""Data models for the v0.2 multimodal ingestion pipeline."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class FileStatus(str, Enum):
    """Lifecycle of an uploaded file.

    Every upload must reach a terminal state (``READY`` or ``FAILED``) so the
    UI never shows an infinite spinner.
    """

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class SourceKind(str, Enum):
    """Where a piece of extracted content came from."""

    DOCUMENT = "document"
    PAGE = "page"
    SHEET = "sheet"
    SLIDE = "slide"
    SECTION = "section"
    ROW = "row"
    ARCHIVE = "archive"
    IMAGE = "image"
    OCR = "ocr"
    VIDEO = "video"
    TRANSCRIPT = "transcript"
    FRAME = "frame"
    TIMESTAMP = "timestamp"


class ExtractedBlock(BaseModel):
    """One addressable piece of content extracted from a file.

    ``locator`` keeps the human-meaningful source reference that later becomes
    a citation, e.g. ``page 12``, ``Sheet "Revenue"``, ``slide 3`` or
    ``00:20-00:45``.
    """

    text: str
    kind: SourceKind = SourceKind.DOCUMENT
    locator: str = ""
    order: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation_label(self) -> str:
        """Locator for citation display, or "" when the file name is the best
        reference we have (a generic kind name like "document" is not)."""
        return self.locator.strip()


class ExtractionResult(BaseModel):
    """Full extraction output for one file."""

    text: str = ""
    blocks: list[ExtractedBlock] = Field(default_factory=list)
    #: Free-form facts about the source (page count, sheet names, resolution...).
    document_metadata: dict[str, Any] = Field(default_factory=dict)
    #: Capabilities that were required but unavailable, e.g. "vision".
    unavailable: dict[str, str] = Field(default_factory=dict)
    #: Non-fatal problems worth surfacing to the user.
    warnings: list[str] = Field(default_factory=list)
    #: Optional generated artifact (thumbnail path, extracted frames...).
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip()) or bool(self.blocks)


def new_id(prefix: str = "") -> str:
    """Return a short, collision-resistant internal id."""
    raw = uuid.uuid4().hex[:20]
    return f"{prefix}{raw}" if prefix else raw


class FileRecord(BaseModel):
    """Persistent metadata for one uploaded file."""

    id: str = Field(default_factory=new_id)
    filename: str
    #: Sanitized, display-only name (never used as a storage path).
    display_name: str = ""
    stored_name: str = ""
    extension: str = ""
    mime_type: str = "application/octet-stream"
    size: int = 0
    sha256: str = ""
    status: FileStatus = FileStatus.UPLOADED
    source_kind: str = "document"
    is_video: bool = False
    is_image: bool = False

    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    processed_at: Optional[float] = None

    #: "session" | "project" | "personal"
    scope: str = "session"
    session_id: str = ""
    project: str = ""
    remembered: bool = False

    error: str = ""
    warnings: list[str] = Field(default_factory=list)
    progress: float = 0.0
    progress_stage: str = ""

    #: Counts and facts produced by the ingestion pipeline.
    chunk_count: int = 0
    block_count: int = 0
    char_count: int = 0
    #: Extracted-content facts (pages, sheets, duration, resolution...).
    document_metadata: dict[str, Any] = Field(default_factory=dict)
    #: Capability shortfalls, surfaced honestly in the UI.
    unavailable: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_public_dict(self) -> dict[str, Any]:
        """Serialise for the API, including derived display fields."""
        return {
            "id": self.id,
            "filename": self.filename,
            "display_name": self.display_name or self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
            "sha256": self.sha256[:16],
            "status": self.status.value,
            "source_kind": self.source_kind,
            "is_video": self.is_video,
            "is_image": self.is_image,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processed_at": self.processed_at,
            "scope": self.scope,
            "session_id": self.session_id,
            "remembered": self.remembered,
            "error": self.error,
            "warnings": self.warnings,
            "progress": self.progress,
            "progress_stage": self.progress_stage,
            "chunk_count": self.chunk_count,
            "block_count": self.block_count,
            "char_count": self.char_count,
            "document_metadata": self.document_metadata,
            "unavailable": self.unavailable,
            "artifacts": self.artifacts,
        }


def storage_path_for(base_dir: Path, file_id: str, filename: str) -> Path:
    """Resolve a safe on-disk path for a stored file.

    Storage is keyed on the generated internal id only; the original filename
    never influences the path, which makes path traversal structurally
    impossible.
    """
    safe_id = "".join(ch for ch in file_id if ch.isalnum() or ch in "-_")
    if not safe_id:
        safe_id = new_id()
    suffix = Path(filename).suffix.lower()
    suffix = "".join(ch for ch in suffix if ch.isalnum() or ch == ".")[:12]
    return base_dir / f"{safe_id}{suffix}"
