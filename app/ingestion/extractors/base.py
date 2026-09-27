"""Extractor interface shared by every content reader."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from app.ingestion.models import ExtractionResult, FileRecord

#: Called with (stage_name, fraction 0..1) while a long job runs.
ProgressFn = Callable[[str, float], None]


@dataclass
class ExtractionContext:
    """Services an extractor may use.

    Deliberately narrow: extractors receive *capability providers* (OCR, vision,
    speech) rather than importing them, so a missing optional dependency turns
    into an honest "unavailable" note instead of a crash.
    """

    settings: Any
    file_record: FileRecord
    progress: ProgressFn = lambda stage, fraction: None
    ocr: Any = None
    vision: Any = None
    speech: Any = None
    ffmpeg: Any = None
    notes: list[str] = field(default_factory=list)


class Extractor(ABC):
    """Read one family of file formats into :class:`ExtractionResult`."""

    #: Machine-readable extractor id, surfaced in API metadata.
    name: str = "base"
    #: Extensions this extractor claims (lowercase, with dot).
    extensions: frozenset[str] = frozenset()

    def supports(self, ext: str) -> bool:
        return ext.lower() in self.extensions

    @abstractmethod
    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        """Extract text/blocks/metadata from ``path``.

        Implementations must not raise for merely-empty content; they raise only
        for genuinely unreadable files, and should populate
        ``result.warnings`` / ``result.unavailable`` for degraded situations.
        """

    # --- shared helpers ---------------------------------------------------

    @staticmethod
    def unavailable_note(ctx: ExtractionContext, capability: str, reason: str) -> None:
        """Record an honest capability gap for the UI and API."""
        ctx.notes.append(f"{capability}: {reason}")

    @staticmethod
    def block_text_limit(ctx: ExtractionContext, text: str) -> str:
        limit = int(getattr(ctx.settings, "max_block_chars", 200_000))
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n[... truncated at {limit} characters ...]"
