"""Extractor registry: maps a file extension to the class that can read it.

Kept deliberately data-driven so ``/health`` and the frontend can report exactly
which formats are supported (and which of them are degraded on this machine).
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger
from app.ingestion.extractors.archive import ZipExtractor
from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.extractors.image import ImageExtractor
from app.ingestion.extractors.office import DocxExtractor, PptxExtractor, XlsxExtractor
from app.ingestion.extractors.pdf import PdfExtractor
from app.ingestion.extractors.structured import (
    CsvExtractor,
    HtmlExtractor,
    JsonExtractor,
    XmlExtractor,
)
from app.ingestion.extractors.text import TextExtractor
from app.ingestion.extractors.video import VideoExtractor
from app.ingestion.models import ExtractionResult

logger = get_logger("msn_ai.ingestion.registry")

#: Ordered by specificity; the first extractor claiming an extension wins.
EXTRACTOR_CLASSES: tuple[type[Extractor], ...] = (
    PdfExtractor,
    DocxExtractor,
    XlsxExtractor,
    PptxExtractor,
    ImageExtractor,
    VideoExtractor,
    ZipExtractor,
    JsonExtractor,
    XmlExtractor,
    HtmlExtractor,
    CsvExtractor,
    TextExtractor,
)


class ExtractorRegistry:
    """Resolve extensions to extractor instances, with no per-request cost."""

    def __init__(self, extractor_classes: tuple[type[Extractor], ...] = EXTRACTOR_CLASSES) -> None:
        self._by_extension: dict[str, Extractor] = {}
        for cls in extractor_classes:
            try:
                instance = cls()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Extractor %s failed to construct: %s", cls.name, exc)
                continue
            for ext in instance.extensions:
                # First registration wins, keeping the ordering meaningful.
                self._by_extension.setdefault(ext.lower(), instance)

    def resolve(self, filename_or_ext: str) -> Extractor | None:
        """Return the extractor for a filename or bare extension, else None."""
        candidate = str(filename_or_ext).strip().lower()
        if not candidate:
            return None
        ext = candidate if candidate.startswith(".") else Path(candidate).suffix.lower()
        if not ext:
            return None
        return self._by_extension.get(ext)

    def supports(self, filename_or_ext: str) -> bool:
        return self.resolve(filename_or_ext) is not None

    def supported_extensions(self) -> list[str]:
        return sorted(self._by_extension)

    def supported_formats(self) -> list[dict[str, str]]:
        """Format list for the API/health payload."""
        seen: dict[str, str] = {}
        for ext, extractor in self._by_extension.items():
            seen.setdefault(extractor.name, ext)
        return [
            {"extractor": name, "example": example}
            for name, example in sorted(seen.items())
        ]

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        """Dispatch to the right extractor, failing loudly for unknown types."""
        extractor = self.resolve(path.name)
        if extractor is None:
            return ExtractionResult(
                text="",
                warnings=[
                    f"No extractor is registered for '{Path(path.name).suffix or path.name}'."
                ],
                unavailable={"extractor": "unsupported extension"},
            )
        logger.debug("Extracting %s with '%s'", path.name, extractor.name)
        return extractor.extract(path, ctx)


#: Shared registry instance; extractors are stateless so one is enough.
registry = ExtractorRegistry()
