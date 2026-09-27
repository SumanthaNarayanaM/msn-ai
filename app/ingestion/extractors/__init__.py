"""Format-specific extractors for MSN AI v0.2 ingestion."""

from __future__ import annotations

from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.extractors.archive import ZipExtractor
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

__all__ = [
    "Extractor",
    "ExtractionContext",
    "TextExtractor",
    "JsonExtractor",
    "XmlExtractor",
    "HtmlExtractor",
    "CsvExtractor",
    "PdfExtractor",
    "DocxExtractor",
    "XlsxExtractor",
    "PptxExtractor",
    "ImageExtractor",
    "VideoExtractor",
    "ZipExtractor",
]
