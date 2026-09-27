"""PDF extraction backed by PyMuPDF, preserving page numbers for citations."""

from __future__ import annotations

from pathlib import Path

from app.core.errors import ExtractionError
from app.core.logging import get_logger
from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.models import ExtractedBlock, ExtractionResult, SourceKind

logger = get_logger("msn_ai.ingestion.pdf")

#: Pages with fewer than this many characters are treated as scanned/image-only.
MIN_CHARS_FOR_TEXT_PAGE = 24
#: Cap on pages rendered for OCR fallback.
MAX_OCR_PAGES = 8


class PdfExtractor(Extractor):
    """Extract text (and optionally OCR) page by page.

    Citations keep the real page number, e.g. ``page 12``.
    """

    name = "pdf"
    extensions = frozenset({".pdf"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("opening pdf", 0.1)
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            return ExtractionResult(
                text="",
                warnings=["PDF support requires PyMuPDF (pip install PyMuPDF)."],
                unavailable={"pdf": str(exc)},
            )

        warnings: list[str] = []
        unavailable: dict[str, str] = {}
        blocks: list[ExtractedBlock] = []
        pages_text: list[str] = []

        doc = None
        try:
            try:
                doc = fitz.open(str(path))
            except Exception as exc:
                # A file that claims to be a PDF but cannot be opened at all is
                # genuinely unreadable, so fail loudly instead of reporting a
                # successful extraction of nothing.
                raise ExtractionError(
                    f"This PDF could not be opened: {exc}. It may be corrupt or "
                    "incompletely uploaded."
                ) from exc
            if doc.needs_pass:
                return ExtractionResult(
                    text="",
                    document_metadata={"encrypted": True, "pages": doc.page_count},
                    warnings=["This PDF is password protected; MSN AI cannot read it."],
                    unavailable={"pdf": "password protected"},
                )

            page_count = doc.page_count
            meta: dict = {
                "pages": page_count,
                "encrypted": False,
                "pdf_metadata": {},
                "scanned_pages": 0,
                "ocr_pages": 0,
            }
            try:
                raw_meta = doc.metadata or {}
                meta["pdf_metadata"] = {
                    k: v
                    for k, v in raw_meta.items()
                    if v and k in ("title", "author", "subject", "keywords", "creator", "producer")
                }
            except Exception:  # pragma: no cover - defensive
                pass

            total = max(page_count, 1)
            ocr_attempted = 0
            for index in range(page_count):
                ctx.progress(f"reading page {index + 1}/{page_count}", 0.1 + 0.7 * (index / total))
                try:
                    page = doc.load_page(index)
                    text = (page.get_text("text") or "").strip()
                except Exception as exc:  # pragma: no cover - malformed page
                    warnings.append(f"Page {index + 1} could not be read: {exc}")
                    continue

                if len(text) < MIN_CHARS_FOR_TEXT_PAGE:
                    meta["scanned_pages"] += 1
                    ocr_text = self._ocr_page(ctx, page, index + 1, ocr_attempted, warnings)
                    if ocr_attempted < MAX_OCR_PAGES:
                        ocr_attempted += 1
                    if ocr_text:
                        meta["ocr_pages"] += 1
                        text = ocr_text
                    else:
                        if "ocr" not in unavailable and ctx.ocr is not None and not ctx.ocr.available:
                            unavailable["ocr"] = ctx.ocr.unavailable_reason
                        if not text:
                            text = ""

                if text:
                    pages_text.append(text)
                    blocks.append(
                        ExtractedBlock(
                            text=self.block_text_limit(ctx, text),
                            kind=SourceKind.PAGE,
                            locator=f"page {index + 1}",
                            order=index,
                            metadata={"page": index + 1},
                        )
                    )

            if page_count == 0:
                warnings.append("The PDF has no pages.")

            full_text = "\n\n".join(pages_text)
            return ExtractionResult(
                text=self.block_text_limit(ctx, full_text),
                blocks=blocks,
                document_metadata=meta,
                unavailable=unavailable,
                warnings=warnings,
            )
        except ExtractionError:
            raise
        except Exception as exc:
            logger.warning("PDF extraction failed for %s: %s", path.name, exc)
            return ExtractionResult(
                text="",
                document_metadata={"pages": 0},
                warnings=[f"The PDF could not be parsed: {exc}"],
                unavailable={"pdf": str(exc)},
            )
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:  # pragma: no cover - defensive
                    pass

    def _ocr_page(
        self,
        ctx: ExtractionContext,
        page,
        page_number: int,
        attempt_index: int,
        warnings: list[str],
    ) -> str:
        """Render a page and run local OCR, when an OCR provider is available."""
        if ctx.ocr is None or not ctx.ocr.available or attempt_index >= MAX_OCR_PAGES:
            return ""
        try:
            import fitz  # type: ignore

            zoom = 2.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            result = ctx.ocr.image_to_text(pix.tobytes("png"), source=f"{ctx.file_record.filename} page {page_number}")
            return (result.text or "").strip()
        except Exception as exc:  # pragma: no cover - optional path
            warnings.append(f"OCR failed on page {page_number}: {exc}")
            return ""
