"""Office document extraction: DOCX paragraphs/tables, XLSX sheets, PPTX slides."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.models import ExtractedBlock, ExtractionResult, SourceKind

#: Cap on spreadsheet rows read per sheet.
MAX_SHEET_ROWS = 2000
#: Cap on cells emitted per row in a text rendering.
MAX_CELLS_PER_ROW = 60


class DocxExtractor(Extractor):
    """Extract paragraphs and tables from a .docx file."""

    name = "docx"
    extensions = frozenset({".docx"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("opening docx", 0.2)
        try:
            import docx  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            return ExtractionResult(
                text="",
                warnings=["DOCX support requires python-docx (pip install python-docx)."],
                unavailable={"docx": str(exc)},
            )

        warnings: list[str] = []
        blocks: list[ExtractedBlock] = []
        try:
            document = docx.Document(str(path))
        except Exception as exc:
            return ExtractionResult(
                text="",
                warnings=[f"The .docx file could not be opened: {exc}"],
                unavailable={"docx": str(exc)},
            )

        try:
            core = document.core_properties
            meta: dict = {
                "paragraphs": len(document.paragraphs),
                "tables": len(document.tables),
                "core_properties": {
                    key: str(getattr(core, key, "") or "")
                    for key in ("title", "author", "subject", "keywords", "category", "comments")
                    if getattr(core, key, None)
                },
            }
        except Exception:  # pragma: no cover - defensive
            meta = {"paragraphs": 0, "tables": 0}

        ctx.progress("reading docx paragraphs", 0.5)
        # Group paragraphs under the nearest heading so citations are useful.
        current_heading = ""
        order = 0
        for para in document.paragraphs:
            text = (para.text or "").strip()
            if not text:
                continue
            style = (getattr(para.style, "name", "") or "").lower()
            if style.startswith("heading"):
                current_heading = text
                locator = f"heading: {text[:60]}"
                kind = SourceKind.SECTION
            else:
                locator = f"under '{current_heading}'" if current_heading else f"paragraph {order + 1}"
                kind = SourceKind.DOCUMENT
            blocks.append(
                ExtractedBlock(
                    text=self.block_text_limit(ctx, text),
                    kind=kind,
                    locator=locator,
                    order=order,
                    metadata={"style": style} if style else {},
                )
            )
            order += 1

        ctx.progress("reading docx tables", 0.75)
        for t_index, table in enumerate(document.tables):
            rows: list[str] = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                rows.append(" | ".join(cells[:MAX_CELLS_PER_ROW]))
            if not rows:
                continue
            table_text = "\n".join(rows)
            blocks.append(
                ExtractedBlock(
                    text=self.block_text_limit(ctx, table_text),
                    kind=SourceKind.SECTION,
                    locator=f"table {t_index + 1}",
                    order=order,
                    metadata={"table": t_index + 1, "rows": len(rows)},
                )
            )
            order += 1

        full_text = "\n\n".join(b.text for b in blocks)
        return ExtractionResult(
            text=self.block_text_limit(ctx, full_text),
            blocks=blocks,
            document_metadata=meta,
            warnings=warnings,
        )


class XlsxExtractor(Extractor):
    """Extract sheets, headers and rows from a .xlsx workbook."""

    name = "xlsx"
    extensions = frozenset({".xlsx", ".xlsm"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("opening workbook", 0.2)
        try:
            import openpyxl  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            return ExtractionResult(
                text="",
                warnings=["XLSX support requires openpyxl (pip install openpyxl)."],
                unavailable={"xlsx": str(exc)},
            )

        try:
            workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        except Exception as exc:
            return ExtractionResult(
                text="",
                warnings=[f"The workbook could not be opened: {exc}"],
                unavailable={"xlsx": str(exc)},
            )

        blocks: list[ExtractedBlock] = []
        sheet_summaries: list[dict] = []
        warnings: list[str] = []
        order = 0
        try:
            names = list(workbook.sheetnames)
            for s_index, name in enumerate(names):
                ctx.progress(f"reading sheet {s_index + 1}/{len(names)}", 0.2 + 0.7 * (s_index / max(len(names), 1)))
                try:
                    sheet = workbook[name]
                except Exception as exc:  # pragma: no cover - defensive
                    warnings.append(f"Sheet '{name}' could not be read: {exc}")
                    continue

                rows_read = 0
                header: list[str] = []
                row_lines: list[str] = []
                try:
                    for r_index, row in enumerate(sheet.iter_rows(values_only=True)):
                        if r_index == 0:
                            header = [str(c) if c is not None else "" for c in row][:MAX_CELLS_PER_ROW]
                            continue
                        if rows_read >= MAX_SHEET_ROWS:
                            warnings.append(
                                f"Sheet '{name}' has more than {MAX_SHEET_ROWS} rows; "
                                "only the first ones were indexed."
                            )
                            break
                        cells = [("" if c is None else str(c)) for c in row][:MAX_CELLS_PER_ROW]
                        if not any(cells):
                            continue
                        pairs = [f"{h}: {v}" for h, v in zip(header, cells)] or cells
                        row_lines.append(" | ".join(pairs))
                        blocks.append(
                            ExtractedBlock(
                                text=self.block_text_limit(ctx, pairs and " | ".join(pairs) or ""),
                                kind=SourceKind.ROW,
                                locator=f'{name} · row {r_index + 1}',
                                order=order,
                                metadata={"sheet": name, "row": r_index + 1},
                            )
                        )
                        order += 1
                        rows_read += 1
                except Exception as exc:  # pragma: no cover - defensive
                    warnings.append(f"Sheet '{name}' was partially read: {exc}")

                sheet_summaries.append(
                    {"name": name, "rows_indexed": rows_read, "columns": header}
                )
                if row_lines:
                    blocks.insert(
                        max(order - rows_read, 0),
                        ExtractedBlock(
                            text=self.block_text_limit(ctx, "\n".join(row_lines[:200])),
                            kind=SourceKind.SHEET,
                            locator=f'Sheet "{name}"',
                            order=order,
                            metadata={"sheet": name, "rows": rows_read},
                        ),
                    )
                    order += 1
        finally:
            try:
                workbook.close()
            except Exception:  # pragma: no cover - defensive
                pass

        full_text = "\n\n".join(b.text for b in blocks if b.text.strip())
        return ExtractionResult(
            text=self.block_text_limit(ctx, full_text),
            blocks=blocks,
            document_metadata={"sheets": sheet_summaries, "sheet_names": [s["name"] for s in sheet_summaries]},
            warnings=warnings,
        )


class PptxExtractor(Extractor):
    """Extract slide text (and speaker notes) from a .pptx deck."""

    name = "pptx"
    extensions = frozenset({".pptx"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("opening presentation", 0.2)
        try:
            from pptx import Presentation  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            return ExtractionResult(
                text="",
                warnings=["PPTX support requires python-pptx (pip install python-pptx)."],
                unavailable={"pptx": str(exc)},
            )

        try:
            presentation = Presentation(str(path))
        except Exception as exc:
            return ExtractionResult(
                text="",
                warnings=[f"The presentation could not be opened: {exc}"],
                unavailable={"pptx": str(exc)},
            )

        blocks: list[ExtractedBlock] = []
        warnings: list[str] = []
        slides = list(presentation.slides)
        for index, slide in enumerate(slides):
            ctx.progress(f"reading slide {index + 1}/{len(slides)}", 0.2 + 0.7 * (index / max(len(slides), 1)))
            parts: list[str] = []
            for shape in slide.shapes:
                try:
                    if getattr(shape, "has_text_frame", False):
                        for paragraph in shape.text_frame.paragraphs:
                            line = "".join(run.text for run in paragraph.runs).strip()
                            if line:
                                parts.append(line)
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            cells = [c.text.strip() for c in row.cells]
                            parts.append(" | ".join(cells))
                except Exception:  # pragma: no cover - odd shapes
                    continue
            notes = ""
            try:
                if slide.has_notes_slide:
                    notes = (slide.notes_slide.notes_text_frame.text or "").strip()
            except Exception:  # pragma: no cover - defensive
                notes = ""

            if not parts and not notes:
                warnings.append(f"Slide {index + 1} contains no extractable text.")
                continue
            body = "\n".join(parts)
            if notes:
                body += ("\n\n[Speaker notes] " + notes if body else "[Speaker notes] " + notes)
            blocks.append(
                ExtractedBlock(
                    text=self.block_text_limit(ctx, body),
                    kind=SourceKind.SLIDE,
                    locator=f"slide {index + 1}",
                    order=index,
                    metadata={"slide": index + 1, "has_notes": bool(notes)},
                )
            )

        full_text = "\n\n".join(b.text for b in blocks)
        return ExtractionResult(
            text=self.block_text_limit(ctx, full_text),
            blocks=blocks,
            document_metadata={"slides": len(slides)},
            warnings=warnings,
        )
