"""Plain-text and Markdown extraction."""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.models import ExtractedBlock, ExtractionResult, SourceKind

#: Try UTF-8 first, then fall back to a few common legacy encodings.
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def read_text_file(path: Path, max_chars: int) -> tuple[str, str, bool]:
    """Read a text file, returning ``(text, encoding_used, truncated)``."""
    raw = path.read_bytes()
    truncated = False
    if len(raw) > max_chars:
        # Read a bounded prefix rather than the whole file into memory.
        raw = raw[: max_chars + 1]
        truncated = True
    for encoding in _ENCODINGS:
        try:
            text = raw.decode(encoding)
            return text, encoding, truncated
        except (UnicodeDecodeError, LookupError):
            continue
    text = raw.decode("utf-8", errors="replace")
    return text, "utf-8/replace", truncated


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into ``(heading, body)`` sections using ATX headings."""
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if match:
            if heading is not None or buffer:
                sections.append((heading or "(preamble)", "\n".join(buffer).strip()))
            heading = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    if heading is not None or buffer:
        sections.append((heading or "(preamble)", "\n".join(buffer).strip()))
    return [s for s in sections if s[1]]


class TextExtractor(Extractor):
    """Read ``.txt`` / ``.md`` / ``.log`` style files, preserving structure."""

    name = "text"
    extensions = frozenset({".txt", ".md", ".markdown", ".log", ".rst"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("reading text", 0.3)
        max_chars = int(getattr(ctx.settings, "max_text_chars", 2_000_000))
        text, encoding, truncated = read_text_file(path, max_chars)
        ext = path.suffix.lower()

        warnings: list[str] = []
        if truncated:
            warnings.append(
                f"File was larger than the {max_chars} character read limit; "
                "only the beginning was indexed."
            )
        if encoding.endswith("replace"):
            warnings.append("Some bytes could not be decoded and were replaced.")

        blocks: list[ExtractedBlock] = []
        if ext in (".md", ".markdown"):
            for order, (heading, body) in enumerate(split_markdown_sections(text)):
                blocks.append(
                    ExtractedBlock(
                        text=self.block_text_limit(ctx, body),
                        kind=SourceKind.SECTION,
                        locator=heading,
                        order=order,
                        metadata={"heading": heading},
                    )
                )
        else:
            # Group plain text into readable paragraphs-sized blocks.
            for order, para in enumerate(_paragraphs(text)):
                blocks.append(
                    ExtractedBlock(
                        text=self.block_text_limit(ctx, para),
                        kind=SourceKind.DOCUMENT,
                        locator=f"lines {para_index_span(text, para)}",
                        order=order,
                    )
                )

        return ExtractionResult(
            text=self.block_text_limit(ctx, text.strip()),
            blocks=blocks,
            document_metadata={"encoding": encoding, "truncated": truncated, "format": "markdown" if ext in (".md", ".markdown") else "text"},
            warnings=warnings,
        )


def _paragraphs(text: str, min_len: int = 1) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if len(p) >= min_len]


def para_index_span(full_text: str, para: str) -> str:
    """Best-effort line span for a paragraph, used in citations."""
    idx = full_text.find(para)
    if idx < 0:
        return "?"
    start_line = full_text.count("\n", 0, idx) + 1
    end_line = start_line + para.count("\n")
    return f"{start_line}-{end_line}"
