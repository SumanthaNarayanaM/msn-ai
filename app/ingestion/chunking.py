"""Text chunking for retrieval.

Chunks are built from :class:`ExtractedBlock` objects rather than the flattened
document text, which is what lets every chunk keep its citation locator
(``page 12``, ``Sheet "Revenue" · row 7``, ``01:20-01:45``). Overlap is applied
*within* a block only, so a chunk never silently merges two unrelated pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from app.ingestion.models import ExtractedBlock, SourceKind

#: Blocks whose text is a data row are kept whole up to a larger size, because
#: splitting a spreadsheet row mid-pair makes it unanswerable.
_ROW_KINDS = {SourceKind.ROW, SourceKind.TRANSCRIPT}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PARAGRAPH = re.compile(r"\n{2,}")


@dataclass
class Chunk:
    """One retrievable unit of knowledge, with a citable source."""

    text: str
    file_id: str
    filename: str
    index: int
    kind: SourceKind = SourceKind.DOCUMENT
    locator: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def source_label(self) -> str:
        """``report.pdf · page 12`` — what the UI shows next to a citation."""
        base = self.filename or self.file_id
        return f"{base} · {self.locator}" if self.locator else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "file_id": self.file_id,
            "filename": self.filename,
            "index": self.index,
            "kind": self.kind.value,
            "locator": self.locator,
            "metadata": self.metadata,
        }


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping paragraph breaks as hard stops."""
    parts: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                parts.append(sentence)
    return parts


def _pack_units(units: list[str], target: int, overlap: int) -> Iterator[str]:
    """Greedily pack units into ``target``-sized windows with ``overlap``."""
    current: list[str] = []
    length = 0
    for unit in units:
        unit_len = len(unit) + 1
        if current and length + unit_len > target:
            yield "\n".join(current)
            if overlap > 0:
                # Carry the tail of the previous window forward.
                tail: list[str] = []
                tail_len = 0
                for previous in reversed(current):
                    if tail_len + len(previous) > overlap:
                        break
                    tail.insert(0, previous)
                    tail_len += len(previous) + 1
                current = list(tail)
                length = tail_len
            else:
                current = []
                length = 0
        current.append(unit)
        length += unit_len
    if current:
        yield "\n".join(current)


def _hard_wrap(text: str, size: int) -> list[str]:
    """Last-resort splitting for a single oversized unit (e.g. a minified JS file)."""
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    for start in range(0, len(text), size):
        parts.append(text[start : start + size])
    return parts


def chunk_blocks(
    blocks: Iterable[ExtractedBlock],
    *,
    file_id: str,
    filename: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
    max_chunks: int = 5000,
) -> list[Chunk]:
    """Turn extracted blocks into retrievable, citable chunks."""
    size = max(int(chunk_size), 200)
    overlap = max(0, min(int(chunk_overlap), size // 2))
    chunks: list[Chunk] = []
    index = 0

    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue

        if len(text) <= size:
            windows: list[str] = [text]
        elif block.kind in _ROW_KINDS:
            # Data rows and transcript lines stay atomic; each is a clean fact.
            windows = [line for line in text.splitlines() if line.strip()]
        else:
            units = split_sentences(text) or [text]
            flat: list[str] = []
            for unit in units:
                flat.extend(_hard_wrap(unit, size))
            windows = list(_pack_units(flat, size, overlap))

        for window in windows:
            window = window.strip()
            if not window:
                continue
            if index >= max_chunks:
                return chunks
            metadata = dict(block.metadata)
            metadata.setdefault("block_order", block.order)
            chunks.append(
                Chunk(
                    text=window,
                    file_id=file_id,
                    filename=filename,
                    index=index,
                    kind=block.kind,
                    locator=block.locator or block.citation_label(),
                    metadata=metadata,
                )
            )
            index += 1
    return chunks


def chunk_plain_text(
    text: str,
    *,
    file_id: str,
    filename: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
    max_chunks: int = 5000,
) -> list[Chunk]:
    """Chunk raw text that arrived without block structure."""
    return chunk_blocks(
        [ExtractedBlock(text=text, kind=SourceKind.DOCUMENT, locator="", order=0)],
        file_id=file_id,
        filename=filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
    )
