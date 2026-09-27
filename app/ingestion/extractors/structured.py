"""Structured data extraction: JSON, XML, HTML and CSV."""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.extractors.text import read_text_file
from app.ingestion.models import ExtractedBlock, ExtractionResult, SourceKind

#: Beyond this many CSV data rows we stop emitting one block per row.
MAX_CSV_ROW_BLOCKS = 400
#: Cap on XML/HTML nodes converted to text.
MAX_NODES = 20_000


def _pretty_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


class JsonExtractor(Extractor):
    """Parse JSON and preserve its key structure."""

    name = "json"
    extensions = frozenset({".json"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("parsing json", 0.35)
        max_chars = int(getattr(ctx.settings, "max_text_chars", 2_000_000))
        raw, encoding, truncated = read_text_file(path, max_chars)
        warnings: list[str] = []
        if truncated:
            warnings.append("File exceeded the character read limit; truncated.")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            warnings.append(f"JSON is malformed ({exc.msg} at line {exc.lineno}); indexed as text.")
            text = self.block_text_limit(ctx, raw.strip())
            return ExtractionResult(
                text=text,
                blocks=[ExtractedBlock(text=text, kind=SourceKind.DOCUMENT, locator="raw", order=0)],
                document_metadata={"valid_json": False, "encoding": encoding},
                warnings=warnings,
            )

        top_level = list(data.keys())[:50] if isinstance(data, dict) else f"array[{len(data)}]"
        blocks = _json_blocks(data)
        return ExtractionResult(
            text=self.block_text_limit(ctx, _pretty_json(data).strip()),
            blocks=blocks,
            document_metadata={
                "valid_json": True,
                "root_type": type(data).__name__,
                "top_level_keys": top_level,
                "encoding": encoding,
            },
            warnings=warnings,
        )


def _json_blocks(data: Any, prefix: str = "") -> list[ExtractedBlock]:
    """Flatten JSON into addressable 'path -> value' blocks."""
    blocks: list[ExtractedBlock] = []
    order = 0

    def walk(node: Any, path: str) -> None:
        nonlocal order
        if order >= 3000:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                walk(value, child)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                child = f"{path}[{index}]"
                if isinstance(value, (dict, list)):
                    walk(value, child)
                else:
                    blocks.append(
                        ExtractedBlock(
                            text=f"{child}: {value}",
                            kind=SourceKind.ROW,
                            locator=child,
                            order=order,
                        )
                    )
                    order += 1
        else:
            blocks.append(
                ExtractedBlock(
                    text=f"{path or '$'}: {node}",
                    kind=SourceKind.ROW,
                    locator=path or "$",
                    order=order,
                    metadata={"value": node if isinstance(node, (int, float, bool)) else None},
                )
            )
            order += 1

    walk(data, prefix)
    return blocks


class XmlExtractor(Extractor):
    """Parse XML into readable text while keeping element paths."""

    name = "xml"
    extensions = frozenset({".xml", ".xsd", ".rss", ".atom", ".svg"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("parsing xml", 0.35)
        max_chars = int(getattr(ctx.settings, "max_text_chars", 2_000_000))
        raw, _encoding, truncated = read_text_file(path, max_chars)
        warnings: list[str] = []
        if truncated:
            warnings.append("File exceeded the character read limit; truncated.")

        blocks: list[ExtractedBlock] = []
        root_tag = ""
        attributes: dict[str, str] = {}
        try:
            root = ET.fromstring(raw)
            root_tag = root.tag
            attributes = {k: v for k, v in list(root.attrib.items())[:20]}
            lines: list[str] = []
            count = 0
            for element in root.iter():
                count += 1
                if count > MAX_NODES:
                    warnings.append(f"Stopped after {MAX_NODES} XML elements.")
                    break
                text = (element.text or "").strip()
                label = _xml_label(element)
                if text:
                    lines.append(f"{label}: {text}")
                    blocks.append(
                        ExtractedBlock(
                            text=f"{label}: {text}",
                            kind=SourceKind.ROW,
                            locator=label,
                            order=len(blocks),
                        )
                    )
            full = "\n".join(lines)
            valid = True
        except ET.ParseError as exc:
            warnings.append(f"XML is malformed ({exc}); indexed as text.")
            full = raw.strip()
            valid = False

        return ExtractionResult(
            text=self.block_text_limit(ctx, full),
            blocks=blocks,
            document_metadata={"valid_xml": valid, "root_tag": root_tag, "attributes": attributes},
            warnings=warnings,
        )


def _xml_label(element: ET.Element) -> str:
    path: list[str] = []
    node: ET.Element | None = element
    depth = 0
    while node is not None and depth < 6:
        path.append(node.tag)
        node = node.getparent() if hasattr(node, "getparent") else None
        depth += 1
        break
    return "/".join(reversed(path)) if path else element.tag


class HtmlExtractor(Extractor):
    """Convert HTML to text using a dependency-free stripper."""

    name = "html"
    extensions = frozenset({".html", ".htm", ".xhtml"})

    _SCRIPT = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
    _BREAK = re.compile(r"</(p|div|section|article|h[1-6]|li|tr|br)\s*>", re.IGNORECASE)
    _TAG = re.compile(r"<[^>]+>")
    _WS = re.compile(r"[ \t]+")
    _BLANK = re.compile(r"\n{3,}")

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("parsing html", 0.35)
        max_chars = int(getattr(ctx.settings, "max_text_chars", 2_000_000))
        raw, _enc, truncated = read_text_file(path, max_chars)
        warnings: list[str] = []
        if truncated:
            warnings.append("File exceeded the character read limit; truncated.")

        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
        title = self._TAG.sub("", title_match.group(1)).strip() if title_match else ""

        body = self._SCRIPT.sub(" ", raw)
        body = self._BREAK.sub("\n", body)
        body = self._TAG.sub(" ", body)
        import html as html_mod

        body = html_mod.unescape(body)
        body = self._WS.sub(" ", body)
        body = "\n".join(line.strip() for line in body.splitlines())
        body = self._BLANK.sub("\n\n", body).strip()

        blocks = [
            ExtractedBlock(text=self.block_text_limit(ctx, para), kind=SourceKind.SECTION,
                           locator=f"paragraph {i + 1}", order=i)
            for i, para in enumerate(p for p in body.split("\n\n") if p.strip())
        ]
        return ExtractionResult(
            text=self.block_text_limit(ctx, body),
            blocks=blocks,
            document_metadata={"title": title, "format": "html"},
            warnings=warnings,
        )


class CsvExtractor(Extractor):
    """Extract CSV headers and rows, with one block per data row."""

    name = "csv"
    extensions = frozenset({".csv", ".tsv"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("parsing csv", 0.35)
        max_chars = int(getattr(ctx.settings, "max_text_chars", 2_000_000))
        raw, encoding, truncated = read_text_file(path, max_chars)
        warnings: list[str] = []
        if truncated:
            warnings.append("File exceeded the character read limit; truncated.")

        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
        try:
            rows = list(reader)
        except csv.Error as exc:
            warnings.append(f"CSV parse issue: {exc}")
            rows = []

        if not rows:
            return ExtractionResult(
                text="", blocks=[], document_metadata={"rows": 0, "columns": 0},
                warnings=warnings + ["The CSV file is empty."],
            )

        header = [h.strip() for h in rows[0]]
        data_rows = rows[1:]
        warnings_out = list(warnings)
        if len(header) <= 1 and data_rows:
            warnings_out.append("No header row detected; columns are positional.")

        blocks: list[ExtractedBlock] = []
        for index, row in enumerate(data_rows):
            if index >= MAX_CSV_ROW_BLOCKS:
                warnings_out.append(
                    f"Only the first {MAX_CSV_ROW_BLOCKS} of {len(data_rows)} rows "
                    "were indexed individually."
                )
                break
            pairs = []
            for col, value in zip(header, row):
                pairs.append(f"{col}: {value}")
            for col, value in zip(header[len(row):], [""] * (len(header) - len(row))):
                pairs.append(f"{col}: ")
            record = " | ".join(pairs) if pairs else " | ".join(row)
            blocks.append(
                ExtractedBlock(
                    text=record,
                    kind=SourceKind.ROW,
                    locator=f"row {index + 2}",
                    order=index,
                    metadata={"row_number": index + 2},
                )
            )

        header_line = ", ".join(header)
        preview = "\n".join(
            [header_line] + [", ".join(r) for r in data_rows[:20]]
        )
        return ExtractionResult(
            text=self.block_text_limit(ctx, preview.strip()),
            blocks=blocks,
            document_metadata={
                "columns": header,
                "column_count": len(header),
                "rows": len(data_rows),
                "encoding": encoding,
                "delimiter": delimiter,
            },
            warnings=warnings_out,
        )
