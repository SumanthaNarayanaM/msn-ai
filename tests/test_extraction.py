"""Extraction quality per format, with honest degradation for the rest.

These tests build real files (real PDF via PyMuPDF, real DOCX/XLSX/PPTX via
their libraries, a real ZIP, a hand-built ISO-BMFF MP4) so the extractors are
exercised end to end rather than mocked. Video tests assert that genuine
container metadata is extracted and that a missing ffmpeg/vision/speech model
is reported rather than faked.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from app.ingestion.models import FileRecord, FileStatus, SourceKind
from conftest import SAMPLE_CSV, SAMPLE_JSON, SAMPLE_NOTE


def make_record(filename: str, file_id: str = "f1") -> FileRecord:
    return FileRecord(id=file_id, filename=filename, display_name=filename)


def ingest(service, payload: bytes | Path, filename: str, session_id: str = "s1"):
    """Store a payload and run extraction synchronously."""
    if isinstance(payload, Path):
        payload = payload.read_bytes()
    record = service.save_upload(io.BytesIO(payload), filename, session_id=session_id)
    return service.process(record.id)


# --- real-file builders --------------------------------------------------


def build_pdf(path: Path, pages: list[str]) -> Path:
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 100), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def build_docx(path: Path) -> Path:
    import docx

    document = docx.Document()
    document.add_heading("Architecture", level=1)
    document.add_paragraph("The backend exposes a FastAPI application.")
    document.add_heading("Testing", level=1)
    document.add_paragraph("Pytest covers the ingestion pipeline.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Layer"
    table.cell(0, 1).text = "Count"
    table.cell(1, 0).text = "Extractors"
    table.cell(1, 1).text = "11"
    document.save(str(path))
    return path


def build_xlsx(path: Path) -> Path:
    import openpyxl

    book = openpyxl.Workbook()
    revenue = book.active
    revenue.title = "Revenue"
    revenue.append(["quarter", "amount"])
    revenue.append(["Q1", 1200])
    revenue.append(["Q2", 1500])
    costs = book.create_sheet("Costs")
    costs.append(["quarter", "amount"])
    costs.append(["Q1", 900])
    book.save(str(path))
    return path


def build_pptx(path: Path) -> Path:
    from pptx import Presentation

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[0])
    slide.shapes.title.text = "Quarterly Review"
    slide.placeholders[1].text = "Revenue grew to 4.2M"
    second = deck.slides.add_slide(deck.slide_layouts[1])
    second.shapes.title.text = "Risks"
    second.placeholders[1].text = "Vendor contract expires in December"
    deck.save(str(path))
    return path


def build_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("myproject/README.md", "# My Project\nA demo project.\n")
        archive.writestr("myproject/src/main.py", "def main():\n    return 'hi'\n")
        archive.writestr("myproject/data/values.json", '{"a": 1}')
    return path


def build_zip_with_traversal(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../../escape.txt", "should never be extracted")
        archive.writestr("safe.txt", "fine")
    return path


def build_png(path: Path, width: int = 64, height: int = 32) -> Path:
    from PIL import Image

    Image.new("RGB", (width, height), (200, 30, 30)).save(str(path))
    return path


# --- minimal ISO-BMFF writer ---------------------------------------------
# Written by hand so the MP4 container parser is exercised against a real
# ftyp/moov/trak layout without requiring a video encoder.


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def build_mp4(path: Path, width: int = 320, height: int = 240, seconds: float = 2.0) -> Path:
    timescale = 15
    duration = int(timescale * seconds)

    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")

    mvhd = _box(
        b"mvhd",
        struct.pack(">I", 0)
        + struct.pack(">II", 0, 0)
        + struct.pack(">I", timescale)
        + struct.pack(">I", duration)
        + struct.pack(">I", 0x00010000)
        + struct.pack(">H", 0x0100)
        + b"\x00" * 10
        + struct.pack(">9i", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
        + b"\x00" * 24
        + struct.pack(">I", 2),
    )

    tkhd = _box(
        b"tkhd",
        struct.pack(">I", 0x00000007)
        + struct.pack(">II", 0, 0)
        + struct.pack(">I", 1)
        + struct.pack(">I", 0)
        + struct.pack(">I", duration)
        + b"\x00" * 8
        + struct.pack(">H", 0)
        + struct.pack(">H", 0)
        + struct.pack(">H", 0)
        + struct.pack(">H", 0)
        + struct.pack(">9i", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
        + struct.pack(">II", width << 16, height << 16),
    )

    mdhd = _box(
        b"mdhd",
        struct.pack(">I", 0)
        + struct.pack(">II", 0, 0)
        + struct.pack(">I", timescale)
        + struct.pack(">I", duration)
        + struct.pack(">H", 0x55C4)
        + struct.pack(">H", 0),
    )

    hdlr = _box(
        b"hdlr",
        struct.pack(">I", 0) + b"\x00" * 4 + b"vide" + b"\x00" * 12 + b"VideoHandler\x00",
    )

    stsd = _box(b"stsd", struct.pack(">I", 0) + struct.pack(">I", 1) + b"\x00" * 78)
    stbl = _box(b"stbl", stsd)
    minf = _box(b"minf", stbl)
    mdia = _box(b"mdia", mdhd + hdlr + minf)
    trak = _box(b"trak", tkhd + mdia)
    moov = _box(b"moov", mvhd + trak)

    path.write_bytes(ftyp + moov)
    return path


# --- text / structured ---------------------------------------------------


def test_text_extraction_preserves_headings(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    assert record.status is FileStatus.READY
    assert record.char_count > 0
    assert record.chunk_count >= 1
    extracted = service.store.extracted_path(record).read_text(encoding="utf-8")
    assert "Berlin" in extracted and "vendor contract" in extracted


def test_json_extraction_produces_key_path_blocks(service) -> None:
    record = ingest(service, SAMPLE_JSON, "config.json")
    assert record.status is FileStatus.READY
    assert record.document_metadata["valid_json"] is True
    assert "features" in record.document_metadata["top_level_keys"]
    text = service.store.extracted_path(record).read_text(encoding="utf-8")
    assert "max_upload_mb" in text
    # Key paths become addressable retrieval targets.
    hits = service.search("limits max_upload_mb", session_id="s1", top_k=10)
    assert any("max_upload_mb" in h.chunk.locator for h in hits)


def test_malformed_json_degrades_with_a_warning(service) -> None:
    record = ingest(service, b'{"a": 1,,,}', "broken.json")
    assert record.status is FileStatus.READY
    assert record.document_metadata["valid_json"] is False
    assert any("malformed" in w.lower() for w in record.warnings)


def test_csv_extraction_uses_headers_for_each_row(service) -> None:
    record = ingest(service, SAMPLE_CSV, "team.csv")
    assert record.status is FileStatus.READY
    meta = record.document_metadata
    assert meta["columns"] == ["name", "role", "team"]
    assert meta["rows"] == 3
    # Rows are indexed as "header: value" facts, so a question like "who is on
    # the platform team" is answerable without the column positions.
    hits = service.search("Ada engineer platform", session_id="s1", top_k=10)
    assert hits
    assert "name: Ada" in hits[0].chunk.text
    assert "team: platform" in hits[0].chunk.text


# --- office --------------------------------------------------------------


def test_docx_extraction_keeps_headings_and_tables(service, tmp_path) -> None:
    record = ingest(service, build_docx(tmp_path / "spec.docx"), "spec.docx")
    assert record.status is FileStatus.READY
    text = service.store.extracted_path(record).read_text(encoding="utf-8")
    assert "Architecture" in text
    assert "FastAPI" in text
    assert "Extractors" in text
    assert record.document_metadata["tables"] >= 1


def test_xlsx_extraction_cites_sheet_and_row(service, tmp_path) -> None:
    record = ingest(service, build_xlsx(tmp_path / "book.xlsx"), "book.xlsx")
    assert record.status is FileStatus.READY
    assert record.document_metadata["sheet_names"] == ["Revenue", "Costs"]
    hits = service.search("Q2 amount", session_id="s1", top_k=10)
    assert hits, "spreadsheet rows should be retrievable"
    locators = " ".join(h.chunk.locator for h in hits)
    assert "Revenue" in locators or "Costs" in locators


def test_pptx_extraction_cites_slides(service, tmp_path) -> None:
    record = ingest(service, build_pptx(tmp_path / "deck.pptx"), "deck.pptx")
    assert record.status is FileStatus.READY
    assert record.document_metadata["slides"] == 2
    hits = service.search("vendor contract expires", session_id="s1")
    assert any("slide" in h.chunk.locator for h in hits)


# --- pdf -----------------------------------------------------------------


def test_pdf_extraction_cites_pages(service, tmp_path) -> None:
    path = build_pdf(tmp_path / "doc.pdf", ["Page one about revenue.", "Page two about risks."])
    record = ingest(service, path, "doc.pdf")
    assert record.status is FileStatus.READY
    assert record.document_metadata["pages"] == 2
    hits = service.search("risks", session_id="s1")
    assert any("page" in h.chunk.locator for h in hits)


def test_corrupt_pdf_fails_with_an_honest_message(service, tmp_path) -> None:
    record = ingest(service, b"%PDF-1.7\nthis is not a real pdf body", "broken.pdf")
    assert record.status is FileStatus.FAILED
    assert record.error


# --- images --------------------------------------------------------------


def test_image_metadata_and_thumbnail_are_real(service, tmp_path) -> None:
    record = ingest(service, build_png(tmp_path / "pic.png", 120, 60), "pic.png")
    assert record.status is FileStatus.READY
    assert record.is_image is True
    assert record.document_metadata["width"] == 120
    assert record.document_metadata["height"] == 60
    thumb = service.settings.thumbnails_path() / str(record.artifacts["thumbnail"])
    assert thumb.exists() and thumb.stat().st_size > 0


def test_image_without_vision_model_says_so_and_never_invents(service, tmp_path) -> None:
    """A text-only chat model must not pretend it saw the picture."""
    record = ingest(service, build_png(tmp_path / "cat.png"), "cat.png")
    assert record.status is FileStatus.READY
    assert "vision" in record.unavailable
    assert "installed" in record.unavailable["vision"].lower()
    text = service.store.extracted_path(record).read_text(encoding="utf-8").lower()
    # It may state real image facts, but must not assert scene content.
    assert "cat" not in text
    assert any("vision" in w.lower() or "cannot visually" in w.lower() for w in record.warnings)


def test_ocr_absence_is_reported_honestly(service, tmp_path) -> None:
    record = ingest(service, build_png(tmp_path / "scan.png"), "scan.png")
    if not service.capabilities()["ocr"]["available"]:
        assert "ocr" in record.unavailable


# --- archives ------------------------------------------------------------


def test_zip_extraction_explains_project_structure(service, tmp_path) -> None:
    record = ingest(service, build_zip(tmp_path / "project.zip"), "project.zip")
    assert record.status is FileStatus.READY
    meta = record.document_metadata
    assert meta["file_count"] == 3
    assert any("src" in folder for folder in meta["folders"])
    text = service.store.extracted_path(record).read_text(encoding="utf-8")
    assert "My Project" in text
    assert "def main" in text


def test_zip_traversal_entries_are_skipped_not_extracted(service, tmp_path) -> None:
    record = ingest(service, build_zip_with_traversal(tmp_path / "evil.zip"), "evil.zip")
    assert record.document_metadata["skipped_unsafe_entries"] == 1
    assert any("traversal" in w.lower() or "absolute" in w.lower() for w in record.warnings)
    # Nothing was written outside the ingestion tree.
    assert not (tmp_path / "escape.txt").exists()


# --- video ---------------------------------------------------------------


def test_video_metadata_is_parsed_without_ffmpeg(service, tmp_path) -> None:
    """Container metadata must be genuine even with no ffmpeg installed."""
    record = ingest(service, build_mp4(tmp_path / "clip.mp4"), "clip.mp4")
    assert record.status is FileStatus.READY, record.error
    assert record.is_video is True
    meta = record.document_metadata
    assert meta["width"] == 320 and meta["height"] == 240
    assert meta["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert meta["duration_human"] == "00:02"
    assert meta["video_codec"]
    # Missing capabilities are declared, never simulated.
    assert "video_frames" in record.unavailable
    assert "not installed" in record.unavailable["video_frames"].lower()


def test_video_never_claims_a_transcript_without_speech_model(service, tmp_path) -> None:
    record = ingest(service, build_mp4(tmp_path / "silent.mp4"), "silent.mp4")
    assert record.status is FileStatus.READY, record.error
    hits = service.search("transcript speech words", session_id="s1")
    assert not any(h.chunk.kind is SourceKind.TRANSCRIPT for h in hits)
    assert "transcript_segments" not in record.document_metadata


def test_unparseable_video_reports_an_honest_failure(service) -> None:
    record = ingest(service, b"\x00\x00\x00\x18ftypmp42rubbish", "junk.mp4")
    assert record.status is FileStatus.FAILED
    assert "could not be parsed" in record.error
    assert record.chunk_count == 0
