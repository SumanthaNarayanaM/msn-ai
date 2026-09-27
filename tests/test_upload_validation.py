"""Upload validation: filename safety, type allow-listing, size limits."""

from __future__ import annotations

import io

import pytest

from app.core.errors import (
    FileTooLargeError,
    UnsafeFileNameError,
    UnsupportedFileTypeError,
)
from app.ingestion.security import (
    normalize_extension,
    resolve_media_type,
    sanitize_filename,
    sniff_kind,
    verify_content_matches_extension,
)
from conftest import SAMPLE_NOTE


# --- filename safety -----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",
        "C:\\Users\\victim\\secret.txt",
        "\\\\server\\share\\file.txt",
        "~/notes.txt",
    ],
)
def test_path_like_filenames_are_reduced_or_rejected(raw: str) -> None:
    """A path is either stripped to a bare basename or rejected outright."""
    try:
        cleaned = sanitize_filename(raw)
    except UnsafeFileNameError:
        return  # rejection is the strongest acceptable outcome
    assert "/" not in cleaned and "\\" not in cleaned
    assert ".." not in cleaned


def test_traversal_payload_is_rejected() -> None:
    with pytest.raises(UnsafeFileNameError):
        sanitize_filename("../../secret.txt")


def test_control_characters_are_rejected() -> None:
    with pytest.raises(UnsafeFileNameError):
        sanitize_filename("bad\x00name.txt")


def test_empty_filename_is_rejected() -> None:
    with pytest.raises(UnsafeFileNameError):
        sanitize_filename("")
    with pytest.raises(UnsafeFileNameError):
        sanitize_filename(None)


def test_display_name_keeps_spaces_and_unicode() -> None:
    assert sanitize_filename("Q3 revenue – final.txt") == "Q3 revenue – final.txt"


def test_windows_reserved_characters_are_replaced() -> None:
    assert sanitize_filename('a<b>c:d"e|f?g*h.txt') == "a_b_c_d_e_f_g_h.txt"


# --- type allow-listing --------------------------------------------------


@pytest.mark.parametrize("ext", [".txt", ".md", ".pdf", ".docx", ".xlsx", ".png", ".mp4", ".zip"])
def test_supported_types_resolve(ext: str) -> None:
    resolved_ext, mime = resolve_media_type(f"sample{ext}")
    assert resolved_ext == ext
    assert mime and mime != "application/octet-stream"


@pytest.mark.parametrize("name", ["virus.exe", "script.bat", "script.ps1", "app.dll", "noext"])
def test_unsupported_types_are_rejected_with_a_clear_message(name: str) -> None:
    with pytest.raises(UnsupportedFileTypeError) as exc:
        resolve_media_type(name)
    assert "supported" in str(exc.value).lower()


def test_normalize_extension() -> None:
    assert normalize_extension("Report.PDF") == ".pdf"
    assert normalize_extension("archive.tar.gz") == ".gz"
    assert normalize_extension("noextension") == ""


# --- content sniffing ----------------------------------------------------


def test_sniff_identifies_real_containers() -> None:
    assert sniff_kind(b"%PDF-1.7 ...") == "pdf"
    assert sniff_kind(b"PK\x03\x04rest") == "zip-container"
    assert sniff_kind(b"\x89PNG\r\n\x1a\n") == "png"
    assert sniff_kind(b"\xff\xd8\xff\xe0") == "jpeg"
    assert sniff_kind(b'{"a": 1}') == "json"
    assert sniff_kind(b"<?xml version='1.0'?>") == "xml"
    assert sniff_kind(b"RIFF\x00\x00\x00\x00WEBP") == "webp"


def test_mismatched_signature_is_reported_not_hidden(tmp_path) -> None:
    """A text file renamed .pdf must be flagged, not silently accepted."""
    fake = tmp_path / "evil.pdf"
    fake.write_bytes(b"this is plainly not a pdf")
    warnings = verify_content_matches_extension(".pdf", fake)
    assert warnings, "a content/extension mismatch must produce a warning"


def test_matching_signature_produces_no_warning(tmp_path) -> None:
    real = tmp_path / "fine.pdf"
    real.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n")
    assert verify_content_matches_extension(".pdf", real) == []


def test_empty_file_is_reported(tmp_path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    assert verify_content_matches_extension(".txt", empty)


# --- size limits ---------------------------------------------------------


def test_size_limit_enforced_while_streaming(tmp_path) -> None:
    from app.ingestion.security import stream_to_disk

    target = tmp_path / "big.bin"
    with pytest.raises(FileTooLargeError):
        stream_to_disk(io.BytesIO(b"x" * 5000), open(target, "wb"), max_bytes=1000)
    # A rejected upload must not leave a partial file behind.
    assert not target.exists() or target.stat().st_size == 0


def test_upload_over_limit_is_rejected(service) -> None:
    with pytest.raises(FileTooLargeError):
        service.save_upload(
            io.BytesIO(b"y" * (service.settings.max_upload_bytes() + 10)),
            "huge.txt",
        )


def test_video_gets_its_own_larger_limit(settings) -> None:
    assert settings.upload_limit_for(True) == settings.max_video_upload_bytes()
    assert settings.upload_limit_for(False) == settings.max_upload_bytes()
    assert settings.max_video_upload_mb > settings.max_upload_mb


def test_empty_upload_is_rejected(service) -> None:
    from app.core.errors import ExtractionError

    with pytest.raises(ExtractionError):
        service.save_upload(io.BytesIO(b""), "nothing.txt")


# --- storage paths -------------------------------------------------------


def test_traversal_filename_is_rejected_outright(service) -> None:
    with pytest.raises(UnsafeFileNameError):
        service.save_upload(io.BytesIO(SAMPLE_NOTE), "../../evil.txt")


def test_storage_path_ignores_the_original_filename(service) -> None:
    """The on-disk name is id-keyed, so the original name cannot steer the path."""
    record = service.save_upload(
        io.BytesIO(SAMPLE_NOTE), "Q3 revenue – final report.txt", session_id="s1"
    )
    path = service.store.storage_path(record)
    assert path.parent == service.store.uploads_dir()
    assert path.name.startswith(record.id)
    assert " " not in path.name
    # The display name is preserved for the UI even though storage ignores it.
    assert record.display_name == "Q3 revenue – final report.txt"


def test_upload_warns_when_signature_does_not_match(service) -> None:
    record = service.save_upload(io.BytesIO(b"not really a pdf"), "fake.pdf", session_id="s1")
    assert any("signature" in w.lower() for w in record.warnings)
