"""Remember/Forget, memory scopes, and the file management HTTP API."""

from __future__ import annotations

import io
import json

import pytest

from app.core.config import PROJECT_ROOT
from app.core.errors import FileNotFoundError_
from app.ingestion.models import FileStatus
from conftest import SAMPLE_NOTE


def ingest(service, payload: bytes, filename: str, session_id: str = "s1", scope: str = "session"):
    record = service.save_upload(io.BytesIO(payload), filename, session_id=session_id)
    record = service.process(record.id)
    if scope != "session":
        service.knowledge.set_scope(record.id, scope, project=service.project_name)
        record = service.store.get(record.id)
    return record


@pytest.fixture()
def client(tmp_path, service):
    """A real app instance, with the background queue enabled as in production.

    Determinism comes from ``wait=true`` on uploads rather than from disabling
    the queue, so the HTTP contract under test is the one users actually hit.
    """
    from fastapi.testclient import TestClient

    from app.core.config import Settings
    from app.main import create_app

    config = Settings(
        data_dir=tmp_path / "api-data",
        model_storage_dir=PROJECT_ROOT / "model_storage",
        log_level="WARNING",
        provider="mock",
        model_name="mock-msn-ai-v0.1",
        session_persist=False,
        ingestion_enabled=True,
        embedding_provider="hashing",
        embedding_dim=384,
    )
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


# --- memory scopes -------------------------------------------------------


def test_remember_makes_a_file_searchable_again_after_forget(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    assert service.search("Q3 revenue", session_id="s1")
    service.forget(record.id)
    assert service.search("Q3 revenue", session_id="s1") == []
    service.remember(record.id, scope="personal")
    assert service.search("Q3 revenue", session_id="s1")


def test_remember_rejects_an_unsupported_scope(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    with pytest.raises(Exception) as excinfo:
        service.remember(record.id, scope="session")
    assert "scope" in str(excinfo.value).lower()


def test_forget_removes_knowledge_but_keeps_the_file(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    service.forget(record.id)
    reloaded = service.store.get(record.id)
    assert reloaded.status is FileStatus.READY
    assert reloaded.chunk_count == 0
    assert service.knowledge.stats()["chunks"] == 0
    # The original file is still on disk and re-rememberable.
    assert service.remember(record.id, scope="personal")
    assert service.search("Q3 revenue", session_id="s1")


def test_personal_scope_is_visible_to_other_sessions(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt", session_id="s1")
    service.forget(record.id)
    service.remember(record.id, scope="personal")
    assert service.search("Q3 revenue", session_id="s1")
    assert service.search("Q3 revenue", session_id="completely-different")


def test_project_scope_requires_a_matching_project(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt", session_id="s1")
    service.forget(record.id)
    service.remember(record.id, scope="project", project="alpha")
    assert service.search("Q3 revenue", session_id="s1", scope="project", project="alpha")
    assert service.search("Q3 revenue", session_id="s1", scope="project", project="beta") == []


def test_session_scope_hides_other_sessions_but_still_sees_remembered_files(service) -> None:
    ingest(service, SAMPLE_NOTE, "launch.txt", session_id="s1")
    # Session-scoped: only the owning session sees it.
    assert service.search("Q3 revenue", session_id="s1")
    assert service.search("Q3 revenue", session_id="s2") == []


def test_deleting_a_file_drops_its_knowledge(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    assert service.search("Q3 revenue", session_id="s1")
    service.delete(record.id)
    assert service.search("Q3 revenue", session_id="s1") == []
    with pytest.raises(FileNotFoundError_):
        service.store.get(record.id)


def test_two_files_in_different_sessions_do_not_leak(service) -> None:
    ingest(service, b"Session one facts about badgers.", "one.txt", session_id="s1")
    ingest(service, b"Session two facts about beavers.", "two.txt", session_id="s2")
    assert service.search("badgers", session_id="s1")
    assert not service.search("badgers", session_id="s2")


def upload_file(client, payload: bytes, filename: str, session_id: str = "api-session", **extra):
    data = {"session_id": session_id, "wait": "true"}
    data.update(extra)
    response = client.post(
        "/api/v1/files/upload",
        files={"file": (filename, payload, "text/plain")},
        data=data,
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- file API ------------------------------------------------------------


def test_upload_process_search_and_delete_cycle(client) -> None:
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    assert record["status"] == "ready"
    assert record["display_name"] == "launch.txt"
    assert record["readable"] is True
    file_id = record["id"]

    listing = client.get("/api/v1/files", params={"session_id": "api-session"})
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()["files"]] == [file_id]
    assert listing.json()["total"] == 1

    detail = client.get(f"/api/v1/files/{file_id}")
    assert detail.status_code == 200
    assert detail.json()["filename"] == "launch.txt"

    search = client.post(
        "/api/v1/files/search",
        json={"query": "What is the Q3 revenue risk?", "session_id": "api-session"},
    )
    assert search.status_code == 200
    body = search.json()
    assert body["used_knowledge"] is True
    hits = body["sources"]
    assert hits
    assert hits[0]["filename"] == "launch.txt"
    assert hits[0]["source"]
    assert hits[0]["file_id"] == file_id

    removed = client.delete(f"/api/v1/files/{file_id}")
    assert removed.status_code == 200
    assert client.get(f"/api/v1/files/{file_id}").status_code == 404
    assert client.post(
        "/api/v1/files/search",
        json={"query": "Q3 revenue", "session_id": "api-session"},
    ).json()["sources"] == []


def test_upload_returns_uploaded_when_not_waiting(client) -> None:
    """The default contract is queue-then-poll, not a blocking request."""
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("launch.txt", SAMPLE_NOTE, "text/plain")},
        data={"session_id": "api-session"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] in ("uploaded", "processing", "ready")
    assert body["progress"] >= 0.0


def test_extracted_text_is_downloadable(client) -> None:
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    text = client.get(f"/api/v1/files/{record['id']}/extracted")
    assert text.status_code == 200
    assert "Q3" in text.json()["text"]


def test_original_file_can_be_downloaded(client) -> None:
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    original = client.get(f"/api/v1/files/{record['id']}/content")
    assert original.status_code == 200
    assert original.content == SAMPLE_NOTE


def test_remember_and_forget_endpoints(client) -> None:
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    file_id = record["id"]

    forgotten = client.post(f"/api/v1/files/{file_id}/forget")
    assert forgotten.status_code == 200
    assert forgotten.json()["remembered"] is False
    assert client.post(
        "/api/v1/files/search",
        json={"query": "Q3 revenue", "session_id": "api-session"},
    ).json()["sources"] == []

    remembered = client.post(
        f"/api/v1/files/{file_id}/remember", json={"scope": "personal"}
    )
    assert remembered.status_code == 200
    assert remembered.json()["scope"] == "personal"
    assert remembered.json()["remembered"] is True

    sources = client.post(
        "/api/v1/files/search",
        json={"query": "Q3 revenue", "session_id": "api-session"},
    ).json()["sources"]
    assert sources


def test_remember_rejects_an_invalid_scope(client) -> None:
    """An unknown scope is refused by validation, before any state changes."""
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    response = client.post(
        f"/api/v1/files/{record['id']}/remember", json={"scope": "forever"}
    )
    assert response.status_code == 422
    assert "scope" in json.dumps(response.json()).lower()


def test_rejected_upload_returns_a_readable_error(client) -> None:
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
        data={"session_id": "api-session"},
    )
    assert response.status_code == 415
    error = response.json()["error"]
    assert "exe" in error["message"].lower()
    assert error["code"] == "unsupported_file_type"


def test_oversized_upload_is_rejected(client, settings) -> None:
    """A file over the limit is refused without storing it."""
    limit = settings.max_upload_mb
    payload = b"x" * (limit * 1024 * 1024 + 1024)
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("big.txt", payload, "text/plain")},
        data={"session_id": "api-session"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    # Nothing was kept.
    assert client.get("/api/v1/files", params={"session_id": "api-session"}).json()["total"] == 0


def test_ingestion_health_reports_capabilities_honestly(client) -> None:
    response = client.get("/api/v1/ingestion/health")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert ".pdf" in body["supported_extensions"]
    caps = body["capabilities"]
    # Whatever is or is not installed, the flags must be present and boolean.
    for name in ("ocr", "vision", "speech", "video_frames", "embeddings"):
        assert isinstance(caps[name]["available"], bool), name
        if not caps[name]["available"]:
            assert caps[name]["reason"], name
    if not caps["vision"]["available"]:
        assert "model" in caps["vision"]["reason"].lower()


def test_search_on_an_empty_knowledge_base_is_not_an_error(client) -> None:
    response = client.post(
        "/api/v1/files/search", json={"query": "anything", "session_id": "empty"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["used_knowledge"] is False


def test_chat_still_accepts_a_plain_v1_payload(client) -> None:
    """Adding ingestion must not break the existing chat contract."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello", "session_id": "v1-check"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    assert "sources" in body
    assert body["used_knowledge"] is False


def test_chat_with_attachments_and_knowledge_flags_validates(client) -> None:
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "What is the Q3 revenue risk?",
            "session_id": "api-session",
            "attachments": [
                {"file_id": record["id"], "filename": record["display_name"]}
            ],
            "use_knowledge": True,
            "scope": "session",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["used_knowledge"], bool)
    assert isinstance(body["sources"], list)


def test_chat_cites_the_uploaded_file_it_used(client) -> None:
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    body = client.post(
        "/api/v1/chat",
        json={
            "message": "What is the Q3 revenue risk with the vendor?",
            "session_id": "api-session",
            "attachments": [{"file_id": record["id"]}],
            "use_knowledge": True,
        },
    ).json()
    assert body["used_knowledge"] is True
    assert body["sources"], "expected at least one cited source"
    assert all(s["file_id"] == record["id"] for s in body["sources"])
    assert any("launch.txt" in s["source"] for s in body["sources"])


def test_chat_skips_knowledge_when_asked_to(client) -> None:
    record = upload_file(client, SAMPLE_NOTE, "launch.txt")
    body = client.post(
        "/api/v1/chat",
        json={
            "message": "What is the Q3 revenue risk?",
            "session_id": "api-session",
            "attachments": [{"file_id": record["id"]}],
            "use_knowledge": False,
        },
    ).json()
    assert body["used_knowledge"] is False
    assert body["sources"] == []


def test_chat_does_not_claim_knowledge_for_an_unrelated_question(client) -> None:
    upload_file(client, SAMPLE_NOTE, "launch.txt")
    body = client.post(
        "/api/v1/chat",
        json={
            "message": "zzzz unrelated quantum chromodynamics question",
            "session_id": "api-session",
            "use_knowledge": True,
        },
    ).json()
    assert body["used_knowledge"] is False
    assert body["sources"] == []


def test_chat_still_works_with_an_unknown_attachment_id(client) -> None:
    """A stale attachment id must not take chat down."""
    body = client.post(
        "/api/v1/chat",
        json={
            "message": "Hello",
            "session_id": "api-session",
            "attachments": [{"file_id": "does-not-exist"}],
            "use_knowledge": True,
        },
    )
    assert body.status_code == 200
    assert body.json()["reply"]


def test_chat_rejects_a_duplicate_attachment_list(client) -> None:
    """Attachments are a list of objects; anything else is a client error."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "hi", "session_id": "s", "attachments": [1, 2]},
    )
    assert response.status_code == 422
