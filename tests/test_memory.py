"""Tests for the long-term memory abstraction and its API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.memory import LocalMemoryStore, MemoryManager
from app.memory import store as store_module


def test_store_add_and_persist(tmp_path) -> None:
    store = LocalMemoryStore(storage_dir=tmp_path)
    record = store.add(
        "My favorite project is MSN AI.",
        type="project",
        metadata={"topic": "chat"},
        source_session_id="abc",
    )
    assert record.id
    assert record.type == "project"

    reloaded = LocalMemoryStore(storage_dir=tmp_path)
    records = reloaded.all()
    assert len(records) == 1
    assert records[0].content == "My favorite project is MSN AI."
    assert records[0].source_session_id == "abc"


def test_store_validates_type(tmp_path) -> None:
    store = LocalMemoryStore(storage_dir=tmp_path)
    try:
        store.add("nonsense", type="bogus")
    except ValueError as exc:
        assert "Unknown memory type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid memory type")


def test_store_search_keyword_overlap(tmp_path) -> None:
    store = LocalMemoryStore(storage_dir=tmp_path)
    store.add("My name is Sumantha.", type="fact")
    store.add("I like hiking in the mountains.", type="preference")

    results = store.search("name Sumantha")
    contents = [r.content for r in results]
    assert "My name is Sumantha." in contents
    assert "hiking" not in contents


def test_uncached_store_keeps_records_off_screen(tmp_path) -> None:
    store = LocalMemoryStore(storage_dir=None)
    store.add("ephemeral fact", type="fact")
    assert len(store.all()) == 1


def test_manager_remember_recall_and_context(tmp_path) -> None:
    manager = MemoryManager(storage_dir=tmp_path)
    manager.remember("The user works at Acme Corp.", type="fact")
    matches = manager.recall("Acme Corp", limit=5)
    assert len(matches) == 1
    context = manager.recall_for_context("Where does the user work?")
    assert "Acme Corp" in context

    assert manager.recall_for_context("nothing matches this query") == ""


def test_manager_extract_fact_heuristics(tmp_path) -> None:
    manager = MemoryManager(storage_dir=tmp_path)
    assert manager.extract_fact("My name is Sumantha") == "my name is sumantha"
    assert manager.extract_fact("I like Python") == "i like python"
    assert manager.extract_fact("What is the weather like?") is None
    assert manager.extract_fact("My favorite movie is Inception") is not None


def test_memory_api_endpoints(tmp_path, monkeypatch) -> None:
    from app.main import create_app

    app = create_app()
    app.state.memory_manager = MemoryManager(
        store=LocalMemoryStore(storage_dir=tmp_path)
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/memory",
            json={"content": "User prefers dark mode.", "type": "preference"},
        )
        assert created.status_code == 200
        record_id = created.json()["record"]["id"]

        listing = client.get("/api/v1/memory")
        assert listing.status_code == 200
        assert listing.json()["count"] == 1

        recalled = client.get("/api/v1/memory/recall", params={"query": "dark mode"})
        assert recalled.status_code == 200
        assert recalled.json()["count"] == 1

        deleted = client.delete(f"/api/v1/memory/{record_id}")
        assert deleted.status_code == 200
        assert client.get("/api/v1/memory").json()["count"] == 0


def test_memory_api_rejects_invalid_type(tmp_path) -> None:
    from app.main import create_app

    app = create_app()
    app.state.memory_manager = MemoryManager(
        store=LocalMemoryStore(storage_dir=tmp_path)
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memory", json={"content": "x", "type": "nonsense"}
        )
        assert response.status_code == 422


def test_store_module_exports() -> None:
    assert store_module.VALID_TYPES == {"fact", "preference", "project", "story"}