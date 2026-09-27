"""Tests for the session manager."""

from __future__ import annotations

from app.core.errors import SessionNotFoundError
from app.core.session import SessionManager


def test_create_session_has_id_and_timestamps() -> None:
    manager = SessionManager()
    session = manager.create()
    assert session.id
    assert session.created_at > 0
    assert session.messages == []


def test_get_unknown_session_raises() -> None:
    manager = SessionManager()
    try:
        manager.get("nope")
    except SessionNotFoundError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected SessionNotFoundError")


def test_add_and_retrieve_messages() -> None:
    manager = SessionManager()
    session = manager.create()
    manager.add_message(session.id, "user", "hello")
    manager.add_message(session.id, "assistant", "hi")
    history = manager.history(session.id)
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content == "hello"
    assert session.updated_at >= session.created_at


def test_recent_history_limits_and_order() -> None:
    manager = SessionManager()
    session = manager.create()
    for i in range(5):
        manager.add_message(session.id, "user", f"msg-{i}")
    recent = manager.recent_history(session.id, max_messages=2)
    assert [m.content for m in recent] == ["msg-3", "msg-4"]
    assert [m.content for m in manager.history(session.id)][-1] == "msg-4"


def test_delete_removes_session() -> None:
    manager = SessionManager()
    session = manager.create()
    manager.delete(session.id)
    assert session.id not in manager.list_ids()


def test_get_or_create_respects_existing() -> None:
    manager = SessionManager()
    created = manager.create()
    again = manager.get_or_create(created.id)
    assert again.id == created.id


def test_persistence_roundtrip(tmp_path) -> None:
    manager = SessionManager(storage_dir=tmp_path, persist=True)
    session = manager.create()
    manager.add_message(session.id, "user", "persist me")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1

    reloaded = SessionManager(storage_dir=tmp_path, persist=True)
    restored = reloaded.get(session.id)
    assert restored.messages[0].content == "persist me"


def test_instances_are_independent() -> None:
    a = SessionManager()
    b = SessionManager()
    a.create()
    assert b.list_ids() == []