"""Tests for conversation context, session persistence, coding requests,
provider fallback and the real provider configuration -- no model download
is required (the model runtime is stubbed or skipped)."""

from __future__ import annotations

from typing import Any, Iterable

from fastapi.testclient import TestClient

from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.inference.registry import create_provider, create_provider_safe
from app.main import create_app
from app.models.schemas import ModelInfo


class _RecordedProvider(ModelProvider):
    """Test provider that records every call and answers deterministically."""

    name = "recorder"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[dict[str, str]]] = []
        self._loaded = True

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def generate(self, messages: Iterable[dict[str, str]], **kwargs: Any) -> str:
        history = [dict(m) for m in messages]
        self.calls.append(history)
        return "reply-from-recorder"

    def info(self) -> ModelInfo:
        return ModelInfo(name="recorder", provider=self.name, loaded=self.is_loaded())


class _ClientWithRecorder:
    """TestClient that swaps in a recording provider after startup."""

    def __init__(self) -> None:
        self.app = create_app()
        self.provider = _RecordedProvider()
        self.client = TestClient(self.app)

    def __enter__(self) -> tuple[TestClient, _RecordedProvider]:
        self.client.__enter__()
        self.app.state.provider = self.provider
        return self.client, self.provider

    def __exit__(self, *exc) -> None:
        self.client.__exit__(*exc)


def test_chat_injects_personality_and_full_history() -> None:
    with _ClientWithRecorder() as (client, provider):
        first = client.post("/api/v1/chat", json={"message": "turn one"}).json()
        sid = first["session_id"]
        client.post("/api/v1/chat", json={"message": "turn two", "session_id": sid})

    last_call = provider.calls[-1]
    roles = [m["role"] for m in last_call]
    assert roles[0] == "system"
    assert "MSN AI" in last_call[0]["content"]
    contents = [m["content"] for m in last_call]
    assert "turn one" in contents
    assert "turn two" in contents


def test_conversation_context_memory() -> None:
    with _ClientWithRecorder() as (client, _):
        first = client.post(
            "/api/v1/chat", json={"message": "My favorite project is MSN AI."}
        ).json()
        sid = first["session_id"]
        second = client.post(
            "/api/v1/chat",
            json={"message": "What project did I mention?", "session_id": sid},
        ).json()
        assert second["session_id"] == sid
        detail = client.get(f"/api/v1/sessions/{sid}").json()
        contents = [m["content"] for m in detail["messages"]]
        assert "My favorite project is MSN AI." in contents
        assert "What project did I mention?" in contents


def test_coding_request_accepted_via_api() -> None:
    with _ClientWithRecorder() as (client, provider):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Write a Python function that reverses a string."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["reply"]
        call = provider.calls[-1]
        assert "Python function" in call[-1]["content"]


def test_session_persistence_roundtrip(tmp_path) -> None:
    manager = SessionManager(storage_dir=tmp_path, persist=True)
    session = manager.create()
    manager.add_message(session.id, "user", "Hello")
    manager.add_message(session.id, "assistant", "Hi there")
    saved_path = tmp_path / f"{session.id}.json"
    assert saved_path.exists()

    restored = SessionManager(storage_dir=tmp_path, persist=True)
    history = restored.history(session.id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hello"


def test_session_no_persist_writes_nothing(tmp_path) -> None:
    manager = SessionManager(storage_dir=tmp_path, persist=False)
    session = manager.create()
    manager.add_message(session.id, "user", "Hello")
    assert not any(tmp_path.iterdir())


def test_provider_fallback_to_mock() -> None:
    provider, meta = create_provider_safe(
        name="llamacpp",
        model_name="qwen-test",
        provider_config={"model_path": "X:\\does-not-exist.gguf"},
    )
    assert meta["fallback"] is True
    assert meta["resolved"] == "mock"
    assert provider.name == "mock"


def test_provider_fallback_uses_mock_without_model_runtime() -> None:
    provider, meta = create_provider_safe(
        name="local",
        model_name="qwen-test",
        provider_config={"model_path": "X:\\missing.gguf"},
    )
    assert provider.name == "mock"
    assert meta["fallback"] is True


def test_real_provider_configuration_mapping() -> None:
    from app.inference.providers.local_llm import RealLocalModelProvider

    provider = RealLocalModelProvider(
        model_config={
            "model_name": "qwen2.5-1.5b-instruct",
            "model_path": "X:\\missing.gguf",
            "n_ctx": 2048,
            "n_threads": 8,
            "max_tokens": 512,
            "temperature": 0.7,
        }
    )
    assert provider.name == "llamacpp"
    assert provider.model_name == "qwen2.5-1.5b-instruct"
    assert "missing.gguf" in str(provider.model_path)
    assert provider.n_ctx == 2048
    assert provider.n_threads == 8
    assert provider.max_tokens == 512
    assert provider.temperature == 0.7
    assert provider.is_loaded() is False


def test_local_alias_points_to_real_provider() -> None:
    from app.inference.registry import _PROVIDERS

    from app.inference.providers.local_llm import RealLocalModelProvider

    assert _PROVIDERS["llamacpp"] is RealLocalModelProvider
    assert _PROVIDERS["local"] is RealLocalModelProvider


def test_unknown_provider_raises() -> None:
    try:
        create_provider(name="nope")
    except Exception as exc:
        assert "Unknown provider" in str(exc)
    else:
        raise AssertionError("Expected an error for an unknown provider name")


def test_session_schema_fits_pydantic_model() -> None:
    session = SessionManager(storage_dir=None).create()
    session.add_message("user", "abc")
    dump = session.model_dump()
    assert "user" in [m["role"] for m in dump["messages"]]