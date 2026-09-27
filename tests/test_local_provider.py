"""Tests for the real local provider (no model download required).

The real llama.cpp runtime is stubbed out; unit tests verify the provider's
contract, error paths and message passing without touching a model file.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from app.core.errors import InferenceError
from app.inference.providers.local_llm import RealLocalModelProvider
from app.inference.registry import create_provider, get_available_providers

TARGET = "llama_cpp"


def test_info_before_load() -> None:
    provider = RealLocalModelProvider(
        model_config={"model_name": "qwen-test", "model_path": "X:\\missing.gguf"}
    )
    info = provider.info()
    assert info.name == "qwen-test"
    assert info.provider == "llamacpp"
    assert info.loaded is False


def test_generate_before_load_raises() -> None:
    provider = RealLocalModelProvider(
        model_config={"model_path": "X:\\missing.gguf"}
    )
    with pytest.raises(InferenceError):
        provider.generate([{"role": "user", "content": "hi"}])


def test_load_missing_file_raises_informative_error(tmp_path) -> None:
    missing = tmp_path / "nope.gguf"
    provider = RealLocalModelProvider(model_config={"model_path": str(missing)})
    with pytest.raises(InferenceError) as excinfo:
        provider.load()
    message = str(excinfo.value)
    assert "not found" in message
    assert "fetch_model.ps1" in message


class _Channel:
    """Fake Llama runtime recording calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello from fake llama"}}
            ]
        }


@pytest.fixture
def fake_llama(monkeypatch):
    channel = _Channel()
    mod = ModuleType(TARGET)
    mod.Llama = lambda **kw: channel

    class _Ctx:
        def __enter__(self):
            monkeypatch.setitem(sys.modules, TARGET, mod)
            return channel

        def __exit__(self, *exc):
            monkeypatch.delitem(sys.modules, TARGET, raising=False)
            return False

    return _Ctx()


def test_load_and_generate_with_stubbed_runtime(tmp_path, fake_llama) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake-gguf-bytes")

    with fake_llama as channel:
        provider = RealLocalModelProvider(
            model_config={"model_path": str(gguf), "n_ctx": 128, "n_threads": 2}
        )
        assert not provider.is_loaded()
        provider.load()
        assert provider.is_loaded()

        reply = provider.generate(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=64,
            temperature=0.5,
        )

    assert reply == "Hello from fake llama"
    assert len(channel.calls) == 1
    call = channel.calls[0]
    assert call["messages"][-1]["content"] == "hello"
    assert call["kwargs"]["max_tokens"] == 64
    assert call["kwargs"]["temperature"] == 0.5


def test_unload_frees_runtime(tmp_path, fake_llama) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake-gguf-bytes")
    with fake_llama:
        provider = RealLocalModelProvider(model_config={"model_path": str(gguf)})
        provider.load()
        provider.unload()
        assert not provider.is_loaded()


def test_registry_lists_real_provider() -> None:
    names = set(get_available_providers())
    assert "llamacpp" in names
    assert "local" in names
    assert "mock" in names


def test_registry_creation_fails_cleanly_without_model(tmp_path) -> None:
    with pytest.raises(InferenceError) as excinfo:
        create_provider(
            name="local",
            model_name="qwen-test",
            force_new=True,
            provider_config={"model_path": str(tmp_path / "missing.gguf")},
        )
    assert "not found" in str(excinfo.value)