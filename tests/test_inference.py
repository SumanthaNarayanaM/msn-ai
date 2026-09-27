"""Tests for the inference layer."""

from __future__ import annotations

import pytest

from app.core.errors import InferenceError, ProviderNotFoundError
from app.inference.providers.mock import MockLocalModelProvider
from app.inference.registry import create_provider, get_available_providers


def test_mock_load_unload() -> None:
    provider = MockLocalModelProvider()
    assert not provider.is_loaded()
    provider.load()
    assert provider.is_loaded()
    provider.unload()
    assert not provider.is_loaded()


def test_mock_generate_requires_load() -> None:
    provider = MockLocalModelProvider()
    with pytest.raises(InferenceError):
        provider.generate([{"role": "user", "content": "hi"}])


def test_mock_generate_echoes_latest_user_message() -> None:
    provider = MockLocalModelProvider()
    provider.load()
    reply = provider.generate(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "hello there"},
        ]
    )
    assert "hello there" in reply
    assert "[mock:" in reply


def test_mock_generate_empty() -> None:
    provider = MockLocalModelProvider()
    provider.load()
    reply = provider.generate([])
    assert "Tell me something" in reply


def test_mock_info() -> None:
    provider = MockLocalModelProvider(model_config={"model_name": "m-test"})
    provider.load()
    info = provider.info()
    assert info.name == "m-test"
    assert info.provider == "mock"
    assert info.loaded is True


def test_registry_available_providers_include_mock() -> None:
    assert "mock" in get_available_providers()


def test_create_provider_returns_loaded_mock() -> None:
    provider = create_provider(name="mock", force_new=True)
    assert provider.is_loaded()


def test_create_provider_unknown_raises() -> None:
    with pytest.raises(ProviderNotFoundError):
        create_provider(name="does-not-exist")