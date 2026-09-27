"""Tests for configuration."""

from __future__ import annotations

from app.core.config import PROJECT_ROOT, Settings


def test_defaults_point_at_project_root() -> None:
    s = Settings()
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.model_storage_dir == PROJECT_ROOT / "model_storage"


def test_conversations_path_defaults() -> None:
    s = Settings()
    assert s.conversations_path() == PROJECT_ROOT / "data" / "conversations"


def test_env_prefix_override(monkeypatch) -> None:
    monkeypatch.setenv("MSN_AI_PORT", "9000")
    monkeypatch.setenv("MSN_AI_PROVIDER", "mock")
    s = Settings()
    assert s.port == 9000
    assert s.provider == "mock"


def test_log_level_normalized() -> None:
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"


def test_resolved_options_are_flat() -> None:
    s = Settings()
    options = s.resolved_options()
    assert options["provider"] == "mock"
    assert "data_dir" in options
    assert "conversations_dir" in options