"""Pydantic settings backed by .env support."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="MSN_AI_",
    )

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True

    # Application
    app_name: str = "MSN AI"
    env: str = "development"
    debug: bool = True
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    model_storage_dir: Path = PROJECT_ROOT / "model_storage"
    conversations_dir: Path | None = None

    # Inference
    provider: str = "mock"
    model_name: str = "mock-msn-ai-v0.1"
    max_tokens: int = 1024
    temperature: float = 0.7
    request_timeout_seconds: float = 60.0

    # Conversation / memory
    session_persist: bool = True
    memory_dir: Path | None = None
    system_prompt: str = (
        "You are MSN AI, a friendly, natural and helpful local assistant. "
        "You are comfortable with casual conversation AND detailed technical "
        "help, switching between the two naturally. Be conversational and warm, "
        "not robotic. Do not constantly say 'As an AI...'. Be concise when a "
        "short answer is enough, and go into detail when the user asks for it. "
        "You are technically capable: you can write and explain code (Python, "
        "JavaScript, React and more), explain errors, and reason about problems. "
        "Never invent personal facts about the user; remember only what they "
        "explicitly share. Use the conversation history to stay consistent."
    )

    # Real local model (llama.cpp backend)
    model_path: Path | None = None
    n_ctx: int = 2048
    n_threads: int = 0  # 0 == auto (logical CPU count)
    model_hf_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    model_hf_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    @field_validator("conversations_dir", mode="before")
    @classmethod
    def _default_conversations_dir(cls, value: object) -> object:
        return value

    @field_validator("memory_dir", mode="before")
    @classmethod
    def _default_memory_dir(cls, value: object) -> object:
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper()

    def conversations_path(self) -> Path:
        return self.conversations_dir or (self.data_dir / "conversations")

    def memory_path(self) -> Path:
        return self.memory_dir or (self.data_dir / "memory")

    def resolved_model_path(self) -> Path:
        """Explicit model path, or the default GGUF location under model_storage."""
        return self.model_path or (self.model_storage_dir / "base" / self.model_hf_file)

    def resolved_n_threads(self) -> int | None:
        """Return the thread count for llama.cpp (None means auto)."""
        return None if self.n_threads <= 0 else self.n_threads

    def ensure_directories(self) -> None:
        """Create all runtime directories that the application expects."""
        from app.utils.paths import ensure_standard_directories

        ensure_standard_directories(
            data_dir=self.data_dir,
            model_storage_dir=self.model_storage_dir,
            conversations_dir=self.conversations_path(),
        )

    def resolved_options(self) -> dict:
        """Flatten settings into a plain dict (useful for dependency injection)."""
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "app_name": self.app_name,
            "env": self.env,
            "debug": self.debug,
            "version": self.version,
            "log_level": self.log_level,
            "data_dir": str(self.data_dir),
            "model_storage_dir": str(self.model_storage_dir),
            "conversations_dir": str(self.conversations_path()),
            "session_persist": self.session_persist,
            "memory_dir": str(self.memory_path()),
            "system_prompt": self.system_prompt,
            "provider": self.provider,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model_path": str(self.resolved_model_path()),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "model_hf_repo": self.model_hf_repo,
            "model_hf_file": self.model_hf_file,
        }


@lru_cache
def get_settings() -> Settings:
    """Return a cached, lazily-initialized Settings instance."""
    return Settings()


settings = get_settings()