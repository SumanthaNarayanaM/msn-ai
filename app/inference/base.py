"""Abstract ModelProvider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from app.core.errors import InferenceError
from app.models.schemas import ModelInfo


class ModelProvider(ABC):
    """Interface every MSN AI model provider must implement.

    A provider wraps one model backend. To connect a real local coding model
    later, implement a new provider (e.g. ``LocalModelProvider``) exposing the
    same interface and register it in :mod:`app.inference.registry`.
    """

    #: Machine-readable provider identifier, e.g. "mock" or "llamacpp".
    name: str = "base"

    #: Human-readable family, used by platform capabilities (code/image/etc.).
    family: str = "text-generation"

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        self.model_config = model_config or {}

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Idempotent; safe to call repeatedly."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True when the model is ready to serve requests."""

    @abstractmethod
    def generate(
        self, messages: Iterable[dict[str, str]], **kwargs: Any
    ) -> str:
        """Generate a reply for a conversation of ``[{role, content}, ...]``.

        Keyword arguments may carry generation options (max_tokens,
        temperature, stop sequences, ...) and are provider-specific.
        """

    @abstractmethod
    def info(self) -> ModelInfo:
        """Describe this provider/model for diagnostics and /health."""

    # --- optional lifecycle hooks ---

    def unload(self) -> None:
        """Release the model. Defaults to a no-op."""

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise InferenceError("Model is not loaded yet.")