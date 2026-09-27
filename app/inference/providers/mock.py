"""Mock local model provider.

Serves deterministic, clearly-marked replies so the whole platform stack can
run end-to-end before a real local coding model is connected.

It must be replaced by a real local model provider (MSN AI Code) later; the
ModelProvider interface in :mod:`app.inference.base` is the only contract the
rest of the platform depends on.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from app.core.logging import get_logger
from app.inference.base import ModelProvider
from app.models.schemas import ModelInfo

logger = get_logger("msn_ai.provider.mock")

_HELP_HINT = (
    "\n\n[MSN AI mock provider] Replace MSN_AI_PROVIDER with a real local "
    "provider (e.g. 'llamacpp' or 'transformers') to connect an actual model."
)


class MockLocalModelProvider(ModelProvider):
    """Deterministic text provider used in place of a real model (v0.1)."""

    name = "mock"

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        super().__init__(model_config)
        self.model_name = self.model_config.get("model_name", "mock-msn-ai-v0.1")
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        logger.info("Loading mock provider '%s' (no real weights).", self.model_name)
        time.sleep(0.05)  # Simulate load work.
        self._loaded = True
        logger.info("Mock provider ready.")

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def generate(
        self, messages: Iterable[dict[str, str]], **kwargs: Any
    ) -> str:
        self._require_loaded()

        history = list(messages)
        if not history:
            return "Tell me something!" + _HELP_HINT

        user_messages = [m["content"] for m in history if m.get("role") == "user"]
        latest = user_messages[-1] if user_messages else history[-1]["content"]

        max_tokens = int(kwargs.get("max_tokens") or self.model_config.get("max_tokens", 1024))
        temperature = float(
            kwargs.get("temperature") or self.model_config.get("temperature", 0.7)
        )

        logger.info(
            "Mock generate: %,d messages, %d tokens, temperature=%.2f",
            len(history),
            max_tokens,
            temperature,
        )

        preview = " ".join(latest.split())[:200]
        reply = (
            f"[mock:{self.model_name}] Echo: {preview!r}\n"
            f"[provider=mock | messages={len(history)} | "
            f"max_tokens={max_tokens} | temperature={temperature:.2f}]"
        )
        return reply + _HELP_HINT

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model_name,
            provider=self.name,
            family=self.family,
            loaded=self.is_loaded(),
            description=(
                "Mock local provider. No real weights; replace with a local "
                "coding model provider (MSN AI Code) in a later release."
            ),
        )