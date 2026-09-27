"""Real local model provider backed by llama.cpp (llama-cpp-python).

MSN AI's initial local language engine. Wraps llama.cpp's ``Llama`` runtime so
a small GGUF chat model (e.g. Qwen2.5-1.5B-Instruct Q4_K_M) can serve the API
fully offline on CPU.

The ``llama_cpp`` package is imported lazily inside :meth:`load` so that the
rest of the platform (and the unit tests) never require the model runtime to
be installed.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from app.core.errors import InferenceError
from app.core.logging import get_logger
from app.inference.base import ModelProvider
from app.models.schemas import ModelInfo

logger = get_logger("msn_ai.provider.local_llm")


class RealLocalModelProvider(ModelProvider):
    """Serve a local GGUF chat model through the llama.cpp runtime."""

    name = "llamacpp"
    family = "text-generation"

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        super().__init__(model_config)
        config = self.model_config

        self.model_name: str = config.get("model_name", "local-chat-model")
        self.model_path: Path = Path(str(config.get("model_path") or ""))
        self.n_ctx: int = int(config.get("n_ctx", 2048))
        self.n_threads: int | None = config.get("n_threads") or None
        self.max_tokens: int = int(config.get("max_tokens", 1024))
        self.temperature: float = float(config.get("temperature", 0.7))

        self._llm: Any | None = None
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if self._llm is not None:
                return

            if not self.model_path or not self.model_path.is_file():
                raise InferenceError(
                    "Local GGUF model not found at "
                    f"'{self.model_path or '<unset>'}'.\n"
                    "Download it first, e.g.:\n"
                    "  powershell -ExecutionPolicy Bypass -File "
                    "D:\\MSN AI\\scripts\\fetch_model.ps1\n"
                    "or set MSN_AI_MODEL_PATH to an existing .gguf file."
                )

            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise InferenceError(
                    "llama-cpp-python is not installed. Install it with:\n"
                    "  pip install -r D:\\MSN AI\\requirements-local.txt"
                ) from exc

            started = time.time()
            logger.info(
                "Loading %s from %s (n_ctx=%d, n_threads=%s) ...",
                self.model_name,
                self.model_path,
                self.n_ctx,
                self.n_threads,
            )
            try:
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                    n_gpu_layers=0,
                )
            except Exception as exc:
                raise InferenceError(
                    f"Failed to load the GGUF model: {exc}"
                ) from exc
            logger.info(
                "Model ready in %.1fs (%s).", time.time() - started, self.model_name
            )

    def is_loaded(self) -> bool:
        return self._llm is not None

    def unload(self) -> None:
        with self._lock:
            self._llm = None
            logger.info("Local model unloaded.")

    # --- inference -------------------------------------------------------

    def generate(
        self, messages: Iterable[dict[str, str]], **kwargs: Any
    ) -> str:
        self._require_loaded()
        assert self._llm is not None

        history = [dict(m) for m in messages]
        max_tokens = int(kwargs.get("max_tokens") or self.max_tokens)
        temperature = float(kwargs.get("temperature") or self.temperature)

        try:
            response = self._llm.create_chat_completion(
                messages=history,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=kwargs.get("stop"),
            )
        except Exception as exc:
            raise InferenceError(f"Local model inference failed: {exc}") from exc

        try:
            reply = response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                f"Unexpected response shape from llama.cpp: {response!r}"
            ) from exc

        logger.debug(
            "Local generate: %d turns, %d tokens, temperature=%.2f",
            len(history),
            max_tokens,
            temperature,
        )
        return reply

    # --- diagnostics -----------------------------------------------------

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model_name,
            provider=self.name,
            family=self.family,
            loaded=self.is_loaded(),
            description=(
                "Real local chat model served by llama.cpp "
                + (f"({os.path.basename(str(self.model_path))})" if self.model_path else "")
            ),
        )