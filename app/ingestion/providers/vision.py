"""Local vision-language provider (llama.cpp multimodal / Ollama compatible).

MSN AI's chat model (Qwen2.5 1.5B) is text-only, so image understanding requires a
*separate* local vision model. This provider loads one on demand; when no vision
model is configured or installed it reports itself unavailable and the ingestion
pipeline surfaces that verbatim instead of inventing a description.
"""

from __future__ import annotations

import base64
import mimetypes
import threading
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.ingestion.providers.base import CapabilityProvider, VisionResult

logger = get_logger("msn_ai.ingestion.vision")

#: Prompt used to elicit a factual description of an uploaded image/frame.
VISION_PROMPT = (
    "Describe this image in detail. Transcribe any visible text exactly. "
    "Mention objects, people, charts, tables, code on screen and the setting. "
    "Be factual and specific; if the image is unclear, say so."
)

_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a careful image analyst. Describe only what is "
    "actually visible.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class LocalVisionProvider(CapabilityProvider):
    """Describe images with a locally installed multimodal GGUF model."""

    name = "local-vision"

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = model_config or {}
        self.model_name: str = str(config.get("model_name") or "none")
        self.model_path: Path | None = config.get("model_path") or None
        self.mmproj_path: Path | None = config.get("mmproj_path") or None
        self.n_ctx: int = int(config.get("n_ctx") or 4096)
        self.n_threads: int | None = config.get("n_threads") or None
        self.max_tokens: int = int(config.get("max_tokens") or 512)
        self.temperature: float = float(config.get("temperature") or 0.2)
        self._llm: Any | None = None
        self._lock = threading.Lock()
        self._load()

    # --- lifecycle -------------------------------------------------------

    def _load(self) -> None:
        if not self.model_path or not self.model_path.is_file():
            self.set_unavailable(
                "No local vision model is installed. MSN AI's chat model is "
                "text-only, so it cannot see images. To enable image "
                "understanding, place a small multimodal GGUF (for example a "
                "LLaVA/Qwen2-VL mmproj model) under model_storage\\vision and set "
                "MSN_AI_VISION_MODEL_PATH."
            )
            return
        if not self.mmproj_path or not self.mmproj_path.is_file():
            self.set_unavailable(
                "The vision model is present but its multimodal projector "
                "(mmproj) file is missing. Set MSN_AI_VISION_MMPROJ_PATH to the "
                "matching .mmproj file."
            )
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            self.set_unavailable(
                "llama-cpp-python is not installed, so no vision model can be "
                f"loaded ({exc})."
            )
            return
        handler = self._build_chat_handler()
        if handler is None:
            # Loading the weights without a projector would produce a model that
            # accepts images and answers from its text prior. Better to say so.
            self.set_unavailable(
                "No compatible multimodal chat handler was found in the "
                "installed llama-cpp-python, so the projector file cannot be "
                "attached. Upgrade llama-cpp-python to a build that ships the "
                "chat handler matching this model."
            )
            return
        try:
            with self._lock:
                self._llm = Llama(
                    model_path=str(self.model_path),
                    chat_handler=handler,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                    n_gpu_layers=0,
                )
            self.set_available()
            logger.info(
                "Vision model ready: %s (projector %s)",
                self.model_name,
                getattr(self.mmproj_path, "name", self.mmproj_path),
            )
        except Exception as exc:
            self._llm = None
            self.set_unavailable(f"The vision model failed to load: {exc}")

    def _build_chat_handler(self) -> Any | None:
        """Attach the multimodal projector to the model.

        llama-cpp-python needs a chat handler that knows how to load the
        ``.mmproj`` weights; without one the model is blind. The handler class
        depends on the model family, so the configured name is used as a hint
        and the remaining candidates are tried in order.
        """
        try:
            from llama_cpp import llama_chat_format as chat_format
        except ImportError:
            return None

        assert self.mmproj_path is not None
        mmproj = str(self.mmproj_path)
        model = self.model_name.lower()

        # Ordered candidates; the first importable class wins.
        candidates: list[str] = []
        if "qwen2-vl" in model or "qwen2vl" in model:
            candidates.append("Qwen2VLChatHandler")
        if "minicpm" in model:
            candidates.append("MiniCPMv26ChatHandler")
        if "llava" in model or "bakllava" in model:
            candidates.append("Llava15ChatHandler")
        candidates += ["Qwen2VLChatHandler", "Llava15ChatHandler", "MiniCPMv26ChatHandler"]

        for name in dict.fromkeys(candidates):
            handler_cls = getattr(chat_format, name, None)
            if handler_cls is None:
                continue
            try:
                return handler_cls(clip_model_path=mmproj, verbose=False)
            except TypeError:
                # Older signatures take no keyword arguments.
                try:
                    return handler_cls(mmproj)
                except Exception as exc:  # pragma: no cover - version dependent
                    logger.debug("Chat handler %s could not be built: %s", name, exc)
            except Exception as exc:  # pragma: no cover - version dependent
                logger.debug("Chat handler %s could not be built: %s", name, exc)
        return None

    def status(self) -> dict[str, Any]:
        data = super().status()
        data.update(
            {
                "model": self.model_name,
                "model_path": str(self.model_path) if self.model_path else None,
                "mmproj_path": str(self.mmproj_path) if self.mmproj_path else None,
                "loaded": self._llm is not None,
            }
        )
        return data

    # --- inference -------------------------------------------------------

    def describe(self, image_path: str | Path, prompt: str = VISION_PROMPT) -> VisionResult:
        if self._llm is None:
            raise RuntimeError(self.unavailable_reason)
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        with self._lock:
            try:
                response = self._llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as exc:
                raise RuntimeError(f"Vision inference failed: {exc}") from exc
        try:
            text = response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected vision response shape: {response!r}") from exc
        return VisionResult(text=text, model_name=self.model_name, engine=self.name)


def build_vision_provider(settings) -> CapabilityProvider:
    """Build the vision provider described by application settings."""
    model_path = getattr(settings, "vision_model_path", None)
    if not model_path:
        return LocalVisionProvider(
            {"model_name": "none", "model_path": None, "mmproj_path": None}
        )
    return LocalVisionProvider(
        {
            "model_name": getattr(settings, "vision_model_name", "none"),
            "model_path": model_path,
            "mmproj_path": getattr(settings, "vision_mmproj_path", None),
            "n_ctx": getattr(settings, "vision_n_ctx", 4096),
            "n_threads": settings.n_threads or None,
            "max_tokens": getattr(settings, "vision_max_tokens", 512),
        }
    )
