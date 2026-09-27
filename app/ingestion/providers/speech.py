"""Local speech-to-text providers (faster-whisper, openai-whisper).

Whisper models are *not* downloaded automatically: the provider reports
unavailable until the package and a local model are both present, and the
ingestion pipeline then says so plainly instead of returning an empty
transcript.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.ingestion.providers.base import CapabilityProvider, TranscriptSegment

logger = get_logger("msn_ai.ingestion.speech")


class FasterWhisperSpeechProvider(CapabilityProvider):
    """Transcribe media with ``faster-whisper`` (CTranslate2 int8 on CPU)."""

    name = "faster-whisper"

    def __init__(self, model_name: str = "base", compute_type: str = "int8") -> None:
        super().__init__()
        self.model_name = model_name or "base"
        self.compute_type = compute_type or "int8"
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            self.set_unavailable(
                "faster-whisper is not installed (pip install faster-whisper). "
                "Speech-to-text for video and audio is therefore unavailable."
            )
            return
        # ``local_files_only`` is what actually enforces the no-auto-download
        # rule: without it, constructing the model silently fetches weights from
        # HuggingFace the first time it is used.
        try:
            with self._lock:
                try:
                    self._model = WhisperModel(
                        self.model_name,
                        device="cpu",
                        compute_type=self.compute_type,
                        local_files_only=True,
                    )
                except TypeError:
                    # Older builds have no local_files_only switch; refuse rather
                    # than risk a download.
                    self._model = None
                    raise
        except Exception as exc:
            self._model = None
            self.set_unavailable(
                f"The speech model '{self.model_name}' is not available locally, so "
                f"MSN AI did not load it ({exc}). MSN AI never downloads models on "
                "your behalf. Pre-download it once with: python -c \"from "
                "faster_whisper import WhisperModel; WhisperModel('"
                f"{self.model_name}', device='cpu', compute_type='{self.compute_type}')"
                "\" -- then restart MSN AI."
            )
            return
        self.set_available()
        logger.info(
            "Speech model ready: faster-whisper %s (%s)", self.model_name, self.compute_type
        )

    def status(self) -> dict[str, Any]:
        data = super().status()
        data.update({"model": self.model_name, "compute_type": self.compute_type})
        return data

    def transcribe(self, media_path: str | Path, language: str | None = None) -> list[TranscriptSegment]:
        if self._model is None:
            raise RuntimeError(self.unavailable_reason)
        segments, _info = self._model.transcribe(str(media_path), language=language, vad_filter=True)
        return [
            TranscriptSegment(start=float(s.start), end=float(s.end), text=(s.text or "").strip())
            for s in segments
        ]


class OpenAiWhisperSpeechProvider(CapabilityProvider):
    """Transcribe media with the reference ``openai-whisper`` package."""

    name = "whisper"

    def __init__(self, model_name: str = "base") -> None:
        super().__init__()
        self.model_name = model_name or "base"
        self._model: Any | None = None
        # ``whisper.load_model`` downloads the weights when they are missing, so
        # presence is checked first and nothing is fetched.
        if Path(model_name).is_file():
            self._weights = Path(model_name)
        else:
            self._weights = Path.home() / ".cache" / "whisper" / f"{self.model_name}.pt"
        if not self._weights.is_file():
            self.set_unavailable(
                f"The Whisper model '{self.model_name}' is not present at "
                f"{self._weights}, so MSN AI did not load it. MSN AI never "
                "downloads models on your behalf. Pre-download it once with: "
                f"python -c \"import whisper; whisper.load_model('{self.model_name}')\""
                " -- then restart MSN AI."
            )
            return
        try:
            import whisper  # type: ignore

            self._model = whisper.load_model(str(self._weights))
            self.set_available()
        except Exception as exc:
            self._model = None
            self.set_unavailable(
                f"openai-whisper is not usable ({exc}). "
                "Install it with: pip install openai-whisper"
            )

    def status(self) -> dict[str, Any]:
        data = super().status()
        data.update({"model": self.model_name, "weights": str(self._weights)})
        return data

    def transcribe(self, media_path: str | Path, language: str | None = None) -> list[TranscriptSegment]:
        if self._model is None:
            raise RuntimeError(self.unavailable_reason)
        result = self._model.transcribe(str(media_path), language=language)
        return [
            TranscriptSegment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=(s.get("text") or "").strip())
            for s in result.get("segments", [])
        ]


class UnavailableSpeechProvider(CapabilityProvider):
    """Placeholder when speech-to-text is disabled in settings."""

    name = "none"

    def __init__(self, reason: str = "Speech-to-text is disabled in settings.") -> None:
        super().__init__()
        self.set_unavailable(reason)

    def transcribe(self, media_path: str | Path, language: str | None = None) -> list[TranscriptSegment]:
        raise RuntimeError(self.unavailable_reason)


def build_speech_provider(settings) -> CapabilityProvider:
    """Build the speech provider named by ``settings.speech_provider``."""
    provider_name = (getattr(settings, "speech_provider", "") or "").lower().strip()
    if not provider_name or provider_name in {"none", "disabled", "off"}:
        return UnavailableSpeechProvider()
    if provider_name == "faster-whisper":
        return FasterWhisperSpeechProvider(
            model_name=getattr(settings, "speech_model", "base"),
            compute_type=getattr(settings, "speech_compute_type", "int8"),
        )
    if provider_name == "whisper":
        return OpenAiWhisperSpeechProvider(model_name=getattr(settings, "speech_model", "base"))
    return UnavailableSpeechProvider(
        f"Unknown speech provider '{provider_name}'. Use 'faster-whisper', 'whisper' or 'none'."
    )
