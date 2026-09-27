"""Optional capability providers used during ingestion.

Each factory is cheap: it inspects settings, checks for a binary or an import,
and returns an object that knows whether it can actually work. Nothing is
downloaded or installed here.
"""

from __future__ import annotations

from app.ingestion.providers.base import (
    CapabilityProvider,
    OcrResult,
    TranscriptSegment,
    VisionResult,
    resolve_binary,
)
from app.ingestion.providers.ffmpeg import FFmpegProvider, build_ffmpeg_provider
from app.ingestion.providers.ocr import (
    RapidOcrProvider,
    TesseractOcrProvider,
    UnavailableOcrProvider,
    build_ocr_provider,
)
from app.ingestion.providers.speech import (
    FasterWhisperSpeechProvider,
    OpenAiWhisperSpeechProvider,
    UnavailableSpeechProvider,
    build_speech_provider,
)
from app.ingestion.providers.vision import LocalVisionProvider, build_vision_provider

__all__ = [
    "CapabilityProvider",
    "OcrResult",
    "TranscriptSegment",
    "VisionResult",
    "resolve_binary",
    "FFmpegProvider",
    "RapidOcrProvider",
    "TesseractOcrProvider",
    "UnavailableOcrProvider",
    "FasterWhisperSpeechProvider",
    "OpenAiWhisperSpeechProvider",
    "UnavailableSpeechProvider",
    "LocalVisionProvider",
    "build_ocr_provider",
    "build_speech_provider",
    "build_vision_provider",
    "build_ffmpeg_provider",
]
