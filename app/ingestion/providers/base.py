"""Capability providers used by extractors (OCR, vision, speech, ffmpeg).

Every provider here follows the same contract: it either *works* or it declares
itself ``available = False`` with a human-readable
:attr:`unavailable_reason`. Nothing in this package ever returns invented
content, which is what lets the ingestion pipeline degrade honestly instead of
pretending it analysed an image or heard a video.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class OcrResult:
    """Text recognised in an image, plus optional mean confidence."""

    text: str = ""
    confidence: Optional[float] = None
    engine: str = ""


@dataclass
class VisionResult:
    """A description produced by a real local vision-language model."""

    text: str = ""
    model_name: str = ""
    engine: str = ""


@dataclass
class TranscriptSegment:
    """One timestamped piece of recognised speech."""

    start: float = 0.0
    end: float = 0.0
    text: str = ""


class CapabilityProvider(ABC):
    """Base class: availability plus a status payload for ``/health``."""

    name: str = "capability"

    def __init__(self) -> None:
        self._available = False
        self._unavailable_reason = "not initialised"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def set_available(self, reason_if_unavailable: str = "") -> None:
        self._available = True
        self._unavailable_reason = ""

    def set_unavailable(self, reason: str) -> None:
        self._available = False
        self._unavailable_reason = reason or "unavailable"

    def status(self) -> dict[str, Any]:
        """Describe this capability for diagnostics and the UI."""
        return {
            "provider": self.name,
            "available": self._available,
            "reason": self._unavailable_reason,
        }


def resolve_binary(command: str) -> Optional[str]:
    """Locate an external executable, or None when it is not installed.

    Uses :func:`shutil.which` so an explicit absolute path in settings works too.
    """
    if not command:
        return None
    return shutil.which(command)
