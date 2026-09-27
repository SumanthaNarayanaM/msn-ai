"""Local OCR providers: Tesseract binary and RapidOCR (onnxruntime).

Both are optional. When neither is installed the provider stays unavailable and
its reason is surfaced to the user, so the UI can say "install Tesseract" instead
of silently returning no text.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.ingestion.providers.base import CapabilityProvider, OcrResult, resolve_binary

logger = get_logger("msn_ai.ingestion.ocr")


class TesseractOcrProvider(CapabilityProvider):
    """OCR via the locally installed ``tesseract`` binary."""

    name = "tesseract"

    def __init__(self, command: str = "tesseract", languages: str = "eng") -> None:
        super().__init__()
        self.command = command or "tesseract"
        self.languages = languages or "eng"
        self._binary = resolve_binary(self.command)
        if self._binary:
            self.set_available()
        else:
            self.set_unavailable(
                f"The Tesseract binary ('{self.command}') is not installed. "
                "Install Tesseract OCR to read text from images and video frames."
            )

    def image_to_text(self, image_path: str | Path, source: str = "") -> OcrResult:
        if not self._available:
            raise RuntimeError(self.unavailable_reason)
        with tempfile.TemporaryDirectory(prefix="msn-ocr-") as tmp:
            out_base = Path(tmp) / "out"
            cmd = [
                self._binary,
                str(image_path),
                str(out_base),
                "-l",
                self.languages,
            ]
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, check=False
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"tesseract failed ({completed.returncode}): "
                    f"{(completed.stderr or '').strip()[:300]}"
                )
            text_file = out_base.with_suffix(".txt")
            text = text_file.read_text(encoding="utf-8", errors="replace") if text_file.exists() else ""
            return OcrResult(text=self._clean(text), confidence=None, engine=self.name)

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("\x0c", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class RapidOcrProvider(CapabilityProvider):
    """OCR via the pure-Python ``rapidocr_onnxruntime`` package.

    Preferred on machines without a Tesseract install because it needs no
    external binary, but it still requires an explicit pip install.
    """

    name = "rapidocr"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._engine = None
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._engine = RapidOCR()
            self.set_available()
        except Exception as exc:
            logger.debug("RapidOCR unavailable: %s", exc)
            self.set_unavailable(
                "RapidOCR is not installed. Run 'pip install rapidocr-onnxruntime' "
                "for OCR without an external binary."
            )

    def image_to_text(self, image_path: str | Path, source: str = "") -> OcrResult:
        if self._engine is None:
            raise RuntimeError(self.unavailable_reason)
        result, _ = self._engine(str(image_path))
        if not result:
            return OcrResult(text="", confidence=None, engine=self.name)
        lines: list[str] = []
        scores: list[float] = []
        for item in result:
            if len(item) >= 3:
                lines.append(str(item[1]))
                try:
                    scores.append(float(item[2]))
                except (TypeError, ValueError):
                    pass
            elif len(item) >= 2:
                lines.append(str(item[1]))
        mean = sum(scores) / len(scores) if scores else None
        return OcrResult(text="\n".join(lines).strip(), confidence=mean, engine=self.name)


class UnavailableOcrProvider(CapabilityProvider):
    """Placeholder used when OCR is switched off in settings."""

    name = "none"

    def __init__(self, reason: str = "OCR is disabled (MSN_AI_OCR_ENABLED=false).") -> None:
        super().__init__()
        self.set_unavailable(reason)

    def image_to_text(self, image_path: str | Path, source: str = "") -> OcrResult:
        raise RuntimeError(self.unavailable_reason)


def build_ocr_provider(settings) -> CapabilityProvider:
    """Pick an OCR provider from settings, preferring the binary-free option."""
    if not getattr(settings, "ocr_enabled", True):
        return UnavailableOcrProvider()

    provider_name = (getattr(settings, "ocr_provider", "tesseract") or "tesseract").lower()
    if provider_name == "rapidocr":
        return RapidOcrProvider()
    if provider_name == "tesseract":
        tesseract = TesseractOcrProvider(
            command=getattr(settings, "tesseract_cmd", "tesseract"),
            languages=getattr(settings, "ocr_languages", "eng"),
        )
        if tesseract.available:
            return tesseract
        return tesseract
    return UnavailableOcrProvider(
        f"Unknown OCR provider '{provider_name}'. Use 'tesseract' or 'rapidocr'."
    )
