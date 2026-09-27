"""Image extraction: metadata, thumbnail, optional OCR and optional vision.

MSN AI never pretends a text-only model can see an image. If a local vision
model or OCR engine is not installed, the extraction completes with the real
image facts (dimensions, format, EXIF) and an explicit ``unavailable`` note that
the UI surfaces verbatim.
"""

from __future__ import annotations

import base64
from pathlib import Path

from app.core.logging import get_logger
from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.models import ExtractedBlock, ExtractionResult, SourceKind

logger = get_logger("msn_ai.ingestion.image")

#: Longest edge of the generated preview thumbnail.
THUMBNAIL_MAX_EDGE = 512
#: Hard ceiling on bytes handed to a vision model.
MAX_VISION_BYTES = 6 * 1024 * 1024

#: EXIF tags worth keeping as searchable text.
_INTERESTING_EXIF = {
    271: "Make",
    272: "Model",
    306: "DateTime",
    270: "ImageDescription",
    315: "Artist",
    316: "Copyright",
}


class ImageExtractor(Extractor):
    """Read image metadata, make a thumbnail, and try OCR/vision providers."""

    name = "image"
    extensions = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("opening image", 0.2)
        try:
            from PIL import Image, ImageOps  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            return ExtractionResult(
                text="",
                warnings=["Image support requires Pillow (pip install Pillow)."],
                unavailable={"image": str(exc)},
            )

        warnings: list[str] = []
        unavailable: dict[str, str] = {}
        blocks: list[ExtractedBlock] = []
        artifacts: dict = {}
        meta: dict = {"format": path.suffix.lstrip(".").lower()}

        image = None
        try:
            image = Image.open(path)
            image.load()
        except Exception as exc:
            return ExtractionResult(
                text="",
                document_metadata=meta,
                warnings=[f"The image could not be decoded: {exc}"],
                unavailable={"image": str(exc)},
            )

        try:
            width, height = image.size
            meta.update(
                {
                    "width": width,
                    "height": height,
                    "megapixels": round((width * height) / 1_000_000, 2),
                    "mode": image.mode,
                    "format": (image.format or meta["format"]).lower(),
                    "is_animated": bool(getattr(image, "is_animated", False)),
                    "frames": getattr(image, "n_frames", 1),
                }
            )

            exif_text: list[str] = []
            try:
                exif = image.getexif()
                for tag, label in _INTERESTING_EXIF.items():
                    value = exif.get(tag)
                    if value:
                        exif_text.append(f"{label}: {value}")
                if exif_text:
                    meta["exif"] = dict(
                        line.split(": ", 1) for line in exif_text if ": " in line
                    )
            except Exception:  # pragma: no cover - defensive
                pass

            # Orientation-aware, size-bounded thumbnail for the UI.
            try:
                ctx.progress("making thumbnail", 0.35)
                thumb = ImageOps.exif_transpose(image).convert("RGB")
                thumb.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE), Image.LANCZOS)
                out_dir = ctx.settings.thumbnails_path()
                out_dir.mkdir(parents=True, exist_ok=True)
                thumb_path = out_dir / f"{ctx.file_record.id}.jpg"
                thumb.save(thumb_path, format="JPEG", quality=82, optimize=True)
                artifacts["thumbnail"] = thumb_path.name
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Thumbnail generation failed: {exc}")

            description_parts: list[str] = [
                f"Image: {meta.get('format', 'image').upper()}, "
                f"{width}x{height} pixels ({meta.get('megapixels')} MP), mode {image.mode}."
            ]
            if exif_text:
                description_parts.append("EXIF: " + "; ".join(exif_text))
            if meta.get("is_animated"):
                description_parts.append(f"Animated image with {meta.get('frames')} frames.")

            # --- OCR (real text recognition, when installed) ---
            ctx.progress("running OCR", 0.5)
            ocr_text = ""
            if ctx.ocr is not None and ctx.ocr.available:
                try:
                    ocr = ctx.ocr.image_to_text(str(path), source=ctx.file_record.filename)
                    ocr_text = (ocr.text or "").strip()
                    if ocr.confidence is not None:
                        meta["ocr_confidence"] = round(ocr.confidence, 3)
                except Exception as exc:
                    warnings.append(f"OCR failed: {exc}")
            else:
                reason = (
                    ctx.ocr.unavailable_reason
                    if ctx.ocr is not None
                    else "No OCR provider is configured."
                )
                unavailable["ocr"] = reason
                warnings.append(
                    "Local OCR is not installed, so no text was read from this image."
                )

            if ocr_text:
                meta["ocr_char_count"] = len(ocr_text)
                blocks.append(
                    ExtractedBlock(
                        text=self.block_text_limit(ctx, ocr_text),
                        kind=SourceKind.OCR,
                        locator="image text (OCR)",
                        order=0,
                        metadata={"provider": ctx.ocr.name if ctx.ocr else "ocr"},
                    )
                )

            # --- Vision (real local VLM only; never faked) ---
            ctx.progress("running vision analysis", 0.7)
            if ctx.vision is not None and ctx.vision.available:
                try:
                    described = ctx.vision.describe(str(path))
                    vision_text = (described.text or "").strip()
                    if vision_text:
                        meta["vision_model"] = ctx.vision.model_name
                        blocks.insert(
                            0,
                            ExtractedBlock(
                                text=self.block_text_limit(ctx, vision_text),
                                kind=SourceKind.IMAGE,
                                locator="image description (vision model)",
                                order=0,
                                metadata={"provider": ctx.vision.name},
                            ),
                        )
                except Exception as exc:
                    warnings.append(f"Vision analysis failed: {exc}")
            else:
                unavailable["vision"] = (
                    ctx.vision.unavailable_reason
                    if ctx.vision is not None
                    else "No local vision model is installed."
                )
                meta["vision_supported"] = False
                if ocr_text:
                    warnings.append(
                        "Only the text layer was read; full visual understanding "
                        "requires a local vision model, which is not installed."
                    )
                else:
                    warnings.append(
                        "MSN AI cannot visually understand this image: no local "
                        "vision model is installed."
                    )

            ctx.progress("assembling image blocks", 0.9)
            summary_block = ExtractedBlock(
                text=self.block_text_limit(ctx, "\n".join(description_parts)),
                kind=SourceKind.IMAGE,
                locator="image properties",
                order=len(blocks),
                metadata={"width": width, "height": height},
            )
            blocks.append(summary_block)

            full_text = "\n\n".join(b.text for b in blocks)
            return ExtractionResult(
                text=self.block_text_limit(ctx, full_text),
                blocks=blocks,
                document_metadata=meta,
                unavailable=unavailable,
                warnings=warnings,
                artifacts=artifacts,
            )
        finally:
            if image is not None:
                try:
                    image.close()
                except Exception:  # pragma: no cover - defensive
                    pass
