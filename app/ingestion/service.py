"""Ingestion service: the object the API layer talks to.

Responsibilities:

* Accept a validated upload and persist it under an internal id.
* Run extraction in a background worker so ``POST /files/upload`` returns fast
  and the UI can show progress.
* Chunk, embed and index the result, always ending in a terminal state.
* Manage memory scope (session / project / personal) and deletion.

Every failure path is terminal and reported: the UI never shows a spinner that
cannot resolve.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from app.core.errors import ExtractionError, MSNError
from app.core.logging import get_logger
from app.ingestion.chunking import Chunk, chunk_blocks
from app.ingestion.extractors.base import ExtractionContext
from app.ingestion.knowledge import KnowledgeBase, SearchHit
from app.ingestion.models import (
    ExtractedBlock,
    ExtractionResult,
    FileRecord,
    FileStatus,
    new_id,
)
from app.ingestion.providers.ocr import build_ocr_provider
from app.ingestion.providers.speech import build_speech_provider
from app.ingestion.providers.ffmpeg import build_ffmpeg_provider
from app.ingestion.providers.vision import build_vision_provider
from app.ingestion.registry import registry as extractor_registry
from app.ingestion.security import (
    extension_source_kind,
    is_image_extension,
    is_video_extension,
    normalize_extension,
    resolve_media_type,
    sanitize_filename,
    stream_to_disk,
    verify_content_matches_extension,
)
from app.ingestion.store import FileStore

logger = get_logger("msn_ai.ingestion.service")

#: Only one extraction per file may run at a time.
_ACTIVE: set[str] = set()
_ACTIVE_LOCK = threading.Lock()

#: Valid memory scopes.
SCOPES = ("session", "project", "personal")


class IngestionService:
    """Coordinates upload, extraction, indexing and retrieval."""

    def __init__(self, settings, embedder, store: FileStore, knowledge: KnowledgeBase) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.knowledge = knowledge
        self._executor = ThreadPoolExecutor(
            max_workers=int(getattr(settings, "ingestion_workers", 2) or 2),
            thread_name_prefix="msn-ingest",
        )
        self._ocr = build_ocr_provider(settings)
        self._vision = build_vision_provider(settings)
        self._speech = build_speech_provider(settings)
        self._ffmpeg = build_ffmpeg_provider(settings)
        logger.info(
            "Ingestion capabilities -> ocr=%s vision=%s speech=%s ffmpeg=%s",
            self._ocr.available,
            self._vision.available,
            self._speech.available,
            self._ffmpeg.available,
        )

    # --- capabilities ----------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """Capability status for /health and the frontend.

        Keys are named after what the user gets, not after the tool that
        provides it: ``video_frames`` is the user-visible feature, while the
        underlying binary is still reported inside that entry.
        """
        ffmpeg = self._ffmpeg.status()
        return {
            "ocr": self._ocr.status(),
            "vision": self._vision.status(),
            "speech": self._speech.status(),
            "video_frames": ffmpeg,
            "embeddings": self.embedder.status(),
        }

    def supported_extensions(self) -> list[str]:
        return extractor_registry.supported_extensions()

    # --- upload ----------------------------------------------------------

    def save_upload(
        self,
        stream,
        filename: str,
        *,
        declared_mime: str | None = None,
        session_id: str = "",
        scope: str = "session",
        project: str = "",
        enqueue: bool = True,
    ) -> FileRecord:
        """Validate and store an upload, then queue processing.

        Runs the security checks *before* the bytes land on disk, and streams
        with a hard size ceiling so an oversized upload is aborted early.
        Pass ``enqueue=False`` to store the file without starting background
        work, e.g. when the caller intends to process it inline itself.
        """
        display_name = sanitize_filename(filename)
        ext, mime = resolve_media_type(display_name, declared_mime)
        if scope not in SCOPES:
            scope = "session"

        record = FileRecord(
            id=new_id(),
            filename=display_name,
            display_name=display_name,
            extension=ext,
            mime_type=mime,
            source_kind=extension_source_kind(ext),
            is_video=is_video_extension(ext),
            is_image=is_image_extension(ext),
            status=FileStatus.UPLOADED,
            scope=scope,
            session_id=session_id,
            project=project if scope == "project" else "",
        )
        record.stored_name = self.store.storage_path(record).name
        target = self.store.storage_path(record)
        limit = self.settings.upload_limit_for(record.is_video)

        written, sha256 = stream_to_disk(stream, open(target, "wb"), max_bytes=limit)
        record.size = written
        record.sha256 = sha256

        if written == 0:
            target.unlink(missing_ok=True)
            raise ExtractionError("The uploaded file is empty.")

        record.warnings = verify_content_matches_extension(ext, target)
        self.store.add(record)
        logger.info(
            "Stored upload %s (%s, %d bytes) as %s",
            display_name, ext, written, record.id,
        )
        if enqueue:
            self.enqueue(record.id)
        return record

    # --- processing ------------------------------------------------------

    def enqueue(self, file_id: str) -> bool:
        """Queue background extraction. False when already running or disabled."""
        if not getattr(self.settings, "ingestion_enabled", True):
            return False
        with _ACTIVE_LOCK:
            if file_id in _ACTIVE:
                return False
            _ACTIVE.add(file_id)
        self._executor.submit(self._run_guarded, file_id)
        return True

    def _run_guarded(self, file_id: str) -> None:
        try:
            self.process(file_id)
        except Exception as exc:  # pragma: no cover - last-resort guard
            logger.exception("Ingestion crashed for %s: %s", file_id, exc)
            try:
                record = self.store.get(file_id)
                record.status = FileStatus.FAILED
                record.error = f"Ingestion failed: {exc}"
                self.store.update(record)
            except Exception:  # pragma: no cover - defensive
                pass
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(file_id)

    def process(self, file_id: str) -> FileRecord:
        """Extract, chunk and index one file. Always leaves a terminal state."""
        record = self.store.get(file_id)
        if record.status is FileStatus.PROCESSING:
            return record

        record.status = FileStatus.PROCESSING
        record.progress = 0.02
        record.progress_stage = "preparing"
        record.error = ""
        self.store.update(record)

        path = self.store.storage_path(record)
        if not path.exists():
            record.status = FileStatus.FAILED
            record.error = "The stored file is missing from disk."
            self.store.update(record)
            return record

        def on_progress(stage: str, fraction: float) -> None:
            record.progress_stage = stage
            record.progress = max(0.02, min(float(fraction), 0.99))
            self.store.update(record)

        ctx = ExtractionContext(
            settings=self.settings,
            file_record=record,
            progress=on_progress,
            ocr=self._ocr,
            vision=self._vision,
            speech=self._speech,
            ffmpeg=self._ffmpeg,
        )

        try:
            on_progress("extracting", 0.1)
            result: ExtractionResult = extractor_registry.extract(path, ctx)
        except MSNError as exc:
            record.status = FileStatus.FAILED
            record.error = str(exc)
            record.progress_stage = "failed"
            self.store.update(record)
            return record
        except Exception as exc:
            logger.exception("Extraction failed for %s: %s", file_id, exc)
            record.status = FileStatus.FAILED
            record.error = f"Extraction failed: {exc}"
            record.progress_stage = "failed"
            self.store.update(record)
            return record

        on_progress("saving extracted text", 0.8)
        blocks = result.blocks or (
            [ExtractedBlock(text=result.text, order=0)] if result.text.strip() else []
        )
        try:
            self.store.extracted_path(record).write_text(
                result.text, encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not persist extracted text for %s: %s", file_id, exc)

        on_progress("chunking", 0.85)
        chunks: list[Chunk] = chunk_blocks(
            blocks,
            file_id=record.id,
            filename=record.display_name,
            chunk_size=int(self.settings.chunk_size),
            chunk_overlap=int(self.settings.chunk_overlap),
        )

        on_progress("embedding and indexing", 0.9)
        indexed = self.knowledge.index_chunks(
            record.id,
            chunks,
            scope=record.scope,
            session_id=record.session_id,
            project=record.project,
        )

        record.chunk_count = indexed
        record.block_count = len(blocks)
        record.char_count = len(result.text)
        record.document_metadata = result.document_metadata
        record.unavailable = result.unavailable
        record.artifacts = result.artifacts
        record.warnings = list(dict.fromkeys(record.warnings + result.warnings))
        record.status = FileStatus.READY
        record.progress = 1.0
        record.progress_stage = "ready"
        record.processed_at = time.time()
        record.error = ""
        self.store.update(record)
        logger.info(
            "Ready %s: %d blocks, %d chunks, %d chars",
            record.display_name, len(blocks), indexed, record.char_count,
        )
        return record

    def reanalyze(self, file_id: str) -> FileRecord:
        """Re-run extraction (after a capability was installed, for example)."""
        record = self.store.get(file_id)
        with _ACTIVE_LOCK:
            if file_id in _ACTIVE:
                return record
            _ACTIVE.add(file_id)
        try:
            self.store.update(record)
            return self.process(file_id)
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(file_id)

    # --- memory scope ----------------------------------------------------

    def remember(self, file_id: str, scope: str = "project", project: str = "") -> FileRecord:
        """Promote a file's knowledge to a broader memory scope.

        If the file was forgotten earlier, its extracted text is still on disk,
        so it is re-chunked and re-indexed here; otherwise Remember would only
        flip a flag and the content would stay unsearchable.
        """
        if scope not in ("project", "personal"):
            raise ExtractionError(
                f"Invalid memory scope '{scope}'. Use 'project' or 'personal'."
            )
        record = self.store.get(file_id)
        if record.status is not FileStatus.READY:
            raise ExtractionError(
                f"'{record.display_name}' is not ready yet, so there is nothing to remember."
            )
        record.scope = scope
        record.remembered = True
        record.project = project if scope == "project" else ""

        if not self.knowledge.chunk_count_for(record.id):
            reindexed = self._reindex_from_text(record)
            if reindexed == 0:
                record.remembered = False
                record.scope = "session"
                self.store.update(record)
                raise ExtractionError(
                    f"'{record.display_name}' has no extracted text to remember. "
                    "Re-analyse the file first."
                )
            record.chunk_count = reindexed
        else:
            self.knowledge.set_scope(record.id, scope, project=record.project)
        self.store.update(record)
        logger.info("Remembered %s in scope '%s'", record.display_name, scope)
        return record

    def _reindex_from_text(self, record: FileRecord) -> int:
        """Re-chunk and re-index a file from its stored extracted text."""
        text_path = self.store.extracted_path(record)
        if not text_path.exists():
            return 0
        try:
            text = text_path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - defensive
            return 0
        if not text.strip():
            return 0
        blocks = [ExtractedBlock(text=text, order=0)]
        chunks = chunk_blocks(
            blocks,
            file_id=record.id,
            filename=record.display_name,
            chunk_size=int(self.settings.chunk_size),
            chunk_overlap=int(self.settings.chunk_overlap),
        )
        return self.knowledge.index_chunks(
            record.id,
            chunks,
            scope=record.scope,
            session_id=record.session_id,
            project=record.project,
        )

    def forget(self, file_id: str) -> FileRecord:
        """Remove indexed knowledge, keeping the original file on disk."""
        record = self.store.get(file_id)
        removed = self.knowledge.remove_file(file_id)
        record.scope = "session"
        record.remembered = False
        record.project = ""
        record.chunk_count = 0
        record.warnings = list(
            dict.fromkeys(
                record.warnings
                + [
                    f"Forgotten: {removed} indexed chunk(s) removed. The original "
                    "file was kept on disk."
                ]
            )
        )
        self.store.update(record)
        logger.info("Forgot %s (%d chunks removed)", record.display_name, removed)
        return record

    def delete(self, file_id: str, *, remove_original: bool = True) -> FileRecord:
        """Delete the record, its knowledge and (by default) the stored bytes."""
        self.knowledge.remove_file(file_id)
        return self.store.delete(file_id, remove_original=remove_original)

    # --- retrieval -------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        scope: str = "session",
        session_id: str = "",
        session_ids: Optional[list[str]] = None,
        project: str = "",
        top_k: int | None = None,
        file_ids: Optional[list[str]] = None,
    ) -> list[SearchHit]:
        return self.knowledge.search(
            query,
            scope=scope,
            session_id=session_id,
            session_ids=session_ids,
            project=project,
            top_k=top_k or int(self.settings.retrieval_top_k),
            min_score=float(self.settings.retrieval_min_score),
            file_ids=file_ids,
        )

    def build_context(
        self,
        query: str,
        *,
        scope: str = "session",
        session_id: str = "",
        session_ids: Optional[list[str]] = None,
        project: str = "",
        top_k: int | None = None,
        file_ids: Optional[list[str]] = None,
    ) -> tuple[str, list[SearchHit]]:
        """Return a prompt-ready context block plus the hits that produced it.

        Explicitly referenced files are searched first so a question about a
        just-attached file always wins over older background knowledge.
        """
        hits = self.search(
            query, scope=scope, session_id=session_id, session_ids=session_ids,
            project=project, top_k=top_k, file_ids=file_ids,
        )
        if not hits:
            return "", []

        max_chars = int(self.settings.max_context_chars)
        lines: list[str] = []
        used = 0
        used_hits: list[SearchHit] = []
        for hit in hits:
            snippet = hit.chunk.text.strip()
            entry = f"[{hit.source_label()}]\n{snippet}"
            if used + len(entry) > max_chars and used_hits:
                break
            lines.append(entry)
            used += len(entry) + 2
            used_hits.append(hit)
        return "\n\n".join(lines), used_hits

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.knowledge.close()
