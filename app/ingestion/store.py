"""Persistent file metadata store.

File records are the source of truth for the upload lifecycle. They survive
restarts (one JSON document per file, written atomically), so an upload that was
still ``processing`` when the server stopped is recovered as ``failed`` on the
next start rather than being stuck in a spinner forever.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from app.core.errors import FileNotFoundError_
from app.core.logging import get_logger
from app.ingestion.models import FileRecord, FileStatus, storage_path_for

logger = get_logger("msn_ai.ingestion.store")


class FileStore:
    """Thread-safe JSON-backed persistence for :class:`FileRecord` objects."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._dir = settings.files_metadata_path()
        self._uploads = settings.uploads_path()
        self._extracted = settings.extracted_path()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._uploads.mkdir(parents=True, exist_ok=True)
        self._extracted.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, FileRecord] = {}
        self._load_all()

    # --- paths -----------------------------------------------------------

    def _record_path(self, file_id: str) -> Path:
        return self._dir / f"{file_id}.json"

    def uploads_dir(self) -> Path:
        return self._uploads

    def extracted_dir(self) -> Path:
        return self._extracted

    def storage_path(self, record: FileRecord) -> Path:
        """Where the original bytes live (always id-keyed)."""
        if not record.stored_name:
            record.stored_name = storage_path_for(self._uploads, record.id, record.filename).name
        return self._uploads / record.stored_name

    def extracted_path(self, record: FileRecord) -> Path:
        """Where the extracted text for this file lives."""
        return self._extracted / f"{record.id}.txt"

    # --- persistence -----------------------------------------------------

    def _load_all(self) -> None:
        with self._lock:
            self._cache.clear()
            for path in self._dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    record = FileRecord.model_validate(data)
                except Exception as exc:
                    logger.warning("Skipping unreadable file record %s: %s", path.name, exc)
                    continue
                if record.status in (FileStatus.UPLOADED, FileStatus.PROCESSING):
                    # A crash mid-processing leaves no worker behind; be honest
                    # about it instead of showing a permanent spinner.
                    record.status = FileStatus.FAILED
                    record.error = (
                        record.error
                        or "Processing was interrupted when MSN AI last stopped."
                    )
                    record.progress = 0.0
                    record.progress_stage = ""
                    self._write(record)
                self._cache[record.id] = record
            logger.info("Loaded %d file record(s) from %s", len(self._cache), self._dir)

    def _write(self, record: FileRecord) -> None:
        """Atomically write one record so a crash cannot corrupt it."""
        target = self._record_path(record.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump(mode="json")
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(target.parent), delete=False, suffix=".tmp"
        )
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, target)

    # --- CRUD ------------------------------------------------------------

    def add(self, record: FileRecord) -> FileRecord:
        with self._lock:
            self._cache[record.id] = record
            self._write(record)
            return record

    def get(self, file_id: str) -> FileRecord:
        with self._lock:
            record = self._cache.get(file_id)
        if record is None:
            raise FileNotFoundError_(f"No file with id '{file_id}'.")
        return record

    def find(self, file_id: str) -> FileRecord | None:
        with self._lock:
            return self._cache.get(file_id)

    def update(self, record: FileRecord) -> FileRecord:
        record.touch()
        with self._lock:
            self._cache[record.id] = record
            self._write(record)
        return record

    def list(
        self,
        *,
        session_id: str | None = None,
        scope: str | None = None,
        statuses: Iterable[FileStatus] | None = None,
        limit: int = 200,
    ) -> list[FileRecord]:
        """Newest-first list, optionally filtered."""
        with self._lock:
            records = list(self._cache.values())
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        if scope:
            records = [r for r in records if r.scope == scope]
        if statuses:
            allowed = set(statuses)
            records = [r for r in records if r.status in allowed]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[: max(int(limit), 0)]

    def delete(self, file_id: str, *, remove_original: bool = True) -> FileRecord:
        """Delete the record and, by default, its stored bytes and artifacts."""
        with self._lock:
            record = self.get(file_id)
            if remove_original:
                self._remove_artifact_paths(record)
            target = self._record_path(file_id)
            self._cache.pop(file_id, None)
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not remove metadata for %s: %s", file_id, exc)
        return record

    def _remove_artifact_paths(self, record: FileRecord) -> None:
        """Best-effort cleanup of an original file, extraction and artifacts."""
        settings = self._settings
        candidates = [
            self.storage_path(record),
            self.extracted_path(record),
        ]
        thumbnail = record.artifacts.get("thumbnail")
        if thumbnail:
            candidates.append(settings.thumbnails_path() / str(thumbnail))
        frames_dir = record.artifacts.get("video_frames_dir")
        if frames_dir:
            candidates.append(settings.video_frames_path() / str(frames_dir))
        for path in candidates:
            try:
                if path.is_dir():
                    import shutil

                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Could not remove %s: %s", path, exc)

    def find_by_hash(self, sha256: str) -> FileRecord | None:
        """Reuse detection: same bytes already uploaded."""
        if not sha256:
            return None
        with self._lock:
            for record in self._cache.values():
                if record.sha256 == sha256:
                    return record
        return None

    def counts(self) -> dict[str, Any]:
        """Summary for the UI header and /health."""
        with self._lock:
            records = list(self._cache.values())
        by_status: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        total_bytes = 0
        for record in records:
            by_status[record.status.value] = by_status.get(record.status.value, 0) + 1
            by_kind[record.source_kind] = by_kind.get(record.source_kind, 0) + 1
            total_bytes += record.size
        return {
            "total": len(records),
            "by_status": by_status,
            "by_kind": by_kind,
            "total_bytes": total_bytes,
        }
