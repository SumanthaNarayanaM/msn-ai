"""Local persistent memory store.

A small JSON-file-backed store for long-term memory records (user
preferences, project information, important facts, story memory). The most
likely later evolution is to back this with an embedder + vector index (RAG);
the interface below already describes everything that will need one.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("msn_ai.memory.store")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Acceptable memory record types. Enforced so the store stays clean.
VALID_TYPES = {"fact", "preference", "project", "story"}


class MemoryRecord(BaseModel):
    """A single long-term memory item."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    type: str = Field(default="fact", description="fact|preference|project|story")
    content: str = Field(..., description="The remembered fact or preference")
    metadata: dict = Field(default_factory=dict)
    source_session_id: str = Field(default="", description="Session that created it")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @classmethod
    def validate_type(cls, type_: str) -> str:
        if type_ not in VALID_TYPES:
            raise ValueError(f"Unknown memory type '{type_}'. Valid: {sorted(VALID_TYPES)}")
        return type_


class LocalMemoryStore:
    """Thread-safe memory records persisted as a single JSON file."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._file = self._storage_dir / "memory.json" if self._storage_dir else None
        self._records: dict[str, MemoryRecord] = {}
        self._lock = threading.RLock()

        if self._storage_dir is not None:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # --- lifecycle -------------------------------------------------------

    def load(self) -> None:
        """(Re)load records from disk."""
        with self._lock:
            self._records.clear()
            self._load()

    def unload(self) -> None:
        with self._lock:
            self._records.clear()

    # --- writes ----------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        type: str = "fact",
        metadata: dict | None = None,
        source_session_id: str = "",
    ) -> MemoryRecord:
        type_ = MemoryRecord.validate_type(type)
        record = MemoryRecord(
            type=type_,
            content=content.strip(),
            metadata=metadata or {},
            source_session_id=source_session_id,
        )
        with self._lock:
            self._records[record.id] = record
            self._save()
        logger.debug("Stored memory %s (%s).", record.id, type_)
        return record

    def update(self, record_id: str, content: str) -> MemoryRecord | None:
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return None
            record.content = content.strip()
            record.updated_at = time.time()
            self._save()
            return record

    def delete(self, record_id: str) -> bool:
        with self._lock:
            removed = self._records.pop(record_id, None) is not None
            if removed:
                self._save()
            return removed

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._save()

    # --- reads -----------------------------------------------------------

    def all(self, type: str | None = None) -> list[MemoryRecord]:
        with self._lock:
            records = list(self._records.values())
        if type:
            records = [r for r in records if r.type == type]
        return sorted(records, key=lambda r: r.created_at)

    def get(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            return self._records.get(record_id)

    def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        """Keyword-overlap recall. Swap for embeddings + vector search later."""
        query = query.strip().lower()
        if not query:
            return []
        tokens = {t for t in re.split(r"\W+", query) if len(t) > 1}
        if not tokens:
            return []

        scored: list[tuple[int, float, MemoryRecord]] = []
        for record in self.all():
            haystack = f"{record.content} {' '.join(record.metadata.values())}".lower()
            text_tokens = {t for t in re.split(r"\W+", haystack) if len(t) > 1}
            overlap = len(tokens & text_tokens)
            if overlap > 0:
                scored.append((overlap, record.created_at, record))

        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [record for _, _, record in scored[:limit]]

    # --- persistence -----------------------------------------------------

    def _save(self) -> None:
        if self._file is None:
            return
        payload = {
            "version": 1,
            "records": [r.model_dump() for r in self._records.values()],
        }
        try:
            self._file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not persist memory store: %s", exc)

    def _load(self) -> None:
        if self._file is None or not self._file.exists():
            return
        try:
            payload = json.loads(self._file.read_text(encoding="utf-8"))
            for raw in payload.get("records", []):
                record = MemoryRecord.model_validate(raw)
                self._records[record.id] = record
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not load memory store: %s", exc)