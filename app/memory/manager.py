"""Memory manager: the bridge between conversation and long-term storage.

Conversation data lives in sessions (per-turn history). The memory manager
decides which facts are worth promoting into durable long-term memory and
offers recall so future conversations can reference what the user already told
us.

Nothing is stored automatically: callers (an endpoint or, later, an agent)
explicitly call :meth:`remember`. This avoids polluting permanent memory with
every random message.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.logging import get_logger
from app.memory.store import LocalMemoryStore, MemoryRecord

logger = get_logger("msn_ai.memory.manager")

#: Phrases users typically use when stating a fact about themselves.
_STATEMENT_HINTS = (
    "my name is",
    "i am",
    "i'm",
    "i like",
    "i love",
    "i prefer",
    "my favorite",
    "my favourite",
    "i live in",
    "i work",
    "my project",
)


class MemoryManager:
    """Thin orchestration layer over the local memory store."""

    def __init__(self, store: LocalMemoryStore | None = None, storage_dir: Path | None = None) -> None:
        self.store = store or LocalMemoryStore(storage_dir=storage_dir)

    def remember(
        self,
        content: str,
        *,
        type: str = "fact",
        metadata: dict | None = None,
        source_session_id: str = "",
    ) -> MemoryRecord:
        record = self.store.add(
            content,
            type=type,
            metadata=metadata,
            source_session_id=source_session_id,
        )
        logger.info("Remembered (%s): %s", record.type, record.content[:80])
        return record

    def recall(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        return self.store.search(query, limit=limit)

    def recall_for_context(self, query: str, *, limit: int = 3) -> str:
        """Build a short 'what we already know' block for the system prompt."""
        matches = self.recall(query, limit=limit)
        if not matches:
            return ""
        lines = [f"- ({m.type}) {m.content}" for m in matches]
        return "Known user context:\n" + "\n".join(lines)

    def all(self, type: str | None = None) -> list[MemoryRecord]:
        return self.store.all(type=type)

    def delete(self, record_id: str) -> bool:
        return self.store.delete(record_id)

    def clear(self) -> None:
        self.store.clear()

    def extract_fact(self, message: str) -> str | None:
        """Heuristic: does a message state a personal fact worth remembering?

        Returns the normalized fact (lowercased statement) or None. Deliberately
        conservative so trivial chatter is not stored permanently.
        """
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        if not normalized:
            return None
        for hint in _STATEMENT_HINTS:
            if normalized.startswith(hint):
                return normalized
        return None