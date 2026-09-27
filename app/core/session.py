"""Conversation session manager.

Keeps per-session message history in memory and can persist conversations
to disk as JSON (UTF-8) under the configured conversations directory.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.errors import SessionNotFoundError
from app.core.logging import get_logger

logger = get_logger("msn_ai.session")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
MAX_SESSIONS = 1000


class ChatMessage(BaseModel):
    role: str = Field(description="One of system, user, assistant")
    content: str = Field(description="Message text")


_MAX_TITLE_CHARS = 60


def _derive_title(session: "Session") -> str:
    """Derive a conversation title from the first non-empty user message."""
    for message in session.messages:
        if message.role == "user":
            title = re.sub(r"\s+", " ", message.content).strip()
            if title:
                pivot = title[: _MAX_TITLE_CHARS]
                return pivot if len(title) <= _MAX_TITLE_CHARS else pivot.rstrip() + "…"
    return "New conversation"


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: list[ChatMessage] = Field(default_factory=list)

    def add_message(self, role: str, content: str) -> ChatMessage:
        msg = ChatMessage(role=role, content=content)
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg


class SessionManager:
    """Thread-safe in-memory session store with optional JSON persistence."""

    def __init__(self, storage_dir: Path | None = None, persist: bool = False) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._storage_dir = storage_dir
        self._persist = persist
        self.started_at: float = time.time()

        if self._storage_dir is not None:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    # --- lifecycle ---

    def create(self) -> Session:
        with self._lock:
            self._evict_if_needed()
            session = Session()
            self._sessions[session.id] = session
            self._save(session)
            logger.debug("Created session %s", session.id)
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                # Try loading from disk as a fallback.
                session = self._load(session_id)
                if session is None:
                    raise SessionNotFoundError(
                        f"Session '{session_id}' does not exist."
                    )
                self._sessions[session_id] = session
            return session

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            try:
                return self.get(session_id)
            except SessionNotFoundError:
                logger.info("Unknown session %s; creating a new one.", session_id)
        return self.create()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            if self._storage_dir is not None:
                path = self._path_for(session_id)
                if path.exists():
                    path.unlink()

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def summaries(self) -> list[dict]:
        """Return lightweight, time-ordered metadata for every session.

        Sessions persisted on disk (but not yet loaded) are loaded lazily so
        conversation history survives server restarts. Backwards compatible
        with :meth:`list_ids`.
        """
        with self._lock:
            if self._storage_dir is not None:
                for path in sorted(self._storage_dir.glob("*.json")):
                    sid = path.stem
                    if sid in self._sessions:
                        continue
                    session = self._load(sid)
                    if session is not None:
                        self._sessions[sid] = session

            rows: list[dict] = []
            for sid, session in self._sessions.items():
                rows.append(
                    {
                        "id": sid,
                        "title": _derive_title(session),
                        "created_at": session.created_at,
                        "updated_at": session.updated_at,
                        "message_count": len(session.messages),
                    }
                )
            rows.sort(key=lambda row: row["updated_at"], reverse=True)
            return rows

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    # --- message operations ---

    def add_message(self, session_id: str, role: str, content: str) -> ChatMessage:
        with self._lock:
            session = self.get(session_id)
            msg = session.add_message(role, content)
            self._save(session)
            return msg

    def history(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            return list(self.get(session_id).messages)

    def recent_history(
        self, session_id: str, max_messages: int | None = None
    ) -> list[ChatMessage]:
        with self._lock:
            messages = self.history(session_id)
            if max_messages is not None and max_messages > 0:
                messages = messages[-max_messages:]
            return messages

    # --- internals ---

    def _evict_if_needed(self) -> None:
        if len(self._sessions) >= MAX_SESSIONS:
            oldest_id = min(
                self._sessions, key=lambda sid: self._sessions[sid].updated_at
            )
            removed = self._sessions.pop(oldest_id)
            logger.warning("Evicted oldest session %s (capacity reached).", removed.id)

    def _path_for(self, session_id: str) -> Path:
        if self._storage_dir is None:
            raise RuntimeError("SessionManager has no storage_dir configured.")
        safe = _SAFE_NAME.sub("_", session_id) or "session"
        return self._storage_dir / f"{safe}.json"

    def _save(self, session: Session) -> None:
        if not self._persist or self._storage_dir is None:
            return
        try:
            self._path_for(session.id).write_text(
                session.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not persist session %s: %s", session.id, exc)

    def _load(self, session_id: str) -> Session | None:
        if self._storage_dir is None:
            return None
        path = self._path_for(session_id)
        if not path.exists():
            return None
        try:
            return Session.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not load session %s: %s", session_id, exc)
            return None