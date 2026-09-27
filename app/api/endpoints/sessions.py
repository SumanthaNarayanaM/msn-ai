"""Conversation session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_session_manager
from app.core.session import SessionManager
from app.models.schemas import ChatMessage

router = APIRouter(tags=["sessions"])


@router.get("/sessions", summary="List active sessions")
def list_sessions(
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    items = manager.summaries()
    return {
        "count": len(items),
        "sessions": [item["id"] for item in items],
        "items": items,
    }


@router.get("/sessions/{session_id}", summary="Get session history")
def get_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    try:
        messages = manager.history(session_id)
        summary = manager.summaries()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    meta = next((item for item in summary if item["id"] == session_id), {})
    return {
        "session_id": session_id,
        "title": meta.get("title", "New conversation"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "message_count": len(messages),
        "messages": [ChatMessage(**m.model_dump()).model_dump() for m in messages],
    }


@router.delete("/sessions/{session_id}", summary="Delete a session")
def delete_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    manager.delete(session_id)
    return {"deleted": session_id}