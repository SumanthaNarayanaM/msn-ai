"""Long-term memory management endpoints.

The memory manager is the abstraction between conversations and durable
facts/preferences. Storing is explicit: only records a client actually asks
to remember. Recall supports keyword search for now (embeddings later).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_memory_manager
from app.memory import MemoryManager
from app.memory.store import MemoryRecord

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, description="What should be remembered")
    type: str = Field(default="fact", description="fact|preference|project|story")
    metadata: dict = Field(default_factory=dict)
    source_session_id: str = Field(default="")


@router.get("", summary="List stored memory records")
def list_memory(
    type: str | None = None,
    manager: MemoryManager = Depends(get_memory_manager),
) -> dict:
    records = manager.all(type=type)
    return {"count": len(records), "records": [r.model_dump() for r in records]}


@router.post("", summary="Store a long-term memory record")
def store_memory(
    payload: MemoryCreate,
    manager: MemoryManager = Depends(get_memory_manager),
) -> dict:
    try:
        record = manager.remember(
            payload.content,
            type=payload.type,
            metadata=payload.metadata,
            source_session_id=payload.source_session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"record": record.model_dump()}


@router.get("/recall", summary="Recall relevant long-term memory")
def recall_memory(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
    manager: MemoryManager = Depends(get_memory_manager),
) -> dict:
    records = manager.recall(query, limit=limit)
    return {"query": query, "count": len(records), "records": [r.model_dump() for r in records]}


@router.delete("/{record_id}", summary="Delete a memory record")
def delete_memory(
    record_id: str,
    manager: MemoryManager = Depends(get_memory_manager),
) -> dict:
    removed = manager.delete(record_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Memory record '{record_id}' not found.")
    return {"deleted": record_id}


def _dump(record: MemoryRecord) -> dict:
    return record.model_dump()