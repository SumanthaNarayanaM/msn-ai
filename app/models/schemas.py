"""Pydantic request/response schema definitions for the v1 API.

Ordering matters: the v0.2 ingestion schemas are declared before the chat
schemas because ``ChatRequest``/``ChatResponse`` reference them in annotations.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    provider: str
    family: str = "text-generation"
    loaded: bool = False
    description: str = ""


class ChatMessage(BaseModel):
    role: str = Field(description="One of system, user, assistant")
    content: str = Field(description="Message text")


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str
    provider: Optional[str] = None
    model: Optional[str] = None
    uptime_seconds: Optional[float] = None


# --- v0.2 multimodal ingestion -------------------------------------------


class FileAttachment(BaseModel):
    """A file attached to a chat message.

    Optional and additive: v0.1 clients keep working because every field on the
    chat schemas has a default.
    """

    file_id: str = Field(..., description="Internal file id from POST /files/upload")
    filename: str = Field(default="", description="Display name, for the UI only.")


class KnowledgeSource(BaseModel):
    """One retrieval hit, returned alongside a chat reply."""

    file_id: str
    filename: str = ""
    locator: str = Field(
        default="", description="page 12 / Sheet 'Revenue' row 7 / 01:20-01:45"
    )
    kind: str = "document"
    score: float = 0.0
    matched_by: str = "vector"
    text: str = ""
    source: str = ""


class FileResponse(BaseModel):
    """A stored upload and its ingestion state."""

    id: str
    filename: str
    display_name: str = ""
    mime_type: str = "application/octet-stream"
    size: int = 0
    sha256: str = ""
    status: str = "uploaded"
    source_kind: str = "document"
    is_video: bool = False
    is_image: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    processed_at: Optional[float] = None
    scope: str = "session"
    session_id: str = ""
    remembered: bool = False
    error: str = ""
    warnings: list[str] = Field(default_factory=list)
    progress: float = 0.0
    progress_stage: str = ""
    chunk_count: int = 0
    block_count: int = 0
    char_count: int = 0
    document_metadata: dict = Field(default_factory=dict)
    unavailable: dict = Field(default_factory=dict)
    artifacts: dict = Field(default_factory=dict)
    #: Convenience booleans so the UI need not interpret these itself.
    has_thumbnail: bool = False
    readable: bool = True


class FileListResponse(BaseModel):
    files: list[FileResponse] = Field(default_factory=list)
    total: int = 0


class RememberRequest(BaseModel):
    """Promote a file's knowledge to a broader memory scope."""

    scope: Literal["project", "personal"] = "project"
    project: str = Field(
        default="", max_length=120, description="Project name for scope='project'."
    )


class SearchRequest(BaseModel):
    """Explicit knowledge search (chat does this implicitly)."""

    query: str = Field(..., min_length=1, description="Search text.")
    scope: Literal["session", "project", "personal", "all"] = "session"
    session_id: Optional[str] = None
    project: str = ""
    top_k: int = Field(default=5, ge=1, le=50)
    file_ids: Optional[list[str]] = None


class SearchResponse(BaseModel):
    query: str
    sources: list[KnowledgeSource] = Field(default_factory=list)
    scope: str = "session"
    used_knowledge: bool = False


class CapabilityStatus(BaseModel):
    """Honest per-capability availability for the frontend and /health."""

    provider: str
    available: bool
    reason: str = ""


class IngestionHealth(BaseModel):
    """Everything the UI needs to explain what MSN AI can currently do."""

    enabled: bool = True
    supported_extensions: list[str] = Field(default_factory=list)
    limits: dict = Field(default_factory=dict)
    files: dict = Field(default_factory=dict)
    knowledge: dict = Field(default_factory=dict)
    capabilities: dict = Field(default_factory=dict)


# --- chat (v0.1 contract, extended additively in v0.2) --------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(
        default=None, description="Existing conversation session; omit to create one."
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, le=4096, description="Override token budget."
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Override sampling temperature."
    )
    stream: bool = Field(
        default=False, description="Reserved; streaming not implemented in v0.1."
    )
    # --- v0.2 additions; all defaulted so v0.1 clients are unaffected ---
    attachments: list[FileAttachment] = Field(
        default_factory=list,
        description="Files to ground this reply in; their text is searched first.",
    )
    use_knowledge: bool = Field(
        default=True,
        description="Set false to answer without consulting uploaded files at all.",
    )
    scope: Literal["session", "project", "personal", "all"] = Field(
        default="session", description="Which memory scopes may be consulted."
    )
    project: str = Field(default="", max_length=120)
    top_k: Optional[int] = Field(
        default=None, ge=1, le=50, description="Override retrieved chunk count."
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model: str
    provider: str
    usage: Literal["mock"]
    created_at: float
    # --- v0.2 additions (defaulted) ---
    sources: list[KnowledgeSource] = Field(
        default_factory=list, description="Knowledge chunks used for this reply."
    )
    used_knowledge: bool = False
    attachments: list[FileAttachment] = Field(default_factory=list)
    #: Honest notes about capabilities that were missing for this turn.
    unavailable: dict = Field(default_factory=dict)


class RenameRequest(BaseModel):
    session_id: str
    name: str = Field(default="", max_length=120, description="New session name.")
