"""Pydantic request/response schema definitions for the v1 API."""

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


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model: str
    provider: str
    usage: Literal["mock"]
    created_at: float


class RenameRequest(BaseModel):
    session_id: str
    name: str = Field(default="", max_length=120, description="New session name.")