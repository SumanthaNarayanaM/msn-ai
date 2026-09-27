"""Liveness / readiness endpoint, including v0.2 ingestion capabilities."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_provider, get_session_manager
from app.core.config import settings
from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(
    request: Request,
    manager: SessionManager = Depends(get_session_manager),
    provider: ModelProvider = Depends(get_provider),
) -> HealthResponse:
    """Return service status, version, active provider and uptime."""
    info = provider.info()
    config = getattr(request.app.state, "config", None) or settings
    return HealthResponse(
        status="ok",
        app="MSN AI",
        version=config.version,
        environment=config.env,
        provider=info.provider,
        model=info.name,
        uptime_seconds=round(time.time() - manager.started_at, 3),
    )


@router.get("/ingestion/health", summary="File ingestion capabilities")
def ingestion_health(request: Request) -> dict[str, Any]:
    """Report exactly what MSN AI can read right now, and what it cannot.

    Every optional capability is reported as available/unavailable with a
    human-readable reason, so the frontend never has to guess and the API never
    implies an image was understood when no vision model is installed.
    """
    service = getattr(request.app.state, "ingestion_service", None)
    config = getattr(request.app.state, "config", None) or settings
    limits = {
        "max_upload_mb": config.max_upload_mb,
        "max_video_upload_mb": config.max_video_upload_mb,
        "max_archive_expanded_mb": config.max_archive_expanded_mb,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "max_context_chars": config.max_context_chars,
        "retrieval_top_k": config.retrieval_top_k,
    }
    if service is None:
        return {
            "enabled": False,
            "supported_extensions": [],
            "limits": limits,
            "files": {},
            "knowledge": {},
            "capabilities": {
                "ingestion": {
                    "provider": "ingestion",
                    "available": False,
                    "reason": (
                        "File ingestion is unavailable: the ingestion service "
                        "failed to start. See the server log for the cause."
                    ),
                }
            },
        }
    return {
        "enabled": bool(getattr(config, "ingestion_enabled", True)),
        "supported_extensions": service.supported_extensions(),
        "limits": limits,
        "files": service.store.counts(),
        "knowledge": service.knowledge.stats(),
        "capabilities": service.capabilities(),
    }
