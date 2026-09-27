"""Liveness / readiness endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.dependencies import get_session_manager, get_provider
from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(
    manager: SessionManager = Depends(get_session_manager),
    provider: ModelProvider = Depends(get_provider),
) -> HealthResponse:
    """Return service status, version, active provider and uptime."""
    info = provider.info()
    return HealthResponse(
        status="ok",
        app="MSN AI",
        version="0.1.0",
        environment="development",
        provider=info.provider,
        model=info.name,
        uptime_seconds=round(time.time() - manager.started_at, 3),
    )