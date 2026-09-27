"""Version 1 API routes."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.health import router as health_router
from app.api.endpoints.memory import router as memory_router
from app.api.endpoints.sessions import router as sessions_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(chat_router)
api_router.include_router(sessions_router)
api_router.include_router(memory_router)