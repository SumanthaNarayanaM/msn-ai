"""FastAPI dependency providers.

The app stores the runtime singletons (SessionManager and the active
ModelProvider) on ``app.state`` at startup; these functions expose them as
FastAPI dependencies so the endpoint code stays decoupled from construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.memory import MemoryManager

if TYPE_CHECKING:  # pragma: no cover
    from starlette.applications import Starlette


def _require_state(request: Request) -> "Starlette":
    if not hasattr(request.app.state, "msn_ai_ready"):
        raise RuntimeError("Application has not finished initializing.")
    return request.app


def get_session_manager(request: Request) -> SessionManager:
    app = _require_state(request)
    return app.state.session_manager


def get_provider(request: Request) -> ModelProvider:
    app = _require_state(request)
    return app.state.provider


def get_memory_manager(request: Request) -> MemoryManager:
    app = _require_state(request)
    return app.state.memory_manager