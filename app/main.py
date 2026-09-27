"""MSN AI FastAPI application factory and entrypoint."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import PROJECT_ROOT, settings
from app.core.errors import register_error_handlers
from app.core.logging import get_logger, setup_logging
from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.inference.registry import create_provider_safe
from app.memory import LocalMemoryStore, MemoryManager
from app.utils.paths import ensure_standard_directories

logger = get_logger("msn_ai.app")


def create_app() -> FastAPI:
    """Build and configure the MSN AI application."""
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _bootstrap(app)
        try:
            yield
        finally:
            _shutdown(app)

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "MSN AI - fully local AI platform (Code / Image / Video / Agent / "
            "RAG / Tools). Local chat + coding assistant with conversation and "
            "long-term memory."
        ),
        lifespan=lifespan,
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(api_router)

    _mount_web_ui(app)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


def _bootstrap(app: FastAPI) -> None:
    """Wire runtime singletons onto app.state."""
    started = time.time()
    ensure_standard_directories(
        data_dir=settings.data_dir,
        model_storage_dir=settings.model_storage_dir,
        conversations_dir=settings.conversations_path(),
    )
    settings.memory_path().mkdir(parents=True, exist_ok=True)

    session_manager = SessionManager(
        storage_dir=settings.conversations_path(),
        persist=settings.session_persist,
    )
    memory_manager = MemoryManager(
        store=LocalMemoryStore(storage_dir=settings.memory_path())
    )
    provider, provider_meta = create_provider_safe(
        name=settings.provider,
        model_name=settings.model_name,
    )

    app.state.session_manager = session_manager
    app.state.memory_manager = memory_manager
    app.state.provider = provider
    app.state.provider_meta = provider_meta
    app.state.msn_ai_ready = True

    provider_state = "fallback->mock" if provider_meta.get("fallback") else provider_meta["resolved"]
    logger.info(
        "MSN AI %s started (%s) in %.2fs | provider=%s model=%s",
        settings.version,
        settings.env,
        time.time() - started,
        provider_state,
        provider.info().name,
    )


def _mount_web_ui(app: FastAPI) -> None:
    """Serve the local web chat interface from the ``web`` directory."""
    web_dir: Path = PROJECT_ROOT / "web"
    if not web_dir.is_dir():
        logger.warning("web/ directory missing; skip mounting web UI.")
        return
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")


def _shutdown(app: FastAPI) -> None:
    provider: ModelProvider | None = getattr(app.state, "provider", None)
    if provider is not None:
        provider.unload()
    memory: MemoryManager | None = getattr(app.state, "memory_manager", None)
    if memory is not None:
        memory.store.unload()
    logger.info("MSN AI shut down cleanly.")


app = create_app()


def run() -> None:
    """Start the uvicorn server with settings from the environment/.env."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()