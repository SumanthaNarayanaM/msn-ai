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
from app.core.config import PROJECT_ROOT, Settings, settings
from app.core.errors import register_error_handlers
from app.core.logging import get_logger, setup_logging
from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.inference.registry import create_provider_safe
from app.ingestion.embeddings import build_embedding_provider
from app.ingestion.knowledge import KnowledgeBase
from app.ingestion.registry import ExtractorRegistry
from app.ingestion.service import IngestionService
from app.ingestion.store import FileStore
from app.memory import LocalMemoryStore, MemoryManager
from app.utils.paths import ensure_ingestion_directories, ensure_standard_directories

logger = get_logger("msn_ai.app")


def create_app(config: Settings | None = None) -> FastAPI:
    """Build and configure the MSN AI application.

    ``config`` overrides the process-wide settings; tests use it to point the
    app at a temporary data directory.
    """
    setup_logging()
    config = config or settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _bootstrap(app, config)
        try:
            yield
        finally:
            _shutdown(app)

    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description=(
            "MSN AI - fully local AI platform (Code / Image / Video / Agent / "
            "RAG / Tools). Local chat + coding assistant with conversation and "
            "long-term memory."
        ),
        lifespan=lifespan,
        debug=config.debug,
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
            "service": config.app_name,
            "version": config.version,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


def _bootstrap(app: FastAPI, config: Settings) -> None:
    """Wire runtime singletons onto app.state."""
    started = time.time()
    ensure_standard_directories(
        data_dir=config.data_dir,
        model_storage_dir=config.model_storage_dir,
        conversations_dir=config.conversations_path(),
    )
    ensure_ingestion_directories(config)
    config.memory_path().mkdir(parents=True, exist_ok=True)

    session_manager = SessionManager(
        storage_dir=config.conversations_path(),
        persist=config.session_persist,
    )
    memory_manager = MemoryManager(
        store=LocalMemoryStore(storage_dir=config.memory_path())
    )
    provider, provider_meta = create_provider_safe(
        name=config.provider,
        model_name=config.model_name,
    )

    app.state.config = config
    app.state.session_manager = session_manager
    app.state.memory_manager = memory_manager
    app.state.provider = provider
    app.state.provider_meta = provider_meta
    app.state.msn_ai_ready = True

    # v0.2 ingestion: optional dependencies must never stop the server, so a
    # failure here degrades the file features instead of taking chat offline.
    ingestion_error = ""
    try:
        embedder = build_embedding_provider(config)
        store = FileStore(config)
        knowledge = KnowledgeBase(config, embedder)
        app.state.ingestion_service = IngestionService(
            config, embedder, store, knowledge
        )
        logger.info(
            "Ingestion ready: %d format(s), knowledge=%d chunk(s)",
            len(ExtractorRegistry().supported_extensions()),
            int(knowledge.stats()["chunks"]),
        )
    except Exception as exc:  # pragma: no cover - startup guard
        app.state.ingestion_service = None
        ingestion_error = str(exc)
        logger.error("Ingestion unavailable: %s", exc, exc_info=True)

    provider_state = "fallback->mock" if provider_meta.get("fallback") else provider_meta["resolved"]
    logger.info(
        "MSN AI %s started (%s) in %.2fs | provider=%s model=%s | ingestion=%s",
        config.version,
        config.env,
        time.time() - started,
        provider_state,
        provider.info().name,
        "error" if ingestion_error else "ready",
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
    ingestion = getattr(app.state, "ingestion_service", None)
    if ingestion is not None:
        ingestion.close()
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