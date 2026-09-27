"""Chat endpoint backed by the active ModelProvider."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import (
    get_ingestion_service_or_none,
    get_memory_manager,
    get_provider,
    get_session_manager,
)
from app.core.logging import get_logger
from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.ingestion.service import IngestionService
from app.memory import MemoryManager
from app.models.schemas import ChatRequest, ChatResponse, KnowledgeSource

logger = get_logger("msn_ai.api.chat")

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
def chat(
    request: ChatRequest,
    http_request: Request,
    manager: SessionManager = Depends(get_session_manager),
    provider: ModelProvider = Depends(get_provider),
    memory: MemoryManager = Depends(get_memory_manager),
    ingestion: IngestionService | None = Depends(get_ingestion_service_or_none),
) -> ChatResponse:
    """Process one user message within a conversation session.

    - ``session_id`` is optional: a new session is created when omitted or
      when the given id does not exist.
    - The full session history is handed to the provider as context, together
      with the MSN AI personality system prompt and any relevant long-term
      memory recalled for this message.
    - When ``use_knowledge`` is true (the default) and files are attached or in
      scope, relevant excerpts are added as a cited context block.
    """
    if request.stream:
        raise NotImplementedError("Streaming is reserved for a later release.")

    session = manager.get_or_create(request.session_id)

    manager.add_message(session.id, "user", request.message)

    history = manager.history(session.id)
    context = [message.model_dump() for message in history]

    # Personality: MSN AI's configurable system prompt comes 'first' in the
    # context so the model always knows how to behave.
    context.insert(0, {"role": "system", "content": provider_system_prompt(http_request)})

    # Long-term memory: only *recalled* relevant facts are included; nothing
    # is stored automatically during chat.
    memory_context = memory.recall_for_context(request.message)
    if memory_context:
        context.insert(1, {"role": "system", "content": memory_context})

    knowledge_text, sources = _retrieve(request, ingestion, session)
    if knowledge_text:
        context.insert(1, {"role": "system", "content": knowledge_text})

    info = provider.info()
    reply = provider.generate(
        context,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    manager.add_message(session.id, "assistant", reply)

    logger.info(
        "chat session=%s provider=%s knowledge=%d source(s)",
        session.id,
        provider.name,
        len(sources),
    )

    return ChatResponse(
        session_id=session.id,
        reply=reply,
        model=info.name,
        provider=provider.name,
        usage="mock",
        created_at=time.time(),
        sources=sources,
        used_knowledge=bool(sources),
    )


def _retrieve(
    request: ChatRequest,
    ingestion: IngestionService | None,
    session: Any,
) -> tuple[str, list[KnowledgeSource]]:
    """Build the cited knowledge context for this message.

    Both the server-assigned session id and the id the client sent are
    searched: the frontend pins its own id when uploading files, and the
    session manager may hand out a different one, so files uploaded under the
    client's id must stay visible.

    Returns an empty context when knowledge is unavailable, so a broken or
    missing ingestion service degrades to plain chat instead of failing.
    """
    if not request.use_knowledge or ingestion is None:
        return "", []

    owner_ids = [session.id]
    if request.session_id and request.session_id not in owner_ids:
        owner_ids.append(request.session_id)
    file_ids = [a.file_id for a in request.attachments if a.file_id]
    try:
        text, hits = ingestion.build_context(
            request.message,
            scope=request.scope,
            session_id=session.id,
            session_ids=owner_ids,
            project=request.project,
            top_k=request.top_k,
            file_ids=file_ids or None,
        )
    except Exception as exc:  # pragma: no cover - retrieval must never break chat
        logger.warning("Knowledge retrieval failed, answering without it: %s", exc)
        return "", []
    if not text:
        return "", []
    sources = [KnowledgeSource(**hit.to_dict()) for hit in hits]
    return _knowledge_block(text, sources), sources


def _knowledge_block(text: str, sources: list[KnowledgeSource]) -> str:
    """Wrap retrieved excerpts in instructions the small chat model can follow."""
    lines = [
        "Knowledge retrieved from the user's uploaded files. Use it when it is "
        "relevant, ignore it when it is not, and never invent facts that are "
        "not in it. When you rely on it, name the source in square brackets, "
        "for example [report.pdf - page 3].",
        "",
        text,
        "",
        "Sources:",
    ]
    lines += [f"- {source.source}" for source in sources]
    return "\n".join(lines)


def provider_system_prompt(http_request: Request | None = None) -> str:
    """Return the configured MSN AI personality system prompt."""
    config = None
    if http_request is not None:
        config = getattr(http_request.app.state, "config", None)
    if config is None:
        from app.core.config import settings as config
    return config.system_prompt.strip()