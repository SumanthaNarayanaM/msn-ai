"""Chat endpoint backed by the active ModelProvider."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_memory_manager,
    get_provider,
    get_session_manager,
)
from app.core.logging import get_logger
from app.core.session import SessionManager
from app.inference.base import ModelProvider
from app.memory import MemoryManager
from app.models.schemas import ChatRequest, ChatResponse

logger = get_logger("msn_ai.api.chat")

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
def chat(
    request: ChatRequest,
    manager: SessionManager = Depends(get_session_manager),
    provider: ModelProvider = Depends(get_provider),
    memory: MemoryManager = Depends(get_memory_manager),
) -> ChatResponse:
    """Process one user message within a conversation session.

    - ``session_id`` is optional: a new session is created when omitted or
      when the given id does not exist.
    - The full session history is handed to the provider as context, together
      with the MSN AI personality system prompt and any relevant long-term
      memory recalled for this message.
    """
    if request.stream:
        raise NotImplementedError("Streaming is reserved for a later release.")

    session = manager.get_or_create(request.session_id)

    manager.add_message(session.id, "user", request.message)

    history = manager.history(session.id)
    context = [message.model_dump() for message in history]

    # Personality: MSN AI's configurable system prompt comes 'first' in the
    # context so the model always knows how to behave.
    context.insert(0, {"role": "system", "content": provider_system_prompt()})

    # Long-term memory: only *recalled* relevant facts are included; nothing
    # is stored automatically during chat.
    memory_context = memory.recall_for_context(request.message)
    if memory_context:
        context.insert(1, {"role": "system", "content": memory_context})

    info = provider.info()
    reply = provider.generate(
        context,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    manager.add_message(session.id, "assistant", reply)

    logger.info(
        "chat session=%s user_tokens_approx=%d provider=%s",
        session.id,
        len(request.message.split()),
        provider.name,
    )

    return ChatResponse(
        session_id=session.id,
        reply=reply,
        model=info.name,
        provider=provider.name,
        usage="mock",
        created_at=time.time(),
    )


def provider_system_prompt() -> str:
    """Return the configured MSN AI personality system prompt."""
    from app.core.config import settings

    return settings.system_prompt.strip()