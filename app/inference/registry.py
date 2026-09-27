"""Provider factory and registry.

New real model providers (local coding model, image, video, RAG embedder, ...)
must be registered here so they can be selected via ``MSN_AI_PROVIDER``.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.errors import ProviderNotFoundError
from app.core.logging import get_logger
from app.inference.base import ModelProvider
from app.inference.providers.local_llm import RealLocalModelProvider
from app.inference.providers.mock import MockLocalModelProvider

logger = get_logger("msn_ai.provider.registry")

#: name -> provider class. Extend to register real providers later.
_PROVIDERS: dict[str, type[ModelProvider]] = {
    MockLocalModelProvider.name: MockLocalModelProvider,
    RealLocalModelProvider.name: RealLocalModelProvider,
    # Convenience alias so both MSN_AI_PROVIDER=llamacpp and =local work.
    "local": RealLocalModelProvider,
}

#: Lazy references to instantiated (singleton) providers, keyed by config hash.
_instances: dict[tuple, ModelProvider] = {}


def get_available_providers() -> list[str]:
    """Return the names of all registered providers."""
    return list(_PROVIDERS)


def _config_key(model_name: str) -> tuple:
    return (
        model_name,
        settings.max_tokens,
        settings.temperature,
        str(settings.resolved_model_path()),
    )


def _base_config(provider_model_name: str) -> dict:
    """Build the provider constructor config from application settings."""
    return {
        "model_name": provider_model_name,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "model_path": str(settings.resolved_model_path()),
        "n_ctx": settings.n_ctx,
        "n_threads": settings.n_threads,
        "model_hf_repo": settings.model_hf_repo,
        "model_hf_file": settings.model_hf_file,
    }


def create_provider(
    name: str | None = None,
    model_name: str | None = None,
    force_new: bool = False,
    provider_config: dict | None = None,
) -> ModelProvider:
    """Instantiate (or reuse) the requested provider.

    - ``name`` selects the provider class; defaults to ``settings.provider``.
    - ``model_name`` overrides the model identifier; defaults to ``settings.model_name``.
    - ``provider_config`` overrides constructor options (useful in tests).
    """
    provider_name = (name or settings.provider).strip().lower()
    cls = _PROVIDERS.get(provider_name)
    if cls is None:
        raise ProviderNotFoundError(
            f"Unknown provider '{provider_name}'. Available: "
            f"{', '.join(get_available_providers())}."
        )

    provider_model_name = model_name or settings.model_name
    key = (provider_name, * _config_key(provider_model_name))

    if not force_new and key in _instances:
        provider = _instances[key]
        if not provider.is_loaded():
            provider.load()
        logger.debug("Reusing cached provider instance for '%s'.", provider_name)
        return provider

    config = {**_base_config(provider_model_name), **(provider_config or {})}
    instance = cls(model_config=config)
    instance.load()
    if not force_new:
        _instances[key] = instance
    return instance


def create_provider_safe(
    name: str | None = None,
    model_name: str | None = None,
    provider_config: dict | None = None,
) -> tuple[ModelProvider, dict[str, str]]:
    """Create the requested provider, falling back to the mock provider.

    Returns ``(provider, meta)`` where ``meta`` explains the resolution so the
    caller can log whether a real model or the mock fallback is in use. The
    mock provider is always available, keeping the platform alive even when a
    real model file or the llama.cpp runtime is missing.
    """
    provider_name = (name or settings.provider).strip().lower()
    try:
        return create_provider(
            name=provider_name,
            model_name=model_name,
            provider_config=provider_config,
        ), {"resolved": provider_name, "fallback": False}
    except Exception as exc:
        logger.warning(
            "Provider '%s' unavailable (%s); falling back to mock provider.",
            provider_name,
            exc,
        )
        return create_provider(
            name="mock",
            model_name=model_name or settings.model_name,
            force_new=True,
            provider_config=provider_config,
        ), {"resolved": "mock", "fallback": True, "reason": str(exc)}