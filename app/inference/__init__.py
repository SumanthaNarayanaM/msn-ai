"""ModelProvider abstraction and provider registry.

The provider interface is deliberately minimal and provider-agnostic so that
providers can be swapped (mock <-> real local model) without touching the API
layer.
"""

from app.inference.base import ModelProvider
from app.inference.providers.local_llm import RealLocalModelProvider
from app.inference.providers.mock import MockLocalModelProvider
from app.inference.registry import (
    create_provider,
    create_provider_safe,
    get_available_providers,
)

__all__ = [
    "ModelProvider",
    "MockLocalModelProvider",
    "RealLocalModelProvider",
    "create_provider",
    "create_provider_safe",
    "get_available_providers",
]