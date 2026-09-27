"""Inference providers package."""

from app.inference.providers.local_llm import RealLocalModelProvider
from app.inference.providers.mock import MockLocalModelProvider

__all__ = ["MockLocalModelProvider", "RealLocalModelProvider"]