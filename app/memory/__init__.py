"""Memory abstraction: Conversation -> MemoryManager -> LocalMemoryStore.

Long-term memory foundation. Current scope is a small keyword-based local
store; the interface is designed to later back onto embeddings + vector search
for real RAG without changing callers.
"""

from app.memory.manager import MemoryManager
from app.memory.store import LocalMemoryStore, MemoryRecord

__all__ = ["LocalMemoryStore", "MemoryManager", "MemoryRecord"]