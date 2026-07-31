"""Adaptive RAG Knowledge Assistant.

A modular retrieval-augmented generation stack with live document management
and built-in performance benchmarking.

Every external dependency (embedding model, vector store, LLM) sits behind a
small interface plus a registry, so backends are swappable through config only.
"""

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.types import Chunk, Document, RAGAnswer, SearchHit

__version__ = "1.0.0"

__all__ = [
    "Chunk",
    "Document",
    "RAGAnswer",
    "SearchHit",
    "Settings",
    "get_settings",
    "__version__",
]
