"""Pluggable vector stores. `build_vector_store(...)` is the entry point."""

from rag_assistant.vectorstores import (  # noqa: F401  (registration side effects)
    chroma,
    faiss_store,
    numpy_store,
)
from rag_assistant.vectorstores.base import VECTOR_STORES, VectorStore
from rag_assistant.vectorstores.factory import available_vector_stores, build_vector_store

__all__ = [
    "VECTOR_STORES",
    "VectorStore",
    "available_vector_stores",
    "build_vector_store",
]
