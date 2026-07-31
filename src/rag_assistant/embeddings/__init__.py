"""Embedding backends behind one interface.

`build_embedder(config)` is the only entry point the rest of the app uses.
"""

from rag_assistant.embeddings.base import EMBEDDERS, CachedEmbedder, Embedder
from rag_assistant.embeddings.factory import available_embedders, build_embedder

from rag_assistant.embeddings import hashing, huggingface, openai  # noqa: F401  (isort: skip)

__all__ = [
    "EMBEDDERS",
    "CachedEmbedder",
    "Embedder",
    "available_embedders",
    "build_embedder",
]
