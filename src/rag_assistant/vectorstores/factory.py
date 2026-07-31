"""Build a vector store from config."""

from __future__ import annotations

from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.core.logging_config import get_logger
from rag_assistant.vectorstores.base import VECTOR_STORES, VectorStore

logger = get_logger(__name__)


def build_vector_store(
    settings: Settings,
    dimension: int,
    *,
    collection: str | None = None,
    directory: Path | None = None,
) -> VectorStore:
    cfg = settings.vector_store
    cls = VECTOR_STORES.get(cfg.provider)
    store = cls(
        collection=collection or cfg.collection,
        dimension=dimension,
        directory=directory or settings.vector_dir,
    )
    logger.info(
        "Vector store ready: %s (collection=%s, dim=%d, vectors=%d)",
        store.provider,
        store.collection,
        dimension,
        store.count(),
    )
    return store


def available_vector_stores() -> list[str]:
    return VECTOR_STORES.names
