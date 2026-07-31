"""Build an embedder from config."""

from __future__ import annotations

from rag_assistant.config import EmbeddingConfig
from rag_assistant.core.logging_config import get_logger
from rag_assistant.embeddings.base import EMBEDDERS, CachedEmbedder, Embedder

logger = get_logger(__name__)


def build_embedder(config: EmbeddingConfig, *, cache: bool = True) -> Embedder:
    embedder = EMBEDDERS.get(config.provider)(config)
    logger.info("Embedder ready: %s", embedder.label)
    return CachedEmbedder(embedder, config.cache_size) if cache else embedder


def available_embedders() -> list[str]:
    return EMBEDDERS.names
