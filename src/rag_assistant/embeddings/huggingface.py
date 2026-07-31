"""Local sentence-transformers embeddings (no API key, no per-token cost)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from rag_assistant.config import EmbeddingConfig
from rag_assistant.core.exceptions import DependencyMissingError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.embeddings.base import EMBEDDERS, Embedder

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> Any:
    """Model loading is slow (~seconds) — cache per process, keyed by name."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise DependencyMissingError(
            "sentence-transformers", "HuggingFace embeddings", extra="hf"
        ) from exc
    logger.info("Loading sentence-transformers model %s (first run downloads it)", model_name)
    return SentenceTransformer(model_name)


@EMBEDDERS.register("huggingface", "hf", "sentence-transformers")
class HuggingFaceEmbedder(Embedder):
    provider = "huggingface"

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model = _load_model(config.model)

    @property
    def native_dimension(self) -> int:
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        return int(getter())

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
