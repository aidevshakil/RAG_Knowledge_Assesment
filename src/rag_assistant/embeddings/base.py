"""Embedder interface.

Contract: every implementation returns **L2-normalised float32** vectors, so a
dot product is cosine similarity everywhere downstream. That single invariant is
what lets the vector stores, MMR re-ranker and evaluation metrics share code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from rag_assistant.config import EmbeddingConfig
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.registry import Registry

logger = get_logger(__name__)

EMBEDDERS: Registry[Embedder] = Registry("embedding provider")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation that is safe for zero vectors."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0.0 else (matrix / norm).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    """Turns text into normalised vectors."""

    provider: str = "base"

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch; may return unnormalised vectors of native width."""

    @property
    @abstractmethod
    def native_dimension(self) -> int:
        """Vector width before optional truncation."""

    @property
    def dimension(self) -> int:
        target = self.config.dimensions
        native = self.native_dimension
        return min(target, native) if target else native

    @property
    def label(self) -> str:
        return f"{self.config.label} ({self.dimension}d)"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode many texts in batches. Returns shape (len(texts), dimension)."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        batch_size = max(1, self.config.batch_size)
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            chunks.append(self._post_process(self._encode(batch)))
        return np.vstack(chunks)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query. Returns shape (dimension,)."""
        return self.embed_documents([text])[0]

    def _post_process(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        target = self.config.dimensions
        if target and vectors.shape[1] > target:
            vectors = vectors[:, :target]
        return l2_normalize(vectors)


class CachedEmbedder(Embedder):
    """LRU-caching decorator — repeat queries skip the model entirely.

    Chunk embedding is not cached: ingestion streams unique text and caching it
    would only grow memory.
    """

    def __init__(self, inner: Embedder, max_size: int = 512) -> None:
        super().__init__(inner.config)
        self.inner = inner
        self.provider = inner.provider
        self._max_size = max(0, max_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def native_dimension(self) -> int:
        return self.inner.native_dimension

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def label(self) -> str:
        return self.inner.label

    def _encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - delegation
        return self.inner._encode(texts)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self._max_size == 0:
            return self.inner.embed_query(text)
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            self.hits += 1
            return cached
        self.misses += 1
        vector = self.inner.embed_query(text)
        self._cache[text] = vector
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return vector

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
