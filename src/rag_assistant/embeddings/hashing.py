"""Dependency-free hashing embedder.

Not a semantic model — it projects token counts into a fixed-width space with a
hashing trick (like scikit-learn's HashingVectorizer). It exists so that:

* the app boots and the tests run with zero network access or model downloads;
* dimensionality experiments are cheap (set `EMBEDDING_DIMENSIONS` to 64, 256,
  1024 … and watch retrieval quality move in the benchmark tab).

It matches on shared vocabulary, so it retrieves keyword-overlapping chunks
reasonably well and paraphrases poorly. Use a real model for actual work.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from rag_assistant.config import EmbeddingConfig
from rag_assistant.core.text import tokenize
from rag_assistant.embeddings.base import EMBEDDERS, Embedder

_DEFAULT_DIM = 512


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if value & 1 else -1.0
    return (value >> 1) % dim, sign


@EMBEDDERS.register("hashing", "hash", "offline")
class HashingEmbedder(Embedder):
    provider = "hashing"

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._dim = config.dimensions or _DEFAULT_DIM

    @property
    def native_dimension(self) -> int:
        return self._dim

    @property
    def label(self) -> str:
        return f"hashing:bag-of-words ({self._dim}d)"

    @property
    def dimension(self) -> int:
        return self._dim

    def _post_process(self, vectors: np.ndarray) -> np.ndarray:
        from rag_assistant.embeddings.base import l2_normalize

        return l2_normalize(vectors)

    def _encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                index, sign = _bucket(token, self._dim)
                matrix[row, index] += sign * (1.0 + math.log(count))
        return matrix
