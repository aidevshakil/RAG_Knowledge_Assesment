"""Vector store interface.

Deliberately narrow, because the app's requirements are narrow:

* `add` chunks with their vectors,
* `search` by vector,
* `delete_document` — a *real* delete: the vectors leave the index, so removed
  files can never be retrieved again,
* introspection (`count`, `document_ids`) for the UI and the health checks.

All stores assume normalised vectors and return **cosine similarity in [-1, 1]**
(higher is better), regardless of what the backend natively reports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from rag_assistant.core.exceptions import DimensionMismatchError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.registry import Registry
from rag_assistant.core.types import Chunk, SearchHit

logger = get_logger(__name__)

VECTOR_STORES: Registry[VectorStore] = Registry("vector store")

_META_KEYS = ("document_id", "document_name", "index")


class VectorStore(ABC):
    """Persistent similarity index over document chunks."""

    provider: str = "base"

    def __init__(self, *, collection: str, dimension: int, directory: Path) -> None:
        self.collection = collection
        self.dimension = dimension
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Index `chunks`; returns the number stored. Upserts by chunk id."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove every vector belonging to `document_id`; returns count removed."""

    @abstractmethod
    def reset(self) -> None:
        """Drop the whole collection."""

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        k: int,
        *,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        """Top-`k` most similar chunks, optionally restricted to some documents."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed vectors."""

    @abstractmethod
    def document_ids(self) -> set[str]:
        """Distinct document ids currently represented in the index."""

    def persist(self) -> None:  # noqa: B027 - intentionally optional, not abstract
        """Flush to disk. No-op for stores that write through."""

    def close(self) -> None:
        self.persist()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _validate(self, chunks: list[Chunk], vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D vector matrix, got shape {vectors.shape}")
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors"
            )
        if vectors.shape[1] != self.dimension:
            raise DimensionMismatchError(self.dimension, int(vectors.shape[1]))
        return vectors

    @staticmethod
    def _serialize_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
        """Flatten chunk metadata — backends only accept scalar values."""
        meta: dict[str, str | int | float | bool] = {
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "index": chunk.index,
        }
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)) and key not in meta:
                meta[key] = value
        return meta

    @staticmethod
    def _deserialize_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
        extra = {k: v for k, v in meta.items() if k not in _META_KEYS}
        return Chunk(
            id=chunk_id,
            document_id=str(meta.get("document_id", "")),
            document_name=str(meta.get("document_name", "unknown")),
            text=text,
            index=int(meta.get("index", 0)),
            metadata=extra,
        )

    def stats(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "collection": self.collection,
            "dimension": self.dimension,
            "vectors": self.count(),
            "documents": len(self.document_ids()),
            "location": str(self.directory),
        }
