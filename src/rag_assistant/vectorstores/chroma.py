"""ChromaDB backend — persistent, metadata-filtered, cosine space.

Chroma returns *distances*; with `hnsw:space=cosine` that is `1 - cosine`, so we
convert back to similarity to honour the `VectorStore` contract.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.core.exceptions import DependencyMissingError, VectorStoreError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.types import Chunk, SearchHit
from rag_assistant.vectorstores.base import VECTOR_STORES, VectorStore

logger = get_logger(__name__)

_MAX_BATCH = 2000


@VECTOR_STORES.register("chroma", "chromadb")
class ChromaVectorStore(VectorStore):
    provider = "chroma"

    def __init__(self, *, collection: str, dimension: int, directory: Path) -> None:
        super().__init__(collection=collection, dimension=dimension, directory=directory)
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise DependencyMissingError(
                "chromadb", "the Chroma vector store", extra="chroma"
            ) from exc

        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=str(self.directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine", "dimension": dimension},
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        vectors = self._validate(chunks, vectors)
        with self._lock:
            for start in range(0, len(chunks), _MAX_BATCH):
                batch = chunks[start : start + _MAX_BATCH]
                self._collection.upsert(
                    ids=[c.id for c in batch],
                    embeddings=vectors[start : start + len(batch)].tolist(),
                    documents=[c.text for c in batch],
                    metadatas=[self._serialize_metadata(c) for c in batch],
                )
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            before = self._collection.count()
            self._collection.delete(where={"document_id": document_id})
            return before - self._collection.count()

    def reset(self) -> None:
        with self._lock:
            self._client.delete_collection(self.collection)
            self._collection = self._client.get_or_create_collection(
                name=self.collection,
                metadata={"hnsw:space": "cosine", "dimension": self.dimension},
            )

    def search(
        self, vector: np.ndarray, k: int, *, document_ids: list[str] | None = None
    ) -> list[SearchHit]:
        if k <= 0:
            return []
        with self._lock:
            total = self._collection.count()
            if total == 0:
                return []
            where: dict[str, Any] | None = None
            if document_ids:
                where = (
                    {"document_id": document_ids[0]}
                    if len(document_ids) == 1
                    else {"document_id": {"$in": list(document_ids)}}
                )
            try:
                result = self._collection.query(
                    query_embeddings=[np.asarray(vector, dtype=np.float32).ravel().tolist()],
                    n_results=min(k, total),
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:
                raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[SearchHit] = []
        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(ids, docs, metas, distances, strict=False)
        ):
            hits.append(
                SearchHit(
                    chunk=self._deserialize_chunk(chunk_id, text or "", meta or {}),
                    score=1.0 - float(distance),
                    rank=rank,
                )
            )
        return hits

    def count(self) -> int:
        with self._lock:
            return int(self._collection.count())

    def document_ids(self) -> set[str]:
        with self._lock:
            if self._collection.count() == 0:
                return set()
            result = self._collection.get(include=["metadatas"])
        return {
            str(meta["document_id"])
            for meta in (result.get("metadatas") or [])
            if meta and meta.get("document_id")
        }
