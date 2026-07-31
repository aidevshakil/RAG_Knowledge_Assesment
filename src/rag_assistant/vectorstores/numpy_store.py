"""Dependency-free vector store: a NumPy matrix plus a JSON sidecar.

Exact (brute-force) cosine search. A single matrix multiply over ~100k chunks is
still only a few milliseconds, which covers personal and small-team knowledge
bases comfortably — and it makes the project runnable with zero extra installs.
Move to Chroma/FAISS/Qdrant when the corpus outgrows memory.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from rag_assistant.core.exceptions import VectorStoreError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.types import Chunk, SearchHit
from rag_assistant.vectorstores.base import VECTOR_STORES, VectorStore

logger = get_logger(__name__)


@VECTOR_STORES.register("numpy", "memory", "local")
class NumpyVectorStore(VectorStore):
    provider = "numpy"

    def __init__(self, *, collection: str, dimension: int, directory: Path) -> None:
        super().__init__(collection=collection, dimension=dimension, directory=directory)
        self._lock = threading.RLock()
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._records: list[dict] = []
        self._index: dict[str, int] = {}
        self._dirty = False
        self._load()

    @property
    def _vectors_path(self) -> Path:
        return self.directory / f"{self.collection}.vectors.npy"

    @property
    def _records_path(self) -> Path:
        return self.directory / f"{self.collection}.records.json"

    def _load(self) -> None:
        if not (self._vectors_path.exists() and self._records_path.exists()):
            return
        try:
            vectors = np.load(self._vectors_path).astype(np.float32)
            payload = json.loads(self._records_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s, starting empty: %s", self.collection, exc)
            return
        records = payload.get("records", [])
        stored_dim = int(payload.get("dimension", vectors.shape[1] if vectors.size else 0))
        if len(records) != vectors.shape[0]:
            logger.warning(
                "Corrupt store (%d records / %d vectors), starting empty",
                len(records),
                vectors.shape[0],
            )
            return
        if stored_dim and stored_dim != self.dimension:
            logger.warning(
                "Collection '%s' holds %dd vectors but the current model emits %dd",
                self.collection,
                stored_dim,
                self.dimension,
            )
            self.dimension = stored_dim
            self._vectors = np.zeros((0, stored_dim), dtype=np.float32)
        self._vectors = vectors
        self._records = records
        self._index = {rec["id"]: row for row, rec in enumerate(records)}

    def persist(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp_vectors = self._vectors_path.with_name(self._vectors_path.name + ".tmp.npy")
            tmp_records = self._records_path.with_name(self._records_path.name + ".tmp")
            np.save(tmp_vectors, self._vectors)
            tmp_records.write_text(
                json.dumps({"dimension": self.dimension, "records": self._records}),
                encoding="utf-8",
            )
            tmp_vectors.replace(self._vectors_path)
            tmp_records.replace(self._records_path)
            self._dirty = False

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        vectors = self._validate(chunks, vectors)
        with self._lock:
            new_rows: list[np.ndarray] = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                record = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": self._serialize_metadata(chunk),
                }
                existing = self._index.get(chunk.id)
                if existing is not None:
                    self._records[existing] = record
                    self._vectors[existing] = vector
                    continue
                self._index[chunk.id] = len(self._records)
                self._records.append(record)
                new_rows.append(vector)
            if new_rows:
                self._vectors = np.vstack([self._vectors, np.array(new_rows, dtype=np.float32)])
            self._dirty = True
        self.persist()
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            keep = [
                row
                for row, rec in enumerate(self._records)
                if rec["metadata"].get("document_id") != document_id
            ]
            removed = len(self._records) - len(keep)
            if not removed:
                return 0
            self._vectors = (
                self._vectors[keep] if keep else np.zeros((0, self.dimension), dtype=np.float32)
            )
            self._records = [self._records[row] for row in keep]
            self._index = {rec["id"]: row for row, rec in enumerate(self._records)}
            self._dirty = True
        self.persist()
        return removed

    def reset(self) -> None:
        with self._lock:
            self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
            self._records = []
            self._index = {}
            self._dirty = True
            for path in (self._vectors_path, self._records_path):
                path.unlink(missing_ok=True)
            self._dirty = False

    def search(
        self, vector: np.ndarray, k: int, *, document_ids: list[str] | None = None
    ) -> list[SearchHit]:
        with self._lock:
            if not self._records or k <= 0:
                return []
            query = np.asarray(vector, dtype=np.float32).ravel()
            if query.shape[0] != self._vectors.shape[1]:
                raise VectorStoreError(
                    f"query has {query.shape[0]} dims, index has {self._vectors.shape[1]}"
                )
            scores = self._vectors @ query
            candidates = np.arange(scores.shape[0])
            if document_ids:
                allowed = set(document_ids)
                mask = np.array(
                    [rec["metadata"].get("document_id") in allowed for rec in self._records]
                )
                candidates = candidates[mask]
                scores = scores[mask]
            if candidates.size == 0:
                return []
            top = min(k, candidates.size)
            picked = np.argpartition(-scores, top - 1)[:top]
            picked = picked[np.argsort(-scores[picked])]
            return [
                SearchHit(
                    chunk=self._deserialize_chunk(
                        self._records[candidates[i]]["id"],
                        self._records[candidates[i]]["text"],
                        self._records[candidates[i]]["metadata"],
                    ),
                    score=float(scores[i]),
                    rank=rank,
                )
                for rank, i in enumerate(picked)
            ]

    def count(self) -> int:
        return len(self._records)

    def document_ids(self) -> set[str]:
        with self._lock:
            return {
                str(rec["metadata"].get("document_id"))
                for rec in self._records
                if rec["metadata"].get("document_id")
            }
