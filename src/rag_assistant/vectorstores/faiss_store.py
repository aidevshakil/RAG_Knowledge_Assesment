"""FAISS backend — `IndexFlatIP` wrapped in `IndexIDMap2`.

FAISS stores vectors only, so chunk text and metadata live in a JSON sidecar
keyed by the integer id we assign. `IndexIDMap2.remove_ids` gives true deletion,
which is what the dynamic knowledge base needs.

Inner product over normalised vectors == cosine similarity, so no conversion is
needed on the way out.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from rag_assistant.core.exceptions import DependencyMissingError, VectorStoreError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.types import Chunk, SearchHit
from rag_assistant.vectorstores.base import VECTOR_STORES, VectorStore

logger = get_logger(__name__)


@VECTOR_STORES.register("faiss")
class FaissVectorStore(VectorStore):
    provider = "faiss"

    def __init__(self, *, collection: str, dimension: int, directory: Path) -> None:
        super().__init__(collection=collection, dimension=dimension, directory=directory)
        try:
            import faiss
        except ImportError as exc:
            raise DependencyMissingError(
                "faiss-cpu", "the FAISS vector store", extra="faiss"
            ) from exc
        self._faiss = faiss
        self._lock = threading.RLock()
        self._records: dict[int, dict] = {}
        self._by_chunk: dict[str, int] = {}
        self._next_id = 0
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self._load()

    @property
    def _index_path(self) -> Path:
        return self.directory / f"{self.collection}.faiss"

    @property
    def _meta_path(self) -> Path:
        return self.directory / f"{self.collection}.meta.json"

    def _load(self) -> None:
        if not (self._index_path.exists() and self._meta_path.exists()):
            return
        try:
            index = self._faiss.read_index(str(self._index_path))
            payload = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load FAISS index, starting empty: %s", exc)
            return
        if index.d != self.dimension:
            logger.warning(
                "FAISS index '%s' is %dd but the current model emits %dd",
                self.collection,
                index.d,
                self.dimension,
            )
            self.dimension = index.d
        self._index = index
        self._records = {int(k): v for k, v in payload.get("records", {}).items()}
        self._by_chunk = {rec["id"]: fid for fid, rec in self._records.items()}
        self._next_id = int(payload.get("next_id", max(self._records, default=-1) + 1))

    def persist(self) -> None:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self._index, str(self._index_path))
            self._meta_path.write_text(
                json.dumps(
                    {
                        "dimension": self.dimension,
                        "next_id": self._next_id,
                        "records": {str(k): v for k, v in self._records.items()},
                    }
                ),
                encoding="utf-8",
            )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        vectors = self._validate(chunks, vectors)
        with self._lock:
            stale = [self._by_chunk[c.id] for c in chunks if c.id in self._by_chunk]
            if stale:
                self._index.remove_ids(np.array(stale, dtype=np.int64))
                for fid in stale:
                    self._records.pop(fid, None)

            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype=np.int64)
            self._next_id += len(chunks)
            self._index.add_with_ids(np.ascontiguousarray(vectors), ids)
            for chunk, fid in zip(chunks, ids, strict=True):
                self._records[int(fid)] = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": self._serialize_metadata(chunk),
                }
                self._by_chunk[chunk.id] = int(fid)
        self.persist()
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            targets = [
                fid
                for fid, rec in self._records.items()
                if rec["metadata"].get("document_id") == document_id
            ]
            if not targets:
                return 0
            self._index.remove_ids(np.array(targets, dtype=np.int64))
            for fid in targets:
                rec = self._records.pop(fid)
                self._by_chunk.pop(rec["id"], None)
        self.persist()
        return len(targets)

    def reset(self) -> None:
        with self._lock:
            self._index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.dimension))
            self._records.clear()
            self._by_chunk.clear()
            self._next_id = 0
            for path in (self._index_path, self._meta_path):
                path.unlink(missing_ok=True)

    def search(
        self, vector: np.ndarray, k: int, *, document_ids: list[str] | None = None
    ) -> list[SearchHit]:
        with self._lock:
            if self._index.ntotal == 0 or k <= 0:
                return []
            query = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(1, -1))
            if query.shape[1] != self._index.d:
                raise VectorStoreError(
                    f"query has {query.shape[1]} dims, index has {self._index.d}"
                )
            fetch = min(self._index.ntotal, k * 8 if document_ids else k)
            scores, ids = self._index.search(query, fetch)

            allowed = set(document_ids) if document_ids else None
            hits: list[SearchHit] = []
            for score, fid in zip(scores[0], ids[0], strict=True):
                if fid == -1:
                    continue
                record = self._records.get(int(fid))
                if record is None:
                    continue
                if allowed and record["metadata"].get("document_id") not in allowed:
                    continue
                hits.append(
                    SearchHit(
                        chunk=self._deserialize_chunk(
                            record["id"], record["text"], record["metadata"]
                        ),
                        score=float(score),
                        rank=len(hits),
                    )
                )
                if len(hits) == k:
                    break
            return hits

    def count(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    def document_ids(self) -> set[str]:
        with self._lock:
            return {
                str(rec["metadata"].get("document_id"))
                for rec in self._records.values()
                if rec["metadata"].get("document_id")
            }
