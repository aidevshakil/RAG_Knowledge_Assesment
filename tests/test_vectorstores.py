"""Contract tests — every registered store must behave identically.

Backends that aren't installed are skipped rather than failing the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_assistant.core.types import Chunk, new_id
from rag_assistant.embeddings.base import l2_normalize
from rag_assistant.vectorstores.base import VECTOR_STORES

DIM = 32
BACKENDS = ["numpy", "chroma", "faiss"]


def _store(name: str, tmp_path):
    try:
        return VECTOR_STORES.get(name)(
            collection="contract", dimension=DIM, directory=tmp_path / name
        )
    except Exception as exc:
        pytest.skip(f"{name} unavailable: {exc}")


def _chunks(document_id: str, count: int, offset: int = 0) -> tuple[list[Chunk], np.ndarray]:
    chunks = [
        Chunk(
            id=new_id(),
            document_id=document_id,
            document_name=f"{document_id}.txt",
            text=f"chunk {offset + i} of {document_id}",
            index=i,
        )
        for i in range(count)
    ]
    rng = np.random.default_rng(offset + 7)
    return chunks, l2_normalize(rng.normal(size=(count, DIM)).astype(np.float32))


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_search_and_count(backend, tmp_path):
    store = _store(backend, tmp_path)
    chunks, vectors = _chunks("docA", 5)
    assert store.add(chunks, vectors) == 5
    assert store.count() == 5

    hits = store.search(vectors[2], k=3)
    assert hits, "expected results"
    assert hits[0].chunk.id == chunks[2].id, "the exact vector must rank first"
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits == sorted(hits, key=lambda h: -h.score)


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_document_removes_vectors(backend, tmp_path):
    """The core requirement: deleted files must become unretrievable."""
    store = _store(backend, tmp_path)
    a_chunks, a_vectors = _chunks("docA", 4)
    b_chunks, b_vectors = _chunks("docB", 3, offset=100)
    store.add(a_chunks, a_vectors)
    store.add(b_chunks, b_vectors)
    assert store.count() == 7

    removed = store.delete_document("docA")
    assert removed == 4
    assert store.count() == 3
    assert store.document_ids() == {"docB"}

    hits = store.search(a_vectors[0], k=10)
    assert all(hit.chunk.document_id == "docB" for hit in hits)


@pytest.mark.parametrize("backend", BACKENDS)
def test_metadata_filter_scopes_results(backend, tmp_path):
    store = _store(backend, tmp_path)
    a_chunks, a_vectors = _chunks("docA", 4)
    b_chunks, b_vectors = _chunks("docB", 4, offset=50)
    store.add(a_chunks, a_vectors)
    store.add(b_chunks, b_vectors)

    hits = store.search(a_vectors[0], k=5, document_ids=["docB"])
    assert hits
    assert {hit.chunk.document_id for hit in hits} == {"docB"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_upsert_does_not_duplicate(backend, tmp_path):
    store = _store(backend, tmp_path)
    chunks, vectors = _chunks("docA", 3)
    store.add(chunks, vectors)
    store.add(chunks, vectors)
    assert store.count() == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_persistence_across_instances(backend, tmp_path):
    store = _store(backend, tmp_path)
    chunks, vectors = _chunks("docA", 3)
    store.add(chunks, vectors)
    store.close()

    reopened = _store(backend, tmp_path)
    assert reopened.count() == 3
    assert reopened.document_ids() == {"docA"}
    assert reopened.search(vectors[0], k=1)[0].chunk.document_id == "docA"


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset_clears_everything(backend, tmp_path):
    store = _store(backend, tmp_path)
    chunks, vectors = _chunks("docA", 3)
    store.add(chunks, vectors)
    store.reset()
    assert store.count() == 0
    assert store.search(vectors[0], k=3) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_dimension_mismatch_is_rejected(backend, tmp_path):
    from rag_assistant.core.exceptions import DimensionMismatchError

    store = _store(backend, tmp_path)
    chunks, _ = _chunks("docA", 2)
    with pytest.raises(DimensionMismatchError):
        store.add(chunks, np.zeros((2, DIM + 8), dtype=np.float32))


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_store_returns_no_hits(backend, tmp_path):
    store = _store(backend, tmp_path)
    assert store.search(np.zeros(DIM, dtype=np.float32), k=5) == []
    assert store.document_ids() == set()
