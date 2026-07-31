"""Query-time retrieval.

Pipeline: embed query → over-fetch candidates → drop low-similarity noise →
MMR re-rank for diversity → trim to the context budget.

Each stage is optional via config, which is what makes the retrieval-quality
comparisons in the benchmark tab meaningful.
"""

from __future__ import annotations

import numpy as np

from rag_assistant.config import RetrievalConfig
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.timing import timed
from rag_assistant.core.types import SearchHit
from rag_assistant.embeddings.base import Embedder
from rag_assistant.retrieval.mmr import mmr_select
from rag_assistant.vectorstores.base import VectorStore

logger = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        config: RetrievalConfig,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        min_score: float | None = None,
        use_mmr: bool = True,
    ) -> tuple[list[SearchHit], np.ndarray, float, float]:
        """Return (hits, query_vector, embed_ms, retrieve_ms)."""
        cfg = self.config
        top_k = top_k or cfg.top_k
        min_score = cfg.min_score if min_score is None else min_score

        with timed() as embed_ms:
            query_vector = self.embedder.embed_query(question)

        with timed() as retrieve_ms:
            fetch_k = max(top_k, top_k * cfg.fetch_k_multiplier) if use_mmr else top_k
            candidates = self.store.search(query_vector, fetch_k, document_ids=document_ids)

            if min_score > 0:
                kept = [hit for hit in candidates if hit.score >= min_score]
                if not kept:
                    logger.debug("no hit cleared min_score=%.2f; using raw ranking", min_score)
                candidates = kept or candidates

            hits = self._rerank(query_vector, candidates, top_k) if use_mmr else candidates[:top_k]
            hits = self._apply_context_budget(hits)
            for rank, hit in enumerate(hits):
                hit.rank = rank

        logger.debug(
            "retrieved %d/%d hits for %r (embed %.1fms, search %.1fms)",
            len(hits),
            len(candidates),
            question[:60],
            embed_ms[0],
            retrieve_ms[0],
        )
        return hits, query_vector, embed_ms[0], retrieve_ms[0]

    def _rerank(
        self, query_vector: np.ndarray, candidates: list[SearchHit], top_k: int
    ) -> list[SearchHit]:
        if len(candidates) <= top_k or self.config.mmr_lambda >= 1.0:
            return candidates[:top_k]
        vectors = self.embedder.embed_documents([hit.text for hit in candidates])
        scores = np.array([hit.score for hit in candidates], dtype=np.float32)
        order = mmr_select(query_vector, vectors, scores, top_k, self.config.mmr_lambda)
        return [candidates[i] for i in order]

    def _apply_context_budget(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Drop the weakest tail if the context would overflow the budget."""
        budget = self.config.max_context_chars
        kept: list[SearchHit] = []
        used = 0
        for hit in hits:
            length = len(hit.text)
            if kept and used + length > budget:
                break
            kept.append(hit)
            used += length
        return kept
