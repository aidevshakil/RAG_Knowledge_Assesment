"""Benchmark runner.

Answers a labelled question suite and aggregates the results into one row. Two
uses:

1. **Before/after a knowledge-base change** — run the same suite, save both
   snapshots, and see whether adding or deleting files helped or hurt.
2. **Config comparison** — `compare_configs` sweeps embedding models, vector
   stores, dimensions and top-k against a *temporary* collection, so the live
   knowledge base is never disturbed.

Suite format (JSON or YAML-free dicts):

    [{"question": "...",
      "expected_sources": ["handbook.pdf"],   # optional, enables recall/nDCG
      "expected_answer_terms": ["30 days"]}]  # optional, answer correctness
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rag_assistant.config import EmbeddingConfig, Settings, VectorStoreConfig
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.types import JSONDict
from rag_assistant.evaluation import metrics as M

if TYPE_CHECKING:
    from rag_assistant.pipeline.engine import RAGEngine

logger = get_logger(__name__)


@dataclass(slots=True)
class BenchmarkCase:
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_answer_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: JSONDict | str) -> BenchmarkCase:
        if isinstance(data, str):
            return cls(question=data)
        return cls(
            question=data["question"],
            expected_sources=list(data.get("expected_sources", []) or []),
            expected_answer_terms=list(data.get("expected_answer_terms", []) or []),
        )


@dataclass(slots=True)
class BenchmarkResult:
    label: str
    fingerprint: str
    summary: JSONDict
    details: list[JSONDict]

    def to_dict(self) -> JSONDict:
        return asdict(self)


def load_suite(path: str | Path) -> list[BenchmarkCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("questions", payload) if isinstance(payload, dict) else payload
    return [BenchmarkCase.from_dict(case) for case in cases]


class BenchmarkRunner:
    def __init__(self, engine: RAGEngine) -> None:
        self.engine = engine

    def run(
        self,
        cases: Sequence[BenchmarkCase | JSONDict | str],
        *,
        label: str = "run",
        generate: bool = True,
        top_k: int | None = None,
        progress: Callable[[float, str], None] | None = None,
        persist: bool = True,
    ) -> BenchmarkResult:
        """Execute the suite and (optionally) store the snapshot."""
        parsed = [c if isinstance(c, BenchmarkCase) else BenchmarkCase.from_dict(c) for c in cases]
        details: list[JSONDict] = []

        for position, case in enumerate(parsed):
            if progress:
                progress(position / max(1, len(parsed)), case.question[:70])
            details.append(self._run_case(case, generate=generate, top_k=top_k))

        summary = self._aggregate(details)
        result = BenchmarkResult(
            label=label,
            fingerprint=self.engine.settings.fingerprint,
            summary=summary,
            details=details,
        )
        if persist:
            stats = self.engine.repository.document_stats()
            self.engine.repository.save_benchmark(
                label=label,
                fingerprint=result.fingerprint,
                doc_count=int(stats.get("documents", 0)),
                chunk_count=int(stats.get("chunks", 0)),
                summary=summary,
                details=details,
            )
        if progress:
            progress(1.0, "Done")
        return result

    def _run_case(self, case: BenchmarkCase, *, generate: bool, top_k: int | None) -> JSONDict:
        row: JSONDict = {"question": case.question}
        try:
            if generate:
                answer = self.engine.ask(case.question, top_k=top_k, log=False)
                hits = answer.hits
                row.update(
                    answer=answer.answer,
                    **answer.timings.to_dict(),
                    **answer.metrics.to_dict(),
                )
                if case.expected_answer_terms:
                    row["answer_correctness"] = M.answer_contains(
                        answer.answer, case.expected_answer_terms
                    )
            else:
                hits = self.engine.retrieve_only(case.question, top_k=top_k)
                mean_score, max_score = M.score_stats(hits)
                row.update(
                    mean_score=mean_score,
                    max_score=max_score,
                    precision_at_k=M.context_precision(
                        hits, self.engine.settings.evaluation.relevance_threshold
                    ),
                )
        except Exception as exc:
            logger.warning("Benchmark case failed (%s): %s", case.question[:50], exc)
            return {**row, "error": str(exc)}

        row["retrieved"] = [hit.chunk.citation for hit in hits]
        if case.expected_sources:
            row["hit_rate"] = M.hit_rate(hits, case.expected_sources)
            row["recall_at_k"] = M.recall_at_k(hits, case.expected_sources)
            row["mrr"] = M.mrr(hits, case.expected_sources)
            row["ndcg"] = M.ndcg(hits, case.expected_sources)
        return row

    @staticmethod
    def _aggregate(details: list[JSONDict]) -> JSONDict:
        numeric_keys = (
            "mean_score",
            "max_score",
            "precision_at_k",
            "faithfulness",
            "answer_relevance",
            "hit_rate",
            "recall_at_k",
            "mrr",
            "ndcg",
            "answer_correctness",
            "embed_ms",
            "retrieve_ms",
            "generate_ms",
            "total_ms",
        )
        summary: JSONDict = {
            "cases": len(details),
            "errors": sum(1 for d in details if "error" in d),
        }
        for key in numeric_keys:
            values = [d[key] for d in details if isinstance(d.get(key), (int, float))]
            if values:
                summary[key] = round(sum(values) / len(values), 4)
        totals = sorted(
            d["total_ms"] for d in details if isinstance(d.get("total_ms"), (int, float))
        )
        if totals:
            summary["p95_total_ms"] = round(
                totals[min(len(totals) - 1, int(0.95 * (len(totals) - 1)))], 2
            )
        risky = [d for d in details if d.get("hallucination_risk")]
        if details:
            summary["hallucination_rate"] = round(len(risky) / len(details), 4)
        return summary


@dataclass(slots=True)
class ConfigVariant:
    """One point in the comparison grid."""

    label: str
    embedding_provider: str | None = None
    embedding_model: str | None = None
    dimensions: int | None = None
    vector_store: str | None = None
    top_k: int | None = None

    def apply(self, base: Settings) -> Settings:
        embedding = EmbeddingConfig(
            provider=self.embedding_provider or base.embedding.provider,
            model=self.embedding_model or base.embedding.model,
            dimensions=self.dimensions
            if self.dimensions is not None
            else base.embedding.dimensions,
            batch_size=base.embedding.batch_size,
        )
        provider = self.vector_store or base.vector_store.provider
        vector_store = VectorStoreConfig(
            provider=provider,
            collection=f"bench_{_slug(self.label)}",
            directory=base.data_dir / "benchmarks" / provider,
        )
        retrieval = base.retrieval
        if self.top_k:
            retrieval = type(base.retrieval)(
                top_k=self.top_k,
                min_score=base.retrieval.min_score,
                mmr_lambda=base.retrieval.mmr_lambda,
                fetch_k_multiplier=base.retrieval.fetch_k_multiplier,
                max_context_chars=base.retrieval.max_context_chars,
            )
        return base.with_overrides(
            embedding=embedding, vector_store=vector_store, retrieval=retrieval
        )


def compare_configs(
    base_settings: Settings,
    variants: Sequence[ConfigVariant],
    cases: Sequence[BenchmarkCase | JSONDict | str],
    documents: Sequence[tuple[str, bytes]],
    *,
    generate: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> list[JSONDict]:
    """Index `documents` and run `cases` under each variant; one row per variant.

    Retrieval-only by default: it isolates the effect of the embedding model,
    dimensionality and store from LLM variance, and costs no API tokens.
    """
    from rag_assistant.embeddings.factory import build_embedder
    from rag_assistant.llm.factory import build_llm
    from rag_assistant.pipeline.engine import RAGEngine
    from rag_assistant.storage.repository import KnowledgeBaseRepository
    from rag_assistant.vectorstores.factory import build_vector_store

    rows: list[JSONDict] = []
    for position, variant in enumerate(variants):
        if progress:
            progress(position / max(1, len(variants)), f"Building {variant.label}")
        settings = variant.apply(base_settings)
        try:
            embedder = build_embedder(settings.embedding)
            store = build_vector_store(settings, embedder.dimension)
            store.reset()
            engine = RAGEngine(
                settings,
                embedder,
                store,
                build_llm(settings.llm),
                KnowledgeBaseRepository(settings.data_dir / "benchmarks" / "bench.sqlite3"),
            )
            engine.clear_knowledge_base()

            if progress:
                progress(position / max(1, len(variants)), f"Indexing for {variant.label}")
            ingestion = engine.ingest_files(list(documents), save_copy=False)

            result = BenchmarkRunner(engine).run(
                cases, label=variant.label, generate=generate, persist=False
            )
            rows.append(
                {
                    "variant": variant.label,
                    "embedding": embedder.label,
                    "dimension": embedder.dimension,
                    "vector_store": store.provider,
                    "top_k": settings.retrieval.top_k,
                    "index_ms": round(ingestion.duration_ms, 1),
                    "chunks": ingestion.chunks_added,
                    **result.summary,
                }
            )
            engine.close()
        except Exception as exc:
            logger.warning("Variant %s failed: %s", variant.label, exc)
            rows.append({"variant": variant.label, "error": str(exc)})
    if progress:
        progress(1.0, "Done")
    return rows


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:40]
