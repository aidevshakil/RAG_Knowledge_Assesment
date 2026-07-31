"""Domain types shared by every layer.

These are plain dataclasses with no third-party imports, so the contract
between ingestion, retrieval, generation and evaluation stays stable even if a
backend library is swapped out.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

JSONDict = dict[str, Any]


def new_id() -> str:
    return uuid.uuid4().hex


def now_ts() -> float:
    return time.time()


class SourceType(StrEnum):
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "md"
    DOCX = "docx"
    CSV = "csv"
    HTML = "html"
    URL = "url"
    RAW = "raw"


@dataclass(slots=True)
class Document:
    """A source of knowledge, identified by content hash so re-uploads dedupe."""

    id: str
    name: str
    source_type: str
    content_hash: str
    size_bytes: int = 0
    char_count: int = 0
    chunk_count: int = 0
    path: str | None = None
    uri: str | None = None
    metadata: JSONDict = field(default_factory=dict)
    created_at: float = field(default_factory=now_ts)

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    """An embeddable slice of a document."""

    id: str
    document_id: str
    document_name: str
    text: str
    index: int
    metadata: JSONDict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        page = self.metadata.get("page")
        locator = f"p.{page}" if page else f"#{self.index + 1}"
        return f"{self.document_name} ({locator})"

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    """A retrieved chunk plus its similarity score (cosine, higher is better)."""

    chunk: Chunk
    score: float
    rank: int = 0

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> JSONDict:
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "document_name": self.chunk.document_name,
            "citation": self.chunk.citation,
            "score": round(self.score, 4),
            "rank": self.rank,
            "text": self.chunk.text,
        }


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    truncated: bool = False


@dataclass(slots=True)
class Timings:
    """Latency breakdown in milliseconds — surfaced in the performance panel."""

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    evaluate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.retrieve_ms + self.generate_ms + self.evaluate_ms

    def to_dict(self) -> JSONDict:
        return {
            "embed_ms": round(self.embed_ms, 2),
            "retrieve_ms": round(self.retrieve_ms, 2),
            "generate_ms": round(self.generate_ms, 2),
            "evaluate_ms": round(self.evaluate_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass(slots=True)
class QualityMetrics:
    """Per-query quality signals computed by `evaluation.evaluator`."""

    mean_score: float = 0.0
    max_score: float = 0.0
    precision_at_k: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_diversity: int = 0
    hallucination_risk: bool = False
    abstained: bool = False
    judge_verdict: str | None = None

    def to_dict(self) -> JSONDict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 4)
        return data


@dataclass(slots=True)
class RAGAnswer:
    """Everything a single query produced — the unit stored and displayed."""

    id: str
    question: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    llm: LLMResult | None = None
    config_fingerprint: str = ""
    created_at: float = field(default_factory=now_ts)

    @property
    def citations(self) -> list[str]:
        seen: dict[str, None] = {}
        for hit in self.hits:
            seen.setdefault(hit.chunk.citation, None)
        return list(seen)

    def to_dict(self) -> JSONDict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "hits": [hit.to_dict() for hit in self.hits],
            "timings": self.timings.to_dict(),
            "metrics": self.metrics.to_dict(),
            "llm": asdict(self.llm) if self.llm else None,
            "config_fingerprint": self.config_fingerprint,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class IngestionResult:
    """Outcome of one ingestion run, per source."""

    documents: list[Document] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [f"{len(self.documents)} document(s), {self.chunks_added} chunk(s) indexed"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return " · ".join(parts) + f" in {self.duration_ms / 1000:.1f}s"
