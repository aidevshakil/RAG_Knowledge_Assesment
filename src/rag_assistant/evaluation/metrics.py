"""Metric primitives.

Two families:

* **Reference-free** (`context_precision`, `faithfulness_lexical`,
  `answer_relevance`) — computable on every live query, which is what the
  performance monitor uses.
* **Reference-based** (`recall_at_k`, `ndcg`, `mrr`) — need a labelled suite of
  question → expected-source pairs, used by the benchmark runner.
"""

from __future__ import annotations

import math

import numpy as np

from rag_assistant.core.text import content_words, coverage, split_sentences
from rag_assistant.core.types import SearchHit


def context_precision(hits: list[SearchHit], threshold: float) -> float:
    """Share of retrieved chunks whose similarity clears `threshold`.

    A proxy for "are the top-k chunks actually relevant?" — low values mean the
    retriever is padding the context with noise, which both costs tokens and
    invites hallucination.
    """
    if not hits:
        return 0.0
    return sum(1 for hit in hits if hit.score >= threshold) / len(hits)


def score_stats(hits: list[SearchHit]) -> tuple[float, float]:
    """(mean, max) similarity of the returned hits."""
    if not hits:
        return 0.0, 0.0
    scores = [hit.score for hit in hits]
    return sum(scores) / len(scores), max(scores)


def faithfulness_lexical(answer: str, hits: list[SearchHit]) -> float:
    """Fraction of the answer's content words that appear in the context.

    Cheap, deterministic and surprisingly effective at catching invented
    specifics (names, numbers, product features). It under-scores heavy
    paraphrase, so use the LLM judge when precision matters.
    """
    if not answer.strip() or not hits:
        return 0.0
    context_terms = content_words(" ".join(hit.text for hit in hits))
    if not context_terms:
        return 0.0
    sentences = split_sentences(answer) or [answer]
    grounded: list[float] = []
    for sentence in sentences:
        terms = content_words(sentence)
        if len(terms) < 3:
            continue
        grounded.append(coverage(terms, context_terms))
    if not grounded:
        return coverage(content_words(answer), context_terms)
    return sum(grounded) / len(grounded)


def unsupported_sentences(answer: str, hits: list[SearchHit], threshold: float = 0.4) -> list[str]:
    """Sentences whose grounding falls below `threshold` — shown in the UI."""
    if not hits:
        return []
    context_terms = content_words(" ".join(hit.text for hit in hits))
    flagged = []
    for sentence in split_sentences(answer):
        terms = content_words(sentence)
        if len(terms) >= 4 and coverage(terms, context_terms) < threshold:
            flagged.append(sentence)
    return flagged


def answer_relevance(query_vector: np.ndarray, answer_vector: np.ndarray) -> float:
    """Cosine similarity between question and answer embeddings.

    Detects the "fluent but off-topic" failure mode that faithfulness misses:
    an answer can be perfectly grounded in the context and still not respond to
    what was asked.
    """
    if query_vector is None or answer_vector is None:
        return 0.0
    query = np.asarray(query_vector, dtype=np.float32).ravel()
    answer = np.asarray(answer_vector, dtype=np.float32).ravel()
    if query.size == 0 or answer.size != query.size:
        return 0.0
    denominator = float(np.linalg.norm(query) * np.linalg.norm(answer))
    return float(np.dot(query, answer) / denominator) if denominator else 0.0


def context_diversity(hits: list[SearchHit]) -> int:
    """How many distinct documents backed the answer."""
    return len({hit.chunk.document_id for hit in hits})


def _matches(hit: SearchHit, expected: str) -> bool:
    """A hit counts as relevant if the label matches its document or its text."""
    needle = expected.lower().strip()
    if not needle:
        return False
    return (
        needle in hit.chunk.document_name.lower()
        or needle == hit.chunk.document_id.lower()
        or needle in hit.text.lower()
    )


def hit_rate(hits: list[SearchHit], expected: list[str]) -> float:
    """1.0 if any expected source appears in the results, else 0.0."""
    if not expected:
        return 0.0
    return 1.0 if any(_matches(hit, exp) for hit in hits for exp in expected) else 0.0


def recall_at_k(hits: list[SearchHit], expected: list[str]) -> float:
    """Share of expected sources found anywhere in the top-k."""
    if not expected:
        return 0.0
    found = sum(1 for exp in expected if any(_matches(hit, exp) for hit in hits))
    return found / len(expected)


def mrr(hits: list[SearchHit], expected: list[str]) -> float:
    """Mean reciprocal rank of the first relevant hit (rank sensitivity)."""
    for position, hit in enumerate(hits, start=1):
        if any(_matches(hit, exp) for exp in expected):
            return 1.0 / position
    return 0.0


def ndcg(hits: list[SearchHit], expected: list[str]) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not expected or not hits:
        return 0.0
    gains = [1.0 if any(_matches(hit, exp) for exp in expected) else 0.0 for hit in hits]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    relevant = int(sum(gains))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, relevant + 1))
    return dcg / ideal if ideal else 0.0


def answer_contains(answer: str, expected_terms: list[str]) -> float:
    """Share of expected key facts present in the answer (correctness proxy)."""
    if not expected_terms:
        return 0.0
    lowered = answer.lower()
    return sum(1 for term in expected_terms if term.lower().strip() in lowered) / len(
        expected_terms
    )
