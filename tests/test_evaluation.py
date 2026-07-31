from __future__ import annotations

import numpy as np
import pytest

from rag_assistant.config import EvaluationConfig
from rag_assistant.core.types import Chunk, SearchHit
from rag_assistant.evaluation import metrics as M
from rag_assistant.evaluation.benchmark import BenchmarkRunner
from rag_assistant.evaluation.evaluator import Evaluator
from rag_assistant.retrieval.mmr import mmr_select


def _hit(text: str, score: float, document: str = "doc.txt", index: int = 0) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"{document}-{index}",
            document_id=document,
            document_name=document,
            text=text,
            index=index,
        ),
        score=score,
        rank=index,
    )


def test_context_precision_counts_relevant_hits():
    hits = [_hit("a", 0.8, index=0), _hit("b", 0.5, index=1), _hit("c", 0.1, index=2)]
    assert M.context_precision(hits, threshold=0.4) == 2 / 3
    assert M.context_precision([], threshold=0.4) == 0.0


def test_recall_and_ndcg_use_expected_sources():
    hits = [_hit("x", 0.9, "handbook.pdf"), _hit("y", 0.7, "notes.txt", 1)]
    assert M.recall_at_k(hits, ["handbook.pdf"]) == 1.0
    assert M.recall_at_k(hits, ["handbook.pdf", "missing.pdf"]) == 0.5
    assert M.mrr(hits, ["handbook.pdf"]) == 1.0
    assert M.mrr(hits, ["notes.txt"]) == 0.5
    assert M.ndcg(hits, ["handbook.pdf"]) > M.ndcg(hits, ["notes.txt"])


def test_ndcg_stays_within_range_when_one_label_matches_many_chunks():
    hits = [_hit("a", 0.9, "handbook.pdf", i) for i in range(3)]
    assert M.ndcg(hits, ["handbook.pdf"]) == pytest.approx(1.0)

    mixed = [_hit("a", 0.9, "notes.txt", 0), *[_hit("b", 0.8, "handbook.pdf", i) for i in (1, 2)]]
    assert 0.0 < M.ndcg(mixed, ["handbook.pdf"]) < 1.0


def test_grounded_answer_scores_higher_than_invented_one():
    hits = [_hit("Refunds are available within 30 days of purchase.", 0.9)]
    grounded = M.faithfulness_lexical("Refunds are available within 30 days of purchase.", hits)
    invented = M.faithfulness_lexical(
        "Our flagship spacecraft ships with unlimited quantum warranty coverage.", hits
    )
    assert grounded > 0.8
    assert invented < 0.4
    assert grounded > invented


def test_unsupported_sentences_are_flagged():
    hits = [_hit("The warranty lasts two years and covers manufacturing defects.", 0.9)]
    flagged = M.unsupported_sentences(
        "The warranty lasts two years. Free helicopter delivery is included worldwide.", hits
    )
    assert any("helicopter" in sentence for sentence in flagged)


def test_answer_relevance_is_cosine_similarity():
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert M.answer_relevance(vector, vector) == 1.0
    assert M.answer_relevance(vector, np.array([0.0, 1.0, 0.0], dtype=np.float32)) == 0.0


def test_evaluator_marks_hallucination_risk(engine):
    evaluator = Evaluator(EvaluationConfig(faithfulness_threshold=0.6), engine.embedder)
    hits = [_hit("Standard shipping takes three to seven business days.", 0.8)]
    result = evaluator.evaluate(
        "How long does shipping take?",
        "Shipping is completed by drone within eleven minutes for platinum subscribers.",
        hits,
    )
    assert result.hallucination_risk
    assert result.faithfulness < 0.6


def test_evaluator_treats_abstention_as_faithful(engine):
    evaluator = Evaluator(EvaluationConfig(), engine.embedder)
    result = evaluator.evaluate(
        "Unknown topic?",
        "I don't have information about that in the current knowledge base.",
        [_hit("unrelated text", 0.2)],
    )
    assert result.abstained
    assert result.faithfulness == 1.0
    assert not result.hallucination_risk


def test_mmr_prefers_diverse_candidates():
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = np.array([0.95, 0.94, 0.60], dtype=np.float32)

    relevance_only = mmr_select(vectors[0], vectors, scores, k=2, lambda_mult=1.0)
    diverse = mmr_select(vectors[0], vectors, scores, k=2, lambda_mult=0.3)
    assert relevance_only == [0, 1]
    assert diverse == [0, 2]


def test_benchmark_runner_aggregates_and_persists(loaded_engine):
    runner = BenchmarkRunner(loaded_engine)
    result = runner.run(
        [
            {"question": "What is the refund window?", "expected_sources": ["policy.txt"]},
            {"question": "When is support available?", "expected_sources": ["support.txt"]},
        ],
        label="baseline",
        generate=False,
    )
    assert result.summary["cases"] == 2
    assert "recall_at_k" in result.summary
    assert loaded_engine.repository.list_benchmarks()[0]["label"] == "baseline"


def test_benchmark_does_not_pollute_the_live_query_log(loaded_engine):
    before = len(loaded_engine.repository.recent_queries(limit=100))
    BenchmarkRunner(loaded_engine).run(
        [{"question": "What is the refund window?"}], generate=True, persist=False
    )
    assert len(loaded_engine.repository.recent_queries(limit=100)) == before
