"""Retrieval and answer quality evaluation."""

from rag_assistant.evaluation.benchmark import BenchmarkResult, BenchmarkRunner, load_suite
from rag_assistant.evaluation.evaluator import Evaluator
from rag_assistant.evaluation.metrics import (
    answer_relevance,
    context_precision,
    faithfulness_lexical,
    ndcg,
    recall_at_k,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "Evaluator",
    "answer_relevance",
    "context_precision",
    "faithfulness_lexical",
    "load_suite",
    "ndcg",
    "recall_at_k",
]
