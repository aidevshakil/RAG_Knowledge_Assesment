"""Per-query evaluation.

Runs on every answer and fills `QualityMetrics`. Cheap by default (embedding
reuse + lexical overlap); the optional LLM judge adds one extra call and a
graded verdict.
"""

from __future__ import annotations

import re

import numpy as np

from rag_assistant.config import EvaluationConfig
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.types import QualityMetrics, SearchHit
from rag_assistant.embeddings.base import Embedder
from rag_assistant.evaluation import metrics as M
from rag_assistant.generation.prompts import (
    ABSTAIN_MESSAGE,
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)
from rag_assistant.llm.base import LLMClient

logger = get_logger(__name__)

_VERDICT = re.compile(r"VERDICT:\s*(\w+)", re.I)
_SCORE = re.compile(r"SCORE:\s*([01](?:\.\d+)?)", re.I)
_REASON = re.compile(r"REASON:\s*(.+)", re.I)

_ABSTAIN_MARKERS = (
    ABSTAIN_MESSAGE[:40].lower(),
    "i don't have information",
    "i do not have information",
    "not contain",
    "could not find an answer",
    "no relevant information",
)


class Evaluator:
    def __init__(
        self,
        config: EvaluationConfig,
        embedder: Embedder,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.llm = llm

    def evaluate(
        self,
        question: str,
        answer: str,
        hits: list[SearchHit],
        *,
        query_vector: np.ndarray | None = None,
    ) -> QualityMetrics:
        mean_score, max_score = M.score_stats(hits)
        abstained = self._is_abstention(answer)

        result = QualityMetrics(
            mean_score=mean_score,
            max_score=max_score,
            precision_at_k=M.context_precision(hits, self.config.relevance_threshold),
            context_diversity=M.context_diversity(hits),
            abstained=abstained,
        )

        if abstained or not hits:
            result.faithfulness = 1.0 if abstained else 0.0
            result.answer_relevance = 0.0
            result.hallucination_risk = False
            return result

        result.faithfulness = M.faithfulness_lexical(answer, hits)
        result.answer_relevance = self._answer_relevance(question, answer, query_vector)
        result.hallucination_risk = result.faithfulness < self.config.faithfulness_threshold

        if self.config.enable_llm_judge and self.llm is not None:
            self._apply_judge(question, answer, hits, result)
        return result

    def unsupported_claims(self, answer: str, hits: list[SearchHit]) -> list[str]:
        return M.unsupported_sentences(answer, hits)

    @staticmethod
    def _is_abstention(answer: str) -> bool:
        lowered = answer.lower()
        return any(marker in lowered for marker in _ABSTAIN_MARKERS)

    def _answer_relevance(
        self, question: str, answer: str, query_vector: np.ndarray | None
    ) -> float:
        try:
            if query_vector is None:
                query_vector = self.embedder.embed_query(question)
            answer_vector = self.embedder.embed_query(answer[:2000])
            return M.answer_relevance(query_vector, answer_vector)
        except Exception as exc:
            logger.warning("answer_relevance failed: %s", exc)
            return 0.0

    def _apply_judge(
        self, question: str, answer: str, hits: list[SearchHit], result: QualityMetrics
    ) -> None:
        try:
            judgement = self.llm.complete(  # type: ignore[union-attr]
                JUDGE_SYSTEM_PROMPT,
                build_judge_prompt(question, answer, hits),
                max_tokens=200,
                temperature=0.0,
            ).text
        except Exception as exc:
            logger.warning("LLM judge failed: %s", exc)
            return

        verdict = match.group(1).lower() if (match := _VERDICT.search(judgement)) else "unknown"
        reason = match.group(1).strip() if (match := _REASON.search(judgement)) else ""
        result.judge_verdict = f"{verdict}: {reason}" if reason else verdict

        if match := _SCORE.search(judgement):
            judge_score = float(match.group(1))
            result.faithfulness = (result.faithfulness + judge_score) / 2
        result.hallucination_risk = (
            verdict == "unsupported" or result.faithfulness < self.config.faithfulness_threshold
        )
