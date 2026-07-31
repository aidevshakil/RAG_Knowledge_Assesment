"""Prompt templates.

Grounding rules live here, in one place, because they are the main lever on
hallucination rate — the metric the performance tab tracks.
"""

from __future__ import annotations

from rag_assistant.core.text import truncate
from rag_assistant.core.types import SearchHit

ABSTAIN_MESSAGE = (
    "I don't have information about that in the current knowledge base. "
    "Try rephrasing the question, or add a document that covers it."
)

RAG_SYSTEM_PROMPT = f"""You are a knowledge assistant that answers strictly from
the provided sources.

Rules:
1. Use ONLY the numbered sources below. Never rely on outside knowledge.
2. Cite the sources you use inline as [1], [2] — every factual claim needs one.
3. If the sources don't contain the answer, say exactly: "{ABSTAIN_MESSAGE}" — do not guess.
4. If the sources conflict, present both readings and attribute each.
5. Be concise and direct. Use short paragraphs or bullets; skip preambles.
6. Quote exact figures, names and dates as they appear; never round or invent them.
"""

_CONTEXT_TEMPLATE = """[{number}] Source: {citation}
{text}"""

_USER_TEMPLATE = """Sources:
{context}

{history}Question: {question}"""

JUDGE_SYSTEM_PROMPT = """You grade the faithfulness of an answer against its sources.

Return exactly three lines, nothing else:
VERDICT: supported | partially_supported | unsupported
SCORE: <number between 0.0 and 1.0>
REASON: <one short sentence>

"supported" means every claim traces to the sources. "unsupported" means the
answer introduces facts the sources do not contain."""

_JUDGE_TEMPLATE = """Sources:
{context}

Question: {question}

Answer to grade:
{answer}"""


def format_context(hits: list[SearchHit], *, max_chars_per_hit: int = 2400) -> str:
    """Render hits as the numbered source block the prompts refer to."""
    blocks = [
        _CONTEXT_TEMPLATE.format(
            number=index,
            citation=hit.chunk.citation,
            text=truncate(hit.text.strip(), max_chars_per_hit),
        )
        for index, hit in enumerate(hits, start=1)
    ]
    return "\n\n".join(blocks) if blocks else "(no sources retrieved)"


def _format_history(history: list[tuple[str, str]] | None, max_turns: int = 3) -> str:
    """Recent turns, so follow-up questions resolve pronouns correctly."""
    if not history:
        return ""
    lines = [
        f"Q: {question}\nA: {truncate(answer, 400)}" for question, answer in history[-max_turns:]
    ]
    return "Earlier in this conversation:\n" + "\n\n".join(lines) + "\n\n"


def build_generation_prompt(
    question: str,
    hits: list[SearchHit],
    *,
    history: list[tuple[str, str]] | None = None,
) -> str:
    return _USER_TEMPLATE.format(
        context=format_context(hits),
        history=_format_history(history),
        question=question.strip(),
    )


def build_judge_prompt(question: str, answer: str, hits: list[SearchHit]) -> str:
    return _JUDGE_TEMPLATE.format(
        context=format_context(hits, max_chars_per_hit=1200),
        question=question.strip(),
        answer=answer.strip(),
    )
