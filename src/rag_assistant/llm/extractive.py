"""Zero-dependency, zero-key "LLM": extractive summarisation of the context.

It scores each sentence of the retrieved context against the question and
returns the best ones verbatim with citations. That makes it useless for
synthesis but perfect as a fallback: the app installs and demos with no API key,
and it is by construction 100% faithful — a helpful baseline to compare a real
model's faithfulness score against.
"""

from __future__ import annotations

import re

from rag_assistant.core.text import content_words, split_sentences
from rag_assistant.llm.base import LLMS, LLMClient

_SOURCE_BLOCK = re.compile(r"^\[(\d+)\]\s*(?:Source:\s*)?(.*)$", re.M)
_MAX_SENTENCES = 5


@LLMS.register("extractive", "offline", "none")
class ExtractiveLLM(LLMClient):
    provider = "extractive"
    requires_api_key = False

    @property
    def label(self) -> str:
        return "extractive:sentence-selection"

    def _complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> tuple[str, int | None, int | None]:
        question, context = _split_prompt(user)
        query_terms = content_words(question)

        scored: list[tuple[float, str, str]] = []
        for label, body in _iter_sources(context):
            for sentence in split_sentences(body):
                terms = content_words(sentence)
                if not terms:
                    continue
                overlap = len(terms & query_terms)
                if not overlap:
                    continue
                scored.append((overlap / (len(terms) ** 0.5), sentence, label))

        if not scored:
            return (
                "I could not find an answer to that in the current knowledge base.",
                None,
                None,
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        lines: list[str] = []
        for _, sentence, label in scored:
            key = sentence[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {sentence} [{label}]")
            if len(lines) == _MAX_SENTENCES:
                break

        return "Based on the retrieved documents:\n\n" + "\n".join(lines), None, None


def _split_prompt(user: str) -> tuple[str, str]:
    """Recover (question, context) from the generation prompt."""
    marker = "Question:"
    index = user.rfind(marker)
    if index == -1:
        return user, user
    return user[index + len(marker) :].strip(), user[:index]


def _iter_sources(context: str) -> list[tuple[str, str]]:
    """Yield (citation label, body) pairs from the numbered context block."""
    matches = list(_SOURCE_BLOCK.finditer(context))
    if not matches:
        return [("context", context)]
    sources: list[tuple[str, str]] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(context)
        body = context[match.end() : end].strip()
        sources.append((match.group(1), body))
    return sources
