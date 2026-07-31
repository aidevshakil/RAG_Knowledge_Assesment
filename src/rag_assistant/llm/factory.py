"""Build an LLM client, with a graceful fallback when no key is configured."""

from __future__ import annotations

from rag_assistant.config import LLMConfig
from rag_assistant.core.exceptions import ConfigurationError, RAGError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.llm.base import LLMS, LLMClient

logger = get_logger(__name__)


def build_llm(config: LLMConfig, *, allow_fallback: bool = True) -> LLMClient:
    """Instantiate the configured backend.

    With `allow_fallback`, a missing key or missing SDK degrades to the
    extractive backend instead of breaking the app — retrieval and the
    performance dashboard stay fully usable offline.
    """
    try:
        return LLMS.get(config.provider)(config)
    except (ConfigurationError, RAGError) as exc:
        if not allow_fallback or config.provider == "extractive":
            raise
        logger.warning(
            "%s unavailable (%s) — falling back to extractive answers", config.provider, exc
        )
        from rag_assistant.llm.extractive import ExtractiveLLM

        return ExtractiveLLM(
            LLMConfig(
                provider="extractive",
                model="extractive-summarizer",
                temperature=0.0,
                max_tokens=config.max_tokens,
            )
        )


def available_llms() -> list[str]:
    return LLMS.names
