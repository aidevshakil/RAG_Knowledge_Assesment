"""LLM backends behind one interface. `build_llm(config)` is the entry point."""

from rag_assistant.llm import (  # noqa: F401  (registration side effects)
    anthropic,
    extractive,
    groq,
    openai,
)
from rag_assistant.llm.base import LLMS, LLMClient
from rag_assistant.llm.factory import available_llms, build_llm

__all__ = ["LLMS", "LLMClient", "available_llms", "build_llm"]
