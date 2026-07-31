"""Anthropic (Claude) backend.

Note the shape difference from the OpenAI-style APIs: the system prompt is a
top-level parameter rather than a message, and `max_tokens` is required.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from rag_assistant.config import LLMConfig
from rag_assistant.core.exceptions import ConfigurationError, DependencyMissingError
from rag_assistant.llm.base import LLMS, LLMClient


@LLMS.register("anthropic", "claude")
class AnthropicLLM(LLMClient):
    provider = "anthropic"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            import anthropic as anthropic_sdk
        except ImportError as exc:
            raise DependencyMissingError(
                "anthropic", "the Anthropic backend", extra="anthropic"
            ) from exc
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic_sdk.Anthropic(api_key=api_key, timeout=config.timeout)

    def _complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> tuple[str, int | None, int | None]:
        response = self._client.messages.create(
            model=self.config.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = getattr(response, "usage", None)
        return (
            text,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )

    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.config.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        ) as stream:
            yield from stream.text_stream
