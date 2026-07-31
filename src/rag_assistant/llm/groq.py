"""Groq backend — fastest inference of the hosted options, OpenAI-style API."""

from __future__ import annotations

import os
from collections.abc import Iterator

from rag_assistant.config import LLMConfig
from rag_assistant.core.exceptions import ConfigurationError, DependencyMissingError
from rag_assistant.llm.base import LLMS, LLMClient


@LLMS.register("groq")
class GroqLLM(LLMClient):
    provider = "groq"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from groq import Groq
        except ImportError as exc:
            raise DependencyMissingError("groq", "the Groq backend", extra="groq") from exc
        api_key = config.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys"
            )
        self._client = Groq(api_key=api_key, timeout=config.timeout)

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> tuple[str, int | None, int | None]:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=self._messages(system, user),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        return (
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=self._messages(system, user),
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta
