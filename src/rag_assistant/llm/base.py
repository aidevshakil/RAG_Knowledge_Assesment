"""LLM client interface.

Only one capability is required — `complete(system, user) -> LLMResult` — since
that is all a RAG answer needs. Streaming is optional and defaults to yielding
the completed text in one piece, so the UI can always use the streaming path.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

from rag_assistant.config import LLMConfig
from rag_assistant.core.exceptions import LLMError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.registry import Registry
from rag_assistant.core.types import LLMResult

logger = get_logger(__name__)

LLMS: Registry[LLMClient] = Registry("LLM provider")

_MAX_ATTEMPTS = 3
_RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "overloaded",
    "connection",
    "temporarily",
)


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


class LLMClient(ABC):
    provider: str = "base"
    requires_api_key: bool = True

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def _complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> tuple[str, int | None, int | None]:
        """Return (text, prompt_tokens, completion_tokens)."""

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        max_tokens = max_tokens or self.config.max_tokens
        temperature = self.config.temperature if temperature is None else temperature
        start = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                text, prompt_tokens, completion_tokens = self._complete(
                    system, user, max_tokens=max_tokens, temperature=temperature
                )
            except Exception as exc:
                last_error = exc
                if attempt == _MAX_ATTEMPTS or not _is_retryable(exc):
                    break
                delay = min(8.0, 0.75 * 2 ** (attempt - 1)) * (0.5 + random.random())
                logger.warning(
                    "%s call failed (attempt %d/%d): %s — retrying in %.1fs",
                    self.provider,
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            if not text.strip():
                raise LLMError(f"{self.provider} returned an empty response")
            return LLMResult(
                text=text.strip(),
                model=self.config.model,
                provider=self.provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        raise LLMError(f"{self.provider} call failed: {last_error}") from last_error

    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Token stream. Backends that can't stream yield the full text once."""
        yield self.complete(system, user, max_tokens=max_tokens, temperature=temperature).text

    @property
    def supports_streaming(self) -> bool:
        return type(self).stream is not LLMClient.stream

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.config.model}"
