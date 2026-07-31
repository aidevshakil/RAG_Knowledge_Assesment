"""OpenAI embeddings (`text-embedding-3-*` support native dimension shortening)."""

from __future__ import annotations

import os

import numpy as np

from rag_assistant.config import EmbeddingConfig
from rag_assistant.core.exceptions import ConfigurationError, DependencyMissingError
from rag_assistant.embeddings.base import EMBEDDERS, Embedder

_NATIVE_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_SUPPORTS_SHORTENING = ("text-embedding-3-",)


@EMBEDDERS.register("openai")
class OpenAIEmbedder(Embedder):
    provider = "openai"

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DependencyMissingError("openai", "OpenAI embeddings", extra="openai") from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI embeddings")
        self._client = OpenAI(api_key=api_key)
        self._native = _NATIVE_DIMS.get(config.model, 1536)
        self._server_side_dims = (
            config.dimensions
            if config.dimensions and config.model.startswith(_SUPPORTS_SHORTENING)
            else None
        )

    @property
    def native_dimension(self) -> int:
        return self._native

    def _encode(self, texts: list[str]) -> np.ndarray:
        kwargs = {"model": self.config.model, "input": texts}
        if self._server_side_dims:
            kwargs["dimensions"] = self._server_side_dims
        response = self._client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.array([item.embedding for item in ordered], dtype=np.float32)
