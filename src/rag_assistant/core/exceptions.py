"""Exception hierarchy — one root so the UI can catch everything we raise."""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(RAGError):
    """Invalid or incomplete configuration (bad provider name, missing key)."""


class DependencyMissingError(ConfigurationError):
    """An optional third-party package is required but not installed."""

    def __init__(self, package: str, purpose: str, extra: str | None = None) -> None:
        hint = (
            f'pip install "rag-knowledge-assistant[{extra}]"' if extra else f"pip install {package}"
        )
        super().__init__(f"{purpose} requires the '{package}' package. Install it with: {hint}")
        self.package = package


class UnsupportedFileTypeError(RAGError):
    """No loader is registered for the given file extension."""


class DocumentLoadError(RAGError):
    """A document could not be parsed or produced no extractable text."""


class VectorStoreError(RAGError):
    """The vector store rejected an operation."""


class DimensionMismatchError(VectorStoreError):
    """Embedding width does not match the existing collection."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"This collection stores {expected}-dimensional vectors but the current "
            f"embedding model produces {actual}. Switch back to the original model, "
            f"use a different collection name, or reset the knowledge base."
        )
        self.expected = expected
        self.actual = actual


class LLMError(RAGError):
    """The LLM backend failed or returned nothing usable."""


class RetrievalError(RAGError):
    """Retrieval could not be completed."""
