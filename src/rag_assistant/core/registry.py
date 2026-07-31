"""A tiny plugin registry.

Every swappable backend registers itself under a name:

    @EMBEDDERS.register("openai")
    class OpenAIEmbedder(Embedder): ...

Adding a new vector store or LLM therefore means writing one class and one
decorator line — no factory `if/elif` chain to edit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from rag_assistant.core.exceptions import ConfigurationError

T = TypeVar("T")


class Registry(Generic[T]):
    __slots__ = ("_kind", "_items")

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, *names: str) -> Callable[[type[T]], type[T]]:
        if not names:
            raise ValueError("at least one name is required")

        def decorator(cls: type[T]) -> type[T]:
            for name in names:
                self._items[name.lower()] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name.lower().strip()]
        except KeyError:
            raise ConfigurationError(
                f"Unknown {self._kind} '{name}'. Available: {', '.join(self.names)}"
            ) from None

    @property
    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower().strip() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self._items)
