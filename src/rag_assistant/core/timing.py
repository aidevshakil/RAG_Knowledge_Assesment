"""Millisecond timers used by the performance monitor."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class Stopwatch:
    """Accumulates named durations: `sw.mark("embed")` then `sw["embed"]`."""

    laps: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    @contextmanager
    def mark(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.laps[name] = self.laps.get(name, 0.0) + elapsed

    def __getitem__(self, name: str) -> float:
        return self.laps.get(name, 0.0)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


@contextmanager
def timed() -> Iterator[list[float]]:
    """`with timed() as t: ...` — afterwards `t[0]` holds elapsed milliseconds."""
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = (time.perf_counter() - start) * 1000.0
