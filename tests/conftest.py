"""Shared fixtures.

Every test runs fully offline: the hashing embedder needs no downloads, the
numpy store needs no server, and the extractive LLM needs no key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.config import (  # noqa: E402
    EmbeddingConfig,
    EvaluationConfig,
    LLMConfig,
    Settings,
    VectorStoreConfig,
)
from rag_assistant.pipeline.engine import RAGEngine, build_engine  # noqa: E402

SAMPLE_TEXT = """
Refund policy

Customers may request a full refund within 30 days of purchase. Refunds are
processed to the original payment method within five business days.

Shipping

Standard shipping takes three to seven business days. Express shipping is
delivered the next business day and costs an additional twelve euros.

Warranty

All hardware is covered by a two year limited warranty against manufacturing
defects. The warranty does not cover accidental damage or water ingress.
"""

OTHER_TEXT = """
Support hours

The support desk is staffed Monday to Friday, from nine in the morning until
six in the evening, Central European Time. Weekend support is available to
enterprise customers only.
"""


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding=EmbeddingConfig(provider="hashing", model="hashing-bow", dimensions=256),
        vector_store=VectorStoreConfig(
            provider="numpy", collection="test", directory=tmp_path / "vectors"
        ),
        llm=LLMConfig(provider="extractive", model="extractive-summarizer"),
        evaluation=EvaluationConfig(enable_llm_judge=False),
        data_dir=tmp_path,
    )


@pytest.fixture()
def engine(settings: Settings) -> RAGEngine:
    engine = build_engine(settings)
    yield engine
    engine.close()


@pytest.fixture()
def loaded_engine(engine: RAGEngine) -> RAGEngine:
    engine.ingest_files(
        [("policy.txt", SAMPLE_TEXT.encode()), ("support.txt", OTHER_TEXT.encode())]
    )
    return engine
