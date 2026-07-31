from __future__ import annotations

import pytest

from rag_assistant.config import ChunkingConfig
from rag_assistant.ingestion.chunking import RecursiveChunker
from rag_assistant.ingestion.loaders import LoadedDocument


def test_short_text_yields_one_chunk():
    chunker = RecursiveChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    chunks = chunker.split_text("A short note about refunds and shipping.")
    assert len(chunks) == 1
    assert chunks[0].index == 0


def test_chunks_respect_size_limit():
    text = " ".join(f"sentence number {i} with some filler words." for i in range(400))
    chunker = RecursiveChunker(ChunkingConfig(chunk_size=300, chunk_overlap=50))
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 + 60 for chunk in chunks)


def test_overlap_preserves_context_across_boundaries():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(12))
    with_overlap = RecursiveChunker(ChunkingConfig(chunk_size=400, chunk_overlap=150)).split_text(
        text
    )
    without = RecursiveChunker(ChunkingConfig(chunk_size=400, chunk_overlap=0)).split_text(text)
    assert len(with_overlap) >= len(without)


def test_page_numbers_are_attached():
    document = LoadedDocument(
        text="first page content\n\nsecond page content",
        source_type="pdf",
        page_offsets=[(0, 1), (20, 2)],
    )
    chunks = RecursiveChunker(
        ChunkingConfig(chunk_size=20, chunk_overlap=0, min_chunk_chars=1)
    ).split(document, document_id="doc", document_name="file.pdf")
    assert any("page" in chunk.metadata for chunk in chunks)
    assert chunks[0].citation.startswith("file.pdf (p.")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_no_content_is_lost():
    text = " ".join(f"token{i}" for i in range(500))
    chunks = RecursiveChunker(ChunkingConfig(chunk_size=200, chunk_overlap=20)).split_text(text)
    combined = " ".join(chunk.text for chunk in chunks)
    assert "token0" in combined
    assert "token499" in combined
