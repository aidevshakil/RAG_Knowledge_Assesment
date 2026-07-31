from __future__ import annotations

import numpy as np
import pytest

from rag_assistant.config import EmbeddingConfig
from rag_assistant.core.exceptions import UnsupportedFileTypeError
from rag_assistant.embeddings.base import CachedEmbedder
from rag_assistant.embeddings.factory import build_embedder
from rag_assistant.ingestion.loaders import load_bytes, supported_extensions


def test_vectors_are_normalised_and_correctly_shaped():
    embedder = build_embedder(EmbeddingConfig(provider="hashing", dimensions=128), cache=False)
    vectors = embedder.embed_documents(["first document", "second document"])
    assert vectors.shape == (2, 128)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert embedder.dimension == 128


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = build_embedder(EmbeddingConfig(provider="hashing", dimensions=512), cache=False)
    query = embedder.embed_query("refund policy within thirty days")
    related = embedder.embed_query("the refund policy allows thirty days")
    unrelated = embedder.embed_query("shipping crates of industrial machinery")
    assert float(query @ related) > float(query @ unrelated)


def test_dimension_choice_changes_vector_width():
    for dim in (64, 256, 1024):
        embedder = build_embedder(EmbeddingConfig(provider="hashing", dimensions=dim), cache=False)
        assert embedder.embed_query("hello world").shape == (dim,)


def test_query_cache_avoids_recomputation():
    embedder = build_embedder(EmbeddingConfig(provider="hashing", dimensions=64))
    assert isinstance(embedder, CachedEmbedder)
    embedder.embed_query("same question")
    embedder.embed_query("same question")
    assert embedder.hits == 1
    assert embedder.hit_rate == 0.5


def test_empty_input_returns_empty_matrix():
    embedder = build_embedder(EmbeddingConfig(provider="hashing", dimensions=32), cache=False)
    assert embedder.embed_documents([]).shape == (0, 32)


def test_text_and_markdown_are_supported():
    loaded = load_bytes(b"# Title\n\nSome   body   text.", "notes.md")
    assert "Title" in loaded.text
    assert "  " not in loaded.text


def test_csv_rows_become_self_describing_lines():
    loaded = load_bytes(b"name,role\nAda,engineer\nGrace,admiral\n", "team.csv")
    assert "name: Ada; role: engineer" in loaded.text
    assert loaded.metadata["rows"] == 2


def test_html_tags_are_stripped():
    loaded = load_bytes(
        b"<html><body><script>x=1</script><p>Hello world</p></body></html>", "p.html"
    )
    assert "Hello world" in loaded.text
    assert "script" not in loaded.text.lower()


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileTypeError):
        load_bytes(b"binary", "archive.zip")


def test_common_formats_are_registered():
    assert {"pdf", "txt", "docx", "md", "csv"} <= set(supported_extensions())
