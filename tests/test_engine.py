"""End-to-end engine behaviour, especially the dynamic data-source guarantees."""

from __future__ import annotations

from tests.conftest import OTHER_TEXT, SAMPLE_TEXT


def test_ingest_indexes_chunks_and_documents(engine):
    result = engine.ingest_files([("policy.txt", SAMPLE_TEXT.encode())])
    assert result.ok
    assert result.chunks_added > 0
    assert len(engine.list_documents()) == 1
    assert engine.store.count() == result.chunks_added


def test_duplicate_content_is_skipped(engine):
    engine.ingest_files([("policy.txt", SAMPLE_TEXT.encode())])
    again = engine.ingest_files([("copy-of-policy.txt", SAMPLE_TEXT.encode())])
    assert again.skipped
    assert len(engine.list_documents()) == 1


def test_unsupported_file_is_reported_not_raised(engine):
    result = engine.ingest_files([("archive.zip", b"PK\x03\x04")])
    assert result.failed
    assert not result.documents


def test_delete_removes_document_and_its_vectors(loaded_engine):
    engine = loaded_engine
    documents = engine.list_documents()
    target = next(d for d in documents if d.name == "policy.txt")
    before = engine.store.count()

    assert engine.delete_document(target.id) is True

    assert engine.store.count() == before - target.chunk_count
    assert target.id not in {d.id for d in engine.list_documents()}
    assert target.id not in engine.store.document_ids()


def test_deleted_content_is_not_retrievable(loaded_engine):
    """The whole point of the delete path: stale content must vanish."""
    engine = loaded_engine
    question = "How long is the refund window?"
    assert any("refund" in hit.text.lower() for hit in engine.retrieve_only(question))

    target = next(d for d in engine.list_documents() if d.name == "policy.txt")
    engine.delete_document(target.id)

    hits = engine.retrieve_only(question)
    assert all(hit.chunk.document_name != "policy.txt" for hit in hits)


def test_ask_returns_grounded_answer_with_metrics(loaded_engine):
    answer = loaded_engine.ask("What is the refund policy?")
    assert answer.answer
    assert answer.hits
    assert answer.timings.total_ms > 0
    assert answer.timings.embed_ms > 0
    assert 0.0 <= answer.metrics.faithfulness <= 1.0
    assert answer.citations


def test_ask_on_empty_knowledge_base_abstains(engine):
    answer = engine.ask("anything at all?")
    assert not answer.hits
    assert answer.metrics.abstained
    assert not answer.metrics.hallucination_risk


def test_scoped_query_only_searches_selected_documents(loaded_engine):
    engine = loaded_engine
    support = next(d for d in engine.list_documents() if d.name == "support.txt")
    hits = engine.retrieve_only("refund policy", document_ids=[support.id])
    assert hits
    assert all(hit.chunk.document_id == support.id for hit in hits)


def test_queries_are_logged_for_the_dashboard(loaded_engine):
    loaded_engine.ask("What is the warranty period?")
    logged = loaded_engine.repository.recent_queries(limit=5)
    assert logged
    assert logged[0]["total_ms"] > 0
    summary = loaded_engine.repository.performance_summary()
    assert summary["queries"] >= 1


def test_clear_knowledge_base_empties_everything(loaded_engine):
    loaded_engine.clear_knowledge_base()
    assert loaded_engine.list_documents() == []
    assert loaded_engine.store.count() == 0
    assert loaded_engine.is_empty


def test_reconcile_removes_orphaned_vectors(loaded_engine):
    engine = loaded_engine
    target = next(d for d in engine.list_documents() if d.name == "support.txt")
    engine.repository.delete_document(target.id)
    assert target.id in engine.store.document_ids()

    report = engine.reconcile()

    assert report["orphan_vector_documents"] == 1
    assert target.id not in engine.store.document_ids()


def test_stats_reflect_the_current_state(loaded_engine):
    stats = loaded_engine.stats()
    assert stats["documents"] == 2
    assert stats["vectors"] == loaded_engine.store.count()
    assert stats["dimension"] == loaded_engine.embedder.dimension


def test_ingest_text_and_url_name_handling(engine):
    result = engine.ingest_text(OTHER_TEXT, name="support-note")
    assert result.ok
    assert engine.list_documents()[0].name == "support-note"
