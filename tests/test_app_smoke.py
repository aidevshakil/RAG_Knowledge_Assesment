"""Streamlit smoke test.

Runs the real `app/main.py` through Streamlit's AppTest harness so a broken
widget call or a bad import fails CI instead of the user's browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "main.py"


@pytest.fixture()
def app(tmp_path, monkeypatch) -> AppTest:
    """A fresh app instance pointed at an isolated, fully offline data dir."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("VECTOR_STORE", "numpy")
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import streamlit as st

    st.cache_resource.clear()
    return AppTest.from_file(str(APP), default_timeout=120)


def test_app_renders_without_exceptions(app: AppTest):
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert any("RAG Knowledge Assistant" in title.value for title in app.title)
    assert len(app.tabs) >= 5


def test_empty_state_is_shown_before_any_ingestion(app: AppTest):
    app.run()
    assert not app.exception
    rendered = " ".join(block.value for block in app.markdown)
    assert "No documents yet" in rendered or "Nothing indexed yet" in rendered


def test_chat_flow_answers_after_ingestion(app: AppTest, tmp_path):
    """Ingest through the engine, then ask through the UI's chat input."""
    app.run()
    assert not app.exception

    from app.bootstrap import get_engine

    settings = app.session_state["settings"]
    engine = get_engine(settings.fingerprint, settings)
    engine.ingest_text(
        "Customers may request a full refund within 30 days of purchase.", "policy.txt"
    )

    app.chat_input[0].set_value("What is the refund window?").run()
    assert not app.exception, [str(e) for e in app.exception]

    messages = app.session_state["messages"]
    assert len(messages) == 2
    assert messages[1]["role"] == "assistant"
    assert messages[1]["answer"] is not None
    assert messages[1]["answer"].hits


def test_sidebar_and_metrics_render_with_data(app: AppTest):
    app.run()
    from app.bootstrap import get_engine

    settings = app.session_state["settings"]
    engine = get_engine(settings.fingerprint, settings)
    engine.ingest_text("Standard shipping takes three to seven business days.", "shipping.txt")
    engine.ask("How long does shipping take?")

    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    labels = {metric.label for metric in app.metric}
    assert {"Documents", "Chunks indexed"} <= labels
