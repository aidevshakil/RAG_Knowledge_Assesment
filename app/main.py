"""Streamlit entry point.

    streamlit run app/main.py

Thin by design: it wires the cached engine to the tab modules and owns nothing
else. All behaviour lives in `rag_assistant`, so the CLI, tests and this UI
exercise exactly the same code paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.bootstrap import base_settings, get_engine  # noqa: E402
from app.ui import (  # noqa: E402
    benchmark_tab,
    chat_tab,
    documents_tab,
    insights_tab,
    performance_tab,
    sidebar,
)

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    settings = st.session_state.setdefault("settings", base_settings())

    try:
        engine = get_engine(settings.fingerprint, settings)
    except Exception as exc:
        st.error(f"Could not start the engine: {exc}")
        st.info(
            "Fastest way to a working setup: set `EMBEDDING_PROVIDER=hashing`, "
            "`VECTOR_STORE=numpy` and `LLM_PROVIDER=extractive` in `.env` — that path "
            "needs no API keys or model downloads."
        )
        st.stop()

    engine.settings = settings
    engine.retriever.config = settings.retrieval
    engine.evaluator.config = settings.evaluation

    settings = sidebar.render(engine, settings)

    st.title("🧠 RAG Knowledge Assistant")
    st.caption(
        "Grounded answers over your own documents — with live source management "
        "and end-to-end performance measurement."
    )

    chat, documents, performance, insights, benchmark = st.tabs(
        ["💬 Chat", "📚 Knowledge Base", "📊 Performance", "🗂️ Insights", "🧪 Benchmark"]
    )
    with chat:
        chat_tab.render(engine)
    with documents:
        documents_tab.render(engine)
    with performance:
        performance_tab.render(engine)
    with insights:
        insights_tab.render(engine)
    with benchmark:
        benchmark_tab.render(engine, settings)


if __name__ == "__main__":
    main()
