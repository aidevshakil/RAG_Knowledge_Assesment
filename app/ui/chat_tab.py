"""Chat tab: ask questions, stream the answer, inspect the evidence."""

from __future__ import annotations

import streamlit as st

from app.ui.components import empty_state, render_answer_footer, render_sources
from rag_assistant.core.types import RAGAnswer
from rag_assistant.pipeline.engine import RAGEngine

_HISTORY_TURNS = 3


def render(engine: RAGEngine) -> None:
    st.session_state.setdefault("messages", [])
    if engine.llm.provider == "extractive":
        st.info(
            "**Extractive mode** — answers are sentences quoted straight from your "
            "documents. Add a `GROQ_API_KEY` (or OpenAI/Anthropic) and pick the provider "
            "in the sidebar for synthesised answers.",
            icon="ℹ️",
        )
    _controls(engine)

    if not st.session_state["messages"]:
        if engine.is_empty:
            empty_state(
                "No documents yet",
                "Open the <strong>📚 Knowledge Base</strong> tab above → "
                "<strong>📄 Upload files</strong>, add a PDF / DOCX / TXT / CSV "
                "(or a URL), then come back and ask anything.",
                "📥",
            )
        else:
            empty_state(
                "Ask your documents anything",
                "Answers are grounded in your sources and cited.",
                "💬",
            )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            answer: RAGAnswer | None = message.get("answer")
            if answer is not None:
                render_sources(answer.hits, key=answer.id)
                render_answer_footer(answer)
                _feedback(engine, answer)

    question = st.chat_input("Ask a question about your documents…")
    if question:
        _handle(engine, question)


def _controls(engine: RAGEngine) -> None:
    left, right = st.columns([4, 1])
    documents = engine.list_documents()
    with left:
        selected = st.multiselect(
            "Restrict to specific documents (optional)",
            options=[doc.id for doc in documents],
            format_func=lambda doc_id: next((d.name for d in documents if d.id == doc_id), doc_id),
            key="scope_documents",
            placeholder="All documents",
        )
        st.session_state["scope"] = selected or None
    with right:
        st.write("")
        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


def _handle(engine: RAGEngine, question: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        (m["content"], n["content"])
        for m, n in zip(
            st.session_state["messages"][:-1:2],
            st.session_state["messages"][1::2],
            strict=False,
        )
    ][-_HISTORY_TURNS:]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer: RAGAnswer | None = None
        try:
            stream = engine.ask_stream(
                question, document_ids=st.session_state.get("scope"), history=history
            )
            buffer: list[str] = []
            for item in stream:
                if isinstance(item, RAGAnswer):
                    answer = item
                    break
                buffer.append(item)
                placeholder.markdown("".join(buffer) + " ▌")
            text = answer.answer if answer else "".join(buffer)
            placeholder.markdown(text)
        except Exception as exc:
            text = f"Something went wrong while answering: `{exc}`"
            placeholder.error(text)

        if answer is not None:
            render_sources(answer.hits, key=answer.id)
            render_answer_footer(answer)

    st.session_state["messages"].append({"role": "assistant", "content": text, "answer": answer})
    st.rerun()


def _feedback(engine: RAGEngine, answer: RAGAnswer) -> None:
    """Thumbs up/down — stored alongside the automatic metrics."""
    key = f"fb_{answer.id}"
    if st.session_state.get(key):
        st.caption("Thanks for the feedback.")
        return
    up, down, _ = st.columns([1, 1, 8])
    if up.button("👍", key=f"{key}_up", help="Helpful"):
        engine.repository.set_feedback(answer.id, 1)
        st.session_state[key] = True
        st.rerun()
    if down.button("👎", key=f"{key}_down", help="Not helpful"):
        engine.repository.set_feedback(answer.id, -1)
        st.session_state[key] = True
        st.rerun()
