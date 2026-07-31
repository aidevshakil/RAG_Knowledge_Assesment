"""Knowledge base insights: the stored query/response history."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.ui.components import empty_state, fmt_ms, fmt_time, score_color
from rag_assistant.pipeline.engine import RAGEngine


def render(engine: RAGEngine) -> None:
    repository = engine.repository

    search, limit, actions = st.columns([4, 1, 1])
    term = search.text_input("Search questions and answers", placeholder="e.g. refund policy")
    count = limit.number_input("Show", min_value=10, max_value=500, value=50, step=10)
    with actions:
        st.write("")
        clear = st.button("Clear log", use_container_width=True)
    if clear:
        repository.clear_queries()
        st.rerun()

    rows = (
        repository.search_queries(term, limit=int(count))
        if term.strip()
        else repository.recent_queries(limit=int(count))
    )
    if not rows:
        empty_state(
            "No history yet" if not term.strip() else "No matches",
            "Questions and their answers are stored here for reference.",
            "🗂️",
        )
        return

    _top_questions(rows)
    st.divider()

    for row in rows:
        flag = " ⚠️" if row["hallucination_risk"] else ""
        with st.expander(f"**{row['question']}**{flag} · {fmt_time(row['created_at'])}"):
            st.markdown(row["answer"])
            st.caption(
                f"{fmt_ms(row['total_ms'])} · relevance {row['mean_score']:.2f} · "
                f"faithfulness {row['faithfulness']:.0%} · {row['top_k']} chunk(s) · "
                f"{row['llm_provider'] or 'n/a'}:{row['llm_model'] or 'n/a'}"
            )
            if row.get("feedback"):
                st.caption("User rated: " + ("👍" if row["feedback"] > 0 else "👎"))
            if row["hits"]:
                st.markdown("**Sources used**")
                for hit in row["hits"]:
                    st.markdown(
                        f"- {score_color(hit['score'])} `{hit['score']:.3f}` — "
                        f"{hit.get('citation', hit.get('document_name', '?'))}"
                    )
            st.caption(f"Configuration: `{row['config_fingerprint']}`")

    st.download_button(
        "⬇️ Export history (JSON)",
        data=json.dumps(rows, indent=2, default=str),
        file_name="rag_query_history.json",
        mime="application/json",
    )


def _top_questions(rows: list[dict]) -> None:
    """Repeated questions are the strongest signal of what to document better."""
    frame = pd.DataFrame([{"question": r["question"].strip().lower()} for r in rows])
    counts = frame["question"].value_counts()
    repeated = counts[counts > 1].head(5)
    if repeated.empty:
        return
    st.markdown("##### Most asked")
    for question, occurrences in repeated.items():
        st.markdown(f"- **{question}** — asked {occurrences}×")
