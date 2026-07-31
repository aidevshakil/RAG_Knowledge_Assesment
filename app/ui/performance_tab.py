"""Performance Monitor — latency, retrieval quality and hallucination rate."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.ui.components import empty_state, fmt_ms, fmt_pct, metric_row
from rag_assistant.pipeline.engine import RAGEngine

_WINDOWS = {"Last hour": 3600, "Last 24 hours": 86_400, "Last 7 days": 604_800, "All time": None}


def render(engine: RAGEngine) -> None:
    repository = engine.repository

    left, right = st.columns([2, 1])
    window = left.selectbox("Time window", list(_WINDOWS), index=3)
    since = time.time() - _WINDOWS[window] if _WINDOWS[window] else None
    if right.button("Refresh", use_container_width=True):
        st.rerun()

    summary = repository.performance_summary(since=since)
    if not summary.get("queries"):
        empty_state(
            "No queries yet", "Ask something in the Chat tab and metrics appear here.", "📊"
        )
        return

    _headline(summary)
    st.divider()
    _latency_breakdown(summary)
    st.divider()
    _trends(repository)
    st.divider()
    _by_configuration(repository)


def _headline(summary: dict) -> None:
    st.markdown("#### Overview")
    metric_row(
        [
            ("Queries", int(summary["queries"]), None),
            (
                "Avg latency",
                fmt_ms(summary["avg_total_ms"]),
                f"p95 {fmt_ms(summary['p95_total_ms'])}",
            ),
            (
                "Avg top-k relevance",
                f"{summary['avg_mean_score']:.2f}",
                "Mean cosine similarity of retrieved chunks",
            ),
            (
                "Faithfulness",
                fmt_pct(summary["avg_faithfulness"]),
                "Share of each answer grounded in retrieved context",
            ),
        ]
    )
    metric_row(
        [
            (
                "Precision@k",
                fmt_pct(summary["avg_precision"]),
                "Retrieved chunks above the relevance threshold",
            ),
            (
                "Answer relevance",
                fmt_pct(summary["avg_answer_relevance"]),
                "Semantic similarity between question and answer",
            ),
            (
                "Hallucination rate",
                fmt_pct(summary["hallucination_rate"]),
                "Answers whose grounding fell below the threshold — lower is better",
            ),
            (
                "Abstention rate",
                fmt_pct(summary["abstention_rate"]),
                "Answers that correctly declined for lack of context",
            ),
        ]
    )


def _latency_breakdown(summary: dict) -> None:
    st.markdown("#### Where the time goes")
    stages = pd.DataFrame(
        {
            "stage": ["Embedding", "Retrieval", "Generation"],
            "milliseconds": [
                summary["avg_embed_ms"],
                summary["avg_retrieve_ms"],
                summary["avg_generate_ms"],
            ],
        }
    ).set_index("stage")
    st.bar_chart(stages, horizontal=True, height=200)
    generation_share = summary["avg_generate_ms"] / max(1e-6, summary["avg_total_ms"])
    if generation_share > 0.75:
        st.caption(
            "🕒 Generation dominates — a smaller/faster model or a lower top-k will "
            "cut latency more than any retrieval tuning."
        )
    elif summary["avg_embed_ms"] > summary["avg_retrieve_ms"] * 3:
        st.caption(
            "🕒 Query embedding dominates — a smaller embedding model (or a hosted one) "
            "would help most."
        )


def _trends(repository) -> None:
    st.markdown("#### Trends")
    rows = repository.latency_series(limit=200)
    if len(rows) < 2:
        st.caption("At least two queries are needed to plot a trend.")
        return
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["created_at"], unit="s")
    frame = frame.set_index("time")

    latency, quality = st.tabs(["Latency", "Quality"])
    with latency:
        st.line_chart(frame[["total_ms", "generate_ms", "retrieve_ms", "embed_ms"]], height=280)
    with quality:
        st.line_chart(frame[["mean_score", "faithfulness", "answer_relevance"]], height=280)
        st.caption("Knowledge base size over the same period")
        st.area_chart(frame[["doc_count", "chunk_count"]], height=160)


def _by_configuration(repository) -> None:
    st.markdown("#### By configuration")
    st.caption(
        "Every query is tagged with the embedding model, store, top-k and chunking "
        "in force at the time — switch a backend in the sidebar and compare rows."
    )
    rows = repository.summary_by_config()
    if not rows:
        return
    frame = pd.DataFrame(rows).rename(
        columns={
            "config_fingerprint": "configuration",
            "avg_total_ms": "avg latency (ms)",
            "avg_mean_score": "avg relevance",
            "avg_precision": "precision@k",
            "avg_faithfulness": "faithfulness",
            "hallucination_rate": "hallucination rate",
        }
    )
    st.dataframe(
        frame.round(3),
        use_container_width=True,
        hide_index=True,
        column_config={
            "avg relevance": st.column_config.ProgressColumn(
                "avg relevance", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "faithfulness": st.column_config.ProgressColumn(
                "faithfulness", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
