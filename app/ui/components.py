"""Small shared widgets and formatters."""

from __future__ import annotations

import re
import time
from typing import Any

import streamlit as st

from rag_assistant.core.types import RAGAnswer, SearchHit

_GOOD, _OK = 0.45, 0.25
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def score_color(score: float) -> str:
    return "🟢" if score >= _GOOD else ("🟡" if score >= _OK else "🔴")


def fmt_ms(value: float | None) -> str:
    if not value:
        return "0 ms"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


def fmt_bytes(value: int | float) -> str:
    size = float(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def fmt_time(timestamp: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp)) if timestamp else "—"


def fmt_pct(value: float | None) -> str:
    return f"{(value or 0) * 100:.0f}%"


def metric_row(items: list[tuple[str, Any, str | None]]) -> None:
    """Render a row of `st.metric` cards from (label, value, help) tuples."""
    for column, (label, value, helptext) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value, help=helptext)


def render_sources(hits: list[SearchHit], *, expanded: bool = False, key: str = "") -> None:
    """The retrieved-context inspector: what the answer was actually built from."""
    if not hits:
        return
    with st.expander(f"📚 Retrieved context · {len(hits)} chunk(s)", expanded=expanded):
        for hit in hits:
            st.markdown(
                f"**[{hit.rank + 1}] {hit.chunk.citation}** &nbsp; "
                f"{score_color(hit.score)} similarity `{hit.score:.3f}`",
                unsafe_allow_html=True,
            )
            st.caption(hit.text[:1200] + ("…" if len(hit.text) > 1200 else ""))
            st.divider()


def render_answer_footer(answer: RAGAnswer) -> None:
    """Per-answer latency and quality strip."""
    metrics = answer.metrics
    timings = answer.timings
    metric_row(
        [
            (
                "Latency",
                fmt_ms(timings.total_ms),
                f"embed {fmt_ms(timings.embed_ms)} · retrieve {fmt_ms(timings.retrieve_ms)} · "
                f"generate {fmt_ms(timings.generate_ms)} · eval {fmt_ms(timings.evaluate_ms)}",
            ),
            (
                "Top-k relevance",
                f"{metrics.mean_score:.2f}",
                "Mean cosine similarity of the retrieved chunks",
            ),
            (
                "Precision@k",
                fmt_pct(metrics.precision_at_k),
                "Share of retrieved chunks above the relevance threshold",
            ),
            (
                "Faithfulness",
                fmt_pct(metrics.faithfulness),
                "How much of the answer is grounded in the retrieved context",
            ),
        ]
    )
    if metrics.hallucination_risk:
        st.warning(
            "⚠️ Low grounding — parts of this answer may not be supported by your documents. "
            "Check the retrieved context below.",
            icon="⚠️",
        )
    if metrics.judge_verdict:
        st.caption(f"LLM judge: {metrics.judge_verdict}")


def empty_state(title: str, body: str, icon: str = "📄") -> None:
    body = _BOLD.sub(r"<strong>\1</strong>", body)
    st.markdown(
        f"<div style='text-align:center;padding:3rem 1rem;opacity:.75'>"
        f"<div style='font-size:2.5rem'>{icon}</div>"
        f"<h4 style='margin:.5rem 0'>{title}</h4>"
        f"<p style='margin:0'>{body}</p></div>",
        unsafe_allow_html=True,
    )
