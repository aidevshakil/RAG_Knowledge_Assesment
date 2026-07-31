"""Benchmark tab.

Two workflows:

* **Snapshots** — run a question suite against the *current* knowledge base and
  save the result. Add or delete files, run it again, and diff the two rows to
  see exactly how the data-source change affected quality and latency.
* **Config sweep** — index a copy of your documents under several embedding
  models / dimensions / vector stores in a throwaway collection and compare
  retrieval quality side by side.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.bootstrap import Settings
from app.ui.components import empty_state, fmt_ms, fmt_time
from rag_assistant.evaluation.benchmark import (
    BenchmarkRunner,
    ConfigVariant,
    compare_configs,
)
from rag_assistant.pipeline.engine import RAGEngine

_DEFAULT_SUITE = [
    {"question": "What is this document about?"},
    {"question": "Summarise the key points."},
]

_SWEEP_PRESETS: dict[str, ConfigVariant] = {
    "MiniLM 384d (HF)": ConfigVariant(
        label="MiniLM-384d",
        embedding_provider="huggingface",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ),
    "MPNet 768d (HF)": ConfigVariant(
        label="MPNet-768d",
        embedding_provider="huggingface",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
    ),
    "OpenAI 3-small 1536d": ConfigVariant(
        label="OpenAI-1536d",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    ),
    "OpenAI 3-small → 512d": ConfigVariant(
        label="OpenAI-512d",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=512,
    ),
    "Hashing 256d (offline)": ConfigVariant(
        label="Hashing-256d",
        embedding_provider="hashing",
        dimensions=256,
    ),
    "Hashing 1024d (offline)": ConfigVariant(
        label="Hashing-1024d",
        embedding_provider="hashing",
        dimensions=1024,
    ),
}


def render(engine: RAGEngine, settings: Settings) -> None:
    snapshots, sweep, history = st.tabs(
        ["📸 Snapshots", "🔬 Compare configurations", "🗄️ Saved runs"]
    )
    with snapshots:
        _snapshot_panel(engine)
    with sweep:
        _sweep_panel(engine, settings)
    with history:
        _history_panel(engine)


def _snapshot_panel(engine: RAGEngine) -> None:
    st.markdown(
        "Run a fixed question suite against the current knowledge base. Take one "
        "snapshot **before** changing your documents and one **after** — the "
        "comparison below shows what the change actually did."
    )
    if engine.is_empty:
        empty_state("Nothing to benchmark", "Index some documents first.", "🧪")
        return

    suite_text = st.text_area(
        "Question suite (JSON)",
        value=json.dumps(_suite_default(engine), indent=2),
        height=220,
        help='Each item: {"question": "...", "expected_sources": ["file.pdf"], '
        '"expected_answer_terms": ["30 days"]}. The optional fields unlock '
        "recall@k, MRR, nDCG and answer-correctness scoring.",
    )
    left, middle, right = st.columns([2, 1, 1])
    label = left.text_input("Snapshot label", value="before-change")
    retrieval_only = middle.toggle(
        "Retrieval only",
        value=True,
        help="Skip generation: faster, free, isolates retrieval quality.",
    )
    run = right.button("Run benchmark", type="primary", use_container_width=True)

    if run:
        try:
            cases = json.loads(suite_text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return
        bar = st.progress(0.0, text="Starting…")
        try:
            result = BenchmarkRunner(engine).run(
                cases,
                label=label or "run",
                generate=not retrieval_only,
                progress=lambda f, m: bar.progress(min(1.0, f), text=m),
            )
        except Exception as exc:
            bar.empty()
            st.error(f"Benchmark failed: {exc}")
            return
        bar.empty()
        st.success(f"Saved snapshot **{result.label}**", icon="✅")
        st.json(result.summary, expanded=True)
        st.dataframe(pd.DataFrame(result.details), use_container_width=True, hide_index=True)

    _compare_snapshots(engine)


def _suite_default(engine: RAGEngine) -> list[dict]:
    documents = engine.list_documents()
    if not documents:
        return _DEFAULT_SUITE
    return [
        {
            "question": f"What does {documents[0].name} say about its main topic?",
            "expected_sources": [documents[0].name],
        },
        *_DEFAULT_SUITE,
    ]


def _compare_snapshots(engine: RAGEngine) -> None:
    runs = engine.repository.list_benchmarks(limit=50)
    if len(runs) < 2:
        return
    st.divider()
    st.markdown("#### Compare two snapshots")
    options = {
        f"{r['label']} · {fmt_time(r['created_at'])} · {r['doc_count']} docs": r for r in runs
    }
    keys = list(options)
    left, right = st.columns(2)
    before = options[left.selectbox("Baseline", keys, index=min(1, len(keys) - 1))]
    after = options[right.selectbox("Comparison", keys, index=0)]

    rows = []
    metric_keys = sorted(set(before["summary"]) | set(after["summary"]) - {"cases", "errors"})
    for key in metric_keys:
        old, new = before["summary"].get(key), after["summary"].get(key)
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        delta = new - old
        rows.append(
            {
                "metric": key,
                before["label"]: round(old, 4),
                after["label"]: round(new, 4),
                "change": round(delta, 4),
                "direction": _direction(key, delta),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Knowledge base: {before['doc_count']} → {after['doc_count']} documents, "
        f"{before['chunk_count']} → {after['chunk_count']} chunks."
    )


_LOWER_IS_BETTER = ("ms", "hallucination", "errors")


def _direction(metric: str, delta: float) -> str:
    if abs(delta) < 1e-6:
        return "—"
    lower_better = any(marker in metric for marker in _LOWER_IS_BETTER)
    improved = delta < 0 if lower_better else delta > 0
    return "🟢 better" if improved else "🔴 worse"


def _sweep_panel(engine: RAGEngine, settings: Settings) -> None:
    st.markdown(
        "Re-index a copy of your documents under each configuration and score the "
        "same questions against all of them. Runs in a **temporary collection** — "
        "your live knowledge base is untouched."
    )
    documents = engine.list_documents()
    if not documents:
        empty_state("Nothing to sweep", "Index some documents first.", "🔬")
        return

    picked_docs = st.multiselect(
        "Documents to index for the sweep",
        options=[d.id for d in documents],
        default=[d.id for d in documents[:3]],
        format_func=lambda i: next(d.name for d in documents if d.id == i),
        help="Fewer documents means a much faster sweep.",
    )
    picked_variants = st.multiselect(
        "Configurations",
        list(_SWEEP_PRESETS),
        default=list(_SWEEP_PRESETS)[:2],
        help="OpenAI presets need OPENAI_API_KEY; HuggingFace presets download a model once.",
    )
    suite_text = st.text_area(
        "Questions (JSON)",
        value=json.dumps(_suite_default(engine), indent=2),
        height=160,
        key="sweep_suite",
    )
    generate = st.toggle(
        "Also generate answers",
        value=False,
        help="Off by default: retrieval-only isolates the embedding/store effect "
        "and costs nothing.",
    )

    if not st.button("Run sweep", type="primary", disabled=not (picked_docs and picked_variants)):
        return
    try:
        cases = json.loads(suite_text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return

    payload = _read_documents(engine, picked_docs)
    if not payload:
        st.error(
            "Could not read the original files for the selected documents. "
            "Re-upload them (uploads are cached under data/uploads) and try again."
        )
        return

    bar = st.progress(0.0, text="Starting…")
    try:
        rows = compare_configs(
            settings,
            [_SWEEP_PRESETS[name] for name in picked_variants],
            cases,
            payload,
            generate=generate,
            progress=lambda f, m: bar.progress(min(1.0, f), text=m),
        )
    except Exception as exc:
        bar.empty()
        st.error(f"Sweep failed: {exc}")
        return
    bar.empty()

    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    _sweep_takeaway(frame)


def _read_documents(engine: RAGEngine, document_ids: list[str]) -> list[tuple[str, bytes]]:
    from pathlib import Path

    payload: list[tuple[str, bytes]] = []
    for document in engine.list_documents():
        if document.id not in document_ids or not document.path:
            continue
        path = Path(document.path)
        if path.is_file():
            payload.append((document.name, path.read_bytes()))
    return payload


def _sweep_takeaway(frame: pd.DataFrame) -> None:
    if "mean_score" not in frame or frame["mean_score"].isna().all():
        return
    best = frame.loc[frame["mean_score"].idxmax()]
    fastest = frame.loc[frame["index_ms"].idxmin()] if "index_ms" in frame else None
    st.success(
        f"**Best retrieval quality:** {best['variant']} "
        f"({best['dimension']}d, mean score {best['mean_score']:.3f})",
        icon="🏆",
    )
    if fastest is not None and fastest["variant"] != best["variant"]:
        st.info(
            f"**Cheapest to index:** {fastest['variant']} — {fmt_ms(fastest['index_ms'])} for "
            f"{int(fastest['chunks'])} chunks at {fastest['dimension']}d. "
            "Higher dimensions usually buy accuracy at the cost of index time, "
            "memory and query latency.",
            icon="⚖️",
        )


def _history_panel(engine: RAGEngine) -> None:
    runs = engine.repository.list_benchmarks(limit=50)
    if not runs:
        empty_state("No saved runs", "Run a snapshot to store one.", "🗄️")
        return
    table = pd.DataFrame(
        [
            {
                "label": run["label"],
                "when": fmt_time(run["created_at"]),
                "docs": run["doc_count"],
                "chunks": run["chunk_count"],
                "cases": run["query_count"],
                **{k: v for k, v in run["summary"].items() if isinstance(v, (int, float))},
            }
            for run in runs
        ]
    )
    st.dataframe(table.round(3), use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Inspect run", runs, format_func=lambda r: f"{r['label']} · {fmt_time(r['created_at'])}"
    )
    if selected:
        st.dataframe(pd.DataFrame(selected["details"]), use_container_width=True, hide_index=True)
        left, right = st.columns([1, 5])
        if left.button("Delete run"):
            engine.repository.delete_benchmark(selected["id"])
            st.rerun()
        right.download_button(
            "⬇️ Export run (JSON)",
            data=json.dumps(selected, indent=2, default=str),
            file_name=f"benchmark_{selected['label']}.json",
            mime="application/json",
        )
