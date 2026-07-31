"""Knowledge Base tab — the Data Source Manager.

Add files, folders, URLs or pasted text at runtime; delete any document and its
vectors leave the index in the same operation. The table always reflects exactly
what retrieval can see.
"""

from __future__ import annotations

import streamlit as st

from app.ui.components import empty_state, fmt_bytes, fmt_time, metric_row
from rag_assistant.core.types import IngestionResult
from rag_assistant.ingestion.loaders import supported_extensions
from rag_assistant.pipeline.engine import RAGEngine


def render(engine: RAGEngine) -> None:
    _summary(engine)
    st.divider()
    _add_sources(engine)
    st.divider()
    _document_table(engine)


def _summary(engine: RAGEngine) -> None:
    stats = engine.stats()
    metric_row(
        [
            ("Documents", int(stats.get("documents", 0)), None),
            ("Chunks indexed", int(stats.get("vectors", 0)), "Vectors currently searchable"),
            ("Text volume", f"{int(stats.get('characters', 0)):,} chars", None),
            ("Vector width", f"{stats['dimension']}d", stats["embedding"]),
        ]
    )


def _add_sources(engine: RAGEngine) -> None:
    st.markdown("#### ➕ Add sources")
    files_tab, url_tab, text_tab = st.tabs(["📄 Upload files", "🌐 From URL", "✍️ Paste text"])

    with files_tab:
        uploads = st.file_uploader(
            f"Supported: {', '.join(supported_extensions())}",
            type=supported_extensions(),
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.get('uploader_epoch', 0)}",
        )
        if uploads and st.button("Index files", type="primary", key="btn_index_files"):
            _run_ingestion(
                engine,
                lambda progress: engine.ingest_files(
                    [(f.name, f.getvalue()) for f in uploads], progress=progress
                ),
            )

    with url_tab:
        url = st.text_input("Page or PDF URL", placeholder="https://example.com/handbook.pdf")
        if url and st.button("Fetch & index", type="primary", key="btn_index_url"):
            _run_ingestion(engine, lambda progress: engine.ingest_url(url, progress=progress))

    with text_tab:
        name = st.text_input("Name", value="pasted-note", key="paste_name")
        body = st.text_area("Content", height=180, key="paste_body")
        if body.strip() and st.button("Index text", type="primary", key="btn_index_text"):
            _run_ingestion(engine, lambda _: engine.ingest_text(body, name or "pasted-note"))


def _run_ingestion(engine: RAGEngine, action) -> None:
    bar = st.progress(0.0, text="Starting…")

    def progress(fraction: float, message: str) -> None:
        bar.progress(min(1.0, max(0.0, fraction)), text=message)

    try:
        result: IngestionResult = action(progress)
    except Exception as exc:
        bar.empty()
        st.error(f"Ingestion failed: {exc}")
        return
    bar.empty()

    if result.documents:
        st.success(result.summary(), icon="✅")
    for name, reason in result.skipped:
        st.info(f"**{name}** skipped — {reason}", icon="↩️")
    for name, error in result.failed:
        st.error(f"**{name}** failed — {error}", icon="⚠️")

    if result.documents:
        st.session_state["uploader_epoch"] = st.session_state.get("uploader_epoch", 0) + 1
        st.rerun()


def _document_table(engine: RAGEngine) -> None:
    st.markdown("#### 📚 Active documents")
    documents = engine.list_documents()
    if not documents:
        empty_state("Nothing indexed yet", "Add a file above to build your knowledge base.", "📭")
        return

    header = st.columns([4, 1, 1, 2, 1])
    for column, title in zip(header, ("Document", "Type", "Chunks", "Added", ""), strict=True):
        column.markdown(f"**{title}**")

    for document in documents:
        cols = st.columns([4, 1, 1, 2, 1])
        cols[0].markdown(f"**{document.name}**")
        cols[0].caption(
            f"{fmt_bytes(document.size_bytes)} · {document.char_count:,} chars"
            + (f" · [source]({document.uri})" if document.uri else "")
        )
        cols[1].write(document.source_type)
        cols[2].write(document.chunk_count)
        cols[3].write(fmt_time(document.created_at))
        if cols[4].button("🗑️", key=f"del_{document.id}", help="Delete document and its vectors"):
            st.session_state["pending_delete"] = document.id
            st.rerun()

    _confirm_delete(engine, documents)
    st.divider()
    _danger_zone(engine)


def _confirm_delete(engine: RAGEngine, documents) -> None:
    pending = st.session_state.get("pending_delete")
    if not pending:
        return
    target = next((d for d in documents if d.id == pending), None)
    if target is None:
        st.session_state.pop("pending_delete", None)
        return

    st.warning(
        f"Delete **{target.name}** and remove its {target.chunk_count} chunk(s) from the "
        "vector store? Retrieval will stop seeing this content immediately.",
        icon="🗑️",
    )
    confirm, cancel, _ = st.columns([1, 1, 6])
    if confirm.button("Delete", type="primary", key="confirm_delete"):
        with st.spinner("Removing vectors…"):
            engine.delete_document(target.id)
        st.session_state.pop("pending_delete", None)
        st.toast(f"Deleted {target.name}", icon="🗑️")
        st.rerun()
    if cancel.button("Cancel", key="cancel_delete"):
        st.session_state.pop("pending_delete", None)
        st.rerun()


def _danger_zone(engine: RAGEngine) -> None:
    with st.expander("⚠️ Maintenance"):
        left, right = st.columns(2)
        if left.button("Run consistency check", use_container_width=True):
            report = engine.reconcile()
            orphans = report["orphan_vector_documents"]
            missing = report["documents_missing_vectors"]
            if not orphans and not missing:
                st.success("Metadata and vector index agree.", icon="✅")
            else:
                st.warning(
                    f"Removed {orphans} orphaned vector group(s). "
                    f"Documents without vectors: {', '.join(missing) or 'none'}",
                    icon="🔧",
                )
        if right.button("Clear knowledge base", type="secondary", use_container_width=True):
            st.session_state["confirm_clear"] = True
        if st.session_state.get("confirm_clear"):
            st.error("This deletes every document, vector and uploaded copy.", icon="🚨")
            yes, no, _ = st.columns([1, 1, 6])
            if yes.button("Yes, delete everything", type="primary", key="confirm_clear_yes"):
                engine.clear_knowledge_base()
                st.session_state.pop("confirm_clear", None)
                st.session_state["messages"] = []
                st.rerun()
            if no.button("Cancel", key="confirm_clear_no"):
                st.session_state.pop("confirm_clear", None)
                st.rerun()
