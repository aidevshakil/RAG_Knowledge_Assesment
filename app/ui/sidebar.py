"""Sidebar: live configuration + knowledge-base status.

Changing a backend here rewrites the `Settings` object and the engine is rebuilt
on the next rerun (the fingerprint is the cache key). Nothing is mutated in
place, so a bad choice never corrupts the running engine.
"""

from __future__ import annotations

import streamlit as st

from app.bootstrap import (
    EmbeddingConfig,
    EvaluationConfig,
    LLMConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
)
from app.ui.components import fmt_bytes
from rag_assistant.embeddings.factory import available_embedders
from rag_assistant.llm.factory import available_llms
from rag_assistant.pipeline.engine import RAGEngine
from rag_assistant.vectorstores.factory import available_vector_stores

EMBEDDING_PRESETS = {
    "huggingface": [
        ("sentence-transformers/all-MiniLM-L6-v2", 384, "fast, small, good default"),
        ("sentence-transformers/all-mpnet-base-v2", 768, "better recall, ~3x slower"),
        ("BAAI/bge-small-en-v1.5", 384, "strong for its size"),
        ("BAAI/bge-large-en-v1.5", 1024, "best quality, heaviest"),
    ],
    "openai": [
        ("text-embedding-3-small", 1536, "cheap hosted baseline"),
        ("text-embedding-3-large", 3072, "highest quality, shortenable"),
    ],
    "hashing": [("hashing-bow", 512, "offline, keyword-only — no downloads")],
}

LLM_PRESETS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
    "extractive": ["extractive-summarizer"],
}


def render(engine: RAGEngine, settings: Settings) -> Settings:
    """Draw the sidebar; return the (possibly updated) settings."""
    with st.sidebar:
        st.markdown("### 🧠 RAG Assistant")
        _status(engine)
        st.divider()
        settings = _configuration(settings)
        st.divider()
        _footer(engine)
    return settings


def _status(engine: RAGEngine) -> None:
    stats = engine.stats()
    documents, chunks = int(stats.get("documents", 0)), int(stats.get("vectors", 0))
    if documents:
        st.success(f"**{documents}** document(s) · **{chunks}** vectors", icon="✅")
    else:
        st.info("Knowledge base is empty — add a document to begin.", icon="📥")
    st.caption(
        f"{stats['embedding']} → {stats['vector_store']} · "
        f"{fmt_bytes(stats.get('bytes', 0))} ingested"
    )


def _configuration(settings: Settings) -> Settings:
    st.markdown("#### ⚙️ Configuration")
    with st.expander("Embeddings & vector store", expanded=False):
        settings = _embedding_controls(settings)
    with st.expander("Retrieval", expanded=False):
        settings = _retrieval_controls(settings)
    with st.expander("Generation & evaluation", expanded=False):
        settings = _generation_controls(settings)
    return settings


def _embedding_controls(settings: Settings) -> Settings:
    providers = available_embedders()
    provider = st.selectbox(
        "Embedding provider",
        providers,
        index=providers.index(settings.embedding.provider)
        if settings.embedding.provider in providers
        else 0,
        help="`hashing` needs no downloads or keys — useful for a first run.",
    )
    presets = EMBEDDING_PRESETS.get(provider, [])
    if presets:
        labels = [f"{name.rsplit('/', 1)[-1]} · {dim}d — {note}" for name, dim, note in presets]
        current = next(
            (i for i, (name, _, _) in enumerate(presets) if name == settings.embedding.model), 0
        )
        model = presets[
            st.selectbox("Model", range(len(labels)), format_func=labels.__getitem__, index=current)
        ][0]
    else:
        model = st.text_input("Model", value=settings.embedding.model)

    dimensions = st.number_input(
        "Truncate to N dimensions (0 = native)",
        min_value=0,
        max_value=4096,
        step=64,
        value=int(settings.embedding.dimensions or 0),
        help="Shorter vectors mean less storage and faster search, usually at some "
        "accuracy cost. Only Matryoshka-trained models degrade gracefully.",
    )

    stores = available_vector_stores()
    store = st.selectbox(
        "Vector store",
        stores,
        index=stores.index(settings.vector_store.provider)
        if settings.vector_store.provider in stores
        else 0,
    )
    collection = st.text_input("Collection", value=settings.vector_store.collection)

    changed = (
        provider != settings.embedding.provider
        or model != settings.embedding.model
        or (dimensions or None) != settings.embedding.dimensions
        or store != settings.vector_store.provider
        or collection != settings.vector_store.collection
    )
    if changed:
        st.warning(
            "Changing the model or dimension makes existing vectors unreadable — "
            "use a new collection name, or clear the knowledge base and re-index.",
            icon="⚠️",
        )
        if st.button("Apply & reload", use_container_width=True, type="primary"):
            settings = settings.with_overrides(
                embedding=EmbeddingConfig(
                    provider=provider,
                    model=model,
                    dimensions=int(dimensions) or None,
                    batch_size=settings.embedding.batch_size,
                ),
                vector_store=VectorStoreConfig(
                    provider=store,
                    collection=collection,
                    directory=settings.data_dir / "vectorstore" / store,
                ),
            )
            st.session_state["settings"] = settings
            st.rerun()
    return settings


def _retrieval_controls(settings: Settings) -> Settings:
    cfg = settings.retrieval
    top_k = st.slider(
        "Top-k chunks",
        1,
        20,
        cfg.top_k,
        help="More context improves recall but adds latency and noise.",
    )
    min_score = st.slider(
        "Minimum similarity",
        0.0,
        0.9,
        float(cfg.min_score),
        0.05,
        help="Chunks below this cosine score are discarded as noise.",
    )
    mmr = st.slider(
        "Relevance vs. diversity (MMR λ)",
        0.0,
        1.0,
        float(cfg.mmr_lambda),
        0.05,
        help="1.0 = pure similarity. Lower values spread results across sources.",
    )
    updated = RetrievalConfig(
        top_k=top_k,
        min_score=min_score,
        mmr_lambda=mmr,
        fetch_k_multiplier=cfg.fetch_k_multiplier,
        max_context_chars=cfg.max_context_chars,
    )
    if updated != cfg:
        settings = settings.with_overrides(retrieval=updated)
        st.session_state["settings"] = settings
    return settings


def _generation_controls(settings: Settings) -> Settings:
    providers = available_llms()
    provider = st.selectbox(
        "LLM provider",
        providers,
        index=providers.index(settings.llm.provider) if settings.llm.provider in providers else 0,
        help="`extractive` quotes your documents directly and needs no API key.",
    )
    models = LLM_PRESETS.get(provider, [settings.llm.model])
    model = st.selectbox(
        "Model",
        models,
        index=models.index(settings.llm.model) if settings.llm.model in models else 0,
    )
    judge = st.toggle(
        "LLM-as-judge faithfulness",
        value=settings.evaluation.enable_llm_judge,
        help="More accurate hallucination detection, one extra LLM call per query.",
    )

    if judge != settings.evaluation.enable_llm_judge:
        settings = settings.with_overrides(
            evaluation=EvaluationConfig(
                enable_llm_judge=judge,
                relevance_threshold=settings.evaluation.relevance_threshold,
                faithfulness_threshold=settings.evaluation.faithfulness_threshold,
            )
        )
        st.session_state["settings"] = settings

    if (provider, model) != (settings.llm.provider, settings.llm.model) and st.button(
        "Switch model", use_container_width=True
    ):
        settings = settings.with_overrides(
            llm=LLMConfig(
                provider=provider,
                model=model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                timeout=settings.llm.timeout,
                api_key=None,
            )
        )
        st.session_state["settings"] = settings
        st.rerun()
    return settings


def _footer(engine: RAGEngine) -> None:
    stats = engine.stats()
    st.caption(
        f"**LLM** `{stats['llm']}`  \n"
        f"**Dim** `{stats['dimension']}`  \n"
        f"**Store** `{stats['vector_store']}:{stats['collection']}`"
    )
