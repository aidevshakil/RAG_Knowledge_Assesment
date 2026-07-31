"""App bootstrap: import path, settings, and the cached engine.

`build_engine` loads an embedding model and opens a database, so it must run
once per session rather than on every Streamlit rerun. `st.cache_resource`
handles that; the cache key is the config fingerprint, so changing a backend in
the sidebar transparently rebuilds the engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from rag_assistant.config import (  # noqa: E402
    EmbeddingConfig,
    EvaluationConfig,
    LLMConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
)
from rag_assistant.core.logging_config import configure_logging  # noqa: E402
from rag_assistant.pipeline.engine import RAGEngine, build_engine  # noqa: E402

__all__ = [
    "EmbeddingConfig",
    "EvaluationConfig",
    "LLMConfig",
    "RetrievalConfig",
    "Settings",
    "VectorStoreConfig",
    "base_settings",
    "get_engine",
]


def _load_streamlit_secrets() -> None:
    """Copy `st.secrets` into the environment so hosted deploys configure the
    same way local `.env` runs do — `Settings` only ever reads os.environ.

    A real environment variable always wins, and accessing `st.secrets` when no
    secrets are configured raises, so the whole thing is best-effort.
    """
    try:
        secrets = st.secrets
    except Exception:
        return
    try:
        items = list(secrets.items())
    except Exception:
        return
    for key, value in items:
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(str(key), str(value))


@st.cache_resource(show_spinner=False)
def base_settings() -> Settings:
    _load_streamlit_secrets()
    settings = Settings.from_env()
    settings.ensure_dirs()
    configure_logging(settings.log_level)
    return settings


@st.cache_resource(show_spinner="Loading models and index…", max_entries=3)
def get_engine(fingerprint: str, _settings: Settings) -> RAGEngine:
    """Cached engine. `fingerprint` is the cache key; `_settings` is not hashed."""
    return build_engine(_settings)
