"""The RAG engine: the single façade the UI and CLI talk to."""

from rag_assistant.pipeline.engine import RAGEngine, build_engine

__all__ = ["RAGEngine", "build_engine"]
