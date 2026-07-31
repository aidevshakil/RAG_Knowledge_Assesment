"""RAGEngine — composition root and public API.

Everything the app does goes through one object:

    engine = build_engine(settings)
    engine.ingest_files([("notes.pdf", data)])   # add sources
    engine.delete_document(doc_id)               # remove sources + their vectors
    answer = engine.ask("what changed in Q3?")   # retrieve -> generate -> evaluate

The engine owns the invariant that matters most for a *dynamic* knowledge base:
**the SQLite document table and the vector index never disagree.** Ingestion
writes vectors first and metadata second; deletion removes vectors first and
metadata second. A crash in between leaves orphans, so `reconcile()` detects and
repairs them, and the UI calls it on startup.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import RAGError
from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.text import sha256_bytes, sha256_text
from rag_assistant.core.timing import timed
from rag_assistant.core.types import (
    Chunk,
    Document,
    IngestionResult,
    JSONDict,
    LLMResult,
    QualityMetrics,
    RAGAnswer,
    SearchHit,
    Timings,
    new_id,
)
from rag_assistant.embeddings.base import Embedder
from rag_assistant.embeddings.factory import build_embedder
from rag_assistant.evaluation.evaluator import Evaluator
from rag_assistant.generation.prompts import (
    ABSTAIN_MESSAGE,
    RAG_SYSTEM_PROMPT,
    build_generation_prompt,
)
from rag_assistant.ingestion.chunking import RecursiveChunker
from rag_assistant.ingestion.loaders import LoadedDocument, load_bytes, load_path, load_url
from rag_assistant.llm.base import LLMClient
from rag_assistant.llm.factory import build_llm
from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.storage.repository import KnowledgeBaseRepository
from rag_assistant.vectorstores.base import VectorStore
from rag_assistant.vectorstores.factory import build_vector_store

logger = get_logger(__name__)

_INDEX_BATCH = 256


class RAGEngine:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        repository: KnowledgeBaseRepository,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.repository = repository
        self.chunker = RecursiveChunker(settings.chunking)
        self.retriever = Retriever(embedder, store, settings.retrieval)
        self.evaluator = Evaluator(settings.evaluation, embedder, llm)
        self._write_lock = threading.Lock()

    def ingest_files(
        self,
        files: Iterable[tuple[str, bytes]],
        *,
        save_copy: bool = True,
        progress: ProgressFn | None = None,
    ) -> IngestionResult:
        """Add uploaded files. Duplicate content (same sha256) is skipped."""
        result = IngestionResult()
        items = list(files)
        with timed() as elapsed:
            for position, (name, data) in enumerate(items):
                _report(progress, position, len(items), f"Reading {name}")
                try:
                    loaded = load_bytes(data, name)
                except RAGError as exc:
                    logger.warning("Skipping %s: %s", name, exc)
                    result.failed.append((name, str(exc)))
                    continue
                self._ingest_loaded(
                    loaded,
                    name=name,
                    content_hash=sha256_bytes(data),
                    size_bytes=len(data),
                    raw=data if save_copy else None,
                    result=result,
                    progress=progress,
                    position=position,
                    total=len(items),
                )
        result.duration_ms = elapsed[0]
        _report(progress, len(items), len(items), "Done")
        return result

    def ingest_paths(
        self, paths: Iterable[str | Path], *, progress: ProgressFn | None = None
    ) -> IngestionResult:
        """Add files from disk (CLI path). Directories are walked recursively."""
        expanded: list[Path] = []
        for entry in paths:
            path = Path(entry).expanduser()
            if path.is_dir():
                expanded.extend(sorted(p for p in path.rglob("*") if p.is_file()))
            elif path.is_file():
                expanded.append(path)

        result = IngestionResult()
        with timed() as elapsed:
            for position, path in enumerate(expanded):
                _report(progress, position, len(expanded), f"Reading {path.name}")
                try:
                    loaded = load_path(path)
                except RAGError as exc:
                    result.failed.append((path.name, str(exc)))
                    continue
                self._ingest_loaded(
                    loaded,
                    name=path.name,
                    content_hash=sha256_text(loaded.text),
                    size_bytes=path.stat().st_size,
                    path=str(path.resolve()),
                    result=result,
                    progress=progress,
                    position=position,
                    total=len(expanded),
                )
        result.duration_ms = elapsed[0]
        return result

    def ingest_url(self, url: str, *, progress: ProgressFn | None = None) -> IngestionResult:
        result = IngestionResult()
        with timed() as elapsed:
            try:
                loaded = load_url(url)
            except RAGError as exc:
                result.failed.append((url, str(exc)))
            else:
                self._ingest_loaded(
                    loaded,
                    name=_url_title(url),
                    content_hash=sha256_text(loaded.text),
                    size_bytes=len(loaded.text.encode("utf-8")),
                    uri=url,
                    result=result,
                    progress=progress,
                    position=0,
                    total=1,
                )
        result.duration_ms = elapsed[0]
        return result

    def ingest_text(self, text: str, name: str = "pasted-text") -> IngestionResult:
        result = IngestionResult()
        with timed() as elapsed:
            self._ingest_loaded(
                LoadedDocument(text=text, source_type="raw"),
                name=name,
                content_hash=sha256_text(text),
                size_bytes=len(text.encode("utf-8")),
                result=result,
            )
        result.duration_ms = elapsed[0]
        return result

    def _ingest_loaded(
        self,
        loaded: LoadedDocument,
        *,
        name: str,
        content_hash: str,
        size_bytes: int,
        result: IngestionResult,
        raw: bytes | None = None,
        path: str | None = None,
        uri: str | None = None,
        progress: ProgressFn | None = None,
        position: int = 0,
        total: int = 1,
    ) -> None:
        """Chunk → embed → index → record. The one write path for all sources."""
        existing = self.repository.find_by_hash(content_hash)
        if existing:
            result.skipped.append((name, f"identical content already indexed as '{existing.name}'"))
            return

        document_id = new_id()
        chunks = self.chunker.split(loaded, document_id=document_id, document_name=name)
        if not chunks:
            result.failed.append((name, "produced no chunks"))
            return

        with self._write_lock:
            try:
                indexed = self._index_chunks(chunks, progress, position, total, name)
            except Exception as exc:
                logger.exception("Indexing %s failed", name)
                self.store.delete_document(document_id)
                result.failed.append((name, str(exc)))
                return

            stored_path = self._save_copy(raw, name, document_id) if raw else path
            document = Document(
                id=document_id,
                name=name,
                source_type=loaded.source_type,
                content_hash=content_hash,
                size_bytes=size_bytes,
                char_count=len(loaded.text),
                chunk_count=indexed,
                path=stored_path,
                uri=uri,
                metadata=loaded.metadata,
            )
            self.repository.add_document(document)

        result.documents.append(document)
        result.chunks_added += indexed
        logger.info("Indexed %s: %d chunks", name, indexed)

    def _index_chunks(
        self,
        chunks: list[Chunk],
        progress: ProgressFn | None,
        position: int,
        total: int,
        name: str,
    ) -> int:
        indexed = 0
        for start in range(0, len(chunks), _INDEX_BATCH):
            batch = chunks[start : start + _INDEX_BATCH]
            vectors = self.embedder.embed_documents([chunk.text for chunk in batch])
            indexed += self.store.add(batch, vectors)
            if progress:
                inner = (start + len(batch)) / len(chunks)
                _report(
                    progress,
                    position + inner,
                    total,
                    f"Embedding {name} ({start + len(batch)}/{len(chunks)} chunks)",
                )
        self.store.persist()
        return indexed

    def _save_copy(self, data: bytes, name: str, document_id: str) -> str:
        """Keep the original file so it can be previewed or re-indexed later."""
        target = self.settings.upload_dir / f"{document_id}_{Path(name).name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    def delete_document(self, document_id: str, *, delete_file: bool = True) -> bool:
        """Remove a document, its vectors and its cached copy.

        Vectors go first: if the process dies mid-way the document row is still
        present and `reconcile()` can re-index or clean up, whereas the reverse
        order would leave unreachable vectors polluting every future search.
        """
        document = self.repository.get_document(document_id)
        if document is None:
            return False
        with self._write_lock:
            removed = self.store.delete_document(document_id)
            self.store.persist()
            self.repository.delete_document(document_id)
            if delete_file and document.path:
                path = Path(document.path)
                if path.is_file() and path.parent == self.settings.upload_dir:
                    path.unlink(missing_ok=True)
        logger.info("Deleted %s (%d vectors removed)", document.name, removed)
        return True

    def clear_knowledge_base(self) -> None:
        with self._write_lock:
            self.store.reset()
            self.repository.clear_documents()
            for file in self.settings.upload_dir.glob("*"):
                if file.is_file():
                    file.unlink(missing_ok=True)
        logger.info("Knowledge base cleared")

    def reconcile(self) -> JSONDict:
        """Repair drift between the metadata table and the vector index."""
        indexed_ids = self.store.document_ids()
        known = {doc.id: doc for doc in self.repository.list_documents()}

        orphan_vectors = indexed_ids - known.keys()
        for document_id in orphan_vectors:
            self.store.delete_document(document_id)

        missing_vectors = [doc for doc_id, doc in known.items() if doc_id not in indexed_ids]
        if orphan_vectors:
            self.store.persist()
        report = {
            "orphan_vector_documents": len(orphan_vectors),
            "documents_missing_vectors": [doc.name for doc in missing_vectors],
        }
        if orphan_vectors or missing_vectors:
            logger.warning("Reconcile: %s", report)
        return report

    def list_documents(self) -> list[Document]:
        return self.repository.list_documents()

    def stats(self) -> JSONDict:
        docs = self.repository.document_stats()
        return {
            **docs,
            "vectors": self.store.count(),
            "embedding": self.embedder.label,
            "dimension": self.embedder.dimension,
            "vector_store": self.store.provider,
            "collection": self.store.collection,
            "llm": self.llm.label,
            "fingerprint": self.settings.fingerprint,
        }

    @property
    def is_empty(self) -> bool:
        return self.store.count() == 0

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        history: list[tuple[str, str]] | None = None,
        evaluate: bool = True,
        log: bool = True,
    ) -> RAGAnswer:
        """Full RAG turn: retrieve → generate → evaluate → log."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        hits, query_vector, embed_ms, retrieve_ms = self.retriever.retrieve(
            question, top_k=top_k, document_ids=document_ids
        )
        timings = Timings(embed_ms=embed_ms, retrieve_ms=retrieve_ms)

        if not hits:
            return self._finalize(
                question,
                self._empty_answer(),
                [],
                timings,
                QualityMetrics(abstained=True, faithfulness=1.0),
                None,
                log,
            )

        with timed() as generate_ms:
            llm_result = self.llm.complete(
                RAG_SYSTEM_PROMPT, build_generation_prompt(question, hits, history=history)
            )
        timings.generate_ms = generate_ms[0]

        metrics = QualityMetrics()
        if evaluate:
            with timed() as evaluate_ms:
                metrics = self.evaluator.evaluate(
                    question, llm_result.text, hits, query_vector=query_vector
                )
            timings.evaluate_ms = evaluate_ms[0]

        return self._finalize(question, llm_result.text, hits, timings, metrics, llm_result, log)

    def ask_stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> Iterator[str | RAGAnswer]:
        """Yield answer tokens, then the finished `RAGAnswer` as the last item.

        Retrieval and evaluation are identical to `ask`; only generation streams,
        so the UI can render tokens while metrics are computed at the end.
        """
        question = question.strip()
        hits, query_vector, embed_ms, retrieve_ms = self.retriever.retrieve(
            question, top_k=top_k, document_ids=document_ids
        )
        timings = Timings(embed_ms=embed_ms, retrieve_ms=retrieve_ms)

        if not hits:
            message = self._empty_answer()
            yield message
            yield self._finalize(
                question,
                message,
                [],
                timings,
                QualityMetrics(abstained=True, faithfulness=1.0),
                None,
                True,
            )
            return

        prompt = build_generation_prompt(question, hits, history=history)
        parts: list[str] = []
        with timed() as generate_ms:
            for token in self.llm.stream(RAG_SYSTEM_PROMPT, prompt):
                parts.append(token)
                yield token
        text = "".join(parts).strip()
        timings.generate_ms = generate_ms[0]

        with timed() as evaluate_ms:
            metrics = self.evaluator.evaluate(question, text, hits, query_vector=query_vector)
        timings.evaluate_ms = evaluate_ms[0]

        yield self._finalize(
            question,
            text,
            hits,
            timings,
            metrics,
            LLMResult(
                text=text,
                model=self.llm.config.model,
                provider=self.llm.provider,
                latency_ms=timings.generate_ms,
            ),
            True,
        )

    def retrieve_only(
        self, question: str, *, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[SearchHit]:
        """Search without generating — used by the retrieval-quality inspector."""
        hits, _, _, _ = self.retriever.retrieve(question, top_k=top_k, document_ids=document_ids)
        return hits

    def _empty_answer(self) -> str:
        if self.is_empty:
            return (
                "The knowledge base is empty. Upload a document in the "
                "**Knowledge Base** tab and ask again."
            )
        return ABSTAIN_MESSAGE

    def _finalize(
        self,
        question: str,
        answer: str,
        hits: list[SearchHit],
        timings: Timings,
        metrics: QualityMetrics,
        llm_result: LLMResult | None,
        log: bool,
    ) -> RAGAnswer:
        response = RAGAnswer(
            id=new_id(),
            question=question,
            answer=answer,
            hits=hits,
            timings=timings,
            metrics=metrics,
            llm=llm_result,
            config_fingerprint=self.settings.fingerprint,
        )
        if log:
            try:
                stats = self.repository.document_stats()
                self.repository.log_answer(
                    response,
                    doc_count=int(stats.get("documents", 0)),
                    chunk_count=int(stats.get("chunks", 0)),
                )
            except Exception as exc:
                logger.warning("Could not log query: %s", exc)
        return response

    def close(self) -> None:
        self.store.close()
        self.repository.close()


ProgressFn = "callable"


def _report(progress, position: float, total: int, message: str) -> None:
    if progress:
        progress(min(1.0, position / total) if total else 1.0, message)


def _url_title(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1] or url
    return tail[:120]


def build_engine(settings: Settings | None = None) -> RAGEngine:
    """Composition root: wire the configured backends into an engine."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    embedder = build_embedder(settings.embedding)
    store = build_vector_store(settings, embedder.dimension)
    llm = build_llm(settings.llm)
    repository = KnowledgeBaseRepository(settings.db_path)

    engine = RAGEngine(settings, embedder, store, llm, repository)
    engine.reconcile()
    return engine
