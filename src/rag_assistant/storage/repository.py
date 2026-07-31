"""Repository over SQLite.

Thread-safe by construction: one connection per thread (Streamlit reruns
handlers on worker threads), WAL mode so readers never block the writer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rag_assistant.core.logging_config import get_logger
from rag_assistant.core.types import Document, JSONDict, RAGAnswer, new_id, now_ts
from rag_assistant.storage.schema import SCHEMA, SCHEMA_VERSION

logger = get_logger(__name__)


class KnowledgeBaseRepository:
    """Metadata + analytics storage. The vector store holds the embeddings."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        with self._init_lock, self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction. `in_transaction` is checked both ways because
        `executescript` (used for the schema) commits implicitly."""
        conn = self._conn
        if not conn.in_transaction:
            conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if conn.in_transaction:
                conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def add_document(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, name, source_type, content_hash, size_bytes,
                                       char_count, chunk_count, path, uri, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    name = excluded.name,
                    chunk_count = excluded.chunk_count,
                    char_count = excluded.char_count,
                    metadata = excluded.metadata
                """,
                (
                    document.id,
                    document.name,
                    document.source_type,
                    document.content_hash,
                    document.size_bytes,
                    document.char_count,
                    document.chunk_count,
                    document.path,
                    document.uri,
                    json.dumps(document.metadata),
                    document.created_at,
                ),
            )

    def get_document(self, document_id: str) -> Document | None:
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._to_document(row) if row else None

    def find_by_hash(self, content_hash: str) -> Document | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return self._to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        rows = self._conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self._to_document(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def clear_documents(self) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM documents").rowcount

    def document_stats(self) -> JSONDict:
        row = self._conn.execute(
            """SELECT COUNT(*) AS documents,
                      COALESCE(SUM(chunk_count), 0) AS chunks,
                      COALESCE(SUM(char_count), 0) AS characters,
                      COALESCE(SUM(size_bytes), 0) AS bytes
               FROM documents"""
        ).fetchone()
        return dict(row) if row else {"documents": 0, "chunks": 0, "characters": 0, "bytes": 0}

    def log_answer(self, answer: RAGAnswer, *, doc_count: int = 0, chunk_count: int = 0) -> None:
        timings, metrics = answer.timings, answer.metrics
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO queries (
                    id, question, answer, config_fingerprint, llm_provider, llm_model,
                    embed_ms, retrieve_ms, generate_ms, evaluate_ms, total_ms, top_k,
                    mean_score, max_score, precision_at_k, faithfulness, answer_relevance,
                    context_diversity, hallucination_risk, abstained, judge_verdict,
                    hits, doc_count, chunk_count, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    answer.id,
                    answer.question,
                    answer.answer,
                    answer.config_fingerprint,
                    answer.llm.provider if answer.llm else None,
                    answer.llm.model if answer.llm else None,
                    timings.embed_ms,
                    timings.retrieve_ms,
                    timings.generate_ms,
                    timings.evaluate_ms,
                    timings.total_ms,
                    len(answer.hits),
                    metrics.mean_score,
                    metrics.max_score,
                    metrics.precision_at_k,
                    metrics.faithfulness,
                    metrics.answer_relevance,
                    metrics.context_diversity,
                    int(metrics.hallucination_risk),
                    int(metrics.abstained),
                    metrics.judge_verdict,
                    json.dumps([hit.to_dict() for hit in answer.hits]),
                    doc_count,
                    chunk_count,
                    answer.created_at,
                ),
            )

    def set_feedback(self, query_id: str, value: int) -> None:
        """Thumbs up/down (+1 / -1) from the UI."""
        with self._connect() as conn:
            conn.execute("UPDATE queries SET feedback = ? WHERE id = ?", (value, query_id))

    def recent_queries(self, limit: int = 50, *, fingerprint: str | None = None) -> list[JSONDict]:
        sql = "SELECT * FROM queries"
        params: list[Any] = []
        if fingerprint:
            sql += " WHERE config_fingerprint = ?"
            params.append(fingerprint)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._to_query(row) for row in rows]

    def search_queries(self, term: str, limit: int = 50) -> list[JSONDict]:
        rows = self._conn.execute(
            """SELECT * FROM queries
               WHERE question LIKE ? OR answer LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (f"%{term}%", f"%{term}%", limit),
        ).fetchall()
        return [self._to_query(row) for row in rows]

    def performance_summary(self, *, since: float | None = None) -> JSONDict:
        sql = """
            SELECT COUNT(*)                        AS queries,
                   AVG(total_ms)                   AS avg_total_ms,
                   AVG(embed_ms)                   AS avg_embed_ms,
                   AVG(retrieve_ms)                AS avg_retrieve_ms,
                   AVG(generate_ms)                AS avg_generate_ms,
                   AVG(mean_score)                 AS avg_mean_score,
                   AVG(precision_at_k)             AS avg_precision,
                   AVG(faithfulness)               AS avg_faithfulness,
                   AVG(answer_relevance)           AS avg_answer_relevance,
                   AVG(CAST(hallucination_risk AS REAL)) AS hallucination_rate,
                   AVG(CAST(abstained AS REAL))    AS abstention_rate
            FROM queries
        """
        params: tuple = ()
        if since is not None:
            sql += " WHERE created_at >= ?"
            params = (since,)
        row = self._conn.execute(sql, params).fetchone()
        summary = {k: (v if v is not None else 0.0) for k, v in dict(row or {}).items()}
        summary["p95_total_ms"] = self._percentile("total_ms", 0.95, since)
        return summary

    def _percentile(self, column: str, q: float, since: float | None) -> float:
        sql = f"SELECT {column} FROM queries"  # noqa: S608 - column is a literal above
        params: tuple = ()
        if since is not None:
            sql += " WHERE created_at >= ?"
            params = (since,)
        sql += f" ORDER BY {column}"
        values = [row[0] for row in self._conn.execute(sql, params).fetchall()]
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return float(values[index])

    def latency_series(self, limit: int = 200) -> list[JSONDict]:
        """Oldest→newest slice for the trend charts."""
        rows = self._conn.execute(
            """SELECT created_at, total_ms, embed_ms, retrieve_ms, generate_ms,
                      mean_score, faithfulness, answer_relevance, doc_count, chunk_count
               FROM queries ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def summary_by_config(self) -> list[JSONDict]:
        rows = self._conn.execute(
            """SELECT config_fingerprint,
                      COUNT(*)            AS queries,
                      AVG(total_ms)        AS avg_total_ms,
                      AVG(mean_score)      AS avg_mean_score,
                      AVG(precision_at_k)  AS avg_precision,
                      AVG(faithfulness)    AS avg_faithfulness,
                      AVG(CAST(hallucination_risk AS REAL)) AS hallucination_rate
               FROM queries GROUP BY config_fingerprint ORDER BY queries DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def clear_queries(self) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM queries").rowcount

    def save_benchmark(
        self,
        *,
        label: str,
        fingerprint: str,
        doc_count: int,
        chunk_count: int,
        summary: JSONDict,
        details: list[JSONDict],
    ) -> str:
        benchmark_id = new_id()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO benchmarks (id, label, fingerprint, doc_count, chunk_count,
                                           query_count, summary, details, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    benchmark_id,
                    label,
                    fingerprint,
                    doc_count,
                    chunk_count,
                    len(details),
                    json.dumps(summary),
                    json.dumps(details),
                    now_ts(),
                ),
            )
        return benchmark_id

    def list_benchmarks(self, limit: int = 50) -> list[JSONDict]:
        rows = self._conn.execute(
            "SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["summary"] = json.loads(item["summary"])
            item["details"] = json.loads(item["details"])
            result.append(item)
        return result

    def delete_benchmark(self, benchmark_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM benchmarks WHERE id = ?", (benchmark_id,)).rowcount > 0

    @staticmethod
    def _to_document(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            name=row["name"],
            source_type=row["source_type"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            char_count=row["char_count"],
            chunk_count=row["chunk_count"],
            path=row["path"],
            uri=row["uri"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_query(row: sqlite3.Row) -> JSONDict:
        item = dict(row)
        try:
            item["hits"] = json.loads(item.get("hits") or "[]")
        except json.JSONDecodeError:
            item["hits"] = []
        item["hallucination_risk"] = bool(item.get("hallucination_risk"))
        item["abstained"] = bool(item.get("abstained"))
        return item
