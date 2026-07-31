"""SQL schema.

Kept as plain SQL (no ORM) because the model is three flat tables and the
queries are simple aggregates — an ORM would add a dependency and indirection
without removing any work. `user_version` gates forward migrations.
"""

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- The authoritative list of active knowledge-base documents. Deleting a row
-- here always happens together with deleting its vectors.
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    source_type   TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    char_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    path          TEXT,
    uri           TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);

-- Every question/answer pair with its latency breakdown and quality metrics:
-- this table *is* the performance monitor's data source.
CREATE TABLE IF NOT EXISTS queries (
    id                 TEXT PRIMARY KEY,
    question           TEXT NOT NULL,
    answer             TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL DEFAULT '',
    llm_provider       TEXT,
    llm_model          TEXT,
    embed_ms           REAL NOT NULL DEFAULT 0,
    retrieve_ms        REAL NOT NULL DEFAULT 0,
    generate_ms        REAL NOT NULL DEFAULT 0,
    evaluate_ms        REAL NOT NULL DEFAULT 0,
    total_ms           REAL NOT NULL DEFAULT 0,
    top_k              INTEGER NOT NULL DEFAULT 0,
    mean_score         REAL NOT NULL DEFAULT 0,
    max_score          REAL NOT NULL DEFAULT 0,
    precision_at_k     REAL NOT NULL DEFAULT 0,
    faithfulness       REAL NOT NULL DEFAULT 0,
    answer_relevance   REAL NOT NULL DEFAULT 0,
    context_diversity  INTEGER NOT NULL DEFAULT 0,
    hallucination_risk INTEGER NOT NULL DEFAULT 0,
    abstained          INTEGER NOT NULL DEFAULT 0,
    judge_verdict      TEXT,
    -- Full hit list (text + scores) as JSON, for the "sources" expander.
    hits               TEXT NOT NULL DEFAULT '[]',
    doc_count          INTEGER NOT NULL DEFAULT 0,
    chunk_count        INTEGER NOT NULL DEFAULT 0,
    feedback           INTEGER,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_queries_created ON queries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_queries_config ON queries(config_fingerprint);

-- Named benchmark runs, so quality can be compared before/after the knowledge
-- base changes or the configuration is swapped.
CREATE TABLE IF NOT EXISTS benchmarks (
    id           TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    doc_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    query_count  INTEGER NOT NULL DEFAULT 0,
    summary      TEXT NOT NULL DEFAULT '{}',
    details      TEXT NOT NULL DEFAULT '[]',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_benchmarks_created ON benchmarks(created_at DESC);
"""
