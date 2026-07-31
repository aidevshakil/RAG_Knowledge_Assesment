# RAG Assistant with Real-Time Source Control & Performance Benchmarking

A production-shaped Retrieval-Augmented Generation stack: upload documents, ask
questions grounded in them, add or delete sources at any time, and **measure
what every change does to retrieval quality, latency and hallucination rate**.

Two things distinguish it from a tutorial RAG demo:

1. **The knowledge base is genuinely dynamic.** Deleting a file removes its
   vectors from the index in the same operation — it can never be retrieved
   again. A `reconcile()` pass repairs any drift between the metadata table and
   the vector index.
2. **Everything is measured.** Every query records its latency breakdown,
   retrieval scores, grounding and answer relevance. A benchmark runner lets you
   snapshot quality *before* and *after* a change to the documents or the
   configuration and diff the two.

---

## Quick start

```bash
make install          # venv + all dependencies
cp .env.example .env  # then add a GROQ_API_KEY (free at console.groq.com/keys)
make run              # http://localhost:8501
```

**No keys, no downloads?** The app runs fully offline out of the box:

```bash
make install-min
EMBEDDING_PROVIDER=hashing VECTOR_STORE=numpy LLM_PROVIDER=extractive make run
```

That path uses a hashing embedder, a NumPy index and extractive answers — every
feature works, answers are quoted rather than synthesised. Useful for a first
run, for CI, and for dimensionality experiments.

### Command line

```bash
python -m rag_assistant.cli ingest docs/ report.pdf     # files or folders
python -m rag_assistant.cli ingest --url https://…      # web pages / remote PDFs
python -m rag_assistant.cli ls                          # active documents
python -m rag_assistant.cli rm <document-id>            # delete doc + vectors
python -m rag_assistant.cli ask "what is our refund policy?" --sources
python -m rag_assistant.cli benchmark --suite eval/questions.example.json --label before
python -m rag_assistant.cli stats
```

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
 PDF/DOCX/TXT/CSV   │  Ingestion                              │
 HTML/URL/text  ──► │  loaders → RecursiveChunker → Embedder  │
                    └───────────────┬─────────────────────────┘
                                    │ vectors + metadata
                    ┌───────────────▼─────────────────────────┐
                    │  Vector store (chroma | faiss | numpy)  │◄── delete_document()
                    └───────────────┬─────────────────────────┘
                                    │ top-k
 question ─► Embedder ─► Retriever (score filter → MMR → context budget)
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │  LLM (groq | openai | anthropic | extractive)
                    └───────────────┬─────────────────────────┘
                                    │ answer
                    ┌───────────────▼─────────────────────────┐
                    │  Evaluator → SQLite → Streamlit dashboards
                    └─────────────────────────────────────────┘
```

Every swappable part sits behind a small interface plus a registry, so adding a
backend is one class and one decorator — no factory to edit:

```python
@VECTOR_STORES.register("qdrant")
class QdrantVectorStore(VectorStore):
    ...   # add / search / delete_document / count / document_ids / reset
```

### Project layout

```
├── app/                          Streamlit UI (presentation only — no logic)
│   ├── main.py                   entry point: streamlit run app/main.py
│   ├── bootstrap.py              cached engine + settings
│   └── ui/
│       ├── sidebar.py            live configuration + status
│       ├── chat_tab.py           streaming chat, citations, evidence inspector
│       ├── documents_tab.py      Data Source Manager: add / list / delete
│       ├── performance_tab.py    Performance Monitor dashboard
│       ├── insights_tab.py       stored query & answer history
│       ├── benchmark_tab.py      before/after snapshots + config sweeps
│       └── components.py         shared widgets and formatters
│
├── src/rag_assistant/
│   ├── config.py                 typed, env-driven settings (the only env reader)
│   ├── cli.py                    scriptable equivalent of the whole UI
│   ├── core/                     types, exceptions, registry, logging, timing, text
│   ├── ingestion/
│   │   ├── loaders.py            PDF/DOCX/TXT/MD/CSV/HTML/URL → text + page offsets
│   │   └── chunking.py           recursive character chunker with overlap
│   ├── embeddings/               base + huggingface / openai / hashing
│   ├── vectorstores/             base + chroma / faiss / numpy
│   ├── retrieval/                retriever + MMR re-ranking
│   ├── generation/prompts.py     grounding rules, citation format, judge prompt
│   ├── evaluation/               metrics, per-query evaluator, benchmark runner
│   ├── storage/                  SQLite schema + repository
│   └── pipeline/engine.py        RAGEngine — the composition root & public API
│
├── tests/                        68 tests, fully offline
├── eval/questions.example.json   labelled benchmark suite
└── Makefile / pyproject.toml / .env.example
```

---

## How the pieces work

### Ingestion

Files are hashed (sha256) before indexing, so re-uploading the same content is
skipped rather than duplicated. PDFs keep per-page character offsets, which is
how a citation can say `handbook.pdf (p.7)` instead of just naming the file.

Chunking is a recursive character splitter: it tries paragraph → line → sentence
→ word → character, taking the most semantic separator that fits the target
size, then packs pieces greedily with a configurable overlap. Fragments below
`min_chunk_chars` are folded into the previous chunk rather than dropped, so no
text is silently lost.

### Dynamic sources — the delete path

This is the part most RAG demos get wrong: they remove the row from the UI and
leave the vectors in the index, so deleted content keeps surfacing in answers.

Here, `engine.delete_document(id)`:

1. removes every vector for that document from the store (a real delete —
   Chroma `where` filter, FAISS `remove_ids`, NumPy row mask),
2. deletes the metadata row,
3. removes the cached copy of the upload.

Vectors go first on purpose. If the process dies between steps, the document row
still exists and `reconcile()` can see the inconsistency — the reverse order
would leave unreachable vectors polluting every future search. `reconcile()`
runs at startup and is exposed as *Maintenance → Run consistency check*.

`tests/test_engine.py::test_deleted_content_is_not_retrievable` pins the
behaviour end to end.

### Retrieval

```
embed query → over-fetch (top_k × 4) → drop below min_score → MMR re-rank → context budget
```

MMR trades a little similarity for diversity, which stops the context window
filling with five near-identical chunks from the same page. `MMR_LAMBDA=1.0`
disables it; lower values spread hits across sources.

`min_score` is treated as advisory: absolute cosine ranges differ a lot between
embedding models, so if nothing clears the bar the plain ranking is used and the
grounding check decides whether to abstain.

### Evaluation

Reference-free metrics run on **every** query, so the dashboard is always
populated without a labelled dataset:

| Metric | What it catches |
| --- | --- |
| `mean_score` / `max_score` | Is anything in the index actually close to the question? |
| `precision_at_k` | Share of retrieved chunks above the relevance threshold — low means the context is padded with noise |
| `faithfulness` | Fraction of the answer's content words traceable to the context — catches invented specifics |
| `answer_relevance` | Question↔answer cosine — catches the "fluent but off-topic" failure that faithfulness misses |
| `hallucination_risk` | Grounding below the threshold |
| `abstained` | Model correctly declined for lack of context (scored as faithful, not as a failure) |
| latency breakdown | embed / retrieve / generate / evaluate, separately |

Set `ENABLE_LLM_JUDGE=true` to add an LLM-as-judge faithfulness verdict; its
score is averaged with the lexical one so a single noisy judgement can't swing
the metric.

Reference-based metrics (`hit_rate`, `recall@k`, `MRR`, `nDCG`,
`answer_correctness`) activate when a benchmark suite supplies
`expected_sources` / `expected_answer_terms`.

### Benchmarking

**Snapshots** — run a suite against the current knowledge base, save it, change
your documents, run it again. The Benchmark tab diffs any two snapshots and
marks each metric 🟢 better / 🔴 worse, alongside the corpus size at each point.
Benchmark traffic is excluded from the live query log so it can't skew the
dashboard.

**Config sweeps** — index a copy of selected documents under several embedding
models, dimensions and vector stores in a throwaway collection, and score the
same questions against each. Retrieval-only by default: it isolates the
embedding/store effect from LLM variance and costs no tokens. The result is one
row per configuration — shape shown below, numbers depend on your corpus:

```
variant           dimension  vector_store  chunks  index_ms  mean_score  precision_at_k
MiniLM-384d       384        chroma        142     1830      0.612       0.80
MPNet-768d        768        chroma        142     5410      0.671       0.85
OpenAI-1536d      1536       chroma        142     2210      0.704       0.90
OpenAI-512d       512        chroma        142     2180      0.688       0.85
```

That table is the point of the exercise: higher dimensions usually buy accuracy
at the cost of index time, memory and query latency — and the OpenAI 1536 → 512
row shows how gracefully a Matryoshka-trained model degrades when truncated.

---

## Configuration

Everything is environment-driven (see `.env.example` for the annotated list).

| Variable | Default | Notes |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` · `openai` · `hashing` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384d; `all-mpnet-base-v2` = 768d; `bge-large-en-v1.5` = 1024d |
| `EMBEDDING_DIMENSIONS` | *(native)* | Truncate vectors — server-side for OpenAI v3, client-side otherwise |
| `VECTOR_STORE` | `chroma` | `chroma` · `faiss` · `numpy` |
| `LLM_PROVIDER` | `groq` | `groq` · `openai` · `anthropic` · `extractive` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Characters |
| `TOP_K` | `5` | Chunks passed to the LLM |
| `MIN_SCORE` | `0.15` | Cosine floor (advisory) |
| `MMR_LAMBDA` | `0.7` | 1.0 = pure relevance, lower = more diverse |
| `ENABLE_LLM_JUDGE` | `false` | Adds one LLM call per query |

If the configured LLM has no API key or its SDK isn't installed, the engine logs
a warning and falls back to extractive answers rather than failing — retrieval
and the dashboards stay usable.

### Choosing a vector store

| | Chroma | FAISS | NumPy |
| --- | --- | --- | --- |
| Setup | `pip install chromadb` | `pip install faiss-cpu` | none |
| Metadata filtering | native `where` | post-filter (over-fetch) | native |
| Scale | ~10⁶ chunks | ~10⁷ chunks, fastest search | ~10⁵ chunks |
| Best for | the default | large local corpora | demos, CI, experiments |

All three pass the same contract test suite
(`tests/test_vectorstores.py`), so switching is a config change. Pinecone,
Qdrant, Weaviate, LanceDB, pgvector or MongoDB Atlas slot in the same way —
implement six methods, register the class.

---

## Development

```bash
make test     # 68 tests, fully offline (hashing + numpy + extractive)
make lint     # ruff check + format --check
make fmt      # auto-format
```

Tests cover the chunker, the loaders, the embedding contract, a shared vector
store contract run against every installed backend, the engine's ingest/delete/
reconcile lifecycle, the metrics, and a Streamlit `AppTest` smoke test that runs
the real UI and asserts no exceptions in any tab.

### Known limitations

- Scanned PDFs need OCR first (`ocrmypdf`); the loader raises a clear error.
- Absolute similarity scores aren't comparable across embedding models — the
  `hashing` backend in particular produces much lower cosines than a semantic
  model, so `MIN_SCORE` and `RELEVANCE_THRESHOLD` are tuned for the latter.
- MMR re-embeds candidate texts at query time (stores don't return vectors).
  That's a few milliseconds locally but a real cost with a hosted embedding API
  — set `MMR_LAMBDA=1.0` to skip it.
- The extractive LLM quotes sentences; it cannot synthesise across sources.
