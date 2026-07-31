"""Command-line interface — everything the UI can do, scriptable.

python -m rag_assistant.cli ingest docs/ report.pdf
python -m rag_assistant.cli ingest --url https://example.com/page
python -m rag_assistant.cli ls
python -m rag_assistant.cli rm <document-id>
python -m rag_assistant.cli ask "what is our refund policy?"
python -m rag_assistant.cli benchmark --suite eval/questions.json --label before
python -m rag_assistant.cli stats
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_assistant.config import get_settings
from rag_assistant.core.exceptions import RAGError
from rag_assistant.core.logging_config import configure_logging
from rag_assistant.evaluation.benchmark import BenchmarkRunner, load_suite
from rag_assistant.pipeline.engine import build_engine


def _progress(fraction: float, message: str) -> None:
    bar = "█" * int(fraction * 24)
    sys.stderr.write(f"\r[{bar:<24}] {fraction:5.0%} {message[:52]:<52}")
    sys.stderr.flush()
    if fraction >= 1.0:
        sys.stderr.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="add files, folders or URLs")
    ingest.add_argument("paths", nargs="*", help="files or directories")
    ingest.add_argument("--url", action="append", default=[], help="web page to ingest")

    sub.add_parser("ls", help="list indexed documents")

    remove = sub.add_parser("rm", help="delete a document and its vectors")
    remove.add_argument("document_id")

    sub.add_parser("clear", help="delete every document and vector")
    sub.add_parser("stats", help="show knowledge base statistics")
    sub.add_parser("reconcile", help="repair drift between metadata and vectors")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", "--top-k", type=int, default=None)
    ask.add_argument("--json", action="store_true", help="emit the full result as JSON")
    ask.add_argument("--sources", action="store_true", help="print retrieved chunks")

    bench = sub.add_parser("benchmark", help="run a question suite and save a snapshot")
    bench.add_argument("--suite", required=True, help="path to a JSON question suite")
    bench.add_argument("--label", default="run")
    bench.add_argument("--retrieval-only", action="store_true", help="skip LLM generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(args.log_level or settings.log_level, force=bool(args.log_level))

    engine = build_engine(settings)
    try:
        return _dispatch(args, engine)
    except RAGError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        engine.close()


def _dispatch(args: argparse.Namespace, engine) -> int:  # noqa: PLR0911 - flat command table
    if args.command == "ingest":
        if not args.paths and not args.url:
            print("nothing to ingest", file=sys.stderr)
            return 1
        if args.paths:
            result = engine.ingest_paths(args.paths, progress=_progress)
            _print_ingestion(result)
        for url in args.url:
            _print_ingestion(engine.ingest_url(url, progress=_progress))
        return 0

    if args.command == "ls":
        documents = engine.list_documents()
        if not documents:
            print("(knowledge base is empty)")
            return 0
        print(f"{'ID':<34}{'CHUNKS':>7}  {'TYPE':<6} NAME")
        for doc in documents:
            print(f"{doc.id:<34}{doc.chunk_count:>7}  {doc.source_type:<6} {doc.name}")
        return 0

    if args.command == "rm":
        if engine.delete_document(args.document_id):
            print(f"deleted {args.document_id}")
            return 0
        print(f"no such document: {args.document_id}", file=sys.stderr)
        return 1

    if args.command == "clear":
        engine.clear_knowledge_base()
        print("knowledge base cleared")
        return 0

    if args.command == "stats":
        print(json.dumps(engine.stats(), indent=2, default=str))
        return 0

    if args.command == "reconcile":
        print(json.dumps(engine.reconcile(), indent=2))
        return 0

    if args.command == "ask":
        answer = engine.ask(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(answer.to_dict(), indent=2))
            return 0
        print(f"\n{answer.answer}\n")
        if answer.citations:
            print("Sources: " + "; ".join(answer.citations))
        if args.sources:
            for hit in answer.hits:
                print(f"\n  [{hit.rank + 1}] {hit.chunk.citation}  score={hit.score:.3f}")
                print("      " + hit.text[:300].replace("\n", " "))
        timings, metrics = answer.timings, answer.metrics
        print(
            f"\n{timings.total_ms:.0f}ms total "
            f"(embed {timings.embed_ms:.0f} · retrieve {timings.retrieve_ms:.0f} · "
            f"generate {timings.generate_ms:.0f} · eval {timings.evaluate_ms:.0f})"
        )
        print(
            f"mean score {metrics.mean_score:.2f} · precision@k {metrics.precision_at_k:.2f} · "
            f"faithfulness {metrics.faithfulness:.2f}"
            + ("  ⚠ possible hallucination" if metrics.hallucination_risk else "")
        )
        return 0

    if args.command == "benchmark":
        cases = load_suite(args.suite)
        result = BenchmarkRunner(engine).run(
            cases,
            label=args.label,
            generate=not args.retrieval_only,
            progress=_progress,
        )
        print(json.dumps(result.summary, indent=2))
        return 0

    return 1


def _print_ingestion(result) -> None:
    print(result.summary())
    for name, reason in result.skipped:
        print(f"  skipped {name}: {reason}")
    for name, error in result.failed:
        print(f"  failed  {name}: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
