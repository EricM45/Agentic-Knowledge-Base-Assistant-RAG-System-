"""Terminal chat over the indexed documents.

Examples:
    python cli.py
    python cli.py --question "What is the standard warranty on the Helios Arc inverter?"
    python cli.py --docs ./sample_documents --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragbot.config import Settings  # noqa: E402
from ragbot.pipeline import RagPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about a folder of documents.")
    parser.add_argument("--docs", type=Path, help="File or directory to index. Defaults to the sample knowledge base.")
    parser.add_argument("--question", help="Ask one question and exit.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the index even if a saved copy exists.")
    parser.add_argument("--verbose", action="store_true", help="Print retrieval scores.")
    parser.add_argument("--backend", choices=["auto", "openai", "ollama", "local-nlp"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)

    settings = Settings.from_env()
    if args.backend:
        from dataclasses import replace

        settings = replace(settings, backend=args.backend)
    docs = args.docs or settings.sample_dir
    if not docs.exists():
        print(f"No documents at {docs}")
        return 1

    pipeline = RagPipeline(settings)
    print("Loading the embedding model and index...")
    loaded = False if args.reindex else pipeline.load_cached([docs])
    if loaded:
        print(f"Loaded a saved index ({pipeline.stats.chunks} passages).")
        pipeline.ensure_pinecone()
    else:
        stats = pipeline.index_paths([docs])
        print(f"Indexed {stats.files} files into {stats.chunks} passages.")
        for note in stats.skipped:
            print(f"Skipped {note}")
    pipeline.warm()
    print(f"Answer engine: {pipeline.backend_label()}")
    print(f"Vector database: {pipeline.vector_store_label()}")
    if pipeline.vector_store_error:
        print(pipeline.vector_store_error)
    print("Ask a question, or type quit.")

    if args.question:
        answer = pipeline.ask(args.question)
        _print_answer(answer, args.verbose)
        return 0

    history: list[dict] = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"quit", "exit", ":quit"}:
            return 0
        if not question:
            continue
        answer = pipeline.ask(question, history)
        _print_answer(answer, args.verbose)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer.text})


def _print_answer(answer, verbose: bool) -> None:
    print(f"\n{answer.text}")
    if answer.warning:
        print(f"\nNote: {answer.warning}")
    if verbose and answer.standalone_question:
        print(f"\nSearch query: {answer.standalone_question}")
    print(f"\n[{answer.backend}]")
    if not answer.sources:
        return
    print("Nearest passages:" if answer.refused else "Sources:")
    for source in answer.sources:
        label = source.source
        if source.section:
            label += f" · {source.section}"
        if source.page:
            label += f" · p. {source.page}"
        extra = ""
        if verbose:
            extra = f"  dense={source.dense_score:.3f} bm25={source.sparse_score:.2f} fused={source.fused_score:.4f}"
            if source.rerank_score is not None:
                extra += f" rerank={source.rerank_score:.3f}"
        print(f"  [{source.number}] {label}{extra}")


if __name__ == "__main__":
    raise SystemExit(main())
