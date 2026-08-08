"""CLI demonstration of the complete grounded RAG pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .llm_provider import OpenAIResponsesProvider, RecordingDemoProvider
from .rag_pipeline import RAGPipeline
from .retriever import FaissRetriever


DEFAULT_INDEX_DIRECTORY = Path("backend/ml/rag/artifacts/faiss")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve FAISS chunks, build grounded context, and generate "
            "an answer with a pluggable LLM provider."
        )
    )
    parser.add_argument("question", help="Natural-language question to answer.")
    parser.add_argument(
        "--provider",
        choices=("demo", "openai"),
        default="demo",
        help="Use the offline demo provider or the OpenAI Responses API.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIRECTORY,
        help="Directory containing the Day 17 FAISS index artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum retrieved chunks before relevance filtering.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.10,
        help="Minimum similarity score accepted as grounding evidence.",
    )
    parser.add_argument("--page", type=int, help="Optional page metadata filter.")
    parser.add_argument("--section", help="Optional section metadata filter.")
    parser.add_argument(
        "--document-id",
        help="Optional document_id metadata filter.",
    )
    return parser


def build_filters(options: argparse.Namespace) -> dict[str, object]:
    filters: dict[str, object] = {}
    if options.page is not None:
        filters["page"] = options.page
    if options.section:
        filters["section"] = options.section
    if options.document_id:
        filters["document_id"] = options.document_id
    return filters


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)

    if not options.index_dir.exists():
        raise SystemExit(
            f"FAISS index directory not found: {options.index_dir}\n"
            "Build Day 17 first with:\n"
            "python -m backend.ml.rag.search_cli build"
        )

    retriever = FaissRetriever.load(options.index_dir)

    if options.provider == "openai":
        provider = OpenAIResponsesProvider.from_environment()
    else:
        provider = RecordingDemoProvider()

    pipeline = RAGPipeline(
        retriever,
        provider,
        minimum_relevance_score=options.min_score,
    )
    result = pipeline.answer(
        options.question,
        top_k=options.top_k,
        filters=build_filters(options) or None,
    )

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
