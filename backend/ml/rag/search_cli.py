"""CLI for building, inspecting, searching, and benchmarking FAISS retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .benchmark_faiss import run_benchmark
from .faiss_store import FaissDependencyError
from .index_builder import MANIFEST_FILENAME, build_faiss_index
from .retriever import FaissRetriever


DEFAULT_INDEX_DIRECTORY = Path("backend/ml/rag/artifacts/faiss")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and query the AI Workflow Studio FAISS index."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build and persist the sample index.")
    build.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    build.add_argument("--index-type", choices=("flat", "hnsw"), default="flat")
    build.add_argument("--chunk-size", type=int, default=80)
    build.add_argument("--overlap", type=int, default=15)
    build.add_argument("--hnsw-m", type=int, default=32)
    build.add_argument("--ef-construction", type=int, default=80)
    build.add_argument("--ef-search", type=int, default=64)

    search = subparsers.add_parser("search", help="Search indexed chunks.")
    search.add_argument("query")
    search.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--document-id")
    search.add_argument("--section")
    search.add_argument("--page", type=int)

    inspect = subparsers.add_parser("inspect", help="Display persisted index metadata.")
    inspect.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Compare NumPy brute-force search with FAISS flat search.",
    )
    benchmark.add_argument("--vectors", type=int, default=10_000)
    benchmark.add_argument("--dimension", type=int, default=128)
    benchmark.add_argument("--queries", type=int, default=100)
    benchmark.add_argument("--top-k", type=int, default=5)

    return parser


def _format_results(results) -> str:
    if not results:
        return "No matching chunks found."
    lines: list[str] = []
    for result in results:
        chunk = result.chunk
        lines.extend(
            [
                f"{result.rank}. {chunk.chunk_id}",
                f"   Score: {result.score:.4f}",
                (
                    f"   Source: {chunk.document_title} | "
                    f"Pages: {chunk.page_start}-{chunk.page_end}"
                ),
                f"   Sections: {', '.join(chunk.sections) or 'Unknown'}",
                f"   Text: {chunk.text}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)

    try:
        if options.command == "build":
            built = build_faiss_index(
                options.index_dir,
                chunk_size_words=options.chunk_size,
                overlap_words=options.overlap,
                index_type=options.index_type,
                hnsw_m=options.hnsw_m,
                hnsw_ef_construction=options.ef_construction,
                hnsw_ef_search=options.ef_search,
            )
            print(
                f"Built {built.index_type} FAISS index with "
                f"{built.chunk_count} chunks and {built.dimension} dimensions."
            )
            print(f"Index: {built.index_path}")
            print(f"Metadata: {built.metadata_path}")
            return 0

        if options.command == "search":
            filters = {
                key: value
                for key, value in {
                    "document_id": options.document_id,
                    "section": options.section,
                    "page": options.page,
                }.items()
                if value is not None
            }
            retriever = FaissRetriever.load(options.index_dir)
            results = retriever.search(
                options.query,
                top_k=options.top_k,
                filters=filters,
            )
            print(_format_results(results))
            return 0

        if options.command == "inspect":
            path = options.index_dir / MANIFEST_FILENAME
            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
            return 0

        result = run_benchmark(
            vector_count=options.vectors,
            dimension=options.dimension,
            query_count=options.queries,
            top_k=options.top_k,
        )
        print(f"Vectors: {result.vector_count}")
        print(f"Dimensions: {result.dimension}")
        print(f"Queries: {result.query_count}")
        print(f"NumPy search: {result.numpy_seconds:.6f} seconds")
        print(f"FAISS build: {result.faiss_build_seconds:.6f} seconds")
        print(f"FAISS search: {result.faiss_search_seconds:.6f} seconds")
        print(f"Search speedup: {result.search_speedup:.2f}x")
        print(f"Top-1 agreement: {result.top1_agreement:.2%}")
        return 0
    except (FaissDependencyError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
