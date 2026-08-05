"""Command-line interface for the Day 15 semantic-search prototype."""

import argparse
from collections.abc import Sequence

try:
    from .embeddings import SemanticTfidfEmbedder
    from .sample_documents import SAMPLE_DOCUMENTS
    from .vector_store import InMemoryVectorStore, SearchResult
except ImportError:  # Supports: python backend/ml/rag/semantic_search.py
    from embeddings import SemanticTfidfEmbedder
    from sample_documents import SAMPLE_DOCUMENTS
    from vector_store import InMemoryVectorStore, SearchResult


def build_default_store() -> InMemoryVectorStore:
    """Create a ready-to-search store containing the sample knowledge base."""
    store = InMemoryVectorStore(SemanticTfidfEmbedder())
    store.add_documents(SAMPLE_DOCUMENTS)
    return store


def format_results(
    query: str,
    results: Sequence[SearchResult],
) -> str:
    """Create readable ranked output for the terminal."""
    lines = [f'Query: "{query}"', ""]

    for rank, result in enumerate(results, start=1):
        lines.extend(
            [
                f"{rank}. {result.document.title}",
                f"   Score: {result.score:.4f}",
                f"   ID: {result.document.document_id}",
                f"   {result.document.text}",
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def create_argument_parser() -> argparse.ArgumentParser:
    """Define and document all command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search a small knowledge base using local text embeddings.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural-language question or search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked documents to return (default: 5).",
    )
    parser.add_argument(
        "--show-vector",
        action="store_true",
        help="Print non-zero query-vector dimensions for learning.",
    )
    parser.add_argument(
        "--list-documents",
        action="store_true",
        help="List the sample knowledge-base documents and exit.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = create_argument_parser()
    options = parser.parse_args(arguments)

    if options.list_documents:
        for document in SAMPLE_DOCUMENTS:
            print(f"{document.document_id}: {document.title}")
        return 0

    if not options.query:
        parser.error("query is required unless --list-documents is used")

    store = build_default_store()
    results = store.search(options.query, top_k=options.top_k)

    print(
        f"Knowledge base: {store.size} documents | "
        f"Embedding dimensions: {store.dimensions}\n"
    )
    print(format_results(options.query, results))

    if options.show_vector:
        query_vector = store.embed_query(options.query)
        non_zero_values = [
            (index, value)
            for index, value in enumerate(query_vector)
            if value != 0.0
        ]
        print("\nNon-zero query-vector dimensions:")
        if non_zero_values:
            for index, value in non_zero_values:
                print(f"  dimension {index}: {value:.4f}")
        else:
            print("  none: the query contains no features known to the store")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
