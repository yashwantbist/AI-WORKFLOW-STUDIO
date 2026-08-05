"""CLI demo for fixed, sentence, and recursive chunking."""
import argparse
from collections.abc import Sequence
from pathlib import Path
from .chunk_visualizer import format_chunks, render_html
from .chunker import FixedSizeChunker, SentenceChunker
from .chunking_sample_document import SAMPLE_CHUNKING_DOCUMENT
from .recursive_chunker import RecursiveChunker

STRATEGIES = ("fixed", "sentence", "recursive")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a two-page article into retrieval chunks.")
    parser.add_argument("--strategy", choices=(*STRATEGIES, "all"), default="recursive")
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--html", type=Path)
    return parser

def create_chunker(strategy: str, chunk_size_words: int, overlap_words: int):
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size_words, overlap_words)
    if strategy == "sentence":
        return SentenceChunker(chunk_size_words, overlap_words)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size_words, overlap_words)
    raise ValueError(f"unsupported strategy: {strategy}")

def run_strategy(strategy: str, chunk_size_words: int, overlap_words: int):
    return create_chunker(strategy, chunk_size_words, overlap_words).chunk(SAMPLE_CHUNKING_DOCUMENT)

def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    if options.strategy == "all":
        results = {name: run_strategy(name, options.chunk_size, options.overlap) for name in STRATEGIES}
        print(f"Document: {SAMPLE_CHUNKING_DOCUMENT.title}")
        print(f"Chunk size: {options.chunk_size} words | Overlap: {options.overlap} words\n")
        for name, chunks in results.items():
            counts = [c.metadata.word_count for c in chunks]
            print(f"{name:10} | chunks: {len(chunks):2d} | min words: {min(counts):3d} | max words: {max(counts):3d}")
        if options.html:
            print(f"\nHTML visualization written to: {render_html(results['recursive'], options.html)}")
        return 0
    chunks = run_strategy(options.strategy, options.chunk_size, options.overlap)
    print(format_chunks(chunks))
    if options.html:
        print(f"\nHTML visualization written to: {render_html(chunks, options.html)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
