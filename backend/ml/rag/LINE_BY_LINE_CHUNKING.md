# Day 16 Code Walkthrough

## `metadata.py`

`DocumentPage` and `SourceDocument` model an ordered source. `WordUnit` attaches page and section information to each whitespace token. `analyze_document()` detects Markdown headings, paragraphs, and sentence endings and records each boundary as an exclusive Python slice index. `build_chunk()` slices the words and creates immutable citation metadata.

## `chunker.py`

`validate_chunk_configuration()` prevents negative values and infinite loops. Fixed-size chunking calculates `end = min(start + size, total)`, then advances with `start = end - overlap`. Sentence chunking selects the latest sentence ending that fits inside the same budget.

## `recursive_chunker.py`

The recursive chunker tests section boundaries first, then paragraph boundaries, then sentence boundaries. `minimum_fill_ratio` prevents a tiny early heading from producing a nearly empty chunk. If no useful logical boundary fits, it uses the exact word limit.

## `chunk_visualizer.py`

The terminal formatter shows chunk IDs, lengths, overlap, pages, and sections. The HTML renderer escapes source text and wraps the first `overlap_words` in `<mark>`.

## `chunking_demo.py`

`argparse` validates CLI choices. `create_chunker()` is a factory that maps a strategy name to the correct implementation. The `all` mode compares chunk counts and minimum/maximum lengths. The main guard runs the CLI only when executed, not when imported by tests.
