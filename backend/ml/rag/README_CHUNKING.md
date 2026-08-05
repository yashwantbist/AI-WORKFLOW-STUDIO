# Day 16: Configurable Document Chunking Pipeline

This module prepares long documents for Retrieval-Augmented Generation by splitting them into smaller retrieval units before embedding.

```mermaid
flowchart LR
    A[Long document] --> B[Parse pages and headings]
    B --> C[Choose chunking strategy]
    C --> D[Create overlapping chunks]
    D --> E[Attach metadata]
    E --> F[Generate one embedding per chunk]
    F --> G[Store in vector database]
```

## Why chunking improves RAG

One vector for a long document mixes unrelated topics. Chunking lets retrieval return only the relevant section, improves citation precision, and uses less LLM context.

## Strategies

| Strategy | How it splits | Strength | Risk |
|---|---|---|---|
| Fixed-size | Every N whitespace words | Fast and predictable | Can cut a sentence |
| Sentence-based | Latest sentence ending within budget | Readable chunks | Uneven sizes |
| Recursive | Section, paragraph, sentence, then words | Preserves structure | More logic |

The recursive priority is:

```text
section → paragraph → sentence → exact word limit
```

## Overlap

If chunk size is 8 and overlap is 2:

```text
A B C D E F G H
            G H I J K L M N
```

Overlap preserves context at boundaries, but too much overlap increases embedding cost, storage, and duplicate search results. The code enforces:

```text
0 <= overlap_words < chunk_size_words
```

## Metadata

Each chunk stores a stable ID, source, strategy, page range, section names, inclusive start word, exclusive end word, word count, and actual overlap.

```json
{
  "chunk_id": "transformer-rag-guide-recursive-003",
  "document_id": "transformer-rag-guide",
  "source": "sample://transformer-rag-guide",
  "page_start": 1,
  "page_end": 2,
  "sections": ["Embeddings", "Retrieval-Augmented Generation"],
  "word_count": 80,
  "overlap_words": 15
}
```

## Run

```powershell
python -m backend.ml.rag.chunking_demo --strategy recursive --chunk-size 80 --overlap 15
```

Compare all strategies:

```powershell
python -m backend.ml.rag.chunking_demo --strategy all --chunk-size 80 --overlap 15
```

Generate an HTML boundary visualization:

```powershell
python -m backend.ml.rag.chunking_demo --strategy recursive --chunk-size 80 --overlap 15 --html backend/ml/rag/artifacts/chunks.html
Start-Process backend/ml/rag/artifacts/chunks.html
```

Highlighted words are overlap copied from the previous chunk.

## Test

```powershell
python -m pytest backend/tests/test_document_chunking.py -v
```

Tests cover maximum sizes, exact overlap, no-overlap behavior, sentence boundaries, recursive section metadata, page ranges, stable IDs, invalid settings, empty documents, CLI factories, and visualizations.

## Before and after retrieval

**Before chunking:** one embedding contains attention, embeddings, RAG, overlap, and metadata. A question about overlap competes with all other topics.

**After chunking:** retrieval can return only the page-2 `Chunk Size and Overlap` section, producing more focused context and a better citation.

## Chunk-size trade-offs

| Size | Benefit | Risk |
|---|---|---|
| Very small | Precise matching | Missing explanation |
| Medium | Precision/context balance | Must be evaluated |
| Very large | More context per result | Mixed topics |

Production evaluation should measure recall@k, precision@k, reciprocal rank, citation accuracy, duplicate-result rate, latency, and embedding cost.

## Design decisions

- Whitespace word counts keep this lab dependency-free and understandable.
- Production code should usually count the embedding model's real tokens.
- Frozen dataclasses protect citation metadata from accidental mutation.
- The sample article is separate from Day 15's `sample_documents.py`, so semantic-search code is not overwritten.
- HTML text is escaped before display.

## Security

The module requires no API key. Confirm `.gitignore` excludes `.env`, secrets, local vector indexes, uploads, and user data.

## Next improvements

Token-aware chunking, PDF page parsing, character offsets, table-aware splitting, permissions metadata, and retrieval evaluation across chunk sizes.
