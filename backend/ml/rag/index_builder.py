"""Build and persist a FAISS index from the Day 16 chunking pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import SemanticTfidfEmbedder
from .faiss_store import FaissVectorStore
from .metadata_store import IndexedChunk, MetadataStore


INDEX_FILENAME = "chunks.faiss"
METADATA_FILENAME = "chunks.metadata.json"
MANIFEST_FILENAME = "chunks.manifest.json"


@dataclass(frozen=True)
class BuiltIndex:
    index_path: Path
    metadata_path: Path
    manifest_path: Path
    chunk_count: int
    dimension: int
    index_type: str


def _load_chunking_components() -> tuple[Any, Any]:
    """Support both root-level and nested chunking package layouts."""

    try:
        from .chunking_sample_document import SAMPLE_CHUNKING_DOCUMENT
        from .recursive_chunker import RecursiveChunker
    except ModuleNotFoundError:
        from .chunking.chunking_sample_document import (  # type: ignore[import-not-found]
            SAMPLE_CHUNKING_DOCUMENT,
        )
        from .chunking.recursive_chunker import (  # type: ignore[import-not-found]
            RecursiveChunker,
        )
    return SAMPLE_CHUNKING_DOCUMENT, RecursiveChunker


def build_sample_records(
    *,
    chunk_size_words: int = 80,
    overlap_words: int = 15,
) -> tuple[IndexedChunk, ...]:
    document, recursive_chunker = _load_chunking_components()
    chunks = recursive_chunker(
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    ).chunk(document)
    return tuple(IndexedChunk.from_document_chunk(chunk) for chunk in chunks)


def build_faiss_index(
    output_directory: str | Path,
    *,
    records: tuple[IndexedChunk, ...] | None = None,
    chunk_size_words: int = 80,
    overlap_words: int = 15,
    index_type: str = "flat",
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 80,
    hnsw_ef_search: int = 64,
    faiss_module: Any | None = None,
) -> BuiltIndex:
    """Embed chunks, build FAISS, and save index plus metadata."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    chunk_records = records or build_sample_records(
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )
    if not chunk_records:
        raise ValueError("at least one chunk is required")

    texts = [record.text for record in chunk_records]
    embedder = SemanticTfidfEmbedder().fit(texts)
    vectors = np.asarray(embedder.embed_many(texts), dtype=np.float32)

    vector_store = FaissVectorStore(
        embedder.dimensions,
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        faiss_module=faiss_module,
    )
    vector_ids = vector_store.add(vectors)

    metadata_store = MetadataStore(chunk_records)
    if vector_ids != tuple(range(len(metadata_store))):
        raise RuntimeError("FAISS vector IDs and metadata IDs are not aligned")

    index_path = vector_store.save(output / INDEX_FILENAME)
    metadata_path = metadata_store.save(output / METADATA_FILENAME)
    manifest_path = output / MANIFEST_FILENAME
    manifest = {
        "schema_version": 1,
        "index_type": index_type,
        "dimension": embedder.dimensions,
        "chunk_count": len(chunk_records),
        "embedding_model": "SemanticTfidfEmbedder",
        "chunk_size_words": chunk_size_words,
        "overlap_words": overlap_words,
        "hnsw": {
            "m": hnsw_m,
            "ef_construction": hnsw_ef_construction,
            "ef_search": hnsw_ef_search,
        },
        "files": {
            "index": INDEX_FILENAME,
            "metadata": METADATA_FILENAME,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return BuiltIndex(
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        chunk_count=len(chunk_records),
        dimension=embedder.dimensions,
        index_type=index_type,
    )
