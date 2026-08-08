"""High-level semantic retriever combining embeddings, FAISS, and metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .embeddings import SemanticTfidfEmbedder
from .faiss_store import FaissVectorStore
from .index_builder import INDEX_FILENAME, MANIFEST_FILENAME, METADATA_FILENAME
from .metadata_store import IndexedChunk, MetadataStore


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    vector_id: int
    chunk: IndexedChunk


class FaissRetriever:
    """Embed queries, search FAISS, and return traceable chunk metadata."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        metadata_store: MetadataStore,
        embedder: SemanticTfidfEmbedder,
    ) -> None:
        if vector_store.ntotal != len(metadata_store):
            raise ValueError("FAISS and metadata record counts do not match")
        if vector_store.dimension != embedder.dimensions:
            raise ValueError("FAISS and embedder dimensions do not match")
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedder = embedder

    @property
    def chunk_count(self) -> int:
        return self._vector_store.ntotal

    @classmethod
    def load(
        cls,
        index_directory: str | Path,
        *,
        faiss_module: Any | None = None,
    ) -> "FaissRetriever":
        directory = Path(index_directory)
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        metadata_store = MetadataStore.load(directory / METADATA_FILENAME)
        embedder = SemanticTfidfEmbedder().fit(
            [record.text for record in metadata_store.records]
        )
        if embedder.dimensions != int(manifest["dimension"]):
            raise ValueError("reconstructed embedder dimension does not match manifest")

        vector_store = FaissVectorStore.load(
            directory / INDEX_FILENAME,
            faiss_module=faiss_module,
            hnsw_ef_search=int(manifest.get("hnsw", {}).get("ef_search", 64)),
        )
        return cls(vector_store, metadata_store, embedder)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = np.asarray(self._embedder.embed(query), dtype=np.float32)
        if np.linalg.norm(query_vector) == 0.0:
            return ()

        allowed_ids = set(self._metadata_store.matching_ids(filters))
        if not allowed_ids:
            return ()

        candidate_k = self.chunk_count if filters else min(top_k, self.chunk_count)
        neighbors = self._vector_store.search(
            query_vector,
            top_k=candidate_k,
        )[0]

        results: list[RetrievedChunk] = []
        for neighbor in neighbors:
            if neighbor.vector_id not in allowed_ids:
                continue
            results.append(
                RetrievedChunk(
                    rank=len(results) + 1,
                    score=neighbor.score,
                    vector_id=neighbor.vector_id,
                    chunk=self._metadata_store.get(neighbor.vector_id),
                )
            )
            if len(results) == top_k:
                break
        return tuple(results)
