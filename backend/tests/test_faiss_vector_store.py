"""Tests for FAISS persistence, metadata mapping, retrieval, and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pytest

from backend.ml.rag.benchmark_faiss import run_benchmark
from backend.ml.rag.embeddings import SemanticTfidfEmbedder
from backend.ml.rag.faiss_store import (
    FaissVectorStore,
    as_float32_matrix,
    normalize_for_cosine,
)
from backend.ml.rag.index_builder import build_faiss_index
from backend.ml.rag.metadata_store import IndexedChunk, MetadataStore
from backend.ml.rag.retriever import FaissRetriever


class _FakeHnswSettings:
    efConstruction = 0
    efSearch = 0


class _FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self._vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return self._vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.vstack((self._vectors, vectors.copy()))

    def search(self, queries: np.ndarray, top_k: int):
        scores = queries @ self._vectors.T
        ids = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, ids, axis=1)
        return sorted_scores.astype(np.float32), ids.astype(np.int64)


class _FakeIndexHNSWFlat(_FakeIndexFlatIP):
    def __init__(self, dimension: int, m: int, metric: int) -> None:
        super().__init__(dimension)
        self.m = m
        self.metric = metric
        self.hnsw = _FakeHnswSettings()


class FakeFaiss:
    METRIC_INNER_PRODUCT = 0
    IndexFlatIP = _FakeIndexFlatIP
    IndexHNSWFlat = _FakeIndexHNSWFlat

    @staticmethod
    def normalize_L2(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms

    @staticmethod
    def write_index(index, path: str) -> None:
        Path(path).write_bytes(pickle.dumps(index))

    @staticmethod
    def read_index(path: str):
        return pickle.loads(Path(path).read_bytes())


def sample_records() -> tuple[IndexedChunk, ...]:
    return (
        IndexedChunk(
            chunk_id="attention-001",
            text=(
                "Self-attention uses query key and value vectors to connect "
                "tokens and model long-range relationships."
            ),
            document_id="transformers",
            document_title="Transformer Guide",
            source="memory://transformers",
            page_start=1,
            page_end=1,
            sections=("Self-Attention",),
            strategy="recursive",
        ),
        IndexedChunk(
            chunk_id="chunking-001",
            text=(
                "Document chunk overlap preserves context across boundaries "
                "and improves retrieval precision."
            ),
            document_id="rag-guide",
            document_title="RAG Guide",
            source="memory://rag",
            page_start=2,
            page_end=2,
            sections=("Chunk Size and Overlap",),
            strategy="recursive",
        ),
        IndexedChunk(
            chunk_id="metadata-001",
            text=(
                "Chunk metadata stores page numbers section names source paths "
                "and stable identifiers for citations."
            ),
            document_id="rag-guide",
            document_title="RAG Guide",
            source="memory://rag",
            page_start=3,
            page_end=3,
            sections=("Metadata and Citations",),
            strategy="recursive",
        ),
    )


def test_matrix_conversion_is_float32_and_two_dimensional() -> None:
    matrix = as_float32_matrix([1.0, 2.0, 3.0])
    assert matrix.shape == (1, 3)
    assert matrix.dtype == np.float32


def test_normalization_produces_unit_vectors() -> None:
    matrix = normalize_for_cosine(
        [[3.0, 4.0], [1.0, 0.0]],
        faiss_module=FakeFaiss,
    )
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_zero_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        normalize_for_cosine([[0.0, 0.0]], faiss_module=FakeFaiss)


def test_flat_index_returns_most_similar_vector() -> None:
    store = FaissVectorStore(2, faiss_module=FakeFaiss)
    store.add([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    results = store.search([[0.9, 0.1]], top_k=2)[0]
    assert results[0].vector_id == 0
    assert results[0].score > results[1].score


def test_hnsw_settings_are_configured() -> None:
    store = FaissVectorStore(
        3,
        index_type="hnsw",
        hnsw_m=16,
        hnsw_ef_construction=90,
        hnsw_ef_search=70,
        faiss_module=FakeFaiss,
    )
    assert store._index.hnsw.efConstruction == 90
    assert store._index.hnsw.efSearch == 70


def test_index_save_and_reload_preserves_results(tmp_path: Path) -> None:
    path = tmp_path / "index.faiss"
    store = FaissVectorStore(2, faiss_module=FakeFaiss)
    store.add([[1.0, 0.0], [0.0, 1.0]])
    before = store.search([[1.0, 0.0]], top_k=2)
    store.save(path)

    reloaded = FaissVectorStore.load(path, faiss_module=FakeFaiss)
    after = reloaded.search([[1.0, 0.0]], top_k=2)
    assert before == after
    assert reloaded.ntotal == 2


def test_metadata_round_trip_and_filters(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    store = MetadataStore(sample_records())
    store.save(path)
    loaded = MetadataStore.load(path)

    assert loaded.records == store.records
    assert loaded.matching_ids({"document_id": "rag-guide"}) == (1, 2)
    assert loaded.matching_ids({"section": "Citations"}) == (2,)
    assert loaded.matching_ids({"page": 2}) == (1,)


def build_test_retriever() -> FaissRetriever:
    records = sample_records()
    embedder = SemanticTfidfEmbedder().fit([record.text for record in records])
    vectors = np.asarray(
        embedder.embed_many([record.text for record in records]),
        dtype=np.float32,
    )
    vector_store = FaissVectorStore(
        embedder.dimensions,
        faiss_module=FakeFaiss,
    )
    vector_store.add(vectors)
    return FaissRetriever(vector_store, MetadataStore(records), embedder)


def test_retriever_returns_ranked_metadata() -> None:
    retriever = build_test_retriever()
    results = retriever.search("How does query key value attention work?", top_k=2)
    assert results[0].chunk.chunk_id == "attention-001"
    assert results[0].rank == 1
    assert results[0].score >= results[1].score


def test_retriever_applies_metadata_filter() -> None:
    retriever = build_test_retriever()
    results = retriever.search(
        "How is context preserved across chunk boundaries?",
        top_k=3,
        filters={"document_id": "rag-guide", "page": 2},
    )
    assert [result.chunk.chunk_id for result in results] == ["chunking-001"]


def test_builder_and_loader_align_index_with_metadata(tmp_path: Path) -> None:
    built = build_faiss_index(
        tmp_path,
        records=sample_records(),
        faiss_module=FakeFaiss,
    )
    assert built.chunk_count == 3
    retriever = FaissRetriever.load(tmp_path, faiss_module=FakeFaiss)
    results = retriever.search("source page section citations", top_k=1)
    assert results[0].chunk.chunk_id == "metadata-001"


def test_benchmark_exact_search_agrees_with_numpy() -> None:
    result = run_benchmark(
        vector_count=100,
        dimension=16,
        query_count=10,
        top_k=3,
        faiss_module=FakeFaiss,
    )
    assert result.top1_agreement == 1.0
    assert result.numpy_seconds >= 0.0
    assert result.faiss_search_seconds >= 0.0


def test_native_faiss_integration_when_installed(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    store = FaissVectorStore(2, faiss_module=faiss)
    store.add([[1.0, 0.0], [0.0, 1.0]])
    assert store.search([[1.0, 0.0]], top_k=1)[0][0].vector_id == 0

    path = tmp_path / "native.faiss"
    store.save(path)
    loaded = FaissVectorStore.load(path, faiss_module=faiss)
    assert loaded.ntotal == 2