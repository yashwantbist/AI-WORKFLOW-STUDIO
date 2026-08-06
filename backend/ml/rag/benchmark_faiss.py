"""Benchmark exact NumPy list search against a FAISS flat index."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .faiss_store import FaissVectorStore


@dataclass(frozen=True)
class BenchmarkResult:
    vector_count: int
    dimension: int
    query_count: int
    top_k: int
    numpy_seconds: float
    faiss_build_seconds: float
    faiss_search_seconds: float
    top1_agreement: float

    @property
    def search_speedup(self) -> float:
        if self.faiss_search_seconds == 0.0:
            return float("inf")
        return self.numpy_seconds / self.faiss_search_seconds


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def numpy_top_k(
    database: np.ndarray,
    queries: np.ndarray,
    top_k: int,
) -> np.ndarray:
    scores = queries @ database.T
    candidate_ids = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    candidate_scores = np.take_along_axis(scores, candidate_ids, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidate_ids, order, axis=1)


def run_benchmark(
    *,
    vector_count: int = 10_000,
    dimension: int = 128,
    query_count: int = 100,
    top_k: int = 5,
    random_seed: int = 42,
    faiss_module: Any | None = None,
) -> BenchmarkResult:
    if min(vector_count, dimension, query_count, top_k) < 1:
        raise ValueError("benchmark sizes must be positive")
    if top_k > vector_count:
        raise ValueError("top_k cannot exceed vector_count")

    rng = np.random.default_rng(random_seed)
    database = _normalize(
        rng.normal(size=(vector_count, dimension)).astype(np.float32)
    )
    queries = _normalize(
        rng.normal(size=(query_count, dimension)).astype(np.float32)
    )

    start = perf_counter()
    numpy_ids = numpy_top_k(database, queries, top_k)
    numpy_seconds = perf_counter() - start

    start = perf_counter()
    store = FaissVectorStore(
        dimension,
        index_type="flat",
        faiss_module=faiss_module,
    )
    store.add(database)
    faiss_build_seconds = perf_counter() - start

    start = perf_counter()
    faiss_rows = store.search(queries, top_k=top_k)
    faiss_search_seconds = perf_counter() - start
    faiss_ids = np.asarray(
        [[neighbor.vector_id for neighbor in row] for row in faiss_rows],
        dtype=np.int64,
    )

    top1_agreement = float(np.mean(numpy_ids[:, 0] == faiss_ids[:, 0]))
    return BenchmarkResult(
        vector_count=vector_count,
        dimension=dimension,
        query_count=query_count,
        top_k=top_k,
        numpy_seconds=numpy_seconds,
        faiss_build_seconds=faiss_build_seconds,
        faiss_search_seconds=faiss_search_seconds,
        top1_agreement=top1_agreement,
    )
