"""FAISS-backed vector index with cosine-similarity search and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


class FaissDependencyError(RuntimeError):
    """Raised when FAISS is required but not installed."""


def load_faiss() -> ModuleType:
    """Import FAISS lazily and return an actionable error when unavailable."""

    try:
        import faiss  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise FaissDependencyError(
            "FAISS is not installed. Run: "
            "python -m pip install faiss-cpu==1.15.0 numpy"
        ) from error
    return faiss


def as_float32_matrix(
    vectors: Any,
    *,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Convert input into a contiguous two-dimensional float32 matrix."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a one- or two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors cannot be empty")
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise ValueError(
            f"expected vectors with dimension {expected_dimension}, "
            f"received {matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    return np.ascontiguousarray(matrix)


def normalize_for_cosine(
    vectors: Any,
    *,
    faiss_module: ModuleType | Any | None = None,
    reject_zero_vectors: bool = True,
) -> np.ndarray:
    """L2-normalize vectors so inner product is cosine similarity."""

    module = faiss_module or load_faiss()
    matrix = as_float32_matrix(vectors).copy()
    norms = np.linalg.norm(matrix, axis=1)
    if reject_zero_vectors and np.any(norms == 0.0):
        raise ValueError("zero vectors cannot be used for cosine similarity")
    module.normalize_L2(matrix)
    return matrix


@dataclass(frozen=True)
class FaissNeighbor:
    """One nearest-neighbor match returned by FAISS."""

    vector_id: int
    score: float


class FaissVectorStore:
    """Own a FAISS index and expose safe add, search, save, and load methods."""

    SUPPORTED_INDEX_TYPES = frozenset({"flat", "hnsw"})

    def __init__(
        self,
        dimension: int,
        *,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 80,
        hnsw_ef_search: int = 64,
        faiss_module: ModuleType | Any | None = None,
        index: Any | None = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        if index_type not in self.SUPPORTED_INDEX_TYPES and index_type != "loaded":
            raise ValueError(
                f"index_type must be one of {sorted(self.SUPPORTED_INDEX_TYPES)}"
            )
        if hnsw_m < 2:
            raise ValueError("hnsw_m must be at least 2")

        self._faiss = faiss_module or load_faiss()
        self._dimension = dimension
        self._index_type = index_type
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._index = index or self._create_index()

        index_dimension = int(getattr(self._index, "d", dimension))
        if index_dimension != dimension:
            raise ValueError(
                f"loaded index dimension {index_dimension} does not match {dimension}"
            )
        self._configure_hnsw_if_present()

    def _create_index(self) -> Any:
        if self._index_type == "flat":
            return self._faiss.IndexFlatIP(self._dimension)

        metric = getattr(self._faiss, "METRIC_INNER_PRODUCT", 0)
        return self._faiss.IndexHNSWFlat(
            self._dimension,
            self._hnsw_m,
            metric,
        )

    def _configure_hnsw_if_present(self) -> None:
        hnsw = getattr(self._index, "hnsw", None)
        if hnsw is None:
            return
        hnsw.efConstruction = self._hnsw_ef_construction
        hnsw.efSearch = self._hnsw_ef_search

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def index_type(self) -> str:
        return self._index_type

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: Any) -> tuple[int, ...]:
        """Normalize and add vectors, returning their sequential FAISS IDs."""

        matrix = as_float32_matrix(
            vectors,
            expected_dimension=self._dimension,
        )
        normalized = normalize_for_cosine(
            matrix,
            faiss_module=self._faiss,
        )
        first_id = self.ntotal
        self._index.add(normalized)
        return tuple(range(first_id, first_id + normalized.shape[0]))

    def search(
        self,
        query_vectors: Any,
        *,
        top_k: int = 5,
    ) -> tuple[tuple[FaissNeighbor, ...], ...]:
        """Return ranked cosine-similarity neighbors for each query vector."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.ntotal == 0:
            raise RuntimeError("cannot search an empty FAISS index")

        matrix = as_float32_matrix(
            query_vectors,
            expected_dimension=self._dimension,
        )
        normalized = normalize_for_cosine(
            matrix,
            faiss_module=self._faiss,
        )
        actual_k = min(top_k, self.ntotal)
        scores, ids = self._index.search(normalized, actual_k)

        rows: list[tuple[FaissNeighbor, ...]] = []
        for score_row, id_row in zip(scores, ids):
            rows.append(
                tuple(
                    FaissNeighbor(
                        vector_id=int(vector_id),
                        score=float(score),
                    )
                    for score, vector_id in zip(score_row, id_row)
                    if int(vector_id) >= 0
                )
            )
        return tuple(rows)

    def save(self, path: str | Path) -> Path:
        """Persist the native FAISS index to disk."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(output_path))
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        faiss_module: ModuleType | Any | None = None,
        hnsw_ef_search: int = 64,
    ) -> "FaissVectorStore":
        """Reload a native FAISS index from disk."""

        module = faiss_module or load_faiss()
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {input_path}")
        index = module.read_index(str(input_path))
        return cls(
            int(index.d),
            index_type="loaded",
            hnsw_ef_search=hnsw_ef_search,
            faiss_module=module,
            index=index,
        )
