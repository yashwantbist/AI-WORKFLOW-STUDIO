"""Deterministic retrieval metrics for RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def _validate(k: int, expected_ids: Iterable[str]) -> tuple[str, ...]:
    if k < 1:
        raise ValueError("k must be at least 1")
    expected = tuple(dict.fromkeys(str(x).strip() for x in expected_ids))
    if not expected or any(not x for x in expected):
        raise ValueError("expected_ids cannot be empty")
    return expected


def hit_at_k(retrieved_ids: Sequence[str], expected_ids: Iterable[str], k: int) -> float:
    expected = set(_validate(k, expected_ids))
    return 1.0 if expected & set(retrieved_ids[:k]) else 0.0


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: Iterable[str], k: int) -> float:
    expected = set(_validate(k, expected_ids))
    return len(expected & set(retrieved_ids[:k])) / len(expected)


def precision_at_k(retrieved_ids: Sequence[str], expected_ids: Iterable[str], k: int) -> float:
    expected = set(_validate(k, expected_ids))
    retrieved = tuple(retrieved_ids[:k])
    if not retrieved:
        return 0.0
    return sum(chunk_id in expected for chunk_id in retrieved) / len(retrieved)


def reciprocal_rank(retrieved_ids: Sequence[str], expected_ids: Iterable[str], k: int) -> float:
    expected = set(_validate(k, expected_ids))
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float

    @property
    def retrieval_passed(self) -> bool:
        return self.hit_at_k == 1.0

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "hit_at_k": self.hit_at_k,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "retrieval_passed": self.retrieval_passed,
        }


def evaluate_retrieval(retrieved_ids, expected_ids, *, k: int) -> RetrievalMetrics:
    return RetrievalMetrics(
        k=k,
        hit_at_k=hit_at_k(retrieved_ids, expected_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, expected_ids, k),
        precision_at_k=precision_at_k(retrieved_ids, expected_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_ids, expected_ids, k),
    )
