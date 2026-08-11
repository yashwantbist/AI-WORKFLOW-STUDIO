"""Deterministic retrieval metrics for RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def _clean_ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError(f"{name} cannot contain empty IDs")
    return cleaned


def _top_k(retrieved_ids: Sequence[str], k: int) -> tuple[str, ...]:
    _validate_k(k)
    return _clean_ids(retrieved_ids, name="retrieved_ids")[:k]


def _relevant_set(relevant_ids: Iterable[str]) -> set[str]:
    return set(_clean_ids(relevant_ids, name="relevant_ids"))


def _unique_relevant_hits(
    top_k: Sequence[str],
    relevant_ids: set[str],
) -> set[str]:
    return set(top_k) & relevant_ids


def hit_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    top_k = _top_k(retrieved_ids, k)
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    return 1.0 if _unique_relevant_hits(top_k, relevant) else 0.0


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    top_k = _top_k(retrieved_ids, k)
    if not top_k:
        return 0.0

    relevant = _relevant_set(relevant_ids)
    relevant_hits = _unique_relevant_hits(top_k, relevant)
    return len(relevant_hits) / len(top_k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    top_k = _top_k(retrieved_ids, k)
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0

    relevant_hits = _unique_relevant_hits(top_k, relevant)
    return len(relevant_hits) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    top_k = _top_k(retrieved_ids, k)
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0

    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    relevant_retrieved: int
    retrieved_count: int
    total_relevant: int
    duplicate_count: int

    @property
    def retrieval_passed(self) -> bool:
        return self.hit_at_k == 1.0

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "k": self.k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "hit_at_k": self.hit_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "relevant_retrieved": self.relevant_retrieved,
            "retrieved_count": self.retrieved_count,
            "total_relevant": self.total_relevant,
            "duplicate_count": self.duplicate_count,
            "retrieval_passed": self.retrieval_passed,
        }


RetrievalEvaluation = RetrievalMetrics


def evaluate_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k: int = 5,
) -> RetrievalMetrics:
    top_k = _top_k(retrieved_ids, k)
    relevant = _relevant_set(relevant_ids)
    hits = _unique_relevant_hits(top_k, relevant)

    return RetrievalMetrics(
        k=k,
        precision_at_k=precision_at_k(top_k, relevant, k),
        recall_at_k=recall_at_k(top_k, relevant, k),
        hit_at_k=hit_at_k(top_k, relevant, k),
        reciprocal_rank=reciprocal_rank(top_k, relevant, k),
        relevant_retrieved=len(hits),
        retrieved_count=len(top_k),
        total_relevant=len(relevant),
        duplicate_count=len(top_k) - len(set(top_k)),
    )
