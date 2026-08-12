"""Run labelled retrieval cases and aggregate deterministic RAG metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Protocol

from .offline_dataset import OfflineEvaluationCase, OfflineEvaluationDataset
from .retrieval_metrics import RetrievalMetrics, evaluate_retrieval


class SearchRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        ...


@dataclass(frozen=True)
class RetrievedEvaluationItem:
    rank: int
    score: float
    chunk_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class QueryEvaluationResult:
    query_id: str
    query: str
    relevant_ids: tuple[str, ...]
    retrieved: tuple[RetrievedEvaluationItem, ...]
    metrics: RetrievalMetrics
    matched_relevant_ids: tuple[str, ...]
    missed_relevant_ids: tuple[str, ...]
    retrieval_latency_ms: float

    @property
    def retrieved_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.retrieved)

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "relevant_ids": list(self.relevant_ids),
            "retrieved_ids": list(self.retrieved_ids),
            "retrieved": [item.to_dict() for item in self.retrieved],
            "metrics": self.metrics.to_dict(),
            "matched_relevant_ids": list(self.matched_relevant_ids),
            "missed_relevant_ids": list(self.missed_relevant_ids),
            "retrieval_latency_ms": self.retrieval_latency_ms,
        }


@dataclass(frozen=True)
class EvaluationReport:
    dataset_name: str
    k: int
    total_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    results: tuple[QueryEvaluationResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "k": self.k,
            "total_queries": self.total_queries,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "results": [result.to_dict() for result in self.results],
        }


class OfflineRetrievalEvaluator:
    def __init__(self, retriever: SearchRetriever, *, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self._retriever = retriever
        self._k = k

    def evaluate_case(
        self,
        case: OfflineEvaluationCase,
    ) -> QueryEvaluationResult:
        started = perf_counter()
        raw_results = tuple(
            self._retriever.search(case.query, top_k=self._k)
        )
        latency_ms = (perf_counter() - started) * 1000.0

        retrieved = tuple(
            RetrievedEvaluationItem(
                rank=int(result.rank),
                score=float(result.score),
                chunk_id=str(result.chunk.chunk_id),
            )
            for result in raw_results
        )
        retrieved_ids = tuple(item.chunk_id for item in retrieved)

        metrics = evaluate_retrieval(
            retrieved_ids,
            case.relevant_ids,
            k=self._k,
        )

        retrieved_set = set(retrieved_ids)
        relevant_set = set(case.relevant_ids)

        return QueryEvaluationResult(
            query_id=case.case_id,
            query=case.query,
            relevant_ids=case.relevant_ids,
            retrieved=retrieved,
            metrics=metrics,
            matched_relevant_ids=tuple(
                sorted(retrieved_set & relevant_set)
            ),
            missed_relevant_ids=tuple(
                sorted(relevant_set - retrieved_set)
            ),
            retrieval_latency_ms=latency_ms,
        )

    def evaluate_dataset(
        self,
        dataset: OfflineEvaluationDataset,
    ) -> EvaluationReport:
        results = tuple(
            self.evaluate_case(case)
            for case in dataset.cases
        )
        count = len(results)

        return EvaluationReport(
            dataset_name=dataset.name,
            k=self._k,
            total_queries=count,
            mean_precision_at_k=sum(
                result.metrics.precision_at_k
                for result in results
            ) / count,
            mean_recall_at_k=sum(
                result.metrics.recall_at_k
                for result in results
            ) / count,
            hit_rate_at_k=sum(
                result.metrics.hit_at_k
                for result in results
            ) / count,
            mean_reciprocal_rank=sum(
                result.metrics.reciprocal_rank
                for result in results
            ) / count,
            results=results,
        )
