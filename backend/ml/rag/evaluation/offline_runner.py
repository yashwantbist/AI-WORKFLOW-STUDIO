"""Run labelled retrieval cases and aggregate diagnostic RAG metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Protocol

from .diagnostics import (
    RetrievalStatus,
    classify_retrieval,
    irrelevant_retrieved_ids,
)
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
    irrelevant_retrieved_ids: tuple[str, ...]
    status: RetrievalStatus
    retrieval_latency_ms: float

    @property
    def retrieved_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.retrieved)

    @property
    def has_irrelevant_results(self) -> bool:
        return bool(self.irrelevant_retrieved_ids)

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
            "irrelevant_retrieved_ids": list(self.irrelevant_retrieved_ids),
            "has_irrelevant_results": self.has_irrelevant_results,
            "status": self.status.value,
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
    successful_queries: int
    partial_queries: int
    complete_misses: int
    no_result_queries: int
    irrelevant_only_queries: int
    queries_with_irrelevant_results: int
    results: tuple[QueryEvaluationResult, ...]

    @property
    def failures(self) -> tuple[QueryEvaluationResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.status is not RetrievalStatus.SUCCESS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "k": self.k,
            "total_queries": self.total_queries,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "successful_queries": self.successful_queries,
            "partial_queries": self.partial_queries,
            "complete_misses": self.complete_misses,
            "no_result_queries": self.no_result_queries,
            "irrelevant_only_queries": self.irrelevant_only_queries,
            "queries_with_irrelevant_results": (
                self.queries_with_irrelevant_results
            ),
            "failure_count": len(self.failures),
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
            irrelevant_retrieved_ids=irrelevant_retrieved_ids(
                retrieved_ids,
                case.relevant_ids,
            ),
            status=classify_retrieval(
                retrieved_ids,
                case.relevant_ids,
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
        if not results:
            raise ValueError("dataset must contain at least one case")

        count = len(results)

        def count_status(status: RetrievalStatus) -> int:
            return sum(
                result.status is status
                for result in results
            )

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
            successful_queries=count_status(RetrievalStatus.SUCCESS),
            partial_queries=count_status(RetrievalStatus.PARTIAL),
            complete_misses=count_status(RetrievalStatus.COMPLETE_MISS),
            no_result_queries=count_status(RetrievalStatus.NO_RESULTS),
            irrelevant_only_queries=count_status(
                RetrievalStatus.IRRELEVANT_RESULTS
            ),
            queries_with_irrelevant_results=sum(
                result.has_irrelevant_results
                for result in results
            ),
            results=results,
        )
