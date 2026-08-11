"""Inference orchestration with optional labelled RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Mapping

from .evaluation.groundedness import (
    ClaimInput,
    GroundednessEvaluation,
    evaluate_groundedness,
)
from .evaluation.retrieval_metrics import (
    RetrievalMetrics,
    evaluate_retrieval,
)
from .rag_pipeline import RAGPipeline, Retriever
from .schemas import RAGAnswer


@dataclass(frozen=True)
class RetrievedItemTelemetry:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_title: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
        }


@dataclass(frozen=True)
class RetrievalTelemetry:
    requested_k: int
    retrieved_count: int
    used_context_count: int
    retrieval_latency_ms: float
    items: tuple[RetrievedItemTelemetry, ...]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_k": self.requested_k,
            "retrieved_count": self.retrieved_count,
            "used_context_count": self.used_context_count,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "chunk_ids": list(self.chunk_ids),
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class EvaluationMetadata:
    retrieval: RetrievalMetrics | None = None
    groundedness: GroundednessEvaluation | None = None

    @property
    def available(self) -> bool:
        return self.retrieval is not None or self.groundedness is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval": (
                self.retrieval.to_dict()
                if self.retrieval is not None
                else None
            ),
            "groundedness": (
                self.groundedness.to_dict()
                if self.groundedness is not None
                else None
            ),
        }


@dataclass(frozen=True)
class EvaluatedRAGResult:
    answer: RAGAnswer
    retrieval: RetrievalTelemetry
    evaluation: EvaluationMetadata

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "evaluation": (
                self.evaluation.to_dict()
                if self.evaluation.available
                else None
            ),
        }


class EvaluatedRAGPipeline:
    """Run one retrieval event, generation, telemetry, and optional evaluation."""

    def __init__(
        self,
        retriever: Retriever,
        rag_pipeline: RAGPipeline,
    ) -> None:
        self._retriever = retriever
        self._rag_pipeline = rag_pipeline

    @staticmethod
    def _validate_query(query: str) -> str:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query cannot be empty")
        return clean_query

    @staticmethod
    def _telemetry_item(result: Any) -> RetrievedItemTelemetry:
        chunk = result.chunk
        return RetrievedItemTelemetry(
            rank=int(result.rank),
            score=float(result.score),
            chunk_id=str(chunk.chunk_id),
            document_id=str(chunk.document_id),
            document_title=str(chunk.document_title),
        )

    def run(
        self,
        query: str,
        *,
        k: int = 5,
        relevant_ids: Iterable[str] | None = None,
        claim_labels: Iterable[ClaimInput] | None = None,
        filters: Mapping[str, Any] | None = None,
        minimum_relevance_score: float | None = None,
    ) -> EvaluatedRAGResult:
        clean_query = self._validate_query(query)
        if k < 1:
            raise ValueError("k must be at least 1")

        retrieval_started = time.perf_counter()
        retrieved = tuple(
            self._retriever.search(
                clean_query,
                top_k=k,
                filters=filters,
            )
        )
        retrieval_latency_ms = (
            time.perf_counter() - retrieval_started
        ) * 1000.0

        answer = self._rag_pipeline.answer_from_retrieved(
            clean_query,
            retrieved,
            minimum_relevance_score=minimum_relevance_score,
        )

        items = tuple(
            self._telemetry_item(result)
            for result in retrieved
        )

        telemetry = RetrievalTelemetry(
            requested_k=k,
            retrieved_count=len(retrieved),
            used_context_count=answer.used_context_count,
            retrieval_latency_ms=retrieval_latency_ms,
            items=items,
        )

        retrieval_evaluation = (
            evaluate_retrieval(
                telemetry.chunk_ids,
                relevant_ids,
                k=k,
            )
            if relevant_ids is not None
            else None
        )

        groundedness_evaluation = (
            evaluate_groundedness(claim_labels)
            if claim_labels is not None
            else None
        )

        return EvaluatedRAGResult(
            answer=answer,
            retrieval=telemetry,
            evaluation=EvaluationMetadata(
                retrieval=retrieval_evaluation,
                groundedness=groundedness_evaluation,
            ),
        )
