"""Run a dataset through raw retrieval and the complete RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Protocol

from .answer_metrics import AnswerMetrics, evaluate_answer
from .dataset import EvaluationCase, EvaluationDataset
from .generation_quality import (
    FailureCategory,
    GenerationEvaluation,
    evaluate_generation_quality,
)
from .groundedness import (
    GroundednessEvaluation,
    evaluate_labelled_groundedness,
)
from .retrieval_metrics import RetrievalMetrics, evaluate_retrieval


class SearchRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ):
        ...


class AnswerPipeline(Protocol):
    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        minimum_relevance_score: float | None = None,
    ):
        ...


@dataclass(frozen=True)
class RetrievedTrace:
    rank: int
    score: float
    chunk_id: str
    text: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    question: str
    expected_chunk_ids: tuple[str, ...]
    reference_answer: str
    retrieved: tuple[RetrievedTrace, ...]
    retrieval_metrics: RetrievalMetrics
    generated_answer: str
    used_source_ids: tuple[str, ...]
    answer_metrics: AnswerMetrics
    groundedness: GroundednessEvaluation | None
    generation_evaluation: GenerationEvaluation
    insufficient_context: bool
    retrieval_latency_ms: float
    pipeline_latency_ms: float
    failure_stage: str
    status: str

    @property
    def failure_categories(self) -> tuple[FailureCategory, ...]:
        return self.generation_evaluation.failure_categories

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "expected_chunk_ids": list(
                self.expected_chunk_ids
            ),
            "reference_answer": self.reference_answer,
            "retrieved": [
                item.to_dict()
                for item in self.retrieved
            ],
            "retrieval_metrics": (
                self.retrieval_metrics.to_dict()
            ),
            "generated_answer": self.generated_answer,
            "used_source_ids": list(
                self.used_source_ids
            ),
            "answer_metrics": (
                self.answer_metrics.to_dict()
            ),
            "groundedness": (
                self.groundedness.to_dict()
                if self.groundedness is not None
                else None
            ),
            "generation_evaluation": (
                self.generation_evaluation.to_dict()
            ),
            "failure_categories": [
                category.value
                for category in self.failure_categories
            ],
            "insufficient_context": (
                self.insufficient_context
            ),
            "retrieval_latency_ms": (
                self.retrieval_latency_ms
            ),
            "pipeline_latency_ms": (
                self.pipeline_latency_ms
            ),
            "failure_stage": self.failure_stage,
            "status": self.status,
        }


def _mean_optional(
    values: tuple[float, ...],
) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


@dataclass(frozen=True)
class EvaluationRun:
    dataset_name: str
    top_k: int
    minimum_relevance_score: float
    answer_f1_review_threshold: float
    cases: tuple[CaseEvaluation, ...]

    @property
    def summary(self) -> dict:
        n = len(self.cases)

        if not n:
            return {
                "case_count": 0,
                "retrieval_hit_rate": 0.0,
                "groundedness_case_count": 0,
                "mean_groundedness_score": None,
                "answer_relevance_case_count": 0,
                "mean_answer_relevance": None,
                "correctness_case_count": 0,
                "mean_correctness": None,
                "generation_failure_counts": {},
            }

        failure_counts: dict[str, int] = {}
        generation_failure_counts: dict[str, int] = {}

        for case in self.cases:
            failure_counts[case.failure_stage] = (
                failure_counts.get(
                    case.failure_stage,
                    0,
                )
                + 1
            )

            for category in case.failure_categories:
                generation_failure_counts[category.value] = (
                    generation_failure_counts.get(
                        category.value,
                        0,
                    )
                    + 1
                )

        groundedness_cases = tuple(
            case.groundedness
            for case in self.cases
            if case.groundedness is not None
        )

        groundedness_scores = tuple(
            result.groundedness_score
            for result in groundedness_cases
        )

        relevance_scores = tuple(
            case.generation_evaluation.answer_relevance
            for case in self.cases
            if (
                case.generation_evaluation.answer_relevance
                is not None
            )
        )

        correctness_scores = tuple(
            case.generation_evaluation.correctness
            for case in self.cases
            if (
                case.generation_evaluation.correctness
                is not None
            )
        )

        return {
            "case_count": n,
            "retrieval_hit_rate": sum(
                case.retrieval_metrics.hit_at_k
                for case in self.cases
            )
            / n,
            "mean_recall_at_k": sum(
                case.retrieval_metrics.recall_at_k
                for case in self.cases
            )
            / n,
            "mean_precision_at_k": sum(
                case.retrieval_metrics.precision_at_k
                for case in self.cases
            )
            / n,
            "mean_reciprocal_rank": sum(
                case.retrieval_metrics.reciprocal_rank
                for case in self.cases
            )
            / n,
            "mean_reference_token_f1": sum(
                case.answer_metrics.reference_token_f1
                for case in self.cases
            )
            / n,
            "mean_context_token_support": sum(
                case.answer_metrics.context_token_support
                for case in self.cases
            )
            / n,
            "groundedness_case_count": len(
                groundedness_scores
            ),
            "mean_groundedness_score": (
                _mean_optional(groundedness_scores)
            ),
            "answer_relevance_case_count": len(
                relevance_scores
            ),
            "mean_answer_relevance": (
                _mean_optional(relevance_scores)
            ),
            "correctness_case_count": len(
                correctness_scores
            ),
            "mean_correctness": (
                _mean_optional(correctness_scores)
            ),
            "mean_retrieval_latency_ms": sum(
                case.retrieval_latency_ms
                for case in self.cases
            )
            / n,
            "mean_pipeline_latency_ms": sum(
                case.pipeline_latency_ms
                for case in self.cases
            )
            / n,
            "failure_counts": failure_counts,
            "generation_failure_counts": (
                generation_failure_counts
            ),
        }

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "top_k": self.top_k,
            "minimum_relevance_score": (
                self.minimum_relevance_score
            ),
            "answer_f1_review_threshold": (
                self.answer_f1_review_threshold
            ),
            "summary": self.summary,
            "cases": [
                case.to_dict()
                for case in self.cases
            ],
        }


class RAGEvaluator:
    def __init__(
        self,
        retriever: SearchRetriever,
        pipeline: AnswerPipeline,
        *,
        top_k: int = 5,
        minimum_relevance_score: float = 0.10,
        answer_f1_review_threshold: float = 0.20,
    ) -> None:
        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )
        if not -1 <= minimum_relevance_score <= 1:
            raise ValueError(
                "minimum_relevance_score must be "
                "between -1 and 1"
            )
        if not 0 <= answer_f1_review_threshold <= 1:
            raise ValueError(
                "answer_f1_review_threshold must be "
                "between 0 and 1"
            )

        self.retriever = retriever
        self.pipeline = pipeline
        self.top_k = top_k
        self.minimum_relevance_score = (
            minimum_relevance_score
        )
        self.answer_f1_review_threshold = (
            answer_f1_review_threshold
        )

    def evaluate_case(
        self,
        case: EvaluationCase,
    ) -> CaseEvaluation:
        started = time.perf_counter()

        raw = self.retriever.search(
            case.question,
            top_k=self.top_k,
        )

        retrieval_ms = (
            time.perf_counter() - started
        ) * 1000

        retrieved = tuple(
            RetrievedTrace(
                rank=int(result.rank),
                score=float(result.score),
                chunk_id=str(
                    result.chunk.chunk_id
                ),
                text=str(result.chunk.text),
            )
            for result in raw
        )

        metrics = evaluate_retrieval(
            tuple(
                item.chunk_id
                for item in retrieved
            ),
            case.expected_chunk_ids,
            k=self.top_k,
        )

        started = time.perf_counter()

        answer = self.pipeline.answer(
            case.question,
            top_k=self.top_k,
            minimum_relevance_score=(
                self.minimum_relevance_score
            ),
        )

        pipeline_ms = (
            time.perf_counter() - started
        ) * 1000

        generated_answer = str(answer.answer)

        source_ids = tuple(
            str(source.chunk_id)
            for source in answer.sources
        )

        answer_metrics = evaluate_answer(
            generated_answer,
            case.reference_answer,
            retrieved_contexts=(
                item.text
                for item in retrieved
            ),
            used_source_ids=source_ids,
            expected_chunk_ids=(
                case.expected_chunk_ids
            ),
        )

        groundedness = (
            evaluate_labelled_groundedness(
                generated_answer,
                source_ids,
                case.groundedness_claims,
            )
            if case.groundedness_claims
            else None
        )

        groundedness_score = (
            groundedness.groundedness_score
            if (
                groundedness is not None
                and groundedness.total_claims > 0
            )
            else None
        )

        generation_evaluation = (
            evaluate_generation_quality(
                generated_answer,
                hit_at_k=metrics.hit_at_k,
                groundedness=groundedness_score,
                relevance_phrases=(
                    case.answer_relevance_phrases
                ),
                correctness_required_phrases=(
                    case.correctness_required_phrases
                ),
                correctness_forbidden_phrases=(
                    case.correctness_forbidden_phrases
                ),
            )
        )

        categories = set(
            generation_evaluation.failure_categories
        )

        if FailureCategory.RETRIEVAL_FAILURE in categories:
            failure_stage, status = (
                "retrieval",
                "FAIL",
            )
        elif (
            bool(answer.insufficient_context)
            or not answer_metrics.expected_source_used
        ):
            failure_stage, status = (
                "augmentation",
                "FAIL",
            )
        elif (
            FailureCategory.UNGROUNDED_GENERATION
            in categories
        ):
            failure_stage, status = (
                "groundedness",
                "FAIL",
            )
        elif FailureCategory.IRRELEVANT_ANSWER in categories:
            failure_stage, status = (
                "relevance",
                "FAIL",
            )
        elif FailureCategory.INCORRECT_ANSWER in categories:
            failure_stage, status = (
                "correctness",
                "FAIL",
            )
        elif (
            answer_metrics.reference_token_f1
            < self.answer_f1_review_threshold
        ):
            failure_stage, status = (
                "generation_proxy",
                "REVIEW",
            )
        else:
            failure_stage, status = (
                "none",
                "PASS",
            )

        return CaseEvaluation(
            case_id=case.case_id,
            question=case.question,
            expected_chunk_ids=(
                case.expected_chunk_ids
            ),
            reference_answer=(
                case.reference_answer
            ),
            retrieved=retrieved,
            retrieval_metrics=metrics,
            generated_answer=generated_answer,
            used_source_ids=source_ids,
            answer_metrics=answer_metrics,
            groundedness=groundedness,
            generation_evaluation=(
                generation_evaluation
            ),
            insufficient_context=bool(
                answer.insufficient_context
            ),
            retrieval_latency_ms=retrieval_ms,
            pipeline_latency_ms=pipeline_ms,
            failure_stage=failure_stage,
            status=status,
        )

    def evaluate(
        self,
        dataset: EvaluationDataset,
    ) -> EvaluationRun:
        return EvaluationRun(
            dataset_name=dataset.name,
            top_k=self.top_k,
            minimum_relevance_score=(
                self.minimum_relevance_score
            ),
            answer_f1_review_threshold=(
                self.answer_f1_review_threshold
            ),
            cases=tuple(
                self.evaluate_case(case)
                for case in dataset.cases
            ),
        )
