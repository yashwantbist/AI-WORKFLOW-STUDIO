"""RAG evaluation utilities."""

from .answer_metrics import (
    AnswerMetrics,
    evaluate_answer,
)
from .dataset import (
    EvaluationCase,
    EvaluationDataset,
)
from .evaluator import (
    CaseEvaluation,
    EvaluationRun,
    RAGEvaluator,
)
from .generation_quality import (
    DimensionEvaluation,
    FailureCategory,
    GenerationEvaluation,
    classify_failures,
    evaluate_answer_relevance,
    evaluate_correctness,
    evaluate_generation_quality,
)
from .groundedness import (
    ClaimEvaluation,
    GroundednessClaim,
    GroundednessEvaluation,
    evaluate_groundedness,
    evaluate_labelled_groundedness,
)
from .retrieval_metrics import (
    RetrievalMetrics,
    evaluate_retrieval,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RetrievalEvaluation = RetrievalMetrics

__all__ = [
    "AnswerMetrics",
    "CaseEvaluation",
    "ClaimEvaluation",
    "DimensionEvaluation",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationRun",
    "FailureCategory",
    "GenerationEvaluation",
    "GroundednessClaim",
    "GroundednessEvaluation",
    "RAGEvaluator",
    "RetrievalEvaluation",
    "RetrievalMetrics",
    "classify_failures",
    "evaluate_answer",
    "evaluate_answer_relevance",
    "evaluate_correctness",
    "evaluate_generation_quality",
    "evaluate_groundedness",
    "evaluate_labelled_groundedness",
    "evaluate_retrieval",
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
