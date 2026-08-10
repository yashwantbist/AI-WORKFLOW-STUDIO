"""RAG evaluation utilities."""

from .answer_metrics import AnswerMetrics, evaluate_answer
from .dataset import EvaluationCase, EvaluationDataset
from .evaluator import CaseEvaluation, EvaluationRun, RAGEvaluator
from .groundedness import (
    ClaimEvaluation,
    GroundednessEvaluation,
    evaluate_groundedness,
)
from .retrieval_metrics import (
    RetrievalEvaluation,
    RetrievalMetrics,
    evaluate_retrieval,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "AnswerMetrics",
    "CaseEvaluation",
    "ClaimEvaluation",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationRun",
    "GroundednessEvaluation",
    "RAGEvaluator",
    "RetrievalEvaluation",
    "RetrievalMetrics",
    "evaluate_answer",
    "evaluate_groundedness",
    "evaluate_retrieval",
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
