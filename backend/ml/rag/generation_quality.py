"""Deterministic generation-quality evaluation for labelled RAG fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable


WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value.strip()).casefold()


def _clean_phrases(
    values: Iterable[str],
    *,
    name: str,
) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip() for value in values)

    if any(not value for value in cleaned):
        raise ValueError(f"{name} cannot contain empty phrases")

    normalized = tuple(_normalize_text(value) for value in cleaned)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique phrases")

    return cleaned


class FailureCategory(str, Enum):
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    UNGROUNDED_GENERATION = "UNGROUNDED_GENERATION"
    IRRELEVANT_ANSWER = "IRRELEVANT_ANSWER"
    INCORRECT_ANSWER = "INCORRECT_ANSWER"
    SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class DimensionEvaluation:
    """Structured diagnostics for one deterministic generation dimension."""

    score: float
    passed: bool
    matched_phrases: tuple[str, ...] = ()
    missing_phrases: tuple[str, ...] = ()
    conflicting_phrases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "passed": self.passed,
            "matched_phrases": list(self.matched_phrases),
            "missing_phrases": list(self.missing_phrases),
            "conflicting_phrases": list(self.conflicting_phrases),
        }


@dataclass(frozen=True)
class GenerationEvaluation:
    groundedness: float | None
    answer_relevance: float | None
    correctness: float | None
    failure_categories: tuple[FailureCategory, ...]
    relevance_details: DimensionEvaluation | None = None
    correctness_details: DimensionEvaluation | None = None

    @property
    def successful(self) -> bool:
        return self.failure_categories == (FailureCategory.SUCCESS,)

    def to_dict(self) -> dict[str, object]:
        return {
            "groundedness": self.groundedness,
            "answer_relevance": self.answer_relevance,
            "correctness": self.correctness,
            "failure_categories": [
                category.value
                for category in self.failure_categories
            ],
            "successful": self.successful,
            "relevance_details": (
                self.relevance_details.to_dict()
                if self.relevance_details is not None
                else None
            ),
            "correctness_details": (
                self.correctness_details.to_dict()
                if self.correctness_details is not None
                else None
            ),
        }


def evaluate_answer_relevance(
    answer: str,
    required_phrases: Iterable[str],
) -> DimensionEvaluation | None:
    """Evaluate labelled answer relevance using required answer phrases.

    This is a deterministic v1 fixture evaluator, not a semantic judge.
    All labelled phrases must appear in the answer for the binary score
    to be 1.0. No labels means the dimension is not evaluated.
    """

    required = _clean_phrases(
        required_phrases,
        name="required_phrases",
    )
    if not required:
        return None

    normalized_answer = _normalize_text(answer)

    matched = tuple(
        phrase
        for phrase in required
        if _normalize_text(phrase) in normalized_answer
    )
    missing = tuple(
        phrase
        for phrase in required
        if _normalize_text(phrase) not in normalized_answer
    )

    passed = not missing

    return DimensionEvaluation(
        score=1.0 if passed else 0.0,
        passed=passed,
        matched_phrases=matched,
        missing_phrases=missing,
    )


def evaluate_correctness(
    answer: str,
    *,
    required_phrases: Iterable[str] = (),
    forbidden_phrases: Iterable[str] = (),
) -> DimensionEvaluation | None:
    """Evaluate deterministic correctness against labelled fact phrases.

    Required phrases represent facts that must appear.
    Forbidden phrases represent explicitly wrong facts that must not appear.
    No labels means correctness is not evaluated.
    """

    required = _clean_phrases(
        required_phrases,
        name="required_phrases",
    )
    forbidden = _clean_phrases(
        forbidden_phrases,
        name="forbidden_phrases",
    )

    if not required and not forbidden:
        return None

    normalized_answer = _normalize_text(answer)

    matched = tuple(
        phrase
        for phrase in required
        if _normalize_text(phrase) in normalized_answer
    )
    missing = tuple(
        phrase
        for phrase in required
        if _normalize_text(phrase) not in normalized_answer
    )
    conflicting = tuple(
        phrase
        for phrase in forbidden
        if _normalize_text(phrase) in normalized_answer
    )

    passed = not missing and not conflicting

    return DimensionEvaluation(
        score=1.0 if passed else 0.0,
        passed=passed,
        matched_phrases=matched,
        missing_phrases=missing,
        conflicting_phrases=conflicting,
    )


def _validate_optional_score(
    value: float | None,
    *,
    name: str,
) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def classify_failures(
    *,
    hit_at_k: float,
    groundedness: float | None,
    answer_relevance: float | None,
    correctness: float | None,
) -> tuple[FailureCategory, ...]:
    """Return every applicable failure category for one query."""

    if hit_at_k not in (0, 0.0, 1, 1.0):
        raise ValueError("hit_at_k must be 0 or 1")

    _validate_optional_score(
        groundedness,
        name="groundedness",
    )
    _validate_optional_score(
        answer_relevance,
        name="answer_relevance",
    )
    _validate_optional_score(
        correctness,
        name="correctness",
    )

    failures: list[FailureCategory] = []

    if hit_at_k == 0:
        failures.append(FailureCategory.RETRIEVAL_FAILURE)

    if groundedness is not None and groundedness < 1.0:
        failures.append(FailureCategory.UNGROUNDED_GENERATION)

    if answer_relevance is not None and answer_relevance < 1.0:
        failures.append(FailureCategory.IRRELEVANT_ANSWER)

    if correctness is not None and correctness < 1.0:
        failures.append(FailureCategory.INCORRECT_ANSWER)

    if not failures:
        return (FailureCategory.SUCCESS,)

    return tuple(failures)


def evaluate_generation_quality(
    answer: str,
    *,
    hit_at_k: float,
    groundedness: float | None,
    relevance_phrases: Iterable[str] = (),
    correctness_required_phrases: Iterable[str] = (),
    correctness_forbidden_phrases: Iterable[str] = (),
) -> GenerationEvaluation:
    """Compose deterministic generation-quality evaluators."""

    relevance = evaluate_answer_relevance(
        answer,
        relevance_phrases,
    )
    correctness_result = evaluate_correctness(
        answer,
        required_phrases=correctness_required_phrases,
        forbidden_phrases=correctness_forbidden_phrases,
    )

    relevance_score = (
        relevance.score
        if relevance is not None
        else None
    )
    correctness_score = (
        correctness_result.score
        if correctness_result is not None
        else None
    )

    return GenerationEvaluation(
        groundedness=groundedness,
        answer_relevance=relevance_score,
        correctness=correctness_score,
        failure_categories=classify_failures(
            hit_at_k=hit_at_k,
            groundedness=groundedness,
            answer_relevance=relevance_score,
            correctness=correctness_score,
        ),
        relevance_details=relevance,
        correctness_details=correctness_result,
    )
