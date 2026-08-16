import pytest

from backend.ml.rag.evaluation.generation_quality import (
    FailureCategory,
    classify_failures,
    evaluate_answer_relevance,
    evaluate_correctness,
    evaluate_generation_quality,
)


def test_answer_relevance_passes_when_required_phrase_is_present():
    result = evaluate_answer_relevance(
        "Refund requests are accepted within 30 days.",
        ["30 days"],
    )

    assert result is not None
    assert result.score == 1.0
    assert result.passed is True
    assert result.matched_phrases == ("30 days",)
    assert result.missing_phrases == ()


def test_answer_relevance_fails_when_answer_does_not_address_requirement():
    result = evaluate_answer_relevance(
        "Our company offers refunds.",
        ["30 days"],
    )

    assert result is not None
    assert result.score == 0.0
    assert result.passed is False
    assert result.missing_phrases == ("30 days",)


def test_answer_relevance_without_labels_returns_none():
    assert evaluate_answer_relevance(
        "Any answer.",
        [],
    ) is None


def test_correctness_passes_with_required_fact_and_no_forbidden_fact():
    result = evaluate_correctness(
        "Refund requests are accepted within 30 days.",
        required_phrases=["30 days"],
        forbidden_phrases=["60 days"],
    )

    assert result is not None
    assert result.score == 1.0
    assert result.matched_phrases == ("30 days",)
    assert result.conflicting_phrases == ()


def test_correctness_fails_when_wrong_fact_is_present():
    result = evaluate_correctness(
        "Refund requests are accepted within 60 days.",
        required_phrases=["30 days"],
        forbidden_phrases=["60 days"],
    )

    assert result is not None
    assert result.score == 0.0
    assert result.missing_phrases == ("30 days",)
    assert result.conflicting_phrases == ("60 days",)


def test_correctness_without_labels_returns_none():
    assert evaluate_correctness("Any answer.") is None


def test_phrase_matching_is_case_insensitive_and_whitespace_tolerant():
    result = evaluate_answer_relevance(
        "Customers have   THIRTY DAYS to request a refund.",
        ["thirty days"],
    )

    assert result is not None
    assert result.score == 1.0


def test_good_retrieval_and_good_answer_classifies_success():
    categories = classify_failures(
        hit_at_k=1.0,
        groundedness=1.0,
        answer_relevance=1.0,
        correctness=1.0,
    )

    assert categories == (FailureCategory.SUCCESS,)


def test_each_failure_category_is_independent():
    assert classify_failures(
        hit_at_k=0.0,
        groundedness=1.0,
        answer_relevance=1.0,
        correctness=1.0,
    ) == (FailureCategory.RETRIEVAL_FAILURE,)

    assert classify_failures(
        hit_at_k=1.0,
        groundedness=0.0,
        answer_relevance=1.0,
        correctness=1.0,
    ) == (FailureCategory.UNGROUNDED_GENERATION,)

    assert classify_failures(
        hit_at_k=1.0,
        groundedness=1.0,
        answer_relevance=0.0,
        correctness=1.0,
    ) == (FailureCategory.IRRELEVANT_ANSWER,)

    assert classify_failures(
        hit_at_k=1.0,
        groundedness=1.0,
        answer_relevance=1.0,
        correctness=0.0,
    ) == (FailureCategory.INCORRECT_ANSWER,)


def test_multiple_simultaneous_failures_are_preserved():
    categories = classify_failures(
        hit_at_k=0.0,
        groundedness=0.5,
        answer_relevance=0.0,
        correctness=0.0,
    )

    assert categories == (
        FailureCategory.RETRIEVAL_FAILURE,
        FailureCategory.UNGROUNDED_GENERATION,
        FailureCategory.IRRELEVANT_ANSWER,
        FailureCategory.INCORRECT_ANSWER,
    )


def test_generation_evaluation_does_not_invent_unlabelled_scores():
    result = evaluate_generation_quality(
        "A generated answer.",
        hit_at_k=1.0,
        groundedness=None,
    )

    assert result.groundedness is None
    assert result.answer_relevance is None
    assert result.correctness is None
    assert result.failure_categories == (FailureCategory.SUCCESS,)


def test_invalid_score_is_rejected():
    with pytest.raises(ValueError, match="correctness"):
        classify_failures(
            hit_at_k=1.0,
            groundedness=1.0,
            answer_relevance=1.0,
            correctness=1.5,
        )
