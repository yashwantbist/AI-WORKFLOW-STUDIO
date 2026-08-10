import pytest

from backend.ml.rag.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_day20_example_precision_is_60_percent():
    retrieved = ["c1", "c2", "c7", "c3", "c9"]
    relevant = {"c1", "c2", "c3", "c4"}
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.60)


def test_day20_example_recall_is_75_percent():
    retrieved = ["c1", "c2", "c7", "c3", "c9"]
    relevant = {"c1", "c2", "c3", "c4"}
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(0.75)


def test_empty_retrieval_returns_zero():
    assert precision_at_k([], {"c1"}, 5) == 0.0
    assert recall_at_k([], {"c1"}, 5) == 0.0


def test_zero_relevant_documents_returns_zero_metrics():
    assert precision_at_k(["c1"], set(), 1) == 0.0
    assert recall_at_k(["c1"], set(), 1) == 0.0
    assert hit_at_k(["c1"], set(), 1) == 0.0


def test_k_larger_than_results_uses_actual_count():
    assert precision_at_k(["c1", "x"], {"c1"}, 5) == pytest.approx(0.5)


def test_duplicates_consume_slots_without_extra_credit():
    retrieved = ["c1", "c1", "c2"]
    relevant = {"c1", "c2", "c3"}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_structured_result_includes_counts():
    result = evaluate_retrieval(
        ["c1", "c1", "wrong", "c2"],
        {"c1", "c2", "c3"},
        k=4,
    )
    assert result.relevant_retrieved == 2
    assert result.retrieved_count == 4
    assert result.total_relevant == 3
    assert result.duplicate_count == 1
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["wrong", "c1"], {"c1"}, 2) == pytest.approx(0.5)


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["c1"], {"c1"}, 0)
