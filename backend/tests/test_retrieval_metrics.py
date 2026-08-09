import pytest

from backend.ml.rag.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k_present():
    assert hit_at_k(["a", "expected", "c"], ["expected"], 3) == 1.0


def test_hit_at_k_respects_boundary():
    assert hit_at_k(["a", "b", "expected"], ["expected"], 2) == 0.0


def test_recall_multiple_expected():
    assert recall_at_k(["a", "x"], ["x", "y"], 2) == pytest.approx(0.5)


def test_precision():
    assert precision_at_k(["x", "wrong", "other"], ["x"], 3) == pytest.approx(1 / 3)


def test_precision_short_result_set():
    assert precision_at_k(["x"], ["x"], 5) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["wrong", "x"], ["x"], 2) == pytest.approx(0.5)


def test_reciprocal_rank_missing():
    assert reciprocal_rank(["a", "b"], ["x"], 2) == 0.0


def test_combined_metrics():
    metrics = evaluate_retrieval(["wrong", "x"], ["x"], k=2)
    assert metrics.hit_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == pytest.approx(0.5)
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    assert metrics.retrieval_passed


def test_invalid_k():
    with pytest.raises(ValueError):
        hit_at_k(["x"], ["x"], 0)


def test_empty_expected():
    with pytest.raises(ValueError):
        recall_at_k(["x"], [], 1)
