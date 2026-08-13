from dataclasses import replace

import pytest

from backend.ml.rag.evaluation.baseline import (
    RegressionStatus,
    RetrievalBaseline,
    compare_to_baseline,
    save_baseline,
)
from backend.ml.rag.evaluation.offline_runner import EvaluationReport


def candidate(
    *,
    precision=0.32,
    recall=1.0,
    hit_rate=1.0,
    mrr=1.0,
    dataset_name="day22-transformer-retrieval-golden-set",
    k=5,
    total_queries=5,
):
    return EvaluationReport(
        dataset_name=dataset_name,
        k=k,
        total_queries=total_queries,
        mean_precision_at_k=precision,
        mean_recall_at_k=recall,
        hit_rate_at_k=hit_rate,
        mean_reciprocal_rank=mrr,
        successful_queries=5,
        partial_queries=0,
        complete_misses=0,
        no_result_queries=0,
        irrelevant_only_queries=0,
        queries_with_irrelevant_results=5,
        results=(),
    )


def measured_baseline():
    return RetrievalBaseline(
        schema_version=1,
        dataset_name="day22-transformer-retrieval-golden-set",
        k=5,
        total_queries=5,
        mean_precision_at_k=0.32,
        mean_recall_at_k=1.0,
        hit_rate_at_k=1.0,
        mean_reciprocal_rank=1.0,
    )


def test_equal_candidate_passes():
    comparison = compare_to_baseline(
        candidate(),
        measured_baseline(),
        tolerance=0.01,
    )

    assert comparison.status is RegressionStatus.PASS
    assert comparison.regressed_metrics == ()
    assert comparison.recall_delta == 0.0


def test_small_floating_change_within_tolerance_passes():
    comparison = compare_to_baseline(
        candidate(precision=0.315),
        measured_baseline(),
        tolerance=0.01,
    )

    assert comparison.status is RegressionStatus.PASS


def test_metric_drop_beyond_tolerance_is_regression():
    comparison = compare_to_baseline(
        candidate(recall=0.80),
        measured_baseline(),
        tolerance=0.01,
    )

    assert comparison.status is RegressionStatus.REGRESSION
    assert "mean_recall_at_k" in comparison.regressed_metrics
    assert comparison.recall_delta == pytest.approx(-0.20)


def test_mismatched_k_is_rejected():
    with pytest.raises(ValueError, match="k values do not match"):
        compare_to_baseline(
            candidate(k=3),
            measured_baseline(),
        )


def test_mismatched_dataset_is_rejected():
    with pytest.raises(ValueError, match="dataset names do not match"):
        compare_to_baseline(
            candidate(dataset_name="different"),
            measured_baseline(),
        )


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="tolerance cannot be negative"):
        compare_to_baseline(
            candidate(),
            measured_baseline(),
            tolerance=-0.1,
        )


def test_baseline_round_trip(tmp_path):
    path = save_baseline(
        measured_baseline(),
        tmp_path / "baseline.json",
    )
    loaded = RetrievalBaseline.load(path)

    assert loaded == measured_baseline()


def test_missing_baseline_is_not_fabricated(tmp_path):
    with pytest.raises(FileNotFoundError, match="baseline not found"):
        RetrievalBaseline.load(tmp_path / "missing.json")
