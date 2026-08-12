from dataclasses import dataclass

import pytest

from backend.ml.rag.evaluation.offline_dataset import (
    OfflineEvaluationCase,
    OfflineEvaluationDataset,
)
from backend.ml.rag.evaluation.offline_report import (
    format_terminal_report,
    save_json_report,
)
from backend.ml.rag.evaluation.offline_runner import (
    OfflineRetrievalEvaluator,
)


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str


@dataclass(frozen=True)
class FakeResult:
    rank: int
    score: float
    chunk: FakeChunk


class FakeRetriever:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def search(self, query, *, top_k=5, filters=None):
        self.calls.append((query, top_k))
        ids = self.responses.get(query, ())
        return tuple(
            FakeResult(
                rank=index,
                score=1.0 - (index - 1) * 0.1,
                chunk=FakeChunk(chunk_id),
            )
            for index, chunk_id in enumerate(ids[:top_k], start=1)
        )


def make_dataset():
    return OfflineEvaluationDataset(
        name="runner-test",
        description="deterministic",
        cases=(
            OfflineEvaluationCase(
                case_id="perfect",
                query="perfect",
                relevant_ids=("c1", "c2"),
            ),
            OfflineEvaluationCase(
                case_id="partial",
                query="partial",
                relevant_ids=("c1", "c2"),
            ),
            OfflineEvaluationCase(
                case_id="irrelevant",
                query="irrelevant",
                relevant_ids=("c1",),
            ),
            OfflineEvaluationCase(
                case_id="empty",
                query="empty",
                relevant_ids=("c1",),
            ),
            OfflineEvaluationCase(
                case_id="multiple",
                query="multiple",
                relevant_ids=("c1", "c2", "c3"),
            ),
        ),
    )


def make_retriever():
    return FakeRetriever(
        {
            "perfect": ("c1", "c2"),
            "partial": ("c1", "x"),
            "irrelevant": ("x", "y"),
            "empty": (),
            "multiple": ("c3", "c1", "x"),
        }
    )


def test_per_query_results_cover_required_cases():
    report = OfflineRetrievalEvaluator(
        make_retriever(),
        k=2,
    ).evaluate_dataset(make_dataset())

    by_id = {result.query_id: result for result in report.results}

    assert by_id["perfect"].metrics.precision_at_k == 1.0
    assert by_id["perfect"].metrics.recall_at_k == 1.0

    assert by_id["partial"].metrics.precision_at_k == pytest.approx(0.5)
    assert by_id["partial"].metrics.recall_at_k == pytest.approx(0.5)

    assert by_id["irrelevant"].metrics.precision_at_k == 0.0
    assert by_id["irrelevant"].metrics.recall_at_k == 0.0

    assert by_id["empty"].retrieved_ids == ()
    assert by_id["empty"].metrics.precision_at_k == 0.0
    assert by_id["empty"].metrics.recall_at_k == 0.0

    assert by_id["multiple"].metrics.precision_at_k == 1.0
    assert by_id["multiple"].metrics.recall_at_k == pytest.approx(2 / 3)
    assert by_id["multiple"].missed_relevant_ids == ("c2",)


def test_aggregate_metrics_are_means_of_raw_results():
    report = OfflineRetrievalEvaluator(
        make_retriever(),
        k=2,
    ).evaluate_dataset(make_dataset())

    expected_precision = (1.0 + 0.5 + 0.0 + 0.0 + 1.0) / 5
    expected_recall = (1.0 + 0.5 + 0.0 + 0.0 + (2 / 3)) / 5

    assert report.total_queries == 5
    assert report.mean_precision_at_k == pytest.approx(expected_precision)
    assert report.mean_recall_at_k == pytest.approx(expected_recall)


def test_retriever_runs_once_per_case():
    fake = make_retriever()
    OfflineRetrievalEvaluator(fake, k=3).evaluate_dataset(make_dataset())

    assert len(fake.calls) == 5
    assert all(top_k == 3 for _, top_k in fake.calls)


def test_report_preserves_failure_details_and_exports_json(tmp_path):
    report = OfflineRetrievalEvaluator(
        make_retriever(),
        k=2,
    ).evaluate_dataset(make_dataset())

    text = format_terminal_report(report)
    output = save_json_report(report, tmp_path / "report.json")

    assert "Missed relevant IDs" in text
    assert "Mean Precision@2" in text
    assert output.exists()
    assert '"query_id": "partial"' in output.read_text(encoding="utf-8")


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError, match="k must be at least 1"):
        OfflineRetrievalEvaluator(make_retriever(), k=0)
