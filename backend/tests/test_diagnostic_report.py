from dataclasses import dataclass

from backend.ml.rag.evaluation.diagnostics import RetrievalStatus
from backend.ml.rag.evaluation.offline_dataset import (
    OfflineEvaluationCase,
    OfflineEvaluationDataset,
)
from backend.ml.rag.evaluation.offline_report import (
    format_diagnostic_report,
    save_diagnostic_json_report,
)
from backend.ml.rag.evaluation.offline_runner import (
    OfflineRetrievalEvaluator,
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str


@dataclass(frozen=True)
class Result:
    rank: int
    score: float
    chunk: Chunk


class Retriever:
    def __init__(self):
        self.responses = {
            "success": ("c1", "x"),
            "partial": ("c2", "x"),
            "miss": ("x", "y"),
            "empty": (),
        }

    def search(self, query, *, top_k=5, filters=None):
        return tuple(
            Result(index, 1.0 / index, Chunk(chunk_id))
            for index, chunk_id in enumerate(
                self.responses[query][:top_k],
                start=1,
            )
        )


def build_report():
    dataset = OfflineEvaluationDataset(
        name="report-test",
        description="",
        cases=(
            OfflineEvaluationCase(
                "success",
                "success",
                ("c1",),
            ),
            OfflineEvaluationCase(
                "partial",
                "partial",
                ("c1", "c2"),
            ),
            OfflineEvaluationCase(
                "miss",
                "miss",
                ("c1",),
            ),
            OfflineEvaluationCase(
                "empty",
                "empty",
                ("c1",),
            ),
        ),
    )
    return OfflineRetrievalEvaluator(
        Retriever(),
        k=2,
    ).evaluate_dataset(dataset)


def test_report_counts_statuses():
    report = build_report()

    assert report.successful_queries == 1
    assert report.partial_queries == 1
    assert report.complete_misses == 1
    assert report.no_result_queries == 1
    assert len(report.failures) == 3


def test_failures_preserve_expected_and_retrieved_ids():
    report = build_report()
    by_id = {result.query_id: result for result in report.failures}

    assert by_id["partial"].status is RetrievalStatus.PARTIAL
    assert by_id["partial"].relevant_ids == ("c1", "c2")
    assert by_id["partial"].retrieved_ids == ("c2", "x")
    assert by_id["partial"].missed_relevant_ids == ("c1",)


def test_terminal_report_surfaces_failures():
    text = format_diagnostic_report(build_report())

    assert "RETRIEVAL STATUS" in text
    assert "Complete misses: 1" in text
    assert "[partial] partial" in text
    assert "Status: partial" in text
    assert "Expected: c1, c2" in text
    assert "Retrieved: c2, x" in text


def test_diagnostic_json_contains_raw_results(tmp_path):
    output = save_diagnostic_json_report(
        build_report(),
        tmp_path / "diagnostic.json",
    )

    content = output.read_text(encoding="utf-8")
    assert '"failure_count": 3' in content
    assert '"status": "complete_miss"' in content
    assert '"missed_relevant_ids"' in content
