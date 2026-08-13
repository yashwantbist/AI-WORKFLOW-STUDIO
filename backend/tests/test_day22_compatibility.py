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
from backend.ml.rag.evaluation.offline_runner import OfflineRetrievalEvaluator


@dataclass(frozen=True)
class Chunk:
    chunk_id: str


@dataclass(frozen=True)
class Result:
    rank: int
    score: float
    chunk: Chunk


class Retriever:
    def search(self, query, *, top_k=5, filters=None):
        return (
            Result(1, 0.9, Chunk("c1")),
            Result(2, 0.4, Chunk("x")),
        )[:top_k]


def test_day22_public_behavior_remains_available(tmp_path):
    dataset = OfflineEvaluationDataset(
        name="compat",
        description="",
        cases=(
            OfflineEvaluationCase(
                "q1",
                "question",
                ("c1",),
            ),
        ),
    )

    report = OfflineRetrievalEvaluator(
        Retriever(),
        k=2,
    ).evaluate_dataset(dataset)

    assert report.mean_precision_at_k == pytest.approx(0.5)
    assert report.mean_recall_at_k == 1.0
    assert report.hit_rate_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0

    text = format_terminal_report(report)
    assert "Mean Precision@2" in text

    output = save_json_report(
        report,
        tmp_path / "day22.json",
    )
    assert output.exists()
