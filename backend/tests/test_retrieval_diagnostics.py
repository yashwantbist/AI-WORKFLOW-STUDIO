from dataclasses import dataclass

import pytest

from backend.ml.rag.evaluation.diagnostics import (
    RetrievalStatus,
    classify_retrieval,
)
from backend.ml.rag.evaluation.offline_dataset import (
    OfflineEvaluationCase,
    OfflineEvaluationDataset,
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

    def search(self, query, *, top_k=5, filters=None):
        ids = self.responses.get(query, ())
        return tuple(
            FakeResult(
                rank=index,
                score=1.0 - ((index - 1) * 0.1),
                chunk=FakeChunk(chunk_id),
            )
            for index, chunk_id in enumerate(ids[:top_k], start=1)
        )


def test_classifies_success():
    assert (
        classify_retrieval(["c1", "c2"], {"c1", "c2"})
        is RetrievalStatus.SUCCESS
    )


def test_classifies_no_results():
    assert (
        classify_retrieval([], {"c1"})
        is RetrievalStatus.NO_RESULTS
    )


def test_classifies_partial_retrieval():
    assert (
        classify_retrieval(["c1", "x"], {"c1", "c2"})
        is RetrievalStatus.PARTIAL
    )


def test_classifies_complete_miss():
    assert (
        classify_retrieval(["x", "y"], {"c1", "c2"})
        is RetrievalStatus.COMPLETE_MISS
    )


def test_no_answer_case_with_results_is_irrelevant_results():
    assert (
        classify_retrieval(["x"], set())
        is RetrievalStatus.IRRELEVANT_RESULTS
    )


def test_no_answer_case_with_no_results_is_success():
    assert (
        classify_retrieval([], set())
        is RetrievalStatus.SUCCESS
    )


def test_success_can_still_contain_irrelevant_noise():
    dataset = OfflineEvaluationDataset(
        name="diagnostic-test",
        description="",
        cases=(
            OfflineEvaluationCase(
                case_id="q1",
                query="success-with-noise",
                relevant_ids=("c1",),
            ),
        ),
    )
    retriever = FakeRetriever(
        {"success-with-noise": ("c1", "x", "y")}
    )

    report = OfflineRetrievalEvaluator(
        retriever,
        k=3,
    ).evaluate_dataset(dataset)

    result = report.results[0]
    assert result.status is RetrievalStatus.SUCCESS
    assert result.irrelevant_retrieved_ids == ("x", "y")
    assert result.has_irrelevant_results is True
    assert report.successful_queries == 1
    assert report.queries_with_irrelevant_results == 1
