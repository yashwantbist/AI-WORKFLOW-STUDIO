"""Day 24 ranking-aware RAG evaluation tests."""

from dataclasses import dataclass
import pytest

from backend.ml.rag.evaluation.offline_dataset import (
    OfflineEvaluationCase,
    OfflineEvaluationDataset,
)
from backend.ml.rag.evaluation.offline_runner import OfflineRetrievalEvaluator
from backend.ml.rag.evaluation.retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k_for_ranks_1_2_and_3():
    relevant = {"c3"}

    assert hit_at_k(["c3", "c1", "c2"], relevant, 3) == 1.0
    assert hit_at_k(["c1", "c3", "c2"], relevant, 3) == 1.0
    assert hit_at_k(["c1", "c2", "c3"], relevant, 3) == 1.0


def test_reciprocal_rank_for_ranks_1_2_and_3():
    relevant = {"c3"}

    assert reciprocal_rank(["c3", "c1", "c2"], relevant, 3) == 1.0
    assert reciprocal_rank(
        ["c1", "c3", "c2"],
        relevant,
        3,
    ) == pytest.approx(0.5)
    assert reciprocal_rank(
        ["c1", "c2", "c3"],
        relevant,
        3,
    ) == pytest.approx(1 / 3)


def test_miss_returns_zero_hit_and_rr():
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c9"}

    assert hit_at_k(retrieved, relevant, 3) == 0.0
    assert reciprocal_rank(retrieved, relevant, 3) == 0.0


def test_empty_retrieval_and_empty_relevant_set():
    assert hit_at_k([], {"c1"}, 5) == 0.0
    assert reciprocal_rank([], {"c1"}, 5) == 0.0

    assert hit_at_k(["c1"], set(), 1) == 0.0
    assert reciprocal_rank(["c1"], set(), 1) == 0.0


def test_multiple_relevant_ids_use_first_relevant_rank():
    retrieved = ["x", "c7", "c3", "c9"]
    relevant = {"c3", "c7"}

    assert reciprocal_rank(
        retrieved,
        relevant,
        4,
    ) == pytest.approx(0.5)


def test_k_boundary_excludes_relevant_item_outside_top_k():
    retrieved = ["c1", "c2", "c3", "c7"]
    relevant = {"c7"}

    assert hit_at_k(retrieved, relevant, 3) == 0.0
    assert reciprocal_rank(retrieved, relevant, 3) == 0.0

    assert hit_at_k(retrieved, relevant, 4) == 1.0
    assert reciprocal_rank(
        retrieved,
        relevant,
        4,
    ) == pytest.approx(0.25)


def test_same_recall_can_have_different_reciprocal_rank():
    relevant = {"c7"}

    retriever_a = ["c7", "c2", "c9", "c4", "c1"]
    retriever_b = ["c2", "c9", "c4", "c1", "c7"]

    assert recall_at_k(retriever_a, relevant, 5) == 1.0
    assert recall_at_k(retriever_b, relevant, 5) == 1.0

    assert reciprocal_rank(retriever_a, relevant, 5) == 1.0
    assert reciprocal_rank(
        retriever_b,
        relevant,
        5,
    ) == pytest.approx(0.2)


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str


@dataclass(frozen=True)
class FakeResult:
    rank: int
    score: float
    chunk: FakeChunk


class RankingFakeRetriever:
    def __init__(self, responses):
        self.responses = responses

    def search(self, query, *, top_k=5, filters=None):
        ids = self.responses.get(query, ())[:top_k]
        return tuple(
            FakeResult(
                rank=rank,
                score=1.0 / rank,
                chunk=FakeChunk(chunk_id),
            )
            for rank, chunk_id in enumerate(ids, start=1)
        )


def test_mrr_includes_misses_in_denominator():
    dataset = OfflineEvaluationDataset(
        name="day24-mrr-demo",
        description="Deterministic ranking example.",
        cases=(
            OfflineEvaluationCase("rank-1", "rank-1", ("c7",)),
            OfflineEvaluationCase("rank-2", "rank-2", ("c7",)),
            OfflineEvaluationCase("rank-3", "rank-3", ("c7",)),
            OfflineEvaluationCase("miss", "miss", ("c7",)),
        ),
    )

    retriever = RankingFakeRetriever(
        {
            "rank-1": ("c7", "x", "y"),
            "rank-2": ("x", "c7", "y"),
            "rank-3": ("x", "y", "c7"),
            "miss": ("x", "y", "z"),
        }
    )

    report = OfflineRetrievalEvaluator(
        retriever,
        k=3,
    ).evaluate_dataset(dataset)

    expected_mrr = (1.0 + 0.5 + (1 / 3) + 0.0) / 4

    assert report.hit_rate_at_k == pytest.approx(3 / 4)
    assert report.mean_reciprocal_rank == pytest.approx(expected_mrr)
