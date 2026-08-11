from dataclasses import dataclass

import pytest

from backend.ml.rag.evaluated_pipeline import EvaluatedRAGPipeline
from backend.ml.rag.rag_pipeline import RAGPipeline


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    text: str
    document_id: str = "doc-1"
    document_title: str = "Policy Guide"
    source: str = "sample://policy"
    page_start: int = 1
    page_end: int = 1
    sections: tuple[str, ...] = ("Policies",)


@dataclass(frozen=True)
class FakeRetrieved:
    rank: int
    score: float
    chunk: FakeChunk


class FakeRetriever:
    def __init__(self, results):
        self.results = tuple(results)
        self.calls = []

    def search(self, query, *, top_k=5, filters=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
            }
        )
        return self.results[:top_k]


class FakeProvider:
    def __init__(self, answer="Refunds are available for 30 days [Source 1]."):
        self.answer = answer
        self.calls = []

    def generate(self, *, instructions, prompt):
        self.calls.append(
            {
                "instructions": instructions,
                "prompt": prompt,
            }
        )
        return self.answer


def make_results():
    return (
        FakeRetrieved(
            rank=1,
            score=0.91,
            chunk=FakeChunk(
                "c1",
                "Refunds are available for 30 days.",
            ),
        ),
        FakeRetrieved(
            rank=2,
            score=0.72,
            chunk=FakeChunk(
                "c2",
                "Shipping takes 3-5 business days.",
            ),
        ),
    )


def build_pipeline(results=None, answer=None):
    retriever = FakeRetriever(results if results is not None else make_results())
    provider = FakeProvider(
        answer=answer
        if answer is not None
        else "Refunds are available for 30 days [Source 1]."
    )
    rag_pipeline = RAGPipeline(
        retriever,
        provider,
        minimum_relevance_score=0.10,
    )
    evaluated = EvaluatedRAGPipeline(
        retriever,
        rag_pipeline,
    )
    return retriever, provider, evaluated


def test_unlabelled_request_returns_telemetry_without_fake_metrics():
    retriever, provider, pipeline = build_pipeline()

    result = pipeline.run("What is the refund policy?", k=2)

    assert result.answer.answer.startswith("Refunds are available")
    assert result.retrieval.chunk_ids == ("c1", "c2")
    assert result.retrieval.retrieved_count == 2
    assert result.evaluation.retrieval is None
    assert result.evaluation.groundedness is None
    assert result.to_dict()["evaluation"] is None
    assert len(retriever.calls) == 1
    assert len(provider.calls) == 1


def test_labelled_request_calculates_precision_and_recall():
    _, _, pipeline = build_pipeline()

    result = pipeline.run(
        "What is the refund policy?",
        k=2,
        relevant_ids={"c1", "c3"},
    )

    metrics = result.evaluation.retrieval
    assert metrics is not None
    assert metrics.precision_at_k == pytest.approx(0.5)
    assert metrics.recall_at_k == pytest.approx(0.5)
    assert metrics.relevant_retrieved == 1
    assert metrics.total_relevant == 2


def test_explicit_empty_relevant_set_is_evaluated_not_treated_as_missing():
    _, _, pipeline = build_pipeline()

    result = pipeline.run(
        "What is the refund policy?",
        k=2,
        relevant_ids=set(),
    )

    metrics = result.evaluation.retrieval
    assert metrics is not None
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.total_relevant == 0


def test_same_single_retrieval_reaches_generation_prompt():
    retriever, provider, pipeline = build_pipeline()

    pipeline.run("What is the refund policy?", k=2)

    assert len(retriever.calls) == 1
    prompt = provider.calls[0]["prompt"]
    assert "chunk_id: c1" in prompt
    assert "Refunds are available for 30 days." in prompt
    assert "chunk_id: c2" in prompt


def test_filters_and_k_are_forwarded_to_retriever():
    retriever, _, pipeline = build_pipeline()

    pipeline.run(
        "What is the refund policy?",
        k=1,
        filters={"document_id": "doc-1"},
    )

    assert retriever.calls == [
        {
            "query": "What is the refund policy?",
            "top_k": 1,
            "filters": {"document_id": "doc-1"},
        }
    ]


def test_empty_retrieval_returns_insufficient_context_without_generation():
    retriever, provider, pipeline = build_pipeline(results=())

    result = pipeline.run(
        "What is the refund policy?",
        k=3,
        relevant_ids={"c1"},
    )

    assert result.answer.insufficient_context is True
    assert result.answer.used_context_count == 0
    assert result.retrieval.retrieved_count == 0
    assert result.evaluation.retrieval is not None
    assert result.evaluation.retrieval.recall_at_k == 0.0
    assert len(retriever.calls) == 1
    assert provider.calls == []


def test_thresholded_out_chunk_is_visible_in_telemetry_but_not_context():
    results = (
        FakeRetrieved(
            rank=1,
            score=0.05,
            chunk=FakeChunk("c-low", "Low score evidence."),
        ),
    )
    _, provider, pipeline = build_pipeline(results=results)

    result = pipeline.run(
        "Question?",
        k=1,
        minimum_relevance_score=0.10,
    )

    assert result.retrieval.chunk_ids == ("c-low",)
    assert result.retrieval.retrieved_count == 1
    assert result.retrieval.used_context_count == 0
    assert result.answer.insufficient_context is True
    assert provider.calls == []


def test_claim_labels_add_groundedness_only_when_supplied():
    _, _, pipeline = build_pipeline()

    result = pipeline.run(
        "What is the refund policy?",
        k=2,
        claim_labels=[
            {
                "claim": "Refunds are available for 30 days.",
                "supported": True,
                "evidence_ids": ["c1"],
            },
            {
                "claim": "Refunds are instant.",
                "supported": False,
            },
        ],
    )

    groundedness = result.evaluation.groundedness
    assert groundedness is not None
    assert groundedness.groundedness_score == pytest.approx(0.5)
    assert groundedness.partially_grounded is True


def test_query_and_k_validation():
    _, _, pipeline = build_pipeline()

    with pytest.raises(ValueError):
        pipeline.run("   ")

    with pytest.raises(ValueError):
        pipeline.run("valid", k=0)


def test_result_serialization_keeps_optional_evaluation_separate():
    _, _, pipeline = build_pipeline()

    result = pipeline.run(
        "What is the refund policy?",
        k=2,
        relevant_ids={"c1"},
    ).to_dict()

    assert result["answer"]["answer"].startswith("Refunds")
    assert result["retrieval"]["chunk_ids"] == ["c1", "c2"]
    assert result["evaluation"]["retrieval"]["precision_at_k"] == pytest.approx(0.5)
    assert result["evaluation"]["groundedness"] is None
