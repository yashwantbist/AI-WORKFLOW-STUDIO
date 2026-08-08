"""Tests for retrieval -> augmentation -> generation orchestration."""

from dataclasses import dataclass

import pytest

from backend.ml.rag.llm_provider import OpenAIResponsesProvider, RecordingDemoProvider
from backend.ml.rag.rag_pipeline import INSUFFICIENT_CONTEXT_ANSWER, RAGPipeline


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    text: str
    document_id: str = "transformers"
    document_title: str = "Transformers Guide"
    source: str = "transformers.pdf"
    page_start: int = 12
    page_end: int = 12
    sections: tuple[str, ...] = ("Self-Attention",)


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
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self.results[:top_k]


def make_result(score=0.9, *, chunk_id="chunk-1", text="Self-attention works."):
    return FakeRetrieved(
        rank=1,
        score=score,
        chunk=FakeChunk(chunk_id=chunk_id, text=text),
    )


def test_pipeline_passes_top_k_and_filters_to_retriever() -> None:
    retriever = FakeRetriever([make_result()])
    provider = RecordingDemoProvider(answer_text="Grounded answer.")
    pipeline = RAGPipeline(retriever, provider)

    pipeline.answer(
        "How does attention work?",
        top_k=3,
        filters={"page": 12},
    )

    assert retriever.calls == [
        {"query": "How does attention work?", "top_k": 3, "filters": {"page": 12}}
    ]


def test_pipeline_propagates_source_metadata() -> None:
    retriever = FakeRetriever(
        [
            make_result(
                score=0.92,
                chunk_id="chunk-24",
                text="Queries and keys produce attention weights.",
            )
        ]
    )
    provider = RecordingDemoProvider(
        answer_text="Attention uses queries and keys [Source 1]."
    )
    pipeline = RAGPipeline(retriever, provider)

    result = pipeline.answer("Explain attention.", top_k=5)

    assert result.grounded is True
    assert result.insufficient_context is False
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "chunk-24"
    assert result.sources[0].page_start == 12
    assert result.sources[0].score == pytest.approx(0.92)


def test_low_score_results_trigger_insufficient_context_without_llm_call() -> None:
    retriever = FakeRetriever([make_result(score=0.04)])
    provider = RecordingDemoProvider(answer_text="This should never be generated.")
    pipeline = RAGPipeline(
        retriever,
        provider,
        minimum_relevance_score=0.20,
    )

    result = pipeline.answer("What is quantum gravity?")

    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.insufficient_context is True
    assert result.grounded is False
    assert result.sources == ()
    assert result.retrieved_count == 1
    assert result.used_context_count == 0
    assert provider.last_prompt is None


def test_relevant_results_reach_generation_prompt() -> None:
    retriever = FakeRetriever(
        [make_result(text="Self-attention allows tokens to examine other tokens.")]
    )
    provider = RecordingDemoProvider(answer_text="Generated grounded answer.")
    pipeline = RAGPipeline(retriever, provider)

    result = pipeline.answer("How does self-attention work?")

    assert result.answer == "Generated grounded answer."
    assert provider.last_prompt is not None
    assert "Self-attention allows tokens to examine other tokens." in provider.last_prompt
    assert "How does self-attention work?" in provider.last_prompt


def test_pipeline_rejects_invalid_input() -> None:
    pipeline = RAGPipeline(FakeRetriever([make_result()]), RecordingDemoProvider())

    with pytest.raises(ValueError):
        pipeline.answer("")
    with pytest.raises(ValueError):
        pipeline.answer("question", top_k=0)
    with pytest.raises(ValueError):
        pipeline.answer("question", minimum_relevance_score=2.0)


def test_result_is_api_friendly() -> None:
    pipeline = RAGPipeline(
        FakeRetriever([make_result()]),
        RecordingDemoProvider(answer_text="Answer [Source 1]."),
    )

    payload = pipeline.answer("Question?").to_dict()

    assert payload["answer"] == "Answer [Source 1]."
    assert payload["sources"][0]["document_id"] == "transformers"
    assert payload["sources"][0]["sections"] == ["Self-Attention"]
    assert payload["grounded"] is True


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": "Provider answer"})()


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_openai_provider_uses_responses_interface() -> None:
    client = FakeOpenAIClient()
    provider = OpenAIResponsesProvider("test-model", client=client)

    answer = provider.generate(
        instructions="Use context only.",
        prompt="Context and question.",
    )

    assert answer == "Provider answer"
    assert client.responses.calls == [
        {
            "model": "test-model",
            "instructions": "Use context only.",
            "input": "Context and question.",
        }
    ]
