from dataclasses import dataclass

from backend.ml.rag.rag_pipeline import RAGPipeline


@dataclass(frozen=True)
class Chunk:
    chunk_id: str = "c1"
    text: str = "Evidence text."
    document_id: str = "doc"
    document_title: str = "Doc"
    source: str = "sample://doc"
    page_start: int = 1
    page_end: int = 1
    sections: tuple[str, ...] = ("Section",)


@dataclass(frozen=True)
class Result:
    rank: int = 1
    score: float = 0.9
    chunk: Chunk = Chunk()


class Retriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, *, top_k=5, filters=None):
        self.calls += 1
        return (Result(),)


class Provider:
    def __init__(self):
        self.calls = 0

    def generate(self, *, instructions, prompt):
        self.calls += 1
        return "Grounded answer [Source 1]."


def test_original_answer_api_still_works():
    retriever = Retriever()
    provider = Provider()
    pipeline = RAGPipeline(retriever, provider)

    answer = pipeline.answer("Question?", top_k=1)

    assert answer.answer.startswith("Grounded answer")
    assert answer.retrieved_count == 1
    assert answer.used_context_count == 1
    assert retriever.calls == 1
    assert provider.calls == 1


def test_answer_from_retrieved_does_not_search_again():
    retriever = Retriever()
    provider = Provider()
    pipeline = RAGPipeline(retriever, provider)

    answer = pipeline.answer_from_retrieved(
        "Question?",
        (Result(),),
    )

    assert answer.used_context_count == 1
    assert retriever.calls == 0
    assert provider.calls == 1
