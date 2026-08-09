from dataclasses import dataclass
import json

from backend.ml.rag.evaluation.answer_metrics import (
    context_token_support,
    token_precision_recall_f1,
)
from backend.ml.rag.evaluation.dataset import EvaluationCase, EvaluationDataset
from backend.ml.rag.evaluation.evaluator import RAGEvaluator
from backend.ml.rag.evaluation.report import format_terminal_report, save_json_report


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class FakeRetrieved:
    rank: int
    score: float
    chunk: FakeChunk


@dataclass(frozen=True)
class FakeSource:
    chunk_id: str


@dataclass(frozen=True)
class FakeAnswer:
    answer: str
    sources: tuple[FakeSource, ...]
    insufficient_context: bool = False


class FakeRetriever:
    def __init__(self, results):
        self.results = tuple(results)

    def search(self, query, *, top_k=5, filters=None):
        return self.results[:top_k]


class FakePipeline:
    def __init__(self, result):
        self.result = result

    def answer(self, question, *, top_k=5, filters=None, minimum_relevance_score=None):
        return self.result


def case():
    return EvaluationCase(
        case_id="attention",
        question="What does attention do?",
        expected_chunk_ids=("expected",),
        reference_answer="Attention models relationships between tokens.",
    )


def test_identical_reference_f1():
    p, r, f1 = token_precision_recall_f1(
        "Attention models relationships between tokens.",
        "Attention models relationships between tokens.",
    )
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_context_support():
    assert context_token_support(
        "Attention models token relationships.",
        ["Attention models relationships between tokens."],
    ) == 1.0


def test_retrieval_failure():
    evaluator = RAGEvaluator(
        FakeRetriever([FakeRetrieved(1, 0.9, FakeChunk("wrong", "Wrong."))]),
        FakePipeline(FakeAnswer("Correct sounding.", (FakeSource("wrong"),))),
        top_k=1,
    )
    result = evaluator.evaluate_case(case())
    assert result.failure_stage == "retrieval"
    assert result.status == "FAIL"


def test_augmentation_failure():
    evaluator = RAGEvaluator(
        FakeRetriever([FakeRetrieved(1, 0.9, FakeChunk("expected", "Attention models relationships between tokens."))]),
        FakePipeline(FakeAnswer("Insufficient.", (), True)),
        top_k=1,
    )
    result = evaluator.evaluate_case(case())
    assert result.failure_stage == "augmentation"


def test_generation_review():
    evaluator = RAGEvaluator(
        FakeRetriever([FakeRetrieved(1, 0.9, FakeChunk("expected", "Attention models relationships between tokens."))]),
        FakePipeline(FakeAnswer("Unrelated response.", (FakeSource("expected"),))),
        top_k=1,
        answer_f1_review_threshold=0.5,
    )
    result = evaluator.evaluate_case(case())
    assert result.failure_stage == "generation_proxy"
    assert result.status == "REVIEW"


def test_success_and_report(tmp_path):
    evaluator = RAGEvaluator(
        FakeRetriever([FakeRetrieved(1, 0.9, FakeChunk("expected", "Attention models relationships between tokens."))]),
        FakePipeline(FakeAnswer("Attention models relationships between tokens [Source 1].", (FakeSource("expected"),))),
        top_k=1,
    )
    run = evaluator.evaluate(EvaluationDataset("test", "tiny", (case(),)))
    assert run.cases[0].status == "PASS"
    assert run.cases[0].answer_metrics.citation_count == 1
    path = save_json_report(run, tmp_path/"report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["retrieval_hit_rate"] == 1.0
    assert "RAG EVALUATION: test" in format_terminal_report(run)
