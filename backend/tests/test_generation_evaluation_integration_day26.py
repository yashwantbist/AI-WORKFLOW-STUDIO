from dataclasses import dataclass
import json

from backend.ml.rag.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
)
from backend.ml.rag.evaluation.evaluator import RAGEvaluator
from backend.ml.rag.evaluation.generation_quality import FailureCategory
from backend.ml.rag.evaluation.groundedness import GroundednessClaim
from backend.ml.rag.evaluation.report import (
    format_terminal_report,
    save_json_report,
)


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

    def answer(
        self,
        question,
        *,
        top_k=5,
        filters=None,
        minimum_relevance_score=None,
    ):
        return self.result


def good_retriever():
    return FakeRetriever(
        [
            FakeRetrieved(
                1,
                0.95,
                FakeChunk(
                    "refund-policy",
                    "Refunds are available within 30 days.",
                ),
            )
        ]
    )


def labelled_case(**overrides):
    values = {
        "case_id": "refund-001",
        "question": "What is the refund period?",
        "expected_chunk_ids": ("refund-policy",),
        "reference_answer": (
            "Refund requests are accepted within 30 days."
        ),
        "groundedness_claims": (
            GroundednessClaim(
                "Refund requests are accepted within 30 days.",
                ("refund-policy",),
            ),
        ),
        "answer_relevance_phrases": ("30 days",),
        "correctness_required_phrases": ("30 days",),
        "correctness_forbidden_phrases": ("60 days",),
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_good_retrieval_and_good_answer_is_success():
    result = RAGEvaluator(
        good_retriever(),
        FakePipeline(
            FakeAnswer(
                "Refund requests are accepted within 30 days.",
                (FakeSource("refund-policy"),),
            )
        ),
        top_k=1,
    ).evaluate_case(labelled_case())

    generation = result.generation_evaluation

    assert generation.groundedness == 1.0
    assert generation.answer_relevance == 1.0
    assert generation.correctness == 1.0
    assert generation.failure_categories == (
        FailureCategory.SUCCESS,
    )
    assert result.status == "PASS"


def test_good_retrieval_but_irrelevant_answer_is_classified():
    case = labelled_case(
        groundedness_claims=(),
        correctness_required_phrases=(),
        correctness_forbidden_phrases=(),
    )

    result = RAGEvaluator(
        good_retriever(),
        FakePipeline(
            FakeAnswer(
                "Our company offers refunds.",
                (FakeSource("refund-policy"),),
            )
        ),
        top_k=1,
    ).evaluate_case(case)

    assert result.generation_evaluation.answer_relevance == 0.0
    assert (
        FailureCategory.IRRELEVANT_ANSWER
        in result.failure_categories
    )
    assert result.failure_stage == "relevance"
    assert result.status == "FAIL"


def test_good_retrieval_but_wrong_answer_is_incorrect():
    case = labelled_case(
        groundedness_claims=(),
        answer_relevance_phrases=(),
    )

    result = RAGEvaluator(
        good_retriever(),
        FakePipeline(
            FakeAnswer(
                "Refund requests are accepted within 60 days.",
                (FakeSource("refund-policy"),),
            )
        ),
        top_k=1,
    ).evaluate_case(case)

    assert result.generation_evaluation.correctness == 0.0
    assert (
        FailureCategory.INCORRECT_ANSWER
        in result.failure_categories
    )
    assert result.failure_stage == "correctness"
    assert result.status == "FAIL"


def test_unsupported_irrelevant_and_incorrect_can_all_be_reported():
    case = labelled_case(
        groundedness_claims=(
            GroundednessClaim(
                "Refund requests are accepted within 60 days.",
                (),
            ),
        ),
    )

    result = RAGEvaluator(
        good_retriever(),
        FakePipeline(
            FakeAnswer(
                "Refund requests are accepted within 60 days.",
                (FakeSource("refund-policy"),),
            )
        ),
        top_k=1,
    ).evaluate_case(case)

    assert result.failure_categories == (
        FailureCategory.UNGROUNDED_GENERATION,
        FailureCategory.IRRELEVANT_ANSWER,
        FailureCategory.INCORRECT_ANSWER,
    )
    assert result.failure_stage == "groundedness"


def test_bad_retrieval_and_wrong_answer_preserve_multiple_failures():
    retriever = FakeRetriever(
        [
            FakeRetrieved(
                1,
                0.90,
                FakeChunk("wrong", "Unrelated context."),
            )
        ]
    )

    case = labelled_case(
        groundedness_claims=(),
        answer_relevance_phrases=(),
    )

    result = RAGEvaluator(
        retriever,
        FakePipeline(
            FakeAnswer(
                "Refund requests are accepted within 60 days.",
                (FakeSource("wrong"),),
            )
        ),
        top_k=1,
    ).evaluate_case(case)

    assert (
        FailureCategory.RETRIEVAL_FAILURE
        in result.failure_categories
    )
    assert (
        FailureCategory.INCORRECT_ANSWER
        in result.failure_categories
    )
    assert result.failure_stage == "retrieval"


def test_unlabelled_dimensions_stay_none_in_summary():
    case = EvaluationCase(
        case_id="unlabelled",
        question="What is the refund period?",
        expected_chunk_ids=("refund-policy",),
        reference_answer=(
            "Refund requests are accepted within 30 days."
        ),
    )

    run = RAGEvaluator(
        good_retriever(),
        FakePipeline(
            FakeAnswer(
                "Refund requests are accepted within 30 days.",
                (FakeSource("refund-policy"),),
            )
        ),
        top_k=1,
    ).evaluate(
        EvaluationDataset("unlabelled", "tiny", (case,))
    )

    assert run.summary["mean_answer_relevance"] is None
    assert run.summary["mean_correctness"] is None
    assert run.summary["answer_relevance_case_count"] == 0
    assert run.summary["correctness_case_count"] == 0


def test_report_and_json_separate_retrieval_and_generation(tmp_path):
    run = RAGEvaluator(
        good_retriever(),
        FakePipeline(
            FakeAnswer(
                "Refund requests are accepted within 30 days.",
                (FakeSource("refund-policy"),),
            )
        ),
        top_k=1,
    ).evaluate(
        EvaluationDataset(
            "day26-report",
            "tiny",
            (labelled_case(),),
        )
    )

    text = format_terminal_report(run)

    assert "RETRIEVAL EVALUATION" in text
    assert "GENERATION EVALUATION" in text
    assert "Answer relevance: 1.000" in text
    assert "Correctness: 1.000" in text
    assert "Failure categories: SUCCESS" in text

    output = save_json_report(
        run,
        tmp_path / "report.json",
    )
    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    generation = payload["cases"][0]["generation_evaluation"]
    assert generation["answer_relevance"] == 1.0
    assert generation["correctness"] == 1.0
    assert generation["failure_categories"] == ["SUCCESS"]
