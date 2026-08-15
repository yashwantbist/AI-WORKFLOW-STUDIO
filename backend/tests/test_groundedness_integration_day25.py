from dataclasses import dataclass
import json

from backend.ml.rag.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
)
from backend.ml.rag.evaluation.evaluator import RAGEvaluator
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


def make_retriever():
    return FakeRetriever(
        [
            FakeRetrieved(
                1,
                0.95,
                FakeChunk(
                    "refund-01",
                    "Refunds may be requested within 30 days.",
                ),
            )
        ]
    )


def test_grounded_case_is_attached_to_evaluation_and_summary():
    case = EvaluationCase(
        case_id="refund-supported",
        question="How long is the refund period?",
        expected_chunk_ids=("refund-01",),
        reference_answer="Customers may request refunds within 30 days.",
        groundedness_claims=(
            GroundednessClaim(
                "Customers may request refunds within 30 days.",
                ("refund-01",),
            ),
        ),
    )

    evaluator = RAGEvaluator(
        make_retriever(),
        FakePipeline(
            FakeAnswer(
                "Customers may request refunds within 30 days.",
                (FakeSource("refund-01"),),
            )
        ),
        top_k=1,
    )

    run = evaluator.evaluate(
        EvaluationDataset(
            "groundedness-test",
            "tiny",
            (case,),
        )
    )

    result = run.cases[0]
    assert result.groundedness is not None
    assert result.groundedness.groundedness_score == 1.0
    assert result.status == "PASS"
    assert run.summary["groundedness_case_count"] == 1
    assert run.summary["mean_groundedness_score"] == 1.0


def test_unsupported_claim_becomes_groundedness_failure():
    case = EvaluationCase(
        case_id="refund-unsupported",
        question="How long is the refund period?",
        expected_chunk_ids=("refund-01",),
        reference_answer="Customers may request refunds within 90 days.",
        groundedness_claims=(
            GroundednessClaim(
                "Customers may request refunds within 90 days.",
                (),
            ),
        ),
    )

    evaluator = RAGEvaluator(
        make_retriever(),
        FakePipeline(
            FakeAnswer(
                "Customers may request refunds within 90 days.",
                (FakeSource("refund-01"),),
            )
        ),
        top_k=1,
    )

    result = evaluator.evaluate_case(case)

    assert result.groundedness is not None
    assert result.groundedness.groundedness_score == 0.0
    assert result.failure_stage == "groundedness"
    assert result.status == "FAIL"


def test_missing_evidence_is_preserved_for_debugging():
    case = EvaluationCase(
        case_id="missing-evidence",
        question="How long is the refund period?",
        expected_chunk_ids=("refund-01",),
        reference_answer="Customers may request refunds within 30 days.",
        groundedness_claims=(
            GroundednessClaim(
                "Customers may request refunds within 30 days.",
                ("refund-01",),
            ),
        ),
    )

    evaluator = RAGEvaluator(
        make_retriever(),
        FakePipeline(
            FakeAnswer(
                "Customers may request refunds within 30 days.",
                (),
            )
        ),
        top_k=1,
    )

    result = evaluator.evaluate_case(case)

    # Augmentation fails first because the pipeline supplied no source.
    assert result.failure_stage == "augmentation"
    assert result.groundedness is not None
    claim = result.groundedness.claims[0]
    assert claim.supported is False
    assert claim.missing_evidence_ids == ("refund-01",)


def test_unlabelled_case_does_not_fabricate_groundedness():
    case = EvaluationCase(
        case_id="unlabelled",
        question="How long is the refund period?",
        expected_chunk_ids=("refund-01",),
        reference_answer="Customers may request refunds within 30 days.",
    )

    evaluator = RAGEvaluator(
        make_retriever(),
        FakePipeline(
            FakeAnswer(
                "Customers may request refunds within 30 days.",
                (FakeSource("refund-01"),),
            )
        ),
        top_k=1,
    )

    run = evaluator.evaluate(
        EvaluationDataset(
            "unlabelled-test",
            "tiny",
            (case,),
        )
    )

    assert run.cases[0].groundedness is None
    assert run.summary["groundedness_case_count"] == 0
    assert run.summary["mean_groundedness_score"] is None


def test_report_and_json_include_claim_evidence(tmp_path):
    case = EvaluationCase(
        case_id="report-case",
        question="How long is the refund period?",
        expected_chunk_ids=("refund-01",),
        reference_answer="Customers may request refunds within 30 days.",
        groundedness_claims=(
            GroundednessClaim(
                "Customers may request refunds within 30 days.",
                ("refund-01",),
            ),
        ),
    )

    run = RAGEvaluator(
        make_retriever(),
        FakePipeline(
            FakeAnswer(
                "Customers may request refunds within 30 days.",
                (FakeSource("refund-01"),),
            )
        ),
        top_k=1,
    ).evaluate(
        EvaluationDataset(
            "report-test",
            "tiny",
            (case,),
        )
    )

    text = format_terminal_report(run)
    assert "Groundedness score: 1.000" in text
    assert "[SUPPORTED] Customers may request refunds within 30 days." in text
    assert "Evidence IDs: refund-01" in text

    output = save_json_report(
        run,
        tmp_path / "report.json",
    )
    payload = json.loads(
        output.read_text(encoding="utf-8")
    )
    assert payload["summary"]["mean_groundedness_score"] == 1.0
    assert (
        payload["cases"][0]["groundedness"]["claims"][0]["evidence_ids"]
        == ["refund-01"]
    )
