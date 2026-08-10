import pytest

from backend.ml.rag.evaluation.groundedness import (
    ClaimEvaluation,
    evaluate_groundedness,
)


def test_fully_grounded_answer():
    result = evaluate_groundedness([
        {"claim": "Pro costs $20/month.", "supported": True},
        {"claim": "Pro includes 100 GB storage.", "supported": True},
    ])
    assert result.groundedness_score == 1.0
    assert result.fully_grounded is True


def test_partially_grounded_answer():
    result = evaluate_groundedness([
        {
            "claim": "Pro costs $20/month.",
            "supported": True,
            "evidence_ids": ["pricing-1"],
        },
        {
            "claim": "Pro includes 100 GB storage.",
            "supported": True,
            "evidence_ids": ["pricing-2"],
        },
        {
            "claim": "Pro is the most popular plan.",
            "supported": False,
        },
    ])
    assert result.total_claims == 3
    assert result.supported_claims == 2
    assert result.unsupported_claims == 1
    assert result.groundedness_score == pytest.approx(2 / 3)
    assert result.partially_grounded is True
    assert result.fully_grounded is False


def test_all_unsupported():
    result = evaluate_groundedness([
        {"claim": "Unsupported one.", "supported": False},
        {"claim": "Unsupported two.", "supported": False},
    ])
    assert result.groundedness_score == 0.0
    assert result.fully_grounded is False


def test_empty_claims_are_not_called_fully_grounded():
    result = evaluate_groundedness([])
    assert result.groundedness_score == 0.0
    assert result.fully_grounded is False


def test_claim_dataclass_and_evidence_ids():
    result = evaluate_groundedness([
        ClaimEvaluation(
            claim="Supported claim.",
            supported=True,
            evidence_ids=("chunk-1",),
        )
    ])
    assert result.claims[0].evidence_ids == ("chunk-1",)


def test_missing_supported_label_is_rejected():
    with pytest.raises(ValueError):
        evaluate_groundedness([{"claim": "Missing support label"}])


def test_empty_claim_text_is_rejected():
    with pytest.raises(ValueError):
        evaluate_groundedness([{"claim": " ", "supported": True}])


def test_result_is_json_friendly():
    payload = evaluate_groundedness([
        {"claim": "Supported.", "supported": True, "evidence_ids": ["c1"]},
        {"claim": "Unsupported.", "supported": False},
    ]).to_dict()
    assert payload["groundedness_score"] == pytest.approx(0.5)
    assert payload["claims"][0]["evidence_ids"] == ["c1"]
