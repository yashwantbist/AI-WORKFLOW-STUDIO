import pytest

from backend.ml.rag.evaluation.groundedness import (
    GroundednessClaim,
    evaluate_labelled_groundedness,
)


def test_fully_supported_answer():
    result = evaluate_labelled_groundedness(
        "Customers may request refunds within 30 days.",
        ["refund-01"],
        [
            {
                "text": "Customers may request refunds within 30 days.",
                "supported_by": ["refund-01"],
            }
        ],
    )

    assert result.groundedness_score == 1.0
    assert result.fully_grounded is True
    assert result.supported_claims == 1
    assert result.claims[0].evidence_ids == ("refund-01",)


def test_partially_supported_answer():
    answer = (
        "Standard shipping takes 3–5 days. "
        "International shipping is free."
    )

    result = evaluate_labelled_groundedness(
        answer,
        ["shipping-01"],
        [
            GroundednessClaim(
                "Standard shipping takes 3–5 days.",
                ("shipping-01",),
            ),
            GroundednessClaim(
                "International shipping is free.",
                (),
            ),
        ],
    )

    assert result.total_claims == 2
    assert result.supported_claims == 1
    assert result.unsupported_claims == 1
    assert result.groundedness_score == pytest.approx(0.5)
    assert result.partially_grounded is True
    assert result.fully_grounded is False


def test_completely_unsupported_when_required_evidence_missing():
    result = evaluate_labelled_groundedness(
        "Customers may request refunds within 30 days.",
        [],
        [
            GroundednessClaim(
                "Customers may request refunds within 30 days.",
                ("refund-01",),
            )
        ],
    )

    assert result.groundedness_score == 0.0
    assert result.fully_grounded is False
    assert result.claims[0].supported is False
    assert result.claims[0].evidence_ids == ()
    assert result.claims[0].missing_evidence_ids == ("refund-01",)


def test_multiple_required_evidence_chunks_must_all_be_available():
    claim = GroundednessClaim(
        "The policy has two eligibility requirements.",
        ("policy-01", "policy-02"),
    )

    partial = evaluate_labelled_groundedness(
        claim.claim,
        ["policy-01"],
        [claim],
    )
    complete = evaluate_labelled_groundedness(
        claim.claim,
        ["policy-01", "policy-02"],
        [claim],
    )

    assert partial.groundedness_score == 0.0
    assert partial.claims[0].evidence_ids == ("policy-01",)
    assert partial.claims[0].missing_evidence_ids == ("policy-02",)

    assert complete.groundedness_score == 1.0
    assert complete.claims[0].evidence_ids == (
        "policy-01",
        "policy-02",
    )


def test_empty_answer_has_zero_evaluated_claims():
    result = evaluate_labelled_groundedness(
        "",
        ["refund-01"],
        [
            GroundednessClaim(
                "Refunds are allowed within 30 days.",
                ("refund-01",),
            )
        ],
    )

    assert result.total_claims == 0
    assert result.groundedness_score == 0.0
    assert result.fully_grounded is False


def test_zero_labelled_claims_is_explicit_zero_case():
    result = evaluate_labelled_groundedness(
        "An answer.",
        ["c1"],
        [],
    )

    assert result.total_claims == 0
    assert result.supported_claims == 0
    assert result.groundedness_score == 0.0
    assert result.fully_grounded is False


def test_labelled_claim_not_present_in_answer_is_not_evaluated():
    result = evaluate_labelled_groundedness(
        "The answer discusses something else.",
        ["refund-01"],
        [
            GroundednessClaim(
                "Refunds are allowed within 30 days.",
                ("refund-01",),
            )
        ],
    )

    assert result.total_claims == 0


def test_duplicate_evidence_ids_are_rejected():
    with pytest.raises(ValueError, match="supported_by must contain unique IDs"):
        GroundednessClaim(
            "A claim.",
            ("c1", "c1"),
        )
