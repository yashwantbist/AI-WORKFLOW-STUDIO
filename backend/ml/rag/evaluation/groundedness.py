"""Deterministic claim-level groundedness for labelled RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence


WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value.strip()).casefold()


def _clean_ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError(f"{name} cannot contain empty IDs")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} must contain unique IDs")
    return cleaned


@dataclass(frozen=True)
class ClaimEvaluation:
    claim: str
    supported: bool
    evidence_ids: tuple[str, ...] = ()
    missing_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("claim cannot be empty")
        _clean_ids(self.evidence_ids, name="evidence_ids")
        _clean_ids(
            self.missing_evidence_ids,
            name="missing_evidence_ids",
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "ClaimEvaluation":
        if "claim" not in value:
            raise ValueError("claim mapping must contain 'claim'")
        if "supported" not in value:
            raise ValueError("claim mapping must contain 'supported'")

        evidence = value.get("evidence_ids", ())
        missing = value.get("missing_evidence_ids", ())

        if isinstance(evidence, str):
            raise ValueError(
                "evidence_ids must be a sequence, not a string"
            )
        if isinstance(missing, str):
            raise ValueError(
                "missing_evidence_ids must be a sequence, not a string"
            )

        return cls(
            claim=str(value["claim"]),
            supported=bool(value["supported"]),
            evidence_ids=tuple(str(item) for item in evidence),
            missing_evidence_ids=tuple(
                str(item)
                for item in missing
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "supported": self.supported,
            "evidence_ids": list(self.evidence_ids),
            "missing_evidence_ids": list(
                self.missing_evidence_ids
            ),
        }


@dataclass(frozen=True)
class GroundednessClaim:
    """Human-labelled claim and the evidence required to support it.

    v1 semantics deliberately require every ID in `supported_by` to be
    available for the claim to count as supported.
    """

    claim: str
    supported_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("groundedness claim cannot be empty")
        _clean_ids(
            self.supported_by,
            name="supported_by",
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "GroundednessClaim":
        raw_claim = value.get("claim", value.get("text"))
        if raw_claim is None:
            raise ValueError(
                "groundedness claim must contain 'claim' or 'text'"
            )

        raw_supported_by = value.get(
            "supported_by",
            value.get("evidence_ids", ()),
        )
        if isinstance(raw_supported_by, str):
            raise ValueError(
                "supported_by must be a sequence, not a string"
            )

        return cls(
            claim=str(raw_claim),
            supported_by=tuple(
                str(item)
                for item in raw_supported_by
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "supported_by": list(self.supported_by),
        }


@dataclass(frozen=True)
class GroundednessEvaluation:
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    groundedness_score: float
    fully_grounded: bool
    claims: tuple[ClaimEvaluation, ...]

    @property
    def partially_grounded(self) -> bool:
        return (
            self.supported_claims > 0
            and self.unsupported_claims > 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "groundedness_score": self.groundedness_score,
            "fully_grounded": self.fully_grounded,
            "partially_grounded": self.partially_grounded,
            "claims": [
                claim.to_dict()
                for claim in self.claims
            ],
        }


ClaimInput = ClaimEvaluation | Mapping[str, object]
GroundednessClaimInput = GroundednessClaim | Mapping[str, object]


def evaluate_groundedness(
    claims: Iterable[ClaimInput],
) -> GroundednessEvaluation:
    """Aggregate already-labelled claim support.

    This preserves the pre-Day-25 API for existing tests and callers.
    """

    normalized: list[ClaimEvaluation] = []

    for value in claims:
        if isinstance(value, ClaimEvaluation):
            normalized.append(value)
        elif isinstance(value, Mapping):
            normalized.append(
                ClaimEvaluation.from_mapping(value)
            )
        else:
            raise TypeError(
                "claims must contain ClaimEvaluation objects "
                "or mappings"
            )

    total = len(normalized)
    supported = sum(
        int(claim.supported)
        for claim in normalized
    )
    unsupported = total - supported

    return GroundednessEvaluation(
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        groundedness_score=(
            supported / total
            if total
            else 0.0
        ),
        fully_grounded=(
            total > 0
            and unsupported == 0
        ),
        claims=tuple(normalized),
    )


def evaluate_labelled_groundedness(
    answer: str,
    available_evidence_ids: Iterable[str],
    claims: Iterable[GroundednessClaimInput],
) -> GroundednessEvaluation:
    """Evaluate explicit answer claims against labelled evidence IDs.

    This is intentionally deterministic. It does not attempt semantic claim
    extraction or LLM-as-judge evaluation.

    Rules:
    - Only labelled claims whose text appears in the answer are evaluated.
    - A present claim is supported only when it has at least one labelled
      evidence ID and *all* labelled evidence IDs are available.
    - A labelled claim with no supporting evidence IDs is an explicit
      unsupported-claim fixture.
    - Missing evidence IDs are retained for diagnostics.
    - An empty answer therefore produces zero evaluated claims, score 0.0,
      and `fully_grounded=False`.
    """

    normalized_answer = _normalize_text(answer)
    available = set(
        _clean_ids(
            available_evidence_ids,
            name="available_evidence_ids",
        )
    )

    specs: list[GroundednessClaim] = []
    for value in claims:
        if isinstance(value, GroundednessClaim):
            specs.append(value)
        elif isinstance(value, Mapping):
            specs.append(
                GroundednessClaim.from_mapping(value)
            )
        else:
            raise TypeError(
                "claims must contain GroundednessClaim objects "
                "or mappings"
            )

    evaluated: list[ClaimEvaluation] = []

    if normalized_answer:
        for spec in specs:
            if _normalize_text(spec.claim) not in normalized_answer:
                continue

            required = set(spec.supported_by)
            matched = tuple(
                evidence_id
                for evidence_id in spec.supported_by
                if evidence_id in available
            )
            missing = tuple(
                evidence_id
                for evidence_id in spec.supported_by
                if evidence_id not in available
            )

            supported = (
                bool(required)
                and not missing
            )

            evaluated.append(
                ClaimEvaluation(
                    claim=spec.claim,
                    supported=supported,
                    evidence_ids=matched,
                    missing_evidence_ids=missing,
                )
            )

    return evaluate_groundedness(evaluated)
