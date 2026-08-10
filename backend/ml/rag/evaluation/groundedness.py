"""Deterministic claim-level groundedness for RAG Evaluation Module v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ClaimEvaluation:
    claim: str
    supported: bool
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("claim cannot be empty")
        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids cannot contain empty IDs")

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
        if isinstance(evidence, str):
            raise ValueError("evidence_ids must be a sequence, not a string")

        return cls(
            claim=str(value["claim"]),
            supported=bool(value["supported"]),
            evidence_ids=tuple(str(item) for item in evidence),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "supported": self.supported,
            "evidence_ids": list(self.evidence_ids),
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
        return self.supported_claims > 0 and self.unsupported_claims > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "groundedness_score": self.groundedness_score,
            "fully_grounded": self.fully_grounded,
            "partially_grounded": self.partially_grounded,
            "claims": [claim.to_dict() for claim in self.claims],
        }


ClaimInput = ClaimEvaluation | Mapping[str, object]


def evaluate_groundedness(
    claims: Iterable[ClaimInput],
) -> GroundednessEvaluation:
    normalized: list[ClaimEvaluation] = []

    for value in claims:
        if isinstance(value, ClaimEvaluation):
            normalized.append(value)
        elif isinstance(value, Mapping):
            normalized.append(ClaimEvaluation.from_mapping(value))
        else:
            raise TypeError(
                "claims must contain ClaimEvaluation objects or mappings"
            )

    total = len(normalized)
    supported = sum(claim.supported for claim in normalized)
    unsupported = total - supported

    return GroundednessEvaluation(
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        groundedness_score=(supported / total if total else 0.0),
        fully_grounded=(total > 0 and unsupported == 0),
        claims=tuple(normalized),
    )
