"""Load and validate a hand-written RAG evaluation dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .groundedness import GroundednessClaim


def _validate_phrases(
    values: tuple[str, ...],
    *,
    name: str,
) -> None:
    cleaned = tuple(value.strip() for value in values)

    if any(not value for value in cleaned):
        raise ValueError(f"{name} cannot contain empty phrases")

    normalized = tuple(value.casefold() for value in cleaned)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique phrases")


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_chunk_ids: tuple[str, ...]
    reference_answer: str
    tags: tuple[str, ...] = ()
    groundedness_claims: tuple[GroundednessClaim, ...] = ()
    answer_relevance_phrases: tuple[str, ...] = ()
    correctness_required_phrases: tuple[str, ...] = ()
    correctness_forbidden_phrases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not self.question.strip():
            raise ValueError("question cannot be empty")
        if not self.expected_chunk_ids:
            raise ValueError(
                "expected_chunk_ids cannot be empty"
            )
        if not self.reference_answer.strip():
            raise ValueError(
                "reference_answer cannot be empty"
            )
        if (
            len(self.expected_chunk_ids)
            != len(set(self.expected_chunk_ids))
        ):
            raise ValueError(
                "expected_chunk_ids must be unique"
            )

        _validate_phrases(
            self.answer_relevance_phrases,
            name="answer_relevance_phrases",
        )
        _validate_phrases(
            self.correctness_required_phrases,
            name="correctness_required_phrases",
        )
        _validate_phrases(
            self.correctness_forbidden_phrases,
            name="correctness_forbidden_phrases",
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "EvaluationCase":
        raw_claims = value.get(
            "groundedness_claims",
            value.get("claims", ()),
        )
        if isinstance(raw_claims, (str, bytes)):
            raise ValueError(
                "groundedness_claims must be a sequence"
            )

        return cls(
            case_id=str(value["case_id"]),
            question=str(value["question"]),
            expected_chunk_ids=tuple(
                str(chunk_id)
                for chunk_id in value["expected_chunk_ids"]
            ),
            reference_answer=str(
                value["reference_answer"]
            ),
            tags=tuple(
                str(tag)
                for tag in value.get("tags", ())
            ),
            groundedness_claims=tuple(
                GroundednessClaim.from_mapping(claim)
                for claim in raw_claims
            ),
            answer_relevance_phrases=tuple(
                str(phrase)
                for phrase in value.get(
                    "answer_relevance_phrases",
                    (),
                )
            ),
            correctness_required_phrases=tuple(
                str(phrase)
                for phrase in value.get(
                    "correctness_required_phrases",
                    (),
                )
            ),
            correctness_forbidden_phrases=tuple(
                str(phrase)
                for phrase in value.get(
                    "correctness_forbidden_phrases",
                    (),
                )
            ),
        )


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    description: str
    cases: tuple[EvaluationCase, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "dataset name cannot be empty"
            )
        if not self.cases:
            raise ValueError(
                "dataset must contain at least one case"
            )

        ids = [
            case.case_id
            for case in self.cases
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "case IDs must be unique"
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "EvaluationDataset":
        path = Path(path)

        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        return cls(
            name=str(
                payload.get("name", path.stem)
            ),
            description=str(
                payload.get("description", "")
            ),
            cases=tuple(
                EvaluationCase.from_dict(case)
                for case in payload["cases"]
            ),
        )
