"""Schema and loader for labelled offline retrieval-evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class OfflineEvaluationCase:
    case_id: str
    query: str
    relevant_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        if any(not value.strip() for value in self.relevant_ids):
            raise ValueError("relevant_ids cannot contain empty IDs")
        if len(self.relevant_ids) != len(set(self.relevant_ids)):
            raise ValueError("relevant_ids must be unique")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("tags cannot contain empty values")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        index: int,
    ) -> "OfflineEvaluationCase":
        required = {"id", "query", "relevant_ids"}
        missing = required - set(value)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"case at index {index} is missing required field(s): {names}"
            )

        relevant_ids = value["relevant_ids"]
        if not isinstance(relevant_ids, list):
            raise ValueError(
                f"case {value.get('id', index)!r}: relevant_ids must be a list"
            )
        if not all(isinstance(item, str) for item in relevant_ids):
            raise ValueError(
                f"case {value.get('id', index)!r}: "
                "relevant_ids must contain only strings"
            )

        tags = value.get("tags", [])
        if not isinstance(tags, list) or not all(
            isinstance(item, str) for item in tags
        ):
            raise ValueError(
                f"case {value.get('id', index)!r}: tags must be a list of strings"
            )

        return cls(
            case_id=str(value["id"]),
            query=str(value["query"]),
            relevant_ids=tuple(relevant_ids),
            tags=tuple(tags),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "query": self.query,
            "relevant_ids": list(self.relevant_ids),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class OfflineEvaluationDataset:
    name: str
    description: str
    cases: tuple[OfflineEvaluationCase, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset name cannot be empty")
        if not self.cases:
            raise ValueError("dataset must contain at least one evaluation case")

        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case IDs must be unique")

    @classmethod
    def load(cls, path: str | Path) -> "OfflineEvaluationDataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"evaluation dataset not found: {path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in evaluation dataset: {error.msg}"
            ) from error

        if not isinstance(payload, dict):
            raise ValueError("evaluation dataset root must be a JSON object")

        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("evaluation dataset must contain a cases list")

        cases = []
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                raise ValueError(
                    f"case at index {index} must be a JSON object"
                )
            cases.append(
                OfflineEvaluationCase.from_dict(raw_case, index=index)
            )

        return cls(
            name=str(payload.get("name", path.stem)),
            description=str(payload.get("description", "")),
            cases=tuple(cases),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "cases": [case.to_dict() for case in self.cases],
        }
