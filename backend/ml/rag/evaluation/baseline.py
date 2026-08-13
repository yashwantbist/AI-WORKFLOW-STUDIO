"""Save and compare measured retrieval-evaluation baselines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .offline_runner import EvaluationReport


class RegressionStatus(str, Enum):
    PASS = "pass"
    REGRESSION = "regression"


@dataclass(frozen=True)
class RetrievalBaseline:
    schema_version: int
    dataset_name: str
    k: int
    total_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float

    @classmethod
    def from_report(
        cls,
        report: EvaluationReport,
    ) -> "RetrievalBaseline":
        return cls(
            schema_version=1,
            dataset_name=report.dataset_name,
            k=report.k,
            total_queries=report.total_queries,
            mean_precision_at_k=report.mean_precision_at_k,
            mean_recall_at_k=report.mean_recall_at_k,
            hit_rate_at_k=report.hit_rate_at_k,
            mean_reciprocal_rank=report.mean_reciprocal_rank,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalBaseline":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"baseline not found: {path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in baseline: {error.msg}"
            ) from error

        required = {
            "schema_version",
            "dataset_name",
            "k",
            "total_queries",
            "mean_precision_at_k",
            "mean_recall_at_k",
            "hit_rate_at_k",
            "mean_reciprocal_rank",
        }
        missing = required - set(payload)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"baseline is missing required field(s): {names}"
            )

        baseline = cls(
            schema_version=int(payload["schema_version"]),
            dataset_name=str(payload["dataset_name"]),
            k=int(payload["k"]),
            total_queries=int(payload["total_queries"]),
            mean_precision_at_k=float(payload["mean_precision_at_k"]),
            mean_recall_at_k=float(payload["mean_recall_at_k"]),
            hit_rate_at_k=float(payload["hit_rate_at_k"]),
            mean_reciprocal_rank=float(payload["mean_reciprocal_rank"]),
        )

        if baseline.schema_version != 1:
            raise ValueError(
                f"unsupported baseline schema_version: "
                f"{baseline.schema_version}"
            )
        if baseline.k < 1:
            raise ValueError("baseline k must be at least 1")
        if baseline.total_queries < 1:
            raise ValueError("baseline total_queries must be at least 1")

        return baseline

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "k": self.k,
            "total_queries": self.total_queries,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
        }


@dataclass(frozen=True)
class BaselineComparison:
    baseline: RetrievalBaseline
    tolerance: float
    precision_delta: float
    recall_delta: float
    hit_rate_delta: float
    reciprocal_rank_delta: float
    regressed_metrics: tuple[str, ...]

    @property
    def status(self) -> RegressionStatus:
        return (
            RegressionStatus.REGRESSION
            if self.regressed_metrics
            else RegressionStatus.PASS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "tolerance": self.tolerance,
            "precision_delta": self.precision_delta,
            "recall_delta": self.recall_delta,
            "hit_rate_delta": self.hit_rate_delta,
            "reciprocal_rank_delta": self.reciprocal_rank_delta,
            "regressed_metrics": list(self.regressed_metrics),
            "baseline": self.baseline.to_dict(),
        }


def compare_to_baseline(
    candidate: EvaluationReport,
    baseline: RetrievalBaseline,
    *,
    tolerance: float = 0.01,
) -> BaselineComparison:
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")

    if candidate.dataset_name != baseline.dataset_name:
        raise ValueError(
            "candidate and baseline dataset names do not match"
        )
    if candidate.k != baseline.k:
        raise ValueError(
            "candidate and baseline k values do not match"
        )
    if candidate.total_queries != baseline.total_queries:
        raise ValueError(
            "candidate and baseline query counts do not match"
        )

    deltas = {
        "mean_precision_at_k": (
            candidate.mean_precision_at_k
            - baseline.mean_precision_at_k
        ),
        "mean_recall_at_k": (
            candidate.mean_recall_at_k
            - baseline.mean_recall_at_k
        ),
        "hit_rate_at_k": (
            candidate.hit_rate_at_k
            - baseline.hit_rate_at_k
        ),
        "mean_reciprocal_rank": (
            candidate.mean_reciprocal_rank
            - baseline.mean_reciprocal_rank
        ),
    }

    regressed = tuple(
        metric
        for metric, delta in deltas.items()
        if delta < -tolerance
    )

    return BaselineComparison(
        baseline=baseline,
        tolerance=tolerance,
        precision_delta=deltas["mean_precision_at_k"],
        recall_delta=deltas["mean_recall_at_k"],
        hit_rate_delta=deltas["hit_rate_at_k"],
        reciprocal_rank_delta=deltas["mean_reciprocal_rank"],
        regressed_metrics=regressed,
    )


def save_baseline(
    baseline: RetrievalBaseline,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            baseline.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
