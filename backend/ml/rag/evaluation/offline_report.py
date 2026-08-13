"""Human-readable and JSON diagnostics for offline retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from .baseline import BaselineComparison
from .offline_runner import EvaluationReport


def format_terminal_report(report: EvaluationReport) -> str:
    """Backward-compatible Day 22 report formatter."""
    return format_diagnostic_report(report)


def format_diagnostic_report(
    report: EvaluationReport,
    comparison: BaselineComparison | None = None,
) -> str:
    lines = [
        "=" * 78,
        f"RAG RETRIEVAL DIAGNOSTIC REPORT: {report.dataset_name}",
        "=" * 78,
        f"Queries evaluated: {report.total_queries}",
        f"K: {report.k}",
        "",
        "AGGREGATE METRICS",
        f"Mean Precision@{report.k}: {report.mean_precision_at_k:.3f}",
        f"Mean Recall@{report.k}: {report.mean_recall_at_k:.3f}",
        f"Hit rate@{report.k}: {report.hit_rate_at_k:.3f}",
        f"Mean reciprocal rank: {report.mean_reciprocal_rank:.3f}",
        "",
        "RETRIEVAL STATUS",
        f"Successful: {report.successful_queries}",
        f"Partial: {report.partial_queries}",
        f"Complete misses: {report.complete_misses}",
        f"No results: {report.no_result_queries}",
        f"Irrelevant-only no-answer cases: {report.irrelevant_only_queries}",
        (
            "Queries containing extra irrelevant results: "
            f"{report.queries_with_irrelevant_results}"
        ),
        "",
    ]

    if report.failures:
        lines.append("FAILURES")
        lines.append("-" * 78)

        for result in report.failures:
            lines.extend(
                [
                    f"[{result.query_id}] {result.query}",
                    f"Status: {result.status.value}",
                    (
                        "Expected: "
                        + (", ".join(result.relevant_ids) or "(none)")
                    ),
                    (
                        "Retrieved: "
                        + (", ".join(result.retrieved_ids) or "(none)")
                    ),
                    (
                        "Matched: "
                        + (
                            ", ".join(result.matched_relevant_ids)
                            or "(none)"
                        )
                    ),
                    (
                        "Missed: "
                        + (
                            ", ".join(result.missed_relevant_ids)
                            or "(none)"
                        )
                    ),
                    (
                        "Irrelevant retrieved: "
                        + (
                            ", ".join(result.irrelevant_retrieved_ids)
                            or "(none)"
                        )
                    ),
                    (
                        f"Precision@{report.k}: "
                        f"{result.metrics.precision_at_k:.3f}"
                    ),
                    (
                        f"Recall@{report.k}: "
                        f"{result.metrics.recall_at_k:.3f}"
                    ),
                    "-" * 78,
                ]
            )
    else:
        lines.extend(
            [
                "FAILURES",
                "(none)",
                "",
            ]
        )

    if comparison is not None:
        lines.extend(
            [
                "BASELINE COMPARISON",
                (
                    f"Status: {comparison.status.value.upper()} "
                    f"(tolerance={comparison.tolerance:.4f})"
                ),
                f"Precision delta: {comparison.precision_delta:+.4f}",
                f"Recall delta: {comparison.recall_delta:+.4f}",
                f"Hit-rate delta: {comparison.hit_rate_delta:+.4f}",
                (
                    "Reciprocal-rank delta: "
                    f"{comparison.reciprocal_rank_delta:+.4f}"
                ),
                (
                    "Regressed metrics: "
                    + (
                        ", ".join(comparison.regressed_metrics)
                        or "(none)"
                    )
                ),
                "",
            ]
        )

    lines.append(
        "These results describe this labelled offline dataset only; "
        "they are not production-traffic metrics."
    )
    return "\n".join(lines)


def save_json_report(
    report: EvaluationReport,
    path: str | Path,
) -> Path:
    """Backward-compatible Day 22 JSON export."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def save_diagnostic_json_report(
    report: EvaluationReport,
    path: str | Path,
    *,
    comparison: BaselineComparison | None = None,
) -> Path:
    payload: dict[str, object] = {
        "evaluation": report.to_dict(),
        "baseline_comparison": (
            comparison.to_dict()
            if comparison is not None
            else None
        ),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
