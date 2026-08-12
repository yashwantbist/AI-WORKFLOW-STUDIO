"""Terminal and JSON reporting for labelled offline retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from .offline_runner import EvaluationReport


def format_terminal_report(report: EvaluationReport) -> str:
    lines = [
        "=" * 78,
        f"OFFLINE RAG RETRIEVAL EVALUATION: {report.dataset_name}",
        "=" * 78,
        f"k={report.k} | queries={report.total_queries}",
        "",
    ]

    for result in report.results:
        metrics = result.metrics
        lines.extend(
            [
                f"[{result.query_id}] {result.query}",
                f"Relevant IDs: {', '.join(result.relevant_ids) or '(none)'}",
                "Retrieved:",
            ]
        )

        if result.retrieved:
            lines.extend(
                f"  {item.rank}. {item.chunk_id} "
                f"(score={item.score:.4f})"
                for item in result.retrieved
            )
        else:
            lines.append("  (none)")

        lines.extend(
            [
                f"Precision@{report.k}: {metrics.precision_at_k:.3f}",
                f"Recall@{report.k}: {metrics.recall_at_k:.3f}",
                f"Hit@{report.k}: {metrics.hit_at_k:.0f}",
                f"Reciprocal rank: {metrics.reciprocal_rank:.3f}",
                (
                    "Matched relevant IDs: "
                    + (
                        ", ".join(result.matched_relevant_ids)
                        or "(none)"
                    )
                ),
                (
                    "Missed relevant IDs: "
                    + (
                        ", ".join(result.missed_relevant_ids)
                        or "(none)"
                    )
                ),
                f"Retrieval latency: {result.retrieval_latency_ms:.2f} ms",
                "-" * 78,
            ]
        )

    lines.extend(
        [
            "AGGREGATE",
            f"Mean Precision@{report.k}: {report.mean_precision_at_k:.3f}",
            f"Mean Recall@{report.k}: {report.mean_recall_at_k:.3f}",
            f"Hit rate@{report.k}: {report.hit_rate_at_k:.3f}",
            f"Mean reciprocal rank: {report.mean_reciprocal_rank:.3f}",
            "",
            (
                "These metrics describe this labelled offline dataset only. "
                "They are not metrics for ordinary production traffic."
            ),
        ]
    )
    return "\n".join(lines)


def save_json_report(
    report: EvaluationReport,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
