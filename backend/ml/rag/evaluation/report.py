"""Terminal and JSON output for RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from .evaluator import EvaluationRun


def format_terminal_report(
    run: EvaluationRun,
) -> str:
    lines = [
        "=" * 72,
        f"RAG EVALUATION: {run.dataset_name}",
        "=" * 72,
        (
            f"top_k={run.top_k} | "
            f"min_score={run.minimum_relevance_score:.2f}"
        ),
        "",
    ]

    for case in run.cases:
        metrics = case.retrieval_metrics
        answer_metrics = case.answer_metrics

        lines.extend(
            [
                f"[{case.status}] {case.case_id}",
                f"Question: {case.question}",
                (
                    "Expected: "
                    + ", ".join(
                        case.expected_chunk_ids
                    )
                ),
                "Retrieved:",
            ]
        )

        if case.retrieved:
            lines.extend(
                (
                    f"  {item.rank}. {item.chunk_id} "
                    f"(score={item.score:.4f})"
                )
                for item in case.retrieved
            )
        else:
            lines.append("  (none)")

        lines.extend(
            [
                (
                    f"Hit@{run.top_k}: "
                    f"{'YES' if metrics.retrieval_passed else 'NO'}"
                ),
                (
                    f"Recall@{run.top_k}: "
                    f"{metrics.recall_at_k:.3f}"
                ),
                (
                    f"Precision@{run.top_k}: "
                    f"{metrics.precision_at_k:.3f}"
                ),
                (
                    "Reciprocal rank: "
                    f"{metrics.reciprocal_rank:.3f}"
                ),
                (
                    f"Failure stage: "
                    f"{case.failure_stage}"
                ),
                f"Answer: {case.generated_answer}",
                (
                    "Reference-token F1 proxy: "
                    f"{answer_metrics.reference_token_f1:.3f}"
                ),
                (
                    "Context-token support proxy: "
                    f"{answer_metrics.context_token_support:.3f}"
                ),
                (
                    "Expected source used: "
                    f"{answer_metrics.expected_source_used}"
                ),
            ]
        )

        if case.groundedness is None:
            lines.append(
                "Groundedness: not evaluated "
                "(no labelled claims)"
            )
        else:
            groundedness = case.groundedness
            lines.extend(
                [
                    (
                        "Groundedness score: "
                        f"{groundedness.groundedness_score:.3f}"
                    ),
                    (
                        "Grounded claims: "
                        f"{groundedness.supported_claims}/"
                        f"{groundedness.total_claims}"
                    ),
                    (
                        "Fully grounded: "
                        f"{groundedness.fully_grounded}"
                    ),
                ]
            )

            for claim in groundedness.claims:
                state = (
                    "SUPPORTED"
                    if claim.supported
                    else "UNSUPPORTED"
                )
                lines.append(
                    f"  [{state}] {claim.claim}"
                )
                lines.append(
                    "    Evidence IDs: "
                    + (
                        ", ".join(
                            claim.evidence_ids
                        )
                        or "(none)"
                    )
                )
                if claim.missing_evidence_ids:
                    lines.append(
                        "    Missing evidence IDs: "
                        + ", ".join(
                            claim.missing_evidence_ids
                        )
                    )

        lines.append("-" * 72)

    summary = run.summary

    lines.extend(
        [
            "SUMMARY",
            f"Cases: {summary['case_count']}",
            (
                "Retrieval hit rate: "
                f"{summary['retrieval_hit_rate']:.3f}"
            ),
            (
                f"Mean Recall@{run.top_k}: "
                f"{summary['mean_recall_at_k']:.3f}"
            ),
            (
                f"Mean Precision@{run.top_k}: "
                f"{summary['mean_precision_at_k']:.3f}"
            ),
            (
                "Mean reciprocal rank: "
                f"{summary['mean_reciprocal_rank']:.3f}"
            ),
            (
                "Mean reference-token F1 proxy: "
                f"{summary['mean_reference_token_f1']:.3f}"
            ),
            (
                "Groundedness-labelled cases: "
                f"{summary['groundedness_case_count']}"
            ),
        ]
    )

    if summary["mean_groundedness_score"] is None:
        lines.append(
            "Mean groundedness score: not evaluated"
        )
    else:
        lines.append(
            "Mean groundedness score: "
            f"{summary['mean_groundedness_score']:.3f}"
        )

    lines.extend(
        [
            (
                "Failure counts: "
                f"{summary['failure_counts']}"
            ),
            "",
            (
                "Note: lexical answer metrics are regression proxies, "
                "not proof of semantic correctness."
            ),
            (
                "Groundedness v1 is deterministic and label-based; "
                "it is not an LLM semantic judge."
            ),
        ]
    )

    return "\n".join(lines)


def save_json_report(
    run: EvaluationRun,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            run.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path
