"""Terminal and JSON output for RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from .evaluator import EvaluationRun


def _format_optional_score(
    value: float | None,
) -> str:
    if value is None:
        return "not evaluated"
    return f"{value:.3f}"


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
        generation = case.generation_evaluation

        lines.extend(
            [
                f"[{case.status}] {case.case_id}",
                f"Question: {case.question}",
                "",
                "RETRIEVAL EVALUATION",
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
                "",
                "GENERATION EVALUATION",
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

        lines.extend(
            [
                (
                    "Answer relevance: "
                    + _format_optional_score(
                        generation.answer_relevance
                    )
                ),
                (
                    "Correctness: "
                    + _format_optional_score(
                        generation.correctness
                    )
                ),
            ]
        )

        if generation.relevance_details is not None:
            details = generation.relevance_details

            lines.append(
                "  Relevance matched phrases: "
                + (
                    ", ".join(
                        details.matched_phrases
                    )
                    or "(none)"
                )
            )

            lines.append(
                "  Relevance missing phrases: "
                + (
                    ", ".join(
                        details.missing_phrases
                    )
                    or "(none)"
                )
            )

        if generation.correctness_details is not None:
            details = generation.correctness_details

            lines.append(
                "  Correctness matched phrases: "
                + (
                    ", ".join(
                        details.matched_phrases
                    )
                    or "(none)"
                )
            )

            lines.append(
                "  Correctness missing phrases: "
                + (
                    ", ".join(
                        details.missing_phrases
                    )
                    or "(none)"
                )
            )

            lines.append(
                "  Correctness conflicting phrases: "
                + (
                    ", ".join(
                        details.conflicting_phrases
                    )
                    or "(none)"
                )
            )

        lines.extend(
            [
                (
                    "Failure categories: "
                    + ", ".join(
                        category.value
                        for category in (
                            generation.failure_categories
                        )
                    )
                ),
                (
                    "Primary failure stage: "
                    f"{case.failure_stage}"
                ),
                "-" * 72,
            ]
        )

    summary = run.summary

    lines.extend(
        [
            "SUMMARY",
            f"Cases: {summary['case_count']}",
            "",
            "RETRIEVAL SUMMARY",
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
            "",
            "GENERATION SUMMARY",
            (
                "Mean reference-token F1 proxy: "
                f"{summary['mean_reference_token_f1']:.3f}"
            ),
            (
                "Groundedness-labelled cases: "
                f"{summary['groundedness_case_count']}"
            ),
            (
                "Mean groundedness score: "
                + _format_optional_score(
                    summary["mean_groundedness_score"]
                )
            ),
            (
                "Answer-relevance-labelled cases: "
                f"{summary['answer_relevance_case_count']}"
            ),
            (
                "Mean answer relevance: "
                + _format_optional_score(
                    summary["mean_answer_relevance"]
                )
            ),
            (
                "Correctness-labelled cases: "
                f"{summary['correctness_case_count']}"
            ),
            (
                "Mean correctness: "
                + _format_optional_score(
                    summary["mean_correctness"]
                )
            ),
            (
                "Generation failure categories: "
                f"{summary['generation_failure_counts']}"
            ),
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
                "Groundedness, answer relevance, and correctness v1 "
                "use deterministic labels; they are not LLM semantic judges."
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
