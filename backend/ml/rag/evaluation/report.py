"""Terminal and JSON output for RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from .evaluator import EvaluationRun


def format_terminal_report(run: EvaluationRun) -> str:
    lines = [
        "=" * 72,
        f"RAG EVALUATION: {run.dataset_name}",
        "=" * 72,
        f"top_k={run.top_k} | min_score={run.minimum_relevance_score:.2f}",
        "",
    ]

    for case in run.cases:
        m = case.retrieval_metrics
        a = case.answer_metrics
        lines.extend([
            f"[{case.status}] {case.case_id}",
            f"Question: {case.question}",
            f"Expected: {', '.join(case.expected_chunk_ids)}",
            "Retrieved:",
        ])
        if case.retrieved:
            lines.extend(
                f"  {r.rank}. {r.chunk_id} (score={r.score:.4f})"
                for r in case.retrieved
            )
        else:
            lines.append("  (none)")
        lines.extend([
            f"Hit@{run.top_k}: {'YES' if m.retrieval_passed else 'NO'}",
            f"Recall@{run.top_k}: {m.recall_at_k:.3f}",
            f"Precision@{run.top_k}: {m.precision_at_k:.3f}",
            f"Reciprocal rank: {m.reciprocal_rank:.3f}",
            f"Failure stage: {case.failure_stage}",
            f"Answer: {case.generated_answer}",
            f"Reference-token F1 proxy: {a.reference_token_f1:.3f}",
            f"Context-token support proxy: {a.context_token_support:.3f}",
            f"Expected source used: {a.expected_source_used}",
            "-" * 72,
        ])

    s = run.summary
    lines.extend([
        "SUMMARY",
        f"Cases: {s['case_count']}",
        f"Retrieval hit rate: {s['retrieval_hit_rate']:.3f}",
        f"Mean Recall@{run.top_k}: {s['mean_recall_at_k']:.3f}",
        f"Mean Precision@{run.top_k}: {s['mean_precision_at_k']:.3f}",
        f"Mean reciprocal rank: {s['mean_reciprocal_rank']:.3f}",
        f"Mean reference-token F1 proxy: {s['mean_reference_token_f1']:.3f}",
        f"Failure counts: {s['failure_counts']}",
        "",
        "Note: lexical answer metrics are regression proxies, not proof of semantic correctness.",
    ])
    return "\n".join(lines)


def save_json_report(run: EvaluationRun, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
