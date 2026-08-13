"""CLI for Day 23 diagnostic and regression retrieval evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..retriever import FaissRetriever
from .baseline import (
    RetrievalBaseline,
    compare_to_baseline,
    save_baseline,
)
from .offline_dataset import OfflineEvaluationDataset
from .offline_report import (
    format_diagnostic_report,
    save_diagnostic_json_report,
)
from .offline_runner import OfflineRetrievalEvaluator


DEFAULT_DATASET = Path(
    "backend/ml/rag/datasets/retrieval_eval.json"
)
DEFAULT_INDEX_DIR = Path(
    "backend/ml/rag/artifacts/faiss"
)
DEFAULT_REPORT = Path(
    "backend/ml/rag/artifacts/evaluation/day23-diagnostic-report.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run labelled offline retrieval diagnostics and optionally "
            "compare the candidate against a measured baseline."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Compare this run with an existing baseline JSON file.",
    )
    parser.add_argument(
        "--save-baseline",
        type=Path,
        default=None,
        help="Save this measured run as a future baseline.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help=(
            "Allowed metric drop before regression is declared. "
            "Default: 0.01"
        ),
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 if a baseline regression is detected.",
    )
    args = parser.parse_args()

    dataset = OfflineEvaluationDataset.load(args.dataset)
    retriever = FaissRetriever.load(args.index_dir)
    report = OfflineRetrievalEvaluator(
        retriever,
        k=args.top_k,
    ).evaluate_dataset(dataset)

    comparison = None
    if args.baseline is not None:
        baseline = RetrievalBaseline.load(args.baseline)
        comparison = compare_to_baseline(
            report,
            baseline,
            tolerance=args.tolerance,
        )

    print(format_diagnostic_report(report, comparison))

    report_path = save_diagnostic_json_report(
        report,
        args.json_report,
        comparison=comparison,
    )
    print(f"\nDiagnostic JSON report written to: {report_path}")

    if args.save_baseline is not None:
        baseline_path = save_baseline(
            RetrievalBaseline.from_report(report),
            args.save_baseline,
        )
        print(f"Baseline written to: {baseline_path}")

    if (
        args.fail_on_regression
        and comparison is not None
        and comparison.regressed_metrics
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
