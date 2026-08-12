"""CLI for Day 22 labelled offline RAG retrieval evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..retriever import FaissRetriever
from .offline_dataset import OfflineEvaluationDataset
from .offline_report import format_terminal_report, save_json_report
from .offline_runner import OfflineRetrievalEvaluator


DEFAULT_DATASET = Path(
    "backend/ml/rag/datasets/retrieval_eval.json"
)
DEFAULT_INDEX_DIR = Path(
    "backend/ml/rag/artifacts/faiss"
)
DEFAULT_REPORT = Path(
    "backend/ml/rag/artifacts/evaluation/day22-retrieval-report.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a labelled offline retrieval dataset against "
            "the AI Workflow Studio FAISS retriever."
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
    args = parser.parse_args()

    dataset = OfflineEvaluationDataset.load(args.dataset)
    retriever = FaissRetriever.load(args.index_dir)
    evaluator = OfflineRetrievalEvaluator(
        retriever,
        k=args.top_k,
    )
    report = evaluator.evaluate_dataset(dataset)

    print(format_terminal_report(report))
    output_path = save_json_report(report, args.json_report)
    print(f"\nJSON report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
