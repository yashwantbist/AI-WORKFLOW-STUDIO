"""Command-line entry point for Day 19 evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..llm_provider import OpenAIResponsesProvider, RecordingDemoProvider
from ..rag_pipeline import RAGPipeline
from ..retriever import FaissRetriever
from .dataset import EvaluationDataset
from .evaluator import RAGEvaluator
from .report import format_terminal_report, save_json_report

DEFAULT_DATASET = Path("data/rag_eval.json")
DEFAULT_INDEX_DIR = Path("backend/ml/rag/artifacts/faiss")
DEFAULT_REPORT = Path("backend/ml/rag/artifacts/evaluation/day19-report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local RAG pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--provider", choices=("demo", "openai"), default="demo")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--answer-f1-review-threshold", type=float, default=0.20)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    dataset = EvaluationDataset.load(args.dataset)
    retriever = FaissRetriever.load(args.index_dir)
    provider = (
        OpenAIResponsesProvider.from_environment()
        if args.provider == "openai"
        else RecordingDemoProvider()
    )
    pipeline = RAGPipeline(
        retriever,
        provider,
        minimum_relevance_score=args.min_score,
    )
    evaluator = RAGEvaluator(
        retriever,
        pipeline,
        top_k=args.top_k,
        minimum_relevance_score=args.min_score,
        answer_f1_review_threshold=args.answer_f1_review_threshold,
    )
    run = evaluator.evaluate(dataset)
    print(format_terminal_report(run))
    report_path = save_json_report(run, args.json_report)
    print(f"\nJSON report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
