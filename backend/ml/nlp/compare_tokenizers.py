"""Compare token counts produced by all included tokenizers."""

from __future__ import annotations

import argparse

try:
    from .token_visualizer import print_comparison_table
    from .tokenizer_demo import compare_all_tokenizers
except ImportError:
    from backend.ml.nlp.token_visualizer import print_comparison_table
    from backend.ml.nlp.tokenizer_demo import compare_all_tokenizers


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple tokenization strategies.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to compare. When omitted, the program prompts for text.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=4096,
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    text = arguments.text or input("Enter text to compare: ")
    results = compare_all_tokenizers(
        text,
        context_window=arguments.context_window,
    )
    print_comparison_table(results)


if __name__ == "__main__":
    main()
