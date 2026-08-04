"""Beginner-friendly tokenization explorer with a command-line interface."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol

try:
    from .token_visualizer import (
        print_comparison_table,
        print_token_table,
        save_html_visualization,
    )
    from .vocabulary import Vocabulary
except ImportError:
    from backend.ml.nlp.token_visualizer import (
        print_comparison_table,
        print_token_table,
        save_html_visualization,
    )
    from backend.ml.nlp.vocabulary import Vocabulary


WORD_OR_PUNCTUATION_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]",
)

DEFAULT_BPE_CORPUS = (
    "artificial intelligence helps people learn",
    "intelligent systems process language",
    "token tokenization tokenizer tokens",
    "learning learners learned reusable pieces",
    "amazing unbelievable believable belief",
    "models read text as tokens not words",
)


class Tokenizer(Protocol):
    """Common interface implemented by every tokenizer strategy."""

    name: str

    def tokenize(self, text: str) -> list[str]:
        """Split text into a list of token strings."""


@dataclass(frozen=True)
class TokenizationResult:
    """The tokens, local IDs, and context estimate for one strategy."""

    strategy: str
    text: str
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    vocabulary_size: int
    context_window: int

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def remaining_tokens(self) -> int:
        return self.context_window - self.token_count

    @property
    def fits_context(self) -> bool:
        return self.token_count <= self.context_window

    @property
    def context_percentage(self) -> float:
        if self.context_window == 0:
            return 0.0
        return self.token_count / self.context_window * 100


class WhitespaceTokenizer:
    """Split only where Python sees whitespace."""

    name = "whitespace"

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class WordPunctuationTokenizer:
    """Separate words, numbers, and punctuation."""

    name = "word-punctuation"

    def tokenize(self, text: str) -> list[str]:
        return WORD_OR_PUNCTUATION_PATTERN.findall(text)


class CharacterTokenizer:
    """Treat every Unicode character, including spaces, as a token."""

    name = "character"

    def tokenize(self, text: str) -> list[str]:
        return list(text)


class BPETokenizer:
    """A small educational Byte Pair Encoding tokenizer.

    It learns frequent character-pair merges from a tiny built-in corpus.
    Production BPE tokenizers use much larger corpora and carefully defined
    byte-level rules, but the repeated-pair merge idea is the same.
    """

    name = "toy-bpe"
    END_OF_WORD = "</w>"

    def __init__(
        self,
        training_corpus: Iterable[str] = DEFAULT_BPE_CORPUS,
        number_of_merges: int = 60,
    ) -> None:
        if number_of_merges < 0:
            raise ValueError("number_of_merges cannot be negative")

        self._merge_rules = self._learn_merges(
            training_corpus,
            number_of_merges,
        )

    @property
    def merge_rules(self) -> tuple[tuple[str, str], ...]:
        """Return the learned pair-merging order."""
        return self._merge_rules

    @classmethod
    def _initial_word_vocabulary(
        cls,
        training_corpus: Iterable[str],
    ) -> Counter[tuple[str, ...]]:
        word_counts: Counter[tuple[str, ...]] = Counter()
        for text in training_corpus:
            for token in WORD_OR_PUNCTUATION_PATTERN.findall(text.lower()):
                symbols = tuple(token) + (cls.END_OF_WORD,)
                word_counts[symbols] += 1
        return word_counts

    @staticmethod
    def _pair_counts(
        word_vocabulary: Counter[tuple[str, ...]],
    ) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for symbols, frequency in word_vocabulary.items():
            for left, right in zip(symbols, symbols[1:]):
                pair_counts[(left, right)] += frequency
        return pair_counts

    @staticmethod
    def _merge_pair_in_symbols(
        symbols: tuple[str, ...],
        pair: tuple[str, str],
    ) -> tuple[str, ...]:
        merged: list[str] = []
        index = 0

        while index < len(symbols):
            if (
                index < len(symbols) - 1
                and symbols[index] == pair[0]
                and symbols[index + 1] == pair[1]
            ):
                merged.append(symbols[index] + symbols[index + 1])
                index += 2
            else:
                merged.append(symbols[index])
                index += 1

        return tuple(merged)

    @classmethod
    def _learn_merges(
        cls,
        training_corpus: Iterable[str],
        number_of_merges: int,
    ) -> tuple[tuple[str, str], ...]:
        word_vocabulary = cls._initial_word_vocabulary(training_corpus)
        merge_rules: list[tuple[str, str]] = []

        for _ in range(number_of_merges):
            pair_counts = cls._pair_counts(word_vocabulary)
            if not pair_counts:
                break

            highest_frequency = max(pair_counts.values())
            best_pair = min(
                pair
                for pair, frequency in pair_counts.items()
                if frequency == highest_frequency
            )
            merge_rules.append(best_pair)

            updated_vocabulary: Counter[tuple[str, ...]] = Counter()
            for symbols, frequency in word_vocabulary.items():
                updated_symbols = cls._merge_pair_in_symbols(
                    symbols,
                    best_pair,
                )
                updated_vocabulary[updated_symbols] += frequency
            word_vocabulary = updated_vocabulary

        return tuple(merge_rules)

    def _tokenize_piece(self, piece: str) -> list[str]:
        symbols = tuple(piece.lower()) + (self.END_OF_WORD,)

        for pair in self._merge_rules:
            symbols = self._merge_pair_in_symbols(symbols, pair)

        cleaned_tokens: list[str] = []
        for symbol in symbols:
            cleaned = symbol.replace(self.END_OF_WORD, "")
            if cleaned:
                cleaned_tokens.append(cleaned)
        return cleaned_tokens

    def tokenize(self, text: str) -> list[str]:
        output: list[str] = []
        for piece in WORD_OR_PUNCTUATION_PATTERN.findall(text):
            output.extend(self._tokenize_piece(piece))
        return output


TOKENIZER_FACTORIES = {
    "whitespace": WhitespaceTokenizer,
    "word": WordPunctuationTokenizer,
    "character": CharacterTokenizer,
    "bpe": BPETokenizer,
}


def analyze_text(
    text: str,
    tokenizer: Tokenizer,
    context_window: int = 4096,
) -> TokenizationResult:
    """Tokenize text, assign local IDs, and estimate context usage."""
    if context_window < 1:
        raise ValueError("context_window must be at least 1")

    tokens = tokenizer.tokenize(text)
    vocabulary = Vocabulary.build([tokens])
    token_ids = vocabulary.encode(tokens)

    return TokenizationResult(
        strategy=tokenizer.name,
        text=text,
        tokens=tuple(tokens),
        token_ids=tuple(token_ids),
        vocabulary_size=len(vocabulary),
        context_window=context_window,
    )


def compare_all_tokenizers(
    text: str,
    context_window: int = 4096,
) -> list[TokenizationResult]:
    """Analyze the same text with every included tokenization strategy."""
    tokenizers: Sequence[Tokenizer] = (
        WhitespaceTokenizer(),
        WordPunctuationTokenizer(),
        BPETokenizer(),
        CharacterTokenizer(),
    )
    return [
        analyze_text(text, tokenizer, context_window)
        for tokenizer in tokenizers
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explore tokens, local token IDs, token counts, and context usage."
        ),
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to tokenize. When omitted, the program prompts for text.",
    )
    parser.add_argument(
        "--strategy",
        choices=("all", *TOKENIZER_FACTORIES),
        default="all",
        help="Tokenizer strategy to run.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=4096,
        help="Maximum number of tokens allowed by the target model.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="Optional output path for an HTML token visualization.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    text = arguments.text or input("Enter text to tokenize: ")

    if arguments.strategy == "all":
        results = compare_all_tokenizers(
            text,
            context_window=arguments.context_window,
        )
        print_comparison_table(results)
        print()
        selected_result = results[1]
        print(
            "Detailed token view uses the word-punctuation strategy.\n",
        )
    else:
        tokenizer = TOKENIZER_FACTORIES[arguments.strategy]()
        selected_result = analyze_text(
            text,
            tokenizer,
            context_window=arguments.context_window,
        )

    print_token_table(selected_result)

    if arguments.html:
        saved_path = save_html_visualization(
            selected_result,
            arguments.html,
        )
        print(f"\nHTML visualization saved to: {saved_path}")


if __name__ == "__main__":
    main()
