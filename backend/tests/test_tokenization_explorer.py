"""Tests for the Day 12 tokenization explorer."""

from pathlib import Path

from backend.ml.nlp.token_visualizer import (
    save_html_visualization,
    visible_token,
)
from backend.ml.nlp.tokenizer_demo import (
    BPETokenizer,
    CharacterTokenizer,
    WhitespaceTokenizer,
    WordPunctuationTokenizer,
    analyze_text,
    compare_all_tokenizers,
)
from backend.ml.nlp.vocabulary import (
    PAD_TOKEN,
    UNKNOWN_TOKEN,
    Vocabulary,
)


def test_whitespace_tokenizer_keeps_punctuation_attached() -> None:
    tokenizer = WhitespaceTokenizer()

    assert tokenizer.tokenize("Hello, world!") == ["Hello,", "world!"]


def test_word_tokenizer_separates_words_numbers_and_punctuation() -> None:
    tokenizer = WordPunctuationTokenizer()

    assert tokenizer.tokenize("AI costs $1,000.50!") == [
        "AI",
        "costs",
        "$",
        "1,000.50",
        "!",
    ]


def test_character_tokenizer_preserves_spaces() -> None:
    tokenizer = CharacterTokenizer()

    assert tokenizer.tokenize("A B") == ["A", " ", "B"]
    assert visible_token(" ") == "␠"


def test_vocabulary_ids_are_reserved_and_deterministic() -> None:
    first = Vocabulary.build([["cat", "dog", "cat"]])
    second = Vocabulary.build([["cat", "dog", "cat"]])

    assert first.to_dict()[PAD_TOKEN] == 0
    assert first.to_dict()[UNKNOWN_TOKEN] == 1
    assert first.to_dict() == second.to_dict()
    assert first.encode(["missing"]) == [first.unknown_id]


def test_analyze_text_calculates_context_usage() -> None:
    result = analyze_text(
        "one two three",
        WhitespaceTokenizer(),
        context_window=4,
    )

    assert result.token_count == 3
    assert result.remaining_tokens == 1
    assert result.fits_context is True
    assert result.context_percentage == 75.0


def test_toy_bpe_is_repeatable_and_produces_subwords() -> None:
    first = BPETokenizer(number_of_merges=30)
    second = BPETokenizer(number_of_merges=30)

    assert first.merge_rules == second.merge_rules
    assert first.tokenize("tokenization")
    assert all(
        token
        for token in first.tokenize("tokenization")
    )


def test_comparison_runs_every_strategy() -> None:
    results = compare_all_tokenizers("Hello world!", context_window=32)

    assert [result.strategy for result in results] == [
        "whitespace",
        "word-punctuation",
        "toy-bpe",
        "character",
    ]


def test_html_visualization_escapes_user_text(tmp_path: Path) -> None:
    result = analyze_text(
        "<script>alert('x')</script>",
        WordPunctuationTokenizer(),
    )
    output_path = save_html_visualization(
        result,
        tmp_path / "tokens.html",
    )
    html = output_path.read_text(encoding="utf-8")

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "Tokenization Explorer" in html
