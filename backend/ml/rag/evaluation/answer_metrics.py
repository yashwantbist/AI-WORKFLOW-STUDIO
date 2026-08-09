"""Lexical answer-quality proxies; these are not semantic judges."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
CITATION_PATTERN = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "with",
})


def normalize_token(token: str) -> str:
    """Apply tiny deterministic normalization for common English plurals.

    This is intentionally small and transparent; it is not a stemmer or
    lemmatizer. It only makes lexical regression metrics less brittle for
    pairs such as token/tokens and query/queries.
    """

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        normalize_token(token)
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOP_WORDS
    )


def token_precision_recall_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    p_tokens = content_tokens(prediction)
    r_tokens = content_tokens(reference)
    if not p_tokens or not r_tokens:
        return 0.0, 0.0, 0.0

    overlap = sum((Counter(p_tokens) & Counter(r_tokens)).values())
    precision = overlap / len(p_tokens)
    recall = overlap / len(r_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def context_token_support(answer: str, contexts: Iterable[str]) -> float:
    answer_tokens = set(content_tokens(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(content_tokens(" ".join(contexts)))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


@dataclass(frozen=True)
class AnswerMetrics:
    reference_token_precision: float
    reference_token_recall: float
    reference_token_f1: float
    context_token_support: float
    citation_count: int
    expected_source_used: bool

    def to_dict(self) -> dict:
        return {
            "reference_token_precision": self.reference_token_precision,
            "reference_token_recall": self.reference_token_recall,
            "reference_token_f1": self.reference_token_f1,
            "context_token_support": self.context_token_support,
            "citation_count": self.citation_count,
            "expected_source_used": self.expected_source_used,
        }


def evaluate_answer(
    answer: str,
    reference_answer: str,
    *,
    retrieved_contexts: Iterable[str],
    used_source_ids: Iterable[str],
    expected_chunk_ids: Iterable[str],
) -> AnswerMetrics:
    precision, recall, f1 = token_precision_recall_f1(answer, reference_answer)
    used = set(used_source_ids)
    expected = set(expected_chunk_ids)
    return AnswerMetrics(
        reference_token_precision=precision,
        reference_token_recall=recall,
        reference_token_f1=f1,
        context_token_support=context_token_support(answer, retrieved_contexts),
        citation_count=len(CITATION_PATTERN.findall(answer)),
        expected_source_used=bool(used & expected),
    )
