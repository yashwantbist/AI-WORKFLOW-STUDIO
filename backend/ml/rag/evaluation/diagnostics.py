"""Failure classification for labelled retrieval evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence


class RetrievalStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    PARTIAL = "partial"
    COMPLETE_MISS = "complete_miss"
    IRRELEVANT_RESULTS = "irrelevant_results"


def classify_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> RetrievalStatus:
    """Classify coverage of the labelled relevant set.

    A SUCCESS may still contain extra irrelevant chunks. That noise is tracked
    separately because successful coverage and retrieval noise can coexist.
    """

    retrieved = tuple(str(value).strip() for value in retrieved_ids)
    relevant = {str(value).strip() for value in relevant_ids}

    if any(not value for value in retrieved):
        raise ValueError("retrieved_ids cannot contain empty IDs")
    if any(not value for value in relevant):
        raise ValueError("relevant_ids cannot contain empty IDs")

    retrieved_set = set(retrieved)

    # Explicit no-answer golden cases.
    if not relevant:
        return (
            RetrievalStatus.SUCCESS
            if not retrieved
            else RetrievalStatus.IRRELEVANT_RESULTS
        )

    if not retrieved:
        return RetrievalStatus.NO_RESULTS

    matched = retrieved_set & relevant

    if not matched:
        return RetrievalStatus.COMPLETE_MISS

    if matched != relevant:
        return RetrievalStatus.PARTIAL

    return RetrievalStatus.SUCCESS


def irrelevant_retrieved_ids(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> tuple[str, ...]:
    relevant = {str(value).strip() for value in relevant_ids}
    return tuple(
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id not in relevant
    )
