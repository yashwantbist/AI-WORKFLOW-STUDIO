"""Service boundary used by HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RAGServiceResult:
    answer: str


class RAGService(Protocol):
    def answer(
        self,
        question: str,
        *,
        request_id: str,
    ) -> RAGServiceResult:
        """Answer a question while preserving the request trace ID."""
        ...


class ReadinessCheck(Protocol):
    def is_ready(self) -> bool:
        """Return whether required dependencies are ready to serve traffic."""
        ...
