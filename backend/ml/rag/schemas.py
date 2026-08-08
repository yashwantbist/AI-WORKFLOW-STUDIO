"""API-friendly schemas for grounded retrieval-augmented generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RAGSource:
    """Traceable source metadata returned with a generated answer."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    page_start: int
    page_end: int
    sections: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = list(self.sections)
        return data


@dataclass(frozen=True)
class EvidenceChunk:
    """Retrieved text plus the source metadata needed for grounding."""

    text: str
    source: RAGSource


@dataclass(frozen=True)
class RAGAnswer:
    """Final answer returned by the grounded RAG pipeline."""

    answer: str
    sources: tuple[RAGSource, ...]
    grounded: bool
    insufficient_context: bool
    retrieved_count: int
    used_context_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "grounded": self.grounded,
            "insufficient_context": self.insufficient_context,
            "retrieved_count": self.retrieved_count,
            "used_context_count": self.used_context_count,
        }


def evidence_from_retrieved(result: Any) -> EvidenceChunk:
    """Adapt the Day 17 RetrievedChunk shape without tight coupling."""

    chunk = result.chunk
    return EvidenceChunk(
        text=chunk.text,
        source=RAGSource(
            rank=int(result.rank),
            score=float(result.score),
            chunk_id=str(chunk.chunk_id),
            document_id=str(chunk.document_id),
            document_title=str(chunk.document_title),
            source=str(chunk.source),
            page_start=int(chunk.page_start),
            page_end=int(chunk.page_end),
            sections=tuple(str(value) for value in chunk.sections),
        ),
    )
