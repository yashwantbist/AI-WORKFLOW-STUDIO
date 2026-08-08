"""Tests for grounded prompt construction."""

import pytest

from backend.ml.rag.prompt_builder import GroundedPromptBuilder
from backend.ml.rag.schemas import EvidenceChunk, RAGSource


def make_evidence(
    *,
    text: str = "Self-attention compares token relationships.",
    score: float = 0.91,
) -> EvidenceChunk:
    return EvidenceChunk(
        text=text,
        source=RAGSource(
            rank=1,
            score=score,
            chunk_id="chunk-17",
            document_id="transformers",
            document_title="Transformers Guide",
            source="transformers.pdf",
            page_start=12,
            page_end=12,
            sections=("Self-Attention",),
        ),
    )


def test_context_contains_text_and_source_metadata() -> None:
    builder = GroundedPromptBuilder()
    context = builder.build_context((make_evidence(),))

    assert "[Source 1]" in context
    assert "chunk-17" in context
    assert "Transformers Guide" in context
    assert "pages: 12" in context
    assert "Self-Attention" in context
    assert "Self-attention compares token relationships." in context


def test_prompt_requires_grounding_and_insufficient_context_behavior() -> None:
    builder = GroundedPromptBuilder()
    bundle = builder.build(
        "How does self-attention work?",
        (make_evidence(),),
    )

    assert "Use only the supplied retrieved context" in bundle.instructions
    assert "do not contain enough information" in bundle.instructions
    assert "How does self-attention work?" in bundle.prompt
    assert "Do not invent facts" in bundle.prompt


def test_prompt_builder_rejects_empty_question() -> None:
    builder = GroundedPromptBuilder()

    with pytest.raises(ValueError, match="question cannot be empty"):
        builder.build("   ", (make_evidence(),))


def test_context_guard_truncates_one_oversized_source() -> None:
    builder = GroundedPromptBuilder(max_context_characters=180)
    evidence = (make_evidence(text="attention " * 100),)

    context = builder.build_context(evidence)

    assert "context truncated" in context
    assert len(context) <= 210
