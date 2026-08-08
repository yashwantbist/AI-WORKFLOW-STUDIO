"""Grounded RAG orchestration: retrieval -> augmentation -> generation."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .llm_provider import LLMProvider
from .prompt_builder import GroundedPromptBuilder
from .schemas import RAGAnswer, evidence_from_retrieved


INSUFFICIENT_CONTEXT_ANSWER = (
    "The available sources do not contain enough information to answer "
    "this question."
)


class Retriever(Protocol):
    """Minimal interface implemented by the Day 17 FaissRetriever."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        """Return ranked retrieval results."""


class RAGPipeline:
    """Connect retrieval, context augmentation, and LLM generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        *,
        prompt_builder: GroundedPromptBuilder | None = None,
        minimum_relevance_score: float = 0.10,
    ) -> None:
        if not -1.0 <= minimum_relevance_score <= 1.0:
            raise ValueError(
                "minimum_relevance_score must be between -1.0 and 1.0"
            )
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._prompt_builder = prompt_builder or GroundedPromptBuilder()
        self._minimum_relevance_score = minimum_relevance_score

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        minimum_relevance_score: float | None = None,
    ) -> RAGAnswer:
        """Answer a question using only sufficiently relevant retrieved chunks."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        threshold = (
            self._minimum_relevance_score
            if minimum_relevance_score is None
            else minimum_relevance_score
        )
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "minimum_relevance_score must be between -1.0 and 1.0"
            )

        retrieved = self._retriever.search(
            clean_question,
            top_k=top_k,
            filters=filters,
        )

        evidence = tuple(
            evidence_from_retrieved(result)
            for result in retrieved
            if float(result.score) >= threshold
        )

        if not evidence:
            return RAGAnswer(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                sources=(),
                grounded=False,
                insufficient_context=True,
                retrieved_count=len(retrieved),
                used_context_count=0,
            )

        prompt_bundle = self._prompt_builder.build(
            clean_question,
            evidence,
        )
        answer_text = self._llm_provider.generate(
            instructions=prompt_bundle.instructions,
            prompt=prompt_bundle.prompt,
        ).strip()

        if not answer_text:
            raise RuntimeError("LLM provider returned an empty answer")

        return RAGAnswer(
            answer=answer_text,
            sources=tuple(item.source for item in evidence),
            grounded=True,
            insufficient_context=False,
            retrieved_count=len(retrieved),
            used_context_count=len(evidence),
        )
