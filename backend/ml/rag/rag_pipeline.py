"""Grounded RAG orchestration: retrieval -> augmentation -> generation."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .llm_provider import LLMProvider
from .prompt_builder import GroundedPromptBuilder
from .schemas import RAGAnswer, evidence_from_retrieved

INSUFFICIENT_CONTEXT_ANSWER = (
    "The available sources do not contain enough information to answer "
    "this question."
)


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        ...


class RAGPipeline:
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

    @staticmethod
    def _validate_question(question: str) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question cannot be empty")
        return clean_question

    def _resolve_threshold(
        self,
        minimum_relevance_score: float | None,
    ) -> float:
        threshold = (
            self._minimum_relevance_score
            if minimum_relevance_score is None
            else minimum_relevance_score
        )
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "minimum_relevance_score must be between -1.0 and 1.0"
            )
        return threshold

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        minimum_relevance_score: float | None = None,
    ) -> RAGAnswer:
        clean_question = self._validate_question(question)
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        retrieved = self._retriever.search(
            clean_question,
            top_k=top_k,
            filters=filters,
        )
        return self.answer_from_retrieved(
            clean_question,
            retrieved,
            minimum_relevance_score=minimum_relevance_score,
        )

    def answer_from_retrieved(
        self,
        question: str,
        retrieved: Sequence[Any],
        *,
        minimum_relevance_score: float | None = None,
    ) -> RAGAnswer:
        """Generate from a previously computed retrieval result."""

        clean_question = self._validate_question(question)
        threshold = self._resolve_threshold(minimum_relevance_score)
        retrieved_items = tuple(retrieved)

        evidence = tuple(
            evidence_from_retrieved(result)
            for result in retrieved_items
            if float(result.score) >= threshold
        )

        if not evidence:
            return RAGAnswer(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                sources=(),
                grounded=False,
                insufficient_context=True,
                retrieved_count=len(retrieved_items),
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
            retrieved_count=len(retrieved_items),
            used_context_count=len(evidence),
        )
