"""Prompt construction for grounded RAG answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .schemas import EvidenceChunk


DEFAULT_INSTRUCTIONS = """You are a grounded knowledge assistant.

Use only the supplied retrieved context to answer the user's question.
Do not rely on unstated background knowledge.
If the context does not contain enough evidence to answer, say:
\"The available sources do not contain enough information to answer this question.\"

When you use a fact from a source, cite it inline using its source label,
for example [Source 1]. Do not invent source labels or citations.
Be concise and distinguish uncertainty from known facts.
"""


@dataclass(frozen=True)
class PromptBundle:
    """System instructions and user prompt sent to an LLM provider."""

    instructions: str
    prompt: str


class GroundedPromptBuilder:
    """Build context and prompts independently from retrieval and generation."""

    def __init__(
        self,
        instructions: str = DEFAULT_INSTRUCTIONS,
        *,
        max_context_characters: int = 12_000,
    ) -> None:
        if max_context_characters < 1:
            raise ValueError("max_context_characters must be at least 1")
        self._instructions = instructions.strip()
        self._max_context_characters = max_context_characters

    @staticmethod
    def _format_source_label(index: int, evidence: EvidenceChunk) -> str:
        source = evidence.source
        pages = (
            str(source.page_start)
            if source.page_start == source.page_end
            else f"{source.page_start}-{source.page_end}"
        )
        sections = ", ".join(source.sections) if source.sections else "Unknown"
        return (
            f"[Source {index}]\n"
            f"chunk_id: {source.chunk_id}\n"
            f"document: {source.document_title}\n"
            f"pages: {pages}\n"
            f"sections: {sections}\n"
            f"score: {source.score:.4f}\n"
            f"text: {evidence.text.strip()}"
        )

    def build_context(self, evidence: Sequence[EvidenceChunk]) -> str:
        """Format retrieved chunks while respecting a context-size guard."""

        if not evidence:
            return "(no retrieved context)"

        blocks: list[str] = []
        total_characters = 0

        for index, item in enumerate(evidence, start=1):
            block = self._format_source_label(index, item)
            separator_cost = 2 if blocks else 0
            projected = total_characters + separator_cost + len(block)

            if projected > self._max_context_characters:
                if not blocks:
                    suffix = "\n[context truncated]"
                    available = max(1, self._max_context_characters - len(suffix))
                    blocks.append(block[:available] + suffix)
                break

            blocks.append(block)
            total_characters = projected

        return "\n\n".join(blocks)

    def build(
        self,
        question: str,
        evidence: Sequence[EvidenceChunk],
    ) -> PromptBundle:
        """Construct the complete grounded prompt."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question cannot be empty")

        context = self.build_context(evidence)
        prompt = (
            "RETRIEVED CONTEXT\n"
            "=================\n"
            f"{context}\n\n"
            "USER QUESTION\n"
            "=============\n"
            f"{clean_question}\n\n"
            "ANSWER REQUIREMENTS\n"
            "===================\n"
            "- Answer only from the retrieved context.\n"
            "- Cite supporting statements with [Source N].\n"
            "- If evidence is insufficient, say so explicitly.\n"
            "- Do not invent facts, citations, page numbers, or document names."
        )
        return PromptBundle(
            instructions=self._instructions,
            prompt=prompt,
        )
