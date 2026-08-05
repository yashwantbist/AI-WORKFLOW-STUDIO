"""Recursive-style chunker that prefers logical boundaries."""
from dataclasses import dataclass
from .chunker import validate_chunk_configuration
from .metadata import DocumentChunk, DocumentLayout, SourceDocument, analyze_document, build_chunk

@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size_words: int = 100
    overlap_words: int = 20
    minimum_fill_ratio: float = 0.5
    def __post_init__(self) -> None:
        validate_chunk_configuration(self.chunk_size_words, self.overlap_words)
        if not 0.0 < self.minimum_fill_ratio <= 1.0:
            raise ValueError("minimum_fill_ratio must be greater than 0 and at most 1")
    def _choose_boundary(self, layout: DocumentLayout, start: int, target: int) -> int:
        minimum = min(target, start + max(1, int(self.chunk_size_words * self.minimum_fill_ratio)))
        for boundaries in (layout.section_ends, layout.paragraph_ends, layout.sentence_ends):
            choices = [boundary for boundary in boundaries if minimum <= boundary <= target]
            if choices:
                return max(choices)
        return target
    def chunk(self, document: SourceDocument) -> tuple[DocumentChunk, ...]:
        layout = analyze_document(document)
        if not layout.words:
            return ()
        chunks: list[DocumentChunk] = []
        start = previous_end = 0
        while start < len(layout.words):
            target = min(start + self.chunk_size_words, len(layout.words))
            end = self._choose_boundary(layout, start, target)
            chunks.append(build_chunk(document, layout, start_word=start, end_word=end, strategy="recursive", chunk_index=len(chunks)+1, previous_end_word=previous_end))
            if end == len(layout.words):
                break
            previous_end = end
            start = end - self.overlap_words
        return tuple(chunks)
