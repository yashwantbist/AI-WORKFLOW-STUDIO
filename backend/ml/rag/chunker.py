"""Fixed-size and sentence-aware chunkers."""
from dataclasses import dataclass
from .metadata import DocumentChunk, DocumentLayout, SourceDocument, analyze_document, build_chunk

def validate_chunk_configuration(chunk_size_words: int, overlap_words: int) -> None:
    if chunk_size_words < 1:
        raise ValueError("chunk_size_words must be at least 1")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

@dataclass(frozen=True)
class FixedSizeChunker:
    chunk_size_words: int = 100
    overlap_words: int = 20
    def __post_init__(self) -> None:
        validate_chunk_configuration(self.chunk_size_words, self.overlap_words)
    def chunk(self, document: SourceDocument) -> tuple[DocumentChunk, ...]:
        layout = analyze_document(document)
        if not layout.words:
            return ()
        chunks: list[DocumentChunk] = []
        start = previous_end = 0
        while start < len(layout.words):
            end = min(start + self.chunk_size_words, len(layout.words))
            chunks.append(build_chunk(document, layout, start_word=start, end_word=end, strategy="fixed", chunk_index=len(chunks)+1, previous_end_word=previous_end))
            if end == len(layout.words):
                break
            previous_end = end
            start = end - self.overlap_words
        return tuple(chunks)

@dataclass(frozen=True)
class SentenceChunker:
    chunk_size_words: int = 100
    overlap_words: int = 20
    def __post_init__(self) -> None:
        validate_chunk_configuration(self.chunk_size_words, self.overlap_words)
    @staticmethod
    def _end(layout: DocumentLayout, start: int, target: int) -> int:
        choices = [boundary for boundary in layout.sentence_ends if start < boundary <= target]
        return max(choices) if choices else target
    def chunk(self, document: SourceDocument) -> tuple[DocumentChunk, ...]:
        layout = analyze_document(document)
        if not layout.words:
            return ()
        chunks: list[DocumentChunk] = []
        start = previous_end = 0
        while start < len(layout.words):
            target = min(start + self.chunk_size_words, len(layout.words))
            end = self._end(layout, start, target)
            chunks.append(build_chunk(document, layout, start_word=start, end_word=end, strategy="sentence", chunk_index=len(chunks)+1, previous_end_word=previous_end))
            if end == len(layout.words):
                break
            previous_end = end
            start = end - self.overlap_words
        return tuple(chunks)
