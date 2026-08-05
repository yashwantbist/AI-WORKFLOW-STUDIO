"""Data models and layout analysis for document chunking."""
from __future__ import annotations
from dataclasses import dataclass
import re

WORD_PATTERN = re.compile(r"\S+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")

@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str
    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be at least 1")

@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    source: str
    pages: tuple[DocumentPage, ...]
    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.title.strip() or not self.source.strip():
            raise ValueError("document_id, title, and source cannot be empty")
        if not self.pages:
            raise ValueError("pages cannot be empty")
        numbers = [page.page_number for page in self.pages]
        if numbers != sorted(numbers):
            raise ValueError("pages must be ordered by page_number")

@dataclass(frozen=True)
class WordUnit:
    text: str
    page_number: int
    section: str

@dataclass(frozen=True)
class DocumentLayout:
    words: tuple[WordUnit, ...]
    sentence_ends: frozenset[int]
    paragraph_ends: frozenset[int]
    section_ends: frozenset[int]
    page_ends: frozenset[int]

@dataclass(frozen=True)
class ChunkMetadata:
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    strategy: str
    chunk_index: int
    page_start: int
    page_end: int
    sections: tuple[str, ...]
    start_word: int
    end_word: int
    word_count: int
    overlap_words: int

@dataclass(frozen=True)
class DocumentChunk:
    text: str
    metadata: ChunkMetadata

def _sentences(text: str) -> list[str]:
    text = text.strip()
    return [part.strip() for part in SENTENCE_PATTERN.split(text) if part.strip()] if text else []

def _append(words: list[WordUnit], text: str, page: int, section: str) -> None:
    words.extend(WordUnit(m.group(0), page, section) for m in WORD_PATTERN.finditer(text))

def analyze_document(document: SourceDocument) -> DocumentLayout:
    words: list[WordUnit] = []
    sentence_ends: set[int] = set()
    paragraph_ends: set[int] = set()
    section_ends: set[int] = set()
    page_ends: set[int] = set()
    section = "Introduction"

    for page in document.pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]
        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            heading = HEADING_PATTERN.match(lines[0]) if lines else None
            if heading:
                if words:
                    section_ends.add(len(words))
                section = heading.group(1).strip()
                _append(words, section, page.page_number, section)
                sentence_ends.add(len(words)); paragraph_ends.add(len(words))
                body = " ".join(lines[1:])
            else:
                body = " ".join(lines)
            for sentence in _sentences(body):
                _append(words, sentence, page.page_number, section)
                sentence_ends.add(len(words))
            if body:
                paragraph_ends.add(len(words))
        if words:
            page_ends.add(len(words))

    if words:
        total = len(words)
        sentence_ends.add(total); paragraph_ends.add(total); section_ends.add(total); page_ends.add(total)
    return DocumentLayout(tuple(words), frozenset(sentence_ends), frozenset(paragraph_ends), frozenset(section_ends), frozenset(page_ends))

def build_chunk(document: SourceDocument, layout: DocumentLayout, *, start_word: int, end_word: int, strategy: str, chunk_index: int, previous_end_word: int) -> DocumentChunk:
    selected = layout.words[start_word:end_word]
    if not selected:
        raise ValueError("cannot create an empty chunk")
    sections = tuple(dict.fromkeys(word.section for word in selected))
    metadata = ChunkMetadata(
        chunk_id=f"{document.document_id}-{strategy}-{chunk_index:03d}",
        document_id=document.document_id,
        document_title=document.title,
        source=document.source,
        strategy=strategy,
        chunk_index=chunk_index,
        page_start=selected[0].page_number,
        page_end=selected[-1].page_number,
        sections=sections,
        start_word=start_word,
        end_word=end_word,
        word_count=len(selected),
        overlap_words=max(0, previous_end_word - start_word),
    )
    return DocumentChunk(" ".join(word.text for word in selected), metadata)
