"""Tests for the Day 16 chunking pipeline."""
from pathlib import Path
import pytest
from backend.ml.rag.chunk_visualizer import format_chunks, render_html
from backend.ml.rag.chunker import FixedSizeChunker, SentenceChunker
from backend.ml.rag.chunking_demo import create_chunker, run_strategy
from backend.ml.rag.chunking_sample_document import SAMPLE_CHUNKING_DOCUMENT
from backend.ml.rag.metadata import DocumentPage, SourceDocument, analyze_document
from backend.ml.rag.recursive_chunker import RecursiveChunker

def make_document(text: str) -> SourceDocument:
    return SourceDocument("test-document", "Test Document", "memory://test", (DocumentPage(1, text),))

def test_fixed_chunks_respect_size() -> None:
    chunks = FixedSizeChunker(10, 2).chunk(make_document(" ".join(f"word{i}" for i in range(25))))
    assert [c.metadata.word_count for c in chunks] == [10, 10, 9]

def test_fixed_chunks_have_exact_overlap() -> None:
    chunks = FixedSizeChunker(10, 3).chunk(make_document(" ".join(f"word{i}" for i in range(24))))
    assert chunks[1].metadata.overlap_words == 3
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]

def test_zero_overlap() -> None:
    chunks = FixedSizeChunker(3, 0).chunk(make_document("one two three four five six"))
    assert [c.text for c in chunks] == ["one two three", "four five six"]

def test_sentence_chunker_prefers_sentence_end() -> None:
    chunks = SentenceChunker(9, 1).chunk(make_document("One short sentence. Another sentence has several words here. The final sentence is readable."))
    assert chunks[0].text.endswith(".")
    assert all(c.metadata.word_count <= 9 for c in chunks)

def test_recursive_records_sections() -> None:
    chunks = RecursiveChunker(70, 10).chunk(SAMPLE_CHUNKING_DOCUMENT)
    sections = {section for chunk in chunks for section in chunk.metadata.sections}
    assert "Self-Attention" in sections
    assert "Retrieval-Augmented Generation" in sections

def test_metadata_tracks_pages_and_ids() -> None:
    chunks = FixedSizeChunker(80, 10).chunk(SAMPLE_CHUNKING_DOCUMENT)
    assert chunks[0].metadata.chunk_id.startswith("transformer-rag-guide-fixed-")
    assert all(c.metadata.page_end >= c.metadata.page_start >= 1 for c in chunks)

def test_layout_has_boundaries() -> None:
    layout = analyze_document(SAMPLE_CHUNKING_DOCUMENT)
    assert layout.words and layout.sentence_ends and layout.paragraph_ends and layout.section_ends
    assert len(layout.page_ends) == 2

@pytest.mark.parametrize("size,overlap", [(0,0),(10,-1),(10,10),(10,11)])
def test_invalid_settings(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size, overlap)

def test_empty_page_returns_no_chunks() -> None:
    assert RecursiveChunker(20, 5).chunk(make_document("   \n\n   ")) == ()

def test_html_marks_overlap(tmp_path: Path) -> None:
    chunks = FixedSizeChunker(20, 4).chunk(SAMPLE_CHUNKING_DOCUMENT)
    path = render_html(chunks, tmp_path / "chunks.html")
    html = path.read_text(encoding="utf-8")
    assert "Document Chunk Visualization" in html and "<mark>" in html

def test_terminal_report_includes_metadata() -> None:
    report = format_chunks(SentenceChunker(50, 5).chunk(SAMPLE_CHUNKING_DOCUMENT))
    assert "Strategy: sentence" in report and "Pages:" in report and "Sections:" in report

@pytest.mark.parametrize("strategy", ["fixed","sentence","recursive"])
def test_factory(strategy: str) -> None:
    chunker = create_chunker(strategy, 50, 5)
    assert chunker.chunk_size_words == 50 and chunker.overlap_words == 5

@pytest.mark.parametrize("strategy", ["fixed","sentence","recursive"])
def test_every_strategy_runs(strategy: str) -> None:
    chunks = run_strategy(strategy, 60, 10)
    assert chunks and all(c.metadata.strategy == strategy for c in chunks)

def test_document_requires_pages() -> None:
    with pytest.raises(ValueError):
        SourceDocument("empty", "Empty", "memory://empty", ())
