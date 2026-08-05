"""Terminal and HTML visualizations for chunks."""
from collections.abc import Sequence
from html import escape
from pathlib import Path
from .metadata import DocumentChunk

def format_chunks(chunks: Sequence[DocumentChunk]) -> str:
    if not chunks:
        return "No chunks generated."
    lines: list[str] = []
    for chunk in chunks:
        m = chunk.metadata
        lines += [
            f"Chunk {m.chunk_index}: {m.chunk_id}",
            f"Strategy: {m.strategy} | Words: {m.word_count} | Overlap: {m.overlap_words}",
            f"Pages: {m.page_start}-{m.page_end} | Sections: {', '.join(m.sections)}",
            chunk.text,
            "-" * 80,
        ]
    return "\n".join(lines)

def render_html(chunks: Sequence[DocumentChunk], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for chunk in chunks:
        m = chunk.metadata
        words = chunk.text.split()
        marked = " ".join(f"<mark>{escape(word)}</mark>" for word in words[:m.overlap_words])
        body = " ".join(escape(word) for word in words[m.overlap_words:])
        text = " ".join(part for part in (marked, body) if part)
        cards.append(f'''<article><h2>Chunk {m.chunk_index}</h2><code>{escape(m.chunk_id)}</code>
<dl><dt>Strategy</dt><dd>{escape(m.strategy)}</dd><dt>Words</dt><dd>{m.word_count}</dd><dt>Overlap</dt><dd>{m.overlap_words}</dd><dt>Pages</dt><dd>{m.page_start}-{m.page_end}</dd><dt>Sections</dt><dd>{escape(', '.join(m.sections))}</dd></dl><p>{text}</p></article>''')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Document Chunk Visualization</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:2rem;line-height:1.6;background:#f7f7f8}}article{{background:white;border:1px solid #ddd;border-radius:12px;padding:1.25rem;margin:1rem 0}}dl{{display:grid;grid-template-columns:auto 1fr;gap:.25rem 1rem}}dt{{font-weight:700}}dd{{margin:0}}mark{{padding:.08rem .2rem}}</style></head><body><h1>Document Chunk Visualization</h1><p>Highlighted words are overlap copied from the previous chunk.</p>{''.join(cards) if cards else '<p>No chunks generated.</p>'}</body></html>'''
    path.write_text(html, encoding="utf-8")
    return path
