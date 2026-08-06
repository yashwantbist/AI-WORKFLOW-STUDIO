"""Metadata records stored alongside sequential FAISS vector IDs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class IndexedChunk:
    """Text and traceability metadata corresponding to one indexed vector."""

    chunk_id: str
    text: str
    document_id: str
    document_title: str
    source: str
    page_start: int
    page_end: int
    sections: tuple[str, ...] = ()
    strategy: str = "unknown"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty")
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("page range is invalid")

    @classmethod
    def from_document_chunk(cls, chunk: Any) -> "IndexedChunk":
        """Convert the previous day's DocumentChunk without tight coupling."""

        metadata = chunk.metadata
        return cls(
            chunk_id=metadata.chunk_id,
            text=chunk.text,
            document_id=metadata.document_id,
            document_title=metadata.document_title,
            source=metadata.source,
            page_start=metadata.page_start,
            page_end=metadata.page_end,
            sections=tuple(metadata.sections),
            strategy=metadata.strategy,
            extra={
                "chunk_index": metadata.chunk_index,
                "start_word": metadata.start_word,
                "end_word": metadata.end_word,
                "word_count": metadata.word_count,
                "overlap_words": metadata.overlap_words,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = list(self.sections)
        data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndexedChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data["text"]),
            document_id=str(data["document_id"]),
            document_title=str(data["document_title"]),
            source=str(data["source"]),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            sections=tuple(str(value) for value in data.get("sections", [])),
            strategy=str(data.get("strategy", "unknown")),
            extra=dict(data.get("extra", {})),
        )


class MetadataStore:
    """Map each sequential FAISS vector ID to one immutable chunk record."""

    SCHEMA_VERSION = 1

    def __init__(self, records: Iterable[IndexedChunk] = ()) -> None:
        self._records = list(records)
        chunk_ids = [record.chunk_id for record in self._records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk IDs must be unique")

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[IndexedChunk, ...]:
        return tuple(self._records)

    def add(self, records: Iterable[IndexedChunk]) -> tuple[int, ...]:
        additions = list(records)
        existing_ids = {record.chunk_id for record in self._records}
        duplicate_ids = [
            record.chunk_id
            for record in additions
            if record.chunk_id in existing_ids
        ]
        if duplicate_ids:
            raise ValueError(f"duplicate chunk IDs: {sorted(set(duplicate_ids))}")

        first_id = len(self._records)
        self._records.extend(additions)
        return tuple(range(first_id, len(self._records)))

    def get(self, vector_id: int) -> IndexedChunk:
        if vector_id < 0 or vector_id >= len(self._records):
            raise IndexError(f"metadata vector ID is out of range: {vector_id}")
        return self._records[vector_id]

    def matching_ids(
        self,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[int, ...]:
        """Return vector IDs matching document, section, page, or extra fields."""

        if not filters:
            return tuple(range(len(self._records)))

        def matches(record: IndexedChunk) -> bool:
            for key, expected in filters.items():
                if key == "section":
                    if not any(
                        str(expected).lower() in section.lower()
                        for section in record.sections
                    ):
                        return False
                elif key == "page":
                    page = int(expected)
                    if not record.page_start <= page <= record.page_end:
                        return False
                elif hasattr(record, key):
                    actual = getattr(record, key)
                    if actual != expected:
                        return False
                elif record.extra.get(key) != expected:
                    return False
            return True

        return tuple(
            vector_id
            for vector_id, record in enumerate(self._records)
            if matches(record)
        )

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "records": [record.to_dict() for record in self._records],
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "MetadataStore":
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"metadata file not found: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported metadata schema version")
        return cls(
            IndexedChunk.from_dict(record)
            for record in payload["records"]
        )
