"""In-memory storage and top-k retrieval for document embeddings."""

from dataclasses import dataclass

try:
    from .embeddings import SemanticTfidfEmbedder
    from .sample_documents import Document
    from .similarity import cosine_similarity
except ImportError:  # Supports direct script-style imports while learning.
    from embeddings import SemanticTfidfEmbedder
    from sample_documents import Document
    from similarity import cosine_similarity


@dataclass(frozen=True)
class VectorRecord:
    """A document stored beside its precomputed embedding."""

    document: Document
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class SearchResult:
    """One document returned from a similarity search."""

    document: Document
    score: float


class InMemoryVectorStore:
    """Store embeddings in RAM and scan them for the closest documents."""

    def __init__(self, embedder: SemanticTfidfEmbedder) -> None:
        self._embedder = embedder
        self._records: tuple[VectorRecord, ...] = ()

    def add_documents(self, documents: tuple[Document, ...]) -> None:
        """Fit the embedder and replace the store with supplied documents."""
        if not documents:
            raise ValueError("At least one document is required")

        searchable_texts = tuple(
            self._searchable_text(document)
            for document in documents
        )
        self._embedder.fit(searchable_texts)
        embeddings = self._embedder.embed_many(searchable_texts)
        self._records = tuple(
            VectorRecord(document=document, embedding=embedding)
            for document, embedding in zip(documents, embeddings)
        )

    def search(self, query: str, top_k: int = 5) -> tuple[SearchResult, ...]:
        """Return the top-k documents ordered by cosine similarity."""
        if not self._records:
            raise RuntimeError("Add documents before searching")
        if not query.strip():
            raise ValueError("Query must contain non-whitespace text")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = self._embedder.embed(query)
        scored_results = [
            SearchResult(
                document=record.document,
                score=cosine_similarity(query_embedding, record.embedding),
            )
            for record in self._records
        ]
        scored_results.sort(
            key=lambda result: (-result.score, result.document.document_id)
        )
        return tuple(scored_results[: min(top_k, len(scored_results))])

    @property
    def size(self) -> int:
        """Return the number of stored document vectors."""
        return len(self._records)

    @property
    def dimensions(self) -> int:
        """Return the dimension of each vector in the store."""
        return self._embedder.dimensions

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Expose a query vector for the CLI learning display."""
        if not self._records:
            raise RuntimeError("Add documents before embedding a query")
        return self._embedder.embed(query)

    @staticmethod
    def _searchable_text(document: Document) -> str:
        tags = " ".join(document.tags)
        return f"{document.title} {document.text} {tags}"
