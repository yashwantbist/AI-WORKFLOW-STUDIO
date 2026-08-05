"""Educational embedding and semantic-search package for Day 15."""

from .embeddings import SemanticTfidfEmbedder
from .sample_documents import Document, SAMPLE_DOCUMENTS
from .vector_store import InMemoryVectorStore, SearchResult

__all__ = [
    "Document",
    "InMemoryVectorStore",
    "SAMPLE_DOCUMENTS",
    "SearchResult",
    "SemanticTfidfEmbedder",
]
