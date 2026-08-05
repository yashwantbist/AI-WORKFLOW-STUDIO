"""Tests for the Day 15 embedding and semantic-search prototype."""

import pytest

from backend.ml.rag.embeddings import (
    SemanticTfidfEmbedder,
    extract_features,
)
from backend.ml.rag.sample_documents import SAMPLE_DOCUMENTS
from backend.ml.rag.semantic_search import build_default_store, format_results
from backend.ml.rag.similarity import cosine_similarity
from backend.ml.rag.vector_store import InMemoryVectorStore


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_different_dimensions() -> None:
    with pytest.raises(ValueError, match="same number of dimensions"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_semantic_aliases_normalize_related_words() -> None:
    assert "retrieval" in extract_features("find relevant documents")
    assert "semantic" in extract_features("meaningful results")
    assert "deployment" in extract_features("deploy with containers")


def test_embedder_requires_fit_before_embedding() -> None:
    embedder = SemanticTfidfEmbedder()

    with pytest.raises(RuntimeError, match=r"Call fit\(\)"):
        embedder.embed("attention")


def test_embeddings_have_shared_dimensions_and_unit_length() -> None:
    embedder = SemanticTfidfEmbedder().fit(
        ["attention connects tokens", "docker deploys containers"]
    )

    vectors = embedder.embed_many(
        ["focus on tokens", "container deployment"]
    )

    assert all(len(vector) == embedder.dimensions for vector in vectors)
    assert cosine_similarity(vectors[0], vectors[0]) == pytest.approx(1.0)


def test_attention_query_ranks_attention_document_first() -> None:
    store = build_default_store()

    results = store.search("How does attention work?", top_k=3)

    assert results[0].document.document_id == "attention"
    assert results[0].score > results[1].score


def test_synonym_query_can_find_container_deployment() -> None:
    store = build_default_store()

    results = store.search(
        "How can I deploy software using containers in the cloud?",
        top_k=1,
    )

    assert results[0].document.document_id == "docker"


def test_top_k_is_limited_to_available_documents() -> None:
    store = build_default_store()

    results = store.search("vectors", top_k=100)

    assert len(results) == len(SAMPLE_DOCUMENTS)


def test_search_rejects_empty_query_and_invalid_top_k() -> None:
    store = build_default_store()

    with pytest.raises(ValueError, match="Query"):
        store.search("   ")
    with pytest.raises(ValueError, match="top_k"):
        store.search("attention", top_k=0)


def test_vector_store_requires_documents_before_search() -> None:
    store = InMemoryVectorStore(SemanticTfidfEmbedder())

    with pytest.raises(RuntimeError, match="Add documents"):
        store.search("attention")


def test_formatted_results_include_rank_title_and_score() -> None:
    store = build_default_store()
    results = store.search("semantic vectors", top_k=1)

    output = format_results("semantic vectors", results)

    assert output.startswith('Query: "semantic vectors"')
    assert "1." in output
    assert "Score:" in output
    assert results[0].document.title in output
