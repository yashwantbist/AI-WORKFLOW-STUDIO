"""Local, deterministic text embeddings for learning semantic retrieval.

This module intentionally avoids network downloads and API keys. It creates
concept-aware TF-IDF vectors so learners can inspect every retrieval step.
Production systems normally replace this class with a trained neural embedding
model while keeping the vector-store and ranking pipeline largely unchanged.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
import math
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "what",
        "when",
        "which",
        "while",
        "with",
        "work",
        "works",
    }
)

# Words in each set are normalized to the same canonical concept. This small
# teaching lexicon demonstrates why meaning-aware retrieval can outperform
# exact keyword matching without pretending to be a production language model.
SEMANTIC_GROUPS: dict[str, frozenset[str]] = {
    "attention": frozenset(
        {
            "attend",
            "attending",
            "attention",
            "focus",
            "focused",
            "relationship",
            "relationships",
        }
    ),
    "embedding": frozenset(
        {
            "embedding",
            "embeddings",
            "representation",
            "representations",
            "vector",
            "vectors",
        }
    ),
    "retrieval": frozenset(
        {
            "find",
            "finding",
            "retrieve",
            "retrieved",
            "retrieval",
            "search",
            "searches",
        }
    ),
    "semantic": frozenset(
        {
            "meaning",
            "meaningful",
            "semantic",
            "semantics",
            "similar",
            "related",
            "relevant",
        }
    ),
    "token": frozenset(
        {
            "token",
            "tokens",
            "tokenization",
            "tokenizer",
            "subword",
            "subwords",
        }
    ),
    "transformer": frozenset(
        {
            "decoder",
            "encoder",
            "transformer",
            "transformers",
        }
    ),
    "generation": frozenset(
        {
            "answer",
            "generate",
            "generated",
            "generation",
            "output",
            "response",
        }
    ),
    "knowledge": frozenset(
        {
            "context",
            "document",
            "documents",
            "ground",
            "grounded",
            "knowledge",
        }
    ),
    "training": frozenset(
        {
            "backpropagation",
            "gradient",
            "gradients",
            "loss",
            "optimization",
            "optimizer",
            "train",
            "training",
        }
    ),
    "deployment": frozenset(
        {
            "cloud",
            "container",
            "containers",
            "deploy",
            "deployed",
            "deployment",
            "docker",
        }
    ),
    "testing": frozenset(
        {
            "integration",
            "pytest",
            "quality",
            "test",
            "testing",
            "tests",
            "unit",
            "validate",
        }
    ),
    "storage": frozenset(
        {
            "database",
            "index",
            "indexes",
            "persistence",
            "store",
            "storage",
            "stored",
        }
    ),
}

TOKEN_TO_CONCEPT = {
    token: concept
    for concept, group_tokens in SEMANTIC_GROUPS.items()
    for token in group_tokens
}


def tokenize(text: str) -> list[str]:
    """Lowercase text and return simple word and number tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def normalize_token(token: str) -> str:
    """Map known synonyms and word forms to one canonical concept."""
    return TOKEN_TO_CONCEPT.get(token, token)


def extract_features(text: str) -> list[str]:
    """Convert raw text into normalized unigram and bigram features."""
    normalized_tokens = [
        normalize_token(token)
        for token in tokenize(text)
        if token not in STOP_WORDS
    ]

    bigrams = [
        f"{left}::{right}"
        for left, right in zip(normalized_tokens, normalized_tokens[1:])
    ]
    return normalized_tokens + bigrams


class SemanticTfidfEmbedder:
    """Create inspectable TF-IDF vectors with lightweight semantic aliases."""

    def __init__(self) -> None:
        self._feature_to_id: dict[str, int] = {}
        self._inverse_document_frequency: tuple[float, ...] = ()
        self._is_fitted = False

    def fit(self, texts: Iterable[str]) -> "SemanticTfidfEmbedder":
        """Learn a shared feature space and IDF weights from documents."""
        document_features = [set(extract_features(text)) for text in texts]
        if not document_features:
            raise ValueError("At least one document is required to fit embeddings")

        all_features = sorted(
            feature
            for features in document_features
            for feature in features
        )
        unique_features = sorted(set(all_features))
        self._feature_to_id = {
            feature: index
            for index, feature in enumerate(unique_features)
        }

        number_of_documents = len(document_features)
        document_frequency = Counter(
            feature
            for features in document_features
            for feature in features
        )
        self._inverse_document_frequency = tuple(
            math.log(
                (1 + number_of_documents)
                / (1 + document_frequency[feature])
            )
            + 1.0
            for feature in unique_features
        )
        self._is_fitted = True
        return self

    @property
    def dimensions(self) -> int:
        """Return the number of learned embedding dimensions."""
        self._require_fitted()
        return len(self._feature_to_id)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names ordered by their vector dimension."""
        self._require_fitted()
        ordered = sorted(
            self._feature_to_id.items(),
            key=lambda item: item[1],
        )
        return tuple(feature for feature, _ in ordered)

    def embed(self, text: str) -> tuple[float, ...]:
        """Convert one text into a normalized numeric vector."""
        self._require_fitted()
        feature_counts = Counter(extract_features(text))
        vector = [0.0] * self.dimensions

        for feature, count in feature_counts.items():
            feature_id = self._feature_to_id.get(feature)
            if feature_id is None:
                continue

            term_frequency = 1.0 + math.log(count)
            vector[feature_id] = (
                term_frequency
                * self._inverse_document_frequency[feature_id]
            )

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return tuple(vector)

        return tuple(value / magnitude for value in vector)

    def embed_many(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed multiple texts in the already fitted feature space."""
        return tuple(self.embed(text) for text in texts)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before requesting embeddings")
