# Day 15 Line-by-Line Code Guide

This guide explains every meaningful line or small group of inseparable lines.
Blank lines are only visual separators, so they are not discussed individually.

## `sample_documents.py`

```python
"""Small knowledge base used by the semantic-search demonstration."""
```

The module docstring states the file's responsibility.

```python
from dataclasses import dataclass
```

representation methods for a data-only class.

```python
@dataclass(frozen=True)
class Document:
```

`@dataclass` removes repetitive constructor code. `frozen=True` prevents a
stored document from being changed accidentally after creation.

```python
    document_id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()
```

These type-annotated fields define one document. `tags` defaults to an empty,
immutable tuple.

```python
SAMPLE_DOCUMENTS = (
```

Creates an immutable tuple containing the small knowledge base.

Each `Document(...)` call supplies a stable ID, human-readable title, searchable
text, and metadata tags. Adjacent string literals inside parentheses are joined
by Python automatically, allowing readable wrapped lines without newline
characters entering the document text.

## `similarity.py`

```python
from collections.abc import Sequence
import math
```

`Sequence` describes list-like vectors. `math` provides square root.

```python
def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
```

Defines a typed function accepting two numeric vectors and returning one number.

```python
    if len(left) != len(right):
        raise ValueError("Vectors must have the same number of dimensions")
```

Cosine similarity is undefined for differently sized coordinate spaces, so the
function fails early with a useful message.

```python
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
```

`zip` pairs matching dimensions, each pair is multiplied, and `sum` combines
the products into the dot product.

```python
def vector_magnitude(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))
```

Squares every coordinate, adds the squares, and takes the square root to obtain
the vector's length.

```python
left_magnitude = vector_magnitude(left)
right_magnitude = vector_magnitude(right)
```

Calculates both vector lengths once and stores them for reuse.

```python
if left_magnitude == 0.0 or right_magnitude == 0.0:
    return 0.0
```

A zero vector has no direction. Returning zero keeps unknown queries safe and
predictable.

```python
similarity = dot_product(left, right) / (left_magnitude * right_magnitude)
```

Divides shared direction by both lengths so vector size does not dominate the
comparison.

```python
return max(-1.0, min(1.0, similarity))
```

Clamps tiny floating-point rounding errors into cosine's valid range.

## `embeddings.py`

```python
from collections import Counter
from collections.abc import Iterable, Sequence
import math
import re
```

`Counter` counts features, collection types improve type hints, `math` supports
logarithms and normalization, and `re` tokenizes text.

```python
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
```

Compiles one reusable regular expression that recognizes lowercase words,
numbers, and simple contractions.

```python
STOP_WORDS = frozenset({...})
```

Stores frequent low-information words in an immutable set. Set membership is
fast, and immutability prevents accidental changes.

```python
SEMANTIC_GROUPS: dict[str, frozenset[str]] = {...}
```

Maps a canonical concept to related forms and synonyms. This is the teaching
mechanism that lets `find` and `retrieve`, for example, share meaning.

```python
TOKEN_TO_CONCEPT = {
    token: concept
    for concept, group_tokens in SEMANTIC_GROUPS.items()
    for token in group_tokens
}
```

Inverts the semantic dictionary so a token can be normalized with one fast
lookup.

```python
def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
```

Lowercases input and returns every regex match as a token.

```python
def normalize_token(token: str) -> str:
    return TOKEN_TO_CONCEPT.get(token, token)
```

Returns the shared concept when known; otherwise preserves the original token.

```python
normalized_tokens = [
    normalize_token(token)
    for token in tokenize(text)
    if token not in STOP_WORDS
]
```

This list comprehension tokenizes, removes stop words, and normalizes semantic
aliases in one pass.

```python
bigrams = [
    f"{left}::{right}"
    for left, right in zip(normalized_tokens, normalized_tokens[1:])
]
```

Pairs each token with its next token. Bigrams preserve some phrase information,
such as `cosine::semantic`, that isolated words would lose.

```python
return normalized_tokens + bigrams
```

Uses both individual concepts and adjacent concept pairs as embedding features.

```python
class SemanticTfidfEmbedder:
```

Groups fitting and embedding state into one reusable object.

```python
self._feature_to_id: dict[str, int] = {}
self._inverse_document_frequency: tuple[float, ...] = ()
self._is_fitted = False
```

Initializes an empty feature-index mapping, no IDF weights, and a guard showing
that the model has not learned a document space yet. Leading underscores mark
implementation details.

```python
document_features = [set(extract_features(text)) for text in texts]
```

Extracts each document's unique features. A set ensures a feature counts at
most once toward document frequency.

```python
if not document_features:
    raise ValueError(...)
```

Rejects fitting with no documents because no vector space could be learned.

```python
all_features = sorted(
    feature
    for features in document_features
    for feature in features
)
unique_features = sorted(set(all_features))
```

Flattens features from all documents, removes duplicates, and sorts them so the
same corpus always produces the same dimension order.

```python
self._feature_to_id = {
    feature: index
    for index, feature in enumerate(unique_features)
}
```

Assigns every feature one integer vector position.

```python
number_of_documents = len(document_features)
```

Stores corpus size for IDF calculation.

```python
document_frequency = Counter(
    feature
    for features in document_features
    for feature in features
)
```

Counts how many documents contain each feature.

```python
math.log((1 + number_of_documents) / (1 + document_frequency[feature])) + 1.0
```

Computes smoothed inverse document frequency. Rare features receive more weight,
while frequent features receive less. Added ones avoid division by zero and
keep weights positive.

```python
self._is_fitted = True
return self
```

Marks the object ready and returns it, enabling chained code such as
`SemanticTfidfEmbedder().fit(texts)`.

```python
@property
def dimensions(self) -> int:
```

Makes vector length readable as `embedder.dimensions` instead of a method call.

```python
self._require_fitted()
```

Prevents using incomplete learned state.

```python
feature_counts = Counter(extract_features(text))
vector = [0.0] * self.dimensions
```

Counts query/document features and creates a dense zero vector with the shared
corpus dimension.

```python
for feature, count in feature_counts.items():
```

Visits every feature appearing in the current text.

```python
feature_id = self._feature_to_id.get(feature)
if feature_id is None:
    continue
```

Looks up the matching dimension and ignores features unseen during fitting.

```python
term_frequency = 1.0 + math.log(count)
```

Uses log-scaled term frequency so repeated words help without dominating.

```python
vector[feature_id] = term_frequency * self._inverse_document_frequency[feature_id]
```

Places the final TF-IDF weight into the correct coordinate.

```python
magnitude = math.sqrt(sum(value * value for value in vector))
```

Calculates vector length.

```python
if magnitude == 0.0:
    return tuple(vector)
```

Returns a safe zero vector when no query features occur in the fitted corpus.

```python
return tuple(value / magnitude for value in vector)
```

Normalizes the vector to length one and returns an immutable tuple.

```python
def embed_many(...):
    return tuple(self.embed(text) for text in texts)
```

Reuses `embed` for every text and returns all vectors.

```python
def _require_fitted(self) -> None:
    if not self._is_fitted:
        raise RuntimeError(...)
```

Centralizes the guard against calling embedding operations too early.

## `vector_store.py`

```python
from dataclasses import dataclass
```

Imports the utility used for immutable record/result objects.

```python
try:
    from .embeddings ...
except ImportError:
    from embeddings ...
```

Relative imports support module execution from the project root. The fallback
supports direct execution while learning from inside the file directory.

```python
@dataclass(frozen=True)
class VectorRecord:
    document: Document
    embedding: tuple[float, ...]
```

Stores a document beside its precomputed vector.

```python
@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
```

Represents one ranked result without exposing internal storage details.

```python
self._embedder = embedder
self._records: tuple[VectorRecord, ...] = ()
```

Keeps the supplied embedding model and starts with an empty immutable record
collection.

```python
if not documents:
    raise ValueError(...)
```

Prevents creating an unusable empty store.

```python
searchable_texts = tuple(
    self._searchable_text(document)
    for document in documents
)
```

Combines title, body, and tags so all useful document fields can influence
retrieval.

```python
self._embedder.fit(searchable_texts)
embeddings = self._embedder.embed_many(searchable_texts)
```

Learns one shared coordinate space and embeds every document within it.

```python
self._records = tuple(
    VectorRecord(document=document, embedding=embedding)
    for document, embedding in zip(documents, embeddings)
)
```

Pairs every document with the embedding generated at the same position.

```python
if not self._records: ...
if not query.strip(): ...
if top_k < 1: ...
```

Validates that searching is possible and arguments are meaningful.

```python
query_embedding = self._embedder.embed(query)
```

Transforms the user's query into the exact same vector space as documents.

```python
scored_results = [
    SearchResult(
        document=record.document,
        score=cosine_similarity(query_embedding, record.embedding),
    )
    for record in self._records
]
```

Performs exhaustive search: compare the query with every stored vector and keep
its document and score together.

```python
scored_results.sort(
    key=lambda result: (-result.score, result.document.document_id)
)
```

The negative score sorts highest similarity first. Document ID breaks ties
deterministically.

```python
return tuple(scored_results[: min(top_k, len(scored_results))])
```

Returns at most `top_k` results and never indexes beyond available documents.

```python
@property
def size(self) -> int:
```

Exposes record count without allowing callers to mutate records.

```python
@staticmethod
def _searchable_text(document: Document) -> str:
```

Marks a helper that does not use object state and combines searchable fields.

## `semantic_search.py`

```python
import argparse
from collections.abc import Sequence
```

`argparse` builds the CLI. `Sequence` types both result collections and optional
argument lists used by tests.

```python
def build_default_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(SemanticTfidfEmbedder())
    store.add_documents(SAMPLE_DOCUMENTS)
    return store
```

Constructs the embedder, injects it into a store, indexes sample documents, and
returns a ready application object.

```python
def format_results(query: str, results: Sequence[SearchResult]) -> str:
```

Separates formatting from printing so output can be tested.

```python
lines = [f'Query: "{query}"', ""]
```

Starts the output with the original query and a blank line.

```python
for rank, result in enumerate(results, start=1):
```

Walks through already sorted results and creates human-friendly ranks beginning
at one.

```python
lines.extend([...])
```

Adds title, formatted score, ID, text, and spacing for each result.

```python
return "\n".join(lines).rstrip()
```

Joins all output lines and removes the final unnecessary blank line.

```python
def create_argument_parser() -> argparse.ArgumentParser:
```

Encapsulates CLI definition so it can be inspected or tested independently.

Each `parser.add_argument(...)` block defines one user option: the query,
`--top-k`, `--show-vector`, or `--list-documents`.

```python
def main(arguments: Sequence[str] | None = None) -> int:
```

Accepts normal command-line arguments when `None`, but tests may inject a list.
The integer return value is the process exit status.

```python
options = parser.parse_args(arguments)
```

Validates and converts raw strings into typed options.

```python
if options.list_documents:
```

Handles the listing mode before requiring a search query.

```python
if not options.query:
    parser.error(...)
```

Produces standard CLI help and a nonzero exit when a required query is absent.

```python
store = build_default_store()
results = store.search(options.query, top_k=options.top_k)
```

Builds the pipeline and retrieves ranked results.

```python
print(f"Knowledge base: ...")
print(format_results(...))
```

Shows index metadata and the formatted rankings.

```python
if options.show_vector:
```

Enables an optional learning view rather than cluttering normal output.

```python
non_zero_values = [
    (index, value)
    for index, value in enumerate(query_vector)
    if value != 0.0
]
```

Finds only active vector coordinates, making a high-dimensional vector easier
to inspect.

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Runs the CLI only when the file is executed, not when imported by tests. Raising
`SystemExit` passes `main`'s status back to the operating system.

## `test_semantic_search.py`

```python
import pytest
```

Provides approximate floating-point comparisons and exception assertions.

Each test follows Arrange–Act–Assert:

- Arrange inputs or build a store.
- Act by calling one behavior.
- Assert the expected score, exception, vector property, ranking, or output.

`pytest.approx` handles harmless floating-point rounding. `pytest.raises`
confirms both that an error occurs and that its message guides the developer.
The ranking tests prove the components work together, not merely in isolation.
