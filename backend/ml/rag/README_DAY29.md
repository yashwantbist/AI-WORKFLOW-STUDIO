# Day 29 — End-to-End RAG Capstone Integration

## Request path

```text
Authenticated HTTP request
        ↓
Day 28 FastAPI route
        ↓
CapstoneRAGService
        ↓
Retriever
        ↓
Ranked chunks
        ↓
SafePromptBuilder
        ↓
Observable LLM provider
        ↓
Generated answer
        ↓
Optional online evaluator
        ↓
Request observation
        ↓
API response
```

## Important boundary

Live requests do not automatically have relevance labels or reference answers.

Therefore the online service does not fabricate:

- Recall@K
- Precision@K against unknown labels
- correctness against unavailable ground truth

Those remain offline evaluation concerns.

Online observability records only measurements actually available, including:

- request ID
- retrieved chunk count
- retrieval latency
- generation latency
- evaluation latency when evaluation runs
- total service latency
- input/output token usage when returned by the provider
- model name

## Ranking

The service preserves the retriever's order. The prompt builder emits Source 1,
Source 2, and so on in the same ranking order.

## Failure policy

- Retriever exception -> controlled dependency failure
- LLM exception -> controlled dependency failure
- Empty model answer -> controlled dependency failure
- Evaluator failure -> logged/observed but non-fatal for an otherwise valid answer
- No retrieved chunks -> controlled `(no context)` prompt; the model is instructed
  to state that evidence is insufficient

Day 28's API error handler can map dependency failures to safe HTTP 503 responses.

## Prompt injection boundary

Retrieved content is explicitly framed as untrusted data, not instructions.
This is a boundary, not a complete prompt-injection defense.

## Tests

```powershell
python -m pytest `
    backend/tests/test_rag_capstone_day29.py `
    backend/tests/test_production_rag_api_day28.py `
    backend/tests/test_observable_inference_day27.py `
    -v
```

All external systems should be faked/mocked in integration tests.
