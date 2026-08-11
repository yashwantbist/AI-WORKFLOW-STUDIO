# Day 21 — Evaluated RAG Inference Pipeline

Day 21 integrates evaluation into the RAG request lifecycle without pretending
that every production request has ground-truth relevance labels.

## Lifecycle

```text
Question
   |
   v
Retriever --------------------> retrieval telemetry
   |                             IDs, scores, count, latency
   |
   +---------------------------> optional Precision@K / Recall@K
   |                             only when relevant_ids exist
   v
Evidence
   |
   v
Generator
   |
   v
Answer -----------------------> optional claim groundedness
                                 only when claim labels exist
```

## Telemetry vs evaluation

Telemetry can be observed for ordinary production traffic:

- requested `k`;
- retrieved chunk IDs;
- scores;
- retrieval count;
- context count;
- retrieval latency.

Retrieval evaluation requires labelled ground truth.

For example, Recall@K needs:

```text
relevant chunks retrieved
--------------------------
all relevant chunks
```

If the application does not know the complete relevant set, the denominator is
unknown and Recall@K must not be fabricated.

Therefore:

```python
pipeline.run(
    "What is the refund policy?",
    k=5,
)
```

returns retrieval telemetry but no Precision@K or Recall@K.

A labelled offline/evaluation request can use:

```python
pipeline.run(
    "What is the refund policy?",
    k=5,
    relevant_ids={"refund-1", "refund-4"},
)
```

and receives deterministic retrieval metrics.

## One retrieval event

Day 19's standalone evaluator performed raw retrieval and then called a RAG
pipeline that retrieved again.

Day 21 adds:

```python
RAGPipeline.answer_from_retrieved(...)
```

so the evaluated pipeline can:

```text
retrieve once
    |
    +--> evaluate that retrieval
    |
    +--> generate from that exact retrieval
```

This prevents evaluation and generation from observing different search events.

The existing `RAGPipeline.answer()` API remains available and still performs
normal retrieve-then-generate inference.

## Structured result

Conceptually:

```json
{
  "answer": {
    "answer": "Refunds are available for 30 days [Source 1].",
    "sources": [],
    "grounded": true,
    "insufficient_context": false,
    "retrieved_count": 2,
    "used_context_count": 2
  },
  "retrieval": {
    "requested_k": 2,
    "retrieved_count": 2,
    "used_context_count": 2,
    "retrieval_latency_ms": 0.42,
    "chunk_ids": ["c1", "c2"],
    "items": []
  },
  "evaluation": {
    "retrieval": {
      "precision_at_k": 0.5,
      "recall_at_k": 0.5
    },
    "groundedness": null
  }
}
```

For an unlabelled production query:

```json
{
  "evaluation": null
}
```

The pipeline does not fill unavailable metrics with guessed values.

## Groundedness

Day 20's deterministic claim labels can also be attached:

```python
claim_labels=[
    {
        "claim": "Refunds are available for 30 days.",
        "supported": True,
        "evidence_ids": ["c1"],
    },
    {
        "claim": "Refunds are instant.",
        "supported": False,
    },
]
```

This produces claim-level groundedness metadata.

It is still a deterministic labelled evaluation. Day 21 does not claim to
automatically extract claims or prove semantic entailment.

## Empty retrieval

Empty retrieval does not crash or call the generator.

The normal insufficient-context response is returned, retrieval telemetry shows
zero results, and labelled retrieval metrics can still report the retrieval
failure.

## Tests

```powershell
python -m pytest `
  backend/tests/test_evaluated_rag_pipeline.py `
  backend/tests/test_rag_pipeline_day21.py `
  -v
```

Then run the existing Day 20 evaluation tests:

```powershell
python -m pytest `
  backend/tests/test_retrieval_metrics.py `
  backend/tests/test_groundedness.py `
  backend/tests/test_rag_evaluator.py `
  backend/tests/test_evaluated_rag_pipeline.py `
  backend/tests/test_rag_pipeline_day21.py `
  -q --tb=short
```

Only claim repository-wide success after running the suite in the actual
repository.

## Production observability boundary

Good production metrics include:

- retrieval latency;
- error rate;
- number of chunks returned;
- number of chunks used after score filtering;
- generation latency;
- model/provider failures;
- request volume.

Metrics that require labels include:

- Recall@K;
- Precision@K against a curated relevant set;
- reference-answer scores;
- human-labelled faithfulness;
- claim-level support labels.

A dashboard should never present unavailable labelled metrics as though they
were observed from normal traffic.
