# Day 24 — Ranking-Aware RAG Evaluation

AI Workflow Studio already implements `hit_at_k()`, `reciprocal_rank()`,
aggregate `hit_rate_at_k`, and aggregate `mean_reciprocal_rank`.

Day 24 therefore focuses on verification and interpretation rather than
duplicating existing metric code.

## What each metric tells you

```text
Precision@K -> How much retrieved material is relevant?
Recall@K    -> How much labelled relevant evidence did we find?
Hit Rate@K  -> Did each query retrieve at least one relevant item?
MRR         -> How early did the first relevant item appear?
```

## Same Recall, different ranking

```text
Relevant = {c7}

Retriever A:
[c7, c2, c9, c4, c1]

Retriever B:
[c2, c9, c4, c1, c7]
```

At K=5:

```text
Recall@5(A) = 1.0
Recall@5(B) = 1.0

Hit@5(A) = 1
Hit@5(B) = 1

RR(A) = 1.0
RR(B) = 0.2
```

This is why Recall alone cannot measure ranking quality.

## MRR denominator

MRR averages reciprocal rank across all evaluated queries, including misses:

```text
RR values = [1.0, 0.5, 0.333, 0.0]

MRR =
(1.0 + 0.5 + 0.333 + 0.0) / 4
```

The missed query remains in the denominator with RR = 0.

## Run Day 24 tests

```powershell
python -m pytest `
    backend/tests/test_ranking_metrics_day24.py `
    -v
```

Then run your existing RAG regression suite.
