# Day 20 — RAG Evaluation v1

Day 20 extends the Day 19 evaluation harness with explicit retrieval edge-case
behavior and deterministic claim-level groundedness.

## Precision@K and Recall@K

Given:

```text
Retrieved top 5:
c1, c2, c7, c3, c9

Relevant:
c1, c2, c3, c4
```

```text
Precision@5 = 3 / 5 = 0.60
Recall@5    = 3 / 4 = 0.75
```

Precision measures how clean the retrieved set is.

Recall measures how much of the labeled relevant evidence was recovered.

## Edge cases

- Empty retrieval -> precision and recall are `0.0`.
- Zero relevant documents -> recall is defined as `0.0`.
- If `k` is larger than returned results, precision uses the returned count.
- Duplicate IDs consume Top-K positions but do not receive repeated relevance
  credit.

Example:

```text
retrieved = ["c1", "c1", "c2"]
relevant  = {"c1", "c2", "c3"}

Precision@3 = 2/3
Recall@3    = 2/3
```

## Claim-level groundedness

Represent factual claims explicitly:

```python
claims = [
    {
        "claim": "The Pro plan costs $20/month.",
        "supported": True,
        "evidence_ids": ["pricing-1"],
    },
    {
        "claim": "The Pro plan includes 100 GB storage.",
        "supported": True,
        "evidence_ids": ["pricing-2"],
    },
    {
        "claim": "The Pro plan is the most popular plan.",
        "supported": False,
    },
]
```

Then:

```text
groundedness = supported claims / total claims
             = 2 / 3
             = 0.667
```

This is partially grounded.

The support labels are deterministic inputs. Day 20 does not add an LLM judge
or pretend this code can automatically prove semantic entailment.

## Run tests

```powershell
python -m pytest `
  backend/tests/test_retrieval_metrics.py `
  backend/tests/test_groundedness.py `
  -v
```

Then run the broader Day 19/20 evaluation tests:

```powershell
python -m pytest `
  backend/tests/test_retrieval_metrics.py `
  backend/tests/test_groundedness.py `
  backend/tests/test_rag_evaluator.py `
  -q --tb=short
```

Do not claim repository-wide tests pass until you run them in the actual repo.
