# Day 25 — Deterministic Claim-Level Groundedness Evaluation

Day 25 moves AI Workflow Studio from retrieval-only evaluation into generation
evaluation.

The project already had a basic `evaluate_groundedness()` helper that could
aggregate human-provided `supported=True/False` claim labels.

Day 25 extends that work by connecting labelled answer claims to the evidence
IDs actually supplied to the answer.

## v1 definition

For this project:

```text
Groundedness score =
supported evaluated claims
--------------------------
total evaluated claims
```

This is a project-specific deterministic metric, not a universal definition of
groundedness.

## Label format

A labelled claim looks like:

```json
{
  "claim": "Customers may request refunds within 30 days.",
  "supported_by": ["refund-01"]
}
```

The evaluator checks:

1. Does the labelled claim text appear in the generated answer?
2. Are all required `supported_by` evidence IDs available to the answer?

If both are true, the claim is supported.

A labelled claim with an empty `supported_by` list is an explicit unsupported
fixture:

```json
{
  "claim": "Customers may request refunds within 90 days.",
  "supported_by": []
}
```

## Why exact claim matching?

Day 25 deliberately avoids automatic NLP claim extraction or an LLM-as-judge.

Exact labelled claims give us:

- deterministic tests;
- no network dependency;
- no API cost;
- reproducible failures;
- explicit evidence IDs.

The limitation is that paraphrased generated answers may not match a labelled
claim exactly. A future semantic evaluator can address that.

## Empty-input behavior

The behavior is explicit:

```text
Empty answer
-> zero evaluated claims
-> groundedness score 0.0
-> fully_grounded = False

No claim labels on an EvaluationCase
-> groundedness = None
-> no groundedness score is fabricated

Claim present but evidence missing
-> unsupported
-> missing evidence IDs retained
```

## Multiple evidence chunks

Day 25 v1 uses strict semantics:

```text
supported_by = ["c1", "c2"]
```

means both `c1` and `c2` must be available.

If only `c1` is present:

```text
supported = False
evidence_ids = ["c1"]
missing_evidence_ids = ["c2"]
```

This makes failures inspectable.

## RAGEvaluator integration

`EvaluationCase` now optionally accepts:

```python
groundedness_claims=(
    GroundednessClaim(
        "Customers may request refunds within 30 days.",
        ("refund-01",),
    ),
)
```

`CaseEvaluation` now includes:

```text
groundedness
```

The end-to-end report preserves:

```text
RETRIEVAL
- Precision@K
- Recall@K
- Hit@K
- Reciprocal Rank

GENERATION
- reference-token proxy metrics
- deterministic groundedness

DIAGNOSTICS
- claim support
- evidence IDs
- missing evidence IDs
- failure stage
```

If labelled groundedness is present and evaluated claims are not fully
grounded, the case receives:

```text
failure_stage = "groundedness"
status = "FAIL"
```

Retrieval and augmentation failures still take precedence.

## Run Day 25 tests

```powershell
python -m pytest `
    backend/tests/test_groundedness.py `
    backend/tests/test_groundedness_day25.py `
    backend/tests/test_groundedness_integration_day25.py `
    -v
```

Then run the full RAG regression suite.

## Important limitation

This evaluator does not determine whether a claim is universally true.

It evaluates whether an explicitly labelled answer claim is backed by the
labelled evidence supplied to the answer.

Future work can compare this deterministic benchmark against an optional
LLM-as-judge or human semantic evaluator.
