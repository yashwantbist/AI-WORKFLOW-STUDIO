# Day 26 — Unified Retrieval + Generation Evaluation

Day 26 extends AI Workflow Studio beyond retrieval and groundedness into two
additional generation-quality dimensions:

- answer relevance;
- correctness.

The goal is not to create a production semantic judge. The goal is to build
replaceable evaluator interfaces, deterministic fixtures, and useful failure
diagnostics.

## Evaluation dimensions

```text
Retrieval quality
-> Did the retriever find useful evidence?

Groundedness
-> Are generated claims supported by supplied evidence?

Answer relevance
-> Does the answer address the labelled requirement from the question?

Correctness
-> Does the answer agree with labelled trusted facts?
```

These dimensions are intentionally separate.

A response may be:

- grounded but irrelevant;
- relevant but incorrect;
- correct-looking but ungrounded;
- affected by retrieval and generation failures at the same time.

## Deterministic v1 labels

### Answer relevance

An evaluation case may specify:

```python
answer_relevance_phrases=("30 days",)
```

All labelled phrases must appear in the generated answer.

Result:

```text
all present -> 1.0
one or more missing -> 0.0
no labels -> None
```

This is deliberately binary for Day 26.

### Correctness

An evaluation case may specify:

```python
correctness_required_phrases=("30 days",)
correctness_forbidden_phrases=("60 days",)
```

Correctness is `1.0` only when:

- every required phrase appears; and
- no forbidden phrase appears.

Otherwise it is `0.0`.

If neither required nor forbidden labels exist, correctness is `None`.

## Why `None` matters

`None` means:

> This dimension was not evaluated.

It is different from `0.0`, which means:

> This dimension was evaluated and failed.

This prevents the report from inventing evaluation scores.

## Failure taxonomy

`FailureCategory` supports:

```text
RETRIEVAL_FAILURE
UNGROUNDED_GENERATION
IRRELEVANT_ANSWER
INCORRECT_ANSWER
SUCCESS
```

The classifier preserves multiple simultaneous failures.

Example:

```text
retrieval miss
+
unsupported answer
+
answer misses required question detail
+
wrong labelled fact
```

can produce:

```text
RETRIEVAL_FAILURE
UNGROUNDED_GENERATION
IRRELEVANT_ANSWER
INCORRECT_ANSWER
```

The existing `failure_stage` remains as the primary failure for backward
compatibility, while `failure_categories` provides the richer Day 26 view.

## Architecture

```text
Retriever
   |
Retrieval metrics
   |
Generator
   |
Groundedness evaluator
Answer relevance evaluator
Correctness evaluator
   |
Failure classifier
   |
Unified report
```

The generation evaluators live in `generation_quality.py` rather than being
buried inside the RAG pipeline.

That keeps a future LLM-as-a-judge implementation replaceable.

## Dataset example

```python
EvaluationCase(
    case_id="refund-001",
    question="What is the refund period?",
    expected_chunk_ids=("refund-policy",),
    reference_answer="Refund requests are accepted within 30 days.",
    groundedness_claims=(
        GroundednessClaim(
            "Refund requests are accepted within 30 days.",
            ("refund-policy",),
        ),
    ),
    answer_relevance_phrases=("30 days",),
    correctness_required_phrases=("30 days",),
    correctness_forbidden_phrases=("60 days",),
)
```

## Report

Per-query output is separated into:

```text
RETRIEVAL EVALUATION

GENERATION EVALUATION
```

Generation output includes:

- groundedness;
- answer relevance;
- correctness;
- matched/missing relevance phrases;
- matched/missing/conflicting correctness phrases;
- all failure categories;
- the primary failure stage.

Aggregate means use only cases where that dimension was actually labelled.

## Run Day 26 tests

```powershell
python -m pytest `
    backend/tests/test_generation_quality_day26.py `
    backend/tests/test_generation_evaluation_integration_day26.py `
    -v
```

Then run the complete RAG regression suite.

## Limitations

The v1 relevance and correctness evaluators use normalized phrase matching.

They do not understand semantic equivalence such as:

```text
"thirty days"
```

versus:

```text
"one month"
```

unless the fixture explicitly labels the chosen wording.

This limitation is intentional. Day 26 establishes evaluator boundaries and
diagnostics without introducing API cost, network dependency, judge bias, or
model non-determinism.

A future evaluator can implement the same conceptual interface using a human
judge or an LLM-as-a-judge.
