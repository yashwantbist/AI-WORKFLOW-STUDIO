# Day 23 — RAG Retrieval Diagnostics and Regression Reporting

Day 23 turns the Day 22 offline retrieval benchmark into an engineering
diagnostic.

The evaluation system now answers two different questions:

1. **What failed?**
2. **Did a retrieval change regress against a measured baseline?**

## Failure classification

Each labelled query receives one coverage status:

```text
SUCCESS
NO_RESULTS
PARTIAL
COMPLETE_MISS
IRRELEVANT_RESULTS
```

The classification is based on the relationship between retrieved chunk IDs
and the human-labelled relevant set.

### SUCCESS

All labelled relevant chunks were retrieved.

A successful result can still contain extra irrelevant chunks. That retrieval
noise is recorded separately through `irrelevant_retrieved_ids`.

This distinction matters because:

```text
retrieved = [relevant-1, irrelevant-1, irrelevant-2]
relevant  = [relevant-1]
```

has complete relevant-set coverage but still contains noisy context.

### NO_RESULTS

The golden case has relevant evidence, but retrieval returned nothing.

### PARTIAL

At least one relevant chunk was retrieved, but at least one labelled relevant
chunk was missed.

### COMPLETE_MISS

Results were returned, but none intersected the labelled relevant set.

### IRRELEVANT_RESULTS

This status is reserved for a human-labelled no-answer case whose relevant set
is empty but retrieval still returns chunks.

If a no-answer query correctly retrieves nothing, it is a success.

## Diagnostic report

The report retains the existing aggregate metrics:

- Mean Precision@K
- Mean Recall@K
- Hit Rate@K
- Mean Reciprocal Rank

It also adds:

- successful query count;
- partial retrieval count;
- complete misses;
- no-result queries;
- irrelevant-only no-answer cases;
- queries containing extra irrelevant chunks;
- failed query IDs;
- expected IDs;
- retrieved IDs;
- matched IDs;
- missed IDs;
- irrelevant retrieved IDs.

An aggregate alone cannot identify which query failed.

## Measured baseline

The initial baseline in:

```text
backend/ml/rag/baselines/retrieval_baseline.json
```

comes from the actual Day 22 K=5 evaluation run:

```text
Dataset: day22-transformer-retrieval-golden-set
Queries: 5
K: 5

Mean Precision@5: 0.320
Mean Recall@5:    1.000
Hit Rate@5:       1.000
MRR:              1.000
```

These values describe only the five-query labelled offline benchmark.

They are not production metrics.

## Baseline safety checks

A candidate is compared only when:

- dataset names match;
- K values match;
- query counts match.

This prevents comparisons such as K=3 versus K=5 from being treated as a
normal retrieval regression.

## Regression tolerance

The default tolerance is:

```text
0.01
```

For each existing metric:

```text
delta = candidate - baseline
```

A metric is marked regressed only when:

```text
delta < -tolerance
```

Therefore a tiny change such as:

```text
-0.00001
```

does not automatically fail the benchmark.

The comparison currently checks metrics the project already has:

- Mean Precision@K
- Mean Recall@K
- Hit Rate@K
- Mean Reciprocal Rank

Day 23 does not add NDCG.

## Run diagnostic evaluation

```powershell
python -m backend.ml.rag.evaluation.offline_cli `
    --top-k 5
```

The default output is:

```text
backend/ml/rag/artifacts/evaluation/day23-diagnostic-report.json
```

## Compare against the measured Day 22 baseline

```powershell
python -m backend.ml.rag.evaluation.offline_cli `
    --top-k 5 `
    --baseline backend/ml/rag/baselines/retrieval_baseline.json
```

You should not predict the regression result before executing this command.

## Use a custom tolerance

```powershell
python -m backend.ml.rag.evaluation.offline_cli `
    --top-k 5 `
    --baseline backend/ml/rag/baselines/retrieval_baseline.json `
    --tolerance 0.01
```

## Make regression usable in CI

```powershell
python -m backend.ml.rag.evaluation.offline_cli `
    --top-k 5 `
    --baseline backend/ml/rag/baselines/retrieval_baseline.json `
    --tolerance 0.01 `
    --fail-on-regression
```

The CLI returns exit code `2` when a measured regression is detected.

## Save a new measured baseline

Only do this after deciding the candidate should become the new reference:

```powershell
python -m backend.ml.rag.evaluation.offline_cli `
    --top-k 5 `
    --save-baseline backend/ml/rag/baselines/retrieval_baseline_candidate.json
```

Do not overwrite a trusted baseline merely because a candidate performed worse.

Review the failures first.

## Tests

Run Day 23 tests:

```powershell
python -m pytest `
    backend/tests/test_retrieval_diagnostics.py `
    backend/tests/test_diagnostic_report.py `
    backend/tests/test_retrieval_baseline.py `
    -v
```

Then run the RAG evaluation regression suite:

```powershell
python -m pytest `
    backend/tests/test_retrieval_metrics.py `
    backend/tests/test_groundedness.py `
    backend/tests/test_rag_evaluator.py `
    backend/tests/test_offline_eval_dataset.py `
    backend/tests/test_offline_eval_runner.py `
    backend/tests/test_retrieval_diagnostics.py `
    backend/tests/test_diagnostic_report.py `
    backend/tests/test_retrieval_baseline.py `
    -q --tb=short
```

## Interpretation discipline

A regression report does not prove why retrieval changed.

It tells you which measured behavior changed on the labelled benchmark.

Investigation can then focus on:

- embedding changes;
- query processing;
- chunking;
- index changes;
- filters;
- K;
- score thresholds;
- reranking;
- relevance-label quality.

Keep the diagnostic report and the production inference path separate.
