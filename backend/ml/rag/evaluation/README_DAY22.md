# Day 22 — Labelled Offline RAG Retrieval Evaluation

Day 22 makes retrieval evaluation reproducible by separating normal production
telemetry from a human-labelled offline golden dataset.

## Why offline evaluation?

A production request can observe:

```text
query
retrieved chunk IDs
scores
latency
errors
```

But Recall@K needs:

```text
relevant chunks retrieved
--------------------------
all relevant chunks
```

For normal user traffic, the complete relevant set is usually unknown.

The offline dataset supplies that missing denominator through explicit human
relevance labels.

## Architecture

```text
Labelled Dataset
       |
       v
OfflineEvaluationDataset
       |
       v
OfflineRetrievalEvaluator
       |
       v
FaissRetriever.search()
       |
       v
Retrieved IDs
       |
       +---------- compare ----------+
       |                             |
       v                             v
Precision@K                    Relevant IDs
Recall@K
Hit@K
Reciprocal rank
       |
       v
Per-query Results
       |
       v
Aggregate Report
```

The runner imports and reuses the existing Day 20 `evaluate_retrieval()`
implementation. Precision and Recall are not reimplemented inside the runner.

## Golden dataset

The default dataset is:

```text
backend/ml/rag/datasets/retrieval_eval.json
```

It contains five manually labelled questions tied to the sample Transformer/RAG
chunks already used by AI Workflow Studio.

The cases cover:

1. query/key/value roles;
2. paraphrased long-range self-attention;
3. a multi-chunk positional-information question;
4. why long-document chunking improves retrieval;
5. a multi-chunk chunk-size/overlap trade-off.

These labels are part of the benchmark. If the source document or chunking
strategy changes, the labels must be reviewed.

## Dataset schema

```json
{
  "name": "day22-transformer-retrieval-golden-set",
  "description": "Human-labelled retrieval cases.",
  "cases": [
    {
      "id": "eval-001",
      "query": "What roles do queries, keys, and values play in self-attention?",
      "relevant_ids": [
        "transformer-rag-guide-recursive-001"
      ],
      "tags": [
        "direct",
        "attention"
      ]
    }
  ]
}
```

An empty `relevant_ids` list is allowed so a future golden dataset can represent
a human-labelled no-answer query.

## Per-query output

Each result retains:

- query ID;
- query text;
- all relevant IDs;
- retrieved IDs;
- ranks and scores;
- Precision@K;
- Recall@K;
- Hit@K;
- reciprocal rank;
- matched relevant IDs;
- missed relevant IDs;
- retrieval latency.

Keeping raw results is essential because an average can hide a catastrophic
failure on one question.

## Aggregate output

The runner calculates:

- mean Precision@K;
- mean Recall@K;
- Hit rate@K;
- mean reciprocal rank.

These values describe the labelled dataset only. They are not production
traffic metrics.

## Run tests

```powershell
python -m pytest `
  backend/tests/test_offline_eval_dataset.py `
  backend/tests/test_offline_eval_runner.py `
  -v
```

Then run the evaluation regression suite:

```powershell
python -m pytest `
  backend/tests/test_retrieval_metrics.py `
  backend/tests/test_groundedness.py `
  backend/tests/test_rag_evaluator.py `
  backend/tests/test_offline_eval_dataset.py `
  backend/tests/test_offline_eval_runner.py `
  -q --tb=short
```

## Run real FAISS evaluation

Make sure the index exists:

```powershell
python -m backend.ml.rag.search_cli inspect
```

Then:

```powershell
python -m backend.ml.rag.evaluation.offline_cli `
  --top-k 5
```

The runner prints per-query and aggregate metrics and writes:

```text
backend/ml/rag/artifacts/evaluation/day22-retrieval-report.json
```

Do not quote a project Precision/Recall result until this command has actually
run against the real index.

## Compare K

```powershell
python -m backend.ml.rag.evaluation.offline_cli --top-k 1
python -m backend.ml.rag.evaluation.offline_cli --top-k 3
python -m backend.ml.rag.evaluation.offline_cli --top-k 5
```

Increasing K may improve Recall@K while reducing Precision@K.

## Dataset-quality warning

Evaluation results are only as trustworthy as the relevance judgements.

Poor labels can make a good retriever look bad or a weak retriever look good.

Start with manually inspected cases, review disagreements, and expand the
golden set gradually.
