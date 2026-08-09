# Day 19 — RAG Evaluation Harness

Day 19 evaluates the RAG system instead of judging it by fluent output alone.

## Evaluation layers

```text
                 RAG Quality
                     |
        +------------+------------+
        |            |            |
     Retrieval   Grounding      Answer
      quality     traces       relevance
```

A **retrieval failure** means the correct evidence never reaches the prompt.
A **generation failure** means the correct evidence was available but the answer
still failed. This harness also identifies an intermediate augmentation/threshold
failure when evidence was retrieved but did not survive into the RAG sources.

## Files

```text
backend/ml/rag/evaluation/
├── __init__.py
├── dataset.py
├── retrieval_metrics.py
├── answer_metrics.py
├── evaluator.py
├── report.py
└── cli.py

data/rag_eval.json
backend/tests/test_retrieval_metrics.py
backend/tests/test_rag_evaluator.py
```

## Retrieval metrics

**Hit@K**: 1 if at least one expected chunk appears in top K.

**Recall@K**: fraction of known relevant chunks retrieved in top K.

**Precision@K**: fraction of returned top-K chunks labeled relevant.

**Reciprocal rank**: rewards retrieving the first relevant chunk near rank 1.

Increasing K often improves recall but can reduce precision, increase context
size, increase token cost, and introduce distracting evidence.

## Answer metrics

Day 19 deliberately does not add an LLM judge yet.

It records deterministic lexical proxies:
- reference token precision/recall/F1;
- answer-token overlap with retrieved context;
- citation-marker count;
- whether an expected source was used.

These are regression signals, not proof of semantic correctness or true
faithfulness. A correct paraphrase can score low; an unsupported claim can reuse
many context words and score high.

## Known evaluation cases

The hand-written dataset uses stable chunk IDs already present in the sample
FAISS metadata, including:
- `transformer-rag-guide-recursive-001` for query/key/value;
- `...002` for long-range self-attention;
- `...005` for why chunking improves retrieval;
- `...007` for overlap;
- `...008` for metadata and citations.

If the source document or chunking strategy changes, review the labels.

## Run tests

```powershell
python -m pytest `
  backend/tests/test_retrieval_metrics.py `
  backend/tests/test_rag_evaluator.py `
  -v
```

## Run evaluation offline

Make sure the Day 17 index exists:

```powershell
python -m backend.ml.rag.search_cli build
```

Then:

```powershell
python -m backend.ml.rag.evaluation.cli `
  --provider demo `
  --top-k 3 `
  --min-score 0.10
```

The demo provider is not a real LLM, so the retrieval metrics are meaningful
while answer metrics may correctly look poor.

## Run with a real provider

After environment credentials are configured:

```powershell
python -m backend.ml.rag.evaluation.cli `
  --provider openai `
  --top-k 3 `
  --min-score 0.10
```

This can incur external API cost and latency.

## JSON report

The default report path is:

```text
backend/ml/rag/artifacts/evaluation/day19-report.json
```

The report preserves expected IDs, retrieved IDs/ranks/scores, generated answer,
used sources, metrics, latency, and failure stage.

## Compare K

```powershell
python -m backend.ml.rag.evaluation.cli --provider demo --top-k 1
python -m backend.ml.rag.evaluation.cli --provider demo --top-k 3
python -m backend.ml.rag.evaluation.cli --provider demo --top-k 5
```

Do not claim that one K is better until you compare the executed results.

## Latency note

The evaluator measures raw retrieval latency and total `RAGPipeline.answer()`
latency. Because Day 18 performs retrieval internally, total pipeline latency
includes another retrieval call. It is not pure generation latency.

## What retrieval evaluation tells you

It tells you whether your labeled evidence is returned and how highly it ranks.

## What it does not tell you

It does not prove answer correctness, true faithfulness, label completeness,
dataset representativeness, or production generalization.

## Security

Never place API keys, `.env`, credentials, private customer documents, or raw
production conversations in evaluation datasets or committed reports.

## Next steps

- expand the labeled dataset;
- compare K=1/3/5;
- add semantic answer similarity;
- add human faithfulness labels;
- add citation correctness;
- add an optional LLM judge only after deterministic metrics are stable;
- add CI regression thresholds later.
