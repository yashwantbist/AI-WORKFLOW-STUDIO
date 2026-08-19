# Day 27 — Observable LLM Inference v1

Day 27 adds a provider-independent inference boundary for AI Workflow Studio.

## Why this exists

Answer quality is only one part of a production LLM system.

Two requests can both be correct while having very different:

- latency;
- input context size;
- output length;
- token usage;
- cost;
- operational behavior.

The inference layer therefore records runtime measurements separately from RAG
quality evaluation.

## Architecture

```text
RAG / application code
        |
        v
ObservableLLM
        |
        v
LLMProvider protocol
        |
        v
Provider implementation
```

The application does not need to know whether the provider is NVIDIA, OpenAI,
a local model, or something else.

## Generation configuration

`GenerationConfig` currently records:

```text
temperature
max_tokens
top_p
top_k
```

It validates values but does not claim they determine quality or latency.

## Normalized provider result

A concrete provider maps its SDK response into:

```python
ProviderGeneration(
    text="...",
    model="...",
    usage=InferenceUsage(
        input_tokens=...,
        output_tokens=...,
    ),
)
```

If usage metadata is unavailable, leave token fields as `None`.

Do not estimate fake token counts merely to populate the schema.

## Latency

`ObservableLLM` measures only the provider generation call:

```text
start clock
provider.generate(...)
stop clock
```

This is generation latency.

It intentionally excludes retrieval, database access, and unrelated pipeline
operations.

Separate metrics should eventually track:

```text
retrieval latency
generation latency
total pipeline latency
```

## Throughput

Latency and throughput are different.

```text
Latency
= how long one request takes

Throughput
= how much work the system handles over time
```

Day 27 does not claim requests/second because a single synchronous request does
not provide enough information to measure service throughput correctly.

When output-token usage is available, the module exposes output tokens/second
for that request.

This is not the same as requests/second.

## Time to first token

Day 27 does not report TTFT.

The current abstraction represents non-streaming generation. A future streaming
provider can instrument the timestamp of the first received output token.

Until then, TTFT remains `not measured`.

## Optional cost estimation

Pricing is kept outside provider logic:

```python
ModelPricing(
    input_per_million=...,
    output_per_million=...,
)
```

Cost is only estimated when:

1. verified pricing is explicitly supplied; and
2. both input and output token counts are known.

Otherwise:

```text
estimated_cost = None
```

This prevents stale or invented pricing from becoming an evaluation result.

## Safe structured logging

The helper:

```python
generation_completed_event(result)
```

produces a payload resembling:

```json
{
  "event": "llm_generation_completed",
  "model": "provider-model-name",
  "latency_ms": 842.4,
  "input_tokens": 1250,
  "output_tokens": 184,
  "temperature": 0.2
}
```

It deliberately excludes:

- prompt text;
- generated text.

Prompts and answers can contain private or sensitive data.

## Provider errors

`ObservableLLM` does not silently convert provider errors into fake successful
results.

Provider exceptions propagate so higher-level application code can decide its
retry, fallback, or HTTP error policy.

A safe failure-log helper records only the exception type and generation
configuration.

## Tests

Run:

```powershell
python -m pytest `
    backend/tests/test_observable_inference_day27.py `
    -v
```

The tests use fake providers and fake clocks.

No external API calls are required.

Coverage includes:

- successful generation;
- generation configuration propagation;
- measured latency;
- provider exception propagation;
- missing token usage;
- zero output tokens;
- optional cost calculation;
- output tokens/second;
- safe success/failure logs;
- invalid generation configurations;
- empty prompts.

## Limitations

Observable LLM Inference v1 does not yet provide:

- streaming generation;
- time to first token;
- real requests/second throughput;
- retries;
- circuit breakers;
- provider-specific adapters;
- production metrics exporters;
- current model pricing configuration.

Those belong in later production-serving work rather than being fabricated in
Day 27.
