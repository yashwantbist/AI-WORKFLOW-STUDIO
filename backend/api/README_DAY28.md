# Day 28 — Production RAG API v1

Day 28 moves AI Workflow Studio from an internal RAG/inference workflow toward
a production-oriented HTTP service.

## Architecture

```text
Client
  |
FastAPI route
  |
Authentication dependency
  |
RAGService
  |
Retriever / LLM Provider / Evaluator
```

The HTTP route does not contain retrieval, embedding, prompt construction,
generation, or evaluation logic.

That behavior belongs behind the `RAGService` boundary.

## Endpoints

### POST `/api/v1/ask`

Request:

```json
{
  "question": "How does self-attention work?"
}
```

Response:

```json
{
  "request_id": "trace-id",
  "answer": "..."
}
```

Questions must contain 1–2000 characters.

### GET `/health`

Liveness endpoint:

```json
{
  "status": "ok"
}
```

It only answers whether the API process is alive.

It does not call an LLM or vector database.

### GET `/ready`

Readiness endpoint:

```json
{
  "status": "ready"
}
```

A supplied readiness checker may verify lightweight dependency state.

Do not make readiness perform an expensive generation request.

## Request tracing

Every HTTP request receives a request ID.

The same ID is:

- stored on request state;
- returned from `/api/v1/ask`;
- returned in `X-Request-ID`;
- passed into `RAGService.answer()`;
- attached to structured events;
- included in public error responses.

This gives a simple trace correlation mechanism across API, retrieval,
generation, and evaluation layers.

## Structured logs

The log helper records metadata such as:

```json
{
  "event": "rag_request_completed",
  "request_id": "abc123",
  "status_code": 200,
  "total_latency_ms": 1240
}
```

The middleware measures real HTTP request latency using `perf_counter`.

It deliberately avoids logging:

- question text;
- prompt text;
- answer text;
- retrieved context.

Those fields can contain private or sensitive information.

## Error boundaries

Known operational failures map to safe public responses:

```text
401 -> authentication failure
422 -> invalid request
429 -> rate limited
503 -> AI/dependency unavailable
500 -> unexpected application error
```

Internal exception messages are not sent to clients.

Example:

```json
{
  "error": "AI_SERVICE_UNAVAILABLE",
  "message": "The AI service is temporarily unavailable.",
  "request_id": "abc123"
}
```

## Authentication

`require_api_key` is a small demonstration boundary using `X-API-Key` and the
`RAG_API_KEY` environment variable.

It is intentionally separate from RAG logic.

If AI Workflow Studio already has JWT authentication, replace this dependency
with that existing authentication dependency instead of maintaining two auth
systems.

Never hard-code the key in source control.

## Liveness vs readiness

```text
Liveness:
Is the process alive?

Readiness:
Can the application serve traffic?
```

A live application can still be unready because a required dependency is down.

## Prompt injection boundary

Retrieved text and user-provided text are data.

They must not be treated as trusted application instructions.

Day 28 does not implement a complete prompt-injection defense system, but the
service boundary is designed so retrieval/prompt policy remains below the HTTP
route rather than being mixed with web handling.

## Run tests

Install development dependencies if needed:

```powershell
python -m pip install fastapi uvicorn httpx pytest
```

Then:

```powershell
python -m pytest `
    backend/tests/test_production_rag_api_day28.py `
    -v
```

The API tests use fake services.

No external model, vector database, or paid API call is required.

## Production wiring

A future application bootstrap can construct:

```text
real retriever
+ observable LLM provider
+ evaluator
        |
        v
real RAGService
        |
        v
create_app(...)
```

This keeps test wiring and production wiring separate.

## Known limitations

Production RAG API v1 does not yet implement:

- distributed tracing;
- Prometheus/OpenTelemetry exporters;
- persistent rate limiting;
- retries/circuit breakers;
- production JWT integration;
- deployment-specific readiness probes;
- comprehensive prompt-injection defenses.

Those concerns should be added deliberately rather than simulated.
