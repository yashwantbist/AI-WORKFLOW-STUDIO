# Backend — README

> **Status:** Not yet implemented. This document describes the planned architecture.

---

## Overview

The backend is a Python-based REST API service responsible for:

1. **Authentication** — verifying user identity and issuing session tokens.
2. **Document ingestion** — accepting file uploads, chunking text, generating embeddings, and storing them in the vector store with per-user metadata.
3. **Retrieval** — executing similarity searches scoped to the authenticated user's corpus.
4. **Answer generation** — constructing a retrieval-augmented prompt and calling the NVIDIA NIM LLM endpoint.
5. **Audit logging** — recording non-sensitive request metadata for monitoring and incident response.

---

## Planned Technology Stack

| Component | Planned Technology |
|---|---|
| API framework | FastAPI |
| Embedding model | NVIDIA NIM embedding endpoint |
| LLM | NVIDIA NIM inference endpoint |
| Vector store | Chroma (local dev) / pgvector (production) |
| Relational database | PostgreSQL |
| Object storage | S3-compatible (MinIO for local dev) |
| Authentication | JWT (access token) + refresh token |

---

## Directory Structure (Planned)

```
backend/
├── README.md            ← This file
├── requirements.txt     ← Python dependencies
├── app/
│   ├── main.py          ← FastAPI application entry point
│   ├── auth/            ← Authentication middleware and routes
│   ├── documents/       ← Upload, chunking, embedding routes
│   ├── retrieval/       ← Vector store query logic
│   ├── generation/      ← LLM prompt construction and call
│   ├── models/          ← Pydantic schemas
│   └── utils/
│       └── logging.py   ← Structured logging middleware (allowlist-enforced)
└── tests/               ← Backend unit and integration tests
```

---

## Environment Variables

Copy `../.env.example` to `../.env` and set all required values before running the backend. See the [Data Handling Policy](../docs/data-handling-policy.md) for what each variable is used for.

**Never commit `.env` to version control.**

---

## Running Locally (Planned)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn app.main:app --reload --port 8000
```

---

## Security Notes

- All endpoints except `/health` require a valid JWT.
- Every document retrieval query includes a mandatory `user_id` filter — this is not optional and cannot be bypassed by request parameters.
- API keys are loaded from environment variables at startup; they are never logged or returned in API responses.
- See [`../docs/ai-risk-register.md`](../docs/ai-risk-register.md) for the full security risk model.
