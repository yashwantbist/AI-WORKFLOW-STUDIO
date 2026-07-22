# Backend — README

> **Status:** Foundation ready. Health endpoint implemented; RAG pipeline will be built in future lessons.

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

## Directory Structure

```
backend/
├── README.md              ← This file
├── requirements.txt       ← Python dependencies
├── Dockerfile             ← Container image for backend (optional)
├── main.py                ← FastAPI application entry point
├── app/
│   ├── api/               ← API route modules
│   │   └── health.py      ← Health-check endpoint
│   ├── models/            ← Pydantic schemas
│   ├── services/          ← Business logic (future: auth, document ingestion, retrieval)
│   ├── utils/             ← Logging, helpers
│   └── config/
│       └── settings.py    ← Environment-based configuration
└── tests/
    └── test_health.py     ← Tests for /health endpoint
```

**Future services** (to be implemented in upcoming lessons):
- `app/services/auth.py` — JWT issuance and validation
- `app/services/embeddings.py` — NVIDIA NIM embedding calls
- `app/services/generation.py` — NVIDIA NIM LLM calls
- `app/services/retrieval.py` — Vector store queries scoped to user

---

## Environment Variables

Copy `../.env.example` to `../.env` and set all required values before running the backend. See the [Data Handling Policy](../docs/data-handling-policy.md) for what each variable is used for.

**Never commit `.env` to version control.**

---

## Running Locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the development server
uvicorn main:app --reload --port 8000
```

Visit:
- **Interactive docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health endpoint:** http://localhost:8000/health

### Running with Docker (Optional)

```bash
# Build the image
docker build -t ai-workflow-backend:latest .

# Run the container
docker run -p 8000:8000 --env-file ../.env ai-workflow-backend:latest
```

---

## Architecture

### Health Endpoint

`GET /health` → `{"status": "ok"}`

Available without authentication. Used by load balancers, CI pipelines, and Kubernetes readiness/liveness probes.

### Planned Services

| Service | Purpose | Implementation Status |
|---|---|---|
| **Authentication** | JWT issuance and validation; session management | Planned |
| **Document Ingestion** | Upload, chunking, embedding, storage with per-user metadata | Planned |
| **Retrieval** | Similarity search scoped to authenticated user's documents | Planned |
| **Answer Generation** | RAG prompt construction and NVIDIA NIM LLM call | Planned |
| **Audit Logging** | Structured allowlist-based logging for monitoring | Planned |

### Authentication Flow (Planned)

1. User submits credentials via `/auth/login`
2. Backend verifies credentials and issues a JWT access token + refresh token
3. All subsequent requests include `Authorization: Bearer <access_token>`
4. Middleware validates the JWT and extracts `user_id`
5. Every retrieval query automatically filters by `user_id` — no cross-user data exposure

### RAG Pipeline (Planned)

1. **Document Upload** → `/documents/upload`
   - Accept PDF, DOCX, TXT, Markdown
   - Chunk into semantically meaningful passages
   - Generate embeddings via NVIDIA NIM embedding endpoint
   - Store in vector database (Chroma for dev, pgvector for prod) with `user_id` metadata
   
2. **Query** → `/retrieval/query`
   - User submits natural-language question
   - Embed the question using the same NVIDIA NIM embedding model
   - Retrieve top-k chunks from vector store **scoped to the user's corpus only**
   - Construct a retrieval-augmented prompt with retrieved context
   
3. **Generation** → `/generation/answer`
   - Send RAG prompt to NVIDIA NIM LLM inference endpoint
   - Stream or return the generated answer
   - Attach citations (chunk IDs, source filenames) to the response

---

## Security Notes

- All endpoints except `/health` will require a valid JWT (future implementation).
- Every document retrieval query will include a mandatory `user_id` filter — this is not optional and cannot be bypassed by request parameters.
- API keys are loaded from environment variables at startup; they are never logged or returned in API responses.
- See [`../docs/ai-risk-register.md`](../docs/ai-risk-register.md) for the full security risk model.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

All endpoints should have corresponding test coverage in `tests/`.
