# Tests — README

> **Status:** Testing strategy defined. No tests have been written yet.

---

## Testing Philosophy

Every risk identified in the [AI Risk Register](../docs/ai-risk-register.md) must have at least one automated test that verifies its mitigation. Tests are not optional — they are the mechanism by which we confirm that safety controls work as intended and continue to work as the codebase evolves.

---

## Test Layers

### 1. Unit Tests

Verify individual functions and modules in isolation.

| Area | Examples |
|---|---|
| Chunking logic | Correct chunk sizes, no data loss at boundaries |
| Log sanitiser | Sentinel strings are redacted; allowlisted fields pass through |
| Input validation | Oversized files rejected; unsupported file types rejected |
| Prompt builder | Prompt structure matches expected template; injected context is escaped |

### 2. Integration Tests

Verify that components work correctly together against a real (test) database and vector store.

| Area | Examples |
|---|---|
| Document upload pipeline | File → chunks → embeddings stored with correct `user_id` |
| Retrieval isolation | User B's query returns zero results from User A's corpus |
| Authentication flow | Unauthenticated requests return 401; expired tokens return 401 |
| Deletion cascade | After deletion, embeddings and files for the user are gone |

### 3. Security / Adversarial Tests

Verify that known attack vectors are blocked.

| Test | Expected Outcome |
|---|---|
| Prompt injection document | Model response does not echo system prompt or follow injected instructions |
| Cross-user document access | HTTP 403 or empty result set |
| Oversized file upload | HTTP 413 returned before any processing occurs |
| Commit containing credential pattern | Pre-commit hook rejects the commit |
| Log leak sentinel | Sentinel string does not appear in any log output after a query |

### 4. Evaluation Tests (AI Quality)

Run against a labelled Q&A test set to measure AI answer quality.

| Metric | Target (TBD at implementation) |
|---|---|
| Retrieval Recall@5 | TBD |
| Answer faithfulness score | TBD |
| Hallucination rate (human review) | TBD |
| "No information found" accuracy | >95% when answer is absent from corpus |

---

## Planned Test Tools

| Layer | Tool |
|---|---|
| Backend unit / integration | pytest + pytest-asyncio |
| Frontend unit / component | Vitest + React Testing Library |
| End-to-end | Playwright |
| Security / secret scanning | trufflehog, git-secrets |
| Dependency audit | pip-audit (Python), npm audit (Node) |
| Load testing | Locust |

---

## Running Tests (Planned)

```bash
# Backend
cd backend
pytest tests/ -v

# Frontend
cd frontend
npm test

# End-to-end
npx playwright test
```

---

## Acceptance Criteria for Each Risk Mitigation

Before a risk mitigation is marked as **implemented** in the risk register, the corresponding test must:

1. Be committed to the repository.
2. Pass in CI without manual intervention.
3. Be reviewed by a second team member.
4. Be documented with a reference to the risk register entry it covers.
