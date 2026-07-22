# NVIDIA AI Workflow Studio - System Card

> **Version:** 0.1.0-pre-implementation
> **Date:** 2026-07-21
> **Status:** Foundation phase - no application code has been written yet. All controls marked as *planned* have not been implemented.

---

## Purpose

NVIDIA AI Workflow Studio is a retrieval-augmented generation (RAG) platform that allows authenticated users to upload documents and query them through a natural-language interface. The system retrieves relevant document chunks, passes them as context to a large language model (LLM) hosted on NVIDIA NIM endpoints, and returns grounded, cited answers. The purpose of this system card is to document what the system is designed to do, what it must not do, where it currently falls short, and what safeguards are planned or in place.

---

## Intended Users

- Individual knowledge workers who need to search and query their own document collections.
- Small teams conducting internal research or policy review against a private corpus.
- Developers evaluating NVIDIA NIM inference endpoints for document-grounded Q&A use cases.
- Researchers exploring RAG architectures under controlled, consented conditions.

The system is designed for **adult users** who have agreed to the platform terms of service and data-handling policy.

---

## Intended Uses

1. **Document Q&A:** Users upload PDF, TXT, or Markdown files and ask questions answered using retrieved passages from those files only.
2. **Summarisation:** Users request summaries of one or more uploaded documents; the model is constrained to the retrieved context.
3. **Citation retrieval:** The system returns the source document name, page number, and passage alongside every answer so users can verify claims.
4. **Private knowledge management:** Users maintain a personal document corpus that is not shared with other users.

---

## Out-of-Scope Uses

The following uses are explicitly **not supported** and should be prevented by design:

1. **General-purpose web search or open-ended question answering** without a document corpus - the system is not a general chatbot.
2. **Medical, legal, or financial advice** intended to replace qualified professional consultation.
3. **Processing sensitive categories of personal data** (health records, financial account numbers, biometric data, government-issued ID numbers) without a separate data-protection impact assessment.
4. **Automated decision-making that affects individuals rights** (hiring, lending, benefits eligibility) without human review.
5. **Access to another user documents** - the system must never expose one user corpus to another.
6. **Training or fine-tuning any external model** using user-uploaded content without explicit written consent.
7. **Scraping, re-hosting, or republishing** third-party copyrighted material retrieved from user documents.

---

## Current Capabilities

> **Note:** The application has not been built yet. This section describes the *planned* capabilities after implementation is complete, not the current state.

- Planned: Chunking and embedding of uploaded documents.
- Planned: Similarity search over a per-user vector store.
- Planned: Context-grounded answer generation via NVIDIA NIM LLM endpoints.
- Planned: Source citation returned with every answer.
- Planned: User authentication and per-user document isolation.

**Currently implemented:** Repository structure, safety documentation, risk register, and data-handling policy only.

---

## Current Limitations

1. **No hallucination elimination:** The LLM may generate plausible-sounding but incorrect statements even when a retrieved passage is present. Mitigation: citations allow users to verify; the UI will display a disclaimer.
2. **Chunk boundary artefacts:** Important context may be split across chunks, degrading answer quality.
3. **Language support:** Initial implementation targets English only; multilingual document behaviour is untested.
4. **Document format support:** Only PDF, plain text, and Markdown are planned for the initial release. Tables, images, and structured spreadsheets may not parse correctly.
5. **Context window limits:** Very long documents may exceed the model context window; portions will be truncated.
6. **No real-time data:** The system operates only on user-uploaded static documents; it has no internet access or live data feeds.
7. **Evaluation gap:** Automated evaluation of answer quality (faithfulness, relevance) has not yet been implemented.

---

## Data Sources

- **User-uploaded documents only.** The system does not ingest external data sources, web content, or third-party databases.
- Documents are processed locally (chunked and embedded) before being stored in a user-scoped vector store.
- No training data is collected from user queries or documents.
- The LLM is accessed via NVIDIA NIM API endpoints; queries (document chunks + user question) are transmitted to NVIDIA infrastructure in accordance with NVIDIA API terms of service.

---

## Privacy Considerations

1. **Data minimisation:** Only content the user explicitly uploads and the questions they explicitly ask are processed. No behavioural telemetry or device fingerprinting is collected.
2. **User isolation:** *Planned control* - every vector store query will be filtered by the authenticated user ID, ensuring no cross-user document access.
3. **Log sanitisation:** *Planned control* - application logs must not contain document content, query text, or API keys. Only request metadata (timestamp, user ID hash, latency, status code) will be logged.
4. **Retention:** *Planned control* - users will be able to delete their documents and associated embeddings at any time; deletion will cascade to all derived artefacts.
5. **Third-party API exposure:** Document chunks are transmitted to NVIDIA NIM endpoints. Users must be informed of this in the terms of service and during onboarding.
6. **No model training on user data:** User documents and queries will not be used to train, fine-tune, or improve any model unless the user provides explicit, informed, opt-in consent.

---

## Safety and Security Controls

| Control | Status |
|---|---|
| Per-user document isolation at retrieval layer | Planned |
| API key storage in environment variables only (never source code) | Implemented |
| .env excluded from version control via .gitignore | Implemented |
| Input length and file size limits | Planned |
| Prompt injection detection and sanitisation | Planned |
| Rate limiting on API endpoints | Planned |
| Authentication required for all document operations | Planned |
| Audit logging of document upload and deletion events | Planned |
| Dependency vulnerability scanning (pip-audit, npm audit) | Planned |
| Secret scanning in CI pipeline | Planned |

---

## Transparency Measures

1. Every answer will include a citation to the source document and passage (planned).
2. When retrieval finds no relevant content, the system will respond with a clear "no relevant information found" message rather than speculating (planned).
3. Users will be informed that answers are generated by an AI model and may contain errors.
4. This system card is version-controlled and updated with each significant change to the system capabilities or limitations.
5. The risk register and data-handling policy are co-located with the source code and open for review.

---

## Bias and Fairness Considerations

1. **Retrieval bias:** Embedding models may return chunks that reflect biases present in the training data of the embedding model itself. Users should be aware that search ranking is not neutral.
2. **Language bias:** The initial system targets English. Non-English documents may produce degraded or incorrect embeddings and answers.
3. **Document quality bias:** The accuracy of answers depends entirely on the quality and completeness of the uploaded documents. Gaps in the corpus are gaps in the system knowledge.
4. **LLM bias:** The underlying language model may reflect societal biases present in its pre-training data. These biases are not measurable or controllable at the application layer.
5. **Mitigation approach:** The system grounds all answers in retrieved context, which reduces (but does not eliminate) the model ability to introduce out-of-corpus bias.
6. *Planned evaluation:* A bias audit using a diverse test document set will be conducted before production release.

---

## Human Oversight

- The system provides citations with every answer so that humans can verify AI-generated claims against source documents.
- No automated action will be triggered by the system output; all outputs are advisory.
- Administrators will have access to audit logs (excluding document content) to review system behaviour.
- *Planned:* A feedback mechanism (thumbs up/down) will allow users to flag incorrect or harmful responses, which will be reviewed by the development team.
- *Planned:* Flagged responses will be reviewed within 5 business days.

---

## Evaluation Plan

| Metric | Method | Status |
|---|---|---|
| Retrieval relevance (Recall@k) | Labelled Q&A test set | Planned |
| Answer faithfulness | Automated faithfulness scoring vs. retrieved context | Planned |
| Answer completeness | Human evaluation on a sample | Planned |
| Hallucination rate | Human review of randomly sampled answers | Planned |
| Cross-user isolation | Automated integration test (User A requests User B doc) | Planned |
| Prompt injection resistance | Red-team test suite | Planned |
| Response latency (P95) | Load test | Planned |

---

## Known Risks

See [ai-risk-register.md](ai-risk-register.md) for the full risk register with likelihoods, impacts, mitigations, and test criteria.

**Summary of highest-priority risks:**

1. Prompt injection via malicious document content
2. Cross-user document exposure
3. Hallucinated or unsupported answers presented as factual
4. Sensitive information (PII, secrets) appearing in application logs
5. Biased retrieval or response
6. Exposed API credentials in source code or logs
