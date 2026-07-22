# NVIDIA AI Workflow Studio — Data Handling Policy

> **Version:** 0.1.0
> **Date:** 2026-07-21
> **Status:** Preliminary policy. Controls marked as *planned* have not been implemented. This document will be updated as the application is built.

---

## 1. What Data the Application Accepts

The application accepts the following types of data from users:

| Data Type | Format | Purpose |
|---|---|---|
| User-uploaded documents | PDF, plain text (.txt), Markdown (.md) | Source corpus for retrieval and Q&A |
| Natural-language queries | Free text (typed by the user) | Input to the RAG pipeline |
| User credentials | Email address + hashed password (or OAuth token) | Authentication and user isolation |

The application does **not** accept and must actively reject:

- Documents larger than the configured size limit (planned: 20 MB per file, 200 MB per user).
- File types other than the supported formats listed above.
- Documents explicitly labelled as containing health records (HIPAA), financial account numbers (PCI-DSS), or government-issued ID numbers, unless a separate data-protection impact assessment has been completed.

---

## 2. Why the Application Needs That Data

| Data | Reason |
|---|---|
| **Uploaded documents** | The application's core function is to answer questions grounded in the user's own documents. Without the documents, the retrieval pipeline has no corpus to search. |
| **User queries** | Queries are needed to perform similarity search over the document embeddings and to construct the prompt sent to the LLM. |
| **User credentials** | Authentication is required to enforce per-user document isolation — without knowing who the user is, the system cannot prevent one user from accessing another's documents. |

The application does **not** collect data for advertising, profiling, behavioural analytics, or model training.

---

## 3. Where Data Will Be Stored

> All storage locations below are **planned**. No production storage has been configured.

| Data | Storage Location | Isolation |
|---|---|---|
| Uploaded document files (raw) | Object storage bucket (e.g., S3-compatible) | *Planned:* Stored under a per-user key prefix; IAM policy restricts access to the application service account only |
| Document embeddings (vectors) | Vector store (e.g., Chroma, pgvector) | *Planned:* Each embedding record includes a `user_id` metadata field; all queries apply a mandatory `user_id` filter |
| User credentials | Relational database | *Planned:* Passwords stored as bcrypt hashes with cost factor ≥ 12; no plaintext passwords ever stored |
| Query history | Relational database (optional feature) | *Planned:* Scoped to the authenticated user; opt-in only; purged on account deletion |
| Application logs | Log aggregation service | Logs contain only non-sensitive metadata (see Section 7); no document content or query text |

---

## 4. Who Has Access

| Role | Access | Justification |
|---|---|---|
| **End user** | Own documents, own query history, own embeddings only | Core privacy guarantee; no cross-user access |
| **Application service account** | Read/write to the storage systems listed in Section 3 | Required for normal operation |
| **System administrator** | Application logs and infrastructure metrics; **not** document content | Needed for monitoring and incident response |
| **NVIDIA NIM API infrastructure** | Document chunks + user query (transmitted per API call) | Required to generate answers; covered by NVIDIA's API terms of service |
| **Third parties / partners** | None | No data is sold, shared, or licensed to third parties |
| **Development team** | Anonymised or synthetic data in non-production environments only | Real user data must never be used in development or test environments |

---

## 5. Whether Data Will Be Used for Model Training

**No.** User-uploaded documents and user queries will not be used to train, fine-tune, evaluate, or improve any machine learning model — internal or external — without:

1. The user's **explicit, informed, opt-in consent** presented in plain language.
2. A documented data-protection impact assessment.
3. A separate contractual agreement if a third-party model provider is involved.

By default, all user data is used solely to provide the service to that user and for no other purpose.

---

## 6. How Users Can Delete Their Data

> The deletion mechanism is **planned**. The following describes the intended behaviour.

Users will be able to delete their data through two mechanisms:

### 6.1 Self-service deletion (planned)

The application UI will provide a **"Delete my documents"** function that:

1. Removes all raw document files from object storage.
2. Deletes all associated embeddings from the vector store (filtered by `user_id`).
3. Deletes query history records associated with the user (if query history is enabled).
4. Returns HTTP 200 with a confirmation message.

The deletion will be **irreversible** and will complete within 30 days of the request (to allow for backup purge cycles).

### 6.2 Account deletion (planned)

Deleting the user account will trigger a cascade deletion of all data listed in 6.1, plus the credential record. Users will be warned that deletion is permanent before confirmation.

### 6.3 Data retention after deletion

After a confirmed deletion request:

- Live storage: deleted immediately.
- Backup snapshots: purged within 30 days.
- Application logs: log entries containing the user ID hash are retained for up to 90 days for security and audit purposes, then deleted. Log entries do not contain document content.

---

## 7. What Must Never Appear in Application Logs

The following categories of data are **strictly prohibited** from appearing in any application log, regardless of log level:

| Prohibited Data | Example |
|---|---|
| Document content | Any text extracted from a user's uploaded file |
| Query text | The literal question typed by the user |
| Retrieved document chunks | Passages returned by the vector store |
| LLM prompt or response | The full prompt sent to or response received from the NIM API |
| Plaintext passwords | Any credential in readable form |
| API keys or tokens | `LLM_API_KEY`, `OBJECT_STORAGE_SECRET`, JWT tokens |
| Personally identifiable information | Email addresses, full names, phone numbers |
| Payment or financial data | Credit card numbers, bank account details |

**What logs may contain:**

- Request timestamp (UTC)
- Request ID (UUID)
- Authenticated user ID — stored as a one-way hash, not the raw ID
- Endpoint path and HTTP method
- HTTP response status code
- Request latency in milliseconds
- File size of upload (bytes only, not content)
- Error codes and stack traces (stripped of any user-data values)

*Planned control:* A structured-logging middleware will enforce an allowlist of loggable fields and will redact or drop any field not on the allowlist before writing to the log sink.

---

## 8. Compliance and Regulatory Considerations

- The application is not initially designed for use in regulated industries (healthcare, finance, legal) without a separate compliance review.
- If deployed in the EU or for EU users, a GDPR compliance review is required before launch, including appointment of a Data Protection Officer if applicable.
- Processing of special-category personal data (health, biometric, political, religious) is out of scope and must be blocked at the upload layer.

---

## 9. Policy Review

This policy will be reviewed and updated:

- Before the application enters production.
- Whenever a new data type is collected or a new storage location is introduced.
- After any security incident that involves user data.
- At minimum annually.

---

## Change Log

| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-07-21 | Initial policy created |
