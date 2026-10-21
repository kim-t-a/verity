# `verity.api` — the HTTP surface

| Module | Responsibility |
|---|---|
| `main.py` | Builds the FastAPI app, CORS, health check, includes routers. |
| `documents.py` | Upload, list, inspect, delete documents. |
| `sessions.py` | Start a conversation; ask a question; read messages. |
| `ingestion.py` | Trigger corpus ingestion synchronously. |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness: `{"status": "ok"}` |
| `POST` | `/api/workspaces/{workspace_id}/documents` | Upload a file (`202 Accepted`); ingestion runs in the background. |
| `GET` | `/api/workspaces/{workspace_id}/documents` | List documents and statuses. |
| `GET` | `/api/workspaces/{workspace_id}/documents/{document_id}` | One document. |
| `DELETE` | `/api/workspaces/{workspace_id}/documents/{document_id}` | Delete a document and its chunks. |
| `POST` | `/api/sessions` | Start a session. |
| `POST` | `/api/sessions/{session_id}/messages` | Ask a question. |
| `GET` | `/api/sessions/{session_id}/messages` | Read the conversation. |
| `POST` | `/api/ingestion` | Ingest a directory path (synchronous). |

## The answer contract

`POST /api/sessions/{session_id}/messages` returns a discriminated union the
frontend switches on:

```json
{"type": "answer",        "text": "...", "citations": [...]}
{"type": "clarification", "text": "...", "options": [...]}
{"type": "refusal",       "text": "..."}
```

Changing this shape is a public-contract change: it belongs in the spec, not in
a refactor.

## What the API owns

- Authentication when it arrives (none yet; one shared workspace).
- Validation of uploads and request bodies.
- Retrieval and the agent loop.
- The model-provider call. The browser never talks to NVIDIA.

## Dependencies

The API is product code: it may import `fastapi`, `pydantic`, `psycopg2`, and
`python-dotenv`. The engine (`verity.engine`) remains standard-library-only and
must not import anything from here.

## Docs while running

`uvicorn verity.api.main:app --reload --port 8000` serves interactive docs at
<http://127.0.0.1:8000/docs>.
