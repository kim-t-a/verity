# Verity

**Ask your documents. Get answers you can check.**

Verity is a local-first document question-answering product. Upload PDFs,
Word documents, spreadsheets, text, logs, CSVs, or images; ask a question in
plain language; get an answer that cites the exact file and location it came
from — or an honest refusal when the documents do not contain the answer.

It is not a chatbot and not a summariser. The answer to a good question
already exists as text in the uploaded documents. The work is finding it,
saying where it came from, and admitting when it is not there.

---

## Why it is different

- **Citations are verified.** The model may cite only chunks that were
  actually retrieved and shown to it this turn. An invented citation is
  rejected, not passed through.
- **Hybrid retrieval.** Keyword search (BM25) and vector search (pgvector)
  run together and are fused with Reciprocal Rank Fusion. One catches exact
  part numbers and filenames; the other catches paraphrases.
- **Bounded agent loop.** The model may answer, ask one clarifying question,
  or refuse — but clarifying is capped. At the cap it produces a best-effort
  answer or refuses only when evidence is genuinely absent.
- **Honest failures.** An encrypted PDF, an unreadable file, or a file with no
  reader is recorded and skipped. One bad document never aborts the corpus,
  and an unreadable file is never cited.
- **Local-first.** One shared workspace, no auth yet. Runs on your machine
  with Docker Compose. The provider key stays in the backend; the browser
  never talks to the model API.

## How it works

Two pipelines share one PostgreSQL database.

```
INGESTION (once per document)

  file ──► discover ──► read ──► chunk ──► embed ──► store
           walker     readers  chunking  provider   repository

QUESTION ANSWERING (every question)

  question ──► keyword search ──┐
            └► vector search ───┤
                                ▼
                       RRF fusion ──► top-k chunks
                                ▼
                  prompt with numbered chunks
                                ▼
                     LLM decides: answer / clarify / refuse
                                ▼
                  verify every cited chunk was shown
                                ▼
                        response + citations
```

```
┌───────────────┐      /api       ┌──────────────────────┐
│ React + Vite  │ ──────────────► │ FastAPI backend      │
│ (frontend/)   │                 │  api/ agent/ llm/    │
└───────────────┘                 │  worker/ db/         │
                                  │  engine/ (pure)      │
                                  └──────────┬───────────┘
                                             │
                                  ┌──────────▼───────────┐
                                  │ PostgreSQL 16        │
                                  │ pgvector + tsvector  │
                                  └──────────────────────┘
```

The `engine/` package is pure Python standard library. PDF, DOCX, and XLSX
parsing are hand-written against `zipfile` and `xml.etree`, so the engine can
be imported and tested with nothing installed. Only the product layer
(`api/`, `worker/`, `db/`, `llm/`) has third-party dependencies.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12, FastAPI, psycopg2, Alembic |
| Engine | Standard library only (hand-written PDF/DOCX/XLSX parsers) |
| Storage | PostgreSQL 16, pgvector, `tsvector` full-text search |
| Retrieval | BM25 + cosine similarity + Reciprocal Rank Fusion |
| Models | NVIDIA OpenAI-compatible chat and embedding endpoints |
| Frontend | React 18, TypeScript, Vite, Tailwind, TanStack Query |
| Tests | pytest (611 backend tests), `tsc --noEmit` for the frontend |

---

## Quick start — Docker Compose

### Prerequisites

- Docker Desktop with Compose
- An NVIDIA API key and the model IDs your account may invoke

### Run

```bash
git clone <your-repository-url> verity
cd verity

cp .env.example .env
# Edit .env and set:
#   NVIDIA_API_KEY
#   NVIDIA_API_BASE_URL
#   NVIDIA_LLM_MODEL

docker compose up --build
```

Open:

- App: <http://127.0.0.1:5173>
- API docs: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/api/health>

The backend container runs `alembic upgrade head` before serving, so a fresh
database is created automatically. The first build downloads Python and Node
dependencies, so it takes a few minutes.

To find a model ID your key can invoke:

```bash
cd backend
python -m scripts.probe_models
```

That script reads `NVIDIA_API_KEY` from the repository-root `.env`, prints
only model IDs, and needs no database.

### Offline mode (no key)

The fake providers let the plumbing run without a provider key:

```dotenv
VERITY_LLM_PROVIDER=fake
VERITY_EMBEDDING_PROVIDER=fake
```

Ingestion still works, but retrieval degrades to keyword-only (the fake
embedding provider returns zero vectors), and the fake LLM has no scripted
actions, so it refuses. Offline mode is for testing the wiring, not for
getting answers.

---

## Quick start — local development

### Prerequisites

- Python 3.12
- Node.js 20+
- Docker (for PostgreSQL)

### Backend

```bash
cd verity
docker compose up -d db

cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

alembic upgrade head
uvicorn verity.api.main:app --reload --port 8000
```

### Frontend

In a second terminal:

```bash
cd verity/frontend
npm install
npm run dev          # http://127.0.0.1:5173
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so the browser
never needs a CORS exception or an API key.

### Tests

```bash
cd backend
TMPDIR=/tmp python -m pytest -q
```

`TMPDIR=/tmp` is required if the repository lives on a 9p/drvfs mount of
Windows `D:` (as the original development environment did). Without it,
pytest puts `tmp_path` on that mount and reports "no tests ran" alongside a
misleading `FileNotFoundError`.

Database-backed tests use the Compose database and **skip** when it is not
running. Start `docker compose up -d db` for the full suite.

```bash
cd frontend
npm run typecheck
npm run build
```

---

## Using the product

1. **Upload documents.** Open the Documents screen and upload files, or POST
   them to the API. Ingestion runs in the background; the status moves from
   `pending` to `processing` to `ready` (or `failed` with a reason).
2. **Ask a question.** Open Chat and type it in plain language.
3. **Read the evidence.** Every answer cites the file and location it came
   from. The evidence panel shows the retrieved chunks; the trace panel shows
   what was searched and how the rankings were fused.
4. **Accept a refusal.** If the documents do not contain the answer, Verity
   says so instead of guessing.

### Example with curl

```bash
# Upload a document (workspace 1 is the default shared workspace)
curl -F "file=@legacy/mc3-corpus/specs/tq40_datasheet_r2.pdf" \
  http://127.0.0.1:8000/api/workspaces/1/documents

# Poll until status is "ready"
curl http://127.0.0.1:8000/api/workspaces/1/documents

# Start a session
curl -X POST http://127.0.0.1:8000/api/sessions \
  -H "Content-Type: application/json" -d '{"workspace_id": 1}'

# Ask a question (use the session_id returned above)
curl -X POST http://127.0.0.1:8000/api/sessions/1/messages \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maximum junction temperature of the TQ40?"}'
```

The answer response is a discriminated union:

```json
{"type": "answer",        "text": "...", "citations": [...]}
{"type": "clarification", "text": "...", "options": [...]}
{"type": "refusal",       "text": "..."}
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness |
| `POST` | `/api/workspaces/{id}/documents` | Upload a file (`202 Accepted`) |
| `GET` | `/api/workspaces/{id}/documents` | List documents and statuses |
| `GET` | `/api/workspaces/{id}/documents/{document_id}` | One document |
| `DELETE` | `/api/workspaces/{id}/documents/{document_id}` | Delete a document and its chunks |
| `POST` | `/api/sessions` | Start a session |
| `POST` | `/api/sessions/{id}/messages` | Ask a question |
| `GET` | `/api/sessions/{id}/messages` | Read the conversation |
| `POST` | `/api/ingestion` | Ingest a directory path synchronously |

Interactive documentation is served at `/docs` while the API runs.

---

## Repository layout

```
.
├── backend/
│   ├── verity/
│   │   ├── engine/        pure document logic: readers, chunking, retrieval
│   │   ├── db/            PostgreSQL connection + chunk repository
│   │   ├── worker/        the five-stage ingestion pipeline
│   │   ├── agent/         the bounded answer/clarify/refuse loop
│   │   ├── llm/           LLM interface, fake, NVIDIA implementation
│   │   ├── api/           FastAPI routes
│   │   └── provider_factory.py
│   ├── migrations/        Alembic schema history
│   ├── scripts/           manual provider diagnostics
│   └── tests/             the only test tree
├── frontend/              React + Vite application
├── docs/                  design specs, plans, walkthrough
├── legacy/                original challenge material (reference only)
├── AGENTS.md              durable rules for human and AI contributors
└── docker-compose.yml
```

Each backend folder has its own README explaining what lives there. Start at
[`backend/verity/README.md`](backend/verity/README.md).

## Documentation

- [`docs/verity-how-it-works.html`](docs/verity-how-it-works.html) — visual
  walkthrough of the product and its design.
- [`docs/superpowers/specs/`](docs/superpowers/specs/) — the design spec. **The
  spec is the authority** when documents disagree.
- [`docs/superpowers/plans/`](docs/superpowers/plans/) — implementation plans.
- [`AGENTS.md`](AGENTS.md) — project rules and invariants.
- [`backend/README.md`](backend/README.md) — how to run and test the backend.

## Design principles

Five invariants shape the code. Violating one is a bug, not a trade-off.

| # | Invariant |
|---|---|
| **I1** | Retrieve at chunk level; cite only chunks that were actually shown. |
| **I2** | The model never sees text it cannot cite. No silently injected context. |
| **I3** | Clarifying is never a way to avoid answering. The loop is capped. |
| **I4** | Memory shapes the question, never the evidence. Every turn re-retrieves. |
| **I5** | A document is `ready` only when every chunk is embedded. |

## Status

Working end to end:

- Ingestion of PDF, DOCX, XLSX, CSV, text, logs, source, and images (OCR when
  an engine is available)
- PostgreSQL + pgvector storage, with `tsvector` keyword search
- Hybrid retrieval and Reciprocal Rank Fusion
- Bounded agent loop with citation verification
- FastAPI routes and a React interface with evidence and trace panels
- Docker Compose stack, Alembic migrations, 611 passing backend tests

Not yet built:

- Authentication and multi-user workspaces (one shared workspace for now)
- A long-running job worker (the `jobs` table exists; ingestion currently
  runs in a background task)
- An evaluation harness with labelled questions and metrics (spec Phase 6)
- A reranker
- Production frontend serving (the Compose frontend runs the Vite dev server)

`legacy/` holds the original challenge material the engine was ported from. It
is reference only and is not part of the product.

## Contributing

- Read [`AGENTS.md`](AGENTS.md) before changing behaviour. It holds the
  invariants and the rules that must not be broken.
- Keep `verity.engine` free of third-party dependencies.
- Add a test that fails first, then make it pass.
- Run `TMPDIR=/tmp python -m pytest -q` from `backend/` before opening a PR.
- Never commit a provider key. `.env` is git-ignored; `.env.example` carries
  variable names only.

## License

No license has been chosen yet. Add one before making this repository public
if you want others to reuse it.
