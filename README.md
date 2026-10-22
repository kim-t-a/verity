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
- **Follow-ups stay on topic.** A message like "and in celsius?" is rewritten
  into a standalone search query before retrieval. "Explain in simple
  manner" or "I don't understand" re-explains the same cited passage in
  plainer words from the extracts, instead of repeating the previous answer.
- **Saved conversations.** Every chat is kept, listed by its first question
  and ordered by latest activity. Reopen one at any time, or delete it.
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

Two pipelines share one PostgreSQL database. Every stage below names the file
that owns it, so you can follow the code from this document.

### The whole system

```
┌──────────────┐     /api (JSON)     ┌─────────────────────────────┐
│ Browser      │ ──────────────────► │ FastAPI   verity/api/       │
│ React + Vite │ ◄────────────────── │                             │
└──────────────┘                     └───────┬─────────────┬───────┘
                                             │             │
                              ┌──────────────▼──┐   ┌──────▼──────────┐
                              │ PostgreSQL 16   │   │ NVIDIA API      │
                              │ pgvector        │   │ (backend only;  │
                              │                 │   │ key never in    │
                              └─────────────────┘   │ the browser)    │
                                                    └─────────────────┘
```

The browser calls only the backend. The backend calls the model provider.
That is rule 7 in [`AGENTS.md`](AGENTS.md): the key stays on the server.

### Pipeline 1 — ingestion (runs once per document)

```
POST /api/workspaces/1/documents              api/documents.py
  │  store the bytes, compute sha256
  │  INSERT documents (status = pending)
  ▼
background task _ingest_background            api/documents.py
  │
  ├─ 1. sniff    engine/ingestion/filetypes.py
  │       inspect the bytes → MediaType (never trust the filename)
  │
  ├─ 2. read     engine/ingestion/readers/
  │       file → ReadResult { text, spans, status }
  │       each span carries a location: "page 2", "row 14", ...
  │
  ├─ 3. chunk    engine/chunking.py
  │       spans → Chunk { text, rel_path, location }
  │
  ├─ 4. embed    engine/nvidia_embedding_provider.py
  │       each chunk's text → one vector
  │
  └─ 5. store    db/chunk_repository.py
          chunks + vectors → PostgreSQL
          documents.status = ready
```

The same five stages run in `worker/ingest.py`, which ingests a whole
directory. A failure at any stage marks **that document** `failed` with the
reason and the loop continues, so one bad file never costs the corpus.

A document becomes `ready` only after every chunk has been stored (I5).

### Pipeline 2 — answering (runs on every question)

```
POST /api/sessions/1/messages                 api/sessions.py
  │
  ├─ load stored chunks and vectors            db/chunk_repository.py
  │
  ├─ build the retriever                       api/sessions.py
  │     keyword  engine/retrieval/keyword.py   BM25
  │     vector   engine/retrieval/vector.py    cosine
  │     fuse     engine/retrieval/hybrid.py    RRF
  ▼
AgentLoop.run(question)                       agent/loop.py
  │
  ├─ a follow-up becomes a standalone query    agent/rewrite.py
  │     (a first question or a self-contained
  │     question is searched exactly as typed)
  │
  │   repeat at most 5 times:
  │     ┌──────────────────────────────────────────────────────┐
  │     │ retrieve the top chunks                              │
  │     │ show them to the model as numbered chunks [1]..[N]   │
  │     │ the model returns one action:                        │
  │     │   search        → refine the query, loop again       │
  │     │   answer        → text + the chunk numbers it used   │
  │     │   clarification → ask the user a question            │
  │     │   refusal       → say the evidence is not there      │
  │     └──────────────────────────────────────────────────────┘
  ▼
verify citations                              agent/loop.py
  │  every cited number must be 1..N of the chunks shown
  │  an invented number rejects the answer
  ▼
save the turn + citations                     api/sessions.py
  ▼
{"type": "answer" | "clarification" | "refusal", ...}
```

Three details that matter:

- **The model never sees a chunk it cannot cite.** The prompt contains exactly
  the retrieved chunks, numbered. There is no extra context (I2).
- **Memory is not evidence.** A follow-up is rewritten into a standalone query
  for retrieval, but the model still receives the user's literal message, and
  every turn retrieves again. An answer cannot cite a document that was not
  retrieved this turn (I4).
- **A restyle is explained, not repeated.** When the message asks for plain
  words, the same topic is retrieved again and the model is told to teach from
  those extracts — unpack their numbers and terms instead of rewording the
  last answer. If the model tries to clarify, the turn retries with a direct
  instruction to explain: nothing is ambiguous here.

### A question, step by step

1. Before the first search, if this is a follow-up, `agent/rewrite.py` turns it
   into a standalone query ("and in celsius?" becomes a complete question). A
   first question or a self-contained one is searched exactly as typed.
2. `api/sessions.py` loads the stored chunks and builds a `HybridRetriever`.
3. The keyword half ranks chunks by BM25; the vector half ranks them by cosine
   similarity to the question's embedding.
4. `HybridRetriever` fuses the two rankings with Reciprocal Rank Fusion (RRF),
   so a chunk found by both methods outranks one found by only one.
5. `AgentLoop` shows the top chunks to the LLM. The LLM answers with chunk
   numbers, searches again, clarifies, or refuses. If the message asked for an
   explanation in plain words, a system note tells the model to teach from
   those chunks instead of restating its previous answer.
6. The loop verifies every number against the chunks shown **in that
   iteration**. If verification fails, the turn is a refusal, not an answer.
7. `api/sessions.py` stores the user message, the assistant message, and one
   `citations` row per verified citation.
8. The frontend renders the answer, the evidence panel, and the trace.

### What lives in the database

| Table | Holds |
|---|---|
| `workspaces` | The single shared workspace. |
| `documents` | One row per uploaded file: status, hash, size, failure reason. |
| `document_versions` | The content-hash history for a document. |
| `chunks` | Text + vector; the retrieval unit. |
| `sessions` | A conversation, listed by its first question. |
| `messages` | User and assistant turns, with `kind` (answer/clarify/refuse). |
| `citations` | Which chunk and document supported which assistant message. |
| `memories` | Derived context (spec); not populated yet, and never evidence. |
| `jobs` | Queue table for a future long-running worker; not populated yet. |

Keyword ranking is BM25 computed in Python over the chunks loaded by
`db/chunk_repository.py`. The schema also carries a generated `tsv` column;
retrieval does not query it.

### Why the engine is separate

The `engine/` package is pure Python standard library. PDF, DOCX, and XLSX
parsing are hand-written against `zipfile` and `xml.etree`, so the engine can
be imported and tested with nothing installed. Only the product layer
(`api/`, `worker/`, `db/`, `llm/`) has third-party dependencies.

### Why no retrieval framework

Search and the agent loop are written here rather than assembled from
LangChain, LangGraph, or LlamaIndex. The design spec requires the mechanism
to stay visible rather than hidden behind a framework, and the engine must
keep its zero-dependency rule. The citation check also needs exact knowledge
of which chunks were shown, in which order, at which rank — and the trace
panel reports those real numbers. The trade-off is that we maintain the BM25
scorer, the cosine search, and the fusion logic ourselves.

### Reading order

If you are learning the system, read in this order:

1. `engine/chunking.py` — the unit everything else is built on.
2. `engine/ingestion/readers/__init__.py` — a file becomes text and locations.
3. `worker/ingest.py` — the five stages over a whole directory.
4. `engine/retrieval/hybrid.py` — how two rankings become one.
5. `agent/loop.py` — how a question becomes an answer with verified citations.
6. `api/sessions.py` — the HTTP surface around the loop.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12, FastAPI, psycopg2, Alembic |
| Engine | Standard library only (hand-written PDF/DOCX/XLSX parsers) |
| Storage | PostgreSQL 16, pgvector |
| Retrieval | BM25 + cosine similarity + Reciprocal Rank Fusion |
| Models | NVIDIA OpenAI-compatible chat and embedding endpoints |
| Frontend | React 18, TypeScript, Vite, Tailwind, TanStack Query |
| Tests | pytest (647 backend tests), `tsc --noEmit` for the frontend |

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

Or, to have the addresses printed for you:

```bash
bash start.sh --build
```

`start.sh` starts the stack in the background, waits until it is running,
then prints the app, API docs, and health URLs. Use `bash start.sh` on later
runs; add `--build` only after changing code or dependencies. (`chmod` is a
no-op on the Windows-mounted checkout, which is why the script is invoked
through `bash` rather than as `./start.sh`.)

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
2. **Ask a question.** Open Conversation and type it in plain language.
3. **Follow up naturally.** "And in celsius?" or "what about the TQ40-B?" is
   rewritten into a standalone search query. "Explain in simple manner" or
   "I don't understand" re-explains the cited passage in plainer words taken
   from the extracts, rather than repeating the previous answer.
4. **Find your past chats.** The list beside the conversation shows previous
   chats by their first question, newest first. Open one to continue it, or
   delete it — the questions and answers go with it, the documents stay.
5. **Read the evidence.** Every answer cites the file and location it came
   from. The evidence panel shows the retrieved chunks; the trace panel shows
   what was searched and how the rankings were fused.
6. **Accept a refusal.** If the documents do not contain the answer, Verity
   says so instead of guessing.

### Example with curl

```bash
# Upload a document (workspace 1 is the default shared workspace)
curl -F "file=@/path/to/document.pdf" \
  http://127.0.0.1:8000/api/workspaces/1/documents

# Poll until status is "ready"
curl http://127.0.0.1:8000/api/workspaces/1/documents

# Start a session
curl -X POST http://127.0.0.1:8000/api/sessions \
  -H "Content-Type: application/json" -d '{"workspace_id": 1}'

# Ask a question (use the session_id returned above)
curl -X POST http://127.0.0.1:8000/api/sessions/1/messages \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maximum operating temperature?"}'
```

The answer response is a discriminated union:

```json
{"type": "answer",        "text": "...", "citations": [...], "searched": [...], "iterations": 1}
{"type": "clarification", "text": "...", "citations": [],    "searched": [...]}
{"type": "refusal",       "text": "...", "citations": []}
```

`citations` lists only chunks that were shown to the model and verified against
what it cited. A refusal always has an empty `citations` list.

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
| `GET` | `/api/sessions` | List saved chats (`workspace_id`, `limit`; title, message count, last activity) |
| `DELETE` | `/api/sessions/{id}` | Delete a chat and its messages (`204`) |
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
├── legacy/                original challenge material (local only, ignored)
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
- PostgreSQL + pgvector storage, with BM25 keyword retrieval over the chunks
- Hybrid retrieval and Reciprocal Rank Fusion
- Follow-up rewriting before retrieval, and plain-language re-explanation
  grounded in the retrieved extracts
- Saved chats: list by first question, reopen with citations, delete
- Bounded agent loop with citation verification
- FastAPI routes and a React interface with evidence and trace panels
- Docker Compose stack, Alembic migrations, 647 passing backend tests

Not yet built:

- Authentication and multi-user workspaces (one shared workspace for now)
- A long-running job worker (the `jobs` table exists; ingestion currently
  runs in a background task)
- An evaluation harness with labelled questions and metrics (spec Phase 6)
- A reranker
- Clickable citations that open the source document (citations name the file
  and location; opening the file itself is not implemented)
- Production frontend serving (the Compose frontend runs the Vite dev server)

`legacy/` holds the original challenge material the engine was ported from. It
is local reference only: **git-ignored and never published**. The real-corpus
tests skip when it is absent, so a fresh clone still runs green.

## Contributing

- Read [`AGENTS.md`](AGENTS.md) before changing behaviour. It holds the
  invariants and the rules that must not be broken.
- Keep `verity.engine` free of third-party dependencies.
- Add a test that fails first, then make it pass.
- Run `TMPDIR=/tmp python -m pytest -q` from `backend/` before opening a PR.
- Never commit a provider key. `.env` is git-ignored; `.env.example` carries
  variable names only.

## License

MIT — see [`LICENSE`](LICENSE). Anyone may use, copy, modify, and distribute
this software, provided the copyright notice is kept.
