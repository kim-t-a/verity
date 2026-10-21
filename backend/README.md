# Verity backend

The server half of Verity: it ingests documents, stores searchable chunks in
PostgreSQL, retrieves evidence for a question, and asks a language model to
answer using only that evidence.

New to RAG? Read [`verity/README.md`](verity/README.md) first — it explains the
two pipelines and maps every stage to a file.

## What lives here

| Path | What it is |
|---|---|
| `verity/` | **The application.** Everything that runs in production. |
| `tests/` | The only test tree. Run pytest from this directory. |
| `migrations/` | The database schema's change history (Alembic). |
| `scripts/` | Manual developer tools. Never imported by the app. |
| `pyproject.toml` | Package metadata and pytest settings. |
| `alembic.ini` | Tells Alembic where migrations live and how to connect. |
| `Dockerfile` | Builds the backend image; runs migrations before serving. |
| `verity.egg-info/`, `.pytest_cache/` | Generated. Ignore them. |

## Run it

```bash
# From the repository root
docker compose up -d db          # PostgreSQL 16 + pgvector

cd backend
alembic upgrade head             # create/update the schema
uvicorn verity.api.main:app --reload --port 8000
```

The API is then at <http://127.0.0.1:8000>, with interactive docs at
<http://127.0.0.1:8000/docs>. Check it with:

```bash
curl http://127.0.0.1:8000/api/health     # {"status":"ok"}
```

Provider keys are read from `.env` at the repository root. Copy
`.env.example` to `.env` and fill in real values there. Keys never belong in
git and never reach the frontend.

### First run on a brand-new database

The schema stores pgvector `vector` columns, so the extension must exist before
`alembic upgrade head`:

```bash
docker compose exec db psql -U verity -d verity -c 'CREATE EXTENSION IF NOT EXISTS vector'
```

This is idempotent; running it again is harmless.

## Run the tests

```bash
cd backend
TMPDIR=/tmp python -m pytest -q
```

`TMPDIR=/tmp` is required in this repository. It lives on a 9p/drvfs mount of
Windows `D:`; with the inherited `TMPDIR`, pytest puts `tmp_path` on that mount
and reports "no tests ran" alongside a misleading `FileNotFoundError`.

Tests need **no network and no provider key**. Anything that would call a model
is behind an interface with a deterministic fake in `verity/llm/provider.py`.

## Where to look first

1. `verity/engine/chunking.py` — how a document becomes searchable units.
2. `verity/worker/ingest.py` — the five-stage ingestion pipeline.
3. `verity/engine/retrieval/hybrid.py` — how search combines two rankings.
4. `verity/agent/loop.py` — how a question becomes an answer with citations.
