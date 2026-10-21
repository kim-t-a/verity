# `verity.worker` — the ingestion pipeline

`ingest.py` turns a directory of documents into embedded, searchable chunks in
PostgreSQL. It is the orchestrator; the real work lives in `verity.engine` and
`verity.db`.

## The five stages

```
discover → read → chunk → embed → store
```

For each file:

1. **Discover** — `engine.ingestion.walker` lists files and skipped entries.
2. **Read** — `engine.ingestion.readers.read_file` returns text plus spans.
3. **Chunk** — `engine.chunking.chunks_for` splits the text, preserving spans.
4. **Embed** — the configured `EmbeddingProvider` turns chunks into vectors.
5. **Store** — `db.chunk_repository.save_chunks` writes both.

## What the worker adds on top

Before the five stages it hashes the file (`sha256`) and skips duplicates in
the same workspace. After them it records the document's status and error, so
the API can report honestly what happened.

## The status lifecycle

```
pending → processing → ready
                    ↘ failed
```

- `pending` — a row exists, work has not finished.
- `ready` — **every** chunk was embedded and stored. This is invariant I5: a
  document with holes in its index must never look ready.
- `failed` — the error is stored on the row and returned to the caller.

## Per-file isolation

One bad file costs that file and nothing else. The walker never raises on an
unreadable path; each reader returns a status instead of throwing; the worker
catches per-file failures and continues. A corpus of a thousand documents is
not lost to one malformed one.

## Running it

The API's ingestion route (`POST /api/ingestion`) runs this synchronously for
simplicity. The `jobs` table exists for a real queue using
`FOR UPDATE SKIP LOCKED`; wiring a long-running worker process to it is a later
step.
