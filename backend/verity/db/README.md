# `verity.db` — PostgreSQL, and nothing above it

The engine does not know this package exists. Everything here is product code:
connection management and persistence.

| Module | Responsibility |
|---|---|
| `__init__.py` | Connection pool (`psycopg2`), `get_connection()`, `ensure_vector_extension()`. |
| `chunk_repository.py` | Save/load/delete chunks and their vectors. |

## Connection handling

```python
from verity.db import get_connection

with get_connection() as conn:       # pooled connection
    with conn.cursor() as cur:
        cur.execute(...)
    # commits on success, rolls back on an exception
```

The pool is created lazily. Host defaults to `127.0.0.1`, not `localhost`, on
purpose: on Windows, `localhost` resolves to IPv6 first and costs a ~21 s TCP
fallback per connection.

## The chunk repository

| Function | Purpose |
|---|---|
| `save_chunks(document_id, version_id, chunks, vectors)` | Insert chunks and their embeddings; raises if counts differ. |
| `load_all_chunks()` | Build a `ChunkIndex` from every stored chunk. |
| `delete_chunks_for_document(document_id)` | Remove a document's chunks. |
| `chunk_count()` | Count stored chunks. |

A `ChunkIndex` is an engine type (`engine/index_types.py`). The repository is
the bridge between that in-memory shape and the `chunks` table.

## The table

The schema is created by `migrations/versions/*_initial_schema.py`:

- `chunks.text` — the chunk's text.
- `chunks.embedding vector` — the dense vector (width not constrained).
- `chunks.tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`
  — Postgres keeps the full-text index in sync automatically.
- `documents`, `document_versions`, `sessions`, `messages`, `citations`,
  `memories`, `jobs` — the rest of the product.

## pgvector

The initial migration runs `CREATE EXTENSION IF NOT EXISTS vector` before it
creates the `chunks` table, so a fresh database needs no manual step.
`ensure_vector_extension()` remains available for callers that need to enable
the extension independently.
