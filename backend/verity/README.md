# The `verity` package

This is the application. It is split by **job**, not by file type.

```
verity/
├── engine/       pure document logic: discovery, reading, chunking, search
├── db/           PostgreSQL connection and chunk storage
├── worker/       runs the ingestion pipeline
├── agent/        runs the question-answering loop
├── llm/          the language-model interface and the NVIDIA implementation
├── api/          FastAPI routes — the HTTP surface
└── provider_factory.py   chooses fake vs real providers from environment
```

A useful rule: **`engine/` is the only subpackage that is a library.** It knows
nothing about PostgreSQL, FastAPI, or API keys, and it has no third-party
dependencies. Everything else is product code that may import the engine, but
the engine never imports them.

## The two pipelines

### 1. Ingestion — runs once per document

```
file → discover → read → chunk → embed → store
        (engine)  (engine) (engine) (provider) (db)
```

Orchestrated by `worker/ingest.py`:

| Stage | Code |
|---|---|
| discover | `engine/ingestion/walker.py` |
| detect the real format | `engine/ingestion/filetypes.py` |
| read text + locations | `engine/ingestion/readers/` |
| split into chunks | `engine/chunking.py` |
| text → vectors | `engine/embedding_provider.py` (interface) and `engine/nvidia_embedding_provider.py` (real) |
| save chunks + vectors | `db/chunk_repository.py` |

### 2. Question answering — runs on every question

```
question → retrieve (keyword + vector) → fuse → top-k chunks
        → prompt with chunk IDs → LLM decides
        → verify citations → answer / clarify / refuse
```

| Stage | Code |
|---|---|
| keyword search (BM25) | `engine/retrieval/keyword.py` |
| vector search (cosine) | `engine/retrieval/vector.py` |
| merge rankings (RRF) | `engine/retrieval/hybrid.py` |
| the loop and its decisions | `agent/loop.py` |
| the model call | `llm/nvidia_provider.py` |
| HTTP endpoints | `api/sessions.py` |

## Where do I change X?

| I want to change… | Open |
|---|---|
| how text is split into chunks | `engine/chunking.py` |
| chunk size and other limits | `engine/config.py` |
| support for a new file format | `engine/ingestion/readers/` |
| the ingestion pipeline | `worker/ingest.py` |
| how chunks are stored in Postgres | `db/chunk_repository.py` |
| keyword / vector / combined search | `engine/retrieval/` |
| answer, clarify, or refuse behaviour | `agent/loop.py` |
| a model call or prompt | `llm/` |
| an HTTP endpoint | `api/` |
| the database schema | `../migrations/` |

## Names that used to collide

These were renamed to make scanning the package easier:

| Old name | New name | Why |
|---|---|---|
| `engine/chunkstore.py` | `engine/index_types.py` | It holds in-memory index types, not a database store. |
| `engine/embeddings.py` | `engine/embedding_provider.py` | It is the interface, not an implementation. |
| `engine/nvidia_embeddings.py` | `engine/nvidia_embedding_provider.py` | Says explicitly that it is a provider. |
| `db/chunkstore.py` | `db/chunk_repository.py` | A repository reads/writes database rows. |
| `providers.py` | `provider_factory.py` | It is a factory that picks an implementation. |
