# `verity.engine` — the pure document engine

Everything here is standard library only. No database, no HTTP, no API keys,
no model SDK. That is deliberate: the engine can be imported and tested with
nothing installed.

## Modules, top to bottom

| Module | Responsibility | Key entry point |
|---|---|---|
| `config.py` | Every tunable number in one place. | constants such as `MAX_CHUNK_CHARS`, `RETRIEVAL_LIMIT` |
| `models.py` | Shared immutable data types. | `DiscoveredFile`, `ReadResult`, `TextSpan`, `Chunk` |
| `normalisation.py` | The one canonical form used for comparisons. | `normalise` |
| `tokenisation.py` | Text → terms that search can match. | `tokenise` |
| `chunking.py` | Read result → chunks that keep provenance. | `chunks_for` |
| `index_types.py` | The in-memory index types and the legacy JSON/binary save/load. | `ChunkIndex`, `IndexManifest`, `build_manifest` |
| `embedding_provider.py` | The embedding **interface** only. | `EmbeddingProvider` Protocol |
| `nvidia_embedding_provider.py` | The real embedding client (stdlib HTTP). | `NvidiaEmbeddingProvider` |
| `ingestion/` | Discover files and read them into text. | `walker`, `filetypes`, `readers` |
| `retrieval/` | Search chunks three ways. | `KeywordRetriever`, `VectorRetriever`, `HybridRetriever` |

## Suggested reading order

If you are learning retrieval, read these in order:

1. `models.py` — what data moves through the system.
2. `chunking.py` — how a document is cut, and why boundaries matter.
3. `retrieval/keyword.py` — search by exact words (BM25).
4. `retrieval/vector.py` — search by meaning (cosine similarity).
5. `retrieval/hybrid.py` — combine the two without one drowning the other.

## The one rule that explains the shape of this code

Retrieval is scored on whether the right chunk appears at all. A chunk that is
never retrieved can never be cited, and no later stage can recover it. That is
why chunking and retrieval receive so much attention here.

## Hostile input

Every reader and walker treats its input as untrusted: unbounded reads are
capped, decompression bombs are refused, and one bad file never aborts the
corpus. A failure is a **value** (`ReadResult` with a status), not an
exception thrown across a boundary.
