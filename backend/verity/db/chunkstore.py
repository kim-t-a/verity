"""PostgreSQL-backed chunk store.

Replaces the challenge-era JSONL on-disk index (``verity/engine/chunkstore.py``)
with PostgreSQL tables. Writes chunks and their embedding vectors to the
``chunks`` table and reads them back as
:class:`~verity.engine.chunkstore.ChunkIndex` objects so the existing
retrieval layer works unchanged.

The engine (``verity.engine``) is pure standard library and knows nothing
about this module -- the product layer owns storage.
"""

from __future__ import annotations

from typing import Sequence

from psycopg2.extras import execute_values

from verity.db import get_connection
from verity.engine.chunkstore import ChunkIndex, IndexManifest
from verity.engine.models import Chunk, MediaType


def _empty_index(
    provider_id: str | None = None, dimension: int | None = None
) -> ChunkIndex:
    """Return an empty ChunkIndex with the given manifest metadata."""
    return ChunkIndex(
        manifest=IndexManifest(
            format_version=1,
            created_utc="",
            corpus_fingerprint="",
            chunk_count=0,
            file_count=0,
            provider_id=provider_id,
            dimension=dimension,
            status_counts={},
            skipped_count=0,
        ),
        chunks=(),
        vectors=(),
    )


def save_chunks(
    document_id: int,
    version_id: int,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Store chunks and their vectors for a document.

    Vectors are positional with chunks -- entry *i* is the vector for chunk
    *i*. All inserts happen in one transaction.

    Raises:
        ValueError: the number of chunks and vectors do not match.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"{len(chunks)} chunks but {len(vectors)} vectors; "
            "they must pair one-to-one"
        )

    rows = [
        (
            document_id,
            version_id,
            chunk.ordinal,
            chunk.text,
            chunk.location,
            len(chunk.text),
            list(vector),
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO chunks
                   (document_id, version_id, ordinal, text, location,
                    token_count, embedding)
                   VALUES %s""",
                rows,
            )


def _parse_pgvector(embedding) -> list[float] | None:
    """Parse a pgvector value into a list of floats.

    psycopg2 returns pgvector values as strings like ``'[0.1,0.2,-0.3]'``.
    If the value is already a list (some adapters), return it as-is.
    Returns ``None`` for NULL/empty.
    """
    if embedding is None:
        return None
    if isinstance(embedding, str):
        # Strip brackets and split on commas.
        embedding = embedding.strip("[]")
        if not embedding:
            return None
        return [float(v) for v in embedding.split(",")]
    # Already a list-like (some psycopg2/pgvector adapters return this).
    return [float(v) for v in embedding]


def _row_to_chunk(row: tuple) -> Chunk:
    """Convert a database row to a :class:`~verity.engine.models.Chunk`."""
    (
        _id,
        _doc_id,
        _ver_id,
        ordinal,
        text,
        location,
        _token_count,
        _embedding,
        filename,
        content_type,
    ) = row
    return Chunk(
        rel_path=filename,
        media_type=MediaType(content_type),
        location=location,
        ordinal=ordinal,
        text=text,
    )


def load_all_chunks(
    provider_id: str | None = None,
    dimension: int | None = None,
) -> ChunkIndex:
    """Load all chunks as a :class:`ChunkIndex` for retrieval.

    Chunks are ordered by document filename then ordinal for stable,
    reproducible ordering -- the same order every time, which matters for
    reproducible rankings.

    Args:
        provider_id: The embedding provider that built the vectors. Recorded
            in the manifest so
            :class:`~verity.engine.retrieval.vector.VectorRetriever` can
            validate the provider matches before scoring.
        dimension: The vector dimension. Inferred from the vectors if not
            given and every chunk has a vector.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.version_id, c.ordinal, c.text,
                       c.location, c.token_count, c.embedding,
                       d.filename, d.content_type
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY d.filename, c.ordinal
                """
            )
            rows = cur.fetchall()

    if not rows:
        return _empty_index(provider_id, dimension)

    chunks: list[Chunk] = []
    vectors: list[tuple[float, ...]] = []
    all_have_vectors = True
    inferred_dimension = dimension

    for row in rows:
        chunks.append(_row_to_chunk(row))
        embedding = row[7]
        if embedding is not None:
            parsed = _parse_pgvector(embedding)
            # embedding is not None, so _parse_pgvector returns a list, not None.
            assert parsed is not None
            vec = tuple(parsed)
            vectors.append(vec)
            if inferred_dimension is None:
                inferred_dimension = len(vec)
        else:
            all_have_vectors = False
            vectors.append(())

    # If not every chunk has a vector, clear the list so the retrieval layer
    # treats this as a lexical-only index. A mixed state would pair vectors
    # with the wrong chunks.
    if not all_have_vectors:
        vectors = []

    if inferred_dimension is None and vectors:
        inferred_dimension = len(vectors[0])

    manifest = IndexManifest(
        format_version=1,
        created_utc="",
        corpus_fingerprint="",
        chunk_count=len(chunks),
        file_count=len({chunk.rel_path for chunk in chunks}),
        provider_id=provider_id,
        dimension=inferred_dimension,
        status_counts={},
        skipped_count=0,
    )

    return ChunkIndex(
        manifest=manifest,
        chunks=tuple(chunks),
        vectors=tuple(vectors),
    )


def delete_chunks_for_document(document_id: int) -> None:
    """Delete all chunks for a document.

    Called when a document is deleted or re-ingested, so stale chunks do
    not linger in the index.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE document_id = %s",
                (document_id,),
            )


def chunk_count() -> int:
    """Return the total number of chunks in the store."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            row = cur.fetchone()
            return row[0] if row is not None else 0
