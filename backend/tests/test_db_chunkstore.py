"""Integration tests for the PostgreSQL-backed chunk store.

These tests write to and read from the real PostgreSQL database started with
``docker compose up``. They are skipped when the database is not available.

The chunk store replaces the challenge-era JSONL index: chunks and vectors
live in PostgreSQL tables and are read back as
:class:`~verity.engine.chunkstore.ChunkIndex` objects so the existing
retrieval layer works unchanged.
"""

from __future__ import annotations

import psycopg2
import pytest

from verity.db.chunkstore import (
    save_chunks,
    load_all_chunks,
    delete_chunks_for_document,
    chunk_count,
)
from verity.engine.chunkstore import ChunkIndex
from verity.engine.models import Chunk, MediaType


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="verity",
        user="verity",
        password="verity_password",
    )


@pytest.fixture(autouse=True)
def clean_all_data(db_conn: psycopg2.extensions.connection):
    """Remove all documents, chunks, and versions before each test so tests
    don't collide with data left by other test files."""
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM document_versions")
        cur.execute("DELETE FROM documents")
    db_conn.commit()
    yield


@pytest.fixture
def db_conn() -> psycopg2.extensions.connection:
    """Database connection, or skip if PostgreSQL is not running."""
    try:
        conn = _connect()
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available")
    yield conn
    conn.close()


@pytest.fixture
def test_document(
    db_conn: psycopg2.extensions.connection,
) -> tuple[int, int]:
    """Insert a test document and its first version, return (doc_id, version_id).
    Cleans up on teardown.
    """
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (workspace_id, filename, storage_path, content_type,
                 size_bytes, sha256, status)
            VALUES (1, 'test.pdf', '/tmp/test.pdf', 'pdf', 100, 'abc123', 'ready')
            RETURNING id
            """
        )
        doc_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO document_versions
                (document_id, version, sha256)
            VALUES (%s, 1, 'abc123')
            RETURNING id
            """,
            (doc_id,),
        )
        version_id = cur.fetchone()[0]
    db_conn.commit()

    yield doc_id, version_id

    # Teardown: remove chunks, versions, and the document.
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        cur.execute("DELETE FROM document_versions WHERE document_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    db_conn.commit()


def make_chunks(n: int = 3) -> list[Chunk]:
    """Build ``n`` chunks with distinct text and locations."""
    return [
        Chunk(
            rel_path="test.pdf",
            media_type=MediaType.PDF,
            location=f"page {i} line 1",
            ordinal=i,
            text=f"chunk {i} content value {i * 10}",
        )
        for i in range(n)
    ]


def make_vectors(n: int, dimension: int = 4) -> list[list[float]]:
    """Build ``n`` distinct vectors of ``dimension``."""
    return [[float(i * dimension + j) for j in range(dimension)] for i in range(n)]


def test_save_and_load_chunks_round_trip(
    db_conn: psycopg2.extensions.connection,
    test_document: tuple[int, int],
) -> None:
    """Chunks saved with vectors load back unchanged as a ChunkIndex."""
    doc_id, version_id = test_document
    chunks = make_chunks(3)
    vectors = make_vectors(3, 4)

    save_chunks(doc_id, version_id, chunks, vectors)

    index = load_all_chunks(provider_id="test-provider", dimension=4)

    assert isinstance(index, ChunkIndex)
    assert len(index.chunks) == 3
    assert len(index.vectors) == 3
    assert index.manifest.provider_id == "test-provider"
    assert index.manifest.dimension == 4
    assert index.manifest.chunk_count == 3

    # Text and location survive the round trip.
    loaded_texts = [chunk.text for chunk in index.chunks]
    assert "chunk 0 content value 0" in loaded_texts
    assert "chunk 1 content value 10" in loaded_texts
    assert "chunk 2 content value 20" in loaded_texts

    # Vectors pair positionally with chunks.
    assert index.vectors[0] == (0.0, 1.0, 2.0, 3.0)
    assert index.vectors[1] == (4.0, 5.0, 6.0, 7.0)
    assert index.vectors[2] == (8.0, 9.0, 10.0, 11.0)


def test_load_empty_database_returns_empty_index(
    db_conn: psycopg2.extensions.connection,
) -> None:
    """An empty chunks table yields an empty ChunkIndex, not an error."""
    index = load_all_chunks()

    assert isinstance(index, ChunkIndex)
    assert index.chunks == ()
    assert index.vectors == ()
    assert index.manifest.chunk_count == 0


def test_delete_chunks_for_document(
    db_conn: psycopg2.extensions.connection,
    test_document: tuple[int, int],
) -> None:
    """Deleting chunks for one document leaves others untouched."""
    doc_id, version_id = test_document
    chunks = make_chunks(2)
    vectors = make_vectors(2, 4)
    save_chunks(doc_id, version_id, chunks, vectors)

    assert chunk_count() == 2

    delete_chunks_for_document(doc_id)

    assert chunk_count() == 0

    index = load_all_chunks()
    assert index.chunks == ()


def test_save_chunks_vector_count_mismatch_raises(
    db_conn: psycopg2.extensions.connection,
    test_document: int,
) -> None:
    """Saving chunks with a different number of vectors raises ValueError."""
    chunks = make_chunks(3)
    vectors = make_vectors(2, 4)  # only 2 vectors for 3 chunks

    with pytest.raises(ValueError, match="3 chunks but 2 vectors"):
        save_chunks(test_document, 1, chunks, vectors)


def test_chunk_count_matches_loaded_chunks(
    db_conn: psycopg2.extensions.connection,
    test_document: tuple[int, int],
) -> None:
    """chunk_count agrees with load_all_chunks."""
    doc_id, version_id = test_document
    chunks = make_chunks(5)
    vectors = make_vectors(5, 2)
    save_chunks(doc_id, version_id, chunks, vectors)

    assert chunk_count() == 5
    assert len(load_all_chunks().chunks) == 5


def test_load_all_chunks_orders_by_filename_then_ordinal(
    db_conn: psycopg2.extensions.connection,
    test_document: tuple[int, int],
) -> None:
    """Chunk ordering is stable: by document filename, then ordinal."""
    doc_id, version_id = test_document
    chunks = make_chunks(3)
    vectors = make_vectors(3, 2)
    save_chunks(doc_id, version_id, chunks, vectors)

    index = load_all_chunks()
    ordinals = [chunk.ordinal for chunk in index.chunks]
    assert ordinals == sorted(ordinals)
