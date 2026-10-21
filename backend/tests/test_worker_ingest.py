"""Integration tests for the ingestion worker.

These tests ingest real documents through the full pipeline:

    discover → read → chunk → embed → store

They use a hand-written fake embedding provider so no NVIDIA credentials or
network access are needed. Tests that write to PostgreSQL are skipped when the
database is not running.
"""

from __future__ import annotations

import psycopg2
import pytest

from verity.engine.models import MediaType, ReadResult, ReadStatus, TextSpan
from verity.engine.nvidia_embedding_provider import EmbeddingProvider
from verity.worker.ingest import ingest_corpus, IngestionReport


class FakeProvider:
    """Deterministic fake embedding provider for offline tests."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return "fake-provider"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._call_count += 1
        return [[float(i) for i in range(self._dimension)] for _ in texts]


@pytest.fixture(autouse=True)
def clean_documents(db_conn: psycopg2.extensions.connection):
    """Remove all documents and chunks before each test so tests don't collide
    on duplicate sha256 detection from previous runs."""
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM document_versions")
        cur.execute("DELETE FROM documents WHERE workspace_id = 1")
    db_conn.commit()
    yield


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        dbname="verity",
        user="verity",
        password="verity_password",
    )


@pytest.fixture
def db_conn() -> psycopg2.extensions.connection:
    try:
        conn = _connect()
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available")
    yield conn
    conn.close()


def _result_for(texts: list[str], locations: list[str], rel_path: str) -> ReadResult:
    spans = [
        TextSpan(text=t, location=loc, ordinal=i)
        for i, (t, loc) in enumerate(zip(texts, locations))
    ]
    return ReadResult.of(rel_path, MediaType.TEXT, spans)


@pytest.fixture
def ingest_with_fake(db_conn: psycopg2.extensions.connection):
    """Return a helper that ingests a synthetic corpus with a fake provider.

    The helper writes files to a temp directory, inserts a workspace row, and
    calls ingest_corpus with FakeProvider. Returns the IngestionReport.
    """

    def ingest(
        contents: dict[str, str],
        workspace_id: int = 1,
        dimension: int = 4,
    ) -> IngestionReport:
        import tempfile
        from pathlib import Path

        root = Path(tempfile.mkdtemp(prefix="verity-ingest-test-"))
        for rel_path, content in contents.items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")

        # Ensure the workspace exists.
        with db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO workspaces (id, name) VALUES (%s, 'test') "
                "ON CONFLICT (id) DO NOTHING",
                (workspace_id,),
            )
        db_conn.commit()

        provider = FakeProvider(dimension=dimension)
        return ingest_corpus(root, workspace_id, provider)

    return ingest


def test_ingest_small_corpus(
    db_conn: psycopg2.extensions.connection,
    ingest_with_fake,
) -> None:
    """A small corpus ingests end-to-end: all documents ready, chunks stored."""
    report = ingest_with_fake(
        {
            "specs/datasheet.txt": "Maximum junction temperature 94 C\nBoard power 410 W\n",
            "support/notes.txt": "Notice period is 30 days.\n",
        }
    )

    assert isinstance(report, IngestionReport)
    assert report.document_count == 2
    assert report.ready_count == 2
    assert report.failed_count == 0
    assert report.total_chunks >= 2

    # Every document is ready.
    statuses = {d.status for d in report.documents}
    assert statuses == {"ready"}

    # Each document's rel_path is preserved.
    rel_paths = {d.rel_path for d in report.documents}
    assert "specs/datasheet.txt" in rel_paths
    assert "support/notes.txt" in rel_paths


def test_duplicate_document_is_skipped(
    db_conn: psycopg2.extensions.connection,
    ingest_with_fake,
) -> None:
    """The same file content in the same workspace is reported as duplicate."""
    report1 = ingest_with_fake({"specs/datasheet.txt": "value 94\n"})
    assert report1.ready_count == 1

    # Re-ingest the same content → duplicate.
    report2 = ingest_with_fake({"specs/datasheet.txt": "value 94\n"})
    assert report2.document_count == 1
    assert report2.documents[0].status == "duplicate"
    assert report2.documents[0].chunk_count == 0


def test_bad_file_does_not_abort_batch(
    db_conn: psycopg2.extensions.connection,
    ingest_with_fake,
) -> None:
    """A file the reader cannot parse is marked failed; siblings still ingest."""
    report = ingest_with_fake(
        {
            "good.txt": "readable content value 94\n",
            "bad.pdf": b"%PDF-1.4\ngarbage",  # not a real PDF, reader will fail
        }
    )

    statuses = {d.status for d in report.documents}
    assert "ready" in statuses
    assert "failed" in statuses

    # The good document made it through.
    good = [d for d in report.documents if d.status == "ready"]
    assert len(good) == 1
    assert good[0].rel_path == "good.txt"
    assert good[0].chunk_count > 0


def test_document_with_no_text_is_failed(
    db_conn: psycopg2.extensions.connection,
    ingest_with_fake,
) -> None:
    """A file that yields no text is marked failed, not ready with zero chunks."""
    report = ingest_with_fake({"empty.txt": "   \n   \n"})
    # Whitespace-only text → ReadResult.of downgrades to EMPTY → no chunks → failed.
    assert report.document_count == 1
    assert report.documents[0].status == "failed"
    assert report.documents[0].chunk_count == 0


def test_empty_corpus_yields_empty_report(
    db_conn: psycopg2.extensions.connection,
    ingest_with_fake,
) -> None:
    """An empty directory produces no documents and no errors."""
    report = ingest_with_fake({})
    assert report.document_count == 0
    assert report.ready_count == 0
    assert report.failed_count == 0
    assert report.total_chunks == 0


def test_chunks_are_stored_in_database(
    db_conn: psycopg2.extensions.connection,
    ingest_with_fake,
) -> None:
    """Ingested chunks are queryable from the chunks table with embeddings."""
    report = ingest_with_fake({"specs/datasheet.txt": "Maximum junction temperature 94 C\n"})
    assert report.ready_count == 1

    # Look up the document id by filename.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE filename = %s",
            ("specs/datasheet.txt",),
        )
        row = cur.fetchone()
        assert row is not None
        doc_id = row[0]

        # Verify chunks exist with embeddings.
        cur.execute(
            "SELECT COUNT(*), COUNT(embedding) FROM chunks WHERE document_id = %s",
            (doc_id,),
        )
        total, with_vectors = cur.fetchone()
        doc = report.documents[0]
        assert total == doc.chunk_count
        assert with_vectors == doc.chunk_count
