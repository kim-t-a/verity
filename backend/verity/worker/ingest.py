"""Document ingestion worker.

Turns a directory of documents into searchable, embedded chunks stored in
PostgreSQL. The pipeline is:

    discover → read → chunk → embed → store

Each stage is defensive: one bad file costs that file and nothing else. The
engine (``verity.engine``) remains pure standard library and knows nothing
about this module -- the product layer owns ingestion.

The worker is callable directly or from a job processor that pulls
``ingest`` jobs off the ``jobs`` table with ``FOR UPDATE SKIP LOCKED``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from verity.db import get_connection
from verity.db.chunkstore import save_chunks
from verity.engine.embeddings import EmbeddingProvider
from verity.engine.chunking import chunks_for
from verity.engine.ingestion import walker
from verity.engine.ingestion.readers import read_file
from verity.engine.models import MediaType, ReadStatus
from verity.engine.nvidia_embeddings import NvidiaEmbeddingProvider, NvidiaEmbeddingError


@dataclass(frozen=True)
class IngestedDocument:
    """The outcome of ingesting one document."""

    rel_path: str
    media_type: MediaType
    status: str
    chunk_count: int
    error: str | None = None


@dataclass(frozen=True)
class IngestionReport:
    """The outcome of ingesting an entire corpus."""

    workspace_id: int
    documents: list[IngestedDocument]
    skipped: list[str]

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def ready_count(self) -> int:
        return sum(1 for d in self.documents if d.status == "ready")

    @property
    def failed_count(self) -> int:
        return sum(1 for d in self.documents if d.status == "failed")

    @property
    def total_chunks(self) -> int:
        return sum(d.chunk_count for d in self.documents)

    @property
    def total_skipped(self) -> int:
        return len(self.skipped)


def _sha256(path: Path) -> str:
    """Compute the sha256 digest of a file, reading in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_duplicate(
    conn,
    workspace_id: int,
    sha256: str,
) -> int | None:
    """Return the id of an existing document with this sha256 in the workspace,
    or ``None`` when no duplicate exists.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM documents
            WHERE workspace_id = %s AND sha256 = %s
            LIMIT 1
            """,
            (workspace_id, sha256),
        )
        row = cur.fetchone()
        return row[0] if row is not None else None


def _insert_document(
    conn,
    workspace_id: int,
    discovered,
    sha256: str,
) -> int:
    """Insert a ``documents`` row in ``pending`` status and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (workspace_id, filename, storage_path, content_type,
                 size_bytes, sha256, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            RETURNING id
            """,
            (
                workspace_id,
                discovered.rel_path,
                str(discovered.path),
                str(discovered.media_type),
                discovered.size_bytes,
                sha256,
            ),
        )
        return cur.fetchone()[0]


def _insert_document_version(
    conn,
    document_id: int,
    sha256: str,
) -> int:
    """Insert a ``document_versions`` row and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document_versions
                (document_id, version, sha256)
            VALUES (%s, 1, %s)
            RETURNING id
            """,
            (document_id, sha256),
        )
        return cur.fetchone()[0]


def _ingest_one(
    conn,
    workspace_id: int,
    discovered,
    provider: EmbeddingProvider,
) -> IngestedDocument:
    """Ingest one discovered file into the database.

    The document row is created first so we always have a place to record the
    outcome. If anything fails after that, the document is marked ``failed``
    with the error text rather than raising and aborting the whole batch.
    """
    sha256 = _sha256(discovered.path)

    # Duplicate detection: same workspace, same content hash → skip.
    duplicate_id = _find_duplicate(conn, workspace_id, sha256)
    if duplicate_id is not None:
        return IngestedDocument(
            rel_path=discovered.rel_path,
            media_type=discovered.media_type,
            status="duplicate",
            chunk_count=0,
            error=f"duplicate of document {duplicate_id}",
        )

    document_id = _insert_document(conn, workspace_id, discovered, sha256)
    version_id = _insert_document_version(conn, document_id, sha256)
    conn.commit()

    try:
        # Mark processing.
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status = 'processing' WHERE id = %s",
                (document_id,),
            )
        conn.commit()

        # Read → spans with locations.
        result = read_file(discovered)
        if result.status is not ReadStatus.OK:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                    (result.detail or "no text extracted", document_id),
                )
            conn.commit()
            return IngestedDocument(
                rel_path=discovered.rel_path,
                media_type=discovered.media_type,
                status="failed",
                chunk_count=0,
                error=result.detail or "no text extracted",
            )

        # Chunk → Chunk objects.
        chunks = chunks_for(result)
        if not chunks:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                    ("the document contains no indexable text", document_id),
                )
            conn.commit()
            return IngestedDocument(
                rel_path=discovered.rel_path,
                media_type=discovered.media_type,
                status="failed",
                chunk_count=0,
                error="the document contains no indexable text",
            )

        # Embed → one vector per chunk, in order.
        texts = [chunk.text for chunk in chunks]
        try:
            vector_lists = provider.embed(texts)
        except NvidiaEmbeddingError as exc:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                    (f"embedding failed: {exc}", document_id),
                )
            conn.commit()
            return IngestedDocument(
                rel_path=discovered.rel_path,
                media_type=discovered.media_type,
                status="failed",
                chunk_count=0,
                error=f"embedding failed: {exc}",
            )

        # Store → PostgreSQL chunks table.
        save_chunks(document_id, version_id, chunks, vector_lists)

        # Mark ready.
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET status = 'ready', processed_at = NOW()
                WHERE id = %s
                """,
                (document_id,),
            )
        conn.commit()

        return IngestedDocument(
            rel_path=discovered.rel_path,
            media_type=discovered.media_type,
            status="ready",
            chunk_count=len(chunks),
        )

    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                (f"{type(exc).__name__}: {exc}", document_id),
            )
        conn.commit()
        return IngestedDocument(
            rel_path=discovered.rel_path,
            media_type=discovered.media_type,
            status="failed",
            chunk_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def ingest_corpus(
    corpus_root: Path | str,
    workspace_id: int = 1,
    provider: EmbeddingProvider | None = None,
) -> IngestionReport:
    """Ingest every readable file in ``corpus_root`` into ``workspace_id``.

    Walks the directory tree, reads each file, chunks it, embeds each chunk
    with ``provider`` (default: :class:`NvidiaEmbeddingProvider`), and stores
    chunks + vectors in PostgreSQL.

    Duplicate detection: if a document with the same sha256 already exists in
    the workspace, it is reported as ``duplicate`` and not re-indexed.

    Errors are caught per file -- one malformed document never aborts the
    whole corpus. The report lists every document's outcome and every skipped
    file.

    Args:
        corpus_root: Directory to ingest.
        workspace_id: The workspace to associate documents with.
        provider: Embedding backend. Defaults to ``NvidiaEmbeddingProvider()``
            which reads ``NVIDIA_API_KEY`` and ``NVIDIA_API_BASE_URL`` from
            the environment.

    Returns:
        An :class:`IngestionReport` with one entry per discovered file plus
        the skipped-file ledger from the walker.
    """
    if provider is None:
        provider = NvidiaEmbeddingProvider()

    discovery = walker.discover(corpus_root)
    documents: list[IngestedDocument] = []

    with get_connection() as conn:
        for discovered in discovery.files:
            documents.append(_ingest_one(conn, workspace_id, discovered, provider))

    skipped = [f"{entry.rel_path}: {entry.detail}" for entry in discovery.skipped]

    return IngestionReport(
        workspace_id=workspace_id,
        documents=documents,
        skipped=skipped,
    )


def ingest_sequences(
    corpus_root: Path | str,
    workspace_id: int = 1,
    provider: EmbeddingProvider | None = None,
) -> list[IngestedDocument]:
    """Convenience wrapper returning just the per-document outcomes.

    Useful when the caller does not need the skipped-file ledger.
    """
    return ingest_corpus(corpus_root, workspace_id, provider).documents
