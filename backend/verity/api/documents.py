"""Document API routes.

Endpoints for uploading, listing, viewing, and deleting documents.
Upload returns 202 Accepted — honest for "accepted, not yet done" — and
ingestion happens asynchronously via the worker.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from verity.db import get_connection
from verity.db.chunkstore import save_chunks, delete_chunks_for_document
from verity.engine.ingestion import walker
from verity.engine.ingestion.readers import read_file
from verity.engine.chunking import chunks_for
from verity.engine.models import MediaType, ReadStatus
from verity.llm.provider import FakeProvider, FakeEmbeddingProvider
from verity.worker.ingest import _sha256, _insert_document, _insert_document_version

router = APIRouter(prefix="/api/workspaces/{workspace_id}/documents")

#: Where uploaded files are stored on disk.
UPLOAD_DIR = Path(os.environ.get("VERITY_UPLOAD_DIR", "/tmp/verity-uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DocumentStatus(BaseModel):
    id: int
    filename: str
    content_type: str
    size_bytes: int
    status: str  # pending | processing | ready | failed
    error: str | None = None
    chunk_count: int = 0
    created_at: str
    processed_at: str | None = None


class DocumentList(BaseModel):
    documents: list[DocumentStatus]


@router.post("", status_code=202, tags=["documents"])
async def upload_document(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a document for ingestion.

    Returns 202 Accepted — the file is accepted but not yet processed.
    Ingestion runs in a background task. Poll GET /documents for status.
    """
    # Read and hash the file.
    contents = await file.read()
    file_sha256 = hashlib.sha256(contents).hexdigest()

    # Check for duplicate.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status FROM documents
                WHERE workspace_id = %s AND sha256 = %s
                LIMIT 1
                """,
                (workspace_id, file_sha256),
            )
            row = cur.fetchone()
            if row:
                return {
                    "document_id": row[0],
                    "status": row[1],
                    "duplicate": True,
                    "message": "Document already exists with this content.",
                }

    # Store the file on disk.
    storage_path = UPLOAD_DIR / f"{file_sha256}_{file.filename or 'upload'}"
    with storage_path.open("wb") as f:
        f.write(contents)

    # Insert document row in pending status.
    with get_connection() as conn:
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
                    file.filename or "upload",
                    str(storage_path),
                    file.content_type or "application/octet-stream",
                    len(contents),
                    file_sha256,
                ),
            )
            document_id = cur.fetchone()[0]
            version_id = _insert_document_version(conn, document_id, file_sha256)
        conn.commit()

    # Kick off ingestion in the background.
    background_tasks.add_task(
        _ingest_background,
        document_id,
        str(storage_path),
        version_id,
        file_sha256,
    )

    return {
        "document_id": document_id,
        "status": "pending",
        "duplicate": False,
        "message": "Document accepted for ingestion.",
    }


@router.get("", response_model=DocumentList, tags=["documents"])
async def list_documents(workspace_id: int) -> DocumentList:
    """List all documents in a workspace with their processing status."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id, d.filename, d.content_type, d.size_bytes,
                       d.status, d.error, d.created_at, d.processed_at,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.workspace_id = %s
                GROUP BY d.id, d.filename, d.content_type, d.size_bytes,
                         d.status, d.error, d.created_at, d.processed_at
                ORDER BY d.created_at DESC
                """,
                (workspace_id,),
            )
            rows = cur.fetchall()

    documents = [
        DocumentStatus(
            id=row[0],
            filename=row[1],
            content_type=row[2],
            size_bytes=row[3],
            status=row[4],
            error=row[5],
            chunk_count=row[8] or 0,
            created_at=row[6].isoformat() if row[6] else "",
            processed_at=row[7].isoformat() if row[7] else None,
        )
        for row in rows
    ]
    return DocumentList(documents=documents)


@router.get("/{document_id}", tags=["documents"])
async def get_document(workspace_id: int, document_id: int) -> dict[str, Any]:
    """Get document detail including status and chunk count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id, d.filename, d.content_type, d.size_bytes,
                       d.status, d.error, d.created_at, d.processed_at,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = %s AND d.workspace_id = %s
                GROUP BY d.id, d.filename, d.content_type, d.size_bytes,
                         d.status, d.error, d.created_at, d.processed_at
                """,
                (document_id, workspace_id),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": row[0],
        "filename": row[1],
        "content_type": row[2],
        "size_bytes": row[3],
        "status": row[4],
        "error": row[5],
        "chunk_count": row[8] or 0,
        "created_at": row[6].isoformat() if row[6] else "",
        "processed_at": row[7].isoformat() if row[7] else None,
    }


@router.delete("/{document_id}", status_code=204, tags=["documents"])
async def delete_document(workspace_id: int, document_id: int) -> JSONResponse:
    """Delete a document and all its chunks, derived memories, and cached copies.

    Cascades: chunks are deleted (ON DELETE CASCADE), document_versions
    are deleted, derived memories are cleared. The stored file is also
    removed from disk.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get the storage path before deleting.
            cur.execute(
                "SELECT storage_path FROM documents WHERE id = %s AND workspace_id = %s",
                (document_id, workspace_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")
            storage_path = row[0]

            # Delete cascades: chunks, versions, memories via FK constraints.
            cur.execute(
                "DELETE FROM documents WHERE id = %s AND workspace_id = %s",
                (document_id, workspace_id),
            )
        conn.commit()

    # Remove the stored file from disk.
    try:
        Path(storage_path).unlink(missing_ok=True)
    except OSError:
        pass  # Best effort — the DB record is gone regardless.

    return JSONResponse(status_code=204, content=None)


async def _ingest_background(
    document_id: int,
    storage_path: str,
    version_id: int,
    file_sha256: str,
) -> None:
    """Background task: read, chunk, embed, store one document."""
    from pathlib import Path

    from verity.engine.models import DiscoveredFile
    from verity.engine.ingestion.filetypes import sniff
    from verity.db import get_connection as _get_connection

    path = Path(storage_path)
    try:
        # Classify the file type.
        with path.open("rb") as f:
            head = f.read(4096)
        verdict = sniff(head, path.suffix)
        media_type = verdict.media_type

        # Build a DiscoveredFile for the reader.
        size = path.stat().st_size
        discovered = DiscoveredFile(
            path=path,
            rel_path=path.name,
            media_type=media_type,
            extension=path.suffix.lower(),
            size_bytes=size,
            magic_agrees=verdict.agrees,
        )

        # Read → spans.
        result = read_file(discovered)
        if result.status is not ReadStatus.OK:
            _mark_failed(document_id, result.detail or "no text extracted")
            return

        # Chunk.
        chunks = list(chunks_for(result))
        if not chunks:
            _mark_failed(document_id, "no indexable text")
            return

        # Embed (fake provider for now — real NVIDIA provider needs credentials).
        provider = FakeEmbeddingProvider()
        texts = [c.text for c in chunks]
        vector_lists = provider.embed(texts)

        # Store.
        save_chunks(document_id, version_id, chunks, vector_lists)

        # Mark ready.
        with _get_connection() as conn:
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

    except Exception as exc:  # noqa: BLE001
        _mark_failed(document_id, f"{type(exc).__name__}: {exc}")


def _mark_failed(document_id: int, error: str) -> None:
    """Mark a document as failed with an error message."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                (error, document_id),
            )
        conn.commit()
