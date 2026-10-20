"""Ingestion API route.

Triggers corpus ingestion from the API. In practice this would be a job
enqueued to the worker; here it runs the ingestion synchronously for
simplicity and returns the report.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from verity.worker.ingest import ingest_corpus, IngestionReport
from verity.llm.provider import FakeEmbeddingProvider

router = APIRouter(prefix="/api/ingestion")


class IngestRequest(BaseModel):
    corpus_path: str
    workspace_id: int = 1


class IngestResponse(BaseModel):
    document_count: int
    ready_count: int
    failed_count: int
    total_chunks: int
    total_skipped: int
    documents: list[dict[str, Any]]
    skipped: list[str]

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_report(cls, report: IngestionReport) -> "IngestResponse":
        return cls(
            document_count=report.document_count,
            ready_count=report.ready_count,
            failed_count=report.failed_count,
            total_chunks=report.total_chunks,
            total_skipped=report.total_skipped,
            documents=[
                {
                    "rel_path": d.rel_path,
                    "media_type": str(d.media_type),
                    "status": d.status,
                    "chunk_count": d.chunk_count,
                    "error": d.error,
                }
                for d in report.documents
            ],
            skipped=report.skipped,
        )


from typing import Any  # noqa: E402


@router.post("", response_model=IngestResponse, tags=["ingestion"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest all documents in a corpus directory.

    Runs walk → read → chunk → embed → store. Returns a report of what
    was ingested and what was skipped.
    """
    import os
    from pathlib import Path

    corpus_path = Path(request.corpus_path)
    if not corpus_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Corpus path not found: {corpus_path}"
        )

    provider = FakeEmbeddingProvider()
    report = ingest_corpus(corpus_path, request.workspace_id, provider)
    return IngestResponse.from_report(report)
