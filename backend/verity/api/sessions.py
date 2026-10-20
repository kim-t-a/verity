"""Session and message API routes.

Endpoints for starting conversations and asking questions. The message
endpoint returns a discriminated union: ``{"type": "answer"}`` /
``{"type": "clarification"}`` / ``{"type": "refusal"}`` — the frontend
switches on the ``type`` field.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from verity.db import get_connection
from verity.db.chunkstore import load_all_chunks
from verity.agent.loop import AgentLoop, make_retriever_from_index
from verity.llm.provider import FakeProvider
from verity.engine.models import Chunk

router = APIRouter(prefix="/api/sessions")


class StartSession(BaseModel):
    workspace_id: int = 1


class SessionResponse(BaseModel):
    session_id: int


class AskRequest(BaseModel):
    question: str


class MessageResponse(BaseModel):
    type: str  # "answer" | "clarification" | "refusal"
    text: str
    citations: list[dict[str, Any]] = []
    searched: list[dict[str, str]] = []
    iterations: int = 0


@router.post("", response_model=SessionResponse, tags=["sessions"])
async def create_session(request: StartSession) -> SessionResponse:
    """Start a new conversation session."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (workspace_id, state, summary) "
                "VALUES (%s, '{}', NULL) RETURNING id",
                (request.workspace_id,),
            )
            session_id = cur.fetchone()[0]
        conn.commit()
    return SessionResponse(session_id=session_id)


@router.post("/{session_id}/messages", response_model=MessageResponse, tags=["sessions"])
async def ask_question(session_id: int, request: AskRequest) -> MessageResponse:
    """Ask a question. Returns answer / clarification / refusal.

    The agent loop retrieves relevant chunks, asks the model to decide,
    verifies citations, and returns a discriminated response.
    """
    # Verify the session exists.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Session not found")

    # Build retriever from all chunks in the database.
    index = load_all_chunks()
    if not index.chunks:
        return MessageResponse(
            type="refusal",
            text="No documents have been ingested yet. Upload some documents first.",
            searched=[],
            iterations=0,
        )

    retriever = make_retriever_from_index(list(index.chunks))
    provider = FakeProvider()
    loop = AgentLoop(retriever, provider)

    # Get conversation history for context.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = %s ORDER BY created_at ASC LIMIT 10",
                (session_id,),
            )
            history = [
                {"role": row[0], "content": row[1]}
                for row in cur.fetchall()
            ]

    decision = loop.run(request.question, session_history=history)

    # Store the user message.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, role, content, kind) "
                "VALUES (%s, 'user', %s, 'answer') RETURNING id",
                (session_id, request.question),
            )
            user_message_id = cur.fetchone()[0]

            # Store the assistant response.
            kind = decision.kind
            cur.execute(
                "INSERT INTO messages (session_id, role, content, kind) "
                "VALUES (%s, 'assistant', %s, %s) RETURNING id",
                (session_id, decision.text, kind),
            )
            assistant_message_id = cur.fetchone()[0]

            # Store citations for answer type.
            if decision.kind == "answer":
                for citation in decision.citations:
                    cur.execute(
                        "SELECT id FROM chunks WHERE rel_path = %s AND location = %s "
                        "LIMIT 1",
                        (citation["rel_path"], citation["location"]),
                    )
                    chunk_row = cur.fetchone()
                    if chunk_row:
                        cur.execute(
                            "INSERT INTO citations (message_id, chunk_id, document_id) "
                            "VALUES (%s, %s, "
                            "(SELECT id FROM documents WHERE id = %s LIMIT 1))",
                            (assistant_message_id, chunk_row[0], chunk_row[0]),
                        )
            conn.commit()

    # Build the response.
    searched = [
        {"rel_path": c.chunk.rel_path, "location": c.chunk.location}
        for c in decision.retrieved_chunks[:5]
    ]

    return MessageResponse(
        type=decision.kind,
        text=decision.text,
        citations=decision.citations,
        searched=searched,
        iterations=decision.iterations,
    )


@router.get("/{session_id}/messages", tags=["sessions"])
async def get_messages(session_id: int) -> dict[str, Any]:
    """Get conversation history for a session."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, role, content, kind, created_at FROM messages "
                "WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cur.fetchall()

    if not rows:
        # Check if session exists.
        with conn.cursor() as cur2:
            cur2.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
            if not cur2.fetchone():
                raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "kind": row[3],
            "created_at": row[4].isoformat() if row[4] else "",
        }
        for row in rows
    ]
    return {"session_id": session_id, "messages": messages}
