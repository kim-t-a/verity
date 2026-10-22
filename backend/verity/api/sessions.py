"""Session and message API routes.

Endpoints for starting conversations and asking questions. The message
endpoint returns a discriminated union: ``{"type": "answer"}`` /
``{"type": "clarification"}`` / ``{"type": "refusal"}`` — the frontend
switches on the ``type`` field.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from verity.db import get_connection
from verity.db.chunk_repository import load_all_chunks
from verity.agent.loop import AgentDecision, AgentLoop
from verity.llm.nvidia_provider import NvidiaLLMError
from verity.provider_factory import default_embedding_provider, default_llm_provider
from verity.engine.retrieval.keyword import KeywordRetriever
from verity.engine.retrieval.vector import VectorRetriever, VectorUnavailable
from verity.engine.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions")


def _build_retriever():
    """The retriever for this turn, fused from what is already in Postgres.

    The stored vectors are used in place, not re-embedded per question: the
    index knows which backend built them, and VectorRetriever refuses to run
    unless the configured backend matches -- a mismatch would mean cosine
    scores between unrelated coordinate systems, which produce confident
    garbage rather than errors.

    A hybrid with only the keyword half is degraded mode, not failure: the
    fused order is then exactly the keyword order, visible in
    ``HybridRetriever.components``.
    """
    embedding = default_embedding_provider()
    index = load_all_chunks(
        provider_id=embedding.model_id,
        dimension=embedding.dimension,
    )

    components = [("keyword", KeywordRetriever(index))]
    try:
        components.append(("vector", VectorRetriever(index, embedding)))
    except VectorUnavailable:
        # Honest degraded mode: keyword-only. VectorUnavailable names the
        # reason (no vectors, or a provider mismatch) in its message.
        pass
    return HybridRetriever(components)


def _run_loop(
    loop: AgentLoop,
    question: str,
    history: list[dict[str, str]],
    session_id: int,
) -> AgentDecision:
    """Run the loop, turning a model failure into an honest 502.

    The loop already turns an *evidence* problem into a refusal, and that is a
    product answer. A model outage is not evidence: reporting it as a refusal
    would claim the documents do not contain something we never got to read.
    It is an upstream failure, so it says so.
    """
    try:
        return loop.run(question, session_history=history)
    except NvidiaLLMError as exc:
        logger.exception("Language model failed for session %s", session_id)
        raise HTTPException(
            status_code=502,
            detail=f"The language model could not answer: {exc}",
        ) from exc


class StartSession(BaseModel):
    workspace_id: int = 1


class SessionResponse(BaseModel):
    session_id: int


class SessionSummary(BaseModel):
    session_id: int
    title: str
    created_at: str
    last_message_at: str | None
    message_count: int


#: How much of the first question becomes a chat title. Long enough to tell
#: two questions apart, short enough for one line in a narrow list.
TITLE_MAX_CHARS = 80


class AskRequest(BaseModel):
    question: str


class MessageResponse(BaseModel):
    type: str  # "answer" | "clarification" | "refusal"
    text: str
    citations: list[dict[str, Any]] = []
    searched: list[dict[str, str]] = []
    iterations: int = 0
    trace: dict[str, Any] = Field(default_factory=dict)


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


@router.get("", response_model=dict, tags=["sessions"])
async def list_sessions(
    workspace_id: int = 1, limit: int = 50
) -> dict[str, list[SessionSummary]]:
    """The workspace's conversations, most recently used first.

    A chat is a session that actually has messages: sessions are created
    lazily on the first question, and an empty one has nothing to reopen.

    The title is the first question the person asked, not a summary: a
    generated title would be a second model call and a second chance to say
    something the conversation does not. Activity is the newest message, so
    resuming an old chat moves it to the top.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id,
                       s.created_at,
                       COUNT(m.id) AS message_count,
                       MAX(m.created_at) AS last_message_at,
                       (SELECT content FROM messages
                         WHERE session_id = s.id AND role = 'user'
                         ORDER BY id LIMIT 1) AS first_question
                FROM sessions s
                JOIN messages m ON m.session_id = s.id
                WHERE s.workspace_id = %s
                GROUP BY s.id
                ORDER BY MAX(m.created_at) DESC, s.id DESC
                LIMIT %s
                """,
                (workspace_id, limit),
            )
            rows = cur.fetchall()

    return {
        "sessions": [
            SessionSummary(
                session_id=row[0],
                title=_chat_title(row[4]),
                created_at=row[1].isoformat(),
                last_message_at=row[3].isoformat() if row[3] else None,
                message_count=row[2],
            )
            for row in rows
        ]
    }


def _chat_title(question: str | None) -> str:
    """One line for the chat list, cut at a word boundary."""
    text = " ".join((question or "").split())
    if not text:
        return "Untitled chat"
    if len(text) <= TITLE_MAX_CHARS:
        return text
    return text[:TITLE_MAX_CHARS].rsplit(" ", 1)[0].rstrip() + "…"


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

    retriever = _build_retriever()
    provider = default_llm_provider()
    # The same model decides answers and rewrites follow-ups. The loop calls
    # the rewriter only when a message looks like a follow-up; a first
    # question skips the extra call entirely (spec §9).
    loop = AgentLoop(retriever, provider, rewriter=provider)

    # Get conversation history for context. The newest ten turns, oldest
    # first: ordering by created_at alone is not enough because the user and
    # assistant rows of one exchange are written in the same transaction and
    # therefore share a timestamp; the id breaks the tie in insertion order.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM ("
                "  SELECT id, role, content FROM messages "
                "  WHERE session_id = %s ORDER BY id DESC LIMIT 10"
                ") recent ORDER BY id ASC",
                (session_id,),
            )
            history = [
                {"role": row[0], "content": row[1]}
                for row in cur.fetchall()
            ]

    decision = _run_loop(loop, request.question, history, session_id)

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
            # chunks has no rel_path column; the document filename lives on
            # documents, joined here. document_id comes from the same row,
            # not from an unrelated subselect.
            if decision.kind == "answer":
                for citation in decision.citations:
                    cur.execute(
                        """
                        SELECT c.id, d.id
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.filename = %s AND c.location = %s
                        ORDER BY c.ordinal
                        LIMIT 1
                        """,
                        (citation["rel_path"], citation["location"]),
                    )
                    pair = cur.fetchone()
                    if pair:
                        cur.execute(
                            "INSERT INTO citations (message_id, chunk_id, document_id) "
                            "VALUES (%s, %s, %s)",
                            (assistant_message_id, pair[0], pair[1]),
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
        trace=decision.trace,
    )


@router.delete("/{session_id}", status_code=204, tags=["sessions"])
async def delete_session(session_id: int) -> Response:
    """Delete one conversation and everything said in it.

    Messages and their citations go with it through ``ON DELETE CASCADE``.
    Documents and chunks do not: a chat is a record of questions, not of the
    corpus, and deleting a conversation must not remove evidence another
    conversation still cites.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM sessions WHERE id = %s RETURNING id", (session_id,)
            )
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Session not found")
        conn.commit()
    return Response(status_code=204)


@router.get("/{session_id}/messages", tags=["sessions"])
async def get_messages(session_id: int) -> dict[str, Any]:
    """Get conversation history for a session.

    History carries each assistant message's stored citations, so reopening a
    conversation still shows what the answer was based on. It does not carry a
    trace: the trace describes one run of the loop (timings, candidate ranks)
    and is not persisted. The panel says so rather than reconstructing a
    plausible-looking one after the fact.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, role, content, kind, created_at FROM messages "
                "WHERE session_id = %s ORDER BY id ASC",
                (session_id,),
            )
            rows = cur.fetchall()

            if not rows:
                cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Session not found")

            citations_by_message: dict[int, list[dict[str, Any]]] = {}
            assistant_ids = [row[0] for row in rows if row[1] == "assistant"]
            if assistant_ids:
                cur.execute(
                    """
                    SELECT ci.message_id, ci.chunk_id, d.filename,
                           c.location, c.text
                    FROM citations ci
                    JOIN chunks c ON c.id = ci.chunk_id
                    JOIN documents d ON d.id = ci.document_id
                    WHERE ci.message_id = ANY(%s)
                    ORDER BY ci.message_id, d.filename, c.ordinal
                    """,
                    (assistant_ids,),
                )
                for message_id, chunk_id, filename, location, text in cur.fetchall():
                    citations_by_message.setdefault(message_id, []).append(
                        {
                            "chunk_id": chunk_id,
                            "rel_path": filename,
                            "location": location,
                            "text": text,
                        }
                    )

    messages = [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "kind": row[3],
            "created_at": row[4].isoformat() if row[4] else "",
            "citations": citations_by_message.get(row[0], []),
        }
        for row in rows
    ]
    return {"session_id": session_id, "messages": messages}
