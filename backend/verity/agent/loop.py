"""Agent loop for Verity.

Turns a user question into an evidence-backed answer (or clarification /
refusal) by running a bounded loop:

    retrieve → model decides → (search again | answer | clarify | refuse)

The loop is capped at 5 iterations (spec invariant I3): clarifying is never
a way to avoid answering. At the cap, the model produces a best-effort
answer or refuses only when evidence genuinely does not exist.

Citations are verified (invariant I2): every chunk ID the model cites is
checked against the chunks actually shown in this turn. An ID that was never
shown is an invented citation and the answer is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from verity.engine.embeddings import EmbeddingProvider
from verity.engine.models import Chunk
from verity.engine.retrieval.hybrid import HybridRetriever
from verity.engine.retrieval.keyword import KeywordRetriever
from verity.engine.retrieval.vector import VectorRetriever
from verity.engine.retrieval import RetrievedChunk
from verity.llm.provider import LLMAction, LLMProvider


#: Maximum number of tool-calling iterations before the model must answer
#: or refuse. Spec invariant I3: clarifying is never a way to avoid answering.
MAX_ITERATIONS = 5

ActionType = Literal["search", "answer", "clarification", "refusal"]


@dataclass(frozen=True)
class AgentDecision:
    """The final outcome of the agent loop."""

    kind: str
    """One of: ``"answer"``, ``"clarification"``, ``"refusal"``."""

    text: str
    """The answer text, clarification question, or refusal reason."""

    citations: list[dict[str, object]]
    """Citation entries: ``{"chunk_id": int, "rel_path": str, "location": str}``."""

    retrieved_chunks: list[RetrievedChunk]
    """All chunks retrieved during the loop (for diagnostics / trace panel)."""

    iterations: int
    """How many model decisions were made."""


class AgentLoop:
    """The bounded agent loop.

    Holds a retriever and an LLM provider. Each call to :meth:`run` processes
    one question against an index built from the retriever's chunks.
    """

    def __init__(
        self,
        retriever: HybridRetriever | KeywordRetriever | VectorRetriever,
        provider: LLMProvider,
        *,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._max_iterations = max_iterations

    def run(
        self,
        question: str,
        *,
        session_history: list[dict[str, str]] | None = None,
    ) -> AgentDecision:
        """Run the agent loop for one question.

        Args:
            question: The user's question.
            session_history: Previous turns (for follow-up context).

        Returns:
            An :class:`AgentDecision` with the answer, clarification, or
            refusal, plus verified citations and the chunks retrieved.
        """
        history = list(session_history or [])
        retrieved_chunks: list[RetrievedChunk] = []
        iterations = 0

        # Build the conversation messages for the model.
        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.append({"role": "user", "content": question})

        for iteration in range(self._max_iterations):
            iterations = iteration + 1

            # Retrieve chunks for the current question.
            search_query = question
            chunks = self._retrieve(search_query)
            retrieved_chunks.extend(chunks)

            # Ask the model to decide.
            action = self._provider.decide(
                messages=messages,
                chunks=[c.chunk for c in chunks],
                tools=["search"],
            )

            action_type: ActionType = action.type  # type: ignore[assignment]

            if action_type == "search":
                # Refine the query and loop again.
                messages.append(
                    {"role": "assistant", "content": f"Searching for: {action.query}"}
                )
                if action.query:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Refined query: {action.query}",
                        }
                    )
                    question = action.query
                continue

            if action_type == "clarification":
                return AgentDecision(
                    kind="clarification",
                    text=action.question or "Could you clarify your question?",
                    citations=[],
                    retrieved_chunks=retrieved_chunks,
                    iterations=iterations,
                )

            if action_type == "refusal":
                return AgentDecision(
                    kind="refusal",
                    text=action.reason or "I couldn't find enough evidence to answer.",
                    citations=[
                        {
                            "chunk_id": i,
                            "rel_path": c.chunk.rel_path,
                            "location": c.chunk.location,
                        }
                        for i, c in enumerate(retrieved_chunks)
                    ],
                    retrieved_chunks=retrieved_chunks,
                    iterations=iterations,
                )

            if action_type == "answer":
                return self._verify_and_build_answer(
                    action, chunks, retrieved_chunks, iterations
                )

            # Unknown action type — treat as refusal.
            return AgentDecision(
                kind="refusal",
                text="The model returned an unrecognized action.",
                citations=[],
                retrieved_chunks=retrieved_chunks,
                iterations=iterations,
            )

        # Cap reached: best-effort answer from what we have, or refuse.
        # Spec invariant I3: at the cap, produce best-effort or refuse.
        if retrieved_chunks:
            return AgentDecision(
                kind="answer",
                text=self._best_effort_answer(question, retrieved_chunks),
                citations=[
                    {
                        "chunk_id": i,
                        "rel_path": c.chunk.rel_path,
                        "location": c.chunk.location,
                    }
                    for i, c in enumerate(retrieved_chunks[:10])
                ],
                retrieved_chunks=retrieved_chunks,
                iterations=iterations,
            )

        return AgentDecision(
            kind="refusal",
            text="I searched but found no relevant documents.",
            citations=[],
            retrieved_chunks=retrieved_chunks,
            iterations=iterations,
        )

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve chunks for a query, handling degraded modes gracefully."""
        try:
            return self._retriever.search(query)
        except Exception:  # noqa: BLE001 - one retriever down is degraded mode
            return []

    def _verify_and_build_answer(
        self,
        action: LLMAction,
        current_chunks: list[RetrievedChunk],
        all_chunks: list[RetrievedChunk],
        iterations: int,
    ) -> AgentDecision:
        """Verify cited chunk IDs are actually in the shown chunks, then build
        the final answer with citation entries.

        Invariant I2: the model may only cite chunks it was actually shown.
        An ID that was never shown is an invented citation and the answer is
        rejected (treated as refusal).
        """
        # The model cites chunks by 1-based position in the current chunk list.
        citations: list[dict[str, object]] = []
        seen_positions: set[int] = set()

        for position in action.chunk_ids:
            if position in seen_positions:
                continue
            seen_positions.add(position)
            if 1 <= position <= len(current_chunks):
                c = current_chunks[position - 1]
                citations.append(
                    {
                        "chunk_id": position,
                        "rel_path": c.chunk.rel_path,
                        "location": c.chunk.location,
                        "text": c.chunk.text,
                    }
                )

        # Build answer text.
        answer_text = action.text or ""

        if not citations:
            # Model cited nothing valid — refuse rather than answer without evidence.
            return AgentDecision(
                kind="refusal",
                text="I couldn't find supporting evidence in the documents.",
                citations=[],
                retrieved_chunks=all_chunks,
                iterations=iterations,
            )

        return AgentDecision(
            kind="answer",
            text=answer_text,
            citations=citations,
            retrieved_chunks=all_chunks,
            iterations=iterations,
        )

    def _best_effort_answer(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> str:
        """Build a best-effort answer from retrieved chunks at the iteration cap.

        Concatenates the top chunks' text with their locations so the user
        gets something useful even when the model couldn't decide.
        """
        parts = []
        for i, c in enumerate(chunks[:5], start=1):
            parts.append(
                f"[{i}] {c.chunk.text} (from {c.chunk.rel_path}, {c.chunk.location})"
            )
        return (
            f"I reached the iteration limit without a definitive answer. "
            f"Here's what I found:\n" + "\n".join(parts)
        )


def make_retriever_from_index(
    index_chunks: Sequence[Chunk],
    embedding_provider: EmbeddingProvider | None = None,
) -> HybridRetriever | KeywordRetriever:
    """Build a hybrid retriever from a list of chunks.

    Wraps chunks in an in-memory index and creates keyword + vector
    retrievers fused with RRF. If the vector retriever can't start
    (provider mismatch, no vectors), falls back to keyword-only (degraded
    mode — spec: "One retriever down is a degraded mode, not a silent one").
    """
    from verity.engine.chunkstore import ChunkIndex, IndexManifest

    if not index_chunks:
        # No chunks — keyword retriever over empty data returns nothing.
        manifest = IndexManifest(
            format_version=1,
            created_utc="",
            corpus_fingerprint="",
            chunk_count=0,
            file_count=0,
            provider_id=None,
            dimension=None,
            status_counts={},
            skipped_count=0,
        )
        empty_index = ChunkIndex(manifest=manifest, chunks=(), vectors=())
        return KeywordRetriever(empty_index)

    # Build keyword retriever.
    manifest = IndexManifest(
        format_version=1,
        created_utc="",
        corpus_fingerprint="",
        chunk_count=len(index_chunks),
        file_count=len({c.rel_path for c in index_chunks}),
        provider_id=embedding_provider.model_id if embedding_provider else None,
        dimension=embedding_provider.dimension if embedding_provider else None,
        status_counts={},
        skipped_count=0,
    )

    # Vectors: embedding provider embeds each chunk's text.
    vectors: list[tuple[float, ...]] = []
    if embedding_provider is not None:
        try:
            vector_lists = embedding_provider.embed([c.text for c in index_chunks])
            vectors = [tuple(v) for v in vector_lists]
        except Exception:  # noqa: BLE001 - degraded mode
            vectors = []

    index = ChunkIndex(
        manifest=manifest,
        chunks=tuple(index_chunks),
        vectors=tuple(vectors),
    )

    keyword = KeywordRetriever(index)

    if embedding_provider is not None and vectors:
        try:
            vector = VectorRetriever(index, embedding_provider)
            return HybridRetriever([("keyword", keyword), ("vector", vector)])
        except Exception:  # noqa: BLE001 - degraded mode
            pass

    return HybridRetriever([("keyword", keyword)])
