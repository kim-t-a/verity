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

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Literal, Sequence

from verity.engine.embedding_provider import EmbeddingProvider
from verity.engine.models import Chunk
from verity.engine.retrieval.hybrid import HybridRetriever
from verity.engine.retrieval.keyword import KeywordRetriever
from verity.engine.retrieval.vector import VectorRetriever
from verity.engine.retrieval import RetrievedChunk
from verity.llm.provider import LLMAction, LLMProvider
from verity.agent.rewrite import FollowUpRewriter, looks_like_follow_up


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

    trace: dict[str, object] = field(default_factory=dict)
    """What happened, in order, for the trace panel.

    Real measurements only: the retriever components that ran, each step's
    query and result count, the timings, and the fused ranks and scores of the
    chunks that were shown to the model. Nothing here is inferred after the
    fact -- a panel that guessed would be a second, disagreeing account of the
    same turn.
    """


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
        rewriter: FollowUpRewriter | None = None,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._max_iterations = max_iterations
        #: Optional. Without it, every message is searched as written, which
        #: is correct for one-shot questions and useless for follow-ups.
        self._rewriter = rewriter

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
        started = perf_counter()
        steps: list[dict[str, object]] = []

        # Build the conversation messages for the model.
        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.append({"role": "user", "content": question})

        # A follow-up is rewritten into a standalone query before anything is
        # searched. This changes the query, not what the model is told the
        # user said: the model still sees the real message, so it cannot
        # answer a question the user did not ask.
        search_query = question
        original_query: str | None = None
        rewrite_failed = False
        rewrite_attempted = False
        if self._rewriter is not None and looks_like_follow_up(question, history):
            rewrite_attempted = True
            try:
                rewritten = self._rewriter.rewrite_follow_up(
                    question=question, history=history
                ).strip()
            except Exception:  # noqa: BLE001 - a rewrite is an improvement, not a step
                # Falling back to the literal question is degraded, not broken,
                # and the trace says which happened.
                rewrite_failed = True
            else:
                if rewritten and rewritten != question:
                    original_query = question
                    search_query = rewritten

        for iteration in range(self._max_iterations):
            iterations = iteration + 1

            # Retrieve chunks for the query this iteration is working on.
            retrieve_started = perf_counter()
            chunks = self._retrieve(search_query)
            retrieve_ms = _elapsed_ms(retrieve_started)
            retrieved_chunks.extend(chunks)

            # Ask the model to decide.
            decide_started = perf_counter()
            action = self._provider.decide(
                messages=messages,
                chunks=[c.chunk for c in chunks],
                tools=["search"],
            )

            step: dict[str, object] = {
                "index": iterations,
                "query": search_query,
                "action": action.type,
                "results": len(chunks),
                "retrieve_ms": retrieve_ms,
                "decide_ms": _elapsed_ms(decide_started),
                "top": [_trace_chunk(c) for c in chunks[:10]],
            }
            if iteration == 0 and original_query is not None:
                step["original_query"] = original_query
            if iteration == 0 and rewrite_attempted:
                step["rewrite_attempted"] = True
            if iteration == 0 and rewrite_failed:
                step["rewrite_failed"] = True
            steps.append(step)

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
                    search_query = action.query
                continue

            if action_type == "clarification":
                return AgentDecision(
                    kind="clarification",
                    text=action.question or "Could you clarify your question?",
                    citations=[],
                    retrieved_chunks=retrieved_chunks,
                    iterations=iterations,
                    trace=self._trace(steps, started),
                )

            if action_type == "refusal":
                return AgentDecision(
                    kind="refusal",
                    text=action.reason or "I couldn't find enough evidence to answer.",
                    # Retrieved is not cited. These chunks were searched and
                    # rejected; listing them as citations made the page show
                    # "10 sources" for an answer that says there are none.
                    # What was searched travels in ``retrieved_chunks``, which
                    # the API reports separately as ``searched``.
                    citations=[],
                    retrieved_chunks=retrieved_chunks,
                    iterations=iterations,
                    trace=self._trace(steps, started),
                )

            if action_type == "answer":
                step["cited"] = list(action.chunk_ids)
                decision = self._verify_and_build_answer(
                    action, chunks, retrieved_chunks, iterations
                )
                return replace(decision, trace=self._trace(steps, started))

            # Unknown action type — treat as refusal.
            return AgentDecision(
                kind="refusal",
                text="The model returned an unrecognized action.",
                citations=[],
                retrieved_chunks=retrieved_chunks,
                iterations=iterations,
                trace=self._trace(steps, started),
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
                        "ordinal": i,
                        "rel_path": c.chunk.rel_path,
                        "location": c.chunk.location,
                        "text": c.chunk.text,
                    }
                    # Same five chunks the best-effort text quotes, so every
                    # [n] in the answer has exactly one evidence row.
                    for i, c in enumerate(retrieved_chunks[:5], start=1)
                ],
                retrieved_chunks=retrieved_chunks,
                iterations=iterations,
                trace=self._trace(steps, started),
            )

        return AgentDecision(
            kind="refusal",
            text="I searched but found no relevant documents.",
            citations=[],
            retrieved_chunks=retrieved_chunks,
            iterations=iterations,
            trace=self._trace(steps, started),
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
                        # The model cites by position, so the display numeral
                        # must be that same position: the reader finds the
                        # matching [n] in the answer text. Renumbering by
                        # citation order would point at the wrong chunk.
                        "chunk_id": position,
                        "ordinal": position,
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

    def _trace(
        self, steps: list[dict[str, object]], started: float
    ) -> dict[str, object]:
        """The turn's trace, from what actually ran.

        ``degraded`` is reported separately from the score because a
        keyword-only run is not merely worse: a fused RRF score means
        ``1/(60+rank)`` in this configuration, which looks like a real score
        and is not comparable to anything a reader has seen before. Saying the
        vector half is missing is the only honest way to present it.
        """
        components = getattr(self._retriever, "components", None)
        if components is None:
            components = (_retriever_name(self._retriever),)
        retrievers = list(components)
        return {
            "retrievers": retrievers,
            "degraded": "vector" not in retrievers,
            "duration_ms": _elapsed_ms(started),
            "steps": steps,
        }


def _retriever_name(retriever: object) -> str:
    """A short name for a retriever that does not expose ``components``."""
    if isinstance(retriever, VectorRetriever):
        return "vector"
    if isinstance(retriever, KeywordRetriever):
        return "keyword"
    return type(retriever).__name__


def _elapsed_ms(started: float) -> float:
    """Milliseconds since ``started``, rounded to a tenth.

    Rounded because the panel prints them and a raw float suggests a precision
    the machine's clock does not have.
    """
    return round((perf_counter() - started) * 1000, 1)


def _trace_chunk(result: RetrievedChunk) -> dict[str, object]:
    """One retrieved chunk, as the trace panel needs it.

    The preview is truncated: the panel shows what was ranked and why, not a
    second copy of the document. Full text is available through the citation
    for the chunks the model actually used.
    """
    return {
        "rel_path": result.rel_path,
        "location": result.location,
        "score": round(result.score, 6),
        "rank": result.rank,
        "sources": [list(source) for source in result.sources],
        "matched_terms": list(result.matched_terms),
        "preview": result.text[:240],
    }


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
    from verity.engine.index_types import ChunkIndex, IndexManifest

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
