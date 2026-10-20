"""Tests for the agent loop.

Uses a hand-written :class:`~verity.llm.provider.FakeProvider` — no NVIDIA
credentials or network access needed. Tests every agent-loop path: answer,
clarification, refusal, search-then-answer, citation verification (rejects
invented citations), and the 5-iteration cap.
"""

from __future__ import annotations

import pytest

from verity.engine.models import Chunk, MediaType
from verity.engine.retrieval import RetrievedChunk
from verity.llm.provider import FakeProvider, LLMAction
from verity.agent.loop import AgentLoop, MAX_ITERATIONS, make_retriever_from_index


def chunk(text: str, rel_path: str = "specs/datasheet.pdf") -> Chunk:
    return Chunk(
        rel_path=rel_path,
        media_type=MediaType.PDF,
        location="page 1 line 1",
        ordinal=0,
        text=text,
    )


def retrieved(text: str, rel_path: str = "specs/datasheet.pdf", rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk(text, rel_path),
        score=0.9,
        rank=rank,
    )


def test_agent_answers_from_retrieved_chunks() -> None:
    """The model cites chunks it was shown — answer is returned with citations."""
    chunks = [retrieved("value 94", rank=1), retrieved("value 95", rank=2)]
    provider = FakeProvider(
        actions=[
            LLMAction(
                "answer",
                text="The value is 94.",
                chunk_ids=[1],
            )
        ]
    )
    retriever = make_retriever_from_index([c.chunk for c in chunks])
    loop = AgentLoop(retriever, provider)

    decision = loop.run("What is the value?")

    assert decision.kind == "answer"
    assert decision.text == "The value is 94."
    assert len(decision.citations) == 1
    assert decision.citations[0]["chunk_id"] == 1
    assert decision.citations[0]["rel_path"] == "specs/datasheet.pdf"
    assert decision.iterations == 1


def test_agent_refuses_when_no_evidence() -> None:
    """No chunks retrieved → refusal."""
    provider = FakeProvider(
        actions=[
            LLMAction("refusal", reason="no relevant documents found")
        ]
    )
    retriever = make_retriever_from_index([])
    loop = AgentLoop(retriever, provider)

    decision = loop.run("What is the value?")

    assert decision.kind == "refusal"
    assert decision.iterations == 1


def test_agent_clarifies_when_ambiguous() -> None:
    """The model asks a clarification question."""
    chunks = [retrieved("value 94", rank=1)]
    provider = FakeProvider(
        actions=[
            LLMAction(
                "clarification",
                question="Which document do you mean?",
            )
        ]
    )
    retriever = make_retriever_from_index([c.chunk for c in chunks])
    loop = AgentLoop(retriever, provider)

    decision = loop.run("What is the value?")

    assert decision.kind == "clarification"
    assert decision.text == "Which document do you mean?"
    assert decision.citations == []


def test_agent_rejects_invented_citations() -> None:
    """The model cites a chunk it was never shown → treated as refusal.

    Invariant I2: the model may only cite chunks it was actually shown.
    """
    chunks = [retrieved("value 94", rank=1)]
    # Model cites chunk_id 2, but only chunk 1 was shown.
    provider = FakeProvider(
        actions=[
            LLMAction(
                "answer",
                text="The value is 95.",
                chunk_ids=[2],  # invented — chunk 2 was never shown
            )
        ]
    )
    retriever = make_retriever_from_index([c.chunk for c in chunks])
    loop = AgentLoop(retriever, provider)

    decision = loop.run("What is the value?")

    assert decision.kind == "refusal"
    assert decision.citations == []


def test_agent_searches_then_answers() -> None:
    """The model searches once, then answers with the new results."""
    chunks1 = [retrieved("value 94 partial info", rank=1)]
    chunks2 = [retrieved("value 94 full answer", rank=1)]
    provider = FakeProvider(
        actions=[
            LLMAction("search", query="value 94 refined"),
            LLMAction("answer", text="The answer is 94.", chunk_ids=[1]),
        ]
    )
    retriever = make_retriever_from_index([c.chunk for c in chunks1 + chunks2])
    loop = AgentLoop(retriever, provider)

    decision = loop.run("value 94")

    assert decision.kind == "answer"
    assert decision.text == "The answer is 94."
    assert decision.iterations == 2


def test_agent_hits_iteration_cap() -> None:
    """At the 5-iteration cap, the model must answer or refuse."""
    chunks = [retrieved("value 94 some info", rank=1)]
    # Always search — never answer — to hit the cap.
    actions = [LLMAction("search", query=f"value 94 query {i}") for i in range(MAX_ITERATIONS)]
    provider = FakeProvider(actions=actions)
    retriever = make_retriever_from_index([c.chunk for c in chunks])
    loop = AgentLoop(retriever, provider)

    decision = loop.run("value 94")

    # At the cap, produces best-effort answer from retrieved chunks.
    assert decision.kind == "answer"
    assert decision.iterations == MAX_ITERATIONS
    assert "value 94" in decision.text


def test_agent_uses_session_history() -> None:
    """Follow-up questions use session history for context."""
    chunks = [retrieved("notice period 30 days", rank=1)]
    provider = FakeProvider(
        actions=[
            LLMAction("answer", text="30 days.", chunk_ids=[1])
        ]
    )
    retriever = make_retriever_from_index([c.chunk for c in chunks])
    loop = AgentLoop(retriever, provider)

    history = [
        {"role": "user", "content": "notice period"},
        {"role": "assistant", "content": "30 days, per section 4.2."},
    ]
    decision = loop.run("notice period 30 days", session_history=history)

    assert decision.kind == "answer"
    assert len(provider.calls) == 1
    # The model received the history.
    assert provider.calls[0]["messages"][0]["content"] == "notice period"


def test_empty_corpus_refuses() -> None:
    """No chunks at all → refusal."""
    provider = FakeProvider(
        actions=[LLMAction("refusal", reason="no documents in corpus")]
    )
    retriever = make_retriever_from_index([])
    loop = AgentLoop(retriever, provider)

    decision = loop.run("Anything?")

    assert decision.kind == "refusal"
    assert decision.iterations == 1


def test_max_iterations_is_five() -> None:
    """Spec invariant I3: the loop cap is 5."""
    assert MAX_ITERATIONS == 5
