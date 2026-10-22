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
from verity.agent.loop import (
    AgentLoop,
    MAX_ITERATIONS,
    RESTYLE_NOTE,
    RESTYLE_RETRY_NOTE,
    make_retriever_from_index,
)


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


# --- Follow-ups and refusals ---------------------------------------------
#
# A follow-up is where conversational RAG usually breaks: the message carries
# the reference, not the topic, so searching it literally retrieves nothing
# useful (spec §9). These tests pin the two rules that make it work: rewrite
# before retrieval, and never turn a refusal into a list of citations.

HISTORY = [
    {"role": "user", "content": "What is the maximum junction temperature?"},
    {"role": "assistant", "content": "94 C."},
]


class RecordingRetriever:
    """A retriever that remembers every query it was searched with."""

    components = ("keyword",)

    def __init__(self, texts: list[str]) -> None:
        self._results = [retrieved(text, rank=i + 1) for i, text in enumerate(texts)]
        self.queries: list[str] = []

    def search(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self._results


class FixedRewriter:
    """A stand-in for the model's follow-up rewrite, recording its calls."""

    def __init__(self, rewritten: str | None = None, error: Exception | None = None) -> None:
        self._rewritten = rewritten
        self._error = error
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def rewrite_follow_up(
        self, *, question: str, history: list[dict[str, str]]
    ) -> str:
        self.calls.append((question, list(history)))
        if self._error is not None:
            raise self._error
        return self._rewritten if self._rewritten is not None else question


def test_a_follow_up_is_rewritten_before_retrieval() -> None:
    retriever = RecordingRetriever(["maximum junction temperature is 94 C"])
    rewriter = FixedRewriter("maximum junction temperature in the datasheet")
    provider = FakeProvider(
        actions=[LLMAction("answer", text="It is 94 C [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider, rewriter=rewriter)

    decision = loop.run("explain in simple manner", session_history=HISTORY)

    assert decision.kind == "answer"
    assert retriever.queries == ["maximum junction temperature in the datasheet"]
    assert rewriter.calls and rewriter.calls[0][0] == "explain in simple manner"
    # The model is shown what the user actually typed, not the search query.
    assert provider.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "explain in simple manner",
    }
    assert decision.trace["steps"][0]["original_query"] == "explain in simple manner"


def test_a_first_question_is_never_rewritten() -> None:
    retriever = RecordingRetriever(["value 94"])
    rewriter = FixedRewriter("something else entirely")
    provider = FakeProvider(
        actions=[LLMAction("answer", text="94 [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider, rewriter=rewriter)

    loop.run("What is the maximum junction temperature?")

    assert rewriter.calls == []
    assert retriever.queries == ["What is the maximum junction temperature?"]


def test_a_standalone_question_is_not_rewritten() -> None:
    """Long and self-contained: the extra model call would be waste."""
    retriever = RecordingRetriever(["value 94"])
    rewriter = FixedRewriter("maximum junction temperature in the datasheet")
    provider = FakeProvider(
        actions=[LLMAction("answer", text="94 [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider, rewriter=rewriter)

    loop.run(
        "What is the maximum junction temperature of the reactor core?",
        session_history=HISTORY,
    )

    assert rewriter.calls == []
    assert retriever.queries == [
        "What is the maximum junction temperature of the reactor core?"
    ]


def test_a_failed_rewrite_falls_back_to_the_literal_question() -> None:
    """A rewrite is an improvement; losing it must not lose the question."""
    from verity.llm.nvidia_provider import NvidiaLLMError

    retriever = RecordingRetriever(["value 94"])
    rewriter = FixedRewriter(error=NvidiaLLMError("endpoint down"))
    provider = FakeProvider(
        actions=[LLMAction("answer", text="94 [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider, rewriter=rewriter)

    decision = loop.run("explain in simple manner", session_history=HISTORY)

    assert decision.kind == "answer"
    assert retriever.queries == ["explain in simple manner"]
    assert decision.trace["steps"][0]["rewrite_failed"] is True


def test_a_refusal_cites_nothing_even_when_chunks_were_retrieved() -> None:
    """Retrieved is not cited.

    The refusal branch used to return every retrieved chunk as a citation, so
    the API answered ``type: refusal`` with ten "sources" and the page showed
    them as evidence for a claim it had just said it could not make.
    """
    retriever = RecordingRetriever(["unrelated passage"])
    provider = FakeProvider(
        actions=[LLMAction("refusal", reason="the extracts do not say")]
    )
    loop = AgentLoop(retriever, provider)

    decision = loop.run("What is the value?")

    assert decision.kind == "refusal"
    assert decision.citations == []
    assert decision.retrieved_chunks  # still reported as what was searched


def test_an_unchanged_rewrite_is_reported_as_attempted() -> None:
    """The trace says a rewrite ran, so "searched as written" is explained."""
    retriever = RecordingRetriever(["value 94"])
    rewriter = FixedRewriter()  # returns the question unchanged
    provider = FakeProvider(
        actions=[LLMAction("answer", text="94 [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider, rewriter=rewriter)

    decision = loop.run("explain in simple manner", session_history=HISTORY)

    step = decision.trace["steps"][0]
    assert step["rewrite_attempted"] is True
    assert "original_query" not in step
    assert retriever.queries == ["explain in simple manner"]


def test_a_restyle_follow_up_tells_the_model_to_rephrase() -> None:
    """Without the note the model re-extracts the same sentence.

    "Explain in simple manner" retrieves the same chunk as the previous turn;
    left alone, the model reads the same sentence out again. The note is what
    makes the same evidence produce a different wording.
    """
    retriever = RecordingRetriever(["maximum junction temperature is 94 C"])
    provider = FakeProvider(
        actions=[LLMAction("answer", text="It tops out at 94 C [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider)

    decision = loop.run("explain in simple manner", session_history=HISTORY)

    assert decision.kind == "answer"
    messages = provider.calls[0]["messages"]
    assert {"role": "system", "content": RESTYLE_NOTE} in messages
    # The user's own words are still the last thing the model reads.
    assert messages[-1] == {"role": "user", "content": "explain in simple manner"}


def test_a_not_understood_follow_up_gets_the_explanation_note() -> None:
    """The note is about explaining the reference passage, not rewording."""
    retriever = RecordingRetriever(["maximum junction temperature is 94 C"])
    provider = FakeProvider(
        actions=[LLMAction("answer", text="It means 94 C [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider)

    loop.run("i am not understanding", session_history=HISTORY)

    messages = provider.calls[0]["messages"]
    assert {"role": "system", "content": RESTYLE_NOTE} in messages


def test_a_restyle_is_never_answered_with_a_clarification() -> None:
    """Invariant I3 on the one follow-up whose intent is never ambiguous.

    The model read "i am not understanding" as a question it could ask about
    ("which part?") and returned a clarification. The extracts were already
    shown and the previous answer is the reference, so being asked what is
    unclear is avoidance, not a clarification.
    """
    retriever = RecordingRetriever(["maximum junction temperature is 94 C"])
    provider = FakeProvider(
        actions=[
            LLMAction("clarification", question="Which part is unclear?"),
            LLMAction("answer", text="It means 94 C [1].", chunk_ids=[1]),
        ]
    )
    loop = AgentLoop(retriever, provider)

    decision = loop.run("i am not understanding", session_history=HISTORY)

    assert decision.kind == "answer"
    assert len(provider.calls) == 2
    assert {"role": "system", "content": RESTYLE_RETRY_NOTE} in provider.calls[1][
        "messages"
    ]


def test_an_ordinary_follow_up_may_still_clarify() -> None:
    retriever = RecordingRetriever(["value 94"])
    provider = FakeProvider(
        actions=[LLMAction("clarification", question="Do you mean the other one?")]
    )
    loop = AgentLoop(retriever, provider)

    decision = loop.run(
        "Could you tell me whether that limit applies when the board is upside down?",
        session_history=HISTORY,
    )

    assert decision.kind == "clarification"
    assert len(provider.calls) == 1


def test_an_ordinary_follow_up_gets_no_restyle_note() -> None:
    retriever = RecordingRetriever(["maximum junction temperature is 94 C"])
    provider = FakeProvider(
        actions=[LLMAction("answer", text="94 C [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider)

    loop.run("and what about it in celsius?", session_history=HISTORY)

    messages = provider.calls[0]["messages"]
    assert {"role": "system", "content": RESTYLE_NOTE} not in messages


def test_a_restyle_with_no_history_gets_no_note() -> None:
    """There is no previous answer to restate, so the note would be false."""
    retriever = RecordingRetriever(["94 C"])
    provider = FakeProvider(
        actions=[LLMAction("answer", text="94 C [1].", chunk_ids=[1])]
    )
    loop = AgentLoop(retriever, provider)

    loop.run("explain in simple manner")

    messages = provider.calls[0]["messages"]
    assert {"role": "system", "content": RESTYLE_NOTE} not in messages
