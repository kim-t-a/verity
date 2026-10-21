"""LLM provider interface for Verity.

Defines the :class:`LLMProvider` Protocol that the agent loop is written
against, plus a hand-written :class:`FakeProvider` for tests. The real
NVIDIA-backed provider can be swapped in later without touching the agent
loop or retrieval code.

The provider never receives raw document text — only chunks that were
actually retrieved and shown to it. This enforces invariant I2: "The model
never sees text it cannot cite."
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verity.engine.models import Chunk


class LLMAction:
    """A single action the model decides to take.

    ``type`` is one of:
      - ``"search"``: retrieve more chunks for a rewritten query
      - ``"answer"``: produce the final answer with cited chunk IDs
      - ``"clarification"``: ask the user a clarifying question
      - ``"refusal"``: refuse because evidence is insufficient

    ``chunk_ids`` (for ``answer``) must reference chunks that were actually
    shown to the model in the current turn — this is verified downstream.
    """

    def __init__(
        self,
        type: str,
        *,
        query: str | None = None,
        chunk_ids: list[int] | None = None,
        text: str | None = None,
        question: str | None = None,
        options: list[str] | None = None,
        reason: str | None = None,
    ) -> None:
        self.type = type
        self.query = query
        self.chunk_ids = chunk_ids or []
        self.text = text
        self.question = question
        self.options = options or []
        self.reason = reason

    def __repr__(self) -> str:
        return f"LLMAction(type={self.type!r}, ...)"  # pragma: no cover


@runtime_checkable
class LLMProvider(Protocol):
    """Turns a conversation + retrieved chunks into a structured action.

    A protocol rather than a base class so any backend can satisfy it — the
    NVIDIA-backed provider at runtime, a deterministic fake in tests.

    ``model_id`` and ``dimension`` are not relevant here (that's the
    EmbeddingProvider's job); what matters is the ``decide`` method's shape.
    """

    @property
    def model_id(self) -> str:
        """Identifier for the model, recorded in message metadata."""
        ...

    def decide(
        self,
        *,
        messages: list[dict[str, str]],
        chunks: list[Chunk],
        tools: list[str],
    ) -> LLMAction:
        """Given the conversation and retrieved chunks, return one action.

        ``chunks`` are the chunks actually retrieved this turn — the model
        may only cite chunk IDs that appear in this list. The provider must
        never invent citations for chunks it wasn't shown.

        ``tools`` is the list of available tool names (e.g. ``["search"]``).

        Returns an :class:`LLMAction`. The agent loop validates that any
        cited chunk IDs are actually in ``chunks``.
        """
        ...


class FakeProvider:
    """Deterministic fake LLM provider for tests.

    Configured with a list of :class:`LLMAction` objects it returns in order.
    Useful for testing every agent-loop path: answer, clarification, refusal,
    search, and misbehavior (citing chunks that weren't shown).
    """

    def __init__(self, actions: list[LLMAction] | None = None) -> None:
        self._actions = list(actions or [])
        self._index = 0
        self.calls: list[dict[str, object]] = []

    @property
    def model_id(self) -> str:
        return "fake-provider"

    def decide(
        self,
        *,
        messages: list[dict[str, str]],
        chunks: list[Chunk],
        tools: list[str],
    ) -> LLMAction:
        """Return the next pre-configured action and record the call."""
        self.calls.append(
            {
                "messages": list(messages),
                "chunks": [chunk.rel_path for chunk in chunks],
                "tools": list(tools),
            }
        )
        if self._index < len(self._actions):
            action = self._actions[self._index]
            self._index += 1
            return action
        # Default: refuse when no more actions configured.
        return LLMAction(
            "refusal",
            reason="no more actions configured in FakeProvider",
        )


class FakeEmbeddingProvider:
    """Deterministic fake embedding provider for tests.

    Returns the same vector for every input. Satisfies the
    :class:`~verity.engine.embedding_provider.EmbeddingProvider` Protocol.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return "fake-embedding-provider"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]
