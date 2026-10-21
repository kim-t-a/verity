"""The embedding interface, and nothing that implements it.

Kept deliberately separate from any implementation: retrieval is written
against this Protocol, so the provider can be swapped -- NVIDIA at runtime, a
deterministic fake in tests -- without touching retrieval. The challenge-era
local model and its CPU fallback lived here and are gone; the guard that made
a CPU path unselectable went with them, because nothing here runs on a GPU
any more.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    A protocol rather than a base class so that a backend can be any object
    with the right shape -- including the NVIDIA-backed provider, which wraps a
    service this project otherwise does not depend on, and the hand-written
    fakes the tests use in its place.

    The two properties are part of the contract rather than conveniences:
    ``model_id`` is written into the index manifest so an index can be traced
    back to what built it, and ``dimension`` is what lets the vector buffer be
    validated on load. A backend that cannot state both cannot be indexed
    safely.
    """

    @property
    def model_id(self) -> str:
        """Identifier recorded in the index. Must change when vectors would."""

    @property
    def dimension(self) -> int:
        """Length of every vector this backend produces."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per input, in the same order.

        Order is the contract: the caller pairs result *i* with input *i*, so a
        backend that returns them in any other order silently attaches vectors
        to the wrong chunks.
        """
