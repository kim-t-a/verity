"""The embedding interface.

There is no implementation left to test here. The challenge-era local model,
its hashing stand-in and the rule-7 CPU guard all existed to satisfy a GPU
requirement this project no longer has, and they went with it.

What remains is the Protocol, and the one property worth pinning about a
Protocol is what does and does not satisfy it -- because retrieval is written
against this door, and a backend of the wrong shape should be refused at the
door rather than fail at the point of use.

The cross-process determinism test that used to live here is gone with its
subject. It existed because a *local* embedder built on the randomised builtin
``hash()`` would, without raising, produce unrelated vectors in the ``--index``
process and in every query process. Vectors now come from a service, so that
failure mode is not reachable. The general form of it -- an index whose vectors
were built by a different provider than the one answering -- is still tested,
in ``test_retrieval_vector.py``, against the provider id recorded in the
manifest.
"""

from __future__ import annotations

from verity.engine.embedding_provider import EmbeddingProvider


class StubProvider:
    """The smallest thing that can be an embedding backend."""

    @property
    def model_id(self) -> str:
        return "stub-v1"

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_a_hand_written_provider_satisfies_the_protocol() -> None:
    """The tests' fakes and the real NVIDIA provider go through the same door."""
    assert isinstance(StubProvider(), EmbeddingProvider)


def test_an_object_without_embed_does_not_satisfy_it() -> None:
    class NotAProvider:
        model_id = "nope"
        dimension = 4

    assert not isinstance(NotAProvider(), EmbeddingProvider)


def test_a_provider_that_cannot_state_its_identity_does_not_satisfy_it() -> None:
    """``model_id`` and ``dimension`` are contract, not convenience.

    Without them an index cannot record what built its vectors, and a vector
    buffer cannot be validated on load -- so a backend missing either is not
    one, however well it embeds.
    """

    class Anonymous:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    assert not isinstance(Anonymous(), EmbeddingProvider)
