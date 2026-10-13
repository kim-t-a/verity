"""Choosing an embedding backend, and refusing to choose one by accident.

The challenge rejects a run that never uses the GPU, so the embedding backend
is not a free choice: it is the thing that puts a model on the device. Rule 7
of this project says a CPU-only path may exist for local testing but must be
*impossible to select accidentally in the container*.

"Impossible" is a strong word, so it is worth being precise about what guards
what, and why one guard would not be enough:

1. **The default is the real backend.** With no environment at all -- which is
   how the harness runs -- the factory never considers the stand-in. A default
   that happened to be the local one would make the failure mode "works on my
   laptop, rejected on the grader", which is the single worst outcome available.
2. **Naming the stand-in is not enough to get it.** It also requires a second,
   separately-named variable set to ``1``. Two switches, because the risk is a
   stray export left in a shell profile: one variable can be forgotten, and its
   effect is silent.
3. **The index records which backend built it.** A test index therefore
   announces itself in ``manifest.json``, and cannot be mistaken for a real one
   by someone reading the output -- or by a later stage comparing providers.

The rule is about *accidental* selection, not about the code existing. Keeping
the stand-in in the shipped image is what makes the whole pipeline runnable
end-to-end in the container for smoke-testing, which is worth more than the
theoretical safety of deleting it.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from config import (
    ALLOW_LOCAL_BACKEND_ENV,
    DEFAULT_EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDER_ENV,
    HASHING_DIMENSION,
    LOCAL_EMBEDDING_PROVIDER,
)


class EmbeddingError(Exception):
    """An embedding backend could not do its job."""


class LocalBackendNotAllowed(EmbeddingError):
    """The local stand-in was requested without the second switch to allow it."""


class UnknownProvider(EmbeddingError):
    """A backend was named that this build does not know."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    A protocol rather than a base class so that a backend can be any object
    with the right shape -- including the real one Stage 4 adds, which will
    wrap a model from a library this project does not otherwise depend on.

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


def provider_from_environment() -> EmbeddingProvider | None:
    """The configured backend.

    The default is the real one, and it is now a real model rather than a
    placeholder: with no environment set at all -- which is exactly how the
    harness runs -- this loads the local sentence-transformer onto the GPU and
    returns it.

    Returns ``None`` only for a build that has no model at all. That case is
    kept because an index is still perfectly useful without vectors, and
    refusing to build one would block lexical retrieval on a model that may not
    be present in a development checkout. It is not reachable in the container:
    the image ships the weights, and the default provider is the real one.

    Raises:
        LocalModelUnavailable: the real backend was selected but could not start
            -- missing weights, or no usable device. Raised rather than
            returning ``None``, because "the model did not load" and "there is
            no model" must not be the same answer: the first is a fault to fix,
            the second is a legitimate configuration.
        LocalBackendNotAllowed: the stand-in was named without the second switch.
        UnknownProvider: a name this build does not recognise.
    """
    requested = os.environ.get(EMBEDDING_PROVIDER_ENV, DEFAULT_EMBEDDING_PROVIDER)

    if requested == LOCAL_EMBEDDING_PROVIDER:
        if os.environ.get(ALLOW_LOCAL_BACKEND_ENV) != "1":
            raise LocalBackendNotAllowed(
                f"{EMBEDDING_PROVIDER_ENV}={LOCAL_EMBEDDING_PROVIDER!r} also requires "
                f"{ALLOW_LOCAL_BACKEND_ENV}=1. The local backend is CPU-only and a run "
                "that never uses the GPU is rejected, so it is not reachable by "
                "naming it alone."
            )
        # Imported here so the stand-in is not even loaded unless it is asked
        # for by name and permitted.
        from embeddings.hashing import HashingEmbedder

        return HashingEmbedder(dimension=HASHING_DIMENSION)

    if requested == DEFAULT_EMBEDDING_PROVIDER:
        # Also imported here, and for a sharper reason: this module defines the
        # provider *protocol*, so importing the real backend at module level
        # would be a circular import. The side effect is that torch is not
        # imported by `import embeddings`, which keeps the import-graph test
        # from paying several seconds for a library it never calls.
        from embeddings.local_model import LocalModelEmbedder

        return LocalModelEmbedder()

    raise UnknownProvider(
        f"{EMBEDDING_PROVIDER_ENV}={requested!r} is not a backend this build knows; "
        f"expected {DEFAULT_EMBEDDING_PROVIDER!r} or {LOCAL_EMBEDDING_PROVIDER!r}"
    )
