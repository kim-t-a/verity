"""Dense retrieval: compare the question's vector against every chunk's.

**What this adds that BM25 cannot.** Keyword retrieval matches *words*. It
cannot connect "how hot may this part get" to a chunk saying "maximum junction
temperature", because no word is shared. Dense retrieval matches *meaning*, by
placing the question and each chunk in the same vector space and measuring the
angle between them. Whether that actually helps depends entirely on the model
doing the embedding -- see the honest note at the end of this docstring.

**Why the same provider must embed both sides, and what happens if it does
not.** Two vectors are only comparable if the *same* model produced them. A
question embedded by model A and a chunk embedded by model B land in unrelated
coordinate systems, and the cosine between them is a meaningless number that
looks exactly like a score. Nothing raises. Retrieval returns ten chunks in a
confident order and every one of them is arbitrary.

This is the same failure shape as the index-format and rule-7 guards elsewhere
in the project -- code that runs cleanly and produces quietly wrong output --
and it is why :class:`VectorRetriever` refuses to be constructed at all unless
the provider's identity matches what the index manifest recorded. Refusing
loudly costs a keyword-only fallback; mixing silently costs every question.

**Cosine similarity, and why the vectors are normalised up front.** The cosine
of the angle between two vectors is their dot product divided by both lengths.
Normalising every chunk vector once at construction removes the division from
the per-query path, so a search is just a dot product per chunk. It also means
a provider that returns unnormalised vectors is handled correctly rather than
being trusted to have normalised them.

**A zero vector is not an error.** A chunk with no tokens embeds to all zeros,
which has no direction and therefore no angle with anything. Its similarity is
zero and it simply never scores. Dividing by its length to "normalise" it would
be a division by zero.

**What this cannot do, stated plainly.** Dense retrieval can report *how
similar* but never *why*. A BM25 result says "these terms matched"; a cosine
score says only "this number is larger". :attr:`RetrievedChunk.matched_terms`
is therefore empty for every result here, and that is a real loss of
debuggability rather than an oversight.

And the measurement caveat: the only embedding backend that exists so far is
the CPU-only hashing stand-in, which has no notion of meaning. So the tests for
this module verify the *mechanism* -- the arithmetic, the ordering, the
refusals -- and cannot verify retrieval quality. Quality is a property of the
real model, and is unmeasured until one is on the device.
"""

from __future__ import annotations

import math

from chunkstore import ChunkIndex
from config import MIN_VECTOR_SIMILARITY, RETRIEVAL_LIMIT
from embeddings import EmbeddingProvider
from models import Chunk
from retrieval import RetrievalError, RetrievedChunk, rank


class VectorUnavailable(RetrievalError):
    """This index and this provider cannot be used together.

    Every reason this is raised is a condition under which cosine scores would
    still be *produced*, and would be meaningless rather than absent. That is
    the whole point of raising: an absent score is a problem someone notices,
    and a meaningless score is not.
    """


def _unit(vector: tuple[float, ...]) -> tuple[float, ...]:
    """Scale ``vector`` to length one, or return it unchanged if it has none.

    A zero vector has no direction. Scaling it is undefined; leaving it as
    zeros is correct, because every dot product against it is zero and it can
    therefore never be mistaken for a match.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return tuple(value / norm for value in vector)


class VectorRetriever:
    """Ranks an index's chunks by cosine similarity to the embedded question.

    Satisfies :class:`retrieval.Retriever` structurally, so the hybrid retriever
    of the next stage can hold one of these and a
    :class:`~retrieval.keyword.KeywordRetriever` behind the same interface.
    """

    def __init__(self, index: ChunkIndex, provider: EmbeddingProvider) -> None:
        """Bind an index to the provider that embedded it.

        Raises:
            VectorUnavailable: the index holds no vectors, or the provider is
                not the one that produced them, or the two disagree about how
                many chunks or how many dimensions there are.
        """
        if not index.chunks:
            raise VectorUnavailable("the index holds no chunks to search")

        if not index.vectors:
            raise VectorUnavailable(
                "the index holds no vectors; it was built without an embedding "
                "backend, so only keyword retrieval is available for it"
            )

        if len(index.vectors) != len(index.chunks):
            raise VectorUnavailable(
                f"the index is inconsistent: {len(index.chunks)} chunks but "
                f"{len(index.vectors)} vectors. Pairing them would attach "
                "vectors to the wrong chunks."
            )

        manifest = index.manifest
        if manifest.provider_id is None or manifest.dimension is None:
            raise VectorUnavailable(
                "the manifest does not record which backend produced these "
                "vectors, so there is no way to check the question is being "
                "embedded the same way the chunks were"
            )

        if provider.model_id != manifest.provider_id:
            raise VectorUnavailable(
                f"the index was built by {manifest.provider_id!r} but the "
                f"configured backend is {provider.model_id!r}. Vectors from two "
                "different backends are not comparable, and their similarity "
                "would be a meaningless number rather than an error."
            )

        if provider.dimension != manifest.dimension:
            raise VectorUnavailable(
                f"the index holds {manifest.dimension}-dimension vectors but the "
                f"configured backend produces {provider.dimension}"
            )

        self._chunks: tuple[Chunk, ...] = index.chunks
        self._provider = provider
        self._dimension: int = manifest.dimension
        self._vectors: tuple[tuple[float, ...], ...] = tuple(
            _unit(vector) for vector in index.vectors
        )

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        """Vector width, echoed from the manifest the provider was checked against."""
        return self._dimension

    @property
    def model_id(self) -> str:
        """Which backend's vectors these are."""
        return self._provider.model_id

    def _embed_query(self, query: str) -> tuple[float, ...]:
        """The question as one unit vector, with its width checked.

        The width check is not defensive padding. Zipping two vectors of
        different lengths in Python silently stops at the shorter one, so a
        backend returning the wrong width would produce a *plausible* cosine
        computed from the first N dimensions and report it as a full match.
        That is the exact silent-wrong-answer shape this project keeps finding.
        """
        (vector,) = self._provider.embed([query])
        if len(vector) != self._dimension:
            raise VectorUnavailable(
                f"the backend returned a {len(vector)}-dimension query vector "
                f"but the index holds {self._dimension}-dimension vectors; "
                "scoring them together would silently compare only the "
                "leading dimensions"
            )
        return _unit(tuple(vector))

    def search(self, query: str, *, limit: int = RETRIEVAL_LIMIT) -> list[RetrievedChunk]:
        """The ``limit`` chunks closest in meaning to ``query``, closest first.

        Returns an empty list rather than raising when nothing clears
        :data:`config.MIN_VECTOR_SIMILARITY`. A question about a file that could
        not be read looks exactly like this, and the caller's correct response
        to it is to refuse.
        """
        if not self._chunks:
            return []

        question = self._embed_query(query)

        scores: dict[int, float] = {}
        for position, vector in enumerate(self._vectors):
            similarity = sum(a * b for a, b in zip(question, vector))
            if similarity > MIN_VECTOR_SIMILARITY:
                scores[position] = similarity

        return rank(
            self._chunks,
            # No matched terms: see the module docstring. A cosine score has no
            # explanation to offer, and inventing one would be a lie told to
            # whoever reads the diagnostics next.
            ((position, score, ()) for position, score in scores.items()),
        )[:limit]
