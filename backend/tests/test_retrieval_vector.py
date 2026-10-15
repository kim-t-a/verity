"""Dense (vector) retrieval.

The most valuable tests in this file are the ones that assert a **refusal**.
Every condition :class:`VectorUnavailable` covers is one where cosine scores
would still be produced -- just meaningless ones -- so a retriever that
tolerated any of them would return a confident, arbitrary ranking and never
say so. Absence of a score is noticed; a meaningless score is not.

The tests that assert *ranking* verify arithmetic and ordering, not quality.
The only embedding backend available is the hashing stand-in, which has no
notion of meaning, so nothing here can show that dense retrieval finds
anything keyword retrieval would miss. That claim waits for a real model.
"""

from __future__ import annotations

import math

import pytest

from verity.engine.chunkstore import ChunkIndex, build_manifest, load, save
from verity.engine.config import RETRIEVAL_LIMIT
from verity.engine.embeddings.hashing import HashingEmbedder
from verity.engine.models import Chunk, MediaType
from verity.engine.retrieval import Retriever
from verity.engine.retrieval.keyword import KeywordRetriever
from verity.engine.retrieval.vector import VectorRetriever, VectorUnavailable

DIMENSION = 8


def chunk(
    text: str,
    rel_path: str = "specs/datasheet.pdf",
    location: str = "page 1",
    ordinal: int = 0,
) -> Chunk:
    """One chunk.

    ``rel_path`` is the identity tests assert on, not ``ordinal``: the
    retriever preserves the caller's chunk order, so an ordinal taken from a
    helper default would be the same number for every chunk in a test and would
    make "these came back in the right order" pass without checking anything.
    """
    return Chunk(
        rel_path=rel_path,
        media_type=MediaType.PDF,
        location=location,
        ordinal=ordinal,
        text=text,
    )


def chunks_named(*names: str) -> list[Chunk]:
    """One chunk per name, so a test's assertions can name what it expects."""
    return [chunk(f"text of {name}", rel_path=name) for name in names]


class StubProvider:
    """A backend with no model behind it: it returns whatever it is told to.

    Needed because the interesting failures are disagreements between an index
    and a provider, and those cannot be produced by a backend that always
    embeds honestly. ``model_id`` and ``dimension`` are settable so a test can
    lie about them one at a time.
    """

    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        *,
        model_id: str = "stub-v1",
        dimension: int = DIMENSION,
        default: list[float] | None = None,
    ) -> None:
        self._vectors = vectors or {}
        self._model_id = model_id
        self._dimension = dimension
        self._default = default if default is not None else [0.0] * dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors.get(text, self._default)) for text in texts]


def index_of(
    chunks: list[Chunk],
    vectors: list[list[float]] | None = None,
    *,
    provider_id: str | None = "stub-v1",
    dimension: int | None = DIMENSION,
) -> ChunkIndex:
    manifest = build_manifest(
        fingerprint="test",
        chunks=chunks,
        file_count=1,
        provider_id=provider_id,
        dimension=dimension,
        status_counts={"ok": 1},
        skipped_count=0,
    )
    return ChunkIndex(
        manifest=manifest,
        chunks=tuple(chunks),
        vectors=tuple(tuple(v) for v in (vectors or [])),
    )


def axis(position: int, *, dimension: int = DIMENSION, scale: float = 1.0) -> list[float]:
    """A unit vector pointing along one axis -- so cosines are exact and readable."""
    vector = [0.0] * dimension
    vector[position] = scale
    return vector


# --- The arithmetic ---------------------------------------------------------


def test_an_identical_vector_scores_one_and_ranks_first() -> None:
    """The one cosine value that can be stated without measuring anything."""
    retriever = VectorRetriever(
        index_of(chunks_named("decoy.txt", "target.txt"), [axis(0), axis(1)]),
        StubProvider({"question": axis(1)}),
    )

    results = retriever.search("question")

    assert results[0].rel_path == "target.txt"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].rank == 1


def test_the_closer_of_two_vectors_wins() -> None:
    """Two chunks off-axis by different amounts, so the order is not a tie."""
    tilted_towards = [0.9, 0.4358898943540674] + [0.0] * 6
    slightly_off = [0.5, 0.8660254037844386] + [0.0] * 6
    retriever = VectorRetriever(
        index_of(chunks_named("far.txt", "near.txt"), [slightly_off, tilted_towards]),
        StubProvider({"question": axis(0)}),
    )

    results = retriever.search("question")

    assert [hit.rel_path for hit in results] == ["near.txt", "far.txt"]
    assert results[0].score > results[1].score


def test_orthogonal_chunks_are_not_returned() -> None:
    """Cosine zero means no relationship, and the floor is zero.

    Returning them would pad every result list to the limit with chunks that
    have nothing to do with the question -- and a padded list is what tempts an
    answering stage into citing something unsupported.
    """
    retriever = VectorRetriever(
        index_of(chunks_named("related.txt", "unrelated.txt"), [axis(0), axis(3)]),
        StubProvider({"question": axis(0)}),
    )

    results = retriever.search("question")

    assert [hit.rel_path for hit in results] == ["related.txt"]


def test_opposite_vectors_are_not_returned() -> None:
    """A negative cosine is *less* related than a vector picked at random."""
    retriever = VectorRetriever(
        index_of(chunks_named("agreeing.txt", "opposing.txt"), [axis(0), axis(0, scale=-1.0)]),
        StubProvider({"question": axis(0)}),
    )

    results = retriever.search("question")

    assert [hit.rel_path for hit in results] == ["agreeing.txt"]


def test_a_chunk_barely_above_the_floor_is_still_returned() -> None:
    """The boundary from the other side, so the floor is a rule and not a coincidence.

    A chunk barely related to the question is weak evidence, but it *is*
    evidence, and dropping it here would be the retriever making a judgement
    about relevance that belongs to the answering stage.
    """
    barely = axis(0, scale=1.0)
    barely[1] = 1000.0  # cosine with axis(0) is a hair above zero
    retriever = VectorRetriever(
        index_of([chunk("barely")], [barely]),
        StubProvider({"question": axis(0)}),
    )

    (result,) = retriever.search("question")

    assert 0.0 < result.score < 1e-3


def test_a_provider_that_does_not_normalise_is_still_scored_correctly() -> None:
    """The retriever normalises, rather than trusting the backend to have done it.

    A real model may or may not return unit vectors, and a retriever that
    assumed it did would rank by magnitude instead of by angle -- a long chunk
    would beat a relevant one, silently.
    """
    long_agreeing = [0.0] * DIMENSION
    long_agreeing[0] = 500.0
    short_agreeing = axis(0, scale=0.001)
    retriever = VectorRetriever(
        index_of(chunks_named("short.txt", "long.txt"), [short_agreeing, long_agreeing]),
        StubProvider({"question": axis(0)}),
    )

    results = retriever.search("question")

    assert all(hit.score == pytest.approx(1.0) for hit in results)
    # Equal scores, so the corpus-position tie-break decides, not the magnitude.
    assert [hit.rel_path for hit in results] == ["short.txt", "long.txt"]


def test_scores_descend() -> None:
    vectors = [axis(0), axis(1), axis(2)]
    retriever = VectorRetriever(
        index_of([chunk(str(i)) for i in range(3)], vectors),
        StubProvider({"question": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
    )

    scores = [hit.score for hit in retriever.search("question")]

    assert scores == sorted(scores, reverse=True)


# --- Determinism ------------------------------------------------------------


def test_identical_scores_come_back_in_corpus_order() -> None:
    retriever = VectorRetriever(
        index_of([chunk(f"chunk {i}", rel_path=f"f{i}.txt") for i in range(5)], [axis(0)] * 5),
        StubProvider({"question": axis(0)}),
    )

    results = retriever.search("question")

    assert [hit.rel_path for hit in results] == [f"f{i}.txt" for i in range(5)]
    assert [hit.rank for hit in results] == [1, 2, 3, 4, 5]


def test_the_same_query_twice_gives_the_same_order() -> None:
    """One process answers one question, so this is about a rebuilt retriever."""
    chunks = chunks_named("a.txt", "b.txt", "c.txt", "d.txt")
    vectors = [axis(i % 3) for i in range(4)]
    question = StubProvider({"q": [1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]})

    def order() -> list[str]:
        return [h.rel_path for h in VectorRetriever(index_of(chunks, vectors), question).search("q")]

    assert order() == order()


# --- Nothing to return ------------------------------------------------------


def test_a_query_that_embeds_to_a_zero_vector_returns_nothing() -> None:
    """Text with no tokens is a legitimate question, and matches nothing.

    Its vector has no direction, so every cosine against it is zero -- and zero
    is the floor, so nothing is returned rather than everything at score zero.
    """
    retriever = VectorRetriever(
        index_of([chunk("a"), chunk("b")], [axis(0), axis(1)]),
        StubProvider({"": [0.0] * DIMENSION}),
    )

    assert retriever.search("") == []


def test_a_chunk_that_embeds_to_a_zero_vector_never_scores() -> None:
    """Normalising a zero vector would divide by zero; leaving it is correct."""
    retriever = VectorRetriever(
        index_of(chunks_named("empty.txt", "real.txt"), [[0.0] * DIMENSION, axis(0)]),
        StubProvider({"question": axis(0)}),
    )

    results = retriever.search("question")

    assert [hit.rel_path for hit in results] == ["real.txt"]
    assert all(math.isfinite(hit.score) for hit in results)


# --- Limits and the result shape -------------------------------------------


def test_limit_caps_the_result_count() -> None:
    retriever = VectorRetriever(
        index_of([chunk(str(i)) for i in range(20)], [axis(0)] * 20),
        StubProvider({"question": axis(0)}),
    )

    assert len(retriever.search("question", limit=3)) == 3
    assert len(retriever.search("question", limit=50)) == 20


def test_the_default_limit_comes_from_config() -> None:
    retriever = VectorRetriever(
        index_of([chunk(str(i)) for i in range(RETRIEVAL_LIMIT + 5)], [axis(0)] * (RETRIEVAL_LIMIT + 5)),
        StubProvider({"question": axis(0)}),
    )

    assert len(retriever.search("question")) == RETRIEVAL_LIMIT


def test_a_result_carries_its_provenance() -> None:
    retriever = VectorRetriever(
        index_of([chunk("some text", rel_path="specs/tq40.pdf", location="page 3 line 2")], [axis(0)]),
        StubProvider({"question": axis(0)}),
    )

    (result,) = retriever.search("question")

    assert result.rel_path == "specs/tq40.pdf"
    assert result.location == "page 3 line 2"
    assert result.text == "some text"


def test_matched_terms_are_empty_because_a_cosine_has_no_explanation() -> None:
    """Documented rather than accidental: dense retrieval cannot say *why*.

    Asserted so that a future change reporting something here is a deliberate
    decision rather than an accident -- and so nobody mistakes an empty tuple
    for a bug and "fixes" it by inventing terms that were never matched.
    """
    retriever = VectorRetriever(
        index_of([chunk("thermal throttle")], [axis(0)]),
        StubProvider({"thermal throttle": axis(0)}),
    )

    (result,) = retriever.search("thermal throttle")

    assert result.matched_terms == ()


def test_the_retriever_satisfies_the_protocol() -> None:
    retriever = VectorRetriever(index_of([chunk("a")], [axis(0)]), StubProvider())
    assert isinstance(retriever, Retriever)


def test_both_retrievers_are_interchangeable() -> None:
    """The property the hybrid stage depends on: same call, same result shape."""
    chunks = [chunk("alpha beta", rel_path="a.txt"), chunk("gamma", rel_path="b.txt")]
    index = index_of(chunks, [axis(0), axis(1)])
    provider = StubProvider({"alpha beta": axis(0)})

    keyword = KeywordRetriever(index).search("alpha beta")
    vector = VectorRetriever(index, provider).search("alpha beta")

    for results in (keyword, vector):
        assert all(isinstance(hit.rank, int) and hit.rank >= 1 for hit in results)
        assert all(hit.rel_path for hit in results)


# --- The refusals -----------------------------------------------------------


def test_an_index_without_vectors_is_refused() -> None:
    """A lexical-only index: legitimate, and simply not searchable this way."""
    with pytest.raises(VectorUnavailable, match="no vectors"):
        VectorRetriever(index_of([chunk("a")], []), StubProvider())


def test_an_empty_index_is_refused() -> None:
    with pytest.raises(VectorUnavailable, match="no chunks"):
        VectorRetriever(index_of([], []), StubProvider())


def test_a_provider_that_did_not_build_the_index_is_refused() -> None:
    """The guard this module exists for.

    Embedding the question with a different model than the chunks yields
    cosines between unrelated coordinate systems. Nothing raises; the ranking is
    simply arbitrary, and every citation drawn from it is unsupported.
    """
    index = index_of([chunk("a")], [axis(0)], provider_id="hashing-bow-v1-d8")

    with pytest.raises(VectorUnavailable, match="not comparable"):
        VectorRetriever(index, StubProvider(model_id="some-real-model-v2"))


def test_a_dimension_disagreement_is_refused() -> None:
    with pytest.raises(VectorUnavailable, match="dimension"):
        VectorRetriever(
            index_of([chunk("a")], [axis(0)]),
            StubProvider(dimension=DIMENSION * 2),
        )


def test_a_vector_count_mismatch_is_refused() -> None:
    """Two chunks and three vectors would pair each vector with the wrong chunk."""
    with pytest.raises(VectorUnavailable, match="inconsistent"):
        VectorRetriever(
            index_of([chunk("a"), chunk("b")], [axis(0), axis(1), axis(2)]),
            StubProvider(),
        )


def test_a_manifest_without_a_provider_is_refused() -> None:
    """Vectors with no record of what made them cannot be checked, so they go unused."""
    with pytest.raises(VectorUnavailable, match="does not record"):
        VectorRetriever(
            index_of([chunk("a")], [axis(0)], provider_id=None, dimension=None),
            StubProvider(),
        )


def test_a_query_vector_of_the_wrong_width_is_refused_not_truncated() -> None:
    """Zip stops at the shorter vector, so this would otherwise score silently.

    A backend returning the wrong width produces a plausible-looking cosine
    computed from whichever dimensions happen to line up. That is precisely the
    quiet-wrong-answer shape, so it raises with a message naming the widths.
    """
    retriever = VectorRetriever(
        index_of([chunk("a")], [axis(0)]),
        StubProvider({"question": [1.0, 0.0, 0.0]}),
    )

    with pytest.raises(VectorUnavailable, match="query vector"):
        retriever.search("question")


# --- Through the store, and with the only backend that exists ---------------


def test_a_saved_index_reloads_and_remains_searchable(tmp_path) -> None:
    """The full path: embed, persist, read back from disk, search.

    This is the sequence the container actually performs -- ``--index`` writes
    the vectors and each question process reads them -- and it is the one place
    a float-packing or ordering mistake would show up.
    """
    embedder = HashingEmbedder(dimension=64)
    chunks = [chunk("thermal throttle engaged", rel_path="logs/a.log"), chunk("nothing here", rel_path="b.txt")]
    index = index_of(
        chunks,
        embedding_vectors(embedder, chunks),
        provider_id=embedder.model_id,
        dimension=embedder.dimension,
    )

    save(tmp_path, index)
    reloaded = load(tmp_path)

    retriever = VectorRetriever(reloaded, embedder)
    results = retriever.search("thermal throttle engaged")

    assert results
    assert results[0].rel_path == "logs/a.log"


def test_the_hashing_backend_satisfies_the_protocol_it_was_checked_against() -> None:
    """The stand-in must be usable through the same door a real backend will be."""
    embedder = HashingEmbedder(dimension=32)
    chunks = chunks_named("first.txt", "second.txt")
    index = index_of(
        chunks,
        embedding_vectors(embedder, chunks),
        provider_id=embedder.model_id,
        dimension=embedder.dimension,
    )

    results = VectorRetriever(index, embedder).search("text of first.txt")

    assert results[0].rel_path == "first.txt"


def embedding_vectors(provider, chunks: list[Chunk]) -> list[list[float]]:
    return provider.embed([c.text for c in chunks])
