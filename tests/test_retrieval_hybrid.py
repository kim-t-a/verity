"""Hybrid fusion: one ranking from two that disagree.

Two kinds of test here, and the split matters.

The **mechanism** tests take hand-computable rankings and assert the fused
scores and order exactly. RRF's arithmetic is simple enough to work out on
paper, so there is no reason to assert anything looser than equality -- and a
test that only checked "the fused list is not empty" would pass for a fusion
that ignored its inputs entirely.

The **disagreement** tests are the ones this stage exists for. Measurement on
the sample corpus found exactly one question where keyword and vector rank the
same two files in opposite orders, and the files are a live datasheet and its
withdrawn predecessor. The fused result of that disagreement is a dead heat,
and these tests pin that down -- including the part that is uncomfortable: which
of the two ends up first is decided by the order the retriever names were
registered in, not by anything about the documents.

That is not a bug to fix here. It is the reason the answering stage must demote
superseded revisions by an explicit rule instead of trusting retrieval, and a
test that asserts the flakiness is what stops someone later from "tidying up"
the component order and silently changing an answer.
"""

from __future__ import annotations

import pytest

from config import FUSION_CANDIDATE_DEPTH, RETRIEVAL_LIMIT, RRF_K
from models import Chunk, MediaType
from retrieval import RetrievalError, RetrievedChunk, Retriever, rank
from retrieval.hybrid import (
    HybridRetriever,
    HybridUnavailable,
    describe_sources,
    reciprocal_rank_fusion,
)


def chunk(name: str) -> Chunk:
    """One chunk whose path *is* its name, so assertions can name what they mean.

    ``name`` is a whole filename, including its extension, so that a test reads
    as the citation it is asserting.
    """
    return Chunk(
        rel_path=f"specs/{name}",
        media_type=MediaType.PDF,
        location="page 1",
        ordinal=0,
        text=f"text of {name}",
    )


class StubRetriever:
    """A retriever that returns a fixed order, and records what it was asked for.

    Recording ``limit`` is not incidental: the hybrid is supposed to ask each
    component for *more* than the caller wants, and a component asked for
    exactly ten could never contribute an eleventh-place chunk that fusion
    should have rescued.
    """

    def __init__(self, chunks: list[Chunk], *, scores: list[float] | None = None) -> None:
        self._chunks = list(chunks)
        self._scores = scores if scores is not None else [
            float(len(self._chunks) - index) for index in range(len(self._chunks))
        ]
        self.asked: list[int] = []

    def search(self, query: str, *, limit: int = RETRIEVAL_LIMIT) -> list[RetrievedChunk]:
        self.asked.append(limit)
        return [
            RetrievedChunk(chunk=item, score=self._scores[index], rank=index + 1)
            for index, item in enumerate(self._chunks[:limit])
        ]


# --- The arithmetic, on paper -------------------------------------------------


def test_fusion_arithmetic_matches_the_formula() -> None:
    """The scores, computed by hand, to full precision.

    ``1/(60+1) + 1/(60+2)`` for a chunk both retrievers returned, and the single
    term for one only the first returned. Asserted exactly rather than
    approximately because at ``k=60`` the terms are small and close together,
    and a tolerance loose enough to hide an off-by-one in the rank would hide
    the entire structure.
    """
    fused = reciprocal_rank_fusion([("keyword", [7, 3]), ("vector", [3, 9])], k=60)

    assert fused[7] == pytest.approx(1 / 61, abs=1e-12)  # keyword rank 1 only
    assert fused[3] == pytest.approx(1 / 62 + 1 / 61, abs=1e-12)  # both, ranks 2 and 1
    assert fused[9] == pytest.approx(1 / 62, abs=1e-12)  # vector rank 2 only
    assert set(fused) == {7, 3, 9}


def test_ranks_are_one_based() -> None:
    """A 0-based rank would make the leading term ``1/k`` instead of ``1/(k+1)``.

    That is a silent change to every score in the system, and it is the kind of
    off-by-one that no ranking test would catch, because the *order* would be
    identical. So the value is pinned.
    """
    fused = reciprocal_rank_fusion([("keyword", [0])], k=60)

    assert fused[0] == pytest.approx(1 / 61, abs=1e-12)
    assert fused[0] != pytest.approx(1 / 60, abs=1e-12)


# --- The property that motivates RRF -----------------------------------------


def test_agreement_beats_a_single_first_place() -> None:
    """A chunk both retrievers liked beats one only a single retriever loved.

    This is the whole reason for running two retrievers rather than picking the
    better one. A chunk at rank 5 of both lists is corroborated; a chunk at rank
    1 of one list and absent from the other rests on one opinion, which is
    exactly the opinion the other retriever was added to check.
    """
    agreed = chunk("agreed.pdf")
    solo = chunk("solo.pdf")

    keyword = StubRetriever([solo, chunk("filler-a.pdf"), chunk("filler-b.pdf"), chunk("filler-c.pdf"), agreed])
    vector = StubRetriever([chunk("filler-d.pdf"), chunk("filler-e.pdf"), chunk("filler-f.pdf"), chunk("filler-g.pdf"), agreed])

    hybrid = HybridRetriever([("keyword", keyword), ("vector", vector)])
    results = hybrid.search("anything")

    # `agreed` is 5th in both lists (2/65); `solo` is 1st in one (1/61).
    assert 2 / 65 > 1 / 61, "the premise of this test no longer holds"
    assert results[0].rel_path == "specs/agreed.pdf"


def test_only_ranks_matter_and_never_the_scores() -> None:
    """Two retrievers with wildly different score scales fuse identically.

    The defect RRF exists to avoid: BM25 scores are unbounded (the best chunk on
    this corpus scores about 12.5) and cosines are bounded by 1 (about 0.85).
    Any fusion that added them would be decided by whichever scale was larger,
    which here would always be BM25. Fusing the same two orders with the scores
    multiplied by a thousand must change nothing.
    """
    shared = [chunk("a"), chunk("b"), chunk("c")]
    other = [chunk("d"), chunk("c"), chunk("b")]

    plain = HybridRetriever(
        [("keyword", StubRetriever(shared)), ("vector", StubRetriever(other))]
    ).search("q")
    rescaled = HybridRetriever(
        [
            ("keyword", StubRetriever(shared, scores=[12000.0, 9000.0, 3000.0])),
            ("vector", StubRetriever(other, scores=[0.81, 0.80, 0.79])),
        ]
    ).search("q")

    assert [r.rel_path for r in plain] == [r.rel_path for r in rescaled]
    assert [r.score for r in plain] == pytest.approx([r.score for r in rescaled])


# --- The measured disagreement ------------------------------------------------


def _supersession_pair() -> tuple[Chunk, Chunk]:
    """The two files Stage 4b measured the retrievers disagreeing about."""
    return chunk("tq40_datasheet_r2.pdf"), chunk("tq40_datasheet_r1_WITHDRAWN.pdf")


def test_the_measured_disagreement_fuses_to_a_dead_heat() -> None:
    """The Stage 5 finding, pinned.

    Stage 4b measured, on the real corpus with the real model:

        keyword: 1. tq40_datasheet_r2.pdf   2. ..._r1_WITHDRAWN.pdf
        vector:  1. ..._r1_WITHDRAWN.pdf    2. tq40_datasheet_r2.pdf

    Both put the same two files first and second, swapped. Fused, each collects
    one first place and one second place, so the scores are equal *exactly* --
    they are the same two terms added in the other order.

    Fusion therefore does not resolve this, and cannot. The assertion is written
    as equality of the two fused scores rather than as an expected final order,
    because an expected final order would be a claim the arithmetic does not
    support.
    """
    live, withdrawn = _supersession_pair()

    keyword = StubRetriever([live, withdrawn])
    vector = StubRetriever([withdrawn, live])

    hybrid = HybridRetriever([("keyword", keyword), ("vector", vector)])
    results = hybrid.search("maximum junction temperature")

    by_path = {result.rel_path: result for result in results}
    assert by_path["specs/tq40_datasheet_r2.pdf"].score == pytest.approx(
        by_path["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].score, abs=1e-15
    )


def test_the_dead_heat_is_broken_by_registration_order_and_not_by_evidence() -> None:
    """Which revision comes first depends on the component list, not the corpus.

    The uncomfortable half of the finding above. The tie has to be broken by
    something, and what breaks it is corpus position, which the hybrid assigns
    in the order the retrievers were registered. Swap the two components and the
    winner swaps with them -- same documents, same question, same evidence.

    This is asserted as a *fact about the current implementation* so that it
    cannot be discovered later by an answer being wrong. It is precisely why
    Stage 6 must demote withdrawn revisions explicitly: a system whose citation
    for a live datasheet depends on argument order is not one to leave deciding
    that question on its own.
    """
    live, withdrawn = _supersession_pair()

    keyword_first = HybridRetriever(
        [("keyword", StubRetriever([live, withdrawn])), ("vector", StubRetriever([withdrawn, live]))]
    ).search("q")
    vector_first = HybridRetriever(
        [("vector", StubRetriever([withdrawn, live])), ("keyword", StubRetriever([live, withdrawn]))]
    ).search("q")

    assert keyword_first[0].rel_path == "specs/tq40_datasheet_r2.pdf"
    assert vector_first[0].rel_path == "specs/tq40_datasheet_r1_WITHDRAWN.pdf"


def test_the_fused_order_beats_the_corpus_order_when_there_is_no_disagreement() -> None:
    """A tie broken by position must not displace a chunk the evidence favours.

    The counterpart to the two tests above: when the retrievers do *not* tie,
    registration order must be irrelevant. Two chunks both returned by both
    retrievers, at ranks 1 and 2 respectively, keep that order whichever way the
    components are listed.
    """
    first, second = chunk("first.pdf"), chunk("second.pdf")

    forward = HybridRetriever(
        [("keyword", StubRetriever([first, second])), ("vector", StubRetriever([first, second]))]
    ).search("q")
    reverse = HybridRetriever(
        [("vector", StubRetriever([first, second])), ("keyword", StubRetriever([first, second]))]
    ).search("q")

    assert [r.rel_path for r in forward] == ["specs/first.pdf", "specs/second.pdf"]
    assert [r.rel_path for r in reverse] == ["specs/first.pdf", "specs/second.pdf"]


# --- Degraded modes -----------------------------------------------------------


def test_a_single_retriever_preserves_its_own_order() -> None:
    """With one component, fusion must be a no-op on the ordering.

    This is the lexical-only degradation path -- no weights, no GPU, an index
    built without embeddings -- and it must not shuffle a ranking that was
    already correct. It holds because ``1/(k+rank)`` still decreases with rank,
    so the fused order and the input order are the same. Worth asserting rather
    than assuming: a fusion that reordered here would silently degrade every
    keyword-only run.
    """
    original = [chunk(name) for name in ("a", "b", "c", "d")]
    component = StubRetriever(original)

    hybrid = HybridRetriever([("keyword", component)])

    assert hybrid.components == ("keyword",)
    assert [r.rel_path for r in hybrid.search("q")] == [c.rel_path for c in original]


def test_the_components_are_reported_in_order() -> None:
    """The degraded-mode signal: a hybrid naming one component is a partial system."""
    hybrid = HybridRetriever([("keyword", StubRetriever([chunk("a")])), ("vector", StubRetriever([chunk("b")]))])

    assert hybrid.components == ("keyword", "vector")
    assert hybrid.rrf_k == RRF_K


def test_an_empty_hybrid_is_refused() -> None:
    """A retriever that returns nothing for every question must not be buildable.

    Its graded output would be indistinguishable from a corpus that genuinely
    lacks the answer, which is the silent-failure shape this project keeps
    meeting. The caller has to state what no retrievers means instead.
    """
    with pytest.raises(HybridUnavailable):
        HybridRetriever([])

    assert issubclass(HybridUnavailable, RetrievalError)


@pytest.mark.parametrize(("k", "depth"), [(-1, 1), (60, 0), (60, -3)])
def test_nonsensical_fusion_parameters_are_refused(k: int, depth: int) -> None:
    with pytest.raises(HybridUnavailable):
        HybridRetriever([("keyword", StubRetriever([chunk("a")]))], k=k, depth=depth)


def test_components_are_asked_for_more_than_the_caller_wants() -> None:
    """Fusion can only choose between candidates it has been shown."""
    component = StubRetriever([chunk(f"c{index}") for index in range(50)])
    hybrid = HybridRetriever([("keyword", component)], depth=FUSION_CANDIDATE_DEPTH)

    results = hybrid.search("q", limit=5)

    assert len(results) == 5
    assert component.asked == [5 * FUSION_CANDIDATE_DEPTH]


def test_the_limit_is_not_exceeded_even_when_every_component_returns_more() -> None:
    keyword = StubRetriever([chunk(f"k{index}") for index in range(20)])
    vector = StubRetriever([chunk(f"v{index}") for index in range(20)])

    results = HybridRetriever([("keyword", keyword), ("vector", vector)]).search("q", limit=3)

    assert len(results) == 3


def test_a_limit_of_zero_returns_nothing_rather_than_everything() -> None:
    """Slicing ``[:0]`` is safe, but asking components for ``0 * depth`` is not
    obviously so, and an early return keeps a degenerate call cheap."""
    hybrid = HybridRetriever([("keyword", StubRetriever([chunk("a")]))])

    assert hybrid.search("q", limit=0) == []


# --- Provenance ---------------------------------------------------------------


def test_a_chunk_found_by_both_names_both_retrievers() -> None:
    """A fused score is on its own scale, so it cannot be read without origins."""
    both, keyword_only = chunk("both.pdf"), chunk("keyword-only.pdf")

    hybrid = HybridRetriever(
        [
            ("keyword", StubRetriever([keyword_only, both])),
            ("vector", StubRetriever([both])),
        ]
    )
    results = hybrid.search("q")
    by_path = {result.rel_path: result for result in results}

    assert by_path["specs/both.pdf"].sources == (("keyword", 2), ("vector", 1))
    assert by_path["specs/keyword-only.pdf"].sources == (("keyword", 1),)


def test_sources_are_sorted_so_the_output_is_stable() -> None:
    """Dict iteration order must not leak into a printed diagnostic."""
    both = chunk("both")

    forward = HybridRetriever(
        [("vector", StubRetriever([both])), ("keyword", StubRetriever([both]))]
    ).search("q")
    backward = HybridRetriever(
        [("keyword", StubRetriever([both])), ("vector", StubRetriever([both]))]
    ).search("q")

    assert forward[0].sources == backward[0].sources == (("keyword", 1), ("vector", 1))


def test_matched_terms_survive_fusion_from_whichever_half_has_them() -> None:
    """Keyword can explain itself and vector cannot, so the explanation is kept.

    Losing the terms in fusion would throw away the only diagnostic the system
    has for a BM25 ranking, at exactly the point where the ranking becomes
    hardest to read.
    """
    explained = chunk("explained.pdf")
    unexplained = chunk("unexplained.pdf")

    class ExplainingRetriever(StubRetriever):
        def search(self, query: str, *, limit: int = RETRIEVAL_LIMIT) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(chunk=explained, score=5.0, rank=1, matched_terms=("junction", "temperature")),
                RetrievedChunk(chunk=unexplained, score=4.0, rank=2, matched_terms=("junction",)),
            ]

    hybrid = HybridRetriever([("keyword", ExplainingRetriever([explained])), ("vector", StubRetriever([explained]))])
    results = hybrid.search("q", limit=2)
    by_path = {result.rel_path: result for result in results}

    assert by_path["specs/explained.pdf"].matched_terms == ("junction", "temperature")


def test_describe_sources_reads_as_which_half_wanted_it() -> None:
    assert describe_sources(RetrievedChunk(chunk=chunk("a"), score=1.0, rank=1)) == "-"
    assert (
        describe_sources(
            RetrievedChunk(chunk=chunk("a"), score=1.0, rank=1, sources=(("keyword", 2), ("vector", 1)))
        )
        == "keyword#2+vector#1"
    )


# --- The shared ordering ------------------------------------------------------


def test_rank_attaches_origins_without_disturbing_the_existing_callers() -> None:
    """``origins`` is additive: omitting it must leave the old behaviour intact."""
    chunks = [chunk("a"), chunk("b")]

    without = rank(chunks, [(0, 2.0, ("x",)), (1, 1.0, ())])
    with_origins = rank(chunks, [(0, 2.0, ("x",)), (1, 1.0, ())], origins={0: (("keyword", 1),)})

    assert [r.rel_path for r in without] == [r.rel_path for r in with_origins]
    assert all(r.sources == () for r in without)
    assert with_origins[0].sources == (("keyword", 1),)
    assert with_origins[1].sources == ()


def test_a_position_absent_from_a_sparse_origins_map_gets_an_empty_tuple() -> None:
    """A sparse map must not raise for the positions it omits."""
    results = rank([chunk("a"), chunk("b")], [(0, 2.0, ()), (1, 1.0, ())], origins={1: (("vector", 1),)})

    assert results[0].sources == ()
    assert results[1].sources == (("vector", 1),)


# --- The interface ------------------------------------------------------------


def test_the_hybrid_satisfies_the_retriever_protocol() -> None:
    """The answering stage holds one object and must not care how many are behind it."""
    hybrid = HybridRetriever([("keyword", StubRetriever([chunk("a")]))])

    assert isinstance(hybrid, Retriever)


def test_every_component_came_back_empty_so_the_fused_result_is_empty() -> None:
    """A corpus with no match is a normal outcome, not an exception."""
    hybrid = HybridRetriever(
        [("keyword", StubRetriever([])), ("vector", StubRetriever([]))]
    )

    assert hybrid.search("a question nothing answers") == []
