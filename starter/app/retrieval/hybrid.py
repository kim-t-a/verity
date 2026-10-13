"""Hybrid retrieval: one ranking from two that disagree.

**Why there are two retrievers at all.** They fail in opposite directions, and
that is the entire argument for running both.

Keyword retrieval matches *words*. It finds ``E7731`` in a log because the
question spells it, and it finds the file called ``tq40_datasheet`` because the
question names it. But it cannot connect "how hot may this part get" to a chunk
saying "maximum junction temperature", because the two share no term.

Vector retrieval matches *meaning*, and inherits the reverse weakness. It will
place the right chunk near the question when the wording differs completely,
and it is indifferent to a token that must match exactly -- a part number, a
revision, a date. Worse, it has no notion of which document is current, and on
this corpus it ranks the withdrawn datasheet revision *above* the live one.

So the two know different things. Fusion is the attempt to get both.

**Why reciprocal rank fusion, and not adding the scores.** The obvious fusion
is a weighted sum of the two scores. It does not work, and the reason is units:
a BM25 score is an unbounded sum of term weights (the best chunk on this corpus
scores about 12.5) and a cosine similarity is bounded by 1 (the best chunk
scores about 0.85). Adding them adds metres to kilograms. Any weighting that
fixed it would be a number fitted to whichever questions happened to be at hand,
and the graded corpus is different and larger.

RRF discards the scores and keeps the positions. A chunk's fused score is the
sum over the retrievers that returned it of ``1 / (RRF_K + rank)``. Ranks carry
no units, so there is nothing to normalise, no scale to balance, and no fitted
constant -- only :data:`config.RRF_K`, kept at its published value of 60 for
exactly that reason. See :mod:`config` for what that value does at this list
length.

**What fusion cannot do, stated plainly.** It cannot manufacture agreement that
is not there. Measurement on the sample corpus found one question where the two
retrievers disagree, and they disagree in the worst possible way -- keyword
prefers the current datasheet ``r2``, vector prefers the superseded
``r1_WITHDRAWN``, and both put them first and second in swapped order. Fused,
that is a dead heat: each chunk collects one first place and one second, so the
fused scores are identical to the last bit and the tie falls to corpus position.

That is not a defect to be tuned away. It is the honest outcome of combining
two signals of equal standing that genuinely conflict, and it is *recorded as a
measurement* in ``tests/test_retrieval_hybrid.py`` rather than papered over.
Its consequence is a requirement on the next stage: the answering stage must
demote superseded revisions by an explicit rule, because retrieval has been
shown not to do it. A fusion that quietly broke the tie -- by favouring the
keyword half, say -- would be hiding a judgement while pretending to make none.

**What fusion measurably bought, on the sample corpus.** Recall is a useless
measure here -- ten results out of sixty-seven chunks is a sixth of everything,
so with two competent retrievers both find the same citations and all three
configurations score 7 of 9 (the two misses are the image questions, whose files
produce no chunks at all). The sensitive measure is the *rank of the first
chunk belonging to an expected citation*, because that is what the answering
stage reads first. On that:

===========  =========  ========  ========
question     keyword    vector    hybrid
===========  =========  ========  ========
q1 (r2/r1)   1          2         **1**
q9 (two)     5          1         **2**
other five   1          1         1
===========  =========  ========  ========

Neither half is good at both. Keyword nails the supersession question and is
fifth on the two-file one; vector is the reverse. Fusion is the only
configuration that is not bad at either, which is the argument for running two
retrievers rather than picking the better one -- and it is a *robustness*
argument, not an accuracy one.

**Two honest caveats on that table.** The q1 result is not fusion merit: the two
chunks tie exactly, so rank 1 there is the positional tie-break, and keyword
happens to be registered first. Only the q9 result is the consensus effect
working as intended, where a file both halves liked beats chunks only one half
did. And with seven of nine questions already tied at rank 1, the entire signal
comes from two questions -- this corpus is far too small to establish that
fusion helps on the graded one, which is larger and where the two halves should
diverge much more.

**One retriever down is a degraded mode, not a silent one.** If the vector half
cannot start (no weights, no GPU, an index built without embeddings) the hybrid
still answers, and the fused order is then exactly the surviving retriever's
order, since a single ``1 / (RRF_K + rank)`` term still decreases with rank. The
degradation is reported through :attr:`HybridRetriever.components`, so a
lexical-only run is visible rather than merely worse.
"""

from __future__ import annotations

from collections.abc import Sequence

from config import FUSION_CANDIDATE_DEPTH, RETRIEVAL_LIMIT, RRF_K
from models import Chunk
from retrieval import RetrievalError, RetrievedChunk, Retriever, rank


class HybridUnavailable(RetrievalError):
    """No retriever was supplied, so there is nothing to fuse."""


def reciprocal_rank_fusion(
    rankings: Sequence[tuple[str, Sequence[int]]],
    *,
    k: int = RRF_K,
) -> dict[int, float]:
    """Fuse ranked lists of chunk positions into one score per position.

    ``rankings`` pairs a retriever's name with the chunk positions it returned,
    best first. The result maps each position that appeared anywhere to the sum
    of ``1 / (k + rank)`` over the lists containing it -- higher is better, and
    a position found by two retrievers scores roughly twice one found by a
    single retriever at a similar rank.

    This is deliberately a free function taking plain integers rather than one
    taking :class:`~retrieval.RetrievedChunk` objects. The arithmetic is the
    part worth testing on its own, with hand-computable inputs, and it should
    not need a corpus, an index or a retriever to be exercised.

    ``rank`` below is 1-based, so the first item in a list contributes
    ``1 / (k + 1)``. A 0-based rank would make the leading term ``1 / k`` and
    quietly change every score in the system.
    """
    fused: dict[int, float] = {}
    for _name, positions in rankings:
        for rank_number, position in enumerate(positions, start=1):
            fused[position] = fused.get(position, 0.0) + 1.0 / (k + rank_number)
    return fused


class HybridRetriever:
    """Fuses the rankings of several retrievers into one, with RRF.

    Satisfies :class:`retrieval.Retriever` structurally, like the two it wraps,
    so the answering stage holds one object and does not care how many
    retrievers are behind it or whether the vector half started at all.
    """

    def __init__(
        self,
        retrievers: Sequence[tuple[str, Retriever]],
        *,
        k: int = RRF_K,
        depth: int = FUSION_CANDIDATE_DEPTH,
    ) -> None:
        """Wrap the retrievers to fuse, each with a name for the diagnostics.

        Raises:
            HybridUnavailable: no retrievers were supplied. Constructing an
                empty hybrid would produce a retriever that returns nothing for
                every question, which is indistinguishable in the graded output
                from a corpus that does not contain the answer -- the silent
                failure this project keeps meeting. The caller has to decide
                what an empty retriever list means, and say so.
        """
        if not retrievers:
            raise HybridUnavailable(
                "no retrievers to fuse; an empty hybrid would refuse every "
                "question, which is indistinguishable from a genuine miss"
            )

        if k < 0:
            raise HybridUnavailable(f"RRF_K cannot be negative (got {k})")

        if depth < 1:
            raise HybridUnavailable(f"candidate depth must be at least 1 (got {depth})")

        self._retrievers = tuple(retrievers)
        self._k = k
        self._depth = depth

    def __len__(self) -> int:
        """How many retrievers are actually being fused."""
        return len(self._retrievers)

    @property
    def components(self) -> tuple[str, ...]:
        """The names of the fused retrievers, in order.

        The degraded-mode signal. A hybrid reporting ``("keyword",)`` is running
        lexical-only, which is a real loss of recall that no score in the output
        would reveal.
        """
        return tuple(name for name, _ in self._retrievers)

    @property
    def rrf_k(self) -> int:
        """The smoothing constant in use, echoed for the diagnostics."""
        return self._k

    def search(self, query: str, *, limit: int = RETRIEVAL_LIMIT) -> list[RetrievedChunk]:
        """The ``limit`` best chunks for ``query``, fused across the retrievers.

        Returns an empty list rather than raising when every retriever came back
        empty, which is what a question about a file that could not be read
        looks like. The caller's correct response to that is to refuse.
        """
        if limit < 1:
            return []

        # Each retriever is asked for more than the caller wants, because fusion
        # can only choose between candidates it has been shown: a chunk both
        # retrievers placed just outside their top ten deserves a place in the
        # fused ten, and asking for exactly ten would never reveal it.
        wanted = limit * self._depth

        results: dict[str, list[RetrievedChunk]] = {
            name: retriever.search(query, limit=wanted)
            for name, retriever in self._retrievers
        }

        # One identity per distinct chunk, whatever each retriever holds. Keyed
        # on the chunk itself rather than on ``id()`` so that two retrievers
        # wrapping equal-but-separate Chunk objects still fuse as one entry --
        # Chunk is frozen and hashable, so this is exact.
        positions: dict[Chunk, int] = {}
        order: list[Chunk] = []
        rankings: list[tuple[str, list[int]]] = []

        for name, found in results.items():
            listed: list[int] = []
            for result in found:
                position = positions.get(result.chunk)
                if position is None:
                    position = len(order)
                    positions[result.chunk] = position
                    order.append(result.chunk)
                listed.append(position)
            rankings.append((name, listed))

        fused = reciprocal_rank_fusion(rankings, k=self._k)

        # Provenance and explanation, gathered per position. The terms come from
        # whichever retriever could offer them -- keyword has them, vector
        # cannot -- and a chunk found by both keeps the terms that explain it
        # rather than the empty tuple that explains nothing.
        origins: dict[int, tuple[tuple[str, int], ...]] = {}
        terms: dict[int, tuple[str, ...]] = {}

        for name, found in results.items():
            for result in found:
                position = positions[result.chunk]
                origins.setdefault(position, ())
                origins[position] = origins[position] + ((name, result.rank),)
                if result.matched_terms and position not in terms:
                    terms[position] = result.matched_terms

        return rank(
            order,
            (
                (position, score, terms.get(position, ()))
                for position, score in fused.items()
            ),
            origins={position: tuple(sorted(entries)) for position, entries in origins.items()},
        )[:limit]


def describe_sources(result: RetrievedChunk) -> str:
    """``result``'s origins as ``keyword#2+vector#1``, or ``-`` when it has none.

    The fused ranking's one line of justification, in the shape a person reads
    fastest: which half of the system wanted this chunk, and how badly.
    """
    if not result.sources:
        return "-"
    return "+".join(f"{name}#{rank_number}" for name, rank_number in result.sources)
