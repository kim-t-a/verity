"""Retrieval: turning a question into the chunks most likely to answer it.

Retrieval's job is deliberately narrow. It does not decide what the answer *is*
-- that is answer extraction, and it comes later. It decides **what to look at**,
and it is scored on a different thing than the final answer is: a retrieval miss
that never surfaces the right chunk cannot be recovered downstream, because
nothing later can read a chunk that was never returned.

Three shapes are worth stating up front, because they explain the interface.

**Retrieval returns chunks, never answers.** A chunk carries its own provenance
-- which file, which page, which row -- so the citation is carried along with
the text from the moment it is found. Reconstructing provenance afterwards would
mean mapping a result back to a file by guessing.

**Every result carries a score and the terms that matched.** When a ranking is
surprising, the first question is always "why did this score as it did?", and
the answer is almost always "these terms matched, and these did not". Keeping
that is what makes the ranking debuggable rather than merely disappointing.

**Results are ordered deterministically.** The index is read in a fresh process
for every question, and a ranking that varied between runs would make a wrong
answer impossible to reproduce. Ties are broken by position in the corpus, so
the same query against the same index always returns the same list.

The interface is a protocol rather than a base class, for the same reason
:class:`embeddings.EmbeddingProvider` is: the vector and hybrid retrievers of the
next two stages must be able to satisfy it without inheriting from anything here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from config import RETRIEVAL_LIMIT
from models import Chunk


class RetrievalError(Exception):
    """Retrieval could not run."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned for a query, with why it was returned.

    Deliberately not the :class:`~models.Chunk` itself. A chunk is a fact about
    the corpus; this is a *judgement about a chunk for one particular query* --
    the score only means anything relative to the query that produced it, and
    the rank only means anything relative to the other results. Keeping them
    apart stops a score from being mistaken for a property of the document.
    """

    chunk: Chunk
    score: float
    rank: int
    """1-based position in the result list."""

    matched_terms: tuple[str, ...] = ()
    """Which query terms this chunk actually matched, sorted.

    The single most useful field when a ranking looks wrong: a chunk scoring
    lower than expected is nearly always a chunk that matched two terms instead
    of five, and a chunk scoring higher than expected has usually matched one
    very rare term.
    """

    sources: tuple[tuple[str, int], ...] = ()
    """Which retrievers returned this chunk, and at what rank, sorted by name.

    Empty for a result from a single retriever, where it would only restate the
    obvious; populated by the hybrid retriever, where it is the fusion analogue
    of :attr:`matched_terms` and answers the same question -- *why did this
    score as it did?*

    It matters more here than next to a BM25 score, because a fused score is on
    a scale of its own (see :data:`config.RRF_K`): it is not a cosine and not a
    term weight, so without the origins a reader has no way to tell whether a
    number came from one strong vote or two weak ones. Those are very different
    kinds of evidence and the fused score alone cannot distinguish them.
    """

    @property
    def rel_path(self) -> str:
        """The citation string. Proxied so callers do not reach through two objects."""
        return self.chunk.rel_path

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def location(self) -> str:
        return self.chunk.location


@runtime_checkable
class Retriever(Protocol):
    """Ranks an index's chunks against a query.

    ``search`` is a method rather than a free function because every
    implementation has setup worth doing once: term statistics for BM25, a
    loaded vector matrix for the embedding path. The index is loaded once per
    question and answered once, so paying for that setup per *retriever* is
    right, and paying for it per *query* would not be.
    """

    def search(self, query: str, *, limit: int = RETRIEVAL_LIMIT) -> list[RetrievedChunk]:
        """The best ``limit`` chunks for ``query``, best first.

        Returns an empty list rather than raising when nothing matches. A query
        with no matches is a normal outcome -- it is exactly what a question
        about a file that could not be read looks like -- and the caller's
        correct response to it is to refuse, not to handle an exception.
        """


def rank(
    chunks: Sequence[Chunk],
    scored: Iterable[tuple[int, float, tuple[str, ...]]],
    *,
    origins: Mapping[int, tuple[tuple[str, int], ...]] | None = None,
) -> list[RetrievedChunk]:
    """Order scored chunks deterministically and number them.

    ``scored`` pairs a chunk's position in ``chunks`` with its score and the
    terms it matched. Sorting on ``(-score, position)`` rather than score alone
    is the whole reason this is a shared function. Two chunks scoring
    identically are common -- a spreadsheet's rows are near-identical documents
    -- and leaving their order to the sort's stability means leaving it to the
    order a dict happened to be built in. Position in the corpus is a
    tie-breaker that is arbitrary but *reproducible*, which is the property that
    matters: a wrong answer has to be reproducible before it can be debugged.

    ``origins`` optionally maps a chunk's position to the retrievers that
    returned it and where each ranked it, and is copied onto the results. It is
    passed separately from ``scored`` because it is provenance *about* a score
    rather than part of one, and because the single-retriever callers have none
    to give.
    """
    ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
    return [
        RetrievedChunk(
            chunk=chunks[position],
            score=score,
            rank=number,
            matched_terms=terms,
            sources=(origins or {}).get(position, ()),
        )
        for number, (position, score, terms) in enumerate(ordered, start=1)
    ]
