"""Candidates: a value span, plus everything needed to judge it.

A :class:`~answering.shapes.ValueSpan` is a fact about a piece of text -- these
characters are a version number. A :class:`Candidate` is a *claim about an
answer*: this value, in this chunk, in this file, at this position in this
ranking, with this much of the question accounted for around it. Judging one
requires all of that together, which is why it is one object.

**Why the surrounding text is kept rather than a score computed here.** The
score needs the question, and a candidate does not know its question. Keeping
the context means the scoring in :mod:`answering.extract` can be re-run against
a different question without re-extracting anything -- and, more usefully, means
a surprising ranking can be explained by printing what the candidate actually
had to work with rather than by re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass

from answering.shapes import AnswerShape, ValueSpan, content_terms, value_spans
from retrieval import RetrievedChunk


@dataclass(frozen=True)
class Candidate:
    """One possible answer: a value found in one retrieved chunk."""

    value: str
    """The value exactly as the corpus spells it. What would be submitted."""

    shape: AnswerShape
    rel_path: str
    location: str
    retrieval_rank: int
    """The rank of the chunk this value was found in, within the fused ranking."""

    context_terms: tuple[str, ...]
    """Distinct content words of the chunk, excluding the value itself.

    The evidence a candidate offers. "Maximum junction temperature ... 94 C"
    yields *maximum, junction, temperature* -- three of the four content words
    a question about it would use -- and that overlap is the main thing
    distinguishing a real answer from a number that merely happens to be near
    the top of the ranking.

    The value's characters are removed so that a value cannot vouch for itself.
    A question naming ``ORR-FAN-2214-B`` tokenises to *orr, fan, 2214, b,
    orrfan2214b*, and the corpus spelling of that same identifier is exactly
    those terms -- so leaving them in would let every copy of the identifier in
    the corpus claim the question's own words as independent evidence for
    itself, and the identifier would always outrank whatever the question was
    actually asking about.

    Only the value's characters go, not its line: the surrounding words are the
    whole point, so ``94`` is cut out of "Maximum junction temperature: 94 C"
    and the temperature words remain.
    """

    superseded: bool
    """Whether this candidate's file has been replaced by a later revision.

    Carried on the candidate rather than looked up during scoring so that the
    verdict -- which depends on text in *other* files -- is computed once, from
    the whole corpus, rather than inferred from whatever happens to be in this
    result list.
    """

    follows_equals: bool
    """Whether the value sits immediately after an ``=`` sign.

    A structural hint with real weight in this corpus, where spreadsheets,
    CSV rows and source code all spell a value as ``key=value``: ``fixed_in=4.3.2``
    is a field's value, while a ``4.3.2`` inside a sentence is a mention of one.
    The distinction matters -- this corpus contains a row whose *summary* text
    names a version that is not that row's ``fixed_in``.
    """

    @property
    def position(self) -> tuple[str, int]:
        """Where the value is, for a deterministic final tie-break."""
        return (self.location, self.retrieval_rank)


def candidates_from(
    results: list[RetrievedChunk], superseded: frozenset[str]
) -> list[Candidate]:
    """Every candidate value in ``results``, in the order the chunks ranked.

    No filtering and no scoring happens here. A value that cannot possibly be
    right is still cheaper to carry than to argue about, and every filter is a
    rule that has to be right -- so the only thing removed is the value's own
    characters from its context, which is not a judgement but the definition of
    what "the surrounding evidence" means.
    """
    candidates: list[Candidate] = []

    for result in results:
        text = result.text
        for span in value_spans(text):
            candidates.append(
                Candidate(
                    value=span.text,
                    shape=span.shape,
                    rel_path=result.rel_path,
                    location=result.location,
                    retrieval_rank=result.rank,
                    context_terms=content_terms(_without_span(text, span)),
                    superseded=result.rel_path in superseded,
                    follows_equals=_follows_equals(text, span),
                )
            )

    return candidates


def _without_span(text: str, span: ValueSpan) -> str:
    """``text`` with the value's characters deleted, its surroundings intact."""
    return text[: span.start] + text[span.end :]


def _follows_equals(text: str, span: ValueSpan, *, lookback: int = 3) -> bool:
    """Whether only whitespace and an ``=`` (or ``:``) separate the value from its key.

    A short lookback rather than a full parse: the point is to tell
    ``fixed_in=4.3.2`` from ``a ticket list for 4.3.2``, and a few characters
    settle that. A longer window would start reaching back across field
    boundaries in a ``key=value | key=value`` row, which is the one thing this
    must not do -- it would mark the *second* field's value as following the
    *first* field's separator.
    """
    prefix = text[max(0, span.start - lookback) : span.start]
    stripped = prefix.rstrip()
    return stripped.endswith(("=", ":"))
