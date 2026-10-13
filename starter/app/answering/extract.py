"""Choosing the answer: which candidate value the evidence actually supports.

**The comparison is lexicographic, and that is deliberate.** The obvious design
weighs the signals -- so much for the right shape, so much for matching the
question's words -- and sums them. Every one of those weights would be a number
invented here, with nothing to fit it against except questions whose answers are
already known, and the graded corpus is different and larger. Stage 5's fusion
avoided exactly this trap by using ranks instead of scores; this module avoids
it by using an **ordering** instead of weights. Four signals, compared in
priority order, and the first that separates two candidates decides:

1. **Is the file superseded?** First, and ahead of everything, because it is the
   one signal that answers a question the others cannot see. ``94`` in r2 and
   ``105`` in r1 match the question *identically* -- same shape, same label, same
   overlap -- and differ only in which revision is current. This is the signal
   that decides the trap question, so nothing may outrank it.
2. **Does the value have the shape the question asked for?** A question wanting
   a version should prefer ``4.3.2`` to the ``1847`` inside a ticket number,
   even where the ticket number sits closer to the question's words.
3. **How much of the question do the surroundings account for?** The main
   workhorse, and the only signal that distinguishes a real answer from a number
   that merely happened to be in a well-ranked chunk.
4. **Is the value a field's value rather than a mention in prose?** A tie-breaker
   among candidates that explain the question equally well, and deliberately the
   last signal before the ranking's own opinion -- see the measurement note
   below for why it is not higher.

**Shape ranks, it does not filter.** A question with no recognisable shape hint
returns no shapes at all, and then step 2 is a tie for every candidate and step
3 decides -- which is the correct degradation, since a missing guess must not
make an answer unreachable. Likewise a candidate whose shape was not guessed
loses on step 2 but is still there to win if nothing shaped better exists.

**Every step earns its place on a question in the sample corpus.** Supersession
decides q1, where the fused ranking cannot separate the two revisions at all.
Shape decides q5, where a ticket number in a spreadsheet cell competes with the
version the question asked for. Overlap decides most of the rest, including the
multi-hop question, where it is what identifies *which* spreadsheet row is about
the incident at all.

**And one step that does not, which is worth stating.** ``follows_equals`` was
designed for a single row of ``bug_database.csv`` -- a decoy whose *summary*
names a version while its ``fixed_in`` field is empty -- and the intent was that
a field value should beat a prose mention. Measured, it does not, and promoting
it above the overlap term cost two questions outright: a datasheet's ``212``
field beat the ``94`` the question was about, and a *different* ticket's
``fixed_in=4.2.0`` beat the right row's ``4.3.2``. The reason is that
``key=value`` is everywhere in this corpus, so a field is not evidence of being
the answer -- and, more sharply, only topical overlap can tell *which* of five
identically-shaped rows is the one being asked about.

So the step stays, in last place among the signals that mean anything, where it
does the job it is actually good for: separating two candidates that explain the
question equally well, one written as a field and one named in a sentence. It is
a tie-breaker, not a trump.

**Why not an abstention threshold here.** This module always returns its best
candidate when there is one, because "no evidence at all" and "weak evidence"
are different things and only the first is this module's to judge. The second --
refusing when a question's answer genuinely is not in the readable corpus -- is
a judgement about *what the corpus could have contained*, and it belongs with
the citation logic that already has to reason about which files were readable.
See :mod:`answering.citations`.
"""

from __future__ import annotations

from dataclasses import dataclass

from answering.candidates import Candidate, candidates_from
from answering.shapes import AnswerShape, content_terms, question_shapes
from retrieval import RetrievedChunk


@dataclass(frozen=True)
class Extraction:
    """The best answer the evidence supports, and what supported it."""

    answer: str
    """The value as the corpus spells it, or ``""`` when nothing was found."""

    confidence: float
    """Recorded, never scored -- and derived rather than guessed at.

    The fraction of the question's content words that the winning candidate's
    surroundings account for. Crude on purpose: it measures *how much of the
    question this chunk addresses*, which is a real quantity, rather than an
    attempt at a probability, which would be a number with no calibration
    behind it.
    """

    winner: Candidate | None
    """The candidate chosen, or ``None`` when there was nothing to choose from."""

    wanted_shapes: tuple[AnswerShape, ...] = ()
    """What the question was read as asking for, kept for the diagnostics."""

    considered: int = 0
    """How many candidates were weighed."""


def _overlap(candidate: Candidate, question_terms: frozenset[str]) -> int:
    """How many of the question's content words the candidate's surroundings hold.

    A plain count, with no inverse-document-frequency weighting, because the
    extractor compares candidates *within one question* and every candidate is
    scored against the same question terms -- so a term that is rare corpus-wide
    is worth exactly as much here as one that is common, and weighting by rarity
    would change nothing about the ordering. (It would matter if candidates from
    different questions were ever compared. They are not.)
    """
    return sum(1 for term in candidate.context_terms if term in question_terms)


def _rank_key(
    candidate: Candidate,
    wanted: tuple[AnswerShape, ...],
    question_terms: frozenset[str],
) -> tuple[bool, int, int, bool, int, tuple[str, int]]:
    """The lexicographic ordering as a sort key. Lower sorts first, so it wins.

    One definition, used once, because the ordering *is* the algorithm here and
    a reader checking whether supersession really outranks shape -- or whether a
    prose mention can beat a field value -- should be able to read all four
    priorities in one place instead of reconstructing them from a call site.
    """
    not_wanted = len(wanted) if candidate.shape not in wanted else wanted.index(candidate.shape)
    return (
        candidate.superseded,            # 1. a replaced revision never wins
        not_wanted,                      # 2. the shape the question asked for
        -_overlap(candidate, question_terms),  # 3. how much of the question it explains
        not candidate.follows_equals,    # 4. a field's value beats a mention of one
        candidate.retrieval_rank,        #    ... then the fused ranking's own opinion
        candidate.position,              #    ... then a deterministic last resort
    )


def extract(
    results: list[RetrievedChunk], query: str, superseded: frozenset[str]
) -> Extraction:
    """The best answer value supported by ``results``, for ``query``.

    Returns an :class:`Extraction` whose ``answer`` is empty when ``results``
    contained no candidate value at all -- which for a question whose evidence
    is an unreadable file is the honest outcome, not a failure.
    """
    wanted = question_shapes(query)
    terms = frozenset(content_terms(query))

    candidates = candidates_from(results, superseded)
    if not candidates:
        return Extraction(answer="", confidence=0.0, winner=None, wanted_shapes=wanted)

    # Sorted rather than `min`-with-a-key so that the winner and the count of
    # what it beat come out of the same pass and cannot disagree about which
    # candidate was best.
    ordered = sorted(candidates, key=lambda c: _rank_key(c, wanted, terms))
    winner = ordered[0]

    return Extraction(
        answer=winner.value,
        confidence=_overlap(winner, terms) / len(terms) if terms else 0.0,
        winner=winner,
        wanted_shapes=wanted,
        considered=len(candidates),
    )
