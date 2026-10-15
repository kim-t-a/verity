"""Extraction: which candidate the evidence actually supports.

The ordering has four priorities and the tests are grouped by which one is being
asked about, because the interesting failures are never "the wrong answer won"
-- they are "the right signal was outranked by the wrong one", and a test that
only checks the final answer cannot tell those apart.

Every case is synthetic and deliberately shaped like a real one: the first test
is the r1/r2 revision trap, the second is the bug-database row whose summary
names a version that is not its own ``fixed_in``. Where a test mirrors a real
question, it says so, so that a failure points at a corpus situation rather than
at an abstract rule.
"""

from __future__ import annotations

import pytest
from answering.extract import extract
from verity.engine.models import Chunk, MediaType
from verity.engine.retrieval import RetrievedChunk

TEMPERATURE = "What is the maximum junction temperature?"


def retrieved(
    rel_path: str, text: str, *, rank: int = 1, location: str = "page 1"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            rel_path=rel_path,
            media_type=MediaType.PDF,
            location=location,
            ordinal=0,
            text=text,
        ),
        score=1.0,
        rank=rank,
    )


# --- priority 1: a replaced revision never wins -----------------------------


def test_a_superseded_revision_loses_despite_matching_the_question_better() -> None:
    """The revision trap, which is the reason this priority is first.

    ``105`` in the withdrawn r1 matches the question *better* than ``94`` in
    r2 -- its line carries all three of the question's content words while r2's
    carries two -- so every signal except supersession points at the wrong
    answer. Stage 5 measured that the fused ranking cannot separate them at all;
    this is the rule that breaks the tie, and it has to outrank the rest.
    """
    result = extract(
        [
            retrieved("specs/r1_WITHDRAWN.pdf", "Maximum junction temperature: 105 C", rank=1),
            retrieved("specs/r2.pdf", "Junction temperature: 94 C", rank=2),
        ],
        TEMPERATURE,
        frozenset({"specs/r1_WITHDRAWN.pdf"}),
    )

    assert result.answer == "94"
    assert result.winner is not None
    assert result.winner.rel_path == "specs/r2.pdf"
    assert result.winner.superseded is False


def test_the_superseded_candidate_is_still_considered_rather_than_dropped() -> None:
    """Demoted, not filtered -- and the count is what proves it.

    Filtering would hide the case where *every* candidate is superseded, which
    is a question whose answer is in a file the corpus retired: the honest
    response there is to answer from it and say so, not to report no evidence.
    """
    result = extract(
        [retrieved("specs/r1_WITHDRAWN.pdf", "Maximum junction temperature: 105 C")],
        TEMPERATURE,
        frozenset({"specs/r1_WITHDRAWN.pdf"}),
    )

    assert result.considered == 1
    assert result.answer == "105"


# --- priority 2: the shape the question asked for ---------------------------


def test_the_wanted_shape_beats_a_higher_overlap() -> None:
    """The bug-database row, whose version is not named in its own prose.

    A question for a version must prefer ``4.3.2`` from a ``fixed_in`` field
    over the ticket identifier, even though the identifier's line accounts for
    two of the question's six content words and the field accounts for none.
    """
    result = extract(
        [
            retrieved("specs/bug_database.csv", "fixed_in=4.3.2", rank=1),
            retrieved("specs/service.log", "Ticket ORR-1847 was closed.", rank=2),
        ],
        "Which firmware version closed ticket ORR-1847?",
        frozenset(),
    )

    assert result.answer == "4.3.2"
    assert result.winner is not None
    assert result.winner.follows_equals is True


def test_a_candidate_of_an_unwanted_shape_still_wins_when_nothing_else_exists() -> None:
    """Shapes rank, they never filter.

    A wrong guess about what the question wants must not make the answer
    unreachable, so an unmatched candidate is scored as though no shape had been
    guessed at all rather than being discarded.
    """
    result = extract([retrieved("specs/a.pdf", "TQ-40")], TEMPERATURE, frozenset())

    assert result.answer == "TQ-40"


def test_a_question_with_no_recognisable_shape_still_produces_an_answer() -> None:
    """No hint means no shape advantage for anyone, so overlap decides.

    The alternative -- treating an unguessed shape as "no answer" -- would make
    a question phrased unusually unanswerable rather than merely harder.
    """
    result = extract(
        [retrieved("specs/a.pdf", "The serial is B14.")], "Identify the device.", frozenset()
    )

    assert result.wanted_shapes == ()
    assert result.answer == "B14"


# --- priority 3: how much of the question the surroundings explain ----------


def test_more_of_the_question_accounted_for_beats_a_better_retrieval_rank() -> None:
    result = extract(
        [
            retrieved("specs/a.pdf", "Junction temperature: 94 C", rank=1),
            retrieved("specs/b.pdf", "Maximum junction temperature of the sensor: 88 C", rank=2),
        ],
        TEMPERATURE,
        frozenset(),
    )

    assert result.answer == "88"
    assert result.winner is not None
    assert result.winner.retrieval_rank == 2


def test_confidence_is_the_share_of_the_question_the_winner_explains() -> None:
    """Two of the question's three content words -- a fraction, not a guess.

    Derived rather than asserted: the contract records confidence without
    scoring it, so it must be a real quantity and not a number chosen to look
    reassuring in the output file.
    """
    result = extract(
        [retrieved("specs/r2.pdf", "Junction temperature: 94 C")], TEMPERATURE, frozenset()
    )

    assert result.confidence == pytest.approx(2 / 3)


# --- priority 4: a field's value beats a mention, then rank, then position ---


def test_a_field_value_beats_a_prose_mention_at_equal_overlap() -> None:
    """The distinction the whole ``follows_equals`` signal exists for.

    Both lines account for the same three content words, and the prose line
    ranks *first*. What separates them is that one spells the value as a field
    and the other names it in a sentence -- which is exactly how the real corpus
    distinguishes a version that closed a ticket from one named in a summary.
    """
    result = extract(
        [
            retrieved(
                "specs/notes.txt",
                "The recorded maximum junction temperature reached 105 C during the run.",
                rank=1,
            ),
            retrieved("specs/datasheet.txt", "maximum junction temperature=94", rank=2),
        ],
        TEMPERATURE,
        frozenset(),
    )

    assert result.answer == "94"


def test_a_field_value_does_not_beat_a_prose_mention_that_explains_more() -> None:
    """Overlap outranks the structural hint, and that order was measured.

    Promoting the field hint above overlap looked obviously right and cost two
    of ten sample questions: every ``key=value`` in the corpus became a trump,
    so a datasheet's ``212`` field beat the ``94`` being asked about, and a
    *different* ticket's ``fixed_in=4.2.0`` beat the row the incident was about.
    A field is not evidence of being the answer; only the surrounding words can
    say which of several identically-shaped rows the question means.
    """
    result = extract(
        [
            retrieved("specs/datasheet.txt", "junction=94", rank=1),
            retrieved(
                "specs/notes.txt",
                "The recorded maximum junction temperature reached 105 C during the run.",
                rank=2,
            ),
        ],
        TEMPERATURE,
        frozenset(),
    )

    assert result.answer == "105"


def test_the_leftmost_value_wins_when_nothing_else_separates_them() -> None:
    """A deterministic last resort, and a positional one.

    The reversed text is asserted too, because "94 wins" on its own could be a
    preference for the value rather than for its position -- and a rule that
    quietly preferred particular values would be unfalsifiable.
    """
    forwards = extract([retrieved("specs/a.pdf", "Values 94 and 95")], "Identify it.", frozenset())
    backwards = extract(
        [retrieved("specs/a.pdf", "Values 95 and 94")], "Identify it.", frozenset()
    )

    assert forwards.answer == "94"
    assert backwards.answer == "95"


# --- nothing to choose from -------------------------------------------------


def test_no_results_yields_an_empty_answer_rather_than_an_exception() -> None:
    """The refusal case has to be a value, not an error.

    A question about a file that could not be read produces no chunks, and the
    caller's correct response is to write the refusal -- so it must arrive as an
    empty answer rather than as something the caller has to catch.
    """
    result = extract([], TEMPERATURE, frozenset())

    assert result.answer == ""
    assert result.confidence == 0.0
    assert result.winner is None
    assert result.considered == 0


def test_a_result_list_with_no_values_in_it_also_refuses() -> None:
    result = extract(
        [retrieved("specs/a.pdf", "Nothing numeric is recorded in this document.")],
        TEMPERATURE,
        frozenset(),
    )

    assert result.answer == ""
    assert result.winner is None


# --- the value itself -------------------------------------------------------


def test_the_value_is_returned_as_the_corpus_spells_it() -> None:
    """Never normalised, never trimmed to a fragment.

    The answer file has to contain what a human reading the document would see.
    ``Q3`` alone is a different answer from ``Q3 FY27``, and normalisation is a
    comparison form rather than a spelling.
    """
    result = extract(
        [retrieved("specs/plan.txt", "Launched in Q3 FY27 for the datacentre.")],
        "Which fiscal quarter did the TQ-40 launch?",
        frozenset(),
    )

    assert result.answer == "Q3 FY27"
    assert result.answer != "Q3"


def test_an_empty_question_does_not_divide_by_zero() -> None:
    """No content words means no fraction to report, and no crash.

    The confidence is guarded because the query reaching here comes from the
    harness, and a question that tokenises to nothing must degrade to a zero
    confidence rather than to a traceback that writes no output file at all.
    """
    result = extract([retrieved("specs/a.pdf", "The value is 94.")], "", frozenset())

    assert result.answer == "94"
    assert result.confidence == 0.0
