"""Candidates: a value span plus the evidence needed to judge it.

The scoring itself is tested in ``test_answering_extract.py``. This file is about
the *carrying*: that a candidate remembers which file it came from, where it sat
in the ranking, whether that file is still current, and what surrounded it --
because each of those is a signal something downstream decides on, and one that
silently failed to propagate would look exactly like a scoring rule that was
wrong.
"""

from __future__ import annotations

from answering.candidates import candidates_from
from answering.shapes import AnswerShape
from models import Chunk, MediaType
from retrieval import RetrievedChunk


def retrieved(
    rel_path: str, text: str, *, rank: int = 1, location: str = "page 1"
) -> RetrievedChunk:
    """A retrieved chunk, built the way retrieval produces one."""
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


def only_candidate(text: str, *, superseded: frozenset[str] = frozenset()):
    """The single candidate in ``text``, asserting there is exactly one."""
    candidates = candidates_from([retrieved("specs/a.pdf", text)], superseded)
    assert len(candidates) == 1, [candidate.value for candidate in candidates]
    return candidates[0]


# --- provenance -------------------------------------------------------------


def test_a_candidate_remembers_which_file_it_came_from() -> None:
    """The citation is carried with the value, not reconstructed afterwards.

    Reconstructing it later would mean mapping a value back to a file by
    guessing, and the guess would be exactly wrong on the questions where two
    files hold the same number.
    """
    candidates = candidates_from(
        [retrieved("specs/tq40_datasheet_r2.pdf", "Junction temperature 94 C", rank=4)],
        frozenset(),
    )
    assert len(candidates) == 1
    assert candidates[0].rel_path == "specs/tq40_datasheet_r2.pdf"
    assert candidates[0].retrieval_rank == 4
    assert candidates[0].position == ("page 1", 4)


def test_a_superseded_file_marks_every_candidate_in_it() -> None:
    """The verdict arrives from the corpus-wide set, not from a local guess."""
    candidates = candidates_from(
        [retrieved("specs/r1_WITHDRAWN.pdf", "Junction temperature 105 C")],
        frozenset({"specs/r1_WITHDRAWN.pdf"}),
    )
    assert candidates[0].superseded is True


def test_a_live_file_marks_none_of_its_candidates() -> None:
    candidates = candidates_from(
        [retrieved("specs/r2.pdf", "Junction temperature 94 C")], frozenset({"specs/r1.pdf"})
    )
    assert candidates[0].superseded is False


def test_candidates_arrive_in_the_order_the_chunks_ranked() -> None:
    candidates = candidates_from(
        [
            retrieved("specs/first.pdf", "Value 11 here", rank=1),
            retrieved("specs/second.pdf", "Value 22 here", rank=2),
        ],
        frozenset(),
    )
    assert [candidate.value for candidate in candidates] == ["11", "22"]
    assert [candidate.retrieval_rank for candidate in candidates] == [1, 2]


def test_a_chunk_with_no_values_contributes_no_candidates() -> None:
    assert candidates_from([retrieved("specs/a.pdf", "Nothing numeric at all.")], frozenset()) == []


# --- the structural hint ----------------------------------------------------


def test_a_field_value_is_recognised_after_an_equals_sign() -> None:
    """``fixed_in=4.3.2`` is a field's value.

    This is the distinction the hint exists for: a real corpus row spells a
    value as ``key=value``, and a version named in a sentence is a mention of one
    rather than a value.
    """
    assert only_candidate("fixed_in=4.3.2").follows_equals is True


def test_a_field_value_is_recognised_after_a_colon() -> None:
    assert only_candidate("Temperature: 94 C").follows_equals is True


def test_a_version_named_in_prose_does_not_follow_a_separator() -> None:
    """``ticket list for 4.3.2`` is a mention.

    The corpus contains a real row whose *summary* names 4.3.2 while its
    ``fixed_in`` field is empty, and this is the signal that tells them apart.
    """
    assert only_candidate("Release notes omit ticket list for 4.3.2").follows_equals is False


def test_a_row_of_field_values_is_judged_key_by_key() -> None:
    """Every value in ``a=1 | b=2`` follows its own separator.

    Which is the uninteresting half of the rule. The interesting half is the
    next test: a short lookback is what stops a *prose* value sitting after a
    field from inheriting that field's ``=``.
    """
    candidates = candidates_from([retrieved("specs/a.csv", "a=1 | b=2")], frozenset())
    by_value = {candidate.value: candidate for candidate in candidates}
    assert by_value["1"].follows_equals is True
    assert by_value["2"].follows_equals is True


def test_a_prose_value_after_a_field_does_not_inherit_its_separator() -> None:
    """``a=1 | see note 4.3.2`` -- 4.3.2 is a mention, not that field's value.

    The separator is three characters away with other word characters between
    it, so the short window stops the scan and the hint stays False. A longer
    window would reach back past the intervening words and mark it True.
    """
    candidates = candidates_from(
        [retrieved("specs/a.csv", "a=1 | see note 4.3.2")], frozenset()
    )
    by_value = {candidate.value: candidate for candidate in candidates}
    assert by_value["1"].follows_equals is True
    assert by_value["4.3.2"].follows_equals is False


# --- the surrounding evidence ----------------------------------------------


def test_the_value_is_removed_from_its_own_context() -> None:
    """``94`` is cut out; the temperature words around it remain.

    The words are the evidence, and the value cannot be evidence for itself.
    """
    candidate = only_candidate("Maximum junction temperature: 94 C")
    assert candidate.value == "94"
    assert candidate.shape == AnswerShape.NUMBER
    assert candidate.context_terms == ("maximum", "junction", "temperature")


def test_an_identifier_does_not_vouch_for_itself() -> None:
    """The load-bearing case for removing the value's characters.

    A question naming ``ORR-FAN-2214-B`` tokenises to *orr, fan, 2214, b,
    orrfan2214b* -- exactly the terms the corpus spelling of that identifier
    contributes. Left in, every copy of the identifier would claim the
    question's own words as independent evidence and outrank whatever the
    question was actually about.
    """
    candidate = only_candidate("The ORR-FAN-2214-B assembly.")
    assert candidate.value == "ORR-FAN-2214-B"
    assert "orr" not in candidate.context_terms
    assert "fan" not in candidate.context_terms
    assert candidate.context_terms == ("assembly",)


def test_a_multi_token_value_is_removed_whole() -> None:
    """``Q3 FY27`` goes entirely, so neither half can vouch for the other."""
    candidate = only_candidate("Launched in Q3 FY27 for the datacentre.")
    assert candidate.value == "Q3 FY27"
    assert "q3" not in candidate.context_terms
    assert "fy27" not in candidate.context_terms
    assert candidate.context_terms == ("launched", "datacentre")
