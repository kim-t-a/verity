"""Shapes: what a question asks for, and what that looks like in a chunk.

Every case here is synthetic. That is the point of the module's design: the
rules are meant to hold on a corpus nobody has seen, so a test that only proves
"this works on the ten sample questions" would prove nothing about the graded
one. What is asserted is the *mechanism* -- a quarter survives whole, a number
inside an identifier is not offered as a number -- and the sample corpus is
measured separately, as a result rather than as an expectation.
"""

from __future__ import annotations

import pytest
from answering.shapes import (
    AnswerShape,
    content_terms,
    question_shapes,
    value_spans,
)
from verity.engine.tokenisation import tokenise


def shapes_in(text: str) -> list[AnswerShape]:
    return [span.shape for span in value_spans(text)]


def texts_in(text: str) -> list[str]:
    return [span.text for span in value_spans(text)]


# --- what the question is asking for ---------------------------------------


def test_a_temperature_question_wants_a_number() -> None:
    assert question_shapes("What is the maximum junction temperature?") == (
        AnswerShape.NUMBER,
    )


def test_a_firmware_question_wants_a_version_before_a_ticket_code() -> None:
    """The vote counts both hints, so the question's emphasis decides.

    ``firmware`` and ``version`` are two hints for VERSION; ``ticket`` is one for
    CODE. The question names both things, and the ordering says which one it
    wants the *value* of.
    """
    assert question_shapes("Which firmware version closed ticket ORR-1847?") == (
        AnswerShape.VERSION,
        AnswerShape.CODE,
    )


def test_a_part_number_question_wants_an_identifier() -> None:
    assert question_shapes("What is the asset tag on the backplane?") == (
        AnswerShape.IDENTIFIER,
    )


def test_a_question_with_no_hint_wants_nothing_rather_than_guessing() -> None:
    """An empty tuple is a legitimate answer, not a failure.

    The alternative -- defaulting to NUMBER -- would give every candidate of that
    shape a free advantage on a question that never asked for one.
    """
    assert question_shapes("Tell me about the device.") == ()


def test_equally_voted_shapes_fall_back_to_declaration_order() -> None:
    """``part number`` and ``version`` each vote once; the narrower one is declared first.

    A tie has to resolve somewhere, and declaration order is at least visible in
    the source rather than emergent from a dict's iteration.
    """
    assert question_shapes("What is the part number version?") == (
        AnswerShape.VERSION,
        AnswerShape.IDENTIFIER,
    )


@pytest.mark.parametrize(
    "query",
    [
        "Identify the device.",
        "How do I rebuild the index?",
        "Verify the reading.",
    ],
)
def test_a_hint_never_fires_from_inside_a_word(query: str) -> None:
    """``identify`` is not a fiscal year and ``rebuild`` is not a build number.

    Matching hints as substrings of the whitespace-deleted question made both
    fire, and a spurious vote is not harmless: it gives a shape advantage to a
    candidate the question never asked about. All three questions here are
    plausible phrasings for a question about a datacentre, which is why the
    failure would have been easy to ship.
    """
    assert question_shapes(query) == ()


def test_a_two_word_hint_matches_however_the_question_spells_it() -> None:
    """``part number``, ``part_number`` and ``partnumber`` are one hint, not three.

    Word-run matching is what gives this without also matching inside a word --
    the phrase is reconstructed from whole words rather than found as a
    substring of the question.
    """
    for spelling in ("part number", "part_number", "partnumber"):
        assert question_shapes(f"What is the {spelling}?") == (AnswerShape.IDENTIFIER,)


def test_only_a_question_that_asks_for_a_quarter_wants_one() -> None:
    """``quarter`` and ``fiscal`` ask for one; naming a fiscal year does not.

    Both sample-set phrasings reach the hint, and the third case is the one that
    used to vote wrongly: a question that merely *names* FY27 is asking for a
    number -- that year's revenue, say -- and a ``QUARTER`` vote would hand a
    shape advantage to a quarter literal it never asked about.
    """
    assert question_shapes("In which quarter does the TQ-60 enter customer sampling?") == (
        AnswerShape.QUARTER,
    )
    assert question_shapes("Which fiscal quarter did the TQ-40 launch?") == (
        AnswerShape.QUARTER,
    )
    assert AnswerShape.QUARTER not in question_shapes("What was the FY27 revenue?")


# --- what a chunk actually contains ----------------------------------------


def test_a_quarter_survives_whole_and_the_code_inside_it_is_dropped() -> None:
    """``Q3 FY27`` is the answer; ``Q3`` alone is not, and neither is ``FY27``.

    ``FY27`` matches the code pattern, so without the overlap rule it would be a
    competing candidate offering the question a fragment of the real answer.
    """
    assert texts_in("Launched in Q3 FY27.") == ["Q3 FY27"]
    assert shapes_in("Launched in Q3 FY27.") == [AnswerShape.QUARTER]


def test_a_number_inside_an_identifier_is_not_offered_as_a_number() -> None:
    """``1847`` in ``ORR-1847`` is digits, not a value.

    Both patterns match the same characters, and the longer span is the one a
    reader would call the value -- so longest-wins is a statement about the text
    rather than a preference between shapes.
    """
    spans = value_spans("Firmware 4.3.2 fixes ORR-1847 and E7731.")
    assert texts_in("Firmware 4.3.2 fixes ORR-1847 and E7731.") == [
        "4.3.2",
        "ORR-1847",
        "E7731",
    ]
    assert [span.shape for span in spans] == [
        AnswerShape.VERSION,
        AnswerShape.IDENTIFIER,
        AnswerShape.CODE,
    ]


def test_a_version_is_not_fragmented_into_its_components() -> None:
    """``4.3.2`` yields one span, never ``4.3`` or ``3`` -- the lookarounds do that.

    Without them every version in the corpus would arrive as three candidates,
    two of which are single digits that match almost any question.
    """
    assert texts_in("Version 4.3.2 released.") == ["4.3.2"]


def test_a_value_ending_a_sentence_is_still_found() -> None:
    """``is 94.`` and ``as 4.3.2.`` are how prose writes a value.

    Both used to yield *nothing at all*: the trailing guard forbade any
    following full stop, so it could not tell a sentence period from the decimal
    point it exists to protect. The failure was silent -- an empty result, not an
    error -- and it would have refused every question whose answer the corpus
    states in a sentence.
    """
    assert texts_in("The maximum junction temperature is 94.") == ["94"]
    assert texts_in("It was released as 4.3.2.") == ["4.3.2"]


def test_a_ratio_multiplier_is_still_not_a_version() -> None:
    """The case the trailing guard exists for, kept working by the fix.

    ``4.2x`` is a throughput multiplier in the log, not a version, and its
    digits must not reach the candidate list where they would compete with the
    real answer.
    """
    assert value_spans("throughput 4.2x nominal") == ()


def test_a_date_is_not_an_identifier() -> None:
    """The identifier pattern demands a letter, so ``2026-09-02`` is not one.

    Dates are separator-joined runs of digits, and admitting them would put a
    timestamp on the candidate list of every log question.

    The known cost is stated rather than hidden: the date's *components* still
    match the number pattern, so a date contributes number candidates. That is
    tolerable because the overlap rule is what decides, and a date's surrounding
    words rarely match the question's -- but it is a real imperfection, not a
    claim that dates are handled.
    """
    assert AnswerShape.IDENTIFIER not in shapes_in("Logged at 2026-09-02 14:03.")
    assert AnswerShape.NUMBER in shapes_in("Logged at 2026-09-02 14:03.")


def test_a_lone_severity_grade_is_not_an_error_code() -> None:
    """``S2`` has one digit; the code pattern requires two.

    A one-character severity grade competing with a real error code is a
    candidate that can only ever waste the ranking's top slot.
    """
    assert AnswerShape.CODE not in shapes_in("Severity S2 incident.")
    assert AnswerShape.CODE in shapes_in("Severity S2, error E7731.")


def test_a_hyphenated_english_word_is_not_an_identifier() -> None:
    """``follow-on`` and ``field-replaceable`` are words, not part numbers.

    The identifier pattern asks for letters *and* digits joined by separators.
    With only the letter test, every hyphenated adjective in the corpus became a
    candidate -- and measured, one of them won a question it had no business
    competing in, returning ``follow-on`` as an answer. It is also the rule the
    tokeniser already applies, so this keeps the two consistent.
    """
    assert AnswerShape.IDENTIFIER not in shapes_in("A follow-on part under study.")
    assert AnswerShape.IDENTIFIER not in shapes_in("Fan assembly, field-replaceable")
    assert AnswerShape.IDENTIFIER in shapes_in("Part REV-C2 is fitted.")


def test_a_chunk_with_no_values_yields_no_spans() -> None:
    assert value_spans("Nothing numeric is written here at all.") == ()


def test_spans_are_returned_left_to_right() -> None:
    """Position order, not pattern order, so a reader can follow the text.

    The patterns are tried one shape at a time, which would otherwise interleave
    the results in a way that has nothing to do with the chunk.
    """
    starts = [span.start for span in value_spans("A 1 then 22 then 333")]
    assert starts == sorted(starts)


# --- what counts as evidence -----------------------------------------------


def test_a_bare_number_is_not_evidence() -> None:
    """``TQ-40`` contributes ``tq`` and ``tq40``, but never a bare ``40``.

    Counting the model number's digits separately would let the number alone
    match any datasheet row that mentions any forty of anything.
    """
    assert content_terms("TQ-40 has 40 fans") == ("tq", "tq40", "fans")


def test_a_version_joins_to_its_numeric_form_but_is_still_not_evidence() -> None:
    """``4.3.2`` joins to ``432``, and ``432`` is still all digits, so it is dropped.

    Pinned because this is the one place the two consumers of the tokeniser
    disagree, and the disagreement is deliberate. Retrieval uses
    :func:`tokenisation.tokenise` directly and *keeps* ``432``, where it is
    genuinely discriminating. ``content_terms`` feeds the overlap heuristic,
    where a token made only of digits is what would let a row mentioning any
    forty of something match a question about forty of something else.

    So the version contributes no evidence -- and that is affordable, because
    the value's own characters are removed from its candidate's context anyway.
    """
    assert content_terms("fixed_in=4.3.2") == ("fixed",)
    assert "432" in tokenise("fixed_in=4.3.2")


def test_stopwords_and_repeats_are_dropped_but_order_is_kept() -> None:
    assert content_terms("The fan and the fan and the shroud") == ("fan", "shroud")


def test_a_hyphenated_phrase_is_evidence_from_either_spelling() -> None:
    """``field-replaceable`` and ``field replaceable`` must meet.

    The corpus writes it one way and a question is far more likely to write it
    the other, so both have to reach the same terms.
    """
    assert content_terms("field-replaceable") == content_terms("field replaceable")


@pytest.mark.parametrize("text", ["", "   ", "...,,,", "1234"])
def test_text_with_nothing_to_say_yields_no_terms(text: str) -> None:
    assert content_terms(text) == ()
