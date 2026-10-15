"""The tokenizer, which decides what retrieval can and cannot see.

This module carries more weight in this project than a tokenizer normally would,
because the answers here are *values* -- ``4.3.2``, ``ORR-FAN-2214-B``,
``E7731`` -- rather than prose. A conventional tokenizer that splits on every
non-alphanumeric character destroys exactly the strings the challenge is asking
for, and does it invisibly: retrieval would still return results, they would just
be the wrong ones.

So the tests below are written as pairs. Each rule is checked alongside the case
it could plausibly break: separators fuse in a value *and* split in a word.
"""

from __future__ import annotations

import pytest

from verity.engine.tokenisation import joined_form, query_terms, tokenise


# --- Values survive as values ----------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("4.3.2", ["432"]),
        ("94", ["94"]),
        ("E7731", ["e7731"]),
        ("Q3", ["q3"]),
        ("FY27", ["fy27"]),
    ],
)
def test_a_value_is_one_term(written: str, expected: list[str]) -> None:
    """The failure this prevents: ``4.3.2`` -> ``4``, ``3``, ``2``.

    Those three are among the most common terms in any corpus of technical
    documents, so a query for the firmware version would match hundreds of
    chunks about equally and identify none of them. The value has to arrive as
    a single term or it cannot be found.
    """
    assert tokenise(written) == expected


def test_a_value_with_long_parts_keeps_them_as_well() -> None:
    """``2019.10`` yields the whole value *and* ``2019`` and ``10``.

    The parts survive here only because they are two or more characters; in
    ``4.3.2`` they are single characters and are dropped, which is why the
    joined form is the sole route to that value. Both behaviours come from the
    same two rules, and this test is what stops either from drifting.
    """
    assert tokenise("2019.10") == ["2019", "10", "201910"]


def test_a_separated_identifier_keeps_both_its_value_and_its_parts() -> None:
    """``ORR-FAN-2214-B`` is matched whole *and* as ``orr``, ``fan``, ``2214``.

    Whole, so a question quoting the part number finds it. In parts, so a
    question asking about "the fan assembly" finds the row without naming it.
    Both routes matter and neither subsumes the other.
    """
    tokens = tokenise("ORR-FAN-2214-B")

    assert tokens == ["orr", "fan", "2214", "orrfan2214b"]


# --- Separators split words -------------------------------------------------


def test_hyphenation_does_not_hide_a_word() -> None:
    """The case that decided the design.

    ``support/rma_parts.xlsx`` writes "Fan assembly, dual-rotor, field
    replaceable" -- with a space -- while a question is far more likely to
    hyphenate "field-replaceable". Fusing on the hyphen would make the two
    spellings mutually invisible, and the question would never reach its own
    answer.
    """
    assert tokenise("field-replaceable") == ["field", "replaceable"]
    assert tokenise("field replaceable") == ["field", "replaceable"]


def test_a_hyphenated_word_is_not_fused_because_it_has_no_digit() -> None:
    """The digit test is what keeps the two rules apart.

    ``dual-rotor`` has no digit, so no joined form is manufactured. Inventing
    ``dualrotor`` would be a word neither the corpus nor any question uses.
    """
    assert "dualrotor" not in tokenise("dual-rotor")


def test_a_file_extension_does_not_swallow_the_stem() -> None:
    """``r2.pdf`` must still be reachable from a question that says ``r2``."""
    tokens = tokenise("r2.pdf")

    assert "r2" in tokens
    assert "pdf" in tokens


# --- Length, punctuation and noise -----------------------------------------


def test_single_characters_are_dropped() -> None:
    """``ORR-FAN-2214-B`` ends in a bare ``b``; ``tq-40`` starts with ``tq``.

    Keeping one-character terms would put ``b`` in the index of every part
    number, where it would match any question containing the letter -- noise
    that no amount of idf weighting makes useful.
    """
    assert tokenise("a b c") == []
    assert "b" not in tokenise("ORR-FAN-2214-B")


def test_trailing_punctuation_is_not_part_of_the_term() -> None:
    """A sentence ending "94." must yield ``94``, not ``94.``."""
    assert tokenise("94.") == ["94"]
    assert tokenise("(180)") == ["180"]


@pytest.mark.parametrize("text", ["", "   ", "---", "...", "\n\t"])
def test_text_with_no_terms_yields_nothing(text: str) -> None:
    """Punctuation-only chunks exist -- a table's rule line -- and are not terms."""
    assert tokenise(text) == []


def test_case_is_folded() -> None:
    assert tokenise("Thermal THROTTLE") == tokenise("thermal throttle")


def test_non_ascii_words_stay_whole() -> None:
    """``naïve`` must not split at the ``ï``.

    The corpus is English, but a tokenizer that mangles accented text is one
    that will mangle something else later, and the rule costs nothing to hold.
    """
    assert tokenise("naïve") == ["naïve"]


# --- Term frequency ---------------------------------------------------------


def test_a_repeated_term_is_kept_once_per_occurrence() -> None:
    """Term frequency is what BM25 scores on, so repeats have to survive."""
    assert tokenise("batch batch batch") == ["batch", "batch", "batch"]


def test_a_run_is_not_counted_twice_for_being_its_own_joined_form() -> None:
    """``94`` arrives as a part and again as a joined form; only one may count.

    Counting it twice would inflate both its frequency and the chunk's length,
    and length is what BM25 divides by -- so a single number would distort the
    ranking of the chunk it appears in.
    """
    assert tokenise("94") == ["94"]


def test_an_underscored_constant_contributes_its_words() -> None:
    """``DEFAULT_BATCH_TIMEOUT_S`` must be reachable from "default batch timeout"."""
    tokens = tokenise("DEFAULT_BATCH_TIMEOUT_S = 180")

    assert {"default", "batch", "timeout", "180"} <= set(tokens)


# --- Query terms ------------------------------------------------------------


def test_query_terms_are_distinct_and_keep_first_seen_order() -> None:
    """Repetition in a question must not weigh more than a single mention.

    Scoring "the thermal throttle incident thermal" as if ``thermal`` counted
    twice would let a question's phrasing distort the ranking it produces.
    """
    assert query_terms("the thermal throttle incident thermal") == (
        "the",
        "thermal",
        "throttle",
        "incident",
    )


def test_query_terms_of_an_empty_question_are_empty() -> None:
    assert query_terms("?!") == ()


# --- The joined form --------------------------------------------------------


@pytest.mark.parametrize(
    ("run", "expected"),
    [("4.3.2", "432"), ("ORR-FAN-2214-B", "orrfan2214b"), ("r2.pdf", "r2pdf")],
)
def test_joined_form_removes_exactly_the_separators(run: str, expected: str) -> None:
    """It has to agree with what :func:`tokenise` emits, or a later stage that
    recognises a value by this rule would disagree with the index built by the
    other one -- and would disagree silently."""
    assert joined_form(run) == expected
    assert expected in tokenise(run)
