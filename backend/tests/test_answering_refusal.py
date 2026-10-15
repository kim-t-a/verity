"""Refusal: the question the corpus does not answer.

A refusal rule has an unusual cost structure. Refusing an answerable question and
answering it wrongly score identically, so the rule gains nothing by being
cautious in the abstract -- but it can only ever *gain* on questions that are
genuinely unanswerable, while every false refusal costs a point that was
otherwise won. So the tests below are as interested in when the rule stays silent
as in when it fires, and most of them are about silence.

The counter-examples are all real shapes from the corpus: a value written into a
sentence, a value after a colon in a log line, and a field name that must be read
from its own row rather than the row before it.
"""

from __future__ import annotations

from answering.candidates import Candidate, candidates_from
from answering.refusal import refusal
from verity.engine.models import Chunk, MediaType
from verity.engine.retrieval import RetrievedChunk

#: The sample's refusal question, and the reason it is one: the price exists only
#: inside a file that cannot be opened.
PRICE = "What is the unit price of the TQ-40 at 10,000 unit volume?"


def corpus(*rows: tuple[str, str]) -> list[Chunk]:
    """A whole corpus, as ``(rel_path, text)`` pairs."""
    return [
        Chunk(rel_path=path, media_type=MediaType.PDF, location=f"page {n}", ordinal=n, text=text)
        for n, (path, text) in enumerate(rows, start=1)
    ]


def winner_in(chunks: list[Chunk], path: str, value: str) -> Candidate:
    """The candidate for ``value``, built from the corpus's own chunk.

    Taken from ``chunks`` rather than constructed alongside it, so the location
    the rule looks the chunk up by is genuinely the one it was found in -- a test
    that built its own copy could pass while the lookup failed.
    """
    chunk = next(c for c in chunks if c.rel_path == path)
    results = [RetrievedChunk(chunk=chunk, score=1.0, rank=1)]
    return next(c for c in candidates_from(results, frozenset()) if c.value == value)


# --- when the rule fires ------------------------------------------------------


def test_a_value_labelled_as_something_else_is_refused() -> None:
    """The refusal question, in miniature.

    The corpus does not merely lack a price -- it says this value *is* a lead
    time. A value the corpus labels as not being the thing asked for cannot be
    the answer, and saying so is the whole rule.
    """
    row = "part_number=ORR-CBL-3301-C | unit_cost_usd=29.75 | lead_time_days=7"
    chunks = corpus(("support/rma_parts.xlsx", row))
    winner = winner_in(chunks, "support/rma_parts.xlsx", "7")

    reason = refusal(winner, PRICE, chunks)

    assert reason is not None
    assert "lead_time_days" in reason


def test_the_same_value_is_answered_when_the_field_is_about_the_question() -> None:
    """The pair above: same row, same question, a different value winning.

    Both tests read the same spreadsheet row and ask the same question. One takes
    the value filed under ``lead_time_days`` and is refused; this one takes the
    value filed under ``unit_cost_usd``, which is what the question asked for,
    and is answered. Nothing but the field name differs -- so between them they
    show the field name is the whole of the rule, rather than some other property
    of the example.
    """
    row = "part_number=ORR-CBL-3301-C | unit_cost_usd=29.75 | lead_time_days=7"
    chunks = corpus(("support/rma_parts.xlsx", row))
    winner = winner_in(chunks, "support/rma_parts.xlsx", "29.75")

    assert refusal(winner, PRICE, chunks) is None


# --- when the rule must stay silent ------------------------------------------


def test_a_value_written_into_a_sentence_is_never_refused() -> None:
    """Most values in this corpus have no field at all, and none of them is judged.

    ``Maximum junction temperature: 94 C`` states a temperature in prose. There
    is no field name to contradict the question, so the rule has nothing to say
    and the answer stands -- which is what keeps it from refusing on a hunch.
    """
    chunks = corpus(("specs/r2.pdf", "Maximum junction temperature: 94 C"))
    winner = winner_in(chunks, "specs/r2.pdf", "94")

    assert refusal(winner, "What is the maximum junction temperature?", chunks) is None


def test_a_colon_does_not_make_a_field() -> None:
    """A log line's colon introduces a message, not a field name.

    ``ERROR E7731: thermal throttle engaged`` would give the value the field name
    ``ERROR`` under a rule that accepted colons, and ``ERROR`` shares no word with
    a question about an error code -- so treating a colon as a separator would
    refuse a question the corpus answers plainly.
    """
    line = "ERROR E7731: thermal throttle engaged on die 0, clocks reduced to 60%"
    chunks = corpus(("logs/prod.log", line))
    winner = winner_in(chunks, "logs/prod.log", "E7731")

    assert refusal(winner, "What error code is logged when the thermal throttle engages?", chunks) is None


def test_a_field_name_is_read_from_its_own_row_not_the_one_before() -> None:
    """``key=value | key=value`` rows are the shape this must not misread.

    Reading back to the start of the line would label ``7`` with the *part
    number* that precedes it, and a part number shares nothing with a question
    about a price -- so the rule would refuse for entirely the wrong reason, and
    would do so on every multi-field row in the corpus. The assertion is on the
    field the reason names, not just on the fact of a refusal.
    """
    row = "part_number=ORR-CBL-3301-C | lead_time_days=7"
    chunks = corpus(("support/rma_parts.xlsx", row))
    winner = winner_in(chunks, "support/rma_parts.xlsx", "7")

    reason = refusal(winner, PRICE, chunks)

    assert reason is not None
    assert "'lead_time_days'" in reason
    assert "part_number" not in reason


def test_a_question_with_no_content_words_is_not_refused() -> None:
    """Nothing to compare against means no mismatch can be established.

    The query reaches here from the harness, and a question that tokenises to no
    content words at all -- empty, or entirely stopwords -- would otherwise
    compare as "the field shares nothing with the question" and refuse every
    question of that shape.
    """
    chunks = corpus(("support/parts.csv", "lead_time_days=7"))
    winner = winner_in(chunks, "support/parts.csv", "7")

    assert refusal(winner, "", chunks) is None
    assert refusal(winner, "what is the of", chunks) is None


def test_an_unknown_chunk_abstains_rather_than_refusing() -> None:
    """A chunk the rule cannot find is not evidence of anything.

    This is what stops a lookup failure -- a location that does not match, a
    corpus that was re-indexed underneath the candidate -- from silently
    becoming a refusal of every question.
    """
    elsewhere = corpus(("specs/other.pdf", "lead_time_days=7"))
    winner = winner_in(elsewhere, "specs/other.pdf", "7")

    assert refusal(winner, PRICE, corpus(("specs/different.pdf", "unrelated text"))) is None


def test_a_field_name_shares_a_word_in_any_spelling() -> None:
    """The comparison is on words, not on the exact string.

    ``fixed_in`` answers a question that says "fixed"; a corpus spelling the same
    fact ``fixed-in`` or ``FIXED_IN`` is the same field, and the index already
    treats those as one term everywhere else.
    """
    row = "ticket=ORR-1847 | status=closed | FIXED-IN=4.3.2"
    chunks = corpus(("support/bugs.csv", row))
    winner = winner_in(chunks, "support/bugs.csv", "4.3.2")

    assert refusal(winner, "Which firmware version fixed ticket ORR-1847?", chunks) is None


def test_the_reason_names_the_value_and_the_file() -> None:
    """A refusal discards an answer, so the diagnostic has to be actionable.

    "No answer" on its own cannot be told apart from a lookup that went wrong,
    and this rule runs on every question -- so the reason says which value was
    dropped, which field contradicted the question, and where.
    """
    row = "lead_time_days=7"
    chunks = corpus(("support/rma_parts.xlsx", row))
    winner = winner_in(chunks, "support/rma_parts.xlsx", "7")

    reason = refusal(winner, PRICE, chunks)

    assert reason is not None
    assert "'7'" in reason
    assert "'lead_time_days'" in reason
    assert "support/rma_parts.xlsx" in reason
