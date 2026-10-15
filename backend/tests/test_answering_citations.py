"""Citations: which files were necessary, as an exact set.

Every case is synthetic and shaped like a real one, because the graded corpus is
different and larger and a rule that only holds on the ten sample questions is
worthless. Where a case mirrors the corpus, the docstring says so, so that a
failure points at a corpus situation rather than at an abstract rule.

Several tests come in **pairs**: the same setup with one thing changed, where the
first says what happens and the second proves that the *rule* is what made it
happen rather than an accident of the input. Those are the tests worth reading --
and writing them is how two of the ones below were caught asserting the right
answer for the wrong reason.
"""

from __future__ import annotations

from answering.candidates import Candidate, candidates_from
from answering.citations import cited_paths
from verity.engine.models import Chunk, MediaType
from verity.engine.retrieval import RetrievedChunk

#: A question describing an incident without naming the ticket, which is the
#: shape that makes a bridging file necessary. Deliberately *not* the sample
#: question verbatim: the rule has to hold on phrasing nobody has seen.
INCIDENT_QUESTION = (
    "The production log records a thermal incident. "
    "Which firmware release fixed the underlying defect?"
)


def corpus(*rows: tuple[str, str]) -> list[Chunk]:
    """A whole corpus, as ``(rel_path, text)`` pairs."""
    return [
        Chunk(rel_path=path, media_type=MediaType.PDF, location=f"page {n}", ordinal=n, text=text)
        for n, (path, text) in enumerate(rows, start=1)
    ]


def winner_in(path: str, text: str, value: str) -> Candidate:
    """The candidate for ``value``, built the way the pipeline builds it."""
    chunk = Chunk(rel_path=path, media_type=MediaType.PDF, location="page 1", ordinal=0, text=text)
    results = [RetrievedChunk(chunk=chunk, score=1.0, rank=1)]
    return next(c for c in candidates_from(results, frozenset()) if c.value == value)


# --- the single-document case: the winner's file, and nothing else -----------


def test_a_single_document_answer_cites_only_the_file_it_came_from() -> None:
    """The common case, and it needs no search at all.

    Seven of the ten sample questions are this. The value was read out of this
    file, so removing the file removes the value -- and removing any *other* file
    leaves a value that is still sitting here.
    """
    chunks = corpus(
        ("specs/r2.pdf", "Maximum junction temperature: 94 C"),
        ("planning/roadmap.docx", "The TQ-40 is in volume production."),
    )
    winner = winner_in("specs/r2.pdf", "Maximum junction temperature: 94 C", "94")

    assert cited_paths(winner, "What is the maximum junction temperature?", chunks) == ("specs/r2.pdf",)


def test_a_value_that_appears_in_two_files_cites_only_the_one_it_came_from() -> None:
    """Necessity, not presence. A value in two files does not need both.

    Removing the other file leaves the value exactly where it was, so the other
    file is not load-bearing and citing it would lose the question outright --
    an extra citation scores what a missing one scores.
    """
    chunks = corpus(
        ("specs/r2.pdf", "Maximum junction temperature: 94 C"),
        ("specs/notes.pdf", "The revised junction temperature is 94 C."),
    )
    winner = winner_in("specs/r2.pdf", "Maximum junction temperature: 94 C", "94")

    assert cited_paths(winner, "What is the maximum junction temperature?", chunks) == ("specs/r2.pdf",)


# --- the chain: a key the question does not supply ---------------------------


def test_a_key_supplied_by_another_file_makes_that_file_necessary() -> None:
    """The two-hop question, and the reason this module exists.

    The value sits in a database row. The question names neither the ticket nor
    the release; what the question describes is an *incident*, and only the log
    says which ticket it was logged against. Without the log the row cannot be
    identified, so the log is load-bearing even though it holds no part of the
    answer.
    """
    db = "ticket=ORR-1847 | status=closed | fixed_in=4.3.2"
    chunks = corpus(
        ("logs/prod.log", "INFO incident logged against ORR-1847"),
        ("support/bugs.csv", db),
    )
    winner = winner_in("support/bugs.csv", db, "4.3.2")

    assert cited_paths(winner, INCIDENT_QUESTION, chunks) == ("logs/prod.log", "support/bugs.csv")


def test_a_key_the_question_already_supplies_bridges_nothing() -> None:
    """And this is the pair to the test above: only the question changed.

    Here the question names the ticket itself, so the log adds nothing the
    question did not already have -- the row is identified without it, and citing
    it would be an extra citation on a question answerable from one file. Both
    halves are real sample questions, and they differ by one noun.
    """
    db = "ticket=ORR-1847 | status=closed | fixed_in=4.3.2"
    chunks = corpus(
        ("logs/prod.log", "INFO incident logged against ORR-1847"),
        ("support/bugs.csv", db),
    )
    winner = winner_in("support/bugs.csv", db, "4.3.2")

    assert cited_paths(winner, "Which firmware version fixed ticket ORR-1847?", chunks) == (
        "support/bugs.csv",
    )


def test_the_bridge_is_found_even_when_that_file_was_never_retrieved() -> None:
    """Necessity is a fact about the corpus, not about what retrieval fetched.

    Measured on the sample set, this is not hypothetical: the log line naming the
    ticket is *not* among the ten chunks the two-hop question retrieves. A rule
    that searched only the retrieved chunks would name one file of the two that
    are necessary -- so the search runs over the whole corpus, which is what the
    ``chunks`` argument is.
    """
    db = "ticket=ORR-1847 | status=closed | fixed_in=4.3.2"
    winner = winner_in("support/bugs.csv", db, "4.3.2")  # built from a one-chunk result list

    full = corpus(("logs/prod.log", "INFO incident logged against ORR-1847"), ("support/bugs.csv", db))

    assert cited_paths(winner, INCIDENT_QUESTION, full) == ("logs/prod.log", "support/bugs.csv")


def test_a_file_that_merely_names_the_key_is_not_necessary() -> None:
    """Naming a ticket is not connecting it to anything.

    A technical corpus names the same ticket in several places. A parts list that
    carries the number in a field tells the reader nothing about which bug-database
    row the question means, and removing it leaves the answer exactly as
    reachable -- so it is not necessary, and citing it would lose the question.

    This case was found by a test asserting the *opposite* under the first
    version of the rule, which cited any file containing the key.
    """
    db = "ticket=ORR-1847 | status=closed | fixed_in=4.3.2"
    chunks = corpus(
        ("logs/prod.log", "INFO incident logged against ORR-1847"),
        ("support/parts.csv", "part_number=ORR-1847 | lead_time_days=21"),
        ("support/bugs.csv", db),
    )
    winner = winner_in("support/bugs.csv", db, "4.3.2")

    assert cited_paths(winner, INCIDENT_QUESTION, chunks) == ("logs/prod.log", "support/bugs.csv")


# --- what may act as a key ---------------------------------------------------


def test_a_bare_number_is_not_a_key() -> None:
    """A number names no particular thing, so nothing can be looked up by it.

    The other file here *is* about the question, so the only reason it escapes a
    citation is the shape of the value the two share. Measured on the sample
    corpus, this restriction is load-bearing rather than tidy: the log line
    holding the error code also says "clocks reduced to 60%", and ``60`` appears
    in a comment in the ingest service and again in the roadmap. Letting numbers
    bridge would cite three files for a question answerable from one.
    """
    log = "ERROR E7731: thermal throttle engaged, throttled to 60%"
    chunks = corpus(
        ("logs/prod.log", log),
        ("engineering/ingest.py", "# Thermal throttle limits were raised to 60."),
    )
    winner = winner_in("logs/prod.log", log, "E7731")

    assert cited_paths(winner, "What error code is logged when the thermal throttle engages?", chunks) == (
        "logs/prod.log",
    )


def test_an_identifier_in_the_same_place_does_bridge() -> None:
    """The pair to the test above, with only the shared value's *shape* changed.

    Same log, same other file, same question -- the number becomes an identifier
    and the citation appears. Together the two say that it is the shape
    restriction doing the work, and not something else about the example.
    """
    log = "ERROR E7731: thermal throttle engaged on node ORR-2214"
    chunks = corpus(
        ("logs/prod.log", log),
        ("engineering/ingest.py", "# Thermal throttle incidents are tracked under ORR-2214."),
    )
    winner = winner_in("logs/prod.log", log, "E7731")

    assert cited_paths(winner, "What error code is logged when the thermal throttle engages?", chunks) == (
        "engineering/ingest.py",
        "logs/prod.log",
    )


def test_a_version_is_not_a_key() -> None:
    """A version is a quantity, and the class of thing a question asks for.

    The release notes list versions, so a version-shaped key would bridge every
    release question into the notes. The row here carries no identifier at all,
    which isolates the version: the notes are about the question and hold the
    same version, and are still not cited.
    """
    db = "component=power | status=closed | fixed_in=4.3.2"
    chunks = corpus(
        ("support/bugs.csv", db),
        ("engineering/notes.txt", "firmware version 4.3.2 shipped for this release."),
    )
    winner = winner_in("support/bugs.csv", db, "4.3.2")

    assert cited_paths(winner, "Which firmware release closed the defect?", chunks) == (
        "support/bugs.csv",
    )


def test_the_value_is_not_its_own_key() -> None:
    """A value cannot vouch for itself, the same rule the candidate context uses.

    The winning value is a ticket identifier and appears verbatim in another file
    that is genuinely about the question. If a value counted as its own key, that
    file would be cited -- and it should not be, because removing it leaves the
    value sitting where it was.
    """
    db = "ticket=ORR-1847 | status=closed | fixed_in=4.3.2"
    chunks = corpus(
        ("support/bugs.csv", db),
        ("engineering/notes.txt", "Firmware release 4.3.2 closed the ticket ORR-1847."),
    )
    winner = winner_in("support/bugs.csv", db, "ORR-1847")

    assert cited_paths(winner, "Which ticket was closed by firmware release 4.3.2?", chunks) == (
        "support/bugs.csv",
    )


def test_a_shared_fragment_is_not_a_shared_key() -> None:
    """``ORR-1904`` and ``ORR-FAN-2214-B`` share ``orr`` and share nothing else.

    The other file is about the question, so the only thing keeping it out is
    that ``orr`` is not the key. Matching on any token of a key would match the
    ``orr`` of every ``ORR-FAN-*``, ``ORR-HSK-*`` and ``ORR-CBL-*`` part number
    in the corpus -- measured, that is most ticket questions.
    """
    db = "ticket=ORR-1904 | status=closed | fixed_in=4.3.3"
    chunks = corpus(
        ("support/bugs.csv", db),
        ("support/parts.csv", "part_number=ORR-FAN-2214-B | firmware release note for the defect"),
    )
    winner = winner_in("support/bugs.csv", db, "4.3.3")

    assert cited_paths(winner, "Which firmware release closed the defect?", chunks) == (
        "support/bugs.csv",
    )


def test_the_key_and_the_value_must_share_a_chunk() -> None:
    """A key taken from the whole *file* would chain far too much.

    The datasheet's title line names the model and its specification line holds
    the temperature, and no single chunk holds both. So the temperature question
    grows no keys and cites one file -- where a file-wide rule would carry the
    model number into other documents, all of which name it.
    """
    title = "Orrery Systems TQ-40 Accelerator"
    spec = "Maximum junction temperature: 94 C"
    other = "The TQ-40 maximum junction temperature is under review."
    chunks = corpus(
        ("specs/r2.pdf", title),
        ("specs/r2b.pdf", spec),
        ("planning/roadmap.docx", other),
    )
    winner = winner_in("specs/r2b.pdf", spec, "94")

    assert cited_paths(winner, "What is the maximum junction temperature?", chunks) == ("specs/r2b.pdf",)


def test_a_key_in_the_same_chunk_does_bridge() -> None:
    """The pair to the test above, with the model number moved onto the same line.

    Identical files and identical question, except that the model number now
    co-occurs with the value. Together the two say that co-occurrence is the
    rule, rather than the model number being ignored.
    """
    spec = "TQ-40 maximum junction temperature: 94 C"
    other = "The TQ-40 maximum junction temperature is under review."
    chunks = corpus(("specs/r2.pdf", spec), ("planning/roadmap.docx", other))
    winner = winner_in("specs/r2.pdf", spec, "94")

    assert cited_paths(winner, "What is the maximum junction temperature?", chunks) == (
        "planning/roadmap.docx",
        "specs/r2.pdf",
    )


# --- the shape of the answer --------------------------------------------------


def test_citations_are_sorted_and_deduplicated() -> None:
    """Sorted so the output file is stable, and deduplicated because it is a set.

    Determinism matters more than it looks: an answer whose citations reorder
    between runs is one that cannot be diffed when a question starts failing.
    Two chunks of the same file are one citation.
    """
    db = "ticket=ORR-1847 | fixed_in=4.3.2"
    chunks = corpus(
        ("logs/b.log", "firmware release flagged against ORR-1847"),
        ("logs/a.log", "firmware release noted for ORR-1847"),
        ("logs/a.log", "firmware release acknowledged for ORR-1847"),
        ("support/bugs.csv", db),
    )
    winner = winner_in("support/bugs.csv", db, "4.3.2")

    assert cited_paths(winner, INCIDENT_QUESTION, chunks) == (
        "logs/a.log",
        "logs/b.log",
        "support/bugs.csv",
    )
