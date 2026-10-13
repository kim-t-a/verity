"""Supersession: which revision of a document is the live one.

Almost everything here is about **one sentence**. ``tq40_datasheet_r2.pdf``
contains the words "Revision 1 is withdrawn" -- a true statement, and a statement
*about r1*, sitting inside the document that replaced it. A substring scan reads
that as evidence r2 is withdrawn and promotes the revision that actually is,
which inverts the answer to the most trap-laden question in the sample set.

So the tests below are weighted towards that inversion: the successor-claim rule
gets four cases to itself, because getting it backwards is a confident wrong
answer rather than a refusal.
"""

from __future__ import annotations

from answering.supersession import is_superseded, superseded_paths
from models import Chunk, MediaType


def chunk(rel_path: str, text: str, location: str = "page 1") -> Chunk:
    return Chunk(
        rel_path=rel_path,
        media_type=MediaType.PDF,
        location=location,
        ordinal=0,
        text=text,
    )


# --- the path marker --------------------------------------------------------


def test_a_path_that_says_withdrawn_is_superseded() -> None:
    assert is_superseded("specs/tq40_datasheet_r1_WITHDRAWN.pdf", ["105 C"]) is True


def test_a_live_document_is_not_superseded() -> None:
    assert is_superseded("specs/tq40_datasheet_r2.pdf", ["94 C"]) is False


# --- the trap ---------------------------------------------------------------


def test_a_successor_claim_defuses_the_statement_about_the_revision_it_replaced() -> None:
    """The whole reason this module exists, in one assertion.

    r2 says both "supersedes revision 1" and "Revision 1 is withdrawn". The
    first is a claim about r2 itself; the second is a claim about r1. Both
    contain the substring a naive scan looks for, and only the precedence rule
    tells them apart.
    """
    r2_texts = [
        "Maximum junction temperature 94 C.",
        "This datasheet supersedes revision 1. Revision 1 is withdrawn.",
    ]
    assert is_superseded("specs/tq40_datasheet_r2.pdf", r2_texts) is False


def test_the_passive_form_is_not_mistaken_for_a_successor_claim() -> None:
    """``SUPERSEDED BY`` must not match ``SUPERSEDES``.

    They differ by one character in the middle, they mean opposite things, and
    a prefix check or a fuzzy contains would conflate them -- promoting the one
    document that is explicitly marked as replaced.
    """
    assert is_superseded("specs/old.pdf", ["This document is superseded by revision 2."]) is True


def test_a_successor_claim_beats_a_content_marker_in_the_same_file() -> None:
    """A file that both replaces something and says "do not use" is still live.

    Contrived on its own, and load-bearing in combination: it pins the
    precedence as a rule about the *text*, not as an accident of which check
    happens to be written first for the one document that exercises it.
    """
    texts = ["Revision 1 is obsolete.", "This release supersedes revision 1."]
    assert is_superseded("specs/r2.pdf", texts) is False


def test_a_successor_claim_on_a_later_page_is_still_found() -> None:
    """Revision history lives on its own page, not beside the numbers.

    All the document's chunks are consulted rather than just the one holding the
    value, which is what makes the rule work when the claim and the answer are
    in different chunks.
    """
    texts = ["Maximum junction temperature 94 C.", "Revision history: supersedes revision 1."]
    assert is_superseded("specs/r2.pdf", texts) is False


# --- the content markers ----------------------------------------------------


def test_a_file_naming_itself_obsolete_is_superseded() -> None:
    assert is_superseded("specs/plan.pdf", ["This document is obsolete."]) is True


def test_a_file_with_no_marker_anywhere_is_live() -> None:
    assert is_superseded("specs/plan.pdf", ["Nothing here says anything about revisions."]) is False


def test_a_path_marker_outranks_a_successor_claim_in_the_content() -> None:
    """Path first, and that order is a decision rather than an accident.

    A withdrawn revision very plausibly says it superseded the revision *before
    it* -- r1 replacing r0 is an ordinary sentence -- so a content claim cannot
    be allowed to un-retire a document whose own filename says it was retired.
    The rename is the deliberate act; a stale history line is not.
    """
    assert is_superseded("specs/r1_WITHDRAWN.pdf", ["This revision supersedes revision 0."]) is True


def test_markers_are_matched_through_the_shared_normalisation() -> None:
    """Punctuation and case must not decide whether a document is retired.

    ``is  withdrawn.`` and ``IS-WITHDRAWN`` have to reach the same verdict, or a
    corpus that spells the marker differently evades the rule entirely.
    """
    assert is_superseded("specs/a.pdf", ["This revision is  withdrawn."]) is True


# --- over the corpus --------------------------------------------------------


def test_the_verdict_is_taken_over_the_whole_document_not_one_chunk() -> None:
    """A document is superseded if *any* of its chunks says so."""
    chunks = [
        chunk("specs/r1_WITHDRAWN.pdf", "Maximum junction temperature 105 C.", "page 1"),
        chunk("specs/r1_WITHDRAWN.pdf", "Revision history.", "page 4"),
    ]
    assert superseded_paths(chunks) == frozenset({"specs/r1_WITHDRAWN.pdf"})


def test_the_set_contains_only_the_documents_that_are_superseded() -> None:
    chunks = [
        chunk("specs/r1_WITHDRAWN.pdf", "105 C"),
        chunk("specs/r2.pdf", "94 C"),
        chunk("specs/note.txt", "This document is obsolete."),
    ]
    assert superseded_paths(chunks) == frozenset(
        {"specs/r1_WITHDRAWN.pdf", "specs/note.txt"}
    )


def test_an_empty_corpus_yields_an_empty_set() -> None:
    assert superseded_paths([]) == frozenset()
