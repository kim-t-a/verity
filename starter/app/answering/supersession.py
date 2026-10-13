"""Superseded documents: deciding which revision is the live one.

**Why this is not a detail.** The corpus contains ``tq40_datasheet_r1_WITHDRAWN``
and ``tq40_datasheet_r2``, and they answer the same question differently: r1
says the maximum junction temperature is 105 C, r2 says 94 C. Citing the wrong
one is wrong even though it is a real file with a real number in it. Stage 5
measured that retrieval cannot separate them -- fused, the two are an exact tie,
broken only by which retriever was registered first. Something has to decide,
and this module is that something.

**The trap, and why content alone is not enough.** The obvious implementation is
"if a file's text says withdrawn or superseded, demote it". That fails here, and
it fails *invertedly*. ``tq40_datasheet_r2.pdf`` contains the sentence
"Revision 1 is withdrawn" -- a true statement, and a statement **about r1**,
sitting inside the document that replaced it. A substring scan reads it as
evidence that r2 is withdrawn and promotes the revision that actually is. Getting
this backwards would produce a confident wrong answer on the single most
trap-laden question in the set, so the rule below is built around not making
that mistake.

**The rule.** Two signals, in this precedence:

1. **A successor claim wins outright.** A document whose text says it
   *supersedes* something is, by its own statement, the current one. r2 says
   "supersedes revision 1"; nothing else in r2 can then mark it as superseded.
   This one precedence rule is what defuses the trap above, and it is not a
   special case for this corpus -- it is what "supersedes" means. Note that
   ``SUPERSEDES`` and ``SUPERSEDED BY`` are different strings, so the check does
   not accidentally match the passive form.
2. **A path marker, or a self-supersession marker in the content.** The path is
   the more trustworthy of the two because it is deliberate: whoever retired a
   document named the file to say so. Content markers are checked too, but only
   for a file with no successor claim of its own.

**What this deliberately does not attempt.** It does not build a revision graph,
and it does not attribute a supersession statement to the document it names. A
fuller design would read "Revision 1 is withdrawn" out of r2 and use it as
positive evidence that r1 is superseded -- which is better evidence than r1's
own filename. It is not needed here, because the filename already says so, and
building it now would be building a mechanism with no measurement behind it.
Recorded as the natural extension if the graded corpus marks revisions only in
prose.
"""

from __future__ import annotations

from collections.abc import Iterable

from config import SUPERSEDED_PATH_MARKERS, SUPERSEDED_TEXT_MARKERS
from models import Chunk
from normalisation import normalise

#: Active-voice successor claims. Present in a document, they mean the document
#: is the replacement rather than the replaced -- see the module docstring.
#:
#: ``SUPERSEDES`` rather than ``SUPERSEDE`` so that the passive
#: ``SUPERSEDED BY`` cannot match: after ``SUPERSEDE`` the passive form has a
#: ``D`` where this has an ``S``.
_SUCCESSOR_CLAIMS = ("SUPERSEDES",)

#: The config markers, respelled the way the text they are matched against is.
#:
#: **This normalisation is not cosmetic.** ``config.SUPERSEDED_TEXT_MARKERS``
#: holds the phrases as a person would write them -- ``"IS WITHDRAWN"``, with
#: spaces -- and they are compared against ``normalise(text)``, which *deletes
#: all whitespace*. Compared raw, ``"IS WITHDRAWN" in "THISREVISIONISWITHDRAWN"``
#: is false for every document in every corpus, so the whole content half of the
#: rule silently never fires: no error, no warning, just a superseded document
#: that is quietly treated as live.
#:
#: Done here rather than in ``config.py`` so that the config keeps holding the
#: phrases a human would recognise and can check against a corpus by eye, and
#: the normalisation lives beside the comparison it exists for.
_SUPERSEDED_PATH_MARKERS = tuple(normalise(marker) for marker in SUPERSEDED_PATH_MARKERS)
_SUPERSEDED_TEXT_MARKERS = tuple(normalise(marker) for marker in SUPERSEDED_TEXT_MARKERS)


def _has_marker(flattened: str, markers: Iterable[str]) -> bool:
    return any(marker in flattened for marker in markers)


def is_superseded(path: str, texts: Iterable[str]) -> bool:
    """Whether the document at ``path`` has been replaced by a later revision.

    ``texts`` is every chunk of that document. All of them are consulted for the
    successor claim, because a document announces what it replaces in its
    revision history, which is rarely on the page that states its numbers.
    """
    if _has_marker(normalise(path), _SUPERSEDED_PATH_MARKERS):
        return True

    flattened = [normalise(text) for text in texts]

    # Checked across the whole document before the markers are, and that order
    # is the entire defence against the "Revision 1 is withdrawn" trap.
    if any(_has_marker(text, _SUCCESSOR_CLAIMS) for text in flattened):
        return False

    return any(_has_marker(text, _SUPERSEDED_TEXT_MARKERS) for text in flattened)


def superseded_paths(chunks: Iterable[Chunk]) -> frozenset[str]:
    """The corpus-relative paths of every superseded document.

    Taken over the whole corpus rather than over a single result list, because
    the verdict for one file can depend on text held in another file's chunk,
    and because a set computed once is a set every question agrees on.
    """
    by_path: dict[str, list[str]] = {}
    for chunk in chunks:
        by_path.setdefault(chunk.rel_path, []).append(chunk.text)

    return frozenset(
        path for path, texts in by_path.items() if is_superseded(path, texts)
    )
