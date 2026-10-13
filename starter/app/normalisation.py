"""The one normalisation used everywhere in this system.

The contract grades an answer by its *value*, not its spelling, and the corpus
is full of spellings that differ without differing in meaning -- ``94 °C``
against ``94C``, ``ORR-FAN-2214-B`` against ``ORR FAN 2214 B``, ``REV-C2``
against ``rev c2``. Every comparison in this project -- matching an answer
against extracted text, checking whether a value was seen, deciding whether a
document is superseded -- therefore has to be made on a common form.

**There is exactly one implementation of that form, and it is this one.**
``AGENTS.md`` states the rule: one normalisation function, used everywhere.
Two implementations that drift apart produce a system that matches in one code
path and misses in another, which is close to impossible to debug from the
outside because both paths look correct in isolation.

The transformation, from the contract: uppercase, strip whitespace, delete
``-`` ``.`` ``·`` ``_``, and ignore units and degree signs.
"""

from __future__ import annotations

import unicodedata

#: Characters deleted outright: the four separators the contract names.
#:
#: Deleting rather than replacing them is what makes ``4.3.2`` and ``4-3-2``
#: and ``432`` the same string. That is intentional -- the contract says so --
#: and it is why normalisation must never be used to *produce* an answer. It is
#: a comparison form, not a display form: an answer reported to the grader is
#: always the value exactly as it appears in the corpus.
_SEPARATORS = frozenset("-._·")

#: Degree-like signs, deleted before anything else runs.
#:
#: ``°`` and ``∘`` are what a datasheet actually uses. ``º`` (masculine
#: ordinal) and ``˚`` (ring above) are here because they are visually
#: indistinguishable from a degree sign at a glance, are routinely typed in its
#: place, and PDF text extraction occasionally emits them for it.
#:
#: These are removed *before* NFKC rather than after, and that ordering is
#: load-bearing: NFKC folds ``º`` to the letter ``o``, which would leave a
#: stray letter in the middle of a value. Deleting first means there is nothing
#: left for NFKC to fold.
_DEGREES = frozenset("°º˚∘")

_DEGREE_TABLE = {ord(character): None for character in _DEGREES}
_SEPARATOR_TABLE = {ord(character): None for character in _SEPARATORS}


def normalise(value: str) -> str:
    """Reduce ``value`` to the comparison form defined by the contract.

    The steps run in a deliberate order:

    1. **Delete degree signs.** Before NFKC, so they vanish rather than folding
       into letters and digits (see ``_DEGREES``).
    2. **NFKC-fold.** Compatibility normalisation collapses the many Unicode
       spellings of one character onto the one a human would type: full-width
       digits and punctuation, the ligature ``ﬁ``, superscripts, and the
       non-breaking space. This matters here because PDF extraction emits all
       of them, so a value can differ from its query by encoding alone.
    3. **Delete separators.** After NFKC, so that a full-width ``．`` -- which
       NFKC has by now turned into ``.`` -- is removed along with the ASCII
       form. Deleting before NFKC would leave that one behind.
    4. **Uppercase.** After NFKC, so a full-width ``ａ`` is folded to ``a`` and
       then raised, rather than being upper-cased and only then folded.
    5. **Delete whitespace.** ``str.split`` with no argument splits on every
       Unicode whitespace character, not just the ASCII ones, which catches the
       non-breaking and thin spaces that extraction produces. The words are
       then joined with nothing between them.

    Units need no special handling here. An expected answer of ``94`` must match
    a corpus line reading ``Maximum junction temperature: 94 °C``, and it does:
    the line normalises to ``MAXIMUMJUNCTIONTEMPERATURE:94C``, which contains
    ``94``. Substring matching against this form is what makes units and
    surrounding prose irrelevant, so there is nothing to strip.

    Args:
        value: Any string -- a corpus line, a query, an expected answer.

    Returns:
        The comparison form. Never ``None``; may be empty when ``value`` held
        nothing but separators and whitespace.
    """
    without_degrees = value.translate(_DEGREE_TABLE)
    folded = unicodedata.normalize("NFKC", without_degrees)
    without_separators = folded.translate(_SEPARATOR_TABLE)
    return "".join(without_separators.upper().split())


def contains(haystack: str, needle: str) -> bool:
    """Whether ``needle`` appears in ``haystack`` under normalisation.

    The one comparison this project makes. Both sides go through
    :func:`normalise`, so the caller never has to remember to do it -- which is
    the failure this helper exists to prevent, since a comparison that
    normalises one side but not the other looks almost right.

    An empty ``needle`` is never a match. Normalisation can reduce a
    punctuation-only string to nothing, and an empty needle would then be
    "found" in every haystack, silently reporting a value that was never there.
    """
    target = normalise(needle)
    return bool(target) and target in normalise(haystack)
