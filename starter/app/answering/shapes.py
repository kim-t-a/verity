"""Answer shapes: what a question asks for, and what that looks like in text.

Every graded answer in this challenge is a **short literal that already exists
as text in the corpus** -- ``94``, ``Q3 FY27``, ``ORR-FAN-2214-B``, ``REV-C2``,
``4.3.2``. Nothing is composed, paraphrased or inferred. So the extraction
problem is not "write an answer"; it is "find the one span in the retrieved
chunks that is the value the question is asking for".

This module supplies the two halves of that, and they are deliberately the same
idea seen from both ends:

* :func:`question_shapes` reads a question and decides which *kinds* of value
  would answer it -- "which firmware version" wants a dotted version, "how many
  seconds" wants a number.
* :func:`value_spans` reads a chunk and finds every span in it that *is* one of
  those kinds of value.

A shape is therefore both a question's expectation and a pattern over text,
which is why the two live together: a shape defined in one place and matched in
another would be two things that could disagree about what a version looks like.

**Shapes rank, they never filter.** A wrong guess about what a question wants
must not make the answer unreachable, so an unmatched candidate is not
discarded -- it is scored as though no shape had been guessed at all. See
:mod:`answering.extract` for what that costs it.

**What is deliberately not here.** No free-text answers. A question whose answer
is a name or a phrase -- "what is the TQ-40's codename" -- has no shape this
module recognises, and nothing downstream will invent one. That is a real gap,
stated rather than hidden: it is bounded by the challenge's own rule that every
answer is a short literal, but a literal the patterns below do not describe
would be missed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from config import STOPWORDS
from tokenisation import joined_form, tokenise


class AnswerShape(Enum):
    """The kinds of short literal the corpus answers with.

    Ordered from most to least specific, and that order is used as a tie-break
    when a question votes for several: ``Q3 FY27`` is a kind of number-and-code
    arrangement, but a question asking for a *quarter* is asking for something
    narrower than one asking for a number, and the narrower reading is the one
    worth trying first.
    """

    QUARTER = "quarter"
    """``Q3 FY27`` -- a fiscal quarter, which is a compound and must survive whole."""

    VERSION = "version"
    """``4.3.2`` -- dotted numeric, at least two components."""

    IDENTIFIER = "identifier"
    """``ORR-FAN-2214-B``, ``REV-C2`` -- separator-joined, mixing letters and digits."""

    CODE = "code"
    """``E7731`` -- a letter prefix fused to digits, with no separator."""

    NUMBER = "number"
    """``94``, ``84.5`` -- a bare numeral, with any unit left as context."""


#: Which question words ask for which shape.
#:
#: A hint matches a **whole word, or a run of consecutive words**, and never a
#: fragment inside a word. That distinction is not pedantry: matched as a plain
#: substring against the whitespace-deleted question, ``identification`` votes
#: for a *quarter* (it ends in ``fy``) and ``rebuild`` votes for a *version* (it
#: contains ``build``). Both are ordinary words in a question about a datacentre,
#: and a spurious vote is not harmless -- it hands a shape advantage to a
#: candidate the question never asked about.
#:
#: Word runs are what keep the hint list readable as phrases. ``part number``,
#: ``part_number`` and ``partnumber`` all reduce to the words ``part`` ``number``
#: and therefore to the same key, so the hint is spelled once, the way a person
#: would say it, and all three spellings of the question reach it.
_SHAPE_HINTS: dict[AnswerShape, tuple[str, ...]] = {
    # Deliberately no `FY` hint, though a fiscal year is what a quarter literal
    # looks like. It was doing two jobs and only one of them was real: a
    # question that *names* a fiscal year ("What was the FY27 revenue?") is
    # asking for a number, and the question that wants a quarter says so --
    # "in which quarter", "which fiscal quarter". The signal that a question
    # wants a quarter is the word, not the abbreviation.
    AnswerShape.QUARTER: ("QUARTER", "FISCAL"),
    AnswerShape.VERSION: ("VERSION", "FIRMWARE", "RELEASE", "BUILD"),
    AnswerShape.IDENTIFIER: (
        "PARTNUMBER",
        "PARTNO",
        "SKU",
        "MODEL",
        "ASSET",
        "CONNECTOR",
        "CABLE",
        "REVISION",
        "BOARDREV",
    ),
    AnswerShape.CODE: ("ERRORCODE", "CODE", "TICKET", "PIN", "INCIDENT", "SERIAL"),
    AnswerShape.NUMBER: (
        "HOWMANY",
        "HOWMUCH",
        "MAXIMUM",
        "MINIMUM",
        "TEMPERATURE",
        "POWER",
        "VOLTAGE",
        "SECONDS",
        "TIMEOUT",
        "BANDWIDTH",
        "CAPACITY",
        "COST",
        "PRICE",
        "RATE",
        "COUNT",
        "SIZE",
        "LATENCY",
        "WIDTH",
        "HEIGHT",
    ),
}

#: The value patterns, one per shape.
#:
#: Ordered so that the more specific patterns are tried first, though
#: :func:`value_spans` resolves overlaps by length rather than by this order --
#: see its docstring.
_PATTERNS: tuple[tuple[AnswerShape, re.Pattern[str]], ...] = (
    # A fiscal quarter, kept whole. `Q3 FY27` is the answer, not `Q3`: the
    # qualifier is part of the value, which the contract states explicitly.
    (AnswerShape.QUARTER, re.compile(r"\bQ[1-4]\s*FY\s*\d{2,4}\b", re.IGNORECASE)),
    # Dotted numeric with at least one dot. The trailing guard rejects `4.2x`
    # -- the log's latency multiplier -- which is a ratio, not a version, and
    # would otherwise be extracted as `4.2` and compete with the real answer.
    #
    # The guard is two lookaheads rather than one character class, and that is
    # the difference between finding "Version 4.3.2." and finding nothing at
    # all: a plain `(?![\w.])` forbids *any* following full stop, so a version
    # written at the end of a sentence -- which is how release notes write it --
    # matched no pattern in this module. Forbidding a following word character,
    # and a full stop only when a digit follows it, keeps the `4.2x` rejection
    # and admits the sentence.
    (AnswerShape.VERSION, re.compile(r"(?<![\w.])\d+\.\d+(?:\.\d+)*(?!\w)(?!\.\d)")),
    # Separator-joined runs: `ORR-FAN-2214-B`, `REV-C2`, `TQ-40`, `12V-2x6`.
    #
    # Two required ingredients, and both are load-bearing:
    #
    # * **a letter**, which excludes bare dates such as `2026-09-02` -- a date is
    #   not an identifier, and admitting them would put a timestamp on the
    #   candidate list of every log question;
    # * **a digit**, which excludes hyphenated English. Without it every
    #   `field-replaceable` and `follow-on` in the corpus is offered as an
    #   identifier, and one of them duly won a question it had no business
    #   competing in. This is also the rule the tokeniser already applies: a run
    #   only gets a joined, separator-free spelling when it contains a digit, so
    #   a digit-free run is not an identifier anywhere else in the system either.
    (
        AnswerShape.IDENTIFIER,
        re.compile(
            r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
            r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+\b"
        ),
    ),
    # A letter prefix fused to digits: `E7731`. Two digits minimum, so that a
    # one-character severity grade (`S2`) does not masquerade as an error code.
    (AnswerShape.CODE, re.compile(r"\b[A-Z]{1,4}\d{2,6}\b")),
    # A bare numeral, optionally with a fraction. The lookarounds keep it from
    # matching a fragment of a longer token: without them `4.3.2` yields `4.3`,
    # `3` and `2`, and every version in the corpus becomes three candidates.
    #
    # The trailing guard is the version pattern's, for the same reason: `is 94.`
    # ends a sentence, and a guard that forbade every following full stop made
    # the commonest way to write a number in prose unmatchable.
    (AnswerShape.NUMBER, re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)(?!\.\d)")),
)


@dataclass(frozen=True)
class ValueSpan:
    """One candidate value found in a chunk's text.

    ``text`` is the value **exactly as the corpus spells it** -- never
    normalised, never stripped. Normalisation is a comparison form, and the
    value that goes into the answer file has to be the one a human reading the
    document would see. ``94`` and ``94 C`` normalise alike, so normalising here
    would discard a distinction the grader accepts either way and gain nothing.
    """

    text: str
    start: int
    """Offset into the chunk text where the value begins."""

    shape: AnswerShape

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def question_shapes(query: str) -> tuple[AnswerShape, ...]:
    """The shapes that would answer ``query``, best guess first.

    Returns an empty tuple when the question offers no hint at all, which is a
    legitimate outcome rather than a failure: it means no candidate gets a shape
    advantage and the choice falls to how well each one's surroundings account
    for the question.

    The vote is a plain count of how many of a shape's hints the question
    contains, so ``which firmware version`` votes twice for ``VERSION`` (once for
    each hint) and once for ``CODE`` (``ticket``), and the version reading wins.
    Counting rather than first-match is what lets a question name two things and
    still express which it wants the value of.
    """
    keys = _hint_keys(query)

    votes: list[tuple[int, int, AnswerShape]] = []
    for order, (shape, hints) in enumerate(_SHAPE_HINTS.items()):
        hits = sum(1 for hint in hints if hint in keys)
        if hits:
            votes.append((hits, -order, shape))

    votes.sort(reverse=True)
    return tuple(shape for _hits, _order, shape in votes)


#: A word of the question. The same definition of a word the tokeniser uses --
#: an underscore is a separator, not part of a word -- so ``part_number`` gives
#: the same two words here as it does everywhere else in the system.
_WORD = re.compile(r"[^\W_]+")

#: How many consecutive words a hint may span. Must cover the longest hint in
#: :data:`_SHAPE_HINTS`, measured in words rather than in characters: ``PARTNO``
#: and ``ERRORCODE`` each span two. Three is one more than anything currently
#: needs, so adding a two-word hint cannot silently stop matching.
_MAX_HINT_WORDS = 3


def _hint_keys(query: str) -> frozenset[str]:
    """Every word, and every run of adjacent words, of ``query``, as hint keys.

    Built by concatenating up to :data:`_MAX_HINT_WORDS` consecutive words, so
    a hint written as one word (``PARTNUMBER``) matches a question that wrote it
    as two (``part number``) and one that wrote it as one (``partnumber``).

    Concatenation rather than normalisation because normalisation deletes *all*
    whitespace, which merges every word of the question into a single string and
    makes ``build`` match inside ``rebuild``. Keeping the word boundaries is what
    lets a hint be checked against whole words while still tolerating the three
    ways of spelling a two-word phrase.
    """
    words = _WORD.findall(query.lower())

    return frozenset(
        "".join(words[start : start + size]).upper()
        for size in range(1, _MAX_HINT_WORDS + 1)
        for start in range(len(words) - size + 1)
    )


def value_spans(text: str) -> tuple[ValueSpan, ...]:
    """Every candidate value in ``text``, in the order they appear.

    Overlaps are resolved by **longest span wins**, and that is not a
    preference between shapes but a statement about the text: inside
    ``ORR-FAN-2214-B`` the substring ``2214`` is a number only in the sense that
    it is made of digits, and offering it as a candidate for a question about
    numbers would be offering a fragment of an identifier. Where two spans
    cover the same characters the longer one is the one a reader would call the
    value. Ties in length cannot occur at the same offset, and spans at
    different offsets are both kept.
    """
    found: list[ValueSpan] = []
    for shape, pattern in _PATTERNS:
        found.extend(
            ValueSpan(text=match.group(0), start=match.start(), shape=shape)
            for match in pattern.finditer(text)
        )

    # Longest first, then leftmost, so the survivor of an overlap is fixed by
    # the text and not by the order the patterns happen to be declared in.
    found.sort(key=lambda span: (-len(span.text), span.start))

    kept: list[ValueSpan] = []
    for span in found:
        if any(span.start < other.end and other.start < span.end for other in kept):
            continue
        kept.append(span)

    kept.sort(key=lambda span: span.start)
    return tuple(kept)


def content_terms(text: str) -> tuple[str, ...]:
    """The words of ``text`` that carry discriminating power, in first-seen order.

    Tokens that are stopwords or bare numbers are dropped. Numbers because a
    question's ``TQ-40`` contributes ``40`` and ``tq40``, and counting the bare
    ``40`` as a separate piece of evidence would let the model number alone
    match a datasheet row that mentions any forty of anything.
    """
    return tuple(
        token
        for token in dict.fromkeys(tokenise(text))
        if token not in STOPWORDS and not token.isdigit()
    )


def term_forms(text: str) -> frozenset[str]:
    """Every spelling of every term in ``text``, in the index's own joined form.

    Joined rather than raw, so that ``ORR-1847`` in a question, ``ORR-1847`` in a
    log and ``orr_1847`` in a spreadsheet are one term -- and so that the shared
    ``orr`` of every ``ORR-*`` part number is a different one.

    Deliberately *not* the same thing as :func:`content_terms`: this keeps
    stopwords and numbers, because it answers "does this text contain this
    string at all", which is a question about presence rather than about
    evidence. A field named ``fixed_in`` has to be able to match a question that
    says "fixed", and a field named ``part_number`` a question that says "part
    number" -- neither of which is a content-word judgement.
    """
    return frozenset(joined_form(token) for token in tokenise(text))
