"""Deciding which recognised text belongs with which, from position alone.

An OCR engine answers the easy question -- *what does the image say* -- and
leaves the hard one alone: which of those pieces of text are related. On a
datasheet page that matters, because the answer to a question can be a
relationship rather than a word: the signal carried by a particular pin is
stated by a diagram, not by a sentence, and reading the label without reading
the box above it produces a name attached to nothing.

This module recovers that relationship geometrically, and only geometrically.
It never opens a file, never calls an engine and never looks at a pixel; it
takes the detections an adapter already produced and works out what sits under
what. That is a deliberate split. The part that can be subtly *wrong* -- the
association rule, with its tolerances and its refusals -- is then a pure
function of numbers, testable with invented boxes and no image at all, exactly
as the header parsers in :mod:`ingestion.readers.image` are tested with invented
bytes. The engine becomes one narrow seam behind it, and swapping Tesseract for
something else later cannot change the reasoning.

**Never invent.** Every association's text is two recognised lines joined, and
nothing else: no word enters the index that the engine did not report. A
sentence-shaped summary would read better and would be a fabrication.

**Refuse rather than guess.** An ambiguous diagram yields *no* association.
This is the rule the whole module is built around. A wrong pairing is not a
missing answer that might still be recovered elsewhere -- it is a confident
statement that a particular pin carries a particular signal, cited as evidence,
and nothing downstream can tell it apart from a pairing the image really shows.
Where the geometry cannot separate two candidates, saying nothing is the only
honest output, and :class:`Rejection` records that it happened so the silence is
visible rather than looking like an empty image.

**Ordinary text is kept either way.** Lines are returned whether or not they
associate with anything. A flat label has no relationships in it at all and is
still perfectly readable text; withholding it because no diagram was found would
throw away the answer along with the structure.

**Where this attaches.** :func:`layout_from` is called from the vision seam in
:mod:`ingestion.readers.image`, with ``image_width`` taken from the header the
reader already probed. Its output becomes ``TextSpan``s there -- one per line,
one per association, each carrying the ``location`` below -- and then enters
``chunking`` and the index unchanged. Nothing in this module knows about that
schema, so the conversion is a handful of lines at the seam and this stays
testable without it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from verity.engine.config import (
    ASSOCIATION_AMBIGUITY_FACTOR,
    ASSOCIATION_MAX_CENTRE_OFFSET_FRACTION,
    ASSOCIATION_MAX_GAP_FACTOR,
    ASSOCIATION_MIN_GAP_FACTOR,
    CELL_SEPARATOR,
    DUPLICATE_BOX_OVERLAP,
    LINE_MAX_HORIZONTAL_GAP_FACTOR,
    LINE_VERTICAL_OVERLAP,
    MAX_ASSOCIATION_WIDTH_FRACTION,
    MAX_IDENTIFIER_WORDS,
    MAX_LABEL_WORDS,
    MIN_OCR_CONFIDENCE,
)
from verity.engine.normalisation import normalise

#: ``(left, top, right, bottom)``. All the arithmetic below runs on this shape
#: rather than on the dataclasses, so a comparison never has to care whether it
#: was handed a detection or a line.
_Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One piece of text an engine found, and the box it found it in.

    The field names are the ones OCR engines use, so the adapter that fills this
    in is a direct transcription of their output rather than a translation of
    it.

    ``confidence`` is on a 0..1 scale, whatever the engine reports natively.
    Tesseract's TSV confidence column is a percentage and has to be divided
    before it arrives here.
    """

    text: str
    confidence: float
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def centre_x(self) -> float:
        return self.left + self.width / 2


@dataclass(frozen=True)
class TextLine:
    """One line of recognised text: what it says, and the box it occupies.

    Words reported separately are joined with a single space, so a line reads as
    it does on the page. The join is the only text this module ever composes,
    and both halves of it came off the image.
    """

    text: str
    confidence: float
    left: float
    top: float
    width: float
    height: float

    ordinal: int
    """Reading position within the image, counting from 1.

    Assigned after the lines are ordered top-to-bottom and left-to-right, so it
    is the line's place in the image rather than the order the engine happened
    to report its detections in. Unique within one layout, which is what makes
    it usable as the identity of a line when associations are deduplicated.
    """

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def centre_x(self) -> float:
        return self.left + self.width / 2

    @property
    def location(self) -> str:
        """Human-readable provenance, in the shape the other readers use.

        The coordinates are what makes this traceable: given the location, the
        box can be drawn back onto the image and the line checked against what
        the engine claimed. A location of ``"text line 4"`` alone would not
        distinguish a correct read from a misordered one.
        """
        return f"text line {self.ordinal} at ({self.left:.0f}, {self.top:.0f})"


@dataclass(frozen=True)
class Association:
    """Two lines that belong together, and the provenance of the pairing.

    The pairing is the finding. Neither line's text is new, but the fact that
    they are related is, and it exists nowhere in the image as text -- which is
    why the location has to name *both* source lines. A reader checking this
    association needs to see the two boxes it was inferred from.
    """

    upper: TextLine
    lower: TextLine
    text: str

    @property
    def location(self) -> str:
        left = min(self.upper.left, self.lower.left)
        top = min(self.upper.top, self.lower.top)
        right = max(self.upper.right, self.lower.right)
        bottom = max(self.upper.bottom, self.lower.bottom)
        return (
            f"text lines {self.upper.ordinal} and {self.lower.ordinal} "
            f"at region ({left:.0f}, {top:.0f})-({right:.0f}, {bottom:.0f})"
        )


class RejectionReason(StrEnum):
    """Why a plausible association was refused.

    Kept as a closed set so that a test can assert the *reason* rather than only
    the absence, and so a run's diagnostics can say whether an image was
    unreadable or merely ambiguous.
    """

    TOO_CLOSE_TO_CALL = "too_close_to_call"
    """Two candidates were near enough that choosing one would be a coin toss."""

    SHARED_LABEL = "shared_label"
    """Two identifiers claimed the same label, and it cannot belong to both."""

    CHAINED = "chained"
    """The label is itself another association's identifier.

    Only possible when three or more rows of short tokens stack up. Whether the
    middle token is a label for the one above or an identifier for the one below
    is not something the boxes can say, so the associations that would rest on
    answering it are dropped.
    """


@dataclass(frozen=True)
class Rejection:
    """One association that geometry could have made and deliberately did not.

    Recorded only when there was something plausible to refuse. A token with no
    candidate anywhere near it is not a rejection, it is simply text -- and
    recording those would bury the real refusals in noise.
    """

    reason: RejectionReason
    lines: tuple[TextLine, ...]
    detail: str


@dataclass(frozen=True)
class Layout:
    """Everything one image's detections yielded.

    ``lines`` is the whole recognised text of the image, in reading order, and
    is complete on its own -- an image with no diagram in it has lines and no
    associations, and is not a failed read.
    """

    lines: tuple[TextLine, ...] = ()
    associations: tuple[Association, ...] = ()
    rejections: tuple[Rejection, ...] = ()

    dropped_detections: int = 0
    """Detections that were not usable text: malformed boxes, no content, or
    confidence too low to trust. Distinct from the count below because the two
    are different findings -- one is engine noise, the other is engine
    verbosity."""

    duplicate_detections: int = 0
    """Detections dropped as repeats of another, which is ordinary: engines
    report overlapping boxes for the same word, and a double-counted word would
    appear twice in the indexed line."""


def _box_of(item: Detection | TextLine) -> _Box:
    return (item.left, item.top, item.right, item.bottom)


def _area(box: _Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _union(boxes: Iterable[_Box]) -> _Box:
    collected = list(boxes)
    return (
        min(box[0] for box in collected),
        min(box[1] for box in collected),
        max(box[2] for box in collected),
        max(box[3] for box in collected),
    )


def _overlap_ratio(small: _Box, other: _Box) -> float:
    """How much of the smaller box is covered by the other.

    Measured against the smaller box rather than the union so that a detection
    nested inside another still counts as the same text; see
    ``DUPLICATE_BOX_OVERLAP``.
    """
    width = min(small[2], other[2]) - max(small[0], other[0])
    height = min(small[3], other[3]) - max(small[1], other[1])
    if width <= 0 or height <= 0:
        return 0.0
    smallest = min(_area(small), _area(other))
    return (width * height) / smallest if smallest > 0 else 0.0


def _share_a_line(first: _Box, second: _Box) -> bool:
    """Whether two boxes belong to the same line of text.

    Both halves are needed, and the horizontal one is the load-bearing half.
    Every label on a diagram shares a baseline with the labels in the other
    columns, so grouping on vertical overlap alone would fuse a whole row into
    one long line -- and a fused line is too long to be a candidate, which would
    lose every pairing on the image at once rather than one of them.

    The vertical test is measured against the *shorter* box, so a word set in
    smaller type beside a larger one still joins it. Measured against the
    taller, a heading and the caption under it would split apart instead -- and
    the pieces of a split line are short enough to look like candidates.
    """
    overlap = min(first[3], second[3]) - max(first[1], second[1])
    if overlap <= 0:
        return False
    shortest = min(first[3] - first[1], second[3] - second[1])
    if overlap < LINE_VERTICAL_OVERLAP * shortest:
        return False

    gap = max(first[0], second[0]) - min(first[2], second[2])
    return gap <= LINE_MAX_HORIZONTAL_GAP_FACTOR * shortest


def _is_number(value: object) -> bool:
    """True for a finite real coordinate.

    ``bool`` is excluded deliberately: it is an ``int`` in Python, so a ``True``
    that reached a coordinate field through a dict lookup would otherwise be
    read as the number 1 and place a box at the origin.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_usable(detection: Detection) -> bool:
    """Whether a detection is text at all.

    Every rejection here has a counterpart in what images actually produce: a
    box with no extent is a detection the engine failed to fill in, a blank or
    punctuation-only string is a rule line or a stray mark, and low confidence
    is the barcode.
    """
    if not all(
        _is_number(value)
        for value in (
            detection.left,
            detection.top,
            detection.width,
            detection.height,
            detection.confidence,
        )
    ):
        return False
    if detection.width <= 0 or detection.height <= 0:
        return False
    if detection.confidence < MIN_OCR_CONFIDENCE:
        return False
    if not detection.text.strip():
        return False
    return any(character.isalnum() for character in detection.text)


def _without_duplicates(detections: Sequence[Detection]) -> list[Detection]:
    """Drop detections that repeat one already kept.

    Text is compared through :func:`normalisation.normalise`, the one comparison
    form this project uses, because engines differ on the spelling of the same
    word: ``THERM_ALERT#`` and ``THERM ALERT#`` are one string to a reader and
    two to a string comparison.

    Position is compared too, and that is the load-bearing half. The same text
    at the same place is a repeat; the same text at a different place is a real
    second occurrence, and a diagram with two identical labels -- two ground
    pins, say -- depends on keeping both.
    """
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: (-d.confidence, d.top, d.left)):
        target = normalise(detection.text)
        duplicate = any(
            normalise(other.text) == target
            and _overlap_ratio(_box_of(detection), _box_of(other)) >= DUPLICATE_BOX_OVERLAP
            for other in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def _lines_from(detections: Sequence[Detection]) -> tuple[TextLine, ...]:
    """Group detections into lines, then order them as a reader would.

    Works for either kind of engine output without being told which: two
    detections on the same line overlap vertically, and overlapping ones are
    joined left to right. Whether the engine reports words or whole lines
    therefore changes nothing here.

    Groups are merged when a detection turns out to belong to two of them at
    once, and that pass is not optional. A word arriving out of order can sit
    between two fragments of one line and be the only thing linking them: with
    the fragments already started as separate groups, the greedy pass alone
    leaves the line split, and a label split in two is a label the index no
    longer holds whole.
    """
    groups: list[list[Detection]] = []
    extents: list[_Box] = []

    for detection in sorted(detections, key=lambda d: (d.top, d.left)):
        box = _box_of(detection)
        matching = [
            index for index, extent in enumerate(extents) if _share_a_line(extent, box)
        ]

        if not matching:
            groups.append([detection])
            extents.append(box)
            continue

        # Everything this detection shares a line with is one line. Merging the
        # rest into the first, highest-index first so the earlier indices stay
        # valid as the lists are popped.
        merged = matching[0]
        groups[merged].append(detection)
        extents[merged] = _union((extents[merged], box))
        for index in reversed(matching[1:]):
            groups[merged].extend(groups.pop(index))
            extents[merged] = _union((extents[merged], extents.pop(index)))

    order = sorted(
        range(len(groups)), key=lambda index: (extents[index][1], extents[index][0])
    )
    return tuple(
        _line_from(groups[index], ordinal)
        for ordinal, index in enumerate(order, start=1)
    )


def _line_from(group: Sequence[Detection], ordinal: int) -> TextLine:
    ordered = sorted(group, key=lambda detection: detection.left)
    extent = _union(_box_of(detection) for detection in ordered)
    return TextLine(
        text=" ".join(detection.text.strip() for detection in ordered),
        # The weakest word sets the line's confidence: a line is only as
        # trustworthy as the part of it most likely to be wrong.
        confidence=min(detection.confidence for detection in ordered),
        left=extent[0],
        top=extent[1],
        width=extent[2] - extent[0],
        height=extent[3] - extent[1],
        ordinal=ordinal,
    )


def _is_short_text(line: TextLine, words: int, image_width: float | None) -> bool:
    """Whether a line is a short run of text rather than a heading or prose."""
    if len(line.text.split()) > words:
        return False
    if not any(character.isalpha() for character in line.text):
        return False
    if image_width is not None and image_width > 0:
        if line.width > MAX_ASSOCIATION_WIDTH_FRACTION * image_width:
            return False
    return True


def _is_identifier(line: TextLine, image_width: float | None) -> bool:
    """Whether a line could be the token an association hangs off.

    One word, which is the rule that keeps title blocks out of this: no line of
    a title block is a single token, and a title block is otherwise shaped
    almost exactly like a diagram.
    """
    return _is_short_text(line, MAX_IDENTIFIER_WORDS, image_width)


def _is_label(line: TextLine, image_width: float | None) -> bool:
    """Whether a line could be the token beneath an identifier.

    More permissive than :func:`_is_identifier`, because a signal name may run
    to a few words while a designator does not.
    """
    return _is_short_text(line, MAX_LABEL_WORDS, image_width)


@dataclass(frozen=True)
class _Proposal:
    """One identifier's best guess at its label, before conflicts are settled."""

    upper: TextLine
    lower: TextLine
    distance: float


def _proposal(upper: TextLine, lower: TextLine) -> _Proposal | None:
    """The pairing score for two lines, or ``None`` when they cannot pair.

    The score is the length of the leader line that would join them: straight
    down from the bottom of the identifier to the top of the label. Horizontal
    offset and vertical gap are both in it, so a label directly beneath wins
    over one further down, and a distant label that merely lines up column-wise
    does not win at all.
    """
    gap = lower.top - upper.bottom
    if gap < 0:
        return None

    unit = min(upper.height, lower.height)
    if unit <= 0:
        return None
    if not (ASSOCIATION_MIN_GAP_FACTOR * unit <= gap <= ASSOCIATION_MAX_GAP_FACTOR * unit):
        return None

    offset = abs(lower.centre_x - upper.centre_x)
    widest = max(upper.width, lower.width)
    if offset > ASSOCIATION_MAX_CENTRE_OFFSET_FRACTION * widest:
        return None

    return _Proposal(upper=upper, lower=lower, distance=math.hypot(offset, gap))


def _associate(
    lines: Sequence[TextLine], image_width: float | None
) -> tuple[tuple[Association, ...], tuple[Rejection, ...]]:
    """Pair identifiers with the labels beneath them, refusing the unclear ones."""
    identifiers = [line for line in lines if _is_identifier(line, image_width)]
    # A one-word line is both an identifier and a label, so the pool below is
    # the wider test and the line itself is excluded by identity, not by a
    # second rule.
    labels = [line for line in lines if _is_label(line, image_width)]

    proposals: list[_Proposal] = []
    rejections: list[Rejection] = []

    for upper in identifiers:
        ranked = sorted(
            (
                proposal
                for lower in labels
                if lower.ordinal != upper.ordinal
                for proposal in (_proposal(upper, lower),)
                if proposal is not None
            ),
            key=lambda proposal: (proposal.distance, proposal.lower.ordinal),
        )
        if not ranked:
            # Nothing plausible was seen, so there is nothing to refuse. A token
            # with no neighbour is ordinary text, and recording it would drown
            # the real refusals.
            continue

        best = ranked[0]
        if len(ranked) > 1:
            margin = ASSOCIATION_AMBIGUITY_FACTOR * min(best.upper.height, best.lower.height)
            if ranked[1].distance - best.distance <= margin:
                rejections.append(
                    Rejection(
                        reason=RejectionReason.TOO_CLOSE_TO_CALL,
                        lines=(best.upper, best.lower, ranked[1].lower),
                        detail=(
                            f"'{best.upper.text}' is {best.distance:.0f}px from "
                            f"'{best.lower.text}' and {ranked[1].distance:.0f}px from "
                            f"'{ranked[1].lower.text}', too close to choose between them"
                        ),
                    )
                )
                continue

        proposals.append(best)

    # Nearest pairing first, so a contested label goes to whichever identifier
    # it belongs to most strongly rather than to whichever was scanned first.
    proposals.sort(key=lambda proposal: (proposal.distance, proposal.upper.ordinal))

    associations: list[Association] = []
    used_as_upper: set[int] = set()
    used_as_lower: set[int] = set()

    for proposal in proposals:
        upper, lower = proposal.upper, proposal.lower
        if upper.ordinal in used_as_lower or lower.ordinal in used_as_upper:
            rejections.append(
                Rejection(
                    reason=RejectionReason.CHAINED,
                    lines=(upper, lower),
                    detail=(
                        f"'{upper.text}' and '{lower.text}' cannot pair: one of them "
                        "already sits on both sides of an association"
                    ),
                )
            )
            continue
        if lower.ordinal in used_as_lower:
            rejections.append(
                Rejection(
                    reason=RejectionReason.SHARED_LABEL,
                    lines=(upper, lower),
                    detail=(
                        f"'{lower.text}' already belongs to another identifier, "
                        f"so it cannot also belong to '{upper.text}'"
                    ),
                )
            )
            continue

        associations.append(
            Association(
                upper=upper,
                lower=lower,
                # Reading order, and the same key-then-value order every other
                # reader in this project renders a row in. Which side is the
                # question's subject is not something the geometry knows, and
                # this is the one choice in the module that is a convention
                # rather than a measurement -- so it is one line, here, to be
                # revisited against real OCR output rather than argued about.
                text=f"{upper.text}{CELL_SEPARATOR}{lower.text}",
            )
        )
        used_as_upper.add(upper.ordinal)
        used_as_lower.add(lower.ordinal)

    return tuple(associations), tuple(rejections)


def layout_from(
    detections: Iterable[Detection],
    *,
    image_width: float | None = None,
) -> Layout:
    """Everything the detections of one image support.

    The single entry point, and a pure function: it reads no file, calls no
    engine, and returns the same answer for the same input every time.

    Args:
        detections: What an engine reported, in any order. Malformed ones are
            counted and dropped rather than raising -- an engine's output is
            untrusted input, and one bad box must not cost the whole image.
        image_width: The image's width in pixels, when known. The reader takes
            it from the header it already probed. ``None`` means unknown, which
            only weakens the heading test; see
            ``MAX_ASSOCIATION_WIDTH_FRACTION``.

    Returns:
        A :class:`Layout`. ``lines`` holds all the recognised text whether or
        not any association was found, so an image that is a plain label is
        still readable. ``associations`` holds only the pairings the geometry
        could make unambiguously, and ``rejections`` records the ones it could
        have made and declined.
    """
    supplied = list(detections)
    usable = [detection for detection in supplied if _is_usable(detection)]
    kept = _without_duplicates(usable)
    lines = _lines_from(kept)
    associations, rejections = _associate(lines, image_width=image_width)

    return Layout(
        lines=lines,
        associations=associations,
        rejections=rejections,
        dropped_detections=len(supplied) - len(usable),
        duplicate_detections=len(usable) - len(kept),
    )
