"""Tests for the image geometry: which recognised text belongs with which.

Everything here is invented. No image is opened, no engine is called and no
sample corpus file is consulted, because the module under test is a pure
function of boxes -- and that is the whole point of splitting it out. A rule
about where a label sits relative to an identifier can be checked against
coordinates chosen to make one specific thing true, which is impossible when the
coordinates have to come out of a photograph.

**The names are deliberately unlike the sample corpus.** The sample pinout pairs
``B14`` with ``THERM_ALERT#``, and a test written around those two strings would
pass just as happily against a lookup table that returned them whenever it saw
the filename. So the fixtures use designators and signals that appear in no
sample file, and one test asserts the rule works on a layout sharing nothing
with the sample but its shape.

**The invariant under test throughout is that nothing is invented.** A wrong
pairing is not a missing answer: it is a confident statement that a particular
pin carries a particular signal, indistinguishable downstream from one the image
really shows, and cited as evidence. Several tests below exist only to pin down
the cases where the rule must return *nothing*.
"""

from __future__ import annotations

import pytest
from config import CELL_SEPARATOR
from ingestion.readers.image_geometry import (
    Detection,
    RejectionReason,
    layout_from,
)

#: Sizes chosen so the arithmetic in the tests is checkable by hand.
#:
#: An identifier sits at ``top=100`` and is 60x20, so its box ends at 120; a
#: label sits at ``top=200`` and is 120x20. The gap between them is therefore
#: exactly 80 pixels -- four text heights, comfortably inside the 2-to-12 band
#: -- and both are centred on the same x, so the leader line between them is
#: vertical. ``TestTheMeasuredShape`` at the end of this file re-runs the same
#: rules against the sizes measured off the real image, since a tolerance that
#: only ever saw round numbers is not a tested tolerance.
IDENTIFIER_WIDTH = 60.0
LABEL_WIDTH = 120.0
TEXT_HEIGHT = 20.0
IDENTIFIER_TOP = 100.0
LABEL_TOP = 200.0


def boxed(
    text: str,
    centre: float,
    top: float,
    *,
    width: float = IDENTIFIER_WIDTH,
    height: float = TEXT_HEIGHT,
    confidence: float = 0.95,
) -> Detection:
    """A detection of the given size, centred on ``centre``."""
    return Detection(
        text=text,
        confidence=confidence,
        left=centre - width / 2,
        top=top,
        width=width,
        height=height,
    )


def identifier(text: str, centre: float, top: float = IDENTIFIER_TOP) -> Detection:
    return boxed(text, centre, top, width=IDENTIFIER_WIDTH)


def label(
    text: str,
    centre: float,
    top: float = LABEL_TOP,
    *,
    width: float = LABEL_WIDTH,
    confidence: float = 0.95,
) -> Detection:
    return boxed(text, centre, top, width=width, height=TEXT_HEIGHT, confidence=confidence)


def column(designator: str, signal: str, centre: float) -> list[Detection]:
    """One identifier with its label directly beneath, as a diagram draws it."""
    return [identifier(designator, centre), label(signal, centre)]


class TestAssociations:
    def test_identifiers_pair_with_the_labels_beneath_them(self) -> None:
        """The base case, on names that appear in no sample file."""
        layout = layout_from(
            column("J9", "ENABLE", 100)
            + column("K2", "CLOCK", 400)
            + column("AA3", "SHIELD_DRAIN", 700)
        )

        assert [association.text for association in layout.associations] == [
            f"J9{CELL_SEPARATOR}ENABLE",
            f"K2{CELL_SEPARATOR}CLOCK",
            f"AA3{CELL_SEPARATOR}SHIELD_DRAIN",
        ]
        assert layout.rejections == ()

    def test_detections_in_any_order_give_the_same_answer(self) -> None:
        """Ordinal assignment is the geometry's job, not the engine's."""
        detections = column("J9", "ENABLE", 100) + column("K2", "CLOCK", 400)

        forwards = layout_from(detections)
        backwards = layout_from(list(reversed(detections)))

        assert forwards == backwards

    def test_the_pairing_is_rendered_identifier_first(self) -> None:
        """Reading order, and the same key-then-value order the other readers use.

        Which side is the question's subject is not something geometry knows, so
        this is a convention rather than a measurement -- pinned here so that
        changing it is a deliberate act with a failing test attached, not a
        quiet edit.
        """
        layout = layout_from(column("J9", "ENABLE", 100))

        assert layout.associations[0].text == "J9 | ENABLE"

    def test_the_association_keeps_both_source_lines(self) -> None:
        """Provenance has to name the two boxes the pairing was inferred from.

        The pairing is the finding and it exists nowhere in the image as text,
        so a location pointing at one line would leave a reader unable to check
        which two things were related.
        """
        layout = layout_from(column("J9", "ENABLE", 100))
        association = layout.associations[0]

        assert association.upper.text == "J9"
        assert association.lower.text == "ENABLE"
        assert association.upper.ordinal == 1
        assert association.lower.ordinal == 2
        assert "text lines 1 and 2" in association.location
        assert "region" in association.location

    def test_a_diagram_sharing_nothing_with_the_sample_still_works(self) -> None:
        """Guards against a rule that is really a lookup table.

        Different designator shape, a two-word label, a lower-case signal and a
        signal with punctuation: none of it may change the outcome, because the
        rule is about position and nothing else.
        """
        layout = layout_from(
            column("Q17a", "chip select", 200) + column("R12", "vcc_io!", 600)
        )

        assert [association.text for association in layout.associations] == [
            "Q17a | chip select",
            "R12 | vcc_io!",
        ]

    def test_a_plain_label_has_text_but_no_associations(self) -> None:
        """An image with no diagram in it is text, not a failed read.

        Withholding the lines because no relationship was found would throw away
        the answer along with the structure.
        """
        layout = layout_from(
            [
                boxed("Orrery Systems", 380, 40, width=240),
                boxed("WIDGET MODEL QX-9", 380, 66, width=260),
                boxed("SERIAL 9F-000111-22", 380, 92, width=280),
            ]
        )

        assert len(layout.lines) == 3
        assert layout.associations == ()

    def test_the_image_width_is_used_when_it_is_known(self) -> None:
        layout = layout_from(column("J9", "ENABLE", 100), image_width=1000)

        assert [association.text for association in layout.associations] == ["J9 | ENABLE"]


class TestDuplicateLabels:
    def test_the_same_label_in_two_columns_is_kept_twice(self) -> None:
        """Two ground pins are two pins, and collapsing them loses one.

        The deduplication rule compares text *and* position precisely so that
        this case survives: identical spelling at a different place is a real
        second occurrence, not a repeat.
        """
        layout = layout_from(column("P1", "GND", 100) + column("P2", "GND", 400))

        assert [association.text for association in layout.associations] == [
            "P1 | GND",
            "P2 | GND",
        ]
        assert layout.duplicate_detections == 0
        assert len(layout.lines) == 4

    def test_identical_labels_do_not_bleed_across_columns(self) -> None:
        """Each label keeps its own pin, which is the whole difficulty here."""
        layout = layout_from(column("P1", "GND", 100) + column("P2", "GND", 400))

        pairs = {
            association.upper.text: association.lower.ordinal
            for association in layout.associations
        }

        assert len(set(pairs.values())) == 2


class TestDuplicateDetections:
    def test_the_same_box_reported_twice_is_counted_once(self) -> None:
        layout = layout_from(
            [identifier("J9", 100), label("ENABLE", 100), identifier("J9", 100)]
        )

        assert layout.duplicate_detections == 1
        assert [line.text for line in layout.lines] == ["J9", "ENABLE"]
        assert len(layout.associations) == 1

    def test_overlapping_boxes_for_one_word_collapse(self) -> None:
        layout = layout_from([identifier("J9", 100), label("ENABLE", 100), label("ENABLE", 104)])

        assert layout.duplicate_detections == 1
        assert layout.associations[0].lower.text == "ENABLE"

    def test_a_box_nested_inside_another_counts_as_the_same_word(self) -> None:
        """Engines disagree about where a word ends, and both answers arrive.

        The overlap is measured against the smaller box rather than the union
        for exactly this case: a short box wholly inside a long one covers all
        of itself, which is as duplicated as a detection gets.
        """
        layout = layout_from(
            [identifier("J9", 100), label("ENABLE", 100), label("ENABLE", 100, width=60)]
        )

        assert layout.duplicate_detections == 1
        assert [line.text for line in layout.lines] == ["J9", "ENABLE"]

    def test_different_spellings_of_one_word_collapse(self) -> None:
        """Comparison goes through the project's single normalisation."""
        layout = layout_from([identifier("J9", 100), label("EN-ABLE", 100), label("ENABLE", 100)])

        assert layout.duplicate_detections == 1

    def test_the_kept_copy_is_the_most_confident_one(self) -> None:
        layout = layout_from(
            [
                identifier("J9", 100),
                label("ENABLE", 100, confidence=0.60),
                label("ENABLE", 100, confidence=0.99),
            ]
        )

        assert layout.duplicate_detections == 1
        assert layout.lines[1].confidence == pytest.approx(0.99)


class TestWordsBecomeLines:
    def test_words_on_one_baseline_are_joined_left_to_right(self) -> None:
        """Engines report words or lines; both have to arrive the same way."""
        detections = [
            boxed("REV-C2", 600, 100, width=80),
            boxed("BOARD", 400, 100, width=70),
            boxed("REVISION", 480, 104, width=110),
        ]

        layout = layout_from(detections)

        assert [line.text for line in layout.lines] == ["BOARD REVISION REV-C2"]

    def test_two_columns_on_one_baseline_stay_apart(self) -> None:
        """The failure this guards is the expensive one.

        Labels on the same row are on the same baseline, so grouping by vertical
        overlap alone would fuse every column of a diagram into one line -- and
        a fused line is too long to be a candidate, which loses every pairing on
        the image at once.
        """
        layout = layout_from(
            [label("GND", 100), label("SDA", 400), label("SCL", 700)]
        )

        assert [line.text for line in layout.lines] == ["GND", "SDA", "SCL"]

    def test_a_lines_confidence_is_its_weakest_word(self) -> None:
        detections = [
            boxed("BOARD", 400, 100, width=70, confidence=0.99),
            boxed("REVISION", 480, 100, width=110, confidence=0.71),
        ]

        assert layout_from(detections).lines[0].confidence == pytest.approx(0.71)

    def test_a_word_arriving_late_still_joins_the_two_halves_it_bridges(self) -> None:
        """The regression that made the merge pass necessary.

        The engine reports ``NORTH`` and ``SOUTH`` first, left to right, and the
        wider word between them afterwards. Neither outer word shares a line
        with the other -- they are 340 pixels apart, which is why they start as
        separate groups -- so with a single greedy pass the middle word joins
        the first group and the line stays split in two.

        A split line is not a cosmetic problem: both halves are short enough to
        look like a designator or a signal, so the next version of this rule
        could pair a fragment with a fragment and report a relationship between
        two pieces of one line.
        """
        detections = [
            boxed("NORTH", 100, 100, width=60),
            boxed("SOUTH", 500, 100, width=60),
            boxed("CROSSING", 300, 102, width=300),
        ]

        layout = layout_from(detections)

        assert [line.text for line in layout.lines] == ["NORTH CROSSING SOUTH"]
        assert len(layout.lines) == 1

    def test_a_single_detection_per_line_is_unchanged_by_the_join(self) -> None:
        """The join only ever adds a space, so whole-line engines pass through."""
        layout = layout_from([boxed("A WHOLE LINE AS ONE BOX", 500, 100, width=300)])

        assert [line.text for line in layout.lines] == ["A WHOLE LINE AS ONE BOX"]

    def test_reading_order_runs_top_to_bottom_then_left_to_right(self) -> None:
        """Ordering is by position, and the left-to-right half is deliberate.

        The engine's report order is not reading order -- OCR engines emit
        detections in whatever order their layout pass produced -- so a line's
        ordinal has to be recomputed from its box.
        """
        detections = [
            label("SECOND", 700, 300),
            identifier("FIRST", 700),
            label("THIRD", 100, 300),
        ]

        layout = layout_from(detections)

        assert [line.text for line in layout.lines] == ["FIRST", "THIRD", "SECOND"]
        assert [line.ordinal for line in layout.lines] == [1, 2, 3]
        assert layout.lines[0].location.startswith("text line 1 at (")


class TestTheMeasuredShape:
    """The one fixture here that is not invented from scratch.

    The sizes are the ones measured off the sample pinout by decoding the PNG
    and profiling its ink rows: boxes 117 pixels wide on a 150-pixel column
    pitch, designators 16 pixels tall with their tops at y=150, signal labels 10
    pixels tall with their tops at y=222. The *text* is invented -- the sample's
    real designators and signals appear nowhere in this file, for the reason
    given in the module docstring -- but the geometry is the real geometry.

    It exists because every other test in this file uses the round numbers the
    arithmetic is easy on, and a tolerance tuned to round numbers can be wrong
    on the real thing while every test still passes. The gap here is 56 pixels
    against a 10-pixel unit: 5.6 text heights, which is what set the ceiling in
    ``ASSOCIATION_MAX_GAP_FACTOR`` rather than a guess.
    """

    PIN_WIDTH = 117.0
    PIN_PITCH = 150.0
    DESIGNATOR_TOP = 150.0
    DESIGNATOR_HEIGHT = 16.0
    LABEL_TOP = 222.0
    LABEL_HEIGHT = 10.0

    COLUMNS = (
        ("A1", "VIN_MAIN"),
        ("B7", "CLK_OUT"),
        ("C12", "CS_N"),
        ("D4", "RDY"),
        ("E9", "VBAT_SW"),
        ("F2", "ALERT_N"),
    )

    def detections(self) -> list[Detection]:
        shapes = []
        for index, (designator, signal) in enumerate(self.COLUMNS):
            centre = 120.0 + index * self.PIN_PITCH
            shapes.append(
                boxed(
                    designator,
                    centre,
                    self.DESIGNATOR_TOP,
                    width=self.PIN_WIDTH,
                    height=self.DESIGNATOR_HEIGHT,
                )
            )
            shapes.append(
                boxed(
                    signal,
                    centre,
                    self.LABEL_TOP,
                    width=self.PIN_WIDTH - 27,
                    height=self.LABEL_HEIGHT,
                )
            )
        return shapes

    def test_six_columns_pair_exactly_once_each(self) -> None:
        layout = layout_from(self.detections())

        assert [association.text for association in layout.associations] == [
            f"{designator}{CELL_SEPARATOR}{signal}"
            for designator, signal in self.COLUMNS
        ]
        assert layout.rejections == ()
        assert layout.dropped_detections == 0
        assert layout.duplicate_detections == 0

    def test_neighbouring_columns_are_not_fused_into_one_line(self) -> None:
        """33 pixels between boxes is the tightest horizontal gap in the image.

        It is only 2 text heights wide, so it is the case that decides whether
        the line-grouping gap can be loosened. If it ever is, this test is what
        says so.

        Fusion would show up as a line holding several words, and it would be
        silent: the fused line is still text, but it is no longer a candidate
        for either role, so every pairing on the image disappears at once.
        """
        layout = layout_from(self.detections())

        assert len(layout.lines) == 2 * len(self.COLUMNS)
        assert all(len(line.text.split()) == 1 for line in layout.lines)

    def test_a_column_missing_its_label_does_not_borrow_the_next_one(self) -> None:
        """One unreadable box must cost one pairing, not two."""
        detections = [
            detection
            for detection in self.detections()
            if detection.text != self.COLUMNS[2][1]
        ]

        layout = layout_from(detections)

        assert [association.text for association in layout.associations] == [
            f"{designator}{CELL_SEPARATOR}{signal}"
            for designator, signal in self.COLUMNS
            if designator != self.COLUMNS[2][0]
        ]


class TestRefusals:
    def test_two_labels_a_hair_apart_under_one_identifier_are_refused(self) -> None:
        """A wrapped label split into two boxes, and the coin toss it creates.

        Both boxes are centred under the same identifier and 15 pixels apart, so
        the pairing score differs by less than one text height. Picking the
        nearer would be a guess presented as a fact.
        """
        layout = layout_from(
            [identifier("J9", 500), label("ENABLE", 500, 200), label("ENABLE", 500, 215)]
        )

        assert layout.associations == ()
        assert [rejection.reason for rejection in layout.rejections] == [
            RejectionReason.TOO_CLOSE_TO_CALL
        ]
        assert len(layout.rejections[0].lines) == 3

    def test_an_identifier_equidistant_between_two_labels_is_refused(self) -> None:
        """The same rule sideways: neither label is closer than the other."""
        layout = layout_from(
            [
                boxed("WIDE", 500, 100, width=300),
                label("LEFT", 420, 200),
                label("RIGHT", 580, 200),
            ]
        )

        assert layout.associations == ()
        assert [rejection.reason for rejection in layout.rejections] == [
            RejectionReason.TOO_CLOSE_TO_CALL
        ]

    def test_a_clearly_nearest_label_is_still_chosen(self) -> None:
        """The margin refuses ties, not everything -- the contrast case.

        Without this, a rule that simply refused whenever two candidates existed
        would pass the two tests above.
        """
        layout = layout_from(
            [identifier("J9", 500), label("NEAR", 500, 200), label("FAR", 500, 230)]
        )

        assert [association.text for association in layout.associations] == ["J9 | NEAR"]

    def test_two_identifiers_cannot_share_one_label(self) -> None:
        """It belongs to one of them, and the boxes do not say which.

        The label here is 240 wide and centred between the two identifiers, so
        both are within the offset gate and only the distance separates them.
        """
        layout = layout_from(
            [
                identifier("J9", 100),
                identifier("K2", 350),
                label("SHARED", 180, width=240),
            ]
        )

        assert len(layout.associations) == 1
        assert layout.associations[0].upper.text == "J9"
        assert layout.associations[0].lower.text == "SHARED"
        assert [rejection.reason for rejection in layout.rejections] == [
            RejectionReason.SHARED_LABEL
        ]
        assert "K2" in layout.rejections[0].detail

    def test_the_contested_label_goes_to_the_nearer_identifier(self) -> None:
        """Distance decides, not the order the engine happened to report in."""
        closer_to_the_second = layout_from(
            [
                identifier("J9", 100),
                identifier("K2", 350),
                label("SHARED", 250, width=240),
            ]
        )

        assert closer_to_the_second.associations[0].upper.text == "K2"

    def test_three_stacked_tokens_are_not_read_as_two_associations(self) -> None:
        """A token cannot be a label for the one above and an identifier for the
        one below.

        The geometry cannot say which reading is right, so the association that
        would rest on answering it is dropped. Which of the two survives is
        arbitrary -- what matters is that exactly one does, and that the same
        input always gives the same one.
        """
        detections = [
            boxed("ALPHA", 500, 40),
            boxed("BRAVO", 500, 140),
            boxed("CHARLIE", 500, 240),
        ]

        layout = layout_from(detections)

        assert [association.text for association in layout.associations] == ["ALPHA | BRAVO"]
        assert [rejection.reason for rejection in layout.rejections] == [
            RejectionReason.CHAINED
        ]
        assert layout_from(detections) == layout

    def test_adjacent_lines_of_one_block_are_not_a_diagram_relationship(self) -> None:
        """The gap floor, and the reason it is load-bearing.

        Both tokens pass every other test in this module: each is a single
        alphabetic word, each is centred over the other, and the pairing is
        otherwise exactly the shape of a pin and its signal. They are 6 pixels
        apart, which is ordinary line spacing -- and reading line spacing as a
        relationship would turn every block of text in every image into a field
        and a value.
        """
        layout = layout_from(
            [
                boxed("SN9F000111", 500, 100, width=200),
                boxed("MALAYSIA", 500, 126, width=160),
            ]
        )

        assert len(layout.lines) == 2
        assert layout.associations == ()
        assert layout.rejections == ()

    def test_an_identifier_with_nothing_below_it_is_not_a_rejection(self) -> None:
        """A token with no neighbour is ordinary text.

        Rejections record a pairing that was declined, not the absence of one.
        Recording every lonely token would bury the real refusals in noise.
        """
        layout = layout_from([identifier("J9", 500)])

        assert layout.lines[0].text == "J9"
        assert layout.associations == ()
        assert layout.rejections == ()

    def test_a_label_with_nothing_above_it_is_not_a_rejection(self) -> None:
        layout = layout_from([label("ENABLE", 500)])

        assert layout.associations == ()
        assert layout.rejections == ()

    def test_a_missing_label_does_not_pull_one_from_another_column(self) -> None:
        """The identifier is left unpaired rather than reaching sideways."""
        layout = layout_from([identifier("J9", 100), *column("K2", "CLOCK", 700)])

        assert [association.text for association in layout.associations] == ["K2 | CLOCK"]
        assert layout.rejections == ()


class TestLongTextIsNotAToken:
    def test_a_multi_word_heading_is_not_paired(self) -> None:
        """Everything else about this layout is a valid pairing.

        The heading is centred over the label, the gap between them is four text
        heights, and the widths line up. Word count is the only thing standing
        between a title and a fabricated field -- which is why it is tested on
        its own, with the width guard unavailable.
        """
        layout = layout_from(
            [
                boxed("QX-9 connector - pin assignment", 500, 100, width=560, height=24),
                label("ENABLE", 500),
            ]
        )

        assert layout.lines[0].text == "QX-9 connector - pin assignment"
        assert layout.associations == ()

    def test_a_short_but_very_wide_heading_is_not_paired(self) -> None:
        """One word, so the word count cannot be what excludes it.

        A heading set large across the full page width is a single token to a
        word-counting rule, and the gap to the caption under it is a perfectly
        ordinary diagram gap. Width against the canvas is the only thing left,
        which is why the reader has to pass the image width in.
        """
        detections = [
            boxed("OVERVIEW", 500, 100, width=800, height=24),
            label("ENABLE", 500),
        ]

        layout = layout_from(detections, image_width=1000)

        assert len(layout.lines) == 2
        assert layout.associations == ()

    def test_a_distant_heading_is_not_paired_even_without_an_image_width(self) -> None:
        """The gap ceiling, which needs no canvas size to apply.

        A single word again, so that the ceiling is the guard under test rather
        than the word count, and far enough below that no canvas size could make
        the pairing plausible.
        """
        layout = layout_from(
            [
                boxed("OVERVIEW", 500, 0, width=300, height=24),
                label("ENABLE", 500, 320),
            ]
        )

        assert layout.associations == ()

    def test_prose_lines_are_kept_as_text_but_never_become_candidates(self) -> None:
        prose = "The connector is rated for the temperature range given in the table below."

        layout = layout_from(
            [boxed(prose, 500, 100, width=900, height=24), label("ENABLE", 500)]
        )

        assert [line.text for line in layout.lines] == [prose, "ENABLE"]
        assert layout.associations == ()

    def test_a_sentence_under_a_token_is_not_a_label(self) -> None:
        """The label side of the same rule, which is the wider of the two.

        The identifier here is a perfect one and the gap is a real diagram gap,
        so the only thing between this pin and a fabricated row is how long a
        signal name may be. Asserting on the absence of a *rejection* as well
        matters: the pairing was never plausible, so there was nothing to
        refuse, and recording one would make an ordinary page look ambiguous.
        """
        layout = layout_from(
            [
                identifier("J9", 500),
                boxed("pulled up internally to the rail", 500, 200, width=520),
            ]
        )

        assert len(layout.lines) == 2
        assert layout.associations == ()
        assert layout.rejections == ()

    def test_a_number_on_its_own_is_not_a_token(self) -> None:
        """A bare quantity is not a name, and diagrams are full of quantities."""
        layout = layout_from(
            [
                boxed("40", 500, 100, width=40),
                boxed("25", 500, 200, width=40),
            ]
        )

        assert len(layout.lines) == 2
        assert layout.associations == ()


class TestInvalidInput:
    def test_no_detections_yields_an_empty_layout(self) -> None:
        layout = layout_from([])

        assert layout.lines == ()
        assert layout.associations == ()
        assert layout.rejections == ()
        assert layout.dropped_detections == 0
        assert layout.duplicate_detections == 0

    def test_no_detections_yields_an_empty_layout_with_an_image_width(self) -> None:
        assert layout_from([], image_width=1000) == layout_from([])

    @pytest.mark.parametrize(
        "detection",
        [
            pytest.param(boxed("ENABLE", 500, 200, width=0), id="zero-width"),
            pytest.param(boxed("ENABLE", 500, 200, height=0), id="zero-height"),
            pytest.param(boxed("ENABLE", 500, 200, width=-120), id="negative-width"),
            pytest.param(boxed("ENABLE", 500, 200, height=-20), id="negative-height"),
            pytest.param(
                Detection("ENABLE", 0.95, float("nan"), 200, 120, 20), id="nan-left"
            ),
            pytest.param(
                Detection("ENABLE", 0.95, 0, float("inf"), 120, 20), id="infinite-top"
            ),
            pytest.param(
                Detection("ENABLE", float("nan"), 0, 200, 120, 20), id="nan-confidence"
            ),
            pytest.param(
                Detection("ENABLE", 0.10, 0, 200, 120, 20), id="confidence-too-low"
            ),
            pytest.param(boxed("   ", 500, 200), id="blank-text"),
            pytest.param(boxed("---", 500, 200), id="punctuation-only"),
            pytest.param(Detection("ENABLE", 0.95, True, 200, 120, 20), id="bool-left"),
        ],
    )
    def test_a_malformed_detection_is_dropped_not_raised(self, detection: Detection) -> None:
        """An engine's output is untrusted, and one bad box must not cost the image."""
        layout = layout_from([detection])

        assert layout.lines == ()
        assert layout.associations == ()
        assert layout.dropped_detections == 1

    def test_one_bad_detection_does_not_cost_the_good_ones(self) -> None:
        layout = layout_from(
            [
                *column("J9", "ENABLE", 100),
                boxed("ENABLE", 400, 200, width=0),
                Detection("CLOCK", float("nan"), 0, 0, 10, 10),
            ]
        )

        assert [association.text for association in layout.associations] == ["J9 | ENABLE"]
        assert layout.dropped_detections == 2

    def test_low_confidence_noise_never_becomes_a_line(self) -> None:
        """The barcode case: text-shaped detections at near-zero confidence."""
        noise = [
            boxed("ll", 60 + 8 * step, 300, width=4, height=24, confidence=0.20)
            for step in range(20)
        ]

        layout = layout_from([*column("J9", "ENABLE", 100), *noise])

        assert [line.text for line in layout.lines] == ["J9", "ENABLE"]
        assert layout.dropped_detections == len(noise)

    def test_detections_may_arrive_as_any_iterable(self) -> None:
        detections = column("J9", "ENABLE", 100)

        assert layout_from(iter(detections)) == layout_from(tuple(detections))


class TestNothingIsInvented:
    def test_every_association_is_two_recognised_lines_and_nothing_else(self) -> None:
        """The invariant, stated as an equality rather than an absence.

        An association's text has to be exactly a join of two lines the engine
        reported. Anything else -- a connecting word, a rewritten spelling, a
        caption -- would enter the index as corpus text.
        """
        layout = layout_from(
            column("J9", "ENABLE", 100)
            + column("K2", "CLOCK", 400)
            + [boxed("Rated for 25 W continuous", 500, 400, width=400)]
        )
        recognised = {line.text for line in layout.lines}

        for association in layout.associations:
            parts = association.text.split(CELL_SEPARATOR)

            assert len(parts) == 2
            assert set(parts) <= recognised

    def test_every_word_in_a_line_came_off_the_image(self) -> None:
        detections = [
            boxed("BOARD", 400, 100, width=70, confidence=0.95),
            boxed("REVISION", 480, 100, width=110, confidence=0.95),
            boxed("REV-C2", 620, 100, width=80, confidence=0.95),
        ]
        supplied = {detection.text for detection in detections}

        layout = layout_from(detections)

        assert set(layout.lines[0].text.split()) <= supplied

    def test_an_association_never_joins_a_line_to_itself(self) -> None:
        layout = layout_from(column("J9", "ENABLE", 100))

        for association in layout.associations:
            assert association.upper.ordinal != association.lower.ordinal

    def test_the_same_detection_is_never_used_twice(self) -> None:
        """Every line appears in at most one association, in either role."""
        layout = layout_from(
            column("J9", "ENABLE", 100)
            + column("K2", "CLOCK", 400)
            + column("AA3", "READY", 700)
        )
        used = [
            ordinal
            for association in layout.associations
            for ordinal in (association.upper.ordinal, association.lower.ordinal)
        ]

        assert len(used) == len(set(used))

    def test_the_input_is_not_modified(self) -> None:
        detections = column("J9", "ENABLE", 100) + column("K2", "CLOCK", 400)
        original = list(detections)

        layout_from(detections)

        assert detections == original
