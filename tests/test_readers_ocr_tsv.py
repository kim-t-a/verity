"""Tests for the Tesseract TSV adapter: engine output to :class:`Detection`.

Everything here is synthesised. No image is opened, no engine is called and no
sample corpus file is consulted -- the module under test is a pure function of a
string, which is the point of putting it behind its own name. A rule about which
rows are words can be checked by writing the row that makes it true.

**The shape of the fixtures is real; the words are not.** The rows below carry
the column layout, the fractional confidences, the hierarchy of levels and the
space-only rule rows that Tesseract 5.3.4 actually emitted when it was run over
the two sample images in Stage 7B. Those are measurements, and reproducing them
is how the three facts that would otherwise be guessed wrong stay pinned to
something that really came out of an engine. The *text* is invented for the
opposite reason: the sample pinout pairs ``B14`` with ``THERM_ALERT#``, and a
suite built around those two strings would pass against a lookup table that
returned them whenever it saw a pinout. The designators here appear in no sample
file.

**What is being defended.** A detection list is the input to a rule that refuses
to guess, so an adapter that quietly invents a row is worse than one that drops
it: a repaired row is indistinguishable from a read one. Several tests below
exist only to pin down that nothing is repaired, and that a malformed row costs
itself and nothing else.
"""

from __future__ import annotations

import pytest
from config import MIN_OCR_CONFIDENCE
from ingestion.readers.image_geometry import layout_from
from ingestion.readers.ocr_tsv import TSV_HEADER, words_from_tsv

_HEADER = "\t".join(TSV_HEADER)


def _fields(
    text: str,
    *,
    level: int | float = 5,
    left: float = 100,
    top: float = 200,
    width: float = 40,
    height: float = 16,
    conf: float = 96.154,
    line_num: int = 1,
    word_num: int = 1,
) -> dict[str, object]:
    """One TSV row as a mapping, using the real column names."""
    return {
        "level": level,
        "page_num": 1,
        "block_num": 1,
        "par_num": 1,
        "line_num": line_num,
        "word_num": word_num,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "conf": conf,
        "text": text,
    }


def _render(header: tuple[str, ...], rows: list[dict[str, object]]) -> str:
    lines = ["\t".join(header)]
    lines += ["\t".join(str(row.get(name, "")) for name in header) for row in rows]
    return "\n".join(lines)


def _row(text: str, **kwargs: object) -> dict[str, object]:
    """A word-level row -- level 5, which is the only level that is a word."""
    return _fields(text, **kwargs)


def _aggregate(text: str, *, level: int) -> dict[str, object]:
    """One of Tesseract's own summary rows: page, block, paragraph or line.

    Levels 1 to 4 restate the text of the words beneath them, so a row built
    here carries the same words a real aggregate row does.
    """
    return _fields(text, level=level, width=400, height=20)


def _tsv(*rows: dict[str, object], header: tuple[str, ...] = TSV_HEADER) -> str:
    return _render(header, list(rows))


class TestRecognisingTheFormat:
    """Whether the blob is TSV at all, which is not the same as being empty."""

    def test_a_well_formed_blob_is_recognised(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL")))
        assert parsed.header_recognised
        assert [d.text for d in parsed.detections] == ["SENTINEL"]

    def test_the_header_line_is_not_parsed_as_a_row(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL")))
        assert [d.text for d in parsed.detections] == ["SENTINEL"]
        assert parsed.skipped_rows == 0

    def test_an_empty_blob_is_not_a_recognised_header(self) -> None:
        parsed = words_from_tsv("")
        assert not parsed.header_recognised
        assert parsed.detections == ()

    def test_an_error_message_is_not_mistaken_for_a_blank_image(self) -> None:
        # The distinction the field exists for: the engine never produced TSV,
        # so the image has not been read. Reporting this as "no words" would
        # let a broken invocation pass for a blank page.
        parsed = words_from_tsv("Error, cannot read input file /tmp/x.png")
        assert not parsed.header_recognised
        assert parsed.detections == ()

    def test_a_blob_of_tabs_without_a_header_is_not_recognised(self) -> None:
        parsed = words_from_tsv("\t\t\t\t\t\t\t\t\t\t\t\n")
        assert not parsed.header_recognised

    def test_a_header_missing_a_required_column_is_not_recognised(self) -> None:
        header = tuple(name for name in TSV_HEADER if name != "conf")
        parsed = words_from_tsv(_tsv(_row("SENTINEL"), header=header))
        assert not parsed.header_recognised

    def test_an_extra_column_shifts_nothing(self) -> None:
        # Columns are located by name, so a build that appends one must not
        # silently move every value one field to the left.
        header = (*TSV_HEADER, "block_extra")
        parsed = words_from_tsv(_tsv(_row("SENTINEL", left=417, top=93), header=header))
        assert [d.text for d in parsed.detections] == ["SENTINEL"]
        assert parsed.detections[0].left == 417
        assert parsed.detections[0].top == 93

    def test_a_reordered_header_is_still_read_correctly(self) -> None:
        header = tuple(reversed(TSV_HEADER))
        parsed = words_from_tsv(_tsv(_row("SENTINEL", left=417, top=93), header=header))
        assert [d.text for d in parsed.detections] == ["SENTINEL"]
        assert parsed.detections[0].left == 417

    def test_an_absent_optional_column_does_not_stop_a_read(self) -> None:
        # line_num and word_num are the engine's bookkeeping; nothing downstream
        # reads them, so their absence is not a format this adapter rejects.
        header = tuple(name for name in TSV_HEADER if name not in {"line_num", "word_num"})
        parsed = words_from_tsv(_tsv(_row("SENTINEL"), header=header))
        assert parsed.header_recognised
        assert [d.text for d in parsed.detections] == ["SENTINEL"]


class TestOnlyWordLevelRowsAreWords:
    """The hierarchy rule -- the one that silently multiplies the corpus."""

    def test_aggregate_rows_contributing_text_are_not_detections(self) -> None:
        blob = _tsv(
            _aggregate("SENTINEL", level=1),
            _aggregate("SENTINEL", level=2),
            _aggregate("SENTINEL", level=3),
            _aggregate("SENTINEL", level=4),
            _row("SENTINEL"),
        )
        parsed = words_from_tsv(blob)
        # One row is a word. The other four restate it and would otherwise put
        # the same token in the index five times over.
        assert [d.text for d in parsed.detections] == ["SENTINEL"]

    def test_the_aggregate_rows_are_not_counted_as_skipped(self) -> None:
        blob = _tsv(*[_aggregate("SENTINEL", level=n) for n in (1, 2, 3, 4)], _row("SENTINEL"))
        # They are not skipped -- they are not words, which is a different
        # finding, and a large number here would bury real corruption.
        assert words_from_tsv(blob).skipped_rows == 0

    def test_a_level_written_as_a_float_is_still_a_word(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL", level=5.0)))
        assert [d.text for d in parsed.detections] == ["SENTINEL"]
        assert parsed.skipped_rows == 0

    def test_interleaved_levels_do_not_disturb_the_word_order(self) -> None:
        blob = _tsv(
            _row("FIRST", left=10),
            _aggregate("FIRST", level=4),
            _row("SECOND", left=90),
            _aggregate("SECOND", level=4),
        )
        parsed = words_from_tsv(blob)
        assert [d.text for d in parsed.detections] == ["FIRST", "SECOND"]


class TestTheMeasuredConfidenceColumn:
    """The column documented as an integer, which is measurably a float."""

    def test_a_fractional_confidence_is_divided_by_one_hundred(self) -> None:
        # The real value Tesseract 5.3.4 printed for the pinout's title word.
        parsed = words_from_tsv(_tsv(_row("SENTINEL", conf=91.944946)))
        assert parsed.detections[0].confidence == pytest.approx(0.91944946)

    def test_a_whole_number_confidence_is_divided_too(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL", conf=92)))
        assert parsed.detections[0].confidence == pytest.approx(0.92)

    def test_a_confidence_is_not_left_on_the_engine_scale(self) -> None:
        # 96.154 must not arrive as 96.154: on a 0..1 field that is a value no
        # threshold can ever reject, so every detection would look trusted.
        parsed = words_from_tsv(_tsv(_row("SENTINEL", conf=96.154)))
        assert parsed.detections[0].confidence < 1.0

    def test_a_declined_confidence_is_passed_through_scaled(self) -> None:
        # Tesseract writes -1 for things that are not text. Interpreting it is
        # the geometry's job, not this adapter's.
        parsed = words_from_tsv(_tsv(_row(" ", conf=-1)))
        assert parsed.detections[0].confidence == pytest.approx(-0.01)
        assert parsed.skipped_rows == 0


class TestTheMeasuredRuleRows:
    """The division of labour: the adapter reports, the geometry judges."""

    def test_a_space_only_row_is_passed_through_rather_than_filtered(self) -> None:
        # Measured: drawn borders come back as word-level rows whose text is a
        # single space. A ``if not text`` check here would let it through
        # anyway, and a stricter check would duplicate a rule the geometry
        # already owns and documents.
        parsed = words_from_tsv(_tsv(_row(" ")))
        assert [d.text for d in parsed.detections] == [" "]
        assert parsed.skipped_rows == 0

    def test_a_space_only_row_is_dropped_by_the_geometry(self) -> None:
        # The other half of the contract, asserted end to end: what this
        # adapter deliberately keeps does not survive into the layout.
        blob = _tsv(_row(" "), _row("  "), _row("SENTINEL", left=10))
        layout = layout_from(words_from_tsv(blob).detections)
        assert [line.text for line in layout.lines] == ["SENTINEL"]
        assert layout.dropped_detections == 2

    def test_an_empty_text_field_is_passed_through(self) -> None:
        parsed = words_from_tsv(_tsv(_row("")))
        assert [d.text for d in parsed.detections] == [""]

    def test_a_declined_confidence_is_dropped_by_the_geometry(self) -> None:
        blob = _tsv(_row(" ", conf=-1), _row("SENTINEL", left=10))
        layout = layout_from(words_from_tsv(blob).detections)
        assert [line.text for line in layout.lines] == ["SENTINEL"]

    def test_a_punctuation_only_word_reaches_the_geometry_and_is_dropped(self) -> None:
        # Measured on the pinout's title: the lone hyphen is discarded by the
        # geometry's alphanumeric test, not by this adapter.
        parsed = words_from_tsv(_tsv(_row("-")))
        assert [d.text for d in parsed.detections] == ["-"]
        assert layout_from(parsed.detections).lines == ()

    def test_a_real_word_at_the_confidence_floor_is_kept(self) -> None:
        # The floor is a threshold on real text, not on noise: the pinout's
        # least confident genuine line sits just above it.
        conf = (MIN_OCR_CONFIDENCE * 100) + 1
        parsed = words_from_tsv(_tsv(_row("SENTINEL", conf=conf)))
        assert layout_from(parsed.detections, image_width=1000).lines != ()


class TestNothingIsRepaired:
    """A malformed row costs itself, and is never guessed back into shape."""

    def test_a_row_with_too_few_fields_is_counted_and_skipped(self) -> None:
        # Seven fields where the header promises twelve.
        blob = _HEADER + "\n" + "\t".join(["5", "1", "1", "1", "1", "1", "10"])
        parsed = words_from_tsv(blob)
        assert parsed.detections == ()
        assert parsed.skipped_rows == 1

    def test_a_row_with_too_many_fields_is_counted_and_skipped(self) -> None:
        good = _render(TSV_HEADER, [_row("SENTINEL")]).splitlines()[1]
        parsed = words_from_tsv("\n".join([_HEADER, good + "\tEXTRA"]))
        # The tail is not rejoined into the text field: guessing how a corrupt
        # row was shaped is how invented text reaches the index.
        assert parsed.detections == ()
        assert parsed.skipped_rows == 1

    def test_a_row_whose_coordinate_is_not_a_number_is_counted_and_skipped(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL", left="n/a")))
        assert parsed.detections == ()
        assert parsed.skipped_rows == 1

    def test_a_row_whose_confidence_is_not_a_number_is_counted_and_skipped(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL", conf="high")))
        assert parsed.detections == ()
        assert parsed.skipped_rows == 1

    def test_a_bad_row_does_not_cost_the_good_rows_around_it(self) -> None:
        blob = _tsv(
            _row("FIRST", left=10),
            _row("BROKEN", left="n/a"),
            _row("THIRD", left=90),
        )
        parsed = words_from_tsv(blob)
        assert [d.text for d in parsed.detections] == ["FIRST", "THIRD"]
        assert parsed.skipped_rows == 1

    def test_the_text_of_a_corrupt_row_never_appears(self) -> None:
        blob = _tsv(_row("FIRST", left=10), _row("LEAKED", left="n/a"))
        assert "LEAKED" not in [d.text for d in words_from_tsv(blob).detections]

    def test_a_blank_line_between_rows_is_counted_rather_than_ignored(self) -> None:
        parsed = words_from_tsv("\n".join([_HEADER, "", ""]))
        # The blank line is not a word row and is not repaired into one.
        assert parsed.detections == ()
        assert parsed.skipped_rows == 1


class TestInputShapes:
    """Text, bytes and everything a real stdout stream contains."""

    def test_bytes_are_accepted(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL")).encode())
        assert [d.text for d in parsed.detections] == ["SENTINEL"]

    def test_utf8_outside_ascii_survives(self) -> None:
        parsed = words_from_tsv(_tsv(_row("RÉSUMÉ")).encode())
        assert [d.text for d in parsed.detections] == ["RÉSUMÉ"]

    def test_invalid_utf8_is_replaced_rather_than_raised_on(self) -> None:
        # These bytes are undecodable, so they cannot be reported as text: they
        # become the replacement character. Raising here would abort the
        # indexing pass over one damaged glyph, which the walker's contract
        # forbids -- so the row survives with the damage visible in it.
        blob = _tsv(_row("SENTINEL")).encode() + b"\xff\xfe"
        parsed = words_from_tsv(blob)
        assert len(parsed.detections) == 1
        assert parsed.detections[0].text.startswith("SENTINEL")
        assert "�" in parsed.detections[0].text

    def test_a_row_of_invalid_utf8_does_not_cost_the_rows_around_it(self) -> None:
        blob = _tsv(_row("FIRST", left=10), _row("SECOND", left=90)).encode()
        parsed = words_from_tsv(blob.replace(b"FIRST", b"\xff\xfe"))
        assert [d.text for d in parsed.detections] == ["��", "SECOND"]

    def test_windows_line_endings_do_not_leave_a_stray_carriage_return(self) -> None:
        # A CR left on the text field would make every value in the file fail to
        # compare equal to the answer it is supposed to be.
        blob = _tsv(_row("SENTINEL"), _row("SECOND", left=90)).replace("\n", "\r\n")
        parsed = words_from_tsv(blob)
        assert [d.text for d in parsed.detections] == ["SENTINEL", "SECOND"]

    def test_a_trailing_newline_yields_no_phantom_row(self) -> None:
        parsed = words_from_tsv(_tsv(_row("SENTINEL")) + "\n")
        assert [d.text for d in parsed.detections] == ["SENTINEL"]
        assert parsed.skipped_rows == 0

    def test_any_garbage_at_all_returns_a_value_rather_than_raising(self) -> None:
        # The reader contract: one bad file must never abort the indexing pass.
        for blob in ("", "\x00\x01\x02", "not tsv\nat all", "\n\n\n", "\t", "5\t6"):
            parsed = words_from_tsv(blob)
            assert isinstance(parsed.detections, tuple)

    def test_parsing_the_same_blob_twice_gives_the_same_answer(self) -> None:
        blob = _tsv(_row("FIRST", left=10), _row("SECOND", left=90))
        assert words_from_tsv(blob) == words_from_tsv(blob)


class TestAgainstTheGeometry:
    """The adapter and the geometry together, which is how they will be used."""

    #: A six-column diagram at the proportions measured on the real pinout:
    #: designators 150 px apart, a 56 px gap down to the labels. The names are a
    #: JTAG header, not the sample's backplane.
    COLUMNS = (
        ("P1", "TDI"),
        ("P2", "TDO"),
        ("P3", "TCK"),
        ("P4", "TMS"),
        ("P5", "TRST"),
        ("P6", "RESET"),
    )
    PITCH = 150.0
    FIRST_CENTRE = 120.0
    DESIGNATOR_TOP = 150.0
    DESIGNATOR_HEIGHT = 16.0
    LABEL_TOP = 222.0
    LABEL_HEIGHT = 10.0
    IMAGE_WIDTH = 1000.0

    def _blob(self) -> str:
        rows = []
        for index, (designator, label) in enumerate(self.COLUMNS):
            centre = self.FIRST_CENTRE + index * self.PITCH
            rows.append(
                _row(
                    designator,
                    left=centre - 20,
                    top=self.DESIGNATOR_TOP,
                    width=40,
                    height=self.DESIGNATOR_HEIGHT,
                    line_num=index + 1,
                )
            )
            rows.append(
                _row(
                    label,
                    left=centre - 25,
                    top=self.LABEL_TOP,
                    width=50,
                    height=self.LABEL_HEIGHT,
                    line_num=index + 1,
                )
            )
        return _tsv(*rows)

    def test_every_column_pairs_with_its_own_label(self) -> None:
        layout = layout_from(
            words_from_tsv(self._blob()).detections, image_width=self.IMAGE_WIDTH
        )
        assert sorted(a.text for a in layout.associations) == sorted(
            f"{designator} | {label}" for designator, label in self.COLUMNS
        )

    def test_no_association_is_refused(self) -> None:
        layout = layout_from(
            words_from_tsv(self._blob()).detections, image_width=self.IMAGE_WIDTH
        )
        assert layout.rejections == ()

    def test_each_designator_pairs_exactly_once(self) -> None:
        layout = layout_from(
            words_from_tsv(self._blob()).detections, image_width=self.IMAGE_WIDTH
        )
        uppers = [a.upper.text for a in layout.associations]
        assert sorted(uppers) == sorted(designator for designator, _ in self.COLUMNS)

    def test_the_pairing_survives_a_designator_being_read_last(self) -> None:
        # Engines report in their own order. The last column's designator is
        # moved to the end of the stream, after every label.
        rows = [_row(d, left=self.FIRST_CENTRE + i * self.PITCH - 20,
                     top=self.DESIGNATOR_TOP, width=40, height=self.DESIGNATOR_HEIGHT)
                for i, (d, _) in enumerate(self.COLUMNS[:-1])]
        rows += [_row(l, left=self.FIRST_CENTRE + i * self.PITCH - 25,
                      top=self.LABEL_TOP, width=50, height=self.LABEL_HEIGHT)
                 for i, (_, l) in enumerate(self.COLUMNS)]
        rows.append(_row(self.COLUMNS[-1][0],
                         left=self.FIRST_CENTRE + 5 * self.PITCH - 20,
                         top=self.DESIGNATOR_TOP, width=40, height=self.DESIGNATOR_HEIGHT))

        layout = layout_from(words_from_tsv(_tsv(*rows)).detections,
                             image_width=self.IMAGE_WIDTH)
        assert sorted(a.text for a in layout.associations) == sorted(
            f"{designator} | {label}" for designator, label in self.COLUMNS
        )

    def test_a_flat_label_parses_but_pairs_with_nothing(self) -> None:
        # The asset label's shape: three lines, no diagram, and no invention
        # either. The lines are separated vertically so the line rule keeps
        # them apart, which is what makes this a flat label rather than a
        # diagram.
        blob = _tsv(
            _row("PLATE SERIES", left=84, top=93, width=201, height=28),
            _row("UNIT REVISION", left=83, top=151, width=179, height=22),
            _row("ZZ-9", left=83, top=201, width=60, height=25),
        )
        layout = layout_from(words_from_tsv(blob).detections, image_width=900)
        assert [line.text for line in layout.lines] == [
            "PLATE SERIES",
            "UNIT REVISION",
            "ZZ-9",
        ]
        assert layout.associations == ()

    def test_the_words_of_a_line_are_rejoined_by_the_geometry_not_the_adapter(self) -> None:
        # Three words of one printed line, reported as three detections. The
        # adapter must not join them -- it has no view of the reading order --
        # and the geometry must.
        blob = _tsv(
            _row("UNIT", left=83, top=201, width=60, height=25),
            _row("REVISION", left=150, top=201, width=120, height=25),
            _row("ZZ-9", left=277, top=201, width=60, height=25),
        )
        parsed = words_from_tsv(blob)
        assert [d.text for d in parsed.detections] == ["UNIT", "REVISION", "ZZ-9"]
        layout = layout_from(parsed.detections, image_width=900)
        assert [line.text for line in layout.lines] == ["UNIT REVISION ZZ-9"]
