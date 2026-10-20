"""Tests for turning read results into indexable chunks.

Chunk boundaries decide recall before any retrieval code exists, and they fail
quietly: a value split across two chunks is simply never returned, and the
search still looks like it worked. So the properties asserted here are the ones
whose violation would be invisible at query time -- that no text is lost, that
unmatchable text is dropped rather than indexed, and that an unread file cannot
reach the index at all.
"""

from __future__ import annotations

import pytest
from verity.engine.chunking import chunks_for, chunks_for_all
from verity.engine.config import MAX_CHUNK_CHARS
from verity.engine.models import Chunk, MediaType, ReadResult, ReadStatus, TextSpan
from verity.engine.normalisation import normalise


def span(text: str, location: str = "line 1", ordinal: int = 1) -> TextSpan:
    return TextSpan(text=text, location=location, ordinal=ordinal)


def readable(*spans: TextSpan, rel_path: str = "specs/tq40.pdf") -> ReadResult:
    return ReadResult.of(rel_path, MediaType.PDF, list(spans))


class TestUnusableFiles:
    """``is_usable`` is the single gate for "may this be cited"."""

    @pytest.mark.parametrize(
        "status",
        [
            ReadStatus.ENCRYPTED,
            ReadStatus.UNREADABLE,
            ReadStatus.UNSUPPORTED,
            ReadStatus.VISION_UNAVAILABLE,
            ReadStatus.EMPTY,
        ],
    )
    def test_a_file_that_was_not_read_yields_no_chunks(self, status: ReadStatus) -> None:
        """The encrypted agreement must not reach the index by any route.

        Indexing it would make question 10's refusal depend on a retrieval
        accident rather than on the file being genuinely unreadable.
        """
        result = ReadResult.failure("vendor/agreement.pdf", MediaType.PDF, status, "detail")

        assert chunks_for(result) == ()

    def test_status_is_all_that_matters_not_absent_spans(self) -> None:
        """A failure carrying spans is still a failure.

        Readers do not construct this, but nothing in the type system prevents
        it, and a gate that only checked ``spans`` would let it through.
        """
        result = ReadResult(
            rel_path="a.pdf",
            media_type=MediaType.PDF,
            status=ReadStatus.ENCRYPTED,
            spans=(span("secret 6412"),),
        )

        assert chunks_for(result) == ()


class TestContentFiltering:
    def test_a_punctuation_only_span_is_not_indexed(self) -> None:
        """It can never match a query, so it is only noise in the index."""
        result = readable(span("-----"), span("=== === ==="))

        assert chunks_for(result) == ()

    def test_a_rule_of_dashes_is_not_content_but_a_value_is(self) -> None:
        """The test runs on the normalised form, which is why both are true.

        ``-----`` normalises to nothing. ``4.3.2`` normalises to ``432``, which
        is alphanumeric, so it survives -- even though the raw text of the
        second contains only one alphanumeric character in five.
        """
        result = readable(span("-----"), span("4.3.2"))

        chunks = chunks_for(result)

        assert len(chunks) == 1
        assert chunks[0].text == "4.3.2"

    def test_a_span_of_only_spaces_is_not_indexed(self) -> None:
        result = readable(span("   "), span("\t\n"))

        assert chunks_for(result) == ()

    def test_a_degree_only_span_is_not_indexed(self) -> None:
        """``°`` is deleted by normalisation, so a span of them is not content."""
        result = readable(span("°°"))

        assert chunks_for(result) == ()

    def test_content_after_filtering_survives(self) -> None:
        result = readable(span("-----"), span("Maximum junction temperature 94 C"))

        assert [chunk.text for chunk in chunks_for(result)] == [
            "Maximum junction temperature 94 C"
        ]


class TestChunkShape:
    def test_short_spans_group_into_one_chunk(self) -> None:
        """Two short spans that fit together become one chunk, not two."""
        result = readable(
            span("Maximum junction temperature 94 C", "page 3 line 4", 4),
            span("Board power 410 W", "page 3 line 5", 5),
        )

        chunks = chunks_for(result)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == "Maximum junction temperature 94 C\nBoard power 410 W"
        assert chunk.location == "page 3 line 4; page 3 line 5"
        assert chunk.ordinal == 4
        assert chunk.media_type is MediaType.PDF

    def test_provenance_is_carried_onto_every_chunk(self) -> None:
        """Retrieval returns chunks from across the corpus with nothing else
        tying them to a file, so the citation string travels with the text."""
        result = readable(span("value 94", "page 3 line 4", 4))

        chunk = chunks_for(result)[0]

        assert chunk.rel_path == "specs/tq40.pdf"
        assert chunk.location == "page 3 line 4"
        assert chunk.ordinal == 4
        assert chunk.media_type is MediaType.PDF

    def test_text_is_stored_verbatim(self) -> None:
        """Normalisation is a comparison form, never a display form.

        The answer reported to the grader has to be the value as the corpus
        spells it, so the stored text must not be the folded version.
        """
        result = readable(span("Maximum junction temperature: 94 °C"))

        chunk = chunks_for(result)[0]

        assert chunk.text == "Maximum junction temperature: 94 °C"
        assert chunk.normalised == "MAXIMUMJUNCTIONTEMPERATURE:94C"

    def test_grouped_chunk_ordinal_is_first_span_ordinal(self) -> None:
        """The chunk's ordinal is the first span's, so within-file ordering
        stays recoverable even when spans are grouped."""
        result = readable(
            span("alpha 1", "line 1", 1),
            span("beta 2", "line 2", 2),
            span("gamma 3", "line 3", 3),
        )

        chunks = chunks_for(result)

        assert len(chunks) == 1
        assert chunks[0].ordinal == 1
        # Every source location is named, none implied by a range.
        assert chunks[0].location == "line 1; line 2; line 3"


class TestGrouping:
    """Consecutive spans that fit are grouped; oversized spans stand alone."""

    def test_group_stops_before_exceeding_max_chunk_chars(self) -> None:
        """When the next span would overflow the cap, it starts a new chunk."""
        # Span 1: 1000 chars, span 2: 1000 chars. Together with newline = 2001 > 2000.
        span1 = "a " * 500  # 1000 chars
        span2 = "b " * 500  # 1000 chars
        span3 = "c " * 300  # 600 chars; span2+span3 = 1601 <= 2000, grouped
        result = readable(
            span(span1, "line 1", 1),
            span(span2, "line 2", 2),
            span(span3, "line 3", 3),
        )

        chunks = chunks_for(result)

        # span1+span2 = 2001 chars > 2000, so they don't fit together.
        # span1 is alone, span2+span3 grouped.
        assert len(chunks) == 2
        assert chunks[0].text == span1
        assert chunks[0].location == "line 1"
        assert chunks[1].text == span2 + "\n" + span3
        assert chunks[1].location == "line 2; line 3"

    def test_oversized_span_is_not_grouped_with_neighbors(self) -> None:
        """An oversized span is split on its own and never merged with
        adjacent spans, even if they would fit."""
        oversized = "word " * 600  # 3000 chars > MAX_CHUNK_CHARS
        small = "small value 94"
        result = readable(
            span(small, "line 1", 1),
            span(oversized, "line 2", 2),
            span(small, "line 3", 3),
        )

        chunks = chunks_for(result)

        # line 1 stands alone, line 2 is split into 2 chunks, line 3 stands alone.
        assert len(chunks) == 4
        assert chunks[0].text == small
        assert chunks[0].location == "line 1"
        assert chunks[1].location.startswith("line 2 part 1 of 2")
        assert chunks[2].location.startswith("line 2 part 2 of 2")
        assert chunks[3].text == small
        assert chunks[3].location == "line 3"

    def test_table_header_and_first_row_kept_together(self) -> None:
        """A table header span and its first data row, when they fit, are
        grouped into one chunk so labels stay with values."""
        from verity.engine.models import ReadResult

        result = ReadResult.of(
            "support/bug_database.csv",
            MediaType.CSV,
            [
                span("ticket | component | status | fixed_in", "header", 0),
                span(
                    "ticket=ORR-1847 | component=power | status=closed | fixed_in=4.3.2",
                    "row 2",
                    1,
                ),
            ],
        )

        chunks = chunks_for(result)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert "fixed_in=4.3.2" in chunk.text
        assert chunk.location == "header; row 2"
        assert chunk.ordinal == 0
        assert chunk.media_type is MediaType.CSV

    def test_punctuation_only_span_is_dropped_from_group(self) -> None:
        """A punctuation-only span contributes nothing, even inside a group."""
        result = readable(
            span("----", "line 1", 1),
            span("real value 94", "line 2", 2),
        )

        chunks = chunks_for(result)

        assert len(chunks) == 1
        assert chunks[0].text == "real value 94"
        assert chunks[0].location == "line 2"

    def test_all_punctuation_spans_yield_no_chunks(self) -> None:
        """A file of only punctuation spans produces no chunks at all."""
        result = readable(
            span("----", "line 1", 1),
            span("=== ===", "line 2", 2),
        )

        assert chunks_for(result) == ()

    def test_no_text_loss_or_duplication_in_grouped_chunk(self) -> None:
        """Every span's text appears exactly once in the grouped chunk."""
        result = readable(
            span("alpha beta gamma", "line 1", 1),
            span("delta epsilon zeta", "line 2", 2),
            span("eta theta iota", "line 3", 3),
        )

        chunks = chunks_for(result)

        assert len(chunks) == 1
        combined = chunks[0].text
        assert "alpha beta gamma" in combined
        assert "delta epsilon zeta" in combined
        assert "eta theta iota" in combined
        # No duplication: each distinct word appears once.
        assert combined.count("alpha") == 1
        assert combined.count("epsilon") == 1
        assert combined.count("iota") == 1

    def test_group_preserves_rel_path_and_media_type(self) -> None:
        """The citation string and media type travel with the grouped chunk."""
        from verity.engine.models import ReadResult

        result = ReadResult.of(
            "specs/datasheet.pdf",
            MediaType.PDF,
            [
                span("value 94", "page 1 line 1", 1),
                span("value 95", "page 1 line 2", 2),
            ],
        )

        chunks = chunks_for(result)

        assert len(chunks) == 1
        assert chunks[0].rel_path == "specs/datasheet.pdf"
        assert chunks[0].media_type is MediaType.PDF


class TestSplitting:
    """A chunk too large to retrieve well is cut, and nothing may be lost."""

    @staticmethod
    def long_text(words: int = 900) -> str:
        return " ".join(f"value{i:04d}" for i in range(words))

    def test_a_long_span_becomes_several_chunks(self) -> None:
        text = self.long_text()
        assert len(text) > MAX_CHUNK_CHARS

        chunks = chunks_for(readable(span(text)))

        assert len(chunks) > 1
        assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)

    def test_splitting_loses_no_characters(self) -> None:
        """The property that matters. A dropped character could be the answer.

        Compared on the normalised form because the split trims whitespace at
        the boundary, and whitespace is not part of any value.
        """
        text = self.long_text()

        rejoined = "".join(chunk.normalised for chunk in chunks_for(readable(span(text))))

        assert rejoined == normalise(text)

    def test_each_piece_says_which_piece_it_is(self) -> None:
        """Otherwise a diagnostic pointing at "line 42" is ambiguous."""
        chunks = chunks_for(readable(span(self.long_text(), "line 42", 42)))

        locations = [chunk.location for chunk in chunks]

        assert locations[0] == f"line 42 part 1 of {len(chunks)}"
        assert locations[-1] == f"line 42 part {len(chunks)} of {len(chunks)}"
        assert all(chunk.ordinal == 42 for chunk in chunks)

    def test_a_scalar_location_is_left_alone(self) -> None:
        """A span that fits keeps its plain location -- no noise added."""
        chunks = chunks_for(readable(span("small value 94", "line 42", 42)))

        assert chunks[0].location == "line 42"

    def test_text_with_no_spaces_is_still_split(self) -> None:
        """A minified file or a base64 blob has no break to look back for.

        The cut then falls at the limit rather than scanning the whole prefix
        for a space that is not there.
        """
        text = "A" * (MAX_CHUNK_CHARS * 2 + 500)

        chunks = chunks_for(readable(span(text)))

        assert len(chunks) == 3
        assert "".join(chunk.text for chunk in chunks) == text

    def test_a_value_is_not_split_across_a_boundary_when_a_space_is_near(self) -> None:
        """Values are short and space-delimited, so pulling the cut back to a
        space keeps a whole value inside one chunk."""
        text = ("filler " * 400) + "SERIAL-ORR-2214-B " + ("filler " * 400)

        chunks = chunks_for(readable(span(text)))
        holders = [chunk for chunk in chunks if "SERIAL-ORR-2214-B" in chunk.text]

        assert len(holders) == 1


class TestSentenceSplitting:
    """Oversized spans are cut at sentence boundaries before whitespace."""

    def test_sentence_boundary_preferred_over_whitespace(self) -> None:
        """When both a sentence end and a space are near the cap, the sentence
        end wins so a value is not torn mid-sentence."""
        text = "x " * 900  # 1800 chars
        text += "alpha beta gamma. delta epsilon zeta. "  # sentence ends in lookback
        text += "y " * 200  # total > MAX_CHUNK_CHARS, forces a split

        chunks = chunks_for(readable(span(text)))
        assert len(chunks) > 1
        # The last sentence end in the lookback wins, so the chunk ends with
        # "zeta." not "gamma." -- this maximizes chunk size while staying at a
        # sentence boundary.
        assert chunks[0].text.rstrip().endswith("zeta.")

    def test_decimal_number_is_not_a_sentence_boundary(self) -> None:
        """``94.5`` is a measurement, not the end of a sentence."""
        text = "x " * 900  # 1800 chars
        text += "temperature 94.5 degrees. "  # decimal inside lookback
        text += "z " * 200  # total > MAX_CHUNK_CHARS, forces a split

        chunks = chunks_for(readable(span(text)))
        # The decimal should stay inside one chunk with its unit.
        holders = [chunk for chunk in chunks if "94.5" in chunk.text]
        assert len(holders) == 1
        assert any("degrees" in chunk.text for chunk in holders)

    def test_abbreviation_does_not_crash(self) -> None:
        """``e.g.`` mid-text must not raise, even if it looks like a sentence end."""
        text = ("filler " * 300) + "for example e.g. a value. " + ("next " * 400)

        chunks = chunks_for(readable(span(text)))
        assert len(chunks) > 1
        # No characters lost.
        rejoined = "".join(chunk.normalised for chunk in chunks)
        assert rejoined == normalise(text)

    def test_spanish_sentence_boundary_splits(self) -> None:
        """Non-English sentences split at ``. ? !`` without an uppercase lookup."""
        text = ("filler " * 300) + "¿Cómo está? Está bien. ¡Muy bien! " + ("x " * 400)

        chunks = chunks_for(readable(span(text)))
        assert len(chunks) > 1
        # Each sentence should be intact in some chunk.
        full = " ".join(chunk.text for chunk in chunks)
        assert "¿Cómo está?" in full
        assert "Está bien." in full
        assert "¡Muy bien!" in full

    def test_no_chunk_exceeds_max_chunk_chars(self) -> None:
        """The hard ceiling is never violated, whatever the break."""
        text = "word " * 3000

        chunks = chunks_for(readable(span(text)))
        assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)

    def test_no_characters_lost_with_sentence_splitting(self) -> None:
        """The normalised form is preserved across sentence-aware cuts."""
        text = "alpha beta gamma. delta epsilon zeta. " + "value " * 500

        chunks = chunks_for(readable(span(text)))
        rejoined = "".join(chunk.normalised for chunk in chunks)
        assert rejoined == normalise(text)


class TestWholeCorpus:
    def test_order_is_preserved(self) -> None:
        results = [
            readable(span("first file value"), rel_path="a.txt"),
            readable(span("second file value"), rel_path="b.txt"),
        ]

        chunks = chunks_for_all(results)

        assert [chunk.rel_path for chunk in chunks] == ["a.txt", "b.txt"]

    def test_unreadable_files_contribute_nothing_without_shifting_the_rest(self) -> None:
        results = [
            readable(span("readable one"), rel_path="a.txt"),
            ReadResult.failure("b.pdf", MediaType.PDF, ReadStatus.ENCRYPTED, "locked"),
            readable(span("readable two"), rel_path="c.txt"),
        ]

        chunks = chunks_for_all(results)

        assert [chunk.rel_path for chunk in chunks] == ["a.txt", "c.txt"]

    def test_an_empty_corpus_yields_nothing(self) -> None:
        assert chunks_for_all([]) == ()

    def test_chunks_are_frozen(self) -> None:
        chunk = chunks_for(readable(span("value 94")))[0]

        with pytest.raises(Exception):
            chunk.text = "tampered"  # type: ignore[misc]
