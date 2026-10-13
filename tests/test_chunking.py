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
from chunking import chunks_for, chunks_for_all
from config import MAX_CHUNK_CHARS
from models import Chunk, MediaType, ReadResult, ReadStatus, TextSpan
from normalisation import normalise


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
    def test_one_chunk_per_span_when_the_span_fits(self) -> None:
        result = readable(
            span("Maximum junction temperature 94 C", "page 3 line 4", 4),
            span("Board power 410 W", "page 3 line 5", 5),
        )

        chunks = chunks_for(result)

        assert [chunk.text for chunk in chunks] == [
            "Maximum junction temperature 94 C",
            "Board power 410 W",
        ]

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

    def test_ordinal_survives_so_order_within_a_file_is_recoverable(self) -> None:
        result = readable(
            span("alpha 1", "line 1", 1),
            span("beta 2", "line 2", 2),
            span("gamma 3", "line 3", 3),
        )

        assert [chunk.ordinal for chunk in chunks_for(result)] == [1, 2, 3]


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
