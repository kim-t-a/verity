"""Contract tests that every reader must satisfy, regardless of format.

These are the "does the pipeline survive the corpus" tests. The format-specific
suites check whether a reader recovers the *right* text; this one checks that no
reader can take the run down, and that the distinction between "read it" and
"did not read it" cannot be fudged.
"""

from __future__ import annotations

import pytest
from document_fixtures import docx_bytes, pdf_bytes, png_bytes, xlsx_bytes
from ingestion.readers import read_file, reader_for
from models import MediaType, ReadResult, ReadStatus, TextSpan


class TestDispatch:
    def test_every_textual_format_has_a_reader(self) -> None:
        for media_type in (
            MediaType.PDF,
            MediaType.DOCX,
            MediaType.XLSX,
            MediaType.CSV,
            MediaType.TEXT,
            MediaType.LOG,
            MediaType.CODE,
        ):
            assert reader_for(media_type) is not None, media_type

    def test_images_report_themselves_unreadable_rather_than_silent(
        self, build_corpus
    ) -> None:
        """An image is not skipped quietly; it says it cannot be read yet.

        The rule is that image support must never be silently absent. The
        status has to name the *gap* -- a recognised image with no vision
        backend -- rather than dissolving into the same ``UNSUPPORTED`` an
        unknown binary gets, because only one of those is a requirement this
        project has to close.
        """
        files = build_corpus({"specs/pinout.png": png_bytes()})

        result = read_file(files["specs/pinout.png"])

        assert result.status is ReadStatus.VISION_UNAVAILABLE
        assert result.is_usable is False
        assert result.spans == ()

    def test_unknown_binary_type_is_unsupported_not_guessed(self, build_corpus) -> None:
        files = build_corpus({"vendor/capture.dat": bytes(range(256)) * 4})

        result = read_file(files["vendor/capture.dat"])

        assert result.status is ReadStatus.UNSUPPORTED
        assert result.is_usable is False


class TestReadersNeverRaise:
    """A reader that throws during indexing costs the whole corpus, not one file."""

    @pytest.mark.parametrize(
        "rel_path, payload",
        [
            ("broken.docx", b"PK\x03\x04 this is not really a zip"),
            ("broken.xlsx", b"PK\x03\x04 neither is this"),
            ("broken.pdf", b"%PDF-1.4\nbut no objects follow"),
            ("truncated.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog"),
            ("empty.pdf", b""),
            ("empty.txt", b""),
            ("empty.csv", b""),
            ("binary.txt", bytes(range(256))),
            ("nul.csv", b"a,b\n\x00\x01,\x02\x03\n"),
        ],
    )
    def test_malformed_input_yields_a_result_not_an_exception(
        self, build_corpus, rel_path: str, payload: bytes
    ) -> None:
        files = build_corpus({rel_path: payload})

        result = read_file(files[rel_path])

        assert isinstance(result, ReadResult)
        assert result.rel_path == rel_path
        # Never OK: none of these contains recoverable text.
        assert result.is_usable is False

    def test_a_corrupt_container_names_its_failure(self, build_corpus) -> None:
        files = build_corpus({"planning/roadmap.docx": b"PK\x03\x04 truncated"})

        result = read_file(files["planning/roadmap.docx"])

        assert result.status is ReadStatus.UNREADABLE
        assert result.detail


class TestReadResultSemantics:
    def test_ok_requires_at_least_one_non_blank_span(self) -> None:
        result = ReadResult.of("a.txt", MediaType.TEXT, [TextSpan("   ", "line 1", 1)])

        assert result.status is ReadStatus.EMPTY
        assert result.is_usable is False

    def test_a_document_with_no_text_is_empty_not_unreadable(self) -> None:
        """``EMPTY`` and ``UNREADABLE`` must stay distinct.

        Both are unusable, but they mean different things -- "this document
        says nothing" versus "we could not find out what it says". Collapsing
        them would hide a broken reader behind a plausible-looking empty file.
        """
        blank = ReadResult.of("a.txt", MediaType.TEXT, [])
        broken = ReadResult.failure("b.txt", MediaType.TEXT, ReadStatus.UNREADABLE, "boom")

        assert blank.status is ReadStatus.EMPTY
        assert broken.status is ReadStatus.UNREADABLE
        assert blank.status != broken.status

    def test_blank_spans_are_dropped_from_a_successful_read(self) -> None:
        spans = [
            TextSpan("real content", "line 1", 1),
            TextSpan("   ", "line 2", 2),
            TextSpan("more content", "line 3", 3),
        ]

        result = ReadResult.of("a.txt", MediaType.TEXT, spans)

        assert [span.text for span in result.spans] == ["real content", "more content"]

    def test_text_joins_spans_with_newlines(self) -> None:
        spans = [TextSpan("one", "line 1", 1), TextSpan("two", "line 2", 2)]

        assert ReadResult.of("a.txt", MediaType.TEXT, spans).text == "one\ntwo"

    def test_ordinals_are_stable_within_a_file(self, build_corpus) -> None:
        files = build_corpus({"notes.txt": "alpha\nbeta\ngamma\n"})

        result = read_file(files["notes.txt"])

        assert [span.ordinal for span in result.spans] == sorted(
            span.ordinal for span in result.spans
        )


class TestEveryFormatReportsAStatus:
    """A sweep over one file of each format, asserting none escapes the contract."""

    def test_all_formats_produce_a_well_formed_result(self, build_corpus) -> None:
        files = build_corpus(
            {
                "a.txt": "hello\n",
                "b.log": "2026-01-01 INFO started\n",
                "c.py": "X = 1\n",
                "d.csv": "a,b,c\n1,2,3\n",
                "e.docx": docx_bytes("<w:p><w:r><w:t>word</w:t></w:r></w:p>"),
                "f.xlsx": xlsx_bytes([("Sheet1", [["a", "b", "c"], [1, 2, 3]])]),
                "g.pdf": pdf_bytes(text="pdf text"),
                "h.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
            }
        )

        for rel_path, discovered in sorted(files.items()):
            result = read_file(discovered)
            assert isinstance(result, ReadResult), rel_path
            assert result.rel_path == rel_path
            assert result.media_type is discovered.media_type
            assert isinstance(result.status, ReadStatus)
            if result.is_usable:
                assert result.spans, rel_path
                assert result.char_count > 0
