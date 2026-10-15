"""Tests for the plain-text reader and the shared decode path.

Text files look like the easy case, and that is exactly why they need tests:
a reader that mangles line endings or mis-decodes one byte still returns
*almost* the right text, which is the hardest kind of wrong to notice.
"""

from __future__ import annotations

from verity.engine.ingestion.readers import read_file, source
from verity.engine.models import MediaType, ReadStatus


class TestEncoding:
    def test_utf8_is_preferred(self) -> None:
        """UTF-8 must win over the single-byte fallbacks, with or without a mark.

        ``utf-8-sig`` leads ``TEXT_ENCODINGS``, so it is the codec that answers
        here even though there is no mark to strip. That is the point of the
        ordering: ``utf-8-sig`` decodes BOM-less UTF-8 byte-for-byte the same
        way ``utf-8`` does, so it *dominates* the plain codec instead of
        competing with it -- there is no input where plain ``utf-8`` would be
        the better answer, and one (a mark) where it is the worse one.

        The property under test is the decoded text. Had these bytes fallen
        through to cp1252, ``naïve`` would have come back as ``naÃ¯ve``.
        """
        text, encoding = source.decode_text("naïve — 94 °C".encode("utf-8"))

        assert encoding == "utf-8-sig"
        assert text == "naïve — 94 °C"

    def test_utf8_bom_is_stripped(self) -> None:
        """A BOM left in place would glue itself to the first value on line 1."""
        text, encoding = source.decode_text("ORR-FAN-2214-B\n".encode("utf-8-sig"))

        assert encoding == "utf-8-sig"
        assert text.startswith("ORR-FAN-2214-B")

    def test_windows_encoding_is_used_when_utf8_fails(self) -> None:
        raw = "Maximum junction temperature — 94 °C".encode("cp1252")

        text, encoding = source.decode_text(raw)

        assert encoding == "cp1252"
        assert "94" in text

    def test_undecodable_bytes_never_raise(self) -> None:
        """Losing a file to a stray byte is worse than keeping it as mojibake.

        Mojibake still tokenises, so a value elsewhere on the line can still be
        matched. An exception would discard the whole file.
        """
        text, _ = source.decode_text(bytes(range(256)))

        assert isinstance(text, str)
        assert len(text) == 256


class TestLineSpans:
    def test_each_line_becomes_one_span_with_its_line_number(self, build_corpus) -> None:
        files = build_corpus({"notes.txt": "first\nsecond\nthird\n"})

        result = read_file(files["notes.txt"])

        assert [(span.location, span.text) for span in result.spans] == [
            ("line 1", "first"),
            ("line 2", "second"),
            ("line 3", "third"),
        ]

    def test_crlf_line_endings_do_not_leave_carriage_returns(self, build_corpus) -> None:
        """Windows line endings are common and would otherwise corrupt values.

        Splitting on ``\\n`` alone would leave a trailing ``\\r`` on every line,
        so a value at the end of a line would carry an invisible extra
        character into every comparison made against it.
        """
        files = build_corpus({"bug_database.csv": "a,b,c\r\n1,2,3\r\n"})

        result = read_file(files["bug_database.csv"])

        assert all("\r" not in span.text for span in result.spans)

    def test_blank_lines_are_dropped(self, build_corpus) -> None:
        files = build_corpus({"notes.txt": "alpha\n\n\nbeta\n"})

        result = read_file(files["notes.txt"])

        assert [span.text for span in result.spans] == ["alpha", "beta"]

    def test_line_numbers_survive_dropped_blank_lines(self, build_corpus) -> None:
        """The number must still point at the real line in the real file."""
        files = build_corpus({"notes.txt": "alpha\n\n\nbeta\n"})

        result = read_file(files["notes.txt"])

        assert [span.location for span in result.spans] == ["line 1", "line 4"]

    def test_a_file_of_only_blank_lines_is_empty(self, build_corpus) -> None:
        files = build_corpus({"notes.txt": "\n\n   \n"})

        result = read_file(files["notes.txt"])

        assert result.status is ReadStatus.EMPTY

    def test_python_source_keeps_the_comment_and_the_constant_apart(
        self, build_corpus
    ) -> None:
        """The sample corpus needs these separated, not merely present.

        The comment names the *old* value and the assignment carries the new
        one. Both are real text in the same file; only line-level provenance
        lets a later stage tell which is which.
        """
        files = build_corpus(
            {
                "service.py": (
                    "# Raised from 60 after the backlog incident.\n"
                    "DEFAULT_BATCH_TIMEOUT_S = 180\n"
                )
            }
        )

        result = read_file(files["service.py"])
        by_location = {span.location: span.text for span in result.spans}

        assert by_location["line 1"].startswith("#")
        assert "60" in by_location["line 1"]
        assert by_location["line 2"] == "DEFAULT_BATCH_TIMEOUT_S = 180"

    def test_log_lines_keep_their_timestamps(self, build_corpus) -> None:
        files = build_corpus(
            {"run.log": "2026-09-02T03:14:41Z node meridian: ERROR E7731: throttle engaged\n"}
        )

        result = read_file(files["run.log"])

        assert "E7731" in result.text
        assert result.spans[0].location == "line 1"


class TestTruncation:
    def test_an_oversized_file_is_truncated_and_says_so(self, build_corpus, monkeypatch) -> None:
        """Silent truncation would look like a file that simply ends early."""
        monkeypatch.setattr(source, "MAX_TEXT_BYTES", 10)
        files = build_corpus({"big.txt": b"x" * 5000})

        result = read_file(files["big.txt"])

        assert result.status is ReadStatus.OK
        assert "truncated" in result.detail
        assert result.char_count < 5000

    def test_a_normal_file_reports_no_detail(self, build_corpus) -> None:
        files = build_corpus({"notes.txt": "ordinary\n"})

        assert read_file(files["notes.txt"]).detail == ""


class TestMediaTypes:
    def test_txt_log_and_code_all_route_to_the_text_reader(self, build_corpus) -> None:
        files = build_corpus({"a.txt": "x\n", "b.log": "y\n", "c.py": "z = 1\n"})

        assert read_file(files["a.txt"]).media_type is MediaType.TEXT
        assert read_file(files["b.log"]).media_type is MediaType.LOG
        assert read_file(files["c.py"]).media_type is MediaType.CODE
