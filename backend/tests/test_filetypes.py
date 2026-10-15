"""Unit tests for format detection from leading bytes and filename."""

from __future__ import annotations

import pytest
from verity.engine.ingestion.filetypes import magic_type, sniff, type_from_extension
from verity.engine.models import MediaType


class TestMagicType:
    def test_recognises_pdf(self) -> None:
        assert magic_type(b"%PDF-1.7\n...") is MediaType.PDF

    def test_recognises_png(self) -> None:
        assert magic_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) is MediaType.PNG

    def test_recognises_jpeg(self) -> None:
        assert magic_type(b"\xff\xd8\xff\xe0" + b"\x00" * 8) is MediaType.JPG

    def test_recognises_zip_container(self) -> None:
        assert magic_type(b"PK\x03\x04" + b"\x00" * 8) is MediaType.ZIP

    def test_returns_none_for_plain_text(self) -> None:
        assert magic_type(b"DEFAULT_BATCH_TIMEOUT_S = 180\n") is None

    def test_returns_none_for_empty_input(self) -> None:
        assert magic_type(b"") is None


class TestTypeFromExtension:
    @pytest.mark.parametrize(
        ("extension", "expected"),
        [
            (".pdf", MediaType.PDF),
            (".docx", MediaType.DOCX),
            (".xlsx", MediaType.XLSX),
            (".csv", MediaType.CSV),
            (".txt", MediaType.TEXT),
            (".log", MediaType.LOG),
            (".py", MediaType.CODE),
            (".png", MediaType.PNG),
            (".jpg", MediaType.JPG),
            (".jpeg", MediaType.JPG),
        ],
    )
    def test_known_extensions(self, extension: str, expected: MediaType) -> None:
        assert type_from_extension(extension) is expected

    def test_extension_matching_is_case_insensitive(self) -> None:
        assert type_from_extension(".PDF") is MediaType.PDF

    def test_unknown_extension(self) -> None:
        assert type_from_extension(".dat") is MediaType.UNKNOWN


class TestSniff:
    def test_pdf_bytes_win_over_a_text_extension(self) -> None:
        """A PDF named .txt is still a PDF: the bytes are evidence, the name is a claim."""
        verdict = sniff(b"%PDF-1.4\n", ".txt")

        assert verdict.media_type is MediaType.PDF
        assert verdict.extension_type is MediaType.TEXT
        assert verdict.magic_type is MediaType.PDF
        assert verdict.agrees is False

    def test_zip_container_is_resolved_by_extension_for_docx(self) -> None:
        verdict = sniff(b"PK\x03\x04" + b"\x00" * 8, ".docx")

        assert verdict.media_type is MediaType.DOCX
        assert verdict.magic_type is MediaType.ZIP
        assert verdict.agrees is True

    def test_zip_container_is_resolved_by_extension_for_xlsx(self) -> None:
        assert sniff(b"PK\x03\x04" + b"\x00" * 8, ".xlsx").media_type is MediaType.XLSX

    def test_zip_container_with_unknown_extension_stays_zip(self) -> None:
        """We can tell it is a ZIP, but not which office format it holds."""
        verdict = sniff(b"PK\x03\x04" + b"\x00" * 8, ".dat")

        assert verdict.media_type is MediaType.ZIP
        assert verdict.extension_type is MediaType.UNKNOWN
        assert verdict.agrees is False

    def test_plain_text_trusts_the_name(self) -> None:
        """Text has no signature, so absence of evidence is not contradiction."""
        verdict = sniff(b"part,note\nORR-FAN-2214-B,dual rotor\n", ".csv")

        assert verdict.media_type is MediaType.CSV
        assert verdict.magic_type is None
        assert verdict.agrees is True

    def test_unknown_binary_stays_unknown(self) -> None:
        """A file we cannot identify must be reported as unknown, never guessed at."""
        verdict = sniff(bytes(range(256)) * 4, ".dat")

        assert verdict.media_type is MediaType.UNKNOWN
        assert verdict.magic_type is None

    def test_png_bytes_in_a_dat_file_are_still_png(self) -> None:
        assert sniff(b"\x89PNG\r\n\x1a\n", ".dat").media_type is MediaType.PNG
