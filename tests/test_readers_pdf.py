"""Tests for the PDF reader, with the encryption trap as the centrepiece.

The encrypted file is not a curiosity. The sample corpus hides a graded
refusal behind one: a question whose correct answer is *empty* because the
only file holding the value cannot be opened. A reader that produced
plausible-looking output from ciphertext would score zero while reporting
success, so the encryption check is asserted from several directions.
"""

from __future__ import annotations

from document_fixtures import pdf_bytes
from ingestion.readers import pdf, read_file
from models import ReadStatus


class TestEncryption:
    def test_an_encrypted_pdf_is_detected_before_parsing(self) -> None:
        assert pdf.is_encrypted(pdf_bytes(text="secret", encrypted=True)) is True

    def test_an_ordinary_pdf_is_not_flagged(self) -> None:
        assert pdf.is_encrypted(pdf_bytes(text="public")) is False

    def test_an_encrypted_file_reports_encrypted_not_empty(self, build_corpus) -> None:
        """The distinction that decides a graded question."""
        files = build_corpus({"secret.pdf": pdf_bytes(text="prices", encrypted=True)})

        result = read_file(files["secret.pdf"])

        assert result.status is ReadStatus.ENCRYPTED
        assert result.is_usable is False
        assert result.spans == ()

    def test_an_encrypted_file_yields_no_text_at_all(self, build_corpus) -> None:
        """Not one byte of the ciphertext may be presented as content."""
        files = build_corpus({"secret.pdf": pdf_bytes(text="6412", encrypted=True)})

        result = read_file(files["secret.pdf"])

        assert result.text == ""
        assert "6412" not in result.text

    def test_the_refusal_says_it_will_not_decrypt(self, build_corpus) -> None:
        files = build_corpus({"secret.pdf": pdf_bytes(text="x", encrypted=True)})

        assert "not be decrypted" in read_file(files["secret.pdf"]).detail

    def test_the_real_encrypted_corpus_file_is_refused(self) -> None:
        """The actual AES-256 file from the sample corpus."""
        from conftest import CORPUS_DIR
        from ingestion import walker

        discovered = next(
            f
            for f in walker.discover(CORPUS_DIR).files
            if f.rel_path == "vendor/supplier_agreement_ENCRYPTED.pdf"
        )

        result = read_file(discovered)

        assert result.status is ReadStatus.ENCRYPTED
        assert result.is_usable is False


class TestExtraction:
    def test_text_is_recovered_from_a_plain_pdf(self, build_corpus) -> None:
        files = build_corpus({"spec.pdf": pdf_bytes(text="Maximum junction temperature 94 C")})

        result = read_file(files["spec.pdf"])

        assert "Maximum junction temperature 94 C" in result.text

    def test_flate_compressed_content_is_decompressed(self, build_corpus) -> None:
        """Compression is the norm; an uncompressed-only reader gets nothing."""
        files = build_corpus(
            {"spec.pdf": pdf_bytes(text="Board power 410 W", compressed=True)}
        )

        result = read_file(files["spec.pdf"])

        assert "Board power 410 W" in result.text

    def test_compressed_and_uncompressed_agree(self, build_corpus) -> None:
        files = build_corpus(
            {
                "plain.pdf": pdf_bytes(text="same text"),
                "zipped.pdf": pdf_bytes(text="same text", compressed=True),
            }
        )

        assert read_file(files["plain.pdf"]).text == read_file(files["zipped.pdf"]).text

    def test_pages_are_labelled(self, build_corpus) -> None:
        files = build_corpus({"spec.pdf": pdf_bytes(text="page one content")})

        result = read_file(files["spec.pdf"])

        assert result.spans[0].location.startswith("page 1")

    def test_each_line_is_its_own_span(self, build_corpus) -> None:
        """A datasheet page holds a dozen unrelated facts.

        One span per page would make every fact on it equally good evidence
        for the others, which is exactly the ambiguity retrieval has to avoid.
        """
        files = build_corpus({"spec.pdf": pdf_bytes(text="line one")})

        result = read_file(files["spec.pdf"])

        assert len(result.spans) >= 1
        assert all("\n" not in span.text for span in result.spans)

    def test_a_file_without_a_pdf_header_is_rejected(self, build_corpus) -> None:
        files = build_corpus({"fake.pdf": b"this is plain text pretending to be a PDF"})

        result = read_file(files["fake.pdf"])

        assert result.status is ReadStatus.UNREADABLE
        assert "header" in result.detail

    def test_a_pdf_with_no_pages_is_empty_not_unreadable(self, build_corpus) -> None:
        """Nothing to draw is not the same as failing to understand the file."""
        files = build_corpus({"empty.pdf": b"%PDF-1.4\ntrailer\n<< /Size 1 >>\n%%EOF\n"})

        result = read_file(files["empty.pdf"])

        assert result.status is ReadStatus.EMPTY


class TestRealCorpusPdfs:
    """The two readable datasheets, which carry the supersession trap."""

    def _read(self, rel_path: str):
        from conftest import CORPUS_DIR
        from ingestion import walker

        discovered = next(
            f for f in walker.discover(CORPUS_DIR).files if f.rel_path == rel_path
        )
        return read_file(discovered)

    def test_the_current_revision_carries_the_current_value(self) -> None:
        result = self._read("specs/tq40_datasheet_r2.pdf")

        assert result.is_usable
        assert "94" in result.text

    def test_the_withdrawn_revision_carries_the_old_value(self) -> None:
        """Both files are real, readable and contain a real number.

        Only the revision tells them apart, so the reader has to preserve both
        faithfully -- deciding which one counts is a later stage's job.
        """
        result = self._read("specs/tq40_datasheet_r1_WITHDRAWN.pdf")

        assert result.is_usable
        assert "105" in result.text

    def test_the_two_revisions_do_not_contain_each_others_values(self) -> None:
        current = self._read("specs/tq40_datasheet_r2.pdf").text
        withdrawn = self._read("specs/tq40_datasheet_r1_WITHDRAWN.pdf").text

        assert "105" not in current
        assert "94" not in withdrawn

    def test_provenance_points_at_a_page_and_a_line(self) -> None:
        result = self._read("specs/tq40_datasheet_r2.pdf")

        assert all(span.location.startswith("page ") for span in result.spans)
        assert any(" line " in span.location for span in result.spans)
