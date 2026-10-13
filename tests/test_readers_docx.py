"""Tests for the Word reader.

The two things most likely to go wrong are reading the wrong parts of the
archive, and flattening a table so that a row's cells lose their relationship
to each other.
"""

from __future__ import annotations

from document_fixtures import docx_bytes, docx_paragraph, docx_table
from ingestion.readers import read_file
from models import ReadStatus


class TestBody:
    def test_paragraphs_are_read_in_order(self, build_corpus) -> None:
        files = build_corpus(
            {
                "notes.docx": docx_bytes(
                    docx_paragraph("Orrery Systems Roadmap")
                    + docx_paragraph("TQ-60 enters customer sampling in Q3 FY27.")
                )
            }
        )

        result = read_file(files["notes.docx"])

        assert [span.text for span in result.spans] == [
            "Orrery Systems Roadmap",
            "TQ-60 enters customer sampling in Q3 FY27.",
        ]

    def test_locations_name_the_paragraph(self, build_corpus) -> None:
        files = build_corpus(
            {"notes.docx": docx_bytes(docx_paragraph("one") + docx_paragraph("two"))}
        )

        result = read_file(files["notes.docx"])

        assert [span.location for span in result.spans] == ["paragraph 1", "paragraph 2"]

    def test_the_styles_part_is_not_read_as_content(self, build_corpus) -> None:
        """`styles.xml` is hundreds of kilobytes of boilerplate in real files.

        It ships in every document built by this fixture, and none of its
        vocabulary may appear as recovered text.
        """
        files = build_corpus({"notes.docx": docx_bytes(docx_paragraph("real content"))})

        result = read_file(files["notes.docx"])

        assert result.text == "real content"
        assert "Heading1" not in result.text
        assert "styleId" not in result.text

    def test_text_split_across_runs_is_rejoined(self, build_corpus) -> None:
        """Word splits a sentence into runs at arbitrary points, including mid-word."""
        body = (
            "<w:p><w:r><w:t>Maximum junction temp</w:t></w:r>"
            "<w:r><w:t>erature .......... 94 C</w:t></w:r></w:p>"
        )
        files = build_corpus({"spec.docx": docx_bytes(body)})

        result = read_file(files["spec.docx"])

        assert result.text == "Maximum junction temperature .......... 94 C"

    def test_text_inside_a_hyperlink_is_read(self, build_corpus) -> None:
        body = (
            "<w:p><w:hyperlink r:id='rId1'>"
            "<w:r><w:t>ORR-FAN-2214-B</w:t></w:r></w:hyperlink></w:p>"
        )
        files = build_corpus({"notes.docx": docx_bytes(body)})

        assert "ORR-FAN-2214-B" in read_file(files["notes.docx"]).text

    def test_deleted_text_is_not_treated_as_document_content(self, build_corpus) -> None:
        """A tracked deletion is not what the document currently says."""
        body = (
            "<w:p><w:r><w:t>Current value 180</w:t></w:r></w:p>"
            "<w:p><w:del><w:r><w:delText>old value 60</w:delText></w:r></w:del></w:p>"
        )
        files = build_corpus({"notes.docx": docx_bytes(body)})

        result = read_file(files["notes.docx"])

        assert "180" in result.text
        assert "60" not in result.text

    def test_a_document_with_only_formatting_is_empty(self, build_corpus) -> None:
        files = build_corpus({"notes.docx": docx_bytes("")})

        assert read_file(files["notes.docx"]).status is ReadStatus.EMPTY


class TestTables:
    def test_a_table_renders_one_span_per_row(self, build_corpus) -> None:
        files = build_corpus(
            {
                "parts.docx": docx_bytes(
                    docx_table(
                        [
                            ["part_number", "description", "compatible_with"],
                            ["ORR-FAN-2214-B", "Dual-rotor fan", "TQ-40"],
                        ]
                    )
                )
            }
        )

        result = read_file(files["parts.docx"])

        assert "part_number=ORR-FAN-2214-B" in result.text
        assert all("table 1" in span.location for span in result.spans)

    def test_a_two_column_table_is_rendered_positionally(self, build_corpus) -> None:
        """The datasheet key/value shape, which must not become label=label."""
        files = build_corpus(
            {
                "spec.docx": docx_bytes(
                    docx_table(
                        [
                            ["Maximum junction temperature", "94 C"],
                            ["Board power (TBP)", "410 W"],
                        ]
                    )
                )
            }
        )

        result = read_file(files["spec.docx"])

        assert "Maximum junction temperature | 94 C" in result.text
        assert "=" not in result.text

    def test_tables_and_paragraphs_keep_their_relative_order(self, build_corpus) -> None:
        """A table's position in the document is part of its meaning."""
        files = build_corpus(
            {
                "mixed.docx": docx_bytes(
                    docx_paragraph("Before the table.")
                    + docx_table([["a", "b", "c"], ["1", "2", "3"]])
                    + docx_paragraph("After the table.")
                )
            }
        )

        result = read_file(files["mixed.docx"])
        locations = [span.location for span in result.spans]

        assert locations[0] == "paragraph 1"
        assert any("table 1" in location for location in locations)
        assert locations[-1].startswith("paragraph")

    def test_a_nested_table_inside_a_cell_is_read_once(self, build_corpus) -> None:
        inner = docx_table([["inner_a", "inner_b", "inner_c"]])
        body = f"<w:tbl><w:tr><w:tc>{docx_paragraph('outer')}{inner}</w:tc></w:tr></w:tbl>"
        files = build_corpus({"nested.docx": docx_bytes(body)})

        result = read_file(files["nested.docx"])

        assert result.text.count("inner_a") == 1

    def test_an_empty_table_is_not_an_error(self, build_corpus) -> None:
        files = build_corpus({"notes.docx": docx_bytes(docx_table([]))})

        assert read_file(files["notes.docx"]).status is ReadStatus.EMPTY


class TestRefusals:
    def test_a_document_type_definition_is_refused(self, build_corpus) -> None:
        """Entity expansion is the one way a tiny XML file becomes a huge one."""
        bomb = (
            '<!DOCTYPE w:document [<!ENTITY a "aaaaaaaaaa">'
            '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
        )
        files = build_corpus(
            {"evil.docx": docx_bytes(docx_paragraph("text"), doctype=bomb)}
        )

        result = read_file(files["evil.docx"])

        assert result.status is ReadStatus.UNREADABLE
        assert "document type" in result.detail

    def test_a_zip_without_a_document_part_is_reported(self, build_corpus) -> None:
        from document_fixtures import zip_bytes

        files = build_corpus({"notes.docx": zip_bytes({"word/styles.xml": "<w:styles/>"})})

        result = read_file(files["notes.docx"])

        assert result.status is ReadStatus.UNREADABLE
        assert "document.xml" in result.detail

    def test_malformed_xml_is_reported_not_raised(self, build_corpus) -> None:
        from document_fixtures import zip_bytes

        files = build_corpus(
            {"notes.docx": zip_bytes({"word/document.xml": "<w:document><w:body>"})}
        )

        result = read_file(files["notes.docx"])

        assert result.status is ReadStatus.UNREADABLE
        assert "well-formed" in result.detail
