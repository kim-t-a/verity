"""Tests for the CSV and spreadsheet readers.

Both formats are tables, so most of what is asserted here is about the shared
rendering: that a value can never drift under the wrong column name, and that
the rendering never rewrites a value.
"""

from __future__ import annotations

from document_fixtures import xlsx_bytes
from verity.engine.ingestion.readers import read_file, tabular


class TestHeaderDetection:
    def test_a_wide_first_row_is_treated_as_a_header(self) -> None:
        assert tabular.looks_like_header(["part", "description", "cost"]) is True

    def test_a_two_column_first_row_is_not_a_header(self) -> None:
        """The key/value case, and the reason the rule has a width floor.

        A datasheet's "Maximum junction temperature | 94 C" is not a header.
        Reading it as one would render the next row as
        "Maximum junction temperature=Board power", which invents a
        relationship between two unrelated measurements.
        """
        assert tabular.looks_like_header(["Maximum junction temperature", "94 C"]) is False

    def test_a_row_with_a_gap_is_not_a_header(self) -> None:
        assert tabular.looks_like_header(["part", "", "cost"]) is False

    def test_an_empty_row_is_not_a_header(self) -> None:
        assert tabular.looks_like_header([]) is False


class TestCsv:
    def test_header_and_rows_are_labelled(self, build_corpus) -> None:
        files = build_corpus(
            {
                "parts.csv": (
                    "ticket,component,status,fixed_in\n"
                    "ORR-1847,power,closed,4.3.2\n"
                )
            }
        )

        result = read_file(files["parts.csv"])

        assert result.spans[0].location == "header"
        assert result.spans[0].text == "ticket | component | status | fixed_in"
        assert result.spans[1].text == (
            "ticket=ORR-1847 | component=power | status=closed | fixed_in=4.3.2"
        )

    def test_the_column_name_lands_next_to_its_value(self, build_corpus) -> None:
        """This is the whole point of the keyed rendering.

        A query asking what release *fixed* something has the word "fixed" only
        in the header. Rendering the row as bare values would leave that word
        with nothing to match against.
        """
        files = build_corpus({"bugs.csv": "ticket,status,fixed_in\nORR-1847,closed,4.3.2\n"})

        result = read_file(files["bugs.csv"])

        assert "fixed_in=4.3.2" in result.text

    def test_every_cell_value_survives_verbatim(self, build_corpus) -> None:
        """Labels are added; values are never rewritten or paraphrased."""
        file_text = (
            "part_number,description,compatible_with\n"
            '"ORR-FAN-2214-B","Fan assembly, dual-rotor","TQ-40"\n'
        )
        files = build_corpus({"parts.csv": file_text})

        result = read_file(files["parts.csv"])

        assert "ORR-FAN-2214-B" in result.text
        assert "Fan assembly, dual-rotor" in result.text
        assert "TQ-40" in result.text

    def test_a_comma_inside_a_quoted_cell_does_not_split_the_row(self, build_corpus) -> None:
        """Without quote handling the description would become two columns."""
        file_text = (
            "part_number,description,compatible_with\n"
            '"ORR-FAN-2214-B","Fan assembly, dual-rotor","TQ-40"\n'
        )
        files = build_corpus({"parts.csv": file_text})

        result = read_file(files["parts.csv"])

        assert "description=Fan assembly, dual-rotor" in result.text

    def test_a_two_column_file_is_rendered_positionally(self, build_corpus) -> None:
        files = build_corpus(
            {"spec.csv": "Maximum junction temperature,94 C\nBoard power,410 W\n"}
        )

        result = read_file(files["spec.csv"])

        assert [span.text for span in result.spans] == [
            "Maximum junction temperature | 94 C",
            "Board power | 410 W",
        ]
        assert "=" not in result.text

    def test_ragged_rows_do_not_lose_values(self, build_corpus) -> None:
        """A short or long row must not silently drop a cell."""
        files = build_corpus(
            {"parts.csv": "a,b,c\n1,2,3\n4,5\n6,7,8,9\n"}
        )

        result = read_file(files["parts.csv"])
        body = result.text

        assert "a=4 | b=5" in body
        # The fourth cell of the last row has no header name, so it is kept
        # under a generated one rather than discarded.
        assert "column_4=9" in body

    def test_semicolon_delimited_files_are_detected(self, build_corpus) -> None:
        files = build_corpus({"parts.csv": "a;b;c\n1;2;3\n"})

        result = read_file(files["parts.csv"])

        assert "a=1 | b=2 | c=3" in result.text

    def test_tab_delimited_files_are_detected(self, build_corpus) -> None:
        files = build_corpus({"parts.csv": "a\tb\tc\n1\t2\t3\n"})

        result = read_file(files["parts.csv"])

        assert "a=1 | b=2 | c=3" in result.text

    def test_crlf_csv_is_read_correctly(self, build_corpus) -> None:
        files = build_corpus({"parts.csv": "a,b,c\r\n1,2,3\r\n"})

        result = read_file(files["parts.csv"])

        assert "c=3" in result.text
        assert "\r" not in result.text

    def test_a_single_column_file_is_not_mistaken_for_a_header(self, build_corpus) -> None:
        files = build_corpus({"notes.csv": "only one column\nand another line\n"})

        result = read_file(files["notes.csv"])

        assert "=" not in result.text
        assert "and another line" in result.text

    def test_row_numbers_match_the_file(self, build_corpus) -> None:
        files = build_corpus({"parts.csv": "a,b,c\n1,2,3\n4,5,6\n"})

        result = read_file(files["parts.csv"])

        assert [span.location for span in result.spans] == ["header", "row 2", "row 3"]


class TestXlsx:
    ROWS = [
        ["part_number", "description", "compatible_with"],
        ["ORR-FAN-2214-B", "Fan assembly, dual-rotor", "TQ-40"],
        ["ORR-FAN-1990-A", "Fan assembly, single-rotor", "TQ-20"],
    ]

    def test_sheet_name_appears_in_the_location(self, build_corpus) -> None:
        files = build_corpus({"rma.xlsx": xlsx_bytes([("RMA Parts", self.ROWS)])})

        result = read_file(files["rma.xlsx"])

        assert all("'RMA Parts'" in span.location for span in result.spans)

    def test_values_are_read_from_inline_strings(self, build_corpus) -> None:
        """The sample corpus stores strings inline and has no shared table."""
        files = build_corpus({"rma.xlsx": xlsx_bytes([("Parts", self.ROWS)])})

        result = read_file(files["rma.xlsx"])

        assert "part_number=ORR-FAN-2214-B" in result.text

    def test_values_are_read_from_shared_strings(self, build_corpus) -> None:
        """Most Excel-authored workbooks use the shared table instead.

        A reader supporting only one of the two storage methods returns an
        empty column for every file using the other, without erroring.
        """
        files = build_corpus(
            {"rma.xlsx": xlsx_bytes([("Parts", self.ROWS)], use_shared_strings=True)}
        )

        result = read_file(files["rma.xlsx"])

        assert "part_number=ORR-FAN-2214-B" in result.text

    def test_both_storage_methods_agree(self, build_corpus) -> None:
        files = build_corpus(
            {
                "inline.xlsx": xlsx_bytes([("Parts", self.ROWS)]),
                "shared.xlsx": xlsx_bytes([("Parts", self.ROWS)], use_shared_strings=True),
            }
        )

        assert read_file(files["inline.xlsx"]).text == read_file(files["shared.xlsx"]).text

    def test_numeric_cells_are_read(self, build_corpus) -> None:
        files = build_corpus(
            {
                "costs.xlsx": xlsx_bytes(
                    [("Costs", [["part", "lead_time", "cost"], ["FAN", 21, 84.5]])]
                )
            }
        )

        result = read_file(files["costs.xlsx"])

        assert "lead_time=21" in result.text
        assert "cost=84.5" in result.text

    def test_sparse_cells_do_not_shift_columns(self, build_corpus) -> None:
        """The failure this prevents: a value silently under the wrong label.

        A row that stores only cell ``C2`` still means column C. Reading cells
        in the order they appear would place that value in column A.
        """
        files = build_corpus(
            {"gap.xlsx": xlsx_bytes([("Sheet1", [["a", "b", "c"], [None, None, "here"]])])}
        )

        result = read_file(files["gap.xlsx"])
        row = [span.text for span in result.spans if span.location.endswith("row 2")][0]

        assert "c=here" in row
        assert "a=here" not in row

    def test_every_worksheet_is_read(self, build_corpus) -> None:
        """A reader that stops at sheet 1 loses the sample corpus's "Notes"."""
        files = build_corpus(
            {
                "wb.xlsx": xlsx_bytes(
                    [
                        ("RMA Parts", self.ROWS),
                        ("Notes", [["Lead times are supplier-quoted."]]),
                    ]
                )
            }
        )

        result = read_file(files["wb.xlsx"])
        locations = [span.location for span in result.spans]

        assert any("'RMA Parts'" in location for location in locations)
        assert any("'Notes'" in location for location in locations)
        assert "Lead times are supplier-quoted." in result.text

    def test_an_empty_sheet_does_not_break_the_others(self, build_corpus) -> None:
        files = build_corpus(
            {"wb.xlsx": xlsx_bytes([("Empty", []), ("Real", self.ROWS)])}
        )

        result = read_file(files["wb.xlsx"])

        assert "part_number=ORR-FAN-2214-B" in result.text

    def test_worksheets_are_still_found_without_a_relationships_part(
        self, build_corpus
    ) -> None:
        """A missing relationships part must not cost us every sheet."""
        files = build_corpus(
            {"wb.xlsx": xlsx_bytes([("Parts", self.ROWS)], include_relationships=False)}
        )

        result = read_file(files["wb.xlsx"])

        assert result.status.value == "ok"
        assert "part_number=ORR-FAN-2214-B" in result.text

    def test_booleans_render_as_words(self, build_corpus) -> None:
        files = build_corpus(
            {"flags.xlsx": xlsx_bytes([("Flags", [["name", "active", "note"], ["FAN", True, "x"]])])}
        )

        result = read_file(files["flags.xlsx"])

        assert "active=TRUE" in result.text

    def test_a_workbook_with_no_readable_sheet_is_reported(self, build_corpus) -> None:
        from document_fixtures import zip_bytes

        files = build_corpus({"wb.xlsx": zip_bytes({"xl/workbook.xml": "<workbook/>"})})

        result = read_file(files["wb.xlsx"])

        assert result.status.value == "empty"
        assert "worksheet" in result.detail

    def test_a_sheet_name_with_a_quote_is_still_matched(self, build_corpus) -> None:
        files = build_corpus({"wb.xlsx": xlsx_bytes([("It's Parts", self.ROWS)])})

        result = read_file(files["wb.xlsx"])

        assert "part_number=ORR-FAN-2214-B" in result.text
