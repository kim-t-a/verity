"""Turning a grid of cells into text, shared by the CSV and XLSX readers.

Both formats are *tables*, and both present the same problem: a bare row of
values loses the meaning of each column. A bug-database row rendered as
``ORR-1847,power,S2,...,4.3.2`` gives a query containing the word "fixed" nothing
to match, because the word "fixed" only exists in the header. Rendering each row
as ``fixed_in=4.3.2`` puts the column name next to the value, which is what
makes these files retrievable at all.

The one rule this rendering must never break: **every cell value appears
verbatim in the output.** Labels are added; nothing is substituted or
paraphrased. Downstream, "the answer must appear literally in its evidence" is
checked against this text, so a rendering that rewrote a value would manufacture
false evidence.
"""

from __future__ import annotations

from verity.engine.models import TextSpan

from verity.engine.config import CELL_SEPARATOR

#: Prefix given to cells that sit beyond the end of the header row.
UNNAMED_COLUMN_PREFIX = "column_"

#: A row is only treated as a header if it has at least this many columns.
#:
#: The reason is a specific, common failure. A two-column table is far more
#: often a *key/value* table -- "Maximum junction temperature | 94 C" -- than a
#: header followed by data. Treating its first row as a header would render the
#: second row as "Maximum junction temperature=Board power | 94 C=410 W": both
#: values survive, but the labels become nonsense and the row now reads as
#: though it were about a different measurement entirely.
#:
#: Requiring three columns removes that case while keeping the benefit for wide
#: tables, where a genuine header row is the overwhelmingly likely reading.
MIN_COLUMNS_FOR_HEADER = 3


def looks_like_header(row: list[str]) -> bool:
    """Whether a first row is a header rather than data.

    The test is deliberately conservative. A header names every column, has no
    gaps, and comes from a table wide enough that a header is the likely
    reading. Anything else is treated as data, because the cost of mislabelling
    data as a header is higher than the cost of mislabelling a header as data:
    the latter merely renders one row less usefully, while the former invents
    column names that are actually values.
    """
    cells = [cell.strip() for cell in row]
    return len(cells) >= MIN_COLUMNS_FOR_HEADER and all(cells)


def column_labels(header: list[str], width: int) -> list[str]:
    """Labels for a row of ``width`` cells, filling in past the header's end."""
    labels = [cell.strip() for cell in header]
    while len(labels) < width:
        labels.append(f"{UNNAMED_COLUMN_PREFIX}{len(labels) + 1}")
    return labels[:width]


def render_keyed_row(labels: list[str], row: list[str]) -> str:
    """Render one data row as ``label=value`` pairs.

    Cells past the end of ``labels`` are kept under a generated name rather
    than dropped, so a ragged row cannot lose a value.
    """
    pairs = []
    for index in range(max(len(labels), len(row))):
        label = labels[index] if index < len(labels) else f"{UNNAMED_COLUMN_PREFIX}{index + 1}"
        value = row[index].strip() if index < len(row) else ""
        pairs.append(f"{label}={value}")
    return CELL_SEPARATOR.join(pairs)


def render_positional_row(row: list[str]) -> str:
    """Render one row with no labels, used when there is no usable header."""
    return CELL_SEPARATOR.join(cell.strip() for cell in row)


def _location(place: str, suffix: str) -> str:
    """Join a place name and a position, tolerating an empty place.

    The strip matters: a CSV has no place name, so the label is built from an
    empty prefix, and without this a row would be located at " row 2" with a
    leading space.
    """
    return f"{place} {suffix}".strip()


def build_spans(
    rows: list[list[str]],
    *,
    location_prefix: str,
    first_ordinal: int = 1,
) -> list[TextSpan]:
    """Convert a table into spans, one per row, plus a header span.

    ``location_prefix`` names the place the rows came from, so a spreadsheet
    reports ``sheet 'RMA Parts' row 7`` and a CSV reports ``row 7``. It may be
    empty when the file has no sub-structure to name.
    """
    non_empty = [(offset, row) for offset, row in enumerate(rows) if any(c.strip() for c in row)]
    if not non_empty:
        return []

    header: list[str] = []
    body = non_empty
    if len(non_empty) > 1 and looks_like_header(non_empty[0][1]):
        header = non_empty[0][1]
        body = non_empty[1:]

    spans: list[TextSpan] = []

    if header:
        spans.append(
            TextSpan(
                text=render_positional_row(header),
                location=_location(location_prefix, "header"),
                ordinal=first_ordinal + non_empty[0][0],
            )
        )

    for offset, row in body:
        labels = column_labels(header, len(row)) if header else []
        text = render_keyed_row(labels, row) if labels else render_positional_row(row)
        spans.append(
            TextSpan(
                text=text,
                location=_location(location_prefix, f"row {offset + 1}"),
                ordinal=first_ordinal + offset,
            )
        )

    return spans
