"""Reader for Word ``.docx`` documents.

A DOCX is a ZIP whose ``word/document.xml`` holds the body. Only that one part
is read as content: the archive also carries ``styles.xml`` and
``stylesWithEffects.xml``, which in the sample corpus are hundreds of kilobytes
of formatting boilerplate. Parsing those as text would swamp retrieval with
irrelevant tokens -- so the reader names the parts it wants rather than walking
the archive looking for anything that looks like writing.

Two structural facts drive the code:

* **Order matters.** Paragraphs and tables interleave, and a table placed
  between two paragraphs is part of the document's meaning. The body is walked
  as a sequence, so the spans come out in reading order.
* **Text is nested.** A paragraph holds runs, and a run holds the text, but
  runs, hyperlinks, revisions and text boxes all nest differently. The text
  walker therefore recurses and collects rather than matching a fixed shape.

Not yet handled, and deliberately noted rather than silently dropped: content
in headers, footers and footnotes lives in separate parts and is not read.
"""

from __future__ import annotations

import zipfile

from verity.engine.models import DiscoveredFile, ReadResult, ReadStatus, TextSpan

from . import tabular, xmlpart

#: The part holding the document body. Everything else in the archive is
#: formatting, properties or a thumbnail.
DOCUMENT_PART = "word/document.xml"

#: Elements whose text is *not* document content. ``delText`` is text under a
#: tracked deletion and ``instrText`` is a field code; neither is what the
#: document says. ``tbl`` is listed because a nested table is collected by the
#: table walker, not by the inline text walker -- without this, a table inside a
#: paragraph would have its cells harvested twice.
_SKIP_IN_INLINE_WALK = frozenset({"delText", "instrText", "tbl"})

_BREAK_ELEMENTS = frozenset({"br", "cr"})


def _inline_text(element) -> str:
    """Collect the visible text under one paragraph or table cell.

    Recursive because OOXML nests text unevenly: a plain run is
    ``w:r > w:t``, but a hyperlink adds a level, a tracked insertion adds
    another, and a text box hangs off a drawing. Matching those shapes
    individually would mean a new branch for every producer; recursing and
    collecting handles all of them.
    """
    pieces: list[str] = []

    def visit(node) -> None:
        name = xmlpart.local_name(node.tag)
        if name in _SKIP_IN_INLINE_WALK:
            return
        if name == "t":
            pieces.append(node.text or "")
            return
        if name == "tab":
            pieces.append("\t")
            return
        if name in _BREAK_ELEMENTS:
            # A line break inside a paragraph becomes a space rather than a
            # newline, so one paragraph stays one retrievable line.
            pieces.append(" ")
            return
        for child in node:
            visit(child)

    for child in element:
        visit(child)
    return "".join(pieces)


def _cell_text(cell) -> str:
    """Flatten one table cell to a single line."""
    parts: list[str] = []
    for child in cell:
        name = xmlpart.local_name(child.tag)
        if name == "p":
            text = _inline_text(child).strip()
            if text:
                parts.append(text)
        elif name == "tbl":
            for row in _table_rows(child):
                parts.append(" ".join(cell_text for cell_text in row if cell_text))
    return " ".join(parts)


def _table_rows(table) -> list[list[str]]:
    """Extract a table as a grid of strings, preserving row and cell order."""
    rows: list[list[str]] = []
    for row in table:
        if xmlpart.local_name(row.tag) != "tr":
            continue
        cells = [_cell_text(cell) for cell in row if xmlpart.local_name(cell.tag) == "tc"]
        if any(cell.strip() for cell in cells):
            rows.append(cells)
    return rows


def _body_spans(root) -> list[TextSpan]:
    """Walk the document body in reading order."""
    body = None
    for element in root.iter():
        if xmlpart.local_name(element.tag) == "body":
            body = element
            break
    if body is None:
        return []

    spans: list[TextSpan] = []
    paragraph_number = 0
    table_number = 0

    for child in body:
        name = xmlpart.local_name(child.tag)
        ordinal = len(spans)
        if name == "p":
            paragraph_number += 1
            text = " ".join(_inline_text(child).split())
            if text:
                spans.append(
                    TextSpan(text=text, location=f"paragraph {paragraph_number}", ordinal=ordinal)
                )
        elif name == "tbl":
            table_number += 1
            rows = _table_rows(child)
            spans.extend(
                tabular.build_spans(
                    rows,
                    location_prefix=f"table {table_number}",
                    first_ordinal=ordinal,
                )
            )
    return spans


def read(discovered: DiscoveredFile) -> ReadResult:
    try:
        with xmlpart.open_archive(discovered.path) as archive:
            root = xmlpart.parse_member(archive, DOCUMENT_PART)
    except xmlpart.XmlPartError as exc:
        return ReadResult.failure(
            discovered.rel_path, discovered.media_type, ReadStatus.UNREADABLE, str(exc)
        )
    except (OSError, zipfile.BadZipFile) as exc:
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNREADABLE,
            f"{type(exc).__name__}: {exc}",
        )

    return ReadResult.of(discovered.rel_path, discovered.media_type, _body_spans(root))
