"""Reader for Excel ``.xlsx`` workbooks.

Every worksheet is read, not just the first. The sample corpus rewards this
directly: its second sheet is called "Notes" and holds a sentence on its own,
and a reader that stopped at sheet 1 would lose it silently.

Three details are easy to get wrong and are handled explicitly here:

* **The sheet name lives elsewhere.** ``xl/worksheets/sheet1.xml`` does not say
  it is called "RMA Parts" -- that mapping is in ``xl/workbook.xml`` and its
  relationships part. Names matter because they are what the location labels
  quote back in diagnostics.
* **Cells are sparse and absolute.** A row is a list of ``<c r="D7">``
  elements; the columns between them are simply absent. Reading cells in the
  order they appear would slide every value left into the wrong column, so each
  cell is placed at the index its reference actually names.
* **Strings arrive three ways.** Shared strings (``t="s"``, an index into
  ``xl/sharedStrings.xml``), inline strings (``t="inlineStr"``) and formula
  results (``t="str"``). The sample corpus uses inline strings and has no
  shared-string table at all, while most workbooks produced by Excel use the
  shared table -- so both paths must work.
"""

from __future__ import annotations

import re
import zipfile

from verity.engine.models import DiscoveredFile, ReadResult, ReadStatus, TextSpan

from . import tabular, xmlpart

WORKBOOK_PART = "xl/workbook.xml"
WORKBOOK_RELS_PART = "xl/_rels/workbook.xml.rels"
SHARED_STRINGS_PART = "xl/sharedStrings.xml"
WORKSHEETS_DIR = "xl/worksheets/"

_TRUE_VALUES = frozenset({"1", "true"})


def _column_index(reference: str) -> int | None:
    """Turn a cell reference like ``C7`` into a zero-based column index.

    Returns ``None`` when the reference carries no letters, which means the
    producer left the position implicit and the caller should fall back to
    sequential placement.
    """
    letters = ""
    for character in reference:
        if not character.isalpha():
            break
        letters += character
    if not letters:
        return None
    index = 0
    for character in letters.upper():
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1


def _text_of(element) -> str:
    """Concatenate every ``t`` descendant, for inline and shared strings alike.

    A string can be split into formatting runs -- ``<is><r><t>Orrery</t></r>
    <r><t> Systems</t></r></is>`` -- and each run also carries its own ``t``.
    Joining them verbatim reconstructs the string, including any significant
    leading or trailing space.
    """
    return "".join(
        (node.text or "") for node in element.iter() if xmlpart.local_name(node.tag) == "t"
    )


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Load the shared string table, or an empty list when there is none."""
    try:
        root = xmlpart.parse_member(archive, SHARED_STRINGS_PART)
    except xmlpart.XmlPartError:
        return []
    return [
        _text_of(node) for node in root if xmlpart.local_name(node.tag) == "si"
    ]


def _cell_value(cell, shared: list[str]) -> str:
    """The text of one cell, whatever way it stores its value."""
    cell_type = cell.get("t")

    if cell_type == "inlineStr":
        return _text_of(cell)

    raw = None
    for child in cell:
        if xmlpart.local_name(child.tag) == "v":
            raw = child.text or ""
            break

    if raw is None:
        # Some producers omit the ``v`` wrapper for inline content.
        return _text_of(cell)

    if cell_type == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            return ""
    if cell_type == "b":
        return "TRUE" if raw.strip().lower() in _TRUE_VALUES else "FALSE"
    # ``str`` (formula result), ``e`` (error) and the untyped numeric default
    # all render as the text the cell already holds.
    return raw


def _worksheet_rows(root, shared: list[str]) -> list[list[str]]:
    """Extract a worksheet as a grid, placing each cell at its named column."""
    rows: list[list[str]] = []
    for row_element in root.iter():
        if xmlpart.local_name(row_element.tag) != "row":
            continue
        cells: list[str] = []
        for cell in row_element:
            if xmlpart.local_name(cell.tag) != "c":
                continue
            value = _cell_value(cell, shared)
            index = _column_index(cell.get("r") or "")
            if index is None:
                cells.append(value)
                continue
            # Fill the gap left by any columns this row simply does not have,
            # so a value can never drift under the wrong header.
            while len(cells) < index:
                cells.append("")
            cells.append(value)
        if any(cell.strip() for cell in cells):
            rows.append(cells)
    return rows


def _resolve_target(target: str) -> str:
    """Resolve a relationship target to a path inside the archive."""
    if target.startswith("/"):
        return target.lstrip("/")
    segments: list[str] = []
    for segment in ("xl/" + target).split("/"):
        if segment == "..":
            if segments:
                segments.pop()
        elif segment not in ("", "."):
            segments.append(segment)
    return "/".join(segments)


def _sheet_parts(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return ``(sheet name, part path)`` pairs in workbook order.

    Names and order come from ``xl/workbook.xml``. When the relationships part
    is missing or unreadable the worksheets are still enumerated from the
    archive by filename, because reading a sheet under a generic name is far
    better than not reading it at all.
    """
    try:
        workbook = xmlpart.parse_member(archive, WORKBOOK_PART)
    except xmlpart.XmlPartError:
        return _worksheets_by_filename(archive)

    relationships: dict[str, str] = {}
    try:
        rels_root = xmlpart.parse_member(archive, WORKBOOK_RELS_PART)
        for node in rels_root:
            if xmlpart.local_name(node.tag) == "Relationship":
                identifier, target = node.get("Id"), node.get("Target")
                if identifier and target:
                    relationships[identifier] = _resolve_target(target)
    except xmlpart.XmlPartError:
        pass

    sheets: list[tuple[str, str]] = []
    for node in workbook.iter():
        if xmlpart.local_name(node.tag) != "sheet":
            continue
        name = node.get("name") or f"Sheet{len(sheets) + 1}"
        identifier = next(
            (value for key, value in node.attrib.items() if xmlpart.local_name(key) == "id"),
            None,
        )
        part = relationships.get(identifier) if identifier else None
        if part:
            sheets.append((name, part))

    return sheets or _worksheets_by_filename(archive)


def _worksheets_by_filename(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Fall back to enumerating worksheet parts by name.

    Sorted naturally rather than lexically, so ``sheet2`` precedes ``sheet10``
    -- lexically it would not, and the sheets would come back in an order no
    human would recognise.
    """
    names = [
        name
        for name in archive.namelist()
        if name.startswith(WORKSHEETS_DIR) and name.endswith(".xml")
    ]
    names.sort(key=lambda name: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)])
    return [(f"Sheet{index + 1}", name) for index, name in enumerate(names)]


def read(discovered: DiscoveredFile) -> ReadResult:
    try:
        with xmlpart.open_archive(discovered.path) as archive:
            shared = _shared_strings(archive)
            spans: list[TextSpan] = []
            sheets_read = 0
            for name, part in _sheet_parts(archive):
                try:
                    root = xmlpart.parse_member(archive, part)
                except xmlpart.XmlPartError:
                    # One unreadable sheet must not cost us the others.
                    continue
                sheets_read += 1
                spans.extend(
                    tabular.build_spans(
                        _worksheet_rows(root, shared),
                        location_prefix=f"sheet {name!r}",
                        first_ordinal=len(spans),
                    )
                )
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

    return ReadResult.of(
        discovered.rel_path,
        discovered.media_type,
        spans,
        detail="" if sheets_read else "the workbook contains no readable worksheets",
    )
